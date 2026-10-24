#!/usr/bin/python3 -I
"""Synthetic native CLI wrapper, used only by offline relay integration tests.

As CLI 2.1.289 does (claude-relay/SAMPLING.md), it serves turn after turn
while stdin stays open. A user record with a uuid is reported once read
(command_lifecycle queued), and taken up (started) when its turn starts; the turn's first reply echoes the uuid, and its result is
numbered and names every record the turn took up. A record read while tool
calls wait goes to the model with their results. Marker words in a user
message select scenarios; see delivered_turn and after_result.
"""
import json
import os
import queue
import re
import shlex
import sys
import threading
import time
import urllib.request
import uuid as uuids

args = sys.argv[1:]
assert args[0] == '--'  # fixed broker invokes the real wrapper with this barrier
args = args[1:]
if args == ['--version']:
    os.write(1, b'2.1.')
    time.sleep(.01)  # the protocol driver must not mistake a short read for EOF
    print('0-fixture (not Claude)')
    raise SystemExit(0)


emit_lock = threading.Lock()


class Turn:
    """The native turn in progress: the record that started it, and every record
    taken up in it, the opener first and then those read mid-turn."""

    def __init__(self, ident):
        self.uuid = ident
        self.taken = [ident] if ident else []
        self.echo = bool(ident)  # its first reply is still to come


turn = Turn(None)
results_written = 0
message_count = 0


def emit(value):
    """Write one record. The turn's first reply echoes the uuid of the record that
    started it, and a result is numbered and names the turn's records; the
    records it took up are then completed."""
    global results_written
    kind = value.get('type')
    if (kind == 'stream_event' and isinstance(value.get('event'), dict)
            and value['event'].get('type') == 'message_start' and turn.echo):
        turn.echo = False
        value = {**value, 'user_message_uuid': turn.uuid, 'user_message_uuids': [turn.uuid]}
    completed = ()
    if kind == 'result' and turn.uuid:
        value = {'result_index': results_written, 'user_message_uuid': turn.uuid,
                 'user_message_uuids': list(turn.taken), **value}
        results_written += 1
        completed = tuple(turn.taken)
    with emit_lock:
        print(json.dumps(value, ensure_ascii=False), flush=True)
    for ident in completed:
        lifecycle(ident, 'completed')


def lifecycle(ident, state):
    emit({'type': 'command_lifecycle', 'command_uuid': ident, 'state': state, 'uuid': str(uuids.uuid4())})


def heartbeat(block, counter):
    emit({'type': 'tool_progress', 'heartbeat': True,
          'tool_use_id': block['id'] + '-heartbeat-' + str(counter),
          'parent_tool_use_id': block['id'], 'tool_name': block['name'],
          'elapsed_time_seconds': 30 * (counter + 1)})


def option(name):
    return args[args.index(name) + 1]


def stream_message(ident, blocks, usage, reason, details=None, delay=0):
    """Observed 2.1.289 ordering with wholly synthetic contents. ``delay`` is
    the time from the request's start to its first event."""
    emit({'type': 'system', 'subtype': 'status', 'status': 'requesting'})
    time.sleep(delay)
    def ping():
        if text == 'stream-ping':
            emit({'type': 'stream_event', 'event': {'type': 'ping'}})
    ping()
    provisional = {**usage, 'output_tokens': 1}
    emit({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': ident, 'usage': provisional}}})
    ping()
    for i, block in enumerate(blocks):
        start = {**block, 'input': {}} if block['type'] == 'tool_use' else block
        emit({'type': 'stream_event', 'event': {'type': 'content_block_start', 'index': i, 'content_block': start}})
        ping()
        if block['type'] == 'tool_use':
            # 2.1.289 streams a call without arguments as one empty partial_json.
            emit({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': i,
                  'delta': {'type': 'input_json_delta',
                            'partial_json': json.dumps(block['input']) if block['input'] else ''}}})
        if block['type'] == 'thinking':
            emit({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 50})
        emit({'type': 'assistant', 'message': {'id': ident, 'content': [block], 'stop_reason': None, 'usage': provisional}})
        emit({'type': 'stream_event', 'event': {'type': 'content_block_stop', 'index': i}})
        ping()
    delta = {'stop_reason': reason, **({'stop_details': details} if details is not None else {})}
    emit({'type': 'stream_event', 'event': {'type': 'message_delta', 'delta': delta, 'usage': usage}})
    emit({'type': 'stream_event', 'event': {'type': 'message_stop'}})
    ping()


def cut_message(ident, *, tool=False):
    """Observed 2.1.289 shape (untraced), synthetic contents: a long thinking
    block cut short, with no signature or per-block assistant record, then the
    CLI's own content_block_stop and message_stop. No message_delta follows.
    With ``tool``, a complete tool block precedes the cut instead."""
    def event(kind, **fields):
        emit({'type': 'stream_event', 'event': {'type': kind, **fields}})
    emit({'type': 'system', 'subtype': 'status', 'status': 'requesting'})
    event('message_start', message={'id': ident, 'usage': {'input_tokens': 5, 'output_tokens': 1}})
    event('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': '', 'signature': ''})
    for _ in range(3):
        emit({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 50})
        event('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'cut thinking '})
    if tool:
        event('content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': 'synthetic'})
        event('content_block_stop', index=0)
        block = {'type': 'tool_use', 'id': 'toolu_cut', 'name': names[0], 'input': {'value': 1}}
        event('content_block_start', index=1, content_block={**block, 'input': {}})
        event('content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])})
        emit({'type': 'assistant', 'message': {'id': ident, 'content': [block], 'stop_reason': None,
                                               'usage': {'input_tokens': 5, 'output_tokens': 1}}})
        event('content_block_stop', index=1)
    else:
        event('content_block_stop', index=0)
    event('message_stop')


initial = json.loads(sys.stdin.readline())
turn = Turn(initial.get('uuid'))
results_at = 0.0  # when the latest tool result came back (monotonic)
inbox = queue.Queue()  # user records read after the cold import; None at stdin's end
gates = queue.Queue()  # any other line: the tests' synthetic gates; None at stdin's end


def read_input():
    """Read stdin as the CLI does while it works: report each user record once
    read (queued), 50 ms after its write, and hold it until a turn takes it up."""
    for line in sys.stdin:
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not isinstance(record, dict) or record.get('type') != 'user':
            gates.put(line)
            continue
        words = ' '.join(block.get('text', '') for block in record['message']['content'])
        if 'exit-on-delivery' in words:
            os._exit(0)  # gone before reading it: no queued
        if 'no-queued' in words:
            continue  # never reported read
        # The CLI reads stdin some milliseconds after a write; in P1 it read a
        # record written as the results were released 10-15 ms after they
        # arrived, and missed it. Here the tool result's round trip is faster.
        time.sleep(.05)
        if record.get('uuid'):
            lifecycle(record['uuid'], 'queued')
        record['fixture_read_at'] = time.monotonic()
        inbox.put(record)
    inbox.put(None)
    gates.put(None)


def wait_for_end():
    """Wait for stdin's end, then exit: what a run that the relay must end does."""
    while inbox.get() is not None:
        pass
    raise SystemExit(0)


threading.Thread(target=read_input, daemon=True).start()
history = json.loads(initial['message']['content'])['history']
text = next((row['text'] for row in reversed(history) if row.get('role') == 'user'), '')
effort = option('--effort') if '--effort' in args else None
assert effort in (None, 'low', 'medium', 'high', 'xhigh', 'max')
summary = 'You are a context summarization assistant.' in option('--system-prompt')
config = json.loads(option('--mcp-config'))['mcpServers']
settings = json.loads(option('--settings')) if '--settings' in args else {}
tool_timeout_ms = settings.get('env', {}).get('MCP_TOOL_TIMEOUT')
print('fixture-state:' + json.dumps({'pid': os.getpid(), 'effort': effort, 'summary': summary,
                                   'tool_server_count': len(config),
                                   'mcp_tool_timeout': tool_timeout_ms,
                                   'mcp_tool_idle_timeout': settings.get('env', {}).get(
                                       'CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT'),
                                   'history_result_ids': [row['id'] for row in history if row['kind'] == 'result'],
                                   'history_call_names': [row['name'] for row in history if row['kind'] == 'call'],
                                   'history_call_arguments': [row['arguments'] for row in history if row['kind'] == 'call'],
                                   'system_preamble': option('--system-prompt').split('\n\n', 1)[0],
                                   'disable_auto_compact': os.environ.get('DISABLE_AUTO_COMPACT'),
                                   'import_uuid': initial.get('uuid'),
                                   'import_keys': sorted(initial),
                                   'cross_session_inbound': settings.get('crossSessionInbound')}),
      file=sys.stderr, flush=True)
servers, tools = [], []
seq = 0
if config:
    server = config['pythia']
    token = os.environ['PYTHIA_CLAUDE_MCP_TOKEN']
    assert token not in ' '.join(args)
    headers = {key: os.path.expandvars(value) for key, value in server['headers'].items()}
    headers.update({'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'})

    def rpc(method, params=None, notify=False):
        global seq
        seq += 1
        value = {'jsonrpc': '2.0', 'method': method, 'params': params or {}}
        if not notify:
            value['id'] = seq
        request = urllib.request.Request(server['url'], data=json.dumps(value).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
        if data:
            reply = json.loads(data)
            if method == 'server/discover':
                assert reply['error']['code'] == -32601
                return
            return reply['result']

    headers['MCP-Protocol-Version'] = '2026-07-28'
    rpc('server/discover', {'_meta': {'io.modelcontextprotocol/protocolVersion': '2026-07-28'}})
    del headers['MCP-Protocol-Version']
    rpc('initialize', {'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'fixture', 'version': '1'}})
    headers['MCP-Protocol-Version'] = '2025-03-26'
    rpc('notifications/initialized', notify=True)
    tools = rpc('tools/list')['tools']
    servers = [{'name': 'pythia', 'status': 'connected'}]

names = ['mcp__pythia__' + tool['name'] for tool in tools]


def tool_result(block):
    global results_at
    result = rpc('tools/call', {'name': block['name'].removeprefix('mcp__pythia__'), 'arguments': block['input'],
                               '_meta': {'claudecode/toolUseId': block['id']} if text != 'missing-meta' else {}})
    results_at = time.monotonic()
    emit({'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': block['id'],
                                                  'content': result['content'], 'is_error': result['isError']}]}})
    return result['content'][0]['text']


# warmpi's instruction (compaction.WARMPI_INSTRUCTION): answer it as a model would.
WARMPI = 'Your next reply is a context checkpoint'
WARM_SUMMARY = '## Goal\nfixture warm summary'


def pick_up(answer):
    """Records read while the calls waited, before their last result came back,
    go to the model with the results: each is started now, before the next
    request, and the answer shows them, or is a summary for warmpi's
    instruction. A record read after that result, or one saying
    'missed-steer', is read too late: the next request goes without it, and
    the relay must notice (no started before the next reply)."""
    while True:
        try:
            record = inbox.get_nowait()
        except queue.Empty:
            return answer
        if record is None:
            inbox.put(None)
            return answer
        texts = [block['text'] for block in record['message']['content']]
        if record['fixture_read_at'] > results_at or any('missed-steer' in text for text in texts):
            continue
        lifecycle(record['uuid'], 'started')
        turn.taken.append(record['uuid'])
        answer += ''.join(' +' + text for text in texts)
        if any(WARMPI in text for text in texts):
            answer = WARM_SUMMARY


def final_usage(words):
    """The final reply's usage. With 'cache-5m' its request wrote 5-minute cache
    entries, as on usage credits; otherwise it reports no cache writes."""
    if 'cache-5m' not in words:
        return {'input_tokens': 5, 'output_tokens': 4}
    return {'input_tokens': 5, 'cache_creation_input_tokens': 7, 'output_tokens': 4,
            'cache_creation': {'ephemeral_5m_input_tokens': 7, 'ephemeral_1h_input_tokens': 0}}


def batch_usage(words):
    """A tool batch's usage. With 'cache-5m' its request wrote 5-minute cache
    entries, as final_usage says; otherwise it reports no cache writes."""
    usage = {'input_tokens': 10, 'cache_read_input_tokens': 3, 'output_tokens': 2}
    if 'cache-5m' in words:
        usage.update(cache_creation_input_tokens=7,
                     cache_creation={'ephemeral_5m_input_tokens': 7, 'ephemeral_1h_input_tokens': 0})
    return usage


def after_result(words):
    """While idle after a result: 'idle-exit' exits; 'idle-record' writes a record
    no idle CLI should (a compaction's status), and 'idle-telemetry' one that
    is harmless mid-turn (a rate_limit_event). Then serve more turns."""
    if 'idle-exit' in words:
        time.sleep(.2)
        os._exit(0)
    if 'idle-record' in words:
        time.sleep(.2)
        emit({'type': 'system', 'subtype': 'status', 'status': 'compacting'})
    if 'idle-telemetry' in words:
        time.sleep(.2)
        emit({'type': 'rate_limit_event', 'rate_limit_info': {'status': 'allowed'}})
    serve()


def serve():
    """Turn after turn while stdin stays open: each user record starts one."""
    while True:
        record = inbox.get()
        if record is None:
            raise SystemExit(0)
        delivered_turn(record)


def delivered_turn(record):
    """One native turn for a user record written after a result. Its answer is
    'warm:' and the record's texts. Markers: 'warm-tools' calls the first tool
    twice first; 'no-echo' leaves the echo out; 'second-init', 'unknown-uuid'
    and 'bad-result-index' break the protocol mid-turn; and those of
    final_usage, after_result and the 'slow-final' delay. 'bad-result-uuids'
    names no record in the result."""
    global turn, message_count
    texts = [block['text'] for block in record['message']['content']]
    words = ' '.join(texts)
    print('fixture-turn:' + json.dumps({'pid': os.getpid(), 'uuid': record.get('uuid'),
                                        'client_composed': record.get('client_composed'),
                                        'parent_tool_use_id': record.get('parent_tool_use_id', '<absent>'),
                                        'texts': texts}),
          file=sys.stderr, flush=True)
    turn = Turn(record.get('uuid'))
    lifecycle(turn.uuid, 'started')
    emit({'type': 'system', 'subtype': 'init', 'tools': names, 'mcp_servers': servers})
    if 'no-echo' in words:
        turn.echo = False
    if 'second-init' in words:
        emit({'type': 'system', 'subtype': 'init', 'tools': names, 'mcp_servers': servers})
    if 'unknown-uuid' in words:
        lifecycle(str(uuids.uuid4()), 'queued')
    message_count += 1
    answer = WARM_SUMMARY if WARMPI in words else 'warm:' + '|'.join(texts)
    if 'warm-tools' in words and tools:
        blocks = [{'type': 'tool_use', 'id': f'toolu_w{message_count}_{n}', 'name': names[0], 'input': {'value': n}}
                  for n in (1, 2)]
        stream_message(f'm-warm-{message_count}-tools', blocks, {'input_tokens': 10, 'output_tokens': 2}, 'tool_use')
        answer = pick_up('|'.join(tool_result(block) for block in blocks))
    stream_message(f'm-warm-{message_count}', [{'type': 'text', 'text': answer}], final_usage(words), 'end_turn',
                   delay=.4 if 'slow-final' in words else 0)
    result = {'type': 'result', 'subtype': 'success', 'is_error': False}
    if 'bad-result-index' in words:
        result['result_index'] = 99
    if 'bad-result-uuids' in words:
        result['user_message_uuids'] = []
    emit(result)
    after_result(words)


if turn.uuid:
    lifecycle(turn.uuid, 'queued')
    lifecycle(turn.uuid, 'started')
    if 'import-no-echo' in text:
        turn.echo = False  # its first reply answers something else
emit({'type': 'system', 'subtype': 'init', 'tools': names + (['Bash'] if text == 'inventory-bad' else []), 'mcp_servers': servers})
if text.startswith('api-retry-'):
    retry = {'type': 'system', 'subtype': 'api_retry', 'attempt': 1, 'max_retries': 10,
             'retry_delay_ms': 615, 'error_status': 401, 'error': 'authentication_failed',
             'fixture_private': 'PRIVATE_RETRY_PAYLOAD'}
    emit(retry)  # metadata only; no real credential changes or backoff sleeps
    if text == 'api-retry-only':
        for _ in range(200):
            emit(retry)
            time.sleep(.01)
    elif text == 'api-retry-auth-error':
        emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
              'error': 'authentication_failed'})
    elif text == 'api-retry-unknown':
        emit({'type': 'stream_event', 'event': {'type': 'fixture_unknown'}})
    elif text == 'api-retry-exit':
        raise SystemExit(0)
    if text != 'api-retry-success':
        wait_for_end()
if text in ('trace-unknown', 'trace-error', 'trace-missing', 'trace-malformed'):
    os.write(2, b'private-native-stderr\xff\n')
    emit({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': 'failing'}}})
    for _ in range(90):
        emit({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 1})
    if text == 'trace-malformed':
        os.write(1, b'{private-invalid-json\xff}\n')
    else:
        event = ({'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'PRIVATE_UPSTREAM_BODY'}}
                 if text == 'trace-error' else {'type': 'fixture_unknown'} if text == 'trace-unknown' else {})
        emit({'type': 'stream_event', 'event': event})
    wait_for_end()
if text == 'trace-progress':
    os.write(1, b'{')
    for _ in range(200):
        os.write(1, b' ')
        time.sleep(.01)  # receive activity, but never a complete model message
    wait_for_end()
if text == 'ping-only':
    for _ in range(200):
        emit({'type': 'stream_event', 'event': {'type': 'ping'}})
        time.sleep(.01)
    wait_for_end()
if text in ('system-compacting', 'system-compact-boundary', 'system-hook'):
    record = {'type': 'system', 'subtype': {'system-compacting': 'status',
              'system-compact-boundary': 'compact_boundary', 'system-hook': 'hook_started'}[text]}
    if text == 'system-compacting':
        record['status'] = 'compacting'
    emit(record)
    raise SystemExit(0)
if text in ('refused', 'refused-without-details'):
    # Observed 2.1.289 shape: a classifier refusal after thinking with no text,
    # its reason only in message_delta's stop_details. Synthetic explanation.
    details = ({'type': 'refusal', 'category': 'cyber', 'explanation': 'Synthetic refusal explanation.'}
               if text == 'refused' else None)
    stream_message('m-refused', [{'type': 'thinking', 'thinking': '', 'signature': 'synthetic'}],
                   {'input_tokens': 5, 'output_tokens': 4}, 'refusal', details)
    wait_for_end()
if text == 'park':
    time.sleep(60)
if text == 'stream-cut-tools':
    cut_message('m-cut-0', tool=True)  # the relay must fail closed: its call may have been sent
    wait_for_end()
if text.startswith('bare-tool-name-once '):
    # Rejected as bare-tool-name by the first process only: that one creates the
    # marker file the prompt names, and later ones answer as usual.
    marker = text.split(' ', 1)[1]
    if not os.path.exists(marker):
        open(marker, 'x').close()
        text = 'bare-tool-name'
if text in ('bare-tool-name', 'unknown-tool-name', 'tool-input-rejected'):
    # Observed 2.1.289 order, synthetic contents: right after a tool block's
    # snapshot, before the message completes, the CLI answers a call it will
    # not make with one error result. Here: the bare Pythia name of a listed
    # tool, a name it has not got, or (not recognized) another rejection.
    name = 'Bash' if text == 'unknown-tool-name' else tools[0]['name'] if text == 'bare-tool-name' else names[0]
    block = {'type': 'tool_use', 'id': 'toolu_rejected', 'name': name, 'input': {'value': 1}}
    error = (f'Error: No such tool available: {name}' if text != 'tool-input-rejected'
             else 'InputValidationError: synthetic')
    def event(kind, **fields):
        emit({'type': 'stream_event', 'event': {'type': kind, **fields}})
    emit({'type': 'system', 'subtype': 'status', 'status': 'requesting'})
    event('message_start', message={'id': 'm-rejected', 'usage': {'input_tokens': 5, 'output_tokens': 1}})
    event('content_block_start', index=0, content_block={**block, 'input': {}})
    event('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])})
    emit({'type': 'assistant', 'message': {'id': 'm-rejected', 'content': [block], 'stop_reason': None,
                                           'usage': {'input_tokens': 5, 'output_tokens': 1}}})
    emit({'type': 'user', 'message': {'role': 'user', 'content': [{
          'type': 'tool_result', 'content': f'<tool_use_error>{error}</tool_use_error>', 'is_error': True,
          'tool_use_id': block['id']}]}, 'tool_use_result': error})
    event('content_block_stop', index=0)
    event('message_delta', delta={'stop_reason': 'tool_use'}, usage={'output_tokens': 3})
    event('message_stop')
    wait_for_end()
if text in ('early-callback', 'early-callback-invalid'):
    def event(kind, **fields):
        emit({'type': 'stream_event', 'event': {'type': kind, **fields}})
    blocks = [{'type': 'tool_use', 'id': ident, 'name': names[0], 'input': {'value': i + 1}}
              for i, ident in enumerate(('toolu_a', 'toolu_b'))]
    event('message_start', message={'id': 'm-early', 'usage': {'input_tokens': 10}})
    event('content_block_start', index=0, content_block=blocks[0])
    emit({'type': 'assistant', 'message': {'id': 'm-early', 'content': [blocks[0]], 'stop_reason': None}})
    event('content_block_stop', index=0)
    first_result = []
    def early_call():
        try:
            first_result.append(tool_result(blocks[0]))
        except (KeyError, OSError):
            pass  # expected revocation in the invalid/stalled-stream tests
    early = threading.Thread(target=early_call, daemon=True)
    early.start()
    event('content_block_start', index=1, content_block={**blocks[1], 'input': {}})
    event('content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': '{"value":'})
    # Synthetic-only gate: keep a later block incomplete while MCP is waiting.
    assert gates.get().strip() == 'fixture-finish'
    event('content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': '2}'})
    emit({'type': 'assistant', 'message': {'id': 'm-early', 'content': [blocks[1]], 'stop_reason': None}})
    event('content_block_stop', index=1)
    event('message_delta', delta={'stop_reason': 'max_tokens' if text.endswith('-invalid') else 'tool_use'},
          usage={'output_tokens': 2})
    event('message_stop')
    early.join(15)
    if not first_result:
        wait_for_end()
    answer = first_result[0] + '|' + tool_result(blocks[1])
elif text == 'host-command-tools':
    assert set(names) == {'mcp__pythia__' + name for name in ('exec_command', 'write_stdin', 'apply_patch', 'update_plan')}
    assert option('--tools') == ''  # Claude's own built-ins are still disabled
    assert set(option('--allowedTools').split(',')) == set(names)
    # Harmless host payload: report A's cwd/UID, then wait for write_stdin.
    # The fixture proposes it through MCP; it never runs this command itself.
    program = ('import json, os, sys; '
               'print(json.dumps({"host_uid": os.getuid(), "host_cwd": os.getcwd()})); '
               'print(sys.stdin.readline().strip())')
    command = '/usr/bin/python3 -I -S -u -c ' + shlex.quote(program)
    block = {'type': 'tool_use', 'id': 'toolu_exec', 'name': 'mcp__pythia__exec_command',
             'input': {'cmd': command, 'login': False, 'yield_time_ms': 0}}
    stream_message('m-exec', [block], {'input_tokens': 10, 'output_tokens': 2}, 'tool_use')
    output = tool_result(block)
    session = re.search(r'Process running with session ID (\d+)', output)
    assert session, 'exec_command did not return a host session'
    block = {'type': 'tool_use', 'id': 'toolu_stdin', 'name': 'mcp__pythia__write_stdin',
             'input': {'session_id': int(session[1]), 'chars': 'relay-host-session-ok\n', 'yield_time_ms': 1000}}
    stream_message('m-stdin', [block], {'input_tokens': 20, 'output_tokens': 2}, 'tool_use')
    answer = output + '\n' + tool_result(block)
elif text == 'host-command-deadline':
    # Two long command waits in one batch. The native client times each call
    # from sending it, one call at a time: report how long the first one
    # waited. The test scales the relay's native deadline down to match.
    blocks = [{'type': 'tool_use', 'id': ident, 'name': 'mcp__pythia__exec_command',
               'input': {'cmd': 'sleep 30', 'login': False, 'yield_time_ms': 2500}}
              for ident in ('toolu_wait_a', 'toolu_wait_b')]
    stream_message('m-wait', blocks, {'input_tokens': 10, 'output_tokens': 2}, 'tool_use')
    sent = time.monotonic()
    first = tool_result(blocks[0])
    answer = f'first-call-seconds={time.monotonic() - sent:.3f}|{first}|{tool_result(blocks[1])}'
elif text in ('rewritten-tool-input', 'rewritten-tool-input-beyond') and not any(row['kind'] == 'result' for row in history):
    # Observed 2.1.289 behavior, synthetic contents: the model streams literal
    # \u2013 text (one backslash, then two); the CLI's block snapshot and MCP
    # call decode the first into an en dash and leave the escaped backslash.
    # 'beyond' decodes that one too (\ then the en dash): not the known rule.
    streamed = {'value': 1, 'note': 'lines 1\\u20132, kept \\\\u2013'}
    native = {'value': 1, 'note': 'lines 1\u20132, kept \\\u2013' if text.endswith('-beyond')
              else 'lines 1\u20132, kept \\\\u2013'}
    block = {'type': 'tool_use', 'id': 'toolu_rewrite', 'name': names[0], 'input': {}, 'caller': {'type': 'direct'}}
    def event(kind, **fields):
        emit({'type': 'stream_event', 'event': {'type': kind, **fields}})
    emit({'type': 'system', 'subtype': 'status', 'status': 'requesting'})
    event('message_start', message={'id': 'm-rewrite', 'usage': {'input_tokens': 10, 'output_tokens': 1}})
    event('content_block_start', index=0, content_block=block)
    raw = json.dumps(streamed)
    for start in range(0, len(raw), 5):  # fragments split the escapes, as streamed
        event('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': raw[start:start + 5]})
    emit({'type': 'assistant', 'message': {'id': 'm-rewrite', 'content': [{**block, 'input': native}],
                                           'stop_reason': None, 'usage': {'input_tokens': 10, 'output_tokens': 1}}})
    event('content_block_stop', index=0)
    event('message_delta', delta={'stop_reason': 'tool_use'}, usage={'input_tokens': 10, 'output_tokens': 3})
    event('message_stop')
    try:
        tool_result({**block, 'input': native})
    except (KeyError, OSError):
        pass  # the relay retires this process rather than answer it
    wait_for_end()
elif text == 'rewritten-tool-input-kept' and not any(row['kind'] == 'result' for row in history):
    # Observed 2.1.289 behavior, synthetic contents: an input with an escape
    # the rule decodes, which the CLI's copies keep as streamed, as it keeps a
    # whole string at times. Its history is the stream's, so the turn goes on.
    block = {'type': 'tool_use', 'id': 'toolu_kept', 'name': names[0],
             'input': {'value': 1, 'note': 'rlo \\u202e kept'}}
    stream_message('m-kept', [block], {'input_tokens': 10, 'output_tokens': 2}, 'tool_use')
    answer = 'kept:' + tool_result(block)
elif text == 'icron-yield' and not any(row['kind'] == 'result' for row in history):
    # The CLI's scheduler, icron, through the relay: the model schedules a
    # message a second away and yields in one batch. The process waits for the
    # batch's results, which the turn that the message starts brings, with the
    # message (doc/claude-relay-warm-yield-plan.md).
    assert 'mcp__pythia__icron' in names
    blocks = [{'type': 'tool_use', 'id': 'toolu_schedule', 'name': 'mcp__pythia__icron',
               'input': {'action': 'schedule', 'message': 'fixture wake-up', 'after': 1}},
              {'type': 'tool_use', 'id': 'toolu_yield', 'name': 'mcp__pythia__icron',
               'input': {'action': 'yield', 'note': 'waiting for the fixture wake-up'}}]
    stream_message('m-icron', blocks, {'input_tokens': 10, 'output_tokens': 2}, 'tool_use')
    outputs = [json.loads(tool_result(block)) for block in blocks]
    answer = pick_up(f"pid={os.getpid()} yielded={outputs[1].get('yielded')}")
elif tools and text != 'text-only' and not any(row['kind'] == 'result' for row in history):
    arguments = {} if text == 'empty-arguments' else {'value': 1}
    blocks = [{'type': 'tool_use', 'id': ident, 'name': names[0], 'input': arguments} for ident in ('toolu_a', 'toolu_b')]
    batch_reason = 'max_tokens' if text == 'limited-tools' else 'tool_use'
    stream_message('m1', blocks, batch_usage(text), batch_reason)
    emit({'type': 'assistant', 'message': {'id': 'm1', 'content': blocks, 'usage': batch_usage(text), 'stop_reason': batch_reason}})
    if text in ('native-timeout', 'native-tool-timeout'):
        # Synthetic-only gate: the test waits for callback claim / actual host
        # effects, then requests the native error. No wall-clock timeout sleeps.
        def pending_request():
            try:
                rpc('tools/call', {'name': tools[0]['name'], 'arguments': {'value': 1},
                                  '_meta': {'claudecode/toolUseId': blocks[0]['id']}})
            except (KeyError, OSError):
                pass  # mailbox revocation is expected; never echo a fake result
        threading.Thread(target=pending_request, daemon=True).start()
        assert gates.get().strip() == 'fixture-expire'
        expired = 'The operation timed out.'
        if text == 'native-tool-timeout':
            # The client's own tool timeout (MCP_TOOL_TIMEOUT): its cancel notice
            # may reach the server before the result record, as here.
            rpc('notifications/cancelled', {'requestId': 99, 'reason': 'SdkError: Request timed out'}, notify=True)
            expired = (f'MCP server "pythia" tool "{tools[0]["name"]}" '
                       f'timed out after {int(tool_timeout_ms) // 1000}s')
        emit({'type': 'user', 'message': {'content': [{'type': 'tool_result',
              'tool_use_id': blocks[0]['id'], 'is_error': True, 'content': expired}]}})
        stream_message('speculative', [{'type': 'tool_use', 'id': 'must_not_execute',
                       'name': names[0], 'input': {'value': 999}}],
                       {'input_tokens': 3, 'output_tokens': 2}, 'tool_use')
        wait_for_end()
    if text == 'autonomous':
        emit({'type': 'result', 'subtype': 'success', 'is_error': False})
        time.sleep(2)
    done = threading.Event()
    pulse = None
    if text == 'heartbeat-batch':
        def progress():
            counter = 0
            while not done.wait(.02):
                heartbeat(blocks[0], counter)
                counter += 1
        pulse = threading.Thread(target=progress, daemon=True)
        pulse.start()
    outputs = []
    try:
        for block in blocks:
            outputs.append(tool_result(block))
    finally:
        done.set()
        if pulse is not None:
            pulse.join(2)
            heartbeat(blocks[0], 0)  # repeated/late telemetry after actual results
    answer = pick_up('|'.join(outputs))
else:
    answer = ('recovered:' + '|'.join(row['output'] for row in history if row['kind'] == 'result')
              if text in ('native-timeout', 'native-tool-timeout') else 'fixture summary' if summary
              else 'cold:' + text)
reason = text.removeprefix('limited:') if text.startswith('limited:') else 'end_turn'
if summary and 'limit-summary' in text:
    reason = 'max_tokens'
if 'stream-cut-' in text:
    for n in range(4 if 'stream-cut-repeated' in text else 1):
        cut_message(f'm-cut-{n}')
    if 'stream-cut-fallback' in text:
        # A complete record in place of a new stream, as a non-streaming retry might give.
        emit({'type': 'assistant', 'message': {'id': 'm-fallback', 'content': [{'type': 'text', 'text': answer}],
                                               'stop_reason': 'end_turn', 'usage': {'input_tokens': 5, 'output_tokens': 4}}})
        emit({'type': 'result', 'subtype': 'success', 'is_error': False})
        wait_for_end()
    if 'stream-cut-error' in text:
        emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True})
    elif 'stream-cut-ended' in text:
        emit({'type': 'result', 'subtype': 'success', 'is_error': False})
    if any(marker in text for marker in ('stream-cut-error', 'stream-cut-ended', 'stream-cut-repeated')):
        wait_for_end()
    # Otherwise the CLI retries: the complete message follows.
stream_message('m2', [{'type': 'thinking', 'thinking': '', 'signature': 'synthetic'}, {'type': 'text', 'text': answer}],
               final_usage(text), reason,
               delay=.4 if 'slow-final' in text else 0)
if text == 'conflicting-final-stop':
    emit({'type': 'assistant', 'message': {'id': 'm2', 'content': [{'type': 'thinking', 'thinking': '', 'signature': 'synthetic'},
                                                             {'type': 'text', 'text': answer}],
                                           'stop_reason': 'max_tokens'}})
if text == 'late-error':
    emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True})
else:
    emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'usage': {'input_tokens': 999999}})
after_result(text)  # then turn after turn, until stdin's end
