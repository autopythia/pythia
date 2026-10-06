#!/usr/bin/python3 -I
"""Synthetic native CLI wrapper, used only by offline relay integration tests."""
import json
import os
import re
import shlex
import sys
import threading
import time
import urllib.request

args = sys.argv[1:]
assert args[0] == '--'  # fixed broker invokes the real wrapper with this barrier
args = args[1:]
if args == ['--version']:
    os.write(1, b'2.1.')
    time.sleep(.01)  # the protocol driver must not mistake a short read for EOF
    print('0-fixture (not Claude)')
    raise SystemExit(0)


emit_lock = threading.Lock()


def emit(value):
    with emit_lock:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def heartbeat(block, counter):
    emit({'type': 'tool_progress', 'heartbeat': True,
          'tool_use_id': block['id'] + '-heartbeat-' + str(counter),
          'parent_tool_use_id': block['id'], 'tool_name': block['name'],
          'elapsed_time_seconds': 30 * (counter + 1)})


def option(name):
    return args[args.index(name) + 1]


def stream_message(ident, blocks, usage, reason):
    """Observed 2.1.289 ordering with wholly synthetic contents."""
    emit({'type': 'system', 'subtype': 'status', 'status': 'requesting'})
    provisional = {**usage, 'output_tokens': 1}
    emit({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': ident, 'usage': provisional}}})
    for i, block in enumerate(blocks):
        start = {**block, 'input': {}} if block['type'] == 'tool_use' else block
        emit({'type': 'stream_event', 'event': {'type': 'content_block_start', 'index': i, 'content_block': start}})
        if block['type'] == 'tool_use':
            emit({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': i,
                  'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}}})
        if block['type'] == 'thinking':
            emit({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 50})
        emit({'type': 'assistant', 'message': {'id': ident, 'content': [block], 'stop_reason': None, 'usage': provisional}})
        emit({'type': 'stream_event', 'event': {'type': 'content_block_stop', 'index': i}})
    emit({'type': 'stream_event', 'event': {'type': 'message_delta', 'delta': {'stop_reason': reason}, 'usage': usage}})
    emit({'type': 'stream_event', 'event': {'type': 'message_stop'}})


initial = json.loads(sys.stdin.readline())
history = json.loads(initial['message']['content'])['history']
text = next((row['text'] for row in reversed(history) if row.get('role') == 'user'), '')
effort = option('--effort') if '--effort' in args else None
assert effort in (None, 'low', 'medium', 'high', 'xhigh', 'max')
summary = 'You are a context summarization assistant.' in option('--system-prompt')
config = json.loads(option('--mcp-config'))['mcpServers']
print('fixture-state:' + json.dumps({'pid': os.getpid(), 'effort': effort, 'summary': summary,
                                   'tool_server_count': len(config),
                                   'history_result_ids': [row['id'] for row in history if row['kind'] == 'result'],
                                   'disable_auto_compact': os.environ.get('DISABLE_AUTO_COMPACT')}),
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
    result = rpc('tools/call', {'name': block['name'].removeprefix('mcp__pythia__'), 'arguments': block['input'],
                               '_meta': {'claudecode/toolUseId': block['id']} if text != 'missing-meta' else {}})
    emit({'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': block['id'],
                                                  'content': result['content'], 'is_error': result['isError']}]}})
    return result['content'][0]['text']


emit({'type': 'system', 'subtype': 'init', 'tools': names + (['Bash'] if text == 'inventory-bad' else []), 'mcp_servers': servers})
if text in ('trace-ping', 'trace-error', 'trace-missing', 'trace-malformed'):
    os.write(2, b'private-native-stderr\xff\n')
    emit({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': 'failing'}}})
    for _ in range(90):
        emit({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 1})
    if text == 'trace-malformed':
        os.write(1, b'{private-invalid-json\xff}\n')
    else:
        event = ({'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'PRIVATE_UPSTREAM_BODY'}}
                 if text == 'trace-error' else {'type': 'ping'} if text == 'trace-ping' else {})
        emit({'type': 'stream_event', 'event': event})
    sys.stdin.read()
    raise SystemExit(0)
if text == 'trace-progress':
    os.write(1, b'{')
    for _ in range(200):
        os.write(1, b' ')
        time.sleep(.01)  # receive activity, but never a complete model message
    sys.stdin.read()
    raise SystemExit(0)
if text in ('system-compacting', 'system-compact-boundary', 'system-hook'):
    record = {'type': 'system', 'subtype': {'system-compacting': 'status',
              'system-compact-boundary': 'compact_boundary', 'system-hook': 'hook_started'}[text]}
    if text == 'system-compacting':
        record['status'] = 'compacting'
    emit(record)
    raise SystemExit(0)
if text == 'park':
    time.sleep(60)
if text == 'host-command-tools':
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
elif tools and text != 'text-only' and not any(row['kind'] == 'result' for row in history):
    blocks = [{'type': 'tool_use', 'id': ident, 'name': names[0], 'input': {'value': 1}} for ident in ('toolu_a', 'toolu_b')]
    batch_reason = 'max_tokens' if text == 'limited-tools' else 'tool_use'
    stream_message('m1', blocks, {'input_tokens': 10, 'cache_read_input_tokens': 3, 'output_tokens': 2}, batch_reason)
    emit({'type': 'assistant', 'message': {'id': 'm1', 'content': blocks, 'usage': {'input_tokens': 10, 'cache_read_input_tokens': 3, 'output_tokens': 2}, 'stop_reason': batch_reason}})
    if text == 'native-timeout':
        # Synthetic-only gate: the test waits for callback claim / actual host
        # effects, then requests the native error. No wall-clock timeout sleeps.
        def pending_request():
            try:
                rpc('tools/call', {'name': tools[0]['name'], 'arguments': {'value': 1},
                                  '_meta': {'claudecode/toolUseId': blocks[0]['id']}})
            except (KeyError, OSError):
                pass  # mailbox revocation is expected; never echo a fake result
        threading.Thread(target=pending_request, daemon=True).start()
        assert sys.stdin.readline().strip() == 'fixture-expire'
        emit({'type': 'user', 'message': {'content': [{'type': 'tool_result',
              'tool_use_id': blocks[0]['id'], 'is_error': True, 'content': 'The operation timed out.'}]}})
        stream_message('speculative', [{'type': 'tool_use', 'id': 'must_not_execute',
                       'name': names[0], 'input': {'value': 999}}],
                       {'input_tokens': 3, 'output_tokens': 2}, 'tool_use')
        sys.stdin.read()
        raise SystemExit(0)
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
    answer = '|'.join(outputs)
else:
    answer = ('recovered:' + '|'.join(row['output'] for row in history if row['kind'] == 'result')
              if text == 'native-timeout' else 'fixture summary' if summary else 'cold:' + text)
reason = text.removeprefix('limited:') if text.startswith('limited:') else 'end_turn'
if summary and 'limit-summary' in text:
    reason = 'max_tokens'
stream_message('m2', [{'type': 'thinking', 'thinking': '', 'signature': 'synthetic'}, {'type': 'text', 'text': answer}],
               {'input_tokens': 5, 'output_tokens': 4}, reason)
if text == 'conflicting-final-stop':
    emit({'type': 'assistant', 'message': {'id': 'm2', 'content': [{'type': 'thinking', 'thinking': '', 'signature': 'synthetic'},
                                                             {'type': 'text', 'text': answer}],
                                           'stop_reason': 'max_tokens'}})
if text == 'late-error':
    emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True})
else:
    emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'usage': {'input_tokens': 999999}})
sys.stdin.read()  # driver must close stdin after successful terminal result
