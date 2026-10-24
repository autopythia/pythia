"""Stdlib process owner. Relay bytes and native records are separate protocols.

A native process serves turn after turn while its stdin stays open
(doc/claude-relay-warm-delivery-plan.md). Its first turn is the cold import;
after each result it waits idle until ``deliver`` writes the next turn's user
messages. ``deliver`` also adds messages to a turn whose tool results are
pending, before releasing them. Every user record written has a uuid that the
CLI must account for: read (``queued``), taken up (``started``), and, for a
record that starts a turn, echoed on the turn's first reply.
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import contextlib
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import stat
import threading
import time
import uuid

from ..model import ModelError, ModelConfigurationError, ModelResponseError, ModelTransportError, ModelTimeoutError
from ..model import ModelAuthenticationError, ModelContextWindowError
from ._cli_protocol import Assembler, MAX_RECORD, NativeStreamInterrupted, PROFILE, decode
from ._mcp import Mailbox, MCPServer, NativeToolRejected
from ._sampling import ResolvedSampling
from ._diagnostics import Diagnostics
from ._trace import RelayTrace
from ._recovery import MCP_SERVER_NAME, NativeMCPTimeout, NATIVE_MCP_TIMEOUT, native_tool_rejection
from ._recovery import DeliveryUnacknowledged, IDLE_EXIT, IDLE_RECORD


PREAMBLE = ('You are serving a caller-owned conversation. The initial input is a JSON history, '
            'not instructions to replay old tool calls. Continue from that history using only '
            'the supplied tools.')


def system_prompt(instructions, native_names):
    """The native system prompt: the preamble, then the caller's instructions.

    ``native_names`` maps each Pythia tool name to its native callable name.
    The history shows calls to these tools under those names (Snapshot.prompt),
    but instructions may name a tool without the prefix, and CLI 2.1.289
    rejects a call by that name, so the preamble says how names map.
    """
    preamble = PREAMBLE
    if native_names:
        prefix = f'mcp__{MCP_SERVER_NAME}__'
        example = next(iter(native_names.values()))
        preamble += (f' Call tools only by their full names in your tool list, such as {example}.'
                     f' Where the history or the instructions name a tool without the {prefix}'
                     ' prefix, call it by its full name if your tool list has it; otherwise it'
                     ' cannot be called.')
    return preamble + '\n\n' + (instructions if instructions is not None else '')


# A command_lifecycle state, and the state before it: each comes once, in order.
_LIFECYCLE = {'queued': None, 'started': 'queued', 'completed': 'started'}


@dataclass
class _Turn:
    """The native turn in progress, from the user record that started it."""

    uuid: str
    live: bool  # delivered to a running process; not its cold import
    taken: list = field(default_factory=list)  # records the CLI started in it, in order
    replied: bool = False  # its first reply came
    init: bool = False  # its system/init came
    awaiting: str | None = None  # a record delivered mid-turn, not yet started


def init_identity(record):
    """What a turn's system/init must repeat from the process's first one."""
    tools = record.get('tools')
    if isinstance(tools, list) and all(isinstance(name, str) for name in tools):
        tools = sorted(tools)
    return (tools, json.dumps(record.get('mcp_servers'), sort_keys=True),
            record.get('model'), record.get('session_id'))


def cache_lifetime(value, previous):
    """The lifetime of a reply's cache writes, from its message_start: '5m' if
    any were 5-minute entries, else '1h' if any were 1-hour ones, else (no
    writes) ``previous``. No traced reply wrote both."""
    event = value.get('event') if isinstance(value.get('event'), dict) else {}
    message = event.get('message') if isinstance(event.get('message'), dict) else {}
    usage = message.get('usage') if isinstance(message.get('usage'), dict) else {}
    creation = usage.get('cache_creation') if isinstance(usage.get('cache_creation'), dict) else {}
    def wrote(key):
        count = creation.get(key)
        return type(count) is int and count > 0
    if wrote('ephemeral_5m_input_tokens'):
        return '5m'
    if wrote('ephemeral_1h_input_tokens'):
        return '1h'
    return previous


class Runtime:
    def __init__(self, endpoint, tools, generation, *, sampling=None, trace=None):
        self.endpoint = endpoint
        self.sampling = sampling if sampling is not None else ResolvedSampling()
        self.generation = generation
        self.diagnostics = Diagnostics()
        self.trace = RelayTrace(trace, generation)
        self.names = {f'mcp__{MCP_SERVER_NAME}__' + t.name: t.name for t in tools}
        self.native_names = {name: native for native, name in self.names.items()}  # Pythia -> native
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', t.name) for t in tools):
            raise ModelResponseError('MCP tool names must be simple identifiers')
        self.mailbox = Mailbox(tools, wait_seconds=endpoint.parked_timeout_seconds,
                               registration_seconds=endpoint.generation_timeout_seconds,
                               tool_timeout_seconds=endpoint.native_tool_timeout_seconds, trace=self.trace)
        self.mcp = MCPServer(self.mailbox, tool_id_pointer=endpoint.tool_id_pointer) if tools else None
        self.events = queue.Queue(maxsize=32)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._owner, name='claude-relay-runtime', daemon=True)
        self.proc = None
        self.tasks = []
        self.closed = threading.Event()
        self.retirement_complete = False
        self.error = None
        self.initialized = False
        self.result_seen = False
        self.final_message = None
        self.stderr_tail = bytearray()
        self._stop_lock = None
        self._init_turns()
        self.thread.start()
        self.mailbox.on_failure = self.fail

    def _init_turns(self):
        """The state of the process's turns (doc/claude-relay-warm-delivery-plan.md,
        3.1). It changes on the runtime's loop only."""
        self.turn = None  # _Turn
        self.results = 0  # results the process wrote: the next result_index
        self.lifecycle = {}  # each uuid written -> its latest command_lifecycle state
        self.pickup = None  # (uuid, future): a delivery waiting to be read
        self.first_init = None  # init_identity of the first system/init
        self.idle = False  # after a result, until the next delivery
        self.idle_end = None  # IDLE_EXIT or IDLE_RECORD: it ended while idle
        # For the idle deadline: when the latest request started (its
        # system/status requesting), the lifetime of the latest cache writes,
        # and both for the reply streaming, the final message, the turn last
        # answered (kept), and the latest tool batch, whose process a yield
        # parks (batch_origin; doc/claude-relay-warm-yield-plan.md).
        self.request_started = None
        self.cache_lifetime = None
        self.reply_origin = self.final_origin = self.kept = self.batch_origin = None

    @property
    def event_types(self):
        return self.diagnostics.snapshot().event_types

    def begin_sample(self, started, scope):
        with self.diagnostics.lock:
            if self.error is not None:
                raise self.error  # preserve the parked failure's origin and counters
            self.diagnostics.begin(started)
            self.trace.begin_sample(scope)

    def _owner(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()

    def _submit(self, coro, timeout):
        if self.closed.is_set() or self.loop.is_closed():
            coro.close()
            raise ModelTransportError('Claude Relay continuation retired')
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise ModelTimeoutError('Claude Relay startup/control deadline exceeded') from None
        except concurrent.futures.CancelledError:
            raise ModelTransportError('Claude Relay continuation retired') from None  # its loop stopped

    def start(self, snapshot):
        self._submit(self._start(snapshot), self.endpoint.startup_timeout_seconds + 2)

    def deliver(self, texts, *, results=None):
        """Write user messages ``texts`` as one user record, with a uuid.

        In an idle process it starts the next native turn. With ``results``
        (native ID -> (output, success)), it goes with the pending results
        instead, which are released only once the CLI reports reading the
        record (``queued``): written at the same moment, it missed them 3
        times in 3 (doc/claude-relay-warm-delivery-plan.md, Part 2). Raises
        ``DeliveryUnacknowledged`` when the CLI doesn't report reading it
        within ``startup_timeout_seconds``, or the process has ended; a later
        failure to take it up reaches ``next_message`` instead.
        """
        ident = str(uuid.uuid4())
        record = {'type': 'user', 'parent_tool_use_id': None, 'uuid': ident, 'client_composed': True,
                  'message': {'role': 'user', 'content': [{'type': 'text', 'text': text} for text in texts]}}
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
        if len(data) > MAX_RECORD:
            raise ModelResponseError('Delivered record exceeds protocol limit')
        self._submit(self._deliver(ident, data, len(texts), results), self.endpoint.startup_timeout_seconds + 2)

    async def _deliver(self, ident, data, count, results):
        if self.idle_end is not None:
            raise DeliveryUnacknowledged('Native process ended while idle', reason=self.idle_end)
        if self.error is not None or self.proc is None or self.proc.returncode is not None:
            raise DeliveryUnacknowledged('Native process ended before the delivery')
        if results is None:
            if not self.idle:
                raise ModelResponseError('Claude Relay can deliver a turn only to an idle native process')
            self.idle = False
            self.result_seen = False
            self.final_message = None
            self.turn = _Turn(ident, live=True)
            self.diagnostics.set_phase('awaiting_turn')
        else:
            if self.idle or self.turn is None or self.turn.awaiting is not None:
                raise ModelResponseError('Claude Relay can deliver messages only with pending results')
            self.turn.awaiting = ident
        self.lifecycle[ident] = None
        pickup = self.loop.create_future()
        self.pickup = (ident, pickup)
        self.trace.emit('continuation_delivery', uuid=ident, message_count=count, bytes=len(data),
                        mid_turn=results is not None)
        self.trace.data('stdin', data)
        # One deadline for the write and the read, within _submit's own.
        deadline = self.loop.time() + self.endpoint.startup_timeout_seconds
        try:
            self.proc.stdin.write(data)
            await asyncio.wait_for(self.proc.stdin.drain(), self.endpoint.startup_timeout_seconds)
            await asyncio.wait_for(asyncio.shield(pickup), max(0.0, deadline - self.loop.time()))
        except (OSError, asyncio.TimeoutError):
            self.pickup = None
            pickup.cancel()
            if self.closed.is_set():
                raise ModelTransportError('Claude Relay continuation retired') from None
            error = DeliveryUnacknowledged('Native CLI did not report reading the delivered messages')
            self._failure(error)
            raise error from None
        finally:
            self.pickup = None
            if not pickup.done():
                pickup.cancel()
        if results is not None:
            self.mailbox.release(results)

    def environment(self):
        if 'CLAUDE_CONFIG_DIR' in os.environ:
            raise ModelConfigurationError('Client-side CLAUDE_CONFIG_DIR is unsupported; configure the B-side profile')
        # TODO(output-budgets): forward only a verified native per-generation cap,
        # after relay/wrapper capability negotiation; never silently filter it.
        # Explicit generation/Pi-summary budgets still fail in _sampling.py.
        return {
            'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
            'HOME': os.environ.get('HOME', str(Path.home())),
            'CLAUDE_RELAY_SOCKET': self.endpoint.socket_path,
            'CLAUDE_RELAY_SERVER_UID': str(self.endpoint.server_uid),
            'CLAUDE_RELAY_PARENT_PID': str(os.getpid()),
            'CLAUDE_AGENT_SDK_CLIENT_APP': 'pythia-claude-relay',
            'CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS': '0', 'ENABLE_TOOL_SEARCH': 'false',
            'DISABLE_AUTO_COMPACT': '1',
            **({'PYTHIA_CLAUDE_MCP_TOKEN': self.mcp.token} if self.mcp else {}),
        }

    async def _spawn(self, argv):
        if self.closed.is_set():
            raise ModelTransportError('Claude Relay continuation retired')
        launcher = Path(self.endpoint.launcher).resolve(strict=True)
        for part in (*reversed(launcher.parents), launcher):
            info = part.lstat()
            expected = stat.S_ISREG if part == launcher else stat.S_ISDIR
            sticky = part != launcher and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
            if (not expected(info.st_mode) or info.st_uid not in (0, os.getuid())
                    or (info.st_mode & 0o022 and not sticky)):
                raise ModelTransportError('Relay client and parents must be trusted, owned paths')
        # Always a Python client under A. A mistaken native-binary path is never
        # executed as A, and caller PYTHONPATH/user-site imports are ignored.
        environment = self.environment()
        self.trace.invocation(argv, native_version_expected=self.endpoint.expected_version,
                              profile=PROFILE, requested_effort=self.sampling.effort,
                              requested_controls={k: environment[k] for k in (
                                  'DISABLE_AUTO_COMPACT', 'ENABLE_TOOL_SEARCH', 'CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS')},
                              generation_timeout_seconds=self.endpoint.generation_timeout_seconds,
                              parked_timeout_seconds=self.endpoint.parked_timeout_seconds,
                              startup_timeout_seconds=self.endpoint.startup_timeout_seconds,
                              stop_timeout_seconds=self.endpoint.stop_timeout_seconds,
                              native_tool_timeout_seconds=self.endpoint.native_tool_timeout_seconds)
        self.proc = await asyncio.create_subprocess_exec(
            '/usr/bin/python3', '-I', '-S', str(launcher), *argv, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=environment, cwd='/', limit=MAX_RECORD + 1)
        self.trace.emit('relay_client_started', client_pid=self.proc.pid)
        if self.closed.is_set():
            await self._stop_process()
            raise ModelTransportError('Claude Relay continuation retired')
        return self.proc

    async def _start(self, snapshot):
        try:
            self.diagnostics.set_phase('version_query')
            probe = await self._spawn(['--version'])
            probe.stdin.close()
            async def bounded(stream, limit):
                data = bytearray()
                while True:
                    part = await stream.read(min(8192, limit + 1 - len(data)))
                    if not part:
                        return bytes(data)
                    if stream is probe.stdout:
                        self.diagnostics.received()
                    self.trace.data('stdout' if stream is probe.stdout else 'stderr', part)
                    data.extend(part)
                    if len(data) > limit:
                        raise ModelResponseError('Native version output exceeds limit')
            out, err = await asyncio.wait_for(asyncio.gather(bounded(probe.stdout, 4096), bounded(probe.stderr, 65536)),
                                              self.endpoint.startup_timeout_seconds)
            self.stderr_tail.extend(err[:65536])
            rc = await asyncio.wait_for(probe.wait(), self.endpoint.stop_timeout_seconds)
            self.trace.exit(rc)
            self.proc = None
            if rc != 0 or len(out) > 4096 or len(err) > 65536:
                raise ModelTransportError('Relayed native version query failed')
            try:
                version = out.decode('utf-8', 'strict').strip()
            except UnicodeError:
                raise ModelResponseError('Native version query is not UTF-8') from None
            if not version.startswith(self.endpoint.expected_version + ' ') and version != self.endpoint.expected_version:
                raise ModelResponseError('Native CLI version differs from the explicitly selected compatibility version')
            self.diagnostics.native_version = self.endpoint.expected_version
            self.trace.emit('native_version_verified', native_version=self.endpoint.expected_version)
            config = {'mcpServers': {}}
            if self.mcp:
                # Env interpolation was live-checked on 2.1.289; retain the
                # authentication check for every deployment/version. The bearer
                # is never in argv, ordinary logs/history, or binding snapshots.
                # Opt-in raw native traces can still contain sensitive diagnostics.
                config['mcpServers'][MCP_SERVER_NAME] = {
                    'type': 'http', 'url': self.mcp.url,
                    'headers': {'Authorization': 'Bearer ${PYTHIA_CLAUDE_MCP_TOKEN}'},
                }
            system = system_prompt(snapshot.instructions, self.native_names)
            argv = ['--print', '--input-format', 'stream-json', '--output-format', 'stream-json',
                    '--verbose', '--include-partial-messages', '--model', self.endpoint.model,
                    '--system-prompt', system, '--tools', '', '--permission-mode', 'dontAsk',
                    '--strict-mcp-config', '--mcp-config', json.dumps(config, separators=(',', ':')),
                    '--setting-sources', '', '--disable-slash-commands', '--no-session-persistence']
            # Flag settings, which --setting-sources leaves on. A process kept
            # between turns refuses turns from B's other Claude sessions (its
            # cross-session inbox, bound in -p mode). With MCP tools, the CLI's
            # deadline for each call (MCP_TOOL_TIMEOUT; unset, 60 s). The CLI
            # also aborts a call whose server sends nothing for 300 s
            # (CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT), with a result the relay
            # doesn't recognize; 0 turns that off, so the deadline alone applies,
            # as an hour-long wait after a yield needs (probe P5,
            # doc/claude-relay-warm-yield-plan.md). These go through argv
            # because the relay forwards no environment for them.
            settings = {'crossSessionInbound': 'refuse'}
            if self.mcp:
                settings['env'] = {'MCP_TOOL_TIMEOUT': str(self.endpoint.native_tool_timeout_seconds * 1000),
                                   'CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT': '0'}
            argv += ['--settings', json.dumps(settings, separators=(',', ':'))]
            argv += self.sampling.cli_args()
            if self.names:
                argv += ['--allowedTools', ','.join(self.names)]
            self.diagnostics.invocation()
            self.diagnostics.set_phase('native_startup')
            proc = await self._spawn(argv)
            self.tasks = [asyncio.create_task(self._stdout()), asyncio.create_task(self._stderr())]
            # The cold import starts the first turn; its uuid is checked as a
            # delivery's is (_reply, _result).
            ident = str(uuid.uuid4())
            initial = {'type': 'user', 'uuid': ident,
                       'message': {'role': 'user', 'content': snapshot.prompt(self.native_names)}}
            data = (json.dumps(initial, ensure_ascii=False, allow_nan=False) + '\n').encode()
            self.turn = _Turn(ident, live=False)
            self.lifecycle[ident] = None
            if len(data) > MAX_RECORD:
                raise ModelResponseError('Cold-import record exceeds protocol limit')
            self.trace.data('stdin', data)  # attempted write, not proof of native receipt
            proc.stdin.write(data)
            await asyncio.wait_for(proc.stdin.drain(), self.endpoint.startup_timeout_seconds)
        except asyncio.TimeoutError:
            await self._stop_process()
            raise ModelTimeoutError('Native startup exceeded deadline') from None
        except Exception as error:
            await self._stop_process()
            if isinstance(error, ModelError):
                raise
            raise ModelTransportError('Relay startup failed; check executable, socket, UID and native profile') from None
        except BaseException:
            await self._stop_process()
            raise

    async def _stderr(self):
        try:
            while True:
                data = await self.proc.stderr.read(8192)
                if not data:
                    self.trace.emit('claude_eof', stream='stderr')
                    return
                self.trace.data('stderr', data)
                self.stderr_tail.extend(data)
                del self.stderr_tail[:-65536]
        except asyncio.CancelledError:
            raise

    def _put(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self._failure(ModelResponseError('Native event queue exceeded limit'))

    def fail(self, error):
        if not self.loop.is_closed() and not self.closed.is_set():
            self.loop.call_soon_threadsafe(self._failure, error)

    def _failure(self, error):
        if self.idle:
            self._end_idle(IDLE_RECORD)  # no sample to fail: the next starts a new process
            return
        with self.diagnostics.lock:
            if self.error is not None or self.closed.is_set():
                return
            turn = self.turn
            untaken = turn is not None and (turn.awaiting is not None or (turn.live and not turn.taken))
            if untaken and not isinstance(error, DeliveryUnacknowledged):
                # Before the CLI took up a delivery, nothing it writes is
                # adopted: whatever went wrong, the delivery wasn't taken.
                error = DeliveryUnacknowledged(
                    f'Native CLI did not take up the delivered messages ({type(error).__name__})')
            # A recognized native failure names its own code (as on NativeRefusal,
            # NativeStreamInterrupted, NativeToolRejected); others keep the record's.
            code = NATIVE_MCP_TIMEOUT if isinstance(error, NativeMCPTimeout) else getattr(error, 'error_code', None)
            if isinstance(code, str):
                self.diagnostics.error_code = code
            diagnostic = self.diagnostics.snapshot(freeze=True)
            self.error = error
            scope = self.trace.scope()
        self.trace.emit('native_failure', scope=scope, exception_type=type(error).__name__,
                        last_event_type=diagnostic.last_event_type,
                        error_code=diagnostic.error_code, event_count=diagnostic.event_count,
                        invocation_event_count=diagnostic.invocation_event_count)
        if isinstance(error, NativeMCPTimeout):
            self.trace.emit('native_mcp_timeout', scope=self.trace.scope(error.call.native_id),
                            native_tool_use_id=error.call.native_id, released=error.call.released,
                            returned=error.call.returned)
        self.mailbox.close()
        with contextlib.suppress(queue.Empty):
            while True:
                self.events.get_nowait()
        self._put(error)
        if self.pickup is not None and not self.pickup[1].done():
            self.pickup[1].set_exception(error)
        asyncio.create_task(self._stop_process())

    def _end_idle(self, reason):
        """The process ended while idle (``IDLE_EXIT``), or wrote a record no idle
        CLI writes (``IDLE_RECORD``), such as a compaction's: it is gone. No
        sample failed; the next one starts a new process."""
        if self.closed.is_set() or self.idle_end is not None:
            return
        self.idle_end = reason
        self.trace.emit('continuation_abandoned', reason=reason)
        self.mailbox.close()
        asyncio.create_task(self._stop_process())

    def _inventory(self, record):
        tools = record.get('tools')
        if not isinstance(tools, list) or set(tools) != set(self.names):
            raise ModelResponseError('Native tool inventory differs from the supplied MCP catalog')
        if self.names:
            servers = record.get('mcp_servers', [])
            if not any(s.get('name') == 'pythia' and s.get('status') == 'connected' for s in servers if isinstance(s, dict)):
                raise ModelResponseError('Native MCP connection is not ready; check bearer interpolation and MCP compatibility')
        self.initialized = True

    def _system(self, record):
        subtype = record.get('subtype')
        if subtype == 'init' and not self.initialized:
            self._inventory(record)
            self.first_init = init_identity(record)
            if self.turn is not None:
                self.turn.init = True
            return
        turn = self.turn
        if (subtype == 'init' and turn is not None and turn.live and turn.taken == [turn.uuid]
                and not turn.init and not turn.replied and init_identity(record) == self.first_init):
            turn.init = True  # each delivered turn starts with one, as the first
            return
        if self.initialized:
            if subtype == 'api_retry':
                return  # native retry/backoff metadata, not a terminal outcome
            # Observed in 2.1.289. Do not broadly ignore system/status: in
            # particular "compacting" must still retire this continuation.
            if subtype == 'status' and record.get('status') == 'requesting':
                self.request_started = time.monotonic()  # a cache lifetime starts here
                return
            if subtype == 'thinking_tokens':
                if all(type(record.get(k)) is int and record[k] >= 0
                       for k in ('estimated_tokens', 'estimated_tokens_delta')):
                    return  # estimates are NOT per-message usage or visible reasoning
                raise ModelResponseError('Invalid native thinking telemetry')
        raise ModelResponseError('Unexpected system event, hook, or internal compaction')

    def _lifecycle(self, record):
        """A command_lifecycle record: the CLI read a user record the relay wrote
        (queued), took it up (started), or answered it (completed). Each comes
        once, in that order, and only for a uuid the relay wrote. A record is
        taken up only to start the turn it was written for, or, delivered
        mid-turn, to go with the results."""
        ident, state = record.get('command_uuid'), record.get('state')
        if (state not in _LIFECYCLE or not isinstance(ident, str) or ident not in self.lifecycle
                or self.lifecycle[ident] != _LIFECYCLE[state]):
            raise ModelResponseError('Unexpected native command lifecycle record')
        if state == 'started':
            turn = self.turn
            if turn is None or ident not in (turn.awaiting, None if turn.taken else turn.uuid):
                raise ModelResponseError('Native CLI took up a user record out of turn')
            turn.taken.append(ident)
            if ident == turn.awaiting:
                turn.awaiting = None
        self.lifecycle[ident] = state
        if state == 'queued' and self.pickup is not None and self.pickup[0] == ident and not self.pickup[1].done():
            self.pickup[1].set_result(None)

    def _reply(self, record):
        """A message_start. A turn's first reply answers the record that started
        it: the CLI took it up, and echoes its uuid. A record delivered mid-turn
        must have been taken up before the next reply, which carries no echo;
        otherwise the CLI sent the results without it, and would answer it in a
        turn of its own. Also notes when the reply's request started, for the
        idle deadline."""
        turn = self.turn
        if turn is None:
            raise ModelResponseError('Native reply outside a turn')
        if not turn.replied:
            turn.replied = True
            if turn.taken != [turn.uuid] or record.get('user_message_uuid') != turn.uuid:
                if turn.live:
                    raise DeliveryUnacknowledged('Native CLI answered without the delivered messages')
                raise ModelResponseError('Native reply does not answer the cold import')
        elif turn.awaiting is not None:
            raise DeliveryUnacknowledged('Native CLI took the tool results without the delivered messages')
        self.cache_lifetime = cache_lifetime(record, self.cache_lifetime)
        started = self.request_started if self.request_started is not None else time.monotonic()
        self.request_started = None
        self.reply_origin = (started, self.cache_lifetime)

    def _result(self, record):
        """A turn's successful result, after its final message: it is the next
        one the process wrote, for the record that started the turn, naming
        every record the turn took up. The final message goes to the sample, and
        the process waits idle for the next delivery."""
        turn = self.turn
        names = record.get('user_message_uuids')
        if (turn is None or type(record.get('result_index')) is not int or record['result_index'] != self.results
                or record.get('user_message_uuid') != turn.uuid or not isinstance(names, list)
                or not all(isinstance(name, str) for name in names) or sorted(names) != sorted(turn.taken)):
            raise ModelResponseError('Native result does not match its turn')
        self.results += 1
        self.result_seen = True
        self.turn = None
        self.idle = True
        self.kept = self.final_origin
        self.diagnostics.set_phase('idle')
        self._put(self.final_message)

    async def _records(self):
        """Bounded framing, observing partial bytes even without a trailing LF."""
        buffer = bytearray()
        search_from = 0
        while True:
            chunk = await self.proc.stdout.read(8192)
            if not chunk:
                self.trace.emit('claude_eof', stream='stdout')
                if buffer:
                    self.diagnostics.malformed('truncated_record')
                    raise ModelResponseError('Truncated native protocol record')
                return
            self.diagnostics.received()
            self.trace.data('stdout', chunk)
            buffer.extend(chunk)
            while True:
                index = buffer.find(b'\n', search_from)
                if index < 0:
                    search_from = len(buffer)
                    break
                if index + 1 > MAX_RECORD:
                    self.diagnostics.malformed('oversized_record')
                    raise ModelResponseError('Native protocol record exceeds limit')
                line = bytes(buffer[:index + 1])
                del buffer[:index + 1]
                search_from = 0
                yield line
            if len(buffer) > MAX_RECORD:
                self.diagnostics.malformed('oversized_record')
                raise ModelResponseError('Native protocol record exceeds limit')

    async def _stdout(self):
        assembler = Assembler()
        interruptions = 0

        def after_interruption(message):
            """The run's end, named by the interrupted message the CLI did not replace."""
            count = assembler.consecutive_interruptions
            return NativeStreamInterrupted(
                f'{message} after {count} interrupted native message{"" if count == 1 else "s"} '
                '(its stream ended without a stop reason or final usage)')
        try:
            async for line in self._records():
                try:
                    value = decode(line)
                except (ModelResponseError, RecursionError):
                    self.diagnostics.malformed('invalid_record')
                    raise
                label = self.diagnostics.record(value)
                self.trace.emit('native_record', event_type=label)
                kind = value['type']
                if self.idle:
                    # After a result, only the records the CLI took up may
                    # complete. Anything else ends the process (_end_idle).
                    if kind != 'command_lifecycle' or value.get('state') != 'completed':
                        raise ModelResponseError('Native record while idle')
                    self._lifecycle(value)
                    continue
                heartbeat = kind == 'tool_progress' and value.get('heartbeat') is True
                if value.get('parent_tool_use_id') is not None and not heartbeat:
                    raise ModelResponseError('Native subagent output is unsupported')
                if kind == 'command_lifecycle':
                    self._lifecycle(value)  # may come before initialization
                    continue
                if kind == 'system' and value.get('subtype') == 'api_retry':
                    # Its error field describes a retry, not a final failure.
                    # Let the native CLI own backoff/refresh; do not validate its
                    # retry counters, reset deadlines, or add another retry loop.
                    self._system(value)  # still requires verified initialization
                    continue
                error_code = value.get('error')
                if isinstance(error_code, dict):
                    error_code = error_code.get('code', error_code.get('type'))
                if error_code:
                    if error_code == 'authentication_failed':
                        raise ModelAuthenticationError('Authenticate through the sandbox wrapper as the broker account')
                    if error_code in ('context_length_exceeded', 'prompt_too_long'):
                        raise ModelContextWindowError('Native context window exceeded')
                    raise ModelResponseError('Native CLI reported a model error')
                if heartbeat:
                    if not self.initialized:
                        raise ModelResponseError('Native tool heartbeat before verified initialization')
                    # CLI 2.1.289 emits synthetic <parent>-heartbeat-N IDs while
                    # waiting for MCP results. These are telemetry, not subagents
                    # or executable tool IDs. Do not alter mailbox/message state
                    # or extend deadlines in response to them.
                    # TODO(heartbeat-validation): validate the registered parent,
                    # MCP tool name, synthetic ID shape and finite elapsed value.
                    # For now correlation is best-effort tracing only; even an
                    # unknown/malformed parent grants no execution/result authority.
                    self.trace.emit('tool_heartbeat',
                                    scope=self.trace.scope(value.get('parent_tool_use_id')),
                                    validation='deferred')
                    continue
                if kind == 'system':
                    self._system(value)
                    if value.get('subtype') in ('init', 'status'):
                        self.diagnostics.set_phase('awaiting_model_message')
                    continue
                if kind in ('assistant', 'stream_event'):
                    if not self.initialized:
                        raise ModelResponseError('Native model output before verified initialization')
                    if (kind == 'stream_event' and isinstance(value.get('event'), dict)
                            and value['event'].get('type') == 'message_start'):
                        self._reply(value)
                    if assembler.current is not None or (kind == 'stream_event'
                            and isinstance(value.get('event'), dict)
                            and value['event'].get('type') == 'message_start'):
                        self.diagnostics.set_phase('message_stream')
                    message = assembler.feed(value)
                    if assembler.interruptions != interruptions:
                        interruptions = assembler.interruptions
                        # Discarded; the CLI may still complete the response.
                        self.diagnostics.set_phase('awaiting_model_message')
                        info = assembler.last_interruption
                        self.trace.emit('native_message_interrupted', native_message_id=info['native_message_id'],
                                        block_types=list(info['block_types']), open_blocks=info['open_blocks'],
                                        unsigned_thinking_blocks=info['unsigned_thinking_blocks'],
                                        consecutive=info['consecutive'])
                    if message is not None:
                        self.diagnostics.completed()
                        if not self.mailbox.all_returned() or self.final_message is not None:
                            raise ModelResponseError('Native continued before host handoff or run completion')
                        items, calls = message.sample_items(self.generation, self.names)
                        if calls:
                            self.mailbox.register(calls, rewritten=message.rewritten_tool_ids)
                            self.diagnostics.set_phase('awaiting_host_results')
                            self.batch_origin = self.reply_origin
                            self._put((message, items, calls))
                        else:
                            self.diagnostics.set_phase('awaiting_run_result')
                            self.final_message = (message, items, calls)
                            self.final_origin = self.reply_origin
                elif kind == 'user':
                    content = value.get('message', {}).get('content')
                    rejected = native_tool_rejection(value, assembler.open_tool_uses())
                    if rejected is not None:
                        ident, name = rejected  # diagnosis only: this record fails either way
                        self.mailbox.reject_streamed_call(
                            ident, NativeToolRejected(name, self.native_names.get(name)))
                    self.mailbox.validate_native_results(content)
                elif kind == 'result':
                    if value.get('is_error') is not False or value.get('subtype') != 'success':
                        if assembler.consecutive_interruptions and assembler.current is None:
                            raise after_interruption('Native run failed')
                        raise ModelResponseError('Native run failed; inspect explicit private diagnostics for details')
                    if not self.initialized or assembler.current is not None or self.final_message is None or not self.mailbox.all_returned():
                        if (assembler.consecutive_interruptions and assembler.current is None
                                and self.final_message is None):
                            raise after_interruption('Native run ended without a complete message')
                        raise ModelResponseError('Native run ended with incomplete message/handoff state')
                    self._result(value)
                elif kind in ('control_request', 'control_response'):
                    # This print/HTTP profile uses no SDK control handshake.
                    # Never autoapprove a surprise permission or custom MCP request.
                    raise ModelResponseError('Unsupported native control operation; compatibility profile must be revised')
                elif kind == 'rate_limit_event':
                    pass  # harmless quota telemetry, not a model message
                elif kind == 'tool_progress':
                    ident = value.get('tool_use_id')
                    if ident not in self.mailbox.slots:
                        raise ModelResponseError('Progress for an unknown native tool')
                else:
                    raise ModelResponseError('Unsupported native record type')
            if self.idle:
                self._end_idle(IDLE_EXIT)  # it exited between turns
                return
            if assembler.consecutive_interruptions and assembler.current is None:
                raise after_interruption('Native CLI exited without a run result')
            raise ModelTransportError('Native CLI exited without a successful run result')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.idle:
                self._end_idle(IDLE_RECORD)
                return
            if not isinstance(error, (ModelResponseError, ModelTransportError)):
                error = ModelResponseError('Invalid or incompatible native protocol record')
            self._failure(error)

    def next_message(self):
        try:
            event = self.events.get(timeout=self.endpoint.generation_timeout_seconds)
        except queue.Empty:
            diagnostic = self.diagnostics.snapshot(freeze=True)
            self.trace.emit('generation_timeout', budget_seconds=self.endpoint.generation_timeout_seconds,
                            phase=diagnostic.phase, last_byte_age=diagnostic.last_byte_age,
                            last_record_age=diagnostic.last_record_age,
                            last_completion_age=diagnostic.last_completion_age)
            raise ModelTimeoutError('Claude Relay generation wait exceeded deadline '
                                    f'(budget={self.endpoint.generation_timeout_seconds:g}s)') from None
        if isinstance(event, Exception):
            raise event
        if self.closed.is_set():
            raise ModelTransportError('Claude Relay continuation retired')
        if self.error is not None:
            raise self.error
        return event

    def _retired(self):
        """On the loop, at close: a delivery waiting to be read waits no more."""
        if self.pickup is not None and not self.pickup[1].done():
            self.pickup[1].set_exception(ModelTransportError('Claude Relay continuation retired'))

    async def _stop_process(self):
        if self._stop_lock is None:
            self._stop_lock = asyncio.Lock()
        async with self._stop_lock:
            await self._stop_process_locked()

    async def _stop_process_locked(self):
        proc = self.proc
        if proc is None:
            return
        if proc.returncode is None:
            self.trace.emit('process_signal', signal='TERM', client_pid=proc.pid)
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), self.endpoint.stop_timeout_seconds)
        except asyncio.TimeoutError:
            self.trace.emit('process_signal', signal='KILL', client_pid=proc.pid)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await asyncio.wait_for(proc.wait(), self.endpoint.stop_timeout_seconds)
        self.trace.exit(proc.returncode)

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.trace.emit('continuation_retire')
        self.mailbox.close()
        with contextlib.suppress(queue.Full):
            self.events.put_nowait(ModelTransportError('Claude Relay continuation retired'))
        with contextlib.suppress(RuntimeError):  # its loop has already closed
            self.loop.call_soon_threadsafe(self._retired)
        try:
            if self.thread.is_alive():
                future = asyncio.run_coroutine_threadsafe(self._stop_process(), self.loop)
                try:
                    future.result(2 * self.endpoint.stop_timeout_seconds + 1)
                finally:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    self.thread.join(timeout=2)
        finally:
            if self.mcp:
                self.mcp.close()
        mcp_retired = True
        if self.mcp:
            with self.mcp.connection_lock:
                mcp_retired = (self.mcp.closed and not self.mcp.thread.is_alive()
                               and not self.mcp.workers and not self.mcp.connections)
        self.retirement_complete = (
            not self.thread.is_alive() and (self.proc is None or self.proc.returncode is not None)
            and mcp_retired)
