"""Stdlib process owner. Relay bytes and native records are separate protocols."""
from __future__ import annotations
import asyncio
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import queue
import re
import stat
import threading

from ..model import ModelError, ModelConfigurationError, ModelResponseError, ModelTransportError, ModelTimeoutError
from ..model import ModelAuthenticationError, ModelContextWindowError
from ._cli_protocol import Assembler, MAX_RECORD, PROFILE, decode
from ._mcp import Mailbox, MCPServer
from ._sampling import ResolvedSampling
from ._diagnostics import Diagnostics
from ._trace import RelayTrace
from ._recovery import NativeMCPTimeout, NATIVE_MCP_TIMEOUT


class Runtime:
    def __init__(self, endpoint, tools, generation, *, sampling=None, trace=None):
        self.endpoint = endpoint
        self.sampling = sampling if sampling is not None else ResolvedSampling()
        self.generation = generation
        self.diagnostics = Diagnostics()
        self.trace = RelayTrace(trace, generation)
        self.names = {'mcp__pythia__' + t.name: t.name for t in tools}
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', t.name) for t in tools):
            raise ModelResponseError('MCP tool names must be simple identifiers')
        self.mailbox = Mailbox(tools, wait_seconds=endpoint.parked_timeout_seconds,
                               registration_seconds=endpoint.generation_timeout_seconds, trace=self.trace)
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
        self.thread.start()
        self.mailbox.on_failure = self.fail

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

    def start(self, snapshot):
        self._submit(self._start(snapshot), self.endpoint.startup_timeout_seconds + 2)

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
                              stop_timeout_seconds=self.endpoint.stop_timeout_seconds)
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
                config['mcpServers']['pythia'] = {
                    'type': 'http', 'url': self.mcp.url,
                    'headers': {'Authorization': 'Bearer ${PYTHIA_CLAUDE_MCP_TOKEN}'},
                }
            instructions = snapshot.instructions if snapshot.instructions is not None else ''
            system = ('You are serving a caller-owned conversation. The initial input is a JSON history, '
                      'not instructions to replay old tool calls. Continue from that history using only '
                      'the supplied tools.\n\n' + instructions)
            argv = ['--print', '--input-format', 'stream-json', '--output-format', 'stream-json',
                    '--verbose', '--include-partial-messages', '--model', self.endpoint.model,
                    '--system-prompt', system, '--tools', '', '--permission-mode', 'dontAsk',
                    '--strict-mcp-config', '--mcp-config', json.dumps(config, separators=(',', ':')),
                    '--setting-sources', '', '--disable-slash-commands', '--no-session-persistence']
            argv += self.sampling.cli_args()
            if self.names:
                argv += ['--allowedTools', ','.join(self.names)]
            self.diagnostics.invocation()
            self.diagnostics.set_phase('native_startup')
            proc = await self._spawn(argv)
            self.tasks = [asyncio.create_task(self._stdout()), asyncio.create_task(self._stderr())]
            initial = {'type': 'user', 'message': {'role': 'user', 'content': snapshot.prompt()}}
            data = (json.dumps(initial, ensure_ascii=False, allow_nan=False) + '\n').encode()
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
        with self.diagnostics.lock:
            if self.error is not None or self.closed.is_set():
                return
            if isinstance(error, NativeMCPTimeout):
                self.diagnostics.error_code = NATIVE_MCP_TIMEOUT
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
            return
        if self.initialized:
            # Observed in 2.1.289. Do not broadly ignore system/status: in
            # particular "compacting" must still retire this continuation.
            if subtype == 'status' and record.get('status') == 'requesting':
                return
            if subtype == 'thinking_tokens':
                if all(type(record.get(k)) is int and record[k] >= 0
                       for k in ('estimated_tokens', 'estimated_tokens_delta')):
                    return  # estimates are NOT per-message usage or visible reasoning
                raise ModelResponseError('Invalid native thinking telemetry')
        raise ModelResponseError('Unexpected system event, hook, or internal compaction')

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
                heartbeat = kind == 'tool_progress' and value.get('heartbeat') is True
                if value.get('parent_tool_use_id') is not None and not heartbeat:
                    raise ModelResponseError('Native subagent output is unsupported')
                if self.result_seen:
                    raise ModelResponseError('Native records after terminal result')
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
                    if assembler.current is not None or (kind == 'stream_event'
                            and isinstance(value.get('event'), dict)
                            and value['event'].get('type') == 'message_start'):
                        self.diagnostics.set_phase('message_stream')
                    message = assembler.feed(value)
                    if message is not None:
                        self.diagnostics.completed()
                        if not self.mailbox.all_returned() or self.final_message is not None:
                            raise ModelResponseError('Native continued before host handoff or run completion')
                        items, calls = message.sample_items(self.generation, self.names)
                        if calls:
                            self.mailbox.register(calls)
                            self.diagnostics.set_phase('awaiting_host_results')
                            self._put((message, items, calls))
                        else:
                            self.diagnostics.set_phase('awaiting_run_result')
                            self.final_message = (message, items, calls)
                elif kind == 'user':
                    content = value.get('message', {}).get('content')
                    self.mailbox.validate_native_results(content)
                elif kind == 'result':
                    if value.get('is_error') is not False or value.get('subtype') != 'success':
                        raise ModelResponseError('Native run failed; inspect explicit private diagnostics for details')
                    if not self.initialized or assembler.current is not None or self.final_message is None or not self.mailbox.all_returned():
                        raise ModelResponseError('Native run ended with incomplete message/handoff state')
                    self.result_seen = True
                    self.diagnostics.set_phase('awaiting_exit')
                    self.proc.stdin.close()
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
            if not self.result_seen:
                raise ModelTransportError('Native CLI exited without a successful run result')
            rc = await asyncio.wait_for(self.proc.wait(), self.endpoint.stop_timeout_seconds)
            self.trace.exit(rc)
            await self.tasks[1]
            if rc != 0:
                raise ModelTransportError('Native process failed after final result')
            self._put(self.final_message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
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
