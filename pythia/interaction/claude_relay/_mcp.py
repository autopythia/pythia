"""Bounded loopback MCP JSON-response subset, not a tool executor or a general SDK."""
from __future__ import annotations
import contextlib
import hmac
import http.server
import json
import secrets
import socket
import threading
import time
import uuid

from ..model import ModelError, ModelResponseError
from ..model_catalog import parse_json_value, thaw_json
from ._context import canonical
from ._recovery import ExpiredCall, NativeMCPTimeout, is_native_timeout, is_host_timeout_echo

PROTOCOLS = ('2025-03-26', '2025-06-18', '2025-11-25')
MAX_BODY = 4 * 1024 * 1024


class MCPError(ModelResponseError):
    def __init__(self, message, rpc_code=-32602):
        super().__init__(message)
        self.rpc_code = rpc_code


def pointer(document, path):
    current = document
    for part in path.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class Mailbox:
    def __init__(self, tools, *, wait_seconds=1800, trace=None):
        self.trace = trace
        self.tools = {t.name: t for t in tools}
        if len(self.tools) != len(tools):
            raise ValueError('duplicate tool names')
        self.wait_seconds = wait_seconds
        self.condition = threading.Condition()
        self.slots = {}
        self.closed = False
        self.failure = None
        self.on_failure = lambda error: None
        self.early = set()

    def fail(self, message, rpc_code=-32602):
        return self._fail_error(MCPError(message, rpc_code))

    def _fail_error(self, error):
        with self.condition:
            if self.failure is None:
                self.failure = error
            self.closed = True
            self.condition.notify_all()
            first = self.failure
        self.on_failure(first)
        return first

    def register(self, calls):
        with self.condition:
            if self.closed:
                raise self.failure or ModelResponseError('Mailbox retired')
            ids = [ident for ident, _, _ in calls]
            if len(set(ids)) != len(ids) or any(ident in self.slots for ident in ids):
                raise self.fail('Duplicate native tool ID')
            for ident, name, args in calls:
                if name not in self.tools:
                    raise self.fail('Native tool is not in the current catalog')
                self.slots[ident] = dict(name=name, args=canonical(args), claimed=False, returned=False, result=None)
            if self.trace:
                self.trace.register(calls)
            self.condition.notify_all()

    def release(self, results):
        with self.condition:
            if self.closed:
                raise self.failure or ModelResponseError('Mailbox retired')
            outstanding = {k for k, v in self.slots.items() if v['result'] is None}
            if set(results) != outstanding:
                raise self.fail('Result batch does not match outstanding native calls')
            prepared = {}
            for ident, (output, success) in results.items():
                if not isinstance(output, str) or type(success) is not bool or len(output) > MAX_BODY:
                    raise self.fail('MCP result exceeds supported text limits')
                value = {'content': [{'type': 'text', 'text': output}], 'isError': not success}
                try:
                    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_BODY - 4096:
                        raise ValueError()
                except (ValueError, UnicodeError):
                    raise self.fail('MCP result is oversized or not UTF-8') from None
                prepared[ident] = value
            for ident, value in prepared.items():
                self.slots[ident]['result'] = value
            if self.trace:
                self.trace.emit('mailbox_release', native_tool_use_ids=list(prepared))
            self.condition.notify_all()

    def call(self, ident, name, args):
        if not isinstance(ident, str) or not ident or len(ident) > 256:
            raise self.fail('MCP request lacks the configured native tool-use ID; JSON-RPC IDs are not substitutes')
        if name not in self.tools or not isinstance(args, dict):
            raise self.fail('Unknown MCP tool or invalid argument object')
        try:
            expected = canonical(args)
        except (ValueError, TypeError):
            raise self.fail('MCP arguments must be finite JSON') from None
        with self.condition:
            if ident in self.early:
                raise self.fail('Duplicate MCP callback')
            self.early.add(ident)
            registration_deadline = time.monotonic() + min(10, self.wait_seconds)
            while ident not in self.slots and not self.closed:
                remaining = registration_deadline - time.monotonic()
                if remaining <= 0:
                    raise self.fail('MCP native ID was never registered by the CLI stream')
                self.condition.wait(remaining)
            if self.closed:
                raise self.failure or ModelResponseError('Mailbox retired')
            slot = self.slots[ident]
            if slot['claimed'] or slot['name'] != name or slot['args'] != expected:
                raise self.fail('MCP callback does not match its native tool use')
            slot['claimed'] = True
            if self.trace:
                self.trace.emit('mailbox_claim', scope=self.trace.scope(ident), native_tool_use_id=ident)
            deadline = time.monotonic() + self.wait_seconds
            while slot['result'] is None and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self.fail('Parked MCP call expired', -32000)
                self.condition.wait(remaining)
            if self.closed:
                raise self.failure or ModelResponseError('Mailbox retired')
            slot['returned'] = True
            if self.trace:
                self.trace.emit('mailbox_return', scope=self.trace.scope(ident), native_tool_use_id=ident)
            return slot['result']

    def unreleased(self):
        with self.condition:
            return any(v['result'] is None for v in self.slots.values())

    def all_returned(self):
        with self.condition:
            return all(v['returned'] for v in self.slots.values())

    def validate_native_results(self, blocks):
        """Validate the whole envelope before granting any result authority.

        A returned slot means the HTTP callback obtained the host reply, not
        that the native client consumed it. Inspect timeout-shaped echoes even
        after return so a delivery race cannot substitute a native error.
        """
        with self.condition:
            if self.closed:
                raise self.failure or ModelResponseError('Mailbox retired')
            if not isinstance(blocks, list) or not blocks:
                raise self.fail('Unexpected native user content')
            seen, expired = set(), None
            for block in blocks:
                if not isinstance(block, dict) or block.get('type') != 'tool_result':
                    raise self.fail('Unexpected native user content')
                ident = block.get('tool_use_id')
                if not isinstance(ident, str) or ident not in self.slots or ident in seen:
                    raise self.fail('Native tool result has an unknown or duplicate call ID')
                seen.add(ident)
                slot = self.slots[ident]
                if is_native_timeout(block) and not (slot['returned'] and is_host_timeout_echo(slot['result'])):
                    if not slot['claimed']:
                        raise self.fail('Unclaimed native tool result appeared before a host reply')
                    expired = ExpiredCall(ident, slot['result'] is not None, slot['returned'])
                elif not slot['returned']:
                    raise self.fail('Native tool result appeared before the host released it')
            if expired is not None:
                if len(blocks) != 1:
                    raise self.fail('Ambiguous native tool-result envelope at MCP expiry')
                raise self._fail_error(NativeMCPTimeout(expired))

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class MCPServer:
    def __init__(self, mailbox, *, tool_id_pointer='/params/_meta/claudecode~1toolUseId', max_workers=16):
        self.mailbox = mailbox
        self.token = secrets.token_urlsafe(32)
        self.tool_id_pointer = tool_id_pointer
        self.initialized = False
        self.protocol = None
        self.state_lock = threading.Lock()
        self.connection_lock = threading.Lock()
        self.connections = set()
        self.workers = set()
        self.closed = False
        self.capacity = threading.BoundedSemaphore(max_workers)
        owner = self

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False
            allow_reuse_address = False
            def process_request(self, request, client_address):
                if owner.closed or not owner.capacity.acquire(False):
                    request.close()
                    return
                with owner.connection_lock:
                    owner.connections.add(request)
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    with owner.connection_lock:
                        owner.connections.discard(request)
                    owner.capacity.release()
                    raise
            def process_request_thread(self, request, client_address):
                with owner.connection_lock:
                    owner.workers.add(threading.current_thread())
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    with owner.connection_lock:
                        owner.connections.discard(request)
                        owner.workers.discard(threading.current_thread())
                    owner.capacity.release()
            def handle_error(self, request, client_address):
                pass  # never dump request/argument/capability data

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def setup(self):
                super().setup()
                self.connection.settimeout(10)
                self.trace_id = uuid.uuid4().hex
                self.trace_scope = owner.mailbox.trace.scope() if owner.mailbox.trace else {}
                self.trace_authorized = False
            def audit(self, kind, data=None, **fields):
                trace = owner.mailbox.trace
                if trace:
                    fields.update(mcp_exchange_id=self.trace_id)
                    if data is not None:
                        trace.payload(kind, data, scope=self.trace_scope, **fields)
                    else:
                        trace.emit(kind, scope=self.trace_scope, **fields)
            def log_message(self, *args):
                pass
            def respond(self, status, value=None):
                data = b'' if value is None else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                self.audit('mcp_response' if self.trace_authorized else 'mcp_rejected',
                           data=data if self.trace_authorized else None, status=status)
                self.send_response(status)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Content-Type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    if data:
                        self.wfile.write(data)
                    self.audit('mcp_response_written', status=status)
                except OSError as error:
                    self.audit('mcp_response_write_failed', exception_type=type(error).__name__)
                    raise
                self.close_connection = True
            def authorized(self):
                if self.path != '/mcp' or self.headers.get_all('Host') != [f'127.0.0.1:{owner.server.server_port}']:
                    self.respond(403)
                    return False
                if self.headers.get('Origin') is not None:
                    self.respond(403)
                    return False
                auth = self.headers.get_all('Authorization', [])
                if len(auth) != 1 or not hmac.compare_digest(auth[0].encode('latin-1'), ('Bearer ' + owner.token).encode()):
                    self.respond(401)
                    return False
                if owner.closed:
                    self.respond(410)
                    return False
                self.trace_authorized = True
                return True
            def do_GET(self):
                if self.authorized():
                    self.respond(405)  # explicitly no server-initiated SSE stream
            do_DELETE = do_GET
            def do_POST(self):
                if not self.authorized():
                    return
                lengths = self.headers.get_all('Content-Length', [])
                if (len(lengths) != 1 or not lengths[0].isdecimal() or not 0 < int(lengths[0]) <= MAX_BODY
                        or self.headers.get('Transfer-Encoding') is not None
                        or self.headers.get_content_type() != 'application/json'):
                    self.respond(400)
                    return
                if 'application/json' not in self.headers.get('Accept', 'application/json') and '*/*' not in self.headers.get('Accept', ''):
                    self.respond(406)
                    return
                rpc_code = -32700
                try:
                    raw = self.rfile.read(int(lengths[0]))
                    self.audit('mcp_request', data=raw)
                    if len(raw) != int(lengths[0]):
                        raise ValueError()
                    message = parse_json_value(raw.decode())
                    rpc_code = -32600
                    if not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or not isinstance(message.get('method'), str):
                        raise ValueError()
                    ident = message.get('id')
                    if 'id' in message and (isinstance(ident, bool) or not isinstance(ident, (int, str))):
                        raise ValueError()
                    if isinstance(ident, str) and len(ident) > 256 or type(ident) is int and abs(ident) > 2**53 - 1:
                        raise ValueError()
                except (ValueError, UnicodeError):
                    self.respond(400, {'jsonrpc': '2.0', 'id': None, 'error': {'code': rpc_code, 'message': 'Invalid JSON-RPC request'}})
                    return
                native_id = pointer(message, owner.tool_id_pointer)
                if owner.mailbox.trace:
                    self.trace_scope = owner.mailbox.trace.scope(native_id)
                self.audit('mcp_request_meta', method=message['method'], rpc_id=ident,
                           native_tool_use_id=native_id if isinstance(native_id, str) else None,
                           protocol_version=self.headers.get('MCP-Protocol-Version'))
                try:
                    result = owner.dispatch(message, self.headers.get('MCP-Protocol-Version'))
                except (ValueError, ModelError) as error:
                    if 'id' in message:
                        self.respond(200, {'jsonrpc': '2.0', 'id': ident,
                                          'error': {'code': getattr(error, 'rpc_code', -32602), 'message': 'MCP request rejected'}})
                    else:
                        self.respond(202)
                    return
                if 'id' not in message:
                    self.respond(202)
                else:
                    self.respond(200, {'jsonrpc': '2.0', 'id': ident, 'result': result})

        self.server = Server(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/mcp'
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        self.thread.start()

    def dispatch(self, request, version):
        method, params = request['method'], request.get('params', {})
        if not isinstance(params, dict):
            raise self.mailbox.fail('Invalid MCP params')
        if not method.startswith('notifications/') and 'id' not in request:
            raise self.mailbox.fail('MCP request requires a JSON-RPC ID')
        with self.state_lock:
            if method == 'server/discover' and self.protocol is None:
                # 2.1.289 first probes a newer stateless discovery protocol, then
                # falls back to initialize on -32601. Reject that optional probe
                # without killing the mailbox; do NOT claim the newer protocol
                # or accept its capabilities as an authenticated session.
                raise MCPError('Stateless MCP discovery is unsupported', -32601)
            if method == 'initialize':
                info = params.get('clientInfo')
                if (self.protocol is not None or params.get('protocolVersion') not in PROTOCOLS
                        or not isinstance(params.get('capabilities'), dict) or not isinstance(info, dict)
                        or any(not isinstance(info.get(k), str) or not 0 < len(info[k]) <= 128 for k in ('name', 'version'))):
                    raise self.mailbox.fail('Unsupported/repeated MCP initialization')
                self.protocol = params['protocolVersion']
                return {'protocolVersion': self.protocol, 'capabilities': {'tools': {'listChanged': False}},
                        'serverInfo': {'name': 'pythia-claude-relay', 'version': '1'}}
            if self.protocol is None or version != self.protocol:
                raise self.mailbox.fail('MCP protocol header/initialization mismatch')
            if method == 'notifications/initialized' and 'id' not in request:
                self.initialized = True
                return None
            if not self.initialized:
                raise self.mailbox.fail('MCP request before initialized notification')
        if method == 'ping':
            return {}
        if method == 'tools/list':
            if params:
                raise self.mailbox.fail('Unsupported MCP pagination')
            return {'tools': [{'name': t.name, 'description': t.description, 'inputSchema': thaw_json(t.parameters)}
                              for t in self.mailbox.tools.values()]}
        if method == 'tools/call' and 'id' in request:
            return self.mailbox.call(pointer(request, self.tool_id_pointer), params.get('name'), params.get('arguments', {}))
        if method == 'notifications/cancelled':
            raise self.mailbox.fail('Native MCP call cancelled')
        raise self.mailbox.fail('Unsupported MCP method', -32601)

    def close(self):
        if not self.closed:
            self.closed = True
            self.mailbox.close()
            self.server.shutdown()
            with self.connection_lock:
                connections, workers = tuple(self.connections), tuple(self.workers)
            for connection in connections:
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
            self.server.server_close()
            self.thread.join(timeout=2)
            deadline = time.monotonic() + 2
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
