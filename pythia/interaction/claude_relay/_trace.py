"""Native trace attribution. No ambient contextvars in persistent IO threads."""
import threading
import uuid


class RelayTrace:
    def __init__(self, sink, runtime_id):
        self.sink, self.runtime_id = sink, runtime_id
        self.lock = threading.RLock()
        self.active = {}
        self.origins = {}
        self.invocation_id = None
        self.offsets = {}
        self.exited = False

    def begin_sample(self, scope):
        with self.lock:
            self.active = dict(scope)

    def parked(self):
        with self.lock:
            self.active = {}

    def scope(self, native_id=None):
        with self.lock:
            origin = self.origins.get(native_id) if isinstance(native_id, str) else None
            return {"op": None, "operation_id": None, "sample_id": None,
                    **(self.active if origin is None else origin),
                    "runtime_id": self.runtime_id, "invocation_id": self.invocation_id}

    def emit(self, kind, *, scope=None, data=None, **fields):
        if self.sink is not None:
            self.sink.event(kind, scope=self.scope() if scope is None else scope, data=data, **fields)

    def invocation(self, argv, **fields):
        with self.lock:
            self.invocation_id = uuid.uuid4().hex
            self.offsets = {}
            self.exited = False
        self.emit('claude_invocation', argv=list(argv), **fields)

    def exit(self, code):
        with self.lock:
            if self.exited:
                return
            self.exited = True
        self.emit('process_exit', returncode=code)

    def data(self, stream, data):
        if self.sink is None or not self.sink.enabled:
            return
        # Chunk stdin too: a large cold prompt must not become an unbounded row.
        for start in range(0, len(data), 16384):
            chunk = data[start:start + 16384]
            with self.lock:
                offset = self.offsets.get(stream, 0)
                self.offsets[stream] = offset + len(chunk)
                scope = self.scope()
            self.emit('claude_' + stream, scope=scope, data=chunk,
                      stream_offset=offset, byte_count=len(chunk))

    def register(self, calls):
        if self.sink is None:
            return
        with self.lock:
            for ident, _, _ in calls:
                self.origins[ident] = dict(self.active)
        self.emit('mailbox_register', native_tool_use_ids=[call[0] for call in calls])

    def payload(self, kind, data, *, scope, **fields):
        if self.sink is None or not self.sink.enabled:
            return
        for offset in range(0, max(1, len(data)), 16384):
            chunk = data[offset:offset + 16384]
            self.emit(kind, scope=scope, data=chunk, payload_offset=offset,
                      byte_count=len(chunk), payload_end=offset + len(chunk) == len(data), **fields)
