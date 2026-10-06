"""Bounded, payload-free native diagnostics. No raw records survive here."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import re
import threading
import time


def identifier(value):
    if value is None:
        return '<missing>'
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', value):
        return value
    return '<invalid>'


def event_identity(record):
    kind = identifier(record.get('type'))
    parts = [kind]
    error = record.get('error')
    if kind == 'tool_progress' and record.get('heartbeat') is True:
        parts.append('heartbeat')  # classification, not parent/payload validation
    if kind == 'system':
        parts.append(identifier(record.get('subtype')))
        if record.get('subtype') == 'status':
            parts.append(identifier(record.get('status')))
    if kind == 'stream_event':
        event = record.get('event')
        if not isinstance(event, dict):
            parts.append('<invalid>' if event is not None else '<missing>')
        else:
            parts.append(identifier(event.get('type')))
            if event.get('type') == 'content_block_delta':
                delta = event.get('delta')
                parts.append(identifier(delta.get('type')) if isinstance(delta, dict) else '<invalid>')
            if event.get('type') == 'error':
                error = event.get('error')
    if isinstance(error, dict):
        code = identifier(error.get('code', error.get('type')))
    elif isinstance(error, str) and error in (
            'authentication_failed', 'context_length_exceeded', 'prompt_too_long'):
        code = error  # the runtime's explicitly recognized bare-code forms
    else:
        code = None  # a bare error string might be provider text, not a code
    return '/'.join(parts), code


@dataclass(frozen=True)
class FailureSnapshot:
    phase: str
    elapsed_seconds: float
    event_count: int
    invocation_event_count: int
    event_types: tuple
    error_code: str | None
    last_byte_age: float | None
    last_record_age: float | None
    last_completion_age: float | None
    native_version: str | None

    @property
    def last_event_type(self):
        return self.event_types[-1] if self.event_types else None

    def detail(self):
        def age(value):
            return 'none' if value is None else f'{value:.3f}s'
        return (f'phase={self.phase}; elapsed={self.elapsed_seconds:.3f}s; '
                f'last_stdout_byte_age={age(self.last_byte_age)}; '
                f'last_record_age={age(self.last_record_age)}; '
                f'last_message_completion_age={age(self.last_completion_age)}; '
                f'native_version={self.native_version or "unobserved"}')


class Diagnostics:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.frozen = None
        self.total = 0
        self.last_byte = self.last_record = self.last_completion = None
        self.native_version = None
        self.begin()

    def begin(self, started=None):
        with self.lock:
            if self.frozen is not None:
                return  # preserve failures observed between host calls
            self.started = self.clock() if started is None else started
            self.phase = 'starting'
            self.count = 0
            self.tail = deque(maxlen=64)
            self.error_code = None

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def invocation(self):
        with self.lock:
            self.last_byte = self.last_record = self.last_completion = None

    def received(self):
        with self.lock:
            self.last_byte = self.clock()

    def record(self, value):
        label, code = event_identity(value)
        with self.lock:
            self.count += 1
            self.total += 1
            self.tail.append(label)
            self.last_record = self.clock()
            if code is not None:
                self.error_code = code
        return label

    def malformed(self, label):
        self.record({'type': label})  # fixed local category, never raw input

    def completed(self):
        with self.lock:
            self.last_completion = self.clock()

    def snapshot(self, *, freeze=False):
        with self.lock:
            if self.frozen is not None:
                return self.frozen
            now = self.clock()
            def age(value):
                return None if value is None else max(0.0, now - value)
            result = FailureSnapshot(self.phase, max(0.0, now - self.started), self.count,
                                     self.total, tuple(self.tail), self.error_code,
                                     age(self.last_byte), age(self.last_record), age(self.last_completion),
                                     self.native_version)
            if freeze:
                self.frozen = result
            return result
