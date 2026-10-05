"""Experimental stdlib-only model adapter for this repository's claude-relay.

No Claude binary is installed/discovered/run as the caller. Text, MCP handoff and
Pi lifecycles were live-checked with 2.1.289; see claude-relay/SAMPLING.md for limits.
Pin expected_version and recheck the profile after native upgrades. Missing native
correlation or permissions fail closed, never by matching tool names/arguments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import json
from pathlib import Path
import re
import secrets
import threading

from ..model import ModelError, ModelConfigurationError, ModelResponseError, ModelTransportError, ModelSample, _timed_sample
from ..items import ModelFailure, Message
from .._tool_spec import ToolSpec
from ..model_catalog import EndpointSpec, ModelBinding
from ._context import Snapshot, record, handoff
from ._runtime import Runtime
from ._sampling import resolve_extra, resolve_sampling


@dataclass(frozen=True)
class ClaudeRelayEndpoint:
    model: str
    launcher: str
    socket_path: str
    server_uid: int
    expected_version: str
    tool_id_pointer: str = '/params/_meta/claudecode~1toolUseId'
    generation_timeout_seconds: float = 300
    parked_timeout_seconds: float = 1800
    startup_timeout_seconds: float = 30
    stop_timeout_seconds: float = 5
    binding: ModelBinding | None = field(default=None, repr=False)

    def __post_init__(self):
        if (not isinstance(self.model, str) or not self.model.strip() or self.model.startswith('-')
                or any(c.isspace() or ord(c) < 32 for c in self.model)):
            raise ModelConfigurationError('Claude Relay requires a native model ID')
        for name, flag, variable, target in (
                ('launcher', '--claude-relay-launcher', 'CLAUDE_RELAY_LAUNCHER', 'A-side claude_relay.py'),
                ('socket_path', '--claude-relay-socket', 'CLAUDE_RELAY_SOCKET', 'B-side broker socket')):
            value = getattr(self, name)
            hint = (f'Set {flag} or export {variable} before starting Pythia '
                    f'(absolute path to {target}); .env files are not loaded automatically.')
            if value is None or value == '':
                raise ModelConfigurationError(f'Claude Relay {name} is not configured. {hint}')
            if not isinstance(value, str) or not value or '\x00' in value or not Path(value).is_absolute():
                raise ModelConfigurationError(f'Claude Relay {name} must be an absolute path. {hint}')
        if type(self.server_uid) is not int or self.server_uid <= 0:
            raise ModelConfigurationError('Claude Relay requires an expected non-root server UID')
        if not isinstance(self.expected_version, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+-]{0,63}', self.expected_version):
            raise ModelConfigurationError('Set the expected native CLI version before launching')
        if (not isinstance(self.tool_id_pointer, str) or not self.tool_id_pointer.startswith('/params/_meta/')
                or re.search(r'~(?![01])', self.tool_id_pointer)):
            raise ModelConfigurationError('Native tool ID pointer must select request params._meta')
        for name in ('generation_timeout_seconds', 'parked_timeout_seconds', 'startup_timeout_seconds', 'stop_timeout_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ModelConfigurationError(f'{name} must be positive and finite')
        if self.binding is None:
            object.__setattr__(self, 'binding', ModelBinding(self.model, EndpointSpec('claude-relay', None, self.model, 'runtime')))
        if self.binding.api != 'claude-relay' or self.binding.endpoint.model != self.model:
            raise ModelConfigurationError('Wrong Claude Relay binding')
        if self.binding.spec is not None and self.binding.spec.responses is not None:
            raise ModelConfigurationError('Claude Relay does not accept Responses settings')
        resolve_extra(self.binding.extra_sample_params)


class ClaudeRelayModel:
    auto_compaction_owner = 'host'

    def __init__(self, endpoint: ClaudeRelayEndpoint):
        self.endpoint = endpoint
        self.binding = endpoint.binding
        self._sample_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._runtime = None
        self._closed = False
        self._signature = None
        self._acknowledged = ()
        self._pending = {}
        self._stderr = b''
        self._epoch = 0

    @property
    def stderr_tail(self):
        """Explicit private diagnostics ONLY; never automatically log/save these bytes."""
        with self._state_lock:
            return bytes(self._runtime.stderr_tail) if self._runtime is not None else self._stderr

    @_timed_sample
    def sample(self, context, *, tools=(), sample_params=None):
        with self._sample_lock:
            with self._state_lock:
                if self._closed:
                    raise ModelConfigurationError('ClaudeRelayModel is closed')
                runtime = self._runtime
                epoch = self._epoch
                previous_signature, acknowledged, pending = self._signature, self._acknowledged, self._pending
            try:
                sampling = resolve_sampling(self.endpoint.binding, sample_params)
                tools = tuple(tools)
                try:
                    snapshot = Snapshot.build(context, tools)
                except ModelError:
                    raise
                except (TypeError, ValueError, RecursionError):
                    raise ModelConfigurationError('Context/catalog is not representable by the text codec') from None
                signature = (self.endpoint, sampling, snapshot.instructions, snapshot.catalog)
                results = None
                if runtime is not None and signature == previous_signature:
                    results = handoff(snapshot, acknowledged, pending)
                if results is None:
                    epoch = self._retire(expected_epoch=epoch)
                    snapshot.prompt()  # reject unresolved/unsupported cold history before launch
                    # Don't let caller-owned nested schema containers change the
                    # live catalog after the immutable signature was captured.
                    fixed_tools = tuple(ToolSpec(row['name'], row['description'], row['schema'])
                                        for row in json.loads(snapshot.catalog))
                    runtime = Runtime(self.endpoint, fixed_tools, secrets.token_hex(12), sampling=sampling)
                    with self._state_lock:
                        if self._closed or self._epoch != epoch:
                            runtime.close()
                            raise ModelTransportError('Claude Relay was retired before launch')
                        self._runtime = runtime
                        self._signature = signature
                    runtime.start(snapshot)
                else:
                    with self._state_lock:
                        if self._epoch != epoch or self._runtime is not runtime:
                            raise ModelTransportError('Claude Relay was retired before handoff')
                        runtime.mailbox.release(results)
                message, items, calls = runtime.next_message()
                with self._state_lock:
                    if self._runtime is not runtime or self._closed or self._epoch != epoch:
                        raise ModelConfigurationError('Claude Relay continuation was retired during sampling')
                    if calls:
                        self._acknowledged = snapshot.records + tuple(record(item) for item in items)
                        self._pending = {runtime.generation + ':' + ident: ident for ident, _, _ in calls}
                if not calls and not any(isinstance(item, Message) and item.content_text.strip() for item in items):
                    raise ModelResponseError('Native run has no final assistant text')
                sample = ModelSample(items, stop_reason=message.stop_reason, usage=message.usage,
                                     provider_turn_id=message.id)
                if not calls:
                    self.retire()
                return sample
            except ModelError as error:
                if error.failure is None:
                    error.failure = ModelFailure(
                        category=type(error).__name__, message=str(error), provider='claude-relay',
                        model=self.endpoint.model, auth_source='runtime',
                        event_types=tuple(runtime.event_types) if runtime is not None else (),
                        event_count=len(runtime.event_types) if runtime is not None else 0,
                    )
                self.retire()
                raise
            except BaseException:
                self.retire()
                raise

    def retire(self):
        """Cancel a live continuation without waiting for the sampling lock."""
        self._retire()

    def _retire(self, expected_epoch=None):
        with self._state_lock:
            if expected_epoch is not None and expected_epoch != self._epoch:
                raise ModelTransportError('Claude Relay was retired before launch')
            self._epoch += 1
            epoch = self._epoch
            runtime, self._runtime = self._runtime, None
            self._pending, self._acknowledged, self._signature = {}, (), None
        if runtime is not None:
            try:
                runtime.close()
            finally:
                self._stderr = bytes(runtime.stderr_tail)
        return epoch

    def close(self):
        with self._state_lock:
            self._closed = True
        self.retire()
