"""Experimental stdlib-only model adapter for this repository's claude-relay.

No Claude binary is installed/discovered/run as the caller. Text, MCP handoff and
Pi lifecycles were live-checked with 2.1.289; see claude-relay/SAMPLING.md for limits.
Pin expected_version and recheck the profile after native upgrades. Missing native
correlation or permissions fail closed, never by matching tool names/arguments.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
import secrets
import threading
import time

from ..model import ModelError, ModelConfigurationError, ModelResponseError, ModelTransportError, ModelContinuationExpired, ModelSample, _timed_sample
from ..items import ModelFailure, Message
from .._tool_spec import ToolSpec
from ..model_catalog import EndpointSpec, ModelBinding
from ._context import Snapshot, record, handoff
from ._cli_protocol import PROFILE
from ._runtime import Runtime
from ._sampling import resolve_extra, resolve_sampling
from ._recovery import NativeMCPTimeout, NATIVE_MCP_TIMEOUT
from .._debug_trace import DebugTrace, capture_trace_scope
from ..timeouts import DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS
from ..timeouts import DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS
from ..timeouts import DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS
from ..timeouts import DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS
from ..timeouts import validate_timeout_seconds


@dataclass(frozen=True)
class ClaudeRelayEndpoint:
    """One relay continuation's launch settings.

    ``generation_timeout_seconds`` bounds each wait for the next completed
    native message (streamed chunks and tool heartbeats do not extend it);
    ``parked_timeout_seconds`` bounds a parked continuation's wait for tool
    results; ``startup``/``stop`` bound the native process lifecycle.
    """

    model: str
    launcher: str
    socket_path: str
    server_uid: int
    expected_version: str
    tool_id_pointer: str = '/params/_meta/claudecode~1toolUseId'
    generation_timeout_seconds: float = DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS
    parked_timeout_seconds: float = DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS
    startup_timeout_seconds: float = DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS
    stop_timeout_seconds: float = DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS
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
            try:
                validate_timeout_seconds(getattr(self, name), name)
            except ValueError as error:
                raise ModelConfigurationError(str(error)) from None
        if self.binding is None:
            object.__setattr__(self, 'binding', ModelBinding(self.model, EndpointSpec('claude-relay', None, self.model, 'runtime')))
        if self.binding.api != 'claude-relay' or self.binding.endpoint.model != self.model:
            raise ModelConfigurationError('Wrong Claude Relay binding')
        if self.binding.spec is not None and self.binding.spec.responses is not None:
            raise ModelConfigurationError('Claude Relay does not accept Responses settings')
        resolve_extra(self.binding.extra_sample_params)


class ClaudeRelayModel:
    auto_compaction_owner = 'host'

    def __init__(self, endpoint: ClaudeRelayEndpoint, *, trace=None):
        if trace is not None and (not isinstance(trace, DebugTrace) or trace.event_path is None):
            raise ModelConfigurationError('Claude Relay tracing requires a DebugTrace with events=True')
        self.trace = trace
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
        started = time.monotonic()
        with self._sample_lock:
            with self._state_lock:
                if self._closed:
                    raise ModelConfigurationError('ClaudeRelayModel is closed')
                runtime = self._runtime
                epoch = self._epoch
                previous_signature, acknowledged, pending = self._signature, self._acknowledged, self._pending
            scope = {**capture_trace_scope(), 'sample_id': secrets.token_hex(16)}
            executing = None
            results = snapshot = None
            try:
                try:
                    revision = len(context)
                except Exception:
                    revision = None  # diagnostics must not add a new context requirement
                scope['model_context_revision'] = revision
                scope.setdefault('context_revision', revision)  # retain Pi's source revision
                if self.trace:
                    self.trace.event('sample_begin', scope=scope, model=self.endpoint.model)
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
                    if runtime is not None and isinstance(runtime.error, NativeMCPTimeout):
                        runtime.trace.emit('continuation_abandoned', reason=NATIVE_MCP_TIMEOUT,
                                           replacement='context_or_policy_change')
                    epoch = self._retire(expected_epoch=epoch)
                    snapshot.prompt()  # reject unresolved/unsupported cold history before launch
                    # Don't let caller-owned nested schema containers change the
                    # live catalog after the immutable signature was captured.
                    fixed_tools = tuple(ToolSpec(row['name'], row['description'], row['schema'])
                                        for row in json.loads(snapshot.catalog))
                    runtime = Runtime(self.endpoint, fixed_tools, secrets.token_hex(12), sampling=sampling, trace=self.trace)
                    executing = runtime
                    runtime.begin_sample(started, scope)
                    with self._state_lock:
                        if self._closed or self._epoch != epoch:
                            runtime.close()
                            raise ModelTransportError('Claude Relay was retired before launch')
                        self._runtime = runtime
                        self._signature = signature
                    runtime.start(snapshot)
                else:
                    executing = runtime
                    runtime.begin_sample(started, scope)
                    runtime.trace.emit('continuation_reuse')
                    runtime.diagnostics.set_phase('awaiting_model_message')
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
                runtime.trace.emit('sample_end', scope={**runtime.trace.scope(), **scope},
                                   stop_reason=message.stop_reason,
                                   input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
                if calls:
                    runtime.trace.parked()
                return sample
            except ModelError as error:
                if error.failure is None:
                    diagnostic = executing.diagnostics.snapshot(freeze=True) if executing is not None else None
                    message = str(error)
                    if diagnostic is not None:
                        message = (f'{message[:512]} ({diagnostic.detail()}; codec={PROFILE}; '
                                   f'last_event={diagnostic.last_event_type or "none"})')
                    error.failure = ModelFailure(
                        category=type(error).__name__, message=message, provider='claude-relay',
                        model=self.endpoint.model, auth_source='runtime',
                        event_types=diagnostic.event_types if diagnostic else (),
                        event_count=diagnostic.event_count if diagnostic else 0,
                        last_event_type=diagnostic.last_event_type if diagnostic else None,
                        error_code=(NATIVE_MCP_TIMEOUT if isinstance(error, NativeMCPTimeout)
                                    else diagnostic.error_code if diagnostic else None),
                        elapsed_seconds=diagnostic.elapsed_seconds if diagnostic else time.monotonic() - started,
                    )
                eligible = (isinstance(error, NativeMCPTimeout) and executing is runtime
                            and runtime is not None and bool(pending) and results is not None
                            and error.call.native_id in pending.values() and not error.completed_items)
                if eligible:
                    try:
                        context.assert_model_ready()
                        snapshot.prompt()  # no guessed/missing outcomes in the cold input
                    except (AttributeError, TypeError, ValueError, ModelError):
                        eligible = False
                retired_epoch = self._retire_after_failure(error, expected_epoch=epoch if eligible else None)
                with self._state_lock:
                    recoverable = (eligible and retired_epoch is not None and not self._closed
                                   and self._epoch == retired_epoch and runtime.retirement_complete)
                if recoverable:
                    error = ModelContinuationExpired(str(error), failure=replace(
                        error.failure, category='ModelContinuationExpired'))
                if self.trace:
                    if isinstance(error, (NativeMCPTimeout, ModelContinuationExpired)):
                        self.trace.event('continuation_recovery_eligibility', scope=scope,
                                         runtime_id=runtime.generation if runtime else None,
                                         eligible=recoverable, cleanup_confirmed=(
                                             runtime.retirement_complete if runtime else False))
                    self.trace.event('sample_end', scope=scope,
                                     runtime_id=runtime.generation if runtime else None,
                                     exception_type=type(error).__name__,
                                     last_event_type=error.failure.last_event_type,
                                     elapsed_seconds=error.failure.elapsed_seconds,
                                     event_count=error.failure.event_count, error_code=error.failure.error_code)
                raise error
            except BaseException as error:
                if self.trace:
                    self.trace.event('sample_end', scope=scope, exception_type=type(error).__name__)
                self._retire_after_failure(error)
                raise

    def _retire_after_failure(self, error, *, expected_epoch=None):
        try:
            return self._retire(expected_epoch=expected_epoch)
        except Exception as cleanup:
            # Keep the causal error, but never claim that teardown succeeded.
            if isinstance(error, ModelError) and error.failure is not None:
                error.failure = replace(error.failure, message=error.failure.message[:880]
                                        + f'; cleanup failed ({type(cleanup).__name__[:64]})')
            if self.trace:
                self.trace.event('cleanup_failure', exception_type=type(cleanup).__name__)
            return None

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
