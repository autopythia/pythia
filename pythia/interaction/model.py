from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import wraps
from time import perf_counter
from typing import Callable
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Mapping

from .context import InteractionContext
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import SampleMetadata
from .items import _validate_elapsed_seconds
from .usage import TokenUsage
from .model_catalog import PROMPT_CACHING_VALUES
from .model_catalog import SAMPLE_PARAM_DEFAULT_FIELDS
from .model_catalog import freeze_extra_sample_params
from .model_catalog import thaw_json
from .model_catalog import validate_temperature
from .model_catalog import validate_top_p
from .timeouts import validate_timeout_seconds

if TYPE_CHECKING:
    from .display import DisplayItem
    from .environment import ToolSpec
    from .model_catalog import ModelBinding


class RetryState(enum.Enum):
    """Whether the turn loop may sample a failed sample's context again by
    itself: ``ModelError.retry_state``. The values are for traces.

    It only moves forward: from ``NONE`` to ``ELIGIBLE``, when a model adapter
    vouches for a retry, and from ``ELIGIBLE`` to ``BLOCKED``, ``RETRIED``,
    ``EXHAUSTED`` or ``CANCELLED``. So an error a frontend sees is ``NONE``,
    ``BLOCKED``, ``EXHAUSTED`` or ``CANCELLED``: the turn loop settles an
    eligible one before raising it.
    """

    NONE = "none"  # no retry is claimed (the default)
    # The adapter vouches that nothing from the attempt was published or run,
    # so the same context may be sampled again: set by the error's class
    # (NativeToolRejected, ModelContinuationExpired) or by the adapter once
    # it has checked.
    ELIGIBLE = "eligible"
    # Eligible, but the retry couldn't be set up: the adapter's teardown of
    # the attempt failed or wasn't confirmed, or the turn loop's own checks
    # failed.
    BLOCKED = "blocked"
    RETRIED = "retried"  # the turn loop sampled again after it
    EXHAUSTED = "exhausted"  # eligible, but the sample's attempts were used up
    # Eligible, but the user stopped the automatic retries (the CLI's /cancel).
    CANCELLED = "cancelled"


_RETRY_MOVES = {
    RetryState.NONE: (RetryState.ELIGIBLE,),
    RetryState.ELIGIBLE: (RetryState.BLOCKED, RetryState.RETRIED, RetryState.EXHAUSTED,
                          RetryState.CANCELLED),
}


class ModelError(RuntimeError):
    """Base model failure with optional safe diagnostics and completed output.

    ``retry_state`` (``RetryState``) says whether the turn loop may sample the
    same context again by itself. A new error is ``NONE`` or ``ELIGIBLE``;
    assigning it later moves it on, as the Claude relay and the turn loop do,
    and a move that ``RetryState`` doesn't allow raises ValueError.
    """

    def __init__(
        self,
        *args,
        failure: Optional[ModelFailure] = None,
        completed_items: Sequence[InteractionItem] = (),
        retry_state: RetryState = RetryState.NONE,
    ) -> None:
        super().__init__(*args)
        if failure is not None and not isinstance(failure, ModelFailure):
            raise TypeError("failure must be ModelFailure or None")
        items = tuple(completed_items)
        for index, item in enumerate(items):
            if not isinstance(item, (Message, Reasoning, ToolCall, OpaqueCompaction)):
                raise TypeError(
                    "completed_items must contain model output items; "
                    f"item {index} is {type(item).__name__}"
                )
        if not isinstance(retry_state, RetryState):
            raise TypeError("retry_state must be a RetryState")
        if retry_state not in (RetryState.NONE, RetryState.ELIGIBLE):
            raise ValueError("A new ModelError's retry_state must be NONE or ELIGIBLE")
        self.failure = failure
        self.completed_items = items
        self._retry_state = retry_state

    @property
    def retry_state(self) -> RetryState:
        return self._retry_state

    @retry_state.setter
    def retry_state(self, state: RetryState) -> None:
        if not isinstance(state, RetryState):
            raise TypeError("retry_state must be a RetryState")
        if state not in _RETRY_MOVES.get(self._retry_state, ()):
            raise ValueError(
                f"retry_state can't move from {self._retry_state.name} to {state.name}"
            )
        self._retry_state = state


class ModelConfigurationError(ModelError, ValueError):
    pass


class ModelTransportError(ModelError):
    pass


class ModelCancelled(ModelTransportError):
    """A request that ``retire_model`` or ``close_model`` cancelled; not a failure.

    ``completed_items`` holds output that completed before the cancel (from a
    stream), and ``failure`` describes the cancelled request (category
    ``request_cancelled``). Neither is set when a closed model refused a call
    that sent nothing. The turn loop treats it like any failure caused by a
    preempting steer or stop: it is saved, never retried.
    """


class ModelContinuationExpired(ModelTransportError):
    """A retired continuation can be replaced from checkpointed host history.

    Providers may raise this only after confirming cleanup and complete actual
    host outcomes. It grants no tool replay or generic transport-retry authority.
    It is ``RetryState.ELIGIBLE``: the shared turn loop samples again, within
    the sample's attempts, as for any eligible failure (``loop.kernel``).
    """

    def __init__(self, *args, failure=None, completed_items=(),
                 retry_state=RetryState.ELIGIBLE):
        if not isinstance(failure, ModelFailure):
            raise TypeError("Continuation recovery requires a safe ModelFailure")
        if tuple(completed_items):
            raise ValueError("Continuation recovery cannot carry completed output")
        super().__init__(*args, failure=failure, retry_state=retry_state)


class ModelAuthenticationError(ModelTransportError):
    pass


class ModelThinkingBindingError(ModelTransportError):
    """The provider rejected a signed thinking block made for another prefix.

    Anthropic binds each thinking block to the conversation before it, the
    tools offered included, and answers a request that changed them with HTTP
    400 ("Invalid `signature` in `thinking` block. The block is bound to a
    different conversation."). Sending the same request again fails the same
    way; a request without the block doesn't. Callers that can't send one
    without it see a ``ModelTransportError``, as before; warmpi sends pi's
    request instead, which carries no signed blocks.
    """


class ModelTimeoutError(ModelTransportError):
    pass


class ModelRateLimitError(ModelTransportError):
    """The provider rate-limited the request, or reported an exhausted quota or
    plan, after the adapter's own retries: HTTP 429, or such a code in a stream.

    Nothing of the response was generated, so it is ``RetryState.ELIGIBLE``:
    the turn loop samples again under its rate-limit policy, as many times as
    the config's ``endpoint_rate_limit_retries`` allows (``loop.kernel``).
    That includes a quota, such as OpenAI's ``insufficient_quota``: a top-up, a
    plan reset or another account may lift it meanwhile. Its failure says
    ``quota exceeded`` instead of ``rate limited``. ``retry_after_seconds`` is
    the last response's ``Retry-After``, uncapped, or None.
    """

    def __init__(self, *args, failure=None, completed_items=(),
                 retry_state=RetryState.ELIGIBLE, retry_after_seconds=None):
        if retry_after_seconds is not None and (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, (int, float))
                or not math.isfinite(retry_after_seconds) or retry_after_seconds < 0):
            raise ValueError("retry_after_seconds must be a non-negative number or None")
        super().__init__(*args, failure=failure, completed_items=completed_items,
                         retry_state=retry_state)
        self.retry_after_seconds = (None if retry_after_seconds is None
                                    else float(retry_after_seconds))


class ModelResponseError(ModelError):
    pass


class ModelContextWindowError(ModelResponseError):
    pass


def _one_line(value: object, limit: int) -> Optional[str]:
    """An untrusted provider string as one bounded line, or None."""
    if not isinstance(value, str):
        return None
    text = " ".join("".join(c if c.isprintable() else " " for c in value).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text or None


class ModelRefusal(ModelResponseError):
    """A provider stopped a response with a refusal, leaving no answer to adopt.

    The reason is the provider's stop details, as sent: on Anthropic's APIs,
    ``stop_details`` with a type ("refusal"), a category such as "cyber", and
    an explanation. Any type or category is reported as given, never matched
    against a list. Adapters raise it when the refused response has no answer
    text (thinking alone is not one) or has tool calls, which must never run;
    nothing from that response is adopted. A refusal after answer text is not
    this error: the adapter returns the text with ``stop_reason`` "refusal".
    The failure's ``error_code`` is ``refusal``. The turn loop saves it like
    any failure, and a retry samples the same context again.
    """

    error_code = "refusal"

    def __init__(self, details: object = None, *,
                 failure: Optional[ModelFailure] = None,
                 retry_state: RetryState = RetryState.NONE) -> None:
        details = details if isinstance(details, Mapping) else {}
        self.kind = _one_line(details.get("type"), 64)
        self.category = _one_line(details.get("category"), 64)
        self.explanation = _one_line(details.get("explanation"), 400)
        tags = [f"{name}={value}" for name, value in (
            ("type", None if self.kind == "refusal" else self.kind),
            ("category", self.category)) if value]
        message = "Request refused" + (f" ({', '.join(tags)})" if tags else "")
        if self.explanation:
            message += ": " + self.explanation
        elif not tags:
            message += " (no reason given)"
        super().__init__(message, failure=failure, retry_state=retry_state)


@dataclass(frozen=True)
class SampleParams:
    """Per-call parameters of ``Model.sample()``.

    Every ``None`` field (and an empty ``stop``) inherits the model's default,
    with one precedence order: this call, then adapter settings such as a
    Messages endpoint's budget or compaction trigger, then the model binding
    and catalog (``temperature`` and ``top_p`` from the catalog's
    ``sample_params.*``). Adapters reject typed fields that their API cannot
    send.

    ``extra`` holds request-body extensions for this call. A mapping replaces
    the binding's ``extra_sample_params`` (``{}`` sends none); ``None``
    inherits them.

    ``request_timeout_seconds`` is host transport policy, never a wire field:
    the HTTP blocking-I/O timeout of each operation of this call's requests.
    ``None`` uses the adapter endpoint's timeout.

    ``prompt_caching`` is the Messages prompt-caching policy of this call's
    requests (``5m``, ``1h``, or ``none``, which sends no ``cache_control``);
    ``None`` uses the endpoint's. The CLI projects ``/config
    endpoint_prompt_caching``. Other adapters ignore it.
    """

    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Tuple[str, ...] = ()
    seed: Optional[int] = None
    # Host/provider context-management control, never a sampling wire field.
    enable_auto_compaction: Optional[bool] = None
    # Host context-management threshold override, never a sampling wire field.
    auto_compaction_context_tokens: Optional[int] = None
    extra: Optional[Mapping] = None
    request_timeout_seconds: Optional[float] = None
    prompt_caching: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None:
            if (
                isinstance(self.max_output_tokens, bool)
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            ):
                raise ValueError(
                    "max_output_tokens must be a positive integer or None"
                )

        if self.temperature is not None:
            object.__setattr__(self, "temperature", validate_temperature(self.temperature))
        if self.top_p is not None:
            object.__setattr__(self, "top_p", validate_top_p(self.top_p))

        stop = tuple(self.stop)
        for index, value in enumerate(stop):
            if not isinstance(value, str):
                raise TypeError(f"stop[{index}] must be a string")
        object.__setattr__(self, "stop", stop)

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if self.enable_auto_compaction is not None and not isinstance(
            self.enable_auto_compaction,
            bool,
        ):
            raise TypeError(
                "enable_auto_compaction must be a bool or None"
            )
        if self.auto_compaction_context_tokens is not None and (
            isinstance(self.auto_compaction_context_tokens, bool)
            or not isinstance(self.auto_compaction_context_tokens, int)
            or self.auto_compaction_context_tokens <= 0
        ):
            raise ValueError(
                "auto_compaction_context_tokens must be a positive integer or None"
            )
        if self.extra is not None:
            object.__setattr__(self, "extra", freeze_extra_sample_params(self.extra))
        if self.request_timeout_seconds is not None:
            object.__setattr__(self, "request_timeout_seconds", validate_timeout_seconds(
                self.request_timeout_seconds, "request_timeout_seconds",
            ))
        if self.prompt_caching is not None and self.prompt_caching not in PROMPT_CACHING_VALUES:
            raise ValueError("prompt_caching must be 5m, 1h, none, or None")


def _request_timeout_seconds(endpoint, sample_params: Optional[SampleParams]) -> float:
    """One call's HTTP timeout: the per-call value, else the endpoint's."""
    timeout = None if sample_params is None else sample_params.request_timeout_seconds
    return endpoint.request_timeout_seconds if timeout is None else timeout


def _timeout_message(message: str, timeout_seconds: Optional[float]) -> str:
    """Name the budget a timed-out request used; the value is not a secret."""
    if timeout_seconds is None:
        return message
    return f"{message} (request_timeout_seconds={timeout_seconds:g})"


def _with_sample_param_defaults(
    binding: "ModelBinding",
    sample_params: Optional[SampleParams],
) -> Optional[SampleParams]:
    """This call's params, with the binding's catalog temperature and top_p where unset.

    A per-call value, 0 included, takes precedence over the catalog's
    ``sample_params.*``. Adapters apply this when they encode their typed
    fields, so every request gets it, compaction requests included.
    """
    defaults = binding.sample_params
    changes = {
        name: getattr(defaults, name)
        for name in SAMPLE_PARAM_DEFAULT_FIELDS
        if getattr(defaults, name) is not None
        and (sample_params is None or getattr(sample_params, name) is None)
    }
    if not changes:
        return sample_params
    return replace(sample_params or SampleParams(), **changes)


def _apply_extra_sample_params(
    payload: dict,
    binding: "ModelBinding",
    sample_params: Optional[SampleParams],
) -> None:
    """Add one request's body extensions without replacing adapter fields.

    A per-call ``extra`` map replaces the binding's map, even when empty;
    ``None`` params or ``extra`` use the binding's map. Per-call maps are
    validated without an API, so the names reserved by this binding's API are
    checked here before sending.
    """
    override = None if sample_params is None else sample_params.extra
    if override is None:
        params = binding.extra_sample_params
    else:
        try:
            params = freeze_extra_sample_params(override, binding.api)
        except ValueError as exc:
            raise ModelConfigurationError(str(exc)) from None
    extensions = thaw_json(params)
    # Typed and structural fields stay authoritative; never override silently.
    replaced = sorted(extensions.keys() & payload.keys())
    if replaced:
        raise ModelConfigurationError(
            "extra_sample_params cannot replace adapter-owned request fields: "
            + ", ".join(replaced)
        )
    payload.update(extensions)


@dataclass(frozen=True)
class ModelSample:
    """One model response, with metadata for saving and for the turn loop.

    ``results_due`` is a ``time.monotonic()`` value by which the model needs
    the results of this sample's tool calls, or None. A Claude relay model sets
    it (the native client gives up on its first MCP call after a fixed time);
    the turn loop gives it to each call (``results_due_scope``). It is a
    scheduling hint: not saved, and not part of equality.
    """

    items: Tuple[InteractionItem, ...]
    stop_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()
    results_due: Optional[float] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ModelResponseError("model sample must contain at least one item")
        for index, item in enumerate(items):
            if not isinstance(
                item,
                (Message, Reasoning, ToolCall, OpaqueCompaction),
            ):
                raise ModelResponseError(
                    "model sample items must be assistant messages, reasoning, "
                    "tool calls, or opaque compactions; "
                    f"item {index} is {type(item).__name__}"
                )
            if isinstance(item, Message) and item.role != "assistant":
                raise ModelResponseError(
                    f"model message at item {index} must have role 'assistant'"
                )
        object.__setattr__(self, "items", items)

        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        object.__setattr__(
            self, "elapsed_seconds", _validate_elapsed_seconds(self.elapsed_seconds)
        )
        if (
            isinstance(self.request_attempts, bool)
            or not isinstance(self.request_attempts, int)
            or self.request_attempts <= 0
        ):
            raise ValueError("request_attempts must be a positive integer")
        recovery = tuple(self.recovery)
        for index, value in enumerate(recovery):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"recovery[{index}] must be a non-empty string")
            if "\r" in value or "\n" in value:
                raise ValueError(f"recovery[{index}] must not contain newlines")
        object.__setattr__(self, "recovery", recovery)
        due = self.results_due
        if due is not None:
            if (isinstance(due, bool) or not isinstance(due, (int, float))
                    or not math.isfinite(due)):
                raise ValueError("results_due must be a finite monotonic time or None")
            object.__setattr__(self, "results_due", float(due))
        for field_name in (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")

    @property
    def tool_calls(self) -> Tuple[ToolCall, ...]:
        return tuple(item for item in self.items if isinstance(item, ToolCall))

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return this sample's output, sample metadata, and durable boundary."""
        return (
            *self.items,
            self._metadata(),
            ModelSampleBoundary(),
        )

    def _metadata(self) -> SampleMetadata:
        return SampleMetadata(
            usage=self.usage,
            provider_session_id=self.provider_session_id,
            provider_turn_id=self.provider_turn_id,
            provider_turn_state=self.provider_turn_state,
            elapsed_seconds=self.elapsed_seconds,
            request_attempts=self.request_attempts,
            recovery=self.recovery,
        )

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(
            (*self.items, self._metadata()),
        )

    @property
    def assistant_messages(self) -> Tuple[Message, ...]:
        return tuple(item for item in self.items if isinstance(item, Message))

    @property
    def last_assistant_text(self) -> Optional[str]:
        for item in reversed(self.items):
            if isinstance(item, Message) and item.content_text:
                return item.content_text
        return None


def _timed_sample(method: Callable[..., ModelSample]) -> Callable[..., ModelSample]:
    """Measure a complete adapter call, including its response cleanup.

    Timing starts inside the calling worker, not while waiting for one. Errors
    propagate unchanged and do not produce a sample or fabricated usage data.
    """
    @wraps(method)
    def measured(*args, **kwargs) -> ModelSample:
        started = perf_counter()
        sample = method(*args, **kwargs)
        return replace(sample, elapsed_seconds=perf_counter() - started)

    return measured


class Model(Protocol):
    def sample(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
    ) -> ModelSample:
        ...


def retire_model(model) -> None:
    """Optional, idempotent cancellation of the model's work in flight.

    That is a Claude relay continuation, or an HTTP model's request in flight.
    Any thread may call it, also while a sample runs; that sample then fails
    soon. The model stays usable: its next sample starts a new continuation or
    request. The turn loop retires at each turn's end except after a final
    answer, when the Claude relay keeps its process for the next turn, and
    after a yield (``park_model``); at the end of each compaction; and to
    cancel the sample in flight for a preempting steer or stop. A preempting
    stop also closes the model (see ``close_model``).
    """
    retire = getattr(model, "retire", None)
    if callable(retire):
        retire()


def park_model(model) -> None:
    """After a turn yielded: let the model keep what can serve the next turn.

    A yielded turn ends right after its tool batch, before the batch's results
    go back to the model. The Claude relay keeps its native process waiting for
    them, as long as its prompt cache lives, so that the turn that resumes
    delivers them with its message, warm (doc/claude-relay-warm-yield-plan.md).
    A model without ``park`` is retired (``retire_model``), as after a stop;
    for an HTTP model, that cancels nothing at a turn's end.
    """
    park = getattr(model, "park", None)
    if callable(park):
        park()
    else:
        retire_model(model)


def close_model(model) -> None:
    """Optional, idempotent final cleanup; stateless models need no new methods.

    A preempting stop (/exit!!) closes models early, from another thread,
    possibly while a sample or a compaction's request runs. A model with work
    to cancel therefore cancels it as ``retire_model`` does, and fails every
    later sample at once, so that a compaction between its requests stops too.
    """
    close = getattr(model, "close", None)
    if callable(close):
        close()


__all__ = [
    "retire_model", "park_model", "close_model",
    "Model",
    "ModelAuthenticationError",
    "ModelCancelled",
    "ModelConfigurationError",
    "ModelContextWindowError",
    "ModelContinuationExpired",
    "ModelError",
    "ModelRateLimitError",
    "ModelRefusal",
    "ModelResponseError",
    "ModelSample",
    "ModelThinkingBindingError",
    "ModelTimeoutError",
    "ModelTransportError",
    "RetryState",
    "SampleParams",
    "TokenUsage",
]
