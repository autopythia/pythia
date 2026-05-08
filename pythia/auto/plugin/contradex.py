import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from pythia.auto.plugin import AutopythiaPlugin


@dataclass
class Contradex(AutopythiaPlugin):
    contradex_api_base_url: Optional[str] = None
    contradex_model_path: Optional[str] = None
    _contradex_session: Any = None
    _contradex_config: Any = None
    _contradex_lock: Any = None

    @staticmethod
    def __post_init__(self):
        if self._contradex_lock is None:
            self._contradex_lock = asyncio.Lock()

    @staticmethod
    async def contradex(self, step_ctr: int, query: str):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent
        from pythia.term_utils import green

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))
        loop = asyncio.get_running_loop()

        try:
            from contradex.display import TurnEventDisplay

            def emit_to_autopythia(*args, **kwargs) -> None:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                text = sep.join(str(arg) for arg in args)
                chunk = f"{text}{end}"
                if chunk.endswith("\n"):
                    chunk = chunk[:-1]
                if chunk:
                    self._emit_output_threadsafe(loop, chunk)

            def run_turn(config, contradex_session):
                display = TurnEventDisplay(
                    display_level=config.display_level,
                    assistant_mode="message",
                    emit=emit_to_autopythia,
                )
                try:
                    return contradex_session.run_turn(
                        query,
                        event_handler=display.handle,
                    )
                finally:
                    display.close()

            async with self._contradex_lock:
                config, contradex_session = self._load_or_create_contradex_session()
                state = await asyncio.to_thread(run_turn, config, contradex_session)
            usage_summary = (
                " | ".join(
                    [
                        f"output sum={state.output_tokens_sum:,}",
                        f"warm input max={state.cache_hit_input_tokens_max:,} sum={state.cache_hit_input_tokens_sum:,}",
                        f"cold input sum={state.non_cache_hit_input_tokens_sum:,}",
                        f"total input sum={state.input_tokens_sum:,}",
                    ]
                )
                if (
                    state.output_tokens_sum
                    or state.cache_hit_input_tokens_max
                    or state.cache_hit_input_tokens_sum
                    or state.non_cache_hit_input_tokens_sum
                    or state.input_tokens_sum
                )
                else "no detailed token usage metadata"
            )
            compaction_summary = (
                f" | compactions={state.compaction_count:,}"
                if state.compaction_count > 0
                else ""
            )
            self._enqueue_event(
                BasicOutputEvent(
                    text=green(
                        f"contradex: done | context sum={state.total_usage_tokens:,}{compaction_summary} | {usage_summary}",
                        bold=True,
                    ),
                )
            )
        except Exception as exc:
            self._enqueue_event(
                BasicOutputEvent(text=f"contradex failed: {exc.__class__.__name__}: {exc}")
            )
        finally:
            self._enqueue_event(EndControlEvent(step_ctr))

    @staticmethod
    async def quota(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent

        del query

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))

        try:
            from contradex.quota import format_quota_status

            async with self._contradex_lock:
                _, contradex_session = self._load_or_create_contradex_session()
                quota_status = await asyncio.to_thread(contradex_session.get_quota)
            if quota_status is None:
                text = (
                    "contradex quota unavailable: current model client or auth provider "
                    "does not support codex usage queries"
                )
            else:
                text = format_quota_status(quota_status)
            self._enqueue_event(BasicOutputEvent(text=text))
        except Exception as exc:
            self._enqueue_event(
                BasicOutputEvent(text=f"contradex quota failed: {exc.__class__.__name__}: {exc}")
            )
        finally:
            self._enqueue_event(EndControlEvent(step_ctr))
