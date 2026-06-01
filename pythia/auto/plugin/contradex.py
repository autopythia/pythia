import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from pythia.auto.plugin import AutopythiaPlugin

REAUTH_TIMEOUT_SECONDS = 120.0


@dataclass
class Contradex(AutopythiaPlugin):
    contradex_api_base_url: Optional[str] = None
    contradex_model_path: Optional[str] = None
    _contradex_session: Any = None
    _contradex_config: Any = None
    _contradex_lock: Any = None

    @staticmethod
    def _post_init(self):
        if self._contradex_lock is None:
            self._contradex_lock = asyncio.Lock()

    @staticmethod
    def _pre_shutdown(self):
        pass

    @staticmethod
    def _post_shutdown(self):
        pass

    @staticmethod
    def _load_contradex_config(self):
        from contradex.integrations.autopythia import AutopythiaContradexConfig

        return AutopythiaContradexConfig(**self._load_contradex_config_from_env())

    @staticmethod
    def _resolve_contradex_session(self, *, isolated_session: bool = False):
        from contradex.integrations.autopythia import AutopythiaContradexSession

        if not isolated_session:
            return self._load_or_create_contradex_session()
        config = Contradex._load_contradex_config(self)
        return config, AutopythiaContradexSession.create(config)

    @staticmethod
    async def default(
        self,
        step_ctr: int,
        query: str,
        *,
        post_user_developer_prompt: str | None = None,
        isolated_session: bool = False,
    ):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent
        from pythia.term_utils import green
        from contradex.protocol import Message

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
                post_user_items = None
                if (
                    post_user_developer_prompt is not None
                    and post_user_developer_prompt.strip()
                ):
                    post_user_items = [
                        Message(role="developer", text=post_user_developer_prompt)
                    ]
                try:
                    return contradex_session.run_turn(
                        query,
                        post_user_items=post_user_items,
                        event_handler=display.handle,
                    )
                finally:
                    display.close()

            async with self._contradex_lock:
                config, contradex_session = Contradex._resolve_contradex_session(
                    self,
                    isolated_session=isolated_session,
                )
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
    async def snapshot(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent

        del query

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))

        try:
            async with self._contradex_lock:
                snapshot_path = Contradex._snapshot(self)
            if snapshot_path is None:
                text = "contradex snapshot unavailable: no active contradex session"
            else:
                text = f"contradex snapshot saved: {snapshot_path}"
            self._enqueue_event(BasicOutputEvent(text=text))
        except Exception as exc:
            self._enqueue_event(
                BasicOutputEvent(
                    text=f"contradex snapshot failed: {exc.__class__.__name__}: {exc}"
                )
            )
        finally:
            self._enqueue_event(EndControlEvent(step_ctr))

    @staticmethod
    def _snapshot(self) -> str | None:
        if self._contradex_session is None or self._session is None:
            return None

        state = getattr(self._contradex_session, "state", None)
        if state is None:
            return None

        to_wire_dict = getattr(state, "to_wire_dict", None)
        if not callable(to_wire_dict):
            return None

        try:
            from pythia.auto.kernel import GLOBAL_SESSION_DIR

            session_dir = os.path.join(GLOBAL_SESSION_DIR, f"{self._session}")
            os.makedirs(session_dir, exist_ok=True)
            snapshot_path = os.path.join(session_dir, "snapshot.json")
            temp_path = os.path.join(
                session_dir,
                f".snapshot.json.tmp-{os.getpid()}",
            )
            payload = to_wire_dict()
            with open(temp_path, "w", encoding="utf-8") as snapshot_file:
                snapshot_file.write(json.dumps(payload, indent=2))
                snapshot_file.write("\n")
            os.replace(temp_path, snapshot_path)
            return snapshot_path
        except Exception:
            return None

    @staticmethod
    async def reauth(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent
        from pythia.term_utils import green

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))

        workspace_hint = query.strip() or None
        server = None
        try:
            from contradex.auth import create_chatgpt_signin_request
            from contradex.auth import start_chatgpt_signin_callback_server

            bootstrap_request = create_chatgpt_signin_request(
                allowed_workspace_id=workspace_hint,
            )
            server = start_chatgpt_signin_callback_server(bootstrap_request)
            signin_request = create_chatgpt_signin_request(
                redirect_uri=(
                    f"http://localhost:{server.actual_port}{server.callback_path}"
                ),
                allowed_workspace_id=workspace_hint,
                force_state=bootstrap_request.state,
                force_code_verifier=bootstrap_request.code_verifier,
            )

            self._enqueue_event(
                BasicOutputEvent(
                    text=green(
                        (
                            "contradex reauth: local callback server listening on "
                            f"http://localhost:{server.actual_port}{server.callback_path}"
                        ),
                        bold=True,
                    )
                )
            )
            self._enqueue_event(
                BasicOutputEvent(
                    text=(
                        "Open this URL in your browser to sign in:\n"
                        f"{signin_request.auth_url}"
                    )
                )
            )
            self._enqueue_event(
                BasicOutputEvent(
                    text=(
                        "Waiting for signin callback "
                        f"(timeout: {int(REAUTH_TIMEOUT_SECONDS)}s)..."
                    )
                )
            )

            completion = await asyncio.to_thread(
                server.wait_for_result,
                REAUTH_TIMEOUT_SECONDS,
            )
            if completion.success:
                self._enqueue_event(
                    BasicOutputEvent(
                        text=green(
                            "contradex reauth: success (authorization code captured)",
                            bold=True,
                        )
                    )
                )
            else:
                self._enqueue_event(
                    BasicOutputEvent(
                        text=(
                            "contradex reauth: "
                            f"{completion.error or 'failed'}"
                        )
                    )
                )
        except Exception as exc:
            self._enqueue_event(
                BasicOutputEvent(text=f"contradex reauth failed: {exc.__class__.__name__}: {exc}")
            )
        finally:
            if server is not None:
                try:
                    server.close()
                except Exception:
                    pass
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

    @staticmethod
    async def model(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import BasicOutputEvent, EndControlEvent, StartControlEvent
        from contradex.model import is_supported_agent_model_path
        from contradex.model import supported_agent_model_paths

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))

        try:
            async with self._contradex_lock:
                config, _contradex_session = self._load_or_create_contradex_session()
                current_model = config.model_path
                requested_model = query.strip()

                if requested_model:
                    if not is_supported_agent_model_path(requested_model):
                        supported_models = ", ".join(supported_agent_model_paths())
                        self._enqueue_event(
                            BasicOutputEvent(
                                text=(
                                    f"warning: unsupported model {requested_model!r}; "
                                    f"supported models: {supported_models}; "
                                    f"current model remains: {current_model}"
                                )
                            )
                        )
                    elif requested_model == current_model:
                        self._enqueue_event(
                            BasicOutputEvent(
                                text=f"model unchanged: {current_model} -> {current_model}"
                            )
                        )
                    else:
                        previous_override = self.contradex_model_path
                        self.contradex_model_path = requested_model
                        try:
                            new_config, _new_session = self._load_or_create_contradex_session()
                        except Exception:
                            self.contradex_model_path = previous_override
                            raise
                        self._enqueue_event(
                            BasicOutputEvent(
                                text=(
                                    f"model switched: {current_model} -> "
                                    f"{new_config.model_path}"
                                )
                            )
                        )
                else:
                    self._enqueue_event(BasicOutputEvent(text=f"current model: {current_model}"))
        except Exception as exc:
            self._enqueue_event(
                BasicOutputEvent(text=f"contradex model failed: {exc.__class__.__name__}: {exc}")
            )
        finally:
            self._enqueue_event(EndControlEvent(step_ctr))
