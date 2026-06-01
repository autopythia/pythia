import textwrap
from dataclasses import dataclass
from pathlib import Path

from pythia.auto.plugin import AutopythiaPlugin

AUTO_PROGRAM_FILENAMES = ("PROGRAM.md", "program.md")
AUTO_POST_USER_DEVELOPER_MESSAGE = (
    "Treat the user's PROGRAM.md contents as the active assignment. "
    "Work incrementally, prefer concrete repository progress over discussion, "
    "and end the turn with a brief summary of what moved forward."
)


@dataclass
class Auto(AutopythiaPlugin):
    @staticmethod
    def _load_program_query(cwd: Path) -> tuple[Path, str]:
        for filename in AUTO_PROGRAM_FILENAMES:
            candidate = cwd / filename
            try:
                return candidate, candidate.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
        joined = " or ".join(AUTO_PROGRAM_FILENAMES)
        raise FileNotFoundError(f"auto mode requires {joined} in {cwd}")

    @staticmethod
    def _build_status_text(program_path: Path, query: str) -> str:
        ignored_query = query.strip()
        parts = [
            f"Entering auto mode with initial user query from {program_path.name}.",
            "",
            "Appended developer message:",
            "",
            textwrap.indent(AUTO_POST_USER_DEVELOPER_MESSAGE, "    "),
        ]
        if ignored_query:
            parts.extend(
                [
                    "",
                    "Ignored extra /auto argument text under the current auto-mode behavior:",
                    "",
                    textwrap.indent(ignored_query, "    "),
                ]
            )
        parts.extend(
            [
                "",
                "Control auto mode with /stop and /resume.",
            ]
        )
        return "\n".join(parts)

    @staticmethod
    async def auto(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import BasicOutputEvent

        default_extension = self._resolve_plugin_extension("/default")
        if default_extension is None:
            raise RuntimeError("default plugin extension is unavailable")

        try:
            program_path, contradex_query = Auto._load_program_query(Path.cwd())
        except OSError as exc:
            self._enqueue_event(
                BasicOutputEvent(
                    text=f"auto mode failed: {exc.__class__.__name__}: {exc}"
                )
            )
            Auto._complete_immediately(self, step_ctr)
            return

        self._enqueue_event(
            BasicOutputEvent(
                text=Auto._build_status_text(program_path, query)
            )
        )
        await default_extension(
            step_ctr,
            contradex_query,
            post_user_developer_prompt=AUTO_POST_USER_DEVELOPER_MESSAGE,
            isolated_session=True,
        )

    a = auto

    @staticmethod
    def _complete_immediately(self, step_ctr: int | None) -> None:
        from pythia.auto.kernel import EndControlEvent, StartControlEvent

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))
        self._enqueue_event(EndControlEvent(step_ctr))

    @staticmethod
    async def stop(self, step_ctr: int, query: str = ""):
        del query
        Auto._complete_immediately(self, step_ctr)

    @staticmethod
    async def resume(self, step_ctr: int, query: str = ""):
        del query
        Auto._complete_immediately(self, step_ctr)
