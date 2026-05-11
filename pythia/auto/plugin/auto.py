import textwrap
from dataclasses import dataclass

from pythia.auto.plugin import AutopythiaPlugin


@dataclass
class Auto(AutopythiaPlugin):
    @staticmethod
    async def auto(self, step_ctr: int, query: str):
        from pythia.auto.kernel import BasicOutputEvent

        default_extension = self._resolve_plugin_extension("/default")
        if default_extension is None:
            raise RuntimeError("default plugin extension is unavailable")
        self._enqueue_event(
            BasicOutputEvent(
                text=(
                    "Entering auto mode with query:\n\n"
                    f"{textwrap.indent(query, '    ')}\n\n"
                    "Control auto mode with /stop and /resume."
                )
            )
        )
        await default_extension(step_ctr, query)

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
