"""/auto and /noauto: auto mode, the start of the CLI's replacement for the
auto frontend (doc/cli-auto-plan.md).

/auto [text] turns auto mode on, if it isn't. It saves AFK's state as it is
then, as if /auto's message hadn't been sent (Afk.save), with the tool-call
approval mode, and gives AFK auto's profile (Afk.use_auto_profile): one rule,
on after 30 minutes without a message from the user, and none that turns AFK
off. Tool calls then run without approval. AFK turns on at once, after
/auto's message, if any: cli._UIState._queue_submission calls sent() once the
message is queued. While tool calls need approval, the CLI asks for /yes
before /auto goes ahead (needs_confirmation). /noauto turns auto mode off,
and puts AFK's state (Afk.load) and the approval mode back as they were at
/auto. Auto mode is idempotent, and a reboot carries it over (freeze,
restore).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from ._cli_afk import Afk, _span
from .loop.icron import IcronError
from .model_catalog import AfkWhen


DEFAULT_AUTO_AFTER = AfkWhen("after", seconds=30 * 60)  # auto's duration, until /autoafk changes it
NOAUTO_USAGE = ("Usage: /noauto (no arguments) turns auto mode off, and puts AFK and tool-call "
                "approval back as they were at /auto.")

Approval = Union[bool, str]  # the CLI's tool-call approval mode: False, True, or "exec"


class Auto:
    """Auto mode, over AFK. ``on`` while it is, since ``since``; ``saved`` is
    AFK's state and the approval mode as they were at /auto, which /noauto
    restores. ``approval`` reads the CLI's tool-call approval mode, and
    ``set_approval`` sets it."""

    def __init__(self, afk: Afk, notice: Callable[[str], None], *,
                 approval: Callable[[], Approval],
                 set_approval: Callable[[Approval], None]) -> None:
        self.afk, self._notice = afk, notice
        self._approval, self._set_approval = approval, set_approval
        self.on = False
        self.since: Optional[float] = None
        self.saved: Optional[dict] = None

    def needs_confirmation(self) -> bool:
        """Whether /auto asks for /yes first: it would turn tool-call approval off."""
        return not self.on and bool(self._approval())

    @staticmethod
    def confirmation() -> str:
        """What the CLI asks, when /auto needs_confirmation."""
        return ("auto: auto mode runs tool calls without approval, until /noauto. /yes enters "
                "it, and /no doesn't.")

    def command(self, line: str) -> bool:
        """/auto alone: auto mode on, and AFK on. (/auto <text> goes as typed
        text does, and comes back to sent().)"""
        self.start(message=False)
        return True

    def sent(self) -> None:
        """/auto <text>'s message is queued: auto mode on, and AFK on after it."""
        self.start(message=True)

    def start(self, *, message: bool) -> None:
        """Auto mode on, if it isn't, then AFK on, or over."""
        if not self.on:
            saved = self.afk.save()  # as if /auto's message hadn't been sent
            saved["approval"] = self._approval()
            self.on, self.since, self.saved = True, self.afk.icron.clock(), saved
            self.afk.use_auto_profile(DEFAULT_AUTO_AFTER)
            self._set_approval(False)
            rules = (f"AFK turns on after {_span(DEFAULT_AUTO_AFTER.seconds)} without a message "
                     "from you")
            if saved["approval"]:
                rules += (", only your messages turn it off, and tool calls run without "
                          "approval")
            else:
                rules += ", and only your messages turn it off"
            self._notice(f"auto: on. Until /noauto, {rules}.")
        if message:
            self.afk.wrote()
        try:
            self.afk.enable()
        except IcronError as exc:
            self._notice(f"afk: not turned on: {exc}")

    def noauto_command(self, line: str) -> bool:
        """/noauto: auto mode off, and AFK and the approval mode back as they
        were at /auto. True once the draft is used."""
        if line.split() != ["/noauto"]:
            self._notice(f"{NOAUTO_USAGE} Draft preserved.")
            return False
        if not self.on:
            self._notice("auto: already off.")
            return True
        saved, since = self.saved, self.since
        self.on, self.since, self.saved = False, None, None
        state = self.afk.load(saved)
        self._set_approval(saved["approval"])
        text = f"auto: off after {_span(self.afk.icron.clock() - since)}. AFK is as it was at "
        text += f"/auto: {state}."
        if saved["approval"]:
            text += (" Tool calls need approval again"
                     + (" (exec)." if saved["approval"] == "exec" else "."))
        self._notice(text)
        return True

    def status(self) -> str:
        """The status line's part: "auto", or nothing."""
        return "auto" if self.on else ""

    def freeze(self) -> Optional[dict]:
        """What a reboot carries over, while auto mode is on (cli._reboot_state)."""
        return {"since": self.since, "saved": self.saved} if self.on else None

    def restore(self, carried: dict, *, rules: bool) -> None:
        """In a restarted CLI: auto mode as a reboot carried it over. ``rules``
        says whether AFK's rules came with it (Afk.restore_autoafk); without
        them, AFK gets auto's profile again."""
        self.on, self.since, self.saved = True, carried["since"], carried["saved"]
        if rules:
            self.afk.in_auto = True
        else:
            self.afk.use_auto_profile(DEFAULT_AUTO_AFTER)
        self._set_approval(False)
        self._notice("auto: still on, carried over the reboot; /noauto ends it.")
