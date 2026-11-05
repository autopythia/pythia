"""/afk: the CLI tells main that the user is away (doc/cli-afk-plan.md).

/afk (or /afk on) gives icron a repeating message job of the user's. Its
reminder reaches main at once, then every [afk] interval_seconds, as a
wake-up: a running turn takes it at its next interrupt point, and otherwise
it starts a turn. Firings during the turn that took a reminder merge into its
run, so a long turn hears it once. AFK is on while that job is pending:
/afk off and /cron cancel end it, and main can't, as it can cancel only its
own jobs.

The model catalog's [afk] section (AfkDefaults) sets the automatic rules:
auto_enable_seconds turns AFK on after that long without a key press, once
per absence; auto_disable_on_message (on by default) turns it off when the
user sends main a message they typed; auto_disable_at turns it off at the
next such local time. A reboot carries AFK over (freeze, restore); any other
restart drops it, as it drops every job.

Everything here runs on the CLI's event loop thread: key presses, commands,
and the terminal loop's tick.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from datetime import time as day_time
from datetime import timedelta
import time
from typing import Callable, Optional

from .loop.icron import Icron, IcronError, Job
from .model_catalog import AfkDefaults
from .model_catalog import DEFAULT_AFK_AUTO_DISABLE_ON_MESSAGE
from .model_catalog import DEFAULT_AFK_INTERVAL_SECONDS


VIA = "/afk"  # the job's via: its wake-ups say "the user's, from /afk"
# What a reminder says after its first sentence, unless [afk] message says otherwise.
GUIDANCE = ("They can't answer until they're back, so don't wait for them: carry on where you "
            "can.")
_CHECK_SECONDS = 1.0  # how often tick() checks the job and the automatic rules

# The clock of the time without a key press (auto_enable_seconds); tests replace it.
_idle_clock = time.monotonic


class Afk:
    """AFK's state and rules, over icron. ``job_id`` is its job while it's on,
    ``since`` when the user left (epoch seconds), and ``until`` when
    auto_disable_at ends it, if set. ``carried`` is what a reboot carries
    over, once freeze() has run."""

    def __init__(self, icron: Icron, defaults: AfkDefaults,
                 notice: Callable[[str], None]) -> None:
        self.icron, self._notice = icron, notice
        self.interval = defaults.interval_seconds or DEFAULT_AFK_INTERVAL_SECONDS
        self.auto_enable = defaults.auto_enable_seconds
        self.on_message = (DEFAULT_AFK_AUTO_DISABLE_ON_MESSAGE
                           if defaults.auto_disable_on_message is None
                           else defaults.auto_disable_on_message)
        self.disable_at = defaults.auto_disable_at
        self.guidance = defaults.message or GUIDANCE
        self.job_id: Optional[str] = None
        self.since: Optional[float] = None
        self.until: Optional[float] = None
        self.carried: Optional[dict] = None
        self._still_noticed = False  # without auto_disable_on_message: said once per absence
        # Auto-enable counts from the last key press, and fires once per
        # absence: AFK turning on disarms it, and the user's coming back arms
        # it again: a key press while AFK is off, or the user's turning AFK
        # off (/afk off, /cron cancel, a message). AFK's own ends, at
        # auto_disable_at's time or by code, don't: the absence goes on.
        self._last_key = _idle_clock()
        self._armed = True
        self._next_check = 0.0

    # The user's side: the command, key presses, and messages.

    def command(self, text: str) -> bool:
        """/afk, /afk on, /afk off; True once the draft is used. An IcronError,
        such as the user's job limit, goes to the caller, which keeps the
        draft."""
        words = text.split()
        if words in (["/afk"], ["/afk", "on"]):
            if self.sync(by_user=True):
                self._notice(self._state_text())
            else:
                self.enable()
            return True
        if words == ["/afk", "off"]:
            if self.sync(by_user=True):
                self.disable()
            else:
                self._notice("afk: already off.")
            return True
        self._notice(f"{self.usage()} Draft preserved.")
        return False

    def usage(self) -> str:
        return (f"Usage: /afk or /afk on tells main, now and every {_span(self.interval)}, "
                "that you're away, until /afk off; while it's on, /afk shows it.")

    def key(self) -> None:
        """A key press: the user is here, and auto-enable counts from now;
        with AFK off, it may fire again (see __init__)."""
        self._last_key = _idle_clock()
        if self.job_id is None:
            self._armed = True

    def message(self) -> None:
        """The user queued a message they typed, for main: text, /steer <text>,
        or /task <text>. Wake-ups and the tools' messages never come here.
        With auto_disable_on_message, AFK ends; otherwise a notice says, once
        per absence, that it's still on."""
        if not self.sync(by_user=True):
            return
        if self.on_message:
            self.disable("message")
        elif not self._still_noticed:
            self._still_noticed = True
            self._notice("afk: still on; /afk off ends it.")

    def status(self) -> str:
        """The status line's part: "afk since 14:05", or nothing."""
        return "" if self.since is None else f"afk since {_hm(self.since)}"

    def startup_notice(self) -> Optional[str]:
        """What the startup notices say of AFK's automatic rules, if any is on."""
        if self.auto_enable is None:
            return None
        return (f"afk: turns on after {_span(self.auto_enable)} without a key press "
                "([afk] auto_enable_seconds).")

    # On and off.

    def enable(self, *, since: Optional[float] = None, idle: Optional[float] = None) -> None:
        """Turn AFK on: its job's first reminder fires at once. Auto-enable
        gives ``since``, the last key press, and ``idle``, the time since."""
        now = self.icron.clock()
        job = self._schedule(now if since is None else since,
                             after=self.icron.limits.min_delay)
        self.until = self._next_disable(now)
        self._armed = False
        head = ("afk: on" if idle is None else
                f"afk: on after {_span(idle)} without input ([afk] auto_enable_seconds)")
        self._notice(f"{head}; main hears now, then every {_span(self.interval)}, that you're "
                     f"away (job {job.job_id}). /afk off ends it{self._until_clause(now)}.")

    def disable(self, how: Optional[str] = None) -> None:
        """Turn AFK off: cancel its job. ``how`` is the automatic rule that
        does it: "message" or "time". The user's own ends rearm auto-enable."""
        job_id, since, until = self.job_id, self.since, self.until
        self._clear()
        self._armed = how != "time"
        with suppress(IcronError):  # it ended meanwhile
            self.icron.cancel(job_id, by="user")
        span = _span(self.icron.clock() - since)
        if how == "message":
            self._notice(f"afk: off after {span}: you wrote to main "
                         "([afk] auto_disable_on_message).")
        elif how == "time":
            self._notice(f"afk: off at {_hm(until)} ([afk] auto_disable_at), after {span}.")
        else:
            self._notice(f"afk: off after {span}.")

    def sync(self, *, by_user: bool = False) -> bool:
        """Whether AFK is on: its job is pending. A job that ended any other
        way, by /cron cancel or a failed run, ends AFK, with a notice;
        ``by_user`` says that the user is here, typing a command or a
        message, which rearms auto-enable."""
        if self.job_id is None:
            return False
        job = self._job()
        if job is not None and job.pending:
            return True
        what = {"cancelled": "was cancelled", "failed": "failed"}.get(
            None if job is None else job.state, "ended")
        self._notice(f"afk: off; its job {self.job_id} {what}.")
        self._clear()
        self._armed = by_user
        return False

    def tick(self, *, may_enable: bool) -> None:
        """The terminal loop's check, at most once a second: a job that ended,
        auto_disable_at's time, and, if ``may_enable``, auto-enable."""
        idle_now = _idle_clock()
        if idle_now < self._next_check:
            return
        self._next_check = idle_now + _CHECK_SECONDS
        if self.job_id is not None:
            if self.sync() and self.until is not None and self.icron.clock() >= self.until:
                self.disable("time")
            return
        idle = idle_now - self._last_key
        if (may_enable and self._armed and self.auto_enable is not None
                and idle >= self.auto_enable):
            self._armed = False
            try:
                self.enable(since=self.icron.clock() - idle, idle=idle)
            except IcronError as exc:
                self._notice(f"afk: not turned on: {exc}")

    # Reboots.

    def freeze(self) -> None:
        """Before icron closes: what a reboot carries over, if AFK is on."""
        self.carried = None
        job = None if self.job_id is None else self._job()
        if job is not None and job.pending:
            self.carried = {"since": self.since, "due": job.due, "until": self.until}

    def restore(self, carried: dict) -> None:
        """In a restarted CLI: AFK as a reboot carried it over, from its next
        reminder on (at once if that's past), unless auto_disable_at's time
        passed meanwhile."""
        now, soonest = self.icron.clock(), self.icron.limits.min_delay
        since, due, until = carried["since"], carried["due"], carried["until"]
        if until is not None and until <= now:
            self._notice(f"afk: off at {_hm(until)} ([afk] auto_disable_at), during the reboot.")
            return
        if due is not None and due >= now + soonest:
            job = self._schedule(since, at=datetime.fromtimestamp(due).astimezone())
        else:
            job = self._schedule(since, after=soonest)
        self.until = until
        self._armed = False
        self._notice(f"afk: still on since {_hms(since)}, carried over the reboot (job "
                     f"{job.job_id}); next reminder at {_hms(job.due)}. /afk off ends "
                     f"it{self._until_clause(now)}.")

    # Internals.

    def _schedule(self, since: float, **when) -> Job:
        job = self.icron.schedule(self._text(since), every=self.interval, by="user", via=VIA,
                                  **when)
        self.job_id, self.since, self._still_noticed = job.job_id, since, False
        return job

    def _text(self, since: float) -> str:
        """A reminder: the facts, then the guidance."""
        return (f"The user has been away from the keyboard (AFK) since {_iso(since)}; only "
                f"they can end these reminders. {self.guidance}")

    def _job(self) -> Optional[Job]:
        return next((job for job in self.icron.jobs(by="user") if job.job_id == self.job_id),
                    None)

    def _clear(self) -> None:
        self.job_id = self.since = self.until = None

    def _next_disable(self, now: float) -> Optional[float]:
        """auto_disable_at's next time, local: today's, or tomorrow's once
        today's has passed."""
        if self.disable_at is None:
            return None
        hour, minute = (int(part) for part in self.disable_at.split(":"))
        today = datetime.fromtimestamp(now).date()
        when = datetime.combine(today, day_time(hour, minute)).astimezone()
        if when.timestamp() <= now:
            when = datetime.combine(today + timedelta(days=1), day_time(hour, minute)).astimezone()
        return when.timestamp()

    def _until_clause(self, now: float) -> str:
        if self.until is None:
            return ""
        tomorrow = (" tomorrow" if datetime.fromtimestamp(self.until).date()
                    != datetime.fromtimestamp(now).date() else "")
        return f", and so will {_hm(self.until)}{tomorrow} ([afk] auto_disable_at)"

    def _state_text(self) -> str:
        job, now = self._job(), self.icron.clock()
        due = "" if job is None or job.due is None else f"; next reminder at {_hms(job.due)}"
        return (f"afk: on since {_hms(self.since)} (job {self.job_id}){due}. /afk off ends "
                f"it{self._until_clause(now)}.")


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def _hms(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%H:%M:%S")


def _hm(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%H:%M")


def _span(seconds: float) -> str:
    """A duration such as 10m, 1h 30m, or 23h 59m 10s, as the CLI writes it."""
    seconds = max(0, int(round(seconds)))
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds %= size
    return " ".join(parts[:3]) or "0s"
