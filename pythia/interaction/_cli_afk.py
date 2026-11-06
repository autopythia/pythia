"""/afk and /autoafk: the CLI tells main that the user is away (doc/cli-afk-plan.md).

/afk [note] gives icron a repeating message job of the user's. Its reminder
reaches main at once, then every [afk] interval_seconds, as a wake-up: a
running turn takes it at its next interrupt point, and otherwise it starts a
turn. Firings during the turn that took a reminder merge into its run, so a
long turn hears it once. The note ends every reminder. AFK is on while that
job is pending, and /afk while it's on starts it over.

The user's next message for main ends AFK (Afk.message: text, /steer <text>,
or /task <text>), as do [afk] auto_disable and /cron cancel; main can't, as
it can cancel only its own jobs. Ending AFK cancels its job, which also
withdraws a reminder that wasn't delivered yet.

/autoafk [when] sets the trigger that turns AFK on by itself, which [afk]
auto_enable sets at launch (AfkWhen): after a duration without a message from
the user, once per absence; at a time each day; or once, at a date and time,
after which the trigger is gone. [afk] auto_disable ends AFK in the same
three ways: once it has been on for a duration, at a time each day, or once.

A reboot carries AFK and the trigger's state over (freeze, restore_autoafk,
restore); any other restart drops AFK, as it drops every job. Everything here
runs on the CLI's event loop thread: commands, messages, and the terminal
loop's tick.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from datetime import time as day_time
from datetime import timedelta
import time
from typing import Callable, List, Optional

from .loop.icron import Icron, IcronError, Job
from .model_catalog import AfkDefaults
from .model_catalog import AfkWhen
from .model_catalog import DEFAULT_AFK_INTERVAL_SECONDS
from .model_catalog import afk_same_times
from .model_catalog import parse_afk_when


VIA = "/afk"  # the job's via: its wake-ups say "the user's, from /afk"
# What a reminder says after its first sentence, unless [afk] message says otherwise.
GUIDANCE = ("They can't answer until they're back, so don't wait for them: carry on where you "
            "can.")
AUTOAFK_USAGE = ("Usage: /autoafk <when> turns AFK on by itself: /autoafk 20m after 20 minutes "
                 "without a message from you (s, m, h, or d; 1m to 7d), /autoafk 18:00 at 18:00 "
                 "each day, or /autoafk 2026-11-07 18:00 once; /autoafk off removes it, and "
                 "/autoafk alone shows it.")
# /afk takes no on or off: as a note, either would turn AFK on.
_NOT_A_NOTE = {
    "on": "afk: there's no /afk on: /afk alone turns AFK on.",
    "off": ("afk: there's no /afk off: your next message to main ends AFK, and /cron cancel "
            "ends its job."),
}
AUTO_ENABLE, AUTO_DISABLE, AUTOAFK = "[afk] auto_enable", "[afk] auto_disable", "/autoafk"
_CHECK_SECONDS = 1.0  # how often tick() checks the job and the triggers

# The clock that spaces tick()'s checks; tests replace it.
_check_clock = time.monotonic


class Afk:
    """AFK's state and triggers, over icron.

    While AFK is on, ``job_id`` is its job, ``since`` when the user went away
    (epoch seconds), ``note`` the /afk note, and ``until`` when auto_disable
    ends it, if it does. ``enable_when`` is the trigger that turns AFK on by
    itself, from ``enable_source``: [afk] auto_enable or /autoafk, which
    ``enable_changed`` records for a reboot. ``disable_when`` is [afk]
    auto_disable. ``carried`` is what a reboot carries over, once freeze()
    has run.
    """

    def __init__(self, icron: Icron, defaults: AfkDefaults,
                 notice: Callable[[str], None]) -> None:
        self.icron, self._notice = icron, notice
        self.interval = defaults.interval_seconds or DEFAULT_AFK_INTERVAL_SECONDS
        self.guidance = defaults.message or GUIDANCE
        self.job_id: Optional[str] = None
        self.since: Optional[float] = None
        self.note: Optional[str] = None
        self.until: Optional[float] = None
        self.carried: Optional[dict] = None
        now = icron.clock()
        # The user's last message for main, or the CLI's start: a trigger's
        # since. A duration counts from it, or from the /autoafk that set the
        # duration, if later (_count_from), and fires once per absence: AFK
        # turning on disarms it, and the user's next message arms it again.
        # AFK that ends otherwise (auto_disable, or a job that /cron cancel
        # or code cancelled) leaves it disarmed: the absence goes on.
        self._seen = self._count_from = now
        self._armed = True
        self._next_check = 0.0
        self._passed: List[str] = []  # the catalog's dates and times that had passed
        self.enable_when: Optional[AfkWhen] = None
        self.enable_source = AUTO_ENABLE
        self.enable_changed = False
        self._next_enable: Optional[float] = None  # a time of day's or a date's next firing
        enable = self._unless_passed(defaults.auto_enable, AUTO_ENABLE, now)
        self.disable_when = self._unless_passed(defaults.auto_disable, AUTO_DISABLE, now)
        self._set_enable(enable, AUTO_ENABLE, now)

    # The user's side: the commands, and messages.

    def command(self, text: str) -> bool:
        """/afk [note]: AFK on now, with the note at the end of each reminder;
        while it's on, it starts over. True once the draft is used. An
        IcronError, such as the user's job limit, goes to the caller, which
        keeps the draft."""
        words = text.split(maxsplit=1)
        note = words[1].rstrip() if len(words) == 2 else None
        refusal = None if note is None else _NOT_A_NOTE.get(note.lower())
        if refusal is None and note is not None:
            length, limit = len(self._text(self.icron.clock(), note)), self.icron.limits.max_chars
            if length > limit:
                refusal = (f"afk: the note is too long: a reminder has at most {limit} "
                           f"characters, and this one would have {length}.")
        if refusal is not None:
            self._notice(f"{refusal} Draft preserved.")
            return False
        self.enable(note=note)
        return True

    def autoafk_command(self, text: str) -> bool:
        """/autoafk [when|off]: show, set, or remove the trigger that turns AFK
        on by itself. True once the draft is used."""
        words, now = text.split(maxsplit=1), self.icron.clock()
        if len(words) == 1:
            self._notice(self._triggers_text(now))
            return True
        if words[1].strip() == "off":
            if self.enable_when is None:
                self._notice("autoafk: already off; AFK turns on only with /afk.")
            else:
                was = self._enable_text(self.enable_when, now)
                self._set_enable(None, AUTOAFK, now)
                self._notice(f"autoafk: off; it was {was}.")
            return True
        try:
            when = parse_afk_when(words[1])
        except ValueError:
            self._notice(f"{AUTOAFK_USAGE} Draft preserved.")
            return False
        refusal = None
        if when.kind == "once" and when.moment.timestamp() <= now:
            refusal = f"{when} has passed"
        elif afk_same_times(when, self.disable_when):
            refusal = (f"{when} is [afk] auto_disable's time too, so AFK would end and start "
                       "again there")
        if refusal is not None:
            self._notice(f"autoafk: {refusal}. Draft preserved.")
            return False
        self._set_enable(when, AUTOAFK, now)
        self._notice(f"autoafk: AFK turns on {self._enable_text(when, now, detail=True)}. "
                     "/autoafk off removes this.")
        return True

    def message(self) -> None:
        """The user queued a message they typed, for main: text, /steer <text>,
        or /task <text>; wake-ups and the tools' messages never come here. It
        ends AFK, and a trigger's duration counts from it, armed again."""
        on = self.sync()
        self._seen = self._count_from = self.icron.clock()
        self._armed = True
        if on:
            self.disable("message")

    def status(self) -> str:
        """The status line's part: "afk since 14:05", or nothing."""
        return "" if self.since is None else f"afk since {_hm(self.since)}"

    def startup_notices(self) -> List[str]:
        """What the startup notices say of the triggers: the catalog's dates
        and times that had passed, and when AFK turns on and off by itself."""
        now, notices = self.icron.clock(), list(self._passed)
        on = None if self.enable_when is None else self._enable_text(self.enable_when, now)
        off = None if self.disable_when is None else _disable_text(self.disable_when)
        if on is not None and off is not None:
            notices.append(f"afk: AFK turns on {on} ({AUTO_ENABLE}), and off {off} "
                           f"({AUTO_DISABLE}) or at your next message; /autoafk shows or "
                           "changes when it turns on.")
        elif on is not None:
            notices.append(f"afk: AFK turns on {on} ({AUTO_ENABLE}); /autoafk shows or changes "
                           "this.")
        elif off is not None:
            notices.append(f"afk: AFK turns off {off} ({AUTO_DISABLE}), or at your next message.")
        return notices

    # On and off.

    def enable(self, *, note: Optional[str] = None, since: Optional[float] = None,
               why: Optional[str] = None) -> None:
        """Turn AFK on, or start it over: a new job, whose first reminder fires
        at once; the old one, if any, is cancelled once the new one is in. A
        trigger gives ``since``, the user's last message, and ``why``, which
        the notice says. An IcronError leaves AFK as it was."""
        now = self.icron.clock()
        old = self.job_id if self.sync() else None
        job = self._schedule(now if since is None else since, note,
                             after=self.icron.limits.min_delay)
        if old is not None:
            with suppress(IcronError):  # it ended meanwhile
                self.icron.cancel(old, by="user")
        self.until = self._next_disable(now)
        head = ("afk: on again" if old is not None else "afk: on" if why is None
                else f"afk: on {why}")
        with_note = ", with your note" if note is not None else ""
        self._notice(f"{head}{with_note}; main hears now, then every {_span(self.interval)}, "
                     f"that you're away (job {job.job_id}). Your next message ends "
                     f"it{self._until_clause(now)}.")

    def disable(self, how: str) -> None:
        """Turn AFK off: cancel its job, which also withdraws a reminder that
        wasn't delivered yet. ``how`` is "message", the user's, or "time",
        auto_disable's, which leaves a duration disarmed (see __init__)."""
        job_id, since, until = self.job_id, self.since, self.until
        self._clear()
        if how != "message":
            self._armed = False
        suspended = self.icron.main.state == "suspended"
        with suppress(IcronError):  # it ended meanwhile
            self.icron.cancel(job_id, by="user")
        span = _span(self.icron.clock() - since)
        if how == "message":
            self._notice(f"afk: off after {span}: you wrote to main.")
            return
        # Main that yielded until AFK's next reminder can't wake up now.
        waits = ("; main now waits for your message"
                 if suspended and self.icron.main.state != "suspended" else "")
        self._notice(f"afk: off at {_hm(until)} ({AUTO_DISABLE}), after {span}{waits}.")

    def sync(self) -> bool:
        """Whether AFK is on: its job is pending. A job that ended another way,
        by /cron cancel, code, or a failed run, ends AFK, with a notice, and
        leaves a duration disarmed, as the absence goes on."""
        if self.job_id is None:
            return False
        job = self._job()
        if job is not None and job.pending:
            return True
        what = {"cancelled": "was cancelled", "failed": "failed"}.get(
            None if job is None else job.state, "ended")
        self._notice(f"afk: off; its job {self.job_id} {what}.")
        self._clear()
        self._armed = False
        return False

    def tick(self, *, may_enable: bool) -> None:
        """The terminal loop's check, at most once a second: a job that ended,
        auto_disable's time, and the trigger's time, or its duration without a
        message. While AFK may not turn on (``may_enable``), a time that
        passes waits for it."""
        check = _check_clock()
        if check < self._next_check:
            return
        self._next_check = check + _CHECK_SECONDS
        now = self.icron.clock()
        on = self.sync()
        if on and self.until is not None and now >= self.until:
            self.disable("time")
            on = False
        off = self.disable_when
        if off is not None and off.kind == "once" and now >= off.moment.timestamp():
            self.disable_when = None  # spent, whether AFK was on then or not
        when = self.enable_when
        if when is None:
            return
        if when.kind == "after":
            idle = now - self._count_from
            if not on and may_enable and self._armed and idle >= when.seconds:
                self._armed = False
                self._enable_by_trigger(f"after {_span(idle)} without a message")
            return
        if self._next_enable is None or now < self._next_enable or not (on or may_enable):
            return
        at = self._next_enable
        if when.kind == "daily":  # on already, it waits for the next day
            self._next_enable = _next_local_time(when.clock, now)
        else:
            self._set_enable(None, self.enable_source, now)
        if not on:
            self._enable_by_trigger(f"at {_hm(at)}")
        if when.kind == "once":
            self._notice(f"autoafk: off, since {when} has passed.")

    def _enable_by_trigger(self, why: str) -> None:
        why = f"{why} ({self.enable_source})"
        try:
            self.enable(since=self._seen, why=why)
        except IcronError as exc:
            self._notice(f"afk: not turned on {why}: {exc}")

    # Reboots.

    def freeze(self) -> None:
        """Before icron closes: what a reboot carries over, as the reboot
        state's keys (cli._reboot_state): "autoafk", the trigger's state, and,
        while AFK is on, "afk" (since, due, and until, as the first version
        wrote it) and "afk_note"."""
        carried = {"autoafk": self._autoafk_state()}
        job = None if self.job_id is None else self._job()
        if job is not None and job.pending:
            carried["afk"] = {"since": self.since, "due": job.due, "until": self.until}
            if self.note is not None:
                carried["afk_note"] = self.note
        self.carried = carried

    @property
    def carries_job(self) -> bool:
        """Whether freeze() carries AFK, and so its job, over a reboot."""
        return self.carried is not None and "afk" in self.carried

    def restore_autoafk(self, state: dict) -> None:
        """In a restarted CLI, first: the trigger's state as a reboot carried
        it over: the user's last message, the duration's count, and /autoafk's
        trigger, if it replaced the catalog's."""
        now = self.icron.clock()
        self._seen, self._count_from = state["seen"], state["count_from"]
        self._armed = state["armed"]
        if "when" in state:
            when = None if state["when"] is None else parse_afk_when(state["when"])
            if when is not None and when.kind == "once" and when.moment.timestamp() <= now:
                self._notice(f"autoafk: off, since {when} passed during the reboot.")
                when = None
            self._set_enable(when, AUTOAFK, now, fresh=False)

    def restore(self, carried: dict, note: Optional[str] = None) -> None:
        """In a restarted CLI: AFK as a reboot carried it over, from its next
        reminder on (at once if that's past), unless auto_disable's time
        passed meanwhile."""
        now, soonest = self.icron.clock(), self.icron.limits.min_delay
        since, due, until = carried["since"], carried["due"], carried["until"]
        if until is not None and until <= now:
            self._armed = False
            self._notice(f"afk: off at {_hm(until)} ({AUTO_DISABLE}), during the reboot.")
            return
        if due is not None and due >= now + soonest:
            job = self._schedule(since, note, at=datetime.fromtimestamp(due).astimezone())
        else:
            job = self._schedule(since, note, after=soonest)
        self.until = until
        self._notice(f"afk: still on since {_hms(since)}, carried over the reboot (job "
                     f"{job.job_id}); next reminder at {_hms(job.due)}. Your next message ends "
                     f"it{self._until_clause(now)}.")

    # Internals.

    def _set_enable(self, when: Optional[AfkWhen], source: str, now: float, *,
                    fresh: bool = True) -> None:
        """Make ``when`` the trigger that turns AFK on, or none. A duration
        counts from now, armed (``fresh``), unless a reboot carried its count
        over."""
        self.enable_when, self.enable_source = when, source
        self.enable_changed = self.enable_changed or source == AUTOAFK
        self._next_enable = None
        if when is None:
            return
        if when.kind == "after":
            if fresh:
                self._count_from, self._armed = now, True
        elif when.kind == "daily":
            self._next_enable = _next_local_time(when.clock, now)
        else:
            self._next_enable = when.moment.timestamp()

    def _unless_passed(self, when: Optional[AfkWhen], source: str,
                       now: float) -> Optional[AfkWhen]:
        """The catalog's ``when``, unless it's a date and time that has passed."""
        if when is not None and when.kind == "once" and when.moment.timestamp() <= now:
            self._passed.append(f"afk: {source}'s {when} has passed, so it's ignored.")
            return None
        return when

    def _next_disable(self, now: float) -> Optional[float]:
        """When auto_disable ends AFK that turns on now, if it does."""
        when = self.disable_when
        if when is None:
            return None
        if when.kind == "after":
            return now + when.seconds
        if when.kind == "daily":
            return _next_local_time(when.clock, now)
        moment = when.moment.timestamp()
        return moment if moment > now else None

    def _schedule(self, since: float, note: Optional[str], **when) -> Job:
        job = self.icron.schedule(self._text(since, note), every=self.interval, by="user",
                                  via=VIA, **when)
        self.job_id, self.since, self.note = job.job_id, since, note
        return job

    def _text(self, since: float, note: Optional[str]) -> str:
        """A reminder: the facts, the guidance, then the note, if any."""
        text = (f"The user has been away from the keyboard (AFK) since {_iso(since)}; only "
                f"they can end these reminders. {self.guidance}")
        return text if note is None else f"{text}\nTheir note: {note}"

    def _job(self) -> Optional[Job]:
        return next((job for job in self.icron.jobs(by="user") if job.job_id == self.job_id),
                    None)

    def _clear(self) -> None:
        self.job_id = self.since = self.until = self.note = None

    def _until_clause(self, now: float) -> str:
        if self.until is None:
            return ""
        return f", and so will {_when_text(self.until, now)} ({AUTO_DISABLE})"

    def _enable_text(self, when: AfkWhen, now: float, *, detail: bool = False) -> str:
        """When the trigger turns AFK on, as notices say it; with ``detail``,
        also when it does so next."""
        if when.kind == "after":
            text = f"after {_span(when.seconds)} without a message from you"
            return text + (", once per absence" if detail else "")
        if when.kind == "daily":
            text = f"at {when.clock} each day"
            if detail and self._next_enable is not None:
                day = ("today" if datetime.fromtimestamp(self._next_enable).date()
                       == datetime.fromtimestamp(now).date() else "tomorrow")
                text += f", next {day}"
            return text
        text = f"once, at {when}"
        if detail:
            text += f" (in {_span(when.moment.timestamp() - now)})"
        return text

    def _triggers_text(self, now: float) -> str:
        """/autoafk alone: when AFK turns on and off by itself."""
        off = ("" if self.disable_when is None
               else f", or {_disable_text(self.disable_when)} ({AUTO_DISABLE})")
        if self.enable_when is None:
            return (f"autoafk: off; AFK turns on only with /afk. It turns off at your next "
                    f"message{off}. {AUTOAFK_USAGE}")
        on = self._enable_text(self.enable_when, now, detail=True)
        return (f"autoafk: AFK turns on {on} ({self.enable_source}). It turns off at your "
                f"next message{off}. /autoafk <when> replaces the trigger, and /autoafk off "
                "removes it.")

    def _autoafk_state(self) -> dict:
        state = {"seen": self._seen, "count_from": self._count_from, "armed": self._armed}
        if self.enable_changed:
            state["when"] = None if self.enable_when is None else str(self.enable_when)
        return state


def _disable_text(when: AfkWhen) -> str:
    """When auto_disable turns AFK off, as notices say it."""
    if when.kind == "after":
        return f"once it has been on {_span(when.seconds)}"
    if when.kind == "daily":
        return f"at {when.clock} each day"
    return f"at {when}"


def _next_local_time(clock: Optional[str], now: float) -> Optional[float]:
    """The next ``clock`` (HH:MM, local) after ``now``: today's, or tomorrow's
    once today's has passed; None without one."""
    if clock is None:
        return None
    hour, minute = (int(part) for part in clock.split(":"))
    today = datetime.fromtimestamp(now).date()
    when = datetime.combine(today, day_time(hour, minute)).astimezone()
    if when.timestamp() <= now:
        when = datetime.combine(today + timedelta(days=1), day_time(hour, minute)).astimezone()
    return when.timestamp()


def _when_text(seconds: float, now: float) -> str:
    """A time as notices say it: 08:30 today, 08:30 tomorrow, or with its date."""
    when, today = datetime.fromtimestamp(seconds), datetime.fromtimestamp(now).date()
    if when.date() == today:
        return when.strftime("%H:%M")
    if when.date() == today + timedelta(days=1):
        return when.strftime("%H:%M tomorrow")
    return when.strftime("%Y-%m-%d %H:%M")


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
