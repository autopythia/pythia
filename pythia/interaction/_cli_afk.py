"""AFK: the CLI tells main that the user is away (doc/cli-afk-plan.md and
doc/cli-auto-plan.md).

/afk [text] gives icron a repeating message job of the user's. Its reminder
reaches main at once, then every [afk] interval_seconds, as a wake-up: a
running turn takes it at its next interrupt point, and otherwise it starts a
turn. Firings during the turn that took a reminder merge into its run, so a
long turn hears it once. /afk's text, if any, ends every reminder. AFK is on
while that job is pending, and /afk while it's on starts it over.

The user's next message for main ends AFK (Afk.message: text, /steer <text>,
or /task <text>), as do /cron cancel and the rules that turn it off; main
can't, as it can cancel only its own jobs. Ending AFK cancels its job, which
also withdraws a reminder that wasn't delivered yet.

Rules (AfkWhen) turn AFK on and off by themselves, at most one of each class
for each kind (ON, OFF), and any of them may fire. A duration turns AFK on
after that long without a message from the user, once per absence, and off
once AFK has been on that long; a time of day fires each day; and a date and
time fires once, and is gone. [afk] auto_enable and auto_disable set the
rules at launch; /autoafk and /noautoafk change those that turn AFK on, and
/autonoafk and /noautonoafk those that turn it off. Auto mode (_cli_auto)
gives AFK a profile of its own (save, use_auto_profile, load), in which the
rules that turn AFK off can't change, and its duration can't go.

A reboot carries AFK and its rules over (freeze, restore_autoafk, restore);
any other restart drops AFK, as it drops every job. Everything here runs on
the CLI's event loop thread: commands, messages, and the terminal loop's
tick.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from datetime import time as day_time
from datetime import timedelta
import time
from typing import Callable, Dict, List, Optional, Tuple

from .loop.icron import Icron, IcronError, Job
from .model_catalog import AfkDefaults
from .model_catalog import AfkWhen
from .model_catalog import DEFAULT_AFK_INTERVAL_SECONDS
from .model_catalog import parse_afk_when


VIA = "/afk"  # the job's via: its wake-ups say "the user's, from /afk"
# What a reminder says after its first sentence, unless [afk] message says otherwise.
GUIDANCE = ("They can't answer until they're back, so don't wait for them: carry on where you "
            "can.")
ON, OFF = "on", "off"  # the two kinds of rules: they turn AFK on, or off
# Where a rule came from, as the notices name it.
AUTO_ENABLE, AUTO_DISABLE = "[afk] auto_enable", "[afk] auto_disable"
AUTOAFK, AUTONOAFK, AUTO = "/autoafk", "/autonoafk", "/auto"
SOURCES = (AUTO_ENABLE, AUTO_DISABLE, AUTOAFK, AUTONOAFK, AUTO)
RULE_COMMANDS = ("/autoafk", "/noautoafk", "/autonoafk", "/noautonoafk")
_CLASSES = ("after", "daily", "once")  # AfkWhen.kind, in the order the notices list rules
_LABEL = {ON: "autoafk", OFF: "autonoafk"}  # the commands' names, and their notices'
AUTOAFK_USAGE = (
    "Usage: /autoafk <rule> sets a rule that turns AFK on by itself, one of each class: "
    "/autoafk 20m after 20 minutes without a message from you (s, m, h, or d; 1m to 7d), "
    "/autoafk 18:00 at 18:00 each day, or /autoafk 2026-11-07 18:00 once. /noautoafk <rule> "
    "removes one, /noautoafk alone all, and /autoafk alone shows them.")
AUTONOAFK_USAGE = (
    "Usage: /autonoafk <rule> sets a rule that turns AFK off by itself, one of each class: "
    "/autonoafk 8h once AFK has been on 8 hours, /autonoafk 08:30 at 08:30 each day, or "
    "/autonoafk 2026-11-09 08:00 once. /noautonoafk <rule> removes one, /noautonoafk alone "
    "all, and /autonoafk alone shows them.")
_USAGE = {ON: AUTOAFK_USAGE, OFF: AUTONOAFK_USAGE}
# /afk takes no on or off: as its text, either would turn AFK on.
_NOT_TEXT = {
    "on": "afk: there's no /afk on: /afk alone turns AFK on.",
    "off": ("afk: there's no /afk off: your next message to main ends AFK, and /cron cancel "
            "ends its job."),
}
_CHECK_SECONDS = 1.0  # how often tick() checks the job and the rules

# The clock that spaces tick()'s checks; tests replace it.
_check_clock = time.monotonic


@dataclass
class _Rule:
    """One of AFK's rules: when it fires, where it came from (SOURCES), and
    when it was set. ``next`` is when a rule that turns AFK on at a time of
    day, or at a date and time, fires next."""

    when: AfkWhen
    source: str
    set_at: float
    next: Optional[float] = None


class Afk:
    """AFK's state and rules, over icron.

    While AFK is on, ``job_id`` is its job, ``since`` when the user went away
    (epoch seconds), ``text`` the /afk text, ``on_at`` when it turned on, and
    ``until`` when a rule ends it, if one does, from ``until_source``.
    ``rules`` holds each kind's rules (ON, OFF) by class, and ``changed`` says
    which kinds commands or auto mode changed, for a reboot. ``in_auto`` is
    set in auto mode (_cli_auto). ``carried`` is what a reboot carries over,
    once freeze() has run.
    """

    def __init__(self, icron: Icron, defaults: AfkDefaults,
                 notice: Callable[[str], None]) -> None:
        self.icron, self._notice = icron, notice
        self.interval = defaults.interval_seconds or DEFAULT_AFK_INTERVAL_SECONDS
        self.guidance = defaults.message or GUIDANCE
        self.job_id: Optional[str] = None
        self.since: Optional[float] = None
        self.text: Optional[str] = None
        self.on_at: Optional[float] = None
        self.until: Optional[float] = None
        self.until_source: Optional[str] = None
        self.carried: Optional[dict] = None
        self.in_auto = False
        now = icron.clock()
        # The user's last message for main, or the CLI's start: since, when a
        # rule turns AFK on. A duration counts from it, or from the command
        # that set the duration, if later (_count_from), and turns AFK on once
        # per absence: AFK turning on disarms it, and the user's next message
        # arms it again. AFK that ends otherwise (by a rule, or because /cron
        # cancel or code cancelled its job) leaves it disarmed: the absence
        # goes on.
        self._seen = self._count_from = now
        self._armed = True
        self._next_check = 0.0
        self._passed: List[str] = []  # the catalog's dates and times that had passed
        self.rules: Dict[str, Dict[str, _Rule]] = {ON: {}, OFF: {}}
        self.changed = {ON: False, OFF: False}
        # The catalog's rules, which /noauto puts back where no command changed them.
        self._catalog = {ON: [[str(when), AUTO_ENABLE] for when in defaults.auto_enable or ()],
                         OFF: [[str(when), AUTO_DISABLE] for when in defaults.auto_disable or ()]}
        for kind, rules, source in ((ON, defaults.auto_enable, AUTO_ENABLE),
                                    (OFF, defaults.auto_disable, AUTO_DISABLE)):
            for when in rules or ():
                if _passed(when, now):
                    self._passed.append(f"afk: {source}'s {when} has passed, so it's ignored.")
                else:
                    self._set_rule(kind, when, source, now)

    # The user's side: the commands, and messages.

    def command(self, line: str) -> bool:
        """/afk [text]: AFK on now, with the text at the end of each reminder;
        while it's on, it starts over. True once the draft is used. An
        IcronError, such as the user's job limit, goes to the caller, which
        keeps the draft."""
        words = line.split(maxsplit=1)
        text = words[1].rstrip() if len(words) == 2 else None
        refusal = None if text is None else _NOT_TEXT.get(text.lower())
        if refusal is None and text is not None:
            length = len(self._reminder(self.icron.clock(), text))
            limit = self.icron.limits.max_chars
            if length > limit:
                refusal = (f"afk: the text is too long: a reminder has at most {limit} "
                           f"characters, and this one would have {length}.")
        if refusal is not None:
            return self._refuse(refusal)
        self.enable(text=text)
        return True

    def rules_command(self, head: str, line: str) -> bool:
        """/autoafk and /autonoafk show the rules, or set one of their kind,
        replacing its class's; /noautoafk and /noautonoafk remove one, or, with
        no rule, all of their kind. True once the draft is used. In auto mode,
        the rules that turn AFK off stay as they are, and so does the duration
        that turns it on, but for /autoafk changing it."""
        kind = OFF if head in ("/autonoafk", "/noautonoafk") else ON
        label, remove = _LABEL[kind], head.startswith("/no")
        words = line.split(maxsplit=1)
        rule = words[1].strip() if len(words) == 2 else ""
        now = self.icron.clock()
        if not rule and not remove:
            self._notice(self._rules_text(kind, now))
            return True
        if kind == OFF and self.in_auto:
            return self._refuse(f"{label}: in auto mode, only your messages turn AFK off; "
                                "/noauto ends auto mode.")
        if not rule:
            return self._remove_all(kind, now)
        if kind == ON and not remove and rule.lower() == "off":
            return self._refuse("autoafk: there's no /autoafk off: /noautoafk <rule> removes a "
                                "rule, and /noautoafk alone removes them all.")
        try:
            when = parse_afk_when(rule)
        except ValueError:
            return self._refuse(_USAGE[kind])
        if remove:
            return self._remove_rule(kind, when, now)
        return self._add_rule(kind, when, now)

    def message(self) -> None:
        """The user queued a message they typed, for main: text, /steer <text>,
        or /task <text>; wake-ups and the tools' messages never come here. It
        ends AFK, and a duration counts from it, armed again."""
        on = self.sync()
        self.wrote()
        if on:
            self.disable("message")

    def wrote(self) -> None:
        """The user wrote to main: a duration counts from now, armed. Without
        ending AFK, for /auto's message (_cli_auto)."""
        self._seen = self._count_from = self.icron.clock()
        self._armed = True

    def status(self) -> str:
        """The status line's part: "afk since 14:05", or nothing."""
        return "" if self.since is None else f"afk since {_hm(self.since)}"

    def startup_notices(self) -> List[str]:
        """What the startup notices say of the rules: the catalog's dates and
        times that had passed, and when AFK turns on and off by itself."""
        now, notices = self.icron.clock(), list(self._passed)
        on, off = self._listing(ON, now), self._listing(OFF, now)
        if on and off:
            notices.append(f"afk: AFK turns on {on}, and off {off} or at your next message; "
                           "/autoafk and /autonoafk show or change this.")
        elif on:
            notices.append(f"afk: AFK turns on {on}; /autoafk shows or changes this.")
        elif off:
            notices.append(f"afk: AFK turns off {off}, or at your next message; /autonoafk "
                           "shows or changes this.")
        return notices

    # On and off.

    def enable(self, *, text: Optional[str] = None, since: Optional[float] = None,
               why: Optional[str] = None) -> None:
        """Turn AFK on, or start it over: a new job, whose first reminder fires
        at once; the old one, if any, is cancelled once the new one is in. A
        rule gives ``since``, the user's last message, and ``why``, which the
        notice says. An IcronError leaves AFK as it was."""
        now = self.icron.clock()
        old = self.job_id if self.sync() else None
        job = self._schedule(now if since is None else since, text,
                             after=self.icron.limits.min_delay)
        if old is not None:
            with suppress(IcronError):  # it ended meanwhile
                self.icron.cancel(old, by="user")
        self.on_at = now
        self._refresh_until(now)
        head = ("afk: on again" if old is not None else "afk: on" if why is None
                else f"afk: on {why}")
        with_text = ", with your text" if text is not None else ""
        self._notice(f"{head}{with_text}; main hears now, then every {_span(self.interval)}, "
                     f"that you're away (job {job.job_id}). Your next message ends "
                     f"it{self._until_clause(now)}.")

    def disable(self, how: str) -> None:
        """Turn AFK off: cancel its job, which also withdraws a reminder that
        wasn't delivered yet. ``how`` is "message", the user's, or "time", a
        rule's, which leaves a duration disarmed (see __init__)."""
        job_id, since, until, source = self.job_id, self.since, self.until, self.until_source
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
        self._notice(f"afk: off at {_hm(until)} ({source}), after {span}{waits}.")

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
        the end that a rule set, and the rules that turn AFK on: a time, or a
        duration without a message. While AFK may not turn on
        (``may_enable``), a time that passes waits for it."""
        check = _check_clock()
        if check < self._next_check:
            return
        self._next_check = check + _CHECK_SECONDS
        now = self.icron.clock()
        on = self.sync()
        if on and self.until is not None and now >= self.until:
            self.disable("time")
            on = False
        spent = self.rules[OFF].get("once")
        if spent is not None and now >= spent.when.moment.timestamp():
            del self.rules[OFF]["once"]  # whether AFK was on then or not
        fired: Optional[Tuple[_Rule, str]] = None
        gone = []
        for cls in ("once", "daily", "after"):
            rule = self.rules[ON].get(cls)
            if rule is None:
                continue
            if cls == "after":
                idle = now - self._count_from
                if (not on and fired is None and may_enable and self._armed
                        and idle >= rule.when.seconds):
                    self._armed = False
                    fired = (rule, f"after {_span(idle)} without a message")
                continue
            if now < rule.next or not (on or may_enable):
                continue
            at = rule.next
            if cls == "daily":  # on already, it waits for the next day
                rule.next = _next_local_time(rule.when.clock, now)
            else:
                del self.rules[ON]["once"]
                gone.append(f"autoafk: {rule.when} has passed, so that rule is gone.")
            if not on and fired is None:
                fired = (rule, f"at {_hm(at)}")
        if fired is not None:
            self._enable_by_rule(*fired)
        for text in gone:
            self._notice(text)

    def _enable_by_rule(self, rule: _Rule, why: str) -> None:
        why = f"{why} ({rule.source})"
        try:
            self.enable(since=self._seen, why=why)
        except IcronError as exc:
            self._notice(f"afk: not turned on {why}: {exc}")

    # Auto mode (_cli_auto).

    def save(self) -> dict:
        """AFK's state, which auto mode keeps until /noauto: the rules of both
        kinds, with their sources, and AFK, if it's on. It's JSON, so that a
        reboot can carry it over."""
        on = self.sync()
        return {
            "enable": self._rules_state(ON), "disable": self._rules_state(OFF),
            "enable_changed": self.changed[ON], "disable_changed": self.changed[OFF],
            "afk": ({"since": self.since, "text": self.text, "on_at": self.on_at,
                     "until": self.until} if on else None),
        }

    def use_auto_profile(self, interval: AfkWhen) -> None:
        """Auto mode's rules: ``interval``, a duration that turns AFK on, and
        none that turns it off."""
        now = self.icron.clock()
        self.in_auto = True
        for kind in (ON, OFF):
            self.rules[kind].clear()
            self.changed[kind] = True
        self._set_rule(ON, interval, AUTO, now)
        self._refresh_until(now)

    def load(self, saved: dict) -> str:
        """At /noauto: AFK's state as save() gave it, with its rules' timing
        started afresh: a duration counts from now. A kind of rules that no
        command had changed is the catalog's, as this process read it, which
        a reboot may have read anew. What AFK is now, as auto's notice says
        it."""
        now = self.icron.clock()
        self.in_auto = False
        for kind, key in ((ON, "enable"), (OFF, "disable")):
            self.rules[kind].clear()
            changed = saved[f"{key}_changed"]
            for text, source in saved[key] if changed else self._catalog[kind]:
                when = parse_afk_when(text)
                if not _passed(when, now):
                    self._set_rule(kind, when, source, now)
            self.changed[kind] = changed
        self._count_from, self._armed = now, True
        afk = saved["afk"]
        old = self.job_id if self.sync() else None
        if old is not None:
            self._clear()
            with suppress(IcronError):  # it ended meanwhile
                self.icron.cancel(old, by="user")
        if afk is None:
            state = "off"
        elif afk["until"] is not None and afk["until"] <= now:
            state = f"off, since its end at {_hm(afk['until'])} passed in auto mode"
        else:
            try:
                job = self._schedule(afk["since"], afk["text"],
                                     after=self.icron.limits.min_delay)
            except IcronError as exc:
                state = f"off, since its job couldn't start again: {exc}"
            else:
                self.on_at = afk["on_at"]
                self._refresh_until(now)
                state = f"on since {_hms(afk['since'])} (job {job.job_id})"
        rules = self._listing(ON, now)
        return state + (f", and it turns on {rules}" if rules else "")

    # Reboots.

    def freeze(self) -> None:
        """Before icron closes: what a reboot carries over, as the reboot
        state's keys (cli._reboot_state): "autoafk", the rules' state, and,
        while AFK is on, "afk" (since, due, and until, as the first versions
        wrote it) and "afk_text"."""
        carried = {"autoafk": self._autoafk_state()}
        job = None if self.job_id is None else self._job()
        if job is not None and job.pending:
            carried["afk"] = {"since": self.since, "due": job.due, "until": self.until}
            if self.text is not None:
                carried["afk_text"] = self.text
        self.carried = carried

    @property
    def carries_job(self) -> bool:
        """Whether freeze() carries AFK, and so its job, over a reboot."""
        return self.carried is not None and "afk" in self.carried

    def restore_autoafk(self, state: dict) -> None:
        """In a restarted CLI, first: the rules' state as a reboot carried it
        over: the user's last message, the duration's count, when AFK turned
        on, and the rules that commands or auto mode changed. ``when``, the
        one rule that turned AFK on in 33b31af's shape, reads too."""
        now = self.icron.clock()
        self._seen, self._count_from = state["seen"], state["count_from"]
        self._armed = state["armed"]
        self.on_at = state.get("on_at")
        lists = {ON: state.get("enable"), OFF: state.get("disable")}
        if "when" in state:
            lists[ON] = [] if state["when"] is None else [[state["when"], AUTOAFK]]
        for kind, rules in lists.items():
            if rules is None:
                continue
            self.rules[kind].clear()
            self.changed[kind] = True
            for text, source in rules:
                when = parse_afk_when(text)
                if _passed(when, now):
                    self._notice(f"{_LABEL[kind]}: {when} passed during the reboot, so that "
                                 "rule is gone.")
                else:
                    self._set_rule(kind, when, source, now, fresh=False)

    def restore(self, carried: dict, text: Optional[str] = None) -> None:
        """In a restarted CLI: AFK as a reboot carried it over, from its next
        reminder on (at once if that's past), unless its end passed
        meanwhile."""
        now, soonest = self.icron.clock(), self.icron.limits.min_delay
        since, due, until = carried["since"], carried["due"], carried["until"]
        on_at = now if self.on_at is None else self.on_at
        source = next((src for end, src in self._ends(on_at) if abs(end - (until or 0)) < 1),
                      AUTO_DISABLE)
        if until is not None and until <= now:
            self._armed = False
            self._notice(f"afk: off at {_hm(until)} ({source}), during the reboot.")
            return
        if due is not None and due >= now + soonest:
            job = self._schedule(since, text, at=datetime.fromtimestamp(due).astimezone())
        else:
            job = self._schedule(since, text, after=soonest)
        self.on_at, self.until, self.until_source = on_at, until, source
        self._notice(f"afk: still on since {_hms(since)}, carried over the reboot (job "
                     f"{job.job_id}); next reminder at {_hms(job.due)}. Your next message ends "
                     f"it{self._until_clause(now)}.")

    # Rules.

    def _add_rule(self, kind: str, when: AfkWhen, now: float) -> bool:
        label = _LABEL[kind]
        if _passed(when, now):
            return self._refuse(f"{label}: {when} has passed.")
        other = OFF if kind == ON else ON
        clash = self.rules[other].get(when.kind)
        if when.kind != "after" and clash is not None and clash.when == when:
            return self._refuse(f"{label}: {when} turns AFK {other} too ({clash.source}), so AFK "
                                "would end and start again there.")
        old = self.rules[kind].get(when.kind)
        also = "also " if any(cls != when.kind for cls in self.rules[kind]) else ""
        self._set_rule(kind, when, AUTOAFK if kind == ON else AUTONOAFK, now)
        self.changed[kind] = True
        if kind == OFF:
            self._refresh_until(now)
        instead = (f", instead of {self._describe(kind, old.when, now)}"
                   if old is not None and old.when != when else "")
        if self.in_auto and kind == ON and when.kind == "after":
            tail = "In auto mode it stays; /autoafk <duration> changes it."
        else:
            tail = f"/no{label} {when} removes this."
        self._notice(f"{label}: AFK {also}turns {kind} "
                     f"{self._describe(kind, when, now, detail=True)}{instead}. {tail}")
        return True

    def _remove_rule(self, kind: str, when: AfkWhen, now: float) -> bool:
        label = _LABEL[kind]
        rule = self.rules[kind].get(when.kind)
        if rule is None or rule.when != when:
            return self._refuse(f"{label}: there's no rule {when}; "
                                f"{self._rules_clause(kind, now)}.")
        if self.in_auto and kind == ON and when.kind == "after":
            return self._refuse("autoafk: in auto mode, AFK's duration stays; /autoafk <duration> "
                                "changes it.")
        del self.rules[kind][when.kind]
        self.changed[kind] = True
        if kind == OFF:
            self._refresh_until(now)
        self._notice(f"{label}: removed {self._describe(kind, when, now)}; "
                     f"{self._rules_clause(kind, now, still=True)}.")
        return True

    def _remove_all(self, kind: str, now: float) -> bool:
        label = _LABEL[kind]
        if self.in_auto and kind == ON:
            return self._refuse("autoafk: in auto mode, rules go one at a time, and AFK's duration "
                                "stays; /noautoafk <rule> removes one.")
        rules = self._ordered(kind)
        if not rules:
            self._notice(f"{label}: there are no rules to remove; {self._rules_clause(kind, now)}.")
            return True
        self.rules[kind].clear()
        self.changed[kind] = True
        if kind == OFF:
            self._refresh_until(now)
        removed = ", and ".join(self._describe(kind, rule.when, now) for rule in rules)
        self._notice(f"{label}: removed {removed}; {self._rules_clause(kind, now, still=True)}.")
        return True

    def _set_rule(self, kind: str, when: AfkWhen, source: str, now: float, *,
                  fresh: bool = True) -> _Rule:
        """Make ``when`` its class's rule of ``kind``. A duration that turns
        AFK on counts from now, armed (``fresh``), unless a reboot carried its
        count over."""
        rule = _Rule(when, source, now)
        if kind == ON:
            if when.kind == "after":
                if fresh:
                    self._count_from, self._armed = now, True
            elif when.kind == "daily":
                rule.next = _next_local_time(when.clock, now)
            else:
                rule.next = when.moment.timestamp()
        self.rules[kind][when.kind] = rule
        return rule

    def _refresh_until(self, now: float) -> None:
        """While AFK is on: the soonest end that a rule sets, and its source."""
        if self.job_id is None:
            self.until = self.until_source = None
            return
        ends = self._ends(now if self.on_at is None else self.on_at)
        self.until, self.until_source = min(ends) if ends else (None, None)

    def _ends(self, on_at: float) -> List[Tuple[float, str]]:
        """Where each rule that turns AFK off ends AFK that turned on at
        ``on_at``: a duration after it, a time of day at its next after AFK
        turned on, or the rule was set, and a date and time at it."""
        ends = []
        for rule in self.rules[OFF].values():
            when = rule.when
            if when.kind == "after":
                end = on_at + when.seconds
            elif when.kind == "daily":
                end = _next_local_time(when.clock, max(on_at, rule.set_at))
            else:
                end = when.moment.timestamp()
            ends.append((end, rule.source))
        return ends

    def _ordered(self, kind: str) -> List[_Rule]:
        return [self.rules[kind][cls] for cls in _CLASSES if cls in self.rules[kind]]

    def _rules_state(self, kind: str) -> List[List[str]]:
        return [[str(rule.when), rule.source] for rule in self._ordered(kind)]

    def _autoafk_state(self) -> dict:
        state = {"seen": self._seen, "count_from": self._count_from, "armed": self._armed}
        if self.job_id is not None and self.on_at is not None:
            state["on_at"] = self.on_at
        for kind, key in ((ON, "enable"), (OFF, "disable")):
            if self.changed[kind]:
                state[key] = self._rules_state(kind)
        return state

    # Notices.

    def _describe(self, kind: str, when: AfkWhen, now: float, *, detail: bool = False) -> str:
        """A rule, as the notices say it; with ``detail``, also when it fires."""
        if kind == OFF:
            if when.kind == "after":
                return f"once it has been on {_span(when.seconds)}"
            return f"at {when.clock} each day" if when.kind == "daily" else f"at {when}"
        if when.kind == "after":
            text = f"after {_span(when.seconds)} without a message from you"
            return text + (", once per absence" if detail else "")
        if when.kind == "daily":
            text = f"at {when.clock} each day"
            rule = self.rules[ON].get("daily")
            if detail and rule is not None and rule.when == when:
                day = ("today" if datetime.fromtimestamp(rule.next).date()
                       == datetime.fromtimestamp(now).date() else "tomorrow")
                text += f", next {day}"
            return text
        text = f"once, at {when}"
        return text + (f" (in {_span(when.moment.timestamp() - now)})" if detail else "")

    def _listing(self, kind: str, now: float, *, detail: bool = False) -> str:
        """The rules of ``kind``, with their sources, joined by "or"."""
        return ", or ".join(f"{self._describe(kind, rule.when, now, detail=detail)} "
                            f"({rule.source})" for rule in self._ordered(kind))

    def _rules_clause(self, kind: str, now: float, *, still: bool = False) -> str:
        rules = self._listing(kind, now)
        if kind == ON:
            if rules:
                return f"AFK {'still ' if still else ''}turns on {rules}"
            return f"AFK {'now ' if still else ''}turns on only with /afk"
        if rules:
            return f"AFK {'still ' if still else ''}turns off {rules}, or at your next message"
        return f"AFK {'now ' if still else ''}turns off only at your next message"

    def _rules_text(self, kind: str, now: float) -> str:
        """/autoafk or /autonoafk alone: when AFK turns on and off by itself."""
        label = _LABEL[kind]
        on, off = self._listing(ON, now, detail=True), self._listing(OFF, now)
        mode = "in auto mode, until /noauto, " if self.in_auto else ""
        first = f"AFK turns on {on}" if on else "AFK turns on only with /afk"
        second = f"It turns off at your next message{', or ' + off if off else ''}"
        text = f"{label}: {mode}{first}. {second}."
        if not self.rules[kind]:
            return f"{text} {_USAGE[kind]}"
        return f"{text} /{label} <rule> sets a rule, and /no{label} <rule> removes one."

    def _until_clause(self, now: float) -> str:
        if self.until is None:
            return ""
        return f", and so will {_when_text(self.until, now)} ({self.until_source})"

    def _refuse(self, text: str) -> bool:
        self._notice(f"{text} Draft preserved.")
        return False

    # The job.

    def _schedule(self, since: float, text: Optional[str], **when) -> Job:
        job = self.icron.schedule(self._reminder(since, text), every=self.interval, by="user",
                                  via=VIA, **when)
        self.job_id, self.since, self.text = job.job_id, since, text
        return job

    def _reminder(self, since: float, text: Optional[str]) -> str:
        """A reminder: the facts, the guidance, then the user's text, if any."""
        reminder = (f"The user has been away from the keyboard (AFK) since {_iso(since)}; "
                    f"only they can end these reminders. {self.guidance}")
        return reminder if text is None else f"{reminder}\nBefore leaving, they wrote: {text}"

    def _job(self) -> Optional[Job]:
        return next((job for job in self.icron.jobs(by="user") if job.job_id == self.job_id),
                    None)

    def _clear(self) -> None:
        self.job_id = self.since = self.text = self.on_at = None
        self.until = self.until_source = None


def _passed(when: AfkWhen, now: float) -> bool:
    """Whether ``when`` is a date and time that has passed."""
    return when.kind == "once" and when.moment.timestamp() <= now


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
