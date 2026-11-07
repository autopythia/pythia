"""AFK and its rules: the CLI tells main that the user is away (doc/cli-afk-plan.md,
doc/cli-auto-plan.md)."""

from __future__ import annotations

from datetime import datetime
import json
import re
import unittest
from unittest import mock
import warnings

from pythia.interaction import Icron, IcronError, InteractionConfig, cli
from pythia.interaction import _cli_afk
from pythia.interaction._cli_afk import AUTOAFK_USAGE, AUTONOAFK_USAGE, Afk
from pythia.interaction.loop.icron import WAKE_HEADER
from pythia.interaction.loop.supervision import Yield
from pythia.interaction.model_catalog import AfkDefaults, AfkWhen, parse_afk_when
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import render_model_catalog
from pythia_test.interaction_helpers import patch_compaction
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_cli_icron import NOW, _Clock, _IcronControllerCase, _enter
from pythia_test.test_interaction_cli_icron import _exit, _has, _idle_with, _notices, _once
from pythia_test.test_interaction_cli_icron import _until, _users
from pythia_test.test_interaction_cli_reboot import _Compactor, _RebootCase, _rebootable
from pythia_test.test_interaction_cli_reboot import _steps


def _iso(seconds):
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def _reminder(since, text=None):
    reminder = (f"The user has been away from the keyboard (AFK) since {_iso(since)}; only "
                "they can end these reminders. They can't answer until they're back, so don't "
                "wait for them: carry on where you can.")
    return reminder if text is None else f"{reminder}\nBefore leaving, they wrote: {text}"


def _on(job_id, tail="", head="afk: on"):
    return (f"{head}; main hears now, then every 30m, that you're away (job {job_id}). Your "
            f"next message ends it{tail}.")


def _ended(kind="ended"):
    """main's turn ended: the run of the job that it took a reminder from ends."""
    extra = {"ended": {"final_text": "Carrying on."}, "yielded": {"yield_text": "waiting"}}
    return extra[kind]


class _AfkCase(unittest.TestCase):
    """A CLI state with icron and /afk, on icron's fake clock (wall time).
    tick() lets it go on, then checks at once (_cli_afk._check_clock)."""

    defaults = AfkDefaults()

    def setUp(self):
        self.clock = _Clock()
        self.check = _Clock(100.0)
        patcher = mock.patch.object(_cli_afk, "_check_clock", self.check)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.icron = Icron(clock=self.clock)
        self.icron.start(lambda: None, lambda text: None)
        self.addCleanup(self.icron.close, 1.0)
        self.state = cli._UIState(ready=True, icron=self.icron)
        self.state.afk = self.afk = Afk(self.icron, self.defaults, self.state.notice)
        self.state.set_phase("sampling")  # /afk works while main is busy

    def enter(self, text):
        _enter(self.state, text)
        return _notices(self.state)

    def job(self, job_id):
        return next(job for job in self.icron.jobs() if job.job_id == job_id)

    def turn_on(self, text="/afk"):
        (notice,) = self.enter(text)
        return re.search(r"\(job (\S+)\)", notice).group(1)

    def tick(self, seconds=0.0, *, may_enable=True):
        self.clock.now += seconds
        self.check.now += _cli_afk._CHECK_SECONDS
        self.afk.tick(may_enable=may_enable)
        return _notices(self.state)

    def at(self, hour, minute, *, day=15, may_enable=True):
        """Tick at a local time of NOW's day (January 15, 2027), or another day's."""
        self.clock.now = datetime(2027, 1, day, hour, minute).timestamp()
        return self.tick(may_enable=may_enable)

    def remind(self):
        """Let AFK's reminder fall due, and wait until icron queues it."""
        self.clock.now += 1
        self.icron._poke()
        self.assertTrue(_until(lambda: self.icron.undelivered() == 1))

    def end_turn(self, job_id=None, kind="ended"):
        self.icron.handoff(Yield(context=1, job_id=job_id, job_text="task", kind=kind,
                                 resumable=True, revision=0, **_ended(kind)))


class CommandTests(_AfkCase):
    def test_afk_turns_on_and_again_starts_it_over_with_its_text(self):
        (notice,) = self.enter("/afk")
        job_id = re.fullmatch(r"afk: on; main hears now, then every 30m, that you're away "
                              r"\(job (\S+)\)\. Your next message ends it\.", notice).group(1)
        self.assertEqual(self.state.editor.text, "")
        job = self.job(job_id)
        self.assertEqual((job.origin, job.kind, job.via, job.every, job.due, job.message),
                         ("user", "message", "/afk", 1800.0, NOW + 1, _reminder(NOW)))
        self.assertEqual(self.afk.status(), "afk since 12:00")
        self.clock.now += 600
        (notice,) = self.enter("/afk Back at 13:00; finish the parser.")
        second = self.afk.job_id
        self.assertEqual(notice, _on(second, head="afk: on again, with your text"))
        self.assertEqual(self.job(job_id).state, "cancelled")
        self.assertEqual(self.job(second).message,
                         _reminder(NOW + 600, "Back at 13:00; finish the parser."))
        self.assertEqual(self.afk.status(), "afk since 12:10")
        self.assertEqual(self.enter("/afk"), [_on(self.afk.job_id, head="afk: on again")])
        self.assertEqual(self.job(second).state, "cancelled")
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(NOW + 600))

    def test_its_text_keeps_its_lines(self):
        self.turn_on("/afk  First the parser,\n  then the docs.  ")
        self.assertEqual(self.job(self.afk.job_id).message,
                         _reminder(NOW, "First the parser,\n  then the docs."))

    def test_on_and_off_aren_t_text_and_long_text_or_a_full_job_list_keep_the_draft(self):
        refusals = {
            "/afk off": "afk: there's no /afk off: your next message to main ends AFK, and "
                        "/cron cancel ends its job. Draft preserved.",
            "/afk On": "afk: there's no /afk on: /afk alone turns AFK on. Draft preserved.",
        }
        for text, refusal in refusals.items():
            with self.subTest(text=text):
                self.assertEqual(self.enter(text), [refusal])
                self.assertEqual(self.state.editor.text, text)
                self.assertIsNone(self.afk.job_id)
        job_id = self.turn_on()
        self.assertEqual(self.enter("/afk off"), [refusals["/afk off"]])
        self.assertTrue(self.afk.sync())
        long = "/afk " + "x" * 8000
        (notice,) = self.enter(long)
        self.assertRegex(notice, r"^afk: the text is too long: a reminder has at most 8000 "
                                 r"characters, and this one would have 8\d{3}\. Draft preserved\.$")
        self.assertEqual((self.state.editor.text, self.afk.job_id), (long, job_id))
        self.icron.cancel(job_id, by="user")
        self.afk.sync()
        _notices(self.state)
        for _ in range(16):
            self.icron.schedule("x", after=600, by="user", via="/cron")
        self.assertEqual(self.enter("/afk"), [
            "icron.schedule(): the user has 16 pending jobs already; cancel one first. Draft "
            "preserved."])
        self.assertEqual(self.state.editor.text, "/afk")
        self.assertIsNone(self.afk.job_id)

    def test_it_works_while_the_model_needs_authentication_and_not_without_icron(self):
        self.state.auth_required = True
        self.turn_on()
        self.assertTrue(self.enter("/autoafk 20m")[0].startswith("autoafk: AFK turns on"))
        state = cli._UIState(ready=True, icron=self.icron)  # no /afk: a headless state
        for text in ("/afk", "/autoafk"):
            _enter(state, text)
            self.assertTrue(_notices(state)[0].startswith("Unsupported command. Use /retry"))

    def test_cron_lists_its_job_and_cancels_it_and_main_can_t(self):
        job_id = self.turn_on("/afk Lunch.")
        (listing,) = self.enter("/cron")
        self.assertTrue(listing.startswith(
            f"cron: 1 pending job (now 12:00:00):\n  {job_id} (yours, /afk) 12:00:01, in 1s, "
            "every 30m: The user has been away"), listing)
        with self.assertRaisesRegex(IcronError, "you can cancel only your own jobs"):
            self.icron.cancel(job_id, by="main")
        self.icron.schedule("ping", after=600, by="user", via="/cron")
        self.assertEqual(self.enter("/cron clear"), ["cron: cancelled 1 job of yours."])
        self.assertTrue(self.afk.sync())  # /cron clear cancels /cron's jobs only
        self.assertEqual(self.enter(f"/cron cancel {job_id}"), [
            f"cron: cancelled job {job_id} (yours).", f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.afk.status(), "")
        job_id = self.turn_on()
        self.assertEqual(self.enter("/cron cancel! all"), [
            "cron: cancelled 1 job.", f"afk: off; its job {job_id} was cancelled."])

    def test_a_job_that_ends_another_way_ends_afk_at_the_next_tick(self):
        job_id = self.turn_on()
        self.icron.cancel(job_id, by="user")  # as /py code might
        self.assertEqual(self.afk.status(), "afk since 12:00")  # until the tick
        self.assertEqual(self.tick(), [f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.afk.status(), "")


class ReminderTests(_AfkCase):
    def test_the_first_reminder_fires_at_once_and_the_next_at_its_firing(self):
        job_id = self.turn_on("/afk Back at 13:00.")
        self.remind()
        self.icron.begin_turn()
        wakeup = self.icron.take_wakes()
        self.assertEqual(wakeup.text(), "\n".join([
            f"{WAKE_HEADER} at {_iso(NOW + 1)}.", "",
            f"Job {job_id} (the user's, from /afk; due {_iso(NOW + 1)}, every 1800 s):",
            _reminder(NOW, "Back at 13:00.")]))
        # Its run ended once it queued the reminder; the next slot sends another, which
        # the running turn takes at its next interrupt point, as any repeat's message.
        self.assertTrue(_until(lambda: self.job(job_id).state == "scheduled"))
        self.clock.now += 1800
        self.icron._poke()
        self.assertTrue(_until(lambda: self.icron.undelivered() == 1))
        self.assertEqual(self.job(job_id).due, NOW + 3601)
        self.assertEqual([delivery.due for delivery in self.icron.take_wakes().deliveries],
                         [NOW + 1801])
        self.end_turn()
        self.assertTrue(self.afk.sync())


class EndTests(_AfkCase):
    def test_a_message_for_main_ends_afk(self):
        for text in ("I'm back.", "/steer I'm back.", "/task Next, the docs."):
            with self.subTest(text=text):
                self.turn_on()
                self.clock.now += 60
                self.assertIn("afk: off after 1m: you wrote to main.", self.enter(text))
                self.assertIsNone(self.afk.job_id)

    def test_commands_keys_and_wake_ups_leave_it_on(self):
        job_id = self.turn_on()
        self.state.editor = self.state.editor.edit("<bracketed-paste>", "half a thought")
        for text in ("/cron", "/cron clear", "/autoafk", "/autoafk 20m", "/cron in 1m ping"):
            self.enter(text)
        # Wake-ups, AFK's own reminder and a /cron message among them, reach main
        # without the user's typing: they leave AFK on.
        self.clock.now += 60
        self.icron._poke()
        self.assertTrue(_until(lambda: self.icron.undelivered() == 2))
        cli._queue_wake(self.state)
        self.state.turn_active = True
        interrupt = cli._take_interrupt(self.state)
        self.assertEqual(len(interrupt.messages), 1)
        self.assertIn("from /afk", interrupt.messages[0].content_text)
        self.assertTrue(self.afk.sync())
        self.assertEqual(self.afk.job_id, job_id)
        self.assertFalse(any(notice.startswith("afk: off") for notice in _notices(self.state)))

    def test_ending_afk_stops_its_job_wherever_its_reminder_is(self):
        def stopped(job_id):
            self.assertEqual(self.job(job_id).state, "cancelled")
            self.assertNotIn(job_id, [job.job_id for job in self.icron.status().jobs])
            self.clock.now += 7200  # hours on, nothing fires
            self.icron._poke()
            self.assertFalse(_until(lambda: self.icron.undelivered(), timeout=0.2))

        # Due, not yet delivered: the reminder is withdrawn, and the queued
        # marker delivers nothing.
        job_id = self.turn_on()
        self.remind()
        cli._queue_wake(self.state)
        self.enter("I'm back.")
        self.assertEqual(self.icron.undelivered(), 0)
        self.assertIsNone(self.icron.take_wakes())
        stopped(job_id)
        # Taken by the running turn: the job ends at once.
        job_id = self.turn_on()
        self.remind()
        self.icron.begin_turn()
        self.icron.take_wakes()
        self.assertEqual(self.job(job_id).state, "scheduled")  # its run ended once queued
        self.enter("/steer I'm back.")
        stopped(job_id)
        self.end_turn()
        stopped(job_id)
        # Between reminders.
        job_id = self.turn_on()
        self.remind()
        self.icron.begin_turn()
        self.icron.take_wakes()
        self.end_turn()
        self.assertTrue(_until(lambda: self.job(job_id).state == "scheduled"))
        self.enter("Back.")
        stopped(job_id)


class DisableTests(_AfkCase):
    defaults = AfkDefaults(auto_disable="12:30", auto_enable="10m")

    def test_afk_ends_at_the_time_and_the_duration_waits_for_a_message(self):
        job_id = self.turn_on()
        self.assertEqual(self.job(job_id).state, "scheduled")
        self.assertEqual(self.at(12, 29), [])
        self.assertEqual(self.at(12, 30), ["afk: off at 12:30 ([afk] auto_disable), after 30m."])
        self.assertEqual(self.job(job_id).state, "cancelled")
        self.assertEqual(self.tick(3600), [])  # away for long, but no message since: off
        self.enter("/cron")  # a command isn't a message
        self.assertEqual(self.tick(3600), [])
        self.enter("Still there?")
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith(
            "afk: on after 10m without a message ([afk] auto_enable); "))

    def test_the_notice_says_when_it_ends_and_a_passed_time_is_tomorrow_s(self):
        self.assertEqual(self.enter("/afk")[0], _on(self.afk.job_id, ", and so will 12:30 "
                                                                     "([afk] auto_disable)"))
        afk = Afk(self.icron, AfkDefaults(auto_disable="9:00"), self.state.notice)
        afk.enable()
        self.assertEqual(_notices(self.state), [_on(afk.job_id, ", and so will 09:00 tomorrow "
                                                                "([afk] auto_disable)")])
        self.assertEqual(afk.until, datetime(2027, 1, 16, 9, 0).timestamp())

    def test_a_duration_ends_afk_once_it_has_been_on_that_long(self):
        self.state.afk = self.afk = Afk(self.icron, AfkDefaults(auto_disable="8h"),
                                        self.state.notice)
        self.assertEqual(self.afk.startup_notices(), [
            "afk: AFK turns off once it has been on 8h ([afk] auto_disable), or at your next "
            "message; /autonoafk shows or changes this."])
        self.clock.now += 3600
        self.assertEqual(self.enter("/afk")[0], _on(self.afk.job_id, ", and so will 21:00 "
                                                                     "([afk] auto_disable)"))
        self.assertEqual(self.tick(8 * 3600 - 1), [])
        self.assertEqual(self.tick(1), ["afk: off at 21:00 ([afk] auto_disable), after 8h."])

    def test_a_date_and_time_ends_afk_once_and_is_spent(self):
        self.state.afk = self.afk = Afk(self.icron, AfkDefaults(auto_disable="2027-01-16 9:00"),
                                        self.state.notice)
        tail = ", and so will 09:00 tomorrow ([afk] auto_disable)"
        self.assertEqual(self.enter("/afk")[0], _on(self.afk.job_id, tail))
        self.assertEqual(self.at(9, 0, day=16), [
            "afk: off at 09:00 ([afk] auto_disable), after 21h."])
        self.assertEqual(self.afk.rules["off"], {})
        self.assertEqual(self.enter("/afk")[0], _on(self.afk.job_id))  # no end now
        # Spent also while AFK is off.
        afk = Afk(self.icron, AfkDefaults(auto_disable="2027-01-16 10:00"), self.state.notice)
        self.clock.now = datetime(2027, 1, 16, 10, 0).timestamp()
        afk.tick(may_enable=True)
        self.assertEqual(afk.rules["off"], {})

    def test_main_waiting_for_the_next_reminder_is_released_and_the_notice_says_so(self):
        job_id = self.turn_on()
        self.remind()
        self.icron.begin_turn()
        self.icron.take_wakes()
        self.end_turn(job_id, "yielded")  # main yields until AFK's next reminder
        self.assertTrue(_until(lambda: self.icron.main.state == "suspended"))
        self.assertEqual(self.at(12, 30), [
            "afk: off at 12:30 ([afk] auto_disable), after 30m; main now waits for your "
            "message."])
        self.assertNotEqual(self.icron.main.state, "suspended")
        self.clock.now += 7200
        self.icron._poke()
        self.assertFalse(_until(lambda: self.icron.undelivered(), timeout=0.2))


class RuleCommandTests(_AfkCase):
    def on_rules(self):
        return {cls: (str(rule.when), rule.source) for cls, rule in self.afk.rules["on"].items()}

    def test_autoafk_sets_shows_replaces_and_removes_rules_of_each_class(self):
        self.assertEqual(self.enter("/autoafk"), [
            "autoafk: AFK turns on only with /afk. It turns off at your next message. "
            + AUTOAFK_USAGE])
        self.assertEqual(self.enter("/autoafk 20m"), [
            "autoafk: AFK turns on after 20m without a message from you, once per absence. "
            "/noautoafk 20m removes this."])
        self.assertEqual(self.enter("/autoafk 18:00"), [
            "autoafk: AFK also turns on at 18:00 each day, next today. /noautoafk 18:00 "
            "removes this."])
        self.assertEqual(self.enter("/autoafk 9:00"), [
            "autoafk: AFK also turns on at 09:00 each day, next tomorrow, instead of at 18:00 "
            "each day. /noautoafk 09:00 removes this."])
        self.assertEqual(self.enter("/autoafk 2027-01-16T18:00"), [
            "autoafk: AFK also turns on once, at 2027-01-16 18:00 (in 1d 6h). /noautoafk "
            "2027-01-16 18:00 removes this."])
        self.assertEqual(self.on_rules(), {"after": ("20m", "/autoafk"),
                                           "daily": ("09:00", "/autoafk"),
                                           "once": ("2027-01-16 18:00", "/autoafk")})
        self.assertEqual(self.enter("/autoafk"), [
            "autoafk: AFK turns on after 20m without a message from you, once per absence "
            "(/autoafk), or at 09:00 each day, next tomorrow (/autoafk), or once, at 2027-01-16 "
            "18:00 (in 1d 6h) (/autoafk). It turns off at your next message. /autoafk <rule> "
            "sets a rule, and /noautoafk <rule> removes one."])
        # Removing matches the rule, not its text: 9:00 is 09:00, and 1200s is 20m.
        self.assertEqual(self.enter("/noautoafk 9:00"), [
            "autoafk: removed at 09:00 each day; AFK still turns on after 20m without a message "
            "from you (/autoafk), or once, at 2027-01-16 18:00 (/autoafk)."])
        self.assertEqual(self.enter("/noautoafk 1200s"), [
            "autoafk: removed after 20m without a message from you; AFK still turns on once, "
            "at 2027-01-16 18:00 (/autoafk)."])
        self.enter("/autoafk 20m")
        self.assertEqual(self.enter("/noautoafk"), [
            "autoafk: removed after 20m without a message from you, and once, at 2027-01-16 "
            "18:00; AFK now turns on only with /afk."])
        self.assertEqual(self.enter("/noautoafk"), [
            "autoafk: there are no rules to remove; AFK turns on only with /afk."])
        self.assertEqual(self.state.editor.text, "")

    def test_a_rule_that_matches_none_or_doesn_t_read_keeps_the_draft(self):
        self.enter("/autoafk 20m")
        for text, notice in (
                ("/noautoafk 30m", "autoafk: there's no rule 30m; AFK turns on after 20m "
                                   "without a message from you (/autoafk)."),
                ("/noautoafk 18:00", "autoafk: there's no rule 18:00; AFK turns on after 20m "
                                     "without a message from you (/autoafk)."),
                ("/autoafk off", "autoafk: there's no /autoafk off: /noautoafk <rule> removes a "
                                 "rule, and /noautoafk alone removes them all."),
                ("/autoafk soon", AUTOAFK_USAGE), ("/autoafk 30s", AUTOAFK_USAGE),
                ("/autoafk 8d", AUTOAFK_USAGE), ("/noautoafk soon", AUTOAFK_USAGE),
                ("/autonoafk 24:00", AUTONOAFK_USAGE),
                ("/autoafk 2027-02-30 10:00", AUTOAFK_USAGE),
                ("/autoafk 2027-01-15 11:00", "autoafk: 2027-01-15 11:00 has passed."),
                ("/autonoafk 2027-01-15 11:00", "autonoafk: 2027-01-15 11:00 has passed.")):
            with self.subTest(text=text):
                self.assertEqual(self.enter(text), [f"{notice} Draft preserved."])
                self.assertEqual(self.state.editor.text, text)
        self.assertEqual(self.on_rules(), {"after": ("20m", "/autoafk")})

    def test_autonoafk_sets_shows_and_removes_the_rules_that_turn_afk_off(self):
        self.assertEqual(self.enter("/autonoafk"), [
            "autonoafk: AFK turns on only with /afk. It turns off at your next message. "
            + AUTONOAFK_USAGE])
        self.assertEqual(self.enter("/autonoafk 8h"), [
            "autonoafk: AFK turns off once it has been on 8h. /noautonoafk 8h removes this."])
        self.assertEqual(self.enter("/autonoafk 08:30"), [
            "autonoafk: AFK also turns off at 08:30 each day. /noautonoafk 08:30 removes this."])
        self.assertEqual(self.enter("/autonoafk"), [
            "autonoafk: AFK turns on only with /afk. It turns off at your next message, or once "
            "it has been on 8h (/autonoafk), or at 08:30 each day (/autonoafk). /autonoafk "
            "<rule> sets a rule, and /noautonoafk <rule> removes one."])
        self.assertEqual(self.enter("/noautonoafk 480m"), [
            "autonoafk: removed once it has been on 8h; AFK still turns off at 08:30 each day "
            "(/autonoafk), or at your next message."])
        self.assertEqual(self.enter("/noautonoafk"), [
            "autonoafk: removed at 08:30 each day; AFK now turns off only at your next message."])

    def test_a_time_can_t_turn_afk_both_on_and_off(self):
        self.enter("/autonoafk 08:30")
        self.assertEqual(self.enter("/autoafk 8:30"), [
            "autoafk: 08:30 turns AFK off too (/autonoafk), so AFK would end and start again "
            "there. Draft preserved."])
        self.enter("/autoafk 18:00")
        self.assertEqual(self.enter("/autonoafk 18:00"), [
            "autonoafk: 18:00 turns AFK on too (/autoafk), so AFK would end and start again "
            "there. Draft preserved."])
        self.enter("/autoafk 20m")
        self.assertEqual(self.enter("/autonoafk 20m"), [  # durations may match
            "autonoafk: AFK also turns off once it has been on 20m. /noautonoafk 20m removes "
            "this."])

    def test_a_change_to_the_rules_that_turn_afk_off_applies_to_afk_that_s_on(self):
        job_id = self.turn_on()
        self.clock.now += 3600  # 13:00, and AFK on since 12:00
        self.assertEqual(self.enter("/autonoafk 90m"), [
            "autonoafk: AFK turns off once it has been on 1h 30m. /noautonoafk 90m removes "
            "this."])
        self.assertEqual(self.afk.until, NOW + 5400)  # from when AFK turned on
        self.enter("/autonoafk 13:10")
        self.assertEqual((self.afk.until, self.afk.until_source), (NOW + 4200, "/autonoafk"))
        self.enter("/noautonoafk 13:10")
        self.assertEqual(self.afk.until, NOW + 5400)
        self.assertEqual(self.tick(1799), [])
        self.assertEqual(self.tick(1), ["afk: off at 13:30 (/autonoafk), after 1h 30m."])
        self.assertEqual(self.job(job_id).state, "cancelled")


class CombinedRulesTests(_AfkCase):
    defaults = AfkDefaults(auto_enable="20m, 12:30", auto_disable="1h, 13:10")

    def test_any_rule_turns_afk_on_and_the_first_end_turns_it_off(self):
        self.assertEqual(self.afk.startup_notices(), [
            "afk: AFK turns on after 20m without a message from you ([afk] auto_enable), or at "
            "12:30 each day ([afk] auto_enable), and off once it has been on 1h ([afk] "
            "auto_disable), or at 13:10 each day ([afk] auto_disable) or at your next message; "
            "/autoafk and /autonoafk show or change this."])
        (notice,) = self.at(12, 20)  # the duration comes first
        self.assertEqual(notice, _on(self.afk.job_id, ", and so will 13:10 ([afk] auto_disable)",
                                     head="afk: on after 20m without a message ([afk] "
                                          "auto_enable)"))
        self.assertEqual(self.at(12, 30), [])  # on already: the time of day waits a day
        self.assertEqual(self.at(13, 10), ["afk: off at 13:10 ([afk] auto_disable), after 1h "
                                           "10m."])
        self.assertEqual(self.at(12, 29, day=16), [])  # disarmed: no message since
        (notice,) = self.at(12, 30, day=16)
        self.assertEqual(notice, _on(self.afk.job_id, ", and so will 13:10 ([afk] auto_disable)",
                                     head="afk: on at 12:30 ([afk] auto_enable)"))


class DurationTests(_AfkCase):
    defaults = AfkDefaults(auto_enable="10m")

    def test_it_turns_on_after_the_duration_without_a_message_once_per_absence(self):
        self.assertEqual(self.afk.startup_notices(), [
            "afk: AFK turns on after 10m without a message from you ([afk] auto_enable); "
            "/autoafk shows or changes this."])
        self.assertEqual(self.tick(599), [])
        self.assertEqual(self.tick(1, may_enable=False), [])  # not while it may not
        (notice,) = self.tick(1)
        job_id = self.afk.job_id
        self.assertEqual(notice, _on(job_id, head="afk: on after 10m 1s without a message "
                                                  "([afk] auto_enable)"))
        self.assertEqual(self.job(job_id).message, _reminder(NOW))  # since the CLI's start
        # Ended another way, AFK doesn't come back in the same absence.
        self.icron.cancel(job_id, by="user")
        self.assertEqual(self.tick(), [f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.tick(3600), [])
        self.enter("/cron")  # nor after a command: only a message is coming back
        self.assertEqual(self.tick(3600), [])
        message_at = self.clock.now
        self.enter("Here.")
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 10m without a message "))
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(message_at))
        # Your message ends it, and the duration counts from it again.
        self.assertEqual(self.enter("Back."), ["afk: off after 10m: you wrote to main."])
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 10m without a message "))

    def test_autoafk_s_duration_counts_from_the_command(self):
        self.enter("/noautoafk")
        self.assertEqual(self.tick(3600), [])
        self.enter("/autoafk 20m")
        self.assertEqual(self.tick(1199), [])
        (notice,) = self.tick(1)
        self.assertEqual(notice, _on(self.afk.job_id, head="afk: on after 20m without a "
                                                           "message (/autoafk)"))
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(NOW))  # no message since

    def test_a_failure_to_turn_on_says_why_once(self):
        for _ in range(16):
            self.icron.schedule("x", after=600, by="user", via="/cron")
        self.assertEqual(self.tick(600), [
            "afk: not turned on after 10m without a message ([afk] auto_enable): "
            "icron.schedule(): the user has 16 pending jobs already; cancel one first."])
        self.assertEqual(self.tick(600), [])


class DailyTests(_AfkCase):
    defaults = AfkDefaults(auto_enable="12:30", auto_disable="13:00")

    def test_afk_turns_on_at_the_time_each_day(self):
        self.assertEqual(self.afk.startup_notices(), [
            "afk: AFK turns on at 12:30 each day ([afk] auto_enable), and off at 13:00 each day "
            "([afk] auto_disable) or at your next message; /autoafk and /autonoafk show or "
            "change this."])
        self.assertEqual(self.at(12, 29), [])
        (notice,) = self.at(12, 30)
        job_id = self.afk.job_id
        self.assertEqual(notice, _on(job_id, ", and so will 13:00 ([afk] auto_disable)",
                                     head="afk: on at 12:30 ([afk] auto_enable)"))
        self.assertEqual(self.job(job_id).message, _reminder(NOW))  # since the last message
        # The absence it reports runs from since, the last message.
        self.assertEqual(self.at(13, 0), ["afk: off at 13:00 ([afk] auto_disable), after 1h."])
        self.assertEqual(self.at(12, 29, day=16), [])
        self.assertTrue(self.at(12, 30, day=16)[0].startswith("afk: on at 12:30 "))

    def test_a_message_ends_afk_and_the_trigger_stays(self):
        self.at(12, 30)
        self.assertEqual(self.enter("Back."), ["afk: off after 30m: you wrote to main."])
        self.assertEqual(self.at(12, 31), [])
        self.assertTrue(self.at(12, 30, day=16)[0].startswith("afk: on at 12:30 "))

    def test_on_already_at_the_time_it_waits_for_the_next_day(self):
        self.turn_on()
        self.assertEqual(self.at(12, 30), [])
        self.assertEqual(self.enter("Back."), ["afk: off after 30m: you wrote to main."])
        self.assertEqual(self.at(12, 40), [])  # today's time passed while it was on
        self.assertTrue(self.at(12, 30, day=16)[0].startswith("afk: on at 12:30 "))

    def test_a_start_after_the_time_waits_and_a_time_that_may_not_fire_waits(self):
        self.state.afk = self.afk = Afk(self.icron, AfkDefaults(auto_enable="9:00"),
                                        self.state.notice)
        self.assertEqual(self.tick(), [])  # it's noon: today's 09:00 has passed
        self.assertEqual(self.at(9, 0, day=16, may_enable=False), [])  # a reboot pending, say
        self.assertTrue(self.tick(60)[0].startswith("afk: on at 09:00 ([afk] auto_enable); "))

    def test_a_failure_to_turn_on_says_why_and_waits_for_the_next_day(self):
        for _ in range(16):
            self.icron.schedule("x", after=600, by="user", via="/cron")
        self.assertEqual(self.at(12, 30), [
            "afk: not turned on at 12:30 ([afk] auto_enable): icron.schedule(): the user "
            "has 16 pending jobs already; cancel one first."])
        self.assertEqual(self.tick(), [])


class OnceTests(_AfkCase):
    def test_a_date_and_time_turns_afk_on_once_and_removes_itself(self):
        self.enter("/autoafk 2027-01-15 12:30")
        self.assertEqual(self.at(12, 29), [])
        (on, off) = self.at(12, 30)
        self.assertEqual(on, _on(self.afk.job_id, head="afk: on at 12:30 (/autoafk)"))
        self.assertEqual(off, "autoafk: 2027-01-15 12:30 has passed, so that rule is gone.")
        self.assertEqual(self.afk.rules["on"], {})
        self.enter("Back.")
        self.assertEqual(self.at(12, 30, day=16), [])

    def test_on_already_at_the_time_it_only_removes_itself(self):
        self.enter("/autoafk 2027-01-15 12:30")
        job_id = self.turn_on()
        self.assertEqual(self.at(12, 30), [
            "autoafk: 2027-01-15 12:30 has passed, so that rule is gone."])
        self.assertEqual(self.afk.job_id, job_id)

    def test_the_catalog_s_passed_dates_and_times_are_ignored_at_launch(self):
        afk = Afk(self.icron, AfkDefaults(auto_enable="2027-01-15 11:00",
                                          auto_disable="2027-01-15 11:30"), self.state.notice)
        self.assertEqual(afk.startup_notices(), [
            "afk: [afk] auto_enable's 2027-01-15 11:00 has passed, so it's ignored.",
            "afk: [afk] auto_disable's 2027-01-15 11:30 has passed, so it's ignored."])
        self.assertEqual((afk.rules["on"], afk.rules["off"]), ({}, {}))


class CatalogTests(unittest.TestCase):
    def parse(self, text):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no catalog version
            return parse_model_catalog(text)

    def test_the_afk_section(self):
        catalog = self.parse("[afk]\ninterval_seconds = 900\nauto_enable = 20m\n"
                             'auto_disable = 8:30\nmessage = "Keep to the plan."\n')
        self.assertEqual(catalog.afk, AfkDefaults(
            interval_seconds=900, auto_enable=AfkWhen("after", seconds=1200),
            auto_disable=AfkWhen("daily", clock="08:30"), message="Keep to the plan."))
        self.assertIn('[afk] interval_seconds = 900, auto_enable = 20m, auto_disable = 08:30, '
                      'message = "Keep to the plan."', render_model_catalog(catalog))
        self.assertEqual(self.parse("[afk]\nmessage = Keep going.\n").afk.message, "Keep going.")
        once = self.parse('[afk]\nauto_enable = "2027-01-16T18:00"\nauto_disable = 2d\n').afk
        self.assertEqual(([str(rule) for rule in once.auto_enable],
                          [str(rule) for rule in once.auto_disable]),
                         (["2027-01-16 18:00"], ["2d"]))
        unset = self.parse("[afk]\ninterval_seconds = null\nauto_enable = null\n")
        self.assertEqual(unset.afk, AfkDefaults())
        self.assertNotIn("[afk]", render_model_catalog(unset))
        self.assertEqual(AfkDefaults(auto_enable="18:00"),
                         AfkDefaults(auto_enable=parse_afk_when("18:00")))

    def test_each_kind_takes_a_rule_of_each_class(self):
        catalog = self.parse("[afk]\nauto_enable = 20m, 18:00, 2027-01-16 09:00\n"
                             'auto_disable = "8h, 08:30"\n')
        self.assertEqual([str(rule) for rule in catalog.afk.auto_enable],
                         ["20m", "18:00", "2027-01-16 09:00"])
        self.assertEqual([str(rule) for rule in catalog.afk.auto_disable], ["8h", "08:30"])
        self.assertIn('[afk] auto_enable = "20m, 18:00, 2027-01-16 09:00", auto_disable = '
                      '"8h, 08:30"', render_model_catalog(catalog))
        for line, error in (
                ("auto_enable = 20m, 30m", r"auto_enable in user: expected at most one rule per "
                                           r"class, but 20m and 30m are both durations, or null"),
                ("auto_disable = 08:30, 9:00", r"auto_disable in user: expected at most one rule "
                                               r"per class, but 08:30 and 09:00 are both times "
                                               r"of day"),
                ("auto_enable = 20m, 59s", r"auto_enable in user: 59s: expected a duration from "
                                           r"1m to 7d, or null"),
                ("auto_enable = 20m,", r"auto_enable in user: an empty rule: expected a "
                                       r"duration such as 20m"),
                ("auto_enable = 20m, 18:00\nauto_disable = 8h, 18:00",
                 r"\[afk\] in user: auto_enable and auto_disable both have 18:00")):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, error):
                self.parse(f"[afk]\n{line}\n")
        with self.assertRaisesRegex(ValueError, "afk auto_enable: expected at most one rule"):
            AfkDefaults(auto_enable=(AfkWhen("daily", clock="9:00"),
                                     AfkWhen("daily", clock="10:00")))

    def test_when_reads_its_three_forms_and_writes_them_back(self):
        for text, kind, canonical in (
                ("20m", "after", "20m"), ("90m", "after", "90m"), ("3600s", "after", "1h"),
                ("7d", "after", "7d"), ("8:30", "daily", "08:30"), ("23:59", "daily", "23:59"),
                ("2027-01-16 18:00", "once", "2027-01-16 18:00"),
                (" 2027-01-16T9:05 ", "once", "2027-01-16 09:05")):
            with self.subTest(text=text):
                when = parse_afk_when(text)
                self.assertEqual((when.kind, str(when)), (kind, canonical))
                self.assertEqual(parse_afk_when(str(when)), when)

    def test_bad_keys_and_values_name_what_was_expected(self):
        when = (r"expected a duration such as 20m \(s, m, h, or d; 1m to 7d\), a local time of "
                r"day such as 18:00, or a local date and time such as 2026-11-07 18:00, or null")
        cases = [
            ("when = 1", r"\[afk\] key 'when' in user; expected interval_seconds, auto_enable, "
                         r"auto_disable, or message"),
            ("auto_enable_seconds = 600", r"\[afk\] key 'auto_enable_seconds' in user"),
            ("auto_disable_on_message = true", r"\[afk\] key 'auto_disable_on_message' in user"),
            ("interval_seconds = 29", r"interval_seconds in user: expected a whole number of "
                                      r"seconds from 30 to 604800, or null"),
            ("interval_seconds = 1800.0", r"interval_seconds in user: expected a whole number"),
            ("auto_enable = 59s", r"auto_enable in user: expected a duration from 1m to 7d, or "
                                  r"null"),
            ("auto_enable = 8d", r"auto_enable in user: expected a duration from 1m to 7d"),
            ("auto_disable = 24:00", r"auto_disable in user: " + when),
            ("auto_disable = 8.30", r"auto_disable in user: " + when),
            ("auto_enable = 18", r"auto_enable in user: " + when),
            ("auto_enable = 2027-02-30 10:00", r"auto_enable in user: expected a real date and "
                                               r"time, such as 2026-11-07 18:00, or null"),
            ('auto_enable = "unclosed', r"auto_enable in user: " + when),
            ("auto_enable = 8:30\nauto_disable = 08:30",
             r"\[afk\] in user: auto_enable and auto_disable both have 08:30, so AFK would end "
             r"and start again there"),
            ("auto_enable = 2027-01-16 18:00\nauto_disable = 2027-01-16T18:00",
             r"\[afk\] in user: auto_enable and auto_disable both have 2027-01-16 18:00"),
            ('message = "unclosed', r"message in user: expected nonblank text of at most 4000 "
                                    r"characters, or null"),
            ("message = " + "x" * 4001, r"message in user: expected nonblank text"),
        ]
        for line, error in cases:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, error):
                self.parse(f"[afk]\n{line}\n")
        self.parse("[afk]\nauto_enable = 20m\nauto_disable = 20m\n")  # durations may match
        with self.assertRaisesRegex(ValueError, "afk interval_seconds: expected"):
            AfkDefaults(interval_seconds=True)
        with self.assertRaisesRegex(ValueError, "afk auto_enable and auto_disable both have "
                                                "09:00"):
            AfkDefaults(auto_enable="9:00", auto_disable="09:00")
        with self.assertRaisesRegex(ValueError, "after sets only seconds"):
            AfkWhen("after", seconds=60, clock="09:00")


class RebootTests(_AfkCase):
    def on_rules(self, afk):
        return {cls: (str(rule.when), rule.source) for cls, rule in afk.rules["on"].items()}

    def test_afk_s_job_doesn_t_refuse_a_reboot_but_others_still_do(self):
        self.turn_on()
        with _rebootable():
            self.assertIsNone(cli._reboot_refusal(self.state))
            self.icron.schedule("ping", after=600, by="user", via="/cron")
            self.assertEqual(cli._reboot_refusal(self.state), (
                "icron has 1 pending job, and jobs live only in this process (/cron lists "
                "them)"))

    def test_the_reboot_state_carries_afk_its_text_and_its_rules_over(self):
        self.state.reboot = cli._Reboot("/reboot")
        self.afk.freeze()
        carried = cli._reboot_state(self.state, InteractionConfig())
        self.assertNotIn("afk", carried)
        self.assertEqual(carried["autoafk"], {"seen": NOW, "count_from": NOW, "armed": True})
        self.clock.now += 60
        self.enter("/autoafk 18:00")
        self.enter("/autonoafk 8h")
        self.turn_on("/afk Back at 13:00.")
        self.afk.freeze()
        carried = cli._reboot_state(self.state, InteractionConfig())
        until = NOW + 60 + 8 * 3600
        self.assertEqual(carried["afk"], {"since": NOW + 60, "due": NOW + 61, "until": until})
        self.assertEqual(carried["afk_text"], "Back at 13:00.")
        self.assertEqual(carried["autoafk"], {
            "seen": NOW, "count_from": NOW, "armed": True, "on_at": NOW + 60,
            "enable": [["18:00", "/autoafk"]], "disable": [["8h", "/autonoafk"]]})
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        self.assertEqual((rebooted.afk, rebooted.afk_text, rebooted.autoafk, rebooted.auto,
                          rebooted.afk_warning),
                         (carried["afk"], carried["afk_text"], carried["autoafk"], None, None))
        # The restarted CLI: the rules, and AFK with the end that its rule sets.
        self.icron.cancel(self.afk.job_id, by="user")
        state = cli._UIState(ready=True, icron=self.icron)
        state.afk = afk = Afk(self.icron, AfkDefaults(auto_enable="20m"), state.notice)
        cli._carry_over(state, InteractionConfig(), rebooted)
        self.assertEqual(self.on_rules(afk), {"daily": ("18:00", "/autoafk")})
        self.assertEqual({cls: (str(rule.when), rule.source)
                          for cls, rule in afk.rules["off"].items()},
                         {"after": ("8h", "/autonoafk")})
        self.assertEqual((afk.on_at, afk.until, afk.until_source), (NOW + 60, until, "/autonoafk"))
        self.assertTrue(_notices(state)[-1].endswith(
            "Your next message ends it, and so will 20:01 (/autonoafk)."), _notices(state))

    def test_a_state_from_the_first_version_still_reads(self):
        state = {"version": 1, "cause": "coldstart", "continues": False, "config": {},
                 "draft": {"text": "", "cursor": 0}, "queue": [], "exit_code": 0,
                 "afk": {"since": NOW, "due": NOW + 1801, "until": None}}
        rebooted = cli._parse_reboot_state(state)
        self.assertEqual((rebooted.cause, rebooted.afk, rebooted.afk_text, rebooted.autoafk),
                         ("coldstart", state["afk"], None, None))

    def test_the_shape_that_33b31af_writes_still_reads(self):
        state = {"version": 1, "cause": "/reboot", "continues": False, "config": {},
                 "draft": {"text": "", "cursor": 0}, "queue": [], "exit_code": 0,
                 "afk": {"since": NOW, "due": NOW + 1801, "until": None}, "afk_note": "Lunch.",
                 "autoafk": {"seen": NOW, "count_from": NOW, "armed": True, "when": "18:00"}}
        rebooted = cli._parse_reboot_state(state)
        self.assertEqual((rebooted.afk_text, rebooted.autoafk, rebooted.afk_warning),
                         ("Lunch.", state["autoafk"], None))
        self.afk.restore_autoafk(rebooted.autoafk)
        self.assertEqual(self.on_rules(self.afk), {"daily": ("18:00", "/autoafk")})

    def test_what_can_t_be_read_drops_only_afk(self):
        self.state.reboot = cli._Reboot("/reboot")
        self.turn_on("/afk Lunch.")
        self.afk.freeze()
        carried = json.loads(json.dumps(cli._reboot_state(self.state, InteractionConfig())))
        for bad in ([], {"since": NOW}, {"since": "noon", "due": None, "until": None},
                    {"since": True, "due": None, "until": None},
                    {"since": NOW, "due": float("nan"), "until": None},
                    {"since": NOW, "due": None, "until": None, "text": "x"}):
            with self.subTest(afk=bad):
                rebooted = cli._parse_reboot_state({**carried, "afk": bad})
                self.assertEqual((rebooted.cause, rebooted.afk, rebooted.afk_text,
                                  rebooted.afk_warning), ("/reboot", None, None, "AFK"))
                self.assertEqual(rebooted.autoafk, carried["autoafk"])
        rebooted = cli._parse_reboot_state({**carried, "afk_text": " "})
        self.assertEqual((rebooted.afk, rebooted.afk_warning), (None, "AFK"))
        base = {"seen": NOW, "count_from": NOW, "armed": True}
        for bad in ({"seen": NOW}, {**base, "armed": 1}, {**base, "when": "soon"},
                    {**base, "other": 1}, {**base, "on_at": "noon"},
                    {**base, "enable": [["20m", "/autoafk"], ["30m", "/autoafk"]]},
                    {**base, "enable": [["20m", "nowhere"]]}, {**base, "disable": [["8h"]]},
                    {**base, "disable": "8h"}):
            with self.subTest(autoafk=bad):
                rebooted = cli._parse_reboot_state({**carried, "autoafk": bad})
                self.assertEqual((rebooted.afk, rebooted.autoafk, rebooted.afk_warning),
                                 (carried["afk"], None, "AFK's rules"))
        rebooted = cli._parse_reboot_state({**carried, "afk": 1, "autoafk": 1, "auto": 1})
        self.assertEqual(rebooted.afk_warning, "AFK, AFK's rules or auto mode")
        state = cli._UIState()
        cli._carry_over(state, InteractionConfig(), rebooted)
        self.assertEqual(_notices(state), [
            "Note: the reboot didn't carry over AFK, AFK's rules or auto mode, which couldn't be "
            "read."])

    def test_the_restarted_cli_goes_on_from_the_next_reminder_with_its_text(self):
        self.afk.restore({"since": NOW - 3600, "due": NOW + 600, "until": None}, "Lunch.")
        job_id = self.afk.job_id
        self.assertEqual(_notices(self.state), [
            f"afk: still on since 11:00:00, carried over the reboot (job {job_id}); next "
            "reminder at 12:10:00. Your next message ends it."])
        job = self.job(job_id)
        self.assertEqual((job.due, job.every, job.message),
                         (NOW + 600, 1800.0, _reminder(NOW - 3600, "Lunch.")))
        self.assertEqual(self.tick(3600), [])

    def test_a_reminder_due_meanwhile_fires_at_once_and_a_passed_end_ends_it(self):
        self.afk.restore({"since": NOW - 3600, "due": NOW - 5, "until": NOW + 60})
        self.assertEqual(self.job(self.afk.job_id).due, NOW + 1)
        self.assertTrue(_notices(self.state)[0].endswith(
            "Your next message ends it, and so will 12:01 ([afk] auto_disable)."))
        afk = Afk(self.icron, AfkDefaults(), self.state.notice)
        afk.restore({"since": NOW - 3600, "due": NOW + 600, "until": NOW - 60})
        self.assertEqual(_notices(self.state), [
            "afk: off at 11:59 ([afk] auto_disable), during the reboot."])
        self.assertIsNone(afk.job_id)

    def test_the_rules_state_carries_over(self):
        # The count goes on from before the reboot: 10m without a message.
        self.afk.restore_autoafk({"seen": NOW - 600, "count_from": NOW - 600, "armed": True,
                                  "enable": [["10m", "/autoafk"]]})
        (notice,) = self.tick()
        self.assertEqual(notice, _on(self.afk.job_id, head="afk: on after 10m without a "
                                                           "message (/autoafk)"))
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(NOW - 600))
        # Disarmed, it waits for a message; carried rules replace the catalog's.
        afk = Afk(self.icron, AfkDefaults(auto_enable="10m"), self.state.notice)
        afk.restore_autoafk({"seen": NOW - 3600, "count_from": NOW - 3600, "armed": False})
        afk.tick(may_enable=True)
        self.assertIsNone(afk.job_id)
        afk.restore_autoafk({"seen": NOW, "count_from": NOW, "armed": True, "enable": []})
        self.assertEqual(afk.rules["on"], {})
        afk.restore_autoafk({"seen": NOW, "count_from": NOW, "armed": True,
                             "enable": [["2027-01-15 11:00", "/autoafk"]]})
        self.assertEqual(afk.rules["on"], {})
        self.assertEqual(_notices(self.state)[-1], "autoafk: 2027-01-15 11:00 passed during the "
                                                   "reboot, so that rule is gone.")

    def test_a_cli_without_afk_says_it_didn_t_carry_it_over(self):
        state = cli._UIState()
        cli._carry_over(state, InteractionConfig(), cli._Rebooted(
            cause="/reboot", afk={"since": NOW, "due": NOW + 1, "until": None}))
        self.assertEqual(_notices(state), [
            "Note: the reboot didn't carry over AFK: this CLI has no /afk (icron is off, or it "
            "runs headless)."])


class SessionTests(_IcronControllerCase):
    async def test_a_reminder_starts_a_turn_and_the_user_s_message_ends_afk(self):
        model = _Model(self.path, _answer("Hello."), _answer("Carrying on."),
                       _answer("Welcome back."))
        statuses = []
        terminal = _Terminal(_once([
            (_idle_with("[assistant] Hello."), lambda t: t.submit("/afk Back soon.")),
            (lambda t, s: _has(t, "afk: on, with your text;"), lambda t: self.advance(1)),
            (_idle_with("[assistant] Carrying on."),
             lambda t: (statuses.append(t.frames[-1][1]), t.submit("I'm back."))),
            (_idle_with("[assistant] Welcome back."),
             lambda t: (statuses.append(t.frames[-1][1]), _exit(t)))]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        self.assertEqual(len(model.calls), 3)
        reminder = _users(model.calls[1][0].items)[-1]
        self.assertTrue(reminder.startswith(f"{WAKE_HEADER} at {_iso(NOW + 1)}."), reminder)
        self.assertTrue(reminder.endswith(f"(the user's, from /afk; due {_iso(NOW + 1)}, every "
                                          f"1800 s):\n{_reminder(NOW, 'Back soon.')}"), reminder)
        self.assertEqual(_users(model.calls[2][0].items)[-1], "I'm back.")
        self.assertEqual(statuses[0], "idle | afk since 12:00 | icron: 1 job, next 12:30:01")
        self.assertEqual(statuses[1], "idle")
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] afk: off after 1s: you wrote to main.", texts)

    async def test_a_duration_turns_afk_on_in_a_session(self):
        model = _Model(self.path, _answer("Hello."), _answer("Carrying on."))
        terminal = _Terminal(_once([
            (_idle_with("[assistant] Hello."), lambda t: self.advance(600)),
            (lambda t, s: _has(t, "afk: on after"), lambda t: self.advance(1)),
            (_idle_with("[assistant] Carrying on."), _exit)]))
        args = cli._build_parser().parse_args(["--prompt", "hello"])
        environment = cli.Environment((self.icron.tool(),))
        await cli._run(model, environment, terminal, args, self.path, icron=self.icron,
                       afk=AfkDefaults(auto_enable="10m"))
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] afk: AFK turns on after 10m without a message from you "
                      "([afk] auto_enable); /autoafk shows or changes this.", texts)
        self.assertTrue(_users(model.calls[1][0].items)[-1].endswith(_reminder(NOW)))


class RebootSessionTests(_RebootCase):
    async def test_afk_lets_a_coldstart_compaction_reboot_and_goes_with_it(self):
        self._resumable()
        clock = _Clock()
        icron = Icron(clock=clock)
        self.addCleanup(icron.close, 1.0)

        def remind(terminal):
            clock.now += 1
            icron._poke()

        model = _Model(self.path)
        terminal = _Terminal(_steps(
            (lambda t, s: s == "idle", lambda t: t.submit("/afk Lunch.")),
            (lambda t, s: _has(t, "afk: on, with your text;"), remind)))
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()):
            carried = await self._rebooted(model, terminal, [
                "--resume", "--auto-compaction-context-tokens", "100",
                "--compaction-coldstart-reboot"], icron=icron)
        self.assertEqual(carried["afk"], {"since": NOW, "due": NOW + 1801, "until": None})
        self.assertEqual(carried["afk_text"], "Lunch.")
        self.assertEqual(set(carried["autoafk"]), {"seen", "count_from", "armed", "on_at"})
        texts = [item.text for item in terminal.items]
        self.assertFalse(any("not rebooting" in text or "icron: dropped" in text
                             for text in texts), texts)


if __name__ == "__main__":
    unittest.main()
