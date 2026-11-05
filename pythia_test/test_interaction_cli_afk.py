"""/afk: the CLI tells main that the user is away (doc/cli-afk-plan.md)."""

from __future__ import annotations

from datetime import datetime
import json
import re
import unittest
from unittest import mock
import warnings

from pythia.interaction import Icron, IcronError, InteractionConfig, cli
from pythia.interaction import _cli_afk
from pythia.interaction._cli_afk import Afk
from pythia.interaction.loop.icron import WAKE_HEADER
from pythia.interaction.loop.supervision import Yield
from pythia.interaction.model_catalog import AfkDefaults
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


def _reminder(since):
    return (f"The user has been away from the keyboard (AFK) since {_iso(since)}; only they "
            "can end these reminders. They can't answer until they're back, so don't wait for "
            "them: carry on where you can.")


def _on(job_id, tail=""):
    return (f"afk: on; main hears now, then every 30m, that you're away (job {job_id}). "
            f"/afk off ends it{tail}.")


class _AfkCase(unittest.TestCase):
    """A CLI state with icron and /afk, on fake clocks: icron's (wall time)
    and the idle time's (_cli_afk._idle_clock)."""

    defaults = AfkDefaults()

    def setUp(self):
        self.clock = _Clock()
        self.idle = _Clock(100.0)
        patcher = mock.patch.object(_cli_afk, "_idle_clock", self.idle)
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

    def turn_on(self):
        (notice,) = self.enter("/afk")
        return re.search(r"\(job (\S+)\)", notice).group(1)

    def tick(self, seconds=1.0, *, may_enable=True):
        """Let the idle time go on (a tick checks at most once a second), then tick."""
        self.idle.now += seconds
        self.afk.tick(may_enable=may_enable)
        return _notices(self.state)


class CommandTests(_AfkCase):
    def test_afk_turns_on_shows_its_state_and_afk_off_turns_it_off(self):
        (notice,) = self.enter("/afk")
        job_id = re.fullmatch(r"afk: on; main hears now, then every 30m, that you're away "
                              r"\(job (\S+)\)\. /afk off ends it\.", notice).group(1)
        self.assertEqual(self.state.editor.text, "")
        job = self.job(job_id)
        self.assertEqual((job.origin, job.kind, job.via, job.every, job.due, job.message),
                         ("user", "message", "/afk", 1800.0, NOW + 1, _reminder(NOW)))
        self.assertEqual(self.afk.status(), "afk since 12:00")
        state = (f"afk: on since 12:00:00 (job {job_id}); next reminder at 12:00:01. /afk off "
                 "ends it.")
        self.assertEqual(self.enter("/afk"), [state])
        self.assertEqual(self.enter("/afk on"), [state])  # the same as /afk
        self.clock.now += 5400
        self.assertEqual(self.enter("/afk off"), ["afk: off after 1h 30m."])
        self.assertEqual(self.job(job_id).state, "cancelled")
        self.assertEqual(self.afk.status(), "")
        self.assertEqual(self.enter("/afk off"), ["afk: already off."])
        self.assertEqual(self.enter("/afk on"), [_on(self.afk.job_id)])

    def test_a_malformed_command_or_a_full_job_list_keeps_the_draft(self):
        usage = ("Usage: /afk or /afk on tells main, now and every 30m, that you're away, "
                 "until /afk off; while it's on, /afk shows it. Draft preserved.")
        for text in ("/afk now", "/afk off now", "/afk of"):
            with self.subTest(text=text):
                self.assertEqual(self.enter(text), [usage])
                self.assertEqual(self.state.editor.text, text)
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
        state = cli._UIState(ready=True, icron=self.icron)  # no /afk: a headless state
        _enter(state, "/afk")
        self.assertTrue(_notices(state)[0].startswith("Unsupported command. Use /retry"))

    def test_cron_lists_its_job_and_cancels_it_and_main_can_t(self):
        job_id = self.turn_on()
        (listing,) = self.enter("/cron")
        self.assertTrue(listing.startswith(
            f"cron: 1 pending job (now 12:00:00):\n  {job_id} (yours, /afk) 12:00:01, in 1s, "
            "every 30m: The user has been away"), listing)
        with self.assertRaisesRegex(IcronError, "you can cancel only your own jobs"):
            self.icron.cancel(job_id, by="main")
        cron = self.icron.schedule("ping", after=600, by="user", via="/cron")
        self.state.cron_entries[cron.job_id] = "in 10m ping"
        self.assertEqual(self.enter("/crontab clear"), ["crontab: cancelled 1 job."])
        self.assertTrue(self.afk.sync())  # /crontab clear cancels /cron's jobs only
        self.assertEqual(self.enter(f"/cron cancel {job_id}"), [
            f"cron: cancelled job {job_id} (yours).", f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.afk.status(), "")
        job_id = self.turn_on()
        self.assertEqual(self.enter("/cron cancel all"), [
            "cron: cancelled 1 job.", f"afk: off; its job {job_id} was cancelled."])

    def test_a_job_that_ends_another_way_ends_afk_at_the_next_tick(self):
        job_id = self.turn_on()
        self.icron.cancel(job_id, by="user")  # as /py code might
        self.assertEqual(self.afk.status(), "afk since 12:00")  # until the tick
        self.assertEqual(self.tick(), [f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.afk.status(), "")


class ReminderTests(_AfkCase):
    def test_the_first_reminder_fires_at_once_and_the_next_after_the_turn_that_took_it(self):
        job_id = self.turn_on()
        self.clock.now += 1
        self.icron._poke()
        self.assertTrue(_until(lambda: self.icron.undelivered() == 1))
        self.icron.begin_turn()
        wakeup = self.icron.take_wakes()
        self.assertEqual(wakeup.text(), "\n".join([
            f"{WAKE_HEADER} at {_iso(NOW + 1)}.", "",
            f"Job {job_id} (the user's, from /afk; due {_iso(NOW + 1)}, every 1800 s):",
            _reminder(NOW)]))
        # The turn runs past the next slot, which merges into the run that it took.
        self.clock.now += 1800
        self.icron._poke()
        self.assertFalse(_until(lambda: self.icron.undelivered(), timeout=0.2))
        self.icron.handoff(Yield(context=1, job_id=None, job_text="task", kind="ended",
                                 resumable=True, revision=0, final_text="Carrying on."))
        self.assertTrue(_until(lambda: self.job(job_id).state == "scheduled"))
        self.assertEqual(self.job(job_id).due, NOW + 3601)
        self.clock.now += 1800
        self.icron._poke()
        self.assertTrue(_until(lambda: self.icron.undelivered() == 1))
        self.assertTrue(self.afk.sync())


class MessageTests(_AfkCase):
    def test_a_typed_message_for_main_ends_afk(self):
        for text in ("I'm back.", "/steer I'm back.", "/task Next, the docs."):
            with self.subTest(text=text):
                self.turn_on()
                self.clock.now += 60
                notices = self.enter(text)
                self.assertIn("afk: off after 1m: you wrote to main "
                              "([afk] auto_disable_on_message).", notices)
                self.assertIsNone(self.afk.job_id)

    def test_keys_commands_and_wake_ups_leave_it_on(self):
        job_id = self.turn_on()
        self.state.editor = self.state.editor.edit("<bracketed-paste>", "half a thought")
        for text in ("/cron", "/crontab", "/afk", "/cron in 1m ping"):
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


class KeepOnMessageTests(_AfkCase):
    defaults = AfkDefaults(auto_disable_on_message=False)

    def test_without_auto_disable_a_message_says_once_that_it_is_still_on(self):
        job_id = self.turn_on()
        self.assertIn("afk: still on; /afk off ends it.", self.enter("I'm back."))
        self.assertNotIn("afk: still on; /afk off ends it.", self.enter("Still here."))
        self.assertEqual(self.afk.job_id, job_id)
        self.enter("/afk off")
        self.turn_on()  # a new absence says it again
        self.assertIn("afk: still on; /afk off ends it.", self.enter("Back again."))


class DisableAtTests(_AfkCase):
    defaults = AfkDefaults(auto_disable_at="12:30", auto_enable_seconds=600)

    def test_afk_ends_at_the_time_and_doesn_t_come_back_without_a_key_press(self):
        job_id = self.turn_on()
        tail = ", and so will 12:30 ([afk] auto_disable_at)"
        self.assertEqual(self.enter("/afk"), [
            f"afk: on since 12:00:00 (job {job_id}); next reminder at 12:00:01. /afk off ends "
            f"it{tail}."])
        self.clock.now = NOW + 1799
        self.assertEqual(self.tick(), [])
        self.clock.now = NOW + 1800
        self.assertEqual(self.tick(), ["afk: off at 12:30 ([afk] auto_disable_at), after 30m."])
        self.assertEqual(self.job(job_id).state, "cancelled")
        self.assertEqual(self.tick(3600), [])  # away for long, but no key since: it stays off
        self.afk.key()
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 10m without input"))

    def test_a_time_that_has_passed_today_is_tomorrow_s(self):
        afk = Afk(self.icron, AfkDefaults(auto_disable_at="9:00"), self.state.notice)
        afk.enable()
        (notice,) = _notices(self.state)
        self.assertEqual(notice, _on(afk.job_id, ", and so will 09:00 tomorrow "
                                                 "([afk] auto_disable_at)"))
        self.assertEqual(afk.until, datetime(2027, 1, 16, 9, 0).timestamp())


class AutoEnableTests(_AfkCase):
    defaults = AfkDefaults(auto_enable_seconds=600)

    def test_it_turns_on_after_the_idle_time_once_per_absence(self):
        self.assertEqual(self.afk.startup_notice(), "afk: turns on after 10m without a key "
                                                    "press ([afk] auto_enable_seconds).")
        self.assertEqual(self.tick(599), [])
        self.assertEqual(self.tick(1, may_enable=False), [])  # not while it may not
        (notice,) = self.tick(1)
        job_id = self.afk.job_id
        self.assertEqual(notice, (
            "afk: on after 10m 1s without input ([afk] auto_enable_seconds); main hears now, "
            f"then every 30m, that you're away (job {job_id}). /afk off ends it."))
        # Since the last key press, on icron's clock.
        self.assertEqual(self.job(job_id).message, _reminder(NOW - 601))
        self.icron.cancel(job_id, by="user")
        self.assertEqual(self.tick(), [f"afk: off; its job {job_id} was cancelled."])
        self.assertEqual(self.tick(3600), [])  # the same absence: not again
        self.enter("/afk off")  # a key press, so a new absence counts from here
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 10m without input"))
        # Turning it off yourself is coming back: when you leave again, it may fire.
        self.assertEqual(self.enter("/afk off"), ["afk: off after 10m."])  # since you left
        self.assertEqual(self.tick(599), [])
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 10m without input"))
        # Keys while it's on don't start a new absence: the end at a job's
        # cancel by code leaves it off, however long you stay away.
        self.enter("/cron")
        self.icron.cancel(self.afk.job_id, by="user")
        self.tick()
        self.assertEqual(self.tick(3600), [])

    def test_a_failure_to_turn_on_says_why_once(self):
        for _ in range(16):
            self.icron.schedule("x", after=600, by="user", via="/cron")
        self.assertEqual(self.tick(600), [
            "afk: not turned on: icron.schedule(): the user has 16 pending jobs already; "
            "cancel one first."])
        self.assertEqual(self.tick(600), [])


class CatalogTests(unittest.TestCase):
    def parse(self, text):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no catalog version
            return parse_model_catalog(text)

    def test_the_afk_section(self):
        catalog = self.parse("[afk]\ninterval_seconds = 900\nauto_enable_seconds = 1200\n"
                             "auto_disable_on_message = false\nauto_disable_at = 8:30\n"
                             'message = "Keep to the plan."\n')
        self.assertEqual(catalog.afk, AfkDefaults(900, 1200, False, "08:30", "Keep to the plan."))
        self.assertIn('[afk] interval_seconds = 900, auto_enable_seconds = 1200, '
                      'auto_disable_on_message = false, auto_disable_at = 08:30, '
                      'message = "Keep to the plan."', render_model_catalog(catalog))
        self.assertEqual(self.parse("[afk]\nmessage = Keep going.\n").afk.message, "Keep going.")
        unset = self.parse("[afk]\ninterval_seconds = null\n")
        self.assertEqual(unset.afk, AfkDefaults())
        self.assertNotIn("[afk]", render_model_catalog(unset))
        self.assertEqual(self.parse("[afk]\nauto_disable_on_message = yes\n").afk,
                         AfkDefaults(auto_disable_on_message=True))

    def test_bad_keys_and_values_name_what_was_expected(self):
        cases = [
            ("when = 1", r"\[afk\] key 'when' in user; expected interval_seconds, "
                         r"auto_enable_seconds, auto_disable_on_message, auto_disable_at, "
                         r"or message"),
            ("interval_seconds = 29", r"interval_seconds in user: expected a whole number of "
                                      r"seconds from 30 to 604800, or null"),
            ("interval_seconds = 1800.0", r"interval_seconds in user: expected a whole number"),
            ("auto_enable_seconds = 59", r"auto_enable_seconds in user: expected a whole number "
                                         r"of seconds from 60 to 604800, or null"),
            ("auto_disable_on_message = maybe", r"auto_disable_on_message in user: expected "
                                                r"true, false, or null"),
            ("auto_disable_at = 24:00", r"auto_disable_at in user: expected a local time of "
                                        r"day as HH:MM, such as 08:30, or null"),
            ("auto_disable_at = 8.30", r"auto_disable_at in user: expected a local time"),
            ('message = "unclosed', r"message in user: expected nonblank text of at most 4000 "
                                    r"characters, or null"),
            ("message = " + "x" * 4001, r"message in user: expected nonblank text"),
        ]
        for line, error in cases:
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, error):
                self.parse(f"[afk]\n{line}\n")
        with self.assertRaisesRegex(ValueError, "afk interval_seconds: expected"):
            AfkDefaults(interval_seconds=True)


class RebootTests(_AfkCase):
    def test_afk_s_job_doesn_t_refuse_a_reboot_but_others_still_do(self):
        self.turn_on()
        with _rebootable():
            self.assertIsNone(cli._reboot_refusal(self.state))
            self.icron.schedule("ping", after=600, by="user", via="/cron")
            self.assertEqual(cli._reboot_refusal(self.state), (
                "icron has 1 pending job, and jobs live only in this process (/cron lists "
                "them)"))

    def test_the_reboot_state_carries_afk_over(self):
        self.state.reboot = cli._Reboot("/reboot")
        self.afk.freeze()
        self.assertNotIn("afk", cli._reboot_state(self.state, InteractionConfig()))
        self.turn_on()
        self.afk.freeze()
        carried = cli._reboot_state(self.state, InteractionConfig())
        self.assertEqual(carried["afk"], {"since": NOW, "due": NOW + 1, "until": None})
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        self.assertEqual(rebooted.afk, carried["afk"])
        self.assertIsNone(cli._parse_reboot_state({**carried, "afk": None}).afk)
        for bad in ([], {"since": NOW}, {"since": "noon", "due": None, "until": None},
                    {"since": True, "due": None, "until": None},
                    {"since": NOW, "due": float("nan"), "until": None}):
            with self.subTest(afk=bad), self.assertRaises(ValueError):
                cli._parse_reboot_state({**carried, "afk": bad})

    def test_the_restarted_cli_goes_on_from_the_next_reminder(self):
        self.afk.restore({"since": NOW - 3600, "due": NOW + 600, "until": None})
        job_id = self.afk.job_id
        self.assertEqual(_notices(self.state), [
            f"afk: still on since 11:00:00, carried over the reboot (job {job_id}); next "
            "reminder at 12:10:00. /afk off ends it."])
        job = self.job(job_id)
        self.assertEqual((job.due, job.every, job.message), (NOW + 600, 1800.0,
                                                             _reminder(NOW - 3600)))
        self.assertEqual(self.tick(3600), [])  # restored, it doesn't turn on again by itself

    def test_a_reminder_due_meanwhile_fires_at_once_and_a_passed_end_ends_it(self):
        self.afk.restore({"since": NOW - 3600, "due": NOW - 5, "until": NOW + 60})
        self.assertEqual(self.job(self.afk.job_id).due, NOW + 1)
        self.assertTrue(_notices(self.state)[0].endswith(
            "/afk off ends it, and so will 12:01 ([afk] auto_disable_at)."))
        afk = Afk(self.icron, AfkDefaults(), self.state.notice)
        afk.restore({"since": NOW - 3600, "due": NOW + 600, "until": NOW - 60})
        self.assertEqual(_notices(self.state), [
            "afk: off at 11:59 ([afk] auto_disable_at), during the reboot."])
        self.assertIsNone(afk.job_id)

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
            (_idle_with("[assistant] Hello."), lambda t: t.submit("/afk")),
            (lambda t, s: _has(t, "afk: on;"), lambda t: self.advance(1)),
            (_idle_with("[assistant] Carrying on."),
             lambda t: (statuses.append(t.frames[-1][1]), t.submit("I'm back."))),
            (_idle_with("[assistant] Welcome back."),
             lambda t: (statuses.append(t.frames[-1][1]), _exit(t)))]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        self.assertEqual(len(model.calls), 3)
        reminder = _users(model.calls[1][0].items)[-1]
        self.assertTrue(reminder.startswith(f"{WAKE_HEADER} at {_iso(NOW + 1)}."), reminder)
        self.assertTrue(reminder.endswith(f"(the user's, from /afk; due {_iso(NOW + 1)}, every "
                                          f"1800 s):\n{_reminder(NOW)}"), reminder)
        self.assertEqual(_users(model.calls[2][0].items)[-1], "I'm back.")
        self.assertEqual(statuses[0], "idle | afk since 12:00 | icron: 1 job, next 12:30:01")
        self.assertEqual(statuses[1], "idle")
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] afk: off after 1s: you wrote to main "
                      "([afk] auto_disable_on_message).", texts)

    async def test_the_idle_time_turns_afk_on_in_a_session(self):
        idle = _Clock(100.0)
        model = _Model(self.path, _answer("Hello."), _answer("Carrying on."))
        terminal = _Terminal(_once([
            (_idle_with("[assistant] Hello."), lambda t: setattr(idle, "now", idle.now + 600)),
            (lambda t, s: _has(t, "afk: on after"), lambda t: self.advance(1)),
            (_idle_with("[assistant] Carrying on."), _exit)]))
        args = cli._build_parser().parse_args(["--prompt", "hello"])
        environment = cli.Environment((self.icron.tool(),))
        with mock.patch.object(_cli_afk, "_idle_clock", idle):
            await cli._run(model, environment, terminal, args, self.path, icron=self.icron,
                           afk=AfkDefaults(auto_enable_seconds=600))
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] afk: turns on after 10m without a key press "
                      "([afk] auto_enable_seconds).", texts)
        self.assertTrue(_users(model.calls[1][0].items)[-1].endswith(_reminder(NOW - 600)))


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
            (lambda t, s: s == "idle", lambda t: t.submit("/afk")),
            (lambda t, s: _has(t, "afk: on;"), remind)))
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()):
            carried = await self._rebooted(model, terminal, [
                "--resume", "--auto-compaction-context-tokens", "100",
                "--compaction-coldstart-reboot"], icron=icron)
        self.assertEqual(carried["afk"], {"since": NOW, "due": NOW + 1801, "until": None})
        texts = [item.text for item in terminal.items]
        self.assertFalse(any("not rebooting" in text or "icron: dropped" in text
                             for text in texts), texts)


if __name__ == "__main__":
    unittest.main()
