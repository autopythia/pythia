"""/auto and /noauto: auto mode, with its own AFK (doc/cli-auto-plan.md)."""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import unittest

from pythia.interaction import Icron, InteractionConfig, Message, cli
from pythia.interaction._cli_afk import AUTONOAFK_USAGE, Afk
from pythia.interaction._cli_auto import Auto
from pythia.interaction.model_catalog import AfkDefaults
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_cli_afk import _AfkCase, _on, _reminder
from pythia_test.test_interaction_cli_icron import NOW, _Clock, _IcronControllerCase, _enter
from pythia_test.test_interaction_cli_icron import _exit, _has, _idle_with, _notices, _once
from pythia_test.test_interaction_cli_icron import _users

_AUTO_ON = ("auto: on. Until /noauto, AFK turns on after 30m without a message from you, and "
            "only your messages turn it off.")
_AUTO_ON_APPROVAL = ("auto: on. Until /noauto, AFK turns on after 30m without a message from "
                     "you, only your messages turn it off, and tool calls run without approval.")
_CONFIRM = ("auto: auto mode runs tool calls without approval, until /noauto. /yes enters it, "
            "and /no doesn't.")


def _auto(state, afk):
    return Auto(afk, state.notice, approval=lambda: state.require_tool_approval,
                set_approval=functools.partial(setattr, state, "require_tool_approval"))


def _rules(afk, kind):
    return {cls: (str(rule.when), rule.source) for cls, rule in afk.rules[kind].items()}


class _AutoCase(_AfkCase):
    def setUp(self):
        super().setUp()
        self.state.auto = self.auto = _auto(self.state, self.afk)


class AutoTests(_AutoCase):
    def test_auto_sends_the_text_as_typed_text_would_then_turns_auto_mode_and_afk_on(self):
        typed = cli._UIState(ready=True)  # no AFK: plain typed text
        _enter(typed, "Write the tests.")
        (expected,) = typed.pending
        self.assertEqual(self.enter("/auto Write the tests."), [_AUTO_ON, _on(self.afk.job_id)])
        (submission,) = self.state.pending
        self.assertEqual((submission.text, submission.message, submission.steers),
                         (expected.text, expected.message, expected.steers))
        self.assertEqual(self.state.editor.text, "")
        self.assertTrue(self.auto.on)
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(NOW))  # no /afk text
        self.assertEqual((_rules(self.afk, "on"), _rules(self.afk, "off")),
                         ({"after": ("30m", "/auto")}, {}))

    def test_auto_alone_turns_auto_mode_and_afk_on_without_a_message(self):
        self.assertEqual(self.enter("/auto"), [_AUTO_ON, _on(self.afk.job_id)])
        self.assertEqual(list(self.state.pending), [])

    def test_auto_is_idempotent(self):
        self.enter("/auto First.")
        saved = self.auto.saved
        self.enter("/autoafk 45m")
        self.clock.now += 60
        self.assertEqual(self.enter("/auto Second."), [_on(self.afk.job_id, head="afk: on again")])
        self.assertIs(self.auto.saved, saved)
        self.assertEqual(_rules(self.afk, "on"), {"after": ("45m", "/autoafk")})
        self.assertEqual([entry.text for entry in self.state.pending], ["First.", "Second."])

    def test_the_loop_afk_on_at_once_off_at_a_message_and_on_again_after_30m(self):
        self.enter("/auto Build it.")
        self.assertTrue(self.afk.sync())
        self.clock.now += 600
        self.assertEqual(self.enter("Status?"), ["afk: off after 10m: you wrote to main."])
        self.assertTrue(self.auto.on)
        self.assertEqual(self.tick(1799), [])
        (notice,) = self.tick(1)
        self.assertEqual(notice, _on(self.afk.job_id, head="afk: on after 30m without a message "
                                                           "(/auto)"))
        self.assertEqual(self.job(self.afk.job_id).message, _reminder(NOW + 600))

    def test_not_without_icron(self):
        state = cli._UIState(ready=True, icron=self.icron)  # no AFK: a headless state
        for text in ("/auto", "/auto Build it.", "/noauto"):
            _enter(state, text)
            self.assertTrue(_notices(state)[0].startswith("Unsupported command. Use /retry"))


class ApprovalTests(_AutoCase):
    def test_while_tool_calls_need_approval_auto_waits_for_yes(self):
        self.state.require_tool_approval = "exec"
        self.assertEqual(self.enter("/auto Build it."), [_CONFIRM])
        self.assertEqual((self.state.pending_auto, self.state.editor.text),
                         ("/auto Build it.", ""))
        self.assertEqual(self.enter("Something else."), [
            "auto: /yes enters auto mode, and /no doesn't; answer first. Draft preserved."])
        self.assertEqual(self.enter("/no"), [
            "auto: not entered; your /auto is back in the input line."])
        self.assertEqual((self.state.pending_auto, self.state.editor.text, self.auto.on),
                         (None, "/auto Build it.", False))
        self.assertEqual(list(self.state.pending), [])
        self.assertEqual(self.enter("/auto Build it."), [_CONFIRM])
        self.assertEqual(self.enter("/yes"), [_AUTO_ON_APPROVAL, _on(self.afk.job_id)])
        self.assertEqual((self.state.require_tool_approval, self.state.editor.text),
                         (False, ""))
        self.assertEqual([entry.text for entry in self.state.pending], ["Build it."])
        # In auto mode, /auto needs no /yes; /noauto turns approval on again.
        self.assertEqual(self.enter("/auto"), [_on(self.afk.job_id, head="afk: on again")])
        self.clock.now += 3600
        self.assertEqual(self.enter("/noauto"), [
            "auto: off after 1h. AFK is as it was at /auto: off. Tool calls need approval again "
            "(exec)."])
        self.assertEqual(self.state.require_tool_approval, "exec")

    def test_without_approval_auto_goes_at_once_and_noauto_says_nothing_of_it(self):
        self.assertEqual(self.enter("/auto"), [_AUTO_ON, _on(self.afk.job_id)])
        self.assertEqual(self.enter("/noauto"), [
            "auto: off after 0s. AFK is as it was at /auto: off."])
        self.assertEqual(self.state.require_tool_approval, False)


class ProfileTests(_AutoCase):
    defaults = AfkDefaults(auto_enable="12:30", auto_disable="13:00")

    def test_auto_mode_s_rules_change_but_its_duration_doesn_t_go(self):
        self.enter("/auto")
        self.assertEqual((_rules(self.afk, "on"), _rules(self.afk, "off")),
                         ({"after": ("30m", "/auto")}, {}))
        self.assertEqual(self.enter("/autoafk"), [
            "autoafk: in auto mode, until /noauto, AFK turns on after 30m without a message from "
            "you, once per absence (/auto). It turns off at your next message. /autoafk <rule> "
            "sets a rule, and /noautoafk <rule> removes one."])
        self.assertEqual(self.enter("/autonoafk"), [
            "autonoafk: in auto mode, until /noauto, AFK turns on after 30m without a message "
            "from you, once per absence (/auto). It turns off at your next message. "
            + AUTONOAFK_USAGE])
        for text, refusal in (
                ("/noautoafk 30m", "autoafk: in auto mode, AFK's duration stays; /autoafk "
                                   "<duration> changes it."),
                ("/noautoafk", "autoafk: in auto mode, rules go one at a time, and AFK's "
                               "duration stays; /noautoafk <rule> removes one."),
                ("/autonoafk 8h", "autonoafk: in auto mode, only your messages turn AFK off; "
                                  "/noauto ends auto mode."),
                ("/noautonoafk", "autonoafk: in auto mode, only your messages turn AFK off; "
                                 "/noauto ends auto mode.")):
            with self.subTest(text=text):
                self.assertEqual(self.enter(text), [f"{refusal} Draft preserved."])
        self.assertEqual(self.enter("/autoafk 45m"), [
            "autoafk: AFK turns on after 45m without a message from you, once per absence, "
            "instead of after 30m without a message from you. In auto mode it stays; /autoafk "
            "<duration> changes it."])
        self.assertEqual(self.enter("/autoafk 18:00"), [
            "autoafk: AFK also turns on at 18:00 each day, next today. /noautoafk 18:00 removes "
            "this."])
        self.assertEqual(self.enter("/noautoafk 18:00"), [
            "autoafk: removed at 18:00 each day; AFK still turns on after 45m without a message "
            "from you (/autoafk)."])
        # Nothing but a message ends AFK: not the catalog's 13:00.
        self.assertEqual(self.at(13, 0), [])
        self.assertTrue(self.afk.sync())


class NoautoTests(_AutoCase):
    defaults = AfkDefaults(auto_enable="18:00", auto_disable="08:30")

    def test_noauto_puts_the_rules_back_and_a_duration_counts_from_it(self):
        self.enter("/autoafk 20m")
        self.enter("/auto Build it.")
        self.enter("/autoafk 19:00")  # auto mode's, until /noauto
        self.clock.now += 3600
        self.assertEqual(self.enter("/noauto"), [
            "auto: off after 1h. AFK is as it was at /auto: off, and it turns on after 20m "
            "without a message from you (/autoafk), or at 18:00 each day ([afk] auto_enable)."])
        self.assertEqual(_rules(self.afk, "on"), {"after": ("20m", "/autoafk"),
                                                  "daily": ("18:00", "[afk] auto_enable")})
        self.assertEqual(_rules(self.afk, "off"), {"daily": ("08:30", "[afk] auto_disable")})
        self.assertIsNone(self.afk.job_id)
        self.assertEqual(self.tick(1199), [])  # 20m from /noauto
        self.assertTrue(self.tick(1)[0].startswith("afk: on after 20m without a message "
                                                   "(/autoafk); "))
        self.assertEqual(self.enter("/noauto"), ["auto: already off."])
        self.assertEqual(self.enter("/noauto now"), [
            "Usage: /noauto (no arguments) turns auto mode off, and puts AFK and tool-call "
            "approval back as they were at /auto. Draft preserved."])


class SnapshotTests(_AutoCase):
    def test_afk_that_was_on_at_auto_is_on_again_though_auto_sent_a_message(self):
        self.turn_on("/afk Lunch.")  # on since 12:00, with its text
        self.clock.now += 600
        self.assertEqual(self.enter("/auto Next, the docs."),
                         [_AUTO_ON, _on(self.afk.job_id, head="afk: on again")])
        self.clock.now += 600
        notices = self.enter("/noauto")
        job_id = self.afk.job_id
        self.assertEqual(notices, [
            f"auto: off after 10m. AFK is as it was at /auto: on since 12:00:00 (job {job_id})."])
        self.assertEqual(self.job(job_id).message, _reminder(NOW, "Lunch."))
        self.assertEqual((self.afk.since, self.afk.text), (NOW, "Lunch."))

    def test_afk_whose_end_passed_in_auto_mode_is_off_after_noauto(self):
        self.state.afk = self.afk = Afk(self.icron, AfkDefaults(auto_disable="12:30"),
                                        self.state.notice)
        self.state.auto = self.auto = _auto(self.state, self.afk)
        self.turn_on()
        self.enter("/auto")
        self.assertEqual(self.at(13, 0), [])  # in auto mode, 12:30 doesn't end it
        self.assertEqual(self.enter("/noauto"), [
            "auto: off after 1h. AFK is as it was at /auto: off, since its end at 12:30 passed "
            "in auto mode."])
        self.assertIsNone(self.afk.job_id)


class RebootTests(_AutoCase):
    def test_a_reboot_carries_auto_mode_and_what_it_saved(self):
        self.state.require_tool_approval = "exec"
        self.enter("/auto Build it.")
        self.enter("/yes")
        self.enter("/autoafk 18:00")
        self.state.reboot = cli._Reboot("/reboot")
        self.afk.freeze()
        carried = json.loads(json.dumps(cli._reboot_state(self.state, InteractionConfig())))
        self.assertEqual(carried["auto"]["saved"]["approval"], "exec")
        self.assertEqual((carried["autoafk"]["enable"], carried["autoafk"]["disable"]),
                         ([["30m", "/auto"], ["18:00", "/autoafk"]], []))
        rebooted = cli._parse_reboot_state(carried)
        self.assertEqual((rebooted.auto, rebooted.afk_warning), (carried["auto"], None))
        # The restarted CLI: approval on, from its command line, until auto mode is back.
        self.icron.cancel(self.afk.job_id, by="user")
        state = cli._UIState(ready=True, icron=self.icron, require_tool_approval="exec")
        state.afk = afk = Afk(self.icron, AfkDefaults(auto_disable="13:00"), state.notice)
        state.auto = auto = _auto(state, afk)
        cli._carry_over(state, InteractionConfig(), rebooted)
        self.assertTrue(auto.on and afk.in_auto)
        self.assertEqual(state.require_tool_approval, False)
        self.assertEqual((_rules(afk, "on"), _rules(afk, "off")),
                         ({"after": ("30m", "/auto"), "daily": ("18:00", "/autoafk")}, {}))
        self.assertIn("auto: still on, carried over the reboot; /noauto ends it.",
                      _notices(state))
        self.assertTrue(auto.noauto_command("/noauto"))
        self.assertEqual(state.require_tool_approval, "exec")
        self.assertEqual(_rules(afk, "off"), {"daily": ("13:00", "[afk] auto_disable")})

    def test_without_afk_s_rules_auto_mode_gets_its_profile_again(self):
        self.enter("/auto")
        saved = self.auto.freeze()
        afk = Afk(self.icron, AfkDefaults(auto_enable="18:00"), self.state.notice)
        auto = _auto(self.state, afk)
        auto.restore(json.loads(json.dumps(saved)), rules=False)
        self.assertEqual((_rules(afk, "on"), afk.in_auto), ({"after": ("30m", "/auto")}, True))


class AttachmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.icron = Icron(clock=_Clock())
        self.icron.start(lambda: None, lambda text: None)
        self.addCleanup(self.icron.close, 1.0)
        self.state = cli._UIState(ready=True, icron=self.icron)
        self.state.afk = self.afk = Afk(self.icron, AfkDefaults(), self.state.notice)
        self.state.auto = self.auto = _auto(self.state, self.afk)
        self.state.reads_files = lambda text: text.startswith("@")

    async def read(self):
        for _ in range(300):
            if self.state.reading is None:
                return
            await asyncio.sleep(0.01)
        self.fail("the attachments weren't read")

    async def test_auto_mode_and_afk_wait_for_the_message_s_attachments(self):
        release = threading.Event()

        def read(text):
            release.wait(3)
            return Message("user", f"[read] {text}")

        self.state.read_query = read
        _enter(self.state, "/auto @spec.md Build it.")
        self.assertEqual(self.state.reading, "/auto @spec.md Build it.")
        self.assertEqual((self.auto.on, self.afk.job_id, _notices(self.state)), (False, None, []))
        release.set()
        await self.read()
        (submission,) = self.state.pending
        self.assertEqual(submission.message.content_text, "[read] @spec.md Build it.")
        self.assertTrue(self.auto.on and self.afk.sync())
        self.assertEqual(self.state.editor.text, "")

    async def test_a_message_that_can_t_be_read_leaves_auto_mode_off_and_the_draft(self):
        def read(text):
            raise cli.AttachmentError("Attachment @spec.md: no such file.")

        self.state.read_query = read
        _enter(self.state, "/auto @spec.md Build it.")
        await self.read()
        self.assertEqual(_notices(self.state), ["Attachment @spec.md: no such file."])
        self.assertEqual((self.auto.on, self.afk.job_id, list(self.state.pending)),
                         (False, None, []))
        self.assertEqual(self.state.editor.text, "/auto @spec.md Build it.")


class SessionTests(_IcronControllerCase):
    async def test_auto_at_a_context_s_start_sends_the_task_then_a_reminder(self):
        model = _Model(self.path, _answer("On it."), _answer("Carrying on."))
        statuses = []
        terminal = _Terminal(_once([
            (_idle_with("icron: main can"), lambda t: t.submit("/auto Write the tests.")),
            (lambda t, s: _has(t, "auto: on."), lambda t: self.advance(1)),
            (_idle_with("[assistant] Carrying on."),
             lambda t: (statuses.append(t.frames[-1][1]), _exit(t)))]))
        self.assertEqual(await self._run(model, terminal), 0)
        self.assertEqual(_users(model.calls[0][0].items), ["Write the tests."])
        reminder = _users(model.calls[1][0].items)[-1]
        self.assertTrue(reminder.endswith(_reminder(NOW)), reminder)
        self.assertEqual(statuses[0], "idle | auto | afk since 12:00 | icron: 1 job, next "
                                      "12:30:01")


if __name__ == "__main__":
    unittest.main()
