"""The experimental ui tool (ui_tool.py): the UI plugin, loaded at launch, and
screenshot(), through the media tool's outbox. The plugin is a fake library of
ctypes callbacks (test_plugin.FakeLibrary) that answers as the crate does."""

from __future__ import annotations

import asyncio
import base64
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from pythia.interaction import Environment, Message, ModelSample, ToolCall
from pythia.interaction import UserInteractionBoundary, cli, load_interaction_save
from pythia.interaction import plugin, ui_tool
from pythia.interaction.display import render_interaction_items
from pythia.interaction.loop import Interrupt, Steer
from pythia.interaction.media_tool import MEDIA_HEADER, ImageCap, MediaTool, Outbox
from pythia.interaction.ui_tool import UI_HEADER, UiPluginError, UiTool, call_text
from pythia.interaction.ui_tool import load_ui_plugin
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_media import _PNG, _write_png
from pythia_test.test_plugin import FakeLibrary, json_answer


_OPTION = "--enable-experimental-ui-plugin"
_INSTALL = "python3 install.py -e --enable-experimental-ui-plugin"
_ALLOW = ("macOS doesn't allow screen recording here. Allow screen recording for the app "
          "that runs this, usually the terminal, in System Settings > Privacy & Security > "
          "Screen Recording (Screen & System Audio Recording on macOS 15 and later), then "
          "quit and reopen that app.")


def _info(**changes):
    """The crate's info reply on a Mac, with two displays."""
    info = {
        "ok": True, "name": "pythia_ui", "version": "0.1.0", "abi": 1,
        "ops": ["info", "screenshot"],
        "system": {"os": "macos", "version": "14.5", "arch": "aarch64"},
        "displays": [
            {"id": 1, "main": True, "builtin": True,
             "bounds": {"x": 0, "y": 0, "width": 1512, "height": 982},
             "pixels": {"width": 3024, "height": 1964}, "scale": 2.0},
            {"id": 5, "main": False, "builtin": False,
             "bounds": {"x": 1512, "y": 0, "width": 2560, "height": 1440},
             "pixels": {"width": 2560, "height": 1440}, "scale": 1.0},
        ],
        "capture": {"paths": ["screencapturekit", "coregraphics"],
                    "default": "screencapturekit", "allowed": True},
        "input": {"allowed": False},
    }
    info.update(changes)
    return info


def _shot(image=_PNG, **changes):
    """The crate's screenshot reply for the PNG ``image``, with its size."""
    width, height = struct.unpack(">II", image[16:24])
    reply = {"ok": True, "display": 1, "capture": "screencapturekit", "width": width,
             "height": height, "bounds": {"x": 0, "y": 0, "width": 1512, "height": 982},
             "cursor": True, "media_type": "image/png", "bytes": len(image),
             "png": base64.b64encode(image).decode("ascii")}
    reply.update(changes)
    return reply


class _FakeUi:
    """The UI plugin, faked: info, and the screenshot replies in turn (by
    default, a good one each time). ``requests`` has what it was asked, ids
    left out."""

    def __init__(self, *shots, info=None, abi=1):
        self.info = _info() if info is None else info
        self.shots = list(shots)
        self.requests = []
        self.library = FakeLibrary(json_answer(self._answer), abi=abi)

    def _answer(self, request):
        self.requests.append({key: value for key, value in request.items() if key != "id"})
        if request["op"] == "info":
            return self.info
        return self.shots.pop(0) if self.shots else _shot()

    def asked(self):
        return [request for request in self.requests if request["op"] != "info"]

    def patches(self, path):
        """What makes this the plugin that the CLI loads: on macOS, at path."""
        stack = ExitStack()
        stack.enter_context(mock.patch.object(ui_tool, "_platform", return_value="darwin"))
        stack.enter_context(mock.patch.object(ui_tool, "plugin_path", return_value=path))
        stack.enter_context(mock.patch.object(plugin, "_open_library",
                                              return_value=self.library))
        return stack


class _Case(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dylib = self.root / "libpythia_ui.dylib"
        self.dylib.write_bytes(b"a library")

    def load(self, fake, *, api="messages", default_tools=True):
        with fake.patches(self.dylib):
            return load_ui_plugin(api=api, default_tools=default_tools)

    def tool(self, fake=None, *, outbox=None):
        fake = _FakeUi() if fake is None else fake
        return UiTool(self.load(fake), outbox=Outbox(api="messages") if outbox is None
                      else outbox)

    def run_call(self, tool, call, **fields):
        return tool.tool().handler({"call": call, **fields})

    def refused(self, tool, call, **fields):
        outcome = self.run_call(tool, call, **fields)
        self.assertFalse(outcome.success, outcome.output)
        self.assertEqual(tool.outbox.queued(), 0)
        return outcome.output


class LoadTests(_Case):
    def test_it_loads_the_plugin_and_reads_its_info(self):
        fake = _FakeUi()
        loaded = self.load(fake)
        self.assertEqual(loaded.methods, ("screenshot",))
        self.assertEqual(loaded.plugin.path, self.dylib)
        self.assertEqual(loaded.info["name"], "pythia_ui")
        self.assertEqual(fake.requests, [{"op": "info"}])

    def test_each_reason_that_stops_the_launch(self):
        def refusal(fake=None, **kwargs):
            with self.assertRaises(UiPluginError) as caught:
                self.load(_FakeUi() if fake is None else fake, **kwargs)
            return str(caught.exception)

        self.assertEqual(refusal(default_tools=False),
                         f"{_OPTION} needs the default tools, which --enable-default-tools "
                         "False turns off.")
        self.assertEqual(refusal(api="claude-relay"),
                         f"{_OPTION}: --endpoint-api claude-relay takes no images; use codex, "
                         "responses, chat-completions, or messages.")
        with self.assertRaises(UiPluginError) as caught:
            load_ui_plugin(api="messages", default_tools=True, path=self.dylib)
        self.assertEqual(str(caught.exception),
                         f"{_OPTION}: the UI plugin is for macOS, and this is linux.")
        missing = self.root / "nothing.dylib"
        with mock.patch.object(ui_tool, "_platform", return_value="darwin"), \
                self.assertRaises(UiPluginError) as caught:
            load_ui_plugin(api="messages", default_tools=True, path=missing)
        self.assertEqual(str(caught.exception),
                         f"{_OPTION}: there is no UI plugin at {missing}; build and install it "
                         f"from a pythia checkout on this Mac with `{_INSTALL}`.")
        fake = _FakeUi()
        with fake.patches(self.dylib), \
                mock.patch.object(plugin, "_open_library", side_effect=OSError("not loadable")), \
                self.assertRaises(UiPluginError) as caught:
            load_ui_plugin(api="messages", default_tools=True)
        self.assertEqual(str(caught.exception),
                         f"{_OPTION}: can't load {self.dylib}: not loadable")
        self.assertEqual(refusal(_FakeUi(abi=2)),
                         f"{_OPTION}: {self.dylib} has plugin interface version 2, and this "
                         f"pythia knows version 1. Rebuild it from this pythia's checkout "
                         f"with `{_INSTALL}`.")
        self.assertEqual(refusal(_FakeUi(info={"ok": False, "kind": "failed",
                                               "error": "info: no displays"})),
                         f"{_OPTION}: the UI plugin's info failed: info: no displays")
        self.assertEqual(refusal(_FakeUi(info=_info(ops=["info"]))),
                         f"{_OPTION}: the UI plugin at {self.dylib} offers none of the ui "
                         "tool's methods (screenshot); its ops are info.")
        silent = _FakeUi()
        silent.library.answer = lambda raw: None
        self.assertEqual(refusal(silent), f"{_OPTION}: the plugin made no reply "
                                          "(pythia_plugin_call returned 1).")


class CallTests(_Case):
    def test_the_canonical_form_reads_each_spelling_the_same(self):
        for written, read in {
            "screenshot()": "ui.screenshot()",
            " ui . screenshot( )": "ui.screenshot()",
            "screenshot(display=None, cursor=True)": "ui.screenshot()",
            "screenshot(5)": "ui.screenshot(display=5)",
            "ui.screenshot(5, cursor=False)": "ui.screenshot(display=5, cursor=False)",
            "screenshot(cursor=1)": "ui.screenshot(cursor=1)",  # not True: refused later
        }.items():
            with self.subTest(written=written):
                self.assertEqual(call_text({"call": written}), read)

    def test_inputs_that_mean_no_one_call_are_refused_before_the_plugin_is_asked(self):
        fake = _FakeUi()
        tool = self.tool(fake)
        cases = {
            "screenshot": "ui: call must be one method call, such as screenshot().",
            "click(10, 20)": "ui: there is no method click(); the tool has one, "
                             "screenshot(display=None, cursor=True).",
            "screenshot(display=main)": "ui.screenshot(): display must be a literal, such as 1.",
            "screenshot(window=1)": "ui.screenshot(): got an unexpected keyword argument "
                                    "'window'",
            "screenshot('main')": "ui.screenshot(): display must be a display ID, such as 1, "
                                  "or None for the main display, not 'main'.",
            "screenshot(-1)": "ui.screenshot(): display must be a display ID, such as 1, or "
                              "None for the main display, not -1.",
            "screenshot(True)": "ui.screenshot(): display must be a display ID, such as 1, or "
                                "None for the main display, not True.",
            "screenshot(cursor='yes')": "ui.screenshot(): cursor must be True or False, not "
                                        "'yes'.",
        }
        for call, expected in cases.items():
            with self.subTest(call=call):
                self.assertEqual(self.refused(tool, call), expected)
        self.assertEqual(tool.tool().handler({}).output,
                         "ui: give call, one method call such as screenshot().")
        self.assertEqual(self.refused(tool, "screenshot()", code="x"),
                         "ui: unexpected field 'code'; give call only.")
        self.assertEqual(fake.asked(), [])

    def test_a_call_shows_as_the_call_it_stands_for(self):
        def render(call):
            return [item.text for item in render_interaction_items(
                (ToolCall("ui", "u1", json.dumps({"call": call})),))]

        self.assertEqual(render("screenshot(5)"),
                         ["[tool-call] ui (u1)\nui.screenshot(display=5)"])
        self.assertEqual(render("click(10, 20)"), ["[tool-call] ui (u1)\nui.click(10, 20)"])

    def test_the_spec_names_the_displays_and_the_cap(self):
        spec = self.tool().tool().spec
        self.assertEqual(spec.name, "ui")
        self.assertEqual(spec.parameters["required"], ["call"])
        self.assertFalse(spec.parameters["additionalProperties"])
        self.assertIn("screenshot(display=None, cursor=True)", spec.description)
        self.assertIn("Displays: 1 (main, built-in, 3024×1964 px), 5 (2560×1440 px).",
                      spec.description)
        self.assertIn("goes to the model provider", spec.description)
        self.assertIn("With messages, a request carries at most 20 images", spec.description)
        open_tool = self.tool(outbox=Outbox(api="chat-completions"))
        self.assertNotIn("at most 20 images", open_tool.tool().spec.description)


class ScreenshotTests(_Case):
    def test_a_screenshot_queues_one_message_with_the_header_and_the_image(self):
        fake = _FakeUi()
        tool = self.tool(fake)
        outcome = self.run_call(tool, "screenshot()")
        self.assertTrue(outcome.success, outcome.output)
        self.assertEqual(outcome.output, "Queued a screenshot of display 1 (image/png 1×1 px, "
                                         "69 B), in a user message that arrives right before "
                                         "your next sample, once this tool batch is done.")
        (message,) = tool.outbox.take()
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content_text, f"{UI_HEADER}: display 1, 1×1 px.")
        self.assertEqual(message.parts[1].source_uri,
                         "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii"))
        self.assertEqual(fake.asked(), [{"op": "screenshot"}])

    def test_the_display_and_the_cursor_reach_the_plugin(self):
        fake = _FakeUi(_shot(display=5, cursor=False), _shot(cursor=False))
        tool = self.tool(fake)
        outcome = self.run_call(tool, "screenshot(5, cursor=False)")
        self.assertEqual(fake.asked(), [{"op": "screenshot", "display": 5, "cursor": False}])
        self.assertTrue(outcome.output.startswith("Queued a screenshot of display 5 (image/png "
                                                  "1×1 px, 69 B), in a user message"),
                        outcome.output)
        self.assertEqual(tool.outbox.take()[0].content_text,
                         f"{UI_HEADER}: display 5, 1×1 px.")
        # Asked for, and left out: the CoreGraphics path never draws it.
        self.assertIn("(image/png 1×1 px, 69 B, without the pointer, which this capture path "
                      "leaves out)", self.run_call(tool, "screenshot()").output)

    def test_a_failure_reply_is_the_result_and_queues_nothing(self):
        tool = self.tool(_FakeUi({"ok": False, "kind": "permission",
                                  "error": f"screenshot: {_ALLOW}"}))
        self.assertEqual(self.refused(tool, "screenshot()"), f"ui.screenshot(): {_ALLOW}")

    def test_a_reply_that_would_break_the_context_fails_the_call(self):
        gif = b"GIF89a" + b"\x00" * 30
        cases = {
            "media_type is 'image/jpeg', not 'image/png'": _shot(media_type="image/jpeg"),
            "it has no png": _shot(png=None),
            "png isn't base64": _shot(png="not base64!"),
            "png isn't a PNG image": _shot(png=base64.b64encode(gif).decode(), bytes=len(gif)),
            "png has 69 bytes, not 70": _shot(bytes=70),
            "png is 1×1 px, not 2×1": _shot(width=2),
            "display is None": _shot(display=None),
            "width is -1": _shot(width=-1),
        }
        for what, reply in cases.items():
            with self.subTest(what):
                tool = self.tool(_FakeUi(reply))
                self.assertEqual(self.refused(tool, "screenshot()"),
                                 f"ui.screenshot(): the UI plugin's reply is malformed: {what}.")

    def test_no_reply_fails_the_call(self):
        fake = _FakeUi()
        tool = self.tool(fake)
        fake.library.answer = lambda raw: None
        self.assertEqual(self.refused(tool, "screenshot()"),
                         "ui.screenshot(): the plugin made no reply (pythia_plugin_call "
                         "returned 1).")

    def test_a_screenshot_over_the_size_limit_is_refused(self):
        tool = self.tool()
        with mock.patch.object(ui_tool, "MAX_SCREENSHOT_BYTES", 68):
            self.assertEqual(self.refused(tool, "screenshot()"),
                             "ui.screenshot(): the screenshot of display 1 is 69 B as PNG "
                             "(1×1 px), over the 68 B that one image may have. Nothing was "
                             "queued.")

    def test_the_notice_says_what_the_tool_sees_and_whether_macos_allows_it(self):
        self.assertEqual(self.tool().notice(),
                         "ui: the model can take screenshots of this Mac's displays with the "
                         f"ui tool (pythia_ui 0.1.0, {self.dylib}); each one goes to the model "
                         "provider, with whatever the display shows.")
        capture = {"paths": ["coregraphics"], "default": "coregraphics", "allowed": False}
        self.assertTrue(self.tool(_FakeUi(info=_info(capture=capture))).notice().endswith(
            "Screen recording isn't allowed yet: the first screenshot asks macOS for it, "
            "and fails."))


class OutboxTests(_Case):
    def test_the_media_and_ui_tools_share_one_outbox_and_its_cap(self):
        _write_png(self.root / "pic.png")
        outbox = Outbox(api="messages", cap=ImageCap(2, 10 ** 9))
        media = MediaTool(cwd=self.root, api="messages", outbox=outbox)
        ui = self.tool(outbox=outbox)
        self.assertEqual(outbox.writers, ("media", "ui"))
        self.assertTrue(media.tool().handler({"call": "attach('pic.png')"}).success)
        self.assertTrue(self.run_call(ui, "screenshot()").success)
        self.assertEqual(self.run_call(ui, "screenshot()").output,
                         "ui.screenshot(): the next request would carry 3 images, over this "
                         "session's cap of 2 for messages; the context and the queue hold 2 "
                         "already, the user's included. Nothing was queued.")
        self.assertTrue(media.tool().handler({"call": "attach('pic.png')"}).output.startswith(
            "media.attach(): the next request would carry 3 images"))
        self.assertEqual([message.content_text for message in outbox.take()],
                         [f"{MEDIA_HEADER}: 'pic.png'.", f"{UI_HEADER}: display 1, 1×1 px."])

    def test_a_shared_outbox_brings_its_cap_and_its_api(self):
        outbox = Outbox(api="messages")
        with self.assertRaises(TypeError):
            MediaTool(cwd=self.root, api="messages", outbox=outbox, cap=None)
        with self.assertRaises(ValueError):
            MediaTool(cwd=self.root, api="responses", outbox=outbox)


class _CliCase(_Case):
    def main(self, *argv, fake=None):
        """Run the CLI headless with a stub model; the exit code, stderr, and
        the model."""
        path = self.root / "session.jsonl"
        model = _Model(path, ModelSample(items=(ToolCall("ui", "u1", json.dumps(
            {"call": "screenshot()"})),)), _answer("A screen."))
        stderr = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(cli, "build_model", return_value=model))
            stack.enter_context(redirect_stderr(stderr))
            if fake is not None:
                stack.enter_context(fake.patches(self.dylib))
            code = cli.main(["--headless", "--cwd", str(self.root), "--save", str(path),
                             "--model", "claude-test", "--max-output-tokens", "64", *argv,
                             "--prompt", "What's on my screen?"])
        return code, stderr.getvalue(), model


class GatingTests(_CliCase):
    def test_the_option_is_off_by_default(self):
        parse = cli._build_parser().parse_args
        self.assertIs(parse([]).enable_experimental_ui_plugin, False)
        self.assertIs(parse([_OPTION]).enable_experimental_ui_plugin, True)
        self.assertIs(parse([_OPTION, "False"]).enable_experimental_ui_plugin, False)
        code, _, model = self.main("--endpoint-api", "messages", fake=_FakeUi())
        self.assertEqual(code, 0)
        self.assertNotIn("ui", [spec.name for spec in model.calls[0][1]])

    def test_a_plugin_that_cant_be_used_stops_the_launch_before_the_save(self):
        for argv, reason in {
            (_OPTION, "--endpoint-api", "messages"):
                "the UI plugin is for macOS, and this is linux.",
            (_OPTION, "--endpoint-api", "claude-relay"):
                "--endpoint-api claude-relay takes no images; use codex, responses, "
                "chat-completions, or messages.",
        }.items():
            with self.subTest(argv=argv):
                code, stderr, model = self.main(*argv)
                self.assertEqual(code, 1)
                self.assertIn(f"interaction CLI failed: {_OPTION}: {reason}", stderr)
                self.assertEqual(model.calls, [])
                # Preparing the save would have made its private directory.
                self.assertFalse((self.root / "session.jsonl").exists())
                self.assertFalse((self.root / "session._private").exists())

    def test_a_reboots_start_check_loads_the_plugin(self):
        check = ["--headless", "--save", str(self.root / "s.jsonl"), "--endpoint-api",
                 "messages", _OPTION, "--rebooted=check"]
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(cli.main(check), 1)
        self.assertIn("the UI plugin is for macOS", stderr.getvalue())
        fake = _FakeUi()
        with fake.patches(self.dylib):
            self.assertEqual(cli.main(check), 0)
        self.assertEqual(fake.requests, [{"op": "info"}])

    def test_approval_covers_it_in_both_modes(self):
        self.assertTrue(cli._requires_approval(True, "ui"))
        self.assertTrue(cli._requires_approval(cli._APPROVAL_EXEC, "ui"))


class HeadlessTests(_CliCase):
    def test_the_screenshot_arrives_right_before_the_next_sample(self):
        fake = _FakeUi(info=_info(capture={"paths": ["screencapturekit"],
                                           "default": "screencapturekit", "allowed": False}))
        code, stderr, model = self.main(_OPTION, "--endpoint-api", "messages", fake=fake)
        self.assertEqual(code, 0, stderr)
        self.assertEqual([spec.name for spec in model.calls[0][1]][-1], "ui")
        items = model.calls[1][0].items
        self.assertEqual([type(item).__name__ for item in items[-2:]], ["ToolResult", "Message"])
        self.assertTrue(items[-2].output.startswith("Queued a screenshot of display 1"),
                        items[-2].output)
        self.assertEqual(items[-1].content_text, f"{UI_HEADER}: display 1, 1×1 px.")
        self.assertTrue(items[-1].has_media)
        saved = load_interaction_save(self.root / "session.jsonl").items
        self.assertIn(items[-1], saved)
        self.assertNotIsInstance(saved[saved.index(items[-1]) - 1], UserInteractionBoundary)
        self.assertEqual(fake.requests, [{"op": "info"}, {"op": "screenshot"}])
        self.assertIn("[cli] ui: the model can take screenshots of this Mac's displays", stderr)
        self.assertIn("Screen recording isn't allowed yet", stderr)
        self.assertEqual(fake.library.buffers, {})


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "interaction.jsonl"
        dylib = self.root / "libpythia_ui.dylib"
        dylib.write_bytes(b"a library")
        fake = _FakeUi()
        with fake.patches(dylib):
            self.ui = UiTool(load_ui_plugin(api="messages", default_tools=True),
                             outbox=Outbox(api="messages"))

    async def test_the_host_takes_its_screenshots_without_a_media_tool(self):
        state = cli._UIState(ui=self.ui)
        state.turn_active = True
        state.pending.append(cli._Submission("Hurry.", Message("user", "Hurry.")))
        host = cli._CliHost(state, mock.Mock(), asyncio.get_running_loop(), steering=True)
        self.ui.screenshot()
        answer = await asyncio.to_thread(host.interrupt)
        self.assertIsInstance(answer, Steer)
        self.assertEqual([message.content_text for message in answer.messages],
                         [f"{UI_HEADER}: display 1, 1×1 px.", "Hurry."])
        self.assertEqual(await asyncio.to_thread(host.interrupt), Interrupt.CONTINUE)

    async def test_one_still_queued_at_exit_is_dropped_with_a_notice(self):
        model = _Model(self.path, ModelSample(items=(ToolCall("ui", "u1", json.dumps(
            {"call": "screenshot()"})),)))
        terminal = _Terminal(lambda terminal, editor, status: terminal.key("c-d")
                             if status.split(" | ")[0] == "failed" else None)
        args = cli._build_parser().parse_args(["--prompt", "Look.", "--max-samples", "1"])
        environment = Environment((self.ui.tool(),))
        self.assertEqual(await asyncio.wait_for(cli._run(
            model, environment, terminal, args, self.path, ui=self.ui), timeout=8), 1)
        texts = [item.text for item in terminal.items]
        self.assertIn("[tool-call] ui (u1)\nui.screenshot()", texts)
        self.assertIn("[cli] ui: dropped 1 queued message of the model's attachments, which no "
                      "turn took; they live only in this process.", texts)
        self.assertEqual(self.ui.outbox.queued(), 0)


if __name__ == "__main__":
    unittest.main()
