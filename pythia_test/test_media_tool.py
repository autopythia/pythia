"""The experimental media tool (media_tool.py): attach() and its delivery."""

from __future__ import annotations

import asyncio
import base64
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import Environment, MediaPart, Message, ModelSample, TextPart
from pythia.interaction import ToolCall, ToolResult, UserInteractionBoundary, cli
from pythia.interaction import load_interaction_save
from pythia.interaction.display import render_interaction_items
from pythia.interaction.loop import Interrupt, Steer
from pythia.interaction import media_tool
from pythia.interaction.media import ContentLimits
from pythia.interaction.media_tool import IMAGE_CAPS, MEDIA_HEADER, MEDIA_TOOL_LIMITS
from pythia.interaction.media_tool import ImageCap, MediaTool, call_text
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_media import _PNG, _symlink, _write_png


_GIF = b"GIF89a" + b"\x00" * 16
_WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8


def _media(call_id, call):
    return ToolCall("media", call_id, json.dumps({"call": call}))


def _calls(*calls):
    return ModelSample(items=tuple(calls))


def _image(uri="https://example.com/x.png"):
    return Message("user", (MediaPart(uri), TextPart("look")))


class _ToolCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        _write_png(self.root / "shot.png")

    def tool(self, api="messages", **kwargs):
        return MediaTool(cwd=self.root, api=api, **kwargs)

    def run_call(self, tool, call, **fields):
        return tool.tool().handler({"call": call, **fields})

    def refused(self, tool, call, **fields):
        outcome = self.run_call(tool, call, **fields)
        self.assertFalse(outcome.success, outcome.output)
        self.assertEqual(tool.queued(), 0)
        return outcome.output


class CallReadingTests(_ToolCase):
    def test_the_canonical_form_reads_each_spelling_the_same(self):
        for written in ("attach('shot.png', text='Hi.')", " media.attach( 'shot.png',text='Hi.')",
                        'media . attach("shot.png", text="Hi.")'):
            self.assertEqual(call_text({"call": written}), "media.attach('shot.png', text='Hi.')")
        self.assertEqual(call_text({"call": "attach('a.png', 'https://example.com/b.png')"}),
                         "media.attach('a.png', 'https://example.com/b.png')")

    def test_inputs_that_mean_no_one_call_are_refused_with_what_to_give(self):
        tool = self.tool()
        cases = {
            "": "media: call must be one method call, such as attach('build/shot.png'); ",
            "attach": "media: call must be one method call, such as attach('build/shot.png').",
            "images()": "media: there is no method images(); the tool has one, "
                        "attach(*refs, text=None).",
            "attach(*['shot.png'])": "media.attach(): give each ref by itself, not *sequence.",
            "attach(**{'text': 'Hi'})": "media.attach(): give each argument by itself, "
                                        "not **mapping.",
            "attach(path)": "media.attach(): ref 1 must be a literal, such as 'build/shot.png'.",
            "attach('shot.png', note='Hi')": "media.attach(): got an unexpected keyword "
                                             "argument 'note'",
            "attach('shot.png', text=caption)": "media.attach(): text must be a literal",
        }
        for call, expected in cases.items():
            with self.subTest(call=call):
                self.assertTrue(self.refused(tool, call).startswith(expected),
                                self.refused(tool, call))
                self.assertIsNone(call_text({"call": call}))
        self.assertEqual(tool.tool().handler({}).output, "media: give call, one method call "
                         "such as attach('build/shot.png', text='Is the header green now?').")
        self.assertEqual(self.refused(tool, "attach('shot.png')", code="x"),
                         "media: unexpected field 'code'; give call only.")
        self.assertEqual(tool.tool().handler({"call": 3}).output,
                         "media: call must be text, one method call such as "
                         "attach('build/shot.png').")

    def test_the_spec_takes_one_required_call(self):
        spec = self.tool().tool().spec
        self.assertEqual(spec.name, "media")
        self.assertEqual(spec.parameters["required"], ["call"])
        self.assertFalse(spec.parameters["additionalProperties"])
        self.assertIn("attach(*refs, text=None)", spec.description)


class AttachTests(_ToolCase):
    def test_an_image_queues_one_message_with_the_header_the_image_and_the_text(self):
        tool = self.tool()
        # As JSON's "\\n" reads: an escape in the call, which is Python.
        outcome = self.run_call(tool, "attach('shot.png', text='  Is the header green now?\\n')")
        self.assertTrue(outcome.success, outcome.output)
        self.assertEqual(outcome.output, "Queued 1 image (image/png 69 B), in a user message "
                         "that arrives right before your next sample, once this tool batch "
                         "is done.")
        self.assertEqual(tool.queued(), 1)
        (message,) = tool.take()
        self.assertEqual(tool.queued(), 0)
        self.assertEqual(tool.take(), ())
        encoded = base64.b64encode(_PNG).decode("ascii")
        self.assertEqual(message, Message("user", (
            TextPart(f"{MEDIA_HEADER}: 'shot.png'."),
            MediaPart(f"data:image/png;base64,{encoded}"),
            TextPart("Is the header green now?"))))

    def test_refs_keep_their_order_and_text_files_are_pasted(self):
        (self.root / "notes.md").write_text("# Notes\nGreen header.\n")
        (self.root / "sub dir").mkdir()
        _write_png(self.root / "sub dir" / "b.gif", _GIF)
        _write_png(self.root / "c.webp", _WEBP)
        tool = self.tool()
        outcome = self.run_call(tool, "attach('notes.md', 'sub dir/b.gif', "
                                      "'https://example.com/chart.jpeg', 'c.webp')")
        self.assertEqual(outcome.output, "Queued 3 images (image/gif 22 B, "
                         "https://example.com/chart.jpeg, image/webp 24 B) and 1 text file "
                         "('notes.md'), in a user message that arrives right before your next "
                         "sample, once this tool batch is done.")
        (message,) = tool.take()
        kinds = [part.text if isinstance(part, TextPart) else part.source_uri[:15]
                 for part in message.parts]
        self.assertEqual(kinds, [
            f"{MEDIA_HEADER}: 'notes.md', 'sub dir/b.gif', 'https://example.com/chart.jpeg', "
            "'c.webp'.\n\n# Notes\nGreen header.",
            "data:image/gif;", "https://example", "data:image/webp"])

    def test_text_files_alone_queue_a_plain_message(self):
        (self.root / "a.txt").write_text("alpha\n")
        tool = self.tool()
        self.assertTrue(self.run_call(tool, "attach('a.txt', text='Read it.')").success)
        self.assertEqual(tool.take(), (Message("user", f"{MEDIA_HEADER}: 'a.txt'.\n\nalpha"
                                                       "\n\nRead it."),))

    def test_the_content_decides_a_local_images_type(self):
        _write_png(self.root / "photo.jpg")  # a PNG named .jpg
        (self.root / "chart.svg").write_text("<svg/>")
        (self.root / "notes.png").write_text("not an image")
        tool = self.tool()
        self.assertTrue(self.run_call(tool, "attach('photo.jpg')").success)
        (message,) = tool.take()
        self.assertTrue(message.parts[1].source_uri.startswith("data:image/png;base64,"))
        for ref in ("chart.svg", "notes.png"):
            self.assertEqual(self.refused(tool, f"attach({ref!r})"),
                             f"media.attach(): {ref!r} is not a PNG, JPEG, GIF or WebP image, "
                             "the types that the model APIs take.")
        self.assertEqual(self.refused(tool, "attach('shot.png', 'https://example.com/a.svg')"),
                         "media.attach(): 'https://example.com/a.svg' doesn't name a PNG, JPEG, "
                         "GIF or WebP image.")

    def test_the_users_rules_and_messages_apply_to_each_ref(self):
        tool = self.tool()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        outside = Path(temporary.name)
        _write_png(outside / "far.png")
        _symlink(self.root / "link.png", outside / "far.png")
        self.assertEqual(self.refused(tool, f"attach({str(outside / 'far.png')!r})"),
                         f"media.attach(): attachment path is outside --cwd: "
                         f"{outside / 'far.png'}")
        self.assertEqual(self.refused(tool, "attach('link.png')"),
                         "media.attach(): attachment path is outside --cwd: link.png")
        self.assertEqual(self.refused(tool, "attach('missing.png')"),
                         "media.attach(): cannot read attachment 'missing.png': "
                         "No such file or directory")
        self.assertEqual(self.refused(tool, "attach('file:///etc/passwd')"),
                         "media.attach(): unsupported URL scheme in attachment: file://")
        self.assertEqual(self.refused(tool, "attach('https://example.com/a.md')"),
                         "media.attach(): text files are pasted only from local paths: "
                         "https://example.com/a.md")
        # The policy is read when the call runs: /config can turn it off.
        tool.bind(enable_workspace=lambda: False)
        self.assertTrue(self.run_call(tool, "attach('link.png')").success)

    def test_the_tools_own_limits(self):
        self.assertEqual(MEDIA_TOOL_LIMITS, ContentLimits(
            max_items=4, max_item_bytes=7 * 1024 * 1024, max_total_bytes=14 * 1024 * 1024,
            max_text_bytes=64 * 1024))
        # In base64, under the Claude API's 10 MB per image.
        self.assertLess(-(-MEDIA_TOOL_LIMITS.max_item_bytes // 3) * 4, 10_000_000)
        tool = self.tool()
        self.assertEqual(self.refused(tool, "attach(" + "'shot.png', " * 5 + ")"),
                         "media.attach(): too many attachments: 5 > 4")
        small = self.tool(limits=ContentLimits(max_item_bytes=60))
        self.assertEqual(self.refused(small, "attach('shot.png')"),
                         "media.attach(): attachment is too large (69 bytes; limit 60): "
                         "shot.png")

    def test_an_attach_without_refs_or_with_other_values_is_refused(self):
        tool = self.tool()
        self.assertEqual(self.refused(tool, "attach(text='Hi.')"),
                         "media.attach(): give at least one ref, such as "
                         "attach('build/shot.png').")
        self.assertEqual(self.refused(tool, "attach('shot.png', 3)"),
                         "media.attach(): ref 2 must be text, such as 'build/shot.png', "
                         "not int.")
        self.assertEqual(self.refused(tool, "attach('shot.png', text=['Hi.'])"),
                         "media.attach(): text must be text, not list.")
        self.assertEqual(self.refused(tool, "attach(' shot.png')"),
                         "media.attach(): invalid attachment reference: ' shot.png'")


class CapTests(_ToolCase):
    def test_messages_has_a_cap_and_the_other_apis_none_yet(self):
        self.assertEqual(IMAGE_CAPS, {"messages": ImageCap(20, 24 * 1024 * 1024)})
        self.assertIn("With messages, a request carries at most 20 images, the user's "
                      "included, and 24.0 MiB of image data; past that, attach fails.",
                      self.tool().tool().spec.description)
        open_tool = self.tool(api="chat-completions")
        self.assertNotIn("at most 20 images", open_tool.tool().spec.description)
        for _ in range(25):
            self.assertTrue(self.run_call(open_tool, "attach('shot.png')").success)
        self.assertEqual(open_tool.queued(), 25)

    def test_the_count_includes_the_contexts_images_and_the_queued_ones(self):
        tool = self.tool(cap=ImageCap(3, 10 ** 9))
        tool.bind(context_items=lambda: (_image(), ToolResult("x", "ok"), Message("user", "hi")))
        self.assertTrue(self.run_call(tool, "attach('shot.png')").success)
        self.assertEqual(self.run_call(tool, "attach('shot.png', 'shot.png')").output,
                         "media.attach(): the next request would carry 4 images, over this "
                         "session's cap of 3 for messages; the context and the queue hold 2 "
                         "already, the user's included. Nothing was queued.")
        self.assertEqual(tool.queued(), 1)
        self.assertTrue(self.run_call(tool, "attach('shot.png')").success)

    def test_inline_data_has_its_own_cap_which_urls_dont_count_toward(self):
        size = len(f"data:image/png;base64,{base64.b64encode(_PNG).decode('ascii')}")
        tool = self.tool(cap=ImageCap(20, size + 10))
        self.assertTrue(self.run_call(tool, "attach('shot.png', 'https://example.com/a.png')")
                        .success)
        self.assertEqual(self.run_call(tool, "attach('shot.png')").output,
                         f"media.attach(): the next request's images would hold {2 * size} B of "
                         f"inline data, over this session's cap of {size + 10} B for messages. "
                         "Nothing was queued.")
        self.assertTrue(self.run_call(tool, "attach('https://example.com/b.png')").success)

    def test_text_files_pass_a_full_cap(self):
        (self.root / "a.md").write_text("alpha")
        tool = self.tool(cap=ImageCap(1, 10 ** 9))
        tool.bind(context_items=lambda: (_image(),))
        self.assertTrue(self.run_call(tool, "attach('a.md')").success)
        self.assertTrue(self.run_call(tool, "attach('shot.png')").output.startswith(
            "media.attach(): the next request would carry 2 images, over this session's cap "
            "of 1 for messages"))


class DisplayTests(unittest.TestCase):
    def render(self, call):
        return [item.text for item in render_interaction_items((call,))]

    def test_a_call_shows_as_the_call_it_stands_for(self):
        self.assertEqual(self.render(_media("m1", "attach( 'shot.png',text='Is it red?')")),
                         ["[tool-call] media (m1)\nmedia.attach('shot.png', text='Is it red?')"])

    def test_a_refused_call_shows_as_written(self):
        self.assertEqual(self.render(_media("m1", "images()")),
                         ["[tool-call] media (m1)\nmedia.images()"])
        self.assertEqual(self.render(ToolCall("media", "m2", json.dumps({"path": "a.png"}))),
                         ["[tool-call] media (m2)\nmedia(path='a.png')"])


class GatingTests(unittest.TestCase):
    def media_tool(self, *argv):
        args = cli._build_parser().parse_args(list(argv))
        return cli._media_tool(args, Path("."))

    def test_it_comes_with_experimental_media_on_an_api_that_takes_images(self):
        for api in ("messages", "chat-completions", "responses", "codex"):
            with self.subTest(api=api):
                tool = self.media_tool("--enable-experimental-media", "--endpoint-api", api)
                self.assertIsInstance(tool, MediaTool)
        self.assertIsNone(self.media_tool("--endpoint-api", "messages"))
        self.assertIsNone(self.media_tool("--enable-experimental-media",
                                          "--endpoint-api", "claude-relay"))
        self.assertIsNone(self.media_tool("--enable-experimental-media", "--endpoint-api",
                                          "messages", "--enable-default-tools", "False"))

    def test_approval_covers_it_in_exec_mode_too(self):
        self.assertTrue(cli._requires_approval(True, "media"))
        self.assertTrue(cli._requires_approval(cli._APPROVAL_EXEC, "media"))


class HeadlessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "session.jsonl"
        _write_png(self.root / "pic.png")

    def run_main(self, *outcomes, argv=("--enable-experimental-media",),
                 prompt="Look at pic.png."):
        model = _Model(self.path, *outcomes)
        stderr = io.StringIO()
        with mock.patch.object(cli, "build_model", return_value=model), redirect_stderr(stderr):
            code = cli.main(["--headless", "--cwd", str(self.root), "--save", str(self.path),
                             "--endpoint-api", "messages", "--model", "claude-test",
                             "--max-output-tokens", "64", *argv, "--prompt", prompt])
        self.assertEqual(code, 0, stderr.getvalue())
        return model

    def test_the_image_arrives_right_before_the_next_sample(self):
        model = self.run_main(
            _calls(_media("m1", "attach('pic.png', text='Is it red?')"),
                   ToolCall("missing", "m2", "{}")),
            _answer("It is red."))
        self.assertEqual([spec.name for spec in model.calls[0][1]][-1], "media")
        items = model.calls[1][0].items
        # The whole batch first, then the message, which has no boundary.
        self.assertEqual([type(item).__name__ for item in items[-3:]],
                         ["ToolResult", "ToolResult", "Message"])
        self.assertTrue(items[-3].success, items[-3].output)
        self.assertTrue(items[-3].output.startswith("Queued 1 image (image/png 69 B)"))
        message = items[-1]
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content_text, f"{MEDIA_HEADER}: 'pic.png'.\nIs it red?")
        self.assertTrue(message.parts[1].source_uri.startswith("data:image/png;base64,"))
        saved = load_interaction_save(self.path).items
        self.assertIn(message, saved)
        self.assertNotIsInstance(saved[saved.index(message) - 1], UserInteractionBoundary)

    def test_the_cap_counts_the_users_own_images(self):
        with mock.patch.dict(media_tool.IMAGE_CAPS, {"messages": ImageCap(1, 10 ** 9)}):
            model = self.run_main(_calls(_media("m1", "attach('pic.png')")), _answer("ok"),
                                  prompt="@pic.png Look at it.")
        self.assertEqual(model.calls[1][0].items[-1], ToolResult(
            "m1", "media.attach(): the next request would carry 2 images, over this session's "
            "cap of 1 for messages; the context and the queue hold 1 already, the user's "
            "included. Nothing was queued.", success=False))

    def test_the_workspace_policy_is_the_sessions(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        outside = _write_png(Path(temporary.name) / "far.png")
        call = _media("m1", f"attach({str(outside)!r})")
        model = self.run_main(_calls(call), _answer("ok"))
        self.assertEqual(model.calls[1][0].items[-1].output,
                         f"media.attach(): attachment path is outside --cwd: {outside}")
        self.path = self.root / "second.jsonl"
        model = self.run_main(_calls(call), _answer("ok"), argv=(
            "--enable-experimental-media", "--enable-workspace", "False"))
        self.assertTrue(model.calls[1][0].items[-1].has_media)

    def test_without_the_flag_there_is_no_media_tool(self):
        model = self.run_main(_calls(_media("m1", "attach('pic.png')")), _answer("ok"), argv=())
        self.assertNotIn("media", [spec.name for spec in model.calls[0][1]])
        self.assertEqual(model.calls[1][0].items[-1],
                         ToolResult("m1", "Unknown tool: media", success=False))


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "interaction.jsonl"
        _write_png(self.root / "pic.png")
        self.media = MediaTool(cwd=self.root, api="messages")

    async def test_its_messages_come_before_the_users_steers(self):
        state = cli._UIState(media=self.media)
        state.turn_active = True
        state.pending.append(cli._Submission("Hurry.", Message("user", "Hurry.")))
        host = cli._CliHost(state, mock.Mock(), asyncio.get_running_loop(), steering=True)
        self.media.attach("pic.png")
        answer = await asyncio.to_thread(host.interrupt)
        self.assertIsInstance(answer, Steer)
        self.assertEqual([message.content_text for message in answer.messages],
                         [f"{MEDIA_HEADER}: 'pic.png'.", "Hurry."])
        self.assertEqual(await asyncio.to_thread(host.interrupt), Interrupt.CONTINUE)

    async def test_a_stop_leaves_them_queued(self):
        state = cli._UIState(media=self.media)
        host = cli._CliHost(state, mock.Mock(), asyncio.get_running_loop(), steering=True)
        self.media.attach("pic.png")
        state.closing = True  # as a stop sets it (_UIState.request_stop)
        self.assertEqual(await asyncio.to_thread(host.interrupt), Interrupt.STOP)
        self.assertEqual(self.media.queued(), 1)

    async def test_one_that_no_interrupt_point_took_goes_with_the_next_turns_input(self):
        model = _Model(self.path, _calls(_media("m1", "attach('pic.png')")), _answer("Red."))
        submitted = []

        def frame(terminal, editor, status):
            phase, texts = status.split(" | ")[0], [item.text for item in terminal.items]
            if phase == "failed" and not submitted:
                # The turn ended at --max-samples, right after its tool batch.
                submitted.append(True)
                terminal.submit("And now?")
            elif phase == "idle" and "[assistant] Red." in texts:
                terminal.key("c-d")

        terminal = _Terminal(frame)
        args = cli._build_parser().parse_args(["--prompt", "Look.", "--max-samples", "1"])
        environment = Environment((self.media.tool(),))
        # 1: the first turn failed, at --max-samples (_drive_interaction_body).
        self.assertEqual(await asyncio.wait_for(cli._run(
            model, environment, terminal, args, self.path, media=self.media), timeout=8), 1)
        self.assertEqual(len(model.calls), 2)
        items = model.calls[1][0].items
        self.assertEqual(items[-3], Message("user", "And now?"))
        self.assertIsInstance(items[-2], UserInteractionBoundary)
        self.assertEqual(items[-1].content_text, f"{MEDIA_HEADER}: 'pic.png'.")
        self.assertTrue(items[-1].has_media)
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] SampleLimitExceeded: Model did not produce a final answer within 1 "
                      "samples.", texts)
        self.assertIn("[tool-call] media (m1)\nmedia.attach('pic.png')", texts)
        self.assertIn(f"[user] {MEDIA_HEADER}: 'pic.png'.", texts)
        self.assertIn("[user] [image] image/png 69 B", texts)
        self.assertFalse(any(text.startswith("[cli] media: dropped") for text in texts))

    async def test_one_still_queued_at_exit_is_dropped_with_a_notice(self):
        model = _Model(self.path, _calls(_media("m1", "attach('pic.png')")))
        terminal = _Terminal(lambda terminal, editor, status: terminal.key("c-d")
                             if status.split(" | ")[0] == "failed" else None)
        args = cli._build_parser().parse_args(["--prompt", "Look.", "--max-samples", "1"])
        environment = Environment((self.media.tool(),))
        self.assertEqual(await asyncio.wait_for(cli._run(
            model, environment, terminal, args, self.path, media=self.media), timeout=8), 1)
        self.assertEqual(self.media.queued(), 0)
        self.assertIn("[cli] media: dropped 1 queued message of the model's attachments, which "
                      "no turn took; they live only in this process.",
                      [item.text for item in terminal.items])


if __name__ == "__main__":
    unittest.main()
