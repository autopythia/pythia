"""Experimental media user messages: parsing, codec, encoders, CLI."""

from __future__ import annotations

import base64
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import (
    Environment,
    MediaPart,
    InteractionContext,
    Message,
    ModelConfigurationError,
    SaveError,
    TextPart,
    cli,
    demo,
    interaction_item_from_dict,
    interaction_item_to_dict,
    load_interaction_save,
)
from pythia.interaction.chat_completions import _encode_context_messages
from pythia.interaction.messages import _encode_context
from pythia.interaction.media import (
    AttachmentError,
    ContentLimits,
    DEFAULT_TEXT_SUFFIXES,
    content_item_to_responses,
    parse_user_prompt,
    resolve_content,
    split_leading_references,
)
from pythia.interaction.responses import _encode_context_items
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_png(path, data=_PNG):
    path.write_bytes(data)
    return path


def _symlink(link, target):
    try:
        os.symlink(target, link)
    except (AttributeError, NotImplementedError, OSError) as exc:
        raise unittest.SkipTest(f"symlinks unavailable: {exc}") from None


class SplitLeadingReferencesTests(unittest.TestCase):
    def test_no_leading_reference_is_identity(self):
        for prompt in ("hello", "", "   "):
            with self.subTest(prompt=prompt):
                self.assertEqual(split_leading_references(prompt), ((), prompt))

    def test_leading_references_and_text(self):
        self.assertEqual(
            split_leading_references("@a.png @b.jpg  describe  this"),
            (("a.png", "b.jpg"), "describe  this"),
        )
        self.assertEqual(split_leading_references("  @a.png tail"), (("a.png",), "tail"))

    def test_attachment_only(self):
        self.assertEqual(split_leading_references("@a.png"), (("a.png",), ""))
        self.assertEqual(split_leading_references("@a.png   "), (("a.png",), ""))

    def test_bare_at_and_escape_end_the_run(self):
        self.assertEqual(split_leading_references("@ @x"), ((), "@ @x"))
        self.assertEqual(split_leading_references("@"), ((), "@"))
        self.assertEqual(split_leading_references("@@literal"), ((), "@literal"))
        self.assertEqual(
            split_leading_references("@a.png @@lit"),
            (("a.png",), "@lit"),
        )

    def test_non_at_token_ends_the_run(self):
        self.assertEqual(
            split_leading_references("@a.png\nb.png rest"),
            (("a.png",), "b.png rest"),
        )


class ResolveContentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def resolve(self, reference, **kwargs):
        return resolve_content(
            (reference,),
            cwd=self.cwd,
            enable_workspace=True,
            **kwargs,
        )

    def test_local_image_is_inlined_as_data_url(self):
        _write_png(self.cwd / "pic.png")
        parts = self.resolve("pic.png")
        self.assertEqual(len(parts), 1)
        self.assertIsInstance(parts[0], MediaPart)
        self.assertTrue(parts[0].source_uri.startswith("data:image/png;base64,"))
        encoded = parts[0].source_uri.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), _PNG)

    def test_magic_sniff_when_extension_is_unknown(self):
        _write_png(self.cwd / "blob.dat")
        (part,) = self.resolve("blob.dat")
        self.assertTrue(part.source_uri.startswith("data:image/png;base64,"))

    def test_absolute_path_resolution(self):
        path = _write_png(self.cwd / "pic.png")
        (part,) = self.resolve(str(path))
        self.assertTrue(part.source_uri.startswith("data:image/png;"))

    def test_workspace_containment(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside = _write_png(Path(outside_dir.name) / "out.png")
        with self.assertRaisesRegex(AttachmentError, "outside --cwd"):
            resolve_content((str(outside),), cwd=self.cwd, enable_workspace=True)
        parts = resolve_content((str(outside),), cwd=self.cwd, enable_workspace=False)
        self.assertTrue(parts[0].source_uri.startswith("data:image/png;"))

    def test_missing_directory_and_non_image_are_rejected(self):
        (self.cwd / "empty.png").mkdir()
        (self.cwd / "folder.md").mkdir()
        (self.cwd / "notes.pdf").write_text("hello")
        for reference, message in (
            ("missing.png", "cannot read attachment"),
            ("missing.txt", "cannot read attachment"),
            ("empty.png", "not a regular file"),
            ("folder.md", "not a regular file"),
            ("notes.pdf", r"not a supported image or text file \(\.md, \.txt\): notes\.pdf$"),
        ):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AttachmentError, message):
                    self.resolve(reference)

    def test_limits(self):
        _write_png(self.cwd / "pic.png")
        with self.assertRaisesRegex(AttachmentError, "too large"):
            self.resolve("pic.png", limits=ContentLimits(max_item_bytes=4))
        with self.assertRaisesRegex(AttachmentError, "too many"):
            resolve_content(
                ("a", "b", "c"),
                cwd=self.cwd,
                enable_workspace=True,
                limits=ContentLimits(max_items=2),
            )
        with self.assertRaisesRegex(AttachmentError, "in total"):
            resolve_content(
                ("pic.png", "pic.png"),
                cwd=self.cwd,
                enable_workspace=True,
                limits=ContentLimits(max_total_bytes=len(_PNG)),
            )

    def test_remote_urls(self):
        (part,) = self.resolve("https://example.test/a.png")
        self.assertEqual(part.source_uri, "https://example.test/a.png")
        for reference, message in (
            ("https://example.test/a.pdf", "not a supported image"),
            ("https://x", "not a supported image"),
            ("https:///a.png", "no host"),
            ("https://example.test/a.txt", "pasted only from local paths"),
            ("https://example.test/README.MD?raw=1", "pasted only from local paths"),
        ):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AttachmentError, message):
                    self.resolve(reference)

    def test_other_schemes_rejected(self):
        for reference in ("data:image/png;base64,AA", "file:///etc/passwd"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(AttachmentError, "unsupported URL scheme"):
                    self.resolve(reference)

    def test_errors_never_embed_payload(self):
        _write_png(self.cwd / "pic.png")
        with self.assertRaises(AttachmentError) as raised:
            self.resolve("pic.png", limits=ContentLimits(max_item_bytes=4))
        self.assertNotIn(
            base64.b64encode(_PNG).decode("ascii"), str(raised.exception)
        )


class ParseUserPromptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        _write_png(self.cwd / "pic.png")

    def test_disabled_is_identity_and_reads_nothing(self):
        message = parse_user_prompt(
            "@missing.png describe",
            cwd=self.cwd,
            enabled=False,
            enable_workspace=True,
        )
        self.assertEqual(message, Message(role="user", content="@missing.png describe"))

    def test_enabled_without_references_is_identity(self):
        message = parse_user_prompt(
            "plain text", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(message.content, "plain text")

    def test_enabled_builds_parts_in_order(self):
        message = parse_user_prompt(
            "@pic.png describe it",
            cwd=self.cwd,
            enabled=True,
            enable_workspace=True,
        )
        self.assertIsInstance(message.content, tuple)
        self.assertIsInstance(message.content[0], MediaPart)
        self.assertEqual(message.content[1], TextPart("describe it"))
        self.assertEqual(message.content_text, "describe it")

    def test_attachment_only_and_escape(self):
        message = parse_user_prompt(
            "@pic.png", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(len(message.content), 1)
        escaped = parse_user_prompt(
            "@@literal", cwd=self.cwd, enabled=True, enable_workspace=True
        )
        self.assertEqual(escaped.content, "@literal")


class TextPasteResolveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)

    def resolve(self, *references, **kwargs):
        return resolve_content(
            references, cwd=self.cwd, enable_workspace=True, **kwargs
        )

    def test_text_files_become_rstripped_text_parts(self):
        (self.cwd / "a.txt").write_bytes(b"  indented\r\nline two \t\r\n\n")
        (self.cwd / "b.md").write_text("# Title\n\nbody\n\n\n", encoding="utf-8")
        (self.cwd / "C.TXT").write_text("upper\n", encoding="utf-8")
        self.assertEqual(
            self.resolve("a.txt", "b.md", "C.TXT"),
            (
                TextPart("  indented\r\nline two"),
                TextPart("# Title\n\nbody"),
                TextPart("upper"),
            ),
        )

    def test_text_suffix_is_never_sniffed_as_an_image(self):
        _write_png(self.cwd / "shot.txt")
        with self.assertRaisesRegex(AttachmentError, r"not valid UTF-8: shot\.txt$"):
            self.resolve("shot.txt")

    def test_invalid_utf8_is_rejected_without_echoing_bytes(self):
        (self.cwd / "bad.txt").write_bytes(b"secret \xff tail")
        with self.assertRaises(AttachmentError) as raised:
            self.resolve("bad.txt")
        self.assertEqual(
            str(raised.exception), "text attachment is not valid UTF-8: bad.txt"
        )

    def test_empty_text_is_rejected(self):
        (self.cwd / "empty.md").write_bytes(b"")
        (self.cwd / "blank.txt").write_text(" \n\t\r\n", encoding="utf-8")
        for reference in ("empty.md", "blank.txt"):
            with self.subTest(reference=reference):
                with self.assertRaises(AttachmentError) as raised:
                    self.resolve(reference)
                self.assertEqual(
                    str(raised.exception), f"text attachment is empty: {reference}"
                )

    def test_limits(self):
        (self.cwd / "a.txt").write_text("12345", encoding="utf-8")
        _write_png(self.cwd / "pic.png")
        for limits in (ContentLimits(max_text_bytes=4), ContentLimits(max_item_bytes=4)):
            with self.subTest(limits=limits):
                with self.assertRaisesRegex(
                    AttachmentError, r"too large \(5 bytes; limit 4\): a\.txt$"
                ):
                    self.resolve("a.txt", limits=limits)
        # The text cap does not apply to images.
        (image,) = self.resolve("pic.png", limits=ContentLimits(max_text_bytes=4))
        self.assertIsInstance(image, MediaPart)
        with self.assertRaisesRegex(AttachmentError, "too many"):
            self.resolve("a.txt", "a.txt", limits=ContentLimits(max_items=1))
        with self.assertRaisesRegex(AttachmentError, "in total"):
            self.resolve("a.txt", "a.txt", limits=ContentLimits(max_total_bytes=9))
        self.assertEqual(
            self.resolve("a.txt", "a.txt", limits=ContentLimits(max_total_bytes=10)),
            (TextPart("12345"), TextPart("12345")),
        )

    def test_workspace_containment(self):
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside = Path(outside_dir.name) / "out.txt"
        outside.write_text("outside\n", encoding="utf-8")
        with self.assertRaisesRegex(AttachmentError, "outside --cwd"):
            self.resolve(str(outside))
        self.assertEqual(
            resolve_content((str(outside),), cwd=self.cwd, enable_workspace=False),
            (TextPart("outside"),),
        )

    def test_symlinks_are_classified_by_their_target(self):
        (self.cwd / "README.md").write_text("# readme\n", encoding="utf-8")
        (self.cwd / ".env").write_text("SECRET=1\n", encoding="utf-8")
        _symlink(self.cwd / "README", "README.md")
        _symlink(self.cwd / "notes.txt", ".env")
        self.assertEqual(self.resolve("README"), (TextPart("# readme"),))
        with self.assertRaises(AttachmentError) as raised:
            self.resolve("notes.txt")
        self.assertEqual(
            str(raised.exception),
            "attachment is not a supported image or text file (.md, .txt): "
            "notes.txt (resolves to .env)",
        )

    def test_unresolvable_paths_are_attachment_errors(self):
        # These used to escape as RuntimeError (loop) and plain ValueError (NUL).
        _symlink(self.cwd / "loop1.txt", "loop2.txt")
        _symlink(self.cwd / "loop2.txt", "loop1.txt")
        for reference in ("loop1.txt", "a\x00b.txt", "a\x00b.png"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(
                    AttachmentError, "invalid attachment path|cannot read attachment"
                ):
                    self.resolve(reference)


class TextSuffixesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        (self.cwd / "x.log").write_text("log line\n", encoding="utf-8")
        (self.cwd / "x.md").write_text("# md\n", encoding="utf-8")
        (self.cwd / "a.txt").write_text("text\n", encoding="utf-8")

    def resolve(self, reference, text_suffixes):
        return resolve_content(
            (reference,),
            cwd=self.cwd,
            enable_workspace=True,
            text_suffixes=text_suffixes,
        )

    def test_default_is_md_and_txt(self):
        self.assertEqual(DEFAULT_TEXT_SUFFIXES, frozenset({".md", ".txt"}))

    def test_suffixes_replace_the_default(self):
        self.assertEqual(self.resolve("x.log", {".log"}), (TextPart("log line"),))
        with self.assertRaisesRegex(
            AttachmentError, r"image or text file \(\.log\): x\.md$"
        ):
            self.resolve("x.md", {".log"})
        self.assertEqual(self.resolve("a.txt", (".TXT",)), (TextPart("text"),))
        with self.assertRaisesRegex(AttachmentError, r"not a supported image: a\.txt$"):
            self.resolve("a.txt", ())
        message = parse_user_prompt(
            "@x.log q",
            cwd=self.cwd,
            enabled=True,
            enable_workspace=True,
            text_suffixes={".log"},
        )
        self.assertEqual(message, Message(role="user", content="log line\n\nq"))

    def test_invalid_suffixes_are_configuration_errors(self):
        for suffixes, error in (
            ({"txt"}, ValueError),
            ({".tar.gz"}, ValueError),
            ({""}, ValueError),
            ("txt", TypeError),
            ({1}, TypeError),
        ):
            with self.subTest(suffixes=suffixes):
                with self.assertRaises(error) as raised:
                    self.resolve("a.txt", suffixes)
                self.assertNotIsInstance(raised.exception, AttachmentError)
                # Checked up front, even when nothing would be resolved.
                with self.assertRaises(error):
                    parse_user_prompt(
                        "plain",
                        cwd=self.cwd,
                        enabled=False,
                        enable_workspace=True,
                        text_suffixes=suffixes,
                    )


class TextPasteParseTests(unittest.TestCase):
    A = "alpha\n  indented"
    B = "beta"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        (self.cwd / "a.txt").write_text(self.A + "\n\n", encoding="utf-8")
        (self.cwd / "b.txt").write_text(self.B + "  \t\n", encoding="utf-8")
        _write_png(self.cwd / "p.png")

    def parse(self, prompt, *, enabled=True):
        return parse_user_prompt(
            prompt, cwd=self.cwd, enabled=enabled, enable_workspace=True
        )

    def test_requested_examples(self):
        self.assertEqual(self.parse("@a.txt"), Message(role="user", content=self.A))
        for prompt in ("@a.txt @b.txt", "@a.txt\n@b.txt\n"):
            with self.subTest(prompt=prompt):
                message = self.parse(prompt)
                self.assertEqual(
                    message, Message(role="user", content=self.A + "\n\n" + self.B)
                )
                self.assertFalse(message.has_media)

    def test_typed_text_joins_like_a_file(self):
        self.assertEqual(
            self.parse("@a.txt @b.txt summarize  both ").content,
            self.A + "\n\n" + self.B + "\n\nsummarize  both ",
        )

    def test_mixed_with_images_keeps_order_and_merges_neighbours(self):
        first, image, last = self.parse("@a.txt @p.png q").content
        self.assertEqual((first, last), (TextPart(self.A), TextPart("q")))
        self.assertIsInstance(image, MediaPart)
        image, text = self.parse("@p.png @a.txt q").content
        self.assertIsInstance(image, MediaPart)
        self.assertEqual(text, TextPart(self.A + "\n\nq"))

    def test_pasted_text_is_not_expanded_again(self):
        (self.cwd / "c.md").write_text("@p.png\n/quit\n@@x\n", encoding="utf-8")
        self.assertEqual(self.parse("@c.md").content, "@p.png\n/quit\n@@x")

    def test_escape_and_flag_off(self):
        self.assertEqual(self.parse("@a.txt @@b.txt").content, self.A + "\n\n@b.txt")
        self.assertEqual(
            self.parse("@missing.txt", enabled=False),
            Message(role="user", content="@missing.txt"),
        )


class MessageModelTests(unittest.TestCase):
    def test_string_content_is_unchanged(self):
        message = Message(role="user", content="hello")
        self.assertEqual(message.content, "hello")
        self.assertEqual(message.content_text, "hello")
        self.assertEqual(message.parts, (TextPart("hello"),))
        self.assertFalse(message.has_media)
        self.assertFalse(hasattr(message, "text"))

    def test_text_only_tuple_canonicalizes(self):
        message = Message(role="user", content=(TextPart("a"), TextPart("b")))
        self.assertEqual(message.content, "a\nb")
        self.assertFalse(message.has_media)

    def test_mixed_tuple_preserved_in_order(self):
        parts = (MediaPart("data:image/png;base64,AA"), TextPart("look"))
        message = Message(role="user", content=parts)
        self.assertEqual(message.content, parts)
        self.assertEqual(message.content_text, "look")
        self.assertTrue(message.has_media)

    def test_non_user_media_rejected(self):
        with self.assertRaisesRegex(ValueError, "require role 'user'"):
            Message(role="assistant", content=(MediaPart("x"),))

    def test_invalid_content_rejected(self):
        for content in (None, True, 123, [], {}, [{"type": "text", "text": "x"}]):
            with self.subTest(content=content):
                with self.assertRaisesRegex(TypeError, "content must be a string"):
                    Message(role="user", content=content)


class CodecTests(unittest.TestCase):
    def test_string_message_round_trips_unchanged(self):
        for role in ("user", "assistant"):
            message = Message(role=role, content="hello")
            self.assertEqual(
                interaction_item_to_dict(message),
                {"type": "message", "role": role, "content": "hello"},
            )

    def test_media_message_round_trips_in_memory_shape(self):
        message = Message(
            role="user",
            content=(MediaPart("data:image/png;base64,AA"), TextPart("look")),
        )
        encoded = interaction_item_to_dict(message)
        self.assertEqual(
            encoded,
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "media", "source_uri": "data:image/png;base64,AA"},
                    {"type": "text", "text": "look"},
                ],
            },
        )
        self.assertEqual(interaction_item_from_dict(encoded), message)

    def test_attachment_only_omits_text_part(self):
        message = Message(role="user", content=(MediaPart("http://x/y.png"),))
        encoded = interaction_item_to_dict(message)
        self.assertEqual(
            encoded["content"],
            [{"type": "media", "source_uri": "http://x/y.png"}],
        )
        self.assertEqual(interaction_item_from_dict(encoded), message)

    def test_text_only_array_loads_as_string(self):
        item = interaction_item_from_dict(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        )
        self.assertEqual(item, Message(role="user", content="hi"))

    def test_invalid_arrays_rejected(self):
        for content, message in (
            ([], "must not be empty"),
            ([{"type": "input_image", "image_url": "x"}], "unsupported"),
            ([{"type": "nope"}], "unsupported"),
            ([{"type": "text"}], "must be a string"),
            ([{"type": "media"}], "source_uri"),
        ):
            with self.subTest(content=content):
                with self.assertRaisesRegex(SaveError, message):
                    interaction_item_from_dict(
                        {"type": "message", "role": "user", "content": content}
                    )


class EncoderTests(unittest.TestCase):
    def test_responses_parts_and_role_text(self):
        context = InteractionContext(
            (
                Message(
                    role="user",
                    content=(MediaPart("data:image/png;base64,AA"), TextPart("look")),
                ),
                Message(role="assistant", content="done"),
            )
        )
        encoded = _encode_context_items(context.model_items(), system_role="developer")
        self.assertEqual(
            encoded,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AA"},
                        {"type": "input_text", "text": "look"},
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        )

    def test_chat_completions_parts(self):
        context = InteractionContext(
            (Message(role="user", content=(MediaPart("https://x/y.png"), TextPart("q"))),)
        )
        self.assertEqual(
            _encode_context_messages(context.model_items()),
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                        {"type": "text", "text": "q"},
                    ],
                }
            ],
        )

    def test_string_chat_completions_unchanged(self):
        context = InteractionContext((Message(role="user", content="hi"),))
        self.assertEqual(
            _encode_context_messages(context.model_items()),
            [{"role": "user", "content": "hi"}],
        )

    def test_messages_rejects_media(self):
        context = InteractionContext((Message(role="user", content=(MediaPart("x"),)),))
        with self.assertRaisesRegex(ModelConfigurationError, "not supported"):
            _encode_context(context.model_items())

    def test_content_item_to_responses(self):
        self.assertEqual(
            content_item_to_responses(TextPart("t"), "assistant"),
            {"type": "output_text", "text": "t"},
        )
        self.assertEqual(
            content_item_to_responses(TextPart("t"), "user"),
            {"type": "input_text", "text": "t"},
        )
        self.assertEqual(
            content_item_to_responses(MediaPart("s"), "user"),
            {"type": "input_image", "image_url": "s"},
        )


class OptionParserTests(unittest.TestCase):
    def test_option_forms(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                self.assertIs(
                    frontend._build_parser()
                    .parse_args([])
                    .enable_experimental_media,
                    False,
                )
                self.assertIs(
                    frontend._build_parser()
                    .parse_args(["--enable-experimental-media"])
                    .enable_experimental_media,
                    True,
                )
                self.assertIs(
                    frontend._build_parser()
                    .parse_args(["--enable-experimental-media=False"])
                    .enable_experimental_media,
                    False,
                )


class HeadlessMediaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "session.jsonl"
        _write_png(self.root / "pic.png")

    def run_main(self, argv, *outcomes):
        model = _Model(self.path, *outcomes)
        stderr = io.StringIO()
        with mock.patch.object(cli, "build_model", return_value=model), \
                mock.patch.object(
                    cli, "PosixTerminal",
                    side_effect=AssertionError("TUI constructed"),
                ), \
                redirect_stderr(stderr):
            code = cli.main(
                ["--headless", "--cwd", str(self.root), "--save", str(self.path), *argv]
            )
        return code, model, stderr.getvalue()

    def _saved_user(self):
        items = load_interaction_save(self.path).items
        return [
            item
            for item in items
            if isinstance(item, Message) and item.role == "user"
        ]

    def test_flag_on_saves_responses_array_and_sends_parts(self):
        code, model, stderr = self.run_main(
            ["--enable-experimental-media", "--prompt", "@pic.png describe"],
            _answer("ok"),
        )
        self.assertEqual(code, 0, stderr)
        (user,) = self._saved_user()
        self.assertIsInstance(user.content, tuple)
        self.assertTrue(user.content[0].source_uri.startswith("data:image/png;base64,"))
        self.assertEqual(user.content[-1], TextPart("describe"))
        sent = model.calls[0][0].items
        sent_user = [
            item
            for item in sent
            if isinstance(item, Message) and item.role == "user"
        ][0]
        self.assertEqual(sent_user.content_text, "describe")
        self.assertTrue(sent_user.has_media)

    def test_flag_off_keeps_literal_text_and_reads_nothing(self):
        code, _model, stderr = self.run_main(
            ["--prompt", "@missing.png describe"], _answer("ok")
        )
        self.assertEqual(code, 0, stderr)
        (user,) = self._saved_user()
        self.assertEqual(user.content, "@missing.png describe")

    def test_attachment_error_exits_nonzero_without_sampling(self):
        code, model, stderr = self.run_main(
            ["--enable-experimental-media", "--prompt", "@missing.png describe"],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("cannot read attachment", stderr)
        self.assertEqual(self._saved_user(), [])

    def test_messages_api_rejected_before_sampling(self):
        code, model, stderr = self.run_main(
            [
                "--enable-experimental-media",
                "--endpoint-api",
                "messages",
                "--model",
                "claude-test",
                "--max-output-tokens",
                "77",
                "--prompt",
                "@pic.png describe",
            ],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("does not support media", stderr)

    def test_responses_api_accepts_media(self):
        code, model, stderr = self.run_main(
            [
                "--enable-experimental-media",
                "--endpoint-api",
                "responses",
                "--model",
                "gpt-test",
                "--prompt",
                "@pic.png describe",
            ],
            _answer("ok"),
        )
        self.assertEqual(code, 0, stderr)
        (sent_user,) = [
            item
            for item in model.calls[0][0].items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertTrue(sent_user.has_media)

    def test_text_paste_works_with_messages_api(self):
        (self.root / "a.txt").write_text("alpha\n", encoding="utf-8")
        (self.root / "b.md").write_text("# beta\n\n", encoding="utf-8")
        code, model, stderr = self.run_main(
            [
                "--enable-experimental-media",
                "--endpoint-api",
                "messages",
                "--model",
                "claude-test",
                "--max-output-tokens",
                "77",
                "--prompt",
                "@a.txt @b.md",
            ],
            _answer("ok"),
        )
        self.assertEqual(code, 0, stderr)
        expected = Message(role="user", content="alpha\n\n# beta")
        self.assertEqual(self._saved_user(), [expected])
        sent = [
            item
            for item in model.calls[0][0].items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertEqual(sent, [expected])
        # Saved as a plain string, exactly like typed text.
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertIn(
            {"type": "message", "role": "user", "content": "alpha\n\n# beta"},
            records,
        )

    def test_messages_api_still_rejects_text_mixed_with_images(self):
        (self.root / "a.txt").write_text("alpha\n", encoding="utf-8")
        code, model, stderr = self.run_main(
            [
                "--enable-experimental-media",
                "--endpoint-api",
                "messages",
                "--model",
                "claude-test",
                "--max-output-tokens",
                "77",
                "--prompt",
                "@a.txt @pic.png describe",
            ],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertIn("does not support media", stderr)

    def test_symlink_loop_is_reported_as_an_attachment_error(self):
        _symlink(self.root / "loop1.txt", "loop2.txt")
        _symlink(self.root / "loop2.txt", "loop1.txt")
        code, model, stderr = self.run_main(
            ["--enable-experimental-media", "--prompt", "@loop1.txt hi"],
            _answer("ok"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(model.calls, [])
        self.assertRegex(
            stderr, r"\[cli\] (invalid attachment path|cannot read attachment)"
        )
        self.assertNotIn("RuntimeError", stderr)
        self.assertEqual(self._saved_user(), [])


class InteractiveAttachmentErrorTests(_ControllerTestCase):
    async def test_bad_reference_restores_the_draft_and_keeps_the_session(self):
        root = self.path.parent
        _symlink(root / "loop1.txt", "loop2.txt")
        _symlink(root / "loop2.txt", "loop1.txt")
        (root / "a.txt").write_text("alpha\n", encoding="utf-8")
        draft = "@loop1.txt hi"
        restored = []
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status != "idle":
                return
            if step == 0:
                step = 1
                terminal.submit(draft)
            elif step == 1 and any(
                "invalid attachment path" in item.text
                or "cannot read attachment" in item.text
                for item in terminal.items
            ):
                step = 2
                restored.append(editor.text)
                terminal.submit("@a.txt")
            elif step == 2 and any(
                item.text == "[assistant] ok" for item in terminal.items
            ):
                step = 3
                terminal.key("c-d")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer("ok"))
        code = await self._run(
            model, terminal, ["--enable-experimental-media", "--cwd", str(root)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(restored, [draft])
        self.assertEqual(len(model.calls), 1)
        users = [
            item
            for item in model.calls[0][0].items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertEqual(users, [Message(role="user", content="alpha")])


class DisplayTests(unittest.TestCase):
    def test_media_message_renders_without_payload(self):
        from pythia.interaction import render_interaction_items

        message = Message(
            role="user",
            content=(MediaPart("data:image/png;base64," + "A" * 400), TextPart("look")),
        )
        blocks = [item.text for item in render_interaction_items((message,))]
        self.assertEqual(blocks[0], "[user] look")
        self.assertTrue(blocks[1].startswith("[user] [image] image/png"))
        self.assertNotIn("A" * 8, "".join(blocks))

    def test_text_only_message_unchanged(self):
        from pythia.interaction import render_interaction_items

        message = Message(role="user", content="hello")
        self.assertEqual(
            [item.text for item in render_interaction_items((message,))],
            ["[user] hello"],
        )


class DemoMediaTests(unittest.TestCase):
    def test_run_converts_leading_references(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        _write_png(root / "pic.png")
        path = root / "interaction.jsonl"

        class Model:
            def __init__(self):
                self.calls = []

            def sample(self, context, *, tools=(), sample_params=None):
                self.calls.append(context.copy())
                return _answer("ok")

        model = Model()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            text = demo.run(
                model,
                Environment(),
                prompt="@pic.png hello",
                save_path=path,
                enable_media=True,
                enable_workspace=True,
                cwd=root,
            )
        self.assertEqual(text, "ok")
        (user,) = [
            item
            for item in load_interaction_save(path).items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertIsInstance(user.content, tuple)
        self.assertEqual(user.content_text, "hello")

    def test_run_pastes_text_files(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "a.txt").write_text("alpha\n\n", encoding="utf-8")
        path = root / "interaction.jsonl"
        model = _Model(path, _answer("ok"))
        with redirect_stdout(io.StringIO()):
            text = demo.run(
                model,
                Environment(),
                prompt="@a.txt",
                save_path=path,
                enable_media=True,
                enable_workspace=True,
                cwd=root,
            )
        self.assertEqual(text, "ok")
        (user,) = [
            item
            for item in load_interaction_save(path).items
            if isinstance(item, Message) and item.role == "user"
        ]
        self.assertEqual(user, Message(role="user", content="alpha"))


if __name__ == "__main__":
    unittest.main()
