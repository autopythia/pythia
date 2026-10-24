"""Text from model providers: lone surrogates repaired before it is kept,
unexpected values bounded when an error message quotes them, and characters
that show as nothing or act on a terminal escaped on screen and in files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pythia.interaction import Init, InteractionContext, Message
from pythia.interaction import load_interaction_save, save_interaction_save
from pythia.interaction._cli_editor import safe_text
from pythia.interaction._provider_text import QUOTED_VALUE_CHARS
from pythia.interaction._provider_text import escaped_character, json_text_escaped
from pythia.interaction._provider_text import loads_json, quoted, repaired_json, repaired_json_text
from pythia.interaction._provider_text import repaired_text


class RepairTests(unittest.TestCase):
    def test_lone_surrogates_become_the_replacement_character(self):
        for text, expected in (
            ("ok \udc80", "ok \ufffd"),
            ("\ud800", "\ufffd"),
            ("\ude00\ud83d", "\ufffd\ufffd"),  # halves in the wrong order stay lone
            ("\ud83d\ude00", "\U0001f600"),  # halves next to each other are joined
        ):
            with self.subTest(text=text.encode("utf-16-le", "surrogatepass")):
                self.assertEqual(repaired_text(text), expected)
        plain = "plain é 😀"
        self.assertIs(repaired_text(plain), plain)
        self.assertEqual(
            repaired_json({"k\ud800": ["x\udc80", 1, None, {"n": "\udfff"}], "t": True}),
            {"k\ufffd": ["x\ufffd", 1, None, {"n": "\ufffd"}], "t": True},
        )

    def test_a_provider_s_json_decodes_without_them_and_saves(self):
        for text in ('{"t": "ok \\udc80", "e": "\\ud83d\\ude00"}',
                     b'{"t": "ok \\uDC80", "e": "\\ud83d\\ude00"}'):
            with self.subTest(text=text):
                self.assertEqual(loads_json(text), {"t": "ok \ufffd", "e": "\U0001f600"})
        # An escaped backslash before \u is text, not an escape.
        self.assertEqual(loads_json('{"p": "C:\\\\udir"}'), {"p": "C:\\udir"})
        with self.assertRaises(json.JSONDecodeError):
            loads_json("{")
        # The bug: UTF-8 can't encode a lone surrogate, so saving one failed.
        message = Message("assistant", loads_json('"ok \\udc80"'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            save_interaction_save(path, InteractionContext((Init(model="m"), message)))
            self.assertEqual(load_interaction_save(path).items[-1],
                             Message("assistant", "ok \ufffd"))

    def test_escaped_lone_surrogates_in_json_text_become_replacement_escapes(self):
        # As tool-call arguments come, as JSON text: decoded, an escaped lone
        # surrogate would reach the tool. The rest stays byte for byte.
        for text, expected in (
            ('{"cmd":  "echo \\udc80"}', '{"cmd":  "echo \\ufffd"}'),
            ('{"upper": "\\uDC80"}', '{"upper": "\\ufffd"}'),
            ('["\\ud800 \\uDC80", "\\ude00\\ud83d"]', '["\\ufffd \\ufffd", "\\ufffd\\ufffd"]'),
            ('{"k\\udfff": "\\ud83d\\ud83d\\ude00"}', '{"k\\ufffd": "\\ufffd\\ud83d\\ude00"}'),
            ('{"p": "\\\\\\udc80 \\u0041\\udc80 \\\\\\ud800"}', '{"p": "\\\\\\ufffd \\u0041\\ufffd \\\\\\ufffd"}'),
        ):
            with self.subTest(text=text):
                self.assertEqual(repaired_json_text(text), expected)
                self.assertEqual(json.loads(expected), repaired_json(json.loads(text)))
        # A pair, an escaped backslash before u, plain text, and text that
        # isn't JSON stay as they are.
        for kept in ('{"e": "\\ud83d\\uDE00", "p": "C:\\\\udc80 C:\\\\ud800"}', '{"cmd": "plain é"}',
                     '{"cut": "\\udc80', '"\\udc80" and more'):
            with self.subTest(kept=kept):
                self.assertEqual(repaired_json_text(kept), kept)


class QuotedTests(unittest.TestCase):
    def test_an_unexpected_value_is_quoted_as_a_bounded_repr(self):
        self.assertEqual(QUOTED_VALUE_CHARS, 80)
        self.assertEqual(quoted("text_v2"), "'text_v2'")
        self.assertEqual(quoted(None), "None")
        # repr shows controls and bidi characters as escapes.
        self.assertEqual(quoted("\x1b[2J\u202e"), "'\\x1b[2J\\u202e'")
        for value in ("x" * 100_000, {"type": "y" * 500}, ["z"] * 500):
            with self.subTest(value=type(value).__name__):
                text = quoted(value)
                self.assertEqual(len(text), 80)
                self.assertTrue(text.endswith("..."), text)


class EscapeTests(unittest.TestCase):
    def test_json_text_escapes_what_shows_as_nothing_or_acts_on_a_terminal(self):
        value = {"text\u202e": ["del\x7f nel\x85 csi\x9b shy\u00ad alm\u061c zwsp\u200b lrm\u200e "
                                "ls\u2028 ps\u2029 rlo\u202e lri\u2066 wj\u2060 bom\ufeff "
                                "tag\U000e0041 sym\U0001d173"],
                 "kept": "zwnj\u200c zwj\u200d é – 漢 😀 👨\u200d👩 \x1b"}
        text = json_text_escaped(json.dumps(value, ensure_ascii=False))
        self.assertEqual(text, (
            '{"text\\u202e": ["del\\u007f nel\\u0085 csi\\u009b shy\\u00ad alm\\u061c zwsp\\u200b '
            'lrm\\u200e ls\\u2028 ps\\u2029 rlo\\u202e lri\\u2066 wj\\u2060 bom\\ufeff '
            'tag\\udb40\\udc41 sym\\ud834\\udd73"], '  # beyond U+FFFF: its surrogate pair
            '"kept": "zwnj\u200c zwj\u200d é – 漢 😀 👨\u200d👩 \\u001b"}'))
        self.assertEqual(json.loads(text), value)  # the same value for any reader
        plain = '{"a": ["b", 1]}'
        self.assertIs(json_text_escaped(plain), plain)
        # Indented JSON keeps its layout.
        self.assertEqual(json_text_escaped(json.dumps({"a": "\u2028"}, ensure_ascii=False, indent=1)),
                         '{\n "a": "\\u2028"\n}')
        # A lone surrogate stays: a save refuses it, as UTF-8 can't encode one.
        self.assertEqual(json_text_escaped(json.dumps("x\ud800", ensure_ascii=False)), '"x\ud800"')

    def test_the_screen_and_the_files_escape_the_same_characters(self):
        astral = (0x1F600, 0x1D173, 0x110BD, 0xE0001, 0xE0041, 0xE007F, 0xE0100, 0x10FFFF)
        for code in (*range(0x10000), *astral):
            char = chr(code)
            escaped = escaped_character(char)
            self.assertEqual(safe_text(char) != char, escaped and char not in "\n\t", hex(code))
            if code >= 0x20 and not 0xD800 <= code <= 0xDFFF:  # JSON escapes C0 itself
                written = json_text_escaped(json.dumps(char, ensure_ascii=False))
                self.assertEqual(written != json.dumps(char, ensure_ascii=False), escaped, hex(code))
                self.assertEqual(written.isascii(), escaped or char.isascii(), hex(code))
                self.assertEqual(json.loads(written), char)


if __name__ == "__main__":
    unittest.main()
