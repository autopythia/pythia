"""Markdown in assistant messages, styled for a terminal (display.assistant)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import os
import re
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    DisplayItem, Environment, MediaPart, Message, Reasoning, TextPart, auto, cli, demo,
)
from pythia.interaction._cli_editor import Editor, cell_width
from pythia.interaction._cli_terminal import PosixTerminal
from pythia.interaction._markdown_display import style_markdown
from pythia.interaction.display import render_interaction_items, safe_display_item
from pythia.interaction.model_catalog import DisplayDefaults
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import prepare_namespace, render_model_catalog
from pythia.interaction.runtime_config import ConfigError, InteractionConfig
from pythia.interaction.runtime_config import parse_config_literal
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


_NAMES = {"\x1b[1m": "<b>", "\x1b[22m": "</b>", "\x1b[4m": "<u>", "\x1b[24m": "</u>",
          "\x1b[90m": "<g>", "\x1b[39m": "</g>", "\x1b[0m": "<0>"}
_SGR = re.compile(r"\x1b\[[0-9;]*m")
# A table like the catalog summary's: two columns of identifiers, one of prose.
_TABLE = "\n".join((
    "| Catalog key | What it sets | Launch option |",
    "|---|---|---|",
    "| `endpoint.prompt_caching` | `5m` (the default), `1h` or `none`; Messages only "
    "| `--endpoint-prompt-caching` |",
    "| `limits.max_output_tokens` | the request's output limit; required on **Messages** "
    "| `--max-output-tokens` |",
))


def named(text):
    """``text`` with each escape sequence named; any other escape fails."""
    for code, name in _NAMES.items():
        text = text.replace(code, name)
    if "\x1b" in text:
        raise AssertionError(f"an unexpected escape sequence: {text!r}")
    return text


def plain(text):
    return _SGR.sub("", text)


def columns(line):
    return sum(cell_width(char) for char in plain(line))


def balanced(test, text):
    """Each line closes the styles that it opens, and nothing resets them all."""
    test.assertNotIn("\x1b[0m", text)
    for line in named(text).split("\n"):
        for on, off in (("<b>", "</b>"), ("<u>", "</u>"), ("<g>", "</g>")):
            test.assertEqual(line.count(on), line.count(off), line)


class HeadingTests(unittest.TestCase):
    def test_markers_stay_gray_and_the_text_is_bold_and_underlined_for_one_and_two(self):
        cases = {
            "# Title": "<g>#</g> <b><u>Title</u></b>",
            "## Results ##": "<g>##</g> <b><u>Results</u></b> <g>##</g>",
            "### Third": "<g>###</g> <b>Third</b>",
            "###### Six  ": "<g>######</g> <b>Six</b>  ",
            "   ## Indented": "   <g>##</g> <b><u>Indented</u></b>",
            "##": "<g>##</g>",
            "# C#": "<g>#</g> <b><u>C#</u></b>",
            # Bold inside is already on: its markers just go.
            "## **Bold** and `**code**`": "<g>##</g> <b><u>Bold and `**code**`</u></b>",
        }
        for text, shown in cases.items():
            with self.subTest(text=text):
                self.assertEqual(named(style_markdown(text)), shown)

    def test_a_heading_on_the_first_line_goes_below_the_label_after_a_blank_line(self):
        prefix = "[assistant] "
        self.assertEqual(named(style_markdown("## T\ntext", prefix=prefix)),
                         "[assistant]\n\n<g>##</g> <b><u>T</u></b>\ntext")
        self.assertEqual(named(style_markdown("### Only", prefix=prefix)),
                         "[assistant]\n\n<g>###</g> <b>Only</b>")
        # Only the first line counts: a later heading, or after the message's
        # own line break, leaves the label where it is.
        self.assertEqual(named(style_markdown("Intro\n## T", prefix=prefix)),
                         "[assistant] Intro\n<g>##</g> <b><u>T</u></b>")
        self.assertEqual(named(style_markdown("\n## T", prefix=prefix)),
                         "[assistant] \n<g>##</g> <b><u>T</u></b>")
        for text in ("#tag", "    # four spaces", "```\n# in a fence\n```"):
            with self.subTest(text=text):
                self.assertEqual(style_markdown(text, prefix=prefix), prefix + text)
        # Without a prefix, nothing goes before it.
        self.assertEqual(named(style_markdown("## T")), "<g>##</g> <b><u>T</u></b>")

    def test_what_is_not_a_heading_shows_as_written(self):
        for text in ("#tag", "#5 bolt", "####### seven", "    # four spaces", "Title\n====="):
            with self.subTest(text=text):
                self.assertEqual(style_markdown(text), text)


class BoldTests(unittest.TestCase):
    def test_the_markers_go_and_the_text_is_bold(self):
        cases = {
            "**Bold** text": "<b>Bold</b> text",
            "a **b** c **d**": "a <b>b</b> c <b>d</b>",
            "a**b**c": "a<b>b</b>c",
            "**a **b**": "**a <b>b</b>",  # a closer pairs with the nearest opener
            "x **y**.": "x <b>y</b>.",
            '**"quoted"**': '<b>"quoted"</b>',
            "**`code` inside**": "<b>`code` inside</b>",
            "- **Key:** value": "- <b>Key:</b> value",
        }
        for text, shown in cases.items():
            with self.subTest(text=text):
                self.assertEqual(named(style_markdown(text)), shown)

    def test_what_isnt_bold_shows_as_written(self):
        for text in ("unmatched **bold", "** x**", "***x***", "****", "\\*\\*no\\*\\*",
                     "\\**no**", "`**code**` and ``a ** b``", "__init__ and __bold__",
                     "*italic*"):
            with self.subTest(text=text):
                self.assertEqual(style_markdown(text), text)


class FenceTests(unittest.TestCase):
    def test_nothing_in_a_fence_is_styled(self):
        fenced = "```python\n**not bold**\n# not a heading\n| a | b |\n|---|---|\n```"
        self.assertEqual(named(style_markdown(fenced + "\n**bold**")),
                         fenced + "\n<b>bold</b>")
        tilde = "~~~\n**x**\n```\n~~~~\n**y**"  # a backtick fence doesn't close it
        self.assertEqual(named(style_markdown(tilde)), "~~~\n**x**\n```\n~~~~\n<b>y</b>")
        longer = "````\n```\n**x**\n````\n**y**"
        self.assertEqual(named(style_markdown(longer)), "````\n```\n**x**\n````\n<b>y</b>")
        self.assertEqual(style_markdown("```\n**open to the end**"), "```\n**open to the end**")

    def test_a_backtick_fence_info_string_has_no_backtick(self):
        self.assertEqual(named(style_markdown("``` a`b\n**x**")), "``` a`b\n<b>x</b>")


class TableTests(unittest.TestCase):
    def test_columns_align_with_a_bold_header_and_gray_borders(self):
        text = "| Key | Value |\n|:----|------:|\n| a | 1 |\n| longer **key** | 12345 |"
        self.assertEqual(named(style_markdown(text)), "\n".join((
            "<g>|</g> <b>Key</b>        <g>|</g> <b>Value</b> <g>|</g>",
            "<g>|:-----------|------:|</g>",
            "<g>|</g> a          <g>|</g>     1 <g>|</g>",
            "<g>|</g> longer <b>key</b> <g>|</g> 12345 <g>|</g>",
        )))

    def test_cells(self):
        text = "\n".join(("a | b", ":-:|---", "x | `p|q`", "only one", "| 1 | 2 | 3 |",
                          "| 日本 | \\| |"))
        lines = plain(style_markdown(text)).split("\n")
        # "only one" has no |: the table ends before it.
        self.assertEqual(lines, [
            "| a | b     |",
            "|:-:|-------|",
            "| x | `p|q` |",
            "only one",
            "| 1 | 2 | 3 |",
            "| 日本 | \\| |",
        ])
        text = "\n".join(("a | b", "---|---", "| 1 | 2 | 3 |", "| 日本 | \\| |", "| x |"))
        lines = plain(style_markdown(text)).split("\n")
        self.assertEqual(lines, [
            "| a    | b  |   |",  # a row's extra cell adds a column
            "|------|----|---|",
            "| 1    | 2  | 3 |",
            "| 日本 | \\| |   |",  # wide characters take two columns
            "| x    |    |   |",  # missing cells are empty
        ])
        self.assertEqual({columns(line) for line in lines}, {17})

    def test_where_a_table_starts_and_ends(self):
        not_tables = (
            "| a | b |\n|---|\n| 1 | 2 |",  # the delimiter row's cells don't match
            "a\n---",  # a setext heading or a rule, not a delimiter row
            "    | a |\n    |---|",  # indented code
            "| a |\n| b |",
        )
        for text in not_tables:
            with self.subTest(text=text):
                self.assertEqual(style_markdown(text), text)
        ends = {
            "| a |\n|---|\n| 1 |\n\n**after**": "\n\n<b>after</b>",
            "| a |\n|---|\n| 1 |\n**after**": "\n<b>after</b>",
            "| a |\n|---|\n| 1 |\n## H | x": "\n<g>##</g> <b><u>H | x</u></b>",
            "| a |\n|---|\n| 1 |\n```\n| b |": "\n```\n| b |",
        }
        for text, tail in ends.items():
            with self.subTest(text=text):
                shown = named(style_markdown(text))
                self.assertTrue(shown.endswith(tail), shown)
                self.assertEqual(shown.count("<g>|---|</g>"), 1)
        self.assertEqual(plain(style_markdown("  | a |\n  |---|\n  | 1 |")),
                         "  | a |\n  |---|\n  | 1 |")

    def test_a_table_on_the_first_line_puts_the_label_on_a_line_of_its_own(self):
        self.assertEqual(plain(style_markdown("| h |\n|---|\n| v |", prefix="[assistant] ")),
                         "[assistant]\n| h |\n|---|\n| v |")
        self.assertEqual(plain(style_markdown("Intro\n| h |\n|---|\n| v |",
                                              prefix="[assistant] ")),
                         "[assistant] Intro\n| h |\n|---|\n| v |")


class FittingTests(unittest.TestCase):
    def test_a_table_that_fits_keeps_its_natural_widths(self):
        natural = style_markdown(_TABLE)
        self.assertEqual({columns(line) for line in natural.split("\n")}, {112})
        self.assertEqual(style_markdown(_TABLE, width=112), natural)
        self.assertEqual(len(natural.split("\n")), 4)

    def test_one_that_doesnt_wraps_its_cells_and_the_widest_columns_give_up_most(self):
        # Every column keeps its longest word, and the prose column gives up
        # the most.
        lines = plain(style_markdown(_TABLE, width=100)).split("\n")
        self.assertEqual({columns(line) for line in lines}, {100})
        self.assertEqual(lines[1], "|" + "|".join("-" * n for n in (28, 39, 29)) + "|")
        self.assertEqual(lines[2:4], [
            "| `endpoint.prompt_caching`  | `5m` (the default), `1h` or `none`;   "
            "| `--endpoint-prompt-caching` |",
            "|                            | Messages only                         "
            "|                             |",
        ])
        # Even the longest words don't fit: the columns share the width fairly,
        # and long words break.
        lines = plain(style_markdown(_TABLE, width=70)).split("\n")
        self.assertEqual({columns(line) for line in lines}, {70})
        self.assertEqual(lines[1], "|" + "|".join("-" * 22 for _ in range(3)) + "|")
        self.assertEqual(lines[2], "| `endpoint.prompt_cac | `5m` (the default),  "
                                   "| `--endpoint-prompt-c |")
        self.assertEqual(lines[3], "| hing`                | `1h` or `none`;      "
                                   "| aching`              |")

    def test_wrapped_bold_closes_and_opens_again_on_each_line(self):
        text = "| a | b |\n|---|---|\n| x | **one two three four** |"
        shown = named(style_markdown(text, width=20))
        self.assertIn("<b>one two</b>", shown)
        self.assertIn("<b>three four</b>", shown)
        balanced(self, style_markdown(text, width=20))

    def test_too_narrow_shows_the_table_as_written(self):
        self.assertEqual(plain(style_markdown(_TABLE, width=12)), _TABLE.replace("**", ""))
        self.assertIn("<b>Messages</b>", named(style_markdown(_TABLE, width=12)))


class SafetyTests(unittest.TestCase):
    def test_every_line_closes_its_styles(self):
        text = "\n".join(("# **T**", "## A `**b**`", "a **b** c", "", _TABLE, "",
                          "| **h** | x |", "|:-:|--:|", "| **one two three** | 4 |"))
        for width in (None, 200, 60, 30, 12):
            with self.subTest(width=width):
                balanced(self, style_markdown(text, width=width))

    def test_content_escapes_stay_visible(self):
        # safe_text has made the model's controls visible: they stay text.
        text = "## \\u001b[31mred\n| \\u001b[2J | x |\n|---|---|\n**\\u202e**"
        shown = named(style_markdown(text, width=40))
        self.assertIn("\\u001b[31mred", shown)
        self.assertIn("\\u001b[2J", shown)
        self.assertIn("<b>\\u202e</b>", shown)


class DisplayItemTests(unittest.TestCase):
    def test_the_marker(self):
        with self.assertRaises(TypeError):
            DisplayItem("[assistant] x", is_markdown="yes")
        with self.assertRaisesRegex(ValueError, "a diff or markdown, not both"):
            DisplayItem("+x", is_diff=True, is_markdown=True)
        self.assertNotEqual(DisplayItem("x"), DisplayItem("x", is_markdown=True))

    def test_the_renderer_marks_only_an_assistant_messages_text(self):
        items = render_interaction_items((
            Message("assistant", "**a**"),
            Message("user", "**u**"),
            Reasoning("", summary=("**r**",)),
            Message("user", (TextPart("**t**"), MediaPart("https://example.test/a.png"))),
        ))
        self.assertEqual([(item.text, item.is_markdown) for item in items], [
            ("[assistant] **a**", True),
            ("[user] **u**", False),
            ("[reasoning] **r**", False),
            ("[user] **t**", False),
            ("[user] [image] https://example.test/a.png", False),
        ])

    def test_printing_styles_after_the_label_and_fits_tables_inside_the_gutter(self):
        text = "[assistant] ## T\n| a | b |\n|---|---|\n| one two three | 4 |"
        item = DisplayItem(text, label="assistant", is_markdown=True)
        self.assertEqual(str(item), item.render())
        shown = named(item.render(width=18))
        # The heading goes below the label, after a blank line, which has no
        # gutter.
        self.assertTrue(shown.startswith(
            "   <g>⌜<0>[assistant]\n\n    <g>##</g> <b><u>T</u></b>\n"), shown)
        table = [line for line in item.render(width=18).split("\n") if "|" in line]
        self.assertEqual((len(table), {columns(line) for line in table}), (5, {18}))
        self.assertEqual(item.text, text)
        self.assertEqual(named(DisplayItem("[assistant] **x**", label="assistant").render()),
                         "   <g>[<0>[assistant] **x**")

    def test_a_safe_copy_keeps_the_label_and_the_marker(self):
        item = DisplayItem("[assistant] **a**\x1b[2J", label="assistant", is_markdown=True)
        safe = safe_display_item(item)
        self.assertEqual((safe.text, safe.label, safe.is_markdown),
                         ("[assistant] **a**\\u001b[2J", "assistant", True))
        self.assertEqual(named(safe.render()), "   <g>[<0>[assistant] <b>a</b>\\u001b[2J")
        odd = DisplayItem("[a\x1b] x", label="a\x1b", is_markdown=True)
        self.assertIsNone(safe_display_item(odd).label)


class TerminalTests(unittest.TestCase):
    def render(self, item):
        class Output(io.StringIO):
            def fileno(self):
                return 1

        output = Output()
        terminal = PosixTerminal(io.StringIO(), output)
        with mock.patch("os.get_terminal_size", return_value=os.terminal_size((24, 8))):
            terminal.render(Editor(), "idle", (item,))
        return output.getvalue()

    def test_a_marked_item_is_styled_and_fits_the_terminal(self):
        text = "[assistant] **x**\x1b[2J\n| a | b |\n|---|---|\n| one two three | 4 |"
        rendered = self.render(DisplayItem(text, label="assistant", is_markdown=True))
        self.assertIn("\x1b[1mx\x1b[22m\\u001b[2J", rendered)
        self.assertNotIn("\x1b[2J", rendered)
        table = [line for line in plain(rendered).split("\r\n") if "|" in line]
        self.assertEqual({columns(line) for line in table}, {24})
        unmarked = self.render(DisplayItem(text, label="assistant"))
        self.assertIn("[assistant] **x**\\u001b[2J", unmarked)


class CatalogTests(unittest.TestCase):
    def test_the_display_section(self):
        catalog = parse_model_catalog("[display]\nassistant = plain\n")
        self.assertEqual(catalog.display, DisplayDefaults("plain"))
        self.assertIn("[display] assistant = plain", render_model_catalog(catalog))
        self.assertEqual(parse_model_catalog('[display]\nassistant = "markdown"\n').display,
                         DisplayDefaults("markdown"))
        unset = parse_model_catalog("[display]\nassistant = null\n")
        self.assertEqual(unset.display, DisplayDefaults())
        self.assertNotIn("[display]", render_model_catalog(unset))
        with self.assertRaisesRegex(ValueError, r"\[display\] key 'reasoning' in user; "
                                                r"expected assistant"):
            parse_model_catalog("[display]\nreasoning = markdown\n")
        with self.assertRaisesRegex(ValueError, r"\[display\] assistant in user: expected "
                                                r"markdown, plain, or null"):
            parse_model_catalog("[display]\nassistant = fancy\n")
        with self.assertRaises(ValueError):
            DisplayDefaults("fancy")


class SettingTests(unittest.TestCase):
    def namespace(self, **values):
        return SimpleNamespace(model_api="chat-completions", model=None, enable_workspace=True,
                               max_samples=None, max_output_tokens=None,
                               enable_auto_compaction=True, **values)

    def test_the_option_then_the_catalog_then_markdown_and_null_restores_the_catalogs(self):
        cases = ((None, None, "markdown"), (None, "plain", "plain"),
                 ("markdown", "plain", "markdown"), ("plain", None, "plain"))
        for option, catalog, launch in cases:
            with self.subTest(option=option, catalog=catalog):
                config = InteractionConfig.from_namespace(self.namespace(
                    display_assistant=option, catalog_display_assistant=catalog))
                self.assertEqual(config.get("display_assistant"), launch)
                other = "plain" if launch == "markdown" else "markdown"
                self.assertEqual(config.set("display_assistant", other), other)
                self.assertEqual(config.set("display_assistant", None), catalog or "markdown")

    def test_its_values(self):
        self.assertEqual(parse_config_literal("display_assistant", " plain "), "plain")
        self.assertIsNone(parse_config_literal("display_assistant", "null"))
        for text in ("fancy", "true", "1"):
            with self.subTest(text=text), self.assertRaisesRegex(
                    ConfigError, "^display_assistant requires markdown, plain, or null[.]$"):
                parse_config_literal("display_assistant", text)

    def test_the_options_and_the_catalogs_value_through_prepare_namespace(self):
        for parse in (cli._build_parser().parse_args, demo._build_parser().parse_args,
                      auto.build_parser().parse_args):
            with self.subTest(parser=parse):
                self.assertIsNone(parse([]).display_assistant)
                self.assertEqual(parse(["--display-assistant", "plain"]).display_assistant,
                                 "plain")
                with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse(["--display-assistant", "fancy"])
        catalog = parse_model_catalog("[display]\nassistant = plain\n")
        args = prepare_namespace(argparse.Namespace(model_api=None, model=None), catalog)
        self.assertEqual(args.catalog_display_assistant, "plain")


class CLITests(_ControllerTestCase):
    async def test_assistant_messages_are_markdown_unless_the_option_says_plain(self):
        for argv, marked in (([], True), (["--display-assistant", "plain"], False)):
            with self.subTest(argv=argv):
                terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                model = _Model(self.path, _answer("## Done\n**ok**"))
                self.assertEqual(await self._run(model, terminal, [
                    "--resume=False", "--prompt", "go", "--enable-default-tools=False", *argv,
                ]), 0)
                [answer] = [item for item in terminal.items if item.label == "assistant"]
                self.assertEqual(answer.is_markdown, marked)

    async def test_a_config_change_applies_to_the_next_items_shown(self):
        phase = [0]

        def frame(terminal, editor, status):
            if status != "idle":
                return
            texts = [item.text for item in terminal.items]
            if phase[0] == 0:
                phase[0] = 1
                terminal.submit("/config display_assistant plain")
            elif phase[0] == 1 and any("display_assistant = 'plain'" in text for text in texts):
                phase[0] = 2
                terminal.submit("two")
            elif phase[0] == 2 and any("**second**" in text for text in texts):
                terminal.key("c-d")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer("**first**"), _answer("**second**"))
        self.assertEqual(await self._run(model, terminal, [
            "--resume=False", "--prompt", "one", "--enable-default-tools=False",
        ]), 0)
        answers = [(item.text, item.is_markdown) for item in terminal.items
                   if item.label == "assistant"]
        self.assertEqual(answers, [("[assistant] **first**", True),
                                   ("[assistant] **second**", False)])


class AutoAndDemoTests(unittest.TestCase):
    def test_auto_keeps_the_marker_under_its_labels_unless_plain(self):
        events = (auto._Event(1, render_interaction_items((Message("assistant", "**a**"),))),)
        for session, marked in ((SimpleNamespace(names={1: "main"}), True),
                                (SimpleNamespace(names={1: "main"}, display_assistant="plain"),
                                 False)):
            with self.subTest(marked=marked):
                [item] = auto._display_events(session, events)
                self.assertEqual((item.label, item.is_markdown),
                                 ("#1 (main) - assistant", marked))
        [item] = auto._display_events(SimpleNamespace(names={1: "main"}), events)
        self.assertEqual(named(str(item)), "   <g>[<0>[#1 (main) - assistant] <b>a</b>")

    def test_auto_prints_styled_items(self):
        events = (auto._Event(1, render_interaction_items((Message("assistant", "**a**"),))),)
        session = SimpleNamespace(names={1: "main"}, drain_events=lambda: events)
        output = io.StringIO()
        with redirect_stdout(output):
            auto._print_events(session)
        self.assertEqual(named(output.getvalue()), "   <g>[<0>[#1 (main) - assistant] <b>a</b>\n")

    def test_the_demo_prints_items_with_or_without_the_marker(self):
        [item] = render_interaction_items((Message("assistant", "**a**\x1b[2J"),))
        for markdown in (True, False):
            with self.subTest(markdown=markdown), mock.patch("builtins.print") as printed:
                demo._show(item, markdown=markdown)
            [shown] = printed.call_args.args
            self.assertIsInstance(shown, DisplayItem)
            self.assertEqual((shown.text, shown.is_markdown),
                             ("[assistant] **a**\\u001b[2J", markdown))
        model = SimpleNamespace(sample=lambda *args, **kwargs: None)
        with self.assertRaisesRegex(ValueError, "display_assistant must be markdown or plain"):
            demo.run(model, Environment(), display_assistant="fancy")


if __name__ == "__main__":
    unittest.main()
