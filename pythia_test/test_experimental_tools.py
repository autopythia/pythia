from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from pythia.interaction import DefaultEnvironment
from pythia.interaction import Environment
from pythia.interaction import EnvironmentError
from pythia.interaction import Message
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import create_update_plan_tool
from pythia.interaction import PlanStore
from pythia.interaction import default_environment
from pythia.interaction.experimental_tools import create_inject_user_message_tool


class ExperimentalToolTests(unittest.TestCase):
    def test_factory_schema_and_manual_registration(self):
        tool = create_inject_user_message_tool()
        self.assertEqual(tool.spec.name, "experimental_inject_user_message")
        self.assertIn("synthetic user message", tool.spec.description)
        self.assertEqual(tool.spec.parameters, {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        })
        environment = Environment((tool,))
        self.assertEqual(environment.tool_specs, (tool.spec,))
        # Each distinct call requests a message, even when reusing the tool.
        for call_id in ("1", "2"):
            result = environment.execute_tool_calls((
                ToolCall(tool.spec.name, call_id, "{}"),
            ))
            expected = ToolResult(call_id, "Synthetic user message queued.")
            self.assertEqual(result.items, (expected,))
            self.assertEqual(
                result.context_items(),
                (expected, Message("user", "hello world")),
            )

    def test_invalid_arguments_never_inject(self):
        tool = create_inject_user_message_tool()
        environment = Environment((tool,))
        for arguments in ('{"text":"arbitrary"}', '{"unexpected":null}', "{", "[]", "null", '""'):
            with self.subTest(arguments=arguments):
                result = environment.execute_tool_calls((
                    ToolCall(tool.spec.name, "1", arguments),
                ))
                self.assertFalse(result.items[0].success)
                self.assertEqual(result.user_messages, ())
        with self.assertRaises(TypeError):
            tool.handler([])

    def test_tool_is_absent_from_default_environment(self):
        with tempfile.TemporaryDirectory() as cwd:
            with DefaultEnvironment(cwd) as environment:
                self.assertEqual(tuple(spec.name for spec in environment.tool_specs), (
                    "exec_command", "write_stdin", "update_plan", "apply_patch",
                ))
                result = environment.execute_tool_calls((
                    ToolCall("experimental_inject_user_message", "1", "{}"),
                ))
                self.assertFalse(result.items[0].success)
                self.assertEqual(result.user_messages, ())

    def test_default_environment_accepts_explicit_extra_tools(self):
        tool = create_inject_user_message_tool()
        with tempfile.TemporaryDirectory() as cwd:
            with DefaultEnvironment(cwd, extra_tools=iter((tool,))) as environment:
                self.assertEqual(tuple(spec.name for spec in environment.tool_specs), (
                    "exec_command", "write_stdin", "update_plan", "apply_patch", tool.spec.name,
                ))
                result = environment.execute_tool_calls((ToolCall(tool.spec.name, "1", "{}"),))
                self.assertEqual(result.user_messages, (Message("user", "hello world"),))

    def test_extra_tool_validation_closes_runtime_on_registration_failure(self):
        tool = create_inject_user_message_tool()
        cases = (
            ((tool, tool), EnvironmentError),
            ((create_update_plan_tool(PlanStore()),), EnvironmentError),
            ((object(),), TypeError),
        )
        for tools, exception in cases:
            with self.subTest(tools=tools):
                with mock.patch.object(default_environment, "CommandRuntime", autospec=True) as runtime:
                    with self.assertRaises(exception):
                        DefaultEnvironment(extra_tools=tools)
                    runtime.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
