"""Tool inputs that are one method call, in Python with literal arguments, as
the media and ui tools take them: ``{"call": "attach('build/shot.png')"}``.

A call may start with the tool's name (``media.attach(...)``), as transcripts
show calls. Nothing in it is evaluated: each value must be a literal
(``ast.literal_eval``), and the arguments bind to the method's own signature.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import re
from typing import Callable, Dict


class MethodCallError(ValueError):
    """An input that isn't one call of the tool's methods; the text says what
    to give instead, and is the tool's result."""


@dataclass(frozen=True)
class MethodCall:
    """A call as read: the method, its arguments by parameter name as given
    (no defaults), and the call in canonical form (``text``)."""

    method: str
    arguments: Dict[str, object]
    text: str


@dataclass(frozen=True)
class MethodCallReader:
    """Reads one tool's inputs. ``methods`` maps each method's name to a
    function whose signature, after ``self``, the arguments bind to; the rest
    words the errors: ``example`` is a whole call, ``short_example`` a short
    one, ``listing`` names the methods (``the tool has one, attach(...)``),
    ``positional`` is what a positional argument is called, and ``literal``
    is an example of a value."""

    tool: str
    methods: Mapping[str, Callable]
    example: str
    short_example: str
    listing: str
    positional: str = "argument"
    literal: str = "1"

    def read(self, arguments) -> MethodCall:
        """The call that a tool input stands for; a MethodCallError says what
        to give instead."""
        tool = self.tool
        if not isinstance(arguments, Mapping):
            raise MethodCallError(self._no_call())
        fields = dict(arguments)
        source = fields.pop("call", None)
        if fields:
            raise MethodCallError(f"{tool}: unexpected field {sorted(fields)[0]!r}; "
                                  "give call only.")
        if source is None:
            raise MethodCallError(self._no_call())
        if not isinstance(source, str):
            raise MethodCallError(f"{tool}: call must be text, one method call such as "
                                  f"{self.short_example}.")
        source = re.sub(rf"^{re.escape(tool)}\s*\.\s*", "", source.strip())
        try:
            node = ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise MethodCallError(f"{tool}: call must be one method call, such as "
                                  f"{self.short_example}; {exc.msg}.") from None
        except (ValueError, RecursionError, MemoryError):  # a null character; deep nesting
            node = None
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            raise MethodCallError(f"{tool}: call must be one method call, such as "
                                  f"{self.short_example}.")
        name = node.func.id
        if name not in self.methods:
            raise MethodCallError(f"{tool}: there is no method {name}(); {self.listing}.")
        call = f"{tool}.{name}()"
        args = []
        for index, value in enumerate(node.args, 1):
            if isinstance(value, ast.Starred):
                raise MethodCallError(f"{call}: give each {self.positional} by itself, "
                                      "not *sequence.")
            args.append(self._literal(value, call, f"{self.positional} {index}"))
        keywords = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise MethodCallError(f"{call}: give each argument by itself, not **mapping.")
            keywords[keyword.arg] = self._literal(keyword.value, call, keyword.arg)
        signature = inspect.signature(self.methods[name])
        try:
            bound = signature.bind(None, *args, **keywords)
        except TypeError as exc:
            raise MethodCallError(f"{call}: {exc}") from None
        parameters = list(signature.parameters.values())[1:]  # after self
        given = {parameter.name: bound.arguments[parameter.name] for parameter in parameters
                 if parameter.name in bound.arguments}
        return MethodCall(name, given, _text_form(tool, name, parameters, given))

    def _no_call(self) -> str:
        return f"{self.tool}: give call, one method call such as {self.example}."

    def _literal(self, node: ast.expr, call: str, what: str):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            raise MethodCallError(f"{call}: {what} must be a literal, such as "
                                  f"{self.literal}.") from None


def _text_form(tool: str, method: str, parameters, given: Dict[str, object]) -> str:
    """media.attach('a.png', text='Hi.'): a *args parameter's values by
    position, every other argument by name, and none that is its default."""
    pieces = []
    for parameter in parameters:
        if parameter.name not in given:
            continue
        value = given[parameter.name]
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            pieces.extend(repr(item) for item in value)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            pieces.extend(f"{key}={item!r}" for key, item in value.items())
        elif not _is_default(value, parameter.default):
            pieces.append(f"{parameter.name}={value!r}")
    return f"{tool}.{method}({', '.join(pieces)})"


def _is_default(value, default) -> bool:
    # By type too: cursor=1 isn't cursor=True.
    return value is default or (type(value) is type(default) and value == default)


__all__ = ["MethodCall", "MethodCallError", "MethodCallReader"]
