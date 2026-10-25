"""``apply_patch`` as a shell command, for models that run it through
``exec_command`` instead of calling the tool, as Codex lets them::

    apply_patch <<'EOF'
    *** Begin Patch
    ...
    *** End Patch
    EOF

``python3 install.py -e`` installs it as ``apply_patch``; ``python3 -m
pythia.interaction.apply_patch`` runs it too. It applies the patch with the
apply_patch tool's code (``local_tools/patch.py``): the same formats (V4A and
unified diff), checks and messages as the tool with the workspace policy
off, and all files or none. A relative path starts at the command's working
directory, as in any shell command. The workspace policy doesn't apply: like
any shell command, and like ``exec_command`` itself, it can write wherever
the user can.

The patch is the one argument, or else standard input, in UTF-8. On success
the command prints the tool's lines (``A``, ``M``, ``D``, ``R old -> new``)
and exits 0. A patch that fails prints the reason on standard error, after
the command's name, and exits 1, having written nothing. A usage error exits
2.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence

from .local_tools.patch import _apply_patch_text

USAGE = """\
Usage: apply_patch 'PATCH'
       apply_patch <<'EOF'
       PATCH
       EOF
       apply_patch < FILE
Applies a V4A (*** Begin Patch) or unified-diff patch, as Pythia's apply_patch
tool does. Relative paths start at the current directory."""

EXIT_OK = 0  # applied, or the usage was asked for
EXIT_NOT_APPLIED = 1
EXIT_USAGE = 2


class _UsageError(Exception):
    """Shown with the usage, and exit status 2."""


def _argument_patch(argument: str) -> str:
    """The patch given as the argument, as UTF-8 text."""
    if len(argument.splitlines()) <= 1:
        # A patch has several lines, so a one-line argument is something else.
        if argument.startswith("-"):
            raise _UsageError(f"unknown option {argument}: the only argument is the patch itself")
        if argument.strip() and os.path.lexists(argument):
            quoted = shlex.quote(argument)
            raise _UsageError(f"{quoted} is a file name, not a patch: use apply_patch < {quoted}")
        if argument.strip():
            raise _UsageError("the patch argument is a single line: put each of its lines on "
                              "a line of its own (newlines, not \\n), as in a heredoc: "
                              "apply_patch <<'EOF'")
    try:
        # The bytes as given: Python decodes them with surrogateescape.
        return os.fsencode(argument).decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("the patch argument is not valid UTF-8") from None


def _stdin_patch() -> str:
    """The patch on standard input, as UTF-8 text; empty when it is closed."""
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        return ""
    try:
        return stream.read().decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("the patch on standard input is not valid UTF-8") from None


def _failure(error: Exception) -> str:
    """A failure as the tool's failed result words it, less its prefix: the
    tool's own errors are ValueErrors, and any other kind is named. A message
    that starts with the command's name, as many of the tool's do, keeps it
    as its prefix."""
    if type(error) is ValueError:
        reason = str(error)
    else:
        reason = f"{type(error).__name__}: {error}"
    return reason if reason.startswith("apply_patch ") else f"apply_patch: {reason}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command with ``argv`` (default: ``sys.argv[1:]``); return its
    exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        # Paths that the terminal's encoding lacks still print, escaped: the
        # patch is applied by then.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
    if arguments in (["-h"], ["--help"]):
        print(USAGE)
        return EXIT_OK
    try:
        if len(arguments) > 1:
            raise _UsageError("too many arguments: the only argument is the patch itself")
        patch = _argument_patch(arguments[0]) if arguments else _stdin_patch()
        if not patch.strip():
            raise _UsageError("no patch given")
        try:
            cwd = Path.cwd()
        except OSError as error:
            raise ValueError(f"the current directory is unavailable: {error.strerror}") from None
        changes = _apply_patch_text(cwd, patch, enable_workspace=False)
    except _UsageError as error:
        print(f"apply_patch: {error}\n{USAGE}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:
        print(_failure(error), file=sys.stderr)
        return EXIT_NOT_APPLIED
    print(changes)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
