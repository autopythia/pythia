"""The apply_patch command (pythia.interaction.apply_patch), run as models run
it: as a process, through a shell, and through exec_command once installed.

Everything happens in temporary directories: the installed case uses a
temporary HOME and prefix, never the user's installation.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import CommandRuntime
from pythia.interaction import Environment
from pythia.interaction import ToolCall
from pythia.interaction import create_apply_patch_tool
from pythia.interaction import create_exec_command_tool
from pythia.interaction.apply_patch import USAGE

CHECKOUT = Path(__file__).resolve().parents[1]
INSTALLER = CHECKOUT / "install.py"
COMMAND = (sys.executable, "-m", "pythia.interaction.apply_patch")

V4A_PATCH = "\n".join((
    "*** Begin Patch",
    "*** Update File: update.txt",
    "@@",
    "-old",
    "+new",
    "*** Add File: added.txt",
    "+added",
    "*** Delete File: delete.txt",
    "*** Update File: move.txt",
    "*** Move to: moved.txt",
    "@@",
    "-before",
    "+after",
    "*** End Patch",
))
UNIFIED_PATCH = "\n".join((
    "diff --git a/update.txt b/update.txt",
    "--- a/update.txt",
    "+++ b/update.txt",
    "@@ -1 +1 @@",
    "-old",
    "+new",
    "diff --git a/added.txt b/added.txt",
    "new file mode 100644",
    "--- /dev/null",
    "+++ b/added.txt",
    "@@ -0,0 +1 @@",
    "+added",
    "diff --git a/delete.txt b/delete.txt",
    "deleted file mode 100644",
    "--- a/delete.txt",
    "+++ /dev/null",
    "@@ -1 +0,0 @@",
    "-delete",
))


def _seed(root: Path) -> Path:
    """The files that the patches above change."""
    root.mkdir(parents=True)
    for name, text in (("update.txt", "old\n"), ("delete.txt", "delete\n"),
                       ("move.txt", "before\n")):
        (root / name).write_text(text, encoding="utf-8")
    return root


def _tree(root: Path) -> dict:
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8")
            for path in sorted(root.rglob("*")) if path.is_file()}


def _add(path: str) -> str:
    return "\n".join(("*** Begin Patch", f"*** Add File: {path}", "+x", "*** End Patch"))


def _environ(**changes) -> dict:
    """This process's environment, with the checkout importable."""
    return {**os.environ, "PYTHONPATH": str(CHECKOUT), **changes}


def _execute(environment, name, arguments):
    result = environment.execute_tool_calls(
        (ToolCall(name=name, call_id="call", arguments_json=json.dumps(arguments)),))
    return result.items[0]


class ApplyPatchCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_command(self, *arguments, stdin=b"", cwd=None, **environ):
        return subprocess.run([*COMMAND, *arguments], cwd=cwd or self.root, input=stdin,
                              capture_output=True, env=_environ(**environ), timeout=60)

    def test_it_does_what_the_tool_does_with_the_patch_as_argument_or_input(self):
        for index, patch in enumerate((V4A_PATCH, UNIFIED_PATCH)):
            tool_root = _seed(self.root / f"{index} tool")
            expected = _execute(Environment((create_apply_patch_tool(tool_root),)),
                                "apply_patch", {"patch": patch})
            self.assertTrue(expected.success, expected.output)
            for form in ("argument", "input"):
                with self.subTest(patch=index, form=form):
                    root = _seed(self.root / f"{index} {form}")
                    result = (self.run_command(patch, cwd=root) if form == "argument"
                              else self.run_command(stdin=patch.encode(), cwd=root))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, b"")
                    self.assertEqual(result.stdout.decode(), expected.output + "\n")
                    self.assertEqual(_tree(root), _tree(tool_root))

    def test_through_a_shell_a_heredoc_after_cd_and_a_pipe(self):
        command = " ".join(shlex.quote(part) for part in COMMAND)
        heredoc = _seed(self.root / "heredoc")
        result = subprocess.run(
            ["bash", "-c", f"cd heredoc && {command} <<'EOF'\n{V4A_PATCH}\nEOF\n"],
            cwd=self.root, capture_output=True, env=_environ(), timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode().splitlines(),
                         ["M update.txt", "A added.txt", "D delete.txt", "R move.txt -> moved.txt"])
        self.assertEqual(_tree(heredoc),
                         {"added.txt": "added\n", "moved.txt": "after\n", "update.txt": "new\n"})
        piped = _seed(self.root / "piped")
        (piped / "fix.diff").write_text(
            "--- a/update.txt\n+++ b/update.txt\n@@ -1 +1 @@\n-old\n+piped\n", encoding="utf-8")
        result = subprocess.run(["bash", "-c", f"cat fix.diff | {command}"], cwd=piped,
                                capture_output=True, env=_environ(), timeout=60)
        self.assertEqual((result.returncode, result.stdout), (0, b"M update.txt\n"), result.stderr)
        self.assertEqual((piped / "update.txt").read_text(encoding="utf-8"), "piped\n")

    def test_a_patch_that_fails_writes_nothing_and_says_why(self):
        (self.root / "a.txt").write_text("a\n", encoding="utf-8")
        (self.root / "b.txt").write_text("b\n", encoding="utf-8")
        (self.root / "c.txt").write_bytes(b"\xff\n")
        update_c = "\n".join(("*** Begin Patch", "*** Update File: c.txt", "@@", "-x", "+y",
                              "*** End Patch"))
        cases = (
            # The tool's message, after the command's name.
            ("\n".join(("*** Begin Patch", "*** Update File: a.txt", "@@", "-a", "+changed",
                        "*** Update File: b.txt", "@@", "-missing", "+changed",
                        "*** End Patch")),
             "apply_patch: unable to apply hunk while updating b.txt"),
            # One that starts with the command's name keeps it as its prefix.
            ("*** Begin Patch\n*** End Patch\n", "apply_patch contains no file operations"),
            # Another kind of error is named, as in the tool's failed result.
            (update_c, "apply_patch: UnicodeDecodeError: 'utf-8' codec can't decode byte "
                       "0xff in position 0: invalid start byte"),
        )
        for patch, message in cases:
            with self.subTest(message=message):
                result = self.run_command(patch)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr.decode(), message + "\n")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ["a.txt", "b.txt", "c.txt"])
        self.assertEqual((self.root / "a.txt").read_bytes(), b"a\n")
        self.assertEqual((self.root / "b.txt").read_bytes(), b"b\n")
        self.assertEqual((self.root / "c.txt").read_bytes(), b"\xff\n")

    def test_usage(self):
        (self.root / "my fix.diff").write_text(UNIFIED_PATCH, encoding="utf-8")
        cases = (
            ((), "no patch given"),
            ((V4A_PATCH, V4A_PATCH), "too many arguments: the only argument is the patch itself"),
            (("-p1",), "unknown option -p1: the only argument is the patch itself"),
            (("my fix.diff",),
             "'my fix.diff' is a file name, not a patch: use apply_patch < 'my fix.diff'"),
            # Newlines written as \n, as a model may.
            (("*** Begin Patch\\n*** Add File: x.txt\\n+x\\n*** End Patch",),
             "the patch argument is a single line: put each of its lines on a line of its own "
             "(newlines, not \\n), as in a heredoc: apply_patch <<'EOF'"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.run_command(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr.decode(), f"apply_patch: {message}\n{USAGE}\n")
        result = self.run_command("--help")
        self.assertEqual((result.returncode, result.stdout.decode(), result.stderr),
                         (0, USAGE + "\n", b""))
        self.assertEqual([path.name for path in self.root.iterdir()], ["my fix.diff"])

    def test_the_patch_is_utf8_with_any_character(self):
        added = "\n".join(("*** Begin Patch", "*** Add File: café.txt", "+crème", "*** End Patch"))
        result = self.run_command(added)
        self.assertEqual((result.returncode, result.stdout.decode()), (0, "A café.txt\n"),
                         result.stderr)
        self.assertEqual((self.root / "café.txt").read_text(encoding="utf-8"), "crème\n")
        # A byte that isn't UTF-8, as an argument (which Python decodes with
        # surrogateescape) or on standard input, is refused before any write.
        bad = _add("bad.txt").encode("utf-8").replace(b"+x", b"+\xff")
        for result, source in ((self.run_command(bad), "argument"),
                               (self.run_command(stdin=bad), "on standard input")):
            with self.subTest(source=source):
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr.decode(),
                                 f"apply_patch: the patch {source} is not valid UTF-8\n")
        self.assertFalse((self.root / "bad.txt").exists())

    def test_a_terminal_without_the_characters_gets_them_escaped(self):
        # The patch is applied before the lines print: a path that the
        # terminal's encoding lacks must not turn success into a crash.
        added = "\n".join(("*** Begin Patch", "*** Add File: café.txt", "+x", "*** End Patch"))
        result = self.run_command(stdin=added.encode("utf-8"), PYTHONIOENCODING="ascii")
        self.assertEqual((result.returncode, result.stdout), (0, b"A caf\\xe9.txt\n"),
                         result.stderr)
        self.assertTrue((self.root / "café.txt").exists())

    def test_nothing_bounds_its_paths(self):
        # Like patch or git apply, and like exec_command itself (through cat,
        # for one): the workspace policy bounds only the tool's paths.
        sub = self.root / "workspace" / "sub"
        sub.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (sub.parent / "link").symlink_to(outside, target_is_directory=True)
        paths = ("../up.txt", str(outside / "absolute.txt"), "../link/linked.txt")
        patch = "\n".join(("*** Begin Patch",
                           *(line for path in paths for line in (f"*** Add File: {path}", "+x")),
                           "*** End Patch"))
        result = self.run_command(patch, cwd=sub)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode(), "".join(f"A {path}\n" for path in paths))
        self.assertEqual((sub.parent / "up.txt").read_text(encoding="utf-8"), "x\n")
        self.assertEqual(_tree(outside), {"absolute.txt": "x\n", "linked.txt": "x\n"})


class InstalledCommandTests(unittest.TestCase):
    def test_exec_command_runs_the_installed_command(self):
        # As a model runs it: installed by install.py, here into a temporary
        # prefix and HOME, and found on PATH by exec_command's shell.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            home, prefix, workspace = root / "home", root / "prefix", root / "workspace"
            home.mkdir()
            workspace.mkdir()
            installed = subprocess.run(
                [sys.executable, str(INSTALLER), "-e", "--prefix", str(prefix)], cwd=CHECKOUT,
                env={**os.environ, "HOME": str(home)}, capture_output=True, text=True,
                timeout=60)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            search = f"{prefix / 'bin'}{os.pathsep}{os.environ.get('PATH', os.defpath)}"
            outputs = []
            with mock.patch.dict(os.environ, {"PATH": search}), \
                    CommandRuntime(workspace) as runtime:
                environment = Environment((create_exec_command_tool(runtime),))
                for path in ("new.txt", "../escaped.txt"):
                    outputs.append(_execute(environment, "exec_command", {
                        "cmd": f"apply_patch <<'EOF'\n{_add(path)}\nEOF", "login": False,
                    }).output)
            self.assertEqual((workspace / "new.txt").read_text(encoding="utf-8"), "x\n")
            # The workspace policy, on here, bounds exec_command's workdir, not
            # what its commands write.
            self.assertEqual((root / "escaped.txt").read_text(encoding="utf-8"), "x\n")
        for output, path in zip(outputs, ("new.txt", "../escaped.txt")):
            self.assertIn("Process exited with code 0", output)
            self.assertTrue(output.endswith(f"Output:\nA {path}\n"), output)


if __name__ == "__main__":
    unittest.main()
