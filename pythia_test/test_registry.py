"""The registry of root contexts (registry.py), and the CLI's options for it.

--registry names it, --context opens a save from it, and --list-roots lists it.
"""

from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock
import uuid

from pythia.interaction import Environment, Init, InteractionContext, Message, cli, registry
from pythia.interaction.registry import (
    Registry, RegistryError, abbreviate, key_at, key_of, registry_path, render_roots,
)
from pythia_test.interaction_helpers import full_log, patch_saves, seed_cli_save
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


# A root's links in the registry: its save's three parts, by key and by number.
_SUFFIXES = (".jsonl", ".summary.md", "._private")


def _save(prefix, prefix_id=None, *, field="prefix_id"):
    """A CLI save at ``prefix`` whose full log, and save log, hold only its Init.

    Returns its private directory and its prefix ID.
    """
    private = Path(f"{prefix}._private")
    private.mkdir(parents=True)
    prefix_id = prefix_id or f"context_{uuid.uuid4().hex}"
    line = json.dumps({"type": "init", field: prefix_id}) + "\n"
    (private / "context.jsonl").write_text(line)
    Path(f"{prefix}.jsonl").write_text(line)
    return private, prefix_id


def _root_links(number, key, prefix):
    """The six links of the root ``number``, ``key``, whose save is at ``prefix``."""
    links = {}
    for suffix in _SUFFIXES:
        links[f"{key}{suffix}"] = f"{prefix}{suffix}"
        links[f"{number}{suffix}"] = f"{key}{suffix}"
    return links


# Workers for the concurrency tests, each in a process of its own.
def _register_new_saves(root, base, count):
    numbers = []
    for index in range(count):
        private, prefix_id = _save(f"{base}/{os.getpid()}-{index}")
        numbers.append(Registry(root).register(prefix_id, private).number)
    return numbers


def _register(root, private, prefix_id, wait, start):
    registry.WAIT_SECONDS = wait
    time.sleep(max(0.0, start - time.time()))  # all processes at once
    registration = Registry(root).register(prefix_id, private)
    return registration.status, registration.number


class _Answers:
    """A model that answers each sample from a script, whatever the save."""

    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append(context.copy())
        outcome = self.outcomes.popleft()
        return outcome(context) if callable(outcome) else outcome


def _in_processes(function, arguments):
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        return pool.starmap_async(function, arguments).get(timeout=120)


class _TemporaryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.registry = Registry(self.root / "dot-pythia" / "root")
        self.work = self.root / "ws" / ".interaction"
        self.work.mkdir(parents=True)

    def links(self):
        return {name: os.readlink(self.registry.path / name)
                for name in os.listdir(self.registry.path)}

    def assertFollow(self):
        """Each link that follows a save-log link agrees with it; nothing else is left."""
        links = self.links()
        for name, target in links.items():
            self.assertFalse(name.startswith("."), f"a temporary link was left: {name}")
            if not name.endswith(".jsonl"):
                stem = next(name[:-len(suffix)] for suffix in _SUFFIXES if name.endswith(suffix))
                self.assertIn(f"{stem}.jsonl", links, f"{name} follows no save-log link")
                continue
            stem, log = name[:-len(".jsonl")], target[:-len(".jsonl")]
            for suffix in _SUFFIXES[1:]:
                self.assertEqual(links.get(f"{stem}{suffix}"), f"{log}{suffix}", name)


class KeyTests(unittest.TestCase):
    def test_a_new_init_has_a_context_key(self):
        key = Init().prefix_id
        self.assertRegex(key, r"^context_[0-9a-f]{32}$")
        self.assertEqual(key_of(key), key)

    def test_an_older_session_prefix_id_is_the_same_key_with_context(self):
        digits = uuid.uuid4().hex
        self.assertEqual(key_of(f"session_{digits}"), f"context_{digits}")

    def test_other_prefix_ids_are_not_keys(self):
        for prefix_id in ("saved", "session_../../etc", f"context_{'A' * 32}", "context_1234",
                          f"context_{uuid.uuid4().hex}\n", None, 7):
            with self.subTest(prefix_id=prefix_id), self.assertRaises(RegistryError):
                key_of(prefix_id)

    def test_abbreviations_are_the_shortest_unique_with_at_least_4_digits(self):
        keys = ["context_" + "1fe5" + "0" * 28, "context_" + "1fe6" + "0" * 28,
                "context_" + "9180a" + "0" * 27, "context_" + "9180b" + "0" * 27]
        self.assertEqual([abbreviate(key, keys) for key in keys],
                         ["context_1fe5", "context_1fe6", "context_9180a", "context_9180b"])


class RegistryPathTests(unittest.TestCase):
    def test_the_option_then_the_environment_then_the_default(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(registry.REGISTRY_ENV, None)
            self.assertEqual(registry_path(), Path.home() / ".pythia" / "root")
            self.assertEqual(registry_path("/x/root"), Path("/x/root"))
            self.assertIsNone(registry_path("none"))
            os.environ[registry.REGISTRY_ENV] = "none"
            self.assertIsNone(registry_path())
            self.assertEqual(registry_path("/y/root"), Path("/y/root"))
            os.environ[registry.REGISTRY_ENV] = "/z/root"
            self.assertEqual(registry_path(), Path("/z/root"))
            self.assertEqual(registry_path("~/r"), Path.home() / "r")
            self.assertEqual(registry_path("relative"), Path.cwd() / "relative")

    def test_the_test_suite_turns_the_registry_off(self):
        self.assertEqual(os.environ.get(registry.REGISTRY_ENV), "none")


class RegisterTests(_TemporaryTestCase):
    def test_a_new_save_gets_the_next_number_from_1_and_six_links(self):
        private, key = _save(self.work / "a")
        registration = self.registry.register(key, private)
        self.assertEqual((registration.status, registration.number), ("new", 1))
        self.assertEqual(self.links(), _root_links(1, key, self.work / "a"))
        self.assertEqual(oct(self.registry.path.stat().st_mode & 0o777), "0o700")
        self.assertIn(key, (self.registry.path / "1.jsonl").read_text())
        self.assertIn(key, (self.registry.path / "1._private" / "context.jsonl").read_text())
        summary = self.registry.path / "1.summary.md"
        self.assertFalse(summary.exists())  # until a coldstart compaction writes it
        (self.work / "a.summary.md").write_text("Covers lines 1–1 of a.\n")
        self.assertEqual(summary.read_text(), "Covers lines 1–1 of a.\n")
        other, other_key = _save(self.work / "b")
        self.assertEqual(self.registry.register(other_key, other).number, 2)

    def test_reopening_keeps_the_number_also_through_a_symlinked_directory(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        (self.root / "alias").symlink_to(self.root / "ws")
        for path in (private, self.root / "alias" / ".interaction" / "a._private"):
            registration = self.registry.register(key, path)
            self.assertEqual((registration.status, registration.number), ("same", 1))
        self.assertEqual(self.links(), _root_links(1, key, self.work / "a"))

    def test_links_hold_real_paths(self):
        (self.root / "alias").symlink_to(self.root / "ws")
        private, key = _save(self.work / "a")
        registration = self.registry.register(key, self.root / "alias/.interaction/a._private")
        self.assertEqual(registration.home, private)
        self.assertEqual(self.links()[f"{key}.jsonl"], str(self.work / "a.jsonl"))
        # The directory's real path: a save log that is a link keeps its own path.
        private, key = _save(self.work / "b")
        os.rename(self.work / "b.jsonl", self.root / "elsewhere.jsonl")
        (self.work / "b.jsonl").symlink_to(self.root / "elsewhere.jsonl")
        self.assertEqual(self.registry.register(key, private).home, private)
        self.assertEqual(self.links()[f"{key}.jsonl"], str(self.work / "b.jsonl"))
        self.assertFollow()

    def test_a_moved_save_keeps_its_number_and_its_links_follow(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        os.rename(self.root / "ws", self.root / "moved")
        moved = self.root / "moved" / ".interaction" / "a._private"
        registration = self.registry.register(key, moved)
        self.assertEqual((registration.status, registration.number), ("moved", 1))
        self.assertEqual(registration.previous, private)
        self.assertEqual(self.links(), _root_links(1, key, self.root / "moved/.interaction/a"))

    def test_a_copy_whose_original_holds_the_key_is_not_registered(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        shutil.copytree(self.root / "ws", self.root / "copy")
        registration = self.registry.register(key, self.root / "copy/.interaction/a._private")
        self.assertEqual((registration.status, registration.number), ("copy", 1))
        self.assertEqual(registration.home, private)
        self.assertEqual(self.links(), _root_links(1, key, self.work / "a"))

    def test_the_key_is_read_from_the_full_log_not_the_save_log(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        shutil.copytree(self.root / "ws", self.root / "copy")
        (self.work / "a.jsonl").unlink()  # the original's save log, which the CLI writes anew
        registration = self.registry.register(key, self.root / "copy/.interaction/a._private")
        self.assertEqual((registration.status, registration.number), ("copy", 1))
        self.assertEqual([entry.state for entry in self.registry.entries()], ["ok"])
        self.assertEqual(self.registry.register(key, private).status, "same")

    def test_an_older_save_is_registered_under_context_and_left_as_it_is(self):
        digits = uuid.uuid4().hex
        for field in ("prefix_id", "session_id"):
            with self.subTest(field=field):
                private, _ = _save(self.work / field, f"session_{digits}", field=field)
                before = (private / "context.jsonl").read_bytes()
                registration = self.registry.register(f"session_{digits}", private)
                self.assertEqual(registration.key, f"context_{digits}")
                self.assertEqual(key_at(private), f"context_{digits}")
                self.assertEqual((private / "context.jsonl").read_bytes(), before)
                shutil.rmtree(private)

    def test_a_replaced_save_gets_a_new_number_and_the_old_one_lists_replaced(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        shutil.rmtree(private)
        private, new_key = _save(self.work / "a")  # as --resume=False replaces it
        self.assertEqual(self.registry.register(new_key, private).number, 2)
        self.assertEqual([(entry.number, entry.state) for entry in self.registry.entries()],
                         [(1, "replaced"), (2, "ok")])

    def test_a_key_linked_before_a_crash_gets_its_number_when_next_opened(self):
        private, key = _save(self.work / "a")
        self.registry.path.mkdir(parents=True)
        os.symlink(self.work / "a.jsonl", self.registry.path / f"{key}.jsonl")  # then it died
        with mock.patch.object(registry, "WAIT_SECONDS", 0.05):
            started = time.monotonic()
            registration = self.registry.register(key, private)
        self.assertEqual((registration.status, registration.number), ("same", 1))
        self.assertGreaterEqual(time.monotonic() - started, 0.05)
        self.assertEqual(self.links(), _root_links(1, key, self.work / "a"))

    def test_links_that_a_crash_left_missing_or_stale_are_made_when_next_opened(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        for name in ("1._private", f"{key}.summary.md", "1.summary.md"):
            os.unlink(self.registry.path / name)  # claimed, then the process died
        stale = self.registry.path / f"{key}._private"
        os.unlink(stale)
        stale.symlink_to(self.root / "old" / "a._private")  # died while it moved the root
        failing = mock.patch.object(Registry, "_ensure", side_effect=PermissionError(13, "no"))
        with failing:
            registration = self.registry.register(key, private)  # they can't be made: still 1
        self.assertEqual((registration.status, registration.number), ("same", 1))
        registration = self.registry.register(key, private)
        self.assertEqual((registration.status, registration.number), ("same", 1))
        self.assertEqual(self.links(), _root_links(1, key, self.work / "a"))

    def test_a_claim_that_another_opener_beat_adopts_that_number(self):
        private, key = _save(self.work / "a")
        self.registry.path.mkdir(parents=True)
        os.symlink(self.work / "a.jsonl", self.registry.path / f"{key}.jsonl")
        os.symlink(f"{key}.jsonl", self.registry.path / "1.jsonl")  # another's, after the listing:
        with mock.patch.object(Registry, "_scan", return_value=([], 0)):
            self.assertEqual(self.registry._claim(key), 1)
        self.assertEqual(sorted(name for name in self.links() if name[0].isdigit()), ["1.jsonl"])

    def test_a_second_number_made_by_hand_leaves_the_lower_as_the_roots(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        os.symlink(f"{key}.jsonl", self.registry.path / "9.jsonl")
        self.assertEqual(self.registry.number_of(key), 1)
        self.assertEqual(self.registry.register(key, private).number, 1)
        self.assertEqual([(entry.number, entry.first) for entry in self.registry.entries()],
                         [(1, None), (9, 1)])

    def test_a_number_that_only_a_following_link_names_is_not_claimed_again(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        os.unlink(self.registry.path / "1.jsonl")  # removed by hand; 1._private stays
        other, other_key = _save(self.work / "b")
        self.assertEqual(self.registry.register(other_key, other).number, 2)

    def test_the_directory_holds_only_links_and_ignores_other_names(self):
        private, key = _save(self.work / "a")
        self.registry.register(key, private)
        os.rename(self.root / "ws", self.root / "moved")
        self.registry.register(key, self.root / "moved" / ".interaction" / "a._private")
        self.assertTrue(all(os.path.islink(self.registry.path / name) for name in self.links()))
        self.assertEqual(self.links(), _root_links(1, key, self.root / "moved/.interaction/a"))
        os.symlink("elsewhere", self.registry.path / f".{key}.jsonl.123.abcdef01")  # a crash's
        (self.registry.path / "notes.txt").write_text("not the registry's")
        other, other_key = _save(self.root / "b")
        self.assertEqual(self.registry.register(other_key, other).number, 2)
        self.assertEqual([entry.number for entry in self.registry.entries()], [1, 2])

    def test_refusals_and_failures(self):
        private, key = _save(self.work / "a")
        with self.assertRaises(RegistryError):
            self.registry.register("saved", private)
        self.registry.path.mkdir(parents=True)
        (self.registry.path / f"{key}.jsonl").write_text("not a link")
        with self.assertRaisesRegex(RegistryError, "is not a symbolic link to a save log"):
            self.registry.register(key, private)
        file_registry = Registry(self.root / "a-file")
        file_registry.path.write_text("")
        with self.assertRaisesRegex(RegistryError, "is not a directory"):
            file_registry.register(key, private)
        other, other_key = _save(self.work / "b")
        with mock.patch("os.symlink", side_effect=PermissionError(13, "Permission denied")), \
                self.assertRaises(PermissionError):
            self.registry.register(other_key, other)
        self.assertFalse([name for name in os.listdir(self.registry.path) if other_key in name])


class EntriesTests(_TemporaryTestCase):
    def test_listing_states_and_rendering(self):
        saves = {name: _save(self.work / name) for name in ("a", "b", "c", "d")}
        for private, key in saves.values():
            self.registry.register(key, private)
        shutil.rmtree(saves["b"][0])
        (saves["c"][0] / "context.jsonl").unlink()
        shutil.rmtree(saves["d"][0])
        _save(self.work / "d")
        self.assertEqual([(entry.number, entry.state) for entry in self.registry.entries()],
                         [(1, "ok"), (2, "missing"), (3, "unreadable"), (4, "replaced")])
        text = render_roots(self.registry.path, self.registry.entries(), home=str(self.root))
        lines = text.splitlines()
        self.assertEqual(lines[0], "Root contexts in ~/dot-pythia/root/: 4, from 1 to 4.")
        self.assertRegex(lines[1], r"^   1  context_[0-9a-f]{4,}\s+ok\s+\d{4}-\d\d-\d\d \d\d:\d\d"
                                   r"  ~/ws/\.interaction/a$")
        self.assertRegex(lines[2],
                         r"^   2  context_[0-9a-f]{4,}\s+missing\s+~/ws/\.interaction/b$")
        self.assertIn("unreadable", lines[3])
        self.assertIn("replaced", lines[4])

    def test_an_empty_or_missing_registry_lists_none(self):
        self.assertEqual(self.registry.entries(), [])
        self.assertEqual(render_roots(self.registry.path, []),
                         f"Root contexts in {self.registry.path}/: none yet.")


class FindTests(_TemporaryTestCase):
    def setUp(self):
        super().setUp()
        self.private, self.key = _save(self.work / "a", "context_9180" + "c" * 28)
        self.registry.register(self.key, self.private)
        self.prefix = self.work / "a"

    def test_a_number_or_a_key_with_or_without_1(self):
        for handle in ("1", "1#1", " 1 ", "context_9180", "context_9180#1", "session_9180",
                       self.key, "context_9180" + "c" * 6):
            with self.subTest(handle=handle):
                root = self.registry.find(handle)
                self.assertEqual((root.number, root.key, root.prefix), (1, self.key, self.prefix))

    def test_refusals(self):
        other, other_key = _save(self.work / "b", "context_9181" + "d" * 28)
        self.registry.register(other_key, other)
        cases = {
            "2#3": "has only #1",
            "7": "there's no root context 7",
            "context_918": "names no root context",
            "context_918#1": "names no root context",
            "context_918a": "no key",
            "#1": "names no root context",
            "tests": "names no root context",
            "1#tests": "names no root context",
        }
        for handle, message in cases.items():
            with self.subTest(handle=handle), self.assertRaisesRegex(RegistryError, message):
                self.registry.find(handle)

    def test_an_ambiguous_abbreviation_is_refused(self):
        _, other = _save(self.work / "b", "context_9180" + "d" * 28)
        self.registry.register(other, self.work / "b._private")
        with self.assertRaisesRegex(RegistryError,
                                    "2 keys in .* start with context_9180: give more"):
            self.registry.find("context_9180")
        self.assertEqual(self.registry.find("context_9180c").key, self.key)

    def test_a_root_that_is_not_where_it_is_registered(self):
        shutil.rmtree(self.private)
        with self.assertRaisesRegex(RegistryError, r"root context 1 \(context_9180\) isn't at "):
            self.registry.find("1")
        _save(self.work / "a")
        with self.assertRaisesRegex(RegistryError, "now holds another save, context_"):
            self.registry.find("1")
        (self.private / "context.jsonl").unlink()
        with self.assertRaisesRegex(RegistryError, "has no readable full log"):
            self.registry.find("context_9180")


class ConcurrencyTests(_TemporaryTestCase):
    def test_processes_registering_new_saves_get_distinct_numbers(self):
        self.registry.path.mkdir(parents=True)
        root, base = str(self.registry.path), str(self.root / "race")
        numbers = [number for chunk in _in_processes(_register_new_saves, [(root, base, 30)] * 8)
                   for number in chunk]
        self.assertEqual(sorted(numbers), list(range(1, 241)))
        self.assertEqual(len(self.links()), 240 * 6)
        self.assertFollow()

    def test_processes_registering_one_save_at_once_agree_on_one_number(self):
        private, key = _save(self.work / "fresh")
        arguments = (str(self.registry.path), str(private), key, 1.0, time.time() + 1.5)
        outcomes = _in_processes(_register, [arguments] * 8)
        self.assertEqual({number for _, number in outcomes}, {1})
        self.assertEqual(sum(status == "new" for status, _ in outcomes), 1)
        self.assertEqual(self.links(), _root_links(1, key, self.work / "fresh"))

    def test_processes_opening_a_key_linked_before_a_crash_agree_on_one_number(self):
        # With no wait, all 8 claim at once: they agree by adopting the first claim.
        self.registry.path.mkdir(parents=True)
        for index in range(2):
            private, key = _save(self.work / f"crashed-{index}")
            os.symlink(self.work / f"crashed-{index}.jsonl",
                       self.registry.path / f"{key}.jsonl")  # linked, then the process died
            arguments = (str(self.registry.path), str(private), key, 0.0, time.time() + 1.5)
            outcomes = _in_processes(_register, [arguments] * 8)
            self.assertEqual({number for _, number in outcomes}, {index + 1})
        self.assertEqual(sorted(name for name in self.links() if name[0].isdigit()
                                and name.endswith(".jsonl")), ["1.jsonl", "2.jsonl"])
        self.assertFollow()


class _CLITestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.registry_dir = self.root / "dot-pythia" / "root"
        self.work = self.root / "ws"
        self.work.mkdir()

    def main(self, argv, *outcomes, registry_dir=None):
        """Run the CLI with ``--registry``, by default this test's registry."""
        model = _Answers(*outcomes)
        stdout, stderr = io.StringIO(), io.StringIO()
        directory = self.registry_dir if registry_dir is None else registry_dir
        registry_option = () if directory is False else ("--registry", str(directory))
        with mock.patch.object(cli, "build_model", side_effect=lambda args: model), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([*argv, *registry_option])
        return code, stdout.getvalue(), stderr.getvalue()

    def run_prompt(self, save, prompt="hello", *options):
        return self.main(["--headless", "--cwd", str(self.work), "--save", str(save), *options,
                          "--prompt", prompt], _answer("done"))

    def root_lines(self, stderr):
        return [line for line in stderr.splitlines() if line.startswith("[cli] Root context")]


class CLIRegistrationTests(_CLITestCase):
    def test_a_new_save_registers_with_its_line_its_links_and_a_context_key(self):
        code, _, stderr = self.run_prompt(self.work / "a")
        self.assertEqual(code, 0, stderr)
        key = load_key(self.work / "a")
        self.assertRegex(key, r"^context_[0-9a-f]{32}$")
        self.assertEqual(self.root_lines(stderr),
                         [f"[cli] Root context 1 ({key[:12]}), new in {self.registry_dir}/."])
        links = {name: os.readlink(self.registry_dir / name)
                 for name in os.listdir(self.registry_dir)}
        self.assertEqual(links, _root_links(1, key, self.work / "a"))
        self.assertEqual((self.registry_dir / "1.jsonl").read_text(),
                         (self.work / "a.jsonl").read_text())
        code, _, stderr = self.run_prompt(self.work / "a", "again")
        self.assertEqual(self.root_lines(stderr),
                         [f"[cli] Root context 1 ({key[:12]}), in {self.registry_dir}/."])
        self.assertEqual(len(os.listdir(self.registry_dir)), 6)

    def test_an_older_save_registers_under_context_and_keeps_its_log(self):
        digits = uuid.uuid4().hex
        seed_cli_save(self.work / "old.jsonl", InteractionContext((
            Init(prefix_id=f"session_{digits}"), Message("user", "x"), Message("assistant", "y"))))
        code, _, stderr = self.run_prompt(self.work / "old")
        self.assertEqual(code, 0, stderr)
        self.assertIn(f"Root context 1 (context_{digits[:4]}), new", stderr)
        self.assertEqual(os.readlink(self.registry_dir / "1.jsonl"), f"context_{digits}.jsonl")
        with open(full_log(self.work / "old.jsonl")) as log:
            self.assertEqual(json.loads(log.readline())["prefix_id"], f"session_{digits}")

    def test_a_moved_save_keeps_its_number_and_names_its_old_path(self):
        self.run_prompt(self.work / "a")
        os.rename(self.work, self.root / "moved")
        self.work = self.root / "moved"
        code, _, stderr = self.run_prompt(self.work / "a")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(self.root_lines(stderr)), 1)
        self.assertRegex(self.root_lines(stderr)[0],
                         rf"^\[cli\] Root context 1 \(context_[0-9a-f]{{4}}\), in "
                         rf"{self.registry_dir}/; moved from {self.root}/ws/a\.$")

    def test_a_copy_is_not_registered(self):
        self.run_prompt(self.work / "a")
        shutil.copytree(self.work, self.root / "copy")
        key = load_key(self.work / "a")
        code, _, stderr = self.run_prompt(self.root / "copy" / "a")
        self.assertEqual(code, 0, stderr)
        self.assertIn(f"[cli] Root context: not registered: this save is a copy of 1 "
                      f"({key[:12]}), which is at {self.work}/a.", stderr)
        self.assertEqual(sorted(os.listdir(self.registry_dir)),
                         sorted(_root_links(1, key, self.work / "a")))

    def test_resume_false_registers_a_new_number_and_lists_the_old_one_replaced(self):
        self.run_prompt(self.work / "a")
        code, _, stderr = self.run_prompt(self.work / "a", "anew", "--resume=False")
        self.assertEqual(code, 0, stderr)
        self.assertIn("[cli] Root context 2 (", stderr)
        code, stdout, _ = self.main(["--list-roots"])
        self.assertEqual(code, 0)
        self.assertEqual([line.split()[2] for line in stdout.splitlines()[1:]],
                         ["replaced", "ok"])

    def test_registry_none_writes_nothing_and_shows_no_line(self):
        # The option, and the test suite's PYTHIA_REGISTRY=none without it.
        for directory in ("none", False):
            with self.subTest(directory=directory):
                code, _, stderr = self.main(
                    ["--headless", "--cwd", str(self.work), "--save", str(self.work / "a"),
                     "--prompt", "hello"], _answer("done"), registry_dir=directory)
                self.assertEqual(code, 0, stderr)
                self.assertNotIn("Root context", stderr)
        self.assertFalse(self.registry_dir.exists())

    def test_a_registry_that_cannot_be_written_leaves_the_session_as_it_was(self):
        (self.root / "a-file").write_text("")
        code, _, stderr = self.main(
            ["--headless", "--cwd", str(self.work), "--save", str(self.work / "a"),
             "--prompt", "hello"], _answer("done"), registry_dir=self.root / "a-file")
        self.assertEqual(code, 0, stderr)
        self.assertIn(f"[cli] Root context: not registered: the registry {self.root}/a-file "
                      "is not a directory.", stderr)
        self.assertIn("done", full_log(self.work / "a").read_text())
        if os.geteuid() != 0:
            self.registry_dir.mkdir(parents=True)
            self.registry_dir.chmod(0o500)
            self.addCleanup(self.registry_dir.chmod, 0o700)
            code, _, stderr = self.run_prompt(self.work / "b")
            self.assertEqual(code, 0, stderr)
            self.assertIn("[cli] Root context: not registered: [Errno 13] Permission denied",
                          stderr)

    def test_a_launch_whose_first_save_fails_leaves_no_entry(self):
        with patch_saves(OSError("disk failed")):
            code, _, stderr = self.run_prompt(self.work / "a")
        self.assertEqual(code, 1)
        self.assertNotIn("Root context", stderr)
        self.assertEqual(sorted(os.listdir(self.registry_dir))
                         if self.registry_dir.exists() else [], [])

    def test_a_save_in_its_original_format_is_not_registered(self):
        seed_cli_save(self.work / "old.jsonl", InteractionContext((
            Init(), Message("user", "x"), Message("assistant", "y"))))
        code, _, stderr = self.run_prompt(self.work / "old.jsonl", "hello",
                                          "--auto-migrate-save=False")
        self.assertEqual(code, 0, stderr)
        self.assertIn("[cli] Root context: not registered: a save in its original format "
                      "(--auto-migrate-save=False) isn't registered.", stderr)
        self.assertFalse(self.registry_dir.exists())


class CLIContextTests(_CLITestCase):
    def setUp(self):
        super().setUp()
        self.run_prompt(self.work / "a", "first")
        self.key = load_key(self.work / "a")

    def context(self, handle, prompt="more", *options):
        return self.main(["--headless", "--cwd", str(self.work), "--context", handle, *options,
                          "--prompt", prompt], _answer("done"))

    def test_a_number_or_a_key_resumes_that_save(self):
        for handle in ("1", "1#1", self.key[:12], f"{self.key[:12]}#1",
                       "session_" + self.key[8:16]):
            with self.subTest(handle=handle):
                code, _, stderr = self.context(handle, f"via {handle}")
                self.assertEqual(code, 0, stderr)
                self.assertIn(f"[cli] Save log: {self.work}/a.jsonl", stderr)
                self.assertIn(f"[cli] Root context 1 ({self.key[:12]}), in", stderr)
                self.assertIn(f"via {handle}", full_log(self.work / "a").read_text())
        self.assertEqual(sorted(os.listdir(self.registry_dir)),
                         sorted(_root_links(1, self.key, self.work / "a")))

    def test_a_name_in_the_registry_is_not_a_save_prefix(self):
        # Its ._private is a link, which the CLI refuses as a save's private directory.
        before = full_log(self.work / "a").read_bytes(), sorted(os.listdir(self.registry_dir))
        for name in ("1", "1.jsonl", f"{self.key}.jsonl"):
            with self.subTest(name=name):
                code, _, stderr = self.main(
                    ["--headless", "--cwd", str(self.work),
                     "--save", str(self.registry_dir / name), "--prompt", "p"], _answer("done"))
                self.assertEqual(code, 1)
                self.assertIn("._private is a symbolic link", stderr)
        self.assertEqual((full_log(self.work / "a").read_bytes(),
                          sorted(os.listdir(self.registry_dir))), before)

    def test_refusals_end_the_launch_before_any_save_changes(self):
        before = full_log(self.work / "a").read_bytes()
        cases = (
            (("1", "p", "--save", str(self.work / "a")), "give it or --save, not both"),
            (("1", "p", "--save", "interaction"), "give it or --save, not both"),
            (("1", "p", "--resume=False"), "--resume=False would replace it"),
            (("1#2", "p"), "has only #1"),
            (("9", "p"), "there's no root context 9"),
            (("tests", "p"), "names no root context"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                code, _, stderr = self.context(*argv)
                self.assertEqual(code, 1)
                self.assertIn(message, stderr)
        code, _, stderr = self.main(["--headless", "--context", "1", "--prompt", "p"],
                                    registry_dir="none")
        self.assertEqual(code, 1)
        self.assertIn("--context needs the registry, which is off (none)", stderr)
        self.assertEqual(full_log(self.work / "a").read_bytes(), before)
        shutil.rmtree(self.work / "a._private")
        code, _, stderr = self.context("1")
        self.assertEqual(code, 1)
        self.assertIn(f"isn't at {self.work}/a any more: open its save with --save", stderr)

    def test_a_reboot_check_resolves_the_context(self):
        code, _, stderr = self.main(["--rebooted=check", "--headless", "--context", "1"])
        self.assertEqual(code, 0, stderr)
        code, _, stderr = self.main(["--rebooted=check", "--headless", "--context", "2"])
        self.assertEqual(code, 1)
        self.assertIn("there's no root context 2", stderr)

    def test_list_roots_lists_and_exits_without_a_model(self):
        with mock.patch.object(cli, "build_model", side_effect=AssertionError("model built")):
            code, stdout, stderr = self.main(["--list-roots"])
        self.assertEqual(code, 0, stderr)
        lines = stdout.splitlines()
        self.assertEqual(lines[0], f"Root contexts in {self.registry_dir}/: 1, from 1 to 1.")
        self.assertRegex(lines[1], rf"^   1  {self.key[:12]}  ok  .*  {self.work}/a$")
        code, stdout, _ = self.main(["--list-roots"], registry_dir="none")
        self.assertEqual((code, stdout.strip()),
                         (0, "The registry is off: --registry or PYTHIA_REGISTRY is none."))
        code, stdout, _ = self.main(["--list-roots"], registry_dir=self.root / "empty")
        self.assertEqual(stdout.strip(), f"Root contexts in {self.root}/empty/: none yet.")
        if os.geteuid() != 0:
            self.registry_dir.chmod(0o000)
            self.addCleanup(self.registry_dir.chmod, 0o700)
            code, _, stderr = self.main(["--list-roots"])
            self.assertEqual(code, 1)
            self.assertIn("interaction CLI failed: [Errno 13] Permission denied", stderr)


class CLIInteractiveRegistrationTests(_ControllerTestCase):
    async def test_the_line_shows_in_the_transcript(self):
        directory = Path(self.path).parent / "root"
        model = _Model(self.path, _answer())
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        self.assertEqual(await self._run(model, terminal,
                                         ["--prompt", "hello", "--registry", str(directory)],
                                         Environment()), 0)
        lines = [item.text for item in terminal.items if item.text.startswith("[cli] Root")]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], rf"^\[cli\] Root context 1 \(context_[0-9a-f]{{4}}\), new in "
                                   rf"{directory}/\.$")


def load_key(prefix):
    with open(full_log(Path(f"{prefix}.jsonl"))) as log:
        return json.loads(log.readline())["prefix_id"]


if __name__ == "__main__":
    unittest.main()
