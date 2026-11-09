"""A CLI save's lifetime lock (_save_lock): held while a CLI runs, and tested by its readers."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from pythia.interaction import _save_lock
from pythia.interaction._save_lock import LOCK_NAME, adopt, describe, holder, take

_ROOT = Path(__file__).resolve().parents[1]
# Takes the lock of the save whose private directory is argv[1], says whether it could,
# and holds it until its input ends.
_HOLDER = ("import sys\n"
           "from pythia.interaction._save_lock import take\n"
           "lock, owner = take(sys.argv[1])\n"
           "print('taken' if lock else 'in use', flush=True)\n"
           "sys.stdin.read()\n")


class SaveLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "s._private"
        os.mkdir(self.private, 0o700)

    def take(self):
        lock, owner = take(self.private)
        self.assertIsNone(owner)
        self.addCleanup(lock.close)
        return lock

    def test_a_cli_holds_it_and_its_readers_see_who(self):
        self.assertIsNone(holder(self.private))  # no lock file yet
        lock = self.take()
        owner = holder(self.private)
        self.assertEqual((owner["program"], owner["pid"]), ("pythia", os.getpid()))
        self.assertLess(abs(owner["since"] - time.time()), 60)
        self.assertEqual(os.stat(self.private / LOCK_NAME).st_mode & 0o777, 0o600)
        self.assertEqual(take(self.private), (None, owner))  # a second CLI finds it in use
        lock.close()
        self.assertIsNone(holder(self.private))  # the file stays, free
        self.assertTrue((self.private / LOCK_NAME).is_file())
        self.take()

    def test_another_process_holds_it_until_it_ends(self):
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
        with subprocess.Popen([sys.executable, "-c", _HOLDER, str(self.private)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              env=environment) as child:
            try:
                self.assertEqual(child.stdout.readline(), "taken\n")
                self.assertEqual(holder(self.private)["pid"], child.pid)
                lock, owner = take(self.private)
                self.assertEqual((lock, owner["pid"]), (None, child.pid))
            finally:
                child.kill()  # the kernel lets the lock go with the process
        self.assertIsNone(holder(self.private))

    def test_a_line_that_makes_no_sense_names_no_one(self):
        lock = self.take()
        wrong = b'{"pid": "12", "since": true, "node": "n", "boot": 1}'
        for line, owner in ((wrong, {"node": "n"}), (b'{"pid": 12, "since": -1}', {"pid": 12}),
                            (b"[12]", {}), (b"not json", {}), (b"", {})):
            with self.subTest(line=line):
                os.ftruncate(lock.fileno(), 0)
                os.pwrite(lock.fileno(), line, 0)
                self.assertEqual(holder(self.private), owner)
        self.assertEqual(describe({}), "PID unknown")
        self.assertEqual(describe({"pid": 7, "node": "n", "since": 0}),
                         "PID 7 on n" + time.strftime(", since %Y-%m-%d %H:%M", time.localtime(0)))

    def test_a_restarted_cli_adopts_the_lock_it_inherited(self):
        lock = self.take()
        inherited = os.dup(lock.fileno())  # as exec keeps it: the same open file description
        os.set_inheritable(inherited, True)
        lock.close()
        self.assertIsNotNone(holder(self.private))  # the copy holds it
        adopted = adopt(inherited, self.private)
        self.addCleanup(adopted.close)
        self.assertEqual(adopted.fileno(), inherited)
        self.assertFalse(os.get_inheritable(inherited))
        self.assertEqual(holder(self.private)["pid"], os.getpid())
        # Anything else adopts nothing, and is closed: another file, which it could lock,
        # a directory, or the lock file through another description, which the lock keeps
        # out.
        for other in (os.open(self.private / "other", os.O_RDWR | os.O_CREAT, 0o600),
                      os.open(self.private, os.O_RDONLY),
                      os.open(self.private / LOCK_NAME, os.O_RDWR)):
            self.assertIsNone(adopt(other, self.private))
            with self.assertRaises(OSError):
                os.fstat(other)
        self.assertIsNone(adopt(2, self.private))
        os.fstat(2)  # standard error stays open
        self.assertEqual(holder(self.private)["pid"], os.getpid())

    def test_flock_where_the_system_has_no_open_file_description_locks(self):
        with mock.patch.object(_save_lock, "_OFD", False), \
                mock.patch.object(_save_lock, "_TRIES", 100):
            lock = self.take()
            self.assertEqual(holder(self.private)["pid"], os.getpid())
            self.assertIsNone(take(self.private)[0])
            lock.close()
            self.assertIsNone(holder(self.private))
            # A reader's test, a shared lock for a moment, only delays a CLI that starts then.
            reader = os.open(self.private / LOCK_NAME, os.O_RDONLY)
            fcntl.flock(reader, fcntl.LOCK_SH)
            threading.Timer(0.05, os.close, (reader,)).start()
            self.take()
            self.assertEqual(holder(self.private)["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
