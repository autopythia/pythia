"""Transport tests use a fixed stdlib Python wrapper fixture, NEVER Claude."""
import array
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'claude-relay/claude_relay.py'
spec = importlib.util.spec_from_file_location('relay_fixture_module', SCRIPT)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)

FIXTURE = r'''
import os, subprocess, sys, time
assert sys.argv[1] == '--'
args = sys.argv[2:]
if args == ['--version']:
    print('2.1.0-fixture', flush=True)
elif args == ['wait']:
    # Emulate bwrap's namespace-init descendant lifetime in this Python-only fixture.
    p = subprocess.Popen(['/usr/bin/python3', '-I', '-c',
        'import ctypes, signal, time; ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0); print("ready", flush=True); time.sleep(60)'],
        stdout=subprocess.PIPE)
    p.stdout.readline()
    p.stdout.close()
    print('%s %s' % (os.getpid(), p.pid), flush=True)
    time.sleep(60)
else:
    print(json.dumps(args) if False else 'READY', flush=True)
    data = sys.stdin.buffer.read()
    sys.stderr.buffer.write(b'E' * 200000)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    sys.exit(7)
'''


@unittest.skipUnless(sys.platform == 'linux' and os.getuid() != 0, 'non-root Linux required')
class RelayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wrapper = self.root / 'wrapper.py'
        self.wrapper.write_text(FIXTURE)
        self.socket = self.root / 'control/control.sock'
        self.env = {**relay.SAFE_ENV, 'CLAUDE_RELAY_SOCKET': str(self.socket),
                    'CLAUDE_RELAY_SERVER_UID': str(os.getuid())}
        self.broker = subprocess.Popen([str(SCRIPT), '--relay-serve', '--relay-socket', str(self.socket),
                                       '--relay-allow-uid', str(os.getuid()), '--relay-wrapper', str(self.wrapper),
                                       '--relay-stop-grace', '0.2'], env=relay.SAFE_ENV,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.stop_broker)
        end = time.monotonic() + 5
        while not self.socket.exists() and self.broker.poll() is None and time.monotonic() < end:
            time.sleep(.02)
        self.assertTrue(self.socket.exists(), self.broker.poll())

    def stop_broker(self):
        if self.broker.poll() is None:
            self.broker.terminate()
        self.broker.communicate(timeout=6)

    def invoke(self, *args, **kw):
        return subprocess.run([str(SCRIPT), *args], env=self.env, capture_output=True, timeout=10, **kw)

    def test_health_and_version(self):
        result = self.invoke('--relay-check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['uid'], os.getuid())
        result = self.invoke('--version', input=b'')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'2.1.0-fixture\n')
        self.assertEqual(result.stderr, b'')

    def test_large_binary_streams_eof_exit_and_literal_arguments(self):
        data = bytes(range(256)) * 12000
        result = self.invoke('echo', '--relay-serve', '--sandbox-home', 'a b', '', input=data)
        self.assertEqual(result.returncode, 7, result.stderr[:100])
        self.assertEqual(result.stdout, b'READY\n' + data)
        self.assertEqual(result.stderr, b'E' * 200000)

    def test_wrong_peer_and_missing_endpoint_fail_without_fallback(self):
        env = {**self.env, 'CLAUDE_RELAY_SERVER_UID': str(os.getuid() + 1)}
        result = subprocess.run([str(SCRIPT), '--version'], env=env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, b'')
        self.stop_broker()
        result = self.invoke('--version')
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, b'')

    def test_signal_disconnect_and_broker_death(self):
        for mode in ('term', 'kill', 'broker'):
            with self.subTest(mode=mode):
                process = subprocess.Popen([str(SCRIPT), 'wait'], env=self.env, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    import select
                    self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                    line = process.stdout.readline()
                    self.assertTrue(line.strip(), 'child did not start')
                    children = [int(n) for n in line.split()]
                    if mode == 'broker':
                        self.broker.kill()
                    elif mode == 'kill':
                        process.kill()
                    else:
                        process.terminate()
                    process.communicate(timeout=7)
                    end = time.monotonic() + 5
                    def live(pid):
                        try:
                            return 'Z (zombie)' not in next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:'))
                        except (FileNotFoundError, ProcessLookupError):
                            return False  # reaped before the open, or between it and the read (ESRCH)
                    while any(live(pid) for pid in children) and time.monotonic() < end:
                        time.sleep(.02)
                    self.assertFalse(any(live(pid) for pid in children), children)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def test_bad_descriptor_transfer_is_rejected(self):
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(str(self.socket))
            wire = relay.Wire(connection)
            self.assertEqual(wire.read(len(relay.MAGIC)), relay.MAGIC)
            wire.receive()
            wire.send({'op': 'START', 'argv': ['--version'], 'env': {}})
            self.assertEqual(wire.receive()['op'], 'READY_FOR_FDS')
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                connection.sendmsg([b'F'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd] * 3))])
            finally:
                os.close(fd)
            self.assertEqual(wire.receive()['op'], 'ERROR')
        self.assertEqual(self.invoke('--version').returncode, 0)

    def test_start_guard_refuses_an_already_missing_owner(self):
        result = subprocess.run([str(SCRIPT), '--relay-child', '999999999', str(self.wrapper), '--', '--version'],
                                env=self.env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, b'')

    def test_controller_death_kills_its_client_and_remote_child(self):
        code = '''
import os, subprocess, sys, time
env = dict(os.environ, CLAUDE_RELAY_PARENT_PID=str(os.getpid()))
child = subprocess.Popen([sys.argv[1], 'wait'], env=env, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(str(child.pid) + ' ' + child.stdout.readline().decode().strip(), flush=True)
time.sleep(60)
'''
        with subprocess.Popen(['/usr/bin/python3', '-I', '-c', code, str(SCRIPT)], env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as controller:
            try:
                import select
                self.assertTrue(select.select([controller.stdout], [], [], 5)[0])
                ids = [int(value) for value in controller.stdout.readline().split()]
                self.assertEqual(len(ids), 3)
                controller.kill()
                controller.communicate(timeout=5)
                deadline = time.monotonic() + 5
                def live(pid):
                    try:
                        text = Path(f'/proc/{pid}/status').read_text()
                        return 'Z (zombie)' not in next(line for line in text.splitlines() if line.startswith('State:'))
                    except (FileNotFoundError, ProcessLookupError):
                        return False  # reaped before the open, or between it and the read (ESRCH)
                while any(live(pid) for pid in ids) and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertFalse(any(live(pid) for pid in ids))
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.communicate(timeout=5)


class CodecTests(unittest.TestCase):
    def test_start_rejects_policy_override(self):
        for env in ({'HOME': '/host'}, {'ONE_CLICK_CLAUDE_BIN': '/evil'}, {'LD_PRELOAD': '/evil'}):
            with self.assertRaises(relay.RelayError):
                relay.validate_start({'op': 'START', 'argv': [], 'env': env})
        with self.assertRaises(relay.RelayError):
            relay.parse_json(b'{"op":"START","op":"CHECK"}')


if __name__ == '__main__':
    unittest.main()
