"""No Claude downloads/execution, sudo, account changes, or system installation.

Rootless setup uses only temporary private homes. Integration tests run ONLY
known system Python/Bash/Git payloads in Bubblewrap, never the real Claude source.
"""

import http.server
import threading
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pty
import select
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'claude-relay/one_click_claude.py'
SPEC = importlib.util.spec_from_file_location('one_click_claude_example', SCRIPT)
sandbox = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sandbox
SPEC.loader.exec_module(sandbox)


def elf_fixture(machine=None, entry=0x1000):
    machine = machine or {'x86_64': 62, 'aarch64': 183}.get(platform.machine(), 62)
    ident = b'\x7fELF\x02\x01\x01' + b'\0' * 9
    header = ident + struct.pack('<HHIQQQIHHHHHH', 2, machine, 1, entry, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    return header + struct.pack('<IIQQQQQQ', 1, 5, 0, 0x1000, 0x1000, 120, 120, 4096)


class UnitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_argv_is_preserved_and_wrapper_flags_do_not_consume_cli_help(self):
        options, args = sandbox.parse_args(['--sandbox-home', '/home/claude-private', '-v', '--help', '', 'x y', '--', '--sandbox-doctor'])
        self.assertEqual(options.home, '/home/claude-private')
        self.assertFalse(options.doctor)
        self.assertEqual(args, ['-v', '--help', '', 'x y', '--', '--sandbox-doctor'])
        options, args = sandbox.parse_args(['--sandbox-claude', '/a b/claude', '--sandbox-setup-only', '-p', 'hi'])
        self.assertEqual(options.claude, '/a b/claude')
        self.assertTrue(options.setup_only)
        self.assertEqual(args, ['-p', 'hi'])
        cli_args = ['--system-prompt', '--sandbox-home', '--sandbox-doctor', '--', '--_sandbox-run']
        options, args = sandbox.parse_args(cli_args)
        self.assertEqual(args, cli_args)
        self.assertIsNone(options.home)
        self.assertFalse(options.doctor)
        for args in (['--sandbox-unknown'], ['--sandbox-claude'], ['--_sandbox-run'],
                     ['--sandbox-doctor', '--sandbox-setup-only'], ['--sandbox-setup-for', '1234'], ['--sandbox-yes']):
            with self.subTest(args=args), self.assertRaises((sandbox.SandboxError, ValueError)):
                sandbox.parse_args(args)

    def test_default_uses_home_not_path(self):
        with mock.patch.dict(os.environ, {'HOME': str(self.root), 'PATH': '/arbitrary'}, clear=True):
            self.assertEqual(sandbox.default_source(), self.root / '.local/bin/claude')

    def test_static_native_inspection_and_snapshot_never_launch_source(self):
        real = self.root / 'native'
        real.write_bytes(elf_fixture())
        real.chmod(0o755)
        alias = self.root / 'claude'
        alias.symlink_to(real)
        with mock.patch.object(sandbox.subprocess, 'run', side_effect=AssertionError('must not execute')):
            with sandbox.native_source(alias) as (stream, resolved, info):
                self.assertEqual(sandbox.inspect_elf(stream), (None, []))
                self.assertEqual(resolved, real)
                self.assertEqual(info.st_size, 120)
            with sandbox.native_snapshot(alias) as (fd, resolved, sha):
                self.assertEqual(os.pread(fd, 1000, 0), real.read_bytes())
                self.assertEqual(os.fstat(fd).st_mode & 0o777, 0o555)
                self.assertEqual(resolved, real)
                self.assertEqual(sha, hashlib.sha256(real.read_bytes()).hexdigest())

    def test_rejects_scripts_bad_arch_truncation_no_entry_and_setid(self):
        path = self.root / 'native'
        for data in (b'#!/bin/sh\necho no\n', elf_fixture(machine=3), elf_fixture()[:80], elf_fixture(entry=0)):
            with self.subTest(data=data[:30]):
                path.write_bytes(data)
                path.chmod(0o755)
                with self.assertRaises(sandbox.SandboxError), sandbox.native_source(path):
                    self.fail('accepted bad native file')
        path.write_bytes(elf_fixture())
        path.chmod(0o4755)
        with self.assertRaises(sandbox.SandboxError), sandbox.native_source(path):
            self.fail('accepted setuid file')

    def test_fifo_source_fails_without_blocking(self):
        path = self.root / 'fifo'
        os.mkfifo(path, 0o700)
        with self.assertRaises(sandbox.SandboxError), sandbox.native_source(path):
            self.fail('accepted a FIFO')

    def test_static_system_dependency_inspection(self):
        with sandbox.native_source(Path('/usr/bin/true')) as (stream, _, _):
            _, dependencies = sandbox.inspect_elf(stream)
            sandbox.check_libraries(dependencies)
        with self.assertRaisesRegex(sandbox.SandboxError, 'missing system library'):
            sandbox.check_libraries(['lib_one_click_intentionally_missing.so'])

    def test_environment_is_explicit_and_config_cannot_escape(self):
        home = self.root / 'home'
        home.mkdir()
        env = sandbox.sandbox_env(home, 'claude-test', 1001, {'TERM': 'xterm', 'HTTPS_PROXY': 'http://localhost:1'})
        self.assertEqual(env['HOME'], str(home))
        self.assertEqual(env['HTTPS_PROXY'], 'http://localhost:1')
        self.assertNotIn('ANTHROPIC_API_KEY', env)
        self.assertNotIn('ENABLE_TOOL_SEARCH', env)
        for key in ('PYTHONPATH', 'LD_PRELOAD', 'SSH_AUTH_SOCK', 'ANTHROPIC_API_KEY'):
            with self.subTest(key=key), self.assertRaises(sandbox.SandboxError):
                sandbox.sandbox_env(home, 'claude-test', 1001, {key: 'no'})
        for config in ('relative', '/tmp/outside', str(home / '../escape')):
            with self.subTest(config=config), self.assertRaises(sandbox.SandboxError):
                sandbox.sandbox_env(home, 'claude-test', 1001, {'CLAUDE_CONFIG_DIR': config})
        (home / 'link').symlink_to('/tmp')
        with self.assertRaises(sandbox.SandboxError):
            sandbox.sandbox_env(home, 'claude-test', 1001, {'CLAUDE_CONFIG_DIR': str(home / 'link/cfg')})
        env = sandbox.sandbox_env(home, 'claude-test', 1001, {'CLAUDE_CONFIG_DIR': str(home / 'config')})
        self.assertEqual(env['CLAUDE_CONFIG_DIR'], str(home / 'config'))

    def test_mount_overlap_and_submounts_fail_closed(self):
        for home in ('/', '/usr', '/usr/bin/home', '/tmp', '/etc/home'):
            with self.subTest(home=home), self.assertRaises(sandbox.SandboxError), sandbox.sandbox_command(
                    self.root, Path(home), 'claude-test', os.getuid(), os.getgid(), ['/usr/bin/true']):
                self.fail('accepted overlap')
        with mock.patch.object(sandbox, 'mount_points', return_value=[self.root / 'nested']):
            with self.assertRaises(sandbox.SandboxError):
                sandbox.reject_submounts(self.root)
        with mock.patch.object(sandbox, 'mount_points', return_value=[self.root]):
            sandbox.reject_submounts(self.root)
            with self.assertRaises(sandbox.SandboxError):
                sandbox.reject_submounts(self.root, include_self=True)

    def test_mountinfo_escapes(self):
        value = '44 1 0:1 / /a\\040b\\134c rw - tmpfs tmpfs rw\n'
        with mock.patch.object(Path, 'read_text', return_value=value):
            self.assertEqual(sandbox.mount_points(), [Path('/a b\\c')])

    def test_trusted_path_rejects_symlink_and_nonroot_source(self):
        (self.root / 'link').symlink_to('/usr/bin/true')
        with self.assertRaises(sandbox.SandboxError):
            sandbox.trusted(self.root / 'link')
        if os.getuid() != 0:
            (self.root / 'file').write_text('not root-owned')
            with self.assertRaises(sandbox.SandboxError):
                sandbox.trusted(self.root / 'file')

    def test_no_privileged_identity_is_accepted(self):
        for real, effective in ((0, 0), (1234, 0)):
            with mock.patch.object(sandbox.os, 'getuid', return_value=real), \
                    mock.patch.object(sandbox.os, 'geteuid', return_value=effective):
                with self.assertRaisesRegex(sandbox.SandboxError, 'non-root account'):
                    sandbox.current_identity()

    def test_bwrap_failure_never_launches_payload(self):
        with mock.patch.object(sandbox, 'BWRAP', self.root / 'absent'):
            with self.assertRaisesRegex(sandbox.SandboxError, 'bubblewrap'):
                sandbox.check_bwrap()

    def test_standard_directory_descriptors_cannot_bypass_isolation(self):
        with mock.patch.object(sandbox.os, 'fstat', return_value=SimpleNamespace(st_mode=stat.S_IFDIR | 0o700)):
            with self.assertRaisesRegex(sandbox.SandboxError, 'directory FDs'):
                sandbox.validate_stdio()
        for kind in (stat.S_IFREG, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR):
            with mock.patch.object(sandbox.os, 'fstat', return_value=SimpleNamespace(st_mode=kind | 0o600)):
                sandbox.validate_stdio()

    def test_no_native_launch_error_is_noninteractive_and_stdout_clean(self):
        result = subprocess.run(['/usr/bin/python3', '-I', str(SCRIPT), '--sandbox-claude', str(self.root / 'absent')],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, '')
        self.assertIn('no download', result.stderr)

    def test_isolated_help_ignores_import_injection_and_requires_isolation(self):
        (self.root / 'json.py').write_text("raise AssertionError('unsafe import')\n")
        result = subprocess.run(['/usr/bin/python3', '-I', str(SCRIPT), '--sandbox-help'],
                                cwd=self.root, env={**os.environ, 'PYTHONPATH': str(self.root)},
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Usage:', result.stdout)
        result = subprocess.run(['/usr/bin/python3', str(SCRIPT), '--sandbox-help'],
                                env=sandbox.SAFE_ENV, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 125)
        self.assertIn('python3 -I', result.stderr)

    def test_cli_path_env_defaults_and_cli_override(self):
        env = {'HOME': str(self.root), 'ONE_CLICK_CLAUDE_HOME': str(self.root / 'private'),
               'ONE_CLICK_CLAUDE_BIN': '/usr/bin/python3'}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(sandbox.default_source(), Path('/usr/bin/python3'))
            self.assertEqual(sandbox.private_home(sandbox.Options()), self.root / 'private')
            self.assertEqual(sandbox.private_home(sandbox.Options(home=str(self.root / 'explicit'))),
                             self.root / 'explicit')

    def test_no_privilege_or_package_manager_dependency(self):
        self.assertFalse({'sudo', 'passwd', 'uidmap', 'socat'} & set(sandbox.DEPENDENCIES))
        for removed in ('bootstrap', 'install_from_stream', 'root_setup', 'sudoers_text', 'system_command'):
            self.assertFalse(hasattr(sandbox, removed))

    def test_harness_mcp_tuning_is_forwarded_only_when_supplied(self):
        keys = {'CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS': '0', 'ENABLE_TOOL_SEARCH': 'false'}
        env = sandbox.sandbox_env(self.root, 'claude-test', os.getuid(), keys)
        for key, value in keys.items():
            self.assertEqual(env[key], value)
            self.assertNotIn(key, sandbox.sandbox_env(self.root, 'claude-test', os.getuid(), {}))

    def test_private_layout_excludes_trusted_code_and_binary(self):
        with self.assertRaisesRegex(sandbox.SandboxError, 'launcher must be outside'):
            sandbox.check_private_layout(SCRIPT.parent)
        with self.assertRaisesRegex(sandbox.SandboxError, 'native source must be outside'):
            sandbox.check_private_layout(self.root, self.root / 'bin/claude')
        link = self.root / 'native-link'
        private = self.root / 'private'
        private.mkdir()
        target = private / 'binary'
        target.write_bytes(elf_fixture())
        link.symlink_to(target)
        with self.assertRaisesRegex(sandbox.SandboxError, 'native source must be outside'):
            sandbox.check_private_layout(private, link)

    def test_rejects_hardlinked_trusted_native(self):
        source = self.root / 'native'
        source.write_bytes(elf_fixture())
        source.chmod(0o755)
        os.link(source, self.root / 'alias')
        with self.assertRaisesRegex(sandbox.SandboxError, 'hardlink'), sandbox.native_source(source):
            self.fail('hardlinked source accepted')


class RootlessSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / 'private/home'

    def test_creates_owned_home_and_staging_directory_without_system_changes(self):
        with mock.patch.object(sandbox.subprocess, 'run', side_effect=AssertionError('no setup commands')):
            sandbox.prepare_home(self.home)
            sandbox.prepare_home(self.home)
        self.assertEqual(self.home.stat().st_uid, os.getuid())
        self.assertEqual(self.home.stat().st_mode & 0o777, 0o700)
        self.assertTrue((self.home / 'work').is_dir())
        self.assertEqual((self.home / 'work').stat().st_uid, os.getuid())

    def test_concurrent_preparation_is_idempotent(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(sandbox.prepare_home, [self.home] * 4))
        self.assertTrue((self.home / 'work').is_dir())

    def test_never_repairs_or_adopts_a_shared_home(self):
        self.home.mkdir(parents=True, mode=0o755)
        self.home.chmod(0o755)
        with self.assertRaisesRegex(sandbox.SandboxError, '0700'):
            sandbox.prepare_home(self.home)
        self.assertEqual(self.home.stat().st_mode & 0o777, 0o755)

    def test_work_symlink_is_rejected_without_touching_its_target(self):
        self.home.mkdir(parents=True, mode=0o700)
        outside = self.root / 'outside'
        outside.mkdir()
        (self.home / 'work').symlink_to(outside)
        with self.assertRaisesRegex(sandbox.SandboxError, 'not an alias'):
            sandbox.prepare_home(self.home)
        self.assertEqual(list(outside.iterdir()), [])

    def test_unsafe_ancestor_is_rejected(self):
        (self.root / 'private').mkdir(mode=0o777)
        (self.root / 'private').chmod(0o777)
        with self.assertRaisesRegex(sandbox.SandboxError, 'unsafe ownership'):
            sandbox.prepare_home(self.home)
        self.assertFalse(self.home.exists())

    def test_print_home_is_read_only_and_does_not_need_a_native(self):
        result = subprocess.run(['/usr/bin/python3', '-I', str(SCRIPT), '--sandbox-home', str(self.home),
                                 '--sandbox-print-home'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.home))
        self.assertFalse(self.home.exists())


@unittest.skipUnless(sys.platform == 'linux' and Path('/usr/bin/bwrap').exists(), 'requires Linux Bubblewrap')
class BubblewrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Distinguish an unavailable namespace facility from a broken wrapper.
        result = subprocess.run(['/usr/bin/bwrap', '--unshare-user', '--unshare-pid',
                                 '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib',
                                 '--symlink', 'usr/lib64', '/lib64', '--', '/usr/bin/true'],
                                capture_output=True, text=True)
        if result.returncode:
            raise unittest.SkipTest('user namespaces unavailable: ' + result.stderr)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home_source = self.root / 'home'
        self.home_source.mkdir(mode=0o700)
        self.home = Path('/home/claude-fixture')

    def command(self, code, args=(), **kwargs):
        return sandbox.sandbox_command(self.home_source, self.home, 'claude-fixture', os.getuid(), os.getgid(),
                                       ['/usr/bin/python3', '-I', '-c', code, *args], **kwargs)

    def wrapper_env(self, **extra):
        # An explicit known system executable stands in for Claude in ALL full
        # entrypoint tests. Never discover/read/run the real default source.
        return {**sandbox.SAFE_ENV, 'HOME': str(self.root),
                'ONE_CLICK_CLAUDE_HOME': str(self.home_source),
                'ONE_CLICK_CLAUDE_BIN': '/usr/bin/python3', **extra}

    def test_full_entrypoint_setup_is_unprivileged_and_never_executes_source(self):
        # Synthetic ELF is inspectable but not a runnable program. Setup must
        # succeed with it, proving setup never calls even a version probe on it.
        native = self.root / 'not-a-runnable-program'
        native.write_bytes(elf_fixture())
        native.chmod(0o755)
        result = subprocess.run([str(SCRIPT), '--sandbox-setup-only'],
                                env=self.wrapper_env(ONE_CLICK_CLAUDE_BIN=str(native), TMPDIR=str(self.home_source)),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('no Claude executed', result.stderr)
        self.assertTrue((self.home_source / 'work').is_dir())
        self.assertEqual(native.read_bytes(), elf_fixture())

    def test_full_entrypoint_version_probe_needs_no_bootstrap_or_prompt(self):
        result = subprocess.run([str(SCRIPT), '--version'], env=self.wrapper_env(),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith('Python 3.'), result.stdout)
        self.assertEqual(result.stderr, '')

    def test_full_cli_path_staging_mcp_env_protocol_and_local_http(self):
        sandbox.prepare_home(self.home_source)
        stage = self.home_source / 'runs/generation'
        stage.mkdir(parents=True)
        (stage / 'system.txt').write_text('staged prompt, not executed')
        (stage / 'settings.json').write_text('{"tools": []}')
        (stage / 'config').mkdir()
        outside = self.root / 'host-project'
        outside.mkdir()
        (outside / 'host-secret').write_text('outside private home')

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get('Authorization') != 'Bearer synthetic-generation':
                    self.send_error(403)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'synthetic MCP listener; no tools executed')

            def log_message(self, *args):
                pass

        with http.server.HTTPServer(('127.0.0.1', 0), Handler) as listener:
            thread = threading.Thread(target=listener.serve_forever, daemon=True)
            thread.start()
            code = r'''
import json, os, pathlib, sys, urllib.request
stage, outside, url, *args = sys.argv[1:]
assert not pathlib.Path(outside).exists(), 'host cwd was mounted'
assert os.getcwd() == str(pathlib.Path.home() / 'work')
assert pathlib.Path(stage, 'system.txt').read_text() == 'staged prompt, not executed'
assert json.loads(pathlib.Path(stage, 'settings.json').read_text()) == {'tools': []}
assert os.environ['CLAUDE_CONFIG_DIR'] == stage + '/config'
assert os.environ['CLAUDE_CODE_ENTRYPOINT'] == 'sdk-test'
assert os.environ['CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS'] == '0'
assert os.environ['ENABLE_TOOL_SEARCH'] == 'false'
assert 'ONE_CLICK_CLAUDE_BIN' not in os.environ
assert 'ONE_CLICK_CLAUDE_HOME' not in os.environ
request = urllib.request.Request(url, headers={'Authorization': 'Bearer synthetic-generation'})
with urllib.request.urlopen(request, timeout=3) as response:
    assert response.read() == b'synthetic MCP listener; no tools executed'
for line in sys.stdin:
    print(json.dumps({'input': json.loads(line), 'args': args, 'uid': os.getuid()}), flush=True)
print('payload diagnostic', file=sys.stderr)
raise SystemExit(17)
'''
            env = self.wrapper_env(CLAUDE_CODE_ENTRYPOINT='sdk-test',
                                   CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS='0', ENABLE_TOOL_SEARCH='false',
                                   CLAUDE_CONFIG_DIR=str(stage / 'config'))
            args = ['-v', '--help', '', 'spaces and quotes "', '--sandbox-home', '--sandbox-doctor']
            try:
                result = subprocess.run([str(SCRIPT), '-I', '-c', code, str(stage), str(outside),
                                         f'http://127.0.0.1:{listener.server_port}/mcp', *args],
                                        cwd=outside, env=env, input='{"a":1}\n{"b":2}\n',
                                        text=True, capture_output=True, timeout=10)
            finally:
                listener.shutdown()
                thread.join(timeout=3)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(result.stderr, 'payload diagnostic\n')
        self.assertEqual([json.loads(line) for line in result.stdout.splitlines()],
                         [{'input': value, 'args': args, 'uid': os.getuid()} for value in ({'a': 1}, {'b': 2})])

    def test_full_entrypoint_keeps_only_inside_cwd_and_rejects_external_config(self):
        sandbox.prepare_home(self.home_source)
        inside = self.home_source / 'work/inside'
        inside.mkdir()
        result = subprocess.run([str(SCRIPT), '--', '-I', '-c', 'import os; print(os.getcwd())'],
                                cwd=inside, env=self.wrapper_env(), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(inside))
        result = subprocess.run([str(SCRIPT), '--version'],
                                env=self.wrapper_env(CLAUDE_CONFIG_DIR=str(self.root / 'external-config')),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 125)
        self.assertEqual(result.stdout, '')
        self.assertIn('CLAUDE_CONFIG_DIR', result.stderr)

    def test_full_cli_path_termination_and_interruption_clear_descendants(self):
        code = """
import subprocess, time
subprocess.Popen(['/usr/bin/python3', '-I', '-c', 'import time; time.sleep(60)'])
print('READY', flush=True)
time.sleep(60)
"""

        def descendants(pid):
            try:
                children = [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
            except FileNotFoundError:
                return []
            return children + [grandchild for child in children for grandchild in descendants(child)]

        def live(pid):
            try:
                text = Path(f'/proc/{pid}/status').read_text()
            except FileNotFoundError:
                return False
            state = next(line for line in text.splitlines() if line.startswith('State:'))
            return 'Z (zombie)' not in state

        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig), subprocess.Popen(
                    [str(SCRIPT), '--', '-I', '-c', code], env=self.wrapper_env(),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                try:
                    self.assertTrue(select.select([process.stdout], [], [], 10)[0], 'no startup response')
                    self.assertEqual(process.stdout.readline(), 'READY\n')
                    children = descendants(process.pid)
                    self.assertGreaterEqual(len(children), 2)
                    process.send_signal(sig)
                    _, errors = process.communicate(timeout=5)
                    self.assertIn(process.returncode, (-sig, 128 + sig), errors)
                    deadline = time.monotonic() + 3
                    while any(live(child) for child in children) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertFalse(any(live(child) for child in children), 'live sandbox descendants remain')
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate(timeout=5)

    def test_doctor_without_native(self):
        sandbox.check_bwrap()
        sandbox.probe(self.home_source, self.home, 'claude-fixture', os.getuid(), os.getgid())

    def test_persistence_symlink_escape_stdio_and_exact_argv(self):
        sentinel = self.root / 'secret'
        sentinel.write_text('private outside')
        (self.home_source / 'escape').symlink_to(sentinel)
        args = ['-v', '', 'spaces and "quotes"', '--ro-bind', '/', '/']
        code = """
import json, os, pathlib, sys
home = pathlib.Path.home()
assert not (home / 'escape').exists()
assert not os.path.exists('/etc/shadow')
assert not os.path.exists('/etc/sudoers')
assert not os.path.exists('/run/user/%s/bus' % os.getuid())
assert os.getcwd() == str(home / 'work')
(home / 'persist').write_text('yes')
print(json.dumps([sys.argv[1:], sys.stdin.read()]))
print('stderr-separate', file=sys.stderr)
sys.exit(7)
"""
        with self.command(code, args) as (command, fds):
            self.assertNotIn('--ro-bind / /', ' '.join(command[:command.index('--')]))
            result = subprocess.run(command, pass_fds=fds, input='pipe input', capture_output=True,
                                    text=True, env=sandbox.SAFE_ENV, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(json.loads(result.stdout), [args, 'pipe input'])
        self.assertEqual(result.stderr, 'stderr-separate\n')
        self.assertEqual((self.home_source / 'persist').read_text(), 'yes')
        self.assertEqual(sentinel.read_text(), 'private outside')

    def test_hermetic_environment_ignores_injection_and_hosts_work(self):
        code = """
import os, socket, ssl
assert 'PYTHONPATH' not in os.environ
assert 'LD_PRELOAD' not in os.environ
assert 'ANTHROPIC_API_KEY' not in os.environ
assert 'SSH_AUTH_SOCK' not in os.environ
assert os.environ['TERM'] == 'xterm-256color'
assert socket.getaddrinfo('localhost', 80)
assert ssl.create_default_context().cert_store_stats()['x509_ca'] > 0
print('clean')
"""
        with self.command(code, supplied={'TERM': 'xterm-256color'}) as (command, fds):
            result = subprocess.run(command, pass_fds=fds, capture_output=True, text=True, timeout=10,
                                    env={**sandbox.SAFE_ENV, 'PYTHONPATH': str(self.root),
                                         'ANTHROPIC_API_KEY': 'must-not-leak', 'SSH_AUTH_SOCK': '/host/socket'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'clean\n')

    def test_payload_has_no_inherited_host_fd(self):
        with (self.root / 'outside').open('w') as extra:
            os.set_inheritable(extra.fileno(), True)
            with self.command("import os; print(os.listdir('/proc/self/fd'))") as (command, fds):
                result = subprocess.run(command, pass_fds=fds, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        # listdir temporarily opens fd 3 itself; it has closed by the time probe checks.
        self.assertEqual(result.stdout.strip(), "['0', '1', '2', '3']")

    def test_basic_coding_tools_and_subprocesses_are_confined(self):
        code = """
import subprocess
result = subprocess.run(['/bin/bash', '-c', 'test ! -e /etc/shadow && git --version'],
                        capture_output=True, text=True, check=True)
assert result.stdout.startswith('git version ')
print('tools OK')
"""
        with self.command(code) as (command, fds):
            result = subprocess.run(command, pass_fds=fds, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'tools OK\n')

    def test_bad_bwrap_command_does_not_run_payload(self):
        with self.command("raise AssertionError('payload must never run')") as (command, fds):
            command.insert(1, '--deliberately-invalid-sandbox-option')
            result = subprocess.run(command, pass_fds=fds, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('AssertionError', result.stderr)

    def test_separate_interactive_terminal_resize_and_ctrl_c(self):
        inner = r'''
import fcntl, os, signal, struct, sys, termios
assert os.isatty(0) and os.isatty(1)
with open('/dev/tty', 'wb', buffering=0) as terminal:
    terminal.write(b'TTY-OK\n')
def resized(*args):
    size = struct.unpack('HHHH', fcntl.ioctl(0, termios.TIOCGWINSZ, b'\0' * 8))
    print('SIZE:%s:%s' % size[:2], flush=True)
signal.signal(signal.SIGWINCH, resized)
print('READY', flush=True)
try:
    while True:
        line = input()
        print('ECHO:' + line, flush=True)
except KeyboardInterrupt:
    print('INTERRUPTED', flush=True)
    sys.exit(23)
'''
        driver = f'''
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location('driver_sandbox', {str(SCRIPT)!r})
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
with m.sandbox_command(pathlib.Path({str(self.home_source)!r}), pathlib.Path({str(self.home)!r}),
                       'claude-fixture', os.getuid(), os.getgid(),
                       ['/usr/bin/python3', '-I', '-c', {inner!r}], interactive=True) as (command, fds):
    raise SystemExit(m.terminal_run(command, fds))
'''
        master, slave = pty.openpty()
        initial = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, 80, 0, 0))

        def controlling_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        process = subprocess.Popen(['/usr/bin/python3', '-I', '-c', driver],
                                   stdin=slave, stdout=slave, stderr=slave,
                                   preexec_fn=controlling_terminal)
        output = bytearray()

        def until(expected):
            deadline = time.monotonic() + 10
            while expected not in output and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        data = os.read(master, 65536)
                    except OSError as error:
                        if error.errno == errno.EIO:
                            break
                        raise
                    if not data:
                        break
                    output.extend(data)
                if process.poll() is not None:
                    break
            self.assertIn(expected, output, output.decode(errors='replace'))

        try:
            until(b'READY')
            self.assertIn(b'TTY-OK', output)
            os.write(master, b'hello\n')
            until(b'ECHO:hello')
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', 41, 103, 0, 0))
            until(b'SIZE:41:103')
            os.write(master, b'\x03')
            until(b'INTERRUPTED')
            self.assertEqual(process.wait(timeout=10), 23, output.decode(errors='replace'))
            self.assertEqual(termios.tcgetattr(slave), initial, 'host terminal not restored')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            os.close(master)
            os.close(slave)


if __name__ == '__main__':
    unittest.main()
