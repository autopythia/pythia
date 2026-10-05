#!/usr/bin/python3 -IS
"""Stdlib-only Linux client/broker for one_click_claude.py. No UID switching.

Already as B:
  ./claude_relay.py --relay-serve --relay-socket /tmp/claude-relay-B/control.sock \
      --relay-allow-uid A
As A, set CLAUDE_RELAY_SOCKET and CLAUDE_RELAY_SERVER_UID, then invoke this file
with native CLI arguments. --relay-check checks only the broker, never Claude.
The native source/profile is configured ONLY in B's ONE_CLICK_CLAUDE_* environment.
See README.md for authority boundaries and the experimental native-client status.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import pwd
import queue
import re
import secrets
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time

PYTHON = '/usr/bin/python3'
MAGIC = b'PYCR\x00\x01'
MAX_RECORD = 1_048_576
MAX_ENV = 32_768
SAFE_ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}
# Values are data, not proof that the official SDK is in use. No inherited secrets.
# TODO(output-budgets): negotiate the broker/wrapper capability intersection in
# HELLO/--relay-check before adding a verified output-cap env key here. Old peers
# must reject requested budgets rather than silently dropping them. See SAMPLING.md.
FORWARD_ENV = frozenset((
    'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_AGENT_SDK_CLIENT_APP', 'TRACEPARENT', 'TRACESTATE',
    'CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS', 'ENABLE_TOOL_SEARCH',
    'DISABLE_AUTO_COMPACT', 'PYTHIA_CLAUDE_MCP_TOKEN',
))
SIGNALS = {'INT': signal.SIGINT, 'TERM': signal.SIGTERM, 'HUP': signal.SIGHUP, 'KILL': signal.SIGKILL}


class RelayError(Exception):
    """Only bounded non-secret diagnostics belong in this exception."""


def require(value, message):
    if not value:
        raise RelayError(message)


def log(message):
    os.write(2, ('claude-relay: ' + message + '\n').encode('utf-8', 'replace'))


def parse_json(data):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'duplicate control key')
            result[key] = value
        return result
    def constant(_):
        raise RelayError('nonfinite control value')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise RelayError('invalid control JSON') from None
    require(isinstance(value, dict), 'control record must be an object')
    return value


def encode(value):
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    except (ValueError, TypeError, UnicodeError):
        raise RelayError('invalid control value') from None
    require(0 < len(data) <= MAX_RECORD, 'control record exceeds limit')
    return struct.pack('!I', len(data)) + data


def close_fds(fds):
    for fd in fds:
        with contextlib.suppress(OSError):
            os.close(fd)


def peer_uid(connection):
    return struct.unpack('=iII', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]


def received_rights(ancillary):
    result = []
    invalid = False
    try:
        for level, kind, data in ancillary:
            if (level, kind) != (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                invalid = True
                continue
            length = len(data) - len(data) % array.array('i').itemsize
            invalid |= length != len(data)
            numbers = array.array('i')
            numbers.frombytes(data[:length])
            result.extend(numbers)
        require(not invalid, 'unexpected/malformed ancillary data')
        return result
    except BaseException:
        close_fds(result)
        raise


class Wire:
    """Exact reads avoid crossing the separately acknowledged FD-transfer phase."""
    def __init__(self, connection, tick=lambda: None, timeout=10.0):
        self.socket = connection
        self.tick = tick
        self.deadline = None if timeout is None else time.monotonic() + timeout
        connection.settimeout(0.2)

    def check(self):
        self.tick()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise RelayError('control deadline exceeded')

    def read(self, count):
        parts = bytearray()
        while len(parts) < count:
            self.check()
            try:
                data, ancillary, flags, _ = self.socket.recvmsg(
                    count - len(parts), socket.CMSG_SPACE(16 * array.array('i').itemsize),
                    getattr(socket, 'MSG_CMSG_CLOEXEC', 0))
            except socket.timeout:
                continue
            fds = received_rights(ancillary)
            close_fds(fds)
            require(not ancillary and not flags & socket.MSG_CTRUNC, 'unexpected descriptor transfer')
            require(data, 'control connection closed')
            parts.extend(data)
        return bytes(parts)

    def send_bytes(self, data):
        # Control is tiny and independent of stdio. A non-reader is a failure,
        # never an unbounded blocker on teardown. Never retry a partially sent record.
        end = time.monotonic() + 3
        while data:
            self.check()
            require(time.monotonic() < end, 'control write deadline exceeded')
            try:
                count = self.socket.send(data)
            except socket.timeout:
                continue
            require(count, 'control write failed')
            data = data[count:]

    def send(self, record):
        self.send_bytes(encode(record))

    def receive(self):
        count = struct.unpack('!I', self.read(4))[0]
        require(0 < count <= MAX_RECORD, 'control record exceeds limit')
        return parse_json(self.read(count))

    def receive_fds(self):
        while True:
            self.check()
            try:
                marker, ancillary, flags, _ = self.socket.recvmsg(
                    1, socket.CMSG_SPACE(16 * array.array('i').itemsize),
                    getattr(socket, 'MSG_CMSG_CLOEXEC', 0))
                break
            except socket.timeout:
                continue
        fds = received_rights(ancillary)
        try:
            require(marker == b'F' and not flags & socket.MSG_CTRUNC and len(fds) == 3,
                    'expected exactly three pipe descriptors')
            for fd, mode in zip(fds, (os.O_RDONLY, os.O_WRONLY, os.O_WRONLY)):
                os.set_inheritable(fd, False)
                require(stat.S_ISFIFO(os.fstat(fd).st_mode)
                        and re.fullmatch(r'pipe:\[\d+\]', os.readlink(f'/proc/self/fd/{fd}')),
                        'only anonymous pipes may be delegated')
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                require(flags & os.O_ACCMODE == mode and not flags & os.O_NONBLOCK,
                        'wrong pipe access mode')
            return fds
        except BaseException:
            close_fds(fds)
            raise


def validate_start(record):
    require(set(record) == {'op', 'argv', 'env'} and record['op'] == 'START', 'invalid START')
    argv, env = record['argv'], record['env']
    require(isinstance(argv, list) and len(argv) <= 4096 and
            all(isinstance(arg, str) and '\x00' not in arg for arg in argv), 'invalid argv')
    require(isinstance(env, dict) and all(k in FORWARD_ENV and isinstance(v, str) and '\x00' not in v
                                        for k, v in env.items()), 'unsupported forwarded environment')
    require(len(json.dumps(env).encode()) <= MAX_ENV, 'forwarded environment exceeds limit')
    return argv, env


def owned_path(path, directory=False):
    require(path.is_absolute() and '..' not in path.parts, 'expected an absolute normalized path')
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        expected = stat.S_ISDIR if part != path or directory else stat.S_ISREG
        require(expected(info.st_mode), 'unsafe path type or symlink')
        sticky = part != path and info.st_uid == 0 and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX
        require(info.st_uid in (0, os.getuid()) and (not info.st_mode & 0o022 or sticky),
                'unsafe path ownership/permissions')
    return info


def below(path, parent):
    return path == parent or parent in path.parents


def stop_process(process, grace):
    if process is None or process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace + 1)


def parent_death_guard(parent):
    require(parent > 0 and os.getppid() == parent, 'launch owner disappeared')
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) == 0, 'cannot set parent-death guard')
    require(os.getppid() == parent, 'launch owner disappeared during guard setup')


def child_guard(parent, wrapper, argv):
    parent_death_guard(parent)
    # Nothing user/agent-controlled is imported; command and environment were
    # fixed/filtered by the broker. Only the wrapper will execute native code.
    os.execve(PYTHON, [PYTHON, '-I', '-S', wrapper, '--', *argv], dict(os.environ))


class ChildFinished(Exception):
    pass


class Broker:
    def __init__(self, path, allowed, wrapper, *, limit=8, grace=3.0, profile='default'):
        self.path, self.allowed = Path(path), frozenset(allowed)
        self.wrapper = Path(wrapper).resolve(strict=True)
        owned_path(self.wrapper)
        owned_path(Path(__file__).resolve())
        require(self.allowed and all(type(uid) is int and uid > 0 for uid in self.allowed), 'non-root client UIDs required')
        require(1 <= limit <= 128 and 0 < grace <= 60, 'invalid broker limits')
        self.grace, self.profile = grace, profile
        self.instance = secrets.token_hex(16)
        self.capacity = threading.BoundedSemaphore(limit)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.connections, self.threads = set(), set()
        self.env = {**SAFE_ENV, 'HOME': pwd.getpwuid(os.getuid()).pw_dir}
        for key in ('ONE_CLICK_CLAUDE_HOME', 'ONE_CLICK_CLAUDE_BIN', 'CLAUDE_CONFIG_DIR',
                    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                    'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            if key in os.environ:
                self.env[key] = os.environ[key]
        raw_home = self.env.get('ONE_CLICK_CLAUDE_HOME', self.env['HOME'] + '/.local/share/one-click-claude/home')
        if raw_home == '~' or raw_home.startswith('~/'):
            raw_home = self.env['HOME'] + raw_home[1:]
        home = Path(raw_home).expanduser().resolve()
        for path in (self.path.absolute(), self.wrapper, Path(__file__).resolve()):
            require(not below(path, home), 'control/launcher path overlaps writable Claude home')
        require(not any(below(self.path.absolute(), Path(p)) for p in ('/usr', '/bin', '/lib', '/lib64', '/sbin', '/etc')),
                'control socket must not be in the public sandbox runtime')

    def handle(self, connection):
        process, fds = None, []
        wire = Wire(connection)
        try:
            require(peer_uid(connection) in self.allowed, 'peer UID is not authorized')
            wire.send_bytes(MAGIC)
            wire.send(dict(op='HELLO', version=1, uid=os.getuid(), instance=self.instance,
                           profile=self.profile, max_record=MAX_RECORD))
            request = wire.receive()
            if request == {'op': 'CHECK'}:
                wire.send({'op': 'CHECKED'})
                return
            argv, env = validate_start(request)
            wire.send({'op': 'READY_FOR_FDS'})
            fds = wire.receive_fds()
            command = [PYTHON, '-I', '-S', str(Path(__file__).resolve()), '--relay-child',
                       str(os.getpid()), str(self.wrapper), '--', *argv]
            process = subprocess.Popen(command, stdin=fds[0], stdout=fds[1], stderr=fds[2],
                                       cwd='/', env={**self.env, **env}, close_fds=True, start_new_session=True)
            close_fds(fds)
            fds = []
            wire.send({'op': 'STARTED', 'id': secrets.token_hex(16)})
            wire.deadline = None

            def tick():
                require(not self.stop.is_set(), 'broker is stopping')
                if process.poll() is not None:
                    raise ChildFinished()
            wire.tick = tick
            while True:
                record = wire.receive()
                require(set(record) == {'op', 'signal'} and record['op'] == 'SIGNAL'
                        and record['signal'] in SIGNALS, 'invalid running control record')
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, SIGNALS[record['signal']])
        except ChildFinished:
            wire.tick = lambda: None
            wire.deadline = time.monotonic() + 3
            rc = process.returncode
            with contextlib.suppress(OSError, RelayError):
                wire.send({'op': 'EXIT', 'code': rc if rc >= 0 else None, 'signal': -rc if rc < 0 else None})
        except (OSError, RelayError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
            # Never include argv, native stderr, environment or peer-supplied text.
            stop_process(process, self.grace)
            with contextlib.suppress(OSError, RelayError):
                wire.tick = lambda: None
                wire.deadline = time.monotonic() + 1
                wire.send({'op': 'ERROR', 'message': 'launch/control rejected or interrupted'})
        finally:
            close_fds(fds)
            stop_process(process, self.grace)
            connection.close()
            with self.lock:
                self.connections.discard(connection)
                self.threads.discard(threading.current_thread())
            self.capacity.release()

    def serve(self):
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(mode=0o711)
            parent.chmod(0o711)
        info = owned_path(parent, directory=True)
        require(info.st_uid == os.getuid(), 'socket directory must belong to broker UID')
        lock_fd = os.open(parent / '.broker.lock', os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        listener = socket.socket(socket.AF_UNIX)
        identity = None
        handlers = {}
        try:
            owned_path(parent / '.broker.lock')
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists() or self.path.is_symlink():
                info = self.path.lstat()
                require(stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid(), 'unsafe existing socket')
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(1)
                    try:
                        probe.connect(str(self.path))
                    except OSError as error:
                        require(error.errno == errno.ECONNREFUSED, 'existing socket is not verified stale')
                    else:
                        raise RelayError('socket already has a listener')
                require(self.path.lstat().st_ino == info.st_ino, 'socket changed during startup')
                self.path.unlink()
            listener.bind(str(self.path))
            identity = self.path.lstat().st_ino
            self.path.chmod(0o666)  # peer-UID authorization is mandatory, not optional
            listener.listen(32)
            listener.settimeout(0.2)
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    handlers[sig] = signal.signal(sig, lambda *_: self.stop.set())
            while not self.stop.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                if peer_uid(connection) not in self.allowed or not self.capacity.acquire(blocking=False):
                    connection.close()
                    continue
                thread = threading.Thread(target=self.handle, args=(connection,), daemon=True)
                with self.lock:
                    self.connections.add(connection)
                    self.threads.add(thread)
                thread.start()
        finally:
            self.stop.set()
            listener.close()
            with self.lock:
                connections, threads = list(self.connections), list(self.threads)
            for connection in connections:
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
            for thread in threads:
                thread.join(self.grace + 5)
            if identity is not None:
                with contextlib.suppress(FileNotFoundError):
                    if self.path.lstat().st_ino == identity:
                        self.path.unlink()
            os.close(lock_fd)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def connect():
    path = os.environ.get('CLAUDE_RELAY_SOCKET', '')
    uid = os.environ.get('CLAUDE_RELAY_SERVER_UID', '')
    require(path.startswith('/') and uid.isdecimal() and int(uid) > 0, 'set relay socket and expected non-root server UID')
    info = Path(path).lstat()
    require(stat.S_ISSOCK(info.st_mode) and info.st_uid == int(uid), 'unexpected socket owner/type')
    connection = socket.socket(socket.AF_UNIX)
    try:
        connection.settimeout(10)
        connection.connect(path)
        require(peer_uid(connection) == int(uid), 'unexpected server UID')
        wire = Wire(connection)
        require(wire.read(len(MAGIC)) == MAGIC, 'incompatible relay protocol')
        hello = wire.receive()
        require(hello.get('op') == 'HELLO' and hello.get('version') == 1
                and hello.get('uid') == int(uid), 'incompatible relay greeting')
        return wire, hello
    except BaseException:
        connection.close()
        raise


def client(argv, check=False):
    # The driver supplies its PID to cover death even before this interpreter
    # finishes importing. Manual invocations use their still-live shell parent.
    if 'CLAUDE_RELAY_PARENT_PID' not in os.environ:
        require(os.getppid() > 1, 'invoking owner disappeared before startup')
    parent_death_guard(int(os.environ.get('CLAUDE_RELAY_PARENT_PID', str(os.getppid()))))
    require('CLAUDE_CONFIG_DIR' not in os.environ, 'remote CLAUDE_CONFIG_DIR is unsupported; use B-side configuration')
    wire, hello = connect()
    local, remote, handlers = [], [], {}
    pumps_started = False
    events = queue.Queue(maxsize=16)
    overflow = threading.Event()
    def notify(event):
        try:
            events.put_nowait(event)
        except queue.Full:
            overflow.set()
    try:
        if check:
            wire.send({'op': 'CHECK'})
            require(wire.receive() == {'op': 'CHECKED'}, 'invalid check response')
            print(json.dumps(hello), flush=True)
            return 0
        env = {k: os.environ[k] for k in FORWARD_ENV if k in os.environ}
        request = {'op': 'START', 'argv': argv, 'env': env}
        validate_start(request)
        wire.send(request)
        require(wire.receive().get('op') == 'READY_FOR_FDS', 'broker rejected launch')
        stdin_r, stdin_w = os.pipe2(os.O_CLOEXEC)
        stdout_r, stdout_w = os.pipe2(os.O_CLOEXEC)
        stderr_r, stderr_w = os.pipe2(os.O_CLOEXEC)
        remote, local = [stdin_r, stdout_w, stderr_w], [stdin_w, stdout_r, stderr_r]
        require(wire.socket.sendmsg([b'F'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', remote))]) == 1,
                'descriptor transfer failed')
        close_fds(remote)
        remote = []
        require(wire.receive().get('op') == 'STARTED', 'remote launch failed')
        wire.deadline = None
        done = [threading.Event(), threading.Event()]

        def pump(source, target, direction, finished=None):
            try:
                while True:
                    data = os.read(source, 65536)
                    if not data:
                        break
                    while data:
                        data = data[os.write(target, data):]
            except OSError as error:
                if direction != 'in' or error.errno != errno.EPIPE:
                    notify(('error', None))
                elif direction == 'in':
                    with contextlib.suppress(OSError):
                        os.close(0)
            finally:
                if direction == 'in':
                    with contextlib.suppress(OSError):
                        os.close(target)
                if finished is not None:
                    finished.set()

        for args in ((0, stdin_w, 'in'), (stdout_r, 1, 'out', done[0]), (stderr_r, 2, 'err', done[1])):
            threading.Thread(target=pump, args=args, daemon=True).start()
        pumps_started = True
        for name, sig in SIGNALS.items():
            if sig != signal.SIGKILL:
                handlers[sig] = signal.signal(sig, lambda sig, _frame: notify(('signal', sig)))

        def tick():
            require(not overflow.is_set(), 'local control queue overflow')
            while True:
                try:
                    kind, sig = events.get_nowait()
                except queue.Empty:
                    break
                require(kind != 'error', 'local stdio failed')
                # Avoid recursion through send_bytes -> check -> tick.
                wire.tick = lambda: None
                try:
                    wire.send({'op': 'SIGNAL', 'signal': next(k for k, v in SIGNALS.items() if v == sig)})
                finally:
                    wire.tick = tick
        wire.tick = tick
        result = wire.receive()
        require(result.get('op') == 'EXIT', 'remote invocation failed without terminal status')
        code, sig = result.get('code'), result.get('signal')
        require((type(code) is int and 0 <= code <= 255 and sig is None) or
                (code is None and type(sig) is int and 0 < sig < signal.NSIG), 'invalid terminal status')
        end = time.monotonic() + 5
        while not all(event.is_set() for event in done):
            require(time.monotonic() < end, 'output drain deadline exceeded')
            tick()
            time.sleep(0.01)
        tick()
        return code if code is not None else -sig
    finally:
        wire.socket.close()  # disconnect independently cancels a still-live invocation
        close_fds(remote)
        # stdin writer belongs to its pump; avoid close/reuse races with that thread.
        close_fds(local[1:] if pumps_started else local)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    require(sys.flags.isolated and sys.platform == 'linux', 'use isolated system Python on Linux')
    require(os.getuid() != 0 and os.getuid() == os.geteuid() and os.getgid() == os.getegid(),
            'run already as a non-root account; no UID switching')
    os.umask(0o077)
    args = sys.argv[1:]
    if args[:1] == ['--relay-child']:
        require(len(args) >= 4 and args[3] == '--', 'invalid child guard invocation')
        child_guard(int(args[1]), args[2], args[4:])
    elif args[:1] == ['--relay-serve']:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--relay-serve', action='store_true')
        parser.add_argument('--relay-socket', required=True)
        parser.add_argument('--relay-allow-uid', type=int, action='append', required=True)
        parser.add_argument('--relay-wrapper', default=str(Path(__file__).with_name('one_click_claude.py')))
        parser.add_argument('--relay-max-active', type=int, default=8)
        parser.add_argument('--relay-stop-grace', type=float, default=3)
        parser.add_argument('--relay-profile', default='default')
        options = parser.parse_args(args)
        Broker(options.relay_socket, options.relay_allow_uid, options.relay_wrapper,
               limit=options.relay_max_active, grace=options.relay_stop_grace,
               profile=options.relay_profile).serve()
        return 0
    elif args == ['--relay-help']:
        print(__doc__, flush=True)
        return 0
    elif args == ['--relay-check']:
        return client([], check=True)
    else:
        if args[:1] == ['--']:
            args = args[1:]
        else:
            require(not args or not args[0].startswith('--relay-'), 'unknown relay option')
        return client(args)


if __name__ == '__main__':
    try:
        status = main()
    except RelayError as error:
        log(str(error) + '; no fallback')
        status = 125
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        log('operation failed (check peer identity, broker status, profile and protocol); no fallback')
        status = 125
    except KeyboardInterrupt:
        status = 130
    # Stdin pump may still be waiting for caller input after the native child
    # exited. We own this disposable executable: never wait for that daemon.
    sys.stdout.flush()
    sys.stderr.flush()
    if status < 0:
        sig = -status
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        status = 128 + sig
    os._exit(status)
