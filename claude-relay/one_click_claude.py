#!/usr/bin/python3 -I
"""Unprivileged, single-file, stdlib-only Linux wrapper for an EXISTING native CLI.

Run already as the provisioned non-root account; this file NEVER calls sudo,
changes accounts, installs packages, or downloads/runs a Claude installer.

    ./claude-relay/one_click_claude.py --sandbox-doctor       # no Claude required
    ./claude-relay/one_click_claude.py --sandbox-setup-only   # inspect/probe, no Claude exec
    ./claude-relay/one_click_claude.py [Claude arguments...]  # Claude exec ONLY in bwrap

Native source: $HOME/.local/bin/claude, or ONE_CLICK_CLAUDE_BIN/--sandbox-claude.
Private writable home: $HOME/.local/share/one-click-claude/home, or
ONE_CLICK_CLAUDE_HOME/--sandbox-home. This is NOT the entire account home. Keep
this launcher, the native source, harness/SDK code, venvs, and unrelated secrets
OUTSIDE that writable tree. Root ownership of caller-owned code is not required;
its absence from the sandbox, isolated Python, and trusted host-side use matter.
Do not prearrange hardlink/socket aliases from that tree to other host resources.

SDK cli_path can point directly at this executable; no installation or special
account naming convention is needed. Set the two ONE_CLICK_* variables in BOTH
version probes and main launches. Stage prompt/settings files beneath the private
home at their unchanged absolute paths; host cwd and /tmp are never auto-mounted.
The harness owns tool/backgrounding settings: the supplied values of
CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS and ENABLE_TOOL_SEARCH are passed, not invented.
A different host UID cannot become this account just by using bwrap --uid.

System prerequisites: Python 3.9+, non-setuid bubblewrap, CA roots, and the native
binary's system libraries; Bash/coreutils/Git are the checked coding baseline.
Use an already-provisioned account without privileged groups/capabilities, and
an OS policy permitting unprivileged namespaces. Root launches are refused.
Never retries without confinement. Doctor/setup-only never execute the native
source; its ELF data is inspected and a private byte-identical snapshot mounted.

Only the private home is persistently writable. Public system runtime is read-
only; temporary storage/proc/dev are private. Networking is the HOST network,
including localhost and abstract UNIX sockets. External MCP/harness tools retain
their own authority. This is NOT kernel, resource, network, or harness isolation.
See doc/claude-sandbox-one-click-plan.md for the launch/staging contract and limits.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import pty
import pwd
import re
import secrets
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import tty


BWRAP = Path('/usr/bin/bwrap')
PYTHON = '/usr/bin/python3'
SAFE_ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C.UTF-8'}
MAX_BINARY = 1024 * 1024 * 1024
RUNTIME_DIRS = ('/usr/bin', '/usr/sbin', '/usr/lib', '/usr/lib64', '/usr/share',
                '/bin', '/sbin', '/lib', '/lib64')
# TODO(output-budgets): expose read-only --sandbox-capabilities and strictly
# validate a verified native output-cap setting before adding it to this list.
# Coordinate broker AND wrapper deployment; see SAMPLING.md. No cap is added yet.
ENV_ALLOW = (
    'TERM', 'COLORTERM', 'NO_COLOR', 'FORCE_COLOR',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
    'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
    'CLAUDE_CONFIG_DIR', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_AGENT_SDK_VERSION',
    'CLAUDE_CODE_SDK_READS_SESSION_STATE', 'CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS',
    'CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING', 'CLAUDE_AGENT_SDK_CLIENT_APP',
    'TRACEPARENT', 'TRACESTATE',
    'CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS', 'ENABLE_TOOL_SEARCH',
    # Explicit harness control and generation-scoped MCP capability, not API auth.
    'DISABLE_AUTO_COMPACT', 'PYTHIA_CLAUDE_MCP_TOKEN',
)
DEPENDENCIES = {
    'python3': ('/usr/bin/python3',),
    'bubblewrap': (str(BWRAP),),
    'ca-certificates': ('/etc/ssl/certs/ca-certificates.crt',),
    'bash': ('/usr/bin/bash',),
    'coreutils': ('/usr/bin/env',),
    'git': ('/usr/bin/git',),
}
HELP = """Usage: one_click_claude.py [wrapper options] [Claude arguments...]

  --sandbox-doctor          Check isolation + localhost. NO Claude or source needed.
  --sandbox-setup-only      Prepare private home, inspect source, probe; NO Claude exec.
  --sandbox-home PATH       Private home; also settable with ONE_CLICK_CLAUDE_HOME.
  --sandbox-print-home      Print its canonical absolute path; no creation/execution.
  --sandbox-claude PATH     Existing native source; also ONE_CLICK_CLAUDE_BIN.
  --sandbox-help            This help. Ordinary --help/-v belong to Claude.
  --                       End wrapper parsing; remaining argv passes verbatim.

Defaults: $HOME/.local/bin/claude and $HOME/.local/share/one-click-claude/home.
Run with /usr/bin/python3 -I, or use this file's executable shebang as SDK cli_path.
Already be the intended non-root account. No sudo/root bootstrap, account changes,
package installation, native download, or wrapper prompts, even on first launch.
Missing prerequisites fail clearly. Authentication belongs to Claude, in its
private home, not the wrapper; authenticate interactively before harness use.

Private home/work is the default cwd; only an existing cwd within that tree is
preserved. Stage prompt/config/MCP files there using the SAME absolute paths.
Keep the launcher, native source and trusted harness/SDK code outside that tree.
CLAUDE_CONFIG_DIR must be under it. No host API keys or login files are imported.
Use ONE_CLICK_* env settings consistently for SDK version probes and main runs.
MCP tuning env is passed only when supplied; the harness owns its values.
Doctor does not contact the internet. Wrapper options precede ALL Claude arguments.
"""


class SandboxError(Exception):
    pass


def log(message: str) -> None:
    print(f'one-click-claude: {message}', file=sys.stderr)


def require(condition, message: str) -> None:
    if not condition:
        raise SandboxError(message)


def beneath(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def normalized(value: str) -> Path:
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and '\x00' not in value,
            f'expected an absolute path without ..: {value!r}')
    return path


def trusted(path: Path, *, directory=False, executable=False, allow_setid=False):
    """Check literal components, not a symlink-following ownership illusion."""
    normalized(str(path))
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        expected = stat.S_ISDIR if item != path or directory else stat.S_ISREG
        require(expected(info.st_mode), f'not a real trusted file/directory: {item}')
        require(info.st_uid == 0 and not info.st_mode & 0o022,
                f'must be root-owned and not group/other-writable: {item}')
    if executable:
        require(info.st_mode & 0o111, f'not executable: {path}')
    require(allow_setid or not info.st_mode & 0o6000, f'setuid/setgid is unsupported: {path}')
    return info


def safe_user_path(path: Path, *, directory=False, executable=False, allow_sticky=False):
    """Trusted host-side input: root/current-UID owned, not writable by others.

    Root-owned sticky ancestors (e.g. /tmp) can contain an owned private tree.
    No symlinks are traversed here; callers canonicalize deliberate source aliases.
    """
    normalized(str(path))
    uid = os.getuid()
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        expected = stat.S_ISDIR if item != path or directory else stat.S_ISREG
        require(expected(info.st_mode), f'not a real file/directory: {item}')
        sticky_parent = ((item != path or allow_sticky) and stat.S_ISDIR(info.st_mode)
                         and info.st_uid == 0 and info.st_mode & stat.S_ISVTX)
        require(info.st_uid in (0, uid) and (not info.st_mode & 0o022 or sticky_parent),
                f'unsafe ownership or group/other-writable path: {item}')
    if not directory:
        require(info.st_nlink == 1, f'trusted input must not have hardlink aliases: {path}')
        require(not info.st_mode & 0o6000, f'setuid/setgid input is unsupported: {path}')
    if executable:
        require(info.st_mode & 0o111, f'not executable: {path}')
    return info


def current_identity():
    uid, gid = os.getuid(), os.getgid()
    require(uid != 0 and os.geteuid() == uid and os.getegid() == gid,
            'run already as the intended non-root account; no identity switching is supported')
    require(gid != 0 and 0 not in os.getgroups(), 'root group membership is forbidden')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    require(all(int(status[k].strip(), 16) == 0 for k in ('CapEff', 'CapPrm', 'CapAmb')),
            'run without host capabilities')
    return pwd.getpwuid(uid)


def missing_packages() -> list[str]:
    return [name for name, paths in DEPENDENCIES.items()
            if any(not Path(p).is_file() or
                   (not p.endswith('.crt') and not os.access(p, os.X_OK)) for p in paths)]


def check_platform() -> None:
    require(sys.platform == 'linux' and sys.version_info >= (3, 9),
            'requires Linux and system Python 3.9 or newer')
    require(platform.machine() in ('x86_64', 'aarch64'), 'supported ISAs: x86_64, aarch64')
    require(platform.libc_ver()[0] == 'glibc', 'v1 requires a glibc-based Linux system')


def check_bwrap() -> None:
    require(BWRAP.exists(), 'missing /usr/bin/bwrap; install the bubblewrap package')
    trusted(BWRAP, executable=True)
    result = subprocess.run([str(BWRAP), '--help'], capture_output=True, text=True,
                            env=SAFE_ENV, timeout=10, check=True)
    for flag in ('--disable-userns', '--assert-userns-disabled', '--perms', '--ro-bind-data'):
        require(flag in result.stdout, f'Bubblewrap lacks required {flag}; upgrade bubblewrap')


def inspect_elf(stream, *, executable=True) -> tuple[str | None, list[str]]:
    """Read ELF64 headers/dynamic strings. Never run the file or ldd."""
    size = os.fstat(stream.fileno()).st_size

    def read_at(offset, length):
        require(0 <= offset <= size and 0 <= length <= size - offset,
                'truncated or invalid ELF file')
        stream.seek(offset)
        data = stream.read(length)
        require(len(data) == length, 'ELF changed while reading')
        return data

    header = read_at(0, 64)
    require(header[:7] == b'\x7fELF\x02\x01\x01',
            'source must be a native little-endian ELF64 executable, not an npm/shell wrapper')
    kind, machine, version, entry, phoff, _, _, ehsize, phsize, phnum = struct.unpack_from(
        '<HHIQQQIHHH', header, 16)
    require(kind in (2, 3) and version == 1 and ehsize == 64, 'unsupported ELF header')
    require(machine == {'x86_64': 62, 'aarch64': 183}[platform.machine()], 'wrong ELF architecture')
    require(not executable or entry != 0, 'ELF has no executable entry point')
    require(phsize == 56 and 0 < phnum < 1024, 'unsupported ELF program headers')
    segments = [struct.unpack('<IIQQQQQQ', read_at(phoff + i * phsize, phsize))
                for i in range(phnum)]
    interpreter = None
    needed, string_table = [], None
    for kind, _, offset, _, _, length, _, _ in segments:
        if kind == 3:  # PT_INTERP
            require(interpreter is None and 1 < length < 4096, 'invalid ELF interpreter')
            raw = read_at(offset, length)
            require(raw.endswith(b'\0') and b'\0' not in raw[:-1], 'invalid ELF interpreter')
            interpreter = raw[:-1].decode('ascii')
        elif kind == 2:  # PT_DYNAMIC
            require(length <= 1024 * 1024 and length % 16 == 0, 'invalid ELF dynamic table')
            for tag, value in struct.iter_unpack('<qQ', read_at(offset, length)):
                if tag == 0:
                    break
                if tag == 1:
                    needed.append(value)
                elif tag == 5:
                    string_table = value
    libraries = []
    if needed:
        require(string_table is not None, 'ELF has dependencies without a string table')
        for index in needed:
            address = string_table + index
            mapping = next((s for s in segments if s[0] == 1 and s[3] <= address < s[3] + s[5]), None)
            require(mapping is not None, 'invalid ELF dependency string')
            raw = read_at(mapping[2] + address - mapping[3], min(4096, mapping[3] + mapping[5] - address))
            require(b'\0' in raw, 'unterminated ELF dependency string')
            name = raw.split(b'\0', 1)[0].decode('ascii')
            require(name and '/' not in name, f'unsupported private ELF dependency: {name!r}')
            libraries.append(name)
    if interpreter:
        expected = {'x86_64': '/lib64/ld-linux-x86-64.so.2',
                    'aarch64': '/lib/ld-linux-aarch64.so.1'}[platform.machine()]
        require(interpreter == expected, f'unsupported ELF interpreter: {interpreter}')
        trusted(Path(interpreter).resolve(strict=True), executable=True)
    stream.seek(0)
    return interpreter, libraries


def check_libraries(libraries: list[str]) -> None:
    """Check the system dependency closure. Nonstandard/private runtimes are refused."""
    triplet = {'x86_64': 'x86_64-linux-gnu', 'aarch64': 'aarch64-linux-gnu'}[platform.machine()]
    roots = [Path(p) for p in (f'/lib/{triplet}', f'/usr/lib/{triplet}',
                               '/lib64', '/usr/lib64', '/lib', '/usr/lib')]
    pending, visited = list(libraries), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        require(len(visited) < 512, 'excessive native library closure')
        path = next((root / name for root in roots if (root / name).is_file()), None)
        require(path is not None,
                f'missing system library {name}; install its distro package (often libstdc++6/libgcc-s1)')
        path = path.resolve(strict=True)
        require(any(beneath(path, Path(p)) for p in RUNTIME_DIRS),
                f'library is outside the public runtime: {path}')
        trusted(path)
        with path.open('rb') as stream:
            _, dependencies = inspect_elf(stream, executable=False)
        pending.extend(dependencies)


@contextlib.contextmanager
def native_source(path: Path):
    # Inspect only: no native execution or external ldd, under the current UID.
    path = path.expanduser().resolve(strict=True)
    require(path != Path(__file__).resolve(), 'source points to this wrapper')
    safe_user_path(path, executable=True)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_BINARY,
                'native source must be a regular file no larger than 1 GiB')
        require(info.st_mode & 0o111 and not info.st_mode & 0o6000,
                'native source must be executable without setuid/setgid')
        _, libraries = inspect_elf(stream)
        check_libraries(libraries)
        yield stream, path, info


@dataclass
class Options:
    claude: str | None = None
    home: str | None = None
    doctor: bool = False
    setup_only: bool = False
    print_home: bool = False
    help: bool = False


def parse_args(argv: list[str]) -> tuple[Options, list[str]]:
    options, payload = Options(), []
    flags = {'--sandbox-doctor': 'doctor', '--sandbox-setup-only': 'setup_only',
             '--sandbox-print-home': 'print_home', '--sandbox-help': 'help'}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--':
            payload.extend(argv[i + 1:])
            break
        if arg in flags:
            setattr(options, flags[arg], True)
        elif arg in ('--sandbox-claude', '--sandbox-home'):
            i += 1
            require(i < len(argv), f'{arg} requires a value')
            setattr(options, 'claude' if arg == '--sandbox-claude' else 'home', argv[i])
        else:
            require(not arg.startswith(('--sandbox-', '--_sandbox-')), f'unknown wrapper option: {arg}')
            # Stop at the first CLI argument. A later prompt/setting value named
            # "--sandbox-home" must remain CLI data, never change launch policy.
            payload.extend(argv[i:])
            break
        i += 1
    require(sum((options.doctor, options.setup_only, options.print_home)) <= 1,
            'choose only one of doctor, setup-only, or print-home')
    return options, payload


def account_home() -> Path:
    return normalized(os.environ.get('HOME', pwd.getpwuid(os.getuid()).pw_dir))


def default_source() -> Path:
    value = os.environ.get('ONE_CLICK_CLAUDE_BIN')
    return Path(value).expanduser() if value else account_home() / '.local/bin/claude'


def private_home(options: Options) -> Path:
    value = options.home or os.environ.get('ONE_CLICK_CLAUDE_HOME')
    home = Path(value).expanduser() if value else account_home() / '.local/share/one-click-claude/home'
    return normalized(str(home)).resolve(strict=False)


def check_home_destination(home: Path) -> None:
    # A narrowly selected home BELOW /tmp can support temporary profiles/staging;
    # /tmp itself stays private. Bind the home after creating the private tmpfs.
    reserved = (*RUNTIME_DIRS, '/usr', '/proc', '/dev', '/run', '/etc', '/opt')
    require(home != Path('/tmp') and not any(beneath(home, Path(p)) or beneath(Path(p), home)
                                            for p in reserved),
            'home overlaps a reserved runtime destination')


def check_private_layout(home: Path, source: Path | None = None) -> None:
    check_home_destination(home)
    for label, path in (('launcher', Path(__file__).absolute()), ('Python', Path(sys.executable)),
                        *([('native source', source.absolute())] if source is not None else [])):
        require(not beneath(path, home) and not beneath(path.resolve(strict=False), home),
                f'{label} must be outside the writable Claude home: {path}')
    safe_user_path(Path(__file__).resolve(strict=True))


def validate_home(home: Path, uid: int) -> None:
    safe_user_path(home, directory=True)
    info = home.lstat()
    require(info.st_uid == uid and info.st_mode & 0o777 == 0o700,
            f'private home must be owned by UID {uid}, mode 0700: {home}')
    reject_submounts(home, include_self=True)


def prepare_home(home: Path) -> None:
    # Never repair ownership/permissions silently or chown existing content.
    for item in (*reversed(home.parents), home):
        try:
            item.mkdir(mode=0o700)
        except FileExistsError:
            pass
        # Only the final directory has to be private; /tmp may be a sticky parent.
        safe_user_path(item, directory=True, allow_sticky=item != home)
    validate_home(home, os.getuid())
    fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            os.mkdir('work', mode=0o700, dir_fd=fd)
        except FileExistsError:
            pass
        info = os.stat('work', dir_fd=fd, follow_symlinks=False)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o022,
                'private home/work must be a real owned directory, not an alias')
    finally:
        os.close(fd)


@contextlib.contextmanager
def native_snapshot(source: Path):
    """Anonymous per-launch copy, never persisted or executed on the host."""
    with tempfile.TemporaryFile(dir='/tmp') as image:
        with native_source(source) as (incoming, resolved, before):
            hasher, size = hashlib.sha256(), 0
            for data in iter(lambda: incoming.read(1024 * 1024), b''):
                size += len(data)
                require(size <= MAX_BINARY, 'native source grew beyond the size limit')
                image.write(data)
                hasher.update(data)
            after = os.fstat(incoming.fileno())
            require(size == before.st_size and
                    (before.st_mtime_ns, before.st_ctime_ns) == (after.st_mtime_ns, after.st_ctime_ns),
                    'native source changed while snapshotting; retry after its update finishes')
        image.flush()
        os.fchmod(image.fileno(), 0o555)
        _, libraries = inspect_elf(image)
        check_libraries(libraries)
        yield image.fileno(), resolved, hasher.hexdigest()


def mount_points() -> list[Path]:
    text = Path('/proc/self/mountinfo').read_text()
    return [Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), line.split()[4]))
            for line in text.splitlines()]


def reject_submounts(path: Path, *, include_self=False) -> None:
    for mount in mount_points():
        require(not (beneath(mount, path) and (include_self or mount != path)),
                f'unexpected host mount under sandbox source: {mount}')


def check_policy(path: Path) -> None:
    trusted(path, directory=True)
    reject_submounts(path)
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs:
            trusted(Path(root) / name, directory=True)
        for name in files:
            trusted(Path(root) / name)


def sandbox_env(home: Path, name: str, uid: int, supplied: dict[str, str]) -> dict[str, str]:
    require(all(k in ENV_ALLOW and isinstance(v, str) and '\x00' not in v for k, v in supplied.items()),
            'unexpected forwarded environment')
    env = dict(supplied)
    if 'CLAUDE_CONFIG_DIR' in env:
        config = normalized(env['CLAUDE_CONFIG_DIR'])
        require(beneath(config.resolve(), home), 'CLAUDE_CONFIG_DIR must be inside the dedicated home')
    env.update(HOME=str(home), USER=name, LOGNAME=name, LANG='C.UTF-8',
               PATH=f'/opt/claude:{home}/.local/bin:/usr/bin:/bin', TMPDIR='/tmp',
               XDG_CONFIG_HOME=f'{home}/.config', XDG_CACHE_HOME=f'{home}/.cache',
               XDG_DATA_HOME=f'{home}/.local/share', XDG_STATE_HOME=f'{home}/.local/state',
               XDG_RUNTIME_DIR=f'/run/user/{uid}',
               SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt',
               NODE_EXTRA_CA_CERTS='/etc/ssl/certs/ca-certificates.crt',
               DISABLE_AUTOUPDATER='1', DISABLE_UPDATES='1')
    return env


# This fixed shim only ever runs AFTER Bubblewrap succeeds. It is not an exposed
# host-side entry point. Python isolation prevents importing code from the home.
ENTRY = r'''
import fcntl, os, signal, sys, termios
interactive, cwd, default_cwd, *command = sys.argv[1:]
if interactive == '1':
    if os.getsid(0) != os.getpid():
        os.setsid()
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
if cwd == default_cwd:
    os.makedirs(cwd, mode=0o700, exist_ok=True)
os.chdir(cwd)
for sig in (signal.SIGPIPE, signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, signal.SIG_DFL)
os.execv(command[0], command)
'''


@contextlib.contextmanager
def sandbox_command(home_source: Path, home: Path, name: str, uid: int, gid: int,
                    payload: list[str], *, native: int | None = None,
                    supplied=None, cwd: Path | None = None, interactive=False):
    """Yield argv + narrowly scoped bwrap setup FDs. No payload execution here."""
    require(re.fullmatch(r'[^:\x00\n\r]+', name), 'invalid account name')
    for path in (home_source, home):
        normalized(str(path))
    check_home_destination(home)
    reject_submounts(home_source)
    command = [str(BWRAP), '--unshare-all', '--unshare-user', '--share-net',
               '--disable-userns', '--assert-userns-disabled', '--uid', str(uid), '--gid', str(gid),
               '--cap-drop', 'ALL', '--new-session', '--die-with-parent', '--clearenv']
    for value in RUNTIME_DIRS:
        path = Path(value)
        if not path.exists():
            continue
        if path.is_symlink():
            target = path.resolve(strict=True)
            require(str(target) in RUNTIME_DIRS, f'unsupported runtime symlink: {path} -> {target}')
            command += ['--symlink', str(target), str(path)]
        else:
            trusted(path, directory=True)
            reject_submounts(path)
            command += ['--ro-bind', str(path), str(path)]
    resolver_link = Path('/etc/resolv.conf')
    require(resolver_link.lstat().st_uid == 0, '/etc/resolv.conf must be administrator controlled')
    resolver = resolver_link.resolve(strict=True)
    resolver_info = resolver.stat()
    # systemd-resolved legitimately owns its /run directory/file as a service UID.
    require(stat.S_ISREG(resolver_info.st_mode) and resolver_info.st_uid != uid
            and not resolver_info.st_mode & 0o022, 'unsafe host resolver configuration')
    certificates = Path('/etc/ssl/certs/ca-certificates.crt')
    trusted(certificates)
    for source, destination in ((resolver, '/etc/resolv.conf'),
                                (certificates, '/etc/ssl/certs/ca-certificates.crt')):
        command += ['--ro-bind', str(source), destination]
    policy = Path('/etc/claude-code')
    if policy.exists() or policy.is_symlink():
        check_policy(policy)
        command += ['--ro-bind', str(policy), str(policy)]
    if native is not None:
        require(stat.S_ISREG(os.fstat(native).st_mode), 'native snapshot must be a regular file')
        # --ro-bind-fd in older distro bwrap cannot resolve an unlinked O_TMPFILE.
        # --ro-bind-data consumes anonymous file data without any host pathname.
        os.lseek(native, 0, os.SEEK_SET)
        command += ['--perms', '0555', '--ro-bind-data', str(native), '/opt/claude/claude']
    hosts = b'127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n'
    if Path('/etc/hosts').exists():
        trusted(Path('/etc/hosts'))
        hosts = Path('/etc/hosts').read_bytes()
        require(len(hosts) <= 1024 * 1024, '/etc/hosts is unexpectedly large')
    files = {
        '/etc/passwd': f'{name}:x:{uid}:{gid}:Claude sandbox:{home}:/bin/bash\n'.encode(),
        '/etc/group': f'{name}:x:{gid}:\n'.encode(),
        '/etc/nsswitch.conf': b'passwd: files\ngroup: files\nhosts: files dns\n',
        '/etc/hosts': hosts,
    }
    with contextlib.ExitStack() as stack:
        descriptors = [] if native is None else [native]
        for destination, data in files.items():
            stream = stack.enter_context(tempfile.TemporaryFile(dir='/tmp'))
            stream.write(data)
            stream.seek(0)
            descriptors.append(stream.fileno())
            command += ['--ro-bind-data', str(stream.fileno()), destination]
        command += ['--proc', '/proc', '--dev', '/dev',
                    '--perms', '1777', '--tmpfs', '/tmp', '--perms', '0700', '--tmpfs', '/run',
                    '--perms', '0700', '--dir', f'/run/user/{uid}',
                    '--bind', str(home_source), str(home), '--remount-ro', '/',
                    '--chdir', str(home)]
        for key, value in sandbox_env(home, name, uid, supplied or {}).items():
            command += ['--setenv', key, value]
        default_cwd = home / 'work'
        cwd = cwd or default_cwd
        require(beneath(cwd, home), 'cwd must be inside the dedicated home')
        command += ['--', PYTHON, '-I', '-c', ENTRY, str(int(interactive)), str(cwd),
                    str(default_cwd), *payload]
        yield command, tuple(descriptors)


PROBE = r'''
import ctypes, errno, json, os, pathlib, socket, sys
settings = json.loads(sys.argv[1])
assert os.getuid() == settings['uid'] and os.getgid() == settings['gid'], 'wrong UID/GID'
assert os.readlink('/proc/self/ns/net') == settings['net'], 'network namespace changed'
assert os.readlink('/proc/self/ns/pid') != settings['pid_ns'], 'PID namespace unchanged'
assert not os.path.lexists(settings['outside']), 'host sentinel exposed'
for pid in os.listdir('/proc'):
    if pid.isdigit():
        assert not os.path.exists('/proc/' + pid + '/root' + settings['outside']), 'host process root exposed'
assert not os.path.exists('/usr/local/bin'), 'host /usr/local exposed'
status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines())
assert int(status['CapEff'].strip(), 16) == 0, 'capabilities retained'
assert status['NoNewPrivs'].strip() == '1', 'no_new_privs missing'
for name in os.listdir('/proc/self/fd'):
    if int(name) >= 3:
        try:
            link = os.readlink('/proc/self/fd/' + name)
        except FileNotFoundError:
            continue
        raise AssertionError('inherited FD: ' + link)
for target in ('/outside-probe', '/usr/bin/.one-click-probe'):
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        assert error.errno in (errno.EROFS, errno.EACCES), str(error)
    else:
        os.close(fd)
        os.unlink(target)
        raise AssertionError('writable runtime: ' + target)
path = pathlib.Path.home() / ('.probe-' + settings['token'])
with path.open('x') as stream:
    stream.write(settings['token'])
assert path.read_text() == settings['token']
path.unlink()
for base in ('/tmp', '/run'):
    path = pathlib.Path(base) / settings['token']
    path.write_text('private')
    path.unlink()
with socket.create_connection(('127.0.0.1', settings['tcp']), timeout=5) as conn:
    assert conn.recv(128).decode() == settings['token'], 'localhost TCP failed'
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as conn:
    conn.settimeout(5)
    conn.sendto(settings['token'].encode(), ('127.0.0.1', settings['udp']))
    assert conn.recv(128).decode() == settings['token'], 'localhost UDP failed'
libc = ctypes.CDLL(None, use_errno=True)
assert libc.unshare(0x10000000) == -1, 'nested user namespaces still enabled'
assert ctypes.get_errno() in (errno.EPERM, errno.ENOSPC, errno.EUSERS), 'unexpected unshare failure'
print('filesystem, process/FD isolation, capabilities, userns, localhost TCP/UDP: OK')
'''


def probe(home_source: Path, home: Path, name: str, uid: int, gid: int, *, native=None) -> None:
    """Harmless payload, no authentication, external network, or Claude execution."""
    with tempfile.TemporaryDirectory(prefix='claude-doctor-outside-', dir='/tmp') as outside, \
            socket.socket() as tcp, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        sentinel = Path(outside) / 'hidden'
        sentinel.write_text('must not be mounted')
        tcp.bind(('127.0.0.1', 0))
        tcp.listen(1)
        tcp.settimeout(15)
        udp.bind(('127.0.0.1', 0))
        udp.settimeout(15)
        token = secrets.token_hex(16)

        def tcp_serve():
            with contextlib.suppress(OSError):
                connection, _ = tcp.accept()
                with connection:
                    connection.sendall(token.encode())

        def udp_serve():
            with contextlib.suppress(OSError):
                data, address = udp.recvfrom(128)
                udp.sendto(data, address)

        workers = [threading.Thread(target=f, daemon=True) for f in (tcp_serve, udp_serve)]
        for worker in workers:
            worker.start()
        settings = dict(uid=uid, gid=gid, net=os.readlink('/proc/self/ns/net'),
                        pid_ns=os.readlink('/proc/self/ns/pid'), outside=str(sentinel), token=token,
                        tcp=tcp.getsockname()[1], udp=udp.getsockname()[1])
        with sandbox_command(home_source, home, name, uid, gid,
                             [PYTHON, '-I', '-c', PROBE, json.dumps(settings)], native=native,
                             cwd=home) as (command, fds):
            result = subprocess.run(command, pass_fds=fds, capture_output=True, text=True,
                                    env=SAFE_ENV, timeout=20)
        require(result.returncode == 0,
                'sandbox probe failed (check kernel/userns/AppArmor/container policy):\n' + result.stderr)
        log(result.stdout.strip())
        for worker in workers:
            worker.join(timeout=0.2)


def check_dependencies() -> None:
    missing = missing_packages()
    require(not missing, 'missing preinstalled system packages: ' + ' '.join(missing)
            + '; provision them separately (the wrapper never elevates or installs packages)')
    check_bwrap()


def doctor(home: Path, account) -> None:
    check_dependencies()
    if home.exists() or home.is_symlink():
        validate_home(home, os.getuid())
        probe(home, home, account.pw_name, os.getuid(), os.getgid())
    else:
        with tempfile.TemporaryDirectory(prefix='claude-doctor-home-', dir='/tmp') as temporary:
            probe(Path(temporary), home, account.pw_name, os.getuid(), os.getgid())
    log(f'current UID {os.getuid()}; private home={home}; no elevation or Claude execution')


def close_descriptors(keep=()) -> None:
    for value in os.listdir('/proc/self/fd'):
        fd = int(value)
        if fd >= 3 and fd not in keep:
            with contextlib.suppress(OSError):
                os.close(fd)


def validate_stdio() -> None:
    for fd in (0, 1, 2):
        mode = os.fstat(fd).st_mode
        require(any(check(mode) for check in (stat.S_ISREG, stat.S_ISFIFO, stat.S_ISSOCK, stat.S_ISCHR)),
                f'stdio FD {fd} is not a stream (directory FDs would bypass path confinement)')


def exit_status(returncode: int) -> int:
    if returncode < 0:
        sig = -returncode
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        return 128 + sig
    return returncode


def terminal_run(command: list[str], fds: tuple[int, ...]) -> int:
    """Bridge an independent PTY, never grant the payload the host controlling TTY."""
    master, slave = pty.openpty()
    previous = termios.tcgetattr(0)
    handlers = {}
    process = None

    def resize(*_):
        size = fcntl.ioctl(0, termios.TIOCGWINSZ, b'\0' * 8)
        fcntl.ioctl(master, termios.TIOCSWINSZ, size)

    def forward(sig, _):
        try:
            group = os.tcgetpgrp(master)
            if group > 0:
                os.killpg(group, sig)
            elif process is not None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    def suspend(sig, frame):
        forward(sig, frame)
        termios.tcsetattr(0, termios.TCSADRAIN, previous)
        os.kill(os.getpid(), signal.SIGSTOP)
        tty.setraw(0)
        resize()
        forward(signal.SIGCONT, frame)

    try:
        termios.tcsetattr(slave, termios.TCSANOW, previous)
        resize()
        # Preserve an explicitly redirected stderr; only terminal streams share
        # the private PTY. Never hand the payload the original terminal FD.
        process = subprocess.Popen(command, stdin=slave, stdout=slave,
                                   stderr=slave if os.isatty(2) else None,
                                   pass_fds=fds, env=SAFE_ENV)
        os.close(slave)
        slave = -1
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGWINCH, signal.SIGTSTP):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, resize if sig == signal.SIGWINCH else suspend if sig == signal.SIGTSTP else forward)
        tty.setraw(0)
        os.set_blocking(master, False)
        pending_input = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(master, selectors.EVENT_READ)
            selector.register(0, selectors.EVENT_READ)
            stdin_active = True
            stdin_closed = False
            while True:
                for key, events in selector.select():
                    if key.fd == master and events & selectors.EVENT_WRITE:
                        try:
                            count = os.write(master, pending_input)
                            del pending_input[:count]
                        except BlockingIOError:
                            pass
                        except OSError as error:
                            if error.errno != errno.EIO:
                                raise
                            return process.wait()
                        if not pending_input:
                            selector.modify(master, selectors.EVENT_READ)
                        if not stdin_active and not stdin_closed and len(pending_input) < 65536:
                            selector.register(0, selectors.EVENT_READ)
                            stdin_active = True
                    if not events & selectors.EVENT_READ:
                        continue
                    try:
                        data = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    except OSError as error:
                        if key.fd == master and error.errno == errno.EIO:
                            data = b''
                        else:
                            raise
                    if not data:
                        if key.fd == master:
                            return process.wait()
                        selector.unregister(0)
                        stdin_active = False
                        stdin_closed = True
                        forward(signal.SIGHUP, None)
                        continue
                    if key.fd == master:
                        while data:
                            data = data[os.write(1, data):]
                    else:
                        # Never block on a large paste while the child is trying
                        # to write output; keep draining both PTY directions.
                        pending_input.extend(data)
                        selector.modify(master, selectors.EVENT_READ | selectors.EVENT_WRITE)
                        if len(pending_input) >= 65536:
                            selector.unregister(0)
                            stdin_active = False
    finally:
        termios.tcsetattr(0, termios.TCSADRAIN, previous)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        os.close(master)
        if slave >= 0:
            os.close(slave)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_local(options: Options, payload: list[str], account, home: Path) -> int:
    source = Path(options.claude).expanduser() if options.claude else default_source()
    require(source.exists(), f'native source not found: {source}; no download will be attempted')
    check_private_layout(home, source)
    check_dependencies()
    supplied = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    sandbox_env(home, account.pw_name, os.getuid(), supplied)  # fail before preparing mutable state
    with native_snapshot(source) as (native, resolved, sha):
        prepare_home(home)
        if options.setup_only:
            probe(home, home, account.pw_name, os.getuid(), os.getgid(), native=native)
            log(f'ready; home={home}; source={resolved}; sha256={sha}; no Claude executed')
            return 0
        # The actual bwrap launch enforces the same mandatory flags as doctor.
        # Do not start additional probe listeners or print diagnostics on SDK -v.
        requested_cwd = Path.cwd()
        cwd = requested_cwd if beneath(requested_cwd, home) else home / 'work'
        interactive = os.isatty(0) and os.isatty(1)
        with sandbox_command(home, home, account.pw_name, os.getuid(), os.getgid(),
                             ['/opt/claude/claude', *payload], native=native, supplied=supplied,
                             cwd=cwd, interactive=interactive) as (command, fds):
            if interactive:
                return exit_status(terminal_run(command, fds))
            os.chdir('/')
            close_descriptors(fds)
            for fd in fds:
                os.set_inheritable(fd, True)
            os.execve(str(BWRAP), command, SAFE_ENV)
    return 125


def main(argv=None) -> int:
    try:
        require(sys.flags.isolated, 'use /usr/bin/python3 -I FILE (or the executable shebang)')
        check_platform()
        validate_stdio()
        os.umask(0o077)
        options, payload = parse_args(list(sys.argv[1:] if argv is None else argv))
        if options.help:
            print(HELP)
            return 0
        account = current_identity()
        home = private_home(options)
        check_private_layout(home)
        if options.print_home:
            print(home)
            return 0
        if options.doctor:
            doctor(home, account)
            return 0
        return run_local(options, payload, account, home)
    except (SandboxError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        log(str(error))
        return 125
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
