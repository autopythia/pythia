"""The node server's settings: ``~/.pythia/node.ini``, read as the server starts, if
it's there, whose ``[node]`` gives the defaults of its options, and whose ``[remote
NAME]`` sections its remote nodes. ``--config FILE`` reads FILE in its place, and
``--config none`` reads none; so does ``$PYTHIA_NODE_CONFIG``, without the option.

A remote node is another machine with pythia, as pythia's ``install.py -e`` installs
it, which ssh reaches from this one, logging in by a key or an agent, never by a
password. The server shows its root contexts too, after this node's, at
``/TOKEN/n/NAME/``, by a node server that ssh starts there, on its standard streams,
at the first request (``remote.py``), which reads the remote's own node.ini, but for
its remotes. NAME is a host's name: letters, digits, '.', '_' and '-', from a letter or
a digit. For example::

    [node]
    port = 8766

    [remote cloud-0]
    host = cloud-0.example.org
    user = me

Every key is optional. A path is absolute, or from the home directory, as ``~/`` may
say: this node's for its own, the remote's for a remote's. ``[node]``'s keys, which the
options override, each as its option says:

- ``registry = DIR``: the registry whose root contexts to show, or ``none``; else
  ``$PYTHIA_REGISTRY``, else ``~/.pythia/root``.
- ``store = DIR``: where the glui view keeps what it keeps, or ``none``, to keep
  nothing; else ``~/.pythia/node``.
- ``port = N``: the port to serve on, on 127.0.0.1 only, or 0 for a free one; else
  8765.
- ``style = NAME``: the style that plain pages show in; else log-dark.

``[remote NAME]``'s, for the remote named NAME, which ``--remote NAME`` takes the place
of:

- ``host = HOST``: what ssh reaches: a host's name or address, or a ``Host`` of
  ``~/.ssh/config``, maybe as ``USER@HOST``; else NAME.
- ``user = USER``, ``port = N``, ``identity = FILE``: ssh's ``-l USER``, ``-p N`` and
  ``-i FILE``, FILE being a private key's, on this node.
- ``options = ...``: more of ssh's options, split into words as a shell splits them,
  such as ``-o ProxyJump=gateway``.
- ``pythia = DIR``: where pythia is there, the directory that holds its package, which
  goes on its Python's PYTHONPATH; else ``.pythia/lib/pythia-dev``, which pythia's
  installer links to its checkout; empty, for a Python that finds pythia alone.
- ``python = COMMAND``: the Python there, 3.9 or later; else ``python3``.
- ``registry = DIR``: the registry there; else the remote's own, as its node.ini, or
  its default, says.
- ``timeout = SECONDS``: how long a request waits for it, 3600 at most, before its
  process is ended, to start again at the next request; else 30.
- ``command = ...``: the command that reaches it, in the place of those keys but
  ``timeout``, split into words as a shell splits them: for a node that ssh doesn't
  reach so, or one tried on this node, such as ``python3 -m pythia.interaction.node
  --stdio --registry DIR``.

A value may go on over lines that start with a space. A file that isn't as this says
stops the server, which says where.
"""

from __future__ import annotations

import configparser
import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .._config_file import read_config_bytes
from .contexts import shown_path
from .remote import NAME, PYTHIA, PYTHON, TIMEOUT, Way, ssh_command

# What names the settings' file in place of ~/.pythia/node.ini, or none.
ENV = "PYTHIA_NODE_CONFIG"
OFF = "none"
MAX_BYTES = 64 * 1024
NODE_KEYS = ("registry", "store", "port", "style")
REMOTE_KEYS = ("host", "user", "port", "identity", "options", "pythia", "python",
               "registry", "timeout", "command")
MAX_TIMEOUT = 3600.0
_DIGITS = re.compile(r"[0-9]+\Z")
_SECONDS = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")
# A host or a user, as ssh takes one: a word, which doesn't start as an option does.
_SSH_WORD = re.compile(r"[^\s-]\S*\Z")


class Invalid(ValueError):
    """Why the settings can't be read: where, and what's wrong there."""


class Settings(NamedTuple):
    """What the settings say, None for what they don't, and the file they're from."""

    path: Optional[Path] = None
    registry: Optional[str] = None
    store: Optional[str] = None
    port: Optional[int] = None
    style: Optional[str] = None
    remotes: Tuple[Way, ...] = ()


def default_path() -> Path:
    return Path.home() / ".pythia" / "node.ini"


def read(option: Optional[str] = None) -> Settings:
    """The settings: from ``option`` (``--config``), else ``$PYTHIA_NODE_CONFIG``, else
    ``~/.pythia/node.ini``, if it's there; ``none`` reads none. A relative path is from
    the launch directory. Raises Invalid."""
    value = option if option is not None else os.environ.get(ENV)
    if value is not None and value.strip() == OFF:
        return Settings()
    given = value is not None and bool(value.strip())
    path = Path(value).expanduser().absolute() if given else default_path()
    shown = shown_path(path)
    try:
        data = read_config_bytes(path, MAX_BYTES)
    except FileNotFoundError:
        if given:
            raise Invalid(f"{shown}: there's no such file") from None
        return Settings()
    except OSError as exc:
        raise Invalid(f"{shown}: it can't be read: {exc.strerror or exc}") from None
    except ValueError as exc:  # not a file, or too long
        raise Invalid(f"{shown}: it can't be read: {str(exc).rstrip('.')}") from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Invalid(f"{shown}: it isn't UTF-8 text") from None
    return parse(text, shown)._replace(path=path)


def parse(text: str, source: str = "node.ini", home: Optional[Path] = None) -> Settings:
    """The settings that ``text`` says, ``source`` being where it's from, as an error
    says, and ``home`` the home directory. Raises Invalid."""
    home = Path.home() if home is None else home
    parser = configparser.ConfigParser(
        interpolation=None, strict=True, delimiters=("=",), comment_prefixes=("#", ";"),
        inline_comment_prefixes=None, empty_lines_in_values=False, default_section="\0")
    parser.optionxform = str  # keys as written
    try:
        parser.read_string(text, source=source)
    except configparser.Error as exc:
        raise Invalid(f"{source}: {_syntax(exc, text.splitlines())}") from None
    node: Optional[dict] = None
    remotes: Dict[str, Way] = {}
    for section in parser.sections():
        values = {key: parser.get(section, key) for key in parser.options(section)}
        where, words = f"{source}: [{section}]", section.split()
        if words == ["node"]:
            if node is not None:
                raise Invalid(f"{where}: [node] comes twice")
            node = _node(values, where, home)
        elif len(words) == 2 and words[0] == "remote":
            name = words[1]
            if not NAME.match(name):
                raise Invalid(f"{where}: {name!r} isn't a node's name: letters, digits, "
                              "'.', '_' and '-', from a letter or a digit")
            if name in remotes:
                raise Invalid(f"{where}: remote {name} comes twice")
            remotes[name] = _remote(name, values, where, home)
        else:
            raise Invalid(f"{where} isn't a section of node.ini: give [node], or "
                          "[remote NAME]")
    return Settings(remotes=tuple(remotes.values()), **(node or {}))


def _syntax(exc: configparser.Error, lines: Sequence[str]) -> str:
    """What's wrong with the text as INI, and where."""
    lineno = getattr(exc, "lineno", None)
    if lineno is None and getattr(exc, "errors", None):
        lineno = exc.errors[0][0]
    line = lines[lineno - 1].strip() if lineno and lineno <= len(lines) else ""
    if isinstance(exc, configparser.DuplicateSectionError):
        what = f"[{exc.section}] comes twice"
    elif isinstance(exc, configparser.DuplicateOptionError):
        what = f"[{exc.section}] has {exc.option} twice"
    elif isinstance(exc, configparser.MissingSectionHeaderError):
        what = f"{line!r} comes before any [section]"
    elif isinstance(exc, configparser.ParsingError):
        what = f"{line!r} isn't a [section], or a key = value"
    else:
        what = str(exc).splitlines()[0]
    return f"line {lineno}: {what}" if lineno else what


def _node(values: Dict[str, str], where: str, home: Path) -> dict:
    """[node]'s settings, by name."""
    _known(values, NODE_KEYS, where)
    node: dict = {}
    for key in ("registry", "store"):
        if key in values:
            node[key] = _here(values[key], f"{where} {key}", home, off=True)
    if "port" in values:
        node["port"] = _port(values["port"], f"{where} port", lowest=0)
    if "style" in values:
        style = values["style"]
        if not style or any(each.isspace() for each in style):
            raise Invalid(f"{where} style: give a style's name")
        node["style"] = style
    return node


def _remote(name: str, values: Dict[str, str], where: str, home: Path) -> Way:
    """A [remote NAME]'s way to it."""
    _known(values, REMOTE_KEYS, where)
    timeout = (_seconds(values["timeout"], f"{where} timeout") if "timeout" in values
               else TIMEOUT)
    if "command" in values:
        also = [key for key in REMOTE_KEYS if key in values and key not in
                ("command", "timeout")]
        if also:
            raise Invalid(f"{where}: its command reaches it in the place of {_listed(also)}: "
                          "give one or the other")
        return Way(name, _words(values["command"], f"{where} command"), timeout)
    host = _ssh_word(values.get("host", name), f"{where} host", "host")
    user = _ssh_word(values["user"], f"{where} user", "user") if "user" in values else None
    port = _port(values["port"], f"{where} port", lowest=1) if "port" in values else None
    identity = (_here(values["identity"], f"{where} identity", home)
                if "identity" in values else None)
    options = _words(values.get("options", ""), f"{where} options", empty=True)
    pythia = _there(values.get("pythia", PYTHIA), f"{where} pythia", empty=True)
    python = (_words(values["python"], f"{where} python") if "python" in values
              else list(PYTHON))
    registry = (_there(values["registry"], f"{where} registry") if "registry" in values
                else None)
    return Way(name, ssh_command(host, user=user, port=port, identity=identity,
                                 options=options, pythia=pythia, python=python,
                                 registry=registry), timeout)


def _known(values: Dict[str, str], keys: Sequence[str], where: str) -> None:
    for key in values:
        if key not in keys:
            raise Invalid(f"{where} has no {key}: its keys are {_listed(keys)}")


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, or ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _here(value: str, where: str, home: Path, *, off: bool = False) -> str:
    """A path on this node, from ``home`` if it's relative; or none, if it may be."""
    if off and value == OFF:
        return OFF
    if not value or "\n" in value:
        raise Invalid(f"{where}: give a path, on one line{', or none' if off else ''}")
    if value == "~" or value.startswith("~/"):
        return str(home / value[2:])
    try:
        path = Path(value).expanduser()
    except RuntimeError:  # ~user, of no such user
        raise Invalid(f"{where}: {value!r} names no home directory") from None
    return str(path if path.is_absolute() else home / path)


def _there(value: str, where: str, *, empty: bool = False) -> str:
    """A path on a remote, as its shell takes one there: absolute, or from its home
    directory, where ssh starts; or nothing, if it may be."""
    if not value and empty:
        return ""
    if not value or "\n" in value:
        raise Invalid(f"{where}: give a path, on one line")
    if value == "~" or value.startswith("~/"):
        return value[2:].lstrip("/") or "."
    if value.startswith("~"):
        raise Invalid(f"{where}: {value!r}: give a path from the home directory there, "
                      "or an absolute one")
    return value


def _port(value: str, where: str, *, lowest: int) -> int:
    port = int(value) if _DIGITS.match(value) else -1
    if not lowest <= port <= 65535:
        raise Invalid(f"{where}: {value!r} isn't a port: give {lowest} to 65535")
    return port


def _seconds(value: str, where: str) -> float:
    seconds = float(value) if _SECONDS.match(value) else 0.0
    if not 0 < seconds <= MAX_TIMEOUT:
        raise Invalid(f"{where}: {value!r} isn't a timeout: give seconds, more than 0 and "
                      f"{MAX_TIMEOUT:g} at most")
    return seconds


def _ssh_word(value: str, where: str, what: str) -> str:
    if not _SSH_WORD.match(value):
        raise Invalid(f"{where}: {value!r} isn't a {what} that ssh takes: a word that "
                      "doesn't start with '-'")
    return value


def _words(value: str, where: str, *, empty: bool = False) -> List[str]:
    """``value``, split into words as a shell splits them."""
    try:
        words = shlex.split(value)
    except ValueError as exc:  # No closing quotation
        raise Invalid(f"{where}: {exc}") from None
    if not words and not empty:
        raise Invalid(f"{where}: give a command")
    return words
