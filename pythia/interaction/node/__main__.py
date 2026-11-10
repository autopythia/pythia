"""``python3 -m pythia.interaction.node``: serve this node's root contexts, read-only.

The root contexts are those that the node's registry lists, wherever their
saves are: ``--registry``, else the settings' (``config.py``), else
``$PYTHIA_REGISTRY``, else ``~/.pythia/root``, as the CLI finds it. The glui
view's settings are kept in ``--store``, else the settings', else
``~/.pythia/node``; ``--store none`` keeps nothing. A relative path is taken
from the launch directory. With ``--stdio``, the server answers on its
standard streams instead, as lines of JSON, for a node server elsewhere that
reaches this node by ssh (``stdio.py``); and with ``--remote HOST``, or a
``[remote HOST]`` section of its settings, a server shows HOST's root contexts
too, reaching it so (``remote.py``). Its settings, in ``~/.pythia/node.ini``,
give its options' defaults, which the options override (``config.py``).
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .._cli_editor import safe_text
from ..registry import abbreviate, registry_path, save_prefix_of
from .contexts import Roots, shown_path
from .pages import _KEY, absent_note, when
from . import config, remote, stdio
from .server import ADDRESS, App, Server
from .store import Store
from .styles import DEFAULT_STYLE, Styles

PROG = "python3 -m pythia.interaction.node"
DEFAULT_PORT = 8765
# --list cuts a row's last column to the terminal's width, but leaves at least this much.
MIN_TITLE = 12
# What --help says of the settings, after the options.
SETTINGS = f"""\
settings: ~/.pythia/node.ini, if it's there, unless --config or
${config.ENV} names another file, or none. Its [node] section gives the
defaults of --registry, --store, --port and --style, as registry = DIR, and so
on; and each [remote NAME] section, a remote node to show, which ssh reaches,
and runs a node server on, with pythia where its install.py -e links it:

  [remote NAME]
  host = HOST          what ssh reaches (default: NAME)
  user = USER          ssh -l USER
  port = N             ssh -p N
  identity = FILE      ssh -i FILE
  options = ...        more of ssh's options
  pythia = DIR         where pythia is there (default: {remote.PYTHIA})
  python = COMMAND     its Python (default: python3)
  registry = DIR       its registry (default: its own)
  timeout = SECONDS    how long a request waits for it (default: {remote.TIMEOUT:g})
  command = ...        the command that reaches it, in the place of all but timeout

pythia/interaction/node/config.py says more."""


def remote_node(text: str) -> remote.Way:
    """A --remote's value, as remote.parse takes it."""
    try:
        return remote.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def port_number(text: str) -> int:
    try:
        port = int(text, 10)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a port: give 0 to 65535")
    return port


# Where the glui view keeps what it keeps, unless --store says otherwise.
STORE = Path.home() / ".pythia" / "node"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, epilog=SETTINGS, formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Serve this node's root contexts, those that its registry lists, as web "
                     "pages, read-only but for the view's store, on 127.0.0.1."))
    parser.add_argument("--registry", metavar="DIR", default=None, help=(
        "the registry whose root contexts to show, read and never written. A relative "
        "path is from the launch directory (default: the settings' registry, else "
        "$PYTHIA_REGISTRY, else ~/.pythia/root)"))
    parser.add_argument("--port", metavar="N", type=port_number, default=None, help=(
        "the port to serve on, on 127.0.0.1 only; 0 picks a free one (default: "
        f"{DEFAULT_PORT})"))
    way = parser.add_mutually_exclusive_group()
    way.add_argument("--list", action="store_true", help=(
        "print the root contexts and exit, serving nothing"))
    way.add_argument("--stdio", action="store_true", help=(
        "serve on standard input and output, as lines of JSON, for a node server elsewhere "
        "that reaches this node by ssh, with no port and no token"))
    parser.add_argument("--style", metavar="NAME", default=None, help=(
        "the style that pages show in, unless their URL names another: log-dark, "
        "log-light, chat, or another in the server's css/. The menu at each page's top "
        f"right switches (default: {DEFAULT_STYLE})"))
    parser.add_argument("--remote", metavar="HOST", type=remote_node, action="append",
                        default=[], help=(
        "show HOST's root contexts too, at /TOKEN/n/HOST/, as a node server that it runs "
        "with --stdio, reached by 'ssh -o BatchMode=yes HOST', as a [remote HOST] section "
        "of the settings with no keys does, in that section's place; NAME=COMMAND reaches "
        "a node named NAME by COMMAND instead, such as 'python3 -m pythia.interaction.node "
        "--stdio --registry DIR'. Give it once for each remote node"))
    parser.add_argument("--store", metavar="DIR", default=None, help=(
        "where the glui view keeps what it keeps across the server's runs, such as its "
        "settings, as a browser keeps its profile; 'none' keeps nothing. A relative path is "
        "from the launch directory (default: ~/.pythia/node)"))
    parser.add_argument("--config", metavar="FILE", default=None, help=(
        "the settings to read, in place of ~/.pythia/node.ini, as below; 'none' reads none. "
        f"A relative path is from the launch directory (default: ${config.ENV}, else "
        "~/.pythia/node.ini, if it's there)"))
    return parser


def _say(text: str) -> None:
    print(f"node: {safe_text(text)}", flush=True)


def _error(text: str) -> None:
    print(f"node: {safe_text(text)}", file=sys.stderr, flush=True)


def _counted(roots: Roots) -> str:
    """How many root contexts the registry lists, and how many of them aren't where
    they're registered: the line that serving starts with."""
    entries, found = roots.entries(), roots.contexts()
    absent = len(entries) - len(found)
    text = (f"{len(entries)} root context{'' if len(entries) == 1 else 's'} in "
            f"{shown_path(roots.path)}/")
    if absent == 1:
        text += "; 1 isn't where it's registered"
    elif absent:
        text += f"; {absent} aren't where they're registered"
    return text + "."


def render_list(roots: Roots, width: int) -> str:
    """``--list``: the root contexts, the last changed first, then those that aren't where
    they're registered, with their state and save, after a heading."""
    entries = roots.entries()
    if not entries:
        return safe_text(roots.problem() or
                         f"Root contexts in {shown_path(roots.path)}/: none yet.")
    contexts = roots.contexts()
    problems: Dict[str, str] = {}
    for context in contexts:
        try:
            context.refresh()
        except OSError as exc:
            problems[context.id] = str(exc)
    keys = [entry.key for entry in entries]
    short = {entry.number: abbreviate(entry.key, keys) if _KEY.match(entry.key) else entry.key
             for entry in entries}
    names = {context.number: safe_text(context.name) for context in contexts}
    name_width = max(7, *(len(name) for name in names.values())) if names else 7
    key_width = max(3, *(len(key) for key in short.values()))
    found = {context.number for context in contexts}
    absent = [entry for entry in entries if entry.number not in found]
    lines = [safe_text(f"Root contexts in {shown_path(roots.path)}/: {len(entries)}, the last "
                       f"changed first.{absent_note(len(absent))}"),
             f"{'#':>4}  {'context':<{name_width}}  {'key':<{key_width}}  {'changed':<16}  "
             f"{'turns':>5}  title"]

    def row(head: str, last: str, *, path: bool = False) -> str:
        """A row, its last column cut to fit: a title at its end, a path at its start."""
        text = safe_text(last)
        room = max(width - len(head), MIN_TITLE)
        if len(text) <= room:
            return head + text
        return head + ("\u2026" + text[1 - room:] if path else text[:room - 1] + "\u2026")

    for context in sorted(contexts, key=lambda each: (-each.index.changed, each.number)):
        if context.id in problems:
            changed, turns = "?", "?"
            title = f"can't read it: {problems[context.id]}"
        else:
            changed, turns = when(context.index.changed), str(len(context.turns()))
            title = context.title() or "-"
        lines.append(row(f"{context.number:>4}  {names[context.number]:<{name_width}}  "
                         f"{safe_text(short[context.number]):<{key_width}}  {changed:<16}  "
                         f"{turns:>5}  ", title))
    for entry in absent:
        where = "-" if entry.home is None else shown_path(save_prefix_of(Path(entry.home)))
        lines.append(row(f"{entry.number:>4}  {'-':<{name_width}}  "
                         f"{safe_text(short[entry.number]):<{key_width}}  "
                         f"{entry.state:<16}  {'':>5}  ", where, path=True))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = config.read(args.config)
    except config.Invalid as exc:
        _error(f"{exc}.")
        return 2
    where = shown_path(settings.path) if settings.path else "node.ini"
    styles = Styles()
    style = next(each for each in (args.style, settings.style, DEFAULT_STYLE)
                 if each is not None)
    if styles.get(style) is None:
        names = ", ".join(each.name for each in styles.all())
        if args.style is not None:
            parser.error(f"argument --style: {style!r} isn't a style: give one of {names}")
        _error(f"{where}: [node] style: {style!r} isn't a style: give one of {names}.")
        return 2
    option, off = args.registry, "--registry none"
    if option is None:
        option, off = ((settings.registry, f"registry = none in {where}")
                       if settings.registry is not None else (None, "PYTHIA_REGISTRY=none"))
    path = registry_path(option)
    if path is None:
        _error(f"the registry is off ({off}), and the node server shows its root contexts: "
               "give one with --registry.")
        return 1
    roots = Roots(path)
    if args.list:
        width = shutil.get_terminal_size((80, 24)).columns
        print(render_list(roots, width), flush=True)
        return 0
    kept = args.store if args.store is not None else settings.store
    stored = Path(kept).expanduser().absolute() if kept else STORE
    store = None if kept == "none" else Store(stored)
    if args.stdio and args.remote:
        parser.error("argument --remote: not allowed with argument --stdio")
    if args.stdio:
        # A node reached so shows its own roots alone: its settings' remotes are for its
        # own server.
        return stdio.serve(roots, store=store, style=style, styles=styles)
    names = [way.name for way in args.remote]
    twice = next((name for name in names if names.count(name) > 1), None)
    if twice is not None:
        parser.error(f"argument --remote: {twice} is given twice")
    # The settings' remotes, each in its place, unless --remote names it, then the others
    # that --remote names.
    given = {way.name: way for way in args.remote}
    remotes = [given.pop(way.name, way) for way in settings.remotes] + list(given.values())
    port = next(each for each in (args.port, settings.port, DEFAULT_PORT) if each is not None)
    app = App(roots, style=style, styles=styles, store=store, remotes=remotes)
    try:
        server = Server(app, port)
    except OSError as exc:
        _error(f"can't serve on {ADDRESS}:{port}: {exc}; give another port with --port, "
               "or --port 0 for a free one.")
        return 1
    try:
        return _serve(server, roots, settings.path)
    finally:
        app.close()


def _serve(server: Server, roots: Roots, settings: Optional[Path] = None) -> int:
    with server:
        if settings is not None:
            _say(f"settings: {shown_path(settings)}")
        _say(roots.problem() or _counted(roots))
        _say(server.url)
        files = server.app.view.files
        if files.missing():
            _say(f"glui's view can't be shown: it needs files that aren't in "
                 f"{shown_path(files.directory)}/: {', '.join(files.missing())}")
        else:
            _say(f"in glui's view: {server.url}lem/")
        for name in server.app.remotes:
            _say(f"remote node {name}: {server.url}n/{name}/, and in glui's view: "
                 f"{server.url}n/{name}/lem/")
        port = server.server_port
        _say(f"from another machine: ssh -L {port}:{ADDRESS}:{port} HOST")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
