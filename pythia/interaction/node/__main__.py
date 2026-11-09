"""``python3 -m pythia.interaction.node``: serve this node's root contexts, read-only.

The root contexts are those that the node's registry lists, wherever their
saves are: ``--registry``, else ``$PYTHIA_REGISTRY``, else ``~/.pythia/root``,
as the CLI finds it. A relative path is taken from the launch directory.
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
from .server import ADDRESS, App, Server
from .styles import DEFAULT_STYLE, Styles

PROG = "python3 -m pythia.interaction.node"
DEFAULT_PORT = 8765
# --list cuts a row's last column to the terminal's width, but leaves at least this much.
MIN_TITLE = 12


def port_number(text: str) -> int:
    try:
        port = int(text, 10)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a port: give 0 to 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=("Serve this node's root contexts, those that its registry lists, as web "
                     "pages, read-only, on 127.0.0.1."))
    parser.add_argument("--registry", metavar="DIR", default=None, help=(
        "the registry whose root contexts to show, read and never written. A relative "
        "path is from the launch directory (default: $PYTHIA_REGISTRY, else "
        "~/.pythia/root)"))
    parser.add_argument("--port", metavar="N", type=port_number, default=DEFAULT_PORT, help=(
        "the port to serve on, on 127.0.0.1 only; 0 picks a free one (default: "
        "%(default)s)"))
    parser.add_argument("--list", action="store_true", help=(
        "print the root contexts and exit, serving nothing"))
    parser.add_argument("--style", metavar="NAME", default=DEFAULT_STYLE, help=(
        "the style that pages show in, unless their URL names another: log-dark, "
        "log-light, chat, or another in the server's css/. The menu at each page's top "
        "right switches (default: %(default)s)"))
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
    styles = Styles()
    if styles.get(args.style) is None:
        names = ", ".join(style.name for style in styles.all())
        parser.error(f"argument --style: {args.style!r} isn't a style: give one of {names}")
    path = registry_path(args.registry)
    if path is None:
        _error("the registry is off (--registry none, or PYTHIA_REGISTRY=none), and the node "
               "server shows its root contexts: give one with --registry.")
        return 1
    roots = Roots(path)
    if args.list:
        width = shutil.get_terminal_size((80, 24)).columns
        print(render_list(roots, width), flush=True)
        return 0
    try:
        server = Server(App(roots, style=args.style, styles=styles), args.port)
    except OSError as exc:
        _error(f"can't serve on {ADDRESS}:{args.port}: {exc}; give another port with --port, "
               "or --port 0 for a free one.")
        return 1
    with server:
        _say(roots.problem() or _counted(roots))
        _say(server.url)
        files = server.app.view.files
        if files.missing():
            _say(f"glui's view can't be shown: it needs files that aren't in "
                 f"{shown_path(files.directory)}/: {', '.join(files.missing())}")
        else:
            _say(f"in glui's view: {server.url}lem/")
        port = server.server_port
        _say(f"from another machine: ssh -L {port}:{ADDRESS}:{port} HOST")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
