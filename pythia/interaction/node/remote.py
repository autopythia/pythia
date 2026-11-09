"""The node server's remotes (``--remote``): other nodes, each reached by a command that
runs a node server there on its standard streams (``stdio.py``), as ssh does, whose
pages the local server answers ``/TOKEN/n/NAME/...`` with (``server.py``).

``--remote HOST`` reaches HOST by ``ssh -o BatchMode=yes HOST python3 -m
pythia.interaction.node --stdio``; ``--remote NAME=COMMAND`` by COMMAND, split into words
as a shell splits them, for a node that's reached otherwise, or one tried on this node. A
remote's process starts at its first request, and again after it ends, once in ``RETRY``
seconds at most, and it ends with the local server, as its input does. A request that it
doesn't answer in ``TIMEOUT`` seconds gets 504, and the process ends, to start again;
one while it can't be reached gets 502, which says why, by what it said last on its
standard error.
"""

from __future__ import annotations

import base64
import collections
import itertools
import json
import math
import queue
import re
import shlex
import subprocess
import threading
import time
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .pages import Response

# The protocol of a node on its standard streams, which its first line says.
PROTOCOL = 1
SSH = ("ssh", "-o", "BatchMode=yes")
STDIO = ("python3", "-m", "pythia.interaction.node", "--stdio")
# A node's name, which its pages' paths hold: as a host's, with no user and no port.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
TIMEOUT = 30.0
RETRY = 5.0
# An answer's line at most: the longest of a log's pages, with room to spare.
MAX_ANSWER = 64 * 1024 * 1024
# The lines of its standard error that a remote's failure can say.
SAID = 20


def parse(text: str) -> Tuple[str, List[str]]:
    """A ``--remote``'s value, as the node's name and the command that reaches it.
    Raises ValueError, which says why it isn't one."""
    name, given, command = text.partition("=")
    if not NAME.match(name):
        raise ValueError(f"{name!r} isn't a node's name: letters, digits, '.', '_' and '-', "
                         "from a letter or a digit; for a host that isn't, give "
                         "NAME=COMMAND")
    if not given:
        return name, [*SSH, name, *STDIO]
    argv = shlex.split(command)
    if not argv:
        raise ValueError(f"{name}'s command is empty")
    return name, argv


def _plain(status: int, text: str) -> Response:
    return Response(status, text + "\n", content_type="text/plain")


class Remote:
    """A remote node, named ``name``, reached by ``argv``, whose pages start with ``base``,
    which it's given."""

    def __init__(self, name: str, argv: Sequence[str], base: str, *,
                 timeout: float = TIMEOUT, retry: float = RETRY) -> None:
        self.name = name
        self.argv = list(argv)
        self.base = base
        self.timeout = timeout
        self.retry = retry
        # What it said first, its host's name and how many roots it has, once it has;
        # and why it can't be reached, if it can't.
        self.hello: Optional[dict] = None
        self.why: Optional[str] = None
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._started = -math.inf
        self._closed = False
        self._ids = itertools.count(1)
        self._waiting: Dict[int, "queue.Queue[Response]"] = {}
        self._said: Deque[str] = collections.deque(maxlen=SAID)

    def request(self, method: str, target: str, host: str, *, origins: Sequence[str] = (),
                marked: bool = False, content_type: Optional[str] = None,
                body: Optional[bytes] = None) -> Response:
        """Its answer to a request, as App.respond takes one, or why there's none."""
        answer: "queue.Queue[Response]" = queue.Queue(maxsize=1)
        with self._lock:
            if self._closed:
                return self._unreachable("the server is stopping")
            if self._process is None:
                wait = self._started + self.retry - time.monotonic()
                if wait > 0:
                    return self._unreachable(f"{self.why}; it's tried again in "
                                             f"{math.ceil(wait)} s")
                if not self._start():
                    return self._unreachable(self.why)
            process = self._process
            rid = next(self._ids)
            request = {"id": rid, "method": method, "target": target, "host": host}
            if method == "POST":
                request.update(origins=list(origins), marked=marked, type=content_type,
                               body=None if body is None else base64.b64encode(body).decode())
            self._waiting[rid] = answer
            try:
                process.stdin.write(json.dumps(request).encode("ascii") + b"\n")
                process.stdin.flush()
            except (OSError, ValueError):  # it ended, and its reader will say why
                pass
        try:
            return answer.get(timeout=self.timeout)
        except queue.Empty:
            with self._lock:
                self._waiting.pop(rid, None)
            self._end(process, f"it didn't answer in {self.timeout:g} s")
            return _plain(504, f"The remote node {self.name} didn't answer in "
                          f"{self.timeout:g} s; it's started again for the next request.")

    def close(self) -> None:
        """Ends its process, if it runs, by the end of its input, else by a signal."""
        with self._lock:
            self._closed = True
            process = self._process
        if process is not None:
            self._end(process, "the server is stopping")

    def _unreachable(self, why: Optional[str]) -> Response:
        return _plain(502, f"The remote node {self.name} can't be reached: {why}.")

    def _start(self) -> bool:
        """Starts its process, with its lock held: whether it started."""
        self._started = time.monotonic()
        self._said.clear()
        self.hello = None
        try:
            process = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
        except OSError as exc:
            self.why = f"its command couldn't start: {exc}"
            return False
        self._process = process
        self.why = None
        errors = threading.Thread(target=self._read_errors, args=(process,), daemon=True,
                                  name=f"node remote {self.name} errors")
        errors.start()
        threading.Thread(target=self._read, args=(process, errors), daemon=True,
                         name=f"node remote {self.name}").start()
        try:
            process.stdin.write(json.dumps({"node": PROTOCOL, "base": self.base})
                                .encode("ascii") + b"\n")
            process.stdin.flush()
        except OSError:
            pass  # it ended, and its reader will say why
        return True

    def _read(self, process: subprocess.Popen, errors: threading.Thread) -> None:
        """Hands each answer that the process gives to its request, until it ends."""
        why = None
        try:
            first = process.stdout.readline(MAX_ANSWER + 1)
            hello = _json(first)
            if not first:
                pass  # it ended, and what it said last says why
            elif not isinstance(hello, dict) or hello.get("node") != PROTOCOL:
                why = "it didn't answer as a node does"
            elif "error" in hello:
                why = f"it refused: {hello['error']}"
            else:
                with self._lock:
                    if self._process is process:
                        self.hello = hello
                while why is None:
                    line = process.stdout.readline(MAX_ANSWER + 1)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        why = f"an answer was longer than {MAX_ANSWER:,} bytes"
                        break
                    value = _json(line)
                    rid = value.get("id") if isinstance(value, dict) else None
                    with self._lock:
                        waiting = self._waiting.pop(rid, None) if type(rid) is int else None
                    if waiting is not None:
                        waiting.put(_answer(self.name, value))
        except (OSError, ValueError):
            pass
        self._end(process, why, errors)
        process.stdout.close()

    def _read_errors(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stderr:
                text = line.decode("utf-8", "replace").strip()
                if text:
                    self._said.append(text)
        except (OSError, ValueError):
            pass
        process.stderr.close()

    def _end(self, process: subprocess.Popen, why: Optional[str],
             errors: Optional[threading.Thread] = None) -> None:
        """Ends ``process``, if it's still the one that runs, for ``why``, else for what it
        said last; and answers the requests that wait for it."""
        with self._lock:
            if self._process is not process:
                return
            self._process = None
            waiting, self._waiting = self._waiting, {}
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            status = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            status = process.wait()
        if errors is not None:
            errors.join(timeout=1)
        if why is None:
            said = self._said[-1] if self._said else "it said nothing"
            why = f"it ended, with status {status}: {said}"
        self.why = why
        for answer in waiting.values():
            answer.put(self._unreachable(why))


def _json(line: bytes) -> object:
    try:
        return json.loads(line) if line else None
    except ValueError:
        return None


def _answer(name: str, value: dict) -> Response:
    """An answer's line as a response; or, if it isn't one, a 502 that says so."""
    status, body, kind = value.get("status"), value.get("body"), value.get("type")
    location, headers, csp = value.get("location"), value.get("headers", []), value.get("csp")
    if (type(status) is not int or not 100 <= status <= 599 or not isinstance(body, str)
            or not isinstance(kind, str) or not isinstance(location, (str, type(None)))
            or not isinstance(csp, (str, type(None))) or not isinstance(headers, list)
            or not all(isinstance(header, list) and len(header) == 2
                       and all(isinstance(part, str) for part in header)
                       for header in headers)):
        return _plain(502, f"The remote node {name} answered with a line that isn't an "
                      "answer.")
    return Response(status, body, content_type=kind, location=location,
                    headers=[tuple(header) for header in headers], csp=csp)
