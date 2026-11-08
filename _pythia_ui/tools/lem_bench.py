#!/usr/bin/env python3
"""Time lem on the shape of pythia's contexts, and on a join that needs a
good order ("Evaluation at log size" and "Joins after meringue" in pythia's
doc/pythia-ui-gui-design.md).

usage: lem_bench.py PYTHIA_UI

The log: a log's turns, and each turn's items, as glem widgets. For each
size, up to 13,394 items, the most that one context in pythia's
.interaction/ had when this was written, it runs `PYTHIA_UI lem` twice: with
the facts and a first tick, and with ten more ticks, each of which changes
one item. It prints the first run's time, which includes reading every fact,
and a tick's, from the difference; both include printing the widgets.

The star: triangles in a star, a hub and its leaves, with edges both ways
and no triangles, where a join that takes its atoms as written walks every
path of two steps through the hub, n² of them, and one that takes the
smallest bucket first walks about 4n facts.

It checks that every tick prints what it should, and exits 1 if one
doesn't.
"""

import os
import subprocess
import sys
import tempfile
import time

SIZES = (1000, 2000, 4000, 8000, 13394)
ITEMS_PER_TURN = 50
MORE = 10

PROGRAM = """\
-- A log's shape: turns, and each turn's items, as glem widgets.
surface Page(title: Text)
widget Group(title: Text)
widget Line(text: Text)
id Top, T(Int), I(Int)
input Turn(n: Int)
input Item(line: Int, turn: Int, text: Text)
Page(Top, title="log"):
  for Turn(n):
    Group(T(n), title="turn", order=n):
      for Item(l, n, text):
        Line(I(l), text=text, order=l)
"""


STAR_SIZES = (1000, 4000, 16000, 64000)

STAR = """\
-- Triangles, the classic case for worst-case optimal joins.
input E(a: Int, b: Int)
rel Tri(a: Int, b: Int, c: Int)
Tri(a, b, c) :- E(a, b), E(b, c), E(a, c)
"""


def ticks(items, turns, more):
    """The ticks, as `pythia-ui lem` reads them: the facts, then `more` ticks
    that change item 0 back and forth."""
    lines = [f"+ Turn({t})" for t in range(turns)]
    lines += [f'+ Item({i}, {i % turns}, "item {i}")' for i in range(items)]
    text = "\n".join(lines) + "\n"
    for k in range(more):
        old, new = ("item 0", "changed") if k % 2 == 0 else ("changed", "item 0")
        text += f'\n- Item(0, 0, "{old}")\n+ Item(0, 0, "{new}")\n'
    return text


def run(exe, program, text, want):
    """Seconds that `exe lem program` takes on `text`; it must print `want`
    lines and nothing on stderr."""
    start = time.perf_counter()
    done = subprocess.run([exe, "lem", program], input=text.encode(),
                          capture_output=True, timeout=600)
    seconds = time.perf_counter() - start
    got = done.stdout.count(b"\n")
    if done.returncode != 0 or done.stderr or got != want:
        sys.exit(f"lem_bench: exit {done.returncode}, {got} lines rather than {want}: "
                 f"{done.stderr[:300].decode(errors='replace')}")
    return seconds


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        program = os.path.join(tmp, "log.lem")
        with open(program, "w") as f:
            f.write(PROGRAM)
        print(f"{'items':>6}  {'first run':>10}  {'a tick':>8}")
        for items in SIZES:
            turns = max(1, items // ITEMS_PER_TURN)
            # A tick prints the page, its turns and items, and a blank line.
            lines = 1 + turns + items + 1
            first = run(exe, program, ticks(items, turns, 0), lines)
            more = run(exe, program, ticks(items, turns, MORE), lines * (1 + MORE))
            tick = max(0.0, more - first) / MORE
            print(f"{items:>6,}  {first:>8.3f} s  {tick * 1000:>5.1f} ms", flush=True)
        star = os.path.join(tmp, "star.lem")
        with open(star, "w") as f:
            f.write(STAR)
        print(f"\n{'leaves':>6}  {'a tick':>10}")
        for leaves in STAR_SIZES:
            edges = [f"+ E(0, {i})\n+ E({i}, 0)\n" for i in range(1, leaves + 1)]
            # No triangles: the tick prints no widgets, and no facts.
            seconds = run(exe, star, "".join(edges) + "? Tri\n", 2)
            print(f"{leaves:>6,}  {seconds:>8.3f} s", flush=True)


if __name__ == "__main__":
    main()
