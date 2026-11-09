#!/usr/bin/env python3
"""Write a sample of pythia's contexts, as the glui view of pythia's node
server sends them (pythia/interaction/node/view.py), for pages that show
them in a browser without it: web/contexts-index.html, an index of
contexts, and web/contexts-log.html, a log of ITEMS items, by
default 13,394, the most that one of pythia's contexts has had. `make web`
writes it, and `make web-serve` serves the pages.

usage: contexts_sample.py DIR [ITEMS]

In DIR: index.json and log.json, each page's facts as JSON steps
(src/lem/host.rs), the log's its turns; turn/TURN.json, each turn's items,
which the log's page asks for when the turn opens; and body/LINE.json, the
body of each item that has one, as Body facts, which the log's page asks
for when the item's entry opens.
"""

import json
import os
import shutil
import sys

ITEMS = 13394
ITEMS_PER_TURN = 50

# A turn's items, in a cycle, as a log has them: (tone, label, gist, body),
# where a body is (text, whether it's a diff), or None.
CYCLE = (
    ("User", "[user]", "Let's look at item {n}.", None),
    ("Dim", "[reasoning]", "Thinking about item {n}",
     ("First read item {n}, then change it.", False)),
    ("Tool", "[tool-call] exec_command", "git diff item_{n}.txt",
     ("git diff item_{n}.txt", False)),
    ("Tool", "[tool-ret] exec_command [ok]", "5 lines",
     ("diff --git a/item_{n}.txt b/item_{n}.txt\n@@ -1,2 +1,2 @@\n-old {n}\n+new {n}\n same", True)),
    ("Assistant", "[assistant]", "Item {n} is changed.", None),
    ("Dim", "[sample]", "1,234 in, 56 out", None),
)

LOG_PAGE = "/web/contexts-log.html"
INDEX_PAGE = "/web/contexts-index.html"


def write(path, steps):
    with open(path, "w") as f:
        json.dump(steps, f, separators=(",", ":"))


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__.split("\n\n")[1])
    out = sys.argv[1]
    items = int(sys.argv[2]) if len(sys.argv) == 3 else ITEMS
    turns = max(1, items // ITEMS_PER_TURN)
    for sub in ("body", "turn"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
        os.makedirs(os.path.join(out, sub))
    size = f"{items * 820 / 1e6:.1f} MB"
    index = [["+", "Page", {"title": "Contexts"}]]
    for rank, (cid, title, model, changed) in enumerate((
            ("sample-0", "A sample log", "opus", "now"),
            ("sample-1", "The same log, by another name", "opus", "5 min ago"),
            ("sample-2", "And again", "sonnet", "2 h ago"))):
        index.append(["+", "Context", {
            "id": cid, "title": title, "model": model, "turns": turns,
            "lines": items, "size": size, "changed": changed, "rank": rank,
            "to": LOG_PAGE}])
    write(os.path.join(out, "index.json"), index)
    log = [["+", "Page", {"title": f"A sample log: {items:,} items",
                          "items": LOG_PAGE + "#L", "turns": LOG_PAGE + "#T"}],
           ["+", "Nav", {"n": 0, "text": "Contexts", "to": INDEX_PAGE}]]
    by_turn = [[["+", "Loaded", {"turn": t}]] for t in range(turns)]
    bodies = 0
    for line in range(1, items + 1):
        tone, label, gist, body = CYCLE[(line - 1) % len(CYCLE)]
        turn = min((line - 1) // ITEMS_PER_TURN, turns - 1)
        by_turn[turn].append(["+", "Item", {
            "line": line, "turn": turn, "tone": tone, "label": label,
            "gist": gist.format(n=line), "body": body is not None}])
        if body is not None:
            text, diff = body
            write(os.path.join(out, "body", f"{line}.json"),
                  [["+", "Body", {"line": line, "part": 0, "text": text.format(n=line),
                                  "diff": diff}]])
            bodies += 1
    for turn, steps in enumerate(by_turn):
        count = len(steps) - 1
        log.append(["+", "Turn", {"n": turn, "label": f"Turn {turn + 1} \u00b7 {count:,} "
                                  f"item{'' if count == 1 else 's'}",
                                  "title": f"Let's look at items {steps[1][2]['line']:,} and on."}])
        write(os.path.join(out, "turn", f"{turn}.json"), steps)
    write(os.path.join(out, "log.json"), log)
    print(f"contexts_sample: {out}: an index, and a log of {items:,} items, "
          f"{bodies:,} with a body")


if __name__ == "__main__":
    main()
