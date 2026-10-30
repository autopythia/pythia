"""Write a save's token traces as JSONL: ``python3 -m pythia.interaction.tokens SAVE``.

SAVE is a CLI save (its prefix, or its PREFIX.jsonl, whose full log
PREFIX._private/context.jsonl holds the traces) or another full log, as the
demo writes. Each line is one JSON object:

- ``{"type": "record", ...}``, for each sample that has a token trace: its
  metadata's ``line`` in the full log, its input tokens, whether its whole
  prompt was ``rebuilt`` from the records before it, and if not the
  ``problem``, then the record's fields, with arrays as lists (NaN as null);
- ``{"type": "segment", ...}``, after the last sample of each run that makes
  one sequence (``pythia.interaction.tokens.segments``): its first and last
  ``lines``, its ``token_ids``, the ``sampled`` spans as ``[start, end]``
  pairs, and ``log_probs``, a sampled token's at its position, null
  elsewhere.

``--records`` or ``--segments`` writes only those. Exit status: 0 when the
save was read, 1 when it couldn't be, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._cli_save import full_log_path
from ..save import SaveError
from ..save import load_interaction_save
from . import TOKEN_RECORD_FIELDS
from . import LogProbs
from . import Segment
from . import TokenIds
from . import TracedSample
from . import segments
from . import traced_samples


def _values(array) -> List[Any]:
    values = array.tolist()
    if isinstance(array, LogProbs):
        return [None if math.isnan(value) else value for value in values]
    return values


def record_json(sample: TracedSample) -> Dict[str, Any]:
    """A record line: where its sample is, how its rebuild went, its fields."""
    encoded: Dict[str, Any] = {"type": "record", "line": sample.line,
                               "input_tokens": sample.input_tokens,
                               "rebuilt": sample.prompt is not None}
    if sample.problem is not None:
        encoded["problem"] = sample.problem
    for name in TOKEN_RECORD_FIELDS:
        value = getattr(sample.record, name)
        if value is not None:
            encoded[name] = _values(value) if isinstance(value, (TokenIds, LogProbs)) else value
    return encoded


def segment_json(segment: Segment) -> Dict[str, Any]:
    """A segment line: its samples' lines, and its sequence."""
    return {"type": "segment", "lines": [segment.samples[0].line, segment.samples[-1].line],
            "samples": len(segment.samples), "token_ids": _values(segment.token_ids),
            "sampled": [list(span) for span in segment.sampled],
            "log_probs": _values(segment.log_probs)}


def _full_log(argument: str) -> Path:
    """The full log that SAVE names: a CLI save's, once it exists, else SAVE."""
    candidate = full_log_path(argument)
    return candidate if candidate.exists() else Path(argument)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m pythia.interaction.tokens",
        description="Write a save's token traces (endpoint.token_trace) as JSONL: a line per "
                    "record, and a line per segment, a run of samples that make one sequence.")
    parser.add_argument("save", metavar="SAVE",
                        help="a CLI save's prefix or PREFIX.jsonl, or a full log")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--records", action="store_true", help="write only the record lines")
    only.add_argument("--segments", action="store_true", help="write only the segment lines")
    args = parser.parse_args(argv)
    path = _full_log(args.save)
    try:
        context = load_interaction_save(path)
    except (OSError, SaveError) as exc:
        print(f"tokens: can't read {path}: {exc}", file=sys.stderr)
        return 1
    samples = [sample for sample in traced_samples(context.items)]
    out = sys.stdout
    # Each segment's line follows its last sample's record line.
    pending = list(segments(samples)) if not args.records else []
    ends = {segment.samples[-1].line: segment for segment in pending}
    for sample in samples:
        if sample.record is not None and not args.segments:
            out.write(json.dumps(record_json(sample), ensure_ascii=False) + "\n")
        if sample.line in ends:
            out.write(json.dumps(segment_json(ends[sample.line]), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
