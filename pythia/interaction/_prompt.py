"""Launch-time prompt arguments and one-time UTF-8 file loading."""

import argparse
import os
from pathlib import Path
import stat


def _prompt_file_argument(value):
    if not value.strip() or "\x00" in value or value == "-":
        raise argparse.ArgumentTypeError(
            "expected a non-empty filesystem path (--prompt-file does not read stdin)"
        )
    return Path(value)


def add_prompt_arguments(parser, *, allow_file=False, prompt_help=None):
    group = parser.add_mutually_exclusive_group() if allow_file else parser
    group.add_argument("--prompt", help=prompt_help)
    if allow_file:
        group.add_argument(
            "--prompt-file", type=_prompt_file_argument, metavar="PATH",
            help=("load the initial prompt once from a nonempty UTF-8 file; "
                  "strip trailing whitespace; "
                  "relative to the launch directory, not --cwd; mutually "
                  "exclusive with --prompt; '-' is not stdin"),
        )


def load_prompt(args):
    """Strip trailing whitespace from file input; preserve literal --prompt."""
    source = getattr(args, "prompt_file", None)
    if source is None:
        return args.prompt
    if args.prompt is not None:
        raise ValueError("--prompt and --prompt-file are mutually exclusive.")
    path = Path(source).expanduser().absolute()
    try:
        if not path.is_file():
            raise ValueError("--prompt-file must be a regular file.")
        # Nonblocking open + fstat also rejects a FIFO swapped in after path
        # validation, rather than hanging headless startup while reading it.
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as file:
            if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                raise ValueError("--prompt-file must be a regular file.")
            prompt = file.read().decode("utf-8").rstrip()
    except (OSError, UnicodeError):
        # Do not echo file contents or decoder diagnostics into startup output.
        raise ValueError(f"Could not read --prompt-file {str(path)!r} as UTF-8 text.") from None
    if not prompt.strip():
        raise ValueError("--prompt-file must contain a non-empty prompt.")
    return prompt
