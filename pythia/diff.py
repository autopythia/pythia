from typing import Optional, TypedDict

from pythia.types import Result, _py_version

class DiffHunk(TypedDict):
    src_line_start: int
    src_line_count: int
    src_lines: list[tuple[int, str]]
    dst_line_start: int
    dst_line_count: int
    dst_lines: list[tuple[int, str]]

class DiffFile(TypedDict):
    src: Optional[str]
    dst: Optional[str]
    hunks: list[DiffHunk]

class Diff(TypedDict):
    files: list[DiffFile]

if _py_version >= (3, 11):
    DiffResult = Result[Diff, dict]
else:
    DiffResult = Result

def parse_diff(haystack: str, path_root: Optional[str]) -> DiffResult:
    try:
        value = _parse_diff(haystack, path_root)
        result = {"ok": value}
    except Exception as e:
        result = {"err: {"exc_type": f"{type(e).__name__}", "exc_value": f"{e}"}}
    return result

def _parse_diff(haystack: str, path_root: Optional[str]) -> DiffResult:
    # INIT_STATE = 0
    # SRC_STATE = 1
    # DST_STATE = 2
    # HUNK_STATE = 3
    state = 0
    diff = {
        "files": [],
    }
    file = None
    hunk = None
    for line_idx, line in enumerate(haystack.splitlines()):
        if (
            (state == 0) and
            line.startswith("diff --git ")
        ):
            pass
        elif (
            (state == 0) and
            line.startswith("index ")
        ):
            pass
        elif (
            (state == 0) and
            line.startswith("--- ")
        ):
            if hunk is not None:
                file["hunks"].append(hunk)
                hunk = None
            if file is not None:
                diff["files"].append(file)
            file = {
                "src": None,
                "dst": None,
                "hunks": [],
            }
            raw_path = line[4:]
            if path_root is not None:
                path_parts = raw_path.split("/", maxsplit=1)
                fix_path = f"{path_root}/{path_parts[1]}"
            else:
                fix_path = raw_path
            file["src"] = fix_path
            state = 1
        elif (
            (state == 1) and
            line.startswith("+++ ")
        ):
            raw_path = line[4:]
            if path_root is not None:
                path_parts = raw_path.split("/", maxsplit=1)
                fix_path = f"{path_root}/{path_parts[1]}"
            else:
                fix_path = raw_path
            file["dst"] = fix_path
            state = 2
        elif (
            (state == 0 or state == 2) and
            line.startswith("@@ ")
        ):
            haystack = line[3:].split(" @@", maxsplit=1)[0]
            parts = haystack.split(maxsplit=1)
            src_parts = parts[0].split(",", maxsplit=1)
            dst_parts = parts[1].split(",", maxsplit=1)
            src_line_start = -int(src_parts[0])
            src_line_count = int(src_parts[1])
            dst_line_start = int(dst_parts[0])
            dst_line_count = int(dst_parts[1])
            if file is None:
                file = {
                    "src": None,
                    "dst": None,
                    "hunks": [],
                }
            if hunk is not None:
                file["hunks"].append(hunk)
            hunk = {
                "src_line_start": src_line_start,
                "src_line_count": src_line_count,
                "src_lines": [],
                "dst_line_start": dst_line_start,
                "dst_line_count": dst_line_count,
                "dst_lines": [],
            }
            state = 3
            if (
                len(hunk["src_lines"]) >= hunk["src_line_count"] and
                len(hunk["dst_lines"]) >= hunk["dst_line_count"]
            ):
                state = 0
        elif (
            (state == 3) and
            line.startswith(" ")
        ):
            hunk["src_lines"].append(line[1:])
            hunk["dst_lines"].append(line[1:])
            if (
                len(hunk["src_lines"]) >= hunk["src_line_count"] and
                len(hunk["dst_lines"]) >= hunk["dst_line_count"]
            ):
                state = 0
        elif (
            (state == 3) and
            line.startswith("-")
        ):
            hunk["src_lines"].append(line[1:])
            if (
                len(hunk["src_lines"]) >= hunk["src_line_count"] and
                len(hunk["dst_lines"]) >= hunk["dst_line_count"]
            ):
                state = 0
        elif (
            (state == 3) and
            line.startswith("+")
        ):
            hunk["dst_lines"].append(line[1:])
            if (
                len(hunk["src_lines"]) >= hunk["src_line_count"] and
                len(hunk["dst_lines"]) >= hunk["dst_line_count"]
            ):
                state = 0
        else:
            raise ValueError(f"parse_diff: line={line_idx+1} state={state}")
    if hunk is not None:
        file["hunks"].append(hunk)
        hunk = None
    if file is not None:
        diff["files"].append(file)
        file = None
    return diff

if __name__ == "__main__":
    with open("test_diff_1.diff", "r") as f:
        diff = parse_diff(f.read(), ".")
        print(diff)
