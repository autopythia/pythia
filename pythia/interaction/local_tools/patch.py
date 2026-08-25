from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ..environment import Tool
from ..environment import ToolOutcome
from ..environment import ToolSpec


@dataclass(frozen=True)
class _Hunk:
    rows: Tuple[Tuple[str, str], ...]
    end_of_file: bool = False


@dataclass(frozen=True)
class _PatchOperation:
    kind: str
    path: str
    move_to: Optional[str] = None
    content_lines: Tuple[str, ...] = ()
    hunks: Tuple[_Hunk, ...] = ()


@dataclass(frozen=True)
class _ResolvedOperation:
    kind: str
    path: str
    source: Path
    move_to: Optional[str] = None
    destination: Optional[Path] = None
    content_lines: Tuple[str, ...] = ()
    hunks: Tuple[_Hunk, ...] = ()


@dataclass(frozen=True)
class _OriginalFile:
    exists: bool
    content: bytes = b""
    mode: Optional[int] = None


_FILE_DIRECTIVES = (
    "*** Add File: ",
    "*** Delete File: ",
    "*** Update File: ",
)

_UNIFIED_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?:.*)$"
)


def _is_file_directive(line: str) -> bool:
    return line.startswith(_FILE_DIRECTIVES)


def _parse_hunks(lines: List[str], path: str) -> Tuple[_Hunk, ...]:
    hunks = []
    index = 0
    while index < len(lines):
        while index < len(lines) and not lines[index]:
            index += 1
        if index >= len(lines):
            break
        header = lines[index]
        if not header.startswith("@@"):
            raise ValueError(f"invalid update hunk header for {path}: {header}")
        index += 1

        rows = []
        end_of_file = False
        while index < len(lines) and not lines[index].startswith("@@"):
            row = lines[index]
            if row == "*** End of File":
                end_of_file = True
                index += 1
                continue
            if row == "":
                rows.append((" ", ""))
                index += 1
                continue
            prefix = row[0]
            if prefix not in {" ", "+", "-"}:
                raise ValueError(f"invalid hunk line for {path}: {row!r}")
            rows.append((prefix, row[1:]))
            index += 1

        if not rows and not end_of_file:
            raise ValueError(f"empty update hunk for {path}")
        hunks.append(_Hunk(rows=tuple(rows), end_of_file=end_of_file))
    return tuple(hunks)


def _parse_v4a_patch(patch_text: str) -> Tuple[_PatchOperation, ...]:
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise ValueError("apply_patch requires non-empty patch text")
    lines = patch_text.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("apply_patch input must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("apply_patch input must end with *** End Patch")

    operations = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if not line:
            index += 1
            continue

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: ") :].strip()
            index += 1
            content_lines = []
            while index < len(lines) - 1 and not _is_file_directive(lines[index]):
                row = lines[index]
                if row.startswith("@@"):
                    index += 1
                    continue
                if not row.startswith("+"):
                    raise ValueError("add file patch lines must start with '+'")
                content_lines.append(row[1:])
                index += 1
            if not content_lines:
                raise ValueError(f"add file patch has no content lines: {path}")
            operations.append(
                _PatchOperation(
                    kind="add",
                    path=path,
                    content_lines=tuple(content_lines),
                )
            )
            continue

        if line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: ") :].strip()
            operations.append(_PatchOperation(kind="delete", path=path))
            index += 1
            continue

        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: ") :].strip()
            index += 1
            move_to = None
            if (
                index < len(lines) - 1
                and lines[index].startswith("*** Move to: ")
            ):
                move_to = lines[index][len("*** Move to: ") :].strip()
                index += 1

            hunk_lines = []
            while index < len(lines) - 1 and not _is_file_directive(lines[index]):
                hunk_lines.append(lines[index])
                index += 1
            hunks = _parse_hunks(hunk_lines, path) if hunk_lines else ()
            if not hunks and move_to is None:
                raise ValueError(f"update file patch has no hunks: {path}")
            operations.append(
                _PatchOperation(
                    kind="update",
                    path=path,
                    move_to=move_to,
                    hunks=hunks,
                )
            )
            continue

        raise ValueError(f"unsupported patch directive: {line}")

    if not operations:
        raise ValueError("apply_patch contains no file operations")
    return tuple(operations)


def _parse_unified_path(header_value: str) -> str:
    value = header_value.split("\t", 1)[0].strip()
    if not value:
        raise ValueError("unified diff file path must not be empty")
    if value.startswith('"'):
        raise ValueError("quoted unified diff paths are not supported")
    if value == "/dev/null":
        return value
    for prefix in ("a/", "b/"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _parse_unified_metadata_path(header_value: str) -> str:
    value = header_value.strip()
    if not value:
        raise ValueError("unified diff rename path must not be empty")
    if value.startswith('"'):
        raise ValueError("quoted unified diff paths are not supported")
    return value


def _parse_unified_hunk(
    lines: List[str],
    index: int,
    path: str,
) -> Tuple[_Hunk, int]:
    header = lines[index]
    match = _UNIFIED_HUNK_RE.match(header)
    if match is None:
        raise ValueError(f"invalid unified diff hunk header for {path}: {header}")
    old_count = int(match.group("old_count") or "1")
    new_count = int(match.group("new_count") or "1")
    index += 1

    old_seen = 0
    new_seen = 0
    rows = []
    while old_seen < old_count or new_seen < new_count:
        if index >= len(lines):
            raise ValueError(f"incomplete unified diff hunk for {path}")
        row = lines[index]
        if row == "\\ No newline at end of file":
            index += 1
            continue
        if row.startswith("@@"):
            raise ValueError(f"incomplete unified diff hunk for {path}")
        if row == "":
            prefix, text = " ", ""
        else:
            prefix, text = row[0], row[1:]
        if prefix == " ":
            old_seen += 1
            new_seen += 1
        elif prefix == "-":
            old_seen += 1
        elif prefix == "+":
            new_seen += 1
        else:
            raise ValueError(f"invalid unified diff line for {path}: {row!r}")
        if old_seen > old_count or new_seen > new_count:
            raise ValueError(f"unified diff hunk line count mismatch for {path}")
        rows.append((prefix, text))
        index += 1

    while (
        index < len(lines)
        and lines[index] == "\\ No newline at end of file"
    ):
        index += 1
    if not rows and old_count == 0 and new_count == 0:
        raise ValueError(f"empty unified diff hunk for {path}")
    return _Hunk(rows=tuple(rows)), index


def _parse_unified_patch(patch_text: str) -> Tuple[_PatchOperation, ...]:
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise ValueError("apply_patch requires non-empty patch text")
    lines = patch_text.splitlines()
    if any(
        line.startswith("GIT binary patch")
        or line.startswith("Binary files ")
        for line in lines
    ):
        raise ValueError("apply_patch does not support binary diffs")

    operations = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("diff --git "):
            block_end = index + 1
            while (
                block_end < len(lines)
                and not lines[block_end].startswith("diff --git ")
            ):
                block_end += 1
            has_file_headers = any(
                lines[position].startswith("--- ")
                and position + 1 < block_end
                and lines[position + 1].startswith("+++ ")
                for position in range(index + 1, block_end)
            )
            if not has_file_headers:
                rename_from = None
                rename_to = None
                for metadata_line in lines[index + 1 : block_end]:
                    if metadata_line.startswith("rename from "):
                        rename_from = _parse_unified_metadata_path(
                            metadata_line[len("rename from ") :]
                        )
                    elif metadata_line.startswith("rename to "):
                        rename_to = _parse_unified_metadata_path(
                            metadata_line[len("rename to ") :]
                        )
                if rename_from is not None or rename_to is not None:
                    if rename_from is None or rename_to is None:
                        raise ValueError(
                            "unified diff rename requires both rename from "
                            "and rename to"
                        )
                    operations.append(
                        _PatchOperation(
                            kind="update",
                            path=rename_from,
                            move_to=rename_to,
                        )
                    )
                index = block_end
                continue

        if not lines[index].startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            index += 1
            continue

        source_path = _parse_unified_path(lines[index][4:])
        destination_path = _parse_unified_path(lines[index + 1][4:])
        display_path = (
            destination_path
            if destination_path != "/dev/null"
            else source_path
        )
        index += 2
        hunks = []

        while index < len(lines):
            if lines[index].startswith("diff --git "):
                break
            if (
                lines[index].startswith("--- ")
                and index + 1 < len(lines)
                and lines[index + 1].startswith("+++ ")
            ):
                break
            if lines[index].startswith("@@"):
                hunk, index = _parse_unified_hunk(
                    lines,
                    index,
                    display_path,
                )
                hunks.append(hunk)
                continue
            index += 1

        if source_path == "/dev/null":
            content_lines = tuple(
                text
                for hunk in hunks
                for prefix, text in hunk.rows
                if prefix in {" ", "+"}
            )
            operations.append(
                _PatchOperation(
                    kind="add",
                    path=destination_path,
                    content_lines=content_lines,
                )
            )
            continue

        if destination_path == "/dev/null":
            operations.append(
                _PatchOperation(
                    kind="delete",
                    path=source_path,
                )
            )
            continue

        move_to = (
            destination_path
            if destination_path != source_path
            else None
        )
        if not hunks and move_to is None:
            raise ValueError(
                f"unified diff update has no hunks: {source_path}"
            )
        operations.append(
            _PatchOperation(
                kind="update",
                path=source_path,
                move_to=move_to,
                hunks=tuple(hunks),
            )
        )

    if not operations:
        raise ValueError("apply_patch contains no unified diff file operations")
    return tuple(operations)


def _parse_patch(patch_text: str) -> Tuple[_PatchOperation, ...]:
    lines = patch_text.splitlines()
    first_line = lines[0].strip() if lines else ""
    if first_line == "*** Begin Patch":
        return _parse_v4a_patch(patch_text)
    return _parse_unified_patch(patch_text)


def _resolve_workspace_path(root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("apply_patch path must not be empty")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError("apply_patch paths must be relative")
    if ".." in relative.parts:
        raise ValueError(f"apply_patch path escapes workspace: {raw_path}")

    candidate = root.joinpath(relative)
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"apply_patch path escapes workspace: {raw_path}")
    return candidate


def _resolve_operations(
    root: Path,
    operations: Tuple[_PatchOperation, ...],
) -> Tuple[_ResolvedOperation, ...]:
    resolved = []
    for operation in operations:
        source = _resolve_workspace_path(root, operation.path)
        destination = None
        if operation.move_to is not None:
            destination = _resolve_workspace_path(root, operation.move_to)
        resolved.append(
            _ResolvedOperation(
                kind=operation.kind,
                path=operation.path,
                source=source,
                move_to=operation.move_to,
                destination=destination,
                content_lines=operation.content_lines,
                hunks=operation.hunks,
            )
        )
    return tuple(resolved)


def _find_subsequence(
    haystack: List[str],
    needle: List[str],
    *,
    start: int,
    end_of_file: bool,
) -> Optional[int]:
    if end_of_file:
        position = len(haystack) - len(needle)
        if position < 0:
            return None
        if haystack[position:] == needle:
            return position
        return None
    if not needle:
        return min(start, len(haystack))
    max_start = len(haystack) - len(needle)
    for position in range(max(0, start), max_start + 1):
        if haystack[position : position + len(needle)] == needle:
            return position
    return None


def _apply_hunks(raw: str, hunks: Tuple[_Hunk, ...], path: str) -> str:
    had_trailing_newline = raw.endswith("\n")
    current_lines = raw.splitlines()
    cursor = 0

    for hunk in hunks:
        old_block = [
            text
            for prefix, text in hunk.rows
            if prefix in {" ", "-"}
        ]
        new_block = [
            text
            for prefix, text in hunk.rows
            if prefix in {" ", "+"}
        ]
        start = _find_subsequence(
            current_lines,
            old_block,
            start=cursor,
            end_of_file=hunk.end_of_file,
        )
        if start is None:
            raise ValueError(f"unable to apply hunk while updating {path}")
        end = start + len(old_block)
        current_lines = [
            *current_lines[:start],
            *new_block,
            *current_lines[end:],
        ]
        cursor = start + len(new_block)

    rendered = "\n".join(current_lines)
    if had_trailing_newline and current_lines:
        rendered = f"{rendered}\n"
    return rendered


def _read_virtual_file(
    path: Path,
    staged: Dict[Path, Optional[str]],
    display_path: str,
) -> str:
    if path in staged:
        content = staged[path]
        if content is None:
            raise ValueError(f"cannot access deleted file: {display_path}")
        return content
    if path.is_symlink():
        raise ValueError(f"apply_patch does not modify symlink files: {display_path}")
    if not path.exists():
        raise ValueError(f"cannot access missing file: {display_path}")
    if path.is_dir():
        raise ValueError(f"cannot modify directory with apply_patch: {display_path}")
    return path.read_text(encoding="utf-8")


def _stage_operations(
    operations: Tuple[_ResolvedOperation, ...],
) -> Tuple[Dict[Path, Optional[str]], Tuple[str, ...]]:
    staged: Dict[Path, Optional[str]] = {}
    changes = []

    for operation in operations:
        source = operation.source
        if operation.kind == "add":
            if source in staged:
                if staged[source] is not None:
                    raise ValueError(f"cannot add existing file: {operation.path}")
            elif source.exists() or source.is_symlink():
                raise ValueError(f"cannot add existing file: {operation.path}")
            body = "\n".join(operation.content_lines)
            if operation.content_lines:
                body = f"{body}\n"
            staged[source] = body
            changes.append(f"A {operation.path}")
            continue

        if operation.kind == "delete":
            _read_virtual_file(source, staged, operation.path)
            staged[source] = None
            changes.append(f"D {operation.path}")
            continue

        if operation.kind == "update":
            raw = _read_virtual_file(source, staged, operation.path)
            rendered = _apply_hunks(raw, operation.hunks, operation.path)
            if operation.destination is None:
                staged[source] = rendered
                changes.append(f"M {operation.path}")
                continue

            destination = operation.destination
            assert operation.move_to is not None
            if destination != source:
                if destination in staged:
                    destination_exists = staged[destination] is not None
                else:
                    destination_exists = (
                        destination.exists() or destination.is_symlink()
                    )
                if destination_exists:
                    raise ValueError(
                        f"cannot move to existing file: {operation.move_to}"
                    )
                staged[destination] = rendered
                staged[source] = None
            else:
                staged[source] = rendered
            changes.append(f"R {operation.path} -> {operation.move_to}")
            continue

        raise ValueError(f"unsupported patch operation: {operation.kind}")

    return staged, tuple(changes)


def _snapshot_originals(
    staged: Mapping[Path, Optional[str]],
) -> Dict[Path, _OriginalFile]:
    originals = {}
    for path in staged:
        if path.is_symlink():
            raise ValueError(f"apply_patch does not modify symlink files: {path}")
        if not path.exists():
            originals[path] = _OriginalFile(exists=False)
            continue
        if path.is_dir():
            raise ValueError(f"cannot modify directory with apply_patch: {path}")
        mode = path.stat().st_mode & 0o7777
        originals[path] = _OriginalFile(
            exists=True,
            content=path.read_bytes(),
            mode=mode,
        )
    return originals


def _restore_originals(originals: Mapping[Path, _OriginalFile]) -> None:
    for path, original in reversed(tuple(originals.items())):
        try:
            if original.exists:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original.content)
                if original.mode is not None:
                    os.chmod(path, original.mode)
                continue
            if path.exists() or path.is_symlink():
                if not path.is_dir():
                    path.unlink()
        except OSError:
            pass


def _commit_staged(
    staged: Mapping[Path, Optional[str]],
    originals: Mapping[Path, _OriginalFile],
) -> None:
    try:
        for path, content in staged.items():
            if content is None:
                if path.exists() or path.is_symlink():
                    if path.is_dir():
                        raise ValueError(
                            f"cannot delete directory with apply_patch: {path}"
                        )
                    path.unlink()
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            original = originals[path]
            if original.exists and original.mode is not None:
                os.chmod(path, original.mode)
    except Exception:
        _restore_originals(originals)
        raise


def _apply_patch_text(workspace_root: Path, patch_text: str) -> str:
    operations = _parse_patch(patch_text)
    resolved = _resolve_operations(workspace_root, operations)
    staged, changes = _stage_operations(resolved)
    originals = _snapshot_originals(staged)
    _commit_staged(staged, originals)
    return "\n".join(changes)


def create_apply_patch_tool(
    workspace_root: Union[str, Path],
    *,
    timeout_seconds: Optional[float] = None,
) -> Tool:
    root = Path(workspace_root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"workspace_root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"workspace_root is not a directory: {root}")
    apply_lock = Lock()

    def apply_patch(
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        del timeout_seconds
        if not isinstance(arguments, Mapping):
            raise TypeError("apply_patch arguments must be a mapping")
        patch_text = arguments.get("patch")
        if not isinstance(patch_text, str) or not patch_text.strip():
            raise ValueError("apply_patch requires a non-empty patch string")
        with apply_lock:
            return ToolOutcome(output=_apply_patch_text(root, patch_text))

    return Tool(
        spec=ToolSpec(
            name="apply_patch",
            description=(
                "Apply file edits in the configured workspace. The patch may "
                "use either the V4A (*** Begin Patch) format or a standard "
                "unified diff produced by diff -u or git diff."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": (
                            "Patch text in either V4A (*** Begin Patch) or "
                            "standard unified-diff format."
                        ),
                    },
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        ),
        handler=apply_patch,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "create_apply_patch_tool",
]
