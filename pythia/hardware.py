"""Small, dependency-free hardware queries used by local tooling.

The operating system can usually tell us how many CPUs it presents, but a
guest cannot in general determine how those CPUs map to physical host cores.
Consequently ``PerformanceLevel.core_count`` means a useful unit of local
parallelism: physical cores when that information is trustworthy, and visible
vCPUs when virtualization is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PerformanceLevel:
    label: str
    core_count: int

    def __post_init__(self):
        if not self.label or self.core_count < 1:
            raise ValueError("A performance level needs a label and at least one core.")


@dataclass(frozen=True)
class CpuTopology:
    levels: tuple[PerformanceLevel, ...]
    # Keep metadata grouped and named like its source rather than imposing a
    # cross-platform virtualization schema.  Absence is not proof of bare metal.
    virtualization: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.levels:
            raise ValueError("CPU topology must contain at least one performance level.")

    @property
    def core_count(self) -> int:
        return sum(level.core_count for level in self.levels)


def _command_output(argv: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, ValueError):
        return None
    return result.stdout if result.returncode == 0 else None


def _parse_key_values(text: str) -> dict[str, str]:
    values = {}
    for raw_line in text.splitlines():
        key, separator, value = raw_line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _performance_label(name: str, counters: dict[str, int]) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("performance") or normalized in {"p", "big"}:
        prefix = "P"
    elif normalized.startswith("efficiency") or normalized in {"e", "little"}:
        prefix = "E"
    else:
        # Unknown sysctl names remain useful levels, but do not guess that they
        # are efficiency cores.  P is also the homogeneous-machine fallback.
        prefix = "P"
    ordinal = counters.get(prefix, 0)
    counters[prefix] = ordinal + 1
    return f"{prefix}{ordinal}"


def _parse_sysctl_topology(text: str) -> Optional[CpuTopology]:
    values = _parse_key_values(text)
    try:
        count = int(values["hw.nperflevels"])
    except (KeyError, TypeError, ValueError):
        count = 0

    levels = []
    counters: dict[str, int] = {}
    for index in range(count):
        prefix = f"hw.perflevel{index}"
        try:
            core_count = int(values[f"{prefix}.physicalcpu"])
        except (KeyError, TypeError, ValueError):
            return None
        if core_count < 1:
            return None
        name = values.get(f"{prefix}.name", "Performance")
        levels.append(PerformanceLevel(_performance_label(name, counters), core_count))

    if levels:
        return CpuTopology(tuple(levels))

    try:
        physical = int(values["hw.physicalcpu"])
    except (KeyError, TypeError, ValueError):
        return None
    if physical < 1:
        return None
    return CpuTopology((PerformanceLevel("P0", physical),))


@dataclass(frozen=True)
class _LscpuEntry:
    cpu: int
    core: int
    socket: int


def _parse_lscpu_entries(text: str) -> tuple[_LscpuEntry, ...]:
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3:
            raise ValueError("Expected CPU, core and socket columns from lscpu.")
        try:
            cpu, core, socket = (int(fields[i]) for i in range(3))
        except ValueError as exc:
            raise ValueError("Invalid numeric lscpu topology field.") from exc
        if min(cpu, core, socket) < 0:
            raise ValueError("Negative lscpu topology field.")
        entries.append(_LscpuEntry(cpu, core, socket))
    return tuple(entries)


def _parse_lscpu_virtualization(
    summary: str,
    *,
    dmi: Optional[Mapping[str, str]] = None,
) -> dict[str, object]:
    values = _parse_key_values(summary)
    lscpu = {}
    for key in ("Hypervisor vendor", "Virtualization type"):
        value = values.get(key)
        if value:
            lscpu[key] = value
    if not lscpu:
        return {}
    metadata = {"lscpu": lscpu}
    dmi_values = {key: value for key, value in (dmi or {}).items() if value}
    if dmi_values:
        metadata["dmi"] = dmi_values
    return metadata


def _available_cpus(entries: Sequence[_LscpuEntry]) -> set[int]:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    cpus = {entry.cpu for entry in entries}
    if affinity is not None:
        cpus.intersection_update(affinity)
    return cpus


def _read_linux_level(cpu: int, root: Path = Path("/sys/devices/system/cpu")) -> tuple[str, Optional[int]]:
    directory = root / f"cpu{cpu}"
    try:
        core_type = (directory / "topology" / "core_type").read_text().strip()
    except OSError:
        core_type = ""
    # Linux uses 1 for Intel Atom and 2 for Intel Core.  Other architectures
    # commonly expose capacity without core_type.
    kind = "E" if core_type == "1" else "P"
    capacity = None
    for path in (directory / "cpu_capacity", directory / "cpufreq" / "cpuinfo_max_freq"):
        try:
            capacity = int(path.read_text().strip())
            break
        except (OSError, ValueError):
            pass
    return kind, capacity


def _linux_levels(
    entries: Sequence[_LscpuEntry],
    available: set[int],
    *,
    level_reader: Callable[[int], tuple[str, Optional[int]]] = _read_linux_level,
) -> tuple[PerformanceLevel, ...]:
    # Choose one representative logical CPU for each physical core.
    cores: dict[tuple[int, int], list[int]] = {}
    for entry in entries:
        if entry.cpu in available:
            cores.setdefault((entry.socket, entry.core), []).append(entry.cpu)
    if not cores:
        return ()

    groups: dict[tuple[str, Optional[int]], int] = {}
    for cpus in cores.values():
        try:
            kind, capacity = level_reader(min(cpus))
        except (OSError, ValueError):
            kind, capacity = "P", None
        kind = "E" if str(kind).upper().startswith("E") else "P"
        groups[(kind, capacity)] = groups.get((kind, capacity), 0) + 1

    # With no level information this is the normal homogeneous P0 case.
    def group_key(item):
        (kind, capacity), _count = item
        return (0 if kind == "P" else 1, -(capacity if capacity is not None else -1))

    counters = {"P": 0, "E": 0}
    levels = []
    for (kind, _capacity), core_count in sorted(groups.items(), key=group_key):
        label = f"{kind}{counters[kind]}"
        counters[kind] += 1
        levels.append(PerformanceLevel(label, core_count))
    return tuple(levels)


def _dmi_metadata(root: Path = Path("/sys/class/dmi/id")) -> dict[str, str]:
    values = {}
    for name in ("sys_vendor", "product_name", "product_version", "board_vendor"):
        try:
            value = (root / name).read_text().strip()
        except OSError:
            continue
        if value:
            values[name] = value
    return values


def _fallback_count() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = 0
    if count < 1:
        try:
            count = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        except (AttributeError, OSError, TypeError, ValueError):
            count = 0
    if count < 1:
        count = os.cpu_count() or 1
    return max(1, count)


def _query_macos(command: Callable[[Sequence[str]], Optional[str]]) -> Optional[CpuTopology]:
    text = command(("sysctl", "-a"))
    return None if text is None else _parse_sysctl_topology(text)


def _query_linux(command: Callable[[Sequence[str]], Optional[str]]) -> Optional[CpuTopology]:
    entries_text = command(("lscpu", "-p=CPU,CORE,SOCKET,NODE"))
    summary = command(("lscpu",)) or ""
    try:
        entries = _parse_lscpu_entries(entries_text or "")
    except ValueError:
        entries = ()
    available = _available_cpus(entries)
    virtualization = _parse_lscpu_virtualization(summary, dmi=_dmi_metadata())

    if virtualization:
        # A guest-visible "core" is hypervisor topology, not evidence about a
        # host core.  Every affinity-visible vCPU is nevertheless schedulable.
        count = len(available) or _fallback_count()
        return CpuTopology((PerformanceLevel("P0", count),), virtualization)

    levels = _linux_levels(entries, available)
    if levels:
        return CpuTopology(levels)
    return None


def cpu_topology(
    *,
    system: Optional[str] = None,
    command: Callable[[Sequence[str]], Optional[str]] = _command_output,
) -> CpuTopology:
    """Return the locally usable CPU levels without claiming unknowable host topology."""
    system = platform.system() if system is None else system
    topology = None
    if system == "Darwin":
        topology = _query_macos(command)
    elif system == "Linux":
        topology = _query_linux(command)
    if topology is not None:
        return topology
    return CpuTopology((PerformanceLevel("P0", _fallback_count()),))


def performance_levels() -> tuple[PerformanceLevel, ...]:
    return cpu_topology().levels

