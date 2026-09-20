"""TAP frontend and process-isolated parallel runner for unittest suites.

The test definitions remain ordinary ``unittest`` tests.  This module supplies
the frontend: it runs each selected module in a clean interpreter, combines the
results, and emits one deterministic TAP test point per unittest test method.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Mapping, Optional, Sequence, TextIO
import unittest

from .hardware import CpuTopology, cpu_topology


_RESULT_VERSION = 1
_DIAGNOSTIC_LIMIT = 256 * 1024


@dataclass(frozen=True)
class TestRecord:
    test_id: str
    status: str
    diagnostic: str = ""
    directive: str = ""
    duration_seconds: float = 0.0


@dataclass
class _PendingTest:
    test_id: str
    started: float
    status: Optional[str] = None
    diagnostic: str = ""
    directive: str = ""


class TapTestResult(unittest.TestResult):
    """Collect unittest callbacks as transportable, one-method records."""

    def __init__(self):
        super().__init__()
        self.records: list[TestRecord] = []
        self._pending: dict[int, _PendingTest] = {}

    def startTest(self, test):
        super().startTest(test)
        self._pending[id(test)] = _PendingTest(test.id(), time.monotonic())

    def stopTest(self, test):
        pending = self._pending.pop(id(test), None)
        if pending is None:
            pending = _PendingTest(test.id(), time.monotonic(), "error",
                                   "Test stopped without a matching start callback.")
        status = pending.status or "success"
        self.records.append(TestRecord(
            pending.test_id,
            status,
            pending.diagnostic,
            pending.directive,
            max(0.0, time.monotonic() - pending.started),
        ))
        super().stopTest(test)

    def _set(self, test, status: str, diagnostic: str = "", directive: str = ""):
        pending = self._pending.get(id(test))
        if pending is None:
            # Class/module fixture errors are represented by unittest's
            # _ErrorHolder and do not always have normal start/stop callbacks.
            self.records.append(TestRecord(test.id(), status, diagnostic, directive, 0.0))
            return
        if pending.status in {"error", "failure", "unexpected_success"}:
            if diagnostic:
                pending.diagnostic = _join_diagnostics(pending.diagnostic, diagnostic)
            return
        pending.status = status
        pending.diagnostic = diagnostic
        pending.directive = directive

    def addSuccess(self, test):
        super().addSuccess(test)
        self._set(test, "success")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._set(test, "failure", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._set(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._set(test, "skip", directive=_single_line(reason))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._set(test, "expected_failure", self._exc_info_to_string(err, test),
                  "expected failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._set(test, "unexpected_success", "Test was expected to fail but succeeded.")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is None:
            return
        try:
            is_failure = issubclass(err[0], test.failureException)
        except TypeError:
            is_failure = False
        status = "failure" if is_failure else "error"
        diagnostic = f"Subtest: {subtest.id()}\n{self._exc_info_to_string(err, subtest)}"
        pending = self._pending.get(id(test))
        if pending is None:
            self.records.append(TestRecord(test.id(), status, diagnostic))
            return
        if pending.status == "error":
            pending.diagnostic = _join_diagnostics(pending.diagnostic, diagnostic)
            return
        if status == "error" or pending.status is None:
            pending.status = status
        pending.diagnostic = _join_diagnostics(pending.diagnostic, diagnostic)


def _join_diagnostics(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return left + "\n\n" + right


def _single_line(value) -> str:
    return " ".join(str(value).splitlines()).strip()


def _synthetic_failure(test_id: str, message: str) -> TestRecord:
    return TestRecord(test_id, "error", message)


def _write_worker_result(path: Path, records: Sequence[TestRecord]):
    payload = {
        "version": _RESULT_VERSION,
        "records": [asdict(record) for record in records],
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_worker(selectors: Sequence[str], result_path: Path) -> int:
    records: Sequence[TestRecord]
    try:
        suite = unittest.defaultTestLoader.loadTestsFromNames(list(selectors))
        result = TapTestResult()
        suite.run(result)
        records = result.records
        if not records:
            records = (_synthetic_failure(
                ".".join(selectors) or "unittest.discovery",
                "No tests were discovered for this worker.",
            ),)
    except BaseException as exc:
        records = (_synthetic_failure(
            ".".join(selectors) or "unittest.worker",
            "Worker infrastructure failed.\n" + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)),
        ),)
    try:
        _write_worker_result(result_path, records)
    except BaseException:
        traceback.print_exc()
        return 2
    return 0


@dataclass(frozen=True)
class _Job:
    order: int
    selectors: tuple[str, ...]

    @property
    def name(self) -> str:
        return ",".join(self.selectors)


@dataclass(frozen=True)
class _WorkerSlot:
    level: str
    index: int

    @property
    def name(self) -> str:
        return f"{self.level}/{self.index}"


@dataclass(frozen=True)
class _JobResult:
    job: _Job
    slot: _WorkerSlot
    records: tuple[TestRecord, ...]
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0


def _load_worker_result(path: Path) -> tuple[TestRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != _RESULT_VERSION or not isinstance(payload.get("records"), list):
        raise ValueError("Invalid worker result document.")
    records = []
    fields = {"test_id", "status", "diagnostic", "directive", "duration_seconds"}
    for item in payload["records"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Invalid worker test record.")
        record = TestRecord(**item)
        if record.status not in {
            "success", "skip", "expected_failure", "failure", "error", "unexpected_success",
        }:
            raise ValueError("Unknown worker test status.")
        records.append(record)
    if not records:
        raise ValueError("Worker returned no test records.")
    return tuple(records)


def _bounded_output(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    omitted = max(0, len(data) - _DIAGNOSTIC_LIMIT)
    if omitted:
        data = data[-_DIAGNOSTIC_LIMIT:]
    text = data.decode("utf-8", errors="replace")
    if omitted:
        text = f"[{omitted} earlier bytes omitted]\n" + text
    return text


class _ProcessRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._terminating = False

    def add(self, process: subprocess.Popen):
        with self._lock:
            terminate = self._terminating
            if not terminate:
                self._processes.add(process)
        if terminate:
            _terminate_process(process)

    def discard(self, process: subprocess.Popen):
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self):
        with self._lock:
            self._terminating = True
            processes = tuple(self._processes)
        for process in processes:
            _terminate_process(process)


def _terminate_process(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _execute_job(
    job: _Job,
    slot: _WorkerSlot,
    directory: Path,
    registry: _ProcessRegistry,
) -> _JobResult:
    stem = f"job-{job.order}"
    result_path = directory / f"{stem}.json"
    stdout_path = directory / f"{stem}.stdout"
    stderr_path = directory / f"{stem}.stderr"
    argv = [
        sys.executable, "-m", "pythia.testing", "--worker",
        "--result", str(result_path), *job.selectors,
    ]
    started = time.monotonic()
    return_code = None
    launch_error = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            registry.add(process)
            try:
                return_code = process.wait()
            finally:
                registry.discard(process)
        except BaseException as exc:
            launch_error = exc
    elapsed = time.monotonic() - started
    stdout_text, stderr_text = _bounded_output(stdout_path), _bounded_output(stderr_path)
    try:
        if launch_error is not None:
            raise launch_error
        if return_code != 0:
            raise RuntimeError(f"Worker exited with status {return_code}.")
        records = _load_worker_result(result_path)
    except BaseException as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        diagnostic = f"Worker {slot.name} failed for {job.name}: {detail}"
        if stderr_text:
            diagnostic += "\n\nWorker stderr:\n" + stderr_text
        if stdout_text:
            diagnostic += "\n\nWorker stdout:\n" + stdout_text
        records = (_synthetic_failure(f"{job.name}.__worker__", diagnostic),)
        stdout_text = stderr_text = ""
    return _JobResult(job, slot, records, stdout_text, stderr_text, elapsed)


def _worker_slots(topology: CpuTopology, jobs: Optional[int] = None) -> tuple[_WorkerSlot, ...]:
    slots = tuple(
        _WorkerSlot(level.label, index)
        for level in topology.levels
        for index in range(level.core_count)
    )
    if jobs is None:
        return slots
    if jobs < 1:
        raise ValueError("jobs must be at least one")
    if jobs <= len(slots):
        return slots[:jobs]
    return slots + tuple(_WorkerSlot("J", index) for index in range(jobs - len(slots)))


def _run_jobs(
    jobs: Sequence[_Job],
    slots: Sequence[_WorkerSlot],
    *,
    execute: Callable[[_Job, _WorkerSlot, Path, _ProcessRegistry], _JobResult] = _execute_job,
) -> tuple[_JobResult, ...]:
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    results = []
    result_lock = threading.Lock()
    registry = _ProcessRegistry()
    cancelled = threading.Event()

    with tempfile.TemporaryDirectory(prefix="pythia-tests-") as temporary:
        directory = Path(temporary)

        def loop(slot):
            while not cancelled.is_set():
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    try:
                        result = execute(job, slot, directory, registry)
                    except BaseException as exc:
                        diagnostic = "Worker slot failed.\n" + "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__))
                        result = _JobResult(
                            job, slot,
                            (_synthetic_failure(f"{job.name}.__worker__", diagnostic),),
                        )
                    with result_lock:
                        results.append(result)
                finally:
                    work.task_done()

        threads = [threading.Thread(target=loop, args=(slot,), name=f"test-{slot.name}")
                   for slot in slots[:max(1, min(len(slots), len(jobs)))]]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                while thread.is_alive():
                    thread.join(0.1)
        except BaseException:
            cancelled.set()
            registry.terminate_all()
            for thread in threads:
                thread.join()
            raise
    return tuple(sorted(results, key=lambda result: result.job.order))


def _discover_modules(package: str) -> tuple[str, ...]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        raise ValueError(f"Test package not found: {package}")
    modules = set()
    for location in spec.submodule_search_locations:
        root = Path(location)
        for path in root.rglob("test_*.py"):
            relative = path.relative_to(root).with_suffix("")
            if all(part.isidentifier() for part in relative.parts):
                modules.add(".".join((package, *relative.parts)))
    return tuple(sorted(modules))


def _format_virtualization(metadata: Mapping[str, object]) -> str:
    fields = []

    def append(prefix: str, value):
        if isinstance(value, Mapping):
            for key, nested in value.items():
                append(f"{prefix}.{_single_line(key)}" if prefix else _single_line(key), nested)
        else:
            fields.append(f"{prefix}={_single_line(value)}")

    for source, value in metadata.items():
        append(_single_line(source), value)
    return ", ".join(fields)


def _tap_name(value: str) -> str:
    return _single_line(value).replace(" # ", " ")


def _write_comments(stream: TextIO, heading: str, text: str):
    if not text:
        return
    print(f"# {heading}", file=stream)
    for line in text.splitlines() or ("",):
        print(f"# {line}", file=stream)


def _write_tap_header(topology: CpuTopology, slot_count: int, stream: TextIO):
    print("TAP version 13", file=stream)
    levels = ", ".join(f"{level.label}={level.core_count}" for level in topology.levels)
    print(f"# workers: {slot_count}; levels: {levels}; isolation: module", file=stream)
    virtualization = _format_virtualization(topology.virtualization)
    if virtualization:
        print(f"# virtualization: {virtualization}", file=stream)
    stream.flush()


def _render_tap(
    results: Sequence[_JobResult],
    topology: CpuTopology,
    slot_count: int,
    stream: TextIO,
    *,
    write_header: bool = True,
) -> int:
    if write_header:
        _write_tap_header(topology, slot_count, stream)

    records = []
    seen = Counter()
    for result in results:
        for record in result.records:
            seen[record.test_id] += 1
            if seen[record.test_id] > 1:
                records.append(_synthetic_failure(
                    record.test_id + ".__duplicate__",
                    f"Duplicate test ID returned by workers: {record.test_id}",
                ))
            else:
                records.append(record)
        _write_comments(stream, f"stdout from {result.job.name}", result.stdout)
        _write_comments(stream, f"stderr from {result.job.name}", result.stderr)

    failed = False
    for number, record in enumerate(records, start=1):
        name = _tap_name(record.test_id)
        if record.status in {"success", "skip"}:
            line = f"ok {number} - {name}"
            if record.status == "skip":
                line += f" # SKIP {_tap_name(record.directive)}"
        elif record.status == "expected_failure":
            line = f"not ok {number} - {name} # TODO expected failure"
        else:
            line = f"not ok {number} - {name}"
            failed = True
        print(line, file=stream)
        if record.status not in {"success", "skip"} and record.diagnostic:
            for diagnostic_line in record.diagnostic.splitlines():
                print(f"# {diagnostic_line}", file=stream)
    print(f"1..{len(records)}", file=stream)
    stream.flush()
    return 1 if failed else 0


def _build_parser(package: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python3 -m {package}",
        description="Run unittest modules through a process-isolated TAP frontend.",
    )
    parser.add_argument("selectors", nargs="*", help="test modules or unittest test IDs")
    parser.add_argument(
        "-j", "--jobs", type=int,
        help="worker count (default: all queried cores across performance levels)",
    )
    return parser


def main(
    package: str = "pythia_test",
    argv: Optional[Sequence[str]] = None,
    *,
    stream: Optional[TextIO] = None,
    topology: Optional[CpuTopology] = None,
) -> int:
    args = _build_parser(package).parse_args(argv)
    if args.jobs is not None and args.jobs < 1:
        _build_parser(package).error("--jobs must be at least one")
    output = sys.stdout if stream is None else stream
    tap_started = False
    try:
        selectors = tuple(args.selectors) or _discover_modules(package)
        if not selectors:
            raise ValueError("No test modules were discovered.")
        topology = cpu_topology() if topology is None else topology
        slots = _worker_slots(topology, args.jobs)
        jobs = tuple(_Job(index, (selector,)) for index, selector in enumerate(selectors))
        _write_tap_header(topology, len(slots), output)
        tap_started = True
        results = _run_jobs(jobs, slots)
        return _render_tap(results, topology, len(slots), output, write_header=False)
    except KeyboardInterrupt:
        if not tap_started:
            print("TAP version 13", file=output)
        print("Bail out! interrupted", file=output)
        output.flush()
        return 130
    except BaseException as exc:
        if not tap_started:
            print("TAP version 13", file=output)
        print(f"Bail out! {_single_line(type(exc).__name__ + ': ' + str(exc))}", file=output)
        output.flush()
        return 2


def _internal_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m pythia.testing")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result")
    parser.add_argument("selectors", nargs="*")
    args = parser.parse_args(argv)
    if not args.worker or args.result is None or not args.selectors:
        parser.error("this module's direct entry point is reserved for test workers")
    return _run_worker(args.selectors, Path(args.result))


if __name__ == "__main__":
    raise SystemExit(_internal_main())

