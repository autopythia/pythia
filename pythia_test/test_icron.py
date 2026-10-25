"""icron: the scheduler, its code, and main's tool (pythia.interaction.loop.icron)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import inspect
import json
import re
import threading
import time
import unittest

from pythia.interaction import Environment, Init, InteractionContext, Message, ModelSample
from pythia.interaction import ToolCall, ToolResult, UserInteractionBoundary
from pythia.interaction.environment import CancelToken, cancel_scope, results_due_scope
from pythia.interaction.loop import Icron, IcronError, IcronLimits, TurnHost, Yield, run_turn
from pythia.interaction.loop import turn_yield
from pythia.interaction.loop.icron import WAKE_HEADER, call_text, log_state
from pythia.interaction.loop.kernel import TurnResult
from pythia.interaction.runtime_config import InteractionConfigSnapshot


NOW = 1_800_000_000.0  # 2027-01-15T08:00:00Z


class _Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def _until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _yield(kind="ended", **fields):
    extra = {"ended": {"final_text": "done"}, "yielded": {"yield_text": "waiting"},
             "failed": {"reason": "ValueError"}, "stopped": {}}[kind]
    return Yield(context=1, job_id=None, job_text="task", kind=kind,
                 resumable=kind != "stopped", revision=0, **{**extra, **fields})


class _IcronCase(unittest.TestCase):
    limits = IcronLimits()

    def setUp(self):
        self.clock = _Clock()
        self.notices = []
        self.delivered = threading.Event()
        self.deliveries = 0
        self.icron = Icron(clock=self.clock, limits=self.limits)
        self.addCleanup(self.icron.close, 1.0)

    def start(self):
        def deliver():
            self.deliveries += 1
            self.delivered.set()
        self.icron.start(deliver, self.notices.append)
        return self.icron

    def advance(self, seconds):
        self.clock.now += seconds
        self.icron._poke()

    def job(self, job_id):
        return next(job for job in self.icron.jobs() if job.job_id == job_id)

    def wait_for(self, job_id, *states):
        self.assertTrue(_until(lambda: self.job(job_id).state in states), self.job(job_id))
        return self.job(job_id)

    def wait_delivery(self):
        self.assertTrue(self.delivered.wait(3), "no wake-up was sent")
        self.delivered.clear()


class ScheduleTests(_IcronCase):
    def test_due_times_from_after_at_and_every(self):
        icron = self.icron
        self.assertEqual(icron.schedule("a", after=600).due, NOW + 600)
        for at in ("2027-01-15T10:30:00+01:00", "2027-01-15T09:30:00Z",
                   datetime(2027, 1, 15, 9, 30, tzinfo=timezone.utc)):
            self.assertEqual(icron.schedule("b", at=at).due, NOW + 5400, at)
        local = (datetime.fromtimestamp(NOW) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        self.assertAlmostEqual(icron.schedule("c", at=local).due, NOW + 7200, delta=3600)
        every = icron.schedule("d", every=3600)
        self.assertEqual((every.due, every.every, every.remaining), (NOW + 3600, 3600.0, None))
        counted = icron.schedule("e", after=10, every=30, count=3)  # the shortest repeat
        self.assertEqual((counted.due, counted.every, counted.remaining), (NOW + 10, 30.0, 3))
        once = icron.schedule(code="pass", after=5)
        self.assertEqual((once.kind, once.state, once.remaining, once.code),
                         ("code", "scheduled", 1, "pass"))

    def test_refused_calls_name_the_call(self):
        icron = self.icron
        cases = [
            (dict(message="x", after=5, at="2027-01-15T09:00:00Z"), "give after or at, not both"),
            (dict(message="x"), "give after, at, or every"),
            (dict(message="x", every=29), "every must be from 30"),
            (dict(message="x", after=0.5), "after must be from 1"),
            (dict(message="x", after=8 * 86400), "after must be from 1"),
            (dict(message="x", after=True), "after must be a number"),
            (dict(message="x", at="2027-01-15T07:00:00Z"), "at is in the past (now "),
            (dict(message="x", at="2027-01-30T07:00:00Z"), "at most"),
            (dict(message="x", at="2027-01-16"), "ISO 8601 date and time"),
            (dict(message="x", at=datetime(2027, 1, 16)), "aware datetime"),
            (dict(message="x", after=5, count=2), "count needs every"),
            (dict(message="x", every=60, count=0), "count must be a positive integer"),
            (dict(message="x", code="pass", after=5), "give a message or code, not both"),
            (dict(after=5), "give a message or code."),
            (dict(message="  ", after=5), "message must be nonblank"),
            (dict(message="x" * 8001, after=5), "at most 8000 characters"),
            (dict(code=lambda run: None, after=5, by="main"), "code must be Python source"),
            (dict(code=3, after=5), "code must be nonblank Python source"),
            (dict(code="def f(:\n  pass", after=5), "the code doesn't compile: "),
            (dict(message="x", after=5, by="model"), "by must be one of main, user, host"),
        ]
        for kwargs, text in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(IcronError) as caught:
                icron.schedule(**kwargs)
            self.assertIn(text, str(caught.exception))
            self.assertTrue(str(caught.exception).startswith("icron.schedule(): "))
        self.assertEqual(icron.jobs(), ())

    def test_pending_jobs_are_limited_per_origin(self):
        icron = Icron(clock=self.clock, limits=IcronLimits(max_jobs=3))
        for _ in range(3):
            icron.schedule("x", after=5, by="main")
        with self.assertRaisesRegex(IcronError, "main has 3 pending jobs already"):
            icron.schedule("x", after=5, by="main")
        for _ in range(3):  # main can't use up the user's room
            icron.schedule("x", after=5, by="user")
        with self.assertRaisesRegex(IcronError, "the user has 3 pending jobs already"):
            icron.schedule("x", after=5, by="user")
        for _ in range(5):  # host jobs are bound only by the time limits
            icron.schedule("x", after=5)

    def test_ids_are_unique_and_share_a_random_prefix(self):
        ids = [self.icron.schedule("x", after=5).job_id for _ in range(5)]
        self.assertEqual(len(set(ids)), 5)
        self.assertTrue(all(re.fullmatch(r"[a-z0-9]{4}-\d+", job_id) for job_id in ids))
        self.assertEqual(len({job_id.split("-")[0] for job_id in ids}), 1)
        self.assertNotEqual(Icron()._prefix + Icron()._prefix + Icron()._prefix,
                            self.icron._prefix * 3)


class FiringTests(_IcronCase):
    def test_a_message_job_sends_a_wake_up_and_ends_at_the_next_handoff(self):
        icron = self.start()
        job = icron.schedule("Check CI run 123.", after=600, by="main")
        self.advance(599)
        time.sleep(0.05)
        self.assertEqual((self.deliveries, icron.undelivered()), (0, 0))
        self.advance(1)
        self.wait_delivery()
        self.assertEqual(self.job(job.job_id).state, "running")
        wakeup = icron.take_wakes()
        self.assertEqual([(d.job.job_id, d.text, d.kind, d.due) for d in wakeup.deliveries],
                         [(job.job_id, "Check CI run 123.", "message", NOW + 600)])
        self.assertFalse(wakeup.resumes_yield)
        self.assertIsNone(icron.take_wakes())
        self.assertEqual(self.job(job.job_id).state, "running")  # until the turn ends
        icron.begin_turn()
        icron.handoff(_yield())
        self.assertEqual(self.wait_for(job.job_id, "done").remaining, 0)
        self.assertEqual(self.notices, [])  # a message job's end needs no notice

    def test_code_that_awaits_main_resume_is_what_a_message_job_does(self):
        icron = self.start()
        job = icron.schedule(code="y = await main.resume('m')\nprint(y.final_text)",
                             after=60, by="main")
        self.advance(60)
        self.wait_delivery()
        self.assertEqual([d.text for d in icron.take_wakes().deliveries], ["m"])
        icron.handoff(_yield(final_text="answered"))
        self.assertEqual(self.wait_for(job.job_id, "done").output, "answered\n")
        self.assertEqual(self.notices, [f"icron: job {job.job_id} (main's code) finished."])

    def test_repeats_run_at_a_fixed_rate_until_their_count(self):
        icron, calls = self.start(), []
        job = icron.schedule(code=lambda run: calls.append((run.due, run.merged)), every=60,
                             count=3)
        for expected in (1, 2, 3):
            self.advance(60)
            self.assertTrue(_until(lambda: len(calls) == expected), calls)
        self.assertEqual(calls, [(NOW + 60, 0), (NOW + 120, 0), (NOW + 180, 0)])
        self.assertEqual(self.wait_for(job.job_id, "done").due, None)

    def test_missed_slots_merge_into_one_run(self):
        icron, calls = self.start(), []
        job = icron.schedule(code=lambda run: calls.append((run.due, run.merged)), every=60,
                             count=10)
        self.advance(250)  # slots at 60, 120, 180, and 240 are due
        self.assertTrue(_until(lambda: calls), calls)
        self.assertEqual(calls, [(NOW + 240, 3)])
        scheduled = self.wait_for(job.job_id, "scheduled")
        self.assertEqual((scheduled.due, scheduled.remaining), (NOW + 300, 6))

    def test_a_firing_during_a_run_merges_into_it(self):
        icron, started, gate, seen = self.start(), threading.Event(), threading.Event(), []

        async def slow(run):
            started.set()
            while not gate.is_set():
                await asyncio.sleep(0.005)
            seen.append(run.merged)

        job = icron.schedule(code=slow, every=60, count=3)
        self.advance(60)
        self.assertTrue(started.wait(3))
        self.advance(60)
        self.assertTrue(_until(lambda: self.job(job.job_id).remaining == 1))
        gate.set()
        self.assertTrue(_until(lambda: seen == [1]), seen)
        self.assertEqual(self.wait_for(job.job_id, "scheduled").due, NOW + 180)

    def test_cancel_withdraws_an_undelivered_wake_up(self):
        icron = self.start()
        job = icron.schedule("x", after=60, by="user")
        self.advance(60)
        self.wait_delivery()
        self.assertEqual(icron.cancel(job.job_id, by="user").state, "cancelled")
        self.assertIsNone(icron.take_wakes())
        with self.assertRaisesRegex(IcronError, "is already cancelled|has already ended"):
            icron.cancel(job.job_id)  # whether or not its run has unwound yet
        self.assertEqual(self.wait_for(job.job_id, "cancelled").due, None)
        self.assertTrue(_until(lambda: not icron._jobs))
        with self.assertRaisesRegex(IcronError, "has already ended"):
            icron.cancel(job.job_id)
        with self.assertRaisesRegex(IcronError, "there is no job nope-1"):
            icron.cancel("nope-1")
        slow = icron.exec("try:\n    await asyncio.sleep(30)\nexcept asyncio.CancelledError:\n"
                          "    import time\n    time.sleep(0.3)\n    raise", wait=0.05, by="main")
        self.assertEqual(icron.cancel(slow.job.job_id).state, "cancelled")
        with self.assertRaisesRegex(IcronError, "is already cancelled"):
            icron.cancel(slow.job.job_id)  # its run still unwinds
        self.assertEqual(self.wait_for(slow.job.job_id, "cancelled").state, "cancelled")
        self.assertEqual([job.state for job in icron.jobs() if job.job_id == slow.job.job_id],
                         ["cancelled"])

    def test_a_failed_run_ends_its_job_with_a_notice(self):
        icron = self.start()
        job = icron.schedule(code="1 / 0", every=60, by="user")
        self.advance(60)
        failed = self.wait_for(job.job_id, "failed")
        self.assertIn("ZeroDivisionError", failed.error)
        self.assertIn(f'File "<icron {job.job_id}>", line 1', failed.error)
        self.assertEqual(self.notices, [f"icron: job {job.job_id} (the user's code) failed: "
                                        "ZeroDivisionError: division by zero"])
        self.advance(60)
        time.sleep(0.05)
        self.assertEqual(self.job(job.job_id).state, "failed")  # it fires no more

    def test_stop_cancels_running_code_and_starts_nothing_new(self):
        icron = self.start()
        running = icron.exec("await asyncio.sleep(30)", wait=0, by="main")
        later = icron.schedule("x", after=60, by="main")
        icron.stop()
        self.assertEqual(self.wait_for(running.job.job_id, "cancelled").error, None)
        self.advance(60)
        time.sleep(0.05)
        self.assertEqual((self.job(later.job_id).state, self.deliveries), ("scheduled", 0))
        with self.assertRaisesRegex(IcronError, "icron isn't running"):
            icron.exec("1", by="main")
        with self.assertRaisesRegex(IcronError, "icron is stopping"):
            icron.schedule("y", after=5, by="main")

    def test_close_drops_jobs_and_nothing_runs_after_it(self):
        icron = self.start()
        job = icron.schedule("x", after=60, by="main")
        running = icron.exec("await asyncio.sleep(30)", wait=0, by="main")
        dropped = icron.close()
        self.assertEqual(sorted(j.job_id for j in dropped),
                         sorted((job.job_id, running.job.job_id)))
        self.assertFalse(icron._thread.is_alive())
        self.advance(60)
        time.sleep(0.05)
        self.assertEqual((self.deliveries, icron.jobs(), icron.close()), (0, (), ()))

    def test_close_abandons_code_that_never_awaits(self):
        icron = self.start()
        blocking = icron.exec("import time\ntime.sleep(3)", wait=0, by="main")
        self.assertTrue(_until(lambda: self.job(blocking.job.job_id).started is not None))
        started = time.monotonic()
        icron.close(timeout=0.1)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(self.notices, ["icron: code that never awaited still held the watcher "
                                        "thread at exit; it was abandoned."])


class CodeTests(_IcronCase):
    def test_exec_returns_output_a_final_value_and_a_traceback(self):
        icron = self.start()
        done = icron.exec("print('hi')\nx = 41\nx + 1", by="main")
        self.assertEqual((done.state, done.output, done.value, done.error, done.reason),
                         ("done", "hi\n", "42", None, None))
        self.assertEqual(icron.exec("x = 2").value, None)  # no final expression
        awaited = icron.exec("await asyncio.sleep(0.01)\nx * 2", by="main")
        self.assertEqual(awaited.value, "82")
        failed = icron.exec("y = 1\nraise ValueError('boom')", by="main")
        self.assertEqual((failed.state, failed.reason), ("failed", None))
        self.assertEqual(failed.error.splitlines(), [
            "Traceback (most recent call last):",
            f'  File "<icron {failed.job.job_id}>", line 2, in <module>',
            "    raise ValueError('boom')",
            "ValueError: boom"])
        bad = icron.exec("def f(:\n  pass", by="main")
        self.assertEqual((bad.state, bad.job, bad.reason), ("failed", None, "it didn't run"))
        self.assertIn("SyntaxError", bad.error)
        self.assertEqual(icron.jobs(), ())  # runs that ended within their call aren't listed

    def test_each_origin_has_one_namespace_for_the_session(self):
        icron = self.start()
        icron.exec("x = 1\ndef double(v):\n    return 2 * v", by="main")
        self.assertEqual(icron.exec("double(x)", by="main").value, "2")
        self.assertIn("NameError", icron.exec("x", by="host").error)
        names = icron.exec("sorted(n for n in globals() if not n.startswith('__'))", by="user")
        self.assertEqual(names.value, "['asyncio', 'help', 'icron', 'main', 'print']")

    def test_runs_that_interleave_each_print_to_their_own_output(self):
        icron = self.start()
        code = ("for i in range(3):\n"
                "    print(icron.current.job_id)\n"
                "    await asyncio.sleep(0.01)\n"
                "await asyncio.to_thread(lambda: print(icron.current.job_id, 'thread'))")
        jobs = [icron.schedule(code=code, after=60, by="main") for _ in range(2)]
        self.advance(60)
        for job in jobs:
            self.assertEqual(self.wait_for(job.job_id, "done").output,
                             f"{job.job_id}\n" * 3 + f"{job.job_id} thread\n")

    def test_exec_waits_for_its_run_and_no_longer(self):
        icron = self.start()
        outlived = icron.exec("await asyncio.sleep(1)", wait=0.05, by="main")
        self.assertEqual((outlived.state, outlived.reason), ("running", "the wait ran out"))
        self.assertEqual(self.job(outlived.job.job_id).state, "running")  # listed now
        cases = [
            ("main.resume('hi')\nawait asyncio.sleep(5)",
             "it sent you a message, which you get after this tool batch"),
            ("await main.handoff()", "it awaits the end of your turn"),
        ]
        for code, reason in cases:
            started = time.monotonic()
            result = icron.exec(code, wait=5, by="main")
            self.assertLess(time.monotonic() - started, 2)
            self.assertEqual((result.state, result.reason), ("running", reason))
        self.assertEqual(icron.undelivered(), 1)
        with CancelToken() as token, cancel_scope(token):
            threading.Timer(0.05, token.cancel).start()
            started = time.monotonic()
            result = icron.exec("await asyncio.sleep(5)", wait=5, by="main")
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(result.reason, "the user interrupted the wait")
        with results_due_scope(time.monotonic() + 0.05):
            result = icron.exec("await asyncio.sleep(5)", wait=5, by="main")
        self.assertEqual(result.reason, "the call reached its deadline")
        with self.assertRaisesRegex(IcronError, "wait must be at most 59 seconds"):
            icron.exec("1", wait=60, by="main")

    def test_a_busy_watcher_leaves_an_exec_run_pending(self):
        icron = self.start()
        icron.exec("import time\ntime.sleep(0.4)", wait=0, by="main")
        pending = icron.exec("41 + 1", wait=0.05, by="main")
        # The reason stays the one the call returned for.
        self.assertEqual((pending.state, pending.reason), ("pending", "the wait ran out"))
        self.assertEqual(self.wait_for(pending.job.job_id, "done").state, "done")

    def test_exec_says_whether_a_stop_cancelled_its_run(self):
        icron = self.start()
        itself = icron.exec("print('start')\nraise asyncio.CancelledError()", by="main")
        self.assertEqual((itself.state, itself.output, itself.reason),
                         ("cancelled", "start\n", None))  # it ended by itself
        # The user's runs are listed from the start, so cancel() can reach one in its call.
        threading.Timer(0.2, lambda: icron.cancel(icron.jobs()[0].job_id, by="user")).start()
        cancelled = icron.exec("await asyncio.sleep(5)", wait=5, by="user")
        self.assertEqual((cancelled.state, cancelled.reason), ("cancelled", None))  # no stop
        threading.Timer(0.2, icron.stop).start()
        stopped = icron.exec("await asyncio.sleep(5)", wait=5, by="main")
        self.assertEqual((stopped.state, stopped.reason), ("cancelled", "a stop cancelled it"))

    def test_main_resume_sends_at_once_and_returns_main_s_next_handoff(self):
        icron = self.start()
        unawaited = icron.exec("main.resume('one')\nmain.state", by="main")
        self.assertEqual((unawaited.state, unawaited.value), ("done", "'quiescent'"))
        self.wait_delivery()
        awaiting = icron.exec("y = await main.resume('two')\nprint(y.kind, y.final_text)",
                              wait=0, by="main")
        self.wait_delivery()
        self.assertEqual([d.text for d in icron.take_wakes().deliveries], ["one", "two"])
        icron.begin_turn()
        self.assertEqual(icron.main.state, "running")
        icron.handoff(_yield(final_text="fin"), (Message("user", "q"), Message("assistant", "fin")))
        self.assertEqual(self.wait_for(awaiting.job.job_id, "done").output, "ended fin\n")
        later = icron.exec("y = await main.handoff()\nprint(y.kind, len(main.view()))", wait=0,
                           by="main")
        self.assertTrue(_until(lambda: icron._handoff_waiters))
        icron.handoff(_yield("stopped"))
        self.assertEqual(self.wait_for(later.job.job_id, "done").output, "stopped 0\n")
        self.assertEqual(icron.exec("main.last().kind", by="main").value, "'stopped'")
        with self.assertRaisesRegex(IcronError, r"main.resume\(\): text must be nonblank"):
            icron.main.resume(" ")

    def test_cancel_from_code(self):
        icron = self.start()
        own = icron.schedule(code="icron.cancel(icron.current.job_id)\n"
                                  "await asyncio.sleep(0.01)\nprint('went on')",
                             every=60, by="main")
        self.advance(60)
        ended = self.wait_for(own.job_id, "done")
        self.assertEqual((ended.output, ended.due), ("went on\n", None))
        other = icron.exec("await asyncio.sleep(30)", wait=0, by="main")
        icron.exec(f"icron.cancel({other.job.job_id!r})", by="main")
        self.assertEqual(self.wait_for(other.job.job_id, "cancelled").state, "cancelled")

    def test_the_stall_check_names_the_run_that_holds_the_watcher(self):
        self.limits = IcronLimits(stall_seconds=0.2)
        self.icron = Icron(clock=self.clock, limits=self.limits)
        self.addCleanup(self.icron.close, 2.0)
        icron = self.start()
        self.assertIsNone(icron.status().stalled)
        blocking = icron.exec("import time\ntime.sleep(1)", wait=0, by="main")
        self.assertTrue(_until(lambda: icron.status().stalled is not None))
        self.assertEqual(icron.status().stalled[0], blocking.job.job_id)
        self.assertTrue(_until(lambda: icron.status().stalled is None))

    def test_code_s_icron_has_no_exec_or_yield_and_the_watcher_thread_can_t_exec(self):
        icron = self.start()
        self.assertEqual(icron.exec("hasattr(icron, 'exec'), hasattr(icron, 'yield_')",
                                    by="main").value, "(False, False)")
        job = icron.schedule(code=lambda run: icron.exec("1"), after=60)
        self.advance(60)
        self.assertIn("can't wait for code on the watcher thread",
                      self.wait_for(job.job_id, "failed").error)

    def test_the_user_s_runs_show_their_output_as_notices(self):
        icron = self.start()
        ok = icron.exec("print('a')\nprint('b', end='')\n6 * 7", wait=0, by="user")
        self.wait_for(ok.job.job_id, "done")
        bad = icron.exec("raise KeyError('k')", wait=0, by="user")
        self.wait_for(bad.job.job_id, "failed")
        self.assertEqual(self.notices[:3], [f"py {ok.job.job_id}: a", f"py {ok.job.job_id}: b",
                                            f"py {ok.job.job_id}: done: 42"])
        self.assertTrue(self.notices[3].startswith(f"py {bad.job.job_id}: failed:\n  Traceback"))
        self.assertTrue(self.notices[3].endswith("KeyError: 'k'"))


class ToolTests(_IcronCase):
    def call(self, **arguments):
        """A call's success and its result's lines: every result is text."""
        outcome = self.icron.tool().handler(arguments)
        return outcome.success, outcome.output.splitlines()

    def exec_lines(self, **arguments):
        """exec's result for code; wait, if given, goes in the call."""
        wait = arguments.pop("wait", None)
        return self.call(**arguments, **({} if wait is None else {"call": f"exec(wait={wait})"}))

    def test_each_action_is_one_api_call(self):
        self.start()
        tool = self.icron.tool()
        # The tool timeout, as the command tools': its exec waits end before it.
        self.assertEqual((tool.spec.name, tool.timeout_seconds), ("icron", 59.0))
        # The input is a call in Python, and code: the keywords are the methods' own,
        # which the description and help() list. Other fields reach the handler,
        # which reads the first version's form too.
        parameters = tool.spec.parameters
        self.assertEqual(set(parameters["properties"]), {"call", "code"})
        self.assertNotIn("required", parameters)
        self.assertNotIn("additionalProperties", parameters)
        ok, scheduled = self.call(call="schedule('Check CI run 123.', after=600)")
        self.assertTrue(ok)
        job_id = scheduled[0].split()[2]
        self.assertEqual(scheduled, [f"Scheduled job {job_id} (yours; due {_iso(NOW + 600)}, "
                                     "in 600 s)", f"Now: {_iso(NOW)}"])
        self.assertEqual(self.job(job_id).origin, "main")
        _, listed = self.call(call="jobs()")
        self.assertEqual(listed, ["1 pending job", f"Now: {_iso(NOW)}",
                                  f"Job {job_id} (yours; due {_iso(NOW + 600)}, in 600 s):",
                                  "  Message:", "    Check CI run 123."])
        self.assertEqual(self.call(call=f"cancel({job_id!r})"),
                         (True, [f"Cancelled job {job_id} (yours)", f"Now: {_iso(NOW)}"]))
        ok, done = self.exec_lines(code="print('x')\n1 + 1")
        self.assertEqual((ok, done[0], done[3:]),
                         (True, "Script completed", ["Value: 2", "Output:", "  x"]))
        ok, failed = self.exec_lines(code="1 / 0")
        self.assertEqual((ok, failed[0], failed[-1]),
                         (False, "Script failed", "  ZeroDivisionError: division by zero"))
        ok, running = self.exec_lines(code="await asyncio.sleep(5)", wait=0.1)
        job_id = re.fullmatch(r"Script running with job ID (\S+)", running[0]).group(1)
        self.assertTrue(ok)
        self.icron.begin_turn()
        ok, yielded = self.call(call="yield_('Waiting.')")
        self.assertEqual((ok, yielded[0], yielded[1].split(" (")[0]),
                         (True, "Yielded: your turn ends once this tool batch is saved",
                          f"Job {job_id}"))
        self.assertEqual(self.icron.request(), "Waiting.")
        ok, helped = self.call(call="help('cancel')")
        self.assertEqual((ok, helped[0]), (True, "cancel(job_id)"))

    def test_every_form_that_means_one_call_reads_as_that_call(self):
        self.start()
        forms = [  # (input, the call as the transcript shows it, its result's first line)
            ({"call": "schedule('Hi.', after=600)"},
             "icron.schedule(message='Hi.', after=600)", "Scheduled job"),
            ({"call": "icron.schedule('Hi.', after=10*60)"},
             "icron.schedule(message='Hi.', after=600)", "Scheduled job"),
            ({"call": "schedule(message='Hi.', after=-(-5)*2/1+0.5)"},
             "icron.schedule(message='Hi.', after=10.5)", "Scheduled job"),
            ({"action": "schedule", "message": "Hi.", "after": 600},
             "icron.schedule(message='Hi.', after=600)", "Scheduled job"),
            ({"call": "schedule('''Line 1.\nLine 2.''', every=270)"},
             "icron.schedule(message='Line 1.\\nLine 2.', every=270)", "Scheduled job"),
            ({"call": "schedule(every=60, count=3)", "code": "print('poll')"},
             "icron.schedule(every=60, count=3)", "Scheduled job"),
            ({"code": "1 + 1"}, "icron.exec()", "Script completed"),
            ({"call": "exec()", "code": "1 + 1"}, "icron.exec()", "Script completed"),
            ({"call": "exec(wait=5)", "code": "1 + 1"}, "icron.exec(wait=5)", "Script completed"),
            ({"call": "exec('1 + 1')"}, "icron.exec()", "Script completed"),
            ({"call": "exec(code='1 + 1', wait=5)"}, "icron.exec(wait=5)", "Script completed"),
            ({"action": "exec", "code": "1 + 1", "wait": 5}, "icron.exec(wait=5)",
             "Script completed"),
            ({"call": "jobs()"}, "icron.jobs()", "6 pending jobs"),
            ({"action": "jobs"}, "icron.jobs()", "6 pending jobs"),
            ({"call": "help()"}, "icron.help()", "icron: this session's in-process scheduler."),
            ({"call": "icron.help('Run')"}, "icron.help(name='Run')", "Run"),
        ]
        for arguments, shown, first in forms:
            with self.subTest(arguments=arguments):
                self.assertEqual(call_text(arguments)[0], shown)
                ok, lines = self.call(**arguments)
                self.assertTrue(ok)
                self.assertTrue(lines[0].startswith(first), lines)
        for arguments in ({"call": "yield('Waiting.')"},
                          {"call": " icron . yield(note='Waiting.') "},
                          {"call": "yield_('Waiting.')"}, {"action": "yield", "note": "Waiting."}):
            with self.subTest(arguments=arguments):
                self.assertEqual(call_text(arguments), ("icron.yield_(note='Waiting.')", None))
                self.icron.begin_turn()
                self.assertEqual(self.call(**arguments)[1][0],
                                 "Yielded: your turn ends once this tool batch is saved")
                self.icron.handoff(_yield("ended"))
        job_id = self.icron.jobs()[0].job_id
        for arguments in ({"call": f"cancel({job_id!r})"}, {"action": "cancel", "job_id": job_id}):
            self.assertEqual(call_text(arguments)[0], f"icron.cancel(job_id={job_id!r})")
        self.assertTrue(self.call(call=f"cancel(job_id={job_id!r})")[1][0].startswith(
            f"Cancelled job {job_id} "))

    def test_exec_s_result_reads_like_code_mode_s(self):
        # The wording of Codex's and pi's code mode: the run's state, the call's wall time
        # with the reason when the code didn't run to its end, then the value, and the
        # output and the error as blocks, indented so that no line in them reads as a label.
        self.start()
        wall = r"Wall time \d+\.\d seconds"
        ok, done = self.exec_lines(code="class R:\n    def __repr__(self):\n"
                                        "        return 'first\\n\\nthird'\n"
                                        "print('Output:\\n\\nx')\nR()")
        self.assertTrue(ok)
        self.assertEqual(done[0], "Script completed")
        self.assertRegex(done[1], f"^{wall}$")
        self.assertEqual(datetime.fromisoformat(done[2].removeprefix("Now: ")).timestamp(), NOW)
        self.assertEqual(done[3:], ["Value:", "  first", "", "  third",
                                    "Output:", "  Output:", "", "  x"])
        # A failed run's output and error are separate blocks.
        ok, failed = self.exec_lines(code="print('before', end='')\n{}['k']")
        self.assertFalse(ok)
        self.assertEqual((failed[0], failed[3:7], failed[-1]),
                         ("Script failed",
                          ["Output:", "  before", "Error:", "  Traceback (most recent call last):"],
                          "  KeyError: 'k'"))
        ok, broken = self.exec_lines(code="def f(:\n    pass")
        self.assertFalse(ok)
        self.assertRegex(broken[1], rf"^{wall} \(it didn't run\)$")
        self.assertEqual(broken[3], "Error:")  # nothing ran, so no Output block
        self.assertTrue(broken[-1].startswith("  SyntaxError"), broken)
        ok, running = self.exec_lines(code="print('start')\nawait asyncio.sleep(5)", wait=0.1)
        self.assertTrue(ok)
        self.assertRegex(running[0], r"^Script running with job ID [a-z0-9]{4}-\d+$")
        self.assertRegex(running[1], rf"^{wall} \(the wait ran out\)$")
        self.assertEqual(running[3:], ["Output so far:", "  start"])
        self.icron.exec("import time\ntime.sleep(0.4)", wait=0, by="main")  # holds the watcher
        ok, pending = self.exec_lines(code="41 + 1", wait=0.05)
        self.assertTrue(ok)
        self.assertRegex(pending[0], r"^Script pending with job ID [a-z0-9]{4}-\d+ until "
                                     r"icron's event loop is free$")
        self.assertRegex(pending[1], rf"^{wall} \(the wait ran out\)$")
        self.assertEqual(pending[3:], [])  # an empty block is left out

    def test_schedule_and_cancel_results_name_the_job_as_wake_ups_do(self):
        self.start()
        ok, once = self.call(action="schedule", message="Check CI run 123.", after=600)
        self.assertEqual((ok, once[0]), (True, f"Scheduled job {once[0].split()[2]} (yours; due "
                                               f"{_iso(NOW + 600)}, in 600 s)"))
        _, counted = self.call(action="schedule", code="print('poll')", every=60, count=3)
        _, endless = self.call(action="schedule", message="Keep going.", every=270)
        _, last = self.call(action="schedule", message="Last call.", every=30, count=1,
                            at=_iso(NOW + 90))
        polls, keeps, lasts = (lines[0].split()[2] for lines in (counted, endless, last))
        self.assertEqual([counted[0], endless[0], last[0]], [
            f"Scheduled job {polls} (your code; due {_iso(NOW + 60)}, in 60 s, every 60 s, "
            "3 firings left)",
            f"Scheduled job {keeps} (yours; due {_iso(NOW + 270)}, in 270 s, every 270 s)",
            f"Scheduled job {lasts} (yours; due {_iso(NOW + 90)}, in 90 s, every 30 s, "
            "1 firing left)"])
        self.assertEqual({lines[1] for lines in (once, counted, endless, last)},
                         {f"Now: {_iso(NOW)}"})
        # Cancelling running code says when it stops; anything else is just cancelled.
        _, running = self.exec_lines(code="await asyncio.sleep(5)", wait=0.05)
        code = re.match(r"Script (?:running|pending) with job ID (\S+)", running[0]).group(1)
        self.assertTrue(_until(lambda: self.job(code).started is not None))
        self.assertEqual(self.call(action="cancel", job_id=code)[1][0],
                         f"Cancelled job {code} (your code, from exec): its code stops at its "
                         "next await")
        self.assertEqual(self.call(action="cancel", job_id=polls)[1][0],
                         f"Cancelled job {polls} (your code)")

    def test_yield_s_result_says_what_resumes_main(self):
        icron = self.start()
        head = "Yielded: your turn ends once this tool batch is saved"

        def yielded(*cleanup):
            icron.begin_turn()
            ok, lines = self.call(action="yield", note="Waiting.")
            self.assertTrue(ok)
            self.assertEqual((lines[0], lines[-1]), (head, f"Now: {_iso(icron.clock())}"))
            icron.handoff(_yield("ended"))
            for job_id in cleanup:
                icron.cancel(job_id)
            return lines[1]

        later = icron.schedule("Check CI run 123.", after=600, by="main")
        self.assertEqual(yielded(later.job_id),
                         f"Job {later.job_id} (yours; due {_iso(NOW + 600)}, in 600 s) resumes "
                         "you when it fires, unless the user writes first.")
        code = icron.exec("await asyncio.sleep(5)", wait=0.01, by="main").job
        self.assertTrue(_until(lambda: self.job(code.job_id).started is not None))
        self.assertEqual(yielded(code.job_id),
                         f"Job {code.job_id} (your code, from exec; running since {_iso(NOW)}) "
                         "resumes you when it sends you a message or ends, unless the user "
                         "writes first.")
        # A message that already waits for main: from a job, or from host code.
        fired = icron.schedule("Look at the logs.", after=30, by="main")
        self.advance(30)
        self.assertTrue(_until(lambda: icron.undelivered() == 1))
        self.assertEqual(yielded(),
                         f"A wake-up from job {fired.job_id} (yours; sent {_iso(NOW + 30)}) "
                         "already waits for you.")
        icron.take_wakes()
        icron.main.resume("From the host.")
        self.assertEqual(yielded(), "A wake-up already waits for you.")

    def test_jobs_lists_each_job_with_its_code_message_output_and_error_as_blocks(self):
        icron = self.start()
        self.assertEqual(self.call(action="jobs")[1], ["No jobs", f"Now: {_iso(NOW)}"])
        broken = icron.schedule(code="print('fetching')\nraise KeyError('missing')", after=10,
                                by="main")
        self.advance(10)
        self.wait_for(broken.job_id, "failed")
        self.assertEqual(self.call(action="jobs")[1][0], "No pending jobs, and 1 that ended")
        running = icron.exec("print('step 0')\nprint()\nprint('step 1')\n"
                             "await asyncio.sleep(5)", wait=0.05, by="main").job
        self.assertTrue(_until(lambda: "step 1" in self.job(running.job_id).output))
        poll = icron.schedule(code="print('poll')", every=60, count=3, by="main")
        users = icron.schedule("Stand-up in 5 minutes.\n  Bring notes.", after=1800, by="user")
        _, listed = self.call(action="jobs")
        self.assertEqual(listed, [
            "3 pending jobs, and 1 that ended", f"Now: {_iso(NOW + 10)}",
            f"Job {running.job_id} (your code, from exec; running since {_iso(NOW + 10)}):",
            "  Code:", "    print('step 0')", "    print()", "    print('step 1')",
            "    await asyncio.sleep(5)",
            "  Output so far:", "    step 0", "", "    step 1",
            f"Job {poll.job_id} (your code; due {_iso(NOW + 70)}, in 60 s, every 60 s, "
            "3 firings left):",
            "  Code:", "    print('poll')",
            f"Job {users.job_id} (the user's, from /cron; due {_iso(NOW + 1810)}, in 1800 s):",
            "  Message:", "    Stand-up in 5 minutes.", "      Bring notes.",
            f"Job {broken.job_id} (your code; failed at {_iso(NOW + 10)}):",
            "  Code:", "    print('fetching')", "    raise KeyError('missing')",
            "  Output:", "    fetching",
            "  Error:", "    Traceback (most recent call last):",
            f'      File "<icron {broken.job_id}>", line 2, in <module>',
            "        raise KeyError('missing')", "    KeyError: 'missing'"])

    def test_jobs_shows_the_start_of_long_code_and_the_end_of_long_output(self):
        icron = self.start()
        long = icron.exec("print('o' * 5000)\nawait asyncio.sleep(5)  # " + "c" * 3000,
                          wait=0.05, by="main").job
        self.assertTrue(_until(lambda: self.job(long.job_id).output))
        _, listed = self.call(action="jobs")
        # The clipping notices stay inside their blocks.
        self.assertEqual(listed[2:], [
            f"Job {long.job_id} (your code, from exec; running since {_iso(NOW)}):",
            "  Code:", "    print('o' * 5000)",
            "    await asyncio.sleep(5)  # " + "c" * 1956 + "… [1044 more characters]",
            "  Output so far:", "    [1001 earlier characters omitted]", "    " + "o" * 3999])

    def test_inputs_that_are_not_a_call_fail(self):
        handler = self.icron.tool().handler
        # Each refusal says what to give instead; nothing in a call is evaluated.
        cases = [
            ({}, "icron: give call, one method call such as jobs(), or code to run now."),
            ({"call": "jobs()", "action": "jobs"}, "icron: give call or action, not both."),
            ({"call": "exec(code='print(2)')", "code": "print(1)"},
             "icron.exec(): give the code in the code field only."),
            ({"call": "exec('print(2)')", "code": "print(1)"},
             "icron.exec(): give the code in the code field only."),
            ({"call": "jobs()", "code": "print(1)"},
             "icron.jobs(): it takes no code; leave the code field out."),
            ({"call": "exec(wait=30)"}, "icron.exec(): give the code to run in the code field."),
            ({"call": "shedule('Hi.', after=600)"},
             "icron: there is no method shedule(); did you mean schedule()?"),
            ({"call": "launch()"}, "icron: there is no method launch()."),
            ({"call": "schedule('Hi.', after=delay)"},
             "icron.schedule(): after must be a literal, such as 600 or 'text'."),
            ({"call": "schedule(open('x').read())"},
             "icron.schedule(): message must be a literal, such as 600 or 'text'."),
            ({"call": "cancel('a'); jobs()"},
             "icron: call must be one method call, such as jobs(); invalid syntax."),
            ({"call": "jobs"}, "icron: call must be one method call, such as jobs()."),
            ({"call": "main.resume('x')"}, "icron: call must be one method call, such as jobs()."),
            ({"call": "schedule(**options)"},
             "icron.schedule(): give each argument by itself, not **mapping."),
            ({"call": "schedule(*options)"},
             "icron.schedule(): give each argument by itself, not *sequence."),
            ({"call": "jobs()", "wait": 3},
             "icron: unexpected field 'wait'; give call, and code for exec() or schedule()."),
            ({"call": 5}, "icron: call must be text, one method call such as jobs()."),
            ({"call": "yield_()"}, "icron.yield_(): missing a required argument: 'note'"),
            ({"call": "jobs(limit=3)"}, "icron.jobs(): got an unexpected keyword argument 'limit'"),
            ({"call": "jobs(by='user')"}, "icron.jobs(): by is for host code only."),
            ({"code": 5}, "icron.exec(): code must be a string."),
            ({"call": "help('nothing')"}, "icron.help(): there is no name 'nothing'. help() lists "
                                          "them."),
            ({"call": "help('shedule')"}, "icron.help(): there is no name 'shedule'; did you mean "
                                          "'schedule'? help() lists them."),
            # The first version's form.
            ({"action": "nope"}, "icron: action must be one of schedule, exec, yield, cancel, "
                                 "jobs."),
            ({"action": "schedule", "message": "x", "after": 5, "by": "user"},
             "icron.schedule(): by is for host code only."),
            ({"action": "exec", "code": ["print(1)"]}, "icron.exec(): code must be a string."),
            ({"action": "yield"}, "icron.yield_(): missing a required argument: 'note'"),
            ({"action": "jobs", "limit": 3},
             "icron.jobs(): got an unexpected keyword argument 'limit'"),
            ({"action": "cancel", "job_id": 7}, "icron.cancel(): job_id must be a string.")]
        for arguments, text in cases:
            with self.subTest(arguments=arguments), self.assertRaises(IcronError) as caught:
                handler(arguments)
            self.assertEqual(str(caught.exception), text)
        self.assertEqual(self.icron.jobs(), ())  # nothing ran, and nothing was scheduled

    def test_a_call_too_deep_to_read_is_refused(self):
        handler = self.icron.tool().handler
        for call in ("schedule('x', after=" + "+".join(["1"] * 5000) + ")",
                     "schedule('x', after=" + "(" * 300 + "1" + ")" * 300 + ")",
                     "jobs(\0)"):
            with self.subTest(call=call[:40]), self.assertRaises(IcronError) as caught:
                handler({"call": call})
            self.assertTrue(str(caught.exception).startswith(
                "icron: call must be one method call, such as jobs()"), str(caught.exception))

    def test_the_description_has_a_line_for_each_method_made_from_its_signature(self):
        description = self.icron.tool().spec.description
        self.assertEqual([line.split(": ", 1)[0] for line in description.splitlines()
                          if line.startswith("- ")], [
            "- exec(wait=10)",
            "- schedule(message=None, *, after=None, at=None, every=None, count=None)",
            "- yield_(note)", "- cancel(job_id)", "- jobs()", "- help(name=None)"])
        self.assertIn("(at most 59)", description)
        self.assertIn('  {"code": "import os\\nprint(os.getloadavg())"}', description)
        # The bounds are the instance's.
        narrow = Icron(limits=IcronLimits(max_wait=20))
        self.assertIn("(at most 20)", narrow.tool().spec.description)
        self.assertIn("exec's wait up to 20 seconds", narrow.help())

    def test_help_shows_an_overview_and_each_name_in_full(self):
        overview = self.icron.help()
        self.assertTrue(overview.startswith("icron: this session's in-process scheduler."))
        lines = overview.splitlines()
        for title in ("Methods:", "In code:", "Messages:", "Limits:", "Results:"):
            self.assertIn(title, lines)
        for signature in ("exec(wait=10)", "yield_(note)", "help(name=None)"):
            self.assertIn(f"  {signature}", lines)
        self.assertIn("16 pending jobs and 16 undelivered messages each", overview)
        for name in ("exec", "schedule", "yield_", "cancel", "jobs", "help", "icron", "main",
                     "main.resume", "Job", "Run"):
            with self.subTest(name=name):
                text = self.icron.help(name)
                self.assertEqual(text.splitlines()[1][:2], "  ")
                if name in ("exec", "schedule", "yield_", "cancel", "help"):
                    # Each keyword of the method has its sentence, so a new one needs one.
                    method = getattr(Icron, name)
                    for parameter in inspect.signature(method).parameters:
                        if parameter not in ("self", "by"):
                            self.assertIn(f"\n  {parameter}: ", text)
        self.assertEqual(self.icron.help("schedule").splitlines()[0],
                         "schedule(message=None, *, after=None, at=None, every=None, count=None)")
        for alias, name in (("yield", "yield_"), (" icron.schedule() ", "schedule"),
                            ("main.handoff", "main"), ("current", "icron")):
            self.assertEqual(self.icron.help(alias), self.icron.help(name))
        # A method's result, for example, is the formatter's own.
        self.assertIn("Result, for example:\n  Cancelled job k3f9-3 (your code, from exec): its "
                      "code stops at its next await\n", self.icron.help("cancel"))
        with self.assertRaisesRegex(IcronError, r"^icron\.help\(\): name must be text"):
            self.icron.help(5)

    def test_code_s_help_prints_the_same_reference_or_python_s_help(self):
        self.start()
        for code, name in (("help('cancel')", "cancel"), ("help(main.resume)", "main.resume"),
                           ("help(icron.schedule)", "schedule"), ("help(icron)", "icron"),
                           ("help(icron.current)", "Run"), ("help()", None)):
            with self.subTest(code=code):
                ok, lines = self.exec_lines(code=code)
                self.assertEqual((ok, lines[3]), (True, "Output:"))
                self.assertEqual(lines[4:], [f"  {line}" if line else ""
                                             for line in self.icron.help(name).splitlines()])
        ok, lines = self.exec_lines(code="help(asyncio.sleep)")
        self.assertTrue(any("Python Library Documentation: function sleep" in line
                            for line in lines), lines)
        ok, lines = self.exec_lines(code="help('no.such.thing')")
        self.assertEqual((ok, lines[4]), (True, "  No Python documentation found for "
                                                "'no.such.thing'; help() shows icron's reference."))
        ok, lines = self.exec_lines(code="import inspect\n[inspect.getdoc(f) for f in "
                                         "(icron.schedule, icron.cancel, icron.jobs)]")
        self.assertNotIn("None", lines[3])

    def test_main_sees_its_jobs_and_the_user_s_and_cancels_only_its_own(self):
        icron = self.icron
        users = icron.schedule("u", after=60, by="user")
        icron.schedule("h", after=60)
        mine = icron.schedule("m", after=30, by="main")
        _, listed = self.call(action="jobs")
        self.assertEqual([line.split()[1] for line in listed if line.startswith("Job ")],
                         [mine.job_id, users.job_id])
        with self.assertRaisesRegex(IcronError, "is the user's; you can cancel only your own"):
            self.call(action="cancel", job_id=users.job_id)
        self.assertEqual(len(icron.jobs()), 3)


class YieldTests(_IcronCase):
    def test_a_yield_needs_a_turn_and_something_to_resume_main(self):
        icron = self.icron
        with self.assertRaisesRegex(IcronError, "no turn is open"):
            icron.yield_("x")
        icron.begin_turn()
        with self.assertRaisesRegex(IcronError, "nothing could resume you"):
            icron.yield_("x")
        users = icron.schedule("u", after=60, by="user")
        self.assertEqual(icron.yield_("Waiting for the user's reminder.").job_id, users.job_id)
        self.assertEqual(icron.request(), "Waiting for the user's reminder.")
        with self.assertRaisesRegex(IcronError, "a yield is already recorded"):
            icron.yield_("again")
        with self.assertRaisesRegex(IcronError, "note must be nonblank"):
            icron.yield_(" ")
        icron.handoff(_yield("ended"))
        self.assertIsNone(icron.request())  # the window closed with the turn

    def test_a_yield_suspends_main_until_a_turn_begins_or_nothing_is_left(self):
        icron = self.icron
        later = icron.schedule("later", after=600, by="main")
        sooner = icron.schedule("sooner", after=60, by="main")
        icron.begin_turn()
        icron.yield_("Waiting.")
        self.assertEqual(icron.handoff(_yield("yielded")).job_id, sooner.job_id)
        status = icron.status()
        self.assertEqual((status.main_state, status.suspended_on.job_id),
                         ("suspended", sooner.job_id))
        icron.cancel(sooner.job_id)
        self.assertEqual(icron.status().suspended_on.job_id, later.job_id)
        icron.cancel(later.job_id)
        self.assertEqual(icron.main.state, "quiescent")  # released: nothing is left
        icron.schedule("again", after=60, by="main")
        icron.begin_turn()
        icron.yield_("Waiting.")
        icron.handoff(_yield("yielded"))
        icron.begin_turn()  # the user wrote first
        self.assertEqual(icron.main.state, "running")
        icron.handoff(_yield("ended"))
        self.assertEqual(icron.main.state, "quiescent")

    def test_a_suspended_main_hears_how_its_code_jobs_end(self):
        icron = self.start()
        finished = icron.exec("await asyncio.sleep(0.2)\nprint('tests pass')", wait=0.01,
                              by="main")
        icron.begin_turn()
        icron.yield_("Waiting for the tests.")
        self.assertEqual(icron.handoff(_yield("yielded")).job_id, finished.job.job_id)
        self.wait_delivery()
        wakeup = icron.take_wakes()
        self.assertTrue(wakeup.resumes_yield)
        self.assertEqual(wakeup.deliveries[0].kind, "end")
        self.assertEqual(wakeup.text().splitlines()[2:], [
            f"Job {finished.job.job_id} (your code, from exec at "
            f"{datetime.fromtimestamp(NOW).strftime('%H:%M:%S')}) ended without a message for "
            "you. Its last output:", "tests pass"])
        self.assertEqual(self.notices, [])
        icron.begin_turn()
        icron.handoff(_yield("ended"))
        # A failure while main is suspended goes to main; one that already
        # messaged main during the suspension sends nothing more.
        failing = icron.exec("await asyncio.sleep(0.2)\n1 / 0", wait=0.01, by="main")
        messaging = icron.exec("await asyncio.sleep(0.2)\nmain.resume('progress')", wait=0.01,
                               by="main")
        icron.begin_turn()
        icron.yield_("Waiting.")
        icron.handoff(_yield("yielded"))
        self.wait_for(failing.job.job_id, "failed")
        self.wait_for(messaging.job.job_id, "done")
        texts = [(d.job.job_id, d.kind, d.text.splitlines()[0])
                 for d in icron.take_wakes().deliveries]
        self.assertEqual(sorted(texts), sorted([(failing.job.job_id, "end", "failed:"),
                                                (messaging.job.job_id, "message", "progress")]))

    def test_the_wake_up_text_names_each_job_and_who_scheduled_it(self):
        icron = self.start()
        mine = icron.schedule("Poll the log.", every=600, by="main")
        users = icron.schedule("Remind me to review the PR.", after=1200, by="user")
        self.advance(1200)  # main's job merges the slot at 600 into the one at 1200
        self.assertTrue(_until(lambda: icron.undelivered() == 2))
        wakeup = icron.take_wakes()
        self.assertEqual(wakeup.text().splitlines(), [
            f"{WAKE_HEADER} at {_iso(NOW + 1200)}.", "",
            f"Job {mine.job_id} (yours; due {_iso(NOW + 1200)}, every 600 s; 1 earlier firing "
            "merged):", "Poll the log.", "",
            f"Job {users.job_id} (the user's, from /cron; due {_iso(NOW + 1200)}):",
            "Remind me to review the PR."])


def _iso(seconds):
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def _call(call_id, **arguments):
    return ToolCall("icron", call_id, json.dumps(arguments))


class LogTests(unittest.TestCase):
    def test_a_yield_is_recorded_only_in_the_last_turn_and_only_with_success(self):
        yielded = (_call("a", action="yield", note="x"), ToolResult("a", "{}"))
        user = (Message("user", "q"), UserInteractionBoundary())
        self.assertTrue(Icron.yield_recorded_in((*user, *yielded)))
        self.assertFalse(Icron.yield_recorded_in((*yielded, *user)))
        self.assertFalse(Icron.yield_recorded_in((*user, yielded[0],
                                                  ToolResult("a", "no", success=False))))
        self.assertFalse(Icron.yield_recorded_in((*user, _call("b", action="jobs"),
                                                  ToolResult("b", "{}"))))
        self.assertFalse(Icron.yield_recorded_in((*user, ToolCall("yield", "c", "{}"),
                                                  ToolResult("c", "ok"))))
        # Any form that the tool reads as a yield: the call, its shorthands, the first version's.
        for arguments in ({"call": "yield_('x')"}, {"call": "yield('x')"},
                          {"call": "icron.yield_(note='x')"}, {"action": "yield", "note": "x"}):
            with self.subTest(arguments=arguments):
                self.assertTrue(Icron.yield_recorded_in((*user, _call("a", **arguments),
                                                         ToolResult("a", "Yielded: …"))))
        for arguments in ({"call": "jobs()"}, {"call": "yield_()"}, {"call": "yeild('x')"},
                          {"code": "yield_('x')"}):
            with self.subTest(arguments=arguments):
                self.assertFalse(Icron.yield_recorded_in((*user, _call("a", **arguments),
                                                          ToolResult("a", "…"))))

    def test_log_state_names_what_a_restart_lost(self):
        def scheduled(due, every=None):  # a result as saved before October 25: JSON
            return (_call("s", action="schedule", message="x", after=5),
                    ToolResult("s", json.dumps({"job_id": "k-1", "due": _iso(due),
                                                "every": every})))
        self.assertIsNone(log_state((Message("user", "q"),), now=NOW))
        self.assertEqual(log_state(scheduled(NOW + 60), now=NOW), "jobs")
        self.assertIsNone(log_state(scheduled(NOW - 60), now=NOW))
        self.assertEqual(log_state(scheduled(NOW - 60, every=60), now=NOW), "jobs")
        self.assertEqual(log_state((_call("e", action="exec", code="1"),
                                    ToolResult("e", "{}")), now=NOW), "jobs")
        self.assertEqual(log_state((_call("y", action="yield", note="x"),
                                    ToolResult("y", "{}")), now=NOW), "yielded")

    def test_log_state_reads_the_schedule_results_that_main_gets(self):
        icron = Icron(clock=lambda: NOW)
        self.addCleanup(icron.close, 1.0)

        def scheduled(**arguments):
            arguments = {"action": "schedule", "message": "x", **arguments}
            return (_call("s", **arguments),
                    ToolResult("s", icron.tool().handler(arguments).output))
        once, repeat, counted = scheduled(after=60), scheduled(every=60), \
            scheduled(every=60, count=2)
        self.assertTrue(once[1].output.startswith("Scheduled job "), once[1].output)
        self.assertEqual(log_state(once, now=NOW + 59), "jobs")
        self.assertIsNone(log_state(once, now=NOW + 61))
        self.assertEqual(log_state(repeat, now=NOW + 3600), "jobs")
        self.assertEqual(log_state(counted, now=NOW + 90), "jobs")  # its second firing is due
        self.assertIsNone(log_state((once[0], ToolResult("s", "Scheduled job k-1 (yours; due "
                                                              "soon, in 60 s)")), now=NOW))
        # New-form calls: code alone ran code, and so did exec().
        new = (_call("s", call="schedule('x', after=60)"), once[1])
        self.assertEqual(log_state(new, now=NOW + 59), "jobs")
        self.assertIsNone(log_state(new, now=NOW + 61))
        for arguments in ({"code": "1"}, {"call": "exec()", "code": "1"}):
            self.assertEqual(log_state((_call("e", **arguments),
                                        ToolResult("e", "Script completed")), now=NOW), "jobs")
        self.assertIsNone(log_state((_call("j", call="jobs()"), ToolResult("j", "No jobs")),
                                    now=NOW))


class _Host(TurnHost):
    def __init__(self):
        self.saved = []

    def append(self, context, items):
        context.extend(items)
        self.saved.append(len(context))


class _Model:
    def __init__(self, *samples):
        self.samples = list(samples)

    def sample(self, context, *, tools=(), sample_params=None):
        return self.samples.pop(0)


class KernelTests(_IcronCase):
    def test_an_icron_yield_ends_the_turn_once_its_batch_is_saved(self):
        icron = self.start()
        icron.schedule("Check CI run 123.", after=600, by="main")
        context = InteractionContext((Init(model="scripted"), Message("user", "task")))
        model = _Model(ModelSample((_call("y", call="yield_('Waiting for CI.')"),
                                    _call("j", call="jobs()"))),
                       ModelSample((Message("assistant", "never"),)))
        icron.begin_turn()
        result = run_turn(context, model, Environment((icron.tool(),)),
                          InteractionConfigSnapshot(), _Host(), control=icron)
        self.assertEqual((result.kind, result.note), ("yielded", "Waiting for CI."))
        self.assertEqual([item.call_id for item in context.items if isinstance(item, ToolResult)],
                         ["y", "j"])  # the whole batch ran and was saved
        self.assertEqual(len(model.samples), 1)
        self.assertTrue(Icron.yield_recorded_in(context.items))
        handoff = turn_yield(result, context, job_text="task")
        self.assertEqual((handoff.kind, handoff.yield_text), ("yielded", "Waiting for CI."))
        self.assertIsNotNone(icron.handoff(handoff, context.items))
        self.assertEqual(icron.main.state, "suspended")

    def test_turn_yield_describes_each_way_a_turn_stops(self):
        context = InteractionContext((Init(model="scripted"), Message("user", "task")))
        ended = turn_yield(TurnResult("ended", "done"), context, job_text="task", resumes=2)
        self.assertEqual((ended.kind, ended.final_text, ended.resumes, ended.revision),
                         ("ended", "done", 2, 2))
        self.assertEqual(turn_yield(TurnResult("stopped"), context, job_text="t").resumable,
                         False)
        failed = turn_yield(ValueError("x"), context, job_text="t")
        self.assertEqual((failed.kind, failed.reason, failed.resumable),
                         ("failed", "ValueError", True))


if __name__ == "__main__":
    unittest.main()
