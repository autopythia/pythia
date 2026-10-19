"""icron: the scheduler, its code, and main's tool (pythia.interaction.loop.icron)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
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
from pythia.interaction.loop.icron import WAKE_HEADER, log_state
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
        self.assertEqual((done.state, done.output, done.value, done.error),
                         ("done", "hi\n", "42", None))
        self.assertEqual(icron.exec("x = 2").value, None)  # no final expression
        awaited = icron.exec("await asyncio.sleep(0.01)\nx * 2", by="main")
        self.assertEqual(awaited.value, "82")
        failed = icron.exec("y = 1\nraise ValueError('boom')", by="main")
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.error.splitlines(), [
            "Traceback (most recent call last):",
            f'  File "<icron {failed.job.job_id}>", line 2, in <module>',
            "    raise ValueError('boom')",
            "ValueError: boom"])
        bad = icron.exec("def f(:\n  pass", by="main")
        self.assertEqual((bad.state, bad.job), ("failed", None))
        self.assertIn("SyntaxError", bad.error)
        self.assertEqual(icron.jobs(), ())  # runs that ended within their call aren't listed

    def test_each_origin_has_one_namespace_for_the_session(self):
        icron = self.start()
        icron.exec("x = 1\ndef double(v):\n    return 2 * v", by="main")
        self.assertEqual(icron.exec("double(x)", by="main").value, "2")
        self.assertIn("NameError", icron.exec("x", by="host").error)
        names = icron.exec("sorted(n for n in globals() if not n.startswith('__'))", by="user")
        self.assertEqual(names.value, "['asyncio', 'icron', 'main', 'print']")

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
        self.assertEqual(outlived.state, "running")
        self.assertIn("so it goes on as a job", outlived.detail)
        self.assertEqual(self.job(outlived.job.job_id).state, "running")  # listed now
        cases = [
            ("main.resume('hi')\nawait asyncio.sleep(5)", None, "It sent you a message"),
            ("await main.handoff()", None, "It waits for the end of your turn"),
        ]
        for code, _scope, detail in cases:
            started = time.monotonic()
            result = icron.exec(code, wait=5, by="main")
            self.assertLess(time.monotonic() - started, 2)
            self.assertEqual(result.state, "running")
            self.assertTrue(result.detail.startswith(detail), result.detail)
        self.assertEqual(icron.undelivered(), 1)
        with CancelToken() as token, cancel_scope(token):
            threading.Timer(0.05, token.cancel).start()
            started = time.monotonic()
            result = icron.exec("await asyncio.sleep(5)", wait=5, by="main")
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(result.detail, "The wait was interrupted; the code goes on as a job.")
        with results_due_scope(time.monotonic() + 0.05):
            result = icron.exec("await asyncio.sleep(5)", wait=5, by="main")
        self.assertEqual(result.detail, "The model needs this result now; the code goes on as "
                                        "a job.")
        with self.assertRaisesRegex(IcronError, "wait must be at most 59 seconds"):
            icron.exec("1", wait=60, by="main")

    def test_a_busy_watcher_leaves_an_exec_run_pending(self):
        icron = self.start()
        icron.exec("import time\ntime.sleep(0.4)", wait=0, by="main")
        pending = icron.exec("41 + 1", wait=0.05, by="main")
        self.assertEqual(pending.state, "pending")
        self.assertIn("hasn't started", pending.detail)
        self.assertEqual(self.wait_for(pending.job.job_id, "done").state, "done")

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
        outcome = self.icron.tool().handler(arguments)
        return outcome.success, json.loads(outcome.output)

    def test_each_action_is_one_api_call(self):
        self.start()
        tool = self.icron.tool()
        self.assertEqual((tool.spec.name, tool.timeout_seconds), ("icron", 60.0))
        self.assertEqual(tool.spec.parameters["properties"]["action"]["enum"],
                         ["schedule", "exec", "yield", "cancel", "jobs"])
        ok, scheduled = self.call(action="schedule", message="Check CI run 123.", after=600)
        self.assertTrue(ok)
        self.assertEqual(set(scheduled), {"job_id", "state", "due", "in_seconds", "every",
                                          "remaining", "now"})
        self.assertEqual((scheduled["state"], scheduled["in_seconds"], scheduled["remaining"]),
                         ("scheduled", 600, 1))
        self.assertEqual(datetime.fromisoformat(scheduled["due"]).timestamp(), NOW + 600)
        self.assertEqual(self.job(scheduled["job_id"]).origin, "main")
        _, listed = self.call(action="jobs")
        self.assertEqual([(job["job_id"], job["origin"], job["message"]) for job in listed["jobs"]],
                         [(scheduled["job_id"], "main", "Check CI run 123.")])
        self.assertEqual(self.call(action="cancel", job_id=scheduled["job_id"])[1]["cancelled"],
                         scheduled["job_id"])
        ok, done = self.call(action="exec", code="print('x')\n1 + 1")
        self.assertEqual((ok, done["state"], done["output"], done["value"]), (True, "done",
                                                                              "x\n", "2"))
        ok, failed = self.call(action="exec", code="1 / 0")
        self.assertEqual((ok, failed["state"]), (False, "failed"))
        self.assertIn("ZeroDivisionError", failed["error"])
        ok, running = self.call(action="exec", code="await asyncio.sleep(5)", wait=0.1)
        self.assertEqual((ok, running["state"]), (True, "running"))
        self.assertIn("job_id", running)
        self.icron.begin_turn()
        ok, yielded = self.call(action="yield", note="Waiting.")
        self.assertEqual((yielded["yielded"], yielded["waits_for"]["job_id"]),
                         (True, running["job_id"]))
        self.assertEqual(self.icron.request(), "Waiting.")

    def test_inputs_that_are_not_a_call_fail(self):
        handler = self.icron.tool().handler
        cases = [({"action": "nope"}, "icron: action must be one of schedule, exec, yield, "
                                      "cancel, jobs."),
                 ({}, "icron: action must be one of"),
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
            self.assertTrue(str(caught.exception).startswith(text), str(caught.exception))

    def test_main_sees_its_jobs_and_the_user_s_and_cancels_only_its_own(self):
        icron = self.icron
        users = icron.schedule("u", after=60, by="user")
        icron.schedule("h", after=60)
        mine = icron.schedule("m", after=30, by="main")
        _, listed = self.call(action="jobs")
        self.assertEqual([job["job_id"] for job in listed["jobs"]], [mine.job_id, users.job_id])
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

    def test_log_state_names_what_a_restart_lost(self):
        def scheduled(due, every=None):
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
        model = _Model(ModelSample((_call("y", action="yield", note="Waiting for CI."),
                                    _call("j", action="jobs"))),
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
