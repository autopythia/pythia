from __future__ import annotations

import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

from pythia.hardware import CpuTopology, PerformanceLevel
from pythia import testing


class TapResultTests(unittest.TestCase):
    def run_cases(self, case):
        result = testing.TapTestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(case).run(result)
        return result.records

    def test_unittest_outcomes_and_subtests_are_recorded_once_per_method(self):
        class Cases(unittest.TestCase):
            def test_success(self):
                pass

            @unittest.skip("not on this platform")
            def test_skip(self):
                pass

            @unittest.expectedFailure
            def test_expected_failure(self):
                self.fail("expected detail")

            @unittest.expectedFailure
            def test_unexpected_success(self):
                pass

            def test_failure(self):
                self.assertEqual(1, 2)

            def test_subtests(self):
                for value in (1, 2, 3):
                    with self.subTest(value=value):
                        self.assertNotEqual(value, 2)

        records = self.run_cases(Cases)
        by_name = {record.test_id.rsplit(".", 1)[-1]: record for record in records}
        self.assertEqual(len(records), 6)
        self.assertEqual(by_name["test_success"].status, "success")
        self.assertEqual(by_name["test_skip"].status, "skip")
        self.assertEqual(by_name["test_skip"].directive, "not on this platform")
        self.assertEqual(by_name["test_expected_failure"].status, "expected_failure")
        self.assertIn("expected detail", by_name["test_expected_failure"].diagnostic)
        self.assertEqual(by_name["test_unexpected_success"].status, "unexpected_success")
        self.assertEqual(by_name["test_failure"].status, "failure")
        self.assertEqual(by_name["test_subtests"].status, "failure")
        self.assertIn("(value=2)", by_name["test_subtests"].diagnostic)

    def test_setup_error_is_an_error_record(self):
        class Cases(unittest.TestCase):
            def setUp(self):
                raise RuntimeError("setup failed")

            def test_never_runs(self):
                self.fail("body ran")

        records = self.run_cases(Cases)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "error")
        self.assertIn("setup failed", records[0].diagnostic)


class TapRenderingTests(unittest.TestCase):
    def topology(self):
        return CpuTopology(
            (PerformanceLevel("P0", 2), PerformanceLevel("E0", 1)),
            {"lscpu": {"Hypervisor vendor": "KVM", "Virtualization type": "full"}},
        )

    def result(self, *records, stdout="", stderr=""):
        return testing._JobResult(
            testing._Job(0, ("example.tests",)), testing._WorkerSlot("P0", 0),
            tuple(records), stdout, stderr, 0.1,
        )

    def test_virtualization_metadata_remains_generic_and_source_shaped(self):
        self.assertEqual(testing._format_virtualization({
            "lscpu": {"Hypervisor vendor": "KVM"},
            "systemd-detect-virt": "kvm",
        }), "lscpu.Hypervisor vendor=KVM, systemd-detect-virt=kvm")

    def test_tap_maps_directives_diagnostics_metadata_and_exit_status(self):
        output = io.StringIO()
        status = testing._render_tap((self.result(
            testing.TestRecord("case.success", "success"),
            testing.TestRecord("case.skip", "skip", directive="reason\ncontinued"),
            testing.TestRecord("case.todo", "expected_failure", "expected traceback"),
            testing.TestRecord("case.failure", "failure", "first\nsecond"),
        ),), self.topology(), 3, output)
        text = output.getvalue()
        self.assertEqual(status, 1)
        self.assertTrue(text.startswith("TAP version 13\n"))
        self.assertIn("# workers: 3; levels: P0=2, E0=1", text)
        self.assertIn("lscpu.Hypervisor vendor=KVM", text)
        self.assertIn("ok 1 - case.success", text)
        self.assertIn("ok 2 - case.skip # SKIP reason continued", text)
        self.assertIn("not ok 3 - case.todo # TODO expected failure", text)
        self.assertIn("not ok 4 - case.failure\n# first\n# second", text)
        self.assertTrue(text.endswith("1..4\n"))

    def test_expected_failures_do_not_make_run_fail_and_output_is_commented(self):
        output = io.StringIO()
        status = testing._render_tap((self.result(
            testing.TestRecord("case.todo", "expected_failure", "known"),
            stdout="ordinary\noutput\n", stderr="warning\n",
        ),), self.topology(), 3, output)
        self.assertEqual(status, 0)
        text = output.getvalue()
        self.assertIn("# stdout from example.tests\n# ordinary\n# output", text)
        self.assertIn("# stderr from example.tests\n# warning", text)
        self.assertNotIn("\nordinary\n", text)

    @unittest.skipUnless(shutil.which("prove"), "requires a TAP parser")
    def test_successful_document_is_accepted_by_prove(self):
        output = io.StringIO()
        self.assertEqual(testing._render_tap((self.result(
            testing.TestRecord("case.success", "success"),
            testing.TestRecord("case.skip", "skip", directive="portable skip"),
            testing.TestRecord("case.todo", "expected_failure", "known"),
        ),), self.topology(), 3, output), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.tap"
            path.write_text(output.getvalue())
            result = subprocess.run(["prove", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class WorkerAndSchedulerTests(unittest.TestCase):
    def test_worker_round_trip_uses_structured_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            self.assertEqual(testing._run_worker(
                ("pythia_test.test_transport_retry",), path,
            ), 0)
            records = testing._load_worker_result(path)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record.status == "success" for record in records))
        self.assertTrue(all(record.test_id.startswith("pythia_test.test_transport_retry.")
                            for record in records))

    def test_scheduler_uses_every_slot_without_exceeding_slot_count(self):
        jobs = tuple(testing._Job(index, (f"module{index}",)) for index in range(9))
        slots = tuple(testing._WorkerSlot("P0", index) for index in range(3))
        lock = threading.Lock()
        active = maximum = 0
        used = set()

        def execute(job, slot, directory, registry):
            nonlocal active, maximum
            self.assertTrue(directory.is_dir())
            self.assertIsInstance(registry, testing._ProcessRegistry)
            with lock:
                active += 1
                maximum = max(maximum, active)
                used.add(slot.name)
            time.sleep(0.02)
            with lock:
                active -= 1
            return testing._JobResult(
                job, slot, (testing.TestRecord(job.name, "success"),), elapsed_seconds=0.02,
            )

        results = testing._run_jobs(jobs, slots, execute=execute)
        self.assertEqual([result.job.order for result in results], list(range(9)))
        self.assertEqual(maximum, 3)
        self.assertEqual(used, {"P0/0", "P0/1", "P0/2"})

    def test_scheduler_turns_slot_exceptions_into_failures_and_continues(self):
        jobs = tuple(testing._Job(index, (f"module{index}",)) for index in range(3))
        slots = (testing._WorkerSlot("P0", 0),)

        def execute(job, slot, directory, registry):
            if job.order == 1:
                raise RuntimeError("slot exploded")
            return testing._JobResult(
                job, slot, (testing.TestRecord(job.name, "success"),),
            )

        results = testing._run_jobs(jobs, slots, execute=execute)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1].records[0].status, "error")
        self.assertIn("slot exploded", results[1].records[0].diagnostic)
        self.assertEqual(results[2].records[0].status, "success")

    def test_slots_include_all_levels_and_explicit_override(self):
        topology = CpuTopology((PerformanceLevel("P0", 2), PerformanceLevel("E0", 2)))
        self.assertEqual([slot.name for slot in testing._worker_slots(topology)],
                         ["P0/0", "P0/1", "E0/0", "E0/1"])
        self.assertEqual([slot.name for slot in testing._worker_slots(topology, 3)],
                         ["P0/0", "P0/1", "E0/0"])
        self.assertEqual(len(testing._worker_slots(topology, 6)), 6)

    def test_discovery_finds_test_modules_but_not_unused_suffixes(self):
        modules = testing._discover_modules("pythia_test")
        self.assertIn("pythia_test.test_transport_retry", modules)
        self.assertIn("pythia_test.test_testing", modules)
        self.assertNotIn("pythia_test.test_messages_model_limits", modules)
        self.assertEqual(len(modules), len(set(modules)))

    def test_public_frontend_runs_selected_module_in_subprocess(self):
        output = io.StringIO()
        topology = CpuTopology((PerformanceLevel("P0", 2),))
        status = testing.main(
            "pythia_test", ["--jobs", "2", "pythia_test.test_transport_retry"],
            stream=output, topology=topology,
        )
        text = output.getvalue()
        self.assertEqual(status, 0, text)
        self.assertEqual(text.count("\nok "), 4)
        self.assertIn("# workers: 2; levels: P0=2", text)
        self.assertTrue(text.endswith("1..4\n"))


if __name__ == "__main__":
    unittest.main()
