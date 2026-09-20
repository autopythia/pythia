from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia import hardware


MAC_THREE_LEVELS = """
hw.nperflevels: 3
hw.perflevel0.name: Performance
hw.perflevel0.physicalcpu: 4
hw.perflevel0.logicalcpu: 4
hw.perflevel1.name: Performance
hw.perflevel1.physicalcpu: 4
hw.perflevel1.logicalcpu: 4
hw.perflevel2.name: Efficiency
hw.perflevel2.physicalcpu: 2
hw.perflevel2.logicalcpu: 2
hw.physicalcpu: 10
"""

LSCPU_SMT = """
# CPU,Core,Socket,Node
0,0,0,0
1,0,0,0
2,1,0,0
3,1,0,0
4,0,1,1
5,0,1,1
"""


class HardwareParserTests(unittest.TestCase):
    def test_sysctl_preserves_multiple_performance_and_efficiency_levels(self):
        topology = hardware._parse_sysctl_topology(MAC_THREE_LEVELS)
        self.assertEqual(topology.levels, (
            hardware.PerformanceLevel("P0", 4),
            hardware.PerformanceLevel("P1", 4),
            hardware.PerformanceLevel("E0", 2),
        ))
        self.assertEqual(topology.core_count, 10)
        self.assertEqual(topology.virtualization, {})

    def test_sysctl_homogeneous_fallback(self):
        topology = hardware._parse_sysctl_topology("hw.physicalcpu: 8\n")
        self.assertEqual(topology.levels, (hardware.PerformanceLevel("P0", 8),))
        self.assertIsNone(hardware._parse_sysctl_topology(
            "hw.nperflevels: 2\nhw.perflevel0.physicalcpu: 4\n"
        ))

    def test_lscpu_parser_deduplicates_by_socket_and_core(self):
        entries = hardware._parse_lscpu_entries(LSCPU_SMT)
        self.assertEqual(len(entries), 6)
        with mock.patch("os.sched_getaffinity", return_value={0, 1, 2, 3, 4, 5}):
            available = hardware._available_cpus(entries)
        levels = hardware._linux_levels(
            entries, available, level_reader=lambda _cpu: ("P", None),
        )
        # Socket 1/core 0 is distinct from socket 0/core 0.
        self.assertEqual(levels, (hardware.PerformanceLevel("P0", 3),))

    def test_linux_levels_keep_explicit_core_types_and_capacities(self):
        entries = hardware._parse_lscpu_entries("\n".join(
            f"{cpu},{cpu},0,0" for cpu in range(7)
        ))
        values = {
            0: ("P", 300), 1: ("P", 300),
            2: ("P", 200), 3: ("P", 200),
            4: ("E", 100), 5: ("E", 100), 6: ("E", 100),
        }
        levels = hardware._linux_levels(
            entries, set(range(7)), level_reader=values.__getitem__,
        )
        self.assertEqual(levels, (
            hardware.PerformanceLevel("P0", 2),
            hardware.PerformanceLevel("P1", 2),
            hardware.PerformanceLevel("E0", 3),
        ))

    def test_virtualization_metadata_retains_source_field_names(self):
        metadata = hardware._parse_lscpu_virtualization(
            "Hypervisor vendor: KVM\nVirtualization type: full\nVirtualization: AMD-V\n",
            dmi={"sys_vendor": "Hetzner", "product_name": "vServer"},
        )
        self.assertEqual(metadata, {
            "lscpu": {
                "Hypervisor vendor": "KVM",
                "Virtualization type": "full",
            },
            "dmi": {
                "sys_vendor": "Hetzner",
                "product_name": "vServer",
            },
        })
        # CPU virtualization extensions alone do not establish that this is a guest.
        self.assertEqual(hardware._parse_lscpu_virtualization(
            "Virtualization: AMD-V\n", dmi={"product_name": "host"}), {})

    def test_dmi_metadata_uses_native_file_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sys_vendor").write_text("Vendor\n")
            (root / "product_name").write_text("Guest\n")
            self.assertEqual(hardware._dmi_metadata(root), {
                "sys_vendor": "Vendor", "product_name": "Guest",
            })

    def test_virtual_linux_counts_visible_vcpus_not_synthetic_physical_cores(self):
        outputs = {
            ("lscpu", "-p=CPU,CORE,SOCKET,NODE"): LSCPU_SMT,
            ("lscpu",): "Hypervisor vendor: KVM\nVirtualization type: full\n",
        }
        with mock.patch.object(hardware, "_dmi_metadata", return_value={"board_vendor": "KVM"}), \
             mock.patch("os.sched_getaffinity", return_value={0, 1, 2, 3}):
            topology = hardware.cpu_topology(
                system="Linux", command=lambda argv: outputs.get(tuple(argv)),
            )
        self.assertEqual(topology.levels, (hardware.PerformanceLevel("P0", 4),))
        self.assertEqual(topology.virtualization["lscpu"]["Hypervisor vendor"], "KVM")
        self.assertEqual(topology.virtualization["dmi"], {"board_vendor": "KVM"})

    def test_query_falls_back_to_affinity_as_one_level(self):
        with mock.patch("os.sched_getaffinity", return_value={3, 4, 5}):
            topology = hardware.cpu_topology(system="Other", command=lambda _argv: None)
        self.assertEqual(topology.levels, (hardware.PerformanceLevel("P0", 3),))
        self.assertEqual(topology.virtualization, {})

    def test_current_machine_query_is_nonempty_and_consistent(self):
        topology = hardware.cpu_topology()
        self.assertGreaterEqual(topology.core_count, 1)
        self.assertEqual(topology.core_count,
                         sum(level.core_count for level in topology.levels))
        self.assertTrue(all(level.label for level in topology.levels))


if __name__ == "__main__":
    unittest.main()
