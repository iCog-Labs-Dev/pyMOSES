"""Tests for tools/server/resources.py on fake /sys and /proc roots."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "server"))
import resources  # noqa: E402

GB = 2**30


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class FakeMachine:
    """A fake /sys and /proc: `cores` physical cores with `threads` hyper-threads each
    (logical CPU c belongs to core c // threads), MemTotal `mem_gb`."""

    def __init__(self, tmp: Path, cores=8, threads=2, mem_gb=64, cgroup="0::/\n",
                 topology=True):
        self.sys, self.proc = tmp / "sys", tmp / "proc"
        self.cpus = set(range(cores * threads))
        for c in self.cpus:
            if topology:
                _write(self.sys, f"devices/system/cpu/cpu{c}/topology/core_id", f"{c // threads}\n")
                _write(self.sys, f"devices/system/cpu/cpu{c}/topology/physical_package_id", "0\n")
        _write(self.proc, "meminfo",
               f"MemTotal:       {mem_gb * 2**20} kB\nMemAvailable:   {mem_gb * 2**19} kB\n")
        _write(self.proc, "self/cgroup", cgroup)

    def decide(self, env=None, affinity=None):
        return resources.decide(self.sys, self.proc,
                                affinity=self.cpus if affinity is None else affinity,
                                env=env or {})


class TestResources(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_physical_cores_without_cgroup(self):
        m = FakeMachine(self.tmp, cores=8, threads=2, mem_gb=64)
        d = m.decide()
        self.assertEqual(d["physical_cores_in_affinity"], 8)
        self.assertEqual(d["logical_cpus_in_affinity"], 16)
        self.assertEqual(d["usable_cpus"], 8)
        self.assertEqual(d["workers"], 7)  # 8 - 1; memory allows (65536 - 6553) // 3072 = 19
        self.assertEqual(d["limited_by"], "cpu")
        self.assertEqual(d["reserve_mb"], 6553)

    def test_affinity_subset_counts_its_own_cores(self):
        m = FakeMachine(self.tmp, cores=8, threads=2, mem_gb=64)
        d = m.decide(affinity={0, 1, 2, 3})  # two cores, both threads
        self.assertEqual(d["usable_cpus"], 2)
        self.assertEqual(d["workers"], 1)

    def test_no_topology_falls_back_to_logical(self):
        m = FakeMachine(self.tmp, cores=4, threads=2, mem_gb=64, topology=False)
        d = m.decide()
        self.assertIsNone(d["physical_cores_in_affinity"])
        self.assertEqual(d["usable_cpus"], 8)
        self.assertEqual(d["workers"], 7)

    def test_memory_limits_workers(self):
        m = FakeMachine(self.tmp, cores=16, threads=1, mem_gb=14)
        d = m.decide()
        # 14336 MB - max(2048, 1433) = 12288 -> 4 runs of 3072 MB
        self.assertEqual(d["reserve_mb"], 2048)
        self.assertEqual(d["workers"], 4)
        self.assertEqual(d["limited_by"], "memory")

    def test_cgroup_v2_quota_and_memory(self):
        m = FakeMachine(self.tmp, cores=32, threads=2, mem_gb=256, cgroup="0::/\n")
        _write(m.sys, "fs/cgroup/cgroup.controllers", "cpu memory\n")
        _write(m.sys, "fs/cgroup/cpu.max", "600000 100000\n")
        _write(m.sys, "fs/cgroup/memory.max", f"{32 * GB}\n")
        d = m.decide()
        self.assertEqual(d["cgroup_cpu_quota"], 6.0)
        self.assertEqual(d["usable_cpus"], 6)
        self.assertEqual(d["mem_mb"], 32768)
        self.assertIn("cgroup", d["mem_source"])
        self.assertEqual(d["workers"], 5)  # 6 - 1; memory: (32768 - 3276) // 3072 = 9

    def test_cgroup_v2_fractional_quota_rounds_down(self):
        m = FakeMachine(self.tmp, cores=8, threads=1, mem_gb=64)
        _write(m.sys, "fs/cgroup/cgroup.controllers", "cpu\n")
        _write(m.sys, "fs/cgroup/cpu.max", "250000 100000\n")
        d = m.decide()
        self.assertEqual(d["usable_cpus"], 2)
        self.assertEqual(d["workers"], 1)

    def test_cgroup_v2_unlimited(self):
        m = FakeMachine(self.tmp, cores=4, threads=1, mem_gb=64)
        _write(m.sys, "fs/cgroup/cgroup.controllers", "cpu memory\n")
        _write(m.sys, "fs/cgroup/cpu.max", "max 100000\n")
        _write(m.sys, "fs/cgroup/memory.max", "max\n")
        d = m.decide()
        self.assertIsNone(d["cgroup_cpu_quota"])
        self.assertIsNone(d["cgroup_mem_limit_mb"])
        self.assertEqual(d["usable_cpus"], 4)
        self.assertEqual(d["workers"], 3)

    def test_cgroup_v2_nested_path_takes_tightest(self):
        m = FakeMachine(self.tmp, cores=16, threads=1, mem_gb=128,
                        cgroup="0::/kubepods/pod1/c1\n")
        _write(m.sys, "fs/cgroup/kubepods/cpu.max", "400000 100000\n")
        _write(m.sys, "fs/cgroup/kubepods/pod1/c1/cpu.max", "max 100000\n")
        _write(m.sys, "fs/cgroup/kubepods/pod1/memory.max", f"{20 * GB}\n")
        d = m.decide()
        self.assertEqual(d["usable_cpus"], 4)
        self.assertEqual(d["cgroup_mem_limit_mb"], 20480)
        self.assertEqual(d["workers"], 3)  # memory: (20480 - 2048) // 3072 = 6

    def test_cgroup_v1(self):
        m = FakeMachine(self.tmp, cores=16, threads=2, mem_gb=128,
                        cgroup="5:memory:/\n4:cpu,cpuacct:/\n")
        _write(m.sys, "fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "800000\n")
        _write(m.sys, "fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "100000\n")
        _write(m.sys, "fs/cgroup/memory/memory.limit_in_bytes", f"{16 * GB}\n")
        d = m.decide()
        self.assertEqual(d["usable_cpus"], 8)
        self.assertEqual(d["mem_mb"], 16384)
        self.assertEqual(d["workers"], 4)  # (16384 - 2048) // 3072 = 4 < 7
        self.assertEqual(d["limited_by"], "memory")

    def test_cgroup_v1_unlimited(self):
        m = FakeMachine(self.tmp, cores=4, threads=1, mem_gb=64,
                        cgroup="5:memory:/\n4:cpu,cpuacct:/\n")
        _write(m.sys, "fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "-1\n")
        _write(m.sys, "fs/cgroup/cpu,cpuacct/cpu.cfs_period_us", "100000\n")
        _write(m.sys, "fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n")
        d = m.decide()
        self.assertIsNone(d["cgroup_cpu_quota"])
        self.assertIsNone(d["cgroup_mem_limit_mb"])
        self.assertEqual(d["workers"], 3)

    def test_quota_capped_by_physical_cores(self):
        m = FakeMachine(self.tmp, cores=4, threads=2, mem_gb=64)
        _write(m.sys, "fs/cgroup/cgroup.controllers", "cpu\n")
        _write(m.sys, "fs/cgroup/cpu.max", "1600000 100000\n")
        self.assertEqual(m.decide()["usable_cpus"], 4)

    def test_cgroup_memory_above_memtotal_is_ignored(self):
        m = FakeMachine(self.tmp, cores=8, threads=1, mem_gb=16)
        _write(m.sys, "fs/cgroup/cgroup.controllers", "memory\n")
        _write(m.sys, "fs/cgroup/memory.max", f"{64 * GB}\n")
        d = m.decide()
        self.assertEqual(d["mem_mb"], 16384)
        self.assertIn("MemTotal", d["mem_source"])

    def test_at_least_one_worker(self):
        m = FakeMachine(self.tmp, cores=1, threads=1, mem_gb=2)
        d = m.decide()
        self.assertEqual(d["workers"], 1)
        self.assertIn("floor of 1", d["limited_by"])

    def test_env_overrides(self):
        m = FakeMachine(self.tmp, cores=16, threads=1, mem_gb=64)
        self.assertEqual(m.decide()["workers"], 15)
        d = m.decide(env={"MAX_WORKERS": "3"})
        self.assertEqual((d["workers"], d["limited_by"]), (3, "MAX_WORKERS"))
        self.assertEqual(m.decide(env={"MAX_WORKERS": "40"})["workers"], 15)
        d = m.decide(env={"PER_RUN_MB": "8192", "RESERVE_MB": "1024"})
        self.assertEqual((d["per_run_mb"], d["reserve_mb"], d["workers"]), (8192, 1024, 7))


if __name__ == "__main__":
    unittest.main()
