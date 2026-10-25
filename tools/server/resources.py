"""How many runs a phase may run in parallel on this machine (SERVER-ROUND-PREREGISTRATION.md,
"Resources"). Prints one JSON object: the decision `workers` and every input to it.

    python3 tools/server/resources.py

    W = max(1, min(usable_cpus - 1, floor((mem - reserve) / per_run_mb)))

* usable_cpus: the cgroup CPU quota (v2 `cpu.max`, v1 `cpu.cfs_quota_us / cpu.cfs_period_us`,
  rounded down, at least 1) if one is set, else the physical cores in the process's
  affinity set (`os.sched_getaffinity`; distinct (`physical_package_id`, `core_id`) pairs
  from /sys topology, or the logical count when the topology is unreadable). A quota is
  also capped at those physical cores.
* mem: the cgroup memory limit (v2 `memory.max`, v1 `memory.limit_in_bytes`) if set and
  smaller than MemTotal, else MemTotal. reserve = max(2048 MB, 10 % of mem).
* per_run_mb: 3072, the per-run RLIMIT_AS of every manifest.

Environment overrides: MAX_WORKERS (an upper cap on W), PER_RUN_MB, RESERVE_MB.
Standard library only; the roots of /sys and /proc are parameters for the tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_PER_RUN_MB = 3072
MIN_RESERVE_MB = 2048


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_paths(proc: Path) -> dict[str, str]:
    """{controller: path} from /proc/self/cgroup; the v2 unified entry is key ""."""
    out: dict[str, str] = {}
    text = _read(proc / "self" / "cgroup") or ""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrls, path = parts
        if ctrls == "":
            out[""] = path
        for c in ctrls.split(","):
            if c:
                out[c] = path
    return out


def _candidates(base: Path, rel: str | None) -> list[Path]:
    """The cgroup directory of this process and its ancestors (as visible under `base`),
    then `base` itself: in a container the namespaced cgroup is usually mounted at the
    root, on a host it is a sub-directory."""
    dirs = []
    if rel:
        p = Path(rel.lstrip("/"))
        while True:
            dirs.append(base / p)
            if p == Path(".") or str(p) == "":
                break
            p = p.parent
    dirs.append(base)
    seen, uniq = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def cgroup_cpu_quota(sys_root: Path, proc_root: Path) -> tuple[float | None, str | None]:
    """(CPUs allowed by the tightest CPU quota on the way up, source) or (None, None)."""
    cg = _cgroup_paths(proc_root)
    base = sys_root / "fs" / "cgroup"
    best: tuple[float | None, str | None] = (None, None)

    def take(v: float, src: str):
        nonlocal best
        if best[0] is None or v < best[0]:
            best = (v, src)

    if (base / "cgroup.controllers").exists() or "" in cg:
        for d in _candidates(base, cg.get("")):
            t = _read(d / "cpu.max")
            if t:
                q, _, per = t.partition(" ")
                if q != "max" and per:
                    try:
                        take(int(q) / int(per), f"cgroup v2 {d / 'cpu.max'}")
                    except (ValueError, ZeroDivisionError):
                        pass
    rel = cg.get("cpu")
    for mount in ("cpu,cpuacct", "cpu", "cpuacct,cpu"):
        for d in _candidates(base / mount, rel):
            q, per = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
            if q and per:
                try:
                    if int(q) > 0 and int(per) > 0:
                        take(int(q) / int(per), f"cgroup v1 {d / 'cpu.cfs_quota_us'}")
                except ValueError:
                    pass
    return best


def cgroup_mem_limit(sys_root: Path, proc_root: Path) -> tuple[int | None, str | None]:
    """(tightest cgroup memory limit in bytes, source) or (None, None)."""
    cg = _cgroup_paths(proc_root)
    base = sys_root / "fs" / "cgroup"
    best: tuple[int | None, str | None] = (None, None)

    def take(v: int, src: str):
        nonlocal best
        if best[0] is None or v < best[0]:
            best = (v, src)

    for d in _candidates(base, cg.get("")):
        t = _read(d / "memory.max")
        if t and t != "max":
            try:
                take(int(t), f"cgroup v2 {d / 'memory.max'}")
            except ValueError:
                pass
    for d in _candidates(base / "memory", cg.get("memory")):
        t = _read(d / "memory.limit_in_bytes")
        if t:
            try:
                v = int(t)
                if v < 2**60:  # v1 "unlimited" is about 2**63
                    take(v, f"cgroup v1 {d / 'memory.limit_in_bytes'}")
            except ValueError:
                pass
    return best


def cgroup_mem_current(sys_root: Path = Path("/sys"), proc_root: Path = Path("/proc")) -> int | None:
    """Current cgroup memory use in bytes (v2 `memory.current`, v1 `memory.usage_in_bytes`)."""
    cg = _cgroup_paths(proc_root)
    base = sys_root / "fs" / "cgroup"
    for d in _candidates(base, cg.get("")):
        t = _read(d / "memory.current")
        if t and t.isdigit():
            return int(t)
    for d in _candidates(base / "memory", cg.get("memory")):
        t = _read(d / "memory.usage_in_bytes")
        if t and t.isdigit():
            return int(t)
    return None


def meminfo(proc_root: Path) -> dict[str, int]:
    """/proc/meminfo as {key: bytes}."""
    out = {}
    for line in (_read(proc_root / "meminfo") or "").splitlines():
        k, _, rest = line.partition(":")
        toks = rest.split()
        if toks and toks[0].isdigit():
            out[k.strip()] = int(toks[0]) * (1024 if len(toks) > 1 and toks[1] == "kB" else 1)
    return out


def physical_cores(sys_root: Path, cpus: set[int]) -> int | None:
    """Distinct (package, core) pairs among `cpus`, or None if any topology is missing."""
    cores = set()
    for c in cpus:
        t = sys_root / "devices" / "system" / "cpu" / f"cpu{c}" / "topology"
        core, pkg = _read(t / "core_id"), _read(t / "physical_package_id")
        if core is None or pkg is None:
            return None
        cores.add((pkg, core))
    return len(cores) or None


def decide(
    sys_root: Path = Path("/sys"),
    proc_root: Path = Path("/proc"),
    affinity: set[int] | None = None,
    env: dict | None = None,
) -> dict:
    env = os.environ if env is None else env
    if affinity is None:
        affinity = set(os.sched_getaffinity(0))
    logical = len(affinity)
    phys = physical_cores(sys_root, affinity)
    per_core = phys if phys is not None else logical
    quota, quota_src = cgroup_cpu_quota(sys_root, proc_root)
    if quota is not None:
        usable = max(1, min(int(quota), per_core))
        cpu_source = f"cgroup CPU quota ({quota_src})"
    elif phys is not None:
        usable, cpu_source = phys, "physical cores in the affinity set (/sys topology)"
    else:
        usable, cpu_source = logical, "logical CPUs in the affinity set (no /sys topology)"

    mi = meminfo(proc_root)
    memtotal = mi.get("MemTotal")
    lim, lim_src = cgroup_mem_limit(sys_root, proc_root)
    if lim is not None and (memtotal is None or lim < memtotal):
        mem, mem_source = lim, f"cgroup memory limit ({lim_src})"
    elif memtotal is not None:
        mem, mem_source = memtotal, "MemTotal (/proc/meminfo)"
    else:
        mem, mem_source = 0, "unknown (no MemTotal, no cgroup limit)"
    mem_mb = mem // 2**20

    per_run_mb = int(env.get("PER_RUN_MB") or DEFAULT_PER_RUN_MB)
    if env.get("RESERVE_MB"):
        reserve_mb = int(env["RESERVE_MB"])
    else:
        reserve_mb = max(MIN_RESERVE_MB, mem_mb // 10)
    by_cpu = usable - 1
    by_mem = (mem_mb - reserve_mb) // per_run_mb
    w = max(1, min(by_cpu, by_mem))
    limited_by = "cpu" if by_cpu <= by_mem else "memory"
    if min(by_cpu, by_mem) < 1:
        limited_by += " (floor of 1)"
    cap = env.get("MAX_WORKERS")
    if cap:
        if int(cap) < w:
            w, limited_by = max(1, int(cap)), "MAX_WORKERS"
    return {
        "workers": w,
        "limited_by": limited_by,
        "usable_cpus": usable,
        "cpu_source": cpu_source,
        "logical_cpus_in_affinity": logical,
        "physical_cores_in_affinity": phys,
        "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_quota": quota,
        "mem_mb": mem_mb,
        "mem_source": mem_source,
        "memtotal_mb": memtotal // 2**20 if memtotal is not None else None,
        "mem_available_mb": mi["MemAvailable"] // 2**20 if "MemAvailable" in mi else None,
        "cgroup_mem_limit_mb": lim // 2**20 if lim is not None else None,
        "reserve_mb": reserve_mb,
        "per_run_mb": per_run_mb,
        "workers_by_cpu": by_cpu,
        "workers_by_memory": by_mem,
        "max_workers_env": int(cap) if cap else None,
    }


def main() -> int:
    print(json.dumps(decide(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
