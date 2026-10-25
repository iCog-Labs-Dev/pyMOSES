"""Describe this machine and checkout as JSON (server-run/environment.json).

    python3 tools/server/envinfo.py [OUT.json]

hostname, /etc/os-release, kernel, CPU model, logical and physical CPU counts, the cgroup
CPU and memory limits, MemTotal, the venv's Python version, git HEAD and
`git status --porcelain`, `pip freeze` of the venv, the uv version, and the worker
decision of resources.py. Standard library only; every field is best effort.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resources  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _cmd(*argv, cwd=None) -> str | None:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=120)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def os_release() -> dict:
    out = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            k, _, v = line.partition("=")
            if k:
                out[k] = v.strip().strip('"')
    except OSError:
        pass
    return out


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def collect() -> dict:
    venv_py = REPO / ".venv" / "bin" / "python"
    uv = os.environ.get("UV_BIN") or shutil.which("uv")
    if not uv:  # setup.sh's pinned copy under the results tree
        found = sorted(glob.glob(str(REPO.parent / "gomea-results" / "toolchain" / "uv-*" / "uv")))
        uv = found[-1] if found else "uv"
    res = resources.decide()
    return {
        "hostname": socket.gethostname(),
        "os_release": os_release(),
        "kernel": platform.release(),
        "uname": " ".join(platform.uname()),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "logical_cpus_in_affinity": res["logical_cpus_in_affinity"],
        "physical_cores_in_affinity": res["physical_cores_in_affinity"],
        "cgroup_cpu_quota": res["cgroup_cpu_quota"],
        "cgroup_mem_limit_mb": res["cgroup_mem_limit_mb"],
        "memtotal_mb": res["memtotal_mb"],
        "resources": res,
        "python": _cmd(str(venv_py), "-c", "import sys; print(sys.version)")
        if venv_py.exists() else None,
        "python_executable": _cmd(str(venv_py), "-c", "import sys; print(sys.base_prefix)")
        if venv_py.exists() else None,
        "git_head": _cmd("git", "rev-parse", "HEAD", cwd=REPO),
        "git_branch": _cmd("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=REPO),
        "git_status_porcelain": _cmd("git", "status", "--porcelain", cwd=REPO),
        "pip_freeze": (_cmd(str(venv_py), "-m", "pip", "freeze", "--all") or "").splitlines()
        if venv_py.exists() else None,
        "uv_version": _cmd(uv, "--version"),
        "repo": str(REPO),
        "results": str((REPO.parent / "gomea-results").resolve()),
    }


def main() -> int:
    info = collect()
    text = json.dumps(info, indent=1, sort_keys=True) + "\n"
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
