"""Phase 1w deviation (2026-10-19): stop the mux11 runner once the primary verdict is decided.

The registered mux11 metric is CPU seconds to solve (unsolved = +inf). TOPK-L runs cost about
5x GOMEA's CPU per evaluation, and would run up to the 36,000 s wall cap. As soon as
* every GOMEA run has a summary, and
* every TOPK-L run either has a summary or has already spent more CPU than its paired GOMEA
  run needed to solve,
every pair's sign is fixed, so the Wilcoxon signs, and the medians' order, can no longer
change. The script then writes `earlystop.json` (each pair's status and CPU lower bound) and
terminates the mux11 runner's child runs and then the runner, so `phase_run.sh` goes on to the Con part.

Usage: phase1w_earlystop.py RESULT_DIR RUNNER_PID   (polls every 60 s)
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def last_cpu(run_dir: Path) -> float:
    cpu = 0.0
    trace = run_dir / "trace.jsonl"
    if trace.exists():
        for line in trace.open():
            try:
                cpu = max(cpu, float(json.loads(line).get("cpu") or 0))
            except (ValueError, json.JSONDecodeError):
                pass
    return cpu


def summary(run_dir: Path):
    p = run_dir / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ppid(p: int) -> int:
    try:
        return int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return -1


def main() -> int:
    runs, pid = Path(sys.argv[1]) / "runs", int(sys.argv[2])
    seeds = range(3001, 3011)
    while True:
        g = {s: summary(runs / f"mux11__GOMEA__s{s}") for s in seeds}
        state, decided = {}, all(v is not None for v in g.values())
        for s in seeds:
            t = summary(runs / f"mux11__TOPK-L__s{s}")
            g_cpu = g[s].get("cpu_to_target") if g[s] else None
            if t is not None:
                state[s] = {"topk": "done", "topk_cpu_to_target": t.get("cpu_to_target"),
                            "gomea_cpu_to_target": g_cpu}
                continue
            lb = last_cpu(runs / f"mux11__TOPK-L__s{s}")
            ok = g_cpu is not None and lb > g_cpu
            state[s] = {"topk": "running", "topk_cpu_lower_bound": lb,
                        "gomea_cpu_to_target": g_cpu, "decided": ok}
            decided = decided and ok
        if decided:
            (Path(sys.argv[1]) / "earlystop.json").write_text(json.dumps(
                {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pairs": state}, indent=2) + "\n")
            print("decided; stopping runner", pid, flush=True)
            # SIGINT is ignored by background jobs of a non-interactive shell, so terminate
            # the runner's child processes (its runs), then the runner itself.
            kids = [int(d) for d in os.listdir("/proc") if d.isdigit()
                    and _ppid(int(d)) == pid]
            for k in kids:
                os.kill(k, signal.SIGTERM)
            time.sleep(5)
            os.kill(pid, signal.SIGTERM)
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
