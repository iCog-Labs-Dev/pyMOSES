"""Phase 1aa deviation (2026-10-20): end -B 3 runs whose pair is already decided.

The registered mux11 metric is CPU seconds to solve (unsolved = +inf). -B 3 runs cost up to
about 1 s of CPU per evaluation and held every worker slot, so 12 runs could not start for
many hours. A -B 3 run whose CPU already exceeds its paired base run's CPU to solve has a
fixed sign: it is slower whatever happens next. Such a run is terminated (SIGTERM to its own
process). The runner stays alive, records it as `error` (unsolved, the outcome the rule
assigns) and starts the next queued run in the freed slot. Every action is logged.

Usage: phase1aa_earlystop.py RESULT_DIR RUNNER_PID   (polls every 60 s until the runner exits)
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

PAIRS = {"HC-B3": "HC", "GOMEA-B3": "GOMEA"}


def ppid(p: int) -> int:
    try:
        return int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return -1


def run_of(p: int, runs: Path):
    try:
        for fd in os.listdir(f"/proc/{p}/fd"):
            t = os.readlink(f"/proc/{p}/fd/{fd}")
            if t.startswith(str(runs) + "/"):
                return t[len(str(runs)) + 1:].split("/")[0]
    except OSError:
        pass
    return None


def last_cpu(d: Path) -> float:
    cpu = 0.0
    try:
        for line in (d / "trace.jsonl").open():
            try:
                cpu = max(cpu, float(json.loads(line).get("cpu") or 0))
            except (ValueError, json.JSONDecodeError):
                pass
    except OSError:
        pass
    return cpu


def main() -> int:
    runs, runner = Path(sys.argv[1]) / "runs", int(sys.argv[2])
    done = set()
    while os.path.exists(f"/proc/{runner}"):
        kids = [int(x) for x in os.listdir("/proc") if x.isdigit() and ppid(int(x)) == runner]
        for k in kids:
            rid = run_of(k, runs)
            if not rid or rid in done:
                continue
            task, arm, seed = rid.split("__")
            base = PAIRS.get(arm)
            if base is None:
                continue
            bs = runs / f"{task}__{base}__{seed}" / "summary.json"
            if not bs.exists():
                continue
            b_cpu = json.loads(bs.read_text()).get("cpu_to_target")
            lb = last_cpu(runs / rid)
            if b_cpu is not None and lb > b_cpu:
                print(f"{time.strftime('%H:%M:%S')} end {rid} pid {k}: cpu {lb:.0f} > "
                      f"{base} {b_cpu:.0f}", flush=True)
                os.kill(k, signal.SIGTERM)
                done.add(rid)
        time.sleep(60)
    print("runner exited", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
