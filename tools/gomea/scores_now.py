"""Live scoreboard of the experiment queue: per phase, problem and arm, the finished runs and
running medians of the registered metric, plus what is running now.

    watch -n 60 .venv/bin/python tools/gomea/scores_now.py [PHASE ...]

Without PHASE it shows the server round's phases (`DEFAULT`, in run order) that have
results. Regression (pyMoses and stand-alone GP-GOMEA runs alike): test NMSE, train NMSE,
test - train, size, evaluations used and CPU per run; a GP-GOMEA run's size is its own node
count (`final_program_nodes`). Boolean: solved, CPU to solve (unsolved = inf), evaluations
to solve and CPU per run. Medians are of finished runs only, so they move.

Results root: `$GOMEA_RESULTS` if set, else `../gomea-results` next to the repository.
"""
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(os.environ.get("GOMEA_RESULTS")
           or Path(__file__).resolve().parents[3] / "gomea-results")
DEFAULT = ["phase1ah", "phase1ai", "phase1af-pilot", "phase1af", "phase1aj", "phase1ag",
           "phase1ae"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=1):
    if x is None:
        return "-"
    if math.isinf(x):
        return "inf"
    return f"{x:,.{nd}f}"


def size(r):
    """MOSES combo program: "(" plus atoms; GP-GOMEA (infix): its own node count."""
    if r.get("final_program_nodes") is not None:
        return r["final_program_nodes"]
    prog = r.get("final_program")
    return prog.count("(") + len(re.findall(r"[^\s()]+", prog)) if prog else None


def evals(r):
    """Evaluations used: pyMoses `n_evals_total` (seeded) or `n_evals`, GP-GOMEA
    `evaluations`."""
    for k in ("n_evals_total", "n_evals", "evaluations"):
        if r.get(k) is not None:
            return r[k]
    return None


def summaries(phase):
    files = glob.glob(str(RES / phase / "*" / "runs" / "*" / "summary.json"))
    files += glob.glob(str(RES / "1" / f"{phase}-bool-*" / "runs" / "*" / "summary.json"))
    for f in files:
        try:
            yield json.load(open(f))
        except (OSError, ValueError):
            continue


def table(phase):
    rows = defaultdict(list)
    for s in summaries(phase):
        rows[(s.get("task"), s.get("arm"))].append(s)
    if not rows:
        return
    print(f"\n== {phase} ==")
    for task in sorted({t for t, _ in rows}):
        arms = sorted(a for t, a in rows if t == task)
        reg = "test_nmse" in rows[(task, arms[0])][0]
        if reg:
            print(f"  {task:6} {'arm':18} {'runs':>4} {'fail':>4} {'test':>7} {'train':>7} "
                  f"{'gap':>6} {'size':>6} {'evals':>7} {'cpu s':>7}")
        else:
            print(f"  {task:6} {'arm':18} {'runs':>4} {'solved':>6} {'cpu-to-solve':>12} "
                  f"{'evals':>7} {'cpu s':>7}")
        for a in arms:
            rs = rows[(task, a)]
            ok = [r for r in rs if r.get("status") == "ok"]
            if reg:
                test = [r["test_nmse"] if r.get("status") == "ok" and r.get("test_nmse") is not None
                        else math.inf for r in rs]
                gap = [r["test_nmse"] - r["train_nmse"] for r in ok
                       if r.get("test_nmse") is not None and r.get("train_nmse") is not None]
                print(f"  {'':6} {a:18} {len(rs):>4} {len(rs) - len(ok):>4} {fmt(med(test)):>7} "
                      f"{fmt(med([r.get('train_nmse') for r in ok])):>7} {fmt(med(gap)):>6} "
                      f"{fmt(med([size(r) for r in ok]), 0):>6} "
                      f"{fmt(med([evals(r) for r in ok]), 0):>7} "
                      f"{fmt(med([r.get('cpu_s') for r in rs]), 0):>7}")
            else:
                solved = [r for r in rs if r.get("success")]
                cts = [r["cpu_to_target"] if r.get("success") else math.inf for r in rs]
                print(f"  {'':6} {a:18} {len(rs):>4} {len(solved):>6} {fmt(med(cts), 0):>12} "
                      f"{fmt(med([r.get('evals_to_target') for r in solved]), 0):>7} "
                      f"{fmt(med([r.get('cpu_s') for r in rs]), 0):>7}")
    for stage in (1, 2):
        c = RES / phase / f"report-stage{stage}" / "contrasts.csv"
        if c.exists():
            print(f"  stage {stage} verdicts:")
            lines = c.read_text().splitlines()
            head = lines[0].split(",")
            for line in lines[1:]:
                d = dict(zip(head, line.split(",")))
                print(f"    {d['problem']:6} {d['a']} vs {d['b']} ({d['family']}): "
                      f"{d['a_lower']}/{d['b_lower']} lower, Holm p {float(d['p_holm']):.3g} "
                      f"-> {d['verdict']}")


def running():
    out = subprocess.run(["ps", "-eo", "etime=,args="], capture_output=True, text=True).stdout
    procs = [l.strip() for l in out.splitlines()
             if re.search(r"queue_run\.sh|handover_1af|phase_run\.sh|moses_sr_runner\.py "
                          r"|gpgomea_sr_runner\.py |benchmarks\.gomea\.runner", l)]
    n = sum("spawn_main" in l for l in out.splitlines())
    print("running:" if procs else "running: no experiment")
    for p in procs:
        print("  " + re.sub(r"\S*/", "", p)[:110])
    if procs:
        print(f"  ({n} worker processes)")


if __name__ == "__main__":
    running()
    for p in sys.argv[1:] or DEFAULT:
        table(p)
