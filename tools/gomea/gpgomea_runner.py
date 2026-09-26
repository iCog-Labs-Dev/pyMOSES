"""Run stand-alone GP-GOMEA (the authors' C++ code) on the Phase 1b Boolean tasks.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/gpgomea_runner.py MANIFEST OUTPUT_ROOT [--workers N]

A manifest (JSON) has these keys:

* `name`, `binary`, `budget`, `seeds`, `wall_cap_s`;
* `tasks`: `{task: {"data": path, "rows": int}}`;
* `arms`: `{arm: {"argv": [GP-GOMEA flags]}}`.

Every run executes `binary --prob accuracy --train D --test D --evaluations BUDGET
--generations -1 --seed S --parallel 1 <arm argv>` as its own process. The command's
stdout is stored and parsed into `runs/<run_id>/summary.json`. The summary uses the same
field names as the MOSES harness (`benchmarks/gomea/runner.py`), so both can be analysed
side by side.

Budget semantics:

* GP-GOMEA checks its evaluation limit only between generations, so a run can overshoot.
* The instrumented binary prints the cumulative evaluation count on each generation line.
* Every metric is read at the **last generation boundary within the budget**:
  * `best_raw` is minus the wrong-row count;
  * `success` means a zero-error elitist within budget;
  * `evals_to_target` is the cumulative evaluations at the first such generation.

  This is conservative against GP-GOMEA by less than one generation.
* A run that never completes a generation within the budget has `best_raw = None`, and
  the analysis counts it as the worst error.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

GEN_RE = re.compile(r"> generation (\d+) - best fit: ([0-9.eE+-]+|inf|nan) - evaluations: (\d+)")


def parse_stdout(text: str, rows: int, budget: int) -> dict:
    gens = [(int(g), float(f), int(e)) for g, f, e in GEN_RE.findall(text)]
    within = [(g, f, e) for g, f, e in gens if e <= budget]
    best = min((f for _, f, _ in within), default=None)
    hit = next(((g, e) for g, f, e in within if f == 0.0), None)
    total = re.search(r"^Evaluations:\s*(\d+)", text, re.M)
    nodes = re.search(r"^Number of nodes:\s*(\d+)", text, re.M)
    prog = re.search(r"^Best solution found:\s*(.*)$", text, re.M)
    return {
        "generations_logged": len(gens),
        "generations_within_budget": len(within),
        "best_error_rate": best,
        "best_raw": (-round(best * rows)) if best is not None else None,
        "success": hit is not None,
        "evals_to_target": hit[1] if hit else None,
        "generation_of_target": hit[0] if hit else None,
        "n_calls": int(total.group(1)) if total else None,
        "evals_within_budget": within[-1][2] if within else 0,
        "final_program": prog.group(1).strip() if prog else None,
        "final_program_nodes": int(nodes.group(1)) if nodes else None,
    }


def _run_one(job: dict) -> dict:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    argv = job["argv"]
    (run_dir / "config.json").write_text(json.dumps(job, indent=2))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    status, exit_code = "ok", None
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=job["wall_cap_s"],
                              env={**os.environ, "OMP_NUM_THREADS": "1"})
        out, exit_code = proc.stdout, proc.returncode
        if exit_code != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = "timeout"
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (run_dir / "stdout.txt").write_text(out)
    summary = {
        "run_id": job["run_id"], "task": job["task"], "arm": job["arm"], "seed": job["seed"],
        "argv": argv, "status": status, "exit_code": exit_code, "wall_s": wall,
        "cpu_s": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
        "ru_maxrss_kb": after.ru_maxrss,
        "budget": job["budget"], "rows": job["rows"],
        **parse_stdout(out, job["rows"], job["budget"]),
    }
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def expand(manifest: dict, out_root: Path, repo: Path) -> list[dict]:
    jobs = []
    for task, t in manifest["tasks"].items():
        data = str((repo / t["data"]).resolve())
        for arm, a in manifest["arms"].items():
            for seed in manifest["seeds"]:
                run_id = f"{task}__{arm}__s{seed}"
                argv = [str((repo / manifest["binary"]).resolve()) if not os.path.isabs(manifest["binary"])
                        else manifest["binary"],
                        "--prob", "accuracy", "--train", data, "--test", data,
                        "--evaluations", str(manifest["budget"]), "--generations", "-1",
                        "--seed", str(seed), "--parallel", "1", *a["argv"]]
                jobs.append({"run_id": run_id, "task": task, "arm": arm, "seed": seed,
                             "argv": argv, "rows": t["rows"], "budget": manifest["budget"],
                             "wall_cap_s": manifest["wall_cap_s"],
                             "run_dir": str(out_root / manifest["name"] / "runs" / run_id)})
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output_root")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.output_root).resolve()
    jobs = expand(manifest, out_root, repo)
    (out_root / manifest["name"]).mkdir(parents=True, exist_ok=True)
    (out_root / manifest["name"] / "manifest.json").write_text(json.dumps(manifest, indent=2))
    done = 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for s in pool.imap_unordered(_run_one, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {s['run_id']}: {s['status']} best_raw={s['best_raw']} "
                  f"success={s['success']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
