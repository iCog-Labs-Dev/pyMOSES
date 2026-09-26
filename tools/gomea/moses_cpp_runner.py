"""Run classic C++ MOSES (the reference binary) on demo Boolean problems.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/moses_cpp_runner.py MANIFEST OUTPUT_ROOT [--workers N]

Manifest keys: `name`, `binary`, `binary_sha256`, `budget`, `seeds`, `wall_cap_s`,
`tasks` (`{task: {"argv": ["-H", "pa", "-k", "4"], "rows": 16}}`) and `arms`
(`{arm: {"argv": [...]}}`).

Each run is `binary <task argv> <arm argv> -m BUDGET -r SEED -V1 -c1 -l WARN -f moses.log`,
executed as its own process inside its own run directory.

The summaries use the same fields as `tools/gomea/gpgomea_runner.py`:

* Classic MOSES stops as soon as the best score reaches the maximum (0 wrong rows for these
  problems), and `-V1` prints the evaluations it used (`n_evals`).
* On a solved run, `evals_to_target` is therefore `n_evals`, measured at deme granularity.
* `best_raw` is the raw score of the top printed candidate, i.e. minus the wrong rows.
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

LINE_RE = re.compile(r"^(-?[0-9.eE+-]+)\s+(\S.*)$")


def parse_stdout(text: str, budget: int) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    best = next((LINE_RE.match(ln) for ln in lines if LINE_RE.match(ln)), None)
    n = re.search(r"^n_evals:\s*(\d+)", text, re.M)
    n_calls = int(n.group(1)) if n else None
    raw = float(best.group(1)) if best else None
    success = raw is not None and raw >= 0 and n_calls is not None and n_calls <= budget
    return {
        "best_raw": raw,
        "success": success,
        "evals_to_target": n_calls if success else None,
        "n_calls": n_calls,
        "final_program": best.group(2).strip() if best else None,
    }


def _run_one(job: dict) -> dict:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    (run_dir / "config.json").write_text(json.dumps(job, indent=2))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    status, exit_code = "ok", None
    try:
        proc = subprocess.run(
            job["argv"],
            capture_output=True,
            text=True,
            timeout=job["wall_cap_s"],
            cwd=run_dir,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
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
        "run_id": job["run_id"],
        "task": job["task"],
        "arm": job["arm"],
        "seed": job["seed"],
        "argv": job["argv"],
        "status": status,
        "exit_code": exit_code,
        "wall_s": wall,
        "cpu_s": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
        "ru_maxrss_kb": after.ru_maxrss,
        "budget": job["budget"],
        "rows": job["rows"],
        **parse_stdout(out, job["budget"]),
    }
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def expand(manifest: dict, out_root: Path) -> list[dict]:
    jobs = []
    for task, t in manifest["tasks"].items():
        for arm, a in manifest["arms"].items():
            for seed in manifest["seeds"]:
                run_id = f"{task}__{arm}__s{seed}"
                run_dir = out_root / manifest["name"] / "runs" / run_id
                argv = [
                    manifest["binary"],
                    *t["argv"],
                    *a["argv"],
                    "-m",
                    str(manifest["budget"]),
                    "-r",
                    str(seed),
                    "-V1",
                    "-c1",
                    "-l",
                    "WARN",
                    "-f",
                    str(run_dir / "moses.log"),
                ]
                jobs.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "arm": arm,
                        "seed": seed,
                        "argv": argv,
                        "rows": t["rows"],
                        "budget": manifest["budget"],
                        "wall_cap_s": manifest["wall_cap_s"],
                        "run_dir": str(run_dir),
                    }
                )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output_root")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.output_root).resolve()
    jobs = expand(manifest, out_root)
    (out_root / manifest["name"]).mkdir(parents=True, exist_ok=True)
    (out_root / manifest["name"] / "manifest.json").write_text(json.dumps(manifest, indent=2))
    done = 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for s in pool.imap_unordered(_run_one, jobs):
            done += 1
            print(
                f"[{done}/{len(jobs)}] {s['run_id']}: {s['status']} best_raw={s['best_raw']} "
                f"success={s['success']} n_evals={s['n_calls']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
