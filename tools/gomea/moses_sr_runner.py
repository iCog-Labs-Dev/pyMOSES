"""Run MOSES (classic C++ or pyMoses) on the Phase 1f regression datasets.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/moses_sr_runner.py MANIFEST OUTPUT_ROOT [--workers N]
        [--only REGEX]

A manifest (JSON) has these keys:

* `name`, `datasets_dir` (built by `sr_datasets.py`), `datasets`, `splits`,
  `wall_cap_s`, and optionally `mem_limit_mb` (RLIMIT_AS per run);
* `arms`: `{arm: {"binary": path, "argv": [extra MOSES flags], "budget": {"evals": N} |
  {"time_s": T}}}`.

**MOSES regression setup** (Phase 1f; see `PHASE1F-PREREGISTRATION.md`). MOSES scores a
regression candidate by raw squared error, while its deme trimming and exemplar selection
windows scale with the complexity temperature `-v`. So, per run, the runner:

1. rescales the training target to y' = 10 (y - mean) / sd, an affine map that leaves
   linearly-scaled NMSE unchanged, and writes it to the run directory;
2. runs `binary -i train_scaled.csv -u y -r SEED -n sin -n exp -n log -v 60*n_train -c 10
   -V1 -l WARN -f RUN_DIR/moses.log <budget> <arm argv>`; the budget is `-m N`, or
   `--max-time T -m 1000000000`;
3. scores each printed candidate with `sr_eval.py` on the ORIGINAL train, val and test
   splits (linear scaling fitted on train), and selects the candidate with the lowest
   validation NMSE. The summary reports that candidate (`test_nmse`, ...) and also the
   first, best-training-score candidate (`first_*`).

The MOSES seed is the split number. Runs are resumable: an existing `summary.json` is
kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_eval  # noqa: E402


def _scale_train(src: Path, dst: Path) -> tuple[int, float, float]:
    with src.open() as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    k = header.index("y")
    ys = [float(r[k]) for r in body]
    mean = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    with dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in body:
            r = list(r)
            r[k] = repr(10.0 * (float(r[k]) - mean) / sd)
            w.writerow(r)
    return len(body), mean, sd


def _limit(mem_mb):
    def fn():
        if mem_mb:
            lim = int(mem_mb) * 2**20
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    return fn


def _run_one(job: dict) -> dict:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    sdir = Path(job["split_dir"])
    n_train, mean, sd = _scale_train(sdir / "train.csv", run_dir / "train_scaled.csv")
    argv = [job["binary"], "-i", str(run_dir / "train_scaled.csv"), "-u", "y"]
    argv += ["-r", str(job["seed"]), "-n", "sin", "-n", "exp", "-n", "log"]
    argv += ["-v", str(60 * n_train), "-c", "10", "-V1", "-l", "WARN"]
    argv += ["-f", str(run_dir / "moses.log")]
    b = job["budget"]
    argv += ["-m", str(b["evals"])] if "evals" in b else ["--max-time", str(b["time_s"])]
    if "evals" not in b:
        argv += ["-m", "1000000000"]
    argv += job["argv"]
    (run_dir / "config.json").write_text(json.dumps({**job, "resolved_argv": argv}, indent=2))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    status, exit_code, error = "ok", None, None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=job["wall_cap_s"],
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
            cwd=run_dir,
            preexec_fn=_limit(job.get("mem_limit_mb")),
        )
        out, exit_code = proc.stdout, proc.returncode
        (run_dir / "stderr.txt").write_text(proc.stderr)
        if exit_code != 0:
            status, error = "error", (proc.stderr.strip().splitlines() or ["?"])[-1][:300]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = "timeout"
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (run_dir / "stdout.txt").write_text(out)
    n_evals = re.search(r"^n_evals:\s*(\d+)", out, re.M)
    cands = sr_eval.parse_moses_output(out) if status == "ok" else []
    train = sr_eval.read_table(str(sdir / "train.csv"))
    val = sr_eval.read_table(str(sdir / "val.csv"))
    test = sr_eval.read_table(str(sdir / "test.csv"))
    scored = []
    for _score, prog in cands:
        on_val = sr_eval.evaluate(prog, train, val)
        on_test = sr_eval.evaluate(prog, train, test)
        scored.append(
            {
                "program": prog,
                "train_nmse": on_val["train"]["nmse_scaled"],
                "val_nmse": on_val["test"]["nmse_scaled"],
                "test_nmse": on_test["test"]["nmse_scaled"],
            }
        )
    (run_dir / "candidates.json").write_text(json.dumps(scored, indent=2))
    if status == "ok" and not scored:
        status, error = "error", "no candidate parsed"
    best = min(scored, key=lambda c: c["val_nmse"]) if scored else {}
    first = scored[0] if scored else {}
    summary = {
        "run_id": job["run_id"],
        "task": job["task"],
        "arm": job["arm"],
        "seed": job["seed"],
        "argv": argv,
        "status": status,
        "error": error,
        "exit_code": exit_code,
        "wall_s": wall,
        "cpu_s": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
        "ru_maxrss_kb": after.ru_maxrss,
        "n_evals": int(n_evals.group(1)) if n_evals else None,
        "n_train": n_train,
        "target_mean": mean,
        "target_sd": sd,
        "n_candidates": len(scored),
        "train_nmse": best.get("train_nmse"),
        "val_nmse": best.get("val_nmse"),
        "test_nmse": best.get("test_nmse"),
        "final_program": best.get("program"),
        "first_train_nmse": first.get("train_nmse"),
        "first_test_nmse": first.get("test_nmse"),
    }
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def expand(manifest: dict, out_root: Path, repo: Path) -> list[dict]:
    ddir = (repo / manifest["datasets_dir"]).resolve()
    jobs = []
    for ds in manifest["datasets"]:
        for arm, a in manifest["arms"].items():
            binary = a["binary"] if os.path.isabs(a["binary"]) else str(repo / a["binary"])
            for seed in manifest["splits"]:
                run_id = f"{ds}__{arm}__s{seed}"
                jobs.append(
                    {
                        "run_id": run_id,
                        "task": ds,
                        "arm": arm,
                        "seed": seed,
                        "binary": str(Path(binary).absolute()),
                        "argv": a.get("argv", []),
                        "budget": a["budget"],
                        "split_dir": str(ddir / ds / "splits" / f"s{seed}"),
                        "wall_cap_s": manifest["wall_cap_s"],
                        "mem_limit_mb": manifest.get("mem_limit_mb"),
                        "run_dir": str(out_root / manifest["name"] / "runs" / run_id),
                    }
                )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output_root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="run only run ids matching this regex")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.output_root).resolve()
    jobs = expand(manifest, out_root, repo)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["run_id"])]
    (out_root / manifest["name"]).mkdir(parents=True, exist_ok=True)
    (out_root / manifest["name"] / "manifest.json").write_text(json.dumps(manifest, indent=2))
    done = 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for s in pool.imap_unordered(_run_one, jobs):
            done += 1
            print(
                f"[{done}/{len(jobs)}] {s['run_id']}: {s['status']} evals={s['n_evals']} "
                f"train={s['train_nmse']} test={s['test_nmse']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
