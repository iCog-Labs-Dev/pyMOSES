"""Run stand-alone GP-GOMEA (the authors' C++ code) on the Phase 1f regression datasets.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/gpgomea_sr_runner.py MANIFEST OUTPUT_ROOT [--workers N]
        [--only REGEX]

A manifest (JSON) has these keys:

* `name`, `datasets_dir` (the `phase1f/datasets` folder built by `sr_datasets.py`),
  `datasets` (abbreviations), `splits` (split seeds), `wall_cap_s`;
* `arms`: `{arm: {"binary": path, "argv": [GP-GOMEA flags], "train": "train" |
  "trainval"}}`. With `"trainval"`, the run gets `trainval.dat` and
  `--validation <gpgomea_validation_fraction>` from `splits.json`. GP-GOMEA then draws its
  own validation rows at random (seeded by `--seed`), so they are not our `val` rows.

Every run executes `binary --prob symbreg --train T --test test.dat --seed S --parallel 1
<arm argv>` in its own directory (GP-GOMEA appends `result.txt` to its cwd). The GP-GOMEA
seed is the split seed, so the arms of one split share a seed.

GP-GOMEA prints linearly-scaled MSEs (the scaling is fitted on the training rows). The
summary converts them to the paper's NMSE, 100 * MSE / var(y), with the population
variance of that split's targets: `train_nmse`, `val_nmse` (GP-GOMEA's internal validation
rows; the variance used is that of the rows it was trained on plus held out, i.e. the
train file) and `test_nmse`.
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

_FIELDS = {
    "train_mse": r"^Train fit:\s*(\S+)",
    "val_mse": r"^Validation fit:\s*(\S+)",
    "test_mse": r"^Test fit:\s*(\S+)",
    "running_time_s": r"^Running time:\s*(\S+)",
    "evaluations": r"^Evaluations:\s*(\S+)",
    "final_program_nodes": r"^Number of nodes:\s*(\S+)",
}


def _var(path: Path) -> float:
    ys = [float(line.split()[-1]) for line in path.read_text().splitlines() if line.strip()]
    m = sum(ys) / len(ys)
    return sum((y - m) ** 2 for y in ys) / len(ys)


def parse_stdout(text: str) -> dict:
    out: dict = {}
    for key, pat in _FIELDS.items():
        m = re.search(pat, text, re.M)
        out[key] = float(m.group(1)) if m else None
    prog = re.search(r"^Best solution found:\s*(.*)$", text, re.M)
    out["final_program"] = prog.group(1).strip() if prog else None
    return out


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
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            cwd=run_dir,
        )
        out, exit_code = proc.stdout, proc.returncode
        (run_dir / "stderr.txt").write_text(proc.stderr)
        if exit_code != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = "timeout"
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (run_dir / "stdout.txt").write_text(out)
    parsed = parse_stdout(out)
    if status == "ok" and parsed["test_mse"] is None:
        status = "error"
    var_train, var_test = _var(Path(job["train_file"])), _var(Path(job["test_file"]))

    def nmse(mse, var):
        return None if mse is None else 100.0 * mse / var

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
        **parsed,
        "train_nmse": nmse(parsed["train_mse"], var_train),
        "val_nmse": nmse(parsed["val_mse"], var_train),
        "test_nmse": nmse(parsed["test_mse"], var_test),
    }
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def expand(manifest: dict, out_root: Path, repo: Path) -> list[dict]:
    ddir = (repo / manifest["datasets_dir"]).resolve()
    jobs = []
    for ds in manifest["datasets"]:
        meta = json.loads((ddir / ds / "splits" / "splits.json").read_text())
        for arm, a in manifest["arms"].items():
            binary = a["binary"] if os.path.isabs(a["binary"]) else str(repo / a["binary"])
            for seed in manifest["splits"]:
                sdir = ddir / ds / "splits" / f"s{seed}"
                train = sdir / f"{a.get('train', 'train')}.dat"
                argv = [str(Path(binary).resolve()), "--prob", "symbreg", "--train", str(train)]
                argv += ["--test", str(sdir / "test.dat"), "--seed", str(seed)]
                argv += ["--parallel", "1", *a["argv"]]
                if a.get("train") == "trainval":
                    argv += ["--validation", repr(meta["gpgomea_validation_fraction"])]
                run_id = f"{ds}__{arm}__s{seed}"
                jobs.append(
                    {
                        "run_id": run_id,
                        "task": ds,
                        "arm": arm,
                        "seed": seed,
                        "argv": argv,
                        "train_file": str(train),
                        "test_file": str(sdir / "test.dat"),
                        "wall_cap_s": manifest["wall_cap_s"],
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
                f"[{done}/{len(jobs)}] {s['run_id']}: {s['status']} "
                f"train={s['train_nmse']} test={s['test_nmse']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
