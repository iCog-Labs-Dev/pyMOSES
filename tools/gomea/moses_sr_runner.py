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
regression candidate by raw squared error (pyMoses: of the least-squares
scaled a + b*f instead, if the arm passes `--linear-scaling 1`), while its deme
trimming and exemplar selection windows scale with the complexity temperature `-v`.
So, per run, the runner:

1. rescales the training target to y' = 10 (y - mean) / sd, an affine map that leaves
   linearly-scaled NMSE unchanged, and writes it to the run directory;
2. runs `binary -i train_scaled.csv -u y -r SEED -n sin -n exp -n log -v 60*n_train -c 10
   -V1 -l WARN -f RUN_DIR/moses.log <budget> <arm argv>`; the budget is `-m N`, or
   `--max-time T -m 1000000000`;
3. scores each printed candidate with `sr_eval.py` on the ORIGINAL train, val and test
   splits (linear scaling fitted on train), and selects the candidate with the lowest
   validation NMSE. The summary reports that candidate (`test_nmse`, ...) and also the
   first, best-training-score candidate (`first_*`).

**Telemetry.** A pyMoses GOMEA arm (`-a gomea` or `-a hc-gomea` in its argv) also gets
`--gomea-log RUN_DIR/events.jsonl`, as the Boolean runner (`benchmarks.gomea.manifest`)
does, so each run writes the per-deme "invocation_start", "generation" and "end" events
(`events_argv`). It is on by default; `"events": false` on an arm turns it off. The log is a
side file: MOSES's output is the same with or without it.

The MOSES seed is the split number. Runs are resumable: an existing `summary.json` is
kept.

**Seeding (backlog item 9).** An arm may carry a `seed` block, `{"binary": gpgomea-seed,
"share": 0.2 | "budget": N, "height": 3, "k": 5, "popsize": .., "functions": ..,
"gpgomea_argv": [..], "with_empty": true}` (evaluation budgets only). The runner then runs
`gpgomea_seed.make_seeds` on `train_scaled.csv` (the file MOSES reads, so the seeds' linear
scaling fits the rescaled target) with `share * N` evaluations and the split number as
GP-GOMEA's seed, gives MOSES `-m (N - used)` and `-e SEED ... -e +`, and writes
`seeds.json`. The summary's `n_evals_total` and `cpu_s` include GP-GOMEA's evaluations and
CPU time (`n_evals`, `moses_cpu_s` are MOSES's own); `seed_*_nmse` report the best seed
(by training score) on the original splits; with pyMoses, `lineage.json`
(`seed_lineage.py`) gives the lineage of every program left in the metapopulation and
`final_lineage` that of the reported program.

Two opt-in seed-block keys (pyMoses only; see `docs/gomea/USAGE.md`):

* `"reserve_empty_every": K` passes `--reserve-empty-every K`: every K-th expansion picks
  its exemplar from the empty program's lineage. The summary's `reserve_empty` gives the
  expansions per lineage and the reserved turns as pyMoses counted them.
* `"reseed_after": S` (with optional `"reseed_share": f`, default 0.2, and
  `"max_reseeds": n`, default 1) installs `reseed.py`: after S expansions without a
  strict improvement of the best score, a fresh GP-GOMEA run (new seed) gets
  `f * (evaluations MOSES has left)` and its elites join the metapopulation. Its
  evaluations are added to MOSES's own count, so they come out of `-m` and `n_evals`
  includes them; the summary adds `n_reseeds`, `reseed_evals`, `reseed_cpu_s` and
  `moses_own_evals`, and `reseed.json` records each re-seed. A re-seed whose share is
  below `"reseed_min_budget"` (default: the initial seeding's budget, since GP-GOMEA
  overspends a small share by up to a generation) is skipped.
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

REPO = Path(__file__).resolve().parents[2]


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


def moses_argv(job: dict, run_dir: Path, n_train: int) -> list[str]:
    """The MOSES command line of one run (before seeding rewrites `-m`): the fixed
    regression setup, the budget, the arm's argv, then `--gomea-log RUN_DIR/events.jsonl`
    when `events_argv` asks for it."""
    argv = [job["binary"], "-i", str(run_dir / "train_scaled.csv"), "-u", "y"]
    argv += ["-r", str(job["seed"]), "-n", "sin", "-n", "exp", "-n", "log"]
    argv += ["-v", str(60 * n_train), "-c", "10", "-V1", "-l", "WARN"]
    argv += ["-f", str(run_dir / "moses.log")]
    b = job["budget"]
    argv += ["-m", str(b["evals"])] if "evals" in b else ["--max-time", str(b["time_s"])]
    if "evals" not in b:
        argv += ["-m", "1000000000"]
    argv += job["argv"]
    return argv + events_argv(job, run_dir)


def _selects_gomea(argv: list[str]) -> bool:
    """Whether `argv` selects a GOMEA optimizer (`-a`/`--opt-algo` gomea or hc-gomea),
    as `benchmarks.gomea.manifest.is_gomea_arm` decides it for the Boolean runner."""
    algos = ("gomea", "hc-gomea")
    for i, tok in enumerate(argv):
        tok = str(tok)
        if tok in ("-a", "--opt-algo") and i + 1 < len(argv) and str(argv[i + 1]) in algos:
            return True
        if tok.startswith("--opt-algo=") and tok.split("=", 1)[1] in algos:
            return True
        if tok.startswith("-a") and len(tok) > 2 and tok[2:] in algos:
            return True
    return False


def events_argv(job: dict, run_dir: Path) -> list[str]:
    """`["--gomea-log", RUN_DIR/events.jsonl]` for a pyMoses GOMEA arm, as the Boolean
    runner passes it, so every regression run writes the same per-deme telemetry
    ("invocation_start", "generation", "end"). On by default; an arm turns it off with
    `"events": false`, and an arm that already passes `--gomea-log` keeps its own."""
    argv = [str(x) for x in job["argv"]]
    if not job.get("events", True) or not _is_pymoses(job["binary"]):
        return []
    if not _selects_gomea(argv) or any(
        x == "--gomea-log" or x.startswith("--gomea-log=") for x in argv
    ):
        return []
    return ["--gomea-log", str(run_dir / "events.jsonl")]


def _run_one(job: dict) -> dict:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    sdir = Path(job["split_dir"])
    n_train, mean, sd = _scale_train(sdir / "train.csv", run_dir / "train_scaled.csv")
    argv = moses_argv(job, run_dir, n_train)
    seed_rec = None
    if job.get("seed_spec"):
        try:
            seed_rec, argv = _apply_seed(job, argv, run_dir)
        except Exception as exc:  # never let one failed seeding kill the pool
            summary = {
                "run_id": job["run_id"], "task": job["task"], "arm": job["arm"],
                "seed": job["seed"], "status": "error",
                "error": f"seeding: {type(exc).__name__}: {exc}"[:300],
                "test_nmse": None, "train_nmse": None, "val_nmse": None, "n_evals": None,
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            return summary
    is_pymoses = _is_pymoses(job["binary"])
    if is_pymoses:
        # pyMoses: run it with a raised recursion limit. Its recursive neighbourhood
        # counting nests once per knob, past Python's default of 1,000 on large
        # representations (Dow). The limit only decides whether a run can finish.
        lineage = ""
        if seed_rec is not None:
            labels = [f"seed{i + 1}" for i in range(len(seed_rec["seeds"]))]
            labels += ["empty"] if seed_rec["with_empty"] else []
            lineage = (
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
                "import atexit, seed_lineage; "
                f"atexit.register(seed_lineage.install({labels!r}).write, "
                f"{str(run_dir / 'lineage.json')!r}); "
            )
        if seed_rec is not None and job["seed_spec"].get("reseed_after"):
            # a re-seed is never given less than the initial seeding was
            spec = {"reseed_min_budget": seed_rec["seed_budget"], **job["seed_spec"]}
            binary = spec["binary"] if os.path.isabs(spec["binary"]) else str(REPO / spec["binary"])
            lineage += (
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
                "import reseed; "
                f"atexit.register(reseed.install({spec!r}, {binary!r}, "
                f"{str(run_dir / 'gpgomea')!r}, {int(job['seed'])!r}, "
                f"train_csv={str(run_dir / 'train_scaled.csv')!r}).write, "
                f"{str(run_dir / 'reseed.json')!r}); "
            )
        boot = (
            "import sys; sys.setrecursionlimit(100000); "
            + lineage
            + "from pymoses.moses.main.moses_exec import moses_exec; "
            "sys.exit(moses_exec(['moses', *sys.argv[1:]]))"
        )
        python = str(Path(job["binary"]).with_name("python"))
        argv = [python, "-c", boot, *argv[1:]]
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
            # PYTHONPATH: run this repository's pymoses (the run's cwd is its run
            # directory, so the venv's editable install would win otherwise)
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                **({"PYTHONPATH": str(REPO)} if is_pymoses else {}),
            },
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
        try:
            on_val = sr_eval.evaluate(prog, train, val)
            on_test = sr_eval.evaluate(prog, train, test)
        except Exception as exc:  # never let one unscorable candidate kill the pool
            scored.append(
                {
                    "program": prog,
                    "train_nmse": math.inf,
                    "val_nmse": math.inf,
                    "test_nmse": math.inf,
                    "eval_error": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
            continue
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
    seeding = _seed_summary(seed_rec, run_dir, best.get("program"), train, val, test)
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
        "n_candidates_unscorable": sum("eval_error" in c for c in scored),
        "train_nmse": best.get("train_nmse"),
        "val_nmse": best.get("val_nmse"),
        "test_nmse": best.get("test_nmse"),
        "final_program": best.get("program"),
        "first_train_nmse": first.get("train_nmse"),
        "first_test_nmse": first.get("test_nmse"),
        **seeding,
    }
    if seed_rec is not None:
        summary["moses_cpu_s"] = summary["cpu_s"]
        summary["cpu_s"] += seed_rec["gpgomea"]["cpu_s"]
        if summary["n_evals"] is not None:
            summary["n_evals_total"] = summary["n_evals"] + seed_rec["evaluations_used"]
        # re-seeding: its GP-GOMEA evaluations are already in MOSES's n_evals (the hook
        # adds them to stats.n_evals), and its CPU time in cpu_s (a grandchild process)
        rs_path = run_dir / "reseed.json"
        if rs_path.exists():
            rs = json.loads(rs_path.read_text())
            summary.update(
                n_reseeds=rs["n_reseeds"],
                reseed_evals=rs["reseed_evals"],
                reseed_cpu_s=rs["reseed_cpu_s"],
            )
            if summary["n_evals"] is not None:
                summary["moses_own_evals"] = summary["n_evals"] - rs["reseed_evals"]
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def _is_pymoses(binary: str) -> bool:
    return Path(binary).parent.name == "bin" and Path(binary).name == "moses"


def _apply_seed(job: dict, argv: list[str], run_dir: Path) -> tuple[dict, list[str]]:
    """GP-GOMEA seeds on `train_scaled.csv`, paid out of the arm's evaluation budget."""
    import gpgomea_seed as gs

    spec, b = job["seed_spec"], job["budget"]
    if "evals" not in b:
        raise ValueError("seeding needs an evaluation budget ({'evals': N})")
    opt_in = [x for x in ("reserve_empty_every", "reseed_after") if spec.get(x)]
    if opt_in and not _is_pymoses(job["binary"]):
        raise ValueError(f"seed block keys {opt_in} need pyMoses, not {job['binary']}")
    total = int(b["evals"])
    b_seed = int(spec["budget"]) if "budget" in spec else int(spec["share"] * total)
    rec = gs.make_seeds(
        "regression",
        spec["binary"] if os.path.isabs(spec["binary"]) else str(REPO / spec["binary"]),
        b_seed,
        int(spec.get("height", 3)),
        int(job["seed"]),
        int(spec.get("k", 5)),
        run_dir / "gpgomea",
        train_csv=run_dir / "train_scaled.csv",
        popsize=spec.get("popsize"),
        functions=spec.get("functions"),
        extra=spec.get("gpgomea_argv"),
        tol=float(spec.get("tol", 1e-5)),
    )
    with_empty = bool(spec.get("with_empty", True))
    moses_m = gs.shared_budget(total, rec["evaluations_used"])
    argv = gs.replace_max_evals(argv, moses_m)
    argv += gs.exemplar_argv("regression", [x["combo"] for x in rec["seeds"]], with_empty)
    if int(spec.get("reserve_empty_every", 0)) > 0:
        argv += ["--reserve-empty-every", str(int(spec["reserve_empty_every"]))]
    rec.update(total_budget=total, seed_budget=b_seed, moses_budget=moses_m,
               with_empty=with_empty)
    (run_dir / "seeds.json").write_text(json.dumps(rec, indent=2))
    return rec, argv


def _seed_summary(rec, run_dir: Path, final_program, train, val, test) -> dict:
    """Seeding fields of a seeded run's summary (empty for an unseeded one)."""
    if rec is None:
        return {}
    out: dict = {
        "seed_evals": rec["evaluations_used"],
        "seed_budget": rec["seed_budget"],
        "moses_budget": rec["moses_budget"],
        "gpgomea_cpu_s": rec["gpgomea"]["cpu_s"],
        "n_seeds": len(rec["seeds"]),
        "seed_n_score_matches": rec.get("n_score_matches"),
        "seed_n_rejected": rec.get("n_rejected"),
    }
    if rec["seeds"]:
        top = rec["seeds"][0]["combo"]
        try:
            on_val = sr_eval.evaluate(top, train, val)
            on_test = sr_eval.evaluate(top, train, test)
            out.update(
                seed_train_nmse=on_val["train"]["nmse_scaled"],
                seed_val_nmse=on_val["test"]["nmse_scaled"],
                seed_test_nmse=on_test["test"]["nmse_scaled"],
            )
        except Exception as exc:  # report, never fail the run
            out["seed_eval_error"] = f"{type(exc).__name__}: {exc}"[:300]
    lin_path = run_dir / "lineage.json"
    if lin_path.exists():
        import seed_lineage

        lin = json.loads(lin_path.read_text())
        out["expansions_by_lineage"] = lin.get("expansions_by_lineage")
        if "reserve_empty" in lin:
            out["reserve_empty"] = lin["reserve_empty"]
        out["best_train_lineages"] = lin.get("best_lineages")
        if final_program is not None:
            key = seed_lineage.key_of(final_program)
            out["final_lineage"] = lin.get("pool", {}).get(key, "not in final metapopulation")
    return out


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
                        "seed_spec": a.get("seed"),
                        "split_dir": str(ddir / ds / "splits" / f"s{seed}"),
                        "wall_cap_s": manifest["wall_cap_s"],
                        "mem_limit_mb": manifest.get("mem_limit_mb"),
                        "run_dir": str(out_root / manifest["name"] / "runs" / run_id),
                        **({"events": a["events"]} if "events" in a else {}),
                    }
                )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("output_root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="run only run ids matching this regex")
    ap.add_argument(
        "--mem-limit-mb", type=int, default=None, help="override the manifest's per-run limit"
    )
    ap.add_argument(
        "--split-major",
        action="store_true",
        help="schedule split by split (mixing datasets and arms in each batch) instead of "
        "dataset by dataset, so memory-heavy runs of one dataset do not all overlap",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.output_root).resolve()
    jobs = expand(manifest, out_root, repo)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["run_id"])]
    if args.mem_limit_mb is not None:
        for j in jobs:
            j["mem_limit_mb"] = args.mem_limit_mb
    if args.split_major:
        jobs.sort(key=lambda j: (j["seed"], j["task"], j["arm"]))
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
