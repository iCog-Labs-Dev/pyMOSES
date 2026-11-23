"""Run MOSES (classic C++ or pyMoses) on the Phase 1f regression datasets.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/moses_sr_runner.py MANIFEST OUTPUT_ROOT [--workers N]
        [--only REGEX]

A manifest (JSON) has these keys:

* `name`, `datasets_dir` (built by `sr_datasets.py`), `datasets`, `splits`,
  `wall_cap_s`, and optionally `mem_limit_mb` (RLIMIT_AS per run);
* `arms`: `{arm: {"binary": path, "argv": [extra MOSES flags], "budget": {"evals": N} |
  {"time_s": T}}}`. An argv element may hold the placeholder `{K*n_train}` (K a positive
  integer; server round 12): the runner puts the integer K times the number of training rows
  of the run's split in its place (`resolve_argv`), so one arm can pass a value that scales
  with the dataset (BASE-DIV: `--diversity-pressure {6*n_train}`). An element without the
  placeholder goes to MOSES unchanged. An arm may also carry `"terms_guard": "val"` (server
  round 13, TLS-G; see "TLS-G" below). `expand` refuses it on an arm without
  `--linear-scaling terms` in its argv, and refuses any other value;
* optionally `"clip_test_nmse": true` (server round 13): every run of the manifest gets
  `test_nmse_clipped` in its summary (see below). Without the key (or with false) the
  summary does not change. `expand` refuses a value that is not true or false;
* optionally `"plain_nmse": true` (backlog 12.2, after server round 13): every
  `--linear-scaling terms` run of the manifest gets the plain-scaling fields (see below).
  Without the key (or with false) no run gets them. `expand` refuses a value that is not true
  or false;
* an arm `{"linear": "ols"}` or `{"linear": "ridge"}` (server round 14, LIN-OLS and LIN-RIDGE;
  see "Linear baselines" below) has no MOSES run. `expand` refuses another value, and refuses a
  linear arm with any other key (`argv`, `binary`, `budget`, ...).
* an arm `{"standalone": {...}}` (the IT linkage study, BACKLOG 13.9; see "Stand-alone GOMEA"
  below) has no MOSES run either. `expand` refuses a wrong block, and a stand-alone arm with any
  other key.

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
   first, best-training-score candidate (`first_*`). An arm with `--linear-scaling terms`
   (server round 12) is scored with the per-term weights refitted on train
   (`sr_eval.evaluate(..., terms=True)`); each candidate and the summary then get a
   `terms` record (`n_terms`, `fallback`, `reason`, `weights`) of that refit.

**Server round 13 (new fields only; the other values do not change).**

* With the manifest key `"clip_test_nmse": true`, every arm: the summary has
  `test_nmse_clipped`, the test NMSE of the reported candidate with its scaled predictions
  clipped to [min - 0.1 R, max + 0.1 R] of the training target (R = max - min;
  `sr_eval.evaluate(..., clip=True)`), in the target space of `test_nmse`. candidates.json
  does not change.
* With the manifest key `"plain_nmse": true`, an arm with `--linear-scaling terms`: each
  candidate and the summary also have `plain_val_nmse` and `plain_test_nmse`, the NMSE of the
  plain a + b*f fitted on train (descriptive). The round 13 manifests have the key. Without
  it, the output of such an arm is that of round 12.
* **TLS-G** (`"terms_guard": "val"` on an arm with `--linear-scaling terms`; the argv and so
  the search are those of BASE-TLS): for each candidate the runner compares the validation
  NMSE of the per-term model with that of the plain a + b*f, both fitted on train. It keeps
  the lower one; a tie keeps plain (the smaller model). The candidate's `train_nmse`,
  `val_nmse` and `test_nmse` (and `test_nmse_clipped`) are then those of the kept scaling, and
  the runner selects the candidate with the lowest `val_nmse`, as for every arm. Each
  candidate's `terms` record gets `guard` (`"terms"` or `"plain"`), so the summary's `terms`
  record has the guard of the reported candidate; each candidate also gets `terms_val_nmse`
  and `terms_test_nmse`, the values of the per-term model. A one-term program or a fit that
  falls back has the same value both ways, so its guard is `"plain"`. The guard compares the
  plain values also without `"plain_nmse": true`; only the fields need the key.

**Linear baselines (server round 14; `sr_linear.py`).** For a job with `"linear"` the runner
starts no process. It fits y = w_0 + sum_i w_i x_i over all input columns of the split (a boolean
input is 0 or 1) to the training target scaled as for the MOSES arms (step 1 above; it also
writes `train_scaled.csv`), by ordinary least squares (`"ols"`) or by ridge regression with the
lambda of the grid 10^-6 .. 10^3 that gives the lowest validation NMSE (`"ridge"`; a tie keeps the
larger lambda). It then scores the model's own predictions (no extra a + b*f fit) with the NMSE
functions of `sr_eval.py` on the original train, val and test splits, so `train_nmse`,
`val_nmse`, `test_nmse` and `test_nmse_clipped` (with `"clip_test_nmse": true`) mean what they
mean for the MOSES arms. The summary has `status`, the NMSE fields, `cpu_s` (the CPU time of the
fit, in the worker process), `wall_s`, `ru_maxrss_kb` (the worker process), `n_evals` null, and
`linear` (the method, the inputs, the weights in the units of the original target, and for OLS
the fallback, for ridge the chosen `lambda` and the validation NMSE of each lambda). For the size
rule of the report (`phase_report.py`, `"size_terms"`) the summary has `final_program`
`+($x1 ... $xk)` over the k inputs and a `terms` record with `n_terms` k and `fallback` false.
`candidates.json` holds the one candidate. A fit that fails gives status "error". The
manifest's `mem_limit_mb` does not apply (there is no child process; a fit uses a few MB).

**Stand-alone GOMEA (NOT A PORT: the IT linkage study, BACKLOG 13.9;
`docs/gomea/IT-LINKAGE-STUDY-PLAN.md`, section 6.4).** An arm `{"standalone": {"genome":
"template" | "it", "fos": "uni" | "rt" | "lt-mi" | "lt-b" | "np", "pop": N, "max_evals": E}}`
runs `python -m pymoses.gomea.standalone` (the runner's own Python, `PYTHONPATH` this
repository, the run directory as its working directory, the manifest's `wall_cap_s` and
`mem_limit_mb`) on the original train, val and test splits, with the split number as its seed.
Optional block keys: `template_height`, `it_terms`, `it_factors`, `it_powers` (a list of
integers), `column_cache` (true or false), `it_cache_max_columns`, `it_cache_max_pairs`, and (server
round 15) `max_generations` (`--max-generations`: end reason "max_generations") and
`stall_generations` (`--stall-generations`: end reason "stalled" after K generations in a row
with no new evaluation), both positive integers. The
runner writes `train_scaled.csv` as for MOSES (the module scales the training target with the
same expressions) and `config.json`, and keeps the module's `summary.json`, `candidates.json`
and `events.jsonl`. A run that ends without a summary (a timeout, a crash) gets a summary with
status "timeout" or "error".

**Telemetry.** A pyMoses GOMEA arm (`-a gomea` or `-a hc-gomea` in its argv) also gets
`--gomea-log RUN_DIR/events.jsonl`, as the Boolean runner (`benchmarks.gomea.manifest`)
does, so each run writes the per-deme "invocation_start", "generation" and "end" events
(`events_argv`). It is on by default; `"events": false` on an arm turns it off. The log is a
side file: MOSES's output is the same with or without it. A pyMoses `-a hc` arm with
`--const-opt` (server round 10: LM for hill climbing) gets the same log, so its "end" events
give `const_opt` (LM calls, guard rejections) and the deme columns of `phase_report.py`;
`-a hc` without `--const-opt` gets no log, as before. With `--exemplar-selection lexicase`
(server round 13) the log ends with one "lexicase" event, which the summary copies as
`lexicase` (`lexicase_summary`).

**Server round 16 (the budget round; `SERVER-ROUND16-PREREGISTRATION.md`; new fields only, the
other values do not change).**

* `rv` (`rv_summary`): when the GOMEA log has "end" events with an "rv" record (the RV option
  `--gomea-rv-step`), the RV totals of the run: `rv_evals` and `rv_share` (of `n_evals`), the
  counters of `RV_COUNTERS` summed over the demes (`generations` is the number of RV
  generations, `kept_p` the samples kept by p_accept, `fi_*` the forced improvements, `skip_*`
  and `skipped_*` the skips), `ams1_rounds` and `ams2_rounds` (RV generations with AMS
  evaluations), `converged_by` (demes by the RV convergence rule), `nis_max_seen`, the times of
  `RV_TIMES` and `demes` (the demes with an "rv" record). Without an "rv" record (the option
  is off) the summary has no `rv` key, as for `lexicase`.
* With the manifest key `"anytime": true`, a pyMoses GOMEA arm (`-a gomea` or `-a hc-gomea`)
  with an evaluation budget and no `seed` block also gets `demes`, `anytime` and
  `anytime_check`. Without the key (or with false) the summary does not change. `expand`
  refuses a value that is not true or false.

  * The run evaluation count of a log record is the sum of the final `ledger.requests` of the
    earlier demes plus the `ledger.requests` of the record's deme (cumulative in the deme).
    The sum over the demes is `n_evals` (checked on the 60 runs of the T3.2 spike 2).
  * `demes` (`deme_summary`): `count` (the demes with evaluations, as `phase_report.py`
    counts them), `start_evals` (the run evaluation at which each of these demes starts),
    `requests` (the evaluations of each), `second_start_evals` (null for one deme) and
    `evals_total`.
  * `anytime` (`anytime_records`): one record for each checkpoint 1, 2, 5, 10, 20, 50 and
    100 % of the budget E (`max(1, round(frac * E))` evaluations), in the format of the
    stand-alone harness (`pymoses/gomea/standalone/run.py`, read by
    `linkage_report.anytime_area`): `evals`, `frac`, `best_train_nmse`, `reached`, and
    `log_evals`. The value is the best raw score of the log records at or before the
    checkpoint: the `best_raw` of each "generation" event and the `best.raw` of each "end"
    event. `log_evals` is the run evaluation count of the last such record. A checkpoint
    before the first record has the value null (the log has the resolution of a generation).
    A checkpoint after the last evaluation of the run has `reached` false and the final best
    value, as in the harness.
  * The conversion (`raw_to_train_nmse`). With `--linear-scaling terms` (and 1), the scorer
    (`pymoses/moses/scoring/bscores.py`, `contin_bscore._terms_bscore`) gives one
    -(fitted - y')^2 for each training row, on the scaled target y' of `train_scaled.csv`; the
    raw score is their sum (`bscore_base.sum_bscore`), without the complexity penalty. So
    the training NMSE of `sr_eval.py` (100 * MSE / var, of the same least-squares fit, which
    an affine map of the target does not change) is 100 * (-raw / n_train) / var(y'), with
    var(y') the population variance of the scaled target (100 up to rounding).
  * `anytime_check`: the conversion of the score of the first printed candidate against
    `first_train_nmse` (the `sr_eval.py` value of the same program): `search_rel_diff`, their
    relative difference (float32 scores: at most 5.2e-8 on the spike runs). Also the best raw
    score of the log, its conversion, whether it is the score of the first printed candidate
    (`log_best_is_first`; MOSES prints by the penalized score, so the log can have a better
    raw score) and its relative difference to `first_train_nmse` (`log_best_rel_diff`), and
    whether the evaluations of the log (`log_evals_total`) are `n_evals`
    (`log_evals_match_n_evals`). No check fails on these values.

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
import sr_linear  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _scale_train(src: Path, dst: Path) -> tuple[int, float, float]:
    with src.open() as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    k = header.index("y")
    ys = [float(r[k]) for r in body]
    mean = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    _write_scaled(header, body, dst, mean, sd)
    return len(body), mean, sd


def _scale_like_train(src: Path, dst: Path, mean: float, sd: float) -> None:
    """`src` (e.g. val.csv) with its target mapped as the training target was,
    y' = 10 (y - mean) / sd with the TRAINING mean and sd."""
    with src.open() as fh:
        rows = list(csv.reader(fh))
    _write_scaled(rows[0], rows[1:], dst, mean, sd)


def _write_scaled(header, body, dst: Path, mean: float, sd: float) -> None:
    k = header.index("y")
    with dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in body:
            r = list(r)
            r[k] = repr(10.0 * (float(r[k]) - mean) / sd)
            w.writerow(r)


def _limit(mem_mb):
    def fn():
        if mem_mb:
            lim = int(mem_mb) * 2**20
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    return fn


# Server round 12: the placeholder `{K*n_train}` in an arm's argv (K a positive integer).
_N_TRAIN_PLACEHOLDER = re.compile(r"\{([1-9][0-9]*)\*n_train\}")


def resolve_argv(argv: list, n_train: int) -> list:
    """`argv` with every `{K*n_train}` replaced by the integer K * `n_train` (server round 12,
    opt-in). An element without the placeholder (or not a string) is returned as it is."""
    return [
        _N_TRAIN_PLACEHOLDER.sub(lambda m: str(int(m.group(1)) * n_train), x)
        if isinstance(x, str) and _N_TRAIN_PLACEHOLDER.search(x) else x
        for x in argv
    ]


def moses_argv(job: dict, run_dir: Path, n_train: int) -> list[str]:
    """The MOSES command line of one run (before seeding rewrites `-m`): the fixed
    regression setup, the budget, the arm's argv (with `{K*n_train}` replaced,
    `resolve_argv`), then `--gomea-log RUN_DIR/events.jsonl` when `events_argv` asks for
    it."""
    argv = [job["binary"], "-i", str(run_dir / "train_scaled.csv"), "-u", "y"]
    argv += ["-r", str(job["seed"]), "-n", "sin", "-n", "exp", "-n", "log"]
    argv += ["-v", str(60 * n_train), "-c", "10", "-V1", "-l", "WARN"]
    argv += ["-f", str(run_dir / "moses.log")]
    b = job["budget"]
    argv += ["-m", str(b["evals"])] if "evals" in b else ["--max-time", str(b["time_s"])]
    if "evals" not in b:
        argv += ["-m", "1000000000"]
    argv += resolve_argv(job["argv"], n_train)
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


def _selects_hc(argv: list[str]) -> bool:
    """Whether `argv` selects `-a hc` (server round 10)."""
    for i, tok in enumerate(argv):
        tok = str(tok)
        if tok in ("-a", "--opt-algo") and i + 1 < len(argv) and str(argv[i + 1]) == "hc":
            return True
        if tok in ("--opt-algo=hc", "-ahc"):
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
    hc_lm = _selects_hc(argv) and _const_opt(argv) != "none"  # server round 10
    if not (_selects_gomea(argv) or hc_lm) or any(
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
    if job.get("linear"):
        return _run_linear(job)  # server round 14: a linear baseline, no MOSES process
    if job.get("standalone"):
        return _run_standalone(job)  # the IT linkage study: stand-alone GOMEA, no MOSES
    sdir = Path(job["split_dir"])
    n_train, mean, sd = _scale_train(sdir / "train.csv", run_dir / "train_scaled.csv")
    argv = moses_argv(job, run_dir, n_train)
    if _const_opt(job["argv"]) == "polish" or _const_opt_guard(job["argv"]) == "val":
        # server rounds 3 and 5: polish and the LM guard keep a tuned program only
        # if the validation NMSE is not worse; they read the validation split,
        # scaled like train
        _scale_like_train(sdir / "val.csv", run_dir / "val_scaled.csv", mean, sd)
        argv += ["--const-opt-val", str(run_dir / "val_scaled.csv")]
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
    terms = _linear_scaling(job["argv"]) == "terms"  # server round 12
    guard = job.get("terms_guard")  # server round 13 (TLS-G): "val" or None
    clip = bool(job.get("clip_test_nmse"))  # server round 13: opt-in (manifest key)
    plain = terms and bool(job.get("plain_nmse"))  # backlog 12.2: opt-in (manifest key)
    want_plain = plain or bool(terms and guard)  # TLS-G compares the plain values
    scored, clipped = [], []  # clipped: test_nmse_clipped of each candidate (round 13)
    for _score, prog in cands:
        try:
            on_val = sr_eval.evaluate(prog, train, val, terms=terms, plain=want_plain)
            on_test = sr_eval.evaluate(prog, train, test, terms=terms, plain=want_plain,
                                       clip=clip)
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
            if plain:
                scored[-1].update(plain_val_nmse=math.inf, plain_test_nmse=math.inf)
            clipped.append(math.inf)
            continue
        scored.append(
            {
                "program": prog,
                "train_nmse": on_val["train"]["nmse_scaled"],
                "val_nmse": on_val["test"]["nmse_scaled"],
                "test_nmse": on_test["test"]["nmse_scaled"],
            }
        )
        clipped.append(on_test["test"].get("nmse_clipped"))  # None without `clip`
        if terms:
            c = scored[-1]
            c["terms"] = on_val["terms"]
            if plain:
                # server round 13: the plain a + b*f values (descriptive; opt-in, backlog 12.2)
                c["plain_val_nmse"] = on_val["plain"]["test"]
                c["plain_test_nmse"] = on_test["plain"]["test"]
            if guard:
                clipped[-1] = _terms_guard(c, on_val, on_test, clipped[-1])
    (run_dir / "candidates.json").write_text(json.dumps(scored, indent=2))
    if status == "ok" and not scored:
        status, error = "error", "no candidate parsed"
    # the first candidate with the lowest validation NMSE, as min() over `scored` gives it
    i_best = min(range(len(scored)), key=lambda i: scored[i]["val_nmse"]) if scored else None
    best = scored[i_best] if scored else {}
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
    if clip:
        # server round 13: the reported candidate's test NMSE with clipped predictions
        summary["test_nmse_clipped"] = clipped[i_best] if scored else None
    if terms:
        # server round 12: the per-term refit of the reported program (round 13 TLS-G: with
        # the guard's choice `guard`)
        summary["terms"] = best.get("terms")
    if plain:
        # server round 13: the plain a + b*f values of the reported program (descriptive;
        # opt-in, backlog 12.2)
        summary["plain_val_nmse"] = best.get("plain_val_nmse")
        summary["plain_test_nmse"] = best.get("plain_test_nmse")
    co = const_opt_summary(run_dir / "events.jsonl", summary["n_evals"])
    if co is not None:
        summary["const_opt"] = co
    lex = lexicase_summary(run_dir / "events.jsonl")
    if lex is not None:
        summary["lexicase"] = lex  # server round 13 (absent without the option)
    rv = rv_summary(run_dir / "events.jsonl", summary["n_evals"])
    if rv is not None:
        summary["rv"] = rv  # server round 16 (absent without the RV option)
    if _wants_anytime(job):
        # server round 16 (opt-in, the manifest key "anytime"): the deme facts and the
        # best-so-far training NMSE at the checkpoints, from the GOMEA log
        events = run_dir / "events.jsonl"
        summary["demes"] = deme_summary(events)
        summary["anytime"], summary["anytime_check"] = anytime_records(
            events, int(job["budget"]["evals"]), n_train,
            _scaled_target_var(run_dir / "train_scaled.csv"),
            first_score=cands[0][0] if cands else None,
            first_train_nmse=first.get("train_nmse"),
            n_evals=summary["n_evals"],
        ) if events.exists() else (None, None)
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


def _fit_linear(method: str, train, val, mean: float, sd: float) -> tuple[list[float], dict]:
    """Server round 14: the weights (w_0, w_1, .., w_k) of the linear baseline `method` in the
    units of the original target, fitted to the training target scaled as for the MOSES arms
    (`mean`, `sd`: the training mean and sd), and the method's part of the `linear` record."""
    y_scaled = [10.0 * (t - mean) / sd for t in train.y]
    cols = sr_linear.columns(train.X)
    if method == "ols":
        w, rec = sr_linear.fit_ols(cols, y_scaled)
        return sr_linear.to_target_units(w, mean, sd), {
            "fallback": rec["fallback"], "fit_reason": rec["fit_reason"],
            "dropped_inputs": [train.labels[j] for j in rec["dropped"]]}
    fits, constant = sr_linear.fit_ridge(cols, y_scaled)
    val_cols = sr_linear.columns(val.X)
    best, by_lambda = None, []
    for lam, w in zip(sr_linear.RIDGE_LAMBDAS, fits, strict=True):
        if w is None:
            by_lambda.append(None)
            continue
        w = sr_linear.to_target_units(w, mean, sd)
        v = sr_eval.nmse(val.y, sr_linear.predict(w, val_cols))
        by_lambda.append(v)
        if best is None or v <= best[0]:  # a tie keeps the larger lambda (the grid ascends)
            best = (v, lam, w)
    if best is None:
        raise ValueError("ridge: no lambda of the grid gives a solution")
    return best[2], {"lambda": best[1], "lambdas": list(sr_linear.RIDGE_LAMBDAS),
                     "val_nmse_by_lambda": by_lambda,
                     "constant_inputs": [train.labels[j] for j in constant]}


def _run_linear(job: dict) -> dict:
    """Server round 14 (LIN-OLS, LIN-RIDGE): fit and score the linear baseline of one split in
    this process, with no MOSES process; write candidates.json and summary.json (see "Linear
    baselines" in the module docstring)."""
    run_dir = Path(job["run_dir"])
    summary_path = run_dir / "summary.json"
    t0, c0 = time.monotonic(), time.process_time()
    sdir = Path(job["split_dir"])
    (run_dir / "config.json").write_text(json.dumps(job, indent=2))
    summary: dict = {
        "run_id": job["run_id"], "task": job["task"], "arm": job["arm"], "seed": job["seed"],
        "status": "ok", "error": None, "exit_code": None, "n_evals": None,
    }
    try:
        n_train, mean, sd = _scale_train(sdir / "train.csv", run_dir / "train_scaled.csv")
        tables = {p: sr_eval.read_table(str(sdir / f"{p}.csv")) for p in ("train", "val", "test")}
        train = tables["train"]
        if not train.labels == tables["val"].labels == tables["test"].labels:
            raise ValueError("train, val and test feature columns differ")
        w, rec = _fit_linear(job["linear"], train, tables["val"], mean, sd)
        q = {p: sr_linear.predict(w, sr_linear.columns(t.X)) for p, t in tables.items()}
        nm = {p: sr_eval.nmse(t.y, q[p]) for p, t in tables.items()}
        program = "+(" + " ".join(f"${lab}" for lab in train.labels) + ")"
        k = len(train.labels)
        summary.update(
            n_train=n_train, target_mean=mean, target_sd=sd, n_candidates=1,
            n_candidates_unscorable=0, train_nmse=nm["train"], val_nmse=nm["val"],
            test_nmse=nm["test"], final_program=program, first_train_nmse=nm["train"],
            first_test_nmse=nm["test"],
            terms={"n_terms": k, "fallback": False, "reason": None, "weights": w},
            linear={"method": job["linear"], "inputs": list(train.labels),
                    "bool_inputs": [train.labels[j] for j in sorted(train.bool_cols)],
                    "target_scaling": "y' = 10 (y - mean) / sd, training mean and sd",
                    "weights": w, **rec},
        )
        if job.get("clip_test_nmse"):
            lo, hi = sr_eval.clip_range(train.y)
            summary["test_nmse_clipped"] = sr_eval.nmse_clipped(tables["test"].y, q["test"], lo, hi)
        cands = [{"program": program, "train_nmse": nm["train"], "val_nmse": nm["val"],
                  "test_nmse": nm["test"]}]
    except Exception as exc:  # never let one failed fit kill the pool
        summary.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300],
                       train_nmse=None, val_nmse=None, test_nmse=None, final_program=None)
        if job.get("clip_test_nmse"):
            summary["test_nmse_clipped"] = None
        cands = []
    (run_dir / "candidates.json").write_text(json.dumps(cands, indent=2))
    summary.update(wall_s=time.monotonic() - t0, cpu_s=time.process_time() - c0,
                   ru_maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


# NOT A PORT: the IT linkage study (BACKLOG 13.9), the arm key "standalone".
STANDALONE_GENOMES = ("template", "it")
STANDALONE_FOS = ("uni", "rt", "lt-mi", "lt-b", "np")  # pymoses.gomea.fos.LinkageModel.MODES
STANDALONE_REQUIRED = ("genome", "fos", "pop", "max_evals")
# optional block key -> the module's option
STANDALONE_OPTIONAL = {
    "template_height": "--template-height",
    "it_terms": "--it-terms",
    "it_factors": "--it-factors",
    "it_powers": "--it-powers",
    "column_cache": "--no-column-cache",
    "it_cache_max_columns": "--it-cache-max-columns",
    "it_cache_max_pairs": "--it-cache-max-pairs",
    "max_generations": "--max-generations",  # server round 15
    "stall_generations": "--stall-generations",  # server round 15
}


def standalone_argv(job: dict, run_dir: Path) -> list[str]:
    """The command line of a stand-alone job (`python -m pymoses.gomea.standalone ...`)."""
    spec, sdir = job["standalone"], Path(job["split_dir"])
    argv = [sys.executable, "-m", "pymoses.gomea.standalone",
            "--genome", spec["genome"], "--fos", spec["fos"],
            "--train", str(sdir / "train.csv"), "--val", str(sdir / "val.csv"),
            "--test", str(sdir / "test.csv"), "--target", "y",
            "--pop", str(spec["pop"]), "--max-evals", str(spec["max_evals"]),
            "--seed", str(job["seed"]), "--out", str(run_dir),
            "--arm", job["arm"], "--task", job["task"], "--run-id", job["run_id"]]
    for key, opt in STANDALONE_OPTIONAL.items():
        if key not in spec:
            continue
        if key == "column_cache":
            argv += [] if spec[key] else [opt]
        elif key == "it_powers":
            argv.append(f"{opt}=" + ",".join(str(int(p)) for p in spec[key]))
        else:
            argv += [opt, str(int(spec[key]))]
    return argv


def _run_standalone(job: dict) -> dict:
    """The IT linkage study: run one stand-alone GOMEA job (see "Stand-alone GOMEA" in the
    module docstring) and return its summary."""
    run_dir = Path(job["run_dir"])
    summary_path = run_dir / "summary.json"
    sdir = Path(job["split_dir"])
    try:
        _scale_train(sdir / "train.csv", run_dir / "train_scaled.csv")
    except ZeroDivisionError:
        pass  # a constant target: the module refuses it and writes the error summary
    argv = standalone_argv(job, run_dir)
    (run_dir / "config.json").write_text(json.dumps({**job, "resolved_argv": argv}, indent=2))
    t0 = time.monotonic()
    status, exit_code, error = "error", None, None
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=job["wall_cap_s"],
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                 "PYTHONPATH": str(REPO)},
            cwd=run_dir, preexec_fn=_limit(job.get("mem_limit_mb")),
        )
        (run_dir / "stdout.txt").write_text(proc.stdout)
        (run_dir / "stderr.txt").write_text(proc.stderr)
        exit_code = proc.returncode
        error = (proc.stderr.strip().splitlines() or ["?"])[-1][:300] if exit_code else None
    except subprocess.TimeoutExpired:
        status, error = "timeout", f"wall cap {job['wall_cap_s']} s reached"
    if status != "timeout" and summary_path.exists():
        return json.loads(summary_path.read_text())
    summary = {
        "run_id": job["run_id"], "task": job["task"], "arm": job["arm"], "seed": job["seed"],
        "status": status, "error": error or "no summary written", "exit_code": exit_code,
        "wall_s": time.monotonic() - t0, "train_nmse": None, "val_nmse": None,
        "test_nmse": None, "test_nmse_clipped": None, "final_program": None, "n_evals": None,
    }
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True))
    os.replace(tmp, summary_path)
    return summary


def _check_standalone(arm: str, a: dict) -> None:
    """The IT linkage study: `"standalone"` must be a block with the required keys and valid
    values, on an arm with no other key."""
    if "standalone" not in a:
        return
    spec = a["standalone"]
    extra = sorted(set(a) - {"standalone"})
    if extra:
        raise ValueError(f"arm {arm}: a stand-alone arm has only the key \"standalone\", "
                         f"got {extra}")
    if not isinstance(spec, dict):
        raise ValueError(f"arm {arm}: standalone must be a block (a JSON object)")
    missing = [k for k in STANDALONE_REQUIRED if k not in spec]
    unknown = sorted(set(spec) - set(STANDALONE_REQUIRED) - set(STANDALONE_OPTIONAL))
    if missing or unknown:
        raise ValueError(f"arm {arm}: standalone block: missing {missing}, unknown {unknown}")
    if spec["genome"] not in STANDALONE_GENOMES:
        raise ValueError(f"arm {arm}: standalone genome must be one of {STANDALONE_GENOMES}")
    if spec["fos"] not in STANDALONE_FOS:
        raise ValueError(f"arm {arm}: standalone fos must be one of {STANDALONE_FOS}")
    ints = [k for k in ("pop", "max_evals", "template_height", "it_terms", "it_factors",
                        "it_cache_max_columns", "it_cache_max_pairs", "max_generations",
                        "stall_generations") if k in spec]
    for k in ints:
        if not isinstance(spec[k], int) or isinstance(spec[k], bool) or spec[k] < 1:
            raise ValueError(f"arm {arm}: standalone {k} must be a positive integer")
    if spec["pop"] < 2 or spec["max_evals"] < spec["pop"]:
        raise ValueError(f"arm {arm}: standalone needs pop >= 2 and max_evals >= pop")
    if "column_cache" in spec and not isinstance(spec["column_cache"], bool):
        raise ValueError(f"arm {arm}: standalone column_cache must be true or false")
    if "it_powers" in spec:
        p = spec["it_powers"]
        if (not isinstance(p, list) or not p
                or any(not isinstance(x, int) or isinstance(x, bool) or x == 0 for x in p)
                or len(set(p)) != len(p)):
            raise ValueError(f"arm {arm}: standalone it_powers must be distinct non-zero "
                             f"integers")


def _terms_guard(c: dict, on_val: dict, on_test: dict, clipped_terms: float) -> float:
    """Server round 13 (TLS-G): keep the per-term model or the plain a + b*f of candidate `c`
    (a scored record of a `--linear-scaling terms` arm), whichever has the lower validation
    NMSE; a tie keeps plain. Rewrites `c`'s `train_nmse`, `val_nmse` and `test_nmse` to the
    kept scaling, adds `terms_val_nmse`, `terms_test_nmse` and `c["terms"]["guard"]`, and
    returns the kept scaling's test NMSE with clipped predictions (`None` when `on_test` has
    no clipped value: the manifest has no `"clip_test_nmse": true`). The plain values come
    from `on_val["plain"]` and `on_test["plain"]` (backlog 12.2: `c` has `plain_val_nmse` and
    `plain_test_nmse` only with `"plain_nmse": true`)."""
    c["terms_val_nmse"], c["terms_test_nmse"] = c["val_nmse"], c["test_nmse"]
    c["terms"] = {**c["terms"], "guard": "terms"}
    if on_val["plain"]["test"] <= c["val_nmse"]:
        c["terms"]["guard"] = "plain"
        c["train_nmse"] = on_val["plain"]["train"]
        c["val_nmse"] = on_val["plain"]["test"]
        c["test_nmse"] = on_test["plain"]["test"]
        return on_test["plain"].get("test_clipped")
    return clipped_terms


def _const_opt(argv: list) -> str:
    """The `--const-opt` mode an arm's argv selects ("none" without it)."""
    argv = [str(x) for x in argv]
    for i, tok in enumerate(argv):
        if tok == "--const-opt" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--const-opt="):
            return tok.split("=", 1)[1]
    return "none"


def _linear_scaling(argv: list) -> str:
    """The `--linear-scaling` value an arm's argv selects ("0" without it; the last one
    wins, as in MOSES's own parser)."""
    argv = [str(x) for x in argv]
    value = "0"
    for i, tok in enumerate(argv):
        if tok == "--linear-scaling" and i + 1 < len(argv):
            value = argv[i + 1]
        elif tok.startswith("--linear-scaling="):
            value = tok.split("=", 1)[1]
    return value


def _const_opt_guard(argv: list) -> str:
    """The `--const-opt-guard` value an arm's argv selects ("none" without it)."""
    argv = [str(x) for x in argv]
    for i, tok in enumerate(argv):
        if tok == "--const-opt-guard" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--const-opt-guard="):
            return tok.split("=", 1)[1]
    return "none"


def const_opt_summary(events: Path, n_evals: int | None) -> dict | None:
    """Server round 3: the run's constant-tuning telemetry from the GOMEA log
    (the `const_opt` record of every deme's "end" event and the "polish"
    event), or `None` when the run did no tuning. `lm_improved_share` is the
    share of LM calls that lowered the training error; `tuning_share` the share
    of the run's evaluations that tuning used. With the guard (server round 5) it
    also has `lm_guard_checked` (LM results that the guard compared on the
    validation rows), `lm_guard_rejected` and their ratio `lm_guard_rejected_share`. With
    the pole check (server round 13, `--const-opt-pole-check box`) it also has
    `lm_pole_checked` (LM results that the check compared on the probe points),
    `lm_pole_rejected` and their ratio `lm_pole_rejected_share`."""
    if not events.exists():
        return None
    ends, polish = [], None
    for line in events.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "end" and ev.get("const_opt"):
            ends.append(ev["const_opt"])
        elif ev.get("type") == "polish":
            polish = ev
    if not ends and polish is None:
        return None
    keys = ("lm_calls", "lm_improved", "lm_not_improved", "lm_singular", "lm_failed",
            "lm_no_constants", "lm_budget", "mutate_attempts", "mutate_accepted",
            "tuning_evals")
    out: dict = {"mode": ends[0]["mode"] if ends else "polish"}
    out.update({k: sum(int(e.get(k, 0)) for e in ends) for k in keys})
    out["lm_improved_share"] = (out["lm_improved"] / out["lm_calls"]) if out["lm_calls"] else None
    out["mutate_accepted_share"] = (
        out["mutate_accepted"] / out["mutate_attempts"] if out["mutate_attempts"] else None
    )
    if polish is not None:
        out["polish"] = {k: v for k, v in polish.items() if k != "type"}
        out["polish_accepted"] = bool(polish.get("accepted"))
        out["tuning_evals"] += int(polish.get("evals", 0))
    out["tuning_share"] = out["tuning_evals"] / n_evals if n_evals else None
    elite = [e for e in ends if "lm_elite_calls" in e]
    if elite:
        # server round 12 (--const-opt-during elite): the LM calls during the search, which
        # lm_calls and tuning_evals above also count (absent without the option)
        for k in ("lm_elite_calls", "lm_elite_evals", "lm_elite_kept", "lm_elite_worse",
                  "lm_elite_guard_rejected", "lm_elite_not_knob"):
            out[k] = sum(int(e.get(k, 0)) for e in elite)
    guards = [e for e in ends if e.get("guard")]
    if guards:
        out["guard"] = guards[0]["guard"]
        out["lm_guard_checked"] = sum(int(e.get("lm_guard_checked", 0)) for e in guards)
        out["lm_guard_rejected"] = sum(int(e.get("lm_guard_rejected", 0)) for e in guards)
        out["lm_guard_rejected_share"] = (
            out["lm_guard_rejected"] / out["lm_guard_checked"] if out["lm_guard_checked"] else None
        )
    poles = [e for e in ends if "lm_pole_checked" in e]
    if poles:
        # server round 13 (--const-opt-pole-check box; absent without the option)
        out["lm_pole_checked"] = sum(int(e.get("lm_pole_checked", 0)) for e in poles)
        out["lm_pole_rejected"] = sum(int(e.get("lm_pole_rejected", 0)) for e in poles)
        out["lm_pole_rejected_share"] = (
            out["lm_pole_rejected"] / out["lm_pole_checked"] if out["lm_pole_checked"] else None
        )
    return out


def lexicase_summary(events: Path) -> dict | None:
    """Server round 13 (`--exemplar-selection lexicase`): the run's "lexicase" event of the
    GOMEA log (`n_cases`, `selections`, `cases_used_mean`, `cases_used_max`,
    `pool_size_mean`, `bscore_recomputes`, `bscore_recompute_cpu_s`; pyMoses writes it once,
    at the end of the run), or `None` when the log has none (the option is off)."""
    if not events.exists():
        return None
    out = None
    for line in events.read_text().splitlines():
        if '"lexicase"' not in line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "lexicase":
            out = {k: v for k, v in ev.items() if k != "type"}
    return out


# ---- server round 16: the RV totals, the deme facts and the best-so-far checkpoints

# The checkpoints of the best-so-far training NMSE, as pymoses/gomea/standalone/run.py has them.
ANYTIME_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
# The counters of the "rv" record of a deme's "end" event (the names of the T3.2 spike 2),
# summed over the demes. A missing counter counts 0.
RV_COUNTERS = (
    "generations", "improving", "kept_p", "kept_p_worse", "rejected", "invalid",
    "ams1_evals", "ams1_improving", "ams1_kept_p", "ams2_evals", "ams2_improving", "ams2_kept_p",
    "fi_triggers", "fi_evals", "fi_improved", "fi_elite_copies",
    "skip_struct", "skip_cache", "skip_dup", "skipped_p_rv", "skipped_gom_improved",
    "models", "sd_degenerate", "near_equal", "cache_marks", "absorbed_cache",
    "pm1_moves", "pm1_dcomplexity", "pm1_complexity_up",
)
RV_TIMES = ("eval_s", "step_s", "analyse_s")


def _read_events(events: Path) -> list[dict]:
    """The records of a GOMEA log in order; a line that is not a JSON object is left out (a
    run that stopped can leave a cut line). No file gives no records."""
    if not events.exists():
        return []
    out = []
    for line in events.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _count(v) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _real(v) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    return None


def rv_summary(events: Path, n_evals: int | None) -> dict | None:
    """Server round 16: the RV totals of a run from its GOMEA log (see the module docstring),
    or `None` when no "end" event has an "rv" record (the RV option is off)."""
    evs = _read_events(events)
    ends = [e["rv"] for e in evs if e.get("type") == "end" and isinstance(e.get("rv"), dict)]
    if not ends:
        return None
    gens = [e["rv"] for e in evs if e.get("type") == "generation"
            and isinstance(e.get("rv"), dict) and e["rv"].get("ran")]
    rv_evals = sum(_count(r.get("rv_evals", r.get("evals"))) for r in ends)
    out: dict = {"mode": ends[0].get("mode"), "budget": ends[0].get("budget"),
                 "demes": len(ends), "rv_evals": rv_evals,
                 "rv_share": rv_evals / n_evals if n_evals else None}
    out.update({k: sum(_count(r.get(k)) for r in ends) for k in RV_COUNTERS})
    out["ams1_rounds"] = sum(_count(r.get("ams1_evals")) > 0 for r in gens)
    out["ams2_rounds"] = sum(_count(r.get("ams2_evals")) > 0 for r in gens)
    conv: dict[str, int] = {}
    for r in ends:
        if r.get("converged_by") is not None:
            k = str(r["converged_by"])
            conv[k] = conv.get(k, 0) + 1
    out["converged_by"] = conv
    out["nis_max_seen"] = max(_count(r.get("nis_max_seen")) for r in ends)
    out.update({k: math.fsum(_real(r.get(k)) or 0.0 for r in ends) for k in RV_TIMES})
    return out


def _deme_walk(evs: list[dict]) -> list[dict]:
    """The demes of a GOMEA log in order. Each one has `start` (the run evaluations before
    it), `requests` (its last `ledger.requests`), `ended` (an "end" event closed it) and
    `points`, the (run evaluations, raw score) of its "generation" records (`best_raw`) and of
    its "end" record (`best.raw`)."""
    demes: list[dict] = []
    cur, offset = None, 0
    for e in evs:
        t = e.get("type")
        if t == "invocation_start" or (cur is None and t in ("generation", "end")):
            if cur is not None:  # a deme without an "end" event
                demes.append(cur)
                offset += cur["requests"]
            cur = {"start": offset, "requests": 0, "ended": False, "points": []}
        if t not in ("generation", "end"):
            continue
        req = _count((e.get("ledger") or {}).get("requests"))
        cur["requests"] = max(cur["requests"], req)
        raw = _real(e.get("best_raw") if t == "generation" else (e.get("best") or {}).get("raw"))
        if raw is not None:
            cur["points"].append((offset + req, raw))
        if t == "end":
            cur["ended"] = True
            demes.append(cur)
            offset += cur["requests"]
            cur = None
    if cur is not None:
        demes.append(cur)
    return demes


def deme_summary(events: Path) -> dict | None:
    """Server round 16: the deme facts of a run (see the module docstring), or `None` when the
    log has no deme with evaluations."""
    walk = _deme_walk(_read_events(events))
    demes = [d for d in walk if d["requests"] > 0]
    if not demes:
        return None
    return {
        "count": len(demes),
        "start_evals": [d["start"] for d in demes],
        "requests": [d["requests"] for d in demes],
        "second_start_evals": demes[1]["start"] if len(demes) > 1 else None,
        "evals_total": sum(d["requests"] for d in walk),
    }


def raw_to_train_nmse(raw: float | None, n_train: int, var_scaled: float | None) -> float | None:
    """Server round 16: the training NMSE (in %, as `sr_eval.py`) of a MOSES raw score with
    linear scaling: 100 * (-raw / n_train) / var(y'), y' the scaled training target (see the
    module docstring). `None` for a score that is not finite."""
    raw = _real(raw)
    if raw is None or not n_train or not var_scaled:
        return None
    return 100.0 * (-raw / n_train) / var_scaled


def _scaled_target_var(path: Path) -> float | None:
    """The population variance of the column `y` of `train_scaled.csv` (the target MOSES
    reads), or `None` when the file is missing."""
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.reader(fh))
    k = rows[0].index("y")
    ys = [float(r[k]) for r in rows[1:]]
    if not ys:
        return None
    mean = math.fsum(ys) / len(ys)
    return math.fsum((y - mean) ** 2 for y in ys) / len(ys)


def _rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return None
    return abs(a - b) / abs(b)


def anytime_records(events: Path, budget_evals: int, n_train: int, var_scaled: float | None,
                    first_score: float | None = None,
                    first_train_nmse: float | None = None,
                    n_evals: int | None = None) -> tuple[list[dict], dict]:
    """Server round 16: the `anytime` records and the `anytime_check` record of a run (see the
    module docstring). `budget_evals` is the run's evaluation budget E; `n_evals` the run's
    evaluation count of MOSES's output (null: not known)."""
    walk = _deme_walk(_read_events(events))
    points = sorted(p for d in walk for p in d["points"])
    total = sum(d["requests"] for d in walk)
    out = []
    for f in ANYTIME_FRACTIONS:
        c = max(1, round(f * budget_evals))
        reached = total >= c
        seen = [p for p in points if p[0] <= (c if reached else total)]
        best = max((raw for _, raw in seen), default=None)
        out.append({
            "evals": c, "frac": f,
            "best_train_nmse": raw_to_train_nmse(best, n_train, var_scaled),
            "reached": reached,
            "log_evals": seen[-1][0] if seen else None,
        })
    log_best = max((raw for _, raw in points), default=None)
    first = raw_to_train_nmse(first_score, n_train, var_scaled)
    log_nmse = raw_to_train_nmse(log_best, n_train, var_scaled)
    check = {
        "rule": "100 * (-raw / n_train) / var(y_scaled)",
        "n_train": n_train,
        "var_scaled_target": var_scaled,
        "first_score": _real(first_score),
        "first_score_train_nmse": first,
        "first_train_nmse": first_train_nmse,
        "search_rel_diff": _rel_diff(first, first_train_nmse),
        "log_best_raw": log_best,
        "log_best_train_nmse": log_nmse,
        "log_best_is_first": (log_best is not None and _real(first_score) is not None
                              and log_best == float(first_score)),
        "log_best_rel_diff": _rel_diff(log_nmse, first_train_nmse),
        "log_evals_total": total,
        "log_evals_match_n_evals": None if n_evals is None else total == n_evals,
    }
    return out, check


def _wants_anytime(job: dict) -> bool:
    """Server round 16: whether a job gets `demes`, `anytime` and `anytime_check` (the
    manifest key "anytime", a pyMoses GOMEA arm, an evaluation budget, no seed block)."""
    return (bool(job.get("anytime")) and _is_pymoses(job["binary"])
            and _selects_gomea([str(x) for x in job["argv"]])
            and "evals" in job["budget"] and not job.get("seed_spec"))


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


def _check_terms_guard(arm: str, a: dict) -> None:
    """Server round 13: `"terms_guard"` must be "val", on an arm with `--linear-scaling terms`."""
    if "terms_guard" not in a:
        return
    if a["terms_guard"] != "val":
        raise ValueError(f"arm {arm}: terms_guard must be \"val\", got {a['terms_guard']!r}")
    if _linear_scaling(a.get("argv", [])) != "terms":
        raise ValueError(f"arm {arm}: terms_guard needs --linear-scaling terms in the argv")


def _check_linear(arm: str, a: dict) -> None:
    """Server round 14: `"linear"` must be "ols" or "ridge", on an arm with no other key (no
    `argv`: a linear arm has no MOSES run)."""
    if "linear" not in a:
        return
    if a["linear"] not in sr_linear.METHODS:
        raise ValueError(f"arm {arm}: linear must be \"ols\" or \"ridge\", got {a['linear']!r}")
    if "argv" in a:
        raise ValueError(f"arm {arm}: a linear arm has no argv (no MOSES run)")
    extra = sorted(set(a) - {"linear"})
    if extra:
        raise ValueError(f"arm {arm}: a linear arm has only the key \"linear\", got {extra}")


def expand(manifest: dict, out_root: Path, repo: Path) -> list[dict]:
    ddir = (repo / manifest["datasets_dir"]).resolve()
    for arm, a in manifest["arms"].items():
        _check_linear(arm, a)  # server round 14: refuse the manifest before any run
        _check_standalone(arm, a)  # the IT linkage study
        _check_terms_guard(arm, a)  # server round 13
    clip = manifest.get("clip_test_nmse", False)  # server round 13 (opt-in)
    if not isinstance(clip, bool):
        raise ValueError(f"clip_test_nmse must be true or false, got {clip!r}")
    plain = manifest.get("plain_nmse", False)  # backlog 12.2 (opt-in)
    if not isinstance(plain, bool):
        raise ValueError(f"plain_nmse must be true or false, got {plain!r}")
    anytime = manifest.get("anytime", False)  # server round 16 (opt-in)
    if not isinstance(anytime, bool):
        raise ValueError(f"anytime must be true or false, got {anytime!r}")
    jobs = []
    for ds in manifest["datasets"]:
        for arm, a in manifest["arms"].items():
            if "linear" in a:
                # server round 14: a linear baseline (no binary, no argv, no budget)
                jobs += [{
                    "run_id": f"{ds}__{arm}__s{seed}", "task": ds, "arm": arm, "seed": seed,
                    "linear": a["linear"],
                    "split_dir": str(ddir / ds / "splits" / f"s{seed}"),
                    "wall_cap_s": manifest["wall_cap_s"],
                    "mem_limit_mb": manifest.get("mem_limit_mb"),
                    "run_dir": str(out_root / manifest["name"] / "runs" / f"{ds}__{arm}__s{seed}"),
                    **({"clip_test_nmse": True} if clip else {}),
                    **({"plain_nmse": True} if plain else {}),
                } for seed in manifest["splits"]]
                continue
            if "standalone" in a:
                # the IT linkage study: stand-alone GOMEA (no binary, no argv, no budget)
                jobs += [{
                    "run_id": f"{ds}__{arm}__s{seed}", "task": ds, "arm": arm, "seed": seed,
                    "standalone": dict(a["standalone"]),
                    "split_dir": str(ddir / ds / "splits" / f"s{seed}"),
                    "wall_cap_s": manifest["wall_cap_s"],
                    "mem_limit_mb": manifest.get("mem_limit_mb"),
                    "run_dir": str(out_root / manifest["name"] / "runs" / f"{ds}__{arm}__s{seed}"),
                    **({"clip_test_nmse": True} if clip else {}),
                    **({"plain_nmse": True} if plain else {}),
                } for seed in manifest["splits"]]
                continue
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
                        **({"terms_guard": a["terms_guard"]} if "terms_guard" in a else {}),
                        **({"clip_test_nmse": True} if clip else {}),
                        **({"plain_nmse": True} if plain else {}),
                        **({"anytime": True} if anytime else {}),
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
