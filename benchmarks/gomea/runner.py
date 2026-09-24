"""The GOMEA benchmark harness runner (Task 9).

CLI: `python -m benchmarks.gomea.runner MANIFEST [--workers N] [--only REGEX]
[--allow-dirty]`, run from the repository root.

Executes every run a manifest expands to (see `manifest.expand_runs`) in its
own freshly-spawned child process (`multiprocessing.get_context("spawn")` --
one process per run, since the MOSES RNG and caches are process-global), with
IDENTICAL scorer instrumentation (`instrument.py`) installed inside every
child. The parent enforces `wall_cap_s` per run by killing a child that runs
too long; the resulting run is CENSORED (`status: "timeout"`), and its
partial trace, if any, is discarded. Resumable: a run whose `summary.json`
already exists with `status` in `{"ok", "timeout", "error"}` is skipped.
`SIGINT` stops launching new runs, terminates active children, and exits
cleanly -- runs that had already completed keep their summaries.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from benchmarks.gomea.instrument import ScorerTrace, instrument_scorer
from benchmarks.gomea.manifest import (
    expand_runs,
    git_head,
    load_manifest,
    result_dir,
    write_environment,
)

__all__ = ["main", "run_manifest"]


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _first_nonempty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped.strip():
                    return stripped
    except OSError:
        return None
    return None


def _read_gomea_events(run_dir: Path) -> dict | None:
    """Best-effort summary of `<run_dir>/events.jsonl`, if a GOMEA arm wrote
    one (`--gomea-log`, wired by `manifest.expand_runs`). Reads defensively:
    the event format is produced by another task's code, so a malformed or
    unexpected line is skipped rather than raising. Per-type line counts,
    plus the `termination_reason` of every event of type "end"."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None
    counts: dict[str, int] = {}
    termination_reasons: list = []
    try:
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                etype = ev.get("type")
                if etype is None:
                    continue
                counts[etype] = counts.get(etype, 0) + 1
                if etype == "end" and "termination_reason" in ev:
                    termination_reasons.append(ev["termination_reason"])
    except OSError:
        return None
    return {"event_counts": counts, "termination_reasons": termination_reasons}


def _build_summary(
    run: dict,
    trace: ScorerTrace,
    status: str,
    exit_code: int,
    error_text: str | None,
    wall_s: float,
    run_dir: Path,
    stdout_path: Path,
) -> dict:
    best_raw = trace.best_raw
    target_raw = run["target_raw"]
    success = status == "ok" and best_raw is not None and best_raw >= target_raw
    hit = trace.first_hit(target_raw) if best_raw is not None else None
    evals_to_target, time_to_target = hit if hit is not None else (None, None)
    return {
        "run_id": run["run_id"],
        "task": run["task"],
        "arm": run["arm"],
        "seed": run["seed"],
        "argv": run["argv"],
        "status": status,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "n_calls": trace.n_calls,
        "best_raw": best_raw,
        "best_pen": trace.best_pen(),
        "best_cpx": trace.best_cpx(),
        "success": success,
        "evals_to_target": evals_to_target,
        "time_to_target": time_to_target,
        "stdout_best_line": _first_nonempty_line(stdout_path),
        "error": error_text,
        "gomea": _read_gomea_events(run_dir),
    }


def _child_main(run: dict) -> None:
    """Entry point for one run's child process (top-level, so it is
    picklable by reference under `multiprocessing.get_context("spawn")`).
    Writes `config.json`, `stdout.txt`, `trace.jsonl` and `summary.json`
    under `run["run_dir"]`, then returns (the process then exits)."""
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config.json",
        {
            "run_id": run["run_id"],
            "task": run["task"],
            "arm": run["arm"],
            "seed": run["seed"],
            "argv": run["argv"],
            "target_raw": run["target_raw"],
        },
    )

    # Isolate this run's own relative-path outputs (`problem_params.py`'s
    # own `-f moses.log` default writes to the CWD) -- every child would
    # otherwise race the SAME relative path against every sibling worker
    # started from the same working directory.
    os.chdir(run_dir)

    from pymoses.moses.main.moses_exec import moses_exec

    trace = ScorerTrace()
    status = "ok"
    exit_code = 0
    error_text: str | None = None
    stdout_path = run_dir / "stdout.txt"
    start = time.time()
    with open(stdout_path, "w") as out_f:
        with instrument_scorer(trace), contextlib.redirect_stdout(out_f):
            try:
                rc = moses_exec(["moses", *run["argv"]])
                exit_code = rc if isinstance(rc, int) else 0
            except SystemExit as exc:
                code = exc.code
                exit_code = code if isinstance(code, int) else (0 if code is None else 1)
            except Exception:
                status = "error"
                error_text = traceback.format_exc()
                exit_code = 1
    wall_s = time.time() - start

    summary = _build_summary(
        run, trace, status, exit_code, error_text, wall_s, run_dir, stdout_path
    )
    _write_json(run_dir / "summary.json", summary)
    with open(run_dir / "trace.jsonl", "w") as f:
        for ev in trace.events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")


def _write_timeout_summary(run: dict, wall_s: float) -> None:
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()  # discard the partial trace
    summary = {
        "run_id": run["run_id"],
        "task": run["task"],
        "arm": run["arm"],
        "seed": run["seed"],
        "argv": run["argv"],
        "status": "timeout",
        "exit_code": None,
        "wall_s": wall_s,
        "n_calls": None,
        "best_raw": None,
        "best_pen": None,
        "best_cpx": None,
        "success": False,
        "evals_to_target": None,
        "time_to_target": None,
        "stdout_best_line": _first_nonempty_line(run_dir / "stdout.txt"),
        "error": "timeout: child process exceeded wall_cap_s and was killed",
        "gomea": _read_gomea_events(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)


def _is_resumable_done(run: dict) -> bool:
    summary_path = Path(run["run_dir"]) / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("status") in ("ok", "timeout", "error")


def _read_status(run: dict) -> str:
    summary_path = Path(run["run_dir"]) / "summary.json"
    try:
        with open(summary_path) as f:
            return json.load(f).get("status", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def run_manifest(manifest: dict, workers: int, only: str | None, wall_cap_s: float) -> int:
    rd = result_dir(manifest)
    rd.mkdir(parents=True, exist_ok=True)
    _write_json(rd / "manifest.json", manifest)
    write_environment(rd)

    all_runs = expand_runs(manifest)
    if only:
        pattern = re.compile(only)
        all_runs = [r for r in all_runs if pattern.search(r["run_id"])]

    pending = deque(r for r in all_runs if not _is_resumable_done(r))
    total = len(all_runs)
    skipped = total - len(pending)
    if skipped:
        print(f"resuming: {skipped}/{total} run(s) already complete, skipping")

    ctx = mp.get_context("spawn")
    active: dict[str, tuple] = {}
    completed = skipped
    interrupted = False

    try:
        while pending or active:
            while pending and len(active) < max(workers, 1):
                run = pending.popleft()
                p = ctx.Process(target=_child_main, args=(run,))
                p.start()
                active[run["run_id"]] = (p, time.monotonic(), run)

            if not active:
                break

            time.sleep(0.05)
            finished = []
            for run_id, (p, start_t, run) in active.items():
                elapsed = time.monotonic() - start_t
                if not p.is_alive():
                    p.join()
                    finished.append(run_id)
                elif elapsed > wall_cap_s:
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join()
                    _write_timeout_summary(run, elapsed)
                    finished.append(run_id)

            for run_id in finished:
                _, _, run = active.pop(run_id)
                completed += 1
                print(f"[{completed}/{total}] {run_id}: {_read_status(run)}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted: stopping launches and terminating active runs...", file=sys.stderr)
        for _run_id, (p, _start_t, _run) in active.items():
            p.terminate()
        for _run_id, (p, _start_t, _run) in active.items():
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()

    return 130 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.gomea.runner")
    parser.add_argument("manifest_path")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.allow_dirty:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            head = git_head(repo_root)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"could not determine git HEAD: {exc}", file=sys.stderr)
            return 1
        if head != manifest["commit"]:
            print(
                f"manifest commit {manifest['commit']} does not match HEAD {head} "
                "(pass --allow-dirty to run anyway)",
                file=sys.stderr,
            )
            return 1

    # Set once, in the PARENT, before any child is spawned: `PYTHONHASHSEED`
    # only takes effect at a fresh interpreter's startup, so setting it
    # inside an already-running child (or here, after children exist) would
    # be too late. `multiprocessing`'s spawn context launches each child by
    # inheriting the parent's `os.environ` at `Process.start()` time, so
    # setting it here, once, before the scheduling loop starts, reaches
    # every child.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"

    workers = args.workers if args.workers is not None else manifest["workers"]
    return run_manifest(
        manifest, workers=workers, only=args.only, wall_cap_s=manifest["wall_cap_s"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
