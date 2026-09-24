"""The GOMEA benchmark harness runner (Task 9).

CLI: `python -m benchmarks.gomea.runner MANIFEST [--workers N] [--only REGEX]
[--allow-dirty]`, run from the repository root.

Executes every run a manifest expands to (see `manifest.expand_runs`) in its
own freshly-spawned child process (`multiprocessing.get_context("spawn")` --
one process per run, since the MOSES RNG and caches are process-global), with
IDENTICAL scorer instrumentation (`instrument.py`) installed inside every
child. The parent enforces `wall_cap_s` per run by killing a child that runs
too long; the resulting run is CENSORED (`status: "timeout"`), with
`best_raw`/`best_pen`/`best_cpx`/`n_calls`/`evals_to_target` recovered
best-effort from the run's own streamed `trace.jsonl` rather than discarded
(fix round 1, finding 4) -- note the wall-cap clock starts at process spawn,
so it includes interpreter startup and import time, which is identical for
every arm and does not bias the comparison. Resumable: a run whose
`summary.json` already exists with `status` in `{"ok", "timeout", "error"}`
is skipped. `SIGINT` stops launching new runs, terminates active children,
and exits cleanly -- runs that had already completed keep their summaries.

Every run is GUARANTEED to end up with a `summary.json`: a child that exits
(for any reason, including a hard crash) without writing one itself gets a
synthesized `status: "error"` summary from the PARENT (fix round 1,
finding 3), and the runner's own process exit code is non-zero if any run in
the manifest -- this invocation's or an earlier, resumed one -- ended up
anything other than `status: "ok"`.

No run directory is ever the process's own working directory: `run_dir` is
always an ABSOLUTE path (`manifest.result_dir`/`run_dir_for`), so no chdir is
needed to write into it. The one thing that DOES need the process's actual
cwd is a relative task argv (e.g. `-i tests/fixtures/...`); each CHILD
chdirs, once, to the manifest's "launch directory" (the repository root by
default -- see `main`'s `launch_dir` parameter) before calling into MOSES,
and MOSES's own default relative `-f moses.log` output is isolated per run by
an explicit, runner-added `-f <run_dir>/moses.log` suffix (fix round 1,
finding 1) -- never by chdir-ing into `run_dir` itself, which used to break
every OTHER relative path in the run's own argv.
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

from benchmarks.gomea.instrument import ScorerTrace, instrument_scorer, summarize_events
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


def _write_json_atomic(path: Path, data: object) -> None:
    """Fix round 1, finding 7: write to a sibling temp file and `os.replace`
    it into place, so `summary.json` -- the file `_is_resumable_done` polls
    to decide a run is finished -- never exists half-written on disk."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


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


def _read_trace_events(run_dir: Path) -> list[dict]:
    """Read back `<run_dir>/trace.jsonl` (the same events
    `ScorerTrace.record`'s `sink` streamed to disk as they happened), for
    `summarize_events` to recover from. Skips a malformed line rather than
    raising -- a process killed mid-write could leave its LAST line
    truncated (unlikely with `flush()` after every line, but not
    impossible), and this must never be the reason a summary fails to
    write."""
    trace_path = run_dir / "trace.jsonl"
    events: list[dict] = []
    if not trace_path.exists():
        return events
    try:
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    events.append(ev)
    except OSError:
        return events
    return events


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
        "censored": False,
    }


def _child_main(run: dict) -> None:
    """Entry point for one run's child process (top-level, so it is
    picklable by reference under `multiprocessing.get_context("spawn")`).
    Writes `config.json`, `stdout.txt`, `trace.jsonl` (streamed as the run
    progresses) and, LAST and atomically, `summary.json` under
    `run["run_dir"]`, then returns (the process then exits).
    """
    # Fix round 1, finding 1: chdir to the manifest's launch directory --
    # NOT to `run_dir` -- so a relative task argv (`-i tests/fixtures/...`)
    # resolves exactly as it would running `moses` by hand from that
    # directory. `run_dir` itself is always absolute, so nothing else here
    # depends on the process's cwd at all.
    if run.get("launch_dir"):
        os.chdir(run["launch_dir"])

    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    moses_log = run_dir / "moses.log"
    # Fix round 1, finding 1: isolate MOSES's own default relative `-f
    # moses.log` output with an explicit, absolute, runner-added suffix --
    # every worker would otherwise race the SAME relative path in the
    # shared launch directory.
    full_argv = list(run["argv"]) + ["-f", str(moses_log)]
    _write_json(
        run_dir / "config.json",
        {
            "run_id": run["run_id"],
            "task": run["task"],
            "arm": run["arm"],
            "seed": run["seed"],
            "argv": run["argv"],
            "resolved_argv": full_argv,
            "moses_log": str(moses_log),
            "target_raw": run["target_raw"],
        },
    )

    from pymoses.moses.main.moses_exec import moses_exec

    status = "ok"
    exit_code = 0
    error_text: str | None = None
    stdout_path = run_dir / "stdout.txt"
    trace_path = run_dir / "trace.jsonl"
    start = time.time()

    with open(stdout_path, "w") as out_f, open(trace_path, "w") as trace_f:

        def _emit(ev: dict) -> None:
            # Fix round 1, finding 4: stream every improvement to disk
            # immediately, so a run killed at the wall cap (or one that
            # crashes outright) leaves a usable partial trace behind for
            # the parent to recover from, instead of an empty file.
            trace_f.write(json.dumps(ev, sort_keys=True) + "\n")
            trace_f.flush()

        trace = ScorerTrace(sink=_emit)
        with instrument_scorer(trace), contextlib.redirect_stdout(out_f):
            try:
                rc = moses_exec(["moses", *full_argv])
                exit_code = rc if isinstance(rc, int) else 0
            except SystemExit as exc:
                code = exc.code
                exit_code = code if isinstance(code, int) else (0 if code is None else 1)
            except Exception:
                status = "error"
                error_text = traceback.format_exc()
                exit_code = 1
    # `trace_f` (and `out_f`) are closed here, before the summary is
    # written -- fix round 1, finding 7's "trace first, summary last".
    wall_s = time.time() - start

    # Fix round 1, finding 2: `moses_exec` returning a non-zero code
    # in-band (e.g. `-H <unsupported>` -> 1, moses_exec.py's own
    # documented in-band failure path) is a real failure, not `"ok"`, even
    # though nothing raised.
    if status == "ok" and exit_code != 0:
        status = "error"
        error_text = f"moses_exec returned a non-zero exit code ({exit_code}) without raising"

    summary = _build_summary(
        run, trace, status, exit_code, error_text, wall_s, run_dir, stdout_path
    )
    _write_json_atomic(run_dir / "summary.json", summary)


def _recovered_summary(
    run: dict, status: str, exit_code: int | None, wall_s: float, error_text: str
) -> dict:
    """Shared by `_write_timeout_summary` and `_write_missing_summary`:
    both are written by the PARENT, about a child it no longer has any
    in-memory state for, so both recover what they can from the child's own
    streamed `trace.jsonl` (fix round 1, finding 4) rather than reporting
    an empty result. `censored: true` marks both as right-censored --
    `analyze.py` treats a `None` `best_raw` here as the worst observed
    value in its task, not as missing data to be silently dropped."""
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    recovered = summarize_events(_read_trace_events(run_dir), run["target_raw"])
    return {
        "run_id": run["run_id"],
        "task": run["task"],
        "arm": run["arm"],
        "seed": run["seed"],
        "argv": run["argv"],
        "status": status,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "n_calls": recovered["n_calls"],
        "best_raw": recovered["best_raw"],
        "best_pen": recovered["best_pen"],
        "best_cpx": recovered["best_cpx"],
        "success": False,
        "evals_to_target": recovered["evals_to_target"],
        "time_to_target": recovered["time_to_target"],
        "stdout_best_line": _first_nonempty_line(run_dir / "stdout.txt"),
        "error": error_text,
        "gomea": _read_gomea_events(run_dir),
        "censored": True,
    }


def _write_timeout_summary(run: dict, wall_s: float) -> None:
    run_dir = Path(run["run_dir"])
    if (run_dir / "summary.json").exists():
        # Fix round 1, finding 6: the child raced ahead of the wall-cap
        # check and already finished (successfully or not) -- keep ITS
        # summary rather than clobbering it with a synthetic timeout.
        return
    summary = _recovered_summary(
        run, "timeout", None, wall_s,
        "timeout: child process exceeded wall_cap_s and was killed",
    )
    _write_json_atomic(run_dir / "summary.json", summary)


def _write_missing_summary(run: dict, exitcode: int | None, wall_s: float) -> None:
    """Fix round 1, finding 3: a child that exited (crashed, was
    OOM-killed, segfaulted, ...) WITHOUT writing its own summary.json must
    not simply vanish from the results."""
    run_dir = Path(run["run_dir"])
    if (run_dir / "summary.json").exists():
        return  # lost a benign race against the child's own final write
    summary = _recovered_summary(
        run, "error", exitcode, wall_s,
        f"child process exited (code {exitcode!r}) without writing summary.json",
    )
    _write_json_atomic(run_dir / "summary.json", summary)


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


def run_manifest(
    manifest: dict,
    workers: int,
    only: str | None,
    wall_cap_s: float,
    launch_dir: str | Path | None = None,
    child_target=None,
) -> int:
    """`launch_dir` (default: the repository root) is the directory a
    relative `output_root` and every run's relative task argv resolve
    against; forwarded to `manifest.expand_runs` for path resolution and
    recorded into `environment.json`, and each spawned child chdirs to it
    (never to its own `run_dir`) before calling into MOSES.

    `child_target`, if given, replaces `_child_main` as the per-run process
    entry point -- a testing seam (fix round 1, finding 10) for exercising
    the "child died without writing summary.json" path deterministically,
    without needing a real MOSES run to actually crash.

    Returns 0 if every run in the manifest (this invocation's newly-run
    ones AND any resumed-as-already-done ones) ended with `status: "ok"`,
    130 if interrupted by `SIGINT`, else 1 (fix round 1, finding 3).
    """
    launch_dir = Path(launch_dir).resolve() if launch_dir is not None else Path.cwd()

    rd = result_dir(manifest, base=launch_dir)
    rd.mkdir(parents=True, exist_ok=True)
    _write_json(rd / "manifest.json", manifest)
    write_environment(rd, launch_dir=launch_dir)

    all_runs = expand_runs(manifest, base=launch_dir)
    for run in all_runs:
        run["launch_dir"] = str(launch_dir)
    if only:
        pattern = re.compile(only)
        all_runs = [r for r in all_runs if pattern.search(r["run_id"])]

    pending = deque(r for r in all_runs if not _is_resumable_done(r))
    total = len(all_runs)
    skipped = total - len(pending)
    if skipped:
        print(f"resuming: {skipped}/{total} run(s) already complete, skipping")

    target = child_target if child_target is not None else _child_main
    ctx = mp.get_context("spawn")
    active: dict[str, tuple] = {}
    completed = skipped
    interrupted = False

    try:
        while pending or active:
            while pending and len(active) < max(workers, 1):
                run = pending.popleft()
                p = ctx.Process(target=target, args=(run,))
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
                    if not (Path(run["run_dir"]) / "summary.json").exists():
                        _write_missing_summary(run, p.exitcode, elapsed)
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

    if interrupted:
        return 130
    any_bad = any(_read_status(r) != "ok" for r in all_runs)
    return 1 if any_bad else 0


def main(argv: list[str] | None = None, *, launch_dir: str | Path | None = None) -> int:
    """`launch_dir` is a keyword-only override of the default launch
    directory (the repository root), for tests: it lets a test point
    relative task argv / a relative `output_root` at an isolated `tmp_path`
    instead of chdir-ing (and thereby polluting) the real process this
    test suite itself runs in. Real CLI invocations never pass it."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.gomea.runner")
    parser.add_argument("manifest_path")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    # Resolved against the CALLING process's cwd, once, up front -- nothing
    # below ever chdirs the parent's own process (only children do, and
    # only to `resolved_launch_dir`, never to the parent's original cwd).
    manifest_path = Path(args.manifest_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    resolved_launch_dir = Path(launch_dir).resolve() if launch_dir is not None else repo_root

    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.allow_dirty:
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
        manifest,
        workers=workers,
        only=args.only,
        wall_cap_s=manifest["wall_cap_s"],
        launch_dir=resolved_launch_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
