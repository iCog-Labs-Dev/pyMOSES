"""The GOMEA benchmark harness runner (Task 9).

CLI: `python -m benchmarks.gomea.runner MANIFEST [--workers N] [--only REGEX]
[--allow-dirty] [--output-root PATH] [--mem-limit-mb N] [--dry-run]`, run
from the repository root.

Final fix wave additions:

* Time evidence (F1): each child measures `time.process_time()` around
  `moses_exec` (`cpu_s`) and its peak RSS (`ru_maxrss_kb`, from
  `getrusage(RUSAGE_SELF)`), and the trace records CPU time per
  improvement (`cpu_to_target`). Runs are scheduled in a deterministic
  shuffled order (`scheduled_order`, seeded by the manifest hash) recorded
  in `run_order.json`, so arms do not cluster in time.
* Launch settings (F7): `--output-root` and `--workers` override the
  manifest; neither is part of `manifest_hash`, so the results directory
  name is the same on any machine. The effective values go to
  `environment.json`.
* Code check (F8): the manifest `commit` must be an ancestor of HEAD with no
  code change since and no local code modification
  (`manifest.check_code_commit`); otherwise the runner refuses unless
  `--allow-dirty`. The check result goes to `environment.json`.
* Memory (F9): `--mem-limit-mb N` caps each child's address space
  (`RLIMIT_AS`) before MOSES loads; a `MemoryError` is recorded as
  `status: "error"`, `error: "memory_limit"`, and the child exits
  non-zero. `--dry-run` prints the plan and exits.
* Memory inside scoring (F10): classic's `complexity_based_scorer` turns
  ANY `Exception` from building or scoring a candidate into the worst
  score, which would hide a memory-cap failure. Every child, for every arm,
  runs under `instrument.guard_memory_errors`, which re-raises a
  `MemoryError` from `get_candidate`/`get_cscore` as `MemoryLimitHit` (a
  `BaseException` that handler cannot catch). An ok run whose
  `ru_maxrss_kb` reached 90% of the limit is flagged `near_memory_limit`.
* CPU time of a TIMEOUT: `cpu_s` is read from `/proc` just before the kill
  (`cpu_s_source: "proc"`) and so INCLUDES interpreter start-up and
  imports (about a second); an ok/error run's `cpu_s` is `process_time`
  around `moses_exec` only (`cpu_s_source: "process_time"`).

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
import random
import re
import resource
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from benchmarks.gomea.instrument import (
    MemoryLimitHit,
    ScorerTrace,
    guard_memory_errors,
    instrument_scorer,
    summarize_events,
)
from benchmarks.gomea.manifest import (
    check_code_commit,
    expand_runs,
    load_manifest,
    manifest_hash,
    result_dir,
    write_environment,
)

__all__ = ["main", "run_manifest", "scheduled_order"]

# Freed when a child hits its memory limit, so it has room to record that.
_MEMORY_RESERVE_BYTES = 8 * 2**20
# Written into the run directory as soon as a child catches a MemoryError,
# so the parent can still say "memory_limit" if the child then dies.
_MEMORY_MARKER = "memory_limit.marker"


def scheduled_order(manifest: dict, runs: list[dict]) -> list[dict]:
    """`runs` in the order they are scheduled: a deterministic shuffle with
    `random.Random(<int of the first 8 hex digits of manifest_hash>)`, so
    the arms of one task do not cluster in time (the host's P/E cores and
    its load change over a campaign). Returns a new list."""
    order = list(runs)
    random.Random(_order_seed(manifest)).shuffle(order)
    return order


def _order_seed(manifest: dict) -> int:
    return int(manifest_hash(manifest)[:8], 16)


def _proc_usage(pid: int | None) -> tuple[float | None, int | None]:
    """Best-effort (cpu seconds, peak RSS kB) of a live process from
    `/proc` (Linux) -- for a child the parent is about to kill at the wall
    cap, which never gets to measure itself. `(None, None)` elsewhere."""
    if pid is None:
        return None, None
    cpu_s: float | None = None
    rss_kb: int | None = None
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        # utime and stime are fields 14 and 15 (1-based) of the full line;
        # after the ")" split, field 3 is at index 0.
        cpu_s = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    rss_kb = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return cpu_s, rss_kb


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
    cpu_s: float | None = None,
) -> dict:
    best_raw = trace.best_raw
    target_raw = run["target_raw"]
    success = status == "ok" and best_raw is not None and best_raw >= target_raw
    hit = trace.first_hit_event(target_raw) if best_raw is not None else None
    evals_to_target = hit["n"] if hit is not None else None
    time_to_target = hit["t"] if hit is not None else None
    cpu_to_target = hit["cpu"] if hit is not None else None
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    near = status == "ok" and _near_memory_limit(rss_kb, run.get("mem_limit_mb"))
    return {
        "run_id": run["run_id"],
        "task": run["task"],
        "arm": run["arm"],
        "seed": run["seed"],
        "argv": run["argv"],
        "status": status,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "cpu_s_source": "process_time",
        "ru_maxrss_kb": rss_kb,
        "mem_limit_mb": run.get("mem_limit_mb"),
        "near_memory_limit": near,
        "n_calls": trace.n_calls,
        "best_raw": best_raw,
        "best_pen": trace.best_pen(),
        "best_cpx": trace.best_cpx(),
        "success": success,
        "evals_to_target": evals_to_target,
        "time_to_target": time_to_target,
        "cpu_to_target": cpu_to_target,
        "stdout_best_line": _first_nonempty_line(stdout_path),
        "error": error_text,
        "gomea": _read_gomea_events(run_dir),
        "censored": False,
    }


def _near_memory_limit(rss_kb: int | None, mem_limit_mb: int | None) -> bool:
    """Peak RSS at or above 90% of the `--mem-limit-mb` cap (F10)."""
    if not mem_limit_mb or rss_kb is None:
        return False
    return rss_kb >= 0.9 * mem_limit_mb * 1024


def _child_main(run: dict) -> None:
    """Entry point for one run's child process (top-level, so it is
    picklable by reference under `multiprocessing.get_context("spawn")`).

    With `run["mem_limit_mb"]` set, caps this process's address space
    (`RLIMIT_AS`, soft and hard) FIRST -- before MOSES is imported (this
    module imports no MOSES code) -- after setting aside a small reserve
    that is freed on a `MemoryError`, so the child has room to record the
    failure as `status: "error"`, `error: "memory_limit"`. A memory-limit
    failure (a `MemoryError`, or `MemoryLimitHit` from inside scoring)
    makes the child exit with code 1 after writing its summary.
    """
    reserve: list[bytearray] = []
    limit_mb = run.get("mem_limit_mb")
    if limit_mb:
        reserve.append(bytearray(_MEMORY_RESERVE_BYTES))
        limit = int(limit_mb) * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    start = time.time()
    try:
        hit = _child_run(run, reserve)
    except (MemoryError, MemoryLimitHit):
        # Outside `moses_exec` itself (e.g. writing config.json).
        reserve.clear()
        _record_memory_limit(run, time.time() - start)
        hit = True
    if hit:
        raise SystemExit(1)


def _record_memory_limit(run: dict, wall_s: float) -> None:
    run_dir = Path(run["run_dir"])
    with contextlib.suppress(OSError):
        (run_dir / _MEMORY_MARKER).write_text("memory_limit\n")
    summary = _recovered_summary(run, "error", 1, wall_s, "memory_limit")
    summary["censored"] = False
    summary["near_memory_limit"] = False
    _write_json_atomic(run_dir / "summary.json", summary)


def _child_run(run: dict, reserve: list[bytearray]) -> bool:
    """The body of `_child_main`; returns True if the run hit the memory
    limit. Writes `config.json`, `stdout.txt`,
    `trace.jsonl` (streamed as the run progresses) and, LAST and atomically, `summary.json` under
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

    status = "ok"
    exit_code = 0
    error_text: str | None = None
    stdout_path = run_dir / "stdout.txt"
    trace_path = run_dir / "trace.jsonl"
    start = time.time()
    cpu_s: float | None = None

    with open(stdout_path, "w") as out_f, open(trace_path, "w") as trace_f:

        def _emit(ev: dict) -> None:
            # Fix round 1, finding 4: stream every improvement to disk
            # immediately, so a run killed at the wall cap (or one that
            # crashes outright) leaves a usable partial trace behind for
            # the parent to recover from, instead of an empty file.
            trace_f.write(json.dumps(ev, sort_keys=True) + "\n")
            trace_f.flush()

        trace = ScorerTrace(sink=_emit)
        try:
            # Imported here, after `_child_main` set any memory limit.
            from pymoses.moses.main.moses_exec import moses_exec

            with (
                guard_memory_errors(),
                instrument_scorer(trace),
                contextlib.redirect_stdout(out_f),
            ):
                cpu0 = time.process_time()
                try:
                    rc = moses_exec(["moses", *full_argv])
                    exit_code = rc if isinstance(rc, int) else 0
                except SystemExit as exc:
                    code = exc.code
                    exit_code = code if isinstance(code, int) else (0 if code is None else 1)
                finally:
                    cpu_s = time.process_time() - cpu0
        except (MemoryError, MemoryLimitHit):
            reserve.clear()
            with contextlib.suppress(OSError):
                (run_dir / _MEMORY_MARKER).write_text("memory_limit\n")
            status = "error"
            error_text = "memory_limit"
            exit_code = 1
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
        run, trace, status, exit_code, error_text, wall_s, run_dir, stdout_path, cpu_s
    )
    _write_json_atomic(run_dir / "summary.json", summary)
    return error_text == "memory_limit"


def _recovered_summary(
    run: dict,
    status: str,
    exit_code: int | None,
    wall_s: float,
    error_text: str,
    usage: tuple[float | None, int | None] = (None, None),
) -> dict:
    """Shared by `_write_timeout_summary` and `_write_missing_summary`:
    both are written by the PARENT, about a child it no longer has any
    in-memory state for, so both recover what they can from the child's own
    streamed `trace.jsonl` (fix round 1, finding 4) rather than reporting
    an empty result. `censored: true` marks both as right-censored --
    `analyze.py` treats a `None` `best_raw` here as the worst observed
    value in its task, not as missing data to be silently dropped.

    A TIMEOUT whose recovered trace already reached `target_raw` is a
    success (F5), with its recovered `evals_to_target`: the target was hit
    within the budget, only the run's own end was cut off. An error never
    is. `usage` is the child's (cpu seconds, peak RSS kB) as read from
    `/proc` just before a wall-cap kill (`cpu_s_source: "proc"`, which
    includes interpreter startup), or `(None, None)`."""
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    recovered = summarize_events(_read_trace_events(run_dir), run["target_raw"])
    success = status == "timeout" and recovered["evals_to_target"] is not None
    cpu_s, rss_kb = usage
    return {
        "run_id": run["run_id"],
        "task": run["task"],
        "arm": run["arm"],
        "seed": run["seed"],
        "argv": run["argv"],
        "status": status,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "cpu_s_source": "proc" if cpu_s is not None else None,
        "ru_maxrss_kb": rss_kb,
        "mem_limit_mb": run.get("mem_limit_mb"),
        "near_memory_limit": False,
        "n_calls": recovered["n_calls"],
        "best_raw": recovered["best_raw"],
        "best_pen": recovered["best_pen"],
        "best_cpx": recovered["best_cpx"],
        "success": success,
        "evals_to_target": recovered["evals_to_target"],
        "time_to_target": recovered["time_to_target"],
        "cpu_to_target": recovered["cpu_to_target"],
        "stdout_best_line": _first_nonempty_line(run_dir / "stdout.txt"),
        "error": error_text,
        "gomea": _read_gomea_events(run_dir),
        "censored": True,
    }


def _write_timeout_summary(
    run: dict, wall_s: float, usage: tuple[float | None, int | None] = (None, None)
) -> None:
    run_dir = Path(run["run_dir"])
    if (run_dir / "summary.json").exists():
        # Fix round 1, finding 6: the child raced ahead of the wall-cap
        # check and already finished (successfully or not) -- keep ITS
        # summary rather than clobbering it with a synthetic timeout.
        return
    summary = _recovered_summary(
        run, "timeout", None, wall_s,
        "timeout: child process exceeded wall_cap_s and was killed",
        usage,
    )
    _write_json_atomic(run_dir / "summary.json", summary)


def _write_missing_summary(run: dict, exitcode: int | None, wall_s: float) -> None:
    """Fix round 1, finding 3: a child that exited (crashed, was
    OOM-killed, segfaulted, ...) WITHOUT writing its own summary.json must
    not simply vanish from the results."""
    run_dir = Path(run["run_dir"])
    if (run_dir / "summary.json").exists():
        return  # lost a benign race against the child's own final write
    if (run_dir / _MEMORY_MARKER).exists():
        error_text = "memory_limit"  # the child caught a MemoryError, then died
    else:
        error_text = f"child process exited (code {exitcode!r}) without writing summary.json"
    summary = _recovered_summary(run, "error", exitcode, wall_s, error_text)
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


def _plan(
    manifest: dict,
    only: str | None,
    launch_dir: Path,
    output_root: str | Path | None,
    mem_limit_mb: int | None,
) -> tuple[dict, Path, list[dict]]:
    """The effective manifest (`output_root` overridden, F7), its results
    directory, and every selected run in scheduled order (F1)."""
    effective = dict(manifest)
    if output_root is not None:
        effective["output_root"] = str(Path(output_root).resolve())
    rd = result_dir(effective, base=launch_dir)
    all_runs = expand_runs(effective, base=launch_dir)
    for run in all_runs:
        run["launch_dir"] = str(launch_dir)
        run["mem_limit_mb"] = mem_limit_mb
    if only:
        pattern = re.compile(only)
        all_runs = [r for r in all_runs if pattern.search(r["run_id"])]
    return effective, rd, scheduled_order(manifest, all_runs)


def run_manifest(
    manifest: dict,
    workers: int,
    only: str | None,
    wall_cap_s: float,
    launch_dir: str | Path | None = None,
    child_target=None,
    output_root: str | Path | None = None,
    mem_limit_mb: int | None = None,
    env_extra: dict | None = None,
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

    `output_root` overrides the manifest's (F7); `manifest.json` in the
    results directory is still the manifest as given. `mem_limit_mb` caps
    every child's address space (F9). `env_extra` (the code-commit check)
    is merged into `environment.json`, next to the effective
    `output_root`, `workers` and `mem_limit_mb`. Runs are launched in
    `scheduled_order`, recorded in `run_order.json`.

    Returns 0 if every run in the manifest (this invocation's newly-run
    ones AND any resumed-as-already-done ones) ended with `status: "ok"`,
    130 if interrupted by `SIGINT`, else 1 (fix round 1, finding 3).
    """
    launch_dir = Path(launch_dir).resolve() if launch_dir is not None else Path.cwd()
    effective, rd, all_runs = _plan(manifest, only, launch_dir, output_root, mem_limit_mb)

    rd.mkdir(parents=True, exist_ok=True)
    _write_json(rd / "manifest.json", manifest)
    write_environment(
        rd,
        launch_dir=launch_dir,
        extra={
            "output_root": str(Path(launch_dir, effective["output_root"]).resolve()),
            "workers": workers,
            "manifest_output_root": manifest["output_root"],
            "manifest_workers": manifest["workers"],
            "mem_limit_mb": mem_limit_mb,
            "max_load_note": manifest.get("max_load_note"),
            **(env_extra or {}),
        },
    )
    _write_json(
        rd / "run_order.json",
        {"seed": _order_seed(manifest), "order": [r["run_id"] for r in all_runs]},
    )
    if mem_limit_mb:
        print(
            f"warning: --mem-limit-mb {mem_limit_mb} x {max(workers, 1)} workers: "
            f"estimated peak = {mem_limit_mb * max(workers, 1)} MB"
        )

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
                    usage = _proc_usage(p.pid)
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join()
                    _write_timeout_summary(run, elapsed, usage)
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


def main(
    argv: list[str] | None = None,
    *,
    launch_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> int:
    """`launch_dir` is a keyword-only override of the default launch
    directory (the repository root), for tests: it lets a test point
    relative task argv / a relative `output_root` at an isolated `tmp_path`
    instead of chdir-ing (and thereby polluting) the real process this
    test suite itself runs in. `repo_root` likewise overrides the git
    repository the code-commit check (F8) inspects. Real CLI invocations
    never pass either."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.gomea.runner")
    parser.add_argument("manifest_path")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output-root", default=None, help="override the manifest's output_root"
    )
    parser.add_argument(
        "--mem-limit-mb",
        type=int,
        default=None,
        help="cap each run's address space (RLIMIT_AS) at this many MiB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the code check, then exit without running",
    )
    args = parser.parse_args(argv)
    if args.mem_limit_mb is not None and args.mem_limit_mb < 1:
        print("--mem-limit-mb must be >= 1", file=sys.stderr)
        return 1

    # Resolved against the CALLING process's cwd, once, up front -- nothing
    # below ever chdirs the parent's own process (only children do, and
    # only to `resolved_launch_dir`, never to the parent's original cwd).
    manifest_path = Path(args.manifest_path).resolve()
    default_root = Path(__file__).resolve().parents[2]
    repo = Path(repo_root).resolve() if repo_root is not None else default_root
    resolved_launch_dir = Path(launch_dir).resolve() if launch_dir is not None else default_root

    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # F8: the manifest commit is the CODE commit; the manifest itself may
    # be committed later. Refuse unless the code at HEAD is that code.
    try:
        check = check_code_commit(repo, manifest["commit"])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"could not run the code-commit check: {exc}", file=sys.stderr)
        return 1
    refused = not check["ok"] and not args.allow_dirty
    workers = args.workers if args.workers is not None else manifest["workers"]

    if args.dry_run:
        _, rd, runs = _plan(
            manifest, args.only, resolved_launch_dir, args.output_root, args.mem_limit_mb
        )
        done = sum(1 for r in runs if _is_resumable_done(r))
        print(f"result directory: {rd}")
        peak = f"{args.mem_limit_mb * max(workers, 1)} MB" if args.mem_limit_mb else "no limit"
        print(f"workers: {workers}; mem limit: {args.mem_limit_mb} MB; estimated peak = {peak}")
        print(f"already complete: {done}/{len(runs)}")
        verdict = "ok" if check["ok"] else f"FAILED ({check['reason']})"
        print(f"code check: {verdict}; HEAD {check['head']}, manifest {check['manifest_commit']}")
        if refused:
            print("a real launch would refuse (pass --allow-dirty to run anyway)")
        print("runs, in scheduled order:")
        for r in runs:
            print(f"  {r['run_id']}")
        return 1 if refused else 0

    if refused:
        print(
            f"code-commit check failed: {check['reason']} "
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

    return run_manifest(
        manifest,
        workers=workers,
        only=args.only,
        wall_cap_s=manifest["wall_cap_s"],
        launch_dir=resolved_launch_dir,
        output_root=args.output_root,
        mem_limit_mb=args.mem_limit_mb,
        env_extra={
            "head": check["head"],
            "manifest_commit": check["manifest_commit"],
            "code_check": check,
            "allow_dirty": args.allow_dirty,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
