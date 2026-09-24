"""Manifest schema validation, hashing, run expansion and environment capture
for the GOMEA benchmark harness (Task 9).

Fails CLOSED: `load_manifest` refuses a manifest with any missing key or any
`null` value anywhere in its known structure, listing EVERY problem it finds
in one `ValueError` rather than stopping at the first -- so a manifest with
one missing top-level key and one null nested value gets one error message
naming both.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

__all__ = [
    "load_manifest",
    "manifest_hash",
    "expand_runs",
    "write_environment",
    "result_dir",
    "run_dir_for",
    "budget_evals",
    "git_head",
    "is_gomea_arm",
    "check_code_commit",
    "LAUNCH_KEYS",
    "CODE_PATHS",
]

_REQUIRED_TOP_KEYS = (
    "schema_version",
    "name",
    "phase",
    "commit",
    "common_argv",
    "tasks",
    "arms",
    "seeds",
    "budget",
    "wall_cap_s",
    "workers",
    "output_root",
)
# Launch-time settings, not part of the experiment's identity: excluded
# from `manifest_hash`, so the same committed manifest names the same
# `<name>-<hash>` results directory on any machine (`runner.py` can
# override both at launch: `--output-root`, `--workers`).
LAUNCH_KEYS = ("output_root", "workers")

# The paths whose content IS the code under test (`check_code_commit`).
CODE_PATHS = ("pymoses", "benchmarks", "pyproject.toml")

_REQUIRED_TASK_KEYS = ("argv", "target_raw")
_REQUIRED_ARM_KEYS = ("argv",)
_REQUIRED_BUDGET_KEYS = ("argv", "kind")


def _check_present(d: dict, key: str, where: str, errors: list[str]) -> bool:
    if key not in d:
        errors.append(f"{where}: missing required key '{key}'")
        return False
    if d[key] is None:
        errors.append(f"{where}: key '{key}' is null")
        return False
    return True


def _check_argv(value: object, where: str, errors: list[str]) -> None:
    """Fix round 1, finding 8: an `argv` must be a list, and none of its
    elements may be `null` -- a null element would reach `moses_exec` as
    the literal string `"None"` (str-joined into the argv) rather than
    failing loudly here."""
    if not isinstance(value, list):
        errors.append(f"{where} must be an array")
        return
    for i, tok in enumerate(value):
        if tok is None:
            errors.append(f"{where}[{i}] is null")


def _check_numeric(d: dict, key: str, where: str, errors: list[str]) -> None:
    """Fix round 1, finding 8: `wall_cap_s`/`workers`/`target_raw` must be
    numbers -- `bool` is deliberately rejected too, even though Python's
    `bool` is an `int` subclass (`isinstance(True, int)` is `True`), since
    a manifest `true`/`false` there is never a meaningful value."""
    if key not in d or d[key] is None:
        return  # already reported by _check_present
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errors.append(f"{where}: key '{key}' must be numeric, got {type(v).__name__}")


def _check_no_duplicate_seeds(seeds: list, errors: list[str]) -> None:
    seen: set[str] = set()
    dupes: list = []
    for s in seeds:
        # `json.dumps` as the dedupe key handles any JSON-serializable seed
        # value uniformly (ints are the expected case, but nothing here
        # requires it) without risking a `TypeError` from an unhashable one.
        key = json.dumps(s, sort_keys=True)
        if key in seen:
            dupes.append(s)
        seen.add(key)
    if dupes:
        errors.append(f"manifest.seeds contains duplicate value(s): {dupes!r}")


def _validate(m: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(m, dict):
        return [f"manifest root must be a JSON object, got {type(m).__name__}"]

    for key in _REQUIRED_TOP_KEYS:
        _check_present(m, key, "manifest", errors)

    _check_numeric(m, "wall_cap_s", "manifest", errors)
    _check_numeric(m, "workers", "manifest", errors)

    if m.get("tasks") is not None:
        if not isinstance(m["tasks"], dict) or not m["tasks"]:
            errors.append("manifest.tasks must be a non-empty object")
        else:
            for name, cfg in m["tasks"].items():
                where = f"manifest.tasks['{name}']"
                if cfg is None:
                    errors.append(f"{where} is null")
                    continue
                if not isinstance(cfg, dict):
                    errors.append(f"{where} must be an object")
                    continue
                for key in _REQUIRED_TASK_KEYS:
                    _check_present(cfg, key, where, errors)
                if "argv" in cfg and cfg["argv"] is not None:
                    _check_argv(cfg["argv"], f"{where}.argv", errors)
                _check_numeric(cfg, "target_raw", where, errors)

    if m.get("arms") is not None:
        if not isinstance(m["arms"], dict) or not m["arms"]:
            errors.append("manifest.arms must be a non-empty object")
        else:
            for name, cfg in m["arms"].items():
                where = f"manifest.arms['{name}']"
                if cfg is None:
                    errors.append(f"{where} is null")
                    continue
                if not isinstance(cfg, dict):
                    errors.append(f"{where} must be an object")
                    continue
                for key in _REQUIRED_ARM_KEYS:
                    _check_present(cfg, key, where, errors)
                if "argv" in cfg and cfg["argv"] is not None:
                    _check_argv(cfg["argv"], f"{where}.argv", errors)

    if m.get("seeds") is not None:
        if not isinstance(m["seeds"], list) or not m["seeds"]:
            errors.append("manifest.seeds must be a non-empty array")
        else:
            for i, s in enumerate(m["seeds"]):
                if s is None:
                    errors.append(f"manifest.seeds[{i}] is null")
            _check_no_duplicate_seeds(m["seeds"], errors)

    if m.get("budget") is not None:
        if not isinstance(m["budget"], dict):
            errors.append("manifest.budget must be an object")
        else:
            for key in _REQUIRED_BUDGET_KEYS:
                _check_present(m["budget"], key, "manifest.budget", errors)
            if "argv" in m["budget"] and m["budget"]["argv"] is not None:
                _check_argv(m["budget"]["argv"], "manifest.budget.argv", errors)

    if m.get("common_argv") is not None:
        _check_argv(m["common_argv"], "manifest.common_argv", errors)

    # Optional, documentation only (no behaviour): e.g. the machine load
    # the manifest's `workers` was chosen for.
    if "max_load_note" in m and not isinstance(m["max_load_note"], str):
        errors.append("manifest: optional key 'max_load_note' must be a string")

    return errors


def load_manifest(path: str | Path) -> dict:
    """Load and validate a manifest JSON file. Raises `ValueError` listing
    every problem found (missing/null keys, malformed sub-structures) if
    the manifest is invalid; never returns a partially-trusted manifest."""
    p = Path(path)
    text = p.read_text()
    try:
        m = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest {p}: invalid JSON: {exc}") from exc

    errors = _validate(m)
    if errors:
        listing = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"manifest {p} failed validation ({len(errors)} problem(s)):\n{listing}")
    return m


def manifest_hash(m: dict) -> str:
    """sha256 of the manifest's canonical JSON (sort_keys, no whitespace),
    truncated to 12 hex characters. Independent of key insertion order, and
    of the launch-time settings `LAUNCH_KEYS` (`output_root`, `workers`)."""
    identity = {k: v for k, v in m.items() if k not in LAUNCH_KEYS}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def result_dir(m: dict, base: str | Path | None = None) -> Path:
    """`<output_root>/<phase>/<name>-<hash>` -- the top of one manifest's
    output tree (03_EXPERIMENT_PROTOCOL.md Sec 9's suggested layout).

    Always ABSOLUTE. When `output_root` is itself relative, it is resolved
    against `base` (default: the calling process's cwd) -- joining a
    `Path` onto a base discards the base entirely when the joined path is
    already absolute (plain `pathlib` behaviour), so an absolute
    `output_root` is unaffected by `base` either way.

    Fix round 1, finding 1: a relative `output_root` used to resolve
    against whatever directory `runner.py`'s per-run `os.chdir` had most
    recently left the process in -- i.e. a DIFFERENT, run-specific
    directory for every run -- so every run died with `FileNotFoundError`
    writing its own summary. `runner.py` no longer chdirs its own process
    at all; it resolves every run's `run_dir` explicitly against one fixed
    "launch directory" (via `expand_runs`'s own `base`) before ever
    spawning a child, and only the CHILD itself (a fresh, short-lived
    process) chdirs -- to that same launch directory, never to its own
    `run_dir` -- so relative task argv (`-i tests/fixtures/...`) keep
    working exactly as before.
    """
    root = Path(base) if base is not None else Path.cwd()
    return (
        root / m["output_root"] / str(m["phase"]) / f"{m['name']}-{manifest_hash(m)}"
    ).resolve()


def run_dir_for(m: dict, run_id: str, base: str | Path | None = None) -> Path:
    return result_dir(m, base=base) / "runs" / run_id


def budget_evals(budget: dict) -> int | None:
    """Best-effort numeric evaluation budget parsed out of `budget["argv"]`.

    Scans for `-m`/`--max-evals` -- the flags `expand_runs` appends
    verbatim to every run's argv -- and returns the integer that follows.
    Returns `None` when neither flag is present, or the following token is
    not an integer (e.g. a wall-time-only budget); `analyze.py`'s own
    `--budget` CLI override exists for exactly that case.
    """
    argv = budget.get("argv") or []
    for i, tok in enumerate(argv):
        if tok in ("-m", "--max-evals") and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except (TypeError, ValueError):
                return None
    return None


def is_gomea_arm(argv: list) -> bool:
    """True iff `argv` selects the GOMEA inner optimizer: the adjacent pair
    `-a gomea` / `--opt-algo gomea`, or the single token `--opt-algo=gomea`.

    Fix round 1, finding 9: the previous check was `any("gomea" in str(tok)
    for tok in argv)`, a bare substring test that also matched, say, a bare
    `--gomea-fos lt-b` with no `-a`/`--opt-algo` at all, or any token that
    merely happens to CONTAIN "gomea" (a task or arm name, for instance) --
    never mind that `--gomea-fos`/`--gomea-log`/etc. only make sense
    alongside `-a gomea` in the first place. This scans for the actual
    flag that selects the optimizer.
    """
    for i, tok in enumerate(argv):
        if tok in ("-a", "--opt-algo") and i + 1 < len(argv) and argv[i + 1] == "gomea":
            return True
        if tok == "--opt-algo=gomea":
            return True
    return False


def expand_runs(m: dict, base: str | Path | None = None) -> list[dict]:
    """Expand a validated manifest into one run dict per task x arm x seed.

    Iteration order is the manifest's own JSON order -- tasks, then arms,
    then the seeds list, in the order they appear in the manifest -- which
    is deterministic (Python `dict`s preserve insertion order, and
    `json.loads` inserts keys in the order they were parsed).

    `base` is forwarded to `run_dir_for`/`result_dir` -- the directory a
    relative `output_root` resolves against (default: the calling
    process's cwd). `runner.py` always passes its own resolved "launch
    directory" explicitly, so every run's `run_dir` is pinned to one
    unambiguous absolute path regardless of what `output_root` looks like
    in the manifest (fix round 1, finding 1).

    Each run dict:
        run_id     -- f"{task}__{arm}__s{seed}"
        task, arm, seed
        argv       -- common_argv + task argv + arm argv + budget argv +
                      ["-r", str(seed)], plus ["--gomea-log",
                      "<run_dir>/events.jsonl"] appended when the arm's own
                      argv selects `-a gomea` (`is_gomea_arm`, precise --
                      not a bare substring match). Does NOT include a
                      leading program-name argv[0] -- these are the raw
                      moses CLI flags, suitable for a human to reproduce
                      with `moses <flags>`; `runner.py` is responsible for
                      prepending a dummy program name, AND for its own
                      runner-added `-f <run_dir>/moses.log` suffix, before
                      calling the in-process moses entry point.
        target_raw -- the task's success threshold, copied through for
                      convenience.
        run_dir    -- str(path) this run's own output directory, computed
                      from `result_dir(m, base)` so it stays in sync with
                      `runner.py`'s own layout. Always absolute.
    """
    runs: list[dict] = []
    common_argv = list(m["common_argv"])
    budget_argv = list(m["budget"]["argv"])
    for task_name, task_cfg in m["tasks"].items():
        task_argv = list(task_cfg["argv"])
        target_raw = task_cfg["target_raw"]
        for arm_name, arm_cfg in m["arms"].items():
            arm_argv = list(arm_cfg["argv"])
            is_gomea = is_gomea_arm(arm_argv)
            for seed in m["seeds"]:
                run_id = f"{task_name}__{arm_name}__s{seed}"
                rd = run_dir_for(m, run_id, base=base)
                argv = common_argv + task_argv + arm_argv + budget_argv + ["-r", str(seed)]
                if is_gomea:
                    argv = argv + ["--gomea-log", str(rd / "events.jsonl")]
                runs.append(
                    {
                        "run_id": run_id,
                        "task": task_name,
                        "arm": arm_name,
                        "seed": seed,
                        "argv": argv,
                        "target_raw": target_raw,
                        "run_dir": str(rd),
                    }
                )
    return runs


def git_head(cwd: str | Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def check_code_commit(repo_root: str | Path, commit: str) -> dict:
    """Is the code at HEAD the code the manifest's `commit` names?

    The manifest's `commit` is the CODE commit; the committed manifest
    itself lives in a later commit. So the check is not "HEAD == commit"
    but: `commit` is an ancestor of HEAD (`is_ancestor`), nothing under
    `CODE_PATHS` changed between them (`code_unchanged`, `git diff --quiet
    <commit> HEAD -- ...`), and the working tree has no local changes
    under `pymoses`/`benchmarks` (`clean`, `git status --porcelain`,
    untracked files included; `dirty` lists them). `ok` is all three.
    Never raises for a bad commit: an unknown commit is simply not an
    ancestor.
    """
    root = str(repo_root)

    def _rc(*args: str) -> int:
        return subprocess.run(["git", *args], cwd=root, capture_output=True).returncode

    head = git_head(root)
    is_ancestor = _rc("merge-base", "--is-ancestor", commit, "HEAD") == 0
    code_unchanged = is_ancestor and _rc("diff", "--quiet", commit, "HEAD", "--", *CODE_PATHS) == 0
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "pymoses", "benchmarks"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    dirty = [line for line in status.splitlines() if line.strip()]
    ok = is_ancestor and code_unchanged and not dirty
    if ok:
        reason = None
    elif not is_ancestor:
        reason = f"manifest commit {commit} is not an ancestor of HEAD {head}"
    elif not code_unchanged:
        reason = f"code under {', '.join(CODE_PATHS)} changed between {commit} and HEAD {head}"
    else:
        reason = f"local code modifications under pymoses/benchmarks: {dirty}"
    return {
        "head": head,
        "manifest_commit": commit,
        "is_ancestor": is_ancestor,
        "code_unchanged": code_unchanged,
        "clean": not dirty,
        "dirty": dirty,
        "ok": ok,
        "reason": reason,
    }


def _git_status_short(cwd: str | Path) -> str:
    out = subprocess.run(
        ["git", "status", "--short"], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return out.stdout


def write_environment(
    out_dir: str | Path,
    repo_root: str | Path | None = None,
    launch_dir: str | Path | None = None,
    extra: dict | None = None,
) -> None:
    """Write `environment.json`: python version, platform, cpu, git HEAD,
    short status, and (fix round 1, finding 1) the "launch directory" the
    runner fixed its process's cwd to for this invocation -- the directory
    every run's relative task argv (e.g. `-i tests/fixtures/...`) and any
    relative `output_root` resolve against. `repo_root` is independent of
    `launch_dir` and always used for the git commands (the repository's own
    identity is a fixed fact, unrelated to where a test points relative
    task paths). `extra` (the runner's effective `output_root`/`workers`,
    memory limit and code-commit check) is merged in. Deliberately carries
    no process id or other
    per-invocation noise -- two runs of the same code at the same commit
    and the same launch directory produce the same file."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "git_head": git_head(root),
        "git_status_short": _git_status_short(root),
        "launch_dir": str(launch_dir) if launch_dir is not None else None,
    }
    # `runner.py`'s effective launch settings and the code-commit check.
    info.update(extra or {})
    out_path = Path(out_dir) / "environment.json"
    out_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
