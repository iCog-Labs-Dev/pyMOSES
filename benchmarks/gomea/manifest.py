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


def _validate(m: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(m, dict):
        return [f"manifest root must be a JSON object, got {type(m).__name__}"]

    for key in _REQUIRED_TOP_KEYS:
        _check_present(m, key, "manifest", errors)

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

    if m.get("seeds") is not None:
        if not isinstance(m["seeds"], list) or not m["seeds"]:
            errors.append("manifest.seeds must be a non-empty array")
        else:
            for i, s in enumerate(m["seeds"]):
                if s is None:
                    errors.append(f"manifest.seeds[{i}] is null")

    if m.get("budget") is not None:
        if not isinstance(m["budget"], dict):
            errors.append("manifest.budget must be an object")
        else:
            for key in _REQUIRED_BUDGET_KEYS:
                _check_present(m["budget"], key, "manifest.budget", errors)

    if m.get("common_argv") is not None and not isinstance(m["common_argv"], list):
        errors.append("manifest.common_argv must be an array")

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
    truncated to 12 hex characters. Independent of key insertion order."""
    canonical = json.dumps(m, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def result_dir(m: dict) -> Path:
    """`<output_root>/<phase>/<name>-<hash>` -- the top of one manifest's
    output tree (03_EXPERIMENT_PROTOCOL.md Sec 9's suggested layout)."""
    return Path(m["output_root"]) / str(m["phase"]) / f"{m['name']}-{manifest_hash(m)}"


def run_dir_for(m: dict, run_id: str) -> Path:
    return result_dir(m) / "runs" / run_id


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


def expand_runs(m: dict) -> list[dict]:
    """Expand a validated manifest into one run dict per task x arm x seed.

    Iteration order is the manifest's own JSON order -- tasks, then arms,
    then the seeds list, in the order they appear in the manifest -- which
    is deterministic (Python `dict`s preserve insertion order, and
    `json.loads` inserts keys in the order they were parsed).

    Each run dict:
        run_id     -- f"{task}__{arm}__s{seed}"
        task, arm, seed
        argv       -- common_argv + task argv + arm argv + budget argv +
                      ["-r", str(seed)], plus ["--gomea-log",
                      "<run_dir>/events.jsonl"] appended when the arm's own
                      argv contains "gomea" anywhere (a `-a gomea` arm, or
                      any future arm that mentions it). Does NOT include a
                      leading program-name argv[0] -- these are the raw
                      moses CLI flags, suitable for a human to reproduce
                      with `moses <flags>`; `runner.py` is responsible for
                      prepending a dummy program name before calling the
                      in-process moses entry point.
        target_raw -- the task's success threshold, copied through for
                      convenience.
        run_dir    -- str(path) this run's own output directory, computed
                      from `result_dir(m)` so it stays in sync with
                      `runner.py`'s own layout.
    """
    runs: list[dict] = []
    common_argv = list(m["common_argv"])
    budget_argv = list(m["budget"]["argv"])
    for task_name, task_cfg in m["tasks"].items():
        task_argv = list(task_cfg["argv"])
        target_raw = task_cfg["target_raw"]
        for arm_name, arm_cfg in m["arms"].items():
            arm_argv = list(arm_cfg["argv"])
            is_gomea = any("gomea" in str(tok) for tok in arm_argv)
            for seed in m["seeds"]:
                run_id = f"{task_name}__{arm_name}__s{seed}"
                rd = run_dir_for(m, run_id)
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


def _git_status_short(cwd: str | Path) -> str:
    out = subprocess.run(
        ["git", "status", "--short"], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return out.stdout


def write_environment(out_dir: str | Path, repo_root: str | Path | None = None) -> None:
    """Write `environment.json`: python version, platform, cpu, git HEAD
    and short status. Deliberately carries no process id or other
    per-invocation noise -- two runs of the same code at the same commit
    produce the same file."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "git_head": git_head(root),
        "git_status_short": _git_status_short(root),
    }
    out_path = Path(out_dir) / "environment.json"
    out_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
