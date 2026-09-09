#!/usr/bin/env python3
"""Capture classic MOSES's own `-l DEBUG` log as the port's logging oracle.

Runs the classic binary once per configuration, reduces each log to
normalised `LEVEL<TAB>MESSAGE` records with `tools/normalize_log.py`, and
writes them to `tests/fixtures/moses_log.json`. `tests/moses/main/
test_log_differential.py` re-runs the PORT with the same argv and compares.

Why a fixture rather than running classic from the test: `tools/ci.sh` must
not require the classic checkout or a C++ toolchain (see
`tools/_capture_env.py`'s own docstring), and a differential that only
runs where classic is built is a differential nobody runs.

Usage::

    PYMETTA_CLASSIC_DIR=/path/to/Moses python3 tools/capture_moses_log.py

The classic binary is `$PYMETTA_CLASSIC_DIR/build/moses/moses/main/moses`;
it must already be built (`scripts/rebuild-moses-local.sh`). Every run is
bounded by `ulimit -v` and a timeout, per CLAUDE.md.

The captured records are NOT raw log lines. They have been through
`normalize_log`, so the timestamp is gone and the hostname, paths, log file
name, argv[0], the two `DemesHill:` microsecond columns, its RAM column and
`Stats:`' elapsed-seconds column are placeholders. That is deliberate: the
fixture stores exactly what the comparison can legitimately assert, and
nothing that would rot the moment the clock ticks.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import _capture_env  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402
from tools.normalize_log import normalize_log  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "moses_log.json"

# Address-space cap and wall-clock bound for every classic run, per CLAUDE.md
# ("Bound every C++ run: `( ulimit -v 2097152; timeout 300 <cmd> )`").
_ULIMIT_V_BYTES = 4 * 1024 * 1024 * 1024
_TIMEOUT_SECS = 300

# The configurations to capture. The `it` problem on classic's own
# tests/moses/dataset.csv is the primary one: it is small, it terminates on
# "found best score" rather than on a budget, and it exercises the whole
# pipeline -- table loading, scoring, knob building, hill climbing, merging
# and the final report -- in 173 records.
#
# The input file is the PORT's own byte-identical copy of that file (copied
# 2026-09-09 as tests/fixtures/table_data/moses_dataset.csv), so the
# differential test needs no classic checkout at run time. `normalize_log`
# reduces the path to its basename, so both sides record the same
# `Read data file moses_dataset.csv`.
#
# The stored argv keeps the `{DATASET}` placeholder rather than the resolved
# absolute path: a path baked into the fixture would be machine-specific, and
# the test resolves the same placeholder against its own repo root.
DATASET_PLACEHOLDER = "{DATASET}"


def dataset_path(repo_root: Path) -> str:
    """Absolute path of the copied dataset, given a repo root."""
    return str(repo_root / "tests" / "fixtures" / "table_data" / "moses_dataset.csv")


def resolve_argv(argv: list[str], repo_root: Path) -> list[str]:
    """Substitute `{DATASET}` in a stored argv."""
    return [dataset_path(repo_root) if a == DATASET_PLACEHOLDER else a for a in argv]


# `prerec` is the second configuration, and it earns its place: it exercises
# `discriminating_bscore` and the `-q`/`-w` threshold path, and comparing it
# against classic is what exposed a real numeric divergence -- classic logged
# `Score: -nan` where the port logged `Score: -3.40282e+38`, because
# `prerec_bscore`'s `bs / pos_total` used Python's `/` (which raises on 0/0)
# instead of IEEE division. The `-q0.5 -w1.0` thresholds are required: the
# defaults leave `min_threshold` at 0.0, and classic ABORTS on that
# (discriminating_bscore.cc:224).
CONFIGS = {
    "it_dataset_m300": ["-i", DATASET_PLACEHOLDER, "-m300", "-W1"],
    "prerec_dataset_m300": [
        "-Hprerec",
        "-i",
        DATASET_PLACEHOLDER,
        "-m300",
        "-W1",
        "-q0.5",
        "-w1.0",
    ],
}


def _capture_one(binary: Path, cwd: Path, argv: list[str]) -> list[str]:
    """Run classic once and return its log as normalised records."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "classic.log"
        cmd = [str(binary), *argv, "-l", "DEBUG", "-f", str(log)]

        def _limit() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (_ULIMIT_V_BYTES, _ULIMIT_V_BYTES))

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECS,
            preexec_fn=_limit,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"classic exited {proc.returncode}: {proc.stderr[:2000]}")
        return normalize_log(log.read_text(errors="replace"))


def main() -> int:
    classic = _capture_env.classic_dir()
    binary = classic / "build" / "moses" / "moses" / "main" / "moses"
    if not binary.is_file():
        raise SystemExit(f"classic binary not built: {binary}")

    golden = {}
    for name, argv in CONFIGS.items():
        resolved = resolve_argv(argv, REPO_ROOT)
        golden[name] = {"argv": argv, "records": _capture_one(binary, classic, resolved)}
        print(f"{name}: {len(golden[name]['records'])} records")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_moses_log.py",
        capture_script_path=Path(__file__).resolve(),
        probe_kind="classic_cli",
        cli_tool="build/moses/moses/main/moses",
        classic_dir=classic,
        inputs={
            "description": (
                "classic's own -l DEBUG log for each configuration, reduced to "
                "normalised LEVEL<TAB>MESSAGE records by tools/normalize_log.py "
                "(timestamp dropped; hostname, paths, log file name, argv[0], the "
                "two DemesHill microsecond columns, its RAM column and Stats' "
                "elapsed-seconds column replaced by placeholders)"
            ),
            "configs": {name: argv for name, argv in CONFIGS.items()},
            "cwd": "$PYMETTA_CLASSIC_DIR",
        },
    )
    write_sidecar(FIXTURE, manifest)
    print(f"wrote {FIXTURE} and its sidecar manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
