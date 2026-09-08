"""Build and capture the S9 Task 2 hill_climbing oracle fixture.

Mirrors tools/capture_neighborhood.py's shape: compile
tools/cpp_probe/hillclimbing_probe.cc against the installed classic headers
and libraries (moses/comboreduct/cogutil), run it, and save its JSON stdout
as tests/fixtures/hillclimbing.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this session's standing OOM caution.

S9 fix-round hardening (2026-09): two standing weaknesses named in
docs/CARRY-FORWARD.md Sec 5 ("No fixture carries a checksum, version, or
input manifest", "ci.sh cannot rebuild the probes") are addressed for THIS
fixture specifically (the other capture_*.py scripts in this directory
share the same hardcoded-path pattern but are out of this task's scope):

1. `CLASSIC`/`PREFIX` are now overridable via `PYMETTA_CLASSIC_DIR`/
   `PYMETTA_MOSES_PREFIX` environment variables, defaulting to the exact
   absolute paths this script always used -- so behaviour on THIS machine
   is unchanged, but the script is no longer permanently tied to one
   developer's home directory the way it (and every sibling capture
   script, and tests/conftest.py) was before.
2. The written fixture now carries a top-level `_manifest` object (probe
   source path + its sha256, this capture script's own path, the commit
   this repo was at, and a UTC timestamp) recording what produced it --
   `tests/moses/moses/optimization/test_hill_climbing.py`'s own
   `test_fixture_manifest_matches_current_probe_source` cross-checks the
   recorded sha256 against the CURRENT probe source on every test run, so
   an edit to the probe that never re-runs this script is caught without
   needing the classic checkout, a compiler, or the installed prefix to be
   present in the environment running the test suite.

This does not, by itself, make `ci.sh` able to rebuild the fixture (ci.sh
deliberately stays compiler/classic-checkout-free, per its own docstring
constraints and CLAUDE.md's stdlib-only house rule) -- it makes drift
between the checked-in fixture and the checked-in probe source DETECTABLE
by the test suite that already runs there, which is the actionable half of
"ci.sh cannot rebuild the probes".
"""

import hashlib
import json
import os
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIC = Path(
    os.environ.get(
        "PYMETTA_CLASSIC_DIR",
        "/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses",
    )
)
PREFIX = Path(os.environ.get("PYMETTA_MOSES_PREFIX", str(Path.home() / "opt" / "moses-legacy")))
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "hillclimbing_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "hillclimbing_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "hillclimbing.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    lib = PREFIX / "lib"
    lib_opencog = lib / "opencog"
    command = [
        "g++",
        "-O2",
        "-std=c++17",
        "-I",
        str(CLASSIC),
        "-I",
        str(PREFIX / "include"),
        "-o",
        str(PROBE_BIN),
        str(PROBE_SRC),
        "-L",
        str(lib),
        "-L",
        str(lib_opencog),
        "-Wl,-rpath," + str(lib),
        "-Wl,-rpath," + str(lib_opencog),
        "-lmoses",
        "-lcomboreduct",
        "-lcogutil",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMPILE_TIMEOUT_S,
            preexec_fn=_bound_resources,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"hillclimbing probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"hillclimbing probe compilation failed:\n{result.stderr}")


def _git_commit() -> str | None:
    """Best-effort -- returns None (rather than raising) when `git` is
    unavailable or this checkout is not a git repo, so a manifest can
    still be written in either case.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def _build_manifest() -> dict:
    return {
        "schema_version": 1,
        "capture_script": "tools/capture_hillclimbing.py",
        "probe_source": "tools/cpp_probe/hillclimbing_probe.cc",
        "probe_source_sha256": hashlib.sha256(PROBE_SRC.read_bytes()).hexdigest(),
        "repo_commit": _git_commit(),
        "classic_dir": str(CLASSIC),
        "prefix_dir": str(PREFIX),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def capture_fixture() -> None:
    try:
        result = subprocess.run(
            [str(PROBE_BIN)],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"hillclimbing probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    payload["_manifest"] = _build_manifest()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Compact, not pretty-printed: the "deterministic"/"sampled" trajectory
    # arrays carry thousands of instances each, and `indent=` would balloon
    # the file ~3x for no readability benefit no reader actually uses (this
    # file is consumed by json.loads, not eyeballed).
    FIXTURE_PATH.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
