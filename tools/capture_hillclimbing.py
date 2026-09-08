"""Build and capture the S9 Task 2 hill_climbing oracle fixture.

Mirrors tools/capture_neighborhood.py's shape: compile
tools/cpp_probe/hillclimbing_probe.cc against the installed classic headers
and libraries (moses/comboreduct/cogutil), run it, and save its JSON stdout
as tests/fixtures/hillclimbing.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this session's standing OOM caution.

S9 fix-round hardening (2026-09), later generalized to every capture
script (see tools/_capture_env.py, tools/_manifest.py,
tools/check_fixture_manifests.py) for docs/CARRY-FORWARD.md Sec 5 ("No
fixture carries a checksum, version, or input manifest", "ci.sh cannot
rebuild the probes"):

1. `CLASSIC`/`PREFIX` are resolved via `PYMETTA_CLASSIC_DIR`/
   `PYMETTA_MOSES_PREFIX` (tools/_capture_env.py) -- REQUIRED, with a
   clear error if unset, not a personal-machine fallback -- so this
   script (and now every capture_*.py sibling) is not permanently tied to
   one developer's home directory.
2. The written fixture carries a top-level `_manifest` object (schema
   from tools/_manifest.py: probe source path + its sha256, this capture
   script's own path + sha256, the commit this repo was at, a UTC
   timestamp, and the run's inputs) recording what produced it --
   `tests/moses/moses/optimization/test_hill_climbing.py`'s own
   `test_fixture_manifest_matches_current_probe_source` cross-checks the
   recorded sha256 against the CURRENT probe source on every test run, so
   an edit to the probe that never re-runs this script is caught without
   needing the classic checkout, a compiler, or the installed prefix to be
   present in the environment running the test suite. Every other fixture
   now carries the same manifest as a `<name>.manifest.json` sidecar
   (see tools/_manifest.py's docstring for why this one embeds instead),
   checked by `tools/check_fixture_manifests.py`.

This does not, by itself, make `ci.sh` able to rebuild the fixture (ci.sh
deliberately stays compiler/classic-checkout-free, per its own docstring
constraints and CLAUDE.md's stdlib-only house rule) -- it makes drift
between the checked-in fixture and the checked-in probe source DETECTABLE
by the test suite (this fixture) and `tools/check_fixture_manifests.py`
(every fixture) without either needing a C++ toolchain.
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
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


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_hillclimbing.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/hillclimbing_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "cases": {
                "deterministic": {"max_evals": 500, "hc_cross": False},
                "sampled": {"max_evals": 4000, "hc_widen": True, "hc_cross": False},
            }
        },
    )


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
