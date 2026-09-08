"""Build and capture the S10 Task 4 deme_expander oracle fixture.

Mirrors tools/capture_hillclimbing.py's shape (S9 fix-round hardening):
compile tools/cpp_probe/deme_expander_probe.cc against the installed classic
headers and libraries (moses/comboreduct/cogutil), run it once per named
case below, and save the merged JSON as tests/fixtures/deme_expander.json.
Every subprocess is resource-bounded (RLIMIT_AS + a wall-clock timeout) per
this session's standing OOM caution -- both datasets involved
(trainingBooleanXor.dat, trainingBoolean.dat) and every exemplar are small
by construction (arity 2-3), so deme sizes stay in the tens.

`CLASSIC`/`PREFIX` are resolved via `PYMETTA_CLASSIC_DIR`/
`PYMETTA_MOSES_PREFIX` (tools/_capture_env.py) -- REQUIRED, with a clear
error if unset -- and the written fixture carries a top-level `_manifest`
(schema from tools/_manifest.py: probe source path + its sha256, this
script's own path + sha256, the commit this repo was at, a UTC timestamp,
inputs) -- `tests/moses/deme/test_deme_expander.py`'s own
`test_fixture_manifest_matches_current_probe_source` cross-checks the
recorded sha256 against the CURRENT probe source, so an edit to the probe
that never re-runs this script is caught without needing the classic
checkout, a compiler, or the installed prefix present in the environment
running the test suite -- the same actionable half of "ci.sh cannot rebuild
the probes" (docs/CARRY-FORWARD.md Sec 5) capture_hillclimbing.py already
addresses for its own fixture; every OTHER fixture now carries the same
manifest as a `<name>.manifest.json` sidecar, checked by
`tools/check_fixture_manifests.py`.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "deme_expander_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "deme_expander_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "deme_expander.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300

# (name, dataset file under tests/fixtures/table_data, target feature,
# exemplar combo-tree source, arity, randGen() seed, max_evals)
CASES = [
    ("xor_empty_and", "trainingBooleanXor.dat", "y", "and()", 2, 1, 200),
    ("boolean_or_and_seed", "trainingBoolean.dat", "y", "or(and($1 $2) $3)", 3, 42, 300),
    ("boolean_empty_and", "trainingBoolean.dat", "y", "and()", 3, 7, 500),
]


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
            f"deme_expander probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"deme_expander probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_deme_expander.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/deme_expander_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "cases": {
                name: {
                    "dataset": dataset,
                    "target_feature": target,
                    "exemplar": exemplar,
                    "arity": arity,
                    "seed": seed,
                    "max_evals": max_evals,
                }
                for name, dataset, target, exemplar, arity, seed, max_evals in CASES
            }
        },
    )


def run_case(
    dataset: str, target: str, exemplar: str, arity: int, seed: int, max_evals: int
) -> dict:
    csv_path = REPO_ROOT / "tests" / "fixtures" / "table_data" / dataset
    argv = [
        str(PROBE_BIN),
        str(csv_path),
        target,
        exemplar,
        str(arity),
        str(seed),
        str(max_evals),
    ]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"deme_expander probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    payload["dataset"] = dataset
    payload["target_feature"] = target
    payload["exemplar"] = exemplar
    payload["arity"] = arity
    payload["seed"] = seed
    payload["max_evals"] = max_evals
    payload["complexity_ratio"] = 3.5
    return payload


def capture_fixture() -> None:
    cases = {}
    for name, dataset, target, exemplar, arity, seed, max_evals in CASES:
        cases[name] = run_case(dataset, target, exemplar, arity, seed, max_evals)
    payload = {"cases": cases, "_manifest": _build_manifest()}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
