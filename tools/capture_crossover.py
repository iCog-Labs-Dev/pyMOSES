"""Build and capture the S12 crossover oracle fixture.

Compiles tools/cpp_probe/crossover_probe.cc against the installed classic
headers and libraries, runs it FROM THE REPO ROOT (its "local_moses" group
loads the relative path tests/fixtures/table_data/trainingBoolean.dat), and
saves its JSON stdout as tests/fixtures/crossover.json, with a
`<name>.manifest.json` sidecar (tools/_manifest.py) recording exactly what
produced it.

Mirrors tools/capture_hillclimbing.py's shape, with two differences:

1. The link line adds `-lboost_program_options`, which the "local_moses"
   group needs -- see tools/cpp_probe/local_moses_probe.cc's own header for
   the measured link failure without it.
2. The probe takes a log directory as argv[1]. It turns classic's own
   `logger()` up to DEBUG for the hill_climbing/local_moses runs so it can
   read back which iterations really took the `if (xover)` branch
   (hill-climbing.cc:152) and why, then deletes each log file. A temporary
   directory is used so nothing is left behind and nothing in the repo is
   written to.

Every subprocess is resource-bounded (RLIMIT_AS + a wall-clock timeout) per
this session's standing OOM caution; the probe's cases are sized so the
largest deme stays in the low thousands of instances.
"""

import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, sidecar_path, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "crossover_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "crossover_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "crossover.json"
_MEM_LIMIT_BYTES = 2097152 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 600


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
        "-lboost_program_options",
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
            f"crossover probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"crossover probe compilation failed:\n{result.stderr}")


def _build_manifest() -> dict:
    return build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/capture_crossover.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/cpp_probe/crossover_probe.cc",
        probe_source_path=PROBE_SRC,
        probe_kind="cpp_probe",
        classic_dir=CLASSIC,
        prefix_dir=PREFIX,
        inputs={
            "description": (
                "fixed cases hardcoded in crossover_probe.cc: 15 std::partial_sort "
                "tie-order samples, 24 cross_top_*/crossover deme scenarios run "
                "through all four functions, 7 full hill_climbing runs (both gate "
                "entry paths, plus the 400-neighbour boundary either side), and the "
                "S11 five-generation local_moses scenario with crossover on and off"
            ),
            "hill_climbing_cases": {
                "large_nbh_at_distance_2": {
                    "arity": 6,
                    "seed": 1,
                    "max_evals": 3000,
                    "hc_cross": True,
                },
                "large_nbh_at_distance_2_immediate": {
                    "arity": 6,
                    "seed": 1,
                    "max_evals": 3000,
                    "hc_cross": True,
                    "null_target": True,
                },
                "last_chance_only": {
                    "arity": 4,
                    "seed": 1,
                    "max_evals": 3000,
                    "hc_cross": True,
                    "null_target": True,
                },
                "no_crossover": {
                    "arity": 6,
                    "seed": 1,
                    "max_evals": 3000,
                    "hc_cross": False,
                    "null_target": True,
                },
                "large_nbh_below": {
                    "arity": 31,
                    "seed": 3,
                    "max_evals": 400,
                    "hc_cross": True,
                    "hc_max_nn_evals": 40,
                },
                "large_nbh_above": {
                    "arity": 32,
                    "seed": 3,
                    "max_evals": 400,
                    "hc_cross": True,
                    "hc_max_nn_evals": 40,
                },
                "large_nbh_widen": {
                    "arity": 32,
                    "seed": 5,
                    "max_evals": 700,
                    "hc_cross": True,
                    "hc_widen": True,
                    "hc_max_nn_evals": 40,
                    "null_target": True,
                },
            },
            "local_moses_cases": {
                "table": "tests/fixtures/table_data/trainingBoolean.dat",
                "seed": 42,
                "exemplar": "$1",
                "complexity_coef": 3.5,
                "max_gens": 5,
                "hc_cross": [True, False],
            },
        },
    )


def capture_fixture() -> None:
    with tempfile.TemporaryDirectory(prefix="pymoses-crossover-") as logdir:
        try:
            result = subprocess.run(
                [str(PROBE_BIN), logdir],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_S,
                preexec_fn=_bound_resources,
                check=True,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"crossover probe run exceeded {_RUN_TIMEOUT_S}s") from error
    payload = json.loads(result.stdout)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Compact, not pretty-printed: the hill_climbing trajectories carry
    # thousands of instances of up to 128 raw fields each, and `indent=`
    # would balloon the file for no reader's benefit (json.loads consumes
    # this, nobody eyeballs it). Same choice capture_hillclimbing.py makes.
    FIXTURE_PATH.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    write_sidecar(FIXTURE_PATH, _build_manifest())


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH} and {sidecar_path(FIXTURE_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
