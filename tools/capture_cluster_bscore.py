"""Build and capture the `cluster_bscore` oracle fixture.

Mirrors tools/capture_contin_bscore.py's shape: compile
tools/cpp_probe/cluster_bscore_probe.cc against the installed classic
headers and libraries, run it once per mode, and save the merged JSON as
tests/fixtures/cluster_bscore.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this project's standing OOM
caution.

Why a probe when `-H cluster` is drivable through classic's own `moses`
binary end to end (and IS, in tests/moses/main/test_moses_exec.py): the
CLI only ever shows the FINAL composite scores of the candidates that
survive. These cases pin `cluster_bscore::operator()`'s single bscore
element directly, on inputs the search loop will not produce on its own:

  * float32 OVERFLOW at `vals[j] = get_contin(oned[j])` (bscores.cc:967),
    where a perfectly finite `contin_t` (1e60) becomes `+inf` purely
    because `score_t` is binary32 -- and the `std::isinf` guard
    (bscores.cc:982) then fires. A port that keeps double precision here
    returns a finite score instead;
  * float32 ACCUMULATION in `sum[i] += sc` / `squ += sc*sc`
    (bscores.cc:992, :1033), pinned both on a 60-row synthetic column
    just above 2^23 and on trainingRegression.dat's 8000 rows;
  * the `cnt[i] < 3.5` guard (bscores.cc:1008);
  * `final == 0`, i.e. `1.0 / 0.0` -> `+inf` (bscores.cc:1055), which
    Python raises `ZeroDivisionError` on rather than producing;
  * `OC_ASSERT(nclusters < oned.size())` (bscores.cc:944) on a table with
    three rows or fewer;
  * a nan column, which is `std::sort`'s undefined behaviour in C++ --
    recorded to show what this toolchain does, and run in its own process
    so a crash could not take the other modes' output with it.

Modes are separate probe invocations merged into one JSON document here;
the probe writes one JSON object per invocation. See the probe's own
header comment for what each mode contains.
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

CLASSIC = classic_dir()
PREFIX = moses_prefix()
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "cluster_bscore_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "cluster_bscore_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "cluster_bscore.json"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300

# `-u y` for both: the target column is EXCLUDED from the ITable
# `cluster_bscore` reads, so which column is the target still matters even
# though the scorer never looks at the output values.
_CONTIN_SAMPLE_REL = "tests/fixtures/table_data/contin_regression_sample.csv"
_TRAINING_REGRESSION_REL = "trainingRegression.dat"

# Trees are POSITIONAL ($1, $2) -- the probe passes an empty label vector
# to `str2combo_tree`, so the same strings work against either table.
# `+(*(-1 log($2)) $1)` and `log($2)` are not arbitrary: they are two of
# the candidates classic's own `-Hcluster -m200` run actually prints on
# contin_regression_sample.csv, so the fixture pins scores the search loop
# really visits, alongside the degenerate ones it discards.
_SAMPLE_TREES = ["$1", "+($1 $2)", "*($1 $2)", "+(*(-1 log($2)) $1)", "log($2)", "0"]
_TRAINING_TREES = ["$1", "+($1 $2)", "*($1 $2)", "log($2)"]


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
            f"cluster_bscore probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"cluster_bscore probe compilation failed:\n{result.stderr}")


def _run_probe(args: list[str]) -> dict:
    try:
        result = subprocess.run(
            [str(PROBE_BIN), *args],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            preexec_fn=_bound_resources,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"cluster_bscore probe run {args!r} exceeded {_RUN_TIMEOUT_S}s"
        ) from error
    return json.loads(result.stdout)


def capture_fixture() -> None:
    sample_csv = REPO_ROOT / _CONTIN_SAMPLE_REL
    training_dat = CLASSIC / _TRAINING_REGRESSION_REL

    data = {
        "synthetic": _run_probe(["synthetic"]),
        "overflow": _run_probe(["overflow"]),
        "nan": _run_probe(["nan"]),
        "too_small": _run_probe(["too_small"]),
        "tables": {
            "contin_regression_sample": {
                "repo_relpath": _CONTIN_SAMPLE_REL,
                "target": "y",
                "result": _run_probe(["table", str(sample_csv), "y", *_SAMPLE_TREES]),
            },
            "training_regression": {
                "classic_relpath": _TRAINING_REGRESSION_REL,
                "target": "y",
                "result": _run_probe(["table", str(training_dat), "y", *_TRAINING_TREES]),
            },
        },
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_cluster_bscore.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/cluster_bscore_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "hand-built single-column ITables (hardcoded in "
                    "cluster_bscore_probe.cc, dumped alongside each score so "
                    "the test does not re-derive them) plus two real tables"
                ),
                "synthetic_modes": ["synthetic", "overflow", "nan", "too_small"],
                "tables": {
                    "contin_regression_sample": {
                        "path": _CONTIN_SAMPLE_REL,
                        "target": "y",
                        "trees": _SAMPLE_TREES,
                    },
                    "training_regression": {
                        "path": f"$CLASSIC/{_TRAINING_REGRESSION_REL}",
                        "target": "y",
                        "trees": _TRAINING_TREES,
                    },
                },
            },
        ),
    )


def main(argv: list[str]) -> int:
    del argv
    build_probe()
    capture_fixture()
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
