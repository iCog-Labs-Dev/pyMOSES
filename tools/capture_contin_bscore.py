"""Build and capture the S14 Task 3 `contin_bscore` oracle fixture.

Mirrors tools/capture_scorers.py's shape (one probe binary, several
subcommand "modes", one JSON fixture assembled from repeated probe runs)
but drives a SEPARATE probe (tools/cpp_probe/contin_bscore_probe.cc)
rather than editing the shared scorers_probe.cc/capture_scorers.py, which
back the differently-owned discriminating_bscore/select_bscore/
precision_bscore/time_dispersion family -- see
docs/CARRY-FORWARD.md Sec 5.1's shared-index concurrency warning, and
CLAUDE.md's "five shared-index incidents" note for this session.

Four sections, four top-level fixture keys:

* "synthetic" -- a small (5-row) hand-built OTable/ITable run through both
  `squared_error` and `abs_error`, chosen so several rows are genuinely
  irrational-in-binary (1/3, 2/3, 1.23456789) and one row's residual is
  EXACTLY zero (`res == tar`, producing `-0.0` through `sq`/`abs`) -- the
  latter caught a real JSON-round-trip bug in this probe during
  development (see contin_bscore_probe.cc's own `float_json` comment):
  a bare "-0" with no decimal point parses back as Python's int `0` when
  the fixture is loaded, silently discarding the sign of zero and making
  a bit-exactness check pass or fail on JSON FORMATTING rather than the
  actual computed value. Fixed in the probe itself, not worked around
  here.

* "table_default" -- `contin_bscore bsc(table);` constructed with NO
  err_function_type argument at all over
  tests/fixtures/table_data/contin_regression_sample.csv (the first 20
  data rows of classic's own trainingRegression.dat, vendored here) --
  i.e. classic's own real default (squared_error), not a value this
  port or probe chose. This is the task's own "at least one test runs
  classic's own defaults" oracle. Tree is `+($1 $2)`, not `*($1 $2)`:
  trainingRegression.dat's target is exactly `x1*x2` (measured: row 1
  gives `1.567521326983612948e-02 * 5.028885506232726871e-02 ==
  7.882885281978582260e-04`, the recorded `y`), so a `*` tree scores an
  all-zero bscore -- useless for catching a narrowing bug, which only
  shows up on genuinely nonzero residuals.

* "table" -- same csv/table, EXPLICIT err_function_type (both squared and
  abs), tree `+($1 $2)`.

* "complexity"/"complexity_ratio" -- `set_complexity_coef`'s two
  overloads (`(alphabet_size, stdev)` -- contin_bscore's OWN override,
  formula `log(alphabet_size) * 2 * stdev**2`, gated on `stdev > 0` -- and
  the one-argument `(complexity_ratio)` form inherited unchanged from
  `bscore_base` via C++'s `using bscore_base::set_complexity_coef`),
  covering the `stdev <= 0`/`ratio <= 0` singularity gates too.

Every probe run is resource-bounded (`ulimit -v` + `timeout`) per this
session's standing OOM caution.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "contin_bscore_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "contin_bscore_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "contin_bscore.json"
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"
_SAMPLE_CSV = DATA_DIR / "contin_regression_sample.csv"
_TRAINING_REGRESSION = CLASSIC / "trainingRegression.dat"

# ulimit -v 2097152 (KiB) / timeout 30 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 2097152 * 1024
_TIMEOUT_S = 30
_COMPILE_TIMEOUT_S = 300
_N_SAMPLE_ROWS = 20


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
            f"contin_bscore probe compilation exceeded {_COMPILE_TIMEOUT_S}s"
        ) from error
    if result.returncode != 0:
        raise RuntimeError(f"contin_bscore probe compilation failed:\n{result.stderr}")


def _write_sample_csv() -> None:
    """Vendors the first `_N_SAMPLE_ROWS` data rows of classic's own
    trainingRegression.dat (8000 rows total, too slow to load repeatedly
    across several probe invocations) -- same rationale as
    capture_scorers.py's own select_regression.csv. Deterministic: always
    the first N rows after the header, no shuffling/sampling involved.
    """
    if not _TRAINING_REGRESSION.is_file():
        raise FileNotFoundError(f"{_TRAINING_REGRESSION} not found -- is $CLASSIC correct?")
    lines = _TRAINING_REGRESSION.read_text().splitlines()
    header, rows = lines[0], lines[1 : 1 + _N_SAMPLE_ROWS]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content = "\n".join([header, *rows]) + "\n"
    if _SAMPLE_CSV.is_file() and _SAMPLE_CSV.read_text() != content:
        raise RuntimeError(
            f"{_SAMPLE_CSV} already exists with DIFFERENT content -- refusing to "
            "silently overwrite a possibly-hand-edited fixture input"
        )
    _SAMPLE_CSV.write_text(content)


def run_probe(args: list[str]) -> dict:
    result = subprocess.run(
        [str(PROBE_BIN), *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    return json.loads(result.stdout)


def main(argv: list[str]) -> int:
    del argv
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()
    _write_sample_csv()

    fixture = {
        "synthetic": run_probe(["synthetic"]),
        "table_default": run_probe(["table_default", str(_SAMPLE_CSV), "y"]),
        "table_squared": run_probe(["table", str(_SAMPLE_CSV), "y", "squared"]),
        "table_abs": run_probe(["table", str(_SAMPLE_CSV), "y", "abs"]),
        "complexity": run_probe(["complexity", "3", "0.1", "0.0", "-0.5", "2.5", "0.001"]),
        "complexity_ratio": run_probe(["complexity_ratio", "3.5", "0.0", "-1.0", "1.0"]),
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_contin_bscore.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/contin_bscore_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "contin_regression_sample_csv": (
                    "tests/fixtures/table_data/contin_regression_sample.csv"
                ),
                "sample_rows": _N_SAMPLE_ROWS,
                "sample_source": (
                    "first N data rows of $CLASSIC/trainingRegression.dat, no shuffling"
                ),
            },
        ),
    )
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
