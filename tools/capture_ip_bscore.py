"""Build and capture the `-H ip` (interesting-predicates) oracle fixture.

Mirrors tools/capture_cluster_bscore.py's shape: compile
tools/cpp_probe/ip_bscore_probe.cc against the installed classic headers
and libraries, run it once per mode, and save the merged JSON as
tests/fixtures/ip_bscore.json. Every subprocess is resource-bounded
(RLIMIT_AS + a wall-clock timeout) per this project's standing OOM
caution.

The fixture covers the three layers `interesting_predicate_bscore`
(bscores.h:553, bscores.cc:726-908) is built from, because each one is
independently gettable wrong:

  * ``stats`` -- cogutil's `standardizedMannWhitneyU`
    (MannWhitneyU.h:64), `MannWhitneyU` (:44), `ranking` (ranking.h:40)
    and boost's `weighted_skewness` accumulator, on hand-built
    `Counter<contin_t, contin_t>`s. Neither statistic existed anywhere in
    this port before `-H ip`, so they are pinned in isolation, including
    the `boost::accumulate(..., 0)` INT truncation that turns a
    fractional-count `n2` into 0 and the standardized U into `+inf`.
  * ``activation`` -- `get_activation_penalty`'s two statements
    (bscores.cc:896-904) over a 23-point grid. It is `private`, so a
    derived class cannot call it and the probe copies the body verbatim;
    the `table` cases below cross-check the copy, since the LAST bscore
    element is exactly this value. The grid is what pins classic's
    `nan` at the DEFAULT activation bounds (`--min-rand-input 0`,
    `--max-rand-input 1`, where both quotients are 0/0), the `fmax`
    NaN-skipping that keeps a half-degenerate configuration finite, and
    glibc `pow`'s sign_bias rule for a NEGATIVE NaN base.
  * ``tables`` -- the whole `operator()` on real CSVs, element by
    element, alongside the constructor's `_counter`/`_pdf`/`_skewness`
    and the CTable's rows IN `std::map` ORDER. The last of those is not
    decoration: `operator()` builds `pred_counter` with an ASSIGNMENT
    (bscores.cc:800), so the row order decides which duplicate output
    value survives, and this port's `CTable` is an insertion-ordered
    dict. `ip_boolean_contin_reversed.csv` is the same 14 observations
    in reverse file order and must produce an identical fixture entry.

Why a probe at all, when `-H ip` runs end to end through classic's own
`moses` binary (measured: exit 0 on any table whose output column is
contin and whose every input column is boolean -- table-problems.cc:236
aborts otherwise): stdout shows only the final composite scores of the
surviving candidates, which fold the whole bscore into one float. At
classic's DEFAULT bounds every one of those is `nan`, so the CLI alone
would pin almost nothing about the four weighted components. The CLI
oracles live in tests/moses/main/test_ip_problem_cli.py; this fixture is
what makes them more than a `nan` check.
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
PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "ip_bscore_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "ip_bscore_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ip_bscore.json"
TABLE_DATA = REPO_ROOT / "tests" / "fixtures" / "table_data"
_MEM_LIMIT_BYTES = 1048576 * 1024
_COMPILE_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 300

# The trees every `table` case is evaluated on. POSITIONAL ($1, $2, ...),
# never column names -- `eval-table`-style label resolution is broken in
# classic (docs/CARRY-FORWARD.md Sec 3) and the probe passes an empty
# label vector for that reason.
#
#   $1                  a single-argument predicate, ~half the rows
#   and($1 $2)          a narrow selection
#   not($1)             the complement of the first
#   and($1 not($1))     selects NOTHING -- pred_counter is EMPTY, which
#                       the `pred_counter.size() == 1` guard does NOT
#                       catch; every component divides by zero
#   or($1 or($2 $3))    selects nearly everything
#   true                selects EVERY row, so pred_counter == _counter
#                       except where the assignment defect bites
_TREES = ("$1", "and($1 $2)", "not($1)", "and($1 not($1))", "or($1 or($2 $3))", "true")

# (label, csv basename, target, K, J, U, X, min_act, max_act, penalty, positive)
#
# Weights are given as the CLI would: `--ip_kld_weight` and friends
# (table-problems.cc:210-224), all defaulting to 1.0. `min_act`/`max_act`
# are `--min-rand-input`/`--max-rand-input`, `penalty` is `--alpha`, and
# `positive` is `--pre-positive`.
_TABLE_CASES = (
    # reaches: the DEFAULT activation bounds -- what the 16 `-Hip` rows of
    # this project's CLI sweep run, and what makes classic print `nan`.
    ("skewed_default_bounds", "ip_boolean_contin.csv", "y", 1, 1, 1, 1, 0, 1, 1, 1),
    # reaches: real bounds, so every component is finite and the scorer
    # actually discriminates. THE primary case.
    ("skewed_real_bounds", "ip_boolean_contin.csv", "y", 1, 1, 1, 1, 0.2, 0.8, 1, 1),
    # reaches: the SAME 14 observations with the rows reversed. Classic
    # walks the CTable in `std::map` order, so this must be identical to
    # `skewed_real_bounds` element for element -- the check that this
    # port's insertion-ordered dict is being sorted.
    ("skewed_reversed_rows", "ip_boolean_contin_reversed.csv", "y", 1, 1, 1, 1, 0.2, 0.8, 1, 1),
    # reaches: `--pre-positive 0`, i.e. the predicate is INVERTED, so
    # `true` selects nothing and the empty-pred_counter path is reached by
    # an ordinary tree.
    ("skewed_negated", "ip_boolean_contin.csv", "y", 1, 1, 1, 1, 0.2, 0.8, 1, 0),
    # reaches: one weight at a time. `-K0` additionally leaves classic's
    # `_skewness` UNINITIALISED (bscores.cc:749 gates the whole precompute
    # block on `_kld_w > 0`), so those three cases record what this
    # toolchain's uninitialised `double` happened to be -- see the
    # DIVERGENCE note on `interesting_predicate_bscore.__init__`.
    ("weights_kld_only", "ip_boolean_contin.csv", "y", 1, 0, 0, 0, 0.2, 0.8, 1, 1),
    ("weights_skewness_only", "ip_boolean_contin.csv", "y", 0, 1, 0, 0, 0.2, 0.8, 1, 1),
    ("weights_stdU_only", "ip_boolean_contin.csv", "y", 0, 0, 1, 0, 0.2, 0.8, 1, 1),
    ("weights_skew_U_only", "ip_boolean_contin.csv", "y", 0, 0, 0, 1, 0.2, 0.8, 1, 1),
    # reaches: KLD kept (so `_skewness` IS initialised) with each of the
    # other three off in turn -- the block gate at bscores.cc:824 and the
    # `stdU is computed but not pushed` arm at :852.
    ("weights_kld_and_skew_U", "ip_boolean_contin.csv", "y", 1, 0, 0, 1, 0.2, 0.8, 1, 1),
    ("weights_kld_and_stdU", "ip_boolean_contin.csv", "y", 1, 0, 1, 0, 0.2, 0.8, 1, 1),
    # reaches: a NON-UNIT weight, so a mutation that drops the
    # multiplication survives nothing.
    ("weights_scaled", "ip_boolean_contin.csv", "y", 2, 3, 0.5, -1, 0.2, 0.8, 1, 1),
    # reaches: a non-unit `--alpha`, the only place `_penalty` is read.
    ("penalty_three", "ip_boolean_contin.csv", "y", 1, 1, 1, 1, 0.3, 0.6, 3, 1),
    # reaches: the sweep's own table, at both the sweep's bounds and real
    # ones. Eight rows, no duplicate inputs, so no compression -- the
    # complement of the skewed table in every structural way.
    ("sweep_default_bounds", "ip_sweep.csv", "y", 1, 1, 1, 1, 0, 1, 1, 1),
    ("sweep_real_bounds", "ip_sweep.csv", "y", 1, 1, 1, 1, 0.2, 0.8, 1, 1),
)


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
        raise RuntimeError(f"ip_bscore probe compilation exceeded {_COMPILE_TIMEOUT_S}s") from error
    if result.returncode != 0:
        raise RuntimeError(f"ip_bscore probe compilation failed:\n{result.stderr}")


def _run_probe(args: list[str]) -> object:
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
        raise RuntimeError(f"ip_bscore probe run exceeded {_RUN_TIMEOUT_S}s: {args}") from error
    if not result.stdout.strip():
        # An empty capture compares equal to any other empty capture, which
        # is how a harness reports false agreement (docs/CARRY-FORWARD.md
        # Sec 5.10). Refuse it here rather than write a vacuous fixture.
        raise RuntimeError(f"ip_bscore probe produced NO output for {args}: {result.stderr}")
    return json.loads(result.stdout)


def capture_fixture() -> None:
    fixture: dict[str, object] = {
        "stats": _run_probe(["stats"]),
        "activation": _run_probe(["activation"]),
        "tables": {},
    }
    tables: dict[str, object] = fixture["tables"]
    for label, csv, target, k, j, u, x, lo, hi, pen, pos in _TABLE_CASES:
        entry = _run_probe(
            [
                "table",
                str(TABLE_DATA / csv),
                target,
                str(k),
                str(j),
                str(u),
                str(x),
                str(lo),
                str(hi),
                str(pen),
                str(pos),
                *_TREES,
            ]
        )
        # The absolute CSV path the probe was given is not recorded (it is
        # machine-specific); the basename is, so the test can find the
        # committed copy under tests/fixtures/table_data/.
        entry["csv"] = csv
        entry["target"] = target
        tables[label] = entry

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_ip_bscore.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/ip_bscore_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "description": (
                    "hand-built Counter<contin_t,contin_t> pairs and a fixed "
                    "(activation, min, max, penalty) grid, both hardcoded in "
                    "ip_bscore_probe.cc; plus interesting_predicate_bscore over "
                    "tests/fixtures/table_data/ip_boolean_contin.csv, its "
                    "row-reversed twin ip_boolean_contin_reversed.csv, and "
                    "ip_sweep.csv (byte-for-byte tools/flag_sweep.py's "
                    "DATASETS['ip']), at 14 weight/activation/positive "
                    "combinations x 6 fixed combo trees. No RNG anywhere."
                ),
                "csv_files": sorted({case[1] for case in _TABLE_CASES}),
                "trees": list(_TREES),
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
