"""Build tools/cpp_probe/scorers_probe.cc and capture its JSON output into
tests/fixtures/scorers.json -- the differential oracle for
pymetta/moses/scoring/{discriminating_bscore,select_bscore,precision_bscore,
time_dispersion}.py.

The committed tests/fixtures/scorers.json is what the default test suite
actually reads (see tests/moses/scoring/test_*.py), so the suite runs
without a C++ toolchain or the classic checkout at all -- same division of
labor as tools/capture_score.py/score.json.

Four families, four top-level fixture keys:

* "discriminating" -- recall_bscore/prerec_bscore/bep_bscore over
  trainingBoolean.dat and trainingBooleanXor.dat (both already vendored
  under tests/fixtures/table_data/ by tools/capture_eval_boolean.py).
  f_one_bscore is NOT captured here: measured live (see this script's own
  probe run log) to be UNCONSTRUCTIBLE in classic -- its hardcoded
  `discriminating_bscore(ct, 0.0, 1.0, 1.0e-20)` base-class call always
  fails `OC_ASSERT(0.0 < min_threshold, ...)` since min_threshold is
  hardcoded to 0.0 while hardness (1.0e-20) is > 0. Ditto bep_bscore's own
  declared DEFAULTS (min_diff=0.0f, hardness=1.0f) -- also unconstructible
  -- so every bep case below passes an explicit, non-default min_diff.
  See discriminating_bscore.py's module docstring for the full writeup;
  this is a real, reproducible upstream defect, not a probe bug (confirmed
  by direct construction: the probe binary aborts with
  opencog::AssertionException naming discriminating_bscore.cc:224 for
  BOTH cases, verified 2026-09-08).

* "select" -- select_bscore over a small vendored sample of classic's own
  trainingRegression.dat (tests/fixtures/table_data/select_regression.csv,
  40 rows, header + a deterministic `shuf --random-source=<(yes 42)`
  sample -- trainingRegression.dat itself is 8000 rows, too slow to
  compress repeatedly across many probe invocations). Every case here
  ALSO captures `_lower_bound`/`_upper_bound`/`_effective_length`/
  `_weights` via the probe's `probe_select_bscore` subclass: classic's
  `select_bscore` constructor calls the (virtual, already-dispatching-to-
  select_bscore's-own-override) `reset_weights()` BEFORE `_lower_bound`/
  `_upper_bound` are assigned later in the same constructor body --
  `_lower_bound`/`_upper_bound` are plain `score_t` (float) members with
  no initializer, so that premature call reads UNINITIALIZED memory.
  Measured live (2026-09-08, this exact probe, `select_regression.csv`,
  two independent runs): byte-identical output both times, with
  `_weights` uniformly 1.0 and `_effective_length` exactly equal to the
  row count -- the signature of EVERY row being treated as outside the
  (nonsensical, garbage) `[_lower_bound, _upper_bound]` window at that
  point, consistent with (though not provably identical to) a
  zero-initialized or otherwise-degenerate `[0.0, 0.0]` read on this
  toolchain/build. See select_bscore.py's own module docstring for the
  full "DO NOT FIX" writeup and why this port pins 0.0/0.0.

* "precision" -- precision_bscore over precision.csv, replicating
  scoringUTest.cxxtest's own `test_precision_bscore_best_possible_bscore`
  case (activation_pressure=1.0, min_activation=0.5, max_activation=1.0)
  plus additional trees and a `positive=False` (negative-predictive-value)
  case matching MOSESUTest's `test_negative_predictive_value`.

* "powf" -- `std::pow(float, float)`, the exact call at
  time_dispersion.cc:104, with its result's sign bit reported separately
  because JSON "nan" cannot carry it. The oracle for `_powf`
  (time_dispersion.py); see `capture_powf`'s own docstring.

* "time_dispersion" -- `_Hmax` measured on two real (untimestamped)
  CTables (confirming it is always exactly 0.0 in this port -- see
  time_dispersion.py's module docstring for why), plus
  `get_time_dispersion_penalty` on several synthetic CTableTime
  distributions built directly by the probe (bypassing the
  timestamp-collection pipeline this port cannot reach from real data).

Every probe run is resource-bounded (`ulimit -v` + `timeout`): this
machine has been OOM-killed twice by unbounded probe runs (see CLAUDE.md).
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

PROBE_SRC = REPO_ROOT / "tools" / "cpp_probe" / "scorers_probe.cc"
PROBE_BIN = REPO_ROOT / "tools" / "cpp_probe" / "build" / "scorers_probe"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "scorers.json"
DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "table_data"

CLASSIC = classic_dir()
PREFIX = moses_prefix()

# ulimit -v 2097152 (KiB) / timeout 30 -- see CLAUDE.md.
_MEM_LIMIT_BYTES = 2097152 * 1024
_TIMEOUT_S = 30


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def build_probe() -> None:
    PROBE_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
            str(PREFIX / "lib"),
            "-L",
            str(PREFIX / "lib" / "opencog"),
            "-Wl,-rpath," + str(PREFIX / "lib"),
            "-Wl,-rpath," + str(PREFIX / "lib" / "opencog"),
            "-lcomboreduct",
            "-lmoses",
            "-lcogutil",
        ],
        check=True,
    )


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


# ---------------------------------------------------------------------------
# discriminating_bscore family
# ---------------------------------------------------------------------------

_BOOL_CSV = CLASSIC / "trainingBoolean.dat"
_XOR_CSV = CLASSIC / "trainingBooleanXor.dat"

# (dataset, scorer, min_thr, max_thr, hardness, tree, ratio-or-None).
# min_thr is always > 0 -- see the module docstring: bep_bscore's OWN
# default (0.0) and f_one_bscore's hardcoded 0.0 both fail
# discriminating_bscore's own constructor assert, so every real case here
# (and every real case classic's own table-problems.cc REGRESSION call
# sites construct) uses a strictly positive min_thr.
_DISC_CASES = [
    (_BOOL_CSV, "y", "recall", 0.8, 1.0, 1.0, "and($1 $2)", None),
    (_BOOL_CSV, "y", "recall", 0.8, 1.0, 1.0, "or($1 $2)", None),
    (_BOOL_CSV, "y", "recall", 0.8, 1.0, 1.0, "and($1 $2)", 3.5),
    (_BOOL_CSV, "y", "recall", 0.5, 1.0, 1.0, "$3", None),
    (_BOOL_CSV, "y", "recall", 0.8, 1.0, 0.0, "and($1 $2)", None),  # hardness=0: no penalty
    (_XOR_CSV, "y", "recall", 0.8, 1.0, 1.0, "or(and($1 !$2) and(!$1 $2))", None),
    (_BOOL_CSV, "y", "prerec", 0.5, 1.0, 1.0, "and($1 $2)", None),
    (_BOOL_CSV, "y", "prerec", 0.5, 1.0, 1.0, "or($1 $2)", None),
    (_BOOL_CSV, "y", "prerec", 0.5, 1.0, 1.0, "and($1 $2)", 3.5),
    (_XOR_CSV, "y", "prerec", 0.5, 1.0, 1.0, "or(and($1 !$2) and(!$1 $2))", None),
    (_BOOL_CSV, "y", "bep", 0.1, 0.5, 1.0, "or($1 $2)", None),
    (_BOOL_CSV, "y", "bep", 0.1, 0.5, 1.0, "and($1 $2)", None),
    (_BOOL_CSV, "y", "bep", 0.1, 0.5, 1.0, "and($1 $2)", 3.5),
    (_XOR_CSV, "y", "bep", 0.1, 0.5, 1.0, "or(and($1 !$2) and(!$1 $2))", None),
]


def capture_discriminating() -> dict:
    cases = {}
    for path, target, scorer, min_thr, max_thr, hardness, tree, ratio in _DISC_CASES:
        args = ["disc", str(path), target, scorer, str(min_thr), str(max_thr), str(hardness), tree]
        if ratio is not None:
            args.append(str(ratio))
        name = f"{path.name}::{scorer}::{tree}::min={min_thr}::max={max_thr}::hard={hardness}"
        if ratio is not None:
            name += f"::ratio={ratio}"
        case = run_probe(args)
        case.update(
            file=path.name,
            target_feature=target,
            min_threshold=min_thr,
            max_threshold=max_thr,
            hardness=hardness,
            tree=tree,
            complexity_ratio=ratio,
        )
        cases[name] = case
    return cases


# ---------------------------------------------------------------------------
# select_bscore
# ---------------------------------------------------------------------------

_SELECT_CSV = DATA_DIR / "select_regression.csv"

_SELECT_CASES = [
    (0.3, 0.8, 1.0, True, "0<($1)"),
    (0.3, 0.8, 1.0, True, "0<($2)"),
    (0.0, 1.0, 1.0, True, "0<($1)"),
    (0.5, 0.9, 1.0, False, "0<($1)"),
]


def capture_select() -> dict:
    cases = {}
    for lower, upper, hardness, positive, tree in _SELECT_CASES:
        args = [
            "select",
            str(_SELECT_CSV),
            "y",
            str(lower),
            str(upper),
            str(hardness),
            "1" if positive else "0",
            tree,
        ]
        name = f"select::{tree}::lower={lower}::upper={upper}::positive={positive}"
        case = run_probe(args)
        case.update(
            file=_SELECT_CSV.name,
            target_feature="y",
            lower_percentile=lower,
            upper_percentile=upper,
            hardness=hardness,
            positive=positive,
            tree=tree,
        )
        cases[name] = case
    return cases


# ---------------------------------------------------------------------------
# precision_bscore
# ---------------------------------------------------------------------------

_PRECISION_CSV = DATA_DIR / "precision.csv"

_PRECISION_CASES = [
    # Replicates scoringUTest.cxxtest's own test_precision_bscore_best_possible_bscore.
    (1.0, 0.5, 1.0, True, "$2", None),
    (1.0, 0.5, 1.0, True, "$1", None),
    # MOSESUTest's test_negative_predictive_value uses positive=False, tree "$1".
    (1.0, 0.5, 1.0, False, "$1", None),
    (1.0, 0.5, 1.0, True, "and($1 $2)", 3.5),
    (0.0, 0.0, 1.0, True, "$2", None),  # activation_pressure=0: no activation penalty
]


def capture_precision() -> dict:
    cases = {}
    for act_pressure, min_act, max_act, positive, tree, ratio in _PRECISION_CASES:
        args = [
            "precision",
            str(_PRECISION_CSV),
            "target",
            str(act_pressure),
            str(min_act),
            str(max_act),
            "1" if positive else "0",
            tree,
        ]
        if ratio is not None:
            args.append(str(ratio))
        name = f"precision::{tree}::pos={positive}::act_pressure={act_pressure}"
        if ratio is not None:
            name += f"::ratio={ratio}"
        case = run_probe(args)
        case.update(
            file=_PRECISION_CSV.name,
            target_feature="target",
            activation_pressure=act_pressure,
            min_activation=min_act,
            max_activation=max_act,
            positive=positive,
            tree=tree,
            complexity_ratio=ratio,
        )
        cases[name] = case
    return cases


# ---------------------------------------------------------------------------
# time_dispersion
# ---------------------------------------------------------------------------


def capture_time_dispersion() -> dict:
    cases = {}
    for path, target in [(_BOOL_CSV, "y"), (DATA_DIR / "weighted.csv", "r")]:
        case = run_probe(["timedisp_hmax", str(path), target])
        cases[f"hmax::{path.name}"] = {**case, "file": path.name, "target_feature": target}

    # (pressure, exponent, true_counts...) -- false count fixed at 1.0 per
    # bucket by the probe. Boundary cases: uniform distribution (H==Hmax,
    # penalty==0 for any pressure/exponent) and single-bucket (Hmax==0,
    # exercises the 0/0 edge -- see time_dispersion.py's module docstring).
    penalty_cases = [
        ("uniform_2_buckets", 1.0, 1.0, [5.0, 5.0]),
        ("uniform_3_buckets", 1.0, 1.0, [5.0, 5.0, 5.0]),
        ("skewed_2_buckets", 1.0, 1.0, [10.0, 0.0]),
        ("skewed_2_buckets_exponent2", 0.5, 2.0, [10.0, 0.0]),
        ("single_bucket_degenerate", 1.0, 1.0, [7.0]),
        ("zero_pressure", 0.0, 1.0, [10.0, 0.0]),
    ]
    for name, pressure, exponent, true_counts in penalty_cases:
        args = ["timedisp_penalty", str(_BOOL_CSV), "y", str(pressure), str(exponent)]
        args.extend(str(c) for c in true_counts)
        case = run_probe(args)
        cases[f"penalty::{name}"] = {
            **case,
            "pressure": pressure,
            "exponent": exponent,
            "true_counts": true_counts,
        }

    # (pressure, exponent, hmax_buckets, [(true, false), ...]) -- both
    # counts explicit, so a bucket can be ALL TRUE. That is the case the
    # CLI reaches once a table has exactly one timestamp class
    # (`--time-bscore-granularity=month` over a single-month table): _Hmax
    # is log2(1) == 0 AND H is 0, so `H / _Hmax` is 0.0f/0.0f and the
    # penalty is a NaN whose SIGN is what reaches stdout. The probe's
    # `penalty_signbit` carries it, because JSON "nan" cannot. The
    # `_degenerate` case above can never reach this: its false count is
    # hardcoded to 1.0, so H > 0 and the result is -inf, not NaN.
    penalty_tf_cases = [
        ("one_class_all_true", 0.5, 1.0, 1, [(7.0, 0.0)]),
        ("one_class_all_true_exponent2", 0.5, 2.0, 1, [(7.0, 0.0)]),
        ("one_class_mixed", 0.5, 1.0, 1, [(5.0, 2.0)]),
        ("two_classes_all_true", 0.5, 1.0, 2, [(3.0, 0.0), (4.0, 0.0)]),
    ]
    for name, pressure, exponent, hmax_buckets, buckets in penalty_tf_cases:
        args = [
            "timedisp_penalty_tf",
            str(_BOOL_CSV),
            "y",
            str(pressure),
            str(exponent),
            str(hmax_buckets),
        ]
        args.extend(f"{t}:{f}" for t, f in buckets)
        case = run_probe(args)
        cases[f"penalty_tf::{name}"] = {
            **case,
            "pressure": pressure,
            "exponent": exponent,
            "hmax_buckets": hmax_buckets,
            "buckets": [list(b) for b in buckets],
        }
    return cases


def capture_powf() -> dict:
    """`std::pow(float, float)` as time_dispersion.cc:104 calls it.

    Python's `math.pow` differs from glibc's `powf` in four families, and
    every one of them is reachable from `--time-dispersion-exponent`:
    the NaN sign rule, a negative finite base with a non-integer exponent
    (C returns -nan, Python raises `ValueError`), a zero base with a
    negative exponent (C returns a signed infinity, Python raises), and
    overflow (C returns a signed infinity, Python raises `OverflowError`).
    `_powf` (time_dispersion.py) reproduces all four; these are its
    oracle.
    """
    pairs = [
        ("negnan", "1.0"),
        ("negnan", "-1.0"),
        ("negnan", "3.0"),
        ("negnan", "2.0"),
        ("negnan", "0.5"),
        ("negnan", "1.5"),
        ("negnan", "0.0"),
        ("negnan", "1e30"),
        ("posnan", "1.0"),
        ("posnan", "2.0"),
        ("posnan", "0.5"),
        ("-3.0", "0.5"),
        ("-3.0", "1.5"),
        ("-0.5", "0.5"),
        ("-2.0", "3.0"),
        ("0.0", "-1.0"),
        ("-0.0", "-1.0"),
        ("-0.0", "-2.0"),
        ("0.5", "-1e30"),
        ("0.5", "1e30"),
        ("-0.5", "-1e30"),
        ("2.0", "1e30"),
        ("-inf", "1.0"),
        ("-inf", "0.5"),
        ("-inf", "3.0"),
        ("0.5", "2.0"),
    ]
    cases = {}
    for base, exponent in pairs:
        cases[f"{base}^{exponent}"] = run_probe(["powf", base, exponent])
    return cases


def main(argv: list[str]) -> int:
    del argv
    if not PROBE_SRC.is_file():
        raise FileNotFoundError(f"probe source not found: {PROBE_SRC}")
    build_probe()

    for path in (_BOOL_CSV, _XOR_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"fixture input not found: {path}")
    if not _SELECT_CSV.is_file():
        raise FileNotFoundError(
            f"{_SELECT_CSV} not found -- see this script's module docstring for how it "
            "was derived from classic's trainingRegression.dat"
        )
    if not _PRECISION_CSV.is_file():
        raise FileNotFoundError(f"{_PRECISION_CSV} not found")

    # weighted.csv is already vendored by tools/capture_score.py; copy it
    # again here is a no-op if identical, but do NOT overwrite a
    # differently-sourced copy silently.
    weighted_src = CLASSIC / "tests" / "comboreduct" / "table" / "weighted.csv"
    weighted_dst = DATA_DIR / "weighted.csv"
    if weighted_src.is_file() and weighted_src.read_bytes() != weighted_dst.read_bytes():
        raise RuntimeError(f"{weighted_dst} diverges from classic's own {weighted_src}")

    fixture = {
        "discriminating": capture_discriminating(),
        "select": capture_select(),
        "precision": capture_precision(),
        "time_dispersion": capture_time_dispersion(),
        "powf": capture_powf(),
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_scorers.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source="tools/cpp_probe/scorers_probe.cc",
            probe_source_path=PROBE_SRC,
            probe_kind="cpp_probe",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "n_discriminating_cases": len(fixture["discriminating"]),
                "n_select_cases": len(fixture["select"]),
                "n_precision_cases": len(fixture["precision"]),
                "n_time_dispersion_cases": len(fixture["time_dispersion"]),
                "n_powf_cases": len(fixture["powf"]),
                "select_regression_csv": "tests/fixtures/table_data/select_regression.csv",
            },
        ),
    )
    print(
        f"wrote {FIXTURE_PATH}: "
        f"{len(fixture['discriminating'])} discriminating, "
        f"{len(fixture['select'])} select, "
        f"{len(fixture['precision'])} precision, "
        f"{len(fixture['time_dispersion'])} time_dispersion, "
        f"{len(fixture['powf'])} powf cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
