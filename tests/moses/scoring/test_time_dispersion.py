"""C++: moses/moses/scoring/time_dispersion.h and time_dispersion.cc

Differential oracle against tests/fixtures/scorers.json's "time_dispersion"
cases, plus direct unit/boundary/mutation-proof tests.

`bscore_ctable_time_dispersion` is abstract (inherits `bscore_base`'s
pure-virtual `__call__`/`best_possible_bscore`, and does not override
either) -- every test here uses a trivial concrete subclass, matching
tools/cpp_probe/scorers_probe.cc's own `probe_time_dispersion` pattern.
"""

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.moses.moses.types import _to_score_t
from pymoses.moses.scoring.scoring_base import _ieee_div
from pymoses.moses.scoring.time_dispersion import (
    TemporalGranularity,
    _powf,
    bscore_ctable_time_dispersion,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_CASES = json.loads((_FIXTURE_DIR / "scorers.json").read_text())["time_dispersion"]
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"


class _ConcreteTD(bscore_ctable_time_dispersion):
    """Trivial concrete subclass -- `__call__`/`best_possible_bscore` are
    never exercised by anything in this file.
    """

    def __call__(self, tr):
        return []

    def best_possible_bscore(self):
        return []


def _from_json_scalar(x):
    """Undo scorers_probe.cc's `float_json`'s "nan"/"inf"/"-inf" string
    encoding for a non-finite value.
    """
    if x == "nan":
        return math.nan
    if x == "inf":
        return math.inf
    if x == "-inf":
        return -math.inf
    return x


def _assert_matches(actual: float, expected) -> None:
    expected = _from_json_scalar(expected)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


# ---------------------------------------------------------------------------
# _Hmax on real (untimestamped) CTables -- always 0.0 in this port
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case_name", sorted(n for n in _CASES if n.startswith("hmax::")), ids=lambda n: n
)
def test_hmax_matches_cpp_probe(case_name):
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / case["file"]), case["target_feature"])
    ct = table.compressed()
    bsc = _ConcreteTD(ct)
    _assert_matches(bsc._Hmax, case["hmax"])


def test_hmax_is_always_zero_for_any_non_empty_ctable():
    """Not an approximation -- see this module's own docstring: classic's
    OWN untimestamped construction collapses to exactly one timestamp
    class for any input this port can ever produce.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    assert _ConcreteTD(ct)._Hmax == 0.0


def test_hmax_is_negative_infinity_for_an_empty_ctable():
    """Boundary case: `log2(0)` in C++ is `-HUGE_VAL`, not an exception --
    `timestamp_classes.size() == 0` only when the table has zero
    observations at all.
    """
    from pymoses.comboreduct.table.table import CTable

    ct = CTable("y", ["x1"])
    assert _ConcreteTD(ct)._Hmax == -math.inf


def test_multiplier_other_than_one_raises():
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    with pytest.raises(AssertionError, match="Multiplier"):
        _ConcreteTD(ct, multiplier=2)


def test_month_granularity_raises_on_any_non_empty_ctable():
    """MUTATION PROVEN. Classic's constructor calls `get_timestamp_class`
    once per observation (time_dispersion.cc:49-51); at `month` that
    OC_ASSERTs `!timestamp.is_not_a_date()` (:61-64), and every timestamp
    this port can produce IS not-a-date, so the assertion fires on the
    first observation. Deleting the `granularity == month` check in
    `bscore_ctable_time_dispersion.__init__` makes this test fail (no
    exception raised) while every other test in this file still passes --
    which is exactly how the divergence survived: nothing here or in
    test_precision_bscore.py ever passed a non-default granularity.

    Measured against the real binary, 2026-09-09:
    `-Hpre -i$CLASSIC/tests/moses/dataset.csv -m300 -W1 -q0.1 -w1.0
    --time-bscore-granularity=month` exits 134 in classic (`terminate
    called after throwing ... AssertionException / what(): timestamp is
    not a valid date ... (time_dispersion.cc:62)`, core dumped); the same
    command with the flag omitted -- or with `=day` -- exits 0 and prints
    a normal candidate list in both classic and this port.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    with pytest.raises(AssertionError, match="timestamp is not a valid date"):
        _ConcreteTD(ct, granularity=TemporalGranularity.month)


def test_day_granularity_does_not_raise_on_the_same_ctable():
    """The complement, and the thing the gate above must NOT break:
    `get_timestamp_class` at `day` is the identity (time_dispersion.cc:
    59-60) and cannot fail, so the default path is untouched. A gate
    written without the granularity test -- raising for every
    construction -- fails here.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    assert _ConcreteTD(ct, granularity=TemporalGranularity.day)._Hmax == 0.0


def test_month_granularity_on_an_EMPTY_ctable_does_not_raise():
    """The other half of the boundary, and it is classic's behaviour, not
    a convenience: the assertion lives INSIDE the per-observation loop
    (`for (iorow : _ctable) for (tcv : iorow.second)`,
    time_dispersion.cc:49-51), so a table with zero observations never
    evaluates it and `_Hmax` is `log2(0) == -inf` at either granularity.
    A gate written as an unconditional `if granularity == month: raise`
    fails here.
    """
    from pymoses.comboreduct.table.table import CTable

    ct = CTable("y", ["x1"])
    assert _ConcreteTD(ct, granularity=TemporalGranularity.month)._Hmax == -math.inf


def test_get_timestamp_class_day_is_the_identity():
    """`case TemporalGranularity::day: return timestamp;`
    (time_dispersion.cc:59-60) -- including for the not-a-date sentinel,
    which is why an untimestamped table gets exactly one timestamp class.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    bsc = _ConcreteTD(table.compressed(), granularity=TemporalGranularity.day)
    assert bsc.get_timestamp_class(None) is None
    assert bsc.get_timestamp_class(date(2012, 3, 23)) == date(2012, 3, 23)


def test_get_timestamp_class_month_truncates_to_the_first_of_the_month():
    """`return TTable::value_type(timestamp.year(), timestamp.month(), 1);`
    (time_dispersion.cc:65). Two dates in the same month must collapse to
    ONE class and two in different months must not -- a `day`-style
    identity would pass a single-date check.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    bsc = _ConcreteTD(table.compressed(), granularity=TemporalGranularity.day)
    bsc._granularity = TemporalGranularity.month
    assert bsc.get_timestamp_class(date(2012, 3, 23)) == date(2012, 3, 1)
    assert bsc.get_timestamp_class(date(2012, 3, 1)) == date(2012, 3, 1)
    assert bsc.get_timestamp_class(date(2012, 4, 1)) == date(2012, 4, 1)


def test_get_timestamp_class_month_rejects_the_not_a_date_sentinel():
    """`OC_ASSERT(!timestamp.is_not_a_date(), ...)` (time_dispersion.cc:
    61-64), with classic's own message text. This is what makes the
    constructor abort on the first observation of an untimestamped table
    at `month` granularity.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    bsc = _ConcreteTD(table.compressed(), granularity=TemporalGranularity.day)
    bsc._granularity = TemporalGranularity.month
    with pytest.raises(AssertionError, match="are you sure you have loaded a timestamp feature"):
        bsc.get_timestamp_class(None)


def test_get_timestamp_class_rejects_an_unknown_granularity():
    """The `default:` arm (time_dispersion.cc:65-70), unreachable while
    `TemporalGranularity` has two members but ported so a third fails
    loudly instead of silently getting `day`'s answer.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    bsc = _ConcreteTD(table.compressed(), granularity=TemporalGranularity.day)
    bsc._granularity = 7
    with pytest.raises(AssertionError, match="Case 7 not implemented"):
        bsc.get_timestamp_class(date(2012, 3, 23))


def test_hmax_counts_real_timestamp_classes():
    """The constructor loop is now REAL, not counted. A three-date table
    gives `log2(3)` at `day` granularity; the same table gives `log2(2)`
    at `month`, because two of its dates share a month. An implementation
    that hard-coded "1 class" (which is all an untimestamped table can
    produce) passes neither.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "timestamp_bool_dup.csv"), "y", "when")
    ct = table.compressed()
    assert sorted(ct.get_timestamps()) == [date(2012, 1, 3), date(2012, 1, 11), date(2012, 1, 19)]
    assert _ConcreteTD(ct, granularity=TemporalGranularity.day)._Hmax == pytest.approx(
        math.log2(3), abs=1e-7
    )
    assert _ConcreteTD(ct, granularity=TemporalGranularity.month)._Hmax == 0.0


# ---------------------------------------------------------------------------
# get_time_dispersion_penalty: the pure entropy-penalty formula, tested
# directly on a synthetic CTableTime-shaped mapping.
# ---------------------------------------------------------------------------


def _synthetic_ctt(true_counts: list[float]) -> dict:
    """One bucket per `true_counts` entry; false count fixed at 1.0 per
    bucket, matching tools/cpp_probe/scorers_probe.cc's own
    `timedisp_penalty` mode exactly (so the SAME fixture cases apply).
    """
    return {
        i: Counter({builtin.logical_true: tc, builtin.logical_false: 1.0})
        for i, tc in enumerate(true_counts)
    }


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case_name", sorted(n for n in _CASES if n.startswith("penalty::")), ids=lambda n: n
)
def test_get_time_dispersion_penalty_matches_cpp_probe(case_name):
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, case["pressure"], case["exponent"], TemporalGranularity.day)

    n_buckets = len(case["true_counts"])
    # Override _Hmax exactly as tools/cpp_probe/scorers_probe.cc's own
    # probe_time_dispersion.set_hmax does: log2(n_buckets), narrowed to
    # score_t (float32) at that point, matching `(float)log2((double)n)`.
    bsc._Hmax = -math.inf if n_buckets == 0 else _to_score_t(math.log2(n_buckets))
    _assert_matches(bsc._Hmax, case["hmax"])

    ctt = _synthetic_ctt(case["true_counts"])
    penalty = bsc.get_time_dispersion_penalty(ctt)
    _assert_matches(penalty, case["penalty"])


def test_zero_pressure_gives_zero_penalty_regardless_of_entropy():
    """Boundary: `pressure=0.0` scales ANY (finite or infinite) distorted
    penalty down to exactly `0.0` -- `0.0 * anything_finite == 0.0`, and
    classic's own formula does not special-case this, it falls out of the
    multiplication.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, time_dispersion_pressure=0.0, time_dispersion_exponent=1.0)
    bsc._Hmax = _to_score_t(math.log2(2))
    ctt = _synthetic_ctt([10.0, 0.0])
    assert bsc.get_time_dispersion_penalty(ctt) == 0.0


def test_uniform_distribution_gives_zero_penalty():
    """Boundary: H == Hmax exactly gives `normalized_H == 1.0`, so
    `penalty == 0.0` before the `pow`/pressure scaling. NOTE: `act_time_
    dist` is NOT normalized to sum to 1 across buckets (each `prob` is
    divided by the GRAND total of true+false counts over every bucket,
    not by the sum of true counts alone, time_dispersion.cc:84-93) -- so
    "uniform true_counts" alone does not reach this boundary (measured:
    `true_counts=[5.0, 5.0]` gives a nonzero penalty, exactly
    `scorers.json`'s own "uniform_2_buckets" case). Achieving H == Hmax
    with this exact formula requires false counts of 0 (so `act_time_
    dist` genuinely sums to 1) -- built directly here rather than via
    `_synthetic_ctt`, which fixes false=1.0 per bucket to match the probe.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, time_dispersion_pressure=1.0, time_dispersion_exponent=1.0)
    bsc._Hmax = _to_score_t(math.log2(2))
    ctt = {
        0: Counter({builtin.logical_true: 5.0}),
        1: Counter({builtin.logical_true: 5.0}),
    }
    penalty = bsc.get_time_dispersion_penalty(ctt)
    assert penalty == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The C99-domain-error-returns-NaN fix for math.pow (negative base,
# non-integer exponent) -- Python's math.pow raises ValueError instead.
# ---------------------------------------------------------------------------


def test_negative_penalty_with_fractional_exponent_gives_nan_not_a_raise():
    """A FINITE negative `penalty` (not the `-inf` this port's own
    degenerate `_Hmax == 0.0` case produces -- `pow(-inf, 0.5)` is
    IEEE-754-special-cased to `+inf`, not a domain error, so it does NOT
    exercise this fix) with a FRACTIONAL exponent: `std::pow` of a
    negative base and a non-integer exponent is a C99 domain error
    returning `NaN` (does not throw); Python's `math.pow` raises
    `ValueError` for the identical input -- proven here to be caught and
    converted, not left to escape. Reuses "uniform_2_buckets"'s own
    finite `penalty == -0.0525287389755249` (measured against classic
    with `exponent=1.0`) with a FRACTIONAL exponent substituted in.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, time_dispersion_pressure=1.0, time_dispersion_exponent=0.5)
    bsc._Hmax = _to_score_t(math.log2(2))
    ctt = _synthetic_ctt([5.0, 5.0])
    penalty = bsc.get_time_dispersion_penalty(ctt)
    assert math.isnan(penalty)


def test_fractional_counts_are_truncated_to_unsigned_as_cpp_does():
    """`row_count` truncates toward zero, because C++ assigns a double to an
    `unsigned`.

    `time_dispersion.cc:87` is `unsigned row_count = tio.second.total_count();`
    and `total_count()` returns `count_t`, which is `typedef double`
    (`table.h:494`). That implicit conversion truncates. `CTable` counts really
    are floats -- weighted rows produce fractional ones -- so this is reachable,
    not theoretical.

    Every other penalty case here uses whole-number counts, where truncation is
    a no-op, so none of them can tell the faithful code from a version that
    drops it. Found by an independent verification pass: removing `_to_usize`
    left the whole suite green.

    With buckets `[0.5, 2.5]` and a fixed false count of 1.0 each, truncation
    gives row counts 1 and 3; without it, 1.5 and 3.5 -- different
    distributions, different entropy, and a penalty that differs in the second
    decimal place.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, time_dispersion_pressure=1.0, time_dispersion_exponent=1.0)
    bsc._Hmax = _to_score_t(math.log2(2))

    penalty = bsc.get_time_dispersion_penalty(_synthetic_ctt([0.5, 2.5]))

    # Truncating: 0.20120501518249512. Not truncating: 0.1678072214126587.
    assert penalty == pytest.approx(0.20120501518249512, abs=0.0)


# ---------------------------------------------------------------------------
# _powf -- glibc's powf, which Python's math.pow is NOT a drop-in for
# ---------------------------------------------------------------------------

_POWF_CASES = json.loads((_FIXTURE_DIR / "scorers.json").read_text())["powf"]


def _powf_arg(token: str) -> float:
    """Mirror tools/cpp_probe/scorers_probe.cc's own `parse_float_arg`.

    "negnan" must come from a real `0.0 / 0.0`, not from `-math.nan`: the
    former is x86-64's "QNaN floating-point indefinite" (sign bit SET) and
    is exactly what `_ieee_div(H, _Hmax)` produces, which is the value the
    penalty formula actually feeds to `pow`.
    """
    if token == "negnan":
        return _ieee_div(0.0, 0.0)
    if token == "posnan":
        return -_ieee_div(0.0, 0.0)
    if token == "inf":
        return math.inf
    if token == "-inf":
        return -math.inf
    return float(token)


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_POWF_CASES), ids=lambda n: n)
def test_powf_matches_cpp_probe(case_name):
    """`_powf` reproduces `std::pow(float, float)` on this toolchain, SIGN
    INCLUDED. The sign is asserted separately because a JSON "nan" cannot
    carry it, and because the sign is the only thing that differs in the
    case the CLI actually reaches.
    """
    case = _POWF_CASES[case_name]
    got = _powf(_powf_arg(case["base"]), _powf_arg(case["exponent"]))
    _assert_matches(got, case["value"])
    assert (math.copysign(1.0, got) < 0.0) == bool(case["signbit"]), (
        f"{case_name}: sign bit differs from classic's"
    )


def test_powf_fixture_is_not_all_one_sign():
    """Guard against a vacuous parametrisation: the corpus must contain
    BOTH sign outcomes, or `test_powf_matches_cpp_probe`'s sign assertion
    could pass against an implementation that always returns a positive
    result.
    """
    signs = {bool(c["signbit"]) for c in _POWF_CASES.values()}
    assert signs == {False, True}
    nan_signs = {bool(c["signbit"]) for c in _POWF_CASES.values() if c["value"] == "nan"}
    assert nan_signs == {False, True}


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case_name", sorted(n for n in _CASES if n.startswith("penalty_tf::")), ids=lambda n: n
)
def test_get_time_dispersion_penalty_all_true_bucket_matches_cpp_probe(case_name):
    """The `_Hmax == 0 and H == 0` case, i.e. `0.0f / 0.0f`.

    `penalty::single_bucket_degenerate` cannot reach it: the
    `timedisp_penalty` probe mode hardcodes a false count of 1.0 per
    bucket, so `H > 0` always and the answer is `-inf`. These cases come
    from the `timedisp_penalty_tf` mode, which takes both counts, and they
    carry `penalty_signbit` because the NaN's sign is what reaches stdout
    (classic prints `nan`, a sign-preserving port prints `-nan`).
    """
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct, case["pressure"], case["exponent"], TemporalGranularity.day)
    n = case["hmax_buckets"]
    bsc._Hmax = -math.inf if n == 0 else _to_score_t(math.log2(n))
    _assert_matches(bsc._Hmax, case["hmax"])

    ctt = {
        i: Counter({builtin.logical_true: t, builtin.logical_false: f})
        for i, (t, f) in enumerate(case["buckets"])
    }
    penalty = bsc.get_time_dispersion_penalty(ctt)
    _assert_matches(penalty, case["penalty"])
    assert (math.copysign(1.0, penalty) < 0.0) == bool(case["penalty_signbit"])


def test_penalty_tf_fixture_covers_both_nan_signs():
    """Non-vacuity guard for the parametrisation above: the corpus must
    contain a `+nan` case AND a `-nan` case, so a `_powf` that dropped the
    odd-integer sign rule fails at least one of them.
    """
    nan_signs = {
        bool(c["penalty_signbit"])
        for n, c in _CASES.items()
        if n.startswith("penalty_tf::") and c["penalty"] == "nan"
    }
    assert nan_signs == {False, True}
