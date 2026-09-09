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
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.moses.types import _to_score_t
from pymetta.moses.scoring.time_dispersion import TemporalGranularity, bscore_ctable_time_dispersion

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
    from pymetta.comboreduct.table.table import CTable

    ct = CTable("y", ["x1"])
    assert _ConcreteTD(ct)._Hmax == -math.inf


def test_multiplier_other_than_one_raises():
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    with pytest.raises(AssertionError, match="Multiplier"):
        _ConcreteTD(ct, multiplier=2)


def test_get_timestamp_class_is_a_raising_gate():
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = _ConcreteTD(ct)
    with pytest.raises(NotImplementedError):
        bsc.get_timestamp_class(None)


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
