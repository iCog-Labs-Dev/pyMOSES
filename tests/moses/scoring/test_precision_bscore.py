"""C++: moses/moses/scoring/precision_bscore.h and precision_bscore.cc

Differential oracle against:
* scoringUTest.cxxtest's own `test_precision_bscore_best_possible_bscore`
  (precision.csv, activation_pressure=1.0, min_activation=0.5,
  max_activation=1.0, expects `precision = 5.f/7.f`) -- the ONE live
  cxxtest coverage of precision_bscore that exists in classic.
* tests/fixtures/scorers.json's "precision" cases (captured by
  tools/capture_scorers.py from tools/cpp_probe/scorers_probe.cc's
  `precision` mode).

Plus direct unit/boundary/mutation-proof tests for the out-of-scope gates
(time_bscore, dispersion_pressure>0.0, ensemble scoring) and the -inf/NaN
handling in get_activation_penalty.
"""

import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.moses.types import _to_score_t, scored_combo_tree_set
from pymetta.moses.scoring.precision_bscore import precision_bscore, precision_conj_bscore

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_CASES = json.loads((_FIXTURE_DIR / "scorers.json").read_text())["precision"]
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"
_PRECISION_CSV = _TABLE_DATA_DIR / "precision.csv"


def _from_json_scalar(x):
    if x == "nan":
        return math.nan
    if x == "inf":
        return math.inf
    if x == "-inf":
        return -math.inf
    return x


def _assert_eq(actual, expected) -> None:
    expected = _from_json_scalar(expected)
    if isinstance(expected, float) and math.isnan(expected):
        assert isinstance(actual, float) and math.isnan(actual)
    else:
        assert actual == expected


# ---------------------------------------------------------------------------
# scoringUTest.cxxtest's own live coverage, transliterated
# ---------------------------------------------------------------------------


def test_precision_bscore_best_possible_bscore():
    """C++: tests/moses/scoringUTest.cxxtest:47 test_precision_bscore_best_possible_bscore

    `score_t precision = 5.f/7.f, act_penalty = 0.f;
    TS_ASSERT_EQUALS(bsc.front(), precision + act_penalty);` -- `5.f/7.f`
    is a FLOAT32-native division (both operands `float` literals),
    reproduced here via `_to_score_t(5.0/7.0)` (a single division,
    narrowed once -- no double-rounding risk, see discriminating_bscore.
    py's `get_threshold_penalty` docstring for when that risk DOES apply).
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    pbsc = precision_bscore(ct, 1.0, 0.5, 1.0)
    bsc = pbsc.best_possible_bscore()
    precision = _to_score_t(5.0 / 7.0)
    act_penalty = 0.0
    assert bsc[0] == precision + act_penalty


# ---------------------------------------------------------------------------
# Differential oracle
# ---------------------------------------------------------------------------


def test_scorers_fixture_has_precision_cases():
    assert len(_CASES) >= 4


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_CASES), ids=sorted(_CASES))
def test_precision_bscore_matches_cpp_probe(case_name):
    case = _CASES[case_name]
    table = loadTable(str(_TABLE_DATA_DIR / case["file"]), case["target_feature"])
    ct = table.compressed()

    bsc = precision_bscore(
        ct,
        case["activation_pressure"],
        case["min_activation"],
        case["max_activation"],
        case["positive"],
    )
    if case["complexity_ratio"] is not None:
        bsc.set_complexity_coef(case["complexity_ratio"])

    tr = str_to_combo_tree(case["tree"])
    bs = bsc(tr)
    best = bsc.best_possible_bscore()
    worst = bsc.worst_possible_bscore()
    mi = bsc.min_improv()
    coef = bsc.get_complexity_coef()
    ssum = bsc.sum_bscore(bs)

    # bs = one row-keyed element per ctable row, then a fixed 2-element
    # tail (phantom, activation_penalty) -- see mode_disc's own comment
    # in scorers_probe.cc for why the row-keyed prefix is compared as a
    # sorted multiset, not positionally.
    prefix_len = len(ct)
    expected_prefix = sorted(_from_json_scalar(r["value"]) for r in case["bscore_rows"])
    assert sorted(bs[:prefix_len]) == expected_prefix, case_name

    tail = bs[prefix_len:]
    assert len(tail) == len(case["bscore_tail"]), case_name
    for actual, expected in zip(tail, case["bscore_tail"], strict=True):
        _assert_eq(actual, expected)

    assert len(best) == len(case["best_possible_bscore"]) == 1
    _assert_eq(best[0], case["best_possible_bscore"][0])
    assert len(worst) == len(case["worst_possible_bscore"]) == 1
    _assert_eq(worst[0], case["worst_possible_bscore"][0])
    _assert_eq(mi, case["min_improv"])
    _assert_eq(coef, case["complexity_coef"])
    _assert_eq(ssum, case["sum_bscore"])


# ---------------------------------------------------------------------------
# Out-of-scope gates: time_bscore / dispersion_pressure / ensemble scoring
# ---------------------------------------------------------------------------


def test_time_bscore_raises_early():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, time_bscore=True)
    tr = str_to_combo_tree("$1")
    with pytest.raises(NotImplementedError, match="time_bscore"):
        bsc(tr)


def test_dispersion_pressure_raises_early():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, dispersion_pressure=0.5)
    tr = str_to_combo_tree("$1")
    with pytest.raises(NotImplementedError, match="dispersion_pressure"):
        bsc(tr)


def test_dispersion_pressure_zero_does_not_raise():
    """Boundary: `dispersion_pressure` EXACTLY `0.0` (the default) must
    NOT trip the gate -- only a strictly positive value does.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, dispersion_pressure=0.0)
    tr = str_to_combo_tree("$1")
    bsc(tr)  # must not raise


def test_ensemble_scoring_raises():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct)
    with pytest.raises(NotImplementedError, match="Ensemble scoring not implemented"):
        bsc.call_ensemble(scored_combo_tree_set())


# ---------------------------------------------------------------------------
# The boolean-output guard
# ---------------------------------------------------------------------------


def test_rejects_non_boolean_ctable():
    from collections import Counter

    from pymetta.comboreduct.table.table import CTable, type_node
    from pymetta.comboreduct.type_checker.type_signature import TypeSignature

    ct = CTable("y", ["x1"])
    ct[(1.0,)] = Counter({1.0: 1.0})
    ct.set_signature(TypeSignature([type_node.contin_type], type_node.contin_type))
    with pytest.raises(AssertionError, match="must be boolean"):
        precision_bscore(ct)


# ---------------------------------------------------------------------------
# get_activation_penalty boundaries: -inf and NaN, with NO saturation clamp
# ---------------------------------------------------------------------------


def test_activation_penalty_zero_at_exact_boundaries():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, activation_pressure=1.0, min_activation=0.5, max_activation=0.9)
    assert bsc.get_activation_penalty(bsc.min_activation) == 0.0
    assert bsc.get_activation_penalty(bsc.max_activation) == 0.0
    assert bsc.get_activation_penalty(0.7) == 0.0


def test_activation_penalty_minus_infinity_at_zero_activation():
    """UNLIKE `discriminating_bscore.get_threshold_penalty`, this method
    has NO `0.99999999 < dst` saturation clamp -- `activation=0.0`
    against `min_activation=1.0` gives `dst=1.0` exactly, so
    `log(1.0-1.0) == log(0.0)`, which is `-inf` in C (not an exception)
    and would raise `ValueError` from Python's `math.log` if not handled
    explicitly.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, activation_pressure=2.0, min_activation=1.0, max_activation=1.0)
    assert bsc.get_activation_penalty(0.0) == -math.inf


def test_activation_penalty_nan_when_dst_exceeds_one():
    """`dst > 1.0` makes `log_arg = 1.0 - dst` NEGATIVE -- C's
    `log(negative)` is a domain error returning `NaN` (not an exception);
    Python's `math.log` raises `ValueError` for the same input.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, activation_pressure=1.0, min_activation=1.0, max_activation=1.0)
    # activation deeply negative relative to min_activation=1.0 makes
    # dst = 1.0 - activation/min_activation exceed 1.0.
    penalty = bsc.get_activation_penalty(-10.0)
    assert math.isnan(penalty)


def test_zero_activation_pressure_disables_the_penalty():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, activation_pressure=0.0)
    assert bsc.get_activation_penalty(0.0) == 0.0
    assert bsc.get_activation_penalty(-100.0) == 0.0


# ---------------------------------------------------------------------------
# Non-zero complexity penalty: penalized vs raw scores must differ
# ---------------------------------------------------------------------------


def test_set_complexity_coef_makes_penalized_score_differ_from_raw():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    # activation_pressure=0.0 avoids the activation penalty's own -inf
    # boundary (test_activation_penalty_minus_infinity_at_zero_activation
    # covers that separately) so this test isolates the complexity
    # coefficient's own effect on a finite raw score.
    bsc = precision_bscore(ct, activation_pressure=0.0)
    tr = str_to_combo_tree("and($1 $2)")
    bs = bsc(tr)
    raw = bsc.sum_bscore(bs)

    bsc.set_complexity_coef(3.5)
    coef = bsc.get_complexity_coef()
    assert coef > 0.0
    complexity = bsc.get_complexity(tr)
    assert complexity > 0
    penalized = raw - coef * complexity
    assert penalized != raw


def test_set_complexity_coef_two_argument_singularities():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct)
    bsc.set_complexity_coef(2, 0.0)
    assert bsc.get_complexity_coef() == 0.0
    bsc.set_complexity_coef(2, 0.5)
    assert bsc.get_complexity_coef() == 0.0
    bsc.set_complexity_coef(2, 0.1)
    assert bsc.get_complexity_coef() > 0.0


# ---------------------------------------------------------------------------
# precision_conj_bscore -- no time/boosting dependency, fully portable
# ---------------------------------------------------------------------------


def test_precision_conj_bscore_best_possible_and_min_improv():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    assert bsc.best_possible_bscore() == [1.0, 0.0]
    assert bsc.min_improv() == 0.0


def test_precision_conj_bscore_call_counts_and_conjunctions():
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    tr = str_to_combo_tree("and($1 $2)")
    bs = bsc(tr)
    assert len(bs) == 2
    # A single top-level `and` contributes exactly 1 to conj_n -- penalty
    # = hardness * (-1 / (1 + 1)) = -0.5.
    assert bs[1] == pytest.approx(-0.5)


def test_precision_conj_bscore_zero_activation_gives_perfect_precision():
    """"No hits means perfect precision :) Yes, zero hits is common,
    early on." (precision_bscore.cc:798-800's own comment.)
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    tr = str_to_combo_tree("logical_false")
    bs = bsc(tr)
    assert bs[0] == 1.0
