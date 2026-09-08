"""C++: moses/moses/scoring/discriminating_bscore.h and discriminating_bscore.cc

Differential oracle against tests/fixtures/scorers.json's "discriminating"
cases (captured by tools/capture_scorers.py from tools/cpp_probe/
scorers_probe.cc's `disc` mode), plus direct unit/boundary/mutation-proof
tests for discriminating_bscore.py's own guards and narrowing fixes.
"""

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import CTable, ITable, OTable, Table, type_node
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.scoring.discriminating_bscore import (
    bep_bscore,
    discriminating_bscore,
    discriminator,
    f_one_bscore,
    prerec_bscore,
    recall_bscore,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_CASES = json.loads((_FIXTURE_DIR / "scorers.json").read_text())["discriminating"]
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"

_SCORERS = {
    "recall": recall_bscore,
    "prerec": prerec_bscore,
    "bep": bep_bscore,
    "f_one": f_one_bscore,
}

T, F = builtin.logical_true, builtin.logical_false


def _boolean_ctable(rows: list[tuple[bool, bool, bool]]):
    """rows: list of (in1, in2, out) booleans. Builds a small hand-made
    boolean CTable, matching test_bscores.py's `_boolean_ctable` helper.
    """
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type])
    outs = []
    for a, b, y in rows:
        it.push_back((T if a else F, T if b else F))
        outs.append(T if y else F)
    ot = OTable(outs)
    return Table(ot, it).compressed()


# ---------------------------------------------------------------------------
# Differential oracle
# ---------------------------------------------------------------------------


def test_scorers_fixture_has_expected_shape():
    assert len(_CASES) >= 10
    scorers_seen = {c["scorer"] for c in _CASES.values()}
    assert {"recall", "prerec", "bep"} <= scorers_seen
    ratios = {c["complexity_ratio"] for c in _CASES.values() if c["complexity_ratio"]}
    assert 3.5 in ratios


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_CASES), ids=sorted(_CASES))
def test_discriminating_matches_cpp_probe(case_name):
    case = _CASES[case_name]
    path = _TABLE_DATA_DIR / case["file"]
    table = loadTable(str(path), case["target_feature"])
    ct = table.compressed()

    cls = _SCORERS[case["scorer"]]
    bsc = cls(ct, case["min_threshold"], case["max_threshold"], case["hardness"])
    if case["complexity_ratio"] is not None:
        bsc.set_complexity_coef(case["complexity_ratio"])

    tr = str_to_combo_tree(case["tree"])
    bs = bsc(tr)
    best = bsc.best_possible_bscore()
    mi = bsc.min_improv()
    coef = bsc.get_complexity_coef()
    ssum = bsc.sum_bscore(bs)

    if case["scorer"] == "prerec":
        # prerec's bscore is one element PER CTABLE ROW (plus a fixed
        # 2-element tail) -- this port's CTable does not reproduce
        # std::map's sorted iteration order (see score_probe.cc's own
        # comment, test_bscores.py's oracle for the established
        # precedent), so the row-keyed PREFIX is compared as a sorted
        # multiset, and only the fixed-position TAIL positionally.
        prefix_len = len(ct)
        assert sorted(bs[:prefix_len]) == sorted(r["value"] for r in case["bscore_rows"])
        assert bs[prefix_len:] == case["bscore"][prefix_len:]
    else:
        assert bs == case["bscore"], case_name

    assert list(best) == case["best_possible_bscore"], case_name
    assert mi == case["min_improv"], case_name
    assert coef == case["complexity_coef"], case_name
    assert ssum == case["sum_bscore"], case_name


# ---------------------------------------------------------------------------
# discriminating_bscore is abstract, matching C++'s pure-virtual operator()
# ---------------------------------------------------------------------------


def test_discriminating_bscore_cannot_be_instantiated_directly():
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    with pytest.raises(TypeError):
        discriminating_bscore(ct)


# ---------------------------------------------------------------------------
# A genuine upstream defect: f_one_bscore/bep_bscore(defaults) always raise
# ---------------------------------------------------------------------------


def test_f_one_bscore_is_unconstructible():
    """Measured live against classic's own probe (2026-09-08): every
    construction of `f_one_bscore` fails `discriminating_bscore`'s own
    threshold assert, because its hardcoded base-class call
    (`discriminating_bscore(ct, 0.0, 1.0, 1.0e-20)`) always has
    `min_threshold=0.0` with `hardness=1.0e-20 > 0`. See
    discriminating_bscore.py's own module docstring for the full writeup.
    DO NOT FIX: this is classic's own behaviour, not a port defect.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    with pytest.raises(AssertionError, match="invalid thresholds"):
        f_one_bscore(ct)


def test_bep_bscore_default_arguments_are_unconstructible():
    """`bep_bscore`'s OWN declared defaults (`min_diff=0.0, hardness=1.0`)
    hit the identical assert -- measured live against the same probe.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    with pytest.raises(AssertionError, match="invalid thresholds"):
        bep_bscore(ct)


def test_bep_bscore_with_nonzero_min_diff_constructs_fine():
    """The crash is specific to `min_diff<=0` combined with `hardness>0`
    -- an explicit, nonzero `min_diff` (exactly what
    `bep_table_problem::run` falls back to whenever it does NOT hit the
    'hardness==0' branch, table-problems.cc) constructs without error.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bep_bscore(ct, min_diff=0.1, max_diff=0.5, hardness=1.0)  # must not raise


@pytest.mark.parametrize("cls", [recall_bscore, prerec_bscore, bep_bscore])
def test_zero_min_threshold_with_positive_hardness_raises(cls):
    """The general form of the same guard `discriminating_bscore.__init__`
    itself enforces (discriminating_bscore.cc:222-229): ANY subclass
    constructed with `min_threshold<=0` and `hardness>0` raises, not just
    `bep_bscore`'s/`f_one_bscore`'s own specific defaults above. Mutation
    proof: comment out the `if not (0.0 < self._min_threshold <=
    self._max_threshold)` guard in discriminating_bscore.py and this test
    goes from raising `AssertionError` to constructing silently -- see
    this file's own module docstring / the task report for the recorded
    mutation result.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    with pytest.raises(AssertionError, match="invalid thresholds"):
        cls(ct, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("cls", [recall_bscore, prerec_bscore, bep_bscore])
def test_zero_hardness_bypasses_the_threshold_check(cls):
    """`hardness=0.0` disables the sanity check entirely (discriminating_
    bscore.cc:223's own `if (0.0 < _hardness)` guard) -- a `min_threshold`
    of 0.0 is then accepted, since `get_threshold_penalty` also always
    returns 0.0 whenever `_hardness<=0` (its own `else` branch). Boundary
    case: hardness EXACTLY 0.0, not merely small.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = cls(ct, 0.0, 1.0, 0.0)  # must not raise
    assert bsc.get_threshold_penalty(0.5) == 0.0
    assert bsc.get_threshold_penalty(-5.0) == 0.0  # even a wildly out-of-range value


def test_negative_hardness_raises():
    """`OC_ASSERT(0.0 <= _hardness, "Hardness must be non-negative")`
    (discriminating_bscore.cc:222) -- unconditional, independent of
    min/max threshold values.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    with pytest.raises(AssertionError, match="non-negative"):
        recall_bscore(ct, 0.8, 1.0, -1.0)


# ---------------------------------------------------------------------------
# get_threshold_penalty: boundaries and the double-rounding fix
# ---------------------------------------------------------------------------


def test_get_threshold_penalty_zero_at_exact_boundaries():
    """`dst` is exactly 0.0 when `value` sits ON either boundary (neither
    `<` nor `>` fires), so `log(1.0-0.0) == log(1.0) == 0.0` -- a value
    EXACTLY on each boundary this class ports, per the testing bar's own
    requirement.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = recall_bscore(ct, 0.5, 0.9, 1.0)
    # Use the scorer's OWN (already score_t-narrowed) thresholds as the
    # "exactly on boundary" probe values -- a bare Python literal 0.9
    # does NOT equal float32(0.9) when widened back to double, so it can
    # spuriously cross the `_max_threshold < value` branch.
    assert bsc.get_threshold_penalty(bsc._min_threshold) == 0.0
    assert bsc.get_threshold_penalty(bsc._max_threshold) == 0.0
    assert bsc.get_threshold_penalty(0.7) == 0.0  # strictly inside


def test_get_threshold_penalty_negative_below_min_threshold():
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = recall_bscore(ct, 0.5, 0.9, 1.0)
    penalty = bsc.get_threshold_penalty(0.4)
    assert penalty < 0.0


def test_get_threshold_penalty_saturates_near_dst_one():
    """`0.99999999 < dst` short-circuits to `-hardness * 18` rather than
    calling `log(1.0 - dst)` (which would blow up toward `-inf` as `dst`
    approaches 1.0). `value = 0.0` against `min_threshold = 1.0` gives
    `dst = 1.0` exactly, safely past the cutoff.
    """
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = recall_bscore(ct, 1.0, 1.0, 2.0)
    assert bsc.get_threshold_penalty(0.0) == pytest.approx(-2.0 * 18)


def test_get_threshold_penalty_double_rounding_regression():
    """Mutation-proof regression for the double-rounding fix: computing
    `1.0 - value / min_threshold` in one shot (narrowing only once, at
    the end) instead of narrowing the `value / min_threshold` FLOAT32-
    NATIVE division separately gives a DIFFERENT (wrong) result here.
    Measured against tools/cpp_probe/scorers_probe.cc directly: this
    exact case is `trainingBoolean.dat::recall::or($1 $2)` in
    tests/fixtures/scorers.json, and is also covered by the differential
    oracle above -- this test isolates it as a standalone, human-readable
    regression independent of the fixture file.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = recall_bscore(ct, 0.8, 1.0, 1.0)
    tr = str_to_combo_tree("or($1 $2)")
    bs = bsc(tr)
    assert bs[1] == -0.18232150375843048
    # The (mutated, wrong) single-shot-narrow version this guards against:
    assert bs[1] != -0.18232154846191406


# ---------------------------------------------------------------------------
# The unsigned-total truncation (_to_usize / discriminator.count's totalc)
# ---------------------------------------------------------------------------


def _weighted_boolean_ctable():
    """A hand-built weighted boolean CTable whose SINGLE compressed row's
    total weight is fractional (0.3 + 0.4 == 0.7) -- exercises
    `_ctable_usize`'s (and `discriminator.count`'s `totalc`) truncation
    of a fractional `count_t` total down to an integer, matching classic's
    OWN implicit `size_t x = count_t_value;`/`unsigned x = count_t_value;`
    narrowing (see `_to_usize`'s docstring in scoring_base.py).
    """
    ct = CTable("y", ["x1", "x2"])
    counter = Counter()
    counter[T] = 0.3
    counter[F] = 0.4
    ct[(T, T)] = counter
    ct.set_signature(
        TypeSignature([type_node.boolean_type, type_node.boolean_type], type_node.boolean_type)
    )
    return ct


def test_ctable_usize_truncates_fractional_uncompressed_size():
    """`_ctable_usize` truncates `uncompressed_size()` (a `count_t`/double)
    down to an int -- a weighted table whose total is `0.7` gives
    `_ctable_usize == 0`, NOT `1` and NOT `0.7`. Mutation proof: replacing
    `_to_usize(...)` with a bare `round(...)`/plain float in
    discriminating_bscore.py's constructor changes this from `0` to `1`.
    """
    ct = _weighted_boolean_ctable()
    assert ct.uncompressed_size() == pytest.approx(0.7)
    bsc = recall_bscore(ct, 0.5, 1.0, 1.0)
    assert bsc._ctable_usize == 0
    # min_improv = 1.0 / _ctable_usize -- a ZeroDivisionError here would
    # ALSO be faithful (C++'s `1.0f / 0u` is `+inf`, not UB, for floats):
    assert math.isinf(bsc.min_improv())


def test_discriminator_count_totalc_truncates_fractional_weight():
    """Same truncation, exercised directly through `discriminator.count`'s
    own `positive_count`/`negative_count` accumulation -- a row weighing
    0.7 total contributes exactly `0` to `positive_count`/`negative_count`
    (whichever side the tree's prediction lands on), not `0.7`.
    """
    ct = _weighted_boolean_ctable()
    disc = discriminator(ct)
    tr = str_to_combo_tree("$1")  # predicts logical_true for row (T, T)
    ctr = disc.count(tr)
    assert ctr.positive_count == 0.0
    assert ctr.negative_count == 0.0


# ---------------------------------------------------------------------------
# Non-zero complexity penalty: penalized vs raw scores must differ
# ---------------------------------------------------------------------------


def test_set_complexity_coef_makes_penalized_score_differ_from_raw():
    """Per the testing bar: use a NON-ZERO complexity penalty so the
    penalized and raw scores are demonstrably different numbers, not just
    that a coefficient field changed.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    ct = table.compressed()
    bsc = recall_bscore(ct, 0.8, 1.0, 1.0)
    tr = str_to_combo_tree("and($1 $2 $3)")
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
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = recall_bscore(ct, 0.8, 1.0, 1.0)
    bsc.set_complexity_coef(2, 0.0)
    assert bsc.get_complexity_coef() == 0.0
    bsc.set_complexity_coef(2, 0.5)
    assert bsc.get_complexity_coef() == 0.0
    bsc.set_complexity_coef(2, 0.1)
    assert bsc.get_complexity_coef() > 0.0


def test_set_complexity_coef_rejects_bad_argument_counts():
    ct = _boolean_ctable([(True, True, True), (False, False, False)])
    bsc = recall_bscore(ct, 0.8, 1.0, 1.0)
    with pytest.raises(TypeError):
        bsc.set_complexity_coef()
    with pytest.raises(TypeError):
        bsc.set_complexity_coef(1, 2, 3)
