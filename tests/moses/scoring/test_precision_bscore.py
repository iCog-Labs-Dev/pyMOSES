"""C++: moses/moses/scoring/precision_bscore.h and precision_bscore.cc

Differential oracle against:
* scoringUTest.cxxtest's own `test_precision_bscore_best_possible_bscore`
  (precision.csv, activation_pressure=1.0, min_activation=0.5,
  max_activation=1.0, expects `precision = 5.f/7.f`) -- the ONE live
  cxxtest coverage of precision_bscore that exists in classic.
* tests/fixtures/scorers.json's "precision" cases (captured by
  tools/capture_scorers.py from tools/cpp_probe/scorers_probe.cc's
  `precision` mode).

Plus direct unit/boundary/mutation-proof tests for `do_score`'s two
timestamp branches (`time_bscore` and `dispersion_pressure > 0.0`), the
still-out-of-scope ensemble gate, and the -inf/NaN handling in
get_activation_penalty.
"""

import json
import math
from datetime import date
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.moses.types import _to_score_t, scored_combo_tree_set
from pymetta.moses.scoring.precision_bscore import precision_bscore, precision_conj_bscore
from pymetta.moses.scoring.time_dispersion import TemporalGranularity

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
# The two timestamp branches of do_score (precision_bscore.cc:242-263)
# ---------------------------------------------------------------------------

_CLEAN_CSV = _TABLE_DATA_DIR / "timestamp_bool_clean.csv"


def _timed_ctable(name: str = "timestamp_bool_clean.csv"):
    """A CTable with REAL per-observation timestamps.

    `timestamp_bool_clean.csv` was built so that every compressed row's
    observations are ordered consistently under classic's own
    `TimedValue::operator<` (all `true` observations dated before all
    `false` ones), which is what makes classic's `std::map` well defined on
    it -- see `TimedCounter`'s class docstring (table.py). Rows agree on
    inputs and output and differ only in date, which is exactly what makes
    it discriminate.
    """
    table = loadTable(str(_TABLE_DATA_DIR / name), target_feature="y", timestamp_feature="when")
    return table.compressed()


def test_wrk_ctable_keeps_the_timestamps_the_loader_parsed():
    """`bscore_ctable_base.__init__` copies the CTable through
    `scoring_base._copy_ctable`, which used to rebuild each cell as a plain
    `collections.Counter` -- flattening the `TimedValue` keys and losing
    every timestamp. `do_score`'s two timestamp branches read `_wrk_ctable`,
    not `_orig_ctable`, so that copy is on the critical path.

    Mutation-checked: restoring `new[row] = Counter(counter)` in
    `_copy_ctable` makes this test and both branch tests below fail.
    """
    ct = _timed_ctable()
    bsc = precision_bscore(ct, dispersion_pressure=0.5)
    assert ct.get_timestamps() == {date(2012, 1, 3), date(2012, 1, 11), date(2012, 1, 19)}
    assert bsc._wrk_ctable.get_timestamps() == ct.get_timestamps()
    # _Hmax is log2(3) only because the copy kept all three classes.
    assert bsc._Hmax == _to_score_t(math.log2(3))


def test_time_bscore_spreads_the_bscore_over_timestamps_in_date_order():
    """`time_bscore` replaces the per-ROW components with one per DISTINCT
    TIMESTAMP, in `std::map<date, ...>` order (ascending), then keeps the
    trailing 0.5 and activation-penalty components
    (precision_bscore.cc:266-275).

    On `timestamp_bool_clean.csv` `or(!$i1 $i2)` selects rows (0,0), (0,1)
    and (1,1) -- 7 of the 9 observations, so `active == 7`. Per date, the
    (tp, fp) counts over SELECTED rows are 2012-01-03 -> (3, 0),
    2012-01-11 -> (2, 0), 2012-01-19 -> (0, 2), so the raw components are
    1.5, 1.0 and -1.0, each normalised by `iac == 1/7`.
    """
    ct = _timed_ctable()
    plain = precision_bscore(ct)
    timed = precision_bscore(ct, time_bscore=True)
    tr = str_to_combo_tree("or(!$1 $2)")

    bs_plain = plain(tr)
    bs_timed = timed(tr)

    # 4 compressed rows + 0.5 + activation penalty, vs 3 dates + the same 2.
    assert len(bs_plain) == 6
    assert len(bs_timed) == 5
    iac = _to_score_t(1.0 / 7.0)
    assert bs_timed[:3] == [
        _to_score_t(_to_score_t(1.5) * iac),
        _to_score_t(_to_score_t(1.0) * iac),
        _to_score_t(_to_score_t(-1.0) * iac),
    ]
    assert bs_timed[3] == 0.5
    assert bs_timed[4] == bs_plain[5]
    # The two must not sum to different totals: the time-spread bscore is a
    # regrouping of the same per-observation contributions.
    assert sum(bs_plain) == pytest.approx(sum(bs_timed), abs=1e-6)


def test_time_bscore_components_are_in_ascending_date_order_not_encounter_order():
    """`time2res` is a `std::map<TTable::value_type, ...>`
    (precision_bscore.cc:225), so its iteration -- and hence the order of
    the emitted bscore components -- is ascending by DATE, not by the order
    the rows happened to introduce each date.

    `timestamp_bool_desc.csv` exists for exactly this: its first compressed
    row lists 2012-03-09, then 2012-02-07, then 2012-01-05, so the
    encounter order is the REVERSE of the date order. Every other
    timestamped corpus here encounters dates ascending, which makes the two
    orders coincide -- and a mutation that emitted encounter order survived
    the whole suite until this test was added.

    With `true` every row is selected (9 observations, `active == 9`):
    2012-01-05 has 4 true and 0 false, 2012-02-07 has 2 and 0, 2012-03-09
    has 0 and 3 -- so ascending gives 2.0, 1.0, -1.5 over `iac == 1/9`,
    while encounter order would put -1.5 first.
    """
    ct = _timed_ctable("timestamp_bool_desc.csv")
    encountered = []
    for counter in ct.values():
        for tv in counter.timed_keys():
            if tv.timestamp not in encountered:
                encountered.append(tv.timestamp)
    assert encountered == [date(2012, 3, 9), date(2012, 2, 7), date(2012, 1, 5)], (
        "corpus no longer discriminates: encounter order must NOT be date order"
    )

    bs = precision_bscore(ct, time_bscore=True)(str_to_combo_tree("true"))
    iac = _to_score_t(1.0 / 9.0)
    assert bs[:3] == [
        _to_score_t(_to_score_t(2.0) * iac),
        _to_score_t(_to_score_t(1.0) * iac),
        _to_score_t(_to_score_t(-1.5) * iac),
    ]


def test_time_bscore_emits_a_component_for_an_unselected_row_s_timestamp():
    """`count_t count = selected ? tcv.second : 0.0;`
    (precision_bscore.cc:246) still WRITES the zero, so
    `time2res[timestamp]` gets a bucket for a date that no selected row
    touches. Dropping the `+= 0.0` would shorten the bscore.

    `logical_false` selects nothing at all, so EVERY bucket comes from the
    zero write -- there must still be one component per distinct date.
    """
    ct = _timed_ctable()
    bsc = precision_bscore(ct, time_bscore=True)
    bs = bsc(str_to_combo_tree("logical_false"))
    # 3 dates + the 0.0 (inactive) component + the activation penalty.
    assert len(bs) == 5
    assert bs[:3] == [0.0, 0.0, 0.0]


def test_time_bscore_negative_target_inverts_the_tp_fp_split():
    """`if (!positive) target = !target;` (precision_bscore.cc:245).

    With `positive=False` the per-date contribution is `0.5 * (fp - tp)` of
    the positive reading, so every component flips sign. Nothing at the CLI
    level distinguishes this on the committed corpora, so it is asserted
    here directly.
    """
    ct = _timed_ctable()
    tr = str_to_combo_tree("or(!$1 $2)")
    pos = precision_bscore(ct, time_bscore=True, positive=True)(tr)
    neg = precision_bscore(ct, time_bscore=True, positive=False)(tr)
    # Same activity (7 selected observations), opposite tp/fp labelling.
    iac = _to_score_t(1.0 / 7.0)
    assert pos[:3] == [
        _to_score_t(_to_score_t(1.5) * iac),
        _to_score_t(_to_score_t(1.0) * iac),
        _to_score_t(_to_score_t(-1.0) * iac),
    ]
    assert neg[:3] == [
        _to_score_t(_to_score_t(-1.5) * iac),
        _to_score_t(_to_score_t(-1.0) * iac),
        _to_score_t(_to_score_t(1.0) * iac),
    ]


def test_dispersion_pressure_appends_one_extra_component_after_the_activation_penalty():
    """`bs.push_back(dispersion_penalty)` (precision_bscore.cc:343) comes
    AFTER `bs.push_back(activation_penalty)` (:327), inside the same
    non-boosting block. Order matters: `behave_cscore` sums the vector, but
    diversity and `best_possible_bscore` compare it positionally.
    """
    ct = _timed_ctable()
    tr = str_to_combo_tree("or(!$1 $2)")
    plain = precision_bscore(ct, dispersion_pressure=0.0)(tr)
    disp = precision_bscore(ct, dispersion_pressure=0.5)(tr)
    assert len(disp) == len(plain) + 1
    assert disp[:-1] == plain
    assert disp[-1] == pytest.approx(0.029095500707626343, abs=0.0)


def test_dispersion_pressure_zero_appends_nothing():
    """Boundary: `if (_pressure > 0.0)` is STRICT, so exactly `0.0` (the
    default) leaves the bscore untouched -- it does not append a zero.
    """
    ct = _timed_ctable()
    tr = str_to_combo_tree("or(!$1 $2)")
    assert len(precision_bscore(ct, dispersion_pressure=0.0)(tr)) == 6


def test_dispersion_pressure_month_granularity_collapses_to_one_class():
    """`get_timestamp_class` at `month` maps every date in
    `timestamp_bool_clean.csv` onto 2012-01-01, so `_Hmax` is `log2(1) ==
    0`. Which degenerate answer follows depends on the candidate, and BOTH
    are live from the CLI:

    * `true` selects every row, so `ctable_res` holds only
      `logical_true` and `H` is 0 too -- `0.0f / 0.0f`, a NaN. Classic
      prints `nan`, not `-nan` (see `_powf`, time_dispersion.py), and that
      is the whole stdout of a real `-Hpre --time-dispersion-pressure=0.5
      --time-bscore-granularity=month` run on this file.
    * `or(!$i1 $i2)` leaves row (1,0) unselected, so the bucket holds both
      vertices, `H > 0`, and `H / 0.0f` is `+inf` -- penalty `-inf`.
    """
    ct = _timed_ctable()
    bsc = precision_bscore(ct, dispersion_pressure=0.5, granularity=TemporalGranularity.month)
    assert bsc._Hmax == 0.0

    all_selected = bsc(str_to_combo_tree("true"))
    assert math.isnan(all_selected[-1])
    assert math.copysign(1.0, all_selected[-1]) > 0.0, "classic's is a POSITIVE nan"

    partial = bsc(str_to_combo_tree("or(!$1 $2)"))
    assert partial[-1] == -math.inf


def test_month_granularity_over_several_months_keeps_several_classes():
    """The other half of the case above: `timestamp_bool_months.csv` spans
    three months, so `month` granularity keeps three classes and `_Hmax` is
    `log2(3)` -- proving the month arm buckets rather than collapsing
    everything unconditionally.
    """
    ct = _timed_ctable("timestamp_bool_months.csv")
    day = precision_bscore(ct, dispersion_pressure=0.5)
    month = precision_bscore(ct, dispersion_pressure=0.5, granularity=TemporalGranularity.month)
    # Four distinct dates, three distinct months.
    assert len(ct.get_timestamps()) == 4
    assert day._Hmax == _to_score_t(math.log2(4))
    assert month._Hmax == _to_score_t(math.log2(3))


def test_time_bscore_ignores_the_granularity_flag():
    """`time2res` is keyed on `tcv.first.timestamp` RAW
    (precision_bscore.cc:247), not on `get_timestamp_class` -- only the
    dispersion branch buckets. So `--time-bscore-granularity` cannot change
    a `--time-bscore` run, which is what classic does too (measured: its
    stdout is byte-identical with and without `=month`).
    """
    ct = _timed_ctable("timestamp_bool_months.csv")
    tr = str_to_combo_tree("or(!$1 $2)")
    day = precision_bscore(ct, time_bscore=True)(tr)
    month = precision_bscore(ct, time_bscore=True, granularity=TemporalGranularity.month)(tr)
    assert day == month
    # ... and it really is per-DATE, not per-month: four dates, four components.
    assert len(day) == 4 + 2


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


def test_activation_penalty_narrows_the_ratio_before_subtracting():
    """The under-activation branch narrows TWICE, and both narrowings matter.

    `precision_bscore.cc:627` is `dst = 1.0 - activation / min_activation;`.
    Both operands are `float`, so the division happens in float -- one
    narrowing -- then `1.0 - <float>` promotes to double and narrows again on
    assignment to `score_t dst`. Collapsing that into a single narrowing
    changes the result; the identical fix in `discriminating_bscore` is already
    proven necessary by a mutation there.

    Every other activation test here lands on a boundary returning 0.0, so none
    could distinguish the two forms. Found by an independent verification pass:
    collapsing the narrow left the whole suite green.

    At activation 2/7 with min_activation 0.287 (float32 0.28700000047683716)
    the two forms differ in the 8th significant digit -- the double-rounding
    signature.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct, activation_pressure=1.0, min_activation=0.287, max_activation=0.9)
    activation = 2.0 / 7.0

    faithful_dst = _to_score_t(1.0 - _to_score_t(activation / bsc.min_activation))
    collapsed_dst = _to_score_t(1.0 - activation / bsc.min_activation)
    assert faithful_dst != collapsed_dst, "inputs must discriminate the two forms"

    expected = _to_score_t(bsc.activation_pressure * _to_score_t(math.log(1.0 - faithful_dst)))
    assert bsc.get_activation_penalty(activation) == expected


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


def test_precision_conj_bscore_precision_term_is_fractional_not_degenerate():
    """[Important 2, branch review] The precision term (`bs[0]`) is
    computed by exactly ONE existing case in this module,
    `test_precision_conj_bscore_zero_activation_gives_perfect_precision`
    (`bs[0] == 1.0` for `logical_false`, zero activation -- the "no hits
    means perfect precision" branch). `test_precision_conj_bscore_call_
    counts_and_conjunctions` above never asserts `bs[0]` at all, and on
    `precision.csv`, `and($1 $2)` -- the tree it happens to use -- has
    ZERO activation too (no row has both `v1` and `v2` set), so `bs[0]`
    there is ALSO the degenerate `1.0`. Every `bs[0]` this module observes
    is therefore exactly `1.0`, and a mutant that forces `bs[0] = 0.123456`
    whenever the real value is not `1.0` passes the whole suite.

    Derived independently from `precision.csv`'s 10 rows (compressed:
    `(v1,v2)=(0,0)` -> target counts `{0:2, 1:2}`; `(1,0)` -> `{0:3}`;
    `(0,1)` -> `{1:3}`): `and(!$1 !$2)` (true only on `(v1,v2)=(0,0)`) has
    active = 4, target-true count = 2, precision = 2/4 = 0.5 -- a genuine
    fraction, not 0.0 or 1.0, that the degenerate-precision mutant cannot
    reproduce by construction. Cross-checked directly against a live
    `precision_conj_bscore` call, not hand-derived alone: `$1` (true only
    on `(1,0)`, target-true count 0 of 3) independently gives `bs[0] ==
    0.0`, and `$2` (true only on `(0,1)`, target-true count 3 of 3) gives
    `bs[0] == 1.0` -- consistent with the reviewer's own derivation.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    bs = bsc(str_to_combo_tree("and(!$1 !$2)"))
    assert bs[0] == pytest.approx(0.5)


def test_precision_conj_bscore_counts_nested_and_siblings_not_just_the_head():
    """`conj_n` counts the head AND each immediate `and` child.

    `precision_bscore.cc:806-814` increments once for `*head == logical_and`
    and once more per immediate sibling that is also `logical_and`. The test
    above uses `and($1 $2)`, whose only `and` is the head -- so it cannot tell
    the faithful code from one that drops the sibling loop entirely. Found by
    an independent verification pass: deleting that loop left the suite green.

    `and(and($1 $2) $3)` has an `and` head and one `and` child, so conj_n = 2
    and the penalty is `-1 / (1 + 2)`. Without the sibling loop conj_n would be
    1 and the penalty `-0.5`.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    bs = bsc(str_to_combo_tree("and(and($1 $2) $3)"))
    assert len(bs) == 2
    assert bs[1] == pytest.approx(-1.0 / 3.0)


def test_precision_conj_bscore_zero_activation_gives_perfect_precision():
    """ "No hits means perfect precision :) Yes, zero hits is common,
    early on." (precision_bscore.cc:798-800's own comment.)
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_conj_bscore(ct, hardness=1.0)
    tr = str_to_combo_tree("logical_false")
    bs = bsc(tr)
    assert bs[0] == 1.0
