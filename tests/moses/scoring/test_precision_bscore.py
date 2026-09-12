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
from pymetta.moses.moses.types import (
    _to_score_t,
    scored_combo_tree,
    scored_combo_tree_set,
)
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


def test_ensemble_scoring_on_an_unboosted_scorer_raises():
    """`OC_ASSERT(_return_weighted_score, "Unexpected call to ensemble
    scorer!")` (precision_bscore.cc:373). Before the weighted path was
    ported this raised `NotImplementedError` from the inherited
    `bscore_base.call_ensemble`; now the class overrides it and the guard
    is classic's own assertion, which only lets a BOOSTED scorer through.
    """
    table = loadTable(str(_PRECISION_CSV), "target")
    ct = table.compressed()
    bsc = precision_bscore(ct)
    assert bsc._return_weighted_score is False
    with pytest.raises(AssertionError, match="Unexpected call to ensemble scorer"):
        bsc.call_ensemble(scored_combo_tree_set())


# ---------------------------------------------------------------------------
# The weighted (boosting) score path: do_score's `_return_weighted_score`
# branch, call_ensemble, exact_selection, bias_selection.
#
# Every table below is built so that the four compressed rows land in a
# KNOWN order and carry a KNOWN true/false split, because both the row
# order and the per-row `sum_outputs` decide every number asserted here:
#
#   key      counter        sum_outputs = (T-F)/2   total_count
#   (T, T)   {T: 3, F: 1}   +1.0                    4
#   (T, F)   {T: 1, F: 1}    0.0                    2
#   (F, T)   {F: 2}         -1.0                    2
#   (F, F)   {F: 1}         -0.5                    1
#
# `uncompressed_size()` is 9, so `_ctable_weight == 9.0`.
# ---------------------------------------------------------------------------

_WEIGHTED_CSV = "target,x1,x2\n1,1,1\n1,1,1\n1,1,1\n0,1,1\n1,1,0\n0,1,0\n0,0,1\n0,0,1\n0,0,0\n"


@pytest.fixture
def weighted_ctable(tmp_path):
    csv = tmp_path / "weighted.csv"
    csv.write_text(_WEIGHTED_CSV)
    ct = loadTable(str(csv), "target").compressed()
    # Pin the shape every assertion below depends on.
    assert len(ct) == 4
    assert ct.uncompressed_size() == 9.0
    return ct


def _boosted(ct, **kwargs):
    """A scorer in exactly the state `table-problems.cc:385` leaves it in
    for `--boost=1`: constructed, then `use_weighted_scores()`, then
    `ensemble::ensemble` (ensemble.cc:47) calls `reset_weights()`.
    """
    bsc = precision_bscore(ct, **kwargs)
    bsc.use_weighted_scores()
    bsc.reset_weights()
    return bsc


def test_reset_weights_only_populates_once_weighted(weighted_ctable):
    """`precision_bscore::reset_weights` (precision_bscore.cc:698) leaves
    `_weights` EMPTY while `_return_weighted_score` is false, and that is
    what makes `sum_bscore`'s unweighted branch keep working before
    boosting is switched on.
    """
    plain = precision_bscore(weighted_ctable)
    assert plain._weights == []
    assert plain.wnorm == 1.0
    boosted = _boosted(weighted_ctable)
    assert boosted._weights == [1.0, 1.0, 1.0, 1.0]


def test_weighted_do_score_normalises_by_the_WEIGHTED_activity(weighted_ctable):
    """The whole point of precision_bscore.cc:283-289.

    Tree `$1` selects rows (T,T) and (T,F): `active = 4 + 2 = 6`, so the
    flat normalisation is `1/6`.  With weights `[2.0, 0.5, 1.0, 1.0]` the
    boosted normalisation is `1 / (2*4 + 0.5*2) = 1/9` -- a different
    number reached through a different denominator, which is what a
    mutation back to `1.0 / active` cannot fake.
    """
    tr = str_to_combo_tree("$1")

    flat = precision_bscore(weighted_ctable)(tr)
    assert flat[:4] == [_to_score_t(1.0 / 6.0), 0.0, 0.0, 0.0]

    bsc = _boosted(weighted_ctable)
    bsc._weights = [2.0, 0.5, 1.0, 1.0]
    weighted = bsc(tr)
    assert weighted[:4] == [_to_score_t(1.0 / 9.0), 0.0, 0.0, 0.0]
    assert weighted[4] == 0.5


def test_weighted_do_score_omits_the_activation_penalty(weighted_ctable):
    """ "the boosted scorer only wants those selectors that are exactly
    correct, even if they have a terrible activation" (precision_bscore.cc
    :312-317).  `min_activation=0.9` makes the penalty non-zero, so the
    omission is visible in the VALUES and not only in the length.
    """
    tr = str_to_combo_tree("$1")

    flat = precision_bscore(weighted_ctable, min_activation=0.9)(tr)
    assert len(flat) == 6
    assert flat[5] < 0.0  # activation 6/9 is below min_activation

    bsc = _boosted(weighted_ctable, min_activation=0.9)
    weighted = bsc(tr)
    assert len(weighted) == 5
    assert weighted[4] == 0.5


def test_uniform_weights_reproduce_the_flat_normalisation(weighted_ctable):
    """`reset_weights()` sets every weight to 1.0, and then `wactive ==
    active` -- so this case CANNOT tell the weighted branch from the flat
    one.  It is pinned deliberately, as the boundary condition the other
    tests are measured against, and is never the only coverage.
    """
    tr = str_to_combo_tree("$1")
    bsc = _boosted(weighted_ctable)
    assert bsc(tr)[:4] == precision_bscore(weighted_ctable)(tr)[:4]


def test_update_weights_drops_rewards_at_or_above_one(weighted_ctable):
    """`precision_bscore::update_weights` (precision_bscore.cc:707) is NOT
    the base class's plain rescale. `if (rew[i] < 1.0) _weights[i] *=
    rew[i];` throws the `expalpha` half of `ensemble::add_expert`'s reward
    vector away entirely -- "whatever we do, we don't want to let later
    updates push them up" (precision_bscore.cc:713-716) -- and only then
    renormalises to `sum == _size`.

    The exact values are asserted, because the SHAPE is the same either
    way: dropping the guard still leaves weight 0 smallest, the other
    three equal, and the sum at 4.0. Only the numbers separate them
    (4/3.5 vs 4/6.5 as the normaliser).
    """
    bsc = _boosted(weighted_ctable)
    # expalpha = 2.0: selected rows get 1/2, unselected rows get 2.0.
    bsc.update_weights([0.5, 2.0, 2.0, 2.0])
    assert bsc._weights == [0.5 * 4 / 3.5, 4 / 3.5, 4 / 3.5, 4 / 3.5]
    assert sum(bsc._weights) == pytest.approx(4.0)
    assert bsc.wnorm == pytest.approx(3.5 / 4.0)


def test_update_weights_keeps_knocked_down_rows_down(weighted_ctable):
    """Two rounds, which is the only way the `wnorm` bookkeeping can
    matter at all -- after `reset_weights` it is 1.0, so a single call
    cannot tell it apart from nothing.

    MEASURED, and recorded because it is a surprise: `_weights[i] *=
    wnorm` (precision_bscore.cc:722) is ALGEBRAICALLY A NO-OP. It scales
    every weight by the same factor, and the renormalisation two lines
    later divides that same factor straight back out. Removing it leaves
    the weights unchanged to ~1e-15 after eight successive updates over
    200 random reward vectors -- i.e. below anything this port can
    assert. So there is no test here that kills that mutant, and the
    reason is "no trap", not "no coverage".

    What this test DOES pin is the real effect of two rounds: row 0 is
    knocked down twice and ends further below the others than one round
    leaves it.
    """
    bsc = _boosted(weighted_ctable)
    bsc.update_weights([0.5, 2.0, 2.0, 2.0])
    after_one = list(bsc._weights)
    bsc.update_weights([0.5, 2.0, 2.0, 2.0])
    assert bsc._weights == [0.25 * 4 / 3.25, 4 / 3.25, 4 / 3.25, 4 / 3.25]
    assert bsc._weights[0] < after_one[0]
    assert sum(bsc._weights) == pytest.approx(4.0)


def test_update_weights_rejects_a_wrong_sized_reward(weighted_ctable):
    bsc = _boosted(weighted_ctable)
    with pytest.raises(AssertionError, match="Unexpected size of weight array"):
        bsc.update_weights([1.0, 1.0])


def test_update_weights_rejects_an_unboosted_scorer(weighted_ctable):
    bsc = precision_bscore(weighted_ctable)
    with pytest.raises(AssertionError, match="Unexpected use of weights"):
        bsc.update_weights([1.0, 1.0, 1.0, 1.0])


# ---- exact_selection ------------------------------------------------------


def test_exact_selection_of_one_member_is_that_member_s_FLAT_bscore(weighted_ctable):
    """`call_ensemble` toggles `_return_weighted_score` off for the whole
    call (precision_bscore.cc:374-378), so a one-member ensemble must
    reproduce the UNWEIGHTED per-tree bscore -- activation penalty and
    all -- even though the scorer is boosted and carries lop-sided
    weights.  Dropping the toggle changes both the normalisation AND the
    length.
    """
    tr = str_to_combo_tree("$1")
    bsc = _boosted(weighted_ctable)
    bsc._weights = [2.0, 0.5, 1.0, 1.0]

    ens = scored_combo_tree_set()
    ens.insert(scored_combo_tree(tr))

    got = bsc.call_ensemble(ens)
    assert got == precision_bscore(weighted_ctable)(tr)
    assert len(got) == 6
    assert bsc._return_weighted_score is True  # restored


def test_exact_selection_of_two_members_is_the_OR_of_them(weighted_ctable):
    """`tr.set_head(logical_or); append_child(...)` (precision_bscore.cc
    :417-424).  `$1` alone selects (T,T),(T,F); `$2` alone selects
    (T,T),(F,T); the union adds the (F,T) row, whose `sum_outputs` is
    NEGATIVE -- so an `and` there, or a scorer that only looked at the
    first member, gives a visibly different bscore.
    """
    bsc = _boosted(weighted_ctable)
    ens = scored_combo_tree_set()
    ens.insert(scored_combo_tree(str_to_combo_tree("$1")))
    ens.insert(scored_combo_tree(str_to_combo_tree("$2")))

    got = bsc.call_ensemble(ens)
    assert got == precision_bscore(weighted_ctable)(str_to_combo_tree("or($1 $2)"))
    assert got[:4] == [0.125, 0.0, -0.125, 0.0]
    assert got != precision_bscore(weighted_ctable)(str_to_combo_tree("and($1 $2)"))


def test_exact_selection_is_insensitive_to_ensemble_iteration_order(weighted_ctable):
    """The `or` tree is built in `scored_combo_tree_set` order, which
    neither language specifies -- but `logical_or` is commutative and the
    tree is only interpreted, never printed, so the SCORE is order-free.
    This is what lets the CLI-level tests assert a score across an
    ordering classic does not fix.
    """
    bsc = _boosted(weighted_ctable)
    a = scored_combo_tree(str_to_combo_tree("$1"))
    b = scored_combo_tree(str_to_combo_tree("$2"))

    fwd = scored_combo_tree_set()
    fwd.insert(a)
    fwd.insert(b)
    rev = scored_combo_tree_set()
    rev.insert(b)
    rev.insert(a)

    assert bsc.call_ensemble(fwd) == bsc.call_ensemble(rev)


def test_get_error_is_one_minus_precision_and_leaves_the_flag_set(weighted_ctable):
    """`0.5 - sum(bs[:_size])` (precision_bscore.cc:512-517).  `$1` has
    precision `sao/active + 0.5 = 1/6 + 0.5`, so the error is `1/3`.
    `ensemble::add_expert` admits a tree only when this is ~0, which is
    why the toggle inside `get_error` has to disable weighting.
    """
    bsc = _boosted(weighted_ctable)
    bsc._weights = [2.0, 0.5, 1.0, 1.0]
    err = bsc.get_error(str_to_combo_tree("$1"))
    assert err == _to_score_t(0.5 - _to_score_t(1.0 / 6.0))
    assert bsc._return_weighted_score is True


def test_get_error_refuses_inexact_experts(weighted_ctable):
    bsc = _boosted(weighted_ctable, exact_experts=False)
    with pytest.raises(AssertionError, match="inexact experts not yet supported"):
        bsc.get_error(str_to_combo_tree("$1"))


# ---- bias_selection -------------------------------------------------------
#
# Unreachable from the CLI (`get_error` asserts first -- see the module
# docstring of precision_bscore.py), so it is driven directly here.
#
# Tree A = `and(!$1 $2)` selects ONLY row (F,T), whose sum_outputs is -1.0
#          -- so A's weight alone becomes the bias.
# Tree B = `and($1 $2)`  selects ONLY row (T,T), whose sum_outputs is +1.0.
# Every other row has hypoth == 0.0 and can never clear a positive bias.


def _bias_ensemble(ct, *, bias_scale, wa, wb, **kwargs):
    bsc = _boosted(ct, exact_experts=False, bias_scale=bias_scale, **kwargs)
    a = scored_combo_tree(str_to_combo_tree("and(!$1 $2)"))
    a.set_weight(wa)
    b = scored_combo_tree(str_to_combo_tree("and($1 $2)"))
    b.set_weight(wb)
    ens = scored_combo_tree_set()
    ens.insert(a)
    ens.insert(b)
    return bsc, ens


def test_bias_selection_is_reached_only_for_inexact_experts(weighted_ctable):
    bsc, ens = _bias_ensemble(weighted_ctable, bias_scale=0.5, wa=1.0, wb=4.0)
    assert bsc.exact_experts is False
    got = bsc.call_ensemble(ens)
    # bias == 1.0 (row (F,T)); 0.5*1.0 clears both hypoth 4.0 and hypoth 1.0,
    # so rows (T,T) and (F,T) are both selected.
    assert got[:4] == [_to_score_t(1.0 / 6.0), 0.0, _to_score_t(-1.0 / 6.0), 0.0]
    assert bsc._return_weighted_score is True


def test_bias_scale_decides_which_rows_clear_the_bias(weighted_ctable):
    """`bias_scale * bias < hypoth[i++]` (precision_bscore.cc:496) -- the
    ONLY place `--boost-bias` is read.  Three regimes, three answers.
    """
    lo, ens = _bias_ensemble(weighted_ctable, bias_scale=0.5, wa=1.0, wb=4.0)
    mid, ens_mid = _bias_ensemble(weighted_ctable, bias_scale=2.0, wa=1.0, wb=4.0)
    hi, ens_hi = _bias_ensemble(weighted_ctable, bias_scale=5.0, wa=1.0, wb=4.0)

    # 0.5: both the +1.0 row and the -1.0 row are selected.
    assert lo.call_ensemble(ens)[:4] == [
        _to_score_t(1.0 / 6.0),
        0.0,
        _to_score_t(-1.0 / 6.0),
        0.0,
    ]
    # 2.0: only the (T,T) row clears 2.0 -- precision is perfect.
    assert mid.call_ensemble(ens_mid)[:4] == [0.25, 0.0, 0.0, 0.0]
    # 5.0: nothing clears it; `active == 0`, so the 0.5 offset collapses to
    # 0.0 and the activation penalty is -inf.
    out = hi.call_ensemble(ens_hi)
    assert out[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert out[5] == -math.inf


def test_bias_comes_only_from_rows_that_should_NOT_be_selected(weighted_ctable):
    """`if (0.0 > sumo and bias < hypoth[i]) bias = hypoth[i];`
    (precision_bscore.cc:487).  Drop the `0.0 > sumo` guard and the bias
    becomes 4.0 (the (T,T) row's own weight) instead of 1.0, which at
    `bias_scale=0.5` loses the (F,T) row -- a different bscore.
    """
    bsc, ens = _bias_ensemble(weighted_ctable, bias_scale=0.5, wa=1.0, wb=4.0)
    got = bsc.call_ensemble(ens)
    assert got[2] == _to_score_t(-1.0 / 6.0), "the negative row must be selected"


def test_bias_selection_narrows_the_ensemble_weight_to_float32(weighted_ctable):
    """`score_t trweight = sct.get_weight();` (precision_bscore.cc:459)
    rounds the `double` weight to binary32 BEFORE it reaches `hypoth`,
    which is itself a `std::vector<double>`.  So this is the one place
    where the narrowing is the ONLY rounding in play, and it is
    order-independent: each row here is claimed by exactly one tree.

    `0.1` and `0.100000001` are DISTINCT doubles that share one float32
    value, so after narrowing `bias == hypoth[(T,T)]` exactly and the
    strict `<` rejects the row; without narrowing `0.1 < 0.100000001` and
    the row is selected.  The two answers are maximally far apart: nothing
    selected (activation penalty -inf) versus perfect precision.
    """
    bsc, ens = _bias_ensemble(weighted_ctable, bias_scale=1.0, wa=0.1, wb=0.100000001)
    got = bsc.call_ensemble(ens)
    assert got[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert got[5] == -math.inf


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


def test_boosted_time_bscore_refuses_to_guess_at_classic_s_out_of_bounds_read():
    """`--time-bscore=1 --boost=1` is the one reachable combination where
    classic's weighted normalisation indexes past the end of its own
    activity vector: the loop runs to `_size` (4 compressed rows here)
    while `ac` was filled per TIMESTAMP (3 dates). Classic exits 0 and
    prints a reproducible answer computed from whatever follows that
    3-element vector; this port raises instead, naming both sizes. See the
    DIVERGENCE marker in `do_score` for the measurement.

    The guard is deliberately `len(ac) < _size` and not "time_bscore is
    on": a table with at least as many dates as compressed rows is
    in-bounds, defined (if odd) in C++, and must keep working.
    """
    ct = _timed_ctable()
    assert len(ct) == 4
    assert len(ct.get_timestamps()) == 3
    bsc = precision_bscore(ct, time_bscore=True)
    bsc.use_weighted_scores()
    bsc.reset_weights()
    with pytest.raises(IndexError, match="reads 4 activity entries but time_bscore"):
        bsc(str_to_combo_tree("$1"))


def test_boosted_time_bscore_is_fine_when_there_are_enough_timestamps():
    """The other side of the same guard: `timestamp_bool_months.csv` has
    more distinct dates than compressed rows, so classic's loop is
    in-bounds and this port must reproduce it rather than refuse.
    """
    ct = _timed_ctable("timestamp_bool_months.csv")
    assert len(ct.get_timestamps()) >= len(ct)
    bsc = precision_bscore(ct, time_bscore=True)
    bsc.use_weighted_scores()
    bsc.reset_weights()
    got = bsc(str_to_combo_tree("$1"))
    # One component per date, plus the +0.5 offset and NO activation
    # penalty (the boosted path drops it).
    assert len(got) == len(ct.get_timestamps()) + 1
    assert got[-1] == 0.5
