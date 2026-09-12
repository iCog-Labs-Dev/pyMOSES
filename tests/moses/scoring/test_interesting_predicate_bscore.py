"""Oracle tests for `interesting_predicate_bscore` (bscores.h:553) and the
two cogutil statistics it is MOSES's only caller of.

Every expected value here was MEASURED from classic by
`tools/capture_ip_bscore.py` (probe: `tools/cpp_probe/ip_bscore_probe.cc`)
into `tests/fixtures/ip_bscore.json`. Nothing is hand-computed and nothing
uses `pytest.approx`: comparison is BIT-EXACT, through `struct.pack`, so a
NaN's sign bit and a signed zero are both part of the assertion. Three of
the scorer's five bscore elements are a NaN on a reachable input and their
signs differ from each other, so an `==` comparison (where `nan != nan`)
or an `approx` one (where every NaN is equal) would both be vacuous
exactly where the port is hardest to get right.

Layers, matching the fixture's three groups:

* ``stats`` -- `ranking` (ranking.h:40), `MannWhitneyU` (MannWhitneyU.h:44),
  `standardizedMannWhitneyU` (:64) and boost's `weighted_skewness`, on
  hand-built counters with no scorer in sight.
* ``activation`` -- `get_activation_penalty` (bscores.cc:896) over a
  23-point grid, including the DEFAULT bounds that make classic print
  `nan`.
* ``tables`` -- the whole `operator()` over three committed CSVs at 14
  parameter combinations x 6 trees, plus the constructor's
  `_counter`/`_pdf`/`_skewness` and the CTable's `std::map` row order.
"""

import json
import math
import struct
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.moses.moses.types import very_best_score
from pymoses.moses.scoring.bscores import (
    _ctable_rows_in_map_order,
    _mann_whitney_u,
    _ranking,
    _standardized_mann_whitney_u,
    _weighted_skewness,
    interesting_predicate_bscore,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_TABLE_DATA = _FIXTURES / "table_data"
_ORACLE = json.loads((_FIXTURES / "ip_bscore.json").read_text())

_SPECIAL = {"nan": math.nan, "-nan": -math.nan, "inf": math.inf, "-inf": -math.inf}


def _num(value) -> float:
    """The probe writes non-finite doubles as strings so JSON can carry
    them AND so the NaN's sign survives (`float_json`,
    ip_bscore_probe.cc). Decode them back.
    """
    if isinstance(value, str):
        return _SPECIAL[value]
    return float(value)


def _bits(x: float) -> bytes:
    """The comparison key: the IEEE-754 binary64 bit pattern. `==` calls
    every NaN unequal (so a NaN assertion would never fail, and never
    pass), and both `==` and `pytest.approx` call `+0.0` and `-0.0` equal.
    Both distinctions are live in this scorer's output.
    """
    return struct.pack(">d", x)


def _assert_same(got: float, expected, what: str) -> None:
    exp = _num(expected)
    assert _bits(got) == _bits(exp), (
        f"{what}: got {got!r} ({_bits(got).hex()}), expected {exp!r} ({_bits(exp).hex()})"
    )


def _counter_from(pairs) -> dict[float, float]:
    return {_num(k): _num(v) for k, v in pairs}


# ---------------------------------------------------------------------------
# stats -- the two green-field statistics, in isolation
# ---------------------------------------------------------------------------

_STATS_CASES = sorted(_ORACLE["stats"])


@pytest.mark.oracle
@pytest.mark.parametrize("label", _STATS_CASES)
def test_ranking_matches_classic(label):
    """`ranking` (ranking.h:40) over `c1 + c2`, the exact call
    `MannWhitneyU` makes. Pins the running `lrank` and the ascending walk
    together: a `dict`-order walk assigns every rank to the wrong key.
    """
    case = _ORACLE["stats"][label]
    c1 = _counter_from(case["c1"])
    c2 = _counter_from(case["c2"])
    combined = dict(c1)
    for key, count in c2.items():
        combined[key] = combined.get(key, 0.0) + count
    got = _ranking(combined)
    expected = _counter_from(case["ranking_c1_plus_c2"])
    assert list(got) == list(expected), f"{label}: rank ORDER differs"
    for key in expected:
        _assert_same(got[key], expected[key], f"{label} ranking[{key}]")


@pytest.mark.oracle
@pytest.mark.parametrize("label", _STATS_CASES)
def test_weighted_skewness_matches_classic(label):
    """boost's `weighted_skewness` accumulator, fed `(value, weight=count)`
    in ascending-key order. `symmetric` pins the exact 0.0; `empty_c2` and
    `single_point_c2` pin the two `0/0` NaNs, whose sign is what a
    `math.nan` literal would get wrong.
    """
    case = _ORACLE["stats"][label]
    _assert_same(
        _weighted_skewness(_counter_from(case["c1"])),
        case["weighted_skewness_c1"],
        f"{label} weighted_skewness(c1)",
    )
    if case["c2"]:
        _assert_same(
            _weighted_skewness(_counter_from(case["c2"])),
            case["weighted_skewness_c2"],
            f"{label} weighted_skewness(c2)",
        )


@pytest.mark.oracle
@pytest.mark.parametrize("label", _STATS_CASES)
def test_mann_whitney_u_matches_classic(label):
    """`MannWhitneyU` (MannWhitneyU.h:44). `identical` is the case that
    pins the unweighted rank sum: c1 == c2 gives 3.5, not the 20.5 a
    count-weighted sum would.
    """
    case = _ORACLE["stats"][label]
    _assert_same(
        _mann_whitney_u(_counter_from(case["c1"]), _counter_from(case["c2"])),
        case["mann_whitney_u"],
        f"{label} MannWhitneyU",
    )


@pytest.mark.oracle
@pytest.mark.parametrize("label", _STATS_CASES)
def test_standardized_mann_whitney_u_matches_classic(label):
    """`standardizedMannWhitneyU` (MannWhitneyU.h:64). `fractional_counts`
    is the INT-truncation case -- `n2` accumulates to 0 from counts
    totalling exactly 1.0, so `sU` is 0 and the result is `+inf`, not a
    finite z-score. `empty_c2` is the `0/0` that must come back NEGATIVE.
    """
    case = _ORACLE["stats"][label]
    _assert_same(
        _standardized_mann_whitney_u(_counter_from(case["c1"]), _counter_from(case["c2"])),
        case["standardized_mann_whitney_u"],
        f"{label} standardizedMannWhitneyU",
    )


def test_stats_corpus_is_non_vacuous():
    """A corpus on which faithful and broken agree is this project's
    most-repeated defect. Assert the cases actually SPREAD: at least one
    non-finite standardized U, at least one NaN skewness, and at least
    three distinct finite skewness values.
    """
    us = [_num(c["standardized_mann_whitney_u"]) for c in _ORACLE["stats"].values()]
    skews = [_num(c["weighted_skewness_c1"]) for c in _ORACLE["stats"].values()]
    skews += [_num(c["weighted_skewness_c2"]) for c in _ORACLE["stats"].values() if c["c2"]]
    assert any(not math.isfinite(u) for u in us)
    assert any(math.isnan(s) for s in skews)
    assert len({s for s in skews if math.isfinite(s)}) >= 3


# ---------------------------------------------------------------------------
# activation -- get_activation_penalty over a grid
# ---------------------------------------------------------------------------

_ACTIVATION = _ORACLE["activation"]


@pytest.fixture(scope="module")
def _any_scorer():
    """`get_activation_penalty` is a METHOD, so it needs an instance; the
    instance's table is irrelevant to it (it reads only `_min_activation`,
    `_max_activation` and `_penalty`, all overwritten per case below).
    """
    table = loadTable(str(_TABLE_DATA / "ip_sweep.csv"), "y")
    return interesting_predicate_bscore(table.compressed(""))


@pytest.mark.oracle
@pytest.mark.parametrize("index", range(len(_ACTIVATION)), ids=lambda i: str(i))
def test_activation_penalty_matches_classic(index, _any_scorer):
    """The grid at `tools/cpp_probe/ip_bscore_probe.cc`'s `activation`
    mode. Rows 0-4 are classic's DEFAULT bounds, where both quotients are
    `0/0`: the penalty is `nan` for an odd-integer `--alpha` and `-nan`
    otherwise, which is glibc `pow`'s sign_bias rule and the single
    character between classic's `nan true` and `-nan true`.
    """
    case = _ACTIVATION[index]
    _any_scorer._min_activation = _num(case["min_activation"])
    _any_scorer._max_activation = _num(case["max_activation"])
    _any_scorer._penalty = _num(case["penalty"])
    _assert_same(
        _any_scorer._get_activation_penalty(_num(case["activation"])),
        case["activation_penalty"],
        f"activation[{index}] {case}",
    )


def test_activation_grid_is_non_vacuous():
    """The grid must contain all four outcome kinds, or a broken
    `get_activation_penalty` could pass it: a NaN, a `-inf`, an exact
    0.0, and a finite non-zero.
    """
    values = [_num(c["activation_penalty"]) for c in _ACTIVATION]
    assert any(math.isnan(v) for v in values)
    assert any(v == -math.inf for v in values)
    assert any(v == 0.0 for v in values)
    assert any(math.isfinite(v) and v != 0.0 for v in values)
    # ... and both NaN signs, which is the whole point of rows 0-4.
    nans = [v for v in values if math.isnan(v)]
    assert {math.copysign(1.0, v) for v in nans} == {1.0, -1.0}


# ---------------------------------------------------------------------------
# tables -- the whole scorer
# ---------------------------------------------------------------------------

_TABLE_CASES = sorted(_ORACLE["tables"])
_VERTEX_NAME = {builtin.logical_true: "true", builtin.logical_false: "false"}


def _scorer_for(case) -> interesting_predicate_bscore:
    table = loadTable(str(_TABLE_DATA / case["csv"]), case["target"])
    kld_w, skew_w, stdU_w, skewU_w = (_num(w) for w in case["weights"])
    lo, hi = (_num(a) for a in case["activation"])
    return interesting_predicate_bscore(
        table.compressed(""),
        kld_w,
        skew_w,
        stdU_w,
        skewU_w,
        lo,
        hi,
        _num(case["penalty"]),
        case["positive"],
    )


@pytest.mark.oracle
@pytest.mark.parametrize("label", _TABLE_CASES)
def test_ctable_row_order_matches_std_map(label):
    """`CTable` is a `std::map<multi_type_seq, TimedCounter>`, so classic
    walks it ascending by input tuple; this port's is an insertion-ordered
    dict. The order is OBSERVABLE because `operator()` builds
    `pred_counter` with an assignment (bscores.cc:800), so it is asserted
    directly rather than only through the scores.
    """
    case = _ORACLE["tables"][label]
    table = loadTable(str(_TABLE_DATA / case["csv"]), case["target"])
    rows = _ctable_rows_in_map_order(table.compressed(""))
    got = [[_VERTEX_NAME[v] for v in key] for key, _cell in rows]
    assert got == [row["key"] for row in case["ctable_rows"]]


@pytest.mark.oracle
@pytest.mark.parametrize("label", _TABLE_CASES)
def test_constructor_state_matches_classic(label):
    """`_counter` (the `+=` histogram), `_pdf` (still raw COUNTS after
    `set_p_pdf`, which mutates only its own copy) and `_skewness`.

    Compared as MAPPINGS, not as ordered sequences, and deliberately so.
    C++'s `Counter<contin_t, contin_t>` is a `std::map`, so the fixture
    lists it ascending by key, while this port's is a `dict` in
    first-appearance order -- the same choice `kld.py` already made and
    documented: leaving the container unsorted keeps every `sorted()` at
    a USE site an independently killable mutation point instead of
    masking it behind a pre-sorted container. The ordering is not
    observable from `_counter` anyway: its three readers
    (`KLDS.set_p_pdf`, `_weighted_skewness`,
    `_standardized_mann_whitney_u`) each sort on the way in, and
    `test_operator_call_matches_classic` below would fail bit-exactly if
    any of them did not.

    What IS asserted here is the key SET and every count bit-exactly,
    which pins the `+=` accumulation (and its order, since a different
    accumulation order would change the low bits of a non-integral
    count).
    """
    case = _ORACLE["tables"][label]
    bsc = _scorer_for(case)
    expected_counter = _counter_from(case["counter"])
    assert set(bsc._counter) == set(expected_counter), f"{label}: _counter KEYS differ"
    for key, count in expected_counter.items():
        _assert_same(bsc._counter[key], count, f"{label} _counter[{key}]")
    expected_pdf = _counter_from(case["pdf"])
    assert set(bsc._pdf) == set(expected_pdf), f"{label}: _pdf KEYS differ"
    for key, count in expected_pdf.items():
        _assert_same(bsc._pdf[key], count, f"{label} _pdf[{key}]")
    _assert_same(bsc._skewness, case["skewness"], f"{label} _skewness")


@pytest.mark.oracle
@pytest.mark.parametrize("label", _TABLE_CASES)
def test_operator_call_matches_classic(label):
    """Every bscore ELEMENT, plus `sum_bscore`, for six fixed trees.
    `and($1 not($1))` selects no rows and is the case where four of the
    five elements are non-finite with three different signs.
    """
    case = _ORACLE["tables"][label]
    bsc = _scorer_for(case)
    for entry in case["trees"]:
        tr = str_to_combo_tree(entry["tree"])
        got = bsc(tr)
        expected = [_num(x) for x in entry["bscore"]]
        assert len(got) == len(expected), (
            f"{label} {entry['tree']}: bscore LENGTH {len(got)} != {len(expected)}"
        )
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            _assert_same(g, e, f"{label} {entry['tree']} bscore[{i}]")
        _assert_same(
            bsc.sum_bscore(got), entry["sum_bscore"], f"{label} {entry['tree']} sum_bscore"
        )


@pytest.mark.oracle
@pytest.mark.parametrize("label", _TABLE_CASES)
def test_best_possible_bscore_and_min_improv_match_classic(label):
    case = _ORACLE["tables"][label]
    bsc = _scorer_for(case)
    expected_best = [_num(x) for x in case["best_possible_bscore"]]
    assert len(bsc.best_possible_bscore()) == len(expected_best)
    for g, e in zip(bsc.best_possible_bscore(), expected_best, strict=True):
        _assert_same(g, e, f"{label} best_possible_bscore")
    _assert_same(bsc.min_improv(), case["min_improv"], f"{label} min_improv")
    # ... and it is `very_best_score`, not an infinity: the C++ comment
    # says "the maximum value a behavioral_score can represent"
    # (bscores.h:576), which is FLT_MAX.
    assert bsc.best_possible_bscore() == [very_best_score]


def test_row_order_does_not_change_any_score():
    """`ip_boolean_contin_reversed.csv` is `ip_boolean_contin.csv` with
    its 14 data rows reversed. Classic scores the two identically (it
    sorts the CTable); a port that walked the rows in FILE order would
    not, because `pred_counter`'s assignment lets the last-visited
    duplicate win.

    Asserted against the port's OWN two runs rather than only against the
    fixture, so this stays a real test if the fixture is ever
    regenerated from a single CSV.
    """
    forward = _scorer_for(_ORACLE["tables"]["skewed_real_bounds"])
    reversed_ = _scorer_for(_ORACLE["tables"]["skewed_reversed_rows"])
    assert forward._counter == reversed_._counter
    _assert_same(reversed_._skewness, forward._skewness, "_skewness under row reversal")
    trees_seen = 0
    for entry in _ORACLE["tables"]["skewed_real_bounds"]["trees"]:
        tr = str_to_combo_tree(entry["tree"])
        a, b = forward(tr), reversed_(str_to_combo_tree(entry["tree"]))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(y, x, f"{entry['tree']} bscore[{i}] under row reversal")
        trees_seen += 1
    assert trees_seen == 6


def test_table_corpus_is_non_vacuous():
    """The bscores must actually SPREAD -- across trees, across cases and
    across LENGTHS -- or a scorer that ignored its weights could pass.
    """
    lengths = set()
    sums = set()
    for case in _ORACLE["tables"].values():
        for entry in case["trees"]:
            lengths.add(len(entry["bscore"]))
            sums.add(str(entry["sum_bscore"]))
    # 1 is the `pred_counter.size() == 1` punt (bscores.cc:805-811), a
    # lone `very_worst_score`; 2 is `weights_skew_U_only` (skew_U +
    # activation); 5 is all four weights plus activation. All five
    # lengths `operator()` can produce are present.
    assert lengths == {1, 2, 3, 4, 5}, lengths
    assert len(sums) > 40, len(sums)
    # ... and the size-1 punt must really be `very_worst_score`, not some
    # other one-element bscore.
    punts = [
        entry
        for case in _ORACLE["tables"].values()
        for entry in case["trees"]
        if len(entry["bscore"]) == 1
    ]
    assert punts, "the pred_counter.size() == 1 punt is never reached"
    for entry in punts:
        assert _num(entry["bscore"][0]) == -very_best_score


def test_pred_counter_assignment_defect_is_actually_exercised():
    """The DO NOT FIX in `__call__` -- `pred_counter[...] = mv.second`
    rather than `+=` -- is only a real trap on a table where some output
    value occurs in MORE THAN ONE compressed row. Assert the committed
    fixture is such a table, so the note is not guarding a case the
    corpus never reaches.
    """
    table = loadTable(str(_TABLE_DATA / "ip_boolean_contin.csv"), "y")
    ctable = table.compressed("")
    from pymoses.comboreduct.table.table import timed_items_of

    seen: dict[float, int] = {}
    for _row, cell in ctable.items():
        for tv, _count in timed_items_of(cell):
            from pymoses.comboreduct.combo.vertex import get_contin

            value = get_contin(tv.value)
            seen[value] = seen.get(value, 0) + 1
    shared = {v: n for v, n in seen.items() if n > 1}
    assert shared, "no output value spans two compressed rows -- the defect is unreachable"
    # And the two histograms must therefore DISAGREE for the `true`
    # predicate, which selects every row.
    bsc = interesting_predicate_bscore(ctable, 1, 1, 1, 1, 0.2, 0.8, 1, True)
    bsc(str_to_combo_tree("true"))
    # `_counter` sums; a faithful `pred_counter` assigns, so at least one
    # value is under-counted relative to it.
    assert any(bsc._counter[v] > 1.0 for v in shared)
