"""Oracle and guard tests for pymetta.moses.scoring.bscores.cluster_bscore
-- the `-H cluster` scorer (C++ bscores.h:613, bscores.cc:937).

The oracle fixture (tests/fixtures/cluster_bscore.json) is produced by
tools/cpp_probe/cluster_bscore_probe.cc against the real classic library
(see tools/capture_cluster_bscore.py) -- not reimplemented by hand. It has
five sections; `synthetic`, `overflow`, `nan` and `too_small` each carry
the exact input column alongside the score, so no case below re-derives
its own input, and `tables` carries two real tables.

Every numeric comparison is EXACT (`==`), not `pytest.approx`, for the
reason test_contin_bscore.py's docstring gives: `score_t` is genuinely
narrowed to IEEE-754 binary32 wherever classic narrows, so this port's
Python floats hold the same bit pattern (widened) as the probe's `%.17g`
round trip.

`-H cluster` also has an end-to-end CLI oracle in
tests/moses/main/test_moses_exec.py. That one is the proof the whole
problem is wired; THIS one is the proof the arithmetic inside the scorer
is right, on inputs the search loop does not reach: float32 overflow,
float32 accumulation, an underpopulated cluster, an exactly-zero RMS, a
too-small table, and a nan column.
"""

import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table import ITable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.scoring.bscores import cluster_bscore

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = json.loads((_FIXTURE_DIR / "cluster_bscore.json").read_text())
_SAMPLE_CSV = _FIXTURE_DIR / "table_data" / "contin_regression_sample.csv"

# The probe's `float_json` writes non-finite values as JSON strings (a bare
# `inf`/`nan` token is not valid JSON), so decode them back here.
_SPECIAL = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def _num(v):
    return _SPECIAL[v] if isinstance(v, str) else v


def _nums(vs):
    return [_num(v) for v in vs]


def _same(got, want) -> bool:
    """Exact float equality, except that `nan == nan` must count as a
    match: `cluster_bscore` genuinely returns nan for a column whose
    squares overflow binary32 (see the `overflow_control_identity` case),
    and a plain `==` would silently pass a port that returned anything
    else there.
    """
    if len(got) != len(want):
        return False
    return all((math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(got, want, strict=True))


def _itable(column) -> ITable:
    it = ITable()
    for v in column:
        it.push_back((float(v),))
    return it


def _check_case(case) -> None:
    """Drive one probe case -- its own recorded column, its own recorded
    tree -- and compare against its own recorded result.
    """
    bsc = cluster_bscore(_itable(_nums(case["column"])))
    tr = str_to_combo_tree(case["tree"])

    if "exception" in case:
        # classic's OC_ASSERT throws opencog::AssertionException; this
        # port raises AssertionError (docs/CARRY-FORWARD.md Sec 5). The
        # recorded `what()` carries only the file:line, since the
        # OC_ASSERT at bscores.cc:944 has no message -- so pin the LINE,
        # which is the part that identifies the assertion.
        assert "bscores.cc:944" in case["exception"]
        with pytest.raises(AssertionError):
            bsc(tr)
    else:
        assert _same(bsc(tr), _nums(case["bscore"])), case["tree"]

    assert bsc.best_possible_bscore() == _nums(case["best_possible_bscore"])
    assert bsc.min_improv() == _num(case["min_improv"])


# ---------------------------------------------------------------------------
# synthetic: hand-built single-column ITables, tree "$1"
# ---------------------------------------------------------------------------

_SYNTHETIC_CASES = [
    "three_clusters",
    "multi_pass",
    "underpopulated",
    "zero_rms",
    "four_rows",
    "float32_sum",
]


def test_synthetic_corpus_is_non_vacuous():
    """CARRY-FORWARD.md Sec 5/6: an empty corpus a parametrize walks over
    passes silently. Pin the count AND, more importantly, pin that the
    corpus is not all-degenerate: at least three cases must return a
    FINITE score, and at least one must return each of `-inf` and `+inf`.
    A `-H cluster` corpus of nothing but `-inf` would match a completely
    broken port.
    """
    assert set(_GOLDEN["synthetic"]) == set(_SYNTHETIC_CASES)
    scores = [
        _num(_GOLDEN["synthetic"][name]["bscore"][0])
        for name in _SYNTHETIC_CASES
        if "bscore" in _GOLDEN["synthetic"][name]
    ]
    assert sum(math.isfinite(s) for s in scores) >= 3
    assert any(s == -math.inf for s in scores)
    assert any(s == math.inf for s in scores)


@pytest.mark.oracle
@pytest.mark.parametrize("name", _SYNTHETIC_CASES)
def test_synthetic_matches_cpp(name):
    _check_case(_GOLDEN["synthetic"][name])


def test_multi_pass_really_needs_more_than_one_pass():
    """`multi_pass`'s column is not just another finite case: its first
    three rows (10, 14, 30 -- the centroid seed, bscores.cc:946-952) put
    the initial edges in the wrong place, so `while (changed)`
    (bscores.cc:973) has to run more than once. Counted here by
    instrumenting the loop from outside -- `edges` is a local, so the
    count is recovered by re-running the algorithm's own convergence test
    on the recorded column rather than by reaching into the scorer.

    Without this the case would prove only that some finite number came
    out, not that the convergence loop iterates at all -- and a port that
    ran exactly one pass would still pass every `==` above for the five
    other synthetic cases, all of which converge in one.
    """
    column = _nums(_GOLDEN["synthetic"]["multi_pass"]["column"])
    from pymetta.moses.moses.types import _to_score_t

    vals = sorted(_to_score_t(v) for v in column)
    centers = sorted(_to_score_t(v) for v in column[:3])
    edges = [
        _to_score_t(0.5 * _to_score_t(centers[0] + centers[1])),
        _to_score_t(0.5 * _to_score_t(centers[1] + centers[2])),
        math.inf,
    ]
    edge_idx = [0, 0]
    passes = 0
    changed = True
    while changed:
        passes += 1
        cnt = [0.0] * 3
        total = [0.0] * 3
        changed = False
        i = 0
        for j, sc in enumerate(vals):
            if sc <= edges[i]:
                cnt[i] = _to_score_t(cnt[i] + 1.0)
                total[i] = _to_score_t(total[i] + sc)
            else:
                if j != edge_idx[i]:
                    changed = True
                edge_idx[i] = j
                i += 1
        assert all(c >= 3.5 for c in cnt)
        total = [_to_score_t(t / c) for t, c in zip(total, cnt, strict=True)]
        edges[0] = _to_score_t(0.5 * _to_score_t(total[0] + total[1]))
        edges[1] = _to_score_t(0.5 * _to_score_t(total[1] + total[2]))
        assert passes < 50, "convergence loop did not terminate"
    assert passes > 1, "multi_pass converges in one pass -- it is not testing the loop"
    # For contrast: the plain `three_clusters` column converges in one.
    assert _num(_GOLDEN["synthetic"]["three_clusters"]["bscore"][0]) != _num(
        _GOLDEN["synthetic"]["multi_pass"]["bscore"][0]
    )


# ---------------------------------------------------------------------------
# overflow: score_t is binary32, and that decides the answer
# ---------------------------------------------------------------------------

_OVERFLOW_CASES = ["overflow_times", "overflow_control_identity", "overflow_control_scaled"]


@pytest.mark.oracle
@pytest.mark.parametrize("name", _OVERFLOW_CASES)
def test_overflow_matches_cpp(name):
    _check_case(_GOLDEN["overflow"][name])


def test_overflow_trio_discriminates():
    """The three overflow cases must give three DIFFERENT kinds of answer,
    or the pair proves nothing:

      * `overflow_times` (column ~1e30, tree `*($1 $1)`) -> `-inf`: the
        products are finite `contin_t` (1e60) that become `+inf` only
        when narrowed to `score_t` at bscores.cc:967, which trips the
        `std::isinf` guard at bscores.cc:982;
      * `overflow_control_identity` (SAME column, tree `$1`) -> `nan`:
        nothing trips the guard (1e30 is representable in binary32), but
        `squ += sc*sc` at bscores.cc:1033 is float*float, so `squ` and
        `sum*sum` both reach `+inf` and `final` becomes `inf - inf`;
      * `overflow_control_scaled` (same shape at ~1e3, same squaring
        tree) -> finite: nothing overflows anywhere.

    A port that kept `contin_t` (double) precision at bscores.cc:967
    would return the third kind of answer for the first case.
    """
    got = {n: _num(_GOLDEN["overflow"][n]["bscore"][0]) for n in _OVERFLOW_CASES}
    assert got["overflow_times"] == -math.inf
    assert math.isnan(got["overflow_control_identity"])
    assert math.isfinite(got["overflow_control_scaled"])


# ---------------------------------------------------------------------------
# nan: std::sort's UB case, and why it cannot matter
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["nan_everywhere", "nan_control_finite"])
def test_nan_matches_cpp(name):
    _check_case(_GOLDEN["nan"][name])


def test_nan_column_result_does_not_depend_on_sort_order():
    """The written reasoning in `cluster_bscore.__call__`'s docstring for
    why `list.sort()` is a faithful stand-in for `std::sort` rests on
    this: a nan anywhere in the column makes the sweep at
    bscores.cc:982-986 return `[-inf]` wherever the sort put it, so the
    permutation is unobservable -- which matters because nan makes
    `std::sort` undefined behaviour in C++ and no permutation is "the"
    faithful one.

    Demonstrated by feeding the SAME multiset in several different
    orders, including nan first, nan last, and nan interleaved.
    """
    finite = [1.0, 2.0, 3.0, 1.25, 1.5, 1.75, 0.75, 2.25, 2.5, 1.9, 2.1, 3.25, 3.5, 2.9, 3.1]
    tr = str_to_combo_tree("$1")
    orders = [
        [math.nan, *finite],
        [*finite, math.nan],
        [*finite[:7], math.nan, *finite[7:]],
        [*finite[:3], math.nan, *finite[3:]],
    ]
    for column in orders:
        assert cluster_bscore(_itable(column))(tr) == [-math.inf]
    # And the same column WITHOUT the nan is finite -- so the -inf above
    # is the guard, not the shape.
    assert math.isfinite(cluster_bscore(_itable(finite))(tr)[0])


# ---------------------------------------------------------------------------
# too_small: OC_ASSERT(nclusters < oned.size())
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["three_rows", "two_rows"])
def test_too_small_matches_cpp(name):
    _check_case(_GOLDEN["too_small"][name])


def test_four_rows_is_the_first_size_that_does_not_assert():
    """The assert is `nclusters < oned.size()` with `nclusters == 3`
    (bscores.cc:942-944), i.e. strictly greater than three rows -- so four
    rows passes it (and then fails the `cnt[i] < 3.5` guard instead, which
    is a DIFFERENT outcome: `-inf`, not a raise). Pins the boundary
    directly; a `<=` in place of `<` would still raise on three rows.
    """
    tr = str_to_combo_tree("$1")
    with pytest.raises(AssertionError):
        cluster_bscore(_itable([1.0, 2.0, 3.0]))(tr)
    assert cluster_bscore(_itable([1.0, 2.0, 3.0, 4.0]))(tr) == [-math.inf]


# ---------------------------------------------------------------------------
# tables: real datasets
# ---------------------------------------------------------------------------


def _check_table(result, table) -> None:
    assert table.size() == result["n_rows"]
    assert table.get_arity() == result["arity"]
    bsc = cluster_bscore(table.itable)
    seen = 0
    for key, case in result.items():
        if not key.startswith("tree_"):
            continue
        seen += 1
        tr = str_to_combo_tree(case["tree"])
        bs = bsc(tr)
        assert _same(bs, _nums(case["bscore"])), case["tree"]
        assert bsc.sum_bscore(bs) == _num(case["sum_bscore"])
        assert bsc.best_possible_bscore() == _nums(case["best_possible_bscore"])
        assert bsc.min_improv() == _num(case["min_improv"])
    assert seen >= 4


@pytest.mark.oracle
def test_contin_regression_sample_matches_cpp():
    entry = _GOLDEN["tables"]["contin_regression_sample"]
    _check_table(entry["result"], loadTable(str(_SAMPLE_CSV), entry["target"]))


@pytest.mark.oracle
def test_training_regression_matches_cpp(classic_root):
    """8000 rows -- the case where a float32 `sum[i] += sc` accumulator
    (bscores.cc:992) demonstrably drifts from a double one. Reads
    `$CLASSIC/trainingRegression.dat` directly rather than vendoring 8000
    rows into this repo; `classic_root` (tests/conftest.py) skips the test
    if the classic checkout is absent.
    """
    entry = _GOLDEN["tables"]["training_regression"]
    path = classic_root / entry["classic_relpath"]
    if not path.is_file():
        pytest.skip(f"{path} not present")
    _check_table(entry["result"], loadTable(str(path), entry["target"]))


def test_table_corpus_is_not_all_degenerate():
    """The `-inf`-corpus trap, pinned for the table section too: the two
    real tables together must contribute at least four DISTINCT finite
    scores, not merely "no crash".
    """
    finite = set()
    for entry in _GOLDEN["tables"].values():
        for key, case in entry["result"].items():
            if key.startswith("tree_") and "bscore" in case:
                s = _num(case["bscore"][0])
                if math.isfinite(s):
                    finite.add(s)
    assert len(finite) >= 4, finite
