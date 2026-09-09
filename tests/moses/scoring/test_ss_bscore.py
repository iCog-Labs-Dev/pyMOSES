"""Oracle and guard tests for pymetta.moses.scoring.ss_bscore.

The oracle fixture (tests/fixtures/ss_bscore.json) is produced by
tools/cpp_probe/ss_bscore_probe.cc against real `boost::accumulators` and
`opencog::RandGen`/`opencog::rand_element_erase` (see
tools/capture_ss_bscore.py) -- not reimplemented by hand. It carries four
sections: "accumulator" (the `_VarAcc` float32 mean/variance recurrence),
"low_dev_penalty" (`sqrtf`-based penalty), "partition_by_row" (shuffle then
split), and "partition_by_time" (`rand_element_erase` round robin).

`ss_bscore` itself CANNOT be constructed in this port today -- see that
module's own docstring for why (`CTable.get_timestamps()` does not exist,
a pre-existing gap entirely outside this task's owned tree). `_make_
uninitialized` below builds an instance by bypassing `__init__` (`object.
__new__`), documented at every use, to exercise the rest of the class
(`__call__`'s fast path, and every delegating method) against a plain
in-repo test double standing in for `_bscorer`.
"""

import json
from pathlib import Path

import pytest

from pymetta.moses.scoring.ss_bscore import (
    _boost_accumulate,
    _low_dev_penalty,
    _partition_by_row,
    _partition_by_time,
    _shuffle_with_gen,
    _VarAcc,
    ss_bscore,
)
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "ss_bscore.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


# --------------------------------------------------------------------- _VarAcc


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["accumulator"], ids=lambda c: f"samples{c['samples']}"
)
def test_var_acc_matches_cpp(case):
    acc = _VarAcc()
    for i, x in enumerate(case["samples"]):
        acc(x)
        want = case["after_each"][i]
        assert acc.mean() == pytest.approx(want["mean"], rel=0, abs=1e-6)
        assert acc.variance() == pytest.approx(want["variance"], rel=0, abs=1e-6)


def test_var_acc_starts_at_zero_not_first_sample():
    """Direct check of the eager-zero-init reading (module docstring):
    variance after exactly one sample is 0.0, not the sample's own value.
    """
    acc = _VarAcc()
    acc(42.0)
    assert acc.mean() == 42.0
    assert acc.variance() == 0.0


def test_var_acc_mutation_using_lazy_variance_formula_diverges():
    """Mutation proof: the LAZY variance formula (moment<2> - mean^2,
    computed from scratch each time rather than iteratively) is
    mathematically the same statistic but reaches it by a different
    float32 rounding route -- confirming this module's own claim that the
    ROUTE, not just the target statistic, is what must match. Uses a
    sequence long enough (5 samples, non-trivial values) that float32
    rounding differences actually surface.
    """
    case = next(c for c in _GOLDEN["accumulator"] if len(c["samples"]) == 5)
    acc = _VarAcc()
    running_sum_sq = 0.0
    running_sum = 0.0
    n = 0
    lazy_variance_final = None
    for x in case["samples"]:
        acc(x)
        n += 1
        running_sum += x
        running_sum_sq += x * x
        mean = running_sum / n
        lazy_variance_final = running_sum_sq / n - mean * mean
    assert acc.variance() != lazy_variance_final, (
        "iterative and lazy variance formulas coincided exactly -- "
        "this fixture case cannot discriminate the two routes"
    )
    want_final_variance = case["after_each"][-1]["variance"]
    assert acc.variance() == pytest.approx(want_final_variance, rel=0, abs=1e-6)


# ---------------------------------------------------------------- low_dev_penalty


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["low_dev_penalty"],
    ids=lambda c: f"var{c['sc_var']}_pressure{c['low_dev_pressure']}",
)
def test_low_dev_penalty_matches_cpp(case):
    got = _low_dev_penalty(case["sc_var"], case["low_dev_pressure"])
    assert got == pytest.approx(case["low_dev_penalty"], rel=0, abs=1e-6)


def test_low_dev_penalty_mutation_dropping_negation():
    """Mutation proof: dropping the leading `-` changes every non-zero
    result's sign.
    """
    positive_cases = [c for c in _GOLDEN["low_dev_penalty"] if c["low_dev_penalty"] != 0]
    assert positive_cases
    for case in positive_cases:
        mutant = _low_dev_penalty(case["sc_var"], case["low_dev_pressure"]) * -1
        assert mutant != pytest.approx(case["low_dev_penalty"], rel=0, abs=1e-6) or mutant == 0


# ----------------------------------------------------------------- _shuffle_with_gen


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["partition_by_row"], ids=lambda c: f"usize{c['usize']}_seed{c['seed']}"
)
def test_shuffle_with_gen_matches_cpp(case):
    randGen().seed(case["seed"])
    items = list(range(case["usize"]))
    _shuffle_with_gen(items, lambda i: randGen().randint_n(i))
    assert items == case["shuffled"]


def test_shuffle_with_gen_mutation_backward_pass_diverges():
    """Mutation proof: a BACKWARD Fisher-Yates pass (the OTHER common
    shuffle shape, iterating i from n-1 down to 1) draws the SAME number of
    values from the RNG but combines them differently, giving a different
    permutation for at least one fixture case.
    """

    def _backward_shuffle(items, rand):
        n = len(items)
        for i in range(n - 1, 0, -1):
            j = rand(i + 1)
            if i != j:
                items[i], items[j] = items[j], items[i]

    mismatches = 0
    for case in _GOLDEN["partition_by_row"]:
        if case["usize"] < 3:
            continue
        randGen().seed(case["seed"])
        items = list(range(case["usize"]))
        _backward_shuffle(items, lambda i: randGen().randint_n(i))
        if items != case["shuffled"]:
            mismatches += 1
    assert mismatches > 0, "backward shuffle matched forward for every case -- weak fixture"


# ----------------------------------------------------------------- _partition_by_row


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["partition_by_row"],
    ids=lambda c: f"usize{c['usize']}_n{c['n_subsamples']}_seed{c['seed']}",
)
def test_partition_by_row_matches_cpp(case):
    randGen().seed(case["seed"])
    groups = _partition_by_row(case["usize"], case["n_subsamples"])
    want_groups = [set(g) for g in case["groups"]]
    assert groups == want_groups


def test_partition_by_row_usize_less_than_n_subsamples_is_empty():
    """DIVERGENCE-documented edge case (module docstring): usize <
    n_subsamples gives seg_size == 0, C++'s loop runs zero times leaving
    an empty vector that the NEXT loop then indexes out of bounds (UB).
    This port returns an empty list instead of reproducing UB.
    """
    randGen().seed(0)
    groups = _partition_by_row(3, 5)
    assert groups == []


# ---------------------------------------------------------------- _partition_by_time


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["partition_by_time"],
    ids=lambda c: f"ts{c['timestamps']}_n{c['n_subsamples']}_seed{c['seed']}",
)
def test_partition_by_time_matches_cpp(case):
    randGen().seed(case["seed"])
    groups = _partition_by_time(set(case["timestamps"]), case["n_subsamples"])
    want_groups = [set(g) for g in case["groups"]]
    assert groups == want_groups


def test_partition_by_time_ignores_input_insertion_order():
    """Direct check that an out-of-sorted-order input set produces the
    same partition as its sorted equivalent -- `std::set` always iterates
    sorted regardless of insertion order, and this function must too.
    """
    case = next(c for c in _GOLDEN["partition_by_time"] if c["timestamps"] == [1, 3, 5, 7, 9])
    randGen().seed(case["seed"])
    groups_from_shuffled_input = _partition_by_time({9, 1, 5, 3, 7}, case["n_subsamples"])
    want_groups = [set(g) for g in case["groups"]]
    assert groups_from_shuffled_input == want_groups


def test_partition_by_time_mutation_unsorted_list_diverges():
    """Mutation proof: using the set's OWN (hash) iteration order instead
    of sorting first can disagree with `std::set`'s sorted order -- and
    does, for at least one non-trivial fixture case (Python's set iteration
    order for small ints is not the same as C++'s `std::set` sorted order
    in general, though it MAY coincide for very small sets -- this proof
    only claims at least one fixture case discriminates, not all).
    """

    def _partition_unsorted(timestamps, n_subsamples, rng):
        remaining = list(timestamps)  # mutation: NOT sorted
        groups = [set() for _ in range(n_subsamples)]
        tt_size = len(remaining)
        for i in range(tt_size):
            idx = rng.randint_n(len(remaining))
            time = remaining.pop(idx)
            groups[i % n_subsamples].add(time)
        return groups

    mismatches = 0
    for case in _GOLDEN["partition_by_time"]:
        if len(case["timestamps"]) < 3:
            continue
        randGen().seed(case["seed"])
        # A dict (insertion-ordered, not sorted) built from a DELIBERATELY
        # scrambled insertion sequence -- Python's plain `set` for small
        # ints often iterates in numeric order regardless of insertion,
        # which would defeat this mutant; a list preserves the scramble.
        scrambled = sorted(case["timestamps"], key=lambda t: (t * 2654435761) % 1000)
        groups = _partition_unsorted(scrambled, case["n_subsamples"], randGen())
        want_groups = [set(g) for g in case["groups"]]
        if groups != want_groups:
            mismatches += 1
    assert mismatches > 0, "unsorted partition matched sorted for every case -- weak fixture"


# ------------------------------------------------------------------ _boost_accumulate


def test_boost_accumulate_is_naive_fold_not_builtin_sum():
    """Pins the exact counter-example bscore_base.sum_bscore's own
    docstring cites: CPython's compensated `sum()` gives 1.0 exactly for
    this input where a naive fold gives 0.9999999999999999.
    """
    values = [0.1] * 10
    assert _boost_accumulate(values) != sum(values)
    assert _boost_accumulate(values) == 0.9999999999999999


# ------------------------------------------------------------------------- ss_bscore


class _CTableDouble:
    """A minimal stand-in for a real CTable, deliberately WITHOUT
    `get_timestamps` -- matching this port's real `CTable` (table.py),
    which has no such method at all (see ss_bscore.py's own docstring).
    """


class _BscoreDouble:
    """A minimal, real, in-repo test double for `bscore_base` -- exercises
    every method `ss_bscore` delegates to, without needing any actual
    `bscore_ctable_base` subclass's full machinery.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self._ctable = _CTableDouble()
        self._ctable_usize = 10
        self._score_map: dict = {}

    def __call__(self, tr):
        self.calls.append(("call", tr))
        return self._score_map.get(tr, [1.0, 2.0, 3.0])

    def best_possible_bscore(self):
        return [10.0]

    def min_improv(self):
        return 0.5

    def ignore_cols(self, idxs):
        self.calls.append(("ignore_cols", frozenset(idxs)))

    def ignore_rows(self, idxs):
        self.calls.append(("ignore_rows", frozenset(idxs)))

    def ignore_rows_at_times(self, timestamps):
        self.calls.append(("ignore_rows_at_times", frozenset(timestamps)))

    def get_ctable_usize(self):
        return self._ctable_usize

    def get_ctable(self):
        return self._ctable


def _make_uninitialized(bscorer, n_subsamples=0, low_dev_pressure=0.0, by_time=True):
    """Bypasses `ss_bscore.__init__` (`object.__new__`), documented here as
    the ONLY way to obtain an `ss_bscore` instance in this port today --
    see ss_bscore.py's own module docstring for why real construction
    always raises. Sets exactly the attributes `__init__` would have set
    apart from the blocked `self._timestamps` line.
    """
    from pymetta.moses.moses.types import _to_score_t

    inst = object.__new__(ss_bscore)
    inst._bscorer = bscorer
    inst._n_subsamples = n_subsamples
    inst._low_dev_pressure = _to_score_t(low_dev_pressure)
    inst._by_time = by_time
    inst._timestamps = set()
    return inst


def test_construction_raises_because_get_timestamps_is_missing():
    bscorer = _BscoreDouble()
    with pytest.raises(NotImplementedError, match="get_timestamps"):
        ss_bscore(bscorer, n_subsamples=2, low_dev_pressure=0.5)


def test_construction_raises_even_with_default_zero_subsamples():
    """C++'s constructor calls get_ctable().get_timestamps() UNCONDITIONALLY
    -- even the harmless-looking default construction fails in this port.
    """
    bscorer = _BscoreDouble()
    with pytest.raises(NotImplementedError, match="get_timestamps"):
        ss_bscore(bscorer)


def test_call_fast_path_when_n_subsamples_is_zero():
    bscorer = _BscoreDouble()
    tr = "tree-a"
    bscorer._score_map[tr] = [1.0, 2.0]
    ssb = _make_uninitialized(bscorer, n_subsamples=0, low_dev_pressure=0.5)
    result = ssb(tr)
    assert result == [1.0, 2.0]
    assert bscorer.calls == [("call", tr)]


def test_call_fast_path_when_low_dev_pressure_is_zero():
    bscorer = _BscoreDouble()
    tr = "tree-b"
    bscorer._score_map[tr] = [5.0]
    ssb = _make_uninitialized(bscorer, n_subsamples=3, low_dev_pressure=0.0)
    result = ssb(tr)
    assert result == [5.0]
    assert bscorer.calls == [("call", tr)]


def test_call_slow_path_by_row_appends_low_dev_penalty():
    bscorer = _BscoreDouble()
    tr = "tree-c"
    ssb = _make_uninitialized(bscorer, n_subsamples=2, low_dev_pressure=1.0, by_time=False)
    randGen().seed(0)
    result = ssb(tr)
    # 2 subsample calls + 1 final recompute call = 3 total.
    assert len([c for c in bscorer.calls if c[0] == "call"]) == 3
    # 2 groups + final empty restore.
    assert len([c for c in bscorer.calls if c[0] == "ignore_rows"]) == 3
    assert bscorer.calls[-2] == ("ignore_rows", frozenset())
    # Last element is the appended low_dev_penalty; the rest is the final
    # recompute's own bscore (constant [1.0, 2.0, 3.0] from the double).
    assert result[:-1] == [1.0, 2.0, 3.0]
    assert isinstance(result[-1], float)


def test_call_slow_path_by_time_appends_low_dev_penalty():
    bscorer = _BscoreDouble()
    tr = "tree-d"
    ssb = _make_uninitialized(bscorer, n_subsamples=2, low_dev_pressure=1.0, by_time=True)
    ssb._timestamps = {1, 2, 3, 4}
    randGen().seed(0)
    result = ssb(tr)
    assert len([c for c in bscorer.calls if c[0] == "call"]) == 3
    ignore_time_calls = [c for c in bscorer.calls if c[0] == "ignore_rows_at_times"]
    assert len(ignore_time_calls) == 2
    all_timestamps_seen = set().union(*(c[1] for c in ignore_time_calls))
    assert all_timestamps_seen == {1, 2, 3, 4}
    assert result[:-1] == [1.0, 2.0, 3.0]


def test_call_slow_path_zero_variance_when_bscorer_is_constant():
    """When every subsample scores identically, `_VarAcc`'s variance is
    always exactly 0.0 -- so the low_dev_penalty is exactly -0.0/0.0
    regardless of low_dev_pressure (`-p * sqrt(0) == -0.0` or `0.0`).
    """
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer, n_subsamples=3, low_dev_pressure=2.0, by_time=False)
    randGen().seed(1)
    result = ssb("tree-e")
    assert result[-1] == 0.0 or result[-1] == -0.0


def test_ignore_cols_delegates():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    ssb.ignore_cols({1, 2})
    assert bscorer.calls == [("ignore_cols", frozenset({1, 2}))]


def test_best_possible_bscore_delegates():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    assert ssb.best_possible_bscore() == [10.0]


def test_min_improv_delegates():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    assert ssb.min_improv() == 0.5


def test_get_ctable_usize_delegates():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    assert ssb.get_ctable_usize() == 10


def test_get_ctable_delegates():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    assert ssb.get_ctable() is bscorer._ctable


def test_ignore_rows_on_ss_bscore_itself_always_raises():
    """Genuine classic behaviour (OC_ASSERT(false, "Not implemented yet")
    in ss_bscore.cc itself), not a port gap -- see ss_bscore.py's own
    docstring distinguishing this from `_bscorer.ignore_rows`.
    """
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    with pytest.raises(NotImplementedError, match="Not implemented yet"):
        ssb.ignore_rows({1, 2})
    assert bscorer.calls == []  # never reaches _bscorer at all


def test_ignore_rows_at_times_on_ss_bscore_itself_always_raises():
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer)
    with pytest.raises(NotImplementedError, match="Not implemented yet"):
        ssb.ignore_rows_at_times({1, 2})
    assert bscorer.calls == []


def test_mutation_swapping_call_order_in_slow_path_changes_ignore_call_count():
    """Mutation proof for `__call__`'s structure: if the final "recompute
    over all data points" `ignore_rows(set())` + extra `__call__` were
    omitted, exactly 2 fewer bscorer calls would be recorded (this counts
    what the CORRECT implementation does, as a guard against a future
    accidental deletion of that block).
    """
    bscorer = _BscoreDouble()
    ssb = _make_uninitialized(bscorer, n_subsamples=4, low_dev_pressure=1.0, by_time=False)
    randGen().seed(2)
    ssb("tree-f")
    call_count = len([c for c in bscorer.calls if c[0] == "call"])
    # n_subsamples calls + exactly one final recompute call.
    assert call_count == 4 + 1
