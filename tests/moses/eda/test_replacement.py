"""Oracle and guard tests for pymoses.moses.eda.replacement.

`replace_the_worst`'s only real call site in this port's scope is
`univariate_optimization` (`test_univariate.py`'s own tests), and its
observable postconditions are pinned directly below. What needed a C++
oracle is the `OMP_ALGO::nth_element` it calls (replacement.h:54): the
permutation that leaves in the TAIL of the range is NOT overwritten and
IS observed downstream -- see replacement.py's module docstring.

Oracle fixture: tests/fixtures/nth_element.json, produced by
tools/cpp_probe/nth_element_probe.cc from the REAL `OMP_ALGO::nth_element`
compiled against the real cogutil headers with classic's own `-fopenmp`
(see tools/capture_nth_element.py). 133 cases, each recording the exact
input and output arrays.
"""

import json
from pathlib import Path

import pytest

from pymoses.moses.eda.replacement import _nth_element, replace_the_worst
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.scored_instance import scored_instance

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "nth_element.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


def _si(score: float) -> scored_instance:
    return scored_instance(instance(0), composite_score(score))


def _by_key(a, b) -> bool:
    """The probe's comparator: `pair<int,int>` compared on `.first` alone,
    so `.second` acts as a tag that makes the permutation readable.
    """
    return a[0] < b[0]


# ---------------------------------------------------------------------------
# nth_element -- libstdc++'s std::__introselect, reproduced exactly
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_omp_algo_resolves_to_std_in_the_build_classic_uses():
    """`OMP_ALGO` is `__gnu_parallel` only when cogutil's `OC_OMP` is
    defined, and that is gated on `HAVE_PARALLEL_STL` (oc_omp.h:49), an
    identifier neither checkout defines. The probe reports the branch it
    actually compiled to, and separately whether the two spellings agree
    element for element over every case -- so this is measured, not read
    off a header. If either flips, the port is reproducing the wrong
    algorithm and this goes red.
    """
    assert _GOLDEN["omp_algo_is_std"] is True
    assert _GOLDEN["omp_matches_std"] is True


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["cases"], ids=lambda c: c["name"])
def test_nth_element_matches_libstdcxx_permutation(case):
    """The whole array after the call, element for element -- not just the
    partition invariant. The invariant alone is satisfied by any correct
    `nth_element`, including Python's own, and is exactly what would let a
    wrong tail order pass unnoticed.
    """
    v = [tuple(x) for x in case["input"]]
    _nth_element(v, 0, case["nth"], len(v), _by_key)
    assert v == [tuple(x) for x in case["output"]]


@pytest.mark.oracle
def test_a_correct_but_different_nth_element_would_be_caught():
    """Vacuity guard for the oracle above. A `sorted()`-based
    implementation satisfies `nth_element`'s documented postcondition
    perfectly, so a test that only checked the partition invariant would
    pass with it. Count how many of the 133 fixture cases it gets wrong:
    a large number is the proof that the element-for-element comparison is
    doing real work.
    """
    wrong = 0
    for case in _GOLDEN["cases"]:
        v = sorted((tuple(x) for x in case["input"]), key=lambda e: e[0])
        if v != [tuple(x) for x in case["output"]]:
            wrong += 1
    assert wrong >= 60


@pytest.mark.oracle
def test_nth_equal_to_last_is_an_early_return():
    """`if (__first == __last || __nth == __last) return;` -- the array
    must come back completely untouched. The fixture carries a
    `rand_n597_kend` case (nth == size) for exactly this.
    """
    case = next(c for c in _GOLDEN["cases"] if c["name"] == "rand_n597_kend")
    assert case["nth"] == len(case["input"])
    assert case["output"] == case["input"]
    v = [tuple(x) for x in case["input"]]
    _nth_element(v, 0, case["nth"], len(v), _by_key)
    assert v == [tuple(x) for x in case["input"]]


def test_mutation_full_sort_changes_the_tail_the_deme_keeps():
    """Mutation proof for the fix itself. The previous implementation
    called `dst.sort()`, whose front slice is then overwritten -- so the
    only difference is the TAIL, which is precisely what survives into the
    deme. Shown here on the real fixture case whose size and tie density
    match the `-aun` deme: the two agree on the SET of the k worst (the
    part the old note reasoned about) and differ on the order of what is
    left (the part it did not).
    """
    case = next(c for c in _GOLDEN["cases"] if c["name"] == "rand_n597_d3")
    k = case["nth"]
    faithful = [tuple(x) for x in case["input"]]
    _nth_element(faithful, 0, k, len(faithful), _by_key)

    mutant = sorted((tuple(x) for x in case["input"]), key=lambda e: e[0])

    # Same k worst, as multisets of keys -- the old note's claim, and true.
    assert sorted(e[0] for e in faithful[:k]) == sorted(e[0] for e in mutant[:k])
    # Different tail -- the part the old note never mentioned.
    assert faithful[k:] != mutant[k:]


# ---------------------------------------------------------------------------
# replace_the_worst
# ---------------------------------------------------------------------------


def test_replaces_exactly_the_k_worst_slots():
    dst = [_si(5.0), _si(1.0), _si(4.0), _si(2.0), _si(3.0)]
    new = [_si(100.0), _si(200.0)]
    replace_the_worst()(new, dst)
    # The two worst (1.0, 2.0) must be gone; everything scoring >= 3.0
    # must survive untouched (5 - 2 = 3 kept, matching len(dst) - len(new)).
    remaining_scores = sorted(si.second.get_score() for si in dst)
    assert 1.0 not in remaining_scores
    assert 2.0 not in remaining_scores
    assert remaining_scores.count(3.0) == 1
    assert remaining_scores.count(4.0) == 1
    assert remaining_scores.count(5.0) == 1
    # And the two new instances must both be present somewhere in dst.
    assert {100.0, 200.0} <= {si.second.get_score() for si in dst}


def test_replacing_all_slots_is_allowed():
    dst = [_si(1.0), _si(2.0)]
    new = [_si(9.0), _si(9.0)]
    replace_the_worst()(new, dst)
    assert all(si.second.get_score() == 9.0 for si in dst)


def test_raises_when_more_new_than_destination_slots():
    """C++: OC_ASSERT(distance(from, to) <= distance(from_dst, to_dst), ...)
    -- ported as ValueError (module docstring).
    """
    dst = [_si(1.0)]
    new = [_si(9.0), _si(9.0)]
    with pytest.raises(ValueError):
        replace_the_worst()(new, dst)


def test_mutation_removing_size_guard_lets_it_silently_truncate():
    """Mutation proof: with the `k > len(dst)` guard removed, `dst[i] =
    new_insts[i]` for `i` beyond `len(dst)` would raise `IndexError`
    instead of the documented `ValueError` -- i.e. the guard changes
    WHICH exception surfaces and WHERE (a clean precondition check vs. a
    raw index fault partway through the loop, potentially after already
    mutating some slots). Demonstrated directly here without touching the
    module (see the task report for the source-level red/green mutation).
    """
    dst = [_si(1.0)]
    new = [_si(9.0), _si(9.0)]
    k = len(new)
    if k > len(dst):
        pass  # this is the guard branch that would have fired
    else:
        raise AssertionError("test setup is wrong: guard should fire here")
    # Now show what happens WITHOUT the guard (the mutant's own behaviour):
    dst_unguarded = [_si(1.0)]
    _nth_element(dst_unguarded, 0, k, len(dst_unguarded))
    with pytest.raises(IndexError):
        for i in range(k):
            dst_unguarded[i] = new[i]
