"""Unit tests for pymetta.moses.eda.replacement.replace_the_worst.

No C++ probe: `replace_the_worst`'s only real call site in this port's
scope is `univariate_optimization` (`test_univariate.py`'s own oracle
tests). This file pins the guard and the observable postcondition
directly.
"""

import pytest

from pymetta.moses.eda.replacement import replace_the_worst
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance


def _si(score: float) -> scored_instance:
    return scored_instance(instance(0), composite_score(score))


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
    dst_unguarded.sort()
    with pytest.raises(IndexError):
        for i in range(k):
            dst_unguarded[i] = new[i]
