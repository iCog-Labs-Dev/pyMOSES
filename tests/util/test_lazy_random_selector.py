"""Tests for pymetta.util.lazy_random_selector -- the Floyd-style lazy
selector `sample_from_neighborhood` draws its knob indices through.
"""

import pytest

from pymetta.util.lazy_random_selector import lazy_random_selector, lazy_selector
from pymetta.util.rng import MT19937RandGen, randGen


class _FixedSelector(lazy_selector):
    """A `lazy_selector` whose `select()` returns a pre-scripted candidate
    sequence instead of drawing from an RNG, so the base class's Floyd's-
    algorithm bookkeeping can be tested in isolation and by hand.
    """

    def __init__(self, u: int, candidates: list[int], lo: int = 0) -> None:
        super().__init__(u, lo)
        self._candidates = list(candidates)

    def select(self) -> int:
        return self._candidates.pop(0)


def test_constructor_rejects_empty_range():
    with pytest.raises(ValueError):
        lazy_selector(3, 3)
    with pytest.raises(ValueError):
        lazy_selector(3, 5)


def test_empty_and_count_n_free():
    sel = _FixedSelector(5, [0, 1, 2, 3, 4])
    assert not sel.empty()
    assert sel.count_n_free() == 5
    for _ in range(5):
        sel()
    assert sel.empty()
    assert sel.count_n_free() == 0


def test_call_on_empty_raises():
    sel = _FixedSelector(1, [0])
    sel()
    assert sel.empty()
    with pytest.raises(ValueError):
        sel()


def test_reset_range_rejects_lower_l():
    sel = _FixedSelector(5, [])
    sel.reset_range(7, 2)
    assert sel._u == 7
    assert sel._l == 2
    with pytest.raises(ValueError):
        sel.reset_range(7, 1)


def test_floyd_algorithm_hand_traced_sequence():
    """Hand-traced by the docstring's own algorithm description:

    u=5, l=0, candidates = [2, 2, 0, 4, 1].
      draw 2: free -> returns 2.                     picked={2}         l=0
      draw 2: not free -> returns l=0, l advances     picked={2,0}       l=1
        (l=1 is free, stop there)
      draw 0: not free -> returns l=1, l advances     picked={2,0,1}     l=3
        (l=2 not free (in picked), l=3 free, stop)
      draw 4: free -> returns 4.                      picked={2,0,1,4}   l=3
      draw 1: not free -> returns l=3, l advances     picked={0,1,2,3,4} l=5
        (l=4 not free, l=5 free (out of range but unpicked), stop)

    Expected output: [2, 0, 1, 4, 3] -- a full permutation of range(5).
    """
    sel = _FixedSelector(5, [2, 2, 0, 4, 1])
    out = [sel() for _ in range(5)]
    assert out == [2, 0, 1, 4, 3]
    assert sorted(out) == list(range(5))


def test_no_replacement_over_full_range():
    # Every candidate the same value: forces every draw after the first to
    # collide, exercising the _l-advance path repeatedly.
    u = 8
    sel = _FixedSelector(u, [3] * u)
    out = [sel() for _ in range(u)]
    assert sorted(out) == list(range(u))
    assert len(set(out)) == u


def test_lazy_random_selector_draws_via_randint_n():
    rng = MT19937RandGen(42)
    oracle = MT19937RandGen(42)
    sel = lazy_random_selector(100, rng)
    # No collisions expected at these odds; every draw should come straight
    # from randint_n(u) in order, proving select() wires to the real RNG.
    drawn = [sel() for _ in range(5)]
    expected = [oracle.randint_n(100) for _ in range(5)]
    assert drawn == expected


def test_lazy_random_selector_two_arg_overload_dispatches_rng_not_l():
    # C++: lazy_random_selector(unsigned u, RandGen& rng = randGen()) --
    # the second positional argument here must be treated as rng, not l.
    randGen().seed(7)
    sel = lazy_random_selector(50, randGen())
    assert sel._l == 0
    assert sel._rng is randGen()


def test_lazy_random_selector_three_arg_overload_sets_l():
    rng = MT19937RandGen(1)
    sel = lazy_random_selector(50, 10, rng)
    assert sel._l == 10
    assert sel._rng is rng


def test_lazy_random_selector_defaults_to_process_singleton():
    randGen().seed(3)
    sel = lazy_random_selector(10)
    assert sel._rng is randGen()
