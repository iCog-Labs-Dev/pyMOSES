"""Unit tests for pymetta.util.selection.roulette_select.

No C++ probe of its own: `roulette_select`'s only real call site anywhere
this port reaches is `metapopulation::select_exemplar` (metapopulation.cc:
186-187), and that call is exercised end-to-end, bit-for-bit, against a
real classic `local_moses()` run in test_local_moses.py's own oracle tests
(`tools/cpp_probe/local_moses_probe.cc`). This file instead pins the three
properties selection.h's own source comments call out explicitly -- the
`> 0` (not `>= 0`) comparison, the do-while shape, and the `sum =
ScoreT(...)` narrowing -- with a fake RNG whose `randdouble()` is fully
controlled, since reverse-engineering a real MT19937 draw to land exactly
on a boundary value is unnecessary when the function already accepts any
object shaped like `MT19937RandGen` (duck typing, matching every other
`rng: RandGen = randGen()` parameter in this port).
"""

from pymetta.moses.moses.types import _to_score_t
from pymetta.util.selection import roulette_select


class _FixedRng:
    """Not itself a ported C++ symbol -- a `randdouble()`-only stand-in,
    the minimum surface `roulette_select` actually calls.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def randdouble(self) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Basic selection arithmetic
# ---------------------------------------------------------------------------


def test_selects_by_cumulative_threshold():
    # sum=4, rand=0.5 -> threshold 2.0: item0 leaves 1.0 (>0, continue),
    # item1 leaves 0.0 (not >0, stop) -> winning index 1.
    items = [1.0, 1.0, 1.0, 1.0]
    assert roulette_select(items, 4.0, _FixedRng(0.5)) == 1


def test_selects_first_item_for_small_rand():
    items = [1.0, 1.0, 1.0, 1.0]
    assert roulette_select(items, 4.0, _FixedRng(0.01)) == 0


def test_selects_last_item_for_rand_near_one():
    items = [1.0, 1.0, 1.0, 1.0]
    # threshold 3.99999: after 3 items, remaining = 3.99999 - 3.0 = 0.99999
    # (>0, and cursor(3) != n(4)) -> must consume the 4th item too.
    assert roulette_select(items, 4.0, _FixedRng(0.999999)) == 3


def test_all_weight_on_one_item_is_always_selected_regardless_of_rand():
    # `r=0.0` is deliberately excluded: it lands the pre-subtraction
    # `remaining` exactly on the zero-weight FIRST bucket's own boundary
    # (`total * 0.0 == 0.0`), and the do-while's mandatory first
    # subtraction (of `items[0] == 0.0`) leaves `remaining` still exactly
    # `0.0` -- `> 0` correctly stops right there, at index 0, a genuine
    # (if unintuitive) edge case of the real algorithm, not a bug this
    # test should assert against.
    items = [0.0, 0.0, 5.0, 0.0]
    for r in (0.05, 0.3, 0.7, 0.999999):
        assert roulette_select(items, 5.0, _FixedRng(r)) == 2


# ---------------------------------------------------------------------------
# Default RNG plumbing (rng=None -> pymetta.util.rng.randGen())
# ---------------------------------------------------------------------------


def test_default_rng_is_reproducible_for_a_fixed_seed():
    from pymetta.util.rng import randGen

    items = [1.0, 2.0, 3.0]
    randGen().seed(7)
    first = roulette_select(items, 6.0)
    randGen().seed(7)
    second = roulette_select(items, 6.0)
    assert first == second


# ---------------------------------------------------------------------------
# `> 0`, NOT `>= 0` (selection.h:92-95's own comment) -- mutation-pinned
# ---------------------------------------------------------------------------


def test_boundary_sum_exactly_zero_stops_immediately_not_wrap_to_end():
    """selection.h's own comment: "For integer-valued arrays, it can
    happen that *from are all zero just at the point where sum == 0,
    which causes [a `>=0` test] to increment to the end of the array."
    `items = [0.5, 0.5, 0.0, 0.0]`, `total = 1.0`, `rand = 0.5` -- all
    three values exactly representable in float32, so every intermediate
    `_to_score_t` narrowing below is a no-op rounding (no residue to
    obscure the boundary): `remaining` hits EXACTLY 0.0 after consuming
    `items[0]` alone -- `> 0` must stop right there (index 0); `>= 0`
    would consume `items[1]` too and return index 1 instead. Mutating the
    source's `remaining > 0` to `remaining >= 0` flips this test's
    expected result -- see this project's task report for the mutation
    actually applied and reverted.
    """
    items = [0.5, 0.5, 0.0, 0.0]
    assert roulette_select(items, 1.0, _FixedRng(0.5)) == 0


# ---------------------------------------------------------------------------
# Do-while shape: the body always runs at least once
# ---------------------------------------------------------------------------


def test_single_element_always_wins_even_when_scaled_sum_is_nonpositive():
    """A do-while ALWAYS performs its one subtraction before ever testing
    the loop condition; a (buggy) pre-check `while remaining > 0 and
    cursor != n:` rewrite would see `remaining <= 0` before the loop ever
    starts (rand=0.0 drives the pre-scaled `remaining` to exactly 0.0) and
    exit with `cursor` still at 0, returning `cursor - 1 == -1` -- an
    invalid index. The real do-while (and this port's `while True: ...;
    if not (...): break`) instead performs the subtraction FIRST
    regardless, giving `remaining = 0.0 - 5.0 = -5.0`, `cursor = 1`, and a
    now-true `not (remaining > 0 and cursor != n)` (`cursor == n == 1`) --
    stopping with the only valid index, 0.
    """
    items = [5.0]
    assert roulette_select(items, 5.0, _FixedRng(0.0)) == 0


def test_single_element_wins_regardless_of_rand_value():
    items = [3.0]
    for r in (0.0, 0.25, 0.5, 0.999999):
        assert roulette_select(items, 3.0, _FixedRng(r)) == 0


# ---------------------------------------------------------------------------
# `score_t` narrowing (`sum = ScoreT(double(sum) * rng.randdouble())`,
# and each `sum -= *from++` narrowed the same way, per this module's own
# docstring)
# ---------------------------------------------------------------------------


def test_returns_plain_int_index_not_a_score_t_value():
    items = [1.0, 1.0]
    result = roulette_select(items, 2.0, _FixedRng(0.9))
    assert isinstance(result, int)


def test_narrows_total_times_rand_to_score_t():
    """`_to_score_t(_to_score_t(total) * rng.randdouble())` -- if this
    were computed at double precision and narrowed only implicitly (or not
    at all), a `total`/`rand` pair whose double-precision product sits
    exactly ON a value that float32 rounds DOWN across a candidate's
    boundary would select a different index than the narrowed computation
    does. `total = 2.0**24 + 1` (the smallest positive double NOT exactly
    representable as float32 -- float32 has a 24-bit significand) is a
    convenient probe: multiplied by `rand = 1.0`, double precision leaves
    it unchanged, but `_to_score_t` rounds it to `2.0**24` -- a change the
    unnarrowed path could never produce.
    """
    total = float(2**24 + 1)
    items = [float(2**24), 2.0]
    rng = _FixedRng(1.0)

    # Narrowed: _to_score_t(total) == 2**24 exactly (rounds the odd value
    # down), so `remaining = 2**24 - items[0] == 0.0` after one step --
    # `> 0` is False immediately, do-while already ran once -> index 0.
    assert _to_score_t(total) == float(2**24)
    assert roulette_select(items, total, rng) == 0


def test_narrows_the_product_result_not_only_its_operands():
    """Pins the OUTER `_to_score_t` in `_to_score_t(_to_score_t(total) *
    rng.randdouble())` -- the case above (`test_narrows_total_times_rand_
    to_score_t`) only exercises the INNER `_to_score_t(total)`: for that
    test's own values, `_to_score_t(total)` alone already lands on a value
    exactly representable in float32 (`2.0**24`), so multiplying by
    `rand=1.0` needs no further rounding and the outer narrowing is never
    actually exercised -- stripping ONLY the outer wrap still passes that
    test. This is the reviewer's own counterexample for selection.py:106:
    `total = 135229872.0`, `r = 0.8474337369372327`,
    `items = [114598352.0, 1.0]` -- narrowing the PRODUCT
    (`total * rand == 114598349.60148...` at double precision) rounds it to
    float32 `114598352.0`, exactly equal to `items[0]`, so `remaining <= 0`
    immediately (index 0); left at double precision, the product stays
    below `items[0]` by a hair (`remaining > 0`), consuming a second
    element (index 1). Stripping only the outer `_to_score_t` here (keeping
    the inner `_to_score_t(total)`) reproduces exactly that: index 1 instead
    of 0.
    """
    items = [114598352.0, 1.0]
    total = 135229872.0
    rng = _FixedRng(0.8474337369372327)
    assert roulette_select(items, total, rng) == 0


def test_narrows_each_loop_subtraction_not_only_the_final_sum():
    """Pins the `_to_score_t` in `remaining = _to_score_t(remaining -
    items[cursor])` (selection.h:90's loop body) using the reviewer's own
    measured counterexample: `items = [0.6] * 20 + [99999992.0, 1.0]`,
    `total = 266994096.0`, real `MT19937RandGen` seeded 42. Compiled C++
    `roulette_select` and this narrowed port both return index 21;
    stripping this narrowing (summing the twenty `0.6`s and the two large
    tail values at double precision, rounding only implicitly) returns
    20 instead -- the twenty small `_to_score_t(remaining - 0.6)` roundings
    accumulate a float32 rounding residue a double-precision subtraction
    never picks up, and that residue is what keeps `remaining > 0` one
    iteration longer.
    """
    from pymetta.util.rng import randGen

    items = [0.6] * 20 + [99999992.0, 1.0]
    total = 266994096.0
    randGen().seed(42)
    assert roulette_select(items, total) == 21


# ---------------------------------------------------------------------------
# Vacuity guard
# ---------------------------------------------------------------------------


def test_module_exports_only_roulette_select():
    import pymetta.util.selection as mod

    assert mod.__all__ == ["roulette_select"]
