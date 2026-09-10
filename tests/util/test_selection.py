"""Unit tests for pymetta.util.selection.roulette_select and
tournament_selection.

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

`tournament_selection` was added by the P2 optimizer-port task
(star-anneal/univariate/particle-swarm; see `pymetta/util/selection.py`'s
own module docstring) for `univariate.cc:81`'s one reachable call site.
Its own tests pin draw ORDER against a scripted RNG -- `t_size` draws per
tournament, `n_select` tournaments -- the property this task's fidelity
requirements call out most (`randint_n`, not `randint`).
"""

import json
from pathlib import Path

import pytest

from pymetta.moses.moses.types import _to_score_t, composite_score
from pymetta.util.rng import randGen
from pymetta.util.selection import roulette_select, tournament_selection


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


def test_module_exports_roulette_select_and_tournament_selection():
    """Updated by the P2 optimizer-port task (star-anneal/univariate/
    particle-swarm): `tournament_selection` was added alongside
    `roulette_select` -- see `pymetta/util/selection.py`'s own module
    docstring for why (`univariate.cc:81`'s one reachable call site).
    """
    import pymetta.util.selection as mod

    assert mod.__all__ == ["roulette_select", "tournament_selection"]


# ---------------------------------------------------------------------------
# tournament_selection
# ---------------------------------------------------------------------------


class _ScriptedRng:
    """Not itself a ported C++ symbol -- a `randint_n(n)`-only stand-in
    that returns a scripted sequence of draws, recording every `n` it was
    asked for (so a test can also confirm `d = len(items)` was passed
    correctly, not just the resulting winners).
    """

    def __init__(self, draws: list[int]) -> None:
        self._draws = list(draws)
        self.calls: list[int] = []

    def randint_n(self, n: int) -> int:
        self.calls.append(n)
        return self._draws.pop(0)


def _si(score: float) -> composite_score:
    return composite_score(score)


def test_rejects_non_positive_tournament_size():
    with pytest.raises(ValueError):
        tournament_selection(0)
    with pytest.raises(ValueError):
        tournament_selection(-1)


def test_single_tournament_picks_the_larger_of_two_draws():
    items = [_si(1.0), _si(9.0), _si(3.0)]
    rng = _ScriptedRng([0, 2])  # draws index 0 (1.0), then index 2 (3.0)
    sel = tournament_selection(2, rng)
    dst: list = []
    sel(items, dst, 1)
    assert dst == [items[2]]  # 3.0 beats 1.0
    assert rng.calls == [3, 3]  # d == len(items) passed both times


def test_draw_order_is_t_size_draws_per_tournament_n_select_tournaments():
    """Fidelity requirement: `t_size` draws per tournament (the first
    unconditional, the remaining `t_size - 1` each compared against the
    running winner), `n_select` tournaments in sequence -- total draws
    == `t_size * n_select`, in that exact grouping.
    """
    items = [_si(float(i)) for i in range(5)]
    # 2 tournaments of t_size=3: (idx1=1,2,0) then (idx2=4,3,2)
    rng = _ScriptedRng([1, 2, 0, 4, 3, 2])
    sel = tournament_selection(3, rng)
    dst: list = []
    sel(items, dst, 2)
    assert len(rng.calls) == 6  # t_size * n_select
    # Tournament 1: res=items[1](1.0); tmp=items[2](2.0) -> res=items[2];
    #               tmp=items[0](0.0) -> res stays items[2].
    # Tournament 2: res=items[4](4.0); tmp=items[3](3.0) -> res stays;
    #               tmp=items[2](2.0) -> res stays items[4].
    assert dst == [items[2], items[4]]


def test_a_single_element_tournament_size_never_compares():
    """`t_size=1` means the inner `dorepeat(t_size - 1)` loop runs zero
    times -- exactly one draw per tournament, and the drawn element wins
    unconditionally (no `<` comparison at all, so `randdouble`/comparison
    behaviour of `composite_score` is never exercised here).
    """
    items = [_si(1.0), _si(2.0)]
    rng = _ScriptedRng([1, 0])
    sel = tournament_selection(1, rng)
    dst: list = []
    sel(items, dst, 2)
    assert dst == [items[1], items[0]]
    assert rng.calls == [2, 2]


def test_defaults_to_the_process_wide_rng():
    randGen().seed(7)
    sel = tournament_selection(2)
    assert sel._rng is randGen()


def test_mutation_off_by_one_in_inner_loop_changes_draw_count():
    """Mutation proof: replacing `range(self.t_size - 1)` with
    `range(self.t_size)` (one extra draw per tournament) is exactly the
    kind of off-by-one this fidelity requirement guards against --
    demonstrated directly by running the mutated loop shape inline and
    showing it consumes a DIFFERENT number of draws for the same inputs.
    """
    items = [_si(float(i)) for i in range(3)]
    rng = _ScriptedRng([0, 1, 2, 0, 1, 2])

    def mutant_call(items, dst, n_select, t_size, rng):
        d = len(items)
        for _ in range(n_select):
            res = items[rng.randint_n(d)]
            for _ in range(t_size):  # BUG: should be t_size - 1
                tmp = items[rng.randint_n(d)]
                if res < tmp:
                    res = tmp
            dst.append(res)

    dst_mutant: list = []
    mutant_call(items, dst_mutant, 1, 2, rng)
    assert len(rng.calls) == 3  # mutant draws t_size (=2) + 1 = 3, not 2

    rng2 = _ScriptedRng([0, 1])
    dst_real: list = []
    tournament_selection(2, rng2)(items, dst_real, 1)
    assert len(rng2.calls) == 2  # faithful port draws exactly t_size == 2


# ---------------------------------------------------------------------------
# ScoreT is a real parameter -- local_structure.cc:146 instantiates it at int
# ---------------------------------------------------------------------------
#
# Oracle: tests/fixtures/univariate_rng.json, sections "roulette_counts",
# "roulette_draws", "roulette_int" and "roulette_double". They were captured
# from the REAL `opencog::roulette_select` template (tools/cpp_probe/
# univariate_rng_probe.cc) instantiated once at `ScoreT = int` and once at
# `ScoreT = double`, over the same 26 x 239 (counts x randdouble) grid, so the
# disagreement between the two instantiations is a measurement, not a claim.
#
# Note what the fixture's float arm is and is NOT: it is `ScoreT = double`,
# which is the right oracle for `score_type=float` but NOT for this port's
# DEFAULT `score_type=_to_score_t` (`score_t` is binary32, CARRY-FORWARD.md
# 4.2). `test_default_score_type_really_narrows_to_binary32` measures that
# third arm's own divergence from binary64 rather than pretending they agree.


_ROULETTE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "univariate_rng.json"
_ROULETTE_GOLDEN = json.loads(_ROULETTE_FIXTURE.read_text())
_ROULETTE_COUNTS = _ROULETTE_GOLDEN["roulette_counts"]
_ROULETTE_DRAWS = _ROULETTE_GOLDEN["roulette_draws"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", range(len(_ROULETTE_COUNTS)))
def test_score_type_int_matches_the_real_int_instantiation(case):
    """`score_type=int` must reproduce `ScoreT = int` -- the instantiation
    `local_structure.cc:146` actually gets, since `dtree_node` is
    `typedef vector<int>` (local_structure.h:68). Compared index for index
    across all 239 draws.
    """
    counts = _ROULETTE_COUNTS[case]
    total = sum(counts)
    expected = _ROULETTE_GOLDEN["roulette_int"][case]
    got = [roulette_select(counts, total, _FixedRng(r), score_type=int) for r in _ROULETTE_DRAWS]
    assert got == expected


@pytest.mark.oracle
@pytest.mark.parametrize("case", range(len(_ROULETTE_COUNTS)))
def test_score_type_float_matches_the_real_double_instantiation(case):
    """The other half of the same plumbing: `score_type=float` (a no-op
    conversion, i.e. C++'s `double`) must reproduce `ScoreT = double`. This
    is what shows `score_type` is a genuine template parameter rather than
    an `int`-only special case.
    """
    counts = [float(c) for c in _ROULETTE_COUNTS[case]]
    total = float(sum(counts))
    expected = _ROULETTE_GOLDEN["roulette_double"][case]
    got = [roulette_select(counts, total, _FixedRng(r), score_type=float) for r in _ROULETTE_DRAWS]
    assert got == expected


@pytest.mark.oracle
def test_the_two_instantiations_really_do_disagree():
    """Vacuity guard for the two tests above: if `ScoreT = int` and
    `ScoreT = double` agreed everywhere on this grid, both would pass with
    a single implementation and neither would be evidence of anything.
    MEASURED disagreement, read off the C++ side's own two arms: 1,409 of
    6,214 grid points.
    """
    total = 0
    differing = 0
    for case in range(len(_ROULETTE_COUNTS)):
        int_row = _ROULETTE_GOLDEN["roulette_int"][case]
        dbl_row = _ROULETTE_GOLDEN["roulette_double"][case]
        assert len(int_row) == len(dbl_row) == len(_ROULETTE_DRAWS)
        total += len(int_row)
        differing += sum(1 for a, b in zip(int_row, dbl_row, strict=True) if a != b)
    assert total == 6214
    assert differing == 1409


def test_default_score_type_really_narrows_to_binary32():
    """The default arm is `score_t` (binary32), NOT `double`, so it is a
    THIRD behaviour and the fixture's `roulette_double` is not its oracle.
    Measured here: over the same grid the default arm disagrees with
    `score_type=float` at exactly 17 points, and EVERY one of them is one
    of the `k/20 +- 1e-12` draws the probe adds specifically to sit astride
    an integer boundary -- narrowing the product to binary32 snaps it onto
    that boundary. Both facts are asserted, so neither is prose: a non-zero
    count proves the default is not silently `double`, and the location
    check proves the divergence is the expected rounding one rather than
    something structural.
    """
    # The probe emits 201 evenly spaced draws first, then the 38 boundary
    # ones; anything at index > 200 is a boundary draw.
    differing = []
    for counts in _ROULETTE_COUNTS:
        fcounts = [float(c) for c in counts]
        total = float(sum(counts))
        for i, r in enumerate(_ROULETTE_DRAWS):
            narrowed = roulette_select(fcounts, total, _FixedRng(r))
            widened = roulette_select(fcounts, total, _FixedRng(r), score_type=float)
            if narrowed != widened:
                differing.append(i)
    assert len(differing) == 17
    assert all(i > 200 for i in differing)


@pytest.mark.oracle
def test_int_truncation_can_select_a_zero_mass_bucket():
    """The single most surprising consequence of `ScoreT = int`, pinned
    because it looks like a bug and must NOT be "fixed": when
    `int(total * rand)` truncates to 0, the do-while's first subtraction
    leaves `sum` at 0, `(sum > 0)` is false immediately, and the function
    returns index 0 -- EVEN IF bucket 0 has zero mass. For `counts ==
    [0, 1]` that happens for every `rand` STRICTLY below 1.0, so classic
    picks the empty bucket for all of them; `ScoreT = double` picks bucket
    1 for every `rand > 0`. (`randdouble()` is documented "[0,1]", closed
    at both ends, so `rand == 1.0` is reachable and is the one draw where
    the int arm reaches bucket 1 -- excluded explicitly below rather than
    hand-waved.)
    """
    case = _ROULETTE_COUNTS.index([0, 1])
    int_row = _ROULETTE_GOLDEN["roulette_int"][case]
    dbl_row = _ROULETTE_GOLDEN["roulette_double"][case]
    below_one = [i for i, r in enumerate(_ROULETTE_DRAWS) if r < 1.0]
    assert {int_row[i] for i in below_one} == {0}
    assert {int_row[i] for i, r in enumerate(_ROULETTE_DRAWS) if r >= 1.0} == {1}
    assert set(dbl_row) == {0, 1}
    got = [roulette_select([0, 1], 1, _FixedRng(r), score_type=int) for r in _ROULETTE_DRAWS]
    assert got == int_row


def test_mutation_dropping_int_truncation_is_caught_by_the_oracle():
    """Mutation proof for `score_type`: run the SAME loop with a float
    conversion in place of `int` and count the disagreements with the C++
    `int` oracle. A non-zero count is the proof that the oracle tests above
    would go red if `local_structure` stopped passing `score_type=int`.
    """
    mismatches = 0
    for case, counts in enumerate(_ROULETTE_COUNTS):
        total = sum(counts)
        expected = _ROULETTE_GOLDEN["roulette_int"][case]
        mutant = [
            roulette_select(
                [float(c) for c in counts], float(total), _FixedRng(r), score_type=float
            )
            for r in _ROULETTE_DRAWS
        ]
        mismatches += sum(1 for a, b in zip(mutant, expected, strict=True) if a != b)
    assert mismatches == 1409
