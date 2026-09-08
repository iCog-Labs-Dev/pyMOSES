"""C++: opencog/util/selection.h

Ports `roulette_select` -- the sole overload `metapopulation::select_exemplar`
(S11 Task 1, `metapopulation.py`) actually calls -- ONLY the four-argument
form that takes an explicit `sum` (selection.h:88-101):

    template<typename It, typename ScoreT>
    It roulette_select(It from, It to, ScoreT sum, RandGen& rng = randGen())

NOT ported: the three-argument overload (selection.h:104-110) that computes
`sum` itself via `std::accumulate` -- `select_exemplar` always passes its own
already-computed `sum` explicitly (metapopulation.cc:186-187), so that
overload has zero reachable call sites anywhere this whole port's scope
touches (verified: `git grep 'roulette_select'` over `$CLASSIC/moses` finds
exactly one call site, `metapopulation.cc:186`, and it is the four-argument
form). `tournament_selection` (selection.h:39-70) is a different algorithm
entirely with its own zero call sites in this port's scope and is not
touched here either.

# DIVERGENCE(C++ opencog/util/selection.h:89): the real function is
# a template over `It` (any iterator type) and `ScoreT` (any type
# `roulette_select`'s own arithmetic works against). This port specializes
# both to exactly the one instantiation `select_exemplar` needs -- a Python
# `Sequence[score_t]` in place of an iterator PAIR, and `score_t` (`float`,
# narrowed via `_to_score_t`) in place of a generic `ScoreT` -- rather than
# reproducing C++ template genericity Python has no equivalent mechanism
# for. Python has no iterator arithmetic (`from++`, `--from`,
# `std::distance`) either, so this returns the WINNING INDEX directly (the
# same index `select_exemplar`'s own `std::distance(probs.begin(), ...)`
# call immediately re-derives from the iterator C++ returns) -- the direct
# analogue of "the position `--from` ends up pointing at", not a behavioural
# difference: `select_exemplar` is this function's only caller in scope, and
# it needs exactly that index either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pymetta.moses.moses.types import _to_score_t, score_t
from pymetta.util.rng import randGen

if TYPE_CHECKING:
    from pymetta.util.rng import MT19937RandGen

__all__ = ["roulette_select"]


def roulette_select(
    items: Sequence[score_t],
    total: score_t,
    rng: MT19937RandGen | None = None,
) -> int:
    """C++: opencog/util/selection.h:89 roulette_select

    "Select an iterator randomly from the interval [from, to), according to
    the probability distribution described by the elements ... The higher
    the ScoreT, the higher the probability of being selected." (selection.h:
    77-85's own doc comment, condensed.) `items` stands in for the whole
    `[from, to)` range -- see module docstring for why the two-endpoint
    generality is dropped. Returns the selected index into `items`.

    `sum = ScoreT(double(sum) * rng.randdouble());` (selection.h:90) is a
    `score_t` NARROWING conversion -- `ScoreT` is `score_t` at this port's
    one instantiation (module docstring) -- ported via `_to_score_t`, the
    same convention every other `score_t`-typed arithmetic step in this
    port's scope follows (CARRY-FORWARD.md 4.2). `double(sum)` is a no-op
    widening in Python (every value here already IS a Python `float`,
    i.e. what C++ calls `double`); `rng.randdouble()` is already `double`
    in both languages, so the product is computed at double precision and
    narrowed ONCE, matching C++'s own single narrowing at the assignment
    (not per-operand).

    The loop itself:

        do {
            sum -= *from++;
        } while ((sum > 0) && (from != to));
        return --from;

    is a DO-WHILE -- the body (one subtraction) always runs at least once,
    even for a single-element range -- ported as a Python
    `while True: ... ; if not (...): break` with the C++ post-increment's
    two effects (use the CURRENT element, then advance) kept in that exact
    order via a plain index cursor. `sum -= *from++` is `score_t -= score_t`
    -- another float32 operation, narrowed via `_to_score_t` after every
    single subtraction (not once at the end): with `items` holding many
    small values, each individual subtraction's rounding is what a real
    `score_t sum` accumulator would actually store before the NEXT
    subtraction reads it, so narrowing only at the end would let
    double-precision precision leak into intermediate comparisons the `> 0`
    check below makes every iteration.

    `(sum > 0)`, NOT `(sum >= 0)` -- selection.h:92-95's own comment
    explains why: "For integer-valued arrays, it can happen that *from are
    all zero just at the point where sum == 0, which causes the above to
    increment to the end of the array." Ported verbatim, `> 0` and nothing
    else -- flipping this to `>= 0` is exactly the bug that comment warns
    against, and is covered by a mutation test (see test_selection.py).
    """
    if rng is None:
        rng = randGen()

    n = len(items)
    remaining: score_t = _to_score_t(_to_score_t(total) * rng.randdouble())

    cursor = 0
    while True:
        remaining = _to_score_t(remaining - items[cursor])
        cursor += 1
        if not (remaining > 0 and cursor != n):
            break

    return cursor - 1
