"""C++: opencog/util/selection.h

Ports `roulette_select` -- the sole overload `metapopulation::select_exemplar`
(S11 Task 1, `metapopulation.py`) actually calls -- ONLY the four-argument
form that takes an explicit `sum` (selection.h:88-101):

    template<typename It, typename ScoreT>
    It roulette_select(It from, It to, ScoreT sum, RandGen& rng = randGen())

NOT ported: the three-argument overload (selection.h:104-110) that computes
`sum` itself via `std::accumulate` -- `select_exemplar` always passes its own
already-computed `sum` explicitly (metapopulation.cc:185-187), so that
overload has zero reachable call sites anywhere this whole port's scope
touches. `git grep 'roulette_select' $CLASSIC/moses` actually finds TWO call
sites, not one: `metapopulation.cc:185` (this port's own, the four-argument
form with an explicit `rng`) and `moses/moses/eda/local_structure.cc:146`
(also the four-argument form, `roulette_select(dtr->begin(), --(dtr->end()),
dtr->back())` -- three arguments AT THE CALL SITE because `rng` takes its
default there, not a different overload). At the time this note was
written the second call site was outside every file this whole port's
scope touched; the P2 optimizer-port task that added `local_structure.py`
(`pymoses/moses/eda/`) reuses `roulette_select` for exactly that call site
rather than re-porting it -- see that module's own docstring.

`tournament_selection` (selection.h:39-70) WAS also outside scope when the
note above was written ("a different algorithm entirely with its own zero
call sites in this port's scope, not touched here either"). The same P2
optimizer-port task added it below: `univariate.cc:81` is its one reachable
call site (`tournament_selection((unsigned)eda_params.selection)`), needed
because `eda_parameters` defaults to tournament selection
(`eda_parameters::selection(2)`, `is_tournament_selection()` returning
`selection > 1`) and `univariate_optimization::operator()`'s ELSE branch
(truncation selection) is `OC_ASSERT(false, "Trunction selection not
implemented...")` in classic itself -- i.e. truncation selection is
unreachable in ANY real classic build, not merely unported here.

# DIVERGENCE(C++ opencog/util/selection.h:89): the real function is
# a template over `It` (any iterator type) and `ScoreT` (any type
# `roulette_select`'s own arithmetic works against). This port specializes
# `It` to a Python `Sequence` in place of an iterator PAIR, rather than
# reproducing C++ template genericity Python has no equivalent mechanism
# for. Python has no iterator arithmetic (`from++`, `--from`,
# `std::distance`) either, so this returns the WINNING INDEX directly (the
# same index `select_exemplar`'s own `std::distance(probs.begin(), ...)`
# call immediately re-derives from the iterator C++ returns) -- the direct
# analogue of "the position `--from` ends up pointing at", not a behavioural
# difference: both callers need exactly that index either way.
#
# `ScoreT` is NOT specialized away: it is passed explicitly as `score_type`,
# because the two live instantiations are genuinely different arithmetic.
# `select_exemplar` instantiates `ScoreT = score_t` (`float`, binary32);
# `local_structure.cc:146` instantiates `ScoreT = int`, because `dtree_node`
# is `typedef vector<int>` (`local_structure.h:68`) and `dtr->back()` is
# therefore an `int`. At `ScoreT = int`, `sum = ScoreT(double(sum) *
# rng.randdouble())` TRUNCATES TOWARD ZERO and the whole loop runs in exact
# integer arithmetic; at `ScoreT = score_t` it narrows to binary32 and the
# loop runs in float. MEASURED 2026-09-10 over the 6,214-point grid in
# `tests/fixtures/univariate_rng.json` (26 leaf-count vectors x 239
# `randdouble` values, both instantiations dumped from the REAL C++
# template by `tools/cpp_probe/univariate_rng_probe.cc`): the two
# instantiations return a DIFFERENT index at 1,409 of those 6,214 points.
# Collapsing them to one was a real behavioural bug on the `-aun` path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pymoses.moses.moses.types import _to_score_t
from pymoses.util.rng import randGen

if TYPE_CHECKING:
    from pymoses.util.rng import MT19937RandGen

__all__ = ["roulette_select", "tournament_selection"]


def roulette_select(
    items: Sequence[float],
    total: float,
    rng: MT19937RandGen | None = None,
    score_type: Callable[[float], float] = _to_score_t,
) -> int:
    """C++: opencog/util/selection.h:89 roulette_select

    "Select an iterator randomly from the interval [from, to), according to
    the probability distribution described by the elements ... The higher
    the ScoreT, the higher the probability of being selected." (selection.h:
    77-85's own doc comment, condensed.) `items` stands in for the whole
    `[from, to)` range -- see module docstring for why the two-endpoint
    generality is dropped. Returns the selected index into `items`.

    `score_type` IS the template parameter `ScoreT`: the conversion C++
    applies at every step where the standing `sum` is written back. It
    defaults to `_to_score_t` (`ScoreT = score_t`, `select_exemplar`'s
    instantiation); `local_structure.py` passes `int` (`ScoreT = int`,
    `local_structure.cc:146`'s instantiation, whose `dtree_node` element
    type is `int`). See the module docstring for the measured size of the
    difference.

    `sum = ScoreT(double(sum) * rng.randdouble());` (selection.h:90) is the
    conversion above applied to a double product. `double(sum)` is a no-op
    widening in Python (every value here already IS a Python `float`, i.e.
    what C++ calls `double`, or an `int`, which Python widens losslessly);
    `rng.randdouble()` is already `double` in both languages, so the product
    is computed at double precision and converted ONCE, matching C++'s own
    single conversion at the assignment (not per-operand). At `ScoreT =
    score_t` that conversion narrows to binary32; at `ScoreT = int` it
    truncates toward zero, which `int()` also does.

    The loop itself:

        do {
            sum -= *from++;
        } while ((sum > 0) && (from != to));
        return --from;

    is a DO-WHILE -- the body (one subtraction) always runs at least once,
    even for a single-element range -- ported as a Python
    `while True: ... ; if not (...): break` with the C++ post-increment's
    two effects (use the CURRENT element, then advance) kept in that exact
    order via a plain index cursor. `sum -= *from++` is `ScoreT -= ScoreT`,
    so `score_type` is applied after every single subtraction, not once at
    the end: with `items` holding many small values, each individual
    subtraction's rounding is what a real `ScoreT sum` accumulator would
    actually store before the NEXT subtraction reads it, so converting only
    at the end would let double-precision precision leak into the
    intermediate comparisons the `> 0` check below makes every iteration.

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
    # `ScoreT sum` is a BY-VALUE parameter, so C++ converts the caller's
    # argument to `ScoreT` on binding before `double(sum)` widens it back;
    # the inner `score_type` reproduces that binding, and is the identity
    # whenever the caller already holds a `ScoreT`-typed value.
    remaining = score_type(score_type(total) * rng.randdouble())

    cursor = 0
    while True:
        remaining = score_type(remaining - items[cursor])
        cursor += 1
        if not (remaining > 0 and cursor != n):
            break

    return cursor - 1


class tournament_selection:
    """C++: opencog/util/selection.h:39 tournament_selection

    "Selects randomly n_select elements in [from, to) such that each
    element is the winner of a random selection of t_size elements in
    [from, to). That is, a tournament is played n_select times. The winner
    of each tournament is the largest element from a set of t_size
    randomly chosen elements." (the header's own doc comment.)

    `items` stands in for the `[from, to)` iterator pair (this port's usual
    convention, see module docstring); `dst` stands in for the `Out dst`
    output iterator -- an `n_select`-entry APPEND target, matching
    `pymoses/moses/moses/neighborhood_sampling.py`'s own append-based output
    convention rather than a pre-sized index-assignment target.
    """

    def __init__(self, t_size: int, rng: MT19937RandGen | None = None) -> None:
        """C++: opencog/util/selection.h:41 tournament_selection

        `_rng` (`rng` in C++) defaults to `randGen()`, matching every other
        RNG-defaulting constructor already ported in this project
        (`lazy_random_selector.__init__`, `rtr_replacement`'s C++ own).
        """
        # DIVERGENCE(C++ opencog/util/selection.h:44): OC_ASSERT becomes
        # ValueError.
        if t_size <= 0:
            raise ValueError("tournament_selection: t_size must be > 0")
        self.t_size = t_size
        self._rng = rng if rng is not None else randGen()

    def __call__(self, items: Sequence, dst: list, n_select: int) -> None:
        """C++: opencog/util/selection.h:60 operator()

        `dorepeat (n_select) { In res = from + rng.randint(d); dorepeat
        (t_size - 1) { In tmp = from + rng.randint(d); if (*res < *tmp) res
        = tmp; } *dst++ = *res; }` -- `rng.randint(d)` is
        `RandGen::randint(int n)`, this port's already-ported
        `MT19937RandGen.randint_n` (`randint() % n`, `pymoses/util/rng.py`),
        NOT the zero-argument `RandGen::randint()` (`raw >> 1`). Draw order
        preserved exactly: `t_size` draws per tournament (the first
        unconditional, the remaining `t_size - 1` each followed by a `<`
        comparison against the running winner), `n_select` tournaments in
        sequence, each appending exactly one winner.
        """
        d = len(items)
        rng = self._rng
        for _ in range(n_select):
            res = items[rng.randint_n(d)]
            for _ in range(self.t_size - 1):
                tmp = items[rng.randint_n(d)]
                if res < tmp:
                    res = tmp
            dst.append(res)
