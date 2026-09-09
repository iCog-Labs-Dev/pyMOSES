"""C++: moses/moses/eda/termination.h

`terminate_if_gte` and `terminate_if_gte_or_no_improv` -- termination
policies for `optimize.py`'s generic EDA loop. Both are C++ templates over
`ScoreT`; `univariate.cc:76-80` instantiates the second one with
`ScoreT = composite_score`, which is the only instantiation this port needs
(`git grep -rn 'terminate_if_gte' $CLASSIC/moses` finds no other call site).
Ported as plain classes over a `from`/`to` pair of `scored_instance` objects
(a Python list slice stand-in for the C++ iterator range), matching the
project's existing "iterator PAIR -> Python sequence" convention
(`pymetta/util/selection.py`'s own module docstring for `roulette_select`).
"""

from __future__ import annotations

from collections.abc import Sequence

from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.scored_instance import scored_instance

__all__ = ["terminate_if_gte", "terminate_if_gte_or_no_improv"]


class terminate_if_gte:
    """C++: moses/moses/eda/termination.h:37 terminate_if_gte

    "Terminate iteration if the score is greater than the indicated bound."
    (the header's own comment.) Not itself reached by `univariate.cc`
    (which uses `terminate_if_gte_or_no_improv` below), ported alongside it
    since both structs live in the same 56-line header.
    """

    def __init__(self, bound: composite_score) -> None:
        """C++: moses/moses/eda/termination.h:39 terminate_if_gte"""
        self.bound = bound

    def __call__(self, items: Sequence[scored_instance]) -> bool:
        """C++: moses/moses/eda/termination.h:42 operator()

        `*std::max_element(from, to) >= bound` compares the dereferenced max
        element -- literally a whole `scored_instance<ScoreT>` in C++'s own
        template -- against a bare `ScoreT bound`, which only compiles for a
        `ScoreT` implicitly convertible from `scored_instance`, or for a
        range whose value_type already IS `ScoreT`. `git grep -rn
        'terminate_if_gte[^_]' $CLASSIC/moses` finds zero instantiations of
        THIS struct anywhere in scope (only its sibling
        `terminate_if_gte_or_no_improv` below is ever used, by
        `univariate.cc`) -- there is no real call site to pin the intended
        `It`/`ScoreT` pairing down, so this is ported using the one
        interpretation that is actually well-typed for this port's own
        `ScoreT = composite_score`: the max INDIVIDUAL's own `.second`
        against `bound`, matching `terminate_if_gte_or_no_improv`'s
        unambiguous `res >= bound` immediately below.
        """
        best = max(items)
        return not (best.second < self.bound)


class terminate_if_gte_or_no_improv:
    """C++: moses/moses/eda/termination.h:55 terminate_if_gte_or_no_improv

    "Terminate the iteration, if the score is greater than the indicated
    bound, or if there is no forward progress. This class 'remembers'
    previous high scores, and uses these to determine if progress is being
    made." (the header's own comment.) `best_seen` and `at` are C++
    `mutable` members, updated from inside a `const operator()` -- ported
    as plain instance attributes mutated directly, since Python has no
    separate `mutable`/`const` mechanism to preserve.
    """

    def __init__(self, bound: composite_score, n: int) -> None:
        """C++: moses/moses/eda/termination.h:57 terminate_if_gte_or_no_improv

        `at(-1)` is the C++ sentinel meaning "no generation observed yet".
        `best_seen` has no entry in the member-init list at all
        (termination.h:57), but it is a class-typed member (`ScoreT
        best_seen`, not a POD) -- C++ therefore default-CONSTRUCTS it, not
        leaves it as garbage. For this port's sole instantiation
        (`ScoreT = composite_score`), that default construction is
        `composite_score()`, the worst possible score -- ported explicitly
        here rather than as Python `None`, so `res > self.best_seen`
        (below) is a real `composite_score` comparison from the very first
        call, matching what C++ actually evaluates on generation 0.
        """
        self.bound = bound
        self.n_gen = n
        self.best_seen: composite_score = composite_score()
        self.at = -1

    def __call__(self, items: Sequence[scored_instance]) -> bool:
        """C++: moses/moses/eda/termination.h:60 operator()

        `res = std::max_element(from, to)->second` -- the WHOLE
        `composite_score` of the best-scoring individual (`ScoreT = res`),
        not merely its raw score; every comparison below (`res > best_seen`,
        `best_seen >= bound`) therefore goes through `composite_score`'s own
        `<`-derived comparisons, matching `terminate_if_gte` above.

        `at++ > n_gen` is a C++ POST-increment: the comparison reads the
        CURRENT value of `at`, and `at` is incremented exactly once either
        way (unconditionally, before `or`'s short-circuiting is even
        evaluated, since `at++` is itself the left operand's own
        expression) -- ported as a bind-then-increment pair, not
        `self.at += 1` folded into the comparison.
        """
        res = max(items).second
        if res > self.best_seen or self.at == -1:
            self.best_seen = res
            self.at = 0
            return not (res < self.bound)
        at = self.at
        self.at += 1
        return at > self.n_gen or not (self.best_seen < self.bound)
