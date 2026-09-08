"""Floyd-style lazy selection of random integers without replacement.

Ported for `sample_from_neighborhood` (`pymetta/moses/moses/neighborhood_sampling.py`),
which draws its knob-to-vary indices through this exact algorithm -- its draw
order is part of the bit-identical RNG claim S9 exercises for the first time
(see that module's docstring). Two C++ classes are ported here, not one:
`lazy_random_selector` is what the task names, but it is a thin subclass of
`lazy_selector` (a separate, unnamed CogUtil file) that supplies the actual
Floyd's-algorithm bookkeeping -- `lazy_random_selector` alone only knows how
to draw a single candidate index, not how to turn repeated draws into a
selection that never repeats. `lazy_selector` has exactly one other C++
subclass in scope (`lazy_normal_selector`), and it is never used anywhere
under `moses/`, so this port stays with the single subclass this file
actually needs.
"""

from __future__ import annotations

from pymetta.util.rng import MT19937RandGen, randGen


class lazy_selector:
    """C++: opencog/util/lazy_selector.h:41 lazy_selector

    Selects integers in ``[l, u)`` without ever repeating one already
    returned, in O(1) time and O(m) memory for m draws. The trick (Floyd's
    algorithm): draw candidates however `select()` likes from the full range
    ``[l, u)``; if a candidate was already returned, return the current lower
    bound `l` instead (which is always still free by construction) and then
    advance `l` past however many already-returned values now sit at its new
    position. `select()` itself is the one hook a subclass supplies.
    """

    def __init__(self, u: int, lo: int = 0) -> None:
        """C++: opencog/util/lazy_selector.cc:39 lazy_selector::lazy_selector

        ``lo`` is C++'s ``l`` (the lower bound) renamed to satisfy ruff's
        E741 (not part of the ruff naming rules this project's house style
        otherwise disables -- see pyproject.toml's ``ignore`` list, which
        does not include E741).
        """
        # DIVERGENCE(C++ opencog/util/lazy_selector.cc:39-43): OC_ASSERT
        # becomes ValueError.
        if not u - lo > 0:
            raise ValueError("you cannot select any thing from an empty list")
        self._u = u
        self._l = lo
        # DIVERGENCE(C++ opencog/util/lazy_selector.h:73): a std::unordered_set
        # of already-returned indices; Python's set has the same semantics.
        self._picked: set[int] = set()

    def empty(self) -> bool:
        """C++: opencog/util/lazy_selector.cc:45 lazy_selector::empty"""
        return self._l >= self._u

    def count_n_free(self) -> int:
        """C++: opencog/util/lazy_selector.cc:50 lazy_selector::count_n_free"""
        return sum(1 for idx in range(self._l, self._u) if self._is_free(idx))

    def reset_range(self, new_u: int, new_l: int | None = None) -> None:
        """C++: opencog/util/lazy_selector.cc:58 lazy_selector::reset_range
        (``new_l`` omitted) and lazy_selector.cc:63 lazy_selector::reset_range
        (``new_l`` given).
        """
        if new_l is not None:
            # DIVERGENCE(C++ opencog/util/lazy_selector.cc:63-70): OC_ASSERT
            # becomes ValueError.
            if new_l < self._l:
                raise ValueError(
                    "You cannot reset the lower bound by a lower number, "
                    "due to the workings of the algorithm."
                )
            self._l = new_l
        self._u = new_u

    def __call__(self) -> int:
        """C++: opencog/util/lazy_selector.cc:84 lazy_selector::operator()

        The real member is `lazy_selector::operator()` -- an operator
        overload, which `tools/check_alignment.py` cannot match by
        identifier (see CARRY-FORWARD.md 4.5), so the enclosing class is
        cited instead. `__call__` is the natural Python spelling of C++'s
        function-call operator.
        """
        # DIVERGENCE(C++ opencog/util/lazy_selector.cc:86): OC_ASSERT becomes
        # ValueError.
        if self.empty():
            raise ValueError("lazy_selector - selector is empty.")

        sel_idx = self.select()

        # If the selected index points to nothing then the result is
        # itself otherwise it is _l
        res = sel_idx if self._is_free(sel_idx) else self._l

        # Move _l from res so that it is now a free index
        if res == self._l:
            self._increase_l_till_free()

        self._picked.add(res)
        return res

    def select(self) -> int:
        """C++: opencog/util/lazy_selector.h:69 select

        Abstract in C++ (a pure virtual); `lazy_random_selector` below is the
        only concrete subclass this port needs.
        """
        raise NotImplementedError("lazy_selector.select is abstract; use lazy_random_selector")

    def _is_free(self, idx: int) -> bool:
        """C++: opencog/util/lazy_selector.cc:102 lazy_selector::is_free"""
        return idx not in self._picked

    def _increase_l_till_free(self) -> None:
        """C++: opencog/util/lazy_selector.cc:107 lazy_selector::increase_l_till_free"""
        # C++ is a do-while: _l is incremented at least once before the free
        # check, unconditionally.
        while True:
            self._l += 1
            if self._is_free(self._l):
                return


class lazy_random_selector(lazy_selector):
    """C++: opencog/util/lazy_random_selector.h:42 lazy_random_selector

    A lazy selector (see `lazy_selector` above) whose candidate draw is
    uniform over ``[0, u)`` via the process-wide `randGen()` singleton by
    default -- note this draws over ``[0, u)``, not ``[l, u)``; the lower
    bound only ever moves via `lazy_selector`'s own bookkeeping, never
    narrows what `select()` itself draws from. Port faithfully, not "fixed":
    this is exactly what lazy_random_selector.cc:36-39 does.
    """

    def __init__(
        self,
        u: int,
        l_or_rng: int | MT19937RandGen | None = None,
        rng: MT19937RandGen | None = None,
    ) -> None:
        """C++: opencog/util/lazy_random_selector.cc:28
        lazy_random_selector::lazy_random_selector (``l`` omitted, ``rng``
        defaults to ``randGen()``) and lazy_random_selector.cc:32
        lazy_random_selector::lazy_random_selector (``l`` given). One Python
        constructor dispatches on the second positional argument's type
        (an `MT19937RandGen` means the caller used the two-argument C++
        overload and meant it as ``rng``, not ``l``) across both overloads,
        matching the merged-overload convention field_set.py's constructor
        already established.
        """
        if isinstance(l_or_rng, MT19937RandGen):
            lo = 0
            rng = l_or_rng
        else:
            lo = 0 if l_or_rng is None else l_or_rng
        super().__init__(u, lo)
        self._rng = rng if rng is not None else randGen()

    def select(self) -> int:
        """C++: opencog/util/lazy_random_selector.cc:36 lazy_random_selector::select

        `_u` here is a protected `lazy_selector` field re-read afresh on
        every call, so a `reset_range` between draws is honoured exactly the
        way it would be in C++.
        """
        return self._rng.randint_n(self._u)
