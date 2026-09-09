"""C++: moses/moses/eda/replacement.h

`replace_the_worst` only -- see `eda/__init__.py`'s module docstring for why
`rtr_replacement` is not ported (its sole call site in scope,
`univariate.cc:83-84`, is commented out).
"""

from __future__ import annotations

from collections.abc import Sequence

from pymetta.moses.representation.scored_instance import scored_instance

__all__ = ["replace_the_worst"]


class replace_the_worst:
    """C++: moses/moses/eda/replacement.h:41 replace_the_worst

    "Replace the lowest-scored individuals in the current population,
    specified by the iterators Dst, and replace them unconditionally with
    the individuals given by NewInst. The container holding the Dst
    iterators will be partially sorted." (the header's own comment.)
    """

    def __call__(
        self, new_insts: Sequence[scored_instance], dst: list[scored_instance]
    ) -> None:
        """C++: moses/moses/eda/replacement.h:44 operator()

        `dst` stands in for the `[from_dst, to_dst)` iterator PAIR C++
        takes: the whole destination population, mutated in place (matching
        `field_set.py`'s own "iterator pair -> Python sequence" convention).
        `new_insts` stands in for `[from, to)`.

        C++: `OMP_ALGO::nth_element(from_dst, from_dst + distance(from, to),
        to_dst); copy(from, to, from_dst);` -- `std::nth_element` partitions
        `dst` so the `len(new_insts)` SMALLEST elements (by
        `scored_instance::operator<`, i.e. worst-scoring first) land at the
        FRONT, in UNSPECIFIED order among themselves, then `copy` overwrites
        exactly that front slice with `new_insts`.

        # DIVERGENCE(C++ moses/moses/eda/replacement.h:54): `nth_element`'s
        # partition order among the `k` selected (worst) elements is
        # unspecified in the standard, and this port does not reproduce
        # libstdc++'s exact introselect permutation the way `merging.py`'s
        # `_sort`/`hill_climbing.py`'s `_partial_sort` do for `std::sort`/
        # `std::partial_sort` (docs/CARRY-FORWARD.md 5b.1, 5.2) -- ONLY
        # because it does not need to: every element in that front slice is
        # about to be UNCONDITIONALLY OVERWRITTEN by `copy(from, to,
        # from_dst)` on the very next line, so which of the `k` worst
        # elements ends up at which of the `k` front POSITIONS is never
        # observed -- unlike `resize_deme`'s truncating `partial_sort`
        # (CARRY-FORWARD.md 5b.1's own counterexample), nothing here reads
        # the partitioned range before it is destroyed. `sorted(dst,
        # key=...)`'s own tie order among the k worst is therefore provably
        # unobservable here, so a plain Python sort suffices: this is
        # exactly the "the result is not truncated or otherwise
        # order-sensitive" exception CARRY-FORWARD.md 5b.1 itself carves out
        # for `_partial_sort`, applied to the analogous `nth_element` case.
        # What DOES matter, and is preserved exactly: WHICH `k` elements
        # (by value, not by position) are selected as "the worst" -- that
        # set is well-defined by `operator<` regardless of tie order, and
        # `new_insts` overwrites precisely those `k` slots, positions
        # `[0, k)` after the sort below.
        """
        k = len(new_insts)
        # DIVERGENCE(C++ moses/moses/eda/replacement.h:46-47): OC_ASSERT
        # becomes ValueError.
        if k > len(dst):
            raise ValueError(
                "ERROR: The number of individuals to replace in a population\n"
                "is larger than the population itself."
            )
        dst.sort()
        for i in range(k):
            dst[i] = new_insts[i]
