"""C++: moses/moses/eda/replacement.h

`replace_the_worst` only -- see `eda/__init__.py`'s module docstring for why
`rtr_replacement` is not ported (its sole call site in scope,
`univariate.cc:83-84`, is commented out).

`replace_the_worst` calls `OMP_ALGO::nth_element` (replacement.h:54), and
BOTH halves of that call matter:

* `OMP_ALGO` is `__gnu_parallel` only when cogutil's `OC_OMP` is defined,
  which happens only `#elif HAVE_PARALLEL_STL` (`opencog/util/oc_omp.h:49`)
  -- an identifier neither checkout ever defines, so the preprocessor
  evaluates it as 0 and `OMP_ALGO` is plain `std`
  (oc_omp.h:53-60). MEASURED, not read off: `tools/cpp_probe/
  nth_element_probe.cc` compiles against the real cogutil headers with
  classic's own `-fopenmp` flag and reports both which branch it got
  (`omp_algo_is_std`) and whether `OMP_ALGO::nth_element` and
  `std::nth_element` agree element for element over 133 cases
  (`omp_matches_std`). `tests/fixtures/nth_element.json` records both.

* `std::nth_element` only PARTITIONS. `[first, nth)` all compare <= `*nth`
  and `(nth, last)` all compare >= it, with the order WITHIN each part
  unspecified by the standard -- and the TAIL is observed: it stays in the
  deme, `sort_demes`' `std::sort` is introsort (not stable), so the tail's
  incoming permutation decides which of several equal-scoring candidates
  survives `keep_top_unique_candidates`' ADJACENT-duplicate scan
  (merging.cc:568-586), and `tournament_selection` then indexes the deme
  positionally (`selection.h:66`). `_nth_element` below therefore
  reproduces libstdc++'s `std::__introselect` exactly, in the same style
  `merging.py`'s `_sort` and `hill_climbing.py`'s `_partial_sort` already
  use for `std::sort`/`std::partial_sort`.

  This replaced a `dst.sort()`, whose accompanying note argued the order
  was unobservable because `copy` overwrites the front slice. That
  argument was correct about the FRONT and silent about the TAIL, which a
  full sort also reorders. MEASURED cost of that silence: with everything
  else in this task fixed, `-aun` was 18/20 against classic over 5 seeds x
  {500, 1500} evals x {trainingBooleanXor.dat, trainingBoolean.dat}; with
  `_nth_element` it is 20/20. CARRY-FORWARD.md 5b.1's "not truncated or
  otherwise order-sensitive" exception does not apply here: the range is
  not truncated, but it IS order-sensitive.

PORTABILITY CAVEAT, inherited from `_partial_sort` and `_sort`
(docs/CARRY-FORWARD.md 5.1/5b.1): this is pinned to THIS libstdc++'s actual
introselect. A different libstdc++ could change the algorithm shape; the
fixture's schema-2 manifest records the toolchain so that shows up as
drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymoses.moses.metapopulation.merging import (
    _insertion_sort,
    _lg,
    _unguarded_partition_pivot,
)
from pymoses.moses.moses.optimization.hill_climbing import _heap_select
from pymoses.moses.representation.scored_instance import scored_instance

__all__ = ["replace_the_worst"]


def _less(a, b) -> bool:
    """Not itself a ported C++ symbol -- `__gnu_cxx::__ops::less()`, the
    default comparator `std::nth_element(first, nth, last)`
    (bits/stl_algo.h:4748) forwards to `__introselect`. `less()(a, b)` is
    `a < b`, which for `deme_inst_t` = `scored_instance<composite_score>`
    resolves to `scored_instance::operator<` (scored_instance.h:43-46),
    i.e. `a.second < b.second`.

    Written as a free function, and passed explicitly at every call below,
    for the same reason `hill_climbing._greater` is: so the algorithm reads
    as the libstdc++ code it reproduces, with the comparator where
    libstdc++ puts it.
    """
    return a.second < b.second


def _introselect(v, first: int, nth: int, last: int, depth_limit: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__introselect`
    (bits/stl_algo.h:1923):

        _RandomAccessIterator __after_nth = __nth;
        ++__after_nth;
        while (__last - __first > 3) {
            if (__depth_limit == 0) {
                std::__heap_select(__first, __after_nth, __last, __comp);
                std::iter_swap(__first, __nth);
                return;
            }
            --__depth_limit;
            _RandomAccessIterator __cut =
                std::__unguarded_partition_pivot(__first, __last, __comp);
            if (__cut <= __nth) __first = __cut; else __last = __cut;
        }
        std::__insertion_sort(__first, __last, __comp);

    `__after_nth` and `__nth` are ABSOLUTE iterators into the original
    range and are NOT re-derived as `__first`/`__last` narrow -- hence the
    absolute `nth` / `after_nth` indices here against a narrowing
    `first`/`last` pair, which is the exact index-based equivalent.

    The `depth_limit == 0` arm is reproduced but is NOT exercised by any
    case in `tests/fixtures/nth_element.json`: `__lg(n) * 2` is a generous
    budget and median-of-3 pivoting survives every input shape the fixture
    tries (sorted, reversed, organ-pipe, all-equal, and pseudo-random with
    1..50 distinct keys, at sizes up to the real deme's 597). Reported as
    untested rather than claimed as covered -- see CARRY-FORWARD.md 5b.6
    for the same disclosure about `cross_top_two`'s unsigned masks.
    """
    after_nth = nth + 1
    while last - first > 3:
        if depth_limit == 0:
            _heap_select(v, first, after_nth, last, comp)
            # "Place the nth largest element in its final position."
            v[first], v[nth] = v[nth], v[first]
            return
        depth_limit -= 1
        cut = _unguarded_partition_pivot(v, first, last, comp)
        if cut <= nth:
            first = cut
        else:
            last = cut
    _insertion_sort(v, first, last, comp)


def _nth_element(v, first: int, nth: int, last: int, comp=_less) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::nth_element`
    (bits/stl_algo.h:4731, the comparator-less overload):

        if (__first == __last || __nth == __last) return;
        std::__introselect(__first, __nth, __last,
                           std::__lg(__last - __first) * 2,
                           __gnu_cxx::__ops::less());

    Operates on a slice of `v` by index rather than on iterators, matching
    `hill_climbing._partial_sort`'s and `merging._sort`'s own convention.
    """
    if first == last or nth == last:
        return
    _introselect(v, first, nth, last, _lg(last - first) * 2, comp)


class replace_the_worst:
    """C++: moses/moses/eda/replacement.h:41 replace_the_worst

    "Replace the lowest-scored individuals in the current population,
    specified by the iterators Dst, and replace them unconditionally with
    the individuals given by NewInst. The container holding the Dst
    iterators will be partially sorted." (the header's own comment.)
    """

    def __call__(self, new_insts: Sequence[scored_instance], dst: list[scored_instance]) -> None:
        """C++: moses/moses/eda/replacement.h:44 operator()

        `dst` stands in for the `[from_dst, to_dst)` iterator PAIR C++
        takes: the whole destination population, mutated in place (matching
        `field_set.py`'s own "iterator pair -> Python sequence" convention).
        `new_insts` stands in for `[from, to)`.

        C++: `OMP_ALGO::nth_element(from_dst, from_dst + distance(from, to),
        to_dst); copy(from, to, from_dst);` -- partition `dst` so the
        `len(new_insts)` SMALLEST elements (by `scored_instance::operator<`,
        i.e. worst-scoring first) land at the FRONT, then overwrite exactly
        that front slice with `new_insts`. The module docstring explains why
        `_nth_element` reproduces libstdc++'s permutation rather than
        substituting a full sort.
        """
        k = len(new_insts)
        # DIVERGENCE(C++ moses/moses/eda/replacement.h:46-47): OC_ASSERT
        # becomes ValueError.
        if k > len(dst):
            raise ValueError(
                "ERROR: The number of individuals to replace in a population\n"
                "is larger than the population itself."
            )
        _nth_element(dst, 0, k, len(dst))
        for i in range(k):
            dst[i] = new_insts[i]
