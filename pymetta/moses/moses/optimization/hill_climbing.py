"""C++: moses/moses/optimization/hill-climbing.{h,cc}

`hc_parameters` and `hill_climbing` -- "search the local neighborhood of an
instance for the highest score, move to that spot, and repeat until no
further improvement is possible" (hill-climbing.h:135-137's own doc
comment).

Crossover (`cross_top_one`, `cross_top_two`, `cross_top_three`, `crossover`,
the five-conjunct gate at hill-climbing.cc:139-152 and the `last_chance`
one-shot at :434-458) is ported here in full, so `hc_parameters(cross=True)`
-- classic's own default, both in the struct (hill-climbing.h:43) and on the
command line (`--hc-crossover`, problem-params.cc:424) -- is the
configuration this port actually runs. S9 shipped this file with the
crossover branch raising `NotImplementedError`; that guard and its
DIVERGENCE marker are gone, replaced by the real mechanics. See
`doc/cross_top_two_explanation.md` and `doc/cross_top_three_explanation.md`
in the classic tree for the intended simplex enumeration, and
`docs/CROSSOVER-INVENTORY.md` here for the defaults/gate/budget summary.

All three `cross_top_*` functions order the sample with
`std::partial_sort(..., std::greater<deme_inst_t>())`, whose tie order the
C++ standard does not specify at all -- yet that tie order decides WHICH
instances get crossed. `_partial_sort` below reproduces libstdc++'s actual
algorithm rather than substituting a Python sort; see its docstring for the
measurement.

NOT ported (pure logging, no observable effect on `deme`/return values --
see `optimization.py`'s module docstring for the same class of omission):
`log_stats_legend` (hill-climbing.cc:825-843), the "Crossover enabled"/
"Crossover disabled"/"Best score:"/"Terminate Local Search: ..." `logger()`
calls inside `operator()` (hill-climbing.cc:162,184,362,415,421,430,462,470,
478), the `hc_params.prefix_stat_deme` deme-statistics line at
hill-climbing.cc:381-397, and the `#ifdef GATHER_STATS` block
(hill-climbing.cc:291-322, dead code by construction -- see
`optimization.py`'s module docstring).

# DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:69,271,278 and
# hill-climbing.h:267): DO NOT FIX -- the "Search started from"/"Instance:"
# `logger().debug() << ...` calls (also dropped, alongside the ones above)
# are NOT in the same "no observable effect" class -- each one
# unconditionally evaluates `iscorer.get_rep()->get_candidate(inst, true)`
# as a plain chained-`<<` argument BEFORE the stream chain's own log-level
# check can skip anything, and `iscorer_base::get_rep()` (instance_scorer.
# h:40) defaults to returning `nullptr`, which `distance_based_scorer`
# (instance_scorer.h:49) never overrides. So in classic, EVERY run of
# `hill_climbing::operator()` (either overload) with a scorer whose
# `get_rep()` is null -- `distance_based_scorer` is exactly that --
# NULL-POINTER-DEREFERENCES on the very first iteration, unconditionally,
# regardless of configured log level. This is not a corner case:
# `tests/moses/optimizationUTest.cxxtest`'s own `test_hc`, classic's own
# REGISTERED unit test, constructs precisely `distance_based_scorer` +
# `hill_climbing` this way and CORE-DUMPS at hill-climbing.cc:69 (confirmed
# by building and running it directly -- see docs/CARRY-FORWARD.md and
# `tools/cpp_probe/hillclimbing_probe.cc`'s own file-header LANDMINE note,
# which works around it with a throwaway non-null `get_rep()`). This port
# has no logging surface at all (dropped uniformly, see above), so it never
# evaluates these `get_candidate()` calls and therefore never crashes here
# -- `tests/moses/moses/optimization/test_hill_climbing.py:152-164` asserts
# exactly this success for a `distance_based_scorer` run that would
# core-dump in classic. Kept as the port's behaviour: DO NOT "fix" this by
# reproducing the crash, and do not re-describe it as merely dropped
# logging with no effect -- the C++ behaviour it replaces is a crash, not a
# no-op.
"""

from __future__ import annotations

import math
import os
import time

from pymetta.moses.moses.neighborhood_sampling import (
    safe_binomial_coefficient,
    sample_new_instances,
)
from pymetta.moses.moses.optimization.optimization import (
    deme_t,
    information_theoretic_bits,
    optim_parameters,
    optimizer_base,
)
from pymetta.moses.moses.types import _to_score_t, score_t, very_worst_score
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance

# hill-climbing.cc's own local #define constants, ported as module-level
# names (Python has no preprocessor) rather than inlined magic numbers, to
# keep the citation trail exact.
_ACCEPTABLE_SIZE = 5000  # hill-climbing.cc:339 ACCEPTABLE_SIZE
_ACCEPTABLE_RAM_FRACTION = 0.5  # hill-climbing.cc:340,798 ACCEPTABLE_RAM_FRACTION
_MINIMUM_DEME_SIZE = 100  # hill-climbing.cc:547 MINIMUM_DEME_SIZE
_MAX_RAM_LIMIT = 0.9  # hill-climbing.cc:571 MAX_RAM_LIMIT
_DONT_BOTHER_SIZE = 500  # hill-climbing.cc:798 DONT_BOTHER_SIZE
_MIN_TO_KEEP = 20  # hill-climbing.cc:769 MIN_TO_KEEP

# hill-climbing.cc:144 xover_min_deme -- a local variable in C++, hoisted to
# a module constant here since nothing else about it varies per call.
_XOVER_MIN_DEME = 3

# Not itself a ported C++ symbol -- the modulus of a C `unsigned` (32-bit on
# every ABI classic is built for, and the type `cross_top_two`/
# `cross_top_three` declare their `max`, `num_to_sort`, `n` and `ntgt` locals
# with, hill-climbing.cc:627,632,646,657,668,673,684,690). Their arithmetic
# wraps mod 2^32 where Python ints would grow without bound; see those
# methods' own DIVERGENCE markers for where that is reachable.
_UNSIGNED_MAX = (1 << 32) - 1


def _c_div2(x: int) -> int:
    """Not itself a ported C++ symbol -- C/C++ integer division by two,
    which truncates TOWARD ZERO, where Python's `//` floors. Only the
    negative arguments differ, and libstdc++'s heap code below really does
    produce them: `__push_heap` computes `(__holeIndex - 1) / 2` with
    `__holeIndex == 0`, and `__adjust_heap` computes `(__len - 1) / 2` and
    `(__len - 2) / 2` with `__len` as small as 0. In C++ those are `0`,
    `0` and `-1`; in Python `//` would give `-1`, `-1` and `-1`, changing
    which branch is taken. Used ONLY by the `_partial_sort` family.
    """
    return -((-x) // 2) if x < 0 else x // 2


def _sqrtf_plus_3(x: int) -> int:
    """Not itself a ported C++ symbol -- `unsigned num_to_sort =
    sqrtf(2 * num_to_make) + 3;` (hill-climbing.cc:632), reproduced with
    every C conversion it performs and Python performs by default.

    Four separate narrowings hide in that one line: the `size_t` argument
    is converted to `float` (binary32, round-to-nearest) before `sqrtf`
    sees it; `sqrtf` itself is single-precision; `+ 3` promotes the int to
    `float` and stays in `float`; and the assignment to `unsigned`
    truncates toward zero. Computing it in Python `float` (binary64) and
    rounding once at the end is NOT the same function.

    `_to_score_t` is reused as the binary32 narrowing primitive here (it is
    a `struct` 'f' round-trip and nothing more); the value being narrowed
    is a C `float`, not a `score_t`.

    MEASURED against classic's own arithmetic rather than reasoned about:
    a standalone C++ probe (tools/cpp_probe/, see the S12 task report)
    evaluated `(unsigned)(sqrtf(2*n) + 3)` for every n in [0, 5000] plus
    eleven larger values up to 10**12, and this function reproduces all
    5012 exactly. Python's `math.sqrt` is IEEE-correctly-rounded in
    binary64, and binary64 has more than the 2p+2 = 50 bits that make
    double-rounding a square root of a binary32 value equivalent to
    rounding it directly, so the agreement is not a coincidence of the
    sample.
    """
    arg = _to_score_t((2 * x) & _SIZE_T_MAX)
    return int(_to_score_t(_to_score_t(math.sqrt(arg)) + 3.0)) & _UNSIGNED_MAX


def _cbrtf_plus_3(x: int) -> int:
    """Not itself a ported C++ symbol -- `unsigned num_to_sort =
    cbrtf(6 * num_to_make) + 3;` (hill-climbing.cc:673). Same conversion
    chain as `_sqrtf_plus_3`; see that docstring.

    `cbrt` is not required to be correctly rounded the way `sqrt` is, so
    agreement here rests on measurement alone, not on an error bound: the
    same probe evaluated `(unsigned)(cbrtf(6*n) + 3)` over the same 5012
    inputs and this function reproduces all of them. The realistic domain
    is far smaller than that -- `crossover` never passes more than
    `crossover_pop_size / 3` (40 by default) -- so the measured range
    covers the reachable one with room to spare.
    """
    arg = _to_score_t((6 * x) & _SIZE_T_MAX)
    return int(_to_score_t(_to_score_t(math.cbrt(arg)) + 3.0)) & _UNSIGNED_MAX


def _greater(a: scored_instance, b: scored_instance) -> bool:
    """Not itself a ported C++ symbol -- `std::greater<deme_inst_t>()`, the
    comparator all three `cross_top_*` functions and `resize_deme` pass to
    `std::partial_sort`. `std::greater<T>::operator()(a, b)` is `a > b`,
    which for `deme_inst_t` = `scored_instance<composite_score>` resolves to
    `scored_instance::operator>` (scored_instance.h:48-51), i.e.
    `a.second > b.second`, i.e. `composite_score`'s boost-generated `>`,
    i.e. `b.second < a.second` (types.cc:131-149). Written as a free
    function rather than reusing `scored_instance.__gt__` so the
    `_partial_sort` family below reads as the C++ algorithm it reproduces,
    with the comparator explicit at every call the way libstdc++ passes it.
    """
    return b.second < a.second


# ---------------------------------------------------------------------------
# libstdc++'s std::partial_sort, reproduced exactly.
#
# All three cross_top_* functions (hill-climbing.cc:604,635,675) and
# resize_deme (:815) order their sample with
# `std::partial_sort(first, middle, last, std::greater<deme_inst_t>())`.
# [alg.sort] gives partial_sort NO stability guarantee, so C++ has no
# *defined* tie order -- but libstdc++'s actual, deterministic ordering is
# what classic produces, and it decides which instances get crossed and in
# which order. Ties are not a corner case here: two composite_scores tie
# under `operator<` whenever their penalized scores AND complexities are
# equal (types.cc:139-148), which is the normal state of a Boolean deme
# where many neighbours score identically.
#
# MEASURED, not assumed: tools/cpp_probe/crossover_probe.cc runs
# `std::partial_sort(v.begin(), v.begin()+mid, v.end(),
# std::greater<deme_inst_t>())` over 15 hand-built samples (all-tied at
# sizes 5/8/12/40, two tied blocks, strictly ascending/descending, equal
# penalized score with differing complexity, penalty-inverted order, NaN,
# +/-inf, mid == 0, mid == size, size == 1, and a shuffled mix), tagging
# each element with an id in its instance bits that the comparator never
# reads, and dumps the resulting permutation. The functions below reproduce
# all 15 element for element (tests/fixtures/crossover.json,
# `test_partial_sort_matches_libstdcxx_order`).
#
# A Python `sorted`/`list.sort` does NOT: it is stable, so `all_tied_8_mid4`
# would come back `[0,1,2,3,4,5,6,7]` where libstdc++ measurably produces
# `[1,3,0,2,4,5,6,7]`. The "a stable sort can only be at least as
# well-ordered as an unspecified one" argument -- used elsewhere in this
# port for `std::sort` -- is therefore FALSE here, and was checked rather
# than inherited.
#
# The source of truth is /usr/include/c++/16/bits/stl_algo.h
# (`__partial_sort`, `__heap_select`) and bits/stl_heap.h (`__make_heap`,
# `__adjust_heap`, `__push_heap`, `__pop_heap`, `__sort_heap`); the four
# helpers below are one-to-one transliterations of those, keeping libstdc++'s
# own parameter names so the correspondence is checkable line by line.
# `_GLIBCXX_MOVE` is a no-op here: Python assignment rebinds, and every
# moved-from slot is overwritten before it is read again, exactly as the
# hole-based C++ algorithms require.
# ---------------------------------------------------------------------------


def _push_heap(v, first: int, holeIndex: int, topIndex: int, value, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__push_heap`
    (bits/stl_heap.h:141). See the block comment above.
    """
    parent = _c_div2(holeIndex - 1)
    while holeIndex > topIndex and comp(v[first + parent], value):
        v[first + holeIndex] = v[first + parent]
        holeIndex = parent
        parent = _c_div2(holeIndex - 1)
    v[first + holeIndex] = value


def _adjust_heap(v, first: int, holeIndex: int, length: int, value, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__adjust_heap`
    (bits/stl_heap.h:228). See the block comment above.
    """
    topIndex = holeIndex
    secondChild = holeIndex
    while secondChild < _c_div2(length - 1):
        secondChild = 2 * (secondChild + 1)
        if comp(v[first + secondChild], v[first + secondChild - 1]):
            secondChild -= 1
        v[first + holeIndex] = v[first + secondChild]
        holeIndex = secondChild
    if (length & 1) == 0 and secondChild == _c_div2(length - 2):
        secondChild = 2 * (secondChild + 1)
        v[first + holeIndex] = v[first + (secondChild - 1)]
        holeIndex = secondChild - 1
    _push_heap(v, first, holeIndex, topIndex, value, comp)


def _pop_heap(v, first: int, last: int, result: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__pop_heap`
    (bits/stl_heap.h:256). See the block comment above.
    """
    value = v[result]
    v[result] = v[first]
    _adjust_heap(v, first, 0, last - first, value, comp)


def _make_heap(v, first: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__make_heap`
    (bits/stl_heap.h:340). See the block comment above.
    """
    if last - first < 2:
        return
    length = last - first
    parent = _c_div2(length - 2)
    while True:
        value = v[first + parent]
        _adjust_heap(v, first, parent, length, value, comp)
        if parent == 0:
            return
        parent -= 1


def _sort_heap(v, first: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__sort_heap`
    (bits/stl_heap.h:417). See the block comment above.
    """
    while last - first > 1:
        last -= 1
        _pop_heap(v, first, last, last, comp)


def _heap_select(v, first: int, middle: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__heap_select`
    (bits/stl_algo.h:1589). See the block comment above.
    """
    _make_heap(v, first, middle, comp)
    i = middle
    while i < last:
        if comp(v[i], v[first]):
            _pop_heap(v, first, middle, i, comp)
        i += 1


def _partial_sort(v, first: int, middle: int, last: int, comp=_greater) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__partial_sort`
    (bits/stl_algo.h:1871), which is what `std::partial_sort(first, middle,
    last, comp)` (bits/stl_algo.h:4696) forwards to. Sorts `v[first:middle]`
    in place so it holds the `middle - first` elements of `v[first:last]`
    that `comp` ranks highest, in `comp` order; `v[middle:last]` is left in
    an implementation-defined -- but here exactly reproduced -- state.

    Operates on a slice of `v` by index rather than on iterators, so the
    same call shape works for `resize_deme`'s whole-deme sort and for the
    `cross_top_*` functions' `[sample_start, sample_start + sample_size)`
    window. See the block comment above for the measurement.
    """
    _heap_select(v, first, middle, last, comp)
    _sort_heap(v, first, middle, comp)

# Not itself a ported C++ symbol -- `sizeof(instance)` (hill-climbing.cc:72,
# 577's own `_instance_bytes = sizeof(instance) + sizeof(packed_t) *
# fields.packed_width();`) has no Python analogue, but `instance` is
# `std::vector<packed_t>` (instance.h:43) and `packed_t` is `unsigned long`
# (instance.h:34, pinned 64-bit -- see instance.py), so `sizeof(instance)`
# is exactly `sizeof(std::vector<unsigned long>)`: measured directly on the
# same 64-bit libstdc++ ABI instance.py's own DIVERGENCE note pins --
# 24 bytes (three 8-byte pointers: begin/end/capacity), independent of
# `packed_width()`. Used only by `_instance_bytes` below, itself dormant
# behind `resize_to_fit_ram=False` for the same reason `_get_total_ram_
# bytes` gives.
_SIZEOF_INSTANCE_BYTES = 24

# DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:541): `size_t
# nleft = max_evals - current_number_of_evals;` -- see `n_new_instances`
# below for why this mask exists and is load-bearing, not decorative.
# Pinned to the same 64-bit ABI `instance.py`'s `packed_t`/`_PACKED_MASK`
# and `neighborhood_sampling.py`'s `_SIZE_T_MAX` already commit to.
_SIZE_T_MAX = (1 << 64) - 1


def _get_total_ram_bytes() -> int:
    """Not itself a ported C++ symbol -- stands in for cogutil's
    `getTotalRAM()` (`$COGUTIL/opencog/util/platform.cc:131-135`,
    `getpagesize() * sysconf(_SC_PHYS_PAGES)`; cogutil's own platform-RAM-
    query helpers are otherwise out of this port's scope). Used ONLY to
    compute `hill_climbing._total_RAM_bytes` (the TOTAL-RAM half of the
    `hc_params.resize_to_fit_ram` branches below) -- for the FREE-RAM half,
    see `_get_free_ram_bytes` below; the two are NOT interchangeable, and
    conflating them was a real bug this fix round corrected (see that
    function's own docstring). `resize_to_fit_ram` defaults to `False`
    (matching C++'s own `hc_parameters` default, hill-climbing.h:50) and no
    test in this port's fidelity suite enables it, since (per
    hill-climbing.h:119-124's own comment) it makes a run's behaviour
    depend on the installed RAM of the machine it happens to run on -- the
    opposite of the bit-identical reproducibility this whole slice exists
    to prove. Best-effort via `os.sysconf`; falls back to a generous fixed
    sentinel on any platform where those keys are absent (`os.sysconf`
    itself is POSIX-only and can raise `ValueError` for an unrecognized
    key).
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 1 << 34  # 16 GiB fallback


def _get_free_ram_bytes() -> int:
    """Not itself a ported C++ symbol -- stands in for cogutil's
    `getFreeRAM()` on the Linux path actually exercised here
    (`$COGUTIL/opencog/util/platform.cc:137-141`, `getpagesize() *
    sysconf(_SC_AVPHYS_PAGES)` -- "AVailable PHYSical pages", distinct
    from `_SC_PHYS_PAGES`'s TOTAL physical pages that
    `_get_total_ram_bytes` above queries). `platform.cc` also has a
    non-Linux branch (`:115-118`, `getTotalRAM() - getMemUsage()`) that is
    not reproduced here -- out of scope the same way the rest of
    `platform.cc` is (see `_get_total_ram_bytes`'s own docstring).

    Used ONLY by `n_new_instances`'s `hc_params.resize_to_fit_ram` branch
    (hill-climbing.cc:577's own `getFreeRAM()` call) to cap
    `number_of_new_instances` against actually-available RAM rather than
    installed RAM -- conflating the two (an earlier version of this file
    called `_get_total_ram_bytes` here instead) makes `free_ram == total`
    always, which always picks `ACCEPTABLE_RAM_FRACTION * total` over
    `MAX_RAM_LIMIT * free` in the `min(...)` two lines below, silently
    discarding the free-RAM half of that cap entirely. Dormant behind
    `resize_to_fit_ram=False` for the exact reproducibility reason
    `_get_total_ram_bytes` gives. Best-effort via `os.sysconf`, same
    fallback convention as `_get_total_ram_bytes`.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 1 << 34  # 16 GiB fallback


class hc_parameters:
    """C++: moses/moses/optimization/hill-climbing.h:41 hc_parameters

    "Hill-climbing parameters" (the header's own comment, with its own
    typo reproduced verbatim in the C++ struct name only, not here).
    """

    def __init__(
        self,
        widen: bool = False,
        step: bool = False,
        cross: bool = True,
        max_evals: int = 20000,
        fraction_of_nn: float = 2.0,
    ) -> None:
        """C++: moses/moses/optimization/hill-climbing.h:42 hc_parameters

        Only `widen`/`step`/`cross`/`max_evals`/`fraction_of_nn` are real
        constructor parameters in C++ (hill-climbing.h:42-45); every other
        field below is a fixed member-initializer default
        (hill-climbing.h:47-50) that C++ callers can only change by
        mutating the constructed object's public field afterward -- ported
        the same way, as plain assignments here rather than additional
        constructor parameters. `OC_ASSERT(0.0 < fraction_of_nn)`
        (hill-climbing.h:51) becomes `ValueError`.
        """
        if not (0.0 < fraction_of_nn):
            raise ValueError("fraction_of_nn must be positive")
        self.widen_search = widen
        self.single_step = step
        self.crossover = cross
        self.crossover_pop_size = 120
        self.crossover_min_neighbors = 400
        self.max_nn_evals = max_evals
        self.fraction_of_nn = fraction_of_nn
        self.score_range: score_t = 5.0
        self.max_allowed_instances = 10000
        self.resize_to_fit_ram = False
        self.prefix_stat_deme = ""


class hill_climbing(optimizer_base):
    """C++: moses/moses/optimization/hill-climbing.h:175 hill_climbing

    See the module docstring for the crossover DIVERGENCE and the dropped
    logging surface.
    """

    def __init__(
        self,
        op: optim_parameters | None = None,
        hc: hc_parameters | None = None,
    ) -> None:
        """C++: moses/moses/optimization/hill-climbing.h:176 hill_climbing"""
        super().__init__(op)
        self.hc_params = hc_parameters() if hc is None else hc
        self._total_RAM_bytes = _get_total_ram_bytes()
        self._instance_bytes = 0

    # ---------------------------------------------------------------- estimate

    def estimate_neighborhood(self, distance: int, fields: field_set) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:486 estimate_neighborhood

        "Return an estimate of the size of the neighborhood at distance
        'distance'" (hill-climbing.h:184-186's own comment). `const size_t
        nn_estimate = information_theoretic_bits(fields);` (hill-climbing.
        cc:491) truncates the `double` result toward zero -- `int(...)`
        matches, since `information_theoretic_bits` is always non-negative
        here.
        """
        if distance == 0:
            return 1

        nn_estimate = int(information_theoretic_bits(fields))

        if nn_estimate < distance:
            # DO NOT FIX: this reassigns the LOCAL `distance`, not
            # `self`/caller state -- C++ takes `distance` by value
            # (hill-climbing.cc:486-487), so this warning-then-clamp is
            # entirely local to this one call, exactly as ported here.
            distance = nn_estimate

        if distance == 1:
            return 2 * nn_estimate  # be large given nn_estimate is only an estimate
        return safe_binomial_coefficient(nn_estimate, distance)

    def n_new_instances(
        self,
        distance: int,
        max_evals: int,
        current_number_of_evals: int,
        total_number_of_neighbors: int,
    ) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:520 n_new_instances

        "Return an estimate of the number of new instances to search"
        (hill-climbing.h:188-189's own comment). The `for` loop
        (hill-climbing.cc:530-531) re-truncates `number_of_new_instances`
        toward zero on EVERY multiplication by `fraction_of_nn`, not once
        at the end -- ported as a literal per-iteration `int(...)` loop,
        not `int(total * fraction_of_nn ** distance)`, since C++'s
        `size_t *= double` truncates at each individual `*=`.
        """
        if distance == 0:
            return 1

        number_of_new_instances = total_number_of_neighbors
        for _ in range(distance):
            number_of_new_instances = int(number_of_new_instances * self.hc_params.fraction_of_nn)

        if number_of_new_instances > self.hc_params.max_nn_evals:
            number_of_new_instances = self.hc_params.max_nn_evals

        # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:540-541):
        # `size_t nleft = max_evals - current_number_of_evals;` -- despite
        # the "avoid overflow" comment there, this is exactly where C++
        # OVERFLOWS: `max_evals` (`unsigned`) promotes to `size_t` for the
        # subtraction, so `max_evals < current_number_of_evals` WRAPS to a
        # huge unsigned value rather than clamping to zero or going
        # negative. This is REACHABLE, not merely a theoretical corner
        # case: the caller's `if (max_evals <= current_number_of_evals)
        # break;` guard (hill-climbing.cc:413-417) is skipped on the very
        # first "distance 0 -> 1" transition by the earlier `if
        # (number_of_new_instances == 1 && distance == 1) continue;`
        # (hill-climbing.cc:408-409, mirrored by this port's own matching
        # `continue` further down in `__call__`), so a second call to
        # `n_new_instances` can and does land with `current_number_of_evals
        # > max_evals` already true -- e.g. `hill_climbing()(deme,
        # distance_based_scorer(fs, target), max_evals=0, 86400)` on a
        # 3-knob `field_set` reaches exactly this call with `(distance=1,
        # max_evals=0, current_number_of_evals=1, total_number_of_
        # neighbors=6)`. Measured directly against classic's own
        # arithmetic (a standalone probe reproducing this function's exact
        # `size_t`/`unsigned` types, not merely reasoned about):
        # `n_new_instances(distance=1, max_evals=1,
        # current_number_of_evals=3, total_number_of_neighbors=6)` == the
        # wrapped `nleft` feeding `min(nleft, MINIMUM_DEME_SIZE)` two lines
        # below, giving **100** -- not a crash, not zero, and not what an
        # unbounded Python `int` subtraction produces (a plain negative
        # value, measured as -2 for that same call before this mask was
        # added). Python ints don't overflow on their own, so reproducing
        # the wrap requires this explicit mask; DO NOT clamp to 0 instead
        # -- that would silently "fix" a real, reachable upstream bug
        # rather than port it, and would not reproduce the measured 100.
        nleft = (max_evals - current_number_of_evals) & _SIZE_T_MAX

        if number_of_new_instances < _MINIMUM_DEME_SIZE:
            number_of_new_instances = min(nleft, _MINIMUM_DEME_SIZE)

        if nleft < number_of_new_instances:
            number_of_new_instances = nleft

        if self.hc_params.resize_to_fit_ram:
            new_usage = self._instance_bytes * number_of_new_instances
            if _ACCEPTABLE_RAM_FRACTION * self._total_RAM_bytes < new_usage:
                # C++: hill-climbing.cc:577 `uint64_t free_ram = getFreeRAM();`
                # -- see `_get_free_ram_bytes`'s own docstring for why this
                # is NOT `_get_total_ram_bytes` (that was a real bug here,
                # since fixed: with free==total, the `min(...)` below always
                # took the ACCEPTABLE_RAM_FRACTION arm).
                free_ram = _get_free_ram_bytes()
                cap = min(
                    _ACCEPTABLE_RAM_FRACTION * self._total_RAM_bytes, _MAX_RAM_LIMIT * free_ram
                )
                number_of_new_instances = int(cap / self._instance_bytes)

        return number_of_new_instances

    # --------------------------------------------------------------- crossover

    def cross_top_one(
        self,
        deme: deme_t,
        deme_size: int,
        num_to_make: int,
        sample_start: int,
        sample_size: int,
        base: instance,
    ) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:593 cross_top_one

        "Cross the single top-scoring instance against the next-highest
        scorers" (hill-climbing.h:194-196's own comment). Appends
        `num_to_make` new instances at `deme[deme_size:]`, each a copy of
        one of the sample's 2nd..(num_to_make+1)th best instances with the
        BEST instance's changes-from-`base` overlaid on top
        (`merge_instance`, field_set.h:468).

        `OC_ASSERT(sample_size > 0, "Cross-over sample size must be
        positive")` (hill-climbing.cc:596) becomes `ValueError`, matching
        this port's established OC_ASSERT-to-exception convention. It is
        load-bearing rather than decorative: `sample_size - 1` one line
        below is `size_t` arithmetic, so with `sample_size == 0` it would
        wrap to 2**64-1 and the clamp would never fire. The assertion makes
        that wrap unreachable, so -- unlike `n_new_instances`'s `nleft` --
        there is nothing to reproduce with a mask here, and the guard is
        ported rather than dropped.

        `reference` is `const instance&` in C++ (:612), bound AFTER the
        `deme.resize` on the line above (which, in C++, may reallocate and
        invalidate any reference taken earlier); the Python alias is taken
        at the same point for the same reason. It is never written by the
        loop -- every target index `j` is `>= deme_size`, and
        `deme_size >= sample_start + sample_size` for the only caller,
        `crossover` -- so the alias-vs-copy distinction is not observable.
        """
        if not sample_size > 0:
            raise ValueError("Cross-over sample size must be positive")
        if sample_size - 1 < num_to_make:
            num_to_make = sample_size - 1

        # "We don't actually need them all sorted; we only need the
        # highest-scoring num_to_make+1 to appear first, and after that, we
        # don't care about the ordering." (hill-climbing.cc:598-601.) The
        # ordering libstdc++ actually produces past `middle` is reproduced
        # anyway -- see `_partial_sort`.
        _partial_sort(
            deme, sample_start, sample_start + num_to_make + 1, sample_start + sample_size
        )

        deme.resize(deme_size + num_to_make)

        # best instance
        reference = deme[sample_start].first

        # Skip the first entry; its the top scorer.
        for i in range(num_to_make):
            j = deme_size + i
            # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:616):
            # C++ declares `unsigned j` there, so `deme_size + i` truncates
            # to 32 bits on assignment; Python's `j` is unmasked. Same
            # family as `cross_top_two`/`cross_top_three`'s masked `n`/
            # `ntgt`, left unmasked here because it is unreachable for the
            # same reason those are untested (CARRY-FORWARD.md 5b.6):
            # `deme.resize(deme_size + num_to_make)` two lines above already
            # succeeded, so `deme_size + i < deme_size + num_to_make` fits
            # comfortably; wrapping at 2**32 would need a deme of billions
            # of instances. Reported rather than fixed -- a marker must
            # state what C++ actually does, not paper over the gap.
            src = deme[sample_start + i + 1]  # +1 to skip the first one
            # `deme[j] = deme[...]` is scored_instance COPY-assignment in
            # C++ (std::pair's, inherited through tagged_item), not an
            # alias -- CARRY-FORWARD.md 2.4's rule. merge_instance then
            # mutates only the copy.
            deme[j] = scored_instance(src.first, src.second)
            # `merge_instance(deme[j], ...)` in C++ reaches `deme[j].first`
            # through tagged_item's implicit `operator Item&()`
            # (opencog/util/functional.h:78-80).
            deme.fields().merge_instance(deme[j].first, base, reference)
        return num_to_make

    def cross_top_two(
        self,
        deme: deme_t,
        deme_size: int,
        num_to_make: int,
        sample_start: int,
        sample_size: int,
        base: instance,
    ) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:624 cross_top_two

        "two-dimensional simplex version of above" (hill-climbing.cc:623).
        For each pair (j, i) with j < i, appends a copy of sample j with
        sample i's changes-from-`base` overlaid, at the triangular index
        `n = i*(i-1)/2 + j`.

        # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:628,632,
        # 652,655): every one of `max`, `num_to_sort`, `n` and `ntgt` is
        # declared `unsigned` in C++ -- 32-bit -- while `sample_size`,
        # `num_to_make` and `deme_size` are `size_t`. So `sample_size *
        # (sample_size - 1) / 2` is evaluated in 64-bit and then TRUNCATED
        # to 32 bits on assignment to `max`; `num_to_sort = sample_size`
        # truncates the same way; `n = i * (i - 1) / 2 + j` wraps mod 2**32
        # in its own right (i * (i-1) exceeds 2**32 at i >= 65537); and
        # `ntgt = deme_size + n` is a 64-bit sum truncated to 32 bits.
        # Python ints are unbounded, so each of those is masked explicitly
        # below rather than clamped -- the same treatment, and for the same
        # reason, as `n_new_instances`'s `nleft`. All four wraps need a
        # deme or a sample of billions of instances to reach and are not
        # reachable in any configuration this port is tested in; they are
        # reproduced so that if a caller ever does reach one, the port
        # wraps where classic wraps instead of quietly computing a
        # different, "correct" answer.
        """
        # sample_size choose two.
        max_ = ((sample_size * ((sample_size - 1) & _SIZE_T_MAX)) & _SIZE_T_MAX) // 2
        max_ &= _UNSIGNED_MAX
        if max_ < num_to_make:
            num_to_make = max_

        # `_sqrtf_plus_3` applies the `2 *` itself -- see its docstring.
        num_to_sort = _sqrtf_plus_3(num_to_make)
        if sample_size < num_to_sort:
            num_to_sort = sample_size & _UNSIGNED_MAX
        _partial_sort(deme, sample_start, sample_start + num_to_sort, sample_start + sample_size)

        deme.resize(deme_size + num_to_make)

        # Summation is over a 2-simplex
        for i in range(1, sample_size):
            # (i+1)th best instance
            reference = deme[sample_start + i].first

            # merge with all best instance (better than reference) but
            # using reference as the one containing the mutations to pass
            # on the other (better) instance.
            for j in range(i):
                n = (((i * ((i - 1) & _UNSIGNED_MAX)) & _UNSIGNED_MAX) // 2 + j) & _UNSIGNED_MAX
                if num_to_make <= n:
                    return num_to_make
                ntgt = (deme_size + n) & _UNSIGNED_MAX
                src = deme[sample_start + j]
                deme[ntgt] = scored_instance(src.first, src.second)
                deme.fields().merge_instance(deme[ntgt].first, base, reference)
        return num_to_make

    def cross_top_three(
        self,
        deme: deme_t,
        deme_size: int,
        num_to_make: int,
        sample_start: int,
        sample_size: int,
        base: instance,
    ) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:664 cross_top_three

        "three-dimensional simplex version of above"
        (hill-climbing.cc:663). Intended to walk strict triples
        `k < j < i` and place each child at the tetrahedral index
        `n = i*(i-1)*(i-2)/6 + j*(j-1)/2 + k`.

        # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:687):
        # DO NOT FIX -- the innermost loop is `for (unsigned k = 0; k < i;
        # k++)` while the index formula on the very next line assumes
        # `k < j`. So `n` is not a bijection, and this is what actually
        # happens (derived from the formula and confirmed against
        # tests/fixtures/crossover.json, `cross_top_three`'s own cases):
        #
        # * NOTHING IS SKIPPED. For a fixed `i` the offsets `j*(j-1)/2 + k`
        #   produced are the union over j in [1,i) of the i consecutive
        #   values [j*(j-1)/2, j*(j-1)/2 + i). Successive j-runs start
        #   j apart with j < i, so the runs overlap and the union is the
        #   CONTIGUOUS range [0, i*(i-1)/2]. The intended range for block i
        #   is [0, i*(i-1)/2), so the block covers all of its own slots and
        #   then one more.
        # * SLOTS ARE WRITTEN REPEATEDLY, last write wins. Offset m in
        #   block i is written once for every j in [1,i) with
        #   0 <= m - j*(j-1)/2 < i, which is up to O(sqrt(i)) times. Only
        #   the final (j, k) pair survives, and it is not the (j, k) the
        #   formula intends: e.g. i=3 writes n=3 twice, first as
        #   (j=1, k=2) and then as (j=2, k=1), and the second wins.
        # * ONE SLOT OF THE NEXT BLOCK IS CLOBBERED EARLY. The block's last
        #   offset, i*(i-1)/2, maps to n = C(i+1,3), i.e. block i+1's slot
        #   0, which block i+1 then rewrites with its own (j=1, k=0).
        # * `k == i` NEVER HAPPENS -- the loop bound is `k < i`, strictly,
        #   so `i` itself is never a value `k` takes. The only degenerate
        #   case is `k == j` (k ranges over [0, i) and j over [1, i), so
        #   k == j occurs exactly once per (i, j) sweep). When it does,
        #   `src = deme[sample_start + k]` is the SAME sample as `jref`
        #   (both read `deme[sample_start + j]`), so `child` starts as a
        #   copy of jref, not a fresh three-way mix.
        # * `merge_instance(child, base, jref)` -- the SECOND merge -- is
        #   NOT a no-op here; it OVERWRITES whatever the first merge
        #   (`merge_instance(child, base, iref)`) left behind, at every
        #   knob where `base != jref`, resetting those knobs to jref's own
        #   values. Measured with the port's own `field_set.merge_instance`
        #   on three ternary knobs (MOSES's actual logical knob, multy=3):
        #   base=[0,0,0], iref=[1,1,1], jref=[2,2,2] -- `child` goes
        #   [2,2,2] (copy of jref) -> [1,1,1] (merge with iref) -> [2,2,2]
        #   (merge with jref). The net effect in this configuration is
        #   sample j UNCHANGED, not a two-way cross: the k == j slot's
        #   "three-way cross" degenerates to writing back the original
        #   jref sample verbatim, because base disagrees with jref at
        #   every knob and the second merge therefore restores all of
        #   them.
        # * `if (num_to_make <= n) return num_to_make;` (:689) still leaves
        #   no hole below `num_to_make`: because the covered offsets always
        #   form a contiguous prefix and rise by one within each j-run, the
        #   first `n >= num_to_make` is reached only after every smaller
        #   slot has already been written at least once.
        #
        # Net effect: the function returns the promised count of
        # initialized instances, but their CONTENT is a different multiset
        # of crosses than `k < j` would produce. Reproduced literally,
        # because bit-identical behaviour against classic is the point.
        #
        # The `unsigned` truncations are the same family
        # `cross_top_two` documents. This one is the most reachable of
        # them, but the onset is i == 1627, not i == 2953 as an earlier
        # version of this note claimed (understating it by ~1.8x): `i`,
        # `j`, `k` are themselves `unsigned` (32-bit) in C++, so
        # `i * (i - 1) * (i - 2)` wraps mod 2**32 AT THE MULTIPLICATION,
        # before the `/ 6` -- and the raw product first exceeds 2**32 at
        # i == 1627 (1627*1626*1625 == 4,298,940,750; i == 1626 gives
        # 4,291,014,000, still under). i == 2953 was instead (an incorrect)
        # attempt at the DIFFERENT threshold where the unwrapped quotient
        # `i*(i-1)*(i-2)/6` itself would cross 2**32 -- that is i == 2955,
        # not 2953 (i == 2954's quotient is 4,291,795,704, still under) --
        # but that quotient threshold was never the relevant one anyway,
        # since the 32-bit wraparound in C++ happens two steps earlier, in
        # the product itself.
        # masked, never clamped.
        """
        # sample_size choose three.
        max_ = sample_size * ((sample_size - 1) & _SIZE_T_MAX) & _SIZE_T_MAX
        max_ = ((max_ * ((sample_size - 2) & _SIZE_T_MAX)) & _SIZE_T_MAX) // 6
        max_ &= _UNSIGNED_MAX
        if max_ < num_to_make:
            num_to_make = max_

        # `_cbrtf_plus_3` applies the `6 *` itself -- see its docstring.
        num_to_sort = _cbrtf_plus_3(num_to_make)
        if sample_size < num_to_sort:
            num_to_sort = sample_size & _UNSIGNED_MAX
        _partial_sort(deme, sample_start, sample_start + num_to_sort, sample_start + sample_size)
        deme.resize(deme_size + num_to_make)

        # Summation is over a 3-simplex
        for i in range(2, sample_size):
            iref = deme[sample_start + i].first
            for j in range(1, i):
                jref = deme[sample_start + j].first
                for k in range(i):  # DO NOT FIX: `k < i`, see the docstring
                    n = (i * ((i - 1) & _UNSIGNED_MAX)) & _UNSIGNED_MAX
                    n = ((n * ((i - 2) & _UNSIGNED_MAX)) & _UNSIGNED_MAX) // 6
                    n = (n + ((j * ((j - 1) & _UNSIGNED_MAX)) & _UNSIGNED_MAX) // 2) & _UNSIGNED_MAX
                    n = (n + k) & _UNSIGNED_MAX
                    if num_to_make <= n:
                        return num_to_make
                    ntgt = (deme_size + n) & _UNSIGNED_MAX
                    src = deme[sample_start + k]
                    deme[ntgt] = scored_instance(src.first, src.second)
                    deme.fields().merge_instance(deme[ntgt].first, base, iref)
                    deme.fields().merge_instance(deme[ntgt].first, base, jref)
        return num_to_make

    def crossover(
        self,
        deme: deme_t,
        deme_size: int,
        max_number_of_new_instances: int,
        sample_start: int,
        sample_size: int,
        base: instance,
    ) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:701 crossover

        "Chain the 3 crossovers methods above and return the number of new
        instances" (hill-climbing.h:232-234's own comment). The budget is
        first capped at `crossover_pop_size` (120 by default), then split
        `num_to_make/3 + num_to_make%3` to `cross_top_one` and
        `num_to_make/3` to each of the other two -- so the whole remainder
        goes to `cross_top_one`, and the default 120 splits 40/40/40.

        Each stage is handed `deme_size` ALREADY ADVANCED by what the
        previous stages actually returned (which can be less than they were
        asked for, since each clamps to its own simplex size), while
        `sample_start`/`sample_size` stay pointing at the same, unchanged
        sample window -- so all three stages re-sort and re-read the SAME
        window, each on top of the previous stage's permutation of it.

        NOT ported: the commented-out `cross_top_one(deme, deme_size,
        num_to_make, sample_start, sample_size, base)` call
        (hill-climbing.cc:709-712) that the live call at :713 replaced.
        """
        # Do not make more than the budget
        num_to_make = min(max_number_of_new_instances, self.hc_params.crossover_pop_size)

        # These cross-over (in the genetic sense) the top-scoring one, two
        # and three instances, respectively.
        number_of_new_instances = self.cross_top_one(
            deme,
            deme_size,
            # Add the remainder so that it all sums up to num_to_make in
            # the end.
            num_to_make // 3 + num_to_make % 3,
            sample_start,
            sample_size,
            base,
        )

        number_of_new_instances += self.cross_top_two(
            deme,
            deme_size + number_of_new_instances,
            num_to_make // 3,
            sample_start,
            sample_size,
            base,
        )

        number_of_new_instances += self.cross_top_three(
            deme,
            deme_size + number_of_new_instances,
            num_to_make // 3,
            sample_start,
            sample_size,
            base,
        )

        return number_of_new_instances

    # ---------------------------------------------------------------- resizing

    def resize_by_score(self, deme: deme_t, cutoff: score_t) -> int:
        """C++: moses/moses/optimization/hill-climbing.cc:741 resize_by_score

        Repeatedly finds and deletes the FIRST contiguous run (scanning
        from index 0) of instances whose penalized score is `<= cutoff`,
        re-scanning from the start after each deletion, until no such run
        remains or `len(deme) < 20` -- NOT a single-pass "delete every
        instance <= cutoff": the `MIN_TO_KEEP` check runs after each
        individual run is deleted, so it can stop early with some
        low-scoring instances still present if deletion drops the deme
        below 20 partway through. Ported as an explicit index scan
        (`del deme[first:last+1]`) rather than C++'s iterator dance, since
        Python has no direct iterator-pair `erase`.

        NOT ported: `ndeleted` (hill-climbing.cc:742,766,773) -- counted
        purely to feed the dropped `logger().debug() << "Trimmed " <<
        ndeleted << ...` line; no other reader.
        """
        while True:
            first = None
            last = None
            contig = 0
            for i in range(len(deme)):
                iscore = deme[i].second.get_penalized_score()
                if iscore <= cutoff:
                    if contig == 0:
                        first = i
                    last = i
                    contig += 1
                elif contig > 0:
                    break

            if contig == 0:
                break

            del deme[first : last + 1]

            if len(deme) < _MIN_TO_KEEP:
                break

        return len(deme)

    def resize_deme(self, deme: deme_t, best_score: score_t) -> bool:
        """C++: moses/moses/optimization/hill-climbing.cc:781 resize_deme

        "Keep the size of the deme at a manageable level" (the header's
        own comment). `usage` is computed ONCE, before either `if` --
        DO NOT FIX: the second `if` reuses that same, now-stale value
        rather than recomputing it after `resize_by_score` may have
        shrunk `deme`, exactly matching hill-climbing.cc:799-814's own
        variable reuse.
        """
        did_resize = False
        cutoff = _to_score_t(best_score - self.hc_params.score_range)
        bad_score_cnt = sum(1 for si in deme if si.second.get_penalized_score() <= cutoff)

        usage = self._instance_bytes * len(deme)

        ram_pressure = _ACCEPTABLE_RAM_FRACTION * self._total_RAM_bytes < usage
        if bad_score_cnt > _DONT_BOTHER_SIZE or ram_pressure:
            self.resize_by_score(deme, cutoff)
            did_resize = True

        if self.hc_params.max_allowed_instances < len(deme) or ram_pressure:
            # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:815-819):
            # this used to be `deme.sort(reverse=True)`, on the argument that
            # a stable sort "can only be at least as well-ordered as" an
            # unspecified one, so there was no C++ tie order to match. That
            # argument is FALSE and was withdrawn in S12 after measuring
            # libstdc++ directly: `std::partial_sort` with ties present
            # returns a genuinely different permutation from a stable sort
            # (`all_tied_8_mid4` measures as [1,3,0,2,...] where a stable
            # sort gives [0,1,2,3,...]), and since the very next line
            # TRUNCATES the deme, tie order decides which tied instances
            # survive. `_partial_sort` reproduces libstdc++ exactly; see its
            # block comment.
            #
            # The one thing not reproduced is C++'s out-of-range `middle`:
            # classic passes `next(deme.begin(), max_allowed_instances)`
            # unconditionally, so when the RAM-pressure arm fires on a deme
            # SMALLER than `max_allowed_instances` (the common case -- the
            # default is 10000), `middle` is past `end()` and the call is
            # undefined behaviour. Python clamps `middle` to `len(deme)`,
            # which is what a full sort of the whole deme does and what the
            # `erase` on the next line then makes a no-op -- the observable
            # outcome any sane libstdc++ build produces, without pretending
            # the C++ is defined.
            _partial_sort(
                deme, 0, min(self.hc_params.max_allowed_instances, len(deme)), len(deme)
            )
            del deme[self.hc_params.max_allowed_instances :]
            did_resize = True

        return did_resize

    # ------------------------------------------------------------------ search

    def __call__(
        self,
        deme: deme_t,
        iscorer,
        max_evals: int,
        max_time: int,
        *,
        init_inst: instance | None = None,
    ) -> None:
        """C++: moses/moses/optimization/hill-climbing.cc:49 hill_climbing::operator()
        (``init_inst`` given -- the full algorithm) and hill-climbing.h:262
        hill_climbing::operator() (``init_inst`` omitted -- "assumes that
        init_inst is null"; builds `instance(deme.fields().packed_width())`
        and forwards, hill-climbing.h:264-268). One Python method merges
        both overloads via a trailing keyword-only parameter, matching this
        port's established overload-merging convention -- ``init_inst`` is
        keyword-only specifically so the positional shape
        ``(deme, iscorer, max_evals, max_time)`` -- the one every call site
        in classic actually uses (hill-climbing.h:260-261's own comment) and
        the one `optimizer_base.__call__`'s abstract signature declares --
        stays exactly the base class's shape.

        Search the local neighborhood of an instance for the highest score,
        move to that spot, and repeat until no further improvement is
        possible (hill-climbing.h:135-146's own doc comment).
        """
        if init_inst is None:
            init_inst = instance(deme.fields().packed_width())

        self.nsteps = 0
        self.demeID = deme.getID()
        self.over_budget = False
        start = time.monotonic()

        fields = deme.fields()

        # C++: hill-climbing.cc:72 `_instance_bytes = sizeof(instance) +
        # sizeof(packed_t) * fields.packed_width();` -- see
        # `_SIZEOF_INSTANCE_BYTES`'s own docstring for the `sizeof(instance)`
        # measurement. This was previously `8 * (1 + fields.packed_width())`
        # here, i.e. `8 + 8w` instead of the correct `24 + 8w` -- a real
        # divergence, since fixed. Only `resize_to_fit_ram` (off by default,
        # see `_get_total_ram_bytes`) reads this value.
        self._instance_bytes = _SIZEOF_INSTANCE_BYTES + 8 * fields.packed_width()

        current_number_of_evals = 0
        current_number_of_instances = 0

        max_distance = self.opt_params.max_distance(fields)

        # center_inst is the current location on the hill. Copy it, don't
        # reference it, since sorting will mess up a ref.
        center_inst = instance(init_inst)
        # NOT ported: `best_cscore` (hill-climbing.cc:82,253,362) -- a pure
        # local bookkeeping variable in C++ whose ONLY consumer, in the
        # entire function, is the dropped `logger().debug() << "Best
        # score: " << best_cscore;` line (hill-climbing.cc:362). It is
        # never compared, never fed into a decision, never returned, and
        # never stored on `self`/the deme -- verified by reading every
        # remaining reference to it in hill-climbing.cc. Provably dead once
        # logging is removed, so it is not tracked here at all, rather than
        # kept as an assign-only local that would exist for no reason.
        best_score: score_t = very_worst_score
        best_raw_score: score_t = very_worst_score

        # Initial distance is zero, so that the first time through the
        # loop, we handle just one instance, the initial instance.
        distance = 0
        iteration = 0

        # Needed for correlated neighborhood exploration
        # (hill-climbing.cc:93-95). ASYMMETRY, easy to get wrong: the
        # crossover branch reads the PREVIOUS iteration's block
        # (`prev_start`, `prev_size`, `prev_center`, hill-climbing.cc:165-
        # 166) while the sampling branch reads `center_inst` and the
        # CURRENT `current_number_of_instances` (:190); both then update
        # all three at :204-206, before scoring and before `center_inst`
        # can move. `prev_center` is copied, not aliased, for the same
        # reason `center_inst` is (:79's "Copy it, don't reference it,
        # since sorting will mess up a ref").
        prev_center = instance(center_inst)
        prev_start = 0
        prev_size = 0

        # keep track whether the population generated from the same center
        # has already been crossover. This is to prevent to redo a useless
        # crossover when distance widening is occurs (hill-climbing.cc:97-99)
        already_xover = False
        last_chance = False

        while True:
            iteration += 1

            total_number_of_neighbors = self.estimate_neighborhood(distance, fields)

            number_of_new_instances = self.n_new_instances(
                distance, max_evals, current_number_of_instances, total_number_of_neighbors
            )

            xover_min_neighbors = self.hc_params.crossover_min_neighbors
            large_nbh = total_number_of_neighbors >= xover_min_neighbors
            xover = (
                self.hc_params.crossover
                and (iteration > 2)
                and not already_xover
                and current_number_of_instances >= _XOVER_MIN_DEME
                and (large_nbh or last_chance)
            )

            if xover:
                # NOT ported: the `why_xover` stringstream and the
                # `OC_ASSERT(false, "There must be a bug")` that closes it
                # (hill-climbing.cc:154-162). The assertion is unreachable
                # -- `xover` is only true when `large_nbh || last_chance`,
                # which is exactly the disjunction the two branches above
                # it test -- and the whole block feeds one dropped
                # `logger().debug()`.
                number_of_new_instances = self.crossover(
                    deme,
                    current_number_of_instances,
                    number_of_new_instances,
                    prev_start,
                    prev_size,
                    prev_center,
                )

                already_xover = True
            else:
                # NOT ported: the matching `whynot_xover` chain and its own
                # unreachable `OC_ASSERT(false, ...)`
                # (hill-climbing.cc:170-184).
                #
                # The current_number_of_instances arg is needed only to be
                # able to manage the size of the deme appropriately.
                number_of_new_instances = sample_new_instances(
                    total_number_of_neighbors,
                    number_of_new_instances,
                    current_number_of_instances,
                    center_inst,
                    deme,
                    distance,
                )

            prev_start = current_number_of_instances
            prev_size = number_of_new_instances
            prev_center = instance(center_inst)

            # score all new instances in the deme (hill-climbing.cc:236-240's
            # OMP_ALGO::transform -- a serial loop is faithful, see the
            # module docstring's parallelism note).
            for i in range(current_number_of_instances, len(deme)):
                deme[i].second = iscorer(deme[i].first)

            prev_hi = best_score

            ibest = current_number_of_instances
            for i in range(current_number_of_instances, len(deme)):
                inst_cscore = deme[i].second
                iscore = inst_cscore.get_penalized_score()
                if iscore > best_score:
                    best_score = iscore
                    ibest = i

                rscore = inst_cscore.get_score()
                if rscore > best_raw_score:
                    best_raw_score = rscore

            has_improved = self.opt_params.score_improved(best_score, prev_hi)

            if has_improved:
                center_inst = instance(deme[ibest].first)
                already_xover = False

            current_number_of_evals += number_of_new_instances
            current_number_of_instances += number_of_new_instances

            if self.hc_params.resize_to_fit_ram:
                deme_usage = self._instance_bytes * current_number_of_instances
                if (
                    current_number_of_instances > _ACCEPTABLE_SIZE
                    or _ACCEPTABLE_RAM_FRACTION * self._total_RAM_bytes < deme_usage
                ):
                    did_resize = self.resize_deme(deme, best_score)
                    # "After resizing, some of the variables set above
                    # become invalid. Some are not needed any more: e.g.
                    # ibest. Others, we need, esp prev_start and prev_size
                    # for the cross-over to work."
                    # (hill-climbing.cc:344-348.)
                    if did_resize:
                        current_number_of_instances = len(deme)
                        prev_start = 0
                        prev_size = current_number_of_instances

            if has_improved:
                distance = 1
                deme.n_best_evals = current_number_of_instances
            elif not xover:
                distance += 1

            self.nsteps += 1
            self.total_steps += 1
            self.total_evals += number_of_new_instances

            now = time.monotonic()
            # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:374-378,
            # 419,424): `gettimeofday`/`timersub` give a `struct timeval`
            # whose `.tv_sec` is the INTEGER seconds elapsed -- the budget
            # check below and the `max_time -=` countdown both use ONLY
            # that truncated integer, discarding the sub-second remainder
            # every iteration. `time.monotonic()` replaces `gettimeofday`
            # (a reasonable host-clock substitution, not a fidelity risk --
            # see the module docstring's parallelism note for the same
            # class of "mechanism differs, semantics don't" swap), but the
            # truncation-to-whole-seconds quirk itself is ported verbatim
            # via `int(...)`, DO NOT FIX: a run whose iterations are all
            # sub-second (true of every test in this port's fidelity
            # suite) never actually decrements `max_time` at all.
            elapsed_tv_sec = int(now - start)
            start = now

            if number_of_new_instances == 1 and distance == 1:
                continue

            if max_evals <= current_number_of_evals:
                self.over_budget = True
                break

            if max_time <= elapsed_tv_sec:
                self.over_budget = True
                break
            max_time -= elapsed_tv_sec

            if self.opt_params.terminate_if_gte <= best_raw_score:
                break

            if self.hc_params.crossover:
                big_step = self.opt_params.score_improved(best_score, prev_hi)
                if (not big_step) and (not last_chance):
                    if not already_xover and (
                        not self.hc_params.widen_search or max_distance < distance
                    ):
                        last_chance = True
                        continue
                has_improved = big_step
                last_chance = False

            if max_distance < distance:
                break

            if not has_improved and not self.hc_params.widen_search:
                break

            if self.hc_params.single_step and has_improved:
                break

        deme.n_evals = current_number_of_evals
