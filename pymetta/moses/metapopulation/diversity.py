"""C++: moses/moses/metapopulation/diversity.cc

P2 metapopulation task: ports `metapopulation::set_diversity`,
`metapopulation::gather_diversity_stats`, and the three `cached_dst::`
methods diversity.cc defines (`operator()`, `erase_ptr_seq`,
`gather_stats`) -- the diversity-PENALTY machinery `merging.py`'s
`merge_demes` has always gated on `diversity_enabled()`
(`metapop_parameters.diversity.enabled()`, `pressure > 0.0`, off by
default) and raised on. `diversity_enabled()` itself, and
`diversity_parameters`/`p_norm_distance`/`tanimoto_distance`/
`angular_distance`, were already ported (`metapopulation.py`,
`metapop_params.py`) -- this file is the part that was missing:
the actual penalty computation.

## `ENABLE_DST_CACHE` is commented out -- the cache never actually caches

`ENABLE_DST_CACHE` is commented out at its own definition site
(`// #define ENABLE_DST_CACHE`, metapopulation.h:53), so in EVERY build
this port targets, `cached_dst::operator()` compiles to its `#else`
branch alone (diversity.cc:254-256): `return _dparams.dst(cl->
get_bscore(), cr->get_bscore());` -- no cache read, no cache write.
`erase_ptr_seq`'s entire body is inside the SAME `#ifdef` (diversity.cc:
264-270), so it compiles to nothing at all -- a true no-op, the same one
`merging.py`'s `resize_metapop` already documented for why it never calls
it. Ported here as exactly that: `__call__` never touches `self.cache`;
`erase_ptr_seq` does nothing.

## `gather_stats()` on a cache that is ALWAYS empty -- measured, not assumed

Because the cache above never gets a single entry in any build this port
targets, `cached_dst.gather_stats()` (called by `gather_diversity_stats`
whenever `n < 0`) always runs boost::accumulators' `count`/`mean`/
`variance`/`min`/`max` over ZERO samples. This is a genuine, real-build
degenerate case in classic itself, not a port artifact -- measured
directly with a standalone C++ probe (`boost::accumulators::
accumulator_set<double, stats<count, mean, variance, min, max>>`, kept
out of this repo) over an empty accumulator:

    count=0 mean=nan variance=0 min=1.7976931348623157e+308
    max=-1.7976931348623157e+308  (std = sqrt(variance) = 0, NOT nan)

`min`/`max` come back as (positive) `DBL_MAX`/`-DBL_MAX`
(`sys.float_info.max`/`-sys.float_info.max` here) -- boost's running
min/max accumulators simply never update their sentinel initial value
when zero samples are pushed. `_gather_stats_over` below reproduces this
exact degenerate case for an empty input, and the ordinary formulas
(population mean/variance, matching a separate probe's measured
non-empty values exactly) for a non-empty one -- `gather_diversity_stats`
with `n >= 0` reaches the non-empty path in every case that is not
itself degenerate (`n == 0` or an empty metapopulation), so the general
formula is real, tested code, not just there for symmetry.

## `dp_t`/`score_t` vs `double` -- two distinct C++ types in this one file

`diversity_parameters::dp_t` IS `score_t` (`typedef score_t dp_t;`,
metapop_params.h:34) -- every individual distance (`dst(...)`) and
uniformity penalty (`dst2dp(...)`, `aggregated_dps(...)`) computed below
is `score_t`-typed and narrowed via `_to_score_t` at each step a real
`float` variable would be re-rounded, the same convention this whole
project applies to every other `score_t` arithmetic chain. `diversity_
stats`'s five fields, by contrast, are plain C++ `double`
(`metapopulation.py`'s own class docstring) -- `boost::accumulators::
accumulator_set<double, ...>` (diversity.cc:190,282) computes in FULL
DOUBLE PRECISION even though every individual sample fed into it
(`_cached_dst(...)`'s return) is `dp_t`/`score_t`: each `float` sample
promotes to `double` the instant it is passed to `acc(...)`, and the
running sum/mean/variance never narrow back down. `_gather_stats_over`
below takes already-`score_t`-narrowed floats as input but performs its
OWN aggregation in plain Python `float` (already double precision, so
this needs no `_to_score_t` calls of its own) -- narrowing the
aggregation itself would be the CARRY-FORWARD 4.2 mistake in reverse
(over-narrowing a `double` computation because a `score_t` fed into it).

## `set_diversity`'s greedy loop is inherently order-sensitive -- unlike domination.py

Unlike `domination.py`'s recursive skyline (provably order-independent,
see that module's own docstring), `set_diversity`'s outer loop picks ONE
best-scoring candidate per iteration and uses it as the sole reference
point (`last_ptr`) for every remaining candidate's NEXT incremental
penalty update -- a genuinely sequential, path-dependent process where
WHICH candidate is picked first changes every later penalty. This port
reproduces the exact selection rule (`sct_score_greater`, the same total
order `_scored_trees` itself is sorted by, imported and reused rather
than reimplemented) via a manual `std::min_element`-with-a-greater-than-
comparator translation: scan left to right, replacing the running "best"
only when the CURRENT candidate compares strictly greater than it --
this keeps the FIRST occurrence on a genuine tie (two candidates with
identical composite_score, raw score AND tree structure -- possible in
principle per `scored_combo_tree_ptr_set`'s own "no way to guarantee a
tree appears only once" caveat, types.py), exactly matching `std::
min_element`'s own left-to-right, first-occurrence-wins tie behaviour.

## Composition

Same free-function + `setattr` shape as `merging.py`/`domination.py`:
`set_diversity`/`gather_diversity_stats`/`aggregated_dps`/`distort_dp`
bind onto `metapopulation`; `__call__`/`erase_ptr_seq`/`gather_stats`
bind onto `cached_dst` (declared, constructor only, in metapopulation.py
-- see that class's own docstring for why). `merging.py` imports this
module for its `setattr` side effect (same chain: `__init__.py` ->
`merging` -> `diversity`, alongside `-> domination`), keeping the
"import the package, get the complete method set" contract
`test_package_binding.py` pins for BOTH classes this module extends.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.metapopulation import (
    cached_dst,
    diversity_stats,
    metapopulation,
)
from pymetta.moses.moses.types import (
    _to_score_t,
    score_t,
    scored_combo_tree,
    scored_combo_tree_ptr_set,
    sct_score_greater,
)

if TYPE_CHECKING:
    pass

__all__: list[str] = []
# Nothing new is exported: every symbol below is bound onto `metapopulation`
# or `cached_dst` (see module docstring's Composition section), matching
# merging.py's/domination.py's own convention.

_bsct_gt = sct_score_greater()
# C++: moses/moses/metapopulation/diversity.cc:140 `sct_score_greater bsct_gt;`
# Constructed once at module scope, not per-call: `sct_score_greater` is
# stateless (types.py), so this is purely an allocation-avoidance detail
# with no observable effect, matching the "cheap, no logic of its own"
# bar this port already applies to similar singletons.


def _gather_stats_over(values: list[float]) -> diversity_stats:
    """Not itself a ported C++ symbol -- shared boost::accumulators
    reproduction backing both `set_diversity`... no, backing `cached_dst.
    gather_stats` and `metapopulation.gather_diversity_stats`'s own
    `n >= 0` branch (diversity.cc:189-224,278-302 -- the SAME five-stat
    accumulator_set, instantiated twice in C++, unified into one helper
    here). Population mean/variance (divide by `len(values)`, not
    `len(values) - 1`) -- measured directly against boost's own output
    for 3 known samples (module docstring); the empty-input case
    reproduces boost's own measured degenerate output for zero samples
    exactly (module docstring), rather than raising or returning zeros.
    """
    n = len(values)
    if n == 0:
        return diversity_stats(
            count=0.0,
            mean=math.nan,
            std=0.0,
            min=sys.float_info.max,
            max=-sys.float_info.max,
        )
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return diversity_stats(
        count=float(n),
        mean=mean,
        std=math.sqrt(variance),
        min=min(values),
        max=max(values),
    )


def __cached_dst_call(self: cached_dst, cl: scored_combo_tree, cr: scored_combo_tree) -> score_t:
    """C++: moses/moses/metapopulation/diversity.cc:228 metapopulation::cached_dst::operator()

    Only the `#else` (no-cache) branch is reachable in any build this
    port targets -- see module docstring. `_dparams.dst(...)` is
    `diversity_parameters.dst` (metapop_params.py, already ported),
    called with each tree's OWN `get_bscore()` directly, matching C++'s
    `cl->get_bscore()`/`cr->get_bscore()`.
    """
    return self._dparams.dst(cl.get_bscore(), cr.get_bscore())


def __cached_dst_erase_ptr_seq(self: cached_dst, ptr_seq: list[scored_combo_tree]) -> None:
    """C++: moses/moses/metapopulation/diversity.cc:262 metapopulation::cached_dst::erase_ptr_seq

    "Remove all keys containing any element of ptr_seq" (the C++ doc
    comment) -- entire body is `#ifdef ENABLE_DST_CACHE` (undefined in
    every build this port targets, see module docstring): a true no-op.
    """


def __cached_dst_gather_stats(self: cached_dst) -> diversity_stats:
    """C++: moses/moses/metapopulation/diversity.cc:278 metapopulation::cached_dst::gather_stats

    "Gather some statistics about the diversity of the population" (the
    C++ doc comment). `self.cache` is always empty in every build this
    port targets (module docstring) -- ported as the general accumulation
    `_gather_stats_over` performs over `self.cache.values()` (structurally
    faithful to `for (const auto& v : cache) acc(v.second);`), which
    degrades to the measured empty-accumulator case in practice.
    """
    return _gather_stats_over(list(self.cache.values()))


def distort_dp(self: metapopulation, dp: score_t) -> score_t:
    """C++: moses/moses/metapopulation/metapopulation.h:290 metapopulation::distort_dp

    "Distort a uniformity penalty component between 2 candidates.
    (actually not used apart from a comment of aggregated_dps)" (the
    header's own doc comment, verbatim -- ZERO real callers anywhere in
    classic, confirmed by grep across the whole `moses/moses/
    metapopulation/` tree). Ported anyway (cheap, one line, matches this
    port's habit of including every inline one-liner a header declares
    alongside the methods that ARE used -- `useful_score_range`'s own
    precedent, `merging.py`) but not independently verifiable: there is
    no oracle path (no caller) to test it against, so this is reported,
    not claimed as covered.

    `pow(dp, _params.diversity.exponent)`: both operands are `dp_t`
    (`score_t`/`float`) in C++, using the `float pow(float, float)`
    overload rather than promoting to `double` -- narrowed via
    `_to_score_t` once, at the return, matching that single-precision
    computation as closely as Python's own `**` (computed in double
    internally regardless) can.
    """
    return _to_score_t(dp ** self._params.diversity.exponent)


def aggregated_dps(self: metapopulation, ddp_sum: score_t, n: int) -> score_t:
    """C++: moses/moses/metapopulation/metapopulation.h:298 metapopulation::aggregated_dps

    "The inverse function of distort_dp normalized by the vector size.
    Basically aggregated_dps(sum_i distort_dp(x_i), N) == generalized_
    mean(x) where N is the size of x" (the header's own doc comment).

    `ddp_sum / N`: `dp_t / unsigned` in C++ stays `float` (an `unsigned`
    operand promotes to `float`, not `double`, when paired with a
    `float`) -- narrowed via `_to_score_t` as its own step. `1.0 /
    _params.diversity.exponent`: a C++ `double` LITERAL divided by a
    `score_t` promotes the WHOLE expression to `double` -- computed here
    in plain Python `float` (already double precision), narrowed only
    where the function's own C++ return type (`dp_t`) applies the FINAL
    narrowing, at the very end. `pow(float_narrowed_ratio,
    double_reciprocal_exponent)`: C++ `std::pow`'s "additional overloads"
    promote the mixed-type call to `double` for the actual computation,
    narrowing back to `dp_t` only via this method's own return -- ported
    as one `_to_score_t` at the end, not at each factor.
    """
    ratio = _to_score_t(ddp_sum / n)
    return _to_score_t(ratio ** (1.0 / self._params.diversity.exponent))


def set_diversity(self: metapopulation) -> None:
    """C++: moses/moses/metapopulation/diversity.cc:44 metapopulation::set_diversity

    "If the uniformity penalty is enabled, then punish the scores of
    those exemplars that are too similar to the previous ones"
    (metapopulation.h:296-298's own doc comment, condensed). See module
    docstring for why this greedy, sequential process is NOT reorderable
    the way `domination.py`'s recursive skyline is.

    `OC_ASSERT(bsct.get_bscore().size(), ...)`/`OC_ASSERT(last_dst >=
    0.0, ...)` (diversity.cc:84-95) -- ported as explicit `AssertionError`s,
    the same convention every other bare `OC_ASSERT` in this project's
    scope uses (`metapopulation.py`'s `select_exemplar`/
    `update_best_candidates`, cited there).

    `pool.transfer(mit->first, _scored_trees)` (diversity.cc:152) MOVES
    the exact same heap-allocated `scored_combo_tree` from `_scored_trees`
    into `pool` -- no copy -- via the ITERATOR `mit->first` directly
    (`psi`, an iterator INTO `_scored_trees` obtained once, up front, at
    diversity.cc:57), never by re-searching `_scored_trees` by value.
    That distinction is load-bearing for THIS port, not merely a style
    choice to mirror: `_scored_trees` (`scored_combo_tree_ptr_set`,
    types.py) is backed by `_sct_comparator_set`, a SORTED PYTHON LIST
    plus BINARY SEARCH, not a tree structure whose nodes are found by
    position the way a C++ iterator is. This function mutates OTHER
    candidates' `composite_score` (hence their `sct_score_greater` sort
    key) on every outer-loop pass, BEFORE any of them have been removed
    -- so a value-based `_scored_trees.erase(sct)` call partway through
    the loop would bisect over a list whose sortedness invariant these
    very mutations have already broken, and could fail to find the
    element it is looking for (or find the wrong one). This port
    sidesteps the hazard entirely instead of risking it: `tmp` (below)
    already holds every original member, extracted ONCE via `list(self.
    _scored_trees)` while the container was still validly sorted;
    `_scored_trees` itself is never read from OR written to again until
    the very end, so no bisection ever runs against a list this
    function has mutated the ordering key of. `pool_members` (a plain
    Python list, order = the order candidates were picked in) plays the
    role of `pool` without needing `scored_combo_tree_ptr_set`'s own
    sorted structure at all until the final rebuild below.

    `_scored_trees.swap(pool)` (diversity.cc:168) -- ported as building a
    FRESH `scored_combo_tree_ptr_set()` from `pool_members` (each
    `insert()` re-sorts by the candidates' NOW-FINAL, fully-updated
    composite scores, which is exactly what real C++ needs `pool` to
    have been built as all along -- `pool.transfer` inserts into a proper
    `boost::ptr_set`, which sorts on every insert too) and reassigning
    `self._scored_trees` to it: Python has no aliasing caller anywhere
    that holds a separate reference to the OLD `_scored_trees` object and
    expects to observe the swap through it (the same reasoning
    `metapopulation.clear()`'s own docstring gives for its own
    reassignment-instead-of-`.clear()` choice).
    """
    dp_max = self._params.diversity.exponent <= 0.0

    tmp: list[list] = [[sct, _to_score_t(0.0)] for sct in list(self._scored_trees)]
    pool_members: list[scored_combo_tree] = []
    last_ptr: scored_combo_tree | None = None

    while tmp:
        if pool_members:
            for v in tmp:
                bsct = v[0]
                if not bsct.get_bscore():
                    raise AssertionError("Behavioral score is needed for diversity!")
                last_dst = self._cached_dst(bsct, last_ptr)
                if last_dst < 0.0:
                    raise AssertionError(
                        "The distance cannot be negative. There could be a bug or "
                        "you are not using a true distance."
                    )
                last_dp = self._params.diversity.dst2dp(last_dst)
                last_ddp = (
                    last_dp if dp_max else _to_score_t(last_dp**self._params.diversity.exponent)
                )

                if dp_max:
                    v[1] = max(v[1], last_ddp)
                    adp = v[1]
                else:
                    v[1] = _to_score_t(v[1] + last_ddp)
                    n = len(pool_members) if self._params.diversity.normalize else 1
                    adp = self.aggregated_dps(v[1], n)

                if (
                    self._params.diversity.dst2dp_type
                    == self._params.diversity.dst2dp_enum_t.pthpower
                ):
                    bsct.get_composite_score().multiply_diversity = True
                bsct.get_composite_score().set_uniformity_penalty(adp)

        best_idx = 0
        for i in range(1, len(tmp)):
            if _bsct_gt(tmp[i][0], tmp[best_idx][0]):
                best_idx = i

        best_sct = tmp[best_idx][0]
        last_ptr = best_sct
        pool_members.append(best_sct)
        del tmp[best_idx]

    new_pool = scored_combo_tree_ptr_set()
    for sct in pool_members:
        new_pool.insert(sct)
    self._scored_trees = new_pool


def gather_diversity_stats(self: metapopulation, n: int) -> diversity_stats:
    """C++: moses/moses/metapopulation/diversity.cc:184 metapopulation::gather_diversity_stats

    "Gather statistics about the diversity of the n best candidates (if n
    is negative then all candidates are included)" (the header's own doc
    comment). `n < 0` delegates to `_cached_dst.gather_stats()` -- always
    the measured empty-cache degenerate case in every build this port
    targets (module docstring); `n >= 0` computes real pairwise distances
    among the top `min(n, size())` candidates in `_scored_trees`'s own
    `sct_score_greater` order.

    `ENABLE_DST_CACHE` branch (diversity.cc:203-208) is not ported --
    dead in every build this port targets, same as `cached_dst`'s own
    `#else`-only reachability (module docstring); this always takes the
    `#else` arm, `acc(_cached_dst(&*from_j, &*from_i))` (diversity.cc:210).
    """
    if n < 0:
        return self._cached_dst.gather_stats()

    ordered = list(self._scored_trees)
    top = ordered[: min(n, self.size())]
    values: list[float] = []
    for i in range(len(top)):
        for j in range(i):
            values.append(self._cached_dst(top[j], top[i]))
    return _gather_stats_over(values)


metapopulation.distort_dp = distort_dp
metapopulation.aggregated_dps = aggregated_dps
metapopulation.set_diversity = set_diversity
metapopulation.gather_diversity_stats = gather_diversity_stats

cached_dst.__call__ = __cached_dst_call
cached_dst.erase_ptr_seq = __cached_dst_erase_ptr_seq
cached_dst.gather_stats = __cached_dst_gather_stats
