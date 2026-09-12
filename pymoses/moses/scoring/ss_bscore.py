"""C++: moses/moses/scoring/ss_bscore.h

"Subsampling behavioral score. Takes a bscore and turn it into subsampling
bscore." (the header's own comment.)

FULLY PORTED AND LIVE. This module used to open with a long note saying
that no `ss_bscore` instance could be constructed in this port, because

    ss_bscore::ss_bscore(...) { _timestamps = _bscorer.get_ctable().
    get_timestamps(); }  (ss_bscore.cc:44-48)

runs UNCONDITIONALLY and `CTable` had no `get_timestamps()`. BOTH halves
of that were stale by the time the `ss-*` task read them, and the second
one was stale when it was written:

* `CTable.get_timestamps()` exists (table.py:1595) and is a real port of
  table.cc:735 -- it returns the empty set for a table loaded without
  `--timestamp-feature` precisely because C++ filters the not-a-date
  sentinel out too (table.cc:740), so the empty result is the CORRECT
  answer for such a table rather than a stub. The `hasattr` guard in
  `__init__` was therefore dead code; it is gone.
* `bscore_ctable_base.ignore_rows`/`ignore_rows_at_times`
  (scoring_base.py) are real now, so `operator()`'s two subsampling
  branches genuinely run against a real bscorer.

`ss_bscore` also now actually SUBCLASSES `bscore_base`, as
ss_bscore.h:38 does -- without that, `behave_cscore ss_cscorer(
ss_bscorer)` (moses_main.h:373) had nothing to bind. See the class
docstring for the two inherited `bscore_base` defaults that turn out to
be load-bearing (`get_complexity_coef()` == 0.0, `size()` == 0).

It is reached from exactly one place, the same one as in classic:
`metapop_moses_results` (moses_main.py / moses_main.h:364), when
`--ss-n-subsample-fitnesses > 1` AND `--ss-low-dev-pressure > 0`.

What is ported, faithfully, and independently testable against a real
`boost::accumulators`/`opencog::RandGen` C++ oracle
(tools/cpp_probe/ss_bscore_probe.cc, tests/fixtures/ss_bscore.json):

* `_VarAcc` -- the `accumulator_set<float, stats<count, mean, variance>>`
  `sc_var_acc` (ss_bscore.cc:57-63) reduces to. `tag::variance` (no
  `(lazy)`/`(immediate)` qualifier) resolves to the ITERATIVE accumulator
  (`boost/accumulators/statistics/variance.hpp`'s `variance_impl`,
  `depends_on<count, immediate_mean>`), not the lazy moment-based one --
  see `_VarAcc`'s own docstring for the exact float32 recurrence this
  reproduces step for step (verified: every individual C++ sub-expression
  rounding reproduced, not just the final per-sample value).
* `_low_dev_penalty` -- `-low_dev_pressure * sqrtf(sc_var)` (ss_bscore.cc:
  122), float32 throughout (`sqrtf`, not `sqrt`: `sc_var`/`low_dev_penalty`
  are both `score_t` in C++, so ADL picks the `float` overload).
* `_partition_by_row` -- the shuffle-then-split branch (ss_bscore.cc:
  82-100): `std::random_shuffle(first, last, gen)` with `gen(i) =
  randGen().randint(i)`, then split into `n_subsamples` equal segments.
* `_partition_by_time` -- the `rand_element_erase`-then-round-robin branch
  (ss_bscore.cc:65-77): repeatedly draws and removes a uniformly-random
  element from a SORTED sequence (modelling `std::set`'s own iteration
  order, since `opencog::rand_element_erase` indexes via
  `std::next(c.begin(), rng.randint(c.size()))`), assigning to subsample
  `i % n_subsamples` in draw order.

`ss_bscore.__call__`'s full body (both `by_time` branches) is below, and
both are now reachable end to end from the CLI. `--ss-by-time` defaults to
FALSE (problem-params.cc), so the DEFAULT SS-fitness run takes the
`_partition_by_row` branch, not the timestamp one.

`ignore_rows`/`ignore_rows_at_times` on `ss_bscore` ITSELF (not the ones it
delegates to on `_bscorer`) are a SEPARATE thing: classic's own
`ss_bscore.cc:143,147` bodies are `OC_ASSERT(false, "Not implemented
yet")` -- genuinely unimplemented IN CLASSIC ITSELF, not a port gap. Ported
verbatim as an unconditional raise.
"""

from __future__ import annotations

import math

from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.moses.moses.types import _to_score_t, behavioral_score, score_t
from pymoses.moses.scoring.scoring_base import bscore_base
from pymoses.util.rng import MT19937RandGen, randGen

__all__ = ["ss_bscore"]


class _VarAcc:
    """C++: boost::accumulators::accumulator_set<float, boost::accumulators::
    stats<tag::count, tag::mean, tag::variance>> -- ss_bscore.cc:57-63's
    `sc_var_acc`, the ONE accumulator instance `operator()` builds per call.

    `tag::variance` with no `(lazy)`/`(immediate)` qualifier resolves to
    `impl::variance_impl` (`boost/accumulators/statistics/variance.hpp:88`),
    which `depends_on<count, immediate_mean>` -- so on every sample, `count`
    and `immediate_mean` update FIRST (mean uses the already-incremented
    count), and `variance_impl::operator()` then reads the ALREADY-UPDATED
    mean. Both `variance_impl` and the OTHER (`lazy_variance_impl`)
    implementation compute the same target statistic, population variance
    (`(1/n) * sum((x_i - mu_n)^2)`, per that header's own doc comment) --
    they differ only in float-rounding ROUTE, and it is the route (not
    merely the statistic) this class reproduces.

    `accumulator_set<float, ...> sc_var_acc;` is DEFAULT-constructed (no
    sample argument) in C++, which the accumulators framework builds
    EAGERLY at that point (not lazily deferred to the first real sample):
    `immediate_mean_impl`'s and `variance_impl`'s constructors both do
    `args[sample | Sample()]`, i.e. "the `sample` named parameter if given,
    else a DEFAULT-CONSTRUCTED `Sample`" -- since no sample is given at
    `accumulator_set`'s own construction, both start at `float() == 0.0`
    (NOT at the first real sample's value, which would be the read if the
    accumulators were instead lazily constructed on first use). Verified
    directly: `tests/fixtures/ss_bscore.json`'s "accumulator" section's
    single-sample cases all show `variance: 0` after exactly one call,
    which only follows from this eager-zero-init reading.

    `immediate_mean_impl::operator()` (mean.hpp:63-71) runs UNCONDITIONALLY
    on every call (including the first, where `cnt == 1`, giving `mean =
    (0*0 + x)/1 == x`); `variance_impl::operator()` (variance.hpp:100-113)
    only updates `variance` when `count(args) > 1` -- on the first sample,
    `variance` keeps its construction-time default (0.0), not `x` itself.

    Formulas reproduced (`Sample = float` throughout, so EVERY named
    intermediate below is its own `_to_score_t` narrowing, matching one
    real C++ sub-expression each -- not one narrowing per Python
    statement):

        mean_new   = (mean_old * (cnt - 1) + x) / cnt                # always
        variance   = variance_old * (cnt - 1) / cnt
                   + (x - mean_new) * (x - mean_new) / (cnt - 1)     # cnt > 1 only

    MEASURED, not merely derived: `tools/cpp_probe/ss_bscore_probe.cc`'s
    "accumulator" section drives the real `boost::accumulators` template
    over six sample sequences (ties, negatives, non-integers, up to 5
    samples) and this class reproduces every intermediate mean/variance
    exactly (`tests/fixtures/ss_bscore.json`, `test_ss_bscore.py`'s
    `test_var_acc_matches_cpp`).
    """

    def __init__(self) -> None:
        self._count = 0
        self._mean: score_t = 0.0
        self._variance: score_t = 0.0

    def __call__(self, x: float) -> None:
        x = _to_score_t(x)
        self._count += 1
        cnt = self._count
        a = _to_score_t(self._mean * (cnt - 1))
        b = _to_score_t(a + x)
        self._mean = _to_score_t(b / cnt)
        if cnt > 1:
            tmp = _to_score_t(x - self._mean)
            term1 = _to_score_t(_to_score_t(self._variance * (cnt - 1)) / cnt)
            term2 = _to_score_t(_to_score_t(tmp * tmp) / (cnt - 1))
            self._variance = _to_score_t(term1 + term2)

    def count(self) -> int:
        return self._count

    def mean(self) -> score_t:
        return self._mean

    def variance(self) -> score_t:
        return self._variance


def _sqrtf(x: float) -> float:
    """Not itself a ported C++ symbol -- `sqrtf`, reproduced the same way
    `hill_climbing.py`'s `_sqrtf_plus_3` reproduces it (see that function's
    own docstring): narrow the argument to float32, take Python's
    (IEEE-correctly-rounded) `math.sqrt`, narrow the result to float32.
    `math.sqrt`'s correct rounding in binary64 makes this exact for
    `sqrtf`, unlike `cbrt` (that docstring's own caveat).
    """
    return _to_score_t(math.sqrt(_to_score_t(x)))


def _boost_accumulate(bs: behavioral_score) -> float:
    """C++: `boost::accumulate(psc, 0.0)` (ss_bscore.cc:74,88,105) -- the
    SAME naive left-to-right double-precision fold `bscore_base.sum_bscore`
    (scoring_base.py) reproduces for its own, textually identical, C++ call
    (`scoring_base.cc:139`). Deliberately NOT Python's builtin `sum()`,
    which (CPython >=3.12) uses compensated summation and is systematically
    more accurate -- see `bscore_base.sum_bscore`'s own docstring for the
    measured counter-example. `ss_bscore.cc` calls the RAW `boost::
    accumulate` directly, not `_bscorer.sum_bscore()` -- bypassing whatever
    weighted-score mode `_bscorer` might itself be in -- so this port must
    not delegate to `self._bscorer.sum_bscore(...)` either; a local,
    unweighted, un-narrowed-per-element accumulator matching that same
    unweighted branch exactly is reproduced here instead.
    """
    total = 0.0
    for x in bs:
        total += x
    return total


def _low_dev_penalty(sc_var: float, low_dev_pressure: float) -> score_t:
    """C++: moses/moses/scoring/ss_bscore.cc:120-122

    "Compute the low deviation penalty. No variance corresponds to no
    penalty, the penalty grows with the variance" (the .cc file's own
    comment.) `score_t sc_var = variance(sc_var_acc), low_dev_penalty =
    -_low_dev_pressure * sqrt(sc_var);` -- `sc_var`/`low_dev_penalty` are
    both `score_t` (float), so `sqrt` resolves to the `float sqrt(float)`
    overload (`sqrtf`), not double `sqrt`.
    """
    return _to_score_t(-_to_score_t(low_dev_pressure) * _sqrtf(sc_var))


def _shuffle_with_gen(items: list, rand) -> None:
    """C++: /usr/include/c++/*/bits/stl_algo.h's 3-argument
    `std::random_shuffle(first, last, rand)` (deprecated post-C++14, still
    compiles; `ss_bscore.cc:97` calls exactly this overload, not
    `boost::random_shuffle` -- contrast `deme_expander.cc`, which is a
    DIFFERENT call site using the boost version, out of this file's scope).

    In-place, forward pass: for `i` in `[1, len(items))`, swap `items[i]`
    with `items[rand(i + 1)]` (`rand(n)` must return a value uniformly in
    `[0, n)`). Verified against the installed libstdc++ header source
    directly (this port's own toolchain, matching this project's
    CARRY-FORWARD.md Sec 5.1 concern about libstdc++-version-dependent
    algorithms -- this one, unlike `std::sort`/`std::partial_sort`, has had
    a STABLE, unchanged body across libstdc++ versions since C++98; it is
    also DEPRECATED and slated for removal, a risk noted here rather than
    hidden).
    """
    n = len(items)
    for i in range(1, n):
        j = rand(i + 1)
        if i != j:
            items[i], items[j] = items[j], items[i]


def _partition_by_row(
    usize: int, n_subsamples: int, rng: MT19937RandGen | None = None
) -> list[set[int]]:
    """C++: moses/moses/scoring/ss_bscore.cc:82-100 (the `by_time == false`
    branch's own row-partitioning prelude, up to but not including the per-
    subsample scoring loop).

    Reproduces, in order: build `[0, usize)`, shuffle it in place via
    `std::random_shuffle(first, last, gen)` with `gen(i) =
    randGen().randint(i)` (`RandGen::randint(int)`, i.e. this port's
    `rng.randint_n`), then slice the shuffled sequence into `n_subsamples`
    consecutive segments of `usize // n_subsamples` elements each --
    `usize // n_subsamples` truncates exactly like C++'s unsigned `/`
    for these always-non-negative operands.

    # DIVERGENCE(C++ moses/moses/scoring/ss_bscore.cc:90-93): when `usize <
    # n_subsamples`, `seg_size` truncates to 0, and C++'s `for (it =
    # begin; it != begin + n_subsamples * 0; it += 0)` condition is
    # `it != begin`, already true is FALSE at entry (`it == begin`), so
    # the loop runs ZERO times and `ignore_row_idxs_per_fitness` stays
    # EMPTY -- yet the very next loop below indexes it
    # `ignore_row_idxs_per_fitness[i]` for `i` in `[0, n_subsamples)`,
    # an out-of-bounds `std::vector::operator[]` (genuine, unchecked UB in
    # C++, not an `OC_ASSERT`). This function returns an EMPTY list for
    # that case rather than reproducing unspecified memory access; a
    # caller indexing the result out of range gets Python's own
    # `IndexError` instead of C++'s undefined behaviour -- a real but
    # strictly-safer failure mode, matching this project's own
    # `move_ontop`/CARRY-FORWARD 3 precedent for the same shape of bug.
    """
    if rng is None:
        rng = randGen()
    row_idxs = list(range(usize))
    _shuffle_with_gen(row_idxs, lambda i: rng.randint_n(i))
    if n_subsamples == 0:
        return []
    seg_size = usize // n_subsamples
    groups: list[set[int]] = []
    if seg_size > 0:
        limit = n_subsamples * seg_size
        for start in range(0, limit, seg_size):
            groups.append(set(row_idxs[start : start + seg_size]))
    return groups


def _partition_by_time(
    timestamps: set, n_subsamples: int, rng: MT19937RandGen | None = None
) -> list[set]:
    """C++: moses/moses/scoring/ss_bscore.cc:65-77 (the `by_time == true`
    branch's own timestamp-partitioning prelude).

    Reproduces:
        std::set<TTable::value_type> timestamps(_timestamps);
        unsigned tt_size = timestamps.size();
        for (unsigned i = 0; i < tt_size; ++i) {
            TTable::value_type time = rand_element_erase(timestamps);
            ignore_timestamps_per_ss_deme[i % _n_subsamples].insert(time);
        }
    `opencog::rand_element_erase` (`opencog/util/random.h:92`) is
    `std::next(c.begin(), rng.randint(c.size()))`, then erase -- since
    `std::set` iterates in SORTED order, this draws a uniformly-random
    SORTED-POSITION index (`rng.randint_n(len(remaining))`) each round and
    removes that element, exactly modelled here with a plain sorted
    `list` standing in for the `std::set` (`.pop(idx)` matching
    `std::next(...)+erase`) -- the input `timestamps` need not already be
    sorted; this function sorts it first, matching how C++'s
    `std::set<TTable::value_type> timestamps(_timestamps)` copy-constructs
    from the (already ordered) `_timestamps` member regardless of any
    caller's own insertion order.

    Draw order matters and is verified element-for-element against a real
    `opencog::rand_element_erase` C++ oracle (`tests/fixtures/ss_bscore.
    json`'s "partition_by_time" section, including an out-of-sorted-order
    input set and an empty-timestamps edge case).
    """
    if rng is None:
        rng = randGen()
    remaining = sorted(timestamps)
    groups: list[set] = [set() for _ in range(n_subsamples)]
    tt_size = len(remaining)
    for i in range(tt_size):
        idx = rng.randint_n(len(remaining))
        time = remaining.pop(idx)
        groups[i % n_subsamples].add(time)
    return groups


class ss_bscore(bscore_base):
    """C++: moses/moses/scoring/ss_bscore.h:38 ss_bscore

    `struct ss_bscore : public bscore_base` (ss_bscore.h:38) -- so the
    Python class subclasses `bscore_base` too. It did NOT, until the
    `ss-*` subsample task, which made the class "implemented but
    unconstructible": every method below existed and was tested in
    isolation, but nothing could put an instance where classic puts one
    (`behave_cscore ss_cscorer(ss_bscorer);`, moses_main.h:373, takes a
    `bscore_base&`).

    Inheriting the base matters for more than typing. `ss_bscore`
    overrides only what ss_bscore.cc overrides; everything else is
    `bscore_base`'s own body, and two of those inherited defaults are
    load-bearing and counter-intuitive:

    * `get_complexity_coef()` returns `bscore_base`'s OWN
      `_complexity_coef`, which its constructor sets to `0.0`
      (scoring_base.h:59) and which nothing ever sets on an `ss_bscore`
      -- NOT the wrapped `_bscorer`'s coefficient, which the CLI did set
      (`problem-params.cc`). So turning on `--ss-n-subsample-fitnesses>1
      --ss-low-dev-pressure>0` silently drops the complexity penalty from
      every composite score for the whole run. Classic's, reproduced, not
      repaired.
    * `size()` returns `_size`, likewise never assigned here, so `0`.

    See this module's own docstring for what is ported and how each piece
    was verified against a real C++ oracle.
    """

    def __init__(
        self,
        bscorer: bscore_base,
        n_subsamples: int = 0,
        low_dev_pressure: float = 0.0,
        by_time: bool = True,
    ) -> None:
        """C++: moses/moses/scoring/ss_bscore.cc:44 ss_bscore::ss_bscore

        `_low_dev_pressure` is declared `float` in C++ (ss_bscore.h:74),
        narrowed here at construction to match, not merely at point of use.

        `_timestamps = _bscorer.get_ctable().get_timestamps();` runs
        UNCONDITIONALLY, exactly as C++ does regardless of `by_time`/
        `n_subsamples`. A `NotImplementedError` used to be raised right
        here, guarded by `hasattr(ctable, "get_timestamps")`, on the
        stated grounds that this port's `CTable` had no such method. That
        guard was already DEAD when the `ss-*` task inherited it:
        `CTable.get_timestamps` exists (table.py) and returns the empty
        set for a table loaded without `--timestamp-feature` -- which is
        the CORRECT answer, not a stub, since every observation then
        carries the not-a-date sentinel the C++ original also filters out
        (table.cc:740). Removed, so the constructor is a plain four-line
        transliteration again.

        `get_ctable()` is `bscore_ctable_base.get_ctable` (scoring_base.py)
        on every real bscorer this reaches, and returns `_orig_ctable` --
        the WHOLE table, never a subsample -- so `_timestamps` is fixed
        for the run, matching C++.
        """
        super().__init__()
        self._bscorer = bscorer
        self._n_subsamples = n_subsamples
        self._low_dev_pressure: score_t = _to_score_t(low_dev_pressure)
        self._by_time = by_time
        self._timestamps = self._bscorer.get_ctable().get_timestamps()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/ss_bscore.cc:52 ss_bscore::operator()

        Fast path (`n_subsamples == 0 or low_dev_pressure == 0.0`, C++'s
        default construction) needs no timestamps at all and is the ONLY
        path reachable without the constructor gap above; see
        test_ss_bscore.py for how it is exercised via a bypassed `__init__`.

        The remainder (both `by_time` branches) is written faithfully,
        using `_partition_by_time`/`_partition_by_row`/`_VarAcc`/
        `_low_dev_penalty` above, but is unreachable through this class
        while `self._timestamps` can never be set -- see this module's own
        docstring.
        """
        if self._n_subsamples == 0 or self._low_dev_pressure == 0.0:
            return self._bscorer(tr)

        sc_var_acc = _VarAcc()

        if self._by_time:
            groups = _partition_by_time(self._timestamps, self._n_subsamples)
            for i in range(self._n_subsamples):
                self._bscorer.ignore_rows_at_times(groups[i])
                psc = self._bscorer(tr)
                sc_var_acc(_boost_accumulate(psc))
        else:
            usize = self._bscorer.get_ctable_usize()
            groups = _partition_by_row(usize, self._n_subsamples)
            for i in range(self._n_subsamples):
                self._bscorer.ignore_rows(groups[i])
                psc = self._bscorer(tr)
                sc_var_acc(_boost_accumulate(psc))

        # Recompute the bscore over all data points.
        self._bscorer.ignore_rows(set())
        psc = list(self._bscorer(tr))

        sc_var = sc_var_acc.variance()
        low_dev_penalty = _low_dev_penalty(sc_var, self._low_dev_pressure)

        psc.append(low_dev_penalty)
        return psc

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/ss_bscore.cc:130 ss_bscore::best_possible_bscore"""
        return self._bscorer.best_possible_bscore()

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/ss_bscore.cc:134 ss_bscore::min_improv

        Plain forward to `_bscorer.min_improv()` -- note the HEADER's own
        comment ("return the min of all min_improv", ss_bscore.h:50) is
        stale/misleading relative to this .cc body: there is only ONE
        wrapped `_bscorer`, not "all" of anything. Ported the .cc
        behaviour (a plain forward), not the header comment's implication.
        """
        return self._bscorer.min_improv()

    def ignore_cols(self, idxs: set) -> None:
        """C++: moses/moses/scoring/ss_bscore.cc:138 ss_bscore::ignore_cols"""
        self._bscorer.ignore_cols(idxs)

    def ignore_rows(self, idxs: set) -> None:
        """C++: moses/moses/scoring/ss_bscore.cc:142 ss_bscore::ignore_rows

        `OC_ASSERT(false, "Not implemented yet")` -- genuinely unimplemented
        IN CLASSIC ITSELF (not a port gap): `ss_bscore`'s OWN `ignore_rows`
        (distinct from `_bscorer.ignore_rows`, which `__call__` above DOES
        call) always fails. Ported as an unconditional raise.
        """
        raise NotImplementedError("Not implemented yet")

    def ignore_rows_at_times(self, timestamps: set) -> None:
        """C++: moses/moses/scoring/ss_bscore.cc:146 ss_bscore::ignore_rows_at_times

        Same status as `ignore_rows` above: `OC_ASSERT(false, "Not
        implemented yet")` in classic itself.
        """
        raise NotImplementedError("Not implemented yet")

    def get_ctable_usize(self) -> int:
        """C++: moses/moses/scoring/ss_bscore.cc:150 ss_bscore::get_ctable_usize"""
        return self._bscorer.get_ctable_usize()

    def get_ctable(self):
        """C++: moses/moses/scoring/ss_bscore.cc:154 ss_bscore::get_ctable"""
        return self._bscorer.get_ctable()
