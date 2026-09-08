"""C++: moses/moses/optimization/hill-climbing.{h,cc}

`hc_parameters` and `hill_climbing` -- "search the local neighborhood of an
instance for the highest score, move to that spot, and repeat until no
further improvement is possible" (hill-climbing.h:135-137's own doc
comment), ported with **crossover left unimplemented**: the crossover
mechanics themselves (`cross_top_one`, `cross_top_two`, `cross_top_three`,
`crossover`) are S12 (see `doc/cross_top_two_explanation.md` and
`doc/cross_top_three_explanation.md` in the classic tree before ever
attempting them), not this task.

# DIVERGENCE(C++ moses/moses/optimization/hill-climbing.h:43): `hc_parameters`'s
# own constructor default is `cross = true` -- ported faithfully as the
# Python default too (see `hc_parameters.__init__` below), matching classic's
# CLI default for `--hc-crossover` (`problem-params.cc:424`, `hc_crossover_opt`,
# also `true`). But this port cannot actually PERFORM crossover (the mechanics
# are not ported -- see above), so `hill_climbing.__call__` raises
# `NotImplementedError` the moment the crossover trigger
# (hill-climbing.cc:147-150) actually fires, rather than silently skipping it
# or silently doing something else. A caller wanting the algorithm this port
# DOES implement must pass `hc_parameters(cross=False)` -- exactly mirroring
# `--hc-crossover=0` on the command line, per this task's own fidelity
# requirement.

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
# hill-climbing.h:267) / DO NOT FIX: the "Search started from"/"Instance:"
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
            # std::partial_sort(..., std::greater<deme_inst_t>()) -- a full
            # stable sort is behaviourally equivalent for "keep the top
            # max_allowed_instances" (we don't need partial_sort's
            # performance benefit), and std::partial_sort's own tie-order
            # is unspecified in C++ itself, so there is no C++ tie-order to
            # match or diverge from here.
            deme.sort(reverse=True)
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

        # NOT ported: `prev_center`/`prev_start`/`prev_size`
        # (hill-climbing.cc:93-95,204-206) -- "needed for correlated
        # neighborhood exploration" (the C++ comment), but their ONLY
        # reader anywhere in the function is the `crossover(...)` call
        # inside `if (xover)` (hill-climbing.cc:165-166), which this port
        # never reaches (the `xover` branch always raises first -- see the
        # module docstring's DIVERGENCE note). Provably write-only/dead in
        # this port for the same reason `best_cscore` above is, so not
        # tracked here at all.
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
                # DIVERGENCE(C++ moses/moses/optimization/hill-climbing.cc:
                # 152-168): `cross_top_one`/`cross_top_two`/`cross_top_three`/
                # `crossover` are S12, not ported (see module docstring).
                # This guard is what proves the crossover trigger never
                # fires in a supposedly crossover-free run, rather than
                # merely assuming it: any run with `hc_params.crossover =
                # True` that reaches this branch raises instead of silently
                # doing nothing or doing the wrong thing.
                raise NotImplementedError(
                    "hill_climbing: crossover triggered (iteration="
                    f"{iteration}, large_nbh={large_nbh}, last_chance="
                    f"{last_chance}) but crossover mechanics are not "
                    "ported (S12); pass hc_parameters(cross=False), "
                    "matching --hc-crossover=0."
                )

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
                    if did_resize:
                        current_number_of_instances = len(deme)
                        # (hill-climbing.cc:351-352 also resets
                        # prev_start/prev_size here -- not tracked, see above)

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
