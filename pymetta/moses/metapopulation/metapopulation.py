"""C++: moses/moses/metapopulation/metapopulation.{h,cc}

Ports the CORE of `metapopulation` only, per this task's own brief:
`__init__`/`init`, `best_candidates`, `best_tree`, `best_composite_score`,
`get_trees`, `size`, `empty`, `clear`, `update_best_candidates`, `rescore`,
`diversity_enabled`, and the `_scored_trees` pool. A handful of other
one-line accessors the header declares right alongside these (`get_ensemble`,
`best_score`, `get_cscorer`) are ported too -- each is a single delegating
call with no logic of its own, so including them costs nothing and lets
`best_composite_score`'s own `best_score()` caller (metapopulation.h:148)
work.

NOT ported here, deliberately, because every method that would read them is
itself out of this task's scope (see the S10 plan's own per-task split):

* `select_exemplar`/`has_been_visited`/`log_selected_exemplar` and the
  `_visited_exemplars` field they share -- exemplar selection is not this
  task.
* `resize_metapop`/`_merge_count` -- population trimming.
* `set_diversity`/`gather_diversity_stats`/`_cached_dst`/`cached_dst` --
  the diversity penalty machinery (`diversity_enabled()` itself, the
  trivial gate accessor, IS ported: it is a one-line delegation to
  `metapop_parameters.diversity.enabled()`, metapop_params.py, and reads
  no diversity-specific state of its own).
* `remove_dominated`/`get_nondominated_disjoint`/`get_nondominated_rec`/
  `merge_nondominated`/`dominates`/`to_set`/`split` -- domination.cc, which
  is not in ANY file this whole S10 slice touches (deme_expander.{h,cc},
  deme_params.h, metapopulation.{h,cc}, merging.cc, metapop_params.{h,cc}
  -- the S10 plan's own file list). `metapopulationUTest.cxxtest` (the
  slice's registered cxxtest oracle) tests exactly these two domination
  methods and nothing else -- see this module's own test file for why that
  makes the packaged oracle inapplicable to Task 2's actual scope, and
  `tools/cpp_probe/metapop_core_probe.cc` for the probe written to cover
  it instead.
* `merge_candidates`/`merge_demes`/`sort_demes`/`keep_top_unique_candidates`/
  `trim_down_deme`/`deme_to_trees`/`get_new_candidates`/subsampling
  (`ss_*`, `recompute_scores_over_whole_dataset`) -- merging.cc's own
  larger machinery, explicitly a LATER task ("Not this task: merge_candidates
  / merge_demes ... Leave clean seams for them"). `init()` below still
  needs SOMETHING to populate `_scored_trees` (this task's own scope names
  that pool explicitly) -- see `init`'s own docstring for the seam this
  leaves.
* `ostream_metapop`/`log_best_candidates`/`log_selected_exemplar` --
  printing/logging, no observable effect on anything this port asserts
  against (the same "logger calls are not ported" convention every prior
  slice in this port has followed; see e.g. `hill_climbing.py`'s own
  module docstring).
* `_merge_mutex`/threading in general -- no file in this port anywhere
  uses `threading.Lock`; C++'s mutexes exist for multi-threaded callers
  this single-threaded port has none of (confirmed by search: `rg
  "threading|Lock\\(\\)|mutex"` over `pymetta/` turns up zero actual uses).

Gates (verified in source, not inferred -- see metapop_params.py's own
module docstring for the same table):

* `do_boosting` (`metapop_parameters.do_boosting`, default `False`,
  `metapop_params.h:170`) gates `best_composite_score`/`best_tree`'s
  boosting arm. Both reach straight into Task 1's `ensemble` (`ensemble.py`)
  rather than raising here separately -- `ensemble.get_ensemble()` itself
  never raises (it is a trivial accessor over an ensemble that can never
  hold anything, since `add_candidates` always raises first), so
  `best_composite_score`'s boosting arm calls it and then feeds the result
  into `behave_cscore.get_cscore`, which has no ported overload for a
  `scored_combo_tree_set`/list of scored trees (`behave_cscore.py`'s own
  module docstring: that overload is "out of scope for this whole cycle")
  -- so the call fails there, naturally, the first time it actually tries
  to use the unported machinery. `best_tree`'s boosting arm calls
  `ensemble.get_weighted_tree()` directly, which always raises
  `NotImplementedError` (ensemble.py) -- per this task's own instruction,
  "The ensemble from Task 1 already raises if driven -- so the boosting
  arm should reach it rather than being stubbed out separately."
* `discard_dominated` (`metapop_parameters.discard_dominated`, default
  `False`, `metapop_params.h:211`) gates the ONE inlined fragment of
  `merge_candidates` this file carries (inside `init`, see its own
  docstring) -- raises `NotImplementedError` on the domination arm, since
  `domination.cc` is outside this whole slice's scope.
* `diversity_enabled()` (`metapop_parameters.diversity.enabled()`, off by
  default -- `pressure == 0.0`) is ported as the trivial one-line
  accessor it is; nothing in this file's own scope branches on it (the
  diversity PENALTY machinery, `set_diversity`, is not ported -- see
  above).

`score_t` is `float` (types.py's own module docstring, CARRY-FORWARD.md
4.2) and every `composite_score` this file stores or returns is checked
against C++'s own copy-vs-reference shape for that exact value (see
`_copy_sct`'s and `best_composite_score`'s own docstrings for the two
places that matters): this file introduces no NEW numeric narrowing sites
of its own (every arithmetic step that produces a `score_t` here --
`sum_bscore`, `get_score`, `set_score` -- already narrows internally,
ported elsewhere), only comparisons and copies of already-narrowed values,
so nothing here needs its own `_to_score_t` call.

Oracle: `metapopulationUTest.cxxtest` is registered, builds, and passes (2
tests) -- but both test `remove_dominated`/`get_nondominated_disjoint`
(domination.cc), neither of which this task ports (see above). Measured,
bounded probe written instead: `tools/cpp_probe/metapop_core_probe.cc`
(prefixed `metapop_` per this task's own instruction), covering the
constructor (both overloads), `init`'s default-path candidate scoring and
insertion, and `update_best_candidates`'s tie-break-by-complexity branch
(public in the header, called directly). Its own build/run recipe and a
measured, load-bearing C++ hazard it caught (a dangling `metapop_parameters`
reference when a caller omits `pa`) are documented in its own header
comment; the captured values are reproduced as literal expected values in
this module's own test file rather than a checked-in fixture (the probe's
scope -- 4 trees, 3 scenarios -- is small enough that a fixture+capture-
script pair would be pure overhead; see `hcextra_max_count_probe.cc` for
the same "probe without a fixture pipeline" precedent elsewhere in this
port). `rescore()` and `init()` itself are both PRIVATE in classic
(`metapopulation.h`, friended only to `::metapopulationUTest`), so neither
can be probed directly from outside the class; both are ported straight
from their own C++ source (merging.cc:126, metapopulation.cc:56) and
exercised with direct Python unit tests instead, relying on `sum_bscore`/
`set_score`/`get_cscore`/`get_bscore` already being oracle-verified
elsewhere (`test_behave_cscore.py`, `test_scoring_base.py`).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.ensemble import ensemble
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.moses.types import (
    composite_score,
    score_t,
    scored_combo_tree,
    scored_combo_tree_ptr_set,
    scored_combo_tree_set,
    worst_composite_score,
)

if TYPE_CHECKING:
    from pymetta.comboreduct.combo.vertex import combo_tree
    from pymetta.moses.scoring.behave_cscore import behave_cscore

__all__ = ["metapopulation"]


def _copy_sct(sct: scored_combo_tree) -> scored_combo_tree:
    """Not itself a ported C++ symbol.

    Every C++ container this class inserts a `scored_combo_tree` into
    copies it independently: `_scored_trees` (`scored_combo_tree_ptr_set`,
    a `boost::ptr_set`) via `new scored_combo_tree(cnd)` at its own insert
    call sites in merging.cc, and `_best_candidates` (`scored_combo_tree_
    set`, `std::unordered_set`) via its own by-value `insert(const
    value_type&)` ("invokes the copy constructor for insertion",
    types.h:362-367's own comment -- see `types.py`'s
    `scored_combo_tree_set` docstring). This helper reproduces that at the
    two call sites in this file that do the Python equivalent of either
    insert (inside `init`, and inside `update_best_candidates`): a fresh
    `scored_combo_tree` wrapper with its OWN `composite_score` (`copy.copy`
    -- sufficient because every `composite_score` field is a plain
    float/int/bool, the same reasoning `metapop_parameters.__init__`'s own
    docstring gives for shallow-copying a `diversity_parameters`) and its
    own `behavioral_score` list (a shallow `list(...)` copy, matching a
    real `std::vector<score_t>` copy constructor). This matters because
    `rescore()` below mutates a stored `scored_combo_tree`'s
    `composite_score` IN PLACE (`set_score`) -- exactly the aliasing shape
    CARRY-FORWARD's S9 fix round closed for `complexity_based_scorer`'s
    own returned `worst_composite_score` (`instance_scorer.py`): without
    this copy, `_scored_trees` and `_best_candidates` could end up sharing
    ONE `composite_score` object between two logically-independent slots,
    and mutating one through `rescore()` would silently corrupt the other.

    The `.tree` (combo_tree) itself is NOT deep-copied here -- a scoped
    simplification, not a claim that C++ shares it (real C++ deep-copies
    the tree too, as part of copying the enclosing `scored_combo_tree`).
    Nothing in this port's current scope ever mutates a `combo_tree` after
    it has been wrapped into a `scored_combo_tree` (tree mutation happens
    during knob-building/representation reconstruction, upstream of
    scoring), so sharing the tree object across independent
    `scored_combo_tree` copies is inert today. Flagged here, the same way
    CARRY-FORWARD 4.1 flags accepted-but-inert aliasing elsewhere, so a
    future task that ever mutates a stored tree in place knows to revisit
    this.
    """
    return scored_combo_tree(
        sct.get_tree(), copy.copy(sct.get_composite_score()), list(sct.get_bscore())
    )


class metapopulation:
    """C++: moses/moses/metapopulation/metapopulation.h:104 metapopulation

    See this module's own docstring for exactly what subset of the real
    class this covers.
    """

    def __init__(
        self,
        exemplars: combo_tree | list[combo_tree],
        sc: behave_cscore,
        pa: metapop_parameters | None = None,
        subp: subsample_deme_filter_parameters | None = None,
    ) -> None:
        """C++: moses/moses/metapopulation/metapopulation.cc:36 metapopulation::metapopulation
        and moses/moses/metapopulation/metapopulation.cc:45 metapopulation::metapopulation

        Merges both C++ constructor overloads (a `vector<combo_tree>` of
        exemplars, or a single `combo_tree`) into one method, the same
        convention `Table.__init__` uses for its own overloads
        (CARRY-FORWARD.md 1.1) -- dispatching on `isinstance(exemplars,
        list)`, since neither overload can be confused for the other
        (`combo_tree` is `tree.py`'s plain `tree` class, never itself a
        `list`).

        `_params`/`_filter_params` are C++ `const metapop_parameters&`/
        `const subsample_deme_filter_parameters&` REFERENCE members
        (metapopulation.h:532,534) bound directly to the caller's `pa`/
        `subp` argument, never copied -- ported as a plain attribute
        assignment (`self._params = pa`), Python's own default aliasing
        assignment, which is therefore an exact match rather than a
        divergence.

        MEASURED HAZARD this port cannot reproduce (there is nothing
        analogous to port): binding a C++ reference MEMBER (not a local
        reference variable) to a default-argument TEMPORARY does not get
        its lifetime extended -- only a local reference gets that -- so a
        real C++ caller that omits `pa`/`subp` (using the default
        `metapop_parameters()`/`subsample_deme_filter_parameters()`) binds
        `_params`/`_filter_params` to an object that is destroyed the
        instant the constructor call's own full expression ends, leaving a
        DANGLING reference for the rest of the `metapopulation`'s
        lifetime. This was not merely inferred: an early version of
        `tools/cpp_probe/metapop_core_probe.cc` called `metapopulation
        mp(t1, csc);` (the default `pa`) and crashed inside the very next
        statement, `mp.best_composite_score()`, with "Ensemble scoring not
        implemented" -- `_params.do_boosting`, read back from
        already-freed stack memory, happened to come back truthy even
        though boosting was never requested. That probe's own header
        comment documents the exact backtrace. Python has no equivalent
        failure mode here: `self._params = pa if pa is not None else
        metapop_parameters()` keeps whichever object it binds alive via
        ordinary reference counting for as long as `self` exists, so
        nothing can dangle -- this is not a deliberate behavioural
        difference (a `# DIVERGENCE` marker would overstate it: there is
        no C++ behaviour here to diverge FROM, only undefined behaviour to
        not reproduce), just a hazard that is real in classic and
        structurally absent in this port.

        `_best_cscore(worst_composite_score)` (metapopulation.cc:40) is a
        COPY-construction from the module-level constant -- ported via
        `copy.copy`, the same shape CARRY-FORWARD's S9 fix documents for
        `complexity_based_scorer`'s own return of this same singleton
        (`instance_scorer.py`): `composite_score.set_score`/
        `set_uniformity_penalty` mutate in place, so a bare alias here
        would let this object's very first mutation (inside
        `update_best_candidates`) also silently corrupt the shared
        `worst_composite_score` singleton every other freshly-constructed
        `metapopulation` reads from.

        `_ensemble(sc, pa.ensemble_params)` (metapopulation.cc:41) --
        Task 1's `ensemble`, constructed unconditionally (it never raises
        unless `ensemble_parameters.do_boosting` -- itself independently
        defaulted `False`, see `ensemble.py`'s own docstring -- is set on
        the params passed in).

        Fields the C++ constructor also initializes that this task does
        NOT port -- `_cached_dst`, `_merge_count`, `_visited_exemplars`,
        `_merge_mutex` -- because every method that reads them is itself
        out of this task's scope (see this module's own docstring). A
        later task adding `select_exemplar`/`resize_metapop`/diversity's
        `set_diversity` should add these alongside those methods, not
        here.
        """
        bases: list[combo_tree] = exemplars if isinstance(exemplars, list) else [exemplars]
        self._params = pa if pa is not None else metapop_parameters()
        self._filter_params = (
            subp if subp is not None else subsample_deme_filter_parameters()
        )
        self._cscorer = sc
        self._best_cscore = copy.copy(worst_composite_score)
        self._scored_trees = scored_combo_tree_ptr_set()
        self._best_candidates = scored_combo_tree_set()
        self._ensemble = ensemble(sc, self._params.ensemble_params)
        self.init(bases)

    def init(self, exemplars: list[combo_tree]) -> None:
        """C++: moses/moses/metapopulation/metapopulation.cc:56 metapopulation::init

        Scores every exemplar (`get_cscore`/`get_bscore`, copied explicitly
        -- see below), records them as the initial best candidates, then
        merges them into `_scored_trees`.

        `composite_score csc(_cscorer.get_cscore(base));` (metapopulation.cc:59)
        copy-constructs from `get_cscore`'s return -- ported via
        `copy.copy`. Real C++ `get_cscore` always returns an independent
        value (even its cache path returns a copy of the cached entry,
        behave_cscore.h:53-56), so this copy is defensive rather than
        load-bearing against anything this port's own `behave_cscore` does
        today (`behave_cscore.py`'s cache, when enabled, returns its
        cached object directly rather than a copy of it -- out of this
        file's scope to fix, but copying HERE means this class's own
        correctness never depends on that detail either way). Likewise
        `behavioral_score bs(_cscorer.get_bscore(base));`
        (metapopulation.cc:63) copy-constructs -- ported via `list(...)`.

        C++ then calls `merge_candidates(candidates)` (metapopulation.cc:71).
        `merge_candidates` itself (merging.cc) is a LATER task's scope --
        "Not this task: merge_candidates / merge_demes ... Leave clean
        seams for them" -- so it is not defined as a method on this class.
        What follows instead is exactly `merge_candidates`'s DEFAULT
        (`not _params.discard_dominated`) arm's body inlined
        (merging.cc:145-149, "insert all candidates in the
        metapopulation": `for (cnd : candidates) _scored_trees.insert(new
        scored_combo_tree(cnd));`) -- just far enough that `init` (this
        task's own scope) actually populates `_scored_trees` (this task's
        own scope, named explicitly in the brief). The domination arm
        (`discard_dominated=True`, merging.cc:150-155's `else`, calling
        `merge_nondominated`) is NOT ported -- `domination.cc` is outside
        every file this whole S10 slice touches -- so it raises instead,
        per this port's "port the gate, raise inside it" convention for
        every other P2 subsystem this slice's own plan names. A later
        task adding the real `merge_candidates`/`merge_demes` methods may
        replace this loop with a call to them.
        """
        candidates = scored_combo_tree_set()
        for base in exemplars:
            csc = copy.copy(self._cscorer.get_cscore(base))
            bs = list(self._cscorer.get_bscore(base))
            candidates.insert(scored_combo_tree(base, csc, bs))

        self.update_best_candidates(candidates)

        if self._params.discard_dominated:
            raise NotImplementedError(
                "metapopulation's merge_candidates domination arm "
                "(merging.cc's discard_dominated branch, merge_nondominated, "
                "domination.cc) is not ported; construct with "
                "metapop_parameters(discard_dominated=False) (the default)"
            )
        for cnd in candidates:
            self._scored_trees.insert(_copy_sct(cnd))

    def get_ensemble(self) -> ensemble:
        """C++: moses/moses/metapopulation/metapopulation.h:140 metapopulation::get_ensemble

        Trivial accessor, returned by reference in C++ (`const ensemble&`)
        -- ported as a direct return, matching that reference semantics.
        """
        return self._ensemble

    def best_composite_score(self) -> composite_score:
        """C++: moses/moses/metapopulation/metapopulation.cc:207 best_composite_score

        Returns BY VALUE in C++ (`composite_score best_composite_score()
        const;`, metapopulation.h:141 -- no `&`), so every real caller
        gets an independent copy of `_best_cscore`. A bare `return
        self._best_cscore` would instead hand back THIS object's own
        internal state, and `composite_score.set_score`/
        `set_uniformity_penalty` mutate in place (see this module's own
        docstring and the `__init__`/`_copy_sct` docstrings for the same
        shape) -- so the default-path arm copies explicitly.

        Gate: `not _params.do_boosting` returns `_best_cscore` outright
        (metapopulation.cc:208-209). The boosting arm
        (metapopulation.cc:213, `_cscorer.get_cscore(_ensemble.
        get_ensemble())`) is reached, not stubbed out separately, per this
        task's own instruction ("The ensemble from Task 1 already raises
        if driven -- so the boosting arm should reach it rather than being
        stubbed out separately"): `ensemble.get_ensemble()` never raises
        (it is a trivial accessor over a permanently-empty ensemble, since
        `add_candidates` always raises first -- ensemble.py), but
        `behave_cscore.get_cscore` has no ported overload taking a
        `scored_combo_tree_set`/list of scored trees at all
        (behave_cscore.py's own module docstring: "NOT ported ... out of
        scope for this whole cycle") -- so this call reaches straight into
        that unported surface and fails there naturally the moment
        anything actually drives boosting, rather than this method adding
        its own separate guard.
        """
        if not self._params.do_boosting:
            return copy.copy(self._best_cscore)
        return self._cscorer.get_cscore(self._ensemble.get_ensemble())

    def best_candidates(self) -> scored_combo_tree_set:
        """C++: moses/moses/metapopulation/metapopulation.cc:222 metapopulation::best_candidates

        Returns `const scored_combo_tree_set&` in C++ -- a genuine
        reference, not a copy (contrast `best_composite_score` above) --
        so this returns `_best_candidates` directly.
        """
        return self._best_candidates

    def best_tree(self) -> combo_tree:
        """C++: moses/moses/metapopulation/metapopulation.cc:229 metapopulation::best_tree

        `best_candidates().begin()->get_tree()` -- `best_candidates()` is
        a `scored_combo_tree_set` (`std::unordered_set`), whose iteration
        order C++ itself does not specify (types.py's own DIVERGENCE note
        for that container); `next(iter(...))` is the direct Python
        analogue of `.begin()` for a container with no ordering promise --
        which specific tied-for-best tree comes back can differ from
        C++'s own run for the same reason C++ itself makes no promise
        about it, not because this port diverges.

        Gate: `_params.do_boosting` routes to `_ensemble.get_weighted_tree()`
        directly (metapopulation.cc:230-231), which always raises
        `NotImplementedError` (ensemble.py) -- reaching the unported
        boosting machinery rather than adding a separate raise here, same
        convention as `best_composite_score` above.
        """
        if self._params.do_boosting:
            return self._ensemble.get_weighted_tree()
        return next(iter(self.best_candidates())).get_tree()

    def best_score(self) -> score_t:
        """C++: moses/moses/metapopulation/metapopulation.h:148 metapopulation::best_score"""
        return self.best_composite_score().get_score()

    def get_cscorer(self) -> behave_cscore:
        """C++: moses/moses/metapopulation/metapopulation.h:150 metapopulation::get_cscorer"""
        return self._cscorer

    def get_trees(self) -> scored_combo_tree_ptr_set:
        """C++: moses/moses/metapopulation/metapopulation.h:172 metapopulation::get_trees"""
        return self._scored_trees

    def size(self) -> int:
        """C++: moses/moses/metapopulation/metapopulation.h:180 metapopulation::size"""
        return len(self._scored_trees)

    def empty(self) -> bool:
        """C++: moses/moses/metapopulation/metapopulation.h:179 metapopulation::empty"""
        return self._scored_trees.empty()

    def clear(self) -> None:
        """C++: moses/moses/metapopulation/metapopulation.h:181 metapopulation::clear

        # DIVERGENCE(C++ moses/moses/metapopulation/metapopulation.h:181):
        # `void clear() { _scored_trees.clear(); }` mutates
        # `_scored_trees` IN PLACE via `boost::ptr_set::clear()`.
        # `scored_combo_tree_ptr_set` (types.py, S10 Task 0's own scope --
        # not this task's file to extend, see this module's docstring)
        # exposes no `clear()` of its own (only `insert`/`erase`/
        # `__contains__`/`__iter__`/`__len__`/`empty`). This rebinds
        # `_scored_trees` to a fresh, empty `scored_combo_tree_ptr_set()`
        # instead, which is observably identical to an in-place clear for
        # every accessor this class exposes (`get_trees`/`size`/`empty`/
        # iteration all read `self._scored_trees` fresh on every call) --
        # the only way this could ever differ is if some OTHER object held
        # a reference to the old `_scored_trees` container itself (not to
        # one of its elements) and expected to observe the clear through
        # that alias; nothing in this port does that.
        """
        self._scored_trees = scored_combo_tree_ptr_set()

    def diversity_enabled(self) -> bool:
        """C++: moses/moses/metapopulation/metapopulation.h:342 metapopulation::diversity_enabled"""
        return self._params.diversity.enabled()

    def update_best_candidates(self, candidates: scored_combo_tree_set) -> None:
        """C++: moses/moses/metapopulation/merging.cc:616 metapopulation::update_best_candidates

        "Candidates are kept in penalized score order, not in absolute
        score order. Thus, we need to search through the first few to
        find the true best score. Also, there may be several candidates
        with the best score." (merging.cc:626-629's own comment.)

        Ported as a direct transcription of the two nested conditions
        (merging.cc:637-647): the OUTER gate (`sc > best_sc or (sc ==
        best_sc and cpx <= best_cpx)`) decides whether `cnd` joins
        `_best_candidates` at all; the INNER gate (strict `<` instead of
        `<=`) decides whether it becomes the NEW sole best, resetting
        `_best_cscore` and clearing `_best_candidates` first. Iterating
        `candidates` (a `scored_combo_tree_set`, unordered in both
        languages) in a different concrete order than C++ does not change
        the FINAL result here: this is a streaming running-minimum-by-
        complexity-at-the-top-score algorithm, provably order-independent
        in both `_best_cscore` and `_best_candidates`' final membership
        (traced by hand for every relative ordering of a same-score,
        different-complexity pair, and measured directly against
        `tools/cpp_probe/metapop_core_probe.cc`'s own
        `update_best_candidates_direct` scenario, which calls this method
        directly since it is public in the header).

        `_best_cscore = csc;` (merging.cc:638) is `composite_score`'s
        (implicit, member-wise) assignment operator -- a genuine VALUE
        copy of `csc` (itself `cnd.get_composite_score()`, a reference
        into `cnd`) into `_best_cscore`, ending any aliasing between them
        -- ported via `copy.copy`, same reasoning as `best_composite_score`/
        `_copy_sct` above. `_best_candidates.insert(cnd)`
        (merging.cc:643) is `std::unordered_set::insert(const
        value_type&)`, which likewise copies `cnd` by value -- ported via
        `_copy_sct` (see its own docstring).
        """
        if candidates.empty():
            return

        best_sc = self._best_cscore.get_score()
        best_cpx = self._best_cscore.get_complexity()

        for cnd in candidates:
            csc = cnd.get_composite_score()
            sc = csc.get_score()
            cpx = csc.get_complexity()
            if (sc > best_sc) or (sc == best_sc and cpx <= best_cpx):
                if (sc > best_sc) or (sc == best_sc and cpx < best_cpx):
                    self._best_cscore = copy.copy(csc)
                    best_sc = self._best_cscore.get_score()
                    best_cpx = self._best_cscore.get_complexity()
                    self._best_candidates = scored_combo_tree_set()
                self._best_candidates.insert(_copy_sct(cnd))

    def rescore(self) -> None:
        """C++: moses/moses/metapopulation/merging.cc:126 metapopulation::rescore

        "Recompute the composite score for each member of the metapop.
        Used only when boosting." (merging.cc:124-125's own comment.) Only
        the `SERIAL_RESCORING` branch is ported (merging.cc:129-133) -- the
        `#else` branch (merging.cc:135-138) is an alternate, disabled-by-
        the-same-`#define` parallel implementation of the IDENTICAL loop
        body (`OMP_ALGO::for_each` over the same lambda), never compiled
        in classic either; porting both would duplicate logic with no
        behavioural difference to capture.

        `_cscorer.get_bscorer()` (merging.cc:127) is spelled
        `self._cscorer._bscorer` here: `behave_cscore.get_bscorer()` was
        deliberately not given its own method in `behave_cscore.py` ("a
        thin one-line delegation ... call `.get_bscorer()` (i.e. the
        `behave_cscore` instance's own `_bscorer` attribute) directly
        instead", that module's own docstring) -- `ensemble.py`'s own
        constructor already establishes this exact replacement for the
        same call.

        `sct.get_composite_score().set_score(new_score)` mutates each
        stored `scored_combo_tree`'s `composite_score` IN PLACE -- this is
        exactly why `_copy_sct` (this module) gives every entry in
        `_scored_trees` its OWN `composite_score` object rather than
        sharing one with `_best_candidates`/the original scoring caller.

        DO NOT FIX (verbatim classic quirk): iterating `_scored_trees` (a
        `scored_combo_tree_ptr_set`, ordered by `sct_score_greater` --
        i.e. by composite score) while mutating each element's OWN score
        in place does not re-sort the container afterwards, in either
        language -- `_sct_comparator_set`'s sorted list (types.py) is
        populated once, at insert time, and never re-bisected on a later
        in-place mutation, matching `boost::ptr_set`'s own real-world
        behaviour under exactly this same call pattern (mutating a live
        element through a reference obtained by iteration, rather than
        via `erase`+`insert`). The container's own sort order can
        therefore go stale relative to its own comparator immediately
        after `rescore()` runs; nothing in THIS task's scope reads
        `_scored_trees` assuming it stays sorted after a `rescore()` call
        (that assumption belongs to `resize_metapop`/`select_exemplar`,
        both out of scope -- see this module's own docstring), so this is
        recorded as a known, ported-as-is quirk rather than "fixed" here.
        """
        bscorer = self._cscorer._bscorer
        for sct in self._scored_trees:
            new_score = bscorer.sum_bscore(sct.get_bscore())
            sct.get_composite_score().set_score(new_score)
