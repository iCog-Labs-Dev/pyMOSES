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

* `has_been_visited`/`log_selected_exemplar` -- the SIBLING accessor and
  logging helper `select_exemplar` also declares alongside `_visited_
  exemplars` (metapopulation.h:558-561,528). `select_exemplar` ITSELF, and
  the `_visited_exemplars` field/`end()`, are ported below -- S11 Task 1's
  own scope (see `select_exemplar`'s own docstring). `has_been_visited`
  has zero callers anywhere in `local_moses.cc`/S11's own file list
  (verified by grep) and `log_selected_exemplar` is logging-only, same
  "no logging surface" convention as `ostream_metapop`/`log_best_
  candidates` below.
* `resize_metapop`/`_merge_count` -- population trimming.
* `set_diversity`/`gather_diversity_stats`/`_cached_dst`/`cached_dst` --
  the diversity penalty machinery (`diversity_enabled()` itself, the
  trivial gate accessor, IS ported: it is a one-line delegation to
  `metapop_parameters.diversity.enabled()`, metapop_params.py, and reads
  no diversity-specific state of its own).
* `remove_dominated`/`get_nondominated_disjoint`/`get_nondominated_rec`/
  `merge_nondominated`/`dominates`/`to_set`/`split` -- domination.cc, which
  was not in ANY file the S10 slice touched (deme_expander.{h,cc},
  deme_params.h, metapopulation.{h,cc}, merging.cc, metapop_params.{h,cc}
  -- the S10 plan's own file list) at S10 Task 2's time.
  `metapopulationUTest.cxxtest` (the slice's registered cxxtest oracle)
  tested exactly these two domination methods and nothing else -- see this
  module's own test file for why that made the packaged oracle
  inapplicable to Task 2's own scope, and
  `tools/cpp_probe/metapop_core_probe.cc` for the probe written to cover
  Task 2's own surface instead.

  UPDATED (P2 metapopulation task): `domination.py` (sibling module) now
  ports this whole file, and `metapopulationUTest.cxxtest`'s two tests are
  its real oracle -- see that module's own docstring. This class's own
  gates that used to raise on `discard_dominated` (`init`, below, and
  `merging.py`'s `merge_candidates`/`merge_demes`) now call the real
  methods.
* `merge_candidates`/`merge_demes`/`sort_demes`/`keep_top_unique_candidates`/
  `trim_down_deme`/`deme_to_trees`/`get_new_candidates`/subsampling
  (`ss_*`, `recompute_scores_over_whole_dataset`) -- merging.cc's own
  larger machinery, explicitly a LATER task ("Not this task: merge_candidates
  / merge_demes ... Leave clean seams for them"). `init()` below still
  needs SOMETHING to populate `_scored_trees` (this task's own scope names
  that pool explicitly) -- see `init`'s own docstring for the seam this
  leaves.
* `ostream_metapop`/`log_selected_exemplar` -- printing/logging, no
  observable effect on anything this port asserts against (the same
  "logger calls are not ported" convention every prior slice in this port
  has followed; see e.g. `hill_climbing.py`'s own module docstring).
  `log_best_candidates` (merging.cc:652) is ADDED, by S11 Task 1, on
  `metapopulation.py`'s sibling module `merging.py` (its own real C++
  home) -- see that module's own docstring for what "port the control
  flow, not the log text" means for a method whose only real purpose is
  printing.
* `_merge_mutex`/threading in general -- no file in this port anywhere
  uses `threading.Lock`; C++'s mutexes exist for multi-threaded callers
  this single-threaded port has none of (confirmed by search: `rg
  "threading|Lock\\(\\)|mutex"` over `pymetta/` turns up zero actual uses).

Gates (verified in source, not inferred -- see metapop_params.py's own
module docstring for the same table):

* `do_boosting` (`metapop_parameters.do_boosting`, default `False`,
  `metapop_params.h:170`) gates `best_composite_score`/`best_tree`'s
  boosting arm. Both reach straight into `ensemble` (`ensemble.py`) rather
  than raising here separately. `best_tree`'s boosting arm (`ensemble.
  get_weighted_tree()`) is now REAL (P2 metapopulation task: `ensemble.py`
  ports the whole of `add_adaboost`/`add_expert`/the weighted-tree
  getters). `best_composite_score`'s boosting arm still raises, for a
  DIFFERENT reason than before: it feeds `ensemble.get_ensemble()` (now a
  real, populatable `scored_combo_tree_set`) into `self._cscorer._bscorer.
  call_ensemble(...)`, and NO concrete scorer anywhere in this port
  overrides that method yet (`bscore_base.call_ensemble`'s own base body
  is itself an unconditional raise, matching classic's own `OC_ASSERT
  (false, ...)` -- `ensemble.py`'s own module docstring, prerequisite gap
  #3, has the full citation and the required follow-up this needs).
  `add_candidates`/`get_ensemble`/`get_weighted_tree`/`flat_score` are all
  real now; only the `call_ensemble`-dependent paths (`best_composite_
  score`'s boosting arm, and `add_expert`'s exact-experts weight update
  and `flat_score` against a real scorer other than a test double) remain
  blocked on that one prerequisite.
* `discard_dominated` (`metapop_parameters.discard_dominated`, default
  `False`, `metapop_params.h:211`) used to gate an inlined fragment of
  `merge_candidates` this file carried directly (inside `init`) with a
  raise on the domination arm. UPDATED (P2 metapopulation task): `init`
  now calls `self.merge_candidates(candidates)` -- a real method, as of
  `merging.py`'s own P2 update -- instead of inlining it; see `init`'s own
  docstring.
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
import math
from typing import TYPE_CHECKING

from pymetta.comboreduct.combo.iostream_combo import tree_to_stream_str
from pymetta.moses.metapopulation.metapop_params import (
    diversity_parameters,
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.moses.types import (
    _to_score_t,
    composite_score,
    score_t,
    scored_combo_tree,
    scored_combo_tree_equal,
    scored_combo_tree_hash,
    scored_combo_tree_ptr_set,
    scored_combo_tree_set,
    worst_composite_score,
)
from pymetta.util.logger import logger
from pymetta.util.rng import randGen
from pymetta.util.selection import roulette_select

if TYPE_CHECKING:
    from pymetta.comboreduct.combo.vertex import combo_tree
    from pymetta.moses.scoring.behave_cscore import behave_cscore

__all__ = ["metapopulation"]


_UNEVALUATED_SCORE: score_t = _to_score_t(-1.0e37)
# C++: moses/moses/metapopulation/metapopulation.cc:128 UNEVALUATED_SCORE
# `#define UNEVALUATED_SCORE -1.0e37`, local to `select_exemplar`. Used only
# as a sentinel `highest_score` starts at, immediately overwritten by the
# first finite candidate score `select_exemplar` sees (`probs` always holds
# at least one finite entry whenever `found_exemplar` is true) -- its exact
# narrowed value never otherwise participates in a comparison a real score
# could win, so the float32 rounding this constant itself picks up is inert.


class _visited_exemplars_counter:
    """Not itself a ported C++ symbol -- backs `_visited_exemplars`
    (`scored_tree_counter`, metapopulation.h:553-557):

        typedef std::unordered_map<scored_combo_tree, unsigned,
                                   scored_combo_tree_hash, scored_combo_tree_equal>
            scored_tree_counter;

    Python cannot use a `scored_combo_tree` as a plain `dict` key at all --
    `scored_combo_tree.__hash__` is explicitly `None` (types.py), precisely
    because its own `__eq__` compares tree AND bscore AND cscore, which
    `scored_combo_tree_hash`/`scored_combo_tree_equal` deliberately do NOT
    (types.py's own docstrings for both: tree-structure only). This mirrors
    `scored_combo_tree_set`'s own bucketed-dict implementation (types.py),
    extended to store a mutable per-entry COUNT rather than just testing
    membership.
    """

    __slots__ = ("_buckets",)

    def __init__(self) -> None:
        self._buckets: dict[int, list[list]] = {}

    def get(self, sct: scored_combo_tree) -> int:
        """C++: std::unordered_map<...>::operator[] (read use)

        `_visited_exemplars[key]` (metapopulation.cc:116,140), read as a
        plain `int` -- NON-const `operator[]`, which inserts a fresh
        zero-value entry for an absent key as a side effect, rather than
        `find()`'s read-only "not present" answer. Reproduced faithfully
        (insert-then-return-0, not merely return-0): a later port of
        `has_been_visited()` (`_visited_exemplars.find(tr) != cend()`,
        metapopulation.cc:77-79 -- itself out of THIS task's scope, see
        module docstring) tests PRESENCE, not count, so this exact
        insert-on-read side effect is what makes an exemplar that was only
        ever SCANNED (never actually selected) still count as "visited" to
        that method -- a genuine, if obscure, classic quirk from mixing a
        non-const accessor into what reads as a query. Not observable
        anywhere in S11 Task 1's own scope (`has_been_visited` has no
        caller in `local_moses.cc`/`expand_deme`), ported anyway per this
        project's "port faithfully" governing rule.
        """
        h = scored_combo_tree_hash()(sct)
        eq = scored_combo_tree_equal()
        bucket = self._buckets.setdefault(h, [])
        for entry in bucket:
            if eq(entry[0], sct):
                return entry[1]
        bucket.append([_copy_sct(sct), 0])
        return 0

    def increment(self, sct: scored_combo_tree) -> None:
        """C++: std::unordered_map<...>::operator[] then `++`

        `_visited_exemplars[*selex]++` (metapopulation.cc:117,194) --
        `operator[]` inserts a zero entry if absent (value-initializing
        `unsigned` to 0), then `++` brings a first-time key to 1. A fresh
        entry copies `sct` via `_copy_sct` (this module's own helper,
        already in scope here) rather than aliasing the live pool object,
        matching every OTHER C++ container this class inserts a
        `scored_combo_tree` into (see `_copy_sct`'s own docstring) --
        `std::unordered_map::operator[]`'s insertion path likewise
        copy-constructs the key from its argument.
        """
        h = scored_combo_tree_hash()(sct)
        eq = scored_combo_tree_equal()
        bucket = self._buckets.setdefault(h, [])
        for entry in bucket:
            if eq(entry[0], sct):
                entry[1] += 1
                return
        bucket.append([_copy_sct(sct), 1])


class diversity_stats:
    """C++: moses/moses/metapopulation/metapopulation.h:319 metapopulation::diversity_stats

    "Structure holding stats about diversity" (the header's own comment).
    All five fields are plain C++ `double` -- NOT `score_t` -- so none of
    them are narrowed via `_to_score_t` anywhere they are set (P2
    metapopulation task, `diversity.py`): CARRY-FORWARD's own rule that
    `score_t`/`count_t`/`contin_t` are three distinct types applies here
    as "this is a fourth, untyped-in-this-port `double`, narrow nothing".
    """

    __slots__ = ("count", "mean", "std", "min", "max")

    def __init__(
        self,
        count: float = 0.0,
        mean: float = 0.0,
        std: float = 0.0,
        min: float = 0.0,  # noqa: A002 (shadows builtin `min`, matching the C++ field name)
        max: float = 0.0,  # noqa: A002 (shadows builtin `max`, matching the C++ field name)
    ) -> None:
        self.count = count
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max


class cached_dst:
    """C++: moses/moses/metapopulation/metapopulation.h:355 metapopulation::cached_dst

    "Cache for bscore distance between (for uniformity penalty) ... We
    don't use {lru,prr}_cache because 1) we don't need a limit on the
    cache. 2) we need to remove the pairs containing deleted pointers"
    (the header's own comment). Declared here (a nested struct in
    metapopulation.h, this file's own C++ home, the same reasoning
    `_visited_exemplars_counter` above already follows for a
    header-declared helper type) with ONLY its constructor -- `__call__`/
    `erase_ptr_seq`/`gather_stats` are `cached_dst::`-qualified DEFINITIONS
    in diversity.cc (P2 metapopulation task), attached here via `setattr`
    by `diversity.py`, mirroring `metapopulation` itself being split
    across `metapopulation.py`/`merging.py`/`domination.py`/`diversity.py`
    the same way. `diversity.py` must be imported for this class to have
    its full method set -- see that module's own docstring for the same
    import-chain guarantee `merging.py`/`domination.py` already document.

    `ENABLE_DST_CACHE` is commented out at its own definition site
    (`// #define ENABLE_DST_CACHE`, metapopulation.h:53), so in every
    build this port targets, `hits`/`misses`/`cache` are declared but
    never actually read from or written to by `__call__` (see its own
    docstring in diversity.py) -- they are ported anyway, as inert
    structural fields, for the same "port the whole struct, not just the
    reachable slice" reasoning `merge_candidates`/`get_new_candidates`
    already apply to fields with a narrow live surface.
    """

    def __init__(self, dparams: diversity_parameters) -> None:
        """C++: moses/moses/metapopulation/metapopulation.h:359 cached_dst::cached_dst

        (`metapopulation::cached_dst::cached_dst` -- class qualifier
        dropped only to fit the line-length limit.)

        `_dparams(dparams)` is a `const diversity_parameters&` REFERENCE
        member bound to whatever `diversity_parameters` object the caller
        passes (`metapopulation`'s own constructor passes `pa.diversity`,
        C++, or `self._params.diversity`, this port -- see `metapopulation.
        __init__`'s own docstring for why those are the SAME object here,
        not merely equal ones, so this reference-vs-value question is
        moot in practice). Ported as a bare attribute assignment, Python's
        own default aliasing -- the same "exact match, not a divergence"
        reasoning `metapopulation.__init__`'s own docstring gives for
        `_params`/`_filter_params`.

        `misses(0), hits(0)` -- ported as plain `int` attributes (`std::
        atomic<unsigned>` in C++, for thread-safety this whole port has no
        need of -- module-wide "no threading" convention).

        `cache` (`std::unordered_map<ptr_pair, dp_t, ...>`, no member
        initializer -- value-initializes to empty) -- ported as an empty
        `dict`.
        """
        self._dparams = dparams
        self.misses = 0
        self.hits = 0
        self.cache: dict[object, float] = {}


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
    copied = scored_combo_tree(
        sct.get_tree(), copy.copy(sct.get_composite_score()), list(sct.get_bscore())
    )
    # `scored_combo_tree`'s constructor always starts `_weight(1.0)`
    # (types.h:266) -- it takes no weight parameter in either language --
    # but the C++ COPY constructor this helper stands in for
    # (`new scored_combo_tree(cnd)`, merging.cc:167; `new scored_combo_tree
    # (*cnd)`, domination.cc:304) is the compiler-generated memberwise one,
    # which copies `_weight` along with everything else. Carried across
    # explicitly, so this helper stays a faithful copy now that
    # `scored_combo_tree` actually has the field (types.py). Inert today
    # -- nothing in this port calls `set_weight` yet -- and deliberately
    # written anyway: the moment `ensemble.py` stops keeping its weights in
    # a side dict, a copy that silently reset every member to 1.0 would be
    # a real, hard-to-find boosting bug.
    #
    # NOT applicable to the OTHER `scored_combo_tree(...)` call sites in
    # this package (`metapopulation.init`, and `merging.merge_demes`'s
    # deme-to-trees and bscore-recompute blocks): each of those is a
    # fresh CONSTRUCTION in C++ too (e.g. merging.cc:279-280 builds
    # `scored_combo_tree sct(cand.get_tree(), cand.get_demeID(),
    # cand.get_composite_score(), bs)`), so classic resets the weight to
    # 1.0 at those points as well and this port matches by doing nothing.
    copied.set_weight(sct.get_weight())
    return copied


# P2 metapopulation task (ensemble): `ensemble.py` now imports `_copy_sct`
# from THIS module (real `add_adaboost`/`add_expert` need it to copy a
# candidate into `_scored_trees`, the same "every container a C++ insert
# copies into needs `_copy_sct`" rule this module's own docstring states).
# This import MUST come after `_copy_sct` is defined above (not with the
# other imports at the top of this file): `ensemble.py` does `from
# pymetta.moses.metapopulation.metapopulation import _copy_sct` at ITS OWN
# top, and THIS module is still mid-execution the first time Python loads
# `ensemble.py` -- moving this import back to the top reproduces the exact
# circular-import failure the analogous `subsample`/`_sort` import in
# `merging.py` already documents (verified live there the same way this
# reordering was verified here: moving it earlier and observing `ImportError:
# cannot import name '_copy_sct' from partially initialized module`, then
# reverting).
from pymetta.moses.metapopulation.ensemble import ensemble  # noqa: E402


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
        NOT port -- `_merge_count`, `_merge_mutex` -- because every
        method that reads them is itself out of this task's scope (see
        this module's own docstring); `resize_metapop` (`merging.py`)
        avoids needing `_merge_count` pre-initialized, see its own
        docstring.

        UPDATED (P2 metapopulation task): `_cached_dst(pa.diversity)`
        (metapopulation.cc:39) IS now ported, as `cached_dst(self._params.
        diversity)` -- `set_diversity`/`gather_diversity_stats`
        (`diversity.py`) need it. `pa.diversity` vs `self._params.
        diversity`: the same object either way, since `self._params`
        already aliases `pa` (this docstring's own paragraph above), so
        this is not a new aliasing decision, just the same one applied to
        a second field.

        `_visited_exemplars` (`scored_tree_counter _visited_exemplars;`,
        metapopulation.h:557, default-constructed to an empty map) IS
        ported here, by S11 Task 1 -- its only reader/writer,
        `select_exemplar`, is this same task's own scope (see that
        method's own docstring for the container this initializes).
        """
        bases: list[combo_tree] = exemplars if isinstance(exemplars, list) else [exemplars]
        self._params = pa if pa is not None else metapop_parameters()
        self._filter_params = subp if subp is not None else subsample_deme_filter_parameters()
        self._cscorer = sc
        self._best_cscore = copy.copy(worst_composite_score)
        self._scored_trees = scored_combo_tree_ptr_set()
        self._best_candidates = scored_combo_tree_set()
        self._ensemble = ensemble(sc, self._params.ensemble_params)
        self._visited_exemplars = _visited_exemplars_counter()
        self._cached_dst = cached_dst(self._params.diversity)
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
        `merge_candidates` (merging.cc) was a LATER task's scope at S10
        Task 2's time -- "Not this task: merge_candidates / merge_demes
        ... Leave clean seams for them" -- so this method used to inline
        `merge_candidates`'s DEFAULT arm's body directly (merging.cc:
        145-149) and raise `NotImplementedError` on its domination arm.

        UPDATED (P2 metapopulation task, now that `merge_candidates` is a
        real method -- `merging.py`): calls `self.merge_candidates(
        candidates)` directly, exactly matching real C++'s one-copy shape
        instead of the port's own former duplicate. This is not a new
        behaviour -- `merge_candidates`'s own default arm is byte-for-byte
        the loop this method used to inline, and its domination arm is
        reached only when `_params.discard_dominated` is `True`, exactly
        the same condition this method used to raise under -- so this
        replaces a raise with the SAME operation done for real
        (`merge_nondominated`, `domination.py`), and the default-OFF path
        (`discard_dominated=False`, unchanged) is unaffected.
        """
        candidates = scored_combo_tree_set()
        for base in exemplars:
            csc = copy.copy(self._cscorer.get_cscore(base))
            bs = list(self._cscorer.get_bscore(base))
            candidates.insert(scored_combo_tree(base, csc, bs))

        self.update_best_candidates(candidates)
        self.merge_candidates(candidates)

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
        scope for this whole cycle").

        CORRECTED (S10 branch review, Minor): this method used to call
        `self._cscorer.get_cscore(...)` directly with that list, on the
        theory that "reaches straight into that unported surface and
        fails there naturally" was good enough -- MEASURED WRONG:
        `get_cscore(tr: combo_tree)` (behave_cscore.py) assumes its
        argument IS a `combo_tree` and reads `.data` off it, so a `list`
        argument fails with an unrelated `AttributeError: 'list' object
        has no attribute 'data'`, not the specific, already-ported
        `NotImplementedError` `bscore_base.call_ensemble`
        (scoring_base.py:124, `moses/moses/scoring/scoring_base.h:66`)
        exists for EXACTLY this case ("Ensemble scoring not implemented
        for bscorer %s"). Real C++'s own `get_cscore(scored_combo_tree_
        set)` overload would itself call `_bscorer(ensemble)` (the
        `operator()` `call_ensemble` ports) as part of its own body before
        ever reaching anything an `AttributeError`-shaped bug could stand
        in for -- so this now calls `call_ensemble` directly on the
        underlying bscorer, reaching the SAME raise real C++ reaches (no
        bscorer this port ships overrides it), one call closer to the
        real control flow instead of an accidental crash one step removed
        from it. `call_ensemble` always raises before returning (no
        subclass overrides it -- see its own docstring), so its declared
        `-> behavioral_score` return type is never actually exercised
        here; nothing in this method's own `-> composite_score` signature
        is violated by a call that never returns.
        """
        if not self._params.do_boosting:
            return copy.copy(self._best_cscore)
        return self._cscorer._bscorer.call_ensemble(self._ensemble.get_ensemble())

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
        directly (metapopulation.cc:230-231) -- now REAL (P2 metapopulation
        task, `ensemble.py`), unlike `best_composite_score` above (which
        still raises: its own do_boosting arm reaches `call_ensemble`, a
        DIFFERENT, still-unported surface -- see that method's own
        docstring).
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

    def get_cached_dst(self) -> cached_dst:
        """C++: moses/moses/metapopulation/metapopulation.h:387 metapopulation::get_cached_dst

        Trivial accessor, returned by reference in C++ (`const cached_dst&`)
        -- ported as a direct return, matching that reference semantics
        (same shape as `get_ensemble`/`get_trees` above).
        """
        return self._cached_dst

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

        CORRECTED (S10 branch review, Minor): `_best_candidates.clear()`
        (merging.cc:642) mutates the member IN PLACE in C++, so any
        earlier `best_candidates()` caller (a genuine `const&` there,
        `best_candidates`'s own docstring) sees the clear too. This method
        used to REBIND `self._best_candidates = scored_combo_tree_set()`
        instead of clearing it -- a different Python object, silently
        stranding any prior `best_candidates()` caller on the OLD, now
        stale set, which never sees the fresh inserts below it. Fixed to
        call `.clear()` (types.py, added for exactly this site), matching
        C++'s own in-place mutation.
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
                    self._best_candidates.clear()
                    # C++: merging.cc:643-644, INSIDE the inner gate and
                    # after the clear -- so it fires once per genuinely new
                    # best, not once per accepted candidate. The `\n\t`
                    # makes this a two-physical-line record.
                    logger().debug(
                        f"New best score: {self._best_cscore}"
                        f"\n\tfor tree: {tree_to_stream_str(cnd.get_tree())}"
                    )
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
        `_scored_trees` assuming it stays FRESHLY sorted after a
        `rescore()` call (that assumption belongs to `resize_metapop`,
        out of scope -- see this module's own docstring; `select_exemplar`
        below makes no such assumption either -- it iterates whatever
        order the container currently holds, exactly as C++'s own
        `for (const scored_combo_tree &bsct : *this)` does, with no
        re-sort of its own), so this is recorded as a known, ported-as-is
        quirk rather than "fixed" here.
        """
        bscorer = self._cscorer._bscorer
        for sct in self._scored_trees:
            new_score = bscorer.sum_bscore(sct.get_bscore())
            sct.get_composite_score().set_score(new_score)

    def end(self) -> scored_combo_tree | None:
        """C++: moses/moses/metapopulation/metapopulation.h:176 metapopulation::end

            scored_combo_tree_ptr_set::const_iterator end() const {
                return _scored_trees.end();
            }

        A one-line delegation to `_scored_trees.end()` -- ported because
        `expand_deme` (S11 Task 1, `local_moses.py`) compares `select_
        exemplar()`'s return against it (`local_moses.cc:61`,
        `exemplar == mp.end()`) -- the ONLY caller of `metapopulation::end`
        anywhere in `local_moses.cc`/`moses_main.cc`. `begin()`
        (metapopulation.h:173-175, the same one-line-delegation shape) has
        ZERO call sites in that same file set (verified by grep) and is
        therefore NOT ported alongside it -- adding it would be scope
        creep with nothing in S11 Task 1 to exercise it.

        # DIVERGENCE(C++ moses/moses/metapopulation/metapopulation.h:176):
        # a real `scored_combo_tree_ptr_set::const_iterator` is a distinct
        # value from every DEREFERENCEABLE iterator, comparable to one via
        # `operator==` without ever being confused for a valid element.
        # This port has no iterator type over `_sct_comparator_set`
        # (types.py) at all -- `select_exemplar` below returns the
        # DEREFERENCED `scored_combo_tree` itself (or `None`) rather than
        # an iterator wrapping it, the same "Python has no pointer/iterator
        # ownership, use the plain object" simplification `_sct_comparator_
        # set`'s own docstring already makes for `boost::ptr_set`. `None`
        # is therefore this port's `end()` sentinel: it can never compare
        # equal to a real `scored_combo_tree` (`scored_combo_tree.__eq__`
        # returns `NotImplemented` for a non-`scored_combo_tree` operand,
        # types.py, which Python resolves to `False` here), exactly the
        # property a real `end()` iterator has against every real element.
        """
        return None

    def log_selected_exemplar(self, exemplar: scored_combo_tree | None, pos: int) -> None:
        """C++: moses/moses/metapopulation/metapopulation.cc:81 log_selected_exemplar

        The real symbol is `metapopulation::log_selected_exemplar`; the
        qualified form does not fit the 100-column limit on one line.

        Emits what `select_exemplar` chose. Note the negated early-out
        (`if (not logger().is_debug_enabled()) return;`, metapopulation.cc:83)
        -- ported as the same guard rather than as four separate ones, since
        C++ leaves `std::distance` and the `_visited_exemplars` lookup below
        it unevaluated when DEBUG is off.

        The final record is empty by construction: metapopulation.cc:100-101
        declares a `std::stringstream ss;`, never writes to it, and logs
        `ss.str()`. So classic really does emit a bare `[DEBUG] ` line here,
        and reproducing the blank record is reproducing the code.

        `pos` replaces C++'s `std::distance(_scored_trees.cbegin(),
        exemplar_it) + 1`: `select_exemplar` returns a dereferenced
        candidate rather than an iterator in this port (see its docstring),
        so the index is passed in from the one place that still has it.
        `exemplar is None` is this port's `end()`, so it selects the
        "No exemplar found" arm.

        # DIVERGENCE(C++ moses/moses/metapopulation/metapopulation.cc:93): the
        # "Selected the Nth exemplar, from deme D, for the Nth time(s)"
        # record is NOT emitted. It streams `xmplr.get_demeID()`, and this
        # port's `scored_combo_tree` carries no `demeID_t` at all -- the
        # same gap that gates `--output-deme-id` (problem_params.py:1116-1126
        # and moses_main.py:127-128). Emitting it with a placeholder deme
        # would be inventing a value; the other three records are emitted,
        # so the only loss is that one line.
        """
        if not logger().is_debug_enabled():
            return

        if exemplar is None:
            logger().debug("No exemplar found")
        else:
            logger().debug("Exemplar tree : " + tree_to_stream_str(exemplar.get_tree()))
            logger().debug(f"With composite score : {exemplar.get_composite_score()}")
            logger().debug("")

    def select_exemplar(self) -> scored_combo_tree | None:
        """C++: moses/moses/metapopulation/metapopulation.cc:105 metapopulation::select_exemplar

        "Select the exemplar from the population, using the diversity
        penalty and the deme it spawns" -- more precisely, per
        metapopulation.h:154-169's own doc comment (condensed): "select
        the exemplar to build the neighborhood for the next deme ...
        computed according to a Boltzmann distribution: the fitter the
        candidate, the more likely it is to be selected ... If an exemplar
        already has a deme, and its parameters (revisit) don't allow it to
        be selected more than once, it is discarded from the search."

        Returns the DEREFERENCED `scored_combo_tree` chosen (or `None`,
        this port's `end()` sentinel -- see that method's own docstring)
        rather than a C++ iterator, the same simplification used
        throughout this port wherever a caller only ever dereferences a
        returned iterator or compares it against `end()` (`expand_deme`,
        `local_moses.py`, does exactly both: `exemplar->get_tree()` and
        `exemplar == mp.end()`).

        `OC_ASSERT(!empty(), ...)` (metapopulation.cc:106) -- ported as an
        explicit `AssertionError`, the same convention every other bare
        `OC_ASSERT` in this port's scope uses (e.g. `deme_expander.py`'s
        three `create_demes` guards).

        ## The `size() == 1` shortcut (metapopulation.cc:110-122)

        "Shortcut for special case, as sometimes, the very first time
        though, the score is invalid" (the C++ comment, verbatim). With
        exactly one candidate in the pool, no roulette selection is
        needed: either that one candidate is still revisitable (per
        `_params.revisit`, see below) and gets selected (incrementing its
        visit count), or it has been visited enough and `end()` is
        returned.

        ## The general case (metapopulation.cc:124-196)

        For each candidate, in `_scored_trees`'s own `sct_score_greater`
        order (best-score first -- `ordered = list(self._scored_trees)`,
        the direct Python analogue of iterating `*this` via `begin()`/
        `end()`): if its penalized score is finite AND it has not been
        visited too often yet (the SAME revisit test the shortcut uses),
        its score joins `probs`; otherwise a NaN placeholder joins `probs`
        instead (metapopulation.cc:135-149) -- `probs` and `ordered` stay
        INDEX-ALIGNED throughout, which is what lets `roulette_select`'s
        returned index double as an index into `ordered` too.

        The REVISIT test itself, both here and in the shortcut above:
        `_params.revisit < 0` means "unlimited revisits" (metapop_params.h:
        186-187's own comment, "If negative then the number of revisit is
        infinite"); otherwise a candidate may be selected as long as it has
        been visited STRICTLY FEWER than `_params.revisit + 1` times so far
        -- `revisit=0` (the default) therefore means "visited at most
        once", matching metapop_params.h:185-186's comment exactly ("If 0
        then an exemplar can only be visited once").

        `_visited_exemplars[bsct]`/`[*selex]` (metapopulation.cc:140,116,
        194) are all NON-const `operator[]` READS or increments -- ported
        via `_visited_exemplars_counter.get`/`.increment` respectively
        (this module's own helper; see its docstring for the insert-on-
        read side effect it faithfully reproduces).

        If NO candidate qualifies (`found_exemplar` stays `False` --
        metapopulation.cc:151-154), `end()` is returned immediately: no
        roulette selection is attempted, matching C++ exactly (it never
        even computes `probs`' probabilities in this case).

        Otherwise, `probs` (still holding raw penalized scores and NaN
        placeholders) is converted into a genuine, non-normalized Boltzmann
        distribution (metapopulation.cc:156-176): "Empirical work on
        4-parity suggests that a temperature of 3 or 4 works best" (the
        C++ comment). Every arithmetic step here is `score_t`-typed in
        C++ (`inv_temp`, `p`, `sum` are all declared `score_t`/`float`) and
        is narrowed via `_to_score_t` at EVERY individual step below, not
        merely once at the end -- `sum` in particular is a running `score_t`
        ACCUMULATOR in C++ (`sum += p` is a float32 in-place update,
        re-read by the NEXT iteration), so narrowing only the final total
        would let double-precision rounding leak into intermediate state a
        real `score_t sum` could never actually hold (the same reasoning
        `merging.py`'s diversity-distance helpers already apply to their own
        running `score_t` accumulators, `metapop_params.py`).

        `OC_ASSERT(sum > 0.0f, ...)` (metapopulation.cc:181) is ported as
        an explicit `AssertionError` -- unreachable in practice (whenever
        `found_exemplar` is `True`, at least one included candidate has
        `p - highest_score == 0`, i.e. `exp(0) == 1 > 0`, so `sum` can
        never end up `<= 0`), kept anyway as a faithful transcription of a
        real invariant check C++ itself keeps.

        `roulette_select(probs.begin(), probs.end(), sum, randGen())`
        (metapopulation.cc:186-187) -- ported via `pymetta.util.selection.
        roulette_select`, this task's other new module; see its own
        docstring for why it returns an index directly rather than an
        iterator `std::distance` would still need unwrapping.
        `std::next(_scored_trees.begin(), fwd)` (metapopulation.cc:191) is
        then simply `ordered[fwd]`, since `ordered` and `probs` share the
        same index space by construction above.

        `log_selected_exemplar` (metapopulation.cc:81-93,121,152,196) is
        NOT ported -- logging-only, no observable effect (module
        docstring's "no logging surface" convention); the POSITION/visit-
        count values it would print have no effect on control flow.
        """
        if self.empty():
            raise AssertionError("Empty metapopulation in select_exemplar().")

        logger().debug("Select exemplar")

        ordered: list[scored_combo_tree] = list(self._scored_trees)

        if len(ordered) == 1:
            selex: scored_combo_tree | None = ordered[0]
            if self._params.revisit < 0 or (
                self._params.revisit + 1 > self._visited_exemplars.get(selex)
            ):
                self._visited_exemplars.increment(selex)
            else:
                selex = None
            # C++: metapopulation.cc:120 -- the shortcut logs before it
            # returns, exactly as the general case does. `cbegin()+1` is 1
            # for the single candidate, and 0 for the `end()` arm (unused).
            self.log_selected_exemplar(selex, 1)
            return selex

        probs: list[score_t] = []
        found_exemplar = False
        highest_score: score_t = _UNEVALUATED_SCORE

        for bsct in ordered:
            sc = bsct.get_penalized_score()
            if math.isfinite(sc) and (
                self._params.revisit < 0
                or self._params.revisit + 1 > self._visited_exemplars.get(bsct)
            ):
                probs.append(sc)
                found_exemplar = True
                if highest_score < sc:
                    highest_score = sc
            else:
                probs.append(math.nan)

        if not found_exemplar:
            # C++: metapopulation.cc:152 `log_selected_exemplar(
            # _scored_trees.cend());` -- the "No exemplar found" arm.
            self.log_selected_exemplar(None, 0)
            return None

        inv_temp: score_t = _to_score_t(100.0 / self._params.complexity_temperature)
        total: score_t = _to_score_t(0.0)
        for i, p in enumerate(probs):
            if math.isfinite(p):
                val: score_t = _to_score_t(
                    math.exp(_to_score_t(_to_score_t(p - highest_score) * inv_temp))
                )
            else:
                val = _to_score_t(0.0)
            probs[i] = val
            total = _to_score_t(total + val)

        if not (total > 0.0):
            raise AssertionError("There is an internal bug, please fix it")

        fwd = roulette_select(probs, total, randGen())
        selex = ordered[fwd]

        self._visited_exemplars.increment(selex)

        # C++: metapopulation.cc:194-195. `fwd` is `std::distance(probs.
        # begin(), roulette_select(...))`, and `probs`/`ordered` are
        # index-aligned, so `fwd + 1` is C++'s own `pos`.
        self.log_selected_exemplar(selex, fwd + 1)

        return selex
