"""C++: moses/moses/metapopulation/merging.cc

Ports `metapopulation::merge_candidates` and `metapopulation::merge_demes`
(S10 Task 3's own brief), plus every PRIVATE helper `merge_demes` itself
calls that is not already ported: `trim_down_deme`, `deme_to_trees`,
`get_new_candidates`, `sort_demes`, `keep_top_unique_candidates`, and
`resize_metapop` (all defined in merging.cc -- the file this task owns).
`rescore()` and `update_best_candidates()` are ALSO declared in merging.cc
(lines 126 and 616) but were already ported onto `metapopulation` by S10
Task 2 (`metapopulation.py`, which cites merging.cc for both) -- not
reported here a second time; this module calls them via `self.` like any
other bound method.

`recompute_scores_over_whole_dataset`/`ss_filter` are declared in
metapopulation.h but DEFINED in subsample.cc -- a file outside this whole
S10 slice (deme_expander.{h,cc}, deme_params.h, metapopulation.{h,cc},
merging.cc, metapop_params.{h,cc}), the same way `remove_dominated`/
`get_nondominated_disjoint` live in domination.cc, also outside the slice.
`merge_demes` (this task's own scope) calls both unconditionally, so they
are ported here too, AS GATES: the disabled arm (`n_subsample_demes <= 1`,
the default) reproduces the real no-op/all-true behaviour exactly (that
part of subsample.cc's own source is trivial and fully visible even though
the file itself is out of scope); the enabled arm raises, since the real
work lives in a file this task does not own.

## S11 Task 1 addition: `log_best_candidates`

`metapopulation::log_best_candidates` (merging.cc:652, declared
metapopulation.h:518) is added here by S11 Task 1, alongside `local_moses.py`
-- `expand_deme`'s own C++ body calls it (`local_moses.cc:87`), though S11's
own port of `expand_deme` does NOT call it back (see `local_moses.py`'s
module docstring for why: it sits inside the SAME `logger().is_info_enabled()`
block as two `logger().info()` calls, and this port's whole `expand_deme`
transcription omits that block entirely -- no logging surface anywhere in
this port, module-wide convention). It is ported here anyway as a directly
callable, independently testable method: see its own docstring for what
"port the control flow, not the log text" means for a method whose ONLY
reason to exist in C++ is to print.

## Composition: module-level functions the class binds (not a mixin)

Real C++ compiles `merging.cc` into a SECOND translation unit that adds
more methods to the ONE class `metapopulation` (declared once, in
metapopulation.h) -- there is no inheritance relationship between
`metapopulation.cc` and `merging.cc`; both simply define members of the
same class. This module reproduces that exact shape: every function below
is written as a free function taking `self: metapopulation` as its first
parameter (with a docstring/breadcrumb exactly like any other ported
method), and the block at the bottom of this file binds each one onto the
`metapopulation` class via `setattr` -- e.g. `metapopulation.merge_demes =
merge_demes`. After this module is imported once, every `metapopulation`
instance has these methods exactly as if they had been defined in the
class body.

This was chosen over a mixin class (`class _MergingMixin: def
merge_demes(self, ...): ...`) combined via inheritance for three reasons:

1. The task boundary forbids editing `metapopulation.py` except for a
   one-line hook if the seam genuinely requires it (see `_merge_count`
   below for the one field this actually needed, resolved WITHOUT such a
   hook). Declaring `class metapopulation(_MergingMixin, ...)` requires
   editing the class statement itself, in `metapopulation.py`, exactly the
   file this task must not touch. Dynamically rewriting `__bases__` from
   here (`metapopulation.__bases__ += (_MergingMixin,)`) is the same idea
   with worse ergonomics: it changes the MRO in place, is unusual enough
   to surprise a future reader, and buys nothing a flat `setattr` does
   not.
2. A plain function is exactly what every OTHER ported method in this
   whole project already looks like: a `def` with a breadcrumb-first
   docstring, at module scope or in the class body it belongs to. Nesting
   these inside a throwaway mixin class would exist purely to satisfy the
   composition mechanism, not to express anything about the port.
3. It mirrors the C++ shape more directly than inheritance would: a
   second translation unit contributing more methods to one class is not
   "merging IS-A specialization of metapopulation" (what a mixin models),
   it is "these methods ALSO belong to metapopulation" -- precisely what
   `setattr(metapopulation, name, func)` says and inheritance does not.

The one caveat this approach has, and inheritance would not: a caller
needs `pymetta.moses.metapopulation.merging` to have been imported at
least once (directly, or transitively through anything that imports it --
every test in `test_merging.py` does) before a bare `metapopulation`
instance has `merge_demes`/`merge_candidates`/etc. Nothing in this port's
current scope constructs a `metapopulation` without also wanting these
methods, so this is recorded here rather than worked around.

## `_min_pool_size`/`_merge_count`/`useful_score_range` -- not merging.cc's
## own symbols, needed anyway

`_min_pool_size` (`static const unsigned _min_pool_size = 20;`,
metapopulation.h:540) and `useful_score_range()` (metapopulation.h:277, an
inline one-liner) are both declared in the HEADER, not merging.cc itself --
but `trim_down_deme`/`resize_metapop` (both merging.cc, both this task's
scope) are their only real callers, and Task 2 (`metapopulation.py`) had no
need of either. Ported here, bound onto the class the same way as
everything else: `_min_pool_size` as a plain class attribute (it is
`static` in C++ -- shared, not per-instance -- so a class attribute is the
more faithful match, not merely a convenient one), `useful_score_range` as
a bound method.

`_merge_count` (`size_t`, metapopulation.h:542) IS genuine per-instance
state, initialized in the constructor's own member-initializer list
(`metapopulation.cc:40,49`, `_merge_count(0)`) -- normally a one-line-hook
candidate. It is avoided here instead: `resize_metapop` (this module's own
scope, its only reader/writer in the whole class) initializes it lazily on
first use (`getattr(self, "_merge_count", 0)`), which is observably
identical to "initialized to 0 at construction time" for a field nothing
else ever reads before `resize_metapop` first runs. See `resize_metapop`'s
own docstring.

## Gates ported (all default OFF; see metapop_params.py's own table)

* `_params.discard_dominated` -- `merge_candidates`'s domination arm
  (merging.cc:168-175) and `merge_demes`'s OWN second, structurally
  redundant `if (_params.discard_dominated)` block (merging.cc:298-328)
  both raise; the latter is UNREACHABLE in this port once the union gate
  below has already raised for the same flag (see `merge_demes`'s own
  docstring) and is therefore not written out as literal dead code.
* `diversity_enabled() or _params.do_boosting or _params.discard_dominated
  or _params.keep_bscore` -- `merge_demes`'s behavioural-score recompute
  block (merging.cc:264-296) raises as a whole; real C++ needs this block
  only to feed diversity/boosting/domination/`keep_bscore`, none of which
  this port drives (see the block's own citation below for why porting the
  block itself was judged out of scope even though it does not depend on
  any OTHER unported file).
* `_params.do_boosting` -- `merge_demes`'s `_ensemble.add_candidates(...)`
  call (merging.cc:344) is reached, not separately guarded: `ensemble.
  add_candidates` (ensemble.py, S10 Task 1) always raises `NotImplementedError`
  already, the same "reach the unported surface" convention
  metapopulation.py's own `best_composite_score`/`best_tree` use for this
  identical flag.
* `diversity_enabled()` -- `merge_demes`'s `set_diversity()` call
  (merging.cc:349-350, diversity.cc, not ported anywhere in this project)
  raises.
* `_params.merge_callback is not None` -- `merge_demes`'s callback hook
  (merging.cc:338-339) raises; always `None` by default and never set by
  anything in this port (metapop_params.py's own docstring).
* `_filter_params.n_subsample_demes > 1` -- gates
  `recompute_scores_over_whole_dataset`/`ss_filter` (see module docstring
  above); the disabled arm's real behaviour (a no-op / an all-`True`
  vector) is ported exactly, not merely stubbed.

## Aliasing discipline

Every `composite_score`/`scored_combo_tree` this module stores into a
container is checked against C++'s own copy-vs-reference shape at that
exact call site, the same discipline Task 2 established:

* `deme_to_trees`: `inst_csc = inst.second` is a reference straight into
  the deme's own `scored_instance` (later trimmed/discarded/rescored).
  `scored_combo_tree.__init__` does NOT copy its `cs` argument (a bare
  `self.cscore = cs` in types.py) -- unlike C++, where `scored_combo_tree`'s
  own constructor member-initializes `_cscore` from its parameter BY
  VALUE. `copy.copy(inst_csc)` is passed explicitly to reproduce that.
* `merge_candidates`/`get_new_candidates`: both use Task 2's own
  `_copy_sct` helper (imported, not reimplemented) at every insert into a
  container the corresponding C++ container copies into
  (`scored_combo_tree_ptr_set`/`scored_combo_tree_set`, per types.py's own
  "invokes the copy constructor for insertion" citation).
* `resize_metapop`: never constructs or stores a NEW `scored_combo_tree`
  or `composite_score` at all -- it only REMOVES existing pool members --
  so there is nothing to copy there; see its own docstring for the one
  aliasing-ADJACENT design choice it does make (removing via the public
  `erase(sct)` API rather than reaching into another module's private
  sorted-list backing store).

## `score_t` narrowing sites this file adds

`useful_score_range()`'s own return (`metapopulation.h:277`, a double
computation narrowed to `score_t` on return) and the two subtractions that
consume it (`trim_down_deme`'s `bot_sc`, `resize_metapop`'s `worst_score`,
both `score_t - score_t` -- a float32 operation in C++, not a
double-precision one) are narrowed here via `_to_score_t`, following
CARRY-FORWARD 4.2's rule for every score_t-typed arithmetic step. None of
these three sites are in types.py's own 14-site list -- `useful_score_range`
and its two callers were all out of every prior task's scope.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.metapopulation import _copy_sct, metapopulation
from pymetta.moses.moses.optimization.hill_climbing import _greater, _partial_sort
from pymetta.moses.moses.types import (
    _combo_tree_structural_eq,
    _is_approx_eq,
    _to_score_t,
    score_t,
    scored_combo_tree,
    scored_combo_tree_set,
    very_worst_score,
)
from pymetta.util.rng import randGen

if TYPE_CHECKING:
    from pymetta.moses.moses.optimization.optimization import deme_t
    from pymetta.moses.representation.representation import representation

__all__: list[str] = []
# Nothing new is exported: every symbol below is bound onto `metapopulation`
# itself (see module docstring's Composition section) rather than being a
# name callers import directly.

_MIN_POOL_SIZE = 20
# C++: moses/moses/metapopulation/metapopulation.h:540
# `static const unsigned _min_pool_size = 20;`
#
# # DIVERGENCE(C++ moses/moses/metapopulation/merging.cc:51): the doc
# # comment directly above `trim_down_deme` in classic itself says
# # "Wild choice ot 250 seems reasonable. (_min_pool_size defaults to
# # 250)" -- but the ACTUAL static const initializer, two hundred and
# # eighty-nine lines later in the SAME header (metapopulation.h:540), is
# # `= 20`, not `250`. This is a stale comment in classic, not a port
# # defect: confirmed by reading metapopulation.h directly, not inferred
# # from the comment. Ported using the real value, 20.

_EPSILON: score_t = _to_score_t(1e-6)
# C++: moses/moses/metapopulation/merging.cc:548
# `#define EPSILON static_cast<score_t>(1e-6)`, local to
# `keep_top_unique_candidates`. Bit-identical to `types.py`'s own
# `_EQ_EPSILON` (`types.cc:158`'s SEPARATE `#define EPSILON` local to
# `composite_score::operator==`) purely by coincidence of both authors
# picking the same round number -- computed fresh here, under this file's
# own citation, rather than importing `_EQ_EPSILON` and relying on that
# coincidence to keep holding.


def useful_score_range(self: metapopulation) -> score_t:
    """C++: moses/moses/metapopulation/metapopulation.h:277 metapopulation::useful_score_range

    "Given the current complexity temp, return the range of scores that
    are likely to be selected by the select_exemplar routine" (the
    header's own comment). Declared inline in the header, not merging.cc --
    see module docstring for why it is ported here regardless.

    `_params.complexity_temperature * 30.0 / 100.0`: `complexity_temperature`
    is `score_t` (float), `30.0`/`100.0` are C++ double literals, so the
    multiplication and division both happen in DOUBLE precision (the
    float32 operand promotes), and only the function's own RETURN narrows
    back to `score_t` -- ported as a double-precision Python expression
    narrowed once via `_to_score_t` at the end, matching that.
    """
    return _to_score_t(self._params.complexity_temperature * 30.0 / 100.0)


def trim_down_deme(self: metapopulation, deme: deme_t) -> None:
    """C++: moses/moses/metapopulation/merging.cc:56 metapopulation::trim_down_deme

    "Trim the demes down to size ... instances below a cut-off score have
    no chance at all of getting selected. So just eliminate them now"
    (merging.cc:41-55's own comment, condensed). Requires `deme` already
    score-sorted descending (`sort_demes`'s own contract, and this
    function's own C++ doc comment: "the instances in the deme must be in
    score-sorted order").

    `for (size_t i = deme.size() - 1; 0 < i; --i)` (merging.cc:72) NEVER
    visits index 0 -- deliberate, not an off-by-one: index 0 is where
    `top_sc` itself came from, and can never be `< bot_sc` for a
    non-negative `useful_score_range()`. `deme.size() - 1` cannot itself
    underflow: the guard just above (`_min_pool_size >= len(deme): return`)
    already handles `len(deme) == 0`. Ported as an explicit index scan
    (matching `hill_climbing.py`'s own established convention for
    erase-during-scan, `resize_by_score`'s own docstring) rather than
    C++'s iterator/`pop_back` dance, which Python has no direct
    equivalent for: `deme.pop()` always removes the CURRENT last element,
    and `i` walks down in lock-step with the shrinking size, so `deme[i]`
    is provably always that same last element (or, once an element
    survives, simply the next one further from the tail) at every step --
    traced by hand for both the "pop every step" and "pop no steps" cases,
    and every mix of the two, in `test_merging.py`.
    """
    if self._min_pool_size >= len(deme):
        return

    top_sc = deme[0].second.get_penalized_score()
    bot_sc = _to_score_t(top_sc - self.useful_score_range())

    i = len(deme) - 1
    while i > 0:
        score = deme[i].second.get_penalized_score()
        if score < bot_sc:
            deme.pop()
        i -= 1


def deme_to_trees(
    self: metapopulation,
    deme: deme_t,
    rep: representation,
    pot_candidates: scored_combo_tree_set,
) -> None:
    """C++: moses/moses/metapopulation/merging.cc:94 metapopulation::deme_to_trees

    Serial port of the OMP-parallel, mutex-guarded `select_candidates`
    lambda (merging.cc:98-114) -- this port has no threading anywhere
    (metapopulation.py's own module docstring makes the same call for
    `_merge_mutex`), so the parallel `for_each` becomes a plain `for`
    loop with no behavioural difference: insertion order into
    `pot_candidates` (`scored_combo_tree_set`, unordered in both
    languages -- types.py's own DIVERGENCE note) is unobservable either
    way.

    `rep.get_candidate(inst, true)` (merging.cc:106) passes a
    `scored_instance<composite_score>` where `get_candidate` expects
    `const instance&` -- an implicit conversion via `tagged_item::operator
    const Item&() const` (opencog/util/functional.h:82, `return this->
    first;`). Python has no implicit-conversion-operator equivalent, so
    this is spelled out explicitly: `rep.get_candidate(inst.first, True)`.

    # DIVERGENCE(C++ moses/moses/metapopulation/merging.cc:110):
    # `scored_combo_tree sct(tr, deme.getID(), inst_csc);` supplies a
    # `demeID_t`. Dropped here -- not a new narrowing this task
    # introduces, but the first call site IN THIS FILE that would have
    # supplied a real one: `scored_combo_tree` itself carries no deme id
    # in this port (types.py's own class docstring, a scope decision from
    # S10 Task 0, before this task).

    `inst_csc = inst.second` is a REFERENCE into the deme's own
    `scored_instance` (later trimmed/discarded/rescored elsewhere).
    `scored_combo_tree.__init__` does not copy its `cs` argument (a bare
    assignment, types.py) where C++'s own constructor member-initializes
    `_cscore` from its parameter BY VALUE -- `copy.copy(inst_csc)` is
    passed explicitly to reproduce that copy (see module docstring's
    Aliasing discipline section).
    """
    for inst in deme:
        inst_csc = inst.second
        inst_sc = inst_csc.get_score()
        if inst_sc <= very_worst_score or not math.isfinite(inst_sc):
            continue

        tr = rep.get_candidate(inst.first, True)
        sct = scored_combo_tree(tr, copy.copy(inst_csc))
        pot_candidates.insert(sct)


def merge_candidates(self: metapopulation, candidates: scored_combo_tree_set) -> None:
    """C++: moses/moses/metapopulation/merging.cc:146 metapopulation::merge_candidates

    "Merge the given set of candidates into the metapopulation. It is
    assumed that these candidates have already be vetted for quality,
    quantity, suitability, etc." (merging.cc:142-145's own comment).
    Logger calls and `_merge_mutex` locking dropped (module-wide
    convention; metapopulation.py's own "no threading" note).

    The DEFAULT arm (`not _params.discard_dominated`, merging.cc:164-167)
    is functionally identical to metapopulation.py's own `init()`, which
    inlines this SAME arm's body because `init()` (Task 2's scope) needed
    something to populate `_scored_trees` before this method existed --
    that inlining is left exactly as committed (`metapopulation.py` is
    out of this task's file ownership) rather than refactored to call
    this method, so the same handful of lines now exist in two places.
    Real C++ has exactly one copy of this logic; `init()`
    (metapopulation.cc:71) calls THIS method. The duplication here is a
    consequence of this task's own file-ownership split, not a C++
    behaviour being reproduced.

    `_scored_trees.insert(new scored_combo_tree(cnd))` heap-allocates an
    independent COPY of `cnd` before inserting -- ported via `_copy_sct`
    (Task 2's helper, imported and reused, not reimplemented; see module
    docstring).
    """
    if self._params.discard_dominated:
        raise NotImplementedError(
            "metapopulation::merge_candidates's domination arm "
            "(merging.cc:168-175, merge_nondominated, domination.cc) is "
            "not ported; leave metapop_parameters.discard_dominated at its "
            "default, False"
        )
    for cnd in candidates:
        self._scored_trees.insert(_copy_sct(cnd))


def merge_demes(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> bool:
    """C++: moses/moses/metapopulation/merging.cc:192 metapopulation::merge_demes

    "Given a vector of demes, and the corresponding representations for
    those demes, convert the instances in the demes into scored combo
    trees, and merge them into the metapopulation" (merging.cc:178-180's
    own comment). `all_demes[j][i]` is breadth-first-deme `j`'s `i`-th
    (sub-sampled) deme; `reps[j]` is the ONE representation shared by
    every deme in `all_demes[j]` -- exactly `deme_expander._demes`/
    `_reps`'s own shape (S10 Task 4, `deme_expander.py`).

    Logger calls dropped throughout (module-wide convention).

    `_params.discard_dominated`, `_params.do_boosting`, `diversity_enabled()`,
    `_filter_params.n_subsample_demes` are all gates that default OFF --
    see module docstring's own Gates section for the full table and each
    one's exact citation. The behavioural-score recompute block
    (merging.cc:258-296) is ported as a SINGLE raise on the union of all
    four conditions (merging.cc:264-265's own `if`), not as the real
    bscore-recompute-and-rebuild machinery: every one of its four gating
    flags is independently out of this port's scope (do_boosting/
    discard_dominated/diversity route into machinery this whole project
    does not implement; `keep_bscore` alone has no OTHER unported
    dependency, but the brief this task was given names this exact block
    and its exact condition for a faithful gate-and-raise, so it is
    treated the same as the other three rather than singled out for a
    partial port).

    Real C++ ALSO has a second, independent `if (_params.discard_dominated)`
    (merging.cc:298-328, guarding `remove_dominated`) further down. In
    this port that block is UNREACHABLE: the union gate above already
    raises whenever `discard_dominated` is True, since it is one of the
    OR'd conditions, and control can never fall through it to reach a
    second check of the same flag. Not written out as a second, literally
    dead `if NotImplementedError` -- would be inert code with no way to
    ever execute, which is worse than the alternative of simply noting
    the correspondence here.
    """
    self.sort_demes(all_demes)
    self.keep_top_unique_candidates(all_demes, reps)

    self.recompute_scores_over_whole_dataset(all_demes, reps)
    pass_filter = self.ss_filter(all_demes, reps)

    pot_candidates = scored_combo_tree_set()
    for j in range(len(all_demes)):
        if not pass_filter[j]:
            continue
        demes = all_demes[j]
        for i in range(len(demes)):
            deme = demes[i]
            self.trim_down_deme(deme)
            self.deme_to_trees(deme, reps[j], pot_candidates)

    candidates = self.get_new_candidates(pot_candidates)

    if (
        self.diversity_enabled()
        or self._params.do_boosting
        or self._params.discard_dominated
        or self._params.keep_bscore
    ):
        raise NotImplementedError(
            "metapopulation::merge_demes's behavioural-score recompute "
            "block (merging.cc:258-296), gated on diversity_enabled() or "
            "do_boosting or discard_dominated or keep_bscore, is not "
            "ported; leave all four at their defaults (diversity disabled, "
            "do_boosting=False, discard_dominated=False, keep_bscore=False)"
        )

    if candidates.empty():
        return False

    self.update_best_candidates(candidates)

    done = False
    if self._params.merge_callback is not None:
        raise NotImplementedError(
            "metapopulation::merge_demes's merge_callback hook "
            "(merging.cc:338-339) is not ported; leave "
            "metapop_parameters.merge_callback at its default, None"
        )
    self.merge_candidates(candidates)

    if self._params.do_boosting:
        self._ensemble.add_candidates(candidates)
        self.rescore()

    if self.diversity_enabled():
        raise NotImplementedError(
            "metapopulation::merge_demes's set_diversity() call "
            "(merging.cc:349-350, diversity.cc) is not ported; leave "
            "metapop_parameters.diversity disabled (pressure == 0.0, the "
            "default)"
        )

    self.resize_metapop()

    return done


def resize_metapop(self: metapopulation) -> None:
    """C++: moses/moses/metapopulation/merging.cc:359 metapopulation::resize_metapop

    "Weed out excessively bad scores ... Keep the minimum metapop size to
    ... 250 (aka _min_pool_size). But if the population starts exploding
    ... apply more and more stringent bounds" (metapopulation.h:240-256's
    own doc comment, condensed) -- phase 1 (merging.cc:375-395) truncates
    the low-scoring tail past `_min_pool_size`; phase 2 (merging.cc:397-436)
    randomly evicts down to a growing cap when the population is still too
    large after phase 1.

    Never constructs a NEW `scored_combo_tree`/`composite_score` -- only
    removes existing pool members -- so there is no copy-vs-alias question
    here (contrast `deme_to_trees`/`merge_candidates` above).

    `scored_combo_tree_ptr_set` (types.py, S10 Task 0's own file, out of
    this task's ownership) exposes only `insert`/`erase(sct)`/iteration/
    `len`/`empty` -- no positional erase, unlike `boost::ptr_set::erase(
    iterator)`. This uses a LOCAL sorted snapshot (`ordered = list(self.
    _scored_trees)`, the exact live objects, not copies) to do the
    position arithmetic C++'s iterators do, then calls the container's own
    `erase(sct)` (equivalence-bisection by `sct_score_greater`) on each
    victim by OBJECT REFERENCE, mirroring the removal in `ordered` in
    lock-step so later index arithmetic stays valid. `erase(sct)`'s
    bisection can only mistake one tied object for another when BOTH
    compare equal under `sct_score_greater`'s full chain (composite_score,
    then raw score, then a total structural tree order,
    `_size_tree_less`) -- which happens only for two objects with
    genuinely identical scores AND identical tree structure, i.e.
    interchangeable duplicates the container's own docstring already
    permits ("there is no way to guarantee that a given tree appears only
    once in the set"). Reaching into `_scored_trees._set._items` directly
    would avoid even that remote ambiguity but was judged a worse
    trade-off: it would coordinate with S10 Task 0's own private
    implementation detail from a file this task does not own, for a
    difference that is unobservable for any two objects that are not
    already indistinguishable duplicates.

    `_merge_count` (metapopulation.h:542, `size_t`) is genuine per-instance
    state C++ initializes to 0 in the constructor's own member-initializer
    list -- this is the one field in this whole module that looked like a
    one-line-hook candidate for `metapopulation.__init__` (out of this
    task's file ownership). Avoided instead: `getattr(self, "_merge_count",
    0)` lazily creates it at 0 on first use, which is observably identical
    to "initialized to 0 at construction" since `resize_metapop` (here) is
    the ONLY reader or writer of this field anywhere in the class (grepped
    against the whole of `merging.cc`/`metapopulation.cc`).

    `randGen().randint(n)` (merging.cc:427) is `randGen().randint_n(n)`
    here -- `rng.py`'s own naming split for `opencog::RandGen::randint()`
    (no argument) vs `randint(int n)`, the same replacement
    `deme_expander.py`/`neighborhood_sampling.py` already use.

    CORRECTED (S10 branch review, [Important 1]): this docstring used to
    claim a 64-bit mask fed C++ "the SAME enormous value ... exactly as
    C++ does". MEASURED FALSE. `MT19937RandGen::randint` takes a plain
    `int`, not a `size_t` (`mt19937ar.h:54`), so `popsz - offset`
    (merging.cc:427, both `size_t`) does underflow to a huge 64-bit value
    in C++'s own subtraction, but that value is NARROWED to 32 bits at the
    call boundary before any modulo happens -- C++'s `randint(int n)` body
    never sees the 64-bit value at all. Confirmed against a compiled probe
    on the real cogutil class (fresh `MT19937RandGen(1)`, three sequential
    calls): `randint(-25) == 22`, `randint(-1) == 0`, `randint(-29) == 6`
    -- C++ receives a SMALL SIGNED int (the underflowed subtraction's low
    32 bits, reinterpreted as two's complement), never an enormous one.
    Reproduced here via an EXPLICIT SIGNED 32-bit mask (replacing the
    unsigned 64-bit one this docstring previously specified), matching the
    narrowing the `int` parameter performs in C++; `rng.py`'s own
    `randint_n` needed a matching fix (see its own docstring) since
    Python's `%` takes the sign of the DIVISOR while C's truncating `%`
    takes the sign of the (here always non-negative) dividend -- the mask
    alone is not sufficient, both fixes together are, and both are what
    `test_merging.py` pins against the measured 22/0/6.

    Still reachable only in principle, exactly as before: `_min_pool_size`
    (20) is strictly less than `offset` (50), and the default
    `_params.cap_coef` (50) never drives phase 2 this far in practice.
    What differs from the retracted claim is only the SIZE of what is
    reproduced -- a small int bounded by `offset` (at most 50 in
    magnitude), not an astronomical one -- while still consuming exactly
    one RNG draw per eviction, same as before.

    What happens NEXT still differs from C++, and for a reason broader
    than the "huge index" story this docstring used to tell: whenever
    `popsz < offset` (the only way `diff` goes negative), `which = offset
    + randint_n(diff)` is ALWAYS `>= offset > popsz` -- not just for the
    formerly-claimed enormous draws, but for EVERY draw reaching this
    branch. In C++, `_scored_trees` is a `boost::ptr_set` over `std::set`
    (`types.h:381-395`): `std::next(it, which)` on that bidirectional
    iterator advances by repeatedly incrementing, walks off `end()`, and
    keeps incrementing an already-past-the-end iterator -- undefined
    behaviour regardless of whether `which` is astronomically large or
    merely a little too big. Python has no raw-iterator analogue to
    corrupt, so any out-of-range `which` here surfaces as a plain
    `IndexError` from `ordered[which]`, a safe, loud failure in place of
    C++'s undefined behaviour, exactly the "reject rather than read out of
    bounds" precedent `representation.get_candidate`'s own docstring
    already sets.
    """
    if self.size() <= self._min_pool_size:
        return

    ordered = list(self._scored_trees)  # sct_score_greater order, best first

    top_score = ordered[0].get_penalized_score()
    worst_score = _to_score_t(top_score - self.useful_score_range())

    idx = self._min_pool_size
    while idx < len(ordered):
        if ordered[idx].get_penalized_score() < worst_score:
            break
        idx += 1

    for victim in ordered[idx:]:
        self._scored_trees.erase(victim)
    del ordered[idx:]

    merge_count = getattr(self, "_merge_count", 0) + 1
    self._merge_count = merge_count

    cap = self._params.cap_coef
    cap *= merge_count + 250.0
    cap *= 1 + 2.0 * math.exp(-merge_count / 500.0)
    popsz_cap = int(cap)
    popsz = len(ordered)

    while popsz_cap < popsz:
        offset = 50
        diff = (popsz - offset) & 0xFFFFFFFF
        if diff & 0x80000000:
            diff -= 0x100000000
        which = offset + randGen().randint_n(diff)
        victim = ordered[which]  # IndexError on an out-of-range draw -- see docstring
        self._scored_trees.erase(victim)
        del ordered[which]
        popsz -= 1

    # `_cached_dst.erase_ptr_seq(ptr_seq)` (merging.cc:440): NOT ported,
    # not even as a no-op stub. `_cached_dst`'s field type, `cached_dst`
    # (metapopulation.h:355-384), has `erase_ptr_seq`'s ENTIRE body inside
    # `#ifdef ENABLE_DST_CACHE ... #endif` (diversity.cc:262-272), and
    # `ENABLE_DST_CACHE` is commented out at its own definition site
    # (`// #define ENABLE_DST_CACHE`, metapopulation.h:53) -- so the real
    # call does nothing in every build this port targets. `_cached_dst`
    # itself is not ported anywhere in this project (diversity is out of
    # scope -- metapopulation.py's own module docstring); omitting a call
    # whose own body is empty is not a gap to flag, since there would be
    # nothing for a stub to do.


def get_new_candidates(
    self: metapopulation, mcs: scored_combo_tree_set
) -> scored_combo_tree_set:
    """C++: moses/moses/metapopulation/merging.cc:460 metapopulation::get_new_candidates

    "Given a set of candidates, return the set of candidates not already
    present in the metapopulation" (merging.cc:456-458's own comment).
    Serial port of the OMP-parallel, mutex-guarded `insert_new_candidate`
    lambda (merging.cc:475-485) -- same "no threading anywhere" reasoning
    as `deme_to_trees` above.

    `tr == v.get_tree()` (merging.cc:479) is real `combo_tree::operator==`
    -- structural comparison (`opencog/util/tree.h:3011`) -- ported via
    `_combo_tree_structural_eq` (types.py), the same helper
    `scored_combo_tree.__eq__`/`scored_combo_tree_equal` already use for
    this exact comparison, imported and reused rather than reimplemented.

    `res.insert(cnd)` (merging.cc:483) copies via `std::unordered_set::
    insert`'s own copy-constructor semantics (types.h:362-367's comment,
    quoted in types.py's `scored_combo_tree_set` docstring) -- ported via
    `_copy_sct`.
    """
    res = scored_combo_tree_set()
    for cnd in mcs:
        tr = cnd.get_tree()
        already_present = any(
            _combo_tree_structural_eq(tr, existing.get_tree()) for existing in self._scored_trees
        )
        if not already_present:
            res.insert(_copy_sct(cnd))
    return res


# ---------------------------------------------------------------------------
# libstdc++'s std::sort (introsort), reproduced exactly.
#
# `sort_demes` (merging.cc:507) calls `boost::sort(deme,
# std::greater<scored_instance<composite_score>>())`. `boost::range::sort`
# (boost/range/algorithm/sort.hpp, read directly on this machine) is a bare
# forward to `std::sort(begin, end, pred)` -- a DIFFERENT algorithm from the
# `std::partial_sort` the `_partial_sort` family above/in hill_climbing.py
# reproduces (heap-select + sort-heap): libstdc++'s `std::sort` is introsort
# -- median-of-3 quicksort down to chunks of `_S_threshold == 16` elements,
# then one final insertion-sort pass over the whole range, with a heapsort
# fallback (`std::__partial_sort(first, last, last, comp)`, i.e. exactly
# `hill_climbing._partial_sort`) if the `2 * __lg(n)` recursion-depth budget
# is exhausted. This was previously ported as `deme.sort(reverse=True)`, on
# the argument that "boost::sort/std::sort makes no stability guarantee, so
# there is no C++ tie-order for a stable sort to diverge from." **That
# argument was measured and is FALSE**, the same way S12 found it false for
# `std::partial_sort` (docs/CARRY-FORWARD.md Sec 5b.1): "no stability
# guarantee" means the order is unspecified, not that libstdc++ doesn't
# produce one, deterministically, every time.
#
# MEASURED, not assumed: tools/cpp_probe/sort_demes_probe.cc runs
# `std::sort(v.begin(), v.end(), std::greater<deme_inst_t>())` -- the exact
# call shape `boost::sort` forwards to -- over hand-built samples straddling
# libstdc++'s own `_S_threshold = 16` boundary (confirmed by reading
# `/usr/include/c++/16/bits/stl_algo.h:1807` on this toolchain, gcc 16.2.1)
# and at realistic MOSES deme sizes (up to 2000; ordinary boolean runs'
# neighbourhood/deme sizes land in the hundreds to low thousands, e.g. the
# 703-instance neighbourhood docs/CARRY-FORWARD.md's own crossover
# measurements record), tagging each element's identity in its INSTANCE BITS
# (which the comparator never reads) so the dumped id order after sorting IS
# the chosen permutation. Result (tests/fixtures/sort_demes.json,
# `test_sort_matches_libstdcxx_order`):
#
# * For n <= 16 (all-tied), std::sort reproduces the input order EXACTLY --
#   `__introsort_loop`'s `while (last - first > 16)` never fires, so no
#   quicksort partition ever runs, and the closing `__final_insertion_sort`
#   makes zero moves when every comparison ties (insertion sort only moves
#   an element strictly before an EARLIER one it's less than).
# * For n >= 17 (all-tied), std::sort measurably reorders: 17 all-tied
#   elements come back `[8, 16, 15, 14, 13, 12, 11, 10, 9, 0, 7, 6, 5, 4, 3,
#   2, 1]`, where a stable sort leaves them `[0, 1, ..., 16]` -- the
#   quicksort partition that runs once `n > 16` swaps pairs from both ends
#   of the range inward even when every comparison is a tie (`comp` never
#   returns true, so `__unguarded_partition`'s two inner `while` loops never
#   advance, and every outer iteration still swaps and steps inward). The
#   same reordering appears at every larger size measured (20, 32, 50, 100,
#   703, 2000) and with NaN ties mixed in.
# * `sort_demes`'s realistic input is essentially NEVER n <= 16: a boolean
#   deme's ordinary size is in the hundreds to low thousands (see above), so
#   this is not a corner case being guarded against -- it is the common
#   case silently producing the wrong permutation under the old code.
#
# That reordering reaches an observable outcome exactly the way it does for
# `resize_deme` (S12): `sort_demes`'s only caller (`merge_demes`) feeds its
# result straight to `keep_top_unique_candidates`, whose own closing line is
# an UNCONDITIONAL `deme.erase(deme.begin() + top_cnd, deme.end())`
# (merging.cc:600) -- a truncation, so tie order decides which tied
# instances occupy the surviving prefix. Measured directly (no dedup
# approximation needed): truncating 100 all-tied instances to the top 17
# keeps ids `{59..75}` (specifically `[62,74,73,72,71,70,69,68,67,66,65,64,
# 63,75,61,60,59]`) under std::sort's real order, versus `{0..16}` under a
# stable sort -- completely disjoint survivor sets from the same input.
#
# The source of truth is /usr/include/c++/16/bits/stl_algo.h; the functions
# below are one-to-one transliterations, keeping libstdc++'s own parameter
# names so the correspondence is checkable line by line. `_GLIBCXX_MOVE` is
# a no-op here for the same reason `_partial_sort`'s own block comment
# (hill_climbing.py) gives: Python assignment rebinds, and every moved-from
# slot is overwritten before it is read again.
#
# PORTABILITY CAVEAT (inherited from `_partial_sort`, docs/CARRY-FORWARD.md
# Sec 5b.1 / S12 task-report.md Sec 9 item 3): this is pinned to THIS
# libstdc++'s (16.2.1) actual introsort implementation, in particular
# `_S_threshold == 16` and the exact partition/pivot/insertion-sort
# functions above. A different libstdc++ version could change the constant
# or the algorithm shape; the fixture manifest records the toolchain paths
# used to capture it, but not the libstdc++ version itself.
# ---------------------------------------------------------------------------


_S_THRESHOLD = 16
# C++: /usr/include/c++/16/bits/stl_algo.h:1807 `enum { _S_threshold = 16 };`
# ONE constant, read by both `_introsort_loop` and `_final_insertion_sort`
# below, exactly as libstdc++'s single `enum` value is read by both of ITS
# functions -- deliberately not two independent literal `16`s, since
# `_final_insertion_sort`'s own correctness (unguarded insertion never
# running off the front of the range) relies on its guarded prefix being AT
# LEAST as wide as the maximum displacement `_introsort_loop`'s own
# threshold can leave behind; two literals that could drift apart
# independently would be a latent correctness hazard `_S_threshold` being
# one `enum` in C++ does not have.


def _lg(n: int) -> int:
    """Not itself a ported C++ symbol -- libstdc++ `std::__lg`
    (bits/stl_algobase.h:1558), `floor(log2(n))` for `n > 0`, used only to
    seed `_sort`'s recursion-depth budget.
    """
    return n.bit_length() - 1


def _move_median_to_first(v, result: int, a: int, b: int, c: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__move_median_to_first`
    (bits/stl_algo.h:88). Swaps whichever of `v[a], v[b], v[c]` is the
    median (under `comp`) into `v[result]`. See the block comment above.
    """
    if comp(v[a], v[b]):
        if comp(v[b], v[c]):
            v[result], v[b] = v[b], v[result]
        elif comp(v[a], v[c]):
            v[result], v[c] = v[c], v[result]
        else:
            v[result], v[a] = v[a], v[result]
    elif comp(v[a], v[c]):
        v[result], v[a] = v[a], v[result]
    elif comp(v[b], v[c]):
        v[result], v[c] = v[c], v[result]
    else:
        v[result], v[b] = v[b], v[result]


def _unguarded_partition(v, first: int, last: int, pivot: int, comp) -> int:
    """Not itself a ported C++ symbol -- libstdc++ `std::__unguarded_partition`
    (bits/stl_algo.h:1833). See the block comment above.
    """
    while True:
        while comp(v[first], v[pivot]):
            first += 1
        last -= 1
        while comp(v[pivot], v[last]):
            last -= 1
        if not (first < last):
            return first
        v[first], v[last] = v[last], v[first]
        first += 1


def _unguarded_partition_pivot(v, first: int, last: int, comp) -> int:
    """Not itself a ported C++ symbol -- libstdc++ `std::__unguarded_partition_pivot`
    (bits/stl_algo.h:1855). See the block comment above.
    """
    mid = first + (last - first) // 2
    second = first + 1
    _move_median_to_first(v, first, second, mid, last - 1, comp)
    return _unguarded_partition(v, second, last, first, comp)


def _unguarded_linear_insert(v, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__unguarded_linear_insert`
    (bits/stl_algo.h:1750). See the block comment above.
    """
    val = v[last]
    next_ = last - 1
    while comp(val, v[next_]):
        v[last] = v[next_]
        last = next_
        next_ -= 1
    v[last] = val


def _insertion_sort(v, first: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__insertion_sort`
    (bits/stl_algo.h:1770). See the block comment above.
    """
    if first == last:
        return
    i = first + 1
    while i != last:
        if comp(v[i], v[first]):
            val = v[i]
            # `std::move_backward(first, i, i + 1)`: shift v[first:i] right
            # by one slot.
            j = i
            while j != first:
                v[j] = v[j - 1]
                j -= 1
            v[first] = val
        else:
            _unguarded_linear_insert(v, i, comp)
        i += 1


def _unguarded_insertion_sort(v, first: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__unguarded_insertion_sort`
    (bits/stl_algo.h:1796). See the block comment above.
    """
    i = first
    while i != last:
        _unguarded_linear_insert(v, i, comp)
        i += 1


def _final_insertion_sort(v, first: int, last: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__final_insertion_sort`
    (bits/stl_algo.h:1813). `_S_threshold` is 16 on this toolchain
    (bits/stl_algo.h:1807). See the block comment above.
    """
    threshold = _S_THRESHOLD
    if last - first > threshold:
        _insertion_sort(v, first, first + threshold, comp)
        _unguarded_insertion_sort(v, first + threshold, last, comp)
    else:
        _insertion_sort(v, first, last, comp)


def _introsort_loop(v, first: int, last: int, depth_limit: int, comp) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__introsort_loop`
    (bits/stl_algo.h:1884). Recurses on the LATER partition and loops on the
    EARLIER one (`_partial_sort` -- `hill_climbing.py`'s heap-select sort --
    is the depth-exhausted fallback, i.e. `std::__partial_sort(first, last,
    last, comp)`), bounding recursion depth to `depth_limit` regardless of
    input size. See the block comment above.

    UNTESTED, reported rather than claimed as covered: the `depth_limit ==
    0` branch exists to bound median-of-3 quicksort's OWN known worst case
    (an adversarially-ordered input can force it to O(n^2) even with
    median-of-3 pivoting -- this fallback is why introsort is safe against
    that in the first place). Deliberately reaching it needs a specific
    adversarial construction (e.g. a McIlroy-style killer sequence), not
    realistic MOSES score data; every fixture case measured here (including
    `all_tied_2000`, tied and therefore among the cheapest inputs for THIS
    exact partition scheme to mis-handle) stays well inside the `2 *
    _lg(n)` budget. Mutating `_lg` to `n.bit_length()` (raising the budget)
    passes every test in this file for exactly that reason -- not because
    the fallback is unreachable in principle, but because none of this
    file's realistic-sized fixtures happen to reach it either way. Kept
    faithful to libstdc++ regardless.
    """
    threshold = _S_THRESHOLD
    while last - first > threshold:
        if depth_limit == 0:
            _partial_sort(v, first, last, last, comp)
            return
        depth_limit -= 1
        cut = _unguarded_partition_pivot(v, first, last, comp)
        _introsort_loop(v, cut, last, depth_limit, comp)
        last = cut


def _sort(v, first: int, last: int, comp=_greater) -> None:
    """Not itself a ported C++ symbol -- libstdc++ `std::__sort`
    (bits/stl_algo.h:1908), which is what `std::sort(first, last, comp)`
    (bits/stl_algo.h:4837) forwards to, which is what `boost::sort(deme,
    comp)` (merging.cc:507) forwards to (`boost::range::sort` is a bare
    forward to `std::sort`, confirmed by reading
    boost/range/algorithm/sort.hpp). Sorts `v[first:last]` in place,
    descending under `comp = _greater` (`std::greater<deme_inst_t>`). See
    the block comment above for the measurement this reproduces.

    Operates on a slice of `v` by index, exactly as `_partial_sort` does,
    so the same call shape works for a whole-deme sort or any sub-range.
    """
    if first != last:
        _introsort_loop(v, first, last, _lg(last - first) * 2, comp)
        _final_insertion_sort(v, first, last, comp)


def sort_demes(self: metapopulation, all_demes: list[list[deme_t]]) -> None:
    """C++: moses/moses/metapopulation/merging.cc:503 metapopulation::sort_demes

    `boost::sort(deme, std::greater<scored_instance<composite_score>>())`
    (merging.cc:507) sorts each deme DESCENDING by `scored_instance::
    operator>` (`this->second > other.second`, i.e. `composite_score::
    operator>`).

    # DIVERGENCE(C++ moses/moses/metapopulation/merging.cc:507): this used
    # to be `deme.sort(reverse=True)`, on the argument that "boost::sort
    # makes no stability guarantee, so there is no C++ tie-order for a
    # stable sort to diverge from." That argument was measured and found
    # FALSE -- see the block comment above `_sort` for the counterexample
    # (17 all-tied elements) and the observable consequence (a truncation
    # downstream in `keep_top_unique_candidates` that lets tie order decide
    # survivors). `_sort` now reproduces libstdc++'s real introsort
    # permutation instead of a merely-valid one.

    The rest of `sort_demes` (merging.cc:509-517) is dead code even in
    C++: a nested loop over every instance in every deme that binds
    `const field_set &fields = deme.fields();` and does nothing else (the
    one line that would use `fields`, a `logger().debug()` call, is
    commented out). Not ported -- it has no observable effect in classic
    either.
    """
    for ss_demes in all_demes:
        for deme in ss_demes:
            _sort(deme, 0, len(deme), _greater)


def keep_top_unique_candidates(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> None:
    """C++: moses/moses/metapopulation/merging.cc:520 metapopulation::keep_top_unique_candidates

    Walks each already-sorted deme with a `(prev, it)` pair, dropping `it`
    whenever it is both approximately score-tied with `prev` (within
    `EPSILON = 1e-6`) AND its decoded, reduced tree is structurally
    identical to `prev`'s -- "Remove duplicates till at most top_cnd
    unique candidates remains" (merging.cc:550-551's own comment) -- then
    truncates whatever remains to `top_cnd` entries.

    `rep.get_candidate(*it, true)` (merging.cc:573) is the same implicit
    `scored_instance -> const instance&` conversion `deme_to_trees`'s own
    docstring explains -- spelled `rep.get_candidate(deme[it_idx].first,
    True)` here. `prev_tr == tr` (merging.cc:577) is real
    `combo_tree::operator==` (structural) -- ported via
    `_combo_tree_structural_eq`, same as `get_new_candidates` above.
    `is_approx_eq(prev_sc, sc, EPSILON)` (merging.cc:572) is ported via
    `_is_approx_eq`/this module's own `_EPSILON` (see their own
    citations above).

    Ported as an explicit index scan (`it_idx`/`prev_idx`), not C++'s
    iterator pair, matching `hill_climbing.py`'s own established
    erase-during-scan convention: `it = deme.erase(it)` (merging.cc:578,
    "erase returns an iterator to the element following the one erased")
    becomes `del deme[it_idx]` with `it_idx` UNCHANGED (the `continue` at
    merging.cc:580 skips `prev_it = it++`, so `prev_idx` stays fixed too)
    -- the element that slides into `it_idx`'s old slot is exactly what
    C++'s returned iterator would have pointed at.

    MEASURED HAZARD this port does not reproduce, and does not need an
    explicit guard against: C++'s own loop starts with `it = deme.begin(),
    prev_it = it++;` (merging.cc:568, `it` at index 1, `prev_it` at index
    0) and its condition is `it != deme.begin() + top_cnd` -- an
    INEQUALITY, not a bound. Whenever `top_cnd == 0` (reachable:
    `_params.max_candidates_per_deme == 0` is a legal, if unusual, value
    the user can set; `deme.n_evals == 0` is reachable too, per this
    function's own comment, "the true number of evals is less than the
    deme size") this condition is satisfied (`it` at index 1 is never
    EQUAL to `deme.begin() + 0` while it only ever increases), so the loop
    runs and immediately dereferences `*it` -- which is `deme.end()` for a
    1-element deme, or already one-past-the-end even sooner for an EMPTY
    deme (`it++` on `deme.begin() == deme.end()` is itself undefined
    behaviour, before the loop is even reached) -- reading/incrementing
    past the container, undefined behaviour in every case. The index-based
    translation below has no analogous failure mode: `it_idx` (starting at
    1) only ever increases and `top_cnd` only ever decreases, so `while
    it_idx < top_cnd` is a plain, well-defined bound for EVERY `top_cnd >=
    0` including 0 and 1 -- `top_cnd == 0` simply never enters the loop
    body at all (`1 < 0` is `False`), falling straight through to the
    unconditional final truncation, `del deme[top_cnd:]` (`del deme[0:]`,
    clearing the deme -- exactly the SANE behaviour the loop is clearly
    intended to have when there is nothing to compare: an empty range has
    no duplicates to remove). No separate early-return for `top_cnd == 0`
    is written here for exactly that reason: it would be dead code, doing
    nothing an ordinary `while` bound does not already do on its own --
    confirmed by mutation, not merely reasoned: deleting an EARLIER
    version of this function's own explicit `if top_cnd == 0: del
    deme[0:]; continue` special case changed no test's outcome.
    """
    for i in range(len(all_demes)):
        rep = reps[i]
        ss_demes = all_demes[i]
        for deme in ss_demes:
            top_cnd = min(deme.n_evals, len(deme))

            if self._params.max_candidates_per_deme >= 0:
                top_cnd = min(top_cnd, self._params.max_candidates_per_deme)
            if self._filter_params.n_subsample_demes > 1:
                top_cnd = min(top_cnd, self._filter_params.n_top_candidates)

            it_idx = 1
            prev_idx = 0
            while it_idx < top_cnd:
                sc = deme[it_idx].second.get_penalized_score()
                prev_sc = deme[prev_idx].second.get_penalized_score()
                if _is_approx_eq(prev_sc, sc, _EPSILON):
                    tr = rep.get_candidate(deme[it_idx].first, True)
                    prev_tr = rep.get_candidate(deme[prev_idx].first, True)
                    if _combo_tree_structural_eq(prev_tr, tr):
                        del deme[it_idx]
                        top_cnd = min(top_cnd, len(deme))
                        continue
                prev_idx = it_idx
                it_idx += 1

            del deme[top_cnd:]


def recompute_scores_over_whole_dataset(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> None:
    """C++: moses/moses/metapopulation/subsample.cc:42 recompute_scores_over_whole_dataset

    Declared in metapopulation.h but DEFINED in subsample.cc -- a file
    outside this whole S10 slice's file list, the same way domination.cc
    is (see module docstring). Called unconditionally from `merge_demes`
    (this task's own scope), so it needs a gate here the way
    `discard_dominated`'s domination arm does. Real C++'s entire body is
    `if (_filter_params.n_subsample_demes > 1) { ... }`
    (subsample.cc:47) -- a pure no-op when subsampling is disabled (the
    default, `n_subsample_demes == 0`). Ported as exactly that gate: the
    disabled arm does nothing, matching real C++ exactly rather than
    approximating it; the enabled arm raises, since subsample.cc's real
    body (re-scoring every candidate over the whole, un-subsampled
    dataset) is out of every file this task owns.
    """
    if self._filter_params.n_subsample_demes > 1:
        raise NotImplementedError(
            "metapopulation::recompute_scores_over_whole_dataset "
            "(subsample.cc, gated on n_subsample_demes > 1) is not "
            "ported; leave subsample_deme_filter_parameters."
            "n_subsample_demes <= 1 (the default)"
        )


def ss_filter(
    self: metapopulation,
    all_demes: list[list[deme_t]],
    reps: list[representation],
) -> list[bool]:
    """C++: moses/moses/metapopulation/subsample.cc:77 metapopulation::ss_filter

    Same out-of-slice-file situation as `recompute_scores_over_whole_dataset`
    above. "Breadth first demes pass vector. By default they all pass the
    filter" (subsample.cc's own comment): `std::vector<bool> bfpass(
    all_demes.size(), true);` is constructed BEFORE the
    `n_subsample_demes > 1` gate and returned unchanged whenever
    subsampling is disabled -- ported as exactly that.
    """
    if self._filter_params.n_subsample_demes > 1:
        raise NotImplementedError(
            "metapopulation::ss_filter (subsample.cc, gated on "
            "n_subsample_demes > 1) is not ported; leave "
            "subsample_deme_filter_parameters.n_subsample_demes <= 1 "
            "(the default)"
        )
    return [True] * len(all_demes)


def log_best_candidates(self: metapopulation) -> None:
    """C++: moses/moses/metapopulation/merging.cc:652 metapopulation::log_best_candidates

    "log the best candidates" (merging.cc:651's own comment). See this
    module's own docstring for why this method is added here rather than
    called back from `expand_deme` (`local_moses.py`).

    NOT ported: the `logger().is_info_enabled()` early-out (merging.cc:
    653-654) and every actual `logger()` call (merging.cc:657,661-663,
    666-668,667-671) -- no logging surface anywhere in this port
    (`metapopulation.py`'s own module docstring: "ostream_metapop/log_best_
    candidates/log_selected_exemplar -- printing/logging, no observable
    effect"). Once every print call is gone, `is_info_enabled()`'s early
    return has nothing left to guard, so dropping it too is not a SEPARATE
    divergence -- there is no remaining behaviour for it to gate.

    What IS ported: the two REAL branches this method's control flow
    walks before it would ever reach a `logger()` call -- `best_candidates
    ().empty()` (merging.cc:656) and `_params.do_boosting` (merging.cc:
    659) -- each reaching the SAME live accessor a real logging call would
    have evaluated as ITS OWN argument: `best_composite_score()` in the
    non-boosting arm (merging.cc:660, `logger().info() << "..." <<
    best_composite_score();`), `_ensemble.get_ensemble()` in the boosting
    arm (merging.cc:664, `_ensemble.get_ensemble().size()`) -- the
    identical "reach the unported boosting surface rather than stub it out
    separately" convention `metapopulation.py`'s own `best_composite_score`/
    `best_tree` already use for this exact `do_boosting` gate. Both
    accessors are pure reads with no side effect of their own
    (`best_composite_score` returns a copy; `ensemble.get_ensemble` is a
    trivial accessor over an ensemble that can never hold anything, since
    `add_candidates` always raises first -- `ensemble.py`) -- calling them
    and discarding the result is therefore observably a no-op UNLESS one
    of them raises, which is exactly the behaviour worth preserving: a
    caller that sets `do_boosting=True` still reaches the same unported
    surface through THIS method that it would through `best_composite_
    score`/`best_tree`, rather than this method silently pretending
    boosting works.

    NOT ported, in EITHER arm: the per-candidate `for` loops (merging.cc:
    661-663,667-668) and the nested `logger().is_debug_enabled()` block
    (merging.cc:665-671) -- every one of them exists SOLELY to format text
    for a `logger()` call that is itself not ported (`cand.get_score()`/
    `cand.get_tree()` in the first loop, `cand.get_weight()` in the second
    -- and `get_weight()` is not even a method this port's `scored_combo_
    tree` HAS: types.py's own class docstring names the boosting weight
    field as out of scope for this whole cycle). Looping over every
    element here would read the whole of `best_candidates()`/the ensemble
    for no reason but log-formatting busywork with nothing left to format
    -- dropped along with the print calls they exist to feed, rather than
    kept as loops that do nothing observable.
    """
    if self.best_candidates().empty():
        return
    if not self._params.do_boosting:
        self.best_composite_score()
    else:
        self._ensemble.get_ensemble()


metapopulation._min_pool_size = _MIN_POOL_SIZE
metapopulation.useful_score_range = useful_score_range
metapopulation.trim_down_deme = trim_down_deme
metapopulation.deme_to_trees = deme_to_trees
metapopulation.merge_candidates = merge_candidates
metapopulation.merge_demes = merge_demes
metapopulation.resize_metapop = resize_metapop
metapopulation.get_new_candidates = get_new_candidates
metapopulation.sort_demes = sort_demes
metapopulation.keep_top_unique_candidates = keep_top_unique_candidates
metapopulation.recompute_scores_over_whole_dataset = recompute_scores_over_whole_dataset
metapopulation.ss_filter = ss_filter
metapopulation.log_best_candidates = log_best_candidates
