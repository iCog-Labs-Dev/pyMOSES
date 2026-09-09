"""C++: moses/moses/metapopulation/domination.cc

P2 Task ("cheapest high-value P2 group"): fills in the domination gates
`merging.py`/`metapopulation.py` already carry as `NotImplementedError`
(`_params.discard_dominated`). Classic's own header comment
(domination.cc:44-58) says this code "is no longer used anywhere (as of
2012)" and is kept only because "there are some useful-seeming
sub-algorithms in it" -- ported anyway, faithfully, because
`metapopulationUTest.cxxtest` (the slice's registered cxxtest oracle,
`tests/moses/metapopulationUTest.cxxtest:74,119`) exercises exactly two of
these methods (`remove_dominated`, `get_nondominated_disjoint`) and
nothing else in the whole metapopulation subsystem -- a real, working
oracle, unlike the probe-based verification every other file in this
group needs.

## Composition: same "module-level functions the class binds" shape as `merging.py`

`domination.cc` is a THIRD translation unit defining more `metapopulation::`
members (see `merging.py`'s own module docstring for the two-translation-unit
precedent this extends to three). Every function below takes `self:
metapopulation` as its first parameter except the four C++ declares
`static` (`dominates`, `to_set`, `split`, `get_nondominated_iter`), which
are bound via `staticmethod` instead -- matching the "no `this`" shape a
real `static` member function has, while still being reachable as
`self.dominates(...)` the way `domination.cc`'s own code calls them
(unqualified, from inside other member functions, since C++ needs no
`this->` for a static sibling either).

`merging.py` imports this module for its `setattr` side effect (see this
file's bottom and `merging.py`'s own updated module docstring), so
`pymetta/moses/metapopulation/__init__.py` -> `merging` -> `domination`
already reaches this module transitively; nothing else needs to change to
keep `tests/moses/metapopulation/test_package_binding.py`'s "importing the
package gives the complete method set" invariant.

## `boost::logic::tribool` -> `True` / `False` / `None`

`dominates` returns a `boost::logic::tribool`: TRUE ("x dominates y"),
FALSE ("y dominates x"), or INDETERMINATE ("neither"). Python has no
three-valued boolean, so this is represented as `True` / `False` / `None`
(indeterminate = `None`) -- BUT every branch below that inspects one of
these results tests `dom is True` / `dom is False` / `dom is None`
explicitly, never bare truthiness (`if dom:` / `if not dom:`), because
`None` and `False` are both falsy in Python and conflating them would
silently turn "indeterminate" into "false" -- the same "type before value"
discipline this project applies to `vertex_equal` (never a bare `==` on a
vertex), applied here to a three-valued result instead of a vertex.

## `scored_combo_tree_ptr_vec` -> `list[scored_combo_tree]`, by reference

C++'s `scored_combo_tree_ptr_vec` (`std::vector<const scored_combo_tree*>`)
holds raw pointers INTO an existing container (`_scored_trees`, or a
caller-supplied `scored_combo_tree_set`) -- never copies -- so that
`boost::sort`+`set_difference`/`set_intersection` over those pointers is,
observably, an IDENTITY operation (two pointers are "the same element" iff
they have the same address; sorting by raw address is only ever used to
satisfy `std::set_difference`/`std::set_intersection`'s "inputs must be
sorted" precondition, never because the SORT ORDER itself is read back).
This port reproduces that with plain Python lists of object references
(no copying during the recursive filtering) and `id()`-keyed set
operations in place of `boost::sort` + `set_difference`/`set_intersection`
over pointers -- verified equivalent, not merely assumed, at every call
site that consumes one of these results:

* `remove_dominated`'s `dif` and `merge_nondominated`'s `diff_bcv_mp` are
  both immediately consumed by a `for` loop that only calls `.erase(...)`
  on a set -- erasing a fixed set of elements from a set has the same
  final set regardless of the ORDER they are erased in.
* `merge_nondominated`'s final `for (cnd : bcv_p.first) _scored_trees.
  insert(...)` inserts into `scored_combo_tree_ptr_set`, a SORTED
  container (`_sct_comparator_set`, types.py) that places every element by
  its own comparator regardless of insertion order.

So nothing here needs to reproduce literal address-order, only IDENTITY-based
set membership -- exactly what `id()`-keyed filtering gives.

## `jobs` (multi-threading) is accepted but not implemented

Every function below accepts `jobs` (default `1`, matching
`metapop_parameters.jobs`'s own default, metapop_params.py) for signature
fidelity, but always takes the SINGLE-THREADED control-flow path
(`get_nondominated_rec`/`get_nondominated_disjoint_rec`'s own `else`
branches) regardless of its value -- the same "no threading anywhere in
this port" convention `metapopulation.py`'s own module docstring states
for `_merge_mutex`. This is safe, not merely convenient: the two branches
are two DIFFERENT algorithms (the multi-threaded one filters both halves
of `bcv1` against the FULL `bcv2` in parallel then intersects the two
surviving-`bcv2` sets; the single-threaded one filters sequentially,
threading the shrinking `bcv2` through), but they are only interchangeable
GIVEN domination.cc's own stated precondition that `bcv1` (and `bcv2`)
individually contain no internally-dominated elements ("assuming that bcs
contains no dominated candidates within itself", domination.cc:284) --
without that precondition, an element of `bcv1`'s first half that is
dominated by (and hence filtered out alongside) some element of `bcv2`
could, in principle, dominate an element of `bcv1`'s second half that the
sequential path would then fail to filter against that removed `bcv2`
element. Every real caller (`remove_dominated`, `merge_nondominated`) only
ever calls this on trees that are already each other's non-dominated
peers going in, so this port always takes the sequential path, which is
this file's OWN oracle-tested path (the `metapopulationUTest.cxxtest`
cases call these methods with the default `jobs=1`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.metapopulation import _copy_sct, metapopulation
from pymetta.moses.moses.types import behavioral_score, scored_combo_tree, scored_combo_tree_set

if TYPE_CHECKING:
    pass

__all__: list[str] = []
# Nothing new is exported: every symbol below is bound onto `metapopulation`
# itself (see module docstring's Composition section), matching merging.py's
# own convention.


def dominates(x: behavioral_score, y: behavioral_score) -> bool | None:
    """C++: moses/moses/metapopulation/domination.cc:70 metapopulation::dominates

    "x dominates y if for all i x_i >= y_i and there exists i such that
    x_i > y_i" (domination.cc:62-64's own doc comment). Returns `True` if
    x dominates y, `False` if y dominates x, `None` (indeterminate)
    otherwise -- see module docstring's tribool section.

    "everything dominates an empty vector" (domination.cc:73's own
    comment): a non-empty `x` against an empty `y` returns `True`
    (unconditionally, without even looking at `x`'s contents) and an
    empty `x` against a non-empty `y` returns `False` -- both empty
    returns indeterminate. These three special cases are NOT redundant
    with the general per-element loop below: an empty `x` makes that loop
    a no-op (nothing to iterate), which would fall through to returning
    the initial indeterminate `res` for ALL THREE empty combinations,
    silently losing the "non-empty dominates empty" cases the C++ function
    special-cases explicitly. Ported as the same explicit special case.

    The general loop walks `x`'s length and assumes `y` is at least as
    long (`for (xit = x.begin(), yit = y.begin(); xit != x.end(); ++xit,
    ++yit)` -- if `y` were shorter, `yit` would run past `y.end()`, which
    is undefined behaviour in C++). Every real caller's `x`/`y` are
    `behavioral_score`s produced by the SAME `behave_cscore` over the SAME
    dataset, so they are always equal length in practice; this port uses
    `zip(x, y)`, which is well-defined for a length mismatch (silently
    stops at the shorter one) rather than reproducing C++'s UB for an
    input shape no real caller produces.

    The short-circuit `return None` the instant a contradiction is seen
    (a `>` in one direction after already committing to the other) is a
    pure optimisation in C++ -- once both directions have been seen, no
    further element can change the answer away from indeterminate, so
    scanning the rest of the vector first would reach the exact same
    final `res`. Ported with the same short-circuit for the same reason
    (there would be no OBSERVABLE difference either way, but this is the
    literal translation).
    """
    if not x:
        if not y:
            return None
        return False
    elif not y:
        return True
    res: bool | None = None
    for xi, yi in zip(x, y, strict=False):
        if xi > yi:
            if res is False:
                return None
            res = True
        elif yi > xi:
            if res is True:
                return None
            res = False
    return res


def to_set(bcv: list[scored_combo_tree]) -> scored_combo_tree_set:
    """C++: moses/moses/metapopulation/domination.cc:202 metapopulation::to_set

    "reciprocal of random_access_view" (domination.cc:200's own comment).
    `res.insert(*cnd)` copy-constructs each element into the result set
    (C++ `std::unordered_set<...>::insert(const value_type&)`) -- ported
    via `_copy_sct` (the same helper `merging.py`/`metapopulation.py`
    already use at every other C++ by-value-container-insert site, not
    reimplemented here).
    """
    res = scored_combo_tree_set()
    for cnd in bcv:
        res.insert(_copy_sct(cnd))
    return res


def split(
    bcv: list[scored_combo_tree],
) -> tuple[list[scored_combo_tree], list[scored_combo_tree]]:
    """C++: moses/moses/metapopulation/metapopulation.h:408 metapopulation::split (static, inline)

    "split in 2 of equal size" (metapopulation.h:407's own comment):
    `middle = bcv.begin() + bcv.size() / 2` -- integer division floors, so
    an odd-length input gives the FIRST half the smaller share (matching
    Python's `//`, no narrowing difference to reproduce).
    """
    mid = len(bcv) // 2
    return list(bcv[:mid]), list(bcv[mid:])


def get_nondominated_iter(bcs: scored_combo_tree_set) -> scored_combo_tree_set:
    """C++: moses/moses/metapopulation/domination.cc:115 get_nondominated_iter

    (`metapopulation::get_nondominated_iter`, `static` in C++ -- dropped
    the class qualifier here only to fit the line-length limit, same as
    `recompute_scores_over_whole_dataset`'s own breadcrumb in merging.py.)

    ZERO callers anywhere in classic -- confirmed by grepping the whole
    `moses/moses/metapopulation/` tree (`merging.cc`/`metapopulation.cc`
    call only `remove_dominated`/`merge_nondominated`, both of which use
    `get_nondominated_rec`, never this method) -- and no oracle test
    touches it either. Ported anyway (domination.cc is a 309-line file
    this task ports in full), verbatim quirk included, per "port classic's
    quirks verbatim" rather than skipping a method just because it is
    unreachable.

    # DO NOT FIX: C++ walks a `std::list` with a nested loop that erases
    # elements in place (domination.cc:121-139). When `it1` itself gets
    # erased (the `else if (!dom)` branch, because `it2` dominates it1),
    # C++ does TWO things: `it1 = mcl.erase(it1)` (which ALREADY advances
    # `it1` to the element following the erased one -- `std::list::erase`
    # returns the next iterator), and THEN, because that same branch also
    # set `it2 = mcl.end()`, the loop's trailing `if (it2 == mcl.end())
    # ++it1;` guard fires and advances `it1` A SECOND TIME. The element
    # `erase` left `it1` pointing to is therefore skipped entirely: it is
    # never compared again as `it1` (the outer loop has already moved past
    # it), and it can never be reached as some LATER `it1`'s `it2` either
    # (`it2` only ever starts at `it1 + 1` and walks forward). A skipped
    # element that some other element genuinely dominates therefore
    # SURVIVES into the result, undetected.
    #
    # Verified empirically with a standalone reproduction of this exact
    # control flow (not this port -- a throwaway C++ probe using
    # `std::list<int>`/a hand-written `dominates` on plain vectors, kept
    # out of the repo): three items A=[1,-1], B=[-1,1], C=[10,10], with
    # `dominates(A,B)` indeterminate, `dominates(A,C)` false (C dominates
    # A) and -- never checked -- `dominates(B,C)` would ALSO be false (C
    # dominates B too). Real classic control flow: it1=A/it2=B
    # (indeterminate, `++it2`), it1=A/it2=C (C dominates A, erase it1=A;
    # it1 now "points to" B; forced `++it1` skips B, landing on C), then
    # C has nothing left to compare (single remaining element). Final
    # result: {B, C} -- B survives even though C actually dominates it.
    # Reproduced exactly below via index bookkeeping that mirrors
    # `std::list::erase`'s "returns the next iterator" contract: `del
    # mcl[i]` leaves `i` pointing at the element that follows (the erased
    # slot's successor shifts down into index `i`), so the trailing `i +=
    # 1` is the exact same double-advance as C++'s `++it1`.
    #
    # `tests/moses/metapopulation/test_domination.py`'s
    # `test_get_nondominated_iter_skips_element_after_erasing_it1`
    # reproduces this A/B/C case directly and asserts B survives, proving
    # this is the quirk being ported rather than a naive skyline.

    C++ copies `bcs` into a `std::list<scored_combo_tree>` BY VALUE
    (`mcl(bcs.begin(), bcs.end())`) -- reproduced as a plain `list(bcs)`
    without `_copy_sct`: unlike every OTHER copy site this port protects
    with `_copy_sct` (guarding against a LATER in-place mutation such as
    `rescore()`'s `set_score`), this function never mutates an element's
    fields, only the container's membership, so there is nothing for a
    missing copy to corrupt.
    """
    mcl: list[scored_combo_tree] = list(bcs)
    i = 0
    while i < len(mcl):
        j = i + 1
        if j < len(mcl):
            while j < len(mcl):
                dom = dominates(mcl[i].get_bscore(), mcl[j].get_bscore())
                if dom is True:
                    del mcl[j]
                elif dom is False:
                    del mcl[i]
                    j = len(mcl)
                else:
                    j += 1
                if j == len(mcl):
                    i += 1
        else:
            i += 1
    return scored_combo_tree_set(mcl)


def get_nondominated_disjoint_rec(
    self: metapopulation,
    bcv1: list[scored_combo_tree],
    bcv2: list[scored_combo_tree],
    jobs: int = 1,
) -> tuple[list[scored_combo_tree], list[scored_combo_tree]]:
    """C++: moses/moses/metapopulation/domination.cc:223 get_nondominated_disjoint_rec

    (`metapopulation::get_nondominated_disjoint_rec` -- class qualifier
    dropped only to fit the line-length limit, same convention as
    `get_nondominated_iter`'s own breadcrumb above.)

    Only the single-threaded (`jobs <= 1`) control-flow path is ported --
    see module docstring's `jobs` section for why that is safe.

    Base cases (domination.cc:227-249): either input empty returns both
    inputs unchanged (identity, no filtering possible); `len(bcv1) == 1`
    scans `bcv2` once, dropping `it2` whenever `it1` dominates it
    (`dom is True`, no action -- simply not appended to `res2`), keeping
    `it2` when indeterminate, and on the FIRST `it2` that dominates `it1`
    (`dom is False`) stopping immediately: `it1` is excluded from `res1`
    and the remainder of `bcv2` from that point on (`it2` onward,
    UNEXAMINED) is appended to `res2` wholesale (domination.cc:242,
    `bcv_res2.insert(bcv_res2.end(), it2, bcv2.end())`) rather than
    continuing to check each one -- once ANYTHING dominates `it1`, `it1`
    cannot dominate any of the remaining `bcv2` elements either (if it
    could, `it1` would have to simultaneously be dominated by, and
    dominate, elements that domination.cc's own precondition -- `bcv2`
    itself is not internally dominated -- rules out contradicting), so
    every remaining `bcv2` element is trivially a survivor.
    """
    if not bcv1 or not bcv2:
        return list(bcv1), list(bcv2)
    if len(bcv1) == 1:
        it1 = bcv1[0]
        res1: list[scored_combo_tree] = []
        res2: list[scored_combo_tree] = []
        it1_insert = True
        for idx, it2 in enumerate(bcv2):
            dom = dominates(it1.get_bscore(), it2.get_bscore())
            if dom is False:
                it1_insert = False
                res2.extend(bcv2[idx:])
                break
            elif dom is None:
                res2.append(it2)
            # dom is True: it2 is dominated by it1 -- dropped.
        if it1_insert:
            res1.append(it1)
        return res1, res2

    bcv1_first, bcv1_second = split(bcv1)
    bcv_m1 = get_nondominated_disjoint_rec(self, bcv1_first, bcv2)
    bcv_m2 = get_nondominated_disjoint_rec(self, bcv1_second, bcv_m1[1])
    res1 = list(bcv_m1[0])
    res1.extend(bcv_m2[0])
    return res1, bcv_m2[1]


def get_nondominated_rec(
    self: metapopulation,
    bcv: list[scored_combo_tree],
    jobs: int = 1,
) -> list[scored_combo_tree]:
    """C++: moses/moses/metapopulation/domination.cc:154 metapopulation::get_nondominated_rec

    Only the single-threaded path is ported (module docstring's `jobs`
    section). Base case: fewer than 2 elements is trivially non-dominated
    (domination.cc:160-162).
    """
    if len(bcv) < 2:
        return list(bcv)
    bcv1, bcv2 = split(bcv)
    bcv1_nd = get_nondominated_rec(self, bcv1)
    bcv2_nd = get_nondominated_rec(self, bcv2)
    res = get_nondominated_disjoint_rec(self, bcv1_nd, bcv2_nd)
    result = list(res[0])
    result.extend(res[1])
    return result


def get_nondominated_disjoint(
    self: metapopulation,
    bcs1: scored_combo_tree_set,
    bcs2: scored_combo_tree_set,
    jobs: int = 1,
) -> tuple[scored_combo_tree_set, scored_combo_tree_set]:
    """C++: moses/moses/metapopulation/domination.cc:211 metapopulation::get_nondominated_disjoint

    "return a pair of sets of nondominated candidates between bcs1 and
    bcs2, assuming none contain dominated candidates" (metapopulation.h:
    421-426's own comment). Oracle-tested directly:
    `metapopulationUTest.cxxtest:119`'s `test_merge_nondominated_rec`
    calls this exact method (`_metapopulation.get_nondominated_disjoint(
    bcs1, bcs2)`, default `jobs=1`).

    `random_access_view(bcsN)` (domination.cc:216-217) is ported as
    `list(bcsN)` -- both `scored_combo_tree_set.__iter__` (this port) and
    `std::unordered_set::begin()/end()` (C++) walk every element exactly
    once; module docstring's identity-based-set-ops section covers why no
    copy is needed for the elements THEMSELVES at this stage (only
    `to_set`, at the very end, copies).
    """
    res1, res2 = get_nondominated_disjoint_rec(self, list(bcs1), list(bcs2), jobs)
    return to_set(res1), to_set(res2)


def remove_dominated(
    self: metapopulation,
    bcs: scored_combo_tree_set,
    jobs: int = 1,
) -> None:
    """C++: moses/moses/metapopulation/domination.cc:101 metapopulation::remove_dominated

    Mutates `bcs` IN PLACE (`scored_combo_tree_set &bcs`, a C++ reference
    -- Python's own by-reference object semantics reproduce this without
    any special handling). Oracle-tested directly:
    `metapopulationUTest.cxxtest:74`'s `test_remove_dominated` calls this
    exact method with the module's own 5-tree fixture and asserts the
    2 dominated trees (`bct2`, `bct5`) are gone and 3 remain.

    `boost::sort(bcv); boost::sort(res); set_difference(bcv, res)`
    (domination.cc:107-108) is ported as `id()`-keyed set membership --
    see module docstring's identity-based-set-ops section for why this is
    equivalent, not merely convenient, at this exact call site (the
    result only ever feeds a `for ... bcs.erase(...)` loop).
    """
    bcv = list(bcs)
    res = get_nondominated_rec(self, bcv, jobs)
    res_ids = {id(cnd) for cnd in res}
    dif = [cnd for cnd in bcv if id(cnd) not in res_ids]
    for cnd in dif:
        bcs.erase(cnd)


def merge_nondominated(
    self: metapopulation,
    bcs: scored_combo_tree_set,
    jobs: int = 1,
) -> None:
    """C++: moses/moses/metapopulation/domination.cc:285 metapopulation::merge_nondominated

    "merge nondominated candidate to the metapopulation assuming that bcs
    contains no dominated candidates within itself" (domination.cc:283-284's
    own comment). Called from `merging.py`'s `merge_candidates`, gated on
    `_params.discard_dominated` (that call site's own docstring).

    `for (const scored_combo_tree& cnd : *this) bcv_mp.push_back(&cnd);`
    (domination.cc:289-290) walks the metapopulation itself, which
    delegates to `_scored_trees` (metapopulation.h's `begin()`/`end()`) --
    ported as `list(self._scored_trees)` directly, matching every other
    site in this file group that reads `_scored_trees` without going
    through a public accessor (e.g. `merging.py`'s own `merge_candidates`).

    `boost::sort(bcv_mp); boost::sort(bcv_p.second); set_difference(...)`
    (domination.cc:295-298) -- same `id()`-keyed equivalent as
    `remove_dominated` (module docstring), and again only ever consumed by
    a `for ... _scored_trees.erase(...)` loop, order-independent.

    `_scored_trees.insert(new scored_combo_tree(*cnd))` (domination.cc:304)
    -- ported via `_copy_sct`, the same helper used at every other C++
    by-value-container-insert site in this file group.
    """
    bcv = list(bcs)
    bcv_mp = list(self._scored_trees)
    res1, res2 = get_nondominated_disjoint_rec(self, bcv, bcv_mp, jobs)

    survivor_ids = {id(cnd) for cnd in res2}
    diff_bcv_mp = [cnd for cnd in bcv_mp if id(cnd) not in survivor_ids]
    for cnd in diff_bcv_mp:
        self._scored_trees.erase(cnd)

    for cnd in res1:
        self._scored_trees.insert(_copy_sct(cnd))


metapopulation.dominates = staticmethod(dominates)
metapopulation.to_set = staticmethod(to_set)
metapopulation.split = staticmethod(split)
metapopulation.get_nondominated_iter = staticmethod(get_nondominated_iter)
metapopulation.get_nondominated_disjoint_rec = get_nondominated_disjoint_rec
metapopulation.get_nondominated_rec = get_nondominated_rec
metapopulation.get_nondominated_disjoint = get_nondominated_disjoint
metapopulation.remove_dominated = remove_dominated
metapopulation.merge_nondominated = merge_nondominated
