"""C++: moses/comboreduct/reduct/logical_rules.{h,cc}

The boolean-logic reduction rules, culminating in `subtree_to_enf`
(reduction to ENF -- elegant normal form). The largest and highest-risk
file in the S7 slice (860 C++ lines).

STRUCTURAL DECISION (this task's own brief, and worth repeating here since
it governs this whole file's shape): `subtree_to_enf.reduce_to_enf` is
ported as ONE nested class mirroring `logical_rules.h:179-270`'s C++
struct member-for-member -- same method names (`operator()` ->
`__call__`), same call graph (`reduce` dispatches to `reduce_and`/
`reduce_or`, which recurse back into `reduce` for their own children),
same nesting of `Result`/`tree_eraser`/`subtree_set_eraser`/
`tree_inserter`. NOT decomposed into separate rule objects or files: the
`metta-moses` port split exactly this function into 12 files and produced
two documented bugs (issues #502/#503) by losing track of the *order* and
*scope* of operations. `MOSES/elegant-normal-forms-python/Utilities/
ReduceToElegance.py` (docs/CARRY-FORWARD.md Β§1.5) was read as a sequencing
reading aid (its 0-subsume/1-subsume/AND-cut/OR-cut/fixpoint order is
faithful) but not imported from -- its data structure (a Python-native
tree, not this port's `tree_node` graph) is unusable here.

Three things earlier tasks flagged as landing on this file, all addressed
below with a dedicated regression test each:

1. `logical_rules.cc:204` (`reduce_nots`' `is_argument` branch: `*it =
   *it.begin(); tr.erase_children(it); get_argument(*it).negate();`) is
   the only in-place vertex mutator in the language. Task 3's root fix
   (`tree.py`'s `_copy_value`, applied at every subtree-copy site) means
   the argument object `it.data` ends up referencing, post-assignment, is
   never shared with anything else live in the tree -- see
   `test_reduce_nots_in_place_negate_does_not_corrupt_a_sibling_occurrence`.
2. Task 3's same root fix also repaired `meta_rules.iterative`'s fixpoint
   snapshot (`combo_tree tmp = combo_tree(it)`): pre-fix, a rule negating
   an argument in place made the fixpoint's own SNAPSHOT report a
   corrupted (already-mutated) value instead of the pre-mutation state it
   is supposed to capture. This is the first real in-place-negating rule
   to exist, so it is also the first to exercise that snapshot path for
   real -- see `test_combo_tree_snapshot_survives_reduce_nots_in_place_
   negate` (that test's own docstring also records a measured, disclosed
   finding: `iterative`'s own CALL COUNT does not itself distinguish
   fixed from broken for this specific rule, because `reduce_nots`'
   `is_argument` branch always changes `it`'s own vertex TYPE, which
   `equal_subtree_vertex_safe`'s type check independently catches
   regardless of leaf aliasing -- the snapshot corruption is real and
   measured directly, just not observable through that one angle).
3. `push_back_negated_arguments` (logical_rules.h:230, reached from
   `reduce_to_enf.reduce_and` at logical_rules.cc:643) does `combo_tree
   (sib)` then negates the COPY's root. Pre-fix this silently mutated the
   source tree too -- see
   `test_push_back_negated_arguments_does_not_mutate_the_source`.

SCOPING (UPDATED by Task 5 -- see below for what changed): one rule class
this file declares still cannot be fully ported, because its own C++ body
calls into a subsystem that is genuinely, permanently out of this
project's scope; the OTHER was fully unblocked once Task 5 landed its own
two dependencies:

* `simplify_predicates` -- its early-return guard (is this even a
  predicate? is its argument already a bare argument?) is fully ported
  and is this rule's real, exercised behaviour for every tree in this
  file's own boolean-only test corpus. Only the TAIL --
  `contin_reduce`/`mixed_reduce` -- is stubbed with `NotImplementedError`
  (a `# DIVERGENCE` marks the exact line): both are PLANNED: S14 per
  docs/MODULE-COVERAGE.md, permanently out of this project's current
  scope, not merely "not yet written". Mirrors meta_rules.py's own
  (now-resolved) `_get_output_type_tree` precedent for a genuinely-blocked
  tail.
* `reduce_remove_subtree_equal_tt` -- FULLY PORTED as of Task 5, which
  landed both forward references this rule needed: `complete_truth_table`
  (`comboreduct/table/table.py`) and `clean_reduce` (`comboreduct/reduct/
  clean_reduction.py`). See its own `apply` docstring for the full
  algorithm, including a real, MEASURED C++ quirk (the walk is not
  actually bounded to `it`'s own subtree) and why `erase_pre_order`
  (`pymetta/util/tree.py`), not plain `erase`, is the correct primitive
  for its `pre_it` walk.

TESTING SCOPE, decided and disclosed rather than silently attempted: this
task's brief asks for a transliteration of `BooleanReductUTest.cxxtest`
and `BooleanFullUTest.cxxtest`, preserving test names. Read in full
($CLASSIC/tests/comboreduct/reduct/):

* `BooleanReductUTest` (6 tests, `test_simplify1`..`test_simplify6`) --
  MOVED to `tests/comboreduct/reduct/test_logical_reduction.py` once Task
  5 landed `logical_reduction.py` (the file this test genuinely needs:
  it calls `logical_reduce(3, tr, ignore_ops)`, effort 3, "complexe",
  which lives there, not here). Ported for REAL there now -- byte-
  identical against the real compiled `libcomboreduct.so` for all 6
  trees at BOTH effort 2 and effort 3 (`tools/cpp_probe/logical_rules_
  probe.cc`'s `boolean_reduct_medium_typed_N`/`boolean_reduct_effort3_
  typed_N`) -- not the untyped, effort-2-only approximation an earlier
  version of this section had to settle for while blocked. See that
  module's own docstring for the measured typed-vs-untyped divergence
  this file's own `insert_ands`/`reduce_nots`/etc. typed dispatch turned
  out to cause (confirmed, not merely hypothesised, once typed `downwards`
  existed to test against).
* `BooleanFullUTest` (`test_simplify1`..`test_simplify9`, `test_simplifya`..
  `test_simplifyd`, `test_simplify14`..`test_simplify20` -- **20** test
  methods total, not a contiguous `1..20`: FIX WAVE, Task 4 review Minor,
  an earlier version of this note miscounted as 24) is, despite its name,
  NOT a pure-boolean corpus: every one of its 20 trees contains `0<(...)`
  (`greater_than_zero`) predicates over `contin_t` arithmetic (`+`, `*`,
  `/`), and its own `base_test` helper calls `infer_type_tree`/
  `is_well_formed` (`type_checker`, not ported -- S8, out of S7 scope
  entirely) plus `ITable`/`OTable` and `logical_reduction`'s full effort
  0-3 sweep including predicate simplification (`simplify_predicates`,
  stubbed above -- S14); 6 of the 20 (`test_simplify15`..`test_simplify20`)
  additionally use `cond` with `definite_object` consequents
  (`POX`/`CYT`/`NUC`/`ME2` string tokens),
  a `vertex` alternative this port's four-way union does not carry
  (vertex.py's own module docstring). **0 of 20 are reachable today.**
  `logical_reduction.py` (Task 5) now exists, but this still needs type
  inference (S8) AND contin/mixed reduction (S14, permanently out of this
  project's current horizon per docs/MODULE-COVERAGE.md) simultaneously --
  landing one of the three original blockers does not unblock the other
  two. NOT PORTED -- disclosed here and in task-4-report.md's Concerns
  section, rather than faked with a hollow substitute the way
  `BooleanReductUTest` legitimately admits one. Whoever eventually
  reaches S14 is the right owner to revisit this.

Every non-obvious algorithmic choice in `reduce_to_enf` was checked
against a bounded probe run against the real compiled `libcomboreduct.so`
(`tools/cpp_probe/logical_rules_probe.cc`,
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/logical_rules_probe
)`) -- see task-4-report.md for the transcript.
"""

from __future__ import annotations

import functools
from enum import Enum, auto

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.assumption import copy_all_assumptions, insert_assumption
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    is_argument,
    lexicographic_subtree_less,
    vertex,
    vertex_equal,
)
from pymetta.comboreduct.reduct.flat_normal_form import (
    clause,
    implications,
    nf,
    nf_mapper,
    number_of_literals,
    pairwise_erase_if,
    subset,
    subset_eq,
    tautology,
)
from pymetta.comboreduct.reduct.reduct import rule
from pymetta.comboreduct.table.table import complete_truth_table
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# vertex.h predicates whose only real call site (within S7's scope) is this
# file -- same narrow-scoping convention `general_rules.py` established for
# `is_associative`/`is_commutative`/etc. (that module's own docstring).
# ---------------------------------------------------------------------------


def is_logical_operator(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:741 is_logical_operator"""
    return (
        vertex_equal(v, builtin.logical_and)
        or vertex_equal(v, builtin.logical_or)
        or vertex_equal(v, builtin.logical_not)
    )


def is_predicate(node: tree_node) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:781 is_predicate

    Takes a node (C++: `const combo_tree::iterator&`), not a bare vertex --
    the `logical_not` case must inspect the node's own child.

    UPSTREAM UB, ported faithfully (Task 4 review Minor): if `node` is a
    CHILDLESS `logical_not` (a malformed `not` with no operand -- real
    combo trees never construct one, but nothing here rejects one),
    `node.first_child.data` raises `AttributeError` (`node.first_child`
    is `None`). C++'s own `*it.begin()` on the same shape is undefined
    behaviour (vertex.h:781-786's own comment gives no guard either) --
    this port's failure mode (a loud, catchable exception) is safer, but
    not something this file corrects.

    Three real call sites in this file CAN reach a childless `logical_not`
    this way (confirmed by construction, not just by inspection --
    `test_is_predicate_raises_for_a_childless_logical_not`,
    `test_enf_crashes_on_a_childless_not_via_reduce_and_res_computation`,
    and (Task 5) `test_reduce_remove_subtree_equal_tt_raises_on_a_
    childless_not_via_its_own_predicate_scan` below):

    * `reduce_to_enf.reduce_and`'s own `if not is_predicate(child.
      last_child):` (the `res`-computation setup) -- reached whenever an
      AND/OR `current` has a direct AND/OR child whose OWN single child
      is a childless `not`.
    * `reduce_to_enf.or_cut`'s `is_argument(child.first_child.data) or
      is_predicate(child.first_child)` -- reached the same way, one level
      up, for an AND/OR `current` child that itself has exactly one
      child (any operator except `logical_or`, which is excluded by the
      preceding `if`).
    * (Task 5) `reduce_remove_subtree_equal_tt`'s own guard scan (`for
      pit in tr: if is_predicate(pit): return`, logical_rules.cc:838-841)
      walks EVERY node in the tree, so a childless `logical_not`
      ANYWHERE in `tr` reaches this, not just at a specific structural
      position the way the two call sites above require.

    `subtree_to_enf.reduce_to_enf.and_cut` is the one place in this file
    that CANNOT reach this: it checks `is_logical_operator(gchild.data)`
    FIRST and only calls `is_predicate` in the sibling `elif` branch,
    which a `logical_not` (itself a logical operator) can never reach.
    """
    if vertex_equal(node.data, builtin.greater_than_zero):
        return True
    if vertex_equal(node.data, builtin.logical_not) and vertex_equal(
        node.first_child.data, builtin.greater_than_zero
    ):
        return True
    return False


def is_complement(x: vertex, y: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:722 is_complement"""
    if isinstance(x, argument) and isinstance(y, argument):
        return x.idx == -y.idx
    return False


def _sort_by_comp(items: list, key_fn, comp) -> None:
    """No direct C++ symbol: `std::sort(negated.begin(), negated.end(),
    comp)` (logical_rules.cc:644) sorts a `vector<combo_tree>` directly
    using `Comp` -- `lexicographic_subtree_order`'s own four overloads
    (opencog/util/tree.h:3257-3278) accept a whole `tree<T>` by implicitly
    taking its `.begin()`, but this port's promoted `comp`
    (`lexicographic_subtree_less`) takes only `tree_node`s -- `key_fn`
    extracts each item's comparable root node before comparing, matching
    `tree.py`'s own `sort_on_subtrees`/`functools.cmp_to_key` idiom.
    """

    def _cmp(a, b):
        ka, kb = key_fn(a), key_fn(b)
        if comp(ka, kb):
            return -1
        if comp(kb, ka):
            return 1
        return 0

    items.sort(key=functools.cmp_to_key(_cmp))


# ---------------------------------------------------------------------------
# opencog/util/algorithm.h merge-style set algorithms `reduce_to_enf` needs.
#
# # DIVERGENCE(C++ opencog/util/algorithm.h, whole section below): counting_iterator collapsed away.
# # C++ wraps a `sib_it` in `boost::counting_iterator` purely so a live tree position can be handed
# # to a generic merge algorithm expecting a forward iterator whose `operator*` yields a comparable
# # value -- since `Comp` (`lexicographic_subtree_order`) already compares `sib_it` values directly,
# # `*counting_iterator(sib_it)` and the wrapped `sib_it` itself are the exact same value; the
# # wrapper adds no behaviour of its own (the same collapse `flat_normal_form.py`'s `implications`
# # DIVERGENCE marker documents for output iterators). This port passes plain Python lists of
# # `tree_node` (this port's stand-in for `sib_it`) directly, with no wrapper. Every real call site
# # takes a FRESH snapshot list of a live sibling range immediately before calling one of these --
# # behaviourally equivalent to a live counting_iterator walk here, because each algorithm below
# # only ever reads forward and erases/inserts at or before its current scan position: an erasure
# # can only remove an element the scan has already passed or is standing on (never one still ahead
# # relative to the OTHER range's pointer), and an insertion via `tree_inserter` either lands before
# # the destination range's CURRENT element (leaving that element's own future traversal, and hence
# # the snapshot's remaining entries, unaffected) or strictly past the snapshotted range's end (an
# # append, which the snapshot -- bounded at the range's original length -- was never going to walk
# # into anyway).
# ---------------------------------------------------------------------------


def is_sorted(nodes: list, comp) -> bool:
    """C++: std::is_sorted(first, last, comp)"""
    return all(not comp(nodes[i + 1], nodes[i]) for i in range(len(nodes) - 1))


def erase_set_intersection(erase, seq1: list, seq2: list, comp) -> None:
    """C++: opencog/util/algorithm.h:60 erase_set_intersection

    Erases (via `erase(node)`) every element of `seq1` that also appears
    in `seq2` (both already sorted per `comp`), leaving the difference.
    """
    i1, n1 = 0, len(seq1)
    i2, n2 = 0, len(seq2)
    while i1 != n1 and i2 != n2:
        a, b = seq1[i1], seq2[i2]
        if comp(a, b):
            i1 += 1
        elif comp(b, a):
            i2 += 1
        else:
            erase(a)
            i1 += 1
            i2 += 1


def erase_set_difference(erase, seq1: list, seq2: list, comp) -> None:
    """C++: opencog/util/algorithm.h:83 erase_set_difference

    Erases every element of `seq1` NOT in `seq2`, leaving the
    intersection.
    """
    i1, n1 = 0, len(seq1)
    i2, n2 = 0, len(seq2)
    while i1 != n1 and i2 != n2:
        a, b = seq1[i1], seq2[i2]
        if comp(a, b):
            erase(a)
            i1 += 1
        elif comp(b, a):
            i2 += 1
        else:
            i1 += 1
            i2 += 1
    while i1 != n1:
        erase(seq1[i1])
        i1 += 1


def insert_set_complement(insert, seq1: list, seq2: list, comp) -> None:
    """C++: opencog/util/algorithm.h:109 insert_set_complement

    `insert(pos, value)`: `pos` is the element of `seq1` to insert
    `value`'s subtree BEFORE, or `None` once `seq1` is exhausted (append
    at the end) -- see `subtree_to_enf.reduce_to_enf.tree_inserter`, the
    one real caller.
    """
    i1, n1 = 0, len(seq1)
    i2, n2 = 0, len(seq2)
    while i1 != n1 and i2 != n2:
        a, b = seq1[i1], seq2[i2]
        if comp(a, b):
            i1 += 1
        elif comp(b, a):
            insert(a, b)
            i2 += 1
        else:
            i1 += 1
            i2 += 1
    while i2 != n2:
        insert(seq1[i1] if i1 < n1 else None, seq2[i2])
        i2 += 1


def has_empty_intersection(seq1: list, seq2: list, comp) -> bool:
    """C++: opencog/util/algorithm.h:136 has_empty_intersection"""
    i1, n1 = 0, len(seq1)
    i2, n2 = 0, len(seq2)
    while i1 != n1 and i2 != n2:
        a, b = seq1[i1], seq2[i2]
        if comp(a, b):
            i1 += 1
        elif comp(b, a):
            i2 += 1
        else:
            return False
    return True


class subtree_set:
    """C++: moses/comboreduct/reduct/logical_rules.h:186 reduce_to_enf::subtree_set

    Promoted to MODULE level (not nested inside `subtree_to_enf.reduce_to_enf`
    as the C++ typedef is) purely for readability -- it carries no state of
    its own tied to a particular reduction run.

    `typedef std::set<sib_it, opencog::lexicographic_subtree_order<vertex>>
    subtree_set` -- an ordered, DEDUPLICATED set of `tree_node`s, ordered
    STRUCTURALLY (by subtree content, via the promoted
    `lexicographic_subtree_less`), not by node identity or tree position.
    Two DIFFERENT nodes holding structurally EQUAL subtrees are the same
    element to this set -- inserting a second one is a no-op, matching
    `std::set`'s own equivalence-class semantics under a `Compare` that
    need not be `==`.

    Implemented as a sorted Python list plus binary search, the same
    approach `flat_normal_form.py`'s `clause` uses for its own
    `std::set<int, absolute_value_order<int>>`, since this port's `tree`
    has no built-in balanced-tree container to reuse.
    """

    __slots__ = ("_items", "_comp")

    def __init__(self, comp, items=()) -> None:
        self._comp = comp
        self._items: list[tree_node] = []
        for x in items:
            self.insert(x)

    def _bisect(self, x: tree_node):
        lo, hi = 0, len(self._items)
        comp = self._comp
        while lo < hi:
            mid = (lo + hi) // 2
            v = self._items[mid]
            if comp(v, x):
                lo = mid + 1
            elif comp(x, v):
                hi = mid
            else:
                return mid, True
        return lo, False

    def insert(self, node: tree_node) -> None:
        """C++: std::set<sib_it, Comp>::insert"""
        idx, found = self._bisect(node)
        if not found:
            self._items.insert(idx, node)

    def erase(self, node: tree_node) -> None:
        """C++: std::set<sib_it, Comp>::erase(iterator)

        A no-op if no structurally-equal element is present, matching
        `std::set::erase`'s own permissive contract for a missing key --
        every real call site here erases something a preceding `find`
        just located, but this stays safe regardless.
        """
        idx, found = self._bisect(node)
        if found:
            del self._items[idx]

    def find(self, node: tree_node) -> tree_node | None:
        """C++: std::set<sib_it, Comp>::find

        Returns the ACTUAL stored element structurally equal to `node`
        (which may be a DIFFERENT `tree_node` object than `node` itself --
        e.g. a `dominant` element from elsewhere in the tree that happens
        to look the same), or `None` if there is none.
        """
        idx, found = self._bisect(node)
        return self._items[idx] if found else None

    def __contains__(self, node: tree_node) -> bool:
        return self._bisect(node)[1]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        """C++: std::set::empty"""
        return not self._items

    def copy(self) -> subtree_set:
        """C++: std::set's own copy constructor (`subtree_set res(other);`
        / `subtree_set child_command(command);`)"""
        new = subtree_set(self._comp)
        new._items = list(self._items)
        return new


# ---------------------------------------------------------------------------
# The rules -- C++ declaration order (logical_rules.h).
# ---------------------------------------------------------------------------


class insert_ands(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:37 insert_ands

    Ensure that all arguments and 'or' nodes have an 'and' node as their
    parent. This is important so that other normalizations can catch all
    cases.
    """

    def __init__(self, name: str = "insert_ands") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:45 insert_ands::operator()

        `tr.prepend_child(it, it.data)` passes `it`'s OWN current value
        (possibly a mutable `argument`) straight through with no defensive
        copy -- safe here without one: `it.data` is REBOUND (not mutated
        in place) to `builtin.logical_and` on the very next line, so the
        new child ends up holding the same value `it` used to, and nothing
        after this ever mutates that value through `it`'s own new
        identity.
        """
        parent = it.parent
        if (is_argument(it.data) or vertex_equal(it.data, builtin.logical_or)) and (
            parent is None
            or vertex_equal(parent.data, builtin.logical_or)
            or vertex_equal(parent.data, builtin.impulse)
            or vertex_equal(parent.data, builtin.logical_not)
        ):
            tr.prepend_child(it, it.data)
            it.data = builtin.logical_and
            new_first = it.first_child
            tr.reparent(new_first, new_first.next_sibling, None)


class remove_unary_junctors(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:45 remove_unary_junctors

    Remove any logical operators that have only one child (i.e. raise up
    the child in the tree).
    """

    def __init__(self, name: str = "remove_unary_junctors") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:85 remove_unary_junctors::operator()"""
        if (
            vertex_equal(it.data, builtin.logical_and)
            or vertex_equal(it.data, builtin.logical_or)
        ) and tr.has_one_child(it):
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))


class simplify_predicates(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:59 simplify_predicates

    Simplify any predicate terms found in the tree. See this module's own
    docstring for what is and is not implemented here.
    """

    def __init__(self, reduct_effort: int, ignore_ops, name: str = "simplify_predicates") -> None:
        """C++: moses/comboreduct/reduct/logical_rules.h:61 simplify_predicates

        The copy constructor (logical_rules.h:66-68) has no separate
        Python equivalent -- `rule.clone()`'s generic `copy.copy` already
        reproduces it, since neither field is ever mutated after
        construction.
        """
        super().__init__(name)
        self.reduct_effort = reduct_effort
        self.ignore_ops = ignore_ops

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:65 simplify_predicates::operator()

        # DIVERGENCE(C++ moses/comboreduct/reduct/logical_rules.cc:77): raises past this point.
        # The guard above (is_predicate / logical_not-stripping / is_argument) is a full, faithful
        # port and IS this rule's entire real behaviour for every tree in this file's own
        # boolean-only test corpus (no `greater_than_zero` node ever appears there, so this always
        # returns on the very first line). Only the TAIL -- `contin_reduce`/`mixed_reduce`,
        # PLANNED: S14 (contin/mixed reduction, out of this project's current scope per
        # docs/MODULE-COVERAGE.md) -- is not implemented. UPDATED (Task 5): meta_rules.py's own
        # analogous `_get_output_type_tree` stub this docstring used to compare against has since
        # been resolved for real (a scoped type_checker slice) -- this tail remains blocked because
        # S14 itself, not a missing type_checker dependency, is out of scope.
        """
        if not is_predicate(it):
            return
        if vertex_equal(it.data, builtin.logical_not):
            it = it.first_child
        cit = it.first_child
        if is_argument(cit.data):
            return
        raise NotImplementedError(
            "simplify_predicates: contin_reduce/mixed_reduce need "
            "comboreduct/reduct/contin_rules.py + mixed_rules.py, "
            "PLANNED: S14 per docs/MODULE-COVERAGE.md -- unreachable from "
            "this project's boolean-only reduction scope today."
        )


class remove_dangling_junctors(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:80 remove_dangling_junctors

    Remove operators 'and', 'or' and 'not' that have no argument. Note
    that this rule does not preserve semantics -- it is used so that
    logical formulae exhibit more diversity.
    """

    def __init__(self, name: str = "remove_dangling_junctors") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:95 remove_dangling_junctors::operator()

        `cond` takes (condition, consequent) PAIRS plus a trailing,
        unpaired "else" -- removing a null condition must also remove its
        paired consequent. `general_rules.py`'s `remove_null_vertices`
        already ported this EXACT pair-stride shape for a different guard
        (a `null_vertex` child, not a childless AND/OR/NOT) and its own
        docstring documents, via a bounded probe
        (`remove_null_vertices_cond_crash_probe.cc`), that real C++
        segfaults if a null CONDITION is also the trailing unpaired "else"
        (`tr.erase(sib++)` twice, the second dereferencing a
        just-invalidated end iterator). By analogy (same pair-stride
        shape, not independently re-probed for THIS guard) this port's
        `tr.erase(None)` would raise `AttributeError` at the same point --
        a strictly safer failure mode, not a fresh divergence.
        """
        if not vertex_equal(it.data, builtin.cond):
            sib = it.first_child
            while sib is not None:
                if (
                    vertex_equal(sib.data, builtin.logical_and)
                    or vertex_equal(sib.data, builtin.logical_or)
                    or vertex_equal(sib.data, builtin.logical_not)
                ) and tr.is_childless(sib):
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
        else:
            sib = it.first_child
            while sib is not None:
                if (
                    vertex_equal(sib.data, builtin.logical_and)
                    or vertex_equal(sib.data, builtin.logical_or)
                    or vertex_equal(sib.data, builtin.logical_not)
                ) and tr.is_childless(sib):
                    sib = tr.erase(sib)
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
                    if sib is not None:
                        sib = sib.next_sibling


class eval_logical_identities(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:90 eval_logical_identities

    and(true X)->X, or(true X)->true; and(false X)->false, or(false X)->X;
    or(X)->X; and(or X) -> and(X), or(and X) -> or(X).
    """

    def __init__(self, name: str = "eval_logical_identities") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:130 eval_logical_identities::operator()

        `it.data = sib.data` (both `*it = *sib` sites) copies an IMMUTABLE
        `builtin` value (`logical_true`/`logical_false`) -- no aliasing
        concern regardless of Task 3's fix, unlike `reduce_nots`' own
        `is_argument` branch below.
        """
        if vertex_equal(it.data, builtin.logical_and):
            sib = it.first_child
            while sib is not None:
                if (
                    vertex_equal(sib.data, builtin.logical_or) and tr.is_childless(sib)
                ) or vertex_equal(sib.data, builtin.logical_true):
                    sib = tr.erase(sib)
                elif vertex_equal(sib.data, builtin.logical_false):
                    it.data = sib.data
                    tr.erase_children(it)
                    break
                else:
                    sib = sib.next_sibling
        elif vertex_equal(it.data, builtin.logical_or):
            if tr.has_one_child(it):
                it.data = it.first_child.data
                tr.erase(tr.flatten(it.first_child))
            else:
                sib = it.first_child
                while sib is not None:
                    if (
                        vertex_equal(sib.data, builtin.logical_and)
                        and tr.is_childless(sib)
                    ) or vertex_equal(sib.data, builtin.logical_false):
                        sib = tr.erase(sib)
                    elif vertex_equal(sib.data, builtin.logical_true):
                        it.data = sib.data
                        tr.erase_children(it)
                        break
                    else:
                        sib = sib.next_sibling


class reduce_nots(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:98 reduce_nots

    Apply distributive law to push `not` towards children:
    !!a->a, !(a&&b)->(!a||!b), !(a||b)->(!a&&!b).
    """

    def __init__(self, name: str = "reduce_nots") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:177 reduce_nots::operator()

        THE ONE IN-PLACE VERTEX MUTATOR IN THE LANGUAGE (this module's own
        docstring, item 1): the `is_argument` branch does `it.data =
        it.first_child.data` (no copy -- see below) then
        `it.data.negate()`, mutating that value IN PLACE. Safe only
        because `tree.py`'s `_copy_value` fix (task-3) guarantees the
        argument object `it.first_child.data` refers to, at this point, is
        never shared with anything else live in the tree -- every subtree
        copy anywhere in this port's history produced an independent
        `argument` object, so `it.data`'s new reference (the same object
        `it.first_child.data` held) has exactly one owner. Pinned directly
        by `test_reduce_nots_in_place_negate_does_not_corrupt_a_sibling_
        occurrence` below. `it.data = it.first_child.data` itself needs no
        defensive copy despite reusing tree.py's aliasing-prone pattern:
        the source (`it.first_child`) is about to be discarded by
        `tr.erase_children(it)` on the very next line, and nothing else in
        the tree references that exact object either (same guarantee).
        """
        if not vertex_equal(it.data, builtin.logical_not):
            return
        if vertex_equal(it.first_child.data, builtin.logical_not):
            # !!a -> a
            tr.erase(tr.flatten(it.first_child))
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))
            self.apply(tr, it)
        elif vertex_equal(it.first_child.data, builtin.logical_and) or vertex_equal(
            it.first_child.data, builtin.logical_or
        ):
            # !(a&&b) -> (!a||!b), !(a||b) -> (!a&&!b)
            tr.flatten(it.first_child)
            was_and = vertex_equal(it.first_child.data, builtin.logical_and)
            it.data = builtin.logical_or if was_and else builtin.logical_and
            tr.erase(it.first_child)
            sib = it.first_child
            while sib is not None:
                sib = tr.insert_above(sib, builtin.logical_not)
                sib = sib.next_sibling
        elif is_argument(it.first_child.data):
            it.data = it.first_child.data
            tr.erase_children(it)
            it.data.negate()
        elif vertex_equal(it.first_child.data, builtin.logical_true):
            tr.erase_children(it)
            it.data = builtin.logical_false
        elif vertex_equal(it.first_child.data, builtin.logical_false):
            tr.erase_children(it)
            it.data = builtin.logical_true


class reduce_and_assumptions(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:109 reduce_and_assumptions

    and(x_1 x_2 x_3 ...): reduce x_1 assuming x_2, x_3 ...; reduce x_2
    assuming x_1, x_3 ...; etc. -- and choose the one that shortens the
    expression the most.
    """

    def __init__(self, r: rule | None = None, name: str = "reduce_and_assumptions") -> None:
        """C++: moses/comboreduct/reduct/logical_rules.h:111 reduce_and_assumptions

        Two C++ constructors: `reduce_and_assumptions(const rule& r)`
        (:111) stores a raw, NON-OWNING `const rule*` -- no clone, unlike
        `downwards`/`upwards`/`sequential`'s own wrapped rules, which DO
        clone at construction -- and the no-arg default (:115,
        `_reduction(this)`, the instance reduces using itself
        recursively).
        """
        super().__init__(name)
        self._reduction: rule = self if r is None else r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:226 reduce_and_assumptions::operator()

        `tr.subtree(sib, tr.next_sibling(sib))` (a sibling RANGE of
        exactly one element) is `combo_tree(sib)` here -- C++'s general
        `subtree(from, to)` degenerates to a single-node standalone copy
        for a one-element range (opencog/util/tree.h:1962-1967's own
        `tmp.replace(tmp.begin(), tmp.end(), from, to)` into a freshly
        head-only tree).

        `for (sib_it a = tr.next_sibling(tr.begin()); a != tr.end(); ++a)
        bna = copy_tr.insert_subtree_after(bna, a);` (logical_rules.cc:
        236-238) walks `tr`'s OWN assumption region as tree-root SIBLINGS
        -- structurally impossible in this port (assumption.py's module
        docstring) -- and is exactly `copy_all_assumptions(copy_tr, tr)`,
        added by Task 1 for this precise call site.
        """
        if not vertex_equal(it.data, builtin.logical_and):
            return

        best_diff = 0
        best_index = -1
        best_tree: combo_tree | None = None

        sib = it.first_child
        while sib is not None:
            copy_tr = combo_tree(sib)
            copy_all_assumptions(copy_tr, tr)

            a = it.first_child
            while a is not None:
                if a is not sib:
                    insert_assumption(copy_tr, a)
                a = a.next_sibling

            s_before = copy_tr.count(copy_tr.begin())
            self._reduction(copy_tr)
            s_after = copy_tr.count(copy_tr.begin())
            diff = s_after - s_before
            if diff < best_diff:
                best_diff = diff
                best_index = tr.sibling_index(sib)
                best_tree = copy_tr

            sib = sib.next_sibling

        if best_index > -1:
            tr.replace(tr.child(it, best_index), best_tree.begin())


class reduce_or_assumptions(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:127 reduce_or_assumptions

    Reduce or(x_1 x_2 x_3 ...) by reducing to not(and(not(x_1)...)) using
    `reduce_and_assumptions`.
    """

    def __init__(self, r: rule | None = None, name: str = "reduce_or_assumptions") -> None:
        """C++: moses/comboreduct/reduct/logical_rules.h:129 reduce_or_assumptions

        Same raw-reference (no clone) `_reduction` convention as
        `reduce_and_assumptions.__init__` above, for the identical reason.
        """
        super().__init__(name)
        self._reduction: rule = self if r is None else r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:264 reduce_or_assumptions::operator()"""
        if not vertex_equal(it.data, builtin.logical_or):
            return
        if tr.is_childless(it):
            raise AssertionError("combo_tree node should not be childless")

        it.data = builtin.logical_not
        new_or = tr.prepend_child(it, builtin.logical_or)
        original_first = new_or.next_sibling
        tr.reparent(new_or, original_first, None)
        first_not = tr.wrap(new_or, builtin.logical_not)
        self._reduction(tr, first_not)
        # to avoid an infinite recursion not(and(x y...)) is reduced to
        # or(not(x) ...), then all not(x), not(y) ... are reduced
        reduce_nots()(tr, it)
        sib = it.first_child
        while sib is not None:
            self._reduction(tr, sib)
            sib = sib.next_sibling


class reduce_ors(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:148 reduce_ors

    Heuristic reduction of ORs based on complementary pairs:
    1) pairwise implications of conjuncts (X&&a)||((Y&&!a) impl X&&Y
    2) for all pairs of conjuncts, including implications (in X) if X is a
       subset of (or equal to) Y, remove Y
    Also, true||X -> true, false||X -> X.
    """

    def __init__(self, name: str = "reduce_ors") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:287 reduce_ors::operator()

        `c1`/`c2` are tracked by OBJECT IDENTITY (matching C++'s
        `nf::iterator`s, which stay valid across erasing OTHER list
        elements): `clause` defines no `__eq__`, so Python's default
        identity-based `is`/`in`/`list.index` already give exactly that.
        `i2`/`i1` are RE-DERIVED via `f.index(...)` after any erasure that
        might shift them, rather than adjusted incrementally -- simpler
        and, since `c1`/`c2` are themselves never among the elements
        erased in this inner loop, always well-defined.
        """
        if not vertex_equal(it.data, builtin.logical_or):
            return

        mapper = nf_mapper()
        f: nf = mapper.add_dnf(it)
        sz = number_of_literals(f)

        f[:] = [c for c in f if not tautology(c)]
        pairwise_erase_if(f, subset_eq)

        i1 = 0
        while i1 < len(f):
            c1 = f[i1]
            i2 = 0
            # FIX WAVE (Task 4 review, Critical): bounded by `f.index(c1)`,
            # RE-DERIVED on every check, not a stale `i1` snapshot -- C++'s
            # `for (c2 = f.begin(); c2 != c1; ++c2)` (logical_rules.cc:311)
            # is bounded by the LIVE `c1`, and the inner loop's own
            # `del f[i]` can erase elements BEFORE c1, shifting its real
            # index down. A stale `i1` bound then lets `i2` walk onto (or
            # past) c1 itself -- IndexError once `i1` exceeds the
            # shrunken `len(f)`, or a silent wrong pairing before that.
            # Measured: `reduce_ors()` on
            # `or(!$1 or(and($2 !$1 !$2)) !$2 $2)` raised IndexError here;
            # C++ returns `true`.
            while i2 < f.index(c1):
                c2 = f[i2]
                for impl in list(implications(c1, c2)):
                    i = 0
                    while i < len(f):
                        c = f[i]
                        if c is not c1 and c is not c2 and subset_eq(impl, c):
                            del f[i]
                        else:
                            i += 1
                    if subset(impl, c1):
                        c1.assign(impl)
                    if subset(impl, c2):
                        c2.assign(impl)
                i2 = f.index(c2) + 1
            i1 = f.index(c1) + 1

        if sz != number_of_literals(f):
            mapper.extract_dnf(f, tr, it)


def _negate_clause(c: clause) -> clause:
    """No direct C++ symbol: `std::transform(c->begin(), c->end(),
    std::inserter(tmp, tmp.begin()), std::negate<int>())` -- negate every
    literal in `c` into a FRESH clause (re-sorted per `clause`'s own
    absolute-value order, which negation never changes since |x|==|-x|).
    Used by `reduce_ands` at every one of its three `*c = tmp` negate-
    in-place sites (logical_rules.cc:352-357, 415-419, 421-424).
    """
    negated = clause()
    for x in c:
        negated.insert(-x)
    return negated


class reduce_ands(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:159 reduce_ands

    Heuristic reduction of ANDs based on complementary pairs:
    1) pairwise implications of conjuncts (X||a)&&((Y||!a) impl (X||Y)
    2) for all pairs of conjuncts, including implications (in X) if X is a
       subset of (or equal to) Y, remove X
    Also, true&&X -> X, false&&X -> false.
    """

    def __init__(self, name: str = "reduce_ands") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:339 reduce_ands::operator()

        Same c1/c2-by-identity, re-derived-index convention as
        `reduce_ors.apply` above, for the identical reason.
        """
        if not vertex_equal(it.data, builtin.logical_and):
            return

        mapper = nf_mapper()
        f: nf = mapper.add_cnf(it)
        sz = number_of_literals(f)

        for c in f:
            c.assign(_negate_clause(c))

        f[:] = [c for c in f if not tautology(c)]

        if not f:  # tautological expression
            tr.erase_children(it)
            it.data = builtin.logical_true
            return

        pairwise_erase_if(f, subset_eq)

        i1 = 0
        while i1 < len(f):
            c1 = f[i1]
            i2 = 0
            # FIX WAVE (Task 4 review, Critical): same re-derived bound as
            # reduce_ors.apply above, for the identical reason -- C++'s
            # `for (c2 = f.begin(); c2 != c1; ++c2)` (logical_rules.cc:373)
            # is bounded by the LIVE `c1`, not a stale index snapshot.
            # Measured: reached via the "medium" pipeline on real trees.
            while i2 < f.index(c1):
                c2 = f[i2]
                for impl in list(implications(c1, c2)):
                    i = 0
                    while i < len(f):
                        c = f[i]
                        if c is not c1 and c is not c2 and subset_eq(impl, c):
                            del f[i]
                        else:
                            i += 1
                    if subset(impl, c1):
                        c1.assign(impl)
                    if subset(impl, c2):
                        c2.assign(impl)
                i2 = f.index(c2) + 1
            i1 = f.index(c1) + 1

        intersect = clause()
        if len(f) > 1 and f[0].size() > 1:
            intersect = clause(tuple(f[0]))
            for c in f[1:]:
                if intersect.empty():
                    break
                if c.size() == 1:
                    intersect = clause()
                    break
                common = clause()
                for x in intersect:
                    if x in c:
                        common.insert(x)
                intersect = common

        if sz != number_of_literals(f) or not intersect.empty():
            for c in f:
                for x in list(intersect):
                    c.erase(x)
                c.assign(_negate_clause(c))
            intersect.assign(_negate_clause(intersect))
            mapper.extract_cnf(f, tr, it)

        if not intersect.empty():
            it.data = builtin.logical_or
            if not tr.is_childless(it):
                tr.prepend_child(it, builtin.logical_and)
                new_first = it.first_child
                tr.reparent(new_first, new_first.next_sibling, None)
            new_conj_node = tr.append_child(it, 0.0)
            mapper.extract_conjunction(intersect, tr, new_conj_node)
            if intersect.size() > 1:
                tr.erase(tr.flatten(it.last_child))


class subtree_to_enf(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:169 subtree_to_enf

    Reduction to ENF (elegant normal form). Note that `reduce_nots()` and
    `eval_logical_identities()` and `level()` should be called first to
    ensure proper reduction. See this module's own top-of-file docstring
    for the structural decision behind `reduce_to_enf`'s shape.
    """

    def __init__(self, name: str = "subtree_to_enf") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.h:173 subtree_to_enf::operator()"""
        self.reduce_to_enf(tr, it)

    class reduce_to_enf:
        """C++: moses/comboreduct/reduct/logical_rules.h:179 reduce_to_enf

        Constructing this class RUNS it immediately (`__init__` calls
        `self(it)` at the end), matching C++'s own constructor
        (logical_rules.h:191-195: `reduce_to_enf(tr_, it) : tr(tr_) {
        (*this)(it); }`).
        """

        class Result(Enum):
            """C++: moses/comboreduct/reduct/logical_rules.h:181 reduce_to_enf::Result"""

            Delete = auto()
            Disconnect = auto()
            Keep = auto()

        class tree_eraser:
            """C++: moses/comboreduct/reduct/logical_rules.h:240 tree_eraser

            # DIVERGENCE(C++ moses/comboreduct/reduct/logical_rules.h:244): takes a plain node.
            # Not an iterator-to-iterator -- see this module's own merge-algorithms section
            # note on the counting_iterator collapse: `tr.erase(*it)` becomes `tr.erase(node)`.
            """

            def __init__(self, tr: combo_tree) -> None:
                self.tr = tr

            def __call__(self, node: tree_node) -> None:
                self.tr.erase(node)

        class subtree_set_eraser:
            """C++: moses/comboreduct/reduct/logical_rules.h:248 subtree_set_eraser

            Same collapse as `tree_eraser` above.
            """

            def __init__(self, se: subtree_set) -> None:
                self.se = se

            def __call__(self, node: tree_node) -> None:
                self.se.erase(node)

        class tree_inserter:
            """C++: moses/comboreduct/reduct/logical_rules.h:256 tree_inserter

            # DIVERGENCE(C++ moses/comboreduct/reduct/logical_rules.h:261): takes (pos, node).
            # Not (It1, It2) -- `pos` is a `tree_node` from the destination range (insert BEFORE
            # it) or `None` for "destination range exhausted" (`it1->node==0`, C++'s own
            # null-position check on the counting_iterator's underlying `sib_it`, collapsed the
            # same way as `tree_eraser`/`subtree_set_eraser` above); `node` is the
            # already-dereferenced source subtree to copy in.
            """

            def __init__(self, tr: combo_tree, it: tree_node) -> None:
                self.tr = tr
                self.it = it

            def __call__(self, pos: tree_node | None, node: tree_node) -> None:
                if pos is None:
                    self.tr.append_child(self.it, node)
                else:
                    self.tr.insert_subtree(pos, node)

        def __init__(self, tr: combo_tree, it: tree_node) -> None:
            """C++: moses/comboreduct/reduct/logical_rules.h:191 reduce_to_enf::reduce_to_enf"""
            self.tr = tr
            self.comp = lexicographic_subtree_less
            self(it)

        def __call__(self, it: tree_node) -> None:
            """C++: moses/comboreduct/reduct/logical_rules.cc:441 reduce_to_enf::operator()

            The C++ body declares `static const type_tree boolean_type_tree
            = type_tree(id::boolean_type);` but never actually reads it in
            this method -- confirmed vestigial by reading the whole method
            body; the SAME dead declaration also appears, equally unused,
            in `build_subtree_sets_upwards` below. Neither is ported.
            """
            tr = self.tr
            comp = self.comp
            if not is_logical_operator(it.data):
                return

            tr.sort_on_subtrees(it, comp, True)
            p = it.parent
            while p is not None and is_logical_operator(p.data):
                if not is_sorted(list(tr.children(p)), comp):
                    tr.sort_on_subtrees(p, comp)
                p = p.parent

            dominant = subtree_set(comp)
            command = subtree_set(comp)
            self.build_subtree_sets_upwards(it, dominant, command)

            result = self.reduce(it, dominant, command)
            if result == self.Result.Delete:  # a contradiction
                tr.erase_children(it)
                it.data = builtin.logical_false
            elif result == self.Result.Disconnect:  # a tautology
                tr.erase_children(it)
                it.data = builtin.logical_true
            # Keep: neither -- no-op

        def build_subtree_sets_upwards(self, up: tree_node, dom_out, cmd_out) -> None:
            """C++: moses/comboreduct/reduct/logical_rules.h:209 build_subtree_sets_upwards

            The C++ `Out` template parameter (an output iterator) collapses
            to `dom_out`/`cmd_out` each being any object with an
            `.insert(node)` method -- both real call sites pass a fresh
            `subtree_set`.
            """
            p = up.parent
            while p is not None and is_logical_operator(p.data):
                if vertex_equal(p.data, builtin.logical_and):
                    sib = p.first_child
                    while sib is not None:
                        if sib is not up:
                            dom_out.insert(sib)
                        sib = sib.next_sibling
                else:
                    if not vertex_equal(p.data, builtin.logical_or):
                        raise AssertionError("*p == id::logical_or")
                    sib = p.first_child
                    while sib is not None:
                        if sib is not up and is_argument(sib.data):
                            cmd_out.insert(sib)
                        sib = sib.next_sibling
                up = p
                p = p.parent

        def push_back_negated_arguments(self, s, negated: list[combo_tree]) -> None:
            """C++: moses/comboreduct/reduct/logical_rules.h:230 push_back_negated_arguments

            `combo_tree(sib)` (a standalone deep copy) then `.negate()`
            the COPY's own root. Relies on `tree.py`'s `_copy_value` fix
            (task-3) to leave `sib` itself untouched -- pre-fix this
            silently mutated the source. Pinned directly by
            `test_push_back_negated_arguments_does_not_mutate_the_source`
            below (this module's own docstring, item 3).
            """
            for sib in s:
                if is_argument(sib.data):
                    new_tree = combo_tree(sib)
                    new_tree.begin().data.negate()
                    negated.append(new_tree)

        def consistent(self, s) -> bool:
            """C++: moses/comboreduct/reduct/logical_rules.cc:475 reduce_to_enf::consistent

            `s` is sorted per `comp` (`lexicographic_subtree_order`),
            which -- for two ARGUMENTS specifically -- reduces to
            `argument`'s own absolute-value order, placing $N and !$N
            adjacent to each other (the same property
            `flat_normal_form.tautology`'s adjacent-pairs-only scan relies
            on). `std::adjacent_find` over `is_complement` is therefore a
            correct and complete scan of ADJACENT pairs only.
            """
            items = list(s)
            return all(
                not is_complement(items[i].data, items[i + 1].data)
                for i in range(len(items) - 1)
            )

        def and_cut(self, child: tree_node) -> bool:
            """C++: moses/comboreduct/reduct/logical_rules.cc:482 reduce_to_enf::and_cut"""
            tr = self.tr
            adopted = False
            gchild = child.first_child
            while gchild is not None:
                if is_logical_operator(gchild.data):
                    if tr.has_one_child(gchild) and (
                        vertex_equal(gchild.data, builtin.logical_and)
                        or vertex_equal(gchild.data, builtin.logical_or)
                    ):
                        if vertex_equal(gchild.first_child.data, builtin.logical_or):
                            tr.erase(tr.flatten(gchild.first_child))
                            if not adopted:
                                x = gchild.first_child
                                while x is not None:
                                    if tr.has_one_child(x):
                                        adopted = True
                                        break
                                    x = x.next_sibling
                        elif not is_predicate(gchild.first_child) and not is_argument(
                            gchild.first_child.data
                        ):
                            raise AssertionError(
                                "Logical reduction: unexpected operator: "
                                f"{gchild.first_child.data!r}"
                            )
                        gchild = tr.erase(tr.flatten(gchild))
                        continue
                elif not is_argument(gchild.data) and not is_predicate(gchild):
                    raise AssertionError(
                        f"Logical reduction: unexpected operator: {gchild.data!r}"
                    )
                gchild = gchild.next_sibling
            return adopted

        def or_cut(self, current: tree_node) -> None:
            """C++: moses/comboreduct/reduct/logical_rules.cc:540 reduce_to_enf::or_cut"""
            tr = self.tr
            comp = self.comp
            child = current.first_child
            while child is not None:
                if tr.has_one_child(child) and (
                    vertex_equal(child.data, builtin.logical_and)
                    or vertex_equal(child.data, builtin.logical_or)
                ):
                    if vertex_equal(child.first_child.data, builtin.logical_or):
                        insert_set_complement(
                            self.tree_inserter(tr, current),
                            list(tr.children(current)),
                            list(tr.children(child.first_child)),
                            comp,
                        )
                        tr.erase(tr.flatten(child.first_child))
                        child = tr.erase(tr.flatten(child))
                    elif is_argument(child.first_child.data) or is_predicate(
                        child.first_child
                    ):
                        child = tr.erase(tr.flatten(child))
                    else:
                        # Just has one child, not argument/predicate/or --
                        # conclude it's a logical_and with a childless
                        # logical_and under it. Delete it.
                        child = tr.erase(child)
                else:
                    child = child.next_sibling

        def reduce(self, current: tree_node, dominant: subtree_set, command: subtree_set):
            """C++: moses/comboreduct/reduct/logical_rules.cc:585 reduce_to_enf::reduce

            First removes duplicate children, assuming `current` is
            already sorted per `comp` (so duplicates are adjacent) --
            matches the C++ comment's own stated assumption verbatim, not
            re-verified here.
            """
            tr = self.tr
            if not tr.is_childless(current):
                sib1 = current.first_child
                sib2 = sib1.next_sibling
                while sib2 is not None:
                    if tr.equal_subtree(sib1, sib2):
                        sib2 = tr.erase(sib2)
                    else:
                        sib1 = sib2
                        sib2 = sib2.next_sibling

            if vertex_equal(current.data, builtin.logical_and):
                return self.reduce_and(current, dominant, command)
            if vertex_equal(current.data, builtin.logical_or):
                return self.reduce_or(current, dominant, command)
            return self.Result.Keep

        def reduce_and(self, current: tree_node, dominant: subtree_set, command: subtree_set):
            """C++: moses/comboreduct/reduct/logical_rules.cc:631 reduce_to_enf::reduce_and"""
            tr = self.tr
            comp = self.comp
            Result = self.Result

            # 0-subsumption of dominant constraints already present verbatim.
            erase_set_intersection(
                self.tree_eraser(tr), list(tr.children(current)), list(dominant), comp
            )

            # 1-subsumption via direct removal of negated command arguments.
            negated: list[combo_tree] = []
            self.push_back_negated_arguments(command, negated)
            _sort_by_comp(negated, lambda t: t.begin(), comp)
            negated_roots = [t.begin() for t in negated]

            erase_set_intersection(
                self.tree_eraser(tr), list(tr.children(current)), negated_roots, comp
            )

            if tr.is_childless(current):
                return Result.Disconnect  # 0-subsume

            if not has_empty_intersection(
                list(tr.children(current)), list(command), comp
            ):
                return Result.Delete  # 1-subsume

            while True:
                prev_guard_set = list(tr.children(current))

                if not is_sorted(list(tr.children(current)), comp):
                    tr.sort_on_subtrees(current, comp)

                handle_set = dominant.copy()  # broom handle
                for node in tr.children(current):
                    handle_set.insert(node)

                if not self.consistent(handle_set):
                    return Result.Delete

                child = current.first_child
                while child is not None:
                    if tr.is_childless(child) or not (
                        vertex_equal(child.data, builtin.logical_and)
                        or vertex_equal(child.data, builtin.logical_or)
                    ):
                        child = child.next_sibling
                        continue

                    tr.validate(child)
                    tr.validate()

                    tmp = handle_set.find(child)
                    add_it = tmp is not None
                    if add_it:
                        handle_set.erase(tmp)

                    result = self.reduce(child, handle_set, command)
                    if result == Result.Delete:
                        return Result.Delete  # current is in all selections with child
                    if result == Result.Disconnect:
                        child = tr.erase(child)
                        add_it = False
                    else:  # Keep
                        if not is_sorted(list(tr.children(current)), comp):
                            tr.sort_on_subtrees(current, comp)

                        if child.first_child is None:
                            raise AssertionError("child should have siblings")

                        # Make res the disjunction of child's children's
                        # guard sets. The last_child won't have children
                        # if it's an argument; treat predicates the same.
                        res = subtree_set(comp)
                        if not is_predicate(child.last_child):
                            pit = child.last_child.first_child
                            while pit is not None:
                                res.insert(pit)
                                pit = pit.next_sibling

                        sib = child.first_child
                        while sib is not child.last_child:
                            erase_set_difference(
                                self.subtree_set_eraser(res),
                                list(res),
                                list(tr.children(sib)),
                                comp,
                            )
                            sib = sib.next_sibling

                        if not res.empty():
                            insert_set_complement(
                                self.tree_inserter(tr, current),
                                list(tr.children(current)),
                                list(res),
                                comp,
                            )

                            gchild = child.first_child
                            while gchild is not None:
                                # Do NOT erase children of predicates.
                                if not is_predicate(gchild):
                                    erase_set_intersection(
                                        self.tree_eraser(tr),
                                        list(tr.children(gchild)),
                                        list(res),
                                        comp,
                                    )
                                gchild = gchild.next_sibling

                            self.and_cut(child)
                            # guard_set has been enlarged, reprocess all kids
                            child = current.first_child
                        else:
                            if not self.and_cut(child):  # if child is adopting a
                                child = child.next_sibling  # terminal 1-constraint AND

                    if add_it:
                        handle_set.insert(tmp)

                self.or_cut(current)

                now = list(tr.children(current))
                if len(now) == len(prev_guard_set) and all(
                    a is b for a, b in zip(prev_guard_set, now, strict=True)
                ):
                    break

            if tr.is_childless(current):
                return Result.Disconnect  # 0-subsumption
            return Result.Keep

        def reduce_or(self, current: tree_node, dominant: subtree_set, command: subtree_set):
            """C++: moses/comboreduct/reduct/logical_rules.cc:800 reduce_to_enf::reduce_or"""
            tr = self.tr
            Result = self.Result
            child = current.first_child
            while child is not None:
                child_command = command.copy()
                sib = current.first_child
                while sib is not None:
                    if (
                        sib is not child
                        and tr.has_one_child(sib)
                        and not is_predicate(sib)
                    ):
                        child_command.insert(sib.first_child)
                    sib = sib.next_sibling

                result = self.reduce(child, dominant, child_command)
                if result == Result.Delete:
                    if tr.is_childless(current) or tr.has_one_child(current):
                        return Result.Delete
                    child = tr.erase(child)
                elif result == Result.Disconnect:
                    return Result.Disconnect
                else:  # Keep
                    child = child.next_sibling
            return Result.Keep


class reduce_remove_subtree_equal_tt(rule):
    """C++: moses/comboreduct/reduct/logical_rules.h:275 reduce_remove_subtree_equal_tt

    (suggested by Moshe) check if the truth table is the same after
    removal of any subtree. See this module's own docstring for what is
    and is not implemented here.
    """

    def __init__(self, name: str = "reduce_remove_subtree_equal_tt") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/logical_rules.cc:833 reduce_remove_subtree_equal_tt
        (the real member is `reduce_remove_subtree_equal_tt::operator()`).

        Task 5 (S7) landed the two forward references this rule was
        blocked on: `complete_truth_table` (`comboreduct/table/table.py`)
        and `clean_reduce` (`comboreduct/reduct/clean_reduction.py`).

        The predicate-scan guard (`for pit in tr: if is_predicate(pit):
        return`) walks the WHOLE tree (`tr.begin()` to `tr.end()`, not
        just `it`'s subtree) -- a full, faithful port of
        `logical_rules.cc:838-841`'s own `for (pre_it pit = tr.begin();
        pit != tr.end(); ++pit)`.

        NOT SCOPED TO `it`'S SUBTREE, a real and measured C++ quirk, not
        an approximation: `for (pre_it pit = it.begin(); pit != it.end();)`
        (logical_rules.cc:844) declares `pit` as a `pre_it` (`combo_tree::
        iterator`) but INITIALIZES it from `it.begin()`/bounds it with
        `it.end()`, both of which are `sibling_iterator`s converted to
        `pre_it` -- and that conversion keeps only the underlying node
        pointer, discarding `sibling_iterator`'s `parent_` field entirely
        (`iterator_base::end()`, opencog/util/tree.h:2356-2362, returns a
        node-less `sibling_iterator` whose ONLY payload after conversion
        is a null node pointer). So `pit != it.end()` reduces to "`pit`'s
        node is not null" -- NOT "`pit` is still inside `it`'s own
        subtree". The walk starts at `it`'s first child and, via ordinary
        whole-tree pre-order stepping, can wander PAST `it`'s own subtree
        boundary into whatever structurally follows it, all the way to
        the true end of the entire tree. Reproduced here by starting at
        `it.first_child` and stepping with the tree's own general
        pre-order primitives (`insert_above`/`erase_pre_order`/
        `next_pre_order`) until the node reference becomes `None` --
        exactly mirroring "keep going until the node pointer itself is
        null", not a subtree-bounded walk. Harmless for every real call
        site in this project's scope (`logical_reduction.cc`'s effort-3
        composition always applies at the tree's own root, where "past
        it's subtree" and "past the whole tree" coincide), but ported
        faithfully rather than narrowed to what today's callers happen to
        need.

        `tr.insert_above(pit, id::null_vertex)` (logical_rules.cc:846)
        returns the NEWLY CREATED null_vertex node (`insert_above`'s own
        docstring, pymetta/util/tree.py) -- `pit` is REASSIGNED to that
        wrapper, with the original content now null_vertex's one and only
        child. The `if`/`else` branches below erase using
        `tr.erase_pre_order`, NOT plain `tr.erase` -- see that method's
        own docstring and the `TODO` `erase()`'s Task 4 fix round left at
        this exact call site: `pit` is a `pre_it` here, and `erase_pre_
        order` is the separately-ported pre-order-successor variant this
        rule needs, not the sibling-successor semantics every OTHER rule
        in this file uses.

        `combo_tree(tr.begin())` is a full snapshot COPY of the WHOLE
        tree (matching C++'s copy-constructor `combo_tree amputated_tr
        (tr);` value-copy semantics) -- `clean_reduce` then runs on that
        COPY alone, so a failed (truth-table-changing) amputation attempt
        never permanently affects the real `tr` beyond the deliberate,
        already-committed `insert_above`/`erase_pre_order`/`flatten`
        splicing this method's own two branches perform on it directly.

        `clean_reduce` is imported HERE, inside the method, rather than
        at module level: `clean_reduction.py` imports `remove_dangling_
        junctors` FROM this module, so a top-level import in the other
        direction would be a real circular import (this port has no
        header/implementation split to break the cycle the way C++'s
        `reduct.h`/`clean_reduction.cc` do). Deferred until first call,
        by which point both modules are fully loaded.
        """
        for pit in tr:
            if is_predicate(pit):
                return

        from pymetta.comboreduct.reduct.clean_reduction import clean_reduce

        tr_tt = complete_truth_table(tr)
        pit = it.first_child
        while pit is not None:
            pit = tr.insert_above(pit, builtin.null_vertex)
            amputated_tr = combo_tree(tr.begin())
            clean_reduce(amputated_tr)
            if tr_tt.same_complete_truth_table(amputated_tr):
                pit = tr.erase_pre_order(pit)
            else:  # remove only the previously added null_vertex
                pit = tr.erase_pre_order(tr.flatten(pit))
                pit = tr.next_pre_order(pit)
