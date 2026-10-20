"""Tests for pymoses/comboreduct/reduct/logical_rules.py -- the boolean
logic reduction rules, culminating in `subtree_to_enf`.

C++: moses/comboreduct/reduct/logical_rules.{h,cc}.

Every literal expected tree string below (the ones NOT computed at
runtime via a truth-table/size check) is copied verbatim from a REAL run
of `tools/cpp_probe/logical_rules_probe.cc` against $CLASSIC's compiled
`libcomboreduct.so` (bounded:
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/logical_rules_probe
)`), not hand-derived. See that file and logical_rules.py's own module
docstring for the full transcript and the reasoning behind the
`BooleanReductUTest` scoping decision (a test-local, untyped "medium"
composition -- `_boolean_reduce_medium` below). MEASURED, not assumed:
this composition agrees byte-for-byte with the real, typed
`logical_reduction`'s effort-2 composition on 5 of that test file's own 6
trees (1, 3, 4, 5, 6) -- tree 2 (`test_simplify2`) genuinely diverges
(FIX WAVE, Task 4 review Important: an earlier version of this note
wrongly claimed all 6 agree, a misread of this same probe's own
already-correct output). `test_simplify2` is pinned against this port's
own untyped value, with the divergence documented directly on that test.
The literal `logical_reduce(3, ...)` effort-3 call the real cxxtest makes
is NOT reproducible here -- see the module docstring's own "TESTING
SCOPE" section for why.

`BooleanFullUTest` is NOT ported (module docstring: 20 test methods, not
24 -- FIX WAVE, Task 4 review Minor, an earlier count was wrong -- and 0
of the 20 are reachable today: every one needs type inference (S8),
contin/mixed reduction (S14, out of this project's scope), and
`logical_reduction.py` (Task 5) all three, and 6 of the 20 additionally
need a `definite_object` vertex this port's union does not carry).
"""

from __future__ import annotations

import itertools

import pytest

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    equal_subtree_vertex_safe,
    vertex_equal,
)
from pymoses.comboreduct.interpreter.interpreter import boolean_interpreter
from pymoses.comboreduct.reduct.general_rules import level
from pymoses.comboreduct.reduct.logical_rules import (
    erase_set_intersection,
    eval_logical_identities,
    has_empty_intersection,
    insert_ands,
    insert_set_complement,
    is_complement,
    is_logical_operator,
    is_predicate,
    is_sorted,
    reduce_and_assumptions,
    reduce_ands,
    reduce_nots,
    reduce_or_assumptions,
    reduce_ors,
    reduce_remove_subtree_equal_tt,
    remove_dangling_junctors,
    remove_unary_junctors,
    simplify_predicates,
    subtree_set,
    subtree_to_enf,
)
from pymoses.comboreduct.reduct.logical_reduction import logical_reduce
from pymoses.comboreduct.reduct.meta_rules import downwards, iterative, upwards
from pymoses.comboreduct.reduct.reduct import rule
from pymoses.comboreduct.table.table import complete_truth_table

# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------


def _apply(rule_obj: rule, tree_str: str) -> str:
    tr = str_to_combo_tree(tree_str)
    rule_obj(tr, tr.begin())
    return combo_tree_to_str(tr).strip()


def _enf(tree_str: str) -> str:
    """`subtree_to_enf`'s own docstring: `reduce_nots()`,
    `eval_logical_identities()` and `level()` should run first."""
    tr = str_to_combo_tree(tree_str)
    it = tr.begin()
    upwards(eval_logical_identities())(tr, it)
    downwards(level())(tr, it)
    subtree_to_enf()(tr, it)
    return combo_tree_to_str(tr).strip()


def _enf_raw(tree_str: str) -> str:
    """`subtree_to_enf()` ALONE, with NO pre-canonicalization -- unlike
    `_enf` above. `level()`'s own flattening (`general_rules.py`'s
    `level`, associative-operator merge) collapses redundant
    same-operator nesting (e.g. `and(and(and()))` -> a single childless
    `and()`) BEFORE `subtree_to_enf` would otherwise see it, which
    silently routes around `reduce_and`'s own TRAILING 0-subsumption
    check and `reduce_or`'s Disconnect-propagation branch -- both
    genuinely reachable in C++ (and in this port) when `subtree_to_enf`
    is called without that pre-pass, exactly as its own docstring already
    frames the pre-pass as needed for "proper reduction" QUALITY, not as
    a precondition for calling it at all. Matches
    `tools/cpp_probe/logical_rules_probe.cc`'s own `run_enf_raw`.
    """
    tr = str_to_combo_tree(tree_str)
    subtree_to_enf()(tr, tr.begin())
    return combo_tree_to_str(tr).strip()


# ---------------------------------------------------------------------------
# vertex.h predicates narrowly scoped to this file
# ---------------------------------------------------------------------------


def test_is_logical_operator():
    assert is_logical_operator(builtin.logical_and)
    assert is_logical_operator(builtin.logical_or)
    assert is_logical_operator(builtin.logical_not)
    assert not is_logical_operator(builtin.logical_true)
    assert not is_logical_operator(argument(1))


def test_is_logical_operator_does_not_confuse_a_contin_with_a_matching_ordinal():
    # builtin.logical_and == 3; type-before-value (CLAUDE.md).
    assert builtin.logical_and == 3
    assert not is_logical_operator(3.0)


def test_is_predicate_true_for_greater_than_zero_and_its_negation():
    tr = str_to_combo_tree("0<($1)")
    assert is_predicate(tr.begin())
    tr2 = str_to_combo_tree("not(0<($1))")
    assert is_predicate(tr2.begin())


def test_is_predicate_false_for_ordinary_logical_nodes():
    tr = str_to_combo_tree("and($1 $2)")
    assert not is_predicate(tr.begin())
    tr2 = str_to_combo_tree("not($1)")
    assert not is_predicate(tr2.begin())


def test_is_predicate_raises_for_a_childless_logical_not():
    """Task 4 review Minor -- inherited C++ UB (vertex.h:781's `*it.
    begin()` on a malformed, childless `not`), ported faithfully as a
    loud `AttributeError` rather than silently corrected. See
    `is_predicate`'s own docstring for the two real call sites in this
    file that can reach this shape.
    """
    childless_not = combo_tree(builtin.logical_not).begin()
    with pytest.raises(AttributeError):
        is_predicate(childless_not)


def test_enf_crashes_on_a_childless_not_via_reduce_and_res_computation():
    """One of the two real reachable call sites documented on
    `is_predicate` -- `and(and(not()))`: the inner `and`'s only child is
    a childless `not`, reached by `reduce_and`'s own `res`-computation
    setup (`if not is_predicate(child.last_child):`) while processing the
    OUTER `and`'s only child (the inner `and`). Not something a
    `str_to_combo_tree` parse can express (its grammar requires `not`'s
    operand); built directly via the node API instead.
    """
    tr = combo_tree(builtin.logical_and)
    root = tr.begin()
    inner_and = tr.append_child(root, builtin.logical_and)
    tr.append_child(inner_and, builtin.logical_not)  # childless -- inner_and's only child

    with pytest.raises(AttributeError):
        subtree_to_enf()(tr, root)


def test_is_complement_true_only_for_two_arguments_of_opposite_sign():
    assert is_complement(argument(1), argument(-1))
    assert is_complement(argument(-3), argument(3))
    assert not is_complement(argument(1), argument(1))
    assert not is_complement(argument(1), argument(2))
    assert not is_complement(builtin.logical_true, builtin.logical_false)
    assert not is_complement(argument(1), builtin.logical_true)


# ---------------------------------------------------------------------------
# insert_ands
# ---------------------------------------------------------------------------


def test_insert_ands_bare_argument_at_root():
    assert _apply(insert_ands(), "$1") == "and($1)"


def test_insert_ands_or_under_not_is_already_fine():
    assert _apply(insert_ands(), "not(or($1 $2))") == "not(or($1 $2))"


def test_insert_ands_argument_under_or():
    assert _apply(insert_ands(), "or($1 $2)") == "and(or($1 $2))"


# ---------------------------------------------------------------------------
# remove_unary_junctors
# ---------------------------------------------------------------------------


def test_remove_unary_junctors_and_one_child():
    assert _apply(remove_unary_junctors(), "and($1)") == "$1"


def test_remove_unary_junctors_or_one_child_promotes_grandchildren():
    assert _apply(remove_unary_junctors(), "or(and($1 $2))") == "and($1 $2)"


def test_remove_unary_junctors_two_children_is_a_noop():
    assert _apply(remove_unary_junctors(), "and($1 $2)") == "and($1 $2)"


# ---------------------------------------------------------------------------
# remove_dangling_junctors
# ---------------------------------------------------------------------------


def test_remove_dangling_junctors_plain():
    assert _apply(remove_dangling_junctors(), "and($1 and() $2)") == "and($1 $2)"


def test_remove_dangling_junctors_cond_drops_paired_consequent():
    """C++: logical_rules.cc:112-124 -- `cond` clauses are (condition,
    consequent) pairs plus a trailing else; a null (here: childless
    `and`) CONDITION drops its paired consequent too.
    """
    tr = combo_tree(builtin.cond)
    root = tr.begin()
    tr.append_child(root, builtin.logical_and)  # childless condition 1 (dropped)
    tr.append_child(root, argument(1))  # consequent 1 (dropped too)
    tr.append_child(root, argument(2))  # condition 2 (kept)
    tr.append_child(root, argument(3))  # consequent 2 (kept)
    tr.append_child(root, argument(4))  # else clause (kept)
    remove_dangling_junctors()(tr, root)
    assert combo_tree_to_str(tr).strip() == "cond($2 $3 $4)"


# ---------------------------------------------------------------------------
# eval_logical_identities
# ---------------------------------------------------------------------------


def test_eval_logical_identities_and_drops_true_and_childless_or():
    assert _apply(eval_logical_identities(), "and(or() $1 true)") == "and($1)"


def test_eval_logical_identities_and_short_circuits_on_false():
    assert _apply(eval_logical_identities(), "and($1 false $2)") == "false"


def test_eval_logical_identities_or_one_child_promotes_it():
    assert _apply(eval_logical_identities(), "or(and($1 $2))") == "and($1 $2)"


def test_eval_logical_identities_or_short_circuits_on_true():
    assert _apply(eval_logical_identities(), "or($1 true $2)") == "true"


def test_eval_logical_identities_or_drops_false_and_childless_and():
    assert _apply(eval_logical_identities(), "or(and() false $1)") == "or($1)"


# ---------------------------------------------------------------------------
# reduce_nots
# ---------------------------------------------------------------------------


def test_reduce_nots_double_negation():
    assert _apply(reduce_nots(), "not(not($1))") == "$1"


def test_reduce_nots_triple_negation_recurses():
    assert _apply(reduce_nots(), "not(not(not($1)))") == "!$1"


def test_reduce_nots_distributes_over_and():
    assert _apply(reduce_nots(), "not(and($1 $2))") == "or(not($1) not($2))"


def test_reduce_nots_distributes_over_or():
    assert _apply(reduce_nots(), "not(or($1 $2))") == "and(not($1) not($2))"


def test_reduce_nots_argument():
    assert _apply(reduce_nots(), "not($1)") == "!$1"


def test_reduce_nots_negated_argument():
    assert _apply(reduce_nots(), "not(!$1)") == "$1"


def test_reduce_nots_true_and_false():
    assert _apply(reduce_nots(), "not(true)") == "false"
    assert _apply(reduce_nots(), "not(false)") == "true"


def test_reduce_nots_in_place_negate_does_not_corrupt_a_sibling_occurrence():
    """logical_rules.py's own module docstring, item 1 -- pins
    `logical_rules.cc:204`'s in-place mutator (`*it = *it.begin();
    tr.erase_children(it); get_argument(*it).negate();`) through THIS
    port's code path. Builds a genuine subtree-copy sibling (via
    `tree.append_child`'s subtree-copy overload, NOT a second parse of
    "$1" -- a second parse would just be a structurally-independent
    `argument` object regardless of any aliasing bug, proving nothing)
    and confirms it is untouched after negating the ORIGINAL in place.

    Measured to actually differentiate: reverting `tree.py`'s
    `_copy_value` fix (task-3) to the identity function makes the COPY
    corrupt too (`and(!$1 !$1)` instead of the correct `and(!$1 $1)`) --
    see task-4-report.md for the transcript.
    """
    tr = str_to_combo_tree("and(not($1))")
    root = tr.begin()
    not_node = root.first_child
    arg_under_not = not_node.first_child
    copy_node = tr.append_child(root, arg_under_not)
    assert copy_node.data is not arg_under_not.data
    assert copy_node.data.idx == 1

    reduce_nots()(tr, not_node)

    assert combo_tree_to_str(tr).strip() == "and(!$1 $1)"
    assert copy_node.data.idx == 1  # untouched


def test_combo_tree_snapshot_survives_reduce_nots_in_place_negate():
    """logical_rules.py's own module docstring, item 2 -- pins the exact
    mechanism `meta_rules.iterative`/`assum_iterative`'s fixpoint depends
    on: `combo_tree(it)` (their own snapshot constructor, meta_rules.cc:
    130/147) must stay independent of a LATER in-place mutation of the
    LIVE tree it was copied from. Builds that exact snapshot directly
    (not through `iterative` itself -- see this test's own docstring
    below for why) and confirms it is unaffected by `reduce_nots`
    negating the live tree's argument in place afterward.

    Measured to actually differentiate: reverting `tree.py`'s
    `_copy_value` fix makes the snapshot's OWN reported value flip too
    (`not(!$1)` instead of the correct, pre-mutation `not($1)`).

    NOT tested via `iterative`'s own call count: measured separately
    (see task-4-report.md) that `iterative(downwards(reduce_nots()))`
    converges in exactly 2 calls (1 real change + 1 confirming no-op)
    REGARDLESS of whether this aliasing fix is present, for every
    reduce_nots scenario tried -- `reduce_nots`' `is_argument` branch
    always changes `it`'s own vertex TYPE (not -> argument), which
    `equal_subtree_vertex_safe`'s per-node type check
    (`vertex_equal`/`isinstance`) independently detects regardless of
    whether the LEAF's underlying value object happens to be aliased.
    That the call count is not a useful observable for THIS specific
    rule does not mean the underlying snapshot bug is unreachable -- this
    test demonstrates the corruption directly, at its source.
    """
    tr = str_to_combo_tree("not($1)")
    it = tr.begin()
    tmp = combo_tree(it)  # meta_rules.iterative's own `combo_tree(it)` snapshot

    reduce_nots()(tr, it)

    assert combo_tree_to_str(tr).strip() == "!$1"
    assert combo_tree_to_str(tmp).strip() == "not($1)"  # untouched


def test_iterative_over_reduce_nots_converges_correctly():
    """A real, working `iterative(downwards(reduce_nots()))` scenario --
    genuine coverage of the combinator interaction, independent of the
    aliasing angle covered by the two tests above (see the previous
    test's own docstring for the measured, disclosed reason this
    particular case's CALL COUNT does not itself distinguish the fix).
    """
    calls = []

    class _Counting(rule):
        def __init__(self) -> None:
            super().__init__("counting")
            self.inner = downwards(reduce_nots())

        def apply(self, tr: combo_tree, it) -> None:
            calls.append(combo_tree_to_str(tr).strip())
            self.inner(tr, it)

    tr = str_to_combo_tree("and(not(and($1 $2)) $3)")
    iterative(_Counting())(tr, tr.begin())

    assert len(calls) == 2  # one real pass, one confirming no-op
    assert combo_tree_to_str(tr).strip() == "and(or(!$1 !$2) $3)"


# ---------------------------------------------------------------------------
# reduce_and_assumptions / reduce_or_assumptions
#
# Neither rule's own assumption-CONSUMING half has a real user anywhere in
# this port's current scope (see logical_rules.py's own docstring on
# reduce_and_assumptions -- every actual assumption-consuming reduction
# lives in mixed_rules.cc/contin_rules.cc, PLANNED: S14). `_AssumeIdentity`
# below is a test-only stand-in exercising the REAL plumbing
# (insert_assumption, copy_all_assumptions, tr.count/sibling_index/child/
# replace, best-diff selection) end-to-end, the way a real S14 consumer
# eventually would.
# ---------------------------------------------------------------------------


class _AssumeIdentity(rule):
    """Test-only: if `it`'s own value structurally matches one of `tr`'s
    assumptions, collapse `it` to `true`."""

    def __init__(self, name: str = "assume_identity") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it) -> None:
        for a in getattr(tr, "assumptions", []):
            if equal_subtree_vertex_safe(it, a.begin()):
                tr.erase_children(it)
                it.data = builtin.logical_true
                return


def test_reduce_and_assumptions_default_reduction_finds_no_improvement():
    assert _apply(reduce_and_assumptions(), "and($1 $2 $3)") == "and($1 $2 $3)"


def test_reduce_and_assumptions_is_a_noop_off_a_non_and_node():
    assert _apply(reduce_and_assumptions(), "or($1 $2)") == "or($1 $2)"


def test_reduce_and_assumptions_picks_the_child_that_shrinks_the_most():
    """Two identical `and($1 $2)` children: reducing EITHER one assuming
    the other (via `_AssumeIdentity`) collapses it to `true` (3 nodes ->
    1), a real size improvement -- `reduce_and_assumptions` picks the
    FIRST such child (tie-break via strict `<`) and replaces it in the
    live tree.
    """
    tr = str_to_combo_tree("and(and($1 $2) and($1 $2))")
    reduce_and_assumptions(_AssumeIdentity())(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "and(true and($1 $2))"


def test_reduce_or_assumptions_default_reduction_is_structurally_inert():
    """The default `_reduction=self` bottoms out immediately (its own
    `*it==OR` guard never matches `not`/argument positions) -- net
    structurally unchanged, matching C++ measured behaviour.
    """
    assert _apply(reduce_or_assumptions(), "or($1 $2)") == "or($1 $2)"


def test_reduce_or_assumptions_is_a_noop_off_a_non_or_node():
    assert _apply(reduce_or_assumptions(), "and($1 $2)") == "and($1 $2)"


def test_reduce_or_assumptions_raises_on_a_childless_or():
    tr = combo_tree(builtin.logical_or)
    with pytest.raises(AssertionError):
        reduce_or_assumptions()(tr, tr.begin())


def test_reduce_or_assumptions_with_reduce_and_assumptions_reduction():
    inner = reduce_and_assumptions()
    assert _apply(reduce_or_assumptions(inner), "or($1 $2 $3)") == "or($1 $2 $3)"


# ---------------------------------------------------------------------------
# reduce_ors / reduce_ands
# ---------------------------------------------------------------------------


def test_reduce_ors_removes_a_duplicate_clause():
    assert _apply(reduce_ors(), "or($1 $1 $2)") == "or($1 $2)"


def test_reduce_ors_removes_a_dominated_clause():
    assert _apply(reduce_ors(), "or(and($1 $2) $1)") == "$1"


def test_reduce_ors_implication_of_complementary_conjuncts():
    assert _apply(reduce_ors(), "or(and($1 $3) and(!$1 $3))") == "or($3 $3)"


def test_reduce_ors_is_a_noop_off_a_non_or_node():
    assert _apply(reduce_ors(), "and($1 $2)") == "and($1 $2)"


def test_reduce_ors_stale_i1_bound_regression():
    """Task 4 review, Critical #2. C++'s inner loop bound is `for (c2 =
    f.begin(); c2 != c1; ++c2)` (logical_rules.cc:311) -- bounded by the
    LIVE position of `c1`, which the loop body's own clause deletions can
    shift left. A first version of this port captured that bound as a
    plain Python int (`i1`) once, at the top of the outer loop, and never
    re-derived it -- pre-fix, this exact input raised `IndexError` once
    `f` shrank below the stale bound; C++ returns `true`.
    """
    assert _apply(reduce_ors(), "or(!$1 or(and($2 !$1 !$2)) !$2 $2)") == "true"


def test_reduce_ands_removes_a_duplicate_clause():
    assert _apply(reduce_ands(), "and($1 $1 $2)") == "and($1 $2)"


def test_reduce_ands_removes_a_dominating_clause():
    assert _apply(reduce_ands(), "and(or($1 $2) $1)") == "$1"


def test_reduce_ands_implication_of_complementary_disjuncts():
    assert _apply(reduce_ands(), "and(or($1 $3) or(!$1 $3))") == "and($3 $3)"


def test_reduce_ands_common_factor_extraction():
    assert _apply(reduce_ands(), "and(or($1 $2) or($1 $3))") == "or(and($2 $3) $1)"


def test_reduce_ands_is_a_noop_off_a_non_and_node():
    assert _apply(reduce_ands(), "or($1 $2)") == "or($1 $2)"


def test_reduce_ands_stale_i1_bound_regression():
    """Task 4 review, Critical #2 -- the AND-dual of
    `test_reduce_ors_stale_i1_bound_regression` above, same mechanism
    (logical_rules.cc:373), reached this time via `reduce_ands` directly.
    Pre-fix: `IndexError`. C++/fixed: `false`.
    """
    assert _apply(reduce_ands(), "and($1 and(or($2 $1 $2)) $2 !$2)") == "false"


# ---------------------------------------------------------------------------
# subtree_to_enf -- reduce_to_enf's own state machine
# ---------------------------------------------------------------------------


def test_enf_and_arg_negarg_at_the_root_is_left_as_is():
    """Measured, not assumed: `and($1 not($1))` applied bare AT THE ROOT
    is NOT converted to `false`. The ONLY contradiction check `reduce_and`
    runs across its own DIRECT children is `consistent(handle_set)`
    (`is_complement`), and `is_complement` requires BOTH sides to already
    be bare `argument` vertices ($N vs -$N) -- it never looks inside an
    explicit `not(...)` wrapper. `$1` and `not($1)` are an `argument` and
    a `builtin.logical_not` respectively, so `is_complement` returns False
    regardless of sign, and nothing else in `reduce_and`'s straight-line
    processing compares two children against each other for this. (A
    prior `reduce_nots()` pass, run before `subtree_to_enf` in real
    `logical_reduction` pipelines, would have turned `not($1)` into the
    bare `!$1` `test_enf_and_bare_complement_is_a_real_contradiction`
    below DOES catch.)
    """
    assert _enf("and($1 not($1))") == "and(not($1) $1)"


def test_enf_and_bare_complement_is_a_real_contradiction():
    """Contrasts with the test above: two BARE `argument` vertices of
    opposite sign ($1, !$1 -- not `not(...)`-wrapped) ARE what
    `is_complement` checks, so `consistent(handle_set)` fires here.
    """
    assert _enf("and($1 !$1)") == "false"


def test_enf_removes_duplicate_children():
    assert _enf("and($1 $1 $2)") == "and($1 $2)"
    assert _enf("or($1 $1 $2)") == "or($1 $2)"


def test_enf_duplicate_removal_does_not_conflate_contin_with_logical_true():
    """S14 Task 0 (CARRY-FORWARD.md section 2.5) pin: `reduce_to_enf.reduce`'s
    duplicate-child removal (logical_rules.cc:604) called bare
    `tr.equal_subtree(sib1, sib2)` -- the no-predicate overload, whose default
    comparator is Python's `==`. `builtin.logical_true` is ordinal 1
    (vertex.py), so `contin_t(1.0) == builtin.logical_true` is `True` in bare
    Python even though C++'s real `boost::variant` never agrees across held
    alternatives (`std::equal_to<vertex>`, C++'s own default, discriminates by
    alternative before ever comparing a value).

    MEASURED, not assumed, via a deliberate revert of this task's own fix (the
    bare `tr.equal_subtree(sib1, sib2)` this test pins against not
    regressing): `_enf_raw("and(1.0 true)")` returns `"and(1)"` under the old
    bare comparison -- `logical_true` is erased as a false "duplicate" of
    `1.0` -- and correctly returns `"and(1 true)"` once the call passes
    `vertex_equal` as its predicate. `sort_on_subtrees` (called earlier in
    `reduce_to_enf.__call__`, using the already-type-safe `vertex_rank`/
    `vertex_less`) places the `contin_t` child before the `builtin` child --
    `vertex_rank`'s ordering, not raw ordinal value -- which is why the
    expected string below is `and(1 true)`, `contin_t` first.
    """
    assert _enf_raw("and(1.0 true)") == "and(1 true)"


def test_enf_duplicate_removal_does_not_conflate_contin_with_logical_false():
    """Same shape as the test above, over `builtin.logical_false` (ordinal 2,
    vertex.py) instead of `logical_true` (ordinal 1) -- the task brief's own
    suggested second pair, "0.0 vs logical_false", does NOT actually collide
    under this port's real (C++-transcribed) ordinals: `null_vertex` is
    ordinal 0, not `logical_false`, so `0.0 == builtin.logical_false` is
    already `False` even under the OLD bare `==` and cannot discriminate the
    fix. `contin_t(2.0)` is `logical_false`'s real same-ordinal collision
    partner (measured: `_enf_raw("or(2.0 false)")` returns `"or(2)"` under the
    reverted bare comparison, erasing `logical_false` as a false duplicate of
    `2.0`), so that is the pair pinned here instead, over `logical_or` rather
    than `logical_and` for path coverage of both duplicate-removal callers.
    """
    assert _enf_raw("or(2.0 false)") == "or(2 false)"


def test_enf_duplicate_removal_contin_against_argument_was_never_at_risk():
    """Completeness case the task brief also names ("a contin value against
    an argument") -- included for coverage, NOT a discriminating mutation
    test: `argument.__eq__` (argument.py) already returns `NotImplemented`
    for a non-`argument` other side, so bare `1.0 == argument(1)` is `False`
    regardless of this task's fix (confirmed identical under both the
    reverted bare comparison and the fixed one -- `vertex_equal`'s own
    docstring names this as the one alternative pair that was never actually
    at risk). Kept as a regression pin, not folded into the two tests above.
    """
    assert _enf_raw("and($1 1.0)") == "and(1 $1)"


def test_enf_and_or_same_argument_reorders_but_cannot_shrink():
    """`and($1 or($1 $2))`: `$1` DOMINATES the `or` branch structurally,
    but `reduce_or`'s own algorithm has no direct child-vs-dominant
    pruning step (only `reduce_and`'s `res`-based common-factor promotion
    looks at GRANDCHILDREN) -- measured, not assumed, that the real
    library leaves this un-shrunk beyond reordering.
    """
    assert _enf("and($1 or($1 $2))") == "and(or($1 $2) $1)"


def test_enf_and_or_negated_argument_command_conflict_kept_as_is():
    assert _enf("and($1 or(not($1) $2))") == "and(or(not($1) $2) $1)"


def test_enf_or_dominant_and_children_kept():
    assert _enf("or(and($1 $2) and($1 $3))") == "or(and($1 $2) and($1 $3))"


def test_enf_nested_and_or_and_promotes_common_factor():
    assert _enf("and($1 or($2 and($1 $3)))") == "and(or($3 $2) $1)"


def test_enf_command_promotion_from_or_sibling():
    assert _enf("or($1 and(not($1) $2))") == "or(and(not($1) $2) $1)"


def test_enf_and_cut_unwraps_single_child_or():
    assert _enf("and($1 and(or($2 $3)))") == "and(or($2 $3) $1)"


def test_enf_all_true_becomes_true():
    assert _enf("and(true true)") == "true"


def test_enf_dominant_command_conflict_keeps_both_branches():
    assert _enf("or($1 and(not($1) not($1)))") == "or(and(not($1)) $1)"


def test_push_back_negated_arguments_does_not_mutate_the_source():
    """logical_rules.py's own module docstring, item 3 --
    `push_back_negated_arguments` (logical_rules.h:230, reached from
    `reduce_and` at logical_rules.cc:643) does `combo_tree(sib)` then
    negates the COPY's own root. Pins that the source (`$1` under the
    FIRST `and()` branch, inserted into `command` via
    `build_subtree_sets_upwards` because it is the OR's own single-child
    sibling) survives the whole `reduce_and` pass over the SECOND branch,
    which is exactly where `push_back_negated_arguments` is reached.
    """
    tr = str_to_combo_tree("or(and($1) and(not($1) $2))")
    first_and = tr.begin().first_child
    original_arg_node = first_and.first_child
    assert original_arg_node.data.idx == 1

    subtree_to_enf()(tr, tr.begin())

    assert original_arg_node.data.idx == 1


def test_enf_is_a_noop_on_a_non_logical_operator():
    assert _enf("$1") == "$1"


# ---------------------------------------------------------------------------
# Task 4 review, Important -- 5 load-bearing branches inside reduce_and that
# survived mutation testing (26 mutants over the original 74 tests: 16
# survived, 5 of them behaviour-changing) because nothing exercised them.
# Each input below was found by instrumenting the Python port with a print()
# at the target line and searching by hand until each one fired, then
# verified against the real library via the probe. This is precisely the
# metta-moses failure mode the task brief warns about: correct code with no
# test that would catch its removal.
# ---------------------------------------------------------------------------


def test_enf_one_subsumption_delete():
    """`reduce_and`'s direct 1-subsumption check
    (`has_empty_intersection(current's children, command)`) deletes the
    WHOLE `and(...)` branch when one of its OWN direct children matches a
    `command` entry verbatim -- here, `and($1)` (a single-child AND
    sibling under the OR) inserts `$1` into `command` for the OTHER
    branch, and `and($1 $2)` (which directly contains `$1`) is deleted
    entirely rather than merely simplified.
    """
    assert _enf("or(and($1) and($1 $2))") == "or(and($1))"


def test_enf_trailing_zero_subsumption_disconnect():
    """The do-while loop's OWN processing (here: `or_cut`'s "childless
    logical_and under a single-child wrapper" delete branch, applied
    twice as the emptying propagates outward) can leave `current` fully
    childless only once the loop has otherwise converged -- the TRAILING
    `if tr.is_childless(current): return Disconnect`, distinct from the
    EARLIER childless check that runs before the loop even starts.

    Uses `_enf_raw` (no pre-canonicalization), not `_enf`: this exact
    redundant-nesting shape is precisely what `level()`'s own flattening
    collapses away first, which would silently route around the branch
    under test -- see `_enf_raw`'s own docstring.
    """
    assert _enf_raw("and(and(and()))") == "true"


def test_enf_reduce_or_disconnect_propagation():
    """`reduce_or` returns `Disconnect` immediately, for the WHOLE `or`,
    the moment any one child reduces to a tautology -- reuses the exact
    nested-empty-`and` mechanism from the test above, this time as an OR
    sibling of `$1`. Uses `_enf_raw` for the same reason as that test.
    """
    assert _enf_raw("or($1 and(and(and())))") == "true"


def test_enf_and_cut_after_res_promotion():
    """`res` (the grandchild-level common factor `$1`, shared by `and($1
    $2)` and `and($1 !$1)`... -- concretely: `and($2 $1)` and `and($2
    !$1)` sharing `$2`) is promoted up to `current`, and `and_cut(child)`
    then unwraps the now-single-child `and($1)`/`and(!$1)` wrappers it
    leaves behind (its own return value is ignored here -- a SEPARATE
    branch, not this one, is the only place that value matters).

    NOT reachable by merely finding an input where this LINE executes:
    an earlier candidate (`and(or(and($1 $2) and($1 $3)))`) reaches it
    but does not distinguish removing it, because the SUBSEQUENT
    `reprocess-all-kids reset` (below) re-examines the same node and its
    OWN "res empty" branch calls the identical `and_cut(child)` again on
    the second pass, silently redoing the cleanup. Found by running the
    Python port against a copy of itself with only this call deleted,
    over randomly generated trees, and keeping the shortest genuine
    divergence.
    """
    assert _enf("and(or($1 $2) or(and($2 $1) and($2 !$1)))") == "and(or(!$1 $1) or($1 $2) $2)"


def test_enf_reprocess_all_kids_reset():
    """`child = current.first_child` (resetting the child loop after a
    `res` promotion enlarges `current`'s own guard set) -- distinct from
    `and_cut(child)` above; found the same way (differential testing the
    port against itself with only this line's effect removed, i.e.
    advancing past `child` instead of restarting from the top). A first,
    shorter candidate reached this same line but its mutation was
    silently absorbed elsewhere, the same trap `test_enf_and_cut_after_
    res_promotion` documents -- this one was kept specifically because it
    genuinely distinguishes the reset from a mere advance.
    """
    assert (
        _enf("and(!$1 and(or(or($3 $2) and($3 !$2)) !$2 or(and($3 !$3) and($3 !$1))))")
        == "and(or($3 $2 $3) !$1 !$2 $3)"
    )


# ---------------------------------------------------------------------------
# subtree_set / the algorithm.h merge helpers `reduce_to_enf` depends on
# ---------------------------------------------------------------------------


def _leaf(idx: int):
    return str_to_combo_tree(f"${idx}").begin()


def test_subtree_set_dedups_structurally_equal_nodes():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less

    s = subtree_set(lexicographic_subtree_less)
    a = _leaf(1)
    b = _leaf(1)  # structurally equal, different object
    c = _leaf(2)
    s.insert(a)
    s.insert(b)
    s.insert(c)
    assert len(s) == 2
    assert s.find(b) is a  # the FIRST-inserted structurally-equal node wins


def test_subtree_set_erase_and_contains():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less

    s = subtree_set(lexicographic_subtree_less)
    a, b = _leaf(1), _leaf(2)
    s.insert(a)
    s.insert(b)
    assert a in s and b in s
    s.erase(a)
    assert a not in s
    assert len(s) == 1
    s.erase(a)  # no-op, matches std::set::erase's permissive contract
    assert len(s) == 1


def test_is_sorted_matches_std_is_sorted_semantics():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less as comp

    a, b, c = _leaf(1), _leaf(2), _leaf(3)
    assert is_sorted([a, b, c], comp) or is_sorted([c, b, a], comp)
    assert not is_sorted([b, a, c], comp) or not is_sorted([a, c, b], comp)


def test_erase_set_intersection_erases_common_elements_only():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less as comp

    a, b, c = _leaf(1), _leaf(2), _leaf(3)
    seq1 = sorted([a, b, c], key=lambda n: n.data.idx)
    seq2 = sorted([b], key=lambda n: n.data.idx)
    erased = []
    erase_set_intersection(erased.append, seq1, seq2, comp)
    assert erased == [b]


def test_insert_set_complement_appends_missing_elements():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less as comp

    a, b = _leaf(1), _leaf(2)
    seq1 = [a]
    seq2 = [a, b]
    calls = []
    insert_set_complement(lambda pos, val: calls.append((pos, val)), seq1, seq2, comp)
    assert calls == [(None, b)]  # b is missing from seq1; a already present


def test_has_empty_intersection():
    from pymoses.comboreduct.combo.vertex import lexicographic_subtree_less as comp

    a, b, c = _leaf(1), _leaf(2), _leaf(3)
    assert has_empty_intersection([a], [b], comp)
    assert not has_empty_intersection([a, b], [b, c], comp)


# ---------------------------------------------------------------------------
# simplify_predicates / reduce_remove_subtree_equal_tt -- disclosed S14/
# Task-5 forward references; see this module's own top docstring.
# ---------------------------------------------------------------------------


def test_simplify_predicates_is_a_noop_when_it_is_not_a_predicate():
    assert _apply(simplify_predicates(3, set()), "and($1 $2)") == "and($1 $2)"


def test_simplify_predicates_returns_early_for_a_bare_argument_operand():
    assert _apply(simplify_predicates(3, set()), "0<($1)") == "0<($1)"


def test_simplify_predicates_runs_its_contin_mixed_tail_for_a_real_predicate():
    """This test used to assert ``pytest.raises(NotImplementedError)``: the
    ``contin_reduce``/``mixed_reduce`` tail (logical_rules.cc:76-81) was
    stubbed. It is ported now, so the tail runs for real.

    Three shapes, all measured against classic
    (``tools/cpp_probe/mixed_predicate_probe.cc``, fixture
    ``tests/fixtures/mixed_predicate.json``): a predicate whose contin
    argument is already reduced comes back unchanged; one that is not is
    rewritten by ``contin_reduce`` (``0<(*(2 $1))`` -> ``0<($1)``); and a
    constant predicate is folded away by ``mixed_reduce`` (``0<(1)`` ->
    ``true``).
    """
    assert _apply(simplify_predicates(2, set()), "0<(+($1 $2))") == "0<(+($1 $2))"
    assert _apply(simplify_predicates(2, set()), "0<(*(2 $1))") == "0<($1)"
    assert _apply(simplify_predicates(2, set()), "0<(1)") == "true"


def test_reduce_remove_subtree_equal_tt_is_a_noop_when_a_predicate_is_present():
    assert _apply(reduce_remove_subtree_equal_tt(), "and(0<($1) $2)") == "and(0<($1) $2)"


def test_reduce_remove_subtree_equal_tt_removes_a_redundant_tautology():
    """Task 5 landed `complete_truth_table` and `clean_reduce`, unblocking
    this rule for real. MEASURED against the real library (`tools/
    cpp_probe/logical_rules_probe.cc`'s `rrset_removes_a_redundant_
    argument`): `or($2 not($2))` is a tautology, so `and($1 or($2 not($2)))`
    has the SAME truth table as `and($1)` alone -- the redundant subtree
    is amputated.
    """
    assert (
        _apply(reduce_remove_subtree_equal_tt(), "and($1 or($2 not($2)))")
        == "and($1)"
    )


def test_reduce_remove_subtree_equal_tt_keeps_a_truth_table_changing_subtree():
    """MEASURED (`rrset_keeps_tt_changing_subtree`): `and($1 $2)` needs
    BOTH children -- removing either changes the truth table, so nothing
    is amputated.
    """
    assert _apply(reduce_remove_subtree_equal_tt(), "and($1 $2)") == "and($1 $2)"


def test_reduce_remove_subtree_equal_tt_preserves_the_truth_table():
    """Truth-table PRESERVATION is this rule's whole point (its own C++
    doc comment: "check if the truth table is the same after removal of
    any subtree") -- verified via the already-ported `boolean_
    interpreter`, independent of the exact string this rule reduces to.
    """
    tr = str_to_combo_tree("and($1 or($2 not($2)) or($3 $1))")
    before = combo_tree(tr.begin())
    reduce_remove_subtree_equal_tt()(tr, tr.begin())
    for bits in itertools.product((False, True), repeat=3):
        inputs = [builtin.logical_true if b else builtin.logical_false for b in bits]
        assert vertex_equal(
            boolean_interpreter(list(inputs))(before), boolean_interpreter(list(inputs))(tr)
        )


def test_reduce_remove_subtree_equal_tt_non_root_it_never_reaches_a_sibling():
    """CORRECTED (Task 5 fix round): this test used to be named
    `..._walks_past_its_own_subtree` and claimed, on the strength of an
    unverified reading of the `sibling_iterator`->`pre_it` conversion,
    that real C++ walks past `it`'s own subtree into structurally later
    siblings (here, `$3`) -- see `reduce_remove_subtree_equal_tt.apply`'s
    own docstring for why that reading was wrong. This exact tree cannot
    actually tell the two hypotheses apart: `$3` survives either way,
    whether because it was never visited (correct, bounded) or because it
    was visited but its removal changes the truth table (the old,
    incorrect story) -- see `..._bounded_to_its_own_subtree_not_the_whole_
    tree` below for a tree that DOES distinguish them. Kept here, renamed
    and re-explained, as a same-answer regression pin plus a `$3`-is-a-
    truth-table-load-bearing-sibling sanity check for the corrected
    boundary logic (`it` is the inner `and(...)`; `$3` is the outer
    `or`'s second child, a sibling of `it`, never entered by the walk).
    """
    tr = str_to_combo_tree("or(and($1 or($2 not($2))) $3)")
    root = tr.begin()
    inner_and = root.first_child
    reduce_remove_subtree_equal_tt()(tr, inner_and)
    assert combo_tree_to_str(tr).strip() == "or(and($1) $3)"


def test_reduce_remove_subtree_equal_tt_bounded_to_its_own_subtree_not_the_whole_tree():
    """The actual distinguishing case for the boundary bug fixed this
    task round (see `reduce_remove_subtree_equal_tt.apply`'s own
    docstring for the full trace). MEASURED via a bounded probe against
    the real compiled `libcomboreduct.so`
    (`( ulimit -v 2097152; timeout 300 <probe> )`), calling
    `reduce_remove_subtree_equal_tt()(tr, it)` directly (not through
    `logical_reduce`) with `it` the `and` node of `or(and($1 $2) $1 $1)`:
    real C++ visits only `it`'s own two children (`$1`, `$2`) and leaves
    the outer `or`'s other two `$1` siblings untouched, giving
    `or(and($1) $1 $1)` (`$2` alone is redundant -- `$1 or ($1 and $2)`
    reduces to `$1` by absorption, same as the original tree's truth
    table). Before this fix, this port's then-unbounded walk continued
    past `it`'s subtree into those two siblings and amputated both
    (nothing downstream of `it`'s own subtree is truth-table load-bearing
    in isolation once the whole tree already reduces to `$1`), giving the
    wrong `or(and($1))`.
    """
    tr = str_to_combo_tree("or(and($1 $2) $1 $1)")
    root = tr.begin()
    inner_and = root.first_child
    reduce_remove_subtree_equal_tt()(tr, inner_and)
    assert combo_tree_to_str(tr).strip() == "or(and($1) $1 $1)"


def test_reduce_remove_subtree_equal_tt_raises_on_a_childless_not_via_its_own_predicate_scan():
    """A THIRD real call site for `is_predicate`'s inherited childless-
    `logical_not` UB (see that function's own docstring, updated by Task
    5): `reduce_remove_subtree_equal_tt`'s own guard scan walks EVERY
    node in the tree, so a malformed childless `not` anywhere reaches it
    -- not built via `str_to_combo_tree` (its grammar cannot express a
    childless `not`), same as the other two documented call sites.
    """
    tr = combo_tree(builtin.logical_and)
    root = tr.begin()
    tr.append_child(root, argument(1))
    tr.append_child(root, builtin.logical_not)  # childless

    with pytest.raises(AttributeError):
        reduce_remove_subtree_equal_tt()(tr, root)


@pytest.mark.parametrize(
    "tree_str, expected",
    [("or(and($1 $2) $1)", "or($1)"), ("and(or($1 $2) $1)", "and($1)")],
)
def test_reduce_remove_subtree_equal_tt_never_leaves_a_childless_root_junctor(tree_str, expected):
    """Regression (BACKLOG section 4 item 2): cutting the last child of the
    root junctor used to leave a childless `or()`/`and()`, whose truth
    table check asserted in `builtin_to_bool`. That cut now counts as
    truth-table-changing (DIVERGENCE: C++ asserts here), so the child stays.
    """
    tr = str_to_combo_tree(tree_str)
    before = complete_truth_table(str_to_combo_tree(tree_str), 2)
    reduce_remove_subtree_equal_tt()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == expected
    assert before.same_complete_truth_table(tr)


@pytest.mark.parametrize(
    "tree_str",
    [
        "or(and($1 $2) $1)",
        "and(or($1 $2) $1)",
        "and(or(or(and(!$4 !$1) or(!$3 !$1)) and(or(!$3 $3 $1) or($1 $3 $3) or($4 $2))"
        " and(!$2 or(!$3 $3 $4) $3)) $1)",
    ],
)
def test_logical_reduce_effort_3_survives_the_childless_root_cut(tree_str):
    """Same regression through full `logical_reduce(3, .)` (the fuzzed
    tree crashed there before the fix); the result keeps the input's
    complete truth table and reduces to `$1`.
    """
    tr = str_to_combo_tree(tree_str)
    before = complete_truth_table(str_to_combo_tree(tree_str), 4)
    logical_reduce(3, tr, ignore_ops=set())
    assert combo_tree_to_str(tr).strip() == "$1"
    assert before.same_complete_truth_table(tr)


# ---------------------------------------------------------------------------
# BooleanReductUTest.cxxtest -- MOVED to test_logical_reduction.py (Task 5).
#
# Task 4's version of this section ported test_simplify1..6 against a
# test-local, UNTYPED `_boolean_reduce_medium` approximation of effort 2,
# because `logical_reduction.py` (this file's own effort-assembly
# counterpart) and a real typed `downwards` did not exist yet -- the real
# cxxtest calls `logical_reduce(3, tr, ignore_ops)` (effort 3), which was
# doubly blocked. Task 5 landed both `logical_reduction.py` and the typed
# `downwards` slice, so `test_simplify1..6` now live in `test_logical_
# reduction.py`, calling the REAL `logical_reduce(3, ...)` (byte-identical
# to real C++, verified directly against `tools/cpp_probe/logical_rules_
# probe.cc`) -- see that module for the faithful cxxtest port (same test
# names, same two assertions: strict size decrease + truth-table
# preservation). Removed from here rather than kept as a redundant,
# now-superseded approximation.
# ---------------------------------------------------------------------------
