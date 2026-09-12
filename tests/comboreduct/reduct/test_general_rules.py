"""Tests for pymoses/comboreduct/reduct/general_rules.py -- the
type-independent reduction rules (`level`, `eval_constants`,
`reorder_commutative`, `remove_null_vertices`, `remove_all_assumptions`)
and the vertex.h algebraic-property predicates they need.

C++: moses/comboreduct/reduct/general_rules.{h,cc}.

Every literal expected tree string below is copied verbatim from a REAL
run of `tools/cpp_probe/general_rules_probe.cc` against $CLASSIC's
compiled `libcomboreduct.so` (bounded:
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/general_rules_probe
)`), not hand-derived from the C++ source comments alone -- see that file
and general_rules.py's own module docstring for the full transcript, and
task-3-report.md for the annotated derivation of each non-obvious case
(especially the `eval_constants` partition/reparent cases, which pin
`tree.partition`'s measured non-stable behaviour at the rule level too).
"""

import pytest

from pymoses.comboreduct.combo.assumption import insert_assumption
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.reduct.general_rules import (
    eval_constants,
    is_associative,
    is_boolean,
    is_commutative,
    is_constant,
    level,
    remove_all_assumptions,
    remove_null_vertices,
    reorder_commutative,
)

# ---------------------------------------------------------------------------
# The algebraic-property predicates
# ---------------------------------------------------------------------------


def test_is_associative_true_for_and_or_plus_times():
    assert is_associative(builtin.logical_and)
    assert is_associative(builtin.logical_or)
    assert is_associative(builtin.plus)
    assert is_associative(builtin.times)


def test_is_associative_false_for_everything_else():
    assert not is_associative(builtin.logical_not)
    assert not is_associative(builtin.div)
    assert not is_associative(builtin.logical_true)


def test_is_associative_does_not_confuse_a_contin_with_a_matching_ordinal():
    # builtin.plus == 6; a bare `==` would let 6.0 spuriously match it.
    # "type before value" (CLAUDE.md) -- vertex_equal must guard this.
    assert builtin.plus == 6
    assert not is_associative(6.0)
    assert not is_commutative(7.0)  # builtin.times == 7


def test_is_commutative_true_for_and_or_plus_times():
    assert is_commutative(builtin.logical_and)
    assert is_commutative(builtin.logical_or)
    assert is_commutative(builtin.plus)
    assert is_commutative(builtin.times)


def test_is_commutative_false_for_non_commutative_operators():
    assert not is_commutative(builtin.logical_not)
    assert not is_commutative(builtin.div)


def test_is_boolean():
    assert is_boolean(builtin.logical_true)
    assert is_boolean(builtin.logical_false)
    assert not is_boolean(builtin.logical_and)
    assert not is_boolean(1.0)


def test_is_constant_true_for_booleans_and_contin():
    assert is_constant(builtin.logical_true)
    assert is_constant(builtin.logical_false)
    assert is_constant(3.5)


def test_is_constant_false_for_an_argument_or_operator():
    from pymoses.comboreduct.combo.argument import argument

    assert not is_constant(argument(1))
    assert not is_constant(builtin.logical_and)


# ---------------------------------------------------------------------------
# level
# ---------------------------------------------------------------------------


def _run(prog, r):
    tr = str_to_combo_tree(prog)
    r(tr, tr.begin())
    return combo_tree_to_str(tr)


def test_level_flattens_one_level_of_nested_and():
    assert _run("and($1 and($2 $3) $4)", level()) == "and($1 $2 $3 $4) "


def test_level_flattens_recursively_in_one_pass():
    # C++: general_rules_probe.cc "level_flatten_deep"
    assert _run("and($1 and($2 and($3 $4)))", level()) == "and($1 $2 $3 $4) "


def test_level_flattens_recursively_when_left_nested_too():
    # C++: general_rules_probe.cc "level_flatten_deep_left_nested" --
    # right-nesting alone (the test above) does not exercise the
    # re-derivation of `sib` after `tr.erase(tr.flatten(sib))`
    # meaningfully differently from a single flatten, since the
    # newly-promoted node is always encountered "for free" at the SAME
    # position; a LEFT-nested tree specifically exercises that the walk
    # continues correctly (task-3 review, Important).
    assert _run("and($1 and(and($2 $3) $4))", level()) == "and($1 $2 $3 $4) "


def test_level_does_not_flatten_a_different_operator():
    assert _run("and($1 or($2 $3))", level()) == "and($1 or($2 $3)) "


def test_level_flattens_plus_too():
    assert _run("plus(1 plus(2 3) 4)", level()) == "+(1 2 3 4) "


# ---------------------------------------------------------------------------
# eval_constants
# ---------------------------------------------------------------------------


def test_eval_constants_folds_an_all_constant_and():
    assert _run("and(true false)", eval_constants()) == "false "


def test_eval_constants_folds_an_all_constant_plus():
    assert _run("plus(1 2 3)", eval_constants()) == "6 "


def test_eval_constants_mixed_and_moves_the_folded_constant_to_the_end():
    # partition's swap-scan (NOT a stable partition) puts the two
    # constants (true, false) first, then evaluates just that pair
    # (and(true,false) = false) and appends it after the surviving
    # variables, in the ORDER partition itself produced them ($2 $1).
    assert _run("and($1 true $2 false)", eval_constants()) == "and($2 $1 false) "


def test_eval_constants_mixed_plus_folds_the_constants_only():
    assert _run("plus(2 $1 1)", eval_constants()) == "+($1 3) "


def test_eval_constants_mixed_plus_interleaved_pins_partition_non_stability():
    # The case that most directly exercises tree.partition's real,
    # non-stable swap scan through eval_constants: THREE constants (2,3,1)
    # interleaved with two variables ($1,$2) at non-adjacent positions.
    assert _run("plus(2 $1 3 $2 1)", eval_constants()) == "+($2 $1 6) "


def test_eval_constants_single_constant_that_is_the_only_child_folds():
    assert _run("and(true)", eval_constants()) == "true "


def test_eval_constants_single_constant_with_siblings_is_a_noop():
    # n_consts == 1 and NOT has_one_child -- the (n_consts==1 &&
    # has_one_child) escape hatch does not apply, so this returns early.
    assert _run("and(true $1)", eval_constants()) == "and(true $1) "


def test_eval_constants_evaluates_a_non_associative_operator():
    assert _run("not(true)", eval_constants()) == "false "


def test_eval_constants_greater_than_zero():
    assert _run("0<(3)", eval_constants()) == "true "


def test_eval_constants_no_constants_is_a_noop():
    assert _run("and($1 $2)", eval_constants()) == "and($1 $2) "


def test_eval_constants_childless_argument_is_a_noop():
    # is_indefinite_object is always False in this port (no such vertex
    # alternative) -- a childless, non-indefinite-object node is left
    # completely untouched.
    assert _run("$1", eval_constants()) == "$1 "


# ---------------------------------------------------------------------------
# reorder_commutative
# ---------------------------------------------------------------------------


def test_reorder_commutative_sorts_positive_arguments_ascending():
    assert _run("and($2 $1 $3)", reorder_commutative()) == "and($1 $2 $3) "


def test_reorder_commutative_orders_a_literal_and_its_own_negation_pinned():
    """REQUIRED test (task-3-brief.md): clause/argument ordering for a
    set containing both `$1` and `!$1` -- the case a natural (signed)
    sort gets wrong. `argument.__lt__`'s absolute-value order
    (docs/CARRY-FORWARD.md §4.4) puts the NEGATED one first: `!$1` before
    `$1`, not the other way around and not grouped by sign. Measured
    directly (general_rules_probe.cc's `reorder_commutative_arg_and_negation`):
    `and($1 !$1)` (the ARGUMENT of idx=1 given BEFORE its negation in the
    source) reorders to `and(!$1 $1)`.
    """
    assert _run("and($1 !$1)", reorder_commutative()) == "and(!$1 $1) "
    # And the reverse input order is already correct -- a stable no-op:
    assert _run("and(!$1 $1)", reorder_commutative()) == "and(!$1 $1) "


def test_reorder_commutative_mixed_signs_multiple_arguments():
    # Full "-1,1,-2,2" pattern (docs/CARRY-FORWARD.md §4.4) with two
    # distinct arguments, each represented by both signs.
    assert _run("and($2 !$1 $1 !$2)", reorder_commutative()) == "and(!$1 $1 !$2 $2) "


def test_reorder_commutative_or_sorts_too_not_just_and():
    assert _run("or($3 $1 $2)", reorder_commutative()) == "or($1 $2 $3) "


def test_reorder_commutative_ranks_a_builtin_subtree_before_an_argument_leaf():
    # vertex_rank: builtin (rank 2) sorts before argument (rank 3) --
    # verified directly against the real 14-way boost::variant declaration
    # order (vertex.py's own note); this is the one relative-order fact
    # from that ordering this port's boolean-only corpus actually needs.
    assert _run("and(or($1 $2) $1)", reorder_commutative()) == "and(or($1 $2) $1) "


def test_reorder_commutative_two_negatives_only_sorts_by_absolute_value():
    assert _run("and(!$2 !$1)", reorder_commutative()) == "and(!$1 !$2) "


def test_reorder_commutative_does_not_touch_a_non_commutative_operator():
    # `not($1)` alone does not distinguish the `is_commutative` guard
    # actually firing from sorting a single-child list (a no-op either
    # way) -- a multi-child non-commutative operator is needed to prove
    # the guard has an effect. `not(...)` normally takes exactly one
    # child in valid combo; built directly (not via str_to_combo_tree)
    # the same way general_rules_probe.cc's own
    # "reorder_commutative_multi_child_non_commutative" case does, since
    # the guard itself (`is_commutative(*it)`) is arity-blind.
    from pymoses.comboreduct.combo.argument import argument
    from pymoses.comboreduct.combo.vertex import combo_tree

    tr = combo_tree(builtin.logical_not)
    root = tr.begin()
    tr.append_child(root, argument(2))
    tr.append_child(root, argument(1))
    reorder_commutative()(tr, root)
    assert combo_tree_to_str(tr) == "not($2 $1) "


# ---------------------------------------------------------------------------
# remove_null_vertices
# ---------------------------------------------------------------------------


def test_remove_null_vertices_plain_operator():
    from pymoses.comboreduct.combo.argument import argument
    from pymoses.comboreduct.combo.vertex import combo_tree

    tr = combo_tree(builtin.logical_and)
    root = tr.begin()
    tr.append_child(root, argument(1))
    tr.append_child(root, builtin.null_vertex)
    tr.append_child(root, argument(2))
    remove_null_vertices()(tr, root)
    assert combo_tree_to_str(tr) == "and($1 $2) "


def test_remove_null_vertices_cond_removes_condition_and_consequent_together():
    # C++: general_rules_probe.cc "remove_null_vertices_cond" -- cond
    # takes (condition, consequent) PAIRS plus a final else clause; a
    # null_vertex condition drops BOTH it and its paired consequent.
    from pymoses.comboreduct.combo.argument import argument
    from pymoses.comboreduct.combo.vertex import combo_tree

    tr = combo_tree(builtin.cond)
    root = tr.begin()
    tr.append_child(root, builtin.null_vertex)  # condition 1 (dropped)
    tr.append_child(root, argument(1))  # consequent 1 (dropped too)
    tr.append_child(root, argument(2))  # condition 2 (kept)
    tr.append_child(root, argument(3))  # consequent 2 (kept)
    tr.append_child(root, argument(4))  # else clause (kept)
    remove_null_vertices()(tr, root)
    assert combo_tree_to_str(tr) == "cond($2 $3 $4) "


def _build_cond(*vals):
    from pymoses.comboreduct.combo.vertex import combo_tree

    tr = combo_tree(builtin.cond)
    root = tr.begin()
    for v in vals:
        tr.append_child(root, v)
    return tr, root


def test_remove_null_vertices_cond_null_as_a_plain_consequent_is_never_inspected():
    # C++: remove_null_vertices_cond_crash_probe.cc "good1" -- a null
    # CONSEQUENT is skipped over by the pair-stride walk, never even
    # dereferenced as a candidate for removal, so it's left completely
    # untouched -- matches C++ byte-for-byte (measured).
    from pymoses.comboreduct.combo.argument import argument

    tr, root = _build_cond(argument(1), builtin.null_vertex, argument(2))
    remove_null_vertices()(tr, root)
    assert combo_tree_to_str(tr) == "cond($1 null_vertex $2) "


def test_remove_null_vertices_cond_null_in_a_safe_trailing_position_is_untouched():
    # C++: remove_null_vertices_cond_crash_probe.cc "good2" -- this
    # particular child count makes the pair-stride walk land on the
    # trailing null as a skipped CONSEQUENT position, not a condition --
    # matches C++ byte-for-byte (measured).
    from pymoses.comboreduct.combo.argument import argument

    tr, root = _build_cond(argument(1), argument(2), argument(3), builtin.null_vertex)
    remove_null_vertices()(tr, root)
    assert combo_tree_to_str(tr) == "cond($1 $2 $3 null_vertex) "


def test_remove_null_vertices_cond_null_in_the_trailing_else_slot_raises():
    """A genuine, disclosed DIVERGENCE (general_rules.py's own docstring
    on this rule): real C++ SEGFAULTS on this exact shape (measured with
    a core dump via
    `tools/cpp_probe/remove_null_vertices_cond_crash_probe.cc`'s "bad1"
    case, run bounded -- `( ulimit -v 1048576; timeout 20 <binary> bad1
    )` exits 139, "dumped core"): a null_vertex sitting in `cond`'s
    trailing, UNPAIRED "else" slot is visited by the pair-stride walk as
    if it were a condition; erasing it lands the walk at `end()`, and the
    SECOND `tr.erase(sib)` in that same iteration then operates on that
    null/end iterator. C++ dereferences a null pointer there (undefined
    behaviour); this port raises a clean, catchable `AttributeError`
    instead -- safer, but a genuinely different KIND of failure, not
    something to silently paper over as "close enough".
    """
    from pymoses.comboreduct.combo.argument import argument

    tr, root = _build_cond(argument(1), argument(2), builtin.null_vertex)
    with pytest.raises(AttributeError):
        remove_null_vertices()(tr, root)


def test_remove_null_vertices_cond_two_nulls_one_trailing_also_raises():
    # C++: remove_null_vertices_cond_crash_probe.cc "bad2" -- a genuine
    # null CONDITION (correctly paired, handled fine on its own) followed
    # by a SECOND null occupying the trailing else slot -- also segfaults
    # real C++ (measured, exit 139), for the same reason as "bad1" above.
    from pymoses.comboreduct.combo.argument import argument

    tr, root = _build_cond(builtin.null_vertex, argument(1), builtin.null_vertex)
    with pytest.raises(AttributeError):
        remove_null_vertices()(tr, root)


# ---------------------------------------------------------------------------
# remove_all_assumptions
# ---------------------------------------------------------------------------


def test_remove_all_assumptions_clears_the_assumption_set():
    tr = str_to_combo_tree("$1")
    a = str_to_combo_tree("$2")
    insert_assumption(tr, a.begin())
    assert len(tr.assumptions) == 1

    remove_all_assumptions()(tr, tr.begin())
    assert tr.assumptions == []


def test_remove_all_assumptions_is_a_noop_with_no_assumption_set_yet():
    # Asserting `getattr(tr, "assumptions", []) == []` alone would pass
    # whether or not `apply()` actually ran (a tree with no assumptions
    # attribute yet already satisfies it) -- `delete_all_assumptions`
    # unconditionally sets `tr.assumptions = []` (assumption.py), so
    # checking `hasattr` too distinguishes "the rule ran" from "the rule
    # was never called at all".
    tr = str_to_combo_tree("$1")
    assert not hasattr(tr, "assumptions")
    remove_all_assumptions()(tr, tr.begin())
    assert hasattr(tr, "assumptions")
    assert tr.assumptions == []
