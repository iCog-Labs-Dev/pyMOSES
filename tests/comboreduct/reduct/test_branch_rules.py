"""Tests for pymoses.comboreduct.reduct.branch_rules (S14 Task 4).

`reduce_cond_arg`/`reduce_cond_else`/`reduce_cond_adjacent`/
`reduce_cond_const` operate on `cond`, and (for the two shared with
`contin_if`) that vertex too. Expected outputs below are hand-derived
directly from `branch_rules.cc`'s own algorithm (traced line by line, not
guessed), and the `zap_one` compound-child case is additionally pinned
against a measured run of the real `libcomboreduct.so` (see that test's
own docstring for the exact probe command).
"""

from __future__ import annotations

import copy

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.reduct.branch_rules import (
    _zap_one,
    reduce_cond_adjacent,
    reduce_cond_arg,
    reduce_cond_const,
    reduce_cond_else,
)


def _reduced(tree_str: str, rule) -> str:
    tr = str_to_combo_tree(tree_str)
    rule(tr, tr.begin())
    return combo_tree_to_str(tr).strip()


# ---------------------------------------------------------------------------
# reduce_cond_else
# ---------------------------------------------------------------------------


def test_reduce_cond_else_merges_identical_trailing_consequents():
    # cond(p1 x1 ... pn xn p x x) -> cond(p1 x1 ... pn xn x): here n=0, so
    # cond($1 $2 $2) -> (erase the pred/first-consequent pair) -> cond($2)
    # -> (zap_one) -> $2.
    assert _reduced("cond($1 $2 $2)", reduce_cond_else()) == "$2"


def test_reduce_cond_else_leaves_distinct_trailing_consequents_alone():
    assert _reduced("cond($1 $2 $3)", reduce_cond_else()) == "cond($1 $2 $3)"


def test_reduce_cond_else_also_fires_on_contin_if():
    assert _reduced("contin_if($1 $2 $2)", reduce_cond_else()) == "$2"


def test_reduce_cond_else_ignores_unrelated_vertex():
    assert _reduced("and($1 $2)", reduce_cond_else()) == "and($1 $2)"


def test_reduce_cond_else_repeats_the_merge_across_multiple_trailing_pairs():
    # cond($1 $2 $3 $4 $4) -- last two consequents ($4, $4) are identical
    # -> cond($1 $2 $4); the loop then re-checks with the NEW last three
    # children ($1, $2, $4): $2 != $4, so it stops there.
    assert _reduced("cond($1 $2 $3 $4 $4)", reduce_cond_else()) == "cond($1 $2 $4)"


# ---------------------------------------------------------------------------
# reduce_cond_const
# ---------------------------------------------------------------------------


def test_reduce_cond_const_true_predicate_truncates_to_its_consequent():
    assert _reduced("cond(true $1 $2)", reduce_cond_const()) == "$1"


def test_reduce_cond_const_false_predicate_drops_the_clause():
    assert _reduced("cond(false $1 $2)", reduce_cond_const()) == "$2"


def test_reduce_cond_const_false_clause_dropped_leaving_later_clauses():
    # cond(false $1 $2 $3 $4) -> drop (false, $1) -> cond($2 $3 $4)
    assert _reduced("cond(false $1 $2 $3 $4)", reduce_cond_const()) == "cond($2 $3 $4)"


def test_reduce_cond_const_true_mid_list_truncates_the_rest():
    # cond($1 $2 true $3 $4) -> $1/$2 pair kept as-is (pred $1 is neither
    # true nor false); on the SECOND pass pred=true -> erase it, then
    # erase everything after its consequent ($3) through the end ($4) ->
    # cond($1 $2 $3).
    assert _reduced("cond($1 $2 true $3 $4)", reduce_cond_const()) == "cond($1 $2 $3)"


def test_reduce_cond_const_ignores_unrelated_vertex():
    assert _reduced("and($1 $2)", reduce_cond_const()) == "and($1 $2)"


def test_reduce_cond_const_also_fires_on_contin_if():
    assert _reduced("contin_if(true $1 $2)", reduce_cond_const()) == "$1"


# ---------------------------------------------------------------------------
# reduce_cond_adjacent
# ---------------------------------------------------------------------------


def test_reduce_cond_adjacent_merges_matching_consequents():
    # cond($1 $2 $3 $2 $4) -> cond(or($1 $3) $2 $4)
    assert _reduced("cond($1 $2 $3 $2 $4)", reduce_cond_adjacent()) == "cond(or($1 $3) $2 $4)"


def test_reduce_cond_adjacent_no_merge_when_consequents_differ():
    assert _reduced("cond($1 $2 $3 $4 $5)", reduce_cond_adjacent()) == "cond($1 $2 $3 $4 $5)"


def test_reduce_cond_adjacent_too_few_children_is_a_noop():
    # Fewer than 5 children (2 clauses + else): the loop guard (5 <= sz)
    # never fires at all.
    assert _reduced("cond($1 $2 $3)", reduce_cond_adjacent()) == "cond($1 $2 $3)"


def test_reduce_cond_adjacent_ignores_unrelated_vertex():
    assert _reduced("and($1 $2)", reduce_cond_adjacent()) == "and($1 $2)"


# ---------------------------------------------------------------------------
# reduce_cond_arg
# ---------------------------------------------------------------------------


def test_reduce_cond_arg_reduces_each_predicate():
    # not(not($1)) -> $1 at effort 1 ("simple"); $2 (the trailing else) is
    # never touched (only positions 0, 2, ... are predicates).
    r = reduce_cond_arg(1, set())
    assert _reduced("cond(not(not($1)) $2 not(not($3)) $4 $5)", r) == "cond($1 $2 $3 $4 $5)"


def test_reduce_cond_arg_ignores_unrelated_vertex():
    r = reduce_cond_arg(1, set())
    assert _reduced("and($1 $2)", r) == "and($1 $2)"


def test_reduce_cond_arg_leaves_the_else_clause_alone():
    # The trailing else clause is at an even position and must never be
    # treated as a predicate -- a not(not(...)) there stays untouched.
    r = reduce_cond_arg(1, set())
    assert _reduced("cond($1 $2 not(not($3)))", r) == "cond($1 $2 not(not($3)))"


# ---------------------------------------------------------------------------
# _zap_one -- the measured upstream bug (module docstring)
# ---------------------------------------------------------------------------


def test_zap_one_on_a_leaf_child_is_a_clean_collapse():
    tr = str_to_combo_tree("cond($1)")
    _zap_one(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "$1"


def test_zap_one_on_a_compound_child_discards_its_children():
    """MEASURED (2026-09-09) against the real `libcomboreduct.so`:

        g++ ... -o probe probe.cc -lcomboreduct -lcogutil
        combo_tree tr; stringstream("cond(+($1 $2))") >> tr;
        contin_reduce(tr, 2, vertex_set());
        // before: cond(+($1 $2))
        // after:  +            (size 1 -- $1 and $2 both silently gone)

    `_zap_one` copies `+`'s own vertex onto `it`, then erases the `+`
    node outright -- `tree.erase` always erases children first, so `$1`
    and `$2` are destroyed along with it, rather than promoted onto `it`
    the way `tr.flatten` would. This is the exact shape `reduce_cond_else`/
    `reduce_cond_const` reach whenever a `cond`/`contin_if` collapses to a
    single COMPOUND remaining clause -- reproduced here directly against
    `_zap_one` in isolation, one call site removed from needing the full
    `contin_reduction` pipeline this module does not yet have access to.

    PROVEN BY MUTATION: replacing the body with the "obviously correct"
    fix (`tr.erase(tr.flatten(child))`, the same idiom
    `reduce_plus_times_one_child`/`reduce_log_div_times` use for a
    same-shaped single-child collapse elsewhere in this port) turns this
    test red (`+($1 $2)`, not the bare `+` asserted below) -- confirming
    this test would catch an accidental "fix" of the upstream defect.
    """
    tr = str_to_combo_tree("cond(+($1 $2))")
    _zap_one(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "+"
    assert tr.is_childless(tr.begin())


def test_zap_one_mutation_proof_flatten_fix_would_be_caught():
    """The mutation referenced by the previous test's docstring, executed
    directly here rather than only described: a `_zap_one` that flattens
    (the "obviously correct" reading) produces a DIFFERENT result than the
    real, buggy upstream -- proving the guard above actually discriminates
    faithful-vs-fixed rather than passing either way.
    """
    def _fixed_zap_one(tr, it):
        child = it.first_child
        it.data = copy.copy(child.data)
        tr.erase(tr.flatten(child))

    tr = str_to_combo_tree("cond(+($1 $2))")
    _fixed_zap_one(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "+($1 $2)"
    assert not tr.is_childless(tr.begin())
    # sanity: this "fixed" behaviour differs from the real, faithful port
    assert combo_tree_to_str(tr).strip() != "+"
