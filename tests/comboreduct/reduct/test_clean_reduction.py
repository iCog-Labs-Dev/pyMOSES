"""Tests for pymetta/comboreduct/reduct/clean_reduction.py -- `clean_reduce`,
which removes `null_vertex` sentinels.

C++: moses/comboreduct/reduct/clean_reduction.cc.

`clean_reduction()` composes exactly two already-fully-ported, already-
fully-tested rules (`general_rules.remove_null_vertices`, `logical_rules.
remove_dangling_junctors`) via two already-tested combinators
(`downwards`/`upwards`/`sequential`) -- this file's own test burden is
the COMPOSITION, not the individual rules. `test_clean_reduce_cascades_
to_a_now_dangling_ancestor` is the load-bearing case: it distinguishes
"just `downwards(remove_null_vertices())` alone" (which would leave a
freshly-childless ancestor `and`/`or`/`not` behind) from the REAL, full
two-pass composition -- MEASURED against real C++
(`tools/cpp_probe/logical_rules_probe.cc`'s `clean_reduce_cascades_to_
dangling_ancestor` case).
"""

from __future__ import annotations

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.reduct.clean_reduction import clean_reduce, clean_reduction
from pymetta.comboreduct.reduct.general_rules import remove_null_vertices


def test_clean_reduce_removes_a_null_vertex_leaf():
    tr = str_to_combo_tree("and($1 $2)")
    tr.insert_above(tr.begin().first_child, builtin.null_vertex)
    assert combo_tree_to_str(tr).strip() == "and(null_vertex($1) $2)"
    clean_reduce(tr)
    assert combo_tree_to_str(tr).strip() == "and($2)"


def test_clean_reduce_cascades_to_a_now_dangling_ancestor():
    """MEASURED against real C++ (`clean_reduce_cascades_to_dangling_
    ancestor`): `and(and(null_vertex($1)))` -> `and`. The inner `and`'s
    ONLY child is the `null_vertex`-wrapped `$1`; removing it (`downwards
    (remove_null_vertices())`) leaves the inner `and` childless, which
    only the SEPARATE, LATER `upwards(remove_dangling_junctors())` pass
    then removes from the OUTER `and` -- a single `downwards(remove_
    null_vertices())` pass alone would stop at `and(and())`, never
    reaching the outer `and`'s own now-dangling child. Directly
    distinguishes the real two-rule composition from just its first half
    (see the comparison test right below).
    """
    tr = str_to_combo_tree("and(and(null_vertex($1)))")
    clean_reduce(tr)
    assert combo_tree_to_str(tr).strip() == "and"


def test_remove_null_vertices_alone_does_not_cascade():
    """The DIRECT counter-case to the test above: applying ONLY the first
    half of `clean_reduction`'s composition (`downwards(remove_null_
    vertices())`, with no subsequent `upwards(remove_dangling_
    junctors())` pass) on the SAME input leaves the inner `and` dangling
    -- `and(and)` (a childless inner `and`), not the fully-collapsed
    `and` clean_reduce's real composition gives. Confirms `clean_reduce`'s
    own two-step composition is what does the cascading, not `remove_
    null_vertices` alone.
    """
    from pymetta.comboreduct.reduct.meta_rules import downwards

    tr = str_to_combo_tree("and(and(null_vertex($1)))")
    downwards(remove_null_vertices())(tr)
    assert combo_tree_to_str(tr).strip() == "and(and)"


def test_clean_reduction_returns_a_rule_object_reusable_across_trees():
    r = clean_reduction()
    tr1 = str_to_combo_tree("and($1 $2)")
    tr1.insert_above(tr1.begin().first_child, builtin.null_vertex)
    r(tr1)
    assert combo_tree_to_str(tr1).strip() == "and($2)"

    tr2 = str_to_combo_tree("or($1 $2)")
    tr2.insert_above(tr2.begin().first_child, builtin.null_vertex)
    r(tr2)
    assert combo_tree_to_str(tr2).strip() == "or($2)"


def test_clean_reduce_explicit_it_only_affects_that_subtree():
    """`clean_reduce(tr, it)` (reduct.h:199) applies only within `it`'s
    own subtree -- a `null_vertex` OUTSIDE `it` must survive untouched.
    """
    tr = str_to_combo_tree("and(and(null_vertex($1)) $2)")
    root = tr.begin()
    outer_and_second_child = root.first_child.next_sibling  # $2 sibling
    # apply clean_reduce scoped to $2 alone (a leaf -- genuinely a no-op,
    # and must NOT touch the null_vertex sitting elsewhere in the tree).
    clean_reduce(tr, outer_and_second_child)
    assert combo_tree_to_str(tr).strip() == "and(and(null_vertex($1)) $2)"

    # Now apply it at the real root: the null_vertex is now in scope and
    # gets removed, cascading exactly as the whole-tree test above shows.
    clean_reduce(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "and($2)"


def test_clean_reduce_explicit_none_it_raises():
    tr = str_to_combo_tree("$1")
    with pytest.raises(ValueError):
        clean_reduce(tr, None)


def test_clean_reduce_on_an_empty_tree_is_a_noop():
    from pymetta.comboreduct.combo.vertex import combo_tree

    tr = combo_tree()
    clean_reduce(tr)  # must not raise
    assert tr.empty()
