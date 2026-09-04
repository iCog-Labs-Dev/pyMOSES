"""C++: moses/comboreduct/combo/assumption.{h,cc}

No classic cxxtest exists for assumption.cc (grep across $CLASSIC/tests
turns up nothing at all) -- these are hand-written tests against
assumption.h's own documented contract (insert_assumption's own comment
block, assumption.h:31-45) rather than a transliterated C++ oracle.

DIVERGENCE(C++ moses/comboreduct/combo/assumption.cc, whole file): classic
stores a combo_tree's assumption set as top-level SIBLINGS of tr.begin()
within the very same combo_tree (a forest: multiple top-level roots
sharing one tree, held together by tree<T>'s head/feet sentinels).
pymetta's util/tree.py deliberately does not support forests -- a decision
already made and regression-tested in the prior (S0-S6) cycle
(tree.py's own module docstring DIVERGENCE marker;
test_insert_after_on_the_tree_root_is_rejected in test_tree.py) -- and
iostream_combo.py's str_to_combo_tree rejects multiple top-level
expressions outright. This task cannot and should not reopen that
decision. Instead, the assumption set is attached to a combo_tree as a
plain Python list, `tr.assumptions` (created lazily on first use), each
element a standalone combo_tree holding a deep copy of one assumption
subtree -- preserving insert_assumption's own set semantics (sorted,
deduplicated) and the other three functions' set-level contract, just
relocated out of the tree's own node graph. See assumption.py's module
docstring for the full reasoning, including why this is safe against
every actual in-scope caller (general_rules.remove_all_assumptions,
meta_rules.assum_iterative, logical_rules.reduce_and_assumptions).
"""

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.assumption import (
    delete_all_assumptions,
    equal_assumptions,
    find_vertices_in_assumptions,
    insert_assumption,
)
from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree

T = builtin.logical_true
F = builtin.logical_false


def _assumption_data(tr) -> list:
    """The vertex held at the root of each attached assumption, in
    whatever order insert_assumption settled on -- the thing every test
    below actually wants to assert against, without hard-coding storage
    shape (`tr.assumptions`) into every single test."""
    return [a.begin().data for a in getattr(tr, "assumptions", [])]


# ---------------------------------------------------------------------------
# insert_assumption -- assumption.cc:35-75
# ---------------------------------------------------------------------------


def test_insert_assumption_on_a_tree_with_none_yet():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    assert _assumption_data(tr) == [T]


def test_insert_assumption_does_not_touch_the_main_tree_body():
    # The DIVERGENCE this module documents: the main tree's own node graph
    # is untouched -- assumptions live in the side list, not as tree
    # siblings (impossible for this port's single-rooted tree anyway).
    tr = str_to_combo_tree("and($1 $2)")
    before = [n.data for n in tr]
    insert_assumption(tr, str_to_combo_tree("true").begin())
    assert [n.data for n in tr] == before


def test_insert_assumption_is_a_set_duplicates_collapse():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    insert_assumption(tr, str_to_combo_tree("true").begin())
    assert _assumption_data(tr) == [T]


def test_insert_assumption_structural_duplicate_also_collapses():
    # Structural equality, not object identity: two independently-built
    # subtrees with the same shape/content are still "the same assumption".
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("and($3 $4)").begin())
    insert_assumption(tr, str_to_combo_tree("and($3 $4)").begin())
    assert len(tr.assumptions) == 1


def test_insert_assumption_maintains_sorted_order_regardless_of_insertion_order():
    tr_ascending = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr_ascending, str_to_combo_tree("1").begin())
    insert_assumption(tr_ascending, str_to_combo_tree("2").begin())

    tr_descending = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr_descending, str_to_combo_tree("2").begin())
    insert_assumption(tr_descending, str_to_combo_tree("1").begin())

    assert _assumption_data(tr_ascending) == _assumption_data(tr_descending)


def test_insert_assumption_copies_rather_than_aliases_the_source():
    tr = str_to_combo_tree("and($1 $2)")
    src = str_to_combo_tree("and($3 $4)")
    insert_assumption(tr, src.begin())
    # Mutating the source after insertion must not be visible.
    src.append_child(src.begin(), builtin.logical_true)
    assert [n.data for n in tr.assumptions[0]] == [builtin.logical_and, argument(3), argument(4)]


def test_insert_assumption_on_an_empty_tree_raises():
    # C++: assumption.cc:38 -- OC_ASSERT(tr.is_valid(main_tree), ...).
    tr = combo_tree()
    with pytest.raises(AssertionError):
        insert_assumption(tr, str_to_combo_tree("true").begin())


# ---------------------------------------------------------------------------
# find_vertices_in_assumptions -- assumption.cc:77-92
# ---------------------------------------------------------------------------


def test_find_vertices_in_assumptions_finds_a_root_level_match():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    res = []
    assert find_vertices_in_assumptions(tr, T, res) is True
    assert [n.data for n in res] == [T]


def test_find_vertices_in_assumptions_searches_the_whole_subtree_not_just_roots():
    # assumption.cc's walk is a full pre-order combo_tree::iterator, not a
    # sibling_iterator over assumption roots only -- it must find a vertex
    # nested inside an assumption subtree too.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("or($3 $4)").begin())
    res = []
    assert find_vertices_in_assumptions(tr, argument(4), res) is True
    assert [n.data for n in res] == [argument(4)]


def test_find_vertices_in_assumptions_collects_every_match():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("or($3 $3)").begin())
    res = []
    assert find_vertices_in_assumptions(tr, argument(3), res) is True
    assert len(res) == 2


def test_find_vertices_in_assumptions_returns_false_when_absent():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    res = []
    assert find_vertices_in_assumptions(tr, F, res) is False
    assert res == []


def test_find_vertices_in_assumptions_with_no_assumptions_at_all_returns_false():
    tr = str_to_combo_tree("and($1 $2)")
    res = []
    assert find_vertices_in_assumptions(tr, T, res) is False
    assert res == []


def test_find_vertices_in_assumptions_does_not_search_the_main_tree_body():
    # $1/$2 live in the main conclusion, not the assumption set.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    res = []
    assert find_vertices_in_assumptions(tr, argument(1), res) is False


# ---------------------------------------------------------------------------
# equal_assumptions -- assumption.cc:94-107
# ---------------------------------------------------------------------------


def test_equal_assumptions_true_for_identical_sets():
    tr1 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr1, str_to_combo_tree("true").begin())
    tr2 = str_to_combo_tree("or($5 $6)")  # a different main tree body entirely
    insert_assumption(tr2, str_to_combo_tree("true").begin())
    assert equal_assumptions(tr1, tr2) is True


def test_equal_assumptions_false_for_different_sets():
    tr1 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr1, str_to_combo_tree("true").begin())
    tr2 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr2, str_to_combo_tree("false").begin())
    assert equal_assumptions(tr1, tr2) is False


def test_equal_assumptions_false_for_different_sized_sets():
    tr1 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr1, str_to_combo_tree("true").begin())
    tr2 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr2, str_to_combo_tree("true").begin())
    insert_assumption(tr2, str_to_combo_tree("false").begin())
    assert equal_assumptions(tr1, tr2) is False


def test_equal_assumptions_true_when_neither_tree_has_any():
    # C++: assumption.cc:98-106 -- when neither tr.next_sibling(tr.begin())
    # is valid (no assumptions at all on either side), the "else return
    # true" branch fires.
    tr1 = str_to_combo_tree("and($1 $2)")
    tr2 = str_to_combo_tree("or($3 $4)")
    assert equal_assumptions(tr1, tr2) is True


def test_equal_assumptions_true_when_only_one_side_has_any():
    # SURPRISING INHERITED BEHAVIOUR (see equal_assumptions's own
    # docstring): C++'s assumption.cc:98-106 returns true whenever EITHER
    # side has zero assumptions, even though the other side has some --
    # not just when both are empty. Ported faithfully, not corrected.
    tr1 = str_to_combo_tree("and($1 $2)")
    tr2 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr2, str_to_combo_tree("true").begin())
    insert_assumption(tr2, str_to_combo_tree("false").begin())
    assert equal_assumptions(tr1, tr2) is True
    assert equal_assumptions(tr2, tr1) is True


def test_equal_assumptions_on_an_empty_tree_raises():
    with pytest.raises(AssertionError):
        equal_assumptions(combo_tree(), str_to_combo_tree("true"))


# ---------------------------------------------------------------------------
# delete_all_assumptions -- assumption.cc:109-115
# ---------------------------------------------------------------------------


def test_delete_all_assumptions_clears_them():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    insert_assumption(tr, str_to_combo_tree("false").begin())
    delete_all_assumptions(tr)
    assert _assumption_data(tr) == []


def test_delete_all_assumptions_leaves_the_main_tree_body_alone():
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    before = [n.data for n in tr]
    delete_all_assumptions(tr)
    assert [n.data for n in tr] == before


def test_delete_all_assumptions_on_a_tree_with_none_yet_is_a_no_op():
    tr = str_to_combo_tree("and($1 $2)")
    delete_all_assumptions(tr)  # must not raise
    assert _assumption_data(tr) == []


def test_delete_all_assumptions_on_an_empty_tree_raises():
    # C++: assumption.cc:110 -- OC_ASSERT(!tr.empty(), ...).
    with pytest.raises(AssertionError):
        delete_all_assumptions(combo_tree())
