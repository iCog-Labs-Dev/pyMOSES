"""C++: moses/comboreduct/combo/assumption.{h,cc}

No classic cxxtest exists for assumption.cc (grep across $CLASSIC/tests
turns up nothing at all) -- these are hand-written tests, several of them
pinned against `tools/cpp_probe/assumption_probe.cc` measurements of the
real compiled `libcomboreduct.so`, not against assumption.h's own
(partly inaccurate -- see `insert_assumption`'s docstring) comment block.

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
subtree. See assumption.py's module docstring for the full reasoning,
including why this reshaping is safe against every actual in-scope
caller, and for what is and is not settled about copying a whole tree's
assumptions along with it.
"""

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.assumption import (
    copy_all_assumptions,
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
#
# These three tests are pinned DIRECTLY against a measured run of the real
# C++ library, not against assumption.h's own comment (which claims a
# property the code does not actually have -- see insert_assumption's
# docstring). Reproduced with `tools/cpp_probe/assumption_probe.cc`:
#
#   $ ( ulimit -v 1048576; timeout 20 ./tools/cpp_probe/build/assumption_probe )
#   case1 insert true,true: [true , true ]
#   case2 insert $1,$2,$3: [$2 , $1 , $3 ]
#   case3 insert $3,$2,$1: [$1 , $2 , $3 ]
#   case4 insert $3,$4,$3(dup): [$3 , $4 , $3 ]
#   case5 insert true,true,true: [true , true ]
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


def test_insert_assumption_with_exactly_one_existing_assumption_does_not_dedup():
    # MEASURED (probe case1): a SECOND insert of an exact duplicate is NOT
    # dropped -- assumption.cc's binary-search dedup only runs once
    # max_pos > 1 (i.e. from the THIRD insert onward). insert_assumption's
    # class-level doc comment (assumption.h:34-37) claims the set is
    # always deduplicated; it is not, for exactly this case.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    insert_assumption(tr, str_to_combo_tree("true").begin())
    assert _assumption_data(tr) == [T, T]


def test_insert_assumption_dedup_at_max_pos_greater_than_one_is_position_dependent_not_guaranteed():
    # MEASURED (probe case4): with a third insert against a 2-element set
    # (max_pos > 1, so the binary search DOES run), a duplicate is NOT
    # necessarily caught. The search's single probe position ($4, not the
    # actual duplicate $3) never lands on the duplicate at all here, and
    # the "search range collapsed" branch that follows inserts
    # unconditionally with no further comparison -- so this duplicate
    # slips through, giving THREE assumptions, not two.
    #
    # A first version of this test asserted the opposite (that max_pos > 1
    # guarantees dedup) without re-checking against the real library --
    # exactly the mistake this whole module's docstring warns about.
    # Corrected after running assumption_probe against libcomboreduct.so:
    #   $ ( ulimit -v 1048576; timeout 20 ./tools/cpp_probe/build/assumption_probe )
    #   case4 insert $3,$4,$3(dup): [$3 , $4 , $3 ]
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("$3").begin())
    insert_assumption(tr, str_to_combo_tree("$4").begin())
    assert len(tr.assumptions) == 2
    insert_assumption(tr, str_to_combo_tree("$3").begin())  # exact duplicate
    assert _assumption_data(tr) == [argument(3), argument(4), argument(3)]


def test_insert_assumption_dedup_does_fire_when_the_search_happens_to_probe_the_duplicate():
    # MEASURED (probe case5): the SAME max_pos > 1 binary search DOES
    # dedup when its very first probed position happens to already BE a
    # structurally-equal element -- here, both existing assumptions are
    # "true", so the third insert's first (and only) comparison hits an
    # exact match immediately and returns without inserting. Contrasted
    # directly with the previous test: whether a duplicate is caught
    # depends on which position the search happens to probe, not on
    # max_pos alone.
    #   $ ( ulimit -v 1048576; timeout 20 ./tools/cpp_probe/build/assumption_probe )
    #   case5 insert true,true,true: [true , true ]
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("true").begin())
    insert_assumption(tr, str_to_combo_tree("true").begin())
    assert len(tr.assumptions) == 2
    insert_assumption(tr, str_to_combo_tree("true").begin())  # exact duplicate
    assert _assumption_data(tr) == [T, T]


def test_insert_assumption_with_zero_or_one_existing_prepends_unconditionally():
    # MEASURED (probe case2/case3): inserting $1, $2, $3 in that order
    # gives [$2, $1, $3] -- NOT sorted -- because the first two inserts
    # each hit the unconditional "prepend, no comparison" bypass
    # (max_pos <= 1, assumption.cc:74) and only the third goes through
    # the (correct) binary search, which searches among an already
    # out-of-order pair.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("$1").begin())
    insert_assumption(tr, str_to_combo_tree("$2").begin())
    insert_assumption(tr, str_to_combo_tree("$3").begin())
    assert _assumption_data(tr) == [argument(2), argument(1), argument(3)]


def test_insert_assumption_result_depends_on_insertion_order_not_just_the_final_set():
    # MEASURED (probe case3): the SAME final set {$1, $2, $3}, inserted in
    # the REVERSE order, lands differently -- [$1, $2, $3] this time --
    # because which pair is already present (and therefore which bypasses
    # ordering) differs. This is the same underlying bug as the previous
    # test, demonstrated the way the review asked: two insertion orders of
    # one set, pinned to their real (and different) results, not asserted
    # to agree.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("$3").begin())
    insert_assumption(tr, str_to_combo_tree("$2").begin())
    insert_assumption(tr, str_to_combo_tree("$1").begin())
    assert _assumption_data(tr) == [argument(1), argument(2), argument(3)]


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
# copy_all_assumptions -- no C++ symbol of its own; mirrors the raw
# insert_subtree_after chain inside logical_rules.cc:236-238
# (reduce_and_assumptions), added for Task 4.
# ---------------------------------------------------------------------------


def test_copy_all_assumptions_is_order_preserving_not_insert_assumption_order():
    # The point of this helper: it does NOT go through insert_assumption's
    # (buggy) ordering rules at all -- it is a raw, order-preserving copy,
    # so a source list built via the "bug path" above copies verbatim,
    # unsorted-looking order and all.
    src = str_to_combo_tree("and($1 $2)")
    insert_assumption(src, str_to_combo_tree("$1").begin())
    insert_assumption(src, str_to_combo_tree("$2").begin())
    insert_assumption(src, str_to_combo_tree("$3").begin())
    assert _assumption_data(src) == [argument(2), argument(1), argument(3)]

    dest = str_to_combo_tree("or($5 $6)")
    copy_all_assumptions(dest, src)
    assert _assumption_data(dest) == [argument(2), argument(1), argument(3)]


def test_copy_all_assumptions_appends_rather_than_replaces():
    src = str_to_combo_tree("and($1 $2)")
    insert_assumption(src, str_to_combo_tree("true").begin())

    dest = str_to_combo_tree("or($5 $6)")
    insert_assumption(dest, str_to_combo_tree("false").begin())
    copy_all_assumptions(dest, src)
    assert _assumption_data(dest) == [F, T]


def test_copy_all_assumptions_copies_rather_than_aliases():
    src = str_to_combo_tree("and($1 $2)")
    insert_assumption(src, str_to_combo_tree("true").begin())
    dest = str_to_combo_tree("or($5 $6)")
    copy_all_assumptions(dest, src)

    delete_all_assumptions(src)
    assert _assumption_data(dest) == [T]


def test_copy_all_assumptions_from_a_source_with_none_is_a_no_op():
    src = str_to_combo_tree("and($1 $2)")
    dest = str_to_combo_tree("or($5 $6)")
    copy_all_assumptions(dest, src)
    assert _assumption_data(dest) == []


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


def test_find_vertices_in_assumptions_uses_vertex_equal_not_bare_equality():
    # Removing the vertex_equal guard (a bare `node.data == v` instead)
    # would report a false match here: builtin.list's ordinal is 15, and
    # a bare `15.0 == builtin.list` is True in Python even though they
    # are different vertex alternatives. Searching for the contin_t 15.0
    # must NOT match an assumption holding the builtin `list` token.
    tr = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr, str_to_combo_tree("list(true)").begin())
    res = []
    assert find_vertices_in_assumptions(tr, 15.0, res) is False
    assert res == []


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


def test_equal_assumptions_uses_vertex_equal_not_bare_equality():
    # Regression pin for the review's Important #4: `tree.equal_subtree`
    # (used by an EARLIER version of this function) compares node data
    # with a bare `!=` and reports [1.0] equal to [logical_true] (ordinal
    # collision, 1.0 == builtin.logical_true in bare Python). This must
    # NOT be considered equal.
    tr1 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr1, str_to_combo_tree("1").begin())  # contin_t 1.0
    tr2 = str_to_combo_tree("and($1 $2)")
    insert_assumption(tr2, str_to_combo_tree("true").begin())  # builtin.logical_true
    assert equal_assumptions(tr1, tr2) is False


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
