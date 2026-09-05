import json

import pytest

from pymetta.util.tree import tree


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "tree.json").read_text())


def to_sexpr(t, node=None):
    node = node if node is not None else t.begin()
    if node is None:
        return None
    kids = [to_sexpr(t, c) for c in t.children(node)]
    return {"data": node.data, "children": kids}


def validate(t, node=None):
    """C++: opencog/util/tree.h:2174 validate (test helper only -- not part
    of the public API surface tested by to_sexpr, which walks only
    first_child/next_sibling and so can never notice a broken back-link).

    Note this is a deliberately *stronger* check than C++'s own
    tree.h:2174 validate(), which shares the blind spot fixed below (it
    never independently checks first_child/last_child against the sibling
    chain reached by forward traversal -- see the round-2 review). That is
    fine: validate() is a test instrument, not ported production code.
    Fidelity binds what this module ports; it does not bind the tools used
    to check the port, and inheriting a C++ blind spot here would only
    cost detection for nothing in return.

    For every node in `node`'s subtree (the whole tree if `node` is
    omitted): `first_child`/`last_child` genuinely have no prev/next
    sibling of their own (independent of forward traversal -- this is what
    catches a corrupted `first_child`/`last_child` pointer that quietly
    orphans the true first/last child); each child's `parent` points back
    to it; `first_child`/`last_child` are the true ends of the sibling
    chain as seen from a child looking up; `next_sibling`/`prev_sibling`
    are mutual inverses. Do not call this on a tree known to contain the
    move_ontop adjacent-sibling self-cycle (see
    test_move_ontop_adjacent_sibling_self_cycle_is_an_inherited_bug) --
    walking children() there does not terminate, by design.
    """
    node = node if node is not None else t.begin()
    if node is None:
        return
    if node.first_child is not None:
        assert node.first_child.prev_sibling is None, (
            f"{node.data!r}.first_child ({node.first_child.data!r}) has a "
            f"non-None prev_sibling -- the true first child was skipped"
        )
    if node.last_child is not None:
        assert node.last_child.next_sibling is None, (
            f"{node.data!r}.last_child ({node.last_child.data!r}) has a "
            f"non-None next_sibling -- the true last child was skipped"
        )
    for child in t.children(node):
        assert child.parent is node, (
            f"{child.data!r}.parent does not point back to {node.data!r}"
        )
    if node.parent is not None:
        if node.prev_sibling is None:
            assert node.parent.first_child is node, (
                f"{node.data!r} has no prev_sibling but is not its parent's first_child"
            )
        if node.next_sibling is None:
            assert node.parent.last_child is node, (
                f"{node.data!r} has no next_sibling but is not its parent's last_child"
            )
    if node.prev_sibling is not None:
        assert node.prev_sibling.next_sibling is node, (
            f"{node.data!r}.prev_sibling.next_sibling does not point back to it"
        )
    if node.next_sibling is not None:
        assert node.next_sibling.prev_sibling is node, (
            f"{node.data!r}.next_sibling.prev_sibling does not point back to it"
        )
    for child in t.children(node):
        validate(t, child)


# ---- oracle: replay each probe case, compare structural / value result ----


@pytest.mark.oracle
def test_append_child_data(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    validate(t)
    assert to_sexpr(t) == golden["append_child_data"]["result"]


@pytest.mark.oracle
def test_append_child_subtree(golden):
    src = tree("s")
    src.append_child(src.begin(), "s1")
    src.append_child(src.begin(), "s2")
    dst = tree("d")
    dst.append_child(dst.begin(), src.begin())
    validate(src)
    validate(dst)
    assert to_sexpr(dst) == golden["append_child_subtree"]["dest"]
    assert to_sexpr(src) == golden["append_child_subtree"]["source_untouched"]


@pytest.mark.oracle
def test_prepend_child(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.prepend_child(t.begin(), "z")
    validate(t)
    assert to_sexpr(t) == golden["prepend_child"]["result"]


@pytest.mark.oracle
def test_append_children(golden):
    t = tree("root")
    t.append_children(t.begin(), ["x", "y", "z"])
    validate(t)
    assert to_sexpr(t) == golden["append_children"]["result"]


@pytest.mark.oracle
def test_insert(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.insert(b, "n")
    validate(t)
    assert to_sexpr(t) == golden["insert"]["result"]


@pytest.mark.oracle
def test_insert_after(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.insert_after(b, "n")
    validate(t)
    assert to_sexpr(t) == golden["insert_after"]["result"]


@pytest.mark.oracle
def test_insert_subtree(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    src = tree("s")
    src.append_child(src.begin(), "s1")
    t.insert_subtree(b, src.begin())
    validate(t)
    assert to_sexpr(t) == golden["insert_subtree"]["result"]


@pytest.mark.oracle
def test_insert_subtree_after(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    src = tree("s")
    src.append_child(src.begin(), "s1")
    t.insert_subtree_after(b, src.begin())
    validate(t)
    assert to_sexpr(t) == golden["insert_subtree_after"]["result"]


@pytest.mark.oracle
def test_insert_above_internal(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(b, "b2")
    t.append_child(t.begin(), "c")
    t.insert_above(b, "n")
    validate(t)
    assert to_sexpr(t) == golden["insert_above_internal"]["result"]


@pytest.mark.oracle
def test_insert_above_root(golden):
    t = tree("R")
    t.append_child(t.begin(), "r1")
    t.append_child(t.begin(), "r2")
    t.insert_above(t.begin(), "N")
    validate(t)
    assert to_sexpr(t) == golden["insert_above_root"]["result"]


@pytest.mark.oracle
def test_erase(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(b, "b2")
    t.append_child(t.begin(), "c")
    ret = t.erase(b)
    validate(t)
    assert to_sexpr(t) == golden["erase"]["result"]
    assert ret.data == golden["erase"]["returned"]


@pytest.mark.oracle
def test_erase_root(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    ret = t.erase(t.begin())
    validate(t)
    assert t.empty() == golden["erase_root"]["empty"]
    assert (not t.is_valid(ret)) == golden["erase_root"]["returned_is_end"]


@pytest.mark.oracle
def test_erase_children(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(b, "b2")
    t.append_child(t.begin(), "c")
    t.erase_children(b)
    validate(t)
    assert to_sexpr(t) == golden["erase_children"]["result"]


@pytest.mark.oracle
def test_erase_children_leaf(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.erase_children(a)
    validate(t)
    assert to_sexpr(t) == golden["erase_children_leaf"]["result"]


@pytest.mark.oracle
def test_replace_data(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.replace(b, "B2")
    validate(t)
    assert to_sexpr(t) == golden["replace_data"]["result"]


@pytest.mark.oracle
def test_replace_subtree(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(t.begin(), "c")
    src = tree("s")
    src.append_child(src.begin(), "s1")
    src.append_child(src.begin(), "s2")
    t.replace(b, src.begin())
    validate(t)
    validate(src)
    assert to_sexpr(t) == golden["replace_subtree"]["result"]
    assert to_sexpr(src) == golden["replace_subtree"]["source_untouched"]


@pytest.mark.oracle
def test_flatten(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(b, "b2")
    t.append_child(t.begin(), "c")
    t.flatten(b)
    validate(t)
    assert to_sexpr(t) == golden["flatten"]["result"]


@pytest.mark.oracle
def test_flatten_childless(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    t.flatten(a)
    validate(t)
    assert to_sexpr(t) == golden["flatten_childless"]["result"]


@pytest.mark.oracle
def test_wrap(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(b, "b2")
    t.append_child(t.begin(), "c")
    t.wrap(b, "W")
    validate(t)
    assert to_sexpr(t) == golden["wrap"]["result"]


@pytest.mark.oracle
def test_reparent(golden):
    t = tree("root")
    src = t.append_child(t.begin(), "src")
    t.append_child(src, "s1")
    t.append_child(src, "s2")
    dst = t.append_child(t.begin(), "dst")
    t.reparent(dst, src)
    validate(t)
    assert to_sexpr(t) == golden["reparent"]["result"]


@pytest.mark.oracle
def test_move_after(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    d = t.append_child(t.begin(), "d")
    t.move_after(b, d)
    validate(t)
    assert to_sexpr(t) == golden["move_after"]["result"]


@pytest.mark.oracle
def test_move_before(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    d = t.append_child(t.begin(), "d")
    t.move_before(b, d)
    validate(t)
    assert to_sexpr(t) == golden["move_before"]["result"]


@pytest.mark.oracle
def test_move_ontop(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    c = t.append_child(t.begin(), "c")
    t.append_child(c, "c1")
    t.append_child(c, "c2")
    t.append_child(t.begin(), "d")
    e = t.append_child(t.begin(), "e")
    t.append_child(e, "e1")
    t.move_ontop(c, e)
    validate(t)
    assert to_sexpr(t) == golden["move_ontop"]["result"]


@pytest.mark.oracle
def test_move_ontop_adjacent_sibling_self_cycle_is_an_inherited_bug(golden):
    # Deliberately reproduces the upstream defect documented on
    # tree.move_ontop -- see its DIVERGENCE comment. Pointer inspection
    # only: DO NOT iterate/traverse/validate `t` after this -- that hangs
    # by design, exactly as the equivalent C++ probe case does (and
    # exactly why the probe itself never destructs the corresponding tree
    # either).
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    c = t.append_child(t.begin(), "c")
    t.append_child(c, "c1")
    t.append_child(c, "c2")
    e = t.append_child(t.begin(), "e")  # e is c's immediate next sibling
    t.append_child(e, "e1")

    prev_of_c = c.prev_sibling
    assert c.next_sibling is e  # matches the probe's target_next_was_source

    t.move_ontop(c, e)

    expected = golden["move_ontop_adjacent_sibling_bug"]
    assert (e.next_sibling is e) == expected["source_next_sibling_is_self"]
    assert e.prev_sibling.data == expected["source_prev_sibling_data"]
    assert t.begin().last_child is prev_of_c  # stale, matches C++
    assert t.begin().last_child.data == expected["parent_last_child_data"]


@pytest.mark.oracle
def test_sort(golden):
    t = tree("root")
    bb = t.append_child(t.begin(), "bb")
    t.append_child(bb, "bb1")
    a = t.append_child(t.begin(), "a")
    t.append_child(a, "a1")
    ccc = t.append_child(t.begin(), "ccc")
    t.append_child(ccc, "ccc1")
    t.sort(t.begin(), key=len)
    validate(t)
    assert to_sexpr(t) == golden["sort"]["result"]


@pytest.mark.oracle
def test_equal_subtree(golden):
    t1 = tree("root")
    a1 = t1.append_child(t1.begin(), "a")
    t1.append_child(a1, "x")
    t1.append_child(a1, "y")

    t2 = tree("root")
    a2 = t2.append_child(t2.begin(), "a")
    t2.append_child(a2, "x")
    t2.append_child(a2, "y")

    t3 = tree("root")
    a3 = t3.append_child(t3.begin(), "a")
    t3.append_child(a3, "x")
    t3.append_child(a3, "DIFFERENT")

    t4 = tree("root")
    a4 = t4.append_child(t4.begin(), "a")
    t4.append_child(a4, "x")

    expected = golden["equal_subtree"]
    assert t1.equal_subtree(t1.begin(), t2.begin()) == expected["identical_shape"]
    assert t1.equal_subtree(t1.begin(), t3.begin()) == expected["different_data"]
    assert t1.equal_subtree(t1.begin(), t4.begin()) == expected["different_children_count"]
    assert t1.equal_subtree(t1.begin(), t1.begin()) == expected["same_object"]


@pytest.mark.oracle
def test_max_depth(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    a1 = t.append_child(a, "a1")
    t.append_child(t.begin(), "leaf")
    expected = golden["max_depth"]
    assert t.max_depth(t.begin()) == expected["root"]
    assert t.max_depth(a) == expected["a"]
    assert t.max_depth(a1) == expected["a1"]


@pytest.mark.oracle
def test_max_depth_single_oracle(golden):
    t = tree("only")
    assert t.max_depth(t.begin()) == golden["max_depth_single"]["result"]


@pytest.mark.oracle
def test_count(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    a1 = t.append_child(a, "a1")
    t.append_child(a, "a2")
    t.append_child(t.begin(), "b")
    expected = golden["count"]
    assert t.count(t.begin()) == expected["root"]
    assert t.count(a) == expected["a"]
    assert t.count(a1) == expected["a1"]


@pytest.mark.oracle
def test_navigation(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    c = t.append_child(t.begin(), "c")
    b = t.child(t.begin(), 1)
    expected = golden["navigation"]
    assert t.child(t.begin(), 0).data == expected["child0"]
    assert t.child(t.begin(), 2).data == expected["child2"]
    assert [n.data for n in t.children(t.begin())] == expected["children"]
    assert t.number_of_children(t.begin()) == expected["number_of_children_root"]
    assert t.number_of_children(a) == expected["number_of_children_a"]
    assert t.number_of_siblings(b) == expected["number_of_siblings_b"]
    assert t.next_sibling(a).data == expected["next_sibling_a"]
    assert t.previous_sibling(c).data == expected["previous_sibling_c"]
    assert t.is_valid(t.next_sibling(c)) == (not expected["next_sibling_c_is_end"])
    assert t.is_valid(t.previous_sibling(a)) == (not expected["previous_sibling_a_is_end"])


@pytest.mark.oracle
def test_traversal_orders(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(a, "a1")
    t.append_child(a, "a2")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    validate(t)
    expected = golden["traversal_orders"]
    assert [n.data for n in t] == expected["pre_order"]
    assert [n.data for n in t.iter_post()] == expected["post_order"]
    assert [n.data for n in t.iter_leaf()] == expected["leaf_order"]


def test_iter_post_at_walks_only_the_given_subtree_in_post_order():
    """CARRY-FORWARD 1.4: `reduct::upwards` (meta_rules.cc:111) needs a
    post-order walk rooted at a subtree, not at the whole tree, which
    `iter_post()` does not offer. Same tree shape as `test_traversal_orders`
    so the whole-tree case (`t.begin()`) can be checked directly against
    `iter_post()`.
    """
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(a, "a1")
    t.append_child(a, "a2")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    validate(t)

    assert [n.data for n in t.iter_post_at(a)] == ["a1", "a2", "a"]
    assert [n.data for n in t.iter_post_at(b)] == ["b1", "b"]
    assert [n.data for n in t.iter_post_at(t.begin())] == [n.data for n in t.iter_post()]


def test_iter_post_at_of_a_leaf_yields_only_itself():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    validate(t)
    assert [n.data for n in t.iter_post_at(a)] == ["a"]


def test_iter_post_at_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        list(t.iter_post_at(None))


def test_iter_post_at_single_child_chain():
    """Task 0's review (progress.md) flagged this exact gap: no dedicated
    test for `iter_post_at` on a single-child CHAIN (`root->x->y->z`) --
    only the leaf and branching shapes above were covered. Hand-verified
    correct at Task 0 time; this pins it. Assigned to Task 2 (this task)
    since it is the consumer of this walk (`reduct.upwards`).

    Post-order on a chain is unambiguous regardless of where in the chain
    the walk is rooted: deepest-first, root last.
    """
    t = tree("root")
    x = t.append_child(t.begin(), "x")
    y = t.append_child(x, "y")
    z = t.append_child(y, "z")
    validate(t)

    assert [n.data for n in t.iter_post_at(t.begin())] == ["z", "y", "x", "root"]
    assert [n.data for n in t.iter_post_at(x)] == ["z", "y", "x"]
    assert [n.data for n in t.iter_post_at(y)] == ["z", "y"]
    assert [n.data for n in t.iter_post_at(z)] == ["z"]


def test_iter_pre_at_walks_only_the_given_subtree_in_pre_order():
    """C++: opencog/util/tree.h:2373 iterator_base::skip_children
    (`reduct::downwards`, meta_rules.cc:79-105). Same tree shape as
    `test_iter_post_at_walks_only_the_given_subtree_in_post_order` so the
    whole-tree case (`t.begin()`) can be checked directly against the
    plain pre-order walk (`__iter__`, via `list(t)`).
    """
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(a, "a1")
    t.append_child(a, "a2")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    validate(t)

    assert [n.data for n in t.iter_pre_at(a)] == ["a", "a1", "a2"]
    assert [n.data for n in t.iter_pre_at(b)] == ["b", "b1"]
    assert [n.data for n in t.iter_pre_at(t.begin())] == [n.data for n in t]


def test_iter_pre_at_single_child_chain():
    """Pre-order counterpart of `test_iter_post_at_single_child_chain`:
    parent-before-child regardless of where in the chain the walk starts.
    """
    t = tree("root")
    x = t.append_child(t.begin(), "x")
    y = t.append_child(x, "y")
    z = t.append_child(y, "z")
    validate(t)

    assert [n.data for n in t.iter_pre_at(t.begin())] == ["root", "x", "y", "z"]
    assert [n.data for n in t.iter_pre_at(x)] == ["x", "y", "z"]
    assert [n.data for n in t.iter_pre_at(y)] == ["y", "z"]
    assert [n.data for n in t.iter_pre_at(z)] == ["z"]


def test_iter_pre_at_of_a_leaf_yields_only_itself():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    validate(t)
    assert [n.data for n in t.iter_pre_at(a)] == ["a"]


def test_iter_pre_at_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        list(t.iter_pre_at(None))


def test_iter_pre_at_is_a_live_walk_not_a_snapshot():
    """`reduct::downwards` mutates the tree (applying its wrapped rule)
    BETWEEN steps of its own traversal, and C++'s live iterator picks up
    the consequences -- measured directly against the real library
    (`tools/cpp_probe/meta_rules_probe.cc`, the
    "downwards(live-mutation insert-after)" case: inserting a new sibling
    immediately after the node just visited gets visited by that SAME
    downwards pass, landing between the current node and its original
    next sibling). This is the same generic tree-level hazard
    `test_hazard_erase_next_sibling`/`test_hazard_replace_next_sibling`
    already pin for `children()`, reproduced here for `iter_pre_at`
    specifically since `downwards` (meta_rules.py) depends on it living
    over the CURRENT structure at each step, not a `list(...)` snapshot.
    """
    t = tree("root")
    x = t.append_child(t.begin(), "x")
    t.append_child(t.begin(), "y")
    t.append_child(t.begin(), "z")
    observed = []
    it = t.iter_pre_at(t.begin())
    for node in it:
        observed.append(node.data)
        if node.data == "x":
            t.insert_after(x, "w")
    validate(t)
    assert observed == ["root", "x", "w", "y", "z"]


@pytest.mark.oracle
def test_hazard_erase_next_sibling(golden):
    t = tree("root")
    t.append_child(t.begin(), "x")
    y = t.append_child(t.begin(), "y")
    t.append_child(t.begin(), "z")
    observed = []
    for child in t.children(t.begin()):
        observed.append(child.data)
        if child.data == "x":
            t.erase(y)
    validate(t)
    assert observed == golden["hazard_erase_next_sibling"]["observed"]


@pytest.mark.oracle
def test_hazard_replace_next_sibling(golden):
    t = tree("root")
    t.append_child(t.begin(), "x")
    y = t.append_child(t.begin(), "y")
    t.append_child(t.begin(), "z")
    observed = []
    for child in t.children(t.begin()):
        observed.append(child.data)
        if child.data == "x":
            t.replace(y, "REPLACED")
    validate(t)
    assert observed == golden["hazard_replace_next_sibling"]["observed"]


# ---- non-oracle unit tests: method edge cases named in the brief ----


def test_erase_the_root_empties_the_tree():
    t = tree("only")
    t.erase(t.begin())
    validate(t)
    assert t.empty()
    assert t.begin() is None


def test_flatten_a_childless_node_is_a_noop():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    same = t.flatten(a)
    validate(t)
    assert same is a
    assert t.number_of_children(t.begin()) == 1


def test_insert_above_on_the_root_makes_a_new_root():
    t = tree("R")
    new_root = t.insert_above(t.begin(), "N")
    validate(t)
    assert t.begin() is new_root
    assert t.begin().data == "N"
    assert t.number_of_children(t.begin()) == 1
    assert t.child(t.begin(), 0).data == "R"


def test_move_ontop_where_source_is_a_descendant_of_target():
    """C++ documents move_ontop as unsafe when `source` is inside `target`'s
    own subtree ("Warning: only use if source is not included in the
    subtree of target"). erase(target) recursively clears every
    descendant's own children *before* move_ontop rescues `source` -- so if
    `source` itself had children, they are destroyed as collateral damage
    even though `source` survives and gets relocated. This asserts that
    concrete, observable consequence (matching the literal erase_children
    recursion) instead of guessing at a crash: Python has no literal
    use-after-free to reproduce, since nothing is actually freed.
    """
    t = tree("root")
    target = t.append_child(t.begin(), "target")
    source = t.append_child(target, "source")
    t.append_child(source, "source_child")  # destroyed as collateral damage

    t.move_ontop(target, source)

    validate(t)
    assert t.number_of_children(t.begin()) == 1
    assert t.child(t.begin(), 0) is source
    assert t.number_of_children(source) == 0  # source's own child did not survive


def test_equal_subtree_structurally_equal_but_distinct_objects():
    t1 = tree("root")
    a1 = t1.append_child(t1.begin(), "a")
    t1.append_child(a1, "x")
    t2 = tree("root")
    a2 = t2.append_child(t2.begin(), "a")
    t2.append_child(a2, "x")
    assert t1.begin() is not t2.begin()
    assert t1.equal_subtree(t1.begin(), t2.begin())


def test_max_depth_of_a_single_node():
    t = tree("only")
    assert t.max_depth(t.begin()) == 0


def test_insert_on_the_tree_root_is_rejected():
    t = tree("root")
    with pytest.raises(ValueError):
        t.insert(t.begin(), "x")


def test_insert_after_on_the_tree_root_is_rejected():
    t = tree("root")
    with pytest.raises(ValueError):
        t.insert_after(t.begin(), "x")


def test_children_generator_is_live_not_a_snapshot():
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    gen = t.children(t.begin())
    first = next(gen)
    assert first.data == "a"
    t.append_child(t.begin(), "c")  # mutate before resuming the generator
    rest = [n.data for n in gen]
    assert rest == ["b", "c"]  # sees the mutation -- not a stale snapshot
    validate(t)  # live splices mid-iteration are exactly where a
    # malformed tree would most likely slip through unnoticed


# ---- new edge cases from the round-1 review ----


def test_validate_helper_catches_a_broken_back_link():
    """Proves the validator actually fires: a tree with a deliberately
    corrupted back-link must fail validate(), not pass silently. See the
    task report for the accompanying "break it for real, in the
    implementation, then restore it" demonstration -- this test instead
    corrupts a link on a throwaway tree object directly, so the
    demonstration doesn't have to live (and be forgotten) in the shipped
    suite.
    """
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    validate(t)  # sane before corruption

    a.parent = None  # corrupt: "a" still lives under root, but forgot to say so
    with pytest.raises(AssertionError):
        validate(t)


def test_validate_catches_a_skipped_first_child():
    """The exact corruption from the round-2 review: set root.first_child
    to `b`, skipping the true first child `a`, while leaving `a`'s own
    links (prev_sibling=None, next_sibling=b, parent=root) untouched.
    validate()'s original version passed this silently -- it only checked
    the reverse direction (parent->first_child/last_child pointing at a
    node reached by forward traversal from first_child), never
    first_child/last_child's own prev/next_sibling independently. `a`
    becomes an invisible orphan and number_of_children quietly drops from
    3 to 2, with nothing catching it.
    """
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    validate(t)  # sane before corruption
    assert t.number_of_children(t.begin()) == 3

    t.begin().first_child = b  # corrupt: skips `a` entirely
    assert t.number_of_children(t.begin()) == 2  # the silent symptom
    with pytest.raises(AssertionError):
        validate(t)


def test_move_after_root_target_is_rejected():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    with pytest.raises(ValueError):
        t.move_after(t.begin(), a)


def test_move_after_root_source_is_rejected():
    t = tree("root")
    other = tree("other")
    with pytest.raises(ValueError):
        t.move_after(other.begin(), t.begin())


def test_move_before_root_target_is_rejected():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    with pytest.raises(ValueError):
        t.move_before(t.begin(), a)


def test_move_before_root_source_is_rejected():
    t = tree("root")
    other = tree("other")
    with pytest.raises(ValueError):
        t.move_before(other.begin(), t.begin())


def test_move_ontop_root_target_is_rejected_and_leaves_tree_untouched():
    """The regression this guards: move_ontop used to call erase(target)
    before validating -- so a rejected move_ontop(root, ...) would empty
    the tree and *then* raise. The guard must run first.
    """
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    with pytest.raises(ValueError):
        t.move_ontop(t.begin(), a)
    assert not t.empty()
    assert t.begin().data == "root"
    assert t.number_of_children(t.begin()) == 1


def test_move_ontop_root_source_is_rejected():
    t = tree("root")
    other = tree("other")
    with pytest.raises(ValueError):
        t.move_ontop(other.begin(), t.begin())


@pytest.mark.parametrize("method", ["move_after", "move_before", "move_ontop"])
def test_move_methods_reject_none(method):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    fn = getattr(t, method)
    with pytest.raises(ValueError):
        fn(None, a)
    with pytest.raises(ValueError):
        fn(b, None)


def test_wrap_rejects_none():
    t = tree("root")
    with pytest.raises(ValueError):
        t.wrap(None, "x")


def test_child_out_of_range_raises():
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    with pytest.raises(IndexError):
        t.child(t.begin(), 2)
    with pytest.raises(IndexError):
        t.child(t.begin(), 5)
    with pytest.raises(IndexError):
        t.child(t.begin(), -1)


def test_child_in_range_still_works():
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    assert t.child(t.begin(), 0).data == "a"
    assert t.child(t.begin(), 1).data == "b"


def test_reparent_into_its_own_first_child_is_rejected():
    """Restores tree.h:1531's tree_assert(first != position.node): moving
    from_node's children onto from_node's own first child would relink
    that first child's prev/next pointers into a self-referential cycle.
    """
    t = tree("root")
    from_node = t.append_child(t.begin(), "from")
    first_child = t.append_child(from_node, "first")
    t.append_child(from_node, "second")
    with pytest.raises(ValueError):
        t.reparent(first_child, from_node)
    # rejected call must leave the tree structurally sound
    validate(t)
    assert t.number_of_children(from_node) == 2


# ---- round-2 review: null-node guards restoring tree.h's own asserts ----
# See tree.py's docstrings for the exact tree.h line each of these mirrors;
# enumerated by grepping every `tree_assert` in tree.h and checking each
# hit's enclosing function against our ported API surface (see the task
# report for the full method-by-method sweep, including what was
# deliberately left unguarded because tree.h itself has no check there).


def test_append_child_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.append_child(None, "x")


def test_prepend_child_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.prepend_child(None, "x")


def test_append_children_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.append_children(None, ["x", "y"])
    # the gap C++ itself doesn't close: an empty iterable used to skip
    # straight past the loop to a null deref
    with pytest.raises(ValueError):
        t.append_children(None, [])


def test_children_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.children(None)
    with pytest.raises(ValueError):
        list(t.children(None))


def test_next_sibling_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.next_sibling(None)


def test_previous_sibling_rejects_none_node():
    t = tree("root")
    with pytest.raises(ValueError):
        t.previous_sibling(None)


def test_replace_rejects_none_node():
    t = tree("root")
    src = tree("s")
    with pytest.raises(ValueError):
        t.replace(None, "x")
    with pytest.raises(ValueError):
        t.replace(None, src.begin())


# ---------------------------------------------------------------------------
# CARRY-FORWARD 1.3: erase()/replace() must not leave a discarded node's own
# links pointing into the live tree. Not cxxtest ports -- regression tests
# for the DIVERGENCE fix (C++ dangles here; this port fails fast instead).
# ---------------------------------------------------------------------------


def test_erase_leaves_the_discarded_nodes_own_links_dead():
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.erase(b)
    validate(t)

    # The tree itself is intact and does not route through the erased node.
    assert [n.data for n in t] == ["root", "a", "c"]

    # But a handle retained past the erase() cannot navigate through it:
    # its own parent/prev_sibling/next_sibling are poisoned, not left
    # pointing at the (still live) neighbors they used to point at.
    with pytest.raises(AttributeError):
        _ = b.next_sibling.data
    with pytest.raises(AttributeError):
        _ = b.prev_sibling.data
    with pytest.raises(AttributeError):
        _ = b.parent.data


def test_replace_subtree_leaves_the_discarded_nodes_own_links_dead():
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(b, "b1")
    t.append_child(t.begin(), "c")
    src = tree("s")
    src.append_child(src.begin(), "s1")
    new_node = t.replace(b, src.begin())
    validate(t)

    assert new_node is not b
    assert [n.data for n in t] == ["root", "a", "s", "s1", "c"]

    with pytest.raises(AttributeError):
        _ = b.next_sibling.data
    with pytest.raises(AttributeError):
        _ = b.prev_sibling.data
    with pytest.raises(AttributeError):
        _ = b.parent.data

    # Only the discarded node's OWN outward links are poisoned -- its own
    # (now orphaned) subtree is still self-consistent and readable.
    assert b.data == "b"
    assert b.first_child.data == "b1"


def test_replace_data_in_place_does_not_poison_the_reused_node():
    """The plain-data overload of replace() mutates `node` in place and
    returns the SAME object (tree.py's replace() docstring) -- unlike the
    subtree-copy overload, there is no discarded handle here to poison.
    """
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    same = t.replace(b, "B2")
    validate(t)
    assert same is b
    assert b.data == "B2"
    assert b.next_sibling.data == "c"
    assert b.prev_sibling.data == "a"


def test_tree_constructed_from_a_node_deep_copies_its_subtree():
    """C++: opencog/util/tree.h:658 tree(const iterator_base&) -- S7's
    eval.py/assumption.py both need this "copy a node's subtree into a
    fresh standalone tree" overload (see tree.py's __init__ docstring)."""
    src = tree("root")
    a = src.append_child(src.begin(), "a")
    src.append_child(a, "a1")
    src.append_child(src.begin(), "b")

    copy = tree(a)
    validate(copy)
    assert [n.data for n in copy] == ["a", "a1"]
    assert copy.begin() is not a


def test_tree_constructed_from_a_node_is_independent_of_the_source():
    """The copy is a genuine deep copy, not aliased: mutating the source
    after construction must not be visible in the new tree, and vice
    versa."""
    src = tree("root")
    a = src.append_child(src.begin(), "a")
    src.append_child(a, "a1")

    copy = tree(a)
    src.append_child(a, "a2")
    assert [n.data for n in copy] == ["a", "a1"]

    copy.append_child(copy.begin(), "only-in-copy")
    assert [n.data for n in src] == ["root", "a", "a1", "a2"]


def test_tree_constructed_from_a_leaf_node_has_no_children():
    src = tree("root")
    leaf = src.append_child(src.begin(), "leaf")
    copy = tree(leaf)
    validate(copy)
    assert [n.data for n in copy] == ["leaf"]
    assert copy.begin().first_child is None


def test_tree_constructed_from_an_erased_node_raises():
    # task-1 review, Minor: without this check, tree(erased_node) was a
    # new silent read path through a dead handle -- erase()'s whole point
    # is that a stale handle fails LOUDLY at the first attempt to
    # navigate through it, and this constructor previously never tried
    # (it only reads .data/.first_child, neither of which erase() itself
    # poisons), so it slipped through undetected.
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.erase(b)

    with pytest.raises(ValueError):
        tree(b)


def test_tree_constructed_from_a_replaced_nodes_discarded_handle_raises():
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    src = tree("s")
    src.append_child(src.begin(), "s1")
    t.replace(b, src.begin())

    with pytest.raises(ValueError):
        tree(b)


def test_tree_constructed_from_a_genuine_root_is_still_fine():
    # The erased-handle guard must not misfire against a legitimately
    # rootless node (parent is None, not _DEAD) -- this is the ordinary,
    # everyday case tree(node) exists for.
    src = tree("root")
    src.append_child(src.begin(), "a")
    copy = tree(src.begin())
    assert [n.data for n in copy] == ["root", "a"]


# ---------------------------------------------------------------------------
# partition / sort_on_subtrees / reparent(3-arg) / is_childless / has_one_child
#
# Added task-3 (S7) for comboreduct/reduct/general_rules.py's
# eval_constants/reorder_commutative. NOT part of the tree_probe.cc +
# tests/fixtures/tree.json golden-fixture convention above -- these follow
# reduct/test_meta_rules.py's own precedent instead (a task-scoped C++ probe,
# tools/cpp_probe/partition_probe.cc, with its measured stdout pinned here as
# literal expected values, not re-run as part of the normal test suite). See
# that probe file and task-3-report.md for the full transcript.
# ---------------------------------------------------------------------------


def test_is_childless():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    assert t.is_childless(a) is True
    assert t.is_childless(t.begin()) is False


def test_has_one_child():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    assert t.has_one_child(t.begin()) is True
    t.append_child(t.begin(), "b")
    assert t.has_one_child(t.begin()) is False
    assert t.has_one_child(a) is False  # a itself is childless


def _is_const(s):
    return s.startswith("C")


def test_partition_case_a_interleaved_single_swap():
    # C++: partition_probe.cc "partition_case_a" ->
    # {"result":["C1","C2","C3","V1","V2"],"boundary":"V1"}
    t = tree("root")
    for s in ("C1", "C2", "V1", "C3", "V2"):
        t.append_child(t.begin(), s)
    boundary = t.partition(t.begin(), _is_const)
    validate(t)
    assert [c.data for c in t.children(t.begin())] == ["C1", "C2", "C3", "V1", "V2"]
    assert boundary.data == "V1"


def test_partition_case_b_is_not_a_stable_partition():
    # C++: partition_probe.cc "partition_case_b" ->
    # {"result":["C1","C3","C2","V2","V1"],"boundary":"V2"}
    # A STABLE partition of ["C1","V1","C2","V2","C3"] would give
    # ["C1","C2","C3","V1","V2"] -- the real algorithm gives something
    # else entirely, with BOTH groups internally reordered.
    t = tree("root")
    for s in ("C1", "V1", "C2", "V2", "C3"):
        t.append_child(t.begin(), s)
    boundary = t.partition(t.begin(), _is_const)
    validate(t)
    assert [c.data for c in t.children(t.begin())] == ["C1", "C3", "C2", "V2", "V1"]
    assert boundary.data == "V2"


def test_partition_all_matching_returns_none_boundary():
    # C++: partition_probe.cc "partition_case_c" ->
    # {"result":["C1","C2","C3"],"boundary":"<end>"}
    t = tree("root")
    for s in ("C1", "C2", "C3"):
        t.append_child(t.begin(), s)
    boundary = t.partition(t.begin(), _is_const)
    assert [c.data for c in t.children(t.begin())] == ["C1", "C2", "C3"]
    assert boundary is None


def test_partition_none_matching():
    # C++: partition_probe.cc "partition_case_d" ->
    # {"result":["V1","V2","V3"],"boundary":"V1"}
    t = tree("root")
    for s in ("V1", "V2", "V3"):
        t.append_child(t.begin(), s)
    boundary = t.partition(t.begin(), _is_const)
    assert [c.data for c in t.children(t.begin())] == ["V1", "V2", "V3"]
    assert boundary.data == "V1"


def test_partition_single_constant_child():
    # C++: partition_probe.cc "partition_case_e" ->
    # {"result":["C1"],"boundary":"<end>"} -- the probe's actual case,
    # previously swapped for an unmeasured empty-children variant below
    # (task-3 review, Minor).
    t = tree("root")
    t.append_child(t.begin(), "C1")
    boundary = t.partition(t.begin(), _is_const)
    assert [c.data for c in t.children(t.begin())] == ["C1"]
    assert boundary is None


def test_partition_no_children_returns_none():
    # Not one of partition_probe.cc's own cases (C++'s `partition` is
    # never called on a genuinely childless node in this port's real
    # call site either) -- a from==to input is degenerate but still
    # worth guarding: `tree.partition`'s own docstring traces exactly why
    # this returns immediately with no relinking attempted.
    t = tree("only")
    boundary = t.partition(t.begin(), _is_const)
    assert boundary is None


def test_sort_on_subtrees_is_stable_for_ties():
    # C++: partition_probe.cc "sort_on_subtrees_ties" ->
    # {"result":["a","bb","cc","ddd"]} -- "bb"/"cc" are both length 2
    # (tied); original relative order (bb before cc) is preserved.
    t = tree("root")
    for s in ("bb", "a", "cc", "ddd"):
        t.append_child(t.begin(), s)

    def by_len(a, b):
        return len(a.data) < len(b.data)

    t.sort_on_subtrees(t.begin(), by_len, False)
    validate(t)
    assert [c.data for c in t.children(t.begin())] == ["a", "bb", "cc", "ddd"]


def test_sort_on_subtrees_deep_sorts_children_before_parent_level():
    # C++: partition_probe.cc "sort_on_subtrees_deep" ->
    # {"top":["P","Q"],"p_children":["x","y","z"],"q_children":["m","n"]}
    t = tree("root")
    p = t.append_child(t.begin(), "P")
    for s in ("z", "y", "x"):
        t.append_child(p, s)
    q = t.append_child(t.begin(), "Q")
    for s in ("n", "m"):
        t.append_child(q, s)

    def by_val(a, b):
        return a.data < b.data

    t.sort_on_subtrees(t.begin(), by_val, True)
    validate(t)
    assert [c.data for c in t.children(t.begin())] == ["P", "Q"]
    assert [c.data for c in t.children(p)] == ["x", "y", "z"]
    assert [c.data for c in t.children(q)] == ["m", "n"]


def test_reparent_two_arg_moves_all_children():
    t = tree("root")
    src = t.append_child(t.begin(), "src")
    t.append_child(src, "s1")
    t.append_child(src, "s2")
    dst = t.append_child(t.begin(), "dst")
    t.reparent(dst, src)
    validate(t)
    assert list(t.children(src)) == []
    assert [c.data for c in t.children(dst)] == ["s1", "s2"]


def test_reparent_three_arg_moves_only_the_given_sibling_range():
    # C++ ground truth: general_rules_probe.cc's eval_constants_mixed_plus
    # case builds exactly this shape by hand (tr.reparent(new_op,
    # it.begin(), to)) -- pinned directly here at the tree.py level too.
    t = tree("and")
    root = t.begin()
    t.append_child(root, "C1")
    t.append_child(root, "C2")
    v1 = t.append_child(root, "V1")
    new_op = t.append_child(root, "and")

    t.reparent(new_op, root.first_child, v1)
    validate(t)
    assert [c.data for c in t.children(root)] == ["V1", "and"]
    assert [c.data for c in t.children(new_op)] == ["C1", "C2"]


def test_reparent_omitted_third_arg_is_not_confused_with_a_range_call():
    # The 2-arg and 3-arg forms must stay genuinely distinguishable --
    # calling with only (new_parent, from_node) must take the "gather ALL
    # of from_node's children" path, never be misread as a 0-length range.
    t = tree("root")
    src = t.append_child(t.begin(), "src")
    only = t.append_child(src, "only")
    dst = t.append_child(t.begin(), "dst")
    result = t.reparent(dst, src)
    assert result is only  # the moved range's start (tree.h:1525's own `first`)
    assert list(t.children(src)) == []
    assert [c.data for c in t.children(dst)] == ["only"]


# ---------------------------------------------------------------------------
# _copy_value -- subtree-copy operations must value-copy mutable node data,
# not alias it (task-3 review, Critical). Uses the REAL `combo_tree`/
# `argument` (imported locally, not at module level, to keep this file's
# own "generic tree<T>" test data -- plain strings elsewhere -- unchanged)
# because `argument` is this port's one concrete, reachable example of
# mutable node data, and the demonstrated failure is combo-shaped:
# Task 4's `push_back_negated_arguments` (logical_rules.h:235,
# logical_rules.cc:204: `*it = *it.begin(); erase_children(it);
# get_argument(*it).negate();`, reached from `reduce_to_enf`,
# logical_rules.cc:643 -- the central function of the whole next task)
# copies a subtree and negates ONLY the copy's own root; the original
# must never observe that mutation.
# ---------------------------------------------------------------------------


def test_combo_tree_copy_constructor_value_copies_a_mutable_argument():
    from pymetta.comboreduct.combo.argument import argument
    from pymetta.comboreduct.combo.vertex import combo_tree

    original = combo_tree(argument(1))
    copy_tr = combo_tree(original.begin())  # tree(const iterator_base&)
    copy_tr.begin().data.negate()

    assert original.begin().data.idx == 1, "negating the COPY corrupted the ORIGINAL"
    assert copy_tr.begin().data.idx == -1


def test_append_child_subtree_copy_value_copies_a_mutable_argument():
    from pymetta.comboreduct.combo.argument import argument
    from pymetta.comboreduct.combo.vertex import combo_tree

    src = combo_tree(argument(1))
    dst = combo_tree("root")
    copied = dst.append_child(dst.begin(), src.begin())
    copied.data.negate()

    assert src.begin().data.idx == 1, "negating the COPY corrupted the ORIGINAL"
    assert copied.data.idx == -1


def test_push_back_negated_arguments_shaped_copy_then_negate_leaves_source_untouched():
    """The exact shape Task 4's `push_back_negated_arguments`
    (logical_rules.h:235) needs: copy ONE child of a live tree out to a
    standalone tree, negate the copy's own root argument in place, and
    confirm the LIVE tree's own child is completely unaffected.
    Demonstrated failing before the `tree.py` `_copy_value` fix
    (task-3-report.md carries the reproduction): `and($1 $2)`'s `$1`
    child, copied and negated this way, used to also flip the ORIGINAL
    `$1` to `!$1` in place, corrupting `and($1 $2)` into `and(!$1 $2)`.
    """
    from pymetta.comboreduct.combo.argument import argument
    from pymetta.comboreduct.combo.vertex import builtin, combo_tree

    live = combo_tree(builtin.logical_and)
    root = live.begin()
    live.append_child(root, argument(1))
    live.append_child(root, argument(2))
    first_child = root.first_child

    standalone = combo_tree(first_child)
    standalone.begin().data.negate()

    assert first_child.data.idx == 1, "the live tree's own child was corrupted"
    assert [c.data.idx for c in live.children(root)] == [1, 2]
    assert standalone.begin().data.idx == -1
