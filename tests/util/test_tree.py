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


# ---- oracle: replay each probe case, compare structural / value result ----


@pytest.mark.oracle
def test_append_child_data(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    assert to_sexpr(t) == golden["append_child_data"]["result"]


@pytest.mark.oracle
def test_append_child_subtree(golden):
    src = tree("s")
    src.append_child(src.begin(), "s1")
    src.append_child(src.begin(), "s2")
    dst = tree("d")
    dst.append_child(dst.begin(), src.begin())
    assert to_sexpr(dst) == golden["append_child_subtree"]["dest"]
    assert to_sexpr(src) == golden["append_child_subtree"]["source_untouched"]


@pytest.mark.oracle
def test_prepend_child(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    t.prepend_child(t.begin(), "z")
    assert to_sexpr(t) == golden["prepend_child"]["result"]


@pytest.mark.oracle
def test_append_children(golden):
    t = tree("root")
    t.append_children(t.begin(), ["x", "y", "z"])
    assert to_sexpr(t) == golden["append_children"]["result"]


@pytest.mark.oracle
def test_insert(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.insert(b, "n")
    assert to_sexpr(t) == golden["insert"]["result"]


@pytest.mark.oracle
def test_insert_after(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.insert_after(b, "n")
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
    assert to_sexpr(t) == golden["insert_above_internal"]["result"]


@pytest.mark.oracle
def test_insert_above_root(golden):
    t = tree("R")
    t.append_child(t.begin(), "r1")
    t.append_child(t.begin(), "r2")
    t.insert_above(t.begin(), "N")
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
    assert to_sexpr(t) == golden["erase"]["result"]
    assert ret.data == golden["erase"]["returned"]


@pytest.mark.oracle
def test_erase_root(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    ret = t.erase(t.begin())
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
    assert to_sexpr(t) == golden["erase_children"]["result"]


@pytest.mark.oracle
def test_erase_children_leaf(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.erase_children(a)
    assert to_sexpr(t) == golden["erase_children_leaf"]["result"]


@pytest.mark.oracle
def test_replace_data(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    t.replace(b, "B2")
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
    assert to_sexpr(t) == golden["flatten"]["result"]


@pytest.mark.oracle
def test_flatten_childless(golden):
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    t.append_child(t.begin(), "b")
    t.flatten(a)
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
    assert to_sexpr(t) == golden["wrap"]["result"]


@pytest.mark.oracle
def test_reparent(golden):
    t = tree("root")
    src = t.append_child(t.begin(), "src")
    t.append_child(src, "s1")
    t.append_child(src, "s2")
    dst = t.append_child(t.begin(), "dst")
    t.reparent(dst, src)
    assert to_sexpr(t) == golden["reparent"]["result"]


@pytest.mark.oracle
def test_move_after(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    d = t.append_child(t.begin(), "d")
    t.move_after(b, d)
    assert to_sexpr(t) == golden["move_after"]["result"]


@pytest.mark.oracle
def test_move_before(golden):
    t = tree("root")
    t.append_child(t.begin(), "a")
    b = t.append_child(t.begin(), "b")
    t.append_child(t.begin(), "c")
    d = t.append_child(t.begin(), "d")
    t.move_before(b, d)
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
    assert to_sexpr(t) == golden["move_ontop"]["result"]


@pytest.mark.oracle
def test_move_ontop_adjacent_sibling_self_cycle_is_an_inherited_bug(golden):
    # Deliberately reproduces the upstream defect documented on
    # tree.move_ontop -- see its DIVERGENCE comment. Pointer inspection
    # only: DO NOT iterate/traverse `t` after this -- that hangs by design,
    # exactly as the equivalent C++ probe case does (and exactly why the
    # probe itself never destructs the corresponding tree either).
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
    expected = golden["traversal_orders"]
    assert [n.data for n in t] == expected["pre_order"]
    assert [n.data for n in t.iter_post()] == expected["post_order"]
    assert [n.data for n in t.iter_leaf()] == expected["leaf_order"]


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
    assert observed == golden["hazard_replace_next_sibling"]["observed"]


# ---- non-oracle unit tests: method edge cases named in the brief ----


def test_erase_the_root_empties_the_tree():
    t = tree("only")
    t.erase(t.begin())
    assert t.empty()
    assert t.begin() is None


def test_flatten_a_childless_node_is_a_noop():
    t = tree("root")
    a = t.append_child(t.begin(), "a")
    same = t.flatten(a)
    assert same is a
    assert t.number_of_children(t.begin()) == 1


def test_insert_above_on_the_root_makes_a_new_root():
    t = tree("R")
    new_root = t.insert_above(t.begin(), "N")
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
