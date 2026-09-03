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

    For every node in `node`'s subtree (the whole tree if `node` is
    omitted): each child's `parent` points back to it; `first_child` /
    `last_child` are the true ends of the sibling chain; `next_sibling` /
    `prev_sibling` are mutual inverses. Do not call this on a tree known to
    contain the move_ontop adjacent-sibling self-cycle (see
    test_move_ontop_adjacent_sibling_self_cycle_is_an_inherited_bug) --
    walking children() there does not terminate, by design.
    """
    node = node if node is not None else t.begin()
    if node is None:
        return
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
