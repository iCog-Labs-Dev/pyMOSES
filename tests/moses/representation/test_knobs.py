import copy
import json

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import (
    combo_tree_to_str,
    str_to_combo_tree,
)
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, vertex_equal
from pymetta.comboreduct.reduct.clean_reduction import clean_reduce
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.moses.representation.knobs import logical_subtree_knob
from pymetta.util.tree import tree


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "logical_knob.json").read_text())


def _render(tr: combo_tree) -> str:
    return combo_tree_to_str(tr).strip()


def _reduced(source: combo_tree) -> str:
    tr = combo_tree(source.begin())
    clean_reduce(tr)
    logical_reduction(set())(2)(tr)
    return _render(tr)


def _logical_knob(base: str, arg_idx: int):
    tr = str_to_combo_tree(base)
    logical_reduction(set())(2)(tr)
    subtree = combo_tree(argument(arg_idx))
    return tr, subtree, logical_subtree_knob(tr, tr.begin(), subtree.begin())


def _node_value(node) -> str:
    if node is None:
        return ""
    return _render(combo_tree(copy.copy(node.data)))


def test_set_head_copies_value_and_rejects_nonempty_without_mutation():
    tr = tree()
    value = argument(4)
    root = tr.set_head(value)
    value.negate()

    assert tr.begin() is root
    assert root.data == argument(4)
    assert tr.size() == 1

    before_root = tr.begin()
    with pytest.raises(ValueError, match="must be empty"):
        tr.set_head(argument(9))
    assert tr.begin() is before_root
    assert tr.begin().data == argument(4)
    assert tr.size() == 1


@pytest.mark.oracle
def test_fixture_schema_and_inputs_are_non_vacuous(golden):
    assert set(golden) == {
        "active",
        "transitions",
        "append",
        "disallow",
        "copy",
        "clear",
    }
    assert golden["active"]["base"] == "and($1 not($2))"
    assert golden["active"]["knobs"] == [
        "[$1 nil !$1]",
        "[!$2 nil $2]",
        "[nil $3 !$3]",
    ]
    assert [(row["knob"], row["idx"]) for row in golden["active"]["turns"]] == [
        (knob, idx) for knob in range(1, 4) for idx in range(3)
    ]

    pinned = [
        ("existing-positive", "and($1 $9)", 1),
        ("existing-negated", "and(not($1) $9)", 1),
        ("missing", "and($1 $9)", 2),
    ]
    assert [(c["name"], c["base"], c["arg"]) for c in golden["transitions"]] == pinned
    assert [(c["name"], c["base"], c["arg"]) for c in golden["append"]] == pinned
    for case in golden["transitions"]:
        assert [row["idx"] for row in case["states"]] == [0, 1, 2, 0, 2, 1, 0]
    for case in golden["append"]:
        assert [(row["idx"], row["nested"]) for row in case["rows"]] == [
            (idx, nested) for idx in range(3) for nested in (False, True)
        ]
    assert [
        (row["name"], row["base"], row["arg"], row["setting"])
        for row in golden["disallow"]
    ] == [
        (name, "and($1 $9)", arg_idx, setting)
        for name, arg_idx in (("missing", 2), ("existing-positive", 1))
        for setting in range(3)
    ]
    assert [
        (
            row["name"],
            row["base"],
            row["arg"],
            row["source_turn"],
            row["disallowed"],
            row["destination_mode"],
        )
        for row in golden["copy"]
    ] == [
        ("existing-absent-disallowed-negated", "and($1 $9)", 1, 1, 2, "source-copy"),
        ("missing-negated-disallowed-present", "and($1 $9)", 2, 2, 1, "fresh-base"),
    ]
    assert [
        (row["name"], row["base"], row["arg"], row["turn"], row["disallowed"])
        for row in golden["clear"]
    ] == [
        ("existing-negated", "and(not($1) $9)", 1, 2, -1),
        ("existing-disallowed-absent", "and($1 $9)", 1, -1, 0),
        ("missing-absent", "and($1 $9)", 2, 0, -1),
        ("missing-present", "and($1 $9)", 2, 1, -1),
        ("missing-negated-disallowed-absent", "and($1 $9)", 2, 1, 0),
    ]


@pytest.mark.oracle
def test_active_knob_turning_cases_match_classic(golden):
    case = golden["active"]
    tr = str_to_combo_tree(case["base"])
    logical_reduction(set())(2)(tr)
    knobs = []
    for idx in range(1, 4):
        subtree = combo_tree(argument(idx))
        knobs.append(logical_subtree_knob(tr, tr.begin(), subtree.begin()))

    assert _render(tr) == case["prototype"]
    assert [knob.toStr() for knob in knobs] == case["knobs"]
    previous_knob = None
    for row in case["turns"]:
        if previous_knob is not None and row["knob"] != previous_knob:
            knobs[previous_knob - 1].turn(0)
        knob = knobs[row["knob"] - 1]
        knob.turn(row["idx"])
        assert _render(tr) == row["tree"]
        assert _reduced(tr) == row["reduced"]
        if row["idx"] == 2:
            knob.turn(row["idx"])
            assert _render(tr) == row["tree"]
        previous_knob = row["knob"]
    for knob in knobs:
        knob.turn(0)


@pytest.mark.oracle
def test_all_unreduced_turn_transitions_match_classic(golden):
    for case in golden["transitions"]:
        tr, _subtree, knob = _logical_knob(case["base"], case["arg"])
        assert _render(tr) == case["initial_tree"]
        assert knob.toStr() == case["knob"]
        assert knob.in_exemplar() is case["in_exemplar"]
        assert knob.multiplicity() == case["multiplicity"]
        assert knob.spec().multy == case["spec_multy"]

        for row in case["states"]:
            knob.turn(row["idx"])
            assert _render(tr) == row["tree"]
            assert _reduced(tr) == row["reduced"]
            assert knob.complexity_bound() == row["complexity"]


@pytest.mark.oracle
def test_append_to_empty_and_nested_candidates_matches_classic(golden):
    for case in golden["append"]:
        tr, _subtree, knob = _logical_knob(case["base"], case["arg"])
        prototype_before = _render(tr)
        for row in case["rows"]:
            candidate = combo_tree()
            parent = None
            if row["nested"]:
                parent = candidate.set_head(builtin.logical_and)
                candidate.append_child(parent, argument(9))

            source, updated_parent = knob.append_to(candidate, parent, row["idx"])

            assert _render(candidate) == row["candidate"]
            classic_absent = row["source_is_candidate_end"] or not row["source_valid"]
            assert (source is None) is classic_absent
            if source is not None:
                assert not row["source_is_exemplar_end"]
                assert _render(combo_tree(source)) == row["source"]
            assert _node_value(updated_parent) == row["parent"]
            assert _render(tr) == prototype_before


def test_append_to_returns_independent_destination_payload():
    tr, _subtree, knob = _logical_knob("and($1 $9)", 2)
    candidate = combo_tree()

    source, parent = knob.append_to(candidate, None, 1)

    assert source is knob.get_loc().first_child
    assert parent is candidate.begin()
    assert parent.data == source.data == argument(2)
    assert parent.data is not source.data
    parent.data.negate()
    assert _render(candidate) == "!$2"
    assert _render(tr) == "and($1 $9 null_vertex($2))"


@pytest.mark.oracle
def test_disallow_preserves_classic_map_idx_quirk(golden):
    for case in golden["disallow"]:
        _tr, _subtree, knob = _logical_knob(case["base"], case["arg"])
        knob.disallow(case["setting"])
        assert knob.multiplicity() == case["multiplicity"]
        assert knob.spec().multy == case["spec_multy"]
        assert knob.toStr() == case["knob"]

    _tr, _subtree, knob = _logical_knob("and($1 $9)", 2)
    knob.disallow(0)
    knob.turn(0)
    assert knob._current == knob.absent
    knob.turn(1)
    assert knob._current == knob.negated

    _tr, _subtree, knob = _logical_knob("and($1 $9)", 1)
    knob.disallow(0)
    knob.turn(0)
    assert knob._current == knob.negated


def test_constructor_uses_first_immediate_match_and_does_not_mutate_probe():
    tr = str_to_combo_tree("and($1 $1 $9)")
    first, second, _third = list(tr.children(tr.begin()))
    subtree = combo_tree(argument(1))
    original_value = subtree.begin().data
    original_text = _render(subtree)

    knob = logical_subtree_knob(tr, tr.begin(), subtree.begin())

    assert knob.get_loc() is first
    assert knob.get_loc() is not second
    assert subtree.begin().data is original_value
    assert _render(subtree) == original_text == "$1"
    assert _render(tr) == "and($1 $1 $9)"


def test_copy_constructor_preserves_existing_current_and_disallowed_state():
    source_tr, _subtree, source = _logical_knob("and($1 $9)", 1)
    source.disallow(2)
    source.turn(1)
    destination = combo_tree(source_tr.begin())

    copied = logical_subtree_knob(destination, destination.begin(), source)

    assert copied._default == source.present
    assert copied._current == source.absent
    assert copied._disallowed == {source.negated}
    assert copied.toStr() == "[$1 nil]"
    assert copied.get_loc() is destination.begin().first_child
    copied.turn(0)
    assert _render(destination) == "and($1 $9)"


def test_copy_constructor_appends_missing_knob_current_shape_and_state():
    source_tr, _subtree, source = _logical_knob("and($1 $9)", 2)
    source.turn(2)
    source.disallow(1)
    destination = str_to_combo_tree("and($1 $9)")

    copied = logical_subtree_knob(destination, destination.begin(), source)

    assert copied._default == source.absent
    assert copied._current == source.negated
    assert copied._disallowed == {source.present}
    assert _render(destination) == "and($1 $9 not($2))"
    candidate = combo_tree()
    source_node, parent = copied.append_to(candidate, None, 1)
    assert source_node is copied.get_loc().first_child
    assert parent is candidate.begin()
    assert _render(candidate) == "$2"
    assert _render(source_tr) == "and($1 $9 not($2))"


@pytest.mark.oracle
def test_copy_constructor_cases_match_classic(golden):
    for case in golden["copy"]:
        source_tr, _subtree, source = _logical_knob(case["base"], case["arg"])
        source.turn(case["source_turn"])
        source.disallow(case["disallowed"])
        if case["destination_mode"] == "source-copy":
            destination = combo_tree(source_tr.begin())
        else:
            destination = str_to_combo_tree(case["base"])
            logical_reduction(set())(2)(destination)

        assert _render(source_tr) == case["source_tree"]
        assert _render(destination) == case["destination_before"]
        copied = logical_subtree_knob(destination, destination.begin(), source)

        assert _render(destination) == case["destination_after"]
        assert copied.toStr() == case["knob"]
        assert copied.in_exemplar() is case["in_exemplar"]
        assert copied.multiplicity() == case["multiplicity"]
        assert copied.spec().multy == case["spec_multy"]
        for state in case["states"]:
            copied.turn(state["idx"])
            assert _render(destination) == state["tree"]
            assert _reduced(destination) == state["reduced"]
            assert copied.complexity_bound() == state["complexity"]


@pytest.mark.oracle
def test_clear_exemplar_cases_match_classic(golden):
    for case in golden["clear"]:
        tr, _subtree, knob = _logical_knob(case["base"], case["arg"])
        if case["disallowed"] >= 0:
            knob.disallow(case["disallowed"])
        if case["turn"] >= 0:
            knob.turn(case["turn"])
        assert _render(tr) == case["before"]

        knob.clear_exemplar()

        assert _render(tr) == case["after"]
        assert knob.in_exemplar() is case["retained"]
        if case["retained"]:
            assert knob.toStr() == case["after_knob"]
        else:
            assert case["after_knob"] == ""
            with pytest.raises(ValueError, match="erased or unset"):
                knob.get_loc()


@pytest.mark.parametrize("state", [0, 1, 2])
def test_clear_missing_knob_erases_every_current_shape_and_rejects_reuse(state):
    tr, _subtree, knob = _logical_knob("and($1 $9)", 2)
    knob.turn(state)
    stale = knob.get_loc()

    knob.clear_exemplar()

    assert _render(tr) == "and($1 $9)"
    assert not tr.is_valid(stale)
    for operation in (
        knob.get_loc,
        knob.toStr,
        knob.complexity_bound,
        knob.clear_exemplar,
        lambda: knob.turn(0),
        lambda: knob.append_to(combo_tree(), None, 0),
    ):
        with pytest.raises(ValueError, match="erased or unset"):
            operation()


def test_clear_existing_knob_restores_default_and_stays_live():
    tr, _subtree, knob = _logical_knob("and($1 $9)", 1)
    knob.turn(2)

    knob.clear_exemplar()

    assert _render(tr) == "and($1 $9)"
    assert knob.get_loc() is tr.begin().first_child
    assert knob._current == knob.present


def test_discrete_setting_validation_is_explicit():
    _tr, _subtree, knob = _logical_knob("and($1 $9)", 2)
    for bad in (-1, 3, True, 1.0, "1"):
        with pytest.raises(ValueError, match="setting"):
            knob.disallow(bad)
        with pytest.raises(ValueError, match="setting"):
            knob.allow(bad)
        with pytest.raises(ValueError, match="index"):
            knob.turn(bad)

    knob.disallow(2)
    with pytest.raises(ValueError, match="index"):
        knob.turn(2)


def test_allow_reenables_a_valid_setting_before_mapper_insertion():
    _tr, _subtree, knob = _logical_knob("and($1 $9)", 2)
    knob.disallow(2)
    assert knob.multiplicity() == 2

    knob.allow(2)
    knob.allow(2)

    assert knob.multiplicity() == 3
    assert knob.spec().multy == 3
    assert knob.toStr() == "[nil $2 !$2]"


def test_loc_str_null_wrapper_check_discriminates_contin_zero_from_null_vertex():
    """Important 3 repro #1 (review knobs.py:325): `loc.data ==
    builtin.null_vertex` was bare `==`. `builtin.null_vertex` is ordinal 0
    and `contin_t` is a bare `float`, so a live contin leaf valued `0.0`
    satisfied that bare comparison exactly like a real null-vertex wrapper
    node would. After `turn(1)` flattens an absent-default knob to
    `present`, `_loc` points directly at that childless contin leaf -- the
    bare check misread it as a still-wrapped null marker and demanded
    (wrongly) that it have exactly one child, raising instead of rendering.
    `vertex_equal` fixes this by refusing to equate a `float` with a
    `builtin` regardless of ordinal. 7.0 (a non-colliding ordinal) already
    worked before the fix and is kept here as the control.
    """

    def _knob_on_contin(value: float) -> logical_subtree_knob:
        tr = combo_tree()
        target = tr.set_head(builtin.logical_and)
        subtree = combo_tree(float(value))
        return logical_subtree_knob(tr, target, subtree.begin())

    control = _knob_on_contin(7.0)
    control.turn(1)
    assert control.toStr() == "[nil 7 !7]"

    knob = _knob_on_contin(0.0)
    knob.turn(1)
    assert knob.toStr() == "[nil 0 !0]"


def test_constructor_binds_to_matching_subtree_not_a_type_coincident_sibling():
    """Important 3 repro #2 (review knobs.py:232 -> tree.py's equal_subtree):
    `tr.equal_subtree` compared `.data` with bare `!=`/`==`. `builtin` is an
    `IntEnum` (`logical_true == 1`) and `contin_t` is a bare `float`, so a
    contin sibling valued `1.0` compared equal to a probed `logical_true`
    subtree purely by ordinal coincidence -- something `boost::variant`'s
    real `operator==` can never do. The constructor then bound `_loc` to
    that unrelated contin sibling instead of appending a fresh null_vertex
    wrapper: `in_exemplar()` reported `True` and `toStr()` was `[1 nil !1]`
    for what should be an absent, freshly-built knob. Passing `vertex_equal`
    as `equal_subtree`'s predicate fixes it.
    """
    tr = combo_tree()
    target = tr.set_head(builtin.logical_and)
    tr.append_child(target, 1.0)
    subtree = combo_tree(builtin.logical_true)

    knob = logical_subtree_knob(tr, target, subtree.begin())

    assert knob.in_exemplar() is False
    assert knob.toStr() == "[nil true !true]"
    assert not isinstance(knob.get_loc().data, float)

    # Same coincidence, exercised directly against the generic tree
    # primitive the constructor delegates to: bare equality must not treat
    # a contin `1.0` node as equal to a `logical_true` node.
    contin_only = combo_tree(1.0)
    logical_true_only = combo_tree(builtin.logical_true)
    assert not tr.equal_subtree(
        contin_only.begin(), logical_true_only.begin(), vertex_equal
    )


def test_constructor_rejects_dead_target_and_copy_source():
    tr = str_to_combo_tree("and($1 $9)")
    dead_target = tr.erase(tr.begin().first_child)
    # erase returns the next sibling; retain and erase that live handle too.
    tr.erase(dead_target)
    subtree = combo_tree(argument(2))
    with pytest.raises(ValueError, match="target is erased"):
        logical_subtree_knob(tr, dead_target, subtree.begin())

    source_tr, _subtree, source = _logical_knob("and($1 $9)", 2)
    source.clear_exemplar()
    destination = str_to_combo_tree("and($1 $9)")
    with pytest.raises(ValueError, match="erased or unset"):
        logical_subtree_knob(destination, destination.begin(), source)

    dead_subtree = combo_tree(argument(2))
    stale_subtree = dead_subtree.begin()
    dead_subtree.erase(stale_subtree)
    with pytest.raises(ValueError, match="cannot copy from an erased"):
        logical_subtree_knob(destination, destination.begin(), stale_subtree)
