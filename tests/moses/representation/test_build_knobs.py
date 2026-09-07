"""Oracle and boundary tests for deterministic pure-Boolean knob building."""

import copy
import json

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import (
    combo_tree_to_str,
    str_to_combo_tree,
)
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, vertex_equal
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    gen_signature,
    type_node,
)
from pymetta.moses.representation.build_knobs import build_knobs
from pymetta.moses.representation.knob_mapper import knob_mapper

_CASE_KEYS = {
    "name",
    "source",
    "arity",
    "ignored_arguments",
    "decorated_exemplar",
    "prototype",
    "probe_calls",
    "knobs",
}
_KNOB_KEYS = {
    "order",
    "multy",
    "text",
    "in_exemplar",
    "location_preorder",
    "location_subtree",
}
_EXPECTED_CASES = (
    ("constant-true", "true", 0, [], 0, 0),
    ("bare-argument", "$1", 1, [], 10, 1),
    ("small-and", "and($1 $2)", 2, [], 24, 2),
    ("small-or-negated", "or($1 !$2)", 2, [], 24, 2),
    ("nested-three", "and(or($1 $2) !$3)", 3, [], 54, 7),
    ("ignored-argument", "and($1 $2)", 2, [2], 12, 1),
)

_FUTURE_OUTPUT_TYPES = {
    type_node.contin_type,
    type_node.action_result_type,
}


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "boolean_builder.json").read_text())


def _render(tr: combo_tree) -> str:
    return combo_tree_to_str(tr).strip()


def _render_vertex(value: object) -> str:
    return _render(combo_tree(copy.copy(value)))


def _prototype(exemplar: combo_tree, mapper: knob_mapper) -> str:
    def render_node(node) -> str:
        knob = mapper.find_disc_knob(node)
        text = knob.toStr() if knob is not None else _render_vertex(node.data)
        effective = node
        if vertex_equal(node.data, builtin.null_vertex):
            assert exemplar.has_one_child(node)
            effective = node.first_child
        children = list(exemplar.children(effective))
        if children:
            text += "(" + " ".join(render_node(child) for child in children) + ")"
        return text

    return render_node(exemplar.begin())


def _python_case(case: dict[str, object]) -> dict[str, object]:
    exemplar = str_to_combo_tree(case["source"])
    signature = gen_signature(type_node.boolean_type, case["arity"])
    ignored = {argument(idx) for idx in case["ignored_arguments"]}
    reduction = logical_reduction(ignored)(2)
    probe_calls: list[dict[str, str]] = []

    def simplify_knob_building(tr: combo_tree) -> None:
        before = _render(tr)
        reduction(tr)
        probe_calls.append({"before": before, "after": _render(tr)})

    mapper = knob_mapper()
    build_knobs(
        exemplar,
        signature,
        mapper,
        simplify_knob_building,
        ignore_ops=ignored,
        perm_ratio=-1,
    )
    preorder = {node: idx for idx, node in enumerate(exemplar)}
    knobs = [
        {
            "order": order,
            "multy": entry.spec.multy,
            "text": entry.knob.toStr(),
            "in_exemplar": entry.knob.in_exemplar(),
            "location_preorder": preorder[entry.knob.get_loc()],
            "location_subtree": _render(combo_tree(entry.knob.get_loc())),
        }
        for order, entry in enumerate(mapper.disc)
    ]
    return {
        "name": case["name"],
        "source": case["source"],
        "arity": case["arity"],
        "ignored_arguments": case["ignored_arguments"],
        "decorated_exemplar": _render(exemplar),
        "prototype": _prototype(exemplar, mapper),
        "probe_calls": probe_calls,
        "knobs": knobs,
    }


@pytest.mark.oracle
def test_boolean_builder_fixture_schema_and_inputs_are_non_vacuous(golden):
    assert set(golden) == {"perm_ratio", "cases"}
    assert golden["perm_ratio"] == -1
    assert isinstance(golden["cases"], list)
    assert [
        (
            case["name"],
            case["source"],
            case["arity"],
            case["ignored_arguments"],
            len(case["probe_calls"]),
            len(case["knobs"]),
        )
        for case in golden["cases"]
    ] == list(_EXPECTED_CASES)
    assert all(set(case) == _CASE_KEYS for case in golden["cases"])
    assert all(
        set(knob) == _KNOB_KEYS
        for case in golden["cases"]
        for knob in case["knobs"]
    )
    assert sum(len(case["probe_calls"]) for case in golden["cases"]) == 124
    assert sum(len(case["knobs"]) for case in golden["cases"]) == 13
    assert any(
        knob["multy"] == 2 for case in golden["cases"] for knob in case["knobs"]
    )


@pytest.mark.oracle
def test_deterministic_boolean_builder_matches_classic(golden):
    assert [_python_case(case) for case in golden["cases"]] == golden["cases"]


@pytest.mark.parametrize(
    "output_type", [node for node in type_node if node is not type_node.boolean_type]
)
def test_builder_rejects_every_non_boolean_output_before_mutation(output_type):
    exemplar = str_to_combo_tree("$1")
    before = _render(exemplar)
    signature = gen_signature(type_node.boolean_type, output_type, 1)
    message = "future allowed domain slice" if output_type in _FUTURE_OUTPUT_TYPES else "excluded"

    with pytest.raises(NotImplementedError, match=message):
        build_knobs(exemplar, signature, knob_mapper(), lambda _tr: None)

    assert _render(exemplar) == before


@pytest.mark.parametrize(
    "input_type", [node for node in type_node if node is not type_node.boolean_type]
)
def test_builder_rejects_every_non_boolean_input_before_mutation(input_type):
    exemplar = str_to_combo_tree("$1")
    before = _render(exemplar)
    signature = TypeSignature((input_type,), type_node.boolean_type)

    with pytest.raises(NotImplementedError, match="non-Boolean inputs"):
        build_knobs(exemplar, signature, knob_mapper(), lambda _tr: None)

    assert _render(exemplar) == before


@pytest.mark.parametrize(
    ("source", "arity", "message"),
    [
        ("not($1)", 1, "unsupported or non-reduced"),
        ("null_vertex($1)", 1, "unsupported or non-reduced"),
        ("0", 0, "unsupported or non-reduced"),
        # 3.0 equals IntEnum builtin.logical_and under bare Python equality.
        ("3", 0, "unsupported or non-reduced"),
        ("plus($1)", 1, "unsupported or non-reduced"),
        ("$2", 1, "outside signature arity"),
    ],
)
def test_builder_rejects_unsupported_or_mistyped_vertices_without_mutation(
    source, arity, message
):
    exemplar = str_to_combo_tree(source)
    before = _render(exemplar)

    with pytest.raises(ValueError, match=message):
        build_knobs(
            exemplar,
            gen_signature(type_node.boolean_type, arity),
            knob_mapper(),
            lambda _tr: None,
        )

    assert _render(exemplar) == before


def test_builder_rejects_empty_or_structurally_invalid_exemplars():
    signature = gen_signature(type_node.boolean_type, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        build_knobs(combo_tree(), signature, knob_mapper(), lambda _tr: None)

    argument_parent = combo_tree(argument(1))
    argument_parent.append_child(argument_parent.begin(), argument(1))
    with pytest.raises(ValueError, match="must be childless"):
        build_knobs(argument_parent, signature, knob_mapper(), lambda _tr: None)

    constant_parent = combo_tree(builtin.logical_true)
    constant_parent.append_child(constant_parent.begin(), argument(1))
    with pytest.raises(ValueError, match="must be childless"):
        build_knobs(constant_parent, signature, knob_mapper(), lambda _tr: None)


@pytest.mark.parametrize("perm_ratio", [0, 1, -0.5, True, "-1"])
def test_builder_rejects_non_deterministic_or_mistyped_perm_ratio(perm_ratio):
    exemplar = str_to_combo_tree("$1")
    with pytest.raises(ValueError, match="perm_ratio=-1"):
        build_knobs(
            exemplar,
            gen_signature(type_node.boolean_type, 1),
            knob_mapper(),
            lambda _tr: None,
            perm_ratio=perm_ratio,
        )


def test_builder_rejects_bad_collaborators_before_mutation():
    exemplar = str_to_combo_tree("$1")
    signature = gen_signature(type_node.boolean_type, 1)
    before = _render(exemplar)

    with pytest.raises(TypeError, match="TypeSignature"):
        build_knobs(exemplar, object(), knob_mapper(), lambda _tr: None)
    with pytest.raises(TypeError, match="knob_mapper"):
        build_knobs(exemplar, signature, object(), lambda _tr: None)
    with pytest.raises(TypeError, match="must be callable"):
        build_knobs(exemplar, signature, knob_mapper(), None)

    finalized = knob_mapper()
    finalized.finalize()
    with pytest.raises(ValueError, match="fresh, empty"):
        build_knobs(exemplar, signature, finalized, lambda _tr: None)
    assert _render(exemplar) == before
