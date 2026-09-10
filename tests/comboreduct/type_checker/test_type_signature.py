"""Parity tests for the S8 immutable signature prerequisite.

The oracle rows were captured from ``tools/cpp_probe/type_signature_probe.cc``
linked against the classic ``libcomboreduct`` build.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import (
    CTable,
    ITable,
    OTable,
    Signature,
    Table,
    TimedCounter,
)
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    action_result_arity,
    boolean_arity,
    contin_arity,
    gen_signature,
    get_signature_inputs,
    get_signature_output,
    ostream_type_node,
    type_node,
    type_tree_arity,
)
from pymetta.comboreduct.type_checker.type_tree import type_tree

_NAME_TO_NODE = {
    "boolean": type_node.boolean_type,
    "contin": type_node.contin_type,
    "action_result": type_node.action_result_type,
}
_EXPECTED_CASE_NAMES = (
    "boolean_zero",
    "boolean_uniform_three",
    "contin_to_boolean_two",
    "mixed_to_action",
    "action_uniform_two",
    "leaf_contin_to_action_two",
    "leaf_sequence_to_boolean",
)
_EXPECTED_CASE_FORMS = (
    "type_node_uniform",
    "type_node_uniform",
    "type_node_io_arity",
    "type_node_seq",
    "type_node_uniform",
    "type_tree_io_arity",
    "type_tree_seq",
)
_EXPECTED_CASE_KEYS = {
    "name",
    "form",
    "inputs",
    "output",
    "arity",
    "boolean_arity",
    "contin_arity",
    "action_result_arity",
}


def _make_oracle_signature(case: dict[str, object]) -> TypeSignature:
    inputs = [_NAME_TO_NODE[name] for name in case["inputs"]]
    output = _NAME_TO_NODE[case["output"]]
    form = case["form"]
    if form == "type_node_uniform":
        return gen_signature(output, case["arity"])
    if form == "type_node_io_arity":
        return gen_signature(inputs[0], output, case["arity"])
    if form == "type_node_seq":
        return gen_signature(inputs, output)
    if form == "type_tree_io_arity":
        return gen_signature(type_tree(inputs[0]), type_tree(output), case["arity"])
    if form == "type_tree_seq":
        return gen_signature(tuple(type_tree(node) for node in inputs), type_tree(output))
    raise AssertionError(f"unknown oracle signature form: {form!r}")


def test_cpp_signature_oracle(fixture_dir):
    cases = json.loads((fixture_dir / "type_signature.json").read_text())
    assert isinstance(cases, list)
    assert len(cases) == len(_EXPECTED_CASE_NAMES)
    assert tuple(case["name"] for case in cases) == _EXPECTED_CASE_NAMES
    assert tuple(case["form"] for case in cases) == _EXPECTED_CASE_FORMS
    assert all(set(case) == _EXPECTED_CASE_KEYS for case in cases)
    assert all(
        isinstance(case["name"], str)
        and isinstance(case["form"], str)
        and isinstance(case["inputs"], list)
        and all(isinstance(name, str) for name in case["inputs"])
        and isinstance(case["output"], str)
        and all(
            isinstance(case[key], int)
            for key in ("arity", "boolean_arity", "contin_arity", "action_result_arity")
        )
        for case in cases
    )

    for case in cases:
        signature = _make_oracle_signature(case)
        assert [leaf.node for leaf in get_signature_inputs(signature)] == [
            _NAME_TO_NODE[name] for name in case["inputs"]
        ]
        assert get_signature_output(signature).node == _NAME_TO_NODE[case["output"]]
        assert type_tree_arity(signature) == case["arity"]
        assert boolean_arity(signature) == case["boolean_arity"]
        assert contin_arity(signature) == case["contin_arity"]
        assert action_result_arity(signature) == case["action_result_arity"]


def test_gen_signature_uniform_overloads_round_trip_leaf_types():
    assert gen_signature(type_node.boolean_type, 3) == TypeSignature(
        (type_node.boolean_type,) * 3, type_node.boolean_type
    )
    assert gen_signature(type_node.contin_type, type_node.boolean_type, 2) == TypeSignature(
        (type_node.contin_type,) * 2, type_node.boolean_type
    )
    action = type_tree(type_node.action_result_type)
    assert gen_signature(action, action, 2) == TypeSignature(
        (type_node.action_result_type,) * 2, type_node.action_result_type
    )
    assert gen_signature(
        (type_tree(type_node.boolean_type), type_tree(type_node.contin_type)),
        type_tree(type_node.boolean_type),
    ) == TypeSignature((type_node.boolean_type, type_node.contin_type), type_node.boolean_type)


def test_signature_is_deeply_immutable():
    source = [type_node.boolean_type]
    signature = TypeSignature(source, type_node.contin_type)
    source.append(type_node.contin_type)

    assert signature.itypes == (type_node.boolean_type,)
    with pytest.raises(FrozenInstanceError):
        signature.otype = type_node.boolean_type


def test_returned_leaf_wrappers_are_independent_copies():
    signature = gen_signature(
        (type_node.boolean_type, type_node.contin_type), type_node.action_result_type
    )
    inputs = get_signature_inputs(signature)
    output = get_signature_output(signature)

    # `type_tree` is a real `tree[type_node]` now, so mutating a returned
    # leaf means writing its root node's data -- there is no `.node` setter
    # (the `.node` property is read-only, kept only for the two `main/`
    # call sites that read it).
    inputs[0].begin().data = type_node.contin_type
    output.begin().data = type_node.boolean_type

    assert signature == TypeSignature(
        (type_node.boolean_type, type_node.contin_type), type_node.action_result_type
    )
    assert tuple(leaf.node for leaf in get_signature_inputs(signature)) == (
        type_node.boolean_type,
        type_node.contin_type,
    )
    assert get_signature_output(signature).node == type_node.action_result_type


def test_table_signature_is_the_canonical_model():
    assert Signature is TypeSignature
    assert Signature([type_node.boolean_type], type_node.boolean_type) == gen_signature(
        type_node.boolean_type, 1
    )


def test_table_get_signature_returns_the_canonical_model():
    itable = ITable(
        types=(type_node.boolean_type, type_node.contin_type),
        labels=("flag", "value"),
    )
    table = Table(OTable([builtin.logical_true]), itable)

    signature = table.get_signature()
    assert type(signature) is TypeSignature
    assert signature == gen_signature(
        (type_node.boolean_type, type_node.contin_type), type_node.boolean_type
    )


def test_ctable_filtered_preserves_a_filtered_canonical_signature():
    signature = gen_signature(
        (type_node.boolean_type, type_node.contin_type), type_node.boolean_type
    )
    table = CTable("result", ("flag", "value"), signature)
    # A CTable cell is a `TimedCounter` (table.h:496), not a bare
    # `Counter`: since timestamps landed, `filtered` merges per
    # OBSERVATION rather than per vertex, so it reads the timed view.
    cell = TimedCounter()
    cell.add(builtin.logical_false, None, 1.0)  # None = the default not-a-date
    table[(builtin.logical_true, 2.5)] = cell

    filtered = table.filtered((1,))
    assert filtered.get_signature() == gen_signature(
        (type_node.contin_type,), type_node.boolean_type
    )
    assert list(filtered) == [(2.5,)]
    assert filtered[(2.5,)].get(builtin.logical_false) == 1.0


def test_type_tree_leaf_node_is_revalidated():
    invalid_leaf = type_tree(type_node.boolean_type)
    # -1, not 999. Since `type_node` gained the `_missing_` hook that mints
    # the unnamed argument ordinals past `argument_type` (type_signature.py),
    # 999 IS a valid `type_node` -- it is `$982`, exactly as classic's
    # `(type_node)999` would be. A NEGATIVE ordinal is still rejected, and
    # that is what this guard is about: `TypeSignature` revalidates the leaf
    # rather than storing whatever it was handed.
    invalid_leaf.begin().data = -1

    with pytest.raises(ValueError, match="-1 is not a valid type_node"):
        TypeSignature((invalid_leaf,), type_node.boolean_type)


def test_an_ordinal_past_argument_type_is_a_valid_type_node():
    """The other half of the guard above, and the reason it moved to -1.

    `type_node`'s own comment (type_tree_def.h:70-78) says `argument_type`
    must be last precisely so that every larger integer names a variable.
    `type_node(19)` is `$2`, `type_node(999)` is `$982`, and neither is a
    named member -- `list(type_node)` still shows exactly classic's 19
    enumerators.
    """
    assert ostream_type_node(type_node(19)) == "$2"
    assert ostream_type_node(type_node(999)) == "$982"
    assert len(list(type_node)) == 19


@pytest.mark.parametrize("arity", [-1, -2])
def test_uniform_signature_rejects_negative_arity(arity):
    with pytest.raises(ValueError, match="non-negative"):
        gen_signature(type_node.boolean_type, arity)
