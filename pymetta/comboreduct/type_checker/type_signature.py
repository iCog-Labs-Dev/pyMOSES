"""Immutable function signatures for the supported MOSES domains.

Classic stores a signature as a ``type_tree`` rooted at ``lambda_type``.
The Python port deliberately keeps ``type_tree`` as the single-leaf wrapper
needed by reduction, so a separate immutable value represents the one compound
shape required by representation: ordered input leaves followed by one output
leaf.

Only the signature and arity helpers needed by Boolean representation and its
continuous/action follow-ons are ported here.  This is not a general type-tree
inference engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from pymetta.comboreduct.type_checker.type_tree import type_tree


class type_node(IntEnum):
    """C++: moses/comboreduct/type_checker/type_tree_def.h:32 type_node

    The full enum is retained as transport compatibility for existing table
    loading and stable classic ordinals.  Its presence does not enable the
    excluded enum, list, ANN, object, message, or procedure domains.  The
    representation entry point is responsible for admitting only Boolean,
    continuous, and action-result signatures when those builders are added.
    """

    lambda_type = 0
    application_type = 1
    union_type = 2
    arg_list_type = 3
    boolean_type = 4
    contin_type = 5
    enum_type = 6
    list_type = 7
    action_result_type = 8
    definite_object_type = 9
    action_definite_object_type = 10
    indefinite_object_type = 11
    message_type = 12
    action_symbol_type = 13
    wild_card_type = 14
    ann_type = 15
    unknown_type = 16
    ill_formed_type = 17
    argument_type = 18


def _type_node_of(leaf: type_node | type_tree) -> type_node:
    if isinstance(leaf, type_node):
        return leaf

    # Importing type_tree at module load time would create a cycle: that
    # wrapper itself uses this canonical type_node definition.
    from pymetta.comboreduct.type_checker.type_tree import type_tree

    if isinstance(leaf, type_tree):
        return type_node(leaf.node)
    raise TypeError(f"expected type_node or type_tree leaf, got {type(leaf).__name__}")


# DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree_def.h:89): classic
# represents signatures as mutable compound `opencog::tree<type_node>` values
# rooted at `lambda_type`; this port deliberately stores the only required
# compound shape as immutable `(itypes, otype)` data while retaining the narrow
# single-leaf `type_tree` wrapper used by reduction.
@dataclass(frozen=True, slots=True)
class TypeSignature:
    """C++: moses/comboreduct/type_checker/type_tree_def.h:89 type_tree

    An ordered sequence of input type leaves and one output type leaf. See the
    adjacent divergence marker for the deliberate immutable representation.
    """

    itypes: tuple[type_node, ...]
    otype: type_node

    def __init__(
        self,
        itypes: Sequence[type_node | type_tree],
        otype: type_node | type_tree,
    ) -> None:
        object.__setattr__(self, "itypes", tuple(_type_node_of(t) for t in itypes))
        object.__setattr__(self, "otype", _type_node_of(otype))


def get_signature_inputs(signature: TypeSignature) -> tuple[type_tree, ...]:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1282 get_signature_inputs

    C++ returns independent leaf ``type_tree`` values.  The tuple preserves
    input order while making the Python result immutable as well.
    """
    from pymetta.comboreduct.type_checker.type_tree import type_tree

    return tuple(type_tree(node) for node in signature.itypes)


def get_signature_output(signature: TypeSignature) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1272 get_signature_output"""
    from pymetta.comboreduct.type_checker.type_tree import type_tree

    return type_tree(signature.otype)


@overload
def gen_signature(
    inputs: Sequence[type_node | type_tree], output: type_node | type_tree
) -> TypeSignature: ...


@overload
def gen_signature(iotype: type_node | type_tree, arity: int) -> TypeSignature: ...


@overload
def gen_signature(
    itype: type_node | type_tree, otype: type_node | type_tree, arity: int
) -> TypeSignature: ...


def gen_signature(
    inputs_or_itype: Sequence[type_node | type_tree] | type_node | type_tree,
    output_or_arity: type_node | type_tree | int,
    arity: int | None = None,
) -> TypeSignature:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1299 gen_signature

    Supports the three classic call shapes used by the scoped port:
    ``(inputs, output)``, ``(input_output_type, arity)``, and
    ``(input_type, output_type, arity)``.
    """
    if arity is not None:
        # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:1325):
        # classic passes `arity + 1` into a child-count API and does not reject
        # negative arities explicitly; Python raises before allocation rather
        # than permitting signed-to-unsigned expansion or invalid signatures.
        if arity < 0:
            raise ValueError("signature arity must be non-negative")
        itype = _type_node_of(inputs_or_itype)  # type: ignore[arg-type]
        return TypeSignature((itype,) * arity, _type_node_of(output_or_arity))  # type: ignore[arg-type]

    if isinstance(output_or_arity, int) and not isinstance(output_or_arity, type_node):
        if output_or_arity < 0:
            raise ValueError("signature arity must be non-negative")
        iotype = _type_node_of(inputs_or_itype)  # type: ignore[arg-type]
        return TypeSignature((iotype,) * output_or_arity, iotype)

    if isinstance(inputs_or_itype, (type_node,)):
        raise TypeError("two-type gen_signature form requires an explicit arity")
    return TypeSignature(inputs_or_itype, output_or_arity)  # type: ignore[arg-type]


def type_tree_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:352 type_tree_arity"""
    return len(signature.itypes)


def contin_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:316 contin_arity"""
    return signature.itypes.count(type_node.contin_type)


def boolean_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:328 boolean_arity"""
    return signature.itypes.count(type_node.boolean_type)


def action_result_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:340 action_result_arity"""
    return signature.itypes.count(type_node.action_result_type)
