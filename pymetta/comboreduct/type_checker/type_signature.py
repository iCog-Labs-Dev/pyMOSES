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
        # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:1330):
        # classic passes `arity + 1` (arity_t, a plain signed `int`,
        # common_def.h:31) straight into `append_children(iter, int n)`
        # (opencog/util/tree.h:1072) -- there is no signed-to-unsigned
        # conversion anywhere in this call. CORRECTED (S10 branch review,
        # Minor): this note used to claim a negative arity "silently
        # produces a degenerate lambda_type node with no children" --
        # MEASURED FALSE, traced through cogutil's tree.h by hand.
        # `append_children`'s loop (`while ((n--) > 0)`) does append zero
        # children for a non-positive `n`, and the FOR loop right after it
        # (`type_tree.cc:1332-1333`) correctly does nothing when `root` is
        # childless (`begin(root) == end(root)` for a childless node,
        # tree.h:915-916, so the loop's own `sib != root.last_child()`
        # condition is already false) -- but `type_tree.cc:1334`,
        # `res.replace(sib, otype.begin())`, runs UNCONDITIONALLY right
        # after that loop, using `sib` still holding `root.begin()`'s
        # NULL-wrapping iterator (`tree.h:912-919`: `begin()` on a
        # childless node returns `end()`, `.node == 0`). `replace`'s own
        # `iterator_base` overload (`tree.h:1396-1424`) has no null guard
        # at all -- its own `tree_assert(position.node != head)` checks
        # against the tree's `head` SENTINEL, not against a null node, so
        # it passes -- and then dereferences `position.node` directly
        # (`erase_children(position)`, then `current_to->prev_sibling`
        # with `current_to = position.node`). Classic SEGFAULTS here, it
        # does not degrade gracefully to a smaller signature. Raising
        # before that point is still the right call -- there is no
        # graceful C++ behaviour to reproduce, only a crash to avoid.
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
    # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:366):
    # classic returns a NEGATIVE arity (`-a`) when the signature's last input
    # is `id::arg_list_type` -- a deliberate sentinel meaning "variadic, |a|
    # named arguments plus a trailing arg_list" (see also convert_index's use
    # of that sign, type_tree.cc:379-382). `arg_list_type` has no
    # representation in this scoped port (term-algebra/enum/etc. are
    # excluded outside the Boolean+continuous domain), so `len(...)` here is
    # always the plain non-negative count and never reproduces that
    # negative-arity variadic encoding.
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


# ---------------------------------------------------------------------------
# Printing -- needed because several MOSES log records stream a type_tree
# ---------------------------------------------------------------------------

# C++: type_tree.cc:1360-1410 `operator<<(ostream&, const type_node&)`. A
# switch, not a table, and the spellings are NOT the enumerator names: the
# enum is `boolean_type` but it prints as `boolean`, and `enum_type` prints
# as `enum_t`. Reproduced verbatim, including that asymmetry.
_TYPE_NODE_STRINGS = {
    type_node.lambda_type: "->",
    type_node.application_type: "application",
    type_node.union_type: "union",
    type_node.arg_list_type: "arg_list",
    type_node.boolean_type: "boolean",
    type_node.contin_type: "contin",
    type_node.enum_type: "enum_t",
    type_node.action_result_type: "action_result",
    type_node.definite_object_type: "definite_object",
    type_node.action_definite_object_type: "action_definite_object",
    type_node.indefinite_object_type: "indefinite_object",
    type_node.message_type: "message",
    type_node.action_symbol_type: "action_symbol",
    type_node.wild_card_type: "wild_card",
    type_node.unknown_type: "unknown",
    type_node.ill_formed_type: "ill_formed",
}


def ostream_type_node(n: type_node) -> str:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1360 operator<<

    The `type_node` half of printing a signature. Two cases the table above
    cannot hold: an `argument_type`-or-higher ordinal prints as `$1`, `$2`,
    ... (`out << "$" << n - (int)id::argument_type + 1`), and anything else
    prints `UNKNOWN_HANDLE`. `list_type` and `ann_type` fall in that second
    group -- classic's switch has no `case` for either, which is why they are
    absent from the table rather than merely unported here.
    """
    if n in _TYPE_NODE_STRINGS:
        return _TYPE_NODE_STRINGS[n]
    if int(n) >= int(type_node.argument_type):
        return f"${int(n) - int(type_node.argument_type) + 1}"
    return "UNKNOWN_HANDLE"


def ostream_type_signature(signature: TypeSignature) -> str:
    """C++: opencog/util/tree.h:3396 operator<<

    Prints a signature the way classic's log does: `->(boolean boolean
    boolean)` for a two-input Boolean table.

    The real symbol is `operator<<(std::ostream&, const tree<T>&)`
    (tree.h:3396), which delegates to `subtree_to_string` (tree.h:3353):
    a childless node prints bare, and a node with children prints
    `node(child child ...)` with SINGLE spaces and no trailing space. A
    signature is a `lambda_type` root -- which `ostream_type_node` renders
    as `->` -- whose children are the input types followed by the output
    type, so a zero-input signature is `->(boolean)`, not `->()`.

    # DIVERGENCE(opencog/util/tree.h:3396): this port has no compound
    # `tree<type_node>`; `TypeSignature` stores `(itypes, otype)` flat (see
    # this module's own DIVERGENCE marker above `TypeSignature`). The arrow
    # form is therefore reconstructed here rather than produced by a generic
    # tree printer, which also means a nested signature -- an input that is
    # itself an arrow type, e.g. `->(->(boolean boolean) boolean)` -- cannot
    # be represented, let alone printed. Nothing this port loads produces
    # one: `gen_signature` builds leaf inputs only.
    """
    children = [ostream_type_node(t) for t in signature.itypes]
    children.append(ostream_type_node(signature.otype))
    return ostream_type_node(type_node.lambda_type) + "(" + " ".join(children) + ")"
