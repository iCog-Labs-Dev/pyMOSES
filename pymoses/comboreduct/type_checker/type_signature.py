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
    from pymoses.comboreduct.type_checker.type_tree import type_tree


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

    @classmethod
    def _missing_(cls, value: object) -> type_node | None:
        """C++: moses/comboreduct/type_checker/type_tree_def.h:32 type_node

        # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree_def.h:70):
        # classic's `type_node` is a plain C enum, so
        # `(type_node)((int)id::argument_type + k)` is a legal value for any
        # k >= 0 and needs no declaration -- that is the entire purpose of
        # `argument_type` being last. Python's `IntEnum` is closed by
        # default and `type_node(19)` would raise `ValueError`, so this hook
        # mints an unnamed pseudo-member for any ordinal past
        # `argument_type`, the same mechanism `enum.IntFlag` uses for
        # unnamed flag combinations. Pseudo-members are cached in
        # `_value2member_map_` (so identity and hashing stay stable) but are
        # deliberately NOT added to `_member_map_`, so `list(type_node)` and
        # `type_node.__members__` still show exactly classic's 19
        # enumerators.
        """
        if isinstance(value, int) and value > cls.argument_type:
            member = int.__new__(cls, value)
            member._name_ = f"argument_type_{value - int(cls.argument_type) + 1}"
            member._value_ = value
            return cls._value2member_map_.setdefault(value, member)
        return None


# C++: moses/comboreduct/type_checker/type_tree.cc:250 get_output_type_tree's
# final `else` and type_tree.cc:243's commented-out one both mean "this vertex
# kind is not handled". This port reaches them for exactly one reason -- its
# `vertex` union spells enum_t, definite_object, action_definite_object,
# indefinite_object, message, action_symbol and wild_card all as a bare `str`
# (vertex.py's own module docstring) and cannot tell them apart -- and every one
# of those domains is out of scope. Shared so the message is identical wherever
# it is raised.
EXCLUDED_STR_VERTEX_MESSAGE = (
    "{fn}: a `str` vertex stands in for enum_t, definite_object, "
    "action_definite_object, indefinite_object, message, action_symbol or "
    "wild_card, which this port's four-way `vertex` union cannot tell apart; "
    "all of those domains are out of scope, so no type is inferred for {v!r}"
)


def argument_type_node(abs_idx_from_zero: int) -> type_node:
    """C++: moses/comboreduct/type_checker/type_tree.cc:158 get_type_tree

    The `(type_node)((int)id::argument_type + a.abs_idx_from_zero())` cast
    that `get_type_tree(const argument&)` performs, and that
    `operator>>(istream&, type_node&)` performs again for a `$N` spelling
    (type_tree.cc:1443). `$1` is `argument_type` itself, `$2` is
    `argument_type + 1`, and so on without bound -- which is exactly what
    `type_node`'s own comment says the enumerator is for ("THIS MUST BE
    LAST IN THE ENUM. The rest of the integers enumerate variables",
    type_tree_def.h:70-78). `type_node._missing_` above is what makes the
    unnamed values past `argument_type` real `type_node`s here, so that
    `set_arg_type`/`get_arg_type` can still tell a type_node apart from a
    plain 1-based index the way C++'s overload resolution does.
    """
    return type_node(int(type_node.argument_type) + abs_idx_from_zero)


def is_argument_type(n: type_node) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree_def.cc:29 is_argument_type

    `((int)n) >= id::argument_type`. `argument_type` is deliberately LAST
    in the enum (type_tree_def.h:70-78) precisely so that every ordinal at
    or beyond it encodes an argument: `argument_type` is `$1`,
    `argument_type + 1` is `$2`, and so on without bound.
    """
    return int(n) >= int(type_node.argument_type)


def arg_to_idx(n: type_node) -> int:
    """C++: moses/comboreduct/type_checker/type_tree_def.cc:34 arg_to_idx

    The 1-based argument index a `type_node` encodes, "as defined in class
    argument in vertex.h" (type_tree_def.h:104). The `OC_ASSERT` on
    `is_argument_type` is real.
    """
    if not is_argument_type(n):
        raise AssertionError("Cannot find the idx of a non-argument type")
    return int(n) - int(type_node.argument_type) + 1


def _type_node_of(leaf: type_node | type_tree) -> type_node:
    if isinstance(leaf, type_node):
        return leaf

    # Importing type_tree at module load time would create a cycle: that
    # wrapper itself uses this canonical type_node definition.
    from pymoses.comboreduct.type_checker.type_tree import type_tree

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

    The `TypeSignature`-shaped entry point; the real one, over a
    `type_tree`, is in `type_tree.py` and this delegates to it. C++
    returns a `type_tree_seq` (a `std::vector`); the tuple preserves input
    order while making the Python result immutable as well.
    """
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return tuple(_tt.get_signature_inputs(signature_to_type_tree(signature)))


def get_signature_output(signature: TypeSignature) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1272 get_signature_output

    The `TypeSignature`-shaped entry point; the real one, over a
    `type_tree`, is in `type_tree.py` and this delegates to it.
    """
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.get_signature_output(signature_to_type_tree(signature))


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


def signature_to_type_tree(signature: TypeSignature) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1299 gen_signature

    No C++ analogue as a separate function -- classic never needs one,
    because a signature simply IS a `type_tree` there. Here it is the
    bridge from this module's flat `TypeSignature` to the real
    `lambda_type`-rooted tree the engine in `type_tree.py` operates on,
    and it is exactly `gen_signature(itypes, otype)`.
    """
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.gen_signature(list(signature.itypes), signature.otype)


def type_tree_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:352 type_tree_arity

    Delegates to the real algorithm in `type_tree.py`, over the arrow tree
    this signature stands for. It used to be `len(signature.itypes)` with
    a DIVERGENCE marker saying the negative `arg_list` sentinel was not
    reproduced; delegating removes the divergence rather than documenting
    it. The two agree on every signature this port constructs -- nothing
    builds a `TypeSignature` whose last input is `arg_list_type` -- so the
    change is behaviour-preserving today and correct if that stops being
    true.
    """
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.type_tree_arity(signature_to_type_tree(signature))


def contin_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:316 contin_arity"""
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.contin_arity(signature_to_type_tree(signature))


def boolean_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:328 boolean_arity"""
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.boolean_arity(signature_to_type_tree(signature))


def action_result_arity(signature: TypeSignature) -> int:
    """C++: moses/comboreduct/type_checker/type_tree.cc:340 action_result_arity"""
    from pymoses.comboreduct.type_checker import type_tree as _tt

    return _tt.action_result_arity(signature_to_type_tree(signature))


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
    """C++: moses/comboreduct/type_checker/type_tree.cc:1360 type_node

    The real symbol is the free `operator<<(ostream&, const type_node&)`
    defined on that line; the crumb cites the type it prints, since
    `tools/check_alignment.py` cannot match an operator name.

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
    """C++: opencog/util/tree.h:3396 tree

    The real symbol is the free `operator<<(std::ostream&, const tree<T>&)`
    declared on that line; the crumb cites the class it prints, since
    `tools/check_alignment.py` cannot match an operator name.

    Prints a signature the way classic's log does: `->(boolean boolean
    boolean)` for a two-input Boolean table.

    The real symbol is `operator<<(std::ostream&, const tree<T>&)`
    (tree.h:3396), which delegates to `subtree_to_string` (tree.h:3353):
    a childless node prints bare, and a node with children prints
    `node(child child ...)` with SINGLE spaces and no trailing space. A
    signature is a `lambda_type` root -- which `ostream_type_node` renders
    as `->` -- whose children are the input types followed by the output
    type, so a zero-input signature is `->(boolean)`, not `->()`.

    # DIVERGENCE(C++ opencog/util/tree.h:3396): this port has no compound
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
