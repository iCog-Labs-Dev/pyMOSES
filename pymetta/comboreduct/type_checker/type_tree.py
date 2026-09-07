"""C++: moses/comboreduct/type_checker/type_tree.{h,cc}

A NARROW SLICE of a 1468-line file (`type_tree.cc`) plus its 260-line
header, scoped in from S8 (PLANNED: S8 per docs/MODULE-COVERAGE.md) into
THIS cycle's Task 5 for exactly one reason: `meta_rules.py`'s `downwards`
TYPED branch (`meta_rules.cc:101`, `get_output_type_tree(*it) ==
type_tree(output)`) is on the critical path for `logical_reduction.py` --
10 of 12 `downwards()` calls in `logical_reduction.cc` pass
`id::boolean_type`, so composing even effort 2 ("medium") needs a real
typed `downwards`, not the `NotImplementedError` stub Task 2 left behind.

What is ported, and why it is enough:

* `type_tree` (type_tree_def.h:89, `typedef opencog::tree<type_node>
  type_tree`) -- but NOT as a general n-ary tree. Every `type_tree` this
  port ever constructs or compares is a SINGLE NODE: `downwards.__init__`
  only ever builds `type_tree(some_type_node)` (meta_rules.h:83/87's
  `input(t)`/`input(input_)`, both single `type_node` arguments), and
  `get_output_type_tree` below -- the only producer this slice needs --
  ALWAYS returns a single-node result too (see its own docstring: even
  compound C++ arrow types like `"->(arg_list(boolean) boolean)"` collapse
  to just their OUTPUT type_node, a leaf). No caller in this port's
  reachable scope ever needs multiple levels, `tree<type_node>`'s
  iterator machinery, or a genuinely empty `type_tree()`.

  # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree_def.h:89): a
  # single-`type_node` value class, not `opencog::tree<type_node>`. This
  # port's own `pymetta.util.tree.tree`/`tree_node` (a real generic n-ary
  # tree, already used for `combo_tree`) is DELIBERATELY not reused here
  # -- it would be strictly more general than anything this scope needs
  # and would resurrect the whole "port the general type-inference engine"
  # scope this task's brief explicitly rejects. `equal_type_tree`
  # (type_tree.h:120, "ty1 inherits ty2 and ty2 inherits ty1", i.e. real
  # multi-node type-tree structural comparison), `get_type_tree` (the
  # general per-vertex TYPE, as opposed to OUTPUT type), `get_input_type_
  # tree`, `get_arity`, and the whole `builtins_properties`/`actions_
  # properties` singleton machinery that PARSES arrow-type strings like
  # `"->(arg_list(boolean) boolean)"` (`descriptions.cc`) are NOT ported.

* `get_output_type_tree(const vertex& v)` (type_tree.cc:250) -- ported for
  the vertex kinds this port's four-way `vertex` union
  (`builtin | argument | contin_t | str`, vertex.py's own module
  docstring) can actually hold. See its own docstring for the per-branch
  detail and what happens for the one branch that IS reachable in
  principle (`str`/`enum_t`) but that even the real C++ function does not
  handle either.

* `type_tree` equality (`__eq__` below) -- used by `downwards.apply`'s
  `input == unknown_type_tree` and `get_output_type_tree(*it) ==
  type_tree(output)` (meta_rules.cc:89,101).

Not ported, and not needed by the reducer's reachable scope (Boolean combo
trees plus continuous predicates). Continuous and action representation are
supported follow-on domains, but their signatures do not require widening
this leaf wrapper into the inference engine: `get_type_tree`,
`get_input_type_tree`, `equal_type_tree`/`inherits`, `is_argument_type`/
`arg_to_idx` (type_tree_def.h), and everything under
`builtin_properties`/`action_properties` (descriptions.h/.cc) beyond the
one per-builtin OUTPUT-type fact `_BUILTIN_OUTPUT_TYPE` below captures as
data.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import builtin, is_argument, is_builtin, is_contin, vertex
from pymetta.comboreduct.type_checker.type_signature import type_node


class type_tree:
    """C++: moses/comboreduct/type_checker/type_tree_def.h:89 type_tree

    See this module's own docstring for why this is a single-`type_node`
    value class rather than a port of `opencog::tree<type_node>`.
    """

    __slots__ = ("node",)

    def __init__(self, node: type_node) -> None:
        self.node: type_node = node

    def __eq__(self, other: object) -> bool:
        """C++: opencog/util/tree.h operator==(tree<T>, tree<T>)

        Real `tree<T>::operator==` compares whole tree SHAPE plus every
        node's value; restricted here to the one shape this port's
        `type_tree` can ever hold (a single node), so value equality on
        `.node` already IS the general comparison for this narrowed type.
        `type_node` is a plain `IntEnum` (type_signature.py) -- CLAUDE.md's "type
        before value" trap does not apply between two `type_tree`
        instances that are already known, by Python's own type system, to
        both be `type_tree`.
        """
        if not isinstance(other, type_tree):
            return NotImplemented
        return self.node == other.node

    def __hash__(self) -> int:
        return hash(self.node)

    def __repr__(self) -> str:
        return f"type_tree({self.node!r})"


# ---------------------------------------------------------------------------
# get_output_type_tree(builtin) -- descriptions.cc's parsed arrow-type
# strings, collapsed to just the one fact this scope needs: which
# type_node a fully-applied builtin's RESULT has. See this module's own
# docstring for what is deliberately not reproduced (the string parser,
# arity, and per-argument types).
# ---------------------------------------------------------------------------

# DIVERGENCE(C++ moses/comboreduct/combo/descriptions.cc:47-96): hardcoded
# per-builtin OUTPUT type_node, not derived by parsing an arrow-type
# string ("->(arg_list(boolean) boolean)" etc, descriptions.cc:47-96)
# through `builtins_properties`'s singleton constructor
# (descriptions.cc:100-142). Each value below is the same fact real C++
# computes -- for an arrow type "->(...  T)", the OUTPUT is the string's
# own LAST top-level element T; for a bare, non-arrow type ("boolean",
# "contin"), the output IS that type -- worked out by hand from the same
# `bd[]` table (descriptions.cc:47-96) real C++ parses at static-init
# time, not independently guessed. Sufficient for `downwards`'s typed
# dispatch (the only real consumer, meta_rules.cc:101), which only ever
# compares a node's OUTPUT type against a single target type_node
# (overwhelmingly `id::boolean_type`, `logical_reduction.cc`'s own 10-of-
# 12 `downwards()` calls) -- never needs arity or per-argument types.
_BUILTIN_OUTPUT_TYPE: dict[builtin, type_node] = {
    builtin.null_vertex: type_node.unknown_type,
    builtin.logical_true: type_node.boolean_type,
    builtin.logical_false: type_node.boolean_type,
    builtin.logical_and: type_node.boolean_type,
    builtin.logical_or: type_node.boolean_type,
    builtin.logical_not: type_node.boolean_type,
    builtin.plus: type_node.contin_type,
    builtin.times: type_node.contin_type,
    builtin.div: type_node.contin_type,
    builtin.exp: type_node.contin_type,
    builtin.log: type_node.contin_type,
    builtin.sin: type_node.contin_type,
    builtin.greater_than_zero: type_node.boolean_type,
    builtin.impulse: type_node.contin_type,
    builtin.rand: type_node.contin_type,
    builtin.list: type_node.list_type,
    builtin.car: type_node.unknown_type,
    builtin.cdr: type_node.list_type,
    builtin.cons: type_node.list_type,
    builtin.foldr: type_node.unknown_type,
    builtin.foldl: type_node.unknown_type,
    builtin.cond: type_node.unknown_type,
    builtin.equ: type_node.boolean_type,
    builtin.lambda_: type_node.lambda_type,
    builtin.apply: type_node.unknown_type,
    builtin.contin_if: type_node.contin_type,
}


def get_output_type_tree(v: vertex) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:250 get_output_type_tree

    Ported branches -- exactly the three `vertex` alternatives this
    port's four-way union (vertex.py) can hold that C++'s own function
    handles: `is_builtin` (type_tree.cc:252-253, via `_BUILTIN_OUTPUT_
    TYPE` above in place of the real `builtins_properties` singleton --
    see this module's own docstring), `is_argument` (type_tree.cc:254-255
    -- an argument's OUTPUT type is ALWAYS `unknown_type`, regardless of
    its declared/inferred type: a real, measured fact about C++, not an
    approximation -- an argument's type is only pinned down by its
    ENCLOSING expression, never in isolation), and `is_contin`
    (type_tree.cc:256-257, always `contin_type`).

    Not ported, matching real C++ exactly rather than "fixing" a gap:
    `is_action`/`is_builtin_action`/`is_perception`/`is_definite_object`/
    `is_indefinite_object`/`is_wild_card`/`is_message`/`is_procedure_call`
    (type_tree.cc:258-273) -- none of this port's `vertex` union members
    can currently satisfy these. A narrow action representation follow-on may
    add its own action vocabulary; that does not make the classic object,
    message, procedure, perception, or legacy embodiment branches part of this
    leaf wrapper's present reducer scope.

    One case is a REAL C++ FACT worth stating plainly, not a port gap:
    `is_enum_type(v)` has NO branch in `get_output_type_tree` AT ALL in
    real C++ (compare `get_type_tree(const vertex&)`, type_tree.cc:213-214,
    which DOES have one) -- an `enum_t`-holding vertex falls through to
    the function's own `else { OC_ASSERT(false, ...); }` tail
    (type_tree.cc:274-279). This port's four-way `vertex` union includes
    `str` (which `enum_t` subclasses), and this function raises for that
    case below via the SAME final `else`, for the SAME reason as real
    C++'s `enum_t` case specifically.

    SOFTENED (Task 5 fix round -- an earlier version of this paragraph
    overclaimed parity for `str` in general): C++'s `get_output_type_tree`
    does NOT send every string-like vertex to that final `else`. It has
    an explicit `is_definite_object(v)` branch (type_tree.cc:264-265,
    `return get_type_tree(v);`) that routes a `definite_object` -- also
    represented as a string in real C++ -- to `definite_object_type`
    instead, never reaching the assert-failure tail at all. This port's
    `str` alternative conflates `enum_t` and `definite_object` into one
    Python type (disclosed in `vertex.py`'s own module docstring and
    `docs/CARRY-FORWARD.md` section 2.1) and so cannot tell, from a bare `str`
    alone, which of the two real C++ types it stands in for -- meaning
    this function's uniform "raise for any `str`" is only PROVABLY
    correct for the `enum_t` case; for a `str` standing in for what would
    have been a real `definite_object`, this port's behaviour (raise)
    genuinely diverges from real C++'s (return `definite_object_type`).
    That divergence is an accepted, already-disclosed consequence of the
    `str` conflation itself (no `definite_object` construct exists
    anywhere reachable in this port's scope), not a new gap -- but it is
    NOT "the same reason" parity the previous wording claimed.
    """
    if is_builtin(v):
        return type_tree(_BUILTIN_OUTPUT_TYPE[v])
    if is_argument(v):
        return type_tree(type_node.unknown_type)
    if is_contin(v):
        return type_tree(type_node.contin_type)
    raise AssertionError(f"get_output_type_tree: Unhandled case '{v}'")
