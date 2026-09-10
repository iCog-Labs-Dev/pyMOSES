"""C++: moses/comboreduct/combo/descriptions.h and descriptions.cc

The per-builtin signature table, and the singleton that parses it.

`descriptions.cc` holds two tables: `bd[]` (builtin_properties, cc:47-96)
and `ad[]` (action_properties, cc:150-165). Only the BUILTIN half is
ported here.

Why the builtin half is in scope even though `docs/MODULE-COVERAGE.md`
marks this file EXCLUDED: that row's stated reason is "Action/perception
I/O descriptions ... only for describing action/perception signatures,
moot once those types are excluded". That is accurate for `ad[]` and
factually wrong for `bd[]` -- `bd[]` describes the signature of every
`id::builtin`, including `and`/`or`/`not`/`plus`/`times`/`div`/`0<`/
`impulse`, and `get_type_tree(builtin)` (type_tree.cc:53),
`get_output_type_tree(builtin)` (:59), `get_input_type_tree(builtin, i)`
(:65) and `get_arity(builtin)` (:47) each read exactly one field of it.
Without `bd[]` there is no type-inference engine at all.

Why the action half is NOT ported: `actions_properties` is keyed by
`id::action` (action.h:37), and this port has no `action` type anywhere --
not in the `vertex` union (`builtin | argument | contin_t | str`,
vertex.py's own module docstring), not as an enum, and there is no
`is_action` predicate. `get_type_tree(action)` and its two siblings
therefore have no argument this port can construct. Porting `ad[]` would
add a table nothing can key. What classic's table actually contains is
RECORDED, measured rather than described, in the `actions` section of
`tests/fixtures/type_tree.json`, and
`tests/comboreduct/type_checker/test_type_tree.py` pins the prerequisite
gap so that adding `action` to the `vertex` union without also porting
those three functions turns red.

Note this file's TABLE is verbatim classic data, including rows for
domains this port excludes (`list`/`car`/`cdr`/`cons`/`foldr`/`foldl` for
list/fold, `equ` for enum, `lambda`/`apply` for the higher-order family).
Those rows are DATA, not inference paths: dropping them would desynchronise
the table from `id::builtin_count` -- classic's own constructor
`OC_ASSERT`s that there is exactly one entry per builtin
(descriptions.cc:105) -- and would make `get_type_tree(builtin.cond)`
silently wrong rather than absent. The excluded domains are gated where
they are actually reached, in `type_tree.py`'s per-vertex dispatch, not by
mutilating this table.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.type_checker.type_signature import type_node
from pymetta.comboreduct.type_checker.type_tree import (
    str_to_type_tree,
    type_tree,
    type_tree_arity,
)

__all__ = [
    "builtins_properties",
    "maximal_builtin_arity",
]

# C++: moses/comboreduct/combo/descriptions.h:40 maximal_builtin_arity
maximal_builtin_arity = 3


# C++: moses/comboreduct/combo/descriptions.cc:47 bd
#
# Verbatim, in classic's own order. `id::lambda` is spelled `lambda_` here
# because `lambda` is a Python keyword -- the same rename `vertex.py`
# already applies to that enumerator.
_BD: tuple[tuple[builtin, str], ...] = (
    (builtin.null_vertex, "unknown"),
    (builtin.logical_and, "->(arg_list(boolean) boolean)"),
    (builtin.logical_or, "->(arg_list(boolean) boolean)"),
    (builtin.logical_not, "->(boolean boolean)"),
    (builtin.logical_true, "boolean"),
    (builtin.logical_false, "boolean"),
    (builtin.plus, "->(arg_list(contin) contin)"),
    (builtin.times, "->(arg_list(contin) contin)"),
    (builtin.div, "->(contin contin contin)"),
    (builtin.exp, "->(contin contin)"),
    (builtin.log, "->(contin contin)"),
    (builtin.sin, "->(contin contin)"),
    (builtin.greater_than_zero, "->(contin boolean)"),
    (builtin.impulse, "->(boolean contin)"),
    (builtin.rand, "contin"),
    (builtin.list, "->(arg_list(unknown) list)"),
    (builtin.car, "->(list unknown)"),
    (builtin.cdr, "->(list list)"),
    (builtin.cons, "->(unknown list list)"),
    (builtin.foldr, "->(unknown unknown list unknown)"),
    (builtin.foldl, "->(unknown unknown list unknown)"),
    (builtin.lambda_, "->(arg_list(unknown) ->)"),
    (builtin.apply, "->(-> unknown unknown)"),
    # cond primitive, in the style of lisp/scheme: cond(p1 v1 .. pn vn y).
    # The 'value' is marked unknown, as it can be of any type.
    (builtin.cond, "->(arg_list(boolean unknown) unknown unknown)"),
    (builtin.contin_if, "->(boolean contin contin contin)"),
    # Tests for equality. Classic's own comment: someday this should also
    # allow booleans and contins, which needs work in the typechecker and
    # the union type.
    (builtin.equ, "->(enum enum boolean)"),
)


class builtins_properties:
    """C++: moses/comboreduct/combo/descriptions.h:44 builtins_properties

    A singleton storing, per builtin, the arity, the output `type_node`,
    the whole signature `type_tree`, and the (at most
    `maximal_builtin_arity`) per-input `type_tree`s -- all derived by
    PARSING the arrow-type strings of `_BD` above, exactly as
    `builtins_properties::builtins_properties` (descriptions.cc:100) does
    at static-initialisation time.

    Classic's three `OC_ASSERT`s on the table -- one entry per builtin, no
    duplicates, every entry a defined builtin (descriptions.cc:105-119) --
    are reproduced in `_build` below.
    """

    _instance: builtins_properties | None = None

    __slots__ = ("_arity", "_argument", "_builtin_type_tree", "_output_type")

    def __init__(self) -> None:
        """C++: moses/comboreduct/combo/descriptions.cc:100 builtins_properties"""
        self._arity: dict[builtin, int] = {}
        self._output_type: dict[builtin, type_node] = {}
        self._builtin_type_tree: dict[builtin, type_tree] = {}
        self._argument: dict[builtin, list[type_tree]] = {}
        self._build()

    def _build(self) -> None:
        """C++: moses/comboreduct/combo/descriptions.cc:100 builtins_properties

        The constructor body. Split out only so the class body reads like
        the C++ header's member list.
        """
        number_of_builtin_descriptions = len(_BD)
        if number_of_builtin_descriptions != int(builtin.builtin_count):
            raise AssertionError("there must be entries for all builtins.")
        seen: set[builtin] = set()
        for b, type_str in _BD:
            if b in seen:
                raise AssertionError("there must not be two entries for one builtin.")
            seen.add(b)
            if not (0 <= int(b) < int(builtin.builtin_count)):
                raise AssertionError("must be one of the defined builtins.")

            t = str_to_type_tree(type_str)
            self._builtin_type_tree[b] = t

            ty_it = t.begin()
            self._arity[b] = type_tree_arity(t)

            if ty_it.data == type_node.lambda_type:
                self._output_type[b] = ty_it.last_child.data
            else:
                self._output_type[b] = ty_it.data

            args: list[type_tree] = []
            last_child = ty_it.last_child
            sib = ty_it.first_child
            while sib is not None and sib is not last_child:
                # descriptions.cc:135: an arg_list input is stored FLATTENED
                # to its first child, so `cond`'s two-element
                # `arg_list(boolean unknown)` is stored as just `boolean`
                # and the `unknown` is dropped. Faithful; the fixture's
                # `builtins` section records it.
                args.append(
                    type_tree(sib.first_child)
                    if sib.data == type_node.arg_list_type
                    else type_tree(sib)
                )
                sib = sib.next_sibling
            self._argument[b] = args

    @classmethod
    def instance(cls) -> builtins_properties:
        """C++: moses/comboreduct/combo/descriptions.h:47 instance

        C++'s `static builtins_properties singleton;` inside the function
        is initialised on first call and never again; this is the same
        thing spelled with a class attribute.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def builtin_arity(self, b: builtin) -> int:
        """C++: moses/comboreduct/combo/descriptions.h:53 builtin_arity"""
        return self._arity[b]

    def builtin_argument(self, b: builtin, i: int) -> type_tree:
        """C++: moses/comboreduct/combo/descriptions.h:56 builtin_argument

        # DIVERGENCE(C++ moses/comboreduct/combo/descriptions.h:56):
        # classic indexes a fixed `type_tree argument[builtin_count][3]`
        # array, so an `i` past the number of inputs actually stored reads
        # a default-constructed (EMPTY) `type_tree` rather than running off
        # the end -- but an `i >= maximal_builtin_arity` IS out of bounds
        # and is undefined behaviour there. Here the stored inputs are a
        # list, so an index past the end returns the same empty `type_tree`
        # for the in-bounds case and raises `IndexError` for the
        # out-of-bounds one instead of reading adjacent memory. No caller
        # reaches either: `get_input_type_tree(builtin, i)`
        # (type_tree.cc:65) only calls this with `i == 0` or with
        # `i < builtin_arity(b) <= maximal_builtin_arity`.
        """
        args = self._argument[b]
        if i >= maximal_builtin_arity:
            raise IndexError(
                f"builtin_argument: index {i} is past maximal_builtin_arity "
                f"({maximal_builtin_arity}); classic reads out of bounds here"
            )
        if i >= len(args):
            return type_tree()
        return type_tree(args[i].begin())

    def type_tree_of_builtin(self, b: builtin) -> type_tree:
        """C++: moses/comboreduct/combo/descriptions.h:59 type_tree_of_builtin

        Returns a COPY: classic's accessor returns `type_tree` by value
        (descriptions.h:59), and `reduce_type_tree` mutates what
        `get_type_tree` hands back, so handing out the singleton's own
        tree would corrupt the table on the first inference.
        """
        return type_tree(self._builtin_type_tree[b].begin())

    def output_type_of_builtin(self, b: builtin) -> type_node:
        """C++: moses/comboreduct/combo/descriptions.h:62 output_type_of_builtin"""
        return self._output_type[b]
