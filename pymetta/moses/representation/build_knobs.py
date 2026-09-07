"""Deterministic Boolean representation builder.

This module ports only classic ``build_knobs``' pure-Boolean path. Continuous
predicates and action expressions remain separate allowed-domain slices; random
logical pairs and every excluded domain are intentionally absent here.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    vertex,
    vertex_equal,
)
from pymetta.comboreduct.reduct.clean_reduction import clean_reduce
from pymetta.comboreduct.reduct.general_rules import is_boolean
from pymetta.comboreduct.reduct.logical_rules import is_logical_operator, is_predicate
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    boolean_arity,
    type_node,
    type_tree_arity,
)
from pymetta.moses.moses.complexity import tree_complexity
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import logical_subtree_knob
from pymetta.util.tree import tree_node

ProbeSimplifier = Callable[[combo_tree], None]

_FUTURE_OUTPUT_TYPES = frozenset(
    {
        type_node.contin_type,
        type_node.action_result_type,
    }
)


# DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:53-123): this
# vertical slice admits only Boolean output with Boolean inputs. Mixed
# Boolean/continuous signatures require the unported predicate and continuous
# builder branches. Continuous and action-result outputs belong to later allowed
# slices; enum, ANN, object, message, and procedure domains remain excluded.
# Pair sampling at build_knobs.cc:439-518 is also omitted: this class requires
# perm_ratio=-1, the classic early-return mode at lines 435-437.
class build_knobs:
    """C++: moses/moses/representation/build_knobs.h:47 build_knobs

    Decorate a reduced, pure-Boolean exemplar with logical knobs. Construction
    is eager, as in classic: on return the exemplar is decorated and ``mapper``
    has been finalized in canonical discrete-spec order.
    """

    def __init__(
        self,
        exemplar: combo_tree,
        signature: TypeSignature,
        mapper: knob_mapper,
        simplify_knob_building: ProbeSimplifier,
        *,
        ignore_ops: Collection[vertex] = (),
        perm_ratio: float = -1.0,
    ) -> None:
        """C++: moses/moses/representation/build_knobs.cc:53 build_knobs::build_knobs"""
        self._validate_request(
            exemplar,
            signature,
            mapper,
            simplify_knob_building,
            perm_ratio,
        )
        self._exemplar = exemplar
        self._signature = signature
        self._mapper = mapper
        self._simplify_knob_building = simplify_knob_building
        self._ignore_ops = tuple(ignore_ops)
        self._arity = type_tree_arity(signature)
        self._skip_disc_probe = not (
            5 * boolean_arity(signature) > type_tree_arity(signature)
        )

        self.logical_canonize(self._exemplar.begin())
        self.build_logical(self._exemplar.begin(), self._exemplar.begin())
        self.logical_cleanup()
        self._mapper.finalize()

    # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:53-123): classic
    # relies on assertions and unchecked caller preconditions across construction.
    # Python validates the entire deterministic request before
    # mutating the exemplar: nonempty/reduced supported shape, in-range
    # arguments, a fresh mapper, an in-place simplifier callback, and exactly
    # perm_ratio=-1. Invalid requests therefore fail atomically rather than
    # partially decorating a tree or reaching undefined iterator behavior.
    @staticmethod
    def _validate_request(
        exemplar: combo_tree,
        signature: TypeSignature,
        mapper: knob_mapper,
        simplify_knob_building: ProbeSimplifier,
        perm_ratio: float,
    ) -> None:
        if not isinstance(signature, TypeSignature):
            raise TypeError("build_knobs requires a TypeSignature")
        if signature.otype is not type_node.boolean_type:
            if signature.otype in _FUTURE_OUTPUT_TYPES:
                raise NotImplementedError(
                    f"{signature.otype.name} output belongs to a future allowed domain slice"
                )
            raise NotImplementedError(
                f"{signature.otype.name} output is excluded from the scoped port"
            )
        if any(input_type is not type_node.boolean_type for input_type in signature.itypes):
            raise NotImplementedError(
                "non-Boolean inputs require their corresponding domain builder slice"
            )
        if exemplar.empty():
            raise ValueError("Boolean exemplar must not be empty")
        if not isinstance(mapper, knob_mapper):
            raise TypeError("build_knobs requires a knob_mapper")
        if mapper.disc or getattr(mapper, "_finalized", False):
            raise ValueError("build_knobs requires a fresh, empty knob_mapper")
        if not callable(simplify_knob_building):
            raise TypeError("simplify_knob_building must be callable")
        if (
            isinstance(perm_ratio, bool)
            or not isinstance(perm_ratio, int | float)
            or float(perm_ratio) != -1.0
        ):
            raise ValueError("deterministic Boolean building requires perm_ratio=-1")

        arity = type_tree_arity(signature)
        allowed_builtins = (
            builtin.logical_true,
            builtin.logical_false,
            builtin.logical_and,
            builtin.logical_or,
        )
        for node in exemplar:
            value = node.data
            if isinstance(value, argument):
                if not exemplar.is_childless(node):
                    raise ValueError("Boolean argument literals must be childless")
                if not 1 <= value.abs_idx() <= arity:
                    raise ValueError(
                        f"Boolean argument ${value.abs_idx()} is outside signature arity {arity}"
                    )
                continue
            if isinstance(value, builtin) and any(
                vertex_equal(value, allowed) for allowed in allowed_builtins
            ):
                if is_boolean(value) and not exemplar.is_childless(node):
                    raise ValueError("Boolean constants must be childless")
                continue
            raise ValueError(
                "Boolean exemplar contains an unsupported or non-reduced vertex: "
                f"{value!r}"
            )

    def permitted_op(self, value: vertex) -> bool:
        """C++: moses/moses/representation/build_knobs.cc:131 build_knobs::permitted_op"""
        return not any(vertex_equal(value, ignored) for ignored in self._ignore_ops)

    def logical_canonize(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:150 build_knobs::logical_canonize"""
        if vertex_equal(it.data, builtin.logical_and):
            self._exemplar.insert_above(it, builtin.logical_or)
        elif vertex_equal(it.data, builtin.logical_or):
            self._exemplar.insert_above(it, builtin.logical_and)
        elif is_boolean(it.data):
            it.data = builtin.logical_and
            self._exemplar.insert_above(it, builtin.logical_or)
        elif isinstance(it.data, argument):
            parent = self._exemplar.insert_above(it, builtin.logical_and)
            self._exemplar.insert_above(parent, builtin.logical_or)
        else:
            raise ValueError("unexpected vertex during Boolean logical canonization")

    def logical_cleanup(self) -> None:
        """C++: moses/moses/representation/build_knobs.cc:269 build_knobs::logical_cleanup"""
        root = self._exemplar.begin()
        if root is not None:
            for node in list(self._exemplar.iter_post_at(root)):
                if (
                    self._exemplar.is_valid(node)
                    and is_logical_operator(node.data)
                    and self._exemplar.is_childless(node)
                ):
                    self._exemplar.erase(node)
        if self._exemplar.empty():
            self._exemplar.set_head(builtin.logical_and)

    def disc_probe(self, subtree: tree_node, knob: logical_subtree_knob) -> bool:
        """C++: moses/moses/representation/build_knobs.cc:311 build_knobs::disc_probe"""
        if self._skip_disc_probe:
            return True

        to_disallow: list[int] = []
        for idx in range(1, knob.multiplicity()):
            knob.turn(idx)
            initial_complexity = tree_complexity(subtree, is_predicate)
            reduced = combo_tree(subtree)
            clean_reduce(reduced)
            self._simplify_knob_building(reduced)
            if initial_complexity > tree_complexity(reduced, is_predicate):
                to_disallow.append(idx)
        knob.turn(0)

        if len(to_disallow) < knob.multiplicity() - 1:
            for idx in to_disallow:
                knob.disallow(idx)
            return True

        knob.clear_exemplar()
        return False

    def sample_logical_perms(self, _it: tree_node) -> list[combo_tree]:
        """C++: moses/moses/representation/build_knobs.cc:411 build_knobs::sample_logical_perms"""
        permutations: list[combo_tree] = []
        for idx in range(1, self._arity + 1):
            arg = argument(idx)
            if self.permitted_op(arg):
                permutations.append(combo_tree(arg))
        return permutations

    def add_logical_knobs(
        self,
        subtree: tree_node,
        it: tree_node,
        add_if_in_exemplar: bool = True,
    ) -> None:
        """C++: moses/moses/representation/build_knobs.cc:530 build_knobs::add_logical_knobs"""
        if not is_logical_operator(it.data):
            return

        collected: list[logical_subtree_knob] = []
        for permutation in self.sample_logical_perms(it):
            knob = logical_subtree_knob(self._exemplar, it, permutation.begin())
            should_build = self.disc_probe(subtree, knob)
            if should_build and (add_if_in_exemplar or not knob.in_exemplar()):
                collected.append(knob)

        for knob in collected:
            self._mapper.add_disc_knob(knob)

    def build_logical(self, subtree: tree_node, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:602 build_knobs::build_logical"""
        if vertex_equal(it.data, builtin.logical_and):
            flip = builtin.logical_or
        elif vertex_equal(it.data, builtin.logical_or):
            flip = builtin.logical_and
        else:
            return

        self.add_logical_knobs(subtree, it)
        current = it.first_child
        while current is not None:
            next_sibling = current.next_sibling
            if isinstance(current.data, argument):
                wrapper = self._exemplar.insert_above(current, flip)
                self.add_logical_knobs(subtree, wrapper, False)
            elif vertex_equal(current.data, builtin.null_vertex):
                break
            else:
                self.build_logical(subtree, current)
            current = next_sibling

        flipped = self._exemplar.append_child(it, flip)
        self.add_logical_knobs(subtree, flipped)
