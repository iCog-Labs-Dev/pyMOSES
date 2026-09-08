"""Immutable Boolean candidate construction from packed MOSES instances."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection

from pymetta.comboreduct.combo.vertex import combo_tree, vertex
from pymetta.comboreduct.reduct.clean_reduction import clean_reduce
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.representation.build_knobs import build_knobs
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.util.tree import tree_node

Simplifier = Callable[[combo_tree], None]


# DIVERGENCE(C++ moses/moses/representation/representation.h:45-205): this
# class contains only the immutable Boolean construction path. The mutable
# transform/get_candidate_lock API is unnecessary in Python and would weaken
# the thread-safe get_candidate design. Continuous and action-result maps are
# deferred to their allowed-domain slices. Term, enum, ANN, object, message,
# procedure, and EXEMPLAR_INST_IS_UNDEAD paths are outside project scope.
class representation(knob_mapper):
    """C++: moses/moses/representation/representation.h:45 representation"""

    def __init__(
        self,
        simplify_candidate: Simplifier,
        simplify_knob_building: Simplifier,
        exemplar: combo_tree,
        signature: TypeSignature,
        *,
        ignore_ops: Collection[vertex] = (),
        perm_ratio: float = 0.0,
    ) -> None:
        """C++: moses/moses/representation/representation.cc:99 representation::representation"""
        if not callable(simplify_candidate):
            raise TypeError("simplify_candidate must be callable")
        if not callable(simplify_knob_building):
            raise TypeError("simplify_knob_building must be callable")
        if not isinstance(exemplar, combo_tree):
            raise TypeError("representation requires a combo_tree exemplar")
        if exemplar.empty():
            raise ValueError("representation exemplar must not be empty")

        super().__init__()
        self._exemplar = combo_tree(exemplar.begin())
        self._simplify_candidate = simplify_candidate
        self._simplify_knob_building = simplify_knob_building

        # Task 3 already runs clean_reduce before invoking this callback.
        build_knobs(
            self._exemplar,
            signature,
            self,
            self._simplify_knob_building,
            ignore_ops=ignore_ops,
            perm_ratio=perm_ratio,
        )

        self._fields = field_set(entry.spec for entry in self.disc)
        start = self._fields.begin_disc_raw_idx()
        self._it_disc_idx = {
            entry.knob.get_loc(): start + offset
            for offset, entry in enumerate(self.disc)
        }

    def fields(self) -> field_set:
        """C++: moses/moses/representation/representation.h:118 representation::fields"""
        return self._fields

    def exemplar(self) -> combo_tree:
        """C++: moses/moses/representation/representation.h:126 representation::exemplar

        C++ returns a const reference. Python has no const-reference view, so
        return a value copy to keep the identity-indexed internal tree immutable.
        """
        return combo_tree(self._exemplar.begin())

    def get_clean_exemplar(
        self, reduce: bool, knob_building: bool = False
    ) -> combo_tree:
        """C++: moses/moses/representation/representation.cc:251 representation::get_clean_exemplar

        Breadcrumb kept on one line for `tools/check_alignment.py`'s parser.
        """
        result = self.exemplar()
        self.clean_combo_tree(result, reduce, knob_building)
        return result

    def clean_combo_tree(
        self, tr: combo_tree, reduce: bool, knob_building: bool = False
    ) -> None:
        """C++: moses/moses/representation/representation.cc:259 representation::clean_combo_tree"""
        if not isinstance(tr, combo_tree):
            raise TypeError("clean_combo_tree requires a combo_tree")
        clean_reduce(tr)
        if reduce:
            reducer = (
                self._simplify_knob_building
                if knob_building
                else self._simplify_candidate
            )
            reducer(tr)

    def get_candidate(self, inst: instance, reduce: bool) -> combo_tree:
        """C++: moses/moses/representation/representation.cc:316 representation::get_candidate

        Rebuild from the immutable decorated exemplar without turning a knob.
        Every call owns its nodes and mutable vertex payloads, so concurrent
        calls cannot exchange tree mutation through this representation.
        """
        # DIVERGENCE(C++ moses/moses/representation/representation.cc:316):
        # classic performs no width check at all here -- not even an
        # OC_ASSERT -- and would read/pack past a too-short `inst` via
        # `field_set::get_raw`'s raw byte-offset arithmetic. Python rejects a
        # wrong-width instance up front instead of reading out of bounds.
        if not isinstance(inst, instance):
            raise TypeError("get_candidate requires an instance")
        expected = self._fields.packed_width()
        if len(inst) != expected:
            raise ValueError(
                f"instance has {len(inst)} words; field set requires {expected}"
            )

        candidate = combo_tree()
        root = self._exemplar.begin()
        if root is not None:
            self._get_candidate_rec(inst, root, None, candidate)
        self.clean_combo_tree(candidate, reduce)
        return candidate

    def _get_candidate_rec(
        self,
        inst: instance,
        src: tree_node,
        parent_dst: tree_node | None,
        candidate: combo_tree,
    ) -> None:
        """C++: moses/moses/representation/representation.cc:331 representation::get_candidate_rec

        Breadcrumb kept on one line for `tools/check_alignment.py`'s parser.
        """
        knob = self.find_disc_knob(src)
        if knob is not None:
            try:
                raw_idx = self._it_disc_idx[src]
            except KeyError as error:
                raise RuntimeError("candidate knob has no raw field index") from error
            raw = self._fields.get_raw(inst, raw_idx)
            new_src, new_parent = knob.append_to(candidate, parent_dst, raw)
            if new_src is not None:
                for child in self._exemplar.children(new_src):
                    self._get_candidate_rec(inst, child, new_parent, candidate)
            return

        if candidate.empty():
            new_parent = candidate.set_head(src.data)
        else:
            if parent_dst is None:
                raise RuntimeError("nonempty candidate has no destination parent")
            # `tree.set_head` value-copies its data (tree.py:1028) but
            # `append_child`'s plain-data path does not (tree.py:1052), so the
            # caller must. Without this copy the candidate shares the
            # exemplar's `argument` object, and reducing a negated composite
            # knob rewrites `$1` to `!$1` IN THE EXEMPLAR via
            # `argument.negate()` -- measured: get_candidate(inst, reduce=True)
            # turned `or(and($1 $2) $3)` into `or(and(!$1 !$2) $3)`. Classic
            # cannot do this; `append_child(pos, const T&)` copies by value,
            # and representation.cc:312 states get_candidate "does not modify
            # the rep at all". `knobs.py`'s `_append_vertex` already copies
            # for the same reason.
            new_parent = candidate.append_child(parent_dst, copy.copy(src.data))
        for child in self._exemplar.children(src):
            self._get_candidate_rec(inst, child, new_parent, candidate)
