"""Discrete and continuous knob primitives for MOSES representation trees.

``contin_knob`` (S14 Task 2) joins ``logical_subtree_knob`` here. Classic's
two action knob classes are still intentionally absent until that domain
slice is implemented; enum, ANN, object, and procedure knobs remain excluded.
"""

# DIVERGENCE(C++ moses/moses/representation/knobs.h:128-333): this slice still
# defers the two action knob classes (action_subtree_knob,
# simple_action_subtree_knob). No knob for an excluded enum, ANN, object, or
# procedure domain is introduced in their place.

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    get_contin,
    is_contin,
    vertex_equal,
)
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.moses.moses.complexity import complexity_t, tree_complexity
from pymetta.moses.representation.field_set import contin_spec, disc_spec
from pymetta.moses.representation.instance import contin_t
from pymetta.util.tree import tree_node


def _vertex_to_string(value: object) -> str:
    one = combo_tree(copy.copy(value))
    return combo_tree_to_str(one).rstrip()


class knob_base(ABC):
    """C++: moses/moses/representation/knobs.h:58 knob_base"""

    def __init__(self, tr: combo_tree, loc: tree_node | None = None) -> None:
        self._tr = tr
        self._loc = loc

    def _require_live_location(self, operation: str) -> tree_node:
        # DIVERGENCE(C++ moses/moses/representation/knobs.h:72-75): classic
        # returns dangling iterators after a knob location is erased; Python
        # rejects them explicitly instead of risking silent tree corruption.
        if self._loc is None or not self._tr.is_valid(self._loc):
            raise ValueError(f"{operation}: knob location is erased or unset")
        return self._loc

    def get_loc(self) -> tree_node:
        """C++: moses/moses/representation/knobs.h:72 knob_base::get_loc"""
        return self._require_live_location("get_loc")

    @abstractmethod
    def in_exemplar(self) -> bool:
        """C++: moses/moses/representation/knobs.h:66 knob_base::in_exemplar"""
        pass

    @abstractmethod
    def clear_exemplar(self) -> None:
        """C++: moses/moses/representation/knobs.h:70 knob_base::clear_exemplar"""
        pass

    @abstractmethod
    def toStr(self) -> str:
        """C++: moses/moses/representation/knobs.h:77 knob_base::toStr"""
        pass

    def __str__(self) -> str:
        return self.toStr()


class disc_knob_base(knob_base, ABC):
    """C++: moses/moses/representation/knobs.h:84 disc_knob_base"""

    @abstractmethod
    def turn(self, idx: int) -> None:
        """C++: moses/moses/representation/knobs.h:92 disc_knob_base::turn"""
        pass

    @abstractmethod
    def disallow(self, setting: int) -> None:
        """C++: moses/moses/representation/knobs.h:93 disc_knob_base::disallow"""
        pass

    @abstractmethod
    def allow(self, setting: int) -> None:
        """C++: moses/moses/representation/knobs.h:94 disc_knob_base::allow"""
        pass

    @abstractmethod
    def append_to(
        self, candidate: combo_tree, parent_dst: tree_node | None, idx: int
    ) -> tuple[tree_node | None, tree_node | None]:
        """C++: moses/moses/representation/knobs.h:112 disc_knob_base::append_to"""
        pass

    @abstractmethod
    def spec(self) -> disc_spec:
        """C++: moses/moses/representation/knobs.h:117 disc_knob_base::spec"""
        pass

    @abstractmethod
    def multiplicity(self) -> int:
        """C++: moses/moses/representation/knobs.h:120 disc_knob_base::multiplicity"""
        pass

    @abstractmethod
    def complexity_bound(self) -> complexity_t:
        """C++: moses/moses/representation/knobs.h:125 disc_knob_base::complexity_bound"""
        pass


class contin_knob(knob_base):
    """C++: moses/moses/representation/knobs.h:128 contin_knob

    Unlike ``discrete_knob``/``logical_subtree_knob``, a contin knob's spec
    is captured once, at construction, straight off the target vertex's
    current value -- there is no probe/disallow step the way discrete knobs
    have (``build_knobs.disc_probe``): ``build_contin`` (build_knobs.cc:758)
    inserts every ``contin_knob`` it builds into the mapper unconditionally.
    """

    def __init__(
        self,
        tr: combo_tree,
        target: tree_node,
        step_size: contin_t,
        expansion: contin_t,
        depth: int,
    ) -> None:
        """C++: moses/moses/representation/knobs.cc:36 contin_knob::contin_knob"""
        super().__init__(tr, target)
        # DIVERGENCE(C++ moses/moses/representation/knobs.cc:36-40): classic
        # relies on the caller (``build_contin``, which already checked
        # ``is_contin(*it)``) to pass a live contin-valued target; a bad
        # target is undefined `combo::get_contin` behaviour there. Python
        # validates explicitly instead.
        if not tr.is_valid(target):
            raise ValueError("contin_knob: target is erased or invalid")
        if not is_contin(target.data):
            raise TypeError("contin_knob: target vertex must hold a contin_t")
        self._spec = contin_spec(get_contin(target.data), step_size, expansion, depth)

    def in_exemplar(self) -> bool:
        """C++: moses/moses/representation/knobs.cc:42 contin_knob::in_exemplar

        Always true: unlike a discrete knob, a contin knob has no "absent"
        setting to compare against -- the exemplar's own value at
        construction time IS the knob's default/center (``_spec.mean``).
        """
        return True

    def clear_exemplar(self) -> None:
        """C++: moses/moses/representation/knobs.cc:44 contin_knob::clear_exemplar

        A literal no-op in C++ too (`{}`) -- classic's own `@todo: it does
        not go back to the initial state` comment (knobs.h:143) is kept
        below rather than silently fixed.
        """
        # @todo: it does not go back to the initial state (knobs.h:143).
        return None

    def turn(self, x: contin_t) -> None:
        """C++: moses/moses/representation/knobs.cc:46 contin_knob::turn"""
        loc = self._require_live_location("turn")
        loc.data = contin_t(x)

    def append_to(
        self, candidate: combo_tree, parent_dst: tree_node | None, c: contin_t
    ) -> tree_node:
        """C++: moses/moses/representation/knobs.cc:48 contin_knob::append_to"""
        if candidate.empty():
            return candidate.set_head(contin_t(c))
        if parent_dst is None:
            raise ValueError("append_to: nonempty candidate requires a destination parent")
        return candidate.append_child(parent_dst, contin_t(c))

    def spec(self) -> contin_spec:
        """C++: moses/moses/representation/knobs.cc:56 contin_knob::spec

        C++ returns a reference into a spec that is NOT part of any field
        set (the class's own comment, knobs.h:151-152); this returns the
        same frozen dataclass value every call, so aliasing is moot.
        """
        return self._spec

    def toStr(self) -> str:
        """C++: moses/moses/representation/knobs.cc:58 contin_knob::toStr"""
        loc = self._require_live_location("toStr")
        return f"[{_vertex_to_string(loc.data)}]"


class discrete_knob(disc_knob_base):
    """C++: moses/moses/representation/knobs.h:172 discrete_knob

    Classic's template ``Multiplicity`` is stored as an instance capacity.
    Task 2 constructs only the three-state logical specialization.
    """

    def __init__(
        self, tr: combo_tree, loc: tree_node | None = None, *, capacity: int
    ) -> None:
        super().__init__(tr, loc)
        if capacity <= 0:
            raise ValueError("discrete knob capacity must be positive")
        self._capacity = capacity
        self._disallowed: set[int] = set()
        self._default = 0
        self._current = 0
        self._spec_frozen = False

    def _require_mutable_spec(self, operation: str) -> None:
        if self._spec_frozen:
            raise RuntimeError(f"{operation}: knob specification is frozen")

    def _freeze_spec(self) -> disc_spec:
        """C++: moses/moses/representation/build_knobs.cc:530 add_logical_knobs"""
        self._require_live_location("freeze specification")
        if self._spec_frozen:
            raise RuntimeError("freeze specification: knob is already inserted")
        result = self.spec()
        self._spec_frozen = True
        return result

    def _validate_setting(self, setting: int) -> None:
        # DIVERGENCE(C++ moses/moses/representation/knobs.h:180-211): classic
        # indexes a fixed bitset and relies on caller/assert preconditions;
        # Python rejects out-of-range and non-integral settings explicitly.
        if (
            not isinstance(setting, int)
            or isinstance(setting, bool)
            or not 0 <= setting < self._capacity
        ):
            raise ValueError(f"invalid discrete knob setting: {setting!r}")

    def disallow(self, setting: int) -> None:
        """C++: moses/moses/representation/knobs.h:180 discrete_knob::disallow"""
        self._require_mutable_spec("disallow")
        self._validate_setting(setting)
        self._disallowed.add(setting)

    def allow(self, setting: int) -> None:
        """C++: moses/moses/representation/knobs.h:183 discrete_knob::allow"""
        self._require_mutable_spec("allow")
        self._validate_setting(setting)
        self._disallowed.discard(setting)

    def multiplicity(self) -> int:
        """C++: moses/moses/representation/knobs.h:187 discrete_knob::multiplicity"""
        return self._capacity - len(self._disallowed)

    def in_exemplar(self) -> bool:
        """C++: moses/moses/representation/knobs.h:191 discrete_knob::in_exemplar"""
        return self._default != 0

    def _map_idx(self, idx: int) -> int:
        """C++: moses/moses/representation/knobs.h:205 discrete_knob::map_idx

        This preserves classic's questionable left-shift mapping literally:
        only disallowed raw positions strictly below the swapped index count.
        It can therefore map to a disallowed state.
        """
        # DIVERGENCE(C++ moses/moses/representation/knobs.h:200-206): classic
        # takes an unchecked `int idx` and relies on caller preconditions;
        # Python rejects a non-integral or out-of-range index before it can
        # reach the (verbatim-faithful, see above) mapping arithmetic below.
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or not 0 <= idx < self.multiplicity()
        ):
            raise ValueError(f"invalid discrete knob index: {idx!r}")
        if idx == self._default:
            idx = 0
        elif idx == 0:
            idx = self._default
        return idx + sum(setting < idx for setting in self._disallowed)


class logical_subtree_knob(discrete_knob):
    """C++: moses/moses/representation/knobs.h:235 logical_subtree_knob"""

    absent = 0
    present = 1
    negated = 2

    def __init__(
        self,
        tr: combo_tree,
        target: tree_node,
        subtree_or_knob: tree_node | logical_subtree_knob,
    ) -> None:
        """C++: moses/moses/representation/knobs.cc:68 logical_subtree_knob
        (the copy-ctor overload, taken when ``subtree_or_knob`` is a
        ``logical_subtree_knob``) and knobs.cc:86 (the subtree-ctor overload,
        taken otherwise) -- one Python method dispatches on argument type
        instead of overloading, per this module's usual convention.
        """
        super().__init__(tr, capacity=3)
        if not tr.is_valid(target):
            raise ValueError("logical_subtree_knob: target is erased or invalid")

        if isinstance(subtree_or_knob, logical_subtree_knob):
            source = subtree_or_knob
            source_loc = source.get_loc()
            if source.in_exemplar():
                self._loc = tr.child(target, source._tr.sibling_index(source_loc))
            else:
                self._loc = tr.append_child(target, source_loc)
            self._disallowed = set(source._disallowed)
            self._default = source._default
            self._current = source._current
            return

        subtree = subtree_or_knob
        if subtree is None:
            raise ValueError("logical_subtree_knob: subtree must not be None")

        negated_subtree = combo_tree(subtree)
        negated_subtree.insert_above(negated_subtree.begin(), builtin.logical_not)
        logical_reduction()(1)(negated_subtree)

        for sibling in tr.children(target):
            if tr.equal_subtree(sibling, subtree, vertex_equal) or tr.equal_subtree(
                sibling, negated_subtree.begin(), vertex_equal
            ):
                self._loc = sibling
                self._current = self.present
                self._default = self.present
                return

        self._loc = tr.append_child(target, builtin.null_vertex)
        tr.append_child(self._loc, subtree)

    def complexity_bound(self) -> complexity_t:
        """C++: moses/moses/representation/knobs.cc:123 logical_subtree_knob::complexity_bound"""
        loc = self._require_live_location("complexity_bound")
        return 0 if self._current == self.absent else tree_complexity(loc)

    def clear_exemplar(self) -> None:
        """C++: moses/moses/representation/knobs.cc:127 logical_subtree_knob::clear_exemplar"""
        loc = self._require_live_location("clear_exemplar")
        if self.in_exemplar():
            self.turn(0)
        else:
            self._tr.erase(loc)

    def turn(self, idx: int) -> None:
        """C++: moses/moses/representation/knobs.cc:134 logical_subtree_knob::turn"""
        loc = self._require_live_location("turn")
        mapped = self._map_idx(idx)
        if mapped == self._current:
            return
        if mapped == self.absent:
            if self._current == self.negated:
                loc.data = builtin.null_vertex
            else:
                self._loc = self._tr.insert_above(loc, builtin.null_vertex)
        elif mapped == self.present:
            self._loc = self._tr.erase(self._tr.flatten(loc))
        elif mapped == self.negated:
            if self._current == self.present:
                self._loc = self._tr.insert_above(loc, builtin.logical_not)
            else:
                loc.data = builtin.logical_not
        else:
            raise ValueError(f"invalid mapped logical knob setting: {mapped}")
        self._current = mapped

    @staticmethod
    def _append_vertex(
        candidate: combo_tree, parent_dst: tree_node | None, value: object
    ) -> tree_node:
        if candidate.empty():
            return candidate.set_head(value)
        if parent_dst is None:
            raise ValueError("append_to: nonempty candidate requires a destination parent")
        return candidate.append_child(parent_dst, copy.copy(value))

    def append_to(
        self, candidate: combo_tree, parent_dst: tree_node | None, idx: int
    ) -> tuple[tree_node | None, tree_node | None]:
        """C++: moses/moses/representation/knobs.cc:164 logical_subtree_knob::append_to

        C++ returns the exemplar source iterator and overwrites ``parent_dst``
        by reference. Python returns both as ``(source, updated_parent)``.
        """
        # DIVERGENCE(C++ moses/moses/representation/knobs.cc:164-190): Python
        # cannot overwrite a caller's local reference, so append_to returns the
        # source and updated destination handles as a pair. Classic
        # initializes its absent result from `parent_dst.end()` (knobs.cc:177),
        # i.e. `iterator_base::end()` (opencog/util/tree.h:2356-2361): that
        # default-constructs a sibling_iterator (`node == 0`, tree.h:2228's
        # default ctor) and assigns `parent_dst`'s node into the returned
        # iterator's `parent_` field, never into its own `node`. So the
        # result's `node` is unconditionally 0 -- NOT "depending on
        # parent_dst" -- and `tree::is_valid` (tree.h:2166-2171, `node == 0`
        # among its checks) always rejects it; it can never pass is_valid.
        # It also does not compare equal to the candidate's own `end()`
        # (a pre/post-order iterator over `feet`, a distinct non-null
        # sentinel, not `node == 0`). Python normalizes every absent source
        # to None, which is the same "definitely not a real node" outcome
        # classic's always-invalid iterator produces -- the normalization is
        # right; the above is what actually justifies it.
        loc = self._require_live_location("append_to")
        mapped = self._map_idx(idx)
        updated_parent = parent_dst
        if mapped == self.negated:
            updated_parent = self._append_vertex(
                candidate, updated_parent, builtin.logical_not
            )
        if mapped == self.absent:
            return None, updated_parent

        new_source = loc if self._default == self.present else loc.first_child
        if new_source is None:
            raise ValueError("append_to: absent-default knob has no stored subtree")
        updated_parent = self._append_vertex(candidate, updated_parent, new_source.data)
        return new_source, updated_parent

    def spec(self) -> disc_spec:
        """C++: moses/moses/representation/knobs.cc:193 logical_subtree_knob::spec"""
        return disc_spec(self.multiplicity())

    def _loc_str(self, negate: bool = False) -> str:
        """C++: moses/moses/representation/knobs.cc:209 logical_subtree_knob::locStr"""
        loc = self._require_live_location("toStr")
        if vertex_equal(loc.data, builtin.null_vertex):
            if not self._tr.has_one_child(loc):
                raise ValueError("null logical knob location must have exactly one child")
            loc = loc.first_child
        value = loc.data
        if isinstance(value, argument):
            value = copy.copy(value)
            if negate:
                value.negate()
            return ("!" if value.is_negated() else "") + f"${value.abs_idx()}"
        return ("!" if negate else "") + _vertex_to_string(value)

    def _pos_str(self, position: int, tag_current: bool = False) -> str:
        """C++: moses/moses/representation/knobs.cc:231 logical_subtree_knob::posStr"""
        if position == self.absent:
            value = "nil"
        elif position == self.present:
            value = self._loc_str()
        elif position == self.negated:
            value = self._loc_str(True)
        else:
            value = "INVALID SETTING"
        return f"({value})" if position == self._current and tag_current else value

    def toStr(self) -> str:
        """C++: moses/moses/representation/knobs.cc:197 logical_subtree_knob::toStr"""
        self._require_live_location("toStr")
        return "[" + " ".join(
            self._pos_str(self._map_idx(idx)) for idx in range(self.multiplicity())
        ) + "]"
