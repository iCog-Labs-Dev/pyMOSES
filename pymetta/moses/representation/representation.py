"""Immutable Boolean and continuous candidate construction from packed MOSES
instances."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, tree_to_stream_str
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_contin, vertex, vertex_equal
from pymetta.comboreduct.reduct.clean_reduction import clean_reduce
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.moses.complexity import tree_complexity
from pymetta.moses.representation.build_knobs import build_knobs
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import contin_t, instance
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.util.logger import logger
from pymetta.util.tree import tree_node

# C++ `sizeof` values for representation.cc:202's `size_t is = sizeof(instance)
# + sizeof(packed_t) * _fields.packed_width();`, measured on this toolchain
# 2026-09-09 by the same probe that produced `field_set.py`'s constants:
# instance=24, packed_t=8. Cross-checked against classic's own log, which
# reports "One instance, in bytes: 32" for a one-word field_set (24 + 8*1).
_SIZEOF_INSTANCE = 24
_SIZEOF_PACKED_T = 8


def _vertex_text(value: object) -> str:
    """No direct C++ symbol: `operator<<(ostream&, const vertex&)`'s single-
    VALUE rendering (not a subtree's), as `ostream_prototype` below uses it
    (representation.h:145/148's `out << *it`). Same one-line idiom
    `knobs.py`'s own private `_vertex_to_string` uses, duplicated here rather
    than imported across modules for that helper's own leading-underscore
    privacy.
    """
    return combo_tree_to_str(combo_tree(copy.copy(value))).rstrip()


Simplifier = Callable[[combo_tree], None]

# C++: moses/moses/representation/representation.cc:59-65 (module-level
# statics `stepsize`, `expansion`, `depth`, and their `set_*` setters).
# Ported as module globals here too, deliberately NOT as `representation.
# __init__` keyword parameters: classic's own constructor does not take
# them either -- `representation::representation` always reads these THREE
# process-wide statics and threads them into `build_knobs`'s ctor
# (representation.cc:113: `build_knobs(_exemplar, tt, *this, ignore_ops,
# perceptions, actions, linear_contin, stepsize, expansion, depth,
# perm_ratio)`). `--contin-depth` (problem-params.cc:506, default 5) is the
# only one a real CLI run ever changes, via `set_depth` (problem-params.cc:
# 2372) -- wiring that flag is `moses/main/`'s job, out of this module's
# owned-file scope; these setters exist so that later slice has something
# to call.
_stepsize: contin_t = 1.0
_expansion: contin_t = 2.0
_depth: int = 5


def set_stepsize(new_ss: float) -> None:
    """C++: moses/moses/representation/representation.cc:67 set_stepsize"""
    global _stepsize
    _stepsize = contin_t(new_ss)


def set_expansion(new_ex: float) -> None:
    """C++: moses/moses/representation/representation.cc:69 set_expansion"""
    global _expansion
    _expansion = contin_t(new_ex)


def set_depth(new_depth: int) -> None:
    """C++: moses/moses/representation/representation.cc:71 set_depth"""
    global _depth
    _depth = int(new_depth)


# DIVERGENCE(C++ moses/moses/representation/representation.h:45-205): this
# class contains only the immutable Boolean (S8) and continuous (S14 Task 2)
# construction paths. The mutable transform/get_candidate_lock API is
# unnecessary in Python and would weaken the thread-safe get_candidate
# design. Action-result maps are deferred to that allowed-domain slice. Term,
# enum, ANN, object, message, procedure, and EXEMPLAR_INST_IS_UNDEAD paths
# are outside project scope.
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
        linear_contin: bool = True,
    ) -> None:
        """C++: moses/moses/representation/representation.cc:99 representation::representation

        # DIVERGENCE(C++ moses/moses/representation/representation.h:61):
        # `linear_contin` defaults to `True`, matching classic's own
        # constructor default verbatim (`bool linear_contin = true`) --
        # NOT the real `moses` CLI's actual default of `False`
        # (`--linear-regression`, problem-params.cc:292), which is a
        # SEPARATE, later layer's decision (`moses/main/`, out of this
        # module's scope) that overrides this constructor default on every
        # real run. `deme_expander.py`'s own DIVERGENCE note already
        # documents that at this exact boundary: it never forwards
        # `linear_contin` at all, relying on it defaulting to `True` here
        # being byte-identical to classic's own `deme_params` default (also
        # `true`, deme_params.h:49) for the one signature it can build
        # (Boolean, which never reads the flag). See CARRY-FORWARD.md
        # Sec 5.4 -- a default differing from classic's is a divergence
        # even when nothing reads it yet.
        """
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

        # C++: representation.cc:110-121. All three records precede
        # `build_knobs`, which is why classic logs "Start knob building"
        # BEFORE the exemplar it is about to decorate.
        logger().info(
            f"Start knob building, rep size={self._exemplar.size()}"
            f" complexity={tree_complexity(self._exemplar)}"
        )
        if logger().is_debug_enabled():
            logger().debug(
                "Building representation from exemplar: " + tree_to_stream_str(self._exemplar)
            )

        if logger().is_debug_enabled():
            # C++: representation.cc:117-120 builds ONE stringstream from two
            # `ostream_prototype` calls, so this is a single record with both
            # halves in it. The second is on `exemplar_`, the caller's
            # ORIGINAL tree -- not `_exemplar`, which is the copy about to be
            # decorated -- so at this point the two read identically and
            # diverge by the time the "Created prototype:" record repeats the
            # pair further down.
            logger().debug(
                "Before Created prototype: "
                + self.ostream_prototype()
                + " Creating From: "
                + self.ostream_prototype(exemplar.begin())
            )

        # Task 3 already runs clean_reduce before invoking this callback.
        build_knobs(
            self._exemplar,
            signature,
            self,
            self._simplify_knob_building,
            ignore_ops=ignore_ops,
            perm_ratio=perm_ratio,
            linear_contin=linear_contin,
            step_size=_stepsize,
            expansion=_expansion,
            depth=_depth,
        )

        logger().info(
            f"After knob building, rep size={self._exemplar.size()}"
            f" complexity={tree_complexity(self._exemplar)}"
        )
        if logger().is_fine_enabled():
            logger().fine("Rep, after knob building: " + tree_to_stream_str(self._exemplar))

        # C++: representation.cc:161-166 (`std::multiset<field_set::spec>
        # tmp; for disc_v: tmp.insert(v.first); for contin_v:
        # tmp.insert(v.first); _fields = field_set(tmp.begin(), tmp.end());`).
        # `field_set.py`'s own constructor already sorts a mixed iterable of
        # contin_spec/disc_spec with contin ordered first regardless of
        # value (matching `boost::variant<term_spec, contin_spec,
        # disc_spec>`'s own `which()`-first ordering -- term_spec is out of
        # this port's scope), stably preserving each group's OWN relative
        # order for exact ties -- so which order the two generators below
        # are chained in does not change the result; this follows classic's
        # own disc-then-contin insertion order for readability only.
        self._fields = field_set(
            [entry.spec for entry in self.disc] + [entry.spec for entry in self.contin]
        )
        # C++: representation.cc:172. `tmp.size()` is the multiset's size,
        # i.e. disc knobs plus contin knobs, before either index map is built.
        logger().info(f"Total number of field specs: {len(self.disc) + len(self.contin)}")

        disc_start = self._fields.begin_disc_raw_idx()
        self._it_disc_idx = {
            entry.knob.get_loc(): disc_start + offset for offset, entry in enumerate(self.disc)
        }
        contin_start = self._fields.begin_contin_raw_idx()
        self._it_contin_idx = {
            entry.knob.get_loc(): contin_start + offset for offset, entry in enumerate(self.contin)
        }

        # C++: representation.cc:200-203. Note representation.cc:189's
        # "Number of disc knobs mapped" twin is COMMENTED OUT upstream, so
        # only the contin count is reported -- reproduced as classic has it.
        logger().info(f"Number of contin knobs mapped: {len(self.contin)}")
        logger().info(f"Field set, in bytes: {self._fields.byte_size()}")
        # C++: `size_t is = sizeof(instance) + sizeof(packed_t) *
        # _fields.packed_width();` -- two more C++ `sizeof` constants, both
        # measured alongside `field_set.py`'s own (instance=24, packed_t=8),
        # and cross-checked against classic's "One instance, in bytes: 32"
        # for a one-word field_set.
        one_instance_bytes = _SIZEOF_INSTANCE + _SIZEOF_PACKED_T * self._fields.packed_width()
        logger().info(f"One instance, in bytes: {one_instance_bytes}")

        if logger().is_debug_enabled():
            # C++: representation.cc:205-210 -- the same two-call pair as the
            # "Before Created prototype:" record above, now that the knobs
            # exist, so the first half shows the decorated prototype and the
            # second the untouched original.
            logger().debug(
                "Created prototype: "
                + self.ostream_prototype()
                + " Created From: "
                + self.ostream_prototype(exemplar.begin())
            )

    def fields(self) -> field_set:
        """C++: moses/moses/representation/representation.h:118 representation::fields"""
        return self._fields

    def exemplar(self) -> combo_tree:
        """C++: moses/moses/representation/representation.h:126 representation::exemplar

        C++ returns a const reference. Python has no const-reference view, so
        return a value copy to keep the identity-indexed internal tree immutable.
        """
        return combo_tree(self._exemplar.begin())

    def ostream_prototype(self, it: tree_node | None = None) -> str:
        """C++: moses/moses/representation/representation.h:182 representation::ostream_prototype

        "Like above but on the _exemplar." Renders the FULL decorated
        exemplar, replacing every knobbed location's own text with its
        knob's ``toStr()`` -- the same "Created prototype: ..." string
        classic's own `-l DEBUG` log line emits. An oracle test can compare
        this single string directly against a real classic run's log line,
        or against a C++ probe calling the same ``rep.ostream_prototype``.

        `it` merges C++'s OTHER overload, `ostream_prototype(out, it)`
        (representation.h:142), which walks an arbitrary iterator instead of
        `_exemplar.begin()` -- the same overload-merging convention used
        elsewhere in this port. The constructor's log records need it: they
        print the caller's original `exemplar_` alongside `_exemplar`, and
        an iterator into the original resolves no knobs (the knob maps key
        on `_exemplar`'s own nodes), so it renders as a plain tree.
        """
        root = self._exemplar.begin() if it is None else it
        if root is None:
            return ""
        return self._ostream_prototype_at(root)

    def _ostream_prototype_at(self, it: tree_node) -> str:
        """C++: moses/moses/representation/representation.h:142 representation::ostream_prototype

        Breadcrumb kept on one line for `tools/check_alignment.py`'s parser.
        """
        # C++'s `disc`/`contin` multimaps are EMPTY until `build_knobs` fills
        # them, and `ostream_prototype` is called once before it runs (the
        # "Before Created prototype:" record, representation.cc:117-120): the
        # lookups there simply miss. This port's `knob_mapper` refuses a
        # lookup before `finalize()` instead (knob_mapper.py's own DIVERGENCE
        # note), so "not finalized yet" is treated as "no knob here", which
        # is what an empty multimap gives classic.
        mapped = self._finalized

        if is_contin(it.data):
            knob = self.find_contin_knob(it) if mapped else None
            text = _vertex_text(it.data) if knob is None else knob.toStr()
        else:
            knob = self.find_disc_knob(it) if mapped else None
            # DIVERGENCE(C++ moses/moses/representation/representation.h:
            # 151-158): classic special-cases `action_subtree_knob` here
            # (returns early rather than recursing into its own already-
            # printed subtree). That knob type is out of this port's scope
            # (knobs.py's own module docstring), so the special case has no
            # counterpart to omit -- every disc knob this port can build
            # (`logical_subtree_knob`) recurses normally, matching classic's
            # own behaviour for that same knob type.
            text = _vertex_text(it.data) if knob is None else knob.toStr()

        # if null_vertex then print its child instead
        node = it
        if vertex_equal(node.data, builtin.null_vertex):
            if not self._exemplar.has_one_child(node):
                raise AssertionError("null_vertex prototype node must have exactly one child")
            node = node.first_child

        if not self._exemplar.is_childless(node):
            parts = [self._ostream_prototype_at(child) for child in self._exemplar.children(node)]
            text += "(" + " ".join(parts) + ")"
        return text

    def get_clean_exemplar(self, reduce: bool, knob_building: bool = False) -> combo_tree:
        """C++: moses/moses/representation/representation.cc:251 representation::get_clean_exemplar

        Breadcrumb kept on one line for `tools/check_alignment.py`'s parser.
        """
        result = self.exemplar()
        self.clean_combo_tree(result, reduce, knob_building)
        return result

    def clean_combo_tree(self, tr: combo_tree, reduce: bool, knob_building: bool = False) -> None:
        """C++: moses/moses/representation/representation.cc:259 representation::clean_combo_tree"""
        if not isinstance(tr, combo_tree):
            raise TypeError("clean_combo_tree requires a combo_tree")
        clean_reduce(tr)
        if reduce:
            reducer = self._simplify_knob_building if knob_building else self._simplify_candidate
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
            raise ValueError(f"instance has {len(inst)} words; field set requires {expected}")

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

        contin_kb = self.find_contin_knob(src)
        if contin_kb is not None:
            # A contin_knob's target is always a childless contin_t leaf
            # (build_knobs.build_contin only ever wraps `is_contin` nodes),
            # so -- unlike the disc branch above -- there is never a
            # `new_src` to recurse into (contin_knob.append_to returns the
            # appended node itself, not a source to descend from).
            try:
                raw_idx = self._it_contin_idx[src]
            except KeyError as error:
                raise RuntimeError("candidate knob has no raw field index") from error
            value = self._fields.get_contin(inst, raw_idx)
            contin_kb.append_to(candidate, parent_dst, value)
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
