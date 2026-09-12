"""Boolean, continuous and mixed representation builder.

This module ports classic ``build_knobs``' Boolean output path (S8), its
continuous output path (S14 Task 2), and -- as of the mixed-predicate slice
-- the MIXED path that joins them: the ``is_predicate`` branches inside
``logical_canonize`` (build_knobs.cc:160) and ``build_logical``
(build_knobs.cc:640), plus the contin branches of ``insert_typed_arg``
(build_knobs.cc:386) and ``sample_logical_perms`` (build_knobs.cc:425).
A Boolean-output problem over contin inputs therefore builds
``greater_than_zero`` perms, canonizes each predicate's contin argument and
adds contin knobs inside it, exactly as classic does. Action expressions
remain a separate allowed-domain slice; every excluded domain is
intentionally absent here.

Random logical PAIR perms (``build_knobs.cc:435-518``) used to be absent too,
with ``perm_ratio`` hard-restricted to ``-1`` (classic's "no pairs"
early-return mode). That was a real, measured behavioural divergence, not a
cosmetic one: classic's own default is ``perm_ratio = 0.0``
(``build_knobs.h:56``, ``representation.h:62``, ``problem-params.cc:299``),
which generates pairs, and without them the ``pa`` (even-parity) demo problem
is UNSOLVABLE by the port at any budget while classic solves it in a single
deme expansion. See ``.superpowers/sdd/2026-09-08-pymetta-s13-cli/
parity-divergence-report.md``. Pair sampling is now ported.

# DIVERGENCE(C++ moses/moses/representation/build_knobs.h:54): classic's own
# `linear_regression`/`linear_contin` constructor parameter DEFAULTS to
# `true`; this is that same, unchanged default -- copied here rather than
# quietly narrowed to whatever a caller happens to always pass, because
# CARRY-FORWARD.md Sec 5.4 is explicit that a default differing from
# classic's is a divergence even when nothing currently reads it. The real
# `moses` binary's OWN CLI default (`--linear-regression`, problem-params.cc:
# 292) is `false`, a SEPARATE, later layer's decision (moses/moses/main/,
# out of this module's scope) that overrides this constructor default on
# every real run -- see `rec_canonize`'s own docstring for where that matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    get_contin,
    is_argument,
    is_contin,
    is_contin_expr,
    swap_and_or,
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
from pymetta.moses.representation.instance import contin_t
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import contin_knob, logical_subtree_knob
from pymetta.util.lazy_random_selector import lazy_random_selector
from pymetta.util.tree import tree_node

ProbeSimplifier = Callable[[combo_tree], None]

_FUTURE_OUTPUT_TYPES = frozenset(
    {
        type_node.action_result_type,
    }
)

# Vertex types `append_linear_combination` (build_knobs.cc:936) accepts for a
# contin exemplar's INPUT arguments -- a contin-typed argument is inserted
# directly, a boolean-typed one is wrapped in `impulse` first. Any other
# input type raises there, matching classic's own
# `OC_ASSERT(0, "... unsupported argument type")` (build_knobs.cc:960-961).
_CONTIN_INPUT_TYPES = frozenset({type_node.contin_type, type_node.boolean_type})

# Builtins a contin exemplar's own nodes are permitted to hold, pre-
# canonicalization (see `_validate_contin_exemplar`). `greater_than_zero`
# is in the set as of the mixed-predicate slice: `rec_canonize`'s own
# `impulse` branch calls `logical_canonize`, whose `is_predicate` case is
# now ported, so a predicate nested under `impulse` inside a contin
# exemplar is handled rather than rejected.
_CONTIN_ALLOWED_BUILTINS = frozenset(
    {
        builtin.plus,
        builtin.times,
        builtin.div,
        builtin.sin,
        builtin.log,
        builtin.exp,
        builtin.impulse,
        builtin.greater_than_zero,
        builtin.logical_and,
        builtin.logical_or,
        builtin.logical_not,
        builtin.logical_true,
        builtin.logical_false,
    }
)

# Builtins a Boolean exemplar's own LOGICAL nodes are permitted to hold
# (see `_validate_boolean_node`). `logical_not` is admitted only directly
# above a predicate, matching build_knobs.cc:608-611's own OC_ASSERT
# ("the tree is supposed to be in normal form; and thus must not contain
# logical_not nodes") -- a negated Boolean literal is an `argument` with a
# negative index in combo, never a `logical_not`.
_BOOLEAN_ALLOWED_BUILTINS = frozenset(
    {
        builtin.logical_true,
        builtin.logical_false,
        builtin.logical_and,
        builtin.logical_or,
    }
)


# DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:53-123): this
# vertical slice admits Boolean-output and contin-output signatures over
# Boolean and/or contin inputs -- including the MIXED case, where a
# Boolean-output tree carries `greater_than_zero` predicates over contin
# sub-expressions (`logical_canonize`/`build_logical`'s own `is_predicate`
# branches, and `insert_typed_arg`/`sample_logical_perms`' contin
# branches, are all ported). Action-result output belongs to a later
# allowed slice; enum, ANN, object, message, and procedure domains remain
# excluded. Pair sampling (build_knobs.cc:435-518) IS ported, including
# its `lazy_random_selector` draws, so `perm_ratio` accepts classic's full
# range.
class build_knobs:
    """C++: moses/moses/representation/build_knobs.h:47 build_knobs

    Decorate a reduced Boolean-output or contin-output exemplar with knobs.
    Either may mix Boolean and contin inputs, and a Boolean-output one may
    carry `greater_than_zero` predicates over contin sub-expressions.
    Construction is eager, as in classic: on return the exemplar is decorated
    and ``mapper`` has been finalized in canonical spec order.
    """

    def __init__(
        self,
        exemplar: combo_tree,
        signature: TypeSignature,
        mapper: knob_mapper,
        simplify_knob_building: ProbeSimplifier,
        *,
        ignore_ops: Collection[vertex] = (),
        perm_ratio: float = 0.0,
        linear_contin: bool = True,
        step_size: contin_t = 1.0,
        expansion: contin_t = 1.0,
        depth: int = 4,
    ) -> None:
        """C++: moses/moses/representation/build_knobs.cc:53 build_knobs::build_knobs

        ``linear_contin``/``step_size``/``expansion``/``depth`` default to
        build_knobs.h:54-56's own literal constructor defaults (``true``,
        ``1.0``, ``1.0``, ``4``) -- NOT to the DIFFERENT values
        (``stepsize=1.0, expansion=2.0, depth=5``) classic's real
        ``representation`` constructor actually passes in from its own
        module-static state (representation.cc:59-65,
        ``representation.py``'s own module docstring). Those are a
        separate layer's decision; this constructor's defaults are
        never overridden by them unless a caller (``representation.py``)
        explicitly passes them through, exactly as classic's own
        `representation::representation` does.
        """
        self._validate_request(
            exemplar,
            signature,
            mapper,
            simplify_knob_building,
            perm_ratio,
        )
        if not isinstance(linear_contin, bool):
            raise TypeError("linear_contin must be a bool")
        self._exemplar = exemplar
        self._signature = signature
        self._mapper = mapper
        self._simplify_knob_building = simplify_knob_building
        self._ignore_ops = tuple(ignore_ops)
        # C++: build_knobs.cc:63 `_perm_ratio(perm_ratio)`. Stored as a
        # Python float, which is a binary64 where C++'s `_perm_ratio` is a
        # binary32 `float` (build_knobs.h:92). That narrowing is NOT
        # reproduced: the only two reads (build_knobs.cc:436's `<= -1.0`
        # test and :476-479's `n_pairs` arithmetic) are a comparison against
        # an exactly-representable constant and a `floor()` of a product,
        # and every value the CLI can supply comes from
        # `po::value<double>` (problem-params.cc:299) before being narrowed
        # to `float` on the way in -- so the port is, if anything, closer to
        # the user's own literal than classic is.
        self._perm_ratio = float(perm_ratio)
        self._arity = type_tree_arity(signature)
        # C++: build_knobs.cc:60 `_skip_disc_probe(true)`, then :95-96's
        # `if (5 * boolean_arity(_signature) > type_tree_arity(_signature))
        # _skip_disc_probe = false;` INSIDE the boolean_type branch. This
        # port folds the two into one unconditional expression: on the
        # Boolean path the value is identical, and on the contin path the
        # member is never READ at all (`disc_probe` is only ever called
        # from `build_logical`, never from `build_contin`), so classic's
        # `true` there and this expression's possible `False` are
        # indistinguishable.
        #
        # NOTE (mixed-predicate slice): the `if self._skip_disc_probe:
        # return True` short-circuit in `disc_probe` is REACHABLE now, and
        # a previous KNOWN-UNEXERCISED note here saying otherwise is
        # deleted as false. It was true only while every input was forced
        # Boolean, which made this `not (5 * arity > arity)`, i.e. `arity
        # == 0` -- the one shape in which `disc_probe` is never called.
        # With contin inputs admitted, a table of (say) two contin columns
        # gives `5 * 0 > 2` -> False -> `_skip_disc_probe` True, with
        # `disc_probe` called for every perm: classic's own comment
        # (build_knobs.cc:88-93) says exactly that the probe is worth its
        # cost only when at least a fifth of the inputs are Boolean.
        self._skip_disc_probe = not (
            5 * boolean_arity(signature) > type_tree_arity(signature)
        )
        # C++: build_knobs.cc:62-63 `_linear_contin(linear_regression),
        # _step_size(step_size), _expansion(expansion), _depth(depth)`. Only
        # read by the contin-output path (`rec_canonize`'s own
        # `!_linear_contin` branch; `build_contin`'s `contin_knob(...,
        # _step_size, _expansion, _depth)`) -- computed unconditionally here
        # too, matching classic's own unconditional member-init-list.
        self._linear_contin = linear_contin
        self._step_size = contin_t(step_size)
        self._expansion = contin_t(expansion)
        self._depth = int(depth)

        if signature.otype is type_node.boolean_type:
            self.logical_canonize(self._exemplar.begin())
            self.build_logical(self._exemplar.begin(), self._exemplar.begin())
            self.logical_cleanup()
        else:
            # `_validate_request` has already rejected every otype except
            # boolean_type and contin_type by this point.
            self.contin_canonize(self._exemplar.begin())
            self.build_contin(self._exemplar.begin())
        self._mapper.finalize()

    # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:53-123): classic
    # relies on assertions and unchecked caller preconditions across construction.
    # Python validates the entire deterministic request before
    # mutating the exemplar: nonempty/reduced supported shape, in-range
    # arguments, a fresh mapper, an in-place simplifier callback, and a
    # real perm_ratio. Invalid requests therefore fail atomically rather than
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
        if signature.otype not in (type_node.boolean_type, type_node.contin_type):
            if signature.otype in _FUTURE_OUTPUT_TYPES:
                raise NotImplementedError(
                    f"{signature.otype.name} output belongs to a future allowed domain slice"
                )
            raise NotImplementedError(
                f"{signature.otype.name} output is excluded from the scoped port"
            )
        if exemplar.empty():
            raise ValueError("exemplar must not be empty")
        if not isinstance(mapper, knob_mapper):
            raise TypeError("build_knobs requires a knob_mapper")
        if mapper.disc or mapper.contin or getattr(mapper, "_finalized", False):
            raise ValueError("build_knobs requires a fresh, empty knob_mapper")
        if not callable(simplify_knob_building):
            raise TypeError("simplify_knob_building must be callable")
        # Classic's own early-return test (build_knobs.cc:435-437) is
        # `_perm_ratio <= -1.0`, not `== -1.0`: any ratio at or below -1
        # (e.g. -2.0) skips pair sampling, everything above it samples
        # pairs, and classic's own default is 0.0 (build_knobs.h:56,
        # representation.h:62, problem-params.cc:299). This port used to
        # accept ONLY -1.0 and reject every other value; that made the
        # default configuration unreachable and silently removed half the
        # knobs from every Boolean representation. The whole range is
        # accepted now. `bool` is still rejected because `isinstance(True,
        # int)` is True in Python and `perm_ratio=True` is far more likely
        # to be a caller mistake than a request for ratio 1.0.
        if isinstance(perm_ratio, bool) or not isinstance(perm_ratio, int | float):
            raise TypeError("perm_ratio must be a real number")
        if math.isnan(float(perm_ratio)):
            raise ValueError("perm_ratio must not be NaN")

        if signature.otype is type_node.boolean_type:
            build_knobs._validate_boolean_exemplar(exemplar, signature)
        else:
            build_knobs._validate_contin_exemplar(exemplar, signature)

    @staticmethod
    def _validate_boolean_exemplar(exemplar: combo_tree, signature: TypeSignature) -> None:
        """Structural check for a Boolean-output exemplar, predicates included.

        A Boolean-output signature's INPUTS may be Boolean or contin
        (build_knobs.cc:82-86's own comment: "Exemplar may consist of a mix
        of boolean and contin-typed variables and functions, but we want the
        end result to be typed boolean"). The walk is recursive rather than
        flat precisely so that the two halves stay separated: logic ops,
        Boolean constants and Boolean-typed argument literals above, and a
        full contin sub-expression only ever BELOW a `greater_than_zero`.

        THERE IS NO SIGNATURE-WIDE INPUT-TYPE GATE HERE, and that is
        deliberate. One used to reject any signature holding an input that
        was neither Boolean nor contin. The CONTIN-output path is right to
        have such a gate -- `append_linear_combination` visits EVERY
        argument index and `OC_ASSERT(0, "... unsupported argument type")`s
        on anything else (build_knobs.cc:958-961), and classic really does
        abort (SIGABRT, rc 134) on such a table. The BOOLEAN path has no
        such assert anywhere: `sample_logical_perms` (build_knobs.cc:422-430)
        and `insert_typed_arg` (:381-393) both have a bare `if boolean /
        else if contin` with NO `else`, so an argument of any other type is
        silently skipped and simply never gets a knob.

        That difference is CLI-reachable and cost a whole run. A CSV column
        whose values parse as none of the known types is inferred
        `ill_formed_type` (a date column such as `2012-01-03` is the easy
        way to get one), and classic then searches happily over the
        remaining columns -- `->(boolean boolean ill_formed contin boolean)`
        on `i1,i2,when,w,y` gives `-2 and($i1 $i2)`. The port refused the
        whole run. Nothing about the weight column (`--score-weight`) was
        involved; that flag is byte-identical with and without.
        """
        build_knobs._validate_boolean_node(
            exemplar, exemplar.begin(), signature, type_tree_arity(signature)
        )

    @staticmethod
    def _validate_boolean_node(
        exemplar: combo_tree,
        node: tree_node,
        signature: TypeSignature,
        arity: int,
    ) -> None:
        value = node.data
        if isinstance(value, argument):
            if not exemplar.is_childless(node):
                raise ValueError("Boolean argument literals must be childless")
            if not 1 <= value.abs_idx() <= arity:
                raise ValueError(
                    f"Boolean argument ${value.abs_idx()} is outside signature arity {arity}"
                )
            if signature.itypes[value.abs_idx() - 1] is not type_node.boolean_type:
                raise ValueError(
                    f"argument ${value.abs_idx()} is contin-typed and may only appear "
                    "inside a predicate's contin sub-expression"
                )
            return
        if isinstance(value, builtin):
            if vertex_equal(value, builtin.greater_than_zero):
                if exemplar.number_of_children(node) != 1:
                    raise ValueError("greater_than_zero must have exactly one child")
                build_knobs._validate_contin_node(exemplar, node.first_child, signature, arity)
                return
            if vertex_equal(value, builtin.logical_not):
                # `is_predicate` is exactly build_knobs.cc:609-610's own
                # assertion: a `not` is legal only directly above a
                # predicate. It reads the child itself, so a childless
                # `not` is rejected here before it can reach that read.
                if exemplar.is_childless(node) or not is_predicate(node):
                    raise ValueError(
                        "logical_not is only permitted directly above a predicate; "
                        "a negated Boolean literal is a negative-index argument"
                    )
                build_knobs._validate_boolean_node(exemplar, node.first_child, signature, arity)
                return
            if any(vertex_equal(value, allowed) for allowed in _BOOLEAN_ALLOWED_BUILTINS):
                if is_boolean(value) and not exemplar.is_childless(node):
                    raise ValueError("Boolean constants must be childless")
                for child in exemplar.children(node):
                    build_knobs._validate_boolean_node(exemplar, child, signature, arity)
                return
        raise ValueError(
            f"Boolean exemplar contains an unsupported or non-reduced vertex: {value!r}"
        )

    # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:53-123): a
    # contin-output signature's INPUTS may be Boolean or contin (`type_tree_
    # arity`/`append_linear_combination`, build_knobs.cc:948-965, dispatch on
    # each input's own declared type). A predicate (`greater_than_zero`) is
    # admitted too as of the mixed-predicate slice: `rec_canonize`'s
    # `impulse` branch hands its Boolean argument to `logical_canonize`,
    # whose own `is_predicate` case is ported. Note what classic does NOT do
    # there -- the contin-output path never calls `build_logical`, so a
    # predicate reached this way is canonized but gets no DISC knobs, only
    # the contin knobs `build_contin`'s whole-tree walk then finds inside it.
    @staticmethod
    def _validate_contin_exemplar(exemplar: combo_tree, signature: TypeSignature) -> None:
        if any(input_type not in _CONTIN_INPUT_TYPES for input_type in signature.itypes):
            raise NotImplementedError(
                "contin output only supports Boolean or contin inputs in this scoped port"
            )
        arity = type_tree_arity(signature)
        for node in exemplar:
            build_knobs._validate_contin_node(exemplar, node, signature, arity, recurse=False)

    @staticmethod
    def _validate_contin_node(
        exemplar: combo_tree,
        node: tree_node,
        signature: TypeSignature,
        arity: int,
        recurse: bool = True,
    ) -> None:
        """Per-node half of `_validate_contin_exemplar`'s own walk.

        `recurse=False` is what `_validate_contin_exemplar` passes: it is
        already iterating every node of the whole tree itself, so each node
        is checked exactly once. `_validate_boolean_node` passes the default
        instead, to check one predicate's contin SUBTREE and nothing above it.
        """
        value = node.data
        if is_contin(value):
            if not exemplar.is_childless(node):
                raise ValueError("contin constants must be childless")
        elif is_argument(value):
            if not exemplar.is_childless(node):
                raise ValueError("argument literals must be childless")
            if not 1 <= value.abs_idx() <= arity:
                raise ValueError(f"argument ${value.abs_idx()} is outside signature arity {arity}")
        elif isinstance(value, builtin) and any(
            vertex_equal(value, allowed) for allowed in _CONTIN_ALLOWED_BUILTINS
        ):
            if is_boolean(value) and not exemplar.is_childless(node):
                raise ValueError("Boolean constants must be childless")
        else:
            raise ValueError(
                f"contin expression contains an unsupported or non-reduced vertex: {value!r}"
            )
        if recurse:
            for child in exemplar.children(node):
                build_knobs._validate_contin_node(exemplar, child, signature, arity)

    def permitted_op(self, value: vertex) -> bool:
        """C++: moses/moses/representation/build_knobs.cc:131 build_knobs::permitted_op"""
        return not any(vertex_equal(value, ignored) for ignored in self._ignore_ops)

    def logical_canonize(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:150 build_knobs::logical_canonize

        The canonization proceeds recursively if the top node is a
        predicate: its contin argument is `linear_canonize`d first, and only
        then is the `and`/`or` pair inserted above the predicate itself --
        the order matters, because `insert_above` would otherwise leave
        `it` no longer pointing at the predicate whose argument is wanted.
        """
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
        elif is_predicate(it):
            cit = it
            if vertex_equal(it.data, builtin.logical_not):
                # skip over negation.
                cit = cit.first_child

            # At this time, we assume that all predicates are of type
            # -> (contin boolean).  So, if they take something else, then
            # the linear_canonize will obviously puke.
            self.linear_canonize(cit.first_child)  # get arg to predicate.

            parent = self._exemplar.insert_above(it, builtin.logical_and)
            self._exemplar.insert_above(parent, builtin.logical_or)
        else:
            raise ValueError("unexpected vertex during Boolean logical canonization")

    def logical_cleanup(self) -> None:
        """C++: moses/moses/representation/build_knobs.cc:269 build_knobs::logical_cleanup"""
        # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:270-276):
        # classic walks the post-order iterator LIVE, erasing through
        # `_exemplar.erase(it++)` as it goes -- per the S0-S6 plan's explicit
        # "do not fix it by snapshotting" rule, that is normally something
        # this port must not paper over. This snapshots the post-order
        # sequence into a plain list up front instead, then re-checks
        # `is_childless` fresh at each node during the loop (not a stale
        # value captured at snapshot time). The two are behaviourally
        # equivalent here specifically because only CHILDLESS nodes are ever
        # erased: post-order guarantees every node is visited strictly after
        # all of its own descendants, so an erased node can never be an
        # ancestor of a later-visited node in this same snapshot, and a
        # parent that becomes childless only because an earlier sibling/child
        # was erased is still observed correctly, since its childless-ness is
        # re-tested live against the real tree, not against the snapshot.
        # Consequence: the `self._exemplar.is_valid(node)` guard below can
        # never actually fire -- no snapshot entry is ever erased before its
        # own turn -- and mutating it away (deleting the conjunct entirely)
        # still leaves the test suite green. Kept anyway as the guard classic
        # gets for free from `it++` capturing the successor before erasing.
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

    def insert_typed_arg(
        self,
        tr: combo_tree,
        arg_type: type_node,
        arg: argument,
        negate: bool = False,
    ) -> None:
        """C++: moses/moses/representation/build_knobs.cc:379 build_knobs::insert_typed_arg

        Classic takes the argument's `type_tree_sib_it`; this port takes the
        `type_node` it dereferences to, since `TypeSignature.itypes` is a
        flat tuple rather than a walkable child list.

        Boolean appends a (possibly negated) copy of the argument to the
        root; contin appends `greater_than_zero($i)`, under an optional
        `not`. Note what the contin arm does when `greater_than_zero` is an
        IGNORED op: nothing at all is appended and `tr` is returned to the
        caller one child short -- classic has no `else` here either
        (build_knobs.cc:386-393), so a pair perm can come back holding one
        literal instead of two, or (both args ignored) a bare childless
        junctor.

        `argument(arg.idx)` is an explicit copy, matching C++'s `argument
        cpy_arg(arg)`: `negate()` mutates in place, and `tree.append_child`'s
        plain-data path stores the caller's object rather than copying it
        (CARRY-FORWARD.md 2.4), so passing `arg` itself would let the knob
        alias -- and then renegate -- the caller's argument. The contin arm
        copies for the same reason, where C++ gets its copy for free from
        `append_child(pos, const T&)`.

        Both copies are DEFENSIVE, and measured to be so: aliasing `arg`
        instead, in either arm, kills nothing across the whole corpus. The
        only caller is `sample_logical_perms`' pair loop, which builds a
        fresh `argument` per iteration and hands each to exactly one
        `insert_typed_arg` call, so nothing observable is left aliased
        today. Kept because C++ copies by value at both sites and because
        CARRY-FORWARD.md 2.4's rule ("appending some_node.data to another
        tree is always a bug unless copied") has no way to notice a new
        caller that does reuse one.
        """
        if arg_type is type_node.boolean_type:
            cpy_arg = argument(arg.idx)
            if negate:
                cpy_arg.negate()
            tr.append_child(tr.begin(), cpy_arg)
        elif self.permitted_op(builtin.greater_than_zero) and (arg_type is type_node.contin_type):
            root = tr.begin()
            if negate:  # insert not
                root = tr.append_child(root, builtin.logical_not)
            gt = tr.append_child(root, builtin.greater_than_zero)
            tr.append_child(gt, argument(arg.idx))

    def sample_logical_perms(self, it: tree_node) -> list[combo_tree]:
        """C++: moses/moses/representation/build_knobs.cc:411 build_knobs::sample_logical_perms

        The `perms` out-parameter C++ appends to becomes this function's
        return value; classic's only caller passes a freshly-declared empty
        vector (build_knobs.cc:545).
        """
        # An argument can be a subtree if it's boolean.
        # If its a contin, then wrap it with "greater_than_zero".
        permutations: list[combo_tree] = []
        for idx in range(1, self._arity + 1):
            arg = argument(idx)
            if self.permitted_op(arg):
                arg_type = self._signature.itypes[idx - 1]
                if arg_type is type_node.boolean_type:
                    permutations.append(combo_tree(arg))
                elif self.permitted_op(builtin.greater_than_zero) and (
                    arg_type is type_node.contin_type
                ):
                    v = combo_tree(builtin.greater_than_zero)
                    v.append_child(v.begin(), argument(idx))
                    permutations.append(v)

        # "Negative one correspnds to no pairs." (build_knobs.cc:435)
        #
        # `<= -1.0`, not `== -1.0`. Written the C++ way even though the two
        # are EQUIVALENT IN PYTHON -- proven by mutation: for any ratio
        # below -1 the `else` branch below computes a NEGATIVE `n_pairs`,
        # and `range(negative)` is empty, so no pair is built either way.
        # In C++ they are not equivalent at all: `static_cast<size_t>` of a
        # negative `floor()` wraps (build_knobs.cc:481's own "Avoid
        # accidental rounding to MAX_UINT" comment), so there this early
        # return is what stops `dorepeat(n_pairs)` from running ~2**64
        # times. Faithful spelling kept; the equivalence is recorded in
        # `test_pair_count_follows_classic_n_pairs_formula`'s docstring
        # rather than papered over with a test that cannot discriminate.
        if self._perm_ratio <= -1.0:
            return permutations

        # "Also create n random pairs op($i $j) out of the total number
        # 2 * choose(n,2) == n * (n-1) of possible pairs."
        # (build_knobs.cc:439-441.)
        permitted_perms: list[tuple[int, int]] = []
        for a in range(self._arity):
            if self.permitted_op(argument(a + 1)):
                for b in range(self._arity):
                    if self.permitted_op(argument(b + 1)) and a != b:
                        permitted_perms.append((a, b))

        max_pairs = len(permitted_perms)
        if max_pairs == 0:
            return permutations

        ps = len(permutations)
        if 0.0 < self._perm_ratio:
            n_pairs = int(math.floor(ps + self._perm_ratio * (max_pairs - ps)))
        else:
            n_pairs = int(math.floor((1.0 + self._perm_ratio) * ps))
            # C++ guards against `floor` of a negative product wrapping when
            # cast to `size_t` ("Avoid accidental rounding to MAX_UINT",
            # build_knobs.cc:481). Python has no such wrap, but the guard is
            # still load-bearing as a plain `ps < n_pairs -> 0` clamp for
            # ratios ABOVE 0 that reach this branch -- none do, since the
            # branch is `_perm_ratio <= 0.0` -- and for the negative ratios
            # that do, `(1 + r) * ps <= ps` already, so it is a no-op in
            # every reachable case. Kept verbatim rather than "simplified"
            # away, since it is the only thing standing between a future
            # branch-condition change and a silent wrap in C++.
            if ps < n_pairs:
                n_pairs = 0

        # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:487):
        # classic's `lazy_random_selector randpair(max_pairs)` uses the
        # default `randGen()` singleton, so `n_pairs` draws are consumed from
        # the process-wide RNG per junctor visited. This port does the same,
        # via the same ported selector, so the draw SEQUENCE matches -- but
        # that also means knob building is RNG-coupled to everything else
        # drawing from `randGen()`. Verified byte-for-byte against classic's
        # own `-l DEBUG` "Created prototype:" line for `-Hpa -k2 -r1`: all
        # four junctors get their two pairs in the same order classic
        # records (`or(!$1 $2)`/`or($1 $2)` under the first `and`,
        # `and($1 $2)`/`and(!$1 $2)` under the nested `or`, and so on).
        randpair = lazy_random_selector(max_pairs)
        for _ in range(n_pairs):
            i = randpair()
            a, b = permitted_perms[i]
            arg_a = argument(a + 1)
            arg_b = argument(b + 1)
            type_a = self._signature.itypes[a]
            type_b = self._signature.itypes[b]

            perm = combo_tree(swap_and_or(it.data))
            if b < a:
                self.insert_typed_arg(perm, type_b, arg_b)
                self.insert_typed_arg(perm, type_a, arg_a)
            else:
                # "As above, but negate arg_a first." (build_knobs.cc:510.)
                self.insert_typed_arg(perm, type_a, arg_a, True)
                self.insert_typed_arg(perm, type_b, arg_b)

            permutations.append(perm)

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
        """C++: moses/moses/representation/build_knobs.cc:602 build_knobs::build_logical

        Classic's `id::builtin flip = id::null_vertex;` sentinel is spelled
        here as an early `return`, with one exception kept literal: the
        `logical_not` arm, which asserts and THEN falls through to the same
        "not canonical, nothing to do" return classic reaches by leaving
        `flip` at its sentinel.

        UNREACHABLE, MEASURED (not merely argued): making that whole arm
        raise on entry leaves the entire mixed-predicate corpus green -- 40
        representation cases, 280 reduct oracle rows, three CLI oracles --
        so `build_logical` is never once called with a `logical_not` at its
        root. The reason: `logical_canonize` wraps a root predicate
        (negated or not) in `or(and(...))`, and a `not(0<(...))` CHILD is
        caught by the `is_predicate` branch below, never by the recursive
        call. The assertion is carried for the same reason classic carries
        it -- it is the only thing that would catch a caller handing this
        function a non-normal-form tree -- and dropping it is therefore a
        surviving mutant by construction, not for want of a test.
        """
        if vertex_equal(it.data, builtin.logical_not):
            # We allow not's, but only if they preceed a predicate. This
            # is similar to the case with arguments: not's are allowed
            # only if they immediately preceed a boolean-typed argument.
            if not is_predicate(it.first_child):
                raise AssertionError(
                    "ERROR: the tree is supposed to be in normal form; "
                    "and thus must not contain logical_not nodes."
                )
            return
        if vertex_equal(it.data, builtin.logical_and):
            flip = builtin.logical_or
        elif vertex_equal(it.data, builtin.logical_or):
            flip = builtin.logical_and
        else:
            # Hmm .. must not have been canonical. Nothing to do.
            return

        self.add_logical_knobs(subtree, it)
        current = it.first_child
        while current is not None:
            # Cache the current sibling before mutating the tree so the
            # outer traversal stays on the original child list.
            next_sibling = current.next_sibling
            if isinstance(current.data, argument):
                wrapper = self._exemplar.insert_above(current, flip)
                self.add_logical_knobs(subtree, wrapper, False)
            elif is_predicate(current):
                wrapper = self._exemplar.insert_above(current, flip)
                self.add_logical_knobs(subtree, wrapper, False)

                # At this time, we assume that the only predicate is
                # "greater_than_zero", and it has a single arg, which
                # is either contin or an argument, or any function
                # returning contin ... So, go and insert contin knobs
                # into that expression.
                pit = current
                if vertex_equal(pit.data, builtin.logical_not):  # skip over the not.
                    pit = pit.first_child
                cit = pit.first_child  # get the arg of predicate.

                # The BRANCH is reached constantly (forcing this condition
                # true fails 24 tests), but the condition itself is never
                # satisfied by any tree the corpus can build: combo's own
                # type checker has already rejected a predicate over a
                # non-contin term long before knob building. Dropping the
                # assertion is therefore a surviving mutant by
                # construction. Classic asserts here for the same reason.
                if not (is_argument(cit.data) or is_contin_expr(cit.data)):
                    raise AssertionError("Error: predicate term must be made of contin")

                # contin_canonize() creates a big pre-knob expr at @it.
                # build_contin() does the actual knob insertion into
                # the field set.
                self.contin_canonize(cit)
                # `pit.first_child` is deliberately RE-READ here, not
                # reused from `cit` above: `contin_canonize` inserts a new
                # root above its argument, so `cit` is no longer the
                # predicate's child by the time this runs. Classic writes
                # the same re-read as `build_contin(pit.begin())`.
                self.build_contin(pit.first_child)
            elif vertex_equal(current.data, builtin.null_vertex):
                # KNOWN-UNEXERCISED (as a mutant, not as dead code): this
                # `break` faithfully mirrors classic's `else if (*cur ==
                # id::null_vertex) break;` (build_knobs.cc:662-663), and the
                # branch itself IS reached by the current corpus (forcing an
                # exception here fails several tests). But replacing `break`
                # with a plain no-op (falling through to `current =
                # next_sibling` and continuing the loop) leaves the full
                # suite passing unchanged -- verified by mutation. Every
                # sibling list this corpus decorates either has no real
                # content after its null_vertex sentinel or does not
                # currently exercise one, so `break` vs "keep going" is not
                # currently distinguishable here, even though the branch
                # itself is.
                break
            else:
                self.build_logical(subtree, current)
            current = next_sibling

        flipped = self._exemplar.append_child(it, flip)
        self.add_logical_knobs(subtree, flipped)

    # ***********************************************************************
    # Continuous trees: linear combinations of arguments, sin/log/exp, and
    # divisions (S14 Task 2).

    def contin_canonize(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:710 build_knobs::contin_canonize"""
        if is_contin(it.data) and get_contin(it.data) == 0:
            it.data = builtin.plus
        if vertex_equal(it.data, builtin.div):
            if self._exemplar.number_of_children(it) != 2:
                raise ValueError("id::div built in must have exactly 2 children.")
            wrapper = self._exemplar.insert_above(it, builtin.plus)
            self._exemplar.append_child(wrapper, contin_t(0))
            self.canonize_div(it)
        elif vertex_equal(it.data, builtin.plus):
            # Move the constant child upwards.
            self.add_constant_child(it, 0)
            self._exemplar.insert_above(it, builtin.plus)
            self._exemplar.move_after(it, it.last_child)
            # Handle any divs: hoist every `div` child of `it` out to become
            # `it`'s own sibling (under the new outer plus), canonizing each.
            #
            # C++'s own loop is `for (sib_it div = partition(...); div !=
            # it.end();) canonize_div(move_after(it, pre_it(div++)))` --
            # `div++` (post-increment) captures div's CURRENT node for the
            # move, then reassigns `div` to that node's ORIGINAL
            # `next_sibling` (computed before the move), all as a single
            # expression evaluated before `move_after` runs. `next_div`
            # below is that same pre-move capture.
            div_node = self._exemplar.partition(
                it, lambda value: not vertex_equal(value, builtin.div)
            )
            while div_node is not None:
                next_div = div_node.next_sibling
                moved = self._exemplar.move_after(it, div_node)
                self.canonize_div(moved)
                div_node = next_div

            # Handle the rest of the children.
            if self.permitted_op(builtin.div):
                wrapper = self._exemplar.insert_above(it, builtin.div)
                self._exemplar.append_child(wrapper, contin_t(1))
                self.canonize_div(it.parent)
            else:
                self.linear_canonize_times(it)
        else:
            # else if we are here, then it must be id::times (or another
            # bare contin expression root -- sin/log/exp/impulse/argument/a
            # nonzero contin constant that survived the zero-rewrite check
            # above; classic's own comment names only `times` but the code
            # itself does not check for it, so this is ported literally).
            wrapper = self._exemplar.insert_above(it, builtin.plus)
            self._exemplar.append_child(wrapper, contin_t(0))
            if self.permitted_op(builtin.div):
                div_wrapper = self._exemplar.insert_above(it, builtin.div)
                self._exemplar.append_child(div_wrapper, contin_t(1))
                self.canonize_div(it.parent)
            else:
                self.linear_canonize_times(it)

    def build_contin(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:758 build_knobs::build_contin

        Loops over every node of the subtree rooted at ``it`` (itself
        included), in pre-order, creating a ``contin_knob``/field spec for
        each contin-valued vertex found. ``iter_pre_at`` is exactly C++'s own
        `pre_it end = it; end.skip_children(); for (++end; it != end; ++it)`
        idiom (see that method's own docstring in ``pymetta/util/tree.py``).
        """
        for node in self._exemplar.iter_pre_at(it):
            if is_contin(node.data):
                # This creates a knob at a particular location in the
                # exemplar inside the representation, and it creates a
                # field spec for the knob. This field spec is not yet a
                # part of any field set; that happens later, when the
                # representation is built.
                knob = contin_knob(
                    self._exemplar, node, self._step_size, self._expansion, self._depth
                )
                self._mapper.add_contin_knob(knob)

    def canonize_div(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:774 build_knobs::canonize_div"""
        self.linear_canonize_times(it.first_child)
        self.linear_canonize(it.last_child)

    def add_constant_child(self, it: tree_node, v: contin_t) -> None:
        """C++: moses/moses/representation/build_knobs.cc:779 build_knobs::add_constant_child"""
        sib = next(
            (child for child in self._exemplar.children(it) if is_contin(child.data)),
            None,
        )
        if sib is None:
            self._exemplar.append_child(it, contin_t(v))
        else:
            _swap_tree_nodes(sib, it.last_child)

    def canonize_times(self, it: tree_node) -> tree_node:
        """C++: moses/moses/representation/build_knobs.cc:788 build_knobs::canonize_times

        Turn ``it`` into a binary ``times`` with a constant second arg,
        returning the (possibly new) ``times`` node.
        """
        is_times = vertex_equal(it.data, builtin.times)
        sib = (
            next(
                (child for child in self._exemplar.children(it) if is_contin(child.data)),
                None,
            )
            if is_times
            else None
        )
        if sib is None:
            it_times = it if is_times else self._exemplar.insert_above(it, builtin.times)
            self._exemplar.append_child(it_times, contin_t(1))
            return it_times
        elif self._exemplar.number_of_children(it) > 2:
            self._exemplar.insert_above(it, builtin.times)
            self._exemplar.move_after(it, sib)
            return it.parent
        else:
            _swap_tree_nodes(sib, it.last_child)
            return it

    def linear_canonize_times(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:808 build_knobs::linear_canonize_times"""
        self.linear_canonize(self.canonize_times(it).first_child)

    def linear_canonize(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:829 build_knobs::linear_canonize"""
        # If we're not appending to plus, then insert one.
        if not vertex_equal(it.data, builtin.plus):
            it = self._exemplar.insert_above(it, builtin.plus)
        self.add_constant_child(it, 0)
        # Add the basic elements and recurse, if necessary.
        self.rec_canonize(it)

    def rec_canonize(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:839 build_knobs::rec_canonize

        The recursive `!self._linear_contin` branch raises the polynomial
        degree of every existing additive term: for every non-constant
        child of a `plus`, its multiplier (introduced by `canonize_times`)
        becomes its own full linear combination rather than a bare scalar.
        Classic's own comment (build_knobs.cc:847-855) names this a real
        arity^degree blow-up hazard -- CLAUDE.md repeats the same warning.
        `deme_expander.py`'s CLI wiring defaults `linear_contin` to
        classic's real `--linear-regression` default, `False`
        (deme_expander.py's own DIVERGENCE note), which is exactly the
        branch below, not the constructor's own `True` default -- see this
        module's docstring.
        """
        if vertex_equal(it.data, builtin.plus):
            # Recurse on whatever's already present, and create a
            # multiplicand for it.
            current = it.first_child
            while current is not None:
                if not is_contin(current.data):
                    current = self.canonize_times(current)
                    if not self._linear_contin:
                        self.rec_canonize(current.first_child)
                        if not is_contin(current.last_child.data):
                            raise AssertionError("Sibling's last child isn't id::contin.")
                        self.rec_canonize(
                            self._exemplar.insert_above(current.last_child, builtin.plus)
                        )
                current = current.next_sibling

            # Add the basic elements: sin, log, exp, and any variables
            # ($1, ..., $n).
            if self.permitted_op(builtin.sin):
                self.append_linear_combination(self.mult_add(it, builtin.sin))
            if self.permitted_op(builtin.log):
                self.append_linear_combination(self.mult_add(it, builtin.log))
            if self.permitted_op(builtin.exp):
                self.append_linear_combination(self.mult_add(it, builtin.exp))
            self.append_linear_combination(it)
        elif (
            vertex_equal(it.data, builtin.sin)
            or vertex_equal(it.data, builtin.log)
            or vertex_equal(it.data, builtin.exp)
        ):
            # functions that take a single contin arg: canonize the arg.
            self.linear_canonize(it.first_child)
        elif vertex_equal(it.data, builtin.times) or vertex_equal(it.data, builtin.div):
            # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:880-903):
            # classic's own body here is `logger().warn("TODO: handle case
            # where it = id::times ...")` followed by `#ifdef LATER` ...
            # `#endif` dead code (per this project's CLAUDE.md: "Code under
            # #ifdef ... LATER is dead and known-inconsistent; don't take it
            # as the current design"). With `LATER` undefined, the compiled
            # behaviour of this branch is the warning log call ALONE -- no
            # tree mutation, no knobs. This is that same no-op, minus the
            # log line (this file has no logging surface anywhere else
            # either).
            pass
        elif vertex_equal(it.data, builtin.impulse):
            # functions that take a single boolean arg: canonize the arg.
            self.logical_canonize(it.first_child)
        else:
            if not is_argument(it.data):
                raise AssertionError(f"{it.data!r} is not a builtin, neither an argument")

    def append_linear_combination(self, it: tree_node) -> None:
        """C++: moses/moses/representation/build_knobs.cc:936 build_knobs::append_linear_combination

        Appends a linear combination of *all* arguments -- i.e. the term
        ``+(*(0 $1) *(0 $2) *(0 $3) ...)``. Later on, the zero constants
        become knobs. A contin-typed argument is inserted directly; a
        Boolean-typed one is run through `impulse` first. Does NOT append
        to a `times` (see classic's own comment, build_knobs.cc:928-934: it
        would raise the polynomial degree by quasi-accident).
        """
        if vertex_equal(it.data, builtin.times):
            return
        if not vertex_equal(it.data, builtin.plus):
            it = self._exemplar.append_child(it, builtin.plus)

        for idx in range(1, self._arity + 1):
            arg = argument(idx)
            if self.permitted_op(arg):
                itype = self._signature.itypes[idx - 1]
                if itype is type_node.contin_type:
                    self.mult_add(it, argument(idx))
                elif itype is type_node.boolean_type:
                    if self.permitted_op(builtin.impulse):
                        imp = self.mult_add(it, builtin.impulse)
                        self._exemplar.append_child(imp, argument(idx))
                else:
                    raise AssertionError(
                        "Error: When building contin expressions, got "
                        "unsupported argument type."
                    )

    def mult_add(self, it: tree_node, v: vertex) -> tree_node:
        """C++: moses/moses/representation/build_knobs.cc:968 build_knobs::mult_add

        Appends ``*(0 v)`` as a child of ``it`` and returns the iterator
        pointing at (the freshly-appended copy of) ``v``.
        """
        const_node = self._exemplar.append_child(it, contin_t(0))
        times_node = self._exemplar.insert_above(const_node, builtin.times)
        return self._exemplar.append_child(times_node, v)


def _swap_tree_nodes(one: tree_node, two: tree_node) -> None:
    """C++: opencog/util/tree.h:2076 tree::swap(iterator, iterator)

    Physically exchanges two nodes' POSITIONS in the tree (parent/prev/next
    links only, never ``.data``) -- every node strictly between them in
    sibling order is left untouched, unlike "move one to the other's slot"
    (which would shift everything in between by one position instead).
    `add_constant_child`/`canonize_times` are this port's only two callers,
    exactly as in classic, and both always pass two DIFFERENT children of
    the SAME parent (never the tree's own root), so only that shape is
    exercised here.

    Implemented directly on the public `tree_node` link attributes
    (`first_child`/`last_child`/`prev_sibling`/`next_sibling`) rather than
    added as a new method on `pymetta/util/tree.py`'s `tree` class: this
    module already reads/writes those same attributes directly elsewhere
    (e.g. `build_logical`'s own `current.next_sibling` walk), and
    `pymetta/util/tree.py` is outside this task's owned-file scope (this
    module's own docstring / the task brief's file ownership list).
    """
    if one is two:
        return
    if one.next_sibling is two:
        _swap_adjacent_tree_nodes(one)
        return
    if two.next_sibling is one:
        _swap_adjacent_tree_nodes(two)
        return

    nxt1, nxt2 = one.next_sibling, two.next_sibling
    pre1, pre2 = one.prev_sibling, two.prev_sibling
    par1, par2 = one.parent, two.parent

    one.parent = par2
    one.next_sibling = nxt2
    if nxt2 is not None:
        nxt2.prev_sibling = one
    else:
        par2.last_child = one
    one.prev_sibling = pre2
    if pre2 is not None:
        pre2.next_sibling = one
    else:
        par2.first_child = one

    two.parent = par1
    two.next_sibling = nxt1
    if nxt1 is not None:
        nxt1.prev_sibling = two
    else:
        par1.last_child = two
    two.prev_sibling = pre1
    if pre1 is not None:
        pre1.next_sibling = two
    else:
        par1.first_child = two


def _swap_adjacent_tree_nodes(node: tree_node) -> None:
    """C++: opencog/util/tree.h:2055 tree::swap(sibling_iterator)

    Swaps ``node`` with its own ``next_sibling`` -- the adjacent-pair
    specialization `_swap_tree_nodes` delegates to (matching classic's own
    `if (one.node->next_sibling == two.node) swap(one);` dispatch). A no-op
    if ``node`` has no next sibling, matching classic's own ``if (nxt) {
    ... }`` guard.
    """
    nxt = node.next_sibling
    if nxt is None:
        return
    if node.prev_sibling is not None:
        node.prev_sibling.next_sibling = nxt
    else:
        node.parent.first_child = nxt
    nxt.prev_sibling = node.prev_sibling
    nxtnxt = nxt.next_sibling
    if nxtnxt is not None:
        nxtnxt.prev_sibling = node
    else:
        node.parent.last_child = node
    nxt.next_sibling = node
    node.prev_sibling = nxt
    node.next_sibling = nxtnxt
