"""Boolean representation builder.

This module ports only classic ``build_knobs``' pure-Boolean path. Continuous
predicates and action expressions remain separate allowed-domain slices; every
excluded domain is intentionally absent here.

Random logical PAIR perms (``build_knobs.cc:435-518``) used to be absent too,
with ``perm_ratio`` hard-restricted to ``-1`` (classic's "no pairs"
early-return mode). That was a real, measured behavioural divergence, not a
cosmetic one: classic's own default is ``perm_ratio = 0.0``
(``build_knobs.h:56``, ``representation.h:62``, ``problem-params.cc:299``),
which generates pairs, and without them the ``pa`` (even-parity) demo problem
is UNSOLVABLE by the port at any budget while classic solves it in a single
deme expansion. See ``.superpowers/sdd/2026-09-08-pymetta-s13-cli/
parity-divergence-report.md``. Pair sampling is now ported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
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
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import logical_subtree_knob
from pymetta.util.lazy_random_selector import lazy_random_selector
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
# Pair sampling (build_knobs.cc:435-518) IS ported, including its
# `lazy_random_selector` draws, so `perm_ratio` accepts classic's full range.
# What is still omitted from `sample_logical_perms`/`insert_typed_arg` is only
# their contin branches (build_knobs.cc:386-393,425-430), unreachable here
# because `_validate_request` rejects non-Boolean inputs outright.
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
        perm_ratio: float = 0.0,
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
        # KNOWN-UNEXERCISED: with every input forced Boolean (this class'
        # scope), `boolean_arity(signature) == type_tree_arity(signature) ==
        # arity`, so this reduces to `not (4 * arity > 0)`, i.e. `arity ==
        # 0`. But `add_logical_knobs` only ever calls `disc_probe` from a
        # `sample_logical_perms` permutation, and that generator is
        # `range(1, self._arity + 1)` -- empty at arity 0. So the one input
        # shape that makes `_skip_disc_probe` True is also the one shape in
        # which `disc_probe` is never called at all: its `if
        # self._skip_disc_probe: return True` branch is unreachable given
        # this scope, not merely untested by the current corpus -- verified
        # by making that branch raise instead and confirming the full suite
        # still passes.
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
        arg: argument,
        negate: bool = False,
    ) -> None:
        """C++: moses/moses/representation/build_knobs.cc:379 build_knobs::insert_typed_arg

        Classic takes the argument's `type_tree_sib_it` and branches on it:
        Boolean appends a (possibly negated) copy of the argument to the
        root, contin appends `greater_than_zero($i)` under an optional
        `not`. `_validate_request` rejects every non-Boolean input in this
        slice, so only the Boolean branch is reachable and the `arg_type`
        parameter carries no information -- it is dropped rather than
        threaded through as an always-`boolean_type` constant. The contin
        branch belongs with the rest of the continuous slice.

        `argument(arg.idx)` is an explicit copy, matching C++'s `argument
        cpy_arg(arg)`: `negate()` mutates in place, and `tree.append_child`'s
        plain-data path stores the caller's object rather than copying it
        (CARRY-FORWARD.md 2.4), so passing `arg` itself would let the knob
        alias -- and then renegate -- the caller's argument.
        """
        cpy_arg = argument(arg.idx)
        if negate:
            cpy_arg.negate()
        tr.append_child(tr.begin(), cpy_arg)

    def sample_logical_perms(self, it: tree_node) -> list[combo_tree]:
        """C++: moses/moses/representation/build_knobs.cc:411 build_knobs::sample_logical_perms

        The `perms` out-parameter C++ appends to becomes this function's
        return value; classic's only caller passes a freshly-declared empty
        vector (build_knobs.cc:545).
        """
        permutations: list[combo_tree] = []
        for idx in range(1, self._arity + 1):
            arg = argument(idx)
            if self.permitted_op(arg):
                # Boolean branch only: the contin branch
                # (build_knobs.cc:425-430) is unreachable here, see
                # `insert_typed_arg`.
                permutations.append(combo_tree(arg))

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

            perm = combo_tree(swap_and_or(it.data))
            if b < a:
                self.insert_typed_arg(perm, arg_b)
                self.insert_typed_arg(perm, arg_a)
            else:
                # "As above, but negate arg_a first." (build_knobs.cc:510.)
                self.insert_typed_arg(perm, arg_a, True)
                self.insert_typed_arg(perm, arg_b)

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
