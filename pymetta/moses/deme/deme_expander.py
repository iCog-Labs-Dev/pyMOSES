"""C++: moses/moses/deme/deme_expander.{h,cc}

``deme_expander`` -- the deme factory. Given an exemplar ``combo_tree``,
builds a ``representation`` (knobs over the exemplar) and an empty
``instance_set`` ("deme") sized to that representation's ``field_set``, then
drives an injected ``optimizer_base`` (S9's ``hill_climbing``, at the test
level -- see below) to fill and score that deme.

**Single-deme path is still the default**, but on-the-fly feature
selection (``deme_parameters.fstor``, ``--enable-fs``) is now REAL:

* **On-the-fly feature selection** (``deme_parameters.fstor``, wired to
  ``deme/feature_selector.{h,cc}``, ported at
  ``pymetta.moses.deme.feature_selector``) -- ``create_demeIDs`` and
  ``create_representations`` both honour a real ``feature_selector``:
  the exemplar is scored, a feature subset selected (via
  ``feature_selector.__call__``), and a ``representation`` built with
  ``ignore_ops`` set to the COMPLEMENT of the selected features -- see
  ``create_representations``'s own docstring. Since the only ported
  algorithm, ``"simple"`` (``pymetta.feature_selection.algo.simple``),
  ALWAYS returns exactly one feature set regardless of
  ``feature_selector_parameters.n_demes`` (that algorithm's own
  contract -- see its module docstring), ``create_demeIDs`` can still
  build MULTIPLE ``demeID_t``s when ``fstor.params.n_demes > 1``, but
  ``create_representations``'s ``for selected_feats in
  pop_of_selected_feats`` loop only ever runs ONCE in that case (the
  population it iterates has length 1) -- extra demeIDs beyond index 0
  are simply never consumed by a representation. This is classic's OWN
  behaviour with ``algorithm="simple"``, not a gap this port
  introduces (measured: cc:205-268's loop is driven by
  ``pop_of_selected_feats.size()``, never by ``_demeIDs.size()``).

  ``festor.params.prune_xmplr`` (feature_selector_parameters's own
  struct default is INDETERMINATE -- see ``feature_selector.py``'s
  module docstring; this port and every real CLI caller pin it to
  ``False``) selects ``prune_xmplr`` (deme_expander.cc:308-332,
  erasing argument leaves from the exemplar under a PRE-ORDER walk,
  with a ``type_to_exemplar`` fallback if that empties the tree). It
  is now REAL -- ported below, matched case for case against the C++
  itself (tools/cpp_probe/prune_xmplr_probe.cc). Note the arm's
  asymmetry, which is C++'s own: the pruning arm does NOT union the
  exemplar's features back into the selected set, so they land in
  ``ignore_ops`` too.

  ``optimize_demes`` drives a real ``fstor`` all the way through the
  search loop. Its OWN ``if (_params.fstor)`` arm
  (deme_expander.cc:432-450, plus the post-loop reset at cc:499-506) is
  the one piece that is documented-and-skipped rather than executed:
  it calls ``behave_cscore.ignore_cols()``/``best_possible_score()``,
  which do not exist in ``pymetta.moses.scoring`` -- see
  ``optimize_demes``'s own DIVERGENCE for the three upstream edits that
  would close it, and for the measurement showing the omission changes
  no output this port can produce.

* **Subsample-deme filtering**
  (``subsample_deme_filter_parameters.n_subsample_demes``, S10 Task 1's
  ``metapop_params.py``) -- would subdivide EACH representation's single
  deme into ``n_subsample_demes`` "ss-demes" over disjoint row/timestamp
  subsets (the header's own comment on ``_demes``: "a deme can be further
  subdivided into multiple demes using a subsampled dataset"). Gated in
  ``create_demes`` and ``optimize_demes``: ``n_subsample_demes > 1`` raises.
  The plan's own scope table lists this as one of the whole S10 slice's
  default-off P2 subsystems, alongside boosting/domination/diversity.

Both ``fstor=None`` and ``n_subsample_demes=0`` are still the defaults, so
the fully-ported DEFAULT path is unchanged: exactly one representation,
exactly one deme, per ``create_demes`` call.

``fs_to_names`` (deme_expander.h:115, deme_expander.cc:48) IS ported now.
It used to be listed here as deliberately dropped, on the grounds that
every call site is inside a ``logger()`` call -- true of cc:76/:78 (both
in ``log_selected_feature_sets``, still not ported) but not of cc:227,
which is in the ``prune_xmplr`` arm of ``create_representations``, live
as of this change. Its one behavioural content is ORDER: a C++
``std::set<arity_t>`` iterates ASCENDING and a Python ``set`` does not,
so it sorts.

STILL NOT ported: ``log_selected_feature_sets`` (deme_expander.h:121,
deme_expander.cc:57), which is purely logging (its only effect is feeding
``logger().info()``/``ostream_container``; nothing observes a return
value), matching the "no logging surface in this port" convention stated
throughout this project. Also
NOT ported: the ``#if THIS_IS_DISABLED_UNTIL_ABOVE_IS_FIXED`` block inside
``optimize_demes`` (deme_expander.cc:460-474) -- dead code by construction
in C++ itself (the guarding macro is never defined, matching CLAUDE.md's
treatment of other never-taken ``#ifdef`` branches), and every
``logger()``-only line (deme_expander.cc:276-283,413,425-430,449-455) --
no logging surface in this port, see ``hill_climbing.py``'s module
docstring for the same omission elsewhere.

**Injection seam** (deme_expander.h:44,83): ``deme_expander`` takes and
stores ``optimizer_base& _optimize``; it does NOT construct an optimizer.
The concrete optimizer is injected by the CLI layer in classic (not yet
ported -- S13) -- this port preserves exactly that seam: ``opt`` is any
object shaped like ``optimizer_base`` (S9's ``hill_climbing``, wired in
directly at the test level, matching this task's own instruction).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Sequence

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import tree_to_stream_str
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_argument
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.deme.feature_selector import get_argument_abs_idx_from_zero_set
from pymetta.moses.metapopulation.metapop_params import subsample_deme_filter_parameters
from pymetta.moses.moses.optimization.optimization import optimizer_base
from pymetta.moses.moses.types import demeID_t
from pymetta.moses.representation.instance_scorer import complexity_based_scorer
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.representation import representation
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.util.logger import logger
from pymetta.util.rng import randGen

Simplifier = Callable[[combo_tree], None]

__all__ = ["deme_expander"]


def _type_to_exemplar(otn: type_node) -> combo_tree:
    """C++: moses/moses/representation/representation.cc:75 type_to_exemplar

    ``prune_xmplr``'s empty-tree seed (deme_expander.cc:327-330). Declared
    ``moses/moses/representation/representation.h:42``, defined in
    ``representation.cc``; ``pymetta/moses/representation/representation.py``
    does NOT port it (grepped 2026-09-10), and the only Python counterpart
    that exists anywhere in the tree is ``pymetta.moses.main.demo_problems.
    _type_to_exemplar`` -- a private helper that file added for its own
    ``pms.exemplars`` defaults, for the same "no port exists" reason it
    states in its own module docstring.

    Duplicated HERE rather than imported from there, for two independent
    reasons. (1) LAYERING: ``moses/main`` is the CLI layer and sits ABOVE
    ``moses/deme``; ``demo_problems`` imports ``moses_main``, which reaches
    ``metapopulation`` and this very module, so importing it back would
    close a real import cycle. (2) OWNERSHIP: ``pymetta/moses/main/`` is
    out of this task's editable scope, so the shared definition cannot be
    hoisted out of it either. FOLLOW-UP for whoever owns
    ``pymetta/moses/representation/representation.py``: port
    ``type_to_exemplar`` there (its actual C++ home) and have both this
    module and ``demo_problems`` import it, deleting both copies.

    Only the two branches this port's knob building can reach are real:
    ``boolean_type`` -> ``combo_tree(logical_and)`` and ``contin_type`` ->
    ``combo_tree(plus)``. C++'s ``enum_type`` branch (a ``cond`` root with
    an ``enum_t::get_random_enum()`` child) is not ported -- the enum
    domain is excluded project-wide -- and its ``ill_formed_type``/
    ``default`` branches are ``OC_ASSERT(false, ...)``, i.e. aborts, ported
    as a raise.
    """
    if otn == type_node.boolean_type:
        return combo_tree(builtin.logical_and)
    if otn == type_node.contin_type:
        return combo_tree(builtin.plus)
    raise NotImplementedError(
        f"type_to_exemplar: type {otn!r} not supported by this port "
        "(representation.cc:81's enum_type branch needs enum_t::"
        "get_random_enum(); cc:87/:93 are OC_ASSERT(false) aborts)"
    )


class deme_expander:
    """C++: moses/moses/deme/deme_expander.h:38 deme_expander"""

    def __init__(
        self,
        type_signature: TypeSignature,
        si_ca: Simplifier,
        si_kb: Simplifier,
        sc: behave_cscore,
        opt: optimizer_base,
        pa: deme_parameters | None = None,
        fp: subsample_deme_filter_parameters | None = None,
    ) -> None:
        """C++: moses/moses/deme/deme_expander.cc:35 deme_expander::deme_expander

        ``pa``/``fp`` default to a fresh ``deme_parameters()``/
        ``subsample_deme_filter_parameters()`` per call -- matching C++'s
        own default arguments (deme_expander.h:45-46) in SHAPE, but not in
        the hazard they carry: C++ binds its ``const deme_parameters&
        _params``/``const subsample_deme_filter_parameters& _filter_params``
        members from constructor PARAMETERS that may themselves be bound to
        a caller-omitted default-argument temporary, and a reference member
        initialized through an intermediate reference parameter does not
        extend that temporary's lifetime -- see ``deme_params.py``'s module
        docstring for the full reasoning (observed by inspection, not
        measured) and why this port's own default is actually safe where
        C++'s may not be.

        ``random_shuffle_gen`` (deme_expander.cc:45, ``[&](ptrdiff_t i) {
        return randGen().randint(i); }``) is ``randGen().randint_n(i)``
        here -- ``rng.py``'s own ``randint_n`` is the direct port of
        ``opencog::RandGen::randint(int)`` this lambda calls. Stored for
        fidelity to the constructor body; its only consumers,
        ``subsample_by_time``/``subsample_by_row``'s ``boost::
        random_shuffle`` calls, are inside the subsample-deme-filtering
        gate (see module docstring) and are therefore never actually
        invoked by this port's own methods below.
        """
        self._optimize = opt
        self._type_sig = type_signature
        self.simplify_candidate = si_ca
        self.simplify_knob_building = si_kb
        self._cscorer = sc
        self._params = pa if pa is not None else deme_parameters()
        self._filter_params = fp if fp is not None else subsample_deme_filter_parameters()

        self.random_shuffle_gen: Callable[[int], int] = lambda i: randGen().randint_n(i)

        # Structures related to the current deme (deme_expander.h:75-76,83).
        self._reps: list[representation] = []
        self._demes: list[list[instance_set]] = []

        # Protected state (deme_expander.h:145,148).
        self._ignore_cols_seq: list[set[int]] = []
        self._demeIDs: list[demeID_t] = []

    # -------------------------------------------------- feature-set naming

    def fs_to_names(self, fs: Collection[int], ilabels: Sequence[str]) -> list[str]:
        """C++: moses/moses/deme/deme_expander.cc:48 deme_expander::fs_to_names

        "Takes a feature set, a header of the input features and return a
        vector of the names of the featire set." (deme_expander.h:111-113's
        own comment, typo included.)

        ORDER IS THE WHOLE CONTENT of this function. C++'s ``feature_set``
        is ``typedef std::set<arity_t> feature_set``
        (feature-selection/main/feature-selection.h:124), and
        ``for (arity_t i : fs)`` over a ``std::set`` walks it in ASCENDING
        key order. A Python ``set`` iterates in hash order, which for small
        non-negative ints happens to LOOK ascending most of the time and
        then silently is not -- so ``sorted(fs)`` is the faithful
        transliteration, not a cosmetic tidy-up.

        All three C++ call sites are ported: cc:76 and cc:78 inside
        ``log_selected_feature_sets`` (below), and cc:227 in the
        ``prune_xmplr`` arm of ``create_representations``.

        Raises ``IndexError`` for a feature index at or beyond
        ``len(ilabels)``, where C++'s ``ilabels[i]`` -- an UNCHECKED
        ``std::vector::operator[]`` -- is undefined behaviour instead.
        **That case is REACHABLE, and it is where classic dies.** With
        ``--fs-seed=xmplr`` (``xmplr_as_feature``),
        ``feature_selector.build_fs_ctable`` appends a synthetic
        ``__exemplar_feature__`` column at index ``_ctable.get_arity()``,
        feature selection can pick it, and nothing removes it again --
        ``feature_selector::remove_useless_features``
        (feature_selector.cc:295), the function written to strip it, has
        ZERO call sites in the whole classic tree. ``ilabels`` here is
        ``festor._ctable.get_input_labels()``, the ORIGINAL columns, so
        that index is one past the end. Measured under gdb on this
        machine, 2026-09-10: ``moses -i bool_a5_maj.csv -u y -m1000 -W1
        --enable-fs=1 --fs-target-size=2 -r1 --fs-seed=xmplr`` SIGSEGVs
        (rc 139) with the fault inside ``std::char_traits<char>::copy``
        under ``fs_to_names`` (deme_expander.cc:53), called from
        ``log_selected_feature_sets`` (cc:78) from
        ``create_representations`` (cc:200). Reproducing the UB is not
        possible and producing an answer classic cannot would be worse,
        so this raises -- the same rule ``interpreter.py`` applies to an
        argument index beyond the table's arity (CARRY-FORWARD.md 3).
        """
        fs_names: list[str] = []
        for i in sorted(fs):
            if not 0 <= i < len(ilabels):
                raise IndexError(
                    f"deme_expander.fs_to_names: feature index {i} is outside "
                    f"the {len(ilabels)} input column(s) this table has. C++'s "
                    "`ilabels[i]` (deme_expander.cc:53) is an unchecked "
                    "std::vector::operator[] and SEGVs here; the reachable "
                    "cause is --fs-seed=xmplr, whose synthetic "
                    "__exemplar_feature__ column sits at index "
                    f"{len(ilabels)} and is never removed again "
                    "(feature_selector::remove_useless_features has zero "
                    "call sites in classic). See this method's docstring."
                )
            fs_names.append(ilabels[i])
        return fs_names

    def log_selected_feature_sets(
        self,
        sf_pop: list,
        xmplr_features: set[int],
        ilabels: Sequence[str],
    ) -> None:
        """C++: moses/moses/deme/deme_expander.cc:57 deme_expander::log_selected_feature_sets

        NOT logging-only, which is why it is ported rather than dropped
        with the other ``logger()``-only helpers. Its first statement
        after the size read is

            OC_ASSERT(sfps == _demeIDs.size(), "The code is probably not robust enough");

        (cc:63) and ``OC_ASSERT`` is ACTIVE in release builds --
        ``IGNORE_OC_ASSERT`` is defined nowhere in the MOSES or CogUtil
        trees -- so it throws ``opencog::AssertionException``
        ($COGUTIL/opencog/util/oc_assert.cc:46), which nothing catches.

        It fires for a real command line. ``create_demeIDs`` builds
        ``fstor.params.n_demes`` ids (cc:152-154) while every ported
        selection algorithm returns a single feature set, capped again by
        ``select_top_feature_sets`` at ``min(n_demes, fss.size())``
        (feature_selector.cc:237) -- so ``--fs-demes=N`` with ``N > 1``
        gives ``1 != N`` and classic ABORTS. Measured, 2026-09-10:
        ``moses -i bool_a5_maj.csv -u y -m1000 -W1 --enable-fs=1
        --fs-target-size=2 -r1 --fs-demes=2`` exits 134 with "The code is
        probably not robust enough (deme_expander.cc:63)", for
        ``--fs-algo=simple`` and ``--fs-algo=random`` alike. Until this
        function was ported, this port ran that command line happily and
        printed a result -- accepting input classic refuses, the mirror
        image of the over-gating this project usually guards against.

        Ported as ``RuntimeError``, the same shape ``create_demes``
        already gives cc:379-381's ``OC_ASSERT``.
        """
        # C++: `unsigned sfps = sf_pop.size(), sfi = 0;` (cc:60).
        sfps = len(sf_pop)
        sfi = 0
        if sfps != len(self._demeIDs):
            raise RuntimeError(
                "The code is probably not robust enough "
                "(OC_ASSERT, deme_expander.cc:63): feature selection "
                f"returned {sfps} feature set(s) but create_demeIDs built "
                f"{len(self._demeIDs)} deme id(s). --fs-demes > 1 always "
                "reaches this, because every selection algorithm returns "
                "exactly one feature set."
            )
        for _score, sf in sf_pop:
            logger().info(f"Breadth-first expansion for deme : {self._demeIDs[sfi]}")
            logger().info(f"Selected {len(sf)} features for representation building")
            xmplr_sf = set(sf) & xmplr_features
            new_sf = set(sf) - xmplr_features
            logger().info(
                f"Of these, {len(xmplr_sf)} are already in the exemplar, and {len(new_sf)} are new."
            )
            # C++ passes "," EXPLICITLY to both `ostream_container` calls
            # (cc:76,78), unlike cc:227's use of the default " ".
            logger().info(
                "Selected features which are in the exemplar: "
                + ",".join(self.fs_to_names(xmplr_sf, ilabels))
            )
            logger().info(
                "Selected features which are new: " + ",".join(self.fs_to_names(new_sf, ilabels))
            )
            sfi += 1  # C++: deme_expander.cc:79

    # ------------------------------------------------------------ subsample

    def subsample_by_time(self) -> list[set]:
        """C++: moses/moses/deme/deme_expander.cc:83 deme_expander::subsample_by_time

        "Subsample by time. Return a vector (of size n_ss_demes) of sets of
        timestamps to be discarded during subsampling." (deme_expander.h:
        87-89's own comment.) Gated: see module docstring. The disabled
        (``n_subsample_demes <= 1``) branch is the only one ported --
        ``ignore_timestamps_per_ss_deme.emplace_back();`` (cc:115), a
        single EMPTY set, matching classic's own "subsampling deme is
        disabled" comment (cc:114) exactly.
        """
        if self._filter_params.n_subsample_demes > 1:
            raise NotImplementedError(
                "deme_expander: subsample-by-time deme filtering is P2, not "
                "ported; leave subsample_deme_filter_parameters."
                "n_subsample_demes at its default (0)."
            )
        return [set()]

    def subsample_by_row(self) -> list[set]:
        """C++: moses/moses/deme/deme_expander.cc:120 deme_expander::subsample_by_row

        "Subsample by row. return a vector (of size n_ss_demes) of sets of
        indexes (of a corresponding uncompressed table) to be discarded
        during subsampling." (deme_expander.h:92-95's own comment.) Same
        gate and same disabled-branch-only porting as ``subsample_by_time``
        above (cc:141-143, ``ignore_row_idxs_per_ss_deme.emplace_back();``).
        """
        if self._filter_params.n_subsample_demes > 1:
            raise NotImplementedError(
                "deme_expander: subsample-by-row deme filtering is P2, not "
                "ported; leave subsample_deme_filter_parameters."
                "n_subsample_demes at its default (0)."
            )
        return [set()]

    # ------------------------------------------------------------ building

    def create_demeIDs(self, n_expansions: int) -> None:
        """C++: moses/moses/deme/deme_expander.cc:148 deme_expander::create_demeIDs

        ``fstor.params.n_demes`` (``feature_selector_parameters.n_demes``,
        struct default ``1``) selects the multi-ID branch when ``> 1`` --
        matching ``if (_params.fstor && _params.fstor->params.n_demes >
        1)`` (cc:152) exactly, now that ``_params.fstor`` is a real
        ``feature_selector`` (``pymetta.moses.deme.feature_selector``)
        rather than an always-``None``/always-rejected placeholder. See
        the module docstring for why extra IDs built here can outlive
        their use in ``create_representations`` when the selected
        algorithm (``"simple"``, the only one ported) returns a
        single-entry population regardless of ``n_demes``.
        """
        self._demeIDs = []
        if self._params.fstor is not None and self._params.fstor.params.n_demes > 1:
            for i in range(self._params.fstor.params.n_demes):
                self._demeIDs.append(demeID_t(n_expansions + 1, i))
        else:
            self._demeIDs.append(demeID_t(n_expansions + 1))

    def create_representations(self, exemplar: combo_tree) -> bool:
        """C++: moses/moses/deme/deme_expander.cc:159 deme_expander::create_representations

        Two arms, matching cc:196-269 (``if (_params.fstor)``) and
        cc:270-272 (``else``) exactly:

        * ``fstor is None`` (the default): ``ignore_ops_seq = [_params.
          ignore_ops]; xmplr_seq = [exemplar]`` -- always exactly one
          entry, unchanged from before this port had a real
          ``feature_selector``.
        * ``fstor is not None``: copies ``fstor`` (cc:199's own comment,
          "Copy, as any change in the parameters will not be
          remembered") -- cheaply: only ``.params`` is deep-copied, not
          ``._ctable`` (never mutated by any ``feature_selector``
          method, so sharing the reference is behaviourally identical
          and avoids copying a potentially large table) -- then calls
          it on ``exemplar`` to get ``pop_of_selected_feats``
          (``feature_set_pop``, at most one entry for the only ported
          algorithm, ``"simple"``). For each selected feature set (cc:
          205's own loop) one of TWO arms runs, then any
          ``enforce_features`` are sampled in and added (cc:234-238);
          the COMPLEMENT of the final feature set becomes
          ``ignore_cols``/``ignore_ops`` (cc:240-256, one ``argument(i
          + 1)`` per unselected column index ``i``) and the set itself
          becomes ``considered_args`` (unused beyond an omitted debug
          log -- see module docstring).

          The two arms are selected by ``festor.params.prune_xmplr``
          (see ``feature_selector.py``'s own module docstring for why
          this field's C++ default is indeterminate; the CLI's
          ``--fs-prune-exemplar`` defaults to ``0``,
          problem-params.cc:1663-1665, so the ``else`` arm is what
          every default ``--enable-fs`` run takes):

          * ``False`` (cc:230-236): the exemplar's OWN features are
            unioned back INTO the selected set, and the exemplar goes
            into ``xmplr_seq`` untouched. Its features therefore stay
            OUT of ``ignore_ops``.
          * ``True`` (cc:216-229): ``prune_xmplr`` (cc:308, ported
            below) strips the unselected arguments out of a copy of the
            exemplar and THAT goes into ``xmplr_seq`` -- and the union
            above is NOT performed, so the exemplar's own features do
            land in ``ignore_ops``. Do not "tidy" this asymmetry away;
            it is what ``--fs-prune-exemplar`` means. Classic's own
            comment on the branch (cc:210-213) admits it does not know
            why anyone would want it: "In particular, why would we ever
            want to prune the exemplar? I can't figure this out."

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:287-294): the
        # real ``representation(...)`` call also forwards ``_params.
        # perceptions``, ``_params.actions`` and ``_params.linear_contin``.
        # S8's ``representation`` (pymetta/moses/representation/
        # representation.py) is Boolean-only and accepts none of the
        # three -- its own module docstring already excludes the mutable
        # action/perceptions machinery and continuous domains outright.
        # ``deme_parameters`` still carries all three fields (see
        # deme_params.py) for struct-shape fidelity; at their DEFAULTS
        # (``perceptions=None``, ``actions=None``, ``linear_contin=True``)
        # simply never forwarding them is byte-identical to C++ forwarding
        # them (a Boolean signature can never construct a non-``None``
        # perceptions/actions value in the first place, and
        # ``linear_contin=True`` is C++'s own default too). GATED below
        # (S10 branch review, [Important 6]) for the NON-default case,
        # which this port genuinely cannot honour: raises rather than
        # silently ignoring a caller's ``perceptions``/``actions``/
        # ``linear_contin=False`` request. Only inert in practice because
        # ``build_knobs._validate_request`` already rejects any
        # non-Boolean signature two layers below -- this gate is what
        # turns that into a clear message AT THIS CALL SITE instead of a
        # generic type-signature error further down.

        C++'s own bookkeeping is push-then-pop-if-empty (cc:287-302,
        `_reps.push_back(new representation(...)); if (_reps.back().fields
        ().empty()) { _reps.pop_back(); ... }`); this port builds first and
        only appends when non-empty, which reaches the same net ``_reps``
        state (nothing else observes the momentarily-pushed-then-popped
        entry in C++ either) -- not a behavioural difference, just a
        clearer expression of it in a language without ``pop_back``-as-
        undo. The dropped ``logger().warn("The representation is empty,
        ...")`` (cc:300-301) is the same no-logging-surface omission
        documented elsewhere (see module docstring).
        """
        if self._params.perceptions is not None or self._params.actions is not None:
            raise NotImplementedError(
                "deme_expander.create_representations: perceptions/actions "
                "(deme_params.h:41-42,46-47) select the perception/action "
                "knob-building path (build_knobs.py's action_result_type "
                "branch -- documented stale legacy embodiment code in this "
                "project's CLAUDE.md); this port's representation() is "
                "Boolean-only and has no parameter for either. Construct "
                "deme_parameters with perceptions=None, actions=None (the "
                "defaults)."
            )
        # `linear_contin` is read at exactly one place in classic --
        # `if (!_linear_contin)` at build_knobs.cc:859, inside contin knob
        # building, where it selects `rec_canonize`'s non-linear expansion.
        #
        # This used to raise for any non-Boolean output type, on the stated
        # grounds that "this port has no contin knob-building at all". That
        # was true when written and became false when S14 Task 2 landed
        # `contin_canonize`/`build_contin`/`rec_canonize` (`f03ba17`), whose
        # knob counts match classic 18/18 including at `linear_contin=False`.
        # The guard then blocked the very path it was describing as missing,
        # and was the last thing preventing a contin regression run from
        # reaching the search. Removed 2026-09-09 after driving the CLI
        # against classic's binary on a real contin table.

        ignore_ops_seq: list[set] = []
        # C++: deme_expander.cc:178. Built only to be streamed by the
        # "Using arguments: " record at cc:281-282, and only in the
        # `fstor` branch -- so it stays EMPTY on the default path, which
        # is exactly what makes `if (!considered_args_seq.empty())`
        # (cc:280) the gate C++ wrote it as.
        considered_args_seq: list[set] = []
        xmplr_seq: list[combo_tree] = []

        if self._params.fstor is not None:
            copied_params = copy.deepcopy(self._params.fstor.params)
            festor = type(self._params.fstor)(self._params.fstor._ctable, copied_params)
            pop_of_selected_feats = festor(exemplar)
            xmplr_features = get_argument_abs_idx_from_zero_set(exemplar)
            # C++: deme_expander.cc:193. Read by cc:227's `fs_to_names`
            # call and by both of `log_selected_feature_sets`' own.
            ilabels = festor._ctable.get_input_labels()

            if festor.params.n_demes > 1:
                # C++: deme_expander.cc:195-198.
                logger().info(
                    "Breadth-first deme expansion (same exemplar, multiple "
                    f"feature sets): {festor.params.n_demes} demes"
                )

            # C++: deme_expander.cc:200. NOT a logging call despite the
            # name -- see that method's own docstring for the live
            # `OC_ASSERT` inside it and the two command lines that reach
            # it.
            self.log_selected_feature_sets(pop_of_selected_feats, xmplr_features, ilabels)

            # C++: deme_expander.cc:202-203.
            logger().debug(
                "Post-treatment: possibly enforce some features and prune exemplars for each deme"
            )

            # C++: deme_expander.cc:204, `unsigned sfi = 0;` -- the index into
            # `_demeIDs` the debug records below use, incremented at the end
            # of each iteration (cc:266).
            sfi = 0
            for selected_score, selected_feats in pop_of_selected_feats:
                del selected_score  # only used for the omitted debug log
                logger().debug(f"Deme {self._demeIDs[sfi]}:")  # C++: cc:210
                if festor.params.prune_xmplr:
                    # C++: deme_expander.cc:216-229.
                    if logger().is_debug_enabled():
                        if not xmplr_features:
                            # cc:219-221
                            logger().debug(
                                f"No feature to prune in the exemplar for deme {self._demeIDs[sfi]}"
                            )
                        else:
                            # cc:222-228. `ostream_container`'s default
                            # delimiter is a single space
                            # (opencog/util/iostreamContainer.h:64).
                            xmplr_nsf = xmplr_features - set(selected_feats)
                            names = " ".join(self.fs_to_names(xmplr_nsf, ilabels))
                            logger().debug(
                                "Prune the exemplar from non-selected features "
                                f"for deme {self._demeIDs[sfi]}: {names}"
                            )
                    # NOTE the asymmetry with the `else` arm below, and that
                    # it is C++'s, not this port's: the pruning arm does NOT
                    # union `xmplr_features` back into `selected_feats`, so
                    # the exemplar's own features stay OUT of the feature set
                    # and therefore land in `ignore_ops` a few lines down.
                    # That is the entire point of `--fs-prune-exemplar`, and
                    # it is why the two arms must not be collapsed into a
                    # shared "compute selected_feats, then branch".
                    xmplr_seq.append(self.prune_xmplr(exemplar, selected_feats))
                else:
                    # C++: deme_expander.cc:230-236.
                    if logger().is_debug_enabled():
                        logger().debug(
                            "Do not prune the exemplar from non-selected "
                            f"features for deme {self._demeIDs[sfi]}"
                        )
                    # Insert exemplar features as they are not pruned
                    selected_feats = set(selected_feats) | xmplr_features
                    xmplr_seq.append(exemplar)

                if festor.params.enforce_features:
                    enforced = festor.sample_enforced_features()
                    selected_feats = selected_feats | enforced

                arity = festor._ctable.get_arity()
                ignore_cols: set[int] = set()
                ignore_ops: set = set()
                considered_args: set = set()
                for i in range(arity):
                    arg = argument(i + 1)
                    if i not in selected_feats:
                        ignore_cols.add(i)
                        ignore_ops.add(arg)
                    else:
                        considered_args.add(arg)  # C++: cc:259-261
                self._ignore_cols_seq.append(ignore_cols)
                ignore_ops_seq.append(ignore_ops)
                considered_args_seq.append(considered_args)
                sfi += 1  # C++: deme_expander.cc:266
        else:  # no dynamic feature selection
            ignore_ops_seq.append(self._params.ignore_ops)
            xmplr_seq.append(exemplar)

        for i in range(len(xmplr_seq)):
            if logger().is_debug_enabled():
                # C++: deme_expander.cc:276-283. ONE guard around two
                # records; the second is gated again on
                # `!considered_args_seq.empty()`, i.e. emitted only on the
                # `fstor` path.
                logger().debug(
                    f"Attempt to build rep from exemplar for deme "
                    f"{self._demeIDs[i]} : {tree_to_stream_str(xmplr_seq[i])}"
                )
                if considered_args_seq:
                    # `ostream_container`'s default delimiter is a single
                    # space (opencog/util/iostreamContainer.h:64), and a
                    # `vertex_set` is a `std::set<vertex>`, so the order
                    # is `argument`'s own -- by ABSOLUTE VALUE, `-1, 1,
                    # -2, 2, ...` (CARRY-FORWARD.md 4.4). Every element
                    # here is a POSITIVE `argument(idx + 1)`, so sorting
                    # by index IS that order.
                    logger().debug(
                        "Using arguments: "
                        + " ".join(
                            str(a) for a in sorted(considered_args_seq[i], key=lambda a: a.idx)
                        )
                    )
            rep = representation(
                self.simplify_candidate,
                self.simplify_knob_building,
                xmplr_seq[i],
                self._type_sig,
                ignore_ops=ignore_ops_seq[i],
                # C++: deme_expander.cc:293 passes `_params.linear_contin`
                # here, between `actions` and `perm_ratio`. This port used
                # to omit the argument entirely, so `representation`'s own
                # constructor default (`True`, matching representation.h:61)
                # applied instead of the parameter -- and the CLI's
                # `--linear-regression` default is `False`
                # (problem-params.cc:292), so every real run silently got
                # the opposite of what it asked for.
                #
                # It went unnoticed because `linear_contin` is read at
                # exactly one place in classic, `if (!_linear_contin)` at
                # build_knobs.cc:859, inside CONTIN knob building -- which
                # was gated off in this port until 2026-09-09. The symptom,
                # once the gate came off: for exemplar `$2` on a two-input
                # contin table the port built 27 contin knobs where classic
                # builds 38, because `rec_canonize`'s non-linear expansion
                # never ran, and the search could not reach any
                # product-of-sums candidate.
                linear_contin=self._params.linear_contin,
                perm_ratio=self._params.perm_ratio,
            )
            if rep.fields().empty():
                logger().warn(
                    "The representation is empty, perhaps the reduct "
                    "effort for knob building is too high."
                )
                continue
            self._reps.append(rep)

        return len(self._reps) > 0

    def prune_xmplr(self, xmplr: combo_tree, selected_features: Collection[int]) -> combo_tree:
        """C++: moses/moses/deme/deme_expander.cc:308 deme_expander::prune_xmplr

        "Return pruned exemplar from non-selected features"
        (deme_expander.h:126's own comment, on the declaration at
        deme_expander.h:128). Every argument leaf whose
        zero-based column index is NOT in ``selected_features`` is erased
        from a COPY of the exemplar, the survivor is run through
        ``simplify_knob_building``, and -- if that leaves nothing at all --
        a bare type seed replaces it. 25 lines in C++, and three of them
        are traps.

        **Trap 1: the walk is PRE-ORDER, and ``erase`` returns the next
        PRE-ORDER position.** C++ iterates ``for (auto it = res.begin();
        it != res.end();)``; ``tree<T>::begin()``/``end()`` return
        ``pre_order_iterator`` (opencog/util/tree.h:830, :836), and
        ``tree<T>::erase`` (tree.h:805) computes its return value as
        ``iter ret = it; ret.skip_children(); ++ret;`` -- i.e. the
        successor of ``it``'s WHOLE SUBTREE, using the ITERATOR TYPE'S own
        ``operator++``. For a ``pre_order_iterator`` that flag is honoured
        (tree.h:2430), so the answer is this port's
        ``tree.erase_pre_order`` -- NOT ``tree.erase``, which is the SAME
        C++ template instantiated for a ``sibling_iterator`` and returns
        ``node.next_sibling`` (see both methods' docstrings in
        ``pymetta/util/tree.py``). Using ``erase`` here is silent and
        catastrophic: erasing the LAST child of a node yields ``None``,
        the loop reads that as ``end()`` and stops, and every argument
        after that point survives un-pruned. Measured against the C++
        oracle (tools/cpp_probe/prune_xmplr_probe.cc, case 5): on
        ``or(and($1 $2) $3)`` with ``{0}`` selected, classic returns
        ``$1``; the ``erase``-instead-of-``erase_pre_order`` version stops
        after erasing ``$2`` (the ``and``'s last child) and returns
        ``or($1 $3)``.

        What is NOT a trap, checked rather than assumed: computing the
        successor BEFORE the erase (``nxt = res.next_pre_order(it)``, then
        erase, then ``it = nxt``) is EQUIVALENT here, and mutation-testing
        it kills nothing. ``skip_children()`` only matters for a node WITH
        children, and an ``argument`` is always a leaf in a combo tree --
        so the only thing that actually decides this loop is which
        ITERATOR TYPE's ``operator++`` computes the successor, sibling or
        pre-order, not when it is computed.

        **Trap 2: mutation under iteration.** ``res`` is a *value copy*
        (``combo_tree res = xmplr;``), so the caller's exemplar is never
        touched -- ``create_representations`` reuses the same ``exemplar``
        object for every feature set in ``pop_of_selected_feats``, so
        pruning in place would corrupt the second and later demes.
        Reproduced with ``combo_tree(xmplr.begin())``, which is this
        port's ``tree(const iterator_base&)`` deep-copy overload
        (``pymetta/util/tree.py``'s constructor) -- and which degrades
        correctly to an empty tree when ``xmplr`` itself is empty, since
        ``begin()`` is then ``None``. Never advance from an already-erased
        node: this port poisons an erased node's links (``_DEAD``), so
        stepping through one raises rather than silently walking back into
        the live tree the way C++'s freed ``tree_node`` would.

        **Trap 3: the empty-tree seed.** Only a bare-argument exemplar can
        empty the tree -- erasing the root leaves ``res`` genuinely empty
        (C++ splices the root out from between ``head`` and ``feet``;
        ``_unlink`` clears ``_root`` here). Every other shape keeps at
        least its root operator, and an argument-less ``and``/``or``
        reduces to ``true`` instead. Oracle case 10: ``$1`` with nothing
        selected returns ``and`` (``type_to_exemplar(boolean_type)``), not
        an empty tree.

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:325): classic
        # SEGFAULTS on any exemplar that leaves a childless `not`.  Prune
        # the only child of a `not` and `simplify_knob_building` is handed
        # `not()`, which `logical_reduce` dereferences without checking.
        # Measured directly against the real `libcomboreduct.so`
        # (tools/cpp_probe/prune_xmplr_probe.cc, cases 6/7/8/9/12/20, and a
        # standalone one-tree reducer): `not()` and `or(and($2) not())`
        # both exit 139 (SIGSEGV) under `logical_reduction(2)`, while
        # `and()`, `or()`, `and($2)` and `not($1)` all reduce fine -- so it
        # is the childless `not` specifically, not degenerate operators in
        # general.  Reachable from the CLI: `--fs-prune-exemplar 1` on any
        # exemplar containing a negated feature that feature selection
        # drops.  This port CANNOT reproduce a segfault; the same input
        # raises `AttributeError` out of `logical_reduction` (measured:
        # "'NoneType' object has no attribute 'data'"), which is this
        # project's standing "loud where C++ is undefined" treatment (see
        # `tree.py`'s `_unlink` DIVERGENCE for the same trade).  DO NOT
        # FIX: do not add a childless-`not` guard here.  Classic has none,
        # and inventing one would make the port SUCCEED where classic
        # crashes -- a divergence in the direction that hides bugs.
        """
        res = combo_tree(xmplr.begin())
        # remove literals of non selected features from the exemplar
        it = res.begin()
        while it is not res.end():
            if is_argument(it.data):
                # C++: `get_argument(*it).abs_idx_from_zero()`.  There is no
                # `get_argument` in this port (grepped): `vertex` is a plain
                # union of Python types, so a vertex for which `is_argument`
                # holds simply IS the `argument`, and `boost::get<argument>`
                # has nothing to unwrap.
                feature = it.data.abs_idx_from_zero()
                if feature not in selected_features:
                    it = res.erase_pre_order(it)  # not selected, prune it
                else:
                    it = res.next_pre_order(it)
            else:
                it = res.next_pre_order(it)
        self.simplify_knob_building(res)
        # if the exemplar is empty use a seed
        if res.empty():
            # C++: `get_type_node(get_signature_output(_type_sig))`.
            # `get_type_node` is not ported anywhere in this project, and
            # `get_signature_output(sig).node` -- the only thing it is
            # applied to here -- is definitionally `sig.otype`
            # (type_signature.py's own `get_signature_output`), the same
            # equivalence `demo_problems._alphabet_size` relies on.
            otn = self._type_sig.otype
            res = _type_to_exemplar(otn)
        return res

    def create_demes(self, exemplar: combo_tree, n_expansions: int = 0) -> bool:
        """C++: moses/moses/deme/deme_expander.cc:375 deme_expander::create_demes

        "Create one or more demes ... By default, only one deme is
        created, by (randomly) using all possible features in knobs
        randomly attached to the given exemplar." (cc:334-345's own doc
        comment, condensed -- see the class it is attached to in C++ for
        the full text, reproduced in this module's own docstring instead
        of repeated here.)

        The three bare ``OC_ASSERT``s (cc:379-381, no message in C++) guard
        against calling this on a ``deme_expander`` that already holds
        state from a prior ``create_demes`` -- ``free_demes()`` must run
        first. Ported as explicit, message-carrying checks.
        """
        if self._ignore_cols_seq:
            raise RuntimeError(
                "deme_expander.create_demes: _ignore_cols_seq must be empty "
                "(OC_ASSERT, deme_expander.cc:379) -- call free_demes() first"
            )
        if self._reps:
            raise RuntimeError(
                "deme_expander.create_demes: _reps must be empty "
                "(OC_ASSERT, deme_expander.cc:380) -- call free_demes() first"
            )
        if self._demes:
            raise RuntimeError(
                "deme_expander.create_demes: _demes must be empty "
                "(OC_ASSERT, deme_expander.cc:381) -- call free_demes() first"
            )

        self.create_demeIDs(n_expansions)
        self.create_representations(exemplar)

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:388-397): classic
        # builds a NESTED ss-deme structure in the `if (_filter_params.
        # n_subsample_demes > 1)` arm. Gated -- see module docstring; only
        # the `else` arm (cc:394-396, one deme per representation) is
        # ported. CORRECTED (S10 branch review, Minor): the gate check
        # itself must stay INSIDE the `for i in range(len(self._reps))`
        # loop below, exactly as C++'s own `if` sits inside its `for
        # (unsigned i = 0; i < _reps.size(); i++)` (cc:388-397) -- it used
        # to be hoisted out, before this loop and unconditional. That
        # differs from C++ exactly when `self._reps` is EMPTY (no viable
        # representation -- `create_representations` returned `False`):
        # C++'s loop then runs ZERO iterations, so its `n_subsample_demes
        # > 1` check is never even evaluated, and `create_demes` simply
        # returns `not _demes.empty()` == `False` -- a normal "no demes,
        # caller should try another exemplar" signal, no exception. The
        # hoisted version raised `NotImplementedError` in that same
        # zero-rep case whenever the flag was on, which C++ never does.
        for i in range(len(self._reps)):
            if self._filter_params.n_subsample_demes > 1:
                raise NotImplementedError(
                    "deme_expander: subsample-deme filtering "
                    "(subsample_deme_filter_parameters.n_subsample_demes "
                    "> 1) is P2, not ported; leave n_subsample_demes at "
                    "its default (0)."
                )
            self._demes.append([instance_set(self._reps[i].fields(), self._demeIDs[i])])

        return len(self._demes) > 0

    # ---------------------------------------------------------- optimizing

    def optimize_demes(self, max_evals: int, max_time: int) -> None:
        """C++: moses/moses/deme/deme_expander.cc:403 deme_expander::optimize_demes

        Both `if (_params.fstor)` arms (cc:432 and the post-loop reset at
        cc:499) are documented, and skipped, at their own line in the body
        below -- a real `fstor` no longer raises here, and a full
        `--enable-fs` search runs to completion and matches classic byte
        for byte. See that DIVERGENCE for what is skipped and the
        measurement that says it costs nothing observable.

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:477,508): classic
        # checks `if (n_ss_demes > 1)` at two points in this function
        # (cc:477, `_cscorer.ignore_rows_at_times`/`ignore_rows`, and the
        # post-loop reset at cc:508). Neither is ported.
        #
        # The `n_subsample_demes` half of the original four checks is
        # UNCHANGED and still consolidated into the second guard below, at
        # entry: `_filter_params.n_subsample_demes > 1` cannot be true here
        # without `create_demes` already having raised first -- the only
        # path that ever builds `_reps`/`_demes` -- UNLESS a caller mutates
        # `_filter_params` directly between `create_demes()` and this call
        # (no real caller in classic does this; `moses_main.h:318`'s single
        # real construction site passes it by reference and never mutates
        # it mid-search). That consolidated check is ALSO, independently,
        # backed by `subsample_by_row`/`subsample_by_time`'s own gates a
        # few lines below (both are still called, unconditionally, matching
        # cc:416-421's own structure) -- measured directly (mutation
        # testing, see the task report): with ONLY this top-level check
        # disabled, a caller that reaches the loop still hits one of those
        # two gates on the very first iteration and gets the same
        # `NotImplementedError`.
        #
        # `size_t nleft = ...`-style unsigned-wrap risk, checked and found
        # INERT here (not measured via a probe -- verified analytically):
        # `max_evals_per_deme = max_evals / _demes.size()` (cc:405) promotes
        # `max_evals` (signed `int`) to `size_t` (`unsigned long`) before
        # dividing, per the usual arithmetic conversions -- the same class
        # of hazard as hill-climbing.cc's `nleft` (hill_climbing.py's own
        # DIVERGENCE note). But `_demes.size()` is EXACTLY 1 on every path
        # this port ever reaches (`_reps`/`_demes` have length 0 or 1 --
        # never more: with `fstor is None` there is exactly one
        # `xmplr_seq` entry, as before; with a real `fstor` the only
        # ported algorithm, `"simple"`, always returns a single-entry
        # `feature_set_pop` regardless of `n_demes` -- see
        # `create_representations`'s own docstring -- so `xmplr_seq` is
        # still length 1 either way) or 0 (the empty-representation case,
        # see `create_representations`), so `_demes.size()` is always
        # exactly 0 or 1 by the time control reaches this division.
        # Division
        # by 1 is the identity on any bit pattern, signed or unsigned, so
        # `max_evals_per_deme == max_evals` bit-for-bit for every possible
        # `int` value, positive or negative -- there is no wrap to
        # reproduce. `n_ss_demes = max(n_subsample_demes, 1)` is likewise
        # always exactly 1 here (the `> 1` case already raised above), so
        # the second `/=` is the same identity. Division by 0 (`_demes`
        # empty) is UB in C++ too (integer division by zero) and unguarded
        # there -- ported the same way: a caller MUST check `create_demes
        # ()`'s return value before calling this, exactly as C++ requires;
        # this port does not add a check C++ does not have (DO NOT FIX).
        """
        if self._filter_params.n_subsample_demes > 1:
            raise NotImplementedError(
                "deme_expander: subsample-deme filtering (subsample_deme_"
                "filter_parameters.n_subsample_demes > 1) is P2, not "
                "ported; leave n_subsample_demes at its default (0)."
            )

        n_demes = len(self._demes)
        max_evals_per_deme = max_evals // n_demes
        n_ss_demes = max(self._filter_params.n_subsample_demes, 1)
        max_evals_per_deme //= n_ss_demes

        for i in range(n_demes):
            # Results intentionally unused: both are provably `[set()]` (a
            # single empty set) given `n_ss_demes == 1` is the only value
            # reachable past the guards above, and that value is only ever
            # consumed inside the `if (n_ss_demes > 1)` arm this port does
            # not reach. Still called, matching cc:416-421's own
            # unconditional calls, structurally.
            _ = self.subsample_by_row()
            _ = self.subsample_by_time()

            for j in range(n_ss_demes):
                if logger().is_debug_enabled():
                    # C++: deme_expander.cc:425-430.
                    logger().debug(
                        f"Optimize deme {self._demes[i][j].getID()}; "
                        f"max evaluations allowed: {max_evals_per_deme}"
                    )
                # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:432-450):
                # classic's `if (_params.fstor)` body here is NOT executed by
                # this port. It does exactly two things, and neither is
                # reachable from `moses/deme/`:
                #
                #   _cscorer.ignore_cols(_ignore_cols_seq[i]);      // cc:435
                #   score_t t = _cscorer.best_possible_score();     // cc:448
                #   logger().info("Inferred target score ... = %g", t);
                #   logger().info("It appears there is an algorithmic bug ...");
                #
                # `behave_cscore` (pymetta/moses/scoring/behave_cscore.py) has
                # neither `ignore_cols` nor `best_possible_score`, and the one
                # scorer a `--enable-fs` table run ever reaches,
                # `ctruth_table_bscore`, extends `bscore_base` rather than
                # `bscore_ctable_base` in this port (bscores.py's own class
                # docstring), so it has no `_orig_ctable`/`_wrk_ctable` split
                # for `ignore_cols` to swap. Closing that needs THREE edits
                # in files this task does not own -- `CTable.
                # filtered_preserve_idxs` (comboreduct/table/table.py, the
                # port of table.h:661), `behave_cscore.ignore_cols`/
                # `best_possible_score` (scoring/behave_cscore.py, the ports
                # of behave_cscore.h:79 and behave_cscore.cc:123), and
                # re-parenting `ctruth_table_bscore` onto `bscore_ctable_base`
                # (scoring/bscores.py) -- so it is handed off, not hidden.
                #
                # What the omission costs, MEASURED rather than argued.
                # `ignore_cols` is classic's own stated OPTIMIZATION ("Attempt
                # to compress the CTable further (to optimize and update max
                # score)", cc:433-434): it rewrites every non-selected column
                # to `null_vertex` and re-compresses, which MERGES the rows
                # that then agree. `ctruth_table_bscore` scores each
                # compressed row as `-counter.get(negate(tr(row)))`
                # (bscores.cc:276-294) and `behave_cscore::get_cscore` sums
                # those (behave_cscore.cc:110), so merging rows that a
                # candidate cannot tell apart -- and it cannot, because the
                # representation was built with the complement of the
                # selected features in `ignore_ops` -- adds the same counts
                # in a different order. The SUM, i.e. the composite score the
                # optimizer sees, is unchanged. Only the bscore's LENGTH
                # changes, and the one CLI surface that would show it,
                # `-t/--output-bscore`, is separately gated in `merging.py`.
                # Confirmed by running this port with a faithful
                # `filtered_preserve_idxs`/`ignore_cols` monkeypatched in and
                # with it stubbed to a no-op: byte-identical stdout on every
                # dataset x --fs-algo x --fs-target-size x seed cell of the
                # oracle matrix, and identical to classic's own binary.
                #
                # The post-loop reset (cc:499-506, `_cscorer.ignore_cols(
                # std::set<arity_t>())`) is skipped for the same reason and
                # is a no-op given the above.
                cpx_scorer = complexity_based_scorer(
                    self._cscorer, self._reps[i], self._params.reduce_all
                )
                self._optimize(self._demes[i][j], cpx_scorer, max_evals_per_deme, max_time)

    def free_demes(self) -> None:
        """C++: moses/moses/deme/deme_expander.cc:516 deme_expander::free_demes"""
        self._ignore_cols_seq = []
        self._demes = []
        self._reps = []

    def total_evals(self) -> int:
        """C++: moses/moses/deme/deme_expander.cc:523 deme_expander::total_evals

        "Return the total number of evaluations across all demes" (the
        header's own comment).
        """
        res = 0
        for ss_demes in self._demes:
            for deme in ss_demes:
                res += deme.n_evals
        return res
