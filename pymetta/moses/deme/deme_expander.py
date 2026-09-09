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

  Two sub-pieces remain gated, both off by default and both narrower
  than the old blanket "fstor is not ported" gate:

  - ``festor.params.prune_xmplr`` (feature_selector_parameters's own
    struct default is INDETERMINATE -- see ``feature_selector.py``'s
    module docstring; this port and every real CLI caller pin it to
    ``False``) selects ``prune_xmplr`` (deme_expander.cc:308-325,
    erasing argument leaves from the exemplar under iteration, with a
    ``type_to_exemplar`` fallback if that empties the tree) -- NOT
    ported; raises in ``create_representations`` if ever set to
    ``True``.
  - ``optimize_demes``'s OWN ``if (_params.fstor)`` branch
    (deme_expander.cc:432-471) needs
    ``behave_cscore.ignore_cols()``/``best_possible_score()``, neither
    ported (``pymetta.moses.scoring`` is not this task's file to edit,
    and is under concurrent edit by another agent as of this task) --
    see ``optimize_demes``'s own docstring. A real ``fstor`` therefore
    builds correct representations/demes but cannot yet be driven
    through a full search loop; report this as a required follow-up
    for whoever owns ``pymetta/moses/scoring/`` if a complete
    ``--enable-fs`` search (not just representation-building) is
    needed.

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

NOT ported at all (not merely gated -- these are feature-selector-only
helpers that are either logging-only or, for ``prune_xmplr``, gated at
its own call site rather than omitted, since it DOES have a live caller
now that ``fstor`` is real): ``fs_to_names`` (deme_expander.h:115,
deme_expander.cc:48) and ``log_selected_feature_sets``
(deme_expander.h:121, deme_expander.cc:57) are both logging-only (their
only effect is feeding ``logger().debug()``/``ostream_container`` --
nothing else observes their return values), matching the "no logging
surface in this port" convention stated throughout this project. Also
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
from collections.abc import Callable

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.deme.feature_selector import get_argument_abs_idx_from_zero_set
from pymetta.moses.metapopulation.metapop_params import subsample_deme_filter_parameters
from pymetta.moses.moses.optimization.optimization import optimizer_base
from pymetta.moses.moses.types import demeID_t
from pymetta.moses.representation.instance_scorer import complexity_based_scorer
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.representation import representation
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.util.rng import randGen

Simplifier = Callable[[combo_tree], None]

__all__ = ["deme_expander"]


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
          205's own loop): the exemplar's own features are inserted
          back in (cc:224-232's ``else`` arm -- ``prune_xmplr`` is the
          gated ``if`` arm, see below), then any ``enforce_features``
          are sampled in and added (cc:234-238); the COMPLEMENT of the
          final feature set becomes ``ignore_cols``/``ignore_ops``
          (cc:240-256, one ``argument(i + 1)`` per unselected column
          index ``i``) and the set itself becomes ``considered_args``
          (unused beyond an omitted debug log -- see module docstring).

          ``festor.params.prune_xmplr`` (see ``feature_selector.py``'s
          own module docstring for why this field's C++ default is
          indeterminate) selects ``prune_xmplr`` (deme_expander.cc:
          308-325) -- NOT ported (needs ``combo_tree`` erase-under-
          iteration plus a ``type_to_exemplar`` empty-tree fallback,
          neither exercised or verified here); raises rather than
          silently falling back to the non-pruning arm. Off by
          default and by every real CLI's own default
          (``--fs-prune-exemplar`` defaults to ``0``,
          problem-params.cc:1663-1665), so this never fires on the
          default ``--enable-fs`` path.

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
        xmplr_seq: list[combo_tree] = []

        if self._params.fstor is not None:
            copied_params = copy.deepcopy(self._params.fstor.params)
            festor = type(self._params.fstor)(self._params.fstor._ctable, copied_params)
            pop_of_selected_feats = festor(exemplar)
            xmplr_features = get_argument_abs_idx_from_zero_set(exemplar)

            for selected_score, selected_feats in pop_of_selected_feats:
                del selected_score  # only used for the omitted debug log
                if festor.params.prune_xmplr:
                    raise NotImplementedError(
                        "deme_expander.create_representations: "
                        "fstor.params.prune_xmplr=True (deme_expander.cc:"
                        "308-325, needs combo_tree erase-under-iteration + "
                        "a type_to_exemplar empty-tree fallback) is not "
                        "ported; leave prune_xmplr at its default (False "
                        "-- see feature_selector.py's module docstring)."
                    )
                selected_feats = set(selected_feats) | xmplr_features
                xmplr_seq.append(exemplar)

                if festor.params.enforce_features:
                    enforced = festor.sample_enforced_features()
                    selected_feats = selected_feats | enforced

                arity = festor._ctable.get_arity()
                ignore_cols: set[int] = set()
                ignore_ops: set = set()
                for i in range(arity):
                    arg = argument(i + 1)
                    if i not in selected_feats:
                        ignore_cols.add(i)
                        ignore_ops.add(arg)
                self._ignore_cols_seq.append(ignore_cols)
                ignore_ops_seq.append(ignore_ops)
        else:  # no dynamic feature selection
            ignore_ops_seq.append(self._params.ignore_ops)
            xmplr_seq.append(exemplar)

        for i in range(len(xmplr_seq)):
            rep = representation(
                self.simplify_candidate,
                self.simplify_knob_building,
                xmplr_seq[i],
                self._type_sig,
                ignore_ops=ignore_ops_seq[i],
                perm_ratio=self._params.perm_ratio,
            )
            if rep.fields().empty():
                continue
            self._reps.append(rep)

        return len(self._reps) > 0

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

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:432,477,499,508): classic
        # checks `if (_params.fstor)` at TWO points in this function
        # (cc:432 and the post-loop reset at cc:499) and `if (n_ss_demes >
        # 1)` at two more (cc:477 and the post-loop reset at cc:508).
        #
        # UNLIKE `create_demeIDs`/`create_representations` (which now honour
        # a real `fstor`, see module docstring), this function's OWN
        # `if (_params.fstor)` body (cc:432-450) is NOT ported: it calls
        # `_cscorer.ignore_cols(...)` and `_cscorer.best_possible_score()`
        # to mask the deme's scorer down to the selected columns, and
        # `pymetta.moses.scoring.behave_cscore` (this task's `moses/deme/`
        # scope does not extend to `moses/scoring/`, which is also under
        # concurrent edit by another agent as of this task) has neither
        # method. A real `fstor` therefore still raises HERE -- narrower
        # than before (create_demeIDs/create_representations no longer
        # raise for it), but this function cannot yet drive a full
        # `--enable-fs` search loop to completion. Report a need for
        # `behave_cscore.ignore_cols`/`best_possible_score` as a required
        # follow-up to whoever owns `pymetta/moses/scoring/` if that is
        # needed. The post-loop reset (cc:499, `_cscorer.ignore_cols({})`)
        # is unreachable for the same reason and is likewise not ported.
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
        # see `create_representations`). This function itself still
        # raises unconditionally whenever `fstor is not None` (see below),
        # so in practice `_demes.size()` is always exactly 0 or 1 by the
        # time control ever reaches this division regardless. Division
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
        if self._params.fstor is not None:
            raise NotImplementedError(
                "deme_expander.optimize_demes: a real fstor's OWN branch "
                "here (deme_expander.cc:432-450) needs "
                "behave_cscore.ignore_cols()/best_possible_score(), "
                "neither ported (pymetta.moses.scoring is out of this "
                "task's scope). create_demeIDs/create_representations "
                "already honour this fstor -- only driving it through a "
                "full search loop is unsupported. Construct "
                "deme_parameters with fstor=None to run optimize_demes."
            )
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
