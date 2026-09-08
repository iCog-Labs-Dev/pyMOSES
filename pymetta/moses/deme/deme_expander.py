"""C++: moses/moses/deme/deme_expander.{h,cc}

``deme_expander`` -- the deme factory. Given an exemplar ``combo_tree``,
builds a ``representation`` (knobs over the exemplar) and an empty
``instance_set`` ("deme") sized to that representation's ``field_set``, then
drives an injected ``optimizer_base`` (S9's ``hill_climbing``, at the test
level -- see below) to fill and score that deme.

**Single-deme path only**, per this task's own scope: the two axes that
turn ONE exemplar into MULTIPLE demes are both P2 subsystems, gated exactly
the way the S10 plan's metapopulation task gates boosting/domination/
diversity -- "port the gate faithfully, raise inside the gated body" rather
than half-implementing either:

* **On-the-fly feature selection** (``deme_parameters.fstor``, wired to
  ``deme/feature_selector.{h,cc}``, 710 lines, ``--enable-fs``) -- would
  build one representation PER selected feature set, i.e. multiple demes
  from one exemplar (deme_expander.cc:205-268's own comment: "a
  representation, and a deme, for each distinct feature set"). Gated in
  ``create_demeIDs`` and ``create_representations``: ``fstor is not None``
  raises ``NotImplementedError`` immediately, matching classic's own
  ``if (_params.fstor)`` branch condition at both call sites.
* **Subsample-deme filtering**
  (``subsample_deme_filter_parameters.n_subsample_demes``, S10 Task 1's
  ``metapop_params.py``) -- would subdivide EACH representation's single
  deme into ``n_subsample_demes`` "ss-demes" over disjoint row/timestamp
  subsets (the header's own comment on ``_demes``: "a deme can be further
  subdivided into multiple demes using a subsampled dataset"). Gated in
  ``create_demes`` and ``optimize_demes``: ``n_subsample_demes > 1`` raises.
  The plan's own scope table lists this as one of the whole S10 slice's
  default-off P2 subsystems, alongside boosting/domination/diversity.

Both gates default OFF (``fstor=None``, ``n_subsample_demes=0``), so the
fully-ported default path is: exactly one representation, exactly one deme,
per ``create_demes`` call.

NOT ported at all (not merely gated -- these are feature-selector-only
helpers with no caller once ``fstor`` is always ``None``, the same
"omit the whole branch, don't add a placeholder" discipline
``build_knobs.py`` uses for enum/ANN/object/message/procedure knob
building): ``fs_to_names`` (deme_expander.h:115, deme_expander.cc:48),
``log_selected_feature_sets`` (deme_expander.h:121, deme_expander.cc:57),
``prune_xmplr`` (deme_expander.h:128, deme_expander.cc:308). Also NOT
ported: the ``#if THIS_IS_DISABLED_UNTIL_ABOVE_IS_FIXED`` block inside
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

from collections.abc import Callable

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.deme.deme_params import deme_parameters
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

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:152-156): classic
        # takes the multi-ID branch only when BOTH ``_params.fstor`` is set
        # AND ``fstor->params.n_demes > 1`` -- with a single-ID fstor
        # (``n_demes <= 1``) it silently falls through to the SAME
        # ``emplace_back(n_expansions + 1)`` the no-fstor case uses. This
        # port cannot draw that distinction: ``feature_selector`` (deme/
        # feature_selector.{h,cc}) is not ported at all, so there is no
        # ``.params.n_demes`` to read off a caller-supplied ``fstor``, and
        # any non-``None`` value is already unsupported regardless of what
        # it would report. Gating on ``fstor is not None`` alone therefore
        # raises in a strict SUPERSET of the cases C++ would take the
        # multi-ID branch for -- never a case C++ would treat as plain
        # single-deme -- so this cannot silently accept something classic
        # would have rejected, only reject something (a single-ID fstor)
        # this port has no way to support in the first place.
        """
        self._demeIDs = []
        if self._params.fstor is not None:
            raise NotImplementedError(
                "deme_expander: on-the-fly feature selection (deme/"
                "feature_selector.{h,cc}, 710 lines) is P2, gated behind "
                "--enable-fs; not ported. Construct deme_parameters with "
                "fstor=None (the default)."
            )
        self._demeIDs.append(demeID_t(n_expansions + 1))

    def create_representations(self, exemplar: combo_tree) -> bool:
        """C++: moses/moses/deme/deme_expander.cc:159 deme_expander::create_representations

        Gated on ``_params.fstor`` (see module docstring); only the
        ``else`` arm -- "no dynamic feature selection" (cc:270's own
        comment) -- is ported: ``ignore_ops_seq.push_back(_params.
        ignore_ops); xmplr_seq.push_back(exemplar);`` (cc:271-272), always
        exactly one entry.

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:287-294): the
        # real ``representation(...)`` call also forwards ``_params.
        # perceptions``, ``_params.actions`` and ``_params.linear_contin``.
        # S8's ``representation`` (pymetta/moses/representation/
        # representation.py) is Boolean-only and accepts none of the
        # three -- its own module docstring already excludes the mutable
        # action/perceptions machinery and continuous domains outright.
        # ``deme_parameters`` still carries all three fields (see
        # deme_params.py) for struct-shape fidelity; they are simply never
        # forwarded here, matching representation.py's own scope exactly
        # (nothing in this port's Boolean-only slice can even construct
        # a non-``None`` ``perceptions``/``actions`` value to forward).

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
        if self._params.fstor is not None:
            raise NotImplementedError(
                "deme_expander: on-the-fly feature selection (deme/"
                "feature_selector.{h,cc}, 710 lines) is P2, gated behind "
                "--enable-fs; not ported. Construct deme_parameters with "
                "fstor=None (the default)."
            )

        ignore_ops_seq = [self._params.ignore_ops]
        xmplr_seq = [exemplar]

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
        # ported.
        if self._filter_params.n_subsample_demes > 1:
            raise NotImplementedError(
                "deme_expander: subsample-deme filtering (subsample_deme_"
                "filter_parameters.n_subsample_demes > 1) is P2, not "
                "ported; leave n_subsample_demes at its default (0)."
            )

        for i in range(len(self._reps)):
            self._demes.append([instance_set(self._reps[i].fields(), self._demeIDs[i])])

        return len(self._demes) > 0

    # ---------------------------------------------------------- optimizing

    def optimize_demes(self, max_evals: int, max_time: int) -> None:
        """C++: moses/moses/deme/deme_expander.cc:403 deme_expander::optimize_demes

        # DIVERGENCE(C++ moses/moses/deme/deme_expander.cc:432,477,499,508): classic
        # checks `if (_params.fstor)` at TWO points in this function
        # (cc:432 and the post-loop reset at cc:499) and `if (n_ss_demes >
        # 1)` at two more (cc:477 and the post-loop reset at cc:508). Given
        # the gates already in `create_demeIDs`/`create_representations`
        # (fstor) and `create_demes` (n_subsample_demes), `_params.fstor`
        # and `_filter_params.n_subsample_demes > 1` cannot be true here
        # without `_reps`/`_demes` already failing to have been populated
        # by an EARLIER raise -- the only path that ever builds them is
        # `create_demes`, above -- UNLESS a caller mutates `_params`/
        # `_filter_params` directly between `create_demes()` and this call
        # (no real caller in classic does this; `moses_main.h:318`'s single
        # real construction site passes both by reference and never
        # mutates either mid-search). This port consolidates all four
        # checks into the two below, at entry: they raise exactly as early
        # as C++ would in every REACHABLE case, without four near-duplicate
        # guards that can only fire via that same pathological direct
        # mutation. The `n_subsample_demes` half of this consolidated
        # check is ALSO, independently, backed by `subsample_by_row`/
        # `subsample_by_time`'s own gates a few lines below (both are
        # still called, unconditionally, matching cc:416-421's own
        # structure) -- measured directly (mutation testing, see the task
        # report): with ONLY this top-level check disabled, a caller that
        # reaches the loop still hits one of those two gates on the very
        # first iteration and gets the same `NotImplementedError`. The
        # `fstor` half has no such backup inside this function -- it is
        # the only thing standing between a mutated `_params.fstor` and
        # running the un-ported `if (_params.fstor)` body past this point.
        #
        # `size_t nleft = ...`-style unsigned-wrap risk, checked and found
        # INERT here (not measured via a probe -- verified analytically):
        # `max_evals_per_deme = max_evals / _demes.size()` (cc:405) promotes
        # `max_evals` (signed `int`) to `size_t` (`unsigned long`) before
        # dividing, per the usual arithmetic conversions -- the same class
        # of hazard as hill-climbing.cc's `nleft` (hill_climbing.py's own
        # DIVERGENCE note). But `_demes.size()` is EXACTLY 1 on every path
        # this port ever reaches (`_reps`/`_demes` have length 0 or 1 --
        # never more, since the only way to get more than one representation
        # is the feature-selection multi-deme path, gated above) or 0 (the
        # empty-representation case, see `create_representations`). Division
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
                "deme_expander: on-the-fly feature selection (deme/"
                "feature_selector.{h,cc}, 710 lines) is P2, gated behind "
                "--enable-fs; not ported. Construct deme_parameters with "
                "fstor=None (the default)."
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
