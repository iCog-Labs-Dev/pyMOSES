"""C++: moses/moses/metapopulation/ensemble.{h,cc}

The FULL AdaBoost/experts ensemble (P2 metapopulation task) -- replacing
the previous minimal stub (S10 Task 1: "enough to be held and constructed
... every method that would require the unported boosting machinery
raises `NotImplementedError`"). Ported whole: `add_adaboost`, `add_expert`,
`get_adaboost_tree`, `get_exact_tree`, `get_expert_tree`, `flat_score`.
The default-OFF path (`ensemble_parameters.do_boosting = False`, the
default) is UNCHANGED byte-for-byte: `__init__`'s own `if not ep.
do_boosting: return` guard still short-circuits before any of the new
code below runs, exactly as it did in the stub.

## Prerequisite check (this task's own brief: "check the already-ported
## `scoring_base.py` weighting hooks before starting; a gap is a
## follow-up, not a file to edit")

`use_weighted_scores`/`reset_weights`/`update_weights`/`sum_bscore`/
`size`/`call_ensemble` (`scoring_base.py`, `bscore_base`) are ALL real,
landed by the P2 scorer-family task (that module's own docstring: "The
`_weights`/`use_weighted_scores`/... apparatus ... IS ported below") --
SUFFICIENT for this file: every one of them is called below exactly as
`ensemble.cc` calls it, on `self._bscorer` (a `bscore_base` reference in
C++; `cs._bscorer` here, `ensemble.py`'s own established naming --
`behave_cscore.get_bscorer()` was never given its own method, see the
class docstring below for the exact citation this predates).

`get_error(const behavioral_score&)`/`get_error(const combo_tree&)`
(`scoring_base.h:142-143`) are NOT on `bscore_base` in this port at all
-- classic's own BASE-class bodies are themselves `OC_ASSERT(false, "...
not implemented for bscorer %s", ...)` (`scoring_base.cc:113-133`), so a
concrete bscorer must override at least one to make boosting usable in
EITHER language; this port already has real overrides ready to drive
this file end-to-end -- `logical_bscore.get_error` (`demo_problems.py`)
and `select_bscore.get_error`/`precision_bscore.get_error` (`select_
bscore.py`/`precision_bscore.py`, both P2 scorer-family) -- so nothing
here is blocked. `test_ensemble.py` drives real `add_adaboost` end to end
against `logical_bscore` (the only one of the three usable standalone,
with no CTable/table-loading setup).

## Two REAL prerequisite gaps found -- reported as follow-ups, not edited
## (both files are outside this task's owned scope, `pymetta/moses/
## metapopulation/`)

1. **[CLOSED] `scored_combo_tree` carries its own boosting weight.**
   It has `weight` (`types.h:274`) with `get_weight`/`set_weight`
   (`types.h:291,295`); it is in `__slots__`, `__eq__` compares it
   (`types.cc:56`) and `_copy_sct` carries it across a copy. **This file
   is now rewired onto it** -- `_get_weight`/`_set_weight` are
   pass-throughs and the private `_sct_weights` dict is gone.

   That dict was keyed by object IDENTITY (`scored_combo_tree.__hash__`
   is `None`), which worked, but made the weight invisible to everything
   outside this class. The place that mattered is `metapop_printer`'s
   `output_ensemble` branch (`moses_main.h:124`), which prints
   `sct.get_weight()` per member: once the accessor existed but this file
   still wrote to the shadow dict, that branch would have printed the
   `1.0` constructor default for every tree while boosting believed
   otherwise -- silently wrong output rather than a crash. It was gated
   to raise for exactly that reason; with the rewiring done, the gate can
   go.

2. **`behave_cscore` has no `get_bscorer()`** -- already noted by the
   PRIOR stub's own docstring (see class docstring below, unchanged):
   `cs._bscorer` stands in for it, matching that module's own documented
   choice not to add "a thin one-line delegation" method.
3. **No concrete scorer in this port overrides `call_ensemble`** (`bscore_
   base.operator()(const scored_combo_tree_set&)`) -- classic's own base
   body is itself `OC_ASSERT(false, "Ensemble scoring not implemented for
   bscorer %s", ...)` (`scoring_base.cc:74-75`), same shape as `get_error`
   above, and EVERY real concrete scorer this project has a partial port
   of overrides it for real in classic (`logical_bscore`, `bscores.cc:85`;
   `ctruth_table_bscore`, `bscores.cc:303`; `select_bscore`, `select_
   bscore.cc:191`; `precision_bscore`, `precision_bscore.cc:368`) -- but
   NONE has been given one in this port yet (`precision_bscore.py`'s own
   docstring and `demo_problems.py`'s own docstring both already note
   this for their own classes: "not overridden here, so they inherit
   `bscore_base.call_ensemble`'s raising"). This blocks `add_expert`'s
   exact-experts weight-update step and `flat_score` against every real
   scorer currently in this port -- `test_ensemble.py` works around it
   with its own test-local `_BoostableBscore.call_ensemble` override (a
   deliberately UNWEIGHTED simplification -- see that test double's own
   docstring for why prerequisite gap #1 above makes the real weighted
   formula unimplementable there too). The real fix belongs in whichever
   of `bscores.py`/`demo_problems.py`/`select_bscore.py`/
   `precision_bscore.py` a caller actually wants boosting to work
   against -- none are edited here (`pymetta/moses/scoring/`/`pymetta/
   moses/main/` are outside this task's owned scope).

## `_scored_trees` is now a real `scored_combo_tree_set`

The prior stub's own DIVERGENCE note (a plain Python `list`) is
RESOLVED: `add_candidates` no longer always raises before anything is
ever inserted, so the dedup-by-tree-structure semantics
`scored_combo_tree_set` (types.py) actually implements now matter --
this class holds one for real, matching classic's own `scored_combo_tree_
set _scored_trees` member (`ensemble.h:83`) exactly, including that
class's own disclosed "no iteration-order promise" divergence (types.py's
own docstring) -- `add_adaboost`'s `std::min_element`-with-a-greater-
comparator scan below inherits that same "no promise in either language"
status, reproduced via the SAME manual left-to-right first-occurrence-
wins translation `diversity.py`'s `set_diversity` already established for
an identical `std::min_element` shape.

## Types: everything here is `double`, not `score_t`

`ensemble_parameters`'s own docstring already establishes `expalpha`/
`bias_scale` as plain C++ `double`, never narrowed; EVERY local this file
computes (`_tolerance`, `_bias`, `_row_bias`, `err`, `alpha`, `expalpha`,
`rcpalpha`, the boosting `weights` list, the ensemble tree's own `weight`/
`bias` vertex constants) is likewise `double` in `ensemble.cc` --
`vertex weight = sct.get_weight()` embeds a `double` directly as a
`contin_t` vertex (`contin_t` IS `double`, types.py's own module
docstring), so no `_to_score_t` narrowing appears anywhere in this file
except `flat_score`'s own final return (`score_t`, matching
`ensemble::flat_score`'s C++ return type and `boost::accumulate(bs, 0.0)`'s
own "accumulate at double, narrow once at return" shape -- the identical
convention `scoring_base.py`'s `sum_bscore` and this port's every other
`boost::accumulate` call site already follow).

## `add_expert`'s inexact-experts branch is a genuine, unconditional crash
## in classic itself -- DO NOT FIX

The `else` arm of `add_expert` (not `exact_experts`) hits `OC_ASSERT(false,
"this doesn't work right now.")` (`ensemble.cc`, its own comment: "XXX
currently broken ...") EVERY time it runs, AFTER `kopy.set_weight(alpha);
_scored_trees.insert(kopy);` have already mutated `_scored_trees` for
real, but BEFORE the bias-adjustment loop and the `update_weights` call
that follow it -- both of which are unreachable dead code in C++ itself,
never executed in any real run. Ported EXACTLY that far and no further:
`_set_weight`/`self._scored_trees.insert(kopy)` run for real, then this
raises `AssertionError` unconditionally, leaving `_scored_trees` mutated
-- reproducing the crash-with-a-side-effect-already-applied shape, not
merely "raises", the same "port what C++ actually does, including its own
bugs" rule CLAUDE.md and this task's own brief both state.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters
from pymetta.moses.metapopulation.metapopulation import _copy_sct
from pymetta.moses.moses.types import (
    _to_score_t,
    epsilon_score,
    score_t,
    scored_combo_tree,
    scored_combo_tree_set,
)

if TYPE_CHECKING:
    from pymetta.comboreduct.combo.vertex import combo_tree
    from pymetta.moses.scoring.behave_cscore import behave_cscore

__all__ = ["ensemble"]


def _is_correct(val: score_t) -> bool:
    """C++: moses/moses/metapopulation/ensemble.cc:71 is_correct (static)

    "Is this behavioral score correct? For boolean scores, correct is 0.0
    and incorrect is -1.0." (ensemble.cc:69-70's own comment.)
    """
    return -0.5 < val


class ensemble:
    """C++: moses/moses/metapopulation/ensemble.h:60 ensemble

    "The ensemble holds a collection of the fittest combo trees ... In
    particular (depending on parameters), the ensemble is designed to
    hold a boosted collection of combo trees" (ensemble.h:35-58's own doc
    comment, condensed).
    """

    def __init__(self, cs: behave_cscore, ep: ensemble_parameters | None = None) -> None:
        """C++: moses/moses/metapopulation/ensemble.cc:39 ensemble::ensemble

        Member-initializer-list order preserved: `_params`/`_bscorer` are
        bound FIRST (`_bscorer(cs.get_bscorer())` runs even when boosting
        is off), before the constructor BODY's `if (not ep.do_boosting)
        return;` guard (ensemble.cc:42-43). `cs.get_bscorer()` is spelled
        `cs._bscorer` here: `behave_cscore.get_bscorer()` was deliberately
        not given its own method in `behave_cscore.py` ("a thin one-line
        delegation ... call `.get_bscorer()` (i.e. the `behave_cscore`
        instance's own `_bscorer` attribute) directly instead", that
        module's own docstring).

        `_tolerance`/`_bias`/`_row_bias` (ensemble.cc:45-58) are set up
        for real now, past the guard: `_tolerance` estimates accumulated
        float-rounding error across the bscore ("Typical values should be
        about 1e-5 for floats, 1e-14 for doubles", ensemble.cc:47-49's own
        comment -- `epsilon_score` here IS `FLT_EPSILON`, matching the
        "for floats" case, since `_bscorer`'s bscore is `score_t`/`float`-
        valued); `_row_bias` is populated ONLY when `not ep.exact_experts`
        (ensemble.cc:56-58), left as an empty list otherwise, matching
        C++'s own conditional `std::vector<double>` construction exactly
        (an untouched `_row_bias` member is empty-by-default-construction
        in C++ too).
        """
        self._params: ensemble_parameters = ep if ep is not None else ensemble_parameters()
        self._bscorer = cs._bscorer
        self._scored_trees: scored_combo_tree_set = scored_combo_tree_set()
        self._weighted_tree: combo_tree | None = None

        if not self._params.do_boosting:
            return

        self._bscorer.reset_weights()

        bslen = self._bscorer.size()
        self._tolerance = 2.0 * epsilon_score
        self._tolerance *= math.sqrt(float(bslen))

        self._bias = 0.0
        self._row_bias: list[float] = []
        if not self._params.exact_experts:
            self._row_bias = [0.0] * bslen

    def _get_weight(self, sct: scored_combo_tree) -> float:
        """C++: moses/moses/moses/types.h:291 scored_combo_tree::get_weight

        A pass-through, kept only so this file reads like classic's, which
        calls `sct.get_weight()` directly at every one of these sites.

        This used to be a lookup in a private `_sct_weights` dict keyed by
        `id(sct)`, because `scored_combo_tree` carried no weight of its
        own. It does now, so the shadow storage is gone: it made the
        weight invisible to everything outside this class -- in particular
        to `metapop_printer`'s `output_ensemble` branch, which prints
        `sct.get_weight()` per member and would have printed the 1.0
        default for every tree while boosting believed otherwise.
        """
        return sct.get_weight()

    def _set_weight(self, sct: scored_combo_tree, weight: float) -> None:
        """C++: moses/moses/moses/types.h:295 scored_combo_tree::set_weight

        See `_get_weight` above.
        """
        sct.set_weight(weight)

    def add_candidates(self, cands: scored_combo_tree_set) -> None:
        """C++: moses/moses/metapopulation/ensemble.cc:79 ensemble::add_candidates

        "Implement a boosted ensemble. Candidate combo trees are added to
        the ensemble one at a time, weights are adjusted, and etc."
        (ensemble.cc:74-76's own comment.)
        """
        if self._params.experts:
            self.add_expert(cands)
            return
        self.add_adaboost(cands)

    def add_adaboost(self, cands: scored_combo_tree_set) -> None:
        """C++: moses/moses/metapopulation/ensemble.cc:90 ensemble::add_adaboost

        "Implement a boosted ensemble, using the classic AdaBoost algo."
        (ensemble.cc:87-89's own comment.) `cands` is mutated in place
        (`cands.erase(best_p)`), matching C++'s own `scored_combo_tree_
        set&` reference parameter.

        `std::min_element(cands.begin(), cands.end(), [](a, b){ return
        a.get_score() > b.get_score(); })` (ensemble.cc:95-97) -- the
        SAME manual left-to-right, first-occurrence-wins translation
        `diversity.py`'s `set_diversity` already established for this
        exact `std::min_element`-with-a-greater-comparator shape: scan
        once, replace the running "best" only on a STRICT improvement.
        `cands` (a `scored_combo_tree_set`) has no C++ OR Python iteration-
        order promise (types.py's own docstring) -- first-occurrence-wins
        is the correct, order-faithful reproduction of what `std::
        min_element` does over WHATEVER order either language's
        container happens to present, not a guess at what that order is.
        """
        promoted = 0

        while True:
            # DIVERGENCE(C++ moses/moses/metapopulation/ensemble.cc:98):
            # classic picks the best candidate with `std::min_element` over
            # `cands`, a `scored_combo_tree_set` -- which is a
            # `std::unordered_set` (types.h:370). `min_element` keeps the
            # FIRST element satisfying the comparator, so **on a tie the
            # winner is decided by hash-bucket iteration order**, which the
            # C++ standard does not specify.
            #
            # The loop below is a faithful `min_element`: `>` keeps the
            # first maximum, exactly as classic's comparator does. What it
            # cannot reproduce is which element comes first.
            #
            # Ties are not rare here -- MEASURED on
            # `-Hmux -k2 --boost=1 -m1000 -r1`: 4 of 4 candidates tied at
            # the top score on the first call, and 12 of 32 on the second.
            # So the choice is real, and the consequence compounds: a
            # different first member changes every subsequent row weight,
            # hence the whole ensemble. Across 18 boosting runs, 11 agree
            # with classic exactly and 7 differ in score, sometimes in the
            # port's favour (-2 where classic gets -3), which is what rules
            # out a systematic error in the arithmetic.
            #
            # DO NOT "fix" by sorting: any total order imposed here would
            # be this port inventing a rule classic does not have. See the
            # `scored_combo_tree_set` DIVERGENCE in types.py for the same
            # issue in its container form.
            best_p: scored_combo_tree | None = None
            for sct in cands:
                if best_p is None or sct.get_score() > best_p.get_score():
                    best_p = sct
            if best_p is None:
                break  # `cands` empty -- C++'s own min_element on an
                # empty range is undefined behaviour; unreachable via
                # this file's own real caller (`merging.py`'s
                # `merge_demes` never calls `add_candidates` with an
                # empty `candidates` -- it returns False first).

            err = self._bscorer.get_error(best_p.get_bscore())
            if not (0.0 <= err < 1.0):
                raise AssertionError(f"boosting score out of range; got {err}")

            if err < self._tolerance:
                # "perfect score" -- wipe out the ensemble; it is now
                # superfluous, replaced by the single best answer
                # (ensemble.cc:107-118). `.clear()` mutates `_scored_
                # trees` IN PLACE (not rebound) so a caller already
                # holding `get_ensemble()`'s return value sees the wipe
                # too -- the same "mutate, don't rebind" rule `scored_
                # combo_tree_set.clear()`'s own docstring (types.py)
                # documents for `update_best_candidates`.
                self._scored_trees.clear()
                best = _copy_sct(best_p)
                self._set_weight(best, 1.0)
                self._scored_trees.insert(best)
                self._bscorer.reset_weights()
                return

            if 0.5 <= err:
                break  # no improvement, ensemble not expanded

            alpha = 0.5 * math.log((1.0 - err) / err)
            expalpha = math.exp(alpha)
            rcpalpha = 1.0 / expalpha

            best = _copy_sct(best_p)
            self._set_weight(best, alpha)
            self._scored_trees.insert(best)

            bs = best_p.get_bscore()
            bslen = self._bscorer.size()
            weights = [rcpalpha if _is_correct(bs[i]) else expalpha for i in range(bslen)]
            self._bscorer.update_weights(weights)

            cands.erase(best_p)

            promoted += 1
            if self._params.num_to_promote <= promoted:
                break
            if cands.empty():
                break

    def add_expert(self, cands: scored_combo_tree_set) -> None:
        """C++: moses/moses/metapopulation/ensemble.cc:179 ensemble::add_expert

        "Add trees that expertly select rows ... At most,
        `_params.num_to_promote` are added." (ensemble.cc:139-150's own
        doc comment, condensed.) `cands` is NEVER mutated here (no
        `erase` call anywhere in this method, in either language) --
        plain iteration is safe.
        """
        promoted = 0
        num = 0
        for sct in cands:
            num += 1
            err = self._bscorer.get_error(sct.get_tree())

            if not (0.0 <= err + self._tolerance and err - self._tolerance <= 1.0):
                raise AssertionError(f"boosting score out of range; got {err}")

            if self._params.exact_experts:
                if self._tolerance < err:
                    continue

                copied = _copy_sct(sct)
                self._scored_trees.insert(copied)
                # No `_set_weight` call here -- matches C++'s own exact-
                # experts branch, which never calls `set_weight` either;
                # `_get_weight` on this entry returns its `1.0` default.

                expalpha = self._params.expalpha
                rcpalpha = 1.0 / expalpha

                treeset = scored_combo_tree_set()
                treeset.insert(_copy_sct(sct))
                bs = self._bscorer.call_ensemble(treeset)
                bslen = self._bscorer.size()
                weights = [rcpalpha if 0.0 < bs[i] else expalpha for i in range(bslen)]
                self._bscorer.update_weights(weights)
            else:
                # "if we are here, its the in-exact experts code.  XXX
                # currently broken ..." (ensemble.cc:216) -- see module
                # docstring's own "DO NOT FIX" section for exactly what
                # happens and why it is ported this way.
                if 0.5 <= err:
                    continue
                if err < self._tolerance:
                    err = self._tolerance
                alpha = 0.5 * math.log((1.0 - err) / err)

                kopy = _copy_sct(sct)
                self._set_weight(kopy, alpha)
                self._scored_trees.insert(kopy)

                raise AssertionError("this doesn't work right now.")

            promoted += 1
            if self._params.num_to_promote <= promoted:
                break

    def get_ensemble(self) -> scored_combo_tree_set:
        """C++: moses/moses/metapopulation/ensemble.h:68 ensemble::get_ensemble

        `const scored_combo_tree_set& get_ensemble() const { return
        _scored_trees; }` -- trivial accessor.
        """
        return self._scored_trees

    def get_weighted_tree(self) -> combo_tree:
        """C++: moses/moses/metapopulation/ensemble.cc:280 ensemble::get_weighted_tree

        "Return the ensemble contents as a single, large tree." Dispatches
        to `get_adaboost_tree`/`get_exact_tree`/`get_expert_tree`, each of
        which ALSO assigns `self._weighted_tree` as a side effect (matching
        `_weighted_tree` being a `mutable combo_tree` MEMBER in C++, not a
        plain return value) -- this method discards each getter's own
        return and reads `self._weighted_tree` back, exactly mirroring
        C++'s own body (`get_adaboost_tree(); ... return _weighted_tree;`).

        # DIVERGENCE(C++ moses/moses/metapopulation/ensemble.h:78):
        # `_weighted_tree` is `mutable combo_tree` -- a single member C++
        # clears and rebuilds IN PLACE on every call, so two `get_
        # weighted_tree()` calls in a row return the SAME object
        # reference (mutated between calls). Each getter below instead
        # constructs a FRESH `combo_tree` and rebinds `self._weighted_
        # tree` to it, so two separate calls never share Python object
        # IDENTITY, only VALUE (structural) equality. No caller anywhere
        # in this port's current scope holds a `get_weighted_tree()`
        # reference across a SECOND call expecting the first one to
        # update alongside it (the method is not currently invoked by
        # `metapopulation.py`/`merging.py` at all outside direct test
        # calls), so this has no observable effect today -- flagged for
        # whoever's future work might change that.
        """
        if self._params.experts:
            if self._params.exact_experts:
                self.get_exact_tree()
            else:
                self.get_expert_tree()
        else:
            self.get_adaboost_tree()
        return self._weighted_tree

    def get_adaboost_tree(self) -> combo_tree:
        """C++: moses/moses/metapopulation/ensemble.cc:298 ensemble::get_adaboost_tree

        "Returns the combo tree expressing (sum_i weight_i * (tree_i ?
        1.0 : -1.0)) > 0), i.e. true if the summation is positive, else
        false, as per standard AdaBoost definition." (ensemble.cc:288-296's
        own doc comment, condensed.)
        """
        from pymetta.comboreduct.combo.vertex import builtin
        from pymetta.comboreduct.combo.vertex import combo_tree as _combo_tree

        if len(self._scored_trees) == 1:
            only = next(iter(self._scored_trees))
            self._weighted_tree = _combo_tree(only.get_tree().begin())
            return self._weighted_tree

        tree = _combo_tree(builtin.greater_than_zero)
        head = tree.begin()
        plus = tree.append_child(head, builtin.plus)

        for sct in self._scored_trees:
            # times is (weight * (tree - 0.5))
            times = tree.append_child(plus, builtin.times)
            tree.append_child(times, self._get_weight(sct))

            # minus is (tree - 0.5): +0.5 if tree is true, -0.5 if false
            minus = tree.append_child(times, builtin.plus)
            tree.append_child(minus, -0.5)
            impulse = tree.append_child(minus, builtin.impulse)
            tree.append_child(impulse, sct.get_tree().begin())

        self._weighted_tree = tree
        return self._weighted_tree

    def get_exact_tree(self) -> combo_tree:
        """C++: moses/moses/metapopulation/ensemble.cc:361 ensemble::get_exact_tree

        "Returns the combo tree expressing or_i tree_i, i.e. true if any
        member of the ensemble picked true." (ensemble.cc:327-332's own
        doc comment.) The commented-out `reduct::logical_reduce` call
        (ensemble.cc:349-351, "Very surprisingly, reduct does not seem to
        make a differrence! Hmmm.") is dead code even in C++ -- not
        ported, same "a commented-out C++ call has no observable
        behaviour to reproduce" convention this whole project applies.
        """
        from pymetta.comboreduct.combo.vertex import builtin
        from pymetta.comboreduct.combo.vertex import combo_tree as _combo_tree

        if len(self._scored_trees) == 1:
            only = next(iter(self._scored_trees))
            self._weighted_tree = _combo_tree(only.get_tree().begin())
            return self._weighted_tree

        tree = _combo_tree(builtin.logical_or)
        head = tree.begin()
        for sct in self._scored_trees:
            tree.append_child(head, sct.get_tree().begin())

        self._weighted_tree = tree
        return self._weighted_tree

    def get_expert_tree(self) -> combo_tree:
        """C++: moses/moses/metapopulation/ensemble.cc:391 ensemble::get_expert_tree

        "Returns the combo tree expressing (sum_i weight_i * (tree_i ?
        1.0 : 0.0)) > _bias), i.e. true if the summation is greater than
        the bias, else false." (ensemble.cc:354-359's own doc comment,
        condensed.)
        """
        from pymetta.comboreduct.combo.vertex import builtin
        from pymetta.comboreduct.combo.vertex import combo_tree as _combo_tree

        if len(self._scored_trees) == 1:
            only = next(iter(self._scored_trees))
            self._weighted_tree = _combo_tree(only.get_tree().begin())
            return self._weighted_tree

        tree = _combo_tree(builtin.greater_than_zero)
        head = tree.begin()
        plus = tree.append_child(head, builtin.plus)

        # score must be better than the bias.
        bias = -self._bias * self._params.bias_scale
        tree.append_child(plus, bias)

        for sct in self._scored_trees:
            # times is (weight * tree)
            times = tree.append_child(plus, builtin.times)
            tree.append_child(times, self._get_weight(sct))

            # impulse is +1.0 if tree is true, else 0.0
            impulse = tree.append_child(times, builtin.impulse)
            tree.append_child(impulse, sct.get_tree().begin())

        self._weighted_tree = tree
        return self._weighted_tree

    def flat_score(self) -> score_t:
        """C++: moses/moses/metapopulation/ensemble.cc:427 ensemble::flat_score

        "Return the plain, unweighted, 'flat' score for the ensemble as a
        whole ... always driven to be 50% wrong, on average [for the
        WEIGHTED score]." (ensemble.cc:419-426's own doc comment,
        condensed.) `boost::accumulate(bs, 0.0)` -- double accumulator,
        narrowed once at return, the same convention `scoring_base.py`'s
        `sum_bscore` (unweighted branch) already follows for an identical
        C++ call shape.
        """
        bs = self._bscorer.call_ensemble(self._scored_trees)
        total = 0.0
        for x in bs:
            total += x
        return _to_score_t(total)
