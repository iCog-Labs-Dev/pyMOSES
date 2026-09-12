"""Unit + mutation-provable tests for pymoses.moses.metapopulation.ensemble.

P2 metapopulation task: `ensemble.py` went from a deliberately minimal
non-boosting stub to the full AdaBoost/experts port (`ensemble.py`'s own
module docstring). Every test below that used to pin `NotImplementedError`
for a boosting call is renamed/rewritten to test the real behaviour
instead (not deleted) -- see each one's own docstring for exactly what
changed.

No cxxtest oracle covers `ensemble.cc` at all (`metapopulationUTest`'s two
tests are both `domination.cc`'s own, per CARRY-FORWARD.md), so every
scenario here is hand-computed against the real ported formulas, the same
bar `test_diversity.py`/`test_subsample.py` (same P2 task, no cxxtest
coverage either) already set.

## `_BoostableBscore` -- a real `get_error`, not `demo_problems.py`'s

`bscore_base.get_error` has no body at all in this port (`ensemble.py`'s
own module docstring: classic's own BASE class is itself an unconditional
`OC_ASSERT(false, ...)`, so a concrete override is required in EITHER
language). `demo_problems.py`'s `logical_bscore` already provides a real
one, but driving it needs that module's own CLI-oriented `demo_params`/
type-signature apparatus for no benefit here: `_BoostableBscore` below
implements the IDENTICAL formula `logical_bscore.get_error` uses
(`bscores.cc:130-133`: `-sum_bscore(bs) / size`) directly against a
SCRIPTED per-tree bscore dict, so every scenario's expected `err`/`alpha`/
weight values can be hand-computed exactly and asserted precisely.
"""

from __future__ import annotations

import math

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.moses.metapopulation.ensemble import ensemble
from pymoses.moses.metapopulation.ensemble_params import ensemble_parameters
from pymoses.moses.moses.types import composite_score, scored_combo_tree, scored_combo_tree_set
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.scoring_base import bscore_base


class _DummyBscore(bscore_base):
    """Minimal concrete bscore_base -- mirrors metapopulationUTest's own
    `dummy_bscore` (tests/moses/metapopulationUTest.cxxtest). No
    `get_error` override: fine for every non-boosting-driven test below,
    since `get_error` is only ever called from `add_adaboost`/
    `add_expert`, both unreachable while `do_boosting=False`.
    """

    def __call__(self, tr):
        return []

    def best_possible_bscore(self):
        return [0.0]


class _BoostableBscore(bscore_base):
    """Not itself a ported C++ symbol -- see module docstring. `bscores`
    maps a combo tree's own string form to its SCRIPTED behavioral score
    (boolean-style: `0.0` correct, `-1.0` incorrect, matching `ensemble.
    cc`'s own `is_correct` convention exactly, the same convention
    `logical_bscore`'s own bscore values use).
    """

    def __init__(self, bscores: dict[str, list[float]], size: int) -> None:
        super().__init__()
        self._bscores = bscores
        self._size = size
        # Real callers always call `use_weighted_scores()` on the bscorer
        # BEFORE constructing `behave_cscore`/`ensemble` when boosting is
        # on (`table_problems.py`'s own `if pms.meta_params.do_boosting:
        # bscore.use_weighted_scores()`, done ahead of `behave_cscore(...)`
        # -- this port's own real wiring, not invented here). Every use of
        # this test double in this file is a boosting scenario, so this
        # mirrors that real call site unconditionally rather than
        # repeating it at every one of this file's own construction
        # points.
        self.use_weighted_scores()

    def __call__(self, tr):
        return list(self._bscores[combo_tree_to_str(tr).strip()])

    def best_possible_bscore(self):
        return [0.0] * self._size

    def get_error(self, bs_or_tr):
        """C++: moses/moses/scoring/bscores.cc:130 logical_bscore::get_error

        Handles BOTH C++ overloads (`get_error(const behavioral_score&)`,
        used by `add_adaboost`; `get_error(const combo_tree&)`, used by
        `add_expert`) by dispatching on the argument's own type -- a
        `list` is already a bscore, anything else is a tree to be scored
        first via `self(tr)` (mirroring the commented-out default body
        `bscore_base::get_error(const combo_tree&)` itself documents:
        "return get_error(operator()(tr))", scoring_base.cc:129).
        """
        bs = bs_or_tr if isinstance(bs_or_tr, list) else self(bs_or_tr)
        return -self.sum_bscore(bs) / float(self._size)

    def call_ensemble(self, ensemble_set):
        """C++: moses/moses/scoring/bscores.cc:85 logical_bscore::operator()
        (scored_combo_tree_set overload)

        `bscore_base.call_ensemble` unconditionally raises in this port
        (no concrete scorer anywhere overrides it yet -- REQUIRED
        FOLLOW-UP, see `ensemble.py`'s own module docstring: every real
        C++ scorer this project has a partial port of -- `logical_bscore`,
        `ctruth_table_bscore`, `select_bscore`, `precision_bscore` --
        overrides `operator()(const scored_combo_tree_set&)` for real in
        classic, but none has been given one in this port). This test
        double's own override is a DELIBERATE SIMPLIFICATION, not a
        faithful port of `logical_bscore`'s real weighted-vote formula:
        the real one needs `sct.get_weight()` per member
        (`bscores.cc:90`), which `scored_combo_tree` cannot supply in
        this port either (the SAME gap `ensemble.py`'s own module
        docstring documents) -- an unweighted sum of each member's own
        scripted bscore is sufficient for every scenario this file
        actually drives through it (both call sites below use a
        SINGLE-member ensemble, where "weighted" and "unweighted" are
        identical).
        """
        result = [0.0] * self._size
        for sct in ensemble_set:
            bs = self(sct.get_tree())
            for i in range(self._size):
                result[i] += bs[i]
        return result


def _sct(tree_str: str, score: float, bscore: list[float]) -> scored_combo_tree:
    return scored_combo_tree(str_to_combo_tree(tree_str), cs=composite_score(score, 0), bs=bscore)


@pytest.fixture
def cscore():
    return behave_cscore(_DummyBscore())


# ---------------------------------------------------------- construction / gate


def test_ensemble_constructs_unconditionally_on_default_params(cscore):
    # metapopulation.cc:41 constructs _ensemble(sc, pa.ensemble_params)
    # regardless of do_boosting -- this must not raise.
    ens = ensemble(cscore)
    assert ens.get_ensemble().empty()


def test_ensemble_constructs_with_explicit_non_boosting_params(cscore):
    ens = ensemble(cscore, ensemble_parameters())
    assert ens.get_ensemble().empty()


def test_ensemble_do_boosting_true_no_longer_raises_sets_up_real_state(cscore):
    """P2 metapopulation task: `do_boosting=True` construction is now
    real -- this used to be `test_ensemble_raises_when_constructed_with_
    do_boosting_true`, pinning `NotImplementedError`. Now asserts the
    REAL setup (ensemble.cc:44-58): `_tolerance == 2*epsilon_score*
    sqrt(bslen)`, `_bias == 0.0`, `_row_bias` populated per `exact_
    experts`. `_DummyBscore` never sets `_size` away from `bscore_base.
    __init__`'s own default `0`, so `bslen == 0` and `_row_bias == []`
    regardless of `exact_experts` here.
    """
    ep = ensemble_parameters()
    ep.do_boosting = True
    ens = ensemble(cscore, ep)
    assert ens._tolerance == pytest.approx(0.0)  # 2*eps*sqrt(0) == 0
    assert ens._bias == 0.0
    assert ens._row_bias == []
    assert ens.get_ensemble().empty()


def test_ensemble_row_bias_empty_when_exact_experts():
    """`_row_bias` is populated ONLY when `not ep.exact_experts`
    (ensemble.cc:56-58) -- `exact_experts` defaults `True`
    (`ensemble_params.py`), so the default-boosting-on construction
    leaves `_row_bias` empty regardless of `bslen`.
    """
    bsc = _BoostableBscore({}, size=5)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ens = ensemble(cs, ep)
    assert ens._row_bias == []


def test_ensemble_row_bias_populated_when_not_exact_experts():
    bsc = _BoostableBscore({}, size=5)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.exact_experts = False
    ens = ensemble(cs, ep)
    assert ens._row_bias == [0.0] * 5


def test_ensemble_stores_the_underlying_bscorer(cscore):
    ens = ensemble(cscore)
    assert ens._bscorer is cscore._bscorer


# --------------------------------------------------------------- get_ensemble


def test_get_ensemble_is_read_only_accessor_and_never_raises(cscore):
    ens = ensemble(cscore)
    assert ens.get_ensemble().empty()
    assert ens.get_ensemble() is ens.get_ensemble()  # same object each call


# --------------------------------------------------------------- add_adaboost


def _adaboost_ensemble(
    bscores: dict[str, list[float]], size: int, num_to_promote: int = 1
) -> ensemble:
    bsc = _BoostableBscore(bscores, size)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.num_to_promote = num_to_promote
    return ensemble(cs, ep)


def test_add_adaboost_perfect_score_wipes_and_replaces_ensemble():
    """One candidate, all-correct bscore (`err == 0 < _tolerance`) --
    ensemble.cc:107-118's "perfect score" branch: `_scored_trees` is
    wiped (even if it already held something) and replaced by exactly
    this one candidate at weight 1.0; scorer weights reset.
    """
    ens = _adaboost_ensemble({"$1": [0.0, 0.0]}, size=2, num_to_promote=5)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, [0.0, 0.0]))
    ens.add_adaboost(cands)

    result = list(ens.get_ensemble())
    assert len(result) == 1
    assert combo_tree_to_str(result[0].get_tree()).strip() == "$1"
    assert ens._get_weight(result[0]) == 1.0


def test_add_adaboost_promotes_with_real_alpha_and_updates_weights():
    """One candidate, bscore `[0,0,0,-1]` (err == 0.25, strictly between
    0 and 0.5 -- promotable, not perfect): `alpha = 0.5*ln(3) ~= 0.5493`,
    `expalpha = 3**0.5 ~= 1.732`, `rcpalpha = 1/expalpha ~= 0.5774`.
    `num_to_promote=1` caps at exactly one promotion.
    """
    bs = [0.0, 0.0, 0.0, -1.0]
    ens = _adaboost_ensemble({"$1": bs}, size=4, num_to_promote=1)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs))
    ens.add_adaboost(cands)

    result = list(ens.get_ensemble())
    assert len(result) == 1
    expected_alpha = 0.5 * math.log(3.0)
    assert ens._get_weight(result[0]) == pytest.approx(expected_alpha)

    expalpha = math.exp(expected_alpha)
    rcpalpha = 1.0 / expalpha
    assert ens._bscorer._weights == pytest.approx([rcpalpha, rcpalpha, rcpalpha, expalpha])
    assert cands.empty()  # erased after promotion


def test_add_adaboost_no_improvement_breaks_without_expanding():
    """`err >= 0.5` -- "no improvement, ensemble not expanded"
    (ensemble.cc:126-129): the loop breaks immediately, `_scored_trees`
    stays empty, `cands` is left UNTOUCHED (no erase before the break).
    """
    bs = [0.0, -1.0]  # err == 0.5
    ens = _adaboost_ensemble({"$1": bs}, size=2, num_to_promote=5)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs))
    ens.add_adaboost(cands)

    assert ens.get_ensemble().empty()
    assert not cands.empty()


def test_add_adaboost_promotes_two_then_a_perfect_third_wipes_them():
    """Three candidates promoted across three loop iterations, selected
    in DESCENDING score order (10, 5, then whatever remains): "$1"
    (score 10, err 0.25 -- promoted), "!$1" (score 5, err 0.25 --
    promoted), "true" (score 1, err 0 -- PERFECT, wipes BOTH prior
    promotions). `num_to_promote=5` never caps this on its own.
    """
    bs1 = [0.0, 0.0, 0.0, -1.0]  # err 0.25
    bs2 = [0.0, 0.0, -1.0, 0.0]  # err 0.25
    bs3 = [0.0, 0.0, 0.0, 0.0]  # err 0.0 -- perfect
    ens = _adaboost_ensemble({"$1": bs1, "!$1": bs2, "true": bs3}, size=4, num_to_promote=5)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs1))
    cands.insert(_sct("!$1", 5.0, bs2))
    cands.insert(_sct("true", 1.0, bs3))

    ens.add_adaboost(cands)

    result = list(ens.get_ensemble())
    assert len(result) == 1
    assert combo_tree_to_str(result[0].get_tree()).strip() == "true"
    assert ens._get_weight(result[0]) == 1.0


def test_add_adaboost_promotes_both_when_neither_is_perfect():
    """Two candidates, BOTH promotable, `num_to_promote=2`: both survive
    in `_scored_trees`. Their alphas are NOT equal despite both having
    the SAME naive (uniform-weight) error of 0.25: `update_weights`
    RESCALES the bscorer's own weights after the FIRST promotion
    (`_weights[i] *= rew[i]`, `scoring_base.py`'s own `update_weights`),
    so the second candidate's error is computed against the ALREADY-
    REWEIGHTED bscorer, not the original uniform one -- the entire point
    of adaptive boosting. Hand-derivation, done by actually tracing the
    weighted `sum_bscore` formula rather than assuming symmetry (an
    earlier draft of this test assumed both alphas equal and was WRONG --
    caught by running it, not by inspection, the exact hazard this
    project's own CARRY-FORWARD.md Sec 6 warns about):

        after promoting "$1" (bs1, naive err 0.25): weights become
        [rcpalpha1, rcpalpha1, rcpalpha1, expalpha1] where
        rcpalpha1 = 1/sqrt(3) ~= 0.57735, expalpha1 = sqrt(3) ~= 1.73205
        (multiplied onto the prior uniform [1,1,1,1] -- same values).
        "!$1"'s bs2 = [0,0,-1,0]: weighted sum_bscore =
        -1 * rcpalpha1 ~= -0.57735, err2 = 0.57735/4 ~= 0.14434,
        alpha2 = 0.5*ln((1-err2)/err2) ~= 0.88986.
    """
    bs1 = [0.0, 0.0, 0.0, -1.0]
    bs2 = [0.0, 0.0, -1.0, 0.0]
    ens = _adaboost_ensemble({"$1": bs1, "!$1": bs2}, size=4, num_to_promote=2)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs1))
    cands.insert(_sct("!$1", 5.0, bs2))

    ens.add_adaboost(cands)

    weight_by_tree = {
        combo_tree_to_str(sct.get_tree()).strip(): ens._get_weight(sct)
        for sct in ens.get_ensemble()
    }
    assert set(weight_by_tree) == {"$1", "!$1"}

    alpha1 = 0.5 * math.log(3.0)
    rcpalpha1 = 1.0 / math.exp(alpha1)
    err2 = rcpalpha1 / 4.0
    alpha2 = 0.5 * math.log((1.0 - err2) / err2)
    assert weight_by_tree["$1"] == pytest.approx(alpha1)
    assert weight_by_tree["!$1"] == pytest.approx(alpha2)


def test_add_adaboost_out_of_range_error_raises():
    """`OC_ASSERT(0.0 <= err and err < 1.0, ...)` (ensemble.cc:99) --
    ported as `AssertionError`. A bscore whose `sum_bscore` is POSITIVE
    (`err` negative) violates this.
    """
    bs = [1.0, 1.0]  # sum_bscore == 2.0 -> err == -1.0, out of [0, 1)
    ens = _adaboost_ensemble({"$1": bs}, size=2)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs))
    with pytest.raises(AssertionError):
        ens.add_adaboost(cands)


# ----------------------------------------------------------------- add_expert


def test_add_expert_exact_mode_admits_only_perfect_trees():
    """`exact_experts=True` (the default): only a tree within `_tolerance`
    of a perfect score (`err <= _tolerance`) is admitted; everything else
    is skipped (`continue`), `_scored_trees` stays untouched for it.
    """
    bsc = _BoostableBscore({"$1": [0.0, 0.0], "!$1": [0.0, -1.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))  # perfect -- admitted
    cands.insert(_sct("!$1", 1.0, [0.0, -1.0]))  # err 0.5 -- skipped

    ens.add_expert(cands)

    result = {combo_tree_to_str(sct.get_tree()).strip() for sct in ens.get_ensemble()}
    assert result == {"$1"}
    only = next(iter(ens.get_ensemble()))
    assert ens._get_weight(only) == 1.0  # never set_weight'd -- C++'s own default


def test_add_expert_exact_mode_updates_weights_via_call_ensemble():
    """After admitting a perfect tree, weights are recomputed via
    `_bscorer.operator()(treeset)` (`call_ensemble` here) over a
    SINGLETON set holding just that tree: `weight[i] = rcpalpha if bs[i]
    > 0 else expalpha`, with `expalpha = ensemble_parameters.expalpha`
    (default `2.0`). `_BoostableBscore.__call__` always returns the same
    scripted bscore for a given tree string, so this is directly
    checkable.
    """
    bsc = _BoostableBscore({"$1": [0.0, 0.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    ens.add_expert(cands)

    expalpha = ep.expalpha  # 2.0
    # bs == [0.0, 0.0]: neither element is > 0.0 -> both weights == expalpha
    assert ens._bscorer._weights == pytest.approx([expalpha, expalpha])


def test_add_expert_num_to_promote_caps_admissions():
    bsc = _BoostableBscore({"$1": [0.0, 0.0], "!$1": [0.0, 0.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.num_to_promote = 1
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    cands.insert(_sct("!$1", 1.0, [0.0, 0.0]))
    ens.add_expert(cands)

    assert len(ens.get_ensemble()) == 1


def test_add_expert_inexact_mode_crashes_after_mutating_scored_trees():
    """`exact_experts=False` -- ensemble.cc's own inexact-experts branch
    is an UNCONDITIONAL `OC_ASSERT(false, "this doesn't work right
    now.")`, reached AFTER `_scored_trees.insert(kopy)` has already run
    for real (see ensemble.py's own module docstring, "DO NOT FIX"
    section). Both facts are asserted: the crash (`AssertionError`) AND
    that `_scored_trees` already holds the mutated entry when it fires.
    """
    bs = [0.0, 0.0, 0.0, -1.0]  # err 0.25 -- promotable, not perfect, not terrible
    bsc = _BoostableBscore({"$1": bs}, size=4)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.exact_experts = False
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, bs))

    with pytest.raises(AssertionError, match="this doesn't work right now"):
        ens.add_expert(cands)

    result = list(ens.get_ensemble())
    assert len(result) == 1
    assert combo_tree_to_str(result[0].get_tree()).strip() == "$1"
    expected_alpha = 0.5 * math.log(3.0)
    assert ens._get_weight(result[0]) == pytest.approx(expected_alpha)


def test_add_expert_inexact_mode_skips_terrible_trees_before_crashing():
    """`0.5 <= err` -- "terrible precision, ensemble not expanded"
    (ensemble.cc:242-245) -- a `continue`, reached BEFORE the crash site,
    so a sufficiently bad tree in inexact mode is simply skipped, not a
    crash.
    """
    bsc = _BoostableBscore({"$1": [0.0, -1.0]}, size=2)  # err 0.5
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.exact_experts = False
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, -1.0]))

    ens.add_expert(cands)  # must NOT raise
    assert ens.get_ensemble().empty()


# ---------------------------------------------------------- add_candidates


def test_add_candidates_dispatches_to_add_adaboost_by_default():
    bs = [0.0, 0.0]
    ens = _adaboost_ensemble({"$1": bs}, size=2)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, bs))
    ens.add_candidates(cands)
    assert len(ens.get_ensemble()) == 1


def test_add_candidates_dispatches_to_add_expert_when_experts_true():
    bsc = _BoostableBscore({"$1": [0.0, 0.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ens = ensemble(cs, ep)

    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    ens.add_candidates(cands)

    assert len(ens.get_ensemble()) == 1
    # add_expert never erases from cands -- add_adaboost always does for
    # a promoted candidate. This distinguishes the two dispatch targets
    # observably, not just by which one happened to run.
    assert not cands.empty()


# ------------------------------------------------------- weighted-tree getters


def test_get_adaboost_tree_single_member_is_a_bare_copy():
    ens = _adaboost_ensemble({"$1": [0.0, 0.0]}, size=2, num_to_promote=5)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    ens.add_adaboost(cands)

    tree = ens.get_adaboost_tree()
    assert combo_tree_to_str(tree).strip() == "$1"
    # NOT `is` -- see ensemble.py's own DIVERGENCE note on `_weighted_tree`:
    # this port builds a FRESH tree object per call rather than mutating
    # one shared member in place, so two separate calls never share
    # identity even though C++'s own `_weighted_tree` reference would.
    assert combo_tree_to_str(ens.get_weighted_tree()).strip() == "$1"


def test_get_adaboost_tree_multi_member_shape():
    """`greater_than_zero(plus(times(w1, plus(-0.5, impulse(t1))), times(w2,
    plus(-0.5, impulse(t2)))))` (ensemble.cc:298-326): root is `greater_
    than_zero` with a single `plus` child, under which sit exactly as
    many `times` children as ensemble members.
    """
    bs1 = [0.0, 0.0, 0.0, -1.0]
    bs2 = [0.0, 0.0, -1.0, 0.0]
    ens = _adaboost_ensemble({"$1": bs1, "!$1": bs2}, size=4, num_to_promote=2)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 10.0, bs1))
    cands.insert(_sct("!$1", 5.0, bs2))
    ens.add_adaboost(cands)

    tree = ens.get_adaboost_tree()
    assert tree.begin().data is builtin.greater_than_zero
    plus_node = tree.begin().first_child
    assert plus_node.data is builtin.plus

    child = plus_node.first_child
    times_count = 0
    while child is not None:
        assert child.data is builtin.times
        times_count += 1
        child = child.next_sibling
    assert times_count == 2


def test_get_exact_tree_multi_member_is_logical_or():
    bsc = _BoostableBscore({"$1": [0.0, 0.0], "!$1": [0.0, 0.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.num_to_promote = 5
    ens = ensemble(cs, ep)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    cands.insert(_sct("!$1", 1.0, [0.0, 0.0]))
    ens.add_expert(cands)

    tree = ens.get_weighted_tree()
    assert tree.begin().data is builtin.logical_or

    child_count = 0
    node = tree.begin().first_child
    while node is not None:
        child_count += 1
        node = node.next_sibling
    assert child_count == 2


def test_get_expert_tree_single_member_is_a_bare_copy():
    """Single-member shortcut (ensemble.cc:365-368) fires before the bias
    is ever read -- checked with the SAME inexact-mode-crash scenario
    `test_add_expert_inexact_mode_crashes_after_mutating_scored_trees`
    uses, since that is the only way this file gets a non-`exact_experts`
    ensemble with exactly one member without also proving the bias-
    accumulation code (dead in C++ itself -- see that test's own
    docstring).
    """
    bs = [0.0, 0.0, 0.0, -1.0]  # err 0.25 -- promotable, not terrible
    bsc = _BoostableBscore({"$1": bs}, size=4)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.exact_experts = False
    ens = ensemble(cs, ep)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, bs))
    with pytest.raises(AssertionError):
        ens.add_expert(cands)

    tree = ens.get_expert_tree()
    assert combo_tree_to_str(tree).strip() == "$1"


def test_get_expert_tree_multi_member_includes_bias_child():
    """Two EXACT experts (`_bias` never touched by the exact-experts
    branch, so it stays `0.0` -- `bias = -0.0 * bias_scale == -0.0`) --
    checks the multi-member shape includes a leading bias child under
    `plus`, distinct from every subsequent `times` child.
    """
    bsc = _BoostableBscore({"$1": [0.0, 0.0], "!$1": [0.0, 0.0]}, size=2)
    cs = behave_cscore(bsc)
    ep = ensemble_parameters()
    ep.do_boosting = True
    ep.experts = True
    ep.num_to_promote = 5
    ens = ensemble(cs, ep)
    cands = scored_combo_tree_set()
    cands.insert(_sct("$1", 1.0, [0.0, 0.0]))
    cands.insert(_sct("!$1", 1.0, [0.0, 0.0]))
    ens.add_expert(cands)

    tree = ens.get_expert_tree()
    assert tree.begin().data is builtin.greater_than_zero
    plus_node = tree.begin().first_child
    assert plus_node.data is builtin.plus

    bias_child = plus_node.first_child
    assert bias_child.data == pytest.approx(0.0)
    times_count = 0
    node = bias_child.next_sibling
    while node is not None:
        assert node.data is builtin.times
        times_count += 1
        node = node.next_sibling
    assert times_count == 2


def test_flat_score_sums_call_ensemble_output():
    bsc = _BoostableBscore({"$1": [0.0, -0.5]}, size=2)
    cs = behave_cscore(bsc)
    ens = ensemble(cs)  # do_boosting default False -- flat_score itself never gated
    ens._scored_trees.insert(_sct("$1", 1.0, [0.0, -0.5]))
    assert ens.flat_score() == pytest.approx(-0.5)
