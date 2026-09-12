"""Unit + RNG-order-oracle tests for pymoses.moses.metapopulation.subsample.

`metapopulationUTest.cxxtest` has no coverage of `subsample.cc` at all
(CARRY-FORWARD.md/p2-metapop-report.md: its two tests are both
`domination.cc`'s own). No cxxtest oracle exists for this file anywhere in
classic -- every scenario here is either a real `tools/cpp_probe/
subsample_selector_probe.cc` oracle (the RNG draw-order claim) or a
hand-computed fixture (everything else), the same bar `test_diversity.py`
(P2 metapopulation, prior task) set for a file with no cxxtest coverage
either.

## `ctruth_table_bscore` cannot back `get_ctable()`/`get_ctable_usize()`
## in THIS port -- required follow-up, not edited here

Real classic `ctruth_table_bscore` extends `bscore_ctable_base`
(`bscores.h:360`, `struct ctruth_table_bscore : public bscore_ctable_base`),
which is where `get_ctable()`/`get_ctable_usize()`'s real bodies live
(`scoring_base.cc:318,323`). THIS port's `bscores.py` (owned by a
different task/agent, `pymoses/moses/scoring/`, not touched here) ports
`ctruth_table_bscore` as extending `bscore_base` DIRECTLY instead (that
module's own docstring: "NOT `bscore_ctable_base`") -- a disclosed
divergence from an earlier task, but one with a real consequence THIS
task is the first to hit: a `ctruth_table_bscore` instance in this port
has no `get_ctable`/`get_ctable_usize` at all (`bscore_base` itself
defines neither), so `subsample.py`'s `ss_average_agreement`/
`ss_tanimoto_stats`/`ss_tanimoto_filter`/`ss_filter`'s `n_best_bfdemes`
branch -- every one of which calls `self._cscorer.get_ctable()` --
`AttributeError` against any REAL `ctruth_table_bscore`-backed
`behave_cscore` today. Every test below that needs a CTable uses
`_CTableBscore`, a minimal test-local double providing both methods
directly, INSTEAD of `ctruth_table_bscore` -- not because `subsample.py`
is wrong, but because its one real CTable-backed scorer in this port
cannot supply what it needs. Reported in the task's own final report as a
required follow-up for whoever next touches `bscores.py`
(a one-line fix: either have `ctruth_table_bscore` extend
`bscore_ctable_base`, matching classic, or add its own thin `get_ctable`/
`get_ctable_usize` override) -- not fixed here, since `pymoses/moses/
scoring/` is outside this task's owned scope.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table import CTable
from pymoses.moses.metapopulation import merging  # noqa: F401 (binds subsample.* et al.)
from pymoses.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymoses.moses.metapopulation.metapopulation import metapopulation
from pymoses.moses.metapopulation.subsample import (
    _decompose_tuple,
    _pow_to_unsigned,
    _score_mean,
    _score_variance_stat,
    _TanimotoAcc,
)
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.scoring_base import bscore_base
from pymoses.util.lazy_random_selector import lazy_random_selector
from pymoses.util.rng import MT19937RandGen

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "subsample_selector.json"
_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "subsample_selector.manifest.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())
_MANIFEST = json.loads(_MANIFEST_PATH.read_text())


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Deme(list):
    """Same shape as test_merging.py's own `_Deme` (deliberately
    duplicated rather than imported -- a 5-line test double, not worth a
    cross-test-file import): a bare `list` of `scored_instance`s with a
    settable `n_evals`.
    """

    n_evals = 0


class _FakeRepByIdentity:
    """Same shape as test_merging.py's own -- see that file's docstring."""

    def __init__(self, tree_by_instance_id: dict[int, object]) -> None:
        self._tree_by_instance_id = tree_by_instance_id

    def get_candidate(self, inst: instance, reduce: bool = True) -> object:
        return self._tree_by_instance_id[id(inst)]


class _CTableBscore(bscore_base):
    """Not itself a ported C++ symbol -- see this module's own docstring
    for why a `ctruth_table_bscore` cannot stand in here. `__call__`/
    `best_possible_bscore` are never exercised by anything in this file
    (`subsample.cc`'s own functions never call the bscorer as a function,
    only `.get_ctable()`/`.get_ctable_usize()`), so both are trivial
    stubs satisfying `bscore_base`'s abstract-method contract.
    """

    def __init__(self, ctable: CTable) -> None:
        super().__init__()
        self._ctable = ctable

    def __call__(self, tr):
        return []

    def best_possible_bscore(self):
        return []

    def get_ctable(self) -> CTable:
        return self._ctable

    def get_ctable_usize(self) -> float:
        return sum(sum(counter.values()) for counter in self._ctable.values())


def _bool_ctable(rows: list[tuple[bool, float]]) -> CTable:
    """One boolean input column (`$1`), one row per `(value, weight)` pair
    -- exactly what `ss_average_agreement`/`ss_tanimoto_stats` iterate via
    `self._cscorer.get_ctable().items()`.
    """
    ct = CTable("y", ["$1"])
    for value, weight in rows:
        key = (builtin.logical_true if value else builtin.logical_false,)
        ct[key] = Counter({builtin.logical_true: weight})
    return ct


def _mp_with(ctable: CTable | None = None, filter_params=None) -> metapopulation:
    bsc = _CTableBscore(ctable if ctable is not None else CTable("y", ["$1"]))
    cscorer = behave_cscore(bsc)
    params = metapop_parameters()
    fp = filter_params if filter_params is not None else subsample_deme_filter_parameters()
    return metapopulation(str_to_combo_tree("$1"), cscorer, params, fp)


def _one_candidate_deme(tree_str: str, rep_map: dict[int, object]) -> _Deme:
    deme = _Deme([scored_instance(instance(), composite_score(1.0, 0))])
    rep_map[id(deme[0].first)] = str_to_combo_tree(tree_str)
    return deme


# ---------------------------------------------------------------------------
# RNG draw-order oracle -- draw ORDER against a real opencog::
# lazy_random_selector + opencog::MT19937RandGen, not merely a resulting
# set (this task's own acceptance criterion).
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("config_name", list(_GOLDEN["configs"].keys()))
def test_lazy_random_selector_draw_order_matches_cpp_probe(config_name):
    """Replays EXACTLY the construction/draw/decompose loop `subsample.
    py`'s own `ss_score_dev_filter`/`ss_average_agreement`/`ss_tanimoto_
    stats` share (`lazy_random_selector(n_possible_tuples, rng)`, drawn
    `sample_tuples` times via `selector()`, each draw decomposed via
    `_decompose_tuple`) against `tools/cpp_probe/subsample_selector_probe.
    cc`'s real `opencog::lazy_random_selector` + `opencog::MT19937RandGen`
    output, asserting equality of BOTH the raw draw sequence and the
    decomposed tuple indices -- draw order, not merely the resulting set.
    Includes two exhaustive (`sample_tuples == n_possible_tuples`)
    heavy-collision configs and two zero/one-arity edge configs (see the
    probe's own header comment for the full list and rationale).
    """
    cfg = _GOLDEN["configs"][config_name]
    rng = MT19937RandGen(cfg["seed"])
    selector = lazy_random_selector(cfg["n_possible_tuples"], rng)

    draws = []
    tuple_idxs = []
    for _ in range(cfg["sample_tuples"]):
        d = selector()
        draws.append(d)
        tuple_idxs.append(_decompose_tuple(d, cfg["n_demes"], cfg["base"]))

    assert draws == cfg["draws"]
    assert tuple_idxs == cfg["tuple_idxs"]


def test_subsample_selector_fixture_manifest_matches_current_probe_source():
    """`tools/check_fixture_manifests.py` covers this in CI; asserted here
    too so a stale fixture fails LOUDLY inside this test file, not just a
    separate script someone has to remember to run."""
    from tools._manifest import sha256_of

    probe_path = _REPO_ROOT / "tools" / "cpp_probe" / "subsample_selector_probe.cc"
    assert _MANIFEST["probe_source_sha256"] == sha256_of(probe_path)


# ---------------------------------------------------------------------------
# _pow_to_unsigned / _decompose_tuple
# ---------------------------------------------------------------------------


def test_pow_to_unsigned_matches_exact_integer_power():
    assert _pow_to_unsigned(4, 3) == 64
    assert _pow_to_unsigned(1, 5) == 1  # C++'s own pow(1, n) == 1
    assert _pow_to_unsigned(0, 0) == 1  # IEEE-754 pow(0,0) == 1, matched
    assert _pow_to_unsigned(0, 3) == 0
    assert _pow_to_unsigned(10, 2) == 100


def test_decompose_tuple_little_endian_base_n():
    # 51 in base 4, 3 digits, little-endian (least-significant first):
    # 51 = 3*16 + 0*4 + 3*1
    assert _decompose_tuple(51, 3, 4) == [3, 0, 3]
    assert _decompose_tuple(0, 3, 4) == [0, 0, 0]
    assert _decompose_tuple(5, 0, 4) == []  # n_demes == 0 -> always []


# ---------------------------------------------------------------------------
# _TanimotoAcc
# ---------------------------------------------------------------------------


def test_tanimoto_acc_hand_computed():
    acc = _TanimotoAcc()
    for x in (0.5, 0.25, 0.1):
        acc.add(x)
    assert acc.count() == 3
    assert acc.mean() == pytest.approx((0.5 + 0.25 + 0.1) / 3)
    # geometric_mean_mirror: 1 - cbrt((1-.5)(1-.25)(1-.1))
    expected_gmm = 1 - (0.5 * 0.75 * 0.9) ** (1.0 / 3.0)
    assert acc.geometric_mean_mirror() == pytest.approx(expected_gmm)
    assert acc.max() == 0.5


def test_tanimoto_acc_empty_mean_is_nan_via_ieee_div():
    acc = _TanimotoAcc()
    assert math.isnan(acc.mean())
    assert acc.max() == -math.inf


# ---------------------------------------------------------------------------
# _score_variance_stat / _score_mean -- population variance, Sample=float
# ---------------------------------------------------------------------------


def test_score_variance_stat_population_formula():
    mean, variance = _score_variance_stat([0.0, 10.0])
    assert mean == 5.0
    assert variance == 25.0  # population (/N), not sample (/N-1, would be 50.0)


def test_score_mean_hand_computed():
    assert _score_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ss_score_dev_filter -- direct, both deterministic (n_top_candidates=1)
# and RNG-driven (n_top_candidates>1) shapes.
# ---------------------------------------------------------------------------


def _ss_params(**overrides) -> subsample_deme_filter_parameters:
    fp = subsample_deme_filter_parameters(
        overrides.pop("n_subsample_demes", 2), overrides.pop("n_top_candidates", 1)
    )
    fp.std_dev_threshold = 3.4028234663852886e38
    fp.tanimoto_mean_threshold = 1.0
    fp.tanimoto_geo_mean_threshold = 1.0
    fp.tanimoto_max_threshold = 1.0
    fp.n_best_bfdemes = 0
    fp.tanimoto_mean_weight = 0.0
    fp.tanimoto_geo_mean_weight = 0.0
    fp.tanimoto_max_weight = 0.0
    for k, v in overrides.items():
        setattr(fp, k, v)
    return fp


def test_ss_score_dev_filter_deterministic_pass():
    mp = _mp_with(filter_params=_ss_params(std_dev_threshold=1.0))
    ss_demes = [
        _Deme([scored_instance(instance(), composite_score(5.0, 0))]),
        _Deme([scored_instance(instance(), composite_score(5.0, 0))]),
    ]
    assert mp.ss_score_dev_filter(None, ss_demes) is True


def test_ss_score_dev_filter_deterministic_fail():
    mp = _mp_with(filter_params=_ss_params(std_dev_threshold=1.0))
    ss_demes = [
        _Deme([scored_instance(instance(), composite_score(0.0, 0))]),
        _Deme([scored_instance(instance(), composite_score(10.0, 0))]),
    ]
    assert mp.ss_score_dev_filter(None, ss_demes) is False


def test_ss_score_dev_filter_boundary_is_strict_less_than():
    """`pass = mean_std < std_dev_threshold` (subsample.cc:187) -- STRICT
    inequality, not `<=`. Scores (0.0, 10.0): mean 5.0, population
    variance ((0-5)**2+(10-5)**2)/2 == 25.0, std EXACTLY 5.0. Setting
    `std_dev_threshold` to that exact value must FAIL.
    """
    mp = _mp_with(filter_params=_ss_params(std_dev_threshold=5.0))
    ss_demes = [
        _Deme([scored_instance(instance(), composite_score(0.0, 0))]),
        _Deme([scored_instance(instance(), composite_score(10.0, 0))]),
    ]
    assert mp.ss_score_dev_filter(None, ss_demes) is False


def test_ss_score_dev_filter_samples_via_rng_not_always_index_zero():
    """`n_top_candidates=2`: the tuple space is `2**2 == 4`, and
    `n_tuples` (unbounded, `_UINT_MAX` default) forces `sample_tuples ==
    4` -- an EXHAUSTIVE draw, so every one of the 4 possible tuples is
    visited regardless of RNG seed (Floyd's algorithm never repeats, and
    `sample_tuples == n_possible_tuples` visits the whole range). Proves
    the loop actually walks index 1 as well as index 0 -- not merely
    always reading `ss_deme[0]` -- by making index 0 and index 1 disagree
    sharply (0.0 vs. 20.0 in EACH deme) and checking the computed
    mean-of-stds sits between the two tuples' own stds (both tuples excluding
    the "both-zero"/"both-20" ones contribute a nonzero std).
    """
    mp = _mp_with(filter_params=_ss_params(n_top_candidates=2, std_dev_threshold=100.0))
    ss_demes = [
        _Deme(
            [
                scored_instance(instance(), composite_score(0.0, 0)),
                scored_instance(instance(), composite_score(20.0, 0)),
            ]
        ),
        _Deme(
            [
                scored_instance(instance(), composite_score(0.0, 0)),
                scored_instance(instance(), composite_score(20.0, 0)),
            ]
        ),
    ]
    # Hand-computed over all 4 tuples: (0,0)->std 0; (0,1)->std 10;
    # (1,0)->std 10; (1,1)->std 0. Mean of stds == 5.0.
    assert mp.ss_score_dev_filter(None, ss_demes) is True  # 5.0 < 100.0
    # And a threshold strictly between 0 and 5 must fail, proving the
    # mean is neither 0 (index-0-only) nor computed some other way.
    mp2 = _mp_with(filter_params=_ss_params(n_top_candidates=2, std_dev_threshold=1.0))
    assert mp2.ss_score_dev_filter(None, ss_demes) is False


# ---------------------------------------------------------------------------
# ss_average_agreement -- dead code in classic itself; direct unit test.
# ---------------------------------------------------------------------------


def test_ss_average_agreement_hand_computed():
    """Two rows, weight 1.0 each. Two ss_demes, `n_top_candidates=1` (the
    tuple space is exactly {0}, deterministic): deme 0's sole candidate is
    `"$1"`, deme 1's is `"!$1"` -- perfectly DISAGREEING trees. Row 1
    ($1=True): $1 says true (+1), !$1 says false (-1) -> agreement 0.
    Row 2 ($1=False): $1 says false (-1), !$1 says true (+1) -> agreement
    0. `normalized_agreement` is 0 for both rows -> `mean_agreement ==
    0.0`.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    mp = _mp_with(ctable, filter_params=_ss_params(n_top_candidates=1))
    rep_map: dict[int, object] = {}
    ss_demes = [
        _one_candidate_deme("$1", rep_map),
        _one_candidate_deme("!$1", rep_map),
    ]
    rep = _FakeRepByIdentity(rep_map)
    assert mp.ss_average_agreement(rep, ss_demes) == pytest.approx(0.0)


def test_ss_average_agreement_full_agreement():
    """Same shape, but BOTH ss_demes hold `"$1"` -- perfect agreement on
    every row -> `|agreement/exponent| == 1.0` for both rows ->
    `mean_agreement == 1.0`.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 2.0)])
    mp = _mp_with(ctable, filter_params=_ss_params(n_top_candidates=1))
    rep_map: dict[int, object] = {}
    ss_demes = [
        _one_candidate_deme("$1", rep_map),
        _one_candidate_deme("$1", rep_map),
    ]
    rep = _FakeRepByIdentity(rep_map)
    assert mp.ss_average_agreement(rep, ss_demes) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ss_tanimoto_stats / ss_tanimoto_filter
# ---------------------------------------------------------------------------


def test_ss_tanimoto_stats_trees_identical_trees_zero_distance():
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    mp = _mp_with(ctable)
    tr = str_to_combo_tree("$1")
    acc = _TanimotoAcc()
    mp._ss_tanimoto_stats_trees([tr, tr], acc)
    assert acc.count() == 1
    assert acc.mean() == pytest.approx(0.0)


def test_ss_tanimoto_stats_trees_opposite_trees_max_distance():
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    mp = _mp_with(ctable)
    tr1 = str_to_combo_tree("$1")
    tr2 = str_to_combo_tree("!$1")
    acc = _TanimotoAcc()
    mp._ss_tanimoto_stats_trees([tr1, tr2], acc)
    assert acc.count() == 1
    assert acc.mean() == pytest.approx(1.0)


def test_ss_tanimoto_stats_trees_asymmetric_pair_hand_computed():
    """`"$1"` (weighted activations `[1.0, 0.0]`, true only on row 1) vs.
    `"true"` (a constant, always-true tree -- weighted activations
    `[1.0, 1.0]`): `ab=1, aa=1, bb=2, numerator=aa+bb-ab=2, distance = 1 -
    ab/numerator == 0.5`. Distinguishes a `vertex_equal(result, builtin.
    logical_true)` polarity bug from a correct one -- unlike the
    identical-trees and perfectly-opposite-trees cases above, which stay
    0.0/1.0 respectively under an inverted true/false rule (the mutation
    just swaps which of two SYMMETRIC vectors is which; tanimoto_distance
    doesn't care), THIS pair is asymmetric enough that inverting the rule
    changes the computed distance to 1.0.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    mp = _mp_with(ctable)
    tr1 = str_to_combo_tree("$1")
    tr2 = str_to_combo_tree("true")
    acc = _TanimotoAcc()
    mp._ss_tanimoto_stats_trees([tr1, tr2], acc)
    assert acc.count() == 1
    assert acc.mean() == pytest.approx(0.5)


def test_ss_tanimoto_filter_pass_and_fail():
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    rep_map: dict[int, object] = {}

    identical = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("$1", rep_map)]
    opposite = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("!$1", rep_map)]
    rep = _FakeRepByIdentity(rep_map)

    mp = _mp_with(ctable, filter_params=_ss_params(n_top_candidates=1, tanimoto_mean_threshold=0.5))
    assert mp.ss_tanimoto_filter(rep, identical) is True  # 0.0 < 0.5
    assert mp.ss_tanimoto_filter(rep, opposite) is False  # 1.0 not < 0.5


def test_ss_tanimoto_filter_requires_all_three_thresholds_not_just_one():
    """`return mean_pass and geo_mean_pass and max_pass` (subsample.cc:
    461) -- ALL three, not any one. Three ss_demes decoding to `"$1"`,
    `"$1"`, `"!$1"` give pairwise distances [0.0, 1.0, 1.0]: mean ==
    0.6667 (passes a 0.7 threshold), but geometric_mean_mirror == 1.0 and
    max == 1.0 (both computed by hand: `gmm = 1 -
    ((1-0)*(1-1)*(1-1))**(1/3) == 1 - 0 == 1.0` since one pairwise
    distance is exactly 1.0) -- both FAIL a 0.5 threshold. The overall
    filter must fail despite the mean alone passing.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    fp = _ss_params(n_top_candidates=1)
    fp.tanimoto_mean_threshold = 0.7
    fp.tanimoto_geo_mean_threshold = 0.5
    fp.tanimoto_max_threshold = 0.5
    mp = _mp_with(ctable, filter_params=fp)

    rep_map: dict[int, object] = {}
    ss_demes = [
        _one_candidate_deme("$1", rep_map),
        _one_candidate_deme("$1", rep_map),
        _one_candidate_deme("!$1", rep_map),
    ]
    rep = _FakeRepByIdentity(rep_map)

    assert mp.ss_tanimoto_filter(rep, ss_demes) is False


# ---------------------------------------------------------------------------
# ss_filter -- n_best_bfdemes branch (needs a real CTable; moved from
# test_merging.py, see this module's own docstring).
# ---------------------------------------------------------------------------


def test_ss_filter_n_best_bfdemes_keeps_lowest_aggregate_tanimoto_distance():
    """`n_best_bfdemes > 0` never calls `ss_score_dev_filter`/
    `ss_tanimoto_filter` at all -- only `ss_tanimoto_stats` (per BF-deme
    aggregated distance) then an ascending `_sort` keeping the
    `n_best_bfdemes` LOWEST. BF-index 0's two ss_demes both decode to
    `"$1"` (identical -> aggregate distance 0.0); BF-index 1's decode to
    `"$1"` and `"!$1"` (opposite -> aggregate distance 1.0, the maximum
    possible). `tanimoto_mean_weight=1.0`, the other two weights 0.0, so
    the aggregate IS the mean distance. `n_best_bfdemes=1` must keep index
    0 (lower distance) and discard index 1.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    fp = _ss_params(n_top_candidates=1)
    fp.n_best_bfdemes = 1
    fp.tanimoto_mean_weight = 1.0
    mp = _mp_with(ctable, filter_params=fp)

    rep_map: dict[int, object] = {}
    low_dist_demes = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("$1", rep_map)]
    high_dist_demes = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("!$1", rep_map)]
    rep = _FakeRepByIdentity(rep_map)

    result = mp.ss_filter([low_dist_demes, high_dist_demes], [rep, rep])
    assert result == [True, False]


def test_ss_filter_n_best_bfdemes_clamped_to_number_of_bfdemes():
    """`n_best_bfdemes = min(n_best_bfdemes, len(all_demes))`
    (subsample.cc:91-92) -- requesting more "best" demes than exist keeps
    every one of them.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    fp = _ss_params(n_top_candidates=1)
    fp.n_best_bfdemes = 99
    fp.tanimoto_mean_weight = 1.0
    mp = _mp_with(ctable, filter_params=fp)

    rep_map: dict[int, object] = {}
    demes_a = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("$1", rep_map)]
    demes_b = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("!$1", rep_map)]
    rep = _FakeRepByIdentity(rep_map)

    result = mp.ss_filter([demes_a, demes_b], [rep, rep])
    assert result == [True, True]


def test_ss_filter_falls_through_to_tanimoto_check_after_std_dev_passes():
    """`if not bfpass[i]: continue` (subsample.cc:132-136) skips the
    Tanimoto check ONLY when the std-dev filter already FAILED for this
    deme -- when it PASSES, control falls through and the Tanimoto check
    can still downgrade `bfpass[i]` to `False`. `std_dev_threshold=100.0`
    (both candidates score 1.0, std 0.0 -> trivially passes);
    `tanimoto_mean_threshold=0.5` with candidates `"$1"`/`"!$1"` (maximal
    distance 1.0, `1.0 < 0.5` is False) must still fail the deme overall.
    """
    ctable = _bool_ctable([(True, 1.0), (False, 1.0)])
    fp = _ss_params(n_top_candidates=1, std_dev_threshold=100.0)
    fp.tanimoto_mean_threshold = 0.5
    mp = _mp_with(ctable, filter_params=fp)

    rep_map: dict[int, object] = {}
    ss_demes = [_one_candidate_deme("$1", rep_map), _one_candidate_deme("!$1", rep_map)]
    rep = _FakeRepByIdentity(rep_map)

    assert mp.ss_filter([ss_demes], [rep]) == [False]
