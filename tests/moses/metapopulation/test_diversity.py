"""Unit tests for pymetta.moses.metapopulation.diversity.

No cxxtest oracle covers this file (`metapopulationUTest.cxxtest`'s two
tests are `domination.cc`'s own -- see test_domination.py's module
docstring); every scenario below is hand-computed against the real
formulas `diversity.py` ports (`p_norm_distance`, `dst2dp`,
`aggregated_dps`), not against a captured C++ trajectory. The empty-cache
degenerate stats (`_gather_stats_over([])`) ARE measured against a
standalone boost::accumulators probe -- see `diversity.py`'s own module
docstring for the exact command and output.

`test_set_diversity_matches_classics_own_defaults` uses `diversity_
parameters()` with ONLY `pressure` set -- `exponent` (-1.0, "max"),
`normalize` (True), `dst` (p_norm/Euclidean, p=2.0) and `dst2dp`
(inverse) all stay at classic's REAL constructor defaults, per this
project's rule 3 (an oracle configured by the port cannot detect the
port's configuration being wrong).
"""

from __future__ import annotations

import math
import sys

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.moses.metapopulation import merging  # noqa: F401 (binds diversity onto metapopulation)
from pymetta.moses.metapopulation.diversity import _gather_stats_over
from pymetta.moses.metapopulation.metapop_params import diversity_parameters, metapop_parameters
from pymetta.moses.metapopulation.metapopulation import cached_dst, metapopulation
from pymetta.moses.moses.types import _to_score_t, composite_score, scored_combo_tree
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.scoring_base import bscore_base


class _FixedBscore(bscore_base):
    """Not itself a ported C++ symbol -- same shape as test_domination.py's
    own `_FixedBscore`/test_metapopulation.py's `_ProbeBscore`: a
    controllable per-tree bscore table, keyed by printed tree form."""

    def __init__(self, scores: dict[str, list[float]]) -> None:
        super().__init__()
        self._scores = scores

    def __call__(self, tr):
        from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str

        return list(self._scores.get(combo_tree_to_str(tr).strip(), []))

    def best_possible_bscore(self):
        return [0.0]


def _sct(label: str, score: float, bs: list[float], complexity_penalty: float = 0.0):
    return scored_combo_tree(
        str_to_combo_tree(label), cs=composite_score(score, 0, complexity_penalty), bs=list(bs)
    )


# ---------------------------------------------------------------------------
# _gather_stats_over -- boost::accumulators reproduction
# ---------------------------------------------------------------------------


def test_gather_stats_over_empty_matches_measured_boost_degenerate_case():
    """Measured directly (diversity.py's own module docstring): an empty
    boost::accumulators accumulator_set<double, stats<count, mean,
    variance, min, max>> returns count=0, mean=nan, variance=0 (so
    std=0, NOT nan), min=DBL_MAX, max=-DBL_MAX."""
    stats = _gather_stats_over([])
    assert stats.count == 0.0
    assert math.isnan(stats.mean)
    assert stats.std == 0.0
    assert stats.min == sys.float_info.max
    assert stats.max == -sys.float_info.max


def test_gather_stats_over_nonempty_matches_measured_boost_values():
    """Measured directly against the same probe with 3 real samples
    pushed (1.0, 2.0, 4.0): count=3, mean=2.3333333333, variance=
    1.5555555556 (population variance, divide by N not N-1), min=1,
    max=4."""
    stats = _gather_stats_over([1.0, 2.0, 4.0])
    assert stats.count == 3.0
    assert stats.mean == 7.0 / 3.0
    assert stats.std == math.sqrt(14.0 / 9.0)
    assert stats.min == 1.0
    assert stats.max == 4.0


# ---------------------------------------------------------------------------
# cached_dst -- no-cache-in-any-real-build behaviour
# ---------------------------------------------------------------------------


def test_cached_dst_call_computes_directly_without_caching():
    dparams = diversity_parameters()  # default dst: p_norm, Euclidean (p=2.0)
    cd = cached_dst(dparams)
    a = _sct("$1", 1.0, [0.0, 0.0])
    b = _sct("$2", 1.0, [3.0, 4.0])

    dist = cd(a, b)

    assert dist == 5.0  # 3-4-5 triangle, Euclidean
    assert cd.cache == {}  # ENABLE_DST_CACHE is undefined -- never populated
    assert cd.hits == 0
    assert cd.misses == 0


def test_cached_dst_erase_ptr_seq_is_a_no_op():
    cd = cached_dst(diversity_parameters())
    cd.cache["anything"] = 1.0  # force non-empty, to prove erase_ptr_seq touches nothing
    cd.erase_ptr_seq([])
    assert cd.cache == {"anything": 1.0}


def test_cached_dst_gather_stats_on_empty_cache_matches_measured_degenerate_case():
    cd = cached_dst(diversity_parameters())
    stats = cd.gather_stats()
    assert stats.count == 0.0
    assert math.isnan(stats.mean)
    assert stats.min == sys.float_info.max
    assert stats.max == -sys.float_info.max


# ---------------------------------------------------------------------------
# set_diversity
# ---------------------------------------------------------------------------


class _EmptyBscore(bscore_base):
    """Not itself a ported C++ symbol -- classic's own `dummy_bscore`
    shape (`metapopulationUTest.cxxtest`): a placeholder scorer used ONLY
    to satisfy `metapopulation.__init__`'s own constructor plumbing.
    `_mp` below immediately `clear()`s whatever this seeds and inserts
    hand-built `scored_combo_tree`s with fully independent, explicit
    score/bscore instead -- `set_diversity`/`gather_diversity_stats`
    never call back into `_cscorer` at all, so this scorer's own output
    is never actually read once construction finishes.
    """

    def __call__(self, tr):
        return []

    def best_possible_bscore(self):
        return [0.0]


def _mp(diversity: diversity_parameters) -> metapopulation:
    """Builds a real `metapopulation` (so `_params`/`_cached_dst` are
    properly wired to `diversity`) with an EMPTY `_scored_trees` -- the
    constructor's own single placeholder exemplar is discarded via
    `clear()` immediately, so score/bscore for every REAL test candidate
    is controlled entirely by direct `_sct(...)` + `.insert(...)` calls,
    decoupled from `_cscorer` (see `_EmptyBscore`'s own docstring for why
    a naive "score derived from bscore via a real scorer" construction
    does not give independent control over both).
    """
    pa = metapop_parameters(diversity=diversity)
    mp = metapopulation(str_to_combo_tree("$1"), behave_cscore(_EmptyBscore()), pa)
    mp.clear()
    return mp


def test_set_diversity_matches_classics_own_defaults():
    """Classic's OWN default `diversity_parameters` (only `pressure` set
    away from 0.0): exponent=-1.0 ("max", dp_max=True), normalize=True
    (unused on the dp_max path), dst=p_norm/Euclidean (p=2.0),
    dst2dp=inverse (`f(x) = pressure / (1+x)`). Hand-computed:

        A: score=10, bscore=[0.0]  (picked first, pool empty -> untouched)
        B: score=8,  bscore=[1.0]
        C: score=5,  bscore=[4.0]
        pressure=2.0

    dst2dp=inverse means `multiply_diversity` stays False (only `pthpower`
    sets it) -- but `update_penalized_score` (types.py, already-ported
    code) has an ELSE branch even in that case: `penalized_score -=
    uniformity_penalty` (types.h:166-171's own `else penalized_score -=
    uniformity_penalty;`, easy to miss reading only the `if` -- an EARLIER
    version of this test's own docstring made exactly that mistake and
    was caught by running it). So `penalized_score` for anything the loop
    actually touches IS reduced by its uniformity penalty here, just not
    MULTIPLIED. In this fixture the reduction never changes PICK order
    (each score gap is bigger than the resulting penalty), matching the
    by-hand trace below -- `test_set_diversity_pthpower_multiplies_the_
    penalized_score` below is the one that exercises multiply_diversity.

    Iter 1 (pool={}): pick A (highest score, never penalised at all).
    Iter 2 (pool={A}): dist(B,A)=1 -> dp=2/(1+1)=1.0 -> B.up=max(0,1)=1.0
                       -> B.penalized = 8 - 1.0 = 7.0.
                       dist(C,A)=4 -> dp=2/(1+4)=0.4 -> C.up=max(0,0.4)=0.4
                       -> C.penalized = 5 - 0.4 = 4.6.
                       pick B (7.0 > 4.6).
    Iter 3 (pool={A,B}): dist(C,B)=3 -> dp=2/(1+3)=0.5 ->
                         C.up=max(0.4,0.5)=0.5 -> C.penalized = 5 - 0.5 = 4.5.
                         pick C (only one left).
    """
    diversity = diversity_parameters()
    diversity.pressure = 2.0
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0]))
    mp._scored_trees.insert(_sct("$2", 8.0, [1.0]))
    mp._scored_trees.insert(_sct("$3", 5.0, [4.0]))

    mp.set_diversity()

    from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str

    by_label = {combo_tree_to_str(sct.get_tree()).strip(): sct for sct in mp.get_trees()}

    assert by_label["$1"].get_composite_score().get_uniformity_penalty() == 0.0
    assert by_label["$2"].get_composite_score().get_uniformity_penalty() == 1.0
    assert by_label["$3"].get_composite_score().get_uniformity_penalty() == 0.5

    # dst2dp=inverse never sets multiply_diversity, but the ELSE branch of
    # update_penalized_score still SUBTRACTS the uniformity penalty.
    assert by_label["$1"].get_composite_score().multiply_diversity is False
    assert by_label["$1"].get_penalized_score() == 10.0  # never touched
    assert by_label["$2"].get_penalized_score() == 7.0  # 8.0 - 1.0
    assert by_label["$3"].get_penalized_score() == 4.5  # 5.0 - 0.5


def test_set_diversity_generalized_mean_branch_when_exponent_is_positive():
    """`exponent > 0` (a real, user-settable, non-default configuration --
    `dp_max = exponent <= 0.0` is False) takes the OTHER branch:
    `v.second` accumulates (SUMS) `last_ddp` across iterations rather than
    taking the max, and `aggregated_dps` (a generalized-mean formula) --
    not the raw sum -- becomes the stored uniformity penalty. `dst2dp=
    inverse` still means `multiply_diversity` stays False, but (see the
    classic-defaults test above for why) `penalized_score` is still
    reduced by the SUBTRACT branch of `update_penalized_score`.
    Hand-computed with exponent=1.0, normalize=True (default),
    pressure=2.0, same trees as the classic-defaults test above:

    Iter 2 (pool={A}, N=1): dist(B,A)=1 -> dp=1.0 -> ddp=1.0^1=1.0 ->
                            B.sum=0+1=1.0 -> agg=(1.0/1)^(1/1)=1.0 ->
                            B.penalized = 8 - 1.0 = 7.0.
                            dist(C,A)=4 -> dp=0.4 -> ddp=0.4 -> C.sum=0.4 ->
                            agg=(0.4/1)^1=0.4 -> C.penalized = 5 - 0.4 = 4.6.
                            pick B (7.0 > 4.6).
    Iter 3 (pool={A,B}, N=2): dist(C,B)=3 -> dp=0.5 -> ddp=0.5 ->
                             C.sum=0.4+0.5=0.9 -> agg=(0.9/2)^(1/1)=0.45 ->
                             C.penalized = 5 - 0.45 = 4.55.

    `0.45` itself is not exactly representable in `score_t` (float32) --
    compared against `_to_score_t(0.45)`, the same rounded value
    `aggregated_dps`'s own narrowing would actually produce, rather than
    the exact decimal literal.
    """
    diversity = diversity_parameters()
    diversity.pressure = 2.0
    diversity.exponent = 1.0
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0]))
    mp._scored_trees.insert(_sct("$2", 8.0, [1.0]))
    mp._scored_trees.insert(_sct("$3", 5.0, [4.0]))

    mp.set_diversity()

    from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str

    by_label = {combo_tree_to_str(sct.get_tree()).strip(): sct for sct in mp.get_trees()}

    assert by_label["$1"].get_composite_score().get_uniformity_penalty() == 0.0
    assert by_label["$2"].get_composite_score().get_uniformity_penalty() == 1.0
    assert by_label["$3"].get_composite_score().get_uniformity_penalty() == _to_score_t(0.45)

    assert by_label["$1"].get_penalized_score() == 10.0
    assert by_label["$2"].get_penalized_score() == 7.0
    assert by_label["$3"].get_penalized_score() == _to_score_t(5.0 - _to_score_t(0.45))


def test_set_diversity_pthpower_multiplies_the_penalized_score():
    """`dst2dp_type == pthpower` is the ONE configuration that sets
    `multiply_diversity = True` (diversity.cc:116-117) -- the only case
    where the uniformity penalty actually changes `penalized_score`
    (`composite_score.update_penalized_score`, already-ported code).
    Hand-computed: A (score=10, bscore=[0.0], picked first, untouched);
    B (score=4, bscore=[9.0]). `dst2dp_type=pthpower` means `dst2dp(x) =
    pow(x, pressure)`, pressure=2.0: dist(B,A)=9.0 -> dp=9.0**2=81.0 ->
    dp_max (exponent=-1.0 default) -> ddp=81.0 -> B.up=max(0,81)=81.0 ->
    multiply_diversity=True -> penalized_score = (4.0 - 0) * 81.0 = 324.0.
    """
    diversity = diversity_parameters()
    diversity.pressure = 2.0
    diversity.set_dst2dp(diversity.dst2dp_enum_t.pthpower)
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0]))
    mp._scored_trees.insert(_sct("$2", 4.0, [9.0]))

    mp.set_diversity()

    from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str

    by_label = {combo_tree_to_str(sct.get_tree()).strip(): sct for sct in mp.get_trees()}

    b_csc = by_label["$2"].get_composite_score()
    assert b_csc.multiply_diversity is True
    assert b_csc.get_uniformity_penalty() == 81.0
    assert b_csc.get_penalized_score() == 324.0

    a_csc = by_label["$1"].get_composite_score()
    assert a_csc.multiply_diversity is False
    assert a_csc.get_penalized_score() == 10.0


def test_set_diversity_raises_when_a_pool_member_has_no_bscore():
    """`OC_ASSERT(bsct.get_bscore().size(), "Behavioral score is needed
    for diversity!")` (diversity.cc:84-85) -- ported as `AssertionError`.

    THREE candidates, not two: the empty-bscore one ("$3") must survive
    to be checked as `bsct` itself in a LATER iteration (the check only
    ever runs on the element CURRENTLY being updated, never on
    `last_ptr` -- a real classic hazard this test does not exercise: if
    the FIRST-picked element itself had an empty bscore, no check would
    ever catch it, and `_cached_dst`'s own `p_norm_distance` would raise
    a length-mismatch `ValueError` instead once some LATER candidate
    tried to measure its distance to it). "$1" (score 10, picked first,
    pool empty, never checked) and "$2" (score 8, valid 1-D bscore,
    checked and passes against last_ptr="$1") both go first; "$3" (score
    5, bscore=[]) is checked last and must raise before ever reaching
    `_cached_dst`.
    """
    diversity = diversity_parameters()
    diversity.pressure = 1.0
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0]))
    mp._scored_trees.insert(_sct("$2", 8.0, [1.0]))
    mp._scored_trees.insert(_sct("$3", 5.0, []))

    try:
        mp.set_diversity()
    except AssertionError as exc:
        assert "Behavioral score is needed for diversity" in str(exc)
    else:
        raise AssertionError("expected set_diversity to raise for an empty bscore")


# ---------------------------------------------------------------------------
# gather_diversity_stats
# ---------------------------------------------------------------------------


def test_gather_diversity_stats_n_non_negative_computes_real_pairwise_distances():
    """n >= 0: pairwise Euclidean distances among the top `min(n, size())`
    candidates, in `_scored_trees`'s own `sct_score_greater` order (best
    score first). A=[0,0] score 10 (best), B=[3,4] score 8, C=[0,5] score
    5. Top 2 = {A, B} (highest two scores) -> exactly ONE pair -> dist(A,B)
    = 5.0 (3-4-5 triangle) -> count=1, mean=5.0, std=0.0 (a single sample
    has zero population variance), min=max=5.0."""
    diversity = diversity_parameters()  # pressure stays 0.0 -- unused by gather_diversity_stats
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0, 0.0]))
    mp._scored_trees.insert(_sct("$2", 8.0, [3.0, 4.0]))
    mp._scored_trees.insert(_sct("$3", 5.0, [0.0, 5.0]))

    stats = mp.gather_diversity_stats(2)

    assert stats.count == 1.0
    assert stats.mean == 5.0
    assert stats.std == 0.0
    assert stats.min == 5.0
    assert stats.max == 5.0


def test_gather_diversity_stats_n_negative_is_always_the_measured_empty_cache_case():
    """`n < 0` delegates to `_cached_dst.gather_stats()`, which is ALWAYS
    the measured empty-cache degenerate case in every build this port
    targets (diversity.py's own module docstring) -- regardless of how
    many real candidates are in the pool, since the cache itself is never
    populated by `__call__`. Pinned explicitly because it is genuinely
    counter-intuitive: a caller might expect `n=-1` ("all candidates") to
    give the SAME answer `n=size()` would."""
    diversity = diversity_parameters()
    mp = _mp(diversity)
    mp._scored_trees.insert(_sct("$1", 10.0, [0.0, 0.0]))
    mp._scored_trees.insert(_sct("$2", 8.0, [3.0, 4.0]))

    stats = mp.gather_diversity_stats(-1)

    assert stats.count == 0.0
    assert math.isnan(stats.mean)
    assert stats.min == sys.float_info.max
    assert stats.max == -sys.float_info.max
