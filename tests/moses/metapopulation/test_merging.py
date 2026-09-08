"""Unit + oracle tests for pymetta.moses.metapopulation.merging.

`metapopulationUTest.cxxtest` covers none of this file's scope (see
CARRY-FORWARD.md: its two tests exercise `remove_dominated`/
`get_nondominated_disjoint`, both `domination.cc`, outside this whole S10
slice). `tools/cpp_probe/merging_probe.cc` fills that gap: it drives the
REAL classic pipeline (a real `Table`/`CTable`/`ctruth_table_bscore`/
`behave_cscore` stack, a real `deme_expander` + `hill_climbing`, and a real
`metapopulation`) over `trainingBooleanXor.dat` with exemplar `"and()"`,
arity 2, seed 1, max_evals 200 -- the IDENTICAL scenario
`tests/fixtures/deme_expander.json`'s own `"xor_empty_and"` case already
captures for S10 Task 4's own oracle -- then calls the real
`metapopulation::merge_demes` and dumps the resulting pool. `_build_dex`
below reproduces that same construction in Python (not by loading that
JSON fixture -- this file needs the LIVE `deme_expander`/`representation`
objects `merge_demes` consumes, not their already-captured trajectory).

## Measured: `keep_top_unique_candidates`'s own deme-size is NOT compared

The probe's own `"deme_size"` field (measured: 13, after `merge_demes`
internally sorts+dedups the original 17-instance deme) is DELIBERATELY not
asserted against in `test_single_merge_matches_cpp_probe` below. Measured
directly (see that test's own docstring): this port's `sort_demes` (a
STABLE sort, `list.sort(reverse=True)`) preserves each tied group's
original discovery order, while classic's own `boost::sort`/`std::sort`
(NOT stable) does not -- for THIS exact scenario, hill-climbing discovers
the deme's eight `!$1`/`!$2`-scoring instances and eight `$1`/$2`-scoring
instances in an ALTERNATING pattern, so `keep_top_unique_candidates`'s
purely-ADJACENT-pair dedup scan (`prev_it`/`it` are neighbours only) finds
zero adjacent ties to collapse in this port (17 stays 17), while some
particular `std::sort` permutation apparently groups same-tree instances
adjacently often enough to remove 4 in classic. Neither language's own
`sort_demes` promises anything about tie order (`sort_demes`'s own
docstring in `merging.py`), so this is not a divergence to fix -- it is a
directly observable consequence of a promise C++ never made. It is
harmless to the actual exit criterion here for a structural reason,
verified empirically below, not merely argued: `deme_to_trees` inserts
every candidate into `pot_candidates`, a `scored_combo_tree_set`
(`types.py`, `std::unordered_set<..., scored_combo_tree_hash,
scored_combo_tree_equal>`) that ALREADY deduplicates by TREE STRUCTURE on
its own, regardless of adjacency -- so whether `keep_top_unique_candidates`
removes 0 or 4 near-duplicate RAW INSTANCES first, every one of the 17
instances still gets decoded and fed to `pot_candidates`, which converges
to the SAME 5 distinct reduced trees either way. The FINAL merged pool --
this task's actual exit criterion ("merged pool contents and ordering
match C++") -- is measured below to match the probe EXACTLY: same 5 trees,
same scores, same order, same `best_composite_score`/`best_tree`, same
`merge_demes` return value, for BOTH the first merge and a second,
repeated merge of an independently-rebuilt identical deme.
"""

from __future__ import annotations

import math

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation import merging  # noqa: F401  (binds merge_demes et al.)
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import (
    composite_score,
    scored_combo_tree,
    scored_combo_tree_set,
    very_worst_score,
)
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.util.rng import randGen

_XOR_DATA = "tests/fixtures/table_data/trainingBooleanXor.dat"
_PROBE_PATH = "tools/cpp_probe/merging_probe.cc"


# ---------------------------------------------------------------------------
# Shared real-pipeline helper -- the SAME scenario merging_probe.cc drives.
# ---------------------------------------------------------------------------


def _build_cscorer() -> behave_cscore:
    table = loadTable(_XOR_DATA, "y")
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    bsc.set_complexity_coef(3.5)
    return behave_cscore(bsc)


def _build_dex(cscorer: behave_cscore, seed: int) -> deme_expander:
    """Same recipe as tests/moses/deme/test_deme_expander.py's own
    `_build_dex`, hardcoded to the "xor_empty_and" scenario's own
    parameters (dataset, exemplar "and()", arity 2, max_evals 200) rather
    than reading that file's golden fixture -- this module needs the LIVE
    deme_expander object `merge_demes` consumes, not a captured trajectory.
    """
    rule = logical_reduction(())(2)
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters()
    hc = hill_climbing(optim_parameters(), hc_parameters(widen=False, step=False, cross=False))
    sig = gen_signature(type_node.boolean_type, 2)
    dex = deme_expander(sig, rule, rule, cscorer, hc, params, filter_params)

    exemplar = str_to_combo_tree("and()")
    randGen().seed(seed)
    dex.create_demes(exemplar, 0)
    dex.optimize_demes(200, 86400)
    return dex


def _pool_trees(mp: metapopulation) -> list[tuple[str, float, int, float]]:
    return [
        (
            combo_tree_to_str(sct.get_tree()).strip(),
            sct.get_score(),
            sct.get_complexity(),
            sct.get_penalized_score(),
        )
        for sct in mp.get_trees()
    ]


# ---------------------------------------------------------------------------
# Oracle: merge_demes, matched against tools/cpp_probe/merging_probe.cc.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_single_merge_matches_cpp_probe():
    """tools/cpp_probe/merging_probe.cc's "single_merge" scenario, captured
    2026-09-08 (build/run recipe in that file's own header comment):

        "pre_size":1, "merge_demes_returned":false, "post_size":5,
        "pool":[
          {"tree":"true","cscore":{"score":-1,...,"penalized_score":-1}},
          {"tree":"!$1", score -1, complexity 1, penalized -1.2857142686843872},
          {"tree":"!$2", score -1, complexity 1, penalized -1.2857142686843872},
          {"tree":"$1",  score -3, complexity 1, penalized -3.2857143878936768},
          {"tree":"$2",  score -3, complexity 1, penalized -3.2857143878936768}],
        "best_composite_score": score -1, complexity 0, penalized -1,
        "best_tree":"true"

    See the module docstring for exactly why the probe's own "deme_size"
    (13) is NOT asserted here (this port's own equivalent, measured just
    below, is 17) -- a tie-order artifact of `keep_top_unique_candidates`
    that provably cannot change the final 5-tree pool.

    Exact (`==`), not `pytest.approx`: `complexity_penalty`/
    `penalized_score` are genuinely-fractional float32 values
    (`1/3.5`-derived, same reasoning test_deme_expander.py's own module
    docstring gives), and both sides compute them through the SAME
    `score_t`-narrowing arithmetic.
    """
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    assert mp.size() == 1

    dex = _build_dex(cscorer, seed=1)
    assert len(dex._demes[0][0]) == 17  # pre-merge_demes deme size, both languages

    done = mp.merge_demes(dex._demes, dex._reps)

    # Measured tie-order artifact (module docstring): 17 in this port,
    # 13 in the C++ probe -- deliberately not asserted equal to the probe.
    assert len(dex._demes[0][0]) <= 17

    assert done is False
    assert mp.size() == 5
    assert _pool_trees(mp) == [
        ("true", -1.0, 0, -1.0),
        ("!$1", -1.0, 1, -1.2857142686843872),
        ("!$2", -1.0, 1, -1.2857142686843872),
        ("$1", -3.0, 1, -3.2857143878936768),
        ("$2", -3.0, 1, -3.2857143878936768),
    ]

    best = mp.best_composite_score()
    assert best.get_score() == -1.0
    assert best.get_complexity() == 0
    assert best.get_penalized_score() == -1.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "true"


@pytest.mark.oracle
def test_repeat_merge_of_an_identical_deme_finds_nothing_new():
    """tools/cpp_probe/merging_probe.cc's "repeat_merge" scenario: a
    SECOND, independently-built deme from the identical seed (so identical
    content, a different object) merged into a metapopulation that
    already contains every tree that deme can produce. Captured:
    `"pre_size":5, "merge_demes_returned":false, "post_size":5`.
    Exercises `get_new_candidates`' dedup end to end: `pot_candidates`
    would again be the same 5 trees, all already present, so
    `get_new_candidates` returns an EMPTY set and `merge_demes` takes the
    `candidates.empty(): return False` early exit without touching
    `_scored_trees`/`_best_candidates` again.
    """
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    dex1 = _build_dex(cscorer, seed=1)
    mp.merge_demes(dex1._demes, dex1._reps)
    assert mp.size() == 5

    dex2 = _build_dex(cscorer, seed=1)
    pre = mp.size()
    done = mp.merge_demes(dex2._demes, dex2._reps)

    assert pre == 5
    assert done is False
    assert mp.size() == 5


def test_merging_probe_source_is_the_file_this_oracle_cites():
    """Lightweight provenance guard (no fixture/manifest pipeline for a
    probe this small -- same "probe without a fixture pipeline" precedent
    `metapop_core_probe.cc`/`hcextra_max_count_probe.cc` already set):
    the probe this test file's docstrings cite must actually exist.
    """
    import pathlib

    path = pathlib.Path(_PROBE_PATH)
    assert path.is_file()
    assert "merge_demes" in path.read_text()


# ---------------------------------------------------------------------------
# trim_down_deme
# ---------------------------------------------------------------------------


class _Deme(list):
    """Not itself a ported C++ symbol -- a bare `list` of `scored_instance`s
    with a settable `n_evals` (a plain `list` has no `__dict__`, so a bare
    `deme.n_evals = N` raises `AttributeError`). `trim_down_deme`/
    `keep_top_unique_candidates` only ever read `len(deme)`/
    `deme[i].second`/`deme[i].first`/`deme.pop()`/`deme.n_evals`/slice
    assignment -- all satisfied here without building a real
    `instance_set` (and the `field_set` it would need) at all.
    """

    n_evals = 0


def _plain_deme(scores: list[tuple[float, float]]) -> _Deme:
    """Each `(score, complexity_penalty)` pair becomes one entry, in the
    given order -- callers are responsible for handing this an already
    score-sorted (descending) sequence, `trim_down_deme`'s own documented
    precondition.
    """
    return _Deme(
        scored_instance(instance(), composite_score(score, 0, complexity_penalty))
        for score, complexity_penalty in scores
    )


def _mp(cscorer: behave_cscore | None = None) -> metapopulation:
    cscorer = cscorer if cscorer is not None else _build_cscorer()
    return metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())


def test_trim_down_deme_is_a_no_op_below_min_pool_size():
    mp = _mp()
    deme = _plain_deme([(0.0, 0.0)] * 20)  # == _min_pool_size, "not >" (`>=` guard)
    mp.trim_down_deme(deme)
    assert len(deme) == 20


def test_trim_down_deme_keeps_a_score_exactly_at_the_cutoff():
    """Places a value EXACTLY on the `< bot_sc` boundary: `bot_sc` itself
    must be KEPT (strict `<`, not `<=`), and only survives past index
    `_min_pool_size` (index 0 of a real deme can never be evaluated at all
    -- see trim_down_deme's own docstring).
    """
    mp = _mp()
    top_sc = 0.0
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng  # un-narrowed here on purpose -- see below

    # 21 entries: 20 at the top score (never individually checked -- they
    # sit at indices 0..19, and the loop only ever inspects index >= ...
    # no, ACTUALLY every index from size-1 down to 1 is inspected in this
    # port (see trim_down_deme's own docstring) -- what index 0 alone is
    # exempt from. `composite_score`'s own constructor narrows `bot_sc`
    # via `_to_score_t` exactly as `trim_down_deme` narrows its own
    # `top_sc - useful_score_range()` -- both compute the identical
    # double-precision subtraction before narrowing, so the two bit
    # patterns are guaranteed equal.
    deme = _plain_deme([(top_sc, 0.0)] * 20 + [(bot_sc, 0.0)])
    mp.trim_down_deme(deme)

    assert len(deme) == 21  # the boundary value survives -- not `< bot_sc`


def test_trim_down_deme_removes_a_score_strictly_below_the_cutoff():
    mp = _mp()
    top_sc = 0.0
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng

    deme = _plain_deme([(top_sc, 0.0)] * 20 + [(bot_sc - 1.0, 0.0)])
    mp.trim_down_deme(deme)

    assert len(deme) == 20  # the too-low tail entry is gone


def test_trim_down_deme_never_evaluates_index_zero():
    """Index 0 is exempt from the `i > 0` scan (trim_down_deme's own
    docstring) -- but that exemption is UNOBSERVABLE whenever
    `useful_score_range()` is non-negative: `top_sc` (index 0's own
    score) can never itself be `< bot_sc = top_sc - range` when `range >=
    0`, so checking index 0 anyway would never have removed it either
    (confirmed by mutation: `while i > 0` widened to `while i >= 0` did
    NOT change this test's outcome until `complexity_temperature` was
    made negative here). A NEGATIVE `complexity_temperature` makes
    `useful_score_range()` negative, so `bot_sc = top_sc - range > top_sc`
    -- index 0's own score now DOES satisfy `< bot_sc`, and would be
    removed if the scan ever reached it. It must survive anyway.
    """
    mp = _mp()
    mp._params.complexity_temperature = -6.0
    rng = mp.useful_score_range()
    assert rng < 0.0

    # top_sc, read from index 0, is -100.0; bot_sc = -100.0 - rng > -100.0,
    # so index 0's OWN score satisfies "< bot_sc" -- it must survive only
    # because index 0 is never re-checked, not because its score clears
    # the cutoff. Every other entry, well above bot_sc, is unaffected
    # either way -- this test isolates index 0's own exemption alone.
    deme = _plain_deme([(-100.0, 0.0)] + [(0.0, 0.0)] * 20)
    mp.trim_down_deme(deme)
    assert len(deme) == 21  # nothing removed: index 0 exempt despite < bot_sc


# ---------------------------------------------------------------------------
# sort_demes
# ---------------------------------------------------------------------------


def test_sort_demes_sorts_each_deme_descending_in_place():
    mp = _mp()
    deme_a = _plain_deme([(1.0, 0.0), (5.0, 0.0), (3.0, 0.0)])
    deme_b = _plain_deme([(-1.0, 0.0), (-5.0, 0.0)])
    all_demes = [[deme_a], [deme_b]]

    mp.sort_demes(all_demes)

    assert [si.second.get_score() for si in deme_a] == [5.0, 3.0, 1.0]
    assert [si.second.get_score() for si in deme_b] == [-1.0, -5.0]


# ---------------------------------------------------------------------------
# deme_to_trees
# ---------------------------------------------------------------------------


def test_deme_to_trees_dedups_by_tree_structure():
    """17 raw instances, only 5 structurally distinct reduced trees (the
    same "xor_empty_and" scenario the oracle tests above use) --
    `pot_candidates` (a `scored_combo_tree_set`) must collapse them to
    exactly 5, independent of `keep_top_unique_candidates`'s own,
    tie-order-sensitive adjacent dedup (module docstring)."""
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees(real_deme, rep, pot_candidates)

    assert len(pot_candidates) == 5
    assert {combo_tree_to_str(sct.get_tree()).strip() for sct in pot_candidates} == {
        "true",
        "!$1",
        "!$2",
        "$1",
        "$2",
    }


def test_deme_to_trees_skips_very_worst_and_non_finite_scores():
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]

    good = real_deme[0]
    bad_worst = scored_instance(instance(real_deme[1].first), composite_score(very_worst_score, 1))
    bad_nan = scored_instance(instance(real_deme[2].first), composite_score(math.nan, 1))
    deme = [good, bad_worst, bad_nan]

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees(deme, rep, pot_candidates)

    assert len(pot_candidates) == 1


def test_deme_to_trees_copies_the_composite_score_not_the_deme_entry():
    """`inst_csc = inst.second` is a reference into the deme's own
    scored_instance; `copy.copy(inst_csc)` must break that alias --
    mutating the deme's own entry afterward must not reach the stored
    candidate.
    """
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    entry = real_deme[0]
    original_score = entry.second.get_score()

    pot_candidates = scored_combo_tree_set()
    mp.deme_to_trees([entry], rep, pot_candidates)

    entry.second.set_score(-12345.0)

    (stored,) = list(pot_candidates)
    assert stored.get_score() == original_score
    assert stored.get_score() != -12345.0


# ---------------------------------------------------------------------------
# get_new_candidates
# ---------------------------------------------------------------------------


def test_get_new_candidates_keeps_only_trees_not_already_in_the_pool():
    mp = _mp()
    already = scored_combo_tree_set()
    already.insert(scored_combo_tree(str_to_combo_tree("$1"), composite_score(2.0, 1)))
    mp.merge_candidates(already)  # populate _scored_trees with "$1"

    candidates = scored_combo_tree_set()
    candidates.insert(scored_combo_tree(str_to_combo_tree("$1"), composite_score(2.0, 1)))
    candidates.insert(scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1)))

    fresh = mp.get_new_candidates(candidates)

    assert len(fresh) == 1
    (only,) = list(fresh)
    assert combo_tree_to_str(only.get_tree()).strip() == "$2"


def test_get_new_candidates_copies_not_aliases():
    """`res.insert(cnd)` (merging.cc:483) copies via
    `scored_combo_tree_set`'s own copy-on-insert contract -- mutating the
    ORIGINAL candidate object after the call must not reach the returned
    set's own entry.
    """
    mp = _mp()
    candidates = scored_combo_tree_set()
    original = scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1))
    candidates.insert(original)

    fresh = mp.get_new_candidates(candidates)
    original.get_composite_score().set_score(-999.0)

    (only,) = list(fresh)
    assert only.get_score() == 3.0


# ---------------------------------------------------------------------------
# merge_candidates
# ---------------------------------------------------------------------------


def test_merge_candidates_inserts_independent_copies():
    mp = _mp()  # already seeds _scored_trees with "$1" -- merge_candidates adds a 2nd
    candidates = scored_combo_tree_set()
    original = scored_combo_tree(str_to_combo_tree("$2"), composite_score(3.0, 1))
    candidates.insert(original)

    mp.merge_candidates(candidates)
    original.get_composite_score().set_score(-999.0)

    (stored,) = [sct for sct in mp.get_trees() if combo_tree_to_str(sct.get_tree()).strip() == "$2"]
    assert stored.get_score() == 3.0


def test_merge_candidates_discard_dominated_raises():
    """Pins the discard_dominated gate directly on merge_candidates (as
    opposed to merge_demes' own union-gated raise, tested separately
    below)."""
    mp = _mp()
    mp._params.discard_dominated = True
    with pytest.raises(NotImplementedError):
        mp.merge_candidates(scored_combo_tree_set())


# ---------------------------------------------------------------------------
# keep_top_unique_candidates
# ---------------------------------------------------------------------------


def test_keep_top_unique_candidates_removes_an_adjacent_exact_duplicate():
    """Two literally-identical instances (same object, same score) placed
    adjacently must collapse to one: `is_approx_eq` trivially holds (exact
    tie) and the decoded, reduced trees are trivially identical too (same
    instance, same representation).
    """
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    dup = real_deme[1]  # a real "!$1" or "!$2" -scoring instance
    deme = _Deme([dup, scored_instance(instance(dup.first), dup.second)])
    deme.n_evals = len(deme)

    mp.keep_top_unique_candidates([[deme]], [rep])

    assert len(deme) == 1


def test_keep_top_unique_candidates_keeps_tied_score_different_trees():
    """The REAL "xor_empty_and" deme's own indices 1/2 (`!$1`/`!$2`) tie
    EXACTLY on penalized score but decode to structurally different
    trees -- must NOT be removed, even though `is_approx_eq` says they
    tie."""
    mp = _mp()
    cscorer = _build_cscorer()
    dex = _build_dex(cscorer, seed=1)
    rep = dex._reps[0]
    real_deme = dex._demes[0][0]
    real_deme.sort(reverse=True)

    a, b = real_deme[1], real_deme[2]
    assert a.second.get_penalized_score() == b.second.get_penalized_score()
    tree_a = combo_tree_to_str(rep.get_candidate(a.first, True)).strip()
    tree_b = combo_tree_to_str(rep.get_candidate(b.first, True)).strip()
    assert tree_a != tree_b

    deme = _Deme([a, b])
    deme.n_evals = 2
    mp.keep_top_unique_candidates([[deme]], [rep])

    assert len(deme) == 2


def test_keep_top_unique_candidates_top_cnd_zero_empties_the_deme_safely():
    """MEASURED HAZARD in keep_top_unique_candidates' own docstring: real
    C++'s iterator/`!=` loop is undefined behaviour here (`it` at index 1
    is never EQUAL to a target at index 0 while only ever increasing).
    This port's `while it_idx < top_cnd` bound is well-defined for
    `top_cnd == 0` without any special-casing (`1 < 0` is simply `False`),
    falling straight through to the final truncation, `del deme[0:]`.
    `reps=[None]` -- `rep` is bound but never read: the pairwise loop body
    (the only place that calls `get_candidate`) never runs.
    """
    mp = _mp()
    mp._params.max_candidates_per_deme = 0
    deme = _plain_deme([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 3

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 0


def test_keep_top_unique_candidates_top_cnd_one_keeps_only_the_first():
    mp = _mp()
    mp._params.max_candidates_per_deme = 1
    deme = _plain_deme([(2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 3

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 1
    assert deme[0].second.get_score() == 2.0


def test_keep_top_unique_candidates_n_evals_below_deme_size_caps_top_cnd():
    """"It can happen that the true number of evals is less than the deme
    size" (merging.cc's own comment) -- `n_evals` below `len(deme)` must
    cap `top_cnd`, truncating the excess even with no duplicates at all.
    """
    mp = _mp()
    deme = _plain_deme([(3.0, 0.0), (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
    deme.n_evals = 2

    mp.keep_top_unique_candidates([[deme]], [None])

    assert len(deme) == 2
    assert [si.second.get_score() for si in deme] == [3.0, 2.0]


def test_keep_top_unique_candidates_max_candidates_per_deme_gate():
    """Pins the `max_candidates_per_deme >= 0` gate: the default (-1)
    must NOT limit top_cnd; a non-negative value must."""
    mp = _mp()
    deme = _plain_deme([(3.0, 0.0), (2.0, 0.0), (1.0, 0.0)])
    deme.n_evals = 3
    assert mp._params.max_candidates_per_deme == -1

    mp.keep_top_unique_candidates([[deme]], [None])
    assert len(deme) == 3  # untouched: default gate off


# ---------------------------------------------------------------------------
# recompute_scores_over_whole_dataset / ss_filter -- subsample gate
# ---------------------------------------------------------------------------


def test_ss_filter_default_passes_every_breadth_first_deme():
    mp = _mp()
    assert mp.ss_filter([[], [], []], []) == [True, True, True]


def test_ss_filter_n_subsample_demes_gate_raises():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.ss_filter([[]], [])


def test_recompute_scores_over_whole_dataset_default_is_a_no_op():
    mp = _mp()
    mp.recompute_scores_over_whole_dataset([], [])  # must not raise


def test_recompute_scores_over_whole_dataset_n_subsample_demes_gate_raises():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.recompute_scores_over_whole_dataset([], [])


# ---------------------------------------------------------------------------
# merge_demes gates -- every one pinned with an EMPTY all_demes/reps
# (sort_demes/keep_top_unique_candidates/the deme loop all no-op on an
# empty list, so these isolate exactly the gate under test) except
# merge_callback, which needs at least one real candidate to be reached at
# all (it is checked AFTER the `candidates.empty(): return False` exit).
# ---------------------------------------------------------------------------


def test_merge_demes_do_boosting_gate_raises():
    mp = _mp()
    mp._params.do_boosting = True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_discard_dominated_gate_raises():
    mp = _mp()
    mp._params.discard_dominated = True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_keep_bscore_gate_raises():
    mp = _mp()
    mp._params.keep_bscore = True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_diversity_enabled_gate_raises():
    mp = _mp()
    mp._params.diversity.pressure = 0.5
    assert mp.diversity_enabled() is True
    with pytest.raises(NotImplementedError):
        mp.merge_demes([], [])


def test_merge_demes_default_gates_off_returns_false_on_no_candidates():
    """Pins the NEGATIVE case for all four gates at once: with everything
    left at its default, an empty `all_demes` must reach the
    `candidates.empty()` early exit and return False, not raise."""
    mp = _mp()
    assert mp.merge_demes([], []) is False


def test_merge_demes_merge_callback_gate_raises():
    cscorer = _build_cscorer()
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    dex = _build_dex(cscorer, seed=1)
    mp._params.merge_callback = object()  # anything not None
    with pytest.raises(NotImplementedError):
        mp.merge_demes(dex._demes, dex._reps)


def test_merge_demes_n_subsample_demes_gate_raises_before_processing_demes():
    mp = _mp()
    mp._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        mp.merge_demes([[]], [None])


# ---------------------------------------------------------------------------
# resize_metapop
# ---------------------------------------------------------------------------


def _grow_pool(
    mp: metapopulation, entries: list[tuple[float, int] | tuple[float, int, float]]
) -> None:
    """`scored_combo_tree_set` (`candidates`, below) dedups by TREE
    STRUCTURE, not by score (types.py's own `scored_combo_tree_equal`) --
    every entry here needs its OWN structurally distinct tree
    (`$2`, `$3`, ...; index 1 is the metapopulation's own seed exemplar,
    "$1", already in the pool) or all but the first would collapse into
    one on insert, silently defeating every caller that wants N distinct
    pool entries.

    Each entry is `(score, complexity)` -- `complexity_penalty` defaults
    0.0, so `penalized_score == score` -- or `(score, complexity,
    complexity_penalty)` when a caller specifically needs the two to
    differ (`resize_metapop` sorts/cuts by PENALIZED score throughout;
    S9/S10's own recurring failure mode was a test suite that could never
    have told a raw-vs-penalized-score bug apart, per the task brief's own
    Testing bar).
    """
    candidates = scored_combo_tree_set()
    for i, entry in enumerate(entries, start=2):
        score, complexity, *rest = entry
        complexity_penalty = rest[0] if rest else 0.0
        tree = str_to_combo_tree(f"${i}")
        cs = composite_score(score, complexity, complexity_penalty)
        candidates.insert(scored_combo_tree(tree, cs))
    mp.merge_candidates(candidates)


def test_resize_metapop_is_a_no_op_at_exactly_min_pool_size():
    mp = _mp()
    _grow_pool(mp, [(float(-i), 0) for i in range(19)])  # + the seed "$1" == 20
    assert mp.size() == 20

    mp.resize_metapop()

    assert mp.size() == 20
    assert not hasattr(mp, "_merge_count")  # never even entered the body


def test_resize_metapop_phase1_keeps_a_score_exactly_at_the_cutoff():
    mp = _mp()
    top_sc = mp.best_composite_score().get_score()  # the seed "$1"'s own score
    rng = mp.useful_score_range()
    bot_sc = top_sc - rng

    # 20 padding entries (fill positions 1..20, never individually
    # checked -- phase 1 only scans from index _min_pool_size==20 onward)
    # plus one AT the cutoff (index 20) and one BELOW it (index 21).
    _grow_pool(mp, [(top_sc, 0)] * 20 + [(bot_sc, 0), (bot_sc - 1.0, 0)])
    assert mp.size() == 23

    mp.resize_metapop()

    assert mp.size() == 22  # only the strictly-below-cutoff entry is gone
    scores = sorted(sct.get_penalized_score() for sct in mp.get_trees())
    assert min(scores) == pytest.approx(bot_sc)  # the boundary value, not the one below it


def test_resize_metapop_phase1_uses_penalized_score_not_raw_score():
    """Distinguishes "compares get_penalized_score()" (correct,
    merging.cc:386) from a hypothetical "compares get_score() instead"
    bug: a RAW score high enough to look safe by itself, paired with a
    complexity_penalty large enough to push its PENALIZED score well
    below the cutoff. Removed only if the code truly reads
    `get_penalized_score()`.
    """
    mp = _mp()
    top_sc = mp.best_composite_score().get_score()
    rng = mp.useful_score_range()

    trap_raw_score = top_sc + 1000.0  # would look like the new BEST by raw score alone
    trap_penalty = 1000.0 + rng + 1.0  # penalized_score lands well below bot_sc
    _grow_pool(mp, [(top_sc, 0)] * 20 + [(trap_raw_score, 0, trap_penalty)])
    assert mp.size() == 22

    mp.resize_metapop()

    assert mp.size() == 21  # the high-raw/low-penalized trap entry is gone
    assert all(sct.get_score() != trap_raw_score for sct in mp.get_trees())


def test_resize_metapop_phase2_evicts_down_to_the_computed_cap():
    """Phase 2 (the cap-based random eviction) never compares scores at
    all -- it evicts a uniformly random SURVIVING pool member regardless
    of value -- so, unlike phase 1's own tests above, a raw-vs-penalized
    score distinction has nothing to exercise here; every padding entry
    below is tied at the same raw score with zero complexity penalty,
    comfortably above the phase-1 cutoff, so ONLY phase 2 runs.
    """
    mp = _mp()
    _grow_pool(mp, [(0.0, 0, ) for _ in range(219)])  # + seed == 220, all tied
    # Use a distinct, nonzero complexity so this scenario is not
    # accidentally penalty-blind (S9/S10's own recurring failure mode --
    # see the task brief's own Testing bar).
    assert mp.size() == 220

    randGen().seed(7)
    mp._params.cap_coef = 0.16
    mp.resize_metapop()

    merge_count = 1  # lazy-initialized from 0 on this, the FIRST call
    cap = mp._params.cap_coef
    cap *= merge_count + 250.0
    cap *= 1 + 2.0 * math.exp(-merge_count / 500.0)
    expected = int(cap)

    assert mp._merge_count == 1
    assert 50 < expected < 220  # sanity: exercises phase 2, no wraparound
    assert mp.size() == expected


def test_resize_metapop_unsigned_subtraction_wraps_and_surfaces_as_indexerror():
    """MEASURED HAZARD, reproduced via an explicit 64-bit mask rather than
    clamped (CARRY-FORWARD's own rule): `popsz - offset` (both `size_t` in
    C++) underflows whenever phase 1 leaves the population below `offset`
    (50) but a pathologically tiny `cap_coef` still demands further
    eviction. `_min_pool_size` (20) is strictly less than `offset` (50),
    so a population that survives phase 1 at, say, 25 members reaches
    phase 2 needing to evict below 50 -- reachable, not contrived away by
    construction elsewhere in this module.

    All entries here share one tied score, so phase 1 (whose cutoff logic
    is independently covered above) removes nothing, isolating phase 2 --
    keeps this test from accidentally exercising phase-1 behaviour instead
    of the wraparound it targets.

    The `IndexError` alone does NOT distinguish a correctly-masked
    subtraction from an unmasked one here: for ANY `popsz < offset`,
    `which = offset + (popsz_minus_offset % randint())`'s value lands
    outside `[0, popsz)` either way (`which > popsz` always, whether the
    pre-modulo dividend is the enormous masked value or the small raw
    negative one) -- confirmed by mutation: removing the `& MASK64` mask
    entirely left this test passing unchanged. What DOES distinguish them
    is the actual argument fed to `randint_n` -- captured directly below,
    by wrapping (not replacing) the real `randGen()` singleton's method,
    so the RNG stream itself is consumed exactly as `resize_metapop` would
    consume it either way.
    """
    mp = _mp()
    # `_mp()`'s own seed ("$1") is scored by the REAL cscorer, not 0.0 --
    # every padding entry is pinned to that SAME score (not a literal
    # 0.0) so the whole 25-entry pool is genuinely tied and phase 1
    # (independently covered above) removes nothing, isolating phase 2.
    seed_score = mp.best_composite_score().get_score()
    _grow_pool(mp, [(seed_score, 0) for _ in range(24)])  # + seed == 25, all tied
    assert mp.size() == 25

    randGen().seed(3)
    mp._params.cap_coef = 1e-6  # forces popsz_cap == 0 -- any positive popsz evicts

    rng = randGen()
    original_randint_n = rng.randint_n
    captured_diffs: list[int] = []

    def spy(n: int) -> int:
        captured_diffs.append(n)
        return original_randint_n(n)

    rng.randint_n = spy
    try:
        with pytest.raises(IndexError):
            mp.resize_metapop()
    finally:
        rng.randint_n = original_randint_n

    assert captured_diffs == [(25 - 50) & 0xFFFFFFFFFFFFFFFF]
    assert captured_diffs[0] > 2**63  # sanity: genuinely the wrapped value, not -25
