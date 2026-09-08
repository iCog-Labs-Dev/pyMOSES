"""Unit + oracle tests for pymetta.moses.metapopulation.metapopulation.

`metapopulationUTest.cxxtest` (the S10 slice's registered cxxtest oracle) is
built and passes (2 tests) but exercises only `remove_dominated`/
`get_nondominated_disjoint` -- domination.cc code outside every file this
whole slice touches (see metapopulation.py's own module docstring and
CARRY-FORWARD.md's oracle map). It covers none of THIS task's scope
(`init`, `best_candidates`, `best_tree`, `best_composite_score`,
`get_trees`, `size`, `empty`, `clear`, `update_best_candidates`, `rescore`,
`diversity_enabled`).

`tools/cpp_probe/metapop_core_probe.cc` fills that gap: a small, bounded
probe (build/run recipe in its own header comment) covering the
constructor (both overloads), `init`'s default-path scoring+insertion, and
`update_best_candidates`'s tie-break-by-complexity branch (public in the
header, called directly). Its measured output is reproduced as literal
`@pytest.mark.oracle` expected values below rather than a checked-in
fixture+capture-script pair -- the probe's own scope (4 trees, 3
scenarios) is small enough that the extra pipeline would be pure overhead
(see `hcextra_max_count_probe.cc` for the same precedent elsewhere in this
port: a probe written and run once for a specific, narrow question,
without a fixture file of its own).

`rescore()` and `init()` are both PRIVATE in classic (friended only to
`::metapopulationUTest`), so neither can be probed directly from outside
the class; they are exercised here with direct Python unit tests instead,
built on `sum_bscore`/`set_score`/`get_cscore`/`get_bscore` already being
oracle-verified elsewhere (test_behave_cscore.py, test_scoring_base.py).

Every `composite_score`-copying call site this module's own docstring
identifies (`best_composite_score`, the constructor's `_best_cscore` seed,
`update_best_candidates`, `_copy_sct`) gets a direct aliasing-proof test
below: mutate the object on one side, assert the other side is unaffected.
"""

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.moses.metapopulation.metapop_params import metapop_parameters
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.types import (
    composite_score,
    scored_combo_tree,
    scored_combo_tree_set,
    worst_composite_score,
)
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.scoring_base import bscore_base

# --------------------------------------------------------------------------
# Shared fixtures: the same 4 trees / scores tools/cpp_probe/metapop_core_
# probe.cc uses, built the same way (str_to_combo_tree, matching printed
# form exactly -- verified directly against the probe's own `show()` output
# when this test file was written: "$1 ", "and($1 $2) ", "and($1 $2 $3) ",
# "or($1 $2) " modulo the trailing space `ostream_combo_tree` appends).
# --------------------------------------------------------------------------

T1 = str_to_combo_tree("$1")  # argument leaf -- tree_complexity 1
T2 = str_to_combo_tree("and($1 $2)")  # tree_complexity 2
T3 = str_to_combo_tree("and($1 $2 $3)")  # tree_complexity 3
T4 = str_to_combo_tree("or($1 $2)")  # tree_complexity 2


class _ProbeBscore(bscore_base):
    """Mirrors tools/cpp_probe/metapop_core_probe.cc's own `probe_bscore`:
    a fixed score per tree, looked up by printed form, with `get_complexity`
    left at bscore_base's own default (`tree_complexity`) so complexity
    values are measured (against the C++ probe), not asserted.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__()
        self._scores = scores

    def __call__(self, tr):
        return [self._scores.get(combo_tree_to_str(tr).strip(), 0.0)]

    def best_possible_bscore(self):
        return [0.0]


def _cscorer() -> behave_cscore:
    scores = {
        combo_tree_to_str(T1).strip(): 2.0,
        combo_tree_to_str(T2).strip(): 5.0,
        combo_tree_to_str(T3).strip(): 5.0,
        combo_tree_to_str(T4).strip(): 1.0,
    }
    return behave_cscore(_ProbeBscore(scores))


# Sanity check on the fixture itself: tree_complexity must reproduce the
# probe's own measured values, or every oracle assertion below would be
# comparing against the wrong baseline without ever noticing.
def test_fixture_complexities_match_the_cpp_probe():
    from pymetta.moses.moses.complexity import tree_complexity

    assert tree_complexity(T1) == 1
    assert tree_complexity(T2) == 2
    assert tree_complexity(T3) == 3
    assert tree_complexity(T4) == 2


# ============================================================ constructor


@pytest.mark.oracle
def test_single_tree_constructor_matches_cpp_probe():
    """tools/cpp_probe/metapop_core_probe.cc's "single_ctor" scenario:
    size=1, best_tree="$1", best_composite_score score=2/complexity=1/
    penalized_score=2.
    """
    mp = metapopulation(T1, _cscorer())
    assert mp.size() == 1
    assert combo_tree_to_str(mp.best_tree()).strip() == "$1"
    csc = mp.best_composite_score()
    assert csc.get_score() == 2.0
    assert csc.get_complexity() == 1
    assert csc.get_complexity_penalty() == 0.0
    assert csc.get_uniformity_penalty() == 0.0
    assert csc.get_penalized_score() == 2.0


@pytest.mark.oracle
def test_multi_tree_constructor_matches_cpp_probe():
    """tools/cpp_probe/metapop_core_probe.cc's "multi_ctor" scenario. T2
    and T3 tie for the best raw score (5.0); T3 is strictly more complex,
    so the tie-break must keep only T2 in best_candidates/best_composite_score,
    and _scored_trees must sort T2 before T3 (sct_score_greater's own
    complexity tie-break, ascending: lower complexity sorts first at a
    tied score) before the strictly-lower-scoring T1 and T4.
    """
    mp = metapopulation([T1, T2, T3, T4], _cscorer())
    assert mp.size() == 4

    trees_in_order = [
        (combo_tree_to_str(sct.get_tree()).strip(), sct.get_score(), sct.get_complexity())
        for sct in mp.get_trees()
    ]
    assert trees_in_order == [
        ("and($1 $2)", 5.0, 2),
        ("and($1 $2 $3)", 5.0, 3),
        ("$1", 2.0, 1),
        ("or($1 $2)", 1.0, 2),
    ]

    csc = mp.best_composite_score()
    assert csc.get_score() == 5.0
    assert csc.get_complexity() == 2

    assert combo_tree_to_str(mp.best_tree()).strip() == "and($1 $2)"

    best = list(mp.best_candidates())
    assert len(best) == 1
    assert combo_tree_to_str(best[0].get_tree()).strip() == "and($1 $2)"
    assert best[0].get_complexity() == 2


def test_single_combo_tree_argument_is_not_confused_with_a_list():
    """The two C++ constructor overloads are merged on `isinstance(exemplars,
    list)` -- a bare `combo_tree` (util/tree.py's `tree` class) must never
    be mistaken for the vector overload.
    """
    mp = metapopulation(T1, _cscorer())
    assert mp.size() == 1


# ============================================================ pool accessors


def test_empty_and_clear():
    mp = metapopulation(T1, _cscorer())
    assert not mp.empty()
    mp.clear()
    assert mp.empty()
    assert mp.size() == 0
    assert list(mp.get_trees()) == []


def test_get_trees_returns_the_live_pool_object():
    mp = metapopulation(T1, _cscorer())
    assert mp.get_trees() is mp._scored_trees


# ============================================================ update_best_candidates


@pytest.mark.oracle
def test_update_best_candidates_direct_matches_cpp_probe():
    """tools/cpp_probe/metapop_core_probe.cc's "update_best_candidates_direct"
    scenario: start from a metapopulation over the single low-scoring T4
    (score 1.0), then call update_best_candidates (public in the header)
    directly with a hand-built set containing a same-score, lower- and
    higher-complexity pair. Final: best score=5.0, complexity=2, and
    best_candidates holds exactly one tree.
    """
    mp = metapopulation(T4, _cscorer())

    more = scored_combo_tree_set()
    more.insert(scored_combo_tree(T2, composite_score(5.0, 2)))
    more.insert(scored_combo_tree(T3, composite_score(5.0, 3)))
    mp.update_best_candidates(more)

    csc = mp.best_composite_score()
    assert csc.get_score() == 5.0
    assert csc.get_complexity() == 2
    assert len(list(mp.best_candidates())) == 1


def test_update_best_candidates_ties_at_same_complexity_are_all_kept():
    """Two candidates tied on BOTH score and complexity: the outer gate
    (`cpx <= best_cpx`) admits the second one without the inner gate
    (`cpx < best_cpx`) ever clearing best_candidates -- so both survive.
    """
    mp = metapopulation(T4, _cscorer())

    more = scored_combo_tree_set()
    more.insert(scored_combo_tree(T2, composite_score(5.0, 2)))
    more.insert(scored_combo_tree(T3, composite_score(5.0, 2)))
    mp.update_best_candidates(more)

    assert len(list(mp.best_candidates())) == 2


def test_update_best_candidates_is_order_independent():
    """Traced by hand in metapopulation.py's own docstring: this is a
    streaming running-minimum-by-complexity algorithm, so feeding the same
    two same-score/different-complexity candidates in either order must
    converge to the identical final (best_cscore, best_candidates).
    Ordering is controlled directly here via two SEPARATE calls (each a
    one-element set) rather than relying on scored_combo_tree_set's own
    unordered iteration.
    """
    a = scored_combo_tree(T2, composite_score(5.0, 2))
    b = scored_combo_tree(T3, composite_score(5.0, 3))

    mp_ab = metapopulation(T4, _cscorer())
    s1 = scored_combo_tree_set()
    s1.insert(a)
    mp_ab.update_best_candidates(s1)
    s2 = scored_combo_tree_set()
    s2.insert(b)
    mp_ab.update_best_candidates(s2)

    mp_ba = metapopulation(T4, _cscorer())
    s3 = scored_combo_tree_set()
    s3.insert(b)
    mp_ba.update_best_candidates(s3)
    s4 = scored_combo_tree_set()
    s4.insert(a)
    mp_ba.update_best_candidates(s4)

    for mp in (mp_ab, mp_ba):
        csc = mp.best_composite_score()
        assert csc.get_score() == 5.0
        assert csc.get_complexity() == 2
        best = list(mp.best_candidates())
        assert len(best) == 1
        assert best[0].get_complexity() == 2


def test_update_best_candidates_on_empty_candidates_is_a_no_op():
    mp = metapopulation(T1, _cscorer())
    before = mp.best_composite_score()
    mp.update_best_candidates(scored_combo_tree_set())
    after = mp.best_composite_score()
    assert before.get_score() == after.get_score()
    assert before.get_complexity() == after.get_complexity()


# ============================================================ rescore


def test_rescore_updates_score_from_current_bscore():
    """rescore() (merging.cc:126) is private in C++ and cannot be probed
    directly; ported straight from its own source and exercised here
    against sum_bscore/set_score, both already oracle-verified elsewhere.
    """
    mp = metapopulation(T1, _cscorer())
    (sct,) = list(mp.get_trees())
    assert sct.get_score() == 2.0

    # Change the underlying bscore in place -- rescore() must recompute
    # the composite score's `score` field from THIS bscore, not the one
    # originally captured at init() time.
    sct.set_bscore([10.0, -1.0])
    mp.rescore()

    assert sct.get_score() == pytest.approx(9.0)  # sum_bscore([10.0, -1.0])
    # update_penalized_score() must have re-run too (set_score's own
    # contract) -- complexity_penalty/uniformity_penalty are both 0 here,
    # so penalized_score tracks score exactly.
    assert sct.get_penalized_score() == pytest.approx(9.0)


def test_rescore_does_not_touch_best_candidates_scores():
    """Proves _copy_sct's own purpose: `_scored_trees` and `_best_candidates`
    must hold INDEPENDENT composite_score objects, so rescore()'s in-place
    mutation of a _scored_trees entry must NOT be visible through the
    corresponding _best_candidates entry.
    """
    mp = metapopulation(T2, _cscorer())
    (pool_sct,) = list(mp.get_trees())
    (best_sct,) = list(mp.best_candidates())
    assert pool_sct.get_score() == 5.0
    assert best_sct.get_score() == 5.0

    pool_sct.set_bscore([100.0])
    mp.rescore()

    assert pool_sct.get_score() == pytest.approx(100.0)
    assert best_sct.get_score() == 5.0  # unaffected


# ============================================================ aliasing proofs
#
# Every composite_score this class stores or returns, checked for the same
# copy-vs-reference shape C++ has at that exact call site (this module's
# own docstring names each one).


def test_best_composite_score_returns_an_independent_copy():
    """best_composite_score() returns BY VALUE in C++ (metapopulation.h:141,
    no `&`) -- mutating the returned object must not corrupt the
    metapopulation's own internal `_best_cscore`.
    """
    mp = metapopulation(T1, _cscorer())
    returned = mp.best_composite_score()
    returned.set_score(999.0)

    again = mp.best_composite_score()
    assert again.get_score() == 2.0
    assert again.get_score() != 999.0


def test_constructor_does_not_alias_worst_composite_score_singleton():
    """`_best_cscore(worst_composite_score)` copy-constructs in C++
    (metapopulation.cc:40) -- mutating a freshly-constructed metapopulation's
    best_composite_score must never reach the shared module-level
    `worst_composite_score` singleton (types.py), the exact aliasing shape
    CARRY-FORWARD's S9 fix round closed for `complexity_based_scorer`.
    """
    original_score = worst_composite_score.get_score()

    mp = metapopulation(T1, _cscorer())
    # T1 (score 2.0) beats worst_composite_score during init(), so
    # _best_cscore has already been replaced by the time we get here --
    # exercise the aliasing risk at its actual origin point instead: the
    # SEED value itself, via a metapopulation whose single exemplar can
    # never beat very_worst_score.
    from pymetta.moses.moses.types import very_worst_score

    losing_scorer = behave_cscore(_ProbeBscore({combo_tree_to_str(T1).strip(): very_worst_score}))
    mp_never_beats = metapopulation(T1, losing_scorer)
    assert mp_never_beats._best_cscore is not worst_composite_score
    seeded = mp_never_beats.best_composite_score()
    seeded.set_score(12345.0)

    assert worst_composite_score.get_score() == original_score
    assert worst_composite_score.get_score() != 12345.0
    del mp  # silence "unused" -- constructed only to prove it doesn't crash


def test_update_best_candidates_copies_the_composite_score_not_the_caller_object():
    """`_best_cscore = csc;` (merging.cc:638) copies -- mutating the
    ORIGINAL composite_score object passed into update_best_candidates
    after the call must not affect the metapopulation's own state.
    """
    mp = metapopulation(T4, _cscorer())
    original_csc = composite_score(5.0, 2)
    cand = scored_combo_tree_set()
    cand.insert(scored_combo_tree(T2, original_csc))
    mp.update_best_candidates(cand)

    original_csc.set_score(-1.0)

    assert mp.best_composite_score().get_score() == 5.0


def test_update_best_candidates_copies_the_candidate_not_the_caller_object():
    """`_best_candidates.insert(cnd)` copies the whole scored_combo_tree
    (_copy_sct) -- mutating the caller's own scored_combo_tree object
    after insertion must not affect what's stored in best_candidates.
    """
    mp = metapopulation(T4, _cscorer())
    cand_tree = scored_combo_tree(T2, composite_score(5.0, 2))
    cand = scored_combo_tree_set()
    cand.insert(cand_tree)
    mp.update_best_candidates(cand)

    cand_tree.get_composite_score().set_score(-1.0)

    (stored,) = list(mp.best_candidates())
    assert stored.get_score() == 5.0


# ============================================================ gates


def test_diversity_enabled_default_is_false():
    mp = metapopulation(T1, _cscorer())
    assert mp.diversity_enabled() is False


def test_diversity_enabled_mutation_flips_true():
    """Pins the diversity_enabled() gate: flipping `pressure` above 0.0
    must flip the accessor's own return value."""
    pa = metapop_parameters()
    mp = metapopulation(T1, _cscorer(), pa)
    assert mp.diversity_enabled() is False

    pa.diversity.pressure = 0.5
    assert mp.diversity_enabled() is True


def test_do_boosting_default_false_returns_best_cscore_without_raising():
    mp = metapopulation(T1, _cscorer())
    csc = mp.best_composite_score()
    assert csc.get_score() == 2.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "$1"


def test_do_boosting_true_makes_best_composite_score_raise():
    """Pins the do_boosting gate on best_composite_score(): flipping it
    must reach the unported ensemble/get_cscore(scored_combo_tree_set)
    path and fail there, rather than silently still returning
    _best_cscore.
    """
    pa = metapop_parameters()
    mp = metapopulation(T1, _cscorer(), pa)
    assert mp.best_composite_score().get_score() == 2.0  # gate off: fine

    pa.do_boosting = True
    with pytest.raises(AttributeError):
        mp.best_composite_score()


def test_do_boosting_true_makes_best_tree_raise():
    """Pins the do_boosting gate on best_tree(): it must route into
    ensemble.get_weighted_tree(), which always raises NotImplementedError.
    """
    pa = metapop_parameters()
    mp = metapopulation(T1, _cscorer(), pa)
    assert combo_tree_to_str(mp.best_tree()).strip() == "$1"  # gate off: fine

    pa.do_boosting = True
    with pytest.raises(NotImplementedError):
        mp.best_tree()


def test_discard_dominated_default_false_constructs_and_populates_pool():
    pa = metapop_parameters()
    assert pa.discard_dominated is False
    mp = metapopulation([T1, T2], _cscorer(), pa)
    assert mp.size() == 2


def test_discard_dominated_true_makes_init_raise():
    """Pins the discard_dominated gate: flipping it must make init()'s
    (and therefore the constructor's) merge step raise, instead of
    silently taking the insert-all default path.
    """
    pa = metapop_parameters()
    pa.discard_dominated = True
    with pytest.raises(NotImplementedError):
        metapopulation(T1, _cscorer(), pa)


# ============================================================ trivial accessors


def test_get_ensemble_and_get_cscorer_and_best_score():
    cscorer = _cscorer()
    mp = metapopulation(T1, cscorer)
    assert mp.get_cscorer() is cscorer
    assert mp.get_ensemble() is mp._ensemble
    assert mp.best_score() == mp.best_composite_score().get_score()
