"""Unit tests for pymoses.gomea.stats and pymoses.gomea.fos."""

import math
import random

import pytest

from pymoses.gomea.fos import (
    LinkageModel,
    linkage_tree_fos,
    random_tree_fos,
    univariate_fos,
)
from pymoses.gomea.stats import (
    biased_similarity,
    entropy_matrix,
    marginal_entropies,
    mi_similarity,
)
from pymoses.util.rng import MT19937RandGen


def _pop(seed, n, alphabets):
    r = random.Random(seed)
    return [tuple(r.randrange(k) for k in alphabets) for _ in range(n)]


def _dependent_pop(seed, n=80):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        x, y = r.randrange(3), r.randrange(2)
        rows.append((x, (x + 1) % 3, 0, y, 1 - y, r.randrange(4), 2))
    return rows


# ------------------------------------------------------------------ stats


def test_entropy_known_values():
    pop = [(0, 0), (1, 1), (0, 1), (1, 0)]
    H = entropy_matrix(pop, [0, 1])
    assert H[0][0] == pytest.approx(math.log(2), abs=1e-15)
    assert H[1][1] == pytest.approx(math.log(2), abs=1e-15)
    assert H[0][1] == pytest.approx(math.log(4), abs=1e-15)
    assert H[1][0] == H[0][1]


def test_entropy_uses_positions_subset():
    pop = _pop(1, 30, [2, 3, 4, 5])
    full = entropy_matrix(pop, range(4))
    sub = entropy_matrix(pop, [3, 1])
    assert sub[0][0] == full[3][3]
    assert sub[1][1] == full[1][1]
    assert sub[0][1] == pytest.approx(full[1][3], abs=1e-15)


def test_label_permutation_within_field_leaves_entropy_unchanged():
    pop = _pop(2, 50, [3, 4, 2])
    relabel = {0: 2, 1: 0, 2: 3, 3: 1}  # a bijection on field 1's codes
    permuted = [(a, relabel[b], c) for a, b, c in pop]
    H = entropy_matrix(pop, range(3))
    Hp = entropy_matrix(permuted, range(3))
    for i in range(3):
        for j in range(3):
            assert Hp[i][j] == pytest.approx(H[i][j], abs=1e-12)


def test_mi_similarity_definition():
    H = [[1.0, 1.5, 2.0], [1.5, 0.8, 1.2], [2.0, 1.2, 1.1]]
    S = mi_similarity(H)
    assert S[0][0] == 1.0 and S[1][1] == 0.8 and S[2][2] == 1.1
    assert S[0][1] == S[1][0] == pytest.approx(1.0 + 0.8 - 1.5)
    assert S[1][2] == S[2][1] == pytest.approx(0.8 + 1.1 - 1.2)


def test_biased_similarity_identity_is_zero_off_diagonal():
    pop = _pop(3, 60, [2, 3, 5, 2])
    H0 = entropy_matrix(pop, range(4))
    S = biased_similarity(H0, H0)
    for i in range(4):
        assert S[i][i] == pytest.approx(1.0, abs=1e-15)
        for j in range(4):
            if i != j:
                assert S[i][j] == pytest.approx(0.0, abs=1e-12)


def test_biased_similarity_negative_values_are_not_clamped():
    # identical columns at H0, independent now: 1 + 1 - 2 * 2 = -2
    h = math.log(2)
    H0 = [[h, h], [h, h]]
    H = [[h, 2 * h], [2 * h, h]]
    S = biased_similarity(H, H0)
    assert S[0][1] == pytest.approx(-2.0)
    assert S[1][0] == S[0][1]


def test_biased_similarity_rejects_zero_h0():
    H = [[0.5, 0.7], [0.7, 0.4]]
    with pytest.raises(ValueError):
        biased_similarity(H, [[0.0, 0.7], [0.7, 0.4]])
    with pytest.raises(ValueError):
        biased_similarity(H, [[0.5, 0.0], [0.0, 0.4]])


# ------------------------------------------------------------------ fos


def test_linkage_tree_negative_similarity_still_merges():
    S = [[0.0, -1.0, -5.0], [-1.0, 0.0, -3.0], [-5.0, -3.0, 0.0]]
    fos = linkage_tree_fos(S, [0, 1, 2])
    assert fos[:3] == [[0], [1], [2]]
    assert fos[3] == [0, 1]
    assert sorted(fos[4]) == [0, 1, 2]
    assert len(fos) == 5


def test_linkage_tree_returns_all_sets_including_root():
    S = [[0.0, 0.9, 0.1, 0.2], [0.9, 0.0, 0.3, 0.1], [0.1, 0.3, 0.0, 0.8], [0.2, 0.1, 0.8, 0.0]]
    fos = linkage_tree_fos(S, [2, 0, 3, 1])
    assert fos[:4] == [[2], [0], [3], [1]]
    assert len(fos) == 7
    assert sorted(fos[-1]) == [0, 1, 2, 3]
    assert {tuple(sorted(s)) for s in fos[4:6]} == {(0, 1), (2, 3)}


def test_linkage_tree_degenerate_sizes():
    assert linkage_tree_fos([], []) == []
    assert linkage_tree_fos([[0.3]], [0]) == [[0]]


def test_univariate_fos():
    assert univariate_fos(0) == []
    assert univariate_fos(3) == [[0], [1], [2]]


def test_random_tree_fos_structure_and_determinism():
    a = random_tree_fos(7, MT19937RandGen(5))
    b = random_tree_fos(7, MT19937RandGen(5))
    assert a == b
    assert a[:7] == [[i] for i in range(7)]
    # merges while more than 2 clusters: 7 - 2 merges, no root
    assert len(a) == 7 + 5
    assert all(len(s) < 7 for s in a)
    assert random_tree_fos(2, MT19937RandGen(1)) == [[0], [1]]


# ------------------------------------------------------------------ LinkageModel


def _covered(fos):
    return sorted({p for s in fos for p in s})


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_learn_covers_search_positions_and_excludes_root(mode):
    pop = _dependent_pop(7)
    domains = [3, 3, 1, 2, 2, 4, 3]
    model = LinkageModel(mode, MT19937RandGen(11))
    model.calibrate(pop, domains)
    fos = model.learn(pop)
    search = list(model.search_positions)
    assert _covered(fos) == search
    assert all(sorted(s) != search for s in fos)  # no root


def test_constant_fields_excluded_from_search_positions():
    pop = _dependent_pop(8)
    model = LinkageModel("lt-b", MT19937RandGen(1))
    model.calibrate(pop, [3, 3, 1, 2, 2, 4, 3])
    assert model.search_positions == (0, 1, 3, 4, 5)
    assert model.constant_positions == (2, 6)
    for s in model.learn(pop):
        assert 2 not in s and 6 not in s


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_single_search_position(mode):
    pop = [(0, 1, 2), (0, 0, 2), (0, 1, 2)]
    model = LinkageModel(mode, MT19937RandGen(3))
    model.calibrate(pop, [1, 2, 3])
    assert model.search_positions == (1,)
    assert model.learn(pop) == [[1]]


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_no_search_positions(mode):
    pop = [(0, 1), (0, 1)]
    model = LinkageModel(mode, MT19937RandGen(3))
    model.calibrate(pop, [1, 2])
    assert model.search_positions == ()
    assert model.learn(pop) == []


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_seeded_learning_is_deterministic(mode):
    # exact ties: duplicated full-factorial columns
    rows = [((m >> 0) & 1, (m >> 1) & 1, (m >> 2) & 1) for m in range(8)]
    pop = [(a, a, b, b, c, c) for a, b, c in rows] * 2
    runs = []
    for _ in range(2):
        model = LinkageModel(mode, MT19937RandGen(42))
        model.calibrate(pop, [2] * 6)
        runs.append([model.learn(pop) for _ in range(3)])
    assert runs[0] == runs[1]


def test_lt_modes_use_rng_order():
    pop = _pop(9, 40, [2, 3, 2, 4, 3])
    fos_by_seed = set()
    for seed in range(6):
        model = LinkageModel("lt-mi", MT19937RandGen(seed))
        model.calibrate(pop, [2, 3, 2, 4, 3])
        fos = model.learn(pop)
        fos_by_seed.add(tuple(tuple(s) for s in fos[:5]))
    assert len(fos_by_seed) > 1  # the singleton order follows the rng permutation


def test_learn_before_calibrate_raises():
    model = LinkageModel("lt-b", MT19937RandGen(0))
    with pytest.raises(RuntimeError):
        model.learn([(0, 1), (1, 0)])


def test_calibrate_twice_raises():
    model = LinkageModel("uni", MT19937RandGen(0))
    model.calibrate([(0, 1), (1, 0)], [2, 2])
    with pytest.raises(RuntimeError):
        model.calibrate([(0, 1), (1, 0)], [2, 2])


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        LinkageModel("lt", MT19937RandGen(0))


def test_calibrate_validates_domains():
    model = LinkageModel("uni", MT19937RandGen(0))
    with pytest.raises(ValueError):
        model.calibrate([(0, 2), (1, 0)], [2, 2])
    with pytest.raises(ValueError):
        LinkageModel("uni", MT19937RandGen(0)).calibrate([(0, 1)], [2])


def _lt_b_case():
    """A deterministic d>=3 case where LT-B learned on P1 with H0 from P0
    differs from LT-B calibrated on P1 itself (searched over small seeds)."""
    domains = [2, 3, 2, 3]
    for seed in range(200):
        r = random.Random(seed)
        pop0 = [tuple(r.randrange(k) for k in domains) for _ in range(12)]
        pop1 = [tuple(r.randrange(k) for k in domains) for _ in range(12)]
        if any(len({row[p] for row in pop}) < 2 for pop in (pop0, pop1) for p in range(4)):
            continue  # keep every position a search position under both
        cross = LinkageModel("lt-b", MT19937RandGen(0))
        cross.calibrate(pop0, domains)
        own = LinkageModel("lt-b", MT19937RandGen(0))
        own.calibrate(pop1, domains)
        if cross.learn(pop1) != own.learn(pop1):
            return seed, domains, pop0, pop1
    raise AssertionError("no distinguishing LT-B case found")


def test_lt_b_uses_calibration_h0():
    """F4(b): ``LinkageModel('lt-b')`` scores similarity against the H0 of
    the CALIBRATION population, not of the population it learns from."""
    _seed, domains, pop0, pop1 = _lt_b_case()
    positions = range(len(domains))

    model = LinkageModel("lt-b", MT19937RandGen(0))
    model.calibrate(pop0, domains)
    fos = model.learn(pop1)

    own = LinkageModel("lt-b", MT19937RandGen(0))
    own.calibrate(pop1, domains)
    assert fos != own.learn(pop1)  # the chosen case really distinguishes the two

    # The expected FOS, built by hand from H(P1) and H0 = H(P0), with the
    # same seeded random order LinkageModel draws.
    order_rng = MT19937RandGen(0)
    order = list(range(len(domains)))
    for i in range(len(order) - 1, 0, -1):
        j = order_rng.randint_n(i + 1)
        order[i], order[j] = order[j], order[i]
    S = biased_similarity(entropy_matrix(pop1, positions), entropy_matrix(pop0, positions))
    assert fos == linkage_tree_fos(S, order)[:-1]


# ------------------------------------------------- port-only optimisations


def test_marginal_entropies_equal_the_entropy_matrix_diagonal_bit_for_bit():
    pop = _pop(21, 37, [2, 3, 1, 5, 4, 2])
    got = marginal_entropies(pop, range(6))
    assert got == [entropy_matrix(pop, [p])[0][0] for p in range(6)]
    assert marginal_entropies(pop, [4, 1]) == [got[4], got[1]]
    assert marginal_entropies(pop, []) == []


def _without_reuse(model):
    """Drop the calibration matrix kept for the first ``learn``, so the
    model recomputes it like before the optimisation."""
    model._first_learn = None
    return model


def _sym(pop):
    # a pure recoding of positions 0 and 1 (like ``contin_symbols``)
    return [(r[0] // 2, r[1] % 2, *r[2:]) for r in pop]


@pytest.mark.parametrize("mode", LinkageModel.MODES)
@pytest.mark.parametrize("symbolize", [None, _sym])
def test_first_learn_reuse_gives_the_identical_fos_and_rng_state(mode, symbolize):
    domains = [4, 3, 2, 3, 4, 2]
    pops = [_pop(s, 30, domains) for s in (1, 2, 3)]
    # exact ties too: duplicated full-factorial columns
    rows = [((m >> 0) & 1, (m >> 1) & 1, (m >> 2) & 1) for m in range(8)]
    tie_pop = [(a, a, b, b, c, c) for a, b, c in rows] * 2
    for pop0, later, doms in ((pops[0], pops[1:], domains), (tie_pop, [tie_pop], [2] * 6)):
        runs = []
        for reuse in (True, False):
            rng = MT19937RandGen(13)
            model = LinkageModel(mode, rng, symbolize)
            model.calibrate(pop0, doms)
            if not reuse:
                _without_reuse(model)
            fos = [model.learn(list(pop0))] + [model.learn(p) for p in later]
            runs.append((fos, rng.randint_n(1 << 30)))
        assert runs[0] == runs[1]


def test_first_learn_on_a_different_population_does_not_reuse_h0():
    domains = [2, 3, 2, 3]
    pop0 = _pop(4, 20, domains)
    pop1 = _pop(5, 20, domains)
    a = LinkageModel("lt-mi", MT19937RandGen(0))
    a.calibrate(pop0, domains)
    b = LinkageModel("lt-mi", MT19937RandGen(0))
    b.calibrate(pop0, domains)
    _without_reuse(b)
    assert a.learn(pop1) == b.learn(pop1)
    assert a._first_learn is None


def test_calibrate_reports_the_first_out_of_domain_value_as_before():
    model = LinkageModel("uni", MT19937RandGen(0))
    with pytest.raises(ValueError, match="value 3 at position 1 is outside domain 3"):
        model.calibrate([(0, 1), (1, 3), (2, 0)], [2, 3])
