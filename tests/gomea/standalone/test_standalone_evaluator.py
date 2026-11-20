"""The IT linkage study (BACKLOG 13.9, plan sections 5.5 and 8.1): ``RegressionEvaluator`` and
the scorers. The budget raises ``BudgetExhausted`` at exactly E; cache hits are not counted;
NaN maps to -inf; the IT fit (the drop of constant and aliased columns, the least-squares
identity) agrees with ``least_squares.fit_terms``; the cached and uncached IT scorers give
bit-identical scores."""

from __future__ import annotations

import math
import random

import pytest
from standalone_data import synthetic_rows

from pymoses.gomea.oracle import BudgetExhausted
from pymoses.gomea.standalone.evaluator import (
    ITScorer,
    RegressionEvaluator,
    TemplateScorer,
    TrainData,
    linear_scaling,
    nmse,
)
from pymoses.gomea.standalone.genome_it import ITGenome
from pymoses.gomea.standalone.genome_template import TemplateGenome
from pymoses.gomea.standalone.run import scaled_target
from pymoses.moses.scoring.least_squares import fit_terms


def _data(n=80, seed=2):
    rows = synthetic_rows(n, seed)
    cols = [[float(r[j]) for r in rows] for j in range(5)]
    y, _, _ = scaled_target([float(r[5]) for r in rows])
    return TrainData(cols, y)


def _it_genome(**kw):
    return ITGenome(5, zero_inputs=frozenset({2, 4}), zero_one_inputs=frozenset({4}), **kw)


def _random(genome, r):
    return tuple(r.randrange(k) for k in genome.domains)


# ---- the budget, the cache and NaN


class _ConstScorer:
    def __init__(self, value=-1.0):
        self.value = value
        self.calls = 0

    def score(self, model):
        self.calls += 1
        return self.value


def test_the_budget_raises_at_exactly_E_and_cache_hits_are_not_counted():
    genome = TemplateGenome(4, height=1)
    sc = _ConstScorer()
    ev = RegressionEvaluator(genome, sc, max_evals=5)
    distinct = [(f, a, b) for f in range(4) for a in range(4) for b in range(4)][:5]
    ev.evaluate(distinct[0])
    for _ in range(3):
        ev.evaluate(distinct[0])  # a repeated phenotype: a cache hit
    assert ev.n_evals == 1 and ev.cache_hits == 3 and sc.calls == 1
    for g in distinct[1:]:
        assert ev.check() is None
        ev.evaluate(g)
    assert ev.n_evals == 5 and sc.calls == 5
    assert ev.check() == "global_budget"
    with pytest.raises(BudgetExhausted) as exc:
        ev.evaluate(distinct[0])  # even a cache hit: the budget is checked first
    assert exc.value.reason == "global_budget"
    assert ev.n_evals == 5


def test_a_different_genotype_with_the_same_phenotype_is_a_cache_hit():
    genome = TemplateGenome(4, height=1)
    ev = RegressionEvaluator(genome, _ConstScorer(), max_evals=10)
    a = (4 + 1, 0, 0)  # the root holds x1: the leaves are introns
    b = (4 + 1, 3, 2)
    assert ev.phenotype_key(a) == ev.phenotype_key(b) == "x1"
    ev.evaluate(a)
    ev.evaluate(b)
    assert ev.n_evals == 1 and ev.cache_hits == 1


def test_nan_maps_to_minus_infinity():
    genome = TemplateGenome(4, height=1)
    ev = RegressionEvaluator(genome, _ConstScorer(math.nan), max_evals=10)
    assert ev.evaluate((0, 1, 2)) == -math.inf
    assert ev.worse(math.nan, -1e300) and not ev.worse(-1e300, math.nan)
    assert not ev.worse(math.nan, -math.inf) and not ev.worse(-math.inf, -math.inf)
    assert ev.worse(-math.inf, 0.0)


def test_checkpoints_and_trace():
    genome = TemplateGenome(4, height=1)

    class Seq:
        def __init__(self):
            self.v = iter([-5.0, -7.0, -3.0, -3.0, -1.0])

        def score(self, model):
            return next(self.v)

    ev = RegressionEvaluator(genome, Seq(), max_evals=5, checkpoints=[1, 2, 4, 5, 99])
    for g in [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0)]:
        ev.evaluate(g)
    assert ev.trace == [(1, -5.0), (3, -3.0), (5, -1.0)]
    assert ev.anytime == [(1, -5.0), (2, -5.0), (4, -3.0), (5, -1.0)]


def test_max_evals_must_be_positive():
    with pytest.raises(ValueError):
        RegressionEvaluator(TemplateGenome(2, height=1), _ConstScorer(), max_evals=0)


# ---- the template scorer


def test_the_template_score_is_minus_the_mse_of_the_scaled_program():
    data = _data()
    genome = TemplateGenome(5)
    sc = TemplateScorer(genome, data)
    r = random.Random(4)
    vy = sum((v - sum(data.y) / data.n) ** 2 for v in data.y) / data.n
    seen = 0
    for _ in range(200):
        g = _random(genome, r)
        _, post = genome.decode(g)
        s = sc.score(post)
        p = genome.predict(post, data.cols)
        if not all(math.isfinite(x) for x in p):
            assert s == -math.inf
            continue
        a, b = linear_scaling(data.y, p)
        assert -s == pytest.approx(nmse(data.y, p, a, b) * vy / 100.0, rel=1e-9, abs=1e-12)
        seen += 1
    assert seen > 150
    # a constant output: the mean model
    x0 = (4 + 0,) + (0,) * 30  # the root holds x0 ...
    same = TemplateScorer(genome, TrainData([[2.0] * data.n] * 5, data.y))
    assert same.score(genome.decode(x0)[1]) == -same.syy / data.n


def test_the_aq_column_equals_the_rule_of_each_row():
    from pymoses.gomea.standalone.genome_template import _aq, _aq_column

    r = random.Random(1)
    vals = [0.0, -0.0, 1.5, -2.0, 1e200, -1e160, 3e-200, math.inf, -math.inf, math.nan]
    for _ in range(300):
        n = 6
        pool = vals if r.random() < 0.5 else [r.uniform(-5, 5) for _ in range(4)] + [0.0]
        A = [r.choice(pool) for _ in range(n)]
        B = [r.choice(pool) for _ in range(n)]
        got = _aq_column(A, B)
        want = [_aq(a, b) for a, b in zip(A, B, strict=True)]
        assert all(x == y or (x != x and y != y) for x, y in zip(got, want, strict=True)), (A, B)


# ---- the IT scorer


def _exact_sse(scorer, kept):
    cols = [scorer.column(m) for m in kept]
    fit = fit_terms(scorer.y, cols)
    assert fit.weights is not None
    return math.fsum((t - f) ** 2 for t, f in zip(scorer.y, fit.fitted, strict=True))


def test_the_it_score_agrees_with_fit_terms():
    data = _data()
    genome = _it_genome()
    sc = ITScorer(genome, data)
    r = random.Random(5)
    modes = set()
    for _ in range(400):
        m = genome.decode(_random(genome, r)).monomials
        f = sc.fit(m)
        modes.add(f.mode)
        if f.mode in ("terms", "single"):
            assert f.sse == pytest.approx(_exact_sse(sc, f.kept), rel=1e-9, abs=1e-9)
            # the kept columns are a subset of the model's terms, in key order
            assert list(f.kept) == [t for t in m if t in f.kept]
    assert "terms" in modes
    for one in ((((0, 1),),), (((1, 2), (2, 1)),)):
        f = sc.fit(one)
        assert f.mode == "single" and f.kept == one
        assert f.sse == pytest.approx(_exact_sse(sc, f.kept), rel=1e-9)


def test_the_it_fit_drops_constant_and_aliased_columns():
    data = _data()
    genome = _it_genome()
    sc = ITScorer(genome, data)
    x4 = ((3, 1),)       # two values: aliased with x4^2 and 1/x4 (and the intercept)
    x4sq = ((3, 2),)
    x4inv = ((3, -1),)
    x1 = ((0, 1),)
    f = sc.fit((x1, x4, x4sq, x4inv))
    assert f.mode == "terms" and f.dropped_aliased == 2
    assert f.kept == (x1, x4)
    assert f.sse == pytest.approx(_exact_sse(sc, f.kept), rel=1e-9)
    # a constant column: the model of x3 / x3 does not exist; a column of a constant input
    const = ITScorer(genome, TrainData(data.cols[:4] + [[1.0] * data.n], data.y))
    f = const.fit((x1, ((4, 1),)))
    assert f.mode == "single" and f.kept == (x1,) and f.dropped_constant == 1
    f = const.fit((((4, 1),),))
    assert f.mode == "mean" and f.kept == () and f.sse == pytest.approx(sc.syy)


def test_the_empty_it_model_scores_as_the_mean():
    data = _data()
    sc = ITScorer(_it_genome(), data)
    f = sc.fit(())
    assert f.mode == "mean" and f.kept == ()
    assert sc.score(()) == -sc.syy / data.n


def test_a_non_finite_it_column_makes_the_model_invalid():
    data = _data()
    genome = ITGenome(5)  # no zero rule: 1/x3 has 1/0 on the rows where x3 = 0
    sc = ITScorer(genome, data)
    f = sc.fit((((2, -1),),))
    assert f.mode == "invalid" and sc.score((((2, -1),),)) == -math.inf


def test_cached_and_uncached_it_scorers_give_the_same_floats():
    data = _data(n=120, seed=6)
    genome = _it_genome()
    scorers = [ITScorer(genome, data, column_cache=False),
               ITScorer(genome, data, column_cache=True),
               ITScorer(genome, data, column_cache=True, max_columns=8, max_pairs=10)]
    r = random.Random(8)
    for _ in range(600):
        m = genome.decode(_random(genome, r)).monomials
        fits = [s.fit(m) for s in scorers]
        assert fits[0] == fits[1] == fits[2]
        scores = [s.score(m) for s in scorers]
        assert scores[0] == scores[1] == scores[2] or all(math.isnan(x) for x in scores)
    assert scorers[1].stats["column_hits"] > 0 and scorers[1].stats["pair_hits"] > 0
    assert scorers[2].stats["cache_clears"] > 0 and scorers[2].stats["pair_clears"] > 0
    assert scorers[0].stats["column_hits"] == 0
