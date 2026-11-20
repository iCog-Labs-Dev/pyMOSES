"""The IT linkage study (BACKLOG 13.9): the opt-in read-only access in existing code. The
``on_trial`` observer of ``gom`` and ``GomeaEngine`` changes no result; ``LinkageModel``'s
``last_fos`` is the FOS that ``learn`` returns and ``last_similarity`` the matrix that the
linkage tree used (``lt-mi`` and ``lt-b`` only); neither draws from the RNG."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.gom import Individual
from pymoses.gomea.stats import biased_similarity, entropy_matrix, mi_similarity
from pymoses.util.rng import MT19937RandGen


class _Trap:
    """Concatenated 4-bit deceptive traps over 0/1 codes (higher is better)."""

    def __init__(self, budget=10**9):
        self.n = 0
        self.budget = budget

    def check(self):
        return "global_budget" if self.n >= self.budget else None

    def evaluate(self, g):
        from pymoses.gomea.oracle import BudgetExhausted

        if self.n >= self.budget:
            raise BudgetExhausted("global_budget")
        self.n += 1
        s = 0
        for i in range(0, len(g), 4):
            u = sum(g[i:i + 4])
            s += 4 if u == 4 else 3 - u
        return float(s)

    def phenotype_key(self, g):
        return g

    def worse(self, a, b):
        return a < b

    def record_noop_skip(self):
        pass

    def record_phenotype_skip(self):
        pass


def _run(mode, on_trial=None, seed=4, n=40, length=16):
    r = random.Random(seed)
    ev = _Trap(budget=4000)
    pop = []
    for _ in range(n):
        g = tuple(r.randrange(2) for _ in range(length))
        pop.append(Individual(g, ev.evaluate(g), g))
    kw = {} if on_trial is None else {"on_trial": on_trial}
    dist = None
    if mode == "np":
        dist = lambda ps: [[abs(a - b) for b in ps] for a in ps]  # noqa: E731
    eng = GomeaEngine([2] * length, EngineParams(fos_mode=mode, max_generations=6,
                                                 stall_generations=100), ev,
                      MT19937RandGen(seed + 1), MT19937RandGen(seed + 2), distance=dist, **kw)
    res = eng.run(pop)
    return eng, res, ev


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_the_on_trial_observer_changes_nothing(mode):
    seen = Counter()
    _, a, ev_a = _run(mode)
    _, b, ev_b = _run(mode, on_trial=lambda size, outcome: seen.update([outcome]))

    def strip(h):
        return [{k: v for k, v in r.items() if not k.endswith("_s")} for r in h]

    assert [i.genotype for i in a.population] == [i.genotype for i in b.population]
    assert strip(a.history) == strip(b.history) and ev_a.n == ev_b.n
    assert a.termination_reason == b.termination_reason
    stats = Counter()
    for r in b.history:
        stats.update(r["stats"])
    assert seen["improving"] == stats["accepted_improving"]
    assert seen["rejected"] == stats["rejected"]
    assert seen["neutral"] + seen["skip"] == stats["accepted_neutral"]
    assert seen["skip"] == stats["phenotype_skips"]


def _pop(seed=3, n=30, length=10):
    r = random.Random(seed)
    return [tuple(r.randrange(3) for _ in range(length)) for _ in range(n)]


@pytest.mark.parametrize("mode", LinkageModel.MODES)
def test_last_fos_and_last_similarity(mode):
    pop = _pop()
    dist = lambda ps: [[abs(a - b) for b in ps] for a in ps]  # noqa: E731
    model = LinkageModel(mode, MT19937RandGen(5), distance=dist)
    twin = LinkageModel(mode, MT19937RandGen(5), distance=dist)
    assert model.last_fos is None and model.last_similarity is None
    for m in (model, twin):
        m.calibrate(pop, [3] * 10)
    pop2 = _pop(seed=9)
    for p in (pop, pop2):
        fos = model.learn(p)
        assert model.last_fos == fos
        assert twin.learn(p) == fos  # the same draws as a second model
        if mode in ("lt-mi", "lt-b"):
            positions, S = model.last_similarity
            assert positions == model.search_positions
            H = entropy_matrix(p, positions)
            want = mi_similarity(H) if mode == "lt-mi" else biased_similarity(H, model._H0)
            assert S == want
        else:
            assert model.last_similarity is None
    assert model.rng.randint() == twin.rng.randint()
