"""End-to-end GOMEA engine behaviour on synthetic objectives: OneMax and a
planted concatenated deceptive trap."""

from __future__ import annotations

import pytest
from diagnostic_objectives import CountingEvaluator, onemax, random_population, trap

import pymoses.gomea.engine as engine_mod
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.util.rng import MT19937RandGen

TRAP_K = 4
TRAP_BLOCKS = 4
TRAP_L = TRAP_K * TRAP_BLOCKS
TRAP_N = 64
TRAP_BUDGET = 20000
TRAP_SEEDS = range(10)
TRUE_BLOCKS = [list(range(b * TRAP_K, (b + 1) * TRAP_K)) for b in range(TRAP_BLOCKS)]


@pytest.mark.parametrize("mode", ["uni", "rt", "lt-mi", "lt-b"])
def test_onemax_30_ternary_fields_reaches_optimum(mode):
    fn = onemax(2)
    ev = CountingEvaluator(fn)
    rng = MT19937RandGen(101)
    pop = [ev.individual(g) for g in random_population(rng, 32, [3] * 30)]
    res = GomeaEngine(
        [3] * 30,
        EngineParams(fos_mode=mode, max_generations=100),
        ev,
        MT19937RandGen(102),
        MT19937RandGen(103),
        target=lambda s: s == 30.0,
    ).run(pop)
    assert res.termination_reason == "target_reached", res.termination_reason
    assert res.best.genotype == (2,) * 30 and res.best.score == 30.0


def _run_trap(mode, seed, stop_at_optimum=True, max_generations=1000):
    """One trap run; the budget counts the initial population too."""
    ev = CountingEvaluator(trap(TRAP_K))
    rng = MT19937RandGen(1000 + seed)
    pop = [ev.individual(g) for g in random_population(rng, TRAP_N, [2] * TRAP_L)]
    ev.budget = TRAP_BUDGET
    eng = GomeaEngine(
        [2] * TRAP_L,
        EngineParams(fos_mode=mode, max_generations=max_generations),
        ev,
        MT19937RandGen(2000 + seed),
        MT19937RandGen(3000 + seed),
        target=(lambda s: s == float(TRAP_L)) if stop_at_optimum else None,
    )
    res = eng.run(pop)
    return res, ev


class _RecordingModel(LinkageModel):
    log: list[tuple[int, list[list[int]]]] = []

    def __init__(self, mode, rng):
        super().__init__(mode, rng)
        self._gen = 0

    def learn(self, pop):
        fos = super().learn(pop)
        type(self).log.append((self._gen, fos))
        self._gen += 1
        return fos


@pytest.fixture(scope="module")
def trap_results():
    """Success counts per mode, plus every lt-b FOS learned at generation
    index >= 2. Successful runs stop at the optimum (often in generation 2
    or 3), so the FOS sample comes from separate lt-b runs without a
    target, capped at 4 generations (FOS of generations 2 and 3)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(engine_mod, "LinkageModel", _RecordingModel)
    wins = {}
    lt_b_fos = []
    try:
        for mode in ("lt-b", "lt-mi", "uni"):
            wins[mode] = 0
            for seed in TRAP_SEEDS:
                res, ev = _run_trap(mode, seed)
                assert ev.calls <= TRAP_BUDGET
                if res.best.score == float(TRAP_L):
                    wins[mode] += 1
        for seed in TRAP_SEEDS:
            _RecordingModel.log = []
            _run_trap("lt-b", seed, stop_at_optimum=False, max_generations=4)
            lt_b_fos.extend(f for g, f in _RecordingModel.log if g >= 2)
    finally:
        mp.undo()
    return wins, lt_b_fos


def test_trap_linkage_learning_beats_univariate(trap_results):
    wins, _ = trap_results
    assert max(wins["lt-b"], wins["lt-mi"]) > wins["uni"], wins
    assert max(wins["lt-b"], wins["lt-mi"]) >= len(TRAP_SEEDS) // 2, wins


def test_lt_b_recovers_the_planted_blocks(trap_results):
    _, fos_list = trap_results
    assert len(fos_list) >= len(TRAP_SEEDS), "too few lt-b runs reached generation 2"
    for block in TRUE_BLOCKS:
        hits = sum(1 for fos in fos_list if any(sorted(s) == block for s in fos))
        assert hits / len(fos_list) >= 0.8, (block, hits, len(fos_list))
