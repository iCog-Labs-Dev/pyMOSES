"""GOM, forced improvement and engine boundaries, driven by a scripted
evaluator with exact call counts."""

from __future__ import annotations

import dataclasses

import pytest
from diagnostic_objectives import CountingEvaluator, onemax

import pymoses.gomea.engine as engine_mod
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.gom import (
    GomInterrupted,
    GomStats,
    Individual,
    forced_improvement,
    gom,
)
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.util.rng import MT19937RandGen


class ScriptedEvaluator:
    """Each genotype's score is predetermined; an unscripted genotype is a
    test failure. ``budget`` caps the number of ``evaluate`` calls."""

    def __init__(self, scores, pkey=None, budget=None):
        self.scores = {tuple(k): v for k, v in scores.items()}
        self.pkey = pkey
        self.budget = budget
        self.calls: list[tuple[int, ...]] = []
        self.noop_skips = 0
        self.phenotype_skips = 0

    def check(self):
        if self.budget is not None and len(self.calls) >= self.budget:
            return "global_budget"
        return None

    def evaluate(self, g):
        reason = self.check()
        if reason is not None:
            raise BudgetExhausted(reason)
        g = tuple(g)
        assert g in self.scores, f"unscripted genotype {g}"
        self.calls.append(g)
        return self.scores[g]

    def phenotype_key(self, g):
        return None if self.pkey is None else self.pkey(tuple(g))

    def worse(self, a, b):
        return a < b

    def record_noop_skip(self):
        self.noop_skips += 1

    def record_phenotype_skip(self):
        self.phenotype_skips += 1

    def ind(self, g, score=None):
        g = tuple(g)
        return Individual(g, self.scores[g] if score is None else score, self.phenotype_key(g))


def _rng(seed=1):
    return MT19937RandGen(seed)


# ------------------------------------------------------------------ gom


def test_worsening_trials_are_rolled_back():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 9.0, (1, 0): 3.0, (0, 1): 4.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0], [1]], ev, _rng(), stats, True)
    assert off.genotype == (0, 0) and off.score == 5.0
    assert (changed, improved) == (False, False)
    assert sorted(ev.calls) == [(0, 1), (1, 0)]
    assert stats.evaluated == 2 and stats.rejected == 2 and stats.trials == 2
    assert stats.accepted_improving == stats.accepted_neutral == 0


def test_rollback_restores_state_between_subsets():
    # FOS [[0], [0, 1]]: whichever runs first, a rejected trial must not leak
    # into the next subset's trial.
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 1.0, (1, 0): 1.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    off, changed, _ = gom(rec, [donor], [[0], [0, 1]], ev, _rng(), GomStats(), True)
    assert off == rec and not changed
    assert sorted(ev.calls) == [(1, 0), (1, 1)]


def test_neutral_trials_are_accepted():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 5.0, (1, 0): 5.0, (0, 1): 5.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0], [1]], ev, _rng(), stats, True)
    assert off.genotype == (1, 1) and off.score == 5.0
    assert (changed, improved) == (True, False)
    assert len(ev.calls) == 2
    assert stats.accepted_neutral == 2 and stats.rejected == 0


def test_strict_improvement_is_reported():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 9.0, (1, 0): 6.0, (0, 1): 6.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0], [1]], ev, _rng(), stats, True)
    # The first subset improves to 6; the second then yields (1, 1) = 9.
    assert off.genotype == (1, 1) and off.score == 9.0
    assert (changed, improved) == (True, True)
    assert stats.accepted_improving == 2 and len(ev.calls) == 2


def test_noop_subsets_are_skipped_without_evaluation():
    ev = ScriptedEvaluator({(0, 0): 5.0})
    rec = ev.ind((0, 0))
    stats = GomStats()
    off, changed, improved = gom(rec, [rec], [[0], [1], [0, 1]], ev, _rng(), stats, True)
    assert off == rec and (changed, improved) == (False, False)
    assert ev.calls == []
    assert stats.noop_skips == 3 and ev.noop_skips == 3 and stats.evaluated == 0


def test_proven_phenotype_skip_accepts_without_evaluation():
    # Field 1 never affects the phenotype: pkey is field 0 alone.
    ev = ScriptedEvaluator({(0, 0): 5.0, (0, 1): 1.0}, pkey=lambda g: g[0])
    rec, donor = ev.ind((0, 0)), ev.ind((0, 1), score=1.0)
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[1]], ev, _rng(), stats, True)
    assert off.genotype == (0, 1) and off.score == 5.0 and off.pkey == 0
    assert (changed, improved) == (True, False)
    assert ev.calls == []
    assert stats.phenotype_skips == 1 and ev.phenotype_skips == 1
    assert stats.accepted_neutral == 1 and stats.evaluated == 0


def test_phenotype_skip_disabled_evaluates():
    ev = ScriptedEvaluator({(0, 0): 5.0, (0, 1): 1.0}, pkey=lambda g: g[0])
    rec, donor = ev.ind((0, 0)), ev.ind((0, 1))
    stats = GomStats()
    off, _, _ = gom(rec, [donor], [[1]], ev, _rng(), stats, False)
    assert ev.calls == [(0, 1)] and off == rec
    assert stats.phenotype_skips == 0 and stats.rejected == 1


def test_none_pkey_never_proves_a_skip():
    ev = ScriptedEvaluator({(0, 0): 5.0, (0, 1): 1.0}, pkey=lambda g: None)
    rec, donor = ev.ind((0, 0)), ev.ind((0, 1))
    gom(rec, [donor], [[1]], ev, _rng(), GomStats(), True)
    assert ev.calls == [(0, 1)]


def test_donors_and_recipient_are_never_mutated():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 9.0, (1, 0): 6.0, (0, 1): 6.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    donors = [rec, donor]
    snapshot = [dataclasses.replace(d) for d in donors]
    for seed in range(5):
        off, _, _ = gom(rec, donors, [[0], [1]], ev, _rng(seed), GomStats(), True)
        assert off is not rec
    assert donors == snapshot and donors[0] is rec and donors[1] is donor
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.genotype = (1, 1)  # type: ignore[misc]


def test_engine_passes_the_generation_start_snapshot_to_every_recipient(monkeypatch):
    seen = []
    real_gom = engine_mod.gom

    def spy(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance="nonworse"):
        seen.append([d.genotype for d in donors])
        return real_gom(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance)

    monkeypatch.setattr(engine_mod, "gom", spy)
    ev = CountingEvaluator(onemax(1))
    rng = MT19937RandGen(3)
    pop = [ev.individual(tuple(rng.randint_n(2) for _ in range(6))) for _ in range(8)]
    eng = GomeaEngine(
        [2] * 6,
        EngineParams(fos_mode="uni", max_generations=1),
        ev,
        MT19937RandGen(4),
        MT19937RandGen(5),
    )
    start = [ind.genotype for ind in pop]
    res = eng.run(pop)
    assert res.generations == 1 and len(seen) == 8
    assert all(s == start for s in seen)
    assert [ind.genotype for ind in res.population] != start  # something changed


def test_gom_budget_interrupt_carries_last_accepted_state():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 1): 9.0, (1, 0): 6.0, (0, 1): 6.0}, budget=1)
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    with pytest.raises(GomInterrupted) as info:
        gom(rec, [donor], [[0], [1]], ev, _rng(), GomStats(), True)
    assert info.value.reason == "global_budget"
    off = info.value.offspring
    assert off.score == 6.0 and off.genotype in {(1, 0), (0, 1)}
    assert isinstance(info.value.__cause__, BudgetExhausted)


# ------------------------------------------------------------------ FI


def test_fi_without_strict_improvement_copies_the_elite():
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (1, 0): 1.0, (0, 1): 1.0})
    off, elite = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    out = forced_improvement(off, elite, [[0], [1]], ev, _rng(), stats)
    assert (out.genotype, out.score, out.pkey) == (elite.genotype, elite.score, elite.pkey)
    assert sorted(ev.calls) == [(0, 1), (1, 0)]  # neutral results are not accepted
    assert stats.fi_events == 1 and stats.fi_elite_copies == 1
    assert stats.fi_evaluated == 2 and stats.fi_improved == 0 and stats.fi_trials == 2
    assert stats.evaluated == stats.rejected == stats.accepted_neutral == 0


def test_fi_accepts_first_strict_improvement_and_stops():
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (1, 0): 2.0, (0, 1): 2.0})
    off, elite = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    out = forced_improvement(off, elite, [[0], [1]], ev, _rng(), stats)
    assert len(ev.calls) == 1
    assert out.genotype == ev.calls[0] and out.score == 2.0
    assert stats.fi_events == 1 and stats.fi_elite_copies == 0
    assert stats.fi_improved == 1 and stats.fi_evaluated == 1
    assert stats.accepted_improving == stats.evaluated == stats.trials == 0


def test_fi_skips_noops_and_proven_equal_phenotypes():
    ev = ScriptedEvaluator({(0, 0): 1.0, (0, 1): 3.0}, pkey=lambda g: g[0])
    off, elite = ev.ind((0, 0)), ev.ind((0, 1))
    stats = GomStats()
    out = forced_improvement(off, elite, [[0], [1]], ev, _rng(), stats)
    assert ev.calls == []
    assert stats.noop_skips == 1 and stats.phenotype_skips == 1
    assert out.genotype == (0, 1) and stats.fi_elite_copies == 1


# ------------------------------------------------------------------ engine boundaries


def _engine(domains, ev, **kw):
    target = kw.pop("target", None)
    on_generation = kw.pop("on_generation", None)
    return GomeaEngine(
        domains,
        EngineParams(**kw),
        ev,
        MT19937RandGen(11),
        MT19937RandGen(12),
        target=target,
        on_generation=on_generation,
    )


def test_budget_exhaustion_mid_generation_keeps_a_valid_population():
    fn = onemax(2)
    ev = CountingEvaluator(fn)
    rng = MT19937RandGen(7)
    pop = [ev.individual(tuple(rng.randint_n(3) for _ in range(20))) for _ in range(16)]
    ev.budget = ev.calls + 37  # ends part-way through the first generation
    res = _engine([3] * 20, ev, fos_mode="lt-b").run(pop)
    assert res.termination_reason == "global_budget"
    assert ev.calls == ev.budget
    assert len(res.population) == 16
    for ind in res.population:
        assert ind.score == fn(ind.genotype)
    # untouched parents are carried over unchanged, in order
    assert res.population[-1] is pop[-1]
    assert res.best.score == max(ind.score for ind in res.population)
    assert res.generations == 1 and len(res.history) == 1


def test_no_search_fields_when_population_is_constant():
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual((1, 0, 2)) for _ in range(5)]
    before = ev.calls
    res = _engine([2, 2, 3], ev).run(pop)
    assert res.termination_reason == "no_search_fields"
    assert res.generations == 0 and ev.calls == before and res.history == []


def test_single_search_field_works():
    ev = CountingEvaluator(onemax(2))
    pop = [ev.individual((0, v)) for v in (0, 1, 2, 1)]
    res = _engine([1, 3], ev, stall_generations=3).run(pop)
    assert res.termination_reason in {"converged", "stagnation"}
    assert res.generations >= 1
    assert all(h["fos_size"] == 1 for h in res.history)
    for parent, child in zip(pop, res.population, strict=True):
        assert child.score >= parent.score and child.genotype[0] == 0


def test_target_met_by_initial_population():
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual(g) for g in ((0, 0), (1, 1))]
    res = _engine([2, 2], ev, target=lambda s: s >= 2.0).run(pop)
    assert res.termination_reason == "target_reached" and res.generations == 0
    assert res.best.genotype == (1, 1)


def test_single_individual_has_no_search_fields():
    ev = CountingEvaluator(onemax(1))
    res = _engine([2, 2], ev).run([ev.individual((0, 1))])
    assert res.termination_reason == "no_search_fields"


def test_max_generations_and_history():
    ev = CountingEvaluator(onemax(2))
    rng = MT19937RandGen(9)
    pop = [ev.individual(tuple(rng.randint_n(3) for _ in range(40))) for _ in range(6)]
    records = []
    res = _engine(
        [3] * 40,
        ev,
        fos_mode="uni",
        max_generations=2,
        stall_generations=100,
        on_generation=lambda rec, p: records.append((rec, list(p))),
    ).run(pop)
    assert res.termination_reason in {"max_generations", "converged"}
    assert res.generations == len(res.history) == len(records) <= 2
    h = res.history[0]
    assert set(h) == {
        "generation",
        "fos_size",
        "fos_mean_subset_size",
        "stats",
        "nis",
        "best_improved",
        "linkage_s",
        "variation_s",
    }
    assert h["generation"] == 0 and h["fos_size"] == 40 and h["fos_mean_subset_size"] == 1.0
    assert set(h["stats"]) == {f.name for f in dataclasses.fields(GomStats)}


def test_keyboard_interrupt_is_recorded_and_reraised():
    class Boom(CountingEvaluator):
        def evaluate(self, g):
            raise KeyboardInterrupt

    ev = Boom(onemax(1))
    pop = [Individual((0, 1), 1.0, None), Individual((1, 0), 1.0, None)]
    eng = _engine([2, 2], ev)
    with pytest.raises(KeyboardInterrupt):
        eng.run(pop)
    assert eng.last_result.termination_reason == "interrupted"
    assert len(eng.last_result.population) == 2


def test_forced_improvement_in_engine():
    ev = CountingEvaluator(onemax(2))
    # Seed chosen so that FI fires and strictly improves an offspring.
    rng = MT19937RandGen(22)
    pop = [ev.individual(tuple(rng.randint_n(3) for _ in range(12))) for _ in range(10)]
    records = []
    eng = _engine(
        [3] * 12,
        ev,
        fos_mode="uni",
        forced_improvement=True,
        target=lambda s: s == 12.0,
        max_generations=50,
        stall_generations=20,
        on_generation=lambda rec, p: records.append(rec),
    )
    res = eng.run(pop)
    assert sum(r["stats"]["fi_events"] for r in records) > 0
    assert sum(r["stats"]["fi_improved"] for r in records) > 0
    assert res.termination_reason == "target_reached"
    assert res.best.score >= max(i.score for i in pop)
    assert eng.elite.score == res.best.score


# ------------------------------------------------------------------ fix round 1


class _StubModel:
    """A linkage model whose ``learn`` returns a fixed FOS."""

    def __init__(self, fos):
        self.fos = fos

    def learn(self, pop):
        return [list(s) for s in self.fos]


def test_fi_never_returns_something_worse_than_the_offspring():
    # The offspring (10) beats the elite (5): no trial improves, and the
    # result is the offspring unchanged, not an elite copy.
    ev = ScriptedEvaluator({(0, 0): 10.0, (1, 1): 5.0, (1, 0): 7.0, (0, 1): 7.0})
    off, elite = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    out = forced_improvement(off, elite, [[0], [1]], ev, _rng(), stats)
    assert out is off
    assert stats.fi_events == 1 and stats.fi_elite_copies == 0


def test_fi_counters_are_separate_from_gom_counters():
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (1, 0): 1.0, (0, 1): 2.0})
    off, elite = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    forced_improvement(off, elite, [[0], [1], [0, 1]], ev, _rng(), stats)
    assert stats.trials == stats.evaluated == 0
    assert stats.accepted_improving == stats.rejected == stats.accepted_neutral == 0
    assert stats.fi_trials >= 1 and stats.fi_evaluated == len(ev.calls) >= 1
    assert stats.fi_improved == 1


def test_fi_phenotype_skip_is_counted_and_reported():
    ev = ScriptedEvaluator({(0, 0): 1.0, (0, 1): 3.0}, pkey=lambda g: g[0])
    off, elite = ev.ind((0, 0)), ev.ind((0, 1))
    stats = GomStats()
    forced_improvement(off, elite, [[1]], ev, _rng(), stats)
    assert ev.calls == [] and stats.phenotype_skips == 1 and ev.phenotype_skips == 1
    assert stats.fi_trials == 1 and stats.fi_evaluated == 0


def _gen_engine(ev, fi=True, target=None):
    return GomeaEngine(
        [2, 2],
        EngineParams(forced_improvement=fi),
        ev,
        MT19937RandGen(11),
        MT19937RandGen(12),
        target=target,
    )


def _script_gom(monkeypatch, results):
    """Replace ``engine.gom`` with one returning ``results`` in call order."""
    it = iter(results)
    calls = []

    def fake(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance="nonworse"):
        calls.append(recipient)
        return next(it)

    monkeypatch.setattr(engine_mod, "gom", fake)
    return calls


def _spy_fi(monkeypatch):
    seen = []
    real = engine_mod.forced_improvement

    def spy(offspring, elite, fos, ev, rng, stats):
        seen.append((offspring, elite))
        return real(offspring, elite, fos, ev, rng, stats)

    monkeypatch.setattr(engine_mod, "forced_improvement", spy)
    return seen


def test_improved_offspring_becomes_the_elite_within_the_generation(monkeypatch):
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 2.0, (1, 0): 5.0})
    a, b = ev.ind((0, 0)), ev.ind((1, 1))
    better = ev.ind((1, 0))
    # Recipient 0 strictly improves to (1, 0) = 5, past the start elite (1, 1).
    # Recipient 1 is unchanged, so FI fires and must use (1, 0) as the elite.
    _script_gom(monkeypatch, [(better, True, True), (b, False, False)])
    seen = _spy_fi(monkeypatch)
    eng = _gen_engine(ev)
    eng.elite = b
    new_pop, _, stop = engine_mod.perform_generation(eng, [a, b], _StubModel([[0], [1]]), 0, 0)
    assert stop is None
    assert len(seen) == 1 and seen[0][1] is better
    # FI of (1, 1) with elite (1, 0): subset [1] yields (1, 0) = 5, a strict improvement.
    assert new_pop[1].genotype == (1, 0) and new_pop[1].score == 5.0
    assert eng.elite.score == 5.0


def test_fi_result_better_than_elite_becomes_the_elite(monkeypatch):
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (0, 1): 4.0})
    a, b = ev.ind((0, 0)), ev.ind((1, 1))
    _script_gom(monkeypatch, [(a, False, False), (b, False, False)])
    eng = _gen_engine(ev)
    eng.elite = b
    new_pop, _, _ = engine_mod.perform_generation(eng, [a, b], _StubModel([[1]]), 0, 0)
    # FI of (0, 0) with elite (1, 1) over [[1]] gives (0, 1) = 4 > 3.
    assert new_pop[0].genotype == (0, 1)
    assert eng.elite.genotype == (0, 1) and eng.elite.score == 4.0


def test_engine_initialises_the_elite_from_the_initial_population():
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual(g) for g in ((0, 0), (1, 0), (0, 1))]
    eng = _engine([2, 2], ev, max_generations=1)
    eng.run(pop)
    assert eng.elite.score >= 1.0
    assert not any(eng.ev.worse(eng.elite.score, ind.score) for ind in eng.last_result.population)


@pytest.mark.parametrize(
    ("changed", "nis", "fires"),
    [(False, 0, True), (True, 2, True), (True, 1, False), (True, 0, False)],
)
def test_fi_trigger(monkeypatch, changed, nis, fires):
    # N = 2, so the threshold is 1 + floor(log10(2)) = 1: FI fires iff the
    # offspring did not change or nis > 1.
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (0, 1): 2.0, (1, 0): 2.0})
    a, b = ev.ind((0, 0)), ev.ind((1, 1))
    off = ev.ind((0, 1)) if changed else a
    _script_gom(monkeypatch, [(off, changed, False), (b, True, False)])
    seen = _spy_fi(monkeypatch)
    eng = _gen_engine(ev)
    eng.elite = b
    engine_mod.perform_generation(eng, [a, b], _StubModel([[0], [1]]), nis, 0)
    assert [s[0] for s in seen] == ([off] if fires else [])


def test_fi_is_skipped_when_the_offspring_is_the_elite(monkeypatch):
    ev = ScriptedEvaluator({(0, 0): 1.0, (1, 1): 3.0, (1, 0): 0.5})
    a, b = ev.ind((0, 0)), ev.ind((1, 1))
    _script_gom(monkeypatch, [(a, True, False), (b, False, False)])
    seen = _spy_fi(monkeypatch)
    eng = _gen_engine(ev)
    eng.elite = b
    _, rec, _ = engine_mod.perform_generation(eng, [a, b], _StubModel([[0]]), 5, 0)
    assert seen[0][0] is a and len(seen) == 1  # b is the elite: no FI for it
    assert rec["stats"]["fi_events"] == 1


def test_target_reached_mid_generation_keeps_untouched_parents(monkeypatch):
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual(g) for g in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    winner = Individual((1, 1, 1), 3.0, None)
    calls = _script_gom(monkeypatch, [(winner, True, True)])
    res = _engine([2, 2, 2], ev, target=lambda s: s >= 3.0).run(pop)
    assert res.termination_reason == "target_reached"
    assert len(calls) == 1 and res.generations == 1
    assert res.population[0] is winner
    assert res.population[1] is pop[1] and res.population[2] is pop[2]


def test_empty_fos_stops_with_no_valid_moves(monkeypatch):
    class EmptyModel(engine_mod.LinkageModel):
        def learn(self, pop):
            return []

    monkeypatch.setattr(engine_mod, "LinkageModel", EmptyModel)
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual(g) for g in ((0, 1), (1, 0))]
    before = ev.calls
    res = _engine([2, 2], ev).run(pop)
    assert res.termination_reason == "no_valid_moves"
    assert res.generations == 0 and ev.calls == before


# ------------------------------------------------ acceptance (Phase 1d, hypothesis E)


def test_equal_score_trial_is_accepted_under_nonworse():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 0): 5.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 0))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0]], ev, _rng(), stats, False, "nonworse")
    assert off.genotype == (1, 0) and off.score == 5.0
    assert (changed, improved) == (True, False)
    assert stats.accepted_neutral == 1 and stats.rejected == 0


def test_equal_score_trial_is_rejected_under_strict():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 0): 5.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 0))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0]], ev, _rng(), stats, False, "strict")
    assert off == rec  # restored
    assert (changed, improved) == (False, False)
    assert ev.calls == [(1, 0)]  # still evaluated once
    assert stats.evaluated == 1 and stats.rejected == 1 and stats.accepted_neutral == 0


def test_strict_still_accepts_a_strict_improvement_and_rejects_a_worse_trial():
    ev = ScriptedEvaluator({(0, 0): 5.0, (1, 0): 6.0, (0, 1): 4.0, (1, 1): 4.0})
    rec, donor = ev.ind((0, 0)), ev.ind((1, 1))
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[0], [1]], ev, _rng(), stats, False, "strict")
    assert off.genotype == (1, 0) and off.score == 6.0
    assert (changed, improved) == (True, True)
    assert stats.accepted_improving == 1 and stats.rejected == 1


def test_strict_rejects_a_proven_phenotype_skip_without_evaluation():
    ev = ScriptedEvaluator({(0, 0): 5.0, (0, 1): 1.0}, pkey=lambda g: g[0])
    rec, donor = ev.ind((0, 0)), ev.ind((0, 1), score=1.0)
    stats = GomStats()
    off, changed, improved = gom(rec, [donor], [[1]], ev, _rng(), stats, True, "strict")
    assert off == rec and (changed, improved) == (False, False)
    assert ev.calls == []
    assert stats.phenotype_skips == 1 and ev.phenotype_skips == 1
    assert stats.rejected == 1 and stats.accepted_neutral == 0 and stats.evaluated == 0


def test_unknown_acceptance_is_rejected():
    ev = ScriptedEvaluator({(0, 0): 5.0})
    with pytest.raises(ValueError):
        gom(ev.ind((0, 0)), [ev.ind((0, 0))], [[0]], ev, _rng(), GomStats(), False, "bogus")


def test_engine_defaults_to_nonworse_and_passes_acceptance_to_gom(monkeypatch):
    assert EngineParams().acceptance == "nonworse"
    seen = []
    real_gom = engine_mod.gom

    def spy(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance="nonworse"):
        seen.append(acceptance)
        return real_gom(recipient, donors, fos, ev, rng, stats, skip_inactive, acceptance)

    monkeypatch.setattr(engine_mod, "gom", spy)
    for mode in ("nonworse", "strict"):
        seen.clear()
        ev = CountingEvaluator(onemax(1))
        rng = MT19937RandGen(3)
        pop = [ev.individual(tuple(rng.randint_n(2) for _ in range(6))) for _ in range(8)]
        GomeaEngine(
            [2] * 6,
            EngineParams(fos_mode="uni", max_generations=1, acceptance=mode),
            ev,
            MT19937RandGen(4),
            MT19937RandGen(5),
        ).run(pop)
        assert seen and set(seen) == {mode}
