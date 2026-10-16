"""Top-K FOS subsets per generation (``--gomea-fos-topk``).

Off by default (K = 0): no RNG draw, no new telemetry, identical runs.
"""

from __future__ import annotations

import argparse
import math

import pytest
from diagnostic_objectives import CountingEvaluator, onemax, random_population, trap

import pymoses.gomea.engine as engine_mod
from pymoses.gomea import gomea_parameters
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel, linkage_tree_fos, select_fos_topk
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.util.rng import MT19937RandGen
from tests.gomea.test_integration import _cli_parser, _events, _run

# ---------------------------------------------------------------------------
# Merge similarities and strengths
# ---------------------------------------------------------------------------


def test_merge_similarity_is_recorded_without_changing_the_tree():
    S = [
        [0.0, 0.9, 0.1, 0.2],
        [0.9, 0.0, 0.3, 0.1],
        [0.1, 0.3, 0.0, 0.7],
        [0.2, 0.1, 0.7, 0.0],
    ]
    order = [2, 0, 3, 1]
    sims: list[float] = []
    fos = linkage_tree_fos(S, order, sims)
    assert fos == linkage_tree_fos(S, order)
    assert len(sims) == len(fos) - len(order) == 3
    by_set = {frozenset(s): v for s, v in zip(fos[len(order) :], sims, strict=True)}
    assert by_set[frozenset({0, 1})] == 0.9
    assert by_set[frozenset({2, 3})] == 0.7
    # Root: the cardinality-weighted (UPGMA) average of the four cross pairs.
    assert by_set[frozenset({0, 1, 2, 3})] == pytest.approx((0.1 + 0.2 + 0.3 + 0.1) / 4)


def _calibrated(mode, seed=7, n=24, d=12):
    rng = MT19937RandGen(seed)
    pop = random_population(rng, n, [2] * d)
    model = LinkageModel(mode, MT19937RandGen(seed + 1))
    model.calibrate(pop, [2] * d)
    return model, pop


@pytest.mark.parametrize("mode", ["lt-b", "lt-mi"])
def test_strengths_align_with_the_linkage_tree(mode):
    model, pop = _calibrated(mode)
    fos = model.learn(pop)
    s = model.last_strengths
    assert len(s) == len(fos)
    for subset, v in zip(fos, s, strict=True):
        assert (v is None) == (len(subset) == 1)
        assert v is None or math.isfinite(v)
    # UPGMA never inverts: a merge is no more similar than the merges below it.
    strength = {frozenset(x): v for x, v in zip(fos, s, strict=True) if v is not None}
    for a in strength:
        for b in strength:
            if b < a:
                assert strength[b] >= strength[a] - 1e-12


@pytest.mark.parametrize("mode", ["uni", "rt"])
def test_strengths_are_none_without_statistics(mode):
    model, pop = _calibrated(mode)
    fos = model.learn(pop)
    assert model.last_strengths == [None] * len(fos)


def test_strengths_do_not_change_the_fos_or_the_rng():
    a, pop = _calibrated("lt-b")
    b, _ = _calibrated("lt-b")
    fa = a.learn(pop)
    sims: list[float] = []
    # The same draws and the same tree as building it by hand.
    from pymoses.gomea.stats import biased_similarity, entropy_matrix

    d = len(b.search_positions)
    order = b._random_order(d)
    S = biased_similarity(entropy_matrix(pop, b.search_positions), b._H0)
    fb = linkage_tree_fos(S, order, sims)[:-1]
    assert fa == [[b.search_positions[i] for i in s] for s in fb]
    assert a.rng.randint() == b.rng.randint()


# ---------------------------------------------------------------------------
# select_fos_topk
# ---------------------------------------------------------------------------


def _same_state(r1, r2):
    return [r1.randint() for _ in range(5)] == [r2.randint() for _ in range(5)]


@pytest.mark.parametrize("mode", ["linkage", "random"])
@pytest.mark.parametrize("k", [0, 5, 9])
def test_off_or_k_at_least_n_returns_everything_without_draws(mode, k):
    rng, ref = MT19937RandGen(3), MT19937RandGen(3)
    assert select_fos_topk(5, [None] * 5, k, mode, rng) == [0, 1, 2, 3, 4]
    assert _same_state(rng, ref)


def test_linkage_ranks_merges_by_strength_ties_by_index():
    strengths = [None, None, None, 0.5, 0.9, 0.5, 0.2]
    rng, ref = MT19937RandGen(3), MT19937RandGen(3)
    assert select_fos_topk(7, strengths, 3, "linkage", rng) == [4, 3, 5]
    assert select_fos_topk(7, strengths, 1, "linkage", rng) == [4]
    assert _same_state(rng, ref)  # linkage mode draws nothing


def test_linkage_fills_missing_slots_with_singletons_in_fos_order():
    strengths = [None, 0.1, None, None, 0.4]
    assert select_fos_topk(5, strengths, 4, "linkage", MT19937RandGen(1)) == [4, 1, 0, 2]
    assert select_fos_topk(5, None, 2, "linkage", MT19937RandGen(1)) == [0, 1]


def test_random_draws_exactly_k_and_is_a_partial_fisher_yates():
    rng, ref = MT19937RandGen(9), MT19937RandGen(9)
    sel = select_fos_topk(10, None, 4, "random", rng)
    idx = list(range(10))
    for i in range(4):
        j = i + ref.randint_n(10 - i)
        idx[i], idx[j] = idx[j], idx[i]
    assert sel == idx[:4] and len(set(sel)) == 4
    assert _same_state(rng, ref)


def test_random_is_roughly_uniform():
    rng = MT19937RandGen(5)
    counts = [0] * 6
    for _ in range(3000):
        for i in select_fos_topk(6, None, 2, "random", rng):
            counts[i] += 1
    assert all(abs(c - 1000) < 120 for c in counts), counts


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_fos_topk(5, None, 2, "best", MT19937RandGen(1))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _run_engine(params, seed=21, gens=None):
    fn = trap(4)
    ev = CountingEvaluator(fn, budget=4000)
    rng = MT19937RandGen(seed)
    pop = [ev.individual(g) for g in random_population(rng, 16, [2] * 16)]
    records = []
    eng = GomeaEngine(
        [2] * 16,
        params,
        ev,
        MT19937RandGen(seed + 1),
        MT19937RandGen(seed + 2),
        target=lambda s: s == 16.0,
        on_generation=lambda rec, P: records.append(rec),
    )
    res = eng.run(pop)
    return res, records, ev, eng


def _strip_timing(records):
    return [{k: v for k, v in r.items() if not k.endswith("_s")} for r in records]


def test_k_zero_is_identical_whatever_the_mode():
    for params in (EngineParams(fos_topk=0), EngineParams(fos_topk=0, fos_topk_mode="random")):
        base = _run_engine(EngineParams())
        other = _run_engine(params)
        assert other[0].termination_reason == base[0].termination_reason
        assert other[0].population == base[0].population
        assert other[2].evaluated == base[2].evaluated
        assert _strip_timing(other[1]) == _strip_timing(base[1])
        assert _same_state(other[3].rng_var, base[3].rng_var)
        assert _same_state(other[3].rng_link, base[3].rng_link)
    assert all("fos_topk" not in r for r in base[1])


@pytest.mark.parametrize("mode", ["linkage", "random"])
def test_topk_limits_every_generation_and_is_logged(mode):
    k = 5
    res, records, ev, _ = _run_engine(EngineParams(fos_topk=k, fos_topk_mode=mode))
    assert records
    for r in records:
        assert r["fos_topk"] == k and r["fos_topk_mode"] == mode
        assert r["fos_effective_size"] == len(r["fos_selected"]) == min(k, r["fos_size"])
        assert r["stats"]["trials"] <= 16 * k
        assert len(r["fos_selected_strength"]) == r["fos_effective_size"]
        if mode == "linkage":
            st = r["fos_selected_strength"]
            assert all(len(s) >= 2 for s in r["fos_selected"])
            assert st == sorted(st, reverse=True)
        else:
            # Random picks may be singletons (no strength) or merges.
            for sub, v in zip(r["fos_selected"], r["fos_selected_strength"], strict=True):
                assert (v is None) == (len(sub) == 1)
    # The generation's effective mean subset size matches the subsets logged.
    r = records[0]
    assert r["fos_effective_mean_subset_size"] == pytest.approx(
        sum(len(s) for s in r["fos_selected"]) / len(r["fos_selected"])
    )


def test_linkage_selection_is_the_strongest_merges_of_the_learned_tree(monkeypatch):
    seen = {}
    real = LinkageModel.learn

    def spy(self, pop):
        fos = real(self, pop)
        seen["fos"], seen["strengths"] = fos, list(self.last_strengths)
        return fos

    monkeypatch.setattr(LinkageModel, "learn", spy)
    _, records, _, _ = _run_engine(EngineParams(fos_topk=3, max_generations=1))
    fos, st = seen["fos"], seen["strengths"]
    merges = sorted((i for i, v in enumerate(st) if v is not None), key=lambda i: (-st[i], i))
    assert records[0]["fos_selected"] == [fos[i] for i in merges[:3]]


def test_random_mode_draws_k_from_the_linkage_stream(monkeypatch):
    draws = []
    real = engine_mod.select_fos_topk

    def spy(n, strengths, k, mode, rng):
        draws.append(rng)
        return real(n, strengths, k, mode, rng)

    monkeypatch.setattr(engine_mod, "select_fos_topk", spy)
    _, _, _, eng = _run_engine(EngineParams(fos_topk=4, fos_topk_mode="random"))
    assert draws and all(r is eng.rng_link for r in draws)


def test_forced_improvement_uses_the_restricted_fos(monkeypatch):
    seen = []
    real = engine_mod.forced_improvement

    def spy(offspring, elite, fos, ev, rng, stats, **kw):
        seen.append(len(fos))
        return real(offspring, elite, fos, ev, rng, stats, **kw)

    monkeypatch.setattr(engine_mod, "forced_improvement", spy)
    _run_engine(EngineParams(fos_topk=3, forced_improvement=True, max_generations=3))
    assert seen and all(n == 3 for n in seen)


def test_topk_still_solves_onemax():
    ev = CountingEvaluator(onemax(1))
    pop = [ev.individual(g) for g in random_population(MT19937RandGen(4), 16, [2] * 20)]
    res = GomeaEngine(
        [2] * 20,
        EngineParams(fos_topk=6, fos_topk_mode="random", max_generations=200, stall_generations=50),
        ev,
        MT19937RandGen(5),
        MT19937RandGen(6),
        target=lambda s: s == 20.0,
    ).run(pop)
    assert res.termination_reason == "target_reached"


# ---------------------------------------------------------------------------
# Parameters and CLI
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_left_out_of_telemetry():
    gp = gomea_parameters()
    assert (gp.fos_topk, gp.fos_topk_mode) == (0, "linkage")
    d = gp.as_dict()
    assert "fos_topk" not in d and "fos_topk_mode" not in d
    on = gomea_parameters(fos_topk=8, fos_topk_mode="random").as_dict()
    assert (on["fos_topk"], on["fos_topk_mode"]) == (8, "random")


@pytest.mark.parametrize(
    "kw",
    [
        {"fos_topk": -1},
        {"fos_topk_mode": "best"},
        {"fos_topk": 4, "fos": "uni"},
        {"fos_topk": 4, "fos": "rt", "fos_topk_mode": "linkage"},
    ],
)
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        gomea_parameters(**kw)


def test_random_mode_works_with_any_fos():
    for fos in ("uni", "rt", "lt-b", "lt-mi"):
        gomea_parameters(fos=fos, fos_topk=4, fos_topk_mode="random")


def test_cli_options_reach_parameters():
    gp = _gomea_params_from_args(
        _cli_parser(True).parse_args(["--gomea-fos-topk", "8", "--gomea-fos-topk-mode", "random"])
    )
    assert (gp.fos_topk, gp.fos_topk_mode, gp.fos) == (8, "random", "lt-b")
    gp0 = _gomea_params_from_args(_cli_parser(True).parse_args([]))
    assert (gp0.fos_topk, gp0.fos_topk_mode) == (0, "linkage")
    gpn = _gomea_params_from_args(argparse.Namespace())
    assert (gpn.fos_topk, gpn.fos_topk_mode) == (0, "linkage")


@pytest.mark.parametrize(
    "argv",
    [
        ["--gomea-fos-topk", "-1"],
        ["--gomea-fos-topk-mode", "best"],
        ["--gomea-fos-topk", "4", "--gomea-fos", "uni"],
    ],
)
def test_invalid_cli_values_rejected(argv):
    with pytest.raises(ValueError):
        _gomea_params_from_args(_cli_parser(True).parse_args(argv))


def test_end_to_end_run_logs_topk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "g.jsonl"
    argv = ["-H", "pa", "-k", "3", "-a", "gomea", "-m", "600", "-r", "1",
            "--gomea-max-evals-per-deme", "200", "--gomea-fos-topk", "4",
            "--gomea-log", str(log)]  # fmt: skip
    rc, stdout = _run(argv)
    assert rc == 0 and stdout
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert starts and all(e["params"]["fos_topk"] == 4 for e in starts)
    gens = [e for e in events if e["type"] == "generation"]
    assert gens
    for g in gens:
        assert g["fos_effective_size"] == min(4, g["fos_size"])
        assert g["stats"]["trials"] <= 16 * 4
