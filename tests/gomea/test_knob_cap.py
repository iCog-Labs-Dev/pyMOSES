"""The knob cap (``--gomea-knob-cap``, SERVER-ROUND2-PREREGISTRATION.md).

Off by default (K = 0): no draw, no new telemetry, identical runs (the CLI-level
byte-identity check is ``test_knob_cap_identity.py``).
"""

from __future__ import annotations

import argparse
import copy
import math

import pytest
from diagnostic_objectives import CountingEvaluator, random_population, trap

from pymoses.gomea import gomea_parameters
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.knob_cap import (
    KnobSelector,
    default_explore,
    draw,
    fitness_contrast,
    rank,
)
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.util.rng import MT19937RandGen
from tests.gomea.test_integration import _cli_parser, _events, _run


def _same_state(r1, r2):
    a, b = copy.deepcopy(r1), copy.deepcopy(r2)
    return [a.randint() for _ in range(5)] == [b.randint() for _ in range(5)]


# ---------------------------------------------------------------------------
# Scores, ranking and draws
# ---------------------------------------------------------------------------


def test_fitness_contrast():
    pop = [(0, 0, 1), (1, 0, 1), (1, 0, 1), (0, 0, 1)]
    fit = [1.0, 5.0, 3.0, 2.0]
    # knob 0: differ {5, 3} mean 4, same {1, 2} mean 1.5 -> 2.5; knob 1: none differ;
    # knob 2: all differ.
    assert fitness_contrast(pop, fit, [0, 1, 2]) == [2.5, 0.0, 0.0]
    # The exemplar's value need not be 0.
    assert fitness_contrast(pop, fit, [0], exemplar=(1, 0, 0)) == [2.5]
    # A NaN difference (two infinite means) scores 0.
    assert fitness_contrast(pop, [-math.inf] * 4, [0]) == [0.0]


def test_rank_orders_by_score_and_breaks_ties_at_random():
    items, scores = [10, 11, 12, 13, 14], [1.0, 3.0, 3.0, 3.0, float("nan")]
    orders = {tuple(rank(items, scores, MT19937RandGen(s))) for s in range(30)}
    assert all(o[3:] == (10, 14) for o in orders)  # 1.0, then NaN last
    assert {o[:3] for o in orders} > {(11, 12, 13)}  # the tied three in several orders
    assert all(sorted(o[:3]) == [11, 12, 13] for o in orders)


def test_draw_is_distinct_and_bounded():
    rng = MT19937RandGen(3)
    for k in range(7):
        d = draw(range(5), k, rng)
        assert len(d) == min(k, 5) == len(set(d)) and set(d) <= set(range(5))


def test_default_explore_is_ceil_quarter():
    assert [default_explore(k) for k in (1, 4, 5, 8, 16, 32, 33)] == [1, 1, 2, 2, 4, 8, 9]


# ---------------------------------------------------------------------------
# The selector on a calibrated model
# ---------------------------------------------------------------------------


def _model(pop, mode="lt-b", seed=5):
    m = LinkageModel(mode, MT19937RandGen(seed))
    m.calibrate(pop, [2] * len(pop[0]))
    return m


def _random_model(d=24, n=16, seed=7, mode="lt-b"):
    pop = random_population(MT19937RandGen(seed), n, [2] * d)
    return _model(pop, mode), pop


def _fit(pop):
    return lambda: [float(sum(g)) for g in pop]


def test_static_contrast_picks_the_k_highest_contrasts_and_keeps_them():
    model, pop = _random_model()
    cands = model.search_positions
    c = dict(zip(cands, fitness_contrast(pop, _fit(pop)(), cands), strict=True))
    rng = MT19937RandGen(1)
    s = KnobSelector(5, "static", 0, "contrast", rng)
    sel = s.select(0, pop, _fit(pop), model)
    assert len(sel.selected) == 5 and sel.selected == tuple(sorted(sel.core))
    assert min(c[p] for p in sel.core) >= max(c[p] for p in cands if p not in sel.core)
    assert [c[p] for p in sel.core] == sorted((c[p] for p in sel.core), reverse=True)
    before = copy.deepcopy(rng)
    for g in (1, 2, 3):
        later = s.select(g, pop, _fit(pop), model)
        assert later.selected == sel.selected and later.entered == 0
    assert _same_state(rng, before)  # fixed: nothing drawn after generation 0


def test_entropy_rank0_picks_the_highest_initial_entropies():
    # knob i has i + 1 ones among 8 rows: entropies rise with i up to 4 ones.
    pop = [tuple(1 if r <= i else 0 for i in range(4)) for r in range(8)]
    model = _model(pop)
    s = KnobSelector(2, "static", 0, "entropy", MT19937RandGen(2))
    sel = s.select(0, pop, lambda: pytest.fail("entropy uses no fitness"), model)
    assert sel.core == (3, 2)


def test_random_rank0_is_a_random_order():
    model, pop = _random_model()
    a = KnobSelector(6, "static", 0, "random", MT19937RandGen(9))
    sel = a.select(0, pop, lambda: pytest.fail("random uses no fitness"), model)
    expect = rank(model.search_positions, [0.0] * len(model.search_positions),
                  MT19937RandGen(9))[:6]  # fmt: skip
    assert list(sel.core) == expect
    other = KnobSelector(6, "static", 0, "random", MT19937RandGen(10)).select(0, pop, None, model)
    assert other.selected != sel.selected


def test_random_select_draws_k_once_and_keeps_them():
    model, pop = _random_model()
    s = KnobSelector(7, "random", 0, "contrast", MT19937RandGen(4))
    sel = s.select(0, pop, lambda: pytest.fail("random selection uses no fitness"), model)
    assert list(sel.core) == draw(model.search_positions, 7, MT19937RandGen(4))
    assert all(s.select(g, pop, None, model).selected == sel.selected for g in (1, 2, 3))


def test_explore_core_explore_and_redraws():
    model, pop = _random_model()
    cands = model.search_positions
    k, e = 8, 2
    s = KnobSelector(k, "explore", e, "contrast", MT19937RandGen(11))
    g0 = s.select(0, pop, _fit(pop), model)
    c = dict(zip(cands, fitness_contrast(pop, _fit(pop)(), cands), strict=True))
    assert len(g0.core) == k - e and len(g0.explore) == e and g0.entered == 0
    assert not set(g0.core) & set(g0.explore)
    assert g0.selected == tuple(sorted(g0.core + g0.explore))
    assert min(c[p] for p in g0.core) >= max(c[p] for p in cands if p not in g0.core)
    # From generation 1: the core is the K - E strongest by LT-B linkage strength.
    strengths, _ = model.knob_strengths(pop)
    explores, cores = [], []
    for g in range(1, 7):
        sel = s.select(g, pop, lambda: pytest.fail("no fitness after generation 0"), model)
        assert len(sel.core) == k - e and len(sel.explore) == e
        assert min(strengths[p] for p in sel.core) >= max(
            strengths[p] for p in cands if p not in sel.core
        )
        explores.append(frozenset(sel.explore))
        cores.append(frozenset(sel.core))
    assert len(set(explores)) > 1  # re-drawn every generation
    assert cores[0] != frozenset(g0.core)  # the core changed (contrast -> linkage)


def test_entered_counts_new_core_knobs():
    model, pop = _random_model()
    s = KnobSelector(8, "explore", 2, "random", MT19937RandGen(12))
    prev = s.select(0, pop, None, model)
    total = 0
    for g in range(1, 4):
        sel = s.select(g, pop, None, model)
        assert sel.entered == len(set(sel.core) - set(prev.core))
        total += sel.entered
        prev = sel
    assert s.entered_total == total


def test_cap_above_candidates_selects_all():
    model, pop = _random_model(d=6)
    for mode in ("explore", "static", "random"):
        sel = KnobSelector(10, mode, 3, "contrast", MT19937RandGen(1)).select(
            0, pop, _fit(pop), model
        )
        assert sel.selected == tuple(model.search_positions)


def test_selection_never_draws_from_the_linkage_stream():
    model, pop = _random_model()
    before = copy.deepcopy(model.rng)
    s = KnobSelector(8, "explore", 2, "random", MT19937RandGen(3))
    for g in range(4):
        s.select(g, pop, _fit(pop), model)
    assert _same_state(model.rng, before)


# ---------------------------------------------------------------------------
# LinkageModel.learn_subset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["lt-b", "lt-mi", "rt", "uni"])
def test_learn_subset_over_every_knob_is_learn(mode):
    a, pop = _random_model(mode=mode)
    b, _ = _random_model(mode=mode)
    pop2 = random_population(MT19937RandGen(99), 16, [2] * 24)
    assert a.learn(pop) == b.learn_subset(pop, b.search_positions)
    assert a.last_strengths == b.last_strengths
    assert a.learn(pop2) == b.learn_subset(pop2, b.search_positions)
    assert _same_state(a.rng, b.rng)


@pytest.mark.parametrize("mode,size", [("lt-b", 14), ("lt-mi", 14), ("rt", 14), ("uni", 8)])
def test_fos_size_is_2k_minus_2_for_trees_and_k_for_uni(mode, size):
    model, pop = _random_model(mode=mode)
    sel = model.search_positions[3:11]
    fos = model.learn_subset(pop, sel)
    assert len(fos) == size
    assert {p for s in fos for p in s} == set(sel)
    assert len(model.last_strengths) == len(fos)


def test_learn_subset_ltb_uses_the_h0_rows_of_the_selected_knobs():
    model, pop = _random_model()
    pop2 = random_population(MT19937RandGen(5), 16, [2] * 24)
    sel = list(model.search_positions[2:9])
    _, H = model.knob_strengths(pop2)
    a = copy.deepcopy(model)
    with_h = model.learn_subset(pop2, sel, H)
    without = a.learn_subset(pop2, sel)
    assert with_h == without and model.last_strengths == a.last_strengths


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _run_engine(params, seed=21, d=16, fi=False, rng_knob=True):
    ev = CountingEvaluator(trap(4), budget=4000)
    pop = [ev.individual(g) for g in random_population(MT19937RandGen(seed), 16, [2] * d)]
    gens = [list(pop)]
    records = []

    def on_gen(rec, P):
        records.append(rec)
        gens.append(list(P))

    eng = GomeaEngine(
        [2] * d,
        params,
        ev,
        MT19937RandGen(seed + 1),
        MT19937RandGen(seed + 2),
        target=lambda s: s == float(d),
        on_generation=on_gen,
        rng_knob=MT19937RandGen(seed + 3) if rng_knob else None,
    )
    res = eng.run(pop)
    return res, records, gens, eng


@pytest.mark.parametrize("fos,size", [("lt-b", 10), ("rt", 10), ("uni", 6), ("lt-mi", 10)])
@pytest.mark.parametrize("select", ["explore", "static", "random"])
def test_engine_fos_size_and_outside_knobs_unchanged(fos, size, select):
    # Forced improvements stay off (as in every round-2 arm): when FI finds no
    # improvement it copies the whole elite, which the knob cap does not restrict.
    params = EngineParams(fos_mode=fos, knob_cap=6, knob_select=select,
                          max_generations=8, stall_generations=8)  # fmt: skip
    _, records, gens, eng = _run_engine(params)
    assert records
    for r, before, after in zip(records, gens, gens[1:], strict=False):
        assert r["fos_size"] == size and r["knob_cap"] == 6 and r["knob_select"] == select
        sel = set(r["knob_selected"])
        assert len(sel) == 6
        for x, y in zip(before, after, strict=True):
            assert all(x.genotype[p] == y.genotype[p] for p in range(16) if p not in sel)
    if select != "explore":
        assert len({tuple(r["knob_selected"]) for r in records}) == 1


def test_engine_explore_rotates_and_logs():
    params = EngineParams(knob_cap=6, max_generations=8, stall_generations=8)
    _, records, _, eng = _run_engine(params)
    assert len(records) >= 3
    assert all(r["knob_explore_n"] == 2 and len(r["knob_explore"]) == 2 for r in records)
    assert len({tuple(r["knob_explore"]) for r in records}) > 1
    assert records[0]["knob_core_entered"] == 0
    assert sum(r["knob_core_entered"] for r in records) == eng.knobs.entered_total


def test_engine_k_zero_draws_nothing_new():
    base = _run_engine(EngineParams(), rng_knob=False)
    other = _run_engine(EngineParams(), rng_knob=True)
    assert other[0].population == base[0].population
    assert [{k: v for k, v in r.items() if not k.endswith("_s")} for r in other[1]] == [
        {k: v for k, v in r.items() if not k.endswith("_s")} for r in base[1]
    ]
    assert all(not any(k.startswith("knob_") for k in r) for r in base[1])
    assert _same_state(other[3].rng_knob, MT19937RandGen(24))  # untouched


def test_engine_needs_the_knob_stream():
    with pytest.raises(ValueError, match="rng_knob"):
        _run_engine(EngineParams(knob_cap=4), rng_knob=False)


# ---------------------------------------------------------------------------
# Parameters and CLI
# ---------------------------------------------------------------------------


def test_parameters_default_off_and_left_out_of_telemetry():
    gp = gomea_parameters()
    assert (gp.knob_cap, gp.knob_select, gp.knob_explore, gp.knob_rank0) == (
        0, "explore", None, "contrast")  # fmt: skip
    assert not any(k.startswith("knob_") for k in gp.as_dict())
    on = gomea_parameters(knob_cap=16).as_dict()
    assert (on["knob_cap"], on["knob_select"], on["knob_explore"], on["knob_rank0"]) == (
        16, "explore", 4, "contrast")  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [
        {"knob_cap": -1},
        {"knob_cap": 8, "knob_select": "best"},
        {"knob_cap": 8, "knob_rank0": "fitness"},
        {"knob_cap": 8, "knob_explore": 9},
        {"knob_cap": 8, "knob_explore": -1},
        {"knob_select": "static"},
        {"knob_explore": 2},
        {"knob_rank0": "entropy"},
    ],
)
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        gomea_parameters(**kw)


def test_cli_options_reach_parameters():
    p = _cli_parser(True)
    gp = _gomea_params_from_args(p.parse_args(
        ["--gomea-knob-cap", "16", "--gomea-knob-select", "static", "--gomea-knob-explore", "3",
         "--gomea-knob-rank0", "entropy"]))  # fmt: skip
    assert (gp.knob_cap, gp.knob_select, gp.knob_explore, gp.knob_rank0) == (
        16, "static", 3, "entropy")  # fmt: skip
    gp = _gomea_params_from_args(p.parse_args(["--gomea-knob-cap", "32"]))
    assert (gp.knob_cap, gp.knob_explore) == (32, 8)
    for ns in (p.parse_args([]), argparse.Namespace()):
        gp = _gomea_params_from_args(ns)
        assert (gp.knob_cap, gp.knob_explore) == (0, None)


@pytest.mark.parametrize(
    "argv",
    [
        ["--gomea-knob-cap", "-1"],
        ["--gomea-knob-cap", "8", "--gomea-knob-select", "best"],
        ["--gomea-knob-cap", "8", "--gomea-knob-rank0", "best"],
        ["--gomea-knob-cap", "8", "--gomea-knob-explore", "9"],
        ["--gomea-knob-select", "static"],
        ["--gomea-knob-explore", "2"],
        ["--gomea-knob-rank0", "random"],
    ],
)
def test_invalid_cli_values_rejected(argv):
    with pytest.raises(ValueError):
        _gomea_params_from_args(_cli_parser(True).parse_args(argv))


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--gomea-knob-select", "static"],
        ["--gomea-knob-select", "random"],
        ["--gomea-knob-rank0", "entropy"],
        ["--gomea-knob-rank0", "random"],
        ["--gomea-fos", "uni"],
    ],
)
def test_end_to_end_run_logs_the_knob_cap(tmp_path, monkeypatch, extra):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "g.jsonl"
    argv = ["-H", "mux", "-k", "2", "-a", "gomea", "-m", "1500", "-r", "3001",
            "--gomea-max-evals-per-deme", "250", "--gomea-knob-cap", "8",
            "--gomea-log", str(log), *extra]  # fmt: skip
    rc, stdout = _run(argv)
    assert rc == 0 and stdout
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert starts and all(e["params"]["knob_cap"] == 8 for e in starts)
    gens = [e for e in events if e["type"] == "generation"]
    assert gens
    size = 8 if "uni" in extra else 14
    for g in gens:
        assert len(g["knob_selected"]) == min(8, g["knob_n_candidates"])
        if g["knob_n_candidates"] >= 8:
            assert g["fos_size"] == size
    ends = [e for e in events if e["type"] == "end"]
    assert ends and all("generations" in e and e["knob_cap"] == 8 for e in ends)
    assert all(e["knob_core_entered_total"] >= 0 for e in ends)
