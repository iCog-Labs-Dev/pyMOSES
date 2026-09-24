"""Tests for pymoses.gomea.optimizer (the MOSES optimizer seam) and
pymoses.gomea.telemetry.

Every test drives ``gomea_optimization`` exactly the way
``deme_expander.optimize_demes`` drives hill climbing
(``optimizer(deme, iscorer, max_evals, max_time)``, see
``docs/gomea/P1.0-AUDIT.md`` Sec 3), against a real Boolean representation
(``make_boolean_rep``, conftest.py) and a real ``complexity_based_scorer``
over a 3-input even-parity (``pa`` -3) truth table.
"""

import itertools
import json
import math

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table import ITable, OTable, Table, type_node
from pymoses.comboreduct.type_checker.type_signature import gen_signature
from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.gomea.optimizer import TERMINATION_REASONS
from pymoses.gomea.schema import FrozenSchema, UnsupportedRepresentation
from pymoses.gomea.telemetry import TelemetrySink
from pymoses.moses.main.demo_problems import even_parity
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.instance_scorer import complexity_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.representation import representation
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import ctruth_table_bscore
from pymoses.util.rng import randGen

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parity_ctable(arity: int):
    """The full even-parity truth table of ``arity`` Boolean inputs (the
    ``pa`` demo problem's target, ``demo_problems.even_parity``)."""
    t, f = builtin.logical_true, builtin.logical_false
    it = ITable(types=[type_node.boolean_type] * arity)
    outs = []
    for bits in itertools.product([False, True], repeat=arity):
        it.push_back(tuple(t if b else f for b in bits))
        outs.append(t if even_parity(bits) else f)
    return Table(OTable(outs), it).compressed()


def _iscorer(rep, arity: int) -> complexity_based_scorer:
    bsc = ctruth_table_bscore(_parity_ctable(arity))
    cscorer = behave_cscore(bsc)
    return complexity_based_scorer(cscorer, rep, True)


@pytest.fixture
def pa3(make_boolean_rep):
    """A fresh ``pa`` -3 problem: (representation, iscorer, empty deme).
    Exemplar ``and`` (dim 12, every field a 3-valued logical knob)."""

    def _make(exemplar: str = "and", arity: int = 3):
        rep = make_boolean_rep(exemplar, arity)
        return rep, _iscorer(rep, arity), instance_set(rep.fields())

    return _make


def _deme_contents(deme):
    return [
        (
            tuple(si.first),
            si.second.get_score(),
            si.second.get_complexity(),
            si.second.get_penalized_score(),
        )
        for si in deme
    ]


def _without_timings(result: dict) -> dict:
    out = dict(result)
    out.pop("timings")
    return out


def _run(make, seed, gp=None, op=None, max_evals=200, max_time=10**6, exemplar="and", arity=3):
    rep, isc, deme = make(exemplar, arity)
    randGen().seed(seed)
    opt = gomea_optimization(op, gp)
    opt(deme, isc, max_evals, max_time)
    return opt, deme


def _events(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_default_parameters():
    gp = gomea_parameters()
    assert gp.pop_size == 64
    assert gp.fos == "lt-b"
    assert gp.forced_improvement is False
    assert gp.skip_inactive is False
    assert gp.radii == "1:0.4,2:0.3,4:0.2,max:0.1"
    assert gp.dedupe is True
    assert gp.max_generations == 100
    assert gp.stall_generations == 5
    assert gp.max_evals_per_deme is None
    assert gp.telemetry_path is None


def test_cli_skip_inactive_defaults_off_and_stays_available():
    """F6: ``--gomea-skip-inactive`` defaults to 0 (off), both for a parsed
    command line without the flag and for a namespace built without it,
    and ``1`` still turns it on."""
    import argparse

    from pymoses.moses.main.problem_params import (
        _GOMEA_OPTIONS,
        _gomea_params_from_args,
        problem_params,
    )

    assert {dest: default for _, dest, _, default in _GOMEA_OPTIONS}["gomea_skip_inactive"] == 0
    assert _gomea_params_from_args(argparse.Namespace()).skip_inactive is False

    parser = argparse.ArgumentParser()
    problem_params().add_gomea_options(parser)
    assert _gomea_params_from_args(parser.parse_args([])).skip_inactive is False
    on = parser.parse_args(["--gomea-skip-inactive", "1"])
    assert _gomea_params_from_args(on).skip_inactive is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pop_size": 1},
        {"fos": "bogus"},
        {"max_generations": 0},
        {"stall_generations": 0},
        {"max_evals_per_deme": 0},
        {"radii": "1:0.5,x"},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        gomea_parameters(**kwargs)


@pytest.mark.parametrize("fos", ["lt-b", "lt-mi", "rt", "uni"])
def test_every_fos_mode_is_accepted_and_runs(pa3, fos):
    opt, deme = _run(pa3, 3, gomea_parameters(pop_size=8, fos=fos), max_evals=60)
    assert deme.n_evals == len(deme) == opt.last_result["ledger"]["requests"]


# ---------------------------------------------------------------------------
# The seam contract (docs/gomea/P1.0-AUDIT.md Sec 3)
# ---------------------------------------------------------------------------


def test_deme_holds_exactly_the_evaluated_trials(pa3):
    rep, isc, deme = pa3()
    randGen().seed(1)
    opt = gomea_optimization(gp=gomea_parameters(pop_size=16))
    opt(deme, isc, 300, 10**6)
    res = opt.last_result
    requests = res["ledger"]["requests"]
    assert requests > 0
    assert deme.n_evals == len(deme) == requests
    # Every archived trial carries exactly the score the scorer gives it.
    fresh = _iscorer(rep, 3)
    for si in deme:
        assert fresh(si.first) == si.second
    # The first archived trial is the exemplar (the default genotype).
    schema = FrozenSchema(rep)
    assert schema.to_genotype(deme[0].first) == schema.default_genotype()
    # optim_stats, as hill climbing sets them.
    assert opt.nsteps == res["generations"]
    assert opt.total_steps == res["generations"]
    assert opt.total_evals == requests
    assert opt.demeID == deme.getID()
    assert opt.over_budget == (
        res["termination_reason"] in {"global_budget", "inner_budget", "time_limit"}
    )


@pytest.mark.parametrize("max_evals", [10, 50, 200])
def test_budget_is_never_exceeded(pa3, max_evals):
    opt, deme = _run(pa3, 5, gomea_parameters(pop_size=32), max_evals=max_evals)
    requests = opt.last_result["ledger"]["requests"]
    assert requests <= max_evals
    assert deme.n_evals == len(deme) == requests


class _CountingScorer:
    """Wraps a real ``complexity_based_scorer`` and counts its calls. Keeps
    ``get_rep`` and ``_cscorer`` so the optimizer and oracle see the same
    interface as the wrapped scorer."""

    def __init__(self, inner):
        self.inner = inner
        self._cscorer = inner._cscorer
        self.calls = 0

    def get_rep(self):
        return self.inner.get_rep()

    def __call__(self, inst):
        self.calls += 1
        return self.inner(inst)


def test_contract5_budget_sweep_across_forced_improvement_boundaries(pa3, monkeypatch):
    """F4(a), Contract 5: with forced improvements on and a small N, sweep
    budgets from 1 to past the unbudgeted run's own request count, covering
    every budget that runs out inside a forced improvement. For every
    budget, scorer calls == deme.n_evals == len(deme) == ledger.requests
    <= budget; and whenever the budget cut the run short (the unbudgeted
    run of the same seed made more requests), the reason is the budget's."""
    import pymoses.gomea.engine as engine_mod
    from pymoses.gomea.gom import GomInterrupted
    from pymoses.gomea.optimizer import BUDGET_REASONS

    real_fi = engine_mod.forced_improvement
    fi_log: list[tuple[int, int, bool]] = []  # (requests before, after, interrupted)

    def spy(off, elite, fos, ev, rng, stats):
        before = ev.oracle.ledger.requests
        try:
            out = real_fi(off, elite, fos, ev, rng, stats)
        except GomInterrupted:
            fi_log.append((before, ev.oracle.ledger.requests, True))
            raise
        fi_log.append((before, ev.oracle.ledger.requests, False))
        return out

    monkeypatch.setattr(engine_mod, "forced_improvement", spy)
    gp = gomea_parameters(
        pop_size=6, forced_improvement=True, stall_generations=50, max_generations=50
    )

    def run(budget):
        rep, isc, deme = pa3()
        counting = _CountingScorer(isc)
        randGen().seed(2)
        opt = gomea_optimization(gp=gp)
        opt(deme, counting, budget, 10**6)
        return opt, deme, counting

    ref_opt, _, _ = run(10**6)
    ref_requests = ref_opt.last_result["ledger"]["requests"]
    fi_windows = [(a, b) for a, b, _ in fi_log if b > a]
    assert fi_windows, "the reference run must evaluate inside a forced improvement"

    # Every budget strictly inside an FI evaluation window (and its edges),
    # plus a stride over the whole run and the budgets around its end.
    inside = {b for a, e in fi_windows for b in range(a, e)}
    budgets = sorted(
        inside
        | {e for _, e in fi_windows}
        | set(range(1, ref_requests, 9))
        | {ref_requests - 1, ref_requests, ref_requests + 1}
    )
    interrupted_in_fi = []
    for budget in budgets:
        fi_log.clear()
        opt, deme, counting = run(budget)
        res = opt.last_result
        requests = res["ledger"]["requests"]
        assert counting.calls == deme.n_evals == len(deme) == requests <= budget, budget
        if ref_requests > budget:
            assert requests == budget, budget
            assert res["termination_reason"] in BUDGET_REASONS, (budget, res)
            assert res["termination_reason"] == "global_budget"
            assert opt.over_budget is True
        else:
            assert res["termination_reason"] == ref_opt.last_result["termination_reason"]
            assert requests == ref_requests
        if any(interrupted for _, _, interrupted in fi_log):
            interrupted_in_fi.append(budget)

    # Every budget strictly inside an FI evaluation window interrupts FI.
    assert inside and inside <= set(interrupted_in_fi)


def test_budget_exhausted_during_initial_evaluation_skips_the_engine(pa3):
    opt, deme = _run(pa3, 5, gomea_parameters(pop_size=32), max_evals=10)
    res = opt.last_result
    assert res["termination_reason"] == "global_budget"
    assert res["generations"] == 0
    assert len(deme) == deme.n_evals == 10
    assert opt.over_budget is True
    assert opt.nsteps == 0


def test_inner_cap_limits_the_deme(pa3):
    gp = gomea_parameters(pop_size=16, max_evals_per_deme=40)
    opt, deme = _run(pa3, 5, gp, max_evals=10**6)
    res = opt.last_result
    assert res["ledger"]["requests"] <= 40
    assert len(deme) == deme.n_evals == res["ledger"]["requests"]
    assert res["termination_reason"] == "inner_budget"
    assert opt.over_budget is True


def test_zero_time_limit_evaluates_nothing(pa3):
    opt, deme = _run(pa3, 5, gomea_parameters(pop_size=8), max_time=0)
    assert opt.last_result["termination_reason"] == "time_limit"
    assert len(deme) == deme.n_evals == 0
    assert opt.last_result["best"] is None


def test_deterministic_given_the_global_seed(pa3):
    gp = gomea_parameters(pop_size=16)
    a_opt, a_deme = _run(pa3, 42, gp, max_evals=250)
    b_opt, b_deme = _run(pa3, 42, gp, max_evals=250)
    assert _deme_contents(a_deme) == _deme_contents(b_deme)
    assert _without_timings(a_opt.last_result) == _without_timings(b_opt.last_result)

    c_opt, c_deme = _run(pa3, 43, gp, max_evals=250)
    assert _deme_contents(a_deme) != _deme_contents(c_deme)


def test_real_runs_produce_distinct_real_termination_reasons(pa3):
    """F4(c): real runs end for at least three different reasons, and each
    one is a declared ``TERMINATION_REASONS`` value."""
    assert len(set(TERMINATION_REASONS)) == len(TERMINATION_REASONS)

    reasons = {}
    opt, _ = _run(pa3, 1, gomea_parameters(pop_size=16), max_evals=10)
    reasons["budget"] = opt.last_result["termination_reason"]

    rep, isc, deme = pa3()
    schema = FrozenSchema(rep)
    exemplar_raw = isc(schema.to_instance(schema.default_genotype())).get_score()
    randGen().seed(7)
    opt = gomea_optimization(
        optim_parameters(terminate_if_gte=exemplar_raw), gomea_parameters(pop_size=16)
    )
    opt(deme, isc, 1000, 10**6)
    reasons["target"] = opt.last_result["termination_reason"]

    opt, _ = _run(pa3, 3, gomea_parameters(pop_size=16, max_generations=1), max_evals=10**6)
    reasons["max_gens"] = opt.last_result["termination_reason"]

    assert reasons == {
        "budget": "global_budget",
        "target": "target_reached",
        "max_gens": "max_generations",
    }
    assert set(reasons.values()) <= set(TERMINATION_REASONS)


class _InterruptingScorer(_CountingScorer):
    """Raises ``KeyboardInterrupt`` on call ``k + 1`` (after ``k`` real calls)."""

    def __init__(self, inner, k):
        super().__init__(inner)
        self.k = k

    def __call__(self, inst):
        if self.calls >= self.k:
            raise KeyboardInterrupt
        return super().__call__(inst)


@pytest.mark.parametrize("k", [3, 250])  # during the initial evaluation / generation 2
def test_keyboard_interrupt_finalizes_and_reraises(pa3, tmp_path, k):
    """F4(c): a ``KeyboardInterrupt`` from the scorer runs ``_finish``: the
    deme is finalized, ``last_result`` says ``interrupted``, the ``end``
    event is written, and the exception is re-raised."""
    path = str(tmp_path / "t.jsonl")
    rep, isc, deme = pa3()
    scorer = _InterruptingScorer(isc, k)
    randGen().seed(5)
    opt = gomea_optimization(gp=gomea_parameters(pop_size=16, telemetry_path=path))
    with pytest.raises(KeyboardInterrupt):
        opt(deme, scorer, 10**6, 10**6)
    res = opt.last_result
    assert res is not None
    assert res["termination_reason"] == "interrupted"
    assert deme.n_evals == len(deme) == res["ledger"]["requests"] == scorer.calls == k
    assert opt.total_evals == k
    assert opt.over_budget is False
    if k > res["pop_size_effective"]:
        assert res["generations"] >= 1  # the interrupt came from inside the engine
    else:
        assert res["generations"] == 0
    end = _events(path)[-1]
    assert end["type"] == "end" and end["termination_reason"] == "interrupted"


def test_contin_representation_raises_unsupported():
    rep = representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 1),
        linear_contin=True,
    )

    class _isc:
        def get_rep(self):
            return rep

        def __call__(self, inst):  # pragma: no cover - never reached
            raise AssertionError("scorer must not be called")

    deme = instance_set(rep.fields())
    with pytest.raises(UnsupportedRepresentation):
        gomea_optimization()(deme, _isc(), 100, 100)
    assert len(deme) == 0


def test_target_reached_with_a_trivially_satisfied_threshold(pa3):
    rep, isc, deme = pa3()
    schema = FrozenSchema(rep)
    exemplar_raw = isc(schema.to_instance(schema.default_genotype())).get_score()
    randGen().seed(7)
    opt = gomea_optimization(
        optim_parameters(terminate_if_gte=exemplar_raw), gomea_parameters(pop_size=16)
    )
    opt(deme, isc, 1000, 10**6)
    res = opt.last_result
    assert res["termination_reason"] == "target_reached"
    assert res["generations"] == 0
    assert res["ledger"]["requests"] <= res["pop_size_effective"]
    assert opt.over_budget is False


def test_last_result_shape(pa3):
    opt, deme = _run(pa3, 9, gomea_parameters(pop_size=16), max_evals=200)
    res = opt.last_result
    for key in (
        "termination_reason",
        "generations",
        "ledger",
        "schema_id",
        "dim",
        "n_search_fields",
        "n_constant_fields",
        "pop_size_effective",
        "init",
        "timings",
        "best",
    ):
        assert key in res
    assert res["dim"] == 12
    assert res["n_search_fields"] + res["n_constant_fields"] == res["dim"]
    for key in ("attempts", "duplicates_rejected", "coverage_replacements", "feasible_short"):
        assert key in res["init"]
    assert "radius_histogram" in res["init"]
    for key in ("init_s", "eval_s", "pkey_s", "linkage_s", "variation_s", "total_s"):
        assert res["timings"][key] >= 0.0
    best = res["best"]
    assert set(best) == {"raw", "penalized", "complexity"}
    assert best["raw"] == max(si.second.get_score() for si in deme)


# ---------------------------------------------------------------------------
# Per-call rebuilding and telemetry
# ---------------------------------------------------------------------------


def test_population_and_linkage_are_rebuilt_per_call(pa3, tmp_path):
    path = str(tmp_path / "t.jsonl")
    gp = gomea_parameters(pop_size=16, telemetry_path=path)

    opt = gomea_optimization(gp=gp)
    rep_a, isc_a, deme_a = pa3("and", 3)
    randGen().seed(11)
    opt(deme_a, isc_a, 200, 10**6)
    rep_b, isc_b, deme_b = pa3("or(and($1 $2) $3)", 3)
    randGen().seed(12)
    opt(deme_b, isc_b, 200, 10**6)
    second = opt.last_result

    events = _events(path)
    starts = [e for e in events if e["type"] == "invocation_start"]
    inits = [e for e in events if e["type"] == "init"]
    assert len(starts) == len(inits) == 2
    assert starts[0]["schema_id"] != starts[1]["schema_id"]
    assert starts[0]["dim"] == 12 and starts[1]["dim"] == 7
    assert inits[1]["schema_id"] == starts[1]["schema_id"]
    assert len(inits[1]["diagnostics"]["per_field"]) == 7

    # The second call is exactly what a fresh optimizer does on that rep:
    # no population, linkage or schema state carries over between calls.
    fresh = gomea_optimization(gp=gomea_parameters(pop_size=16))
    rep_c, isc_c, deme_c = pa3("or(and($1 $2) $3)", 3)
    randGen().seed(12)
    fresh(deme_c, isc_c, 200, 10**6)
    assert _deme_contents(deme_b) == _deme_contents(deme_c)
    assert _without_timings(second) == _without_timings(fresh.last_result)

    # Only the optim_stats counters accumulate.
    assert opt.total_evals == len(deme_a) + len(deme_b)


def test_telemetry_events_per_call(pa3, tmp_path):
    path = str(tmp_path / "t.jsonl")
    gp = gomea_parameters(pop_size=16, telemetry_path=path)
    opt, deme = _run(pa3, 21, gp, max_evals=400)
    events = _events(path)
    types = [e["type"] for e in events]
    assert types[0] == "invocation_start"
    assert types[1] == "init"
    assert types[-1] == "end"
    assert types.count("invocation_start") == types.count("init") == types.count("end") == 1
    gens = [e for e in events if e["type"] == "generation"]
    assert len(gens) == opt.last_result["generations"]
    for g in gens:
        assert {"best_raw", "best_penalized", "best_complexity", "ledger"} <= set(g)
    end = events[-1]
    end.pop("type")
    assert end == json.loads(json.dumps(opt.last_result, default=str, sort_keys=True))


def test_population_diversity_hand_computed():
    """F3: the per-generation diversity numbers on a hand-built population."""
    from pymoses.gomea.optimizer import population_diversity

    genotypes = [(0, 1, 2), (0, 1, 2), (1, 1, 0), (0, 1, 1)]
    raws = [-2.0, -1.0, -1.0, -3.0]
    d = population_diversity(genotypes, raws, positions=(0, 2))
    assert d["unique_genotypes"] == 3
    # position 0: {0: 3, 1: 1}; position 2: {2: 2, 0: 1, 1: 1}
    h0 = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    h2 = -(0.5 * math.log2(0.5) + 2 * 0.25 * math.log2(0.25))
    assert d["mean_entropy_bits"] == pytest.approx((h0 + h2) / 2)
    assert d["n_varying_fields"] == 2
    assert d["best_raw_count"] == 2
    # No search positions: nothing varies and the mean entropy is 0.
    empty = population_diversity(genotypes, raws, positions=())
    assert empty["mean_entropy_bits"] == 0.0 and empty["n_varying_fields"] == 0


def test_generation_events_carry_diversity_telemetry(pa3, tmp_path):
    """F3: every "generation" event reports the population's diversity
    after that generation, over the model's search positions, and the
    numbers are consistent on a real representation."""
    path = str(tmp_path / "t.jsonl")
    gp = gomea_parameters(pop_size=16, telemetry_path=path)
    opt, _ = _run(pa3, 21, gp, max_evals=400)
    res = opt.last_result
    gens = [e for e in _events(path) if e["type"] == "generation"]
    assert gens, "the run must reach at least one generation"
    n = res["pop_size_effective"]
    n_search = res["n_search_fields"]
    for g in gens:
        assert 1 <= g["unique_genotypes"] <= n
        assert 0 <= g["n_varying_fields"] <= n_search
        assert 0.0 <= g["mean_entropy_bits"] <= math.log2(3) + 1e-12  # 3-valued knobs
        assert 1 <= g["best_raw_count"] <= n
        assert (g["n_varying_fields"] == 0) == (g["mean_entropy_bits"] == 0.0)
        if g["unique_genotypes"] == 1:
            assert g["n_varying_fields"] == 0
    # The initial population is diverse over its search positions.
    assert gens[0]["n_varying_fields"] > 0
    assert gens[0]["unique_genotypes"] > 1


def test_telemetry_sink_without_path_is_a_noop(tmp_path):
    sink = TelemetrySink(None)
    sink.emit("anything", x=1)
    assert list(tmp_path.iterdir()) == []


def test_telemetry_sink_appends_json_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    sink = TelemetrySink(str(path))
    sink.emit("a", x=1)
    sink.emit("b", y={2: 3})
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"type": "a", "x": 1},
        {"type": "b", "y": {"2": 3}},
    ]
