"""Tests for pymoses.gomea.hybrid (``-a hc-gomea``, Phase 1e).

The hybrid runs MOSES's own hill climbing on a deme, then, when hill
climbing stopped at a local optimum with budget left, continues the SAME
deme with GOMEA seeded from the instances hill climbing already evaluated.
Driven exactly like ``deme_expander.optimize_demes`` drives any optimizer,
against a real Boolean representation and a real scorer (even parity).
"""

import json

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea import gomea_parameters
from pymoses.gomea.hybrid import HC_STOP_REASONS, hc_gomea_optimization
from pymoses.gomea.schema import FrozenSchema, UnsupportedRepresentation
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import randGen
from tests.gomea.test_optimizer import _deme_contents, _iscorer


class _Counting:
    """Counts scorer calls around a real ``complexity_based_scorer``."""

    def __init__(self, isc):
        self.isc = isc
        self.calls = 0

    def get_rep(self):
        return self.isc.get_rep()

    def __call__(self, inst):
        self.calls += 1
        return self.isc(inst)


@pytest.fixture
def problem(make_boolean_rep):
    def _make(exemplar: str = "and", arity: int = 3):
        rep = make_boolean_rep(exemplar, arity)
        return rep, _Counting(_iscorer(rep, arity)), instance_set(rep.fields())

    return _make


def _hc(problem, seed, op=None, max_evals=10**4, **kw):
    rep, isc, deme = problem(**kw)
    randGen().seed(seed)
    opt = hill_climbing(op if op is not None else optim_parameters(), hc_parameters())
    opt(deme, isc, max_evals, 10**6)
    return opt, deme, randGen().randint()


def _hybrid(problem, seed, op=None, gp=None, max_evals=10**4, **kw):
    rep, isc, deme = problem(**kw)
    randGen().seed(seed)
    opt = hc_gomea_optimization(op, hc_parameters(), gp)
    opt(deme, isc, max_evals, 10**6)
    return opt, deme, isc, randGen().randint()


def _raw(isc, rep):
    schema = FrozenSchema(rep)
    return isc(schema.to_instance(schema.default_genotype())).get_score()


def test_hc_reaching_the_target_is_plain_hill_climbing(problem):
    """HC reaches the target: the deme, its counts and the RNG state after
    the call are exactly ``-a hc``'s, and GOMEA never runs."""
    rep, isc, _ = problem()
    op = optim_parameters(terminate_if_gte=_raw(isc, rep))
    hc, hc_deme, hc_next = _hc(problem, 5, op)
    hy, hy_deme, _, hy_next = _hybrid(problem, 5, op)
    assert _deme_contents(hy_deme) == _deme_contents(hc_deme)
    assert (hy_deme.n_evals, hy_deme.n_best_evals) == (hc_deme.n_evals, hc_deme.n_best_evals)
    assert hy_next == hc_next
    res = hy.last_result
    assert (res["hc_stop"], res["gomea_ran"], res["gomea_evals"]) == ("target", False, 0)
    assert res["gomea_termination_reason"] is None


@pytest.mark.parametrize("max_evals", [1, 10, 40])
def test_hc_spending_the_budget_is_plain_hill_climbing(problem, max_evals):
    hc, hc_deme, hc_next = _hc(problem, 2, max_evals=max_evals)
    hy, hy_deme, _, hy_next = _hybrid(problem, 2, max_evals=max_evals)
    assert _deme_contents(hy_deme) == _deme_contents(hc_deme)
    assert hy_deme.n_evals == hc_deme.n_evals and hy_next == hc_next
    assert hy.last_result["hc_stop"] == "budget" and not hy.last_result["gomea_ran"]
    assert hy.over_budget is True


def test_gomea_phase_continues_the_deme_without_recharging(problem, tmp_path):
    """HC stops at a local optimum; GOMEA then runs on the same deme. HC's
    part of the deme is exactly ``-a hc``'s; the seeded population is taken
    from it uncharged (every scorer call after HC is a charged GOMEA
    request, appended once); ``n_evals == len(deme)``."""
    log = tmp_path / "g.jsonl"
    gp = gomea_parameters(fos="uni", max_evals_per_deme=300, telemetry_path=str(log))
    hc, hc_deme, _ = _hc(problem, 4)
    hy, deme, isc, _ = _hybrid(problem, 4, gp=gp)
    res = hy.last_result
    assert res["hc_stop"] == "local_optimum" and res["gomea_ran"] is True
    n_hc = res["hc_evals"]
    assert n_hc == hc_deme.n_evals
    assert _deme_contents(deme)[:n_hc] == _deme_contents(hc_deme)
    requests = res["gomea"]["ledger"]["requests"]
    assert res["gomea_evals"] == requests > 0
    assert isc.calls == n_hc + requests
    assert deme.n_evals == len(deme) == n_hc + requests
    assert requests <= 300

    events = [json.loads(line) for line in log.read_text().splitlines()]
    types = [e["type"] for e in events]
    assert types[0] == "invocation_start" and types[-1] == "hybrid"
    init = next(e for e in events if e["type"] == "init")
    assert init["init_mode"] == "deme"
    distinct_hc = len({tuple(si.first) for si in hc_deme})
    assert init["from_deme"] == min(16, distinct_hc)
    start = events[0]
    assert start["prior_evals"] == n_hc and start["max_evals"] == 10**4 - n_hc
    hybrid = events[-1]
    assert hybrid["hc_evals"] == n_hc and hybrid["hc_stop"] == "local_optimum"
    assert hybrid["gomea_ran"] is True and hybrid["gomea_evals"] == requests
    assert hybrid["gomea_termination_reason"] == res["gomea"]["termination_reason"]


def test_seeded_population_is_the_best_distinct_hc_instances(problem, monkeypatch):
    import pymoses.gomea.engine as engine_mod

    captured = {}
    real_run = engine_mod.GomeaEngine.run

    def spy(self, population):
        captured["pop"] = [(ind.genotype, ind.score) for ind in population]
        return real_run(self, population)

    monkeypatch.setattr(engine_mod.GomeaEngine, "run", spy)
    hy, deme, isc, _ = _hybrid(problem, 4, gp=gomea_parameters(fos="uni", max_evals_per_deme=50))
    n_hc = hy.last_result["hc_evals"]
    schema = FrozenSchema(isc.get_rep())
    hc_part = [(schema.to_genotype(si.first), si.second) for si in list(deme)[:n_hc]]
    ranked = sorted(range(n_hc), key=lambda i: hc_part[i][1], reverse=True)
    expected, seen = [], set()
    for i in ranked:
        if hc_part[i][0] not in seen:
            seen.add(hc_part[i][0])
            expected.append(hc_part[i])
    pop = captured["pop"]
    k = min(16, len(expected))
    assert [g for g, _ in pop[:k]] == [g for g, _ in expected[:k]]
    assert [s for _, s in pop[:k]] == [s for _, s in expected[:k]]


@pytest.mark.parametrize("max_evals", [30, 120, 400, 1500])
def test_budget_is_never_exceeded(problem, max_evals):
    hy, deme, isc, _ = _hybrid(problem, 6, gp=gomea_parameters(fos="uni"), max_evals=max_evals)
    assert deme.n_evals == len(deme) == isc.calls <= max_evals
    assert hy.last_result["hc_stop"] in HC_STOP_REASONS


def test_stall_rule_applies_in_the_gomea_phase(problem):
    gp = gomea_parameters(fos="uni", stall_evals=25)
    hy, deme, _, _ = _hybrid(problem, 4, gp=gp)
    res = hy.last_result
    assert res["gomea_ran"] and res["gomea_termination_reason"] == "stall_evals"
    assert hy.over_budget is False
    assert deme.n_evals == len(deme)


def test_contin_representation_is_refused_before_hill_climbing():
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
        hc_gomea_optimization()(deme, _isc(), 100, 100)
    assert len(deme) == 0
