"""The deme-local complexity penalty (``--gomea-accept-complexity-ratio``)
and the accepted-only deme store (``--gomea-deme-store accepted``).

Both are off by default; off, parameters, telemetry and runs are unchanged.
"""

from __future__ import annotations

import argparse
import itertools
import math

import pytest

import pymoses.gomea.optimizer as optimizer_mod
from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.gom import Individual
from pymoses.gomea.optimizer import _OracleEvaluator, _WorseKey, accept_worse
from pymoses.gomea.oracle import EvaluationOracle
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.main.problem_params import _gomea_params_from_args
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import MT19937RandGen, randGen
from tests.gomea.test_integration import _cli_parser, _events, _run
from tests.gomea.test_optimizer import _deme_contents, _iscorer

NAN = float("nan")


def cs(raw: float, cx: int, penalty: float = 0.0) -> composite_score:
    return composite_score(raw, cx, penalty)


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------


def test_penalty_orders_by_raw_minus_complexity_over_ratio():
    # ratio 0.125: one complexity unit costs 8 raw units.
    small, big = cs(-10, 1), cs(-4, 2)  # -18 vs -20
    assert accept_worse(big, small, 0.125)
    assert not accept_worse(small, big, 0.125)
    # MOSES's own comparator (no penalty stored here) prefers the big one.
    assert small < big
    # A gain of more than 8 raw units per unit does pay.
    assert accept_worse(small, cs(-1, 2), 0.125)


def test_moses_penalty_fields_are_ignored():
    # The stored complexity/uniformity penalties play no part.
    a, b = cs(-10, 1, penalty=100.0), cs(-10, 1, penalty=0.0)
    assert not accept_worse(a, b, 2.0) and not accept_worse(b, a, 2.0)
    assert a < b  # MOSES sees a difference


def test_equal_penalized_value_breaks_ties_on_complexity():
    a, b = cs(-8, 1), cs(0, 2)  # both -16 at ratio 0.125
    assert accept_worse(b, a, 0.125)
    assert not accept_worse(a, b, 0.125)
    same = cs(-8, 1)
    assert not accept_worse(a, same, 0.125) and not accept_worse(same, a, 0.125)


def test_nan_handling_mirrors_composite_score():
    x, n = cs(-3, 2), cs(NAN, 2)
    assert accept_worse(n, x, 1.0)  # NaN is worse than anything ...
    assert not accept_worse(x, n, 1.0)
    assert not accept_worse(n, cs(NAN, 0), 1.0)  # ... except NaN
    assert accept_worse(n, cs(-math.inf, 5), 1.0)
    # The same cases under MOSES's comparator, for reference.
    assert n < x and not x < n and not n < cs(NAN, 0)


def test_evaluator_without_ratio_is_the_moses_comparator():
    scores = [cs(r, c, p) for r, c, p in itertools.product((-5, -4, NAN), (1, 3), (0.0, 0.5))]
    ev = _OracleEvaluator(None, None, False)
    for a, b in itertools.product(scores, repeat=2):
        assert ev.worse(a, b) == (a < b)
    ev_r = _OracleEvaluator(None, None, False, 0.5)
    for a, b in itertools.product(scores, repeat=2):
        assert ev_r.worse(a, b) == accept_worse(a, b, 0.5)


def test_worse_key_sort_equals_the_composite_sort():
    scores = [cs(r, c) for r, c in itertools.product((-5, -4, -4, NAN), (1, 3, 3))]
    ev = _OracleEvaluator(None, None, False)
    by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    by_key = sorted(range(len(scores)), key=lambda i: _WorseKey(scores[i], ev.worse), reverse=True)
    assert by_key == by_score


def test_engine_best_and_elite_follow_the_evaluator():
    ev = _OracleEvaluator(None, None, False, 0.125)
    eng = GomeaEngine([2], EngineParams(), ev, MT19937RandGen(1), MT19937RandGen(2))
    small = Individual((0,), cs(-10, 1), None)
    big = Individual((1,), cs(-4, 2), None)
    assert eng.best_of([big, small]) is small
    eng.offer_elite(small)
    eng.offer_elite(big)
    assert eng.elite is small


# ---------------------------------------------------------------------------
# Parameters and CLI
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_left_out_of_telemetry():
    gp = gomea_parameters()
    assert gp.accept_complexity_ratio == 0.0 and gp.deme_store == "all"
    d = gp.as_dict()
    assert "accept_complexity_ratio" not in d and "deme_store" not in d
    on = gomea_parameters(accept_complexity_ratio=0.125, deme_store="accepted").as_dict()
    assert on["accept_complexity_ratio"] == 0.125 and on["deme_store"] == "accepted"


@pytest.mark.parametrize(
    "kw",
    [
        {"accept_complexity_ratio": -1.0},
        {"accept_complexity_ratio": NAN},
        {"accept_complexity_ratio": math.inf},
        {"deme_store": "rejected"},
    ],
)
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        gomea_parameters(**kw)


def test_cli_options_reach_parameters():
    gp = _gomea_params_from_args(
        _cli_parser(True).parse_args(
            ["--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
        )
    )
    assert (gp.accept_complexity_ratio, gp.deme_store) == (0.125, "accepted")
    gp0 = _gomea_params_from_args(_cli_parser(True).parse_args([]))
    assert (gp0.accept_complexity_ratio, gp0.deme_store) == (0.0, "all")
    gpn = _gomea_params_from_args(argparse.Namespace())
    assert (gpn.accept_complexity_ratio, gpn.deme_store) == (0.0, "all")


@pytest.mark.parametrize(
    "argv",
    [
        ["--gomea-accept-complexity-ratio", "-0.5"],
        ["--gomea-accept-complexity-ratio", "nan"],
        ["--gomea-deme-store", "some"],
    ],
)
def test_invalid_cli_values_rejected(argv):
    with pytest.raises(ValueError):
        _gomea_params_from_args(_cli_parser(True).parse_args(argv))


# ---------------------------------------------------------------------------
# The oracle: stall rule and accepted-only store
# ---------------------------------------------------------------------------


class _Scores:
    """Returns pre-set scores in call order."""

    def __init__(self, scores):
        self.scores = list(scores)

    def __call__(self, inst):
        return self.scores.pop(0)


@pytest.fixture
def schema(make_boolean_rep):
    return FrozenSchema(make_boolean_rep("and", 2))


def _genotypes(schema, n):
    base = list(schema.default_genotype())
    out = []
    for i in range(n):
        g = list(base)
        g[i % schema.dim] = (g[i % schema.dim] + 1 + i // schema.dim) % schema.domains[
            i % schema.dim
        ]
        out.append(tuple(g))
    return out


def test_stall_rule_uses_the_penalized_best(schema):
    # g1 is better for MOSES but worse under ratio 0.125.
    seq = [cs(-10, 1), cs(-4, 3), cs(-20, 1)]
    gs = _genotypes(schema, 3)
    for stall_worse, expected in (
        (None, None),
        (lambda a, b: accept_worse(a, b, 0.125), "stall_evals"),
    ):
        deme = instance_set(schema.fields)
        o = EvaluationOracle(
            schema, _Scores(seq), deme, 100, 10**6, stall_evals=2, stall_worse=stall_worse
        )
        for g in gs:
            o.evaluate(g)
        assert o.check() == expected
        o.finalize()
        assert deme.n_best_evals == 2  # MOSES's best either way
        assert o.best_so_far().get_score() == -4


def test_store_accepted_keeps_only_marked_entries_and_counts_everything(schema):
    gs = _genotypes(schema, 4)
    deme = instance_set(schema.fields)
    o = EvaluationOracle(
        schema, _Scores([cs(-i, 1) for i in range(4)]), deme, 100, 10**6, store_accepted=True
    )
    for g in gs:
        o.evaluate(g)
    o.keep(gs[0])
    o.keep(gs[2])
    o.keep(tuple(schema.default_genotype()))  # never evaluated: ignored
    assert len(deme) == 4  # nothing dropped before finalize
    o.finalize()
    assert [si.second.get_score() for si in deme] == [0, -2]
    assert deme.n_evals == 4 and o.ledger.requests == 4


def test_store_accepted_keeps_preloaded_entries(schema):
    gs = _genotypes(schema, 3)
    deme = instance_set(schema.fields)
    pre = EvaluationOracle(schema, _Scores([cs(-1, 1)]), deme, 100, 10**6)
    pre.evaluate(gs[0])
    pre.finalize()
    o = EvaluationOracle(
        schema, _Scores([cs(-5, 1), cs(-6, 1)]), deme, 100, 10**6, store_accepted=True
    )
    o.preload([(gs[0], cs(-1, 1))], deme.n_evals)
    o.evaluate(gs[1])
    o.evaluate(gs[2])
    o.keep(gs[2])
    o.finalize()
    assert [si.second.get_score() for si in deme] == [-1, -6]
    assert deme.n_evals == 3


def test_store_all_keep_is_a_noop(schema):
    gs = _genotypes(schema, 2)
    deme = instance_set(schema.fields)
    o = EvaluationOracle(schema, _Scores([cs(-1, 1), cs(-2, 1)]), deme, 100, 10**6)
    for g in gs:
        o.evaluate(g)
    o.keep(gs[0])
    o.finalize()
    assert len(deme) == 2 == deme.n_evals


# ---------------------------------------------------------------------------
# Whole demes
# ---------------------------------------------------------------------------


@pytest.fixture
def pa3(make_boolean_rep):
    def _make():
        rep = make_boolean_rep("and", 3)
        return rep, _iscorer(rep, 3), instance_set(rep.fields())

    return _make


def _deme_run(pa3, gp, seed=7, max_evals=300, log=None, monkeypatch=None):
    if log is not None:
        gp.telemetry_path = str(log)
    captured = {}
    if monkeypatch is not None:

        class _Capturing(GomeaEngine):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                captured["engine"] = self

        monkeypatch.setattr(optimizer_mod, "GomeaEngine", _Capturing)
    _, isc, deme = pa3()
    randGen().seed(seed)
    opt = gomea_optimization(None, gp)
    opt(deme, isc, max_evals, 10**6)
    return opt, deme, captured.get("engine")


def test_explicit_defaults_change_nothing(pa3):
    a_opt, a = _deme_run(pa3, gomea_parameters())[:2]
    b_opt, b = _deme_run(pa3, gomea_parameters(accept_complexity_ratio=0.0, deme_store="all"))[:2]
    assert _deme_contents(a) == _deme_contents(b)
    assert (a.n_evals, a.n_best_evals) == (b.n_evals, b.n_best_evals)
    strip = lambda r: {k: v for k, v in r.items() if k != "timings"}  # noqa: E731
    assert strip(a_opt.last_result) == strip(b_opt.last_result)


@pytest.mark.parametrize("fi", [False, True])
@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_accepted_store_keeps_initial_and_accepted_trials(pa3, tmp_path, monkeypatch, fi, ratio):
    kw = {"forced_improvement": fi, "accept_complexity_ratio": ratio, "pop_size": 8}
    _, full, _ = _deme_run(pa3, gomea_parameters(**kw))
    log = tmp_path / "g.jsonl"
    opt, kept, eng = _deme_run(
        pa3, gomea_parameters(deme_store="accepted", **kw), log=log, monkeypatch=monkeypatch
    )
    # Storing does not change the search: same evaluations, same counts.
    assert kept.n_evals == full.n_evals == len(full)
    assert kept.n_best_evals == full.n_best_evals
    assert opt.last_result["ledger"]["requests"] == full.n_evals
    # The kept entries are a subsequence of the full deme, in order.
    full_c, kept_c = _deme_contents(full), _deme_contents(kept)
    it = iter(full_c)
    assert all(any(x == y for y in it) for x in kept_c)
    # Exactly the initial population plus every accepted evaluated trial.
    gens = [e for e in _events(log) if e["type"] == "generation"]
    accepted = sum(
        e["stats"]["accepted_improving"]
        + e["stats"]["accepted_neutral"]
        + e["stats"]["fi_improved"]
        for e in gens
    )
    assert sum(e["stats"]["phenotype_skips"] for e in gens) == 0
    assert len(kept) == opt.last_result["pop_size_effective"] + accepted
    assert len(kept) < len(full)
    # The final population and the elite are all in the deme.
    schema = FrozenSchema(pa3()[0])
    stored = {c[0] for c in kept_c}
    for ind in [*eng.last_result.population, eng.elite]:
        assert tuple(schema.to_instance(ind.genotype)) in stored


def test_accepted_store_with_the_genotype_memo(pa3, monkeypatch):
    kw = {"genotype_memo": True, "accept_complexity_ratio": 0.25, "pop_size": 8}
    _, full, _ = _deme_run(pa3, gomea_parameters(**kw))
    opt, kept, eng = _deme_run(
        pa3, gomea_parameters(deme_store="accepted", **kw), monkeypatch=monkeypatch
    )
    assert kept.n_evals == full.n_evals == len(full)
    assert 0 < len(kept) < len(full)
    schema = FrozenSchema(pa3()[0])
    stored = {tuple(si.first) for si in kept}
    for ind in eng.last_result.population:
        assert tuple(schema.to_instance(ind.genotype)) in stored


def test_penalty_changes_the_deme_search(pa3):
    _, a, _ = _deme_run(pa3, gomea_parameters(pop_size=8))
    _, b, _ = _deme_run(pa3, gomea_parameters(pop_size=8, accept_complexity_ratio=0.05))
    assert a.n_evals == b.n_evals
    assert _deme_contents(a) != _deme_contents(b)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_both_options_run_end_to_end_and_are_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep moses.log out of the repository
    log = tmp_path / "g.jsonl"
    argv = ["-H", "pa", "-k", "3", "-a", "gomea", "-m", "800", "-r", "1",
            "--gomea-max-evals-per-deme", "200", "--gomea-log", str(log)]  # fmt: skip
    rc, stdout = _run(
        [*argv, "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
    )
    assert rc == 0 and stdout
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert starts
    for e in starts:
        assert e["params"]["accept_complexity_ratio"] == 0.125
        assert e["params"]["deme_store"] == "accepted"
    ends = [e for e in events if e["type"] == "end"]
    assert sum(e["ledger"]["requests"] for e in ends) <= 800

    log0 = tmp_path / "g0.jsonl"
    rc0, _ = _run([*argv[:-1], str(log0)])
    assert rc0 == 0
    for e in _events(log0):
        if e["type"] == "invocation_start":
            assert "accept_complexity_ratio" not in e["params"]
            assert "deme_store" not in e["params"]
