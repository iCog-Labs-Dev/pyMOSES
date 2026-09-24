"""Tests for pymoses.gomea.oracle.EvaluationOracle and EvaluationLedger.

Builds real Boolean representations (``make_boolean_rep``, conftest.py) and
wraps them in ``FrozenSchema`` (Task 2), then drives ``EvaluationOracle``
against either a real ``complexity_based_scorer``/``behave_cscore`` chain
(matching how hill climbing itself scores a genotype -- see
``docs/gomea/P1.0-AUDIT.md`` Sec 3/5) or a small stand-in ``iscorer`` for
tests that only need control over the returned scores.
"""

import copy
import random

import pytest

from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table import ITable, OTable, Table, type_node
from pymoses.gomea.oracle import BudgetExhausted, EvaluationLedger, EvaluationOracle
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.moses.types import composite_score, very_worst_score, worst_composite_score
from pymoses.moses.representation.instance_scorer import complexity_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import ctruth_table_bscore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _boolean_ctable_arity2():
    """A minimal hand-built two-row boolean CTable, arity 2 -- reuses the
    pattern ``test_scoring_base.py``'s own ``_boolean_ctable`` uses to avoid
    a filesystem fixture: ``ctruth_table_bscore`` only ever needs the rows
    actually present in the table, not full domain coverage.
    """
    t, f = builtin.logical_true, builtin.logical_false
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type])
    it.push_back((t, t))
    it.push_back((f, f))
    ot = OTable([t, f])
    return Table(ot, it).compressed()


def _make_deme(schema: FrozenSchema) -> instance_set:
    return instance_set(schema.fields)


def _real_iscorer(rep, cache_size: int = 0) -> complexity_based_scorer:
    bsc = ctruth_table_bscore(_boolean_ctable_arity2())
    bsc.set_complexity_coef(3.5)
    cscorer = behave_cscore(bsc, cache_size=cache_size)
    return complexity_based_scorer(cscorer, rep, True)


class _sequence_iscorer:
    """Not a ported C++ symbol -- returns pre-set ``composite_score``
    objects in call order, one per call, so oracle bookkeeping (budget,
    best-so-far, invalid counting) can be tested independently of the real
    Boolean scoring pipeline. Records every ``instance`` it was called
    with, for aliasing checks.
    """

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = []

    def __call__(self, inst):
        self.calls.append(inst)
        return self._scores.pop(0)


class _stub_cscore:
    """Not a ported C++ symbol -- mirrors
    ``test_instance_scorer.py``'s own ``_stub_cscore``: a minimal
    ``behave_cscore`` stand-in with no cache counters at all.
    """

    def __init__(self, result: composite_score):
        self.result = result

    def get_cscore(self, tr):
        return self.result


class _raising_representation:
    """Not a ported C++ symbol -- mirrors
    ``test_instance_scorer.py``'s own ``_raising_representation``: forces
    ``complexity_based_scorer.__call__``'s except branch (-> a copy of
    ``worst_composite_score``) without a full representation/knob-building
    setup.
    """

    def get_candidate(self, inst, reduce):
        raise ValueError("synthetic evaluation failure")


class _fake_clock:
    """Not a ported C++ symbol -- a settable stand-in for
    ``time.monotonic``, so the time-limit budget check can be tested
    deterministically.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_ledger_starts_at_all_zeros():
    ledger = EvaluationLedger()
    assert ledger.as_dict() == {
        "requests": 0,
        "invalid": 0,
        "noop_skips": 0,
        "phenotype_skips": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }


# ---------------------------------------------------------------------------
# Contract 3: oracle score == iscorer(inst), calling HC's scorer directly
# ---------------------------------------------------------------------------


def test_oracle_score_matches_direct_iscorer_call_for_50_random_genotypes(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    rng = random.Random(20260924)
    for _ in range(50):
        g = tuple(rng.randrange(domain) for domain in schema.domains)
        expected = iscorer(schema.to_instance(g))
        got = oracle.evaluate(g)
        assert got == expected

    assert oracle.ledger.requests == 50


# ---------------------------------------------------------------------------
# Contract 2: no aliasing between evaluations
# ---------------------------------------------------------------------------


def test_two_evaluations_of_the_same_genotype_never_alias_scores(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    g = schema.default_genotype()
    score1 = oracle.evaluate(g)
    score2 = oracle.evaluate(g)

    assert score1 == score2
    assert score1 is not score2
    assert deme[0].second is not score1
    assert deme[1].second is not score2
    assert deme[0].second is not deme[1].second

    before = copy.copy(deme[0].second)
    score1.set_score(-999.0)
    assert deme[0].second == before
    assert deme[0].second.get_score() != -999.0
    assert deme[1].second.get_score() != -999.0


def test_two_evaluations_of_the_same_genotype_never_alias_instances(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    g = schema.default_genotype()
    oracle.evaluate(g)
    oracle.evaluate(g)

    assert deme[0].first is not deme[1].first
    schema.fields.set_raw(deme[0].first, 0, 1)
    assert schema.to_genotype(deme[1].first) == g


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_max_evals_zero_raises_global_budget_before_calling_the_scorer(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=0, max_time=3600.0)

    assert oracle.check() == "global_budget"
    with pytest.raises(BudgetExhausted) as exc_info:
        oracle.evaluate(schema.default_genotype())

    assert exc_info.value.reason == "global_budget"
    assert iscorer.calls == []
    assert oracle.ledger.requests == 0
    assert len(deme) == 0


def test_inner_cap_stops_at_exactly_three_requests_even_with_a_larger_global_budget(
    make_boolean_rep,
):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0) for _ in range(10)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(
        schema, iscorer, deme, max_evals=100, max_time=3600.0, inner_cap=3
    )

    g = schema.default_genotype()
    for _ in range(3):
        oracle.evaluate(g)

    assert oracle.check() == "inner_budget"
    with pytest.raises(BudgetExhausted) as exc_info:
        oracle.evaluate(g)
    assert exc_info.value.reason == "inner_budget"
    assert oracle.ledger.requests == 3
    assert len(iscorer.calls) == 3


def test_global_budget_reason_takes_priority_over_inner_budget(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([])
    deme = _make_deme(schema)
    # requests (0) already >= max_evals (0) AND >= inner_cap (0): global
    # must win per the documented check() order.
    oracle = EvaluationOracle(
        schema, iscorer, deme, max_evals=0, max_time=3600.0, inner_cap=0
    )
    assert oracle.check() == "global_budget"


def test_time_limit_uses_injected_fake_clock(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0)])
    deme = _make_deme(schema)
    clock = _fake_clock(100.0)
    oracle = EvaluationOracle(
        schema, iscorer, deme, max_evals=1000, max_time=5.0, clock=clock
    )

    assert oracle.check() is None
    clock.now = 106.0
    assert oracle.check() == "time_limit"
    with pytest.raises(BudgetExhausted) as exc_info:
        oracle.evaluate(schema.default_genotype())
    assert exc_info.value.reason == "time_limit"
    assert iscorer.calls == []


def test_check_returns_none_when_nothing_is_exhausted(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=10, max_time=3600.0)
    assert oracle.check() is None


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------


def test_finalize_sets_deme_n_evals_to_requests_and_len(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    rng = random.Random(1)
    for _ in range(7):
        g = tuple(rng.randrange(domain) for domain in schema.domains)
        oracle.evaluate(g)

    oracle.finalize()

    assert oracle.ledger.requests == 7
    assert len(deme) == 7
    assert deme.n_evals == 7


def test_finalize_sets_deme_n_best_evals_to_evals_at_best(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    scores = [
        composite_score(-5.0, 0, 0.0, 0.0),
        composite_score(-3.0, 0, 0.0, 0.0),
        composite_score(-3.0, 0, 0.0, 0.0),  # tie: earlier (#2) stays best
        composite_score(-1.0, 0, 0.0, 0.0),  # strictly better -> new best
        composite_score(-4.0, 0, 0.0, 0.0),
    ]
    iscorer = _sequence_iscorer(scores)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    g = schema.default_genotype()
    for _ in range(5):
        oracle.evaluate(g)

    assert oracle.evals_at_best() == 4
    assert oracle.best_so_far().get_score() == -1.0

    oracle.finalize()
    assert deme.n_best_evals == 4


# ---------------------------------------------------------------------------
# best_so_far() / evals_at_best()
# ---------------------------------------------------------------------------


def test_best_so_far_keeps_the_earlier_of_two_tied_scores(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    tied = composite_score(-2.0, 1, 0.0, 0.0)
    iscorer = _sequence_iscorer([copy.copy(tied), copy.copy(tied)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    g = schema.default_genotype()
    oracle.evaluate(g)
    oracle.evaluate(g)

    assert oracle.evals_at_best() == 1


def test_best_so_far_is_a_defensive_copy(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)
    oracle.evaluate(schema.default_genotype())

    best = oracle.best_so_far()
    best.set_score(-999.0)
    assert oracle.best_so_far().get_score() != -999.0


# ---------------------------------------------------------------------------
# Invalid counting
# ---------------------------------------------------------------------------


def test_invalid_counts_scorer_exceptions_turned_into_worst_score(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = complexity_based_scorer(
        _stub_cscore(composite_score(-1.0, 0, 0.0, 0.0)), _raising_representation(), True
    )
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    got = oracle.evaluate(schema.default_genotype())

    assert got == worst_composite_score
    assert got.get_penalized_score() == very_worst_score
    assert oracle.ledger.invalid == 1
    assert oracle.ledger.requests == 1


def test_invalid_does_not_count_valid_scores(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    oracle.evaluate(schema.default_genotype())

    assert oracle.ledger.invalid == 0


# ---------------------------------------------------------------------------
# record_noop_skip / record_phenotype_skip
# ---------------------------------------------------------------------------


def test_record_noop_skip_only_increments_the_ledger(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    oracle.record_noop_skip()
    oracle.record_noop_skip()

    assert oracle.ledger.noop_skips == 2
    assert oracle.ledger.requests == 0
    assert len(deme) == 0
    assert iscorer.calls == []


def test_record_phenotype_skip_only_increments_the_ledger(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    oracle.record_phenotype_skip()

    assert oracle.ledger.phenotype_skips == 1
    assert oracle.ledger.requests == 0
    assert len(deme) == 0
    assert iscorer.calls == []


# ---------------------------------------------------------------------------
# Cache hit/miss delta
# ---------------------------------------------------------------------------


def test_cache_hits_and_misses_are_zero_when_the_iscorer_has_no_cscorer(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    oracle.evaluate(schema.default_genotype())

    assert oracle.ledger.cache_hits == 0
    assert oracle.ledger.cache_misses == 0


def test_cache_hits_and_misses_track_the_real_cscorer_counters(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _real_iscorer(rep, cache_size=3000)
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=1000, max_time=3600.0)

    g = schema.default_genotype()
    oracle.evaluate(g)  # first time: a cache miss
    oracle.evaluate(g)  # same genotype -> same reduced tree -> a cache hit

    assert oracle.ledger.cache_misses == 1
    assert oracle.ledger.cache_hits == 1
    assert oracle.ledger.requests == 2


# ---------------------------------------------------------------------------
# check() called before EVERY evaluation, even mid-run
# ---------------------------------------------------------------------------


def test_evaluate_checks_the_budget_every_call_not_just_once(make_boolean_rep):
    rep = make_boolean_rep("and($1 $2)", 2)
    schema = FrozenSchema(rep)
    iscorer = _sequence_iscorer([composite_score(-1.0, 0, 0.0, 0.0) for _ in range(2)])
    deme = _make_deme(schema)
    oracle = EvaluationOracle(schema, iscorer, deme, max_evals=2, max_time=3600.0)

    g = schema.default_genotype()
    oracle.evaluate(g)
    oracle.evaluate(g)

    with pytest.raises(BudgetExhausted) as exc_info:
        oracle.evaluate(g)
    assert exc_info.value.reason == "global_budget"
    assert oracle.ledger.requests == 2
