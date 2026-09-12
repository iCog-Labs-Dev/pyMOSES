"""Guard tests for pymoses.feature_selection.algo.deme_optimize (the "hc"
feature-selection algorithm).

## No C++ oracle is possible here -- and that is itself a documented finding

Classic's OWN "--algo hc" standalone path segfaults UNCONDITIONALLY, on
every input, confirmed two independent ways (see
`tools/cpp_probe/fs_standalone_probe.cc`'s own header comment for the
full trace):

1. The REAL installed binary --
       feature-selection -i trainingBoolean.dat -u y --algo hc
           --target-size=1 -m 200 -o /dev/null -l ERROR -r 1
   dumps core every time (`-l ERROR` included, so this is not a
   debug-logging-level issue).
2. A gdb backtrace of that exact crash: `hill_climbing::operator()`
   (hill-climbing.cc:69) calls `iscorer.get_rep()->get_candidate(...)`
   UNCONDITIONALLY as an argument to its own `logger().debug() << ...`
   line (no `is_debug_enabled()` guard), and `iscorer.get_rep()` is
   `nullptr` for every scorer this algorithm ever builds
   (`deme_based_scorer`/`iscorer_cache`, neither overrides
   `iscorer_base::get_rep()`'s `nullptr` default) -- a 100% reproducible
   null-pointer dereference through
   `main -> feature_selection() -> select_features() ->
   select_feature_sets() -> moses_select_feature_sets() ->
   create_deme_select_feature_sets() ->
   optimize_deme_select_feature_sets() -> hill_climbing::operator()() ->
   representation::get_candidate() [this=0x0]`.

This port's OWN `hill_climbing.py` does not crash here only because an
EARLIER, already-reviewed slice (S9's optimizer port) already drops every
`logger()` call project-wide as a matter of established convention -- not
a fix applied for this task. The consequence: this port's "hc" algorithm
genuinely runs to completion where classic itself cannot, a real
behavioural DIVERGENCE (in the port's favour) that has no golden value to
compare against. These tests instead prove internal correctness by
construction and by mutation.
"""

import pytest

from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.algo.deme_optimize import (
    create_deme_select_feature_sets,
    initial_instance,
    iscorer_cache,
    moses_select_feature_sets,
    optimize_deme_select_feature_sets,
)
from pymoses.feature_selection.main.feature_selection import select_feature_sets
from pymoses.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen

_TABLE_DATA_DIR = None


@pytest.fixture()
def boolean_ctable() -> CTable:
    from pathlib import Path

    table_dir = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
    return loadTable(str(table_dir / "trainingBoolean.dat"), "y").compressed()


def _hc(max_evals: int = 200) -> hill_climbing:
    """Exactly feature-selection.cc:141-159's own "hc" branch construction."""
    op_params = optim_parameters(
        opt_algo="hc", pop_size_ratio=20.0, terminate_if_gte=1.0e50, max_dist=4,
        min_score_improv=0.0,
    )
    hc_params = hc_parameters()
    hc_params.widen_search = True
    hc_params.fraction_of_nn = 2.0
    hc_params.single_step = False
    hc_params.crossover = True
    hc_params.crossover_pop_size = 300
    hc_params.crossover_min_neighbors = 1000
    hc_params.prefix_stat_deme = "FSDemes"
    return hill_climbing(op_params, hc_params)


def test_initial_instance_sets_only_the_forced_features(boolean_ctable):
    fields = field_set(field_set.disc_spec(2), boolean_ctable.get_arity())
    params = feature_selection_parameters(initial_features=["x1", "x3"])
    inst = initial_instance(params, fields, boolean_ctable.get_input_labels())
    from pymoses.feature_selection.scorers.moses_optim import get_feature_set

    assert get_feature_set(fields, inst) == {0, 2}


def test_initial_instance_empty_when_no_forced_features(boolean_ctable):
    fields = field_set(field_set.disc_spec(2), boolean_ctable.get_arity())
    labels = boolean_ctable.get_input_labels()
    inst = initial_instance(feature_selection_parameters(), fields, labels)
    assert inst == instance(fields.packed_width())


def test_iscorer_cache_returns_same_value_and_counts_hits_misses():
    calls = []

    def base_scorer(inst):
        calls.append(tuple(inst))
        from pymoses.moses.moses.types import composite_score

        return composite_score(1.0, len(inst), 0)

    cached = iscorer_cache(100, base_scorer)
    inst = instance([5])
    r1 = cached(inst)
    r2 = cached(inst)
    assert r1.get_score() == r2.get_score() == 1.0
    assert cached.get_misses() == 1
    assert cached.get_hits() == 1
    assert len(calls) == 1  # the underlying scorer was called only ONCE


def test_moses_select_feature_sets_runs_to_completion_and_returns_valid_feature_sets(
    boolean_ctable,
):
    randGen().seed(1)
    params = feature_selection_parameters(target_size=1, hc_max_evals=100)
    hc = _hc(max_evals=100)
    pop = moses_select_feature_sets(boolean_ctable, hc, params)
    assert len(pop) >= 1
    arity = boolean_ctable.get_arity()
    for score, fs in pop:
        assert isinstance(score, float)
        assert fs <= set(range(arity))


def test_create_deme_select_feature_sets_without_cache(boolean_ctable):
    """`hc_cache_size <= 0` takes the OTHER branch in
    `create_deme_select_feature_sets` (no `iscorer_cache` wrapper) --
    proven by mutation: this test alone would still pass if the
    ``if fs_params.hc_cache_size > 0`` gate were deleted (both branches
    call the same underlying scorer), so it is paired with
    `test_iscorer_cache_...` above, which DOES distinguish cached from
    uncached (call count), and with the branch-selection test below.
    """
    randGen().seed(1)
    params = feature_selection_parameters(target_size=1, hc_max_evals=50, hc_cache_size=0)
    hc = _hc(max_evals=50)
    pop = create_deme_select_feature_sets(boolean_ctable, hc, lambda fs: 0.0, params)
    assert len(pop) >= 1


def test_create_deme_select_feature_sets_hc_cache_size_gate(monkeypatch, boolean_ctable):
    """Mutation-style proof that `hc_cache_size > 0` really does select
    the `iscorer_cache`-wrapping branch: monkeypatch `iscorer_cache` to a
    spy and confirm it is constructed exactly when `hc_cache_size > 0`.
    """
    import pymoses.feature_selection.algo.deme_optimize as mod

    built = []
    real_cache = mod.iscorer_cache

    class _SpyCache(real_cache):
        def __init__(self, n, sc):
            built.append(n)
            super().__init__(n, sc)

    monkeypatch.setattr(mod, "iscorer_cache", _SpyCache)

    randGen().seed(1)
    hc = _hc(max_evals=50)
    params_cached = feature_selection_parameters(target_size=1, hc_max_evals=50, hc_cache_size=5)
    mod.create_deme_select_feature_sets(boolean_ctable, hc, lambda fs: 0.0, params_cached)
    assert built == [5]

    built.clear()
    randGen().seed(1)
    hc2 = _hc(max_evals=50)
    params_uncached = feature_selection_parameters(target_size=1, hc_max_evals=50, hc_cache_size=0)
    mod.create_deme_select_feature_sets(boolean_ctable, hc2, lambda fs: 0.0, params_uncached)
    assert built == []


def test_optimize_deme_select_feature_sets_deduplicates_exact_pairs():
    """`if pair not in fs_pop: fs_pop.append(pair)` -- proven directly by
    handing a deme that already contains two DISTINCT-score entries and
    one EXACT (score, feature_set) duplicate of the first, without
    running any real optimizer at all (`optimize` is a no-op stub)."""
    fields = field_set(field_set.disc_spec(2), 2)
    deme = instance_set(fields)

    inst_a = instance(fields.packed_width())
    fields.set_raw(inst_a, fields.begin_bit_raw_idx(), 1)  # {0}
    inst_b = instance(fields.packed_width())
    fields.set_raw(inst_b, fields.begin_bit_raw_idx() + 1, 1)  # {1}

    from pymoses.moses.moses.types import composite_score
    from pymoses.moses.representation.scored_instance import scored_instance

    # score_t is float32-narrowed (composite_score's own convention); 0.5
    # and 0.75 both round-trip exactly, avoiding a spurious float32 vs.
    # float64 literal mismatch unrelated to what this test actually checks.
    deme.append(scored_instance(inst_a, composite_score(0.5, 1, 0)))
    deme.append(scored_instance(inst_b, composite_score(0.75, 1, 0)))
    deme.append(scored_instance(instance(inst_a), composite_score(0.5, 1, 0)))  # exact duplicate

    def noop_optimize(deme, scorer, max_evals, max_time, *, init_inst=None):
        pass

    pop = optimize_deme_select_feature_sets(
        fields, deme, instance(fields.packed_width()), noop_optimize, lambda fs: 0.0,
        feature_selection_parameters(),
    )
    assert pop == [(0.75, {1}), (0.5, {0})]  # sorted descending; duplicate collapsed


def test_select_feature_sets_hc_end_to_end_via_dispatcher(boolean_ctable):
    randGen().seed(1)
    pop = select_feature_sets(
        boolean_ctable,
        feature_selection_parameters(algorithm="hc", target_size=1, hc_max_evals=100),
    )
    assert len(pop) >= 1
