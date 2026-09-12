"""Unit, gate and live-oracle tests for pymetta.moses.moses.moses_main.

Three layers, in order:

1. Pure unit tests of `adjust_termination_criteria`/`autoscale_diversity`
   and every gate (`run_moses`'s mpi/distributed branches,
   `metapop_moses_results_b`'s UNRECOGNISED-optimizer branch,
   `metapop_moses_results`'s SS-fitness branch, `metapop_printer`'s
   `output_demeID`) -- fast, in-memory `CTable`s, no file I/O, no
   dependency on `local_moses` being ready. Note "unrecognised", not
   "non-hc": all four names classic recognises (`hc`/`sa`/`un`/`ps`) now
   construct a real optimizer, which the parametrised dispatch test
   below asserts BY TYPE.

2. A wiring/integration test that drives the REAL `metapop_moses_results`
   through a FAKE `local_moses` (injected into `sys.modules`, undone by
   `monkeypatch` at teardown -- not a reimplementation of the real
   algorithm, just a recording stub) to prove the pipeline builds the
   right objects in the right order and the parameter-copying
   (`metapop_moses_results`'s own docstring) actually isolates the
   caller's `opt_params`/`meta_params`/`moses_params` from mutation.

3. Live oracle tests (`test_it`/`test_it_target_ignore`) that reproduce
   two S11-usable `MOSESUTest` cases end to end through the REAL
   `local_moses`, comparing the final score and winning tree against
   values captured by actually running classic's own `moses` binary
   (recorded in each test's own docstring, not re-derived here) --
   `dataset.csv`'s content below is copied byte-for-byte from
   `$CLASSIC/tests/moses/dataset.csv`. Bounded: `-m1000` in classic maps
   to `moses_parameters(max_evals=1000)` here, and the whole suite is run
   under `( ulimit -v 2097152; timeout 300 pytest ... )` per this task's
   own memory-safety instruction.
"""

from __future__ import annotations

import sys
import types
from collections import Counter

import pytest

from pymetta.comboreduct.combo.iostream_combo import (
    combo_tree_to_str,
    output_format,
    str_to_combo_tree,
)
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table import CTable, TimedCounter
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses import moses_main
from pymetta.moses.moses.moses_main import metapop_printer
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.optimization.particle_swarm import (
    particle_swarm,
    ps_parameters,
)
from pymetta.moses.moses.optimization.star_anneal import simulated_annealing
from pymetta.moses.moses.optimization.univariate import univariate_optimization
from pymetta.moses.moses.types import (
    _to_score_t,
    composite_score,
    demeID_t,
    score_t,
    scored_combo_tree,
    very_best_score,
)
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.moses.scoring.ss_bscore import ss_bscore

T, F = builtin.logical_true, builtin.logical_false

# ---------------------------------------------------------------------------
# Small, in-memory scorer builders (no file I/O) for the unit-test layer.
# ---------------------------------------------------------------------------


def _timed(counts: dict) -> TimedCounter:
    """A `TimedCounter` cell with every observation at the not-a-date
    sentinel -- exactly what `Table.compressed` produces for a table
    loaded WITHOUT `--timestamp-feature` (table.cc:512's `++res[*in_it]
    [TimedValue(*out_it)]` branch).
    """
    tc = TimedCounter()
    for value, count in counts.items():
        tc.add(value, None, count)
    return tc


def _tiny_cscorer() -> behave_cscore:
    """One boolean input, x1: x1=False always maps to True (3/3); x1=True
    is split 3/3 -- the same shape as the real `test_it_target_ignore`
    oracle below (verified independently against classic's own dataset),
    just built by hand instead of loaded from a file. `best_possible_score
    == -3`, `min_improv == 1.5` -- both re-derived by the tests below, not
    hardcoded twice.
    """
    ct = CTable(olabel="y", ilabels=["x1"], signature=gen_signature(type_node.boolean_type, 1))
    # `TimedCounter`, not a plain `collections.Counter`: that is what
    # `Table.compressed` builds for every real table, and `ss_bscore`'s
    # constructor calls `CTable.get_timestamps()`, which walks
    # `TimedCounter.timed_keys()`. A hand-built plain `Counter` cell has no
    # such view and makes this helper unusable by the SS-fitness tests.
    ct[(F,)] = _timed({T: 3.0, F: 0.0})
    ct[(T,)] = _timed({T: 3.0, F: 3.0})
    bsc = ctruth_table_bscore(ct)
    bsc.set_complexity_coef(3.5)
    return behave_cscore(bsc)


def _rule():
    return logical_reduction(())(2)


def _fresh_deme_parameters() -> deme_parameters:
    pa = deme_parameters()
    pa.perm_ratio = -1.0  # required: see deme_params.py's own DIVERGENCE note
    return pa


# ---------------------------------------------------------------------------
# run_moses: dispatch + gates
# ---------------------------------------------------------------------------


def test_run_moses_raises_for_mpi():
    pa = moses_parameters(local=False)
    pa.mpi = True
    with pytest.raises(NotImplementedError, match="mpi_moses"):
        moses_main.run_moses(None, None, pa, moses_statistics())


def test_run_moses_raises_for_distributed():
    pa = moses_parameters(local=False)
    assert pa.mpi is False
    with pytest.raises(NotImplementedError, match="distributed_moses"):
        moses_main.run_moses(None, None, pa, moses_statistics())


def test_run_moses_dispatches_to_local_moses(monkeypatch):
    """Injects a fake `local_moses` module into `sys.modules` rather than
    depending on the real S11 Task 1 module (owned by a concurrently
    implemented task, may not exist yet -- see this file's own module
    docstring and moses_main.py's `run_moses` docstring for why the real
    import is deferred to call time). `monkeypatch.setitem` undoes this
    automatically at teardown, so it cannot leak into other tests or
    collide with the real module once it lands.
    """
    calls = []

    def fake_local_moses(mp, dex, pa, stats):
        calls.append((mp, dex, pa, stats))

    fake_module = types.ModuleType("pymetta.moses.moses.local_moses")
    fake_module.local_moses = fake_local_moses
    monkeypatch.setitem(sys.modules, "pymetta.moses.moses.local_moses", fake_module)

    pa = moses_parameters(local=True)
    stats = moses_statistics()
    moses_main.run_moses("MP", "DEX", pa, stats)

    assert calls == [("MP", "DEX", pa, stats)]


# ---------------------------------------------------------------------------
# adjust_termination_criteria
# ---------------------------------------------------------------------------


def test_adjust_termination_criteria_infers_target_when_max_score_is_sentinel():
    """C++'s `--max-score` CLI default IS `very_best_score`
    (problem-params.cc:253) -- moses_parameters(max_score=very_best_score)
    reproduces that "infer automatically" configuration exactly.
    """
    sc = _tiny_cscorer()
    opt_params = optim_parameters()
    moses_params = moses_parameters(max_score=very_best_score)

    moses_main.adjust_termination_criteria(sc, opt_params, moses_params)

    best_possible: score_t = sc._bscorer.sum_bscore(sc._bscorer.best_possible_bscore())
    min_improv: score_t = sc._bscorer.min_improv()
    expected = _to_score_t(best_possible - max(min_improv, _to_score_t(0)))

    assert best_possible == -3.0
    assert min_improv == 1.5
    assert opt_params.terminate_if_gte == expected == -4.5
    assert moses_params.max_score == expected
    assert opt_params.min_score_improv() == min_improv


def test_adjust_termination_criteria_honors_explicit_max_score():
    """An explicit, non-sentinel `max_score` "always over-rides the
    inferred score" (moses_main.cc:78's own comment) -- the inferred
    `best_possible_score()` (-3.0 for this scorer) must NOT appear
    anywhere in the result.
    """
    sc = _tiny_cscorer()
    opt_params = optim_parameters()
    moses_params = moses_parameters(max_score=_to_score_t(10.0))

    moses_main.adjust_termination_criteria(sc, opt_params, moses_params)

    # 10.0 (explicit) - 1.5 (min_improv), NOT -3.0 (inferred) - 1.5.
    assert opt_params.terminate_if_gte == 8.5
    assert moses_params.max_score == 8.5


def test_adjust_termination_criteria_clamps_negative_min_improv_to_zero():
    """ "negative min_improv is interpreted as percentage of improvement,
    if so then don't subtract anything" (moses_main.cc:87-89) --
    `bscore_base.min_improv`'s own base-class default is 0.0
    (scoring_base.py), never negative, so this exercises the `max(...)`
    clamp directly against a scorer whose `min_improv` IS negative.
    """

    class _NegativeMinImprovBscore(ctruth_table_bscore):
        def min_improv(self) -> score_t:
            return _to_score_t(-2.0)

    ct = CTable(olabel="y", ilabels=["x1"], signature=gen_signature(type_node.boolean_type, 1))
    ct[(F,)] = Counter({T: 1.0})
    bsc = _NegativeMinImprovBscore(ct)
    bsc.set_complexity_coef(3.5)
    sc = behave_cscore(bsc)

    opt_params = optim_parameters()
    moses_params = moses_parameters(max_score=very_best_score)
    moses_main.adjust_termination_criteria(sc, opt_params, moses_params)

    # best_possible_score is 0.0 (the single row is already unanimous);
    # actual_min_improv must be clamped to 0, not -2.0.
    assert opt_params.terminate_if_gte == 0.0
    assert opt_params.min_score_improv() == -2.0  # set_min_score_improv is NOT clamped


# ---------------------------------------------------------------------------
# autoscale_diversity
# ---------------------------------------------------------------------------


def test_autoscale_diversity_noop_when_disabled():
    sc = _tiny_cscorer()
    meta_params = metapop_parameters()
    assert meta_params.diversity.enabled() is False  # pressure == 0.0, the default

    moses_main.autoscale_diversity(sc, meta_params)

    assert meta_params.diversity.pressure == 0.0


def test_autoscale_diversity_scales_pressure_when_enabled_and_autoscale_set():
    # Deliberately asymmetric (unlike `_tiny_cscorer`, whose best- and
    # worst-possible scores happen to coincide at -3.0 each, making a
    # magnitude of 0.0 a poor witness for the multiplication itself):
    # row x1=False is unanimous (best=0, worst=-3); row x1=True is a
    # 3-1 split (best=0, worst=-1). best_possible_score=0, worst=-4.
    ct = CTable(olabel="y", ilabels=["x1"], signature=gen_signature(type_node.boolean_type, 1))
    ct[(F,)] = Counter({T: 3.0, F: 0.0})
    ct[(T,)] = Counter({T: 1.0, F: 0.0})
    bsc = ctruth_table_bscore(ct)
    bsc.set_complexity_coef(3.5)
    sc = behave_cscore(bsc)

    meta_params = metapop_parameters()
    meta_params.diversity.pressure = _to_score_t(2.0)
    meta_params.diversity.autoscale = True
    assert meta_params.diversity.enabled() is True

    moses_main.autoscale_diversity(sc, meta_params)

    best = sc._bscorer.sum_bscore(sc._bscorer.best_possible_bscore())
    worst = sc._bscorer.sum_bscore(sc._bscorer.worst_possible_bscore())
    magnitude = _to_score_t(best - worst)
    assert (best, worst, magnitude) == (0.0, -4.0, 4.0)
    assert meta_params.diversity.pressure == _to_score_t(2.0 * magnitude) == 8.0


def test_autoscale_diversity_reads_autoscale_only_when_enabled():
    """Proves the short-circuit this function's own docstring claims:
    `.autoscale` is left deliberately UNSET on a fresh
    `diversity_parameters` (metapop_params.py's own DIVERGENCE note), so
    reading it eagerly would raise `AttributeError` even when diversity is
    disabled. With `pressure == 0.0` (disabled), that must never happen.
    """
    sc = _tiny_cscorer()
    meta_params = metapop_parameters()
    assert not hasattr(meta_params.diversity, "autoscale")

    moses_main.autoscale_diversity(sc, meta_params)  # must not raise

    # Enabling diversity WITHOUT setting .autoscale now must raise --
    # proving the attribute really is read once enabled() is True.
    meta_params.diversity.pressure = _to_score_t(1.0)
    with pytest.raises(AttributeError):
        moses_main.autoscale_diversity(sc, meta_params)


# ---------------------------------------------------------------------------
# metapop_moses_results_b: optimizer gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opt_algo", "expected_type"),
    [
        ("hc", hill_climbing),
        ("sa", simulated_annealing),
        ("un", univariate_optimization),
        ("ps", particle_swarm),
    ],
)
def test_metapop_moses_results_b_accepts_every_algo_classic_recognises(opt_algo, expected_type):
    """All four of classic's optimizer branches are ported, and each one
    builds the type classic builds.

    `moses_main.h:288-299` recognises exactly `hc`, `sa`, `un` and `ps`
    (names declared at `optimization.h:66-69`) and constructs
    `hill_climbing`/`simulated_annealing`/`univariate_optimization`/
    `particle_swarm` respectively. This test previously asserted that
    every non-`hc` value raised, which was true only while star-anneal,
    univariate and particle-swarm were unported.

    The type assertion is the point: an earlier version only asserted
    that the call did not raise, which a dispatch collapsed to
    `hill_climbing(...)` in all four arms would satisfy. The constructed
    optimizer is not returned, so it is recovered through the printer
    callback -- `metapop_moses_results_b` hands the printer the very
    `deme_expander` it injected the optimizer into (moses_main.h:318,322),
    and `deme_expander` stores it as `_optimize` (`optimizer_base&
    _optimize` in C++).

    Asserting construction rather than a full run: the point here is the
    dispatch, and a real search for each algo is covered by their own
    modules' tests.
    """
    seen = []
    opt_params = optim_parameters(opt_algo=opt_algo)
    moses_main.metapop_moses_results_b(
        [],
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer(),
        opt_params,
        hc_parameters(),
        ps_parameters(),
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(),
        lambda mp, dex, stats: seen.append(dex._optimize),
    )

    assert len(seen) == 1
    assert type(seen[0]) is expected_type


def test_metapop_moses_results_b_gives_particle_swarm_the_CALLERS_ps_params():
    """MUTATION PROVEN: replacing `particle_swarm(opt_params, ps_params)`
    with `particle_swarm(opt_params, ps_parameters())` -- the exact code
    that stood here until 2026-09-09 -- makes this fail (50 != 7).

    `ps_params` is the ONLY optimizer-parameter bundle that used to be
    dropped from this signature, and `moses_main.h:298` reads its VALUE.
    `calc_swarm_size` clamps to `[4, max_parts]` (particle-swarm.cc:248),
    so `max_parts` decides how many particles the search actually gets.
    """
    seen = []
    ps_params = ps_parameters()
    ps_params.max_parts = 7
    moses_main.metapop_moses_results_b(
        [],
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer(),
        optim_parameters(opt_algo="ps"),
        hc_parameters(),
        ps_params,
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(),
        lambda mp, dex, stats: seen.append(dex._optimize),
    )
    assert len(seen) == 1
    assert seen[0].ps_params.max_parts == 7


def test_metapop_moses_results_passes_ps_params_through_unmodified(monkeypatch):
    """`metapop_moses_results` is a pure pass-through for `ps_params`
    (moses_main.h:376,382) -- it is not one of the "parameters that might
    get tweaked" (moses_main.h:357), so it is neither copied nor altered
    on the way to `metapop_moses_results_b`. Dropping it from the
    forwarding call, or forwarding a fresh `ps_parameters()`, fails here.
    """
    received = []
    monkeypatch.setattr(
        moses_main,
        "metapop_moses_results_b",
        lambda *args: received.append(args),
    )
    ps_params = ps_parameters()
    ps_params.max_parts = 11
    moses_main.metapop_moses_results(
        [],
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer(),
        optim_parameters(),
        hc_parameters(),
        ps_params,
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(),
        lambda mp, dex, stats: None,
    )
    assert len(received) == 1
    # Position 7, between hc_params (6) and deme_params (8) -- C++'s own
    # argument order at moses_main.h:376-379.
    assert received[0][7] is ps_params
    assert received[0][7].max_parts == 11


def test_metapop_moses_results_b_rejects_an_unrecognised_algo():
    """An unknown name raises, carrying classic's own message text.

    DIVERGENCE, and it is deliberate: C++ (`moses_main.h:300-306`) writes
    "Unknown optimization algo ..." to stderr and calls `exit(1)` -- a hard
    process exit, not a throw, and not catchable in either language. A
    library cannot faithfully reproduce a process exit, so this port raises
    `ValueError` with the same text. Silently accepting the name would be
    worse than either.
    """
    opt_params = optim_parameters(opt_algo="bogus")
    with pytest.raises(ValueError, match="Unknown optimization algo bogus"):
        moses_main.metapop_moses_results_b(
            [],
            gen_signature(type_node.boolean_type, 1),
            _rule(),
            _rule(),
            _tiny_cscorer(),
            opt_params,
            hc_parameters(),
            ps_parameters(),
            _fresh_deme_parameters(),
            subsample_deme_filter_parameters(),
            metapop_parameters(),
            moses_parameters(),
            lambda mp, dex, stats: None,
        )


def test_metapop_moses_results_b_accepts_hc_optimizer(monkeypatch):
    """The complement of the gate test above: `opt_algo="hc"` (the
    default) must NOT raise, and must reach `run_moses` -- proven via the
    same fake-`local_moses` injection `test_run_moses_dispatches_to_
    local_moses` uses.
    """
    fake_module = types.ModuleType("pymetta.moses.moses.local_moses")
    fake_module.local_moses = lambda mp, dex, pa, stats: None
    monkeypatch.setitem(sys.modules, "pymetta.moses.moses.local_moses", fake_module)

    printed = []
    moses_main.metapop_moses_results_b(
        [combo_tree(builtin.logical_and)],
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer(),
        optim_parameters(opt_algo="hc"),
        hc_parameters(),
        ps_parameters(),
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(),
        lambda mp, dex, stats: printed.append(True),
    )
    assert printed == [True]


def test_metapop_moses_results_b_does_not_mutate_caller_bases(monkeypatch):
    """`combo_tree siba(xmplr); si_ca(siba);` (moses_main.h:313-314)
    copy-constructs before simplifying -- the caller's own `and()`
    exemplar must still print as `and()` afterward, even though its
    SIMPLIFIED copy (which is what actually gets scored) reduces to
    `true` (verified directly: `logical_reduction(())(2)` folds a
    childless `and()` to `true`).
    """
    fake_module = types.ModuleType("pymetta.moses.moses.local_moses")
    fake_module.local_moses = lambda mp, dex, pa, stats: None
    monkeypatch.setitem(sys.modules, "pymetta.moses.moses.local_moses", fake_module)

    exemplar = combo_tree(builtin.logical_and)
    bases = [exemplar]

    moses_main.metapop_moses_results_b(
        bases,
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer(),
        optim_parameters(opt_algo="hc"),
        hc_parameters(),
        ps_parameters(),
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(),
        lambda mp, dex, stats: None,
    )

    # `ostream_combo_tree` prints a childless `and` as "and " (no
    # parentheses), so this checks tree STRUCTURE directly rather than
    # relying on a printed-form assumption: still exactly the childless
    # `logical_and` root the caller passed in, not the `true` its
    # simplified COPY reduces to.
    root = exemplar.begin()
    assert root.data == builtin.logical_and
    assert root.first_child is None
    assert bases[0] is exemplar


# ---------------------------------------------------------------------------
# metapop_moses_results: parameter copying + SS-fitness gate
# ---------------------------------------------------------------------------


def _wire_with_fake_local_moses(
    monkeypatch, *, filter_params=None, meta_params=None, printer=None, c_scorer=None
):
    fake_module = types.ModuleType("pymetta.moses.moses.local_moses")

    def fake_local_moses(mp, dex, pa, stats):
        stats.n_evals = 42

    fake_module.local_moses = fake_local_moses
    monkeypatch.setitem(sys.modules, "pymetta.moses.moses.local_moses", fake_module)

    opt_params = optim_parameters()
    moses_params = moses_parameters(max_evals=1, max_score=very_best_score)
    if meta_params is None:
        meta_params = metapop_parameters()
    if filter_params is None:
        filter_params = subsample_deme_filter_parameters()

    calls = []
    if printer is not None:
        real_printer = printer
    else:

        def real_printer(mp, dex, stats):
            calls.append((mp, dex, stats))

    moses_main.metapop_moses_results(
        [combo_tree(builtin.logical_and)],
        gen_signature(type_node.boolean_type, 1),
        _rule(),
        _rule(),
        _tiny_cscorer() if c_scorer is None else c_scorer,
        opt_params,
        hc_parameters(),
        ps_parameters(),
        _fresh_deme_parameters(),
        filter_params,
        meta_params,
        moses_params,
        real_printer,
    )
    return opt_params, moses_params, meta_params, calls


def test_metapop_moses_results_wires_pipeline_and_calls_printer(monkeypatch):
    opt_params, moses_params, meta_params, calls = _wire_with_fake_local_moses(monkeypatch)

    assert len(calls) == 1
    mp, dex, stats = calls[0]
    assert isinstance(mp, metapopulation)
    assert isinstance(dex, deme_expander)
    assert isinstance(stats, moses_statistics)
    assert stats.n_evals == 42  # proves run_moses handed the fake the SAME stats object


def test_metapop_moses_results_copies_opt_and_moses_params(monkeypatch):
    """`twk_opt_params`/`twk_moses_params` must be independent copies:
    `adjust_termination_criteria` mutates them internally, and the
    caller's own `opt_params`/`moses_params` must come back untouched.
    """
    opt_params, moses_params, _meta_params, _calls = _wire_with_fake_local_moses(monkeypatch)

    assert opt_params.terminate_if_gte == 0.0  # optim_parameters()'s own default, untouched
    assert moses_params.max_score == very_best_score  # untouched sentinel, not the inferred target


def test_metapop_moses_results_copies_diversity_params(monkeypatch):
    """The aliasing trap this function's own docstring calls out: a bare
    `copy.copy(meta_params)` would alias `.diversity`, so
    `autoscale_diversity`'s in-place mutation of the COPY would corrupt
    the caller's own `meta_params.diversity` too. Enabling diversity with
    autoscale on makes this directly observable: the caller's `pressure`
    must stay exactly 2.0, while the METAPOPULATION actually built (whose
    `._params` IS the `twk_meta_params` copy) must show the scaled value.
    """
    meta_params = metapop_parameters()
    meta_params.diversity.pressure = _to_score_t(2.0)
    meta_params.diversity.autoscale = True

    _opt, _moses, meta_params_after, calls = _wire_with_fake_local_moses(
        monkeypatch, meta_params=meta_params
    )

    assert meta_params_after.diversity.pressure == 2.0  # caller's own object: untouched
    mp, _dex, _stats = calls[0]
    assert mp._params.diversity.pressure != 2.0  # the copy actually used: scaled
    assert mp._params is not meta_params_after


def test_metapop_moses_results_wraps_in_ss_bscore_when_configured(monkeypatch):
    """C++: moses_main.h:364-378. `n_subsample_fitnesses > 1 and
    low_dev_pressure > 0.0` must hand `metapop_moses_results_b` a DIFFERENT
    `behave_cscore`, built around an `ss_bscore` wrapping the caller's own
    bscorer -- not the caller's cscorer. This used to be a
    `NotImplementedError` gate.
    """
    filter_params = subsample_deme_filter_parameters()
    filter_params.n_subsample_fitnesses = 2
    filter_params.low_dev_pressure = 0.1
    filter_params.by_time = False

    c_scorer = _tiny_cscorer()
    calls = _wire_with_fake_local_moses(
        monkeypatch, filter_params=filter_params, c_scorer=c_scorer
    )[3]

    assert len(calls) == 1
    mp, _dex, _stats = calls[0]
    used = mp._cscorer
    assert used is not c_scorer, "the SS branch must build its own behave_cscore"
    assert isinstance(used._bscorer, ss_bscore)
    # `get_bscorer()` hands back the very object, not a copy
    # (behave_cscore.h:128 returns `bscore_base&`).
    assert used._bscorer._bscorer is c_scorer._bscorer
    assert used._bscorer._n_subsamples == 2
    assert used._bscorer._by_time is False


def test_metapop_moses_results_ss_bscore_reads_by_time_from_filter_params(monkeypatch):
    """`filter_params.by_time` is the FOURTH constructor argument
    (moses_main.h:372) and is read from the caller's own `filter_params`,
    not from any tweaked copy.
    """
    filter_params = subsample_deme_filter_parameters()
    filter_params.n_subsample_fitnesses = 3
    filter_params.low_dev_pressure = 0.25
    filter_params.by_time = True

    calls = _wire_with_fake_local_moses(monkeypatch, filter_params=filter_params)[3]
    ssb = calls[0][0]._cscorer._bscorer
    assert ssb._by_time is True
    assert ssb._n_subsamples == 3
    assert ssb._low_dev_pressure == _to_score_t(0.25)


def test_metapop_moses_results_ss_bscore_inherits_zero_complexity_coef(monkeypatch):
    """The counter-intuitive consequence of `ss_bscore : public
    bscore_base` (ss_bscore.h:38): `get_complexity_coef()` is
    `bscore_base`'s own `_complexity_coef`, `0.0` (scoring_base.h:59), NOT
    the wrapped bscorer's -- so an SS-fitness run silently loses the
    complexity penalty. Classic's, reproduced.
    """
    filter_params = subsample_deme_filter_parameters()
    filter_params.n_subsample_fitnesses = 2
    filter_params.low_dev_pressure = 1.0
    filter_params.by_time = False

    c_scorer = _tiny_cscorer()
    # `set_complexity_coef(3.5)` stores the RECIPROCAL (scoring_base.cc:50).
    assert c_scorer._bscorer.get_complexity_coef() != 0.0

    calls = _wire_with_fake_local_moses(
        monkeypatch, filter_params=filter_params, c_scorer=c_scorer
    )[3]
    assert calls[0][0]._cscorer._bscorer.get_complexity_coef() == 0.0


@pytest.mark.parametrize(
    ("n_subsample_fitnesses", "low_dev_pressure"),
    [(1, 0.1), (2, 0.0), (0, 0.0)],
)
def test_metapop_moses_results_ss_bscore_gate_does_not_fire_below_threshold(
    monkeypatch, n_subsample_fitnesses, low_dev_pressure
):
    filter_params = subsample_deme_filter_parameters()
    filter_params.n_subsample_fitnesses = n_subsample_fitnesses
    filter_params.low_dev_pressure = low_dev_pressure

    _opt, _moses, _meta, calls = _wire_with_fake_local_moses(
        monkeypatch, filter_params=filter_params
    )
    assert len(calls) == 1  # reached the printer; did not raise


def test_metapop_moses_results_ss_bscore_gate_default_filter_params_never_raises_attributeerror(
    monkeypatch,
):
    """`subsample_deme_filter_parameters()` leaves `n_subsample_fitnesses`/
    `low_dev_pressure` UNSET by design (metapop_params.py's own
    DIVERGENCE note) -- `getattr(..., 0)`/`getattr(..., 0.0)` must treat
    that as "not configured" rather than propagating `AttributeError`.
    """
    filter_params = subsample_deme_filter_parameters()
    assert not hasattr(filter_params, "n_subsample_fitnesses")
    assert not hasattr(filter_params, "low_dev_pressure")

    _opt, _moses, _meta, calls = _wire_with_fake_local_moses(
        monkeypatch, filter_params=filter_params
    )
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Live oracle: reproduces MOSESUTest's test_it / test_it_target_ignore.
#
# `_DATASET_CSV` below is byte-for-byte `$CLASSIC/tests/moses/dataset.csv`
# (9 rows, columns res,arg1,arg2 -- confirmed via `diff` when this file was
# written). Oracle values captured by running classic's own binary,
# bounded, under this task's own memory/time limits:
#
#   ( ulimit -v 2097152; timeout 120 ./build/moses/moses/main/moses \
#       -i tests/moses/dataset.csv -m1000 -r0 -W1 )
#   -> "-3 and($arg1 $arg2)" is the top (highest-scoring) printed line.
#
#   ( ulimit -v 2097152; timeout 120 ./build/moses/moses/main/moses \
#       -i tests/moses/dataset.csv -m1000 -uarg2 -Yres -r0 -W1 -c50 )
#   -> "-3 true" is the ONLY printed line (arity-1 search, single result
#      at the best score).
#
# Both were ALSO re-run against classic with --hc-crossover=0 and produced
# BYTE-IDENTICAL output to the crossover-enabled (default) run.
#
# That does NOT mean this port's own `hill_climbing` can be driven with
# `hc_parameters()`'s real default (`cross=True`) here, though: measured
# directly (first draft of this test), the FULL multi-generation
# `local_moses` loop -- unlike a single `deme_expander.optimize_demes`
# call -- genuinely reaches hill_climbing.py's crossover-trigger guard on
# a later generation (`iteration=4, last_chance=True`) and raises, exactly
# as that guard is supposed to when crossover mechanics are needed but not
# ported (S12). Since classic's own `--hc-crossover=0` run is confirmed
# byte-identical to its default-crossover run for both cases above,
# passing `hc_parameters(cross=False)` here reproduces the verified
# oracle exactly while staying inside what this port actually implements
# -- not a silent behavioural change, a disclosed substitution for an
# input already shown to make no difference to the real answer.
# ---------------------------------------------------------------------------

_DATASET_CSV = """res,arg1,arg2
0,0,1
0,1,0
1,1,1
0,0,1
0,1,0
1,1,1
1,0,1
1,1,0
0,1,1
"""


def _load_dataset_cscorer(tmp_path, target_feature, ignore_features=()):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(_DATASET_CSV)
    table = loadTable(str(csv_path), target_feature, "", list(ignore_features))
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    bsc.set_complexity_coef(3.5)  # CLI default --complexity-ratio (problem-params.cc:335)
    return behave_cscore(bsc), table.get_arity()


def _run_it_style(tmp_path, target_feature, ignore_features=()):
    sc, arity = _load_dataset_cscorer(tmp_path, target_feature, ignore_features)
    sig = gen_signature(type_node.boolean_type, arity)
    rule = _rule()

    result = {}
    moses_main.metapop_moses_results(
        [combo_tree(builtin.logical_and)],  # type_to_exemplar(boolean_type)
        sig,
        rule,
        rule,
        sc,
        optim_parameters(),
        hc_parameters(cross=False),  # see this section's own module-docstring note
        ps_parameters(),
        _fresh_deme_parameters(),
        subsample_deme_filter_parameters(),
        metapop_parameters(),
        moses_parameters(max_evals=1000, max_score=very_best_score),  # -m1000
        lambda mp, dex, stats: result.update(mp=mp, dex=dex, stats=stats),
    )
    return result["mp"]


def test_it_reproduces_mosesutest(tmp_path):
    # `tree` (util/tree.py) defines no `__eq__`, so `==` on two separately
    # constructed trees is identity comparison, not structural -- the same
    # reason test_deme_expander.py compares printed forms
    # (`combo_tree_to_str(...).rstrip()`) rather than `tree == tree`.
    mp = _run_it_style(tmp_path, "res")
    assert mp.best_score() == -3.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "and($1 $2)"


def test_it_target_ignore_reproduces_mosesutest(tmp_path):
    mp = _run_it_style(tmp_path, "arg2", ignore_features=["res"])
    assert mp.best_score() == -3.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "true"


# ---------------------------------------------------------------------------
# metapop_printer (S13 Task 3) -- pure unit tests against directly-built
# `scored_combo_tree`/composite_score objects and a lightweight duck-typed
# `_FakeMetapop`/`_FakeEnsemble`, deliberately NOT routed through a real
# search (`_run_it_style`/`_fresh_deme_parameters` above hardcodes
# `perm_ratio = -1.0`, the exact S8 scope decision docs/CARRY-FORWARD.md
# Sec 5.4 records as "a scope decision that silently became a correctness
# bug" -- an oracle built on it is exactly what that section warns is
# unable to detect the port's own configuration being wrong). Every
# EXPECTED string below is copied verbatim from a real, captured run of
# `$CLASSIC/build/moses/moses/main/moses` (command quoted in each test's
# own docstring) -- the genuine "classic's own defaults, no configuration
# supplied by this port" oracle for `metapop_printer` itself lives in
# `tests/moses/main/test_moses_exec.py` instead (the REAL CLI path,
# `problem_params.py` wiring `perm_ratio` to its real default 0.0).
# ---------------------------------------------------------------------------


class _FakeEnsemble:
    """Not a ported C++ symbol -- a duck-typed stand-in for
    `pymetta.moses.metapopulation.ensemble.ensemble` whose `flat_score`/
    `get_weighted_tree` raise, so a test can assert
    `metapop_printer.__call__` propagates that failure unmodified rather
    than masking or mis-handling it.

    CORRECTED: this docstring used to say it "raises exactly where the
    real class's own `flat_score`/`get_weighted_tree` do (both unported
    stubs, `ensemble.py`'s own docstring)". `ensemble.py` is fully ported
    now -- `flat_score` (ensemble.cc:427) and `get_weighted_tree`
    (ensemble.cc:280) are real implementations. The real `flat_score`
    still fails, but one layer further down and with a DIFFERENT message
    (`bscore_base.call_ensemble`, which no concrete scorer in this port
    overrides), so this double no longer reproduces the real failure
    text. `test_metapop_printer_output_ensemble_flat_score_dies_in_
    call_ensemble_for_real` below pins the real object's real failure;
    this double stays only to prove the printer does not swallow an
    exception raised from that call site.
    """

    def __init__(self, members=()):
        self._members = list(members)

    def get_ensemble(self):
        return self._members

    def flat_score(self):
        raise NotImplementedError("ensemble::flat_score (ensemble.cc) is not ported")

    def get_weighted_tree(self):
        raise NotImplementedError("ensemble::get_weighted_tree (ensemble.cc) is not ported")


class _FakeMetapop:
    """Not a ported C++ symbol -- exposes exactly the five methods
    `metapop_printer.__call__` calls on its `metapop` argument
    (`size`, `get_ensemble`, `best_candidates`, `get_trees`,
    `best_score`), so printer-formatting tests do not need a real search
    (see this section's own header comment for why that matters here).
    """

    def __init__(self, trees, best=None, ensemble_members=(), best_score_val=0.0, size=None):
        self._trees = list(trees)
        self._best = list(best) if best is not None else list(self._trees)
        self._ensemble = _FakeEnsemble(ensemble_members)
        self._best_score_val = best_score_val
        self._size = size if size is not None else len(self._trees)

    def get_trees(self):
        return self._trees

    def best_candidates(self):
        return self._best

    def get_ensemble(self):
        return self._ensemble

    def best_score(self):
        return self._best_score_val

    def size(self):
        return self._size


def _sct(combo_str, score, complexity, complexity_penalty, uniformity_penalty=0.0, bscore=None):
    tr = str_to_combo_tree(combo_str)
    cs = composite_score(score, complexity, complexity_penalty, uniformity_penalty)
    return scored_combo_tree(tr, cs=cs, bs=bscore if bscore is not None else [])


def _printer(**overrides):
    kwargs = dict(
        result_count=10,
        output_score=True,
        output_cscore=False,
        output_bscore=False,
        output_only_best=False,
        output_ensemble=False,
        output_eval_number=False,
        output_with_labels=False,
        output_demeID=False,
        ilabels=[],
        output_file="",
        fmt=output_format.combo,
        is_mpi=False,
    )
    kwargs.update(overrides)
    return metapop_printer(**kwargs)


def test_metapop_printer_combo_format_no_labels(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000` (no `-W`) -> top line `-3 and($1 $2) `.
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    _printer()(mp, None, moses_statistics())
    assert capsys.readouterr().out == "-3 and($1 $2) \n"


def test_metapop_printer_combo_format_with_labels(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000 -W1` -> top line `-3 and($arg1 $arg2) `.
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    printer = _printer(output_with_labels=True, ilabels=["arg1", "arg2"])
    printer(mp, None, moses_statistics())
    assert capsys.readouterr().out == "-3 and($arg1 $arg2) \n"


def test_metapop_printer_labels_gate_is_mutation_detectable(capsys):
    """Same fixture as the no-labels test above but with
    `output_with_labels=False` and `ilabels` populated anyway (exactly
    what a real `it_table_problem.run()` does: `pms.printer.ilabels` is
    always set to the table's real column labels regardless of `-W`,
    table-problems.cc:164 -- only `_labels()`'s own gate decides whether
    they are actually used). Proves the gate by construction: a version
    of `_labels()` that ignored `output_with_labels` (returned
    `self.ilabels` unconditionally) would print `$arg1`/`$arg2` here
    instead of `$1`/`$2` -- verified by mutation (see task-3-report.md).
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    printer = _printer(output_with_labels=False, ilabels=["arg1", "arg2"])
    printer(mp, None, moses_statistics())
    assert capsys.readouterr().out == "-3 and($1 $2) \n"


def test_metapop_printer_cscore(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000 -x1` -> `-3 and($1 $2)  [score=-3, penalized score=
    -3.57142853736877441, complexity=2, complexity penalty=
    0.571428596973419189, uniformity penalty=0]`.
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    _printer(output_cscore=True)(mp, None, moses_statistics())
    out = capsys.readouterr().out
    assert out == (
        "-3 and($1 $2)  [score=-3, penalized score=-3.57142853736877441, "
        "complexity=2, complexity penalty=0.571428596973419189, "
        "uniformity penalty=0]\n"
    )


def test_metapop_printer_bscore(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000 -t1` -> `-3 and($1 $2)  behavioral score: [-1 -1 -1]`.
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714, bscore=[-1.0, -1.0, -1.0])])
    _printer(output_bscore=True)(mp, None, moses_statistics())
    assert capsys.readouterr().out == "-3 and($1 $2)  behavioral score: [-1 -1 -1]\n"


def test_metapop_printer_python_format(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000 --output-format=python`, first generated program (verbatim,
    including its own boilerplate).
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    _printer(fmt=output_format.python)(mp, None, moses_statistics())
    assert capsys.readouterr().out == (
        "#!/usr/bin/env python\n"
        "#score: -3\n"
        "import operator as op\n"
        "from functools import reduce\n"
        "from math import log, exp, sin\n"
        "def l0(i): return 0 < i\n"
        "def adds(*args): return sum(args)\n"
        "def muls(*args): return reduce(op.mul, args)\n"
        "def pdiv(a, b): return a / (b + 0.000001)\n"
        "def impulse(a): return 1.0 if a else 0.0\n"
        "def moses_eval(i):\n"
        "    return (i[0] and i[1]) \n"
    )


@pytest.mark.parametrize(
    ("result_count", "expected_count"),
    [(0, 0), (1, 1), (2, 2), (3, 3), (-1, 3)],
)
def test_metapop_printer_result_count_boundaries(capsys, result_count, expected_count):
    """`if (result_count == cnt++) break;` (moses_main.h:159) compares the
    PRE-increment `cnt` -- `result_count=0` prints NOTHING (breaks before
    the very first candidate), `result_count=N>0` prints exactly N, and a
    NEGATIVE `result_count` prints every candidate (`-c`'s own help text:
    "If negative, then return all results.", problem-params.cc:1083-1084)
    since `cnt` never goes negative. Mutation-provable: swapping to
    post-increment comparison (`cnt == result_count` after `cnt += 1`)
    would print one FEWER candidate at every positive `result_count` --
    verified red/green (task-3-report.md).
    """
    trees = [
        _sct("and($1 $2)", -3.0, 2, 0.5714285714285714),
        _sct("$1", -4.0, 1, 0.2857142857142857),
        _sct("$2", -4.0, 1, 0.2857142857142857),
    ]
    mp = _FakeMetapop(trees)
    _printer(result_count=result_count)(mp, None, moses_statistics())
    printed_lines = capsys.readouterr().out.splitlines()
    assert len(printed_lines) == expected_count


def test_metapop_printer_output_only_best_selects_best_candidates(capsys):
    all_trees = [
        _sct("and($1 $2)", -3.0, 2, 0.5714285714285714),
        _sct("$1", -4.0, 1, 0.2857142857142857),
    ]
    best_only = [all_trees[0]]
    mp = _FakeMetapop(all_trees, best=best_only)
    _printer(output_only_best=True)(mp, None, moses_statistics())
    assert capsys.readouterr().out == "-3 and($1 $2) \n"


def test_metapop_printer_output_eval_number(capsys):
    """Oracle: `$CLASSIC/build/moses/moses/main/moses -i dataset.csv
    -m1000 -V1` -> last line `n_evals: 33`.
    """
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    stats = moses_statistics()
    stats.n_evals = 33
    _printer(output_eval_number=True)(mp, None, stats)
    assert capsys.readouterr().out == "-3 and($1 $2) \nn_evals: 33\n"


def test_metapop_printer_output_file_writes_file_not_stdout(capsys, tmp_path):
    out_path = tmp_path / "results.txt"
    mp = _FakeMetapop([_sct("and($1 $2)", -3.0, 2, 0.5714285714285714)])
    _printer(output_file=str(out_path))(mp, None, moses_statistics())
    assert capsys.readouterr().out == ""
    assert out_path.read_text() == "-3 and($1 $2) \n"


def test_metapop_printer_is_mpi_empty_metapop_produces_no_output(capsys):
    mp = _FakeMetapop([], size=0)
    _printer(is_mpi=True, output_eval_number=True)(mp, None, moses_statistics())
    assert capsys.readouterr().out == ""


def test_metapop_printer_output_demeID_prints_the_deme_the_candidate_came_from(capsys):
    """The arm raised while `scored_combo_tree` carried no `demeID_t`. It
    carries one now, and `demeID_t` subclasses `str` (types.h:190 makes
    C++'s a `std::string` subclass), so `out << sct.get_demeID()` is a
    direct append.

    Two candidates with DIFFERENT ids, so a printer that emitted one
    constant -- or the default -- could not pass.
    """
    a = _sct("and($1 $2)", -3.0, 2, 0.5714285714285714)
    b = _sct("$1", -4.0, 1, 0.2857142857142857)
    a.deme_id = demeID_t(1)
    b.deme_id = demeID_t(2, 3)
    mp = _FakeMetapop([a, b])

    _printer(output_demeID=True, output_cscore=False)(mp, None, moses_statistics())
    out = capsys.readouterr().out

    assert out == "-3 and($1 $2)  demeID: 1\n-4 $1  demeID: 2.3\n"


def test_metapop_printer_python_format_ensemble_gates_on_the_boosting_weight():
    """The python-format ensemble arm prints `sct.get_weight()` per member
    (moses_main.h:124), and this port gates it rather than printing a
    wrong number.

    `scored_combo_tree.get_weight()` exists (types.py, ported off
    types.h:291) and returns the constructor's own 1.0 default. But
    `ensemble.py` stores its real AdaBoost/experts weights in a private
    `_sct_weights` dict keyed by object identity, not on the
    `scored_combo_tree` -- so without the gate every member would print a
    flat `1.0` where classic prints its real alpha. Before `get_weight`
    was ported this line failed with `AttributeError`; the gate exists so
    that adding the accessor could not silently convert an accidental
    failure into wrong output.

    The `if tree_set:` guard is deliberate and asserted below: an EMPTY
    ensemble reaches no `get_weight()` call at all, so gating it would
    raise on a case that has nothing wrong with it.
    """
    member = _sct("and($1 $2)", -3.0, 2, 0.5714285714285714)
    mp = _FakeMetapop([], ensemble_members=[member])
    with pytest.raises(NotImplementedError, match="get_weight"):
        _printer(output_ensemble=True, fmt=output_format.python)(mp, None, moses_statistics())


def test_metapop_printer_python_format_empty_ensemble_is_not_gated(capsys):
    """The negative half of the gate above: with NO members there is no
    `get_weight()` call to be wrong about, so the branch must run and
    emit classic's boilerplate -- including the `return (0.0 < val)` bug
    reproduced verbatim off moses_main.h:129 (`val` is never defined).
    """
    mp = _FakeMetapop([], ensemble_members=[], best_score_val=-3.0)
    _printer(output_ensemble=True, fmt=output_format.python)(mp, None, moses_statistics())
    out = capsys.readouterr().out
    assert out.startswith("#!/usr/bin/env python\n#score: -3\n")
    assert out.endswith("    return (0.0 < val)\n")


def test_metapop_printer_output_ensemble_flat_score_dies_in_call_ensemble_for_real():
    """Pins what ACTUALLY stops `output_ensemble=True` in combo format,
    using the real `ensemble` rather than `_FakeEnsemble`.

    `metapop_printer.__call__` does not gate this branch at all (it is
    ported inline off moses_main.h:111-136, with no raise in it). The
    combo-format arm prints `ensm.flat_score()`, and the real
    `ensemble.flat_score` (ensemble.cc:427) calls
    `bscore_base.call_ensemble` -- whose base body raises in BOTH
    languages (`OC_ASSERT(false, "Ensemble scoring not implemented for
    bscorer %s", ...)`, scoring_base.cc:74-75) because no concrete
    scorer in this port overrides it yet.

    Asserting the real message text, not just the exception type: that is
    what distinguishes "stopped one layer down, in the scorer" (true)
    from "gated in metapop_printer" (what this module's docstring used to
    claim).
    """
    from pymetta.moses.metapopulation.ensemble import ensemble

    ensm = ensemble(_tiny_cscorer())
    with pytest.raises(NotImplementedError, match="Ensemble scoring not implemented"):
        ensm.flat_score()


def test_metapop_printer_output_ensemble_propagates_unported_failure():
    mp = _FakeMetapop([], ensemble_members=[])
    with pytest.raises(NotImplementedError, match="flat_score"):
        _printer(output_ensemble=True)(mp, None, moses_statistics())
