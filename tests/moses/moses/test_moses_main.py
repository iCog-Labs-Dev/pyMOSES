"""Unit, gate and live-oracle tests for pymetta.moses.moses.moses_main.

Three layers, in order:

1. Pure unit tests of `adjust_termination_criteria`/`autoscale_diversity`
   and every gate (`run_moses`'s mpi/distributed branches,
   `metapop_moses_results_b`'s non-"hc" optimizer branch,
   `metapop_moses_results`'s SS-fitness branch) -- fast, in-memory
   `CTable`s, no file I/O, no dependency on `local_moses` being ready.

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

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table import CTable
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
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import _to_score_t, score_t, very_best_score
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore

T, F = builtin.logical_true, builtin.logical_false

# ---------------------------------------------------------------------------
# Small, in-memory scorer builders (no file I/O) for the unit-test layer.
# ---------------------------------------------------------------------------


def _tiny_cscorer() -> behave_cscore:
    """One boolean input, x1: x1=False always maps to True (3/3); x1=True
    is split 3/3 -- the same shape as the real `test_it_target_ignore`
    oracle below (verified independently against classic's own dataset),
    just built by hand instead of loaded from a file. `best_possible_score
    == -3`, `min_improv == 1.5` -- both re-derived by the tests below, not
    hardcoded twice.
    """
    ct = CTable(olabel="y", ilabels=["x1"], signature=gen_signature(type_node.boolean_type, 1))
    ct[(F,)] = Counter({T: 3.0, F: 0.0})
    ct[(T,)] = Counter({T: 3.0, F: 3.0})
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
    """"negative min_improv is interpreted as percentage of improvement,
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


@pytest.mark.parametrize("opt_algo", ["sa", "un", "ps", "bogus"])
def test_metapop_moses_results_b_raises_for_non_hc_optimizer(opt_algo):
    opt_params = optim_parameters(opt_algo=opt_algo)
    with pytest.raises(NotImplementedError, match=opt_algo):
        moses_main.metapop_moses_results_b(
            [],
            gen_signature(type_node.boolean_type, 1),
            _rule(),
            _rule(),
            _tiny_cscorer(),
            opt_params,
            hc_parameters(),
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


def _wire_with_fake_local_moses(monkeypatch, *, filter_params=None, meta_params=None, printer=None):
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
        _tiny_cscorer(),
        opt_params,
        hc_parameters(),
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


def test_metapop_moses_results_ss_bscore_gate_raises_when_configured(monkeypatch):
    filter_params = subsample_deme_filter_parameters()
    filter_params.n_subsample_fitnesses = 2
    filter_params.low_dev_pressure = 0.1

    with pytest.raises(NotImplementedError, match="SS-fitness"):
        _wire_with_fake_local_moses(monkeypatch, filter_params=filter_params)


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
