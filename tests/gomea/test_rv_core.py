"""NOT A PORT, T3.2 (server round 16): the RV-GOMEA core (``pymoses/gomea/rv_core.py``,
``--gomea-rv-step rvg``) and its seed fix.

The first tests come from the T3.2 spike 2 (commit be83e08, ``tests/gomea/test_rv_core.py``),
adapted to the flags of the branch: the ``cap`` budget rule and ``--gomea-rv-cap`` are left
out, so ``test_rvg_end_to_end`` has the cases ``balance`` and ``late`` only, and
``test_flag_values_are_checked`` checks that ``cap`` is refused.

The seed-fix tests check the rule of ``SERVER-ROUND16-PREREGISTRATION.md`` ("The seed fix"):
the RV stream comes from the run seed (``-r``) and the deme index, and never draws from the
global generator. So the first deme of a run with ``--gomea-rv-step rvg`` starts from the same
initial population as the run with the flag off.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.gomea import optimizer as gomea_optimizer
from pymoses.gomea import rv_core
from pymoses.gomea.oracle import EvaluationLedger
from pymoses.gomea.rv_core import ETA_DEC, ETA_INC, NIS_MAX, RV_SEED_SALT, RVCore
from pymoses.gomea.schema import FrozenSchema
from pymoses.util.rng import MT19937RandGen, randGen

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"


class _Schema:
    """The parts of ``FrozenSchema`` that the free codes use."""

    contin_positions = (0, 1)
    contin_depths = (2, 2)
    dim = 2

    def __init__(self):
        self._n_free = 0
        self._free_values = [[], []]
        self._free_codes = [{}, {}]

    register_contin_value = FrozenSchema.register_contin_value


class _Oracle:
    def __init__(self):
        self.ledger = EvaluationLedger()
        self.iscorer = object()
        self._memo = {}
        self._seen = set()
        self._last_index = {}


def _core():
    return RVCore(_Schema(), _Oracle(), None, MT19937RandGen(1), None)


def test_adapt_multiplier_follows_algorithm_3():
    core = _core()
    core.c[0] = 1.0
    # An improvement far from the mean (SDR 2 > 1): c grows, pop-NIS resets.
    core.pop_nis = 5
    core._adapt(0, 0.0, 1.0, [2.0])
    assert core.c[0] == pytest.approx(ETA_INC) and core.pop_nis == 0
    # An improvement near the mean (SDR 0.5): c stays.
    core._adapt(0, 0.0, 1.0, [0.5])
    assert core.c[0] == pytest.approx(ETA_INC)
    # No improvement with c > 1: c shrinks; pop-NIS does not count while c > 1.
    core._adapt(0, 0.0, 1.0, [])
    assert core.c[0] == pytest.approx(ETA_INC * ETA_DEC) and core.pop_nis == 0
    # No improvement with c <= 1 and pop-NIS < NIS_MAX: c is set back to 1.
    core.c[0] = 1.0
    core._adapt(0, 0.0, 1.0, [])
    assert core.c[0] == 1.0 and core.pop_nis == 1
    # pop-NIS reaches NIS_MAX: c can fall below 1.
    core.pop_nis = NIS_MAX - 1
    core._adapt(0, 0.0, 1.0, [])
    assert core.c[0] == pytest.approx(ETA_DEC)
    # An improvement sets c < 1 back to 1 first.
    core._adapt(0, 0.0, 1.0, [0.1])
    assert core.c[0] == 1.0


def test_rollback_restores_the_free_codes_and_the_oracle():
    core = _core()
    s, o = core.schema, core.oracle
    kept = s.register_contin_value(0, 3.5)  # a value kept earlier
    trial, new = core._trial((0, 0), {0: 3.5, 1: -2.25})
    assert trial == (kept, 16) and new == [(1, 16, -2.25)]
    o._memo[trial] = "score"
    o._seen.add(trial)
    o._last_index[trial] = 7
    core._rollback(trial, new)
    assert s._free_values == [[3.5], []] and s._free_codes == [{3.5: kept}, {}]
    assert s._n_free == 1
    assert trial not in o._memo and trial not in o._seen and trial not in o._last_index
    # The code is free again for the next value.
    assert s.register_contin_value(1, 9.0) == 16


def _run(argv, tmp_path):
    from pymoses.moses.main.moses_exec import moses_exec

    log = tmp_path / "events.jsonl"
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()]


def _argv(seed: str = "1") -> list[str]:
    return [
        "-i", str(DATA / "train.csv"), "-u", "y", "-r", seed, "-n", "sin", "-n", "exp",
        "-n", "log", "-v", "2400", "-c", "3", "-V1", "-l", "WARN", "-m", "1500", "-a", "gomea",
        "--gomea-pop-size", "24", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "terms",
        "--gomea-deme-store", "accepted", "--gomea-accept-complexity-ratio", "0.125",
    ]  # fmt: skip


_ARGV = _argv()


@pytest.mark.parametrize("budget", ["balance", "late"])
def test_rvg_end_to_end(budget, tmp_path):
    out, events = _run(_ARGV + ["--gomea-rv-step", "rvg", "--gomea-rv-budget", budget], tmp_path)
    ends = [e for e in events if e["type"] == "end"]
    n_evals = int(out.rsplit("n_evals:", 1)[1])
    assert n_evals <= 1500
    assert n_evals == sum(e["ledger"]["requests"] for e in ends)
    ran = 0
    for e in ends:
        rv = e["rv"]
        assert rv["mode"] == "rvg" and rv["budget"] == budget
        assert rv["rv_evals"] <= e["ledger"]["requests"]
        # The fields of the left-out cap rule stay, with no value.
        assert rv["cap"] is None and rv["skipped_cap"] == 0 and rv["cap_hits"] == 0
        ran += rv["generations"]
        if e["termination_reason"] == "converged":
            assert rv["converged_by"] in ("structure", "no_evals")
    assert ran > 0
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert all(s["params"]["rv_step"] == "rvg" for s in starts)
    assert all(s["params"]["rv_budget"] == budget and s["params"]["rv_seed"] == 1 for s in starts)


def test_rvg_is_reproducible(tmp_path):
    outs = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        outs.append(_run(_ARGV + ["--gomea-rv-step", "rvg"], d)[0])
    assert outs[0] == outs[1]


def test_flag_values_are_checked():
    from pymoses.gomea.optimizer import gomea_parameters

    with pytest.raises(ValueError):
        gomea_parameters(contin="copy", rv_step="rvg", rv_budget="nope")
    # The spike's "cap" rule and its first-spike modes are left out.
    with pytest.raises(ValueError):
        gomea_parameters(contin="copy", rv_step="rvg", rv_budget="cap")
    for mode in ("uni", "block"):
        with pytest.raises(ValueError):
            gomea_parameters(contin="copy", rv_step=mode)
    # The spike's refusals.
    with pytest.raises(ValueError, match="contin 'copy'"):
        gomea_parameters(rv_step="rvg")
    with pytest.raises(ValueError, match="const_opt 'none'"):
        gomea_parameters(contin="copy", rv_step="rvg", const_opt="mutate")
    with pytest.raises(ValueError):
        gomea_parameters(contin="copy", rv_step="rvg", rv_seed="1")
    d = gomea_parameters(contin="copy").as_dict()
    assert "rv_step" not in d and "rv_budget" not in d and "rv_seed" not in d
    d = gomea_parameters(contin="copy", rv_budget="late", rv_seed=7).as_dict()
    assert "rv_step" not in d and "rv_budget" not in d and "rv_seed" not in d
    d = gomea_parameters(contin="copy", rv_step="rvg", rv_budget="late", rv_seed=7).as_dict()
    assert (d["rv_step"], d["rv_budget"], d["rv_seed"]) == ("rvg", "late", 7)
    assert rv_core.P_ACCEPT == 0.05 and rv_core.TAU == 0.35


@pytest.mark.parametrize("flag", ["--gomea-rv-cap", "--gomea-rv-inert", "--gomea-rv-nis"])
def test_the_left_out_spike_flags_are_unknown(flag, tmp_path):
    with pytest.raises(SystemExit):
        _run(_ARGV + ["--gomea-rv-step", "rvg", flag, "1"], tmp_path)


def test_the_cli_refuses_the_cap_rule(tmp_path):
    with pytest.raises(ValueError, match="rv_budget"):
        _run(_ARGV + ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "cap"], tmp_path)


# -- the seed fix ----------------------------------------------------------------------------


def _state(rng: MT19937RandGen) -> tuple:
    return (tuple(rng._mt), rng._mti)


def test_the_seed_rule():
    """The seed of deme k is draw k + 1 of the master generator of the run seed."""
    for run_seed in (0, 1, 7, 2**32 + 3, -5):
        master = MT19937RandGen((run_seed ^ RV_SEED_SALT) & 0xFFFFFFFF)
        draws = [master.randint() for _ in range(4)]
        assert [rv_core.rv_seed(run_seed, k) for k in range(4)] == draws
        a, b = rv_core.rv_stream(run_seed, 2), MT19937RandGen(draws[2])
        assert [a.randint() for _ in range(5)] == [b.randint() for _ in range(5)]
    assert len({rv_core.rv_seed(1, k) for k in range(50)}) == 50
    assert rv_core.rv_seed(1, 0) != rv_core.rv_seed(2, 0)
    with pytest.raises(ValueError):
        rv_core.rv_seed(1, -1)


def test_the_rv_stream_does_not_change_the_global_generator():
    g = randGen()
    g.seed(12345)
    g.randint()  # a state in the middle of the stream
    before = _state(g)
    for k in range(3):
        rv_core.rv_stream(12345, k)
    assert _state(g) == before
    # The optimizer's stream of each call (deme), with the flag on.
    gp = gomea_optimizer.gomea_parameters(contin="copy", rv_step="rvg", rv_seed=12345)
    opt = gomea_optimizer.gomea_optimization(None, gp)
    streams = [opt._rv_stream() for _ in range(3)]
    assert _state(g) == before
    assert opt._rv_demes == 3
    expect = [rv_core.rv_stream(12345, k).randint() for k in range(3)]
    assert [s.randint() for s in streams] == expect
    # With the flag off: no stream and no count.
    off = gomea_optimizer.gomea_optimization(None, gomea_optimizer.gomea_parameters())
    assert off._rv_stream() is None and off._rv_demes == 0
    assert _state(g) == before
    # The spike's rule (one draw from randGen()) changed the global generator.
    MT19937RandGen(g.randint())
    assert _state(g) != before


def _run_with_inits(argv, tmp_path, monkeypatch):
    """``_run``, plus the result of each ``build_population`` call (the initial populations,
    in deme order) and the (run seed, deme index) of each RV stream."""
    inits, streams = [], []
    real_build, real_stream = gomea_optimizer.build_population, gomea_optimizer.rv_stream

    def build(*a, **kw):
        res = real_build(*a, **kw)
        inits.append(res)
        return res

    def stream(run_seed, k):
        streams.append((run_seed, k))
        return real_stream(run_seed, k)

    monkeypatch.setattr(gomea_optimizer, "build_population", build)
    monkeypatch.setattr(gomea_optimizer, "rv_stream", stream)
    tmp_path.mkdir(parents=True, exist_ok=True)
    out, events = _run(argv, tmp_path)
    monkeypatch.undo()
    return out, events, inits, streams


def _first(events, kind):
    return next(e for e in events if e["type"] == kind)


@pytest.mark.parametrize("seed", ["1", "2"])
@pytest.mark.parametrize("budget", ["balance", "late"])
def test_the_first_deme_starts_as_in_the_off_run(budget, seed, tmp_path, monkeypatch):
    """With the flag on, the first deme's initial population (genotypes, attempts,
    duplicates rejected) and its first GOM generation are those of the off run."""
    _, ev_off, init_off, st_off = _run_with_inits(_argv(seed), tmp_path / "off", monkeypatch)
    argv = _argv(seed) + ["--gomea-rv-step", "rvg", "--gomea-rv-budget", budget]
    _, ev_on, init_on, st_on = _run_with_inits(argv, tmp_path / "on", monkeypatch)
    a, b = init_off[0], init_on[0]
    assert b.genotypes == a.genotypes
    assert (b.attempts, b.duplicates_rejected) == (a.attempts, a.duplicates_rejected)
    assert (b.radii, b.coverage_replacements) == (a.radii, a.coverage_replacements)
    # The "init" event of the first deme, field for field.
    assert _first(ev_on, "init") == _first(ev_off, "init")
    s_off, s_on = _first(ev_off, "invocation_start"), _first(ev_on, "invocation_start")
    assert {k: v for k, v in s_on.items() if k != "params"} == {
        k: v for k, v in s_off.items() if k != "params"
    }
    # The first GOM generation is the same; the RV generation comes after it.
    g_off, g_on = _first(ev_off, "generation"), _first(ev_on, "generation")
    assert g_on["stats"] == g_off["stats"] and g_on["fos_size"] == g_off["fos_size"]
    assert "rv" in g_on and "rv" not in g_off
    # Each deme of the on run has the stream of its index; the off run has none.
    n_demes = sum(1 for e in ev_on if e["type"] == "invocation_start")
    assert st_on == [(int(seed), k) for k in range(n_demes)]
    assert st_off == []


# -- the RV16 review (nits 3 to 6) -----------------------------------------------------------


@pytest.mark.parametrize("algo", ["hc", "hc-capped", "hc-gomea", "sa", "un", "ps"])
def test_the_cli_refuses_rvg_without_gomea(algo, tmp_path):
    """Nit 3: -a hc and hc-capped ignored the flag; -a hc-gomea ran RV with a deme index that
    counts the GOMEA phases. Every algorithm other than gomea is refused."""
    argv = list(_ARGV)
    argv[argv.index("-a") + 1] = algo
    with pytest.raises(ValueError, match=f"--gomea-rv-step rvg needs -a gomea, got -a {algo}"):
        _run(argv + ["--gomea-rv-step", "rvg"], tmp_path)


@pytest.mark.parametrize("ls", ["0", "1"])
def test_the_cli_refuses_rvg_without_linear_scaling_terms(ls, tmp_path):
    """Nit 3: the inert rule assumes the per-term fit (it skipped live genes as "absorbed")."""
    argv = list(_ARGV)
    argv[argv.index("--linear-scaling") + 1] = ls
    want = f"needs --linear-scaling terms, got --linear-scaling {ls}"
    with pytest.raises(ValueError, match=want):
        _run(argv + ["--gomea-rv-step", "rvg"], tmp_path)
    # the off value is accepted as before
    out, _ = _run(argv + ["--gomea-rv-step", "off", "-m", "200"], tmp_path)
    assert "n_evals:" in out


def test_the_hybrid_refuses_rvg():
    from pymoses.gomea.hybrid import hc_gomea_optimization

    gp = gomea_optimizer.gomea_parameters(contin="copy", rv_step="rvg")
    with pytest.raises(ValueError, match="rv_step needs -a gomea, not the hc-gomea hybrid"):
        hc_gomea_optimization(None, None, gp)
    hc_gomea_optimization(None, None, gomea_optimizer.gomea_parameters(contin="copy"))


def test_nis_max_seen_is_a_running_maximum():
    """Nit 4: NIS goes back to 0 after an improvement (and after a forced improvement), so the
    highest NIS seen can be above max(nis) at the end of the deme."""
    core = _core()
    assert core.record()["nis_max_seen"] == 0
    core.nis = [0, 0, 0]
    for _ in range(3):
        assert core._step_nis(1, False) is False
    assert core._step_nis(1, True) is False and core.nis == [0, 0, 0]
    core._step_nis(2, False)
    assert core.record()["nis_max_seen"] == 3  # max(self.nis) gives 1
    core.nis = [NIS_MAX, 0]  # the population size changed: the maximum stays
    assert core._step_nis(0, False) is True  # NIS_MAX + 1: a forced improvement is due
    core.nis[0] = 0  # as _generation does after the forced improvement
    assert core.record()["nis_max_seen"] == NIS_MAX + 1


class _FISchema(_Schema):
    """``_Schema`` with a decoder: a code below 4**depth is its own value."""

    def contin_value(self, p, code):
        base = 4 ** self.contin_depths[p]
        return self._free_values[p][code - base] if code >= base else float(code)


class _FIEv:
    """An evaluator over ``_Oracle``: each trial is worse than the recipient; every second call
    is a memo hit (charged 0), like the oracle's memo."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.calls = 0

    def evaluate(self, g):
        self.calls += 1
        if self.calls % 2:
            self.oracle.ledger.requests += 1
        return -100.0

    def worse(self, a, b):
        return a < b

    def phenotype_key(self, g):
        return None


class _FIEngine:
    def __init__(self, elite):
        self.elite = elite

    def offer_elite(self, ind):
        pass


def test_forced_improvement_counts_only_charged_evaluations():
    """Nit 5: fi_evals and evals counted each trial, also a memo hit; now they count the
    charged evaluations, as _try does. rv_evals does not change."""
    from collections import Counter

    from pymoses.gomea.gom import Individual

    oracle = _Oracle()
    ev = _FIEv(oracle)
    core = RVCore(_FISchema(), oracle, ev, MT19937RandGen(1), None)
    elite = Individual((3, 5), -1.0, None)
    P = [elite, Individual((1, 1), -50.0, None)]
    cnt = Counter()
    core._forced_improvement(_FIEngine(elite), P, 1, [{0, 1}, {0, 1}], [{0, 1}, {0, 1}], cnt)
    # 6 blend factors (0.5 down to 0.015625) x 2 genes, none improves: P_1 becomes the elite
    assert ev.calls == 12 and oracle.ledger.requests == 6 and core.rv_evals == 6
    assert cnt["fi_triggers"] == 1 and cnt["fi_improved"] == 0 and cnt["fi_elite_copies"] == 1
    assert cnt["fi_evals"] == 6 and cnt["evals"] == 6  # was 12 and 12
    assert P[1] is elite
    # every trial was rolled back: no free code is left
    assert core.schema._n_free == 0


def test_a_deme_that_ends_before_the_engine_has_no_rv_record(tmp_path):
    """Nit 6 (the stale record fix of clone K): a deme that ends in its initial population
    (here the budget runs out) gets "rv": null, not the totals of the deme before it."""
    (tmp_path / "full").mkdir()
    _, events = _run(_ARGV + ["--gomea-rv-step", "rvg"], tmp_path / "full")
    first = next(e for e in events if e["type"] == "end")
    assert isinstance(first["rv"], dict) and first["rv"]["generations"] > 0
    used = first["ledger"]["requests"]
    # the same run with a budget 5 evaluations above the first deme: the second deme stops
    # after 5 evaluations of its initial population of 24
    argv = _ARGV + ["--gomea-rv-step", "rvg"]
    argv[argv.index("-m") + 1] = str(used + 5)
    (tmp_path / "cut").mkdir()
    out, events = _run(argv, tmp_path / "cut")
    ends = [e for e in events if e["type"] == "end"]
    assert len(ends) == 2 and _same_but_clocks(ends[0]["rv"], first["rv"])
    assert ends[0]["ledger"] == first["ledger"]
    assert ends[1]["termination_reason"] == "global_budget" and ends[1]["generations"] == 0
    assert ends[1]["ledger"]["requests"] == 5 and ends[1]["rv"] is None
    assert int(out.rsplit("n_evals:", 1)[1]) == used + 5
    # the unit view of the fix: a new call clears the core of the call before it
    gp = gomea_optimizer.gomea_parameters(contin="copy", rv_step="rvg", rv_seed=1)
    opt = gomea_optimizer.gomea_optimization(None, gp)
    opt._rv = "stale"
    opt._rv_stream()
    assert opt._rv is None


_CLOCKS = frozenset({"eval_s", "step_s", "analyse_s", "timings"})


def _same_but_clocks(a: dict, b: dict) -> bool:
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in _CLOCKS}
        return x

    return strip(a) == strip(b)
