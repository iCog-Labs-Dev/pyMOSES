"""Server round 13, BASE-PS: the pole check after LM (``--const-opt-pole-check box``).

After LM improves a program at the end of a deme, the old and the tuned program are
predicted on 256 probe points in the box of the training inputs (each range widened by 10 %
on each side), each with its linear scaling fitted on the training rows. The tuned program
is rejected (the old one is kept) if a probe prediction of it is not finite while all of
the old one are finite, or if its largest distance from the mean training target is above
10 times the target range and above that of the old program. See
``docs/gomea/SERVER-ROUND13-PREREGISTRATION.md`` ("BASE-PS").
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.gomea import optimizer as gomea_optimizer
from pymoses.gomea.const_opt import (
    POLE_PROBES,
    LMResult,
    RegressionRows,
    pole_check,
    pole_probes,
    probe_spread,
)
from pymoses.gomea.hc_lm import hc_lm_optimization
from pymoses.gomea.hybrid import hc_gomea_optimization
from pymoses.gomea.optimizer import (
    _new_const_opt_counts,
    gomea_optimization,
    gomea_parameters,
)
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import contin_bscore
from pymoses.util.rng import randGen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

TRAIN = Path(__file__).resolve().parent / "data" / "const_opt_none" / "train.csv"

# The old program has no pole (x1^2 + 30 > 0). The tuned one has poles at x1 = +-3.5: inside
# the probe box (x1 in [-3.6, 3.6]), outside the training rows (x1 in [-3, 3]), where it is
# finite.
_OLD = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) 30)))"
_POLE = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) -12.25)))"
_HARMLESS = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) 29)))"
# log(x1 + 3.05) is defined on every training row, not on the probes with x1 < -3.05.
_DOMAIN = "+(sin(*(1.3 $1)) log(+($1 3.05)))"


def _bscore():
    t = loadTable(str(TRAIN), "y")
    b = contin_bscore(t.otable, t.itable)
    b.set_linear_scaling(True)
    return b


def _rows():
    return RegressionRows.of_bscore(_bscore())


def _t(s):
    return str_to_combo_tree(s)


# --- the probes ------------------------------------------------------------------------


def test_the_probes_fill_the_widened_box_of_the_training_inputs():
    rows = _rows()
    probes = pole_probes(rows, 1)
    assert len(probes) == POLE_PROBES == 256
    for j in range(2):
        col = [r[j] for r in rows.inputs]
        lo, hi = min(col), max(col)
        m = 0.1 * (hi - lo)
        vals = [p[j] for p in probes]
        assert all(lo - m <= v <= hi + m for v in vals)
        assert min(vals) < lo and max(vals) > hi  # the margins are probed too


def test_the_probes_come_from_their_own_generator():
    """The same seed gives the same probes, another seed others; the global generator
    (``randGen()``, every other random stream of the run) does not move."""
    rows = _rows()
    randGen().seed(5)
    ref = [randGen().randint() for _ in range(4)]
    randGen().seed(5)
    a = pole_probes(rows, 1)
    assert [randGen().randint() for _ in range(4)] == ref
    assert pole_probes(rows, 1) == a
    assert pole_probes(rows, 2) != a


def test_the_probes_need_numeric_inputs():
    rows = RegressionRows([[1.0, True], [2.0, False]], [0.0, 1.0], True)
    with pytest.raises(ValueError, match="numeric inputs"):
        pole_probes(rows, 1)


# --- the check -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_tuning_that_moves_a_pole_into_the_probe_box_is_rejected(seed):
    rows = _rows()
    probes = pole_probes(rows, seed)
    assert all(math.isfinite(x) for x in rows.residuals(_t(_POLE))[0])  # LM sees no pole
    v = pole_check(_t(_OLD), _t(_POLE), rows, probes, lambda: None)
    r = max(rows.targets) - min(rows.targets)
    assert not v["accepted"]
    assert v["pole_s_new"] > 10 * r and v["pole_s_new"] > v["pole_s_old"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_harmless_tuning_is_accepted(seed):
    rows = _rows()
    v = pole_check(_t(_OLD), _t(_HARMLESS), rows, pole_probes(rows, seed), lambda: None)
    assert v["accepted"]
    assert v["pole_s_new"] == pytest.approx(v["pole_s_old"], rel=0.01)


def test_a_prediction_that_is_not_finite_rejects_the_tuning():
    rows = _rows()
    probes = [[-3.5, 1.0], [0.0, 1.0]]
    s, finite = probe_spread(_t(_DOMAIN), rows, probes)
    assert s == math.inf and not finite
    v = pole_check(_t(_OLD), _t(_DOMAIN), rows, probes, lambda: None)
    assert not v["accepted"] and v["pole_s_new"] == math.inf


def test_when_both_programs_have_a_prediction_that_is_not_finite_the_check_accepts():
    """Agent completion of the rule: a non-finite prediction counts as +inf in S, so (a)
    does not apply and (b) gives inf > inf, false."""
    rows = _rows()
    probes = [[-3.5, 1.0], [0.0, 1.0]]
    other = "+(sin(*(1.2 $1)) log(+($1 3.05)))"
    v = pole_check(_t(_DOMAIN), _t(other), rows, probes, lambda: None)
    assert v["accepted"] and v["pole_s_old"] == v["pole_s_new"] == math.inf


def test_a_large_spread_that_is_not_larger_than_the_old_one_is_accepted():
    """Rule (b) needs both S(new) > 10 R and S(new) > S(old)."""
    rows = _rows()
    probes = pole_probes(rows, 1)
    v = pole_check(_t(_POLE), _t(_POLE), rows, probes, lambda: None)
    r = max(rows.targets) - min(rows.targets)
    assert v["pole_s_new"] > 10 * r and v["accepted"]


def test_the_check_charges_two_passes_and_stops_at_the_budget():
    rows = _rows()
    probes = pole_probes(rows, 1)
    n = [0]

    def charge():
        n[0] += 1

    pole_check(_t(_OLD), _t(_POLE), rows, probes, charge)
    assert n[0] == 2
    for left in (0, 1):
        used = [0]

        def limited(left=left, used=used):
            if used[0] >= left:
                raise BudgetExhausted("global_budget")
            used[0] += 1

        with pytest.raises(BudgetExhausted):
            pole_check(_t(_OLD), _t(_POLE), rows, probes, limited)
        assert used[0] == left


# --- at the end of a deme (``_tune_best``) --------------------------------------------


class _Deme(list):
    n_evals = 0


class _Schema:
    """The part of ``FrozenSchema`` that ``_tune_best`` uses; the program is fixed."""

    def __init__(self, prog):
        self._tree = _t(prog)

    def candidate(self, g, reduce=True):
        return combo_tree(self._tree.begin())

    def to_instance(self, g):
        return instance(list(g))


class _Oracle:
    """The part of ``EvaluationOracle`` that ``_tune_best`` uses, with a budget of
    ``max_evals`` charged evaluations."""

    def __init__(self, cscorer, max_evals=10**6):
        self.iscorer = SimpleNamespace(_cscorer=cscorer, _reduce=True)
        self.ledger = SimpleNamespace(requests=0, tuning_requests=0)
        self.max_evals = max_evals

    def best_genotype(self):
        return (7,)

    def charge_tuning(self):
        if self.ledger.requests >= self.max_evals:
            raise BudgetExhausted("global_budget")
        self.ledger.requests += 1
        self.ledger.tuning_requests += 1


def _tune(monkeypatch, new_prog, guard=True, max_evals=10**6, lm_evals=5):
    """``_tune_best`` with an LM that "improves" the old program to ``new_prog`` after
    ``lm_evals`` charged passes. Returns (deme, record, counts, oracle)."""
    cscorer = behave_cscore(_bscore())

    def fake_lm(tree, rows, charge, *a, **k):
        for _ in range(lm_evals):
            charge()
        return LMResult("improved", 2, 1.0, 0.5, evals=lm_evals, tree=_t(new_prog))

    monkeypatch.setattr(gomea_optimizer, "levenberg_marquardt", fake_lm)
    kw = dict(const_opt_guard="val", const_opt_val=str(TRAIN)) if guard else {}
    opt = gomea_optimization(gp=gomea_parameters(
        const_opt="lm", const_opt_pole_check="box", const_opt_pole_seed=1, **kw))
    opt._co = _new_const_opt_counts()
    deme = _Deme()
    old_score = cscorer.get_cscore(_t(_OLD))
    deme.append(scored_instance(instance([7]), old_score))
    oracle = _Oracle(cscorer, max_evals)
    opt._tune_best(deme, oracle, _Schema(_OLD))
    return deme, opt._co["lm"], opt._co, oracle


def test_tune_best_rejects_a_pole_and_keeps_the_old_program(monkeypatch):
    """The check runs before the guard (no guard keys): two probe passes on top of LM."""
    deme, rec, co, oracle = _tune(monkeypatch, _POLE)
    assert rec["pole"] == "rejected" and rec["written_back"] is False
    assert "guard" not in rec and "lm_guard_checked" not in co
    assert len(deme) == 1 and getattr(deme[0], "tree", None) is None
    assert co["lm_pole_checked"] == 1 and co["lm_pole_rejected"] == 1
    assert rec["evals"] == oracle.ledger.requests == 5 + 2
    assert deme.n_evals == 7


def test_tune_best_accepts_a_harmless_tuning_then_runs_the_guard(monkeypatch):
    deme, rec, co, oracle = _tune(monkeypatch, _HARMLESS)
    assert rec["pole"] == "accepted" and rec["guard"] in ("accepted", "rejected")
    assert co["lm_pole_checked"] == 1 and co.get("lm_pole_rejected", 0) == 0
    assert co["lm_guard_checked"] == 1
    # LM, two probe passes, two validation passes, and the score if the guard accepts
    assert rec["evals"] == 5 + 2 + 2 + (1 if rec["guard"] == "accepted" else 0)
    if rec["guard"] == "accepted":
        assert rec["written_back"] and combo_tree_to_str(deme[-1].tree).strip() == _HARMLESS


def test_tune_best_without_the_guard(monkeypatch):
    deme, rec, co, _oracle = _tune(monkeypatch, _HARMLESS, guard=False)
    assert rec["pole"] == "accepted" and rec["written_back"] and rec["evals"] == 5 + 2 + 1
    assert len(deme) == 1 and deme[0].tree is not None


@pytest.mark.parametrize("max_evals", [5, 6])
def test_tune_best_at_the_budget_end_keeps_the_old_program(monkeypatch, max_evals):
    """The budget ends before or between the two probe passes: as for the guard, the
    record has ``budget`` and the old program stays; no pass goes over the budget."""
    deme, rec, co, oracle = _tune(monkeypatch, _POLE, max_evals=max_evals)
    assert rec["budget"] is True and rec["written_back"] is False
    assert "pole" not in rec and "lm_pole_checked" not in co
    assert oracle.ledger.requests == max_evals
    assert len(deme) == 1 and getattr(deme[0], "tree", None) is None


def test_tune_best_draws_the_probes_once_per_run(monkeypatch):
    calls = []
    orig = gomea_optimizer.pole_probes

    def spy(rows, seed, *a, **k):
        calls.append(seed)
        return orig(rows, seed, *a, **k)

    monkeypatch.setattr(gomea_optimizer, "pole_probes", spy)
    cscorer = behave_cscore(_bscore())
    opt = gomea_optimization(gp=gomea_parameters(
        const_opt="lm", const_opt_pole_check="box", const_opt_pole_seed=9))
    for _ in range(3):
        opt._co = _new_const_opt_counts()
        deme = _Deme()
        deme.append(scored_instance(instance([7]), cscorer.get_cscore(_t(_OLD))))
        opt._tune_best(deme, _Oracle(cscorer), _Schema(_OLD))
    assert calls == [9]


# --- parameters, telemetry and the command line ----------------------------------------


def test_parameters_off_by_default_and_left_out_of_as_dict():
    d = gomea_parameters().as_dict()
    assert "const_opt_pole_check" not in d and "const_opt_pole_seed" not in d
    d = gomea_parameters(const_opt="lm", const_opt_pole_seed=4).as_dict()
    assert "const_opt_pole_check" not in d and "const_opt_pole_seed" not in d
    d = gomea_parameters(const_opt="lm", const_opt_pole_check="box", const_opt_pole_seed=4)
    assert d.as_dict()["const_opt_pole_check"] == "box"
    assert d.as_dict()["const_opt_pole_seed"] == 4


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(const_opt_pole_check="grid", const_opt="lm"), "must be one of"),
        (dict(const_opt_pole_check="box"), "needs const_opt 'lm'"),
        (dict(const_opt_pole_check="box", const_opt="lm", const_opt_top_k=2), "top_k"),
        (dict(const_opt_pole_check="box", const_opt="lm", contin="copy",
              const_opt_during="elite"), "const_opt_during"),
        (dict(const_opt_pole_check="box", const_opt="lm", const_opt_pole_seed=1.5), "integer"),
    ],
)  # fmt: skip
def test_parameters_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        gomea_parameters(**kw)


def test_hc_lm_and_the_hybrid_refuse_the_check():
    gp = gomea_parameters(const_opt="lm", const_opt_pole_check="box")
    with pytest.raises(ValueError, match="not -a hc"):
        hc_lm_optimization(gp=gp)
    with pytest.raises(ValueError, match="hc-gomea"):
        hc_gomea_optimization(gp=gp)


_REG = ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", "2400", "-c", "10", "-V1", "-l", "WARN"]  # fmt: skip
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
    "--const-opt-val", str(TRAIN),
]  # fmt: skip
_PS = ["--const-opt-pole-check", "box"]


@pytest.mark.parametrize(
    "extra, match",
    [
        (["-a", "gomea", "--const-opt-pole-check", "grid", "--const-opt", "lm"],
         "--const-opt-pole-check grid is not recognized"),
        (["-a", "gomea", *_PS], "--const-opt-pole-check box needs --const-opt lm"),
        (["-a", "gomea", "--const-opt", "mutate", "--gomea-contin", "copy", *_PS],
         "needs --const-opt lm"),
        (["-a", "hc", "--const-opt", "lm", *_PS], "needs -a gomea, got -a hc"),
        (["-a", "hc-gomea", "--const-opt", "lm", *_PS], "needs -a gomea, got -a hc-gomea"),
        (["-a", "gomea", "--const-opt", "lm", "--const-opt-top-k", "2", *_PS],
         "--const-opt-pole-check box and --const-opt-top-k exclude each other"),
        (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "lm",
          "--const-opt-during", "elite", *_PS],
         "--const-opt-pole-check box and --const-opt-during elite exclude each other"),
        (["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "lm", *_PS],
         "--linear-scaling terms does not support --const-opt-pole-check box"),
    ],
)  # fmt: skip
def test_cli_refusals(extra, match):
    with pytest.raises(ValueError, match=match):
        moses_exec(["moses", *_REG, "-m", "50", *extra])


def _run(argv, tmp_path, name):
    """One in-process run in ``tmp_path``; returns (stdout, events)."""
    log = tmp_path / f"{name}.jsonl"
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    return buf.getvalue(), [json.loads(x) for x in log.read_text().splitlines()], log


def _ends(events):
    return [e for e in events if e["type"] == "end"]


def test_without_an_lm_improvement_the_check_changes_nothing_but_its_telemetry(
    tmp_path, monkeypatch
):
    """BASE at -m 1500 on this table: LM never improves, so no check runs. The probes are
    still drawn (at the first LM call), from their own generator: the search, the output
    and every event are those of BASE, apart from the option in the parameters and the two
    zero counters of each "end" event."""
    calls = []
    orig = gomea_optimizer.pole_probes

    def spy(rows, seed, *a, **k):
        calls.append(seed)
        return orig(rows, seed, *a, **k)

    monkeypatch.setattr(gomea_optimizer, "pole_probes", spy)
    out0, ev0, _ = _run(_REG + ["-m", "1500"] + _BASE, tmp_path, "base")
    assert calls == []
    out1, ev1, _ = _run(_REG + ["-m", "1500"] + _BASE + _PS, tmp_path, "ps")
    assert calls == [1]
    assert out1 == out0
    assert not any(e["const_opt"]["lm_improved"] for e in _ends(ev0))
    assert len(ev0) == len(ev1)
    clock = {"linkage_s", "variation_s", "timings", "max_time"}
    for a, b in zip(ev0, ev1, strict=True):
        a = {k: v for k, v in a.items() if k not in clock}
        b = {k: v for k, v in b.items() if k not in clock}
        if b["type"] == "invocation_start":
            assert b["params"].pop("const_opt_pole_check") == "box"
            assert b["params"].pop("const_opt_pole_seed") == 1
            a["params"].pop("telemetry_path"), b["params"].pop("telemetry_path")
        if b["type"] == "end":
            assert b["const_opt"].pop("lm_pole_checked") == 0
            assert b["const_opt"].pop("lm_pole_rejected") == 0
        assert a == b


def test_a_run_with_the_check_counts_it_and_stays_in_the_budget(tmp_path):
    """With a deme cap, LM improves in several demes: each check is in the record and the
    counts, and the run's evaluations never pass -m. The runner sums the counts."""
    m = 3000
    argv = _REG + ["-m", str(m)] + _BASE + _PS + ["--gomea-max-evals-per-deme", "300"]
    out, events, log = _run(argv, tmp_path, "ps")
    ends = _ends(events)
    checked = sum(e["const_opt"]["lm_pole_checked"] for e in ends)
    rejected = sum(e["const_opt"]["lm_pole_rejected"] for e in ends)
    improved = sum(e["const_opt"]["lm_improved"] for e in ends)
    assert checked >= 2 and checked <= improved
    for e in ends:
        lm = e["const_opt"]["lm"]
        if lm is not None and lm["status"] == "improved" and not lm["budget"]:
            assert lm["pole"] in ("accepted", "rejected")
            assert lm["pole_s_new"] >= 0 and lm["pole_s_old"] >= 0
    assert sum(e["ledger"]["requests"] for e in ends) <= m
    n_evals = int(out.strip().splitlines()[-1].split()[-1])
    assert n_evals <= m
    s = runner.const_opt_summary(log, n_evals)
    assert s["lm_pole_checked"] == checked and s["lm_pole_rejected"] == rejected
    assert s["lm_pole_rejected_share"] == pytest.approx(rejected / checked)


def test_the_runner_summary_has_no_pole_keys_without_the_option(tmp_path):
    _out, _events, log = _run(_REG + ["-m", "600"] + _BASE, tmp_path, "base")
    s = runner.const_opt_summary(log, 600)
    assert not [k for k in s if "pole" in k]
