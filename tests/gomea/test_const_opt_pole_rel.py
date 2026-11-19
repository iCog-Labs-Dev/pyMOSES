"""Server round 14, NP-LMR-PSR: the relative pole rule (``--const-opt-pole-check box-rel``).

``box-rel`` is ``box`` (server round 13, BASE-PS, ``test_const_opt_pole.py``) with one other
rule. The probes, the place (after LM improves a program, before the guard), the two
charged probe passes, the non-finite rule, the write-back and the telemetry do not change.
With S = the probe spread (``probe_spread``; a prediction that is not finite counts as
+inf) and R = the range of the training target, the tuned program is rejected if one of its
predictions is not finite while all of the old one are finite, or if
S(tuned) > 10 * max(S(old), R). See ``docs/gomea/SERVER-ROUND14-PREREGISTRATION.md``
("NP-LMR-PSR: the relative pole rule").
"""

from __future__ import annotations

import contextlib
import io
import itertools
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
    POLE_CHECK_MODES,
    POLE_FACTOR,
    LMResult,
    RegressionRows,
    pole_check,
    pole_probes,
    pole_rejects,
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner as runner  # noqa: E402

TRAIN = Path(__file__).resolve().parent / "data" / "const_opt_none" / "train.csv"
REF = Path(__file__).resolve().parent / "data" / "pole_check_dee8101"

INF = math.inf

# The programs of test_const_opt_pole.py. The old program has no pole; the tuned one has
# poles at x1 = +-3.5 (inside the probe box, outside the training rows).
_OLD = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) 30)))"
_POLE = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) -12.25)))"
_HARMLESS = "+(sin(*(1.3 $1)) /(10 +(*($1 $1) 29)))"
_DOMAIN = "+(sin(*(1.3 $1)) log(+($1 3.05)))"


def _near(c: str) -> str:
    """A program with poles at x1 = +-sqrt(-c), near the probe x1 = 3.45 of ``_NEAR``."""
    return f"+(sin(*(1.3 $1)) /(10 +(*($1 $1) {c})))"


# Two fixed probes: x1 = 3.45 is near the poles of the ``_near`` programs. On this table
# (R = 5.858), S(_near("-12.0")) = 128.5 > 10 R, S(_near("-11.95")) = 259.2 (in
# (S(old), 10 S(old)]) and S(_near("-11.9")) = 4837.7 (> 10 S(old)).
_NEAR = [[3.45, 1.0], [0.0, 1.0]]
_NEAR_OLD = _near("-12.0")
_NEAR_GROWS = _near("-11.95")
_NEAR_EXPLODES = _near("-11.9")


def _bscore():
    t = loadTable(str(TRAIN), "y")
    b = contin_bscore(t.otable, t.itable)
    b.set_linear_scaling(True)
    return b


def _rows():
    return RegressionRows.of_bscore(_bscore())


def _t(s):
    return str_to_combo_tree(s)


def _range(rows):
    return max(rows.targets) - min(rows.targets)


# --- the rule (``pole_rejects``) -------------------------------------------------------

# R = 1 and the factor 10, so the box-rel threshold is 10 * max(S(old), 1). All values are
# exact in binary floating point.
_UP_10 = math.nextafter(10.0, INF)
_DOWN_10 = math.nextafter(10.0, 0.0)
_UP_30 = math.nextafter(30.0, INF)
_DOWN_30 = math.nextafter(30.0, 0.0)

# (case, s_old, finite_old, s_new, finite_new, box-rel rejects, box rejects)
RULE_CASES = [
    # S(old) below R: the threshold is 10 R = 10
    ("old<R, new just below 10R", 0.5, True, _DOWN_10, True, False, False),
    ("old<R, new at 10R", 0.5, True, 10.0, True, False, False),
    ("old<R, new just above 10R", 0.5, True, _UP_10, True, True, True),
    ("old<R, new far above 10R", 0.5, True, 1e6, True, True, True),
    ("old<R, new below old", 0.5, True, 0.25, True, False, False),
    # S(old) at R: the threshold is 10 R = 10 S(old)
    ("old=R, new just above 10R", 1.0, True, _UP_10, True, True, True),
    # S(old) above R (but below 10 R): the threshold is 10 S(old) = 30
    ("R<old<10R, new just below 10 old", 3.0, True, _DOWN_30, True, False, True),
    ("R<old<10R, new at 10 old", 3.0, True, 30.0, True, False, True),
    ("R<old<10R, new just above 10 old", 3.0, True, _UP_30, True, True, True),
    ("R<old<10R, new between old and 10R", 3.0, True, 5.0, True, False, False),
    ("R<old<10R, new between 10R and 10 old", 3.0, True, 20.0, True, False, True),
    # S(old) above 10 R: the threshold is 10 S(old) = 300
    ("old>10R, new between old and 10 old", 30.0, True, 200.0, True, False, True),
    ("old>10R, new at 10 old", 30.0, True, 300.0, True, False, True),
    ("old>10R, new just above 10 old", 30.0, True, math.nextafter(300.0, INF), True, True, True),
    ("old>10R, new below old", 30.0, True, 20.0, True, False, False),
    # the non-finite rule
    ("new not finite, old finite", 2.0, True, INF, False, True, True),
    ("new not finite, old finite and large", 1e300, True, INF, False, True, True),
    ("both not finite", INF, False, INF, False, False, False),
    ("old not finite, new finite", INF, False, 1e6, True, False, False),
]  # fmt: skip


@pytest.mark.parametrize(
    "s_old, fin_old, s_new, fin_new, rel, box",
    [c[1:] for c in RULE_CASES],
    ids=[c[0] for c in RULE_CASES],
)
def test_the_rule_on_a_table_of_cases(s_old, fin_old, s_new, fin_new, rel, box):
    assert pole_rejects(s_old, fin_old, s_new, fin_new, 1.0, "box-rel") is rel
    assert pole_rejects(s_old, fin_old, s_new, fin_new, 1.0, "box") is box


def test_the_factor_is_10_and_the_default_mode_is_box():
    assert POLE_FACTOR == 10.0
    assert POLE_CHECK_MODES == ("none", "box", "box-rel")
    # box rejects (20 > 10 R and 20 > S(old)); box-rel accepts (20 <= 10 * 3)
    assert pole_rejects(3.0, True, 20.0, True, 1.0) is True
    assert pole_rejects(3.0, True, 20.0, True, 1.0, "box-rel") is False
    # the factor is a parameter: with 5, the box-rel threshold is 15
    assert pole_rejects(3.0, True, 20.0, True, 1.0, "box-rel", factor=5.0) is True


@pytest.mark.parametrize("mode", ["none", "grid", "", None])
def test_the_rule_refuses_any_other_mode(mode):
    with pytest.raises(ValueError, match="pole check mode"):
        pole_rejects(1.0, True, 2.0, True, 1.0, mode)
    with pytest.raises(ValueError, match="pole check mode"):
        pole_rejects(1.0, True, INF, False, 1.0, mode)


_GRID_S = [0.0, 0.5, 1.0, 3.0, 9.0, 10.0, 11.0, 30.0, 100.0, 1e6, INF]


def _grid():
    for s_old, s_new in itertools.product(_GRID_S, repeat=2):
        for r in (0.0, 1.0, 2.5):
            yield s_old, math.isfinite(s_old), s_new, math.isfinite(s_new), r


def test_box_is_the_round_13_expression_on_a_grid():
    """``pole_rejects(..., "box")`` is the inline expression of ``pole_check`` at dee8101."""
    for so, fo, sn, fn, r in _grid():
        old = (not fn and fo) or (sn > 10.0 * r and sn > so)
        assert pole_rejects(so, fo, sn, fn, r, "box") is old, (so, sn, r)


def test_box_rel_rejects_only_what_box_rejects_on_a_grid():
    """S >= 0, so S(new) > 10 max(S(old), R) gives S(new) > 10 R and S(new) > S(old)."""
    n_differ = 0
    for so, fo, sn, fn, r in _grid():
        rel = pole_rejects(so, fo, sn, fn, r, "box-rel")
        box = pole_rejects(so, fo, sn, fn, r, "box")
        assert box or not rel, (so, sn, r)
        n_differ += box and not rel
    assert n_differ > 0


# --- the check on programs (``pole_check`` with ``mode="box-rel"``) --------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_tuning_that_moves_a_pole_into_the_probe_box_is_rejected(seed):
    """S(old) < R here, so the box-rel threshold is 10 R, as for box."""
    rows = _rows()
    r = _range(rows)
    v = pole_check(_t(_OLD), _t(_POLE), rows, pole_probes(rows, seed), lambda: None,
                   mode="box-rel")  # fmt: skip
    assert v["pole_s_old"] < r
    assert not v["accepted"] and v["pole_s_new"] > 10 * r


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_harmless_tuning_is_accepted(seed):
    rows = _rows()
    v = pole_check(_t(_OLD), _t(_HARMLESS), rows, pole_probes(rows, seed), lambda: None,
                   mode="box-rel")  # fmt: skip
    assert v["accepted"]


def test_a_prediction_that_is_not_finite_rejects_the_tuning():
    rows = _rows()
    probes = [[-3.5, 1.0], [0.0, 1.0]]
    v = pole_check(_t(_OLD), _t(_DOMAIN), rows, probes, lambda: None, mode="box-rel")
    assert not v["accepted"] and v["pole_s_new"] == INF and math.isfinite(v["pole_s_old"])


def test_when_both_programs_have_a_prediction_that_is_not_finite_the_check_accepts():
    rows = _rows()
    probes = [[-3.5, 1.0], [0.0, 1.0]]
    other = "+(sin(*(1.2 $1)) log(+($1 3.05)))"
    v = pole_check(_t(_DOMAIN), _t(other), rows, probes, lambda: None, mode="box-rel")
    assert v["accepted"] and v["pole_s_old"] == v["pole_s_new"] == INF


def test_box_rel_accepts_a_tuning_that_box_rejects():
    """S(old) > 10 R and S(old) < S(tuned) <= 10 S(old): box rejects, box-rel accepts. The
    spreads are the same in both modes (the same probes, the same programs)."""
    rows = _rows()
    r = _range(rows)
    old, new = _t(_NEAR_OLD), _t(_NEAR_GROWS)
    box = pole_check(old, new, rows, _NEAR, lambda: None, mode="box")
    rel = pole_check(old, new, rows, _NEAR, lambda: None, mode="box-rel")
    assert box["pole_s_old"] == rel["pole_s_old"] and box["pole_s_new"] == rel["pole_s_new"]
    s_old, s_new = rel["pole_s_old"], rel["pole_s_new"]
    assert s_old > 10 * r and s_old < s_new <= 10 * s_old
    assert not box["accepted"] and rel["accepted"]


def test_box_rel_rejects_a_spread_above_10_times_the_old_one():
    rows = _rows()
    r = _range(rows)
    v = pole_check(_t(_NEAR_OLD), _t(_NEAR_EXPLODES), rows, _NEAR, lambda: None,
                   mode="box-rel")  # fmt: skip
    assert v["pole_s_old"] > 10 * r and v["pole_s_new"] > 10 * v["pole_s_old"]
    assert not v["accepted"]


@pytest.mark.parametrize("mode", ["box", "box-rel"])
def test_the_check_charges_two_passes_and_stops_at_the_budget(mode):
    rows = _rows()
    probes = pole_probes(rows, 1)
    n = [0]

    def charge():
        n[0] += 1

    pole_check(_t(_OLD), _t(_POLE), rows, probes, charge, mode=mode)
    assert n[0] == 2
    for left in (0, 1):
        used = [0]

        def limited(left=left, used=used):
            if used[0] >= left:
                raise BudgetExhausted("global_budget")
            used[0] += 1

        with pytest.raises(BudgetExhausted):
            pole_check(_t(_OLD), _t(_POLE), rows, probes, limited, mode=mode)
        assert used[0] == left


def test_the_spreads_do_not_depend_on_the_mode():
    rows = _rows()
    probes = pole_probes(rows, 3)
    for old, new in [(_OLD, _POLE), (_OLD, _HARMLESS), (_POLE, _OLD)]:
        a = pole_check(_t(old), _t(new), rows, probes, lambda: None, mode="box")
        b = pole_check(_t(old), _t(new), rows, probes, lambda: None, mode="box-rel")
        assert (a["pole_s_old"], a["pole_s_new"]) == (b["pole_s_old"], b["pole_s_new"])
        assert a["pole_s_old"] == probe_spread(_t(old), rows, probes)[0]


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
    """The part of ``EvaluationOracle`` that ``_tune_best`` uses."""

    def __init__(self, cscorer):
        self.iscorer = SimpleNamespace(_cscorer=cscorer, _reduce=True)
        self.ledger = SimpleNamespace(requests=0, tuning_requests=0)

    def best_genotype(self):
        return (7,)

    def charge_tuning(self):
        self.ledger.requests += 1
        self.ledger.tuning_requests += 1


def _tune(monkeypatch, mode, old_prog, new_prog, probes=None, guard=False, lm_evals=5):
    """``_tune_best`` with an LM that "improves" ``old_prog`` to ``new_prog`` after
    ``lm_evals`` charged passes. ``probes`` (if given) replaces the probes of the run.
    Returns (deme, record, counts, oracle)."""
    cscorer = behave_cscore(_bscore())

    def fake_lm(tree, rows, charge, *a, **k):
        for _ in range(lm_evals):
            charge()
        return LMResult("improved", 2, 1.0, 0.5, evals=lm_evals, tree=_t(new_prog))

    monkeypatch.setattr(gomea_optimizer, "levenberg_marquardt", fake_lm)
    kw = dict(const_opt_guard="val", const_opt_val=str(TRAIN)) if guard else {}
    opt = gomea_optimization(gp=gomea_parameters(
        const_opt="lm", const_opt_pole_check=mode, const_opt_pole_seed=1, **kw))  # fmt: skip
    opt._co = _new_const_opt_counts()
    if probes is not None:
        opt._pole_probes = probes
    deme = _Deme()
    deme.append(scored_instance(instance([7]), cscorer.get_cscore(_t(old_prog))))
    oracle = _Oracle(cscorer)
    opt._tune_best(deme, oracle, _Schema(old_prog))
    return deme, opt._co["lm"], opt._co, oracle


def test_tune_best_box_rel_rejects_a_pole_and_keeps_the_old_program(monkeypatch):
    deme, rec, co, oracle = _tune(monkeypatch, "box-rel", _OLD, _POLE, guard=True)
    assert rec["pole"] == "rejected" and rec["written_back"] is False
    assert "guard" not in rec and "lm_guard_checked" not in co
    assert len(deme) == 1 and getattr(deme[0], "tree", None) is None
    assert co["lm_pole_checked"] == 1 and co["lm_pole_rejected"] == 1
    assert rec["evals"] == oracle.ledger.requests == 5 + 2


def test_tune_best_box_rel_accepts_a_harmless_tuning_then_runs_the_guard(monkeypatch):
    deme, rec, co, oracle = _tune(monkeypatch, "box-rel", _OLD, _HARMLESS, guard=True)
    assert rec["pole"] == "accepted" and rec["guard"] in ("accepted", "rejected")
    assert co["lm_pole_checked"] == 1 and co.get("lm_pole_rejected", 0) == 0
    assert rec["evals"] == 5 + 2 + 2 + (1 if rec["guard"] == "accepted" else 0)


@pytest.mark.parametrize("mode, accepted", [("box", False), ("box-rel", True)])
def test_tune_best_writes_back_the_tuning_that_only_box_rel_accepts(monkeypatch, mode, accepted):
    """The same LM result in both modes: box keeps the old program, box-rel writes back the
    tuned one. The record has the same spreads."""
    deme, rec, co, oracle = _tune(monkeypatch, mode, _NEAR_OLD, _NEAR_GROWS, probes=_NEAR)
    assert rec["pole"] == ("accepted" if accepted else "rejected")
    assert rec["pole_s_old"] == pytest.approx(128.47145915647673)
    assert rec["pole_s_new"] == pytest.approx(259.1887078822298)
    assert co["lm_pole_checked"] == 1 and co.get("lm_pole_rejected", 0) == (0 if accepted else 1)
    assert rec["written_back"] is accepted
    # LM, two probe passes, and the score of the tuned program if it is written back
    assert rec["evals"] == oracle.ledger.requests == 5 + 2 + (1 if accepted else 0)
    if accepted:
        assert len(deme) == 1 and combo_tree_to_str(deme[0].tree).strip() == _NEAR_GROWS
    else:
        assert len(deme) == 1 and getattr(deme[0], "tree", None) is None


def test_tune_best_box_rel_draws_the_probes_of_box_once_per_run(monkeypatch):
    calls = []
    orig = gomea_optimizer.pole_probes

    def spy(rows, seed, *a, **k):
        calls.append(seed)
        return orig(rows, seed, *a, **k)

    monkeypatch.setattr(gomea_optimizer, "pole_probes", spy)
    cscorer = behave_cscore(_bscore())
    opt = gomea_optimization(gp=gomea_parameters(
        const_opt="lm", const_opt_pole_check="box-rel", const_opt_pole_seed=9))  # fmt: skip
    for _ in range(3):
        opt._co = _new_const_opt_counts()
        deme = _Deme()
        deme.append(scored_instance(instance([7]), cscorer.get_cscore(_t(_OLD))))
        opt._tune_best(deme, _Oracle(cscorer), _Schema(_OLD))
    assert calls == [9]
    assert opt._pole_probes == pole_probes(_rows(), 9)


# --- parameters ------------------------------------------------------------------------


def test_parameters_accept_box_rel_and_as_dict_names_it():
    d = gomea_parameters(const_opt="lm", const_opt_pole_check="box-rel", const_opt_pole_seed=4)
    assert d.as_dict()["const_opt_pole_check"] == "box-rel"
    assert d.as_dict()["const_opt_pole_seed"] == 4


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(const_opt_pole_check="box-rel"), "needs const_opt 'lm'"),
        (dict(const_opt_pole_check="box-rel", const_opt="mutate", contin="copy"),
         "needs const_opt 'lm'"),
        (dict(const_opt_pole_check="box-rel", const_opt="lm", const_opt_top_k=2), "top_k"),
        (dict(const_opt_pole_check="box-rel", const_opt="lm", contin="copy",
              const_opt_during="elite"), "const_opt_during"),
        (dict(const_opt_pole_check="box-rel", const_opt="lm", const_opt_pole_seed=1.5),
         "integer"),
        (dict(const_opt_pole_check="box_rel", const_opt="lm"), "must be one of"),
    ],
)  # fmt: skip
def test_parameters_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        gomea_parameters(**kw)


def test_hc_lm_and_the_hybrid_refuse_box_rel():
    gp = gomea_parameters(const_opt="lm", const_opt_pole_check="box-rel")
    with pytest.raises(ValueError, match="not -a hc"):
        hc_lm_optimization(gp=gp)
    with pytest.raises(ValueError, match="hc-gomea"):
        hc_gomea_optimization(gp=gp)


# --- the command line ------------------------------------------------------------------

_REG = ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", "2400", "-c", "10", "-V1", "-l", "WARN"]  # fmt: skip
# The round 12 BASE flags, as in test_const_opt_pole.py and test_pole_check_identity.py.
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
    "--const-opt-val", str(TRAIN),
]  # fmt: skip
_REL = ["--const-opt-pole-check", "box-rel"]


@pytest.mark.parametrize(
    "extra, match",
    [
        (["-a", "gomea", *_REL], "--const-opt-pole-check box-rel needs --const-opt lm"),
        (["-a", "gomea", "--const-opt", "none", *_REL],
         "--const-opt-pole-check box-rel needs --const-opt lm, got --const-opt none"),
        (["-a", "gomea", "--const-opt", "mutate", "--gomea-contin", "copy", *_REL],
         "--const-opt-pole-check box-rel needs --const-opt lm, got --const-opt mutate"),
        (["-a", "gomea", "--const-opt", "polish", "--const-opt-val", str(TRAIN), *_REL],
         "--const-opt-pole-check box-rel needs --const-opt lm, got --const-opt polish"),
        (["-a", "hc", "--const-opt", "lm", *_REL], "needs -a gomea, got -a hc"),
        (["-a", "hc-gomea", "--const-opt", "lm", *_REL], "needs -a gomea, got -a hc-gomea"),
        (["--const-opt", "lm", *_REL], "needs -a gomea, got -a hc"),
        (["-a", "gomea", "--const-opt", "lm", "--const-opt-top-k", "2", *_REL],
         "--const-opt-pole-check box-rel and --const-opt-top-k exclude each other"),
        (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "lm",
          "--const-opt-during", "elite", *_REL],
         "--const-opt-pole-check box-rel and --const-opt-during elite exclude each other"),
        (["-a", "gomea", "--linear-scaling", "terms", "--const-opt", "lm", *_REL],
         "--linear-scaling terms does not support --const-opt-pole-check box-rel"),
        (["-a", "gomea", "--const-opt", "lm", "--const-opt-pole-check", "box_rel"],
         "--const-opt-pole-check box_rel is not recognized. Supported: none, box, box-rel"),
    ],
)  # fmt: skip
def test_cli_refusals(extra, match):
    with pytest.raises(ValueError, match=match):
        moses_exec(["moses", *_REG, "-m", "50", *extra])


def test_the_help_names_box_rel():
    """The help text describes box-rel next to box. The comparison leaves out all white
    space: argparse wraps the text (also at a hyphen)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        moses_exec(["moses", "--help"])
    text = "".join(buf.getvalue().split())
    for want in (
        "counts as one evaluation); box-rel (as box, but the size rule keeps the old program",
        "10 times the larger of that of the old one and the range of the target).",
    ):
        assert "".join(want.split()) in text


def _run(argv, tmp_path, name):
    """One in-process run in ``tmp_path``; returns (stdout, events, log path)."""
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


def _checks(events):
    """The ``const_opt.lm`` records of the "end" events that hold a pole verdict."""
    out = []
    for e in _ends(events):
        lm = e["const_opt"]["lm"]
        if lm is not None and "pole" in lm:
            out.append(lm)
    return out


def test_a_box_rel_run_ends_ok_counts_its_checks_and_stays_in_the_budget(tmp_path):
    """The setting of the box-r1 reference (``test_pole_check_identity.py``) with box-rel.
    The run ends ok, the checks run and are counted, the evaluations never pass -m, and the
    parameters name box-rel. The search before the first check is that of box, so the first
    check has the spreads of box-r1; box rejected that tuning (S(old) < 10 R < S(new) <=
    10 S(old)), box-rel accepts it."""
    m = 3000
    argv = _REG + ["-m", str(m)] + _BASE + _REL + ["--gomea-max-evals-per-deme", "300"]
    out, events, log = _run(argv, tmp_path, "rel")
    ends = _ends(events)
    checked = sum(e["const_opt"]["lm_pole_checked"] for e in ends)
    rejected = sum(e["const_opt"]["lm_pole_rejected"] for e in ends)
    improved = sum(e["const_opt"]["lm_improved"] for e in ends)
    assert 0 < checked <= improved
    assert len(_checks(events)) == checked
    for lm in _checks(events):
        assert lm["pole"] in ("accepted", "rejected")
        assert lm["pole_s_new"] >= 0 and lm["pole_s_old"] >= 0
    assert sum(e["ledger"]["requests"] for e in ends) <= m
    n_evals = int(out.strip().splitlines()[-1].split()[-1])
    assert n_evals <= m
    start = [e for e in events if e["type"] == "invocation_start"]
    assert start and all(e["params"]["const_opt_pole_check"] == "box-rel" for e in start)
    assert all(e["params"]["const_opt_pole_seed"] == 1 for e in start)
    s = runner.const_opt_summary(log, n_evals)
    assert s["lm_pole_checked"] == checked and s["lm_pole_rejected"] == rejected

    ref = [json.loads(x) for x in (REF / "box-r1.events.jsonl").read_text().splitlines()]
    first_box, first_rel = _checks(ref)[0], _checks(events)[0]
    assert first_box["pole"] == "rejected" and first_rel["pole"] == "accepted"
    assert first_rel["pole_s_old"] == first_box["pole_s_old"]
    assert first_rel["pole_s_new"] == first_box["pole_s_new"]
    r = _range(_rows())
    assert first_rel["pole_s_old"] < 10 * r < first_rel["pole_s_new"]
    assert first_rel["pole_s_new"] <= 10 * first_rel["pole_s_old"]
