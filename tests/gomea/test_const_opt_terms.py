"""Server round 13, T3.1 (arm TLS-LM): LM with ``--linear-scaling terms``.

LM at the end of a deme tunes the constants inside the terms. At each pass the term weights
are fitted again by least squares, by the scorer's rule (``bscores.fit_program_terms``). A
program with one term, or whose per-term fit falls back, gets the plain ``a + b f``
residuals. The validation guard applies the weights of the training rows to the validation
rows. See ``docs/gomea/SERVER-ROUND13-PREREGISTRATION.md`` ("TLS-LM") and
``docs/gomea/T3.1-FEASIBILITY.md``.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.crutil.exception import OverflowException
from pymoses.comboreduct.table.table import ITable, OTable
from pymoses.gomea import optimizer as gomea_optimizer
from pymoses.gomea.const_opt import (
    RegressionRows,
    constant_nodes,
    guard_accepts,
    levenberg_marquardt,
    mse,
    validation_nmse,
)
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.moses.types import _to_score_t
from pymoses.moses.scoring.bscores import FALLBACK_ERROR, contin_bscore

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import sr_eval  # noqa: E402

TRAIN = Path(__file__).resolve().parent / "data" / "const_opt_none" / "train.csv"


def _rows(n: int = 20, shift: float = 0.0):
    return [(0.3 * i - 2.0 + shift, float(i % 5) - 1.5) for i in range(n)]


def _target(rows):
    """y = 1 + 2 sin(1.3 x1) - 0.5 x2^2: an exact sum of terms of ``_TWO``, with the inner
    constant 1.3."""
    return [1.0 + 2.0 * math.sin(1.3 * a) - 0.5 * b * b for a, b in rows]


def _scorer(rows, ys, mode: str):
    it = ITable()
    for r in rows:
        it.push_back(tuple(r))
    b = contin_bscore(OTable(list(ys)), it)
    if mode == "terms":
        b.set_linear_scaling_terms(True)
    elif mode == "1":
        b.set_linear_scaling(True)
    return b


def _terms_rows(rows=None):
    rows = rows if rows is not None else _rows()
    b = _scorer(rows, _target(rows), "terms")
    return b, RegressionRows.of_bscore(b)


# Two terms with an inner constant; the same with a constant that a term weight absorbs
# (3 * t); one term; and three fallbacks: a numeric leaf term, collinear terms, and a term
# that cannot be evaluated (log of a negative number is an evaluation error).
_TWO = "+(sin(*(0.8 $1)) *($2 $2))"
_ABSORBED = "+(*(3 sin(*(0.8 $1))) *($2 $2))"
_ONE = "sin(*(0.8 $1))"
_FALLBACKS = ("+(sin(*(0.8 $1)) 0.25)", "+($1 *(2 $1))", "+(sin($1) log($1))")


def test_of_bscore_reads_the_terms_state_of_the_scorer():
    """``set_linear_scaling_terms`` turns ``_linear_scaling`` off. ``of_bscore`` still builds
    rows with the per-term rule, and with the scaled plain path for the fallbacks."""
    rows = _rows()
    b, r = _terms_rows(rows)
    assert b._linear_scaling is False
    assert r.terms is True and r.scaled is True
    one = RegressionRows.of_bscore(_scorer(rows, _target(rows), "1"))
    assert one.terms is False and one.scaled is True
    plain = RegressionRows.of_bscore(_scorer(rows, _target(rows), "0"))
    assert plain.terms is False and plain.scaled is False


@pytest.mark.parametrize("prog", [_TWO, _ABSORBED, _ONE, *_FALLBACKS[:2]])
def test_the_lm_training_error_is_the_scorer_terms_error(prog):
    """For the same tree and rows, each squared residual of the LM rows is the scorer's
    per-row error (the scorer narrows each one to ``score_t``), and the LM training MSE is
    the scorer's MSE."""
    b, r = _terms_rows()
    tr = str_to_combo_tree(prog)
    res, _w = r.residuals(tr)
    bs = b(tr)
    assert [-_to_score_t(x * x) for x in res] == bs
    assert mse(res) == pytest.approx(-math.fsum(bs) / len(bs), rel=1e-6)


def test_a_program_that_cannot_be_evaluated_is_the_worst_for_both():
    """The scorer raises (its caller gives the worst score); the LM rows give infinite
    residuals (LM then reports ``failed``)."""
    b, r = _terms_rows()
    tr = str_to_combo_tree(_FALLBACKS[2])
    with pytest.raises(OverflowException):
        b(tr)
    res, _ab = r.residuals(tr)
    assert mse(res) == math.inf
    with_constant = str_to_combo_tree("+(sin(*(0.8 $1)) log($1))")
    assert levenberg_marquardt(with_constant, r, lambda: None).status == "failed"


def test_lm_in_terms_mode_lowers_the_training_mse_of_a_two_term_program():
    """LM tunes the inner constant (0.8 -> 1.3); the weights come from the least-squares
    fit at each pass. The tuned tree has the scorer's error too."""
    b, r = _terms_rows()
    tr = str_to_combo_tree(_TWO)
    res = levenberg_marquardt(tr, r, lambda: None)
    assert res.status == "improved"
    assert res.mse0 > 1.0 and res.mse < 1e-20
    assert res.values == [pytest.approx(1.3, abs=1e-9)]
    bs = b(res.tree)
    assert res.mse == pytest.approx(-math.fsum(bs) / len(bs), rel=1e-6, abs=1e-12)
    assert combo_tree_to_str(tr).strip() == _TWO  # the input tree is not changed


def test_a_constant_that_a_term_weight_absorbs_stays_fixed():
    """In ``*(3 sin(...))`` the 3 only scales the term; the term weight absorbs it, so its
    Jacobian column is zero and the existing inert rule holds it fixed (``n_inert``)."""
    _b, r = _terms_rows()
    res = levenberg_marquardt(str_to_combo_tree(_ABSORBED), r, lambda: None)
    assert res.status == "improved" and res.n_inert == 1
    assert res.values[0] == 3.0
    assert res.values[1] == pytest.approx(1.3, abs=1e-9)
    assert [float(n.data) for n in constant_nodes(res.tree)][0] == 3.0


@pytest.mark.parametrize("prog", [_ONE, *_FALLBACKS])
def test_one_term_and_fallback_programs_get_the_plain_residuals(prog):
    """The residuals of the rows with ``terms`` equal those of plain linear scaling."""
    rows = _rows()
    _b, r = _terms_rows(rows)
    plain = RegressionRows(rows, _target(rows), True)
    tr = str_to_combo_tree(prog)
    k, fit = r.term_fit(tr)
    assert fit is None or fit.weights is None
    if prog == _FALLBACKS[2]:
        assert fit.reason == FALLBACK_ERROR
    assert r.residuals(tr) == plain.residuals(tr)


def test_explicit_scaling_bypasses_the_terms_fit():
    """``residuals(tree, (a, b))`` (the validation rows of the guard) uses ``(a, b)``."""
    _b, r = _terms_rows()
    tr = str_to_combo_tree(_TWO)
    p = r.predictions(tr)
    res, ab = r.residuals(tr, (0.5, 2.0))
    assert ab == (0.5, 2.0)
    assert res == [0.5 + 2.0 * x - y for x, y in zip(p, r.targets, strict=True)]


def _sr_table(rows):
    return sr_eval.Table(["x1", "x2"], [list(x) for x in rows], _target(rows))


@pytest.mark.parametrize("prog", [_TWO, _ABSORBED, _ONE, *_FALLBACKS[:2]])
def test_the_guard_applies_the_training_weights_to_the_validation_rows(prog):
    """``validation_nmse`` in terms mode is the report's NMSE (``sr_eval.evaluate(...,
    terms=True)``, which also fits the weights on the training rows), divided by 100."""
    train_rows, val_rows = _rows(), _rows(13, shift=0.17)
    _b, train = _terms_rows(train_rows)
    val = RegressionRows(val_rows, _target(val_rows), True)
    v = validation_nmse(str_to_combo_tree(prog), train, val)
    rep = sr_eval.evaluate(prog, _sr_table(train_rows), _sr_table(val_rows), terms=True)
    assert 100.0 * v == pytest.approx(rep["test"]["nmse_scaled"], rel=1e-12, abs=1e-24)


def test_the_guard_in_terms_mode_differs_from_plain_scaling_for_two_terms():
    train_rows, val_rows = _rows(), _rows(13, shift=0.17)
    _b, train = _terms_rows(train_rows)
    plain = RegressionRows(train_rows, _target(train_rows), True)
    val = RegressionRows(val_rows, _target(val_rows), True)
    tr = str_to_combo_tree(_TWO)
    assert validation_nmse(tr, train, val) < validation_nmse(tr, plain, val)
    one = str_to_combo_tree(_ONE)
    assert validation_nmse(one, train, val) == validation_nmse(one, plain, val)


def test_a_validation_row_where_a_term_fails_makes_the_nmse_infinite():
    """``log(x2 + 3)`` is defined on every training row (x2 >= -1.5), not on a validation
    row with x2 = -4."""
    train_rows = _rows()
    _b, train = _terms_rows(train_rows)
    val_rows = [(0.1, 0.5), (0.2, -4.0), (0.3, 1.0)]
    val = RegressionRows(val_rows, _target(val_rows), True)
    tr = str_to_combo_tree("+(sin($1) log(+($2 3)))")
    k, fit = train.term_fit(tr)
    assert k == 2 and fit.weights is not None
    assert validation_nmse(tr, train, val) == math.inf


def test_guard_accepts_in_terms_mode_charges_two_passes():
    train_rows, val_rows = _rows(), _rows(13, shift=0.17)
    _b, train = _terms_rows(train_rows)
    val = RegressionRows(val_rows, _target(val_rows), True)
    old = str_to_combo_tree(_TWO)
    new = levenberg_marquardt(old, train, lambda: None).tree
    n = [0]

    def charge():
        n[0] += 1

    v = guard_accepts(old, new, train, val, charge)
    assert n[0] == 2
    assert v["accepted"] and v["val_nmse_new"] < 1e-20 < v["val_nmse_old"]
    back = guard_accepts(new, old, train, val, charge)
    assert not back["accepted"] and n[0] == 4


# --- the search: the tree that LM tunes is the tree that the scorer splits ---------------

_REG = ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", "2400", "-c", "10", "-V1", "-l", "WARN"]  # fmt: skip
_SEARCH = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced",
]  # fmt: skip
_TAIL = ["--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
_TLS_LM = (_SEARCH + ["--linear-scaling", "terms", "--const-opt", "lm", "--const-opt-guard",
                      "val"] + _TAIL + ["--const-opt-val", str(TRAIN)])  # fmt: skip


def _run(argv, tmp_path, spy=None):
    """One in-process run; returns (stdout, events). ``spy`` collects each LM call of
    ``_tune_best``: (tree, rows, LM result, scorer)."""
    log = tmp_path / "events.jsonl"
    log.unlink(missing_ok=True)
    patches = contextlib.ExitStack()
    if spy is not None:
        mp = patches.enter_context(pytest.MonkeyPatch.context())
        seen = {}
        orig_find, orig_lm = gomea_optimizer.find_contin_bscore, gomea_optimizer.levenberg_marquardt

        def find(cscorer):
            seen["b"] = orig_find(cscorer)
            return seen["b"]

        def lm(tree, rows, charge, *a, **k):
            res = orig_lm(tree, rows, charge, *a, **k)
            spy.append((tree, rows, res, seen["b"]))
            return res

        mp.setattr(gomea_optimizer, "find_contin_bscore", find)
        mp.setattr(gomea_optimizer, "levenberg_marquardt", lm)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        with patches:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    events = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return buf.getvalue(), events


def _check_lm_calls(spy):
    """Each LM call got the per-term rows, and its first training MSE is the MSE of the
    scorer's terms bscore of the same tree."""
    assert spy, "LM was never called"
    for tree, rows, res, b in spy:
        assert rows.terms is True and b._linear_scaling_terms is True
        if math.isfinite(res.mse0):
            bs = b(tree)
            assert res.mse0 == pytest.approx(-math.fsum(bs) / len(bs), rel=1e-6)


def test_tls_lm_tunes_the_tree_that_the_scorer_splits_into_terms(tmp_path):
    """In a GOMEA run, LM at the end of each deme gets the reduced tree of the deme's best
    genotype: its first training MSE times the rows is minus the raw score of that best
    (``end`` event), so the scorer and LM split the same tree by the same rule."""
    spy: list = []
    cap = ["--gomea-max-evals-per-deme", "300"]  # more demes, so more LM calls
    _out, events = _run(_REG + ["-m", "3000"] + _TLS_LM + cap, tmp_path, spy)
    _check_lm_calls(spy)
    n = sum(1 for _ in TRAIN.open()) - 1
    ends = [e for e in events if e["type"] == "end"]
    checked = 0
    for e in ends:
        lm = e["const_opt"]["lm"]
        if lm is not None and math.isfinite(lm["mse0"]):
            assert lm["mse0"] * n == pytest.approx(-e["best"]["raw"], rel=1e-5)
            checked += 1
    assert checked >= 1
    assert any(e["const_opt"]["lm_improved"] for e in ends), "no LM call improved"
    assert any(e["const_opt"]["lm"].get("guard") == "accepted" for e in ends)
    assert all(e["const_opt"]["guard"] == "val" for e in ends)
    assert all(e["params"]["const_opt"] == "lm" for e in events if e["type"] == "invocation_start")


@pytest.mark.parametrize(
    "algo",
    [
        ["-a", "hc", "--knob-probe-baseline", "reduced", "--const-opt-guard", "val",
         "--const-opt-val", str(TRAIN)],
        ["-a", "hc-gomea", "--gomea-pop-size", "16", "--gomea-contin", "copy",
         "--gomea-max-evals-per-deme", "400"],
        _SEARCH + _TAIL + ["--const-opt-reserve", "200"],
    ],
    ids=["hc", "hc-gomea", "gomea-reserve"],
)  # fmt: skip
def test_tls_lm_with_hc_hc_gomea_and_the_reserve(algo, tmp_path):
    """``-a hc`` (HC-LM borrows ``_tune_best``), ``-a hc-gomea`` and ``--const-opt-reserve``
    need no code of their own: LM gets the per-term rows and the scorer's error."""
    spy: list = []
    argv = _REG + ["-m", "1500", "--linear-scaling", "terms", "--const-opt", "lm"] + algo
    out, events = _run(argv, tmp_path, spy)
    assert out.strip()
    _check_lm_calls(spy)
    ends = [e for e in events if e.get("type") == "end"]
    assert sum(e["const_opt"]["lm_calls"] for e in ends) == len(spy)


# --- the command line ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, match",
    [
        (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "mutate"],
         "--linear-scaling terms needs --const-opt none or lm"),
        (["-a", "gomea", "--const-opt", "polish", "--const-opt-val", str(TRAIN)],
         "--linear-scaling terms needs --const-opt none or lm"),
        (["-a", "gomea", "--const-opt", "lm", "--const-opt-top-k", "3"],
         "--linear-scaling terms does not support --const-opt-top-k"),
        (["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "lm",
          "--const-opt-during", "elite"],
         "--linear-scaling terms does not support --const-opt-during elite"),
    ],
)  # fmt: skip
def test_cli_refusals_with_terms(extra, match):
    with pytest.raises(ValueError, match=match):
        moses_exec(["moses", *_REG, "-m", "50", "--linear-scaling", "terms", *extra])


def test_cli_accepts_terms_with_lm_and_with_none(tmp_path):
    """``--const-opt none`` (BASE-TLS) and ``--const-opt lm`` (TLS-LM) both run."""
    out, _ = _run(_REG + ["-m", "300"] + _SEARCH + ["--linear-scaling", "terms",
                                                    "--const-opt", "none"] + _TAIL, tmp_path)
    assert out.strip()
    out, events = _run(_REG + ["-m", "300"] + _TLS_LM, tmp_path)
    assert out.strip() and any(e["type"] == "end" for e in events)
