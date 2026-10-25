"""`--linear-scaling 1` (pyMoses extension, not in classic MOSES).

With the option on, `contin_bscore` scores a program f on a + b*f, where
(a, b) is the least-squares fit of f's predictions to the training target
(`bscores.linear_scaling`, the formula `tools/gomea/sr_eval.py` applies when it
reports NMSE). Off (the default) the scorer is classic's, unchanged.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import random
import sys
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.table.table import ITable, OTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.moses.types import _to_score_t
from pymoses.moses.scoring.bscores import contin_bscore, linear_scaling

_REPO = Path(__file__).resolve().parents[3]
_SAMPLE_CSV = _REPO / "tests" / "fixtures" / "table_data" / "contin_regression_sample.csv"


def _sr_eval():
    spec = importlib.util.spec_from_file_location(
        "sr_eval_for_test", _REPO / "tools" / "gomea" / "sr_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def _xs():
    return [0.5, 1.0, 1.5, 2.25, -0.75, 3.0, 4.5, -2.0]


def _scorer(ys, xs=None):
    it = ITable()
    for x in xs if xs is not None else _xs():
        it.push_back((x,))
    return contin_bscore(OTable(list(ys)), it)


# ------------------------------------------------------------ the formula


def test_linear_scaling_recovers_known_a_and_b():
    xs = _xs()
    ys = [3.0 + 2.0 * x for x in xs]
    a, b = linear_scaling(ys, xs)
    assert a == pytest.approx(3.0, abs=1e-12)
    assert b == pytest.approx(2.0, abs=1e-12)
    a, b = linear_scaling([-1.5 - 0.25 * x for x in xs], xs)
    assert (a, b) == pytest.approx((-1.5, -0.25), abs=1e-12)


def test_linear_scaling_constant_prediction_gives_mean():
    ys = [1.0, 2.0, 6.0]
    assert linear_scaling(ys, [4.0, 4.0, 4.0]) == (3.0, 0.0)


def test_linear_scaling_nonfinite_prediction_is_left_unscaled():
    assert linear_scaling([1.0, 2.0], [1.0, math.nan]) == (0.0, 1.0)
    assert linear_scaling([1.0, 2.0], [math.inf, 1.0]) == (0.0, 1.0)


def test_linear_scaling_matches_sr_eval():
    sr_eval = _sr_eval()
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(2, 40)
        p = [rng.uniform(-10, 10) for _ in range(n)]
        y = [rng.uniform(-5, 5) + 0.3 * x for x in p]
        assert linear_scaling(y, p) == sr_eval.linear_scaling(y, p)
    assert linear_scaling([1.0, 5.0], [2.0, 2.0]) == sr_eval.linear_scaling([1.0, 5.0], [2.0, 2.0])


# ------------------------------------------------------------ the scorer


def test_off_by_default_and_unchanged():
    table = loadTable(str(_SAMPLE_CSV), "y")
    progs = ["$1", "*(2 $1)", "+($1 $2)", "0.5", "*($1 $1 $2)"]
    fresh = contin_bscore(table)
    toggled = contin_bscore(table)
    toggled.set_linear_scaling(True)
    toggled.set_linear_scaling(False)
    for p in progs:
        tr = str_to_combo_tree(p)
        assert fresh(tr) == toggled(tr)
    # the classic definition: -(f(x) - y)^2 per row, narrowed to score_t
    want = [
        -_to_score_t((float(row[0]) - float(t)) ** 2)
        for row, t in zip(fresh._cti, fresh._target, strict=True)
    ]
    assert fresh(str_to_combo_tree("$1")) == want


def test_scaled_score_of_an_affine_target_is_zero():
    xs = _xs()
    bsc = _scorer([3.0 + 2.0 * x for x in xs], xs)
    tr = str_to_combo_tree("$1")
    raw = bsc(tr)
    assert sum(raw) < -1.0
    bsc.set_linear_scaling(True)
    scaled = bsc(tr)
    assert len(scaled) == len(raw)
    assert all(abs(v) < 1e-9 for v in scaled)


def test_scaled_score_is_per_row_error_of_a_plus_b_f():
    xs = _xs()
    ys = [math.sin(x) + 0.1 * x * x for x in xs]
    bsc = _scorer(ys, xs)
    bsc.set_linear_scaling(True)
    tr = str_to_combo_tree("*($1 $1)")
    p = [x * x for x in xs]
    a, b = linear_scaling(ys, p)
    want = [-_to_score_t((a + b * f - y) ** 2) for f, y in zip(p, ys, strict=True)]
    assert bsc(tr) == want
    # never worse than unscaled: (a, b) = (0, 1) is one of the fits considered
    bsc.set_linear_scaling(False)
    assert sum(bsc.__call__(tr)) <= sum(want) + 1e-6


def test_constant_program_scores_the_target_mean():
    xs = _xs()
    ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    bsc = _scorer(ys, xs)
    bsc.set_linear_scaling(True)
    m = sum(ys) / len(ys)
    want = [-_to_score_t((m - y) ** 2) for y in ys]
    assert bsc(str_to_combo_tree("42")) == want
    assert bsc(str_to_combo_tree("-3.5")) == want


def test_best_possible_and_min_improv_unchanged():
    bsc = _scorer([1.0] * 8)
    before = (bsc.best_possible_bscore(), bsc.min_improv())
    bsc.set_linear_scaling(True)
    assert (bsc.best_possible_bscore(), bsc.min_improv()) == before


# ------------------------------------------------------------ the CLI


_CSV = "y,x1,x2\n" + "".join(
    f"{3.0 * i + 7.0 + 0.01 * (i % 3)},{i},{i % 4}\n" for i in range(1, 21)
)


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


@pytest.mark.parametrize("algo", [[], ["-a", "hc-capped", "--gomea-max-evals-per-deme", "50"]])
def test_cli_flag_zero_is_the_default(tmp_path, algo):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    base = ["moses", f"-i{p}", "-uy", "-m200", "-c3", "-r1", *algo]
    rc0, out0 = _run(base)
    rc1, out1 = _run([*base, "--linear-scaling", "0"])
    assert rc0 == rc1 == 0 and out0 == out1


@pytest.mark.parametrize(
    "algo",
    [
        ["-a", "hc"],
        ["-a", "hc-capped", "--gomea-max-evals-per-deme", "50"],
        ["-a", "gomea", "--gomea-max-evals-per-deme", "50", "--gomea-contin", "copy"],
    ],
)
def test_cli_flag_on_runs_and_changes_scores(tmp_path, algo):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    base = ["moses", f"-i{p}", "-uy", "-m200", "-c3", "-r1", *algo]
    rc0, out0 = _run(base)
    rc1, out1 = _run([*base, "--linear-scaling", "1"])
    assert rc0 == rc1 == 0 and out1.strip()
    assert out0 != out1


def test_cli_rejects_bad_value(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    with pytest.raises(ValueError, match="--linear-scaling"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--linear-scaling", "2"])


def test_cli_rejects_boolean_problem():
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", "-Hpa", "-k3", "-m50", "--linear-scaling", "1"])


def test_cli_rejects_boolean_table(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("y,x1\n0,0\n1,1\n")
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--linear-scaling", "1"])


def test_cli_rejects_abs_error(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    with pytest.raises(ValueError, match="it-abs-err"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--it-abs-err", "1", "--linear-scaling", "1"])
