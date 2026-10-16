"""Tests for pymoses.moses.deme.numeric_simplify (NOT A PORT: the opt-in
`--simplify-exemplar-numeric` pass, no C++ counterpart)."""

from __future__ import annotations

import math

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.reduct.contin_reduction import (
    _reset_cache_for_testing,
    contin_reduction,
)
from pymoses.comboreduct.table.table import ITable, OTable
from pymoses.moses.deme.deme_params import deme_parameters
from pymoses.moses.deme.numeric_simplify import exemplar_numeric_simplifier, subtree_outputs
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import contin_bscore


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_cache_for_testing()
    yield
    _reset_cache_for_testing()


def _s(tr) -> str:
    return combo_tree_to_str(tr).strip()


def _no_reduce(tr) -> None:
    pass


def _problem(rows, ys):
    """(simplifier factory, cscore function) for a small regression table."""
    it = ITable()
    for r in rows:
        it.push_back(tuple(float(v) for v in r))
    ot = OTable([float(y) for y in ys])
    cs = behave_cscore(contin_bscore(ot, it))
    return it, cs.get_cscore_nocache


# x1 varies; x2 is the constant 5 on every row.
_ROWS_CONST_X2 = [(i, 5.0) for i in range(1, 11)]
_Y_LINEAR = [3.0 * i + 0.005 for i in range(1, 11)]


def test_off_by_default():
    assert deme_parameters().numeric_simplifier is None


def test_tol_must_be_positive_and_finite():
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            exemplar_numeric_simplifier([(1.0,)], [1.0], bad)


def test_scale_is_population_std_of_target():
    s = exemplar_numeric_simplifier([(0.0,)] * 4, [1.0, 3.0, 1.0, 3.0], 0.5)
    assert s.scale == 1.0
    assert s.threshold == 0.5


def test_constant_subtree_is_replaced_by_its_mean():
    it, score = _problem(_ROWS_CONST_X2, _Y_LINEAR)
    simp = exemplar_numeric_simplifier(it, _Y_LINEAR, 0.01)
    tr = str_to_combo_tree("+(*(3 $1) *(0.001 $2))")
    out = simp(tr, score, _no_reduce)
    assert out is not tr
    assert _s(out) == "+(*(3 $1) 0.005)"
    # The original exemplar is untouched.
    assert _s(tr) == "+(*(3 $1) *(0.001 $2))"
    assert (simp.calls, simp.fired, simp.candidates, simp.accepted, simp.rejected) == (
        1,
        1,
        1,
        1,
        0,
    )
    assert simp.nodes_removed == tr.size() - out.size() == 2
    assert not score(out) < score(tr)


def test_non_constant_subtrees_are_kept():
    it, score = _problem(_ROWS_CONST_X2, _Y_LINEAR)
    simp = exemplar_numeric_simplifier(it, _Y_LINEAR, 0.01)
    tr = str_to_combo_tree("+(*(3 $1) $1)")
    out = simp(tr, score, _no_reduce)
    assert out is tr
    assert (simp.candidates, simp.fired, simp.nodes_removed) == (0, 0, 0)


def test_near_constant_needs_range_below_threshold():
    # x2 varies by 0.01 in total; y's std is about 8.6.
    rows = [(i, 5.0 + 0.001 * i) for i in range(1, 11)]
    it, score = _problem(rows, _Y_LINEAR)
    sub = str_to_combo_tree("*(1 $2)")
    rng = max(subtree_outputs(sub.begin(), it)) - min(subtree_outputs(sub.begin(), it))
    tight = exemplar_numeric_simplifier(it, _Y_LINEAR, 0.5 * rng / 8.616843969807043)
    loose = exemplar_numeric_simplifier(it, _Y_LINEAR, 2.0 * rng / 8.616843969807043)
    assert tight.near_constant(sub.begin()) is None
    mean = loose.near_constant(sub.begin())
    assert mean == pytest.approx(5.0055)


def test_guard_rejects_a_worse_tree_and_tries_descendants():
    # y = 1000 * (x2 + 1): the child +($2 1) is near-constant on its own
    # (range 0.009 << 0.01 * std(y)), but the parent multiplies it by 1000,
    # so replacing it makes the fit much worse and the guard refuses.
    rows = [(i, 0.001 * i) for i in range(10)]
    ys = [1000.0 * (0.001 * i + 1.0) for i in range(10)]
    it, score = _problem(rows, ys)
    simp = exemplar_numeric_simplifier(it, ys, 0.5)
    tr = str_to_combo_tree("*(1000 +($2 1))")
    out = simp(tr, score, _no_reduce)
    assert out is tr
    assert simp.candidates >= 1
    assert simp.rejected == simp.candidates
    assert simp.accepted == 0 and simp.fired == 0


def test_rejected_parent_does_not_block_a_child():
    # The outer *(1000 ...) term is rejected (its constant loses the fit),
    # but its constant grandchild *(0.5 $3) is still replaced.
    rows = [(i, 0.001 * i, 2.0) for i in range(10)]
    ys = [1000.0 * (0.001 * i + 1.0) for i in range(10)]
    it, score = _problem(rows, ys)
    simp = exemplar_numeric_simplifier(it, ys, 0.5)
    tr = str_to_combo_tree("*(1000 +($2 *(0.5 $3)))")
    out = simp(tr, score, _no_reduce)
    assert _s(out) == "*(1000 +($2 1))"
    assert simp.accepted == 1 and simp.rejected >= 1


def test_simplified_tree_is_reduced():
    it, score = _problem(_ROWS_CONST_X2, _Y_LINEAR)
    simp = exemplar_numeric_simplifier(it, _Y_LINEAR, 0.01)
    reduce = contin_reduction(2, set())
    # *(2 $2) is the constant 10, and the reducer folds 10 + 5 into 15.
    tr = str_to_combo_tree("+(*(3 $1) *(2 $2) 5)")
    out = simp(tr, score, reduce)
    assert _s(out) == "+(15 *(3 $1))"
    assert not score(out) < score(tr)


def test_nonfinite_subtree_is_never_a_candidate():
    rows = [(0.0, 5.0)] * 3
    it, _ = _problem(rows, [1.0, 2.0, 3.0])
    simp = exemplar_numeric_simplifier(it, [1.0, 2.0, 3.0], 1.0)
    assert simp.near_constant(str_to_combo_tree("log($1)").begin()) is None


def test_constant_target_disables_the_pass():
    it, score = _problem(_ROWS_CONST_X2, [2.0] * 10)
    simp = exemplar_numeric_simplifier(it, [2.0] * 10, 1.0)
    tr = str_to_combo_tree("+(*(3 $1) *(0.001 $2))")
    assert simp(tr, score, _no_reduce) is tr
    assert simp.calls == 1 and simp.candidates == 0


# --------------------------------------------------------------- CLI


_CSV = "y,x1,x2\n" + "".join(f"{3.0 * i + 0.005},{i},5\n" for i in range(1, 21))


def _run(argv):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


def test_cli_rejects_negative_tol(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    with pytest.raises(ValueError, match="simplify-exemplar-numeric"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--simplify-exemplar-numeric", "-1"])


def test_cli_rejects_boolean_problem():
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", "-Hpa", "-k3", "-m50", "--simplify-exemplar-numeric", "0.1"])


def test_cli_rejects_boolean_table(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("y,x1\n0,0\n1,1\n")
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--simplify-exemplar-numeric", "0.1"])


def test_cli_regression_run_with_and_without(tmp_path, caplog):
    p = tmp_path / "t.csv"
    p.write_text(_CSV)
    base = ["moses", f"-i{p}", "-uy", "-m300", "-c3", "-r1"]
    rc0, out0 = _run(base)
    rc0b, out0b = _run([*base, "--simplify-exemplar-numeric", "0"])
    assert rc0 == rc0b == 0 and out0 == out0b
    log = tmp_path / "m.log"
    rc1, out1 = _run([*base, "--simplify-exemplar-numeric", "0.01", "-lINFO", f"-f{log}"])
    assert rc1 == 0 and out1
    assert "Numeric exemplar simplification:" in log.read_text()
