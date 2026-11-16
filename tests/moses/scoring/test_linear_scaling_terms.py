"""`--linear-scaling terms` (server round 12, T2.2; pyMoses extension, not in classic MOSES).

With the option on, `contin_bscore` splits a program at a top-level `+` into terms (else one
term) and scores w_0 + sum w_i t_i, the least-squares fit of the term outputs to the
training target (`least_squares.fit_terms`). A program with one term, or whose fit falls
back, gets exactly the `--linear-scaling 1` score.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.comboreduct.crutil.exception import OverflowException
from pymoses.comboreduct.table.table import ITable, OTable
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.moses.types import _to_score_t
from pymoses.moses.scoring import least_squares as ls
from pymoses.moses.scoring.bscores import (
    FALLBACK_ERROR,
    TERMS_STATS_KEYS,
    contin_bscore,
    top_level_terms,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAIN = _REPO / "tests" / "gomea" / "data" / "const_opt_none" / "train.csv"


def _rows():
    return [(0.3 * i - 2.0, float(i % 5) - 1.5) for i in range(20)]


def _scorer(ys, rows=None):
    it = ITable()
    for r in rows if rows is not None else _rows():
        it.push_back(tuple(r))
    return contin_bscore(OTable(list(ys)), it)


def _target(rows=None):
    """y = 1 + 2 sin(x1) - 0.5 x2^2, an exact combination of the terms of `_SUM`."""
    return [1.0 + 2.0 * math.sin(a) - 0.5 * b * b for a, b in (rows or _rows())]


_SUM = "+(sin($1) *($2 $2))"


def _plain(ys, prog, rows=None):
    """The `--linear-scaling 1` bscore of `prog`."""
    b = _scorer(ys, rows)
    b.set_linear_scaling(True)
    return b(str_to_combo_tree(prog))


def _terms(ys, rows=None):
    b = _scorer(ys, rows)
    b.set_linear_scaling_terms(True)
    return b


# ------------------------------------------------------------ the term split


@pytest.mark.parametrize(
    ("prog", "terms"),
    [
        ("+($1 $2)", ["$1", "$2"]),
        ("+(sin($1) *($2 $2) 0.5)", ["sin($1)", "*($2 $2)", "0.5"]),
        ("+(+($1 $2) $1)", ["+($1 $2)", "$1"]),  # only the top level
        ("*(2 +($1 $2))", ["*(2 +($1 $2))"]),  # the root is not +
        ("$1", ["$1"]),
        ("6", ["6"]),  # a contin leaf equal to the ordinal of + is not +
    ],
)
def test_top_level_terms(prog, terms):
    tr = str_to_combo_tree(prog)
    got = [combo_tree_to_str(combo_tree(n)) for n in top_level_terms(tr)]
    assert got == [combo_tree_to_str(str_to_combo_tree(t)) for t in terms]


# ------------------------------------------------------------ the scorer


def test_off_by_default_and_after_turning_it_off():
    ys = _target()
    fresh = _scorer(ys)
    toggled = _scorer(ys)
    toggled.set_linear_scaling_terms(True)
    toggled.set_linear_scaling_terms(False)
    for p in ["$1", _SUM, "+($1 *(2 $1))", "0.5"]:
        tr = str_to_combo_tree(p)
        assert fresh(tr) == toggled(tr)
    assert toggled._terms_stats is None


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_sum_of_k_terms_gets_k_weights_and_an_intercept(k):
    terms = ["sin($1)", "*($2 $2)", "$1", "*($1 $2)"][:k]
    prog = f"+({' '.join(terms)})"
    b = _terms([math.cos(a) * b for a, b in _rows()])
    n, fit = b.term_fit(str_to_combo_tree(prog))
    assert n == k
    assert fit.reason is None and len(fit.weights) == k + 1


def test_a_known_combination_of_terms_scores_zero():
    ys = _target()
    b = _terms(ys)
    tr = str_to_combo_tree(_SUM)
    n, fit = b.term_fit(tr)
    assert n == 2 and fit.weights == pytest.approx([1.0, 2.0, -0.5], abs=1e-9)
    bs = b(tr)
    assert len(bs) == len(ys)
    assert all(abs(v) < 1e-12 for v in bs)  # NMSE about 0
    # the single a + b*f fit cannot do that
    assert sum(_plain(ys, _SUM)) < -1.0


def test_the_score_is_the_per_row_error_of_the_fitted_model():
    rows = _rows()
    ys = [math.exp(0.1 * a) + 0.2 * b for a, b in rows]
    b = _terms(ys, rows)
    tr = str_to_combo_tree(_SUM)
    _, fit = b.term_fit(tr)
    want = [-_to_score_t((f - y) ** 2) for f, y in zip(fit.fitted, ys, strict=True)]
    assert b(tr) == want
    # never worse than a + b*f, a special case of the per-term model
    assert sum(b(tr)) >= sum(_plain(ys, _SUM, rows)) - 1e-6


@pytest.mark.parametrize(
    ("prog", "reason"),
    [
        ("+($1 *(2 $1))", ls.FALLBACK_SINGULAR),  # collinear terms
        ("+($2 *(-3 $2) sin($1))", ls.FALLBACK_SINGULAR),
        ("+(0.25 sin($1) $2)", ls.FALLBACK_CONSTANT),  # a numeric leaf
        ("+(sin(3) $1)", ls.FALLBACK_CONSTANT),  # constant on the rows
    ],
)
def test_a_fallback_gets_the_plain_linear_scaling_score(prog, reason):
    ys = _target()
    b = _terms(ys)
    tr = str_to_combo_tree(prog)
    n, fit = b.term_fit(tr)
    assert n >= 2 and fit.weights is None and fit.reason == reason
    assert b(tr) == _plain(ys, prog)


def test_a_term_output_that_is_not_finite_falls_back():
    rows = _rows()
    rows[3] = (1e200, 0.5)  # *($1 $1) is inf on this row (no overflow check in *)
    ys = _target(rows)
    prog = "+(*($1 $1) $2)"
    b = _terms(ys, rows)
    _, fit = b.term_fit(str_to_combo_tree(prog))
    assert fit.reason == ls.FALLBACK_NONFINITE
    got, want = b(str_to_combo_tree(prog)), _plain(ys, prog, rows)
    assert str(got) == str(want)  # the same, -inf (and nan) included


def test_a_term_that_cannot_be_evaluated_fails_as_the_plain_path_does():
    ys = _target()
    prog = "+(log($1) $2)"  # log of a negative x1 raises OverflowException
    b = _terms(ys)
    _, fit = b.term_fit(str_to_combo_tree(prog))
    assert fit.reason == FALLBACK_ERROR
    with pytest.raises(OverflowException):
        _plain(ys, prog)
    with pytest.raises(OverflowException):
        b(str_to_combo_tree(prog))


@pytest.mark.parametrize("prog", ["*(2 +($1 $2))", "$1", "sin($1)", "42", "+($1)"])
def test_one_term_gets_the_plain_linear_scaling_score(prog):
    ys = _target()
    b = _terms(ys)
    n, fit = b.term_fit(str_to_combo_tree(prog))
    assert n == 1 and fit is None
    assert b(str_to_combo_tree(prog)) == _plain(ys, prog)


def test_the_score_is_a_pure_function_of_the_program():
    ys = _target()
    b = _terms(ys)
    progs = [_SUM, "+($1 *(2 $1))", "$1", _SUM, "+(0.25 sin($1) $2)", _SUM]
    first = {}
    for p in progs:
        bs = b(str_to_combo_tree(p))
        assert first.setdefault(p, bs) == bs
    # inner multiplicative constants do not change the score of a fitted program
    assert b(str_to_combo_tree("+(sin($1) *(-7 $2 $2))")) == pytest.approx(
        first[_SUM], abs=1e-9
    )


def test_the_counters():
    ys = _target()
    b = _terms(ys)
    for p in [_SUM, _SUM, "$1", "+($1 *(2 $1))", "+(0.25 sin($1) $2)", "+(log($1) $2)"]:
        try:
            b(str_to_combo_tree(p))
        except OverflowException:
            pass
    st = b._terms_stats
    assert set(st) == set(TERMS_STATS_KEYS)
    assert st == {
        "calls": 6,
        "terms": 2 + 2 + 1 + 2 + 3 + 2,
        "one_term": 1,
        "fitted": 2,
        "fallback_constant": 1,
        "fallback_singular": 1,
        "fallback_nonfinite": 0,
        "fallback_error": 1,
    }


def test_terms_and_plain_scaling_exclude_each_other():
    b = _scorer(_target())
    b.set_linear_scaling(True)
    b.set_linear_scaling_terms(True)
    assert b._linear_scaling is False and b._linear_scaling_terms is True


def test_best_possible_and_min_improv_unchanged():
    b = _scorer(_target())
    before = (b.best_possible_bscore(), b.min_improv())
    b.set_linear_scaling_terms(True)
    assert (b.best_possible_bscore(), b.min_improv()) == before


# ------------------------------------------------------------ the CLI


def _run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


_REG = ["moses", "-i", str(_TRAIN), "-uy", "-r1", "-n", "sin", "-n", "exp", "-n", "log",
        "-v", "2400", "-c3", "-m400", "--knob-probe-baseline", "reduced"]  # fmt: skip


@pytest.mark.parametrize(
    "algo",
    [
        ["-a", "hc"],
        ["-a", "gomea", "--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05",
         "--gomea-contin", "copy", "--gomea-deme-store", "accepted"],
    ],
)  # fmt: skip
def test_cli_terms_runs_and_changes_scores(algo):
    rc1, out1 = _run([*_REG, *algo, "--linear-scaling", "1"])
    rc2, out2 = _run([*_REG, *algo, "--linear-scaling", "terms", "--const-opt", "none"])
    assert rc1 == rc2 == 0 and out2.strip()
    assert out1 != out2


@pytest.mark.parametrize(
    "co",
    [
        ["--const-opt", "lm"],
        ["-a", "gomea", "--const-opt", "lm"],
        ["-a", "gomea", "--const-opt", "lm", "--const-opt-guard", "val",
         "--const-opt-val", str(_TRAIN)],
        ["-a", "gomea", "--gomea-contin", "copy", "--const-opt", "mutate"],
        ["-a", "gomea", "--const-opt", "polish", "--const-opt-val", str(_TRAIN)],
    ],
)  # fmt: skip
def test_cli_refuses_constant_tuning(co):
    with pytest.raises(ValueError, match="--linear-scaling terms needs --const-opt none"):
        _run([*_REG, "--linear-scaling", "terms", *co])


def test_cli_refuses_a_boolean_problem():
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", "-Hpa", "-k3", "-m50", "--linear-scaling", "terms"])


def test_cli_refuses_a_boolean_table(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("y,x1\n0,0\n1,1\n")
    with pytest.raises(ValueError, match="regression only"):
        _run(["moses", f"-i{p}", "-uy", "-m50", "--linear-scaling", "terms"])


def test_cli_refuses_abs_error():
    with pytest.raises(ValueError, match="it-abs-err"):
        _run([*_REG, "--it-abs-err", "1", "--linear-scaling", "terms"])


@pytest.mark.parametrize("value", ["term", "Terms", "1.0"])
def test_cli_refuses_other_words(value, capsys):
    with pytest.raises(SystemExit):
        _run([*_REG, "--linear-scaling", value])
    assert "--linear-scaling: invalid value" in capsys.readouterr().err


def test_terms_are_evaluated_as_the_program_evaluates_them():
    """On contin rows the terms share one interpreter per row; their sum, in term order, is
    exactly the program's own output."""
    from pymoses.comboreduct.interpreter.interpreter import interpreter_visitor

    rows = _rows()
    prog = "+(sin(*(1.3 $1)) *($2 $2 0.5) /(1 +(4 $1)) exp(*(0.1 $2)))"
    tr = str_to_combo_tree(prog)
    b = _terms([0.0] * len(rows), rows)
    seen = {}
    orig = ls.fit_terms

    def spy(y, cols):
        seen["cols"] = cols
        return orig(y, cols)

    import pymoses.moses.scoring.bscores as bscores_mod

    bscores_mod.fit_terms = spy
    try:
        b.term_fit(tr)
    finally:
        bscores_mod.fit_terms = orig
    iv = interpreter_visitor(tr)
    for r, row in enumerate(rows):
        s = 0.0
        for c in seen["cols"]:
            s += c[r]
        assert s == iv(list(row))


def test_rows_with_logical_inputs_use_one_visitor_per_term():
    from pymoses.comboreduct.combo.vertex import builtin

    rows = [(0.5 * i - 3.0, builtin.logical_true if i % 3 else builtin.logical_false)
            for i in range(15)]  # fmt: skip
    ys = [1.0 + 2.0 * a + 3.0 * (1.0 if t is builtin.logical_true else 0.0) for a, t in rows]
    b = _terms(ys, rows)
    tr = str_to_combo_tree("+($1 impulse($2))")
    n, fit = b.term_fit(tr)
    assert n == 2 and fit.weights == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)
    assert all(abs(v) < 1e-12 for v in b(tr))
