"""`pymoses.moses.scoring.least_squares` (server round 12, T2.2): the solver that moved from
`pymoses.gomea.const_opt`, and the per-term fit of `--linear-scaling terms`."""

from __future__ import annotations

import math

import pytest

from pymoses.gomea import const_opt
from pymoses.moses.scoring import least_squares as ls
from pymoses.moses.scoring.least_squares import TermFit, fit_terms, predict_terms

_N = 24


def _cols():
    """Three independent columns on 24 rows."""
    t1 = [math.sin(0.7 * i) for i in range(_N)]
    t2 = [0.1 * i * i - 1.0 for i in range(_N)]
    t3 = [math.cos(1.3 * i) + 0.05 * i for i in range(_N)]
    return [t1, t2, t3]


def test_solve_is_the_constant_tuning_solver():
    assert const_opt.solve is ls.solve
    assert const_opt.PIVOT_TOL == ls.PIVOT_TOL == 1e-12
    assert ls.solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]) == pytest.approx([0.8, 1.4])
    assert ls.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]) is None


@pytest.mark.parametrize("k", [1, 2, 3])
def test_k_terms_get_k_weights_and_an_intercept(k):
    cols = _cols()[:k]
    y = [1.5 + sum((j + 2.0) * c[r] for j, c in enumerate(cols)) for r in range(_N)]
    fit = fit_terms(y, cols)
    assert fit.reason is None
    assert len(fit.weights) == k + 1
    assert fit.weights == pytest.approx([1.5, *(j + 2.0 for j in range(k))], abs=1e-9)
    assert fit.fitted == pytest.approx(y, abs=1e-9)
    assert fit.fitted == predict_terms(fit.weights, cols)


def test_fit_is_the_least_squares_solution():
    cols = _cols()
    y = [math.exp(0.05 * r) + (-1) ** r * 0.3 for r in range(_N)]
    fit = fit_terms(y, cols)
    res = [f - t for f, t in zip(fit.fitted, y, strict=True)]
    # normal equations: the residual is orthogonal to the intercept and to every column
    assert abs(math.fsum(res)) < 1e-9
    for c in cols:
        assert abs(math.fsum(r * x for r, x in zip(res, c, strict=True))) < 1e-9


def test_predict_terms_uses_the_weights_on_new_rows():
    assert predict_terms((1.0, 2.0, -1.0), [[1.0, 0.5], [3.0, 4.0]]) == [0.0, -2.0]


@pytest.mark.parametrize(
    "make",
    [
        lambda t: [2.0 * x for x in t],  # a multiple
        lambda t: [x + 3.0 for x in t],  # a shift (collinear with the intercept)
        lambda t: [-0.37 * x + 11.0 for x in t],  # an affine map
    ],
)
def test_collinear_terms_fall_back(make):
    t1 = _cols()[0]
    fit = fit_terms([float(i) for i in range(_N)], [t1, make(t1)])
    assert fit == TermFit(None, ls.FALLBACK_SINGULAR)


def test_collinear_with_three_terms_falls_back():
    t1, t2, _ = _cols()
    t3 = [a - 2.0 * b + 0.5 for a, b in zip(t1, t2, strict=True)]
    fit = fit_terms([float(i) for i in range(_N)], [t1, t2, t3])
    assert fit.reason == ls.FALLBACK_SINGULAR


def _pair(one_minus_r2: float):
    """Two columns whose correlation r has 1 - r^2 = `one_minus_r2` (up to rounding)."""
    u = [1.0, -1.0] * (_N // 2)  # centred
    t = [float(i) - (_N - 1) / 2 for i in range(_N)]  # centred
    tt, uu = math.fsum(x * x for x in t), math.fsum(x * x for x in u)
    tu = math.fsum(x * y for x, y in zip(t, u, strict=True))
    u = [x - tu / tt * y for x, y in zip(u, t, strict=True)]  # orthogonal to t
    uu = math.fsum(x * x for x in u)
    # 1 - r^2 = e^2 uu / (tt + e^2 uu)
    e = math.sqrt(one_minus_r2 * tt / ((1 - one_minus_r2) * uu))
    return t, [x + e * y for x, y in zip(t, u, strict=True)]


def test_the_rank_rule_for_two_terms_is_one_minus_r_squared():
    y = [math.sin(i) for i in range(_N)]
    assert fit_terms(y, list(_pair(1e-6))).reason is None
    assert fit_terms(y, list(_pair(1e-10))).reason is None
    assert fit_terms(y, list(_pair(1e-14))).reason == ls.FALLBACK_SINGULAR


@pytest.mark.parametrize("value", [4.0, -0.1, 0.0, 1e6])
def test_a_constant_term_falls_back(value):
    t1 = _cols()[0]
    fit = fit_terms([float(i) for i in range(_N)], [t1, [value] * _N])
    assert fit == TermFit(None, ls.FALLBACK_CONSTANT)


def test_the_constant_rule_is_relative_to_the_size_of_the_term():
    t1 = _cols()[0]
    y = [float(i) for i in range(_N)]
    wiggle = [(-1) ** i for i in range(_N)]
    # centred sum of squares / uncentred = 1e-14: constant
    near = [100.0 + 1e-5 * w for w in wiggle]
    assert fit_terms(y, [t1, near]).reason == ls.FALLBACK_CONSTANT
    # 1e-10: a (small) real term
    small = [100.0 + 1e-3 * w for w in wiggle]
    assert fit_terms(y, [t1, small]).reason is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_term_output_that_is_not_finite_falls_back(bad):
    t1, t2, _ = _cols()
    t2 = list(t2)
    t2[5] = bad
    fit = fit_terms([float(i) for i in range(_N)], [t1, t2])
    assert fit == TermFit(None, ls.FALLBACK_NONFINITE)


def test_an_overflowing_fit_falls_back():
    t1 = _cols()[0]
    huge = [1e200 * (i + 1) for i in range(_N)]  # its sum of squares is inf
    assert fit_terms([1.0] * _N, [t1, huge]).reason == ls.FALLBACK_NONFINITE


def test_rows_must_match():
    with pytest.raises(ValueError):
        fit_terms([1.0, 2.0], [[1.0, 2.0, 3.0]])
