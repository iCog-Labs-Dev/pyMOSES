"""Server round 14: the linear baselines LIN-OLS and LIN-RIDGE (`SERVER-ROUND14-PREREGISTRATION.md`,
"LIN-OLS and LIN-RIDGE: linear baselines").

`moses_sr_runner.py` calls `fit` for an arm with the key `"linear": "ols"` or `"linear": "ridge"`.
There is no MOSES process. Pure Python (`math.fsum`), as `pymoses.moses.scoring.least_squares`:
the server venv has no numpy.

The model is y = w_0 + sum_i w_i x_i over all k input columns of the split. A boolean input
(every token 0 or 1, for example Bos `x4`) is the number 0 or 1 (`sr_eval.read_table` reads every
input as a float). The runner fits the model to the training target scaled as for the MOSES arms,
y' = 10 (y - mean) / sd with the training mean and sd, and then gives the weights in the units of
the original target. The model has no extra a + b*f fit: its own predictions are scored.

* **OLS** (`fit_ols`): least squares with an intercept on the training rows, by
  `least_squares.fit_terms` (centred unit-length columns, the normal equations, `solve`).
  **Fallback** when `fit_terms` gives no weights (a constant input, or collinear inputs): the
  runner drops each input that modified Gram-Schmidt finds aliased (`independent_columns`: a
  constant column, or a column whose centred unit-length vector keeps a squared norm of at most
  `PIVOT_TOL` = 1e-12 after the projection on the columns kept before it; the rule of `fit_terms`
  for two terms), fits the other inputs with `fit_terms`, and gives each dropped input the weight
  0 (as R's `lm` drops an aliased column). The record names the dropped inputs and the reason of
  the first fit.
* **Ridge** (`fit_ridge`): the inputs are standardized with the training mean and the training
  standard deviation (population form, as the target scaling). The fit minimizes
  sum_rows (y' - b_0 - z . beta)^2 + lambda * |beta|^2 (the sum form; the intercept b_0 is not
  penalized, so b_0 = mean(y')), for each lambda of `RIDGE_LAMBDAS` = 10^k, k = -6, ..., 3. A
  constant input gets the weight 0. The runner keeps the lambda with the lowest validation NMSE;
  a tie keeps the larger lambda. The fit of each lambda uses the training rows only, so the
  "refit on the training rows" is the fit of the chosen lambda.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pymoses.moses.scoring.least_squares import PIVOT_TOL, fit_terms, predict_terms, solve

METHODS = ("ols", "ridge")
RIDGE_LAMBDAS = tuple(10.0**k for k in range(-6, 4))


def columns(X: Sequence[Sequence[float]]) -> list[list[float]]:
    """The input columns of the row-major matrix `X`."""
    return [list(c) for c in zip(*X, strict=True)]


def is_constant(col: Sequence[float]) -> bool:
    """The rule of `fit_terms`: the centred sum of squares is at most `PIVOT_TOL` times the
    uncentred one (an all-zero column also counts)."""
    n = len(col)
    m = math.fsum(col) / n
    return math.fsum((x - m) ** 2 for x in col) <= PIVOT_TOL * math.fsum(x * x for x in col)


def independent_columns(cols: Sequence[Sequence[float]]) -> list[int]:
    """The indices of the columns that modified Gram-Schmidt keeps, in order. A constant column
    is dropped. Each other column is centred and scaled to unit length; its projection on the
    columns kept before it is removed twice; the column is kept if the squared norm that stays
    is above `PIVOT_TOL`."""
    basis: list[list[float]] = []
    keep = []
    for j, col in enumerate(cols):
        if is_constant(col):
            continue
        m = math.fsum(col) / len(col)
        v = [x - m for x in col]
        d = math.sqrt(math.fsum(x * x for x in v))
        v = [x / d for x in v]
        for _ in range(2):
            for q in basis:
                p = math.fsum(a * b for a, b in zip(q, v, strict=True))
                v = [a - p * b for a, b in zip(v, q, strict=True)]
        r2 = math.fsum(x * x for x in v)
        if r2 <= PIVOT_TOL:
            continue
        r = math.sqrt(r2)
        basis.append([x / r for x in v])
        keep.append(j)
    return keep


def fit_ols(cols: Sequence[Sequence[float]], y: Sequence[float]) -> tuple[list[float], dict]:
    """`(weights, record)`: the least-squares weights (w_0, w_1, .., w_k) of y on the columns,
    and the record of the fit (`fallback`: None or "aliased"; `fit_reason`: the reason of the
    first `fit_terms` call that gave no weights; `dropped`: the indices of the dropped columns).
    """
    k = len(cols)
    fit = fit_terms(y, cols)
    if fit.weights is not None:
        return list(fit.weights), {"fallback": None, "fit_reason": None, "dropped": []}
    keep = independent_columns(cols)
    w = [0.0] * (k + 1)
    if keep:
        fit2 = fit_terms(y, [cols[j] for j in keep])
        if fit2.weights is None:
            raise ValueError(f"OLS: no fit after the aliased inputs are dropped ({fit2.reason})")
        w[0] = fit2.weights[0]
        for i, j in enumerate(keep):
            w[j + 1] = fit2.weights[i + 1]
    else:
        w[0] = math.fsum(y) / len(y)  # no usable input: the mean
    return w, {
        "fallback": "aliased",
        "fit_reason": fit.reason,
        "dropped": [j for j in range(k) if j not in keep],
    }


def fit_ridge(
    cols: Sequence[Sequence[float]], y: Sequence[float], lambdas: Sequence[float] = RIDGE_LAMBDAS
) -> tuple[list[list[float] | None], list[int]]:
    """`(fits, constant)`: for each lambda the weights (w_0, w_1, .., w_k) in the units of the
    columns (`None` if `solve` refuses the system), and the indices of the constant columns
    (weight 0). See the module docstring for the objective."""
    n, k = len(y), len(cols)
    constant = [j for j in range(k) if is_constant(cols[j])]
    use = [j for j in range(k) if j not in constant]
    means = {j: math.fsum(cols[j]) / n for j in use}
    sds = {j: math.sqrt(math.fsum((x - means[j]) ** 2 for x in cols[j]) / n) for j in use}
    z = [[(x - means[j]) / sds[j] for x in cols[j]] for j in use]
    my = math.fsum(y) / n
    yc = [t - my for t in y]
    m = len(use)
    gram = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(a, m):
            gram[a][b] = gram[b][a] = math.fsum(p * q for p, q in zip(z[a], z[b], strict=True))
    rhs = [math.fsum(p * q for p, q in zip(za, yc, strict=True)) for za in z]
    fits: list[list[float] | None] = []
    for lam in lambdas:
        a_mat = [row[:] for row in gram]
        for a in range(m):
            a_mat[a][a] += lam
        beta = solve(a_mat, rhs) if m else []
        if beta is None:
            fits.append(None)
            continue
        w = [0.0] * (k + 1)
        for b, j in zip(beta, use, strict=True):
            w[j + 1] = b / sds[j]
        w[0] = my - math.fsum(w[j + 1] * means[j] for j in use)
        fits.append(w)
    return fits, constant


def predict(weights: Sequence[float], cols: Sequence[Sequence[float]]) -> list[float]:
    """w_0 + sum_i w_i cols[i][r] for each row r (`least_squares.predict_terms`)."""
    return predict_terms(weights, cols)


def to_target_units(w_scaled: Sequence[float], mean: float, sd: float) -> list[float]:
    """The weights of the model of y' = 10 (y - mean) / sd as weights of the model of y."""
    f = sd / 10.0
    return [mean + f * w_scaled[0], *(f * x for x in w_scaled[1:])]
