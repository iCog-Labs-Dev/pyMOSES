"""Linear least squares for regression scoring.

NOT A PORT (pyMoses extension; no C++ counterpart). Pure Python (``math.fsum``), as the
rest of the regression extensions: the package declares no numpy dependency.

* ``solve``: Gaussian elimination with partial pivoting. Constant tuning
  (``pymoses.gomea.const_opt``, server round 3) used it first. It moved here in server
  round 12, so that the scorer (``bscores.contin_bscore``) can use it too: ``const_opt``
  imports ``bscores``, so ``bscores`` cannot import ``const_opt``. The code and
  ``PIVOT_TOL`` did not change, and ``const_opt`` imports both names from here.
* ``fit_terms`` and ``predict_terms``: the per-term least-squares fit of
  ``--linear-scaling terms`` (server round 12, T2.2,
  ``docs/gomea/SERVER-ROUND12-PREREGISTRATION.md``). The scorer and
  ``tools/gomea/sr_eval.py`` both call them, so the search and the report fit the same
  weights by the same rule.

**The per-term fit.** Columns ``t_1 .. t_k`` (the outputs of the k terms on the training
rows) and the target ``y``. ``fit_terms`` returns ``(w_0, w_1, .., w_k)`` that minimise
``sum (y - (w_0 + sum_i w_i t_i))^2``. It centres each column, scales it to unit length,
solves the normal equations of the scaled columns (a Gram matrix with a unit diagonal) with
``solve``, then maps the solution back to the raw columns. It gives no weights (a fallback
reason instead) in these cases:

* ``nonfinite``: a term output, or a number of the fit (a mean, a sum of squares, a weight,
  a fitted value), is not finite.
* ``constant``: a term is constant on the rows, so the intercept and that term are
  collinear. The rule: the centred sum of squares of the column is at most ``PIVOT_TOL``
  (1e-12) times its uncentred sum of squares (an all-zero column also counts).
* ``singular``: the terms are collinear. The rule: ``solve`` refuses the Gram matrix of the
  centred unit-length columns, so a pivot of the elimination is at most ``PIVOT_TOL`` (the
  largest diagonal entry is 1). For two terms this is exactly ``1 - r^2 <= 1e-12``, with
  ``r`` the correlation of the two terms on the rows. ``solve`` also refuses a result that
  is not finite.

The caller then scores the program with the plain ``a + b f`` fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# A pivot below PIVOT_TOL times the largest diagonal entry is "singular".
PIVOT_TOL = 1e-12

FALLBACK_NONFINITE = "nonfinite"
FALLBACK_CONSTANT = "constant"
FALLBACK_SINGULAR = "singular"


def solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """``a x = b`` by Gaussian elimination with partial pivoting; ``None``
    when a pivot is below ``PIVOT_TOL`` times the largest diagonal entry
    (numerically singular) or the result is not finite."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    scale = max((abs(a[i][i]) for i in range(n)), default=0.0)
    if not (scale > 0 and math.isfinite(scale)):
        return None
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) <= PIVOT_TOL * scale:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (m[c][n] - math.fsum(m[c][k] * x[k] for k in range(c + 1, n))) / m[c][c]
    return x if all(math.isfinite(v) for v in x) else None


@dataclass(frozen=True)
class TermFit:
    """The result of ``fit_terms``: ``weights`` = ``(w_0, w_1, .., w_k)`` and ``fitted``
    (the fitted value of each row, ``predict_terms``), or ``None`` for both with the
    fallback ``reason`` (``nonfinite``, ``constant`` or ``singular``)."""

    weights: tuple[float, ...] | None
    reason: str | None = None
    fitted: list[float] | None = None


def _finite(v) -> bool:
    return all(math.isfinite(x) for x in v)


def fit_terms(y: Sequence[float], cols: Sequence[Sequence[float]]) -> TermFit:
    """The least-squares weights of ``y ~ w_0 + sum_i w_i cols[i]`` (see the module
    docstring for the method and the fallback rules)."""
    n = len(y)
    if n == 0 or any(len(c) != n for c in cols):
        raise ValueError("fit_terms: every column needs one value per target row")
    if not all(_finite(c) for c in cols):
        return TermFit(None, FALLBACK_NONFINITE)
    means = [math.fsum(c) / n for c in cols]
    centred = [[x - m for x in c] for c, m in zip(cols, means, strict=True)]
    ss = [math.fsum(x * x for x in c) for c in centred]
    raw = [math.fsum(x * x for x in c) for c in cols]
    if not (_finite(means) and _finite(ss) and _finite(raw)):
        return TermFit(None, FALLBACK_NONFINITE)
    if any(s <= PIVOT_TOL * r for s, r in zip(ss, raw, strict=True)):
        return TermFit(None, FALLBACK_CONSTANT)
    norms = [math.sqrt(s) for s in ss]
    z = [[x / d for x in c] for c, d in zip(centred, norms, strict=True)]
    my = math.fsum(y) / n
    yc = [t - my for t in y]
    k = len(cols)
    gram = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            gram[i][j] = gram[j][i] = math.fsum(
                p * q for p, q in zip(z[i], z[j], strict=True)
            )
    rhs = [math.fsum(p * q for p, q in zip(zi, yc, strict=True)) for zi in z]
    if not (all(_finite(row) for row in gram) and _finite(rhs) and math.isfinite(my)):
        return TermFit(None, FALLBACK_NONFINITE)
    v = solve(gram, rhs)
    if v is None:
        return TermFit(None, FALLBACK_SINGULAR)
    w = [vi / d for vi, d in zip(v, norms, strict=True)]
    w0 = my - math.fsum(wi * m for wi, m in zip(w, means, strict=True))
    weights = (w0, *w)
    if not _finite(weights):
        return TermFit(None, FALLBACK_NONFINITE)
    fitted = predict_terms(weights, cols)
    if not _finite(fitted):
        return TermFit(None, FALLBACK_NONFINITE)
    return TermFit(weights, None, fitted)


def predict_terms(weights: Sequence[float], cols: Sequence[Sequence[float]]) -> list[float]:
    """``w_0 + sum_i w_i cols[i][r]`` for each row ``r`` (an exactly rounded sum)."""
    w0, w = weights[0], weights[1:]
    if len(w) != len(cols):
        raise ValueError("predict_terms: one weight per column, plus the intercept")
    n = len(cols[0]) if cols else 0
    return [
        math.fsum([w0, *(wi * c[r] for wi, c in zip(w, cols, strict=True))]) for r in range(n)
    ]
