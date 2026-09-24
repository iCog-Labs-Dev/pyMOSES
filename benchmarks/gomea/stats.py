"""Pure statistics functions for the GOMEA benchmark harness (Task 9).

Stdlib only (`math`, `random`) -- no numpy/scipy. Every function here is a
plain function over plain Python numbers/sequences, independently testable
against hand-computed cases (see `tests/gomea/test_benchmarks.py`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

__all__ = [
    "wilson_ci",
    "wilcoxon_signed_rank",
    "holm",
    "bootstrap_median_ci",
    "median_iqr",
    "ert",
]


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion `k`/`n` (default
    `z` = 1.959964, the two-sided 95% normal quantile).

    Returns `(0.0, 1.0)` for `n == 0` (no information). At `k == 0` the
    lower bound is exactly 0.0 and at `k == n` the upper bound is exactly
    1.0 (both fall out of the formula's own algebra, not a special case).
    """
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = center - margin
    hi = center + margin
    return max(0.0, lo), min(1.0, hi)


def _midranks(abs_vals: Sequence[float]) -> list[float]:
    """1-based midranks of `abs_vals` (ties share the average of the ranks
    they span), matching the classic Wilcoxon tie-handling convention."""
    n = len(abs_vals)
    order = sorted(range(n), key=lambda i: abs_vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_vals[order[j + 1]] == abs_vals[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _exact_null_distribution(ranks: Sequence[float]) -> dict[int, int]:
    """Counts of achievable `sum(positive-signed ranks)` values over all
    `2**n` equally-likely sign assignments, via subset-sum DP (every
    achievable positive-rank sum is exactly a subset sum of `ranks`, and
    the number of sign assignments giving that sum equals the number of
    subsets summing to it) -- exact and fast even at `n == 25`, unlike
    literally enumerating `2**25` sign patterns. Ranks are scaled by 2 first
    so midranks (always integer or half-integer) become exact integers.
    """
    scaled = [round(r * 2) for r in ranks]
    dp: dict[int, int] = {0: 1}
    for w in scaled:
        new_dp: dict[int, int] = dict(dp)
        for s, c in dp.items():
            new_dp[s + w] = new_dp.get(s + w, 0) + c
        dp = new_dp
    return dp


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    diffs: Sequence[float], method: str = "auto"
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test, two-sided. Returns `(W, p)` where
    `W = min(W+, W-)` (the usual Wilcoxon T statistic) and `p` is the
    two-sided p-value.

    Zero differences are dropped before ranking (Wilcoxon's own method).
    `p = 1.0` when no nonzero differences remain. Ties among the nonzero
    absolute differences use midranks.

    `method`: `"auto"` (default) uses the exact permutation distribution
    (via subset-sum DP over the actual, possibly tied, ranks) when
    `n_nonzero <= 25`, and a normal approximation with a tie correction and
    a continuity correction otherwise; `"exact"`/`"normal"` force one or
    the other regardless of `n_nonzero` (used to compare the two directly
    at the same n).
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0

    abs_vals = [abs(d) for d in nz]
    signs = [1 if d > 0 else -1 for d in nz]
    ranks = _midranks(abs_vals)

    w_pos = sum(r for r, s in zip(ranks, signs, strict=True) if s > 0)
    w_neg = sum(r for r, s in zip(ranks, signs, strict=True) if s < 0)
    w = min(w_pos, w_neg)

    use_exact = method == "exact" or (method == "auto" and n <= 25)
    if method not in ("auto", "exact", "normal"):
        raise ValueError(f"wilcoxon_signed_rank: unknown method {method!r}")

    if use_exact:
        dist = _exact_null_distribution(ranks)
        total = sum(dist.values())
        t_obs_scaled = round(w_pos * 2)
        le = sum(c for s, c in dist.items() if s <= t_obs_scaled)
        ge = sum(c for s, c in dist.items() if s >= t_obs_scaled)
        p = min(1.0, 2 * min(le, ge) / total)
    else:
        mean = n * (n + 1) / 4.0
        # Tie correction: sum over tie groups of (t**3 - t), t = group size.
        tie_term = 0.0
        sorted_abs = sorted(abs_vals)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
                j += 1
            t = j - i + 1
            tie_term += t**3 - t
            i = j + 1
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
        sd = math.sqrt(var) if var > 0 else 0.0
        diff = w_pos - mean
        if sd == 0.0:
            z_stat = 0.0
        elif diff > 0:
            z_stat = (diff - 0.5) / sd
        elif diff < 0:
            z_stat = (diff + 0.5) / sd
        else:
            z_stat = 0.0
        p = min(1.0, 2 * (1 - _norm_cdf(abs(z_stat))))

    return w, p


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down multiple-comparison adjustment. Returns adjusted
    p-values in the SAME order as `pvals`, monotone non-decreasing when
    read in ascending order of the original p-values, and capped at 1.0."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        running_max = max(running_max, factor * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _quantile_sorted(xs_sorted: Sequence[float], q: float) -> float:
    """Type-7 quantile (linear interpolation between order statistics --
    R's and numpy's default), given an ALREADY-sorted sequence."""
    n = len(xs_sorted)
    if n == 0:
        return float("nan")
    if n == 1:
        return xs_sorted[0]
    h = (n - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs_sorted[int(h)]
    frac = h - lo
    return xs_sorted[lo] + frac * (xs_sorted[hi] - xs_sorted[lo])


def median_iqr(xs: Sequence[float]) -> tuple[float, float, float]:
    """`(median, q1, q3)` of `xs`, type-7 (linear interpolation) quantiles.
    `(nan, nan, nan)` for an empty sequence."""
    xs_sorted = sorted(xs)
    if not xs_sorted:
        nan = float("nan")
        return nan, nan, nan
    return (
        _quantile_sorted(xs_sorted, 0.5),
        _quantile_sorted(xs_sorted, 0.25),
        _quantile_sorted(xs_sorted, 0.75),
    )


def bootstrap_median_ci(
    diffs: Sequence[float], n: int = 10000, seed: int = 12345
) -> tuple[float, float]:
    """Bootstrap 95% CI of the median of `diffs` (used for the paired
    median difference between two arms), `n` resamples with replacement,
    reproducible via `random.Random(seed)`. `(nan, nan)` for an empty
    input."""
    m = len(diffs)
    if m == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    values = list(diffs)
    medians = []
    for _ in range(n):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        medians.append(_quantile_sorted(sorted(sample), 0.5))
    medians.sort()
    return _quantile_sorted(medians, 0.025), _quantile_sorted(medians, 0.975)


def ert(evals_used: Sequence[float], successes: Sequence[bool]) -> float:
    """Expected running cost / evaluations-to-target:
    `sum(evals_used) / count(successes)`, `inf` when there are zero
    successes. `evals_used[i]` must already be right-censored by the
    caller -- the true evaluations-to-target for a successful run, and the
    evaluation budget/cap for an unsuccessful one (including a timeout or
    an error), per 03_EXPERIMENT_PROTOCOL.md Sec 8's "retain unsuccessful
    runs as right-censored observations".
    """
    n_success = sum(1 for s in successes if s)
    if n_success == 0:
        return float("inf")
    return sum(evals_used) / n_success
