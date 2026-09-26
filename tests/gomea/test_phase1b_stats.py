"""Unit tests for the unpaired statistics in tools/gomea/phase1b_report.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gomea"))
from phase1b_report import boot_median_diff, fisher_exact, mann_whitney  # noqa: E402


def test_fisher_exact_textbook_table():
    # [[1, 9], [11, 3]]: the classic example, two-sided p = 0.002759 (R: fisher.test).
    assert math.isclose(fisher_exact(1, 9, 11, 3), 0.0027594, rel_tol=1e-4)


def test_fisher_exact_identical_rows_is_one():
    assert fisher_exact(5, 5, 5, 5) == 1.0
    assert fisher_exact(0, 30, 0, 30) == 1.0


def test_fisher_exact_extreme_split():
    # 30/30 vs 0/30: p = 2 / C(60, 30).
    assert math.isclose(fisher_exact(30, 0, 0, 30), 2 / math.comb(60, 30), rel_tol=1e-9)


def test_mann_whitney_separated_samples():
    # x = 1..3 vs y = 4..6: U = 0; normal approximation with continuity correction:
    # z = (4.5 - 0.5) / sqrt(5.25) = 1.7457, p = 0.0809.
    u, p = mann_whitney([1, 2, 3], [4, 5, 6])
    assert u == 0
    assert math.isclose(p, 0.08086, rel_tol=1e-3)


def test_mann_whitney_all_tied_is_one():
    assert mann_whitney([2, 2, 2], [2, 2]) == (3.0, 1.0)


def test_mann_whitney_symmetric():
    x, y = [1, 3, 3, 5, 7], [2, 4, 4, 6, 8, 9]
    assert math.isclose(mann_whitney(x, y)[1], mann_whitney(y, x)[1])


def test_boot_median_diff_brackets_true_shift():
    lo, hi = boot_median_diff([10.0] * 20 + [11.0] * 20, [0.0] * 20 + [1.0] * 20, n=2000)
    assert lo <= 10.0 <= hi or lo <= 10.5 <= hi
