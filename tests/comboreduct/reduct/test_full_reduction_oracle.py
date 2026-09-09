"""S14 Task 4's oracle for full_reduction.py, against `tests/fixtures/
full_reduction.json` (19 trees mixing logical/contin/mixed structure,
captured from the real compiled `libcomboreduct.so` via `tools/
capture_full_reduction.py`, which extends `contin_mixed_reduct_probe.cc`
with a "full" mode).

The corpus specifically includes `and(0<(+($1 1.0)) 0<(+($1 2.0)))`,
whose full-mode result (`0<(+(1 $1))`) DIFFERS from mixed_reduce's own
result on the identical input (`and(0<(+($1 1)))`, see test_contin_mixed_
reduction_oracle.py) -- measured confirmation that full_reduction.py's
three documented omissions relative to mixed_reduction.py (reduce_gt_
zero_log, reduce_junction_gt_zero_sum_constant, reduce_gt_division_of_
constants -- see full_reduction.py's own module docstring) are real and
change observable output, not merely a paper difference.

Uses classic's own real default `ignore_ops` (empty).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.full_reduction import _reset_cache_for_testing, full_reduce

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
FIXTURE_PATH = _FIXTURES_DIR / "full_reduction.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_cache_for_testing()
    yield
    _reset_cache_for_testing()


def test_fixture_is_non_empty():
    assert len(FIXTURE["full"]) >= 15


@pytest.mark.parametrize(
    "case",
    FIXTURE["full"],
    ids=[c["input"] for c in FIXTURE["full"]],
)
def test_full_reduce_byte_identical_against_classic(case: dict) -> None:
    assert case["ok"], f"oracle capture itself failed for {case['input']!r}: {case.get('error')}"
    tr = str_to_combo_tree(case["input"])
    full_reduce(tr)
    assert combo_tree_to_str(tr).strip() == case["reduced"]


def test_full_reduction_diverges_from_mixed_reduction_on_junction_sum_constant():
    """PROVEN BY MEASUREMENT, not just a docstring claim: full_reduction's
    missing `reduce_junction_gt_zero_sum_constant` (and the two other
    omissions) produce a genuinely DIFFERENT result than mixed_reduction
    on the exact same input.
    """
    from pymetta.comboreduct.reduct.mixed_reduction import (
        _reset_cache_for_testing as _reset_mixed,
    )
    from pymetta.comboreduct.reduct.mixed_reduction import mixed_reduce

    tree_str = "and(0<(+($1 1.0)) 0<(+($1 2.0)))"

    tr_full = str_to_combo_tree(tree_str)
    full_reduce(tr_full)

    _reset_mixed()
    tr_mixed = str_to_combo_tree(tree_str)
    mixed_reduce(tr_mixed)
    _reset_mixed()

    full_result = combo_tree_to_str(tr_full).strip()
    mixed_result = combo_tree_to_str(tr_mixed).strip()
    assert full_result == "0<(+(1 $1))"
    assert mixed_result == "and(0<(+($1 1)))"
    assert full_result != mixed_result
