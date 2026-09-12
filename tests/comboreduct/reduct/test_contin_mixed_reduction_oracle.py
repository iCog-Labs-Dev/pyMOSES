"""S14 Task 4's PRIMARY exit-criterion oracle: pymoses's full-pipeline
`contin_reduce` (contin_reduction.py) and `mixed_reduce`
(mixed_reduction.py) against `tests/fixtures/contin_mixed_reduction.json`
-- 53 trees x 2 modes (contin effort 2, mixed) plus 4 assumption-bearing
cases, captured directly from the real compiled `libcomboreduct.so` (see
`tools/capture_contin_mixed_reduction.py` and the two probes it drives).

`test_reduct.cc` (the pre-existing, already-built binary named in this
task's own brief as "the oracle test_reduct at contin and mixed effort
levels") does NOT drive contin/mixed reduction at all -- verified
directly (read its source, and fed it a `+(...)` tree live): it calls
ONLY `logical_reduce`, and its truth-table check assumes a boolean-typed
tree. `contin_mixed_reduct_probe.cc`/`mixed_assumption_probe.cc` (this
module's own oracle sources) exist because of that gap, in the same
"a new cpp_probe in place of a C++-side test file" pattern every other
slice in this project already uses.

Every case here uses classic's own REAL DEFAULT `ignore_ops` (empty) --
this satisfies CLAUDE.md's "at least one test runs classic's own
defaults" requirement at the FULL-PIPELINE level (test_contin_rules_
oracle.py already satisfies it at the per-rule level).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.assumption import insert_assumption
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.reduct.contin_reduction import (
    _reset_cache_for_testing as _reset_contin_cache,
)
from pymoses.comboreduct.reduct.contin_reduction import contin_reduce
from pymoses.comboreduct.reduct.mixed_reduction import (
    _reset_cache_for_testing as _reset_mixed_cache,
)
from pymoses.comboreduct.reduct.mixed_reduction import mixed_reduce

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
FIXTURE_PATH = _FIXTURES_DIR / "contin_mixed_reduction.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(autouse=True)
def _reset_caches():
    """`contin_reduction`/`mixed_reduction` are process-lifetime
    singletons in real C++, ported faithfully as module-level caches
    (see those modules' own docstrings) -- reset both before and after
    every test so this file's many differently-configured cases (empty
    vs. assumption-bearing) each get a freshly-built rule tree, matching
    what a fresh classic process would do for each independent run.
    """
    _reset_contin_cache()
    _reset_mixed_cache()
    yield
    _reset_contin_cache()
    _reset_mixed_cache()


def test_fixture_has_all_three_sections():
    assert len(FIXTURE["contin_effort_2"]) >= 50
    assert len(FIXTURE["mixed"]) >= 50
    assert len(FIXTURE["mixed_with_assumptions"]) == 4


@pytest.mark.parametrize(
    "case",
    FIXTURE["contin_effort_2"],
    ids=[c["input"] for c in FIXTURE["contin_effort_2"]],
)
def test_contin_reduce_byte_identical_against_classic(case: dict) -> None:
    assert case["ok"], f"oracle capture itself failed for {case['input']!r}: {case.get('error')}"
    tr = str_to_combo_tree(case["input"])
    contin_reduce(tr, reduct_effort=case["effort"], ignore_ops=set())
    assert combo_tree_to_str(tr).strip() == case["reduced"]


@pytest.mark.parametrize(
    "case",
    FIXTURE["mixed"],
    ids=[c["input"] for c in FIXTURE["mixed"]],
)
def test_mixed_reduce_byte_identical_against_classic(case: dict) -> None:
    assert case["ok"], f"oracle capture itself failed for {case['input']!r}: {case.get('error')}"
    tr = str_to_combo_tree(case["input"])
    mixed_reduce(tr)
    assert combo_tree_to_str(tr).strip() == case["reduced"]


@pytest.mark.parametrize(
    "case",
    FIXTURE["mixed_with_assumptions"],
    ids=[c["label"] for c in FIXTURE["mixed_with_assumptions"]],
)
def test_mixed_reduce_with_assumptions_byte_identical_against_classic(case: dict) -> None:
    """Covers `reduce_from_assumptions`/`reduce_gt_zero_sum`'s assumption-
    drawing path end to end -- not reachable through the plain-text
    corpus above, since combo-tree text syntax has no assumption
    literal.
    """
    tr = str_to_combo_tree(case["main_tree"])
    for assumption_str in case["assumptions"]:
        assumption_tr = str_to_combo_tree(assumption_str)
        insert_assumption(tr, assumption_tr.begin())
    mixed_reduce(tr)
    assert combo_tree_to_str(tr).strip() == case["reduced"]


def test_measured_agreement_summary():
    """Not a real assertion beyond "the fixture loaded and every case
    above passed" -- exists so the measured count is visible in test
    output/reports without having to count `-k` matches by hand.
    """
    total = (
        len(FIXTURE["contin_effort_2"])
        + len(FIXTURE["mixed"])
        + len(FIXTURE["mixed_with_assumptions"])
    )
    assert total >= 100
