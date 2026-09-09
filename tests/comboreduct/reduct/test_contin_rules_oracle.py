"""S14 Task 4 differential oracle: pymetta's contin_rules.py/branch_rules.py
rule classes against `tests/fixtures/contin_branch_rules.json` (62 cases,
captured from the real compiled `libcomboreduct.so` via
`tools/cpp_probe/contin_rule_probe.cc` -- see `tools/
capture_contin_branch_rules.py` for the exact corpus and capture method).

Each fixture row applies ONE named rule, pre-order (`downwards(rule())`,
matching the probe's own application method exactly), to a freshly-parsed
combo tree, and records the resulting tree text. This is the file that
satisfies CLAUDE.md's "at least one test runs classic's own defaults"
requirement for the per-rule layer: every rule here runs with NO
`ignore_ops` exclusions and at classic's actual default effort level (1,
for the two rules -- `reduce_cond_arg`/`reduce_impulse_arg` -- that take
one; not exercised by this fixture, which only covers rules with a
no-argument constructor, but the full-pipeline oracle in
`test_contin_reduction_oracle.py`/`test_mixed_reduction_oracle.py` does
cover them, at classic's real default `contin_reduction`/
`mixed_reduction` composition).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct import branch_rules, contin_rules
from pymetta.comboreduct.reduct.meta_rules import downwards

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
FIXTURE_PATH = _FIXTURES_DIR / "contin_branch_rules.json"

_RULE_CLASSES = {
    "reduce_plus_zero": contin_rules.reduce_plus_zero,
    "reduce_times_one_zero": contin_rules.reduce_times_one_zero,
    "reduce_plus_times_one_child": contin_rules.reduce_plus_times_one_child,
    "reduce_invert_constant": contin_rules.reduce_invert_constant,
    "reduce_fraction": contin_rules.reduce_fraction,
    "reduce_factorize": contin_rules.reduce_factorize,
    "reduce_factorize_fraction": contin_rules.reduce_factorize_fraction,
    "reduce_times_div": contin_rules.reduce_times_div,
    "reduce_sum_log": contin_rules.reduce_sum_log,
    "reduce_log_div_times": contin_rules.reduce_log_div_times,
    "reduce_exp_times": contin_rules.reduce_exp_times,
    "reduce_exp_div": contin_rules.reduce_exp_div,
    "reduce_exp_log": contin_rules.reduce_exp_log,
    "reduce_sin": contin_rules.reduce_sin,
    "reduce_cond_else": branch_rules.reduce_cond_else,
    "reduce_cond_const": branch_rules.reduce_cond_const,
    "reduce_cond_adjacent": branch_rules.reduce_cond_adjacent,
}


def _load_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())["cases"]


CASES = _load_cases()


def test_fixture_is_non_empty():
    assert len(CASES) >= 60


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{c['rule']}:{c['input']}" for c in CASES],
)
def test_byte_identical_against_classic(case: dict) -> None:
    rule_cls = _RULE_CLASSES[case["rule"]]
    tr = str_to_combo_tree(case["input"])
    downwards(rule_cls())(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == case["output"]


def test_every_rule_class_has_at_least_one_fixture_case():
    covered = {c["rule"] for c in CASES}
    assert covered == set(_RULE_CLASSES)
