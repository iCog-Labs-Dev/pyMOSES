"""Task 6 (S7 cycle): the differential oracle that closes the cycle.

C++: moses/comboreduct/reduct/reduct.h:124 logical_reduce(int, combo_tree&,
const vertex_set&) -- the 3-arg overload actually exercised throughout
this module and by every `logical_reduce(effort, tr, ignore_ops=set())`
call below (reduct.h:116 is a DIFFERENT, 4-arg overload additionally
taking a `combo_tree::iterator`; not used here), moses/comboreduct/
reduct/logical_rules.h:169 subtree_to_enf.

Verifies `pymoses.comboreduct.reduct.logical_reduction.logical_reduce`
against `tests/fixtures/reduct.json` (the committed 103-tree corpus + real
`test_reduct`/`logical_reduce` capture -- see `tools/cpp_probe/
reduct_corpus.py`, `tools/capture_reduct.py`,
`tests/fixtures/test_reduct_shape.py` (this fixture's own shape guard,
which this module does not repeat), and `.superpowers/sdd/
2026-09-04-pymoses-s7-reduct/task6-corpus-report.md` for how it was
built) across four independent layers:

1. BYTE-IDENTICAL (`test_byte_identical_*`): for every one of the 237 (79
   runnable trees x effort 1/2/3) captured cases, pyMoses's own
   `logical_reduce` output matches classic's real `logical_reduce` output
   character-for-character. This is the cycle's PRIMARY exit criterion
   (docs/superpowers/plans/2026-09-04-pymoses-s7-reduct.md's own "Cycle
   exit criterion" #2).

2. SEMANTIC (truth-table preservation), amended 2026-09-07 per docs/
   CARRY-FORWARD.md Β§3: classic's own `eval_logical_identities` is
   unsound on trees where a junctor's entire (non-empty) child list
   consists of childless junctors, and the port must match that
   unsoundness byte-for-byte (layer 1 already does) rather than silently
   "fixing" it. `enf_checkers.tt_unsound_by_construction` is a
   STRUCTURAL indicator for that shape, computed independently
   of this fixture's own `known_tt_divergent` field --
   `test_predicate_independently_reproduces_the_known_tt_divergent_split`
   is the cross-check the task brief asks for; the tree-shape unit tests
   further down independently prove the predicate discriminates (not a
   constant, not copied from the fixture) on synthetic input the fixture
   does not contain.

   Review found that the predicate is neither necessary nor sufficient for
   end-to-end divergence: parent context can mask an immediately-present
   occurrence, while earlier effort-1 rules can create the shape from an
   input the predicate does not flag. Semantic correctness is therefore
   checked directly against C++'s captured `tt_computed`/`tt_preserved`
   diagnostics. `test_pinned_context_dependent_rows` preserves the concrete
   counterexamples so this limitation cannot regress back into an iff or
   one-way guarantee.

3. STRUCTURAL (the seven ENF axioms, `enf_checkers.py`), applied ONLY at
   effort 3 -- see `test_effort_2_output_can_still_violate_enf_measured`
   for why: MEASURED, not assumed, effort 2's "medium" pipeline leaves 3
   of the 79 corpus trees (2x rule_six, 1x rule_one) in a shape that
   effort 3's own extra `iterative(sequential(p_medium, reduce_remove_
   subtree_equal_tt()))` wrapping still simplifies further -- e.g.
   `GeneralTestCases.py#35`'s effort-2 output keeps a `!$1` nested inside
   a branch whose ancestor already asserts `$1` (a real, if transient,
   ENF violation), which effort 3 removes. This is "the output of
   subtree_to_enf" in the sense the task brief's own note means: the
   COMPLETE effort pipeline that actually reaches ENF, not every effort
   that merely touches `subtree_to_enf` somewhere along the way.

4. The randomized differential harness against the full effort pipelines
   (`tools/cpp_probe/run_differential_effort.py`, this task's own
   addition) is run separately, over a corpus far larger than this
   fixture's hand-picked 79 trees -- see task-6-report.md for the
   transcript and divergence count; it is a developer tool (needs the
   classic C++ build), not wired into this file or `tools/ci.sh`, the
   same precedent `tools/cpp_probe/run_differential.py` (Task 4) set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from enf_checkers import (
    ROOT_TOKEN,
    enf_violations,
    evaluate_truth_table,
    rule_five,
    rule_four,
    rule_one,
    rule_seven,
    rule_six,
    rule_three,
    rule_two,
    truth_table_preserved,
    tt_unsound_by_construction,
)

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduce

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_FIXTURE = json.loads((_FIXTURE_DIR / "reduct.json").read_text())
_CASES = _FIXTURE["cases"]

# capture_reduct.py only ever records a case it could actually run to
# completion (BooleanFullUTest.cxxtest's 20 not-yet-runnable trees are in
# `_FIXTURE["skipped"]` instead, covered by test_reduct_shape.py, not
# here) -- every entry in `_CASES` has `ok: true`. Asserted once, at
# import time, rather than trusted silently: every test below relies on
# `case["reduced"]`/`case["tt_preserved"]` existing unconditionally.
assert all(c["ok"] for c in _CASES), "reduct.json has a non-ok case -- capture_reduct.py changed"
assert len(_CASES) > 0, "reduct.json's cases list is empty"

# One entry per unique corpus TREE (not per effort) -- first occurrence
# wins; `known_tt_divergent` is identical across all 3 efforts for a
# given tree (test_reduct_shape.py's own
# test_known_tt_divergent_ids_match_and_are_internally_consistent already
# pins this), so any one is representative.
_TREES_BY_ID: dict[str, dict] = {}
for _c in _CASES:
    _TREES_BY_ID.setdefault(_c["id"], _c)
assert len(_TREES_BY_ID) == 79


def _case_id(case: dict) -> str:
    return f"{case['id']}@effort{case['effort']}"


# ---------------------------------------------------------------------------
# 1. Byte-identical -- the cycle's primary exit criterion.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_byte_identical_to_classic_logical_reduce(case):
    tr = str_to_combo_tree(case["input"])
    logical_reduce(case["effort"], tr, ignore_ops=set())
    assert combo_tree_to_str(tr) == case["reduced"]


def test_byte_identical_check_actually_discriminates():
    """Prove `test_byte_identical_to_classic_logical_reduce`'s own
    assertion is not vacuous -- i.e. that deleting it would fail a test.
    Reduces one real case, confirms the REAL output equals the fixture's
    `reduced` value (the check we rely on above), and separately confirms
    the unreduced input does NOT equal that expected value. This proves a
    no-op implementation cannot satisfy the oracle.
    """
    # Pick a case for which a no-op reducer is observably wrong. This
    # makes the check fail if `logical_reduce` is accidentally bypassed,
    # unlike comparing a value with `actual + "!"` (which is tautological).
    case = next(c for c in _CASES if c["input"].strip() != c["reduced"].strip())
    tr = str_to_combo_tree(case["input"])
    logical_reduce(case["effort"], tr, ignore_ops=set())
    actual = combo_tree_to_str(tr)
    assert actual == case["reduced"]

    no_op_result = combo_tree_to_str(str_to_combo_tree(case["input"]))
    assert no_op_result != case["reduced"], "selected case cannot detect a no-op reducer"


# ---------------------------------------------------------------------------
# 2. Semantic: reproduce C++'s measured truth-table outcome directly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tree_id,case",
    sorted(_TREES_BY_ID.items()),
    ids=[tid for tid, _ in sorted(_TREES_BY_ID.items())],
)
def test_predicate_independently_reproduces_the_known_tt_divergent_split(tree_id, case):
    """`enf_checkers.tt_unsound_by_construction` is computed purely from
    tree SHAPE, with no reference to this fixture's own
    `known_tt_divergent` flag anywhere in its implementation (see that
    function's own docstring). It reproduces this fixture's historical
    13-tree split, but is intentionally not asserted as a complete classifier
    for inputs outside the fixture; the pinned emergent-shape counterexample
    below proves that limitation explicitly.
    """
    tr = str_to_combo_tree(case["input"])
    assert tt_unsound_by_construction(tr) == case["known_tt_divergent"], tree_id


@pytest.mark.oracle
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_truth_table_outcome_matches_classic(case):
    """Match C++'s measured semantic outcome without classifier assumptions."""
    original = str_to_combo_tree(case["input"])

    reduced = str_to_combo_tree(case["input"])
    logical_reduce(case["effort"], reduced, ignore_ops=set())

    preserved = truth_table_preserved(original, reduced)
    if case.get("tt_computed", True):
        assert preserved == case["tt_preserved"], case["id"]
    else:
        assert preserved is None, case["id"]


@pytest.mark.parametrize(
    "effort,combo_str,expected_flagged,expected_reduced,expected_preserved",
    [
        (3, "and(or())", True, "true ", True),
        (3, "and(or() or())", True, "true ", True),
        (3, "or(and(or()) $1)", True, "$1 ", False),
        (3, "and($1 or(and(or()) $1))", True, "$1 ", True),
        (1, "or(and(or($1 and(!$1))) !$2)", False, "!$2 ", False),
    ],
)
def test_pinned_context_dependent_rows(
    effort, combo_str, expected_flagged, expected_reduced, expected_preserved
):
    """Pin false positives and a reduction-created false negative."""
    original = str_to_combo_tree(combo_str)
    reduced = str_to_combo_tree(combo_str)
    logical_reduce(effort, reduced, ignore_ops=set())

    assert tt_unsound_by_construction(original) is expected_flagged
    assert combo_tree_to_str(reduced) == expected_reduced
    assert truth_table_preserved(original, reduced) is expected_preserved


# ---------------------------------------------------------------------------
# Unit-level proof that `tt_unsound_by_construction` actually discriminates
# on tree SHAPE -- synthetic trees the corpus does not contain, so this
# cannot be satisfied by coincidentally matching the fixture's own flags.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "combo_str,expected",
    [
        # The documented shape itself: a junctor (and) whose entire,
        # non-empty child list is childless junctors (a single or()).
        ("and(or())", True),
        ("or(and())", True),
        # A junctor with SEVERAL childless-junctor children -- still
        # matches (>=1 non-empty list, ALL childless junctors).
        ("and(or() or())", True),
        # NOT a match: the or() has a real child, so it is not itself
        # childless -- the outer and's one child fails "childless".
        ("and(or($1))", False),
        # NOT a match: and's children are a real literal plus a childless
        # or() -- not ALL of and's children are childless junctors.
        ("and($1 or())", False),
        # NOT a match: a bare trailing empty and()/or() next to real
        # content at the ROOT is exactly what most of the corpus's
        # non-divergent testTrees.sh entries look like -- the ROOT here
        # has a real ($1-bearing) sibling, so its own child list is not
        # ALL childless junctors.
        ("or($1 and())", False),
        # A plain childless junctor alone is not itself a MATCH (no
        # nonempty child list to inspect) -- the predicate is about a
        # PARENT's children, not about being childless oneself.
        ("and()", False),
        ("or()", False),
        # A deeply nested match must still be found (whole-tree scan, not
        # root-only).
        ("or($1 and(and(or())))", True),
    ],
)
def test_predicate_discriminates_on_synthetic_shapes(combo_str, expected):
    tr = str_to_combo_tree(combo_str)
    assert tt_unsound_by_construction(tr) is expected, combo_str


# ---------------------------------------------------------------------------
# The distinct third outcome: a raw operator token reaching the root.
# No corpus tree (original OR reduced, at any effort) actually reaches
# this -- confirmed by test_truth_table_preserved_except_on_the_unsound_
# class's own `assert preserved is not None` above holding for all 237
# cases -- so it is proven directly against a synthetic minimal tree.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo_str", ["and()", "or()"])
def test_root_token_is_a_distinct_third_outcome(combo_str):
    tr = str_to_combo_tree(combo_str)
    result = evaluate_truth_table(tr)
    assert result is not True
    assert result is not False
    assert result is ROOT_TOKEN


def test_root_token_is_not_confused_with_an_ordinary_crash():
    """A childless and()/or() at the root is a WELL-DEFINED distinct
    outcome (ROOT_TOKEN), not an exception -- `evaluate_truth_table` must
    return normally, not raise, for this input; some other tree causing a
    genuine crash (e.g. malformed input) is a different failure mode
    entirely and must not be silently absorbed into this one."""
    tr = str_to_combo_tree("and()")
    evaluate_truth_table(tr)  # must not raise


def test_truth_table_preserved_returns_none_for_a_root_token_tree():
    before = str_to_combo_tree("and()")
    after = str_to_combo_tree("and()")
    assert truth_table_preserved(before, after) is None


# ---------------------------------------------------------------------------
# 3. Structural: the seven ENF axioms, at effort 3 only.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "tree_id,case",
    sorted(_TREES_BY_ID.items()),
    ids=[tid for tid, _ in sorted(_TREES_BY_ID.items())],
)
def test_effort_3_output_satisfies_enf(tree_id, case):
    tr = str_to_combo_tree(case["input"])
    logical_reduce(3, tr, ignore_ops=set())
    violations = enf_violations(tr)
    assert violations == [], f"{tree_id}: {violations} on {combo_tree_to_str(tr)!r}"


def test_effort_2_does_not_yet_guarantee_enf_measured():
    """Prove `test_effort_3_output_satisfies_enf`'s scoping to effort 3
    (not effort 2) is a measured necessity, not a convenience: running
    the SAME seven checkers over effort 2's ("medium") own real output
    finds exactly 3 genuine violations across the 79-tree corpus, pinned
    here by id and rule so a future change either preserves this exact,
    understood gap or is a real signal something moved. This is also the
    "prove it fires" check for `enf_checkers.py` itself: if rule_six/
    rule_one were stubbed to always return True, this test would go from
    3 failures found to 0, and fail outright.
    """
    expected = {
        "GeneralTestCases.py#11": "rule_six",
        "GeneralTestCases.py#34": "rule_six",
        "GeneralTestCases.py#35": "rule_one",
    }
    found: dict[str, str] = {}
    for tree_id, case in _TREES_BY_ID.items():
        tr = str_to_combo_tree(case["input"])
        logical_reduce(2, tr, ignore_ops=set())
        violations = enf_violations(tr)
        if violations:
            assert len(violations) == 1, (tree_id, violations)
            found[tree_id] = violations[0]
    assert found == expected


def test_rule_six_and_rule_one_are_not_vacuously_true():
    """Direct unit proof that `rule_six`/`rule_one` can each independently
    return False on a small, hand-built tree -- not just inferred from the
    corpus-driven test above. `and(and($1) $1)` nests a real ancestor AND
    asserting $1 above a descendant AND re-asserting the SAME literal --
    exactly rule_six's "no literal re-asserted by a node whose ancestor
    already guarantees it"."""
    tr = str_to_combo_tree("and($1 and($1 $2))")
    assert rule_six(tr) is False

    tr2 = str_to_combo_tree("and($1 or(!$1 $2))")
    assert rule_one(tr2) is False


@pytest.mark.parametrize(
    "checker,combo_str",
    [
        (rule_two, "and()"),
        (rule_three, "or($1)"),
        (rule_four, "or(and(or($1 $2)) $3)"),
        (rule_five, "or(and($1 $2) and($1 $3))"),
        (rule_seven, "or($1 and($2 or($1 $3)))"),
    ],
)
def test_each_remaining_enf_checker_is_falsifiable(checker, combo_str):
    """Each checker must reject at least one direct counterexample.

    These cases make replacing any checker with ``lambda tr: True`` an
    observable test failure; effort-3 outputs alone are all valid ENF and
    therefore cannot establish that the checkers discriminate.
    """
    assert checker(str_to_combo_tree(combo_str)) is False


# ---------------------------------------------------------------------------
# Cross-cutting regression the ENF translation's own module docstring
# relies on: reduce_nots + subtree_to_enf never leave a `not(...)`
# wrapper in effort-2/3 output (negation is always a signed `argument`),
# which is why `enf_checkers.py`'s `_is_literal` only ever needs to check
# `is_argument`, never also unwrap a `not(argument)`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", [2, 3])
def test_no_residual_not_wrapper_in_enf_shaped_output(effort):
    for case in _TREES_BY_ID.values():
        tr = str_to_combo_tree(case["input"])
        logical_reduce(effort, tr, ignore_ops=set())
        assert "not(" not in combo_tree_to_str(tr), case["id"]
