"""C++: tests/comboreduct/combo/interpreterUTest.cxxtest

Transliteration of interpreterUTest.cxxtest (112 lines), scoped to the
classes task-4-brief.md actually asks this task to implement:
`boolean_interpreter` and `interpreter_visitor`.

Test names are preserved 1:1 with the C++ file so a failure maps back to
its original. Only `test_boolean_interpreter` is a live port --
`test_contin_interpreter` and `test_mixed_interpreter` are kept as named,
explicitly skipped stubs: `contin_interpreter` and `mixed_interpreter` are
out of scope for the *entire* S0-S6 foundation cycle, not just this task
(docs/superpowers/plans/2026-09-03-pymetta-s0-s6-foundation.md's "Known gap,
deliberate" note scopes this cycle to the boolean scoring path only).
Skipping-by-name rather than omitting keeps the 1:1 mapping to the C++ file
visible in `pytest -v` output instead of silently vanishing.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.interpreter.interpreter import boolean_interpreter, interpreter_visitor

# tests/fixtures/, computed directly (not via the `fixture_dir` fixture)
# because @pytest.mark.parametrize below needs the corpus at collection
# time, before any fixture would run -- same rationale as
# test_iostream_combo.py's _ROUNDTRIP_CASES.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_EVAL_BOOLEAN_PATH = _FIXTURE_DIR / "eval_boolean.json"
_EVAL_BOOLEAN_CASES = (
    json.loads(_EVAL_BOOLEAN_PATH.read_text()) if _EVAL_BOOLEAN_PATH.exists() else []
)


# ---------------------------------------------------------------------------
# interpreterUTest.cxxtest, transliterated
# ---------------------------------------------------------------------------


def test_boolean_interpreter():
    """C++: tests/comboreduct/combo/interpreterUTest.cxxtest:11 test_boolean_interpreter"""
    tr = str_to_combo_tree("and(or($1 $2) not($3))")

    F = builtin.logical_false
    T = builtin.logical_true

    # (inputs, expected output) pairs, transliterated verbatim from
    # io1..io4 (interpreterUTest.cxxtest:20-31).
    irs = [
        ([F, F, F], F),
        ([F, T, T], F),
        ([F, T, F], T),
        ([T, F, F], T),
    ]

    for inputs, expected in irs:
        res = boolean_interpreter(inputs)(tr)
        assert res == expected


@pytest.mark.skip(
    reason="contin_interpreter is out of scope for the S0-S6 foundation cycle "
    "(boolean scoring path only); see task-4-brief.md and the plan's "
    "'Known gap, deliberate' note."
)
def test_contin_interpreter():
    """C++: tests/comboreduct/combo/interpreterUTest.cxxtest:44 test_contin_interpreter"""


@pytest.mark.skip(
    reason="mixed_interpreter is out of scope for the S0-S6 foundation cycle "
    "(boolean scoring path only); see task-4-brief.md and the plan's "
    "'Known gap, deliberate' note."
)
def test_mixed_interpreter():
    """C++: tests/comboreduct/combo/interpreterUTest.cxxtest:83 test_mixed_interpreter"""


# ---------------------------------------------------------------------------
# Unit tests beyond the interpreterUTest corpus -- empty and/or identities,
# argument negation, unsupported-builtin/vertex errors, interpreter_visitor
# dispatch (including its documented non-root-construction blind spot).
# ---------------------------------------------------------------------------


def test_logical_and_over_no_children_returns_the_and_sentinel():
    # C++: interpreter.cc:66-68 -- "For correct foldr behaviour" a
    # *childless* and() does not return id::logical_true: it returns the
    # id::logical_and operator token itself. This only behaves like the
    # AND identity (true) from the perspective of a PARENT and/or loop,
    # which never compares a child's result against anything but
    # id::logical_false (AND) or id::logical_true (OR) -- a sentinel that
    # is neither simply fails to trigger either short-circuit. Calling
    # boolean_eval directly on a bare and() (no parent to interpret the
    # sentinel) surfaces that raw token. Verified independently: eval-table
    # itself refuses to run a *whole program* that is just "and()" (it
    # fails combo type-checking before boolean_eval ever runs -- see
    # task-4-report.md) -- this is a real, if unusual, corner of the
    # interpreter, not a case classic MOSES's own CLI can exercise as a
    # top-level program either.
    tr = str_to_combo_tree("and()")
    assert boolean_interpreter([])(tr) == builtin.logical_and


def test_logical_or_over_no_children_returns_the_or_sentinel():
    # C++: interpreter.cc:75-77 -- symmetric to the and() case above: a
    # childless or() returns id::logical_or, not id::logical_false.
    tr = str_to_combo_tree("or()")
    assert boolean_interpreter([])(tr) == builtin.logical_or


def test_logical_and_identity_holds_when_nested_with_a_real_operand():
    # The practically meaningful form of the "and() over no children is
    # true" identity (CLAUDE.md): and($1 and()) equals plain $1, because
    # the AND-loop's own early-exit only fires on a literal
    # id::logical_false, and the nested empty and()'s sentinel is never
    # equal to that -- so the loop's outcome is driven entirely by the
    # real operand. Verified against eval-table: `and($x1 and())` produces
    # exactly the same column as `$x1` over trainingBoolean.dat (see
    # task-4-report.md).
    tr = str_to_combo_tree("and($1 and())")
    assert boolean_interpreter([builtin.logical_true])(tr) == builtin.logical_true
    assert boolean_interpreter([builtin.logical_false])(tr) == builtin.logical_false


def test_logical_or_identity_holds_when_nested_with_a_real_operand():
    # Symmetric to the and() case above: or($1 or()) equals plain $1.
    tr = str_to_combo_tree("or($1 or())")
    assert boolean_interpreter([builtin.logical_true])(tr) == builtin.logical_true
    assert boolean_interpreter([builtin.logical_false])(tr) == builtin.logical_false


def test_logical_true_and_false_constants_evaluate_to_themselves():
    assert boolean_interpreter([])(str_to_combo_tree("true")) == builtin.logical_true
    assert boolean_interpreter([])(str_to_combo_tree("false")) == builtin.logical_false


def test_positive_argument_returns_input_unchanged():
    tr = str_to_combo_tree("$1")
    assert boolean_interpreter([builtin.logical_true])(tr) == builtin.logical_true
    assert boolean_interpreter([builtin.logical_false])(tr) == builtin.logical_false


def test_negated_argument_returns_negation_of_input():
    # "!$1" parses to argument(-1) (a single leaf, is_negated() True) --
    # see test_iostream_combo.py's
    # test_negated_argument_is_distinct_from_a_not_node.
    tr = str_to_combo_tree("!$1")
    assert boolean_interpreter([builtin.logical_true])(tr) == builtin.logical_false
    assert boolean_interpreter([builtin.logical_false])(tr) == builtin.logical_true


def test_negated_third_argument_indexes_correctly():
    # idx = -3 -> boolean_inputs[-(-3) - 1] == boolean_inputs[2], negated.
    tr = str_to_combo_tree("!$3")
    inputs = [builtin.logical_true, builtin.logical_true, builtin.logical_false]
    assert boolean_interpreter(inputs)(tr) == builtin.logical_true


def test_unsupported_builtin_raises_naming_the_vertex():
    # C++: interpreter.cc:87-93 -- boolean_interpreter's default switch
    # case OC_ASSERTs out naming the offending builtin. `plus` is a real
    # builtin this interpreter never handles.
    tr = str_to_combo_tree("plus($1 $2)")
    with pytest.raises(ValueError, match="plus"):
        boolean_interpreter([builtin.logical_true, builtin.logical_true])(tr)


def test_unsupported_vertex_raises_naming_the_vertex():
    # C++: interpreter.cc:95-101 -- the `else` branch for a vertex that is
    # not even a builtin (here: a bare contin_t leaf).
    tr = str_to_combo_tree("3.14")
    with pytest.raises(ValueError, match="3.14"):
        boolean_interpreter([])(tr)


# ---------------------------------------------------------------------------
# interpreter_visitor (table.h:304)
# ---------------------------------------------------------------------------


def test_interpreter_visitor_dispatches_pure_boolean_tree_to_boolean_interpreter():
    tr = str_to_combo_tree("and(or($1 $2) not($3))")
    iv = interpreter_visitor(tr)
    inputs = [builtin.logical_false, builtin.logical_true, builtin.logical_false]
    assert iv(inputs) == builtin.logical_true


def test_interpreter_visitor_from_iterator_evaluates_that_subtree():
    tr = str_to_combo_tree("and($1 $2)")
    it = tr.begin()
    iv = interpreter_visitor(it)
    assert iv([builtin.logical_true, builtin.logical_true]) == builtin.logical_true
    assert iv([builtin.logical_true, builtin.logical_false]) == builtin.logical_false


def test_interpreter_visitor_marks_contin_expression_tree_as_mixed():
    # C++: table.h:306-317 -- walks the whole tree; `plus` anywhere makes
    # is_contin_expr true and sets mixed.
    tr = str_to_combo_tree("and($1 plus($2 $2))")
    iv = interpreter_visitor(tr)
    assert iv.mixed is True
    # mixed_interpreter is out of scope for this port (see module
    # docstring) -- interpreter_visitor must say so rather than silently
    # mishandling it.
    with pytest.raises(NotImplementedError):
        iv([builtin.logical_true, builtin.logical_true])


def test_interpreter_visitor_marks_greater_than_zero_tree_as_mixed():
    tr = str_to_combo_tree("0<($1)")
    iv = interpreter_visitor(tr)
    assert iv.mixed is True


def test_interpreter_visitor_constructed_from_non_root_iterator_can_miss_mixedness():
    # KNOWN SHARP EDGE, ported faithfully: table.h:319-324's iterator
    # constructor only inspects the single vertex `it` points at ("Cannot
    # check the entire tree with this ctor" -- the C++ comment itself).
    # Here `it` is the `and` node; the `plus` that would make this tree
    # `mixed` sits deeper in `and`'s own subtree, so the iterator ctor
    # misses it even though the whole-tree ctor (previous test) does not.
    tr = str_to_combo_tree("and($1 plus($2 $2))")
    it = tr.begin()  # the `and` node itself
    iv = interpreter_visitor(it)
    assert iv.mixed is False


def test_interpreter_visitor_rejects_non_builtin_input_sequences():
    # Only the vector<builtin> overload (table.h:328) is implemented in
    # this port; the vector<contin_t>/vector<vertex>/string_seq/
    # vector<combo_tree> overloads all unconditionally need
    # mixed_interpreter, which is out of scope here.
    tr = str_to_combo_tree("$1")
    iv = interpreter_visitor(tr)
    with pytest.raises(NotImplementedError):
        iv([1.0, 2.0])


# ---------------------------------------------------------------------------
# Differential oracle: Python boolean_interpreter vs. eval-table, over the
# Task 3 combo_roundtrip corpus crossed with trainingBoolean.dat /
# trainingBooleanXor.dat. See tools/capture_eval_boolean.py.
# ---------------------------------------------------------------------------


def test_golden_fixture_has_expected_shape():
    assert len(_EVAL_BOOLEAN_CASES) > 0
    for entry in _EVAL_BOOLEAN_CASES:
        assert set(entry) == {"dataset", "tree", "row", "expected"}
        assert isinstance(entry["dataset"], str)
        assert isinstance(entry["tree"], str)
        assert isinstance(entry["row"], list)
        assert isinstance(entry["expected"], str)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _EVAL_BOOLEAN_CASES,
    ids=lambda case: f"{case['dataset']}:{case['tree']}:{case['row']}",
)
def test_boolean_interpreter_matches_eval_table(case):
    inputs = [builtin.logical_true if x == 1 else builtin.logical_false for x in case["row"]]
    tr = str_to_combo_tree(case["tree"])
    expected = builtin.logical_true if case["expected"] == "1" else builtin.logical_false
    assert boolean_interpreter(inputs)(tr) == expected
