"""C++: tests/comboreduct/combo/interpreterUTest.cxxtest

Transliteration of interpreterUTest.cxxtest (112 lines). All three test
methods are now LIVE ports: `test_boolean_interpreter` (S4/S0-S6
foundation), `test_contin_interpreter` and `test_mixed_interpreter` (S14
Task 1 -- previously kept as named, explicitly-skipped stubs, since
`contin_interpreter`/`mixed_interpreter` were out of scope for the entire
S0-S6 foundation cycle).

Test names are preserved 1:1 with the C++ file so a failure maps back to
its original. `test_contin_interpreter`/`test_mixed_interpreter` are, in
particular, run UNCHANGED from classic's own hardcoded expected values
(computed there by real `log`/`exp`/`sin`, confirmed against a rebuilt
`interpreterUTest` binary on this machine -- see task report) -- this is
the S14 acceptance criterion "at least one test must run classic's own
defaults", not a configuration this port chose.
"""

import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import _contin_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, enum_t, is_boolean, vertex_to_bool
from pymetta.comboreduct.crutil.exception import OverflowException
from pymetta.comboreduct.interpreter.interpreter import (
    boolean_interpreter,
    contin_interpreter,
    interpreter_visitor,
    mixed_interpreter,
)
from pymetta.util.rng import MT19937RandGen, randGen

# tests/fixtures/, computed directly (not via the `fixture_dir` fixture)
# because @pytest.mark.parametrize below needs the corpus at collection
# time, before any fixture would run -- same rationale as
# test_iostream_combo.py's _ROUNDTRIP_CASES.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_EVAL_BOOLEAN_PATH = _FIXTURE_DIR / "eval_boolean.json"
_EVAL_BOOLEAN_CASES = (
    json.loads(_EVAL_BOOLEAN_PATH.read_text()) if _EVAL_BOOLEAN_PATH.exists() else []
)
_CONTIN_PROBE_PATH = _FIXTURE_DIR / "contin_probe.json"
_CONTIN_PROBE_CASES = (
    json.loads(_CONTIN_PROBE_PATH.read_text()) if _CONTIN_PROBE_PATH.exists() else {}
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


def test_contin_interpreter():
    """C++: tests/comboreduct/combo/interpreterUTest.cxxtest:44 test_contin_interpreter

    Transliterated verbatim, including the expected outputs: these are
    NOT recomputed by this port's own math, they are copied from the C++
    test's own hardcoded `o1`..`o6` literals (which the compiled test,
    rebuilt and rerun on this machine for this task, confirms it still
    produces via real `log`/`exp`/`sin` -- see task report). `math.log`/
    `math.exp`/`math.sin` are expected to match C's libm bit-for-bit on
    this platform (both ultimately call the same glibc); this test would
    fail loudly, not silently, if that assumption were ever wrong.
    """
    tr = str_to_combo_tree("+(*($1 $2) /($3 $4) log($5) exp($6) sin($7) 10.0)")

    irs = [
        ([1, 2, 0, 0, 1, 0, 0], 1 * 2 + 1 + 10.0),
        ([0, 0, 3, 4, 1, 0, 0], 3.0 / 4 + 1 + 10.0),
        ([0, 0, 0, 0, 5, 0, 0], math.log(5) + 1 + 10.0),
        ([0, 0, 0, 0, 1, 6, 0], math.exp(6) + 10.0),
        ([0, 0, 0, 0, 1, 0, 7], math.sin(7) + 1 + 10.0),
        (
            [1, 2, 3, 4, 5, 6, 7],
            1 * 2 + 3.0 / 4 + math.log(5) + math.exp(6) + math.sin(7) + 10.0,
        ),
    ]

    for inputs, expected in irs:
        res = contin_interpreter([float(x) for x in inputs])(tr)
        assert res == expected


def test_mixed_interpreter():
    """C++: tests/comboreduct/combo/interpreterUTest.cxxtest:83 test_mixed_interpreter

    Transliterated verbatim. `inputs_t = vector<vertex>` in C++ -- `i1`
    etc. mix bare numeric literals (contin_t, via boost::variant's
    implicit conversion) with `id::logical_true`/`id::logical_false`
    (builtin) in the SAME list, exactly the "mix contin_t and builtin in
    the same tree" case the S14 task brief calls for. This exercises the
    default `vector<vertex>` constructor (`mixed_interpreter(ir.first)`),
    the `plus`/`greater_than_zero`/`and`/`impulse`/`sin` dispatch chain,
    and (io2) a negative-valued (not negated-index) contin argument.
    """
    tr = str_to_combo_tree("sin(impulse(and(0<(+($1 $2)) $3)))")

    T = builtin.logical_true
    F = builtin.logical_false

    irs = [
        ([1.0, 2.0, T], math.sin(1.0)),
        ([1.0, -2.0, T], 0.0),
        ([1.0, 2.0, F], 0.0),
    ]

    for inputs, expected in irs:
        res = mixed_interpreter(inputs)(tr)
        assert res == expected


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
# Regression tests for the final review's Critical 2. This guard test
# passes only for a NON-integral contin like 3.14 above -- `3.14` never
# equals any `builtin` member's ordinal, so it was already routed
# correctly even before the fix. An INTEGRAL contin whose value equals a
# builtin's ordinal (logical_true=1, logical_false=2, logical_and=3,
# logical_or=4, logical_not=5 -- vertex.py) is what the bug actually
# needed: `match v` compared value patterns (`case builtin.logical_false
# | builtin.logical_true`, etc.) directly against `v` with no preceding
# `isinstance(v, builtin)` check, so `1.0 == builtin.logical_true` (True
# in Python, since `contin_t` is a bare `float` and `builtin` an
# `IntEnum`) silently misrouted these into the wrong branch. Verified
# directly against the real C++ (a bounded g++ probe linking
# libcomboreduct/libmoses/libcogutil, this fix wave's own
# final-fix-wave-report.md): C++ throws `AssertionException`
# ("boolean_interpreter does not handle vertex ...") for every one of
# these, via `boolean_interpreter::boolean_eval`'s `else` branch
# (interpreter.cc:95-101) -- `boost::variant` alternatives never compare
# equal across types, so a contin can never even enter C++'s `switch`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contin_literal", "colliding_builtin_name"),
    [
        ("1", "logical_true"),
        ("2", "logical_false"),
        ("3", "logical_and"),
        ("4", "logical_or"),
        ("5", "logical_not"),
    ],
)
def test_unsupported_integral_contin_vertex_raises_naming_the_vertex(
    contin_literal, colliding_builtin_name
):
    """Critical 2 regression. Before the fix, `boolean_interpreter([])
    (str_to_combo_tree("1"))` returned `1.0` (a `float`, violating the
    declared `-> builtin` contract) rather than raising; `"2"` returned
    `2.0`; `"3"`/`"4"`/`"5"` were evaluated as `logical_and`/`logical_or`/
    `logical_not` respectively (`"5"` -- `logical_not` -- then crashed
    trying to recurse into `it.first_child` of a childless contin leaf,
    `AttributeError: 'NoneType' object has no attribute 'data'`).
    """
    tr = str_to_combo_tree(contin_literal)
    # Sanity check on the fixture itself: confirms the parsed root really
    # is the colliding contin_t value this test means to exercise, not
    # something str_to_combo_tree's grammar happened to parse differently.
    assert tr.begin().data == float(contin_literal)
    with pytest.raises(ValueError, match=contin_literal) as exc_info:
        boolean_interpreter([])(tr)
    assert colliding_builtin_name not in str(exc_info.value)


def test_interpreter_visitor_iterator_sharp_edge_lets_integral_contin_reach_boolean_interpreter():
    """Critical 2 regression, the `interpreter_visitor(iterator)` sharp
    edge named in the final review: constructed from a bare iterator
    pointing at a node whose OWN vertex is not contin/greater_than_zero
    but which has a contin CHILD the shallow, single-node check cannot
    see (table.h:321-326's own comment: "Cannot check the entire tree
    with this ctor" -- see also
    test_interpreter_visitor_constructed_from_non_root_iterator_can_miss_mixedness
    above for the same sharp edge with a `plus` instead). `mixed` is
    wrongly `False` here, so `__call__` proceeds straight into
    `boolean_interpreter`, which must now raise on the `3` child rather
    than silently evaluating it as `logical_and` (3's colliding builtin).
    """
    tr = str_to_combo_tree("and($1 3)")
    it = tr.begin()  # the `and` node itself: not contin, not greater_than_zero
    iv = interpreter_visitor(it)
    assert iv.mixed is False  # the sharp edge: a whole-tree ctor would say True
    with pytest.raises(ValueError, match="3"):
        iv([builtin.logical_true])


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


def test_interpreter_visitor_routes_mixed_tree_with_boolean_inputs_to_mixed_interpreter():
    # S14 Task 1: `mixed_interpreter` is now ported, so a `vector<builtin>`
    # call against a `mixed` tree is a REAL evaluation
    # (`mixed_interpreter.from_boolean_inputs`, table.h:328-331), not a
    # NotImplementedError. `impulse` alone is `is_contin_expr` (mixed),
    # and this tree stays WELL-TYPED under boolean inputs: `impulse`'s
    # own argument is used only as a boolean, never handed to a genuine
    # contin operator (contrast the tree above, "and($1 plus($2 $2))",
    # where `plus` DOES read its argument as a contin -- see
    # test_interpreter_visitor_boolean_inputs_against_a_genuinely_contin_argument_raises
    # below for what that does once mixed_interpreter is real).
    tr = str_to_combo_tree("impulse($1)")
    iv = interpreter_visitor(tr)
    assert iv.mixed is True
    assert iv([builtin.logical_true]) == 1.0
    assert iv([builtin.logical_false]) == 0.0


def test_interpreter_visitor_boolean_inputs_against_a_genuinely_contin_argument_raises():
    # The real C++ behaviour this replaces the old NotImplementedError
    # expectation with: `mixed_interpreter(vector<builtin>)`'s argument
    # case returns a raw `builtin` for ANY argument (interpreter.cc:286),
    # even one a `plus` node then tries to use as a contin_t. In C++ that
    # is `get_contin(mixed_eval(it))` -> `boost::get<contin_t>` on a
    # variant actually holding a `builtin` -> `boost::bad_get`. This
    # port's `get_contin` raises `TypeError` for exactly the same reason
    # (vertex.py) -- a genuine type mismatch, not a scope gap.
    tr = str_to_combo_tree("and($1 plus($2 $2))")
    iv = interpreter_visitor(tr)
    assert iv.mixed is True
    with pytest.raises(TypeError):
        iv([builtin.logical_true, builtin.logical_true])


def test_interpreter_visitor_vector_contin_overload_always_uses_mixed_interpreter():
    # C++: table.h:332-336 -- `operator()(const vector<contin_t>&)` ALWAYS
    # constructs a `mixed_interpreter`, regardless of `mixed`, "since the
    # output might be non-contin, e.g. a boolean, or an enum." Here the
    # tree is not itself `mixed` ($1 is a bare argument), but the
    # vector<contin_t> overload is still the one that runs.
    tr = str_to_combo_tree("$1")
    iv = interpreter_visitor(tr)
    assert iv.mixed is False
    assert iv([3.0, 4.0]) == 3.0


def test_interpreter_visitor_vector_vertex_overload_handles_a_genuine_mix():
    # A heterogeneous vector<vertex> input (contin_t AND builtin in the
    # same sequence) -- unambiguous evidence this is routed to the
    # `vector<vertex>` overload (table.h:337-339), not one of the
    # homogeneous ones. Same tree/values as test_mixed_interpreter above.
    tr = str_to_combo_tree("sin(impulse(and(0<(+($1 $2)) $3)))")
    iv = interpreter_visitor(tr)
    assert iv.mixed is True
    assert iv([1.0, 2.0, builtin.logical_true]) == math.sin(1.0)


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


def test_interpreter_visitor_rejects_string_seq_input():
    # C++: table.h:341-343 -- `operator()(const string_seq&)` is
    # `OC_ASSERT(false, "Not implemented")` in classic ITSELF, not merely
    # out of scope for this port. `enum_t` is deliberately excluded from
    # this check (see interpreter_visitor.__call__'s own docstring) -- a
    # bare `str` here has to be a `definite_object`/`message` stand-in to
    # be unambiguous, hence the plain (non-enum_t) strings below.
    tr = str_to_combo_tree("$1")
    iv = interpreter_visitor(tr)
    with pytest.raises(NotImplementedError):
        iv(["a", "b"])


def test_interpreter_visitor_rejects_combo_tree_seq_input():
    # C++: table.h:344-350 -- `operator()(const vector<combo_tree>&)` is
    # likewise `OC_ASSERT(false, "Not implemented")` in classic itself.
    tr = str_to_combo_tree("$1")
    iv = interpreter_visitor(tr)
    with pytest.raises(NotImplementedError):
        iv([str_to_combo_tree("$1"), str_to_combo_tree("$2")])


# ---------------------------------------------------------------------------
# contin_interpreter / mixed_interpreter: unit tests beyond the
# interpreterUTest corpus -- foldr identities, overflow reconstruction,
# the UB-mirroring negative-index argument case, cond/contin_if, the
# definite_object passthrough, and unsupported builtin/vertex errors.
# ---------------------------------------------------------------------------


def test_contin_plus_over_no_children_returns_the_ordinal_as_a_contin_t():
    # C++: interpreter.cc:133-140 -- `return *b;` with `contin_eval`'s
    # return type being `contin_t` (double, NOT `builtin`) implicitly
    # converts the builtin enum -> int -> double, so a childless plus()
    # returns id::plus's ORDINAL (6) as a genuine contin_t 6.0 -- NOT a
    # symbolic sentinel the way boolean_interpreter's and()/or() childless
    # case is (there the return type really is `builtin`, so no
    # conversion happens). Measured directly against the real C++
    # (tools/cpp_probe/contin_interpreter_probe.cc,
    # "plus_over_no_children=VALUE:6"). `type(...) is float`, not merely
    # `== 6.0`, is asserted because `builtin.plus == 6.0` is ALSO True in
    # bare Python (IntEnum-vs-float numeric equality) -- a value-only
    # assertion would not have caught this port returning the wrong TYPE.
    tr = str_to_combo_tree("+()")
    res = contin_interpreter([])(tr)
    assert type(res) is float
    assert res == float(builtin.plus) == 6.0


def test_contin_times_over_no_children_returns_the_ordinal_as_a_contin_t():
    # Same mechanism as plus above (interpreter.cc:143-155); measured:
    # "times_over_no_children=VALUE:7" (id::times's ordinal).
    tr = str_to_combo_tree("*()")
    res = contin_interpreter([])(tr)
    assert type(res) is float
    assert res == float(builtin.times) == 7.0


def test_contin_plus_leaks_the_ordinal_into_a_nested_sum():
    # The practical consequence of the quirk above: plus($1 plus()) adds
    # 6.0 (NOT 0.0) into the sum -- this is a real, if surely unintended,
    # upstream behaviour, ported verbatim rather than "fixed" into a true
    # additive identity. Not directly probed (the probe only measures the
    # bare childless case), but it follows deterministically from the
    # same C++ conversion measured there, and Python's own IntEnum
    # arithmetic reproduces it via ordinary `+=` for the same reason.
    tr = str_to_combo_tree("+($1 +())")
    assert contin_interpreter([10.0])(tr) == 16.0


def test_contin_div_by_zero_raises_overflow_with_the_signed_infinity():
    # C++: interpreter.cc:157-168 -- x/0.0 with x>0 is IEEE754 +inf,
    # which the isnan/isinf check turns into a raised OverflowException.
    tr = str_to_combo_tree("/($1 $2)")
    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([1.0, 0.0])(tr)
    assert math.isinf(exc_info.value.get_vertex())
    assert exc_info.value.get_vertex() > 0

    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([-1.0, 0.0])(tr)
    assert math.isinf(exc_info.value.get_vertex())
    assert exc_info.value.get_vertex() < 0


def test_contin_div_by_zero_numerator_shortcuts_without_evaluating_denominator():
    # C++: interpreter.cc:161 -- "avoid pointless evals": if the
    # numerator is exactly 0.0, div returns 0.0 immediately without ever
    # evaluating the denominator at all. Proven here by feeding a
    # denominator that would itself raise if evaluated (log(0) would
    # raise OverflowException) -- if the shortcut did not fire, this test
    # would raise instead of returning 0.0.
    tr = str_to_combo_tree("/($1 log($2))")
    assert contin_interpreter([0.0, 0.0])(tr) == 0.0


def test_contin_log_of_zero_and_negative_raise_overflow():
    # C++: interpreter.cc:170-180 -- log(0)=-inf, log(negative)=nan in
    # IEEE754; both isnan/isinf-checked and raised as OverflowException.
    tr = str_to_combo_tree("log($1)")
    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([0.0])(tr)
    assert exc_info.value.get_vertex() == -math.inf

    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([-1.0])(tr)
    assert math.isnan(exc_info.value.get_vertex())


def test_contin_exp_overflow_raises():
    # C++: interpreter.cc:182-188 -- only isinf is checked (exp() of a
    # finite real is never nan).
    tr = str_to_combo_tree("exp($1)")
    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([1000.0])(tr)
    assert exc_info.value.get_vertex() == math.inf


def test_contin_sin_of_a_non_finite_input_returns_nan_not_raised():
    # C++: interpreter.cc:190-192 -- sin() has NO isnan/isinf check at
    # all; sin(inf) is silently nan in C, never raised. This is a real
    # DIVERGENCE at the math.sin() call (Python raises ValueError for a
    # non-finite argument), reconstructed by hand -- see the class's own
    # docstring. `$1` feeds inf directly through contin_inputs (the
    # argument case has no isnan/isinf guard, matching interpreter.cc:
    # 123-125 exactly), so this exercises the reconstruction, not the
    # contin-constant guard.
    tr = str_to_combo_tree("sin($1)")
    res = contin_interpreter([math.inf])(tr)
    assert math.isnan(res)


def test_contin_constant_nan_and_inf_raise_overflow_immediately():
    # C++: interpreter.cc:208-212 -- a bare contin_t LEAF (not an
    # argument) is isnan/isinf-checked directly.
    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([])(str_to_combo_tree("nan"))
    assert math.isnan(exc_info.value.get_vertex())

    with pytest.raises(OverflowException) as exc_info:
        contin_interpreter([])(str_to_combo_tree("inf"))
    assert exc_info.value.get_vertex() == math.inf


def test_contin_negative_argument_index_mirrors_python_negative_indexing_not_cpp_ub():
    # DO NOT FIX, see contin_interpreter.contin_eval's own docstring:
    # C++'s `contin_inputs[idx - 1]` has NO sign check (unlike
    # boolean_interpreter's argument case) -- a negative idx is
    # undefined behaviour in C++ (huge unsigned index), not a defined
    # negation. This port mirrors the literal expression, which for a
    # negative idx performs Python's OWN negative-index wraparound
    # instead -- a different, but at least deterministic, outcome.
    # `!$2` parses to argument(-2); `contin_inputs[-2-1] ==
    # contin_inputs[-3]`, which for this 3-element list is index 0.
    tr = str_to_combo_tree("!$2")
    assert contin_interpreter([10.0, 20.0, 30.0])(tr) == 10.0


def test_contin_unsupported_builtin_raises_naming_the_vertex():
    # C++: interpreter.cc:197-204 -- `cond` is a MIXED operator
    # (interpreter.cc requires mixed_interpreter for it), not a plain
    # contin one; contin_interpreter's own switch has no case for it.
    tr = str_to_combo_tree("cond($1 $2 $3)")
    with pytest.raises(ValueError, match="cond"):
        contin_interpreter([1.0, 2.0, 3.0])(tr)


def test_contin_unsupported_vertex_raises_naming_the_vertex():
    # C++: interpreter.cc:213-219 -- a definite_object/enum_t string
    # reaching contin_eval is neither argument, builtin nor contin_t.

    tr = combo_tree("some_definite_object")
    with pytest.raises(ValueError, match="some_definite_object"):
        contin_interpreter([])(tr)


@pytest.mark.parametrize(
    ("contin_literal", "colliding_builtin_name"),
    [
        ("6", "plus"),
        ("7", "times"),
        ("8", "div"),
        ("9", "exp"),
        ("10", "log"),
        ("11", "sin"),
        ("12", "greater_than_zero"),
        ("14", "rand"),
    ],
)
def test_contin_integral_collision_with_builtin_ordinal_does_not_misroute(
    contin_literal, colliding_builtin_name
):
    """The task brief's central trap, at contin_interpreter's own outer
    type-first `match` (this module's FIX WAVE comment, same shape as
    boolean_interpreter's own Critical-2 regression above): a bare
    contin_t leaf whose value happens to equal a builtin's ordinal (6.0
    == builtin.plus, 7.0 == builtin.times, ...) must be evaluated as the
    NUMBER it is, not misrouted into that builtin's arithmetic case --
    which for e.g. `div`/`log`/`sin` would immediately crash trying to
    recurse into `it.first_child` of a childless leaf. Proven by mutation
    (see task report): removing the outer `case builtin():` type guard
    and matching these values directly makes this test fail with an
    AttributeError instead of returning the plain float.
    """
    tr = str_to_combo_tree(contin_literal)
    assert tr.begin().data == float(contin_literal)
    res = contin_interpreter([])(tr)
    assert type(res) is float
    assert res == float(contin_literal)
    assert colliding_builtin_name not in repr(res)


def test_contin_rand_delegates_to_randgen_randfloat():
    # C++: interpreter.cc:194-195, `return randGen().randfloat();` -- ONE
    # draw off the process-wide singleton, and specifically `randfloat`
    # (binary32) rather than `randdouble`, which is a different value from
    # the same raw draw. Pinned by seeding the singleton and comparing
    # against an independently seeded generator, so a swap to randdouble or
    # an extra/missing draw both fail.
    tr = str_to_combo_tree("rand")
    randGen().seed(99)
    got = contin_interpreter([])(tr)
    expected = MT19937RandGen(99)
    assert got == expected.randfloat()
    assert got != MT19937RandGen(99).randdouble()


def test_mixed_greater_than_zero_recovers_the_sign_from_an_overflow():
    # C++: interpreter.cc:316-329's own comment: "A divide-by-zero in the
    # contin could throw. We want to return a boolean in this case,
    # anyway. Values might be +inf -inf or nan and we can still get a
    # sign bit off two of these cases..." -- proven here with the two
    # signed-infinity cases (nan has no sign bit to recover, so is not
    # exercised: `0 < nan` is False either way in both languages).
    tr_pos = str_to_combo_tree("0<(/(1.0 0.0))")
    assert mixed_interpreter([])(tr_pos) == builtin.logical_true

    tr_neg = str_to_combo_tree("0<(/(-1.0 0.0))")
    assert mixed_interpreter([])(tr_neg) == builtin.logical_false


def test_mixed_cond_takes_the_universal_else_when_the_predicate_is_false():
    # C++: interpreter.cc:337-359 -- "the very last value is the
    # universal else clause, taken when none of the previous predicates
    # were true." Ported via vertex_equal (interpreter.cc:354's bare
    # `vcond == id::logical_true` is exactly the "type before value"
    # shape this port never does bare).
    tr = str_to_combo_tree("cond($1 10.0 20.0)")
    assert mixed_interpreter([builtin.logical_true])(tr) == 10.0
    assert mixed_interpreter([builtin.logical_false])(tr) == 20.0


def test_mixed_cond_malformed_raises():
    # C++: interpreter.cc:341 -- `OC_ASSERT(sib != it.end(), "Error:
    # mal-formed cond statement")`, hit here with zero children at all.
    tr = str_to_combo_tree("cond()")
    with pytest.raises(AssertionError, match="mal-formed"):
        mixed_interpreter([])(tr)


def test_mixed_definite_object_passes_through_unevaluated():
    # C++: interpreter.cc:374-377 -- "string constant": a definite_object
    # (a bare, non-enum_t str -- see vertex.py's module docstring) is
    # returned as-is by mixed_eval, not evaluated further. str_to_combo_tree's
    # grammar has no token form for this (its own module docstring), so
    # the tree is built directly.

    tr = combo_tree("some_definite_object")
    assert mixed_interpreter([])(tr) == "some_definite_object"


def test_mixed_enum_t_is_neither_contin_nor_definite_object_and_raises():
    # Real, verified C++ behaviour, not a port gap: `enum_t` and
    # `definite_object` are DIFFERENT boost::variant alternatives, so an
    # enum_t reaching mixed_eval matches neither `is_contin` nor
    # `is_definite_object` and falls to the final OC_ASSERT(false)
    # (interpreter.cc:378-384).

    tr = combo_tree(enum_t("some_enum"))
    with pytest.raises(ValueError, match="some_enum"):
        mixed_interpreter([])(tr)


def test_mixed_unsupported_builtin_raises_naming_the_vertex():
    # C++: interpreter.cc:361-367 -- `list` is a real builtin this
    # port's builtin enum carries, but mixed_interpreter's switch has no
    # case for it.
    tr = str_to_combo_tree("list($1 $2)")
    with pytest.raises(ValueError, match="list"):
        mixed_interpreter([1.0, 2.0])(tr)


@pytest.mark.parametrize(
    ("contin_literal", "colliding_builtin_name"),
    [
        ("6", "plus"),
        ("11", "sin"),
        ("12", "greater_than_zero"),
    ],
)
def test_mixed_integral_collision_with_builtin_ordinal_does_not_misroute(
    contin_literal, colliding_builtin_name
):
    # Same trap as contin_interpreter's own regression above, at
    # mixed_eval's outer type-first match (interpreter.cc:294's own
    # `boost::get<builtin>(&v)`). Proven by mutation (see task report).
    tr = str_to_combo_tree(contin_literal)
    res = mixed_interpreter([])(tr)
    assert type(res) is float
    assert res == float(contin_literal)
    assert colliding_builtin_name not in repr(res)


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


# ---------------------------------------------------------------------------
# Differential oracle: Python contin_interpreter/mixed_interpreter vs. a
# direct C++ probe (tools/cpp_probe/contin_interpreter_probe.cc), for the
# edge cases (overflow reconstruction, the builtin->contin_t ordinal leak,
# greater_than_zero's OverflowException recovery, cond) that
# interpreterUTest.cxxtest's own three tests do not reach. See
# tools/capture_contin_probe.py.
# ---------------------------------------------------------------------------

# label -> (combo tree, contin_interpreter inputs, mixed_interpreter
# inputs) -- kept in lockstep with contin_interpreter_probe.cc's own
# case list by name; a label present in the fixture but missing here (or
# vice versa) fails loudly via the two shape-guard tests below rather
# than silently comparing fewer cases than the fixture actually has.
_CONTIN_PROBE_TREES: dict[str, tuple[str, list[float]]] = {
    "div_by_zero_positive_numerator": ("/($1 $2)", [1.0, 0.0]),
    "div_by_zero_negative_numerator": ("/($1 $2)", [-1.0, 0.0]),
    "div_zero_numerator_shortcuts_denominator": ("/($1 log($2))", [0.0, 0.0]),
    "log_of_zero": ("log($1)", [0.0]),
    "log_of_negative": ("log($1)", [-1.0]),
    "exp_overflow": ("exp($1)", [1000.0]),
    "sin_of_infinity_no_throw": ("sin($1)", [math.inf]),
    "contin_constant_nan_leaf": ("nan", []),
    "contin_constant_inf_leaf": ("inf", []),
    "plus_over_no_children": ("+()", []),
    "times_over_no_children": ("*()", []),
}

_MIXED_PROBE_TREES: dict[str, tuple[str, list]] = {
    "mixed_gtz_recovers_positive_infinity": ("0<(/(1.0 0.0))", []),
    "mixed_gtz_recovers_negative_infinity": ("0<(/(-1.0 0.0))", []),
    "mixed_cond_true_predicate": ("cond($1 10.0 20.0)", [builtin.logical_true]),
    "mixed_cond_false_predicate_falls_to_else": ("cond($1 10.0 20.0)", [builtin.logical_false]),
}


def test_contin_probe_fixture_has_expected_shape():
    assert len(_CONTIN_PROBE_CASES) > 0
    assert set(_CONTIN_PROBE_CASES) == set(_CONTIN_PROBE_TREES) | set(_MIXED_PROBE_TREES)
    for entry in _CONTIN_PROBE_CASES.values():
        assert set(entry) == {"status", "value"}
        assert entry["status"] in ("VALUE", "OVERFLOW")


def _parse_probe_contin(value: str) -> float:
    # C++'s default ostream formatting prints a negative NaN as "-nan";
    # math.isnan is sign-agnostic, and float("-nan") already parses fine
    # in Python, so no special-casing is needed here beyond a straight
    # float() call.
    return float(value)


@pytest.mark.oracle
@pytest.mark.parametrize("label", sorted(_CONTIN_PROBE_TREES), ids=sorted(_CONTIN_PROBE_TREES))
def test_contin_interpreter_matches_cpp_probe(label):
    case = _CONTIN_PROBE_CASES[label]
    tree, inputs = _CONTIN_PROBE_TREES[label]
    tr = str_to_combo_tree(tree)
    expected = _parse_probe_contin(case["value"])
    if case["status"] == "OVERFLOW":
        with pytest.raises(OverflowException) as exc_info:
            contin_interpreter(inputs)(tr)
        got = exc_info.value.get_vertex()
    else:
        got = contin_interpreter(inputs)(tr)
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected


@pytest.mark.oracle
@pytest.mark.parametrize("label", sorted(_MIXED_PROBE_TREES), ids=sorted(_MIXED_PROBE_TREES))
def test_mixed_interpreter_matches_cpp_probe(label):
    case = _CONTIN_PROBE_CASES[label]
    tree, inputs = _MIXED_PROBE_TREES[label]
    tr = str_to_combo_tree(tree)
    got = mixed_interpreter(inputs)(tr)
    # Every mixed case in this fixture is a VALUE, never an OVERFLOW --
    # greater_than_zero always resolves an OverflowException into a
    # plain boolean before it can propagate out of mixed_eval.
    assert case["status"] == "VALUE"
    if case["value"] in ("true", "false"):
        expected = builtin.logical_true if case["value"] == "true" else builtin.logical_false
    else:
        expected = _parse_probe_contin(case["value"])
    assert got == expected


# ---------------------------------------------------------------------------
# Differential oracle: Python interpreter_visitor/mixed_interpreter vs.
# eval-table, over a fixed 9-tree contin/mixed corpus against a SAMPLE of
# trainingRegression.dat's 8,000 rows (the S14 Task 1 plan's own named
# oracle). See tools/capture_eval_contin.py for the full rationale,
# including why every row here routes through mixed_interpreter -- even
# the purely-arithmetic trees -- exactly as classic's own OTable
# constructor does for an all-contin ITable (table.h:332-336).
# ---------------------------------------------------------------------------

_EVAL_CONTIN_PATH = _FIXTURE_DIR / "eval_contin.json"
_EVAL_CONTIN_CASES = (
    json.loads(_EVAL_CONTIN_PATH.read_text()) if _EVAL_CONTIN_PATH.exists() else []
)


def _format_like_table_fmt_vertex_to_str(v) -> str:
    """Not a ported symbol -- a minimal, local stand-in for
    `table_fmt_vertex_to_str` (table.cc:63, "Boolean values are output in
    binary form (0 for false, 1 for true)") built from vertex.py's own
    `is_boolean`/`vertex_to_bool` (added this task) plus
    `iostream_combo.py`'s existing `_contin_to_str`, since `ostreamTable`/
    `table_fmt_vertex_to_str` themselves are out of this task's scope
    (table.py's module docstring) -- this test only needs the FORMATTING
    convention, not a ported `OTable`-writing pipeline.
    """
    if is_boolean(v):
        return "1" if vertex_to_bool(v) else "0"
    return _contin_to_str(v)


def test_eval_contin_golden_fixture_has_expected_shape():
    assert len(_EVAL_CONTIN_CASES) > 0
    for entry in _EVAL_CONTIN_CASES:
        assert set(entry) == {"tree", "row_index", "row", "expected"}
        assert isinstance(entry["tree"], str)
        assert isinstance(entry["row_index"], int)
        assert isinstance(entry["row"], list)
        assert isinstance(entry["expected"], str)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _EVAL_CONTIN_CASES,
    ids=lambda case: f"{case['tree']}:{case['row_index']}",
)
def test_interpreter_visitor_matches_eval_table_over_training_regression(case):
    tr = str_to_combo_tree(case["tree"])
    iv = interpreter_visitor(tr)
    result = iv(case["row"])
    assert _format_like_table_fmt_vertex_to_str(result) == case["expected"]


# ---------------------------------------------------------------------------
# The same oracle, but through the REAL table_io.loadTable -> ITable
# pipeline instead of a hand-parsed CSV row -- closes the loop that
# table_io.py's own contin-column parsing (already in place before this
# task -- see task report's inventory) really does feed
# interpreter_visitor/mixed_interpreter identically to classic, not just
# that a bespoke float() parse does. tests/fixtures/table_data/
# regression_sample.csv is the SAME 40 sampled rows as eval_contin.json,
# with ORIGINAL tokens (tools/capture_eval_contin.py writes both from the
# one capture run), so row order lines up by position.
# ---------------------------------------------------------------------------


def test_loadtable_itable_rows_feed_interpreter_visitor_matching_eval_table(fixture_dir):
    from pymetta.comboreduct.table.table_io import loadTable

    csv_path = fixture_dir / "table_data" / "regression_sample.csv"
    table = loadTable(str(csv_path), "y")
    assert table.get_arity() == 2

    row_indices_in_fixture_order = sorted({c["row_index"] for c in _EVAL_CONTIN_CASES})
    assert len(table.itable) == len(row_indices_in_fixture_order)
    position_of_row_index = {
        row_idx: pos for pos, row_idx in enumerate(row_indices_in_fixture_order)
    }

    checked = 0
    for case in _EVAL_CONTIN_CASES:
        tr = str_to_combo_tree(case["tree"])
        iv = interpreter_visitor(tr)
        itable_row = list(table.itable[position_of_row_index[case["row_index"]]])
        result = iv(itable_row)
        assert _format_like_table_fmt_vertex_to_str(result) == case["expected"]
        checked += 1
    assert checked == len(_EVAL_CONTIN_CASES)


# ---------------------------------------------------------------------------
# Out-of-range argument index -- C++ has undefined behaviour here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", [3, 99, -3, -99])
def test_boolean_eval_rejects_an_out_of_range_argument(idx):
    """`interpreter.cc:54` indexes a `std::vector` with a bare
    `operator[]` and no bounds check, so an argument beyond the table's
    arity reads past the end -- undefined behaviour.

    Measured on a 2-input boolean table with `-e 'and($1 $99)'`: classic
    prints `0 and($1 $99)` and exits 0 (a PERFECT score for a tree it
    cannot evaluate), while the same command with `-W1` segfaults, both
    deterministically over five runs. A port cannot faithfully reproduce
    undefined behaviour, and reproducing the perfect score would be worse
    than failing -- it would silently rank an ill-formed exemplar first.

    So the port raises, and this pins the message: a bare `IndexError:
    list index out of range` from inside a recursive evaluator tells the
    user nothing.
    """
    inputs = [builtin.logical_true, builtin.logical_false]
    tree = str_to_combo_tree(f"${idx}" if idx > 0 else f"!${-idx}")
    with pytest.raises(IndexError, match=r"out of range for a 2-input row"):
        boolean_interpreter(inputs)(tree)


@pytest.mark.parametrize("idx", [1, 2, -1, -2])
def test_boolean_eval_accepts_every_in_range_argument(idx):
    """The other half: the guard must not fire on any valid index, in
    either sign. Without this, a guard that rejected everything would
    satisfy the test above.
    """
    inputs = [builtin.logical_true, builtin.logical_false]
    tree = str_to_combo_tree(f"${idx}" if idx > 0 else f"!${-idx}")
    assert boolean_interpreter(inputs)(tree) in (
        builtin.logical_true,
        builtin.logical_false,
    )
