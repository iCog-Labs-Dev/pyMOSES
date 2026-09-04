"""C++: moses/comboreduct/interpreter/eval.{h,cc}

No dedicated classic cxxtest exists for eval.cc (grep across $CLASSIC/tests
turns up nothing) -- `eval` has no direct oracle test of its own, unlike
interpreter.cc's interpreterUTest.cxxtest. These are hand-written unit
tests against eval.cc's own documented behaviour (argument binding, the
list/fold operators, contin arithmetic and its overflow handling, and the
raise-not-return-default contract for unbound arguments and unhandled
vertices), transliterated line-by-line from eval.cc's own logic rather
than from any C++ test file.
"""

import math

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.crutil.exception import EvalException, OverflowException
from pymetta.comboreduct.interpreter.eval import (
    arity,
    eval_binding,
    eval_procedure_tree,
    eval_throws_binding,
    eval_throws_tree,
    eval_throws_vertex,
    is_procedure_call,
)

F = builtin.logical_false
T = builtin.logical_true


# ---------------------------------------------------------------------------
# Argument binding -- eval.cc:119-125 (eval_throws_vertex's argument branch)
# ---------------------------------------------------------------------------


def test_positive_argument_binds_to_the_matching_bmap_entry():
    tr = str_to_combo_tree("$1")
    assert eval_binding([T], tr) == T
    assert eval_binding([F], tr) == F


def test_negative_argument_binds_negated():
    # C++: eval.cc:122-124 -- "when idx is negative, the argument is
    # necessarily boolean, and must be negated."
    tr = str_to_combo_tree("!$1")
    assert eval_binding([T], tr) == F
    assert eval_binding([F], tr) == T


def test_argument_selects_by_one_based_index_not_position_in_expression():
    tr = str_to_combo_tree("and($2 $1)")
    assert eval_binding([F, T], tr) == F  # $2 selects bmap[1]=T, $1 selects bmap[0]=F


def test_tree_with_no_arguments_needs_no_bmap():
    tr = str_to_combo_tree("and(true false)")
    assert eval_binding([], tr) == F


def test_unbound_argument_raises_rather_than_defaulting():
    # "Semantics to get right": unbound arguments must raise, not return a
    # default. C++'s vertex_seq is a std::vector<vertex>; bmap[idx-1] is
    # bounds-UNCHECKED operator[] (undefined behaviour on an out-of-range
    # idx, not a throw). Python's list[idx] raises IndexError for the same
    # out-of-range access -- strictly safer than C++'s UB, and satisfies
    # "must raise" without any extra bounds-check code of this port's own.
    tr = str_to_combo_tree("$2")
    with pytest.raises(IndexError):
        eval_throws_binding([T], tr)
    # eval_binding only rescues OverflowException/EvalException
    # (eval.cc:290-299) -- IndexError is neither, so it propagates through
    # eval_binding too, not silently swallowed into a default.
    with pytest.raises(IndexError):
        eval_binding([T], tr)


def test_unbound_negative_argument_also_raises():
    tr = str_to_combo_tree("!$3")
    with pytest.raises(IndexError):
        eval_throws_binding([T, F], tr)


# ---------------------------------------------------------------------------
# Logical operators -- eval.cc:130-153
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "bmap", "expected"),
    [
        ("and($1 $2)", [T, T], T),
        ("and($1 $2)", [T, F], F),
        ("or($1 $2)", [F, F], F),
        ("or($1 $2)", [F, T], T),
        ("not($1)", [T], F),
        ("not($1)", [F], T),
    ],
)
def test_logical_operators(expr, bmap, expected):
    assert eval_binding(bmap, str_to_combo_tree(expr)) == expected


def test_logical_and_over_no_children_returns_the_and_sentinel():
    # C++: eval.cc:136-137 -- "For correct foldr behaviour": empty and()
    # returns the id::logical_and token itself, not logical_true.
    assert eval_binding([], str_to_combo_tree("and()")) == builtin.logical_and


def test_logical_or_over_no_children_returns_the_or_sentinel():
    assert eval_binding([], str_to_combo_tree("or()")) == builtin.logical_or


# ---------------------------------------------------------------------------
# Mixed operators -- eval.cc:157-176
# ---------------------------------------------------------------------------


def test_greater_than_zero():
    assert eval_binding([], str_to_combo_tree("0<(1)")) == T
    assert eval_binding([], str_to_combo_tree("0<(-1)")) == F
    assert eval_binding([], str_to_combo_tree("0<(0)")) == F


def test_greater_than_zero_recovers_from_an_overflowing_operand():
    # C++: eval.cc:160-168 -- a divide-by-zero inside 0<(...) is caught
    # and its vertex payload (here, +inf) used as the operand's value
    # instead of propagating the exception.
    assert eval_binding([], str_to_combo_tree("0<(/(1 0))")) == T
    assert eval_binding([], str_to_combo_tree("0<(/(-1 0))")) == F


def test_impulse():
    assert eval_binding([], str_to_combo_tree("impulse(true)")) == 1.0
    assert eval_binding([], str_to_combo_tree("impulse(false)")) == 0.0


# ---------------------------------------------------------------------------
# Continuous operators -- eval.cc:179-250
# ---------------------------------------------------------------------------


def test_plus_and_times():
    assert eval_binding([], str_to_combo_tree("+(2 3 1)")) == 6.0
    assert eval_binding([], str_to_combo_tree("*(2 3 4)")) == 24.0


def test_plus_over_no_children_returns_the_plus_operator():
    # C++: eval.cc:180-182 -- "if plus does not have any argument, return
    # plus operator" (for correct foldr behaviour, same identity-sentinel
    # pattern as and()/or() above).
    assert eval_binding([], str_to_combo_tree("+()")) == builtin.plus


def test_times_over_no_children_returns_the_times_operator():
    assert eval_binding([], str_to_combo_tree("*()")) == builtin.times


def test_times_short_circuits_on_a_zero_operand():
    # C++: eval.cc:203 -- "avoid pointless evals": once the running
    # product hits 0.0, times() returns without evaluating the rest.
    # $1 is never bound, so evaluating it would raise IndexError -- if
    # times() failed to short-circuit, this would raise instead of
    # returning 0.0.
    assert eval_binding([], str_to_combo_tree("*(0 $1)")) == 0.0


def test_div():
    assert eval_binding([], str_to_combo_tree("/(6 3)")) == 2.0


def test_div_by_zero_overflows():
    # C++: eval.cc:213-220 -- x==0.0 short-circuits before the divide; a
    # genuine div-by-zero (x!=0, y==0) produces +-inf via IEEE754, checked
    # via isinf and raised as OverflowException. DIVERGENCE(C++ eval.cc:
    # 217): Python's float '/' raises ZeroDivisionError for a zero
    # divisor instead of returning a signed infinity (unlike C++'s IEEE754
    # division) -- eval.py must catch it and reconstruct the signed
    # infinity itself; this test is what pins that reconstruction down.
    with pytest.raises(OverflowException) as exc_info:
        eval_throws_binding([], str_to_combo_tree("/(1 0)"))
    assert math.isinf(exc_info.value.get_vertex())
    assert exc_info.value.get_vertex() > 0

    with pytest.raises(OverflowException) as exc_info:
        eval_throws_binding([], str_to_combo_tree("/(-1 0)"))
    assert exc_info.value.get_vertex() < 0


def test_div_by_zero_numerator_zero_short_circuits_before_dividing():
    # x==0.0 returns immediately (eval.cc:213), never reaching the y==0
    # divide at all -- 0/0 must NOT raise.
    assert eval_binding([], str_to_combo_tree("/(0 0)")) == 0.0


def test_eval_binding_recovers_the_overflow_vertex_instead_of_raising():
    # C++: eval.cc:290-299 -- eval_binding is the "never throws" entry
    # point: it catches OverflowException/EvalException and returns the
    # payload vertex instead.
    result = eval_binding([], str_to_combo_tree("/(1 0)"))
    assert math.isinf(result) and result > 0


def test_log_of_zero_overflows():
    # C++: eval.cc:223-233 -- log(0) is -inf in IEEE754 (Python's
    # math.log raises ValueError instead; see eval.py's DIVERGENCE note).
    with pytest.raises(OverflowException) as exc_info:
        eval_throws_binding([], str_to_combo_tree("log(0)"))
    assert exc_info.value.get_vertex() == -math.inf


def test_log_of_negative_overflows_as_nan():
    with pytest.raises(OverflowException) as exc_info:
        eval_throws_binding([], str_to_combo_tree("log(-1)"))
    assert math.isnan(exc_info.value.get_vertex())


def test_log_of_positive_is_ordinary():
    assert eval_binding([], str_to_combo_tree("log(1)")) == 0.0


def test_exp_overflows_on_a_large_argument():
    # C++: eval.cc:235-242 -- Python's math.exp raises OverflowError for a
    # too-large argument instead of returning +inf (see eval.py's
    # DIVERGENCE note); eval.cc only checks isinf here (exp is never nan).
    with pytest.raises(OverflowException) as exc_info:
        eval_throws_binding([], str_to_combo_tree("exp(10000)"))
    assert exc_info.value.get_vertex() == math.inf


def test_exp_ordinary():
    assert eval_binding([], str_to_combo_tree("exp(0)")) == 1.0


def test_sin_ordinary():
    assert eval_binding([], str_to_combo_tree("sin(0)")) == 0.0


def test_rand_is_p2_deferred():
    # C++: eval.cc:249-250 -- randGen().randfloat(). rng.py's randfloat()
    # is itself an already-established NotImplementedError stub (P2, see
    # its own docstring) -- eval.py must call straight through to it, not
    # work around the gap.
    with pytest.raises(NotImplementedError):
        eval_binding([], str_to_combo_tree("rand"))


def test_equ_is_not_handled_yet():
    # C++: eval.cc:254-257 -- OC_ASSERT(false, "The equ operator is not
    # handled yet..."). "equ" has no combo-text form (iostream_combo.py's
    # own docstring), so this is built directly, not via str_to_combo_tree.
    tr = combo_tree(builtin.equ)
    with pytest.raises(AssertionError):
        eval_throws_binding([], tr)


# ---------------------------------------------------------------------------
# List operators -- eval.cc:326-528
# ---------------------------------------------------------------------------


def test_list_of_constants():
    result = eval_throws_tree([], str_to_combo_tree("list(true false)"))
    assert [n.data for n in result] == [builtin.list, T, F]


def test_car_of_a_list():
    assert eval_throws_binding([], str_to_combo_tree("car(list(true false))")) == T


def test_car_evaluates_a_non_list_expression_first():
    # C++: eval.cc:346-349 -- if the argument isn't literally a list node
    # (here, a `cond` that evaluates to one), it is evaluated first, and
    # the RESULT must be a list.
    tr = str_to_combo_tree("car(cond(true list(true false) list(false true)))")
    assert eval_binding([], tr) == T


def test_car_of_an_empty_list_returns_the_empty_list():
    # C++: eval.cc:353-356.
    result = eval_throws_tree([], str_to_combo_tree("car(list())"))
    assert [n.data for n in result] == [builtin.list]


def test_car_of_a_non_list_raises():
    # C++: eval.cc:350-351 -- throw ComboException(TRACE_INFO, "not a list!").
    with pytest.raises(ValueError, match="not a list"):
        eval_throws_binding([], str_to_combo_tree("car(true)"))


def test_cdr_of_a_list():
    result = eval_throws_tree([], str_to_combo_tree("cdr(list(true false true))"))
    assert [n.data for n in result] == [builtin.list, F, T]


def test_cdr_of_a_singleton_list_is_the_empty_list():
    result = eval_throws_tree([], str_to_combo_tree("cdr(list(true))"))
    assert [n.data for n in result] == [builtin.list]


def test_cdr_of_a_non_list_raises():
    with pytest.raises(ValueError, match="not a list"):
        eval_throws_binding([], str_to_combo_tree("cdr(true)"))


def test_cons_prepends_to_a_list():
    result = eval_throws_tree([], str_to_combo_tree("cons(true list(false))"))
    assert [n.data for n in result] == [builtin.list, T, F]


def test_cons_over_no_children_returns_the_cons_operator():
    # C++: eval.cc:392-395.
    result = eval_throws_tree([], str_to_combo_tree("cons()"))
    assert [n.data for n in result] == [builtin.cons]


def test_foldr_base_case_returns_v_for_an_empty_list():
    # C++: eval.cc:414-421 -- foldr(f v list) = v when list is empty.
    result = eval_throws_tree([], str_to_combo_tree("foldr(and true list())"))
    assert [n.data for n in result] == [T]


def test_foldr_over_a_list():
    # foldr(and true list(true true false)) = and(true and(true and(false true)))
    result = eval_binding([], str_to_combo_tree("foldr(and true list(true true false))"))
    assert result == F


def test_foldl_base_case_returns_v_for_an_empty_list():
    result = eval_throws_tree([], str_to_combo_tree("foldl(and true list())"))
    assert [n.data for n in result] == [T]


def test_foldl_over_a_list():
    result = eval_binding([], str_to_combo_tree("foldl(and true list(true true false))"))
    assert result == F


# ---------------------------------------------------------------------------
# Control operators -- eval.cc:505-556
# ---------------------------------------------------------------------------


def test_lambda_returns_the_tree_unevaluated():
    # C++: eval.cc:506-509 -- lambda is a constructor, not evaluated.
    result = eval_throws_tree([], str_to_combo_tree("->($1 and($1 $1))"))
    assert [n.data for n in result] == [builtin.lambda_, argument(1), builtin.logical_and,
                                         argument(1), argument(1)]


def test_apply_is_not_implemented():
    # C++: eval.cc:513-514 -- OC_ASSERT(false, "apply() is not implemented").
    with pytest.raises(AssertionError):
        eval_throws_binding([], str_to_combo_tree("apply($1 true)"))


def test_cond_picks_the_first_true_branch():
    tr = str_to_combo_tree("cond(false 1 true 2 3)")
    assert eval_binding([], tr) == 2.0


def test_cond_falls_through_to_the_else_clause():
    tr = str_to_combo_tree("cond(false 1 false 2 3)")
    assert eval_binding([], tr) == 3.0


def test_cond_malformed_without_an_else_clause_raises():
    # C++: eval.cc:535 -- OC_ASSERT(sib != it.end(), "Error: mal-formed
    # cond statement"). combo-text can't express a bare, unparenthesized
    # trailing predicate, so this is built directly.
    tr = combo_tree(builtin.cond)
    tr.append_child(tr.begin(), F)
    tr.append_child(tr.begin(), 1.0)
    with pytest.raises(AssertionError):
        eval_throws_binding([], tr)


def test_contin_if_is_the_same_dispatch_as_cond():
    # C++: eval.cc:531-532 -- id::contin_if falls into the same case as
    # id::cond ("XXX TODO: contin_if should go away").
    tr = str_to_combo_tree("contin_if(true 1 2)")
    assert eval_binding([], tr) == 1.0


# ---------------------------------------------------------------------------
# Constants and unrecognised vertices -- eval.cc:264-283
# ---------------------------------------------------------------------------


def test_contin_constant():
    assert eval_binding([], str_to_combo_tree("3.5")) == 3.5


def test_nan_contin_constant_overflows():
    # C++: eval.cc:265-269 -- a contin_t LEAF that is itself nan/inf also
    # raises OverflowException, not just an operator RESULT that overflows.
    tr = combo_tree(math.nan)
    with pytest.raises(OverflowException):
        eval_throws_binding([], tr)


def test_unrecognized_vertex_raises_via_eval_throws_vertex_directly():
    # C++: eval.cc:275-281 -- a bare, non-enum_t string vertex (this
    # port's stand-in for definite_object/message, see vertex.py's module
    # docstring) is none of argument/builtin/contin_t/enum_type.
    # eval_throws_vertex is the only place this is actually visible as a
    # raised EvalException in practice: its only caller, eval_throws_tree,
    # catches it internally (eval.cc:576-582) and returns the offending
    # vertex wrapped in a combo_tree instead of propagating -- see the
    # next test.
    tr = combo_tree("some-definite-object")
    with pytest.raises(EvalException):
        eval_throws_vertex([], tr.begin())


def test_eval_throws_tree_swallows_eval_exception_and_returns_the_vertex():
    # C++: eval.cc:571-582 -- eval_throws_tree's own catch block never
    # lets EvalException escape; it wraps the offending vertex in a
    # single-node combo_tree instead. Consequently eval_throws_binding
    # and eval_binding never see one bubble through a normal call either.
    tr = combo_tree("some-definite-object")
    result = eval_throws_tree([], tr)
    assert [n.data for n in result] == ["some-definite-object"]
    assert eval_throws_binding([], tr) == "some-definite-object"
    assert eval_binding([], tr) == "some-definite-object"


# ---------------------------------------------------------------------------
# eval_throws_binding's whole-tree guard -- eval.cc:104-110
# ---------------------------------------------------------------------------


def test_eval_throws_binding_rejects_a_result_with_children():
    # list(...) evaluates to a combo_tree with children, not a lone
    # vertex -- eval_throws_binding's own OC_ASSERT (eval.cc:106-108)
    # forbids exactly this misuse.
    with pytest.raises(AssertionError):
        eval_throws_binding([], str_to_combo_tree("list(true false)"))


# ---------------------------------------------------------------------------
# eval_procedure_tree / is_procedure_call -- eval.cc:33-82
# ---------------------------------------------------------------------------


def test_is_procedure_call_is_always_false_in_this_port():
    # procedure_call is not one of this port's four vertex alternatives
    # (vertex.py's module docstring) -- see is_procedure_call's own
    # docstring for why this is a faithful consequence, not a fresh gap.
    assert is_procedure_call(T) is False
    assert is_procedure_call(argument(1)) is False
    assert is_procedure_call(1.0) is False


def test_eval_procedure_tree_always_raises_in_this_port():
    tr = str_to_combo_tree("true")
    with pytest.raises(ValueError, match="does not represent a combo procedure call"):
        eval_procedure_tree([], tr.begin())


# ---------------------------------------------------------------------------
# arity() -- eval.h:72-81
# ---------------------------------------------------------------------------


def test_arity_of_a_tree_with_no_arguments_is_zero():
    assert arity(str_to_combo_tree("and(true false)")) == 0


def test_arity_is_the_largest_absolute_argument_index():
    assert arity(str_to_combo_tree("and($1 !$3 $2)")) == 3
