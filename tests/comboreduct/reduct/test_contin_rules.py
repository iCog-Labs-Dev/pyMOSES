"""Tests for pymetta.comboreduct.reduct.contin_rules (S14 Task 4).

Expected outputs are hand-derived directly from `contin_rules.cc`'s own
algorithm (traced line by line). The highest-risk rules (`reduce_factorize`,
`reduce_factorize_fraction`, `reduce_times_div`, `reduce_log_div_times`,
`reduce_exp_log`, `reduce_sum_log`, `reduce_fraction`) are additionally
cross-checked against the real `libcomboreduct.so` -- see
`test_contin_rules_oracle.py` for that harness.
"""

from __future__ import annotations

import math

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.reduct.contin_rules import (
    _LexicographicSubtreeMap,
    _LexicographicSubtreeSet,
    reduce_exp_div,
    reduce_exp_log,
    reduce_exp_times,
    reduce_factorize,
    reduce_factorize_fraction,
    reduce_fraction,
    reduce_impulse_arg,
    reduce_invert_constant,
    reduce_log_div_times,
    reduce_plus_times_one_child,
    reduce_plus_zero,
    reduce_sin,
    reduce_sum_log,
    reduce_times_div,
    reduce_times_one_zero,
)


def _reduced(tree_str: str, r) -> str:
    tr = str_to_combo_tree(tree_str)
    r(tr, tr.begin())
    return combo_tree_to_str(tr).strip()


# ---------------------------------------------------------------------------
# reduce_plus_zero / reduce_times_one_zero
# ---------------------------------------------------------------------------


def test_reduce_plus_zero_drops_zero_terms():
    assert _reduced("+(0.0 $1 0.0)", reduce_plus_zero()) == "+($1)"


def test_reduce_plus_zero_all_zero_collapses_to_zero():
    assert _reduced("+(0.0 0.0)", reduce_plus_zero()) == "0"


def test_reduce_plus_zero_non_finite_child_collapses_to_zero_not_nan():
    """Proves the DO NOT FIX quirk: real C++ writes `FP_NAN` (=0), not an
    actual NaN.
    """
    tr = str_to_combo_tree("+(inf $1 $2)")
    reduce_plus_zero()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "0"
    assert tr.begin().data == 0.0
    assert not math.isnan(tr.begin().data)


def test_reduce_plus_zero_ignores_unrelated_vertex():
    assert _reduced("*($1 $2)", reduce_plus_zero()) == "*($1 $2)"


def test_reduce_times_one_zero_drops_one_factors():
    assert _reduced("*(1.0 $1 1.0)", reduce_times_one_zero()) == "*($1)"


def test_reduce_times_one_zero_zero_factor_collapses_to_zero():
    assert _reduced("*($1 0.0 $2)", reduce_times_one_zero()) == "0"


def test_reduce_times_one_zero_non_finite_collapses_to_one_not_infinity():
    tr = str_to_combo_tree("*(inf $1)")
    reduce_times_one_zero()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "1"
    assert tr.begin().data == 1.0
    assert not math.isinf(tr.begin().data)


# ---------------------------------------------------------------------------
# reduce_plus_times_one_child
# ---------------------------------------------------------------------------


def test_reduce_plus_times_one_child_collapses_plus():
    assert _reduced("+($1)", reduce_plus_times_one_child()) == "$1"


def test_reduce_plus_times_one_child_collapses_times_with_compound_child():
    assert _reduced("*(+($1 $2))", reduce_plus_times_one_child()) == "+($1 $2)"


def test_reduce_plus_times_one_child_ignores_multi_child():
    assert _reduced("+($1 $2)", reduce_plus_times_one_child()) == "+($1 $2)"


# ---------------------------------------------------------------------------
# reduce_invert_constant
# ---------------------------------------------------------------------------


def test_reduce_invert_constant_zero_over_x_is_zero():
    assert _reduced("/(0.0 $1)", reduce_invert_constant()) == "0"


def test_reduce_invert_constant_x_over_constant():
    assert _reduced("/($1 2.0)", reduce_invert_constant()) == "*($1 0.5)"


def test_reduce_invert_constant_x_over_zero_is_infinite_marker():
    tr = str_to_combo_tree("/($1 0.0)")
    reduce_invert_constant()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "1"


def test_reduce_invert_constant_x_over_c_times_y():
    # x/(2*y) -> 1/2 * x/y
    assert _reduced("/($1 *(2.0 $2))", reduce_invert_constant()) == "*(/($1 $2) 0.5)"


def test_reduce_invert_constant_x_over_c_times_y_z_multiple_factors():
    # x/(2*y*z) -> 1/2 * x/(y*z)
    assert _reduced(
        "/($1 *(2.0 $2 $3))", reduce_invert_constant()
    ) == "*(/($1 *($2 $3)) 0.5)"


def test_reduce_invert_constant_ignores_unrelated_vertex():
    assert _reduced("+($1 $2)", reduce_invert_constant()) == "+($1 $2)"


# ---------------------------------------------------------------------------
# reduce_fraction (measured multiset-erase-by-key bug)
# ---------------------------------------------------------------------------


def test_reduce_fraction_cancels_common_factor():
    assert _reduced("/(*($1 $2) *($1 $3))", reduce_fraction()) == "/(*($2) *($3))"


def test_reduce_fraction_single_denominator_factor():
    assert _reduced("/(*($1 $2) $1)", reduce_fraction()) == "/(*($2) 1)"


def test_reduce_fraction_no_common_factor_is_a_noop():
    assert _reduced("/(*($1 $2) *($3 $4))", reduce_fraction()) == "/(*($1 $2) *($3 $4))"


def test_reduce_fraction_duplicated_factor_only_cancels_once():
    """MEASURED (2026-09-09) against the real `libcomboreduct.so`
    (`downwards(reduce_fraction())` in isolation):

        before: /(*($1 $1 $2) *($1 $1))
        after:  /(*($1 $2) *($1))

    A naive full-cancellation reading would give `/($2 1)`; the real
    `std::multiset::erase(key)` bug leaves exactly one `$1` on each side.
    See `reduce_fraction`'s own docstring.
    """
    assert _reduced(
        "/(*($1 $1 $2) *($1 $1))", reduce_fraction()
    ) == "/(*($1 $2) *($1))"


def test_reduce_fraction_mutation_proof_full_cancellation_would_differ():
    """Proves the previous test actually discriminates: a hand-rolled
    "correct" single-erase-per-match reducer (using `discard_one` instead
    of the real `discard_all_equal`) fully cancels both copies -- a
    DIFFERENT, larger-cancellation result than the real port's output.
    """
    tr = str_to_combo_tree("/(*($1 $1 $2) *($1 $1))")
    it = tr.begin()
    numerator = it.first_child
    denominator = it.last_child
    nc = _LexicographicSubtreeSet()
    sib = numerator.first_child
    while sib is not None:
        nc.add(sib)
        sib = sib.next_sibling
    sib = denominator.first_child
    while sib is not None and len(nc):
        denom_el = sib
        sib = sib.next_sibling
        mirror_el = nc.find(denom_el)
        if mirror_el is not None:
            nc.discard_one(denom_el)  # the "fixed" single-erase variant
            tr.erase(mirror_el)
            if tr.is_childless(numerator):
                numerator.data = 1.0
            tr.erase(denom_el)
            if tr.is_childless(denominator):
                denominator.data = 1.0
    result = combo_tree_to_str(tr).strip()
    assert result == "/(*($2) 1)"
    assert result != "/(*($1 $2) *($1))"


# ---------------------------------------------------------------------------
# reduce_factorize / reduce_factorize_fraction
# ---------------------------------------------------------------------------


def test_reduce_factorize_common_factor_two_terms():
    # x*y + x*z -> x*(y+z)
    assert _reduced(
        "+(*($1 $2) *($1 $3))", reduce_factorize()
    ) == "+(*($1 +(*($2) *($3))))"


def test_reduce_factorize_bare_term_and_product_share_factor():
    # x + x*z -> x*(1+z)
    assert _reduced("+($1 *($1 $3))", reduce_factorize()) == "+(*($1 +(1 *($3))))"


def test_reduce_factorize_no_shared_factor_is_a_noop():
    assert _reduced("+(*($1 $2) *($3 $4))", reduce_factorize()) == "+(*($1 $2) *($3 $4))"


def test_reduce_factorize_childless_plus_raises():
    tr = str_to_combo_tree("$1")
    tr.begin().data = builtin.plus
    with pytest.raises(AssertionError):
        reduce_factorize()(tr, tr.begin())


def test_reduce_factorize_fraction_common_denominator():
    # x/z + y/z -> (x+y)/z
    assert _reduced(
        "+(/($1 $3) /($2 $3))", reduce_factorize_fraction()
    ) == "+(/(+($1 $2) $3))"


def test_reduce_factorize_fraction_no_shared_denominator_is_a_noop():
    assert _reduced(
        "+(/($1 $3) /($2 $4))", reduce_factorize_fraction()
    ) == "+(/($1 $3) /($2 $4))"


# ---------------------------------------------------------------------------
# reduce_times_div
# ---------------------------------------------------------------------------


def test_reduce_times_div_nested_div_numerator():
    # (x/y)/z -> x/(y*z)
    assert _reduced("/(/($1 $2) $3)", reduce_times_div()) == "/($1 *($3 $2))"


def test_reduce_times_div_nested_div_denominator():
    # x/(y/z) -> (x*z)/y
    assert _reduced("/($1 /($2 $3))", reduce_times_div()) == "/(*($1 $3) $2)"


def test_reduce_times_div_times_of_one_div():
    # x*(y/z) -> (x*y)/z
    assert _reduced("*($1 /($2 $3))", reduce_times_div()) == "/(*($1 $2) $3)"


def test_reduce_times_div_times_of_two_divs():
    # a*b*(c/d)*(e/f) -> (a*b*c*e)/(d*f)
    assert _reduced(
        "*($1 $2 /($3 $4) /($5 $6))", reduce_times_div()
    ) == "/(*($1 $2 $3 $5) *($4 $6))"


def test_reduce_times_div_ignores_plain_times():
    assert _reduced("*($1 $2)", reduce_times_div()) == "*($1 $2)"


# ---------------------------------------------------------------------------
# reduce_sum_log
# ---------------------------------------------------------------------------


def test_reduce_sum_log_merges_two_logs():
    # log(x) + log(y) -> log(x*y)
    assert _reduced("+(log($1) log($2))", reduce_sum_log()) == "+(log(*($1 $2)))"


def test_reduce_sum_log_subtracts_minus_log():
    # log(x) + (-1*log(y)) -> log(x/y)
    assert _reduced(
        "+(log($1) *(-1.0 log($2)))", reduce_sum_log()
    ) == "+(log(/($1 $2)))"


def test_reduce_sum_log_no_log_present_is_a_noop():
    assert _reduced("+($1 $2)", reduce_sum_log()) == "+($1 $2)"


def test_reduce_sum_log_leaves_a_bare_times_minus_log_alone_when_not_matching():
    # a -1*x term where x is NOT log() should not trigger the minus-log path
    assert _reduced("+(log($1) *(-1.0 $2))", reduce_sum_log()) == "+(log($1) *(-1 $2))"


# ---------------------------------------------------------------------------
# reduce_log_div_times
# ---------------------------------------------------------------------------


def test_reduce_log_div_times_log_of_constant_over_x():
    # log(2/x) -> -log(0.5*x)
    assert _reduced("log(/(2.0 $1))", reduce_log_div_times()) == "*(log(*(0.5 $1)) -1)"


def test_reduce_log_div_times_log_of_zero_over_x_is_infinite_marker():
    tr = str_to_combo_tree("log(/(0.0 $1))")
    reduce_log_div_times()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "1"


def test_reduce_log_div_times_log_of_exp_collapses():
    assert _reduced("log(exp($1))", reduce_log_div_times()) == "$1"


def test_reduce_log_div_times_log_of_exp_times_y():
    # log(exp(x)*y) -> x + log(y)
    assert _reduced(
        "log(*(exp($1) $2))", reduce_log_div_times()
    ) == "+(log(*($2)) $1)"


def test_reduce_log_div_times_all_exp_factors_drops_log_entirely():
    # log(exp(x)*exp(y)) -> x + y (the leftover log(*()) term vanishes)
    assert _reduced(
        "log(*(exp($1) exp($2)))", reduce_log_div_times()
    ) == "+($1 $2)"


def test_reduce_log_div_times_ignores_unrelated_vertex():
    assert _reduced("exp($1)", reduce_log_div_times()) == "exp($1)"


# ---------------------------------------------------------------------------
# reduce_exp_times / reduce_exp_div / reduce_exp_log
# ---------------------------------------------------------------------------


def test_reduce_exp_times_merges_two_exps():
    assert _reduced("*(exp($1) exp($2))", reduce_exp_times()) == "*(exp(+($1 $2)))"


def test_reduce_exp_times_merges_three_exps_and_other_factor():
    assert _reduced(
        "*(exp($1) $3 exp($2))", reduce_exp_times()
    ) == "*(exp(+($1 $2)) $3)"


def test_reduce_exp_times_single_exp_is_a_noop():
    assert _reduced("*(exp($1) $2)", reduce_exp_times()) == "*(exp($1) $2)"


def test_reduce_exp_div_bare_exponent():
    assert _reduced("/($1 exp($2))", reduce_exp_div()) == "*($1 exp(*($2 -1)))"


def test_reduce_exp_div_times_exponent():
    assert _reduced("/($1 exp(*($2 $3)))", reduce_exp_div()) == "*($1 exp(*($2 -1 $3)))"


def test_reduce_exp_div_ignores_non_exp_denominator():
    assert _reduced("/($1 $2)", reduce_exp_div()) == "/($1 $2)"


def test_reduce_exp_log_single_log():
    assert _reduced("exp(log($1))", reduce_exp_log()) == "$1"


def test_reduce_exp_log_log_plus_y():
    # exp(log(x)+y) -> x*exp(y)
    assert _reduced("exp(+(log($1) $2))", reduce_exp_log()) == "*(exp(+($2)) $1)"


def test_reduce_exp_log_two_logs_plus_y():
    # exp(log(x)+log(w)+y) -> x*w*exp(y)
    assert _reduced(
        "exp(+(log($1) log($3) $2))", reduce_exp_log()
    ) == "*(exp(+($2)) $1 $3)"


def test_reduce_exp_log_all_logs_drops_exp_entirely():
    # exp(log(x)+log(w)) -> x*w (no leftover exp() term)
    assert _reduced("exp(+(log($1) log($2)))", reduce_exp_log()) == "*($1 $2)"


def test_reduce_exp_log_ignores_unrelated_vertex():
    assert _reduced("log($1)", reduce_exp_log()) == "log($1)"


# ---------------------------------------------------------------------------
# reduce_sin
# ---------------------------------------------------------------------------


def test_reduce_sin_folds_constant_into_range():
    tr = str_to_combo_tree("sin(+($1 4.0))")
    reduce_sin()(tr, tr.begin())
    const_val = tr.begin().first_child.last_child.data
    expected = math.fmod(4.0 + math.pi, 2.0 * math.pi) - math.pi
    assert const_val == pytest.approx(expected)


def test_reduce_sin_leaves_in_range_constant_alone():
    tr = str_to_combo_tree("sin(+($1 1.0))")
    reduce_sin()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "sin(+($1 1))"


def test_reduce_sin_merges_multiple_constants_first():
    tr = str_to_combo_tree("sin(+($1 1.0 2.0))")
    reduce_sin()(tr, tr.begin())
    # 1.0 + 2.0 = 3.0, already in (-pi, pi], so left as-is once merged
    assert combo_tree_to_str(tr).strip() == "sin(+($1 3))"


def test_reduce_sin_non_finite_constant_collapses_to_zero_not_nan():
    tr = str_to_combo_tree("sin(inf)")
    reduce_sin()(tr, tr.begin())
    assert combo_tree_to_str(tr).strip() == "0"
    assert not math.isnan(tr.begin().data)


def test_reduce_sin_bare_constant_argument():
    tr = str_to_combo_tree("sin(4.0)")
    reduce_sin()(tr, tr.begin())
    expected = math.fmod(4.0 + math.pi, 2.0 * math.pi) - math.pi
    assert tr.begin().first_child.data == pytest.approx(expected)


def test_reduce_sin_ignores_unrelated_vertex():
    assert _reduced("exp($1)", reduce_sin()) == "exp($1)"


# ---------------------------------------------------------------------------
# reduce_impulse_arg
# ---------------------------------------------------------------------------


def test_reduce_impulse_arg_reduces_boolean_argument():
    r = reduce_impulse_arg(1, set())
    assert _reduced("impulse(not(not($1)))", r) == "impulse($1)"


def test_reduce_impulse_arg_ignores_unrelated_vertex():
    r = reduce_impulse_arg(1, set())
    assert _reduced("exp($1)", r) == "exp($1)"


# ---------------------------------------------------------------------------
# _LexicographicSubtreeMap / _LexicographicSubtreeSet -- direct unit tests
# ---------------------------------------------------------------------------


def test_lexicographic_subtree_map_find_or_insert():
    tr1 = str_to_combo_tree("+($1 $2)")
    tr2 = str_to_combo_tree("+($1 $2)")  # structurally equal, different object
    m = _LexicographicSubtreeMap()
    m.insert(tr1.begin(), "first")
    entry = m.find(tr2.begin())
    assert entry is not None
    assert entry[0] is tr1.begin()
    assert entry[1] == "first"


def test_lexicographic_subtree_map_distinguishes_different_shapes():
    tr1 = str_to_combo_tree("+($1 $2)")
    tr2 = str_to_combo_tree("+($1 $3)")
    m = _LexicographicSubtreeMap()
    m.insert(tr1.begin(), "first")
    assert m.find(tr2.begin()) is None


def test_lexicographic_subtree_set_contains_and_discard():
    tr1 = str_to_combo_tree("$1")
    tr2 = str_to_combo_tree("$1")
    s = _LexicographicSubtreeSet()
    s.add(tr1.begin())
    assert s.contains(tr2.begin())
    removed = s.discard_one(tr2.begin())
    assert removed is tr1.begin()
    assert not s.contains(tr2.begin())
