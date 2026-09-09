"""Direct unit tests for pymetta.comboreduct.reduct.mixed_rules (S14
Task 4), covering the SIMPLE (no recursive `rule` argument) classes in
isolation. The recursive/assumption-based classes (`reduce_from_
assumptions`, `reduce_contin_if_not`, `reduce_gt_zero_sum`, `reduce_gt_
zero_prod`, `reduce_gt_zero_div`, `reduce_gt_zero_sum_sin`, `reduce_gt_
zero_sin`, `reduce_gt_division_of_constants`, and `reduce_inequality_
from_assumptions`'s own `operator()`) are covered end to end through the
real `mixed_reduction()` pipeline in `test_contin_mixed_reduction_oracle.py`
instead -- testing them in true isolation would need a throwaway
reduction rule anyway, and the pipeline-level oracle already measures the
real composition byte-identical against classic.

Every expected value below was captured against the compiled
`libcomboreduct.so` via `tools/cpp_probe/contin_rule_probe.cc`'s dispatch
extended locally for this file's own verification pass (not committed as
a fixture here -- the full-pipeline fixture in `test_contin_mixed_
reduction_oracle.py` already carries the durable, checked-in oracle
coverage; these are additional isolated-rule spot checks).
"""

from __future__ import annotations

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.mixed_rules import (
    _LexicographicSubtreeSet,
    _tree_swap,
    may_have_side_effects,
    reduce_contin_if,
    reduce_contin_if_inner_op,
    reduce_contin_if_to_impulse,
    reduce_gt_zero_const_div,
    reduce_gt_zero_exp,
    reduce_gt_zero_impulse,
    reduce_gt_zero_minus_exp,
    reduce_gt_zero_pair_power,
    reduce_gt_zero_times_const,
    reduce_impulse_power,
    reduce_impulse_prod,
    reduce_impulse_sum,
    reduce_junction_gt_zero_sum_constant,
    reduce_op_contin_if,
)


def _reduced(tree_str: str, r) -> str:
    tr = str_to_combo_tree(tree_str)
    r(tr, tr.begin())
    return combo_tree_to_str(tr).strip()


def test_may_have_side_effects_is_always_false():
    tr = str_to_combo_tree("$1")
    assert may_have_side_effects(tr.begin()) is False


# ---------------------------------------------------------------------------
# reduce_gt_zero_times_const / reduce_gt_zero_const_div (deprecated but ported)
# ---------------------------------------------------------------------------


def test_reduce_gt_zero_times_const_positive_constant():
    assert _reduced("0<(*(2.0 $1))", reduce_gt_zero_times_const()) == "0<(*($1))"


def test_reduce_gt_zero_times_const_negative_constant_flips_sign():
    assert _reduced("0<(*(-2.0 $1))", reduce_gt_zero_times_const()) == "0<(*(-1 $1))"


def test_reduce_gt_zero_times_const_zero_constant_is_false():
    assert _reduced("0<(*(0.0 $1))", reduce_gt_zero_times_const()) == "false"


def test_reduce_gt_zero_const_div_positive_constant():
    assert _reduced("0<(/(2.0 $1))", reduce_gt_zero_const_div()) == "0<($1)"


def test_reduce_gt_zero_const_div_negative_constant():
    assert _reduced("0<(/(-2.0 $1))", reduce_gt_zero_const_div()) == "0<(*(-1 $1))"


# ---------------------------------------------------------------------------
# reduce_gt_zero_exp / reduce_gt_zero_minus_exp
# ---------------------------------------------------------------------------


def test_reduce_gt_zero_exp_is_always_true():
    assert _reduced("0<(exp($1))", reduce_gt_zero_exp()) == "true"


def test_reduce_gt_zero_minus_exp_is_false():
    assert _reduced("0<(*(-1.0 exp($1)))", reduce_gt_zero_minus_exp()) == "false"


def test_reduce_gt_zero_minus_exp_positive_is_a_noop():
    assert _reduced("0<(*(1.0 exp($1)))", reduce_gt_zero_minus_exp()) == "0<(*(1 exp($1)))"


# ---------------------------------------------------------------------------
# reduce_gt_zero_pair_power
# ---------------------------------------------------------------------------


def test_reduce_gt_zero_pair_power_negative_even_powers_is_false():
    # 0 < -1 * x * x -> false (x appears twice: even parity)
    assert _reduced("0<(*(-1.0 $1 $1))", reduce_gt_zero_pair_power()) == "false"


def test_reduce_gt_zero_pair_power_odd_power_is_a_noop():
    # x appears only once: odd parity, rule cannot conclude anything
    assert _reduced("0<(*(-1.0 $1))", reduce_gt_zero_pair_power()) == "0<(*(-1 $1))"


def test_reduce_gt_zero_pair_power_positive_constant_is_a_noop():
    assert _reduced("0<(*(1.0 $1 $1))", reduce_gt_zero_pair_power()) == "0<(*(1 $1 $1))"


# ---------------------------------------------------------------------------
# reduce_gt_zero_impulse
# ---------------------------------------------------------------------------


def test_reduce_gt_zero_impulse_unwraps():
    assert _reduced("0<(impulse($1))", reduce_gt_zero_impulse()) == "$1"


def test_reduce_gt_zero_impulse_unwraps_compound_argument():
    assert _reduced("0<(impulse(and($1 $2)))", reduce_gt_zero_impulse()) == "and($1 $2)"


# ---------------------------------------------------------------------------
# reduce_impulse_power / reduce_impulse_prod / reduce_impulse_sum
# ---------------------------------------------------------------------------


def test_reduce_impulse_power_dedups_repeated_impulse():
    assert _reduced("*(impulse($1) impulse($1) $2)", reduce_impulse_power()) == "*(impulse($1) $2)"


def test_reduce_impulse_power_collapses_to_bare_impulse_when_sole_survivor():
    assert _reduced("*(impulse($1) impulse($1))", reduce_impulse_power()) == "impulse($1)"


def test_reduce_impulse_prod_merges_two_impulses():
    assert _reduced("*(impulse($1) impulse($2))", reduce_impulse_prod()) == "impulse(and($1 $2))"


def test_reduce_impulse_prod_merges_three_impulses():
    assert _reduced(
        "*(impulse($1) impulse($2) impulse($3))", reduce_impulse_prod()
    ) == "impulse(and($1 $3 $2))"


def test_reduce_impulse_prod_leaves_other_factors_alone():
    assert _reduced(
        "*(impulse($1) impulse($2) $3)", reduce_impulse_prod()
    ) == "*(impulse(and($1 $2)) $3)"


def test_reduce_impulse_sum_becomes_or():
    assert _reduced("0<(+(impulse($1) impulse($2)))", reduce_impulse_sum()) == "or($1 $2)"


def test_reduce_impulse_sum_ignores_mixed_terms():
    assert _reduced("0<(+(impulse($1) $2))", reduce_impulse_sum()) == "0<(+(impulse($1) $2))"


# ---------------------------------------------------------------------------
# reduce_contin_if_to_impulse / reduce_contin_if
# ---------------------------------------------------------------------------


def test_reduce_contin_if_to_impulse_true_false_branches():
    assert _reduced("contin_if($1 1.0 0.0)", reduce_contin_if_to_impulse()) == "impulse($1)"


def test_reduce_contin_if_to_impulse_false_true_branches_negates():
    assert _reduced("contin_if($1 0.0 1.0)", reduce_contin_if_to_impulse()) == "impulse(not($1))"


def test_reduce_contin_if_to_impulse_ignores_non_01_branches():
    assert _reduced("contin_if($1 2.0 0.0)", reduce_contin_if_to_impulse()) == "contin_if($1 2 0)"


def test_reduce_contin_if_true_condition():
    assert _reduced("contin_if(true $1 $2)", reduce_contin_if()) == "$1"


def test_reduce_contin_if_false_condition():
    assert _reduced("contin_if(false $1 $2)", reduce_contin_if()) == "$2"


def test_reduce_contin_if_nested_then_branch():
    assert (
        _reduced("contin_if($1 contin_if($1 $2 $3) $4)", reduce_contin_if())
        == "contin_if($1 $2 $4)"
    )


def test_reduce_contin_if_nested_else_branch():
    assert (
        _reduced("contin_if($1 $2 contin_if($1 $3 $4))", reduce_contin_if())
        == "contin_if($1 $2 $4)"
    )


def test_reduce_contin_if_nested_mismatched_condition_is_a_noop():
    tr_str = "contin_if($1 contin_if($2 $3 $4) $5)"
    assert _reduced(tr_str, reduce_contin_if()) == tr_str


# ---------------------------------------------------------------------------
# reduce_op_contin_if / reduce_contin_if_inner_op
# ---------------------------------------------------------------------------


def test_reduce_op_contin_if_div_of_two_matching_conditions():
    assert (
        _reduced("/(contin_if($1 $2 $3) contin_if($1 $4 $5))", reduce_op_contin_if())
        == "contin_if($1 /($2 $4) /($3 $5))"
    )


def test_reduce_op_contin_if_plus_merges_matching_conditions():
    assert (
        _reduced(
            "+(contin_if($1 $2 $3) contin_if($1 $4 $5))", reduce_op_contin_if()
        )
        == "contin_if($1 +($4 $2) +($5 $3))"
    )


def test_reduce_op_contin_if_mismatched_conditions_is_a_noop():
    tr_str = "/(contin_if($1 $2 $3) contin_if($6 $4 $5))"
    assert _reduced(tr_str, reduce_op_contin_if()) == tr_str


def test_reduce_contin_if_inner_op_shared_numerator():
    # contin_if(x /(y z1) /(y z2)) -> /(y contin_if(x z1 z2))
    assert (
        _reduced("contin_if($1 /($2 $3) /($2 $4))", reduce_contin_if_inner_op())
        == "/($2 contin_if($1 $3 $4))"
    )


def test_reduce_contin_if_inner_op_shared_denominator():
    # contin_if(x /(y1 z) /(y2 z)) -> /(contin_if(x y1 y2) z)
    assert (
        _reduced("contin_if($1 /($2 $4) /($3 $4))", reduce_contin_if_inner_op())
        == "/(contin_if($1 $2 $3) $4)"
    )


# ---------------------------------------------------------------------------
# reduce_junction_gt_zero_sum_constant
# ---------------------------------------------------------------------------


def test_reduce_junction_gt_zero_sum_constant_and_keeps_min():
    assert (
        _reduced(
            "and(0<(+($1 1.0)) 0<(+($1 2.0)))", reduce_junction_gt_zero_sum_constant()
        )
        == "and(0<(+($1 1)))"
    )


def test_reduce_junction_gt_zero_sum_constant_or_keeps_max():
    assert (
        _reduced(
            "or(0<(+($1 1.0)) 0<(+($1 2.0)))", reduce_junction_gt_zero_sum_constant()
        )
        == "or(0<(+($1 2)))"
    )


def test_reduce_junction_gt_zero_sum_constant_ignores_unrelated_vertex():
    tr_str = "not(0<(+($1 1.0)))"
    assert _reduced(tr_str, reduce_junction_gt_zero_sum_constant()) == "not(0<(+($1 1)))"


# ---------------------------------------------------------------------------
# _tree_swap / _LexicographicSubtreeSet -- direct unit + mutation tests
# ---------------------------------------------------------------------------


def test_tree_swap_adjacent_siblings():
    tr = str_to_combo_tree("+($1 $2 $3)")
    a = tr.begin().first_child
    b = a.next_sibling
    _tree_swap(a, b)
    assert combo_tree_to_str(tr).strip() == "+($2 $1 $3)"


def test_tree_swap_non_adjacent_nodes_across_different_parents():
    tr = str_to_combo_tree("+(*($1 $2) *($3 $4))")
    left_times = tr.begin().first_child
    right_times = left_times.next_sibling
    a = left_times.first_child  # $1, child of the left *
    b = right_times.first_child  # $3, child of the right *
    _tree_swap(a, b)
    assert combo_tree_to_str(tr).strip() == "+(*($3 $2) *($1 $4))"


def test_lexicographic_subtree_set_discard_one_vs_discard_all_equal():
    """PROVEN BY MUTATION: `reduce_contin_if_inner_op` needs `discard_one`
    (C++'s `std::multiset::erase(iterator)`, mixed_rules.cc:574);
    `reduce_fraction` (contin_rules.py) needs `discard_all_equal` (C++'s
    KEY-based `erase`, contin_rules.cc's own measured bug). Using the
    WRONG one for a given caller changes results whenever a duplicate
    exists -- demonstrated directly here rather than only asserted in
    each rule's own docstring.
    """
    tr1 = str_to_combo_tree("$1")
    tr2 = str_to_combo_tree("$1")
    tr3 = str_to_combo_tree("$1")

    s = _LexicographicSubtreeSet()
    s.add(tr1.begin())
    s.add(tr2.begin())
    s.add(tr3.begin())
    assert len(s) == 3

    probe = str_to_combo_tree("$1")
    s.discard_one(probe.begin())
    assert len(s) == 2  # exactly one removed

    s2 = _LexicographicSubtreeSet()
    s2.add(tr1.begin())
    s2.add(tr2.begin())
    s2.add(tr3.begin())
    s2.discard_all_equal(probe.begin())
    assert len(s2) == 0  # every equal element removed
