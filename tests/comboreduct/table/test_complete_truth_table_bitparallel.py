"""Port-only tests, no C++ counterpart: the bit-parallel evaluator behind
`complete_truth_table` (table.py's `_bitparallel_truth_table`) must give
exactly the table the per-row `boolean_interpreter` loop gives
(table.h:1495's `populate`, ported as the fallback in `_populate`), and
must hand every tree it does not fully support back to that loop.
"""

from __future__ import annotations

import random

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import bool_to_vertex, builtin_to_bool
from pymoses.comboreduct.interpreter.interpreter import boolean_interpreter
from pymoses.comboreduct.table.table import (
    _bitparallel_columns,
    _bitparallel_truth_table,
    complete_truth_table,
)


def _per_row_table(tr, arity):
    """The pre-optimisation `_populate` loop, verbatim."""
    out = [False] * (2**arity)
    for i in range(len(out)):
        inputs = [bool_to_vertex(bool((i >> j) % 2)) for j in range(arity)]
        out[i] = builtin_to_bool(boolean_interpreter(inputs)(tr))
    return out


def _random_tree_str(rng: random.Random, arity: int, depth: int) -> str:
    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.08:
            return rng.choice(["true", "false"])
        k = rng.randint(1, arity)
        return f"!${k}" if rng.random() < 0.3 else f"${k}"
    op = rng.choice(["and", "or", "and", "or", "not"])
    if op == "not":
        return f"not({_random_tree_str(rng, arity, depth - 1)})"
    n = rng.randint(1, 4)
    kids = " ".join(_random_tree_str(rng, arity, depth - 1) for _ in range(n))
    return f"{op}({kids})"


def test_columns_follow_populate_bit_order():
    for arity in range(0, 7):
        columns, full = _bitparallel_columns(arity)
        assert full == (1 << (1 << arity)) - 1
        for j, col in enumerate(columns):
            for i in range(1 << arity):
                assert bool((col >> i) & 1) == bool((i >> j) % 2)


@pytest.mark.parametrize("seed", range(40))
def test_random_boolean_trees_give_identical_tables(seed):
    rng = random.Random(seed)
    arity = rng.randint(0, 7) if seed % 5 else 11
    if arity == 0:
        tr_str = rng.choice(["true", "false", "and(true not(false))", "or(false false)"])
    else:
        tr_str = _random_tree_str(rng, arity, rng.randint(1, 6))
    tr = str_to_combo_tree(tr_str)
    expected = _per_row_table(tr, arity)
    fast = _bitparallel_truth_table(tr, arity)
    assert fast is not None, tr_str
    assert fast == expected, tr_str
    assert all(type(b) is bool for b in fast)
    tt = complete_truth_table(tr, arity)
    assert list(tt) == expected
    assert len(tt) == 2**arity


@pytest.mark.parametrize(
    "tr_str",
    [
        "and($1 $1 not($1))",
        "or($1 !$1)",
        "not(not(not($2)))",
        "and(or($1 $2 $3) or(!$1 $3) not(and($2 $2)))",
        "and(true $1)",
        "or(false $2)",
        "and($3)",
    ],
)
def test_handwritten_trees_and_inferred_arity(tr_str):
    tr = str_to_combo_tree(tr_str)
    tt = complete_truth_table(tr)  # arity inferred from the tree
    assert list(tt) == _per_row_table(tr, tt._arity)


def test_same_complete_truth_table_matches_per_row_answer():
    rng = random.Random(7)
    for _ in range(200):
        arity = rng.randint(1, 5)
        a = str_to_combo_tree(_random_tree_str(rng, arity, 3))
        b = str_to_combo_tree(_random_tree_str(rng, arity, 3))
        tt = complete_truth_table(a, arity)
        expected = list(tt) == _per_row_table(b, arity)
        assert tt.same_complete_truth_table(b) is expected
        assert tt.same_complete_truth_table(a) is True


@pytest.mark.parametrize(
    "tr_str",
    [
        "or(and() $1)",  # childless and: evaluates to the builtin itself
        "and(or() $1)",
        "0<($1)",  # contin predicate
        "and(0<(+($1 $2)) $1)",
    ],
)
def test_unsupported_trees_fall_back(tr_str):
    tr = str_to_combo_tree(tr_str)
    assert _bitparallel_truth_table(tr, 2) is None
    try:
        expected = _per_row_table(tr, 2)
    except Exception as exc:  # the fallback must raise exactly as before
        with pytest.raises(type(exc)):
            complete_truth_table(tr, 2)
    else:
        assert list(complete_truth_table(tr, 2)) == expected


def test_out_of_range_argument_falls_back_and_still_raises():
    tr = str_to_combo_tree("and($1 $5)")
    assert _bitparallel_truth_table(tr, 2) is None
    with pytest.raises(IndexError):
        complete_truth_table(tr, 2)
    tt = complete_truth_table(str_to_combo_tree("and($1 $2)"), 2)
    with pytest.raises(IndexError):
        tt.same_complete_truth_table(tr)
