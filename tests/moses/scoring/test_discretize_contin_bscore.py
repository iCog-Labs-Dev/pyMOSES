"""Unit tests for `discretize_contin_bscore` (`bscores.cc:196`).

`tests/moses/main/test_discretize_threshold.py` covers it end to end against
classic's binary. These pin three behaviours that the CLI corpus provably
does NOT reach — each was found by mutation: reverting it left all eight
oracle cases green.

That is the distinction this project cares about. A zero-kill mutation means
either no coverage or no real trap, and all three here were the former.
"""

import struct

import pytest

from pymoses.comboreduct.table.table import ITable, OTable, type_node
from pymoses.moses.scoring.bscores import discretize_contin_bscore


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _tables(outputs, n_inputs: int = 1):
    it = ITable([type_node.contin_type] * n_inputs, [f"x{i + 1}" for i in range(n_inputs)])
    for i in range(len(outputs)):
        it.push_back([float(i)] * n_inputs)
    ot = OTable(list(outputs), "y")
    return ot, it


def test_thresholds_are_sorted_by_the_constructor():
    """`boost::sort(thresholds)` (bscores.cc:204) sorts the MEMBER, after it
    was copy-initialised from the argument. Every CLI case passes `-R` values
    already in ascending order, so nothing there can tell a sorted member
    from an unsorted one.
    """
    ot, it = _tables([0.0, 1.0, 2.0])
    unsorted = [2.0, 0.5]
    sc = discretize_contin_bscore(ot, it, unsorted)

    assert sc._thresholds == [0.5, 2.0]
    # The caller's list is untouched -- C++ sorts its own copy.
    assert unsorted == [2.0, 0.5]
    # And classification follows the SORTED order: 1.0 sits between them.
    assert sc.class_idx(0.0) == 0
    assert sc.class_idx(1.0) == 1
    assert sc.class_idx(2.0) == 2


def test_class_idx_upper_bound_is_inclusive():
    """`if (v >= thresholds[s - 1]) return s;` (bscores.cc:238) -- a value
    EXACTLY equal to the last threshold belongs to the top class, not the one
    below. Contin data from a real table essentially never lands exactly on a
    threshold, so the CLI corpus cannot distinguish `>=` from `>`.
    """
    ot, it = _tables([0.0, 1.0])
    sc = discretize_contin_bscore(ot, it, [0.5, 1.5])

    assert sc.class_idx(1.5) == 2, "a value equal to the last threshold is the top class"
    assert sc.class_idx(1.4999999) == 1
    # The lower bound is exclusive by contrast (`v < thresholds[0]`).
    assert sc.class_idx(0.5) == 1
    assert sc.class_idx(0.4999999) == 0


def test_weighted_accuracy_narrows_each_weight_to_float32():
    """`classes.size() / (float)(weights.size() * cs.count(i))`
    (bscores.cc:212) is a FLOAT divide, so each weight is binary32.

    Chosen so the narrowing is observable: 3 rows, 3 classes, 1 row per
    class gives 3/(3*1) = 1.0 for each -- exact, and useless as a test. Two
    rows in one class and none in another is what produces a non-representable
    quotient, and an empty class divides by zero, which C++ does in float and
    yields +inf rather than trapping.
    """
    # 3 rows, thresholds cut into 3 classes; rows land 2 in class 0, 1 in
    # class 2, and NONE in class 1.
    ot, it = _tables([0.0, 0.1, 5.0])
    sc = discretize_contin_bscore(ot, it, [1.0, 2.0], True)

    assert sc._classes == [0, 0, 2]
    # class 0: 3/(3*2) = 0.5 -- exactly representable either way.
    assert sc._weights[0] == pytest.approx(0.5)
    # class 1 is EMPTY: C++ divides by zero in float and gets +inf.
    assert sc._weights[1] == float("inf")
    # class 2: 3/(3*1) = 1.0.
    assert sc._weights[2] == pytest.approx(1.0)

    # And the narrowing itself, on a quotient that is not representable in
    # binary32: 3 rows over 3 classes with 3 in one class gives 1/3.
    ot2, it2 = _tables([0.0, 0.1, 0.2])
    sc2 = discretize_contin_bscore(ot2, it2, [1.0, 2.0], True)
    assert sc2._classes == [0, 0, 0]
    assert sc2._weights[0] == _f32(3 / (3 * 3))
    assert sc2._weights[0] != 3 / (3 * 3), (
        "this assertion is the whole point: float64 and float32 must differ here, "
        "or the narrowing is untested"
    )


def test_a_perfect_candidate_scores_all_zeros():
    """Unlike every other bscore in this file, a perfect candidate here
    scores ZERO and a bad one scores POSITIVE -- `best_possible_bscore`
    returns all zeros (bscores.cc:222-225). Pinned because it inverts the
    convention a reader would assume.
    """
    ot, it = _tables([0.0, 5.0])
    sc = discretize_contin_bscore(ot, it, [1.0])
    assert list(sc.best_possible_bscore()) == [0.0, 0.0]
    assert sc.min_improv() == 0.0


def test_mismatched_table_sizes_are_rejected():
    ot, it = _tables([0.0, 1.0, 2.0])
    short = ITable([type_node.contin_type], ["x1"])
    short.push_back([0.0])
    with pytest.raises(ValueError, match="size do not match"):
        discretize_contin_bscore(ot, short, [1.0])
