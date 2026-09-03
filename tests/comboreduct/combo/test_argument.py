import pytest

from pymetta.comboreduct.combo.argument import argument


def test_sign_encodes_negation():
    a = argument(3)
    assert not a.is_negated()
    assert a.abs_idx() == 3
    a.negate()
    assert a.idx == -3
    assert a.is_negated()
    assert a.abs_idx() == 3


def test_index_zero_is_rejected():
    with pytest.raises(AssertionError):
        argument(0)


def test_ordering_is_by_absolute_value():
    # C++: argument.h:61-64 uses opencog::absolute_value_order<int>
    # (opencog/util/numeric.h:56-60): `(a == -b) ? a < b : abs(a) < abs(b)`.
    # For same-magnitude pairs (a == -b, e.g. 1 and -1) the negative index
    # sorts first; across magnitudes it is a plain absolute-value sort. So
    # $1/!$1 (idx +-1) sort before $2/!$2 (idx +-2), and within each pair
    # the negated one (!$n, idx -n) comes first: -1, 1, -2, 2. A signed-int
    # comparison would instead give -2, -1, 1, 2 -- both orderings are
    # tested against directly below, not with an `or`.
    args = [argument(2), argument(-1), argument(1), argument(-2)]
    assert [a.idx for a in sorted(args)] == [-1, 1, -2, 2]
    assert [a.abs_idx() for a in sorted(args)] == [1, 1, 2, 2]


def test_ordering_matches_absolute_value_order_directly():
    # A second, independent check straight from absolute_value_order's own
    # operator(): (a == -b) ? a < b : abs(a) < abs(b), swept over a small
    # grid of nonzero indices.
    def cpp_absolute_value_order(a: int, b: int) -> bool:
        return a < b if a == -b else abs(a) < abs(b)

    values = [-3, -2, -1, 1, 2, 3]
    for a in values:
        for b in values:
            assert (argument(a) < argument(b)) == cpp_absolute_value_order(a, b), (a, b)


def test_abs_idx_from_zero():
    assert argument(1).abs_idx_from_zero() == 0
    assert argument(-1).abs_idx_from_zero() == 0
    assert argument(2).abs_idx_from_zero() == 1
    assert argument(-2).abs_idx_from_zero() == 1


def test_idx_from_zero_to_idx_is_the_inverse():
    for idx in (1, 2, 3, 5):
        assert argument.idx_from_zero_to_idx(argument(idx).abs_idx_from_zero()) == idx


def test_equality_ignores_ordering_sign_rule():
    # operator== (argument.h:66-68) is plain idx equality, unrelated to the
    # absolute-value ordering above: $1 != !$1.
    assert argument(1) == argument(1)
    assert argument(-1) == argument(-1)
    assert argument(1) != argument(-1)
    assert argument(1) != argument(2)


def test_hashable_and_consistent_with_equality():
    assert hash(argument(3)) == hash(argument(3))
    assert len({argument(1), argument(1), argument(-1)}) == 2


def test_total_ordering_derives_le_gt_ge():
    # functools.total_ordering (Python's analogue of C++'s
    # boost::less_than_comparable/equality_comparable mixins on argument,
    # argument.h:44-45) derives >, <=, >= from __lt__ and __eq__.
    assert argument(-1) <= argument(1)
    assert argument(1) <= argument(1)
    assert argument(1) > argument(-1)
    assert argument(2) >= argument(-2)


def test_repr_is_informative():
    assert repr(argument(-2)) == "argument(-2)"
