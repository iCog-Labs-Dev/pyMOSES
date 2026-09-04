import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    enum_t,
    get_contin,
    is_argument,
    is_builtin,
    is_contin,
    negate_vertex,
    vertex,
)


def test_builtin_member_count_matches_cpp_enum():
    # C++: vertex.h:69-105 -- null_vertex through builtin_count, 27 members.
    assert len(builtin) == 27


@pytest.mark.parametrize(
    ("member", "ordinal"),
    [
        (builtin.null_vertex, 0),
        (builtin.logical_true, 1),
        (builtin.logical_false, 2),
        (builtin.logical_and, 3),
        (builtin.logical_or, 4),
        (builtin.logical_not, 5),
        (builtin.plus, 6),
        (builtin.times, 7),
        (builtin.div, 8),
        (builtin.exp, 9),
        (builtin.log, 10),
        (builtin.sin, 11),
        (builtin.greater_than_zero, 12),
        (builtin.impulse, 13),
        (builtin.rand, 14),
        (builtin.list, 15),
        (builtin.car, 16),
        (builtin.cdr, 17),
        (builtin.cons, 18),
        (builtin.foldr, 19),
        (builtin.foldl, 20),
        (builtin.cond, 21),
        (builtin.equ, 22),
        (builtin.lambda_, 23),
        (builtin.apply, 24),
        (builtin.contin_if, 25),
        (builtin.builtin_count, 26),
    ],
)
def test_builtin_ordinals_match_cpp_enum(member, ordinal):
    # C++: vertex.h:69-105, transcribed verbatim (implicit C enum
    # auto-increment ordinals). `lambda_` is this port's name for C++'s
    # `lambda` -- a hard Python keyword; see vertex.py's DIVERGENCE note.
    assert member.value == ordinal


def test_builtin_is_int_enum_and_comparable_as_int():
    # Ordinals participate in comparisons elsewhere in classic MOSES (see
    # vertex.py's class docstring on lexicographic_subtree_order); IntEnum
    # is what makes `builtin.plus < builtin.times` etc. work directly.
    assert builtin.null_vertex < builtin.logical_and
    assert int(builtin.cond) == 21


def test_vertex_is_a_four_way_union():
    # Widened from three to four alternatives by Task 5 (comboreduct/table):
    # `str` stands in for `enum_t` (out of scope for Task 3, but needed by
    # table.py's type inference over CSV columns) -- see vertex.py's module
    # docstring.
    assert isinstance(builtin.plus, vertex)
    assert isinstance(argument(1), vertex)
    assert isinstance(3.14, vertex)
    assert isinstance("sad", vertex)
    assert not isinstance([], vertex)


def test_enum_t_is_a_str_subclass_but_stays_distinguishable():
    # Fix round 1 (task-5 review): enum_t must be its own type, not bare
    # str, so it can be told apart from definite_object/message (both
    # also represented as plain str in this port -- see vertex.py's
    # module docstring). isinstance/equality/hashing all fall through to
    # str's own.
    e = enum_t("sad")
    assert isinstance(e, str)
    assert isinstance(e, vertex)
    assert e == "sad"
    assert "sad" == e
    assert hash(e) == hash("sad")
    assert not isinstance("sad", enum_t)


def test_combo_tree_is_tree_subscripted_with_vertex():
    # C++: vertex.h:527 `typedef tree<vertex> combo_tree;` -- this port's
    # tree carries no runtime type parameter (see tree.__class_getitem__),
    # so `tree[vertex]` is accepted syntax that resolves to `tree` itself.
    from pymetta.util.tree import tree

    assert combo_tree is tree


def test_is_builtin_is_argument_is_contin_are_mutually_exclusive():
    b, a, c = builtin.plus, argument(1), 2.5
    assert is_builtin(b) and not is_argument(b) and not is_contin(b)
    assert is_argument(a) and not is_builtin(a) and not is_contin(a)
    assert is_contin(c) and not is_builtin(c) and not is_argument(c)


def test_negate_vertex_swaps_the_two_logical_constants():
    assert negate_vertex(builtin.logical_true) == builtin.logical_false
    assert negate_vertex(builtin.logical_false) == builtin.logical_true


def test_negate_vertex_rejects_non_constant_builtin():
    with pytest.raises(AssertionError):
        negate_vertex(builtin.plus)


def test_negate_vertex_rejects_non_builtin_vertex():
    with pytest.raises(TypeError):
        negate_vertex(argument(1))
    with pytest.raises(TypeError):
        negate_vertex(1.0)


def test_get_contin_returns_the_held_float():
    assert get_contin(2.5) == 2.5


def test_get_contin_rejects_non_contin_vertex():
    with pytest.raises(TypeError):
        get_contin(builtin.logical_true)
    with pytest.raises(TypeError):
        get_contin(argument(1))
    with pytest.raises(TypeError):
        get_contin("sad")
