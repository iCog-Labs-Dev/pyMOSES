import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    enum_t,
    get_builtin,
    get_contin,
    is_argument,
    is_boolean,
    is_builtin,
    is_contin,
    is_definite_object,
    negate_vertex,
    vertex,
    vertex_to_bool,
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


# ---------------------------------------------------------------------------
# S14 Task 1 additions: get_builtin, is_definite_object, is_boolean,
# vertex_to_bool -- needed by interpreter.py's mixed_interpreter and (for
# is_boolean/vertex_to_bool) table.py's table_fmt_vertex_to_str.
# ---------------------------------------------------------------------------


def test_get_builtin_returns_the_held_builtin():
    assert get_builtin(builtin.logical_true) is builtin.logical_true
    assert get_builtin(builtin.plus) is builtin.plus


def test_get_builtin_rejects_non_builtin_vertex():
    with pytest.raises(TypeError):
        get_builtin(2.5)
    with pytest.raises(TypeError):
        get_builtin(argument(1))
    with pytest.raises(TypeError):
        get_builtin("sad")


def test_is_definite_object_accepts_bare_str_but_not_enum_t():
    # A bare `str` in this port's vertex union stands for `definite_object`
    # (see vertex.py's module docstring); `enum_t` is a genuinely different
    # C++ variant alternative even though it subclasses `str` here.
    assert is_definite_object("some_object") is True
    assert is_definite_object(enum_t("some_enum")) is False


def test_is_definite_object_rejects_non_str_vertex():
    assert is_definite_object(builtin.logical_true) is False
    assert is_definite_object(argument(1)) is False
    assert is_definite_object(2.5) is False


def test_is_boolean_true_only_for_the_two_logical_constants():
    assert is_boolean(builtin.logical_true) is True
    assert is_boolean(builtin.logical_false) is True
    assert is_boolean(builtin.plus) is False
    assert is_boolean(argument(1)) is False
    # Critical-2-shaped trap: a contin_t whose value equals a logical
    # constant's ordinal (1.0 == builtin.logical_true) must NOT be treated
    # as boolean -- is_boolean goes through vertex_equal, not bare `==`.
    assert is_boolean(1.0) is False
    assert is_boolean(2.0) is False


def test_vertex_to_bool_matches_logical_true_and_false():
    assert vertex_to_bool(builtin.logical_true) is True
    assert vertex_to_bool(builtin.logical_false) is False


def test_vertex_to_bool_rejects_non_boolean_vertex():
    with pytest.raises(AssertionError):
        vertex_to_bool(builtin.plus)
    with pytest.raises(AssertionError):
        vertex_to_bool(argument(1))


def test_vertex_to_bool_rejects_colliding_contin_not_just_wrong_type():
    # The same Critical-2-shaped trap as test_is_boolean above, at the
    # sibling function: 1.0/2.0 must not be silently accepted as
    # logical_true/logical_false merely because their ordinals coincide.
    with pytest.raises(AssertionError):
        vertex_to_bool(1.0)
    with pytest.raises(AssertionError):
        vertex_to_bool(2.0)


def test_is_contin_expr_is_immune_to_the_intenum_value_collision():
    """`is_contin_expr` cannot be fooled by a contin whose value equals a
    builtin's ordinal.

    `contin_t` is `float` and `builtin` is an `IntEnum`, so in Python
    `8.0 == builtin.div` is `True` where C++'s variant alternatives never
    compare equal across types. Both prior Criticals in this project were
    that confusion, so the function's chain of bare `==` comparisons looks
    like a hazard.

    It is not, and this pins why: `is_contin(v)` is a genuine type check and
    it comes FIRST, so any contin value short-circuits before a comparison
    can misfire. `builtin.div`'s ordinal is 8, and `contin_t(8.0)` is the
    exact input that would expose an ordering mistake.

    Note the result is stable even under reordering here, because a contin
    value's correct answer is `True` either way -- so this test pins the
    outcome, not the mechanism. Two agents disagreed about whether this
    function was safe; measured, it is.
    """
    from pymetta.comboreduct.combo.common_def import contin_t
    from pymetta.comboreduct.combo.vertex import builtin, is_contin, is_contin_expr

    assert int(builtin.div) == 8
    colliding = contin_t(8.0)

    # The collision is real at the Python level ...
    assert colliding == builtin.div
    # ... but the type check fires first, so the answer is right for the
    # right reason: it is contin-typed, not "equal to div".
    assert is_contin(colliding) is True
    assert is_contin_expr(colliding) is True

    # A builtin that is NOT a contin operator must still be rejected, which
    # is what a value-only comparison would get wrong.
    assert is_contin(builtin.logical_and) is False
    assert is_contin_expr(builtin.logical_and) is False
    assert is_contin_expr(argument(1)) is False
