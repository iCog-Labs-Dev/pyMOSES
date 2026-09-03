"""C++: moses/comboreduct/combo/vertex.h

`builtin` (vertex.h:69-105) and `vertex` (vertex.h:128-141), narrowed to the
subset this port carries: `builtin | argument | contin_t | str`. The C++
`vertex` is a 14-way `boost::variant`; the other nine alternatives --
wild_card, action, builtin_action, perception, definite_object,
indefinite_object, message, procedure_call, action_symbol, ann_type -- are
out of scope for this task (see task-3-brief.md) and are not constructed
anywhere in this port.

Task 5 (comboreduct/table) widens this beyond Task 3's original
`builtin | argument | contin_t`: classic's type inference over CSV columns
(table_io.cc's `token_to_vertex`) produces `enum_t` values for both
genuinely enum-typed columns and, via what table_io.cc's own comment calls
"an ugly hack", for columns that fail every other type guess (dates,
free text -- see `id::ill_formed_type` in table_io.cc:298-301). `enum_t`
itself (moses/comboreduct/combo/enum_type.h:39) is, by its own header
comment, "essentially a string... coded as a different type... because it
semantically denotes something else" -- an interned wrapper around
`std::string` whose `operator==`/`operator<` (enum_type.h:75-82) reduce to
string identity/lexicographic order. Since this port has no use for the
interning (a performance optimisation for C++'s `boost::get<T>` dispatch,
irrelevant in Python) or for `enum_t` as a distinct wrapper type, it is
narrowed to plain `str` here -- disclosed scope decision, see
task-5-report.md.
"""

from enum import IntEnum

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.util.tree import tree


class builtin(IntEnum):
    """C++: moses/comboreduct/combo/vertex.h:69 builtin

    Transcribed verbatim from the C `enum` at vertex.h:69-105: same member
    names, same implicit ordinals (C enum auto-increment, `null_vertex = 0`
    first). Do not reorder or omit members even where unimplemented in this
    port -- ordinals participate directly in comparisons elsewhere (the
    C++ comment at vertex.h:74-78 notes the boolean constants are ordered
    first specifically so lexicographic_subtree_order runs a bit faster).

    `id::builtin_count` (vertex.h:105) is not itself printed or parsed by
    ostream_builtin/builtin_str_to_vertex (iostream_combo.py) -- it exists
    only so C++ code can say "number of builtin members" without a
    separate constant; ported for ordinal fidelity, not for any printable
    meaning of its own.
    """

    null_vertex = 0

    # simple boolean functions -- constants come first, see class docstring
    logical_true = 1
    logical_false = 2

    logical_and = 3
    logical_or = 4
    logical_not = 5

    # contin functions (take contin as arg, or return contin)
    plus = 6
    times = 7
    div = 8
    exp = 9
    log = 10  # if ABS_LOG is enabled then log(x) := log(abs(x)) (C++ only)
    sin = 11
    greater_than_zero = 12
    impulse = 13  # 1.0 if true else 0.0
    rand = 14  # random contin_t in [0,1)

    list = 15  # List constructor
    car = 16
    cdr = 17
    cons = 18  # Functions on lists
    foldr = 19
    foldl = 20
    # Generic functions (currently take enum as arg or return enum)
    cond = 21
    equ = 22

    # DIVERGENCE(C++ moses/comboreduct/combo/vertex.h:98): `lambda` is a
    # hard Python keyword -- it starts a lambda expression grammatically,
    # so `lambda = 23` is a SyntaxError and cannot appear as a class-body
    # assignment target the way every other member here does. Renamed
    # `lambda_`, the standard PEP 8 convention for exactly this collision
    # (compare `class_`, `type_` elsewhere in the Python ecosystem). Only
    # the Python identifier is affected: the combo surface syntax this
    # builtin maps to/from ("->" and "lambda", both plain string literals,
    # see iostream_combo.py) is unchanged.
    lambda_ = 23
    apply = 24

    # Should be obsoleted by cond, at some point (C++ comment, still true).
    contin_if = 25

    builtin_count = 26  # to get the number of builtin


# C++: moses/comboreduct/combo/vertex.h:128 vertex
#
# `boost::variant<contin_t, enum_t, builtin, wild_card, argument, action,
# builtin_action, perception, definite_object, indefinite_object, message,
# procedure_call, action_symbol, ann_type>`, narrowed to this port's four
# live alternatives (see module docstring, `str` standing in for `enum_t`).
# `X | Y` (PEP 604) builds a runtime `types.UnionType`, so
# `isinstance(v, vertex)` works exactly like a manual
# `isinstance(v, (builtin, argument, contin_t, str))` would.
vertex = builtin | argument | contin_t | str

# C++: moses/comboreduct/combo/vertex.h:527 combo_tree
#
# `typedef tree<vertex> combo_tree;` -- see tree.__class_getitem__
# (pymetta/util/tree.py) for why `tree[vertex]` is accepted syntax despite
# this port's tree carrying no runtime type parameter.
combo_tree = tree[vertex]


def is_builtin(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:98 is_builtin"""
    return isinstance(v, builtin)


def is_argument(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:135 is_argument

    C++'s own doc comment (vertex.h:653-659) warns this does not take into
    account the argument's declared type (boolean vs. contin); any caller
    that blindly trusts this risks accepting an argument of the wrong
    type. Ported as-is -- that caveat is inherited, not something this
    wrapper corrects.
    """
    return isinstance(v, argument)


def is_contin(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:118 is_contin"""
    return isinstance(v, contin_t)


def get_contin(v: vertex) -> contin_t:
    """C++: moses/comboreduct/combo/vertex.cc:122 get_contin

    `boost::get<contin_t>(v)` -- throws `boost::bad_get` (ported as
    `TypeError`) if `v` does not hold a `contin_t`. Added for Task 5's
    `Table.compressed()` (table.cc:501), which uses this exact function
    to pull a row's weight out of its weighting-feature column.
    """
    if not is_contin(v):
        raise TypeError(f"vertex {v!r} does not hold a contin_t")
    return v


def is_contin_expr(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:764 is_contin_expr

    Added for task 4 (`interpreter_visitor`'s `mixed` computation at
    table.h:306-317, which calls this on every vertex of the tree to decide
    whether the boolean or the mixed interpreter is needed) -- not in
    task-3-brief.md's file list, but a direct, narrow port of a template
    function vertex.h itself declares, scoped the same way every other
    helper in this module is scoped (narrowed to `builtin | argument |
    contin_t`; `is_contin(v)` alone covers the `contin_t` alternative,
    `argument` never matches any branch below, so only `builtin` members
    are actually distinguished here).

    Returns true if `v` is itself contin-typed, or is a builtin operator
    that takes/returns contin (`div`, `exp`, `impulse`, `log`, `plus`,
    `rand`, `sin`, `times`). Deliberately excludes `greater_than_zero`
    (a predicate, not a contin expression) -- table.h's mixed check tests
    that separately, right after this one.
    """
    return (
        is_contin(v)
        or v == builtin.div
        or v == builtin.exp
        or v == builtin.impulse
        or v == builtin.log
        or v == builtin.plus
        or v == builtin.rand
        or v == builtin.sin
        or v == builtin.times
    )


def negate_vertex(v: vertex) -> vertex:
    """C++: moses/comboreduct/combo/vertex.cc:236 negate_vertex

    Returns logical_false for logical_true and vice versa -- nothing else.
    This is *not* a general vertex negation (that is what argument.negate()
    is for, mutating the argument itself, not a vertex holding one): it
    mirrors get_builtin(v) (vertex.cc:100-104, `boost::get<builtin>(v)`,
    which throws boost::bad_get if v does not hold a builtin at all)
    followed by negate_builtin's own OC_ASSERT (vertex.cc:241-251) that the
    builtin it got really is id::logical_true or id::logical_false.
    """
    if not is_builtin(v):
        raise TypeError(f"vertex {v!r} does not hold a builtin")
    if v == builtin.logical_true:
        return builtin.logical_false
    if v == builtin.logical_false:
        return builtin.logical_true
    raise AssertionError(
        f'builtin "{v.name}" should be id::logical_true or id::logical_false'
    )
