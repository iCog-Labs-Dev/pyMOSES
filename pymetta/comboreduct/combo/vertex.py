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
string identity/lexicographic order. This port has no use for the
interning (a performance optimisation for C++'s `boost::get<T>` dispatch,
irrelevant in Python), so `enum_t` here (below) is a bare `str` subclass
carrying no extra state -- but it IS kept as its own class, not narrowed
to plain `str` outright: `str` in the `vertex` union also stands for
`definite_object` (a genuine, distinct `std::string`-typed alternative in
C++'s variant -- table_io.cc:299-300's `definite_object_type` case) and,
if this port ever grows one, `message`. Collapsing `enum_t` into bare
`str` would make those indistinguishable from an enum value by
`isinstance` alone, which Task 6's scoring code needs to do (see
task-5-report.md's fix-round-1 section: this was flagged as a live
conflation, not a hypothetical one, before Task 6 could build on it).
`isinstance(v, str)` still recognises `enum_t` values (it subclasses
`str`), so the `vertex` union itself does not need to change.
"""

from enum import IntEnum

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.util.tree import tree, tree_node


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


class enum_t(str):
    """C++: moses/comboreduct/combo/enum_type.h:39 enum_t

    A bare `str` subclass -- see the module docstring for why this is a
    distinct class rather than plain `str`. `isinstance`/`==`/`<`/hashing
    all fall straight through to `str`'s own (enum_type.h:75-82's
    `operator==`/`operator<` reduce to exactly that: id/content identity
    and lexicographic order, which is what `str` already gives for free),
    so no members are added.
    """


# C++: moses/comboreduct/combo/vertex.h:128 vertex
#
# `boost::variant<contin_t, enum_t, builtin, wild_card, argument, action,
# builtin_action, perception, definite_object, indefinite_object, message,
# procedure_call, action_symbol, ann_type>`, narrowed to this port's four
# live alternatives (see module docstring). `str` covers both `enum_t`
# (via subclassing -- `isinstance(enum_t(...), vertex)` is True through
# the `str` check) and `definite_object`/`message` (genuine `str`s, not
# `enum_t`). `X | Y` (PEP 604) builds a runtime `types.UnionType`, so
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


def get_builtin(v: vertex) -> builtin:
    """C++: moses/comboreduct/combo/vertex.cc:103 get_builtin

    `boost::get<builtin>(v)` -- throws `boost::bad_get` (ported as
    `TypeError`, matching `get_contin`'s own convention) if `v` does not
    hold a `builtin`. Added for S14 Task 1's `mixed_interpreter` (interpreter.py),
    whose `boolean_eval` override is `get_builtin(mixed_eval(it))`
    (interpreter.cc:258) -- unlike `negate_vertex` (this module), which
    already folds "extract the builtin, then act on it" into one step,
    `get_builtin` on its own is genuinely needed here because the caller
    wants the extracted `builtin` itself, not a derived value.
    """
    if not is_builtin(v):
        raise TypeError(f"vertex {v!r} does not hold a builtin")
    return v


def is_definite_object(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:195 is_definite_object

    `boost::get<definite_object>(&v)` -- in C++, `definite_object` and
    `enum_t` are genuinely different `boost::variant` alternatives, so
    this can never be `True` for an `enum_t`. This port's narrowed vertex
    union collapses both onto plain `str` (`enum_t` a subclass of it --
    see this module's own docstring), so the `enum_t` exclusion has to be
    explicit here: `isinstance(v, str) and not isinstance(v, enum_t)`.
    Added for S14 Task 1's `mixed_interpreter::mixed_eval`
    (interpreter.cc:375), whose "string constant" fallback branch
    (`is_definite_object(v)`) returns the definite_object vertex
    unevaluated.
    """
    return isinstance(v, str) and not isinstance(v, enum_t)


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


def swap_and_or(v: vertex) -> vertex:
    """C++: moses/comboreduct/combo/vertex.cc:257 swap_and_or

    Returns id::logical_or for id::logical_and and vice versa. Classic's
    `OC_ASSERT(v == id::logical_or || v == id::logical_and)` is active in
    release builds (it is an OC_ASSERT, not a bare `assert`), so it is
    raised explicitly here rather than written as a Python `assert` that
    `python -O` would strip -- the same reasoning `argument.__init__`'s own
    docstring gives for its idx-zero check.
    """
    if vertex_equal(v, builtin.logical_and):
        return builtin.logical_or
    if vertex_equal(v, builtin.logical_or):
        return builtin.logical_and
    raise AssertionError("v == id::logical_or || v == id::logical_and")


def is_enum_type(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:87 is_enum_type

    `boost::get<enum_t>(&v)` -- specifically the `enum_t` alternative, not
    any `str`. Added for Task 1 (interpreter/eval.py's `eval_throws_vertex`,
    eval.cc:272, mirrors "enums are constants too"). Deliberately narrower
    than `isinstance(v, str)`: a bare, non-`enum_t` `str` in this port's
    vertex union also stands for `definite_object`/`message` (see this
    module's docstring), which are genuinely different `boost::variant`
    alternatives in C++ and do not satisfy `is_enum_type` there either.
    `enum_t` subclassing `str` (rather than being a disjoint type) is
    exactly why this needs its own function instead of reusing `is_contin`
    et al.'s plain-`isinstance` shape unguarded.
    """
    return isinstance(v, enum_t)


def bool_to_vertex(b: bool) -> vertex:
    """C++: moses/comboreduct/combo/vertex.h:698 bool_to_vertex

    Inlined in C++ as `bool_to_builtin(b)` (vertex.h:694-696:
    `b ? id::logical_true : id::logical_false`); ported as one function
    since this port has no separate `bool_to_builtin` call site to
    preserve. Added for Task 1 (interpreter/eval.py's `eval_throws_vertex`,
    eval.cc:169, the `greater_than_zero` case).
    """
    return builtin.logical_true if b else builtin.logical_false


def builtin_to_bool(b: builtin) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:708 builtin_to_bool

    The inverse of `bool_to_vertex`/`bool_to_builtin`. `b` must already be
    KNOWN to hold a `builtin` (the C++ parameter type is `const builtin&`,
    not `vertex` -- no discrimination to do), so a plain `==` between two
    `builtin` `IntEnum` members is safe here (CLAUDE.md's "type before
    value" trap is about comparing across DIFFERENT vertex alternatives,
    e.g. `contin_t == builtin`, which cannot arise once `b`'s static type
    is already `builtin`).

    FIX WAVE convention (argument.py's `__init__`, vertex.h:710's
    `OC_ASSERT`): a bare Python `assert` is stripped under `python -O`,
    so the guard is a `raise AssertionError` instead, to survive
    `-O` the same way `OC_ASSERT` survives `NDEBUG`. Added for Task 5
    (`comboreduct/table/table.py`'s `complete_truth_table`, table.h:1436),
    which needs this to translate `boolean_interpreter`'s `builtin`
    result back into a plain `bool` for the truth table it accumulates.
    """
    if b != builtin.logical_true and b != builtin.logical_false:
        raise AssertionError("builtin should be 'id::logical_true' or 'id::logical_false'.")
    return b == builtin.logical_true


def is_boolean(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:732 is_boolean

    `template<typename T> bool is_boolean(const T& v) { return (v ==
    id::logical_true || v == id::logical_false); }` -- a bare `==` in
    C++, safe there because `boost::variant`/a concrete `builtin` can
    never spuriously compare equal to a different-typed value the way
    `contin_t`/`IntEnum` can here (CLAUDE.md's "type before value" trap).
    Ported via `vertex_equal` rather than bare `==` for exactly that
    reason -- added for S14 Task 1's `table_fmt_vertex_to_str`
    (table.py, table.cc:63/table.cc:72), which needs to tell a genuine
    boolean output column apart from a contin one that merely happens to
    hold an integral value.
    """
    return vertex_equal(v, builtin.logical_true) or vertex_equal(v, builtin.logical_false)


def vertex_to_bool(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:701 vertex_to_bool

    The `vertex`-typed sibling of `builtin_to_bool` above: same OC_ASSERT
    shape (`v == id::logical_true || v == id::logical_false`), but `v` is
    NOT already known to hold a `builtin` here -- it is a full `vertex`,
    so the comparison goes through `vertex_equal` (type-safe) rather than
    bare `==`, unlike `builtin_to_bool`'s own justified use of a direct
    `builtin`-to-`builtin` comparison. Added for S14 Task 1's
    `table_fmt_vertex_to_str` (table.py, table.cc:63/:72), used there
    exactly as C++ uses it: `ss << vertex_to_bool(v)` prints a genuine
    boolean output column as "0"/"1", the "Boolean values are output in
    binary form" convention `ostreamTable`'s own doc comment names
    (table_io.h:192-193).
    """
    if not is_boolean(v):
        raise AssertionError("vertex should be 'id::logical_true' or 'id::logical_false'.")
    return vertex_equal(v, builtin.logical_true)


def vertex_equal(a: vertex, b: vertex) -> bool:
    """No direct C++ symbol: `vertex` (vertex.h:128) is a `boost::variant`,
    and `boost::variant<...>::operator==` is compiler-generated -- it
    compares which alternative is *held* (`which()`) before ever comparing
    a value, so it is unconditionally `false` whenever `a` and `b` hold
    different alternatives. This reproduces exactly that discrimination
    for this port's four surviving alternatives.

    This closes the trap the task brief names directly: `contin_t` is a
    bare `float` and `builtin` an `IntEnum`, so bare `a == b` between one
    of each can be `True` in Python (`1.0 == builtin.logical_true`) purely
    because their ordinals coincide -- something `boost::variant` can
    never do. `argument.__eq__` (argument.py) and `str.__eq__` already
    return `NotImplemented`/`False` for a mismatched type on their own, so
    only the `builtin`/`contin_t` pair is actually at risk here -- but all
    four alternatives are guarded uniformly so a future reader does not
    have to rediscover which pair was the dangerous one. Use this (not
    bare `==`) anywhere a vertex of *unproven* type is compared against a
    specific value, e.g. a sub-evaluation result compared against
    `builtin.logical_true` -- see interpreter.py's own module docstring
    and eval.py's call sites for the general rule.
    """
    if isinstance(a, builtin) != isinstance(b, builtin):
        return False
    if isinstance(a, contin_t) != isinstance(b, contin_t):
        return False
    if isinstance(a, argument) != isinstance(b, argument):
        return False
    if isinstance(a, str) != isinstance(b, str):
        return False
    return a == b


# ---------------------------------------------------------------------------
# lexicographic_subtree_order -- promoted here (not private to any one
# reduct file) because THREE separate call sites across S7/S8 instantiate
# it as `lexicographic_subtree_order<vertex>` against this exact vertex
# type, and its actual output order is load-bearing, not incidental:
#
#   * combo/assumption.cc:36 (`insert_assumption`'s `Comp`)
#   * reduct/general_rules.cc:106 (`reorder_commutative`'s
#     `tr.sort_on_subtrees(..., lexicographic_subtree_order<vertex>())`)
#     -- this one SORTS THE PRINTED CHILDREN of every commutative node in
#     the final reduced tree, so its output feeds Task 6's byte-identical
#     oracle directly.
#   * reduct/logical_rules.h:185 (`reduce_to_enf`'s
#     `std::set<sib_it, lexicographic_subtree_order<value_type>>`)
#
# RETRACTION: an earlier version of this comparator (when it lived only in
# assumption.py) claimed the exact order was "not load-bearing... can be
# changed if it's important" -- quoting C++'s own comment on the class,
# but not checking whether OTHER callers depended on the actual order.
# They do (`reorder_commutative` above). That claim is wrong and is
# retracted here, not repeated.
#
# What IS still true, and is the actual scope limit of this port: only
# RELATIVE order among this port's four vertex alternatives is
# reproduced, matching their real position in the 14-way `boost::variant`
# (vertex.h:128: `contin_t, enum_t, builtin, wild_card, argument, ...`),
# not the literal `boost::variant::which()` integer. For the domain S7's
# byte-identical oracle actually exercises -- boolean combo trees, whose
# vertices are only ever `builtin` and `argument` (no `contin_t`/`enum_t`
# literal ever appears as a clause in the boolean corpus) -- `builtin`
# ranking below `argument` is the one relative-order fact that matters,
# and it is verified correct against the real variant declaration order.
# The `contin_t`/`str` ranks relative to `builtin`/`argument` are ported
# the same way but have no boolean-corpus call site to verify against;
# revisit if/when S14 (contin/mixed reduction, out of scope per CLAUDE.md)
# needs this comparator over a genuinely mixed vertex set.
# ---------------------------------------------------------------------------


def vertex_rank(v: vertex) -> int:
    """No direct C++ symbol: reproduces the RELATIVE order of this port's
    four vertex alternatives as they appear in the real 14-way
    `boost::variant` (vertex.h:128) -- `contin_t`=0, `enum_t`/`str`=1,
    `builtin`=2, `argument`=3 (the intervening `wild_card`=3 in the real
    variant is skipped, so `argument`'s rank here is 3, not 4; absolute
    values are never compared or serialised, only relative order). See
    this section's own module-level note for what is and is not verified
    about this ordering.
    """
    if isinstance(v, contin_t):
        return 0
    if isinstance(v, str):
        return 1
    if isinstance(v, builtin):
        return 2
    if isinstance(v, argument):
        return 3
    raise TypeError(f"vertex {v!r} is not one of this port's four alternatives")


def vertex_less(v1: vertex, v2: vertex) -> bool:
    """No direct C++ symbol: a total order over single vertices standing
    in for `boost::variant`'s own default `operator<` (rank/`which()`
    first, then each alternative's own `operator<` for a same-rank pair)
    -- `argument.__lt__` (`absolute_value_order`, argument.py) for two
    arguments, plain `<` for two `contin_t`/`str`/`builtin` values (the
    last by ordinal, `builtin` being an `IntEnum`). Safe by construction:
    `vertex_rank` already confirmed both operands share the same Python
    type before either `<` here ever runs, so this never risks the
    `contin_t`/`builtin` cross-type trap `vertex_equal` guards against.
    """
    r1, r2 = vertex_rank(v1), vertex_rank(v2)
    if r1 != r2:
        return r1 < r2
    return v1 < v2


def lexicographic_subtree_cmp(n1: tree_node, n2: tree_node) -> int:
    """C++: opencog/util/tree.h:3277 lexicographic_subtree_order::cmp

    Recursive pre-order comparison: `1` if `n1`'s subtree sorts first,
    `-1` if `n2`'s does, `0` if they are structurally equal. A node with
    fewer children than its counterpart (once their own vertices compare
    equal) sorts first -- ported directly from the C++ recursion using
    this port's `first_child`/`next_sibling` node links in place of
    `sibling_iterator`.
    """
    if vertex_less(n1.data, n2.data):
        return 1
    if vertex_less(n2.data, n1.data):
        return -1
    sib1, sib2 = n1.first_child, n2.first_child
    while True:
        if sib1 is None:
            return 0 if sib2 is None else 1
        if sib2 is None:
            return -1
        res = lexicographic_subtree_cmp(sib1, sib2)
        if res != 0:
            return res
        sib1, sib2 = sib1.next_sibling, sib2.next_sibling


def lexicographic_subtree_less(n1: tree_node, n2: tree_node) -> bool:
    """C++: opencog/util/tree.h:3273 lexicographic_subtree_order::operator()"""
    return lexicographic_subtree_cmp(n1, n2) > 0


# ---------------------------------------------------------------------------
# equal_subtree_vertex_safe -- promoted here (from combo/assumption.py, where
# it started life as a private `_equal_subtree_vertex_safe`) because a SECOND
# call site now needs it: reduct/meta_rules.py's `iterative`/`assum_iterative`
# fixpoint checks (Task 2, S7). Both are the same "no direct C++ symbol: a
# type-safe replacement for tree.equal_subtree" need assumption.py's own
# docstring already explained -- `tree.equal_subtree` (tree.py:493) compares
# node data with a bare `a.data != b.data`, which is exactly the
# `contin_t`/`IntEnum` collision the module docstring's "type before value"
# constraint warns about (verified live: `equal_subtree(combo_tree(1.0).
# begin(), combo_tree(builtin.logical_true).begin())` returns `True`).
# `tree.equal_subtree` is shared, foundational S0-S6 code; fixing it globally
# is out of scope for either task that needs this, so both close the hole at
# their own call sites via this one shared helper instead of two copies.
# ---------------------------------------------------------------------------


def equal_subtree_vertex_safe(a: tree_node, b: tree_node) -> bool:
    """No direct C++ symbol: see the section note above.

    CORRECTED (Task 2, S7) from an earlier claim in this function's
    previous home (assumption.py) that it shares "the same
    recursive-descent shape as tree.equal_subtree" -- it does not. This is
    a SIMPLER pure children-recursion walk (`a.first_child`/
    `b.first_child`, stepping `.next_sibling` on each side in lockstep),
    never touching `a`'s or `b`'s own `next_sibling`/`parent` at the top
    level, unlike `tree.equal_subtree`'s pre-order-stepping-with-boundary
    algorithm (which computes `_skip_children_successor(a)` up front
    specifically because ITS pre-order stepping would otherwise walk past
    `a`'s own subtree into `a`'s real siblings). Because this walk only
    ever DESCENDS into children, it is already, unconditionally, bounded
    to exactly `a`'s (respectively `b`'s) own subtree, regardless of
    what real siblings `a`/`b` might have in an enclosing tree -- no
    boundary computation needed, and (a provable improvement, not merely
    a difference) it does not share `tree.equal_subtree`'s own documented
    "accepted, inherited imprecision" for shape-mismatched subtrees whose
    `b` side has real trailing content (this function's `while` loop
    correctly returns `False` the moment one side runs out of children
    before the other does, via the final `ca is None and cb is None`).

    That inherited imprecision is a REAL C++ behaviour (tree.h's own
    `equal()`/`equal_subtree()`), not a Python artifact, so reusing it
    verbatim would matter for faithfulness IF this function's own callers
    could ever hit it -- they cannot: every current call site
    (assumption.py's per-element standalone copies; meta_rules.py's
    `tmp.begin()`, always the root of a freshly-copied STANDALONE
    combo_tree) passes a `b` with no real siblings of its own to read
    past in the first place, so the two algorithms are provably
    equivalent for every case this function is ever actually asked to
    decide. See task-2-report.md for the argument in full.
    """
    if a is b:
        return True
    if not vertex_equal(a.data, b.data):
        return False
    ca, cb = a.first_child, b.first_child
    while ca is not None and cb is not None:
        if not equal_subtree_vertex_safe(ca, cb):
            return False
        ca, cb = ca.next_sibling, cb.next_sibling
    return ca is None and cb is None
