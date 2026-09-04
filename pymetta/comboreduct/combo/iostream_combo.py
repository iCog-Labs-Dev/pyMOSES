"""C++: moses/comboreduct/combo/iostream_combo.{h,cc}

Combo tree printing and parsing, scoped to the default `output_format::combo`
format only (iostream_combo.h:42-48's `python`/`python3`/`scheme` formats are
out of scope for this task) and to the subset of combo-tree GRAMMAR this
port's parser recognises -- builtin | argument | contin_t token forms only
(see vertex.py). Stale as of Task 5, corrected here: `vertex.py`'s `vertex`
alias itself is no longer this narrow (Task 5 widened it to include `str`/
`enum_t`, for table columns -- see its module docstring), but nothing in
THIS module's grammar ever produces or expects an enum_t/message/ann_type
token form regardless -- combo-tree syntax and table-column syntax are
different grammars, and only the latter was widened.

Parsing does not go through an intermediate `tree<std::string>` the way
classic does (`stream_to_combo_tree`, iostream_combo.h:219, first parses into
a generic string tree via opencog::tree's own `operator>>`, then converts
node-by-node with `str_to_vertex`). That two-stage design exists in C++ so
`tree<std::string>`'s parser can be reused for other node payloads; ported
directly here as a single recursive-descent pass, but the grammar it
implements -- what counts as a token, when a trailing `(...)` opens a
subtree, and the empty/whitespace-only-parens special case -- is taken
directly from CogUtil's tree.cc (see _parse_expr below), not reinvented.

Also scoped: a `combo_tree` is always single-rooted (see pymetta/util/
tree.py's module docstring), so str_to_combo_tree parses exactly one
top-level expression and rejects trailing content, unlike classic's
strtree_to_combo_tree (iostream_combo.h:203-216) which folds a forest of
sibling trees under one implicit root.
"""

import re

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.comboreduct.combo.vertex import builtin, combo_tree

# ---------------------------------------------------------------------------
# vertex <-> token string tables (combo format only)
# ---------------------------------------------------------------------------

# C++: moses/comboreduct/combo/iostream_combo.cc:36 builtin_str_to_vertex
#
# Restricted to the branches of builtin_str_to_vertex (iostream_combo.cc:
# 36-124) that assign a genuine `id::builtin` (vertex.py's `builtin` enum):
# the C++ function is actually shared across every "id::"-style enum in the
# full vertex variant (ann_type, wild_card::asterisk, and a long run of
# id::action members like sequential_and/action_not/...), none of which
# this port's vertex carries (see vertex.py's module docstring) -- those
# branches are simply absent below.
#
# "equ" is deliberately absent, matching a real asymmetry in the C++: it is
# printed by ostream_builtin (iostream_combo.cc:321-322) but has no string
# form recognised by builtin_str_to_vertex at all, so classic MOSES itself
# cannot parse "equ" back from combo text either.
_STR_TO_BUILTIN: dict[str, builtin] = {
    "and": builtin.logical_and,
    "logical_and": builtin.logical_and,
    "or": builtin.logical_or,
    "logical_or": builtin.logical_or,
    "not": builtin.logical_not,
    "logical_not": builtin.logical_not,
    "true": builtin.logical_true,
    "logical_true": builtin.logical_true,
    "false": builtin.logical_false,
    "logical_false": builtin.logical_false,
    "cond": builtin.cond,
    "contin_if": builtin.contin_if,
    "contin_boolean_if": builtin.contin_if,
    "+": builtin.plus,
    "plus": builtin.plus,
    "*": builtin.times,
    "times": builtin.times,
    "/": builtin.div,
    "div": builtin.div,
    "log": builtin.log,
    "exp": builtin.exp,
    "sin": builtin.sin,
    "0<": builtin.greater_than_zero,
    "impulse": builtin.impulse,
    "rand": builtin.rand,
    "list": builtin.list,
    "car": builtin.car,
    "cdr": builtin.cdr,
    "cons": builtin.cons,
    "foldr": builtin.foldr,
    "foldl": builtin.foldl,
    "->": builtin.lambda_,
    "lambda": builtin.lambda_,
    "apply": builtin.apply,
    "null_vertex": builtin.null_vertex,
}

# C++: moses/comboreduct/combo/iostream_combo.cc:266 ostream_builtin
#
# The `case output_format::combo:` branch only (iostream_combo.cc:269-325).
# Unlike _STR_TO_BUILTIN above, "equ" *is* here (see that table's docstring
# for the asymmetry) -- this is the reverse map, keyed by builtin member.
_BUILTIN_TO_STR: dict[builtin, str] = {
    builtin.null_vertex: "null_vertex",
    builtin.logical_and: "and",
    builtin.logical_or: "or",
    builtin.logical_not: "not",
    builtin.logical_true: "true",
    builtin.logical_false: "false",
    builtin.plus: "+",
    builtin.times: "*",
    builtin.div: "/",
    builtin.log: "log",
    builtin.exp: "exp",
    builtin.sin: "sin",
    builtin.greater_than_zero: "0<",
    builtin.impulse: "impulse",
    builtin.rand: "rand",
    builtin.list: "list",
    builtin.car: "car",
    builtin.cdr: "cdr",
    builtin.cons: "cons",
    builtin.foldr: "foldr",
    builtin.foldl: "foldl",
    builtin.lambda_: "->",
    builtin.apply: "apply",
    builtin.contin_if: "contin_if",
    builtin.cond: "cond",
    builtin.equ: "equ",
}


def _builtin_str_to_vertex(token: str) -> builtin | None:
    """C++: moses/comboreduct/combo/iostream_combo.cc:36 builtin_str_to_vertex

    Returns None (C++: returns false, leaving `v` untouched) rather than
    raising, so callers can fall through to the next vertex kind -- exactly
    how builtin_str_to_vertex's boolean return is used at its own call site
    (str_to_vertex, iostream_combo.h:136).
    """
    return _STR_TO_BUILTIN.get(token)


def _argument_str_to_vertex(token: str) -> argument | None:
    """C++: moses/comboreduct/combo/iostream_combo.cc:155 argument_str_to_vertex

    Arguments come in two surface forms: $n and !$n, n an integer.
    boost::lexical_cast<arity_t> failing (not a clean integer) falls
    through to None, matching the C++ catch-and-return-false; but arg == 0
    *after* a successful parse is OC_ASSERT's territory in C++, not a
    fallthrough -- ported as a raised AssertionError, not a None return,
    for the same reason (see argument.__init__, which would raise the
    identical AssertionError one call further down if this let 0 through).
    """
    if token.startswith("!$"):
        digits = token[2:]
        sign = -1
    elif token.startswith("$"):
        digits = token[1:]
        sign = 1
    else:
        return None
    try:
        arg = int(digits)
    except ValueError:
        return None
    assert arg != 0, "arg value should be different than zero."
    return argument(sign * arg)


def _contin_str_to_vertex(token: str) -> contin_t | None:
    """C++: moses/comboreduct/combo/iostream_combo.cc:185 contin_str_to_vertex

    A contin is any naked number, with or without a decimal point.
    """
    try:
        return float(token)
    except ValueError:
        return None


def _token_to_vertex(token: str):
    """C++: moses/comboreduct/combo/iostream_combo.h:131 str_to_vertex

    Tries builtin, then argument, then contin, in that order -- the order
    is significant in C++ (e.g. a token starting with '$' must fail
    argument_str_to_vertex, via a non-numeric suffix, before ann_str_to_vertex
    or the enum/message/definite_object fallbacks even get a look; here,
    failing all three of the alternatives this port's vertex actually has
    is a hard parse error, not a silent fallback to a fourth vertex kind
    this port does not carry).
    """
    v = _builtin_str_to_vertex(token)
    if v is not None:
        return v
    v = _argument_str_to_vertex(token)
    if v is not None:
        return v
    v = _contin_str_to_vertex(token)
    if v is not None:
        return v
    raise ValueError(
        f"cannot parse combo token {token!r} into a vertex -- combo-tree "
        "grammar in this port recognises builtin | argument | contin_t "
        "token forms only (see vertex.py's module docstring); enum/"
        "message/ann_type/definite_object are not part of that grammar here"
    )


def _vertex_to_str(v) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:428 ostream_vertex

    The combo-format cases only, in the same argument/builtin/contin order
    ostream_vertex itself checks them in (iostream_combo.cc:433-464, ending
    in the try/get<contin_t> fallback at 463-467) -- ported as a `match`
    (mirroring boost::apply_visitor, per task-3-brief.md) rather than a
    chain of isinstance checks.
    """
    match v:
        case argument():
            return _argument_to_str(v)
        case builtin():
            return _builtin_to_str(v)
        case float():
            return _contin_to_str(v)
        case _:
            raise TypeError(f"don't know how to print vertex {v!r} (combo format)")


def _argument_to_str(a: argument) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:622 ostream_abbreviate_literal

    ostream_argument's `case output_format::combo:` (iostream_combo.cc:
    386-403) branches on ABBREVIATE_NEGATIVE_LITERAL, #defined at
    iostream_combo.cc:33 with no corresponding #undef anywhere in the file
    -- so classic MOSES always takes the ostream_abbreviate_literal path
    ("!$n") for combo-format output, never the "not($n)" branch guarded by
    #else (iostream_combo.cc:389-402); only the abbreviated path is ported.
    The `labels` parameter (empty by default, substituting a variable name
    for the bare index) is not in this task's scope -- see the module
    docstring.
    """
    sign = "!" if a.is_negated() else ""
    return f"{sign}${a.abs_idx()}"


def _builtin_to_str(b: builtin) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:266 ostream_builtin

    The `case output_format::combo:` branch (iostream_combo.cc:269-325)
    only, including its `default:` fallback (iostream_combo.cc:323-324,
    "Builtin %u unknown") for the one enumerator with no case of its own,
    id::builtin_count.
    """
    return _BUILTIN_TO_STR.get(b, f"Builtin {int(b)} unknown")


def _contin_to_str(c: contin_t) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:428 ostream_vertex

    The fallback at the end of ostream_vertex, `out << get<contin_t>(v)`
    (iostream_combo.cc:463-464) -- reached once none of the earlier
    `get<T>(&v)` checks match, which for this port's narrowed vertex means
    "not an argument and not a builtin." Prints a bare double
    through the default C++ stream formatting: `defaultfloat`, precision 6
    -- equivalent to C's "%.6g", which Python's ".6g" format spec matches
    byte-for-byte (verified against combo-fmt-converter over 3.14, 3.0,
    0.0001, 0.00001, 1234567, 1e10, 1e-100, -0.0, ... -- see task-3-report.md).
    """
    return format(c, ".6g")


def _combo_it_to_str(t: combo_tree, node) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.h:69 ostream_combo_it

    The `case(output_format::combo):` branch (iostream_combo.h:75-85) only.
    """
    s = _vertex_to_str(node.data)
    children = list(t.children(node))
    if children:
        s += "(" + " ".join(_combo_it_to_str(t, child) for child in children) + ")"
    return s


def combo_tree_to_str(t: combo_tree) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:472 ostream_combo_tree

    Iterates `ct.begin()..ct.end()` appending a trailing " " after each
    top-level tree (iostream_combo.cc:475-479); since a combo_tree here is
    always single-rooted (pymetta/util/tree.py's module docstring), that
    loop runs at most once -- an empty tree (ct.begin() == ct.end()) prints
    nothing at all, exactly as the C++ loop body never executes for one.
    """
    root = t.begin()
    if root is None:
        return ""
    return _combo_it_to_str(t, root) + " "


# ---------------------------------------------------------------------------
# parsing: combo text -> combo_tree
# ---------------------------------------------------------------------------

# A token is a maximal run of characters that are none of '(', ')', or
# whitespace. C++: opencog/util/tree.cc:50 term / tree.cc:53 beg (the
# boost::spirit lexemes `+( anychar_p - ch_p('(') - ch_p(')') - space_p )`);
# the alternate "message:\"...\"" lexeme at tree.cc:45-49 is not ported --
# no vertex alternative in this port can ever hold a message (see the
# module docstring).
_TOKEN_RE = re.compile(r"[^()\s]+")


def _skip_ws(s: str, pos: int) -> int:
    while pos < len(s) and s[pos].isspace():
        pos += 1
    return pos


def _parse_expr(s: str, pos: int) -> tuple:
    """C++: opencog/util/tree.cc:54 expr

    `expr = (beg[&begin_internal] >> +expr >> ')'[&end_internal]) | term`,
    combined with the whitespace-before-open-paren hack that classic's own
    `operator>>` applies to the *whole string* before handing it to this
    grammar (opencog/util/tree.cc:96-120, "we must ignore whitespace after
    a function name, but before a parenthesis") -- ported here as a
    directly whitespace-tolerant recursive descent instead of a
    preprocessing pass, since the two are equivalent for any string this
    grammar can otherwise parse: skip whitespace before checking for '(',
    and skip whitespace right inside '(' too.

    That same C++ comment documents the "nothing but whitespace before the
    close paren" special case (tree.cc:106-115: "the operator is
    effectively child-less in this situation... some operators can be
    zero-ary") -- e.g. "and()" or "and(   )" parses as a bare leaf `and`
    with zero children, identical to a leaf `and` with no parens at all
    (confirmed against combo-fmt-converter: "and()" round-trips to "and").
    Without this case the grammar would otherwise recurse into `+expr`
    expecting at least one child and fail, or (per the C++ comment) treat
    the operator as its own child.

    Returns (vertex_value, [(child_vertex, child_children), ...], next_pos).
    """
    pos = _skip_ws(s, pos)
    match = _TOKEN_RE.match(s, pos)
    if not match:
        raise ValueError(f"expected a combo token at position {pos} in {s!r}")
    token = match.group(0)
    pos = match.end()

    lookahead = _skip_ws(s, pos)
    if lookahead >= len(s) or s[lookahead] != "(":
        return _token_to_vertex(token), [], pos

    pos = _skip_ws(s, lookahead + 1)
    if pos < len(s) and s[pos] == ")":
        # Nothing but whitespace inside the parens: zero-ary, same as a
        # bare leaf token (see docstring above).
        return _token_to_vertex(token), [], pos + 1

    children = []
    while True:
        child_value, grandchildren, pos = _parse_expr(s, pos)
        children.append((child_value, grandchildren))
        pos = _skip_ws(s, pos)
        if pos >= len(s):
            raise ValueError(f"unterminated '(' for token {token!r} in {s!r}")
        if s[pos] == ")":
            pos += 1
            break
    return _token_to_vertex(token), children, pos


def _materialize(t: combo_tree, node, children: list) -> None:
    """Append a parsed (vertex, grandchildren) forest as descendants of
    `node`, whose own `.data` the caller has already set via `tree(value)`
    or `t.append_child`. No direct C++ counterpart (classic builds the
    combo_tree by converting an already-built `tree<std::string>` node by
    node, sub_strtree_to_combo_tree, iostream_combo.h:191-201); this is the
    equivalent step for a parser that builds the combo_tree directly.
    """
    for child_value, grandchildren in children:
        child_node = t.append_child(node, child_value)
        _materialize(t, child_node, grandchildren)


def str_to_combo_tree(s: str) -> combo_tree:
    """C++: moses/comboreduct/combo/iostream_combo.cc:483 str2combo_tree

    The `labels` parameter is not in this task's scope (see module
    docstring) -- always the label-less, placeholder ($n) form.

    A combo_tree here is always single-rooted (pymetta/util/tree.py's
    module docstring), so exactly one top-level expression is parsed and
    any trailing non-whitespace content is a hard error -- unlike classic's
    strtree_to_combo_tree (iostream_combo.h:203-216), which loops over
    however many top-level sibling trees a `tree<std::string>` parsed and
    folds them under one implicit root.
    """
    value, children, pos = _parse_expr(s, 0)
    pos = _skip_ws(s, pos)
    if pos != len(s):
        raise ValueError(
            f"trailing content after the first combo expression at position "
            f"{pos} in {s!r} -- a combo_tree is single-rooted (see module docstring)"
        )
    t = combo_tree(value)
    _materialize(t, t.begin(), children)
    return t
