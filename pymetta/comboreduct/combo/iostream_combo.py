"""C++: moses/comboreduct/combo/iostream_combo.{h,cc}

Combo tree printing and parsing, scoped to the subset of combo-tree GRAMMAR
this port's parser recognises -- builtin | argument | contin_t token forms
only (see vertex.py). Stale as of Task 5, corrected here: `vertex.py`'s
`vertex` alias itself is no longer this narrow (Task 5 widened it to include
`str`/`enum_t`, for table columns -- see its module docstring), but nothing
in THIS module's grammar ever produces or expects an enum_t/message/ann_type
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

## S13 Task 3 addition: `output_format`, labels, python-format printing

Originally (S3) this module was scoped to `output_format::combo` only, with
no `labels` substitution -- both explicitly flagged in `_argument_to_str`'s
own docstring as "not in this task's scope". S13 Task 3 (`moses_exec.py`/
`metapop_printer`, `moses_main.py`) is the first caller that needs `-W`
(label substitution, `$arg1` instead of `$1`) and `--output-format=python`
(`test_output_labels`/`test_python_output`, `MOSESUTest.cxxtest`), so both
land here now: `output_format` (iostream_combo.h:42), `parse_output_format`
(iostream_combo.cc:252), and `labels`/`fmt` parameters threaded through
`_vertex_to_str`/`_argument_to_str`/`_builtin_to_str`/`_combo_it_to_str`/
`combo_tree_to_str`, mirroring `ostream_vertex`/`ostream_argument`/
`ostream_builtin`/`ostream_combo_it`/`ostream_combo_tree`'s own `labels`/
`fmt` parameters exactly. `output_format::scheme` printing is NOT ported
(no test in this port's scope needs it): `parse_output_format("scheme")`
still succeeds, matching C++ (the enum value is real and the string parses),
but every printing function raises `NotImplementedError` the moment it is
actually asked to render a `scheme`-format vertex or tree -- gate-then-raise
for an engaged-but-unported format, not a silent no-op, per this whole S13
slice's design rule. `output_format::python3` has NO string form in C++
itself: `parse_output_format` (iostream_combo.cc:252-263) is a plain
if/else-if chain over `"combo"`/`"python"`/`"scheme"` that never tests for
`"python3"`, so a CLI user cannot reach it that way in classic either --
confirmed by reading the function, not inferred. Ported faithfully:
`parse_output_format("python3")` falls through to the same `OC_ASSERT(false)`
path as any other unrecognized string (raised here as `ValueError`).
"""

import re
from collections.abc import Sequence
from enum import Enum, auto

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.comboreduct.combo.vertex import builtin, combo_tree

# ---------------------------------------------------------------------------
# output_format -- C++: moses/comboreduct/combo/iostream_combo.h:42
# ---------------------------------------------------------------------------


class output_format(Enum):
    """C++: moses/comboreduct/combo/iostream_combo.h:42 output_format

    `enum class output_format { combo, python, python3, scheme,
    output_format_count };` -- `output_format_count` (a C++ enum-counting
    idiom, "to get the number of formats" per its own comment) has no
    Python equivalent and is not ported: `len(output_format)` is the
    direct substitute for any caller that would need it, and nothing in
    this port ever did (grepped: zero references to
    `output_format_count` outside iostream_combo.h itself).
    """

    combo = auto()
    python = auto()
    python3 = auto()
    scheme = auto()


def parse_output_format(fmt_str: str) -> output_format:
    """C++: moses/comboreduct/combo/iostream_combo.cc:252 parse_output_format

        combo::output_format fmt = combo::output_format::combo;
        if (fmt_str == "combo") fmt = combo::output_format::combo;
        else if (fmt_str == "python") fmt = combo::output_format::python;
        else if (fmt_str == "scheme") fmt = combo::output_format::scheme;
        else OC_ASSERT(false, "Format %s not supported", fmt_str.c_str());
        return fmt;

    Ported verbatim, including the fact that `"python3"` is NOT one of the
    recognised strings (see module docstring) -- it raises exactly like any
    other unrecognized string, not like `"python"`. `OC_ASSERT(false, ...)`
    on bad CLI input is this port's usual `ValueError` (matching
    `problem_params.py`'s own convention for malformed option strings).
    """
    if fmt_str == "combo":
        return output_format.combo
    if fmt_str == "python":
        return output_format.python
    if fmt_str == "scheme":
        return output_format.scheme
    raise ValueError(f"Format {fmt_str} not supported")


# C++'s `enum class output_format { combo, python, python3, scheme,
# output_format_count };` has no explicit initializers, so the compiler
# assigns 0, 1, 2, 3 in declaration order -- what `static_cast<unsigned>`
# reads back in `ostream_builtin`/`ostream_argument`'s own "Format N
# unknown" fallbacks (see those two functions' docstrings for exactly
# where this is read). Python's `Enum` with `auto()` (the `output_format`
# class above) starts at 1, so `.value` alone would NOT reproduce these
# ordinals -- this table bridges that gap.
_OUTPUT_FORMAT_ORDINAL: dict[output_format, int] = {
    output_format.combo: 0,
    output_format.python: 1,
    output_format.python3: 2,
    output_format.scheme: 3,
}


def _output_format_to_int(fmt: output_format) -> int:
    """C++: moses/comboreduct/combo/iostream_combo.h:42 output_format

    `static_cast<unsigned>(fmt)` -- see `_OUTPUT_FORMAT_ORDINAL`'s own
    comment for why this indirection exists instead of `fmt.value`.
    """
    return _OUTPUT_FORMAT_ORDINAL[fmt]


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

# C++: moses/comboreduct/combo/iostream_combo.cc:266 ostream_builtin
#
# The `case output_format::python:` branch (iostream_combo.cc:326-354) only.
# Restricted, like `_BUILTIN_TO_STR` above, to the builtin members this
# port's `builtin` enum actually carries -- the C++ branch also handles
# `id::rand`/`id::list`/`id::car`/.../`id::equ` by falling to its own
# `default:` ("Builtin %u unknown"), so leaving them out of this table has
# the same observable effect (`_builtin_to_str`'s `.get(b, ...)` fallback
# below reproduces that default case) without needing dead entries.
_BUILTIN_TO_PYTHON_STR: dict[builtin, str] = {
    builtin.null_vertex: "null_vertex",
    builtin.logical_and: "and",
    builtin.logical_or: "or",
    builtin.logical_not: "not",
    builtin.logical_true: "True",
    builtin.logical_false: "False",
    builtin.plus: "adds",
    builtin.times: "muls",
    builtin.div: "pdiv",
    builtin.greater_than_zero: "l0",
    builtin.log: "log",
    builtin.exp: "exp",
    builtin.sin: "sin",
    builtin.impulse: "impulse",
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

    FIX WAVE (final review, Important): that AssertionError is now raised
    explicitly (`if arg == 0: raise ...`), not via a bare Python `assert`
    -- `OC_ASSERT` (iostream_combo.cc:167) is active in release C++
    builds, but a bare `assert` is stripped by `python -O`, which would
    have silently let `arg == 0` fall through to `argument(sign * arg)`
    below and hit `argument.__init__`'s own (also-now-explicit) check
    instead -- correct by accident, not by this function's own contract.
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
    if arg == 0:
        raise AssertionError("arg value should be different than zero.")
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


# C++ streams default to `defaultfloat` with precision 6, which is what a
# bare `std::cout << some_double` uses. That default is NOT universal: any
# caller that has done `out << std::setprecision(n)` changes it for
# everything subsequently written to that stream, INCLUDING contin
# constants nested inside a combo tree. `metapop_printer` is exactly such a
# caller (moses_main.h:110, `setprecision(io_score_precision)` = 18), so
# printing a candidate goes through `_contin_to_str` at precision 18, not 6.
DEFAULT_OSTREAM_PRECISION = 6


def _vertex_to_str(
    v,
    labels: Sequence[str] = (),
    fmt: output_format = output_format.combo,
    precision: int = DEFAULT_OSTREAM_PRECISION,
) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:428 ostream_vertex

    The argument/builtin/contin branches this port's narrowed vertex can
    ever hit, in the same order ostream_vertex itself checks them in
    (iostream_combo.cc:433-464, ending in the try/get<contin_t> fallback at
    463-467) -- ported as a `match` (mirroring boost::apply_visitor, per
    task-3-brief.md) rather than a chain of isinstance checks. `labels`/
    `fmt` (S13 Task 3 addition -- see module docstring) are threaded
    through to `_argument_to_str`/`_builtin_to_str` unchanged;
    `_contin_to_str` takes neither, matching ostream_vertex's own fallback
    (`out << get<contin_t>(v)`, iostream_combo.cc:463-464) printing a
    contin_t identically regardless of `f`.
    """
    match v:
        case argument():
            return _argument_to_str(v, labels, fmt)
        case builtin():
            return _builtin_to_str(v, fmt)
        case float():
            return _contin_to_str(v, precision)
        case _:
            raise TypeError(f"don't know how to print vertex {v!r} (combo format)")


def _argument_to_str(
    a: argument, labels: Sequence[str] = (), fmt: output_format = output_format.combo
) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:381 ostream_argument

    `case output_format::combo:` (iostream_combo.cc:386-403) branches on
    ABBREVIATE_NEGATIVE_LITERAL, #defined at iostream_combo.cc:33 with no
    corresponding #undef anywhere in the file -- so classic MOSES always
    takes the ostream_abbreviate_literal path ("!$n") for combo-format
    output, never the "not($n)" branch guarded by #else
    (iostream_combo.cc:389-402); only the abbreviated path is ported
    (`ostream_abbreviate_literal`, iostream_combo.cc:622-633: "!" prefix if
    negated else nothing, then "$" then either `abs_idx()` -- 1-based, no
    labels -- or `labels[abs_idx_from_zero()]` -- 0-based -- exactly what
    `argument.abs_idx`/`abs_idx_from_zero` (argument.py) compute).

    `case output_format::python:` (iostream_combo.cc:405-408) ignores
    `labels` entirely (classic's own python-format branch never reads its
    `labels` parameter either) and prints the argument as an index into a
    python function's `i` parameter: `not(i[-idx-1])` when negated, else
    `i[idx-1])` -- `idx` here is the raw SIGNED C++ member (`a.idx`, this
    port's `argument.idx`), not `abs_idx()`.

    # DIVERGENCE(C++ moses/comboreduct/combo/iostream_combo.cc:384-424):
    # NOT a divergence in this port -- recorded here because it is easy to
    # get wrong by analogy with `_combo_it_to_str`/`ostream_combo_it`
    # (iostream_combo.h:84-102), which DOES explicitly merge
    # `output_format::python` and `::python3` into one `case`. This
    # function's own C++ switch on `fmt` has exactly THREE cases --
    # `combo`, `python`, `scheme` -- each ending in `return`, with NO
    # `case output_format::python3:` label at all; `python3` therefore
    # falls to the switch's own `default:` (iostream_combo.cc:423-424),
    # `"Format " << static_cast<unsigned>(fmt) << "unknown"` (note: NO
    # space before "unknown" here, unlike `ostream_builtin`'s otherwise
    # near-identical default two functions down -- a real, verified
    # inconsistency between the two, not a transcription slip in this
    # docstring). Ported below by falling through to the same
    # `_UNKNOWN_FORMAT` branch `scheme` printing's own unported case does
    # NOT take -- `python3` is not "unimplemented", it behaves exactly
    # like any other value C++ itself does not recognise in this
    # particular function. Unreachable from any real CLI run either way:
    # `parse_output_format` (this module) never returns `python3` (see
    # its own docstring), so no port test can reach this path from
    # `moses_exec.py`; included for a caller that constructs
    # `output_format.python3` directly.
    """
    if fmt == output_format.combo:
        sign = "!" if a.is_negated() else ""
        idx_str = labels[a.abs_idx_from_zero()] if labels else str(a.abs_idx())
        return f"{sign}${idx_str}"
    if fmt == output_format.python:
        if a.is_negated():
            return f"not(i[{-a.idx - 1}])"
        return f"i[{a.idx - 1}]"
    if fmt == output_format.scheme:
        raise NotImplementedError(
            "ostream_argument: output_format.scheme printing is not ported "
            "(iostream_combo.cc:410-421); no test in this port's scope "
            "needs scheme-format output"
        )
    return f"Format {_output_format_to_int(fmt)}unknown"


def _builtin_to_str(b: builtin, fmt: output_format = output_format.combo) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:266 ostream_builtin

    `case output_format::combo:` (iostream_combo.cc:269-325) and
    `case output_format::python:` (iostream_combo.cc:326-354), each
    including its own `default:` fallback ("Builtin %u unknown") for a
    builtin member with no case of its own -- see `_BUILTIN_TO_STR`/
    `_BUILTIN_TO_PYTHON_STR`'s own docstrings for why that never actually
    fires for any member this port's `builtin` enum carries.

    Like `_argument_to_str` above, `output_format::python3` is NOT merged
    with `python` here: this function's outer switch on `f` has no
    `case output_format::python3:` label either, so it falls to the outer
    `default:` (iostream_combo.cc:374-375), `"Format " <<
    static_cast<unsigned>(f) << " unknown"` -- WITH a space this time,
    unlike `ostream_argument`'s otherwise near-identical default. See that
    function's own docstring for the full explanation; same unreachable-
    from-any-real-CLI-run caveat applies here too.
    """
    if fmt == output_format.combo:
        return _BUILTIN_TO_STR.get(b, f"Builtin {int(b)} unknown")
    if fmt == output_format.python:
        return _BUILTIN_TO_PYTHON_STR.get(b, f"Builtin {int(b)} unknown")
    if fmt == output_format.scheme:
        raise NotImplementedError(
            "ostream_builtin: output_format.scheme printing is not ported "
            "(iostream_combo.cc:355-370); no test in this port's scope "
            "needs scheme-format output"
        )
    return f"Format {_output_format_to_int(fmt)} unknown"


def _contin_to_str(c: contin_t, precision: int = DEFAULT_OSTREAM_PRECISION) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:428 ostream_vertex

    The fallback at the end of ostream_vertex, `out << get<contin_t>(v)`
    (iostream_combo.cc:463-464) -- reached once none of the earlier
    `get<T>(&v)` checks match, which for this port's narrowed vertex means
    "not an argument and not a builtin." Prints a bare double through the
    enclosing stream's float formatting: `defaultfloat` at whatever
    precision that stream currently carries, equivalent to C's "%.<p>g",
    which Python's ".{p}g" format spec matches byte-for-byte (verified
    against combo-fmt-converter at the default precision 6 over 3.14, 3.0,
    0.0001, 0.00001, 1234567, 1e10, 1e-100, -0.0, ...).

    `precision` used to be hardcoded at 6. That is right for a
    default-configured stream such as `combo-fmt-converter`'s, and wrong
    for `metapop_printer`'s, which sets 18. Measured divergence before the
    fix: `moses -Hsr -k3 -m600 -W1 -r9` printed
    `*(0.543044125185779492 $1)` in classic and `*(0.543044 $1)` here --
    same tree, same score, truncated constant. It surfaced only at that
    seed because every other run in the sweep produced constants like
    `0.5`/`2`/`-1`, which render identically at both precisions.
    """
    return format(c, f".{precision}g")


def _combo_it_to_str(
    t: combo_tree,
    node,
    labels: Sequence[str] = (),
    fmt: output_format = output_format.combo,
    precision: int = DEFAULT_OSTREAM_PRECISION,
) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.h:69 ostream_combo_it

    `case(output_format::combo):` (iostream_combo.h:75-85): vertex text
    then, if any children, "(" + space-joined recursive child text + ")".

    `case(output_format::python): case(output_format::python3):`
    (iostream_combo.h:86-101) -- the ONE place in this whole module where
    `python`/`python3` really are merged into one code path (contrast
    `_argument_to_str`/`_builtin_to_str`'s docstrings, where they are NOT):
    `logical_and`/`logical_or` are "infix" -- nothing is written for the
    vertex itself, only a separator (" and "/" or ", built by writing a
    leading and trailing space around the vertex's own python text) that
    joins the children inside parens; a childless infix vertex therefore
    prints as the EMPTY STRING (matches iostream_combo.h:88-93 precisely:
    `is_infix` writes only to the local `seperator` stream, never to
    `out`, and the `out << "("` block only runs `if
    (it.number_of_children() > 0)`). Every other vertex is "non-infix":
    its own python text IS written first, separator is ", ", and -- same
    as combo format -- children (if any) follow in parens.

    `case(output_format::scheme):` (iostream_combo.h:103-108) is not
    ported (module docstring); raises rather than silently doing
    something else.
    """
    if fmt == output_format.combo:
        s = _vertex_to_str(node.data, labels, fmt, precision)
        children = list(t.children(node))
        if children:
            s += (
                "("
                + " ".join(_combo_it_to_str(t, child, labels, fmt, precision) for child in children)
                + ")"
            )
        return s
    if fmt in (output_format.python, output_format.python3):
        v = node.data
        is_infix = isinstance(v, builtin) and v in (builtin.logical_and, builtin.logical_or)
        children = list(t.children(node))
        if is_infix:
            s = ""
            sep = " " + _vertex_to_str(v, labels, fmt, precision) + " "
        else:
            s = _vertex_to_str(v, labels, fmt, precision)
            sep = ", "
        if children:
            s += (
                "("
                + sep.join(_combo_it_to_str(t, c, labels, fmt, precision) for c in children)
                + ")"
            )
        return s
    if fmt == output_format.scheme:
        raise NotImplementedError(
            "ostream_combo_it: output_format.scheme printing is not ported "
            "(iostream_combo.h:103-108); no test in this port's scope needs "
            "scheme-format output"
        )
    raise ValueError(f"Unsupported case: {fmt!r}")


def tree_to_stream_str(t: combo_tree, precision: int = DEFAULT_OSTREAM_PRECISION) -> str:
    """C++: opencog/util/tree.h:3396 operator<<

    What `logger().debug() << some_combo_tree` actually produces. It is NOT
    `ostream_combo_tree`: `iostream_combo.h` declares `operator<<` for
    `ann_type`, `builtin`, `wild_card`, `argument` and `vertex`, but NOT for
    `combo_tree`, so streaming a tree resolves to the generic
    `operator<<(ostream&, const tree<T>&)` in cogutil's tree.h, which calls
    `subtree_to_string` (tree.h:3353) and prints each node through
    `operator<<(ostream&, const vertex&)`.

    The observable difference from `combo_tree_to_str` is the TRAILING
    SPACE: `ostream_combo_tree` appends one after every top-level tree
    (iostream_combo.cc:478, unconditionally), `subtree_to_string` does not.
    Measured against classic's own log 2026-09-09: `[DEBUG] Exemplar tree :
    true` and `[DEBUG] and($1 $2)` both end at the last character of the
    tree, with no trailing blank.

    The two agree on everything else for `output_format.combo` with no
    labels -- `subtree_to_string`'s `node(child child ...)` shape is the same
    shape `ostream_combo_it` builds, and both delegate leaf rendering to the
    same `vertex` printer -- so this is implemented by reusing
    `_combo_it_to_str` rather than by writing a second tree walker.
    """
    root = t.begin()
    if root is None:
        return ""
    return _combo_it_to_str(t, root, (), output_format.combo, precision)


def combo_tree_to_str(
    t: combo_tree,
    labels: Sequence[str] = (),
    fmt: output_format = output_format.combo,
    precision: int = DEFAULT_OSTREAM_PRECISION,
) -> str:
    """C++: moses/comboreduct/combo/iostream_combo.cc:472 ostream_combo_tree

    Iterates `ct.begin()..ct.end()` appending a trailing " " after each
    top-level tree UNCONDITIONALLY, regardless of `fmt` (iostream_combo.cc:
    475-479, `it.skip_children(); out << " ";` runs after every top-level
    tree with no format check at all); since a combo_tree here is always
    single-rooted (pymetta/util/tree.py's module docstring), that loop
    runs at most once -- an empty tree (ct.begin() == ct.end()) prints
    nothing at all, exactly as the C++ loop body never executes for one.
    `labels`/`fmt` (S13 Task 3 addition -- see module docstring) default
    to empty/`output_format.combo`, matching every existing caller's
    label-free combo-format-only usage exactly.
    """
    root = t.begin()
    if root is None:
        return ""
    return _combo_it_to_str(t, root, labels, fmt, precision) + " "


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
