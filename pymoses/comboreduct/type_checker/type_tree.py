"""C++: moses/comboreduct/type_checker/type_tree.h and type_tree.cc

The combo type-inference engine.

This module REPLACES the narrow S7 slice that used to live here. That
slice defined `type_tree` as a single-`type_node` value class and ported
three branches of `get_output_type_tree`; `docs/MODULE-COVERAGE.md:213`
already said in as many words that a real port "needs a genuinely
different, more general representation (most naturally built on this
port's own `pymoses.util.tree.tree`/`tree_node`) under either a different
name or a deliberate, documented replacement of the S7 class, not a silent
second definition." This is that deliberate replacement: `type_tree` is
now `tree[type_node]`, exactly as `type_tree_def.h:89` says it is, and the
41 unported definitions of `type_tree.cc` plus the two of
`type_tree_def.cc` are ported here.

The two things S7's class gave callers are both preserved, so the
replacement is source-compatible for every existing call site:

* `type_tree(some_type_node)` still builds a one-node tree, which is what
  `meta_rules.py:143` and `mixed_rules.py:1228` construct.
* `type_tree.node` (below) still reads that one node's `type_node`, which
  is what `demo_problems.py` and `table_problems.py` do to
  `get_output_type_tree(...)`. It is a port-local accessor with no C++
  analogue -- classic writes `*tt.begin()` -- kept precisely so this
  replacement needs no edit in files this task does not own.

## What is ported, and what is gated

Boolean, continuous and action-RESULT typing are ported in full: every
function below is generic over `type_node`, and `action_result_type`
flows through `type_tree_arity`, `action_result_arity`,
`inherit_type_tree`, `get_intersection`, `reduce_type_tree`,
`gen_signature` and both stream operators with no special case. The
fixture exercises it (`inherit`'s `action_result` rows,
`action_result_arity` on every program).

What is NOT ported, per branch, and why -- this list is about the
per-VERTEX dispatch functions (`get_type_tree`, `get_output_type_tree`,
`get_input_type_tree`, `get_arity`), which are the only place a domain
can enter at all:

* `is_action` / `is_builtin_action` / `is_perception` / `is_procedure_call`
  (type_tree.cc:220-232, :258-263, :294-299, and their `get_arity` twins
  at :410-416, :423). These are not "excluded inference paths" -- they are
  unreachable for want of a prerequisite TYPE. This port's `vertex` is
  `builtin | argument | contin_t | str` (vertex.py's own module
  docstring); there is no `action`, `builtin_action`, `perception` or
  `procedure_call` alternative, and no `is_action`/`is_perception`/
  `is_builtin_action` predicate exists anywhere in the package. A vertex
  therefore cannot BE one of these, so no branch can be written for it
  that any input could reach. See `descriptions.py`'s own module docstring
  for the same conclusion about `actions_properties`, and the `actions`
  section of `tests/fixtures/type_tree.json` for a measured record of what
  classic's action table actually holds.
* `is_enum_type` (type_tree.cc:213), `is_definite_object` (:226, :264),
  `is_message` (:230, :270), `is_indefinite_object` (:228, :266),
  `is_wild_card` (:234, :268) and `is_action_symbol` (:232). All six are
  spelled `str` in this port's `vertex` union, which cannot tell them
  apart, and all six domains -- enum, object, message -- are out of scope.
  A `str` vertex raises here rather than guessing which of the six it
  stands in for. This is the SAME behaviour the S7 slice already had
  (`get_output_type_tree` raised `AssertionError` for any `str`), so no
  caller changes.
* `ann_type` has no `type_node` case in classic's own stream operator
  (:1360's switch prints `UNKNOWN_HANDLE` for it) and no vertex
  alternative here; nothing to gate.

`list`/`car`/`cdr`/`cons`/`foldr`/`foldl` (list and fold),
`lambda`/`apply` (higher-order) and `equ` (enum) DO have rows in
`descriptions.py`'s builtin table, and their signatures are therefore
inferred correctly by the generic machinery below. That is deliberate:
those rows are table DATA, and dropping them would desynchronise the table
from `id::builtin_count` and make `get_type_tree(builtin.cond)` silently
wrong rather than absent. See `descriptions.py`.

## Relationship to `type_signature.py`

`type_signature.py` holds `type_tree_def.h`'s `type_node` enum, the port's
own flat `TypeSignature` value (a deliberate, marked divergence from
classic's `lambda_type`-rooted tree), and `TypeSignature`-shaped versions
of the arity and signature accessors. Those now DELEGATE to the real
algorithms in this module rather than reimplementing them, so there is one
implementation of each. This module's versions are the C++ ones and take a
real `type_tree`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.common_def import arity_t, contin_t
from pymoses.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    is_argument,
    is_builtin,
    is_contin,
    vertex,
)
from pymoses.comboreduct.type_checker.type_signature import (
    EXCLUDED_STR_VERTEX_MESSAGE,
    arg_to_idx,
    argument_type_node,
    is_argument_type,
    ostream_type_node,
    type_node,
)
from pymoses.util.tree import tree, tree_node

__all__ = [
    "abs_min_arity",
    "action_result_arity",
    "argument_type_list_input_type",
    "boolean_arity",
    "contin_arity",
    "convert_index",
    "default_vertex_value",
    "does_contain_all_arg_up_to",
    "equal_type_tree",
    "explicit_arity",
    "gen_signature",
    "get_arg_type",
    "get_argument_abs_idx_from_zero_set",
    "get_argument_abs_idx_set",
    "get_arity",
    "get_input_type_tree",
    "get_intersection",
    "get_output_type_node",
    "get_output_type_tree",
    "get_signature_inputs",
    "get_signature_output",
    "get_type_node",
    "get_type_tree",
    "infer_arg_type_tree",
    "infer_arity",
    "infer_type_tree",
    "infer_vertex_type",
    "inherit_type_tree",
    "insert_arg_type_tree",
    "is_well_formed",
    "istream_type_node",
    "ostream_type_tree",
    "reduce_type_tree",
    "set_arg_type",
    "str_to_type_tree",
    "type_tree",
    "type_tree_arity",
]


class type_tree(tree):
    """C++: moses/comboreduct/type_checker/type_tree_def.h:89 type_tree

    `typedef opencog::tree<type_node> type_tree`. A real n-ary tree of
    `type_node`, as classic's typedef says: `boolean`, `->(contin
    boolean)`, `application(->(arg_list(contin) contin) contin contin)`
    and `union(boolean contin)` are all `type_tree` values, and the engine
    below rewrites them in place.

    See this module's own docstring for why this REPLACES the earlier
    single-`type_node` value class of the same name, and what was kept
    source-compatible.
    """

    def __init__(self, root_data: type_node | tree_node | type_tree | None = None) -> None:
        """C++: opencog/util/tree.h:646 tree

        Three C++ constructors collapse here, the same way
        `pymoses.util.tree.tree.__init__` already collapses them:
        `type_tree()` (empty), `type_tree(id::boolean_type)` (a one-node
        tree) and `explicit tree(const iterator_base&)` (a deep copy of
        that node's whole subtree -- classic writes `type_tree(it1)` all
        over `get_intersection` and `get_signature_output`). The copy
        constructor `type_tree(other)` is folded in as a fourth case,
        since a `type_tree` is not a `tree_node` and would otherwise be
        stored as a root VALUE.
        """
        if isinstance(root_data, type_tree):
            super().__init__(root_data.begin())
            return
        super().__init__(root_data)

    @property
    def node(self) -> type_node:
        """C++: moses/comboreduct/type_checker/type_tree_def.h:89 type_tree

        No C++ analogue -- classic writes `*tt.begin()`. Kept from the S7
        slice this module replaces, because `demo_problems.py` and
        `table_problems.py` both read `get_output_type_tree(v).node`, and
        neither file belongs to this task. Unlike `get_type_node` (:372)
        this does NOT assert the root is childless, so it reads the ROOT
        of a compound type rather than refusing; every current caller
        passes a `get_output_type_tree` result, which is always a leaf.
        """
        head = self.begin()
        if head is None:
            raise ValueError("type_tree.node: tree is empty")
        return head.data

    def __eq__(self, other: object) -> bool:
        """C++: opencog/util/tree.h:2916 tree

        The real symbol is the free `operator==(const tree<T>&, const
        tree<T>&)` defined on that line; the crumb cites the class it
        compares, since `tools/check_alignment.py` cannot match an
        operator name.

        `t1.empty() ? t2.empty() : (t2.empty() ? false :
        t1.equal_subtree(t1.begin(), t2.begin()))` -- full structural
        comparison, not just the root. `type_node` is a plain `IntEnum`,
        so CLAUDE.md's "type before value" trap (which is about
        `contin_t`/`builtin` sharing Python's `==`) does not arise
        between two `type_node`s.
        """
        if not isinstance(other, type_tree):
            return NotImplemented
        if self.empty():
            return other.empty()
        if other.empty():
            return False
        return self.equal_subtree(self.begin(), other.begin())

    def __hash__(self) -> int:
        """C++: opencog/util/tree.h:2916 tree

        No C++ analogue (`tree<T>` is not hashable); carried over from the
        S7 class this replaces, and widened to agree with the structural
        `__eq__` above rather than hashing the root alone.
        """
        return hash(tuple((n.data, self.number_of_children(n)) for n in self))

    def __repr__(self) -> str:
        return f"type_tree({ostream_type_tree(self)!r})"


# ---------------------------------------------------------------------------
# Printing and parsing
# ---------------------------------------------------------------------------


def _subtree_to_string(tr: tree, it: tree_node) -> str:
    """C++: opencog/util/tree.h:3353 subtree_to_string

    A childless node prints bare; a node with children prints
    `node(child child ...)` with single spaces and no trailing space.
    """
    n_children = tr.number_of_children(it)
    if n_children == 0:
        return ostream_type_node(it.data)
    parts = [_subtree_to_string(tr, sib) for sib in tr.children(it)]
    return ostream_type_node(it.data) + "(" + " ".join(parts) + ")"


def ostream_type_tree(tt: tree) -> str:
    """C++: opencog/util/tree.h:3305 tree

    The real symbol is the free `operator<<(std::ostream&, const
    tree<T>&)` defined on that line; the crumb cites the class it prints,
    since `tools/check_alignment.py` cannot match an operator name.

    `sibling_iterator it = tr.begin(); if (it != tr.end()) out <<
    subtree_to_string(it);` -- so an EMPTY tree prints as the empty
    string, which is exactly how `get_input_type_tree`'s past-the-end
    `return type_tree()` shows up in the fixture.
    """
    head = tt.begin()
    if head is None:
        return ""
    return _subtree_to_string(tt, head)


def istream_type_node(s: str) -> type_node:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1409 type_node

    The real symbol is the free `operator>>(std::istream&,
    opencog::combo::type_node&)` defined on that line; the crumb cites the
    type it reads, since `tools/check_alignment.py` cannot match an
    operator name.

    Note two things classic does that look like bugs and are not being
    "fixed" here. `"$-1"` is accepted -- `boost::lexical_cast<int>("-1")`
    succeeds, the `arg != 0` assert passes, and the result is
    `argument_type + (-1) - 1 == unknown_type`; measured, and pinned by
    the fixture's `type_node_io.istream` section. And `"list"` maps to
    `id::list_type` on the way IN even though `operator<<` has no case for
    it on the way out, so `list` does not round-trip.
    """
    if not s:
        raise AssertionError("str representation must not be empty.")

    # Type operators
    if s in ("->", "lambda", "lambda_type"):
        return type_node.lambda_type
    if s in ("application", "application_type"):
        return type_node.application_type
    if s in ("union", "union_type"):
        return type_node.union_type
    if s in ("arg_list", "arg_list_type"):
        return type_node.arg_list_type

    # Elementary types
    if s in ("boolean", "boolean_type"):
        return type_node.boolean_type
    if s in ("contin", "contin_t", "contin_type"):
        return type_node.contin_type
    if s in ("enum", "enum_t", "enum_type"):
        return type_node.enum_type
    if s in ("list", "list_t", "list_type"):
        return type_node.list_type

    # Types for motor control, sensory data.
    if s in ("action_result", "action_result_type"):
        return type_node.action_result_type
    if s in ("definite_object", "definite_object_type"):
        return type_node.definite_object_type
    if s in ("action_definite_object", "action_definite_object_type"):
        return type_node.action_definite_object_type
    if s in ("indefinite_object", "indefinite_object_type"):
        return type_node.indefinite_object_type
    if s in ("message", "message_type"):
        return type_node.message_type
    if s in ("action_symbol", "action_symbol_type"):
        return type_node.action_symbol_type
    if s in ("wild_card", "wild_card_type"):
        return type_node.wild_card_type
    if s in ("unknown", "unknown_type"):
        return type_node.unknown_type
    if s in ("ill_formed", "ill_formed_type"):
        return type_node.ill_formed_type
    if s[0] == "$":
        arg = int(s[1:])
        if arg == 0:
            raise AssertionError("arg value should be different from zero.")
        return argument_type_node(arg - 1)
    raise AssertionError(f"type tree input stream fatal error : {s} no such case")


def _collapse_empty_parens(s: str) -> str:
    """C++: opencog/util/tree.cc:70 tree

    The real symbol is the free `operator>>(std::istream&,
    opencog::tree<std::string>&)` defined on that line; the crumb cites
    the class it fills, since `tools/check_alignment.py` cannot match an
    operator name.

    Classic's own hand-written pre-pass (tree.cc:93-119), reproduced
    because it is load-bearing rather than cosmetic: it drops whitespace
    that precedes an open paren (so `"and  ($1 $2)"` parses as
    `and($1 $2)`), and it drops BOTH parens of a `(` ... `)` containing
    only whitespace (so `"+()"` becomes `"+"`, a childless operator,
    rather than the `"+(+)"` classic's own comment says the parser would
    otherwise produce). That second half is why `and()` infers as
    `->(arg_list(boolean) boolean)` with arity -1, which the fixture
    records.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n and s[i] in " \t":
        i += 1
    while i < n:
        ix = i
        while ix < n and s[ix] in " \t":
            ix += 1
        if ix < n and s[ix] == "(":
            i = ix
            ix += 1
            while ix < n and s[ix] in " \t":
                ix += 1
            if ix < n and s[ix] == ")":
                ix += 1
                i = ix
        if i >= n:
            break
        out.append(s[i])
        i += 1
    return "".join(out)


def str_to_type_tree(s: str) -> type_tree:
    """C++: opencog/util/tree.h:3312 tree

    The real symbol is the free `operator>>(std::istream&,
    opencog::tree<T>&)` defined on that line; the crumb cites the class it
    fills, since `tools/check_alignment.py` cannot match an operator name.

    Classic reads the text into a `tree<std::string>` with a boost::spirit
    grammar (tree.cc:34-61) and then `tree_convert`s it (tree.h:3142),
    which `boost::lexical_cast<type_node>`s every node -- i.e. runs
    `istream_type_node` on it.

    # DIVERGENCE(C++ opencog/util/tree.cc:34): the boost::spirit::classic
    # grammar is transliterated as a recursive-descent parser over the
    # same token rule (a token is a maximal run of characters that are
    # not '(', ')' or whitespace; `beg` is such a token immediately
    # followed by '('), rather than reproduced through a parser combinator
    # library this port does not depend on. The `message:"..."`
    # alternative of that grammar (tree.cc:44-47, classic's own comment
    # calls it "A HACK") is NOT reproduced: it exists to parse `has_said`
    # PERCEPTIONS, an excluded embodiment domain, and no `type_node`
    # spelling contains a quote or a space. The paren-balance multi-line
    # read loop (tree.cc:74-82) is also not reproduced: it exists to keep
    # reading further LINES off an istream, and this function takes a
    # complete string.
    """
    text = _collapse_empty_parens(s)
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t":
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < n and text[j] not in "() \t":
            j += 1
        tokens.append(text[i:j])
        i = j

    pos = 0

    def parse_expr(tt: type_tree, parent: tree_node | None) -> tree_node:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"str_to_type_tree: unexpected end of input in {s!r}")
        tok = tokens[pos]
        if tok in "()":
            raise ValueError(f"str_to_type_tree: unexpected {tok!r} in {s!r}")
        pos += 1
        node_value = istream_type_node(tok)
        node = tt.set_head(node_value) if parent is None else tt.append_child(parent, node_value)
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1
            if pos < len(tokens) and tokens[pos] == ")":
                raise ValueError(f"str_to_type_tree: empty argument list in {s!r}")
            while pos < len(tokens) and tokens[pos] != ")":
                parse_expr(tt, node)
            if pos >= len(tokens):
                raise ValueError(f"str_to_type_tree: unbalanced parens in {s!r}")
            pos += 1
        return node

    result = type_tree()
    if not tokens:
        return result
    parse_expr(result, None)
    if pos != len(tokens):
        raise ValueError(f"str_to_type_tree: trailing content in {s!r}")
    return result


# ---------------------------------------------------------------------------
# Per-vertex facts
# ---------------------------------------------------------------------------


def default_vertex_value(tn: type_node) -> vertex:
    """C++: moses/comboreduct/type_checker/type_tree.cc:35 default_vertex_value

    The `default:` arm logs an error and returns a default-constructed
    `vertex()`. `vertex` is a `boost::variant` whose FIRST alternative is
    `contin_t` (vertex.h:126), so that default is the contin `0.0`, not a
    null or an exception -- measured, and pinned by the fixture's
    `misc.default_vertex_value` section, where every ordinal except
    `boolean_type` prints "0". The `logger().error()` call is dropped, as
    everywhere in this port.
    """
    if tn == type_node.contin_type:
        return contin_t(0.0)
    if tn == type_node.boolean_type:
        return builtin.logical_false
    return contin_t(0.0)


def _get_type_tree_vertex(v: vertex) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:201 get_type_tree

    The `const vertex&` overload -- classic's own comment calls it a
    "Giant case statement". Three of its thirteen arms are reachable in
    this port; see this module's docstring for the other ten, per branch.

    Note the shape of the final `else` (cc:243-247): its `OC_ASSERT(false)`
    is COMMENTED OUT and it returns `unknown_type`. That arm is
    unreachable here anyway -- `vertex` has exactly four alternatives and
    the fourth (`str`) is caught by the raise below.
    """
    from pymoses.comboreduct.combo.descriptions import builtins_properties

    if is_builtin(v):
        # cc:53, get_type_tree(builtin)
        return builtins_properties.instance().type_tree_of_builtin(v)
    if is_argument(v):
        # cc:158, get_type_tree(const argument&)
        return type_tree(argument_type_node(v.abs_idx_from_zero()))
    if is_contin(v):
        # cc:163, get_type_tree(contin_t)
        return type_tree(type_node.contin_type)
    raise NotImplementedError(EXCLUDED_STR_VERTEX_MESSAGE.format(fn="get_type_tree", v=v))


def get_type_tree(tr: vertex | tree, it: tree_node | None = None) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1169 get_type_tree

    Six C++ overloads collapse into one function, dispatching on the
    argument's Python type the way this port does everywhere overloads
    meet (see `pymoses.util.tree.tree.__init__`'s own docstring):

    * `get_type_tree(const combo_tree&)` (cc:1169) and
      `get_type_tree(const combo_tree&, combo_tree::iterator)` (cc:1177)
      -- when `tr` is a tree. The result is the UNREDUCED
      `application(...)` shape, not a signature.
    * `get_type_tree(const vertex&)` (cc:201) and, through it,
      `get_type_tree(builtin)` (cc:53), `get_type_tree(const argument&)`
      (cc:158) and `get_type_tree(contin_t)` (cc:163) -- when `tr` is a
      vertex.

    cc:1169 returns an EMPTY type_tree for an empty combo_tree; cc:1177
    asserts on one.
    """
    if not isinstance(tr, tree):
        if it is not None:
            raise TypeError("get_type_tree: `it` is only meaningful for a combo_tree")
        return _get_type_tree_vertex(tr)

    if it is None:
        # cc:1169
        tmp = type_tree()
        if not tr.empty():
            tmp = get_type_tree(tr, tr.begin())
        return tmp

    # cc:1177
    if tr.empty():
        raise AssertionError("tr cannot be empty")
    if not tr.is_valid(it):
        raise AssertionError("it must be valid")
    tmp = _get_type_tree_vertex(it.data)
    if tmp.empty():
        raise AssertionError("tmp cannot be empty")
    if it.first_child is None:
        return tmp
    head = tmp.begin()
    head = tmp.wrap(head, type_node.application_type)
    for sib in tr.children(it):
        arg_app = get_type_tree(tr, sib)
        tmp.append_child(head, arg_app.begin())
    return tmp


def get_output_type_tree(v: vertex) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:250 get_output_type_tree

    Collapses `get_output_type_tree(const vertex&)` (cc:250) with
    `get_output_type_tree(builtin)` (cc:59), which is the only one of its
    delegates this port can reach.

    Three of the eleven arms are reachable: `is_builtin` (cc:251-252),
    `is_argument` (cc:253-254, an argument's OUTPUT type is ALWAYS
    `unknown_type` -- its type is pinned down only by its ENCLOSING
    expression, never in isolation) and `is_contin` (cc:255-256).

    The final `else` (cc:273-279) is a real `OC_ASSERT(false, "Unhandled
    case '%s'")`, unlike `get_type_tree`'s commented-out one, and that is
    what a `str` vertex hits here.
    """
    from pymoses.comboreduct.combo.descriptions import builtins_properties

    if is_builtin(v):
        return type_tree(builtins_properties.instance().output_type_of_builtin(v))
    if is_argument(v):
        return type_tree(type_node.unknown_type)
    if is_contin(v):
        return type_tree(type_node.contin_type)
    raise AssertionError(f"get_output_type_tree: Unhandled case '{v}'")


def get_output_type_node(v: vertex) -> type_node:
    """No C++ counterpart: the single node of `get_output_type_tree(v)`.

    Every reachable arm of `get_output_type_tree` returns a one-node tree,
    so `get_output_type_tree(v) == type_tree(t)` holds exactly when
    `get_output_type_node(v) == t`. Typed `downwards` asks that once per
    visited node, and this answers it without building two trees.
    """
    from pymoses.comboreduct.combo.descriptions import builtins_properties

    if is_builtin(v):
        return builtins_properties.instance().output_type_of_builtin(v)
    if is_argument(v):
        return type_node.unknown_type
    if is_contin(v):
        return type_node.contin_type
    return get_output_type_tree(v).node  # raises, as the tree version does


def get_input_type_tree(v: vertex, i: arity_t) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:283 get_input_type_tree

    Collapses `get_input_type_tree(const vertex&, arity_t)` (cc:283) with
    `get_input_type_tree(builtin, arity_t)` (cc:65).

    The builtin arm has three cases: an arity of -1 (a variadic
    `arg_list(T)` operator such as `and`/`or`/`plus`/`times`) always
    answers with input 0, whatever `i` is; an `i` below the arity answers
    with that input; and anything past the end returns an EMPTY
    `type_tree`, which prints as the empty string. A contin vertex also
    returns the empty tree (cc:287-288); an argument returns
    `unknown_type` (cc:285-286).
    """
    from pymoses.comboreduct.combo.descriptions import builtins_properties

    if is_builtin(v):
        props = builtins_properties.instance()
        if props.builtin_arity(v) == -1:
            return props.builtin_argument(v, 0)
        if i < props.builtin_arity(v):
            return props.builtin_argument(v, i)
        return type_tree()
    if is_argument(v):
        return type_tree(type_node.unknown_type)
    if is_contin(v):
        return type_tree()
    raise AssertionError(f"get_input_type_tree: Unhandled case '{v}'")


def get_arity(v: vertex) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:403 get_arity

    Collapses `get_arity(const vertex&)` (cc:403) with `get_arity(builtin)`
    (cc:47). "+1 if it is an argument, 0 if it is a constant, and -1 if it
    is an arg list" (type_tree.h:392). The final `else` is an
    `OC_ASSERT(false, "Unhandled case")`.
    """
    from pymoses.comboreduct.combo.descriptions import builtins_properties

    if is_builtin(v):
        return builtins_properties.instance().builtin_arity(v)
    if is_argument(v):
        return 1
    if is_contin(v):
        return 0
    raise AssertionError("get_arity: Unhandled case")


# ---------------------------------------------------------------------------
# Arity of a signature
# ---------------------------------------------------------------------------


def _count_inputs_of_type(ty: type_tree, wanted: type_node) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:316 contin_arity

    `contin_arity` (cc:316), `boolean_arity` (cc:328) and
    `action_result_arity` (cc:340) are the same eight lines three times
    over, differing only in the `type_node` they count; the shared body
    lives here and each keeps its own C++ name and breadcrumb below.
    """
    res: arity_t = 0
    ty_it = ty.begin()
    if ty_it is not None and ty_it.data == type_node.lambda_type:
        sib = ty_it.first_child
        while sib is not None and sib is not ty_it.last_child:
            if sib.data == wanted:
                res += 1
            sib = sib.next_sibling
    return res


def contin_arity(ty: type_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:316 contin_arity"""
    return _count_inputs_of_type(ty, type_node.contin_type)


def boolean_arity(ty: type_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:328 boolean_arity"""
    return _count_inputs_of_type(ty, type_node.boolean_type)


def action_result_arity(ty: type_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:340 action_result_arity"""
    return _count_inputs_of_type(ty, type_node.action_result_type)


def type_tree_arity(ty: type_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:352 type_tree_arity

    Given `->(T_1 ... T_n T)`, returns `n` -- or `-n` when `T_n` is
    `arg_list(T')`, the negative being a deliberate "variadic" sentinel
    that `convert_index` (cc:380) and `abs_min_arity` (cc:389) both read.
    A non-lambda type has arity 0.

    That sentinel IS reachable from a real combo program: `and()` parses
    to a childless `and` (see `_collapse_empty_parens`), whose inferred
    signature is `->(arg_list(boolean) boolean)` and whose arity is -1.
    """
    if ty.empty():
        raise AssertionError("Not sure this assert should not be replaced by a conditional")
    ty_it = ty.begin()
    if ty_it.data == type_node.lambda_type:
        noc = ty.number_of_children(ty_it)
        if noc <= 0:
            raise AssertionError("Lambda must not be childless")
        ty_last_child = ty_it.last_child
        if noc == 1:
            return 0
        a: arity_t = noc - 1
        last_arg = ty.previous_sibling(ty_last_child)
        return -a if last_arg.data == type_node.arg_list_type else a
    return 0


def get_type_node(tt: type_tree) -> type_node:
    """C++: moses/comboreduct/type_checker/type_tree.cc:372 get_type_node

    "Return the root of the type tree, convenient function when one
    assumes that the type is elementary" (type_tree.h:371). Both
    `OC_ASSERT`s are real: an empty tree and a tree whose root has
    children both abort classic (the fixture records the second as
    `threw`).
    """
    if tt.empty():
        raise AssertionError("get_type_node: tt must not be empty")
    head_it = tt.begin()
    if head_it.first_child is not None:
        raise AssertionError("get_type_node: head must be childless")
    return head_it.data


def convert_index(arity: arity_t, index: arity_t) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:380 convert_index

    For a positive (explicit) arity the index is unchanged; for a negative
    (variadic) arity every index at or past `|arity|` is clamped to the
    LAST declared input, `|arity| - 1`.
    """
    ap = -arity
    if ap > 0 and index >= ap:
        return ap - 1
    return index


def abs_min_arity(arity: arity_t) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:389 abs_min_arity"""
    if arity < 0:
        return -arity - 1
    return arity


def argument_type_list_input_type(
    atl: Sequence[type_tree], arity: arity_t, index: arity_t
) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:396
    argument_type_list_input_type

    # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:400):
    # classic is a bare `atl[convert_index(arity, index)]` on a
    # `std::vector`, so an out-of-range converted index is undefined
    # behaviour; Python raises `IndexError` instead. Reproducing UB is not
    # possible and would be worse than failing -- the same call this
    # project already made for `interpreter.cc:54` (see
    # docs/CARRY-FORWARD.md section 3).
    """
    return atl[convert_index(arity, index)]


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def equal_type_tree(ty1: type_tree, ty2: type_tree) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree.cc:435 equal_type_tree

    "returns true iff ty1 inherits ty2 and ty2 inherits ty1"
    (type_tree.h:119). NOT the same as `==`: `->(boolean boolean)` and
    `->(unknown boolean)` are `equal_type_tree` (the `unknown_type`
    exception documented at type_tree.h:143) while comparing unequal
    structurally.
    """
    return inherit_type_tree(ty1, ty2) and inherit_type_tree(ty2, ty1)


def inherit_type_tree(
    ty1: type_tree,
    ty2: type_tree | tree_node,
    ty2_or_none: type_tree | None = None,
    it2: tree_node | None = None,
) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree.cc:440 inherit_type_tree

    Both overloads: the two-argument whole-tree form (cc:440) and the
    four-argument "consider it1 and it2 as roots" form (cc:446), which is
    the recursive one. Called as `inherit_type_tree(ty1, ty2)` or
    `inherit_type_tree(ty1, it1, ty2, it2)`.

    The rules are laid out at length in type_tree.h:135-166. The one worth
    restating is its third: `unknown_type` on the RIGHT of a lambda input
    position makes the comparison succeed regardless, which is why
    `->(boolean boolean)` and `->(unknown boolean)` inherit BOTH ways.
    """
    if ty2_or_none is None:
        return inherit_type_tree(ty1, ty1.begin(), ty2, ty2.begin())

    it1 = ty2
    ty2 = ty2_or_none

    # --------------
    # recursive case
    # --------------

    # union case
    if it1.data == type_node.union_type:
        # all type in the union must inherit from ty2 at it2
        for sib in ty1.children(it1):
            if not inherit_type_tree(ty1, sib, ty2, it2):
                return False
        return True
    if it2.data == type_node.union_type:
        if it1.data == type_node.union_type:
            raise AssertionError(
                "Due to the recursive structure of that function"
                " it is impossible that *it1 is id::union_type"
            )
        # the type of it1 must inherit at least one type of the union of it2
        for sib in ty2.children(it2):
            if inherit_type_tree(ty1, it1, ty2, sib):
                return True
        return False
    # lambda case
    if it1.data == type_node.lambda_type and it2.data == type_node.lambda_type:
        if ty1.number_of_children(it1) == ty2.number_of_children(it2):
            # all input arguments sib2 must inherit sib1, or sib2 must be
            # unknown_type, and output of ty1 must inherit from ty2
            sib1 = it1.first_child
            sib2 = it2.first_child
            while sib2 is not it2.last_child:
                if (
                    not inherit_type_tree(ty2, sib2, ty1, sib1)
                    and sib2.data != type_node.unknown_type
                ):
                    return False
                sib1 = sib1.next_sibling
                sib2 = sib2.next_sibling
            return bool(inherit_type_tree(ty1, sib1, ty2, sib2))
        return False
    # application case
    if it1.data == type_node.application_type:
        return False  # from the assumption that type_tree must be first reduced
    # arg_list case
    if it1.data == type_node.arg_list_type and it2.data == type_node.arg_list_type:
        if not (ty1.has_one_child(it1) and ty2.has_one_child(it2)):
            raise AssertionError("arg_list_type takes only one argument")
        return inherit_type_tree(ty1, it1.first_child, ty2, it2.first_child)
    # ----------
    # base cases
    # ----------
    # Nothing can inherit or be inherited from an ill-formed type
    if it1.data == type_node.ill_formed_type or it2.data == type_node.ill_formed_type:
        return False
    # Everything except ill_formed_type inherits from unknown_type
    if it2.data == type_node.unknown_type:
        return True
    # All other cases.
    return it1.data == it2.data


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def reduce_type_tree(
    tt: type_tree,
    arg_types: Sequence[type_tree] | None = None,
    tr: combo_tree | None = None,
    proc_name: str = "PROCEDURE NAME UNKNOWN",
) -> None:
    """C++: moses/comboreduct/type_checker/type_tree.cc:517 reduce_type_tree

    Collapses the two public overloads: `reduce_type_tree(tt, arg_types,
    tr, proc_name)` (cc:517) and `reduce_type_tree(tt, tr, proc_name)`
    (cc:529), which just supplies an empty `arg_types`. Reduces `tt` IN
    PLACE, as classic does.

    The five reductions are listed at type_tree.h:180-192: collapse
    application/abstraction chains, `lambda(X) -> X`, `application(X) -> X`,
    `union(X Y) -> union(X)` when Y inherits X, and replace a variable by
    its type from `arg_types`.

    An empty `tt` is a no-op plus a `logger().warn()` (cc:525), and the
    warn is dropped as everywhere in this port.
    """
    if arg_types is None:
        arg_types = []
    if tr is None:
        tr = combo_tree()
    if not tt.empty():
        _reduce_type_tree_at(tt, tt.begin(), arg_types, tr, tr.begin(), proc_name)


# DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:637): every
# `stringstream message; ... logger().error(message.str().c_str());` block
# in `_reduce_type_tree_at` is dropped, along with the `ct_it_child`
# walking (cc:673-674, cc:825-826) that exists ONLY to name the offending
# operand in those messages. This port drops `logger()` calls
# package-wide, so nothing reads them. `tr`/`ct_it`/`proc_name` are kept in
# the signature so call sites still read like classic's; the `if
# (!tr.empty())` guards they gate change nothing else.
def _reduce_type_tree_at(
    tt: type_tree,
    it: tree_node,
    arg_types: Sequence[type_tree],
    tr: combo_tree,
    ct_it: tree_node | None,
    proc_name: str,
) -> None:
    """C++: moses/comboreduct/type_checker/type_tree.cc:537 reduce_type_tree

    The subtree-rooted overload, and the whole of the reduction. Private
    here because its C++ name is a third overload of the public one above,
    which Python cannot express; `reduce_type_tree` dispatches into it.
    """
    # ---------------------
    #    recursive cases
    # ---------------------

    # -----------
    # lambda case
    # -----------
    if it.data == type_node.lambda_type:
        if it.first_child is None:
            raise AssertionError("lambda_type must have at least a child")
        # if lambda has only one child then reduce lambda(X) -> X
        if tt.has_one_child(it):
            it_child = it.first_child
            it.data = it_child.data
            tt.erase(tt.flatten(it_child))
            _reduce_type_tree_at(tt, it, arg_types, tr, ct_it, proc_name)
        else:
            for sib in tt.children(it):
                _reduce_type_tree_at(tt, sib, arg_types, tr, ct_it, proc_name)
    # ----------------
    # application case
    # ----------------
    elif it.data == type_node.application_type:
        if it.first_child is None:
            raise AssertionError("application_type must have at least a child")

        it_child = it.first_child

        # If application has only one child, then a trivial reduction
        # application(X) -> X
        if tt.has_one_child(it):
            it.data = it_child.data
            tt.erase(tt.flatten(it_child))
            _reduce_type_tree_at(tt, it, arg_types, tr, ct_it, proc_name)
        elif it_child.data == type_node.lambda_type:
            # Check that the output type of all siblings of it_child inherit
            # from the children of it_child -- all but the last, which is the
            # output. If the next-to-last child of it_child is arg_list, the
            # earlier siblings inherit from the earlier children and all the
            # rest from the children of arg_list.
            if it_child.first_child is None:
                raise AssertionError("it_child must have at least one child")
            # cia stands for current input argument
            cia_it = it_child.first_child

            # number of input arguments the function takes
            arg_count = tt.number_of_children(it_child) - 1

            # number of arguments applied to that function
            arg_count_app = tt.number_of_siblings(it_child)

            # check whether the last input argument, or the second-from-last
            # input argument, under the lambda, is arg_list
            last_arg_sib = tt.previous_sibling(it_child.last_child)
            second_last_arg_sib = (
                tt.previous_sibling(last_arg_sib) if last_arg_sib is not None else None
            )

            # ila is true if the last input argument under lambda is arg_list
            ila = tt.is_valid(last_arg_sib) and last_arg_sib.data == type_node.arg_list_type

            # ils is true if the second-from-last input argument under lambda
            # is arg_list. Need this for the cond signature.
            ils = (
                tt.is_valid(second_last_arg_sib)
                and second_last_arg_sib.data == type_node.arg_list_type
            )

            # Check if the number of applied arguments are incorrect
            if ((ila or ils) and arg_count_app < arg_count - 1) or (
                not ila and not ils and arg_count_app != arg_count
            ):
                it.data = type_node.ill_formed_type
                tt.erase_children(it)
                return
            # The number of applied arguments is correct

            # Will be set to true if at some point arg_list is reached
            is_arg_list_reached = False
            arg_list_head: tree_node | None = None
            input_arg_it: tree_node | None = cia_it

            # Iterate over the applied arguments.
            arg_app = tt.next_sibling(it_child)
            while arg_app is not None:
                # input_arg_it is either cia_it, or if it is arg_list, then it
                # will cycle over the arg_list contents.
                if not is_arg_list_reached:
                    input_arg_it = cia_it

                # Advance the arg list counter. In almost all cases the
                # arglist has just one child total, so the below loops back to
                # that child -- but the cond statement has a pair of
                # alternating types to cycle through.
                if is_arg_list_reached:
                    input_arg_it = input_arg_it.next_sibling
                    # If at the end of the list, start again at the beginning.
                    if input_arg_it is None:
                        input_arg_it = arg_list_head.first_child

                # Check if cia_it is arg_list(T)
                elif cia_it.data == type_node.arg_list_type:
                    is_arg_list_reached = True
                    arg_list_head = cia_it
                    input_arg_it = arg_list_head.first_child

                # Reduce both input and applied arguments
                _reduce_type_tree_at(tt, arg_app, arg_types, tr, ct_it, proc_name)
                _reduce_type_tree_at(tt, cia_it, arg_types, tr, ct_it, proc_name)

                arg_inherits = inherit_type_tree(tt, arg_app, tt, input_arg_it)
                # Check for a funky special case, occurring for the cond
                # operator: the arg_list is empty, but the input might still
                # match the last argument (the "else" clause, which comes
                # after the arg_list but before the output type).
                if not arg_inherits and is_arg_list_reached and ils:
                    arg_inherits = inherit_type_tree(tt, arg_app, tt, last_arg_sib)
                    # And if it matches, then we've moved past the end of the
                    # arg_list.
                    if arg_inherits:
                        is_arg_list_reached = False

                if not arg_inherits:
                    # Check if the output argument of arg_app inherits from
                    # cia_it (or cia_it's child if cia_it is arg_list)
                    if arg_app.data == type_node.lambda_type:
                        if arg_app.first_child is None:
                            raise AssertionError("lambda must have at least one child")
                        output_it = arg_app.last_child
                        if inherit_type_tree(tt, output_it, tt, input_arg_it):
                            # insert all inputs before cia_it
                            input_sib = arg_app.first_child
                            while input_sib is not arg_app.last_child:
                                tt.insert_subtree(cia_it, input_sib)
                                input_sib = input_sib.next_sibling
                        else:
                            # If it does not inherit, then the type checker has
                            # found an error, and the resulting type is
                            # ill_formed
                            it.data = type_node.ill_formed_type
                            tt.erase_children(it)
                            return
                    else:
                        # If it's not a lambda then it is ill formed, because
                        # arg_app does not inherit input_arg_it
                        it.data = type_node.ill_formed_type
                        tt.erase_children(it)
                        return

                # if cia_it is not an arg_list then erase it and its children,
                # because this input has been substituted by the inputs of the
                # application (if there are some), and cia_it will now point to
                # the next input argument
                if not is_arg_list_reached:
                    cia_it = tt.erase(cia_it)

                arg_app = arg_app.next_sibling

            # Remove a possibly remaining arg_list(T) in the input arguments
            if is_arg_list_reached or ila:
                tt.erase(cia_it)

            # Remove application_type and its argument now that the
            # substitution has been done.
            sib = tt.next_sibling(it_child)
            while sib is not None:
                sib = tt.erase(sib)
            it.data = it_child.data
            tt.erase(tt.flatten(it_child))

            # Check if it is in the form lambda(X) and remove lambda if so
            if tt.has_one_child(it):
                inner_child = it.first_child
                it.data = inner_child.data
                tt.erase(tt.flatten(inner_child))
                _reduce_type_tree_at(tt, it, arg_types, tr, ct_it, proc_name)
        else:
            # the first of the application is not lambda, so there is a type
            # checker error because the so-called operator is not a function
            it.data = type_node.ill_formed_type
            tt.erase_children(it)
    # ----------
    # union case
    # ----------
    elif it.data == type_node.union_type:
        if it.first_child is None:
            raise AssertionError("union_type must have at least a child")
        # apply reduce recursively to the children of the union
        for sib in tt.children(it):
            _reduce_type_tree_at(tt, sib, arg_types, tr, ct_it, proc_name)
        # remove sib if it inherits from other siblings
        sib = it.first_child
        while sib is not None:
            found_inherit = False
            sib2 = it.first_child
            while sib2 is not None and not found_inherit:
                found_inherit = sib is not sib2 and inherit_type_tree(tt, sib, tt, sib2)
                sib2 = sib2.next_sibling
            if found_inherit:
                sib = tt.erase(sib)
            else:
                sib = sib.next_sibling
        # if union has only one child then reduce union(X) -> X
        if tt.has_one_child(it):
            it_child = it.first_child
            it.data = it_child.data
            tt.erase(tt.flatten(it_child))
    # -------------
    # arg_list case
    # -------------
    elif it.data == type_node.arg_list_type:
        # Usually an arg_list will have only one type in it. However, the cond
        # statement can have a repeated list of two alternating types.
        #
        # NOT a subtree walk, despite reading like one. `tit` is a
        # `type_tree_pre_it` starting at `it.begin()` and the loop runs while
        # `tit != it.end()`, which compares only the NODE pointer -- and
        # `it.end()`'s node is null. A pre-order `++` walks up and out of the
        # subtree, so this reduces every node from the arg_list's first child
        # to the end of the WHOLE tree, not just the arg_list's own children.
        # Transliterated as written; `tt.iter_pre_at(it)` would be the
        # subtree-restricted walk and is the wrong one.
        # The successor is taken AFTER the body, exactly as the `++tit` of a
        # C++ for-loop is: reducing a node can erase that node's CHILDREN
        # (and its later siblings), so a successor computed before the body
        # can be one of the nodes the body then erases. Taking it after means
        # the step reads the links the reduction just rewrote, which is what
        # classic does and what the fixture's `+(6 *(6 $1))` row depends on.
        tit = it.first_child
        while tit is not None:
            _reduce_type_tree_at(tt, tit, arg_types, tr, ct_it, proc_name)
            tit = tt.next_pre_order(tit)

    # ----------------
    #    base case
    # ----------------

    # at this stage an argument is considered as constant, so it is simply
    # replaced by its type. The binding between considering argument as input
    # variable in the signature of the function is made at the end in
    # infer_type_tree.
    elif is_argument_type(it.data):
        tmp = get_arg_type(it.data, arg_types)
        tmp_head = tmp.begin()
        it.data = tmp_head.data
        # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:934):
        # classic writes `tt.reparent(it, tmp_head)`, MOVING `tmp`'s children
        # into `tt`. `tmp` is a local copy (`get_arg_type` returns a const
        # reference and that line copies it) destroyed on the next statement,
        # so moving out of it is safe there but leaves two trees sharing
        # nodes for an instant. This appends deep copies instead, which
        # reaches the same shape without cross-tree relinking; the copies are
        # of a copy, so nothing else can observe the difference.
        for child in list(tmp.children(tmp_head)):
            tt.append_child(it, child)


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


def _lexicographic_subtree_cmp(it1: tree_node, it2: tree_node) -> int:
    """C++: opencog/util/tree.h:3257 lexicographic_subtree_order

    The real symbol is that struct's `cmp` member template, whose name the
    breadcrumb matcher cannot reach through the enclosing template; the
    crumb cites the struct.

    Returns 1 when `it1`'s value sorts first, -1 when `it2`'s does, and
    recurses over the children -- with the C++ quirk that whichever node
    runs out of children first is the one reported as sorting FIRST.
    """
    if it1.data < it2.data:
        return 1
    if it2.data < it1.data:
        return -1
    sib1 = it1.first_child
    sib2 = it2.first_child
    while True:
        if sib1 is None:
            return 0 if sib2 is None else 1
        if sib2 is None:
            return -1
        res = _lexicographic_subtree_cmp(sib1, sib2)
        sib1 = sib1.next_sibling
        sib2 = sib2.next_sibling
        if res:
            return res


def _size_tree_order_less(t1: type_tree, t2: type_tree) -> bool:
    """C++: opencog/util/tree.h:3306 size_tree_order

    The real symbol is that struct's `operator()(const tree<T>&, const
    tree<T>&)`, which the breadcrumb matcher cannot name; the crumb cites
    the struct. "if size(tr1)!=size(tr2) then tr1 < tr2, otherwise
    lexicographic_subtree_order" -- a smaller tree sorts first, and
    same-size trees sort by the comparator above.

    This is the comparator of the `std::set<type_tree,
    size_tree_order<type_node>>` in `get_intersection`'s union-of-unions
    branch (type_tree.cc:961), so it decides both which intersections
    count as duplicates and the ORDER the resulting union's children are
    written in.
    """
    s1 = t1.size()
    s2 = t2.size()
    if s1 == s2:
        return _lexicographic_subtree_cmp(t1.begin(), t2.begin()) > 0
    return s1 < s2


def _ordered_set_insert(items: list[type_tree], value: type_tree) -> None:
    """C++: opencog/util/tree.h:3306 size_tree_order

    `std::set<type_tree, size_tree_order<type_node>>::insert` over a
    sorted list: an element already EQUIVALENT under the comparator
    (`!(a<b) && !(b<a)`) is dropped rather than added, and the list stays
    in comparator order, which is the order `get_intersection` then writes
    the union's children in.
    """
    lo = 0
    hi = len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if _size_tree_order_less(items[mid], value):
            lo = mid + 1
        else:
            hi = mid
    if lo < len(items) and not _size_tree_order_less(value, items[lo]):
        return
    items.insert(lo, value)


def get_intersection(
    tt1: type_tree,
    it1: type_tree | tree_node,
    tt2: type_tree | None = None,
    it2: tree_node | None = None,
) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:941 get_intersection

    Both overloads: the whole-tree form (cc:941) and the four-argument
    subtree-rooted form (cc:951). "the highest type (in the inheritance
    hierarchy) that inherits both T1 and T2" (type_tree.h:196), with
    `intersection(unknown, T) == T`, `intersection(ill_formed, T) ==
    ill_formed`, and no detected intersection also giving `ill_formed`.

    This is where a CROSS-ARGUMENT type mismatch becomes `ill_formed`:
    `infer_arg_type_tree` (cc:1135) intersects the type an argument
    already has with the type its next occurrence demands, so a `$1` used
    once as a boolean and once as a contin lands on `ill_formed` here,
    never reaching `reduce_type_tree` at all.
    """
    if tt2 is None:
        if tt1.empty() or it1.empty():
            raise AssertionError("neither tt1 nor tt2 must be empty")
        return get_intersection(tt1, tt1.begin(), it1, it1.begin())

    # check whether one inherits from the other
    if inherit_type_tree(tt1, it1, tt2, it2):
        return type_tree(it1)
    if inherit_type_tree(tt2, it2, tt1, it1):
        return type_tree(it2)
    # check if tt1 and tt2 are unions, if so the intersection of the unions is
    # the union of the intersections
    if it1.data == type_node.union_type and it2.data == type_node.union_type:
        union_of_inter: list[type_tree] = []
        for sib1 in tt1.children(it1):
            for sib2 in tt2.children(it2):
                inter_tt = get_intersection(tt1, sib1, tt2, sib2)
                if is_well_formed(inter_tt):
                    _ordered_set_insert(union_of_inter, inter_tt)
        if not union_of_inter:
            return type_tree(type_node.ill_formed_type)
        if len(union_of_inter) == 1:
            return type_tree(union_of_inter[0])
        tmp = type_tree()
        head = tmp.set_head(type_node.union_type)
        for its in union_of_inter:
            tmp.append_child(head, its.begin())
        return tmp
    # if both are lambda, then calculate the intersection of their outputs and
    # the union of their inputs
    if it1.data == type_node.lambda_type and it2.data == type_node.lambda_type:
        n1 = tt1.number_of_children(it1)
        n2 = tt2.number_of_children(it2)
        if n1 <= 0 or n2 <= 0:
            raise AssertionError("lambda must not be childless")
        if n1 != n2:
            return type_tree(type_node.ill_formed_type)
        tmp = type_tree(type_node.lambda_type)
        tmp_head = tmp.begin()
        sib1 = it1.first_child
        sib2 = it2.first_child
        # perform union of arguments
        while sib1 is not it1.last_child:
            union_it = tmp.append_child(tmp_head, type_node.union_type)
            tmp.append_child(union_it, sib1)
            tmp.append_child(union_it, sib2)
            sib1 = sib1.next_sibling
            sib2 = sib2.next_sibling
        output_tt = get_intersection(tt1, sib1, tt2, sib2)
        tmp.append_child(tmp_head, output_tt.begin())
        reduce_type_tree(tmp)
        return tmp
    # check if one is unknown_type
    if it1.data == type_node.unknown_type:
        return type_tree(it2)
    if it2.data == type_node.unknown_type:
        return type_tree(it1)
    # check if one is ill_formed_type
    if it1.data == type_node.ill_formed_type or it2.data == type_node.ill_formed_type:
        return type_tree(type_node.ill_formed_type)
    # otherwise there is no detected intersection and it returns ill_formed
    return type_tree(type_node.ill_formed_type)


# ---------------------------------------------------------------------------
# Per-argument type vectors
# ---------------------------------------------------------------------------


def set_arg_type(
    tt: type_tree, arg: argument | type_node | int, arg_types: list[type_tree]
) -> None:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1031 set_arg_type

    All three overloads: by `argument` (cc:1031), by an argument
    `type_node` (cc:1038) and by a 1-based index (cc:1045). Grows
    `arg_types` with `unknown_type` entries up to `idx` when needed, then
    writes slot `idx - 1`.
    """
    if isinstance(arg, argument):
        set_arg_type(tt, arg.abs_idx(), arg_types)
        return
    if isinstance(arg, type_node):
        set_arg_type(tt, arg_to_idx(arg), arg_types)
        return
    idx = arg
    if idx <= 0:
        raise AssertionError("set_arg_type: idx must be positive")
    idxfz = idx - 1
    s = len(arg_types)
    if s < idx:
        for _ in range(s, idx):
            arg_types.append(type_tree(type_node.unknown_type))
    arg_types[idxfz] = tt


_UNKNOWN_TYPE_TREE = type_tree(type_node.unknown_type)


def get_arg_type(arg: argument | type_node | int, arg_types: Sequence[type_tree]) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1059 get_arg_type

    All three overloads: by `argument` (cc:1059), by an argument
    `type_node` (cc:1066) and by a 1-based index (cc:1073). An index past
    the end of `arg_types` answers with `unknown_type` -- classic returns
    a reference to a function-local `static const type_tree`, which is the
    module-level `_UNKNOWN_TYPE_TREE` here.

    All three return by CONST REFERENCE in classic, and every caller
    either copies the result (`type_tree tmp = get_arg_type(...)`, cc:931)
    or passes it straight into `get_intersection`, which does not mutate
    it. This returns the stored object for the same reason, so
    `_UNKNOWN_TYPE_TREE` must never be mutated by a caller.
    """
    if isinstance(arg, argument):
        return get_arg_type(arg.abs_idx(), arg_types)
    if isinstance(arg, type_node):
        return get_arg_type(arg_to_idx(arg), arg_types)
    idx = arg
    if idx <= len(arg_types):
        return arg_types[idx - 1]
    return _UNKNOWN_TYPE_TREE


def infer_vertex_type(tr: combo_tree, it: tree_node, atl: Sequence[type_tree] = ()) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1082 infer_vertex_type

    "infer type of a vertex based on its contextual use" (type_tree.h:216)
    -- the OUTPUT type comes from what the PARENT expects in this vertex's
    sibling position, and the INPUT types (if the vertex has children)
    from each child's own output type, wrapped in a lambda.

    Note cc:1090's arity guard: a vertex sitting at a sibling index at or
    past a POSITIVE parent arity is `ill_formed` immediately. That is the
    arity half of type checking, and it is why `not($1 $2)`'s second
    argument, `/($1 $2 $3)`'s third and `0<($1 $2)`'s second all come back
    ill formed.
    """
    it_parent = it.parent
    res = type_tree(type_node.unknown_type)
    # set output type
    if tr.is_valid(it_parent):
        si = tr.sibling_index(it)
        a = get_arity(it_parent.data)
        if a < 0 or (a > 0 and si < a):
            res = get_input_type_tree(it_parent.data, tr.sibling_index(it))
        else:
            return type_tree(type_node.ill_formed_type)
    # set input types
    if it.first_child is not None:
        # wrap lambda over res
        head = res.wrap(res.begin(), type_node.lambda_type)
        for sib in tr.children(it):
            # if argument then uses atl to get the input type
            if is_argument(sib.data):
                child_type = get_arg_type(sib.data, atl)
            else:
                child_type = get_output_type_tree(sib.data)
            res.insert_subtree(head.last_child, child_type.begin())
    return res


def get_argument_abs_idx_set(tr: combo_tree) -> set[int]:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1115
    get_argument_abs_idx_set

    # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:1115):
    # classic returns `arity_set`, a `std::set<arity_t>` (common_def.h),
    # which is ORDERED; this returns a Python `set`, which is not. Its one
    # classic call site, `table.cc:353`, only tests membership, so the
    # order is unobservable there -- and this port has no `arity_set`
    # typedef at all (common_def.py holds only `arity_t` and `contin_t`).
    """
    res: set[int] = set()
    for node in tr.iter_leaf():
        if is_argument(node.data):
            res.add(node.data.abs_idx())
    return res


def get_argument_abs_idx_from_zero_set(tr: combo_tree) -> set[int]:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1124
    get_argument_abs_idx_from_zero_set

    As above but zero-based. `pymoses/moses/deme/feature_selector.py:66`
    holds an earlier, identical copy of this function, put there because
    "editing pymoses/comboreduct/ is out of bounds for this task" (its own
    docstring). This is its real home; that file is not this task's to
    edit either, so the duplicate is reported rather than removed.
    """
    res: set[int] = set()
    for node in tr.iter_leaf():
        if is_argument(node.data):
            res.add(node.data.abs_idx_from_zero())
    return res


def infer_arg_type_tree(tr: combo_tree, arg_types: list[type_tree]) -> None:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1135 infer_arg_type_tree

    Walks the LEAVES left to right and, for each argument, intersects the
    type that argument already has with the type its enclosing expression
    demands here. Fills `arg_types` in place.

    This is the whole of cross-argument type checking: two incompatible
    uses of the same `$n` meet in `get_intersection` and produce
    `ill_formed` for that slot -- `+(impulse($1) $1)` is
    `->(ill_formed ill_formed)` and `*(impulse($2) $2)` is
    `->(unknown ill_formed ill_formed)`, both measured.
    """
    if tr.empty():
        raise AssertionError("cannot infer arg types on an empty combo_tree")
    for lit in tr.iter_leaf():
        if is_argument(lit.data):
            a = lit.data
            att = get_arg_type(a, arg_types)
            itt = get_intersection(att, infer_vertex_type(tr, lit, arg_types))
            set_arg_type(itt, a, arg_types)


def insert_arg_type_tree(arg_types: Sequence[type_tree], tt2: type_tree) -> None:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1152 insert_arg_type_tree

    Turns a reduced OUTPUT type into a full signature by prepending the
    inferred argument types, wrapping in `lambda_type` first if it is not
    already one. An empty `arg_types` leaves `tt2` alone -- which is why a
    constant program such as `true` infers as bare `boolean`, not
    `->(boolean)`.
    """
    if tt2.empty():
        raise AssertionError("tt2 is supposed to contain a type")
    if arg_types:
        head = tt2.begin()
        if head.data != type_node.lambda_type:
            head = tt2.wrap(head, type_node.lambda_type)
        first_var_arg = head.first_child
        for itt in arg_types:
            tt2.insert_subtree(first_var_arg, itt.begin())


def infer_type_tree(tr: combo_tree) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1200 infer_type_tree

    The whole engine, in four lines: build the unreduced
    `application(...)` type of the tree, infer each argument's type,
    reduce, then splice the argument types back in as the signature's
    inputs.

    "If a type error is detected then the type_tree will contain
    ill_formed_type" (type_tree.h:274) -- which is what `is_well_formed`
    then looks for.
    """
    tt = get_type_tree(tr)
    arg_types: list[type_tree] = []
    infer_arg_type_tree(tr, arg_types)
    reduce_type_tree(tt, arg_types, tr)
    insert_arg_type_tree(arg_types, tt)
    return tt


def is_well_formed(tt: type_tree) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1210 is_well_formed

    An EMPTY type tree is not well formed, and neither is one containing
    `ill_formed_type` anywhere -- input position or output position.
    """
    if tt.empty():
        return False
    for it in tt:
        if it.data == type_node.ill_formed_type:
            return False
    return True


def does_contain_all_arg_up_to(tr: combo_tree, n: arity_t) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1220
    does_contain_all_arg_up_to

    True iff every index `$1..$n` appears at least once AND no index above
    `n` appears. `n == 0` is trivially true whatever the tree contains --
    the `if (n > 0)` guard skips the scan entirely.
    """
    if tr.empty():
        raise AssertionError("cannot infer arg types on an empty combo_tree")
    if n < 0:
        raise AssertionError("Must be positive or null")
    bv = [False] * n
    if n > 0:
        for lit in tr.iter_leaf():
            if is_argument(lit.data):
                idx = lit.data.abs_idx_from_zero()
                if idx < n:
                    bv[idx] = True
                else:
                    return False
        return bv == [True] * n
    return True


def infer_arity(tr: combo_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1242 infer_arity

    "the sum of the arities of all the leaves" (type_tree.h:381), with an
    early return of `va - a` on the first NEGATIVE leaf arity -- so a
    childless variadic operator anywhere in the leaves makes the whole
    answer negative.
    """
    a: arity_t = 0
    for l_it in tr.iter_leaf():
        va = get_arity(l_it.data)
        if va > 0:
            a += va
        elif va < 0:
            return va - a
    return a


def explicit_arity(tr: combo_tree) -> arity_t:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1260 explicit_arity

    "the highest argument index of combo tree tr; for instance if tr ==
    +($1 $5) it returns 5" (type_tree.h:385). Note it does NOT require the
    indices in between to appear.

    `abs_idx()` (type_tree.cc:1267), NOT `idx`. A NEGATED argument is
    stored with a negative index -- `!$1` is `argument(-1)` -- so reading
    `idx` makes `max()` ignore it entirely. Measured before the fix:
    `!$1` gave 0 where classic gives 1, `or($1 !$3)` gave 1 where classic
    gives 3, and `or(and($1 !$2) and($3 $4) !$5)` gave 4 where classic
    gives 5. See docs/CARRY-FORWARD.md 4.4 -- `argument`'s ordering is by
    absolute value, and this is the same trap in a different function.
    """
    res: arity_t = 0
    for l_it in tr.iter_leaf():
        if is_argument(l_it.data):
            res = max(res, l_it.data.abs_idx())
    return res


# ---------------------------------------------------------------------------
# Signature accessors and constructors
# ---------------------------------------------------------------------------


def get_signature_output(ty: type_tree) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1272 get_signature_output

    The last child of a `lambda_type` root, or the whole tree when the
    root is not a lambda (a constant program's type IS its output type).
    """
    if ty.empty():
        raise AssertionError("ty must not be empty")
    ty_it = ty.begin()
    if ty_it.data == type_node.lambda_type:
        return type_tree(ty_it.last_child)
    return type_tree(ty_it)


def get_signature_inputs(ty: type_tree) -> list[type_tree]:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1282 get_signature_inputs

    Every child of a `lambda_type` root except the last, with an
    `arg_list(T)` input unwrapped to `T`. A non-lambda type has no inputs
    at all -- an empty list, not a one-element one.
    """
    if ty.empty():
        raise AssertionError("ty must not be empty")
    ty_it = ty.begin()

    tts: list[type_tree] = []
    if ty_it.data != type_node.lambda_type:
        return tts

    if ty_it.first_child is None:
        raise AssertionError("Lambda must not be childless")
    sib = ty_it.first_child
    while sib is not ty_it.last_child:
        tts.append(
            type_tree(sib.first_child) if sib.data == type_node.arg_list_type else type_tree(sib)
        )
        sib = sib.next_sibling
    return tts


def _replace_with(res: type_tree, sib: tree_node, value: type_node | type_tree) -> tree_node:
    """C++: opencog/util/tree.h:1388 replace

    `gen_signature`'s six overloads differ only in whether the thing
    written into a slot is a plain `type_node` (`res.replace(sib, *vi)`,
    tree.h:1388) or a whole subtree (`res.replace(sib, (*vi).begin())`,
    tree.h:1397). Both return the node that now occupies the slot, which
    the callers advance from.
    """
    if isinstance(value, type_tree):
        return res.replace(sib, value.begin())
    return res.replace(sib, value)


def gen_signature(
    inputs: Sequence[type_node] | Sequence[type_tree] | type_node | type_tree,
    otype: type_node | type_tree | arity_t | None = None,
    arity: arity_t | None = None,
) -> type_tree:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1299 gen_signature

    All six overloads (cc:1299, :1312, :1325, :1338, :1344, :1349), which
    differ only in whether they take `type_node`s or `type_tree`s and
    whether the inputs are a sequence, a repeated type plus an arity, or a
    single type used for both inputs and output:

    * `gen_signature(inputs_seq, otype)`
    * `gen_signature(itype, otype, arity)`
    * `gen_signature(iotype, arity)`

    `gen_signature([contin, contin], boolean)` gives
    `->(contin contin boolean)`.

    # DIVERGENCE(C++ moses/comboreduct/type_checker/type_tree.cc:1330): a
    # NEGATIVE arity segfaults classic. `append_children(root, arity + 1)`
    # appends nothing for `arity + 1 <= 0` (tree.h:1072's `while ((n--) >
    # 0)`), the following loop does nothing on a childless root, and then
    # `res.replace(sib, otype.begin())` runs UNCONDITIONALLY with `sib`
    # still holding `begin()`'s null-wrapping iterator, which `replace`
    # dereferences with no null guard. There is no graceful C++ behaviour
    # to reproduce, only a crash to avoid, so this raises instead. The
    # same reasoning and the same trace are already recorded on
    # `type_signature.gen_signature`.
    """
    res = type_tree(type_node.lambda_type)
    root = res.begin()

    if arity is None and isinstance(otype, int) and not isinstance(otype, type_node):
        # gen_signature(iotype, arity) -- cc:1344 / :1349
        arity = otype
        otype = inputs

    if arity is not None:
        # gen_signature(itype, otype, arity) -- cc:1325 / :1338
        if arity < 0:
            raise ValueError("signature arity must be non-negative")
        res.append_children(root, [type_node.lambda_type] * (arity + 1))
        sib = root.first_child
        while sib is not root.last_child:
            sib = _replace_with(res, sib, inputs)
            sib = sib.next_sibling
        _replace_with(res, sib, otype)
        return res

    # gen_signature(inputs_seq, otype) -- cc:1299 / :1312
    res.append_children(root, [type_node.lambda_type] * (len(inputs) + 1))
    sib = root.first_child
    for vi in inputs:
        sib = _replace_with(res, sib, vi)
        sib = sib.next_sibling
    _replace_with(res, sib, otype)
    return res
