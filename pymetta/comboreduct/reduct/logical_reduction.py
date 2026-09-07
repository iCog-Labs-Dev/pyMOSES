"""C++: moses/comboreduct/reduct/logical_reduction.cc

`logical_reduction` assembles the boolean-logic rule set at each of four
effort levels (0 "extra_simple" .. 3 "complexe") -- this is what
`problem-params.cc:2385-2389` calls to build `bool_reduct`/`bool_reduct_
rep`, and Task 6's own exit criterion (byte-identical output at efforts
1, 2 AND 3 against real `logical_reduce`). COMPOSITION ORDER IS THE
NORMAL FORM: every effort level here is a fixed sequence of already-ported
rules and combinators (`meta_rules.py`, `general_rules.py`,
`logical_rules.py`) -- getting the order, nesting, or typed/untyped
dispatch of any one of them wrong changes the fixpoint the whole
composition converges to, silently, with no single-rule test able to
catch it (the same reasoning `meta_rules.py`'s own module docstring gives
for why VISIT ORDER, not just end results, is under test).

10 of the 12 `downwards()` calls this file makes pass `id::boolean_type`
-- this is why Task 5 scoped in a slice of `type_checker/type_tree`
(`pymetta.comboreduct.type_checker.type_tree`) rather than leaving
`meta_rules.py`'s typed `downwards` branch stubbed. See that module's own
docstring for exactly what is and is not ported.

MEASURED, not assumed: typed `downwards` CHANGES results relative to an
all-untyped approximation, starting at EFFORT 2, not only effort 3 (a
correction to an earlier, wrong claim -- see `git log` / task-4-report.md
for the history). Verified directly against the real compiled
`libcomboreduct.so` via `tools/cpp_probe/logical_rules_probe.cc`'s
`boolean_reduct_medium_typed_N`/`boolean_reduct_effort3_typed_N` cases
(`( ulimit -v 2097152; timeout 300 tools/cpp_probe/build/
logical_rules_probe )`): of `BooleanReductUTest`'s 6 real trees, tree 2
diverges from an untyped approximation at BOTH effort 2 and effort 3 (the
same underlying mechanism -- an `argument`'s `get_output_type_tree` is
ALWAYS `unknown_type` in isolation, so typed `downwards(insert_ands(),
boolean_type)` skips wrapping a bare argument an untyped `downwards
(insert_ands())` would have wrapped), and tree 6 additionally diverges at
effort 3 only (introduced by `reduce_remove_subtree_equal_tt`'s own
repeated re-entry into "medium" via the outer `iterative`, not by
anything in this file). `test_logical_reduction.py` pins THIS port's real
typed output for all 6 trees at both efforts directly against those
measured C++ values -- not an untyped approximation.

A hardcoded-but-real upstream ODDITY, ported faithfully (`logical_
reduction.cc:115-116`'s own comment: "Arghh .. XXX should use reduct_
effort==3 for the complexe rule"): `p_medium`'s inner `simplify_
predicates` is always constructed with `reduct_effort = 2`, EVEN when
`p_medium` is being reused (verbatim, via `sequential`'s cloning
constructor) to build `p_complexe` (effort 3). Inert for every tree in
this project's actual reachable scope -- `simplify_predicates`'s own
early-return guard means `reduct_effort` is only ever consulted past a
`greater_than_zero` predicate, which never appears in a pure boolean
tree (see `logical_rules.py`'s own module docstring) -- but a real,
disclosed upstream fact, not silently "fixed" to `effort` here.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import combo_tree, vertex
from pymetta.comboreduct.reduct.general_rules import level
from pymetta.comboreduct.reduct.logical_rules import (
    eval_logical_identities,
    insert_ands,
    reduce_ands,
    reduce_nots,
    reduce_ors,
    reduce_remove_subtree_equal_tt,
    remove_unary_junctors,
    simplify_predicates,
    subtree_to_enf,
)
from pymetta.comboreduct.reduct.meta_rules import downwards, iterative, sequential, upwards
from pymetta.comboreduct.reduct.reduct import _NO_IT, rule
from pymetta.comboreduct.table.table import type_node
from pymetta.util.tree import tree_node


class logical_reduction:
    """C++: moses/comboreduct/reduct/reduct.h:76 logical_reduction

    `@ignore_ops` (the constructor argument, C++'s own `vertex_set&`
    parameter comment, reduct.h:72-75) is the set of operators to ignore
    -- `logical_reduction` doesn't use it itself, but does pass it on to
    `simplify_predicates` for the `contin_reduction` step of any
    predicates encountered.

    `p_extra_simple`/`p_simple` are C++ CLASS-level `static rule*`
    members (reduct.h:90-91), shared across every `logical_reduction`
    instance and built exactly once, lazily, by `_do_init` (below) --
    ported as Python class attributes for the same "build once, share
    everywhere" semantics. `p_medium`/`p_complexe` are C++ per-INSTANCE
    `const rule*` members (reduct.h:87-88, private) -- ordinary instance
    attributes here, `None` unless the `ignore_ops` constructor argument
    was given (see `__init__`'s own docstring).
    """

    p_extra_simple: rule | None = None
    p_simple: rule | None = None

    @classmethod
    def _do_init(cls) -> None:
        """C++: moses/comboreduct/reduct/logical_reduction.cc:36 logical_reduction::do_init

        C++'s own comment explains a double-checked-locking guard here
        (`if (p_extra_simple != NULL) return;` plus a `std::mutex`) --
        protecting against two threads racing to construct the same
        static rule the first time this runs. Nothing in this port is
        multi-threaded (module-level Python execution, and every call
        into this classmethod, is single-threaded), so only the
        OBSERVABLE half survives: build `p_extra_simple`/`p_simple`
        exactly once, on first use, and every later call is a cheap
        early-return -- matching the guard's own effect, not its
        mechanism (see `clean_reduction.py`'s `clean_reduction()` for the
        same pattern, spelled out in more detail there).
        """
        if cls.p_extra_simple is not None:
            return
        cls.p_extra_simple = downwards(
            reduce_nots(), type_node.boolean_type, name="extra_simple"
        )
        cls.p_simple = sequential(
            downwards(reduce_nots(), type_node.boolean_type),
            iterative(
                sequential(
                    upwards(eval_logical_identities()),
                    downwards(level()),
                    downwards(reduce_ands(), type_node.boolean_type),
                    downwards(reduce_ors(), type_node.boolean_type),
                )
            ),
            name="simple",
        )

    def __init__(self, ignore_ops: set[vertex] | None = None) -> None:
        """C++: moses/comboreduct/reduct/reduct.h:78 logical_reduction

        Two C++ constructors collapsed into one Python signature (this
        port's usual convention for an overloaded C++ constructor): the
        default constructor (reduct.h:78, `logical_reduction()` -- no
        `ignore_ops`, `p_medium`/`p_complexe` left unbuilt, only efforts 0
        and 1 usable) and the `vertex_set` constructor (reduct.h:80,
        `logical_reduction(const vertex_set& ignore_ops)` -- ALWAYS
        builds `p_medium`/`p_complexe`, even for an EMPTY `ignore_ops`).
        `ignore_ops is None` (not "falsy", e.g. an empty set) is the
        discriminator, so `logical_reduction(set())` correctly takes the
        vertex_set-constructor branch with an empty set, matching C++'s
        own `vertex_set ignore_ops;` (default-constructed, empty) callers
        (`BooleanReductUTest`'s own `vertex_set ignore_ops;`).

        The copy constructor/assignment operator (reduct.h:79/81,
        `p_medium(rhs.p_medium->clone())`) have no separate Python
        equivalent -- nothing in this file's own reachable scope ever
        copies a `logical_reduction` instance (grep confirms zero such
        call sites in `logical_reduction.cc`/`reduct.h` itself; the only
        real construction pattern anywhere is `logical_reduction
        r(ignore_ops);`, a fresh instance per use, e.g. `logical_reduce`
        below).
        """
        type(self)._do_init()
        self.p_medium: rule | None = None
        self.p_complexe: rule | None = None
        if ignore_ops is None:
            return

        # medium
        pre_subtree_to_enf = sequential(
            upwards(eval_logical_identities()),
            downwards(level()),
            downwards(insert_ands(), type_node.boolean_type),
        )
        post_subtree_to_enf = sequential(
            downwards(reduce_ands(), type_node.boolean_type),
            downwards(reduce_ors(), type_node.boolean_type),
        )

        # Arghh .. XXX should use reduct_effort==3 for the complexe rule.
        # See this module's own docstring -- ported verbatim, a real
        # upstream oddity, not fixed here.
        reduct_effort = 2

        self.p_medium = sequential(
            downwards(simplify_predicates(reduct_effort, ignore_ops), type_node.boolean_type),
            downwards(reduce_nots(), type_node.boolean_type),
            iterative(sequential(pre_subtree_to_enf, subtree_to_enf(), post_subtree_to_enf)),
            downwards(remove_unary_junctors(), type_node.boolean_type),
            name="medium",
        )

        # complexe
        self.p_complexe = iterative(
            sequential(self.p_medium, reduce_remove_subtree_equal_tt()),
            name="complexe",
        )

    def __call__(self, effort: int = 2) -> rule:
        """C++: moses/comboreduct/reduct/reduct.h:84 logical_reduction::operator()

        Effort 0 (extra simple) to 3 (complex). Returns the RULE for that
        effort level -- the caller applies it separately (`r(effort)(tr,
        it)`, matching every real C++ call site, e.g. `logical_reduce`
        below).
        """
        if effort == 0:
            return type(self).p_extra_simple
        if effort == 1:
            return type(self).p_simple

        # The higher reduction efforts require the use of a different
        # constructor, the ctor that had initialized medium.
        if self.p_medium is None:
            raise AssertionError(
                "Error: logical reduction effort 2 and greater requires ignore_ops"
            )

        if effort == 2:
            return self.p_medium
        if effort == 3:
            return self.p_complexe
        raise AssertionError("Error: logical reduction: no such effort")


def logical_reduce(
    effort: int,
    tr: combo_tree,
    it: tree_node = _NO_IT,
    ignore_ops: set[vertex] | None = None,
) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:116 logical_reduce

    Three C++ overloads collapsed into one Python signature (this port's
    usual convention): `logical_reduce(effort, tr, it, ignore_ops)`
    (reduct.h:116-122), `logical_reduce(effort, tr, ignore_ops)`
    (reduct.h:124-129, `it` omitted -> the 1-arg `rule.__call__` overload,
    itself `tr.begin()` if `tr` is non-empty), and `logical_reduce(effort,
    tr)` (reduct.h:131-135, `ignore_ops` ALSO omitted -> the default
    `logical_reduction()` constructor, usable only for effort 0/1 -- see
    `logical_reduction.__call__`'s own `AssertionError` for effort >= 2
    without `ignore_ops`). `it`'s default is `reduct.py`'s own `_NO_IT`
    sentinel, matching `clean_reduce`'s identical reasoning.

    `reduct.py` itself declares this is where the `xxx_reduce` inline
    convenience wrappers would eventually live once Tasks 3-5 landed the
    rule sets they need (its own module docstring); this port places
    `logical_reduce` here, alongside the class it wraps, instead --
    `reduct.py` importing `logical_reduction` at module level would be a
    circular import (`logical_reduction.py` already imports FROM
    `reduct.py`, for `rule`/`_NO_IT`), and this port has no header/
    implementation split to break that cycle the way C++'s `reduct.h`/
    `logical_reduction.cc` do.
    """
    r = logical_reduction(ignore_ops)
    r(effort)(tr, it)
