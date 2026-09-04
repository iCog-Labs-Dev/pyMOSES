"""C++: moses/comboreduct/reduct/meta_rules.{h,cc}

`downwards`, `upwards`, `sequential`, `iterative`, `assum_iterative` -- the
traversal and fixpoint combinators every rule set (Tasks 3-5) is built
from. THIS ORDERING IS DELIBERATE: these decide traversal direction, where
a wrapped rule is applied, and when a fixpoint terminates, so they are
ported -- and their VISIT ORDER tested -- before any actual rule exists to
wrap. Getting this file wrong would let a fully-correct rule set still
converge to a different normal form than C++, silently, with no
rule-level test able to catch it (see the task brief and
docs/superpowers/plans/2026-09-04-pymetta-s7-reduct.md's Task 2 section).

Two C++ symbols this same header declares are DELIBERATELY NOT PORTED:

* `when` (meta_rules.h:44-56) and `ignore_size_increase` (meta_rules.h:
  60-71) -- both real meta-rules, but zero call sites anywhere within S7's
  actual scope: `grep -rln "when(\\|ignore_size_increase(" logical_
  reduction.cc general_rules.cc logical_rules.cc clean_reduction.cc`
  under $CLASSIC's moses/comboreduct/reduct/ returns nothing. Both are
  used only by `contin_reduction.cc` (S14, mixed/contin reduction --
  CLAUDE.md's own "Known gap, deliberate"). `ignore_size_increase` is
  also the ONLY real call site of reduct.h's `replace_without_changing_it`
  (see reduct.py's own module docstring), which is why that helper is
  likewise not ported here.

VISIT ORDER, verified against the real compiled library, not inferred
from the C++ comments alone (`tools/cpp_probe/meta_rules_probe.cc`,
built against $CLASSIC's actual `libcomboreduct.so` and run bounded --
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/meta_rules_probe
)`). Every literal expected sequence in this task's test files is a
value from that probe's real stdout, not a hand-derived guess:

    downwards:              [and(...), $1, or($2 $3), $2, $3, not($4), $4]
    upwards:                [$1, $2, $3, or($2 $3), $4, not($4), and(...)]
    sequential(A,B,C):      [A:..., B:..., C:...]           (same node)
    downwards(sequential(A,B)):
                            [A:root, B:root, A:$1, B:$1, A:$2, B:$2]
    iterative(step $1->$2->$3, cap 3):     calls=3, final=$3
    assum_iterative(grow assumptions to 3):
                            initial=1, calls=3, final=3
    assum_iterative(no-op rule):           calls=1, final=1 (control)
    downwards + insert-sibling-after mid-walk:
                            [and($1 $2 $3), $1, $2, $99, $3]  (the new
                            sibling IS visited by the SAME pass -- live
                            re-derivation, not a snapshot)

`downwards`'s typed constructors (meta_rules.h:83,87 -- `downwards(r,
combo::type_node t)` / `downwards(r, input_, output_)`) are ported for
SHAPE (the constructors accept and store the type arguments, and
`apply`'s branch structure matches C++'s `if (input == unknown_type_tree)
... else ...` exactly), but the TYPED branch's actual comparison --
`opencog::combo::get_output_type_tree(*it) == type_tree(output)`
(meta_rules.cc:101) -- is NOT implemented; see `_get_output_type_tree`
below. This is a genuine, disclosed gap, not a silent one: `type_checker/
type_tree.{h,cc}` is PLANNED: S8 per docs/MODULE-COVERAGE.md (the same
kind of ordering hazard already documented there for `build_knobs`), so
`get_output_type_tree` has nothing to call. Every one of THIS task's own
tests exercises only the untyped (default) `downwards(r)` form -- the one
path that never reaches `_get_output_type_tree` -- so this gap does not
block Task 2. It WILL block Task 5 (`logical_reduction.py` constructs
`downwards(rule, id::boolean_type)` extensively, e.g.
logical_reduction.cc:56,61,64-65); flagged prominently in
task-2-report.md's Concerns section for Task 5 to resolve deliberately
(most plausibly a minimal, scoped slice of `get_output_type_tree` --
mirroring how S5 copied only `type_node`'s enum into table.py rather than
porting type_checker/type_tree.h wholesale -- rather than the general
1468-line type-inference engine that file also contains).
"""

from pymetta.comboreduct.combo.assumption import copy_all_assumptions, equal_assumptions
from pymetta.comboreduct.combo.vertex import combo_tree, equal_subtree_vertex_safe
from pymetta.comboreduct.reduct.reduct import rule
from pymetta.comboreduct.table.table import type_node
from pymetta.util.tree import tree_node


def _get_output_type_tree(v):
    """C++: moses/comboreduct/type_checker/type_tree.cc:250 get_output_type_tree

    NOT PORTED -- see this module's own docstring for the full reasoning
    (an S8 ordering hazard, not a silent gap). Reached only from
    `downwards.apply`'s typed branch, which none of Task 2's own tests
    exercise.

    # DIVERGENCE(C++ moses/comboreduct/reduct/meta_rules.cc:101): the
    # typed-dispatch comparison `get_output_type_tree(*it) ==
    # type_tree(output)` is not implemented -- this stub raises instead
    # of comparing. Observable impact: `downwards(rule, some_real_type)`
    # (not the default untyped `downwards(rule)`) raises
    # `NotImplementedError` on first use instead of filtering by output
    # type, for EVERY real (non-`unknown_type`) type argument. Added (Task
    # 2 review, Important 2) specifically so this shows up in
    # `docs/ALIGNMENT.md`'s divergence table -- a bare `NotImplementedError`
    # with no marker left `meta_rules.py` reporting 100% coverage with
    # zero divergences, blind to the file's largest known gap. `type_
    # checker/type_tree.{h,cc}` is PLANNED: S8 per docs/MODULE-COVERAGE.md
    # (the same kind of ordering hazard already documented there for
    # `build_knobs`); Task 5 (`logical_reduction.py`, still S7 scope)
    # constructs `downwards(rule, id::boolean_type)` extensively and WILL
    # hit this -- see this module's own docstring and task-2-report.md's
    # Concerns section.
    """
    raise NotImplementedError(
        "downwards' typed-dispatch branch needs comboreduct/type_checker/"
        "type_tree.get_output_type_tree, PLANNED: S8 per "
        "docs/MODULE-COVERAGE.md. See meta_rules.py's module docstring and "
        "task-2-report.md's Concerns section (flagged for Task 5)."
    )


class downwards(rule):
    """C++: moses/comboreduct/reduct/meta_rules.h:77 downwards

    Apply the wrapped rule in PRE-ORDER (left-to-right, parents before
    children) over `it`'s own subtree -- `it` itself is visited FIRST.
    """

    def __init__(
        self,
        r: rule,
        input_type: type_node = type_node.unknown_type,
        output_type: type_node | None = None,
        name: str = "downwards",
    ) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.h:79 downwards

        Three constructor overloads (79, 83, 87), collapsed into one
        Python signature via defaults (the convention this port uses
        throughout for overloaded C++ constructors -- see e.g. tree.py's
        own combined `tree(iterator)` constructor):

            downwards(r)                 -> input=output=unknown_type
                                             (meta_rules.h:79)
            downwards(r, t)               -> input=output=t
                                             (meta_rules.h:83; `output_type`
                                             left `None` resolves to
                                             `input_type` below)
            downwards(r, input_, output_) -> both given explicitly
                                             (meta_rules.h:87)
        """
        super().__init__(name)
        self.r = r.clone()
        self.input = input_type
        self.output = input_type if output_type is None else output_type

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.cc:79 downwards::operator()

        `tr.iter_pre_at(it)` (pymetta/util/tree.py, added this task as
        the pre-order counterpart of Task 0's `iter_post_at`) is exactly
        `combo_tree::iterator end = it; end.skip_children(); ++end;` plus
        the `for (; it != end; ++it)` loop it bounds. It is a LIVE walk,
        not a snapshot -- `list(tr.iter_pre_at(it))` up front would be a
        real behavioural DIVERGENCE here, silently changing what a
        mutating wrapped rule can and cannot affect mid-walk (see
        `iter_pre_at`'s own docstring, and the module docstring's
        "insert-sibling-after mid-walk" probe result, for the measured
        real behaviour this reproduces).

        Only `self.input` is compared against `unknown_type` here --
        `self.output` is IGNORED whenever `self.input` is `unknown_type`,
        even if `self.output` was set to something else via the 3-arg
        constructor. This exactly mirrors C++'s own asymmetric check
        (`if (input == unknown_type_tree)`, meta_rules.cc:89 -- `output`
        never even appears in that branch), not something this port
        smooths over.
        """
        if self.input == type_node.unknown_type:
            for node in tr.iter_pre_at(it):
                self.r(tr, node)
        else:
            for node in tr.iter_pre_at(it):
                if _get_output_type_tree(node.data) == self.output:
                    self.r(tr, node)


class upwards(rule):
    """C++: moses/comboreduct/reduct/meta_rules.h:107 upwards

    Apply the wrapped rule in POST-ORDER (left-to-right, children before
    parent, deepest first) over `it`'s own subtree -- `it` itself is
    visited LAST.
    """

    def __init__(self, r: rule, name: str = "upwards") -> None:
        """C++: moses/comboreduct/reduct/meta_rules.h:108 upwards::upwards"""
        super().__init__(name)
        self.r = r.clone()

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.cc:108 upwards::operator()

        `tr.iter_post_at(it)` (Task 0) IS `combo_tree::post_order_iterator
        at=it, end=it; ++end; at.descend_all();` plus the loop it bounds
        -- see that method's own docstring.
        """
        for node in tr.iter_post_at(it):
            self.r(tr, node)


class sequential(rule):
    """C++: moses/comboreduct/reduct/meta_rules.h:149 sequential

    Apply every wrapped rule, IN ORDER, at the SAME point of application.
    """

    def __init__(self, *rules: rule, name: str = "sequential") -> None:
        """C++: moses/comboreduct/reduct/meta_rules.h:151 sequential

        The copy constructor (151) plus the 50 boost-preprocessor-generated
        variadic constructors (157-165: `sequential(r, r1)`, `sequential(r,
        r1, r2)`, ... up to 50 rules) collapse to Python's native `*args`;
        `boost::ptr_vector<rule> rules` (meta_rules.h:171) is a plain
        Python `list`.
        """
        super().__init__(name)
        self.rules = [r.clone() for r in rules]

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.cc:122 sequential::operator()

        `r(tr, it)` for each wrapped rule `r`, in declaration order --
        this is the 2-arg call (`__call__` with `it` already given),
        dispatching straight to that rule's own `apply`, never the 1-arg
        empty-tree-guarded wrapper.
        """
        for r in self.rules:
            r(tr, it)


class iterative(rule):
    """C++: moses/comboreduct/reduct/meta_rules.h:121 iterative

    Apply the wrapped rule repeatedly AT the point of application `it`
    until `it`'s own subtree stops changing (a fixpoint).
    """

    def __init__(self, r: rule | None = None, name: str = "iterative") -> None:
        """C++: moses/comboreduct/reduct/meta_rules.h:122 iterative

        Lines 122-125, including the C++ default constructor
        (`iterative()`, no wrapped rule at all). Verified by
        grep across every `*.cc` under $CLASSIC's moses/comboreduct/
        reduct/ that this default constructor is never actually invoked
        anywhere real (every real use passes a rule: `iterative(sequential
        (...))` etc.) -- kept here purely for interface fidelity, not
        because it is reachable. Calling `apply` on an `iterative()` built
        this way fails (there is no wrapped rule to call), matching what a
        null-`shared_ptr` dereference in C++ would do if this path were
        ever actually exercised, which real $CLASSIC never does.
        """
        super().__init__(name)
        self.r = None if r is None else r.clone()

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.cc:131 iterative::operator()

        `combo_tree(it)` (tree.py's iterator-copy constructor, added
        Task 1) is C++'s `combo_tree tmp = combo_tree(it)` -- a
        STANDALONE deep copy of `it`'s CURRENT subtree, snapshotted BEFORE
        the wrapped rule runs. `equal_subtree_vertex_safe` (vertex.py),
        NOT `tree.equal_subtree`, compares the post-apply subtree at `it`
        against that snapshot -- the same type-before-value reason
        `assumption.equal_assumptions` uses it, and load-bearing here for
        the identical reason: `tree.equal_subtree`'s bare `a.data !=
        b.data` was flagged by Task 1's review as reachable from exactly
        this fixpoint check, and a false "already equal" from a
        `contin_t`/`builtin` collision would terminate the loop one
        iteration too early.
        """
        while True:
            tmp = combo_tree(it)
            self.r(tr, it)
            if equal_subtree_vertex_safe(it, tmp.begin()):
                break


class assum_iterative(rule):
    """C++: moses/comboreduct/reduct/meta_rules.h:135 assum_iterative

    Like `iterative`, but the fixpoint check ALSO covers the assumption
    set -- keeps iterating if either the main tree OR the assumptions
    changed.
    """

    def __init__(self, r: rule | None = None, name: str = "assum_iterative") -> None:
        """C++: moses/comboreduct/reduct/meta_rules.h:136 assum_iterative

        Lines 136-141 -- see `iterative.__init__`'s docstring; the same
        never-actually-invoked default constructor exists here too, for
        the same reason.
        """
        super().__init__(name)
        self.r = None if r is None else r.clone()

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/meta_rules.cc:143 assum_iterative::operator()

        ASSUM_ITERATIVE DECISION (deferred by Task 1's review, settled
        here): C++'s `tmp = tr;` (meta_rules.cc:147) is a WHOLE-TREE value
        copy. For a real C++ `combo_tree` that copies BOTH the main
        "conclusion" tree AND every assumption sibling held under the
        shared head/feet sentinels (assumption.py's own module
        docstring) -- there is only ONE tree object in C++, and
        assignment copies all of it. This port's `combo_tree` carries
        assumptions as a SIDE LIST, `tr.assumptions` (assumption.py), not
        real tree-node siblings, and a bare `combo_tree(some_node)` copy
        (the constructor `iterative.apply` uses above) never carries that
        list forward -- a bare `tree_node` has no back-reference to
        whatever tree it came from (assumption.py's own "UNSETTLED" note,
        now settled). Reproducing `tmp = tr` therefore needs TWO Python
        steps in place of C++'s one assignment: `combo_tree(tr.begin())`
        (copies the main tree) PLUS `copy_all_assumptions(tmp, tr)`
        (assumption.py, added by Task 1 for exactly this dest-always-empty
        call pattern) to carry `.assumptions` across too.

        DECISION: carry them. Not a `# DIVERGENCE` -- this reproduces
        C++'s real `tmp = tr` semantics exactly, via the two primitives
        this port has in place of one C++ copy-assignment operator.
        Skipping the second step would make `tmp` report zero assumptions
        on every iteration, and `equal_assumptions`'s own documented
        behaviour ("true whenever EITHER side has zero assumptions") would
        then make the assumption half of the fixpoint check
        UNCONDITIONALLY pass every time -- silently degrading
        `assum_iterative` to plain `iterative`'s main-tree-only fixpoint.
        That is exactly the "converges to a different normal form"
        failure this whole cycle exists to prevent, and it is MEASURED,
        not hypothetical: `tools/cpp_probe/meta_rules_probe.cc`'s
        "assum_iterative" case (pre-seed 1 assumption, a rule that grows
        the set to 3 before going quiet) gets calls=3/final=3 against the
        real library; a `tmp` that never carries assumptions forward would
        make an EQUIVALENT Python rule stop after just 1 call, matching
        this module's own "no-op rule" control case (calls=1/final=1)
        instead -- observably wrong, not just theoretically so.

        `tr != tmp` (meta_rules.cc:150, `tree<T>::operator!=`) is, per
        opencog/util/tree.h:3011-3018, `!t1.equal_subtree(t1.begin(),
        t2.begin())` -- comparing ONLY `t1.begin()`'s subtree, NEVER the
        assumption siblings (confirmed by reading that operator directly).
        That is exactly why `assum_iterative` needs the SEPARATE
        `equal_assumptions` check at all: plain tree comparison is blind
        to the assumption region. It is also why this comparison is
        against the WHOLE tree's OWN root (`tr.begin()`), regardless of
        what `it` is -- unlike `iterative.apply`, which compares AT the
        point of application `it` (meta_rules.cc:139). A real, narrower-
        vs-wider distinction between the two combinators, preserved here
        deliberately rather than unified with `iterative`'s own check.
        `equal_subtree_vertex_safe`, not `tree.equal_subtree`, is used for
        the main-tree half too, for the same type-before-value reason as
        `iterative.apply`.
        """
        while True:
            tmp = combo_tree(tr.begin())
            copy_all_assumptions(tmp, tr)
            self.r(tr, it)
            tr_root, tmp_root = tr.begin(), tmp.begin()
            same_tree = (tr_root is None and tmp_root is None) or (
                tr_root is not None
                and tmp_root is not None
                and equal_subtree_vertex_safe(tr_root, tmp_root)
            )
            if same_tree and equal_assumptions(tmp, tr):
                break
