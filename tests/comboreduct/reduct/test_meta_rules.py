"""Tests for pymoses/comboreduct/reduct/meta_rules.py -- the traversal and
fixpoint combinators (`downwards`, `upwards`, `sequential`, `iterative`,
`assum_iterative`).

C++: moses/comboreduct/reduct/meta_rules.{h,cc}.

VISIT ORDER is the property under test, not just the end result -- per
this task's brief. Every literal expected value below is copied verbatim
from a REAL run of `tools/cpp_probe/meta_rules_probe.cc` against
$CLASSIC's compiled `libcomboreduct.so` (bounded:
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/meta_rules_probe
)`), not hand-derived from the C++ source comments alone. See that file
and meta_rules.py's own module docstring for the full transcript.
"""

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.assumption import insert_assumption
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.meta_rules import (
    assum_iterative,
    downwards,
    iterative,
    sequential,
    upwards,
)
from pymoses.comboreduct.reduct.reduct import rule
from pymoses.comboreduct.table.table import type_node


class _RecordVisit(rule):
    """C++ probe counterpart: `record_visit` (tools/cpp_probe/
    meta_rules_probe.cc). Appends `tag + combo_tree_to_str(combo_tree(it))`
    to a shared log at every point of application -- non-mutating.
    `clone()` (the generic default from `rule`) shares the SAME `log`
    list by reference (shallow copy), matching the C++ probe's clones
    sharing one `vector<string>&` -- so wrapping this in downwards/
    upwards/sequential (which each clone their wrapped rule at
    construction) still logs into ONE sequence.
    """

    def __init__(self, log, tag="", name="record_visit"):
        super().__init__(name)
        self.log = log
        self.tag = tag

    def apply(self, tr, it):
        self.log.append(self.tag + combo_tree_to_str(combo_tree(it)))


class _StepArg(rule):
    """C++ probe counterpart: `step_arg`. Bumps an `argument`'s index by 1
    each call, until it reaches `cap`, then no-ops. Used to give
    `iterative`'s fixpoint a deterministic, countable number of real
    iterations.
    """

    def __init__(self, cap, calls, name="step_arg"):
        super().__init__(name)
        self.cap = cap
        self.calls = calls  # a single-element list used as a mutable counter

    def apply(self, tr, it):
        self.calls[0] += 1
        if isinstance(it.data, argument) and it.data.idx < self.cap:
            tr.replace(it, argument(it.data.idx + 1))


class _StepFixedTarget(rule):
    """C++ probe counterpart: `step_target` (tools/cpp_probe/
    meta_rules_probe.cc). Always bumps a FIXED node captured at
    construction, regardless of the point of application `it` the
    combinator actually passes -- used to check what `iterative`/
    `assum_iterative` do when the wrapped rule mutates something OUTSIDE
    `it`'s own subtree (Task 2 review, Important 1).
    """

    def __init__(self, target, cap, calls, name="step_target"):
        super().__init__(name)
        self.target = target
        self.cap = cap
        self.calls = calls

    def apply(self, tr, it):
        self.calls[0] += 1
        if isinstance(self.target.data, argument) and self.target.data.idx < self.cap:
            tr.replace(self.target, argument(self.target.data.idx + 1))


class _GrowAssumptions(rule):
    """C++ probe counterpart: `grow_assumptions`. Inserts one new
    (distinct) assumption per call until `tr` already has `cap`
    assumptions, then no-ops. Never touches the main tree.
    """

    def __init__(self, cap, calls, start_idx, name="grow_assumptions"):
        super().__init__(name)
        self.cap = cap
        self.calls = calls
        self.next_idx = [start_idx]

    def apply(self, tr, it):
        self.calls[0] += 1
        count = len(getattr(tr, "assumptions", []))
        if count < self.cap:
            a = combo_tree(argument(self.next_idx[0]))
            self.next_idx[0] += 1
            insert_assumption(tr, a.begin())


class _InsertSiblingOnce(rule):
    """C++ probe counterpart: `insert_sibling_once`. The first time it
    sees `argument(2)`, inserts a brand-new sibling ($99) immediately
    after it, and records the visit sequence -- the LIVE-MUTATION hazard
    for `downwards`'s own subtree-bounded walk.
    """

    def __init__(self, log, name="insert_sibling_once"):
        super().__init__(name)
        self.log = log
        self.done = [False]

    def apply(self, tr, it):
        self.log.append(combo_tree_to_str(combo_tree(it)))
        if isinstance(it.data, argument) and it.data.idx == 2 and not self.done[0]:
            self.done[0] = True
            tr.insert_after(it, argument(99))


def assumption_count(tr):
    return len(getattr(tr, "assumptions", []))


# ---------------------------------------------------------------------------
# downwards: pre-order (parents before children)
# ---------------------------------------------------------------------------


def test_downwards_visits_pre_order():
    tr = str_to_combo_tree("and($1 or($2 $3) not($4))")
    log = []
    downwards(_RecordVisit(log))(tr)
    assert log == [
        "and($1 or($2 $3) not($4)) ",
        "$1 ",
        "or($2 $3) ",
        "$2 ",
        "$3 ",
        "not($4) ",
        "$4 ",
    ]


# ---------------------------------------------------------------------------
# upwards: post-order (children before parents)
# ---------------------------------------------------------------------------


def test_upwards_visits_post_order():
    tr = str_to_combo_tree("and($1 or($2 $3) not($4))")
    log = []
    upwards(_RecordVisit(log))(tr)
    assert log == [
        "$1 ",
        "$2 ",
        "$3 ",
        "or($2 $3) ",
        "$4 ",
        "not($4) ",
        "and($1 or($2 $3) not($4)) ",
    ]


# ---------------------------------------------------------------------------
# sequential: declaration order, same point of application
# ---------------------------------------------------------------------------


def test_sequential_applies_each_rule_in_order_at_the_same_node():
    tr = str_to_combo_tree("and($1 $2)")
    log = []
    sequential(
        _RecordVisit(log, tag="A:"),
        _RecordVisit(log, tag="B:"),
        _RecordVisit(log, tag="C:"),
    )(tr)
    assert log == ["A:and($1 $2) ", "B:and($1 $2) ", "C:and($1 $2) "]


def test_sequential_propagates_a_mutation_from_one_rule_to_the_next():
    """Minor (Task 2 review): every prior `sequential` test used only
    non-mutating `_RecordVisit` sub-rules, leaving the actual PROPAGATION
    of a mutation from one wrapped rule to the next entirely unpinned --
    the first rule mutates `it` in place; the second, applied right after
    at the SAME live `tr`/`it` (`r(tr, it)`, meta_rules.cc:122-128, not a
    snapshot each takes independently), must observe the POST-mutation
    state.
    """
    tr = str_to_combo_tree("$1")
    calls = [0]
    log = []
    sequential(_StepArg(3, calls), _RecordVisit(log))(tr)
    assert calls[0] == 1
    assert log == ["$2 "]  # the record rule sees $1 already bumped to $2


def test_downwards_of_sequential_interleaves_both_rules_at_each_node():
    """The combinators compose: `downwards` picks WHERE, `sequential`
    picks WHAT happens at each of those points, fully, before advancing.
    """
    tr = str_to_combo_tree("and($1 $2)")
    log = []
    downwards(sequential(_RecordVisit(log, tag="A:"), _RecordVisit(log, tag="B:")))(tr)
    assert log == [
        "A:and($1 $2) ",
        "B:and($1 $2) ",
        "A:$1 ",
        "B:$1 ",
        "A:$2 ",
        "B:$2 ",
    ]


# ---------------------------------------------------------------------------
# downwards: LIVE walk, not a snapshot
# ---------------------------------------------------------------------------


def test_downwards_is_a_live_walk_a_mutation_mid_pass_is_visited():
    tr = str_to_combo_tree("and($1 $2 $3)")
    log = []
    downwards(_InsertSiblingOnce(log))(tr)
    assert log == ["and($1 $2 $3) ", "$1 ", "$2 ", "$99 ", "$3 "]
    assert combo_tree_to_str(tr) == "and($1 $2 $99 $3) "


# ---------------------------------------------------------------------------
# downwards: the typed-dispatch branch (Task 2 review, Important 2; the
# real slice of type_checker/type_tree landed by Task 5 -- see meta_rules.py's
# own module docstring and pymoses.comboreduct.type_checker.type_tree's).
# ---------------------------------------------------------------------------


def test_downwards_typed_boolean_type_skips_bare_argument_leaves():
    """MEASURED against the real library (`tools/cpp_probe/
    meta_rules_probe.cc`'s case 11, `downwards(typed boolean_type)`):
    `get_output_type_tree(argument)` is ALWAYS `unknown_type`, so a typed
    `downwards(r, id::boolean_type)` over `and($1 $2)` visits ONLY the
    root -- never the two argument leaves an UNTYPED `downwards(r)` would
    also visit (`test_downwards_visits_pre_order` above). This is the
    exact mechanism `logical_reduction.py`'s effort-2/3 compositions
    measurably diverge from an untyped approximation on.
    """
    tr = str_to_combo_tree("and($1 $2)")
    log = []
    downwards(_RecordVisit(log), type_node.boolean_type)(tr)
    assert log == ["and($1 $2) "]


def test_downwards_typed_boolean_type_visits_a_nested_boolean_operator_too():
    """MEASURED (`meta_rules_probe.cc` case 12,
    `downwards(typed boolean_type, nested or)`): the root `and` AND the
    nested `or` both resolve to `boolean_type` (both are logical
    operators), so both are visited -- only the argument leaves are
    skipped.
    """
    tr = str_to_combo_tree("and($1 or($2 $3))")
    log = []
    downwards(_RecordVisit(log), type_node.boolean_type)(tr)
    assert log == ["and($1 or($2 $3)) ", "or($2 $3) "]


def test_downwards_ignores_output_when_input_is_unknown_type():
    """C++'s own asymmetric check (`if (input == unknown_type_tree)`,
    meta_rules.cc:89) tests ONLY `input` -- `output` is never even
    consulted when `input` is `unknown_type`, so the untyped (visit
    everything) path is taken even if `output` is set to a real type via
    the 3-arg constructor. This must NOT raise (the typed branch, and
    `_get_output_type_tree` with it, must never be reached).
    """
    tr = str_to_combo_tree("and($1 $2)")
    log = []
    downwards(_RecordVisit(log), type_node.unknown_type, type_node.boolean_type)(tr)
    assert log == ["and($1 $2) ", "$1 ", "$2 "]


# ---------------------------------------------------------------------------
# iterative: fixpoint AT the point of application
# ---------------------------------------------------------------------------


def test_iterative_reapplies_until_the_subtree_stops_changing():
    tr = str_to_combo_tree("$1")
    calls = [0]
    iterative(_StepArg(3, calls))(tr)
    assert calls[0] == 3
    assert combo_tree_to_str(tr) == "$3 "


def test_iterative_calls_the_wrapped_rule_at_least_once():
    """Do-while: the wrapped rule always runs at least once, even if it
    never changes anything (cap=0 means the very first call is already a
    no-op).
    """
    tr = str_to_combo_tree("$1")
    calls = [0]
    iterative(_StepArg(0, calls))(tr)
    assert calls[0] == 1
    assert combo_tree_to_str(tr) == "$1 "


def _build_and_of_three_args():
    """`and($1 $2 $3)`, built node by node so the caller keeps direct
    handles to each child (mirroring `tools/cpp_probe/meta_rules_probe.cc`
    cases 9/10, which need the same direct handles in C++).
    """
    tr = combo_tree(builtin.logical_and)
    root = tr.begin()
    c1 = tr.append_child(root, argument(1))
    c2 = tr.append_child(root, argument(2))
    c3 = tr.append_child(root, argument(3))
    return tr, c1, c2, c3


def test_iterative_at_a_non_root_it_ignores_mutations_outside_it():
    """Task 2 review, Important 1: NOTHING in this file previously passed
    a non-root point of application -- `iterative`'s "compares AT `it`"
    scope (vs `assum_iterative`'s "compares the whole tree") was entirely
    unpinned; swapping either combinator's comparison scope to the
    other's left all prior tests green. `it = c2` (the middle child, a
    non-root node with real siblings); the wrapped rule mutates `c1` (a
    SIBLING of `it`, never `it` itself). `iterative`'s fixpoint check
    (`tr.equal_subtree(it, tmp.begin())`, meta_rules.cc:139) is LOCAL to
    `it`'s own subtree, so it must never even notice `c1` changing --
    exactly one call, even though the rule itself, if left running,
    would keep mutating `c1` for another three calls.

    Verified against the real library
    (`tools/cpp_probe/meta_rules_probe.cc`, case 9): `calls=1
    final=and($2 $2 $3)` -- one call DOES happen (do-while) and DOES
    mutate `c1` once (1->2), but the loop stops right there because the
    check it actually runs never looks at `c1` at all.
    """
    tr, c1, c2, _c3 = _build_and_of_three_args()
    calls = [0]
    iterative(_StepFixedTarget(c1, 4, calls))(tr, c2)
    assert calls[0] == 1
    assert combo_tree_to_str(tr) == "and($2 $2 $3) "


# ---------------------------------------------------------------------------
# assum_iterative: fixpoint over BOTH the main tree and the assumption set
# ---------------------------------------------------------------------------


def test_assum_iterative_keeps_iterating_while_assumptions_keep_changing():
    """THE assum_iterative DECISION, pinned: without carrying `tr.
    assumptions` forward into each iteration's snapshot, this would stop
    after 1 call (see the next test's control case) instead of 3.
    """
    tr = str_to_combo_tree("and($1 $2)")
    seed = combo_tree(argument(9))
    insert_assumption(tr, seed.begin())
    assert assumption_count(tr) == 1

    calls = [0]
    assum_iterative(_GrowAssumptions(3, calls, 10))(tr)
    assert calls[0] == 3
    assert assumption_count(tr) == 3
    # the main "conclusion" tree itself was never touched by the wrapped rule
    assert combo_tree_to_str(tr) == "and($1 $2) "


def test_assum_iterative_stops_after_one_call_when_nothing_changes():
    """Control case: a wrapped rule that never touches the main tree NOR
    the assumption set must still run once (do-while), then stop.
    """
    tr = str_to_combo_tree("and($1 $2)")
    seed = combo_tree(argument(9))
    insert_assumption(tr, seed.begin())

    calls = [0]
    assum_iterative(_GrowAssumptions(0, calls, 10))(tr)  # cap=0: instant no-op
    assert calls[0] == 1
    assert assumption_count(tr) == 1


def test_assum_iterative_at_a_non_root_it_still_sees_the_whole_tree():
    """Counterpart of `test_iterative_at_a_non_root_it_ignores_mutations_
    outside_it`, same tree/rule/target, same `it = c2`: `assum_iterative`'s
    fixpoint check compares the WHOLE main tree at `tr.begin()`
    (`tr!=tmp`, meta_rules.cc:150 -- `tree<T>::operator!=`, confirmed by
    reading it directly to be `!equal_subtree(t1.begin(), t2.begin())`,
    $COGUTIL's opencog/util/tree.h:3011-3018), REGARDLESS of `it`. So,
    unlike `iterative` above, it DOES notice `c1` changing even though
    `it = c2` never changes itself, and keeps iterating until `c1`
    actually stops (cap=4): 1->2, 2->3, 3->4, 4->4 (no-op, detected) = 4
    calls.

    Verified against the real library
    (`tools/cpp_probe/meta_rules_probe.cc`, case 10): `calls=4
    final=and($4 $2 $3)`.
    """
    tr, c1, c2, _c3 = _build_and_of_three_args()
    calls = [0]
    assum_iterative(_StepFixedTarget(c1, 4, calls))(tr, c2)
    assert calls[0] == 4
    assert combo_tree_to_str(tr) == "and($4 $2 $3) "
