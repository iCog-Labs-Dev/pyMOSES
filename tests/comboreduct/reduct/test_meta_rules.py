"""Tests for pymetta/comboreduct/reduct/meta_rules.py -- the traversal and
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

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.assumption import insert_assumption
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.reduct.meta_rules import (
    assum_iterative,
    downwards,
    iterative,
    sequential,
    upwards,
)
from pymetta.comboreduct.reduct.reduct import rule


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
