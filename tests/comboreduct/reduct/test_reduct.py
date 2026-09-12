"""Tests for pymoses/comboreduct/reduct/reduct.py -- the `rule` base class.

C++: moses/comboreduct/reduct/reduct.h:35 rule.
"""

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.comboreduct.reduct.reduct import rule


class _RecordApply(rule):
    """Minimal concrete `rule`: records every (tr, it) it is applied to,
    as the it node's `.data`, without mutating anything.
    """

    def __init__(self, log, name="record_apply"):
        super().__init__(name)
        self.log = log

    def apply(self, tr, it):
        self.log.append(it.data)


def test_rule_cannot_be_instantiated_directly():
    """`rule.apply` is a pure virtual in C++ (reduct.h:41, `=0`) -- ABC's
    `@abstractmethod` reproduces that: no concrete `.apply` means no
    instance.
    """
    with pytest.raises(TypeError):
        rule("bare")


def test_call_with_explicit_it_dispatches_straight_to_apply():
    """C++: reduct.h:41 rule::operator()(combo_tree&, combo_tree::iterator)
    -- the 2-arg call never consults `tr.empty()`, it goes straight to the
    override.

    FIX WAVE (Task 2 review, Important 1): the original version of this
    test passed `tr.begin()` as `it` -- indistinguishable from `__call__`
    silently IGNORING `it` and always applying at the root anyway. Passing
    a NON-root node (`tr.begin()`'s second child) and asserting `apply`
    received exactly THAT node's data (not the root's) actually pins that
    `it` is used, not merely accepted.
    """
    tr = str_to_combo_tree("and($1 $2)")
    non_root = list(tr.children(tr.begin()))[1]  # the second child, $2
    log = []
    r = _RecordApply(log)
    r(tr, non_root)
    assert log == [non_root.data]
    assert log != [tr.begin().data]


def test_call_with_one_arg_resolves_to_tr_begin_when_non_empty():
    """C++: reduct.h:44 rule::operator()(combo_tree&) -- `if (!tr.empty())
    (*this)(tr, tr.begin());`.
    """
    tr = str_to_combo_tree("and($1 $2)")
    log = []
    r = _RecordApply(log)
    r(tr)
    assert log == [tr.begin().data]


def test_call_with_one_arg_on_an_empty_tree_is_a_silent_no_op():
    """C++: reduct.h:46 `if (!tr.empty())` -- an empty tree never reaches
    `apply` at all, and this must not raise.
    """
    tr = combo_tree()
    log = []
    r = _RecordApply(log)
    r(tr)
    assert log == []


def test_call_with_explicit_none_it_raises():
    """FIX WAVE (Task 2 review, Important 3): an explicit `it=None` must
    be rejected, not silently treated the same as omitting `it` entirely
    (which would silently apply at `tr.begin()` instead -- exactly the
    hazard a real `r(tr, tree.next_sibling(x))` call at the end of a
    sibling list would hit, since `next_sibling` legitimately returns
    `None` for "no such node"). C++ has no way to conflate these: the two
    `operator()` overloads are distinct at compile time.
    """
    tr = str_to_combo_tree("and($1 $2)")
    r = _RecordApply([])
    with pytest.raises(ValueError):
        r(tr, None)


def test_get_name_returns_the_constructor_name():
    """C++: reduct.h:50 rule::get_name"""
    r = _RecordApply([], name="my_rule")
    assert r.get_name() == "my_rule"


def test_clone_returns_a_new_object_with_the_same_fields():
    """C++: reduct.h:42 rule::clone [pure virtual] / reduct.h:67
    crule<T>::clone. See rule.clone()'s own docstring for why this port
    collapses crule<T> into one generic base-class implementation: a new
    object (`clone() is not r`), same field values -- everything a caller
    can actually observe about a C++ `new T(*this)` clone.
    """
    log = []
    r = _RecordApply(log, name="my_rule")
    c = r.clone()
    assert c is not r
    assert isinstance(c, _RecordApply)
    assert c.get_name() == "my_rule"
    # the clone's own log reference is shared (shallow copy) -- appending
    # through either is observable through both, matching the fact that
    # no rule ever mutates its own fields after construction (see
    # rule.clone()'s DIVERGENCE note).
    assert c.log is r.log
