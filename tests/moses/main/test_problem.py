"""Guard tests for pymetta.moses.main.problem (S13 Task 0).

No C++ oracle/fixture is built for this module: `register_problem`'s
duplicate-key behaviour is `std::map::insert`'s textbook, unambiguous
"no-op if the key exists" contract (cppreference, not a project-specific
quirk), and `find_problem`'s miss behaviour is a plain `NULL` return --
neither has any undefined-behaviour or floating-point subtlety the way
this project's other oracles (crossover, hill-climbing, ...) do. Fidelity
is established by direct citation of problem.cc's exact lines (see
problem.py's module and method docstrings) rather than a compiled probe.

Every guard below was proven by mutation during development: break the
guarded line, run this file, see the specific test go red; revert; see it
green again. See task-0-report.md for the list and results.
"""

from __future__ import annotations

import pytest

from pymetta.moses.main.problem import (
    _register_unported_table_problems,
    _unported_table_problem,
    _unported_table_problems,
    problem_base,
    problem_manager,
)

# ---------------------------------------------------------------------------
# problem_base: abstract interface
# ---------------------------------------------------------------------------


def test_problem_base_cannot_be_instantiated_directly():
    """C++ cannot instantiate a class with an unoverridden pure virtual
    (`name`/`description`/`run` are all `= 0`, problem.h:56-58); `ABC` +
    `@abstractmethod` is this port's equivalent enforcement mechanism.
    """
    with pytest.raises(TypeError):
        problem_base()


def test_problem_base_subclass_missing_run_cannot_be_instantiated():
    class _incomplete(problem_base):
        def name(self) -> str:
            return "x"

        def description(self) -> str:
            return "x"

        # run() deliberately not overridden

    with pytest.raises(TypeError):
        _incomplete()


def test_problem_base_subclass_implementing_all_three_is_instantiable():
    class _complete(problem_base):
        def name(self) -> str:
            return "x"

        def description(self) -> str:
            return "y"

        def run(self, ob: object) -> None:
            del ob

    prob = _complete()
    assert prob.name() == "x"
    assert prob.description() == "y"
    assert prob.run(None) is None


# ---------------------------------------------------------------------------
# Test doubles for problem_manager's own registry logic
# ---------------------------------------------------------------------------


class _dummy_problem(problem_base):
    def __init__(self, name: str, description: str = "", tag: object = None) -> None:
        self._name = name
        self._description = description
        self.tag = tag
        self.ran_with: list[object] = []

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def run(self, ob: object) -> None:
        self.ran_with.append(ob)


# ---------------------------------------------------------------------------
# problem_manager.register_problem / find_problem: basic round trip
# ---------------------------------------------------------------------------


def test_register_then_find_returns_the_same_object():
    pmr = problem_manager()
    prob = _dummy_problem("foo", "does foo things")
    pmr.register_problem(prob)
    found = pmr.find_problem("foo")
    assert found is prob
    assert found.description() == "does foo things"


def test_find_problem_returns_none_for_unknown_name():
    """C++: problem.cc:99-104 `find_problem` returns `NULL` on a miss --
    no assert, no exception, no exit(). Faithful translation: `None`.

    MUTATION PROVEN: changing `find_problem` to
    `return self._problem_set[name]` (a bare `dict.__getitem__`, matching
    what an *incorrect* port might do) makes this test fail with
    `KeyError` instead of passing; reverting to `.get(name)` restores the
    pass. See task-0-report.md.
    """
    pmr = problem_manager()
    pmr.register_problem(_dummy_problem("foo"))
    assert pmr.find_problem("does-not-exist") is None


def test_find_problem_on_a_manager_with_nothing_registered():
    pmr = problem_manager()
    assert pmr.find_problem("anything") is None


def test_register_multiple_distinct_names_are_all_findable():
    pmr = problem_manager()
    a, b, c = _dummy_problem("a"), _dummy_problem("b"), _dummy_problem("c")
    for prob in (a, b, c):
        pmr.register_problem(prob)
    assert pmr.find_problem("a") is a
    assert pmr.find_problem("b") is b
    assert pmr.find_problem("c") is c


# ---------------------------------------------------------------------------
# problem_manager.register_problem: duplicate-name semantics
# ---------------------------------------------------------------------------


def test_register_problem_keeps_first_registered_on_duplicate_name():
    """C++: problem.cc:93-97 `register_problem` builds a `std::pair` and
    calls `_problem_set.insert(pr)` -- `std::map::insert` is a documented
    no-op when the key is already present, so the FIRST-registered
    `problem_base` for a given `name()` wins permanently; a later
    registration under the same name is silently dropped.

    MUTATION PROVEN: changing `register_problem` from
    `self._problem_set.setdefault(prob.name(), prob)` to the plain-
    assignment `self._problem_set[prob.name()] = prob` (Python dict's
    normal "last write wins" semantics -- the natural but WRONG port of
    this line) makes this test fail (`found is first` becomes False,
    `found is second` becomes True); reverting to `setdefault` restores
    the pass. See task-0-report.md.
    """
    pmr = problem_manager()
    first = _dummy_problem("dup", "first description")
    second = _dummy_problem("dup", "second description")
    pmr.register_problem(first)
    pmr.register_problem(second)

    found = pmr.find_problem("dup")
    assert found is first
    assert found is not second
    assert found.description() == "first description"


def test_register_problem_duplicate_does_not_raise():
    """C++ raises/asserts nothing for a duplicate registration -- the
    discarded object is simply never inserted. No exception here either.
    """
    pmr = problem_manager()
    pmr.register_problem(_dummy_problem("dup", "one"))
    pmr.register_problem(_dummy_problem("dup", "two"))  # must not raise


def test_register_same_object_twice_under_same_name_is_a_harmless_noop():
    pmr = problem_manager()
    prob = _dummy_problem("x")
    pmr.register_problem(prob)
    pmr.register_problem(prob)
    assert pmr.find_problem("x") is prob


# ---------------------------------------------------------------------------
# The nine unreachable table-problems.cc stubs
# ---------------------------------------------------------------------------

# (name, description) copied verbatim from table-problems.h's own
# name()/description() overrides -- see problem.py's
# `_unported_table_problems` docstring for the exact citations.
_EXPECTED_STUBS = {
    "ip": "Find interesting patterns",
    "ann-it": "ANN-based regression on input table",
    "pre": "Precision-Activation scoring",
    "pre-conj": "Precision-Conjunction-Maximization",
    "prerec": "Precision Maximization (holding recall constant)",
    "recall": "Recall Maximization (holding precision constant)",
    "bep": "Maximize Break-even Point",
    "f_one": "Maximize F_1 score",
    "select": "Select range of rows",
    "cluster": "Discover clustering function",
}


def test_unported_table_problems_is_non_vacuous():
    """CARRY-FORWARD.md Sec 5/6: an empty corpus a parametrize walks over
    passes silently. Pin the count directly, independent of `parametrize`.
    """
    assert len(_unported_table_problems()) == 10 == len(_EXPECTED_STUBS)


def test_unported_table_problems_names_and_descriptions_match_classic():
    stubs = _unported_table_problems()
    assert {p.name() for p in stubs} == set(_EXPECTED_STUBS)
    for prob in stubs:
        assert prob.description() == _EXPECTED_STUBS[prob.name()]


def test_unported_table_problems_excludes_it():
    """The clean seam S13 Task 2 fills in: `it_table_problem` is the one
    reachable problem and is deliberately never named here.
    """
    assert "it" not in {p.name() for p in _unported_table_problems()}


@pytest.mark.parametrize("name", sorted(_EXPECTED_STUBS))
def test_unported_table_problem_run_raises_not_implemented(name):
    """MUTATION PROVEN: replacing `_unported_table_problem.run`'s body
    with `pass` (a stub that silently does nothing, exactly the failure
    mode this task's brief calls out: "-H recall must fail loudly rather
    than ... acting as a no-op") makes every one of these ten parametrized
    cases fail (no exception raised); restoring the `raise
    NotImplementedError(...)` makes all ten pass again. See
    task-0-report.md.
    """
    prob = next(p for p in _unported_table_problems() if p.name() == name)
    with pytest.raises(NotImplementedError) as excinfo:
        prob.run(None)
    assert name in str(excinfo.value)


def test_unported_table_problem_run_raises_regardless_of_argument():
    prob = _unported_table_problem("x", "y", "table-problems.h:1 x_problem")
    with pytest.raises(NotImplementedError):
        prob.run(None)
    with pytest.raises(NotImplementedError):
        prob.run(object())


def test_register_unported_table_problems_registers_all_ten_and_none_run():
    pmr = problem_manager()
    _register_unported_table_problems(pmr)
    for name, description in _EXPECTED_STUBS.items():
        found = pmr.find_problem(name)
        assert found is not None, f"{name!r} was not registered"
        assert found.description() == description
        with pytest.raises(NotImplementedError):
            found.run(None)
    assert pmr.find_problem("it") is None
    assert pmr.find_problem("totally-bogus-name") is None


def test_register_unported_table_problems_is_idempotent_shape():
    """Calling it twice on two independent managers must not make the
    second call see the first manager's registrations (no shared mutable
    state leaking between `problem_manager` instances via the stub
    factory).
    """
    pmr_a = problem_manager()
    pmr_b = problem_manager()
    _register_unported_table_problems(pmr_a)
    _register_unported_table_problems(pmr_b)
    assert pmr_a.find_problem("pre") is not pmr_b.find_problem("pre")
