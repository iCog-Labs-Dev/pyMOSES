"""Unit tests for pymetta.moses.eda.termination.

No C++ probe: `terminate_if_gte_or_no_improv` is exercised end-to-end by
`test_univariate.py`'s own oracle-shaped tests, but its edge behaviour
(post-increment order, the `best_seen` default-construction quirk) is
easiest to pin directly against hand-built `scored_instance` populations.
"""

from pymetta.moses.eda.termination import terminate_if_gte, terminate_if_gte_or_no_improv
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance


def _si(score: float) -> scored_instance:
    return scored_instance(instance(0), composite_score(score))


# --------------------------------------------------------------- terminate_if_gte


def test_terminate_if_gte_true_when_best_meets_bound():
    term = terminate_if_gte(composite_score(5.0))
    assert term([_si(1.0), _si(5.0), _si(2.0)]) is True


def test_terminate_if_gte_false_when_best_below_bound():
    term = terminate_if_gte(composite_score(5.0))
    assert term([_si(1.0), _si(4.9), _si(2.0)]) is False


# ------------------------------------------------------- terminate_if_gte_or_no_improv


def test_first_call_always_treated_as_improvement_over_worst_default():
    """`best_seen` is default-constructed to the WORST composite_score
    (C++ leaves it default-constructed, not garbage -- see the class
    docstring). Any real score on generation 0 must therefore be judged
    an "improvement", taking the `res >= bound` branch, regardless of
    `at == -1` too.
    """
    term = terminate_if_gte_or_no_improv(composite_score(100.0), n=3)
    # best score (-1.0) is far below bound (100.0) -> not terminated,
    # but `at` must have been reset to 0 (improvement path taken).
    assert term([_si(-1.0)]) is False
    assert term.at == 0
    assert term.best_seen == composite_score(-1.0)


def test_terminates_when_bound_reached():
    term = terminate_if_gte_or_no_improv(composite_score(5.0), n=3)
    assert term([_si(5.0)]) is True


def test_no_improvement_counter_uses_post_increment_order():
    """`at++ > n_gen`: the comparison reads `at` BEFORE incrementing.
    With `n_gen=2`, three consecutive non-improving calls should see
    `at` compared as 0, 1, 2 (not-terminated for the first two, since
    `0 > 2` and `1 > 2` are both false) and TERMINATE on the third
    (`2 > 2` is false too -- termination requires `at > n_gen` strictly,
    so the FOURTH non-improving call, comparing `3 > 2`, is the one that
    terminates).
    """
    term = terminate_if_gte_or_no_improv(composite_score(1000.0), n=2)
    term([_si(0.0)])  # generation 0: improvement path, at=0, best_seen=0.0
    assert term.at == 0
    assert term([_si(-5.0)]) is False  # at compared as 0 -> at becomes 1
    assert term.at == 1
    assert term([_si(-5.0)]) is False  # at compared as 1 -> at becomes 2
    assert term.at == 2
    assert term([_si(-5.0)]) is False  # at compared as 2 (2 > 2 is False) -> at becomes 3
    assert term.at == 3
    assert term([_si(-5.0)]) is True  # at compared as 3 (3 > 2 is True) -> terminate


def test_mutation_dropping_best_seen_default_breaks_generation_zero():
    """Mutation proof, performed directly (see task report for the
    source-level red/green cycle): with `best_seen` left as Python's
    bare, un-default-constructed `None` (the mutant this test detects --
    i.e. a hypothetical port that forgot C++'s implicit class-member
    default-construction, module docstring), `res > self.best_seen`
    recurses infinitely through `composite_score.__gt__`'s `other < self`
    (`None.__lt__` is NotImplemented, so Python retries via the reflected
    `self.__gt__(None)`, calling right back into the same method) rather
    than cleanly comparing against the worst score. The faithful port
    (`self.best_seen = composite_score()`, tested above) never reaches
    this at all.
    """
    term = terminate_if_gte_or_no_improv(composite_score(100.0), n=3)
    term.best_seen = None  # simulate the broken "no default" mutant
    try:
        term([_si(-1.0)])
    except RecursionError:
        return
    raise AssertionError("expected the mutant to recurse comparing against None")
