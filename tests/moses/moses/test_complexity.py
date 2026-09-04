"""C++: moses/moses/moses/complexity.cc

No `complexityUTest.cxxtest` exists in classic to transliterate (checked:
`$CLASSIC/tests/moses/` has no such file) -- this module is therefore
direct unit tests of `tree_complexity`'s individual branches (hand-verified
against complexity.cc's own algorithm, reproduced in complexity.py's
docstring) plus a differential oracle against every case in
tests/fixtures/score.json (captured from tools/cpp_probe/score_probe.cc,
which calls `bscore_base::get_complexity` -- itself just `tree_complexity`,
scoring_base.py's own docstring -- over every tree in the corpus).
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.moses.moses.complexity import tree_complexity

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_SCORE_CASES = json.loads((_FIXTURE_DIR / "score.json").read_text())

# One (tree, expected complexity) pair per distinct tree in score.json,
# de-duplicated (tree_complexity depends only on the tree, so every
# dataset that scores the same tree records the same value -- asserted
# below, not just assumed).
_TREE_COMPLEXITIES = {case["tree"]: case["tree_complexity"] for case in _SCORE_CASES.values()}


def test_tree_complexity_is_dataset_independent_in_the_oracle_fixture():
    """Sanity check on the fixture itself: the same tree string must carry
    the same tree_complexity in every case it appears in, regardless of
    dataset -- tree_complexity (complexity.cc:113) takes only a
    combo_tree, never a table.
    """
    by_tree: dict[str, set[int]] = {}
    for case in _SCORE_CASES.values():
        by_tree.setdefault(case["tree"], set()).add(case["tree_complexity"])
    for tree, values in by_tree.items():
        assert len(values) == 1, f"{tree!r}: inconsistent tree_complexity {values}"


@pytest.mark.oracle
@pytest.mark.parametrize("tree", sorted(_TREE_COMPLEXITIES), ids=sorted(_TREE_COMPLEXITIES))
def test_tree_complexity_matches_cpp_probe(tree):
    tr = str_to_combo_tree(tree)
    assert tree_complexity(tr) == _TREE_COMPLEXITIES[tree]


# ---------------------------------------------------------------------------
# Direct unit tests of branches the boolean-only oracle corpus above never
# reaches (no contin/div/sin/rand/equ/cond appears in a purely boolean
# combo tree, and no bare true/false leaf appears in the Task 3 corpus
# either) -- hand-verified against complexity.cc's own algorithm.
# ---------------------------------------------------------------------------


def test_tree_complexity_of_bare_boolean_constant_is_zero():
    """complexity.cc:62-65: `*it == id::logical_true || ... logical_false
    || ... null_vertex` is the first base case, returning 0.
    """
    assert tree_complexity(str_to_combo_tree("true")) == 0
    assert tree_complexity(str_to_combo_tree("false")) == 0


def test_tree_complexity_of_bare_argument_is_one():
    """complexity.cc:81-86: is_argument(*it) returns 1."""
    assert tree_complexity(str_to_combo_tree("$1")) == 1


def test_tree_complexity_of_div_counts_one_plus_children():
    """complexity.cc:100-106: `div` is one of the operators costing 1
    (`int(*it==id::div || ...)`), on top of which its children's own
    complexities are summed (complexity.cc:108-109) -- 1 (div itself) + 1
    ($1) + 1 ($2) = 3.
    """
    assert tree_complexity(str_to_combo_tree("div($1 $2)")) == 3


def test_tree_complexity_of_sin_counts_one_plus_child():
    """Same cost-one operator list as div, above; 1 (sin) + 1 ($1) = 2."""
    assert tree_complexity(str_to_combo_tree("sin($1)")) == 2


def test_tree_complexity_of_and_or_is_zero_plus_children():
    """complexity.cc:92-95's own comment: "Note that logical_and,
    logical_or are not listed, these have a complexity of zero." -- 0
    (and itself) + 1 ($1) + 1 ($2) = 2.
    """
    assert tree_complexity(str_to_combo_tree("and($1 $2)")) == 2


def test_tree_complexity_of_logical_not_delegates_to_its_child():
    """complexity.cc:89-90: `logical_not` recurses directly into its one
    child rather than falling into the general "cost + sum of children"
    path -- so `not($1)` costs exactly what `$1` alone costs (1), not
    1 + 1.
    """
    assert tree_complexity(str_to_combo_tree("not($1)")) == 1


def test_tree_complexity_stopper_halts_recursion_immediately():
    """complexity.cc:67-70: "If the stopper function is defined, and it
    returns true, we halt recursion." -- checked before every other case
    except the bare-boolean-constant one, so a stopper that always
    returns True makes even a non-trivial tree cost 0.
    """
    tr = str_to_combo_tree("div($1 $2)")
    assert tree_complexity(tr, stopper=lambda it: True) == 0


# ---------------------------------------------------------------------------
# Regression tests for the final review's Critical 1: `v == builtin.X`
# comparisons in tree_complexity's base cases ran with no preceding
# `isinstance(v, builtin)` guard. Since `contin_t` is a bare Python `float`
# and `builtin` an `IntEnum`, a contin whose value equals a builtin's
# ordinal (null_vertex=0, logical_true=1, logical_false=2, logical_and=3,
# logical_or=4, logical_not=5, times=7 -- vertex.py) was silently
# misrouted into the wrong branch. C++'s `boost::variant` alternatives can
# never compare equal across types, so classic cannot exhibit this at all
# -- verified directly against the real C++ (a bounded g++ probe linking
# libcomboreduct/libmoses/libcogutil, this fix wave's own final-fix-wave
# report): `tree_complexity("0")` is 1 in C++ (is_contin's branch), and
# `tree_complexity("plus(0 $1)")` is 2 (1 for the 0 contin child + 1 for
# the $1 argument child, "plus" itself costing 0) -- both reproduced
# exactly below.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contin_literal",
    ["0", "1", "2", "3", "4", "5", "7"],
    ids=[
        "0-collides-with-null_vertex",
        "1-collides-with-logical_true",
        "2-collides-with-logical_false",
        "3-collides-with-logical_and",
        "4-collides-with-logical_or",
        "5-collides-with-logical_not",
        "7-collides-with-times",
    ],
)
def test_tree_complexity_of_bare_contin_is_one_even_when_it_collides_with_a_builtin_ordinal(
    contin_literal,
):
    """Critical 1 regression. Before the fix, `tree_complexity("0")`
    returned 0 (wrongly hitting the null_vertex base case) instead of the
    correct 1 (the is_argument/is_contin base case, complexity.cc:78-86)
    -- verified against real C++ via a bounded probe: `tree_complexity
    ("0")` is 1 there, never 0.
    """
    assert tree_complexity(str_to_combo_tree(contin_literal)) == 1


def test_tree_complexity_of_plus_zero_and_argument_matches_cpp():
    """Critical 1 regression, the exact case the final review verified
    against real C++: `tree_complexity("plus(0 $1)")` is 2 in C++ (0 for
    `plus` itself + 1 for the `0` contin child + 1 for the `$1` argument
    child), not 1 (what the pre-fix port returned, because the `0` child
    was misrouted into the null_vertex base case and contributed 0 instead
    of 1).
    """
    assert tree_complexity(str_to_combo_tree("plus(0 $1)")) == 2


def test_tree_complexity_of_times_zero_knob_marker_is_still_zero():
    """The `*(0 stuff)` contin-knob-marker base case (complexity.cc:72-76)
    must still short-circuit to 0 after the isinstance guard was added --
    this is the one case where `v == builtin.times` is a real (non-contin)
    builtin comparison and must still succeed.
    """
    assert tree_complexity(str_to_combo_tree("times(0 $1)")) == 0
