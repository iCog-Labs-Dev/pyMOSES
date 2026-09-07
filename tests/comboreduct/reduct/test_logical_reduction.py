"""Tests for pymetta/comboreduct/reduct/logical_reduction.py -- the
per-effort-level rule-set assembly (0 "extra_simple" .. 3 "complexe").

C++: moses/comboreduct/reduct/logical_reduction.cc.

Every literal expected value below is copied verbatim from a REAL run of
`tools/cpp_probe/logical_rules_probe.cc` against $CLASSIC's compiled
`libcomboreduct.so` (bounded: `( ulimit -v 2097152; timeout 300 tools/
cpp_probe/build/logical_rules_probe )`), not hand-derived -- including
the full 6-tree `BooleanReductUTest` corpus, AT BOTH EFFORT 2 AND EFFORT
3 (via `boolean_reduct_medium_typed_N`/`boolean_reduct_effort3_typed_N`,
which the probe computes via the REAL `logical_reduce(N, tr, ignore_ops)`
and `logical_reduction(ignore_ops)(N)`), not a test-local approximation
the way Task 4's own `_boolean_reduce_medium` had to be (`test_logical_
rules.py`'s own module docstring) before this module existed.

COMPOSITION ORDER IS WHAT IS UNDER TEST. Three specific mutations were
made and MEASURED (not merely reasoned about) to change output before
writing the tests below, confirming each corresponding pinned value is
load-bearing:

* Making the inner `insert_ands()` call UNTYPED (dropping `id::
  boolean_type`) in `p_medium`'s `pre_subtree_to_enf` changes tree 2's
  effort-2 result from the real typed value pinned below
  (`and(or(and(!$3 !$5) !$1) or($2 !$3 $5))`) to the UNTYPED
  approximation (`and(or(!$1 and(!$3 !$5)) or($2 !$3 $5))`) -- this is
  the exact mechanism `logical_rules.py`'s and `meta_rules.py`'s own
  module docstrings describe.
* Dropping the trailing `downwards(remove_unary_junctors(), boolean_
  type)` from `p_medium` changes tree 4's effort-2 result from the
  pinned `$3` to `and($3)`.
* Trees 5 and 6's OWN pinned effort-2 and effort-3 values (both measured
  independently against real C++, see the two lists below) genuinely
  differ from each other -- see `test_effort2_and_effort3_genuinely_
  disagree_on_trees_5_and_6` -- so a `p_complexe` that dropped `reduce_
  remove_subtree_equal_tt()` (making it behave like `p_medium` alone)
  would produce the WRONG, effort-2 value at effort 3 for those two trees
  and fail their effort-3 tests above.

CORRECTED (self-review): an earlier draft of this module's docstring, and
a test now named `test_effort2_and_effort3_genuinely_disagree_on_trees_
5_and_6`, wrongly claimed trees 2 and 6 were the ones that disagree
between effort 2 and effort 3, based on misreading the probe's
`boolean_reduct_effort3_agree_N`/`boolean_reduct_medium_agree_N` fields
as an effort-2-vs-effort-3 comparison -- they are not (they measure
TYPED-vs-UNTYPED agreement at ONE fixed effort). Caught by the test
itself failing on first run, not by inspection; fixed by comparing the
two pinned lists directly instead of re-deriving from a different
measurement.
"""

from __future__ import annotations

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduce, logical_reduction
from pymetta.comboreduct.table.table import complete_truth_table

# ---------------------------------------------------------------------------
# BooleanReductUTest's own 6 trees (logical_rules_probe.cc's own corpus,
# copied verbatim -- same trees test_logical_rules.py's test_simplify1..6
# already carry, repeated here since this module is the one that actually
# reproduces the real logical_reduce/logical_reduction pipeline).
# ---------------------------------------------------------------------------

_TREE_1 = (
    "and(and(not(not(and($4 $3))) or(not(or($5 $4)) or(and($1 $5) and($1 $5)))) "
    "and(and(and(not($1) not($2)) not(not($3))) or(and(not($4) not($5)) "
    "or(or($2 $1) not($5)))))"
)
_TREE_2 = (
    "not(or(and(and(not($5) or($5 $3)) not(or($5 $2))) "
    "and(or(and($5 $3) or($5 $3)) not(not($1)))))"
)
_TREE_3 = (
    "and(or(not(and(or($5 $3) not($2))) or(or(or($1 $2) and($2 $3)) "
    "or(and($2 $4) not($5)))) not(not(and(or($3 $3) or($3 $4)))))"
)
_TREE_4 = (
    "or(and(not($1) $3 or($3 not($1)) or(not($1) and($2 $1) and($3 $2))) "
    "and($3 $1) and(not($3) or($3 $1) not(or($3 not($1))) not(or($2 $1))))"
)
_TREE_5 = "or(and(or(not($1) not($2)) $3) and(or(not($1) $3) not($2)))"
_TREE_6 = "and(or(and(or(not($1) $3) not($2)) and(not($1) $3)) or(not($2) $3))"

_TREES = [_TREE_1, _TREE_2, _TREE_3, _TREE_4, _TREE_5, _TREE_6]

# boolean_reduct_medium_typed_1..6 (real, typed effort 2 "medium").
_EFFORT2_EXPECTED = [
    "false",
    "and(or(and(!$3 !$5) !$1) or($2 !$3 $5))",
    "and(or($1 $2 !$5) $3)",
    "$3",
    "or(and(or(!$1 !$2) $3) and(or(!$1 $3) !$2))",
    "and(or(and(or(!$1 $3) !$2) and(!$1 $3)) or(!$2 $3))",
]

# boolean_reduct_effort3_typed_1..6 (real, typed effort 3 "complexe").
_EFFORT3_EXPECTED = [
    "false",
    "and(or(and(!$3 !$5) !$1) or($2 !$3 $5))",
    "and(or($1 $2 !$5) $3)",
    "$3",
    "or(and(or(!$1 $3) !$2) and(!$1 $3))",
    "and(or(and(!$2 $3) !$1) or(!$2 $3))",
]


_EFFORT2_CASES = list(zip(_TREES, _EFFORT2_EXPECTED, strict=True))
_EFFORT3_CASES = list(zip(_TREES, _EFFORT3_EXPECTED, strict=True))


@pytest.mark.parametrize(("tree_str", "expected"), _EFFORT2_CASES)
def test_effort2_matches_c_plus_plus(tree_str, expected):
    tr = str_to_combo_tree(tree_str)
    logical_reduce(2, tr, ignore_ops=set())
    assert combo_tree_to_str(tr).strip() == expected


def test_three_argument_overload_treats_third_positional_argument_as_ignore_ops():
    positional = str_to_combo_tree(_TREE_2)
    keyword = str_to_combo_tree(_TREE_2)

    logical_reduce(2, positional, set())
    logical_reduce(2, keyword, ignore_ops=set())

    assert combo_tree_to_str(positional) == combo_tree_to_str(keyword)


@pytest.mark.parametrize(("tree_str", "expected"), _EFFORT3_CASES)
def test_effort3_matches_c_plus_plus(tree_str, expected):
    tr = str_to_combo_tree(tree_str)
    logical_reduce(3, tr, ignore_ops=set())
    assert combo_tree_to_str(tr).strip() == expected


def test_effort2_and_effort3_genuinely_disagree_on_trees_5_and_6():
    """MEASURED directly from the two pinned lists above (both
    independently verified against real C++): trees 1, 2, 3, 4 give the
    SAME string at effort 2 and effort 3; trees 5 and 6 do NOT. The only
    compositional difference between `p_medium` (effort 2) and
    `p_complexe` (effort 3) is the outer `iterative` re-entering `p_medium`
    plus `reduce_remove_subtree_equal_tt` -- so this is exactly the load-
    bearing evidence that `reduce_remove_subtree_equal_tt` is wired into
    `p_complexe` for real, not a no-op: a `p_complexe` that dropped it
    (degenerating to plain `p_medium`) would give the WRONG, effort-2
    value at effort 3 for trees 5 and 6 and fail the effort-3 tests
    above.

    CORRECTED (self-review): an earlier draft of this test predicted
    trees 2 and 6 based on a MISREADING of the probe's own `boolean_
    reduct_effort3_agree_N`/`boolean_reduct_medium_agree_N` fields --
    those measure TYPED-vs-UNTYPED agreement AT a single fixed effort,
    not effort-2-vs-effort-3 equality, and do not answer this test's own
    question at all. This test now compares only the two lists actually
    pinned in this file, directly.
    """
    for i in (0, 1, 2, 3):  # trees 1, 2, 3, 4 (0-indexed)
        assert _EFFORT2_EXPECTED[i] == _EFFORT3_EXPECTED[i]
    for i in (4, 5):  # trees 5, 6
        assert _EFFORT2_EXPECTED[i] != _EFFORT3_EXPECTED[i]


# ---------------------------------------------------------------------------
# Effort 0 ("extra_simple") and effort 1 ("simple") -- usable via the
# DEFAULT constructor (no ignore_ops needed). Each case below was picked
# to isolate exactly one wrapped rule inside the composition; see this
# module's own docstring and logical_reduction.py's for the measured
# mutation evidence at effort 2 (the same discipline applied here).
# ---------------------------------------------------------------------------


def test_effort0_double_negation():
    tr = str_to_combo_tree("not(not($1))")
    logical_reduce(0, tr)
    assert combo_tree_to_str(tr).strip() == "$1"


def test_effort0_reduce_nots_pushes_negation_through_or_demorgan():
    """`reduce_nots` (the ONLY rule effort 0 wraps) pushes negation
    through `or` (De Morgan), not merely `not(not(x))->x` -- MEASURED
    against C++ (`logical_reduction_effort0_reduce_nots_demorgan`).
    """
    tr = str_to_combo_tree("and(not(or($1 $2)) $3)")
    logical_reduce(0, tr)
    assert combo_tree_to_str(tr).strip() == "and(and(!$1 !$2) $3)"


def test_effort0_is_typed_boolean_only_and_skips_no_reachable_leaves():
    """`downwards(reduce_nots(), boolean_type)` -- typed. `reduce_nots`
    only ever fires when `it`'s own vertex is `logical_not`, which
    always resolves to `boolean_type` (`get_output_type_tree`'s own
    per-builtin table), so typed vs untyped dispatch is UNOBSERVABLE for
    this specific rule/effort combination -- ported as typed to match
    `logical_reduction.cc:56` verbatim, not because a test can currently
    distinguish it from the untyped form.
    """
    tr = str_to_combo_tree("not($1)")
    logical_reduce(0, tr)
    assert combo_tree_to_str(tr).strip() == "!$1"


def test_effort1_eval_logical_identities_and_true():
    """MEASURED (`logical_reduction_effort1_and_true`): `and(true X)->X`
    needs `upwards(eval_logical_identities())`, the FIRST rule inside
    effort 1's inner `iterative`.
    """
    tr = str_to_combo_tree("and(true $1)")
    logical_reduce(1, tr)
    assert combo_tree_to_str(tr).strip() == "and($1)"


def test_effort1_level_flattens_nested_associative_operators():
    """MEASURED (`logical_reduction_effort1_level_flattens`):
    `and(and($1 $2) $3)` -> `and($1 $2 $3)` needs `downwards(level())` --
    without it the nesting survives (verified directly against a
    same-shaped composition with that one step removed: stays
    `and(and($1 $2) $3)`).
    """
    tr = str_to_combo_tree("and(and($1 $2) $3)")
    logical_reduce(1, tr)
    assert combo_tree_to_str(tr).strip() == "and($1 $2 $3)"


def test_effort1_reduce_ors_subset_absorption():
    """MEASURED (`logical_reduction_effort1_subset_absorption`):
    `or(and($1 $2) and($1 $2 $3))` -> `and($1 $2)` (the superset clause
    is absorbed) needs `downwards(reduce_ors(), boolean_type)`.
    """
    tr = str_to_combo_tree("or(and($1 $2) and($1 $2 $3))")
    logical_reduce(1, tr)
    assert combo_tree_to_str(tr).strip() == "and($1 $2)"


def test_effort1_does_not_run_insert_ands_or_subtree_to_enf():
    """MEASURED (`logical_reduction_effort1_no_insert_ands`): unlike
    effort 2/3, effort 1's composition never wraps a bare argument or
    `or` node under an `and` -- `insert_ands`/`subtree_to_enf` are NOT
    part of `p_simple` at all (`logical_reduction.cc:60-67`).
    """
    tr = str_to_combo_tree("or($1 $2)")
    logical_reduce(1, tr)
    assert combo_tree_to_str(tr).strip() == "or($1 $2)"


def test_effort1_composition_order_and_fixpoint_are_load_bearing():
    """FIX ROUND: every effort-1 test above uses a one-operator tree
    (a single `and`, a single `or`, ...), which cannot distinguish
    `p_simple`'s ACTUAL composition (`downwards(reduce_nots()) ->
    iterative(sequential(eval_logical_identities, level, reduce_ands,
    reduce_ors))`, logical_reduction.cc:60-65) from two mutants: swapping
    `reduce_ands`/`reduce_ors`'s order inside that inner `sequential`, or
    dropping the outer `iterative` (running the inner sequence only ONCE
    instead of to a fixpoint). Both mutations were APPLIED to this port
    directly and re-run against the four tests above: neither changes
    any of their four pinned values, so both mutants silently survived
    until now.

    Both expected values below are MEASURED against real C++
    (`( ulimit -v 2097152; timeout 300 <probe> )`, `logical_reduce(1, tr,
    ignore_ops=set())`) and this port's own CORRECT (unmutated) output
    matches both exactly. Each case was then independently re-run against
    BOTH mutants above (applied one at a time to this port) to confirm
    which one it actually kills -- a test asserting a "should distinguish
    this" value without checking is exactly the unverified-claim mistake
    this fix round exists to correct elsewhere in this file's own module:

    * `and(not(not($2)) or(or($2 $1) not($1)))` -> `$2`. Kills the
      ORDER-SWAP mutant (measured: swapped gives `and($2)` -- the
      `or($2 $1 !$1)` tautology is never collapsed to `true`, so
      `and($2 true)` never reduces its own `true` argument away).
      Measured NOT to kill the DROP-`iterative` mutant (still gives the
      correct `$2` with only one pass of the inner sequence).
    * `or($1 false)` -> `$1`. Kills the DROP-`iterative` mutant
      (measured: one pass gives `or($1)` -- `reduce_ors` drops the
      `false` argument on that first pass, but nothing in `p_simple`'s
      rule set unwraps the resulting single-child `or` without a SECOND
      pass through `level`/`eval_logical_identities`). Measured NOT to
      kill the ORDER-SWAP mutant (still gives the correct `$1`: swapping
      which of `reduce_ands`/`reduce_ors` runs first each pass does not
      change how many passes this particular input needs).

    Together the two cases cover both mutations; neither alone would.
    """
    tr1 = str_to_combo_tree("and(not(not($2)) or(or($2 $1) not($1)))")
    logical_reduce(1, tr1)
    assert combo_tree_to_str(tr1).strip() == "$2"

    tr2 = str_to_combo_tree("or($1 false)")
    logical_reduce(1, tr2)
    assert combo_tree_to_str(tr2).strip() == "$1"


# ---------------------------------------------------------------------------
# logical_reduction's own construction/dispatch contract
# ---------------------------------------------------------------------------


def test_p_extra_simple_and_p_simple_are_shared_across_instances():
    """`p_extra_simple`/`p_simple` are C++ CLASS-level `static rule*`
    members (reduct.h:90-91) -- built once, shared by every instance.
    Pins the STATIC/shared-across-instances contract itself (matching
    C++'s `static` keyword), not a behavioural necessity: nothing in
    either rule mutates any state, so a non-shared rebuild would behave
    identically. Still a real structural fact worth locking in.
    """
    r1 = logical_reduction()
    r2 = logical_reduction()
    assert r1(0) is r2(0)
    assert r1(1) is r2(1)


def test_p_medium_and_p_complexe_are_per_instance_not_shared():
    """Unlike `p_extra_simple`/`p_simple`, `p_medium`/`p_complexe` are
    C++ per-INSTANCE members (reduct.h:87-88, private, non-static) --
    built fresh (via `new`) by each constructor call that supplies
    `ignore_ops`. Two separate `logical_reduction(set())` instances must
    NOT share the same `p_medium`/`p_complexe` object.
    """
    r1 = logical_reduction(set())
    r2 = logical_reduction(set())
    assert r1(2) is not r2(2)
    assert r1(3) is not r2(3)


def test_default_constructor_leaves_medium_and_complexe_unusable():
    """C++'s default constructor (reduct.h:78) leaves `p_medium`/
    `p_complexe` `NULL` -- only effort 0/1 are usable. Effort 2 or 3
    without `ignore_ops` hits the SAME `OC_ASSERT(p_medium, ...)`
    (reduct.h's `operator()`) regardless of which of the two it is --
    ported as `AssertionError`, matching the OC_ASSERT convention this
    whole port uses (argument.py's `__init__`).
    """
    r = logical_reduction()
    with pytest.raises(AssertionError, match="requires ignore_ops"):
        r(2)
    with pytest.raises(AssertionError, match="requires ignore_ops"):
        r(3)


def test_an_invalid_effort_without_ignore_ops_still_reports_missing_ignore_ops():
    """A REAL, faithfully-ported upstream quirk, not a bug introduced
    here: C++'s `operator()` checks `OC_ASSERT(p_medium, ...)` for EVERY
    effort other than 0/1 BEFORE checking whether that effort is
    otherwise valid (reduct.h's own `switch`/`OC_ASSERT`/`switch`
    sequence) -- so an invalid effort (e.g. 4) with no `ignore_ops`
    reports "requires ignore_ops", not "no such effort".
    """
    r = logical_reduction()
    with pytest.raises(AssertionError, match="requires ignore_ops"):
        r(4)


def test_an_invalid_effort_with_ignore_ops_reports_no_such_effort():
    r = logical_reduction(set())
    with pytest.raises(AssertionError, match="no such effort"):
        r(4)
    with pytest.raises(AssertionError, match="no such effort"):
        r(-1)


def test_ignore_ops_is_none_vs_empty_set_are_different_constructors():
    """`ignore_ops is None` (the Python "omitted" sentinel) takes C++'s
    DEFAULT constructor (`p_medium` stays unbuilt); `ignore_ops=set()`
    (an explicit, merely EMPTY `vertex_set`) takes the OTHER constructor
    and DOES build `p_medium`/`p_complexe` -- these are not the same
    thing, matching `BooleanReductUTest`'s own real `vertex_set
    ignore_ops;` (default-constructed, empty, but a REAL vertex_set).
    """
    assert logical_reduction().p_medium is None
    assert logical_reduction(set()).p_medium is not None


# ---------------------------------------------------------------------------
# logical_reduce -- the free-function convenience wrapper (reduct.h:116-135)
# ---------------------------------------------------------------------------


def test_logical_reduce_it_omitted_applies_at_the_root():
    tr = str_to_combo_tree("not(not($1))")
    logical_reduce(0, tr)
    assert combo_tree_to_str(tr).strip() == "$1"


def test_logical_reduce_explicit_it_applies_at_that_node_only():
    tr = str_to_combo_tree("and(not(not($1)) $2)")
    it = tr.begin()
    inner_not = it.first_child  # not(not($1))
    logical_reduce(0, tr, inner_not, set())
    assert combo_tree_to_str(tr).strip() == "and($1 $2)"


def test_logical_reduce_explicit_none_it_raises():
    tr = str_to_combo_tree("$1")
    with pytest.raises(ValueError):
        logical_reduce(0, tr, None)


# ---------------------------------------------------------------------------
# BooleanReductUTest.cxxtest -- a faithful port, MOVED here from Task 4's
# test_logical_rules.py once the real pipeline it needs (this file) landed.
# Same 6 trees, same test names (test_simplify1..6), same TWO assertions
# the real cxxtest makes (test_simplify_tree_str, BooleanReductUTest.
# cxxtest:20-36): strict size decrease, and truth-table preservation --
# via `complete_truth_table`, now genuinely ported (table.py), not a
# `boolean_interpreter`-based stand-in. Calls the REAL `logical_reduce(3,
# tr, ignore_ops)`, matching the cxxtest's own call exactly -- not an
# approximation of it.
# ---------------------------------------------------------------------------


def _test_simplify_tree_str(tree_str: str) -> None:
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:20
    test_simplify_tree_str -- the real cxxtest helper every test_simplifyN
    below calls, ported line-for-line (parse, snapshot the complete truth
    table and size, reduce at effort 3, re-measure, assert both)."""
    tr = str_to_combo_tree(tree_str)
    tt1 = complete_truth_table(tr)
    size_before = tr.size()
    logical_reduce(3, tr, ignore_ops=set())
    size_after = tr.size()
    tt2 = complete_truth_table(tr, arity=_integer_log2(len(tt1)))

    assert size_after < size_before
    assert list(tt1) == list(tt2)


def _integer_log2(n: int) -> int:
    """No direct C++ symbol of its own -- a narrow stand-in for
    `opencog::integer_log2` (the real cxxtest's own `integer_log2(tt1.
    size())`, opencog/util/numeric.h), scoped to exactly this call's own
    shape: `n` is always an exact power of two here (`complete_truth_
    table`'s own `2**arity` sizing), so a plain bit-length count suffices.
    """
    return n.bit_length() - 1


def test_simplify1():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:39 test_simplify1"""
    _test_simplify_tree_str(_TREE_1)


def test_simplify2():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:42 test_simplify2"""
    _test_simplify_tree_str(_TREE_2)


def test_simplify3():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:45 test_simplify3"""
    _test_simplify_tree_str(_TREE_3)


def test_simplify4():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:48 test_simplify4"""
    _test_simplify_tree_str(_TREE_4)


def test_simplify5():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:51 test_simplify5"""
    _test_simplify_tree_str(_TREE_5)


def test_simplify6():
    """$CLASSIC: tests/comboreduct/reduct/BooleanReductUTest.cxxtest:54 test_simplify6"""
    _test_simplify_tree_str(_TREE_6)
