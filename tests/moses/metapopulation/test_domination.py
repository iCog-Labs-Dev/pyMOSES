"""Unit + oracle tests for pymoses.moses.metapopulation.domination.

`metapopulationUTest.cxxtest` (`tests/moses/metapopulationUTest.cxxtest`
in `$CLASSIC`) is THE real oracle for this whole module -- its two tests,
`test_remove_dominated` (line 74) and `test_merge_nondominated_rec` (line
119), exercise exactly `remove_dominated` and `get_nondominated_disjoint`,
the two methods this file ports. Both are reproduced here verbatim (same
5-tree / 3-tree fixtures, same assertions), calling the SAME entry points
with the SAME default `jobs` classic's own test calls with (neither test
passes a `jobs` argument at all -- classic's own default, `metapop_
parameters::jobs == 1`, `metapop_params.h:235`, is what both the C++ test
and this file exercise; nothing here configures `jobs` itself). Built and
run directly to confirm:

    ( ulimit -v 2097152; timeout 300 make -C $CLASSIC/build metapopulationUTest )
    ( ulimit -v 2097152; timeout 300 $CLASSIC/build/tests/moses/metapopulationUTest )
    -> "Running cxxtest tests (2 tests)..OK!"

`get_nondominated_iter` has no oracle (zero callers anywhere in classic --
domination.py's own module docstring) and is verified instead against a
throwaway, standalone C++ reproduction of its exact control flow (a
`std::list<int>`-based rebuild of the same nested loop, kept out of this
repo, run once to observe the erase-then-double-`++it1` skip -- see
domination.py's own DO NOT FIX comment on that function for the full
trace). `test_get_nondominated_iter_skips_element_after_erasing_it1`
below reproduces that exact case.
"""

from __future__ import annotations

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.moses.metapopulation import merging  # noqa: F401  (binds merge_candidates et al.)
from pymoses.moses.metapopulation.domination import dominates, get_nondominated_iter
from pymoses.moses.metapopulation.metapop_params import metapop_parameters
from pymoses.moses.metapopulation.metapopulation import metapopulation
from pymoses.moses.moses.types import composite_score, scored_combo_tree, scored_combo_tree_set
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.scoring_base import bscore_base

# ---------------------------------------------------------------------------
# dominates() -- direct unit tests on the tribool logic itself
# ---------------------------------------------------------------------------


def test_dominates_strictly_greater_everywhere():
    assert dominates([2.0, 3.0], [1.0, 1.0]) is True
    assert dominates([1.0, 1.0], [2.0, 3.0]) is False


def test_dominates_equal_everywhere_is_indeterminate():
    """ "there exists i such that x_i > y_i" (domination.cc:64) -- equal
    everywhere fails that existence requirement in EITHER direction."""
    assert dominates([1.0, 2.0], [1.0, 2.0]) is None


def test_dominates_mixed_signs_is_indeterminate():
    assert dominates([2.0, -1.0], [1.0, 5.0]) is None


def test_dominates_ge_and_gt_somewhere_dominates():
    """Componentwise >=, with a strict > in at least one place -- the
    literal definition (domination.cc:62-64), not just the "everywhere
    strictly greater" special case test_dominates_strictly_greater_
    everywhere already covers."""
    assert dominates([1.0, 5.0], [1.0, 2.0]) is True
    assert dominates([1.0, 2.0], [1.0, 5.0]) is False


def test_dominates_empty_vs_empty_is_indeterminate():
    assert dominates([], []) is None


def test_dominates_nonempty_dominates_empty():
    """ "everything dominates an empty vector" (domination.cc:73's own
    comment) -- a NON-empty x against an empty y is True regardless of
    x's own contents (even all-zero/negative)."""
    assert dominates([-5.0, 0.0], []) is True
    assert dominates([], [-5.0, 0.0]) is False


def test_dominates_result_is_never_conflated_with_falsy_none():
    """Guards the "type before value" discipline domination.py's own
    module docstring states: None and False are both falsy in Python, so
    a bare `if dom:` / `if not dom:` would silently treat indeterminate as
    false. `is True`/`is False`/`is None` must be able to tell all three
    apart -- this pins that the three cases really are three DISTINCT
    Python objects, not just "the assertions above happened to pass"."""
    indeterminate = dominates([1.0, 2.0], [1.0, 2.0])
    false_case = dominates([1.0, 1.0], [2.0, 3.0])
    assert indeterminate is None
    assert false_case is False
    assert indeterminate is not False
    assert false_case is not None


# ---------------------------------------------------------------------------
# metapopulationUTest.cxxtest's own two fixtures, reproduced verbatim
# ---------------------------------------------------------------------------


def _csc(score: float, complexity: int, complexity_penalty: float = 0.0) -> composite_score:
    return composite_score(score, complexity, complexity_penalty)


def _sct(label: str, cs: composite_score, bs: list[float]) -> scored_combo_tree:
    # combo_tree content is irrelevant to domination (only bscore matters,
    # exactly as metapopulationUTest.cxxtest's own comment says: "the
    # combo_tree themselves are dummy (1, 2, ...) as only their bscore
    # matter") -- distinct `$N` argument leaves stand in for classic's
    # distinct contin-constant dummy trees (1.0, 2.0, ...), giving each a
    # distinct, easily-labelled identity for the set-membership asserts
    # below.
    return scored_combo_tree(str_to_combo_tree(label), cs=cs, bs=list(bs))


def test_remove_dominated_matches_cxxtest_oracle():
    """metapopulationUTest.cxxtest:74 test_remove_dominated, verbatim."""
    bcs = scored_combo_tree_set()
    bct1 = _sct("$1", _csc(-2, -1), [-1, 0, -1, 0])
    bcs.insert(bct1)
    bct2 = _sct("$2", _csc(-3, -1), [-1, -1, -1, 0])
    bcs.insert(bct2)
    bct3 = _sct("$3", _csc(-2, -1), [-1, 0, -1, 0])
    bcs.insert(bct3)
    bct4 = _sct("$4", _csc(-2, -1), [0, -1, 0, -1])
    bcs.insert(bct4)
    bct5 = _sct("$5", _csc(-3, -1), [0, -1, -1, -1])
    bcs.insert(bct5)

    mp = metapopulation.__new__(metapopulation)  # remove_dominated reads no instance state
    mp.remove_dominated(bcs)  # classic's own default jobs (1) -- not overridden here

    assert len(bcs) == 3  # 2 are removed
    assert bct2 not in bcs
    assert bct5 not in bcs
    assert bct1 in bcs
    assert bct3 in bcs
    assert bct4 in bcs


def test_get_nondominated_disjoint_matches_cxxtest_oracle():
    """metapopulationUTest.cxxtest:119 test_merge_nondominated_rec, verbatim."""
    bcs1 = scored_combo_tree_set()
    bct4 = _sct("$4", _csc(-2, -1), [0, -1, 0, -1])
    bcs1.insert(bct4)

    bcs2 = scored_combo_tree_set()
    bct5 = _sct("$5", _csc(-3, -1), [0, -1, -1, -1])
    bcs2.insert(bct5)
    bct2 = _sct("$2", _csc(-3, -1), [-1, -1, -1, 0])
    bcs2.insert(bct2)

    mp = metapopulation.__new__(metapopulation)
    res1, res2 = mp.get_nondominated_disjoint(bcs1, bcs2)  # classic's own default jobs (1)
    union = list(res1) + list(res2)

    assert len(union) == 2  # 1 is removed
    assert bct5 not in union
    assert any(sct.get_tree() == bct4.get_tree() for sct in union)
    assert any(sct.get_tree() == bct2.get_tree() for sct in union)


def test_to_set_copies_not_aliases():
    """`res.insert(*cnd)` (domination.cc:206) copy-constructs -- `to_set`
    (called from `get_nondominated_disjoint`) must hand back INDEPENDENT
    `scored_combo_tree` objects, not the same ones `bcs1`/`bcs2` hold.
    Mutating the original after the call must not reach the result."""
    original = _sct("$1", _csc(1.0, 0), [5.0, 5.0])
    bcs1 = scored_combo_tree_set([original])
    bcs2 = scored_combo_tree_set()

    mp = metapopulation.__new__(metapopulation)
    res1, _res2 = mp.get_nondominated_disjoint(bcs1, bcs2)
    (copy,) = list(res1)
    assert copy is not original

    original.get_composite_score().set_score(-999.0)
    assert copy.get_score() == 1.0  # unaffected by the mutation above


# ---------------------------------------------------------------------------
# get_nondominated_rec -- a case the two 2-input oracle fixtures above
# cannot reach: an odd-length input forcing more than one level of
# recursive splitting (`split`'s own floor-division halves).
# ---------------------------------------------------------------------------


def test_get_nondominated_rec_recursive_split_case():
    """5 candidates -> split() recurses at least twice. Domination
    verified directly via `dominates` (not hand-derived): B dominates
    {A, E}, C dominates {A, E}, D dominates nobody and is dominated by
    nobody, B/C/D are pairwise indeterminate. Expected survivors:
    {B, C, D} (A and E eliminated), regardless of how `split` partitions
    the 5-element list, since the final SURVIVING SET is a mathematical
    function of the (transitive) dominance relation alone -- see
    domination.py's own module docstring for why the jobs<=1 path is safe
    to be the only one ported."""
    mp = metapopulation.__new__(metapopulation)
    a = _sct("$1", _csc(0, 0), [1.0, 1.0])
    b = _sct("$2", _csc(0, 0), [5.0, 20.0])
    c = _sct("$3", _csc(0, 0), [10.0, 1.0])
    d = _sct("$4", _csc(0, 0), [-1.0, 25.0])
    e = _sct("$5", _csc(0, 0), [0.0, 0.0])
    bcv = [a, b, c, d, e]

    # Sanity on the fixture itself, independent of get_nondominated_rec:
    assert dominates(b.get_bscore(), a.get_bscore()) is True
    assert dominates(c.get_bscore(), a.get_bscore()) is True
    assert dominates(b.get_bscore(), e.get_bscore()) is True
    assert dominates(c.get_bscore(), e.get_bscore()) is True
    for x, y in ((b, c), (b, d), (c, d)):
        assert dominates(x.get_bscore(), y.get_bscore()) is None

    result = mp.get_nondominated_rec(bcv)

    result_trees = {combo_tree_to_str(sct.get_tree()).strip() for sct in result}
    assert result_trees == {"$2", "$3", "$4"}  # B, C, D survive; A, E eliminated


def test_get_nondominated_disjoint_rec_base_case_appends_the_whole_remaining_tail():
    """domination.cc:242, `bcv_res2.insert(bcv_res2.end(), it2, bcv2.end())`:
    once ANYTHING in bcv2 dominates the single bcv1 element, every
    REMAINING bcv2 element (not just the one that triggered the break) is
    appended wholesale, unexamined -- see get_nondominated_disjoint_rec's
    own docstring for why that is provably safe, not merely convenient.
    bcv2 = [P, D, Q, R]: P is indeterminate vs X (kept before the break);
    D is the first to dominate X (triggers the break at index 1); Q and R
    come AFTER D and must survive via the wholesale-tail-append, not by
    being individually re-examined."""
    x = _sct("$1", _csc(0, 0), [5.0, 5.0])
    p = _sct("$2", _csc(0, 0), [1.0, 9.0])
    d = _sct("$3", _csc(0, 0), [10.0, 10.0])
    q = _sct("$4", _csc(0, 0), [-1.0, -1.0])
    r = _sct("$5", _csc(0, 0), [-2.0, -2.0])

    # Sanity on the fixture itself:
    assert dominates(x.get_bscore(), p.get_bscore()) is None
    assert dominates(x.get_bscore(), d.get_bscore()) is False  # D dominates X
    assert dominates(x.get_bscore(), q.get_bscore()) is True  # irrelevant to the break itself
    assert dominates(x.get_bscore(), r.get_bscore()) is True

    mp = metapopulation.__new__(metapopulation)
    res1, res2 = mp.get_nondominated_disjoint_rec([x], [p, d, q, r])

    assert res1 == []  # X is dominated (by D) -- excluded
    assert res2 == [p, d, q, r]  # P kept, then D/Q/R appended wholesale from the break point on


# ---------------------------------------------------------------------------
# get_nondominated_iter -- dead code (zero callers, no C++ oracle), verified
# against a standalone reproduction of the exact same control flow instead.
# ---------------------------------------------------------------------------


def test_get_nondominated_iter_skips_element_after_erasing_it1():
    """DO NOT FIX: see domination.py's own docstring on this function for
    the full trace. A=[1,-1], B=[-1,1] (indeterminate vs A), C=[10,10]
    (dominates A, and WOULD dominate B if ever compared to it). Real
    classic control flow erases A (it1) when compared against C, then
    double-advances `it1` past B without ever comparing B against
    anything again -- B survives even though C actually dominates it.
    A correct skyline would return {C} alone; this pins the ACTUAL
    (buggy) classic behaviour, {B, C}."""
    a = _sct("$1", _csc(0, 0), [1.0, -1.0])
    b = _sct("$2", _csc(0, 0), [-1.0, 1.0])
    c = _sct("$3", _csc(0, 0), [10.0, 10.0])
    bcs = scored_combo_tree_set([a, b, c])

    result = get_nondominated_iter(bcs)

    result_trees = {combo_tree_to_str(sct.get_tree()).strip() for sct in result}
    assert result_trees == {"$2", "$3"}  # B, C -- NOT {"$3"} alone, see docstring
    # Sanity: C really does dominate B (confirms this is a genuine skip,
    # not a case where B legitimately survives on the merits).
    assert dominates(c.get_bscore(), b.get_bscore()) is True


def test_get_nondominated_iter_correct_skyline_when_no_erase_reorders_it1():
    """Contrast case: when domination is discovered via the `it2`-erase
    branch (dominated elements come AFTER, not before), no double-advance
    ever happens, and the function computes the ordinary correct skyline
    -- confirming the previous test's {B, C} result is specifically about
    the it1-erase-then-skip path, not get_nondominated_iter being broken
    in general."""
    a = _sct("$1", _csc(0, 0), [10.0, 10.0])  # dominates both B and C
    b = _sct("$2", _csc(0, 0), [1.0, -1.0])
    c = _sct("$3", _csc(0, 0), [-1.0, 1.0])
    bcs = scored_combo_tree_set([a, b, c])

    result = get_nondominated_iter(bcs)

    result_trees = {combo_tree_to_str(sct.get_tree()).strip() for sct in result}
    assert result_trees == {"$1"}


# ---------------------------------------------------------------------------
# merge_nondominated -- the metapopulation-mutating half (bcs vs _scored_trees,
# not bcs1 vs bcs2), exercised through a real metapopulation instance.
# ---------------------------------------------------------------------------


class _FixedBscore(bscore_base):
    """Not itself a ported C++ symbol -- a controllable per-tree bscore
    table, the same shape test_metapopulation.py's own `_ProbeBscore`
    uses, so domination tests can pick exact bscore vectors instead of
    depending on a real dataset scorer."""

    def __init__(self, scores: dict[str, list[float]]) -> None:
        super().__init__()
        self._scores = scores

    def __call__(self, tr):
        return list(self._scores.get(combo_tree_to_str(tr).strip(), []))

    def best_possible_bscore(self):
        return [0.0]


def test_merge_nondominated_evicts_dominated_pool_member_and_inserts_survivor():
    """Builds a REAL metapopulation (constructor, not `__new__`) with one
    pre-existing pool member ("$1"), then merges a candidate set containing
    a tree that dominates it ("$2") and one that is incomparable ("$3").
    Expected: "$1" evicted, "$2" and "$3" both present afterward."""
    scores = {
        "$1": [0.0, 0.0],
        "$2": [1.0, 1.0],  # dominates $1
        "$3": [-1.0, 1.0],  # incomparable with $1 and $2
    }
    cscorer = behave_cscore(_FixedBscore(scores))
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, metapop_parameters())
    assert {combo_tree_to_str(t).strip() for t in [sct.get_tree() for sct in mp.get_trees()]} == {
        "$1"
    }

    candidates = scored_combo_tree_set()
    candidates.insert(_sct("$2", _csc(1.0, 0), scores["$2"]))
    candidates.insert(_sct("$3", _csc(-1.0, 0), scores["$3"]))

    mp.merge_nondominated(candidates)  # classic's own default jobs (1)

    trees = {combo_tree_to_str(sct.get_tree()).strip() for sct in mp.get_trees()}
    assert trees == {"$2", "$3"}  # $1 evicted, $2 and $3 both survive


# ---------------------------------------------------------------------------
# The gate wiring itself (merging.py), not just domination.py's own math --
# `metapop_parameters()` constructed with NO override at all, pinning
# classic's real default (jobs == 1) is what actually reaches
# merge_nondominated/remove_dominated through this path.
# ---------------------------------------------------------------------------


def test_merge_candidates_discard_dominated_wiring_uses_classics_default_jobs():
    """Rule 3 (an oracle configured by the port cannot detect the port's
    own configuration being wrong): this test never sets `.jobs` on
    `metapop_parameters` at all, so if `merge_candidates`/`merge_
    nondominated`'s wiring ever hardcoded a different jobs value, or read
    the wrong params object, this would still exercise classic's actual
    default rather than a value this test chose."""
    params = metapop_parameters()
    assert params.jobs == 1  # pin classic's own default (metapop_params.h:235) explicitly
    params.discard_dominated = True

    scores = {"$1": [0.0, 0.0], "$2": [1.0, 1.0]}
    cscorer = behave_cscore(_FixedBscore(scores))
    mp = metapopulation(str_to_combo_tree("$1"), cscorer, params)

    candidates = scored_combo_tree_set()
    candidates.insert(_sct("$2", _csc(1.0, 0), scores["$2"]))
    mp.merge_candidates(candidates)  # -> merge_nondominated(candidates, self._params.jobs)

    trees = {combo_tree_to_str(sct.get_tree()).strip() for sct in mp.get_trees()}
    assert trees == {"$2"}  # $1 evicted (dominated by $2)
