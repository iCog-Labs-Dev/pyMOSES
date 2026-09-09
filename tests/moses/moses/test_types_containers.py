"""C++: moses/moses/moses/types.h and types.cc -- S10 Task 0 container plumbing.

Tests `demeID_t`'s move to this module (exactly one definition left), the
four "collections of trees" functors (`sct_score_greater`, `sct_tree_
greater`, `scored_combo_tree_hash`, `scored_combo_tree_equal`), the three
custom set types (`scored_combo_tree_set`, `scored_combo_tree_tset`,
`scored_combo_tree_ptr_set`), and the `scored_combo_tree.__eq__` fix this
task made along the way (see that method's own FIX note in types.py).

Oracle rows come from `tools/cpp_probe/containers_sct_probe.cc`, run
through `tools/capture_containers_sct.py` into
`tests/fixtures/containers_sct.json` -- the SAME classic
`sct_score_greater`/`sct_tree_greater`/`scored_combo_tree_ptr_set`/
`scored_combo_tree_tset`/`scored_combo_tree_set` this module ports,
measured directly against the real compiled `libmoses.so`/
`libcomboreduct.so`, not inferred from reading types.h/types.cc alone.
"""

import json
import math

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.moses.moses.types import (
    _combo_tree_structural_eq,
    _sct_comparator_set,
    composite_score,
    demeID_t,
    scored_combo_tree,
    scored_combo_tree_equal,
    scored_combo_tree_hash,
    scored_combo_tree_ptr_set,
    scored_combo_tree_set,
    scored_combo_tree_tset,
    sct_score_greater,
    sct_tree_greater,
)
from pymetta.moses.representation import instance_set as instance_set_module
from pymetta.util.tree import tree

# ---------------------------------------------------------------------------
# demeID_t: exactly one definition, reached two different ways.
# ---------------------------------------------------------------------------


def test_demeID_t_has_exactly_one_definition():
    """`instance_set.py` imports `demeID_t` from `types.py` (its canonical
    C++ home, types.h:190) rather than redefining it -- both import paths
    must resolve to the literal SAME class object, not two classes that
    merely behave alike.
    """
    assert instance_set_module.demeID_t is demeID_t


# ---------------------------------------------------------------------------
# Building the pinned scored_combo_tree table -- mirrors
# containers_sct_probe.cc's own `table` byte-for-byte (same names, same
# trees, same scores/complexities) so the oracle tests below can replay
# the exact same rows classic was measured on.
# ---------------------------------------------------------------------------


def _leaf(value) -> tree:
    return tree(value)


def _and_with_arg1() -> tree:
    t = tree(builtin.logical_and)
    t.append_child(t.begin(), argument(1))
    return t


def _build_table() -> dict[str, scored_combo_tree]:
    nan = float("nan")
    return {
        "a-worse-score": scored_combo_tree(
            _leaf(builtin.logical_and), composite_score(-10.0, 0)
        ),
        "b-better-score": scored_combo_tree(
            _leaf(builtin.logical_and), composite_score(-2.0, 0)
        ),
        "c-tie-score-high-complexity": scored_combo_tree(
            _leaf(builtin.logical_and), composite_score(-5.0, 10)
        ),
        "d-tie-score-low-complexity": scored_combo_tree(
            _leaf(builtin.logical_and), composite_score(-5.0, 3)
        ),
        "e-tie-everything-bigger-tree": scored_combo_tree(
            _and_with_arg1(), composite_score(-5.0, 3)
        ),
        "f-tie-everything-arg1": scored_combo_tree(_leaf(argument(1)), composite_score(-5.0, 3)),
        "g-tie-everything-arg2": scored_combo_tree(_leaf(argument(2)), composite_score(-5.0, 3)),
        "h-tie-everything-nan-leaf": scored_combo_tree(_leaf(nan), composite_score(-5.0, 3)),
        "i-tie-everything-five-leaf": scored_combo_tree(_leaf(5.0), composite_score(-5.0, 3)),
    }


@pytest.fixture
def table() -> dict[str, scored_combo_tree]:
    return _build_table()


# ---------------------------------------------------------------------------
# scored_combo_tree.__eq__'s FIX: structural, not identity, tree comparison.
# ---------------------------------------------------------------------------


def test_scored_combo_tree_eq_is_structural_not_identity():
    """Two INDEPENDENTLY built trees with identical shape/content must
    compare equal -- real C++ `combo_tree::operator==` is structural
    (opencog/util/tree.h:3011). Before this task's fix, `scored_combo_tree.
    __eq__` fell back to Python's bare `self.tree == other.tree`, which --
    since `tree`/`tree_node` define no `__eq__` of their own -- is
    IDENTITY comparison: this would have been `False` for two distinct
    tree objects even though they hold exactly the same combo program.
    """
    a = scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))
    b = scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))
    assert a.tree is not b.tree, "fixture must use two DISTINCT tree objects"
    assert a == b


def test_scored_combo_tree_eq_type_before_value_on_the_tree():
    """The exact trap `vertex_equal`'s own docstring names: `1.0 ==
    builtin.logical_true` is `True` in bare Python despite the two values
    meaning completely different things. Real C++ `combo_tree::operator==`
    NEVER conflates them (`boost::variant`'s compiler-generated
    `operator==` discriminates by held alternative first) -- so a
    `contin_t` leaf holding `1.0` must NOT compare equal to a `builtin`
    leaf holding `logical_true`, even though `builtin.logical_true == 1`.
    """
    assert 1.0 == builtin.logical_true, "fixture must actually hit the trap"
    a = scored_combo_tree(tree(1.0), composite_score(-1.0, 3))
    b = scored_combo_tree(tree(builtin.logical_true), composite_score(-1.0, 3))
    assert a != b


def test_combo_tree_structural_eq_helper_directly():
    t1 = _and_with_arg1()
    t2 = _and_with_arg1()
    assert t1 is not t2
    assert _combo_tree_structural_eq(t1, t2) is True
    assert _combo_tree_structural_eq(t1, tree(builtin.logical_or)) is False


# ---------------------------------------------------------------------------
# scored_combo_tree_hash / scored_combo_tree_equal: dedup by TREE ALONE,
# not by scored_combo_tree.__eq__ (which also compares bscore/cscore).
# ---------------------------------------------------------------------------


def test_scored_combo_tree_equal_ignores_score():
    """types.cc:70-74's `scored_combo_tree_equal::operator()` is
    `tr1.get_tree() == tr2.get_tree()` ALONE -- score/bscore are not
    consulted at all, unlike `scored_combo_tree.__eq__`.
    """
    a = scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))
    b = scored_combo_tree(tree(builtin.logical_and), composite_score(-9.0, 7))
    assert a != b, "fixture must actually differ under scored_combo_tree.__eq__"
    assert scored_combo_tree_equal()(a, b) is True


def test_scored_combo_tree_hash_is_consistent_with_equal_over_a_multi_node_tree():
    """Hash/equal consistency (`a == b` under `scored_combo_tree_equal`
    implies `hash(a) == hash(b)`) is what makes `scored_combo_tree_set`'s
    O(1) dedup correct at all -- exercised over a genuine multi-node tree
    (`and($1)`, not a trivial single-vertex one) so the fold-over-every-
    node loop (`scored_combo_tree_hash.__call__`, types.cc:60-68's `for
    (combo::vertex vtx : tr)`) actually has more than one node to walk.
    """
    a = scored_combo_tree(_and_with_arg1(), composite_score(-1.0, 3))
    b = scored_combo_tree(_and_with_arg1(), composite_score(-9.0, 7))
    assert scored_combo_tree_equal()(a, b) is True
    assert scored_combo_tree_hash()(a) == scored_combo_tree_hash()(b)


def test_scored_combo_tree_set_dedups_by_tree_ignoring_score():
    s = scored_combo_tree_set()
    a = scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))
    b = scored_combo_tree(tree(builtin.logical_and), composite_score(-9.0, 7))
    c = scored_combo_tree(tree(builtin.logical_or), composite_score(-1.0, 3))
    assert s.insert(a) is True
    assert s.insert(b) is False, "same tree, different score -- must be rejected as a duplicate"
    assert s.insert(c) is True
    assert len(s) == 2


def test_scored_combo_tree_set_clear_empties_in_place():
    """S10 branch review, Minor: `clear()` (added for `metapopulation.
    update_best_candidates`, which used to rebind to a fresh
    `scored_combo_tree_set()` instead of mutating the existing one in
    place, unlike C++'s `_best_candidates.clear()`) must empty the SAME
    object -- not just make `len()` report 0, but leave a PRIOR reference
    to it observing the same emptiness, and allow fresh inserts
    afterward.
    """
    s = scored_combo_tree_set()
    a = scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))
    s.insert(a)
    assert len(s) == 1

    held = s
    s.clear()

    assert len(s) == 0
    assert s.empty() is True
    assert list(s) == []
    assert held is s and len(held) == 0  # the SAME object, not a new one

    b = scored_combo_tree(tree(builtin.logical_or), composite_score(-1.0, 3))
    assert s.insert(b) is True  # still usable after clear()
    assert len(s) == 1


def test_scored_combo_tree_is_unhashable_so_a_bare_set_cannot_hold_it():
    """The concrete reason a bare Python `set()` cannot stand in for
    `scored_combo_tree_set`: `scored_combo_tree.__hash__` is explicitly
    `None` (types.py).
    """
    sct = scored_combo_tree(tree(builtin.logical_and))
    with pytest.raises(TypeError):
        {sct}  # noqa: B018 -- deliberately constructing a bare set to prove it fails


# ---------------------------------------------------------------------------
# The comparator-set dedup rule pinned deliberately: "neither compares
# less than the other" (std::set's own equivalence-class semantics) is NOT
# the same test as `==`. A contin_t NaN vs. a finite contin_t leaf makes
# this concrete: IEEE-754 NaN comparisons are false in BOTH directions, so
# `sct_tree_greater`'s underlying `size_tree_order<vertex>` ties -- even
# though the two leaves are patently not equal (NaN != 5.0, and NaN is not
# even equal to itself).
# ---------------------------------------------------------------------------


def test_comparator_dedup_is_not_eq_the_nan_five_tie(table):
    nan_sct = table["h-tie-everything-nan-leaf"]
    five_sct = table["i-tie-everything-five-leaf"]

    # Not equal by ANY of this port's equality notions.
    assert nan_sct != five_sct
    assert scored_combo_tree_equal()(nan_sct, five_sct) is False
    assert math.isnan(nan_sct.tree.begin().data)

    # Yet neither compares less than the other under sct_tree_greater --
    # the exact "neither a<b nor b<a" equivalence std::set/boost::ptr_set
    # use for deduplication, ported by `_sct_comparator_set`.
    tree_gt = sct_tree_greater()
    assert tree_gt(nan_sct, five_sct) is False
    assert tree_gt(five_sct, nan_sct) is False

    cs = _sct_comparator_set(tree_gt)
    assert cs.insert(nan_sct) is True
    assert cs.insert(five_sct) is False, (
        "neither-compares-less must dedup this pair even though they are not `==`"
    )
    assert len(cs) == 1


def test_comparator_dedup_mutation_using_eq_instead_of_neither_less_would_not_dedup(table):
    """Guard-fires-by-mutation check for the exact rule above: an
    `_sct_comparator_set` variant that dedups by `==` instead of "neither
    compares less" would WRONGLY keep both the NaN and five-leaf rows
    (they are not `==`) -- i.e. it would fail to reproduce
    `scored_combo_tree_ptr_set`/`_tset`'s real C++ dedup behaviour. This
    proves the distinction matters, not just that the current
    implementation happens to use the right one.
    """
    nan_sct = table["h-tie-everything-nan-leaf"]
    five_sct = table["i-tie-everything-five-leaf"]

    class _EqBasedSet:
        def __init__(self):
            self._items = []

        def insert(self, x):
            if any(x == item for item in self._items):
                return False
            self._items.append(x)
            return True

    buggy = _EqBasedSet()
    assert buggy.insert(nan_sct) is True
    assert buggy.insert(five_sct) is True, "sanity: an == -based set does NOT dedup this pair"
    assert len(buggy._items) == 2, "mutation confirms == alone gives the WRONG (non-C++) answer"


# ---------------------------------------------------------------------------
# scored_combo_tree_tset / scored_combo_tree_ptr_set: basic container
# contract (insert/erase/contains/iterate/len/empty).
# ---------------------------------------------------------------------------


def test_ptr_set_orders_best_score_first(table):
    pset = scored_combo_tree_ptr_set()
    pset.insert(table["a-worse-score"])
    pset.insert(table["b-better-score"])
    pset.insert(table["c-tie-score-high-complexity"])
    scores = [sct.get_score() for sct in pset]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == table["b-better-score"].get_score()


def test_ptr_set_erase_and_contains(table):
    pset = scored_combo_tree_ptr_set()
    a = table["a-worse-score"]
    pset.insert(a)
    assert a in pset
    pset.erase(a)
    assert a not in pset
    assert pset.empty()
    pset.erase(a)  # no-op, matching std::set::erase's permissive contract
    assert pset.empty()


def test_tset_dedups_by_tree_shape_not_score(table):
    tset = scored_combo_tree_tset()
    a = table["a-worse-score"]  # and(), size 1
    same_shape_different_score = scored_combo_tree(
        tree(builtin.logical_and), composite_score(-999.0, 0)
    )
    assert tset.insert(a) is True
    assert tset.insert(same_shape_different_score) is False
    assert len(tset) == 1


# ---------------------------------------------------------------------------
# C++ oracle: full pairwise sct_score_greater/sct_tree_greater truth
# tables, and ptr_set/tset/unordered_set insertion behaviour, measured
# directly against the real compiled classic library
# (containers_sct_probe.cc via capture_containers_sct.py).
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_cpp_sct_score_greater_matches_every_pinned_pair(fixture_dir, table):
    fixture = json.loads((fixture_dir / "containers_sct.json").read_text())
    assert fixture["names"] == list(table.keys()), "fixture/table names must line up 1:1"
    score_gt = sct_score_greater()
    checked = 0
    for row in fixture["sct_score_greater_pairs"]:
        a, b, expected = table[row["a"]], table[row["b"]], row["gt"]
        assert score_gt(a, b) is expected, f"{row['a']} > {row['b']} (by score) mismatch"
        checked += 1
    assert checked == len(table) * (len(table) - 1)


@pytest.mark.oracle
def test_cpp_sct_tree_greater_matches_every_pinned_pair(fixture_dir, table):
    fixture = json.loads((fixture_dir / "containers_sct.json").read_text())
    tree_gt = sct_tree_greater()
    checked = 0
    for row in fixture["sct_tree_greater_pairs"]:
        a, b, expected = table[row["a"]], table[row["b"]], row["gt"]
        assert tree_gt(a, b) is expected, f"{row['a']} > {row['b']} (by tree) mismatch"
        checked += 1
    assert checked == len(table) * (len(table) - 1)


@pytest.mark.oracle
def test_cpp_ptr_set_insertion_matches_including_the_discarded_tie(fixture_dir, table):
    """The measured proof of the "DO NOT FIX" quirk's real consequence:
    real `boost::ptr_set<scored_combo_tree, sct_score_greater>` really
    does discard the second of two fully-tied candidates (types.cc:86-88's
    own comment: "we are desperate to break a tie ... otherwise ... the
    scored_combo_tree_ptr_set will discard anything that compares equal").
    """
    fixture = json.loads((fixture_dir / "containers_sct.json").read_text())
    pset = scored_combo_tree_ptr_set()
    results = [pset.insert(table[name]) for name in fixture["ptr_set_insert_order"]]
    assert results == fixture["ptr_set_insert_results"]
    assert len(pset) == fixture["ptr_set_size"]


@pytest.mark.oracle
def test_cpp_tset_insertion_matches_including_the_discarded_tie(fixture_dir, table):
    fixture = json.loads((fixture_dir / "containers_sct.json").read_text())
    tset = scored_combo_tree_tset()
    results = [tset.insert(table[name]) for name in fixture["tset_insert_order"]]
    assert results == fixture["tset_insert_results"]
    assert len(tset) == fixture["tset_size"]


@pytest.mark.oracle
def test_cpp_unordered_set_dedup_matches(fixture_dir):
    fixture = json.loads((fixture_dir / "containers_sct.json").read_text())
    uset = scored_combo_tree_set()
    results = [
        uset.insert(scored_combo_tree(tree(builtin.logical_and), composite_score(-1.0, 3))),
        uset.insert(scored_combo_tree(tree(builtin.logical_and), composite_score(-9.0, 7))),
        uset.insert(scored_combo_tree(tree(builtin.logical_or), composite_score(-1.0, 3))),
    ]
    assert results == fixture["unordered_set_insert_results"]
    assert len(uset) == fixture["unordered_set_size"]


# ---------------------------------------------------------------------------
# scored_combo_tree's boosting weight (types.h:274,291,295; types.cc:56)
# ---------------------------------------------------------------------------


def _sct(combo_str: str, score: float = 0.0, complexity: int = 0) -> scored_combo_tree:
    from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree

    return scored_combo_tree(str_to_combo_tree(combo_str), composite_score(score, complexity))


def test_weight_defaults_to_one():
    """`_weight(1.0)` is the last entry of the constructor's own
    member-initializer list (types.h:266) -- a fixed initial value, so a
    freshly built `scored_combo_tree` weighs exactly 1.0.

    Asserted as `== 1.0` AND as a `float`, so a mutant defaulting to 0.0
    (the other obvious choice, and the one that would silently zero every
    AdaBoost term) dies here.
    """
    sct = _sct("$1")
    assert sct.get_weight() == 1.0
    assert isinstance(sct.get_weight(), float)


def test_weight_is_not_a_constructor_parameter():
    """C++'s constructor takes `(tr, id, cs, bs)` (types.h:262-265) -- no
    weight argument in either language; only `set_weight` changes it.
    """
    from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree

    with pytest.raises(TypeError):
        scored_combo_tree(str_to_combo_tree("$1"), composite_score(), [], 2.0)


def test_set_weight_round_trips_at_double_precision_not_score_t():
    """`_weight` is `double` (types.h:274), NOT `score_t`.

    0.1 is chosen because it is the standard witness that binary64 and
    binary32 disagree: narrowing it changes the value. A mutant that
    routed `set_weight` (or the field) through `_to_score_t` would be
    invisible to a round-trip on, say, 0.5 -- this one kills it.
    """
    from pymetta.moses.moses.types import _to_score_t

    assert _to_score_t(0.1) != 0.1  # the witness is a real witness

    sct = _sct("$1")
    sct.set_weight(0.1)
    assert sct.get_weight() == 0.1
    assert sct.get_weight() != _to_score_t(0.1)


def test_eq_compares_weight():
    """`scored_combo_tree::operator==` compares `get_weight()`
    (types.cc:56) alongside tree/bscore/cscore.

    Nothing in this port sets a weight yet, so this term is inert in
    ordinary use -- which is exactly why it needs a test of its own
    rather than being assumed covered by the existing `__eq__` tests
    (every one of which compares two default-weighted trees).
    """
    a, b = _sct("$1", -1.0, 1), _sct("$1", -1.0, 1)
    assert a == b  # identical in every field, weights both 1.0

    b.set_weight(0.25)
    assert a != b
    assert b != a  # symmetric

    a.set_weight(0.25)
    assert a == b  # equal again once the weights agree


def test_scored_combo_tree_equal_functor_still_ignores_weight():
    """The functor is `tr1.get_tree() == tr2.get_tree()` and nothing else
    (types.cc:70-74) -- deliberately NOT `scored_combo_tree::operator==`.

    So two trees that differ ONLY in weight must still be "equal" to it,
    which is what keeps `scored_combo_tree_set`'s dedup meaning "same
    combo tree regardless of score" (that class's own docstring). A
    mutant that routed the functor through `__eq__` -- now that `__eq__`
    compares weight -- dies here.
    """
    a, b = _sct("$1", -1.0, 1), _sct("$1", -1.0, 1)
    b.set_weight(0.25)

    assert a != b  # differ under scored_combo_tree.__eq__ ...
    assert scored_combo_tree_equal()(a, b) is True  # ... but not to the functor
    assert scored_combo_tree_hash()(a) == scored_combo_tree_hash()(b)


def test_copy_sct_carries_the_weight():
    """`_copy_sct` (metapopulation.py) stands in for C++'s COPY
    constructor -- `new scored_combo_tree(cnd)` (merging.cc:167),
    `new scored_combo_tree(*cnd)` (domination.cc:304) -- which is the
    compiler-generated memberwise copy and therefore copies `_weight`.

    Lives in this module rather than test_metapopulation.py because what
    it pins is a `scored_combo_tree` field's semantics: the helper builds
    its copy field by field, so every field added to the class is one the
    helper can silently drop. Inert today (nothing calls `set_weight` in
    the package), and pinned precisely because an inert dropped field is
    the shape of bug docs/CARRY-FORWARD.md Sec 5.2/5.4 both record.
    """
    from pymetta.moses.metapopulation.metapopulation import _copy_sct

    original = _sct("$1", -1.0, 1)
    original.set_weight(0.375)

    copied = _copy_sct(original)

    assert copied is not original
    assert copied.get_weight() == 0.375
    assert copied == original


def test_fresh_construction_resets_the_weight_the_way_classic_does():
    """The complement of the copy test above, and the reason it is not a
    blanket "always carry the weight" rule.

    `merging.cc:279-280` builds `scored_combo_tree sct(cand.get_tree(),
    cand.get_demeID(), cand.get_composite_score(), bs)` -- a fresh
    CONSTRUCTION from another tree's parts, not a copy -- so classic
    resets that one to 1.0 too. `merge_demes`'s bscore-recompute block
    (merging.py) and `metapopulation.init` mirror it by doing nothing,
    and this pins that a "helpfully" propagated weight there would be a
    divergence.
    """
    original = _sct("$1", -1.0, 1)
    original.set_weight(0.375)

    rebuilt = scored_combo_tree(
        original.get_tree(),
        original.get_composite_score(),
        list(original.get_bscore()),
    )

    assert rebuilt.get_weight() == 1.0
