"""C++: moses/moses/scoring/behave_cscore.cc

No cxxtest exists for `behave_cscore` in classic (`$CLASSIC/tests/moses/`
has no `behave_cscoreUTest.cxxtest`) -- this module is a differential
oracle against every case in tests/fixtures/score.json (captured from
tools/cpp_probe/score_probe.cc, which builds a `behave_cscore` over the
same `ctruth_table_bscore` and dumps `get_cscore(tr)`'s full
`composite_score`) plus a couple of direct unit tests of the caching
behaviour `behave_cscore.__init__`'s `cache_size` parameter controls
(pure Python bookkeeping, not itself something score_probe.cc's one-shot
process can observe -- see behave_cscore.py's module docstring on why the
cache is a plain dict rather than the real `prr_cache_threaded`).

The two oracle tests assert EXACT (`==`) equality against the probe's
`%.17g` output -- see test_bscores.py's module docstring for why that is
now correct rather than merely convenient, now that `score_t` is
genuinely narrowed to IEEE-754 binary32 (types.py's `to_score_t`) at
every point classic's own arithmetic narrows.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import get_contin, is_contin
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.moses.types import _to_score_t
from pymetta.moses.scoring.behave_cscore import _cache_key, behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.moses.scoring.scoring_base import bscore_base

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_SCORE_CASES = json.loads((_FIXTURE_DIR / "score.json").read_text())
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"


def _bscorer_for(case: dict) -> ctruth_table_bscore:
    path = _TABLE_DATA_DIR / case["file"]
    table = loadTable(str(path), case["target_feature"])
    ctable = table.compressed(case["weighting_feature"])
    bsc = ctruth_table_bscore(ctable)
    if case["complexity_ratio"] is not None:
        bsc.set_complexity_coef(case["complexity_ratio"])
    return bsc


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_SCORE_CASES), ids=sorted(_SCORE_CASES))
def test_get_cscore_matches_cpp_probe(case_name):
    case = _SCORE_CASES[case_name]
    bsc = _bscorer_for(case)
    tr = str_to_combo_tree(case["tree"])

    behsc = behave_cscore(bsc)
    cs = behsc.get_cscore(tr)

    expected = case["composite_score"]
    assert cs.get_score() == expected["score"]
    assert cs.get_complexity() == expected["complexity"]
    assert cs.get_complexity_penalty() == expected["complexity_penalty"]
    assert cs.get_uniformity_penalty() == expected["uniformity_penalty"]
    assert cs.get_penalized_score() == expected["penalized_score"]


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_SCORE_CASES), ids=sorted(_SCORE_CASES))
def test_get_bscore_matches_get_cscore_sum(case_name):
    """`get_cscore(tr).get_score()` is exactly `sum_bscore(get_bscore(tr))`
    (behave_cscore.cc:87, `_bscorer.sum_bscore(bs)`) -- checked directly
    against `get_bscore`'s own return value, independent of the C++
    fixture, since this identity holds by construction in both languages.
    """
    case = _SCORE_CASES[case_name]
    bsc = _bscorer_for(case)
    tr = str_to_combo_tree(case["tree"])

    behsc = behave_cscore(bsc)
    bs = behsc.get_bscore(tr)
    cs = behsc.get_cscore(tr)

    assert cs.get_score() == bsc.sum_bscore(bs)


# ---------------------------------------------------------------------------
# Direct unit tests of the cache_size-gated caching behaviour
# ---------------------------------------------------------------------------


def test_get_cscore_without_cache_recomputes_every_call():
    """`cache_size=0` (the class default) disables caching altogether --
    `_have_cache` stays False, so get_cscore always takes the
    get_cscore_nocache path (behave_cscore.cc:54-55). Observed here via a
    bscorer whose complexity coefficient is mutated between calls: without
    caching, the second call must see the new coefficient.
    """
    case = _SCORE_CASES[next(iter(_SCORE_CASES))]
    bsc = _bscorer_for(case)
    tr = str_to_combo_tree(case["tree"])

    behsc = behave_cscore(bsc)  # cache_size=0 by default
    cs1 = behsc.get_cscore(tr)

    bsc.set_complexity_coef(1.0)
    cs2 = behsc.get_cscore(tr)

    assert cs2.get_complexity_penalty() != pytest.approx(cs1.get_complexity_penalty())


def test_get_cscore_with_cache_returns_the_same_value_on_repeat_calls():
    """`cache_size>0` enables the cache (`_have_cache = 0 < initial_cache_
    size`, behave_cscore.cc:35) -- a second call for the SAME tree must
    return the cached composite_score even after the underlying bscorer's
    complexity coefficient changes underneath it (the whole point of a
    cache: it doesn't re-derive on cache hit).
    """
    case = _SCORE_CASES[next(iter(_SCORE_CASES))]
    bsc = _bscorer_for(case)
    tr = str_to_combo_tree(case["tree"])

    behsc = behave_cscore(bsc, cache_size=16)
    cs1 = behsc.get_cscore(tr)

    bsc.set_complexity_coef(1.0)
    cs2 = behsc.get_cscore(tr)

    assert cs2.get_complexity_penalty() == pytest.approx(cs1.get_complexity_penalty())


# ---------------------------------------------------------------------------
# Regression test for the final review's Important finding: the cache used
# to be keyed on `combo_tree_to_str(tr)`, which prints a contin leaf at
# ~6 significant digits (iostream_combo.py's `_contin_to_str`) -- so two
# structurally DIFFERENT trees that print identically collided on one
# cache slot, and the second tree silently got back the first tree's
# `composite_score`. `ctruth_table_bscore` is boolean-only (bscores.py's
# module docstring), so this uses a minimal fake `bscore_base` whose score
# depends on the tree's own contin leaf value, to demonstrate the
# collision end-to-end through `behave_cscore.get_cscore` rather than only
# unit-testing the cache-key helper in isolation.
# ---------------------------------------------------------------------------


class _FakeContinBscorer(bscore_base):
    """Minimal `bscore_base`: the "score" is just the tree's own bare
    contin leaf value, so two different contin values must produce two
    different `composite_score.get_score()` results.
    """

    def __call__(self, tr):
        return [get_contin(tr.begin().data)]

    def best_possible_bscore(self):
        return [0.0]


def test_get_cscore_cache_does_not_collide_on_contin_leaves_that_print_identically():
    # 1.0 and 1.0000002 both print as "1 " via combo_tree_to_str's ~6
    # significant-digit formatting, but remain distinct once narrowed to
    # float32 (1.0 vs 1.0000002384185791...) -- so a correct,
    # collision-free cache key must still tell them apart.
    tr1 = str_to_combo_tree("1.0")
    tr2 = str_to_combo_tree("1.0000002")
    assert combo_tree_to_str(tr1) == combo_tree_to_str(tr2), (
        "fixture must actually collide under combo_tree_to_str, or this test proves nothing"
    )
    assert _to_score_t(tr1.begin().data) != _to_score_t(tr2.begin().data)

    behsc = behave_cscore(_FakeContinBscorer(), cache_size=16)
    cs1 = behsc.get_cscore(tr1)
    cs2 = behsc.get_cscore(tr2)

    assert cs1.get_score() == _to_score_t(1.0)
    assert cs2.get_score() == _to_score_t(1.0000002)
    assert cs1.get_score() != cs2.get_score()

    # Repeat calls must still hit the (correct) cached value for each tree
    # independently, not silently fall back to whichever was scored first.
    assert behsc.get_cscore(tr1).get_score() == cs1.get_score()
    assert behsc.get_cscore(tr2).get_score() == cs2.get_score()


# ---------------------------------------------------------------------------
# Regression test for the ROUND 2 review Critical: the round-1 fix above
# closed the contin-print-precision collision but opened a wider one --
# `_cache_key` paired each node with a bare `node.data`, and `contin_t`
# (a bare `float`) compares AND HASHES equal to `builtin` (an `IntEnum`)
# at matching ordinals (`1.0 == builtin.logical_true`, `hash(1.0) ==
# hash(builtin.logical_true)`), the exact "vertex compared by value
# without checking its type first" trap the whole fix wave exists to
# close. `_cache_key` now pairs every node's value with `type(node.data)`
# too, so two vertices of different variant TYPES can never produce equal
# keys, no matter what their values compare as.
# ---------------------------------------------------------------------------


def test_cache_key_does_not_collide_across_vertex_types_at_matching_ordinals():
    # contin_t(1.0) vs builtin.logical_true (ordinal 1) -- the exact pair
    # the round-2 review demonstrated.
    contin_one = _cache_key(str_to_combo_tree("1"))
    bool_true = _cache_key(str_to_combo_tree("true"))
    assert contin_one != bool_true
    assert hash(contin_one) != hash(bool_true) or contin_one != bool_true

    # contin_t(2.0) vs builtin.logical_false (ordinal 2).
    contin_two = _cache_key(str_to_combo_tree("2"))
    bool_false = _cache_key(str_to_combo_tree("false"))
    assert contin_two != bool_false

    # argument($1, idx=1) vs contin_t(1.0) and vs builtin.logical_true --
    # already safe even pre-round-2 (argument.__eq__ returns NotImplemented
    # for a non-argument other, so it never compares equal to a bare int/
    # float/builtin regardless of any hash collision), reasserted here so
    # a future change to argument's __eq__/__hash__ cannot silently
    # reopen this without a test noticing.
    arg_one = _cache_key(str_to_combo_tree("$1"))
    assert arg_one != contin_one
    assert arg_one != bool_true

    # Sanity check on the fixture itself: two identical trees must still
    # collapse to the SAME key -- this fix must not turn the cache into a
    # no-op that never hits.
    assert _cache_key(str_to_combo_tree("1")) == contin_one


def test_get_cscore_cache_does_not_collide_across_vertex_types_end_to_end():
    """The end-to-end demonstration: a contin-leaf tree and a
    boolean-leaf tree, scored through the SAME cache-enabled
    `behave_cscore`, must each get back their OWN `composite_score` --
    not silently the other's, the way `_cache_key`'s round-1 collision
    let happen (verified directly against the round-1 code: scoring the
    contin tree first, then the boolean tree, made the boolean tree's
    `get_cscore` return the contin tree's cached 999.0 instead of its own
    1.0 -- see final-fix-wave-report.md for the full transcript).
    """

    class _FakeTypeSensitiveBscorer(bscore_base):
        """Scores a contin leaf as 999.0 and anything else (here, a
        boolean constant) as 1.0 -- deliberately far apart so a collision
        is unmistakable, not a rounding artifact.
        """

        def __call__(self, tr):
            v = tr.begin().data
            return [999.0] if is_contin(v) else [1.0]

        def best_possible_bscore(self):
            return [0.0]

    behsc = behave_cscore(_FakeTypeSensitiveBscorer(), cache_size=16)
    tr_contin = str_to_combo_tree("1")  # contin_t(1.0)
    tr_bool = str_to_combo_tree("true")  # builtin.logical_true (ordinal 1)

    cs_contin = behsc.get_cscore(tr_contin)
    cs_bool = behsc.get_cscore(tr_bool)

    assert cs_contin.get_score() == _to_score_t(999.0)
    assert cs_bool.get_score() == _to_score_t(1.0)
    assert cs_contin.get_score() != cs_bool.get_score()

    # Repeat calls (now both cache hits) must stay correct and independent.
    assert behsc.get_cscore(tr_contin).get_score() == _to_score_t(999.0)
    assert behsc.get_cscore(tr_bool).get_score() == _to_score_t(1.0)
