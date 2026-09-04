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

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore

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
