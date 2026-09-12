"""Oracle and guard tests for pymoses.feature_selection.scorers.moses_matrix
(pre_scorer) and pymoses.feature_selection.scorers.fs_scorer_base.

## Oracle: tools/cpp_probe/fs_standalone_probe.cc -> tests/fixtures/fs_standalone.json

The fixture's `"smd_pre"` group runs classic's own
`smd_select_feature_sets(const CTable&, const feature_selection_parameters&)`
with `scorer="pre"` -- reachable because "smd" never touches
hill_climbing/get_rep() (unlike "hc" -- see deme_optimize's own module
docstring), so `fs_scorer<FeatureSet>`'s OTHER branch (`pre_scorer`) can
actually be exercised end to end against live classic. The probe found and
worked around a real trap here: `feature_selection_parameters`'s
`pre_penalty`/`pre_min_activation`/`pre_max_activation`/`pre_positive`
fields are OMITTED from the C++ struct's own constructor -- genuinely
indeterminate stack garbage on a bare `feature_selection_parameters()` --
so the probe sets all four explicitly to the real CLI defaults before
calling `scorer="pre"` (confirmed necessary: omitting them crashed the
probe with an unrelated-looking assertion from garbage memory).
"""

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.algo.stochastic_max_dependency import smd_select_feature_sets
from pymoses.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymoses.feature_selection.scorers.fs_scorer_base import fs_scorer_base
from pymoses.feature_selection.scorers.moses_matrix import pre_scorer
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_standalone.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())["smd_pre"]


@pytest.fixture()
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


@pytest.mark.oracle
@pytest.mark.parametrize("target_size", [1, 2])
def test_smd_with_pre_scorer_matches_classic(boolean_ctable, target_size):
    randGen().seed(1)
    params = feature_selection_parameters(
        target_size=target_size,
        scorer="pre",
        pre_penalty=1.0,
        pre_min_activation=0.5,
        pre_max_activation=1.0,
        pre_positive=True,
    )
    pop = smd_select_feature_sets(boolean_ctable, params)
    got = [(score, sorted(fs)) for score, fs in pop]
    want = [(e["score"], sorted(e["fs"])) for e in _GOLDEN["boolean"][str(target_size)]]
    assert got == want


def test_pre_scorer_is_a_fs_scorer_base(boolean_ctable):
    scorer = pre_scorer(boolean_ctable)
    assert isinstance(scorer, fs_scorer_base)


def test_pre_scorer_single_feature_scores(boolean_ctable):
    """Direct (non-dispatcher) call, matching the golden target_size=1
    entries one feature at a time -- x3 (index 2) alone gets a perfect
    precision of 1.0 on this fixture.
    """
    scorer = pre_scorer(boolean_ctable, confi=50.0)
    assert scorer({2}) == pytest.approx(1.0)
    assert scorer({0}) == pytest.approx(0.75)
    assert scorer({1}) == pytest.approx(0.75)


def test_fs_scorer_base_confidence_formula():
    """`fs_scorer_base.confidence` in isolation, independent of which
    subclass calls it (MICScorerCTable's own tests already cover the
    formula through that subclass; this proves the shared base directly).
    """
    import math

    class _DummyScorer(fs_scorer_base):
        def __call__(self, features):
            return 0.0

    ctable = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()
    scorer = _DummyScorer(ctable, confi=0.1)
    n = ctable.uncompressed_size()
    expected = n / (n + math.exp(-0.1 * 2))
    assert scorer.confidence(2) == pytest.approx(expected, rel=1e-12)


def test_fs_scorer_base_operator_call_is_abstract():
    with pytest.raises(TypeError):
        fs_scorer_base(None, 100.0)  # type: ignore[abstract]
