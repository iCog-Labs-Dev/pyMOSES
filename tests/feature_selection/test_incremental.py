"""Oracle and guard tests for pymetta.feature_selection.algo.incremental.

## Oracle: tools/cpp_probe/fs_standalone_probe.cc -> tests/fixtures/fs_standalone.json

The fixture's `"inc"` group runs classic's own
`incremental_select_feature_sets(const CTable&, const
feature_selection_parameters&)` directly (incremental.cc:31) for seven
`(table, target_size, threshold, inc_interaction_terms)` combinations.
Fully deterministic -- no RNG at all in this algorithm -- so `randGen().
seed(1)` before each call is for uniformity only, not a fidelity
requirement (the module docstring makes the same point).
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.feature_selection.algo.incremental import (
    adaptive_incremental_selection,
    incremental_select_feature_sets,
    incremental_selection,
    redundant_features,
)
from pymetta.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_standalone.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())["inc"]


@pytest.fixture()
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


@pytest.fixture()
def xor_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y").compressed()


_CASES = [
    ("boolean", 1, 0.0, 1),
    ("boolean", 2, 0.0, 1),
    ("boolean", 0, 0.0, 1),
    ("boolean", 0, 0.1, 1),
    ("boolean", 2, 0.0, 2),
    ("xor", 1, 0.0, 1),
    ("xor", 0, 0.0, 1),
]


@pytest.mark.oracle
@pytest.mark.parametrize(("table_name", "target_size", "threshold", "terms"), _CASES)
def test_incremental_select_feature_sets_matches_classic(
    boolean_ctable, xor_ctable, table_name, target_size, threshold, terms
):
    ctable = boolean_ctable if table_name == "boolean" else xor_ctable
    index = _CASES.index((table_name, target_size, threshold, terms))
    randGen().seed(1)
    pop = incremental_select_feature_sets(
        ctable,
        feature_selection_parameters(
            target_size=target_size, threshold=threshold, inc_interaction_terms=terms
        ),
    )
    got = [(score, sorted(fs)) for score, fs in pop]
    want = [(e["score"], sorted(e["fs"])) for e in _GOLDEN[index]["pop"]]
    assert got == want


def test_redundant_features_first_occurrence_tie_break():
    """`max_element`'s first-occurrence-under-strict-less tie-break,
    exercised directly: three features scoring identically as pairs, so
    every 1-sized subset's complement scores identically too -- the
    result must be the complement of the FIRST 1-sized subset in
    ascending order, `{0}`, i.e. `features - {0} == {1, 2}`.
    """

    def scorer(fs):
        return 0.0  # every subset scores identically -> first tie wins

    result = redundant_features({0, 1, 2}, scorer, threshold=1.0)
    assert result == {1, 2}


def test_redundant_features_returns_empty_when_nothing_qualifies():
    def scorer(fs):
        return len(fs) * 10.0  # bigger subsets always score much higher

    result = redundant_features({0, 1, 2}, scorer, threshold=0.001)
    assert result == set()


def test_incremental_selection_is_deterministic_pure_function_of_inputs():
    scores = {frozenset(): 0.0, frozenset({0}): 0.6, frozenset({1}): 0.2, frozenset({0, 1}): 0.9}

    def scorer(fs):
        return scores[frozenset(fs)]

    result = incremental_selection({0, 1}, scorer, threshold=0.5, max_interaction_terms=1)
    assert result == {0}


def test_adaptive_incremental_selection_bisects_toward_target_size():
    def scorer(fs):
        # Monotonic in threshold position: score(feature i) decreases as i grows,
        # so lower thresholds keep more features.
        return {0: 0.9, 1: 0.5, 2: 0.1}.get(next(iter(fs), -1), 0.0) if len(fs) == 1 else 0.0

    result = adaptive_incremental_selection(
        {0, 1, 2}, scorer, features_size_target=2, max_interaction_terms=1, epsilon=0.001
    )
    assert len(result) == 2
    assert result == {0, 1}
