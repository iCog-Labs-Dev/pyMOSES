"""Tests for pymetta.feature_selection.main.feature_selection (the
algorithm dispatcher, select_feature_sets/select_features, and the two
library helpers initial_features/score_individual_features).

## Oracle: the standalone `feature-selection` binary

`score_individual_features` is compared directly against
`$CLASSIC/build/moses/feature-selection/main/feature-selection`'s own
`-l INFO` log line (`log_selected_features(): <label> <score>`), captured
by hand once (see this module's own docstrings below) rather than through
the JSON fixture -- the numbers match `tests/fixtures/feature_selection.json`'s
`"mi"` group exactly (see test_mutual_info.py's own confidence-saturation
note for why `MICScorerCTable`, which this function actually uses, agrees
with bare MI here), so no second probe run was needed.

    $ ./feature-selection -i trainingBoolean.dat -u y --target-size=2 \\
          -o /dev/null -l INFO -F /dev/stderr
    ... log_selected_features(): x2 0.0487949
    ... log_selected_features(): x3 0.548795
"""

import json
import re
from pathlib import Path

import pytest

from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.feature_selection.main.feature_selection import (
    initial_features,
    score_individual_features,
    select_feature_sets,
    select_features,
)
from pymetta.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymetta.util.rng import randGen

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "feature_selection.json"
)
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture()
def boolean_table():
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")


@pytest.fixture()
def boolean_ctable(boolean_table) -> CTable:
    return boolean_table.compressed()


@pytest.mark.oracle
def test_select_feature_sets_dispatches_to_simple_and_matches_classic(boolean_ctable):
    randGen().seed(1)
    pop = select_feature_sets(boolean_ctable, feature_selection_parameters(target_size=2))
    assert sorted(pop[0][1]) == _GOLDEN["simple"]["boolean"]["2"][0]


@pytest.mark.parametrize(
    "algorithm,expected_file",
    [
        ("inc", "incremental.cc"),
        ("smd", "stochastic_max_dependency.cc"),
        ("random", "random.cc"),
        ("hc", "deme_optimize.cc"),
    ],
)
def test_select_feature_sets_raises_for_unported_algorithms(
    boolean_ctable, algorithm, expected_file
):
    # The C++ FILE CITATION is asserted, not just the exception type or the
    # algorithm name: `select_feature_sets`'s final "unknown algorithm"
    # branch ALSO raises NotImplementedError with the algorithm name in its
    # own message (`{fs_params.algorithm!r}`), so neither a bare
    # `pytest.raises(NotImplementedError)` NOR a match on the algorithm name
    # alone can tell "this specific algorithm's own gate fired" from "the
    # dispatch fell all the way through to the unknown-algorithm branch" --
    # proven by mutation: removing the `algorithm in _PORTED_ALGORITHMS`
    # gate still raised NotImplementedError, with the algorithm name still
    # in the message, from the OTHER (unknown-algorithm) branch, and both
    # weaker assertions missed it. Only the specific C++ file name
    # discriminates the two branches.
    with pytest.raises(NotImplementedError, match=re.escape(expected_file)):
        select_feature_sets(boolean_ctable, feature_selection_parameters(algorithm=algorithm))


def test_select_feature_sets_raises_for_an_unknown_algorithm(boolean_ctable):
    with pytest.raises(NotImplementedError, match="unknown algorithm"):
        select_feature_sets(boolean_ctable, feature_selection_parameters(algorithm="bogus"))


def test_select_features_ctable_returns_the_top_entry(boolean_ctable):
    randGen().seed(1)
    fs = select_features(boolean_ctable, feature_selection_parameters(target_size=1))
    assert fs == {2}


def test_select_features_table_compresses_first(boolean_table):
    randGen().seed(1)
    fs = select_features(boolean_table, feature_selection_parameters(target_size=1))
    assert fs == {2}


def test_select_features_raises_for_subsampling_ratio_below_one(boolean_table):
    with pytest.raises(NotImplementedError):
        select_features(
            boolean_table, feature_selection_parameters(target_size=1, subsampling_ratio=0.5)
        )


@pytest.mark.oracle
def test_score_individual_features_matches_classic_log_line(boolean_table):
    """See this module's own docstring for the classic invocation this
    compares against.
    """
    scores = score_individual_features(boolean_table, feature_selection_parameters())
    assert scores == pytest.approx([0.04879494069539869, 0.04879494069539869, 0.5487949406953987])


def test_initial_features_maps_known_labels_to_indices(boolean_table):
    params = feature_selection_parameters(initial_features=["x3", "x1"])
    fs = initial_features(boolean_table.itable.get_labels(), params)
    assert fs == {0, 2}


def test_initial_features_ignores_unknown_labels(boolean_table):
    params = feature_selection_parameters(initial_features=["x3", "not-a-real-feature"])
    fs = initial_features(boolean_table.itable.get_labels(), params)
    assert fs == {2}


def test_feature_selection_parameters_defaults_match_the_struct_ctor():
    """Pins every field from feature-selection.h:47-68's own constructor
    -- see feature_selection_params.py's own module docstring for the
    handful that diverge from BOTH classic CLIs' defaults, and why.
    """
    p = feature_selection_parameters()
    assert p.algorithm == "simple"
    assert p.scorer == "mi"
    assert p.target_size == 1
    assert p.exp_distrib is False
    assert p.threshold == 0.0
    assert p.jobs == 1
    assert p.subsampling_ratio == 1.0
    assert p.inc_target_size_epsilon == 1.0e-10
    assert p.inc_red_intensity == -1.0
    assert p.inc_interaction_terms == 1
    assert p.smd_top_size == 100
    assert p.hc_max_evals == 10000
    assert p.max_time == 2_147_483_647
    assert p.hc_max_score == 1.0e50
    assert p.hc_cache_size == 1000
    assert p.hc_fraction_of_remaining == 1.0
    assert p.hc_crossover is True
    assert p.hc_crossover_pop_size == 300
    assert p.hc_crossover_min_neighbors == 1000
    assert p.hc_widen_search is True
    assert p.hc_fraction_of_nn == 2.0
    assert p.mi_confi == 50.0
    assert p.pre_penalty == 1.0
    assert p.pre_min_activation == 0.5
    assert p.pre_max_activation == 1.0
    assert p.pre_positive is True
