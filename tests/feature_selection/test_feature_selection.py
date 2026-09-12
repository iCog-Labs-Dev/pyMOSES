"""Tests for pymoses.feature_selection.main.feature_selection (the
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
from pathlib import Path

import pytest

from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.main.feature_selection import (
    initial_features,
    score_individual_features,
    select_feature_sets,
    select_features,
)
from pymoses.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymoses.util.rng import randGen

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "feature_selection.json"
)
_STANDALONE_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_standalone.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())
_STANDALONE_GOLDEN = json.loads(_STANDALONE_FIXTURE_PATH.read_text())


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


@pytest.mark.oracle
@pytest.mark.parametrize(
    "algorithm,golden_group",
    [("random", "random"), ("smd", "smd"), ("inc", "inc")],
)
def test_select_feature_sets_dispatches_to_every_ported_algorithm(
    boolean_ctable, algorithm, golden_group
):
    """All five algorithms are now real dispatch targets. Checks the
    EXACT oracle value per algorithm at ``target_size=1`` (not just "is a
    well-formed population") -- proven necessary by mutation: a version
    of this test that only checked shape (score is a float, feature set
    is a subset of the table's columns) did NOT go red when the
    dispatcher's own "inc"/"smd" branches were swapped (both algorithms
    happen to produce shape-valid, but algorithm-WRONG, results on this
    fixture at ``target_size=1``) -- only comparing against each
    algorithm's own fixture group catches that. See ``test_random.py``/
    ``test_stochastic_max_dependency.py``/``test_incremental.py`` for
    each algorithm's own full oracle coverage; this test is scoped to
    proving the DISPATCH itself, not re-deriving that coverage.
    """
    randGen().seed(1)
    pop = select_feature_sets(
        boolean_ctable, feature_selection_parameters(algorithm=algorithm, target_size=1)
    )
    got = [(score, sorted(fs)) for score, fs in pop]
    if golden_group == "inc":
        # fs_standalone.json's own "inc" group is a plain list of cases,
        # not keyed by target_size -- case 0 is (boolean, target_size=1,
        # threshold=0, terms=1), see test_incremental.py's own _CASES.
        want = [
            (e["score"], sorted(e["fs"])) for e in _STANDALONE_GOLDEN["inc"][0]["pop"]
        ]
    else:
        want = [
            (e["score"], sorted(e["fs"]))
            for e in _STANDALONE_GOLDEN[golden_group]["boolean"]["1"]
        ]
    assert got == want


def test_select_feature_sets_hc_dispatch_reaches_deme_optimize(boolean_ctable):
    """Classic's OWN "hc" path segfaults unconditionally (see
    ``pymoses.feature_selection.algo.deme_optimize``'s module docstring
    and ``tools/cpp_probe/fs_standalone_probe.cc``'s header comment for
    the confirmed upstream bug) -- there is no golden value to compare
    against, so this only proves the dispatcher builds a real
    ``hill_climbing`` optimizer and reaches ``moses_select_feature_sets``
    without raising, and that the result is a well-formed population.
    See ``tests/feature_selection/test_deme_optimize.py`` for the
    algorithm's own dedicated coverage.
    """
    randGen().seed(1)
    pop = select_feature_sets(
        boolean_ctable,
        feature_selection_parameters(algorithm="hc", target_size=1, hc_max_evals=100),
    )
    assert len(pop) >= 1
    for _score, fs in pop:
        assert fs <= set(range(boolean_ctable.get_arity()))


def test_select_feature_sets_exits_for_an_unknown_algorithm(boolean_ctable, capsys):
    """C++: feature-selection.cc:169-174 -- `cerr << "Fatal Error: ..." ;
    exit(1);`. Ported for real since 2026-09-10 (it used to raise
    `NotImplementedError`), so the message is asserted VERBATIM against
    what the live classic binary writes:

        $ feature-selection -i trainingBoolean.dat -u y -l NONE -C2 -a bogus
        Fatal Error: Algorithm 'bogus' is unknown, please consult the help
        for the list of algorithms.
        $ echo $?
        1
    """
    with pytest.raises(SystemExit) as excinfo:
        select_feature_sets(boolean_ctable, feature_selection_parameters(algorithm="bogus"))
    assert excinfo.value.code == 1
    assert capsys.readouterr().err == (
        "Fatal Error: Algorithm 'bogus' is unknown, please consult the help "
        "for the list of algorithms.\n"
    )


def test_select_feature_sets_unknown_algorithm_message_is_not_confused_with_a_real_gate(
    boolean_ctable, capsys
):
    """Mutation-proven distinguishing test (mirrors this test module's
    former unported-algorithm test, now that all five algorithms are
    real): a typo'd algorithm name must NOT silently match one of the
    five real dispatch branches (which all use exact ``==`` string
    comparison, not e.g. a prefix or case-insensitive match).
    """
    with pytest.raises(SystemExit) as excinfo:
        select_feature_sets(boolean_ctable, feature_selection_parameters(algorithm="Simple"))
    assert excinfo.value.code == 1
    assert "Algorithm 'Simple' is unknown" in capsys.readouterr().err


def test_select_features_ctable_returns_the_top_entry(boolean_ctable):
    randGen().seed(1)
    fs = select_features(boolean_ctable, feature_selection_parameters(target_size=1))
    assert fs == {2}


def test_select_features_table_compresses_first(boolean_table):
    randGen().seed(1)
    fs = select_features(boolean_table, feature_selection_parameters(target_size=1))
    assert fs == {2}


def test_select_features_subsamples_the_table_before_selecting(boolean_table):
    """`select_features(const Table&, ...)` (feature-selection.cc:187-196)
    compresses, then subsamples IN PLACE when `subsampling_ratio < 1.0`,
    then delegates to the CTable overload. It raised here while
    `subsampleCTable` was unported; it is ported now.

    Asserted through a real selection rather than a smoke call: with a
    seeded RNG the subsampled run must still return a feature set of the
    requested size, and must not raise.
    """
    randGen().seed(1)
    got = select_features(
        boolean_table, feature_selection_parameters(target_size=1, subsampling_ratio=0.5)
    )
    assert len(got) == 1


def test_select_features_leaves_the_table_alone_at_ratio_one(boolean_table):
    """The other half: at the default 1.0 the subsample branch must not
    run at all, so this and the test above cannot both pass against a
    version that subsamples unconditionally.
    """
    randGen().seed(1)
    got = select_features(
        boolean_table, feature_selection_parameters(target_size=1, subsampling_ratio=1.0)
    )
    assert len(got) == 1


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
