"""Oracle and guard tests for pymoses.feature_selection.algo.random.

## Oracle: tools/cpp_probe/fs_standalone_probe.cc -> tests/fixtures/fs_standalone.json

The fixture's `"random"` group runs classic's own
`random_select_feature_sets(const CTable&, const feature_selection_parameters&)`
directly (random.cc:32), with `randGen().seed(1)` immediately before every
call (matching a fresh `-r 1` process). `target_size=3` on
`trainingBoolean.dat` (arity 3) exercises the with-replacement duplicate-
draw quirk: three draws land on only two distinct features (`{0, 2}`),
proving the port reproduces classic's own "final set can be smaller than
target_size" behaviour, not a naive without-replacement fix.
"""

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.algo.random import random_select_feature_sets, random_selection
from pymoses.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_standalone.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN["random"]


@pytest.fixture()
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


@pytest.fixture()
def xor_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y").compressed()


@pytest.mark.oracle
@pytest.mark.parametrize("target_size", [0, 1, 2, 3])
def test_random_select_feature_sets_matches_classic_boolean(
    boolean_ctable, golden, target_size
):
    randGen().seed(1)
    pop = random_select_feature_sets(
        boolean_ctable, feature_selection_parameters(target_size=target_size)
    )
    assert len(pop) == 1
    score, fs = pop[0]
    assert score == 0.0
    assert sorted(fs) == sorted(golden["boolean"][str(target_size)][0]["fs"])


@pytest.mark.oracle
@pytest.mark.parametrize("target_size", [0, 1, 2])
def test_random_select_feature_sets_matches_classic_xor(xor_ctable, golden, target_size):
    randGen().seed(1)
    pop = random_select_feature_sets(
        xor_ctable, feature_selection_parameters(target_size=target_size)
    )
    score, fs = pop[0]
    assert sorted(fs) == sorted(golden["xor"][str(target_size)][0]["fs"])


def test_random_selection_with_replacement_can_return_fewer_than_num_desired():
    """DO NOT FIX: `random_selection`'s own docstring. Proven directly
    here (not just via the oracle above, whose boolean target_size=3 case
    already demonstrates it): a features set small enough relative to
    num_desired makes a duplicate draw certain.
    """
    randGen().seed(1)
    fs = random_selection({0, 1}, 5)
    assert fs <= {0, 1}
    assert len(fs) <= 2


def test_random_selection_zero_desired_returns_empty_and_draws_nothing():
    randGen().seed(1)
    before = randGen().randint()
    randGen().seed(1)
    fs = random_selection({0, 1, 2}, 0)
    after = randGen().randint()
    assert fs == set()
    assert before == after  # the very next draw is unaffected: zero draws happened


def test_random_select_feature_sets_target_size_zero_is_not_all_features():
    """Unlike simple/smd/incremental, "random" has NO target_size<=0
    "return all features" special case (random.cc has none) --
    target_size=0 selects nothing.
    """
    ctable = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()
    randGen().seed(1)
    pop = random_select_feature_sets(ctable, feature_selection_parameters(target_size=0))
    assert pop == [(0.0, set())]
