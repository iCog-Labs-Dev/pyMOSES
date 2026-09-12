"""Oracle and guard tests for pymoses.feature_selection.algo.stochastic_max_dependency.

## Oracles: tools/cpp_probe/fs_standalone_probe.cc -> tests/fixtures/fs_standalone.json

Two independent groups from the SAME fixture:

* `"shuffle"` -- the RAW `std::random_shuffle(v.begin(), v.end(), shr)`
  pass over a plain `{0,...,n-1}` vector, isolated from any scorer or
  table. An element-for-element draw-order check on `_random_shuffle`
  ITSELF (not merely its downstream effect on a selected feature set) --
  the fidelity requirement this project's brief names explicitly ("verify
  draw order element-for-element, not set-wise").
* `"smd"` -- `smd_select_feature_sets(const CTable&, const
  feature_selection_parameters&)` directly (stochastic_max_dependency.cc:
  34), `randGen().seed(1)` immediately before every call. `target_size=2`/
  `3` on `trainingBoolean.dat` exercise both the shuffle AND
  `fs_scorer(mi)` together, including tie ordering (`"2"`'s six entries
  alternate `{1,2}`/`{0,2}` twice then `{0,1}` twice -- an exact
  reproduction of this depends on getting BOTH the shuffle order and the
  `std::multimap`-tie-preserves-insertion-order behaviour right).
"""

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.algo import stochastic_max_dependency as smd_module
from pymoses.feature_selection.algo.stochastic_max_dependency import (
    _random_shuffle,
    smd_select_feature_sets,
    stochastic_max_dependency_selection,
)
from pymoses.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_standalone.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture()
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


@pytest.fixture()
def xor_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y").compressed()


@pytest.mark.oracle
@pytest.mark.parametrize(
    "seed,n,key",
    [
        (1, 5, "n5_seed1"),
        (1, 8, "n8_seed1"),
        (42, 5, "n5_seed42"),
        (1, 1, "n1_seed1"),
        (1, 0, "n0_seed1"),
    ],
)
def test_random_shuffle_matches_classic_element_for_element(seed, n, key):
    randGen().seed(seed)
    v = list(range(n))
    _random_shuffle(v)
    assert v == _GOLDEN["shuffle"][key]


def test_random_shuffle_draws_exactly_n_minus_one_times(monkeypatch):
    """One `randint_n` draw per loop iteration -- `len(seq) - 1` total,
    REGARDLESS of how many actual swaps occur (the `i != j` check only
    guards the swap, not the draw).
    """
    randGen().seed(1)
    rng = randGen()
    orig_randint_n = rng.randint_n
    calls = []

    def counting_randint_n(n):
        calls.append(n)
        return orig_randint_n(n)

    monkeypatch.setattr(rng, "randint_n", counting_randint_n)
    _random_shuffle(list(range(6)))
    assert len(calls) == 5


@pytest.mark.oracle
@pytest.mark.parametrize("target_size", [0, 1, 2, 3])
def test_smd_select_feature_sets_matches_classic_boolean(boolean_ctable, target_size):
    randGen().seed(1)
    pop = smd_select_feature_sets(
        boolean_ctable, feature_selection_parameters(target_size=target_size)
    )
    got = [(score, sorted(fs)) for score, fs in pop]
    want = [(e["score"], sorted(e["fs"])) for e in _GOLDEN["smd"]["boolean"][str(target_size)]]
    assert got == want


@pytest.mark.oracle
@pytest.mark.parametrize("target_size", [0, 1, 2])
def test_smd_select_feature_sets_matches_classic_xor(xor_ctable, target_size):
    randGen().seed(1)
    pop = smd_select_feature_sets(xor_ctable, feature_selection_parameters(target_size=target_size))
    got = [(score, sorted(fs)) for score, fs in pop]
    want = [(e["score"], sorted(e["fs"])) for e in _GOLDEN["smd"]["xor"][str(target_size)]]
    assert got == want


def test_smd_selection_edge_case_returns_empty_when_target_size_le_init_size():
    """DIVERGENCE(C++ stochastic_max_dependency.h:184-194), DO NOT FIX --
    see `stochastic_max_dependency_selection`'s own docstring: classic
    itself returns an EMPTY population, not the initial feature set, when
    the growth loop body never runs at all.
    """

    def scorer(fs):
        return 0.0

    pop = stochastic_max_dependency_selection(
        features={0, 1, 2}, init_features={0, 1}, scorer=scorer, num_features=1
    )
    assert pop == []


def test_shuffle_starts_from_the_sorted_feature_order_not_python_set_order(monkeypatch):
    """`std::vector<feature_id> shuffle(features.begin(), features.end())`
    (stochastic_max_dependency.h:122) copies from a `std::set<arity_t>`
    (feature-selection.h:124), so the vector handed to `random_shuffle` is
    SORTED ascending. This port built it with `list(features_set)`, which
    iterates a Python set in hash order.

    The two coincide for every input the sole caller can produce
    (`feature_set(range(arity))`; CPython iterates small contiguous ints
    ascending -- measured for 0..n up to n=1000, and for 0..49 minus a
    few). So this was right for the wrong reason rather than broken. They
    diverge as soon as the set mixes small and >=32 values, which is the
    case used here.

    `_random_shuffle` is stubbed to a no-op so the order the algorithm
    walks IS the starting order, making the difference directly
    observable rather than smeared through the RNG.
    """
    sparse = {1, 33, 65, 2}
    # Precondition: without this, the test proves nothing.
    assert list(sparse) != sorted(sparse), (
        "CPython's set order matches sorted() for these values; pick others"
    )

    monkeypatch.setattr(smd_module, "_random_shuffle", lambda seq: None)

    scored_singletons: list[int] = []

    def scorer(fs):
        fs = set(fs)
        if len(fs) == 1:
            scored_singletons.append(next(iter(fs)))
        return 0.0

    stochastic_max_dependency_selection(
        set(sparse), set(), scorer, num_features=1, threshold=0.0, top_size=4
    )

    assert scored_singletons == sorted(sparse), (
        f"features were walked in {scored_singletons}, i.e. Python-set order; "
        f"classic walks std::set order {sorted(sparse)}"
    )
