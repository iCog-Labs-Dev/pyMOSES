"""Oracle and guard tests for pymetta.feature_selection.scorers.mutual_info.

## Oracle: tools/cpp_probe/feature_selector_probe.cc -> tests/fixtures/feature_selection.json

The fixture's `"mi"` group runs classic's own `mutualInformation(const
CTable&, const FeatureSet&)` (table.h:1132) directly, one singleton feature
set per column, over trainingBoolean.dat (arity 3, x1/x2 tied, x3 highest)
and trainingBooleanXor.dat (arity 2, x1/x2 tied -- XOR's defining property).
Every comparison is exact equality on the full double, not
`pytest.approx` -- `%.17g` round-trips a double bit-for-bit (see
tools/cpp_probe/deme_expander_probe.cc's own comment on why `%.17g`, not
`%.9g`, and this probe follows the same convention), and this port's
`_entropy`/`_weighted_information` use the same `math.log2` classic's
`std::log2` resolves to on this platform -- there is no floating-point
slop to paper over with a tolerance.
"""

import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.feature_selection.scorers.mutual_info import (
    MICScorer,
    MICScorerCTable,
    MutualInformation,
    mutual_information,
    mutual_information_between_sets,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "feature_selection.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


@pytest.fixture(scope="module")
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


@pytest.fixture(scope="module")
def xor_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y").compressed()


@pytest.mark.oracle
@pytest.mark.parametrize("label,idx", [("x1", 0), ("x2", 1), ("x3", 2)])
def test_mutual_information_matches_classic_boolean(golden, boolean_ctable, label, idx):
    got = mutual_information(boolean_ctable, {idx})
    assert got == golden["mi"]["boolean"][label]


@pytest.mark.oracle
@pytest.mark.parametrize("label,idx", [("x1", 0), ("x2", 1)])
def test_mutual_information_matches_classic_xor(golden, xor_ctable, label, idx):
    got = mutual_information(xor_ctable, {idx})
    assert got == golden["mi"]["xor"][label]


@pytest.mark.oracle
def test_mutual_information_class_wrapper_matches_free_function(boolean_ctable):
    scorer = MutualInformation(boolean_ctable)
    for idx in range(3):
        assert scorer({idx}) == mutual_information(boolean_ctable, {idx})


def test_mutual_information_is_order_independent_over_the_feature_set(boolean_ctable):
    """`fs` is always re-sorted ascending internally (seq_filtered's own
    "in the order of the indexes" contract) -- a caller passing indices in
    a different order, or as a list vs. a set, must get the identical
    result.
    """
    a = mutual_information(boolean_ctable, [2, 0])
    b = mutual_information(boolean_ctable, (0, 2))
    c = mutual_information(boolean_ctable, {0, 2})
    assert a == b == c


def test_mutual_information_no_longer_gates_a_contin_output():
    """Both continuous branches are ported now (see
    tests/feature_selection/test_mutual_info_contin.py for the oracle
    tests). A contin output over discrete inputs takes the DISCRETE
    branch, after binning -- so on an empty ctable it reaches the same
    zero-counter path a boolean-output empty ctable does, rather than
    raising.

    This test used to assert NaN. That was the port's own invention: C++
    never tests `total`, it sizes each probability vector from its counter
    and `transform`s into it, so an empty counter divides zero times and
    `entropy()` returns exactly 0.0 (table.h:1233-1240, numeric.h:229).
    Measured against the real classic code -- see
    tests/feature_selection/test_mutual_info_empty.py and its oracle
    fixture."""
    sig = TypeSignature([type_node.boolean_type], type_node.contin_type)
    ctable = CTable("y", ["x1"], sig)  # empty: no rows to bin or count
    assert mutual_information(ctable, {0}) == 0.0


def test_mutual_information_raises_for_a_contin_input_with_a_boolean_output():
    """Inputs not all discrete AND a non-contin output is C++'s own
    `else`: `OC_ASSERT(0, "Type ... is not supported for mutual
    information")` (table.h:1298-1301). An OC_ASSERT is live in release
    builds, so it raises rather than returning."""
    sig = TypeSignature([type_node.contin_type], type_node.boolean_type)
    ctable = CTable("y", ["x1"], sig)
    with pytest.raises(AssertionError):
        mutual_information(ctable, {0})


def test_mutual_information_between_sets_is_symmetric(boolean_ctable):
    """MI(fs_l, fs_r) = H(fs_l) + H(fs_r) - H(fs_l union fs_r) is
    symmetric in (fs_l, fs_r) by construction (the union and the sum of
    the other two terms are both order-independent) -- true for ANY two
    feature sets, not just an oracle-matched pair, so this is a structural
    property check rather than a value comparison.
    """
    ab = mutual_information_between_sets(boolean_ctable, {0}, {2})
    ba = mutual_information_between_sets(boolean_ctable, {2}, {0})
    assert ab == ba


def test_mutual_information_between_sets_with_the_empty_set_is_zero(boolean_ctable):
    """MI(fs, {}) = H(fs) + H({}) - H(fs union {}) = H(fs) + 0 - H(fs) = 0:
    an empty feature set has exactly one possible "value" (the empty
    tuple), so its entropy is 0, and `fs union {} == fs` makes the other
    two terms cancel exactly -- true for ANY `fs`, unconditionally.
    """
    mi_with_empty = mutual_information_between_sets(boolean_ctable, {2}, set())
    assert mi_with_empty == pytest.approx(0.0, abs=1e-12)


def test_mic_scorer_ctable_matches_score_individual_features_style_confidence(boolean_ctable):
    """`MICScorerCTable`'s confidence factor saturates to (a double that
    prints as) 1.0 for these tiny fixture tables (`mi_confi=50`,
    `exp(-50) approx 1.9e-22`, `N/(N+1.9e-22) == N/N` in double precision
    for any `N` in the single digits) -- so this must equal bare MI
    exactly, not just approximately, for every singleton feature. This is
    also independently confirmed against classic's own `-l INFO`
    `log_selected_features(): <label> <score>` line for
    `--target-size=2` (fs-main.cc/feature-selection.cc, `fs_scorer<FS>`
    with `scorer="mi"` IS `MICScorerCTable`) -- see
    test_feature_selection.py's own `test_score_individual_features_*`.
    """
    scorer = MICScorerCTable(boolean_ctable, confi=50.0)
    for idx in range(3):
        assert scorer({idx}) == mutual_information(boolean_ctable, {idx})


def test_mic_scorer_ctable_confidence_formula_at_a_non_saturating_confi(boolean_ctable):
    """With a small `confi` the confidence factor is NOT ~1, so this
    exercises the actual formula (`N/(N + exp(-confi*|fs|))`), not just
    the saturated case above.
    """
    scorer = MICScorerCTable(boolean_ctable, confi=0.1)
    n = int(boolean_ctable.uncompressed_size())
    expected_confidence = n / (n + math.exp(-0.1 * 1))
    mi = mutual_information(boolean_ctable, {2})
    assert scorer({2}) == pytest.approx(mi * expected_confidence, rel=1e-12)


def test_micscorer_gives_the_same_score_for_a_generator_as_for_a_list():
    """`MICScorer.__call__` used `fs` twice -- once for the mutual
    information and once for `len(list(fs))` in the confidence factor.

    C++'s `FeatureSet` is a `std::set`, so classic can iterate it as often
    as it likes; this signature accepts any `Iterable`. Given a generator,
    the second pass saw an exhausted iterator, so the confidence factor
    was computed as if the feature set were EMPTY and the score came out
    silently wrong rather than raising.

    This is not reachable through `select_feature_sets` (no algorithm in
    `algo/` passes a generator, and none uses this class at all -- it is
    `eval-features`' scorer). It is pinned anyway because the failure is
    silent, and because the sibling `MICScorerCTable` already materialised
    its argument first, so the two classes disagreed on the same input.
    """
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y")
    scorer = MICScorer(table.itable, table.otable, confi=100.0)

    features = [0, 2]
    from_list = scorer(features)
    from_generator = scorer(i for i in features)

    assert from_generator == from_list
    # And the value is not the degenerate empty-set one, which is what the
    # bug produced -- otherwise this test would pass on a scorer that
    # ignored its argument entirely.
    assert from_list != scorer([])
