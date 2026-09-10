"""Oracle and guard tests for pymetta.moses.deme.feature_selector -- the
in-loop `--enable-fs` entry point.

## Oracle: tools/cpp_probe/feature_selector_probe.cc -> tests/fixtures/feature_selection.json

The fixture's `"in_loop"` group runs classic's REAL
`feature_selector::operator()(const combo_tree&)` end to end (not just
`select_feature_sets` -- see test_simple.py for that piece alone), via a
`feature_selector_probe_access` subclass that also exposes the protected
`preprocess_params`/`build_fs_ctable` so the fixture records the POST-
preprocessing `target_size`, the `ignored_features` set, and the filtered
ctable's `uncompressed_size()`/`size()` -- not just the final selected set,
so a `build_fs_ctable` that filtered/blanked the WRONG rows but coincidentally
still selected the right features would still be caught. Four cases on
trainingBoolean.dat (arity 3, `y` target), all `randGen().seed(1)`
immediately before the call that draws:

* `and($1 $2)`, base_target_size=1, restrict_incorrect=True (default),
  ignore_xmplr_features=True (default), increase_target_size=True
  (default) -- effective target_size bumps to 3 (1 + the exemplar's own 2
  features), all 3 features selected regardless of score.
* Same exemplar, restrict_incorrect=False -- exercises the "keep every
  row, not just the exemplar's mistakes" branch of build_fs_ctable
  (uncompressed_size 8 vs. 3).
* `$1`, ignore_xmplr_features=False, increase_target_size=False --
  ignored_features stays EMPTY even though the exemplar uses feature 0;
  effective target_size stays at the base (no bump).
* `$1`, base_target_size=0 (defaults otherwise) -- effective target_size
  bumps from 0 to 1 (0 + the exemplar's own 1 feature); feature 0 is
  ignored (blanked, not dropped) in build_fs_ctable.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.moses.deme.feature_selector import (
    feature_selector,
    feature_selector_parameters,
    get_argument_abs_idx_from_zero_set,
)
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "feature_selection.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


@pytest.fixture()
def boolean_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed()


def _params_for(case: dict) -> feature_selector_parameters:
    params = feature_selector_parameters(
        restrict_incorrect=case["restrict_incorrect"],
        restrict_true=case["restrict_true"],
        ignore_xmplr_features=case["ignore_xmplr_features"],
        increase_target_size=case["increase_target_size"],
    )
    params.fs_params.target_size = case["base_target_size"]
    return params


@pytest.mark.oracle
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_call_matches_classic_end_to_end(golden, boolean_ctable, case_idx):
    case = golden["in_loop"][case_idx]
    params = _params_for(case)
    fs = feature_selector(boolean_ctable, params)
    xmplr = str_to_combo_tree(case["exemplar"])

    randGen().seed(1)
    top = fs(xmplr)

    assert sorted(top[0][1]) == case["selected"]
    assert len(top) == case["n_feature_sets"]
    assert params.fs_params.target_size == case["effective_target_size"]
    assert sorted(params.ignored_features) == case["ignored_features"]


@pytest.mark.oracle
@pytest.mark.parametrize("case_idx", [0, 1, 2, 3])
def test_build_fs_ctable_matches_classic_sizes(golden, boolean_ctable, case_idx):
    """Independent of the final selected set: proves preprocess_params +
    build_fs_ctable filter/blank exactly the rows classic does, not just
    that the downstream algorithm happens to land on the same answer.
    """
    case = golden["in_loop"][case_idx]
    params = _params_for(case)
    fs = feature_selector(boolean_ctable, params)
    xmplr = str_to_combo_tree(case["exemplar"])

    fs.preprocess_params(xmplr)
    fs_ctable = fs.build_fs_ctable(xmplr)

    assert fs_ctable.uncompressed_size() == case["fs_ctable_uncompressed_size"]
    assert len(fs_ctable) == case["fs_ctable_compressed_size"]


# ---------------------------------------------------------------------------
# preprocess_params: individual field effects, isolated from build_fs_ctable
# ---------------------------------------------------------------------------


def test_preprocess_params_bumps_target_size_by_exemplar_arity(boolean_ctable):
    params = feature_selector_parameters()
    params.fs_params.target_size = 5
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(str_to_combo_tree("and($1 $2 $3)"))
    assert params.fs_params.target_size == 8


def test_preprocess_params_does_not_bump_when_increase_target_size_is_false(boolean_ctable):
    params = feature_selector_parameters(increase_target_size=False)
    params.fs_params.target_size = 5
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(str_to_combo_tree("and($1 $2 $3)"))
    assert params.fs_params.target_size == 5


def test_preprocess_params_ignored_features_tracks_ignore_xmplr_features(boolean_ctable):
    xmplr = str_to_combo_tree("and($1 $2)")

    params_on = feature_selector_parameters(ignore_xmplr_features=True)
    feature_selector(boolean_ctable, params_on).preprocess_params(xmplr)
    assert params_on.ignored_features == {0, 1}

    params_off = feature_selector_parameters(ignore_xmplr_features=False)
    feature_selector(boolean_ctable, params_off).preprocess_params(xmplr)
    assert params_off.ignored_features == set()


def test_preprocess_params_init_xmplr_features_seeds_fs_params_and_forces_bump(boolean_ctable):
    """init_xmplr_features=True both appends the exemplar's own feature
    NAMES to fs_params.initial_features and forces increase_target_size
    on, even if the caller set it False (feature_selector.cc:83-90).
    """
    params = feature_selector_parameters(increase_target_size=False, init_xmplr_features=True)
    params.fs_params.target_size = 1
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(str_to_combo_tree("and($1 $3)"))
    assert params.fs_params.initial_features == ["x1", "x3"]  # ascending arg-index order
    assert params.increase_target_size is True
    assert params.fs_params.target_size == 3  # 1 + 2, the bump DID happen


def test_preprocess_params_xmplr_as_feature_appends_synthetic_feature_name(boolean_ctable):
    params = feature_selector_parameters(xmplr_as_feature=True)
    params.fs_params.target_size = 1
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(str_to_combo_tree("$1"))
    assert params.fs_params.initial_features == ["__exemplar_feature__"]
    assert params.fs_params.target_size == 3  # +1 exemplar's own arg, +1 synthetic feature


# ---------------------------------------------------------------------------
# build_fs_ctable: restrict_true / restrict_incorrect / boolean-only gate
# ---------------------------------------------------------------------------


def test_build_fs_ctable_restrict_true_keeps_only_positive_predictions(boolean_ctable):
    """`$1` predicts x1 itself; restrict_true keeps only rows where x1 is
    logical_true -- exactly half of trainingBoolean.dat's 8 rows.
    """
    params = feature_selector_parameters(
        restrict_incorrect=False, restrict_true=True, ignore_xmplr_features=False
    )
    fs = feature_selector(boolean_ctable, params)
    xmplr = str_to_combo_tree("$1")
    fs.preprocess_params(xmplr)
    fs_ctable = fs.build_fs_ctable(xmplr)
    assert fs_ctable.uncompressed_size() == 4.0


def test_build_fs_ctable_accepts_a_non_boolean_output_type(boolean_ctable):
    """This test used to assert `NotImplementedError`, and that was an
    OVER-GATE. C++ reads `cto` (feature_selector.cc:125) and branches on it
    in exactly one place -- inside `xmplr_as_feature`
    (feature_selector.cc:190-196), whose third arm is its own
    `OC_ASSERT(false, "Not implemented")`. Nothing else in
    `build_fs_ctable` cares. Refusing every non-boolean output type here
    made `-Hit` on a contin table and `-Hcluster` -- the
    `REGRESSION(table, ...)` sites at table-problems.cc:519,522,557 --
    reject command lines classic runs; both are pinned byte-for-byte in
    tests/moses/main/test_enable_fs_cli.py now.
    """
    sig = TypeSignature([type_node.boolean_type] * 3, type_node.contin_type)
    boolean_ctable.set_signature(sig)
    params = feature_selector_parameters()
    fs = feature_selector(boolean_ctable, params)
    fs_ctable = fs.build_fs_ctable(str_to_combo_tree("$1"))
    assert fs_ctable.get_signature().otype is type_node.contin_type
    assert len(fs_ctable) > 0


def test_build_fs_ctable_raises_only_for_xmplr_as_feature_on_an_odd_output(boolean_ctable):
    """The gate that IS C++'s: `OC_ASSERT(false, "Not implemented")` at
    feature_selector.cc:196, reached only when `xmplr_as_feature` has to
    push the exemplar's own prediction into the row and the output type is
    neither boolean nor contin."""
    sig = TypeSignature([type_node.boolean_type] * 3, type_node.enum_type)
    boolean_ctable.set_signature(sig)
    params = feature_selector_parameters(xmplr_as_feature=True)
    fs = feature_selector(boolean_ctable, params)
    with pytest.raises(NotImplementedError, match="Not implemented"):
        fs.build_fs_ctable(str_to_combo_tree("$1"))


def test_build_fs_ctable_subsamples_by_row(boolean_ctable):
    """`--fs-subsampling-ratio` alone (the BY-ROW branch,
    feature_selector.cc:217) now runs -- see
    tests/feature_selection/test_ctable_subsample.py and
    tests/moses/deme/test_feature_selector_diversity.py for the oracle
    coverage. Here: it changes the table, and it does not raise.
    """
    xmplr = str_to_combo_tree("$1")

    full = feature_selector(boolean_ctable, feature_selector_parameters())
    full.preprocess_params(xmplr)
    full_size = full.build_fs_ctable(xmplr).uncompressed_size()

    params = feature_selector_parameters(subsampling_ratio=0.5)
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(xmplr)
    randGen().seed(1)
    assert fs.build_fs_ctable(xmplr).uncompressed_size() < full_size


def test_build_fs_ctable_subsampling_by_time_is_a_no_op_without_timestamps(boolean_ctable):
    """The BY-TIME branch (feature_selector.cc:207-215) removes nothing on
    a table loaded without `--timestamp-feature`: `get_timestamps()`
    excludes the not-a-date sentinel, so it is empty, `rm_size` is 0, and
    `remove_rows_at_times({})` does nothing.

    MEASURED against the classic binary, which is why this is a no-op test
    and not a raise: `moses -i bool_a5_maj.csv -u y -m 300 -r 1 -W1
    --enable-fs=1 --fs-subsampling-ratio=0.5 --fs-subsampling-by-time=1`
    exits 0 with the same best candidate as the plain `--enable-fs=1` run
    (`-8 or(and($x2 $x3) $x1)`), where the by-ROW form of the same run
    returns a different one (`-6 or(and($x2 $x4) and($x3 $x5))`).
    """
    xmplr = str_to_combo_tree("$1")

    plain = feature_selector(boolean_ctable, feature_selector_parameters())
    plain.preprocess_params(xmplr)
    expected = plain.build_fs_ctable(xmplr).uncompressed_size()

    params = feature_selector_parameters(subsampling_ratio=0.5, subsampling_by_time=True)
    fs = feature_selector(boolean_ctable, params)
    fs.preprocess_params(xmplr)
    randGen().seed(1)
    assert fs.build_fs_ctable(xmplr).uncompressed_size() == expected


# ---------------------------------------------------------------------------
# select_top_feature_sets / __call__ plumbing
# ---------------------------------------------------------------------------


def test_select_top_feature_sets_slices_to_n_demes(boolean_ctable):
    params = feature_selector_parameters(n_demes=1)
    fs = feature_selector(boolean_ctable, params)
    pop = [(0.9, {0}), (0.5, {1}), (0.1, {2})]
    assert fs.select_top_feature_sets(pop) == [(0.9, {0})]


def test_select_top_feature_sets_ranks_when_diversity_pressure_is_positive(boolean_ctable):
    """`diversity_pressure > 0` takes the rank_feature_sets branch
    (feature_selector.cc:238-244) instead of the plain slice. A
    single-entry population comes back re-keyed on its own
    `get_penalized_score()`, which for one feature set (no other set to be
    penalized against) is its score narrowed to `score_t`.
    """
    params = feature_selector_parameters(diversity_pressure=0.5)
    fs = feature_selector(boolean_ctable, params)
    top = fs.select_top_feature_sets([(0.9, {0})])
    assert len(top) == 1
    assert top[0][1] == {0}
    assert top[0][0] == pytest.approx(0.9, abs=1e-7)
    assert top[0][0] != 0.9  # narrowed to binary32 by composite_score


def test_call_diversity_cap_is_unreachable_for_the_only_ported_algorithm(boolean_ctable):
    """`0 < diversity_cap < len(sf_pop)` (feature_selector.cc:283-284)
    truncates the population BEFORE select_top_feature_sets runs -- but
    `select_feature_sets` (the "simple" algorithm, the only one ported)
    ALWAYS returns a length-1 population (see test_simple.py's own
    `test_simple_select_feature_sets_always_returns_exactly_one_entry`),
    so `len(sf_pop) > diversity_cap` for any `diversity_cap >= 1` is
    never true, and `diversity_cap == 0` disables the check outright
    (feature_selector.cc:283's own `0 < params.diversity_cap`). This is
    classic's OWN behaviour with `algorithm="simple"`, not a gap this
    port introduces -- checked directly here by proving diversity_cap=1
    (the tightest possible non-disabling cap) changes NOTHING about
    `__call__`'s output versus diversity_cap=0 (disabled).
    """
    xmplr = str_to_combo_tree("$1")

    randGen().seed(1)
    uncapped = feature_selector(boolean_ctable, feature_selector_parameters(diversity_cap=0))(xmplr)

    randGen().seed(1)
    capped = feature_selector(boolean_ctable, feature_selector_parameters(diversity_cap=1))(xmplr)

    assert uncapped == capped


def test_get_argument_abs_idx_from_zero_set():
    tr = str_to_combo_tree("and($1 or($2 $1) $3)")
    assert get_argument_abs_idx_from_zero_set(tr) == {0, 1, 2}


def test_sample_enforced_features_is_empty_and_draw_free_by_default(boolean_ctable):
    """The default `enforce_features={}` means the loop runs zero
    iterations -- proven by NOT raising, since the underlying
    `randGen().randfloat()` primitive `biased_randbool` needs is itself
    NotImplementedError (pymetta.util.rng, not this file's own gate).
    """
    fs = feature_selector(boolean_ctable, feature_selector_parameters())
    assert fs.sample_enforced_features() == set()


def test_sample_enforced_features_raises_via_randfloat_when_non_empty(boolean_ctable):
    params = feature_selector_parameters(enforce_features={"x1": 1.0})
    fs = feature_selector(boolean_ctable, params)
    with pytest.raises(NotImplementedError):
        fs.sample_enforced_features()


# ---------------------------------------------------------------------------
# Counter::mode() quirk (DO NOT FIX) -- table.h's own `TimedCounter::mode`,
# via `Counter::mode`'s buggy loop (`cnt` never reassigned). Not exercised
# by either fixture table (every compressed row there has a single-entry
# counter -- unique input rows), so this is checked directly against a
# hand-built Counter matching the documented C++ behaviour, not an oracle.
# ---------------------------------------------------------------------------


def test_timed_counter_mode_quirk_returns_last_entry_exceeding_the_first(boolean_ctable):
    from pymetta.moses.deme.feature_selector import _timed_counter_mode

    # Ascending key order (Counter::mode iterates a std::map, sorted by
    # key): logical_true(1) < logical_false(2). First entry's count is 2.
    # logical_false's count (5) exceeds it -> key becomes logical_false.
    # Only two possible builtin values exist for a boolean output, so a
    # THIRD entry (needed to discriminate the buggy `cnt`-never-updates
    # loop from a correct running-max) is not constructible with real
    # vertex data -- see the plain-key test below for that.
    counter = Counter({builtin.logical_true: 2.0, builtin.logical_false: 5.0})
    assert _timed_counter_mode(counter) == builtin.logical_false


def test_timed_counter_mode_quirk_is_not_a_true_argmax():
    """The discriminating case: THREE entries, ascending key order,
    counts `1 < A`, `5 < B`, `3 < C` (`A < B < C` as keys). The TRUE
    maximum is `B` (count 5). The buggy loop instead returns `C`: `cnt`
    is captured ONCE from the first entry (`A`, count 1) and never
    reassigned, so both `B` (5 > 1) and `C` (3 > 1) satisfy `cnt < v`,
    and the LAST one visited (`C`, in ascending-key order) wins --
    overwriting the correct answer `B` found one iteration earlier. A
    real argmax (`cnt` reassigned on every update, as a "fix" would do)
    would return `B` -- proven by mutation: reassigning `cnt` inside the
    loop makes this test fail because the function then returns `B`, not
    `C`. `_timed_counter_mode` is generic over any orderable key (matching
    `Counter<T, CT>::mode()`'s own template parameter T) -- plain `int`
    keys exercise the algorithm without needing a third real `vertex`
    value the boolean domain cannot supply.
    """
    from pymetta.moses.deme.feature_selector import _timed_counter_mode

    counter = Counter({1: 1.0, 2: 5.0, 3: 3.0})  # keys already ascending
    assert _timed_counter_mode(counter) == 3


def test_timed_counter_mode_quirk_keeps_first_entry_when_nothing_exceeds_it(boolean_ctable):
    # First entry (logical_true, ascending order) already has the max
    # count -- nothing in the loop ever satisfies `cnt < v.second`.
    counter = Counter({builtin.logical_true: 5.0, builtin.logical_false: 2.0})
    from pymetta.moses.deme.feature_selector import _timed_counter_mode

    assert _timed_counter_mode(counter) == builtin.logical_true


def test_timed_counter_mode_single_entry_is_that_entrys_key(boolean_ctable):
    from pymetta.moses.deme.feature_selector import _timed_counter_mode

    counter = Counter({builtin.logical_true: 1.0})
    assert _timed_counter_mode(counter) == builtin.logical_true
