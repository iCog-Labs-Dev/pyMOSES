"""Oracle and guard tests for pymetta.moses.deme.deme_expander/deme_params.

The oracle fixture (tests/fixtures/deme_expander.json) is produced by
tools/cpp_probe/deme_expander_probe.cc against the real classic
deme_expander -- driving a real Table/CTable/ctruth_table_bscore/
behave_cscore stack (trainingBooleanXor.dat, trainingBoolean.dat) and a real
hill_climbing optimizer (hc_parameters(cross=False)) -- via
tools/capture_deme_expander.py, not reimplemented by hand. Three cases:

* "xor_empty_and": arity-2 XOR data, seed exemplar `and()` (build_knobs'
  own logical_canonize/build_logical path on a childless `and`), seed 1.
* "boolean_or_and_seed": arity-3 boolean data, a NON-trivial exemplar
  `or(and($1 $2) $3)` (already contains real argument literals, so
  build_knobs decorates around existing structure rather than a bare
  seed), seed 42 -- exercises the mixed-multiplicity field_set case (5
  three-valued disc knobs AND 2 one-bit knobs, `n_bits=2`).
  * "boolean_empty_and": arity-3 boolean data, seed exemplar `and()`,
  seed 7, a larger max_evals budget.

Every trajectory comparison is EXACT (`==`), not `pytest.approx` --
see test_behave_cscore.py's own module docstring for why this is now
correct rather than merely convenient (`score_t` is genuinely narrowed to
IEEE-754 binary32 at every point classic's own arithmetic narrows), and
deme_expander_probe.cc's own `print_score` comment for why it prints with
`%.17g` (full double round-trip), not the `%.9g` hillclimbing_probe.cc uses
-- this fixture's `complexity_penalty` values are genuinely fractional
(`1/3.5`-derived), where `%.9g` measurably does NOT round-trip to the same
double bit pattern `_to_score_t`'s widened float32 produces.
"""

import hashlib
import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.deme.deme_expander import deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import subsample_deme_filter_parameters
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "deme_expander.json"
_PROBE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "cpp_probe" / "deme_expander_probe.cc"
)
_TABLE_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


# ---------------------------------------------------------------------------
# Fixture shape / provenance
# ---------------------------------------------------------------------------


def test_golden_fixture_has_expected_shape(golden):
    assert set(golden) == {"cases", "_manifest"}
    assert set(golden["cases"]) == {"xor_empty_and", "boolean_or_and_seed", "boolean_empty_and"}
    for case in golden["cases"].values():
        assert case["created"] is True
        assert case["n_reps"] == 1
        # Every knob is either a genuine multi-valued disc knob (listed in
        # disc_multiplicities) or a single-bit knob (counted only via
        # n_bits) -- "boolean_or_and_seed" has both (5 + 2 == 7).
        assert len(case["knobs"]) == (
            len(case["field_shape"]["disc_multiplicities"]) + case["field_shape"]["n_bits"]
        )
        assert case["deme_size"] == case["total_evals"] == case["n_evals"]
        assert len(case["trajectory"]) == case["deme_size"]


def test_fixture_is_non_vacuous():
    """CARRY-FORWARD.md Sec 5/6: a parametrized oracle over an empty corpus
    passes silently. `empty_parameter_set_mark = "fail_at_collect"`
    (pyproject.toml) already guards every `parametrize` collection
    project-wide; this pins the corpus's actual size too, so a future edit
    that accidentally empties `CASES` in tools/capture_deme_expander.py
    without touching this file's own `parametrize` list is still caught.
    """
    assert len(_GOLDEN["cases"]) == 3
    assert sum(len(c["trajectory"]) for c in _GOLDEN["cases"].values()) == 17 + 13 + 49


def test_fixture_manifest_records_what_produced_it(golden):
    manifest = golden["_manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["probe_source"] == "tools/cpp_probe/deme_expander_probe.cc"
    assert manifest["capture_script"] == "tools/capture_deme_expander.py"
    assert len(manifest["probe_source_sha256"]) == 64  # hex sha256
    assert manifest["generated_at_utc"]


def test_fixture_manifest_matches_current_probe_source(golden):
    """The actionable half of "ci.sh cannot rebuild the probes"
    (CARRY-FORWARD.md Sec 5) -- see test_hill_climbing.py's identically-named
    test for the full reasoning this mirrors.
    """
    actual_sha256 = hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
    assert golden["_manifest"]["probe_source_sha256"] == actual_sha256


# ---------------------------------------------------------------------------
# Oracle: create_demes + optimize_demes, matched step for step
# ---------------------------------------------------------------------------


def _build_dex(case: dict) -> deme_expander:
    table = loadTable(str(_TABLE_DATA_DIR / case["dataset"]), case["target_feature"])
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    bsc.set_complexity_coef(case["complexity_ratio"])
    cscorer = behave_cscore(bsc)

    rule = logical_reduction(())(2)

    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters()

    hc = hill_climbing(optim_parameters(), hc_parameters(widen=False, step=False, cross=False))

    sig = gen_signature(type_node.boolean_type, case["arity"])
    return deme_expander(sig, rule, rule, cscorer, hc, params, filter_params)


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN["cases"]))
def test_create_demes_matches_cpp(name, golden):
    case = golden["cases"][name]
    dex = _build_dex(case)
    exemplar = str_to_combo_tree(case["exemplar"])

    randGen().seed(case["seed"])
    created = dex.create_demes(exemplar, 0)

    assert created == case["created"]
    assert len(dex._reps) == case["n_reps"]

    rep = dex._reps[0]
    # combo_tree_to_str's own trailing-space convention (iostream_combo.py,
    # S7 -- see that module's docstring) vs. classic's plain `out <<
    # combo_tree` (verified against $COGUTIL's tree.h:3396
    # subtree_to_string: no trailing space) is a pre-existing, out-of-scope
    # discrepancy unrelated to deme_expander -- stripped on both sides so it
    # cannot mask a REAL divergence in THIS task's own scope.
    assert combo_tree_to_str(rep.exemplar()).rstrip() == case["decorated_exemplar"].rstrip()

    fs = rep.fields()
    assert [spec.multy for spec in fs.disc_and_bit()[: fs.n_disc_fields()]] == (
        case["field_shape"]["disc_multiplicities"]
    )
    assert fs.n_bits() == case["field_shape"]["n_bits"]
    assert fs.packed_width() == case["field_shape"]["packed_width"]

    assert len(rep.disc) == len(case["knobs"])
    for i, expected in enumerate(case["knobs"]):
        entry = rep.disc[i]
        assert entry.spec.multy == expected["multy"], f"{name}: knob {i} multiplicity"
        assert entry.knob.toStr() == expected["text"], f"{name}: knob {i} text"


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN["cases"]))
def test_optimize_demes_matches_cpp_step_for_step(name, golden):
    case = golden["cases"][name]
    dex = _build_dex(case)
    exemplar = str_to_combo_tree(case["exemplar"])

    randGen().seed(case["seed"])
    dex.create_demes(exemplar, 0)
    dex.optimize_demes(case["max_evals"], 86400)

    assert dex.total_evals() == case["total_evals"]
    deme = dex._demes[0][0]
    assert len(deme) == case["deme_size"]
    assert deme.n_evals == case["n_evals"]

    fs = dex._reps[0].fields()
    for i, expected in enumerate(case["trajectory"]):
        got_raw = [fs.get_raw(deme[i].first, k) for k in range(fs.raw_size())]
        assert got_raw == expected["raw"], f"{name}: instance {i} raw mismatch"
        cs = deme[i].second
        exp = expected["score"]
        assert cs.get_score() == exp["score"], f"{name}: instance {i} score"
        assert cs.get_complexity() == exp["complexity"], f"{name}: instance {i} complexity"
        assert cs.get_complexity_penalty() == exp["complexity_penalty"], (
            f"{name}: instance {i} complexity_penalty"
        )
        assert cs.get_uniformity_penalty() == exp["uniformity_penalty"], (
            f"{name}: instance {i} uniformity_penalty"
        )
        assert cs.get_penalized_score() == exp["penalized_score"], (
            f"{name}: instance {i} penalized_score"
        )


# ---------------------------------------------------------------------------
# free_demes / total_evals -- direct unit tests, independent of the oracle
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_free_demes_resets_state_and_permits_a_second_create_demes(golden):
    case = golden["cases"]["xor_empty_and"]
    dex = _build_dex(case)
    exemplar = str_to_combo_tree(case["exemplar"])

    randGen().seed(case["seed"])
    dex.create_demes(exemplar, 0)
    assert dex._reps and dex._demes

    dex.free_demes()
    assert dex._reps == []
    assert dex._demes == []
    assert dex._ignore_cols_seq == []

    # A second create_demes() call must succeed post-reset (this is exactly
    # what the three OC_ASSERTs at the top of create_demes exist to allow --
    # see test_create_demes_without_free_demes_raises below for what happens
    # WITHOUT the reset).
    randGen().seed(case["seed"])
    created_again = dex.create_demes(exemplar, 1)
    assert created_again is True
    assert dex._demeIDs == ["2"]  # demeID_t(n_expansions + 1) == demeID_t(2)


def test_create_demes_without_free_demes_raises(golden):
    case = golden["cases"]["xor_empty_and"]
    dex = _build_dex(case)
    exemplar = str_to_combo_tree(case["exemplar"])
    randGen().seed(case["seed"])
    dex.create_demes(exemplar, 0)

    with pytest.raises(RuntimeError):
        dex.create_demes(exemplar, 1)


def test_total_evals_sums_across_demes_and_ss_demes():
    """Direct unit test of total_evals()'s summation, independent of a real
    optimizer run -- constructs the `_demes` shape by hand (one
    representation, one ss-deme, matching this port's own single-deme
    scope) and pins n_evals is read from each `instance_set`, not
    recomputed.
    """
    case = _GOLDEN["cases"]["xor_empty_and"]
    dex = _build_dex(case)
    exemplar = str_to_combo_tree(case["exemplar"])
    randGen().seed(case["seed"])
    dex.create_demes(exemplar, 0)

    assert dex.total_evals() == 0  # nothing scored yet
    dex._demes[0][0].n_evals = 42
    assert dex.total_evals() == 42


# ---------------------------------------------------------------------------
# Gates: feature selection (fstor) and subsample-deme filtering
# (n_subsample_demes > 1) both default OFF and both raise NotImplementedError
# when driven -- "port the gate, raise inside the gated body" (S10 plan).
# ---------------------------------------------------------------------------


def _dex_with(params: deme_parameters | None = None, filter_params=None) -> deme_expander:
    case = _GOLDEN["cases"]["xor_empty_and"]
    table = loadTable(str(_TABLE_DATA_DIR / case["dataset"]), case["target_feature"])
    bsc = ctruth_table_bscore(table.compressed())
    cscorer = behave_cscore(bsc)
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, case["arity"])
    return deme_expander(sig, rule, rule, cscorer, hc, params, filter_params)


def test_create_demeIDs_raises_when_fstor_is_set():
    params = deme_parameters(fstor="not-a-real-feature-selector")
    dex = _dex_with(params)
    with pytest.raises(NotImplementedError):
        dex.create_demeIDs(0)


def test_create_representations_raises_when_fstor_is_set():
    params = deme_parameters(fstor="not-a-real-feature-selector")
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    with pytest.raises(NotImplementedError):
        dex.create_representations(str_to_combo_tree("and()"))


def test_create_demes_raises_when_fstor_is_set():
    params = deme_parameters(fstor="not-a-real-feature-selector")
    dex = _dex_with(params)
    with pytest.raises(NotImplementedError):
        dex.create_demes(str_to_combo_tree("and()"), 0)


def test_optimize_demes_raises_when_fstor_is_set_after_create_demes():
    """The pathological direct-mutation scenario the module docstring's
    consolidation note names: `_params.fstor` set AFTER a successful
    create_demes() call, before optimize_demes(). No real caller (C++ or
    this port) does this, but the guard exists in optimize_demes() too --
    this proves it independently of create_demes()'s own earlier gate.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    dex._params.fstor = "mutated-in-afterward"
    with pytest.raises(NotImplementedError):
        dex.optimize_demes(200, 86400)


def test_create_demes_raises_when_n_subsample_demes_greater_than_one():
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=2)
    dex = _dex_with(params, filter_params)
    with pytest.raises(NotImplementedError):
        dex.create_demes(str_to_combo_tree("and()"), 0)


def test_optimize_demes_raises_when_n_subsample_demes_greater_than_one_after_create_demes():
    """Isolates optimize_demes' OWN top-of-function guard from
    subsample_by_row/subsample_by_time's independent gate on the exact
    same condition (both are called, unconditionally, inside this same
    function's loop body -- see its own module-docstring DIVERGENCE note).
    Without neutralizing those two first, this scenario is NOT a clean
    isolation: measured directly -- disabling ONLY optimize_demes' own
    check (via mutation, see the task report's mutation log) still leaves
    this exact test passing, because execution falls through into the
    loop and subsample_by_row's OWN `if n_ss_demes > 1: raise` fires
    first, for a different (if equally correct) reason. Monkeypatching
    both to no-ops here makes optimize_demes' own check the ONLY thing
    that can raise, so a future mutation of exactly IT is what this test
    actually detects.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters()
    dex = _dex_with(params, filter_params)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    dex.subsample_by_row = lambda: [set()]
    dex.subsample_by_time = lambda: [set()]

    dex._filter_params.n_subsample_demes = 2
    with pytest.raises(NotImplementedError):
        dex.optimize_demes(200, 86400)


def test_subsample_by_time_raises_when_n_subsample_demes_greater_than_one():
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=2)
    dex = _dex_with(None, filter_params)
    with pytest.raises(NotImplementedError):
        dex.subsample_by_time()


def test_subsample_by_time_returns_one_empty_set_by_default():
    dex = _dex_with()
    assert dex.subsample_by_time() == [set()]


def test_subsample_by_row_raises_when_n_subsample_demes_greater_than_one():
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=2)
    dex = _dex_with(None, filter_params)
    with pytest.raises(NotImplementedError):
        dex.subsample_by_row()


def test_subsample_by_row_returns_one_empty_set_by_default():
    dex = _dex_with()
    assert dex.subsample_by_row() == [set()]


# ---------------------------------------------------------------------------
# perm_ratio: unset by default (deme_params.py's DIVERGENCE note) --
# create_representations reads it unconditionally, so an omitted assignment
# must surface as AttributeError, not a silently-substituted default.
# ---------------------------------------------------------------------------


def test_create_representations_raises_attributeerror_when_perm_ratio_unset():
    dex = _dex_with(deme_parameters())  # perm_ratio never assigned
    with pytest.raises(AttributeError):
        dex.create_representations(str_to_combo_tree("and()"))


def test_deme_expander_default_params_argument_also_leaves_perm_ratio_unset():
    """When `pa` is omitted entirely, `deme_expander.__init__` constructs
    its own fresh `deme_parameters()` -- which, per C++'s own constructor
    (deme_params.h:39-50), never sets `perm_ratio` either. Confirms the
    omitted-argument path and the explicit-`deme_parameters()`-argument
    path behave identically.
    """
    dex = _dex_with(None, None)
    with pytest.raises(AttributeError):
        dex.create_representations(str_to_combo_tree("and()"))


# ---------------------------------------------------------------------------
# Empty-representation branch: an arity-0 signature gives build_knobs
# nothing to attach knobs to (no arguments), so the exemplar decorates to a
# field_set with zero fields -- create_representations' own `if rep.fields
# ().empty(): continue` branch, and create_demes() returning False.
# ---------------------------------------------------------------------------


def test_create_demes_returns_false_for_an_empty_representation():
    # Dataset content is irrelevant here -- only the arity-0 signature below
    # matters; trainingBooleanXor.dat is reused just to get a real CTable.
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y")
    cscorer = behave_cscore(ctruth_table_bscore(table.compressed()))
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, 0)

    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = deme_expander(sig, rule, rule, cscorer, hc, params, subsample_deme_filter_parameters())

    randGen().seed(1)
    created = dex.create_demes(str_to_combo_tree("and()"), 0)

    assert created is False
    assert dex._reps == []
    assert dex._demes == []


# ---------------------------------------------------------------------------
# DO NOT FIX: optimize_demes divides by len(self._demes) with no zero
# guard, exactly as C++'s own `max_evals / _demes.size()` has none -- a
# caller who does not check create_demes()'s return value before calling
# optimize_demes() hits the same unguarded division classic does.
# ---------------------------------------------------------------------------


def test_optimize_demes_raises_zerodivisionerror_when_no_demes_were_created():
    table = loadTable(str(_TABLE_DATA_DIR / "trainingBooleanXor.dat"), "y")
    cscorer = behave_cscore(ctruth_table_bscore(table.compressed()))
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, 0)

    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = deme_expander(sig, rule, rule, cscorer, hc, params, subsample_deme_filter_parameters())

    randGen().seed(1)
    created = dex.create_demes(str_to_combo_tree("and()"), 0)
    assert created is False

    with pytest.raises(ZeroDivisionError):
        dex.optimize_demes(200, 86400)
