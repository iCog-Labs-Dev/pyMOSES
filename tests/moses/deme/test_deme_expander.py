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
from pymetta.moses.deme.feature_selector import feature_selector, feature_selector_parameters
from pymetta.moses.metapopulation.metapop_params import subsample_deme_filter_parameters
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.util.rng import randGen
from tools.check_fixture_manifests import SUPPORTED_SCHEMA_VERSIONS

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "deme_expander.json"
_PROBE_PATH = Path(__file__).resolve().parents[3] / "tools" / "cpp_probe" / "deme_expander_probe.cc"
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
    # Against the SUPPORTED set, not a hardcoded 1: these four fixture
    # tests each pinned `== 1`, so bumping the manifest schema (to add
    # the `toolchain` field) broke a test in an unrelated file for every
    # fixture that had been re-captured. The check that matters here is
    # the probe sha256 below; the version only needs to be one this repo
    # knows how to validate.
    assert manifest["schema_version"] in SUPPORTED_SCHEMA_VERSIONS
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
# fstor: on-the-fly feature selection is now REAL (P2 feature-selection
# port, pymetta.moses.deme.feature_selector). create_demeIDs/
# create_representations both honour a real feature_selector; only
# fstor.params.prune_xmplr=True and driving a real fstor through
# optimize_demes's own search loop remain gated -- see deme_expander.py's
# own module docstring. Subsample-deme filtering (n_subsample_demes > 1)
# is UNCHANGED and still raises NotImplementedError when driven -- "port
# the gate, raise inside the gated body" (S10 plan).
# ---------------------------------------------------------------------------


class _StampedCTable:
    """Stands in for the `CTable` `_cscorer.get_ctable()` returns, supplying
    only `get_timestamps()` -- the single method `subsample_by_time` calls
    on it (deme_expander.cc:89-90). Lets these tests drive a timestamp set
    of a chosen size without committing another CSV fixture.
    """

    def __init__(self, timestamps):
        self._timestamps = set(timestamps)

    def get_timestamps(self):
        return self._timestamps


def _dex_with(params: deme_parameters | None = None, filter_params=None) -> deme_expander:
    case = _GOLDEN["cases"]["xor_empty_and"]
    table = loadTable(str(_TABLE_DATA_DIR / case["dataset"]), case["target_feature"])
    bsc = ctruth_table_bscore(table.compressed())
    cscorer = behave_cscore(bsc)
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, case["arity"])
    return deme_expander(sig, rule, rule, cscorer, hc, params, filter_params)


def _boolean_dex_and_ctable(params: deme_parameters):
    """A deme_expander plus its own CTable, over trainingBoolean.dat
    (arity 3) -- for the fstor tests below, which need a ctable to build
    a real feature_selector over in the first place (the XOR fixture
    `_dex_with` above uses is arity 2, too small to usefully ignore a
    feature).
    """
    case = _GOLDEN["cases"]["boolean_empty_and"]
    table = loadTable(str(_TABLE_DATA_DIR / case["dataset"]), case["target_feature"])
    ctable = table.compressed()
    bsc = ctruth_table_bscore(ctable)
    cscorer = behave_cscore(bsc)
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, case["arity"])
    dex = deme_expander(sig, rule, rule, cscorer, hc, params)
    return dex, ctable


def test_create_demeIDs_uses_real_fstor_n_demes():
    """`create_demeIDs` reads `fstor.params.n_demes` off a REAL
    feature_selector now -- multi-ID branch when > 1 (deme_expander.cc:
    152), single-ID otherwise, matching the no-fstor case's own IDs
    (`demeID_t(n_expansions + 1)`, a bare `"<n>"` string). The dataset
    the feature_selector is built over is irrelevant to this method alone
    (it never reads `fstor._ctable`, only `fstor.params.n_demes`).
    """
    case = _GOLDEN["cases"]["xor_empty_and"]
    ctable = loadTable(str(_TABLE_DATA_DIR / case["dataset"]), case["target_feature"]).compressed()

    festor = feature_selector(ctable, feature_selector_parameters(n_demes=3))
    dex = _dex_with(deme_parameters(fstor=festor))
    dex.create_demeIDs(0)
    assert [str(d) for d in dex._demeIDs] == ["1.0", "1.1", "1.2"]

    festor_single = feature_selector(ctable, feature_selector_parameters(n_demes=1))
    dex2 = _dex_with(deme_parameters(fstor=festor_single))
    dex2.create_demeIDs(0)
    assert [str(d) for d in dex2._demeIDs] == ["1"]


def test_create_representations_builds_a_real_representation_from_fstor():
    """End-to-end: a real `feature_selector` selects a feature subset for
    a real exemplar, and `create_representations` builds `ignore_ops`
    from its COMPLEMENT. Numbers are the "$1", base_target_size=0 case
    from tests/fixtures/feature_selection.json's own "in_loop" cases
    (captured straight from classic's `feature_selector::operator()`,
    tools/cpp_probe/feature_selector_probe.cc) -- restrict_incorrect and
    ignore_xmplr_features at their defaults (True), increase_target_size
    at its default (True): `feature_selector` selects `{2}` (x3) for
    exemplar `$1` against trainingBoolean.dat, then
    `create_representations` inserts the exemplar's own feature `{0}`
    (x1) back in (deme_expander.cc:224-232, the non-`prune_xmplr` arm) --
    considered = `{0, 2}`, so `x2` (index 1) is the only ignored column.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    festor_params = feature_selector_parameters()
    festor_params.fs_params.target_size = 0  # matches the fixture case exactly
    festor = feature_selector(ctable, festor_params)
    dex._params.fstor = festor

    randGen().seed(1)
    # `create_demeIDs` first, exactly as `create_demes` calls the two
    # (deme_expander.cc:383-385). `create_representations` reaches
    # `log_selected_feature_sets`, whose `OC_ASSERT(sfps ==
    # _demeIDs.size())` (cc:63) compares against those ids -- classic
    # would trip it too on a bare `create_representations` call.
    dex.create_demeIDs(0)
    created = dex.create_representations(str_to_combo_tree("$1"))

    assert created is True
    assert len(dex._reps) == 1
    assert dex._ignore_cols_seq == [{1}]


def test_create_demes_succeeds_with_a_real_fstor():
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = feature_selector(ctable, feature_selector_parameters())

    randGen().seed(1)
    assert dex.create_demes(str_to_combo_tree("$1"), 0) is True
    assert len(dex._demes) == 1


def test_create_representations_no_longer_gates_fstor_prune_xmplr():
    """This test used to assert `NotImplementedError`. `prune_xmplr`
    (deme_expander.cc:308) is ported now, so the gate is gone -- kept here,
    inverted, so the removal is recorded at the site that used to pin it.
    The behaviour of the branch itself is covered in
    tests/moses/deme/test_prune_xmplr.py, against the C++ oracle.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = feature_selector(ctable, feature_selector_parameters(prune_xmplr=True))
    dex.create_demeIDs(0)
    assert dex.create_representations(str_to_combo_tree("$1")) is True


def test_optimize_demes_runs_with_a_real_fstor():
    """This test used to assert `NotImplementedError`. `optimize_demes`
    drives a real `fstor` through the whole search loop now; only its own
    `if (_params.fstor)` arm (deme_expander.cc:432-450) is
    documented-and-skipped, and that arm calls nothing this function's
    result depends on -- see its DIVERGENCE. Kept here, inverted, so the
    removal is recorded at the site that used to pin it. The end-to-end
    proof is tests/moses/main/test_enable_fs_cli.py's byte-for-byte oracle.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = feature_selector(ctable, feature_selector_parameters())

    randGen().seed(1)
    assert dex.create_demes(str_to_combo_tree("$1"), 0) is True
    dex.optimize_demes(200, 86400)
    assert dex.total_evals() > 0


# ---------------------------------------------------------------------------
# S10 branch review, [Important 6]: perceptions/actions/linear_contin used to
# be silently dropped on create_representations' default path with no gate
# at all -- deme_parameters still carries and accepts non-default values for
# all three, but representation() (Boolean-only) has no parameter for any of
# them. Gated the same way fstor/n_subsample_demes already are: raise rather
# than silently ignore.
# ---------------------------------------------------------------------------


def test_create_representations_raises_when_perceptions_is_set():
    params = deme_parameters(perceptions="not-a-real-perception-set")
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    with pytest.raises(NotImplementedError):
        dex.create_representations(str_to_combo_tree("and()"))


def test_create_representations_raises_when_actions_is_set():
    params = deme_parameters(actions="not-a-real-action-set")
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    with pytest.raises(NotImplementedError):
        dex.create_representations(str_to_combo_tree("and()"))


def test_linear_contin_false_is_accepted_on_a_boolean_signature():
    """`linear_contin=False` must NOT raise for a Boolean run.

    This test previously asserted the opposite, and that was wrong in a way
    the CLI made reachable. `linear_contin` is read at exactly one place in
    classic -- `if (!_linear_contin)` at build_knobs.cc:859, inside CONTIN
    knob building -- and `build_knobs` dispatches on the signature's output
    type, so a Boolean run never reads it.

    Meanwhile `deme_parameters`' struct default is `linear_contin(true)`
    (deme_params.h:49), but the CLI's `--linear-regression` defaults to
    FALSE (problem-params.cc:292) and is assigned unconditionally
    (`deme_params.linear_contin = linear_regression`, :2264). So a plain
    `-H it` run on a Boolean table -- which classic handles fine -- would
    have hit the old unconditional raise. Found by S13 Task 1 while wiring
    the CLI.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    params.linear_contin = False
    dex = _dex_with(params)
    # No raise: the Boolean path never consults the flag.
    dex.create_representations(str_to_combo_tree("and()"))


def test_create_representations_does_not_raise_at_the_defaults():
    """The gates above must not fire on the untouched defaults
    (`perceptions=None`, `actions=None`, `linear_contin=True`) -- this is
    the byte-identical-to-C++ case the module docstring's own DIVERGENCE
    note describes, not a case that should raise.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    assert params.perceptions is None
    assert params.actions is None
    assert params.linear_contin is True
    dex = _dex_with(params)
    assert dex.create_representations(str_to_combo_tree("and()")) is True


def test_optimize_demes_with_an_fstor_set_after_create_demes_is_not_special_cased():
    """The pathological direct-mutation scenario the module docstring's
    consolidation note names: `_params.fstor` set AFTER a successful
    create_demes() call, before optimize_demes(). This used to assert
    `NotImplementedError`; the guard it pinned is gone, and C++ has no
    guard here either -- the `if (_params.fstor)` at cc:432 is a plain
    branch, so a non-null `fstor` that `create_representations` never saw
    simply indexes `_ignore_cols_seq[i]`, which classic left EMPTY, and
    reads off the end of a `std::vector` (UB). This port's skipped arm
    never indexes it, so the run completes. Kept, inverted, to record that
    the gate went and that nothing replaced it.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    dex._params.fstor = "mutated-in-afterward"
    dex.optimize_demes(200, 86400)
    assert dex.total_evals() > 0


# ---------------------------------------------------------------------------
# Subsample-deme filtering (n_subsample_demes > 1). Every one of these
# raised NotImplementedError until the `ss-*` task. The end-to-end oracle
# for the whole family is tests/moses/main/test_ss_cli.py (classic's own
# stdout, byte for byte); what is checked here is the per-function shape
# and the RNG draw ORDER, which a whole-run comparison would only detect
# indirectly.
# ---------------------------------------------------------------------------


def test_create_demes_builds_n_subsample_demes_ss_demes_per_representation():
    """C++: deme_expander.cc:388-393. One `_demes` entry per
    representation, each holding `n_subsample_demes` ss-demes.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=3)
    dex = _dex_with(params, filter_params)
    randGen().seed(1)

    assert dex.create_demes(str_to_combo_tree("and()"), 0) is True
    assert len(dex._demes) == len(dex._reps) == 1
    assert len(dex._demes[0]) == 3


def test_create_demes_ss_deme_ids_use_n_expansions_not_n_expansions_plus_one():
    """C++: deme_expander.cc:392-393, `demeID_t(n_expansions, i, j)` --
    where `create_demeIDs` one line up uses `n_expansions + 1` (cc:154,
    156). Classic's own asymmetry, reproduced: an SS run's first
    expansion is `0.0.SS-j` where a non-SS run's is `1`.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params, subsample_deme_filter_parameters(n_subsample_demes=2))
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)
    assert [d.getID() for d in dex._demes[0]] == ["0.0.SS-0", "0.0.SS-1"]

    plain = _dex_with(params, subsample_deme_filter_parameters())
    randGen().seed(1)
    plain.create_demes(str_to_combo_tree("and()"), 0)
    assert [d.getID() for d in plain._demes[0]] == ["1"]


def test_subsample_by_row_partitions_the_uncompressed_indexes():
    """C++: deme_expander.cc:120-146. `n_ss_demes` DISJOINT sets of
    `usize // n_ss_demes` uncompressed row indexes each, all drawn from
    `[0, usize)`.
    """
    dex = _dex_with(None, subsample_deme_filter_parameters(n_subsample_demes=3))
    usize = dex._cscorer.get_ctable_usize()
    randGen().seed(1)

    groups = dex.subsample_by_row()

    assert len(groups) == 3
    assert all(len(g) == usize // 3 for g in groups)
    union = set().union(*groups)
    assert len(union) == 3 * (usize // 3), "segments must be disjoint"
    assert union <= set(range(usize))


def test_subsample_by_row_draw_order_is_the_std_random_shuffle_one():
    """The partition is a SLICING of the shuffled sequence, so reproducing
    the shuffle is reproducing the partition. `boost::random_shuffle(rng,
    gen)` compiles to `std::random_shuffle` on libstdc++ (see
    `deme_expander.__init__`'s own docstring for the
    `BOOST_NO_CXX98_RANDOM_SHUFFLE` measurement), i.e. exactly
    `_shuffle_with_gen` -- which has its own C++ oracle in
    tests/moses/scoring/test_ss_bscore.py. Recomputing it here from the
    same seed pins that this function draws from `randGen()` the same
    number of times, in the same order, with nothing else in between.
    """
    from pymetta.moses.scoring.ss_bscore import _shuffle_with_gen

    dex = _dex_with(None, subsample_deme_filter_parameters(n_subsample_demes=2))
    usize = dex._cscorer.get_ctable_usize()

    randGen().seed(7)
    groups = dex.subsample_by_row()

    randGen().seed(7)
    expected_idxs = list(range(usize))
    _shuffle_with_gen(expected_idxs, lambda i: randGen().randint_n(i))
    seg = usize // 2
    assert groups == [set(expected_idxs[:seg]), set(expected_idxs[seg : 2 * seg])]


def test_subsample_by_row_returns_one_empty_set_when_disabled():
    """C++: deme_expander.cc:141-143, the "subsampling deme is disabled"
    arm -- a single EMPTY set, and NO RNG draw at all.
    """
    dex = _dex_with()
    randGen().seed(3)
    before = randGen().randint()
    randGen().seed(3)
    assert dex.subsample_by_row() == [set()]
    assert randGen().randint() == before, "the disabled arm must not touch the RNG"


def test_subsample_by_time_contiguous_keeps_ascending_segments_and_draws_nothing():
    """C++: deme_expander.cc:95-98. `contiguous_time` (the DEFAULT) SKIPS
    the shuffle, so the segments are consecutive runs of the ASCENDING
    timestamp order -- `std::set`'s, which is why this port sorts rather
    than trusting its own `set`.
    """
    dex = _dex_with(None, subsample_deme_filter_parameters(n_subsample_demes=2))
    dex._filter_params.contiguous_time = True
    stamps = {"d5", "d1", "d4", "d2", "d3", "d6"}
    dex._cscorer.get_ctable = lambda: _StampedCTable(stamps)

    randGen().seed(11)
    before = randGen().randint()
    randGen().seed(11)
    groups = dex.subsample_by_time()

    assert groups == [{"d1", "d2", "d3"}, {"d4", "d5", "d6"}]
    assert randGen().randint() == before, "contiguous_time must not touch the RNG"


def test_subsample_by_time_non_contiguous_shuffles_the_ascending_sequence():
    """C++: deme_expander.cc:95-98's `if (!contiguous_time)` arm -- the
    SAME `random_shuffle_gen` `subsample_by_row` uses, over the sorted
    timestamp vector.
    """
    from pymetta.moses.scoring.ss_bscore import _shuffle_with_gen

    dex = _dex_with(None, subsample_deme_filter_parameters(n_subsample_demes=2))
    dex._filter_params.contiguous_time = False
    stamps = {"d5", "d1", "d4", "d2", "d3", "d6"}
    dex._cscorer.get_ctable = lambda: _StampedCTable(stamps)

    randGen().seed(11)
    groups = dex.subsample_by_time()

    randGen().seed(11)
    expected = sorted(stamps)
    _shuffle_with_gen(expected, lambda i: randGen().randint_n(i))
    assert groups == [set(expected[:3]), set(expected[3:6])]
    assert groups != [{"d1", "d2", "d3"}, {"d4", "d5", "d6"}], (
        "this seed must actually reorder, or the test proves nothing"
    )


def test_subsample_by_time_drops_the_remainder_segment():
    """`seg_size = tt_size / n_ss_demes` truncates and the loop stops at
    `n_ss_demes * seg_size` (cc:101-104), so timestamps beyond that are
    kept by EVERY ss-deme.
    """
    dex = _dex_with(None, subsample_deme_filter_parameters(n_subsample_demes=3))
    dex._filter_params.contiguous_time = True
    dex._cscorer.get_ctable = lambda: _StampedCTable({f"d{i}" for i in range(8)})

    groups = dex.subsample_by_time()

    assert [len(g) for g in groups] == [2, 2, 2]
    assert "d6" not in set().union(*groups)
    assert "d7" not in set().union(*groups)


def test_subsample_by_time_returns_one_empty_set_when_disabled():
    dex = _dex_with()
    assert dex.subsample_by_time() == [set()]


def test_optimize_demes_restricts_the_scorer_per_ss_deme_then_restores_it():
    """C++: deme_expander.cc:476-489 and the post-loop reset at cc:508-513.
    The reset is what makes `metapopulation.recompute_scores_over_whole_
    dataset` (subsample.py) see the whole table, so it must be the LAST
    `ignore_rows` call and it must carry the EMPTY set.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params, subsample_deme_filter_parameters(n_subsample_demes=2))
    dex._filter_params.by_time = False  # the CLI's own default (--ss-by-time=0)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    seen: list[tuple[str, frozenset]] = []
    real_rows = dex._cscorer.ignore_rows

    def spy_rows(idxs):
        seen.append(("rows", frozenset(idxs)))
        real_rows(idxs)

    dex._cscorer.ignore_rows = spy_rows
    dex.optimize_demes(200, 86400)

    assert len(seen) == 3, "one call per ss-deme, plus the whole-table reset"
    assert seen[-1] == ("rows", frozenset()), "the reset must come last and be empty"
    assert seen[0][1] and seen[1][1] and not (seen[0][1] & seen[1][1])


def test_optimize_demes_uses_ignore_rows_at_times_when_by_time_is_set():
    """C++: deme_expander.cc:478-484 picks `ignore_rows_at_times` over
    `ignore_rows` on `_filter_params.by_time`. The final whole-table reset
    (cc:512) is `ignore_rows` in BOTH cases.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=2)
    filter_params.by_time = True
    dex = _dex_with(params, filter_params)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    # The fixture table is untimestamped, so the REAL `subsample_by_time`
    # would return zero groups here and trip the cc:479 OC_ASSERT (its own
    # test, below). Substituting two empty groups isolates the ONE thing
    # this test is about: which of the two scorer methods cc:478-488
    # picks. The by-time path over a genuinely timestamped table is
    # covered end to end, byte for byte against classic, in
    # tests/moses/main/test_ss_cli.py.
    dex.subsample_by_time = lambda: [set(), set()]

    seen: list[str] = []
    real_rows, real_times = dex._cscorer.ignore_rows, dex._cscorer.ignore_rows_at_times
    dex._cscorer.ignore_rows = lambda idxs: (seen.append("rows"), real_rows(idxs))[1]
    dex._cscorer.ignore_rows_at_times = lambda ts: (seen.append("times"), real_times(ts))[1]

    dex.optimize_demes(200, 86400)

    assert seen == ["times", "times", "rows"]


def test_optimize_demes_asserts_when_subsample_by_time_cannot_partition():
    """C++: deme_expander.cc:479-482's `OC_ASSERT` -- an untimestamped
    table gives `subsample_by_time()` nothing to split, so it returns
    FEWER than `n_ss_demes` groups and classic aborts with its own
    "Are you sure your data has timestamps" message rather than indexing
    out of bounds. That is exactly what `--ss-by-time=1` on a table
    without `--timestamp-feature` does.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    filter_params = subsample_deme_filter_parameters(n_subsample_demes=2)
    filter_params.by_time = True
    dex = _dex_with(params, filter_params)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    assert dex._cscorer.get_ctable().get_timestamps() == set()
    with pytest.raises(AssertionError, match="timestamps"):
        dex.optimize_demes(200, 86400)


def test_optimize_demes_leaves_the_scorer_alone_when_subsampling_is_off():
    """The `n_ss_demes > 1` guards mean a default run never calls
    `ignore_rows` at all -- not even with the empty set.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params, subsample_deme_filter_parameters())
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    seen: list = []
    dex._cscorer.ignore_rows = lambda idxs: seen.append(idxs)
    dex._cscorer.ignore_rows_at_times = lambda ts: seen.append(ts)
    dex.optimize_demes(200, 86400)

    assert seen == []


def test_optimize_demes_divides_the_eval_budget_by_n_ss_demes():
    """C++: deme_expander.cc:405-408, `max_evals /= _demes.size()` then
    `/= n_ss_demes`. With one representation and three ss-demes each gets
    a third.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params, subsample_deme_filter_parameters(n_subsample_demes=3))
    dex._filter_params.by_time = False  # the CLI's own default (--ss-by-time=0)
    randGen().seed(1)
    dex.create_demes(str_to_combo_tree("and()"), 0)

    budgets: list[int] = []
    dex._optimize = lambda deme, scorer, max_evals, max_time: budgets.append(max_evals)
    dex.optimize_demes(200, 86400)

    assert budgets == [66, 66, 66]


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


# ---------------------------------------------------------------------------
# log_selected_feature_sets (deme_expander.cc:57) -- ported because it is NOT
# logging-only: it carries a live OC_ASSERT that classic actually hits.
# ---------------------------------------------------------------------------


def test_log_selected_feature_sets_assert_fires_when_n_demes_exceeds_one():
    """`OC_ASSERT(sfps == _demeIDs.size(), "The code is probably not robust
    enough")` (deme_expander.cc:63). `create_demeIDs` builds
    `fstor.params.n_demes` ids while every ported selection algorithm
    returns exactly one feature set, so `--fs-demes=2` can never satisfy
    it. Measured on classic 2026-09-10: exit 134,
    `opencog::AssertionException`, for `--fs-algo=simple` and
    `--fs-algo=random` alike. The end-to-end CLI form is pinned in
    tests/moses/main/test_enable_fs_cli.py.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = feature_selector(ctable, feature_selector_parameters(n_demes=2))

    randGen().seed(1)
    with pytest.raises(RuntimeError, match="probably not robust enough"):
        dex.create_demes(str_to_combo_tree("$1"), 0)


def test_log_selected_feature_sets_does_not_fire_at_the_default_n_demes():
    """The guard must not fire on the untouched default (`n_demes=1`) --
    every byte-identical oracle run in test_enable_fs_cli.py goes straight
    through it."""
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = feature_selector(ctable, feature_selector_parameters())
    randGen().seed(1)
    assert dex.create_demes(str_to_combo_tree("$1"), 0) is True


def test_fs_to_names_rejects_an_index_past_the_labels():
    """C++'s `ilabels[i]` (deme_expander.cc:53) is an unchecked
    `std::vector::operator[]`, and the out-of-range case is REACHABLE:
    `--fs-seed=xmplr` appends a synthetic `__exemplar_feature__` column at
    index `get_arity()` that `remove_useless_features`
    (feature_selector.cc:295, zero call sites in the whole classic tree)
    never removes. Classic SIGSEGVs there -- gdb-confirmed, see
    test_enable_fs_cli.py. Undefined behaviour cannot be ported, so this
    raises instead of answering.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex = _dex_with(params)
    assert dex.fs_to_names({0, 2}, ["a", "b", "c"]) == ["a", "c"]
    with pytest.raises(IndexError, match="outside the 3 input column"):
        dex.fs_to_names({0, 3}, ["a", "b", "c"])
    with pytest.raises(IndexError, match="outside the 3 input column"):
        dex.fs_to_names({-1}, ["a", "b", "c"])
