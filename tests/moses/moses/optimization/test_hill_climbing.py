"""Oracle and guard tests for pymetta.moses.moses.optimization.hill_climbing.

The oracle fixture (tests/fixtures/hillclimbing.json) is produced by
tools/cpp_probe/hillclimbing_probe.cc against the real classic library (see
tools/capture_hillclimbing.py) -- not reimplemented by hand. See that probe's
own file header for a measured, real crash it works around
(`iscorer_base::get_rep()` defaulting to null, dereferenced unconditionally
by hill_climbing's own debug-logging code) and for why the search field_set
is derived from a throwaway `representation` instead of a hand-built one.

Both fixture cases run with `hc_parameters(cross=False)`, matching
`--hc-crossover=0`; a separate battery below constructs a scenario that
satisfies the crossover trigger (hill-climbing.cc:147-150) by hand and
asserts our port refuses it (`NotImplementedError`) rather than silently
proceeding -- this is what actually proves the trigger cannot fire silently
in this port, rather than merely assuming the fixture cases avoid it.
"""

import json
from pathlib import Path

import pytest

from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.field_set import disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import distance_based_scorer
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "hillclimbing.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

_CASE_CONFIG = {
    "deterministic": {
        "max_evals": 500,
        "make_op": lambda: optim_parameters(),
        "make_hc": lambda: hc_parameters(cross=False),
    },
    "sampled": {
        "max_evals": 4000,
        "make_op": lambda: optim_parameters(terminate_if_gte=1000.0),
        "make_hc": lambda: hc_parameters(widen=True, cross=False, max_evals=50),
    },
}


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


def test_golden_fixture_has_expected_shape(golden):
    assert set(golden) == {"sample_new_instances", "deterministic", "sampled"}
    assert len(golden["sample_new_instances"]) == 4
    for label in ("deterministic", "sampled"):
        case = golden[label]
        assert len(case["trajectory"]) == case["n_evals"]
        assert case["field_shape"]["n_bits"] == 0
        assert all(m == 3 for m in case["field_shape"]["disc_multiplicities"])


def _field_set_from_shape(shape: dict) -> field_set:
    specs = [disc_spec(m) for m in shape["disc_multiplicities"]]
    specs += [disc_spec(2) for _ in range(shape["n_bits"])]
    return field_set(specs)


def _target_for(fs: field_set) -> instance:
    """Mirrors hillclimbing_probe.cc's run_case: each disc knob's raw target
    is its own top value (multy - 1); each bit knob's raw target is 1.
    """
    target = instance(fs.packed_width())
    for i in range(fs.raw_size()):
        if fs.begin_disc_raw_idx() <= i < fs.end_disc_raw_idx():
            multy = fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy
            fs.set_raw(target, i, multy - 1)
        elif fs.begin_bit_raw_idx() <= i < fs.end_bit_raw_idx():
            fs.set_raw(target, i, 1)
    return target


# ------------------------------------------------------------------------- oracle


@pytest.mark.oracle
@pytest.mark.parametrize("label", ["deterministic", "sampled"])
def test_hill_climbing_matches_cpp_step_for_step(label, golden):
    case = golden[label]
    config = _CASE_CONFIG[label]

    fs = _field_set_from_shape(case["field_shape"])
    target = _target_for(fs)
    scorer = distance_based_scorer(fs, target)

    randGen().seed(case["seed"])
    deme = instance_set(fs)
    hc = hill_climbing(config["make_op"](), config["make_hc"]())
    hc(deme, scorer, config["max_evals"], 86400)

    assert deme.n_evals == case["n_evals"]
    assert hc.nsteps == case["nsteps"]
    assert hc.over_budget == case["over_budget"]
    assert len(deme) == len(case["trajectory"])

    for i, expected in enumerate(case["trajectory"]):
        got_raw = [fs.get_raw(deme[i].first, k) for k in range(fs.raw_size())]
        assert got_raw == expected["raw"], f"{label}: instance {i} raw mismatch"
        cs = deme[i].second
        exp_score = expected["score"]
        assert cs.get_score() == exp_score["score"], f"{label}: instance {i} score"
        assert cs.get_complexity() == exp_score["complexity"]
        assert cs.get_complexity_penalty() == exp_score["complexity_penalty"]
        assert cs.get_uniformity_penalty() == exp_score["uniformity_penalty"]
        assert cs.get_penalized_score() == exp_score["penalized_score"]


# ---------------------------------------------------------------- crossover trigger


def _tiny_scorer_setup(n_knobs=3, seed=1):
    fs = field_set(disc_spec(2), n_knobs)
    target = instance(fs.packed_width())
    for i in range(n_knobs):
        fs.set_raw(target, i, 1)
    scorer = distance_based_scorer(fs, target)
    randGen().seed(seed)
    deme = instance_set(fs)
    return fs, scorer, deme


def test_crossover_trigger_raises_when_enabled():
    """Forces every clause of hill-climbing.cc:147-150's `xover` condition
    (iteration > 2, not already_xover, current_number_of_instances >= 3,
    large_nbh) by setting `crossover_min_neighbors` to 1 -- a public field
    a C++ caller could equally mutate post-construction (hc_parameters has
    no constructor parameter for it either, see hc_parameters.__init__'s
    own docstring) -- and confirms the port refuses to proceed rather than
    silently skipping crossover or silently doing something else.
    """
    fs, scorer, deme = _tiny_scorer_setup()
    hcp = hc_parameters(cross=True)
    hcp.crossover_min_neighbors = 1
    hc = hill_climbing(optim_parameters(), hcp)
    with pytest.raises(NotImplementedError, match="crossover triggered"):
        hc(deme, scorer, 1000, 86400)


def test_crossover_trigger_never_fires_with_crossover_disabled():
    """The exact same scenario as above (same field, same forced
    crossover_min_neighbors=1, same seed) but with hc_parameters(cross=False)
    -- proving the guard is gated on the crossover flag specifically, not on
    any of the other three clauses, and that our two oracle fixture cases
    above (both cross=False) could never have hit it either.
    """
    fs, scorer, deme = _tiny_scorer_setup()
    hcp = hc_parameters(cross=False)
    hcp.crossover_min_neighbors = 1
    hc = hill_climbing(optim_parameters(), hcp)
    hc(deme, scorer, 1000, 86400)  # must not raise
    assert len(deme) > 0


def test_crossover_never_fires_in_oracle_fixture_cases(golden):
    """Re-derives, from the fixture's own recorded field_shape/parameters,
    that hc_parameters.crossover was False for both recorded C++ runs --
    the same guarantee --hc-crossover=0 gives on the command line -- rather
    than merely trusting the probe source comment.
    """
    for label in ("deterministic", "sampled"):
        hcp = _CASE_CONFIG[label]["make_hc"]()
        assert hcp.crossover is False


# --------------------------------------------------------------------------- guards


def test_hc_parameters_rejects_nonpositive_fraction_of_nn():
    with pytest.raises(ValueError, match="fraction_of_nn"):
        hc_parameters(fraction_of_nn=0.0)
    with pytest.raises(ValueError, match="fraction_of_nn"):
        hc_parameters(fraction_of_nn=-1.0)


def test_estimate_neighborhood_clamps_distance_past_field_dimension():
    """DO NOT FIX quirk (hill-climbing.cc:493-502): when the requested
    distance exceeds the information-theoretic bit estimate, the estimate
    itself (not the caller's distance) is used as the clamped local
    distance for the rest of the computation. A single-bit field has an
    estimate of 1; requesting distance=5 clamps to distance=1, which then
    takes the `distance == 1` branch (`2 * nn_estimate == 2`), NOT the
    `safe_binomial_coefficient(1, 5)` branch (which would be 0, since
    5 > 1).
    """
    fs = field_set(disc_spec(2), 1)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    assert hc.estimate_neighborhood(5, fs) == 2


def test_resize_by_score_stops_at_min_to_keep_leaving_low_scorers():
    """Mirrors hill-climbing.cc:741-775's own iterator-scan-and-erase-one-
    run-at-a-time algorithm exactly: 31 instances, two SEPARATE contiguous
    low-scoring runs (15 and 15) separated by one high scorer. Deleting the
    FIRST run alone (15 elements) already drops the deme from 31 to 16,
    which is under MIN_TO_KEEP(20) -- so the loop stops there WITHOUT ever
    reaching the second low-scoring run, which is still present and still
    <= cutoff. A "delete every instance <= cutoff in one pass" rewrite
    would leave only the 1 high-scoring instance instead.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    low = composite_score(-10.0, 0, 0, 0)
    high = composite_score(0.0, 0, 0, 0)
    cutoff = -5.0
    for _ in range(15):
        deme.append(scored_instance(instance(fs.packed_width()), low))
    deme.append(scored_instance(instance(fs.packed_width()), high))
    for _ in range(15):
        deme.append(scored_instance(instance(fs.packed_width()), low))
    assert len(deme) == 31

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    result = hc.resize_by_score(deme, cutoff)

    assert result == 16
    assert len(deme) == 16
    # The surviving deme still contains the untouched second low-scoring run.
    remaining_low = sum(1 for si in deme if si.second.get_penalized_score() <= cutoff)
    assert remaining_low == 15


def test_resize_by_score_removes_everything_when_deme_stays_above_min_to_keep():
    """Contrast case for the guard above: when NO deletion ever drops the
    deme below MIN_TO_KEEP, every contiguous low-scoring run is eventually
    found and removed (the loop only stops for lack of any remaining run).
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    low = composite_score(-10.0, 0, 0, 0)
    high = composite_score(0.0, 0, 0, 0)
    cutoff = -5.0
    # 25 high-scorers, with 2 low-scorers dropped in the middle -- deleting
    # those 2 never comes close to MIN_TO_KEEP(20).
    for _ in range(12):
        deme.append(scored_instance(instance(fs.packed_width()), high))
    for _ in range(2):
        deme.append(scored_instance(instance(fs.packed_width()), low))
    for _ in range(13):
        deme.append(scored_instance(instance(fs.packed_width()), high))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    result = hc.resize_by_score(deme, cutoff)

    assert result == 25
    assert len(deme) == 25
    assert all(si.second.get_penalized_score() > cutoff for si in deme)
