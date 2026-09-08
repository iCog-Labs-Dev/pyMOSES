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

S9 fix-round coverage note (2026-09): instrumenting both fixture runs showed
`has_improved` was True on EVERY iteration in both cases (12/12 and 41/41),
so `distance` never exceeded 1 and both runs exited solely via `max_evals`
-- the 4,500-instance corpus above walks exactly one control-flow path
through `__call__`'s termination tail (hill-climbing.cc:408-480). The
"Termination tail coverage" section below closes that gap with hand-built,
non-fixture scenarios (a constant-score "plateau" scorer forces
`has_improved` False so `distance` actually advances) -- each one is a
direct pytest-level regression test of this PORT's own already-established
control flow, in the same style as the pre-existing `resize_by_score`/
`crossover_trigger` batteries above (not a new C++ oracle fixture): the
underlying algorithm's fidelity to classic is what the fixture-based tests
above already establish, and these tests instead pin BRANCHES of that same
already-faithful algorithm that 69 independent mutation-testing runs found
were reachable by zero test in the repo (27 of those 69 mutants survived).
Each one is verified, by hand, to fail under the exact mutation it is
named for and pass once reverted -- see
`.superpowers/sdd/2026-09-07-pymetta-s10-metapopulation/s9-fix-coverage-report.md`
for the full mutation-by-mutation ledger.
"""

import hashlib
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
    assert set(golden) == {"_manifest", "sample_new_instances", "deterministic", "sampled"}
    assert len(golden["sample_new_instances"]) == 4
    for label in ("deterministic", "sampled"):
        case = golden[label]
        assert len(case["trajectory"]) == case["n_evals"]
        assert case["field_shape"]["n_bits"] == 0
        assert all(m == 3 for m in case["field_shape"]["disc_multiplicities"])


def test_fixture_manifest_records_what_produced_it(golden):
    """S9 fix-round hardening: `tests/fixtures/hillclimbing.json` carried
    no version, checksum, or provenance before this (CARRY-FORWARD.md Sec
    5's own named standing weakness) -- `tools/capture_hillclimbing.py`
    now injects a `_manifest` object recording the probe source path, this
    capture script's own path, and a UTC timestamp; this test just pins
    that the fields the next test depends on actually exist and are
    non-empty.
    """
    manifest = golden["_manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["probe_source"] == "tools/cpp_probe/hillclimbing_probe.cc"
    assert manifest["capture_script"] == "tools/capture_hillclimbing.py"
    assert len(manifest["probe_source_sha256"]) == 64  # hex sha256
    assert manifest["generated_at_utc"]


def test_fixture_manifest_matches_current_probe_source(golden):
    """The actionable half of "ci.sh cannot rebuild the probes"
    (CARRY-FORWARD.md Sec 5): this repo-local, compiler-free check cannot
    itself REBUILD the fixture (that needs the classic checkout, g++, and
    the installed moses-legacy prefix -- none of which `ci.sh` assumes are
    present), but it CAN prove the checked-in fixture and the checked-in
    probe source have not drifted apart, by recomputing the probe
    source's sha256 and comparing it against the one
    `tools/capture_hillclimbing.py` recorded when this fixture was last
    generated. An edit to `hillclimbing_probe.cc` that never re-runs the
    capture script now fails this test instead of silently shipping a
    fixture that no longer matches the probe that supposedly produced it.
    """
    probe_path = (
        Path(__file__).resolve().parents[4] / "tools" / "cpp_probe" / "hillclimbing_probe.cc"
    )
    actual_sha256 = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    assert golden["_manifest"]["probe_source_sha256"] == actual_sha256


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


# ------------------------------------------------------- termination tail coverage
#
# `__call__`'s termination tail (hill-climbing.cc:408-480) is the part the
# S9 fix-round mutation run found completely unpinned: both oracle fixture
# cases above have `has_improved` True on every iteration, so `distance`
# never exceeds 1 and every one of the six checks below is either never
# reached or reached only via the one branch the fixtures happen to take
# (over-budget via max_evals). Every test in this section uses a
# constant-score "plateau" scorer -- every instance in the deme scores
# identically -- which is NOT itself a C++-observable value (no probe
# produced it; it is a hand-built scenario chosen so `has_improved` is
# knowably False from the second iteration on, forcing `distance` to
# actually advance) but drives the SAME already-oracle-verified
# `hill_climbing`/`neighborhood_sampling` machinery the fixture tests above
# already establish is faithful to classic. Each assertion below was
# confirmed, by temporarily applying the exact mutation named in its
# docstring to pymetta/moses/moses/optimization/hill_climbing.py, that the
# test fails, then reverting and confirming it passes again -- see the
# coverage report this task writes to
# .superpowers/sdd/2026-09-07-pymetta-s10-metapopulation/s9-fix-coverage-report.md.


class _ConstScorer:
    """Not a ported C++ symbol -- every instance scores identically, so
    `has_improved` is knowably False as soon as more than one instance has
    ever been seen (score_improved's `best_score > prev_hi + min_score_improv`
    can never hold when every score is equal). Used only to force the
    plateau/widen control flow open for direct inspection.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, inst: instance) -> composite_score:
        return composite_score(self.value, 0, 0, 0)


def _plateau_fs() -> field_set:
    # 3 knobs of multiplicity 2 -- field_set.py's own `_build_disc_spec`
    # folds multiplicity-2 disc_specs into its bit range (width == 1), so
    # these are dim_size()==3, information_theoretic_bits()==3.0, matching
    # classic's own identical fold (field_set.cc's own width-1 handling) --
    # not a divergence, just why `max_dist`(default 4) clamps to
    # `max_distance == 3` for this field (`optim_parameters.max_distance`).
    return field_set(disc_spec(2), 3)


def _run_plateau(*, widen=False, step=False, terminate_if_gte=0.0, max_time=86400,
                  max_evals=100_000, value=-100.0, seed=1):
    fs = _plateau_fs()
    randGen().seed(seed)
    deme = instance_set(fs)
    hc = hill_climbing(
        optim_parameters(terminate_if_gte=terminate_if_gte),
        hc_parameters(cross=False, widen=widen, step=step),
    )
    hc(deme, _ConstScorer(value), max_evals, max_time)
    return hc, deme


def test_terminates_via_no_improvement_and_no_widen_search():
    """Pins hill-climbing.cc:470-473 (`if (!has_improved &&
    !hc_params.widen_search) break;`) AND, as a side effect, closes the
    "no non-over-budget exit is ever asserted" gap (`over_budget` starts
    `False` at hill-climbing.cc:60/hill_climbing.py's own `__call__` --
    every fixture case above ends over-budget, so nothing had ever checked
    it stays `False` on a normal exit).

    With `widen_search=False` (the documented default) and a constant
    scorer, iteration 1 evaluates only the initial instance (distance 0),
    which is a trivial "improvement" over the very-worst initial
    `best_score`; `number_of_new_instances == 1 and distance == 1` then
    fires the hill-climbing.cc:408-409 `continue`, deferring every
    remaining check for that iteration. Iteration 2 evaluates the true
    distance-1 neighborhood (3 neighbors on this field), finds no
    improvement (every score is identical), and hits this exact break --
    BEFORE `max_distance < distance` (distance is only 2, max_distance is
    3) can fire first.

    MUTATION: delete the break (or force the condition `False`) --
    verified this makes the run fall through to the max_distance break
    three iterations later instead, changing `nsteps` 2 -> 4 and
    `len(deme)` 4 -> 8. Confirmed by hand: reverting the mutation restores
    nsteps==2/len==4.
    """
    hc, deme = _run_plateau()
    assert hc.nsteps == 2
    assert hc.over_budget is False
    assert len(deme) == 4
    assert deme.n_evals == 4


def test_terminates_via_max_distance_when_widen_search_stalls():
    """Pins hill-climbing.cc:461-464 (`if (max_distance < distance)
    break;`). With `widen_search=True`, the not-improved break above is
    disabled (`not widen_search` is False), so a plateau keeps incrementing
    `distance` every iteration (elif not xover: distance += 1) until it
    exceeds `max_distance` (3, for this 3-knob field: `min(opt_params.
    max_dist=4, fs.dim_size()=3)`).

    MUTATION: delete the break (or force the condition `False`) --
    verified this makes the run continue widening forever with a constant
    scorer (nothing else can stop it: `widen_search=True` disables the
    line-470 break, `terminate_if_gte`'s default 0 is never <= this
    scorer's -100, and `single_step` is False), until `n_new_instances`'s
    exponentially-growing neighborhood estimate overflows to a Python
    `OverflowError` -- itself a test failure, just via an uncaught
    exception rather than a bad assertion. Confirmed by hand: reverting
    the mutation restores nsteps==4/len==8.
    """
    hc, deme = _run_plateau(widen=True)
    assert hc.nsteps == 4
    assert hc.over_budget is False
    assert len(deme) == 8


def test_terminates_via_terminate_if_gte_before_reaching_later_checks():
    """Pins hill-climbing.cc:427-431 (`if (opt_params.terminate_if_gte <=
    best_raw_score) break;`). `widen_search=True` disables the
    not-improved break (line 470) so that, if this check is mutated away,
    the run would NOT coincidentally stop at the same iteration via a
    different check -- it would keep widening (see the max_distance test
    above) until `max_distance < distance` fires several iterations later,
    changing both `nsteps` and `len(deme)`. A constant score of 0.0 (>=
    `terminate_if_gte=-1.0`) satisfies this check from the very first
    evaluated instance, but hill-climbing.cc:408-409's own `continue`
    (tested separately below) defers the check past iteration 1, so it
    first actually fires at iteration 2.

    MUTATION: delete the break (or force the condition `False`) --
    verified this changes nsteps 2 -> 4 and len(deme) 4 -> 8 (the same
    values the max-distance test above measures for the same reason: with
    both this check AND the not-improved check disabled/bypassed, only
    the max_distance break is left to stop the run). Confirmed by hand:
    reverting the mutation restores nsteps==2/len==4.
    """
    hc, deme = _run_plateau(widen=True, terminate_if_gte=-1.0, value=0.0)
    assert hc.nsteps == 2
    assert hc.over_budget is False
    assert len(deme) == 4


def test_terminates_via_max_time_budget():
    """Pins hill-climbing.cc:419-422 (`if (max_time <= elapsed.tv_sec) {
    over_budget = true; break; }`). `max_time=0` means the FIRST iteration
    to actually reach this check (iteration 2 -- iteration 1's single-
    instance transition is deferred by the line-408 `continue`, same as
    the terminate_if_gte test above) has `elapsed_tv_sec >= 0 == max_time`
    on any run fast enough to have zero whole seconds elapse (true of
    every test in this suite -- see hill_climbing.py's own DIVERGENCE note
    on the whole-seconds truncation).

    Deliberately does NOT merely check that the run stops quickly (a
    `max_time=0` run also stops just as fast via the ordinary
    not-improved-and-not-widen break two lines later, hill-climbing.cc:
    470-473, giving the identical nsteps/len(deme) either way) -- the
    discriminating observation is `over_budget`, which ONLY this break
    sets `True`.

    MUTATION: delete the break/over_budget assignment (or force the
    condition `False`) -- verified nsteps/len(deme) are UNCHANGED (2/4,
    the run still stops at the same point via the not-improved break) but
    `over_budget` becomes `False` instead of `True`. Confirmed by hand:
    reverting the mutation restores over_budget is True.
    """
    hc, deme = _run_plateau(max_time=0)
    assert hc.nsteps == 2
    assert hc.over_budget is True
    assert len(deme) == 4


def test_single_step_breaks_after_first_improving_iteration():
    """Pins hill-climbing.cc:476-480 (`if (hc_params.single_step &&
    has_improved) break;`). Needs an ACTUALLY improving run (a plateau
    scorer's `has_improved` is False everywhere this check could matter,
    since the line-408 `continue` already defers it past the one iteration
    that trivially improves) -- reuses the same `distance_based_scorer` +
    6-bit-knob field shape the oracle fixture's own "deterministic" case
    uses (S9 Task 2), at the same seed, so the underlying trajectory is
    already independently established as faithful; only the `single_step`
    flag differs here.

    MUTATION: delete the break (or force the condition `False`) --
    verified this makes the run continue to full convergence instead
    (nsteps 2 -> 7, len(deme) 7 -> 37 -- identical to the `single_step=
    False` run at the same seed). Confirmed by hand: reverting the
    mutation restores nsteps==2/len==7.
    """
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    for i in range(6):
        fs.set_raw(target, i, 1)
    scorer = distance_based_scorer(fs, target)

    randGen().seed(1)
    deme = instance_set(fs)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False, step=True))
    hc(deme, scorer, 500, 86400)

    assert hc.nsteps == 2
    assert hc.over_budget is False
    assert len(deme) == 7


def test_transitional_continue_defers_termination_checks_for_one_iteration():
    """Pins hill-climbing.cc:408-409 (`if (number_of_new_instances == 1 &&
    distance == 1) continue;`). Uses the SAME scenario as the
    terminate_if_gte test above (a constant score of 0.0, already >=
    `terminate_if_gte=-1.0` from the very first evaluated instance at
    iteration 1) but with `widen_search=False`, so that -- correctly --
    this `continue` is what stands between "the terminate_if_gte condition
    is already satisfiable" and "the run actually stops there": it defers
    every check for iteration 1 (where `number_of_new_instances == 1` and
    `distance` was just set to 1 by the improvement branch), so the deme
    ends up with MORE than the single initial instance once the run
    finally does stop at iteration 2.

    MUTATION: delete the continue (replace with `pass`) -- verified this
    makes the run stop dead after iteration 1 with `nsteps == 1` and
    `len(deme) == 1` (the terminate_if_gte check, no longer deferred,
    fires immediately). Confirmed by hand: reverting the mutation restores
    nsteps==2/len==4.
    """
    hc, deme = _run_plateau(terminate_if_gte=-1.0, value=0.0)
    assert hc.nsteps == 2
    assert hc.over_budget is False
    assert len(deme) == 4


# --------------------------------------------------- penalized-vs-raw-score coverage


def test_best_tracking_uses_penalized_score_and_sets_n_best_evals():
    """Pins hill-climbing.cc:251's `iscore = inst_cscore.
    get_penalized_score()` inside the per-instance best-tracking loop
    (the `get_penalized_score()` -> `get_score()` mutation survived
    because every fixture-recorded instance has zero complexity/uniformity
    penalty, so the two scores never differ), AND, as a side effect, pins
    `deme.n_best_evals = current_number_of_instances`
    (hill-climbing.cc:359) -- never asserted anywhere before this.

    A 1-knob, multiplicity-3 field with a scorer whose RAW-score ranking
    and PENALIZED-score ranking DISAGREE: raw value 0 scores highest by
    raw score (10) but carries a complexity_penalty of 20 (penalized -10,
    the worst of the three); values 1 and 2 carry zero penalty (penalized
    == raw). The correct algorithm's `best_score`/`ibest` must therefore
    prefer value 1 (penalized 5) over value 0 (penalized -10, despite its
    higher RAW score of 10) once value 1 is evaluated at iteration 2.
    `best_raw_score` (a separate, always-raw-score tracker,
    hill-climbing.cc:257-258) correctly reaches 10 regardless -- so the
    run's natural exit is via `terminate_if_gte`'s default 0 <= 10, NOT
    the mutation-sensitive path itself; the discriminating value is
    `deme.n_best_evals`.

    MUTATION (get_penalized_score -> get_score at hill-climbing.cc:243-
    244's Python line): verified `deme.n_best_evals` becomes 1 (value 0,
    the initial instance, stays "best" by raw score forever) instead of 3
    (value 1, correctly adopted as best once its penalized score 5 beats
    value 0's penalized -10, is adopted after 1 initial + 2 distance-1
    neighbors have been evaluated: 1 + 2 == 3).
    MUTATION (delete `deme.n_best_evals = current_number_of_instances`,
    i.e. line 594 becomes a no-op): verified `deme.n_best_evals` stays at
    its `instance_set` default of 0.
    Both confirmed by hand; reverting either restores n_best_evals == 3.
    """
    fs = field_set(disc_spec(3), 1)
    # value -> (raw score, complexity_penalty); penalized = score - penalty.
    table = {0: (10.0, 20.0), 1: (5.0, 0.0), 2: (1.0, 0.0)}

    class _PenaltyScorer:
        def __call__(self, inst: instance) -> composite_score:
            score, penalty = table[fs.get_raw(inst, 0)]
            return composite_score(score, 0, penalty, 0)

    randGen().seed(1)
    deme = instance_set(fs)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    hc(deme, _PenaltyScorer(), 1000, 86400)

    assert hc.nsteps == 2
    assert len(deme) == 3
    assert deme.n_best_evals == 3


def test_resize_by_score_uses_penalized_score_not_raw_score():
    """Pins hill-climbing.cc:748's `iscore = it->second.
    get_penalized_score()` inside `resize_by_score` (the second of the two
    `get_penalized_score()` -> `get_score()` survivors -- the other is
    covered by the best-tracking test above). An instance with raw score
    -3.0 (ABOVE the -5.0 cutoff -- safe by raw score) but a
    complexity_penalty of 5.0 (penalized -8.0 -- BELOW the cutoff) must be
    removed by the correct algorithm and survive under the mutation.

    MUTATION: `get_penalized_score()` -> `get_score()` -- verified the two
    penalized-bad instances (raw -3.0, safe by raw score) are no longer
    removed: `resize_by_score` returns 27 instead of 25. Confirmed by
    hand: reverting the mutation restores 25.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    high = composite_score(0.0, 0, 0, 0)
    penalized_bad = composite_score(-3.0, 0, 5.0, 0)  # raw -3 (safe), penalized -8 (bad)
    cutoff = -5.0
    for _ in range(12):
        deme.append(scored_instance(instance(fs.packed_width()), high))
    for _ in range(2):
        deme.append(scored_instance(instance(fs.packed_width()), penalized_bad))
    for _ in range(13):
        deme.append(scored_instance(instance(fs.packed_width()), high))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    result = hc.resize_by_score(deme, cutoff)

    assert result == 25
    assert len(deme) == 25


# -------------------------------------------------------------- estimate_neighborhood


def test_estimate_neighborhood_hits_binomial_branch_past_distance_one():
    """Pins hill-climbing.cc:516 (`return safe_binomial_coefficient(
    nn_estimate, distance);`) -- the ONLY existing test that exercises
    `estimate_neighborhood` with a distance other than 0 or 1
    (`test_estimate_neighborhood_clamps_distance_past_field_dimension`)
    deliberately clamps down to distance 1 and therefore never reaches
    this line at all, leaving it completely unpinned. A 3-knob,
    multiplicity-2 field has `information_theoretic_bits() == 3.0`
    (nn_estimate 3, not clamped at distance 2), so `estimate_neighborhood(
    2, fs)` reaches this exact line: `safe_binomial_coefficient(3, 2)
    == 3`.

    MUTATION (`nn_estimate + 1`): verified `safe_binomial_coefficient(4,
    2) == 6` instead of 3.
    MUTATION (`distance + 1`): verified `safe_binomial_coefficient(3, 3)
    == 1` instead of 3.
    Both confirmed by hand; reverting either restores 3.
    """
    fs = field_set(disc_spec(2), 3)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    assert hc.estimate_neighborhood(2, fs) == 3


# ----------------------------------------------------------------------- resize_deme


def test_resize_deme_returns_false_when_nothing_needs_resizing():
    """`resize_deme` (hill-climbing.cc:526-561) had ZERO tests before this
    battery -- `resize_by_score` (its own helper, called from inside it)
    had two, but the deme-size-management wrapper itself had none.
    Baseline: a small deme, no bad scorers, no RAM pressure (`_instance_
    bytes` defaults to 0, so `usage` is always 0), and `len(deme)` well
    under `max_allowed_instances` -- neither of `resize_deme`'s two `if`s
    should fire.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    for _ in range(10):
        deme.append(scored_instance(instance(fs.packed_width()), composite_score(0.0, 0, 0, 0)))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    did_resize = hc.resize_deme(deme, 0.0)

    assert did_resize is False
    assert len(deme) == 10


def test_resize_deme_resizes_by_score_when_bad_score_count_exceeds_threshold():
    """Pins hill-climbing.cc:800-801 (`if ((DONT_BOTHER_SIZE < bad_score_cnt)
    or ram_pressure)`)'s `bad_score_cnt > DONT_BOTHER_SIZE` clause: 600
    contiguous bad-scoring instances (well over DONT_BOTHER_SIZE == 500)
    followed by 30 good ones triggers `resize_by_score`, which removes the
    single contiguous bad run.

    MUTATION (drop the `bad_score_cnt > _DONT_BOTHER_SIZE` clause, leaving
    only `ram_pressure`): verified `did_resize` becomes `False` and
    `len(deme)` stays 630 (nothing removed, since `ram_pressure` is False
    here -- `_instance_bytes` defaults to 0). Confirmed by hand: reverting
    the mutation restores did_resize==True/len==30.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    for _ in range(600):
        deme.append(scored_instance(instance(fs.packed_width()), composite_score(-100.0, 0, 0, 0)))
    for _ in range(30):
        deme.append(scored_instance(instance(fs.packed_width()), composite_score(0.0, 0, 0, 0)))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    did_resize = hc.resize_deme(deme, 0.0)

    assert did_resize is True
    assert len(deme) == 30


def test_resize_deme_sorts_and_truncates_when_exceeding_max_allowed_instances():
    """Pins hill-climbing.cc:812-814 (`if ((hc_params.max_allowed_instances <
    deme.size() or ram_pressure)`)'s `max_allowed_instances < deme.size()`
    clause, and the sort-and-truncate that follows it. `max_allowed_
    instances` is a public field a C++ caller can mutate post-construction
    (hc_parameters has no constructor parameter for it, matching
    `crossover_min_neighbors`'s own precedent in the crossover-trigger
    tests above) -- set to 5 here so a 10-instance deme with distinct
    scores exceeds it without needing a genuinely large deme.

    MUTATION (drop the `max_allowed_instances < len(deme)` clause, leaving
    only `ram_pressure`): verified `did_resize` becomes `False` and
    `len(deme)` stays 10, unsorted. Confirmed by hand: reverting the
    mutation restores did_resize==True/len==5/top-5-descending.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    for v in range(10):
        score = composite_score(float(v), 0, 0, 0)
        deme.append(scored_instance(instance(fs.packed_width()), score))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    hc.hc_params.max_allowed_instances = 5
    did_resize = hc.resize_deme(deme, 9.0)

    assert did_resize is True
    assert len(deme) == 5
    assert [si.second.get_score() for si in deme] == [9.0, 8.0, 7.0, 6.0, 5.0]


def test_resize_deme_ram_pressure_forces_both_actions_below_their_own_thresholds():
    """Pins the `ram_pressure` clause of BOTH `if`s in `resize_deme`
    (hill-climbing.cc:800-801,812-814) independently of the count-based thresholds
    those `if`s otherwise gate on: `_instance_bytes`/`_total_RAM_bytes` are
    set here (as `__call__` would set them from real field/RAM values) so
    that `usage = instance_bytes * len(deme)` trivially exceeds
    `ACCEPTABLE_RAM_FRACTION * total_RAM_bytes`, forcing `ram_pressure`
    True on a deme with only 3 bad-scoring instances (far below
    DONT_BOTHER_SIZE==500) and only 13 total instances (far below
    max_allowed_instances==10000).

    MUTATION (drop `ram_pressure` from the FIRST `if`, hill-climbing.cc:
    534): verified the 3 bad instances (score -100.0, cutoff 0.0) are NOT
    removed -- `len(deme)` stays 13 instead of 10.
    MUTATION (drop `ram_pressure` from the SECOND `if`, hill-climbing.cc:
    541): verified the surviving instances are NOT sorted -- ascending
    order [1.0..10.0] is preserved instead of becoming descending.
    Both confirmed by hand; reverting either restores the assertions
    below.
    """
    fs = field_set(disc_spec(2), 1)
    deme = instance_set(fs)
    for v in range(1, 11):
        score = composite_score(float(v), 0, 0, 0)
        deme.append(scored_instance(instance(fs.packed_width()), score))
    bad = composite_score(-100.0, 0, 0, 0)
    for _ in range(3):
        deme.append(scored_instance(instance(fs.packed_width()), bad))

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    hc._instance_bytes = 1_000_000
    hc._total_RAM_bytes = 1
    did_resize = hc.resize_deme(deme, 5.0)

    assert did_resize is True
    assert len(deme) == 10
    expected_descending = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert [si.second.get_score() for si in deme] == expected_descending


# ------------------------------------------------------------------------ optim_stats


def test_total_evals_accumulates_across_multiple_calls():
    """Pins hill-climbing.cc:373 (`total_evals += number_of_new_
    instances`, inherited via `optim_stats`) -- never asserted anywhere.
    `total_evals`/`total_steps` are NOT reset by `__call__` (only
    `nsteps`/`demeID`/`over_budget` are, hill-climbing.cc:174-176), unlike
    `deme.n_evals` (reset fresh on each call, since it belongs to the
    `deme` argument, not `self`) -- so calling the SAME `hill_climbing`
    instance twice, on two fresh demes, must give `total_evals ==
    deme1.n_evals + deme2.n_evals`, strictly more than either call's own
    `n_evals` alone. This is what actually distinguishes "accumulates"
    from "merely gets set once and happens to match the last call".

    MUTATION (delete the `+=`, i.e. `total_evals` never updated): verified
    `hc.total_evals` stays 0 instead of reaching 74 (37 + 37). Confirmed
    by hand: reverting the mutation restores total_evals == 74.
    """
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    for i in range(6):
        fs.set_raw(target, i, 1)
    scorer = distance_based_scorer(fs, target)

    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))

    randGen().seed(1)
    deme1 = instance_set(fs)
    hc(deme1, scorer, 500, 86400)

    randGen().seed(2)
    deme2 = instance_set(fs)
    hc(deme2, scorer, 500, 86400)

    assert deme1.n_evals == 37
    assert deme2.n_evals == 37
    assert hc.total_evals == deme1.n_evals + deme2.n_evals == 74


# -------------------------------------------------------------------- hc_parameters


def test_hc_parameters_pins_every_documented_default():
    """`hc_parameters`'s five member-initializer-only fields (hill-
    climbing.h:47-50 -- not constructor parameters, only settable by
    mutating the constructed object afterward, see `hc_parameters.
    __init__`'s own docstring) plus the two constructor-parameter fields'
    OWN defaults were entirely unpinned before this test, `crossover`
    (`cross=True`) most of all: it is the module's own load-bearing
    documented DIVERGENCE (see the module docstring's DIVERGENCE marker)
    and was asserted by nothing. Values are `docs/CROSSOVER-INVENTORY.md`'s
    own verified-against-source C++ defaults (hill-climbing.h:41-128).

    MUTATION: change any single default below (e.g. `crossover_pop_size =
    120` -> `121`, or `resize_to_fit_ram = False` -> `True`) -- verified
    each one, independently, makes exactly its own assertion fail while
    every other assertion still passes (confirmed by hand for
    `crossover`, `crossover_pop_size`, and `resize_to_fit_ram`; the
    remaining fields are structurally identical plain-attribute
    assignments, so the same mutation-detection applies uniformly).
    """
    hcp = hc_parameters()
    assert hcp.widen_search is False
    assert hcp.single_step is False
    assert hcp.crossover is True
    assert hcp.crossover_pop_size == 120
    assert hcp.crossover_min_neighbors == 400
    assert hcp.max_nn_evals == 20000
    assert hcp.fraction_of_nn == 2.0
    assert hcp.score_range == 5.0
    assert hcp.max_allowed_instances == 10000
    assert hcp.resize_to_fit_ram is False
    assert hcp.prefix_stat_deme == ""
