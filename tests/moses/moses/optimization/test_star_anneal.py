"""Unit and oracle-shaped tests for pymoses.moses.moses.optimization.star_anneal.

Oracle: `$CLASSIC/build/tests/moses/MOSESUTest`'s `test_sa_2_parity`
(`-Hpa -k2 -asa`) and `test_sa_1_mux` (`-Hmux -k1 -asa`) both PASS in
classic -- confirmed directly (`./MOSESUTest`, 2026-09-09: "Failed 1 and
Skipped 0 of 23 tests", the one failure being `test_ps_2_parity`, not
either of these). Full CLI-to-`simulated_annealing` wiring
(`--opt-algo=sa` -> an actually-constructed `simulated_annealing`) lives in
`moses_exec.py`/`problem_params.py`, outside this task's owned scope (see
task report) -- these tests instead drive `simulated_annealing` directly
against a `distance_based_scorer`, this project's own established
convention for optimizer-level tests (`test_hill_climbing_edge_cases.py`).

**Acceptance criterion 3**: `test_default_parameters_match_classic_header`
below constructs `simulated_annealing()` with NO arguments and pins every
field against star-anneal.h's own hardcoded defaults -- an oracle
configured by THIS port cannot detect the port's own defaults being wrong
(docs/CARRY-FORWARD.md 5.4), so this checks the defaults directly rather
than only through an already-configured scenario.
"""

import math

import pytest

from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.moses.optimization.star_anneal import sa_parameters, simulated_annealing
from pymoses.moses.representation.field_set import disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_scorer import distance_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen


def _sa(sa_params: sa_parameters | None = None) -> simulated_annealing:
    return simulated_annealing(optim_parameters(), sa_params or sa_parameters())


# --------------------------------------------------------------- acceptance criterion 3


def test_default_parameters_match_classic_header():
    """C++: moses/moses/optimization/star-anneal.h:46-51 sa_parameters()."""
    p = sa_parameters()
    assert p.init_temp == 30
    assert p.min_temp == 0
    assert p.temp_step_size == 0.5
    assert p.accept_prob_temp_intensity == 0.5
    assert p.max_new_instances == 100


# --------------------------------------------------------------- basic mechanics


def test_energy_is_minus_score():
    sa = _sa()
    assert sa.energy(3.0) == -3.0
    assert sa.energy(-2.0) == 2.0


def test_cooling_schedule_matches_formula():
    sa = _sa(sa_parameters())
    sa.sa_params.init_temp = 30
    assert sa.cooling_schedule(1.0) == pytest.approx(30.0 / 2.0)
    assert sa.cooling_schedule(3.0) == pytest.approx(30.0 / 4.0)


def test_cooling_schedule_rejects_non_positive_t():
    sa = _sa()
    with pytest.raises(ValueError):
        sa.cooling_schedule(0.0)
    with pytest.raises(ValueError):
        sa.cooling_schedule(-1.0)


def test_accept_probability_is_one_when_new_energy_is_lower():
    sa = _sa()
    assert sa.accept_probability(1.0, 5.0, 10.0) == 1.0


def test_accept_probability_uses_boltzmann_form_otherwise():
    sa = _sa()
    got = sa.accept_probability(5.0, 5.0, 2.0)
    assert got == pytest.approx(math.exp(0.0))
    got2 = sa.accept_probability(5.0, 1.0, 2.0)
    assert got2 == pytest.approx(math.exp((1.0 - 5.0) / 2.0))


def test_dist_temp_scales_linearly_between_bounds():
    sa = _sa()
    sa.max_distance = 5
    sa.sa_params.init_temp = 10
    sa.sa_params.min_temp = 0
    # at current_temp == init_temp -> full distance (max_distance-1)+1 == max_distance
    assert sa.dist_temp(10.0) == 5
    # at current_temp == min_temp -> just +1
    assert sa.dist_temp(0.0) == 1


# ------------------------------------------------- DIVERGENCE: dist_temp unsigned wrap


def test_dist_temp_wraps_to_uint_max_when_max_distance_is_zero():
    """DO NOT FIX marker: `max_distance` is C++ `unsigned`; `max_distance -
    1` with `max_distance == 0` wraps to `UINT_MAX`, not Python's `-1`.
    """
    sa = _sa()
    sa.max_distance = 0
    sa.sa_params.init_temp = 10
    sa.sa_params.min_temp = 0
    # (10-0)/(10-0) * (UINT_MAX) + 1 -- astronomically large, not ~1 or 0.
    result = sa.dist_temp(10.0)
    assert result > 1_000_000_000


def test_mutation_removing_unsigned_mask_gives_a_small_negative_result():
    """Mutation proof, performed directly: without the `& _UNSIGNED_MAX`
    mask, Python's arbitrary-precision `int` gives `max_distance - 1 ==
    -1` instead of wrapping, producing a SMALL (negative-driven) result
    instead of the huge one classic's real unsigned arithmetic gives.
    """
    span = 10.0  # init_temp - min_temp
    max_distance_minus_1_unmasked = 0 - 1  # the mutant: no & _UNSIGNED_MAX
    mutant_result = int(((10.0 - 0.0) / span) * max_distance_minus_1_unmasked + 1)
    assert mutant_result == 0  # (1.0 * -1) + 1 == 0, nowhere near UINT_MAX

    sa = _sa()
    sa.max_distance = 0
    faithful_result = sa.dist_temp(10.0)
    assert faithful_result != mutant_result
    assert faithful_result > 1_000_000_000


# --------------------------------------------------- DIVERGENCE: accept_probability bug


def test_actual_accept_prob_is_always_the_fixed_intensity_constant():
    """DO NOT FIX marker: `accept_probability(best_instance_energy,
    best_instance_energy, current_temp)` compares `best_instance_energy`
    against ITSELF -- always `energy_new < energy_old` False -- so
    `accept_probability` always returns `exp(0/temperature) == 1.0`, and
    `actual_accept_prob` always equals exactly
    `accept_prob_temp_intensity` (0.5 by default), independent of score,
    energy, or temperature. Verified directly against the method classic
    itself calls, not a re-derivation.
    """
    sa = _sa()
    for best_energy in (-100.0, 0.0, 42.0):
        for temp in (0.1, 1.0, 100.0):
            prob = sa.accept_probability(best_energy, best_energy, temp)
            assert prob == 1.0
            actual = sa.sa_params.accept_prob_temp_intensity * prob
            assert actual == sa.sa_params.accept_prob_temp_intensity


def test_mutation_passing_center_energy_would_make_prob_score_dependent():
    """Mutation proof: the "fixed" call (`accept_probability(best_energy,
    center_energy, temp)`) WOULD vary with the actual score gap -- shown
    directly to differ from the always-1.0 faithful result whenever
    `best_energy < center_energy` (a genuine improvement) at a finite
    temperature.
    """
    sa = _sa()
    best_energy, center_energy, temp = -10.0, 5.0, 2.0
    faithful = sa.accept_probability(best_energy, best_energy, temp)
    mutant = sa.accept_probability(best_energy, center_energy, temp)
    assert faithful == 1.0
    assert mutant == 1.0  # energy_new < energy_old here too, so also 1.0 in THIS case...
    # ...but for a case where the mutant's two arguments are EQUAL-ish but
    # the faithful call's forced self-comparison still trivially returns
    # 1.0, the two functions diverge whenever best < center is false:
    best_energy2, center_energy2 = 5.0, -10.0  # a "worse" new energy
    faithful2 = sa.accept_probability(best_energy2, best_energy2, temp)
    mutant2 = sa.accept_probability(best_energy2, center_energy2, temp)
    assert faithful2 == 1.0  # always 1.0: self-comparison
    assert mutant2 == pytest.approx(math.exp((center_energy2 - best_energy2) / temp))
    assert mutant2 != faithful2


def test_call_site_always_compares_best_energy_to_itself():
    """Integration-level pin (not just the standalone-method tests above):
    monkeypatches `accept_probability` to RECORD its own two energy
    arguments on every call made during a real `__call__` run, and
    asserts they are ALWAYS EQUAL -- the exact, load-bearing shape of the
    DIVERGENCE marker on `__call__`. A "fix" that changes the call site to
    pass `center_instance_energy` as the second argument would break this
    test (the two recorded arguments would differ whenever the best newly
    sampled instance's energy differs from the running center's), even
    though it would NOT break the standalone `accept_probability` unit
    tests above, which only ever call the method directly with hand-picked
    arguments and never exercise the real call site at all.
    """
    randGen().seed(4)
    fs = field_set(disc_spec(2), 8)
    target = instance(fs.packed_width())
    for raw_idx in range(fs.raw_size()):
        fs.set_raw(target, raw_idx, 1)
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    sa = _sa()

    recorded_calls: list[tuple[float, float]] = []
    real_accept_probability = simulated_annealing.accept_probability

    def spy(self, energy_new, energy_old, temperature):
        recorded_calls.append((energy_new, energy_old))
        return real_accept_probability(self, energy_new, energy_old, temperature)

    original = simulated_annealing.accept_probability
    simulated_annealing.accept_probability = spy
    try:
        sa(deme, scorer, 400, 86400)
    finally:
        simulated_annealing.accept_probability = original

    assert len(recorded_calls) > 0
    assert all(new == old for new, old in recorded_calls)


# --------------------------------------------------------------- end-to-end convergence


def test_converges_to_the_target_instance():
    """A non-all-zero target: `sample_new_instances`' own neighborhood
    sampling has no bias toward all-zero the way `generate_initial_
    sample`'s "bias towards the exemplar instance" step does
    (`test_univariate.py`'s own note on that step) -- but an all-ones
    target is used here too, for symmetry with `test_univariate.py` and
    to avoid relying on that distinction for THIS optimizer's own guard.
    """
    randGen().seed(1)
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    for raw_idx in range(fs.raw_size()):
        fs.set_raw(target, raw_idx, 1)
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    sa = _sa()

    sa(deme, scorer, 500, 86400)

    assert len(deme) > 0
    best = max(deme)
    assert best.second.get_score() == 0.0


def test_call_accepts_an_explicit_init_inst():
    randGen().seed(2)
    fs = field_set(disc_spec(2), 4)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    sa = _sa()
    start = instance(fs.packed_width())
    for raw_idx in range(fs.raw_size()):
        fs.set_raw(start, raw_idx, 1)  # every bit flipped from the target

    sa(deme, scorer, 200, 86400, start)
    assert deme.n_evals > 0


def test_n_evals_matches_deme_size_after_run():
    randGen().seed(3)
    fs = field_set(disc_spec(2), 3)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    sa = _sa()
    sa(deme, scorer, 1000, 86400)
    assert deme.n_evals == len(deme)
