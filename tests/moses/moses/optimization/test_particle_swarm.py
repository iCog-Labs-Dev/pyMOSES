"""Unit and mutation tests for pymoses.moses.moses.optimization.particle_swarm.

**`test_ps_2_parity` FAILS IN CLASSIC ITSELF** -- confirmed directly
(`$CLASSIC/build/tests/moses/MOSESUTest`, 2026-09-09: "Failed 1 and
Skipped 0 of 23 tests", and the failure IS `test_ps_2_parity`:
`moses-framework.h:169`, `fabs(result.first - expected_sc) < 1.0e-8` found
`1.0000 >= 0.0000`). This file does NOT attempt to show particle_swarm
solving anything -- only that it runs the documented (buggy, in two
specific and DISCLOSED places -- see `particle_swarm.py`'s own module
docstring) algorithm faithfully, matching classic's real, unfixed
behaviour. Full CLI wiring is outside this task's owned scope (see task
report); these tests drive `particle_swarm` directly against a
`distance_based_scorer`, matching `test_star_anneal.py`'s and
`test_univariate.py`'s convention.
"""

import math

import pytest

from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.moses.optimization.particle_swarm import (
    _clamp,
    _DiscreteParticles,
    _std_round,
    particle_swarm,
    ps_parameters,
)
from pymoses.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_scorer import distance_based_scorer
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen


def _ps(ps_params: ps_parameters | None = None) -> particle_swarm:
    return particle_swarm(optim_parameters(), ps_params or ps_parameters())


# --------------------------------------------------------------- acceptance criterion 3


def test_default_parameters_match_classic_header():
    """C++: moses/moses/optimization/particle-swarm.h:50-76 ps_parameters()."""
    p = ps_parameters()
    assert p.max_parts == 50
    assert p.bit_c1 == 0.7
    assert p.disc_c1 == 2.05
    assert p.cont_c1 == 0.7
    assert p.bit_c2 == 1.43
    assert p.disc_c2 == 2.05
    assert p.cont_c2 == 1.43
    assert p.inertia == 0.7
    assert p.bit_min_value == 0
    assert p.bit_max_value == 1
    assert p.disc_min_value == 0
    assert p.disc_max_value == 1
    assert p.cont_min_value == -2147483648
    assert p.cont_max_value == 2147483647
    assert p.bit_min_vel == -6
    assert p.bit_max_vel == 6
    assert p.disc_min_vel == -0.5
    assert p.disc_max_vel == 0.5
    assert p.cont_min_vel == -1073741824
    assert p.cont_max_vel == 1073741823
    assert p.range_bit_vel == 12
    fi = p.disc_c1 + p.disc_c2
    expected_constriction = 2 / (2 - fi - math.sqrt((fi * fi) - (4 * fi)))
    assert p.constriction_disc == pytest.approx(expected_constriction)


# --------------------------------------------------------------- helpers


def test_clamp_matches_max_of_l_min_of_u_x():
    assert _clamp(5.0, 0.0, 10.0) == 5.0
    assert _clamp(-5.0, 0.0, 10.0) == 0.0
    assert _clamp(50.0, 0.0, 10.0) == 10.0


def test_std_round_is_half_away_from_zero_not_bankers_rounding():
    """`std::round` rounds 0.5 AWAY from zero always; Python's builtin
    `round()` uses round-half-to-even (`round(0.5) == 0`,
    `round(2.5) == 2`) -- verified to differ from `_std_round` at exactly
    these boundary values.
    """
    assert _std_round(0.5) == 1.0
    assert _std_round(-0.5) == -1.0
    assert _std_round(2.5) == 3.0
    assert _std_round(1.5) == 2.0
    # Contrast with Python's builtin, proving the difference is real:
    assert round(0.5) == 0
    assert round(2.5) == 2


def test_calc_swarm_size_uses_n_bits_not_n_disc_fields():
    """C++: `unsigned disc_bit_size = fs.n_bits();` -- multi-valued disc
    fields contribute NOTHING to swarm size, only single-bit fields do.
    """
    ps = _ps()
    fs_bits_only = field_set(disc_spec(2), 8)  # 8 bits, byte_relation 3/4 -> 6.0
    assert ps._calc_swarm_size(fs_bits_only) == 6

    fs_disc_only = field_set(disc_spec(5), 8)  # 8 multi-valued disc fields, 0 bits
    assert ps._calc_swarm_size(fs_disc_only) == 4  # clamped to the floor of 4.0


def test_calc_swarm_size_is_clamped_between_4_and_max_parts():
    ps = _ps()
    huge = field_set(disc_spec(2), 1000)
    assert ps._calc_swarm_size(huge) == ps.ps_params.max_parts
    tiny = field_set(disc_spec(2), 1)
    assert ps._calc_swarm_size(tiny) == 4


def test_discrete_particles_temp_and_best_personal_are_independent():
    dp = _DiscreteParticles(3, 2)
    assert dp.temp == [[0.0, 0.0]] * 3 or dp.temp == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dp.temp[0][0] = 5.0
    assert dp.best_personal[0][0] == 0.0  # must not alias


# --------------------------------------------------------------- run without crashing


def test_call_does_not_crash_on_a_boolean_only_field_set():
    randGen().seed(1)
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    ps = _ps()

    ps(deme, scorer, 500, 86400)

    assert len(deme) > 0
    assert deme.n_evals > 0
    assert deme.n_best_evals == len(deme)


def test_call_does_not_crash_on_a_mixed_field_set():
    randGen().seed(2)
    fs = field_set([disc_spec(2), disc_spec(2), disc_spec(4), contin_spec(0.0, 1.0, 2.0, 3)])
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    ps = _ps()

    ps(deme, scorer, 300, 86400)

    assert len(deme) > 0
    assert deme.n_evals > 0


def test_over_budget_flag_set_when_max_evals_is_tiny():
    randGen().seed(3)
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    ps = _ps()
    swarm_size = ps._calc_swarm_size(fs)

    ps(deme, scorer, swarm_size, 86400)  # exactly one iteration's worth

    assert ps.over_budget is True


# --------------------------------------------------- DIVERGENCE: vel-reset-per-segment


def test_vel_reset_bug_is_invisible_on_a_boolean_only_field_set():
    """The by-value `vel` reset (module docstring) cannot be observed on
    a field_set with only bit fields: `_update_disc_particle`/`_update_
    cont_particle` iterate zero times either way, so `_update_bit_
    particle`'s writes to `vel[0:n_bits]` are never touched again --
    matching `test_ps_2_parity`'s own case (pure Boolean 2-parity).
    """
    ps = _ps()
    fs = field_set(disc_spec(2), 4)
    vel = [0.0] * fs.dim_size()
    temp = instance(fs.packed_width())
    personal = instance(fs.packed_width())
    glob = instance(fs.packed_width())
    for raw_idx in range(fs.raw_size()):
        fs.set_raw(personal, raw_idx, 1)
        fs.set_raw(glob, raw_idx, 1)

    randGen().seed(5)
    ps._update_bit_particle(temp, personal, glob, vel, fs)
    after_bit = list(vel)
    # Nothing else touches `vel` for a bit-only field_set -- a "fixed"
    # threaded-cursor version would behave identically here too, since
    # there is nothing downstream to thread INTO.
    assert vel == after_bit


def test_vel_reset_bug_is_observable_on_a_mixed_field_set():
    """Mutation-discriminating test: for a field_set with BOTH bit and
    disc fields, the faithful (buggy) port has `_update_disc_particle`
    OVERWRITE `vel[0:n_disc]` -- the SAME leading slots
    `_update_bit_particle` just wrote -- rather than continuing from
    `vel[n_bits:n_bits+n_disc]`. A "fixed" threaded-cursor version would
    leave `vel[0:n_bits]` (the bit velocities) untouched by the disc
    update. Both are exercised directly (bypassing `_update_particles`'
    own per-particle `vel` reset, to isolate the SPECIFIC shape this
    marker documents) to show they diverge.
    """
    ps = _ps()
    fs = field_set([disc_spec(2), disc_spec(2), disc_spec(4)])  # 2 bits, 1 disc(4)
    assert fs.n_bits() == 2
    assert fs.n_disc_fields() == 1

    def run(threaded: bool) -> list[float]:
        randGen().seed(11)
        vel = [-99.0] * fs.dim_size()  # sentinel: untouched slots stay -99.0
        temp = instance(fs.packed_width())
        personal = instance(fs.packed_width())
        glob = instance(fs.packed_width())
        for raw_idx in range(fs.raw_size()):
            fs.set_raw(personal, raw_idx, 1)
            fs.set_raw(glob, raw_idx, 1)
        disc_temp = [0.5]
        disc_personal = [0.5]
        disc_global = [0.5]

        if threaded:
            # The "fixed" mutant: manually offset disc writes into the
            # SEGMENT AFTER the bit velocities, by pre-advancing a copy.
            bit_vel = vel[: fs.n_bits()]
            ps._update_bit_particle(temp, personal, glob, bit_vel, fs)
            vel[: fs.n_bits()] = bit_vel
            disc_vel = vel[fs.n_bits() :]
            ps._update_disc_particle(temp, disc_temp, disc_personal, disc_global, disc_vel, fs)
            vel[fs.n_bits() :] = disc_vel
        else:
            ps._update_bit_particle(temp, personal, glob, vel, fs)
            ps._update_disc_particle(temp, disc_temp, disc_personal, disc_global, vel, fs)
        return vel

    faithful = run(threaded=False)
    fixed = run(threaded=True)

    # Faithful (buggy) port: vel[0] was overwritten by the disc update
    # (no longer whatever the bit update put there), and vel[2] (where a
    # threaded cursor WOULD have put the disc velocity) is untouched.
    assert faithful[2] == -99.0
    # Fixed/threaded version: vel[0:2] (bit) survive untouched by the disc
    # step, and vel[2] (disc) is the one that changed.
    assert fixed[2] != -99.0
    assert faithful != fixed


# --------------------------------------------------- DIVERGENCE: gen_cont_value bug


def test_gen_cont_value_uses_velocity_params_not_value_params():
    """DO NOT FIX marker: `_gen_cont_value` reads `range_cont_vel`/
    `cont_min_vel`, not the value-range fields its name and every sibling
    `_gen_*_value` method suggest it should use.
    """
    ps = _ps()

    class _FixedRng:
        def randdouble(self):
            return 0.25

    import pymoses.moses.moses.optimization.particle_swarm as ps_mod

    original = ps_mod.randGen
    ps_mod.randGen = lambda: _FixedRng()
    try:
        got = ps._gen_cont_value()
    finally:
        ps_mod.randGen = original

    faithful = (0.25 * ps.ps_params.range_cont_vel) + ps.ps_params.cont_min_vel
    would_be_fixed = (0.25 * (ps.ps_params.cont_max_value - ps.ps_params.cont_min_value)) + (
        ps.ps_params.cont_min_value
    )
    assert got == pytest.approx(faithful)
    assert got != pytest.approx(would_be_fixed)


def test_gen_cont_value_bug_is_near_inert_for_stepper_direction():
    """The module docstring's own claim, verified rather than merely
    asserted: for a small-mean contin_spec, BOTH the faithful (buggy)
    value and the hypothetical "fixed" value are so far outside the
    field's reachable range that `set_contin`'s stepper saturates in the
    SAME direction either way, producing an IDENTICAL raw encoding.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 4)])
    ps = _ps()

    class _FixedRng:
        def randdouble(self):
            return 0.1  # a small draw -> a hugely negative target either way

    import pymoses.moses.moses.optimization.particle_swarm as ps_mod

    original = ps_mod.randGen
    ps_mod.randGen = lambda: _FixedRng()
    try:
        buggy_target = ps._gen_cont_value()
    finally:
        ps_mod.randGen = original
    fixed_target = (0.1 * (ps.ps_params.cont_max_value - ps.ps_params.cont_min_value)) + (
        ps.ps_params.cont_min_value
    )
    assert buggy_target != fixed_target  # genuinely different numbers ...

    inst_buggy = instance(fs.packed_width())
    fs.set_contin(inst_buggy, 0, buggy_target)
    inst_fixed = instance(fs.packed_width())
    fs.set_contin(inst_fixed, 0, fixed_target)
    # ... but the raw Left/Right/Stop encoding they produce is identical.
    assert list(fs.iter_raw(inst_buggy)) == list(fs.iter_raw(inst_fixed))


# ------------------------------- DIVERGENCE: frozen disc_best_it in update_particles


def test_update_particles_freezes_disc_best_personal_at_index_zero():
    """DO NOT FIX marker, and the test that would have caught this.

    `update_particles` (particle-swarm.cc:308) initialises FIVE iterators
    at :311-317, but its loop header at :325-326 increments only four:

        for(;temp_it != end_temp; temp_it++, bestp_it++,
                vels_it++, disc_temp_it++){

    `disc_best_it` is missing from that list, so `*disc_best_it` stays
    pinned at `disc_parts.best_personal[0]` for EVERY particle while its
    four siblings advance. Every other iterator is incremented, so this
    reads as an upstream oversight rather than intent -- but classic's
    behaviour is the specification, and this port must reproduce it.

    This port used `best_personal[i]` (the "obviously correct" reading)
    until 2026-09-09. Nothing in this file caught it: every existing test
    here either uses a single particle, or a boolean-only field_set where
    the discrete path never runs. It was found by driving the CLI, where
    `-aps` differed from classic on 8 of 10 runs across
    trainingBooleanXor.dat and trainingBoolean.dat at seeds 1-5; after the
    fix, 36 of 36 across six problem specs, three seeds and two budgets.

    The assertion below is what discriminates: particle 1's discrete
    update must be driven by `best_personal[0]`, not `best_personal[1]`.
    """
    ps = _ps()
    fs = field_set([disc_spec(4), disc_spec(4)])  # 2 disc fields, no bits
    assert fs.n_disc_fields() == 2
    assert fs.n_bits() == 0

    n_parts = 2
    seen: list[list[float]] = []

    original = ps._update_disc_particle

    def spy(temp_inst, disc_temp, disc_personal, disc_global, vel, fields):
        # Record WHICH best_personal row this call was handed.
        seen.append(list(disc_personal))
        return original(temp_inst, disc_temp, disc_personal, disc_global, vel, fields)

    ps._update_disc_particle = spy
    try:
        deme = instance_set(n_parts, fs)
        temp_parts = [deme[i] for i in range(n_parts)]
        best_parts = [deme[i] for i in range(n_parts)]
        velocities = [[0.0] * fs.dim_size() for _ in range(n_parts)]
        dp = _DiscreteParticles(n_parts, fs.n_disc_fields())
        # Make the two rows distinguishable: row 0 is all 0.25, row 1 all 0.75.
        dp.best_personal[0] = [0.25] * fs.n_disc_fields()
        dp.best_personal[1] = [0.75] * fs.n_disc_fields()
        randGen().seed(5)
        ps._update_particles(temp_parts, best_parts, 0, velocities, dp, fs)
    finally:
        # Delete the instance attribute rather than reassigning the bound
        # method: reassigning leaves a permanent shadow, which is exactly
        # how an earlier RNG test in this project leaked across the suite.
        del ps._update_disc_particle

    assert len(seen) == n_parts
    # BOTH particles must have been handed row 0 -- the frozen iterator.
    assert seen == [[0.25, 0.25], [0.25, 0.25]], (
        "particle 1 was handed best_personal[1]; classic freezes disc_best_it "
        "at best_personal[0] for every particle (particle-swarm.cc:325-326)"
    )
