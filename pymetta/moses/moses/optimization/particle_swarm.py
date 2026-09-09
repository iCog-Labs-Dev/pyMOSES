"""C++: moses/moses/optimization/particle-swarm.{h,cc}

`ps_parameters` and `particle_swarm` -- particle swarm optimization over a
mixed bit/discrete/continuous `field_set`, per Nouaouria & Boukadoum,
"Improved global-best particle swarm optimization algorithm with mixed-
attribute data classification capability" (the header's own citation).

**`test_ps_2_parity` fails in classic itself** (`moses-framework.h:169`,
confirmed by the controller before this task started): particle swarm does
not solve 2-parity, and this port reproduces that trajectory, not a "fixed"
one that solves it. Do not chase it.

`velocity = std::vector<double>` (C++ `typedef std::vector<double>
velocity;`) is ported as a plain `list[float]`, one per particle, laid out
`[bit velocities][disc velocities][contin velocities]` in that order,
matching `fields.dim_size()` (`n_bits() + n_disc_fields() + len(contin())`).

# DIVERGENCE(C++ moses/moses/optimization/particle-swarm.h:244,264,290,296,
# 312): DO NOT FIX -- `update_bit_particle`, `update_disc_particle` and
# `update_cont_particle` each take their shared per-particle velocity
# vector's position as `velocity::iterator vel` -- BY VALUE, not `vel&` --
# and `update_particles` (particle-swarm.cc:308-337) calls all three in
# sequence with the SAME caller-side `vel` variable, never reassigning it
# between calls. Because an iterator argument is copied at each call
# boundary, each of the three functions independently starts its own
# internal walk from the position `vel` held when `update_particles`
# itself created it (`velocity::iterator vel = (*vels_it).begin();`,
# particle-swarm.cc:330) -- i.e. from index 0 of that particle's velocity
# vector, EVERY time, not threaded onward from where the previous
# function's internal walk stopped. `initialize_random_particle`
# (particle-swarm.cc:283-306) does NOT have this shape -- there `vit` is
# ONE local variable threaded through all three field-kind loops within a
# SINGLE function body, genuinely walking bit-then-disc-then-contin in
# order across the whole velocity vector. `update_particles` genuinely
# does not: for a field_set with more than one kind of field present, the
# three update calls all write into (an overlapping prefix of) the SAME
# leading indices of `vel`, and only whichever call runs LAST for a given
# index leaves its value there. This is invisible for a Boolean-only
# `field_set` (`test_ps_2_parity`'s own case: no disc or contin fields at
# all, so `update_disc_particle`/`update_cont_particle` iterate zero times
# and never touch `vel`) -- see `test_particle_swarm.py`'s own mutation
# test for a mixed field_set that makes the difference observable, proving
# this port reproduces the by-value reset rather than a threaded-cursor
# fix.

# DIVERGENCE(C++ moses/moses/optimization/particle-swarm.h:221-224): DO NOT
# FIX -- `gen_cont_value()` (used only to seed a PARTICLE'S INITIAL contin
# VALUE, in `initialize_random_particle`) reads `ps_params.range_cont_vel`
# and `ps_params.cont_min_vel` -- the VELOCITY range/floor -- not
# `cont_min_value`/`cont_max_value` (the VALUE range this function's own
# name and every sibling `gen_*_value` suggests it should use). Confirmed
# by reading particle-swarm.h:219-224 directly; not a transcription slip,
# and this fork's `git log` never touches particle-swarm.h. Practically
# near-inert either way: `cont_min_vel`/`cont_max_vel` (`INT_MIN/2`,
# `INT_MAX/2`) and `cont_min_value`/`cont_max_value` (`INT_MIN`, `INT_MAX`)
# are both astronomically wider than any real `contin_spec`'s reachable
# range, so `set_contin`'s stepper saturates toward the field's own step
# bound regardless of which huge target this bug hands it -- and this
# port's own oracle (`test_ps_2_parity`, Boolean-only) never reaches this
# code path at all (no contin fields). Ported verbatim.
"""

from __future__ import annotations

import math
import os
import time

from pymetta.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymetta.moses.moses.types import very_worst_score
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import iscorer_base
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.util.rng import randGen

__all__ = ["ps_parameters", "particle_swarm"]

velocity = list[float]  # C++: moses/moses/optimization/particle-swarm.h:37 velocity


def _get_total_ram_bytes() -> int:
    """C++: opencog/util/platform.cc:131 opencog::getTotalRAM

    Best-effort via `os.sysconf`, matching `hill_climbing.py`'s own
    `_get_total_ram_bytes` (duplicated here rather than imported: it is a
    module-private helper there, and this port's own `_total_RAM_bytes`
    -- see `particle_swarm.__init__` -- is never actually READ anywhere in
    `particle-swarm.cc` after being stored, so exact fidelity to the
    platform value has no observable consequence; kept for structural
    parity with the C++ member, not because anything downstream depends
    on its precision).
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 1 << 34  # 16 GiB fallback sentinel


def _clamp(x: float, lo: float, hi: float) -> float:
    """C++: opencog/util/numeric.h:199 clamp

    `max(l, min(u, x))` (the header's own body) -- a plain templated
    min/max composition, not `std::clamp` (the header's own comment: "TODO:
    replace the following by C++17 std::clamp").
    """
    return max(lo, min(hi, x))


def _std_round(x: float) -> float:
    """Not itself a ported C++ symbol -- `std::round`'s round-HALF-AWAY-
    FROM-ZERO tie-breaking, which Python's builtin `round()` (round-half-
    to-even) does not reproduce. Every `std::round` call in
    particle-swarm.cc (`cont2disc`, `calc_swarm_size`) goes through this
    instead.
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


class ps_parameters:
    """C++: moses/moses/optimization/particle-swarm.h:46 ps_parameters

    "PSO parameters hardcoded just for now." (the header's own comment.)
    Never CLI-configurable in classic (`ps_max_particles` is the sole
    exception -- stored, unwired, per this port's `problem_params.py`'s own
    documented scope decision, outside this task).
    """

    def __init__(self) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:50 ps_parameters"""
        self.max_parts = 50
        self.bit_c1 = 0.7
        self.disc_c1 = 2.05
        self.cont_c1 = 0.7
        self.bit_c2 = 1.43
        self.disc_c2 = 2.05
        self.cont_c2 = 1.43
        self.inertia = 0.7
        self.bit_min_value = 0
        self.bit_max_value = 1
        self.disc_min_value = 0
        self.disc_max_value = 1
        # DIVERGENCE(C++ moses/moses/optimization/particle-swarm.h:63-64):
        # `std::numeric_limits<int>::min()/max()` pinned to Python's exact
        # 32-bit `int` bounds (matching this project's established
        # `_UNSIGNED_MAX`-style ABI-pinning convention elsewhere).
        self.cont_min_value = -2147483648
        self.cont_max_value = 2147483647
        self.bit_min_vel = -6
        self.bit_max_vel = 6
        self.disc_min_vel = -self.disc_max_value / 2
        self.disc_max_vel = self.disc_max_value / 2
        self.cont_min_vel = self.cont_min_value // 2
        self.cont_max_vel = self.cont_max_value // 2
        self.range_bit_vel = self.bit_max_vel - self.bit_min_vel
        self.range_cont_vel = self.cont_max_vel - self.cont_min_vel
        fi = self.disc_c1 + self.disc_c2  # where c1 + c2 > 4
        self.constriction_disc = 2 / (2 - fi - math.sqrt((fi * fi) - (4 * fi)))


class _DiscreteParticles:
    """C++: moses/moses/optimization/particle-swarm.h:169
    particle_swarm::discrete_particles

    "Discrete values have to be outside the deme." (the header's own
    comment.) `temp`/`best_personal` are `vector<vector<double>>` in C++;
    ported as plain nested lists.
    """

    def __init__(self, part_size: int, disc_size: int) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:171
        discrete_particles::discrete_particles
        """
        self.temp: list[list[float]] = [[0.0] * disc_size for _ in range(part_size)]
        self.best_personal: list[list[float]] = [[0.0] * disc_size for _ in range(part_size)]


class particle_swarm(optimizer_base):
    """C++: moses/moses/optimization/particle-swarm.h:155 particle_swarm"""

    def __init__(
        self,
        op: optim_parameters | None = None,
        ps: ps_parameters | None = None,
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:157 particle_swarm

        `_total_RAM_bytes` is `const uint64_t`, computed once here (not
        per-call) via `getTotalRAM()` -- see `_get_total_ram_bytes`'s own
        docstring for why exact fidelity does not matter.
        """
        super().__init__(op)
        self._total_RAM_bytes = _get_total_ram_bytes()
        self.ps_params = ps if ps is not None else ps_parameters()
        self._instance_bytes = 0  # set per-call, see __call__

    # ---- swarm sizing --------------------------------------------------------

    def _calc_swarm_size(self, fs: field_set) -> int:
        """C++: moses/moses/optimization/particle-swarm.cc:239
        particle_swarm::calc_swarm_size

        "For each 4 bit i'll let it similar to a cont. Normally 3x or 4x
        of the dimension." (the header's own comments.) `fs.n_bits()` --
        NOT `fs.n_bits() + fs.n_disc_fields()` -- is deliberate, matching
        C++'s own `unsigned disc_bit_size = fs.n_bits();`: multi-valued
        discrete fields contribute nothing to swarm size here.
        """
        byte_relation = 3.0 / 4.0
        cont_relation = 3.0
        disc_bit_size = fs.n_bits()
        total = disc_bit_size * byte_relation + len(fs.contin()) * cont_relation
        total = _clamp(total, 4.0, float(self.ps_params.max_parts))
        return int(_std_round(total))

    # ---- particle initialization ----------------------------------------------

    def _gen_bit_value(self) -> bool:
        """C++: moses/moses/optimization/particle-swarm.h:217 gen_bit_value"""
        return randGen().randbool()

    def _gen_bit_vel(self) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:203 gen_bit_vel"""
        return (randGen().randdouble() * self.ps_params.range_bit_vel) - self.ps_params.bit_max_vel

    def _gen_disc_value(self) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:219 gen_disc_value"""
        return randGen().randdouble()

    def _gen_disc_vel(self) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:207 gen_disc_vel"""
        return randGen().randdouble() - self.ps_params.disc_max_vel

    def _gen_cont_value(self) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:221 gen_cont_value

        See module docstring's DIVERGENCE marker -- verbatim reuse of the
        VELOCITY range/floor, not the value range, is classic's own bug.
        """
        return (
            randGen().randdouble() * self.ps_params.range_cont_vel
        ) + self.ps_params.cont_min_vel

    def _gen_cont_vel(self) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:210 gen_cont_vel"""
        return (
            randGen().randdouble() * self.ps_params.range_cont_vel
        ) - self.ps_params.cont_max_vel

    def _cont2disc(self, cvalue: float, max_dvalue: int) -> int:
        """C++: moses/moses/optimization/particle-swarm.h:273 cont2disc

        "[min_cvalue, max_cvalue] == [0,1] and min_dvalue == 0 and
        max_dvalue == it.multy()" (the header's own comment, explaining
        why the general rounding-off formula collapses to this one).
        """
        return int(_std_round(cvalue * (max_dvalue - 1)))

    def _initialize_random_particle(
        self, new_inst: instance, vel: velocity, dist_values: list[float], fs: field_set
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:283
        particle_swarm::initialize_random_particle

        ONE shared cursor (`vidx`, standing in for C++'s `vit`) threads
        through bit, then disc, then contin, all within this single
        function body -- contrast `update_particles`' three SEPARATE
        by-value `vel` copies (module docstring's DIVERGENCE marker).
        """
        vidx = 0
        for raw_idx in range(fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()):
            fs.set_raw(new_inst, raw_idx, 1 if self._gen_bit_value() else 0)
            vel[vidx] = self._gen_bit_vel()
            vidx += 1
        cidx = 0
        for raw_idx in range(fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()):
            multy = fs.disc_and_bit()[fs.raw_to_disc_idx(raw_idx)].multy
            dist_values[cidx] = self._gen_disc_value()
            fs.set_raw(new_inst, raw_idx, self._cont2disc(dist_values[cidx], multy))
            vel[vidx] = self._gen_disc_vel()
            vidx += 1
            cidx += 1
        for spec_idx in range(len(fs.contin())):
            fs.set_contin(new_inst, spec_idx, self._gen_cont_value())
            vel[vidx] = self._gen_cont_vel()
            vidx += 1

    def _initialize_particles(
        self,
        swarm_size: int,
        best_parts: deme_t,
        velocities: list[velocity],
        disc_parts: _DiscreteParticles,
        fields: field_set,
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:269
        particle_swarm::initialize_particles
        """
        for i in range(swarm_size):
            new_inst = instance(fields.packed_width())
            self._initialize_random_particle(
                new_inst, velocities[i], disc_parts.best_personal[i], fields
            )
            # C++: `best_parts.push_back(new_inst);` -- implicit
            # `scored_instance(const instance&)` (NOT `explicit`,
            # scored_instance.h:39), same conversion `univariate.py`'s
            # own `deme[i] = scored_instance(inst)` documents.
            best_parts.append(scored_instance(new_inst))

    # ---- per-iteration particle update -----------------------------------------

    def _update_bit_vel(
        self, vel: velocity, k: int, temp_bit: int, personal_bit: int, global_bit: int
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:244 update_bit_vel

        "Bool convertion to int: false to 0, true to 1. Bit vel hasn't
        inertia." (the header's own comment.)
        """
        rng = randGen()
        v = vel[k]
        v += (rng.randdouble() * (int(personal_bit) - int(temp_bit))) + (
            rng.randdouble() * (int(global_bit) - int(temp_bit))
        )
        vel[k] = _clamp(v, self.ps_params.bit_min_vel, self.ps_params.bit_max_vel)

    def _new_bit_value(self, vel: float) -> bool:
        """C++: moses/moses/optimization/particle-swarm.h:254 new_bit_value

        "Sigmoid" (the header's own comment).
        """
        return randGen().randdouble() < (1.0 / (1.0 + math.exp(-vel)))

    def _update_bit_particle(
        self, temp: instance, personal: instance, glob: instance, vel: velocity, fs: field_set
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:339
        particle_swarm::update_bit_particle

        See module docstring's DIVERGENCE marker: `k` starts at 0 on
        EVERY call, matching classic's by-value `velocity::iterator`
        reset.
        """
        k = 0
        for raw_idx in range(fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()):
            temp_bit = fs.get_raw(temp, raw_idx)
            personal_bit = fs.get_raw(personal, raw_idx)
            global_bit = fs.get_raw(glob, raw_idx)
            self._update_bit_vel(vel, k, temp_bit, personal_bit, global_bit)
            fs.set_raw(temp, raw_idx, 1 if self._new_bit_value(vel[k]) else 0)
            k += 1

    def _update_disc_vel(
        self, vel: velocity, k: int, temp_val: float, personal_val: float, global_val: float
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:264 update_disc_vel

        "Disc vel hasn't inertia, but has Constriction Factor" (the
        header's own comment.)
        """
        rng = randGen()
        v = vel[k]
        v += (self.ps_params.disc_c1 * rng.randdouble() * (personal_val - temp_val)) + (
            self.ps_params.disc_c2 * rng.randdouble() * (global_val - temp_val)
        )
        v = v * self.ps_params.constriction_disc
        vel[k] = _clamp(v, self.ps_params.disc_min_vel, self.ps_params.disc_max_vel)

    def _confinement_disc(self, value: float) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:229 confinement_disc"""
        return _clamp(value, self.ps_params.disc_min_value, self.ps_params.disc_max_value)

    def _new_disc_value(self, temp_list: list[float], k: int, vel: float, max_dvalue: int) -> int:
        """C++: moses/moses/optimization/particle-swarm.h:282 new_disc_value

        `cvalue` is `double&` in C++ -- mutated in place here via
        `temp_list[k]`, the actual per-particle running continuous
        position `disc_parts.temp[i]` carries across generations.
        """
        temp_list[k] += vel
        temp_list[k] = self._confinement_disc(temp_list[k])
        return self._cont2disc(temp_list[k], max_dvalue)

    def _update_disc_particle(
        self,
        dtemp: instance,
        temp_list: list[float],
        personal_list: list[float],
        global_list: list[float],
        vel: velocity,
        fs: field_set,
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:351
        particle_swarm::update_disc_particle

        See module docstring's DIVERGENCE marker: `k` starts at 0 on
        EVERY call.
        """
        k = 0
        for raw_idx in range(fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()):
            multy = fs.disc_and_bit()[fs.raw_to_disc_idx(raw_idx)].multy
            self._update_disc_vel(vel, k, temp_list[k], personal_list[k], global_list[k])
            new_val = self._new_disc_value(temp_list, k, vel[k], multy)
            fs.set_raw(dtemp, raw_idx, new_val)
            k += 1

    def _update_cont_vel(
        self, vel: velocity, k: int, temp_val: float, personal_val: float, global_val: float
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.h:296 update_cont_vel"""
        rng = randGen()
        v = vel[k]
        v += self.ps_params.inertia * (
            (self.ps_params.cont_c1 * rng.randdouble() * (personal_val - temp_val))
            + (self.ps_params.cont_c2 * rng.randdouble() * (global_val - temp_val))
        )
        vel[k] = _clamp(v, self.ps_params.cont_min_vel, self.ps_params.cont_max_vel)

    def _confinement_cont(self, value: float) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:233 confinement_cont"""
        return _clamp(value, self.ps_params.cont_min_value, self.ps_params.cont_max_value)

    def _new_cont_value(self, value: float, vel: float) -> float:
        """C++: moses/moses/optimization/particle-swarm.h:304 new_cont_value

        "Wind dispersion enter here." (the header's own comment; wind
        dispersion itself is a TODO in classic, never implemented.)
        """
        return self._confinement_cont(value + vel)

    def _update_cont_particle(
        self, temp: instance, personal: instance, glob: instance, vel: velocity, fs: field_set
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:365
        particle_swarm::update_cont_particle

        See module docstring's DIVERGENCE marker: `k` starts at 0 on
        EVERY call.
        """
        k = 0
        for spec_idx in range(len(fs.contin())):
            temp_val = fs.get_contin(temp, spec_idx)
            personal_val = fs.get_contin(personal, spec_idx)
            global_val = fs.get_contin(glob, spec_idx)
            self._update_cont_vel(vel, k, temp_val, personal_val, global_val)
            fs.set_contin(temp, spec_idx, self._new_cont_value(temp_val, vel[k]))
            k += 1

    def _update_particles(
        self,
        temp_parts: deme_t,
        best_parts: deme_t,
        best_index: int,
        velocities: list[velocity],
        disc_parts: _DiscreteParticles,
        fields: field_set,
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:308
        particle_swarm::update_particles
        """
        best_instance = best_parts[best_index].first
        disc_best_instance = disc_parts.best_personal[best_index]
        for i in range(len(temp_parts)):
            temp_inst = temp_parts[i].first
            bestp_inst = best_parts[i].first
            vel = velocities[i]
            self._update_bit_particle(temp_inst, bestp_inst, best_instance, vel, fields)
            self._update_disc_particle(
                temp_inst,
                disc_parts.temp[i],
                disc_parts.best_personal[i],
                disc_best_instance,
                vel,
                fields,
            )
            self._update_cont_particle(temp_inst, bestp_inst, best_instance, vel, fields)

    # ---- main loop -------------------------------------------------------------

    def __call__(
        self,
        best_parts: deme_t,
        iscorer: iscorer_base,
        max_evals: int,
        max_time: int,
        init_inst: instance | None = None,
    ) -> None:
        """C++: moses/moses/optimization/particle-swarm.cc:46
        particle_swarm::operator() (``init_inst`` given) and
        particle-swarm.h:341 operator() (``init_inst`` omitted -- classic
        builds a null/zero instance for it). One Python method merges both
        overloads.

        `init_inst` is accepted for signature parity but never read past
        this point -- classic's own body never references its `init_inst`
        parameter either (the header's own comment: "In fact, all of the
        current code uses this entry point, no one bothers to supply an
        initial instance... This could help PSO if we maintain the best
        particle" -- an acknowledged, unexploited TODO, not an oversight
        in this port).
        """
        fields = best_parts.fields()
        self.nsteps = 0
        self.demeID = best_parts.getID()
        self.over_budget = False
        start = time.monotonic()

        # sizeof(instance) + sizeof(packed_t) * packed_width -- see
        # `_get_total_ram_bytes`'s docstring: never read after being set,
        # kept only for structural parity with the C++ member.
        self._instance_bytes = 24 + 8 * fields.packed_width()

        swarm_size = self._calc_swarm_size(fields)

        velocities: list[velocity] = [[0.0] * fields.dim_size() for _ in range(swarm_size)]
        disc_parts = _DiscreteParticles(swarm_size, fields.n_disc_fields())
        self._initialize_particles(swarm_size, best_parts, velocities, disc_parts, fields)

        # `auto temp_parts = best_parts;` -- a full value copy (every
        # `scored_instance` independently copied), NOT a Python alias --
        # see `star_anneal.py`'s and `univariate.py`'s own
        # `scored_instance(...)` copy calls for the same aliasing concern
        # (CARRY-FORWARD.md 1.1/2.4).
        temp_parts = instance_set(fields)
        temp_parts.setID(best_parts.getID())
        for si in best_parts:
            temp_parts.append(scored_instance(si.first, si.second))

        best_global = 0  # "Any value" (the C++ comment).
        # `disc_parts.temp = disc_parts.best_personal;` -- a
        # `vector<vector<double>>` assignment, deep-copying every row.
        disc_parts.temp = [list(row) for row in disc_parts.best_personal]

        best_score = very_worst_score
        best_raw_score = very_worst_score
        current_number_of_evals = 0

        not_improving = 0
        while True:
            for entry in temp_parts:
                entry.second = iscorer(entry.first)
            current_number_of_evals += swarm_size

            has_improved = False
            for i in range(swarm_size):
                inst_cscore = temp_parts[i].second
                iscore = inst_cscore.get_penalized_score()
                if iscore > best_parts[i].second.get_penalized_score():
                    best_parts[i] = scored_instance(temp_parts[i].first, temp_parts[i].second)
                    disc_parts.best_personal[i] = list(disc_parts.temp[i])
                    has_improved = True
                    if iscore > best_score:
                        best_score = iscore
                        best_global = i
                # The instance with the best raw score will typically NOT
                # be the same as the one with the best weighted score.
                # We need the raw score for the termination condition.
                rscore = inst_cscore.get_score()
                if rscore > best_raw_score:
                    best_raw_score = rscore

            # DIVERGENCE(C++ moses/moses/optimization/particle-swarm.cc:
            # 170-189,184): `gettimeofday`/`timersub` truncate elapsed time
            # to whole SECONDS before the budget check and the `max_time
            # -=` countdown -- ported the same way, and for the same
            # reason, `hill_climbing.py`'s own `__call__` documents at its
            # matching `elapsed_tv_sec` line: `time.monotonic()` replaces
            # `gettimeofday` (a host-clock substitution, not a fidelity
            # risk), but the truncation-to-whole-seconds quirk is kept
            # verbatim via `int(...)`.
            now = time.monotonic()
            elapsed_tv_sec = int(now - start)
            start = now

            # If we've blown our budget for evaluating the scorer, done.
            if max_evals <= current_number_of_evals:
                self.over_budget = True
                break

            if max_time <= elapsed_tv_sec:
                self.over_budget = True
                break
            max_time -= elapsed_tv_sec

            # If we've already gotten the best possible score, done.
            if self.opt_params.terminate_if_gte <= best_raw_score:
                break

            # TODO: work in a better way to identify convergence (the C++
            # comment, reproduced verbatim -- classic's own admission this
            # heuristic is provisional).
            not_improving = 0 if has_improved else not_improving + 1
            if not_improving > 3:
                break

            self._update_particles(
                temp_parts, best_parts, best_global, velocities, disc_parts, fields
            )

        best_parts.n_best_evals = swarm_size
        best_parts.n_evals = current_number_of_evals
