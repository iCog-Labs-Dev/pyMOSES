"""C++: moses/moses/optimization/star-anneal.{h,cc}

`sa_parameters` and `simulated_annealing` -- "Star-shaped search: Apply a
modified simulated annealing-style search. Deprecated; its not really
simulated annealing, but something else, and this really doesn't work very
well. Its here for backward compatibility, for the moment." (star-anneal.h's
own doc comment, reproduced verbatim, including the header's own "not
really simulated annealing" admission -- port faithfully, not "fixed": see
the DIVERGENCE marker on `operator()` below for the specific, load-bearing
consequence of that admission.)
"""

from __future__ import annotations

import math

from pymetta.moses.moses.neighborhood_sampling import sample_new_instances
from pymetta.moses.moses.optimization.optimization import (
    deme_t,
    optim_parameters,
    optimizer_base,
)
from pymetta.moses.moses.types import score_t
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import iscorer_base
from pymetta.util.rng import randGen

__all__ = ["energy_t", "sa_parameters", "simulated_annealing"]

energy_t = score_t  # C++: moses/moses/optimization/star-anneal.h:87 energy_t

# Not itself a ported C++ symbol -- the modulus of a C++ `unsigned` (32-bit
# on every ABI classic is built for), matching hill_climbing.py's own
# `_UNSIGNED_MAX` convention for the same shape of wraparound. `max_distance`
# (star-anneal.h:154) is declared `unsigned`; see `dist_temp`'s own
# docstring for the one place this actually bites.
_UNSIGNED_MAX = (1 << 32) - 1


class sa_parameters:
    """C++: moses/moses/optimization/star-anneal.h:44 sa_parameters"""

    def __init__(self) -> None:
        """C++: moses/moses/optimization/star-anneal.h:46 sa_parameters"""
        self.init_temp = 30
        self.min_temp = 0
        self.temp_step_size = 0.5
        self.accept_prob_temp_intensity = 0.5
        self.max_new_instances = 100


class simulated_annealing(optimizer_base):
    """C++: moses/moses/optimization/star-anneal.h:85 simulated_annealing"""

    def __init__(
        self,
        op: optim_parameters | None = None,
        sa: sa_parameters | None = None,
    ) -> None:
        """C++: moses/moses/optimization/star-anneal.h:89 simulated_annealing"""
        super().__init__(op)
        self.sa_params = sa if sa is not None else sa_parameters()
        self.max_distance = 0  # C++: star-anneal.h:154 max_distance, protected

    def accept_probability(
        self, energy_new: energy_t, energy_old: energy_t, temperature: float
    ) -> float:
        """C++: moses/moses/optimization/star-anneal.h:93 accept_probability

        Plain `double` arithmetic throughout, matching C++'s own return
        type (`double`, not `score_t`) -- not narrowed via `_to_score_t`.
        """
        if energy_new < energy_old:
            return 1.0
        return math.exp((energy_old - energy_new) / temperature)

    def cooling_schedule(self, t: float) -> float:
        """C++: moses/moses/optimization/star-anneal.h:102 cooling_schedule"""
        # DIVERGENCE(C++ moses/moses/optimization/star-anneal.h:104): OC_ASSERT
        # becomes ValueError.
        if not t > 0:
            raise ValueError("t should greater than 0")
        return float(self.sa_params.init_temp) / (1.0 + t)

    def energy(self, sc_or_inst) -> energy_t:
        """C++: moses/moses/optimization/star-anneal.h:109 energy (``score_t``
        overload) and star-anneal.h:117 energy (``deme_inst_t`` overload).

        One method dispatches on argument shape across both C++ overloads,
        matching the project's usual merged-overload convention. "The
        energy is minus the score. This is because better scores are
        bigger numbers, whereas, in SA, better scores correspond to lower
        energies." (the header's own comment.)
        """
        if isinstance(sc_or_inst, (int, float)):
            return -sc_or_inst
        return -sc_or_inst.second.get_score()

    def dist_temp(self, current_temp: float) -> int:
        """C++: moses/moses/optimization/star-anneal.h:128 dist_temp

        "This method calculate the distance of the jump according to the
        temperature. The higher the temperature the higher the distance."
        (the header's own comment.) `(unsigned)(...)` truncates toward
        zero for the non-negative value this expression normally produces
        (a valid `current_temp` stays within `[min_temp, init_temp]`).

        # DIVERGENCE(C++ moses/moses/optimization/star-anneal.h:130):
        # DO NOT FIX -- `max_distance` (star-anneal.h:154) is declared
        # `unsigned`, and `(max_distance - 1)` is evaluated in UNSIGNED
        # arithmetic before its (implicit, usual-arithmetic-conversions)
        # promotion to `double` for the surrounding multiplication. If
        # `max_distance == 0` (a field_set with zero knobs -- an
        # exemplar with no build_knobs insertion at all), C++ wraps this
        # to `UINT_MAX == 4294967295`, not `-1`: the returned distance
        # then comes out enormous rather than negative. Python's
        # arbitrary-precision `int` would silently give `-1` instead --
        # exactly the S9-Critical-1 `size_t`-wrap shape this project's
        # own guidance calls out -- so the subtraction is masked to
        # 32-bit unsigned explicitly, matching `hill_climbing.py`'s own
        # `_UNSIGNED_MAX` convention for the identical C++ type.
        """
        span = self.sa_params.init_temp - self.sa_params.min_temp
        max_distance_minus_1 = (self.max_distance - 1) & _UNSIGNED_MAX
        return int(
            ((current_temp - self.sa_params.min_temp) / span) * max_distance_minus_1 + 1
        )

    def __call__(
        self,
        deme: deme_t,
        iscorer: iscorer_base,
        max_evals: int,
        max_time: int,
        init_inst: instance | None = None,
    ) -> None:
        """C++: moses/moses/optimization/star-anneal.cc:55
        simulated_annealing::operator() (``init_inst`` given) and
        star-anneal.h:143 operator() (``init_inst`` omitted -- classic
        builds a null/zero instance for it). One Python method merges both
        overloads, matching `hc_parameters`'s own convention elsewhere in
        this port's optimizer family. `max_time` is accepted for signature
        parity but never read -- classic's own `operator()` body never
        checks it either (its only termination conditions are the instance
        budget, the temperature floor, and the energy bound).
        """
        fields = deme.fields()
        if init_inst is None:
            init_inst = instance(fields.packed_width())

        self.max_distance = self.opt_params.max_distance(fields)

        # DIVERGENCE(C++ moses/moses/optimization/star-anneal.cc:67-73):
        # C++ narrows `pop_size` (an `unsigned` from `opt_params.pop_size`)
        # into `size_t max_number_of_instances`, then clamps it against
        # `max_evals` (also unsigned) with a `>` comparison -- no unsigned
        # subtraction occurs here (S9's Critical-1 shape does not apply).
        pop_size = self.opt_params.pop_size(fields)
        max_number_of_instances = pop_size
        if max_number_of_instances > max_evals:
            max_number_of_instances = max_evals

        current_number_of_instances = 0
        step = 1

        # Score the initial instance.
        center_instance = instance(init_inst)
        center_score = iscorer(center_instance)
        center_instance_energy = self.energy(center_score.get_score())
        current_temp = float(self.sa_params.init_temp)

        while True:
            current_distance = self.dist_temp(current_temp)

            # `total_number_of_neighbours=None` selects the merged
            # function's SECOND C++ overload (neighborhood_sampling.cc:261,
            # which computes the total internally) -- the one
            # star-anneal.cc:101-105 actually calls, with
            # `sa_params.max_new_instances` in the `number_of_new_
            # instances` slot (see `sample_new_instances`'s own docstring
            # for why this differs from hill-climbing.cc's six-argument
            # call).
            number_of_new_instances = sample_new_instances(
                None,
                self.sa_params.max_new_instances,
                current_number_of_instances,
                center_instance,
                deme,
                current_distance,
            )

            # If the temperature is too high, then the distance will be too
            # large, and no instances will be found at that distance.
            if number_of_new_instances == 0:
                current_temp = self.cooling_schedule(float(step) * self.sa_params.temp_step_size)
                step += 1
                if not self._loop_condition(
                    current_number_of_instances, max_number_of_instances,
                    current_temp, center_instance_energy,
                ):
                    break
                continue

            # Score all new instances in the deme.
            for i in range(current_number_of_instances, len(deme)):
                deme[i].second = iscorer(deme[i].first)

            # Get the best instance among the newly-scored slice.
            best_scored_instance = max(deme[current_number_of_instances:])
            best_instance = best_scored_instance.first
            best_instance_energy = self.energy(best_scored_instance)

            # Check if the current instance in the deme is better than
            # the center_instance.
            #
            # # DIVERGENCE(C++ moses/moses/optimization/star-anneal.cc:
            # # 133-135): DO NOT FIX -- `accept_probability` is called with
            # # `best_instance_energy` as BOTH its `energy_new` and
            # # `energy_old` arguments; `center_instance_energy` (the
            # # value that reads as though it belongs here) is never
            # # passed at all. Since `energy_new < energy_old` is then
            # # comparing `best_instance_energy` against ITSELF -- always
            # # false -- `accept_probability` always takes the `else`
            # # branch and returns `exp(0 / temperature) == 1.0`,
            # # unconditionally, regardless of how the new best instance
            # # actually compares to the running center. `actual_accept_
            # # prob` therefore always equals exactly
            # # `sa_params.accept_prob_temp_intensity` (0.5 by default),
            # # a fixed coin-flip independent of score, energy or
            # # temperature -- not the score-dependent acceptance the
            # # header's own doc comment describes. Confirmed by reading
            # # star-anneal.cc:133-135 directly; this is real, upstream
            # # behaviour (this fork's `git log` never touches
            # # star-anneal.cc), not a transcription slip, and it is
            # # exactly the kind of thing star-anneal.h's own top-of-file
            # # comment warns about ("Why these two different, unrelated
            # # things should ever have the same temperature is one of
            # # the reasons the implementation below is broken/
            # # deprecated"). Ported verbatim.
            actual_accept_prob = self.sa_params.accept_prob_temp_intensity * (
                self.accept_probability(best_instance_energy, best_instance_energy, current_temp)
            )

            if actual_accept_prob >= randGen().randdouble():
                center_instance_energy = best_instance_energy
                center_instance = instance(best_instance)

            current_number_of_instances += number_of_new_instances

            current_temp = self.cooling_schedule(float(step) * self.sa_params.temp_step_size)
            step += 1

            if not self._loop_condition(
                current_number_of_instances, max_number_of_instances,
                current_temp, center_instance_energy,
            ):
                break

        deme.n_evals = current_number_of_instances

    def _loop_condition(
        self,
        current_number_of_instances: int,
        max_number_of_instances: int,
        current_temp: float,
        center_instance_energy: energy_t,
    ) -> bool:
        """Not itself a ported C++ symbol -- the `do { ... } while(...)`
        loop condition of `simulated_annealing::operator()`
        (star-anneal.cc:156-158), factored out since Python has no
        do-while construct: `while(current_number_of_instances <
        max_number_of_instances && current_temp >= sa_params.min_temp &&
        center_instance_energy > energy(opt_params.terminate_if_gte))`.
        Every C++ do-while elsewhere in this optimizer family
        (`hill_climbing.py`) is likewise ported as "run the body once,
        then check this same condition to decide whether to continue" --
        matching that convention rather than inlining the check twice.
        """
        return (
            current_number_of_instances < max_number_of_instances
            and current_temp >= self.sa_params.min_temp
            and center_instance_energy > self.energy(self.opt_params.terminate_if_gte)
        )
