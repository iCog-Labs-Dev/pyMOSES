"""C++: moses/moses/optimization/univariate.{h,cc}

`eda_parameters` and `univariate_optimization` -- an estimation-of-
distribution algorithm (the "univariate" EDA: no cross-field dependency
modelling, just per-field marginal distributions re-estimated each
generation from a tournament-selected subpopulation; see
`pymetta.moses.eda.local_structure`'s module docstring for the model
itself). `eda_parameters` is never CLI-configurable anywhere in classic
(`git grep -n 'eda_parameters' $CLASSIC/moses/moses/main` finds nothing) --
`univariate_optimization` is always constructed with the header's own
defaults, exactly as this port's `problem_params.py` gate (out of this
task's scope; see this task's report) will have to do too.
"""

from __future__ import annotations

from pymetta.moses.eda.local_structure import local_structure_probs_learning, univariate
from pymetta.moses.eda.logging import cout_log_best_and_gen
from pymetta.moses.eda.optimize import optimize
from pymetta.moses.eda.replacement import replace_the_worst
from pymetta.moses.eda.termination import terminate_if_gte_or_no_improv
from pymetta.moses.moses.neighborhood_sampling import generate_initial_sample
from pymetta.moses.moses.optimization.optimization import (
    deme_t,
    information_theoretic_bits,
    optim_parameters,
    optimizer_base,
)
from pymetta.moses.moses.types import composite_score, worst_composite_score
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import iscorer_base
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.util.selection import tournament_selection

__all__ = ["eda_parameters", "univariate_optimization"]


class eda_parameters:
    """C++: moses/moses/optimization/univariate.h:38 eda_parameters

    "if <=1, truncation selection ratio, if >1, tournament selection size
    (should be int)" (the header's own comment for `selection`).
    """

    def __init__(self) -> None:
        """C++: moses/moses/optimization/univariate.h:40 eda_parameters"""
        self.selection = 2  # if <=1, truncation ratio; if >1, tournament size
        self.selection_ratio = 1  # ratio of population size selected for modeling
        self.replacement_ratio = 0.5  # ratio of population size sampled and integrated
        self.model_complexity = 1  # model parsimony term log(N)*model_complexity

    def is_tournament_selection(self) -> bool:
        """C++: moses/moses/optimization/univariate.h:50 is_tournament_selection"""
        return self.selection > 1

    def is_truncation_selection(self) -> bool:
        """C++: moses/moses/optimization/univariate.h:53 is_truncation_selection"""
        return self.selection <= 1


class univariate_optimization(optimizer_base):
    """C++: moses/moses/optimization/univariate.h:63 univariate_optimization"""

    def __init__(
        self,
        op: optim_parameters | None = None,
        ep: eda_parameters | None = None,
    ) -> None:
        """C++: moses/moses/optimization/univariate.h:65 univariate_optimization"""
        super().__init__(op)
        self.eda_params = ep if ep is not None else eda_parameters()

    def __call__(
        self,
        deme: deme_t,
        iscorer: iscorer_base,
        max_evals: int,
        max_time: int,
    ) -> None:
        """C++: moses/moses/optimization/univariate.cc:48
        univariate_optimization::operator()

        `max_time` is accepted for signature parity with every other
        optimizer's `operator()` (matching `hill_climbing.__call__`'s own
        convention) but never read here, exactly as classic's own
        `univariate.cc` never reads it either -- there is no time-budget
        check anywhere in this function's C++ body.
        """
        fields = deme.fields()
        pop_size = self.opt_params.pop_size(fields)
        # DIVERGENCE(C++ moses/moses/optimization/univariate.cc:53): C++
        # narrows `double information_theoretic_bits(...)` to `unsigned
        # max_gens_total` by implicit truncating conversion. `int(...)`
        # truncates toward zero for the non-negative value this always is,
        # matching that narrowing.
        max_gens_total = int(information_theoretic_bits(fields))
        max_gens_improv = self.opt_params.max_gens_improv(fields)
        # DIVERGENCE(C++ moses/moses/optimization/univariate.cc:55-56): two
        # more double->unsigned truncating narrowings, same reasoning.
        n_select = int(float(pop_size) * self.eda_params.selection_ratio)
        n_generate = int(float(pop_size) * self.eda_params.replacement_ratio)

        # Adjust parameters based on the maximal # of evaluations allowed.
        if max_evals < pop_size:
            pop_size = max_evals
            max_gens_total = 0
        else:
            # DIVERGENCE(C++ moses/moses/optimization/univariate.cc:63-64):
            # unsigned integer division in C++ (truncates toward zero for
            # non-negative operands, same as Python `//`). Neither language
            # guards `n_generate == 0`; classic invokes undefined behaviour
            # there (division by an unsigned zero), this port raises
            # ZeroDivisionError -- both are "do not reach this with
            # n_generate == 0", not a divergence this port papers over.
            max_gens_total = min(max_gens_total, (max_evals - pop_size) // n_generate)

        # Create the initial sample: generate the initial population to
        # populate the deme.
        deme.resize(pop_size)
        buf: list[instance] = []
        generate_initial_sample(fields, pop_size, buf, pop_size)
        for i, inst in enumerate(buf):
            # C++: `*out++ = inst;` through an `instance_set::iterator`
            # implicitly converts via `scored_instance(const instance&)`
            # (scored_instance.h:39, NOT `explicit`) -- ported directly.
            deme[i] = scored_instance(inst)

        if self.eda_params.is_tournament_selection():
            logger = cout_log_best_and_gen()
            bound = composite_score(
                self.opt_params.terminate_if_gte,
                worst_composite_score.get_complexity(),
                0,
            )
            deme.n_evals = optimize(
                deme,
                n_select,
                n_generate,
                max_gens_total,
                iscorer,
                terminate_if_gte_or_no_improv(bound, max_gens_improv),
                tournament_selection(int(self.eda_params.selection)),
                univariate(),
                local_structure_probs_learning(),
                replace_the_worst(),
                logger,
            )
        else:  # truncation selection
            # DIVERGENCE(C++ moses/moses/optimization/univariate.cc:88-90):
            # `OC_ASSERT(false, "Trunction selection not implemented. "
            # "Tournament should be used instead.")` -- unreachable in ANY
            # real classic build, not merely unported here: `eda_parameters`
            # is never wired to a CLI flag anywhere in classic
            # (`git grep -n 'eda_parameters' $CLASSIC/moses/moses/main`
            # finds nothing), so `selection` is always its default `2`
            # (`is_tournament_selection()` true) for every real invocation.
            raise NotImplementedError(
                "Trunction selection not implemented. Tournament should be used instead."
            )
