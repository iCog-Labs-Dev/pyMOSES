"""C++: moses/moses/eda/optimize.h

The generic EDA competition loop `univariate_optimization` (`univariate.py`)
drives. C++ is a function template over seven policy type parameters
(`ScoringPolicy`, `TerminationPolicy`, `SelectionPolicy`,
`StructureLearningPolicy`, `ProbsLearningPolicy`, `ReplacementPolicy`,
`LoggingPolicy`); Python needs no template mechanism for this -- every
policy argument is duck-typed, and this port's one instantiation
(`univariate.cc:74-86`) fixes them to `iscorer_base`, `terminate_if_gte_or_
no_improv<composite_score>`, `tournament_selection`, `univariate`,
`local_structure_probs_learning`, `replace_the_worst`, and
`cout_log_best_and_gen` respectively.

`OMP_ALGO::transform`/`std::bind(std::cref(score), ...)` (the two scoring
passes) are ported as plain sequential loops -- see `hill_climbing.py`'s and
`optimization.py`'s own module docstrings for the same "OMP_ALGO -> plain
loop" convention (this port is single-threaded and `iscorer_base.__call__`
draws no randomness of its own, so iteration order here is not
observable).
"""

from __future__ import annotations

from typing import Protocol

from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.util.logger import logger

__all__ = ["optimize"]


class _IScorer(Protocol):
    def __call__(self, inst) -> object: ...  # returns a composite_score


def optimize(
    current: instance_set,
    n_select: int,
    n_generate: int,
    max_gens: int,
    score: _IScorer,
    termination_criterion,
    select,
    learn_structure,
    learn_probs,
    replace,
    write_log,
) -> int:
    """C++: moses/moses/eda/optimize.h:84 optimize

    "Rough outline: an initial population of individuals is created and
    scored. Then, up to max_gen iterations are performed, or until the
    TerminationPolicy is satisfied, whichever is less. ... Returns the
    number of scoring function evaluations actually performed." (the
    header's own comment, condensed.)

    `current` is mutated in place (matching C++'s `instance_set<ScoreT>&
    current`); `current.fields()` supplies both `StructureLearningPolicy::
    model_type` construction arguments below.
    """
    logger().debug("Probabilistic Learning Optimization")

    logger().debug("Evaluate the initial population (%u individuals)", len(current))
    for entry in current:
        entry.second = score(entry.first)

    fields: field_set = current.fields()
    model_type = learn_structure.model_type

    generation = 0
    while generation < max_gens and not termination_criterion(current):
        write_log(current, fields, generation)

        # Select n_select promising instances to enroll in tournament.
        logger().debug("Select %d promising instances for model building", n_select)
        promising: list[scored_instance] = []
        select(current, promising, n_select)

        # Initialize the model (run its constructor), then update it:
        # first the "structure" of the dependencies between variables
        # (a no-op for univariate()), then the probability distribution
        # over the selected population within that structure.
        logger().debug("Build probabilistic model")
        model = model_type(fields, promising)
        learn_structure(fields, promising, model)
        learn_probs(fields, promising, model)

        # Create new instances from the model and score them.
        logger().debug(
            "Sample and evaluate %d new candidates according to the model", n_generate
        )
        new_instances = instance_set(n_generate, fields)
        for i in range(len(new_instances)):
            new_instances[i] = scored_instance(model())
        for entry in new_instances:
            entry.second = score(entry.first)

        # Fold the new instances into the current population.
        logger().debug("Replace the new candidates")
        replace(new_instances, current)

        generation += 1

    write_log(current, fields, generation)

    return len(current) + generation * n_generate
