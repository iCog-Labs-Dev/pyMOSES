"""Unit tests for pymetta.moses.eda.optimize.optimize.

Exercised end-to-end (all seven real policies) by `test_univariate.py`;
this file isolates `optimize()`'s own contract -- the return-value formula,
and that it correctly threads a population through all seven policy calls
in the documented order -- using small stand-in policies rather than the
full univariate/local_structure stack, so a failure here points at
`optimize()` itself rather than at `local_structure`.
"""

from pymetta.moses.eda.optimize import optimize
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.field_set import disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_set import instance_set
from pymetta.moses.representation.scored_instance import scored_instance


class _CountingScorer:
    """Not itself a ported C++ symbol -- scores by the instance's raw
    field 0 value directly, and counts calls.
    """

    def __init__(self, fs: field_set) -> None:
        self.fs = fs
        self.calls = 0

    def __call__(self, inst: instance) -> composite_score:
        self.calls += 1
        return composite_score(float(self.fs.get_raw(inst, 0)))


class _AlwaysContinue:
    def __call__(self, items) -> bool:
        return False


class _StopAfterOneGen:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, items) -> bool:
        self.calls += 1
        return self.calls > 1


class _TakeFirstNSelect:
    """Not itself a ported C++ symbol -- a trivial SelectionPolicy taking
    the first `n_select` entries, for a test that only needs SOME
    selection to happen deterministically.
    """

    def __call__(self, items, dst, n_select) -> None:
        for i in range(n_select):
            dst.append(items[i])


class _NoStructure:
    model_type = list  # arbitrary; overridden per-test via monkeypatch-free stub

    def __call__(self, fields, items, model) -> None:
        return


class _FixedModel:
    """Not itself a ported C++ symbol -- a stand-in ProbsLearningPolicy
    model that always produces the same instance, so `optimize()`'s new-
    instance-generation step is fully deterministic.
    """

    def __init__(self, fields, items) -> None:
        self.fields = fields

    def __call__(self) -> instance:
        return instance(self.fields.packed_width())


class _NoProbs:
    def __call__(self, fields, items, model) -> None:
        return


class _ReplaceFirst:
    """Not itself a ported C++ symbol -- overwrites the first
    `len(new_insts)` slots of `dst`, a simpler stand-in for
    `replace_the_worst` that lets a test predict exactly which slots
    changed.
    """

    def __call__(self, new_insts, dst) -> None:
        for i in range(len(new_insts)):
            dst[i] = new_insts[i]


class _NoLog:
    def __call__(self, items, fields, gen) -> None:
        return


def _deme(n: int) -> instance_set:
    fs = field_set(disc_spec(3), 2)
    deme = instance_set(fs)
    for i in range(n):
        inst = instance(fs.packed_width())
        fs.set_raw(inst, 0, i % 3)
        deme.append(scored_instance(inst))
    return deme


def test_return_value_is_size_plus_generations_times_n_generate():
    current = _deme(4)
    scorer = _CountingScorer(current.fields())
    structure = _NoStructure()
    structure.model_type = _FixedModel
    n_evals = optimize(
        current,
        n_select=2,
        n_generate=3,
        max_gens=2,
        score=scorer,
        termination_criterion=_AlwaysContinue(),
        select=_TakeFirstNSelect(),
        learn_structure=structure,
        learn_probs=_NoProbs(),
        replace=_ReplaceFirst(),
        write_log=_NoLog(),
    )
    # len(current) is unchanged by replace (same size, just overwritten
    # slots) -- 4 + 2 generations * 3 n_generate == 10.
    assert n_evals == 4 + 2 * 3
    assert len(current) == 4


def test_termination_criterion_stops_the_loop_early():
    current = _deme(3)
    scorer = _CountingScorer(current.fields())
    structure = _NoStructure()
    structure.model_type = _FixedModel
    term = _StopAfterOneGen()
    n_evals = optimize(
        current,
        n_select=1,
        n_generate=2,
        max_gens=100,
        score=scorer,
        termination_criterion=term,
        select=_TakeFirstNSelect(),
        learn_structure=structure,
        learn_probs=_NoProbs(),
        replace=_ReplaceFirst(),
        write_log=_NoLog(),
    )
    # One generation runs (term returns False the first time it's asked,
    # after the initial scoring pass, then True on the second check).
    assert n_evals == 3 + 1 * 2


def test_initial_population_is_scored_before_any_generation_runs():
    current = _deme(5)
    scorer = _CountingScorer(current.fields())
    structure = _NoStructure()
    structure.model_type = _FixedModel
    term = _StopAfterOneGen()
    term.calls = 1  # pretend termination already fires on the FIRST check
    optimize(
        current,
        n_select=1,
        n_generate=1,
        max_gens=5,
        score=scorer,
        termination_criterion=term,
        select=_TakeFirstNSelect(),
        learn_structure=structure,
        learn_probs=_NoProbs(),
        replace=_ReplaceFirst(),
        write_log=_NoLog(),
    )
    # Even with zero generations run, the initial population must have
    # been scored exactly once each (5 calls), matching C++'s
    # unconditional pre-loop OMP_ALGO::transform.
    assert scorer.calls == 5


def test_new_instances_are_scored_before_replace_is_called():
    """The new-instance scoring pass must happen BEFORE `replace` runs,
    so `replace` sees real (non-default-worst) scores to compare against.
    """
    current = _deme(4)
    scorer = _CountingScorer(current.fields())
    structure = _NoStructure()
    structure.model_type = _FixedModel

    seen_scores_at_replace_time = []

    class _RecordingReplace:
        def __call__(self, new_insts, dst) -> None:
            seen_scores_at_replace_time.extend(si.second.get_score() for si in new_insts)
            for i in range(len(new_insts)):
                dst[i] = new_insts[i]

    optimize(
        current,
        n_select=1,
        n_generate=2,
        max_gens=1,
        score=scorer,
        termination_criterion=_AlwaysContinue(),
        select=_TakeFirstNSelect(),
        learn_structure=structure,
        learn_probs=_NoProbs(),
        replace=_RecordingReplace(),
        write_log=_NoLog(),
    )
    # _FixedModel always produces a zero instance -> raw field 0 == 0 ->
    # score 0.0, not the fresh-worst default a not-yet-scored instance
    # would carry.
    assert seen_scores_at_replace_time == [0.0, 0.0]
