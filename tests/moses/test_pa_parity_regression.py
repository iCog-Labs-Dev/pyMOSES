"""End-to-end regression for the even-parity (`pa`) search divergence.

`tests/moses/MOSESUTest.cxxtest:44` (`test_2_parity`) asserts that
`moses -Hpa -k2` reaches score 0. Classic does, in a single deme expansion
(measured live from its own `-l DEBUG` log at `-r1`: 144 evaluations, three
hill-climbing iterations, the third one a `last chance` crossover). The port
used to stall at -1.0 forever, and the cause was NOT in `local_moses` /
`hill_climbing` / `metapopulation` at all -- it was that
`build_knobs::sample_logical_perms`' random PAIR perms
(`build_knobs.cc:435-518`) were never ported, so every Boolean
representation had half the knobs classic builds. Even parity is the first
problem in the corpus that cannot be reached from single-literal knobs
alone, which is why nothing else noticed.

Full write-up, including the per-iteration trajectory diff against classic:
`.superpowers/sdd/2026-09-08-pymetta-s13-cli/parity-divergence-report.md`.

This file deliberately drives the WHOLE stack -- problem registry, knob
building, hill climbing with crossover, metapopulation merging -- because
that is the only level at which the original defect was visible.
"""

from __future__ import annotations

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.main.demo_problems import register_demo_problems
from pymetta.moses.main.problem import problem_manager
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.moses.moses_params import moses_parameters
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.util.rng import randGen

# Classic's own default, `problem-params.cc:299`
# (`po::value<double>(&perm_ratio)->default_value(0.0)`), forwarded to
# `deme_params.perm_ratio` at `problem-params.cc:2265`. This is the value the
# `moses` binary runs `-Hpa -k2` with.
_CLASSIC_DEFAULT_PERM_RATIO = 0.0


def _rule():
    return logical_reduction(())(2)


class _FakeParams:
    """Duck-typed stand-in for `problem_params`, carrying only what
    `demo_problems.py`'s `bool_problem_base.run()` reads. Deliberately a
    local copy rather than an import from another test module -- tests in
    this project do not import from each other (see
    `tests/moses/main/test_demo_problems.py`'s own note on the same
    duplication).
    """

    def __init__(self, *, max_evals: int = 10000, perm_ratio: float) -> None:
        self.exemplars: list[combo_tree] = []
        self.ignore_ops: tuple = ()
        self.bool_reduct = _rule()
        self.bool_reduct_rep = _rule()
        self.meta_params = metapop_parameters()
        self.opt_params = optim_parameters()
        self.hc_params = hc_parameters()
        self.deme_params = deme_parameters()
        self.deme_params.perm_ratio = perm_ratio
        self.filter_params = subsample_deme_filter_parameters()
        self.moses_params = moses_parameters(max_evals=max_evals)
        self.noise = -1.0
        self.complexity_ratio = 3.5
        self.printer = lambda mp, dex, stats: None


def _run_pa(problem_size: int, *, perm_ratio: float, seed: int = 1, max_evals: int = 10000):
    pmr = problem_manager()
    demo = register_demo_problems(pmr)
    demo.problem_size = problem_size
    pms = _FakeParams(max_evals=max_evals, perm_ratio=perm_ratio)
    captured: dict = {}
    pms.printer = lambda mp, dex, stats: captured.update(mp=mp, stats=stats)
    randGen().seed(seed)
    pmr.find_problem("pa").run(pms)
    return captured


@pytest.mark.oracle
def test_2_parity_reaches_score_zero_at_classic_default_perm_ratio():
    """C++: tests/moses/MOSESUTest.cxxtest:44 test_2_parity

    `moses_test_score({"-Hpa", "-k2"}, 0)`. Seed 1 is classic's own CLI
    default (`-r1`). Classic prints, sorted:

        0 and(or(!$1 $2) or($1 !$2))
        0 or(and(!$1 !$2) and($1 $2))

    -- so the tree is not unique and `moses_test_score` never checks it;
    only the score is asserted here, exactly as classic's own harness does.
    """
    mp = _run_pa(2, perm_ratio=_CLASSIC_DEFAULT_PERM_RATIO)["mp"]
    assert mp.best_score() == 0.0
    # Not part of the C++ oracle, but recorded because it is the tree this
    # port actually lands on and it is one of the two classic itself prints.
    assert combo_tree_to_str(mp.best_tree()).strip() == "or(and(!$1 !$2) and($1 $2))"


@pytest.mark.oracle
def test_2_parity_matches_classic_evaluation_budget():
    """C++: moses/moses/optimization/hill-climbing.cc:381 hill_climbing::operator()

    Classic's `-l DEBUG` log for `-Hpa -k2 -W1 -r1` records, for its one and
    only deme, `DemesHill` lines with `current_number_of_evals` 1, 33, 144
    over three iterations, then `Expansion 1 done` and the generation
    line `Stats: 1 144 0 15 0 4 0 3 0` (gen 1, num_evals 144).
    The port reproduces that eval budget exactly -- not just the final score
    -- which is what pins the divergence to knob building rather than to
    anything downstream of it.
    """
    run = _run_pa(2, perm_ratio=_CLASSIC_DEFAULT_PERM_RATIO)
    assert run["stats"].n_evals == 144
    assert run["stats"].n_expansions == 1


def test_2_parity_stalls_without_pair_perms():
    """C++: moses/moses/representation/build_knobs.cc:435 build_knobs::sample_logical_perms

    The counterexample that identified the defect, kept as a live guard: at
    `perm_ratio = -1` (classic's "no pairs" early-return mode, and what this
    port used to hard-require) the identical run cannot get past -1.0. If a
    future change makes this pass, the two tests above have stopped
    discriminating and the corpus needs re-examining.
    """
    mp = _run_pa(2, perm_ratio=-1.0)["mp"]
    assert mp.best_score() == -1.0
