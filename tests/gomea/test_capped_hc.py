"""Tests for pymoses.gomea.capped_hc (``-a hc-capped``, Phase 1g).

``-a hc-capped`` is MOSES's own hill climbing with each deme's evaluation budget
capped (``--gomea-max-evals-per-deme``), the control arm for "is GOMEA's gain
just from short demes?". It must equal plain hill climbing run with budget
``min(max_evals, cap)``, and equal ``-a hc`` when the cap is off.
"""

import pytest

from pymoses.gomea import gomea_parameters
from pymoses.gomea.capped_hc import hc_capped_optimization
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.representation.instance_set import instance_set
from pymoses.util.rng import randGen
from tests.gomea.test_optimizer import _deme_contents, _iscorer


@pytest.fixture
def problem(make_boolean_rep):
    def _make():
        rep = make_boolean_rep("and", 3)
        return _iscorer(rep, 3), instance_set(rep.fields())

    return _make


def _run(opt, problem, seed, max_evals):
    isc, deme = problem()
    randGen().seed(seed)
    opt(deme, isc, max_evals, 10**6)
    return deme, randGen().randint()


@pytest.mark.parametrize("cap", [5, 20, 60])
def test_capped_equals_hill_climbing_with_the_smaller_budget(problem, cap):
    capped = hc_capped_optimization(
        optim_parameters(), hc_parameters(), gomea_parameters(max_evals_per_deme=cap)
    )
    hc = hill_climbing(optim_parameters(), hc_parameters())
    d1, r1 = _run(capped, problem, 3, 10**4)
    d2, r2 = _run(hc, problem, 3, min(10**4, cap))
    assert _deme_contents(d1) == _deme_contents(d2)
    assert d1.n_evals == d2.n_evals <= cap and r1 == r2


def test_no_cap_is_plain_hill_climbing(problem):
    capped = hc_capped_optimization(optim_parameters(), hc_parameters(), gomea_parameters())
    hc = hill_climbing(optim_parameters(), hc_parameters())
    d1, r1 = _run(capped, problem, 7, 10**4)
    d2, r2 = _run(hc, problem, 7, 10**4)
    assert _deme_contents(d1) == _deme_contents(d2) and r1 == r2
