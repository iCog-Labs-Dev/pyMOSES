"""Unit tests for pymetta.moses.metapopulation.ensemble_params.

No C++ probe backs this module -- it is a plain six-field parameter
struct. Every field's default is asserted individually so a change to any
one of them fails a test, per the task's testing bar.
"""

from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters


def test_ensemble_parameters_defaults_match_cpp_constructor():
    ep = ensemble_parameters()
    assert ep.do_boosting is False
    assert ep.experts is False
    assert ep.exact_experts is True
    assert ep.expalpha == 2.0
    assert ep.bias_scale == 1.0
    assert ep.num_to_promote == 1


def test_ensemble_parameters_instances_are_independent():
    # Regression guard: two constructions must not share mutable state.
    a = ensemble_parameters()
    b = ensemble_parameters()
    a.do_boosting = True
    a.num_to_promote = 5
    assert b.do_boosting is False
    assert b.num_to_promote == 1
