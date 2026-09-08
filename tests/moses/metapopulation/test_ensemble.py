"""Unit tests for pymetta.moses.metapopulation.ensemble.

No C++ probe backs this module: it is a deliberately minimal stand-in for
`ensemble.{h,cc}` (98 + 438 lines), not a full port -- see the module's own
docstring. The exit bar this task sets is behavioural, not numeric: an
`ensemble` can be held and constructed on the default (non-boosting) path,
and every method that would require the unported boosting algorithm raises
`NotImplementedError` instead of silently no-opping.
"""

import pytest

from pymetta.moses.metapopulation.ensemble import ensemble
from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.scoring_base import bscore_base


class _DummyBscore(bscore_base):
    """Minimal concrete bscore_base -- mirrors metapopulationUTest's own
    `dummy_bscore` (tests/moses/metapopulationUTest.cxxtest)."""

    def __call__(self, tr):
        return []

    def best_possible_bscore(self):
        return [0.0]


@pytest.fixture
def cscore():
    return behave_cscore(_DummyBscore())


# ---------------------------------------------------------- construction / gate


def test_ensemble_constructs_unconditionally_on_default_params(cscore):
    # metapopulation.cc:41 constructs _ensemble(sc, pa.ensemble_params)
    # regardless of do_boosting -- this must not raise.
    ens = ensemble(cscore)
    assert ens.get_ensemble() == []


def test_ensemble_constructs_with_explicit_non_boosting_params(cscore):
    ens = ensemble(cscore, ensemble_parameters())
    assert ens.get_ensemble() == []


def test_ensemble_raises_when_constructed_with_do_boosting_true(cscore):
    ep = ensemble_parameters()
    ep.do_boosting = True
    with pytest.raises(NotImplementedError):
        ensemble(cscore, ep)


def test_ensemble_stores_the_underlying_bscorer(cscore):
    ens = ensemble(cscore)
    assert ens._bscorer is cscore._bscorer


# --------------------------------------------------------------- get_ensemble


def test_get_ensemble_is_read_only_accessor_and_never_raises(cscore):
    ens = ensemble(cscore)
    # Called unconditionally from best_composite_score() only when
    # do_boosting is true (metapopulation.cc:207) -- but the accessor
    # itself is trivial and must be safe to call regardless.
    assert ens.get_ensemble() == []
    assert ens.get_ensemble() is ens.get_ensemble()  # same object each call


# ------------------------------------------------------- "raises when driven"


def test_add_candidates_always_raises_not_implemented(cscore):
    ens = ensemble(cscore)
    with pytest.raises(NotImplementedError):
        ens.add_candidates([])


def test_get_weighted_tree_raises_not_implemented(cscore):
    ens = ensemble(cscore)
    with pytest.raises(NotImplementedError):
        ens.get_weighted_tree()


def test_flat_score_raises_not_implemented(cscore):
    ens = ensemble(cscore)
    with pytest.raises(NotImplementedError):
        ens.flat_score()
