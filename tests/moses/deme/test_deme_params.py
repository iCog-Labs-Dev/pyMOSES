"""Unit tests for pymetta.moses.deme.deme_params.deme_parameters."""

import pytest

from pymetta.moses.deme.deme_params import deme_parameters


def test_defaults_match_cpp_initializer_list():
    """C++: moses/moses/deme/deme_params.h:39-50 deme_parameters

    `reduce_all(true), ignore_ops(empty_ignore_ops), perceptions(NULL),
    actions(NULL), fstor(NULL), linear_contin(true)` -- the constructor's
    own initializer list, verbatim.
    """
    params = deme_parameters()
    assert params.reduce_all is True
    assert params.ignore_ops == ()
    assert params.perceptions is None
    assert params.actions is None
    assert params.fstor is None
    assert params.linear_contin is True


def test_constructor_parameters_are_stored():
    marker_ops = ("op-a", "op-b")
    params = deme_parameters(
        reduce_all=False,
        ignore_ops=marker_ops,
        perceptions="perceptions-marker",
        actions="actions-marker",
        fstor="fstor-marker",
    )
    assert params.reduce_all is False
    assert params.ignore_ops == marker_ops
    assert params.perceptions == "perceptions-marker"
    assert params.actions == "actions-marker"
    assert params.fstor == "fstor-marker"
    # linear_contin is not a constructor parameter in C++ (deme_params.h:49
    # is a plain member-initializer default) -- always True regardless of
    # what else was passed.
    assert params.linear_contin is True


def test_ignore_ops_accepts_any_iterable_and_stores_a_tuple():
    params = deme_parameters(ignore_ops=iter(["a", "b", "c"]))
    assert params.ignore_ops == ("a", "b", "c")


# ---------------------------------------------------------------------------
# max_candidates_per_deme / perm_ratio: deliberately left unset (see
# deme_params.py's own module docstring DIVERGENCE note) -- reading either
# before an explicit assignment must raise AttributeError, matching C++'s
# genuinely uninitialized (indeterminate) struct members, louder rather
# than silently substituting a plausible default.
# ---------------------------------------------------------------------------


def test_max_candidates_per_deme_is_unset_by_default():
    params = deme_parameters()
    with pytest.raises(AttributeError):
        _ = params.max_candidates_per_deme


def test_perm_ratio_is_unset_by_default():
    params = deme_parameters()
    with pytest.raises(AttributeError):
        _ = params.perm_ratio


def test_perm_ratio_can_be_set_after_construction():
    """Matches every real classic caller (problem-params.cc:2265,
    `deme_params.perm_ratio = perm_ratio;`, unconditionally after
    construction) -- and S8's own `representation`/`build_knobs` only ever
    drive to completion with exactly -1.0 (see build_knobs.py's own
    `_validate_request`).
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    assert params.perm_ratio == -1.0


def test_max_candidates_per_deme_can_be_set_after_construction():
    params = deme_parameters()
    params.max_candidates_per_deme = 50
    assert params.max_candidates_per_deme == 50
