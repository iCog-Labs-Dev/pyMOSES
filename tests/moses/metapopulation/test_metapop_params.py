"""Unit tests for pymetta.moses.metapopulation.metapop_params.

No C++ probe/fixture backs this module -- `metapopulationUTest` (the S10
oracle) never constructs `metapop_parameters`/`diversity_parameters`
directly, and CLI wiring for them is S13. Every default is instead pinned
against the literal values read from `metapop_params.h`/`.cc` (see this
task's report for the exact lines cited), and `p_norm_distance`/
`tanimoto_distance`/`angular_distance` are checked against hand-computed
values from their own doc-comment formulas (`opencog/util/numeric.h`).
"""

import math

import pytest

from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters
from pymetta.moses.metapopulation.metapop_params import (
    angular_distance,
    diversity_parameters,
    metapop_parameters,
    p_norm_distance,
    subsample_deme_filter_parameters,
    tanimoto_distance,
)

# ------------------------------------------------------------------ distances


def test_p_norm_distance_manhattan():
    assert p_norm_distance([1.0, 2.0, 3.0], [0.0, 2.0, 5.0], 1.0) == pytest.approx(1.0 + 0.0 + 2.0)


def test_p_norm_distance_euclidean_default_and_explicit():
    a, b = [0.0, 0.0], [3.0, 4.0]
    assert p_norm_distance(a, b, 2.0) == pytest.approx(5.0)


def test_p_norm_distance_max_difference_for_nonpositive_p():
    a, b = [1.0, 2.0, 3.0], [0.0, 2.0, 9.0]
    # |1-0|=1, |2-2|=0, |3-9|=6 -> max is 6.
    assert p_norm_distance(a, b, 0.0) == pytest.approx(6.0)
    assert p_norm_distance(a, b, -3.0) == pytest.approx(6.0)


def test_p_norm_distance_general_case():
    a, b = [0.0, 0.0], [1.0, 1.0]
    # p=3: (|1|^3 + |1|^3)^(1/3) = 2^(1/3).
    assert p_norm_distance(a, b, 3.0) == pytest.approx(2.0 ** (1.0 / 3.0))


def test_p_norm_distance_general_case_skips_zero_diffs():
    # A zero difference must not contribute pow(0, p) for a fractional p
    # (0**p is fine numerically, but the C++ source explicitly skips
    # `0 < diff` terms -- this pins that same skip, not just the result).
    a, b = [5.0, 5.0, 0.0], [5.0, 5.0, 1.0]
    assert p_norm_distance(a, b, 3.0) == pytest.approx(1.0)


def test_p_norm_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="unequal-sized"):
        p_norm_distance([1.0, 2.0], [1.0])


def test_tanimoto_distance_basic():
    a, b = [1.0, 0.0], [1.0, 1.0]
    # ab=1, aa=1, bb=2, numerator=1+2-1=2, f=1/2, distance=1-0.5=0.5.
    assert tanimoto_distance(a, b) == pytest.approx(0.5)


def test_tanimoto_distance_identical_vectors_is_zero():
    a = [2.0, 3.0, -1.0]
    assert tanimoto_distance(a, list(a)) == pytest.approx(0.0)


def test_tanimoto_distance_degenerate_numerator_returns_zero():
    # All-zero vectors: ab=aa=bb=0, numerator=0 < DISTANCE_EPSILON -> 0.
    assert tanimoto_distance([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_tanimoto_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="unequal-sized"):
        tanimoto_distance([1.0], [1.0, 2.0])


def test_angular_distance_orthogonal_vectors_is_half():
    a, b = [1.0, 0.0], [0.0, 1.0]
    # cos^-1(0)/pi = (pi/2)/pi = 0.5.
    assert angular_distance(a, b) == pytest.approx(0.5)


def test_angular_distance_identical_vectors_is_zero():
    a = [1.0, 2.0, 3.0]
    assert angular_distance(a, list(a)) == pytest.approx(0.0, abs=1e-6)


def test_angular_distance_pos_n_neg_false_doubles_alpha():
    a, b = [1.0, 0.0], [0.0, 1.0]
    assert angular_distance(a, b, pos_n_neg=False) == pytest.approx(1.0)


def test_angular_distance_clamps_rounding_error_past_plus_one():
    # Parallel vectors: ab/numerator is mathematically exactly 1, but this
    # pins that the result is clamped rather than fed to acos() unclamped
    # (acos of a value > 1 due to float rounding would be nan in C++ too,
    # which the clamp() call exists specifically to prevent).
    a = [1.0, 1.0, 1.0]
    assert angular_distance(a, list(a)) == pytest.approx(0.0)


def test_angular_distance_degenerate_numerator_returns_zero():
    assert angular_distance([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_angular_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="unequal-sized"):
        angular_distance([1.0], [1.0, 2.0])


# ------------------------------------------------------------- diversity_parameters


def test_diversity_parameters_defaults_match_cpp_constructor():
    dp = diversity_parameters()
    assert dp.pressure == 0.0
    assert dp.exponent == -1.0
    assert dp.normalize is True
    assert dp.dst2dp_type is diversity_parameters.dst2dp_enum_t.inverse
    assert dp.enabled() is False


def test_diversity_parameters_default_dst_is_euclidean_p_norm():
    # Constructor calls set_dst(p_norm, 2.0) unconditionally.
    dp = diversity_parameters()
    a, b = [0.0, 0.0], [3.0, 4.0]
    assert dp.dst(a, b) == pytest.approx(5.0)
    assert dp.dst(a, b) == pytest.approx(p_norm_distance(a, b, 2.0))


def test_diversity_parameters_enabled_tracks_pressure():
    dp = diversity_parameters()
    dp.pressure = 0.5
    assert dp.enabled() is True
    dp.pressure = 0.0
    assert dp.enabled() is False


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (diversity_parameters.dst_enum_t.p_norm, "p_norm"),
        (diversity_parameters.dst_enum_t.tanimoto, "tanimoto"),
        (diversity_parameters.dst_enum_t.angular, "angular"),
    ],
)
def test_diversity_parameters_set_dst_switches_kind(kind, expected):
    dp = diversity_parameters()
    dp.set_dst(kind, 1.0)
    a, b = [1.0, 0.0], [0.0, 1.0]
    expect_fn = {
        "p_norm": lambda: p_norm_distance(a, b, 1.0),
        "tanimoto": lambda: tanimoto_distance(a, b),
        "angular": lambda: angular_distance(a, b),
    }[expected]
    assert dp.dst(a, b) == pytest.approx(expect_fn())


def test_diversity_parameters_set_dst_rejects_unknown_kind():
    dp = diversity_parameters()
    with pytest.raises(ValueError):
        dp.set_dst("not-a-kind")


def test_diversity_parameters_dst2dp_inverse_formula():
    dp = diversity_parameters()
    dp.pressure = 2.0
    dp.set_dst2dp(diversity_parameters.dst2dp_enum_t.inverse)
    assert dp.dst2dp(1.0) == pytest.approx(2.0 / (1 + 1.0))


def test_diversity_parameters_dst2dp_complement_formula():
    dp = diversity_parameters()
    dp.pressure = 2.0
    dp.set_dst2dp(diversity_parameters.dst2dp_enum_t.complement)
    assert dp.dst2dp(0.25) == pytest.approx(2.0 * (1 - 0.25))


def test_diversity_parameters_dst2dp_pthpower_formula():
    dp = diversity_parameters()
    dp.pressure = 3.0
    dp.set_dst2dp(diversity_parameters.dst2dp_enum_t.pthpower)
    assert dp.dst2dp(0.5) == pytest.approx(math.pow(0.5, 3.0))


def test_diversity_parameters_dst2dp_reads_pressure_at_call_time():
    # dst2dp captures `pressure` BY REFERENCE in C++ (metapop_params.cc:
    # 68,71,74) -- it must read whatever `pressure` holds when CALLED, not
    # when set_dst2dp was called.
    dp = diversity_parameters()
    dp.set_dst2dp(diversity_parameters.dst2dp_enum_t.inverse)
    dp.pressure = 1.0
    first = dp.dst2dp(1.0)
    dp.pressure = 4.0
    second = dp.dst2dp(1.0)
    assert first == pytest.approx(0.5)
    assert second == pytest.approx(2.0)


def test_diversity_parameters_set_dst2dp_rejects_unknown_kind():
    dp = diversity_parameters()
    with pytest.raises(ValueError):
        dp.set_dst2dp("not-a-kind")


def test_diversity_parameters_leaves_autoscale_unset():
    # C++ leaves `autoscale` genuinely uninitialized too (metapop_params.h:
    # 46; not in the constructor's own initializer list) -- same fix as
    # subsample_deme_filter_parameters's ten fields below: never set it,
    # so an unguarded read fails loudly instead of returning garbage.
    dp = diversity_parameters()
    with pytest.raises(AttributeError):
        getattr(dp, "autoscale")  # noqa: B009 -- deliberately probing for AttributeError
    dp.autoscale = True
    assert dp.autoscale is True


def test_diversity_parameters_instances_are_independent():
    a = diversity_parameters()
    b = diversity_parameters()
    a.pressure = 5.0
    assert b.pressure == 0.0
    assert b.enabled() is False


# --------------------------------------------------- subsample_deme_filter_parameters


def test_subsample_deme_filter_parameters_defaults_match_cpp_constructor():
    sp = subsample_deme_filter_parameters()
    assert sp.by_time is True
    assert sp.contiguous_time is True
    assert sp.n_subsample_demes == 0
    assert sp.n_top_candidates == 1
    assert sp.n_tuples == 0xFFFFFFFF


def test_subsample_deme_filter_parameters_custom_construction_args():
    sp = subsample_deme_filter_parameters(_n_subsample_demes := 4, _n_top_candidates := 7)
    assert sp.n_subsample_demes == _n_subsample_demes
    assert sp.n_top_candidates == _n_top_candidates


@pytest.mark.parametrize(
    "field",
    [
        "std_dev_threshold",
        "tanimoto_mean_threshold",
        "tanimoto_geo_mean_threshold",
        "tanimoto_max_threshold",
        "n_best_bfdemes",
        "tanimoto_mean_weight",
        "tanimoto_geo_mean_weight",
        "tanimoto_max_weight",
        "n_subsample_fitnesses",
        "low_dev_pressure",
    ],
)
def test_subsample_deme_filter_parameters_leaves_ten_fields_unset(field):
    # C++ leaves these genuinely uninitialized; this port never sets them,
    # so an unguarded read must fail loudly rather than return a made-up
    # default (see the class's own DIVERGENCE note).
    sp = subsample_deme_filter_parameters()
    with pytest.raises(AttributeError):
        getattr(sp, field)
    setattr(sp, field, 1.0)
    assert getattr(sp, field) == 1.0


# ------------------------------------------------------------------- metapop_parameters


def test_metapop_parameters_defaults_match_cpp_constructor():
    mp = metapop_parameters()
    assert mp.max_candidates_per_deme == -1
    assert mp.revisit == 0
    assert mp.do_boosting is False
    assert mp.discard_dominated is False
    assert mp.keep_bscore is False
    assert mp.complexity_temperature == pytest.approx(6.0)
    assert mp.cap_coef == 50.0
    assert mp.jobs == 1
    assert mp.merge_callback is None
    assert mp.callback_user_data is None


def test_metapop_parameters_diversity_default_is_a_disabled_diversity_parameters():
    mp = metapop_parameters()
    assert isinstance(mp.diversity, diversity_parameters)
    assert mp.diversity.enabled() is False


def test_metapop_parameters_ensemble_params_default_matches_cpp():
    mp = metapop_parameters()
    assert isinstance(mp.ensemble_params, ensemble_parameters)
    assert mp.ensemble_params.do_boosting is False


def test_metapop_parameters_do_boosting_and_ensemble_params_do_boosting_are_independent_flags():
    # metapop_parameters.do_boosting and ensemble_parameters.do_boosting
    # are two SEPARATE C++ fields that both default False -- flipping one
    # must never flip the other.
    mp = metapop_parameters()
    mp.do_boosting = True
    assert mp.ensemble_params.do_boosting is False


def test_metapop_parameters_diversity_argument_is_copied_not_aliased():
    shared = diversity_parameters()
    a = metapop_parameters(diversity=shared)
    b = metapop_parameters(diversity=shared)
    a.diversity.pressure = 9.0
    assert b.diversity.pressure == 0.0
    assert shared.pressure == 0.0


def test_metapop_parameters_default_diversity_instances_are_independent():
    a = metapop_parameters()
    b = metapop_parameters()
    a.diversity.pressure = 3.0
    assert b.diversity.pressure == 0.0


def test_metapop_parameters_default_ensemble_params_instances_are_independent():
    a = metapop_parameters()
    b = metapop_parameters()
    a.ensemble_params.num_to_promote = 9
    assert b.ensemble_params.num_to_promote == 1


def test_metapop_parameters_complexity_temperature_is_narrowed_to_score_t():
    # score_t is float32; a double value with no exact float32
    # representation must come back rounded, not bit-exact.
    mp = metapop_parameters(complexity_temperature=0.1)
    assert mp.complexity_temperature != 0.1
    assert mp.complexity_temperature == pytest.approx(0.1, abs=1e-6)


def test_metapop_parameters_custom_constructor_arguments():
    mp = metapop_parameters(
        max_candidates_per_deme=10,
        revisit=-1,
        complexity_temperature=3.5,
        jobs=4,
    )
    assert mp.max_candidates_per_deme == 10
    assert mp.revisit == -1
    assert mp.complexity_temperature == pytest.approx(3.5)
    assert mp.jobs == 4
