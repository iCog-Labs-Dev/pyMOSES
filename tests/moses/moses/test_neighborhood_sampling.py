"""Oracle and guard tests for pymetta.moses.moses.neighborhood_sampling.

The oracle fixture (tests/fixtures/neighborhood.json) is produced by
tools/cpp_probe/neighborhood_probe.cc against the real classic library (see
tools/capture_neighborhood.py) -- not reimplemented by hand. It carries four
sections: "deterministic" (count_neighborhood_size +
generate_all_in_neighborhood, no RNG), "safe_binomial_coefficient",
"generate_initial_sample", and "sample_from_neighborhood" (the latter two are
RNG-drawn; reproducing them element-for-element is this task's whole point).

`sample_new_instances` (both overloads) is tested separately, near the
bottom of this file, against tests/fixtures/hillclimbing.json -- S9 Task 2's
own fixture (tools/cpp_probe/hillclimbing_probe.cc), since it needs
`instance_set`, which did not exist when this file's original oracle
fixture was captured.
"""

import json
from pathlib import Path

import pytest

from pymetta.moses.moses.neighborhood_sampling import (
    _SIZE_T_MAX,
    _randomize,
    count_neighborhood_size,
    count_neighborhood_size_from_index,
    generate_all_in_neighborhood,
    generate_initial_sample,
    safe_binomial_coefficient,
    sample_from_neighborhood,
    sample_new_instances,
    vary_n_knobs,
)
from pymetta.moses.moses.types import worst_composite_score
from pymetta.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_set import instance_set
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "neighborhood.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

_HC_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "hillclimbing.json"
_HC_GOLDEN = json.loads(_HC_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


def test_golden_fixture_has_expected_shape(golden):
    assert set(golden) == {
        "deterministic",
        "safe_binomial_coefficient",
        "generate_initial_sample",
        "sample_from_neighborhood",
    }
    assert len(golden["deterministic"]) == 11
    assert len(golden["safe_binomial_coefficient"]) == 8
    assert len(golden["generate_initial_sample"]) == 12
    assert len(golden["sample_from_neighborhood"]) == 7
    for case in golden["deterministic"]:
        assert len(case["neighbors"]) == case["count" if case["has_center"] else "count_null"]
    for case in golden["generate_initial_sample"]:
        assert len(case["instances"]) == case["n"]
    for case in golden["sample_from_neighborhood"]:
        assert len(case["instances"]) == case["sample_size"]


def _uniform_fs(multiplicity: int, num_knobs: int) -> field_set:
    return field_set(disc_spec(multiplicity), num_knobs)


def _mixed_fs(multiplicities: list[int]) -> field_set:
    return field_set([disc_spec(m) for m in multiplicities])


def _raw_tuple(fs: field_set, inst: instance) -> tuple[int, ...]:
    return tuple(fs.get_raw(inst, i) for i in range(fs.raw_size()))


# ---------------------------------------------------------------- deterministic


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["deterministic"],
    ids=lambda c: f"m{c['multiplicity']}_k{c['num_knobs']}_d{c['distance']}_c{c['has_center']}",
)
def test_deterministic_matches_cpp(case):
    fs = _uniform_fs(case["multiplicity"], case["num_knobs"])

    if case["has_center"]:
        center = instance(fs.packed_width())
        for idx, value in enumerate(case["center"]):
            fs.set_raw(center, idx, value)
    else:
        center = None

    count_null = count_neighborhood_size(fs, case["distance"])
    assert count_null == case["count_null"]

    if case["has_center"]:
        count_with = count_neighborhood_size(fs, case["distance"], inst=center)
        assert count_with == case["count"]

    out: list[instance] = []
    if case["has_center"]:
        generate_all_in_neighborhood(fs, case["distance"], out, case["count"], center_inst=center)
    else:
        generate_all_in_neighborhood(fs, case["distance"], out, case["count_null"])

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(n) for n in case["neighbors"]]
    assert got == expected


# ---------------------------------------------------------- safe_binomial_coefficient


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["safe_binomial_coefficient"], ids=lambda c: f"k{c['k']}_n{c['n']}"
)
def test_safe_binomial_coefficient_matches_cpp(case):
    assert safe_binomial_coefficient(case["k"], case["n"]) == case["result"]


def test_safe_binomial_coefficient_k_less_than_n_returns_zero():
    # DIVERGENCE: no C++ oracle value exists for this input -- see the
    # function's own docstring marker. k < n never occurs at the real call
    # site (count_neighborhood_size_from_index guards dist <= rb first).
    assert safe_binomial_coefficient(3, 4) == 0


# ------------------------------------------------------------- generate_initial_sample


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["generate_initial_sample"],
    ids=lambda c: f"{c['label']}_seed{c['seed']}",
)
def test_generate_initial_sample_matches_cpp(case):
    fs = _mixed_fs(case["multiplicities"])
    randGen().seed(case["seed"])

    out: list[instance] = []
    generate_initial_sample(fs, case["n"], out, case["n"])

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(row) for row in case["instances"]]
    assert got == expected


# ---------------------------------------------------------- sample_from_neighborhood


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["sample_from_neighborhood"],
    ids=lambda c: f"seed{c['seed']}_d{c['dist']}",
)
def test_sample_from_neighborhood_matches_cpp(case):
    fs = _mixed_fs(case["multiplicities"])
    center = instance(fs.packed_width())
    for idx, value in enumerate(case["center_raw"]):
        fs.set_raw(center, idx, value)

    randGen().seed(case["seed"])
    out: list[instance] = []
    sample_from_neighborhood(
        fs, case["dist"], case["sample_size"], out, case["sample_size"], center
    )

    got = [_raw_tuple(fs, inst) for inst in out]
    expected = [tuple(row) for row in case["instances"]]
    assert got == expected


# --------------------------------------------------------------------- guards


def test_emit_raises_when_buffer_full():
    fs = _uniform_fs(2, 3)
    inst = instance(fs.packed_width())
    with pytest.raises(IndexError):
        vary_n_knobs(fs, inst, 0, 0, [], 0)


def test_generate_all_in_neighborhood_rejects_wrong_center_width():
    fs = _uniform_fs(2, 3)
    bad_center = instance(fs.packed_width() + 1)
    with pytest.raises(ValueError):
        generate_all_in_neighborhood(fs, 1, [], 10, center_inst=bad_center)


def test_sample_from_neighborhood_rejects_wrong_center_width():
    fs = _uniform_fs(2, 3)
    bad_center = instance(fs.packed_width() + 1)
    with pytest.raises(ValueError):
        sample_from_neighborhood(fs, 1, 5, [], 5, bad_center)


def test_sample_from_neighborhood_rejects_distance_over_dimension():
    # match= pins this to the explicit dist > dim guard's own message, not
    # merely "some ValueError somewhere": without the guard, lazy_selector's
    # own "selector is empty" ValueError still fires once all dim indices
    # are exhausted, which would let a bare pytest.raises(ValueError) pass
    # vacuously even with the guard deleted (measured directly: removing the
    # `if dist > dim` check leaves this test green using a bare check).
    fs = _uniform_fs(2, 3)
    center = instance(fs.packed_width())
    with pytest.raises(ValueError, match="cannot be greater than the field dimension"):
        sample_from_neighborhood(fs, fs.dim_size() + 1, 5, [], 5, center)


def test_sample_from_neighborhood_zero_distance_leaves_instance_unchanged():
    fs = _mixed_fs([2, 2, 2, 3, 4, 5])
    center = instance(fs.packed_width())
    fs.set_raw(center, 0, 2)
    randGen().seed(0)
    out: list[instance] = []
    sample_from_neighborhood(fs, 0, 4, out, 4, center)
    for inst in out:
        assert _raw_tuple(fs, inst) == _raw_tuple(fs, center)


def _fs_with_contin() -> field_set:
    return field_set([contin_spec(0.0, 1.0, 2.0, 3), disc_spec(3)])


def test_count_neighborhood_size_raises_on_contin_field():
    fs = _fs_with_contin()
    with pytest.raises(NotImplementedError):
        count_neighborhood_size_from_index(fs, instance(fs.packed_width()), 1, 0)


def test_vary_n_knobs_raises_on_contin_field():
    fs = _fs_with_contin()
    with pytest.raises(NotImplementedError):
        vary_n_knobs(fs, instance(fs.packed_width()), 1, 0, [], 10)


def test_sample_from_neighborhood_raises_on_contin_field():
    fs = _fs_with_contin()
    center = instance(fs.packed_width())
    randGen().seed(0)
    with pytest.raises(NotImplementedError):
        sample_from_neighborhood(fs, fs.dim_size(), 5, [], 5, center)


def test_randomize_raises_on_contin_field():
    fs = _fs_with_contin()
    with pytest.raises(NotImplementedError):
        _randomize(fs, instance(fs.packed_width()))


def test_generate_initial_sample_raises_on_contin_field():
    fs = _fs_with_contin()
    with pytest.raises(NotImplementedError):
        generate_initial_sample(fs, 1, [], 1)


def test_size_t_max_is_64_bit():
    assert _SIZE_T_MAX == 2**64 - 1


# ------------------------------------------------------------ sample_new_instances


def _sni_center(fs: field_set, n_knobs: int) -> instance:
    center = instance(fs.packed_width())
    for i in range(0, n_knobs, 3):
        fs.set_raw(center, i, 1)
    return center


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _HC_GOLDEN["sample_new_instances"],
    ids=lambda c: f"m{c['multiplicity']}_k{c['n_knobs']}_d{c['dist']}_seed{c['seed']}_"
    f"total{c['give_total']}",
)
def test_sample_new_instances_matches_cpp(case):
    fs = _uniform_fs(case["multiplicity"], case["n_knobs"])
    center = _sni_center(fs, case["n_knobs"])
    deme = instance_set(case["current_number_of_instances"], fs)

    randGen().seed(case["seed"])
    if case["give_total"]:
        total = count_neighborhood_size(
            fs, case["dist"], case["number_of_new_instances"], inst=center
        )
        returned = sample_new_instances(
            total,
            case["number_of_new_instances"],
            case["current_number_of_instances"],
            center,
            deme,
            case["dist"],
        )
    else:
        returned = sample_new_instances(
            None,
            case["number_of_new_instances"],
            case["current_number_of_instances"],
            center,
            deme,
            case["dist"],
        )

    assert returned == case["returned"]
    assert len(deme) == case["deme_size"]

    got_new_slice = [
        _raw_tuple(fs, deme[i].first)
        for i in range(case["current_number_of_instances"], len(deme))
    ]
    expected = [tuple(row) for row in case["new_slice"]]
    assert got_new_slice == expected

    # Every newly-written slot must carry a FRESH default (worst)
    # composite_score, matching scored_instance(instance) -- see
    # neighborhood_sampling.py's _DemeWriter docstring -- not whatever
    # score happened to occupy that slot before deme.resize() grew it.
    for i in range(case["current_number_of_instances"], len(deme)):
        assert deme[i].second == worst_composite_score


def test_sample_new_instances_overload_2_matches_overload_1_without_precomputed_total():
    """Not itself an oracle case (no separate C++ value exists beyond the
    fixture's own give_total=false rows already covered above) -- a direct
    unit check that passing total_number_of_neighbours=None recomputes it
    identically to the caller precomputing it, for a case NOT already in
    the fixture, confirming the merge itself (not just the two fixture-
    covered shapes) is correct.
    """
    fs = _uniform_fs(2, 6)
    center = instance(fs.packed_width())

    randGen().seed(7)
    deme_a = instance_set(0, fs)
    returned_a = sample_new_instances(None, 3, 0, center, deme_a, 1)

    randGen().seed(7)
    deme_b = instance_set(0, fs)
    total = count_neighborhood_size(fs, 1, 3, inst=center)
    returned_b = sample_new_instances(total, 3, 0, center, deme_b, 1)

    assert returned_a == returned_b
    assert [_raw_tuple(fs, si.first) for si in deme_a] == [
        _raw_tuple(fs, si.first) for si in deme_b
    ]


def test_sample_new_instances_grows_deme_in_place():
    fs = _uniform_fs(2, 10)
    center = instance(fs.packed_width())
    deme = instance_set(4, fs)  # 4 pre-existing (unrelated) instances
    randGen().seed(1)
    returned = sample_new_instances(None, 5, 4, center, deme, 1)
    assert returned == 5
    assert len(deme) == 9  # 4 pre-existing + 5 new, not replaced/truncated
