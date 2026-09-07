"""Oracle and guard tests for pymetta.moses.moses.neighborhood_sampling.

The oracle fixture (tests/fixtures/neighborhood.json) is produced by
tools/cpp_probe/neighborhood_probe.cc against the real classic library (see
tools/capture_neighborhood.py) -- not reimplemented by hand. It carries four
sections: "deterministic" (count_neighborhood_size +
generate_all_in_neighborhood, no RNG), "safe_binomial_coefficient",
"generate_initial_sample", and "sample_from_neighborhood" (the latter two are
RNG-drawn; reproducing them element-for-element is this task's whole point).
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
    vary_n_knobs,
)
from pymetta.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "neighborhood.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


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
