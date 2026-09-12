"""C++ oracle and aliasing tests for scored_instance."""

import json
import struct

import pytest

from pymoses.moses.moses.types import composite_score, very_worst_score
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.scored_instance import scored_instance


def _hex(x: float) -> str:
    return "0x" + struct.pack(">f", x).hex()


def test_zero_arg_constructs_empty_instance_and_fresh_worst_score():
    si = scored_instance()
    assert si.first == instance()
    assert si.second.get_score() == very_worst_score
    assert si.second.get_complexity() == 0


def test_one_arg_copies_instance_and_default_constructs_fresh_score():
    src = instance([1, 2])
    si = scored_instance(src)
    assert si.first == src
    assert si.first is not src
    src[0] = 9
    assert si.first == [1, 2]  # aliasing guard: mutating src must not reach si.first
    assert si.second.get_score() == very_worst_score


def test_one_arg_gives_each_call_an_independent_score_object():
    # C++'s std::vector fill-construction default-constructs a FRESH ScoreT()
    # per slot; this is the property instance_set's own fill constructors
    # depend on (see instance_set.py's module docstring and the
    # "default_fill" oracle below).
    a = scored_instance(instance(1))
    b = scored_instance(instance(1))
    a.second.set_score(-1.0)
    assert b.second.get_score() == very_worst_score


def test_two_arg_copies_both_instance_and_score():
    src_inst = instance([3])
    src_score = composite_score(-2.0, 5, 0.0, 0.0)
    si = scored_instance(src_inst, src_score)
    assert si.first == src_inst and si.first is not src_inst
    assert si.second is not src_score
    src_inst[0] = 7
    src_score.set_score(-9.0)
    assert si.first == [3]
    assert si.second.get_score() == -2.0


def test_one_arg_rejects_a_score_without_an_instance():
    with pytest.raises(TypeError):
        scored_instance(None, composite_score())


def test_constructor_rejects_non_instance_first_argument():
    with pytest.raises(TypeError):
        scored_instance([1, 2])  # a plain list is not an `instance`


def test_lt_and_gt_delegate_to_second():
    lo = scored_instance(instance(1), composite_score(-10.0, 0, 0.0, 0.0))
    hi = scored_instance(instance(1), composite_score(-2.0, 0, 0.0, 0.0))
    assert lo < hi
    assert hi > lo
    assert not (hi < lo)
    assert not (lo > hi)


def test_le_ge_match_boost_less_than_comparable_derivation():
    a = scored_instance(instance(1), composite_score(-5.0, 3, 0.0, 0.0))
    b = scored_instance(instance(1), composite_score(-5.0, 3, 0.0, 0.0))
    c = scored_instance(instance(1), composite_score(-2.0, 3, 0.0, 0.0))
    assert a <= b and a >= b  # tie: both hold
    assert a <= c and not (a >= c)
    assert c >= a and not (c <= a)


def test_eq_compares_first_and_second_elementwise():
    a = scored_instance(instance([1, 2]), composite_score(-5.0, 3, 0.0, 0.0))
    b = scored_instance(instance([1, 2]), composite_score(-5.0, 3, 0.0, 0.0))
    c = scored_instance(instance([1, 9]), composite_score(-5.0, 3, 0.0, 0.0))
    assert a == b
    assert a != c
    assert a != scored_instance(instance([1, 2]), composite_score(-9.0, 3, 0.0, 0.0))


def test_scored_instance_is_unhashable():
    with pytest.raises(TypeError):
        hash(scored_instance())


@pytest.mark.oracle
def test_cpp_pairwise_ordering_oracle(fixture_dir):
    """Reproduces classic's actual pairwise `<`/`>` results (and
    `std::greater<scored_instance<composite_score>>()`, matching
    merging.cc:507's real comparator) for a curated set of composite_score
    tuples, including an exact tie, a complexity tie-break, a penalized
    score, a within-epsilon near-tie, and NaN -- see containers_probe.cc's
    `emit_pairwise_compare`.
    """
    fixture = json.loads((fixture_dir / "containers.json").read_text())
    section = fixture["pairwise_compare"]

    # Same synthetic table containers_probe.cc builds, by name.
    table = {
        "a-worse-score": (-10.0, 0, 0.0, 0.0),
        "b-better-score": (-2.0, 0, 0.0, 0.0),
        "c-tie-score-high-complexity": (-5.0, 10, 0.0, 0.0),
        "d-tie-score-low-complexity": (-5.0, 3, 0.0, 0.0),
        "e-tie-score-tie-complexity": (-5.0, 3, 0.0, 0.0),
        "f-same-raw-score-but-penalized": (-5.0, 10, 1.0, 0.0),
        "g-near-tie-within-epsilon": (-5.0000005, 3, 0.0, 0.0),
        "h-nan-score": (float("nan"), 5, 0.0, 0.0),
        "i-nan-score-again": (float("nan"), 1, 0.0, 0.0),
    }
    names = section["names"]
    assert names == list(table.keys())

    entries = [
        scored_instance(instance(1), composite_score(*table[name])) for name in names
    ]

    assert len(section["pairs"]) == len(names) * (len(names) - 1)
    for pair in section["pairs"]:
        i, j = pair["i"], pair["j"]
        assert (entries[i] < entries[j]) == pair["lt"]
        assert (entries[i] > entries[j]) == pair["gt"]
        assert pair["std_greater"] == pair["gt"]  # std::greater<T>()(a,b) == (a>b)


@pytest.mark.oracle
def test_cpp_default_fill_scores_oracle(fixture_dir):
    """`scored_instance(instance)`'s default score must match a freshly
    constructed C++ `composite_score()` bit-exactly (see
    containers_probe.cc's `emit_default_fill`, entry 0 before mutation).
    """
    fixture = json.loads((fixture_dir / "containers.json").read_text())
    entry = fixture["default_fill"]["entries"][0]

    si = scored_instance(instance(1))
    assert _hex(si.second.get_score()) == entry["score"]["score"]
    assert si.second.get_complexity() == entry["score"]["complexity"]
    assert _hex(si.second.get_complexity_penalty()) == entry["score"]["complexity_penalty"]
    assert _hex(si.second.get_uniformity_penalty()) == entry["score"]["uniformity_penalty"]
    assert _hex(si.second.get_penalized_score()) == entry["score"]["penalized_score"]
