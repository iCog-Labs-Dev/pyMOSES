"""Oracle and guard tests for pymetta.comboreduct.combo.similarity.

The oracle fixture (tests/fixtures/similarity.json) is produced by
tools/cpp_probe/similarity_probe.cc, which links the REAL exported
`tree_flatten`/`tree_similarity` symbols from libcomboreduct.so (see
tools/capture_similarity.py) -- not reimplemented by hand. It carries two
sections: "flatten" (tree_flatten -> toString, reproducing classic's own
similarityUTest.cxxtest test_simple/test_nest/test_long cases verbatim)
and "similarity" (tree_similarity, reproducing test_distance's own 11
cases -- with bare identifier placeholders substituted 1:1 by arguments,
see the probe's own comment -- plus adversarial size_t-wraparound cases
that pin the genuine bug documented in similarity.py's own module
docstring).
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.similarity import (
    _cvttsd2si_signed,
    _double_to_uint64,
    _first_loop_distance,
    toString,
    tree_flatten,
    tree_similarity,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "similarity.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


# -------------------------------------------------------------------- tree_flatten


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["flatten"], ids=lambda c: c["input"])
def test_tree_flatten_matches_cpp(case):
    assert toString(tree_flatten(case["input"])) == case["result"]


def test_tree_flatten_mutation_swapping_child_order():
    """Mutation proof: swapping which child is `first` vs `second` in the
    2-child terminal branch changes the branch string for an ASYMMETRIC
    pair (a symmetric one like "+(1 1)" could not discriminate this).
    """
    tbv = tree_flatten("+(1 2 3)")
    assert "+(2 3)" in tbv
    assert "+(3 2)" not in tbv


def test_tree_flatten_leaf_count_accumulates():
    tbv = tree_flatten("+(1 1 1 1 1)")
    assert tbv == {"+(1 +)": 3, "+(1 1)": 1}


# ------------------------------------------------------------------ tree_similarity


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["similarity"], ids=lambda c: f"{c['a']}__{c['b']}")
def test_tree_similarity_matches_cpp(case):
    assert tree_similarity(case["a"], case["b"]) == case["dist"]


def test_tree_similarity_accepts_pre_flattened_dicts():
    """The tree_branch_vector overload (similarity.cc:167) -- exercised
    directly here, bypassing the parser entirely.
    """
    av = {"x": 3, "y": 1}
    bv = {"x": 1, "y": 1, "z": 2}
    # x: acnt=3 >= bcnt=1 -> contributes 2. y: equal -> 0. z: only in bv -> 2.
    assert tree_similarity(av, bv) == 4


def test_tree_similarity_is_not_symmetric_for_a_wraparound_case():
    """Direct, minimal reproduction of the documented asymmetry (module
    docstring): tree_similarity(a, b) != tree_similarity(b, a) whenever a
    shared branch is strictly MORE common in the second argument.
    """
    av = {"k": 1}
    bv = {"k": 5}
    assert tree_similarity(av, bv) == 0  # wraps and collapses to 0
    assert tree_similarity(bv, av) == 4  # no wrap: plain 5 - 1


def test_tree_similarity_wraparound_erases_prior_accumulation():
    """A wraparound key that sorts AFTER a normal key still erases the
    normal key's already-accumulated contribution (the "absorption"
    property `_double_to_uint64`'s docstring proves) -- keys chosen so
    "a_normal" < "b_wrap" in sorted (std::map) order.
    """
    av = {"a_normal": 10, "b_wrap": 1}
    bv = {"a_normal": 3, "b_wrap": 5}
    # Processing order: a_normal first (dist = 7), then b_wrap wraps and
    # resets dist to 0 regardless of the 7 already accumulated.
    assert tree_similarity(av, bv) == 0


def test_tree_similarity_wraparound_before_normal_key_still_lets_later_terms_add_normally():
    """After a wraparound resets dist to 0, a LATER (sorted-after) normal
    key adds normally on top of the reset 0, rather than staying corrupted.
    """
    av = {"a_wrap": 1, "b_normal": 10}
    bv = {"a_wrap": 5, "b_normal": 3}
    # a_wrap sorts first: wraps, dist -> 0. b_normal sorts second: 10-3=7,
    # dist -> 0 + 7 = 7.
    assert tree_similarity(av, bv) == 7


def test_tree_similarity_diff_only_in_b_adds_plainly():
    av = {"shared": 2}
    bv = {"shared": 2, "only_in_b": 9}
    assert tree_similarity(av, bv) == 9


# --------------------------------------------------------- _double_to_uint64 / helpers


@pytest.mark.parametrize(
    "d,want",
    [
        (4.0, 4),
        (0.0, 0),
        (18446744073709551616.0, 0),  # 2**64 exactly
        (18446744073709549568.0, 18446744073709549568),  # 2**64 - 2048
        (9223372036854775808.0, 9223372036854775808),  # 2**63 exactly
        (18446744083709550592.0, 0),  # ~2**64 + 1e10
    ],
)
def test_double_to_uint64_matches_measured_cpp_conversions(d, want):
    """Pins the exact (uint64_t)double conversions measured directly
    against a compiled C++ snippet on this project's reference toolchain
    (see similarity.py's own docstring for the measurement)."""
    assert _double_to_uint64(d) == want


def test_cvttsd2si_signed_out_of_range_gives_int64_min():
    assert _cvttsd2si_signed(2.0**63) == -(2**63)
    assert _cvttsd2si_signed(-(2.0**63) - 1.0) == -(2**63)
    assert _cvttsd2si_signed(float("nan")) == -(2**63)


def test_cvttsd2si_signed_in_range_truncates_toward_zero():
    assert _cvttsd2si_signed(3.9) == 3
    assert _cvttsd2si_signed(-3.9) == -3


def test_first_loop_distance_returns_remaining_diff():
    av = {"x": 1}
    bv = {"x": 1, "y": 2, "z": 3}
    dist, diff = _first_loop_distance(av, bv)
    assert dist == 0
    assert diff == {"y": 2, "z": 3}


def test_mutation_using_plain_subtraction_diverges_on_wraparound():
    """Mutation proof: a "fixed" (non-wrapping, mathematically correct)
    implementation of the first loop -- plain `abs(acnt - bcnt)` summed
    normally -- disagrees with classic's own measured, buggy behaviour on
    the exact case this module's docstring cites.
    """

    def fixed_first_loop(av, bv):
        dist = 0
        diff = dict(bv)
        for key in sorted(av):
            acnt = av[key]
            diff.pop(key, None)
            bcnt = bv.get(key, 0)
            dist += abs(acnt - bcnt)
        return dist, diff

    av = {"+(1 +)": 1, "+(1 1)": 1}
    bv = {"+(1 +)": 5, "+(1 1)": 1}
    buggy_dist, _ = _first_loop_distance(av, bv)
    fixed_dist, _ = fixed_first_loop(av, bv)
    assert buggy_dist == 0
    assert fixed_dist == 4
    assert buggy_dist != fixed_dist


def test_toString_sorts_by_key():
    tbv = {"z": 1, "a": 2, "m": 3}
    assert toString(tbv) == "a : 2, m : 3, z : 1"


def test_mutation_unsorted_toString_diverges():
    """Mutation proof: iterating a dict in INSERTION order instead of
    sorted order changes the rendered string for these keys."""

    def unsorted_to_string(tbv):
        return ", ".join(f"{k} : {v}" for k, v in tbv.items())

    tbv = {"z": 1, "a": 2}
    assert toString(tbv) != unsorted_to_string(tbv)
