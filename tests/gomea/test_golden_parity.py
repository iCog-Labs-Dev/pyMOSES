"""Parity of pymoses.gomea.stats / fos with GP-GOMEA's own C++ code.

tests/fixtures/gomea_golden.json is produced by tools/gomea_golden/capture.py,
which compiles tools/gomea_golden/golden.cpp -- verbatim copies of
GP-GOMEA@750220c's ``DetermineNearestNeighbour``,
``BuildLinkageTreeFromSimilarityMatrix`` and the entropy / adjustment loops
of ``GenerateLinkageTreeFOS`` -- and records their output. Similarity
matrices must agree to 1e-12; FOS lists must be identical.
"""

import hashlib
import json
from pathlib import Path

import pytest

from pymoses.gomea.fos import linkage_tree_fos
from pymoses.gomea.stats import biased_similarity, entropy_matrix, mi_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "gomea_golden.json"
GOLDEN = json.loads(FIXTURE.read_text())
CASES = GOLDEN["cases"]
TOL = 1e-12


def _assert_matrix_close(actual, expected):
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected, strict=True):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e, strict=True):
            assert abs(a - e) <= TOL, (a, e)


def test_fixture_provenance():
    assert GOLDEN["upstream"]["commit"] == "750220c23d599e515ca337dcca617ae5faab8b4d"
    assert GOLDEN["upstream"]["file"] == "src/GOMEA/GOMEAFOS.cpp"
    harness = REPO_ROOT / "tools" / "gomea_golden" / "golden.cpp"
    assert GOLDEN["harness_sha256"] == hashlib.sha256(harness.read_bytes()).hexdigest()
    assert GOLDEN["compiler"].startswith("g++")


def test_fixture_covers_required_shapes():
    assert len(CASES) >= 12
    dims = {len(c["order"]) for c in CASES}
    assert {2, 3, 12, 30} <= dims
    tags = {t for c in CASES for t in c["tags"]}
    required = {"unequal_alphabets", "dependent", "second_generation", "negative_biased", "ties"}
    assert required <= tags
    negative = [
        c for c in CASES
        if any(v < -1e-9 for i, row in enumerate(c["S_b"]) for j, v in enumerate(row) if i != j)
    ]
    assert negative, "no golden case produces a negative biased similarity"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_entropy_matches_golden(case):
    d = len(case["order"])
    _assert_matrix_close(entropy_matrix(case["pop1"], range(d)), case["H"])
    _assert_matrix_close(entropy_matrix(case["pop0"], range(d)), case["H0"])


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_mi_similarity_and_tree_match_golden(case):
    d = len(case["order"])
    S = mi_similarity(entropy_matrix(case["pop1"], range(d)))
    _assert_matrix_close(S, case["S_mi"])
    fos = linkage_tree_fos(S, case["order"])
    assert fos == case["lt_mi_full"]
    assert fos[:-1] == case["lt_mi"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_biased_similarity_and_tree_match_golden(case):
    d = len(case["order"])
    H = entropy_matrix(case["pop1"], range(d))
    H0 = entropy_matrix(case["pop0"], range(d))
    S = biased_similarity(H, H0)
    _assert_matrix_close(S, case["S_b"])
    fos = linkage_tree_fos(S, case["order"])
    assert fos == case["lt_b_full"]
    assert fos[:-1] == case["lt_b"]
