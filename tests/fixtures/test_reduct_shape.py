"""Shape guard for tests/fixtures/reduct.json (Task 6's oracle capture,
built by tools/capture_reduct.py from the corpus assembled by
tools/cpp_probe/reduct_corpus.py -- see that module's docstring for where
the 103-tree corpus and the 44-vs-46-raw-entries count come from).

This module deliberately does NOT import anything from `pymoses` and does
NOT exercise reduction logic itself -- that is Task 6's job, wired up
against this same fixture. What lives here is purely a guard against the
fixture JSON silently becoming empty, truncated, or reshaped: this
project's `empty_parameter_set_mark = "fail_at_collect"` (pyproject.toml)
already turns a `@pytest.mark.parametrize` over a fully EMPTY corpus into
a hard failure, but it says nothing about a corpus that is merely
SMALLER than intended (e.g. a capture re-run against a half-built probe,
or a hand-edit that drops a source) -- that still collects >0 parametrized
cases and passes quietly. This is the same role
tests/util/test_rng.py's `test_golden_fixture_has_expected_shape` and
tests/comboreduct/combo/test_iostream_combo.py's fixture-shape test play
for their own fixtures ("has bitten this project twice" -- CLAUDE.md/task
brief); this is reduct.json's.
"""

import json
from pathlib import Path

_FIXTURE_PATH = Path(__file__).resolve().parent / "reduct.json"
_FIXTURE = json.loads(_FIXTURE_PATH.read_text())

# Corpus composition, from tools/cpp_probe/reduct_corpus.py:
#   29 testTrees.sh + 6 BooleanReductUTest.cxxtest + 20 BooleanFullUTest.cxxtest
#   + 4 unique FullReductUTest wrapper shapes + 44 GeneralTestCases.py
#   (unique, after deduping the raw list's 46 entries) = 103 trees total;
#   79 runnable (the 24 skipped cases all require contin/mixed reduction).
_EXPECTED_SOURCE_COUNTS = {
    "testTrees.sh": 29,
    "BooleanReductUTest.cxxtest": 6,
    "BooleanFullUTest.cxxtest": 20,
    "FullReductUTest.cxxtest": 4,
    "GeneralTestCases.py": 44,
}
_EXPECTED_CORPUS_TOTAL = 103
_EXPECTED_RUNNABLE = 79
_EXPECTED_SKIPPED = 24
_EXPECTED_EFFORTS = [1, 2, 3]
_EXPECTED_CASES = _EXPECTED_RUNNABLE * len(_EXPECTED_EFFORTS)  # 237

# The 13 testTrees.sh entries whose reduction, per the REAL (unmodified)
# $CLASSIC/build/tests/comboreduct/reduct/test_reduct binary itself, does
# NOT preserve the tree's truth table at any of efforts 1/2/3 -- verified
# directly against that binary (see reduct_corpus.py's own comment for
# the transcript). This is upstream's own oracle disagreeing with itself,
# not a capture bug, so it is pinned down here rather than silently
# tolerated: if this count or set ever drifts, that is a signal the
# corpus or the probe changed underneath this fixture.
# testTrees.sh's `trees` array indices 16..28 (0-based) -- the 13
# entries shaped `and(or())` (an `and` wrapping a single, EMPTY `or()`)
# alongside a trailing empty `and()`. See reduct_corpus.py's
# TEST_TREES_SH_KNOWN_TT_DIVERGENT for the actual tree strings and the
# verification transcript against the real test_reduct binary.
_EXPECTED_KNOWN_TT_DIVERGENT_IDS = {f"testTrees.sh#{i}" for i in range(16, 29)}
assert len(_EXPECTED_KNOWN_TT_DIVERGENT_IDS) == 13


def test_top_level_shape():
    assert set(_FIXTURE) == {"efforts", "sources", "counts", "cases", "skipped"}
    assert _FIXTURE["efforts"] == _EXPECTED_EFFORTS
    assert set(_FIXTURE["sources"]) == set(_EXPECTED_SOURCE_COUNTS)
    assert _FIXTURE["counts"] == {
        "corpus_total": _EXPECTED_CORPUS_TOTAL,
        "runnable": _EXPECTED_RUNNABLE,
        "skipped": _EXPECTED_SKIPPED,
        "cases": _EXPECTED_CASES,
    }


def test_cases_are_non_empty_and_expected_count():
    cases = _FIXTURE["cases"]
    assert len(cases) > 0, "reduct.json's cases list is empty -- capture_reduct.py did not run"
    assert len(cases) == _EXPECTED_CASES


def test_skipped_are_non_empty_and_expected_count():
    skipped = _FIXTURE["skipped"]
    assert len(skipped) > 0, (
        "reduct.json's skipped list is empty -- BooleanFullUTest.cxxtest's 20 "
        "not-yet-runnable trees are missing"
    )
    assert len(skipped) == _EXPECTED_SKIPPED


def test_every_effort_present_per_runnable_tree():
    """Every runnable corpus id must appear exactly once per effort (1, 2,
    3) -- not just "at least one effort", which a partial capture run
    could satisfy while silently missing coverage at effort 2 or 3.
    """
    by_id: dict[str, set[int]] = {}
    for case in _FIXTURE["cases"]:
        by_id.setdefault(case["id"], set()).add(case["effort"])
    assert len(by_id) == _EXPECTED_RUNNABLE
    for tree_id, efforts in by_id.items():
        assert efforts == set(_EXPECTED_EFFORTS), f"{tree_id}: has efforts {sorted(efforts)}"


def test_source_counts_match_corpus():
    """Cross-check per-source case/skip counts against
    tools/cpp_probe/reduct_corpus.py's own corpus composition, independent
    of the aggregate totals asserted above."""
    case_ids_by_source: dict[str, set[str]] = {}
    for case in _FIXTURE["cases"]:
        case_ids_by_source.setdefault(case["source"], set()).add(case["id"])
    skipped_ids_by_source: dict[str, set[str]] = {}
    for entry in _FIXTURE["skipped"]:
        skipped_ids_by_source.setdefault(entry["source"], set()).add(entry["id"])

    skipped_only_sources = {"BooleanFullUTest.cxxtest", "FullReductUTest.cxxtest"}
    for source, expected_total in _EXPECTED_SOURCE_COUNTS.items():
        n_cases = len(case_ids_by_source.get(source, ()))
        n_skipped = len(skipped_ids_by_source.get(source, ()))
        if source in skipped_only_sources:
            assert n_cases == 0, f"{source}: expected 0 runnable cases, got {n_cases}"
            assert n_skipped == expected_total
        else:
            assert n_skipped == 0, f"{source}: expected 0 skipped, got {n_skipped}"
            assert n_cases == expected_total, (
                f"{source}: expected {expected_total} trees, got {n_cases}"
            )


def test_case_entries_have_expected_fields():
    required_common = {"id", "source", "effort", "input", "known_tt_divergent", "ok"}
    for case in _FIXTURE["cases"]:
        assert required_common <= set(case), f"{case.get('id')}: missing fields, has {sorted(case)}"
        assert case["effort"] in _EXPECTED_EFFORTS
        assert isinstance(case["input"], str) and case["input"]
        if case["source"] == "GeneralTestCases.py":
            assert "enf_original" in case
        if case["ok"]:
            assert {"reduced", "size_before", "size_after"} <= set(case)
            assert isinstance(case["reduced"], str)
            # Legacy captures predate the explicit flag and always computed TT.
            if case.get("tt_computed", True):
                assert "tt_preserved" in case
                assert "tt_error" not in case
            else:
                assert "tt_preserved" not in case
                assert isinstance(case.get("tt_error"), str) and case["tt_error"]
            # ostream_combo_tree (and pymoses's combo_tree_to_str) always
            # appends exactly one trailing space per top-level tree -- see
            # tests/fixtures/combo_roundtrip.json's own committed entries.
            # This fixture preserves that verbatim for byte-identical
            # comparison; a captured entry missing it would signal the
            # probe (or a hand-edit) stripped something it should not have.
            assert case["reduced"].endswith(" "), case["id"]
            assert case["size_before"] >= case["size_after"] >= 1
        else:
            assert "error" in case and isinstance(case["error"], str) and case["error"]


def test_skipped_entries_have_expected_fields():
    required = {"id", "source", "input", "reason"}
    for entry in _FIXTURE["skipped"]:
        assert required <= set(entry)
        assert entry["source"] in {"BooleanFullUTest.cxxtest", "FullReductUTest.cxxtest"}
        assert "0<(" in entry["input"], (
            f"{entry['id']}: expected a 0<(...) contin predicate (the documented "
            "reason this source is not-yet-runnable), found none"
        )
        assert "0<(...)" in entry["reason"] or "contin" in entry["reason"]


def test_known_tt_divergent_ids_match_and_are_internally_consistent():
    """Pins down exactly which corpus entries the real C++ oracle itself
    fails to truth-table-preserve on (see reduct_corpus.py's own
    commentary) -- and, independently, that every case flagged
    known_tt_divergent really does have tt_preserved=False (ok=True; the
    reduction itself completes, it just changes the tree's boolean
    function). Every unflagged case in this FIXED corpus also preserves;
    this is a fixture invariant, not a general classifier guarantee.
    """
    flagged_ids = {c["id"] for c in _FIXTURE["cases"] if c["known_tt_divergent"]}
    assert flagged_ids == _EXPECTED_KNOWN_TT_DIVERGENT_IDS

    for case in _FIXTURE["cases"]:
        if case["known_tt_divergent"]:
            assert case["ok"] is True, case["id"]
            assert case.get("tt_computed", True) is True, case["id"]
            assert case["tt_preserved"] is False, case["id"]
        elif case["ok"] and case.get("tt_computed", True):
            assert case["tt_preserved"] is True, (
                f"{case['id']} (effort {case['effort']}) unexpectedly failed truth-table "
                "preservation and is not in the known-divergent set"
            )
