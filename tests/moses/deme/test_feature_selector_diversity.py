"""Oracle tests for feature_selector's --fs-diversity-pressure branch.

## Oracle: tools/cpp_probe/fs_diversity_probe.cc -> tests/fixtures/fs_diversity.json

The fixture's `"rank"` group calls classic's own
`feature_selector::rank_feature_sets` (feature_selector.cc:343) and
`feature_selector::select_top_feature_sets` (cc:235) through a subclass
that merely un-protects them, over hand-built `feature_set_pop`s. Its
`"end_to_end"` group calls `feature_selector::operator()` with
`diversity_pressure > 0` and/or `subsampling_ratio < 1`.

## Why the populations look the way they do

Ties are where the ordering bugs live, so the corpus is built to have
them rather than to look realistic:

* three equal-score singletons, in two different INSERTION orders (a
  `std::multimap` keeps insertion order among equal keys, and
  `std::max_element` keeps the first maximum -- both are order facts a
  hash-ordered or unstable implementation gets wrong);
* the same feature set twice at the same score (`dup_set_jaccard`), which
  is the only shape where the jaccard index reaches 1.0;
* `bool_a5_maj.csv`, a 5-input MAJORITY function -- every input column is
  interchangeable, so every same-size feature set has EXACTLY the same
  mutual information with the target and with every other same-size set.
  That symmetry is what makes the `a5_*` cases exercise tie-breaking
  rather than merely sorting.

`fs_mi_skewed.csv` exists for the OPPOSITE reason and was added after a
mutation run showed why it was needed. Every other boolean fixture table
in this repo is a full truth-table enumeration, so mutual information
between its input sets is always a whole number of bits -- exact in both
`double` and `float`, and exact in any summation order. Three real traps
were therefore zero-kill: `float sim = mi(...)`'s narrowing, `powerset`'s
`std::set` ordering, and the summation order in `mi()` itself. The skewed
table is 11 hand-picked rows with correlated, unbalanced columns, so its
mutual informations are irrational and all three become observable.

## What is NOT covered, and why

No end-to-end case drives `rank_feature_sets` over a population of more
than one. `simple`/`inc`/`smd`/`random` all `return {sfs}` -- a
single-entry `feature_set_pop` -- and `hc`, the only algorithm that can
return more, SEGFAULTS in classic itself on every invocation (measured
again while building this probe, rc 139; see
tools/cpp_probe/fs_standalone_probe.cc's header comment). So classic has
no reachable end-to-end route to a multi-set ranked population either,
and the `"rank"` group calls the method directly instead.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.deme.feature_selector import (
    feature_selector,
    feature_selector_parameters,
    jaccard_index,
    powerset,
)
from pymetta.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "fs_diversity.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


@pytest.fixture(scope="module")
def tables() -> dict[str, CTable]:
    return {
        "boolean": loadTable(str(_TABLE_DATA_DIR / "trainingBoolean.dat"), "y").compressed(),
        "a5": loadTable(str(_TABLE_DATA_DIR / "bool_a5_maj.csv"), "y").compressed(),
        "skew": loadTable(str(_TABLE_DATA_DIR / "fs_mi_skewed.csv"), "y").compressed(),
    }


def _festor(tables, case) -> feature_selector:
    params = feature_selector_parameters(
        diversity_pressure=case["diversity_pressure"],
        diversity_jaccard=case["diversity_jaccard"],
        diversity_interaction=case["diversity_interaction"],
        n_demes=case["n_demes"],
    )
    return feature_selector(tables[case["table"]], params)


def _pop(case) -> list[tuple[float, set[int]]]:
    """The fixture records the input population in MULTIMAP order (what
    classic's own `feature_set_pop` iterates), which is exactly the order
    this port's list-shaped `feature_set_pop` must be in.
    """
    return [(score, set(fs)) for score, fs in case["pop"]]


# --------------------------------------------------------------- guards


@pytest.mark.oracle
def test_corpus_contains_real_ties(golden):
    """A feature-selection corpus where every feature scores differently
    does not exercise tie-breaking. Assert the corpus does.
    """
    with_score_ties = 0
    with_penalty_ties = 0
    for case in golden["rank"]:
        scores = [score for score, _fs in case["pop"]]
        if len(set(scores)) < len(scores):
            with_score_ties += 1
        penalized = [entry["cs"]["penalized_score"] for entry in case["ranked"]]
        if len(set(penalized)) < len(penalized):
            with_penalty_ties += 1
    assert with_score_ties >= 10, f"only {with_score_ties} case(s) have equal input scores"
    assert with_penalty_ties >= 6, (
        f"only {with_penalty_ties} case(s) end with equal PENALIZED scores -- "
        "those are the ones that actually reach composite_score's tie-break"
    )


@pytest.mark.oracle
def test_corpus_actually_reorders(golden):
    """At least a few cases must come out of `rank_feature_sets` in an
    order DIFFERENT from the order they went in, or "keep insertion order"
    would pass the whole corpus.
    """
    reordered = sum(
        1
        for case in golden["rank"]
        if [sorted(fs) for _score, fs in case["pop"]]
        != [sorted(entry["fs"]) for entry in case["ranked"]]
    )
    assert reordered >= 3, f"only {reordered} case(s) are reordered by ranking"


# ------------------------------------------------------- rank_feature_sets


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case_index", "name"),
    [(i, c["name"]) for i, c in enumerate(_GOLDEN["rank"])],
)
def test_rank_feature_sets_matches_classic(golden, tables, case_index, name):
    case = golden["rank"][case_index]
    ranked = _festor(tables, case).rank_feature_sets(_pop(case))

    assert len(ranked) == len(case["ranked"])
    for got, want in zip(ranked, case["ranked"], strict=True):
        cs, fs = got
        assert sorted(fs) == want["fs"], name
        assert cs.get_score() == pytest.approx(want["cs"]["score"], abs=0.0, rel=0.0), name
        assert cs.get_complexity() == want["cs"]["complexity"], name
        assert cs.get_uniformity_penalty() == want["cs"]["uniformity_penalty"], name
        assert cs.get_penalized_score() == want["cs"]["penalized_score"], name


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case_index", "name"),
    [(i, c["name"]) for i, c in enumerate(_GOLDEN["rank"])],
)
def test_select_top_feature_sets_matches_classic(golden, tables, case_index, name):
    case = golden["rank"][case_index]
    top = _festor(tables, case).select_top_feature_sets(_pop(case))

    assert len(top) == len(case["top"]), name
    for (score, fs), (want_score, want_fs) in zip(top, case["top"], strict=True):
        assert sorted(fs) == want_fs, name
        assert score == want_score, name


@pytest.mark.oracle
def test_scores_are_narrowed_to_binary32(golden, tables):
    """`composite_score(fs.first, fs.second.size())` narrows the multimap's
    `double` key to `score_t` (binary32). Measured in the fixture: a raw
    0.9 comes back as 0.89999997615814209. Without the narrowing every
    such case reports 0.9 and this assertion fails.
    """
    case = next(c for c in golden["rank"] if c["name"] == "distinct_scores_jaccard")
    ranked = _festor(tables, case).rank_feature_sets(_pop(case))
    raw = [score for score, _fs in case["pop"]]
    narrowed = [cs.get_score() for cs, _fs in ranked]
    assert narrowed != raw
    assert narrowed == [entry["cs"]["score"] for entry in case["ranked"]]


# --------------------------------------------------------------- helpers


@pytest.mark.oracle
def test_jaccard_index_edges():
    assert jaccard_index({0, 1}, {1, 2}) == pytest.approx(1.0 / 3.0, abs=1e-7)
    assert jaccard_index({0}, {0}) == 1.0
    assert jaccard_index({0}, {1}) == 0.0
    # DIVERGENCE reproduced: 0/0 is nan in C++, not an exception.
    assert jaccard_index(set(), set()) != jaccard_index(set(), set())


def test_powerset_matches_cpp_shape():
    """`std::set<Set>` iteration order is ascending lexicographic over each
    subset's own ascending elements -- and `mi()` sums in that order.
    """
    assert powerset({2, 0, 1}, 1, True) == [frozenset({0}), frozenset({1}), frozenset({2})]
    assert powerset({2, 0, 1}, 2, True) == [
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 2}),
    ]
    assert powerset({0, 1}, 2, False) == [
        frozenset(),
        frozenset({0}),
        frozenset({0, 1}),
        frozenset({1}),
    ]
    # exact=True with n > |s| is EMPTY in C++ too -- see powerset's docstring.
    assert powerset({0, 1}, 3, True) == []


@pytest.mark.oracle
def test_mi_zero_denominator_returns_nan_not_an_exception(tables):
    """`mi / (pfs_l.size() * pfs_r.size())` with an empty powerset is
    0.0/0 -- nan in C++, ZeroDivisionError in bare Python.
    """
    params = feature_selector_parameters(diversity_pressure=1.0, diversity_interaction=3)
    fs = feature_selector(tables["boolean"], params)
    value = fs.mi({0, 1}, {1, 2})
    assert value != value  # nan


# ------------------------------------------------------------------- mi


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["mi"])))
def test_mi_matches_classic(golden, tables, case_index):
    """`feature_selector::mi` at FULL double precision -- `rank_feature_sets`
    narrows its result to a `float`, which hides the last-ULP differences a
    wrong `powerset` order or a wrong summation order produces.
    """
    case = golden["mi"][case_index]
    params = feature_selector_parameters(
        diversity_pressure=1.0,
        diversity_jaccard=False,
        diversity_interaction=case["diversity_interaction"],
    )
    fs = feature_selector(tables[case["table"]], params)
    assert fs.mi(set(case["fs_l"]), set(case["fs_r"])) == case["mi"]


@pytest.mark.oracle
def test_mi_corpus_is_not_all_whole_bits(golden):
    """Guard: on a full truth-table enumeration every mutual information is
    a whole number of bits, and float32 narrowing / summation order are
    then unobservable. The corpus must contain irrational values.
    """
    irrational = sum(1 for c in golden["mi"] if c["mi"] != int(c["mi"]))
    assert irrational >= 12, f"only {irrational} mi case(s) are not whole bits"


# -------------------------------------------------------------- end to end


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["end_to_end"])))
def test_operator_call_matches_classic(golden, tables, case_index):
    case = golden["end_to_end"][case_index]
    params = feature_selector_parameters(
        increase_target_size=case["increase_target_size"],
        diversity_pressure=case["diversity_pressure"],
        diversity_jaccard=case["diversity_jaccard"],
        diversity_interaction=case["diversity_interaction"],
        n_demes=case["n_demes"],
        diversity_cap=case["diversity_cap"],
        subsampling_ratio=case["subsampling_ratio"],
    )
    params.fs_params.algorithm = case["algorithm"]
    params.fs_params.target_size = case["base_target_size"]

    festor = feature_selector(tables[case["table"]], params)
    randGen().seed(case["seed"])
    top = festor(str_to_combo_tree(case["exemplar"]))

    assert len(top) == len(case["top"])
    for (score, fs), (want_score, want_fs) in zip(top, case["top"], strict=True):
        assert sorted(fs) == want_fs
        assert score == want_score


@pytest.mark.oracle
def test_subsampling_guard_is_strictly_less_than_one(tables):
    """`if (params.subsampling_ratio < 1.0)` (feature_selector.cc:206) is a
    STRICT comparison, and the difference is observable: at ratio exactly
    1.0 classic never calls `subsampleCTable` at all, so an EMPTY
    `fs_ctable` is fine. Relaxing the guard to `<=` would construct a
    `lazy_random_selector(0)`, whose own `OC_ASSERT(u - l > 0)`
    (lazy_selector.cc:41) fires.

    An empty `fs_ctable` is reachable: `restrict_true` keeps only the rows
    where the exemplar predicts true, and `false` predicts true nowhere.
    """
    params = feature_selector_parameters(restrict_true=True, restrict_incorrect=False)
    fs = feature_selector(tables["boolean"], params)
    xmplr = str_to_combo_tree("false")
    fs.preprocess_params(xmplr)
    assert len(fs.build_fs_ctable(xmplr)) == 0


@pytest.mark.oracle
def test_end_to_end_corpus_exercises_both_gates(golden):
    """Guard: the end-to-end group must contain cases with
    diversity_pressure > 0 AND cases with subsampling_ratio < 1, or it
    proves nothing about either gate being wired into `operator()`.
    """
    pressured = sum(1 for c in golden["end_to_end"] if c["diversity_pressure"] > 0.0)
    subsampled = sum(1 for c in golden["end_to_end"] if c["subsampling_ratio"] < 1.0)
    assert pressured >= 3
    assert subsampled >= 3
