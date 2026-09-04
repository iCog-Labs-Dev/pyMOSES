"""C++: tests/moses/scoringUTest.cxxtest

Transliteration of scoringUTest.cxxtest (85 lines). Test names are
preserved 1:1 with the C++ file so a failure maps back to its original.

Only `test_ctruth_table_bscore_best_possible_bscore` is a live port --
`test_precision_bscore_best_possible_bscore` is kept as a named, explicitly
skipped stub: `precision_bscore` is out of scope for this task
(task-6-brief.md ports `ctruth_table_bscore` only; see bscores.py's module
docstring). Skipping-by-name rather than omitting keeps the 1:1 mapping to
the C++ file visible in `pytest -v` output, matching test_interpreter.py's
own convention for out-of-scope cxxtest methods.

Beyond the transliterated cxxtest, this module also carries the
differential oracle against every case in tests/fixtures/score.json
(captured by tools/capture_score.py from tools/cpp_probe/score_probe.cc)
for `ctruth_table_bscore`'s full ported surface: `__call__`,
`best_possible_bscore`, `worst_possible_bscore`, `min_improv`,
`get_complexity`, `get_complexity_coef`.

The oracle asserts EXACT (`==`) equality against the probe's `%.17g`
output, not `pytest.approx` -- now that `score_t` is genuinely narrowed to
IEEE-754 binary32 at every point classic's own arithmetic narrows
(types.py's `to_score_t`), this port's double-precision Python floats
hold the exact same bit pattern (widened) as the C++ `score_t` values the
JSON round-trips, so anything less than exact equality would silently
tolerate a real regression back to double-only arithmetic.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.moses.scoring.bscores import ctruth_table_bscore

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_SCORE_CASES = json.loads((_FIXTURE_DIR / "score.json").read_text())
_TABLE_DATA_DIR = _FIXTURE_DIR / "table_data"

T = builtin.logical_true
F = builtin.logical_false


def _ctable_for(case: dict):
    path = _TABLE_DATA_DIR / case["file"]
    table = loadTable(str(path), case["target_feature"])
    return table.compressed(case["weighting_feature"])


def _row_from_json(inputs: list[bool]) -> tuple:
    return tuple(T if v else F for v in inputs)


# ---------------------------------------------------------------------------
# scoringUTest.cxxtest, transliterated
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="precision_bscore is out of scope for this task -- task-6-brief.md "
    "ports ctruth_table_bscore only; see bscores.py's module docstring."
)
def test_precision_bscore_best_possible_bscore():
    """C++: tests/moses/scoringUTest.cxxtest:46 test_precision_bscore_best_possible_bscore"""


def test_ctruth_table_bscore_best_possible_bscore():
    """C++: tests/moses/scoringUTest.cxxtest:58 test_ctruth_table_bscore_best_possible_bscore

    `expected_bsc = {0.0, 0.0, -2.0}` (scoringUTest.cxxtest:65) is compared
    here as a SORTED multiset rather than TS_ASSERT_EQUALS's
    position-for-position vector equality: `TS_ASSERT_EQUALS(bsc,
    expected_bsc)` relies on C++'s `CTable` (a `std::map`) iterating rows
    in sorted-key order, an ordering guarantee this port's `CTable` (a
    plain, insertion-ordered `dict` -- table.py's own class docstring)
    does not reproduce (see tools/cpp_probe/score_probe.cc's own module
    comment on exactly this point). The *set* of per-row best-possible
    values does not depend on row order, so this is the faithful
    equivalent of the original assertion in a port where dict order is
    not meaningful.
    """
    path = _TABLE_DATA_DIR / "precision.csv"
    table = loadTable(str(path), "target")
    ct = table.compressed()

    ctbsc = ctruth_table_bscore(ct)

    bsc = ctbsc.best_possible_bscore()
    expected_bsc = [0.0, 0.0, -2.0]
    assert sorted(bsc) == sorted(expected_bsc)


# ---------------------------------------------------------------------------
# Differential oracle against tools/cpp_probe/score_probe.cc
# ---------------------------------------------------------------------------


def test_score_fixture_has_expected_shape():
    """Guards against two distinct ways this corpus could silently stop
    testing anything: an EMPTY (or truncated) score.json -- caught by
    `pyproject.toml`'s `empty_parameter_set_mark = "fail"` globally, but
    checked directly here too, matching test_table_io.py's
    `test_golden_fixture_has_expected_shape` convention -- and a corpus
    that is merely non-empty but too NARROW to exercise the float32-
    narrowing fix this file's oracle exists to guard (a fix-round review
    found the float32-vs-double gap was undetectable at complexity_ratio
    2.0 alone, since 1/2.0 is exactly representable in binary32; see
    tools/capture_score.py's own comment on why 3.5/7.0 and complexity
    >=79 are both required to make the gap exceed 1e-6).
    """
    assert len(_SCORE_CASES) >= 90
    ratios = {c["complexity_ratio"] for c in _SCORE_CASES.values() if c["complexity_ratio"]}
    assert {3.5, 7.0} <= ratios
    assert max(c["tree_complexity"] for c in _SCORE_CASES.values()) >= 79
    for name, case in _SCORE_CASES.items():
        assert set(case) >= {
            "rows",
            "sum_bscore",
            "tree_complexity",
            "complexity_coef",
            "min_improv",
            "composite_score",
            "file",
            "target_feature",
            "weighting_feature",
            "complexity_ratio",
            "tree",
        }, name


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_SCORE_CASES), ids=sorted(_SCORE_CASES))
def test_ctruth_table_bscore_matches_cpp_probe(case_name):
    case = _SCORE_CASES[case_name]
    ctable = _ctable_for(case)
    tr = str_to_combo_tree(case["tree"])

    bsc = ctruth_table_bscore(ctable)
    if case["complexity_ratio"] is not None:
        bsc.set_complexity_coef(case["complexity_ratio"])

    bs = bsc(tr)
    best = bsc.best_possible_bscore()
    worst = bsc.worst_possible_bscore()

    # Associate by ROW KEY, not position -- see
    # test_ctruth_table_bscore_best_possible_bscore's docstring above for
    # why: this port's CTable does not reproduce std::map's sorted
    # iteration order, but bs/best/worst were all produced by iterating
    # THIS port's own `ctable` object, so they agree with each other
    # (and with a fresh `ctable.keys()` iteration) positionally.
    actual_by_key = dict(zip(ctable.keys(), zip(bs, best, worst, strict=True), strict=True))
    expected_by_key = {
        _row_from_json(row["inputs"]): (
            row["bscore"],
            row["best_possible_bscore"],
            row["worst_possible_bscore"],
        )
        for row in case["rows"]
    }

    assert actual_by_key.keys() == expected_by_key.keys()
    for key, (exp_b, exp_be, exp_w) in expected_by_key.items():
        act_b, act_be, act_w = actual_by_key[key]
        assert act_b == exp_b, f"{case_name}: bscore at {key}"
        assert act_be == exp_be, f"{case_name}: best_possible at {key}"
        assert act_w == exp_w, f"{case_name}: worst_possible at {key}"

    assert bsc.sum_bscore(bs) == case["sum_bscore"]
    assert bsc.get_complexity(tr) == case["tree_complexity"]
    assert bsc.get_complexity_coef() == case["complexity_coef"]
    assert bsc.min_improv() == case["min_improv"]
