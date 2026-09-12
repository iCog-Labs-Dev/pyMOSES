"""`CTable.balance()` (table.cc:806) and the contin-target discretization
pair `discretize_contin_feature`/`get_discrete_bin` (table.cc:573/:583).

Oracle: tests/fixtures/ctable_balance.json, captured from
tools/cpp_probe/ctable_balance_probe.cc by tools/capture_ctable_balance.py
against the real classic `libcomboreduct` -- not reconstructed by hand.

The load-bearing claim these tests exist to pin is that **classic's
`balance()` changes nothing**: both of its C++ range-for loops bind by
value, so the rescaling it computes is written to temporaries. A port that
"helpfully" implemented a working balance would disagree with every classic
`--balance` run, so `test_balance_multipliers_are_not_all_one` proves the
no-op is non-vacuous (a real balance WOULD move these numbers) before
`test_balance_changes_nothing` asserts it does not happen.
"""

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table import (
    TARGET_DISCRETIZED_BINS_NUM,
    discretize_contin_feature,
    get_discrete_bin,
)
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.comboreduct.type_checker.type_signature import type_node
from pymoses.util.logger import logger

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = json.loads((_FIXTURE_DIR / "ctable_balance.json").read_text())
_BALANCE_CASES = _GOLDEN["balance"]
_DISCRETIZE = _GOLDEN["discretize"]


def _vertex_from_json(v):
    """JSON booleans came from the probe's own typed vertex encoder; a
    `bool` there is a `builtin` logical constant, everything else (float,
    str) round-trips as itself.
    """
    if isinstance(v, bool):
        return builtin.logical_true if v else builtin.logical_false
    return v


def _load_ctable(case):
    path = _FIXTURE_DIR / "table_data" / case["file"]
    table = loadTable(str(path), case["target_feature"], "", [])
    return table.compressed("")


def _snapshot(ctable):
    """A comparable, order-insensitive image of every cell's counts."""
    return {row: dict(counter) for row, counter in ctable.items()}


# ---------------------------------------------------------------------------
# balance()
# ---------------------------------------------------------------------------


def test_golden_fixture_has_expected_shape():
    assert len(_BALANCE_CASES) > 0
    for name, case in _BALANCE_CASES.items():
        assert set(case) >= {
            "output_type",
            "class_count",
            "n_classes",
            "uncompressed_size_before",
            "uncompressed_size_after",
            "ctable_before",
            "ctable_after",
            "balance_changed_anything",
            "file",
            "target_feature",
        }, name
    assert len(_DISCRETIZE["cases"]) > 0


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_BALANCE_CASES), ids=sorted(_BALANCE_CASES))
def test_classic_balance_changed_nothing(case_name):
    """Classic's own measured answer: for every case in the corpus,
    `ctable_before == ctable_after`. This is the fixture-side half of the
    claim -- if a future re-capture ever showed `true` here, the port would
    have to change with it.
    """
    case = _BALANCE_CASES[case_name]
    assert case["balance_changed_anything"] is False
    assert case["ctable_before"] == case["ctable_after"]
    assert case["uncompressed_size_before"] == case["uncompressed_size_after"]


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_BALANCE_CASES), ids=sorted(_BALANCE_CASES))
def test_balance_changes_nothing(case_name):
    case = _BALANCE_CASES[case_name]
    ctable = _load_ctable(case)
    before = _snapshot(ctable)
    usize_before = ctable.uncompressed_size()
    assert before, case_name  # a vacuous (empty) table would pass everything below

    ctable.balance()

    assert _snapshot(ctable) == before, case_name
    assert ctable.uncompressed_size() == usize_before


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_BALANCE_CASES), ids=sorted(_BALANCE_CASES))
def test_post_balance_ctable_matches_classic(case_name):
    """The port's post-`balance()` table equals the one classic's own
    probe dumped after calling `CTable::balance()`, cell for cell.
    Order-insensitive: this port's `CTable` is an insertion-ordered dict
    where classic's is a `std::map` sorted on `multi_type_seq`.
    """
    case = _BALANCE_CASES[case_name]
    ctable = _load_ctable(case)
    ctable.balance()

    expected = {}
    for row in case["ctable_after"]:
        key = tuple(_vertex_from_json(v) for v in row["inputs"])
        expected[key] = {_vertex_from_json(v): c for v, c in row["counts"]}

    actual = _snapshot(ctable)
    assert len(actual) == len(expected), case_name
    for key, counts in expected.items():
        assert key in actual, f"{case_name}: missing row {key!r}"
        assert actual[key].keys() == counts.keys(), case_name
        for v, expected_count in counts.items():
            assert actual[key][v] == pytest.approx(expected_count)


@pytest.mark.parametrize(
    "case_name",
    [n for n, c in _BALANCE_CASES.items() if c["output_type"] == "boolean"],
)
def test_balance_multipliers_are_not_all_one(case_name):
    """Proves the no-op is NOT vacuous: on every discrete case in the
    corpus the multiplier `(usize/n)/class_count[v]` that `balance()`
    computes and discards is genuinely different from 1.0 for at least one
    class, so a working balance would visibly rescale the table and this
    corpus would detect it.

    (`imbalanced_bool.csv` is the extreme: 7 `true` / 1 `false` over 8 rows
    gives multipliers 4/7 and 4/1.)
    """
    case = _BALANCE_CASES[case_name]
    class_count = {_vertex_from_json(v): c for v, c in case["class_count"]}
    usize = case["uncompressed_size_before"]
    n = float(len(class_count))
    multipliers = [(usize / n) / c for c in class_count.values()]
    assert any(m != 1.0 for m in multipliers), case_name


def test_balance_on_contin_output_takes_the_warn_branch(monkeypatch):
    """`get_output_type() == contin_type` selects the `else` arm
    (table.cc:834-836), which logs and returns.

    The counts-unchanged assertion alone CANNOT tell the two arms apart --
    both leave the table untouched -- so the branch is asserted through the
    only thing that actually distinguishes them: the `logger().warn()`
    call and its message. Without this, "take the discrete branch for a
    contin table" is an equivalent mutant.
    """
    warned: list[tuple] = []
    monkeypatch.setattr(logger(), "warn", lambda txt, *a: warned.append((txt, a)))

    case = _BALANCE_CASES["contin_regression_sample.csv"]
    ctable = _load_ctable(case)
    assert ctable.get_output_type() is type_node.contin_type
    before = _snapshot(ctable)
    ctable.balance()
    assert _snapshot(ctable) == before

    assert len(warned) == 1
    txt, args = warned[0]
    assert txt % args == ("CTable::balance() - cannot balance non discrete output type contin")


def test_balance_on_discrete_output_does_not_warn(monkeypatch):
    """The mirror of the test above: the boolean arm must NOT log the
    "cannot balance" warning. Together the two pin which branch ran.
    """
    warned: list[tuple] = []
    monkeypatch.setattr(logger(), "warn", lambda txt, *a: warned.append((txt, a)))
    ctable = _load_ctable(_BALANCE_CASES["imbalanced_bool.csv"])
    assert ctable.get_output_type() is type_node.boolean_type
    ctable.balance()
    assert warned == []


def test_balance_on_an_empty_ctable_is_a_no_op():
    """`n = class_count.size()` is 0 for an empty table, and C++ would
    divide by it -- except the loop that divides never runs, since there
    are no rows. Guards against a port that hoists `usize / n` out of the
    loop and raises `ZeroDivisionError` where classic does nothing.
    """
    from pymoses.comboreduct.table.table import CTable, Signature

    ctable = CTable("y", ["i1"], Signature([type_node.boolean_type], type_node.boolean_type))
    ctable.balance()
    assert len(ctable) == 0


# ---------------------------------------------------------------------------
# discretize_contin_feature / get_discrete_bin
# ---------------------------------------------------------------------------

# C++'s own `operator<<(std::ostream&, builtin)` spelling for the five
# enumerators a bin ordinal 1..5 can land on (vertex.h:72-79).
_CPP_BUILTIN_NAME = {
    builtin.logical_true: "true",
    builtin.logical_false: "false",
    builtin.logical_and: "and",
    builtin.logical_or: "or",
    builtin.logical_not: "not",
}


def test_target_discretized_bins_num_matches_classic():
    assert TARGET_DISCRETIZED_BINS_NUM == _DISCRETIZE["TARGET_DISCRETIZED_BINS_NUM"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _DISCRETIZE["cases"],
    ids=[f"{c['min']}..{c['max']}" for c in _DISCRETIZE["cases"]],
)
def test_discretize_contin_feature_matches_classic(case):
    """BIT-exact, not `approx`: the fixture stores each edge as `%.17g`,
    which round-trips a binary64 exactly, and the whole point of the
    comparison is the `min + i*interval` rounding (e.g. `-1..1` gives
    `-0.59999999999999998`, and `1e-8..1e8` gives `20000000.000000007`).
    A `linspace`-style or compensated implementation would produce
    different last bits and pass an `approx` check.
    """
    actual = discretize_contin_feature(case["min"], case["max"])
    assert actual == case["intervals"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _DISCRETIZE["cases"],
    ids=[f"{c['min']}..{c['max']}" for c in _DISCRETIZE["cases"]],
)
def test_get_discrete_bin_matches_classic(case):
    intvs = discretize_contin_feature(case["min"], case["max"])
    assert case["bins"], case  # a case with no probed values would pass vacuously
    for val, expected_ordinal, expected_name in case["bins"]:
        got = get_discrete_bin(intvs, val)
        assert isinstance(got, builtin)
        assert int(got) == expected_ordinal, (case["min"], case["max"], val)
        # `expected_name` is C++'s own `operator<<(ostream&, builtin)`
        # spelling of the SAME enumerator -- checked so a bin ordinal that
        # silently shifted by one could not pass the ordinal check alone.
        assert _CPP_BUILTIN_NAME[got] == expected_name, (case["min"], case["max"], val)


def test_get_discrete_bin_reinterprets_the_ordinal_as_a_builtin():
    """The `(builtin)i` cast is load-bearing and surprising: bin 5 is
    `logical_not`, an OPERATOR token, not a constant. Pinned explicitly so
    a "cleanup" that returned a plain `int` (or an offset ordinal) fails
    here rather than silently changing the keys
    `mutualInformation(CTable, FeatureSet)` counts by.
    """
    intvs = [0.0, 2.0, 4.0, 6.0, 8.0]
    assert get_discrete_bin(intvs, -100.0) is builtin.logical_true  # bin 1
    assert get_discrete_bin(intvs, 2.0) is builtin.logical_false  # bin 2
    assert get_discrete_bin(intvs, 4.0) is builtin.logical_and  # bin 3
    assert get_discrete_bin(intvs, 6.0) is builtin.logical_or  # bin 4
    assert get_discrete_bin(intvs, 8.0) is builtin.logical_not  # bin 5


def test_get_discrete_bin_never_consults_the_bottom_edge():
    """`for (i = 1; ...)` starts at index 1, so `disc_intvs[0]` is dead.
    A value far below the range still lands in bin 1, exactly as a value
    inside the first bin does -- there is no "below range" bin.
    """
    intvs = discretize_contin_feature(0.0, 10.0)
    assert get_discrete_bin(intvs, -1e9) is get_discrete_bin(intvs, 0.5)
    # And the bottom edge itself is never read: replacing it changes nothing.
    poisoned = [float("nan"), *intvs[1:]]
    assert get_discrete_bin(poisoned, -1e9) is get_discrete_bin(intvs, -1e9)
