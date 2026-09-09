"""C++: tests/comboreduct/table/tableUTest.cxxtest

Transliteration of tableUTest.cxxtest (437 lines). Test names are preserved
1:1 with the C++ file so a failure maps back to its original (ruff's naming
rules are disabled repo-wide for exactly this reason -- see pyproject.toml).

`testIStreamTable` (cxxtest, lines 37-55) feeds a `stringstream` through
`istreamTable`; this port's `loadTable` takes a file path (matching the
brief's interface, `loadTable(path, ...)`), so `_load` below writes the
same text to a temp file first -- same parsing pipeline, same bytes, only
the plumbing to get them there differs.

Three tests are intentionally NOT ported, each for a symbol genuinely
outside this task's produced interface (`ITable`, `OTable`, `Table`,
`CTable`, `loadTable` -- task-5-brief.md's Interfaces section):
`test_loadCTable` (needs `loadCTable`, a distinct entry point that reads a
pre-compressed `{v:n,...}` file format loadTable never touches),
`test_mutualInformation` and `test_mutualInformationBtwSets` (need
`mutualInformation`/`mutualInformationBtwSets`, entropy/feature-selection
utilities declared in table.h but unrelated to loading or compressing a
table). Kept as named, skipped stubs so `pytest -v` still shows the 1:1
mapping to the C++ file rather than silently vanishing (same convention as
test_interpreter.py's `test_contin_interpreter`/`test_mixed_interpreter`).

Two tests are ported in ADAPTED form because they assert on `ostreamTable`/
`table.ttable`, neither of which exists in this port (table.py's module
docstring: no `TTable`, no stream-printing methods are in this task's
Produces list) -- `test_filterTable`, `test_timestamp_feature` and
`test_ostream_timestamp_table` check the underlying `itable`/`otable`/
`target_pos` facts instead of a printed string or a `TTable` object. Each
adapted test says so at the point of adaptation.
"""

from collections import Counter
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import CTable, ITable, OTable, Table, type_node
from pymetta.comboreduct.table.table_io import loadTable

olabel = "happiness"
label1 = "price"
label2 = "temperature"

T = builtin.logical_true
F = builtin.logical_false


def _load(
    tmp_path: Path,
    text: str,
    target_feature: str = "",
    timestamp_feature: str = "",
    ignore_features: tuple[str, ...] = (),
) -> Table:
    """C++: tests/comboreduct/table/tableUTest.cxxtest:37 testIStreamTable

    `text` stands in for the C++ helper's `stringstream ss(ts)`; see
    module docstring for why a temp file is used instead of an in-memory
    stream.
    """
    path = tmp_path / "table_data.csv"
    path.write_text(text)
    return loadTable(str(path), target_feature, timestamp_feature, list(ignore_features))


# ---------------------------------------------------------------------------
# tableUTest.cxxtest, transliterated
# ---------------------------------------------------------------------------


def test_TruthTable(tmp_path):
    """C++: tableUTest.cxxtest:64 test_TruthTable"""
    o = [T, F]
    ot = OTable(o)
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type])
    it.push_back((T, F))
    it.push_back((F, T))
    expected = Table(ot, it)
    table = _load(tmp_path, "1,1,0\n0,0,1\n")
    assert table == expected


def test_ContinTable(tmp_path):
    """C++: tableUTest.cxxtest:77 test_ContinTable"""
    o = [10.1, 6.5]
    ot = OTable(o)
    it = ITable(types=[type_node.contin_type, type_node.contin_type])
    it.push_back((0.4, 244.2))
    it.push_back((9.5, 4.2))
    expected = Table(ot, it)
    table = _load(tmp_path, "10.1,0.4,244.2\n6.5,9.5,4.2\n")
    assert table == expected


def test_ContinTable_type_inference(tmp_path):
    """C++: tableUTest.cxxtest:89 test_ContinTable_type_inference"""
    o = [0.0, 1.0, 10.1, 6.5]
    ot = OTable(o)
    it = ITable(types=[type_node.contin_type, type_node.contin_type])
    it.push_back((1.0, 1.0))
    it.push_back((0.0, 1.0))
    it.push_back((0.4, 244.2))
    it.push_back((9.5, 4.2))
    expected = Table(ot, it)
    table = _load(tmp_path, "0,1,1\n1,0,1\n10.1,0.4,244.2\n6.5,9.5,4.2\n")
    assert table == expected


def test_ContinTableLabels(tmp_path):
    """C++: tableUTest.cxxtest:105 test_ContinTableLabels"""
    o = [0.0, 10.1, 6.5]
    ot = OTable(o, olabel)
    it = ITable(types=[type_node.contin_type, type_node.contin_type])
    it.push_back((1.0, 1.0))
    it.push_back((0.4, 244.2))
    it.push_back((9.5, 4.2))
    it.set_labels([label1, label2])
    expected = Table(ot, it)
    table = _load(tmp_path, "happiness,price,temperature\n0,1,1\n10.1,0.4,244.2\n6.5,9.5,4.2\n")
    assert table == expected


def test_EnumTableLabels(tmp_path):
    """C++: tableUTest.cxxtest:122 test_EnumTableLabels"""
    o = ["sad", "bored", "super-happy"]
    ot = OTable(o, olabel)
    itypes = [type_node.contin_type, type_node.contin_type, type_node.boolean_type]
    il = [label1, label2, "truth"]
    it = ITable(types=itypes, labels=il)
    it.push_back((1.0, 1.0, F))
    it.push_back((0.44, 244.2, T))
    it.push_back((9.5, 4.2, F))
    expected = Table(ot, it)
    table = _load(
        tmp_path,
        "happiness,price,temperature,truth\n"
        "sad,1,1,0\n"
        "bored,0.44,244.2,T\n"
        "super-happy,9.5,4.2,F\n",
    )
    assert table == expected


def test_filterITable():
    """C++: tableUTest.cxxtest:139 test_filterITable"""
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type], labels=["truc", "trac"])
    it.push_back((T, F))
    it.push_back((F, T))
    # ignore second argument trac
    fit_it = it.filtered([0])
    assert fit_it.get_labels() == ["truc"]
    assert list(fit_it) == [(T,), (F,)]


def test_filterTable(fixture_dir):
    """C++: tableUTest.cxxtest:166 test_filterTable

    ADAPTED: classic asserts on `ostreamTable`'s printed string
    ("input1,time,target\\n1,2012-03-11,6\\n0,2012-03-23,7\\n"); this port
    has no `ostreamTable`/`TTable` (table.py's module docstring). Checks
    the same filtering outcome directly on `itable`/`target_pos` instead.
    """
    input_file = fixture_dir / "table_data" / "timestamp_table2.csv"
    table = loadTable(str(input_file), "target", "time")
    filtered_table = table.filtered({1})

    assert filtered_table.itable.get_labels() == ["input1"]
    assert list(filtered_table.itable) == [(T,), (F,)]
    assert list(filtered_table.otable) == [6.0, 7.0]
    assert filtered_table.target_pos == 2


def test_filtered_table_does_not_alias_the_original_otable(fixture_dir):
    """Not a cxxtest port: `Table::filtered` (table.h:987) copies `otable`
    -- a C++ value type -- into the result. `OTable` here is a mutable
    Python list, so `Table.filtered()` must copy it explicitly (see
    `OTable.copy()`) rather than share the same object; otherwise
    mutating one table's otable would silently mutate the other's.
    """
    input_file = fixture_dir / "table_data" / "timestamp_table2.csv"
    table = loadTable(str(input_file), "target", "time")
    filtered_table = table.filtered([1])
    assert filtered_table.otable is not table.otable

    filtered_table.otable.push_back(99.0)
    assert list(table.otable) == pytest.approx([6.0, 7.0])


def test_filterCTable():
    """C++: tableUTest.cxxtest:180 test_filterCTable"""
    it = ITable(types=[type_node.boolean_type, type_node.boolean_type], labels=["truc", "trac"])
    it.push_back((T, F))
    it.push_back((F, T))
    ot = OTable([T, F], "troc")
    table = Table(ot, it)
    ctable = table.compressed()
    # ignore second argument trac
    filtered_ctable = ctable.filtered([0])
    assert filtered_ctable.get_labels() == ["troc", "truc"]
    assert filtered_ctable == {(T,): Counter({T: 1.0}), (F,): Counter({F: 1.0})}


def test_weightCTable(fixture_dir):
    """C++: tableUTest.cxxtest:205 test_weightCTable"""
    input_file = fixture_dir / "table_data" / "weighted.csv"
    tt = loadTable(str(input_file), "r", "", [])
    ctt = tt.compressed("wgt")

    assert ctt.get_labels() == ["r", "a", "b", "c", "d", "e"]
    expected = {
        (T, T, T, T, T): Counter({T: 3.5}),
        (T, T, T, F, F): Counter({T: pytest.approx(2.66)}),
        (F, T, T, T, T): Counter({F: 2.5}),
        (F, F, F, F, F): Counter({T: pytest.approx(0.4), F: pytest.approx(0.8)}),
    }
    assert ctt.keys() == expected.keys()
    for key, counter in expected.items():
        assert ctt[key] == counter


# Set target Price, and ignore column Temperature
def test_target_col_ignore_col(tmp_path):
    """C++: tableUTest.cxxtest:229 test_target_col_ignore_col"""
    o = [0.4, 9.5]  # price column
    ot = OTable(o, label1)
    it = ITable(types=[type_node.contin_type], labels=[olabel])
    it.push_back((10.1,))
    it.push_back((6.5,))
    expected = Table(ot, it)
    expected.target_pos = 1

    table = _load(
        tmp_path,
        "happiness,price,temperature\n10.1,0.4,244.2\n6.5,9.5,4.2\n",
        "price",
        "",
        ("temperature",),
    )
    assert table == expected
    assert table.target_pos == expected.target_pos


def test_ignore_col_complicated(tmp_path):
    """C++: tableUTest.cxxtest:246 test_ignore_col_complicated"""
    o = [1.0, 0.44, 9.5]  # price column
    ot = OTable(o, label1)
    il = [olabel, "truth"]
    itypes = [type_node.enum_type, type_node.boolean_type]
    it = ITable(types=itypes, labels=il)
    it.push_back(("sad", F))
    it.push_back(("bored", T))
    it.push_back(("super-happy", F))
    expected = Table(ot, it)
    expected.target_pos = 1

    table = _load(
        tmp_path,
        "happiness,price,temperature,truth\n"
        "sad,1,1,0\n"
        "bored,0.44,244.2,T\n"
        "super-happy,9.5,4.2,F\n",
        "price",
        "",
        ("temperature",),
    )
    assert table == expected
    assert table.target_pos == expected.target_pos


def test_sparseFile1(fixture_dir):
    """C++: tableUTest.cxxtest:287 test_sparseFile1"""
    input_file = fixture_dir / "table_data" / "sparse.data"
    tt = loadTable(str(input_file), "target1", "", ["a", "b", "c", "target2"])

    assert tt.get_labels() == ["target1", "date", "d", "e", "f", "g", "x", "y", "z"]
    assert list(tt.otable) == ["thing", "some", "other", "last"]
    assert list(tt.itable) == [
        ("12-04-2012", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("12-05-2012", 4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0),
        ("12-06-2012T14:35:02-0500", 0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0),
        ("12-07-2012T16:00:00-0500", 4.0, 5.0, 0.0, 0.0, 24.0, 25.0, 26.0),
    ]


def test_sparseFile2(fixture_dir):
    """C++: tableUTest.cxxtest:305 test_sparseFile2"""
    input_file = fixture_dir / "table_data" / "sparse.data"
    tt = loadTable(str(input_file), "target2", "", ["a", "b", "c", "target1"])

    assert tt.get_labels() == ["target2", "date", "d", "e", "f", "g", "x", "y", "z"]
    assert list(tt.otable) == pytest.approx([3.0, 4.4, 5.5, 6.6])
    assert list(tt.itable) == [
        ("12-04-2012", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("12-05-2012", 4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0),
        ("12-06-2012T14:35:02-0500", 0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0),
        ("12-07-2012T16:00:00-0500", 4.0, 5.0, 0.0, 0.0, 24.0, 25.0, 26.0),
    ]


@pytest.mark.skip(
    reason="mutualInformation is out of scope for task-5-brief.md's Produces "
    "list (ITable/OTable/Table/CTable/loadTable only) -- an entropy/"
    "feature-selection utility, not part of loading or compressing a table."
)
def test_mutualInformation():
    """C++: tableUTest.cxxtest:323 test_mutualInformation"""


@pytest.mark.skip(
    reason="mutualInformationBtwSets is out of scope for task-5-brief.md's "
    "Produces list -- see test_mutualInformation's skip reason."
)
def test_mutualInformationBtwSets():
    """C++: tableUTest.cxxtest:337 test_mutualInformationBtwSets"""


def test_insert_get_pos():
    """C++: tableUTest.cxxtest:353 test_insert_get_pos"""
    it = ITable()
    col1 = [T, F]
    it.insert_col("label1", col1)
    assert it.get_labels() == ["label1"]
    assert it.get_types() == [type_node.boolean_type]
    assert list(it) == [(T,), (F,)]

    col2 = [F, T]
    it.insert_col("label2", col2)
    assert it.get_labels() == ["label1", "label2"]
    assert list(it) == [(T, F), (F, T)]

    lc = it.get_column_data("label2")
    it.insert_col("label3", lc, 0)
    assert it.get_labels() == ["label3", "label1", "label2"]
    assert list(it) == [(F, T, F), (T, F, T)]


@pytest.mark.skip(
    reason="loadCTable is out of scope for task-5-brief.md's Produces list "
    "(ITable/OTable/Table/CTable/loadTable only) -- a distinct entry point "
    "reading a pre-compressed {v:n,...} file format, never called by "
    "loadTable itself."
)
def test_loadCTable():
    """C++: tableUTest.cxxtest:380 test_loadCTable"""


def test_timestamp_feature(fixture_dir):
    """C++: tableUTest.cxxtest:393 test_timestamp_feature

    ADAPTED: classic builds a `TTable` of the three parsed dates and
    asserts `table.ttable == dates`; this port has no `TTable` (table.py's
    module docstring: timestamps are excluded from `itable` but not
    otherwise retained). Checks the thing that IS in scope: the timestamp
    column is excluded from `itable` exactly like an ignored column would
    be, while the target and its values are unaffected.
    """
    input_file = fixture_dir / "table_data" / "timestamp_table.csv"
    table = loadTable(str(input_file), "length", "time")
    assert table.itable.get_labels() == ["width"]
    assert table.otable.get_label() == "length"
    assert list(table.otable) == pytest.approx([12.0, 10.0, 14.0])


def test_ctable_has_no_timestamp_surface(tmp_path):
    """The port's `CTable` carries no timestamp DATA, so it has none of
    classic's three timestamp members -- and the absence is load-bearing,
    not cosmetic.

    Classic's `CTable` is `map<multi_type_seq, TimedCounter>` and its
    `TimedCounter` is `Counter<TimedValue, count_t>`, where `TimedValue`
    is `(vertex value, TTable::value_type timestamp)` with a member-wise
    `operator==` (table.h:472-490,496). `Table::compressed` fills the
    timestamp half only when `ttable` is non-empty (table.cc:534-550),
    which `loadTable` only ever makes it when `--timestamp-feature` was
    given (table_io.cc:1109-1110,1144). This port has no `TTable` at all,
    so every key collapses to a bare vertex and there is nothing for
    `get_timestamps` (table.h:697), `remove_rows_at_time` (table.h:692)
    or `remove_rows_at_times` (table.h:687) to read.

    Why pin the ABSENCE rather than shrug at it: classic's own
    `get_timestamps` skips any default (not-a-date) timestamp
    (table.cc:740), so a vacuous `def get_timestamps(self): return set()`
    would return the numerically correct answer for every table this port
    can build -- and would still be a real defect, because
    `ss_bscore.__init__` (moses/scoring/ss_bscore.py) gates on
    `hasattr(ctable, "get_timestamps")`. Defining one turns that clear
    `NotImplementedError` into an `ss_bscore` that silently believes the
    dataset is untimestamped.

    And measured, not assumed: THIS test is the only thing that catches
    that. `test_construction_raises_because_get_timestamps_is_missing`
    (tests/moses/scoring/test_ss_bscore.py) looks like it guards the same
    property, but it feeds `ss_bscore` a `_CTableDouble` -- an empty
    test-local class that has no `get_timestamps` by construction -- so
    it stays green no matter what the real `CTable` grows. Verified by
    adding a vacuous `CTable.get_timestamps` returning `set()`: this test
    failed, that one did not.
    """
    assert not hasattr(CTable, "get_timestamps")
    assert not hasattr(CTable, "remove_rows_at_time")
    assert not hasattr(CTable, "remove_rows_at_times")
    assert not hasattr(Table(), "ttable")

    # Two rows agreeing on inputs AND output but differing in date. In
    # classic these are two distinct TimedValue keys, each counted 1.0;
    # here they collapse to one bare-vertex key counted 2.0. Scoring is
    # unaffected -- TimedCounter::get(const vertex&) is defined as "the
    # sum of that counter across all timestamps with vertex equal to v"
    # (table.h:497-500), which is exactly what one collapsed entry gives.
    path = tmp_path / "dupe_rows_distinct_dates.csv"
    path.write_text("time,width,length\n2012-Jan-11,6,12\n2012-Feb-18,6,12\n2012-Mar-23,3,10\n")
    table = loadTable(str(path), "length", "time")
    assert table.itable.get_labels() == ["width"]  # timestamp column excluded

    ctable = table.compressed()
    assert dict(ctable) == {(6.0,): Counter({12.0: 2.0}), (3.0,): Counter({10.0: 1.0})}

    # Every counter key is a bare vertex, never a (value, timestamp) pair
    # -- the concrete reason get_timestamps() cannot be implemented here.
    for counter in ctable.values():
        for key in counter:
            assert not isinstance(key, tuple)

    # Total observations are preserved either way, which is why the
    # collapse is scoring-neutral.
    assert ctable.uncompressed_size() == 3.0


def test_ostream_timestamp_table(fixture_dir):
    """C++: tableUTest.cxxtest:411 test_ostream_timestamp_table

    ADAPTED: classic asserts on `ostreamTable`'s printed string (not
    available in this port -- see `test_filterTable`'s adaptation note).
    Checks the same two column-ordering scenarios -- timestamp before the
    target plus an ignored column, and timestamp after the target -- via
    `itable`/`otable`/`target_pos` directly.
    """
    # timestamp feature before target, ignore a second feature
    input_file = fixture_dir / "table_data" / "timestamp_table2.csv"
    table = loadTable(str(input_file), "target", "time", ["input1"])
    assert table.itable.get_labels() == ["input0"]
    assert list(table.itable) == [(F,), (T,)]
    assert list(table.otable) == pytest.approx([6.0, 7.0])
    assert table.target_pos == 2

    # timestamp feature after target
    input_file = fixture_dir / "table_data" / "timestamp_table3.csv"
    table = loadTable(str(input_file), "target", "time")
    assert table.itable.get_labels() == ["input1"]
    assert list(table.itable) == [(T,), (F,)]
    assert list(table.otable) == pytest.approx([6.0, 7.0])
    assert table.target_pos == 1


# ---------------------------------------------------------------------------
# Fix round 1 (review): OTable.__eq__ label comparison, OTable.__init__
# type inference, quote-dialect CSVs, empty trailing target fields.
# Not cxxtest ports -- regression tests for findings from the task-5 review.
# ---------------------------------------------------------------------------


def test_otable_init_infers_type_from_first_value():
    """C++: table.cc:312 OTable(const super&, const string&) sets
    `type = get_type_node(get_type_tree((*this)[0]))` -- a prior version
    of this port left `type` at `unknown_type` for this overload, which
    `loadTable` never hits (it always calls `set_type` explicitly) but
    every hand-built `OTable(values, label)` in this test file does.
    """
    ot = OTable([10.1, 6.5], "happiness")
    assert ot.get_type() == type_node.contin_type

    ot_bool = OTable([T, F], "y")
    assert ot_bool.get_type() == type_node.boolean_type

    ot_enum = OTable(["sad", "bored"], "mood")
    assert ot_enum.get_type() == type_node.enum_type

    ot_empty = OTable([], "y")
    assert ot_empty.get_type() == type_node.unknown_type


def test_otable_eq_compares_the_label():
    """C++: table.cc:396 `return rhs.get_label() == label;` -- the last
    line of `OTable::operator==`, easy to miss since it comes after the
    element-comparison loop. Without it, two OTables with identical
    values but different labels compare equal.
    """
    same_values_different_label = OTable([T, F], "a") == OTable([T, F], "b")
    assert not same_values_different_label
    assert OTable([T, F], "a") == OTable([T, F], "a")


def test_loadTable_handles_quote_all_dialect(tmp_path):
    """The default `QUOTE_ALL` CSV dialect (pandas/R/Excel's default
    export format) wraps every field in double quotes:
    `"1","0","1"`. A prior version of this port silently mis-parsed this
    as `ill_formed_type` string tokens (`'"1"'`) instead of recognising
    `boolean_type` `true`/`false` -- a loadable dataset silently turned
    into garbage that looked like data, with no error raised. Verified
    against the C++ probe directly -- see task-5-report.md's fix-round-1
    section.
    """
    path = tmp_path / "quoted.csv"
    path.write_text('out,in1,in2\n"1","0","1"\n"0","1","0"\n')
    table = loadTable(str(path))
    assert table.itable.get_labels() == ["in1", "in2"]
    assert [t.name for t in table.itable.get_types()] == ["boolean_type", "boolean_type"]
    assert table.otable.get_type() == type_node.boolean_type
    assert list(table.itable) == [(F, T), (T, F)]
    assert list(table.otable) == [T, F]


def test_loadTable_preserves_row_count_when_a_trailing_target_field_is_empty(tmp_path):
    """C++ pre-sizes `otable` to `len(data_lines)` and only conditionally
    overwrites a slot when the target token is non-empty
    (table_io.cc:1105-1106, :1133-1136) -- a prior version of this port
    appended conditionally instead, silently dropping that row from
    `otable` while `itable` kept it, desynchronising the two by one and
    surfacing far downstream as an opaque `zip() argument 2 is shorter`
    from `Table.compressed()`. The blank field must be the LAST data row
    of a file with more than 20 data rows, so it falls outside
    `inferTableAttributes`'s 20-line type-inference sample and the target
    column is still (correctly) inferred as `boolean_type` despite this
    one row's blank token. Verified against the C++ probe directly, byte
    for byte, including the odd-but-faithful consequence that the
    unfilled slot is a raw `float` `0.0` (`vertex()`'s default variant
    alternative), not `builtin.logical_false` -- see
    task-5-report.md's fix-round-1 section.
    """
    lines = ["in1,target"]
    for i in range(25):
        v = i % 2
        lines.append(f"{v},{v}")
    lines.append("1,")  # 26th data row: target field is empty
    path = tmp_path / "blank_target.csv"
    path.write_text("\n".join(lines) + "\n")

    table = loadTable(str(path), "target")
    assert len(table.itable) == 26
    assert len(table.otable) == 26
    assert table.otable[-1] == 0.0
    assert isinstance(table.otable[-1], float)
    assert table.itable[-1] == (T,)

    ctable = table.compressed("")
    assert ctable.uncompressed_size() == pytest.approx(26.0)
    assert ctable[(T,)] == Counter({T: 12.0, 0.0: 1.0})
    assert ctable[(F,)] == Counter({F: 13.0})


# ---------------------------------------------------------------------------
# CARRY-FORWARD 1.1: Table.__init__ must copy, not alias, its arguments.
# Not a cxxtest port -- a regression test for the fix.
# ---------------------------------------------------------------------------


def test_table_init_copies_itable_and_otable_not_aliases_them():
    """C++: table.cc:484 Table(const OTable&, const ITable&) copy-
    constructs `itable`/`otable` from `const&` parameters -- table.h:918's
    fields are plain C++ value members, so mutating the ARGUMENT objects
    after construction can never be observed through the `Table`. A prior
    version of this port aliased both (`self.itable = itable`, `self.otable
    = otable`); this is the regression test for the fix -- see
    `Table.__init__`'s docstring and docs/CARRY-FORWARD.md section 1.1.
    """
    it = ITable(types=[type_node.boolean_type])
    it.push_back((T,))
    ot = OTable([T])
    table = Table(ot, it)

    it.push_back((F,))
    ot.push_back(F)

    assert list(table.itable) == [(T,)]
    assert list(table.otable) == [T]
    assert table.itable is not it
    assert table.otable is not ot
