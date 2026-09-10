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
from datetime import date
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
        "happiness,price,temperature,truth\nsad,1,1,0\nbored,0.44,244.2,T\nsuper-happy,9.5,4.2,F\n",
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
        "happiness,price,temperature,truth\nsad,1,1,0\nbored,0.44,244.2,T\nsuper-happy,9.5,4.2,F\n",
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

    Now the real thing: classic builds a `TTable` of the three parsed
    dates and asserts `table.ttable == dates`, and so does this.
    """
    input_file = fixture_dir / "table_data" / "timestamp_table.csv"
    table = loadTable(str(input_file), "length", "time")
    assert table.itable.get_labels() == ["width"]
    assert table.otable.get_label() == "length"
    assert list(table.otable) == pytest.approx([12.0, 10.0, 14.0])
    assert list(table.ttable) == [date(2012, 1, 11), date(2012, 1, 13), date(2012, 2, 18)]
    assert table.ttable.get_label() == "time"
    assert table.timestamp_pos == 0


def test_ctable_has_the_timestamp_surface(tmp_path):
    """The inverse of what this test used to assert.

    It used to pin the ABSENCE of `get_timestamps`/`remove_rows_at_time`/
    `remove_rows_at_times`/`Table.ttable`, because a vacuous
    `get_timestamps` returning `set()` would have been numerically right
    for every table the port could then build while silently un-gating
    `ss_bscore.__init__` (which tests `hasattr(ctable, "get_timestamps")`).
    All four now exist and are backed by real data, so the property worth
    pinning is the one that made the absence load-bearing: `get_timestamps`
    must be NON-vacuous -- it must return the actual dates for a
    timestamped table, not just `set()` for an untimestamped one.
    """
    assert hasattr(CTable, "get_timestamps")
    assert hasattr(CTable, "remove_rows_at_time")
    assert hasattr(CTable, "remove_rows_at_times")
    assert hasattr(Table(), "ttable")

    # Two rows agreeing on inputs AND output but differing in date. Classic
    # keys them as two distinct `TimedValue`s, each counted 1.0; so does
    # this port now. Before the timestamp slice they collapsed into one
    # bare-vertex key counted 2.0.
    path = tmp_path / "dupe_rows_distinct_dates.csv"
    path.write_text("time,width,length\n2012-Jan-11,6,12\n2012-Feb-18,6,12\n2012-Mar-23,3,10\n")
    table = loadTable(str(path), "length", "time")
    assert table.itable.get_labels() == ["width"]  # timestamp column excluded

    ctable = table.compressed()

    # The UNTIMED view -- what every vertex-keyed consumer sees, and what
    # `TimedCounter::get(const vertex&)` (table.cc:608) is defined to
    # return: "the sum of that counter across all timestamps".
    assert {row: dict(counter) for row, counter in ctable.items()} == {
        (6.0,): {12.0: 2.0},
        (3.0,): {10.0: 1.0},
    }

    # ... and the observation list underneath it, which is where the two
    # 2012 dates survive separately.
    assert {(tv.value, tv.timestamp): c for tv, c in ctable[(6.0,)].timed_items()} == {
        (12.0, date(2012, 1, 11)): 1.0,
        (12.0, date(2012, 2, 18)): 1.0,
    }
    assert ctable[(6.0,)].timed_len() == 2
    assert len(ctable[(6.0,)]) == 1  # one distinct VALUE, two observations

    assert ctable.get_timestamps() == {
        date(2012, 1, 11),
        date(2012, 2, 18),
        date(2012, 3, 23),
    }
    assert ctable.uncompressed_size() == 3.0


def test_get_timestamps_is_empty_without_a_timestamp_feature(fixture_dir):
    """Classic's `get_timestamps` filters out the default not-a-date value
    (table.cc:740), and `Table::compressed` gives every observation of an
    untimestamped table exactly that value (table.cc:513) -- so the empty
    set here is C++'s own answer, reached the same way, not a stub.
    """
    table = loadTable(str(fixture_dir / "table_data" / "trainingBoolean.dat"), "y")
    ctable = table.compressed()
    assert len(ctable) > 0
    assert ctable.get_timestamps() == set()
    # Non-vacuously so: there ARE observations, they just all carry the
    # sentinel.
    assert sum(c.timed_len() for c in ctable.values()) > 0
    assert all(tv.timestamp is None for c in ctable.values() for tv in c.timed_keys())


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


# ---------------------------------------------------------------------------
# ITable's random-sampling constructor (table.cc:93) and OTable's
# combo-tree constructor (table.cc:319). Not in tableUTest.cxxtest -- classic
# has no unit test for either; these are oracle tests against a compiled
# probe.
#
# The probe links the REAL `ITable` and the REAL cogutil `MT19937RandGen`:
#
#   g++ -O2 -std=c++17 -I $CLASSIC -I $PREFIX/include -o itable_rand_probe \
#       itable_rand_probe.cc -L $PREFIX/lib -L $PREFIX/lib/opencog \
#       -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$PREFIX/lib/opencog \
#       -lmoses -lcomboreduct -lcogutil
#
# It seeds `randGen()`, builds one `ITable` and prints every cell at
# `%.17g`; the `sig <seed> <nsamples> <min> <max> <spec>` mode builds a
# `lambda(<spec>) -> contin` signature from a string of `b`/`c`/`u`
# characters. Every literal below is that output, verbatim. 16
# configurations were diffed in total (this file keeps the six that
# discriminate); all 16 were identical element for element.
#
# Draw ORDER and COUNT are the point: per row, in column order, a boolean
# column consumes one `randGen().randint(2)` and a contin column one
# `randGen().randdouble()`, while an `unknown` column consumes NOTHING.
# ---------------------------------------------------------------------------

_T = builtin.logical_true
_F = builtin.logical_false


def _sig(spec, otype=type_node.contin_type):
    from pymetta.comboreduct.type_checker.type_signature import TypeSignature

    m = {
        "b": type_node.boolean_type,
        "c": type_node.contin_type,
        "u": type_node.unknown_type,
        "e": type_node.enum_type,
        "x": type_node.list_type,
    }
    return TypeSignature([m[ch] for ch in spec], otype)


def _seeded(seed):
    from pymetta.util.rng import randGen

    g = randGen()
    g.seed(seed)
    return g


def test_itable_random_ctor_matches_classic_for_the_sr_signature():
    """`itable_rand_probe sr 1 8` -- exactly what `polynomial_problem::run`
    builds (`gen_signature(id::contin_type, 1)` and the constructor's own
    `min_contin=-1.0`/`max_contin=1.0` defaults, demo-problems.cc:261-263),
    at classic's default seed.
    """
    _seeded(1)
    it = ITable(_sig("c"), 8)
    assert [row[0] for row in it] == [
        -0.16595600293156598,
        0.99436961673069035,
        0.44064897891149135,
        0.86511472283515967,
        -0.99977123783896005,
        -0.74375110439577863,
        -0.39533486436012544,
        0.99808103125497727,
    ]


def test_itable_random_ctor_interleaves_boolean_and_contin_draws():
    """`itable_rand_probe sig 1 6 -1 1 bcbc`. The discriminating property
    is the INTERLEAVING: within a row the four columns draw
    randint/randdouble/randint/randdouble in that order, so the contin
    values are NOT the first six of the pure-contin stream above. (They
    are its 2nd, 4th, 6th, 8th, ... -- visible by comparing row 0 here
    with rows 1 and 3 of the previous test.)
    """
    _seeded(1)
    it = ITable(_sig("bcbc"), 6)
    assert list(it[0]) == [_F, 0.99436961673069035, _F, 0.86511472283515967]
    assert list(it[1]) == [_T, -0.74375110439577863, _T, 0.99808103125497727]
    assert list(it[5]) == [_T, -0.37345297387183019, _T, 0.049096325656654338]


def test_itable_random_ctor_enumerates_booleans_when_nsamples_is_pow2_barity():
    """`itable_rand_probe sig 3 8 -1 1 bbb`: `comp_tt` (table.cc:100) is
    true because `8 == pow2(3)`, so the three boolean columns are the
    binary counter `i & (1 << bidx++)` -- LOW bit in column 0 -- and NO
    RNG draw happens at all. Seed 3 is used and is irrelevant: the same
    table comes out under any seed, which is the whole point.

    Mutation-proven: replacing `1 << bidx` with `1 << (barity - 1 - bidx)`
    (the other plausible bit order) makes this test fail (see this task's
    report).
    """
    _seeded(3)
    it = ITable(_sig("bbb"), 8)
    assert [list(r) for r in it] == [
        [_F, _F, _F],
        [_T, _F, _F],
        [_F, _T, _F],
        [_T, _T, _F],
        [_F, _F, _T],
        [_T, _F, _T],
        [_F, _T, _T],
        [_T, _T, _T],
    ]
    # ...and it really did not touch the RNG.
    from pymetta.util.rng import randGen

    g = randGen()
    fresh = ITable(_sig("c"), 1)[0][0]
    g.seed(3)
    assert fresh == (2.0 * g.randdouble() - 1.0)


def test_itable_random_ctor_honours_a_reversed_contin_range():
    """`itable_rand_probe sig 8 4 10 0 c`: `min_contin=10`,
    `max_contin=0`, which is the argument order `combo_problem::run`
    actually produces (demo-problems.cc:429 passes max as min). The
    formula `(max - min) * r + min` then yields `10 - 10r`, DECREASING in
    the draw. Values here are classic's, so they pin the direction.
    """
    _seeded(8)
    it = ITable(_sig("c"), 4, 10.0, 0.0)
    assert [row[0] for row in it] == [
        1.2657059638913974,
        9.8888556193301582,
        0.31459334779404813,
        7.6056042843511342,
    ]


def test_itable_random_ctor_auto_nsamples():
    """`itable_rand_probe sig 11 -1 -1 1 bbc`: `nsamples < 0` ->
    `std::max(pow2(barity), sample_count(carity))` = `max(4, 26)` = 26.
    26, not 26.265233750364456: `sample_count` returns `unsigned`, so
    `20.0 * log(1 + e)` truncates.

    UNREACHABLE from either demo problem (both force a positive
    `nsamples` first -- see `ITable._sample`'s own docstring), so this is
    the only coverage it has.
    """
    _seeded(11)
    it = ITable(_sig("bbc"), -1)
    assert len(it) == 26
    assert list(it[0]) == [_F, _T, -0.96104951947020589]
    assert list(it[25]) == [_F, _F, -0.14569563305603705]

    # carity 0 -> sample_count is 1, so pow2(barity) wins.
    _seeded(13)
    assert len(ITable(_sig("bbbbb"), -1)) == 32


def test_itable_random_ctor_unknown_column_is_a_contin_zero():
    """`itable_rand_probe sig 4 3 -1 1 uu`: `vs.push_back(vertex())` for
    an `unknown_type` column. `vertex` is `boost::variant<contin_t,
    enum_t, builtin, ...>` (vertex.h:128), so a default-constructed one
    holds its FIRST alternative -- a CONTIN 0.0, which the probe reports
    as `c:0`, NOT `id::null_vertex`.

    Mutation-proven: `builtin.null_vertex` (the natural reading of
    "push default vertex") makes this test fail. It was the port's first
    guess, and the probe caught it (see this task's report).
    """
    _seeded(4)
    it = ITable(_sig("uu"), 3)
    assert [list(r) for r in it] == [[0.0, 0.0]] * 3
    assert all(isinstance(v, float) and not isinstance(v, builtin) for v in it[0])

    # ...and it consumed no draws either.
    from pymetta.util.rng import randGen

    g = randGen()
    g.seed(4)
    before = g.randdouble()
    g.seed(4)
    ITable(_sig("uu"), 3)
    assert g.randdouble() == before


def test_itable_random_ctor_leaves_types_and_labels_at_their_lazy_defaults():
    """The C++ constructor sets neither `types` nor `labels` (table.cc:
    93-127), so `get_types()`/`get_labels()` fall through to their lazy
    defaults -- all `unknown_type`, and `i1..iN`.
    """
    _seeded(1)
    it = ITable(_sig("bc"), 3)
    assert it.get_types() == [type_node.unknown_type, type_node.unknown_type]
    assert it.get_labels() == ["i1", "i2"]


def test_itable_random_ctor_rejects_an_enum_column():
    """`enum_t::get_random_enum()` (table.cc:115) has no port -- the enum
    domain is excluded package-wide.
    """
    with pytest.raises(NotImplementedError, match="enum_type"):
        ITable(_sig("e"), 1)


def test_itable_random_ctor_rejects_any_other_column_type():
    """classic's own `OC_ASSERT(false, "Not implemented yet")`
    (table.cc:118) for every `type_node` outside boolean/contin/enum/
    unknown.
    """
    with pytest.raises(NotImplementedError, match="Not implemented yet"):
        ITable(_sig("x"), 1)


def test_otable_from_combo_tree_evaluates_every_row():
    """C++: moses/comboreduct/table/table.cc:319 OTable

    Same configuration `combo_problem::run` produces for
    `-y'+(6 *(6 $1))'` at classic's own defaults: `max_rand_input = 2 *
    largest_const = 12`, `min_rand_input = -12`, and the swapped
    constructor arguments. Both the inputs and the outputs are the
    probe's `cp 1 "+(6 *(6 $1))" 5 -12 12` output.
    """
    from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree

    _seeded(1)
    it = ITable(_sig("c"), 5, 12.0, -12.0)
    ot = OTable(str_to_combo_tree("+(6 *(6 $1))"), it)
    assert [row[0] for row in it] == [
        1.9914720351787913,
        -11.932435400768284,
        -5.2877877469378944,
        -10.381376674021915,
        11.997254854067521,
    ]
    assert list(ot) == [
        17.948832211072748,
        -65.594612404609705,
        -25.726726481627367,
        -56.288260044131491,
        77.983529124405123,
    ]
    # `type = get_type_node(get_type_tree((*this)[0]))` -- table.cc:346.
    assert ot.get_type() == type_node.contin_type
    assert ot.get_label() == "output"


def test_otable_from_combo_tree_rejects_a_non_itable_second_argument():
    from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree

    with pytest.raises(TypeError, match="must be an ITable"):
        OTable(str_to_combo_tree("+(6 $1)"), "not an itable")


# ---------------------------------------------------------------------------
# timed_items_of -- the (TimedValue, count) view precision_bscore reads
# ---------------------------------------------------------------------------


def test_timed_items_of_a_real_cell_yields_every_observation_with_its_date():
    """`timed_items_of` stands in for C++'s
    `for (const CTable::counter_t::value_type& tcv : orow)`
    (precision_bscore.cc:243). Unlike `timed_keys_of` it carries the COUNT
    too, which both of `do_score`'s timestamp branches need.

    Iterating the cell itself would give the UNTIMED view (this port's
    documented divergence, `TimedCounter`'s class docstring), collapsing the
    two dated observations of the same vertex into one entry -- so a
    faithful reading has to go through this function.
    """
    from pymetta.comboreduct.table.table import TimedCounter, timed_items_of

    cell = TimedCounter()
    cell.add(T, date(2012, 1, 3), 1.0)
    cell.add(T, date(2012, 1, 11), 2.0)
    cell.add(F, date(2012, 1, 3), 4.0)

    items = list(timed_items_of(cell))
    assert [(tv.value, tv.timestamp, c) for tv, c in items] == [
        (T, date(2012, 1, 3), 1.0),
        (T, date(2012, 1, 11), 2.0),
        (F, date(2012, 1, 3), 4.0),
    ]
    # The untimed mapping view really does lose the split, which is why
    # this function exists rather than plain iteration.
    assert dict(cell) == {T: 3.0, F: 4.0}


def test_timed_items_of_tolerates_a_plain_counter_cell():
    """A hand-built test table can hold a plain `collections.Counter`.
    Such a cell is semantically a `TimedCounter` all of whose observations
    carry the not-a-date sentinel -- what `Table::compressed` builds for an
    untimestamped table (table.cc:513) -- so it is presented as one rather
    than raising `AttributeError`.
    """
    from pymetta.comboreduct.table.table import timed_items_of

    items = list(timed_items_of(Counter({T: 2.0, F: 5.0})))
    assert [(tv.value, tv.timestamp, c) for tv, c in items] == [
        (T, None, 2.0),
        (F, None, 5.0),
    ]


def test_ctable_cell_copy_keeps_every_observation():
    """`TimedCounter.copy()` is C++'s `std::map` copy constructor -- the one
    `CTable`'s own copy constructor invokes per cell, and the one
    `scoring_base._copy_ctable` now uses. Rebuilding the cell from the
    untimed view instead (`Counter(cell)`, which is what it used to do)
    silently drops every timestamp.
    """
    from pymetta.comboreduct.table.table import TimedCounter

    cell = TimedCounter()
    cell.add(T, date(2012, 1, 3), 1.0)
    cell.add(T, date(2012, 1, 11), 2.0)

    clone = cell.copy()
    assert clone is not cell
    assert clone.timed_len() == 2
    assert {tv.timestamp for tv in clone.timed_keys()} == {date(2012, 1, 3), date(2012, 1, 11)}
    # ... whereas the flattening rebuild keeps only one, sentinel-dated
    # observation, which is the regression this pins.
    assert Counter(cell) == {T: 3.0}
