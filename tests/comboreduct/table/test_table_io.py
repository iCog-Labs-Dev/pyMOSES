"""C++: moses/comboreduct/table/table_io.cc

Two kinds of tests, beyond tableUTest.cxxtest's own transliteration
(test_table.py):

1. Unit tests of the parsing/type-inference primitives (`infer_type_from_
   token(2)`, `token_to_*`, `is_header`, `get_indices`, comment/tokenizing
   helpers) and of the column-ordering/target-resolution questions
   task-5-brief.md asks this task to settle -- see table_io.py's module
   docstring for the definitive write-up.
2. A differential oracle: every case in tests/fixtures/table.json (captured
   by tools/capture_table.py from tools/cpp_probe/table_probe.cc, which
   calls loadTable() then .compressed("") against classic MOSES) compared
   against this port's own `loadTable`/`Table.compressed` over the exact
   same input files (checked into tests/fixtures/table_data/).
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import get_indices, type_node
from pymetta.comboreduct.table.table_io import (
    infer_type_from_token,
    infer_type_from_token2,
    is_header,
    loadTable,
    token_to_boolean,
    token_to_contin,
    token_to_vertex,
)

# tests/fixtures/, computed directly (not via the `fixture_dir` fixture)
# because @pytest.mark.parametrize below needs the corpus at collection
# time -- same rationale as test_interpreter.py's _EVAL_BOOLEAN_CASES.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_TABLE_JSON = _FIXTURE_DIR / "table.json"
_TABLE_CASES = json.loads(_TABLE_JSON.read_text()) if _TABLE_JSON.exists() else {}

T = builtin.logical_true
F = builtin.logical_false


# ---------------------------------------------------------------------------
# infer_type_from_token / infer_type_from_token2 -- table_io.cc:198-260
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["0", "1", "T", "F", "t", "f"])
def test_infer_type_from_token_boolean_literals(token):
    assert infer_type_from_token(token) == type_node.boolean_type


@pytest.mark.parametrize("token", ["sad", "bored", "super-happy", "x", "Thing"])
def test_infer_type_from_token_alpha_leading_is_enum(token):
    assert infer_type_from_token(token) == type_node.enum_type


@pytest.mark.parametrize("token", ["0.4", "244.2", "-3.5", "2", "100", "1e10"])
def test_infer_type_from_token_numeric_is_contin(token):
    assert infer_type_from_token(token) == type_node.contin_type


@pytest.mark.parametrize("token", ["12-04-2012", "12-06-2012T14:35:02-0500", ""])
def test_infer_type_from_token_unparseable_is_ill_formed(token):
    assert infer_type_from_token(token) == type_node.ill_formed_type


def test_infer_type_from_token2_first_token_sets_the_guess():
    """C++: table_io.cc:235 infer_type_from_token2 -- unknown_type -> tokt"""
    assert infer_type_from_token2(type_node.unknown_type, "1") == type_node.boolean_type


def test_infer_type_from_token2_boolean_then_contin_upgrades_to_contin():
    """0/1 seen first, then a real decimal -- classic's "we thought its a
    boolean 0,1 it might be a contin" rule (table_io.cc:252-254).
    """
    guess = infer_type_from_token2(type_node.unknown_type, "1")
    assert guess == type_node.boolean_type
    guess = infer_type_from_token2(guess, "10.1")
    assert guess == type_node.contin_type


def test_infer_type_from_token2_contin_then_boolean_stays_contin():
    """Symmetric: once contin, a later 0/1 does not downgrade it back
    (table_io.cc:249-251).
    """
    guess = infer_type_from_token2(type_node.unknown_type, "10.1")
    assert guess == type_node.contin_type
    guess = infer_type_from_token2(guess, "1")
    assert guess == type_node.contin_type


def test_infer_type_from_token2_inconsistent_types_fall_back_to_enum():
    """A boolean followed by something alpha-led is neither a consistent
    boolean nor a consistent contin column -- table_io.cc:256-259's
    "there's some sort of unexpected inconsistency... presume enum_type".
    """
    guess = infer_type_from_token2(type_node.unknown_type, "1")
    assert guess == type_node.boolean_type
    guess = infer_type_from_token2(guess, "sad")
    assert guess == type_node.enum_type


# ---------------------------------------------------------------------------
# token_to_* -- table_io.cc:262-332
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token, expected", [("0", F), ("F", F), ("f", F), ("1", T), ("T", T), ("t", T)]
)
def test_token_to_boolean(token, expected):
    assert token_to_boolean(token) == expected


def test_token_to_boolean_rejects_non_boolean_token():
    with pytest.raises(ValueError, match="boolean"):
        token_to_boolean("sad")


def test_token_to_contin():
    assert token_to_contin("244.2") == pytest.approx(244.2)


def test_token_to_vertex_enum_returns_plain_str():
    """See vertex.py's module docstring: enum_t is narrowed to plain str."""
    assert token_to_vertex(type_node.enum_type, "sad") == "sad"


def test_token_to_vertex_ill_formed_preserves_raw_token():
    """The "ugly hack" (table_io.cc:298-301): a column that fails every
    other type guess (typically dates) is preserved as its raw string.
    """
    assert token_to_vertex(type_node.ill_formed_type, "12-04-2012") == "12-04-2012"


def test_token_to_vertex_unknown_type_raises():
    with pytest.raises(ValueError):
        token_to_vertex(type_node.unknown_type, "x")


# ---------------------------------------------------------------------------
# is_header -- table_io.cc:678
# ---------------------------------------------------------------------------


def test_is_header_true_when_a_column_types_as_enum_but_isnt():
    """A row whose token types as enum_type in a column whose accumulated
    type is NOT enum_type must be a header (labels are never consistent
    with a numeric/boolean column's real data).
    """
    assert is_header(["happiness", "price"], [type_node.contin_type, type_node.contin_type])


def test_is_header_false_when_types_are_consistent():
    assert not is_header(["1", "0"], [type_node.boolean_type, type_node.boolean_type])


# ---------------------------------------------------------------------------
# get_indices -- table.cc:563
# ---------------------------------------------------------------------------


def test_get_indices_returns_header_order_regardless_of_argument_order():
    """table.h:64-68's own doc comment, ported verbatim in table.py: "The
    labels can be sequenced in any order, it will always return the order
    consistent with the header."
    """
    header = ["a", "b", "c", "d"]
    assert get_indices(["c", "a"], header) == [0, 2]
    assert get_indices(["a", "c"], header) == [0, 2]


def test_get_indices_ignores_names_not_in_header():
    assert get_indices(["z", "a"], ["a", "b"]) == [0]


# ---------------------------------------------------------------------------
# Comment handling -- table_io.cc:84 is_comment / :104 get_data_line
# ---------------------------------------------------------------------------


def test_loadTable_skips_comment_and_blank_lines(tmp_path):
    path = tmp_path / "commented.csv"
    path.write_text(
        "# This is a test dataset\n"
        "; This is a comment\n"
        "! This is another comment\n"
        "!\n"
        "#\n"
        "1,1,1\n"
        "1,1,1\n"
        "0,1,0\n"
    )
    table = loadTable(str(path))
    assert len(table.itable) == 3
    assert list(table.otable) == [T, T, F]


# ---------------------------------------------------------------------------
# Column ordering with ignored columns and a non-last target -- the
# question this task exists to settle (see table_io.py's module docstring
# for the full write-up).
# ---------------------------------------------------------------------------


def test_ignored_and_target_columns_preserve_original_relative_order(tmp_path):
    """5 columns, target in the middle, two ignored columns straddling
    it (one before, one after) -- the surviving itable columns must be
    exactly the remaining ones, left-to-right, unchanged in relative
    order.
    """
    path = tmp_path / "five_col.csv"
    path.write_text("c0,c1,target,c3,c4\n1,0,9,0,1\n0,1,8,1,0\n")
    table = loadTable(str(path), "target", "", ["c1", "c3"])
    assert table.itable.get_labels() == ["c0", "c4"]
    assert list(table.itable) == [(T, T), (F, F)]
    assert list(table.otable) == pytest.approx([9.0, 8.0])


def test_target_by_name_dense_path_rejects_numeric_position(tmp_path):
    """DENSE loading resolves target_feature ONLY by exact header-name
    match -- a numeric string is looked up as a literal column name, not
    a position (table_io.py's module docstring).
    """
    path = tmp_path / "named.csv"
    path.write_text("a,b,c\n1,0,1\n0,1,0\n")
    with pytest.raises(ValueError, match="not found"):
        loadTable(str(path), "2")


def test_target_by_position_is_a_sparse_only_feature(fixture_dir):
    """The SAME numeric string, against the sparse-format fixture,
    resolves POSITIONALLY (ITable.get_column_offset's 1-based-position
    special case for a digit-leading name) -- see table_io.py's module
    docstring. Column "2" (1-based) of sparse.data's fixed columns is
    "target2".
    """
    input_file = fixture_dir / "table_data" / "sparse.data"
    table = loadTable(str(input_file), "2", "", ["a", "b", "c"])
    assert table.otable.get_label() == "target2"


def test_no_header_forbids_a_named_target(tmp_path):
    """istreamDenseTable's own assertion (table_io.cc:1172-1179): without
    a header, target_feature/ignore_features/timestamp_feature must all
    be empty, since there are no names to resolve them against.
    """
    path = tmp_path / "headerless.csv"
    path.write_text("1,1,1\n0,1,0\n")
    with pytest.raises(ValueError):
        loadTable(str(path), "y")


# ---------------------------------------------------------------------------
# CTable weighted counts are floats, not ints -- table.h:494 count_t
# ---------------------------------------------------------------------------


def test_ctable_counts_are_always_float(fixture_dir):
    input_file = fixture_dir / "table_data" / "weighted.csv"
    table = loadTable(str(input_file), "r", "", [])
    ctable = table.compressed("wgt")
    assert len(ctable) > 0
    for counter in ctable.values():
        for count in counter.values():
            assert isinstance(count, float)
            assert not isinstance(count, bool)


# ---------------------------------------------------------------------------
# Differential oracle against tools/cpp_probe/table_probe.cc
# ---------------------------------------------------------------------------


def test_golden_fixture_has_expected_shape():
    assert len(_TABLE_CASES) > 0
    for name, case in _TABLE_CASES.items():
        assert set(case) >= {
            "itable_labels",
            "itable_types",
            "otable_label",
            "otable_type",
            "arity",
            "target_pos",
            "itable",
            "otable",
            "ctable_labels",
            "ctable_uncompressed_size",
            "ctable",
            "file",
        }, name


_TYPE_NAME_TO_NODE = {t.name: t for t in type_node}


def _vertex_from_json(v):
    if isinstance(v, bool):
        return T if v else F
    return v  # float (contin) or str (enum/ill_formed) round-trip as-is


def _values_match(actual, expected) -> bool:
    """Elementwise compare, using pytest.approx for floats (JSON round-
    trips a double through `%.17g`/`json.loads`, which is bit-exact, but
    approx is used defensively) and plain equality otherwise (`builtin`
    enum members, `str`).
    """
    if isinstance(expected, float):
        return actual == pytest.approx(expected)
    return actual == expected


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_TABLE_CASES), ids=sorted(_TABLE_CASES))
def test_loadTable_matches_cpp_probe(case_name):
    case = _TABLE_CASES[case_name]
    path = _FIXTURE_DIR / "table_data" / case["file"]
    table = loadTable(
        str(path), case["target_feature"], case["timestamp_feature"], case["ignore_features"]
    )

    assert table.itable.get_labels() == case["itable_labels"]
    assert [t.name for t in table.itable.get_types()] == case["itable_types"]
    assert table.otable.get_label() == case["otable_label"]
    assert table.otable.get_type().name == case["otable_type"]
    assert table.get_arity() == case["arity"]
    assert table.target_pos == case["target_pos"]

    expected_itable = [tuple(_vertex_from_json(v) for v in row) for row in case["itable"]]
    actual_itable = list(table.itable)
    assert len(actual_itable) == len(expected_itable)
    for actual_row, expected_row in zip(actual_itable, expected_itable, strict=True):
        assert len(actual_row) == len(expected_row)
        for actual_v, expected_v in zip(actual_row, expected_row, strict=True):
            assert _values_match(actual_v, expected_v)

    expected_otable = [_vertex_from_json(v) for v in case["otable"]]
    actual_otable = list(table.otable)
    assert len(actual_otable) == len(expected_otable)
    for actual_v, expected_v in zip(actual_otable, expected_otable, strict=True):
        assert _values_match(actual_v, expected_v)


@pytest.mark.oracle
@pytest.mark.parametrize("case_name", sorted(_TABLE_CASES), ids=sorted(_TABLE_CASES))
def test_ctable_matches_cpp_probe(case_name):
    case = _TABLE_CASES[case_name]
    path = _FIXTURE_DIR / "table_data" / case["file"]
    table = loadTable(
        str(path), case["target_feature"], case["timestamp_feature"], case["ignore_features"]
    )
    ctable = table.compressed("")

    assert ctable.get_labels() == case["ctable_labels"]
    assert ctable.uncompressed_size() == pytest.approx(case["ctable_uncompressed_size"])
    assert len(ctable) == len(case["ctable"])

    actual_by_key = {key: dict(counter) for key, counter in ctable.items()}

    for row in case["ctable"]:
        key = tuple(_vertex_from_json(v) for v in row["inputs"])
        expected_counts = {_vertex_from_json(v): c for v, c in row["counts"]}
        assert key in actual_by_key, f"{case_name}: missing row {key!r}"
        actual_counts = actual_by_key[key]
        assert actual_counts.keys() == expected_counts.keys(), case_name
        for v, expected_count in expected_counts.items():
            assert actual_counts[v] == pytest.approx(expected_count)
