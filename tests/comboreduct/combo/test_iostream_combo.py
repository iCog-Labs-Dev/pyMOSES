import json
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree

# tests/fixtures/, computed directly (not via the `fixture_dir` fixture)
# because @pytest.mark.parametrize below needs the corpus at collection
# time, before any fixture would run.
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"
_ROUNDTRIP_CASES = json.loads((_FIXTURE_DIR / "combo_roundtrip.json").read_text())


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "combo_roundtrip.json").read_text())


def test_golden_fixture_has_expected_shape(golden):
    # 29 trees from $CLASSIC/testTrees.sh's `trees` array + 1 from
    # iostream_comboUTest.cxxtest's constructor -- see task-3-brief.md.
    assert len(golden) == 30
    for entry in golden:
        assert set(entry) == {"input", "expected"}
        assert isinstance(entry["input"], str)
        assert isinstance(entry["expected"], str)


@pytest.mark.oracle
@pytest.mark.parametrize("case", _ROUNDTRIP_CASES, ids=lambda case: case["input"])
def test_round_trip_matches_combo_fmt_converter(case):
    t = str_to_combo_tree(case["input"])
    assert combo_tree_to_str(t) == case["expected"]


# ---------------------------------------------------------------------------
# Unit tests beyond the oracle corpus -- builtin string aliases, contin
# formatting, and parse-error paths the 30-tree corpus does not exercise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("logical_and", "and"),
        ("logical_or", "or"),
        ("logical_not", "not"),
        ("logical_true", "true"),
        ("logical_false", "false"),
        ("plus", "+"),
        ("times", "*"),
        ("div", "/"),
        ("lambda", "->"),
        ("contin_boolean_if", "contin_if"),
    ],
)
def test_builtin_string_aliases_normalize_to_canonical_output(alias, canonical):
    # C++: iostream_combo.cc:36-124 builtin_str_to_vertex accepts several
    # input spellings per builtin; ostream_builtin (iostream_combo.cc:
    # 269-325) always emits the one canonical spelling.
    assert combo_tree_to_str(str_to_combo_tree(alias)).strip() == canonical


def test_equ_prints_but_cannot_be_parsed():
    # A real asymmetry in classic MOSES itself: builtin_str_to_vertex has
    # no "equ" branch (iostream_combo.cc:36-124), so "equ" cannot come back
    # from combo text even though ostream_builtin prints it
    # (iostream_combo.cc:321-322).
    assert combo_tree_to_str(str_to_combo_tree("and($1 $2)")).strip() == "and($1 $2)"
    with pytest.raises(ValueError):
        str_to_combo_tree("equ")


@pytest.mark.parametrize(
    ("contin_str", "expected"),
    [
        ("3.14", "3.14"),
        ("3.0", "3"),
        ("3", "3"),
        ("-2.5", "-2.5"),
        ("1e10", "1e+10"),
        ("0.00001", "1e-05"),
        ("123.456789", "123.457"),
    ],
)
def test_contin_formatting_matches_cpp_default_double_stream(contin_str, expected):
    # C++'s `out << get<contin_t>(v)` (iostream_combo.cc:464) is the
    # default iostream double format: defaultfloat, precision 6 -- "%.6g".
    # Verified against combo-fmt-converter for each of these (see
    # task-3-report.md).
    assert combo_tree_to_str(str_to_combo_tree(contin_str)).strip() == expected


def test_empty_parens_collapse_to_a_zero_arity_leaf():
    t = str_to_combo_tree("and()")
    assert t.number_of_children(t.begin()) == 0
    assert combo_tree_to_str(t).strip() == "and"


def test_whitespace_before_open_paren_is_tolerated():
    assert combo_tree_to_str(str_to_combo_tree("and  ($1 $2)")).strip() == "and($1 $2)"
    assert combo_tree_to_str(str_to_combo_tree("and(  $1 $2)")).strip() == "and($1 $2)"
    assert combo_tree_to_str(str_to_combo_tree("and($1 $2 )")).strip() == "and($1 $2)"


def test_negated_argument_is_distinct_from_a_not_node():
    # "!$3" is a single leaf vertex (argument(-3)); "not($3)" is a builtin
    # logical_not node with one argument(3) child. Structurally different
    # trees that happen to be logically equivalent -- reduct's job, not
    # this module's.
    negated_literal = str_to_combo_tree("!$3")
    assert negated_literal.begin().data == argument(-3)
    assert negated_literal.number_of_children(negated_literal.begin()) == 0

    not_node = str_to_combo_tree("not($3)")
    assert not_node.begin().data == builtin.logical_not
    assert not_node.number_of_children(not_node.begin()) == 1
    assert next(not_node.children(not_node.begin())).data == argument(3)


def test_empty_tree_prints_as_empty_string():
    assert combo_tree_to_str(combo_tree()) == ""


def test_unparseable_token_raises_value_error():
    with pytest.raises(ValueError):
        str_to_combo_tree("this_is_not_a_known_token")


def test_unterminated_paren_raises_value_error():
    with pytest.raises(ValueError):
        str_to_combo_tree("and($1 $2")


def test_trailing_content_after_root_raises_value_error():
    # A combo_tree is single-rooted (pymetta/util/tree.py's module
    # docstring); classic's own strtree_to_combo_tree folds multiple
    # top-level trees under an implicit forest root, which this port does
    # not support (see iostream_combo.py's module docstring).
    with pytest.raises(ValueError):
        str_to_combo_tree("and($1 $2) or($1 $2)")


def test_argument_index_zero_is_rejected_during_parsing():
    with pytest.raises(AssertionError):
        str_to_combo_tree("$0")
