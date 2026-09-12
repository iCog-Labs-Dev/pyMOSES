"""Oracle tests for the combo type-inference engine.

Every assertion here compares `pymoses.comboreduct.type_checker.type_tree`
(and the `builtins_properties` table in
`pymoses.comboreduct.combo.descriptions`) against
`tests/fixtures/type_tree.json`, produced by
`tools/capture_type_tree.py` from `tools/cpp_probe/type_tree_probe.cc` --
classic's own `type_tree.cc` and `type_tree_def.cc`, linked out of
libcomboreduct.so, not a reimplementation. The probe's module comment
lists which C++ arm each case reaches.

WHAT THE CORPUS ACTUALLY COVERS, and how that was established. The point
of this fixture is the ILL-TYPED half: a corpus of only well-typed
programs lets a stub that always answers "well formed" pass every row,
which is this project's most-repeated defect. Every trap below was
verified by MUTATION -- the plausible wrong version was written, the suite
re-run, and the number of failing tests recorded. A mutation that kills
nothing means either no coverage or no trap, and the two demand opposite
responses.

    mutation                                                   tests killed
    ---------------------------------------------------------- ------------
    is_well_formed always returns True                                 22
    infer_type_tree skips infer_arg_type_tree (no cross-arg check)      13
    infer_vertex_type drops the `si < a` arity guard                    18
    inherit_type_tree's `*it2 == unknown` arm returns False             40
    inherit_type_tree's lambda arm drops `*sib2 != unknown_type`         6
    type_tree_arity returns `a` instead of `-a` for arg_list             8
    reduce_type_tree's arg_list walk restricted to the subtree           4
    descriptions.py stops flattening arg_list inputs                    13
    _collapse_empty_parens returns its input unchanged                  16
    convert_index returns `index` unconditionally                        2
    _size_tree_order_less compares `s1 > s2`                             0 (*)

    (*) zero kills is NO COVERAGE, not "no trap": the union-of-unions
    branch of get_intersection only ever collects ONE well-formed
    intersection in this corpus (`union(boolean contin)` against
    `union(contin action_result)` gives just `contin`), so the ordering
    never decides anything. Recorded as uncovered rather than claimed.

The two rows this port cannot reproduce are also asserted, in the
direction of the divergence rather than skipped -- see
`test_equ_token_parses_as_a_definite_object_in_classic_and_is_refused_here`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pymoses.comboreduct.combo import vertex as vertex_module
from pymoses.comboreduct.combo.descriptions import builtins_properties
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.type_checker.type_signature import (
    arg_to_idx,
    is_argument_type,
    ostream_type_node,
    type_node,
)
from pymoses.comboreduct.type_checker.type_tree import (
    abs_min_arity,
    action_result_arity,
    argument_type_list_input_type,
    boolean_arity,
    contin_arity,
    convert_index,
    default_vertex_value,
    does_contain_all_arg_up_to,
    equal_type_tree,
    explicit_arity,
    get_argument_abs_idx_from_zero_set,
    get_argument_abs_idx_set,
    get_arity,
    get_input_type_tree,
    get_intersection,
    get_output_type_tree,
    get_signature_inputs,
    get_signature_output,
    get_type_node,
    get_type_tree,
    infer_arg_type_tree,
    infer_arity,
    infer_type_tree,
    infer_vertex_type,
    inherit_type_tree,
    is_well_formed,
    istream_type_node,
    ostream_type_tree,
    reduce_type_tree,
    str_to_type_tree,
    type_tree,
    type_tree_arity,
)
from tools.check_fixture_manifests import SUPPORTED_SCHEMA_VERSIONS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "type_tree.json"
_MANIFEST_PATH = _REPO_ROOT / "tests" / "fixtures" / "type_tree.manifest.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())
_MANIFEST = json.loads(_MANIFEST_PATH.read_text())

# The only two fixture rows this port's own combo PARSER cannot produce.
# Classic reads the bare token `equ` as a `definite_object` string vertex,
# not as `id::equ` (the fixture's `vertices` row for it says
# `definite_object`, arity 0); this port's grammar recognises builtin,
# argument and contin tokens only (iostream_combo.py:332) and raises. That
# is a property of `iostream_combo.py`, not of this module -- and it is
# asserted below rather than skipped.
_UNPARSEABLE_BY_THIS_PORT = frozenset({"equ", "equ($1 $2)"})


def _vertex_of(token: str):
    return str_to_combo_tree(token).begin().data


# ---------------------------------------------------------------------------
# builtins_properties -- descriptions.cc:100
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["builtins"], ids=lambda c: c["name"])
def test_builtins_properties_matches_cpp(case):
    b = builtin(case["ord"])
    props = builtins_properties.instance()
    assert props.builtin_arity(b) == case["arity"]
    assert ostream_type_tree(get_type_tree(b)) == case["type_tree"]
    assert ostream_type_tree(get_output_type_tree(b)) == case["output_type_tree"]
    assert [ostream_type_tree(get_input_type_tree(b, i)) for i in range(4)] == case[
        "input_type_trees"
    ]


def test_builtin_table_covers_every_builtin_exactly_once():
    """descriptions.cc:105's own `OC_ASSERT` -- one entry per builtin."""
    props = builtins_properties.instance()
    for b in builtin:
        if b is builtin.builtin_count:
            continue
        assert props.builtin_arity(b) is not None
    assert len(_GOLDEN["builtins"]) == int(builtin.builtin_count)


# ---------------------------------------------------------------------------
# Per-vertex facts -- type_tree.cc:201 / :250 / :283 / :403
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    [c for c in _GOLDEN["vertices"] if c["vertex"] not in _UNPARSEABLE_BY_THIS_PORT],
    ids=lambda c: c["vertex"],
)
def test_vertex_type_facts_match_cpp(case):
    v = _vertex_of(case["vertex"])
    assert ostream_type_tree(get_type_tree(v)) == case["get_type_tree"]
    assert ostream_type_tree(get_output_type_tree(v)) == case["get_output_type_tree"]
    assert get_arity(v) == case["get_arity"]
    assert [ostream_type_tree(get_input_type_tree(v, i)) for i in range(4)] == case[
        "input_type_trees"
    ]


# ---------------------------------------------------------------------------
# Whole-program inference -- the heart of the fixture
# ---------------------------------------------------------------------------

_PROGRAMS = [c for c in _GOLDEN["programs"] if c["combo"] not in _UNPARSEABLE_BY_THIS_PORT]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_infer_type_tree_matches_cpp(case):
    tr = str_to_combo_tree(case["combo"])
    assert tree_to_stream_str(tr) == case["parsed"], "the two parsers disagree"
    assert ostream_type_tree(get_type_tree(tr)) == case["get_type_tree"]
    tt = infer_type_tree(tr)
    assert ostream_type_tree(tt) == case["infer_type_tree"]
    assert int(is_well_formed(tt)) == case["is_well_formed"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_program_arities_match_cpp(case):
    tr = str_to_combo_tree(case["combo"])
    tt = infer_type_tree(tr)
    assert type_tree_arity(tt) == case["type_tree_arity"]
    assert contin_arity(tt) == case["contin_arity"]
    assert boolean_arity(tt) == case["boolean_arity"]
    assert action_result_arity(tt) == case["action_result_arity"]
    assert infer_arity(tr) == case["infer_arity"]
    assert explicit_arity(tr) == case["explicit_arity"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_program_signature_accessors_match_cpp(case):
    tt = infer_type_tree(str_to_combo_tree(case["combo"]))
    assert ostream_type_tree(get_signature_output(tt)) == case["signature_output"]
    assert [ostream_type_tree(x) for x in get_signature_inputs(tt)] == case["signature_inputs"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_program_argument_index_facts_match_cpp(case):
    tr = str_to_combo_tree(case["combo"])
    assert sorted(get_argument_abs_idx_set(tr)) == case["arg_abs_idx"]
    assert sorted(get_argument_abs_idx_from_zero_set(tr)) == case["arg_abs_idx_from_zero"]
    assert [int(does_contain_all_arg_up_to(tr, n)) for n in range(6)] == case[
        "contains_all_arg_up_to"
    ]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_infer_arg_type_tree_matches_cpp(case):
    tr = str_to_combo_tree(case["combo"])
    arg_types: list[type_tree] = []
    infer_arg_type_tree(tr, arg_types)
    assert [ostream_type_tree(a) for a in arg_types] == case["infer_arg_types"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _PROGRAMS, ids=lambda c: c["combo"])
def test_infer_vertex_type_matches_cpp(case):
    tr = str_to_combo_tree(case["combo"])
    got = [
        {"idx": i, "type": ostream_type_tree(infer_vertex_type(tr, node))}
        for i, node in enumerate(tr)
    ]
    assert got == [{"idx": r["idx"], "type": r["type"]} for r in case["infer_vertex_type"]]


# ---------------------------------------------------------------------------
# Inheritance and intersection -- type_tree.cc:435 / :440 / :446 / :941 / :951
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["inherit"], ids=lambda c: f"{c['a']}|{c['b']}")
def test_inherit_and_intersection_match_cpp(case):
    a = str_to_type_tree(case["a"])
    b = str_to_type_tree(case["b"])
    assert int(inherit_type_tree(a, b)) == case["a_inherits_b"]
    assert int(inherit_type_tree(b, a)) == case["b_inherits_a"]
    assert int(equal_type_tree(a, b)) == case["equal"]
    assert ostream_type_tree(get_intersection(a, b)) == case["intersection"]


# ---------------------------------------------------------------------------
# reduce_type_tree -- type_tree.cc:517 / :537
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["reduce"], ids=lambda c: f"{c['tt']}|{','.join(c['arg_types'])}"
)
def test_reduce_type_tree_matches_cpp(case):
    tt = str_to_type_tree(case["tt"])
    arg_types = [str_to_type_tree(x) for x in case["arg_types"]]
    reduce_type_tree(tt, arg_types, None, "PROBE")
    assert ostream_type_tree(tt) == case["reduced"]
    assert int(is_well_formed(tt)) == case["is_well_formed"]


# ---------------------------------------------------------------------------
# The type-tree string reader -- opencog/util/tree.cc:70 + tree.h:3312
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["parse"], ids=lambda c: c["text"])
def test_str_to_type_tree_matches_cpp(case):
    if case.get("threw"):
        with pytest.raises((AssertionError, ValueError)):
            str_to_type_tree(case["text"])
        return
    tt = str_to_type_tree(case["text"])
    assert ostream_type_tree(tt) == case["printed"]
    assert tt.size() == case["size"]


# ---------------------------------------------------------------------------
# type_node stream operators -- type_tree.cc:1360 / :1409
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["type_node_io"]["ostream"], ids=lambda c: str(c["ord"]))
def test_ostream_type_node_matches_cpp(case):
    assert ostream_type_node(type_node(case["ord"])) == case["str"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["type_node_io"]["istream"], ids=lambda c: c["str"])
def test_istream_type_node_matches_cpp(case):
    if case.get("threw"):
        with pytest.raises(AssertionError):
            istream_type_node(case["str"])
        return
    assert int(istream_type_node(case["str"])) == case["ord"]


# ---------------------------------------------------------------------------
# misc -- type_tree.cc:35 / :372 / :380 / :389 / :396 and type_tree_def.cc
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["misc"]["convert_index"], ids=lambda c: f"a{c['arity']}i{c['index']}"
)
def test_convert_index_matches_cpp(case):
    assert convert_index(case["arity"], case["index"]) == case["result"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["misc"]["abs_min_arity"], ids=lambda c: str(c["arity"]))
def test_abs_min_arity_matches_cpp(case):
    assert abs_min_arity(case["arity"]) == case["result"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["misc"]["argument_type_list_input_type"],
    ids=lambda c: f"a{c['arity']}i{c['index']}",
)
def test_argument_type_list_input_type_matches_cpp(case):
    atl = [str_to_type_tree(x) for x in ("boolean", "contin", "action_result")]
    got = argument_type_list_input_type(atl, case["arity"], case["index"])
    assert ostream_type_tree(got) == case["result"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["misc"]["get_type_node"], ids=lambda c: c["tt"])
def test_get_type_node_matches_cpp(case):
    tt = str_to_type_tree(case["tt"])
    if case.get("threw"):
        with pytest.raises(AssertionError):
            get_type_node(tt)
        return
    assert int(get_type_node(tt)) == case["result"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", _GOLDEN["misc"]["default_vertex_value"], ids=lambda c: str(c["ord"])
)
def test_default_vertex_value_matches_cpp(case):
    v = default_vertex_value(type_node(case["ord"]))
    # `vertex()`'s default is the contin 0.0, which classic's own
    # `operator<<` prints as "0"; only `boolean_type` prints "false".
    printed = "false" if v is builtin.logical_false else ("0" if v == 0.0 else repr(v))
    assert printed == case["value"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["misc"]["is_argument_type"], ids=lambda c: str(c["ord"]))
def test_is_argument_type_and_arg_to_idx_match_cpp(case):
    n = type_node(case["ord"])
    assert int(is_argument_type(n)) == case["result"]
    if case["result"]:
        assert arg_to_idx(n) == case["arg_to_idx"]
    else:
        with pytest.raises(AssertionError):
            arg_to_idx(n)


# ---------------------------------------------------------------------------
# Guards -- the things a passing oracle above could still be hiding
# ---------------------------------------------------------------------------


def test_the_corpus_actually_contains_ill_typed_programs():
    """The whole point of the fixture.

    A corpus of only well-typed programs would let `is_well_formed`
    returning a constant `True` pass every row above. This asserts the
    fixture still holds all three ill-typed FAMILIES, each of which a
    different partial implementation gets wrong:

    * a cross-argument mismatch, which reaches `ill_formed` through
      `get_intersection` inside `infer_arg_type_tree` and never touches
      `reduce_type_tree`;
    * an operand whose OUTPUT type does not inherit the input its operator
      demands, which reaches it through `reduce_type_tree` and is
      invisible to any per-argument rule;
    * a wrong applied-argument COUNT, which reaches it through
      `infer_vertex_type`'s arity guard.
    """
    by_combo = {c["combo"]: c for c in _GOLDEN["programs"]}
    cross_argument = ["+(impulse($1) $1)", "and($1 0<($1))", "or($1 +($1 1))"]
    operand_type = ["+(and($1 $2) 1)", "and(+($1 $2) $3)", "exp(and($1 $2))"]
    wrong_arity = ["not($1 $2)", "/($1)", "impulse($1 $2)", "0<($1 $2)"]
    for family in (cross_argument, operand_type, wrong_arity):
        for combo in family:
            assert by_combo[combo]["is_well_formed"] == 0, combo
    assert sum(1 for c in _GOLDEN["programs"] if c["is_well_formed"] == 0) >= 20
    assert sum(1 for c in _GOLDEN["programs"] if c["is_well_formed"] == 1) >= 20


@pytest.mark.parametrize(
    "combo",
    ["+(and($1 $2) 1)", "and(+($1 $2) $3)", "not($1 $2)", "/($1)", "+(impulse($1) $1)"],
)
def test_the_engine_rejects_what_the_old_substitute_accepted(combo):
    """The observable defect this module was written to close.

    `demo_problems.py`'s narrow `_infer_signature` substitute caught only
    the cross-argument family: it walked arguments and compared what each
    parent DEMANDED of its children, so `+(impulse($1) $1)` was caught,
    while `+(and($1 $2) 1)` and `and(+($1 $2) $3)` (an operand's own
    OUTPUT type) and `not($1 $2)` and `/($1)` (a wrong applied-argument
    count) were all accepted. Measured before the change: classic exits
    255 with "is not well formed" for all five, while `moses -Hcp -y` on
    this port ran `not($1 $2)` and `/($1)` to completion and printed a
    result, and crashed later on the other two.
    """
    tt = infer_type_tree(str_to_combo_tree(combo))
    assert not is_well_formed(tt)


def test_equ_token_parses_as_a_definite_object_in_classic_and_is_refused_here():
    """The two fixture rows this port cannot reproduce, asserted rather
    than skipped.

    Classic's combo parser reads a bare `equ` as a `definite_object`
    string vertex, NOT as `id::equ` -- the fixture's own `vertices` row
    says `definite_object`, arity 0, no input types -- and `equ($1 $2)`
    is therefore an application of a non-function, `->(ill_formed
    ill_formed ill_formed)`. This port's grammar recognises builtin,
    argument and contin tokens only (`iostream_combo.py:332`) and raises.
    Both reject the program; they disagree about how.

    The BUILTIN-level fact is separately verified and does match: the
    `builtins` section's `equ` row (`->(enum_t enum_t boolean)`, arity 2)
    is asserted by `test_builtins_properties_matches_cpp`.
    """
    for combo in sorted(_UNPARSEABLE_BY_THIS_PORT):
        with pytest.raises(ValueError):
            str_to_combo_tree(combo)
    equ_row = next(c for c in _GOLDEN["vertices"] if c["vertex"] == "equ")
    assert equ_row["get_type_tree"] == "definite_object"
    assert equ_row["get_arity"] == 0
    equ_prog = next(c for c in _GOLDEN["programs"] if c["combo"] == "equ($1 $2)")
    assert equ_prog["is_well_formed"] == 0


def test_the_action_vocabulary_has_no_python_prerequisite():
    """Pins the reason `get_type_tree(action)` and its two siblings are
    not ported, so that removing the reason turns this red.

    `descriptions.py`'s module docstring says `actions_properties` is not
    ported because `id::action` has no representation here at all. If
    someone adds an `action` alternative to the `vertex` union, or an
    `is_action` predicate, then `get_type_tree`/`get_output_type_tree`/
    `get_input_type_tree`/`get_arity` acquire a reachable arm they do not
    implement, and this test is the tripwire. The fixture's `actions`
    section is the measured record of what that arm would have to
    reproduce.
    """
    assert not hasattr(vertex_module, "is_action")
    assert not hasattr(vertex_module, "is_perception")
    assert not hasattr(vertex_module, "is_builtin_action")
    assert not hasattr(vertex_module, "action")
    assert len(_GOLDEN["actions"]) == 14
    assert _GOLDEN["actions"][0]["type_tree"] == "->(arg_list(action_result) action_result)"


def test_a_str_vertex_raises_in_every_per_vertex_dispatch():
    """The excluded enum / object / message domains, gated rather than
    guessed. `str` conflates six C++ vertex kinds here, so none of the
    four dispatch functions can answer for one.
    """
    for fn in (get_type_tree, get_output_type_tree, get_arity):
        with pytest.raises((AssertionError, NotImplementedError)):
            fn("some_definite_object")
    with pytest.raises(AssertionError):
        get_input_type_tree("some_definite_object", 0)


def test_type_tree_equality_is_structural_not_root_only():
    """`type_tree.__eq__` ports `operator==(tree<T>, tree<T>)`, which is a
    whole-shape comparison. The class this module replaces compared only a
    single `type_node`, so a root-only `__eq__` would still pass every
    single-node use in `meta_rules.py` and `mixed_rules.py` while being
    wrong for every compound type the engine now builds.
    """
    a = str_to_type_tree("->(boolean boolean)")
    b = str_to_type_tree("->(boolean contin)")
    c = str_to_type_tree("->(boolean boolean)")
    assert a == c
    assert a != b
    assert type_tree(type_node.boolean_type) == type_tree(type_node.boolean_type)
    assert type_tree() == type_tree()
    assert type_tree() != type_tree(type_node.boolean_type)
    # ... and the S7 accessor the two `main/` call sites still use.
    assert type_tree(type_node.contin_type).node is type_node.contin_type


def test_fixture_manifest_matches_current_probe_source():
    """Compiler-free rot detection: editing `type_tree_probe.cc` without
    re-running `tools/capture_type_tree.py` fails here.
    """
    assert _MANIFEST["schema_version"] in SUPPORTED_SCHEMA_VERSIONS
    assert _MANIFEST["probe_source"] == "tools/cpp_probe/type_tree_probe.cc"
    assert _MANIFEST["capture_script"] == "tools/capture_type_tree.py"
    probe = _REPO_ROOT / "tools" / "cpp_probe" / "type_tree_probe.cc"
    assert _MANIFEST["probe_source_sha256"] == hashlib.sha256(probe.read_bytes()).hexdigest()
