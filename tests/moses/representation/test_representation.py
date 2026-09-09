"""C++ oracle and immutability tests for Boolean candidate construction."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import (
    combo_tree_to_str,
    str_to_combo_tree,
)
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.table.table import complete_truth_table
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import logical_subtree_knob
from pymetta.moses.representation.representation import representation

_CASE_KEYS = {
    "name",
    "source",
    "arity",
    "ignored_arguments",
    "decorated_exemplar",
    "packed_width",
    "raw_size",
    "exhaustive",
    "knobs",
    "rows",
}
_KNOB_KEYS = {"order", "multy", "text", "location_preorder", "raw_idx"}
_ROW_KEYS = {
    "raw",
    "packed",
    "unreduced",
    "reduced",
    "unreduced_tt",
    "reduced_tt",
}


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "representation.json").read_text())


def _make_rep(case):
    ignored = {argument(idx) for idx in case["ignored_arguments"]}
    reducer = logical_reduction(ignored)(2)
    return representation(
        reducer,
        reducer,
        str_to_combo_tree(case["source"]),
        gen_signature(type_node.boolean_type, case["arity"]),
        ignore_ops=ignored,
        perm_ratio=-1,
    )


def _truth_outcome(tr: combo_tree, arity: int) -> tuple[bool, str | None]:
    try:
        return True, "".join("1" if value else "0" for value in complete_truth_table(tr, arity))
    except (AssertionError, ValueError):
        return False, None


@pytest.mark.oracle
def test_representation_fixture_schema_is_bounded_and_non_vacuous(golden):
    assert set(golden) == {"perm_ratio", "cases"}
    assert golden["perm_ratio"] == -1
    assert sum(len(case["rows"]) for case in golden["cases"]) == 1969
    assert max(len(case["rows"]) for case in golden["cases"]) == 972
    assert all(set(case) == _CASE_KEYS for case in golden["cases"])
    assert all(case["exhaustive"] for case in golden["cases"])
    assert all(
        set(knob) == _KNOB_KEYS
        for case in golden["cases"]
        for knob in case["knobs"]
    )
    assert all(
        set(row) == _ROW_KEYS for case in golden["cases"] for row in case["rows"]
    )
    assert sum(
        not row[key]["computed"]
        for case in golden["cases"]
        for row in case["rows"]
        for key in ("unreduced_tt", "reduced_tt")
    ) == 6


@pytest.mark.oracle
@pytest.mark.parametrize("case_idx", range(7))
def test_all_bounded_candidates_match_classic_bytes_indexes_and_truth_tables(
    golden, case_idx
):
    case = golden["cases"][case_idx]
    rep = _make_rep(case)

    assert combo_tree_to_str(rep.exemplar()) == case["decorated_exemplar"]
    assert rep.fields().packed_width() == case["packed_width"]
    assert rep.fields().raw_size() == case["raw_size"]
    preorder = {node: idx for idx, node in enumerate(rep._exemplar)}
    assert [
        {
            "order": order,
            "multy": entry.spec.multy,
            "text": entry.knob.toStr(),
            "location_preorder": preorder[entry.knob.get_loc()],
            "raw_idx": rep._it_disc_idx[entry.knob.get_loc()],
        }
        for order, entry in enumerate(rep.disc)
    ] == case["knobs"]

    for row in case["rows"]:
        inst = rep.fields().pack(row["raw"])
        assert list(inst) == row["packed"]
        for reduce, tree_key, truth_key in (
            (False, "unreduced", "unreduced_tt"),
            (True, "reduced", "reduced_tt"),
        ):
            candidate = rep.get_candidate(inst, reduce)
            assert combo_tree_to_str(candidate) == row[tree_key]
            computed, bits = _truth_outcome(candidate, case["arity"])
            assert computed is row[truth_key]["computed"]
            if computed:
                assert bits == row[truth_key]["bits"]
            else:
                assert isinstance(row[truth_key]["error"], str)
                assert row[truth_key]["error"]


@pytest.mark.oracle
def test_knob_order_case_discriminates_the_descending_multy_sort(golden):
    """Task 3's exit criterion is "knob indexes match C++", but the other six
    fixture cases cannot discriminate `knob_mapper`'s ``sorted(..., reverse=
    True)`` (knob_mapper.py) from a no-op sort: their insertion order already
    equals sorted order. ``and-wraps-or`` (``and($1 or($2 $3))``) does not --
    measured insertion order is ``[3, 3, 3, 2, 2, 3, 3]`` -- so this is the
    first fixture row where a wrong sort direction changes the observable
    knob order. See ``tools/cpp_probe/candidate_roundtrip_probe.cc``'s
    "and-wraps-or" case and its comment.
    """
    case = next(c for c in golden["cases"] if c["name"] == "and-wraps-or")
    rep = _make_rep(case)
    multys = [entry.spec.multy for entry in rep.disc]
    assert multys == [3, 3, 3, 3, 3, 2, 2]
    assert multys != [3, 3, 3, 2, 2, 3, 3]  # the un-sorted insertion order
    assert multys == [knob["multy"] for knob in case["knobs"]]


def test_constructor_and_exemplar_access_do_not_alias_caller_or_internal_tree():
    source = str_to_combo_tree("and($1 $2)")
    reducer = logical_reduction(set())(2)
    rep = representation(
        reducer,
        reducer,
        source,
        gen_signature(type_node.boolean_type, 2),
    )
    source_before = combo_tree_to_str(source)
    first = rep.exemplar()
    second = rep.exemplar()

    first_arg = next(node for node in first if isinstance(node.data, argument))
    first_arg.data.negate()
    first.erase(first_arg)

    assert combo_tree_to_str(source) == source_before
    assert combo_tree_to_str(second) == combo_tree_to_str(rep.exemplar())
    assert first.begin() is not second.begin()


def test_candidates_own_mutable_payloads_and_do_not_change_later_results(golden):
    case = golden["cases"][2]
    rep = _make_rep(case)
    inst = rep.fields().pack([0, 0])
    first = rep.get_candidate(inst, False)
    expected = combo_tree_to_str(first)
    first_arg = next(node for node in first if isinstance(node.data, argument))
    first_arg.data.negate()
    first.erase(first_arg)

    second = rep.get_candidate(inst, False)
    assert combo_tree_to_str(second) == expected
    assert second.begin() is not first.begin()


def test_cleaning_precedes_selected_reducer_and_does_not_run_again_afterward():
    seen: list[str] = []

    def candidate_reducer(tr: combo_tree) -> None:
        seen.append(combo_tree_to_str(tr))
        tr.append_child(tr.begin(), builtin.null_vertex)

    rep = representation(
        candidate_reducer,
        lambda _tr: None,
        str_to_combo_tree("true"),
        gen_signature(type_node.boolean_type, 0),
    )
    target = str_to_combo_tree("and(null_vertex($1) $2)")
    rep.clean_combo_tree(target, True)

    assert seen == ["and($2) "]
    assert combo_tree_to_str(target) == "and($2 null_vertex) "


def test_get_clean_exemplar_selects_the_requested_reducer():
    calls: list[str] = []

    def candidate(_tr: combo_tree) -> None:
        calls.append("candidate")

    def building(_tr: combo_tree) -> None:
        calls.append("building")

    rep = representation(
        candidate,
        building,
        str_to_combo_tree("true"),
        gen_signature(type_node.boolean_type, 0),
    )
    calls.clear()
    rep.get_clean_exemplar(False)
    rep.get_clean_exemplar(True)
    rep.get_clean_exemplar(True, knob_building=True)
    assert calls == ["candidate", "building"]


def test_invalid_instances_fail_without_mutating_representation(golden):
    rep = _make_rep(golden["cases"][2])
    expected = combo_tree_to_str(rep.exemplar())
    with pytest.raises(TypeError, match="requires an instance"):
        rep.get_candidate([0], False)
    with pytest.raises(ValueError, match="field set requires"):
        rep.get_candidate(instance(rep.fields().packed_width() + 1), False)
    invalid = rep.fields().pack([3, 0])
    with pytest.raises(ValueError, match="invalid discrete knob index"):
        rep.get_candidate(invalid, False)
    assert combo_tree_to_str(rep.exemplar()) == expected


def test_representation_rejects_still_future_builder_paths_without_source_mutation():
    """``contin_type`` output moved from "future" to supported in S14 Task 2
    (see ``test_contin_representation.py``'s own oracle tests); this now
    exercises a domain that remains genuinely unported instead.
    """
    source = str_to_combo_tree("$1")
    before = combo_tree_to_str(source)
    with pytest.raises(NotImplementedError, match="future allowed domain slice"):
        representation(
            lambda _tr: None,
            lambda _tr: None,
            source,
            gen_signature(type_node.action_result_type, 1),
        )
    assert combo_tree_to_str(source) == before


def test_concurrent_candidate_calls_are_deterministic_and_isolated(golden):
    case = golden["cases"][4]
    rep = _make_rep(case)
    selected = case["rows"][::37]

    def render(row):
        inst = rep.fields().pack(row["raw"])
        return combo_tree_to_str(rep.get_candidate(inst, False))

    jobs = selected * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(render, jobs))
    assert actual == [row["unreduced"] for row in jobs]


def test_candidate_recursion_descends_into_a_composite_knob_subtree():
    """Pin the knob-subtree recursion the fixture corpus structurally cannot reach.

    ``get_candidate_rec`` recurses into ``new_src``'s exemplar children, but
    only a knob sitting on a *composite* subtree has any. With ``perm_ratio ==
    -1`` classic returns at ``build_knobs.cc:435-437`` before generating the
    ``op($i $j)`` pair permutations, so every knob it builds sits on a single
    literal: measured, 0 of the 20 knobs across all seven fixture cases has a
    child under ``new_src``. The 1969 oracle rows therefore give this branch no
    coverage at all, and both a "skip the recursion" and an "append under the
    original parent_dst" mutant survive them.

    A knob reaches ``_default == present`` on a composite node when its subtree
    equals an existing sibling (``knobs.py`` ctor), which is what a pair perm
    does once sampling is enabled. This test builds that state directly.

    Provenance: MEASURED against classic, not merely derived by reading it.
    ``tools/cpp_probe/composite_knob_probe.cc`` constructs the identical
    composite knob directly (bypassing pair sampling, which this slice defers,
    the same way this test does) and transcribes ``get_candidate_rec``'s
    generic recursion shape around the REAL compiled
    ``logical_subtree_knob::append_to`` (``knobs.cc:164-190``, which
    overwrites ``parent_dst`` through a non-const reference) and
    ``combo_tree::is_valid`` -- i.e. the part actually in question (composite-
    knob descent) runs classic's own code, not a reimplementation of it. Its
    measured output is byte-for-byte this test's three expected rows (module
    trailing-space rendering differences) and an unchanged exemplar. The
    negated row is additionally cross-checked semantically here: ``not(and($1
    $2))`` reduces by De Morgan to ``or(!$1 !$2)``, which only holds if the
    two arguments were really appended beneath the ``not``.
    """
    reducer = logical_reduction(set())(2)
    tr = str_to_combo_tree("or(and($1 $2) $3)")
    subtree = str_to_combo_tree("and($1 $2)")

    rep = representation.__new__(representation)
    knob_mapper.__init__(rep)
    rep._exemplar = tr
    rep._simplify_candidate = reducer
    rep._simplify_knob_building = reducer
    knob = logical_subtree_knob(tr, tr.begin(), subtree.begin())
    rep.add_disc_knob(knob)
    rep.finalize()
    rep._fields = field_set(entry.spec for entry in rep.disc)
    start = rep._fields.begin_disc_raw_idx()
    rep._it_disc_idx = {
        entry.knob.get_loc(): start + offset
        for offset, entry in enumerate(rep.disc)
    }

    # Without both of these the branch under test is not entered at all.
    assert knob._default == knob.present
    assert len(list(tr.children(knob.get_loc()))) == 2

    seen = {}
    for raw in range(knob.multiplicity()):
        inst = rep.fields().pack([raw])
        seen[combo_tree_to_str(rep.get_candidate(inst, False))] = combo_tree_to_str(
            rep.get_candidate(inst, True)
        )

    assert seen == {
        "or(and($1 $2) $3) ": "or(and($1 $2) $3) ",
        "or($3) ": "$3 ",
        "or(not(and($1 $2)) $3) ": "or(!$1 !$2 $3) ",
    }

    # C++ representation.cc:313: get_candidate "does not modify the rep at all".
    # Before the copy at the recursion's append_child, reducing the negated
    # candidate rewrote the shared `argument` objects in place through
    # `argument.negate()`, leaving the exemplar as `or(and(!$1 !$2) $3)`.
    assert combo_tree_to_str(rep.exemplar()) == "or(and($1 $2) $3) "
