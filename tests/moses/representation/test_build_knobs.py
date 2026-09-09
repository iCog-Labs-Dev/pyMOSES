"""Oracle and boundary tests for deterministic pure-Boolean knob building."""

import copy
import inspect
import json

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import (
    combo_tree_to_str,
    str_to_combo_tree,
)
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    swap_and_or,
    vertex_equal,
)
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    gen_signature,
    type_node,
)
from pymetta.moses.representation.build_knobs import build_knobs
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.representation import representation
from pymetta.util.rng import randGen

_CASE_KEYS = {
    "name",
    "source",
    "arity",
    "ignored_arguments",
    "decorated_exemplar",
    "prototype",
    "probe_calls",
    "knobs",
}
_KNOB_KEYS = {
    "order",
    "multy",
    "text",
    "in_exemplar",
    "location_preorder",
    "location_subtree",
}
_EXPECTED_CASES = (
    ("constant-true", "true", 0, [], 0, 0),
    ("bare-argument", "$1", 1, [], 10, 1),
    ("small-and", "and($1 $2)", 2, [], 24, 2),
    ("small-or-negated", "or($1 !$2)", 2, [], 24, 2),
    ("nested-three", "and(or($1 $2) !$3)", 3, [], 54, 7),
    ("ignored-argument", "and($1 $2)", 2, [2], 12, 1),
)

_FUTURE_OUTPUT_TYPES = {
    type_node.action_result_type,
}


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "boolean_builder.json").read_text())


def _render(tr: combo_tree) -> str:
    return combo_tree_to_str(tr).strip()


def _render_vertex(value: object) -> str:
    return _render(combo_tree(copy.copy(value)))


def _prototype(exemplar: combo_tree, mapper: knob_mapper) -> str:
    def render_node(node) -> str:
        knob = mapper.find_disc_knob(node)
        text = knob.toStr() if knob is not None else _render_vertex(node.data)
        effective = node
        if vertex_equal(node.data, builtin.null_vertex):
            assert exemplar.has_one_child(node)
            effective = node.first_child
        children = list(exemplar.children(effective))
        if children:
            text += "(" + " ".join(render_node(child) for child in children) + ")"
        return text

    return render_node(exemplar.begin())


def _python_case(case: dict[str, object]) -> dict[str, object]:
    exemplar = str_to_combo_tree(case["source"])
    signature = gen_signature(type_node.boolean_type, case["arity"])
    ignored = {argument(idx) for idx in case["ignored_arguments"]}
    reduction = logical_reduction(ignored)(2)
    probe_calls: list[dict[str, str]] = []

    def simplify_knob_building(tr: combo_tree) -> None:
        before = _render(tr)
        reduction(tr)
        probe_calls.append({"before": before, "after": _render(tr)})

    mapper = knob_mapper()
    build_knobs(
        exemplar,
        signature,
        mapper,
        simplify_knob_building,
        ignore_ops=ignored,
        perm_ratio=-1,
    )
    preorder = {node: idx for idx, node in enumerate(exemplar)}
    knobs = [
        {
            "order": order,
            "multy": entry.spec.multy,
            "text": entry.knob.toStr(),
            "in_exemplar": entry.knob.in_exemplar(),
            "location_preorder": preorder[entry.knob.get_loc()],
            "location_subtree": _render(combo_tree(entry.knob.get_loc())),
        }
        for order, entry in enumerate(mapper.disc)
    ]
    return {
        "name": case["name"],
        "source": case["source"],
        "arity": case["arity"],
        "ignored_arguments": case["ignored_arguments"],
        "decorated_exemplar": _render(exemplar),
        "prototype": _prototype(exemplar, mapper),
        "probe_calls": probe_calls,
        "knobs": knobs,
    }


@pytest.mark.oracle
def test_boolean_builder_fixture_schema_and_inputs_are_non_vacuous(golden):
    assert set(golden) == {"perm_ratio", "cases"}
    assert golden["perm_ratio"] == -1
    assert isinstance(golden["cases"], list)
    assert [
        (
            case["name"],
            case["source"],
            case["arity"],
            case["ignored_arguments"],
            len(case["probe_calls"]),
            len(case["knobs"]),
        )
        for case in golden["cases"]
    ] == list(_EXPECTED_CASES)
    assert all(set(case) == _CASE_KEYS for case in golden["cases"])
    assert all(
        set(knob) == _KNOB_KEYS
        for case in golden["cases"]
        for knob in case["knobs"]
    )
    assert sum(len(case["probe_calls"]) for case in golden["cases"]) == 124
    assert sum(len(case["knobs"]) for case in golden["cases"]) == 13
    assert any(
        knob["multy"] == 2 for case in golden["cases"] for knob in case["knobs"]
    )


@pytest.mark.oracle
def test_deterministic_boolean_builder_matches_classic(golden):
    assert [_python_case(case) for case in golden["cases"]] == golden["cases"]


@pytest.mark.parametrize(
    "output_type",
    [
        node
        for node in type_node
        if node not in (type_node.boolean_type, type_node.contin_type)
    ],
)
def test_builder_rejects_every_non_boolean_non_contin_output_before_mutation(output_type):
    """``contin_type`` output is excluded from this parametrization -- it is
    now a SUPPORTED output (S14 Task 2), covered by its own positive tests
    below, not by this "every other type is still rejected" sweep.
    """
    exemplar = str_to_combo_tree("$1")
    before = _render(exemplar)
    signature = gen_signature(type_node.boolean_type, output_type, 1)
    message = "future allowed domain slice" if output_type in _FUTURE_OUTPUT_TYPES else "excluded"

    with pytest.raises(NotImplementedError, match=message):
        build_knobs(exemplar, signature, knob_mapper(), lambda _tr: None)

    assert _render(exemplar) == before


@pytest.mark.parametrize(
    "input_type", [node for node in type_node if node is not type_node.boolean_type]
)
def test_builder_rejects_every_non_boolean_input_before_mutation(input_type):
    exemplar = str_to_combo_tree("$1")
    before = _render(exemplar)
    signature = TypeSignature((input_type,), type_node.boolean_type)

    with pytest.raises(NotImplementedError, match="non-Boolean inputs"):
        build_knobs(exemplar, signature, knob_mapper(), lambda _tr: None)

    assert _render(exemplar) == before


@pytest.mark.parametrize(
    ("source", "arity", "message"),
    [
        ("not($1)", 1, "unsupported or non-reduced"),
        ("null_vertex($1)", 1, "unsupported or non-reduced"),
        ("0", 0, "unsupported or non-reduced"),
        # 3.0 equals IntEnum builtin.logical_and under bare Python equality.
        ("3", 0, "unsupported or non-reduced"),
        ("plus($1)", 1, "unsupported or non-reduced"),
        ("$2", 1, "outside signature arity"),
    ],
)
def test_builder_rejects_unsupported_or_mistyped_vertices_without_mutation(
    source, arity, message
):
    exemplar = str_to_combo_tree(source)
    before = _render(exemplar)

    with pytest.raises(ValueError, match=message):
        build_knobs(
            exemplar,
            gen_signature(type_node.boolean_type, arity),
            knob_mapper(),
            lambda _tr: None,
        )

    assert _render(exemplar) == before


def test_builder_rejects_empty_or_structurally_invalid_exemplars():
    signature = gen_signature(type_node.boolean_type, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        build_knobs(combo_tree(), signature, knob_mapper(), lambda _tr: None)

    argument_parent = combo_tree(argument(1))
    argument_parent.append_child(argument_parent.begin(), argument(1))
    with pytest.raises(ValueError, match="must be childless"):
        build_knobs(argument_parent, signature, knob_mapper(), lambda _tr: None)

    constant_parent = combo_tree(builtin.logical_true)
    constant_parent.append_child(constant_parent.begin(), argument(1))
    with pytest.raises(ValueError, match="must be childless"):
        build_knobs(constant_parent, signature, knob_mapper(), lambda _tr: None)


@pytest.mark.parametrize("perm_ratio", [True, "-1", None])
def test_builder_rejects_mistyped_perm_ratio(perm_ratio):
    """C++: moses/moses/representation/build_knobs.cc:53 build_knobs::build_knobs

    Classic takes a `float` and has no validation at all; this port keeps a
    type guard (and rejects `bool`, which `isinstance(True, int)` would
    otherwise smuggle through as ratio 1.0). It no longer rejects legal
    RATIOS -- see `test_builder_accepts_classic_perm_ratio_range`.
    """
    exemplar = str_to_combo_tree("$1")
    with pytest.raises(TypeError, match="perm_ratio must be a real number"):
        build_knobs(
            exemplar,
            gen_signature(type_node.boolean_type, 1),
            knob_mapper(),
            lambda _tr: None,
            perm_ratio=perm_ratio,
        )


def test_builder_rejects_nan_perm_ratio():
    exemplar = str_to_combo_tree("$1")
    with pytest.raises(ValueError, match="must not be NaN"):
        build_knobs(
            exemplar,
            gen_signature(type_node.boolean_type, 1),
            knob_mapper(),
            lambda _tr: None,
            perm_ratio=float("nan"),
        )


@pytest.mark.parametrize("perm_ratio", [-2.0, -1.0, -0.5, 0, 0.0, 0.5, 1, 1.0])
def test_builder_accepts_classic_perm_ratio_range(perm_ratio):
    """C++: moses/moses/representation/build_knobs.cc:435 build_knobs::sample_logical_perms

    Every one of these was rejected outright before pair sampling was
    ported, INCLUDING classic's own default of 0.0
    (build_knobs.h:56, representation.h:62, problem-params.cc:299).
    """
    randGen().seed(1)
    exemplar = str_to_combo_tree("and($1 $2)")
    build_knobs(
        exemplar,
        gen_signature(type_node.boolean_type, 2),
        knob_mapper(),
        logical_reduction(())(2),
        perm_ratio=perm_ratio,
    )


@pytest.mark.parametrize(
    ("source", "arity", "perm_ratio", "expected_knobs"),
    [
        # `_perm_ratio <= -1.0` returns before any pair is built
        # (build_knobs.cc:435-437) -- note -2.0 behaves like -1.0, which the
        # old `perm_ratio == -1.0` VALIDATION guard rejected outright.
        ("true", 2, -2.0, 8),
        ("true", 2, -1.0, 8),
        # `n_pairs = floor((1 + r) * ps)` (build_knobs.cc:479), ps == 2.
        ("true", 2, -0.5, 12),
        # classic's DEFAULT: floor(1.0 * 2) == 2 pairs per junctor.
        ("true", 2, 0.0, 16),
        # `n_pairs = floor(ps + r * (max_pairs - ps))` (build_knobs.cc:477).
        # At arity 2, max_pairs == 2 == ps, so no ratio can exceed 2 pairs.
        ("true", 2, 0.5, 16),
        ("true", 2, 1.0, 16),
        # At arity 3, max_pairs == 6 > ps == 3, so the two branches really
        # do separate: 3, floor(3 + 0.5*3) == 4, and 6 pairs per junctor.
        ("and(or($1 $2) !$3)", 3, -1.0, 7),
        ("and(or($1 $2) !$3)", 3, -0.5, 9),
        ("and(or($1 $2) !$3)", 3, 0.0, 13),
        ("and(or($1 $2) !$3)", 3, 0.5, 19),
        ("and(or($1 $2) !$3)", 3, 1.0, 23),
    ],
)
def test_pair_count_follows_classic_n_pairs_formula(source, arity, perm_ratio, expected_knobs):
    """C++: moses/moses/representation/build_knobs.cc:474 build_knobs::sample_logical_perms

    `sample_logical_perms` runs once per junctor, so `n_pairs` is not
    directly observable; the knob count is (`disc_probe` can still veto
    individual perms, which is why these are measured totals rather than a
    closed form). What matters is that all three of classic's branches --
    the `<= -1.0` early return, the `0.0 < _perm_ratio` branch and the
    `floor((1 + r) * ps)` branch -- are separately visible here.

    EQUIVALENT MUTANT, recorded rather than hidden: rewriting the early
    return's `self._perm_ratio <= -1.0` as `== -1.0` leaves this entire
    file green, including the -2.0 row. It is not a gap in the corpus -- the
    two really are equivalent in Python, because a ratio below -1 makes
    `floor((1 + r) * ps)` negative and `range(negative)` is empty. In C++
    they differ (the negative `floor` is cast to `size_t`), which is why
    the port keeps classic's spelling; see `sample_logical_perms`' own
    comment. Deleting the early return outright IS caught, by every row
    below -1 here and by `test_2_parity_stalls_without_pair_perms`.
    """
    randGen().seed(1)
    exemplar = str_to_combo_tree(source)
    mapper = knob_mapper()
    build_knobs(
        exemplar,
        gen_signature(type_node.boolean_type, arity),
        mapper,
        logical_reduction(())(2),
        perm_ratio=perm_ratio,
    )
    assert len(mapper.disc) == expected_knobs


@pytest.mark.oracle
def test_default_perm_ratio_reproduces_classic_pa_k2_decorated_exemplar():
    """C++: moses/moses/representation/build_knobs.cc:411 build_knobs::sample_logical_perms

    The exact string below was READ OFF classic's own `-l DEBUG` log for
    `moses -Hpa -k2 -W1 -r1` (the final "Subtree:" line emitted by
    `build_logical`, immediately before "After knob building, rep size=52 /
    Total number of field specs: 16"). The inputs reproduce that run's first
    deme exactly: exemplar `true`, arity 2, `logical_reduction` at effort 2,
    classic's default `perm_ratio = 0.0`, and seed 1 (classic's own CLI
    default, `-r1`) -- knob building is the first consumer of `randGen()` in
    that run, so the seed alone pins the `lazy_random_selector` draws.

    This is the regression guard for the parity divergence: before pair
    sampling was ported the port produced an 8-knob tree here instead of
    classic's 16, and the `pa` problem became unsolvable. Every one of the
    four `or(...)`/`and(...)` pair subtrees below, and their ORDER within
    each junctor, is RNG-chosen.
    """
    randGen().seed(1)
    exemplar = str_to_combo_tree("true")
    mapper = knob_mapper()
    build_knobs(
        exemplar,
        gen_signature(type_node.boolean_type, 2),
        mapper,
        logical_reduction(())(2),
        perm_ratio=0.0,
    )

    assert len(mapper.disc) == 16
    assert _render(exemplar) == (
        "or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) "
        "null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) "
        "null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) "
        "null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) "
        "null_vertex(and($1 $2)) and(null_vertex($1) null_vertex($2) "
        "null_vertex(or($1 $2)) null_vertex(or(!$1 $2))))"
    )


def test_perm_ratio_defaults_match_classic():
    """C++: moses/moses/representation/build_knobs.h:56 build_knobs

    `float perm_ratio = 0.0` there, and the same default on
    `representation::representation` (representation.h:62) and on the CLI
    (`problem-params.cc:299`). This port carried -1.0 in both signatures,
    which silently halved every Boolean field set; pinned so it cannot
    drift back unnoticed.
    """
    assert inspect.signature(build_knobs).parameters["perm_ratio"].default == 0.0
    assert inspect.signature(representation).parameters["perm_ratio"].default == 0.0


def test_swap_and_or_rejects_non_junctors():
    """C++: moses/comboreduct/combo/vertex.cc:257 swap_and_or"""
    assert swap_and_or(builtin.logical_and) is builtin.logical_or
    assert swap_and_or(builtin.logical_or) is builtin.logical_and
    with pytest.raises(AssertionError, match="logical_or"):
        swap_and_or(builtin.logical_true)


def test_builder_rejects_bad_collaborators_before_mutation():
    exemplar = str_to_combo_tree("$1")
    signature = gen_signature(type_node.boolean_type, 1)
    before = _render(exemplar)

    with pytest.raises(TypeError, match="TypeSignature"):
        build_knobs(exemplar, object(), knob_mapper(), lambda _tr: None)
    with pytest.raises(TypeError, match="knob_mapper"):
        build_knobs(exemplar, signature, object(), lambda _tr: None)
    with pytest.raises(TypeError, match="must be callable"):
        build_knobs(exemplar, signature, knob_mapper(), None)

    finalized = knob_mapper()
    finalized.finalize()
    with pytest.raises(ValueError, match="fresh, empty"):
        build_knobs(exemplar, signature, finalized, lambda _tr: None)
    assert _render(exemplar) == before
