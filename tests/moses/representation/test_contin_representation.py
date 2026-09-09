"""C++ oracle tests for the contin-output representation path (S14 Task 2).

The exit criterion is "the knobs built for a contin exemplar match C++ in
count, order and spec" -- measured against ``tests/fixtures/
contin_representation.json``, captured by ``tools/capture_contin_
representation.py`` from ``tools/cpp_probe/contin_representation_probe.cc``,
which drives the REAL classic ``representation`` C++ class (never a
reimplementation) at its own real module-static defaults (``stepsize=1.0,
expansion=2.0, depth=5``, ``representation.cc:59-61``).

Half of the fixture's 18 cases (the ``-nonlinear`` ones) run classic at
``linear_contin=False`` -- the real ``moses`` CLI's own default
(``--linear-regression``, problem-params.cc:292), NOT merely this
constructor's own literal default (``True``, matching representation.h:61
verbatim -- see CARRY-FORWARD.md Sec 5.4, "an oracle configured by the port
cannot detect the port's configuration being wrong"). Every one of those
cases is diffed against classic's OWN compiled binary output, not against a
probe this port configured -- see ``test_all_...`` below.
"""

import json
import sys

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.interpreter.interpreter import contin_interpreter
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import contin_knob
from pymetta.moses.representation.representation import representation

# DIVERGENCE(no C++ symbol): `pymetta/moses/representation/__init__.py` does
# `from .representation import representation`, which -- as a side effect of
# how `from X import Y` binds names -- overwrites the package's own
# `representation` ATTRIBUTE (normally the submodule, set by the import
# machinery) with the CLASS of the same name. So
# `import pymetta.moses.representation.representation as m` silently binds
# `m` to the CLASS, not the submodule (`import a.b.c as x` resolves via
# attribute access on `a.b`, not a direct `sys.modules` lookup) -- this is
# NOT this test file's own bug, it is Python's own attribute-shadowing rule
# meeting this package's `__init__.py` shape. `sys.modules` is unaffected by
# the shadowing, so this looks the real submodule up there instead, to reach
# its `set_stepsize`/`set_expansion`/`set_depth`/`_stepsize`/`_expansion`/
# `_depth` module-level state.
representation_module = sys.modules["pymetta.moses.representation.representation"]

_TYPE_NAME = {
    "boolean": type_node.boolean_type,
    "contin": type_node.contin_type,
}

_CASE_KEYS = {
    "name",
    "source",
    "itypes",
    "linear_contin",
    "decorated_exemplar",
    "prototype",
    "disc_knob_count",
    "contin_knobs",
}
_KNOB_KEYS = {
    "order",
    "mean",
    "step_size",
    "expansion",
    "depth",
    "text",
    "location_preorder",
    "location_subtree",
}


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "contin_representation.json").read_text())


def _make_rep(case):
    itypes = [_TYPE_NAME[name] for name in case["itypes"]]
    signature = gen_signature(itypes, type_node.contin_type)
    return representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree(case["source"]),
        signature,
        linear_contin=case["linear_contin"],
    )


@pytest.mark.oracle
def test_contin_representation_fixture_schema_is_bounded_and_non_vacuous(golden):
    assert set(golden) == {"stepsize", "expansion", "depth", "cases"}
    # C++: representation.cc's own module statics -- this fixture's whole
    # point is to measure at classic's REAL defaults, never a probe-chosen
    # substitute for them.
    assert (golden["stepsize"], golden["expansion"], golden["depth"]) == (1.0, 2.0, 5)
    assert len(golden["cases"]) == 18
    assert all(set(case) == _CASE_KEYS for case in golden["cases"])
    assert all(
        set(knob) == _KNOB_KEYS for case in golden["cases"] for knob in case["contin_knobs"]
    )
    # Every case is a PURE contin-output exemplar: zero disc knobs, at least
    # one contin knob (a vacuous "0 knobs either side" fixture could not
    # discriminate this task's own code from a no-op).
    assert all(case["disc_knob_count"] == 0 for case in golden["cases"])
    assert all(len(case["contin_knobs"]) > 0 for case in golden["cases"])
    # Half the corpus is the real CLI default -- see this module's docstring.
    assert sum(not case["linear_contin"] for case in golden["cases"]) == 9
    assert sum(case["linear_contin"] for case in golden["cases"]) == 9
    # The corpus discriminates linear from non-linear knob building: at
    # least one case has strictly more knobs when non-linear (proving
    # `rec_canonize`'s `!_linear_contin` branch is actually exercised, not
    # merely present in the fixture unread).
    by_name = {case["name"]: case for case in golden["cases"]}
    assert (
        len(by_name["sum-two-args-nonlinear"]["contin_knobs"])
        > len(by_name["sum-two-args-linear"]["contin_knobs"])
    )


@pytest.mark.oracle
@pytest.mark.parametrize("case_idx", range(18))
def test_all_contin_exemplars_match_classic_exemplar_prototype_and_knob_specs(
    golden, case_idx
):
    """Runs classic's OWN binary output (half at ``linear_contin=False``,
    classic's real CLI default) against this port -- not a probe this port
    configured. Compares the FULL decorated-exemplar text, the FULL
    ``ostream_prototype`` string (which encodes tree structure AND every
    knob's placement together), and each contin knob's (mean, step_size,
    expansion, depth) IN ORDER -- count, order, and spec, per the exit
    criterion.
    """
    case = golden["cases"][case_idx]
    rep = _make_rep(case)

    assert combo_tree_to_str(rep.exemplar()) == case["decorated_exemplar"]
    assert rep.ostream_prototype() == case["prototype"]
    assert len(rep.disc) == case["disc_knob_count"]
    preorder = {node: idx for idx, node in enumerate(rep._exemplar)}
    assert [
        {
            "order": order,
            "mean": entry.spec.mean,
            "step_size": entry.spec.step_size,
            "expansion": entry.spec.expansion,
            "depth": entry.spec.depth,
            "text": entry.knob.toStr(),
            "location_preorder": preorder[entry.knob.get_loc()],
            "location_subtree": combo_tree_to_str(combo_tree(entry.knob.get_loc())),
        }
        for order, entry in enumerate(rep.contin)
    ] == case["contin_knobs"]


@pytest.mark.oracle
def test_linear_contin_false_is_classicss_real_cli_default_not_just_a_probe_config(golden):
    """Names the specific fixture rows that exercise classic's real CLI
    default directly, so a future reader does not have to re-derive which
    half of the corpus this is from the raw JSON.
    """
    nonlinear_cases = [case for case in golden["cases"] if not case["linear_contin"]]
    assert len(nonlinear_cases) == 9
    for case in nonlinear_cases:
        rep = _make_rep(case)
        assert combo_tree_to_str(rep.exemplar()) == case["decorated_exemplar"]
        assert rep.ostream_prototype() == case["prototype"]


def test_representation_linear_contin_defaults_to_classics_own_constructor_default():
    """DIVERGENCE(C++ moses/moses/representation/representation.h:61):
    pinned directly, not just described in a docstring -- classic's own
    constructor DEFAULT is `true` (not the CLI's `false`, which is a later
    layer's decision this module does not make). Mutation: flip the
    default to `False` in ``representation.__init__`` and this goes red
    without ``linear_contin`` ever being passed here.
    """
    rep = representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 1),
    )
    linear_rep = representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 1),
        linear_contin=True,
    )
    assert combo_tree_to_str(rep.exemplar()) == combo_tree_to_str(linear_rep.exemplar())
    assert len(rep.contin) == len(linear_rep.contin)


def test_module_statics_match_classics_own_representation_cc_defaults():
    """C++: moses/moses/representation/representation.cc:59-61 (`stepsize
    = 1.0`, `expansion = 2.0`, `depth = 5`). These are what every
    representation build reads UNLESS ``set_stepsize``/``set_expansion``/
    ``set_depth`` has been called first -- pinned directly against a fresh
    contin knob's own spec, not merely against the module attributes
    (which a broken accessor could satisfy vacuously).
    """
    assert representation_module._stepsize == 1.0
    assert representation_module._expansion == 2.0
    assert representation_module._depth == 5

    rep = representation(
        lambda _tr: None,
        lambda _tr: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 1),
        linear_contin=True,
    )
    spec = rep.contin[0].spec
    assert (spec.step_size, spec.expansion, spec.depth) == (1.0, 2.0, 5)


def test_set_stepsize_expansion_depth_change_subsequent_field_specs():
    """Proves the setters are actually load-bearing, not merely present:
    changing them changes the NEXT representation's contin field specs,
    and restoring them changes it back.
    """
    try:
        representation_module.set_stepsize(0.5)
        representation_module.set_expansion(3.0)
        representation_module.set_depth(2)
        rep = representation(
            lambda _tr: None,
            lambda _tr: None,
            str_to_combo_tree("$1"),
            gen_signature(type_node.contin_type, 1),
            linear_contin=True,
        )
        spec = rep.contin[0].spec
        assert (spec.step_size, spec.expansion, spec.depth) == (0.5, 3.0, 2)
        assert rep.fields().n_contin_fields() == 2 * len(rep.contin)
    finally:
        representation_module.set_stepsize(1.0)
        representation_module.set_expansion(2.0)
        representation_module.set_depth(5)


def test_a_predicate_anywhere_in_a_contin_exemplar_is_still_rejected():
    """The mixed logical-predicate branches (build_knobs.cc's
    `is_predicate` cases inside `logical_canonize`/`build_logical`) remain
    unported for BOTH output directions -- this pins that a
    `greater_than_zero` predicate is rejected even nested under `impulse`
    inside an otherwise-supported contin exemplar, not merely at top level.
    """
    with pytest.raises(ValueError, match="unsupported.*predicate"):
        representation(
            lambda _tr: None,
            lambda _tr: None,
            str_to_combo_tree("impulse(0<($1))"),
            gen_signature([type_node.contin_type], type_node.contin_type),
        )


def _bare_contin_representation(tr, knobs):
    """Builds a `representation` directly from a hand-decorated exemplar and
    a fixed list of `contin_knob`s, bypassing `build_knobs` entirely --
    same technique test_representation.py's own
    `test_candidate_recursion_descends_into_a_composite_knob_subtree` uses
    for a composite DISC knob, applied here to contin knobs instead.
    """
    rep = representation.__new__(representation)
    knob_mapper.__init__(rep)
    rep._exemplar = tr
    rep._simplify_candidate = lambda _tr: None
    rep._simplify_knob_building = lambda _tr: None
    for kb in knobs:
        rep.add_contin_knob(kb)
    rep.finalize()
    rep._fields = field_set([entry.spec for entry in rep.contin])
    rep._it_disc_idx = {}
    start = rep._fields.begin_contin_raw_idx()
    rep._it_contin_idx = {
        entry.knob.get_loc(): start + offset for offset, entry in enumerate(rep.contin)
    }
    return rep


def test_get_candidate_turns_each_contin_knob_through_field_set_independently():
    """Pins `representation._get_candidate_rec`'s contin branch end to end:
    an all-zero instance (every raw two-bit field at ``Stop``) reproduces
    each knob's own MEAN unchanged; setting one knob's raw fields away from
    its mean changes ONLY that knob's leaf in the rebuilt candidate, and the
    result is a real, evaluable combo tree (checked with the S14 Task 1
    `contin_interpreter`, not just string-compared).
    """
    tr = str_to_combo_tree("plus(0 1)")
    first_leaf, second_leaf = list(tr.children(tr.begin()))
    kb0 = contin_knob(tr, first_leaf, step_size=1.0, expansion=2.0, depth=4)
    kb1 = contin_knob(tr, second_leaf, step_size=1.0, expansion=2.0, depth=4)
    rep = _bare_contin_representation(tr, [kb0, kb1])
    assert len(rep.contin) == 2

    inst = instance(rep.fields().packed_width())
    baseline = rep.get_candidate(inst, False)
    assert combo_tree_to_str(baseline) == "+(0 1) "
    assert contin_interpreter([])(baseline) == 1.0

    index_of = {entry.knob: idx for idx, entry in enumerate(rep.contin)}
    rep.fields().set_contin(inst, index_of[kb0], 5.0)
    turned = rep.get_candidate(inst, False)
    assert combo_tree_to_str(turned) == "+(5 1) "
    assert contin_interpreter([])(turned) == 6.0
    # kb1's own leaf is untouched by turning kb0.
    assert rep.fields().get_contin(inst, index_of[kb1]) == 1.0

    rep.fields().set_contin(inst, index_of[kb1], -2.5)
    achieved = rep.fields().get_contin(inst, index_of[kb1])
    # depth=4 cannot represent -2.5 exactly from a mean of 1.0 at these
    # step/expansion settings; assert against what `set_contin` actually
    # achieved rather than the unreachable exact target.
    assert achieved != 1.0
    both_turned = rep.get_candidate(inst, False)
    assert combo_tree_to_str(both_turned) == f"+(5 {achieved:g}) "
    assert contin_interpreter([])(both_turned) == 5.0 + achieved

    # get_candidate never mutates the exemplar (representation.cc:313's own
    # "does not modify the rep at all").
    assert combo_tree_to_str(rep.exemplar()) == "+(0 1) "
