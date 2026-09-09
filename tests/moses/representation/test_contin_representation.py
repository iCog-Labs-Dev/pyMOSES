"""C++ oracle tests for the contin-output representation path (S14 Task 2).

The exit criterion is "the knobs built for a contin exemplar match C++ in
count, order and spec" -- measured against ``tests/fixtures/
contin_representation.json``, captured by ``tools/capture_contin_
representation.py`` from ``tools/cpp_probe/contin_representation_probe.cc``,
which drives the REAL classic ``representation`` C++ class (never a
reimplementation) at its own real module-static defaults (``stepsize=1.0,
expansion=2.0, depth=5``, ``representation.cc:59-61``).

Half of the fixture's 16 cases (the ``-nonlinear`` ones) run classic at
``linear_contin=False`` -- the real ``moses`` CLI's own default
(``--linear-regression``, problem-params.cc:292), NOT merely this
constructor's own literal default (``True``, matching representation.h:62
verbatim -- see CARRY-FORWARD.md Sec 5.4, "an oracle configured by the port
cannot detect the port's configuration being wrong"). Every one of those
cases is diffed against classic's OWN compiled binary output, not against a
probe this port configured -- see ``test_all_...`` below.
"""

import json

import pytest

import sys

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
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
    assert len(golden["cases"]) == 16
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
    assert sum(not case["linear_contin"] for case in golden["cases"]) == 8
    assert sum(case["linear_contin"] for case in golden["cases"]) == 8
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
@pytest.mark.parametrize("case_idx", range(16))
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
    assert len(nonlinear_cases) == 8
    for case in nonlinear_cases:
        rep = _make_rep(case)
        assert combo_tree_to_str(rep.exemplar()) == case["decorated_exemplar"]
        assert rep.ostream_prototype() == case["prototype"]


def test_representation_linear_contin_defaults_to_classics_own_constructor_default():
    """DIVERGENCE(C++ moses/moses/representation/representation.h:62):
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
