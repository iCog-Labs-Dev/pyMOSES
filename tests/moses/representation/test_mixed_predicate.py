"""C++ oracle tests for the MIXED knob-building path -- a Boolean-output
representation over contin (or mixed contin/Boolean) inputs.

The gap this closes, named in docs/REMAINING-WORK.md Sec 1 as "the largest
remaining functional gap": ``build_logical``'s ``is_predicate(cur)`` branch
(build_knobs.cc:640-663) and ``logical_canonize``'s (build_knobs.cc:160-172),
plus the contin branches of ``insert_typed_arg`` (build_knobs.cc:386-393)
and ``sample_logical_perms`` (build_knobs.cc:425-430). Its shadowed twin --
``simplify_predicates``' ``contin_reduce``/``mixed_reduce`` tail
(logical_rules.cc:76-81) -- is covered by the second half of the same
fixture, because ``disc_probe`` routes every knob probe through exactly that
rule and so the two cannot be separated on this path anyway.

Both halves come from ``tests/fixtures/mixed_predicate.json``, captured by
``tools/capture_mixed_predicate.py`` from
``tools/cpp_probe/mixed_predicate_probe.cc``, which drives the REAL classic
``representation`` and ``logical_reduce`` (never a reimplementation) at
classic's own module statics (stepsize=1.0, expansion=2.0, depth=5) and at
``linear_contin=False``, the real ``moses`` CLI default
(``--linear-regression``, problem-params.cc:292).

Half the representation corpus runs at ``perm_ratio=0`` -- classic's own
constructor AND CLI default (build_knobs.h:56, problem-params.cc:299) --
which is the only regime that runs ``sample_logical_perms``' pair loop, and
therefore the only one that calls ``insert_typed_arg`` at all. Those cases
consume ``randGen()`` draws, so each reseeds the singleton to the fixture's
own recorded seed first, exactly as the probe does.
"""

import io
import json
from contextlib import redirect_stdout

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduce, logical_reduction
from pymetta.comboreduct.reduct.mixed_reduction import mixed_reduction
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.representation.build_knobs import build_knobs
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.representation import representation
from pymetta.util.rng import randGen

_TYPE_NAME = {
    "boolean": type_node.boolean_type,
    "contin": type_node.contin_type,
}

_TOP_KEYS = {
    "stepsize",
    "expansion",
    "depth",
    "linear_contin",
    "reduct_effort",
    "cases",
    "logical_reduce",
}
_CASE_KEYS = {
    "name",
    "otype",
    "source",
    "itypes",
    "ignored_arguments",
    "ignore_greater_than_zero",
    "perm_ratio",
    "seed",
    "decorated_exemplar",
    "prototype",
    "disc_knobs",
    "contin_knobs",
}
_DISC_KEYS = {
    "order",
    "multy",
    "text",
    "in_exemplar",
    "location_preorder",
    "location_subtree",
}
_CONTIN_KEYS = {
    "order",
    "mean",
    "step_size",
    "expansion",
    "depth",
    "text",
    "location_preorder",
    "location_subtree",
}
_REDUCE_KEYS = {"input", "effort", "ok", "reduced", "size_before", "size_after"}

_CASE_COUNT = 40
_REDUCE_COUNT = 40


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "mixed_predicate.json").read_text())


def _ignore_ops(case):
    ops = {argument(idx) for idx in case["ignored_arguments"]}
    if case["ignore_greater_than_zero"]:
        ops.add(builtin.greater_than_zero)
    return ops


def _make_rep(case):
    itypes = [_TYPE_NAME[name] for name in case["itypes"]]
    otype = _TYPE_NAME[case["otype"]]
    signature = gen_signature(itypes, otype)
    ignore_ops = _ignore_ops(case)
    # For a Boolean output: the SAME rule object in both slots, exactly as
    # the probe does and as problem-params.cc hands a Boolean-output
    # problem. It matters -- this is what routes `disc_probe` through
    # `simplify_predicates`' own contin/mixed tail rather than through a
    # no-op lambda. For a contin output: `mixed_reduction()`, which the
    # contin path threads through but never invokes (no `disc_probe`).
    if otype is type_node.boolean_type:
        simplifier = logical_reduction(ignore_ops)(2)
    else:
        simplifier = mixed_reduction()
    randGen().seed(case["seed"])
    return representation(
        simplifier,
        simplifier,
        str_to_combo_tree(case["source"]),
        signature,
        ignore_ops=ignore_ops,
        perm_ratio=case["perm_ratio"],
        linear_contin=False,
    )


@pytest.mark.oracle
def test_mixed_predicate_fixture_schema_is_bounded_and_non_vacuous(golden):
    assert set(golden) == _TOP_KEYS
    assert (golden["stepsize"], golden["expansion"], golden["depth"]) == (1.0, 2.0, 5)
    assert golden["linear_contin"] is False
    assert len(golden["cases"]) == _CASE_COUNT
    assert len(golden["logical_reduce"]) == _REDUCE_COUNT
    assert all(set(case) == _CASE_KEYS for case in golden["cases"])
    assert all(set(knob) == _DISC_KEYS for case in golden["cases"] for knob in case["disc_knobs"])
    assert all(
        set(knob) == _CONTIN_KEYS for case in golden["cases"] for knob in case["contin_knobs"]
    )
    assert all(set(row) == _REDUCE_KEYS for row in golden["logical_reduce"])

    boolean_cases = [case for case in golden["cases"] if case["otype"] == "boolean"]
    contin_cases = [case for case in golden["cases"] if case["otype"] == "contin"]
    assert len(boolean_cases) == 34
    assert len(contin_cases) == 6

    # The one property that makes this corpus discriminate the mixed path
    # from the pure-Boolean one: a BOOLEAN-output representation that also
    # carries contin knobs. Without build_logical's is_predicate branch
    # there would be none.
    with_contin = [case for case in boolean_cases if case["contin_knobs"]]
    assert len(with_contin) == 22
    assert all(case["disc_knobs"] for case in with_contin)

    # The contin-output half pins the OTHER route to a predicate --
    # rec_canonize's `impulse` arm -- and, just as importantly, pins that
    # classic builds NO disc knobs there: the contin path never calls
    # build_logical at all.
    assert all(case["contin_knobs"] and not case["disc_knobs"] for case in contin_cases)

    # And the pair regime is really exercised: at perm_ratio 0 an arity>=2
    # case must have strictly more disc knobs than the same shape at -1,
    # which is the only way `insert_typed_arg` gets called at all.
    by_name = {case["name"]: case for case in golden["cases"]}
    assert len(by_name["bare-and-2contin-pairs"]["disc_knobs"]) > len(
        by_name["bare-and-2contin-nopairs"]["disc_knobs"]
    )


@pytest.mark.oracle
@pytest.mark.parametrize("case_idx", range(_CASE_COUNT))
def test_mixed_exemplars_match_classic_exemplar_prototype_and_knob_specs(golden, case_idx):
    """Compares the FULL decorated-exemplar text, the FULL
    ``ostream_prototype`` string (which encodes tree structure AND every
    knob's placement together), every disc knob's (multy, text,
    in_exemplar, location) IN ORDER, and every contin knob's (mean,
    step_size, expansion, depth, location) IN ORDER -- count, order and
    spec, on both sides of the mixed boundary.
    """
    case = golden["cases"][case_idx]
    rep = _make_rep(case)

    assert combo_tree_to_str(rep.exemplar()) == case["decorated_exemplar"]
    assert rep.ostream_prototype() == case["prototype"]

    preorder = {node: idx for idx, node in enumerate(rep._exemplar)}
    assert [
        {
            "order": order,
            "multy": entry.spec.multy,
            "text": entry.knob.toStr(),
            "in_exemplar": entry.knob.in_exemplar(),
            "location_preorder": preorder[entry.knob.get_loc()],
            "location_subtree": combo_tree_to_str(combo_tree(entry.knob.get_loc())),
        }
        for order, entry in enumerate(rep.disc)
    ] == case["disc_knobs"]
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
@pytest.mark.parametrize("row_idx", range(_REDUCE_COUNT))
def test_logical_reduce_of_predicates_matches_classic(golden, row_idx):
    """``simplify_predicates``' own contin/mixed tail, through the public
    ``logical_reduce`` entry point at both efforts that reach ``p_medium``.

    Mutation-checked: deleting the ``mixed_reduce(tr, it)`` call in
    ``simplify_predicates.apply`` leaves ``0<(1)`` as ``0<(1)`` where
    classic gives ``true``; deleting the ``contin_reduce`` call leaves
    ``0<(*(2 $1))`` unreduced where classic gives ``0<($1)``.
    """
    row = golden["logical_reduce"][row_idx]
    tr = str_to_combo_tree(row["input"])
    assert tr.size() == row["size_before"]
    logical_reduce(row["effort"], tr, ignore_ops=set())
    assert combo_tree_to_str(tr) == row["reduced"]
    assert tr.size() == row["size_after"]


def test_boolean_output_rejects_a_contin_typed_argument_outside_a_predicate():
    """DIVERGENCE guard (build_knobs.py's own `_validate_boolean_node`):
    classic asserts nothing here, but a bare contin-typed ``$1`` sitting at
    the logical level is not a Boolean expression, and letting it through
    would silently build a Boolean knob over a contin term.
    """
    signature = gen_signature([type_node.contin_type], type_node.boolean_type)
    with pytest.raises(ValueError, match="contin-typed"):
        build_knobs(
            str_to_combo_tree("$1"),
            signature,
            knob_mapper(),
            lambda _tr: None,
        )


def test_boolean_output_rejects_a_logical_not_that_does_not_precede_a_predicate():
    """build_knobs.cc:608-611's own OC_ASSERT, hoisted to construction time:
    "the tree is supposed to be in normal form; and thus must not contain
    logical_not nodes" -- except immediately above a predicate.
    """
    signature = gen_signature([type_node.boolean_type], type_node.boolean_type)
    with pytest.raises(ValueError, match="logical_not"):
        build_knobs(
            str_to_combo_tree("and(not($1))"),
            signature,
            knob_mapper(),
            lambda _tr: None,
        )


# ---------------------------------------------------------------------------
# CLI-level oracles. The unit-level fixture above pins knob COUNT, ORDER and
# SPEC; these pin what the user actually sees, byte for byte, for a whole
# contin-input/Boolean-output run.
#
# Each expected block below was captured by ACTUALLY RUNNING
# `$CLASSIC/build/moses/moses/main/moses` with exactly the argv quoted in
# the test, bounded per this project's OOM rule:
#     ( ulimit -v 2097152; timeout 300 <cmd> )
# Nothing here shells out at test time -- `tools/ci.sh` must not require a
# C++ toolchain (tools/_capture_env.py's own module docstring).
# ---------------------------------------------------------------------------

_MIX2_CSV = """x1,x2,y
-1.06,-2.09,0
0.91,-2.57,0
0.22,-0.81,0
-2.65,0.04,0
-2.78,-0.4,0
-2.58,-2.46,0
-0.45,1.96,1
-2.26,-1.66,0
0.76,2.69,1
0.46,-0.62,0
2.86,-2.72,1
2.15,-1.26,1
-2.13,-2.29,0
-1.15,1.9,1
-1.92,0.49,0
0.83,-0.77,1
0.29,-2.62,0
-2.64,-1.76,0
1.08,-0.43,1
-1.12,0.51,0
-0.28,-1.2,0
1.77,1.19,1
-1.54,0.45,0
0.15,2.25,1
"""


def _run_cli(argv):
    from pymetta.moses.main.moses_exec import moses_exec

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = moses_exec(["moses", *argv])
    return rc, buffer.getvalue()


@pytest.mark.oracle
def test_two_contin_columns_boolean_target_matches_classic_byte_for_byte(tmp_path):
    """docs/REMAINING-WORK.md Sec 1's own measured reproducer, inverted.

    It recorded that a two-contin-column table with a Boolean target makes
    classic print ``0 0<(+($x1 $x2))`` at a perfect score while this port
    raised ``NotImplementedError`` at ``build_knobs.py:275``. Classic's
    whole output for

        moses -i mix2.csv -u y -m 1000 -r 1 -W1

    is below, and the port now reproduces it byte for byte.
    """
    csv_path = tmp_path / "mix2.csv"
    csv_path.write_text(_MIX2_CSV)
    rc, out = _run_cli(["-i", str(csv_path), "-u", "y", "-m", "1000", "-r", "1", "-W1"])
    assert rc == 0
    assert out == (
        "0 0<(+($x1 $x2)) \n"
        "-6 0<($x1) \n"
        "-6 and(0<($x1) 0<($x2)) \n"
        "-8 0<($x2) \n"
        "-8 or(0<($x1) 0<($x2)) \n"
        "-8 or(and(not(0<($x1)) 0<($x2)) 0<($x1)) \n"
        "-9 and(not(0<($x2)) 0<($x1)) \n"
        "-11 and(not(0<($x1)) 0<($x2)) \n"
        "-13 or(not(0<($x2)) 0<($x1)) \n"
        "-13 or(and(not(0<($x1)) not(0<($x2))) 0<($x1)) \n"
    )


@pytest.mark.oracle
def test_two_contin_columns_at_a_small_budget_matches_classic_byte_for_byte(tmp_path):
    """The same table at ``-m 200`` -- a budget that stops mid-search, so
    the whole metapopulation (not just a converged winner) is compared:

        moses -i mix2.csv -u y -m 200 -r 1 -W1
    """
    csv_path = tmp_path / "mix2.csv"
    csv_path.write_text(_MIX2_CSV)
    rc, out = _run_cli(["-i", str(csv_path), "-u", "y", "-m", "200", "-r", "1", "-W1"])
    assert rc == 0
    assert out == (
        "-6 0<($x1) \n"
        "-6 0<(+(1 $x1)) \n"
        "-6 and(0<($x1) 0<($x2)) \n"
        "-7 0<(+(*(-1 $x1 $x2) $x1)) \n"
        "-8 0<($x2) \n"
        "-8 or(0<($x1) 0<($x2)) \n"
        "-8 or(and(not(0<($x1)) 0<($x2)) 0<($x1)) \n"
        "-9 and(not(0<($x1)) 0<($x1)) \n"
        "-9 and(not(0<($x2)) 0<($x1)) \n"
        "-10 0<(*(+(1 *(-1 $x1)) $x1)) \n"
    )


@pytest.mark.oracle
def test_select_over_a_contin_table_matches_classic_byte_for_byte():
    """``-H select`` is a Boolean-output problem over a contin-input table
    -- the exact shape ``tests/moses/main/test_table_problems.py`` used to
    assert raised. Classic, for

        moses -i tests/fixtures/table_data/select_regression.csv \\
              -u y -m 30 -q 0.2 -w 0.8 -Hselect
    """
    rc, out = _run_cli(
        [
            "-i",
            "tests/fixtures/table_data/select_regression.csv",
            "-u",
            "y",
            "-m",
            "30",
            "-q",
            "0.2",
            "-w",
            "0.8",
            "-Hselect",
        ]
    )
    assert rc == 0
    assert out == (
        "-15 true \n"
        "-15 0<($1) \n"
        "-15 0<($2) \n"
        "-15 and(0<($1) 0<($2)) \n"
        "-15 or(0<($1) 0<($2)) \n"
        "-15 or(not(0<($1)) 0<($2)) \n"
        "-15 or(not(0<($2)) 0<($1)) \n"
    )
