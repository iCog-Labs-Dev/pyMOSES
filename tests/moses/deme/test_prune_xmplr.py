"""`deme_expander.prune_xmplr` / `.fs_to_names` / `_type_to_exemplar`.

C++ oracle
==========

Every expectation in `_ORACLE` below was produced by the REAL
`opencog::moses::deme_expander::prune_xmplr` and
`::fs_to_names`, driven through `tools/cpp_probe/prune_xmplr_probe.cc`
(both members are `protected` -- deme_expander.h:85's `protected:` section
covers h:115 and h:128 -- so the probe re-exposes them through a derived
struct). Build and run, verbatim::

    P=$HOME/opt/moses-legacy
    C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
    g++ -O2 -std=c++17 -I $C -I $P/include \\
        -o tools/cpp_probe/build/prune_xmplr_probe \\
        tools/cpp_probe/prune_xmplr_probe.cc \\
        -L $P/lib -L $P/lib/opencog \\
        -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \\
        -lmoses -lcomboreduct -lcogutil

    for i in $(seq 0 23); do
      ( ulimit -v 2097152; timeout 60 \\
        ./tools/cpp_probe/build/prune_xmplr_probe \\
        tests/fixtures/table_data/trainingBoolean.dat y 3 "$i" )
      echo "rc=$?"
    done

The corpus is hardcoded INSIDE the .cc rather than passed on argv, and the
loop above passes only an integer index: every exemplar contains `$1`/`$2`/
`$3`, and CARRY-FORWARD.md Sec 5.10 records three separate false readings
in this project caused by a shell expanding exactly those tokens.

The results are embedded here as a literal rather than written to
`tests/fixtures/`: this task's editable scope is `pymoses/moses/deme/
deme_expander.py` plus NEW files under `tests/moses/deme/` and `tools/`,
and a fixture + schema-2 manifest under `tests/fixtures/` is neither.
Everything needed to regenerate and re-check it is in the two commands
above.

Six of the 24 cases SEGFAULT classic (exit 139)
===============================================

`prune_xmplr` can hand `simplify_knob_building` a `not` with zero
children -- prune the only child of a `not` and that is what is left --
and classic's `logical_reduce` dereferences it unchecked. Measured with a
standalone one-tree reducer over the same `logical_reduction(2)`::

    not()                -> SIGSEGV
    or(and($2) not())    -> SIGSEGV
    and()                -> "true"
    or()                 -> "true"
    and($2)              -> "$2"
    not($1)              -> "!$1"

so it is the childless `not` specifically, not degenerate operators in
general. Reachable from the CLI via `--fs-prune-exemplar 1` on any
exemplar with a negated feature that feature selection drops. A Python
port cannot reproduce a segfault; this one raises `AttributeError` out of
`logical_reduction` instead, which is the same "loud where C++ is
undefined" trade `tree.py`'s `_unlink` DIVERGENCE makes. Those cases are
asserted below as raising, so a future "fix" that makes them succeed
fails a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.moses.deme.deme_expander import _type_to_exemplar, deme_expander
from pymoses.moses.deme.deme_params import deme_parameters
from pymoses.moses.deme.feature_selector import feature_selector, feature_selector_parameters
from pymoses.moses.metapopulation.metapop_params import subsample_deme_filter_parameters
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import ctruth_table_bscore
from pymoses.util.logger import Logger, logger
from pymoses.util.rng import randGen

_TABLE_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_DATASET = "trainingBoolean.dat"
_TARGET = "y"
_ARITY = 3
_ILABELS = ["x1", "x2", "x3"]

# (case index, exemplar, selected features, pruned tree, fs_to_names)
# -- straight out of the probe runs described in this module's docstring.
_ORACLE: list[tuple[int, str, set[int], str, list[str]]] = [
    (0, "and($1 $2 $3)", {0, 2}, "and($1 $3)", ["x1", "x3"]),
    (1, "and($1 $2 $3)", set(), "true", []),
    (2, "and($1 $2 $3)", {0, 1, 2}, "and($1 $2 $3)", ["x1", "x2", "x3"]),
    (3, "or(and($1 $2) $3)", set(), "true", []),
    (4, "or(and($1 $2) $3)", {2}, "$3", ["x3"]),
    # THE erase-successor case: `$2` is the `and`'s LAST child, and `$3`
    # comes after it in pre-order and must ALSO go. A sibling-successor
    # erase returns None there, ends the walk, and leaves `or($1 $3)`.
    (5, "or(and($1 $2) $3)", {0}, "$1", ["x1"]),
    # The `type_to_exemplar` seed: pruning the ROOT empties the tree.
    (10, "$1", set(), "and", []),
    (11, "$1", {0}, "$1", ["x1"]),
    (13, "or($1 and($2 or($3 $1)))", {0}, "$1", ["x1"]),
    (14, "or($1 and($2 or($3 $1)))", {2}, "$3", ["x3"]),
    (15, "or($1 and($2 or($3 $1)))", set(), "true", []),
    (16, "and(or($1 $2) or($2 $3))", {1}, "$2", ["x2"]),
    (17, "and($1 $1 $2)", {0}, "$1", ["x1"]),
    (18, "and($1 $1 $2)", {1}, "$2", ["x2"]),
    (19, "or(and($1 $2 $3) and($1 $2 $3))", set(), "true", []),
    (21, "not($1)", {0}, "!$1", ["x1"]),
    (22, "and(not($1) not($2))", {0, 1}, "and(!$1 !$2)", ["x1", "x2"]),
    (23, "or(and($1 $2) $3)", {0, 1, 2}, "or(and($1 $2) $3)", ["x1", "x2", "x3"]),
]

# Cases where classic exits 139 (SIGSEGV) -- see this module's docstring.
_CLASSIC_SEGFAULTS: list[tuple[int, str, set[int]]] = [
    (6, "or(and($1 $2) not($3))", {1}),
    (7, "and(not($1) $2)", set()),
    (8, "and(not($1) $2)", {1}),
    (9, "and(not($1) not($2))", {0}),
    (12, "not($1)", set()),
    (20, "and(not($1) or($2 not($3)))", {2}),
]


def _dex(arity: int = _ARITY) -> deme_expander:
    """The same construction `tools/cpp_probe/prune_xmplr_probe.cc` uses:
    real table -> CTable -> ctruth_table_bscore -> behave_cscore, real
    `logical_reduction(())(2)` for BOTH reduction slots, a real
    `hill_climbing` injected only because the constructor demands an
    optimizer reference (`prune_xmplr` never touches it).
    """
    table = loadTable(str(_TABLE_DATA_DIR / _DATASET), _TARGET)
    cscorer = behave_cscore(ctruth_table_bscore(table.compressed()))
    rule = logical_reduction(())(2)
    params = deme_parameters()
    params.perm_ratio = -1.0
    hc = hill_climbing(optim_parameters(), hc_parameters(widen=False, step=False, cross=False))
    sig = gen_signature(type_node.boolean_type, arity)
    return deme_expander(sig, rule, rule, cscorer, hc, params, subsample_deme_filter_parameters())


def _ids(cases):
    return [f"c{c[0]}" for c in cases]


# ---------------------------------------------------------------------------
# prune_xmplr, matched against the C++ itself
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case", "exemplar", "selected", "expected", "_names"), _ORACLE, ids=_ids(_ORACLE)
)
def test_prune_xmplr_matches_cpp(case, exemplar, selected, expected, _names):
    del case, _names
    dex = _dex()
    got = dex.prune_xmplr(str_to_combo_tree(exemplar), selected)
    # combo_tree_to_str's trailing-space convention (iostream_combo.py) vs.
    # classic's plain `out << combo_tree`: stripped on both sides, exactly as
    # test_deme_expander.py already does, so it cannot mask a real divergence.
    assert combo_tree_to_str(got).rstrip() == expected


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case", "exemplar", "selected", "expected", "_names"), _ORACLE, ids=_ids(_ORACLE)
)
def test_prune_xmplr_never_returns_an_empty_tree(case, exemplar, selected, expected, _names):
    """`pruned_empty` was `false` for every one of the 18 non-crashing
    oracle cases -- including case 10, where the walk DOES empty the tree
    and the `type_to_exemplar` seed puts a root back. Asserted separately
    from the rendering above because an empty tree renders as `""`, which
    would only be caught by a case whose expected text is also `""` --
    there is none.
    """
    del case, expected, _names
    dex = _dex()
    assert dex.prune_xmplr(str_to_combo_tree(exemplar), selected).empty() is False


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case", "exemplar", "selected", "_expected", "names"), _ORACLE, ids=_ids(_ORACLE)
)
def test_fs_to_names_matches_cpp(case, exemplar, selected, _expected, names):
    del case, exemplar, _expected
    assert _dex().fs_to_names(selected, _ILABELS) == names


@pytest.mark.parametrize(
    ("case", "exemplar", "selected"), _CLASSIC_SEGFAULTS, ids=_ids(_CLASSIC_SEGFAULTS)
)
def test_prune_xmplr_raises_where_classic_segfaults(case, exemplar, selected):
    """Classic exits 139 on these (childless `not` reaching
    `logical_reduce`); the port raises instead. DO NOT "fix" this by
    guarding childless `not` in `prune_xmplr` -- classic has no such guard,
    and succeeding where classic crashes is a divergence in the direction
    that hides bugs.
    """
    del case
    dex = _dex()
    with pytest.raises(AttributeError):
        dex.prune_xmplr(str_to_combo_tree(exemplar), selected)


def test_prune_xmplr_does_not_mutate_the_caller_s_exemplar():
    """`combo_tree res = xmplr;` is a VALUE copy (deme_expander.cc:311).
    The probe dumps the exemplar again after the call ("xmplr_after") and
    it is unchanged in all 18 non-crashing cases. It matters:
    `create_representations` hands the SAME `exemplar` object to
    `prune_xmplr` once per entry in `pop_of_selected_feats`, so an
    in-place prune would corrupt every deme after the first.
    """
    dex = _dex()
    xmplr = str_to_combo_tree("or(and($1 $2) $3)")
    for selected in ({0}, {2}, set(), {0, 1, 2}):
        dex.prune_xmplr(xmplr, selected)
        assert combo_tree_to_str(xmplr).rstrip() == "or(and($1 $2) $3)"


def test_prune_xmplr_result_is_independent_of_the_exemplar():
    """The copy is DEEP: mutating the returned tree must not reach back
    into the caller's exemplar. `tree(iterator_base)` copies descendants
    (pymoses/util/tree.py), but a shallower "share the root node" copy
    would still pass every rendering assertion above.
    """
    dex = _dex()
    xmplr = str_to_combo_tree("and($1 $2 $3)")
    pruned = dex.prune_xmplr(xmplr, {0, 1, 2})
    assert combo_tree_to_str(pruned).rstrip() == "and($1 $2 $3)"
    pruned.erase(pruned.begin().first_child)
    assert combo_tree_to_str(xmplr).rstrip() == "and($1 $2 $3)"


def test_prune_xmplr_accepts_an_empty_exemplar():
    """`combo_tree()` is a legal (if useless) input: `xmplr.begin()` is
    `None`, the copy is empty, the walk runs zero times, `rule.__call__`
    is a no-op on an empty tree (reduct.h:46's own `if (!tr.empty())`),
    and the seed fallback fires. Not in the C++ corpus because
    `str2combo_tree("")` cannot express it.
    """
    got = _dex().prune_xmplr(combo_tree(), {0, 1, 2})
    assert combo_tree_to_str(got).rstrip() == "and"


def test_prune_xmplr_seeds_from_the_signature_output_type():
    """The seed is `type_to_exemplar(get_type_node(get_signature_output(
    _type_sig)))` -- it reads the deme_expander's OWN signature, not a
    hardcoded boolean. A contin-output signature seeds `plus`.
    """
    table = loadTable(str(_TABLE_DATA_DIR / _DATASET), _TARGET)
    cscorer = behave_cscore(ctruth_table_bscore(table.compressed()))
    rule = logical_reduction(())(2)
    params = deme_parameters()
    params.perm_ratio = -1.0
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.contin_type, _ARITY)
    dex = deme_expander(sig, rule, rule, cscorer, hc, params)

    got = dex.prune_xmplr(str_to_combo_tree("$1"), set())
    # `builtin.plus` renders as "+", not "plus" -- compare the vertex.
    assert got.size() == 1
    assert got.begin().data == builtin.plus


# ---------------------------------------------------------------------------
# fs_to_names: ORDER is the whole content of the function
# ---------------------------------------------------------------------------


def test_fs_to_names_sorts_ascending_where_a_python_set_does_not():
    """A C++ `std::set<arity_t>` iterates ASCENDING; a Python `set`
    iterates in hash order. `{3, 10}` is a concrete counterexample --
    `list({3, 10})` is `[10, 3]` in CPython -- so an unsorted
    transliteration returns the labels backwards. Small index sets
    (`{0, 2}`, `{1, 2}`) happen to iterate ascending, which is exactly
    why the whole oracle corpus above cannot catch this on its own.
    """
    assert list({3, 10}) == [10, 3], "premise: CPython set order for this literal"
    ilabels = [f"f{i}" for i in range(16)]
    assert _dex().fs_to_names({3, 10}, ilabels) == ["f3", "f10"]
    assert _dex().fs_to_names({1, 8}, ilabels) == ["f1", "f8"]
    assert _dex().fs_to_names({2, 9, 16 - 1}, ilabels) == ["f2", "f9", "f15"]


def test_fs_to_names_of_an_empty_feature_set_is_empty():
    assert _dex().fs_to_names(set(), _ILABELS) == []


def test_fs_to_names_accepts_any_collection_not_only_a_set():
    """C++ takes `const feature_set&` (a `std::set`), so a caller cannot
    pass a duplicate; a Python caller can. Sorting a list keeps the
    ascending contract, and duplicates are simply reproduced -- documented
    rather than defended against, since no real caller can produce one.
    """
    assert _dex().fs_to_names([2, 0], _ILABELS) == ["x1", "x3"]


# ---------------------------------------------------------------------------
# _type_to_exemplar
# ---------------------------------------------------------------------------


def test_type_to_exemplar_boolean_is_logical_and():
    tr = _type_to_exemplar(type_node.boolean_type)
    assert tr.size() == 1
    assert tr.begin().data == builtin.logical_and


def test_type_to_exemplar_contin_is_plus():
    tr = _type_to_exemplar(type_node.contin_type)
    assert tr.size() == 1
    assert tr.begin().data == builtin.plus


def test_type_to_exemplar_enum_raises():
    """representation.cc:81's `enum_type` branch needs
    `enum_t::get_random_enum()`; the enum domain is excluded project-wide.
    """
    with pytest.raises(NotImplementedError):
        _type_to_exemplar(type_node.enum_type)


def test_type_to_exemplar_ill_formed_raises():
    """representation.cc:86-89 is `OC_ASSERT(false, ...)` -- an abort."""
    with pytest.raises(NotImplementedError):
        _type_to_exemplar(type_node.ill_formed_type)


# ---------------------------------------------------------------------------
# create_representations wiring: festor.params.prune_xmplr
# ---------------------------------------------------------------------------


def _boolean_dex_and_ctable(params: deme_parameters):
    table = loadTable(str(_TABLE_DATA_DIR / _DATASET), _TARGET)
    ctable = table.compressed()
    cscorer = behave_cscore(ctruth_table_bscore(ctable))
    rule = logical_reduction(())(2)
    hc = hill_climbing(optim_parameters(), hc_parameters(cross=False))
    sig = gen_signature(type_node.boolean_type, _ARITY)
    return deme_expander(sig, rule, rule, cscorer, hc, params), ctable


def _festor(ctable, **kwargs):
    festor_params = feature_selector_parameters(**kwargs)
    # Matches tests/fixtures/feature_selection.json's "in_loop" `$1`,
    # base_target_size=0 case exactly -- the selector picks `{2}` (x3).
    festor_params.fs_params.target_size = 0
    return feature_selector(ctable, festor_params)


def test_create_representations_prunes_the_exemplar_when_prune_xmplr_is_true():
    """The whole point of `--fs-prune-exemplar`, in one assertion pair.

    For exemplar `$1` on trainingBoolean.dat the selector returns `{2}`
    (x3), and the two arms of deme_expander.cc:216-236 then diverge:

    * prune_xmplr=False (cc:230-236): `{2}` is unioned with the exemplar's
      own `{0}`, so ignore_cols is `{1}` and the exemplar `$1` is used
      as-is.
    * prune_xmplr=True (cc:216-229): NO union, so ignore_cols is `{0, 1}`
      -- the exemplar's own feature is ignored too -- and the exemplar
      itself becomes `prune_xmplr("$1", {2})`, which erases the root and
      falls back to the `and` seed.

    Collapsing the two arms into a shared "compute selected_feats then
    branch" would make both lines below equal.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = _festor(ctable, prune_xmplr=True)

    randGen().seed(1)
    dex.create_demeIDs(0)
    created = dex.create_representations(str_to_combo_tree("$1"))

    assert created is True
    assert dex._ignore_cols_seq == [{0, 1}]

    params2 = deme_parameters()
    params2.perm_ratio = -1.0
    dex2, ctable2 = _boolean_dex_and_ctable(params2)
    dex2._params.fstor = _festor(ctable2, prune_xmplr=False)
    randGen().seed(1)
    dex2.create_demeIDs(0)
    assert dex2.create_representations(str_to_combo_tree("$1")) is True
    assert dex2._ignore_cols_seq == [{1}]


def test_create_representations_prune_arm_uses_the_pruned_exemplar():
    """Not just `ignore_cols`: the REPRESENTATION is built over the pruned
    tree. With exemplar `$1` pruned to the `and` seed and only x3 left, the
    decorated exemplar must not mention `$1` at all.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = _festor(ctable, prune_xmplr=True)

    randGen().seed(1)
    dex.create_demeIDs(0)
    assert dex.create_representations(str_to_combo_tree("$1")) is True
    decorated = combo_tree_to_str(dex._reps[0].exemplar())
    assert "$1" not in decorated
    assert "$2" not in decorated
    assert "$3" in decorated


def test_create_representations_prune_arm_survives_debug_logging():
    """The prune arm carries two debug records (cc:219-221 and cc:222-228,
    the latter the only live caller of `fs_to_names`). They sit behind
    `logger().is_debug_enabled()`, which is False by default -- so every
    other test in this file leaves them unexecuted. Run once with DEBUG on
    so a typo in an f-string there is a test failure, not a surprise at
    `-l DEBUG`.
    """
    lg = logger()
    old_level = lg.get_level()
    lg.set_level(Logger.Level.DEBUG)
    try:
        params = deme_parameters()
        params.perm_ratio = -1.0
        dex, ctable = _boolean_dex_and_ctable(params)
        dex._params.fstor = _festor(ctable, prune_xmplr=True)
        randGen().seed(1)
        dex.create_demeIDs(0)
        assert dex.create_representations(str_to_combo_tree("$1")) is True

        # ... and the cc:219-221 branch, which needs an exemplar with NO
        # arguments at all so `xmplr_features` is empty.
        params2 = deme_parameters()
        params2.perm_ratio = -1.0
        dex2, ctable2 = _boolean_dex_and_ctable(params2)
        dex2._params.fstor = _festor(ctable2, prune_xmplr=True)
        randGen().seed(1)
        dex2.create_demeIDs(0)
        dex2.create_representations(str_to_combo_tree("and"))

        # ... and the non-pruning arm's own record (cc:231-234).
        params3 = deme_parameters()
        params3.perm_ratio = -1.0
        dex3, ctable3 = _boolean_dex_and_ctable(params3)
        dex3._params.fstor = _festor(ctable3, prune_xmplr=False)
        randGen().seed(1)
        dex3.create_demeIDs(0)
        assert dex3.create_representations(str_to_combo_tree("$1")) is True
    finally:
        lg.set_level(old_level)


def test_create_demes_runs_end_to_end_with_prune_xmplr():
    """`create_demes` -> `create_demeIDs` + `create_representations`; the
    prune arm must not break the deme bookkeeping downstream of it.
    """
    params = deme_parameters()
    params.perm_ratio = -1.0
    dex, ctable = _boolean_dex_and_ctable(params)
    dex._params.fstor = _festor(ctable, prune_xmplr=True)

    randGen().seed(1)
    assert dex.create_demes(str_to_combo_tree("$1"), 0) is True
    assert len(dex._demes) == 1
    assert len(dex._reps) == 1
