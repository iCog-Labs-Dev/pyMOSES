"""`--knob-probe-baseline` (extension, not in classic MOSES).

`classic` (the default) probes knobs exactly as classic MOSES does
(`build_knobs.disc_probe`, C++ build_knobs.cc:311). `reduced` disallows a knob
setting only when reducing the decorated exemplar with that setting removes
MORE complexity than reducing it with the knob at 0 does. It differs from
classic only when the baseline itself is reducible, which is what makes knob
counts collapse on some mux11 exemplars (docs/gomea/BACKLOG.md section 4
item 1; exemplars from docs/gomea/knob-collapse-2026-10-16/).
"""

import argparse
import collections

import pytest

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.moses.deme.deme_params import deme_parameters
from pymoses.moses.main.problem_params import problem_params
from pymoses.moses.moses.complexity import tree_complexity
from pymoses.moses.representation.build_knobs import PROBE_BASELINES, build_knobs
from pymoses.moses.representation.knob_mapper import knob_mapper
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import randGen

# Collapsing exemplars (c*) and controls (n*) from mux11 GOMEA runs 3001-3003.
EXEMPLARS = {
    "n3001_6": "and(or(and(or(and(!$1 $4) $3) or(!$1 $9) !$2) and(or(!$1 $7) or($2 !$4 $5) "
    "!$3) and(!$1 $10)) or(and(or(and($9 !$11) $1 $8 $10) $3) and($2 $6) $4))",
    "c3001_7": "and(or(and(or(and(or($2 $8) !$1) $9) or(and(!$1 $4) $3) !$2) and(or(!$1 $7) "
    "or($2 $5 !$7) or($6 $7) !$3) and(!$1 $10)) or(and(or(and(!$5 !$6 $10) $1 $8) $3) "
    "and($1 $11) and($2 $6) $4))",
    "n3002_10": "and(or(and(or(and(or(and(!$3 $9) $8) or(and(!$9 $11) !$2) or(!$2 !$5) $4) "
    "and(or($3 $6) $2)) !$1) and(or(and(!$2 $5) $3 $7) or(and($2 $11) and($3 $9) $5) "
    "or(!$2 $11) or($2 !$3 $9) $1)) or(and(!$2 $4) $1 !$3 $10))",
    "c3002_11": "and(or(and(or(and(or(and(or(!$8 $9 !$11) !$3) $8) or(and(!$9 $11) !$2) "
    "or(!$2 !$5) $4) and(or($3 $6) $2)) !$1) and(or(and(!$2 $5) $3 $7) or(and($2 $11) "
    "and($3 $9) $5) or(!$2 $11) or($2 !$3 $9) $1)) or(and(!$2 $4) $1 !$3 $10))",
    "c3003_11": "and(or(and(or(and(or(and(or($1 $9) or($2 $10) !$7 $11) and($5 $8) !$3) $4) "
    "and($3 !$5 !$7 !$9) $2) or(and(or($8 $9 !$10) !$2 $4 $5 !$7) and(or(!$9 $10 $11) $3) "
    "$6) !$1) and(or(and(or(and($4 $8 $11) $3 !$6) or($4 !$5 $8) or($4 !$6 $8) !$2) "
    "and(!$5 $6 $8) $1) or(and(or(!$7 $10) !$1) !$3 $9) or(and(!$1 $3) $7) or(!$3 $8)) "
    "and($1 $3 $9)) or(and(or(and(or(and(!$2 $7) $1) $3) and(or(and($2 $8) !$6) $1 $5)) "
    "or(and($9 $11) !$2)) and(or(and(!$1 $10) !$3) $2) and(or(and($6 !$10) !$3 $8) "
    "or($3 $4) !$1 !$2) and(or(and($9 !$11) $6) !$2 $5)))",
}

# (classic, reduced) knob counts with --logical-perm-ratio -1 (no pair perms,
# to keep the test fast) and the RNG seeded with 1. c3003_11 (138 nodes:
# classic 50, reduced 1,324) is left out because it takes about 9 minutes;
# with the default perm ratio the counts are 135/1,048, 250/1,234 and
# 114/2,704 for the collapsing exemplars and 832 and 1,193 (both
# modes) for the controls.
KNOB_COUNTS = {
    "n3001_6": (409, 409),
    "c3001_7": (65, 511),
    "n3002_10": (575, 575),
    "c3002_11": (110, 598),
}

_REDUCER = logical_reduction(set())(2)


def _rep(name, probe_baseline):
    randGen().seed(1)
    return representation(
        _REDUCER,
        _REDUCER,
        str_to_combo_tree(EXEMPLARS[name]),
        gen_signature(type_node.boolean_type, 11),
        perm_ratio=-1,
        probe_baseline=probe_baseline,
    )


def _knobs(rep):
    """Multiset of (knob text, disallowed settings) over a representation."""
    return collections.Counter(
        (e.knob.toStr(), tuple(sorted(e.knob._disallowed))) for e in rep.disc
    )


@pytest.mark.parametrize("name", sorted(KNOB_COUNTS))
def test_knob_counts_classic_and_reduced(name):
    classic, reduced = _rep(name, "classic"), _rep(name, "reduced")
    assert (len(classic.disc), len(reduced.disc)) == KNOB_COUNTS[name]
    kc, kr = _knobs(classic), _knobs(reduced)
    if name.startswith("n"):
        # Controls: the baseline is never reducible, so the representation
        # (every knob and its disallowed settings) is exactly classic's.
        assert kc == kr
        assert tree_to_stream_str(classic._exemplar) == tree_to_stream_str(reduced._exemplar)
    else:
        # Collapsing exemplars: every knob classic builds is still built,
        # with the same disallowed settings; the rest are the knobs classic
        # threw away once the baseline became reducible.
        assert not (kc - kr)
        assert len(reduced.disc) >= 4 * len(classic.disc)


def test_default_is_classic():
    assert PROBE_BASELINES == ("classic", "reduced")
    assert deme_parameters().knob_probe_baseline == "classic"
    import inspect

    for fn in (build_knobs.__init__, representation.__init__):
        assert inspect.signature(fn).parameters["probe_baseline"].default == "classic"


def test_invalid_probe_baseline_rejected():
    with pytest.raises(ValueError, match="probe_baseline"):
        build_knobs(
            str_to_combo_tree("and($1 $2)"),
            gen_signature(type_node.boolean_type, 2),
            knob_mapper(),
            _REDUCER,
            probe_baseline="bogus",
        )


def _pp(argv):
    pp = problem_params()
    parser = argparse.ArgumentParser()
    pp.add_options(parser)
    pp.add_gomea_options(parser)
    pp.parse_options(parser.parse_args(argv))
    return pp


def test_cli_option_reaches_deme_params():
    assert _pp([]).deme_params.knob_probe_baseline == "classic"
    assert (
        _pp(["--knob-probe-baseline", "reduced"]).deme_params.knob_probe_baseline == "reduced"
    )
    with pytest.raises(ValueError, match="--knob-probe-baseline"):
        _pp(["--knob-probe-baseline", "bogus"])


def _drop_first_literal(tr: combo_tree) -> None:
    """A fake knob-building reducer that always lowers complexity by exactly
    one: it replaces the first argument (pre-order) with `true`, i.e. every
    tree, the baseline included, looks reducible by one."""
    for node in tr.iter_pre_at(tr.begin()):
        if isinstance(node.data, argument):
            node.data = builtin.logical_true
            return


def _build(exemplar, arity, simplifier, probe_baseline):
    randGen().seed(1)
    tr = str_to_combo_tree(exemplar)
    mapper = knob_mapper()
    build_knobs(
        tr,
        gen_signature(type_node.boolean_type, arity),
        mapper,
        simplifier,
        perm_ratio=-1,
        probe_baseline=probe_baseline,
    )
    return collections.Counter(
        (e.knob.toStr(), tuple(sorted(e.knob._disallowed))) for e in mapper.disc
    )


@pytest.mark.parametrize("exemplar,arity", [("and($1 $2)", 3), ("and(or($1 $2) !$3)", 4)])
def test_uniformly_reducible_baseline(exemplar, arity):
    """When every tree reduces by the same amount, classic drops every knob,
    while `reduced` sees no setting adding reducibility and builds what a
    no-op reducer builds."""
    t = str_to_combo_tree(exemplar)
    before = tree_complexity(t)
    _drop_first_literal(t)
    assert tree_complexity(t) == before - 1

    noop = _build(exemplar, arity, lambda tr: None, "classic")
    assert noop
    assert not _build(exemplar, arity, _drop_first_literal, "classic")
    assert _build(exemplar, arity, _drop_first_literal, "reduced") == noop


@pytest.mark.parametrize("exemplar,arity", [("and($1 $2)", 3), ("and(or($1 $2) !$3)", 4)])
def test_fixpoint_baseline_matches_classic(exemplar, arity):
    """With the real reducer on a small exemplar (baseline a fixpoint), the
    two modes build the same knobs with the same disallowed settings."""
    assert _build(exemplar, arity, _REDUCER, "reduced") == _build(
        exemplar, arity, _REDUCER, "classic"
    )


def test_baseline_key_ignores_null_subtrees_only():
    """The memo key equals for trees that `clean_reduce` makes equal by
    dropping null_vertex subtrees, and differs otherwise."""
    from pymoses.comboreduct.reduct.clean_reduction import clean_reduce
    from pymoses.moses.representation.build_knobs import _baseline_key

    plain = str_to_combo_tree("and(or($1 !$2) $3)")
    marked = str_to_combo_tree("and(or($1 !$2) $3)")
    or_node = marked.begin().first_child
    null = marked.append_child(or_node, builtin.null_vertex)
    marked.append_child(null, argument(4))
    assert _baseline_key(plain.begin()) == _baseline_key(marked.begin())
    clean_reduce(marked)
    assert tree_to_stream_str(marked) == tree_to_stream_str(plain)
    assert _baseline_key(plain.begin()) != _baseline_key(
        str_to_combo_tree("and(or($1 $2) $3)").begin()
    )
    assert _baseline_key(str_to_combo_tree("cond($1 $2 $3)").begin()) is None
