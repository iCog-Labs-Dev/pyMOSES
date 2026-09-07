"""Tests for iscorer_base, distance_based_scorer and complexity_based_scorer."""

import json
import struct

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.moses.types import composite_score, worst_composite_score
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import (
    complexity_based_scorer,
    distance_based_scorer,
    iscorer_base,
)
from pymetta.moses.representation.representation import representation


def _hex(x: float) -> str:
    return "0x" + struct.pack(">f", x).hex()


def _fs(n: int = 8) -> field_set:
    return field_set(field_set.disc_spec(2), n)


# --- iscorer_base ------------------------------------------------------


def test_iscorer_base_is_abstract():
    with pytest.raises(TypeError):
        iscorer_base()


def test_iscorer_base_get_rep_default_is_none():
    class trivial_scorer(iscorer_base):
        def __call__(self, inst):
            return worst_composite_score

    assert trivial_scorer().get_rep() is None


# --- distance_based_scorer -----------------------------------------------


def test_distance_based_scorer_rejects_wrong_types():
    fs = _fs()
    target = instance(fs.packed_width())
    with pytest.raises(TypeError):
        distance_based_scorer("not-a-field-set", target)
    with pytest.raises(TypeError):
        distance_based_scorer(fs, [0, 0])


def test_distance_based_scorer_is_negative_hamming_distance():
    fs = _fs()
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)

    same = instance(fs.packed_width())
    assert scorer(same).get_score() == 0.0

    one_off = instance(fs.packed_width())
    fs.set_raw(one_off, 0, 1)
    cs = scorer(one_off)
    assert cs.get_score() == -1.0
    assert cs.get_complexity() == 0
    assert cs.get_complexity_penalty() == 0.0
    assert cs.get_uniformity_penalty() == 0.0


@pytest.mark.oracle
def test_cpp_distance_based_scorer_oracle(fixture_dir):
    """Reproduces containers_probe.cc's `emit_distance_scorer`: a target
    with alternating disc bits, and the same/one-bit-off/all-bits-off
    probes, against classic's actual `distance_based_scorer::operator()`.
    """
    fixture = json.loads((fixture_dir / "containers.json").read_text())
    expected = fixture["distance_scorer"]["results"]

    fs = field_set(field_set.disc_spec(2), 8)
    target = instance(fs.packed_width())
    for idx in range(8):
        fs.set_raw(target, idx, idx % 2)

    same = instance(target)
    one_off = instance(target)
    fs.set_raw(one_off, 0, (fs.get_raw(one_off, 0) + 1) % 2)
    all_off = instance(fs.packed_width())
    for idx in range(8):
        fs.set_raw(all_off, idx, (fs.get_raw(target, idx) + 1) % 2)

    scorer = distance_based_scorer(fs, target)
    for probe, expect in zip((same, one_off, all_off), expected, strict=True):
        cs = scorer(probe)
        assert _hex(cs.get_score()) == expect["score"]
        assert cs.get_complexity() == expect["complexity"]
        assert _hex(cs.get_penalized_score()) == expect["penalized_score"]


# --- complexity_based_scorer ----------------------------------------------


def _boolean_rep(source: str, arity: int) -> representation:
    reducer = logical_reduction(frozenset())(2)
    return representation(
        reducer,
        reducer,
        str_to_combo_tree(source),
        gen_signature(type_node.boolean_type, arity),
        perm_ratio=-1.0,
    )


class _stub_cscore:
    """Not itself a ported C++ symbol -- a minimal behave_cscore stand-in
    recording which combo_tree it was asked to score, so the reduce flag's
    plumbing through `representation.get_candidate` can be checked without
    a real bscorer/table.
    """

    def __init__(self, result: composite_score):
        self.result = result
        self.seen = []

    def get_cscore(self, tr):
        self.seen.append(tr)
        return self.result


def test_complexity_based_scorer_get_rep_returns_the_representation():
    rep = _boolean_rep("and($1 $2)", 2)
    scorer = complexity_based_scorer(_stub_cscore(worst_composite_score), rep, True)
    assert scorer.get_rep() is rep


def test_complexity_based_scorer_forwards_get_candidate_result_to_cscorer():
    rep = _boolean_rep("and($1 $2)", 2)
    want = composite_score(-3.0, 4, 0.0, 0.0)
    cscore = _stub_cscore(want)
    scorer = complexity_based_scorer(cscore, rep, True)

    inst = instance(rep.fields().packed_width())
    got = scorer(inst)

    assert got is want
    # `tree`/`combo_tree` defines no value `__eq__` (matching C++'s own
    # `tree<T>`, which has neither `operator==` nor a hash -- see
    # `behave_cscore.py`'s `_cache_key` docstring); compare by rendering.
    assert [combo_tree_to_str(tr) for tr in cscore.seen] == [
        combo_tree_to_str(rep.get_candidate(inst, True))
    ]


class _raising_representation:
    """Not itself a ported C++ symbol -- forces
    `complexity_based_scorer.__call__`'s except branch without a full
    representation/knob-building setup."""

    def get_candidate(self, inst, reduce):
        raise ValueError("synthetic evaluation failure")


def test_complexity_based_scorer_swallows_exceptions_and_returns_worst_score():
    scorer = complexity_based_scorer(
        _stub_cscore(composite_score(-1.0, 0, 0.0, 0.0)), _raising_representation(), True
    )
    assert scorer(instance(1)) is worst_composite_score


class _keyboard_interrupting_representation:
    def get_candidate(self, inst, reduce):
        raise KeyboardInterrupt


def test_complexity_based_scorer_does_not_swallow_non_exception_signals():
    # DIVERGENCE guard: `except Exception` (not bare `except:`) must let a
    # process-level control signal propagate rather than treating it as
    # "this instance could not be evaluated". Proves the choice is real,
    # not just documented.
    scorer = complexity_based_scorer(
        _stub_cscore(worst_composite_score), _keyboard_interrupting_representation(), True
    )
    with pytest.raises(KeyboardInterrupt):
        scorer(instance(1))
