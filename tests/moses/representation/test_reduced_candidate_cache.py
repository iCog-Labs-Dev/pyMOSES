"""Tests for `representation`'s reduced-candidate cache (no C++ counterpart).

The cache is a port-only speed-up: `get_candidate(inst, True)` memoises
"unreduced candidate -> cleaned and reduced candidate" per representation.
Every test here compares a cached representation against an identical one
built with the cache disabled, so a cache hit must be indistinguishable from
a fresh reduction.
"""

import importlib
import json
import random

import pytest

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.contin_reduction import contin_reduction
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.moses.representation.representation import (
    _reduction_cache_key,
    representation,
)
from pymoses.moses.scoring.behave_cscore import _cache_key

# The package re-exports the `representation` class under the submodule's
# own name, so fetch the module object explicitly.
representation_module = importlib.import_module("pymoses.moses.representation.representation")


@pytest.fixture(scope="module")
def golden(fixture_dir):
    return json.loads((fixture_dir / "representation.json").read_text())


def _bool_rep(case):
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


def _uncached(build, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(representation_module, "REDUCED_CANDIDATE_CACHE_SIZE", 0)
        rep = build()
    assert rep._reduced_cache is None
    return rep


def _same(a: combo_tree, b: combo_tree) -> None:
    # `_cache_key` keeps every vertex's class and exact value, so this is a
    # stricter comparison than the printed form.
    assert _cache_key(a) == _cache_key(b)
    assert combo_tree_to_str(a) == combo_tree_to_str(b)


def test_every_boolean_fixture_row_reduces_identically_with_and_without_the_cache(
    golden, monkeypatch
):
    rows_checked = 0
    for case in golden["cases"]:
        cached = _bool_rep(case)
        plain = _uncached(lambda case=case: _bool_rep(case), monkeypatch)
        assert cached._reduced_cache is not None
        # Every row twice: the first pass fills the cache, the second hits it.
        for _ in range(2):
            for row in case["rows"]:
                try:
                    inst = cached.fields().pack(row["raw"])
                    expected = plain.get_candidate(inst, True)
                except ValueError:
                    with pytest.raises(ValueError):
                        cached.get_candidate(inst, True)
                    continue
                actual = cached.get_candidate(inst, True)
                _same(actual, expected)
                assert combo_tree_to_str(actual) == row["reduced"]
                rows_checked += 1
        assert cached.reduced_cache_hits > 0
        # The unreduced path never touches the cache.
        before = (cached.reduced_cache_hits, cached.reduced_cache_misses)
        cached.get_candidate(cached.fields().pack(case["rows"][0]["raw"]), False)
        assert (cached.reduced_cache_hits, cached.reduced_cache_misses) == before
    assert rows_checked > 1000


def _random_instance(rep, rng):
    fs = rep.fields()
    raw = []
    disc = fs.disc_and_bit()
    for idx in range(fs.raw_size()):
        if fs.begin_contin_raw_idx() <= idx < fs.end_contin_raw_idx():
            raw.append(rng.randrange(3))
        else:
            raw.append(rng.randrange(disc[fs.raw_to_disc_idx(idx)].multy))
    return fs.pack(raw)


@pytest.mark.parametrize(
    "source", ["plus(times($1 2) $2)", "plus(sin($1) times($2 -1.5) 0)", "times($1 $2 3)"]
)
def test_contin_candidates_reduce_identically_with_and_without_the_cache(source, monkeypatch):
    reducer = contin_reduction(2, set())
    signature = gen_signature([type_node.contin_type] * 2, type_node.contin_type)

    def build():
        return representation(
            reducer, reducer, str_to_combo_tree(source), signature, linear_contin=False
        )

    cached = build()
    plain = _uncached(build, monkeypatch)
    rng = random.Random(7)
    insts = [_random_instance(cached, rng) for _ in range(60)]
    for inst in insts + insts:
        _same(cached.get_candidate(inst, True), plain.get_candidate(inst, True))
    assert cached.reduced_cache_hits >= 60


def test_cached_results_are_fresh_copies_callers_can_mutate(golden):
    case = golden["cases"][2]
    rep = _bool_rep(case)
    inst = rep.fields().pack(case["rows"][0]["raw"])
    first = rep.get_candidate(inst, True)
    # Printed, not `_cache_key`: that key holds the argument objects
    # themselves, which the mutation below negates.
    expected = combo_tree_to_str(first)
    # Negate every argument and graft a node onto the returned tree.
    for node in first:
        if isinstance(node.data, argument):
            node.data.negate()
    first.append_child(first.begin(), builtin.logical_true)

    second = rep.get_candidate(inst, True)
    assert rep.reduced_cache_hits == 1
    assert combo_tree_to_str(second) == expected
    third = rep.get_candidate(inst, True)
    assert third.begin() is not second.begin()
    for a, b in zip(second, third, strict=True):
        assert not isinstance(a.data, argument) or a.data is not b.data


def test_cache_is_lru_bounded(golden, monkeypatch):
    case = max(golden["cases"], key=lambda c: len(c["rows"]))
    monkeypatch.setattr(representation_module, "REDUCED_CANDIDATE_CACHE_SIZE", 3)
    rep = _bool_rep(case)
    plain = _uncached(lambda: _bool_rep(case), monkeypatch)
    for row in case["rows"][:50]:
        inst = rep.fields().pack(row["raw"])
        _same(rep.get_candidate(inst, True), plain.get_candidate(inst, True))
        assert len(rep._reduced_cache) <= 3


def test_only_comboreduct_rules_are_cached():
    calls = []

    def probe(tr):
        calls.append(combo_tree_to_str(tr))

    rep = representation(
        probe, probe, str_to_combo_tree("and($1 $2)"), gen_signature(type_node.boolean_type, 2)
    )
    assert rep._reduced_cache is None
    inst = rep.fields().pack([0] * rep.fields().raw_size())
    calls.clear()
    rep.get_candidate(inst, True)
    rep.get_candidate(inst, True)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("1", "true"),  # contin 1.0 vs builtin logical_true
        ("$1", "!$1"),  # argument sign
        ("and(or($1 $2) $3)", "and(or($1) $2 $3)"),  # shape, same pre-order labels
        ("plus($1 1)", "plus($1 1.0000000001)"),  # contin precision
    ],
)
def test_reduction_key_separates_trees_that_differ(a, b):
    assert _reduction_cache_key(str_to_combo_tree(a)) != _reduction_cache_key(
        str_to_combo_tree(b)
    )


def test_reduction_key_separates_signed_zero_and_holds_no_mutable_vertex():
    pos = combo_tree(builtin.plus)
    pos.append_child(pos.begin(), 0.0)
    neg = combo_tree(builtin.plus)
    neg.append_child(neg.begin(), -0.0)
    assert _reduction_cache_key(pos) != _reduction_cache_key(neg)

    tr = str_to_combo_tree("and($1 $2)")
    key = _reduction_cache_key(tr)
    for node in tr:
        if isinstance(node.data, argument):
            node.data.negate()
    assert _reduction_cache_key(str_to_combo_tree("and($1 $2)")) == key
    assert _reduction_cache_key(combo_tree()) is None
