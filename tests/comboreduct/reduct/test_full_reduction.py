"""Tests for pymetta.comboreduct.reduct.full_reduction (S14 Task 4) that
are not covered by the byte-identical oracle in
test_full_reduction_oracle.py: the caching behaviour itself, and
`clean_and_full_reduce`'s two-call shape.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.reduct.full_reduction import (
    _reset_cache_for_testing,
    clean_and_full_reduce,
    full_reduce,
    full_reduction,
)


def test_full_reduction_caches_across_calls():
    _reset_cache_for_testing()
    first = full_reduction()
    second = full_reduction()
    assert first is second
    _reset_cache_for_testing()


def test_full_reduction_reset_allows_a_fresh_build():
    _reset_cache_for_testing()
    first = full_reduction()
    _reset_cache_for_testing()
    second = full_reduction()
    assert first is not second
    _reset_cache_for_testing()


def test_full_reduce_reduces_a_simple_tautology():
    _reset_cache_for_testing()
    tr = str_to_combo_tree("or($1 not($1))")
    full_reduce(tr)
    assert combo_tree_to_str(tr).strip() == "true"
    _reset_cache_for_testing()


def test_clean_and_full_reduce_removes_null_vertices_then_fully_reduces():
    _reset_cache_for_testing()
    tr = str_to_combo_tree("and($1 $1)")
    clean_and_full_reduce(tr)
    # and($1 $1) has no null vertices to clean, but should still fully
    # reduce via the second (full_reduce) call.
    assert combo_tree_to_str(tr).strip() == "$1"
    _reset_cache_for_testing()
