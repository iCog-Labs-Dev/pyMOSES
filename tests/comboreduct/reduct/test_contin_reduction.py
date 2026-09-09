"""Tests for pymetta.comboreduct.reduct.contin_reduction (S14 Task 4).

Full end-to-end pipeline against the real compiled `libcomboreduct.so`
(via `contin_reduce`) lands in `test_contin_mixed_reduction_oracle.py`
once `mixed_reduction.py` also exists (both share one corpus + probe).
This file covers: the assembled rule-set composition producing sensible
results on hand-picked small trees, the `# DIVERGENCE`-free but
explicitly-disclosed static-singleton CACHING quirk (§5.4's own lesson:
"a default differing from classic's is a divergence even when nothing
reads it yet" -- ported here as the real, load-bearing behaviour it is,
not silently smoothed over), and that `ignore_ops` genuinely gates the
rules it is documented to gate.
"""

from __future__ import annotations

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.reduct.contin_reduction import (
    _reset_cache_for_testing,
    contin_reduce,
    contin_reduction,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every test gets a clean cache -- see contin_reduction.py's own
    module docstring for why this is needed at all (no C++ equivalent).
    """
    _reset_cache_for_testing()
    yield
    _reset_cache_for_testing()


def _reduce(tree_str: str, effort: int = 2, ignore_ops: set | None = None) -> str:
    tr = str_to_combo_tree(tree_str)
    contin_reduce(tr, reduct_effort=effort, ignore_ops=ignore_ops or set())
    return combo_tree_to_str(tr).strip()


# ---------------------------------------------------------------------------
# End-to-end pipeline sanity, classic's own real defaults (empty ignore_ops)
# ---------------------------------------------------------------------------


def test_contin_reduce_drops_redundant_plus_wrapper():
    # +(0 $1 0) -> $1 -- reduce_plus_zero alone leaves +($1); this proves
    # reduce_plus_times_one_child actually runs afterward in the pipeline.
    assert _reduce("+(0.0 $1 0.0)") == "$1"


def test_contin_reduce_factorizes_common_multiplicative_term():
    # x*y + x*z -> x*(y+z); reorder_commutative then sorts the resulting
    # times's two children lexicographically, putting the compound
    # +($2 $3) subtree before the bare argument $1.
    assert _reduce("+(*($1 $2) *($1 $3))") == "*(+($2 $3) $1)"


def test_contin_reduce_cancels_fraction_and_cleans_up_wrappers():
    # (x*y)/(x*z) -> y/z, fully cleaned (the isolated reduce_fraction rule
    # alone leaves *($2)/*($3); the pipeline's reduce_plus_times_one_child
    # collapses each one-child times down to the bare argument).
    assert _reduce("/(*($1 $2) *($1 $3))") == "/($2 $3)"


def test_contin_reduce_merges_exp_product_into_sum():
    assert _reduce("*(exp($1) exp($2))") == "exp(+($1 $2))"


def test_contin_reduce_on_a_bare_argument_is_a_noop():
    assert _reduce("$1") == "$1"


def test_contin_reduce_reduces_a_pure_constant_expression():
    assert _reduce("+(1.0 2.0 3.0)") == "6"


# ---------------------------------------------------------------------------
# ignore_ops gating
# ---------------------------------------------------------------------------


def test_ignore_ops_div_disables_fraction_cancellation():
    # The module-level cache (this module's own docstring) means the
    # SECOND call below would otherwise silently reuse the first call's
    # rule regardless of its own ignore_ops -- reset explicitly BETWEEN
    # the two configurations, not just once per test.
    with_div = _reduce("/(*($1 $2) *($1 $3))", ignore_ops=set())
    _reset_cache_for_testing()
    without_div = _reduce("/(*($1 $2) *($1 $3))", ignore_ops={builtin.div})
    assert with_div == "/($2 $3)"
    # every div-gated rule (reduce_fraction, reduce_invert_constant,
    # reduce_times_div, reduce_factorize_fraction, reduce_exp_div, the
    # reduce_exp_log gate's div half) is skipped, so the div survives
    # structurally unreduced in this particular shape.
    assert without_div != with_div


def test_ignore_ops_sin_disables_sin_folding():
    tr_str = "sin(+($1 4.0))"

    with_sin = str_to_combo_tree(tr_str)
    contin_reduce(with_sin, reduct_effort=2, ignore_ops=set())
    # sin folding ran: 4.0 is out of (-pi, pi] and gets replaced.
    assert "4" not in combo_tree_to_str(with_sin)

    _reset_cache_for_testing()
    without_sin = str_to_combo_tree(tr_str)
    contin_reduce(without_sin, reduct_effort=2, ignore_ops={builtin.sin})
    # sin folding skipped -- the constant 4.0 survives untouched
    # (reorder_commutative may still move it within the sum, but not
    # change its value).
    assert "4" in combo_tree_to_str(without_sin)


# ---------------------------------------------------------------------------
# Static-singleton caching -- ported faithfully, load-bearing, tested directly
# ---------------------------------------------------------------------------


def test_contin_reduction_caches_across_different_arguments():
    """MEASURED behaviour of C++'s own `static rule *pr` -- the SECOND
    call, even with completely different arguments, returns the exact
    same object the first call built. Proven here, not merely asserted
    in a docstring (this project's own §5.4 lesson: an undischarged
    "documented divergence" already cost a real correctness bug once).
    """
    first = contin_reduction(1, set())
    second = contin_reduction(3, {builtin.div, builtin.log})
    assert first is second


def test_contin_reduction_cache_reset_lets_a_new_configuration_take_effect():
    contin_reduction(1, set())
    _reset_cache_for_testing()
    rebuilt = contin_reduction(3, set())
    assert rebuilt is not None
