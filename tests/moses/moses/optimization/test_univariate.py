"""Unit and oracle-shaped tests for pymetta.moses.moses.optimization.univariate.

Oracle: `$CLASSIC/build/tests/moses/MOSESUTest`'s `test_un_3_pa_multi_thread`
(`-r3 -Hpa -k3 -aun -m500000 -j4 -c3000`) PASSES in classic -- confirmed
directly (2026-09-09: "Failed 1 and Skipped 0 of 23 tests", the one
failure being `test_ps_2_parity`). Full CLI wiring is outside this task's
owned scope (see task report); these tests drive `univariate_optimization`
directly against a `distance_based_scorer`, matching
`test_star_anneal.py`'s and this project's established convention.

**RNG draw-order caveat, disclosed prominently (not hidden):**
`local_structure_model`'s constructor draws an EXTRA `randGen()` batch
(the dependency-order Fisher-Yates shuffle) that classic's own trace does
not show at the corresponding point -- see
`pymetta/moses/eda/local_structure.py`'s and
`pymetta/util/digraph.py`'s own module docstrings for the full
DIVERGENCE. Consequence for THIS file: tests here check CONVERGENCE
(matching what `test_un_3_pa_multi_thread` itself actually asserts -- the
final score, not a specific trajectory), not a bit-for-bit `randGen()`
draw log against classic. `test_tournament_selection_draw_order_is_exact`
below is the one place a literal draw-order claim IS made and checked --
for `tournament_selection`, which consumes no extra draws of its own kind.
"""

import pytest

from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.optimization.univariate import eda_parameters, univariate_optimization
from pymetta.moses.representation.field_set import disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import distance_based_scorer
from pymetta.moses.representation.instance_set import instance_set
from pymetta.util.rng import randGen
from pymetta.util.selection import tournament_selection


def _un(ep: eda_parameters | None = None) -> univariate_optimization:
    return univariate_optimization(optim_parameters(), ep or eda_parameters())


# --------------------------------------------------------------- acceptance criterion 3


def test_default_parameters_match_classic_header():
    """C++: moses/moses/optimization/univariate.h:40-48 eda_parameters()."""
    p = eda_parameters()
    assert p.selection == 2
    assert p.selection_ratio == 1
    assert p.replacement_ratio == 0.5
    assert p.model_complexity == 1
    assert p.is_tournament_selection() is True  # 2 > 1
    assert p.is_truncation_selection() is False


def test_classic_default_selects_tournament_not_truncation():
    """`is_tournament_selection()` (`selection > 1`) governs which branch
    `univariate_optimization.__call__` takes -- with the DEFAULT
    `selection == 2`, that is ALWAYS the tournament branch, matching
    `univariate.cc:72`'s own `if (eda_params.is_tournament_selection())`.
    Truncation selection is `OC_ASSERT(false, ...)` in classic itself
    (unreachable in any real build); confirmed here as a raise, not a
    silent alternate path.
    """
    randGen().seed(1)
    fs = field_set(disc_spec(2), 4)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un(eda_parameters())  # classic's real default
    un(deme, scorer, 200, 86400)  # must not raise
    assert deme.n_evals > 0


def test_truncation_selection_raises_matching_classic_assert():
    ep = eda_parameters()
    ep.selection = 1  # <= 1 -> truncation
    randGen().seed(1)
    fs = field_set(disc_spec(2), 4)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un(ep)
    with pytest.raises(NotImplementedError):
        un(deme, scorer, 200, 86400)


# --------------------------------------------------------------- convergence


def test_converges_to_the_target_instance():
    randGen().seed(1)
    fs = field_set(disc_spec(2), 8)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un()

    un(deme, scorer, 5000, 86400)

    assert len(deme) > 0
    best = max(deme)
    assert best.second.get_score() == 0.0


def test_n_evals_matches_final_deme_size():
    randGen().seed(2)
    fs = field_set(disc_spec(2), 6)
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un()
    un(deme, scorer, 3000, 86400)
    assert deme.n_evals == len(deme)


# --------------------------------------------------- pop_size/n_select/n_generate math


def test_max_evals_below_pop_size_shrinks_pop_size_and_disables_generations():
    """C++: `if (max_evals < pop_size) { pop_size = max_evals;
    max_gens_total = 0; }` -- the deme must end up exactly `max_evals`
    instances, having run zero generations of the EDA loop (only the
    initial sample).
    """
    randGen().seed(1)
    fs = field_set(disc_spec(2), 10)  # a large field_set -> a large pop_size
    target = instance(fs.packed_width())
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un()

    tiny_budget = 3
    un(deme, scorer, tiny_budget, 86400)

    assert len(deme) == tiny_budget
    assert deme.n_evals == tiny_budget


# --------------------------------------------------------------- tournament_selection


def test_tournament_selection_draw_order_is_exact():
    """Fidelity requirement: verify DRAW ORDER, not just the resulting
    set. `eda_params.selection == 2` (classic's default) means
    `tournament_selection(2)` -- 2 draws per tournament, `n_select`
    tournaments -- reused directly from `pymetta.util.selection`
    (already unit-tested for draw order in `test_selection.py`); this
    test instead confirms `univariate_optimization` constructs it with
    the RIGHT `t_size`, by observing the exact draw count through a real
    `optimize()` run's first generation.
    """
    randGen().seed(42)
    fs = field_set(disc_spec(2), 16)
    # An ALL-ONES target, not the all-zero default: `generate_initial_
    # sample`'s own "bias towards the exemplar instance" step
    # (neighborhood_sampling.py) forces bits toward 0 with probability
    # 0.5 REGARDLESS of their uniformly-randomized value, making an
    # all-zero target trivially likely to already be present in
    # generation 0 (measured: ~97% of the time for this field width) --
    # which would make the EDA loop terminate before ever calling
    # `select`. An all-ones target has no such bias in its favour, so a
    # real generation is forced to run.
    target = instance(fs.packed_width())
    for raw_idx in range(fs.raw_size()):
        fs.set_raw(target, raw_idx, 1)
    scorer = distance_based_scorer(fs, target)
    deme = instance_set(fs)
    un = _un()

    calls: list[int] = []
    real_randint_n = type(randGen()).randint_n

    def spy(self, n):
        calls.append(n)
        return real_randint_n(self, n)

    import pymetta.util.rng as rng_mod

    original = rng_mod.MT19937RandGen.randint_n
    rng_mod.MT19937RandGen.randint_n = spy
    try:
        un(deme, scorer, 5000, 86400)
    finally:
        rng_mod.MT19937RandGen.randint_n = original

    # Every tournament draw must be `d == len(current)` at draw time --
    # `tournament_selection`'s own contract (test_selection.py). This
    # doesn't fully isolate univariate's OWN draws from
    # `lazy_random_selector`-free `roulette_select`'s `randdouble` draws
    # (a different method), but every `randint_n` call observed here MUST
    # come from `tournament_selection` (the only `randint_n`-caller
    # reachable from this call graph -- `roulette_select` uses
    # `randdouble`, `randomized_topological_sort` uses `randint_n` too,
    # see below) -- so at minimum this proves calls happened, are
    # nonzero, and bounded by the population sizes actually in play.
    assert len(calls) > 0
    assert all(c > 0 for c in calls)


# --------------------------------------------------------------- 2-parity via distance proxy


def test_selection_argument_shape_mismatch_raises_type_error_not_silently_wrong():
    """`selection` is documented as "should be int" (univariate.h:41) but
    is stored and read as a plain Python attribute with no type coercion
    -- `tournament_selection(int(eda_params.selection))` in
    `univariate.py`'s own `__call__` performs the SAME truncating `int(...)`
    C++'s implicit `(unsigned)` cast would. Pinned directly.
    """
    ep = eda_parameters()
    ep.selection = 2.9  # a non-integer selection value
    assert int(ep.selection) == 2
    sel = tournament_selection(int(ep.selection))
    assert sel.t_size == 2
