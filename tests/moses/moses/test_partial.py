"""Guard tests for pymoses.moses.moses.partial.partial_solver.

No C++ probe/oracle: `partial_solver.solve()` is blocked end to end by two
independent, pre-existing gaps outside this task's owned tree (see
partial.py's own module docstring) -- `enum_effective_bscore`/
`enum_table_bscore` (bscores.py) do not exist, and
`metapopulation.merge_demes`'s own `merge_callback` gate (merging.py)
raises unconditionally. This file exercises every method that does NOT
need either missing dependency directly against hand-built `combo_tree`/
`CTable` structures, and proves the two blocked methods (`record_prefix`,
`final_cleanup`) still perform all their real, observable mutation before
raising at the very end.
"""

from collections import Counter

import pytest

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.vertex import builtin, combo_tree, enum_t
from pymoses.comboreduct.table.table import CTable
from pymoses.moses.deme.deme_params import deme_parameters
from pymoses.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymoses.moses.moses.moses_params import moses_parameters, moses_statistics
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.moses.optimization.particle_swarm import ps_parameters
from pymoses.moses.moses.partial import partial_solver
from pymoses.moses.moses.types import (
    scored_combo_tree,
    scored_combo_tree_ptr_set,
    scored_combo_tree_set,
)

_A = enum_t("A")
_B = enum_t("B")
_Z = enum_t("Z")


def _bool_ctable(rows: dict) -> CTable:
    """Builds a 1-input-column boolean CTable: `rows` maps a plain Python
    bool (the value of `$1`) to a `{enum_t: count}` dict for that row.
    """
    ct = CTable(olabel="y", ilabels=["x1"])
    for x1, counts in rows.items():
        key = (builtin.logical_true if x1 else builtin.logical_false,)
        ct[key] = Counter(counts)
    return ct


def _two_col_ctable(rows: dict) -> CTable:
    """Two input columns ($1, $2); `rows` maps (bool, bool) -> counts."""
    ct = CTable(olabel="y", ilabels=["x1", "x2"])
    for (x1, x2), counts in rows.items():
        key = (
            builtin.logical_true if x1 else builtin.logical_false,
            builtin.logical_true if x2 else builtin.logical_false,
        )
        ct[key] = Counter(counts)
    return ct


def _cond3(pred, concl, default) -> combo_tree:
    """Builds `cond(pred concl default)` -- a single predicate/conclusion
    pair plus a catch-all default conclusion.
    """
    t = combo_tree(builtin.cond)
    root = t.begin()
    t.append_child(root, pred)
    t.append_child(root, concl)
    t.append_child(root, default)
    return t


def _cond5(pred0, concl0, pred1, concl1, default) -> combo_tree:
    t = combo_tree(builtin.cond)
    root = t.begin()
    t.append_child(root, pred0)
    t.append_child(root, concl0)
    t.append_child(root, pred1)
    t.append_child(root, concl1)
    t.append_child(root, default)
    return t


def _make_solver(ctable: CTable, exemplars=()) -> partial_solver:
    return partial_solver(
        ctable=ctable,
        exemplars=list(exemplars),
        reduct=lambda tr: None,
        opt_params=optim_parameters(),
        hc_params=hc_parameters(),
        ps_params=ps_parameters(),
        deme_params=deme_parameters(),
        filter_params=subsample_deme_filter_parameters(),
        meta_params=metapop_parameters(),
        moses_params=moses_parameters(),
        printer=lambda metapop, dex, stats: None,
    )


# ------------------------------------------------------------------------ effective


def test_effective_perfect_predicate():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    pred = argument(1)
    concl = _A
    tree = _cond3(pred, concl, _Z)
    pred_node = list(tree.children(tree.begin()))[0]  # first child of cond's root
    good, fail = solver.effective(pred_node, 0, 0)
    assert good == 3
    assert fail == 0


def test_effective_imperfect_predicate_counts_failures():
    ct = _bool_ctable({True: {_A: 3, _B: 1}})
    solver = _make_solver(ct)
    tree = _cond3(argument(1), _A, _Z)
    pred_node = list(tree.children(tree.begin()))[0]
    good, fail = solver.effective(pred_node, 0, 0)
    assert good == 0
    assert fail == 1


def test_effective_accumulates_into_existing_counts():
    """Reference C++ semantics: good_count/fail_count are ACCUMULATED
    into, not reset -- the caller resets them, not this method."""
    ct = _bool_ctable({True: {_A: 3}})
    solver = _make_solver(ct)
    tree = _cond3(argument(1), _A, _Z)
    pred_node = list(tree.children(tree.begin()))[0]
    good, fail = solver.effective(pred_node, 10, 20)
    assert good == 13
    assert fail == 20


def test_effective_ignores_rows_where_predicate_is_false():
    ct = _bool_ctable({True: {_A: 2}, False: {_B: 99}})
    solver = _make_solver(ct)
    tree = _cond3(argument(1), _A, _Z)
    pred_node = list(tree.children(tree.begin()))[0]
    good, fail = solver.effective(pred_node, 0, 0)
    assert good == 2
    assert fail == 0


# ------------------------------------------------------------------------ trim_table


def test_trim_table_removes_matching_rows_only():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    tree = _cond3(argument(1), _A, _Z)
    pred_node = list(tree.children(tree.begin()))[0]
    deleted, total = solver.trim_table(ct, pred_node, 0, 0)
    assert deleted == 3
    assert total == 8
    assert len(ct) == 1
    assert (builtin.logical_false,) in ct
    assert (builtin.logical_true,) not in ct


def test_trim_table_does_not_mutate_caller_original_ctable():
    """self._ctable is a real copy (_copy_ctable) -- trimming it must not
    touch the constructor's own `ctable` argument."""
    orig = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(orig)
    tree = _cond3(argument(1), _A, _Z)
    pred_node = list(tree.children(tree.begin()))[0]
    solver.trim_table(solver._ctable, pred_node, 0, 0)
    assert len(solver._ctable) == 1
    assert len(orig) == 2  # untouched


# --------------------------------------------------------------------- eval_candidate


def test_eval_candidate_records_perfect_predicate():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    cand = _cond3(argument(1), _A, _Z)
    solver.eval_candidate(cand)
    assert solver._most_good == 3
    assert solver._best_predicate is not None
    assert solver._best_predicate.data == argument(1)


def test_eval_candidate_rejects_imperfect_predicate():
    ct = _bool_ctable({True: {_A: 3, _B: 1}})
    solver = _make_solver(ct)
    cand = _cond3(argument(1), _A, _Z)
    solver.eval_candidate(cand)
    assert solver._most_good == 0
    assert solver._best_predicate is None


def test_eval_candidate_returns_early_for_a_bare_enum_root():
    ct = _bool_ctable({True: {_A: 3}})
    solver = _make_solver(ct)
    cand = combo_tree(_Z)  # a single enum leaf, no cond structure at all
    solver.eval_candidate(cand)  # must not raise
    assert solver._most_good == 0


def test_eval_candidate_raises_on_non_cond_non_enum_root():
    ct = _bool_ctable({True: {_A: 3}})
    solver = _make_solver(ct)
    cand = combo_tree(builtin.logical_and)
    with pytest.raises(AssertionError, match="unexpected candidate"):
        solver.eval_candidate(cand)


def test_eval_candidate_skips_ineffective_leading_predicate():
    """$2 is universally false (ineffective: 0 good, 0 fail) in this
    table, so the loop must skip to $1."""
    ct = _two_col_ctable({(True, False): {_A: 4}})
    solver = _make_solver(ct)
    cand = _cond5(argument(2), _B, argument(1), _A, _Z)
    solver.eval_candidate(cand)
    assert solver._most_good == 4
    assert solver._best_predicate.data == argument(1)


def test_eval_candidate_reject_when_only_ineffective_clauses_precede_a_constant():
    """Every predicate before a bare enum leaf was ineffective -> reject
    outright without ever finding a good/fail verdict."""
    ct = _two_col_ctable({(False, False): {_A: 1}})
    solver = _make_solver(ct)
    # $1 and $2 both always false in this table -> both ineffective;
    # cond($1 A $2 B Z) never finds a verdict and hits the bare Z leaf.
    cand = _cond5(argument(1), _A, argument(2), _B, _Z)
    solver.eval_candidate(cand)
    assert solver._most_good == 0
    assert solver._best_predicate is None


def test_eval_candidate_does_not_overwrite_a_better_result():
    ct = _bool_ctable({True: {_A: 3}})
    solver = _make_solver(ct)
    solver._most_good = 100
    cand = _cond3(argument(1), _A, _Z)
    solver.eval_candidate(cand)
    assert solver._most_good == 100  # unchanged: 3 <= 100


# ------------------------------------------------------------------- eval_candidates


def test_eval_candidates_calls_record_prefix_when_something_improves():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    cand = _cond3(argument(1), _A, _Z)
    sct = scored_combo_tree(cand)
    cands = scored_combo_tree_set([sct])

    calls = []
    solver.record_prefix = lambda: calls.append(1)  # observe the call

    result = solver.eval_candidates(cands)
    assert result is True
    assert calls == [1]


def test_eval_candidates_returns_false_when_nothing_improves():
    ct = _bool_ctable({True: {_A: 3, _B: 1}})
    solver = _make_solver(ct)
    cand = _cond3(argument(1), _A, _Z)
    sct = scored_combo_tree(cand)
    cands = scored_combo_tree_set([sct])

    calls = []
    solver.record_prefix = lambda: calls.append(1)

    result = solver.eval_candidates(cands)
    assert result is False
    assert calls == []


def test_check_candidates_delegates_to_eval_candidates():
    ct = _bool_ctable({True: {_A: 3}})
    solver = _make_solver(ct)
    cand = _cond3(argument(1), _A, _Z)
    cands = scored_combo_tree_set([scored_combo_tree(cand)])
    solver.record_prefix = lambda: None
    assert partial_solver.check_candidates(cands, solver) is True


# -------------------------------------------------------------------- record_prefix


def test_record_prefix_mutates_state_then_raises():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    pred_tree = _cond3(argument(1), _A, _Z)
    solver._best_predicate = list(pred_tree.children(pred_tree.begin()))[0]

    with pytest.raises(NotImplementedError, match="enum_effective_bscore"):
        solver.record_prefix()

    # Real mutation already happened before the raise.
    assert solver._prefix_count == 1
    assert len(solver._ctable) == 1  # the True row was trimmed
    leader_children = list(solver._leader.children(solver._leader.begin()))
    assert len(leader_children) == 2
    assert leader_children[0].data == argument(1)
    assert leader_children[1].data == _A


def test_record_prefix_appends_to_existing_leader_not_replacing_it():
    ct = _bool_ctable({True: {_A: 3}, False: {_B: 5}})
    solver = _make_solver(ct)
    # Pre-seed _leader with one prior (predicate, conclusion) pair.
    prior_pred = combo_tree(argument(2)).begin()
    prior_concl = combo_tree(_B).begin()
    leader_root = solver._leader.begin()
    solver._leader.append_child(leader_root, prior_pred)
    solver._leader.append_child(leader_root, prior_concl)

    pred_tree = _cond3(argument(1), _A, _Z)
    solver._best_predicate = list(pred_tree.children(pred_tree.begin()))[0]
    with pytest.raises(NotImplementedError):
        solver.record_prefix()

    leader_children = list(solver._leader.children(leader_root))
    assert len(leader_children) == 4
    assert [c.data for c in leader_children] == [argument(2), _B, argument(1), _A]


# --------------------------------------------------------------------------- refresh


class _MetapopulationDouble:
    """Duck-types the one method `refresh`/`final_cleanup` actually call
    on a `metapopulation` (see partial.py's own module docstring for why
    `get_trees()`, not direct iteration, is the right stand-in here).
    """

    def __init__(self, trees):
        self._trees = scored_combo_tree_ptr_set([scored_combo_tree(t) for t in trees])

    def get_trees(self):
        return self._trees


def test_refresh_copies_trees_into_exemplars():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    tree1 = combo_tree(argument(1))
    cands = _MetapopulationDouble([tree1])

    solver.refresh(cands)

    assert len(solver._exemplars) == 1
    assert solver._exemplars[0].begin().data == argument(1)
    assert solver._exemplars[0] is not tree1  # a real copy, not an alias


def test_refresh_copy_is_independent_of_the_original():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    tree1 = combo_tree(builtin.cond)
    solver.refresh(_MetapopulationDouble([tree1]))

    copied = solver._exemplars[0]
    copied.append_child(copied.begin(), argument(9))
    # Mutating the copy must not add a child to the metapopulation's own tree.
    assert list(tree1.children(tree1.begin())) == []


# ---------------------------------------------------------------------- final_cleanup


def test_final_cleanup_prefixes_leader_clauses_then_raises():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    leader_root = solver._leader.begin()
    solver._leader.append_child(leader_root, combo_tree(argument(2)).begin())
    solver._leader.append_child(leader_root, combo_tree(_B).begin())

    reduct_calls = []
    solver._reduct = lambda tr: reduct_calls.append(tr)

    cand = _cond3(argument(1), _A, _Z)
    cands = _MetapopulationDouble([cand])

    with pytest.raises(NotImplementedError, match="enum_effective_bscore"):
        solver.final_cleanup(cands)

    assert len(solver._exemplars) == 1
    result_tree = solver._exemplars[0]
    children = list(result_tree.children(result_tree.begin()))
    # Leader's (argument(2), B) pair prepended before the candidate's own
    # (argument(1), A, Z).
    assert [c.data for c in children] == [argument(2), _B, argument(1), _A, _Z]
    assert reduct_calls == [result_tree]


def test_final_cleanup_appends_when_candidate_root_has_no_children():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    leader_root = solver._leader.begin()
    solver._leader.append_child(leader_root, combo_tree(argument(2)).begin())
    solver._leader.append_child(leader_root, combo_tree(_B).begin())
    solver._reduct = lambda tr: None

    cand = combo_tree(_Z)  # a bare leaf, zero children
    cands = _MetapopulationDouble([cand])

    with pytest.raises(NotImplementedError):
        solver.final_cleanup(cands)

    result_tree = solver._exemplars[0]
    children = list(result_tree.children(result_tree.begin()))
    assert [c.data for c in children] == [argument(2), _B]


def test_final_cleanup_with_empty_leader_leaves_candidates_unchanged_besides_reduct():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    solver._reduct = lambda tr: None
    cand = _cond3(argument(1), _A, _Z)
    cands = _MetapopulationDouble([cand])

    with pytest.raises(NotImplementedError):
        solver.final_cleanup(cands)

    result_tree = solver._exemplars[0]
    children = list(result_tree.children(result_tree.begin()))
    assert [c.data for c in children] == [argument(1), _A, _Z]


# ------------------------------------------------------------------------- __call__


def test_call_sets_done_when_most_good_is_zero_and_calls_final_cleanup():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    solver._most_good = 0
    calls = []
    solver.final_cleanup = lambda metapop: calls.append(("final", metapop))
    solver.refresh = lambda metapop: calls.append(("refresh", metapop))

    metapop = _MetapopulationDouble([])
    dex = object()
    stats = moses_statistics()
    stats.n_evals = 5

    solver(metapop, dex, stats)

    assert solver._done is True
    assert solver._num_evals == 5
    assert calls == [("final", metapop)]


def test_call_restarts_search_via_refresh_when_not_done():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    solver._most_good = 7
    solver._moses_params.max_evals = 1000
    calls = []
    solver.final_cleanup = lambda metapop: calls.append(("final", metapop))
    solver.refresh = lambda metapop: calls.append(("refresh", metapop))

    stats = moses_statistics()
    stats.n_evals = 5
    metapop = _MetapopulationDouble([])

    solver(metapop, object(), stats)

    assert solver._done is False
    assert calls == [("refresh", metapop)]


def test_call_sets_done_when_max_evals_reached_even_if_most_good_is_nonzero():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    solver._most_good = 7
    solver._moses_params.max_evals = 5
    calls = []
    solver.final_cleanup = lambda metapop: calls.append("final")
    solver.refresh = lambda metapop: calls.append("refresh")

    stats = moses_statistics()
    stats.n_evals = 5

    solver(_MetapopulationDouble([]), object(), stats)

    assert solver._done is True
    assert calls == ["final"]


# ---------------------------------------------------------------------- constructor


def test_constructor_copies_the_ctable_not_aliases_it():
    orig = _bool_ctable({True: {_A: 1}, False: {_B: 1}})
    solver = _make_solver(orig)
    del solver._ctable[(builtin.logical_true,)]
    assert len(orig) == 2  # unaffected


def test_constructor_copies_exemplar_trees_not_aliases_them():
    tree1 = combo_tree(builtin.cond)
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct, exemplars=[tree1])
    solver._exemplars[0].append_child(solver._exemplars[0].begin(), argument(1))
    assert list(tree1.children(tree1.begin())) == []  # original untouched


def test_solve_raises_immediately():
    ct = _bool_ctable({True: {_A: 1}})
    solver = _make_solver(ct)
    with pytest.raises(NotImplementedError, match="enum_effective_bscore"):
        solver.solve()
