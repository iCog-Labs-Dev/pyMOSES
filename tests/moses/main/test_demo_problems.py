"""C++: tests/moses/MOSESUTest.cxxtest, transliterated (plus unit coverage
for pymetta/moses/main/demo_problems.py that has no direct C++ oracle).

This is the ONLY new file this task's own brief authorizes touching under
`tests/moses/main/` (that directory is shared with a concurrent agent
owning `problem_params.py`'s own tests -- see this task's brief).

## What actually reproduces a previously-blocked `MOSESUTest` case, and what does not

Seven `MOSESUTest` cases have been blocked since S11 on two missing
dependencies this task ports: `demo-problems.cc` itself and `logical_bscore`
(see `demo_problems.py`'s own module docstring). Measured here, live,
against `$CLASSIC/build/moses/moses/main/moses` (bounded: `( ulimit -v
2097152; timeout <n> ./build/moses/moses/main/moses ... )`), NOT merely
inferred from `docs/CARRY-FORWARD.md`'s summary that classic's own
`MOSESUTest` passes 22/23:

* `test_1_mux` (`-Hmux -k1`) -- REPRODUCES. Classic (`-r1`, sorted output)
  finds `and(or(!$1 $3) or($1 $2))` at score 0; this port, same seed, same
  parameters, finds the IDENTICAL tree.
* `test_2_mux` (`-Hmux -k2`) -- REPRODUCES the SCORE (0), which is all
  `moses_test_score` checks -- classic finds a different 0-scoring tree
  than this port does (both are valid optima; `moses_test_score` never
  compares trees, only `fabs(score - 0) < 1e-8`).
* `test_cp_simple` (`-Hcp -yand($1 $2)`) -- REPRODUCES exactly: classic
  and this port both return the tree `and($1 $2)` (the input program
  itself), matching `moses_test_combo`'s tree-equality check.
* `test_cp` (`-Hcp -yor(and($2 !$3) $1 !$4 $5) -E3`) -- REPRODUCES
  exactly: classic and this port both return
  `or(and($2 !$3) $1 !$4 $5)` (again the input program itself).
* `test_2_parity`, `test_2_parity_with_labels` (`-Hpa -k2`[`-W1`]) --
  REPRODUCE the score (0). These were blocked when this file was written;
  the cause was NOT in `demo_problems.py`/`logical_bscore`, and not in
  `local_moses`/`hill_climbing`/`metapopulation` either, but in
  `build_knobs::sample_logical_perms`' unported random PAIR perms
  (`build_knobs.cc:435-518`) plus this port's then-default
  `perm_ratio = -1` -- see
  `tests/moses/test_pa_parity_regression.py` and
  `.superpowers/sdd/2026-09-08-pymetta-s13-cli/parity-divergence-report.md`.
  `-W1` only changes what classic PRINTS (labelled vs `$n` argument
  names), not what it searches for or finds, so `test_2_parity_with_labels`
  is covered by the identical assertion as `test_2_parity`.
  `test_3_parity_multi_thread` (`-Hpa -k3 -j4`) additionally uses `-j4`
  (multi-threaded evaluation; `distributed_moses.py`/threading are not part
  of this port), so it is not asserted here as an oracle -- but the
  single-threaded `-Hpa -k3` search it wraps now reaches score 0 and finds
  classic's own `-r1` tree exactly (measured; too slow at ~9.8k evals to
  run on every commit).

`sr` and `cp`'s contin arm were BOTH unblocked on 2026-09-09 (the reason
this docstring used to give -- "S14 contin territory, `contin_bscore`/
`ITable`/`OTable` not built, `problem_params.contin_reduct` is still
`None`" -- was false in every clause by the time it was read). The live,
byte-for-byte oracles for both live in `test_moses_exec.py`'s own
`-H sr`/`-H cp` block, since they are CLI-level runs; what stays here is
the unit-level wiring, including the two things those oracles cannot
isolate: `simple_symbolic_regression` itself, and
`polynomial_problem.run`'s `deme_params.linear_contin = False` override.
"""

from __future__ import annotations

import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.contin_reduction import contin_reduction
from pymetta.comboreduct.reduct.logical_reduction import logical_reduction
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    type_node,
    type_tree_arity,
)
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.main import demo_problems
from pymetta.moses.main.demo_problems import (
    combo_problem,
    demo_params,
    disjunction,
    even_parity,
    logical_bscore,
    majority,
    majority_problem,
    multiplex,
    mux_problem,
    pa_problem,
    polynomial_problem,
    register_demo_problems,
    simple_symbolic_regression,
)
from pymetta.moses.main.problem import problem_manager
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.moses.moses_params import moses_parameters
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.optimization.particle_swarm import ps_parameters
from pymetta.moses.scoring.bscores import ctruth_table_bscore
from pymetta.moses.scoring.scoring_base import bscore_base
from pymetta.util.rng import randGen

# ---------------------------------------------------------------------------
# Shared fixtures. `_rule`/`_fresh_deme_parameters` mirror
# tests/moses/moses/test_moses_main.py's own helpers of the same names and
# purpose (duplicated rather than imported -- tests do not import from other
# tests' modules in this project); `_fake_params` is this file's own.
# ---------------------------------------------------------------------------


def _rule():
    return logical_reduction(())(2)


def _fresh_deme_parameters() -> deme_parameters:
    pa = deme_parameters()
    # Classic's own default (problem-params.cc:299, forwarded to
    # deme_params.perm_ratio at problem-params.cc:2265). This was -1.0
    # ("no pairs") until pair sampling was ported; -1.0 is what made
    # `pa` unsolvable here -- see tests/moses/test_pa_parity_regression.py.
    pa.perm_ratio = 0.0
    return pa


class _FakeParams:
    """Not a C++ type -- a minimal duck-typed stand-in for `problem_params`
    (`dynamic_cast<problem_params*>(ob)` in classic), carrying only the
    attributes `demo_problems.py`'s `run()` methods actually read. See
    `bool_problem_base.run`'s own docstring for why `.printer` stands in
    for classic's `.mmr_pa`.
    """

    def __init__(self, *, max_evals: int = 10000) -> None:
        self.exemplars: list[combo_tree] = []
        self.ignore_ops: tuple = ()
        self.bool_reduct = _rule()
        self.bool_reduct_rep = _rule()
        self.meta_params = metapop_parameters()
        self.opt_params = optim_parameters()
        self.hc_params = hc_parameters()
        self.ps_params = ps_parameters()
        self.deme_params = _fresh_deme_parameters()
        self.filter_params = subsample_deme_filter_parameters()
        self.moses_params = moses_parameters(max_evals=max_evals)
        self.noise = -1.0
        self.complexity_ratio = 3.5
        # `bool_problem_base.run` reads this to decide whether to emit
        # demo-problems.cc:99-100's "Feature selection is not supported for
        # the demo problems" warning. Classic only WARNS -- it does not
        # refuse the flag -- so `False` is the ordinary path.
        self.enable_feature_selection = False
        self.printer = lambda mp, dex, stats: None
        # -- contin-problem fields (`sr`, and `cp`'s contin arm). Values are
        # classic's own CLI defaults: problem-params.cc:150
        # (`default_nsamples(20)`), :227 (`nsamples`, -1), :1147/:1158
        # (`min_rand_input` 0.0 / `max_rand_input` 1.0), and
        # `--it-abs-err`'s own `false`.
        self.nsamples = -1
        self.default_nsamples = 20
        self.min_rand_input = 0.0
        self.max_rand_input = 1.0
        self.it_abs_err = False
        self.contin_reduct = contin_reduction(2, ())


# ---------------------------------------------------------------------------
# demo_params
# ---------------------------------------------------------------------------


def test_demo_params_defaults_match_cli():
    """C++: moses/moses/main/demo-problems.cc:53-59 demo_params::add_options

    `-y`/`--combo-program` defaults to an empty (default-constructed)
    `std::string`; `-k`/`--problem-size` defaults to 5
    (`->default_value(5)`).
    """
    dp = demo_params()
    assert dp.combo_str == ""
    assert dp.problem_size == 5


# ---------------------------------------------------------------------------
# set_noise_or_ratio
# ---------------------------------------------------------------------------


def test_set_noise_or_ratio_uses_alphabet_size_branch_when_noise_non_negative():
    """C++: moses/moses/main/demo-problems.cc:73 set_noise_or_ratio

    Mutation-proven: inverting this `>=` to `<` makes this test (and
    `test_set_noise_or_ratio_uses_ratio_branch_when_noise_negative` below)
    fail -- see this task's own report for the log.
    """
    scorer = ctruth_table_bscore.__new__(ctruth_table_bscore)  # avoid needing a real CTable
    bscore_base.__init__(scorer)
    calls = []
    scorer.set_complexity_coef = lambda *a: calls.append(a)

    demo_problems._set_noise_or_ratio(scorer, 5, 0.3, 3.5)
    assert calls == [(5, 0.3)]


def test_set_noise_or_ratio_uses_ratio_branch_when_noise_negative():
    scorer = ctruth_table_bscore.__new__(ctruth_table_bscore)
    bscore_base.__init__(scorer)
    calls = []
    scorer.set_complexity_coef = lambda *a: calls.append(a)

    demo_problems._set_noise_or_ratio(scorer, 5, -1.0, 3.5)
    assert calls == [(3.5,)]


# ---------------------------------------------------------------------------
# _alphabet_size
# ---------------------------------------------------------------------------


def test_alphabet_size_boolean_type():
    """C++: moses/moses/main/problem.cc:145-147 alphabet_size (boolean_type)

    Mutation-proven: changing `3 + arity` to `2 + arity` makes this test
    fail (see this task's own report).
    """
    sig = TypeSignature((type_node.boolean_type,) * 4, type_node.boolean_type)
    assert demo_problems._alphabet_size(sig, ()) == 7


def test_alphabet_size_contin_type_subtracts_ignore_ops():
    sig = TypeSignature((type_node.contin_type,) * 2, type_node.contin_type)
    assert demo_problems._alphabet_size(sig, (builtin.sin, builtin.log)) == 8 + 2 - 2


def test_alphabet_size_ann_type():
    sig = TypeSignature((type_node.contin_type,) * 3, type_node.ann_type)
    assert demo_problems._alphabet_size(sig, ()) == 2 + 3 * 3


def test_alphabet_size_enum_type_raises_not_implemented():
    sig = TypeSignature((type_node.boolean_type,), type_node.enum_type)
    with pytest.raises(NotImplementedError, match="enum_type"):
        demo_problems._alphabet_size(sig, ())


def test_alphabet_size_unsupported_type_raises_value_error():
    """C++: moses/moses/main/problem.cc:186-195 alphabet_size (default arm)

    Ported as `ValueError`, not classic's `exit(1)` -- see
    `_alphabet_size`'s own docstring.
    """
    sig = TypeSignature((), type_node.lambda_type)
    with pytest.raises(ValueError, match="not supported"):
        demo_problems._alphabet_size(sig, ())


# ---------------------------------------------------------------------------
# _type_to_exemplar
# ---------------------------------------------------------------------------


def test_type_to_exemplar_boolean():
    tr = demo_problems._type_to_exemplar(type_node.boolean_type)
    assert combo_tree_to_str(tr).strip() == "and"


def test_type_to_exemplar_contin():
    tr = demo_problems._type_to_exemplar(type_node.contin_type)
    assert combo_tree_to_str(tr).strip() == "+"


def test_type_to_exemplar_unsupported_raises():
    with pytest.raises(NotImplementedError):
        demo_problems._type_to_exemplar(type_node.enum_type)


# ---------------------------------------------------------------------------
# even_parity / disjunction / multiplex / majority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ((False, False), True),
        ((True, False), False),
        ((False, True), False),
        ((True, True), True),
        ((True, True, True), False),  # 3 trues -- odd -- not even parity
        ((True, True, False), True),  # 2 trues -- even
    ],
)
def test_even_parity(bits, expected):
    """C++: moses/moses/example-progs/scoring_iterators.h:44 even_parity"""
    assert even_parity(bits) is expected


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ((False, False), False),
        ((True, False), True),
        ((False, True), True),
        ((True, True), True),
        ((False, False, False), False),
    ],
)
def test_disjunction(bits, expected):
    """C++: moses/moses/example-progs/scoring_iterators.h:56 disjunction"""
    assert disjunction(bits) is expected


def test_multiplex_selects_addressed_bit():
    """C++: moses/moses/example-progs/scoring_iterators.h:68 multiplex

    arity=2 address bits select one of 4 data bits: address (1,0) -> addr
    1 -> selects data bit at index arity+1.
    """
    mux = multiplex(2)
    # address bits (a0=1,a1=0) -> addr=1; data bits d0..d3 = 0,1,0,0
    assert mux([True, False, 0, 1, 0, 0]) is True
    assert mux([False, False, 0, 1, 0, 0]) == 0  # addr=0 -> d0=0
    assert mux([False, True, 0, 1, 0, 0]) == 0  # addr=2 -> d2=0


def test_majority_strict_majority_of_true():
    """C++: moses/moses/example-progs/scoring_iterators.h:94 majority

    arity=5, floor(5/2)=2: exactly 2 true is NOT a majority (2 > 2 is
    False); 3 true IS.
    """
    maj = majority(5)
    assert maj([True, True, False, False, False]) is False
    assert maj([True, True, True, False, False]) is True


def test_iterator_function_truth_table_bit_order():
    """C++: moses/comboreduct/table/table.h:1458-1468 complete_truth_table(Func, arity)

    i=0..3 for arity=2 maps to ($1,$2) = (0,0),(1,0),(0,1),(1,1) -- bit j
    of i is `(i >> j) & 1`.
    """
    table = demo_problems._iterator_function_truth_table(even_parity, 2)
    assert table == [True, False, False, True]


# ---------------------------------------------------------------------------
# logical_bscore
# ---------------------------------------------------------------------------


def test_logical_bscore_from_func_and_from_equivalent_tree_agree():
    """`logical_bscore(even_parity, 2)` and `logical_bscore(<xnor tree>, 2)`
    must compute the identical target truth table, since both describe
    the same boolean function.
    """
    from_func = logical_bscore(even_parity, 2)
    xnor_tree = combo_tree(builtin.logical_or)
    left = xnor_tree.append_child(xnor_tree.begin(), builtin.logical_and)
    xnor_tree.append_child(left, argument(1))
    xnor_tree.append_child(left, argument(2))
    right = xnor_tree.append_child(xnor_tree.begin(), builtin.logical_and)
    xnor_tree.append_child(right, argument(-1))
    xnor_tree.append_child(right, argument(-2))
    from_tree = logical_bscore(xnor_tree, 2)
    assert from_func._target == from_tree._target == [True, False, False, True]


def test_logical_bscore_call_scores_wrong_rows_negative_one():
    """C++: moses/moses/scoring/bscores.cc:67 logical_bscore::operator()

    Target is even_parity(2) = [T,F,F,T] (see test above). Candidate
    `and($1 $2)` predicts [F,F,F,T] -- wrong only on row 0.

    Mutation-proven: swapping the `-` sign (return `+score_t(...)`
    instead) makes this test fail (see this task's own report).
    """
    bsc = logical_bscore(even_parity, 2)
    tr = combo_tree(builtin.logical_and)
    tr.append_child(tr.begin(), argument(1))
    tr.append_child(tr.begin(), argument(2))
    assert bsc(tr) == [-1.0, 0.0, 0.0, 0.0]


def test_logical_bscore_best_and_worst_possible():
    bsc = logical_bscore(even_parity, 2)
    assert bsc.best_possible_bscore() == [0.0, 0.0, 0.0, 0.0]
    assert bsc.worst_possible_bscore() == [-1.0, -1.0, -1.0, -1.0]


def test_logical_bscore_min_improv():
    """C++: moses/moses/scoring/bscores.cc:125 logical_bscore::min_improv"""
    assert logical_bscore(even_parity, 2).min_improv() == 0.5


def test_logical_bscore_get_error():
    """C++: moses/moses/scoring/bscores.cc:130 logical_bscore::get_error"""
    bsc = logical_bscore(even_parity, 2)
    assert bsc.get_error([-1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)


def test_logical_bscore_call_ensemble_not_overridden_raises():
    """`logical_bscore` does not override `call_ensemble` (boosting is out
    of scope for this whole port cycle) -- falls back to
    `bscore_base.call_ensemble`'s own unconditional raise.
    """
    bsc = logical_bscore(even_parity, 2)
    with pytest.raises(NotImplementedError, match="Ensemble scoring"):
        bsc.call_ensemble(None)


# ---------------------------------------------------------------------------
# pa / dj / maj / mux -- name/description/get_arity, and get_bscore wiring.
# ---------------------------------------------------------------------------


def test_pa_problem_identity():
    dp = demo_params()
    prob = pa_problem(dp)
    assert prob.name() == "pa"
    assert prob.get_arity(4) == 4
    assert prob.get_bscore(2)._target == [True, False, False, True]


def test_mux_problem_get_arity():
    dp = demo_params()
    prob = mux_problem(dp)
    assert prob.get_arity(2) == 2 + 4  # sz + 2**sz


def test_mux_problem_get_arity_rejects_size_above_four():
    """C++: moses/moses/main/demo-problems.cc:202-213 mux_problem::get_arity

    Mutation-proven: changing `sz > 4` to `sz > 5` makes this test fail
    (see this task's own report).
    """
    dp = demo_params()
    prob = mux_problem(dp)
    with pytest.raises(ValueError, match="maximum mux demo problem size is 4"):
        prob.get_arity(5)


def test_majority_problem_get_bscore_matches_majority_func():
    dp = demo_params()
    prob = majority_problem(dp)
    bsc = prob.get_bscore(3)
    assert bsc._target == demo_problems._iterator_function_truth_table(majority(3), 3)


# ---------------------------------------------------------------------------
# bool_problem_base.run -- wiring, exemplar defaulting, boosting gate.
# (monkeypatches `metapop_moses_results` so these are fast unit tests, not
# live searches -- the live searches are the oracle tests further down.)
# ---------------------------------------------------------------------------


def test_bool_problem_base_run_defaults_exemplar_when_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda *args: calls.append(args),
    )
    dp = demo_params(problem_size=2)
    prob = pa_problem(dp)
    pms = _FakeParams()
    assert pms.exemplars == []

    prob.run(pms)

    assert len(pms.exemplars) == 1
    assert combo_tree_to_str(pms.exemplars[0]).strip() == "and"
    assert len(calls) == 1


def test_bool_problem_base_run_does_not_override_existing_exemplar(monkeypatch):
    monkeypatch.setattr(demo_problems, "metapop_moses_results", lambda *args: None)
    dp = demo_params(problem_size=2)
    prob = pa_problem(dp)
    pms = _FakeParams()
    custom = combo_tree(builtin.logical_or)
    pms.exemplars = [custom]

    prob.run(pms)

    assert pms.exemplars == [custom]


def test_bool_problem_base_run_raises_for_boosting(monkeypatch):
    """`bool_problem_base::run` (demo-problems.cc:109) calls
    `bscore.use_weighted_scores()` when boosting is requested -- not
    implemented (see `run`'s own docstring); ported as an explicit raise.

    Mutation-proven: deleting this guard makes this test fail with an
    `AttributeError` instead of the expected `NotImplementedError` (see
    this task's own report).
    """
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda *args: pytest.fail("must not reach metapop_moses_results"),
    )
    dp = demo_params(problem_size=2)
    prob = pa_problem(dp)
    pms = _FakeParams()
    pms.meta_params.do_boosting = True

    with pytest.raises(NotImplementedError, match="boost"):
        prob.run(pms)


def test_bool_problem_base_run_passes_pms_ps_params_in_C_PLUS_PLUS_position(monkeypatch):
    """MUTATION PROVEN. `demo-problems.cc:117-120` calls
    `metapop_moses_results(..., pms.opt_params, pms.hc_params,
    pms.ps_params, pms.deme_params, ...)`; this port dropped
    `ps_params` from all four `demo_problems.py` call sites until
    2026-09-09. Position 7 is asserted, not mere presence -- a swap with
    `hc_params` type-checks fine in Python and would hand
    `particle_swarm` an `hc_parameters`.
    """
    calls = []
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda *args: calls.append(args),
    )
    prob = pa_problem(demo_params(problem_size=2))
    pms = _FakeParams()

    prob.run(pms)

    assert len(calls) == 1
    assert calls[0][5] is pms.opt_params
    assert calls[0][6] is pms.hc_params
    assert calls[0][7] is pms.ps_params
    assert calls[0][8] is pms.deme_params


def test_bool_problem_base_run_passes_correct_signature_and_complexity_coef(monkeypatch):
    captured = {}

    def fake_metapop_moses_results(exemplars, sig, si_ca, si_kb, cscore, *rest):
        captured["sig"] = sig
        captured["cscore"] = cscore

    monkeypatch.setattr(demo_problems, "metapop_moses_results", fake_metapop_moses_results)
    dp = demo_params(problem_size=2)
    prob = pa_problem(dp)
    pms = _FakeParams()

    prob.run(pms)

    assert captured["sig"].otype == type_node.boolean_type
    assert len(captured["sig"].itypes) == 2
    bscorer = captured["cscore"]._bscorer if hasattr(captured["cscore"], "_bscorer") else None
    # alphabet_size(boolean, arity=2) = 5; complexity_ratio path (noise=-1)
    # sets `set_complexity_coef(3.5)` -- confirm SOME non-zero coefficient
    # landed (exact value is `discrete_complexity_coef`'s own concern, not
    # this wiring test's).
    if bscorer is not None:
        assert bscorer.get_complexity_coef() != 0.0


# ---------------------------------------------------------------------------
# combo_problem_base.check_args
# ---------------------------------------------------------------------------


def test_check_args_raises_when_combo_str_empty():
    dp = demo_params(combo_str="")
    base = combo_problem(dp)
    with pytest.raises(ValueError, match="You must specify a combo tree"):
        base.check_args()


def test_check_args_raises_when_no_dollar_in_combo_str():
    """C++: moses/moses/main/demo-problems.cc:337 combo_problem_base::check_args

    Mutation-proven: removing this check makes this test fail (see this
    task's own report).
    """
    dp = demo_params(combo_str="true")
    base = combo_problem(dp)
    with pytest.raises(ValueError, match="forget to escape"):
        base.check_args()


def test_check_args_raises_for_unparseable_combo_str():
    """A genuinely unparseable `combo_str` (mismatched parens) fails
    inside `str_to_combo_tree` itself, unwrapped -- see `check_args`'s own
    docstring for why this is a disclosed, stricter-than-classic failure
    mode, not this method's own "not well formed" message.
    """
    dp = demo_params(combo_str="and($1 $2")  # unbalanced parens
    base = combo_problem(dp)
    with pytest.raises(ValueError, match="unterminated"):
        base.check_args()


def test_check_args_not_well_formed_branch(monkeypatch):
    """`check_args`'s own "not well formed" `ValueError` is currently
    unreachable via any real `combo_str` (see `check_args`'s own
    docstring) -- proven here directly via a monkeypatch of
    `_infer_output_type_and_arity` instead, showing the `except` clause
    itself is wired correctly.

    Mutation-proven: deleting this `try`/`except` makes this test fail
    with the injected `AssertionError` instead of the expected
    `ValueError` (see this task's own report).
    """
    monkeypatch.setattr(
        demo_problems,
        "_infer_output_type_and_arity",
        lambda tr: (_ for _ in ()).throw(AssertionError("boom")),
    )
    dp = demo_params(combo_str="and($1 $2)")
    base = combo_problem(dp)
    with pytest.raises(ValueError, match="not well formed"):
        base.check_args()


def test_check_args_returns_signature_for_boolean_program():
    dp = demo_params(combo_str="and($1 $2)")
    base = combo_problem(dp)
    tr, tt = base.check_args()
    assert tt == TypeSignature(
        (type_node.boolean_type, type_node.boolean_type), type_node.boolean_type
    )
    assert combo_tree_to_str(tr).strip() == "and($1 $2)"


def test_check_args_returns_signature_for_contin_program():
    dp = demo_params(combo_str="+(6 *(6 $1))")
    base = combo_problem(dp)
    _tr, tt = base.check_args()
    assert tt == TypeSignature((type_node.contin_type,), type_node.contin_type)


def test_check_args_arity_is_max_argument_index_not_count():
    """`argument.idx` can skip -- "or($1 !$3)" has TWO arguments but arity
    3 (matching `type_tree_arity`'s "how many input slots" meaning, not
    "how many `$n` tokens appear"), and the unmentioned slot is
    `unknown_type`, exactly as classic's own `infer_type_tree` reports:
    measured against a compiled probe, `or($1 !$3)` is
    `->(boolean unknown boolean boolean)`.
    """
    dp = demo_params(combo_str="or($1 !$3)")
    base = combo_problem(dp)
    _tr, tt = base.check_args()
    assert type_tree_arity(tt) == 3
    assert tt == TypeSignature(
        (type_node.boolean_type, type_node.unknown_type, type_node.boolean_type),
        type_node.boolean_type,
    )


def test_check_args_rejects_a_cross_argument_type_mismatch():
    """`+(impulse($1) $1)` uses `$1` in boolean and contin position at
    once. Classic's own `infer_type_tree` returns `->(ill_formed
    ill_formed)` with `is_well_formed` false (measured against the probe),
    and the binary prints
    "The combo tree +(impulse($1) $1) is not well formed." and exits -1.

    Mutation-proven: deleting the `if not _is_well_formed(tt)` guard makes
    this test fail (see this task's report for the kill count) -- the
    monkeypatch-driven `test_check_args_not_well_formed_branch` below
    covers a DIFFERENT arm (`get_output_type_tree` raising) and does not
    substitute for it.
    """
    dp = demo_params(combo_str="+(impulse($1) $1)")
    base = combo_problem(dp)
    with pytest.raises(ValueError, match="is not well formed"):
        base.check_args()


def test_infer_signature_leaves_a_root_argument_untyped():
    """A bare `$1` has no enclosing expression to pin its type down, and
    classic agrees: `infer_type_tree("$1")` is `->(unknown unknown)`,
    `wf=1` (measured). Not an error -- `combo_problem.run`'s final `else`
    is what rejects it, and only because `unknown_type` is neither
    boolean nor contin.
    """
    tt = demo_problems._infer_signature(str_to_combo_tree("$1"))
    assert tt == TypeSignature((type_node.unknown_type,), type_node.unknown_type)


def test_infer_signature_raises_for_a_position_dependent_builtin():
    """`cond`'s children alternate predicate/branch types, so it has no
    `_BUILTIN_INPUT_TYPE` entry and an argument under it raises rather
    than being guessed at -- see the module docstring's own
    "missing general type-inference engine" section.
    """
    tr = str_to_combo_tree("cond($1 1 2)")
    with pytest.raises(NotImplementedError, match="no input-type entry"):
        demo_problems._infer_signature(tr)


def test_is_well_formed_matches_classic_on_both_answers():
    """C++: moses/comboreduct/type_checker/type_tree.cc:1210 is_well_formed

    "for (it : tt) if (*it == id::ill_formed_type) return false" -- an
    `ill_formed` anywhere, input slot or output, is enough.
    """
    good = TypeSignature((type_node.boolean_type,), type_node.boolean_type)
    bad_input = TypeSignature((type_node.ill_formed_type,), type_node.contin_type)
    bad_output = TypeSignature((type_node.contin_type,), type_node.ill_formed_type)
    assert demo_problems._is_well_formed(good)
    assert not demo_problems._is_well_formed(bad_input)
    assert not demo_problems._is_well_formed(bad_output)


# ---------------------------------------------------------------------------
# combo_problem.run
# ---------------------------------------------------------------------------


def test_combo_problem_run_boolean_wiring(monkeypatch):
    captured = {}

    def fake_metapop_moses_results(exemplars, sig, si_ca, si_kb, cscore, *rest):
        captured["sig"] = sig
        captured["exemplars"] = exemplars

    monkeypatch.setattr(demo_problems, "metapop_moses_results", fake_metapop_moses_results)
    dp = demo_params(combo_str="and($1 $2)")
    prob = combo_problem(dp)
    pms = _FakeParams()

    prob.run(pms)

    assert captured["sig"].otype == type_node.boolean_type
    assert len(captured["sig"].itypes) == 2
    assert combo_tree_to_str(captured["exemplars"][0]).strip() == "and"


def test_combo_problem_run_contin_wiring(monkeypatch):
    """The contin arm's "stupid hack" (demo-problems.cc:403-421), unit
    level. With `-y '+(6 *(6 $1))'` and both range flags at their CLI
    defaults (0.0/1.0) it must set `max_rand_input = 2 * 6 = 12`,
    `min_rand_input = -12`, and -- because `default_nsamples` (20) is
    less than `2 * arity * max_rand_input` (24) -- `nsamples = 24`, the
    TRUNCATION of a `contin_t` into an `int` member.
    """
    captured = {}

    def fake_metapop_moses_results(exemplars, sig, si_ca, si_kb, cscore, *rest):
        captured["sig"] = sig
        captured["cscore"] = cscore
        captured["reduct"] = (si_ca, si_kb)

    monkeypatch.setattr(demo_problems, "metapop_moses_results", fake_metapop_moses_results)
    dp = demo_params(combo_str="+(6 *(6 $1))")
    prob = combo_problem(dp)
    pms = _FakeParams()

    prob.run(pms)

    assert pms.max_rand_input == 12.0
    assert pms.min_rand_input == -12.0
    assert pms.nsamples == 24
    assert captured["sig"] == TypeSignature((type_node.contin_type,), type_node.contin_type)
    # classic passes `*pms.contin_reduct` for BOTH reduct arguments
    # (demo-problems.cc:437-438), not bool_reduct/bool_reduct_rep.
    assert captured["reduct"] == (pms.contin_reduct, pms.contin_reduct)
    assert combo_tree_to_str(pms.exemplars[0]).strip() == "+"


def test_combo_problem_run_contin_leaves_an_explicit_range_alone(monkeypatch):
    """The hack fires ONLY when both range flags are still at their
    defaults (`0.0 == min && 1.0 == max`). `-q0 -w10` must survive
    untouched, and `nsamples` must fall through to `default_nsamples`
    rather than to the `2 * arity * max_rand_input` formula.

    Mutation-proven: dropping the `and pms.max_rand_input == 1.0`
    conjunct makes this test fail (see this task's report).
    """
    monkeypatch.setattr(demo_problems, "metapop_moses_results", lambda *a: None)
    dp = demo_params(combo_str="*(2 $1)")
    prob = combo_problem(dp)
    pms = _FakeParams()
    pms.min_rand_input = 0.0
    pms.max_rand_input = 10.0

    prob.run(pms)

    assert pms.min_rand_input == 0.0
    assert pms.max_rand_input == 10.0
    assert pms.nsamples == 20


def test_combo_problem_run_contin_sampling_range_is_swapped(monkeypatch):
    """DIVERGENCE-pinning test for demo-problems.cc:429. The constructor
    is `ITable(tt, nsamples, min_contin, max_contin)` and classic calls it
    with `(tt, nsamples, max_rand_input, min_rand_input)`, so with
    `-q0 -w10` the sampled column is `10 - 10r`, i.e. it DECREASES as the
    RNG draw increases, and never exceeds `max_rand_input`.

    Checked against the RNG directly rather than against a golden list:
    seed 1's first `randdouble()` is 0.417021998534217 (pinned in
    tests/util/test_rng.py), so the first sampled input must be
    `10 - 10 * 0.417021998534217`. Passing the two bounds the "sensible"
    way round would give `10 * 0.417021998534217` instead.
    """
    captured = {}

    def fake_metapop_moses_results(exemplars, sig, si_ca, si_kb, cscore, *rest):
        captured["cscore"] = cscore

    monkeypatch.setattr(demo_problems, "metapop_moses_results", fake_metapop_moses_results)
    dp = demo_params(combo_str="*(2 $1)")
    prob = combo_problem(dp)
    pms = _FakeParams()
    pms.min_rand_input = 0.0
    pms.max_rand_input = 10.0
    randGen().seed(1)

    prob.run(pms)

    first_row = captured["cscore"]._bscorer._cti[0]
    assert first_row[0] == 10.0 - 10.0 * 0.417021998534217


def test_combo_problem_run_raises_for_boosting():
    dp = demo_params(combo_str="and($1 $2)")
    prob = combo_problem(dp)
    pms = _FakeParams()
    pms.meta_params.do_boosting = True

    with pytest.raises(NotImplementedError, match="Boosting not supported"):
        prob.run(pms)


def test_combo_problem_run_unsupported_output_type_raises_value_error():
    """`combo_str="$1"` alone: root is a bare `argument`, whose OWN output
    type is `unknown_type` (`get_output_type_tree`'s own docstring) --
    neither boolean nor contin, so `combo_problem.run`'s final `else`
    fires. `pms.exemplars` is pre-populated so `_type_to_exemplar` (which
    cannot handle `unknown_type` either, matching classic's own
    `type_to_exemplar`'s `default:` `OC_ASSERT` branch) is never called --
    isolating the `else` branch this test actually targets, exactly as
    classic's real control flow would never reach the "not supported"
    `else` for THIS combo_str either (it would abort inside
    `type_to_exemplar` first) when exemplars is empty.
    """
    dp = demo_params(combo_str="$1")
    prob = combo_problem(dp)
    pms = _FakeParams()
    pms.exemplars = [combo_tree(builtin.logical_and)]

    with pytest.raises(ValueError, match="not supported"):
        prob.run(pms)


def test_ann_combo_problem_raises_without_validating_combo_str():
    """See `ann_combo_problem.run`'s own docstring: it must raise
    `NotImplementedError` even for an EMPTY `combo_str` that `check_args`
    would reject with `ValueError` -- proving `check_args` is never
    called.
    """
    from pymetta.moses.main.demo_problems import ann_combo_problem

    dp = demo_params(combo_str="")
    prob = ann_combo_problem(dp)
    pms = _FakeParams()

    with pytest.raises(NotImplementedError, match="ann_combo_problem"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# simple_symbolic_regression / _dorepeat_count / _largest_const_in_tree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "x", "expected"),
    [
        (0, 3.0, 0.0),  # f(x)_0 = 0
        (1, 3.0, 3.0),  # f(x)_1 = x
        (2, 3.0, 12.0),  # f(x)_2 = x + x^2
        (3, 3.0, 39.0),  # f(x)_3 = x + x^2 + x^3
        (4, 2.0, 30.0),  # the constructor's own default order
        (3, -1.0, -1.0),  # -1 + 1 - 1
        (3, 0.0, 0.0),
    ],
)
def test_simple_symbolic_regression_is_the_polynomial_it_claims(order, x, expected):
    """C++: examples/example-progs/scoring_iterators.h:110

    "f(x)_o = sum_{i={1,o}} x^i" (scoring_iterators.h:104-108's own
    comment). The Horner form `res = (res + 1) * x`, run `order` times,
    must equal that sum -- checked against the sum, not against itself.
    """
    assert simple_symbolic_regression(order)([x]) == expected
    assert expected == sum(x**i for i in range(1, order + 1))


def test_simple_symbolic_regression_reads_only_the_first_input():
    """`res = (res + 1) * get_contin(*from)` never advances `from`, and
    never touches `to`. Extra columns must be ignored entirely -- which
    is why an arity-1 signature and a wider one give the same answer.
    """
    func = simple_symbolic_regression(3)
    assert func([2.0]) == func([2.0, 99.0, -7.0])


def test_simple_symbolic_regression_default_order_is_four():
    """`simple_symbolic_regression(int o = 4)` -- scoring_iterators.h:112.
    Never used by `polynomial_problem::run` (which always passes
    `problem_size`), but it is part of the C++ declaration.
    """
    assert simple_symbolic_regression().order == 4


def test_dorepeat_counts_in_unsigned_int():
    """C++: opencog/util/dorepeat.h:41

    `for (unsigned int i = N; i > 0; --i)`. A negative `N` is NOT zero
    iterations (Python's `range(-1)`) but `N mod 2**32`. Extracted from
    the loop so it can be tested at all: `simple_symbolic_regression(-1)`
    would actually run 4294967295 times per row -- which is exactly what
    classic does, measured (`moses -Hsr -k-1 -m200 -W1` under
    `timeout 20` produces no output and exits 124).

    Mutation-proven: replacing `n & 0xFFFFFFFF` with a plain `n` makes
    this test fail (see this task's report).
    """
    assert demo_problems._dorepeat_count(0) == 0
    assert demo_problems._dorepeat_count(5) == 5
    assert demo_problems._dorepeat_count(-1) == 0xFFFFFFFF
    assert demo_problems._dorepeat_count(-2) == 0xFFFFFFFE


@pytest.mark.parametrize(
    ("combo_str", "expected"),
    [
        ("+(6 *(6 $1))", 6.0),
        ("+(3 *(2 $1) *(4 $2))", 4.0),
        ("*($1 $1)", 0.0),  # no contin constants at all -> 0.0, not a sentinel
        ("+(-3 *(-9 $1))", 0.0),  # all negative -> still 0.0, NOT -3
        ("+(0.5 $1)", 0.5),
    ],
)
def test_largest_const_in_tree(combo_str, expected):
    """C++: moses/moses/main/demo-problems.cc:347 largest_const_in_tree

    `contin_t rc = 0.0; if (rc < val) rc = val;` -- the 0.0 seed means an
    all-negative tree answers 0.0. That is not a rounding detail: it is
    what makes `-y'+(-3 *(-9 $1))'` sample the degenerate range
    `[-0.0, 0.0]` in classic.
    """
    assert demo_problems._largest_const_in_tree(str_to_combo_tree(combo_str)) == expected


# ---------------------------------------------------------------------------
# sr
# ---------------------------------------------------------------------------


def test_polynomial_problem_identity():
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)
    assert prob.name() == "sr"
    assert prob.description() == "Simple regression of f_n(x) = sum_{k={1,n}} x^k"


def test_polynomial_problem_run_wiring(monkeypatch):
    captured = {}

    def fake_metapop_moses_results(exemplars, sig, si_ca, si_kb, cscore, *rest):
        captured["sig"] = sig
        captured["reduct"] = (si_ca, si_kb)
        captured["cscore"] = cscore

    monkeypatch.setattr(demo_problems, "metapop_moses_results", fake_metapop_moses_results)
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)
    pms = _FakeParams()

    prob.run(pms)

    # gen_signature(id::contin_type, 1) -- demo-problems.cc:261.
    assert captured["sig"] == TypeSignature((type_node.contin_type,), type_node.contin_type)
    # *pms.contin_reduct for BOTH arguments -- demo-problems.cc:436-437.
    assert captured["reduct"] == (pms.contin_reduct, pms.contin_reduct)
    # the default contin exemplar, `+` -- demo-problems.cc:255.
    assert combo_tree_to_str(pms.exemplars[0]).strip() == "+"
    # nsamples <= 0 falls through to default_nsamples (20), so the table
    # has 20 rows -- demo-problems.cc:263.
    assert len(captured["cscore"]._bscorer._cti) == 20


def test_polynomial_problem_run_overrides_linear_contin(monkeypatch):
    """demo-problems.cc:257-259: "sr is fundamentally a kind of non-linear
    regression! over-ride any flag settings regarding this."
    `problem_params.parse_options` has already set
    `deme_params.linear_contin = linear_regression`, so a
    `--linear-regression=1` run arrives here with it True; `run` must put
    it back to False before knob building sees it.

    Mutation-proven: deleting the assignment makes this test fail, and so
    does `test_moses_exec_sr_overrides_linear_contin` end to end (see this
    task's report for both kill counts).
    """
    monkeypatch.setattr(demo_problems, "metapop_moses_results", lambda *a: None)
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)
    pms = _FakeParams()
    pms.deme_params.linear_contin = True

    prob.run(pms)

    assert pms.deme_params.linear_contin is False


def test_polynomial_problem_run_respects_explicit_nsamples(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda exemplars, sig, a, b, cscore, *rest: captured.update(cscore=cscore),
    )
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)
    pms = _FakeParams()
    pms.nsamples = 7

    prob.run(pms)

    assert len(captured["cscore"]._bscorer._cti) == 7


def test_polynomial_problem_run_raises_for_boosting(monkeypatch):
    """`OC_ASSERT(not pms.meta_params.do_boosting, "Boosting not supported
    for this problem!")` -- demo-problems.cc:272. Note it sits AFTER the
    scorer is built in C++, so the ITable is already drawn by then; the
    port keeps that order, which this test does not depend on but the
    RNG stream does.
    """
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda *a: pytest.fail("must not reach metapop_moses_results"),
    )
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)
    pms = _FakeParams()
    pms.meta_params.do_boosting = True

    with pytest.raises(NotImplementedError, match="Boosting not supported"):
        prob.run(pms)


def test_polynomial_problem_run_selects_abs_error_from_it_abs_err(monkeypatch):
    """`contin_bscore::err_function_type eft = pms.it_abs_err ?
    abs_error : squared_error` -- demo-problems.cc:267-269. Checked
    behaviourally (the two error functions disagree on any non-unit
    difference) rather than by reading a private attribute.
    """
    captured = {}
    monkeypatch.setattr(
        demo_problems,
        "metapop_moses_results",
        lambda exemplars, sig, a, b, cscore, *rest: captured.update(cscore=cscore),
    )
    dp = demo_params(problem_size=3)
    prob = polynomial_problem(dp)

    pms = _FakeParams()
    prob.run(pms)
    squared = captured["cscore"]._bscorer._err_func(0.0, 3.0)

    pms = _FakeParams()
    pms.it_abs_err = True
    prob.run(pms)
    absolute = captured["cscore"]._bscorer._err_func(0.0, 3.0)

    assert squared == 9.0
    assert absolute == 3.0


# ---------------------------------------------------------------------------
# register_demo_problems
# ---------------------------------------------------------------------------


def test_register_demo_problems_registers_all_seven_names():
    pmr = problem_manager()
    register_demo_problems(pmr)
    for name in ("pa", "dj", "maj", "mux", "sr", "cp", "ann-cp"):
        assert pmr.find_problem(name) is not None, name


def test_register_demo_problems_returns_shared_demo_params():
    """The returned `demo_params` is the SAME instance every registered
    problem's `_dparms` holds -- mutating it through the handle affects
    every problem, exactly as classic's single shared `demo_params*` does.
    """
    pmr = problem_manager()
    dp = register_demo_problems(pmr)
    dp.problem_size = 7
    assert pmr.find_problem("pa")._dparms is dp
    assert pmr.find_problem("pa")._dparms.problem_size == 7


# ---------------------------------------------------------------------------
# Live oracle tests -- reproduce MOSESUTest.cxxtest cases, measured live
# against classic (see this file's own module docstring for the exact
# commands and outputs classic produced).
# ---------------------------------------------------------------------------


def _run_bool_problem(name: str, problem_size: int, *, max_evals: int = 10000, seed: int = 1):
    pmr = problem_manager()
    dp = register_demo_problems(pmr)
    dp.problem_size = problem_size
    prob = pmr.find_problem(name)
    pms = _FakeParams(max_evals=max_evals)
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)
    randGen().seed(seed)
    prob.run(pms)
    return result["mp"]


def _run_cp(combo_str: str, *, max_evals: int = 10000, seed: int = 1):
    pmr = problem_manager()
    dp = register_demo_problems(pmr)
    dp.combo_str = combo_str
    prob = pmr.find_problem("cp")
    pms = _FakeParams(max_evals=max_evals)
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)
    randGen().seed(seed)
    prob.run(pms)
    return result["mp"]


def test_1_mux_reproduces_mosesutest():
    """C++: tests/moses/MOSESUTest.cxxtest:69 test_1_mux
    (`moses_test_score({"-Hmux", "-k1"})`, expected score 0)

    Classic (`-Hmux -k1 -r1`, this task's own live measurement -- see
    module docstring) finds `and(or(!$1 $3) or($1 $2))` at score 0; this
    port, same seed, finds the IDENTICAL tree.
    """
    mp = _run_bool_problem("mux", 1)
    assert mp.best_score() == 0.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "and(or(!$1 $3) or($1 $2))"


def test_2_mux_reproduces_mosesutest():
    """C++: tests/moses/MOSESUTest.cxxtest:73 test_2_mux
    (`moses_test_score({"-Hmux", "-k2"})`, expected score 0)

    `moses_test_score` only checks the SCORE, never the tree -- classic
    and this port find different 0-scoring trees for this larger (arity 6)
    search, which is expected and does not affect this oracle.
    """
    mp = _run_bool_problem("mux", 2)
    assert mp.best_score() == 0.0


def test_cp_simple_reproduces_mosesutest():
    """C++: tests/moses/MOSESUTest.cxxtest:118 test_cp_simple
    (`moses_test_combo({"-Hcp", "-yand($1 $2)"}, {"and($1 $2)"})`)
    """
    mp = _run_cp("and($1 $2)")
    assert mp.best_score() == 0.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "and($1 $2)"


def test_cp_reproduces_mosesutest():
    """C++: tests/moses/MOSESUTest.cxxtest:127 test_cp
    (`moses_test_combo({"-Hcp", "-yor(and($2 !$3) $1 !$4 $5)", "-E3"},
    {"or(and($2 !$3) $1 !$4 $5)"})`)

    `-E3` is not modeled (no equivalent option surface reached by this
    port yet) -- both classic and this port converge to the exact input
    program regardless, so this does not affect the result.
    """
    mp = _run_cp("or(and($2 !$3) $1 !$4 $5)")
    assert mp.best_score() == 0.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "or(and($2 !$3) $1 !$4 $5)"


# ---------------------------------------------------------------------------
# dj / maj -- not MOSESUTest oracles (no `-Hdj`/`-Hmaj` case exists in
# MOSESUtest.cxxtest), but exercise this file's own dj_problem/
# majority_problem end to end at negligible extra cost.
# ---------------------------------------------------------------------------


def test_dj_k3_converges_to_perfect_score():
    mp = _run_bool_problem("dj", 3)
    assert mp.best_score() == 0.0


def test_maj_k3_converges_to_perfect_score():
    mp = _run_bool_problem("maj", 3)
    assert mp.best_score() == 0.0


# ---------------------------------------------------------------------------
# pa -- was blocked; unblocked by porting build_knobs' random pair perms.
# ---------------------------------------------------------------------------


def test_2_parity_reproduces_mosesutest():
    """C++: tests/moses/MOSESUTest.cxxtest:44 test_2_parity
    (`moses_test_score({"-Hpa", "-k2"})`, expected score 0)

    Also covers `test_2_parity_with_labels` (MOSESUTest.cxxtest:49,
    `{"-Hpa", "-k2", "-W1"}`): `-W1` changes only what classic prints.
    `moses_test_score` never compares trees, and classic itself prints two
    distinct score-0 trees for this problem, so only the score is asserted.
    The full trajectory match against classic (144 evals, one expansion) is
    pinned separately in `tests/moses/test_pa_parity_regression.py`.
    """
    mp = _run_bool_problem("pa", 2)
    assert mp.best_score() == 0.0
