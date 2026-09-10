"""Unit, gate and live-oracle tests for pymetta.moses.main.table_problems.

Four layers, in order:

1. `table_problem_params` -- the CLI surface `problem_params.py` deliberately
   does not carry (`-i`/`-u`/`--score-weight`/`--timestamp-feature`/`-Y`),
   parsed on the SAME `argparse.ArgumentParser` as `problem_params` itself
   (`parse_table_args`).
2. `table_problem_base.common_setup`/`common_type_setup` -- table loading,
   arity inference, exemplar defaulting, and every P2/S14 gate
   (`--nsamples`, `--balance`, feature selection, `contin_reduct`).
3. Each concrete `*_table_problem`'s own wiring -- name/description, the
   `hardness == 0` CLI fallback (`prerec`/`recall`/`bep`), and
   `f_one`/`select`'s faithfully-reproduced upstream crashes.
4. Live oracles, measured directly against classic's own compiled binary
   (`$CLASSIC/build/moses/moses/main/moses`), not against a probe this
   port configured -- see docs/CARRY-FORWARD.md 5.4 and this module's own
   test docstrings for the exact commands and outputs.

Bounded per this project's own memory-safety convention: every live-oracle
subprocess measurement behind these tests was run under
`( ulimit -v 2097152; timeout 300 <cmd> )`; the tests themselves run
in-process (no subprocess), driven by small datasets and small `-m` caps,
so they stay fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str
from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymetta.moses.main import table_problems as tp
from pymetta.moses.main.problem import _unported_table_problem, problem_manager
from pymetta.moses.main.table_problems import (
    bep_table_problem,
    cluster_table_problem,
    f_one_table_problem,
    it_table_problem,
    parse_table_args,
    pre_conj_table_problem,
    pre_table_problem,
    prerec_table_problem,
    recall_table_problem,
    register_table_problems,
    select_table_problem,
    table_problem_base,
    table_problem_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures.
#
# `_DATASET_CSV` is byte-for-byte `$CLASSIC/tests/moses/dataset.csv` --
# `tests/moses/moses/test_moses_main.py` already established this exact
# copy for its own `test_it`/`test_it_target_ignore`; duplicated here
# (rather than imported) per this project's "tests do not import from
# other tests' modules" convention (that file's own comment).
# ---------------------------------------------------------------------------

_DATASET_CSV = """res,arg1,arg2
0,0,1
0,1,0
1,1,1
0,0,1
0,1,0
1,1,1
1,0,1
1,1,0
0,1,1
"""


def _write_dataset(tmp_path) -> str:
    p = tmp_path / "dataset.csv"
    p.write_text(_DATASET_CSV)
    return str(p)


# The contin-output table used by the `it` contin-branch tests below: the
# same 20-row vendored slice of classic's `trainingRegression.dat` that
# `capture_contin_bscore.py` already established as this project's contin
# fixture, reused rather than re-vendored.
_CONTIN_SAMPLE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "table_data" / "contin_regression_sample.csv"
)

# The operators `--ignore-operator` can name, in the SAME ORDER as
# `tools/cpp_probe/alphabet_size_probe.cc`'s own `all[]` array -- the
# `_ALPHABET_SIZE_CONTIN_ROWS` table below is indexed by a prefix length
# of this list, so the order is load-bearing, not cosmetic.
_IGNORABLE_OPS = [
    builtin.plus,
    builtin.times,
    builtin.div,
    builtin.exp,
    builtin.log,
    builtin.sin,
    builtin.impulse,
    builtin.logical_and,
    builtin.logical_or,
    builtin.logical_not,
    builtin.greater_than_zero,
    builtin.cond,
]


def _fresh_pms_and_tpp(argv):
    """Build a real `(table_problem_params, problem_params)` pair via the
    SAME `register_table_problems`/`parse_table_args` path a CLI caller
    would use, rather than a hand-rolled fake `problem_params` -- every
    default a test below relies on is therefore the SAME default a real
    `-H <name>` invocation would see (docs/CARRY-FORWARD.md 5.4's own
    rule: "an oracle configured by the port cannot detect the port's
    configuration being wrong").
    """
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    pms = parse_table_args(argv, tpp)
    pms.printer = lambda mp, dex, stats: None
    return pmr, tpp, pms


def _capture_metapop_moses_results(monkeypatch):
    """Replace `table_problems.metapop_moses_results` with a recording
    stub, matching `tests/moses/moses/test_moses_main.py`'s own "FAKE
    local_moses ... a recording stub" convention -- used where a test
    needs to observe the ARGUMENTS a `*_table_problem.run()` builds
    without paying for (or depending on) a real search.
    """
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(tp, "metapop_moses_results", fake)
    return calls


# ---------------------------------------------------------------------------
# table_problem_params
# ---------------------------------------------------------------------------


def test_table_problem_params_defaults():
    tpp = table_problem_params()
    assert tpp.input_data_files == []
    assert tpp.target_feature == ""
    assert tpp.timestamp_feature == ""
    assert tpp.weighting_feature == ""
    assert tpp.ignore_features_str == []


def test_parse_table_args_populates_both_option_surfaces(tmp_path):
    csv_path = _write_dataset(tmp_path)
    tpp = table_problem_params()
    pms = parse_table_args(["-i", csv_path, "-u", "arg2", "-Y", "res"], tpp)
    # table_problem_params's own options:
    assert tpp.input_data_files == [csv_path]
    assert tpp.target_feature == "arg2"
    assert tpp.ignore_features_str == ["res"]
    # problem_params's options, parsed off the SAME shared parser:
    assert pms.problem == "it"
    assert pms.complexity_ratio == 3.5
    assert pms.min_rand_input == 0.0
    assert pms.max_rand_input == 1.0


def test_parse_table_args_accumulates_repeated_input_and_ignore_flags(tmp_path):
    csv_path = _write_dataset(tmp_path)
    tpp = table_problem_params()
    parse_table_args(["-i", csv_path, "-i", csv_path, "-Y", "a", "-Y", "b"], tpp)
    assert tpp.input_data_files == [csv_path, csv_path]
    assert tpp.ignore_features_str == ["a", "b"]


def test_parse_table_args_uses_the_tpp_it_is_given_not_a_fresh_one(tmp_path):
    """`parse_table_args` must populate the SAME `table_problem_params`
    instance `register_table_problems` already wired into every
    registered problem -- passing a second, freshly-constructed one would
    silently parse into an object nothing reads (this function's own
    docstring). Proven directly: the `tpp` returned by
    `register_table_problems` sees the parsed fields.
    """
    csv_path = _write_dataset(tmp_path)
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    parse_table_args(["-i", csv_path], tpp)
    assert tpp.input_data_files == [csv_path]
    it_prob = pmr.find_problem("it")
    assert it_prob._tpp is tpp


# ---------------------------------------------------------------------------
# table_problem_base.common_setup
# ---------------------------------------------------------------------------


def test_common_setup_requires_input_file(tmp_path):
    _pmr, _tpp, pms = _fresh_pms_and_tpp([])
    prob = it_table_problem(table_problem_params())
    with pytest.raises(ValueError, match="no input data file"):
        prob.common_setup(pms)


def test_common_setup_loads_table_and_infers_arity(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")
    prob.common_setup(pms)
    assert prob.arity == 2
    assert prob.table.size() == 9
    assert prob.ctable is not None


def test_common_setup_ilabels_empty_by_default_and_populated_with_output_with_labels(tmp_path):
    csv_path = _write_dataset(tmp_path)

    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")
    prob.common_setup(pms)
    assert prob.ilabels == []
    assert pms.printer.ilabels == []

    pmr2, tpp2, pms2 = _fresh_pms_and_tpp(["-i", csv_path, "-W", "1"])
    prob2 = pmr2.find_problem("it")
    prob2.common_setup(pms2)
    assert prob2.ilabels == ["arg1", "arg2"]
    assert pms2.printer.ilabels == ["arg1", "arg2"]


def test_common_setup_nsamples_gate(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    pms.nsamples = 3
    prob = pmr.find_problem("it")
    with pytest.raises(NotImplementedError, match="subsampleTable"):
        prob.common_setup(pms)


def test_common_setup_balance_gate(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    pms.balance = True
    prob = pmr.find_problem("it")
    with pytest.raises(NotImplementedError, match="CTable.balance"):
        prob.common_setup(pms)


def test_common_setup_multi_file_scores_only_the_first_file(tmp_path):
    """DO NOT FIX: see table_problems.py's own module docstring, "The
    multi-input-file quirk" -- classic validates arity across every `-i`
    file but only ever scores against the front() one.
    """
    csv1 = _write_dataset(tmp_path)
    csv2_path = tmp_path / "dataset2.csv"
    csv2_path.write_text("res,arg1,arg2\n1,1,1\n0,0,0\n0,0,1\n")
    csv2 = str(csv2_path)

    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv1, "-i", csv2])
    prob = pmr.find_problem("it")
    prob.common_setup(pms)
    assert len(prob._tables) == 2
    assert prob.table.size() == 9  # csv1's row count, NOT 9 + 3
    assert prob.ctable is prob._ctables[0]


def test_common_setup_multi_file_arity_mismatch_raises(tmp_path):
    csv1 = _write_dataset(tmp_path)
    csv2_path = tmp_path / "dataset_bad_arity.csv"
    csv2_path.write_text("res,arg1\n1,1\n0,0\n")
    csv2 = str(csv2_path)

    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv1, "-i", csv2])
    prob = pmr.find_problem("it")
    with pytest.raises(ValueError, match="has arity"):
        prob.common_setup(pms)


# ---------------------------------------------------------------------------
# table_problem_base.common_type_setup
# ---------------------------------------------------------------------------


def test_common_type_setup_infers_boolean_and_builds_default_exemplar(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")
    prob.common_setup(pms)
    assert pms.exemplars == []

    prob.common_type_setup(pms)
    assert prob.output_type == type_node.boolean_type
    assert len(pms.exemplars) == 1
    assert type_node(prob.table_type_signature.otype) == type_node.boolean_type


def test_common_type_setup_preserves_an_existing_exemplar(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")
    given = combo_tree(builtin.logical_or)
    pms.exemplars = [given]
    prob.common_setup(pms)
    prob.common_type_setup(pms)
    assert pms.exemplars == [given]
    assert len(pms.exemplars) == 1


def test_common_type_setup_forced_out_type_used_when_exemplar_is_default(tmp_path):
    """`prerec`/`recall`/`bep`/`f_one`/`select` all call
    `common_type_setup(pms, boolean_type)` regardless of the table's own
    output type (table-problems.cc:414,429,441,453,465,542) -- proven
    here with a table whose target column would otherwise infer as
    boolean anyway (dataset.csv), just to pin that the explicit
    `out_type` argument is honoured end to end.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("prerec")
    prob.common_setup(pms)
    prob.common_type_setup(pms, type_node.boolean_type)
    assert prob.output_type == type_node.boolean_type


def test_common_type_setup_exemplar_output_type_overrides_forced_out_type(tmp_path):
    """C++: table-problems.cc:188-190's own comment: "The exemplar output
    type can differ from the table output type for scorers that are
    trying to select rows (the pre and select scorers)" -- a caller
    supplying a CONTIN-rooted `-e` exemplar to a forced-boolean problem
    ends up with a contin `output_type`, not boolean.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("prerec")
    pms.exemplars = [combo_tree(builtin.plus)]
    prob.common_setup(pms)
    prob.common_type_setup(pms, type_node.boolean_type)
    assert prob.output_type == type_node.contin_type


# ---------------------------------------------------------------------------
# it_table_problem -- name/description, output-type dispatch, live oracles.
# ---------------------------------------------------------------------------


def test_it_table_problem_name_and_description():
    prob = it_table_problem(table_problem_params())
    assert prob.name() == "it"
    assert prob.description() == "Maximize Accuracy"


def test_it_table_problem_enum_output_raises_not_implemented(tmp_path, monkeypatch):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")

    def fake_common_type_setup(self, pms, out_type=type_node.unknown_type):
        del pms, out_type
        self.output_type = type_node.enum_type

    monkeypatch.setattr(table_problem_base, "common_type_setup", fake_common_type_setup)
    with pytest.raises(NotImplementedError, match="enum_effective_bscore"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# it_table_problem's CONTIN branch (table-problems.cc:511-525).
#
# This arm raised `NotImplementedError("... S14 territory this port has not
# built")` until 2026-09-09, and a test here asserted that raise. Both were
# wrong by then: `contin_bscore` landed in `07027b0` and `contin_reduction`
# in `d439711`, so the message was false and the gate was blocking a path
# that already existed. The tests below replace that one, and pin what the
# branch actually does rather than that it refuses.
# ---------------------------------------------------------------------------

_CONTIN_CSV = _CONTIN_SAMPLE.read_text()


def _write_contin_dataset(tmp_path) -> str:
    p = tmp_path / "contin.csv"
    p.write_text(_CONTIN_CSV)
    return str(p)


def test_it_table_problem_contin_output_builds_a_contin_bscore(tmp_path, monkeypatch):
    """The contin arm reaches `metapop_moses_results` with a real
    `contin_bscore`, and -- per the `REGRESSION` macro's
    `if (id::boolean_type != output_type)` (table-problems.cc:347-350) --
    with `pms.contin_reduct` as BOTH reductors, not `bool_reduct`.
    """
    csv_path = _write_contin_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-u", "y"])
    calls = _capture_metapop_moses_results(monkeypatch)
    prob = pmr.find_problem("it")

    prob.run(pms)

    assert len(calls) == 1
    args, _ = calls[0]
    reduct_cand, reduct_rep, mbcscore = args[2], args[3], args[4]
    assert reduct_cand is pms.contin_reduct
    assert reduct_rep is pms.contin_reduct
    assert reduct_cand is not pms.bool_reduct
    assert isinstance(mbcscore._bscorer, tp.contin_bscore)


def test_it_table_problem_contin_defaults_to_squared_error(tmp_path, monkeypatch):
    """`eft = pms.it_abs_err ? abs_error : squared_error`
    (table-problems.cc:516-518), and `--it-abs-err` defaults false.

    Checked behaviourally rather than by comparing the stored callable,
    because `contin_bscore.init` binds a lambda: squared error of a
    3.0 residual is 9.0, absolute error is 3.0.
    """
    csv_path = _write_contin_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-u", "y"])
    assert pms.it_abs_err is False
    calls = _capture_metapop_moses_results(monkeypatch)
    pmr.find_problem("it").run(pms)

    bscore = calls[0][0][4]._bscorer
    assert bscore._err_func(3.0, 0.0) == pytest.approx(9.0)


def test_it_table_problem_contin_it_abs_err_selects_abs_error(tmp_path, monkeypatch):
    csv_path = _write_contin_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-u", "y", "--it-abs-err=1"])
    assert pms.it_abs_err is True
    calls = _capture_metapop_moses_results(monkeypatch)
    pmr.find_problem("it").run(pms)

    bscore = calls[0][0][4]._bscorer
    assert bscore._err_func(3.0, 0.0) == pytest.approx(3.0)


def test_discretize_threshold_reaches_discretize_contin_bscore(tmp_path, monkeypatch):
    """`discretize_contin_bscore` (bscores.cc:196) is ported as of
    2026-09-10, so `--discretize-threshold` no longer raises at either the
    parse-time or the run-time gate. Both of those gates had tests here
    asserting the refusal; this replaces them.

    `-R` selects a DIFFERENT scorer from the plain contin arm, so the
    assertion is on which class was built, not merely that the run
    completed.
    """
    csv_path = _write_contin_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-u", "y", "--discretize-threshold=0.5"])
    assert pms.discretize_thresholds == [0.5]
    calls = _capture_metapop_moses_results(monkeypatch)

    pmr.find_problem("it").run(pms)

    assert len(calls) == 1
    bscore = calls[0][0][4]._bscorer
    assert isinstance(bscore, tp.discretize_contin_bscore)
    assert bscore._thresholds == [0.5]


def test_discretize_threshold_takes_precedence_over_the_plain_contin_arm(tmp_path, monkeypatch):
    """Without `-R` the same table builds a `contin_bscore` instead --
    otherwise the test above would pass against a branch that ignored the
    flag entirely.
    """
    csv_path = _write_contin_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-u", "y"])
    assert pms.discretize_thresholds == []
    calls = _capture_metapop_moses_results(monkeypatch)

    pmr.find_problem("it").run(pms)

    assert isinstance(calls[0][0][4]._bscorer, tp.contin_bscore)


# ---------------------------------------------------------------------------
# `_alphabet_size`'s contin branch and its two-stage integer wrap.
#
# Rows measured against the REAL `opencog::moses::alphabet_size` linked out
# of `libmoses_exec.so` -- `tools/cpp_probe/alphabet_size_probe.cc`, run
# 2026-09-09 under `( ulimit -v 2097152; timeout 60 ... )`. Each row is
# `(arity, n_ignored, unsigned_as, int_as)`: the value C++ stores in
# `unsigned as` (problem.cc:139) and the value it then stores in
# `int as = alphabet_size(...)` (table-problems.cc:335,372).
#
# Note what these rows do and do not show. They DO show C++ reaching
# negative `as`. They do NOT show a divergence from naive Python
# arithmetic -- the wrap only becomes observable past -2**31, which needs
# ~2e9 distinct ignored operators. The port masks anyway so the agreement
# follows from C++'s declared types; see the module docstring.
# ---------------------------------------------------------------------------

_ALPHABET_SIZE_CONTIN_ROWS = [
    (1, 0, 9, 9),
    (1, 8, 1, 1),
    (1, 9, 0, 0),
    (1, 10, 4294967295, -1),
    (1, 11, 4294967294, -2),
    (1, 12, 4294967293, -3),
    (2, 10, 0, 0),
    (2, 11, 4294967295, -1),
    (2, 12, 4294967294, -2),
    (3, 11, 0, 0),
    (3, 12, 4294967295, -1),
]


@pytest.mark.parametrize(("arity", "n_ignored", "as_u", "as_i"), _ALPHABET_SIZE_CONTIN_ROWS)
def test_alphabet_size_contin_matches_classic(arity, n_ignored, as_u, as_i):
    sig = gen_signature(type_node.contin_type, type_node.contin_type, arity)
    ignore_ops = set(_IGNORABLE_OPS[:n_ignored])
    assert len(ignore_ops) == n_ignored
    assert tp._alphabet_size(sig, ignore_ops) == as_u
    assert tp._to_int32(tp._alphabet_size(sig, ignore_ops)) == as_i


def test_it_table_problem_unsupported_output_raises_value_error(tmp_path, monkeypatch):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    prob = pmr.find_problem("it")

    def fake_common_type_setup(self, pms, out_type=type_node.unknown_type):
        del pms, out_type
        self.output_type = type_node.ann_type

    monkeypatch.setattr(table_problem_base, "common_type_setup", fake_common_type_setup)
    with pytest.raises(ValueError, match="currently not supported"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# LIVE ORACLES -- reproduce classic's own binary, driven fully through the
# registry + the ported option surface (register_table_problems +
# parse_table_args + find_problem + run), at CLASSIC'S OWN DEFAULTS: no
# option below is set to anything other than what a bare
# `moses -i dataset.csv -m1000 [-u ... -Y ...]` command line would use.
#
# Measured directly against $CLASSIC/build/moses/moses/main/moses (this
# task's own live measurement, 2026-09-08), bounded:
#
#   ( ulimit -v 2097152; timeout 120 ./build/moses/moses/main/moses \
#       -i tests/moses/dataset.csv -m1000 )
#   -> "-3 and($1 $2)" is the top (and only score -3) printed line.
#   Reran with -H it explicitly and with -r0: byte-identical.
#
#   ( ulimit -v 2097152; timeout 120 ./build/moses/moses/main/moses \
#       -i tests/moses/dataset.csv -m1000 -uarg2 -Yres )
#   -> "-3 true" is the ONLY printed line.
#
# Per docs/CARRY-FORWARD.md 5.4's own rule ("an oracle configured by the
# port cannot detect the port's configuration being wrong"), NOTHING
# about these two tests' `problem_params`/`table_problem_params` is
# hand-tuned: `parse_table_args` runs the exact same `add_options`/
# `parse_options` a real `-H it` CLI invocation would run, and every
# default (complexity_ratio=3.5, hc_crossover=True, noise=-1.0, ...) is
# whatever `problem_params.py` says classic's real default is.
# ---------------------------------------------------------------------------


def test_it_table_problem_reproduces_classic_binary_default_run(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    pms = parse_table_args(["-i", csv_path, "-m", "1000"], tpp)
    prob = pmr.find_problem(pms.problem)
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)

    prob.run(pms)

    mp = result["mp"]
    assert mp.best_score() == -3.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "and($1 $2)"


def test_it_table_problem_target_ignore_reproduces_classic_binary(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    pms = parse_table_args(["-i", csv_path, "-m", "1000", "-u", "arg2", "-Y", "res"], tpp)
    prob = pmr.find_problem(pms.problem)
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)

    prob.run(pms)

    mp = result["mp"]
    assert mp.best_score() == -3.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "true"


# ---------------------------------------------------------------------------
# pre_table_problem
# ---------------------------------------------------------------------------


def test_pre_table_problem_name_and_description():
    prob = pre_table_problem(table_problem_params())
    assert prob.name() == "pre"
    assert prob.description() == "Precision-Activation scoring"


def test_pre_table_problem_default_options_crash_like_classic(tmp_path):
    """Classic's OWN default options crash `-H pre` on ANY dataset -- see
    `precision_bscore.py`'s own module docstring for the constructor
    assert this reproduces. Measured directly against the compiled
    binary, bounded (2026-09-08):

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -Hpre )
      -> terminate called after throwing opencog::AssertionException:
         "Precision scorer, invalid activation bounds. The minimum
         activation must be greater than zero, and the maximum activation
         must be greater than or equal to the minimum activation."
         (precision_bscore.cc:163)

    This is NOT a table_problems.py defect: `pms.hardness=1.0` (> 0) and
    `pms.min_rand_input=0.0` are both classic's own CLI defaults
    (problem-params.cc), and `discriminating_bscore`'s/`precision_bscore`'s
    own threshold assert requires `min_activation > 0` whenever
    `activation_pressure > 0`. DO NOT FIX -- reproduced faithfully.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-m", "30"])
    prob = pmr.find_problem("pre")
    with pytest.raises(AssertionError, match="invalid activation bounds"):
        prob.run(pms)


def test_pre_table_problem_reproduces_mosesutest_precision(tmp_path):
    """C++: tests/moses/MOSESUTest.cxxtest:181 test_precision
    (`moses_test_combo({"-Hpre", precision.csv, "-m100", "-q0.2", "-w0.5",
    "-Q1", "-z10000"}, {"$2"})`). `tests/fixtures/table_data/precision.csv`
    is byte-for-byte `$CLASSIC/tests/moses/precision.csv` (diffed when
    this test was written). Measured directly against the compiled
    binary, bounded, with the SAME flags:

      ( ulimit -v 2097152; timeout 60 ./build/moses/moses/main/moses \
          -i tests/moses/precision.csv -m100 -Hpre -q0.2 -w0.5 -Q1 \
          -z10000 )
      -> top line "1 $2" (score 1, tree $2).
    """
    pmr, tpp, pms = _fresh_pms_and_tpp(
        [
            "-i",
            "tests/fixtures/table_data/precision.csv",
            "-m",
            "100",
            "-q",
            "0.2",
            "-w",
            "0.5",
            "-Q",
            "1",
            "-z",
            "10000",
        ]
    )
    prob = pmr.find_problem("pre")
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)

    prob.run(pms)

    mp = result["mp"]
    assert mp.best_score() == 1.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "$2"
    # table-problems.cc:396: pre_table_problem::run always sets this,
    # unconditionally, regardless of --boost.
    assert pms.meta_params.ensemble_params.experts is True


def test_pre_table_problem_reproduces_mosesutest_negative_predictive_value(tmp_path):
    """C++: tests/moses/MOSESUTest.cxxtest:187 test_negative_predictive_value
    (same as above plus `--pre-positive=0`, expected tree `$1`). Measured
    directly against the compiled binary, bounded:

      ( ulimit -v 2097152; timeout 60 ./build/moses/moses/main/moses \
          -i tests/moses/precision.csv -m100 -Hpre -q0.2 -w0.5 -Q1 \
          --pre-positive=0 -z10000 )
      -> top line "1 $1" (score 1, tree $1).
    """
    pmr, tpp, pms = _fresh_pms_and_tpp(
        [
            "-i",
            "tests/fixtures/table_data/precision.csv",
            "-m",
            "100",
            "-q",
            "0.2",
            "-w",
            "0.5",
            "-Q",
            "1",
            "-z",
            "10000",
            "--pre-positive=0",
        ]
    )
    prob = pmr.find_problem("pre")
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)

    prob.run(pms)

    mp = result["mp"]
    assert mp.best_score() == 1.0
    assert combo_tree_to_str(mp.best_tree()).strip() == "$1"


def test_pre_table_problem_feature_selection_gate(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-q", "0.2", "-w", "0.5", "-Q", "1"])
    # A real CLI run already rejects --enable-fs=1 earlier, inside
    # problem_params.parse_options -- flipped by hand here for
    # direct-call fidelity (this file's own module docstring), exactly
    # mirroring how a caller could hand-build a problem_params bypassing
    # parse_options.
    pms.enable_feature_selection = True
    prob = pmr.find_problem("pre")
    with pytest.raises(NotImplementedError, match="feature_selector"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# prerec / recall / bep -- name/description, the hardness==0 CLI fallback,
# the default-options crash, and a real live oracle with non-default (but
# classic-successful) -q/-w/-Q values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem_name", "cls", "expected_min", "expected_max"),
    [
        ("prerec", prerec_table_problem, 0.5, 1.0),
        ("recall", recall_table_problem, 0.8, 1.0),
    ],
)
def test_hardness_zero_fallback(
    tmp_path, monkeypatch, problem_name, cls, expected_min, expected_max
):
    """C++: table-problems.cc:431-432 (prerec), :443-444 (recall) --
    `if (0.0 == pms.hardness) { pms.hardness = 1.0; pms.min_rand_input =
    X; pms.max_rand_input = Y; }`, each problem's own hardcoded fallback
    triple, ported verbatim including the mutation of `pms` itself.
    `bep`'s own identically-shaped fallback is tested separately below
    (`test_bep_hardness_zero_fallback_still_crashes_like_classic`): its
    hardcoded triple (`0.0, 0.5`) does NOT dodge the constructor assert
    the fallback exists to avoid.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-Q", "0"])
    assert pms.hardness == 0.0
    calls = _capture_metapop_moses_results(monkeypatch)

    prob = pmr.find_problem(problem_name)
    assert isinstance(prob, cls)
    prob.run(pms)

    assert pms.hardness == 1.0
    assert pms.min_rand_input == expected_min
    assert pms.max_rand_input == expected_max
    assert len(calls) == 1


def test_bep_hardness_zero_fallback_still_crashes_like_classic(tmp_path):
    """C++: table-problems.cc:455-456: `if (0.0 == pms.hardness) {
    pms.hardness = 1.0; pms.min_rand_input = 0.0; pms.max_rand_input =
    0.5; }` -- unlike `prerec`/`recall`'s own fallback triples, `bep`'s
    own hardcoded `min_rand_input = 0.0` does NOT dodge
    `discriminating_bscore`'s `0.0 < min_threshold` assert (`0.0 < 0.0`
    is false): classic's OWN fallback for `-H bep -Q0` is broken too, not
    just its unmodified defaults (`discriminating_bscore.py`'s own module
    docstring already documents this exact fact). Measured directly
    against the compiled binary, bounded (2026-09-08):

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -Hbep -Q0 )
      -> terminate called after throwing opencog::AssertionException:
         "Discriminating scorer, invalid thresholds. ..."
         (discriminating_bscore.cc:224)

    DO NOT FIX -- the `pms` mutation itself still happens (checked below,
    up to the point of the crash), exactly as classic's own fallback
    assignment runs unconditionally before the scorer is ever
    constructed.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-Q", "0"])
    assert pms.hardness == 0.0
    prob = pmr.find_problem("bep")
    assert isinstance(prob, bep_table_problem)

    with pytest.raises(AssertionError, match="invalid thresholds"):
        prob.run(pms)

    assert pms.hardness == 1.0
    assert pms.min_rand_input == 0.0
    assert pms.max_rand_input == 0.5


@pytest.mark.parametrize("problem_name", ["prerec", "recall", "bep"])
def test_hardness_nonzero_fallback_not_triggered(tmp_path, monkeypatch, problem_name):
    csv_path = _write_dataset(tmp_path)
    # -Q1 (hardness=1.0, not 0) plus classic-valid -q/-w so the scorer
    # itself doesn't also raise before the fallback question is answered.
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-Q", "1", "-q", "0.5", "-w", "1.0"])
    calls = _capture_metapop_moses_results(monkeypatch)
    prob = pmr.find_problem(problem_name)
    prob.run(pms)
    assert pms.hardness == 1.0
    assert pms.min_rand_input == 0.5
    assert pms.max_rand_input == 1.0
    assert len(calls) == 1


@pytest.mark.parametrize("problem_name", ["prerec", "recall", "bep"])
def test_discriminating_problem_default_options_crash_like_classic(tmp_path, problem_name):
    """Classic's own default options (`hardness=1.0`, `min_rand_input=0.0`)
    crash `-H prerec`/`-H recall`/`-H bep` on ANY dataset -- see
    `discriminating_bscore.py`'s own module docstring for the constructor
    assert this reproduces. Measured directly against the compiled
    binary, bounded (2026-09-08), for all three:

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -H<name> )
      -> terminate called after throwing opencog::AssertionException:
         "Discriminating scorer, invalid thresholds.  The minimum
         threshold must be greater than zero, and the maximum threshold
         must be greater than or equal to the minimum threshold."
         (discriminating_bscore.cc:224)

    DO NOT FIX -- classic's own default `-q0.0` combined with `hardness=
    1.0 (!= 0)` never triggers each problem's own hardness==0 fallback,
    so `min_threshold=0.0` reaches `discriminating_bscore`'s constructor
    unmodified and fails its own `0.0 < min_threshold` assert.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-m", "30"])
    prob = pmr.find_problem(problem_name)
    with pytest.raises(AssertionError, match="invalid thresholds"):
        prob.run(pms)


@pytest.mark.parametrize(
    ("problem_name", "expected_score", "expected_tree"),
    [
        ("prerec", 0.6666666865348816, "and($1 $2)"),
        ("recall", 0.8822169899940491, "true"),
        ("bep", 0.7222222089767456, "true"),
    ],
)
def test_discriminating_problem_reproduces_classic_binary(
    tmp_path, problem_name, expected_score, expected_tree
):
    """Measured directly against the compiled binary, bounded
    (2026-09-08), for all three:

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -H<name> -q0.5 -w1.0 -Q1 )
      -> prerec: "0.666666686534881592 and($1 $2)"
         recall: "0.882216989994049072 true"
         bep:    "0.722222208976745605 true"
      (binary32 values printed at double width -- CLAUDE.md/CARRY-FORWARD
      4.2's own note; `expected_score` above is the nearest Python float
      to that same binary32 value.)
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(
        ["-i", csv_path, "-m", "30", "-q", "0.5", "-w", "1.0", "-Q", "1"]
    )
    prob = pmr.find_problem(problem_name)
    result = {}
    pms.printer = lambda mp, dex, stats: result.update(mp=mp)

    prob.run(pms)

    mp = result["mp"]
    assert mp.best_score() == pytest.approx(expected_score, abs=0.0)
    assert combo_tree_to_str(mp.best_tree()).strip() == expected_tree


# ---------------------------------------------------------------------------
# f_one_table_problem -- always crashes, matching classic's own crash.
# ---------------------------------------------------------------------------


def test_f_one_table_problem_name_and_description():
    prob = f_one_table_problem(table_problem_params())
    assert prob.name() == "f_one"
    assert prob.description() == "Maximize F_1 score"


def test_f_one_table_problem_always_raises_like_classic(tmp_path):
    """`f_one_bscore` is unconstructible in classic itself -- see
    `discriminating_bscore.py`'s own module docstring. Measured directly
    against the compiled binary, bounded (2026-09-08):

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -Hf_one )
      -> terminate called after throwing opencog::AssertionException:
         "Discriminating scorer, invalid thresholds. ..."
         (discriminating_bscore.cc:224)

    DO NOT FIX -- `f_one_table_problem::run()` (table-problems.cc:461-468)
    calls `f_one_bscore(ctable)` with no way for ANY CLI option to avoid
    the crash (the three threshold values `f_one_bscore`'s own C++
    constructor passes to its base class are all hardcoded).
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-m", "30"])
    prob = pmr.find_problem("f_one")
    with pytest.raises(AssertionError, match="invalid thresholds"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# select_table_problem -- correctly wired, but blocked from a full
# end-to-end run against a CONTIN-input table by a pre-existing,
# already-disclosed whole-project gap (interpreter_visitor has no
# contin/mixed overload -- select_bscore.py's own module docstring).
# ---------------------------------------------------------------------------


def test_select_table_problem_name_and_description():
    prob = select_table_problem(table_problem_params())
    assert prob.name() == "select"
    assert prob.description() == "Select range of rows"


def test_select_table_problem_requires_contin_output_like_classic(tmp_path):
    """`select_bscore` requires a CONTIN-output ctable -- `dataset.csv`'s
    target is boolean, so this raises, matching classic's own crash.
    Measured directly against the compiled binary, bounded (2026-09-08):

      ( ulimit -v 2097152; timeout 30 ./build/moses/moses/main/moses \
          -i tests/moses/dataset.csv -m30 -Hselect )
      -> terminate called after throwing opencog::AssertionException:
         "The selection scorer can only be used with contin-valued
         tables!" (select_bscore.cc:74)
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-m", "30"])
    prob = pmr.find_problem("select")
    with pytest.raises(AssertionError, match="contin-valued tables"):
        prob.run(pms)


def test_select_table_problem_builds_the_right_scorer_and_calls_metapop_moses_results(
    tmp_path, monkeypatch
):
    """Proves the WIRING (scorer constructed with the right arguments,
    `metapop_moses_results` called) without depending on the SEPARATE
    contin knob-building gap the next test documents -- see
    `build_knobs.py`'s own "non-Boolean inputs require their
    corresponding domain builder slice" gate (S14 Task 2, not yet
    ported). `metapop_moses_results` is monkeypatched here, so this test
    never reaches actual tree evaluation/knob building either way.
    """
    calls = _capture_metapop_moses_results(monkeypatch)
    pmr, tpp, pms = _fresh_pms_and_tpp(
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
        ]
    )
    prob = pmr.find_problem("select")
    prob.run(pms)
    assert len(calls) == 1
    (bases, sig, reduct_cand, reduct_rep, mbcscore, *_rest) = calls[0][0]
    assert bases is pms.exemplars
    assert reduct_cand is pms.bool_reduct
    assert reduct_rep is pms.bool_reduct_rep
    assert mbcscore.get_bscore is not None  # a real behave_cscore, not a placeholder


def test_select_table_problem_end_to_end_matches_classic():
    """UPDATED twice. It first hit `interpreter_visitor`'s missing
    `mixed_interpreter` path; then, once that closed, it hit the
    knob-building gap for a contin-input exemplar ("non-Boolean inputs
    require their corresponding domain builder slice") and asserted that
    `NotImplementedError`. The mixed-predicate slice closed THAT gap too --
    `build_logical`'s `is_predicate` branch (build_knobs.cc:640) --
    so `-Hselect` over a contin-input table now runs end to end.

    `_fresh_pms_and_tpp` installs a null printer, so this test only asserts
    that the run COMPLETES and reaches the printer with a populated
    metapopulation. The byte-for-byte stdout comparison against classic for
    exactly this argv lives at the CLI level, in
    ``tests/moses/representation/test_mixed_predicate.py::
    test_select_over_a_contin_table_matches_classic_byte_for_byte``.
    """
    csv_path = "tests/fixtures/table_data/select_regression.csv"
    pmr, tpp, pms = _fresh_pms_and_tpp(
        ["-i", csv_path, "-u", "y", "-m", "30", "-q", "0.2", "-w", "0.8"]
    )
    seen = []
    pms.printer = lambda mp, dex, stats: seen.append(
        [combo_tree_to_str(sct.get_tree()) for sct in mp.best_candidates()]
    )
    prob = pmr.find_problem("select")
    prob.run(pms)
    assert seen == [["true "]]


# ---------------------------------------------------------------------------
# _regression's contin_reduct gate (S14) -- reachable only through a
# forced-boolean problem given a contin-rooted custom exemplar (see
# test_common_type_setup_exemplar_output_type_overrides_forced_out_type).
#
# `problem_params.contin_reduct` is no longer `None` from a real,
# `parse_options`-built `problem_params` (S13 CLI-gate-reconciliation task:
# `contin_reduction`/`contin_bscore` landed, S14 Tasks 3/4, `d439711`/
# `07027b0`) -- so `_regression`'s own `if pms.contin_reduct is None: raise`
# branch is DEFENSIVE code now, not a live gate a real CLI run can reach.
# This test forces `contin_reduct` back to `None` after building a real
# `problem_params` (rather than asserting it starts that way) specifically
# to keep covering that still-real branch in table_problems.py, which is
# outside this task's owned scope to remove or restructure.
# ---------------------------------------------------------------------------


def test_regression_contin_reduct_gate(tmp_path):
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-q", "0.5", "-w", "1.0", "-Q", "1"])
    pms.exemplars = [combo_tree(builtin.plus)]
    pms.contin_reduct = None  # simulate the pre-wiring state; see comment above
    prob = pmr.find_problem("prerec")
    # ValueError, not NotImplementedError: since `d439711` a
    # `parse_options`-built `problem_params` always carries a real
    # `contin_reduction`, so a `None` here means a caller bypassed the
    # constructor -- a bad argument, not an unported subsystem.
    with pytest.raises(ValueError, match="contin_reduct"):
        prob.run(pms)


def test_regression_with_real_contin_reduct_now_runs_to_completion(tmp_path):
    """This pin fired, and it was right to.

    It used to assert `pytest.raises(ZeroDivisionError)`, with its own note
    that it existed "so a future change ... that silently starts succeeding
    is visible as a test diff rather than discovered by surprise". That is
    exactly what happened: the ZeroDivisionError was a PORT DEFECT, not a
    property of the scenario. `prerec_bscore::operator()` divides its whole
    behavioural score by `pos_total` (discriminating_bscore.cc:513), which
    is zero whenever no ctable row is positive -- IEEE division in C++,
    yielding NaN, but an exception in Python. Comparing classic's own
    `-l DEBUG` log against the port's found it: classic logged
    `Score: -nan` where the port logged `Score: -3.40282e+38`, because the
    exception was caught upstream and turned into the worst possible score,
    and the two runs then kept different numbers of candidates (25 vs 23).

    With `_ieee_div` in place the scenario runs to completion, as it does
    in classic. Pinned in the new direction for the same reason it was
    pinned in the old one.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-q", "0.5", "-w", "1.0", "-Q", "1"])
    pms.exemplars = [combo_tree(builtin.plus)]
    prob = pmr.find_problem("prerec")
    assert pms.contin_reduct is not None
    prob.run(pms)


# ---------------------------------------------------------------------------
# pre_conj_table_problem -- table-problems.cc:410.
# ---------------------------------------------------------------------------


def test_pre_conj_is_registered_under_the_HYPHENATED_name(tmp_path):
    """`pre_conj_table_problem::name()` returns `"pre-conj"`, with a
    hyphen (table-problems.h:134). This is the whole reason `-Hpre_conj`
    is refused by classic AND by this port -- there is no such name in
    either registry, so `find_problem` misses and `moses_exec` prints
    `problem type "pre_conj" is currently unsupported.` and returns 1
    (measured against the real binary, 2026-09-09).
    """
    pmr = problem_manager()
    register_table_problems(pmr)
    assert isinstance(pmr.find_problem("pre-conj"), pre_conj_table_problem)
    assert pmr.find_problem("pre_conj") is None
    assert pmr.find_problem("pre-conj").description() == "Precision-Conjunction-Maximization"


def test_pre_conj_builds_precision_conj_bscore_with_hardness_and_positive(tmp_path, monkeypatch):
    """MUTATION PROVEN. `REGRESSION(ctable, precision_conj_bscore,
    (ctable, pms.hardness, pms.pre_positive))` (table-problems.cc:421-422)
    -- both non-table arguments come from `pms`, not from the scorer's
    own defaults. Hardcoding either (`hardness=1.0`, or dropping
    `pms.pre_positive` so the default `positive=True` applies) makes one
    of these two parametrized cases fail.
    """
    from pymetta.moses.scoring.precision_bscore import precision_conj_bscore

    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, "-Q", "3", "--pre-positive", "0"])
    calls = _capture_metapop_moses_results(monkeypatch)

    prob = pmr.find_problem("pre-conj")
    prob.run(pms)

    assert len(calls) == 1
    bscore = calls[0][0][4]._bscorer
    assert isinstance(bscore, precision_conj_bscore)
    assert bscore.hardness == 3.0
    assert bscore.positive is False


def test_pre_conj_rejects_boosting_like_classic(tmp_path):
    """C++: table-problems.cc:419-420 `OC_ASSERT(not pms.meta_params.
    do_boosting, "Boosting not supported for the pre problem!")`. Classic
    throws `AssertionException` out of `main` and dumps core; this port
    raises `NotImplementedError`, the same mapping `demo_problems.py`
    uses for its own three identical `OC_ASSERT(not do_boosting)` sites.
    Deleting the check lets the run proceed past it and this fails.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path])
    pms.meta_params.do_boosting = True
    prob = pmr.find_problem("pre-conj")
    with pytest.raises(NotImplementedError, match="Boosting not supported"):
        prob.run(pms)


# ---------------------------------------------------------------------------
# ps_params reaches metapop_moses_results from every table-problem caller.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem_name", "extra_argv"),
    [
        ("it", []),
        # `pre` is the HAND-INLINED copy of the REGRESSION macro
        # (table-problems.cc:358-408), a separate call site from
        # `_regression`'s -- the argument had to be added twice.
        ("pre", ["-q", "0.1", "-w", "1.0"]),
        ("pre-conj", []),
        ("prerec", ["-Q", "1", "-q", "0.5", "-w", "1.0"]),
    ],
)
def test_every_table_problem_passes_pms_ps_params_in_C_PLUS_PLUS_position(
    tmp_path, monkeypatch, problem_name, extra_argv
):
    """MUTATION PROVEN. C++'s `REGRESSION` macro and
    `pre_table_problem::run` both call `metapop_moses_results(...,
    pms.opt_params, pms.hc_params, pms.ps_params, pms.deme_params, ...)`
    (table-problems.cc:352-355, :404-407). This port dropped the
    `ps_params` argument entirely until 2026-09-09.

    Position 7 is checked, not just presence: swapping `pms.ps_params`
    and `pms.hc_params` (or inserting it after `deme_params`) type-checks
    fine in Python and would silently hand `particle_swarm` an
    `hc_parameters`. Deleting the argument from any ONE of these call
    sites fails exactly that case.
    """
    csv_path = _write_dataset(tmp_path)
    pmr, tpp, pms = _fresh_pms_and_tpp(["-i", csv_path, *extra_argv])
    calls = _capture_metapop_moses_results(monkeypatch)

    pmr.find_problem(problem_name).run(pms)

    assert len(calls) == 1
    args = calls[0][0]
    assert args[5] is pms.opt_params
    assert args[6] is pms.hc_params
    assert args[7] is pms.ps_params
    assert args[8] is pms.deme_params


# ---------------------------------------------------------------------------
# register_table_problems -- composition on top of problem.py without
# editing it.
# ---------------------------------------------------------------------------


_REAL_NAMES_AND_CLASSES = {
    "it": it_table_problem,
    "pre": pre_table_problem,
    "pre-conj": pre_conj_table_problem,
    "prerec": prerec_table_problem,
    "recall": recall_table_problem,
    "bep": bep_table_problem,
    "f_one": f_one_table_problem,
    "select": select_table_problem,
    "cluster": cluster_table_problem,
}
_STUB_NAMES = ("ip", "ann-it")


def test_register_table_problems_registers_all_eleven_names():
    pmr = problem_manager()
    register_table_problems(pmr)
    for name in (*_REAL_NAMES_AND_CLASSES, *_STUB_NAMES):
        assert pmr.find_problem(name) is not None, name


def test_register_table_problems_wires_the_nine_real_classes():
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    for name, cls in _REAL_NAMES_AND_CLASSES.items():
        prob = pmr.find_problem(name)
        assert isinstance(prob, cls), name
        assert prob._tpp is tpp, name


def test_register_table_problems_leaves_two_stubs_raising():
    pmr = problem_manager()
    register_table_problems(pmr)
    for name in _STUB_NAMES:
        prob = pmr.find_problem(name)
        assert isinstance(prob, _unported_table_problem), name
        with pytest.raises(NotImplementedError, match="not ported"):
            prob.run(object())


def test_register_table_problems_returns_the_shared_tpp():
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    assert isinstance(tpp, table_problem_params)
    for name in _REAL_NAMES_AND_CLASSES:
        assert pmr.find_problem(name)._tpp is tpp
