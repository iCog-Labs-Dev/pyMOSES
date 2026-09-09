"""Unit tests for pymetta.moses.main.problem_params (S13 Task 1).

No single C++ oracle binary run backs this file the way a fixture/probe
does elsewhere in this project: fidelity here means "every default
literal in `problem_params.py` equals the literal `problem-params.cc`
declares", which is a plain source-to-source comparison, not something a
compiled probe would add anything to (`docs/MODULE-COVERAGE.md`'s own
non-goal note: "only the option names and defaults are reproduced").
`problem_params.py`'s own module docstring records the ONE piece of this
task that WAS measured against the real compiled binary directly
(`$CLASSIC/build/moses/moses/main/moses --help` and a `--balance=<token>`
sweep) -- the full 127-flag name list and the bool-option token set.

Every default assertion below is traceable to a specific
`problem-params.cc` line via `problem_params.py`'s own `add_options`
source (each `opt(...)`/`sopt(...)`/`vopt(...)` call cites the exact
flag). A representative sample of these was proven by mutation during
development (change the literal default in the source, watch the
matching test go red, revert, watch it go green again) -- see
`task-1-report.md` for the exact list and results; this docstring does
not re-duplicate that list to avoid it silently drifting out of sync with
what was actually run.
"""

from __future__ import annotations

import argparse

import pytest

from pymetta.moses.main.problem_params import _bool_opt, parse_args, problem_params
from pymetta.moses.metapopulation.metapop_params import diversity_parameters


def _parse(*argv: str) -> problem_params:
    return parse_args(list(argv))


def _get(pp: problem_params, path: str):
    obj = pp
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# _bool_opt -- measured against the compiled binary (problem_params.py's own
# module docstring). Every one of boost's documented accepted tokens plus a
# rejection, both directions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("FALSE", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
    ],
)
def test_bool_opt_accepts_boosts_measured_token_set(token, expected):
    assert _bool_opt(token) is expected


@pytest.mark.parametrize("token", ["2", "-1", "abc", "", "maybe"])
def test_bool_opt_rejects_everything_else(token):
    with pytest.raises(argparse.ArgumentTypeError):
        _bool_opt(token)


# ---------------------------------------------------------------------------
# Every default, pinned. One flat table: (dotted attribute path on the
# returned problem_params, expected value). Each row cites the
# problem-params.cc line its default comes from.
# ---------------------------------------------------------------------------

_DEFAULTS: list[tuple[str, object]] = [
    # -- General / problem-type (problem-params.cc:164-238) --
    ("jobs_str", []),
    ("min_pool", 50),
    ("exemplars_str", []),
    ("exemplars", []),
    ("problem", "it"),
    ("nsamples", -1),
    ("balance", False),
    # -- Algorithm control (problem-params.cc:242-370) --
    ("opt_algo", "hc"),
    ("max_score", pytest.approx(3.4028234663852886e38, rel=1e-6)),
    ("max_evals", 10000),
    ("max_time", 2147483647),
    ("cache_size", 3000),
    ("ignore_ops_str", []),
    ("linear_regression", False),
    ("perm_ratio", 0.0),
    ("rand_seed", 1),
    ("complexity_temperature", pytest.approx(6.0)),
    ("complexity_ratio", pytest.approx(3.5)),
    ("cap_coef", 50.0),
    # -- Large-problem / hill-climbing (problem-params.cc:373-490) --
    ("hc_max_nn", 20000),
    ("hc_frac_of_nn", 2.0),
    ("hc_crossover", True),
    ("hc_crossover_pop_size", 120),
    ("hc_crossover_min_neighbors", 400),
    ("hc_resize_to_fit_ram", False),
    # -- particle swarm / contin depth (stored only) --
    ("ps_max_particles", 50),
    ("contin_depth", 5),
    # -- boosting / reduct / metapop (problem-params.cc:512-1076) --
    ("boosting", False),
    ("num_to_promote", 1),
    ("exact_experts", True),
    ("expalpha", 2.0),
    ("bias_scale", 1.0),
    ("reduct_knob_building_effort", 2),
    ("max_dist", 4),
    ("reduce_all", True),
    ("reduct_candidate_effort", 2),
    ("max_gens", -1),
    ("discard_dominated", False),
    ("hc_single_step", False),
    ("include_only_ops_str", []),
    ("pop_size_ratio", pytest.approx(20.0)),
    ("noise", pytest.approx(-1.0)),
    ("hc_widen_search", False),
    ("use_well_enough", False),
    ("revisit", 0),
    # -- Output control (problem-params.cc:1080-1142) --
    ("result_count", 10),
    ("output_score", True),
    ("output_cscore", False),
    ("output_bscore", False),
    ("output_only_best", False),
    ("output_eval_number", False),
    ("output_with_labels", False),
    ("output_deme_id", False),
    ("output_format_str", "combo"),
    ("output_file", ""),
    # -- The rest (problem-params.cc:1146-1590) --
    ("min_rand_input", 0.0),
    ("max_rand_input", 1.0),
    ("log_level", "INFO"),
    ("log_file_dep_opt", False),
    ("log_file", "moses.log"),
    ("max_candidates_per_deme", -1),
    ("weighted_accuracy", False),
    ("diversity_pressure", pytest.approx(0.0)),
    ("diversity_autoscale", False),
    ("diversity_exponent", pytest.approx(-1.0)),
    ("diversity_normalize", True),
    ("diversity_dst", "p_norm"),
    ("diversity_p_norm", pytest.approx(2.0)),
    ("diversity_dst2dp", "auto"),
    ("discretize_thresholds", []),
    ("hardness", pytest.approx(1.0)),
    ("time_dispersion_pressure", pytest.approx(0.0)),
    ("time_dispersion_exponent", pytest.approx(1.0)),
    ("time_bscore", False),
    ("time_bscore_granularity_str", "day"),
    ("gen_best_tree", False),
    ("it_abs_err", False),
    ("pre_positive", True),
    # -- Feature selection (problem-params.cc:1581-1926) --
    ("enable_feature_selection", False),
    ("fs_params.target_size", 20),
    ("fs_params.exp_distrib", False),
    ("fs_focus", "incorrect"),
    ("fs_seed", "add"),
    ("festor_params.prune_xmplr", False),
    ("festor_params.subsampling_ratio", 1.0),
    ("festor_params.subsampling_by_time", False),
    ("festor_params.n_demes", 1),
    ("fs_params.algorithm", "simple"),
    ("fs_params.scorer", "mi"),
    ("fs_params.threshold", 0.0),
    ("fs_enforce_features_filename", ""),
    ("festor_params.diversity_pressure", 0.0),
    ("festor_params.diversity_cap", 100),
    ("festor_params.diversity_interaction", -1),
    ("festor_params.diversity_jaccard", True),
    ("fs_params.inc_red_intensity", -1.0),
    ("fs_params.inc_target_size_epsilon", 1.0e-6),
    ("fs_params.inc_interaction_terms", 1),
    ("fs_params.pre_penalty", 1.0),
    ("fs_params.pre_min_activation", 0.5),
    ("fs_params.pre_max_activation", 1.0),
    ("fs_params.pre_positive", True),
    ("fs_params.hc_max_score", 1.0),
    ("fs_params.hc_max_evals", 10000),
    ("fs_params.hc_fraction_of_remaining", 0.5),
    ("fs_params.hc_crossover", False),
    ("fs_params.hc_crossover_pop_size", 120),
    ("fs_params.hc_crossover_min_neighbors", 400),
    ("fs_params.hc_widen_search", True),
    ("fs_params.mi_confi", 100.0),
    ("fs_params.smd_top_size", 10),
    # -- Subsample-deme (problem-params.cc:1929-2053) --
    ("ss_n_subsample_demes", 0),
    ("ss_n_top_candidates", 1),
    ("ss_n_tuples", 4294967295),
    ("ss_std_dev_threshold", pytest.approx(3.4028234663852886e38, rel=1e-6)),
    ("ss_tanimoto_mean_threshold", pytest.approx(1.0)),
    ("ss_tanimoto_geo_mean_threshold", pytest.approx(1.0)),
    ("ss_tanimoto_max_threshold", pytest.approx(1.0)),
    ("ss_n_best_bfdemes", 0),
    ("ss_tanimoto_mean_weight", pytest.approx(0.0)),
    ("ss_tanimoto_geo_mean_weight", pytest.approx(0.0)),
    ("ss_tanimoto_max_weight", pytest.approx(0.0)),
    ("ss_n_subsample_fitnesses", 0),
    ("ss_low_dev_pressure", pytest.approx(1.0)),
    ("ss_by_time", False),
    ("ss_contiguous_time", True),
]


@pytest.mark.parametrize("path,expected", _DEFAULTS, ids=[p for p, _ in _DEFAULTS])
def test_default_matches_cpp(path, expected):
    pp = _parse()
    assert _get(pp, path) == expected


def test_empty_default_table_is_not_silently_vacuous():
    """CARRY-FORWARD.md section 6's own warning: a parametrised oracle over
    an empty fixture passes silently. Pin the count so a future edit that
    accidentally empties `_DEFAULTS` turns this red, not green-by-accident.
    """
    assert len(_DEFAULTS) >= 120


# ---------------------------------------------------------------------------
# Wired-struct defaults: moses_parameters / hc_parameters / deme_parameters /
# optim_parameters / metapop_parameters (+ diversity/ensemble/filter),
# checked directly against the constructed sub-objects, not just the flat
# CLI-mirror fields above -- this is what actually proves the WIRING (not
# merely the parsing) matches C++.
# ---------------------------------------------------------------------------


def test_moses_parameters_wiring():
    """C++: problem-params.cc:2375-2382"""
    pp = _parse()
    mp = pp.moses_params
    assert mp.local is True
    assert mp.mpi is False
    assert mp.max_evals == 10000
    assert mp.max_gens == -1
    assert mp.max_score == pytest.approx(3.4028234663852886e38, rel=1e-6)
    assert mp.max_time == 2147483647
    assert mp.max_cnd_output == 10  # == result_count default
    assert mp.jobs == {"localhost": 1}


def test_hc_parameters_wiring():
    """C++: problem-params.cc:2360-2368"""
    pp = _parse()
    hc = pp.hc_params
    assert hc.widen_search is False
    assert hc.single_step is False
    assert hc.crossover is True
    assert hc.crossover_pop_size == 120
    assert hc.crossover_min_neighbors == 400
    assert hc.max_nn_evals == 20000
    assert hc.fraction_of_nn == pytest.approx(2.0)
    assert hc.resize_to_fit_ram is False
    assert hc.prefix_stat_deme == "Demes"


def test_deme_parameters_wiring():
    """C++: problem-params.cc:2262-2265"""
    pp = _parse()
    dp = pp.deme_params
    assert dp.reduce_all is True
    assert dp.ignore_ops == set()
    assert dp.perm_ratio == pytest.approx(0.0)
    # The landmine -- see problem_params.py's module docstring. Pinned here
    # as a regression: a DEFAULT run genuinely produces linear_contin=False,
    # NOT deme_parameters()'s own bare True default.
    assert dp.linear_contin is False


def test_optim_parameters_wiring():
    """C++: problem-params.cc:2358"""
    pp = _parse()
    op = pp.opt_params
    assert op.opt_algo == "hc"
    assert op.pop_size_ratio == pytest.approx(20.0)
    assert op.terminate_if_gte == pytest.approx(3.4028234663852886e38, rel=1e-6)
    assert op.max_dist == 4


def test_metapop_parameters_wiring():
    """C++: problem-params.cc:2277-2291"""
    pp = _parse()
    mp = pp.meta_params
    assert mp.max_candidates_per_deme == -1
    assert mp.revisit == 0
    assert mp.do_boosting is False
    assert mp.discard_dominated is False
    assert mp.keep_bscore is False
    assert mp.complexity_temperature == pytest.approx(6.0)
    assert mp.cap_coef == pytest.approx(50.0)
    assert mp.jobs == 1
    ep = mp.ensemble_params
    assert ep.do_boosting is False
    assert ep.num_to_promote == 1
    assert ep.exact_experts is True
    assert ep.expalpha == pytest.approx(2.0)
    assert ep.bias_scale == pytest.approx(1.0)


def test_diversity_parameters_populated_even_though_disabled():
    """The CARRY-FORWARD obligation this task exists to discharge: the CLI
    populates `autoscale` (and everything else) BEFORE `enabled()` can ever
    matter, even though `enabled()` is False by default. C++:
    problem-params.cc:2293-2326.
    """
    pp = _parse()
    dv = pp.meta_params.diversity
    assert dv.pressure == pytest.approx(0.0)
    assert dv.autoscale is False  # C++'s own genuinely-uninitialized field
    assert dv.exponent == pytest.approx(-1.0)
    assert dv.normalize is True
    assert dv.enabled() is False
    # dst resolves to p_norm/2.0 (the default), matching
    # diversity_parameters()'s own bare-constructor state exactly.
    assert dv.dst([1.0, 0.0], [0.0, 0.0]) == pytest.approx(1.0)


def test_filter_params_populated_even_though_disabled():
    """Same obligation as diversity, for the ten previously-uninitialized
    `subsample_deme_filter_parameters` fields. C++: problem-params.cc:
    2339-2355.
    """
    pp = _parse()
    fp = pp.filter_params
    assert fp.n_subsample_demes == 0
    assert fp.n_top_candidates == 1
    assert fp.n_tuples == 4294967295
    assert fp.by_time is False
    assert fp.contiguous_time is True
    assert fp.std_dev_threshold == pytest.approx(3.4028234663852886e38, rel=1e-6)
    assert fp.tanimoto_mean_threshold == pytest.approx(1.0)
    assert fp.tanimoto_geo_mean_threshold == pytest.approx(1.0)
    assert fp.tanimoto_max_threshold == pytest.approx(1.0)
    assert fp.n_best_bfdemes == 0
    assert fp.tanimoto_mean_weight == pytest.approx(0.0)
    assert fp.tanimoto_geo_mean_weight == pytest.approx(0.0)
    assert fp.tanimoto_max_weight == pytest.approx(0.0)
    assert fp.n_subsample_fitnesses == 0
    assert fp.low_dev_pressure == pytest.approx(1.0)


def test_bool_reduct_and_bool_reduct_rep_built_at_the_default_efforts():
    pp = _parse()
    assert pp.bool_reduct is not None
    assert pp.bool_reduct_rep is not None
    assert pp.contin_reduct is None  # S14, not ported -- see module docstring


# ---------------------------------------------------------------------------
# The hc-allow-resize-deme deprecated alias: same dest as
# hc-resize-to-fit-ram (C++: problem-params.cc:482-490, a SECOND
# add_options entry bound to the same variable).
# ---------------------------------------------------------------------------


def test_hc_allow_resize_deme_is_a_deprecated_alias_for_the_same_dest():
    pp = _parse("--hc-allow-resize-deme", "1")
    assert pp.hc_params.resize_to_fit_ram is True
    pp = _parse("--hc-resize-to-fit-ram", "1")
    assert pp.hc_params.resize_to_fit_ram is True


# ---------------------------------------------------------------------------
# Setting a SUPPORTED option to a non-default value must actually change
# behaviour and must NOT raise -- the flip side of the gate-and-raise bar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,path,expected",
    [
        (["--hc-crossover", "0"], "hc_params.crossover", False),
        (["--hc-crossover-pop-size", "7"], "hc_params.crossover_pop_size", 7),
        (["--hc-crossover-min-neighbors", "9"], "hc_params.crossover_min_neighbors", 9),
        (["--hc-widen-search", "1"], "hc_params.widen_search", True),
        (["--hc-single-step", "1"], "hc_params.single_step", True),
        (["--hc-resize-to-fit-ram", "1"], "hc_params.resize_to_fit_ram", True),
        (["--hc-max-nn-evals", "42"], "hc_params.max_nn_evals", 42),
        (["--hc-fraction-of-nn", "3.5"], "hc_params.fraction_of_nn", pytest.approx(3.5)),
        (["--max-evals", "5"], "moses_params.max_evals", 5),
        (["--max-gens", "3"], "moses_params.max_gens", 3),
        (["--random-seed", "99"], "rand_seed", 99),
        (["--reduce-all", "0"], "deme_params.reduce_all", False),
        (["--logical-perm-ratio", "-1.0"], "deme_params.perm_ratio", pytest.approx(-1.0)),
        (["--linear-regression", "1"], "deme_params.linear_contin", True),
        (["--max-candidates-per-deme", "7"], "meta_params.max_candidates_per_deme", 7),
        (["--revisit", "3"], "meta_params.revisit", 3),
        (
            ["--complexity-temperature", "9.0"],
            "meta_params.complexity_temperature",
            pytest.approx(9.0),
        ),
        (["--cap-coef", "1.0"], "meta_params.cap_coef", pytest.approx(1.0)),
        (["--max-dist", "9"], "opt_params.max_dist", 9),
        (["--pop-size-ratio", "5"], "opt_params.pop_size_ratio", pytest.approx(5.0)),
        (["--reduct-candidate-effort", "0"], "reduct_candidate_effort", 0),
        (["--reduct-knob-building-effort", "1"], "reduct_knob_building_effort", 1),
        (["--diversity-autoscale", "1"], "meta_params.diversity.autoscale", True),
        (["--fs-target-size", "5"], "fs_params.target_size", 5),
        (["--ss-n-subsample-demes", "1"], "filter_params.n_subsample_demes", 1),
    ],
)
def test_supported_non_default_option_applies_and_does_not_raise(argv, path, expected):
    pp = _parse(*argv)
    assert _get(pp, path) == expected


# ---------------------------------------------------------------------------
# Gate-and-raise: every unported subsystem raises NotImplementedError when
# engaged, and each guard is checked NOT to fire at its own default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--opt-algo", "nosuchalgo"],
        ["--enable-fs", "1"],
        ["--boost", "1"],
        ["--discard-dominated", "1"],
        ["--diversity-pressure", "0.1"],
        ["--ss-n-subsample-demes", "2"],
        ["--ss-n-subsample-fitnesses", "2"],
        ["--discretize-threshold", "1.0"],
        ["--contin-depth", "6"],
        ["--it-abs-err", "1"],
        ["--time-bscore", "1"],
        ["--time-dispersion-pressure", "0.1"],
        ["--well-enough", "1"],
        ["--jobs", "2:otherhost"],
        ["-t", "1"],
        ["--output-bscore", "1"],
        ["--output-deme-id", "1"],
    ],
)
def test_unsupported_option_raises(argv):
    with pytest.raises(NotImplementedError):
        _parse(*argv)


def test_output_bscore_raises_at_parse_time_naming_merge_demes():
    """[Important 1] `-t`/`--output-bscore` used to parse fine and only
    crash deep inside a real run (`metapopulation.merge_demes`, several
    stack frames past `-m` evaluations having already executed) with an
    unhandled traceback. Pins that it now fails immediately, at
    `parse_options` time, naming the blocking subsystem -- the same
    contract every other Phase-2 gate in this file already gives.
    """
    with pytest.raises(NotImplementedError, match="merge_demes"):
        _parse("-t", "1")


def test_output_deme_id_raises_at_parse_time_naming_demeid():
    """[Important 1] Same proof as above for `--output-deme-id`, whose
    blocking gap is `scored_combo_tree` carrying no `demeID_t`."""
    with pytest.raises(NotImplementedError, match="demeID"):
        _parse("--output-deme-id", "1")


def test_default_command_line_never_raises():
    """The exit criterion's other half: an unmodified command line must
    never raise. Guards against an over-broad gate condition."""
    _parse()  # must not raise


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--opt-algo", "hc"],
        ["--enable-fs", "0"],
        ["--boost", "0"],
        ["--discard-dominated", "0"],
        ["--diversity-pressure", "0.0"],
        ["--ss-n-subsample-demes", "0"],
        ["--ss-n-subsample-demes", "1"],  # "0 or 1" both disactivated, header's own comment
        ["--ss-n-subsample-fitnesses", "1"],
        ["--contin-depth", "5"],
        ["--it-abs-err", "0"],
        ["--time-bscore", "0"],
        ["--time-dispersion-pressure", "0.0"],
        ["--well-enough", "0"],
        ["--jobs", "4"],  # local job count, not a remote host spec
        ["-t", "0"],  # -t/--output-bscore's own default
        ["--output-bscore", "0"],
        ["--output-deme-id", "0"],  # --output-deme-id's own default
        # Sub-options that are provably inert while their master switch
        # stays at ITS default -- must not raise either (see module
        # docstring's "sub-options... provably inert" paragraph).
        ["--diversity-autoscale", "1"],
        ["--diversity-exponent", "3.0"],
        ["--diversity-normalize", "0"],
        ["--diversity-dst", "tanimoto"],
        ["--fs-target-size", "999"],
        ["--fs-hc-crossover", "1"],
        ["--boost-promote", "5"],
        ["--boost-expalpha", "9.0"],
        ["--weighted-accuracy", "1"],
        ["--time-bscore-granularity", "month"],
        ["--time-dispersion-exponent", "5.0"],
    ],
)
def test_default_value_or_provably_inert_suboption_never_raises(argv):
    _parse(*argv)  # must not raise


# ---------------------------------------------------------------------------
# ValueError: malformed input, validated unconditionally (matching C++'s own
# not_recognized_*/granularity/log-level checks, independent of any P2 gate).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--diversity-dst", "bogus"],
        ["--diversity-dst2dp", "bogus"],
        ["--time-bscore-granularity", "bogus"],
        ["--ignore-operator", "bogus"],
        ["--include-only-operator", "bogus"],
        ["--log-level", "bogus"],
    ],
)
def test_malformed_input_raises_value_error_regardless_of_gate_state(argv):
    with pytest.raises(ValueError):
        _parse(*argv)


def test_diversity_dst_validated_even_while_diversity_disabled():
    """Proves validation is unconditional, not merely a side effect of the
    P2 gate: pressure stays at its default (disabled) and the bad `--dst`
    is still what raises -- and it raises ValueError, not
    NotImplementedError, so a test can tell "malformed" from "unported"
    apart.
    """
    with pytest.raises(ValueError):
        _parse("--diversity-dst", "bogus")  # pressure defaults to 0.0


# ---------------------------------------------------------------------------
# --version: C++ exits(0) before touching anything else; never fires at its
# own default.
# ---------------------------------------------------------------------------


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        _parse("--version")
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# ignore-operator / include-only-operator / linear-regression resolution
# (C++: problem-params.cc:2118-2151).
# ---------------------------------------------------------------------------


def test_ignore_operator_adds_to_the_ignore_set():
    from pymetta.comboreduct.combo.vertex import builtin

    pp = _parse("--ignore-operator", "div", "--ignore-operator", "sin")
    assert pp.deme_params.ignore_ops == {builtin.div, builtin.sin}


def test_include_only_operator_starts_from_full_set_minus_named():
    from pymetta.comboreduct.combo.vertex import builtin

    pp = _parse("--include-only-operator", "plus")
    expected = {builtin.times, builtin.div, builtin.exp, builtin.log, builtin.sin, builtin.impulse}
    assert pp.deme_params.ignore_ops == expected


def test_linear_regression_adds_div_exp_log_sin_to_ignore_ops():
    from pymetta.comboreduct.combo.vertex import builtin

    pp = _parse("--linear-regression", "1")
    assert pp.deme_params.ignore_ops == {builtin.div, builtin.exp, builtin.log, builtin.sin}
    assert pp.deme_params.linear_contin is True


def test_ignore_operator_and_linear_regression_combine():
    from pymetta.comboreduct.combo.vertex import builtin

    pp = _parse("--linear-regression", "1", "--ignore-operator", "plus")
    expected = {builtin.div, builtin.exp, builtin.log, builtin.sin, builtin.plus}
    assert pp.deme_params.ignore_ops == expected


# ---------------------------------------------------------------------------
# jobs / local resolution (C++: problem-params.cc:2157-2172).
# ---------------------------------------------------------------------------


def test_jobs_default_is_localhost_one():
    pp = _parse()
    assert pp.moses_params.jobs == {"localhost": 1}
    assert pp.moses_params.local is True


def test_jobs_bare_number_overwrites_localhost_count():
    pp = _parse("--jobs", "4")
    assert pp.moses_params.jobs == {"localhost": 4}
    assert pp.moses_params.local is True


def test_jobs_host_spec_is_not_local_and_raises():
    # covered by test_unsupported_option_raises too; this test isolates the
    # `jobs`/`local` bookkeeping specifically (message content, not just
    # exception type).
    with pytest.raises(NotImplementedError, match="distributed"):
        _parse("--jobs", "8:remote.example.org")


# ---------------------------------------------------------------------------
# score_t narrowing (CARRY-FORWARD.md 4.2): score_t/float-typed CLI options
# must be the binary32-nearest value, not raw binary64. Distinguishing
# decimal literal picked so float32 and float64 parses disagree.
# ---------------------------------------------------------------------------


def test_score_t_options_are_narrowed_to_binary32():
    from pymetta.moses.moses.types import _to_score_t

    raw = 0.1 + 0.2  # a value whose float64 and float32 nearest reps differ
    pp = _parse("--complexity-temperature", repr(raw))
    assert pp.complexity_temperature == _to_score_t(raw)
    assert pp.complexity_temperature != raw or _to_score_t(raw) == raw


def test_ss_float_fields_are_narrowed_to_binary32():
    from pymetta.moses.moses.types import _to_score_t

    raw = 0.1 + 0.2
    pp = _parse("--ss-std-dev-threshold", repr(raw))
    assert pp.filter_params.std_dev_threshold == _to_score_t(raw)


# ---------------------------------------------------------------------------
# The reference-alias construction: problem_params.fs_params IS
# festor_params.fs_params, the same object (C++: problem-params.cc:151,
# `fs_params(festor_params.fs_params)`).
# ---------------------------------------------------------------------------


def test_fs_params_is_the_same_object_as_festor_params_fs_params():
    pp = problem_params()
    assert pp.fs_params is pp.festor_params.fs_params


def test_diversity_dst_enum_resolution_matches_p_norm_default():
    pp = _parse()
    assert pp.meta_params.diversity._dst_kind is diversity_parameters.dst_enum_t.p_norm
    assert pp.meta_params.diversity.dst2dp_type is diversity_parameters.dst2dp_enum_t.inverse
