"""C++: moses/moses/main/problem-params.{h,cc}

`problem_params` -- the 127-flag CLI option surface (measured: `git grep -c
"po::value<"` gives 127 live sites, one more commented out at
problem-params.cc:1871-1877; the plan's "168" counted something broader and
is not reproduced here) and the `moses_parameters`/`metapop_parameters`/
`hc_parameters`/`deme_parameters`/`optim_parameters` wiring `parse_options`
drives from it. Per `docs/MODULE-COVERAGE.md`'s own non-goal note on this
file ("2407 lines of boost::program_options boilerplate; only the option
names and defaults are reproduced, via argparse"), this is NOT a line-by-line
port: help text is a short paraphrase, not `problem-params.cc`'s full prose.

**Measured against the real compiled binary**, not just the source: every
one of the 127 flag names below (short letters included) was diffed against
`$CLASSIC/build/moses/moses/main/moses --help`'s actual option list and
matches exactly (no flag missing, no invented flag). `--mpi` is deliberately
NOT registered: `CLAUDE.md`'s own "Known sharp edges" notes `HAVE_MPI` is
never defined in any build this project produces (`FIND_PACKAGE(MPI)` is
commented out), so the REAL classic binary's own `--help` does not list it
either -- confirmed directly, not assumed. `problem_params.enable_mpi`
still exists as a plain field (matching the C++ struct member), permanently
`False`, matching `moses_parameters.mpi`'s own hardcoded `False`
(`moses_params.py`).

The bool-option parser (`_bool_opt`) was measured against the compiled
binary too: `--balance=<TRUE|yes|on|1|0|no|off|false>` all succeed (case
folded), while `--balance=2` fails with boost's own message -- "Valid
choices are 'on|off', 'yes|no', '1|0' and 'true|false'" -- quoted verbatim
from the actual error text the binary prints, not from boost's docs.

## Scope: which options are WIRED vs. stored flat

Per this task's own brief, wiring is scoped to exactly the five structs
`parse_options` (problem-params.cc:2058-2404) actually assembles:
`moses_parameters`, `metapop_parameters` (+ its nested `diversity_parameters`/
`subsample_deme_filter_parameters`/`ensemble_parameters`), `hc_parameters`,
`deme_parameters`, `optim_parameters` -- plus the two `reduct::logical_reduction`
rule objects, since `logical_reduction` (S7) is fully ported and directly
analogous. Options whose only C++ consumer lives OUTSIDE `problem-params.cc`
itself (`balance`/`nsamples` -> `table-problems.cc`'s `ctable.balance()`/
`subsampleTable`; `hardness`/`pre_positive`/`gen_best_tree` -> the `pre`/
`prerec`/`recall`/... bscores; `min_rand_input`/`max_rand_input` -> the
synthetic-sampling problems; `log_level`/`log_file` -> a `Logger` class this
port does not have; `result_count`/most of `output_*` -> the S13-Task-3
`metapop_printer`, which HAS since landed -- `moses_exec.py` wires this
struct's `output_score`/`output_cscore`/`output_only_best`/
`output_eval_number`/`output_with_labels`/`output_format_str`/
`output_file` straight into a real `metapop_printer`, which works end to
end (`test_moses_exec.py`'s own oracle tests exercise `-W1`/`-x1`/`-C1`/
`-V1`/`--output-format=python`/`-o<file>` against it directly)) are stored
as PLAIN FLAT FIELDS on `problem_params`, matching the C++ struct's own
field, with NO attempt to wire them into an object that does not exist yet
-- their eventual gate-if-unsupported logic belongs to whichever future
task actually wires them (`table_problems.py`, `moses_exec.py`), the same
"don't invent a consumer" discipline `problem.py`'s own docstring states
for `option_base`. CORRECTED: `ps_max_particles` used to be listed here
too, as "the same story taken to its limit: `ps_parameters` does not
exist ANYWHERE in this port ... so it is pure stored data". Both halves
went stale -- `optimization/particle_swarm.py` ports `ps_parameters` in
full, and `moses_main.py` now threads a `ps_params` bundle from here all
the way to `particle_swarm`. It is built below, next to `hc_params`,
exactly where classic builds it (:2369-2370). None of the fields that
ARE still stored flat needs its own P2 gate here: each is reachable ONLY
through a problem selector (`-H`) or output path this task does not
dispatch through, so a wrong value cannot diverge from classic via this
file alone (this mirrors `-j host:N`'s reasoning below almost exactly).

**Two fields of that same `output_*` group used to be the deliberate
exception, gated here like any other subsystem's master switch**:
`output_bscore` (`-t`) and `output_deme_id` (`--output-deme-id`) each drive
an ALREADY-WIRED consumer (`meta_params.keep_bscore` below;
`metapop_printer`'s own `output_bscore`/`output_demeID` constructor
arguments, `moses_exec.py`) that reached genuinely unported C++ machinery
the instant it was engaged. NEITHER IS GATED ANY MORE: `--output-deme-id`
was ungated once `scored_combo_tree` grew a real `demeID_t`, and `-t` once
`metapopulation.merge_demes`'s `keep_bscore` arm (`merging.cc:258-296`)
stopped raising -- it turned out to be the same behavioural-score recompute
three other flags already drove. This paragraph is kept, rather than
deleted, because the SHAPE of the trap it describes recurs: a flag whose
consumer is wired but whose consumer's consumer is not fails several stack
frames into a live run, as an unhandled traceback with empty stdout, rather
than at option-parse time. Gate at parse time when that is the situation;
do not gate at parse time merely because a flag is unusual.

## Design rule: gate-then-raise for unported SUBSYSTEMS, not for every field

Two-phase `parse_options`:

1. **Populate everything, unconditionally, exactly the way C++'s own
   `parse_options` does** -- including the fields C++ sets regardless of
   whether the subsystem they belong to is ever enabled (`meta_params.
   ensemble_params.*`, all 15 `filter_params.*` fields, all of `meta_params.
   diversity.*` including `autoscale`). This is deliberate: it is what
   discharges the two CARRY-FORWARD obligations for `diversity_parameters.
   autoscale` and the ten `subsample_deme_filter_parameters` fields ("the
   CLI is what sets them") -- see `_wire_diversity`/`_wire_subsample`
   below. It also reproduces C++'s own ALWAYS-run input validation
   (`not_recognized_dst`/`not_recognized_dst2dp`/the time-bscore-granularity
   check/`not_recognized_combo_operator`) as `ValueError`, regardless of
   whether the validated subsystem is itself enabled -- an invalid
   `--diversity-dst=bogus` fails the same way in both classic and this port
   even with `--diversity-pressure=0` (disabled), because classic's own
   `parse_options` validates it unconditionally too.
2. **Only after every field is populated**, run one fixed-order sequence of
   gate checks and raise `NotImplementedError` on the FIRST unported
   subsystem an option actually engages. Because population always
   completes first, a caller can never observe a `problem_params` whose
   `meta_params.diversity.enabled()` is `True` without the raise having
   already fired -- "the CLI populates them before enabled() can be true"
   is satisfied by ORDERING within one function, not by skipping
   population. This is the choice CARRY-FORWARD's S13 note asked to be
   verified rather than assumed: raising first (before populating) was
   rejected because it would make `--diversity-autoscale=1
   --diversity-pressure=0` (disabled, so harmless in classic) raise here
   too, which would reject a command line classic runs to completion --
   exactly the "must never raise on values that don't diverge from
   classic" bar this task sets. Raising only from the DEDICATED gate
   (`diversity_pressure`/`enabled()`) after population is complete matches
   classic's own observable behaviour: harmless while disabled, and the
   port raises at the exact point classic would have gone on to actually
   USE the unported diversity code.

A field belonging to an unported subsystem is gated ONLY when the C++
source shows it is the subsystem's actual master switch
(`enable_feature_selection`, `boosting`, `opt_algo != "hc"`,
`discretize_thresholds` non-empty, and non-local `--jobs`/`-j host:N`).
Nine options have come OFF this list:
`ss_n_subsample_demes > 1` and `ss_n_subsample_fitnesses > 1` -- the two
`ss-*` master switches, straight from the header's own "if 0 or 1 remains
disactivated"/"Ignored is 0 r 1" comments -- when the `ss-*` subsample
task gave both a real consumer (`deme_expander.py`'s ss-deme branch and
`ss_bscore.py`, see the gate section's own comment);
`diversity_pressure > 0` and `discard_dominated` when the P2
metapopulation task ported `diversity.cc`/`domination.cc` for real
(`0c1beda`/`f1c439f`); `contin_depth != 5` and `it_abs_err` when
`it_table_problem.run`'s contin branch was wired (2026-09-09);
`use_well_enough`, which was never a divergence at all -- classic accepts
it, ignores it off the enum path, and produces byte-identical output, so
gating it made this port REFUSE a command line classic runs; and
`time_bscore`/`time_dispersion_pressure` when the CTable grew real
timestamps and `precision_bscore.do_score`'s two timestamp branches were
ported (2026-09-10). Each un-gating is proven end to end against
classic's own binary; see the gate section's own comments.
Sub-options that are PROVABLY inert while their master switch stays at
its default are never independently gated: `--diversity-autoscale`/
`-exponent`/`-normalize`/`-dst`/`-p-norm`/`-dst2dp` (dead unless `pressure
> 0`, `diversity_parameters.enabled()`'s own definition), every `fs-*`
option (dead unless `enable_feature_selection`, `deme_expander.py`'s own
`fstor is not None` gate), every `boost-*` option (dead unless `boosting`,
only consumed via `meta_params.ensemble_params`), `weighted_accuracy`
(the option's OWN C++ help text: "only used for the discretize_contin_
bscore (when --discretize-threshold is used)"), `time_bscore_granularity`
(only read "in case the bscore is spread across time", its own help
text), and `time_dispersion_exponent` (the same penalty as `-pressure`,
inert at `pressure == 0`). `hc-crossover-pop-size`/`-min-neighbors`/
`hc-resize-to-fit-ram` are NOT gated at all -- crossover and hill-climbing
are fully ported (S9/S12) -- so `--hc-crossover-pop-size=999` legitimately
changes behaviour and must never raise.

## A genuine landmine this task found and does NOT fix

`deme_params.linear_contin = linear_regression` (problem-params.cc:2264)
is UNCONDITIONAL: whatever the CLI's `linear_regression` flag holds
overwrites `linear_contin`, every run. `--linear-regression`'s own CLI
default is `false` (problem-params.cc:292). So a REAL classic invocation
with no `--linear-regression` flag at all sets `deme_parameters.
linear_contin = False` -- NOT `True`, `deme_parameters`'s own bare
member-initializer default (`deme_params.h:49`, `linear_contin(true)`,
faithfully reproduced by `deme_params.py`'s constructor). This port's
OWN `deme_expander.create_representations` (S9/S10, a file this task does
not own) raises `NotImplementedError` whenever `linear_contin is not
True` (`deme_expander.py:267-275), on the stated assumption that
"`linear_contin=True` is C++'s own default too" -- true only of the bare
struct, NOT of a real CLI-driven run, as this measurement shows. The
result: wiring `deme_params.linear_contin = linear_regression` faithfully
(as this file does, since that IS what `problem-params.cc` does,
unconditionally, with no gate of its own) means every DEFAULT `-H it` run
assembled by a future caller that also calls `deme_expander.
create_representations` with this `deme_params` would hit that raise --
even though `build_knobs`'s own dispatch never actually consults
`linear_contin` for a Boolean-output exemplar (`CLAUDE.md`'s own knob-
building dispatch table: purely by inferred OUTPUT type). This is a defect
in the ALREADY-MERGED `deme_expander.py` (an over-broad gate that checks
the FLAG's value rather than whether the exemplar's type would ever read
it), not a classic C++ bug and not something introduced here -- ported
faithfully anyway, per this project's "never silently fix" rule, and
flagged prominently in this task's report for whoever next touches
`deme_expander.py` or wires `problem_params` into it (S13 Task 2/3).
"""

from __future__ import annotations

import argparse
import sys

from pymoses.comboreduct.combo.iostream_combo import _builtin_str_to_vertex, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.reduct.contin_reduction import contin_reduction
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.gomea import gomea_parameters
from pymoses.moses.deme.deme_params import deme_parameters
from pymoses.moses.deme.feature_selector import feature_selector_parameters
from pymoses.moses.metapopulation.metapop_params import (
    diversity_parameters,
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymoses.moses.moses.moses_main import version_string
from pymoses.moses.moses.moses_params import moses_parameters
from pymoses.moses.moses.optimization.hill_climbing import hc_parameters
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.moses.optimization.particle_swarm import ps_parameters
from pymoses.moses.moses.types import _to_score_t, very_best_score
from pymoses.moses.representation.representation import set_depth
from pymoses.util.logger import logger, ostream_double
from pymoses.util.rng import randGen

__all__ = ["parse_args", "problem_params"]

# ---------------------------------------------------------------------------
# C++ file-scope statics (problem-params.cc:47-99) and <climits>/<limits>
# constants this module's defaults cite directly.
# ---------------------------------------------------------------------------

_P_NORM = "p_norm"  # C++: problem-params.cc:47 p_norm
_TANIMOTO = "tanimoto"  # C++: problem-params.cc:48 tanimoto
_ANGULAR = "angular"  # C++: problem-params.cc:49 angular

_AUTO = "auto"  # C++: problem-params.cc:52 auto_str
_INVERSE = "inverse"  # C++: problem-params.cc:53 inverse_str
_COMPLEMENT = "complement"  # C++: problem-params.cc:54 complement_str
_POWER = "power"  # C++: problem-params.cc:55 power_str

# Focus types -- "which data points feature selection within moses should
# focus on" (problem-params.cc:61-62's own comment).
_FOCUS_ALL = "all"  # C++: problem-params.cc:63 focus_all
_FOCUS_ACTIVE = "active"  # C++: problem-params.cc:65 focus_active
_FOCUS_INCORRECT = "incorrect"  # C++: problem-params.cc:67 focus_incorrect
_FOCUS_AI = "ai"  # C++: problem-params.cc:70 focus_ai

# Seed types -- "how to use the features of the exemplar to seed feature
# selection" (problem-params.cc:74-75's own comment).
_SEED_NONE = "none"  # C++: problem-params.cc:80 seed_none
_SEED_ADD = "add"  # C++: problem-params.cc:88 seed_add
_SEED_INIT = "init"  # C++: problem-params.cc:93 seed_init
_SEED_XMPLR = "xmplr"  # C++: problem-params.cc:99 seed_xmplr

_DAY = "day"  # C++: problem-params.cc:58 day_str
_MONTH = "month"  # C++: problem-params.cc:59 month_str

_HC = "hc"  # C++: moses/optimization/optimization.h:68 hc
# GOMEA extension (not in classic MOSES): "gomea" selects the population
# optimizer in `pymoses.gomea`; the other four are classic's own names.
_SUPPORTED_OPT_ALGOS = frozenset({"hc", "sa", "un", "ps", "gomea"})

# GOMEA extension (not in classic MOSES): the `--gomea-*` options as
# (flag, dest, type, default). Registered by `problem_params.add_gomea_options`
# and read back by `_gomea_params_from_args`, which falls back to these same
# defaults for a namespace built without them.
_GOMEA_FOS_CHOICES = ("lt-b", "lt-mi", "rt", "uni")
_GOMEA_OPTIONS = (
    ("--gomea-pop-size", "gomea_pop_size", int, 64),
    ("--gomea-fos", "gomea_fos", str, "lt-b"),
    ("--gomea-fi", "gomea_fi", int, 0),
    ("--gomea-radii", "gomea_radii", str, "1:0.4,2:0.3,4:0.2,max:0.1"),
    ("--gomea-skip-inactive", "gomea_skip_inactive", int, 1),
    ("--gomea-max-gens", "gomea_max_gens", int, 100),
    ("--gomea-stall-gens", "gomea_stall_gens", int, 5),
    ("--gomea-max-evals-per-deme", "gomea_max_evals_per_deme", int, 0),
    ("--gomea-log", "gomea_log", str, ""),
)


def _gomea_params_from_args(args: argparse.Namespace) -> gomea_parameters:
    """GOMEA extension (not in classic MOSES): build the `gomea_parameters`
    the `--gomea-*` options describe. Raises `ValueError` for a malformed
    value, like the other enumerated options `parse_options` checks."""
    v = {dest: getattr(args, dest, default) for _, dest, _, default in _GOMEA_OPTIONS}
    if v["gomea_fos"] not in _GOMEA_FOS_CHOICES:
        raise ValueError(
            f"--gomea-fos {v['gomea_fos']} is not recognized. "
            f"Supported: {', '.join(_GOMEA_FOS_CHOICES)}."
        )
    for dest in ("gomea_fi", "gomea_skip_inactive"):
        if v[dest] not in (0, 1):
            raise ValueError(f"--{dest.replace('_', '-')} must be 0 or 1, got {v[dest]}")
    if v["gomea_max_evals_per_deme"] < 0:
        raise ValueError(
            f"--gomea-max-evals-per-deme must be >= 0 (0 = no cap), "
            f"got {v['gomea_max_evals_per_deme']}"
        )
    return gomea_parameters(
        pop_size=v["gomea_pop_size"],
        fos=v["gomea_fos"],
        forced_improvement=bool(v["gomea_fi"]),
        skip_inactive=bool(v["gomea_skip_inactive"]),
        radii=v["gomea_radii"],
        max_generations=v["gomea_max_gens"],
        stall_generations=v["gomea_stall_gens"],
        max_evals_per_deme=v["gomea_max_evals_per_deme"] or None,
        telemetry_path=v["gomea_log"] or None,
    )


_DEFAULT_LOG_FILE = "moses.log"  # C++: moses_exec_def.h:56 default_log_file
_LOCALHOST = "localhost"  # C++: moses_exec_def.h:69 localhost
_JOB_SEPARATOR = ":"  # C++: moses_exec_def.h:68 job_seperator

_INT32_MAX = 2147483647  # C++: <climits> INT_MAX (problem-params.cc:266 max_time default)
_UINT_MAX = 0xFFFFFFFF  # C++: <climits> UINT_MAX (problem-params.cc:1940 ss-n-tuples default)
_FLT_MAX = 3.4028234663852886e38  # C++: std::numeric_limits<float>::max() (ss-std-dev-threshold)

# C++: opencog/util/Logger.h Logger::Level string names (get_level_from_string);
# validated the same way problem-params.cc:2093-2101 validates it unconditionally,
# even though this port has no Logger to actually configure (see module docstring's
# "stored flat" section) -- validate-but-don't-consume, not a fabricated subsystem.
_LOG_LEVELS = frozenset({"NONE", "ERROR", "WARN", "INFO", "DEBUG", "FINE"})

_COMBO_OPERATORS = {
    "plus": builtin.plus,
    "times": builtin.times,
    "div": builtin.div,
    "exp": builtin.exp,
    "log": builtin.log,
    "sin": builtin.sin,
    "impulse": builtin.impulse,
}


def _bool_opt(s: str) -> bool:
    """Not itself a ported C++ symbol -- argparse `type=` callback standing in
    for boost::program_options' `validate(any&, vector<string>&, bool*, int)`.

    MEASURED against `$CLASSIC/build/moses/moses/main/moses` directly (see
    module docstring): `on/off`, `yes/no`, `1/0`, `true/false` all accepted,
    case-folded (`--balance=TRUE` succeeds); anything else fails with boost's
    own message, quoted in the `raise` below verbatim.
    """
    t = s.strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"the argument ('{s}') is invalid. Valid choices are 'on|off', "
        "'yes|no', '1|0' and 'true|false'"
    )


class problem_params:
    """C++: moses/moses/main/problem-params.h:56 problem_params

    See this module's docstring for scope, the two-phase gate design, and
    the `deme_expander.py`/`linear_contin` landmine this task found.
    """

    def __init__(self) -> None:
        """C++: moses/moses/main/problem-params.cc:149 problem_params::problem_params

        `enable_mpi(false), default_nsamples(20), fs_params(festor_params.
        fs_params), max_filename_size(255)` -- ported verbatim. `fs_params`
        is C++'s reference-alias member: the SAME object as
        `self.festor_params.fs_params`, not a copy (see `CARRY-FORWARD.md`
        4.1's "the aliasing is correct" precedent for a `const&` member;
        this one is non-const in C++ too, and nothing here ever reassigns
        either name to a different object, only mutates fields through
        whichever name is in hand).
        """
        self.enable_mpi = False
        self.default_nsamples = 20
        self.max_filename_size = 255
        self.festor_params = feature_selector_parameters()
        self.fs_params = self.festor_params.fs_params

    # ------------------------------------------------------------- add_options

    def add_options(self, parser: argparse.ArgumentParser) -> None:
        """C++: moses/moses/main/problem-params.cc:153 problem_params::add_options

        Registers all 127 flags (measured count, see module docstring) plus
        `--version`/`-F, --log-file-dep-opt` (2 more, no `po::value` of their
        own in C++ either -- plain presence flags here too). `--mpi` is
        deliberately not registered (module docstring). Help text is a short
        paraphrase, not `problem-params.cc`'s full prose -- this file's own
        scope note and `docs/MODULE-COVERAGE.md`'s "not line-by-line" say so
        explicitly.
        """
        add = parser.add_argument

        def opt(*flags: str, dest: str, type_, default, **kw) -> None:
            add(*flags, dest=dest, type=type_, default=default, **kw)

        def sopt(*flags: str, dest: str, default: bool, **kw) -> None:
            opt(*flags, dest=dest, type_=_bool_opt, default=default, **kw)

        def vopt(*flags: str, dest: str, type_=str) -> None:
            add(*flags, dest=dest, type=type_, action="append", default=[])

        # -- General options --
        vopt("-j", "--jobs", dest="jobs_str")
        opt("--min-pool", dest="min_pool", type_=int, default=50)
        vopt("-e", "--exemplar", dest="exemplars_str")

        # -- Problem-type options --
        opt("-H", "--problem", dest="problem", type_=str, default="it")
        opt("--nsamples", dest="nsamples", type_=int, default=-1)
        sopt("--balance", dest="balance", default=False)

        # -- Algorithm control options --
        opt("-a", "--opt-algo", dest="opt_algo", type_=str, default=_HC)
        opt("-A", "--max-score", dest="max_score", type_=float, default=float(very_best_score))
        opt("-m", "--max-evals", dest="max_evals", type_=int, default=10000)
        opt("--max-time", dest="max_time", type_=int, default=_INT32_MAX)
        opt("-s", "--cache-size", dest="cache_size", type_=int, default=3000)
        vopt("-n", "--ignore-operator", dest="ignore_ops_str")
        sopt("--linear-regression", dest="linear_regression", default=False)
        opt("--logical-perm-ratio", dest="perm_ratio", type_=float, default=0.0)
        opt("-r", "--random-seed", dest="rand_seed", type_=int, default=1)
        opt(
            "-v",
            "--complexity-temperature",
            dest="complexity_temperature",
            type_=float,
            default=6.0,
        )
        opt("-z", "--complexity-ratio", dest="complexity_ratio", type_=float, default=3.5)
        opt("--cap-coef", dest="cap_coef", type_=float, default=50.0)

        # -- Large problem / hill-climbing parameters --
        opt("--hc-max-nn-evals", dest="hc_max_nn", type_=int, default=20000)
        opt("--hc-fraction-of-nn", dest="hc_frac_of_nn", type_=float, default=2.0)
        sopt("-Z", "--hc-crossover", dest="hc_crossover", default=True)
        opt("--hc-crossover-pop-size", dest="hc_crossover_pop_size", type_=int, default=120)
        opt(
            "--hc-crossover-min-neighbors",
            dest="hc_crossover_min_neighbors",
            type_=int,
            default=400,
        )
        sopt("--hc-resize-to-fit-ram", dest="hc_resize_to_fit_ram", default=False)
        sopt("--hc-allow-resize-deme", dest="hc_resize_to_fit_ram", default=False)

        # -- Particle swarm (stored only, see module docstring) --
        opt("--ps-max-particles", dest="ps_max_particles", type_=int, default=50)

        opt("--contin-depth", dest="contin_depth", type_=int, default=5)

        # -- Algorithm tuning (boosting / ensembles) --
        sopt("--boost", dest="boosting", default=False)
        opt("--boost-promote", dest="num_to_promote", type_=int, default=1)
        sopt("--boost-exact", dest="exact_experts", default=True)
        opt("--boost-expalpha", dest="expalpha", type_=float, default=2.0)
        opt("--boost-bias", dest="bias_scale", type_=float, default=1.0)

        opt(
            "-B",
            "--reduct-knob-building-effort",
            dest="reduct_knob_building_effort",
            type_=int,
            default=2,
        )
        opt("-D", "--max-dist", dest="max_dist", type_=int, default=4)
        sopt("-d", "--reduce-all", dest="reduce_all", default=True)
        opt("-E", "--reduct-candidate-effort", dest="reduct_candidate_effort", type_=int, default=2)
        opt("-g", "--max-gens", dest="max_gens", type_=int, default=-1)
        sopt("--discard-dominated", dest="discard_dominated", default=False)
        sopt("-L", "--hc-single-step", dest="hc_single_step", default=False)
        vopt("-N", "--include-only-operator", dest="include_only_ops_str")
        opt("-P", "--pop-size-ratio", dest="pop_size_ratio", type_=float, default=20.0)
        opt("-p", "--noise", dest="noise", type_=float, default=-1.0)
        sopt("-T", "--hc-widen-search", dest="hc_widen_search", default=False)
        sopt("--well-enough", dest="use_well_enough", default=False)
        opt("--revisit", dest="revisit", type_=int, default=0)

        # -- Output control options --
        opt("-c", "--result-count", dest="result_count", type_=int, default=10)
        sopt("-S", "--output-score", dest="output_score", default=True)
        sopt("-x", "--output-cscore", dest="output_cscore", default=False)
        sopt("-t", "--output-bscore", dest="output_bscore", default=False)
        sopt("-C", "--output-only-best", dest="output_only_best", default=False)
        sopt("-V", "--output-eval-number", dest="output_eval_number", default=False)
        sopt("-W", "--output-with-labels", dest="output_with_labels", default=False)
        sopt("--output-deme-id", dest="output_deme_id", default=False)
        opt("--output-format", dest="output_format_str", type_=str, default="combo")
        opt("-o", "--output-file", dest="output_file", type_=str, default="")

        # -- The rest --
        opt("-q", "--min-rand-input", dest="min_rand_input", type_=float, default=0.0)
        opt("-w", "--max-rand-input", dest="max_rand_input", type_=float, default=1.0)
        opt("-l", "--log-level", dest="log_level", type_=str, default="INFO")
        add("-F", "--log-file-dep-opt", dest="log_file_dep_opt", action="store_true", default=False)
        opt("-f", "--log-file", dest="log_file", type_=str, default=_DEFAULT_LOG_FILE)
        opt(
            "-M", "--max-candidates-per-deme", dest="max_candidates_per_deme", type_=int, default=-1
        )
        sopt("-G", "--weighted-accuracy", dest="weighted_accuracy", default=False)
        opt("--diversity-pressure", dest="diversity_pressure", type_=float, default=0.0)
        sopt("--diversity-autoscale", dest="diversity_autoscale", default=False)
        opt("--diversity-exponent", dest="diversity_exponent", type_=float, default=-1.0)
        sopt("--diversity-normalize", dest="diversity_normalize", default=True)
        opt("--diversity-dst", dest="diversity_dst", type_=str, default=_P_NORM)
        opt("--diversity-p-norm", dest="diversity_p_norm", type_=float, default=2.0)
        opt("--diversity-dst2dp", dest="diversity_dst2dp", type_=str, default=_AUTO)
        vopt("-R", "--discretize-threshold", dest="discretize_thresholds", type_=float)
        opt("-Q", "--alpha", dest="hardness", type_=float, default=1.0)
        opt("--time-dispersion-pressure", dest="time_dispersion_pressure", type_=float, default=0.0)
        opt("--time-dispersion-exponent", dest="time_dispersion_exponent", type_=float, default=1.0)
        sopt("--time-bscore", dest="time_bscore", default=False)
        opt(
            "--time-bscore-granularity", dest="time_bscore_granularity_str", type_=str, default=_DAY
        )
        sopt("--gen-best-tree", dest="gen_best_tree", default=False)

        # -- Fitness specific params --
        sopt("--it-abs-err", dest="it_abs_err", default=False)
        sopt("--pre-positive", dest="pre_positive", default=True)

        # -- Feature-selection params (P2, stored only) --
        sopt("--enable-fs", dest="enable_feature_selection", default=False)
        opt("--fs-target-size", dest="fs_target_size", type_=int, default=20)
        sopt("--fs-exp-distrib", dest="fs_exp_distrib", default=False)
        opt("--fs-focus", dest="fs_focus", type_=str, default="incorrect")
        opt("--fs-seed", dest="fs_seed", type_=str, default="add")
        sopt("--fs-prune-exemplar", dest="fs_prune_exemplar", default=False)
        opt("--fs-subsampling-ratio", dest="fs_subsampling_ratio", type_=float, default=1.0)
        sopt("--fs-subsampling-by-time", dest="fs_subsampling_by_time", default=False)
        opt("--fs-demes", dest="fs_demes", type_=int, default=1)
        opt("--fs-algo", dest="fs_algo", type_=str, default="simple")
        opt("--fs-scorer", dest="fs_scorer", type_=str, default="mi")
        opt("--fs-threshold", dest="fs_threshold", type_=float, default=0.0)
        opt(
            "--fs-enforce-features-filename",
            dest="fs_enforce_features_filename",
            type_=str,
            default="",
        )
        opt("--fs-diversity-pressure", dest="fs_diversity_pressure", type_=float, default=0.0)
        opt("--fs-diversity-cap", dest="fs_diversity_cap", type_=int, default=100)
        opt("--fs-diversity-interaction", dest="fs_diversity_interaction", type_=int, default=-1)
        sopt("--fs-diversity-jaccard", dest="fs_diversity_jaccard", default=True)
        opt(
            "--fs-inc-redundant-intensity",
            dest="fs_inc_redundant_intensity",
            type_=float,
            default=-1.0,
        )
        opt(
            "--fs-inc-target-size-epsilon",
            dest="fs_inc_target_size_epsilon",
            type_=float,
            default=1.0e-6,
        )
        opt("--fs-inc-interaction-terms", dest="fs_inc_interaction_terms", type_=int, default=1)
        opt("--fs-pre-penalty", dest="fs_pre_penalty", type_=float, default=1.0)
        opt("--fs-pre-min-activation", dest="fs_pre_min_activation", type_=float, default=0.5)
        opt("--fs-pre-max-activation", dest="fs_pre_max_activation", type_=float, default=1.0)
        sopt("--fs-pre-positive", dest="fs_pre_positive", default=True)
        opt("--fs-hc-max-score", dest="fs_hc_max_score", type_=float, default=1.0)
        opt("--fs-hc-max-evals", dest="fs_hc_max_evals", type_=int, default=10000)
        opt(
            "--fs-hc-fraction-of-remaining",
            dest="fs_hc_fraction_of_remaining",
            type_=float,
            default=0.5,
        )
        sopt("--fs-hc-crossover", dest="fs_hc_crossover", default=False)
        opt("--fs-hc-crossover-pop-size", dest="fs_hc_crossover_pop_size", type_=int, default=120)
        opt(
            "--fs-hc-crossover-min-neighbors",
            dest="fs_hc_crossover_min_neighbors",
            type_=int,
            default=400,
        )
        sopt("--fs-hc-widen-search", dest="fs_hc_widen_search", default=True)
        opt("--fs-mi-penalty", dest="fs_mi_penalty", type_=float, default=100.0)
        opt("--fs-smd-top-size", dest="fs_smd_top_size", type_=int, default=10)

        # -- Subsample-deme params (P2, stored only) --
        opt("--ss-n-subsample-demes", dest="ss_n_subsample_demes", type_=int, default=0)
        opt("--ss-n-top-candidates", dest="ss_n_top_candidates", type_=int, default=1)
        opt("--ss-n-tuples", dest="ss_n_tuples", type_=int, default=_UINT_MAX)
        opt("--ss-std-dev-threshold", dest="ss_std_dev_threshold", type_=float, default=_FLT_MAX)
        opt(
            "--ss-tanimoto-mean-threshold",
            dest="ss_tanimoto_mean_threshold",
            type_=float,
            default=1.0,
        )
        opt(
            "--ss-tanimoto-geometric-mean-threshold",
            dest="ss_tanimoto_geo_mean_threshold",
            type_=float,
            default=1.0,
        )
        opt(
            "--ss-tanimoto-max-threshold",
            dest="ss_tanimoto_max_threshold",
            type_=float,
            default=1.0,
        )
        opt("--ss-n-best-bfdemes", dest="ss_n_best_bfdemes", type_=int, default=0)
        opt("--ss-tanimoto-mean-weight", dest="ss_tanimoto_mean_weight", type_=float, default=0.0)
        opt(
            "--ss-tanimoto-geometric-mean-weight",
            dest="ss_tanimoto_geo_mean_weight",
            type_=float,
            default=0.0,
        )
        opt("--ss-tanimoto-max-weight", dest="ss_tanimoto_max_weight", type_=float, default=0.0)
        opt("--ss-n-subsample-fitnesses", dest="ss_n_subsample_fitnesses", type_=int, default=0)
        opt("--ss-low-dev-pressure", dest="ss_low_dev_pressure", type_=float, default=1.0)
        sopt("--ss-by-time", dest="ss_by_time", default=False)
        sopt("--ss-contiguous-time", dest="ss_contiguous_time", default=True)

        add("--version", dest="version", action="store_true", default=False)

    # GOMEA extension (not in classic MOSES).
    def add_gomea_options(self, parser: argparse.ArgumentParser) -> None:
        """Register the `--gomea-*` options (`_GOMEA_OPTIONS`).

        Kept out of `add_options` so that `add_options` still registers
        exactly classic's option surface. `moses_exec` calls this after every
        classic option is registered, so classic's own `--help` lines keep
        their order and wrapping and the GOMEA lines only follow them.
        """
        for flag, dest, type_, default in _GOMEA_OPTIONS:
            parser.add_argument(flag, dest=dest, type=type_, default=default)

    # ------------------------------------------------------------ parse_options

    def parse_options(self, args: argparse.Namespace) -> None:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Two phases -- see module docstring. Phase 1 (population, always
        runs) mirrors `parse_options`' own statement order; `ValueError` is
        raised for malformed input exactly where classic's own
        `not_recognized_*`/granularity/log-level checks would `exit(1)`.
        Phase 2 (gating) raises `NotImplementedError` for the first
        unported subsystem an option actually engages, in the fixed order
        listed in the module docstring.
        """
        # C++: problem-params.cc:2059-2067 -- checked first, exits(0)
        # before touching anything else. Never fires on the default
        # (`False`).
        #
        #     cout << "moses " << version_string << std::endl;
        #     #ifdef HAVE_MPI
        #       cout << "\tMPI support enabled." << std::endl;
        #     #else
        #       cout << "\tNo MPI support." << std::endl;
        #     #endif
        #     exit(0);
        #
        # This used to print NOTHING and only `raise SystemExit(0)`, on the
        # grounds that `version_string` is build-sourced and unportable.
        # Wrong on both lines. The second line is not build-sourced at all:
        # `HAVE_MPI` is chosen at COMPILE time, `FIND_PACKAGE(MPI)` is
        # commented out of $CLASSIC/CMakeLists.txt so no build this checkout
        # produces can define it (CLAUDE.md's own "Known sharp edges" says
        # so, and the compiled binary prints the `#else` line), and this port
        # excludes MPI entirely -- so "\tNo MPI support." is a CONSTANT here,
        # not a variable. The first line is build-sourced only in the weakest
        # sense: 3/7/0 are literal constants in $CLASSIC/CMakeLists.txt:13-15.
        # See `version_string`'s own comment in moses_main.py for why it
        # carries embedded quotes.
        #
        # Printed with `sys.stdout.write`, not `print`, to keep the exact
        # bytes visible at the call site: "moses " + version + "\n", then a
        # literal TAB, then "No MPI support." + "\n" -- 35 bytes total,
        # matching the hexdump of the compiled binary's own stdout.
        if args.version:
            sys.stdout.write(f"moses {version_string}\n")
            sys.stdout.write("\tNo MPI support.\n")
            raise SystemExit(0)

        # -- flat, unwired fields (see module docstring's scope section) --
        self.problem = args.problem
        self.nsamples = args.nsamples
        self.min_rand_input = args.min_rand_input
        self.max_rand_input = args.max_rand_input
        self.balance = args.balance
        self.result_count = args.result_count
        self.output_score = args.output_score
        self.output_cscore = args.output_cscore
        self.output_bscore = args.output_bscore
        self.output_only_best = args.output_only_best
        self.output_eval_number = args.output_eval_number
        self.output_with_labels = args.output_with_labels
        self.output_deme_id = args.output_deme_id
        self.output_format_str = args.output_format_str
        self.output_file = args.output_file
        self.use_well_enough = args.use_well_enough
        self.hardness = _to_score_t(args.hardness)
        self.pre_positive = args.pre_positive
        self.gen_best_tree = args.gen_best_tree
        self.it_abs_err = args.it_abs_err
        self.ps_max_particles = args.ps_max_particles
        self.min_pool = args.min_pool
        # C++: problem-params.cc's own parse_options never reads `noise`
        # (grep-confirmed: zero references in the function body) -- its only
        # consumer is table-problems.cc (an alternative complexity-ratio
        # derivation for the pre/prerec/... bscores), outside this file's
        # scope (module docstring's "stored flat" section).
        self.noise = args.noise

        if args.log_level.strip().upper() not in _LOG_LEVELS:
            raise ValueError(f"Log level {args.log_level} is incorrect (see --help).")
        self.log_level = args.log_level
        self.log_file_dep_opt = args.log_file_dep_opt
        self.log_file = args.log_file
        # C++: moses/moses/main/problem-params.cc:2069-2113 -- classic sets
        # up the logger HERE, inside `parse_options`, right after reading
        # `-l`/`-f`/`-F`, because a later option can still change the log
        # file's location.
        #
        # Called through the namespace rather than imported: `moses_exec`
        # imports this module, so importing it back would be a cycle. It
        # attaches this callable before calling `parse_options`, and the
        # callable is run-once, so this and `moses_exec`'s own fallback call
        # cannot both fire -- moving WHERE the three startup records are
        # emitted without changing WHETHER. The no-op default keeps every
        # caller that hand-builds a namespace working unchanged.
        getattr(args, "_setup_logging", lambda: None)()

        self.enable_feature_selection = args.enable_feature_selection
        self.fs_focus = args.fs_focus
        self.fs_seed = args.fs_seed
        self.fs_enforce_features_filename = args.fs_enforce_features_filename
        self.fs_params.target_size = args.fs_target_size
        self.fs_params.exp_distrib = args.fs_exp_distrib
        self.fs_params.algorithm = args.fs_algo
        self.fs_params.scorer = args.fs_scorer
        self.fs_params.threshold = args.fs_threshold
        self.fs_params.inc_red_intensity = args.fs_inc_redundant_intensity
        self.fs_params.inc_target_size_epsilon = args.fs_inc_target_size_epsilon
        self.fs_params.inc_interaction_terms = args.fs_inc_interaction_terms
        self.fs_params.pre_penalty = args.fs_pre_penalty
        self.fs_params.pre_min_activation = args.fs_pre_min_activation
        self.fs_params.pre_max_activation = args.fs_pre_max_activation
        self.fs_params.pre_positive = args.fs_pre_positive
        self.fs_params.hc_max_score = args.fs_hc_max_score
        self.fs_params.hc_max_evals = args.fs_hc_max_evals
        self.fs_params.hc_fraction_of_remaining = args.fs_hc_fraction_of_remaining
        self.fs_params.hc_crossover = args.fs_hc_crossover
        self.fs_params.hc_crossover_pop_size = args.fs_hc_crossover_pop_size
        self.fs_params.hc_crossover_min_neighbors = args.fs_hc_crossover_min_neighbors
        self.fs_params.hc_widen_search = args.fs_hc_widen_search
        self.fs_params.mi_confi = args.fs_mi_penalty
        self.fs_params.smd_top_size = args.fs_smd_top_size
        self.festor_params.prune_xmplr = args.fs_prune_exemplar
        self.festor_params.subsampling_ratio = args.fs_subsampling_ratio
        self.festor_params.subsampling_by_time = args.fs_subsampling_by_time
        self.festor_params.n_demes = args.fs_demes
        self.festor_params.diversity_pressure = args.fs_diversity_pressure
        self.festor_params.diversity_cap = args.fs_diversity_cap
        self.festor_params.diversity_interaction = args.fs_diversity_interaction
        self.festor_params.diversity_jaccard = args.fs_diversity_jaccard

        # -- random seed (C++: problem-params.cc:2116 randGen().seed(rand_seed);) --
        self.rand_seed = args.rand_seed
        randGen().seed(self.rand_seed)

        # -- operator ignore/include-only resolution (C++: :2118-2151) --
        ignore_ops = self._resolve_ignore_ops(args)

        # -- exemplars (C++: :2154-2155) --
        self.exemplars_str = list(args.exemplars_str)
        self.exemplars = [str_to_combo_tree(s) for s in self.exemplars_str]

        # -- jobs / local (C++: :2157-2172) --
        jobs, local = self._resolve_jobs(args)

        # -- feature-selection focus / seed (C++: :2196-2229) --
        #
        # "Set parameter structures. Please don't systematically use their
        # constructors (if you can avoid it), as it is prone to error,
        # whenever the constructor changes it may silently set the wrong
        # things (as long as they have the same types)." (:2190-2193's own
        # comment.)
        #
        # These two switches are the ONLY writers of `restrict_incorrect`,
        # `restrict_true`, `increase_target_size`, `ignore_xmplr_features`,
        # `init_xmplr_features` and `xmplr_as_feature` -- there is no CLI
        # option bound directly to any of the six. Both chains are plain
        # `if/else if` with NO final `else`, so an unrecognised string
        # silently leaves `feature_selector_parameters`' own
        # member-initializer defaults in place; reproduced exactly, with no
        # validation C++ does not have. (`--fs-focus`/`--fs-seed` are
        # `po::value<string>`, not a checked enum: classic accepts
        # `--fs-focus=nonsense` and runs.)
        if self.fs_focus == _FOCUS_ALL:
            self.festor_params.restrict_incorrect = False
            self.festor_params.restrict_true = False
        elif self.fs_focus == _FOCUS_ACTIVE:
            self.festor_params.restrict_incorrect = False
            self.festor_params.restrict_true = True
        elif self.fs_focus == _FOCUS_INCORRECT:
            self.festor_params.restrict_incorrect = True
            self.festor_params.restrict_true = False
        elif self.fs_focus == _FOCUS_AI:
            self.festor_params.restrict_incorrect = True
            self.festor_params.restrict_true = True

        if self.fs_seed == _SEED_NONE:
            self.festor_params.increase_target_size = False
            self.festor_params.ignore_xmplr_features = True
            self.festor_params.init_xmplr_features = False
            self.festor_params.xmplr_as_feature = False
        elif self.fs_seed == _SEED_ADD:
            self.festor_params.increase_target_size = True
            self.festor_params.ignore_xmplr_features = False
            self.festor_params.init_xmplr_features = False
            self.festor_params.xmplr_as_feature = False
        elif self.fs_seed == _SEED_INIT:
            self.festor_params.increase_target_size = False
            self.festor_params.ignore_xmplr_features = False
            self.festor_params.init_xmplr_features = True
            self.festor_params.xmplr_as_feature = False
        elif self.fs_seed == _SEED_XMPLR:
            self.festor_params.increase_target_size = False
            self.festor_params.ignore_xmplr_features = False
            self.festor_params.init_xmplr_features = False
            self.festor_params.xmplr_as_feature = True

        # -- enforce-features file (C++: :2231-2259) --
        self._parse_enforce_features()

        # -- deme expansion parameters (C++: :2262-2265) --
        self.reduce_all = args.reduce_all
        self.linear_regression = args.linear_regression
        self.perm_ratio = args.perm_ratio
        deme_params = deme_parameters()
        deme_params.reduce_all = self.reduce_all
        deme_params.ignore_ops = ignore_ops
        # DIVERGENCE(C++ moses/moses/main/problem-params.cc:2264): see this
        # module's docstring, "A genuine landmine this task found and does
        # NOT fix" -- ported verbatim, unconditionally, no gate of its own.
        deme_params.linear_contin = self.linear_regression
        deme_params.perm_ratio = self.perm_ratio
        self.deme_params = deme_params

        # -- cache size / boosting cache-clear (C++: :2266-2274,2285-2286) --
        self.cache_size = args.cache_size
        self.ss_n_subsample_demes = args.ss_n_subsample_demes
        self.ss_n_subsample_fitnesses = args.ss_n_subsample_fitnesses
        if self.ss_n_subsample_demes > 1 or self.ss_n_subsample_fitnesses > 1:
            self.cache_size = 0
        self.boosting = args.boosting
        if self.boosting:
            self.cache_size = 0

        # -- metapopulation parameters (C++: :2277-2291) --
        self.max_candidates_per_deme = args.max_candidates_per_deme
        self.revisit = args.revisit
        self.num_to_promote = args.num_to_promote
        self.exact_experts = args.exact_experts
        self.expalpha = args.expalpha
        self.bias_scale = args.bias_scale
        self.discard_dominated = args.discard_dominated
        self.complexity_temperature = _to_score_t(args.complexity_temperature)
        self.complexity_ratio = _to_score_t(args.complexity_ratio)
        self.cap_coef = args.cap_coef
        meta_params = metapop_parameters(
            max_candidates_per_deme=self.max_candidates_per_deme,
            revisit=self.revisit,
            complexity_temperature=self.complexity_temperature,
            jobs=jobs[_LOCALHOST],
        )
        meta_params.do_boosting = self.boosting
        meta_params.ensemble_params.do_boosting = self.boosting
        meta_params.ensemble_params.num_to_promote = self.num_to_promote
        meta_params.ensemble_params.exact_experts = self.exact_experts
        meta_params.ensemble_params.expalpha = self.expalpha
        meta_params.ensemble_params.bias_scale = self.bias_scale
        meta_params.discard_dominated = self.discard_dominated
        meta_params.keep_bscore = self.output_bscore
        meta_params.cap_coef = self.cap_coef
        self.meta_params = meta_params

        # -- diversity parameters (C++: :2293-2326), full population then
        # gate -- see module docstring's design-rule section --
        self._wire_diversity(args, meta_params)

        # -- time bscore granularity (C++: :2328-2337, validated
        # unconditionally like diversity's dst/dst2dp) --
        gran = args.time_bscore_granularity_str
        if gran not in (_DAY, _MONTH):
            raise ValueError(f"Granularity {gran} not implemented")
        self.time_bscore_granularity_str = gran
        self.time_bscore = args.time_bscore
        self.time_dispersion_pressure = _to_score_t(args.time_dispersion_pressure)
        self.time_dispersion_exponent = _to_score_t(args.time_dispersion_exponent)

        # -- subsampling deme/fitness parameters (C++: :2339-2355) --
        self._wire_subsample(args)

        # -- contin / discretization (S14, stored + gated below) --
        self.weighted_accuracy = args.weighted_accuracy
        self.discretize_thresholds = list(args.discretize_thresholds)
        self.contin_depth = args.contin_depth

        # -- optim_parameters (C++: :2358) --
        self.opt_algo = args.opt_algo
        self.pop_size_ratio = args.pop_size_ratio
        self.max_score = _to_score_t(args.max_score)
        self.max_dist = args.max_dist
        self.opt_params = optim_parameters(
            self.opt_algo, self.pop_size_ratio, self.max_score, self.max_dist
        )

        # -- hc_parameters (C++: :2360-2368) --
        self.hc_widen_search = args.hc_widen_search
        self.hc_single_step = args.hc_single_step
        self.hc_crossover = args.hc_crossover
        self.hc_crossover_pop_size = args.hc_crossover_pop_size
        self.hc_crossover_min_neighbors = args.hc_crossover_min_neighbors
        self.hc_max_nn = args.hc_max_nn
        self.hc_frac_of_nn = args.hc_frac_of_nn
        self.hc_resize_to_fit_ram = args.hc_resize_to_fit_ram
        # C++: problem-params.cc:2360-2368. `hc_params` is a DEFAULT-
        # CONSTRUCTED member of `problem_params` (problem-params.h:220) and
        # every field is ASSIGNED afterwards -- eight plain assignments, no
        # constructor call. This port used to route five of them through
        # `hc_parameters(...)`'s constructor instead, which looks equivalent
        # and is not: the constructor carries `OC_ASSERT(0.0 <
        # fraction_of_nn)` (hill-climbing.h:51) and a plain assignment does
        # not. MEASURED consequence before the fix: classic accepts
        # `--hc-fraction-of-nn=0` (and `=-1`) and exits 0, because its own
        # assert ran once, at construction, against the 2.0 default and was
        # never re-run; this port raised `ValueError: fraction_of_nn must be
        # positive` and refused to start. `hc_parameters.__init__` keeps the
        # assert -- it is faithful to the constructor -- and this block stops
        # going through it, which is faithful to the caller.
        #
        # `--hc-fraction-of-nn=0` is not inert, either: `number_of_new_
        # instances *= hc_params.fraction_of_nn` (hill-climbing.cc:531) then
        # zeroes the count. Measured on `-Hpa -k4 -m5000`, classic prints
        # three DIFFERENT candidate lists for 0, 0.5 and 2 -- so this is a
        # value that flows through to the search, not one the optimizer
        # shrugs off.
        hc_params = hc_parameters()
        hc_params.widen_search = self.hc_widen_search
        hc_params.single_step = self.hc_single_step
        hc_params.crossover = self.hc_crossover
        hc_params.crossover_pop_size = self.hc_crossover_pop_size
        hc_params.crossover_min_neighbors = self.hc_crossover_min_neighbors
        hc_params.max_nn_evals = self.hc_max_nn
        hc_params.fraction_of_nn = self.hc_frac_of_nn
        hc_params.resize_to_fit_ram = self.hc_resize_to_fit_ram
        hc_params.prefix_stat_deme = "Demes"
        self.hc_params = hc_params

        # -- ps_parameters (C++: :2369-2370 `ps_params.max_parts =
        # ps_max_particles`) -- `problem_params` owns a whole
        # `ps_parameters` by value in C++ too (problem-params.h:222), and
        # `max_parts` is the ONE field the CLI touches; every other field
        # keeps the class's own default (particle-swarm.h:50-...).
        ps_params = ps_parameters()
        ps_params.max_parts = self.ps_max_particles
        self.ps_params = ps_params

        # GOMEA extension (not in classic MOSES): the `-a gomea` settings,
        # handed to `metapop_moses_results` next to `hc_params`/`ps_params`.
        self.gomea_params = _gomea_params_from_args(args)

        # -- Set depth (C++: :2371-2372 `set_depth(contin_depth)`) --
        # `representation.py`'s module-level `_depth` static, exactly as
        # classic sets its own process-wide `representation.cc:63` static
        # here. That module's own comment said "wiring that flag is
        # moses/main's job, out of this module's owned-file scope; these
        # setters exist so that later slice has something to call" -- this
        # is that call. Until 2026-09-09 it was missing and `--contin-depth`
        # was gated instead, on the (then-true, now-false) grounds that no
        # contin path could read it.
        set_depth(self.contin_depth)

        # -- moses_parameters (C++: :2374-2382) --
        self.max_evals = args.max_evals
        self.max_gens = args.max_gens
        self.max_time = args.max_time
        moses_params = moses_parameters(vm=None, jobs=jobs)
        moses_params.local = local
        moses_params.mpi = self.enable_mpi
        moses_params.max_evals = self.max_evals
        moses_params.max_gens = self.max_gens
        moses_params.max_score = self.max_score
        moses_params.max_time = self.max_time
        moses_params.max_cnd_output = self.result_count
        self.moses_params = moses_params

        # -- reduct rules (C++: :2384-2394) --
        self.reduct_candidate_effort = args.reduct_candidate_effort
        self.reduct_knob_building_effort = args.reduct_knob_building_effort
        self.lr = logical_reduction(ignore_ops)
        self.bool_reduct = self.lr(self.reduct_candidate_effort)
        self.bool_reduct_rep = self.lr(self.reduct_knob_building_effort)
        # `reduct::contin_reduction(reduct_candidate_effort, ignore_ops)`
        # (problem-params.cc:2393-2394), matching C++'s own choice of ONE
        # rule object for both candidate AND knob-building reduction here
        # (unlike bool_reduct/bool_reduct_rep's two separate efforts) --
        # `contin_reduction`'s only two real call sites, table-problems.cc's
        # REGRESSION macro and demo-problems.cc's polynomial_problem, both
        # pass `*pms.contin_reduct` for BOTH `reduct_cand` and `reduct_rep`.
        # `contin_reduction` (S14 Task 4, `d439711`) and `contin_bscore`
        # (S14 Task 3, `07027b0`) are both genuinely ported and tested now
        # -- the DIVERGENCE this replaces ("contin reduction is S14, not
        # ported anywhere in this port") was true when written and is false
        # today; `self.contin_reduct = None` was a stale claim, not a
        # measurement.
        #
        # A DIVERGENCE marker stood here saying that wiring a real
        # `contin_reduct` "does NOT make contin regression reachable end to
        # end through THIS port's CLI", because `it_table_problem.run`'s
        # contin branch and `polynomial_problem.run` (`-H sr`) both raised
        # unconditionally before reading it. That was measured and true on
        # 2026-09-08. `it_table_problem.run`'s contin branch was wired on
        # 2026-09-09, so the claim no longer holds for `-H it`; `-H sr` is
        # tracked separately in `demo_problems.py`.
        self.contin_reduct = contin_reduction(self.reduct_candidate_effort, ignore_ops)

        # ---------------- Phase 2: gate unported subsystems, fixed order ----------------
        # All four algo names classic recognises (`optimization.h:66-69`) are
        # now ported and dispatched at `moses_main.py`'s
        # `metapop_moses_results_b`.
        #
        # DIVERGENCE(C++ moses/moses/moses/moses_main.h:288-306): classic
        # validates an unrecognised `--opt-algo` name only at DISPATCH time,
        # deep inside `metapop_moses_results_b` (:300-306) -- after `-H`'s
        # problem is resolved, the table loaded and the exemplar built --
        # printing "Unknown optimization algo ..." and `exit(1)`. This
        # correction replaces an EARLIER, wrong version of this same
        # comment that claimed the opposite ("NOT rejected here... this
        # port raises there too") while the very next line already raised
        # here -- measured directly: `moses_exec(["moses", "-i", D, "-m",
        # "30", "-a", "zz"])` raises from THIS `parse_options` call
        # (`problem_params.py`), never reaching `moses_main.py`'s dispatch
        # check at all. The raise below is deliberately kept, not deleted:
        # it fails immediately, before any table I/O or evaluation, the
        # same "clean, immediate failure over a deep unhandled traceback"
        # choice already made -- and disclosed -- for `-t/--output-bscore`
        # and `--output-deme-id` further down this same method. The outcome
        # (not just the timing) is identical either way: all four names
        # classic recognises dispatch identically in both languages, and
        # any other name fails in both. `metapop_moses_results_b`
        # (moses_main.py) still carries classic's own dispatch-time check
        # too, with its own correct DIVERGENCE marker (raises `ValueError`,
        # not `exit(1)`) and its own test (`test_moses_main.py`) -- that
        # code path is unreachable from a run that goes through THIS
        # method first, but stays correct for a caller that builds/mutates
        # a `problem_params` directly, bypassing `parse_options` (the same
        # "direct-call fidelity" caveat `table_problems.py`'s own
        # `_regression` docstring states for its own redundant checks).
        if self.opt_algo not in _SUPPORTED_OPT_ALGOS:
            raise NotImplementedError(
                f"opt-algo {self.opt_algo!r} is not a ported optimizer. "
                f"Supported: {', '.join(sorted(_SUPPORTED_OPT_ALGOS))}."
            )
        if not local:
            raise NotImplementedError(
                "-j host:REMOTE_HOST distributed job dispatch is not ported "
                "(distributed_moses.cc is a different, out-of-scope mechanism); "
                "use local job counts only."
            )
        # DIVERGENCE(C++ moses/moses/deme/feature_selector.cc, moses/moses/
        # main/table-problems.cc:328-334 REGRESSION macro): `feature_
        # selector.py` itself and `deme_expander.py`'s CONSUMPTION of a
        # real `fstor` are both genuinely ported now (`f4ca02c`). What
        # blocks this flag end to end is the CONSTRUCTION site: classic's
        # `REGRESSION` macro builds `new feature_selector(TABLE,
        # festor_params)` and assigns it to `pms.deme_params.fstor`
        # whenever `enable_feature_selection and fs_params.target_size >
        # 0` (both true at classic's own defaults: `--fs-target-size`
        # defaults to 20). `table_problems.py`'s own port of that macro
        # (`_regression`, outside this task's owned scope) still only
        # raises `NotImplementedError` at that exact spot instead of
        # constructing the selector -- measured directly: with this gate
        # removed, `moses_exec(["moses", "-i", D, "-m1000",
        # "--enable-fs=1", "-W1"])` raises from `table_problem_base.
        # _regression` ("feature_selector is not ported anywhere in this
        # project"), a claim that is ALSO false today and is reported as
        # a follow-up for whoever owns `table_problems.py` next. Kept
        # gated here because removing this check alone would not make
        # `--enable-fs` reach working behaviour -- it would just move an
        # (inaccurate) raise one file deeper.
        # (gate removed; see the comment above)
        # DIVERGENCE(C++ moses/moses/scoring/scoring_base.h/.cc
        # bscore_base::call_ensemble): `ensemble.py` (the AdaBoost/experts
        # engine itself) is fully ported (`370146c`) and `merging.py`
        # `--boost` is NOT gated. It was, on the grounds that no concrete
        # scorer overrode `bscore_base.call_ensemble`; `logical_bscore`
        # now does (demo_problems.py), which is precisely the set of
        # problems classic supports boosting on. On a TABLE problem
        # classic itself aborts inside `bscore_base::get_error` ("bscore
        # error not implemented for bscorer ... ctruth_table_bscore",
        # scoring_base.cc:115) -- measured, exit 134 -- and this port
        # raises there for the same reason, so the behaviours agree
        # without a gate here.
        # `discard_dominated` is REAL now (P2 metapopulation task,
        # `f1c439f` ports domination.cc; merging.py's merge_candidates/
        # merge_demes both call the real get_nondominated_disjoint/
        # remove_dominated path unconditionally). Proven end to end
        # against classic's own binary: `-i tests/fixtures/table_data/
        # dataset.csv -m1000 --discard-dominated=1 -W1` gives the
        # byte-for-byte identical 2-line result in both languages (see
        # `test_moses_exec.py`'s own oracle test for this flag).
        #
        # `diversity_pressure > 0` is REAL now too (P2 metapopulation
        # task, `0c1beda` ports diversity.cc; merging.py's merge_demes
        # calls the real gather_diversity_stats/set_diversity path
        # unconditionally whenever `diversity_enabled()`). Proven the same
        # way: `-i tests/fixtures/table_data/dataset.csv -m1000
        # --diversity-pressure=0.1 -W1` gives the byte-for-byte identical
        # 10-line result in both languages (`test_moses_exec.py`'s own
        # oracle test for this flag).
        #
        # Neither is gated here any more -- see the module docstring's own
        # "gated ONLY when" paragraph, updated alongside this change.
        # The whole `ss-*` family was gated here until the `ss-*`
        # subsample task, on the grounds that neither master switch had a
        # real consumer. Both do now:
        # `deme_expander.py`'s `n_subsample_demes`-splitting branch
        # (create_demes/optimize_demes/subsample_by_row/subsample_by_time)
        # and the SS-fitness scorer `ss_bscore.py`, wired in
        # `moses_main.metapop_moses_results` exactly where moses_main.h:364
        # wires it. `subsample.py` (merging.py's `ss_filter`/
        # `ss_score_dev_filter`/tanimoto-stats family) was already ported
        # (`283b9e7`) and is what those two feed. Measured end to end
        # against classic's own binary on all 27 previously-gated sweep
        # rows -- see tests/moses/main/test_ss_cli.py for the byte-for-byte
        # oracles. Two `ss-*` options remain UNVERIFIABLE against this
        # oracle build rather than ungated: every tanimoto-based option
        # (`--ss-tanimoto-*`, `--ss-n-best-bfdemes`) makes classic
        # SIGABRT, because `HAVE_BOOST` is defined nowhere in this
        # checkout and cogutil's `tanimoto_distance` throws
        # "Compiled without boost support" (numeric.h:378) in that case.
        # That is a property of the oracle binary, not of this port, and
        # it predates this task -- `--diversity-dst=tanimoto` aborts
        # classic identically at HEAD.
        # `--discretize-threshold` was gated here until 2026-09-10, on the
        # accurate grounds that `discretize_contin_bscore` did not exist.
        # It is ported now (`bscores.py`) and wired into
        # `it_table_problem.run`'s contin arm, so the flag reaches its
        # scorer and the gate is gone.
        # `--contin-depth` and `--it-abs-err` were BOTH gated here until
        # 2026-09-09, each on the same stated grounds: that their only
        # CLI-reachable consumers (`it_table_problem`'s contin branch and
        # `polynomial_problem`/`-H sr`) "raise unconditionally before
        # reading this flag". That was true when the gates were written
        # and false once the contin branch was wired, so both gates are
        # gone: `--contin-depth` now reaches `set_depth` above (the call
        # whose absence was the gate's other stated reason), and
        # `--it-abs-err` now reaches `contin_bscore`'s err-function
        # selector through `it_table_problem.run`.
        # `--time-bscore` and `--time-dispersion-pressure` were BOTH gated
        # here until 2026-09-10, on the grounds that this port's `CTable`
        # carried no timestamp field at all and so `precision_bscore.
        # do_score`'s two timestamp branches (precision_bscore.cc:242-263)
        # could not be reached. That is no longer true: `TTable`/
        # `TimedValue`/`TimedCounter`/`CTableTime` are ported (`table.py`),
        # `loadTable` fills `Table.ttable` from `--timestamp-feature`,
        # `Table.compressed` keys each observation on `TimedValue`,
        # `scoring_base._copy_ctable` now copies each cell through its own
        # `.copy()` instead of flattening it to a plain `Counter`, and both
        # branches of `do_score` are ported. Both gates are gone.
        #
        # `--time-dispersion-pressure` reaches genuinely undefined
        # behaviour in classic on most timestamped tables -- see
        # `precision_bscore.do_score`'s own DIVERGENCE marker, which
        # records the measured SIGSEGV and what this port prints instead.
        # The flag is still accepted here, because refusing it would make
        # the port reject an input classic accepts, which is the thing
        # this whole check exists to avoid.
        # `--well-enough` is deliberately NOT gated. It was, until
        # 2026-09-09, and that was an over-gate: measured, classic accepts
        # `--well-enough=1` on a boolean table, exits 0, and prints output
        # byte-identical to the same run without the flag, while this port
        # refused to start.
        #
        # `use_well_enough` has exactly ONE reader in the whole classic tree
        # (`table-problems.cc:493`, grep-confirmed), and it sits inside
        # `it_table_problem::run`'s `else if (output_type == id::enum_type)`
        # arm. Enum output is EXCLUDED from this port's scope entirely, not
        # deferred, so that arm raises earlier and for its own accurate
        # reason. The flag is therefore inert here for exactly the tables it
        # is inert for in classic, which is what this file's own gating rule
        # asks: gate a field only where the C++ source shows it is the
        # subsystem's actual master switch on a path this port can reach.
        # `-t/--output-bscore` was gated here until the S13-CLI `-t` slice,
        # because `metapopulation.merge_demes`'s own `keep_bscore` arm
        # (merging.cc:258-296) raised. It no longer does -- that arm is the
        # SAME behavioural-score recompute `--discard-dominated`/
        # `--diversity-pressure`/`--boost` already drove, and the printing
        # side (`ostream_scored_combo_tree`'s `output_bscore` branch,
        # types.cc:175 / types.py) was already ported and already wired from
        # this field, so nothing beyond deleting the two gates was needed.
        # Measured byte-identical to classic afterwards across boolean,
        # contin, `pa`, `mux` and `sr` problems and four seeds.
        # DIVERGENCE(C++ moses/moses/moses/types.cc:175): classic's own
        # --output-deme-id succeeds and prints an extra " demeID: N" per result
        # line, rc 0 (measured directly: $CLASSIC/build/moses/moses/main/moses -i
        # tests/fixtures/table_data/dataset.csv -m1000 --output-deme-id=1 ->
        # 10 lines, exit 0 -- corrected from an earlier version of this marker
        # that cited tests/moses/dataset.csv, a path under $CLASSIC but not in
        # this port's own repo, and an unmeasured "9 lines").
        # `--output-deme-id` was gated here until 2026-09-10, accurately:
        # `scored_combo_tree` carried no `demeID_t`/`get_demeID()`. It does
        # now (types.py), `ostream_scored_combo_tree`'s arm prints it, and
        # the two rebuild sites (merging.cc:279-280, domination.cc:304)
        # carry the id across, so the flag reaches a real value.

    # ------------------------------------------------------------ helpers

    def _resolve_ignore_ops(self, args: argparse.Namespace) -> set:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Inline body at :2118-2151. Builds the `ignore_ops` vertex set from
        `--include-only-operator`, `--ignore-operator` and `--linear-regression`, in that order,
        exactly matching C++'s own statement order and
        `not_recognized_combo_operator`'s `ValueError` equivalent.
        """
        self.include_only_ops_str = list(args.include_only_ops_str)
        self.ignore_ops_str = list(args.ignore_ops_str)

        ignore_ops: set = set()
        if self.include_only_ops_str:
            ignore_operators = False
            for s in self.include_only_ops_str:
                v = _builtin_str_to_vertex(s)
                if v is None:
                    raise ValueError(f"{s} is not recognized as combo operator.")
                if not ignore_operators:
                    ignore_ops = set(_COMBO_OPERATORS.values())
                    ignore_operators = True
                ignore_ops.discard(v)

        for s in self.ignore_ops_str:
            v = _builtin_str_to_vertex(s)
            if v is None:
                raise ValueError(f"{s} is not recognized as combo operator.")
            ignore_ops.add(v)

        if args.linear_regression:
            ignore_ops.update((builtin.div, builtin.exp, builtin.log, builtin.sin))

        self.ignore_ops = ignore_ops
        return ignore_ops

    def _parse_enforce_features(self) -> None:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Cites the enclosing member (the breadcrumb convention for code
        that is not a symbol of its own): the real code is the "Parse
        enforce features" block inlined in `problem_params::parse_options`
        at problem-params.cc:2231-2259.

        Line format is "<feature name>[ <weight>]": the FIRST space splits
        name from weight, `stof` parses the rest, and a line with no space
        gets weight `1`. Empty lines are skipped. Faithful details that
        look like sloppiness and are not:

        * `stof` (cc:2248) parses a LEADING-whitespace-tolerant prefix and
          ignores trailing junk, so "x1   0.25 rubbish" is 0.25. Python's
          `float()` accepts the leading whitespace and rejects the
          trailing junk, so the substring is `.split()`-ed first and only
          its first token converted -- reproducing `strtof`'s prefix
          parse without importing C's tolerance for the rest of the line.
        * A duplicate feature name overwrites (`map::operator[]`), it does
          not accumulate.
        * The weight is a C++ `float` (feature_selector.h:166,
          `std::map<std::string,float>`), so `_to_score_t` narrows it
          (`score_t` is the same binary32); the
          value round-trips through `biased_randbool(float)`
          ($COGUTIL/opencog/util/random.h:127) at sampling time.
        * C++'s `while (in) { getline(in, line); ... }` reads ONE PAST the
          last line: the final `getline` fails, leaves `line` empty, and
          the empty-line `continue` swallows it. Iterating the file's real
          lines is equivalent, since the extra iteration contributes
          nothing.
        * `OC_ASSERT(in.is_open(), "Could not open %s", ...)` throws
          `AssertionException`; a missing file raises here too, as the
          `OSError` Python's own `open` produces.

        Called unconditionally; returns immediately when
        `--fs-enforce-features-filename` was not given, matching
        `if (!fs_enforce_features_filename.empty())` (cc:2232).
        """
        if not self.fs_enforce_features_filename:
            return
        with open(self.fs_enforce_features_filename) as handle:
            for line in handle:
                # `getline` strips the newline; Python's iteration does not.
                line = line.rstrip("\n")
                if not line:
                    continue
                whitespace_pos = line.find(" ")
                if whitespace_pos != -1:
                    feature_name = line[:whitespace_pos]
                    weight = _to_score_t(float(line[whitespace_pos:].split()[0]))
                    self.festor_params.enforce_features[feature_name] = weight
                else:
                    self.festor_params.enforce_features[line] = 1.0

        # C++: problem-params.cc:2255-2258. `std::map` iterates in
        # ascending key order; a Python `dict` iterates in insertion
        # order, so this sorts to keep the RECORD order identical (the
        # map itself is consumed by `feature_selector.
        # sample_enforced_features`, whose own iteration-order divergence
        # is documented there).
        for name in sorted(self.festor_params.enforce_features):
            logger().debug(
                "Enforce feature %s with probability %s",
                name,
                ostream_double(self.festor_params.enforce_features[name]),
            )

    def _resolve_jobs(self, args: argparse.Namespace) -> tuple[dict, bool]:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Inline body at :2157-2172. `jobs_t{{localhost, 1}}` then one entry per `--jobs`/`-j` token,
        `HOST:N` selecting a remote host (and turning `local` off) or a
        bare `N` overwriting the localhost count. Gated in phase 2 (module
        docstring) if the result is non-local -- distributed dispatch is
        out of scope.
        """
        self.jobs_str = list(args.jobs_str)
        jobs: dict[str, int] = {_LOCALHOST: 1}
        local = True
        for js in self.jobs_str:
            pos = js.find(_JOB_SEPARATOR)
            if pos != -1:
                jobs[js[pos + 1 :]] = int(js[:pos])
                local = False
            else:
                jobs[_LOCALHOST] = int(js)
        return jobs, local

    def _wire_diversity(self, args: argparse.Namespace, meta_params: metapop_parameters) -> None:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Inline body at :2293-2326. Populates `meta_params.diversity` UNCONDITIONALLY -- including
        `autoscale`, the field CARRY-FORWARD's own DIVERGENCE note on
        `diversity_parameters` names as genuinely uninitialized in C++
        until the CLI sets it -- and validates `--diversity-dst`/
        `--diversity-dst2dp` unconditionally too, matching C++'s own
        `not_recognized_dst`/`not_recognized_dst2dp` calls (which run
        regardless of `--diversity-pressure`). The P2 gate itself
        (`enabled()`) is checked in phase 2, not here -- see the module
        docstring's design-rule section for why population must complete
        first.
        """
        self.diversity_pressure = _to_score_t(args.diversity_pressure)
        self.diversity_autoscale = args.diversity_autoscale
        self.diversity_exponent = _to_score_t(args.diversity_exponent)
        self.diversity_normalize = args.diversity_normalize
        self.diversity_dst = args.diversity_dst
        self.diversity_p_norm = _to_score_t(args.diversity_p_norm)
        self.diversity_dst2dp = args.diversity_dst2dp

        meta_params.diversity.pressure = self.diversity_pressure
        meta_params.diversity.autoscale = self.diversity_autoscale
        meta_params.diversity.exponent = self.diversity_exponent
        meta_params.diversity.normalize = self.diversity_normalize

        if self.diversity_dst == _P_NORM:
            de = diversity_parameters.dst_enum_t.p_norm
        elif self.diversity_dst == _TANIMOTO:
            de = diversity_parameters.dst_enum_t.tanimoto
        elif self.diversity_dst == _ANGULAR:
            de = diversity_parameters.dst_enum_t.angular
        else:
            raise ValueError(
                f"{self.diversity_dst} is not recognized. Valid distances are "
                f"{_P_NORM}, {_TANIMOTO} and {_ANGULAR}"
            )
        meta_params.diversity.set_dst(de, self.diversity_p_norm)

        if self.diversity_dst2dp == _AUTO:
            d2de = (
                diversity_parameters.dst2dp_enum_t.inverse
                if self.diversity_dst == _P_NORM
                else diversity_parameters.dst2dp_enum_t.complement
            )
        elif self.diversity_dst2dp == _INVERSE:
            d2de = diversity_parameters.dst2dp_enum_t.inverse
        elif self.diversity_dst2dp == _COMPLEMENT:
            d2de = diversity_parameters.dst2dp_enum_t.complement
        elif self.diversity_dst2dp == _POWER:
            d2de = diversity_parameters.dst2dp_enum_t.pthpower
        else:
            raise ValueError(
                f"{self.diversity_dst2dp} is not recognized. Valid distances to "
                f"penalty are {_AUTO}, {_INVERSE}, {_COMPLEMENT}, {_POWER}"
            )
        meta_params.diversity.set_dst2dp(d2de)

    def _wire_subsample(self, args: argparse.Namespace) -> None:
        """C++: moses/moses/main/problem-params.cc:2058 problem_params::parse_options

        Inline body at :2339-2355. Populates every one of the 15 `filter_params` fields
        UNCONDITIONALLY, including the ten `subsample_deme_filter_
        parameters` fields CARRY-FORWARD's own DIVERGENCE note documents as
        genuinely uninitialized in C++ until the CLI sets them. The P2 gate
        (`n_subsample_demes|n_subsample_fitnesses > 1`) is checked in
        phase 2 -- see `_wire_diversity`'s docstring for the same ordering
        argument.

        The 8 `ss_*` threshold/weight fields are C++ `float` (problem-
        params.h:193-199,205 -- NOT `double`, unlike every other CLI
        double), so each is narrowed via `_to_score_t` the same way a real
        `score_t`/`float` binding is narrowed elsewhere in this port
        (CARRY-FORWARD.md 4.2); the `unsigned`/bool fields are not.
        """
        self.ss_n_top_candidates = args.ss_n_top_candidates
        self.ss_n_tuples = args.ss_n_tuples
        self.ss_std_dev_threshold = _to_score_t(args.ss_std_dev_threshold)
        self.ss_tanimoto_mean_threshold = _to_score_t(args.ss_tanimoto_mean_threshold)
        self.ss_tanimoto_geo_mean_threshold = _to_score_t(args.ss_tanimoto_geo_mean_threshold)
        self.ss_tanimoto_max_threshold = _to_score_t(args.ss_tanimoto_max_threshold)
        self.ss_n_best_bfdemes = args.ss_n_best_bfdemes
        self.ss_tanimoto_mean_weight = _to_score_t(args.ss_tanimoto_mean_weight)
        self.ss_tanimoto_geo_mean_weight = _to_score_t(args.ss_tanimoto_geo_mean_weight)
        self.ss_tanimoto_max_weight = _to_score_t(args.ss_tanimoto_max_weight)
        self.ss_low_dev_pressure = _to_score_t(args.ss_low_dev_pressure)
        self.ss_by_time = args.ss_by_time
        self.ss_contiguous_time = args.ss_contiguous_time

        filter_params = subsample_deme_filter_parameters(
            n_subsample_demes=self.ss_n_subsample_demes,
            n_top_candidates=self.ss_n_top_candidates,
        )
        filter_params.n_tuples = self.ss_n_tuples
        filter_params.by_time = self.ss_by_time
        filter_params.contiguous_time = self.ss_contiguous_time
        filter_params.std_dev_threshold = self.ss_std_dev_threshold
        filter_params.tanimoto_mean_threshold = self.ss_tanimoto_mean_threshold
        filter_params.tanimoto_geo_mean_threshold = self.ss_tanimoto_geo_mean_threshold
        filter_params.tanimoto_max_threshold = self.ss_tanimoto_max_threshold
        filter_params.n_best_bfdemes = self.ss_n_best_bfdemes
        filter_params.tanimoto_mean_weight = self.ss_tanimoto_mean_weight
        filter_params.tanimoto_geo_mean_weight = self.ss_tanimoto_geo_mean_weight
        filter_params.tanimoto_max_weight = self.ss_tanimoto_max_weight
        filter_params.n_subsample_fitnesses = self.ss_n_subsample_fitnesses
        filter_params.low_dev_pressure = self.ss_low_dev_pressure
        self.filter_params = filter_params


def parse_args(argv: list[str]) -> problem_params:
    """Not itself a ported C++ symbol -- the `problem_params()`;
    `add_options`; `parse_args`/`parse_options` three-step C++ callers
    perform inline (`moses_exec.cc`'s own body), collapsed into one
    convenience entry point for this port's callers and tests.
    """
    pp = problem_params()
    parser = argparse.ArgumentParser(add_help=True)
    pp.add_options(parser)
    ns = parser.parse_args(argv)
    pp.parse_options(ns)
    return pp
