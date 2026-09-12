"""Value-aware specification of classic MOSES's command line, for `flag_sweep.py`.

WHY THIS FILE EXISTS -- the hole it replaces
============================================
The first differential flag sweep this project ran took the list of long options
out of ``moses --help`` and, for each one, tried a fixed ladder of candidate
values -- ``1``, ``2``, ``0.5``, ``3``, ``0``, ``tanimoto``, ``hc`` -- keeping the
first value classic exited 0 on. That design has two holes, and both of them hid
real divergences:

1. **Enumerated options have no numeric value that is legal.** For
   ``--output-format``, ``--time-bscore-granularity``, ``--diversity-dst2dp``,
   ``--score-weight``, ``--timestamp-feature`` and
   ``--fs-enforce-features-filename`` the ladder never produced an accepted
   value, so the sweep printed ``classic-rejects-all-candidates`` -- a *silent*
   bucket that looked like a skip rather than a miss -- and those flags were
   never actually compared.  ``--output-format=scheme`` was a genuine divergence
   sitting inside that bucket.

2. **A value classic merely *parses* is not a value classic *exercises*.** The
   ladder's first hit for ``--fs-algo``/``--fs-scorer``/``--fs-focus``/``--fs-seed``
   was the literal ``1``.  Classic accepts it (rc 0) because those strings are
   only interpreted when ``--enable-fs=1``, which the sweep never passed.  All
   four were reported ``IDENTICAL`` while testing nothing at all.

There was also a third, quieter hole: the option list had been filtered to long
options with no short form, so the 49 options that do have one (``-i``, ``-H``,
``-a``, ``-B``, ``-E``, ``-W``, ...) were never in the sweep to begin with.

So this file states, per option, the values to try -- derived from ``moses --help``
and from ``moses/moses/main/problem-params.cc`` in the C++ checkout, not guessed --
plus the context (``needs``) that makes the option actually reachable, and the
dataset it needs.  ``flag_sweep.py`` treats "classic rejected every value I have
for this flag" as a LOUD, exit-non-zero condition, because that is exactly the
signal the old sweep swallowed.

Conventions
-----------
``values``  every value to try, legal and deliberately illegal.  Illegal ones are
            kept on purpose: classic's reaction to a bad enum (``OC_ASSERT`` ->
            ``SIGABRT``, or a ``std::logic_error`` -> rc 1) is itself behaviour the
            port has to match.
``needs``   extra argv tokens that make the option do something.  Without
            ``--enable-fs=1`` every ``--fs-*`` option is dead text.
``dataset`` ``bool``   res,arg1,arg2            -- a 2-input boolean table
            ``wide``   a1..a6,res               -- 6 inputs, so feature selection bites
            ``ts``     i1,i2,when,w,y           -- adds a timestamp and a weight column
            ``contin`` x1,x2,y                  -- real-valued target
``skip``    a reason string; the option is listed (so the census stays complete)
            but not run, e.g. because it forks ssh or blocks on a socket.
``known_dead``
            a reason string for "classic exits non-zero at every value".  Without
            it, that state raises the loud banner; with it, the option is listed
            quietly.  Never add one without a verified reproducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

BOOL = ("0", "1")


@dataclass(frozen=True)
class FlagSpec:
    """One option and everything needed to sweep it."""

    flag: str
    kind: str  # enum | bool | int | float | string | file | special | novalue
    default: str
    values: tuple[str, ...]
    short: str = ""
    needs: tuple[str, ...] = ()
    dataset: str = "bool"
    note: str = ""
    skip: str = ""
    # Why classic accepts no value at all.  An option with this set is
    # listed in the report but does not fail the sweep; an option WITHOUT it
    # that classic rejects everywhere raises the loud banner.  The point of
    # the split is that the banner must stay rare enough to still be read.
    known_dead: str = ""
    # Values classic is expected to reject (documentation only; the sweep
    # measures, it does not assume).
    expect_reject: tuple[str, ...] = field(default_factory=tuple)


# Contexts that switch a gated subsystem on.  Kept as names so the table below
# reads as intent rather than as argv soup.
FS = ("--enable-fs=1", "--fs-target-size=3")
BOOST = ("--boost=1",)
BOOST_PRE = ("--boost=1", "-Hpre", "-q0.1", "-w0.9")
PRE = ("-Hpre", "-q0.1", "-w0.9")
TS = ("--timestamp-feature=when",)
TSB = ("--timestamp-feature=when", "--time-bscore=1")
SS = ("--ss-n-subsample-demes=2",)
DIV = ("--diversity-pressure=1",)
PS = ("-aps",)

# Classic segfaults the moment --fs-algo=hc actually runs (reproduced
# standalone on the 6-input table).  Every --fs-hc-* option therefore has no
# value at which classic exits 0; that is acknowledged rather than re-reported
# as a fresh discovery on every run.
HC_SEGV = (
    "classic SEGVs whenever --fs-algo=hc actually runs (reproduced standalone); "
    "every value of this option inherits that crash"
)


SPECS: tuple[FlagSpec, ...] = (
    # ---- informational -------------------------------------------------
    FlagSpec(
        "--help",
        "novalue",
        "",
        ("",),
        short="-h",
        known_dead="classic prints the help and returns 1 (problem.cc:50); "
        "there is no value at which it exits 0",
    ),
    FlagSpec("--version", "novalue", "", ("",)),
    # ---- interesting-patterns weights (only live under -Hip) -----------
    FlagSpec(
        "--ip_kld_weight",
        "float",
        "1",
        ("0", "1", "2", "-1"),
        short="-K",
        needs=("-Hip",),
        dataset="ip",
    ),
    FlagSpec(
        "--ip_skewness_weight",
        "float",
        "1",
        ("0", "1", "2", "-1"),
        short="-J",
        needs=("-Hip",),
        dataset="ip",
    ),
    FlagSpec(
        "--ip_stdU_weight",
        "float",
        "1",
        ("0", "1", "2", "-1"),
        short="-U",
        needs=("-Hip",),
        dataset="ip",
    ),
    FlagSpec(
        "--ip_skew_U_weight",
        "float",
        "1",
        ("0", "1", "2", "-1"),
        short="-X",
        needs=("-Hip",),
        dataset="ip",
    ),
    # ---- table / feature plumbing --------------------------------------
    FlagSpec(
        "--input-file",
        "file",
        "",
        ("@DATA@", "@MISSING@"),
        short="-i",
        note="repeatable; @DATA@ passes the dataset a second time",
    ),
    FlagSpec("--target-feature", "string", "", ("@TARGET@", "nosuchcolumn"), short="-u"),
    FlagSpec("--score-weight", "string", "", ("w", "nosuchcolumn"), dataset="ts"),
    FlagSpec("--timestamp-feature", "string", "", ("when", "nosuchcolumn"), dataset="ts"),
    FlagSpec("--ignore-feature", "string", "", ("arg1", "nosuchcolumn"), short="-Y"),
    FlagSpec(
        "--combo-program",
        "string",
        "",
        ("and($1 $2)", "or($1 $2)", "bogus("),
        short="-y",
        needs=("-Hcp", "-k2"),
    ),
    FlagSpec("--problem-size", "int", "5", ("1", "2", "3", "5"), short="-k", needs=("-Hpa",)),
    FlagSpec(
        "--jobs",
        "int",
        "",
        ("1", "2"),
        short="-j",
        skip="spawns worker threads / ssh; not a deterministic differential",
    ),
    FlagSpec("--min-pool", "int", "50", ("0", "1", "50", "100")),
    FlagSpec("--exemplar", "string", "", ("and($1 $2)", "true", "$1", "bogus("), short="-e"),
    FlagSpec(
        "--problem",
        "enum",
        "it",
        (
            "it",
            "pre",
            "prerec",
            "recall",
            "bep",
            "f_one",
            "select",
            "pre-conj",
            "cluster",
            "ip",
            "ann-it",
            "cp",
            "ann-cp",
            "pa",
            "dj",
            "mux",
            "maj",
            "sr",
            "bogus",
        ),
        short="-H",
        note="registry lookup; --help omits pre-conj, cluster and ann-cp, all three valid",
        expect_reject=("bogus",),
    ),
    FlagSpec("--nsamples", "int", "-1", ("-1", "0", "1", "4", "1000")),
    FlagSpec(
        "--balance",
        "bool",
        "0",
        ("0", "1", "true", "false", "yes", "no", "on", "off", "2", "bogus"),
        note="stand-in for the whole po::value<bool> grammar; every other bool "
        "option shares this parser, and an out-of-grammar value aborts at "
        "problem.cc:64",
    ),
    FlagSpec(
        "--opt-algo",
        "enum",
        "hc",
        ("hc", "un", "sa", "ps", "bogus"),
        short="-a",
        expect_reject=("bogus",),
    ),
    FlagSpec("--max-score", "float", "3.40282347e+38", ("-1", "0", "1", "100"), short="-A"),
    FlagSpec("--max-evals", "int", "10000", ("0", "1", "50", "200"), short="-m"),
    FlagSpec(
        "--max-time",
        "int",
        "2147483647",
        ("0", "2147483647"),
        note="wall-clock gated; 0 stops immediately, large is the default",
    ),
    FlagSpec("--cache-size", "int", "3000", ("0", "1", "3000"), short="-s"),
    FlagSpec(
        "--ignore-operator",
        "enum",
        "",
        ("div", "sin", "exp", "log", "plus", "times", "impulse", "cond", "and", "$3", "bogus"),
        short="-n",
        dataset="contin",
        expect_reject=("bogus",),
    ),
    FlagSpec("--linear-regression", "bool", "0", BOOL, dataset="contin"),
    FlagSpec(
        "--logical-perm-ratio",
        "float",
        "0",
        ("-2", "-1", "-0.5", "0", "0.5", "1", "2"),
        note="help says 'valid values in the range -1.0 to +1.0'; -2 and 2 probe outside it",
    ),
    FlagSpec("--random-seed", "int", "1", ("0", "1", "2", "-1"), short="-r"),
    FlagSpec(
        "--complexity-temperature",
        "float",
        "6",
        ("0", "0.5", "6", "100", "-1"),
        short="-v",
        note="help says 'strictly positive'; 0 and -1 probe the boundary",
    ),
    FlagSpec("--complexity-ratio", "float", "3.5", ("0", "1", "3.5", "10", "-1"), short="-z"),
    FlagSpec("--cap-coef", "float", "50", ("0", "1", "50", "100")),
    # ---- hill climbing --------------------------------------------------
    FlagSpec("--hc-max-nn-evals", "int", "20000", ("0", "1", "100", "20000")),
    FlagSpec("--hc-fraction-of-nn", "float", "2", ("0", "0.5", "1", "2", "3")),
    FlagSpec("--hc-crossover", "bool", "1", BOOL, short="-Z"),
    FlagSpec("--hc-crossover-pop-size", "int", "120", ("0", "1", "120")),
    FlagSpec("--hc-crossover-min-neighbors", "int", "400", ("0", "1", "400")),
    FlagSpec("--hc-resize-to-fit-ram", "bool", "0", BOOL),
    FlagSpec("--hc-allow-resize-deme", "bool", "0", BOOL),
    FlagSpec("--hc-single-step", "bool", "0", BOOL, short="-L"),
    FlagSpec("--hc-widen-search", "bool", "0", BOOL, short="-T"),
    FlagSpec("--ps-max-particles", "int", "50", ("1", "10", "50"), needs=PS),
    # ---- representation -------------------------------------------------
    FlagSpec("--contin-depth", "int", "5", ("1", "2", "5", "8"), dataset="contin"),
    FlagSpec(
        "--reduct-knob-building-effort",
        "int",
        "2",
        ("-1", "0", "1", "2", "3", "4"),
        short="-B",
        note="logical_reduction.cc:142 switch default -> OC_ASSERT; -1 and 4 probe outside 0..3",
    ),
    FlagSpec("--max-dist", "int", "4", ("1", "2", "4", "8"), short="-D"),
    FlagSpec("--reduce-all", "bool", "1", BOOL, short="-d"),
    FlagSpec(
        "--reduct-candidate-effort",
        "int",
        "2",
        ("-1", "0", "1", "2", "3", "4"),
        short="-E",
        note="logical_reduction.cc:142 switch default -> OC_ASSERT; -1 and 4 probe outside 0..3",
    ),
    FlagSpec("--max-gens", "int", "-1", ("-1", "0", "1", "2"), short="-g"),
    FlagSpec(
        "--include-only-operator",
        "enum",
        "",
        ("div", "sin", "exp", "log", "plus", "times", "impulse", "cond", "and", "$3", "bogus"),
        short="-N",
        dataset="contin",
        expect_reject=("bogus",),
    ),
    FlagSpec("--pop-size-ratio", "float", "20", ("1", "20", "100")),
    FlagSpec("--noise", "float", "-1", ("-1", "0", "0.5", "1")),
    # ---- boosting -------------------------------------------------------
    FlagSpec("--boost", "bool", "0", BOOL),
    FlagSpec("--boost-promote", "int", "1", ("1", "2", "5"), needs=BOOST_PRE),
    FlagSpec("--boost-exact", "bool", "1", BOOL, needs=BOOST_PRE),
    FlagSpec("--boost-expalpha", "float", "2", ("1", "2", "3"), needs=BOOST_PRE),
    FlagSpec("--boost-bias", "float", "1", ("0", "0.5", "1"), needs=BOOST_PRE),
    # ---- metapopulation -------------------------------------------------
    FlagSpec("--discard-dominated", "bool", "0", BOOL),
    FlagSpec("--well-enough", "bool", "0", BOOL),
    FlagSpec("--revisit", "int", "0", ("0", "1", "2")),
    FlagSpec("--max-candidates-per-deme", "int", "-1", ("-1", "0", "1", "10"), short="-M"),
    FlagSpec("--weighted-accuracy", "bool", "0", BOOL, short="-G"),
    FlagSpec("--gen-best-tree", "bool", "0", BOOL),
    FlagSpec("--it-abs-err", "bool", "0", BOOL, dataset="contin"),
    FlagSpec("--pre-positive", "bool", "1", BOOL, needs=PRE),
    FlagSpec(
        "--alpha",
        "float",
        "1",
        ("-1", "0", "1", "2"),
        short="-Q",
        needs=PRE,
        note="OC_ASSERT(0.0 <= hardness) at discriminating_bscore.cc:222 / precision_bscore.cc:160",
    ),
    FlagSpec(
        "--min-rand-input",
        "float",
        "0",
        ("-1", "0", "0.1", "0.5", "0.95"),
        short="-q",
        needs=PRE,
        note="precision_bscore.cc:162 asserts 0 < min && min <= max when alpha > 0; "
        "the -q0.1 in PRE is dropped when -q is itself swept",
    ),
    FlagSpec(
        "--max-rand-input",
        "float",
        "1",
        ("0", "0.05", "0.9", "1", "2"),
        short="-w",
        needs=PRE,
        note="same assert as -q, from the other side",
    ),
    FlagSpec(
        "--discretize-threshold",
        "float",
        "",
        ("0", "0.5", "1", "2"),
        short="-R",
        dataset="contin",
        note="repeatable: n uses produce n+1 classes",
    ),
    # ---- diversity ------------------------------------------------------
    FlagSpec("--diversity-pressure", "float", "0", ("0", "0.5", "1")),
    FlagSpec("--diversity-autoscale", "bool", "0", BOOL, needs=DIV),
    FlagSpec("--diversity-exponent", "float", "-1", ("-1", "0", "1", "2"), needs=DIV),
    FlagSpec("--diversity-normalize", "bool", "1", BOOL, needs=DIV),
    FlagSpec(
        "--diversity-dst",
        "enum",
        "p_norm",
        ("p_norm", "tanimoto", "angular", "bogus"),
        needs=DIV,
        expect_reject=("bogus",),
    ),
    FlagSpec("--diversity-p-norm", "float", "2", ("0", "0.5", "1", "2", "3"), needs=DIV),
    FlagSpec(
        "--diversity-dst2dp",
        "enum",
        "auto",
        ("auto", "inverse", "complement", "power", "bogus"),
        needs=DIV,
        note="the old ladder never produced a legal value for this one",
        expect_reject=("bogus",),
    ),
    # ---- time / timestamp ----------------------------------------------
    FlagSpec("--time-bscore", "bool", "0", BOOL, needs=TS, dataset="ts"),
    FlagSpec(
        "--time-bscore-granularity",
        "enum",
        "day",
        ("day", "month", "year", "bogus"),
        needs=TSB,
        dataset="ts",
        note="problem-params.cc:2329 accepts day|month and throws otherwise",
        expect_reject=("year", "bogus"),
    ),
    FlagSpec("--time-dispersion-pressure", "float", "0", ("0", "0.5", "1"), needs=TS, dataset="ts"),
    FlagSpec("--time-dispersion-exponent", "float", "1", ("0", "1", "2"), needs=TS, dataset="ts"),
    # ---- output ---------------------------------------------------------
    FlagSpec("--result-count", "int", "10", ("0", "1", "10", "100"), short="-c"),
    FlagSpec("--output-score", "bool", "1", BOOL, short="-S"),
    FlagSpec("--output-cscore", "bool", "0", BOOL, short="-x"),
    FlagSpec("--output-bscore", "bool", "0", BOOL, short="-t"),
    FlagSpec("--output-only-best", "bool", "0", BOOL, short="-C"),
    FlagSpec("--output-eval-number", "bool", "0", BOOL, short="-V"),
    FlagSpec("--output-with-labels", "bool", "0", BOOL, short="-W"),
    FlagSpec("--output-deme-id", "bool", "0", BOOL),
    FlagSpec(
        "--output-format",
        "enum",
        "combo",
        ("combo", "python", "python3", "scheme", "bogus"),
        note="iostream_combo.cc:252 parse_output_format accepts combo|python|scheme only; "
        "python3 is in the enum but unreachable from the CLI",
        expect_reject=("python3", "bogus"),
    ),
    FlagSpec("--output-file", "file", "", ("@OUTFILE@",), short="-o"),
    FlagSpec(
        "--log-level",
        "enum",
        "INFO",
        ("NONE", "ERROR", "WARN", "INFO", "DEBUG", "FINE", "fine", "bogus"),
        short="-l",
        expect_reject=("bogus",),
    ),
    FlagSpec(
        "--log-file",
        "file",
        "moses.log",
        ("@LOGFILE@", "/dev/null"),
        short="-f",
        note="/dev/null is a real divergence probe: classic ignores the failed unlink",
    ),
    FlagSpec("--log-file-dep-opt", "novalue", "", ("",), short="-F"),
    # ---- feature selection ----------------------------------------------
    FlagSpec("--enable-fs", "bool", "0", BOOL),
    FlagSpec("--fs-target-size", "int", "20", ("0", "1", "2", "20"), needs=FS),
    FlagSpec("--fs-exp-distrib", "bool", "0", BOOL, needs=FS),
    FlagSpec(
        "--fs-focus",
        "enum",
        "incorrect",
        ("all", "active", "incorrect", "ai", "bogus"),
        needs=FS,
        note="problem-params.cc:2196 -- the old ladder scored this IDENTICAL at value '1'",
        expect_reject=("bogus",),
    ),
    FlagSpec(
        "--fs-seed",
        "enum",
        "add",
        ("none", "add", "init", "xmplr", "bogus"),
        needs=FS,
        note="problem-params.cc:2210",
        expect_reject=("bogus",),
    ),
    FlagSpec("--fs-prune-exemplar", "bool", "0", BOOL, needs=FS),
    FlagSpec("--fs-subsampling-ratio", "float", "1", ("0", "0.5", "1"), needs=FS),
    FlagSpec(
        "--fs-subsampling-by-time",
        "bool",
        "0",
        BOOL,
        needs=("--enable-fs=1", "--fs-target-size=2", "--timestamp-feature=when"),
        dataset="ts_wide",
        note="MI aborts on a date-string column (table.h:1301), so the timestamp "
        "must be consumed by --timestamp-feature and the rest kept boolean",
    ),
    FlagSpec("--fs-demes", "int", "1", ("0", "1", "2"), needs=FS),
    FlagSpec(
        "--fs-algo",
        "enum",
        "simple",
        ("simple", "inc", "smd", "random", "hc", "bogus"),
        needs=FS,
        expect_reject=("bogus",),
    ),
    FlagSpec(
        "--fs-scorer", "enum", "mi", ("mi", "pre", "bogus"), needs=FS, expect_reject=("bogus",)
    ),
    FlagSpec("--fs-threshold", "float", "0", ("0", "0.5", "1"), needs=FS),
    FlagSpec("--fs-enforce-features-filename", "file", "", ("@FEATFILE@", "@MISSING@"), needs=FS),
    FlagSpec("--fs-diversity-pressure", "float", "0", ("0", "0.5", "1"), needs=FS),
    FlagSpec("--fs-diversity-cap", "int", "100", ("0", "1", "100"), needs=FS),
    FlagSpec("--fs-diversity-interaction", "int", "-1", ("-1", "0", "1"), needs=FS),
    FlagSpec("--fs-diversity-jaccard", "bool", "1", BOOL, needs=FS),
    FlagSpec(
        "--fs-inc-redundant-intensity",
        "float",
        "-1",
        ("-1", "0", "0.5"),
        needs=FS + ("--fs-algo=inc",),
    ),
    FlagSpec(
        "--fs-inc-target-size-epsilon",
        "float",
        "1e-06",
        ("1e-06", "0.1"),
        needs=FS + ("--fs-algo=inc",),
    ),
    FlagSpec(
        "--fs-inc-interaction-terms", "int", "1", ("1", "2", "3"), needs=FS + ("--fs-algo=inc",)
    ),
    FlagSpec("--fs-pre-penalty", "float", "1", ("0", "1", "2"), needs=FS + ("--fs-scorer=pre",)),
    FlagSpec(
        "--fs-pre-min-activation",
        "float",
        "0.5",
        ("0", "0.5", "1"),
        needs=FS + ("--fs-scorer=pre",),
    ),
    FlagSpec(
        "--fs-pre-max-activation", "float", "1", ("0.5", "1"), needs=FS + ("--fs-scorer=pre",)
    ),
    FlagSpec("--fs-pre-positive", "bool", "1", BOOL, needs=FS + ("--fs-scorer=pre",)),
    FlagSpec(
        "--fs-hc-max-score",
        "float",
        "1",
        ("0", "1", "2"),
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-max-evals",
        "int",
        "10000",
        ("1", "100", "10000"),
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-fraction-of-remaining",
        "float",
        "0.5",
        ("0", "0.5", "1"),
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-crossover",
        "bool",
        "0",
        BOOL,
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-crossover-pop-size",
        "int",
        "120",
        ("1", "120"),
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-crossover-min-neighbors",
        "int",
        "400",
        ("1", "400"),
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec(
        "--fs-hc-widen-search",
        "bool",
        "1",
        BOOL,
        needs=FS + ("--fs-algo=hc",),
        known_dead=HC_SEGV,
    ),
    FlagSpec("--fs-mi-penalty", "float", "100", ("0", "1", "100"), needs=FS),
    FlagSpec("--fs-smd-top-size", "int", "10", ("1", "10"), needs=FS + ("--fs-algo=smd",)),
    # ---- subsampling (ss-*) ---------------------------------------------
    FlagSpec("--ss-n-subsample-demes", "int", "0", ("0", "1", "2")),
    FlagSpec("--ss-n-top-candidates", "int", "1", ("1", "2"), needs=SS),
    FlagSpec("--ss-n-tuples", "int", "4294967295", ("1", "2", "4294967295"), needs=SS),
    FlagSpec("--ss-std-dev-threshold", "float", "3.40282347e+38", ("0", "1", "1e30"), needs=SS),
    FlagSpec("--ss-tanimoto-mean-threshold", "float", "1", ("0", "0.5", "1"), needs=SS),
    FlagSpec("--ss-tanimoto-geometric-mean-threshold", "float", "1", ("0", "0.5", "1"), needs=SS),
    FlagSpec("--ss-tanimoto-max-threshold", "float", "1", ("0", "0.5", "1"), needs=SS),
    FlagSpec("--ss-n-best-bfdemes", "int", "0", ("0", "1"), needs=SS),
    FlagSpec("--ss-tanimoto-mean-weight", "float", "0", ("0", "1"), needs=SS),
    FlagSpec("--ss-tanimoto-geometric-mean-weight", "float", "0", ("0", "1"), needs=SS),
    FlagSpec("--ss-tanimoto-max-weight", "float", "0", ("0", "1"), needs=SS),
    FlagSpec("--ss-n-subsample-fitnesses", "int", "0", ("0", "1", "2")),
    FlagSpec(
        "--ss-low-dev-pressure",
        "float",
        "1",
        ("0", "1", "2"),
        needs=("--ss-n-subsample-fitnesses=2",),
    ),
    FlagSpec("--ss-by-time", "bool", "0", BOOL, needs=SS + TS, dataset="ts"),
    FlagSpec(
        "--ss-contiguous-time", "bool", "1", BOOL, needs=SS + TS + ("--ss-by-time=1",), dataset="ts"
    ),
)


# ---------------------------------------------------------------------------
# Combination sweeps: pairs/tuples the single-flag sweep cannot reach, because
# each option only does something in the presence of the other.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComboSpec:
    name: str
    axes: tuple[tuple[str, tuple[str, ...]], ...]
    needs: tuple[str, ...] = ()
    dataset: str = "bool"


COMBOS: tuple[ComboSpec, ...] = (
    ComboSpec(
        "fs-algo-x-scorer-lateness",
        (
            ("--fs-algo", ("simple", "inc", "random", "smd", "hc")),
            ("--fs-scorer", ("mi", "pre", "bogus")),
        ),
        needs=FS,
        dataset="wide",
    ),
    ComboSpec(
        "fs-matrix",
        (
            ("--fs-algo", ("simple", "inc", "smd", "random", "hc")),
            ("--fs-scorer", ("mi", "pre")),
            ("--fs-focus", ("all", "active", "incorrect", "ai")),
            ("--fs-seed", ("none", "add", "init", "xmplr")),
        ),
        needs=FS,
    ),
    ComboSpec(
        "outfmt-x-boost",
        (
            ("--output-format", ("combo", "python", "python3", "scheme")),
            ("--boost", ("0", "1")),
        ),
    ),
    ComboSpec(
        "outfmt-x-labels",
        (
            ("--output-format", ("combo", "python", "python3", "scheme")),
            ("--output-with-labels", ("0", "1")),
        ),
    ),
    ComboSpec(
        "outfmt-x-bscore",
        (
            ("--output-format", ("combo", "python", "python3", "scheme")),
            ("--output-bscore", ("0", "1")),
        ),
    ),
    ComboSpec(
        "granularity-x-dispersion",
        (
            ("--time-bscore-granularity", ("day", "month")),
            ("--time-dispersion-pressure", ("0", "0.5", "1")),
            ("--time-dispersion-exponent", ("1", "2")),
        ),
        needs=TSB,
        dataset="ts",
    ),
    ComboSpec(
        "diversity-dst-x-dst2dp",
        (
            ("--diversity-dst", ("p_norm", "tanimoto", "angular")),
            ("--diversity-dst2dp", ("auto", "inverse", "complement", "power")),
        ),
        needs=DIV,
    ),
)


# ---------------------------------------------------------------------------
# Feature selection is a NO-OP when the target size is >= the arity: classic
# selects every feature and every --fs-focus / --fs-seed / --fs-algo value
# produces byte-identical output.  On the 2-input table the first sweep used,
# that is exactly what happened -- 20 `IDENTICAL` verdicts from a sweep that
# could not have found anything.  So every --fs-* case is re-pointed at the
# 6-input `wide` table, and FS carries --fs-target-size=3 to force a choice.
# Verified: `--fs-focus=all` and `--fs-focus=incorrect` give different output
# on `wide`, and identical output on `bool`.
# ---------------------------------------------------------------------------
SPECS = tuple(
    replace(s, dataset="wide")
    if s.dataset == "bool" and (s.flag == "--enable-fs" or s.flag.startswith("--fs-"))
    else s
    for s in SPECS
)
COMBOS = tuple(replace(c, dataset="wide") if c.name == "fs-matrix" else c for c in COMBOS)
