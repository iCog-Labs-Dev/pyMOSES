"""C++: moses/feature-selection/main/fs-main.cc

The standalone ``feature-selection`` binary's ``main()``: parse argv, set
up the log, seed the RNG, load the table, optionally subsample it, then
dispatch on the inferred OUTPUT type into ``feature_selection()``
(``pymetta.feature_selection.main.feature_selection``).

Structural choices, all following ``pymetta.moses.main.moses_exec``'s
already-established precedent for the analogous C++ ``main()``
(``moses-exec.cc`` / ``moses_exec.cc``) rather than inventing anything:

* ``boost::program_options`` becomes ONE ``argparse.ArgumentParser``.
  ``-h``/``--help`` and malformed-argument handling are left to argparse
  (``SystemExit``), not reproduced from boost's own formatter -- exactly
  the scope note ``moses_exec.py``'s module docstring records.
* ``int main(int, char**)`` becomes ``fs_main(argv)`` where ``argv[0]``
  is the program name, so only ``argv[1:]`` reaches the parser. A bare
  ``str`` is split on whitespace, matching ``moses_exec``'s own third
  overload.
* ``setting_omp(fs_params.jobs)`` (fs-main.cc:420) is not ported: it sets
  OpenMP's thread count, and this port is single-threaded throughout
  (`--jobs` is parsed and stored, and changes nothing here, exactly as it
  changes nothing in a single-threaded classic build). Nothing downstream
  reads ``fs_params.jobs``.

TWO UPSTREAM DEFECTS ARE REPRODUCED HERE, NOT FIXED.

1. **``--subsampling-ratio`` is broken for all but a sliver of its
   range.** fs-main.cc:437 reads

       subsampleTable(fs_params.subsampling_ratio * table.size(), table);

   but the two-argument ``subsampleTable`` overload it resolves to is
   ``subsampleTable(float ratio, Table&)`` (table.cc:892), whose first
   parameter is a RATIO, not a row count -- and which opens with
   ``OC_ASSERT(0.0 <= ratio and ratio <= 1.0)``. So the value that
   reaches the assert is ``ratio * nrows``, and the option only survives
   when ``ratio <= 1/nrows``. Measured against
   ``$CLASSIC/build/moses/feature-selection/main/feature-selection`` on
   the 8-row ``trainingBoolean.dat`` (2026-09-10, each run bounded by
   ``( ulimit -v 2097152; timeout 300 ... )``):

       --subsampling-ratio 0.5   -> abort, "Ratio must be in [0.0, 1.0], but is 4.000000"
       --subsampling-ratio 0.9   -> abort, "... but is 7.200000"
       --subsampling-ratio 0.13  -> abort, "... but is 1.040000"
       --subsampling-ratio 0.125 -> exit 0, all 8 rows kept
       --subsampling-ratio 0.1   -> exit 0, 6 rows kept
       --subsampling-ratio 0.05  -> SIGSEGV (exit 139)

   Ported by transliterating ``subsampleTable`` faithfully -- assert
   included -- and calling it with ``ratio * size`` exactly as
   fs-main.cc does, so the failure arises from the same arithmetic
   rather than from a hand-written gate. **DO NOT FIX.**

2. **The inner ``subsampleTable`` erases at indices drawn from the
   ORIGINAL row count**, never the shrinking one (table.cc:872-889:
   ``unsigned size = max(...)`` is computed once, and every
   ``randGen().randint(size)`` inside the loop uses it). Once a row has
   been erased, a later draw can name a position past the end, and
   ``it.erase(it.begin()+ridx)`` is then undefined behaviour -- that is
   the ``--subsampling-ratio 0.05`` SIGSEGV above. A port cannot
   faithfully reproduce undefined behaviour (docs/CARRY-FORWARD.md
   section 3 settles this for the analogous out-of-range ``$99``
   argument), so ``_subsample_table_nrows`` raises ``IndexError`` naming
   the index, the size and this note instead of silently doing something
   defensible. Every draw that stays in range behaves identically to
   classic, which is what makes ``--subsampling-ratio 0.1`` a real
   byte-for-byte oracle.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Sequence

from pymetta.comboreduct.table.table import ITable, OTable, Table
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import type_node
from pymetta.feature_selection.main.feature_selection import feature_selection
from pymetta.feature_selection.main.feature_selection_params import (
    MI_SCORER,
    feature_selection_parameters,
)
from pymetta.util.log_prog_name import determine_log_name
from pymetta.util.logger import Logger, logger
from pymetta.util.rng import randGen

__all__ = ["fs_main", "subsampleTable", "unsupported_type_exit"]

# C++: moses/moses/moses/moses_main.cc:47 `version_string`, which
# fs-main.cc:369 prints for `--version`. Duplicated from
# `moses_exec.py`'s own copy rather than imported, to keep this package
# independent of `pymetta.moses.main` (importing it would drag in the
# whole problem registry for one string). See that constant's own
# comment for why the value carries embedded quote characters.
_VERSION_STRING = '"3"."7"."0"'

# C++: fs-main.cc:47-49
_DEFAULT_LOG_FILE_PREFIX = "feature-selection"
_DEFAULT_LOG_FILE_SUFFIX = "log"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_FILE_PREFIX + "." + _DEFAULT_LOG_FILE_SUFFIX

# C++: <climits> INT_MAX -- fs-main.cc:246's own `--max-time` default.
_INT_MAX = 2_147_483_647

# C++: feature-selection.h:38 `static const std::string hc="hc";` (the
# `opencog::hc` fs-main.cc:129 splices into its own help text).
_HC = "hc"


def _bool_opt(s: str) -> bool:
    """Not itself a ported C++ symbol -- argparse ``type=`` callback standing
    in for boost::program_options' ``validate(any&, vector<string>&, bool*,
    int)``, which is what ``po::value<bool>`` uses for every ``bool`` option
    fs-main.cc declares (``--exp-distrib``, ``--hc-crossover``,
    ``--hc-widen-search``, ``--pre-positive``).

    Deliberately a local copy of ``problem_params.py``'s identical helper
    rather than an import of it: importing a private name out of
    ``pymetta.moses.main.problem_params`` would make this package depend on
    the whole MOSES problem registry for twelve lines of string matching.
    The accepted spellings and the error text are boost's own, measured
    there.
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


def _subsample_table_nrows(nrows: int, it: ITable, ot: OTable) -> None:
    """C++: moses/comboreduct/table/table.cc:872 subsampleTable

    The ``(unsigned nrows, ITable&, OTable&, TTable&)`` overload -- "Remove
    enough rows randomly so that the table has only nrows" (its own
    comment). No ``TTable`` parameter: this port carries no timestamps at
    all (``pymetta.comboreduct.table.table``'s module docstring), so C++'s
    ``tt`` is unconditionally empty here and every ``tt``-guarded statement
    is statically dead.

    # DIVERGENCE(C++ moses/comboreduct/table/table.cc:872): PLACEMENT ONLY --
    # its C++ home is ``comboreduct/table/table.cc``, whose Python
    # counterpart is off limits to this task's partition; defined here, in
    # its only caller's module, the same way
    # ``pymetta.feature_selection.main.table_output`` hosts ``ostreamTable``.

    ``ridx`` is drawn from the ORIGINAL ``size`` on every iteration, so it
    can name a row that a previous iteration already erased -- see this
    module's docstring, defect 2. C++ walks off the end of the vector;
    this raises.
    """
    # C++: table.cc:874-876 -- three OC_ASSERTs, of which only the it/ot
    # one can fire without a TTable.
    if len(it) != 0 and len(ot) != 0 and len(it) != len(ot):
        raise AssertionError("it.empty() || ot.empty() || it.size() == ot.size()")
    size = max(len(it), len(ot))
    if nrows < size:
        nremove = size - nrows
        for _ in range(nremove):  # C++: dorepeat(nremove)
            ridx = randGen().randint_n(size)
            for seq in (it, ot):
                if len(seq) == 0:
                    continue
                if ridx >= len(seq):
                    raise IndexError(
                        f"subsampleTable: row index {ridx} is past the end of a "
                        f"table of {len(seq)} rows (drawn against the ORIGINAL "
                        f"size {size}). classic dereferences past the end here -- "
                        "undefined behaviour, reproduced as a raise rather than "
                        "guessed at; see fs_main.py's module docstring, defect 2."
                    )
                del seq[ridx]


def subsampleTable(ratio: float, table: Table) -> None:
    """C++: moses/comboreduct/table/table.cc:892 subsampleTable

    The ``(float ratio, Table&)`` overload. In place, like C++'s
    reference parameter.

    ``OC_ASSERT(0.0 <= ratio and ratio <= 1.0, "Ratio must be in [0.0,
    1.0], but is %f", ratio)`` is active in release builds, so it is a
    raise, not a bare ``assert``. Its message is classic's, ``%f``
    formatting included (six decimal places), because that string is what
    a caller comparing against the real binary's stderr will see.

    THIS ASSERT IS THE ``--subsampling-ratio`` BUG. fs-main.cc:437 passes
    ``ratio * table.size()`` as this function's ``ratio``. See this
    module's docstring, defect 1. DO NOT FIX.

    # DIVERGENCE(C++ moses/comboreduct/table/table.cc:892): PLACEMENT ONLY --
    # see ``_subsample_table_nrows`` above for the identical reason.
    """
    if not (0.0 <= ratio <= 1.0):
        raise AssertionError(f"Ratio must be in [0.0, 1.0], but is {ratio:f}")
    # C++: table.cc:896 -- `ratio * table.size()` truncated to `unsigned`.
    _subsample_table_nrows(int(ratio * table.size()), table.itable, table.otable)


def unsupported_type_exit(tipe: type_node) -> None:
    """C++: moses/feature-selection/main/fs-main.cc:103 unsupported_type_exit

    The ``type_node`` overload (fs-main.cc:103-106), which forwards to the
    ``type_tree`` one (fs-main.cc:96-101); both are collapsed here, since
    Python has no overload resolution and this port's ``type_node`` is
    already the only form fs-main.cc's own call site (:451) constructs.

    Note the missing space in classic's own message -- ``"Type " << tt <<
    "currently not supported."`` -- reproduced verbatim, on both the log
    record and stderr.
    """
    logger().error("Type " + str(tipe) + "currently not supported.")
    print(f"Error: type {tipe}currently not supported.", file=sys.stderr)
    raise SystemExit(1)


def _add_options(parser: argparse.ArgumentParser) -> None:
    """C++: moses/feature-selection/main/fs-main.cc:119 main (its
    ``desc.add_options()`` block, fs-main.cc:119-352)

    The breadcrumb cites ``main`` because the option block is not a symbol
    of its own in C++ -- it is a chained call inside ``main``'s body.
    Every flag below carries fs-main.cc's own long name, short name, type
    and ``->default_value(...)``. NO ``help=`` text: ``problem_params.
    add_options`` registers its own 127 flags without any either, and
    argparse's ``--help`` rendering is already a documented divergence
    from boost's (see this module's docstring) -- so a help string here
    would be new prose that matches nothing in classic.

    Three of fs-main.cc's declared options are DEAD in classic itself and
    are still registered here so the CLI surface matches:
    ``--max-time`` ("ONLY WORKS FOR hc" per its own help, and
    ``deme_optimize.h`` never reads it), ``--jobs`` (see module
    docstring), and ``--cache-size`` (``iscorer_cache``,
    feature-selection.h:143, has zero live callers -- see
    ``feature_selection_params.py``'s ``fs_scorer`` docstring).
    """
    add = parser.add_argument

    def opt(*flags: str, dest: str, type_, default) -> None:
        add(*flags, dest=dest, type=type_, default=default)

    def sopt(*flags: str, dest: str, default: bool) -> None:
        opt(*flags, dest=dest, type_=_bool_opt, default=default)

    def vopt(*flags: str, dest: str) -> None:
        add(*flags, dest=dest, type=str, action="append", default=[])

    add("-v", "--version", action="store_true", default=False)
    opt("-a", "--algo", dest="algorithm", type_=str, default="simple")
    opt("-H", "--scorer", dest="scorer", type_=str, default=MI_SCORER)

    # ======= File I/O opts ========= (fs-main.cc:141-192)
    opt("-i", "--input-file", dest="input_file", type_=str, default="")
    opt("-u", "--target-feature", dest="target_feature_str", type_=str, default="")
    opt("--timestamp-feature", dest="timestamp_feature_str", type_=str, default="")
    vopt("-Y", "--ignore-feature", dest="ignore_features_str")
    vopt("-e", "--force-feature", dest="force_features_str")
    opt("-o", "--output-file", dest="output_file", type_=str, default="")
    opt("-l", "--log-level", dest="log_level", type_=str, default="DEBUG")
    opt("-F", "--log-file", dest="log_file", type_=str, default=_DEFAULT_LOG_FILE)
    add("-L", "--log-file-dep-opt", dest="log_file_dep_opt", action="store_true", default=False)

    # ======= Generic algo opts ========= (fs-main.cc:194-255)
    vopt("-f", "--initial-feature", dest="initial_features")
    opt("-j", "--jobs", dest="jobs", type_=int, default=1)
    opt("-C", "--target-size", dest="target_size", type_=int, default=0)
    sopt("--exp-distrib", dest="exp_distrib", default=False)
    opt("-T", "--threshold", dest="threshold", type_=float, default=0.0)
    opt("-r", "--random-seed", dest="rand_seed", type_=int, default=1)
    opt("--max-time", dest="max_time", type_=int, default=_INT_MAX)
    opt("--subsampling-ratio", dest="subsampling_ratio", type_=float, default=1.0)

    # ======= Incremental selection params ======= (fs-main.cc:257-275)
    opt("-D", "--inc-redundant-intensity", dest="inc_red_intensity", type_=float, default=0.1)
    opt(
        "-E",
        "--inc-target-size-epsilon",
        dest="inc_target_size_epsilon",
        type_=float,
        default=0.001,
    )
    opt("-U", "--inc-interaction-terms", dest="inc_interaction_terms", type_=int, default=1)

    # ======= Hill-climbing only params ======= (fs-main.cc:277-315)
    opt("-A", "--max-score", dest="hc_max_score", type_=float, default=1.0)
    opt("-m", "--max-evals", dest="hc_max_evals", type_=int, default=10000)
    opt(
        "-O",
        "--hc-fraction-of-remaining",
        dest="hc_fraction_of_remaining",
        type_=float,
        default=0.5,
    )
    opt("-s", "--cache-size", dest="hc_cache_size", type_=int, default=1000000)
    sopt("--hc-crossover", dest="hc_crossover", default=False)
    opt("--hc-crossover-pop-size", dest="hc_crossover_pop_size", type_=int, default=120)
    opt("--hc-crossover-min-neighbors", dest="hc_crossover_min_neighbors", type_=int, default=400)
    sopt("--hc-widen-search", dest="hc_widen_search", default=True)
    opt("--hc-fraction-of-nn", dest="hc_fraction_of_nn", type_=float, default=2.0)

    # ======= Stochastic max dependency params ======= (fs-main.cc:317-321)
    opt("--smd-top-size", dest="smd_top_size", type_=int, default=10)

    # ================= mi scoring ==================== (fs-main.cc:323-331)
    opt("-c", "--mi-penalty", dest="mi_confi", type_=float, default=100.0)

    # ================= pre scoring ==================== (fs-main.cc:334-351)
    opt("--pre-penalty", dest="pre_penalty", type_=float, default=1.0)
    opt("--pre-min-activation", dest="pre_min_activation", type_=float, default=0.5)
    opt("--pre-max-activation", dest="pre_max_activation", type_=float, default=1.0)
    sopt("--pre-positive", dest="pre_positive", default=True)


def _explicit_options(args: Sequence[str]) -> dict[str, object]:
    """C++: opencog/util/log_prog_name.cc:64 determine_log_name (its
    ``defaulted()`` test)

    Not a C++ symbol of its own; the same reconstruction ``moses_exec.py``
    documents at length. ``boost::program_options::variables_map`` records
    per option whether the user TYPED it, argparse does not, so the filter
    is rebuilt by re-parsing against a parser whose every default is
    ``argparse.SUPPRESS``. Comparing values against the defaults would be
    wrong: boost counts ``-r 1`` as non-defaulted even though 1 IS the
    default, and that option still lands in the file name.

    Returns a mapping from LONG option name (without ``--``), the key
    ``variables_map`` itself uses.
    """
    parser = argparse.ArgumentParser(prog="feature-selection", add_help=False)
    _add_options(parser)
    dest_to_long: dict[str, str] = {}
    for action in parser._actions:
        action.default = argparse.SUPPRESS
        for opt in action.option_strings:
            if opt.startswith("--"):
                dest_to_long[action.dest] = opt[2:]
                break
    ns = parser.parse_args(list(args))
    return {dest_to_long[d]: v for d, v in vars(ns).items() if d in dest_to_long}


def fs_main(argv: Sequence[str] | str) -> int:
    """C++: moses/feature-selection/main/fs-main.cc:108 main

    ``argv[0]`` is the program name, so only ``argv[1:]`` reaches the
    parser -- the convention every ``moses_exec`` overload keeps too. A
    bare ``str`` is split on whitespace first.

    Returns ``1`` for ``--help`` / no arguments (fs-main.cc:362-365, which
    really does ``return 1`` for a successful ``--help``) and ``0`` for
    ``--version`` (:367-372) and for a completed run -- C++'s ``main``
    falls off the end after ``feature_selection(table, fs_params)``,
    which is a ``return 0``. ``unsupported_type_exit`` and
    ``err_empty_features`` raise ``SystemExit`` instead, matching their
    C++ ``exit(1)`` / ``exit(0)``.
    """
    if isinstance(argv, str):
        return fs_main(argv.split())

    argv_list = list(argv)
    parser = argparse.ArgumentParser(prog="feature-selection", add_help=True)
    _add_options(parser)

    # C++: fs-main.cc:362 `if (vm.count("help") || argc == 1)`. argparse
    # handles `--help` itself (SystemExit); `argc == 1` -- the program name
    # alone -- has no argparse equivalent and is reproduced here.
    if len(argv_list) <= 1:
        parser.print_help()
        return 1

    args = parser.parse_args(argv_list[1:])

    # C++: fs-main.cc:367-372
    if args.version:
        print("feature-selection " + _VERSION_STRING)
        return 0

    fs_params = feature_selection_parameters(
        algorithm=args.algorithm,
        scorer=args.scorer,
        input_file=args.input_file,
        target_feature_str=args.target_feature_str,
        timestamp_feature_str=args.timestamp_feature_str,
        ignore_features_str=list(args.ignore_features_str),
        force_features_str=list(args.force_features_str),
        initial_features=list(args.initial_features),
        output_file=args.output_file,
        target_size=args.target_size,
        exp_distrib=args.exp_distrib,
        threshold=args.threshold,
        jobs=args.jobs,
        subsampling_ratio=args.subsampling_ratio,
        inc_target_size_epsilon=args.inc_target_size_epsilon,
        inc_red_intensity=args.inc_red_intensity,
        inc_interaction_terms=args.inc_interaction_terms,
        smd_top_size=args.smd_top_size,
        hc_max_evals=args.hc_max_evals,
        max_time=args.max_time,
        hc_max_score=args.hc_max_score,
        hc_cache_size=args.hc_cache_size,
        hc_fraction_of_remaining=args.hc_fraction_of_remaining,
        hc_crossover=args.hc_crossover,
        hc_crossover_pop_size=args.hc_crossover_pop_size,
        hc_crossover_min_neighbors=args.hc_crossover_min_neighbors,
        hc_widen_search=args.hc_widen_search,
        hc_fraction_of_nn=args.hc_fraction_of_nn,
        mi_confi=args.mi_confi,
        pre_penalty=args.pre_penalty,
        pre_min_activation=args.pre_min_activation,
        pre_max_activation=args.pre_max_activation,
        pre_positive=args.pre_positive,
    )

    # C++: fs-main.cc:374-391 -- "Set log"
    log_file = args.log_file
    if args.log_file_dep_opt:
        # C++: fs-main.cc:376-379 -- BOTH names are ignored here, unlike
        # `moses`, which ignores only `log-file-dep-opt`.
        ignore_opt = {"log-file-dep-opt", "log-file"}
        log_file_prefix = _DEFAULT_LOG_FILE_PREFIX
        if log_file != _DEFAULT_LOG_FILE:
            log_file_prefix = log_file
        log_file = determine_log_name(
            log_file_prefix,
            _explicit_options(argv_list[1:]),
            ignore_opt,
            "." + _DEFAULT_LOG_FILE_SUFFIX,
        )

    # C++: fs-main.cc:394 `remove(log_file.c_str());` -- C's remove() ignores
    # a missing file, and it runs BEFORE set_filename because the writer
    # opens with "a" and would otherwise append to the previous run's log.
    with contextlib.suppress(FileNotFoundError):
        os.remove(log_file)
    logger().set_filename(log_file)

    # C++: fs-main.cc:396-403
    level = Logger.get_level_from_string(args.log_level.strip())
    if level == Logger.Level.BAD_LEVEL:
        print(
            f"Fatal Error: Log level {args.log_level} is incorrect (see --help).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    logger().set_level(level)
    logger().set_backtrace_level(Logger.Level.ERROR)

    # C++: fs-main.cc:406-414 -- "Log command-line args"
    logger().info("feature-selection version " + _VERSION_STRING)
    logger().info("Command line:" + "".join(" " + a for a in argv_list))

    # C++: fs-main.cc:417 -- init random generator
    randGen().seed(args.rand_seed)

    # C++: fs-main.cc:420 `setting_omp(fs_params.jobs);` -- not ported, see
    # this module's docstring.

    # C++: fs-main.cc:423
    logger().info("Read input file %s", fs_params.input_file)

    # C++: fs-main.cc:429-432
    table = loadTable(
        fs_params.input_file,
        fs_params.target_feature_str,
        fs_params.timestamp_feature_str,
        fs_params.ignore_features_str,
    )

    logger().debug("Table size = " + str(table.size()))  # C++: fs-main.cc:434

    # C++: fs-main.cc:436-439. `subsampling_ratio * table.size()` is passed
    # as a RATIO -- see this module's docstring, defect 1. DO NOT FIX.
    if fs_params.subsampling_ratio < 1.0:
        subsampleTable(fs_params.subsampling_ratio * table.size(), table)
        logger().debug("Table has been resampled to size = " + str(table.size()))

    # C++: fs-main.cc:441-443
    inferred_type = table.get_signature().otype

    # C++: fs-main.cc:446-452 -- "Go and do it."
    if inferred_type in (type_node.boolean_type, type_node.enum_type, type_node.contin_type):
        feature_selection(table, fs_params)
    else:
        unsupported_type_exit(inferred_type)
    return 0
