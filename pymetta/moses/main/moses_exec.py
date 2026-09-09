"""C++: moses/moses/main/moses_exec.{h,cc}, moses-exec.cc, moses_exec_def.{h,cc}

The CLI entry point: parse argv, build the problem registry, dispatch to
whichever `problem_base` matches `-H`, and let it print its own results.
`moses_exec.cc`'s ENTIRE dispatch body is:

    option_manager mgr;
    problem_params pms;
    problem_manager pmr;
    register_table_problems(pmr, mgr);
    register_demo_problems(pmr, mgr);
    mgr.register_options(&pms);
    mgr.init_options();
    mgr.parse_options(argc, argv);
    problem_base* probm = pmr.find_problem(pms.problem);
    if (probm) { probm->run(&pms); return 0; }
    logger().error() << "problem type \\"" << pms.problem << "\\" is currently unsupported.";
    std::cerr << "problem type \\"" << pms.problem << "\\" is currently unsupported." << std::endl;
    return 1;

`option_manager`/`option_base` are not ported anywhere in this project
(`problem.py`'s own module docstring): this port's CLI options are
argparse-based, so `mgr.register_options`/`init_options`/`parse_options`
collapse into building ONE shared `argparse.ArgumentParser` and registering
every `option_base`-equivalent's flags onto it directly -- the same "many
option surfaces, one parser" composition `table_problems.py`'s own
`parse_table_args` already uses (and explicitly invites this module to
supersede: "S13 Task 3's `moses_exec.py` (not yet built) is free to
supersede this once it exists").

## The three C++ overloads, merged into one Python function

`moses_exec.h` declares three overloads -- `moses_exec(int argc, char**
argv)` (the one with the real body, moses_exec.cc:38-63), `moses_exec(const
vector<string>&)` (moses_exec.cc:65-74, converts to `char**` and re-calls
the first), and `moses_exec(const string&)` (moses_exec.cc:76-81, splits on
whitespace via `boost::is_space()` and re-calls the second). Python has no
`argc`/`char**` distinction, so the first two collapse into ONE signature,
`moses_exec(argv: Sequence[str])`, dispatched on `isinstance` the same way
`Table.__init__` (table.py) merges two C++ constructor overloads -- see
this function's own docstring for the "index 0 is the program name"
convention both retain (moses_exec.h:35-36's own comment: "note that the
first still represents the name of the supposed executable").

## What is NOT ported here, and why

* `option_manager`/`option_base`, `problem.py`'s own module docstring.
* `-h`/`--help` and malformed-CLI-argument handling are left to argparse's
  own behavior (`SystemExit` on `-h`, `argparse.ArgumentError`-derived
  exceptions on a bad flag) rather than reproducing boost::program_options'
  own error text -- neither `option_manager`'s help/usage formatting nor
  its exact error strings are ported anywhere in this project (see
  `problem.py`'s module docstring for the `option_manager` scope note this
  follows).
* A `ValueError`/`NotImplementedError` raised by `problem_params.parse_options`
  or by a `problem_base.run()` override (both already gate-then-raise
  their own unported subsystems, per those modules' own docstrings) is
  NOT caught and converted to a return code here: C++'s equivalent path is
  `OC_ASSERT`/`exit(1)`, which never returns control to `moses_exec.cc`'s
  own body either -- an uncaught Python exception propagating out of this
  function is the direct analogue of a process that already exited.
  `find_problem` returning `None` is the ONE case classic's own
  `moses_exec.cc` handles with an ordinary `return 1;` (a real, in-band
  return, not an `exit()`), so that is the one case ported as a plain
  return value below rather than a raised exception.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
from collections.abc import Sequence

from pymetta.comboreduct.combo.iostream_combo import parse_output_format
from pymetta.moses.main.demo_problems import register_demo_problems
from pymetta.moses.main.problem import problem_manager
from pymetta.moses.main.problem_params import problem_params
from pymetta.moses.moses.moses_main import metapop_printer
from pymetta.util.log_prog_name import determine_log_name
from pymetta.util.logger import Logger, logger

__all__ = ["moses_exec"]

# C++: moses/moses/moses/moses_main.cc:47 `version_string`. Its home is
# `moses_main.cc`, whose Python counterpart `moses_main.py` this task does not
# own, so the constant is defined here instead of there.
#
# The value looks wrong and is not. `moses/moses/CMakeLists.txt:1` passes
# `-DMOSES_VERSION_MAJOR="3"` -- a define whose expansion is a STRING LITERAL,
# quotes included -- and `moses_main.cc:35-36`'s `stringify(x)` is `#x`, which
# stringifies that literal a second time. So `version_string` is
# `"3"."7"."0"`, WITH the embedded quote characters, and classic's log really
# does read `moses version "3"."7"."0"`. Measured against the binary
# 2026-09-09. The three numbers come from CMakeLists.txt:13-15.
_VERSION_STRING = '"3"."7"."0"'

# C++: moses/moses/main/moses_exec_def.h:54-55 -- `default_log_file_prefix`
# ("moses") and `default_log_file_suffix` ("log"). `problem_params` already
# carries the `default_log_file` built from them as its `-f` default; `-F`
# needs the two halves separately.
_DEFAULT_LOG_FILE_PREFIX = "moses"
_DEFAULT_LOG_FILE_SUFFIX = "log"


def _add_demo_options(parser) -> None:
    """C++: moses/moses/main/demo-problems.cc:46 demo_params::add_options

    `demo_params` (`demo_problems.py`) is explicitly "Not an `option_base`"
    (its own class docstring) -- `register_demo_problems` builds and
    returns one, but nothing registers ITS two CLI flags
    (`-y`/`--combo-program`, `-k`/`--problem-size`) against any parser;
    that module's own docstring names this file, `moses_exec.py`, as the
    caller responsible for doing so. Ported directly from
    demo-problems.cc:49-61: `combo-program,y` (`po::value<string>(&combo_str)`,
    no `->default_value`, i.e. an empty string) and `problem-size,k`
    (`po::value<unsigned int>(&problem_size)->default_value(5)`).
    """
    parser.add_argument("-y", "--combo-program", dest="combo_str", type=str, default="")
    parser.add_argument("-k", "--problem-size", dest="problem_size", type=int, default=5)


def _explicit_options(build_parser, args: Sequence[str]) -> dict[str, object]:
    """C++: opencog/util/log_prog_name.cc:64 determine_log_name (its `defaulted()` test)

    Not a C++ symbol of its own. `determine_log_name` filters
    `boost::program_options::variables_map` with `!it->second.defaulted()`,
    and `variables_map` records that flag per option; argparse does not, so
    the filter has to be reconstructed.

    Comparing the parsed value against `parser.get_default(dest)` would be
    wrong: boost calls an option non-defaulted when the user TYPED it, even
    if they typed its default value, so `-m 100` with a default of 100 still
    lands in the file name. Re-parsing against a parser whose every default
    is `argparse.SUPPRESS` reproduces the real test -- the resulting namespace
    holds exactly the options that appeared on the command line.

    `build_parser` is a zero-argument callable that returns a FRESH parser
    with the same options registered, because the defaults have to be
    suppressed on a parser nobody else is using.

    Returns a mapping from LONG option name (without `--`, the key boost's
    `variables_map` uses) to the parsed value.
    """
    parser = build_parser()
    dest_to_long: dict[str, str] = {}
    for action in parser._actions:
        action.default = argparse.SUPPRESS
        for opt in action.option_strings:
            if opt.startswith("--"):
                dest_to_long[action.dest] = opt[2:]
                break
    ns = parser.parse_args(list(args))
    return {dest_to_long[dest]: value for dest, value in vars(ns).items() if dest in dest_to_long}


def _setup_logging(pms: problem_params, explicit: dict[str, object]) -> None:
    """C++: moses/moses/main/problem-params.cc:2069 problem_params::parse_options

    The `-l` / `-f` / `-F` half of `parse_options`, transliterated from
    problem-params.cc:2069-2113 and :2185-2186:

        bool have_log_file_opt = vm.count(log_file_dep_opt_opt.first) > 0;
        if (have_log_file_opt) { ... log_file = determine_log_name(...); }
        remove(log_file.c_str());
        logger().set_filename(log_file);
        boost::trim(log_level);
        Logger::Level level = logger().get_level_from_string(log_level);
        if (level != Logger::BAD_LEVEL) logger().set_level(level);
        else { cerr << ...; exit(1); }
        logger().set_backtrace_level(Logger::ERROR);
        ...
        logger().info() << "moses version " << version_string;
        gethostname(hname, 256); logger().info("hostname: %s", hname);
        ...
        logger().warn("WARNING: This version of MOSES does NOT have MPI support!\\n");

    Three details are easy to lose and all three are observable:

    * `remove(log_file)` runs BEFORE `set_filename`, because the writer opens
      the file with `fopen(..., "a")` and would otherwise append to the
      previous run's log.
    * `set_backtrace_level(ERROR)` is what makes `Logger::log` flush after
      every ERROR record.
    * the MPI warning ends in an explicit `\\n` ON TOP of the newline
      `Logger::log` adds, so classic's log has a BLANK LINE after it. That
      blank line is real output, not a formatting artifact.

    # DIVERGENCE(moses/moses/main/problem-params.cc:2069): this block's C++
    # home is the middle of `problem_params::parse_options`, i.e.
    # `problem_params.py`, which this task does not own. It is called from
    # `moses_exec` instead, immediately after `parse_options` returns.
    # Ordering is unaffected: NOTHING between problem-params.cc:2069 and the
    # end of `parse_options` writes to the log except the three records
    # emitted here, and `table_problem_params::parse_options` -- the only
    # other `option_base` in the set -- has no `logger()` call at all
    # (grep-verified over table-problems.cc). Moving it to its C++ home is a
    # one-line insert; see this task's report.

    # DIVERGENCE(moses/moses/main/problem-params.cc:2097): C++ prints to
    # cerr and `exit(1)` on an unparseable `-l`. `problem_params.parse_options`
    # already raises `ValueError` for exactly that case and runs FIRST, so the
    # `else` arm here is unreachable from the CLI; it is written as a plain
    # return rather than duplicating either behaviour.

    # DIVERGENCE(moses/moses/main/problem-params.cc:2103): classic installs a
    # SIGHUP handler that logs "Caught SIGHUP" and dumps the stack. Not
    # ported -- `backtrace()` is a no-op in this port (logger.py's own
    # DIVERGENCE), so the handler would log a bare line and nothing else.
    """
    log_file = pms.log_file

    # C++: :2070 `vm.count(log_file_dep_opt_opt.first) > 0` -- the presence of
    # `-F`, not its value; it is declared with no `po::value<>` at all.
    if pms.log_file_dep_opt:
        # C++: :2085 `std::set<std::string> ignore_opt{log_file_dep_opt_opt.first};`
        log_file = determine_log_name(
            _DEFAULT_LOG_FILE_PREFIX,
            explicit,
            {"log-file-dep-opt"},
            "." + _DEFAULT_LOG_FILE_SUFFIX,
        )
        pms.log_file = log_file

    # C++: :2091 `remove(log_file.c_str());` -- C's remove() ignores a
    # missing file, so the FileNotFoundError is suppressed rather than raised.
    with contextlib.suppress(FileNotFoundError):
        os.remove(log_file)

    logger().set_filename(log_file)
    level = Logger.get_level_from_string(pms.log_level.strip())
    if level == Logger.Level.BAD_LEVEL:
        return
    logger().set_level(level)
    logger().set_backtrace_level(Logger.Level.ERROR)

    logger().info("moses version " + _VERSION_STRING)
    logger().info("hostname: %s", socket.gethostname())
    logger().warn("WARNING: This version of MOSES does NOT have MPI support!\n")


def moses_exec(argv: Sequence[str] | str) -> int:
    """C++: moses/moses/main/moses_exec.cc:38,65,76 moses_exec (all three overloads)

    `argv[0]` is the program name, matching every C++ overload's own
    convention (see module docstring) -- only `argv[1:]` is ever handed to
    the argument parser. A bare `str` is split on whitespace first
    (`moses_exec(const string&)`, moses_exec.cc:76-81's
    `boost::split(argvs, args, boost::is_space())` -- Python's `str.split()`
    with no separator argument splits on any run of whitespace the same
    way).

    Returns `0` on a successful run (`probm->run(&pms); return 0;`,
    moses_exec.cc:56-57) or `1` when `-H <name>` names no registered
    problem (moses_exec.cc:60-62) -- the ONE failure path that is a
    genuine in-band return in C++ rather than an `exit()`; see the module
    docstring for why every OTHER failure mode is left to propagate as a
    raised Python exception instead of being caught and turned into a
    return code here.
    """
    if isinstance(argv, str):
        return moses_exec(argv.split())

    argv_list = list(argv)

    # Deferred imports: `table_problems.py` is a concurrently-developed
    # sibling module (this task's own brief: "One concurrent agent owns
    # pymetta/moses/main/table_problems.py"). Importing it lazily, inside
    # the function body rather than at module load time, keeps every
    # OTHER symbol in this file -- and every test of them -- importable
    # and runnable independently of whether that file has landed yet,
    # exactly the precedent `moses_main.py`'s own `run_moses` already set
    # for `local_moses` during S11 ("write your module and tests against
    # it anyway ... do not reimplement it").
    from pymetta.moses.main.table_problems import register_table_problems

    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    dp = register_demo_problems(pmr)

    pp = problem_params()

    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="moses", add_help=True)
        pp.add_options(p)
        tpp.add_options(p)
        _add_demo_options(p)
        return p

    parser = build_parser()

    ns = parser.parse_args(argv_list[1:])
    pp.parse_options(ns)
    tpp.parse_options(ns)
    dp.combo_str = ns.combo_str
    dp.problem_size = ns.problem_size

    # C++: problem-params.cc:2069-2113 and :2185-2186, reached from
    # `option_manager::parse_options`' `for (auto ob : _option_set)
    # ob->parse_options(vm);` loop (problem.cc:75-77). See `_setup_logging`'s
    # own DIVERGENCE marker for why it is called from here rather than from
    # inside `problem_params.parse_options`.
    _setup_logging(pp, _explicit_options(build_parser, argv_list[1:]))

    # C++: moses/moses/main/problem.cc:82-87 -- the tail of
    # `option_manager::parse_options`, whose own comment explains the
    # placement: "Log command-line args. Do this after above, since one of
    # the options might change the log file location." `option_manager` is
    # not ported (see this module's docstring), so its one logging statement
    # lands here, at the point in this function that corresponds to the end
    # of that C++ function. C++ joins ALL of `argv`, including `argv[0]`, each
    # preceded by a space -- so the record reads "Command line: ./moses -i ...".
    logger().info("Command line:" + "".join(" " + a for a in argv_list))

    # C++: problem-params.cc:2396-2403, the tail of
    # `problem_params::parse_options` -- `combo::output_format fmt =
    # parse_output_format(output_format_str); mmr_pa = metapop_printer(
    # result_count, output_score, output_cscore, output_bscore,
    # output_only_best, boosting, output_eval_number, output_with_labels,
    # output_deme_id, col_labels, output_file, fmt, enable_mpi);`.
    # `col_labels` is populated later, by `table_problem_base.common_setup`
    # (`pms.mmr_pa.ilabels = ilabels`, table-problems.cc:164) once the
    # input file's own column labels are known -- an empty list here
    # matches C++'s own `col_labels` at THIS point (`problem-params.h:234`
    # default-constructs an empty `std::vector<std::string>`; nothing
    # populates it before this line runs either).
    #
    # This assignment lives here, in `moses_exec.py`, rather than inside
    # `problem_params.parse_options` itself: `problem_params.py`'s own
    # module docstring places `result_count`/`output_*` wiring explicitly
    # OUT of that file's scope ("their eventual gate-if-unsupported logic
    # belongs to whichever future task actually wires them
    # (`table_problems.py`, `moses_exec.py`)"), and `demo_problems.py`'s
    # `bool_problem_base.run` already reads the printer as `pms.printer`
    # (not `pms.mmr_pa`) -- the exact attribute name this line uses.
    fmt = parse_output_format(pp.output_format_str)
    pp.printer = metapop_printer(
        pp.result_count,
        pp.output_score,
        pp.output_cscore,
        pp.output_bscore,
        pp.output_only_best,
        pp.boosting,
        pp.output_eval_number,
        pp.output_with_labels,
        pp.output_deme_id,
        [],
        pp.output_file,
        fmt,
        pp.enable_mpi,
    )

    probm = pmr.find_problem(pp.problem)
    if probm is not None:
        probm.run(pp)
        return 0

    print(f'problem type "{pp.problem}" is currently unsupported.', file=sys.stderr)
    return 1
