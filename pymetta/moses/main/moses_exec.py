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
import sys
from collections.abc import Sequence

from pymetta.comboreduct.combo.iostream_combo import parse_output_format
from pymetta.moses.main.demo_problems import register_demo_problems
from pymetta.moses.main.problem import problem_manager
from pymetta.moses.main.problem_params import problem_params
from pymetta.moses.moses.moses_main import metapop_printer

__all__ = ["moses_exec"]


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
    parser = argparse.ArgumentParser(prog="moses", add_help=True)
    pp.add_options(parser)
    tpp.add_options(parser)
    _add_demo_options(parser)

    ns = parser.parse_args(argv_list[1:])
    pp.parse_options(ns)
    tpp.parse_options(ns)
    dp.combo_str = ns.combo_str
    dp.problem_size = ns.problem_size

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
