"""Derive a log file name from the program options that were actually given.

C++: opencog/util/log_prog_name.h:36 determine_log_name

This is what `-F` / `--log-file-dep-opt` selects. Instead of writing
`moses.log`, MOSES names the file after every option the user typed, so a
seed sweep leaves one distinctly-named log per run:

    moses_input-file_data.csv_log-level_DEBUG_max-evals_10_random-seed_5.log

Four rules decide that string, all of them from `determine_log_name`
(log_prog_name.cc:55-82) and all observable:

1. Only options the user actually supplied appear -- `!it->second.defaulted()`
   skips anything that fell back to its `->default_value(...)`. `-F` itself is
   skipped separately, via `ignore_opt`.
2. The order is `variables_map`'s own iteration order. `variables_map` derives
   from `std::map<std::string, variable_value>`, so that is LEXICOGRAPHIC by
   LONG option name -- not the order the user typed, and not the order the
   options were declared. Measured against the classic binary 2026-09-09:
   `-i .../dataset.csv -m10 -W1 -F -r5 -l DEBUG` produced
   `input-file`, `log-level`, `max-evals`, `output-with-labels`, `random-seed`.
3. An option is DROPPED, not truncated, when appending it would take the name
   to 255 characters or more -- and the check is per-option, so a long value
   in the middle can be skipped while a later short one still makes it in.
4. `/` becomes `d` last of all, over the WHOLE name including the suffix,
   because a Unix file name cannot contain a path separator. (The `.` in
   `dataset.csv` survives; only slashes are touched.)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = ["determine_log_name", "to_string"]

# C++: log_prog_name.cc:56 `const static unsigned int max_filename_size = 255;`
# -- "this is because OSs usually do not handle file name above 255 chars".
_MAX_FILENAME_SIZE = 255


def to_string(vv: object, separator: str = "_") -> str:
    """C++: opencog/util/log_prog_name.cc:34 to_string

    Renders one option value the way `operator<<` would, trying `int`,
    `unsigned int`, `long`, `unsigned long`, `float`, `double`, `bool`,
    `std::string` in that order, and then the `std::vector<T>` of each
    (log_prog_name.h:45-60), joined with `separator`.

    Two of those conversions are visible in the resulting file name:

    * `bool` streams WITHOUT `std::boolalpha`, so it is `1`/`0`, never
      `true`/`false`. `-W1` really does produce `output-with-labels_1`.
    * `float`/`double` stream at an `ostream`'s default precision of 6
      significant digits, which is Python's `%g`, not `str(float)`.

    The order also matters for `bool`: C++ tries the integral types first, and
    `bool` is a distinct `typeid` from `int`, so a bool never matches earlier.
    Python's `bool` IS an `int` subclass, so the bool test must come FIRST here
    to get `1`/`0` rather than `True`/`False`.
    """
    if isinstance(vv, str):
        return vv
    if isinstance(vv, bool):
        return "1" if vv else "0"
    if isinstance(vv, int):
        return str(vv)
    if isinstance(vv, float):
        return f"{vv:g}"
    if isinstance(vv, Iterable):
        return separator.join(to_string(v, separator) for v in vv)
    # C++: log_prog_name.cc:46 `cerr << "type not handled yet" << endl;` and
    # returns the empty string it started with.
    return ""


def determine_log_name(
    log_file_prefix: str,
    vm: Mapping[str, object],
    ignore_opt: set[str],
    log_file_suffix: str = ".log",
) -> str:
    """C++: opencog/util/log_prog_name.cc:55 determine_log_name

    `vm` must contain ONLY the options that were explicitly supplied, keyed by
    LONG option name: C++ filters `variables_map` with `!it->second.defaulted()`
    inside this loop, but Python's argparse records no such flag, so the caller
    does that half (see `moses_exec.py`'s `_explicit_options`). Iteration is
    over `sorted(vm)` for the same reason -- C++ gets lexicographic order for
    free from `std::map`, and a Python dict would otherwise hand back insertion
    order.

    The size test is C++'s exactly, including its off-by-one flavour: the
    candidate is `log_file.size() + str.size() + log_file_suffix.size()` and it
    must be strictly `< 255`, so the suffix is counted for every option even
    though it is appended only once.

    The trailing `OC_ASSERT(log_file.size() <= max_filename_size)` is kept as
    an assert: it can only fire if `log_file_prefix` alone already exceeds the
    limit, since the loop refuses to cross it.
    """
    log_file = log_file_prefix
    for name in sorted(vm):
        if name in ignore_opt:
            continue
        s = "_" + name + "_" + to_string(vm[name])
        expected_max_size = len(log_file) + len(s) + len(log_file_suffix)
        if expected_max_size < _MAX_FILENAME_SIZE:
            log_file += s
    log_file += log_file_suffix
    # C++: `replace(log_file.begin(), log_file.end(), '/', 'd');` -- a single
    # character replacement over the finished name, suffix included.
    log_file = log_file.replace("/", "d")
    assert len(log_file) <= _MAX_FILENAME_SIZE
    return log_file
