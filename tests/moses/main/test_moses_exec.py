"""Oracle and guard tests for pymoses.moses.main.moses_exec (S13 Task 3).

This is the layer a user actually invokes: `moses_exec(argv)` end to end,
argv-to-stdout, with NO configuration supplied by this test beyond the
input dataset (`docs/CARRY-FORWARD.md` Sec 5.4's own rule -- "at least one
oracle per slice must run classic's own defaults", written precisely
because an oracle the port itself configures cannot detect the port's own
configuration being wrong).

`_DATASET_CSV` is byte-for-byte `$CLASSIC/tests/moses/dataset.csv` (9
rows, columns res,arg1,arg2). Every `_EXPECTED_*` string below was
captured by ACTUALLY RUNNING `$CLASSIC/build/moses/moses/main/moses`
(bounded: `( ulimit -v 2097152; timeout 60 <cmd> )`) with exactly the
flags quoted in each test's own docstring/command comment -- not
reconstructed from documentation or from this port's own understanding of
the format. classic's own stdout is not re-captured at test-run time (no
test here shells out to a compiled C++ binary): `tools/ci.sh` must not
require a C++ toolchain (`tools/_capture_env.py`'s own module docstring),
matching the precedent `test_moses_main.py`'s "Live oracle" section
already set ("captured by actually running classic's own moses binary
... recorded in each test's own docstring, not re-derived here").

`test_output_labels`/`test_python_output` (`MOSESUTest.cxxtest:188,198`)
are BOTH reproduced by `test_moses_exec_reproduces_classic_it_oracle`
below: read literally, `test_python_output`'s C++ body is a byte-for-byte
copy of `test_output_labels` (same arguments -- `-i..., -m1000, -W1`, no
`--output-format`/`-o` involved) -- checked directly against
`MOSESUTest.cxxtest`, not assumed; classic's own test suite does not
actually exercise python-format output under that name. This module
still exercises `--output-format=python` for real
(`test_moses_exec_reproduces_classic_python_format_oracle`), just not
under either of those two names, since neither name's C++ body does.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.moses.main.moses_exec import moses_exec

# ---------------------------------------------------------------------------
# dataset.csv -- byte-for-byte $CLASSIC/tests/moses/dataset.csv
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


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text(_DATASET_CSV)
    return str(path)


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# The required exit-criterion oracle: classic's OWN defaults, this port's
# real CLI, byte-for-byte stdout comparison.
# ---------------------------------------------------------------------------

# Captured: ( ulimit -v 2097152; timeout 60
#   $CLASSIC/build/moses/moses/main/moses -i$CLASSIC/tests/moses/dataset.csv
#   -m1000 -W1 )
# The same run without `-W1` -- `$1`/`$2` instead of `$arg1`/`$arg2`.
# Named because several tests below need the plain-output oracle to prove a
# flag they pass is inert in classic too.
_EXPECTED_IT_NO_LABELS = (
    "-3 and($1 $2) \n"
    "-4 $1 \n"
    "-4 $2 \n"
    "-4 and(!$1 !$2) \n"
    "-4 or(!$1 $2) \n"
    "-4 or($1 !$2) \n"
    "-5 true \n"
    "-5 !$1 \n"
    "-5 !$2 \n"
)

_EXPECTED_IT_LABELS = (
    "-3 and($arg1 $arg2) \n"
    "-4 $arg1 \n"
    "-4 $arg2 \n"
    "-4 and(!$arg1 !$arg2) \n"
    "-4 or(!$arg1 $arg2) \n"
    "-4 or($arg1 !$arg2) \n"
    "-5 true \n"
    "-5 !$arg1 \n"
    "-5 !$arg2 \n"
)


def test_moses_exec_reproduces_classic_it_oracle(dataset_csv):
    """The task's own stated oracle: `-i<dataset.csv> -m1000 -W1`, no
    other configuration supplied. Also IS `test_output_labels` and
    `test_python_output` (`MOSESUTest.cxxtest:188,198`) -- identical
    argv, and the top line's tree (`and($arg1 $arg2)`) is exactly what
    both tests assert via `cheap_moses_test_combo` (see module docstring
    for why they share one body).
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_IT_LABELS


def test_moses_exec_reproduces_classic_it_oracle_no_labels(dataset_csv):
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses -i<dataset.csv> -m1000 ) -- same
    search, `$1`/`$2` instead of `$arg1`/`$arg2` since `-W` is not passed.
    This is the case `test_metapop_printer_labels_gate_is_mutation_
    detectable` (test_moses_main.py) exercises at the unit level; here it
    is proven through the real CLI end to end.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000"])
    assert rc == 0
    assert out == _EXPECTED_IT_NO_LABELS


def test_moses_exec_reproduces_classic_cscore_oracle(dataset_csv):
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses -i<dataset.csv> -m1000 -W1 -x1 )
    -- exercises the binary32-at-double-width formatting
    (`-3.57142853736877441` / `0.571428596973419189`) this task's own
    brief singles out by name.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-W1", "-x1"])
    assert rc == 0
    assert out.splitlines()[0] == (
        "-3 and($arg1 $arg2)  [score=-3, penalized score=-3.57142853736877441, "
        "complexity=2, complexity penalty=0.571428596973419189, "
        "uniformity penalty=0]"
    )
    assert len(out.splitlines()) == 9


def test_moses_exec_reproduces_classic_python_format_oracle(dataset_csv):
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses -i<dataset.csv> -m1000
    --output-format=python ), first generated program only.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "--output-format=python"])
    assert rc == 0
    assert out.startswith(
        "#!/usr/bin/env python\n"
        "#score: -3\n"
        "import operator as op\n"
        "from functools import reduce\n"
        "from math import log, exp, sin\n"
        "def l0(i): return 0 < i\n"
        "def adds(*args): return sum(args)\n"
        "def muls(*args): return reduce(op.mul, args)\n"
        "def pdiv(a, b): return a / (b + 0.000001)\n"
        "def impulse(a): return 1.0 if a else 0.0\n"
        "def moses_eval(i):\n"
        "    return (i[0] and i[1]) \n"
    )


def test_moses_exec_reproduces_classic_only_best_and_eval_number_oracle(dataset_csv):
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses -i<dataset.csv> -m1000 -C1 ) ->
    single line `-3 and($1 $2) `; and ... -V1 ) -> same 9 lines as the
    no-labels oracle above, plus a trailing `n_evals: 33`.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-C1"])
    assert rc == 0
    assert out == "-3 and($1 $2) \n"

    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-V1"])
    assert rc == 0
    lines = out.splitlines()
    assert lines[-1] == "n_evals: 33"
    assert lines[0] == "-3 and($1 $2) "


def test_moses_exec_output_file_matches_stdout_oracle(dataset_csv, tmp_path):
    """`-o<file>` redirects the SAME text to a file instead of stdout
    (moses_main.h:195-201) -- also the mechanism `MOSESUTest`'s own
    `cheap_moses_test_combo` uses (`build_cmd` always appends `-o
    <tempfile>`, moses-framework.h:49-56) to read results back.
    """
    out_path = tmp_path / "results.txt"
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-W1", f"-o{out_path}"])
    assert rc == 0
    assert out == ""
    assert out_path.read_text() == _EXPECTED_IT_LABELS


# ---------------------------------------------------------------------------
# P2 metapopulation un-gates (CLI-gate-reconciliation task): --discard-
# dominated and --diversity-pressure are REAL now (domination.cc/
# diversity.cc, `f1c439f`/`0c1beda`) and no longer raise from
# problem_params.parse_options. `tests/fixtures/table_data/dataset.csv`
# (comment lines, no header, 2-column boolean input) is the task's own
# named oracle dataset for both -- NOT `_DATASET_CSV` above -- because it
# is small enough that `--discard-dominated`'s pruning and
# `--diversity-pressure`'s tie-break reordering are both directly visible
# in a 10-line default `it` run. Every `_EXPECTED_*` string below was
# captured the same way this module's own docstring requires: actually
# running `$CLASSIC/build/moses/moses/main/moses` (bounded: `( ulimit -v
# 2097152; timeout 60 <cmd> )`) with exactly the flags named.
# ---------------------------------------------------------------------------

_COMMENTED_DATASET_PATH = "tests/fixtures/table_data/dataset.csv"


def test_moses_exec_reproduces_classic_discard_dominated_oracle():
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses
    -i tests/fixtures/table_data/dataset.csv -m1000 --discard-dominated=1
    -W1 ) -- domination pruning collapses the usual 10-candidate default
    `it` result down to 2. Before this task, this raised `NotImplementedError`
    at parse time; the un-gate is proven by an exact byte-for-byte match to
    classic's own binary, not merely by "no exception raised".
    """
    rc, out = _run(
        ["moses", f"-i{_COMMENTED_DATASET_PATH}", "-m1000", "--discard-dominated=1", "-W1"]
    )
    assert rc == 0
    assert out == ("-1 $i2 \n-1 and($i1 $i2) \n")


def test_moses_exec_reproduces_classic_diversity_pressure_oracle():
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses
    -i tests/fixtures/table_data/dataset.csv -m1000 --diversity-pressure=0.1
    -W1 ) -- diversity's tie-break reorders two equal-score/equal-complexity
    candidates (`and($i1 !$i2)`/`and(!$i1 !$i2)`) relative to the
    diversity-off default. Before this task, this raised `NotImplementedError`
    at parse time.
    """
    rc, out = _run(
        ["moses", f"-i{_COMMENTED_DATASET_PATH}", "-m1000", "--diversity-pressure=0.1", "-W1"]
    )
    assert rc == 0
    assert out == (
        "-1 $i2 \n"
        "-1 and($i1 $i2) \n"
        "-2 $i1 \n"
        "-2 and(!$i1 $i2) \n"
        "-2 or(!$i1 $i2) \n"
        "-2 or($i1 $i2) \n"
        "-3 true \n"
        "-3 !$i1 \n"
        "-3 and($i1 !$i2) \n"
        "-3 and(!$i1 !$i2) \n"
    )


def test_moses_exec_discard_dominated_and_diversity_pressure_defaults_still_match_default_oracle():
    """The exit criterion's other half for both un-gated flags: leaving
    both at their (shared, off) default on the SAME dataset must still
    give the ordinary un-pruned, un-reordered 10-line default result
    (captured the same way: `-i tests/fixtures/table_data/dataset.csv
    -m1000 -W1`, no --discard-dominated/--diversity-pressure at all).
    """
    rc, out = _run(["moses", f"-i{_COMMENTED_DATASET_PATH}", "-m1000", "-W1"])
    assert rc == 0
    assert out == (
        "-1 $i2 \n"
        "-1 and($i1 $i2) \n"
        "-2 $i1 \n"
        "-2 and(!$i1 $i2) \n"
        "-2 or(!$i1 $i2) \n"
        "-2 or($i1 $i2) \n"
        "-3 true \n"
        "-3 !$i1 \n"
        "-3 and(!$i1 !$i2) \n"
        "-3 and($i1 !$i2) \n"
    )


def test_moses_exec_opt_algo_bad_name_raises_immediately_at_parse_time():
    """[opt-algo comment fix] An unrecognised `--opt-algo` name raises
    `NotImplementedError` from `problem_params.parse_options` itself,
    BEFORE any table I/O or evaluation -- proven end to end through the
    real CLI (no `-i`/dataset needed at all: the raise fires before the
    table would even be opened), not merely at the unit level.

    This is the corrected version of `problem_params.py`'s own comment on
    this gate: an EARLIER version of that comment claimed classic's
    dispatch-time check (`moses_main.h:300-306`) was the ONLY place this
    got validated, and that this port "raises there too" instead of here
    -- false, since the very next line already raised here. The gate
    itself is intentionally KEPT (not moved to the dispatch) as a
    disclosed, deliberate timing divergence -- fail immediately rather
    than deep in a live run, the same choice already made for
    `-t/--output-bscore`/`--output-deme-id`. `moses_main.py`'s own
    dispatch-time check (`test_moses_main.py`'s own test) stays correct
    for a caller that builds/mutates a `problem_params` directly,
    bypassing `parse_options` -- it is simply unreachable from THIS entry
    point, which always goes through `parse_options` first.
    """
    with pytest.raises(NotImplementedError, match="opt-algo 'zz' is not a ported optimizer"):
        moses_exec(["moses", "--opt-algo=zz"])


# ---------------------------------------------------------------------------
# A second, independent oracle that does NOT need table_problems.py: the
# `-Hcp` demo problem (already-committed, S13 P2 track). Exercises the
# SAME moses_exec.py dispatch/printer wiring through a different problem
# registry (register_demo_problems, not register_table_problems).
# ---------------------------------------------------------------------------


def test_moses_exec_reproduces_classic_cp_demo_oracle():
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses -Hcp -y"and(\\$1 \\$2)" -r1 ).
    """
    rc, out = _run(["moses", "-Hcp", "-yand($1 $2)", "-r1"])
    assert rc == 0
    assert out == ("0 and($1 $2) \n-1 $1 \n-1 $2 \n-3 true \n")


def test_moses_exec_wires_demo_params_problem_size(dataset_csv):
    """`-k`/`--problem-size` (`demo_params`, demo_problems.py) is NOT an
    `option_base` in C++ and is not registered by `problem_params.py` or
    `table_problems.py` -- `moses_exec.py`'s own `_add_demo_options` is
    the only place it is wired. `-Hmux -k1` needs a 3-bit input (a mux of
    1 selector bit multiplexes 2^1=2 data bits, `get_arity(1) = 1 + 2 =
    3`) -- an arity mismatch here would raise well before any score is
    printed, so a non-empty, successful run proves the wiring reached
    `demo_params.problem_size` (default 5 would instead build an
    arity-`5 + 2**5 = 37` truth table and very likely not finish inside
    this test's own bound).
    """
    rc, out = _run(["moses", "-Hmux", "-k1", "-r1", "-m200"])
    assert rc == 0
    assert out != ""


# ---------------------------------------------------------------------------
# Dispatch/error-path unit tests -- moses_exec.cc:38-63's own body.
# ---------------------------------------------------------------------------


def test_moses_exec_unknown_problem_returns_1_and_writes_stderr(capsys):
    """`if (probm) {...} logger().error()...; std::cerr << ...; return 1;`
    (moses_exec.cc:53-62) -- the ONE genuine in-band `return` in
    `moses_exec`'s whole body (see module docstring for why every other
    failure mode is left to propagate as an exception instead).
    """
    rc = moses_exec(["moses", "-Hbogus-problem-name"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err == 'problem type "bogus-problem-name" is currently unsupported.\n'


@pytest.mark.parametrize("name", ["ann-it"])
def test_moses_exec_unregistered_stub_problem_raises(name):
    """The problems `table_problems.py` still leaves as `problem.py`'s own
    raising `_unported_table_problem` stub. This test was `-H cluster`
    alone until `cluster_table_problem` was wired for real
    (table_problems.py); `-H cluster` now RUNS, and its oracle is
    `test_moses_exec_reproduces_classic_cluster_*` below. `-H pre-conj`
    was the next to leave this list -- its oracle is
    `test_moses_exec_reproduces_classic_pre_conj_oracle` below -- and
    `-H ip` the most recent, with `tests/moses/main/test_ip_problem_cli.py`
    as its oracle. `ann-it` is the ANN domain, excluded project-wide
    rather than unscheduled, so this list is expected to stay at one.
    """
    with pytest.raises(NotImplementedError, match=name):
        moses_exec(["moses", f"-H{name}"])


def test_moses_exec_ip_is_wired_not_stubbed():
    """The complement of the case above, kept as its own test so the
    parametrize list shrinking to one entry cannot quietly become a
    parametrize over NOTHING (`empty_parameter_set_mark = "fail_at_collect"`
    guards that, but only for an EMPTY list).

    `-H ip` with no `-i` must now fail the way every other wired table
    problem does -- `common_setup`'s "no input data file" ValueError
    (table-problems.cc:101-106) -- NOT with `NotImplementedError`.
    """
    with pytest.raises(ValueError, match="no input data file has been specified"):
        moses_exec(["moses", "-Hip"])


def test_moses_exec_version_flag_exits_zero():
    """`--version` (`problem_params.parse_options`, problem-params.cc:
    2059-2067 / problem_params.py) raises `SystemExit(0)` before any
    other option is even read -- moses_exec.py does not catch it (module
    docstring: an uncaught process-exit is the correct analogue of C++'s
    own `exit(0)`, not something to convert into a return code).
    """
    with pytest.raises(SystemExit) as exc_info:
        moses_exec(["moses", "--version"])
    assert exc_info.value.code == 0


def test_moses_exec_accepts_a_plain_string_argv():
    """The third C++ overload, `moses_exec(const string&)`
    (moses_exec.cc:76-81): splits on whitespace, then re-dispatches --
    ported here as the same function accepting either a `Sequence[str]`
    or a bare `str` (see module docstring's "three overloads, merged"
    section). Uses `-Hpa -k1` (no space-containing option value) rather
    than `-Hcp -y"and($1 $2)"`: naive whitespace splitting (`boost::
    is_space()` in C++, bare `str.split()` here) breaks a space-containing
    combo-program string into two tokens in BOTH languages alike -- a
    faithful, not a divergent, limitation of this particular overload,
    not this test's concern.
    """
    rc, out = _run("moses -Hpa -k1 -r1 -m50")
    assert rc == 0
    assert out == "0 !$1 \n-1 true \n-2 $1 \n"


# ---------------------------------------------------------------------------
# -t/--output-bscore and --output-deme-id both used to fail here: first by
# crashing deep inside a real run (metapopulation.merge_demes /
# ostream_scored_combo_tree, several stack frames past `-m` evaluations
# already having executed), then -- once problem_params.py gated them at
# option-parse time -- by refusing a command line classic runs to
# completion. Both are real now and both are pinned below against the
# compiled binary, byte for byte.
# ---------------------------------------------------------------------------

# -t/--output-bscore's oracles come in three shapes, deliberately, because
# the flag was the FIRST consumer able to see a CTable ordering defect that
# lived in a file the -t slice did not own. A `behavioral_score` over a
# BOOLEAN table has one element per CTable row; C++'s CTable is a
# `std::map<multi_type_seq, TimedCounter>` (table.h:550) and iterates in
# ascending key order, while this port's was a plain `dict` filled in file
# order. Nothing before `-t` could see the difference: every other consumer
# of a bscore (the composite score's sum, `remove_dominated`'s element-wise
# compare, the diversity distances) is permutation-invariant so long as
# every candidate uses the SAME permutation, which they do.
#
# CLOSED: `CTable` now iterates in `std::map` key order
# (`pymoses/comboreduct/table/table.py`, `ctable_row_key` / `CTable._order`;
# regression coverage in `tests/comboreduct/table/test_ctable_order.py`), so
# `test_moses_exec_output_bscore_ctable_row_order_matches_classic` below is
# a plain oracle again rather than a strict xfail. The three shapes are kept
# as they were: they cover a table already in map order, a table that is
# not, and a contin table whose bscore never goes through a CTable at all.

# A boolean table whose distinct input rows happen to appear in the file in
# exactly `std::map` order already, so file order and map order coincide
# here whatever CTable does: combo booleans are `id::logical_true` = 1 and
# `id::logical_false` = 2 (vertex.h:69), so ascending key order is
# (T,T) < (T,F) < (F,T) < (F,F).
_SORTED_BOOL_CSV = """res,arg1,arg2
1,1,1
0,1,1
1,1,0
0,1,0
0,0,1
1,0,1
0,0,0
0,0,0
"""


@pytest.fixture
def sorted_bool_csv(tmp_path):
    path = tmp_path / "sorted_bool.csv"
    path.write_text(_SORTED_BOOL_CSV)
    return str(path)


# A contin table. `contin_bscore` (bscores.h:181) is built from an
# `ITable`/`OTable` pair, NOT a CTable, so its element order is raw row
# order in classic and here alike -- immune to the defect above by
# construction, which is why this one is a plain oracle and not an xfail.
_TINY_CONTIN_CSV = """y,x1
0.1,1
0.3,2
0.7,3
1.3,4
"""


@pytest.fixture
def tiny_contin_csv(tmp_path):
    path = tmp_path / "tiny_contin.csv"
    path.write_text(_TINY_CONTIN_CSV)
    return str(path)


def test_moses_exec_output_bscore_matches_classic(sorted_bool_csv):
    """`-t`/`--output-bscore`: classic exits 0 and appends
    `" behavioral score: [...]"` to every result line. This port refused to
    start at all until the S13-CLI `-t` slice -- `metapopulation.
    merge_demes`'s `keep_bscore` arm (merging.cc:258-296) raised, and
    `problem_params.parse_options` raised earlier still to make that
    failure clean and immediate.

    Captured: ( ulimit -v 2097152; timeout 300
    $CLASSIC/build/moses/moses/main/moses -i<sorted_bool.csv> -m1000 -t1 )

    The TWO SPACES before "behavioral" are classic's, not a typo:
    `ostream_combo_tree` ends every tree with an unconditional trailing
    space (iostream_combo.cc:478) and `ostream_scored_combo_tree` writes
    another one before the bscore (types.cc:197). A port that emitted one
    space would differ here and nowhere else.

    `-0` is likewise classic's: the elements are `score_t` (binary32) and
    an exactly-correct boolean row scores negative zero, which prints with
    its sign.
    """
    rc, out = _run(["moses", f"-i{sorted_bool_csv}", "-m1000", "-t1"])
    assert rc == 0
    assert out == (
        "-3 $1  behavioral score: [-1 -1 -1 -0]\n"
        "-3 $2  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and(!$1 $2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and($1 !$2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and($1 $2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 or($1 $2)  behavioral score: [-1 -1 -1 -0]\n"
        "-5 true  behavioral score: [-1 -1 -1 -2]\n"
    )


def test_moses_exec_output_bscore_with_labels_matches_classic(sorted_bool_csv):
    """Same run with `-W1`: labels substitute into the tree and NOT into the
    behavioral score, which has no argument positions to substitute."""
    rc, out = _run(["moses", f"-i{sorted_bool_csv}", "-m1000", "-t1", "-W1"])
    assert rc == 0
    assert out == (
        "-3 $arg1  behavioral score: [-1 -1 -1 -0]\n"
        "-3 $arg2  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and(!$arg1 $arg2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and($arg1 !$arg2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 and($arg1 $arg2)  behavioral score: [-1 -1 -1 -0]\n"
        "-3 or($arg1 $arg2)  behavioral score: [-1 -1 -1 -0]\n"
        "-5 true  behavioral score: [-1 -1 -1 -2]\n"
    )


def test_moses_exec_output_bscore_contin_matches_classic(tiny_contin_csv):
    """The contin half. This is the oracle that actually pins the NUMERIC
    formatting of a `behavioral_score`: every element is a `score_t`
    (binary32) rendered through the stream `metapop_printer` set to
    `setprecision(io_score_precision)` = 18 (moses_main.h:110), so
    `-0.16f` prints as `-0.159999996423721313` and not `-0.16`.

    Captured: ( ulimit -v 2097152; timeout 300
    $CLASSIC/build/moses/moses/main/moses -i<tiny_contin.csv> -uy -m300
    -t1 -c3 )
    """
    rc, out = _run(["moses", f"-i{tiny_contin_csv}", "-uy", "-m300", "-t1", "-c3"])
    assert rc == 0
    assert out == (
        "-0.879999995231628418 0.5  behavioral score:"
        " [-0.159999996423721313 -0.0399999991059303284"
        " -0.0399999991059303284 -0.639999985694885254]\n"
        "-0.930000007152557373 0.75  behavioral score:"
        " [-0.422500014305114746 -0.202500000596046448"
        " -0.00249999994412064552 -0.302500009536743164]\n"
        "-1.32999992370605469 0.25  behavioral score:"
        " [-0.0225000008940696716 -0.00249999994412064552"
        " -0.202500000596046448 -1.10249996185302734]\n"
    )


def test_moses_exec_output_bscore_ctable_row_order_matches_classic(dataset_csv):
    """Captured: ( ulimit -v 2097152; timeout 300
    $CLASSIC/build/moses/moses/main/moses -i$CLASSIC/tests/moses/dataset.csv
    -m1000 -t1 )

    `dataset.csv`'s three distinct input rows appear in the file as
    (F,T), (T,F), (T,T) and in classic's CTable as (T,T), (T,F), (F,T) --
    the exact reverse. Trees and scores agreed even while the port was
    wrong, because the composite score is a SUM; only the bscore VECTOR
    ORDER differed, which is why this run and not
    `test_moses_exec_output_bscore_matches_classic` was the one that
    failed. Measured scope of the exposure, over `-t1 -m500` on every
    boolean/contin fixture in tests/fixtures/table_data plus
    $CLASSIC/tests/moses/dataset.csv: every boolean table diverged except
    ones already in sorted order; every contin table agreed (contin
    bscores are ITable-ordered, see above).

    This was a `@pytest.mark.xfail(strict=True)` until `CTable` was made to
    iterate in `std::map` key order; it is the end-to-end half of
    `tests/comboreduct/table/test_ctable_order.py`, which pins the rule
    itself.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-t1"])
    assert rc == 0
    assert out == (
        "-3 and($1 $2)  behavioral score: [-1 -1 -1]\n"
        "-4 $1  behavioral score: [-1 -2 -1]\n"
        "-4 $2  behavioral score: [-1 -1 -2]\n"
        "-4 and(!$1 !$2)  behavioral score: [-2 -1 -1]\n"
        "-4 or(!$1 $2)  behavioral score: [-1 -1 -2]\n"
        "-4 or($1 !$2)  behavioral score: [-1 -2 -1]\n"
        "-5 true  behavioral score: [-1 -2 -2]\n"
        "-5 !$1  behavioral score: [-2 -1 -2]\n"
        "-5 !$2  behavioral score: [-2 -2 -1]\n"
    )


# ---------------------------------------------------------------------------
# Three flags classic accepts and this port refused, all found by the
# value-aware sweep that replaced the fixed-candidate one.
# ---------------------------------------------------------------------------

# Captured: ( ulimit -v 2097152; timeout 200
#   $CLASSIC/build/moses/moses/main/moses -Hpa -k4 -m5000
#   --hc-fraction-of-nn=0 )
_EXPECTED_PA_K4_FRAC0 = (
    "-4 or(and(or($1 $3) !$2 $4) and(or($1 $3) $2 !$4) and($1 $3)) \n"
    "-4 or(and(or(and(!$2 $4) $1) $3) and(or($1 $3) $2 !$4) and($1 !$2 $4)) \n"
    "-4 or(and(or(and(!$2 $4) $3) $1) and(or($1 $3) $2 !$4) and(!$2 $3 $4)) \n"
    "-5 or(and(or($1 $3) !$2 $4) and($1 $3) and($2 !$4)) \n"
    "-5 or(and(or($1 $3) $2 !$4) and(or($2 !$4) $1 $3)) \n"
    "-5 or(and(or($1 $3) $2 !$4) and($1 $3) and(!$2 $4)) \n"
    "-5 or(and(or(and(!$2 $4) $3) $1) and(or($1 $3) $2 !$4)) \n"
    "-4 or(and(or(and(!$2 $4) $1) $3) and(or($1 $3) $2 !$4) and($1 !$2 !$3 $4)) \n"
    "-4 or(and(or(and(!$2 $4) $3) $1) and(or($1 $3) $2 !$4) and(!$1 !$2 $3 $4)) \n"
    "-4 or(and(or($1 $3) !$2 $4) and(or($1 $3) $2 !$4) and(or(!$2 $4) $1 $3)) \n"
)


def test_hc_fraction_of_nn_zero_matches_classic():
    """`--hc-fraction-of-nn=0`: classic exits 0; this port raised
    `ValueError: fraction_of_nn must be positive`.

    The port's check was not an invention -- `OC_ASSERT(0.0 <
    fraction_of_nn)` is real (hill-climbing.h:51) -- it was in the wrong
    place. That assert lives in `hc_parameters`' CONSTRUCTOR, and
    `problem-params.cc:2360-2368` never calls the constructor with CLI
    values: `hc_params` is a default-constructed member and every field is
    assigned afterwards. So the assert fires once, against the 2.0 default,
    and a CLI value of 0 (or -1) sails past it.

    `-Hpa -k4 -m5000` is used rather than a table because this is also the
    proof the value REACHES the optimizer: measured on that run, classic
    prints three different candidate lists for 0, 0.5 and 2
    (`number_of_new_instances *= hc_params.fraction_of_nn`,
    hill-climbing.cc:531). A test on a run where 0 changed nothing would
    pass against a port that silently ignored the flag.
    """
    rc, out = _run(["moses", "-Hpa", "-k4", "-m5000", "--hc-fraction-of-nn=0"])
    assert rc == 0
    assert out == _EXPECTED_PA_K4_FRAC0
    rc2, out2 = _run(["moses", "-Hpa", "-k4", "-m5000"])
    assert rc2 == 0
    assert out2 != out  # the flag is load-bearing on this run, not inert


_EXPECTED_PA_K4_FRAC_NEG1 = (
    "-4 or(and(or(and(or($1 !$3) $4) !$2) or(and(!$1 $2) $3)) and(or(!$2 $3) !$1 !$4)) \n"
    "-5 or(and(or(and($1 $4) !$2) $3) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or(!$1 !$4) $2) $3) or(and(!$3 $4) $1 !$2)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or($1 !$2 !$3) $4) !$2) or(and(!$1 $2) $3)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or($1 !$3) $4) !$2) or(and(!$1 $2 $4) $3)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or(!$1 $3) $2) $3) or(and(or($1 !$3) $4) !$2)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or(!$1 !$4) $2) $3) or(and(or($1 !$3) $4) !$2)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or($1 !$3) or(!$2 $4)) !$2) or(and(!$1 $2) $3)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or($1 !$3) $4) and(!$2 $3)) or(and(!$1 $2) $3)) and(or(!$2 $3) !$1 !$4)) \n"
    "-4 or(and(or(and(or($1 !$3) $4) !$2) or(and(or($2 $3) !$1) $3)) and(or(!$2 $3) !$1 !$4)) \n"
)


def test_hc_fraction_of_nn_negative_is_accepted_like_classic():
    """`=-1` is accepted (classic exits 0), and is NOT equivalent to `0`.

    This test used to assert `out == _EXPECTED_PA_K4_FRAC0`, on the stated
    theory that "<= 0 all take the same zeroed path". **Measured false:**
    classic's own output at `-1` differs from its own output at `0`.

    Why: `number_of_new_instances` is a `size_t` and `fraction_of_nn` a
    `double`, so `*=` (hill-climbing.cc:531) promotes, multiplies, and
    converts BACK to `size_t` each iteration. At `0` that yields 0; at a
    negative fraction the first multiply wraps to a huge unsigned value and
    the second overflows int64, giving x86-64's "integer indefinite"
    0x8000000000000000. A plain Python `int()`, which keeps the sign, gave
    neither -- see `_double_to_size_t`.

    DO NOT FIX by restoring a positivity check: `OC_ASSERT(0.0 <
    fraction_of_nn)` (hill-climbing.h:51) is in the CONSTRUCTOR, which the
    CLI bypasses by assigning the field afterwards, so classic really does
    run this command line.
    """
    rc, out = _run(["moses", "-Hpa", "-k4", "-m5000", "--hc-fraction-of-nn=-1"])
    assert rc == 0
    assert out == _EXPECTED_PA_K4_FRAC_NEG1
    assert out != _EXPECTED_PA_K4_FRAC0, "negative is not the same path as zero"


def test_hc_parameters_constructor_still_asserts():
    """The other half of the same fix: `hc_parameters.__init__` KEEPS
    `OC_ASSERT(0.0 < fraction_of_nn)`, because it is faithful to the
    constructor. Only the CLI stopped going through it."""
    from pymoses.moses.moses.optimization.hill_climbing import hc_parameters

    with pytest.raises(ValueError, match="fraction_of_nn"):
        hc_parameters(fraction_of_nn=0.0)


def test_log_file_that_cannot_be_unlinked_matches_classic(dataset_csv):
    """`-f /dev/null`: classic exits 0, this port raised `PermissionError:
    [Errno 13] Permission denied: '/dev/null'` before any search ran.

    C's `remove()` returns -1 on ANY failure and classic discards the
    return value (problem-params.cc:2091), so an unlink that fails for
    ANY reason is ignored -- not just the missing-file case this port was
    suppressing. /dev/null is the reproducer because it is the one such
    target that exists on every machine; the same shape covers a log file
    in a read-only or sticky directory.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-f", "/dev/null"])
    assert rc == 0
    assert out == _EXPECTED_IT_NO_LABELS


@pytest.mark.parametrize(
    "argv,field,expected",
    [
        (["-K", "2"], "ip_kld_weight", 2.0),
        (["--ip_kld_weight=2.5"], "ip_kld_weight", 2.5),
        (["-J", "0.5"], "ip_skewness_weight", 0.5),
        (["--ip_skewness_weight=3"], "ip_skewness_weight", 3.0),
        (["-U", "3"], "ip_stdU_weight", 3.0),
        (["--ip_stdU_weight=0"], "ip_stdU_weight", 0.0),
        (["-X", "-1"], "ip_skew_U_weight", -1.0),
        (["--ip_skew_U_weight=1.25"], "ip_skew_U_weight", 1.25),
    ],
)
def test_ip_problem_params_parse_and_store(argv, field, expected):
    """The four `-H ip` weights (table-problems.cc:210-223) were absent
    from this port's parser entirely: classic exits 0 on each, the port
    exited 2 with argparse's "unrecognized arguments". Both spellings of
    each are exercised, because the short forms (-K/-J/-U/-X) come from
    boost's `"name,X"` syntax and are easy to drop.

    Deliberately NOT gated on `-H ip` being runnable. Classic parses and
    stores them regardless of `-H`, and its own `ip_problem` aborts for a
    reason of its own (table-problems.cc:236) on any table that is not
    contin-output/boolean-input. Gating the weights would reject command
    lines classic accepts.
    """
    from pymoses.moses.main.table_problems import ip_problem_params

    parser = argparse.ArgumentParser(add_help=False)
    ippp = ip_problem_params()
    ippp.add_options(parser)
    ippp.parse_options(parser.parse_args(argv))
    assert getattr(ippp, field) == expected


def test_ip_problem_params_defaults(dataset_csv):
    """All four default to 1.0 (`->default_value(1.0)`,
    table-problems.cc:211/215/219/223) and passing them changes nothing
    about an ordinary `-H it` run -- measured against classic, which also
    prints the same output with and without them."""
    from pymoses.moses.main.table_problems import ip_problem_params

    ippp = ip_problem_params()
    assert (
        ippp.ip_kld_weight,
        ippp.ip_skewness_weight,
        ippp.ip_stdU_weight,
        ippp.ip_skew_U_weight,
    ) == (1.0, 1.0, 1.0, 1.0)

    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "-K2", "-J0.5", "-U3", "-X-1"])
    assert rc == 0
    assert out == _EXPECTED_IT_NO_LABELS


# ---------------------------------------------------------------------------
# --help / no arguments at all / the option surface itself.
#
# Classic's `option_manager::parse_options` (problem.cc:70-73) prints the
# whole options_description and exits ONE -- not zero -- and takes the same
# branch when argc == 1. Both were divergences here: argparse's built-in
# help action exits 0, and a bare `moses` ran the default problem and died
# on the missing -i.
#
# The help TEXT is still argparse's, not boost's (70462 bytes vs ~11KB);
# see the DIVERGENCE marker at moses_exec.py's own help block. What IS
# pinned byte-exactly is the option SURFACE, below, which is the part a
# script or a user can actually trip over.
# ---------------------------------------------------------------------------

# Every option spelling -- long and short -- that
# $CLASSIC/build/moses/moses/main/moses --help lists. Extracted from that
# output with `^  (?:(-[A-Za-z0-9]) \[ )?(--[-\w.]+)`, which is exactly
# boost's own `format_name()` shape ("  -h [ --help ]" / "  --version").
# 188 spellings over 139 options.
#
# Four of these -- -K/--ip_kld_weight, -J/--ip_skewness_weight,
# -U/--ip_stdU_weight, -X/--ip_skew_U_weight -- were MISSING from this port
# entirely until the S13-CLI option-surface slice; classic exits 0 on each
# and the port exited 2 with "unrecognized arguments". They were the only
# four missing, and there were no port-only extras, so with them added the
# two surfaces are exactly equal.
_CLASSIC_OPTION_SPELLINGS = frozenset(
    {
        "--alpha",
        "--balance",
        "--boost",
        "--boost-bias",
        "--boost-exact",
        "--boost-expalpha",
        "--boost-promote",
        "--cache-size",
        "--cap-coef",
        "--combo-program",
        "--complexity-ratio",
        "--complexity-temperature",
        "--contin-depth",
        "--discard-dominated",
        "--discretize-threshold",
        "--diversity-autoscale",
        "--diversity-dst",
        "--diversity-dst2dp",
        "--diversity-exponent",
        "--diversity-normalize",
        "--diversity-p-norm",
        "--diversity-pressure",
        "--enable-fs",
        "--exemplar",
        "--fs-algo",
        "--fs-demes",
        "--fs-diversity-cap",
        "--fs-diversity-interaction",
        "--fs-diversity-jaccard",
        "--fs-diversity-pressure",
        "--fs-enforce-features-filename",
        "--fs-exp-distrib",
        "--fs-focus",
        "--fs-hc-crossover",
        "--fs-hc-crossover-min-neighbors",
        "--fs-hc-crossover-pop-size",
        "--fs-hc-fraction-of-remaining",
        "--fs-hc-max-evals",
        "--fs-hc-max-score",
        "--fs-hc-widen-search",
        "--fs-inc-interaction-terms",
        "--fs-inc-redundant-intensity",
        "--fs-inc-target-size-epsilon",
        "--fs-mi-penalty",
        "--fs-pre-max-activation",
        "--fs-pre-min-activation",
        "--fs-pre-penalty",
        "--fs-pre-positive",
        "--fs-prune-exemplar",
        "--fs-scorer",
        "--fs-seed",
        "--fs-smd-top-size",
        "--fs-subsampling-by-time",
        "--fs-subsampling-ratio",
        "--fs-target-size",
        "--fs-threshold",
        "--gen-best-tree",
        "--hc-allow-resize-deme",
        "--hc-crossover",
        "--hc-crossover-min-neighbors",
        "--hc-crossover-pop-size",
        "--hc-fraction-of-nn",
        "--hc-max-nn-evals",
        "--hc-resize-to-fit-ram",
        "--hc-single-step",
        "--hc-widen-search",
        "--help",
        "--ignore-feature",
        "--ignore-operator",
        "--include-only-operator",
        "--input-file",
        "--ip_kld_weight",
        "--ip_skew_U_weight",
        "--ip_skewness_weight",
        "--ip_stdU_weight",
        "--it-abs-err",
        "--jobs",
        "--linear-regression",
        "--log-file",
        "--log-file-dep-opt",
        "--log-level",
        "--logical-perm-ratio",
        "--max-candidates-per-deme",
        "--max-dist",
        "--max-evals",
        "--max-gens",
        "--max-rand-input",
        "--max-score",
        "--max-time",
        "--min-pool",
        "--min-rand-input",
        "--noise",
        "--nsamples",
        "--opt-algo",
        "--output-bscore",
        "--output-cscore",
        "--output-deme-id",
        "--output-eval-number",
        "--output-file",
        "--output-format",
        "--output-only-best",
        "--output-score",
        "--output-with-labels",
        "--pop-size-ratio",
        "--pre-positive",
        "--problem",
        "--problem-size",
        "--ps-max-particles",
        "--random-seed",
        "--reduce-all",
        "--reduct-candidate-effort",
        "--reduct-knob-building-effort",
        "--result-count",
        "--revisit",
        "--score-weight",
        "--ss-by-time",
        "--ss-contiguous-time",
        "--ss-low-dev-pressure",
        "--ss-n-best-bfdemes",
        "--ss-n-subsample-demes",
        "--ss-n-subsample-fitnesses",
        "--ss-n-top-candidates",
        "--ss-n-tuples",
        "--ss-std-dev-threshold",
        "--ss-tanimoto-geometric-mean-threshold",
        "--ss-tanimoto-geometric-mean-weight",
        "--ss-tanimoto-max-threshold",
        "--ss-tanimoto-max-weight",
        "--ss-tanimoto-mean-threshold",
        "--ss-tanimoto-mean-weight",
        "--target-feature",
        "--time-bscore",
        "--time-bscore-granularity",
        "--time-dispersion-exponent",
        "--time-dispersion-pressure",
        "--timestamp-feature",
        "--version",
        "--weighted-accuracy",
        "--well-enough",
        "-A",
        "-B",
        "-C",
        "-D",
        "-E",
        "-F",
        "-G",
        "-H",
        "-J",
        "-K",
        "-L",
        "-M",
        "-N",
        "-P",
        "-Q",
        "-R",
        "-S",
        "-T",
        "-U",
        "-V",
        "-W",
        "-X",
        "-Y",
        "-Z",
        "-a",
        "-c",
        "-d",
        "-e",
        "-f",
        "-g",
        "-h",
        "-i",
        "-j",
        "-k",
        "-l",
        "-m",
        "-n",
        "-o",
        "-p",
        "-q",
        "-r",
        "-s",
        "-t",
        "-u",
        "-v",
        "-w",
        "-x",
        "-y",
        "-z",
    }
)


def test_help_lists_exactly_classics_options():
    """The option surface, diffed against the compiled binary's own --help.

    This is the guard that would have caught the four missing `-H ip`
    weights on the day they went missing, and it is a MUCH stronger claim
    than "--help exits 1": a flag classic accepts and this port rejects is
    a hard failure for any script driving moses, whatever the help text
    says. Asserted in both directions -- a port-only flag is just as much a
    divergence as a missing one.
    """
    import argparse

    from pymoses.moses.main.demo_problems import register_demo_problems
    from pymoses.moses.main.moses_exec import _add_demo_options
    from pymoses.moses.main.problem import problem_manager
    from pymoses.moses.main.problem_params import problem_params
    from pymoses.moses.main.table_problems import ip_problem_params, register_table_problems

    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    register_demo_problems(pmr)
    parser = argparse.ArgumentParser(prog="moses", add_help=False)
    parser.add_argument("-h", "--help", dest="help", action="store_true", default=False)
    problem_params().add_options(parser)
    tpp.add_options(parser)
    ip_problem_params().add_options(parser)
    _add_demo_options(parser)

    port = {opt for action in parser._actions for opt in action.option_strings}
    assert sorted(port - _CLASSIC_OPTION_SPELLINGS) == []
    assert sorted(_CLASSIC_OPTION_SPELLINGS - port) == []


def test_help_exits_one_like_classic():
    """DO NOT FIX: `moses --help` exits 1 in classic, not 0
    (problem.cc:72's `exit(1)` -- measured, `echo $?` prints 1). argparse's
    built-in help action exits 0, which is why this port registers `-h` as
    an ordinary flag and acts on it after parsing instead."""
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        moses_exec(["moses", "--help"])
    assert exc.value.code == 1
    assert buf.getvalue().startswith("usage: moses")


def test_no_arguments_prints_help_and_exits_one_like_classic():
    """`argc == 1` takes the same branch as `--help` (problem.cc:70).
    Measured: a bare `moses` prints the identical 70462 bytes classic's
    `--help` prints and exits 1. This port used to run the default `-H it`
    problem with no table and raise `ValueError: no input data file has
    been specified (option -i)`."""
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        moses_exec(["moses"])
    assert exc.value.code == 1
    assert buf.getvalue().startswith("usage: moses")
    assert buf.getvalue() == _help_text()


def _help_text() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        moses_exec(["moses", "-h"])
    return buf.getvalue()


def test_help_is_not_consulted_before_the_command_line_parses():
    """C++ checks `vm.count("help")` only AFTER `po::store`/`po::notify`
    (problem.cc:63-70), so an unparseable command line is rejected first
    and `--help` does not rescue it. Pinned because moving the check
    earlier -- the obvious "simplification" -- would silently change
    which of the two failures a user sees."""
    with pytest.raises(SystemExit) as exc:
        moses_exec(["moses", "--help", "--no-such-option"])
    assert exc.value.code != 1  # argparse's own usage error, not the help exit


# ---------------------------------------------------------------------------
# --version. Classic prints two lines and exits 0; this port printed NOTHING
# and exited 0 until the S13-CLI --version slice, on the (half-right)
# grounds that `version_string` is a build-time define. See
# `moses_main.version_string` and `problem_params.parse_options` for which
# half was wrong. Missed by the 90-flag differential sweep because --version
# was on that sweep's explicit SKIP list, so it was never compared at all.
# ---------------------------------------------------------------------------


def test_moses_exec_version_matches_classic():
    """Captured: ( ulimit -v 2097152; timeout 60
    $CLASSIC/build/moses/moses/main/moses --version ) -- 35 bytes, rc 0,
    hexdumped rather than eyeballed because both quirks are invisible
    otherwise:

    * the version components carry LITERAL DOUBLE QUOTES (`"3"."7"."0"`,
      a double-stringify bug in classic's own CMake wiring -- see
      `moses_main.version_string`);
    * the second line begins with a literal TAB, not spaces.

    DO NOT FIX either.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        moses_exec(["moses", "--version"])
    assert exc.value.code == 0
    assert buf.getvalue() == 'moses "3"."7"."0"\n\tNo MPI support.\n'
    assert len(buf.getvalue().encode()) == 35


def test_moses_exec_version_is_checked_before_every_other_option():
    """`parse_options` tests `vm.count("version")` FIRST and `exit(0)`s
    before touching anything else (problem-params.cc:2059-2067), so
    `--version` wins over an option that would otherwise be rejected --
    here a log level classic itself `exit(1)`s on (:2093-2101). Pins the
    ORDER, which is the only thing that makes this observable."""
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        moses_exec(["moses", "--version", "-lBOGUS"])
    assert exc.value.code == 0
    assert buf.getvalue() == 'moses "3"."7"."0"\n\tNo MPI support.\n'


# ---------------------------------------------------------------------------
# --output-format=scheme. `parse_output_format` has always accepted it
# (iostream_combo.cc:258-259) and classic has always run it to completion,
# but every printing function in iostream_combo.py raised
# NotImplementedError until the S13-CLI scheme slice. The 90-flag
# differential sweep never tested it: its fixed candidate list
# (1/2/0.5/3/0/tanimoto/hc) contains no legal --output-format value, so
# the sweep recorded "classic rejects all candidates" and moved on.
# ---------------------------------------------------------------------------


def test_moses_exec_output_format_scheme_matches_classic(dataset_csv):
    """Captured: `moses -i<dataset.csv> -m1000 --output-format=scheme`.

    Three things this pins that combo format cannot:
    * parentheses are UNCONDITIONAL (iostream_combo.h:103-108) -- a leaf
      prints `(PredicateNode "$1")`, not `PredicateNode "$1"`;
    * `logical_true` is `PredicateNode "Top"`, not `true`;
    * negation WRAPS rather than abbreviating: `!$1` becomes
      `(NotLink (PredicateNode "$1"))`, built by wrapping the whole
      PredicateNode (iostream_combo.cc:410-421), so the `!`-prefix
      shorthand `ABBREVIATE_NEGATIVE_LITERAL` gives combo format never
      appears here.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "--output-format=scheme"])
    assert rc == 0
    assert out == (
        '-3 (AndLink (PredicateNode "$1") (PredicateNode "$2")) \n'
        '-4 (PredicateNode "$1") \n'
        '-4 (PredicateNode "$2") \n'
        '-4 (AndLink (NotLink (PredicateNode "$1")) (NotLink (PredicateNode "$2"))) \n'
        '-4 (OrLink (NotLink (PredicateNode "$1")) (PredicateNode "$2")) \n'
        '-4 (OrLink (PredicateNode "$1") (NotLink (PredicateNode "$2"))) \n'
        '-5 (PredicateNode "Top") \n'
        '-5 (NotLink (PredicateNode "$1")) \n'
        '-5 (NotLink (PredicateNode "$2")) \n'
    )


def test_moses_exec_output_format_scheme_with_labels_matches_classic(dataset_csv):
    """Same run with `-W1`. The label replaces the WHOLE `$n` token inside
    the quotes -- `PredicateNode "arg1"`, with no `$` -- because
    `ostream_argument`'s scheme branch writes the `"$"` prefix only on the
    label-less side of its own if/else (iostream_combo.cc:414-417). Combo
    format keeps the `$` in both cases (`$arg1`), so this asymmetry is
    only visible here."""
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "--output-format=scheme", "-W1"])
    assert rc == 0
    assert out == (
        '-3 (AndLink (PredicateNode "arg1") (PredicateNode "arg2")) \n'
        '-4 (PredicateNode "arg1") \n'
        '-4 (PredicateNode "arg2") \n'
        '-4 (AndLink (NotLink (PredicateNode "arg1")) (NotLink (PredicateNode "arg2"))) \n'
        '-4 (OrLink (NotLink (PredicateNode "arg1")) (PredicateNode "arg2")) \n'
        '-4 (OrLink (PredicateNode "arg1") (NotLink (PredicateNode "arg2"))) \n'
        '-5 (PredicateNode "Top") \n'
        '-5 (NotLink (PredicateNode "arg1")) \n'
        '-5 (NotLink (PredicateNode "arg2")) \n'
    )


def test_moses_exec_output_format_scheme_contin_matches_classic():
    """`-Hsr -k3 -m600 -r9 --output-format=scheme`: the contin half, which
    the boolean oracles above cannot reach.

    DO NOT FIX: scheme format has no Atomese name for ANY contin operator.
    `ostream_builtin`'s scheme switch (iostream_combo.cc:355-370) handles
    exactly six builtins and sends every other one to its own `default:`,
    whose text is `"Builtin: " << (unsigned) h << " unknown"` -- with a
    COLON the combo/python defaults do not have. So classic itself prints
    `Builtin: 7 unknown` for `*`, `6` for `+`, `8` for `/`, `11` for
    `sin`. This is classic's output, measured, and is reproduced rather
    than repaired.

    It also pins the sticky-precision rule: `0.543044125185779492` is the
    contin constant rendered at `io_score_precision` (18), because
    `metapop_printer` set that on the stream before the tree was written
    (moses_main.h:110). At the default stream precision of 6 it would be
    `0.543044`.
    """
    rc, out = _run(["moses", "-Hsr", "-k3", "-m600", "-r9", "--output-format=scheme"])
    assert rc == 0
    assert out == (
        '-1.14690244197845459 (PredicateNode "$1") \n'
        '-0.596547305583953857 (Builtin: 8 unknown (PredicateNode "$1")'
        " (Builtin: 6 unknown (1) (Builtin: 11 unknown"
        ' (Builtin: 7 unknown (-0.5) (PredicateNode "$1"))))) \n'
        '-1.09503006935119629 (Builtin: 8 unknown (PredicateNode "$1")'
        " (Builtin: 6 unknown (1) (Builtin: 11 unknown"
        ' (Builtin: 7 unknown (-1) (PredicateNode "$1"))))) \n'
        "-2.61014318466186523 (Builtin: 7 unknown (0.543044125185779492)"
        ' (PredicateNode "$1")) \n'
        '-2.90248513221740723 (Builtin: 7 unknown (0.5) (PredicateNode "$1")) \n'
        '-2.56437349319458008 (Builtin: 8 unknown (PredicateNode "$1")'
        ' (Builtin: 6 unknown (1) (Builtin: 7 unknown (-1) (PredicateNode "$1")))) \n'
        "-3.789306640625 (Builtin: 7 unknown (Builtin: 6 unknown (1)"
        ' (PredicateNode "$1")) (PredicateNode "$1")) \n'
        "-6.95522165298461914 (Builtin: 6 unknown (Builtin: 11 unknown"
        ' (PredicateNode "$1")) (PredicateNode "$1")) \n'
        "-8.24759960174560547 (0) \n"
        '-8.40433311462402344 (Builtin: 7 unknown (2) (PredicateNode "$1")) \n'
    )


# Captured: ( ulimit -v 2097152; timeout 300
#   $CLASSIC/build/moses/moses/main/moses -i<sorted_bool.csv> -m1000 -t1
#   --output-format=scheme )
_EXPECTED_BSCORE_SCHEME = (
    '-3 (PredicateNode "$1")  behavioral score: [-1 -1 -1 -0]\n'
    '-3 (PredicateNode "$2")  behavioral score: [-1 -1 -1 -0]\n'
    '-3 (AndLink (NotLink (PredicateNode "$1")) (PredicateNode "$2"))'
    "  behavioral score: [-1 -1 -1 -0]\n"
    '-3 (AndLink (PredicateNode "$1") (NotLink (PredicateNode "$2")))'
    "  behavioral score: [-1 -1 -1 -0]\n"
    '-3 (AndLink (PredicateNode "$1") (PredicateNode "$2"))'
    "  behavioral score: [-1 -1 -1 -0]\n"
    '-3 (OrLink (PredicateNode "$1") (PredicateNode "$2"))'
    "  behavioral score: [-1 -1 -1 -0]\n"
    '-5 (PredicateNode "Top")  behavioral score: [-1 -1 -1 -2]\n'
)


def test_moses_exec_output_bscore_scheme_combined(sorted_bool_csv):
    """The two flags this slice landed, together: `-t1
    --output-format=scheme`. The bscore is format-independent (it is a
    `behavioral_score`, not a tree), so it prints exactly as it does in
    combo format, after the scheme tree's own trailing space. Runs on the
    already-sorted fixture for the reason the -t block above states."""
    rc, out = _run(["moses", f"-i{sorted_bool_csv}", "-m1000", "-t1", "--output-format=scheme"])
    assert rc == 0
    assert out == _EXPECTED_BSCORE_SCHEME


def test_moses_exec_output_deme_id_matches_classic(dataset_csv):
    """`--output-deme-id` was gated until 2026-09-10 because
    `scored_combo_tree` carried no `demeID_t`. It does now, and
    `deme_to_trees` stamps each candidate with `deme.getID()`
    (merging.cc:110), which is where the id ORIGINATES.

    Captured: ( cd <scratch> && ulimit -v 2097152; timeout 300
    $CLASSIC/build/moses/moses/main/moses -i<dataset.csv> -m1000
    --output-deme-id=1 )

    The expected output carries BOTH `demeID: 1` and `demeID: 0` -- the
    initial exemplar comes from deme 0 -- so a port that stamped one
    constant id everywhere could not match it. That is what makes this an
    oracle rather than a smoke test.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000", "--output-deme-id=1"])
    assert rc == 0
    assert out == (
        "-3 and($1 $2)  demeID: 1\n"
        "-4 $1  demeID: 1\n"
        "-4 $2  demeID: 1\n"
        "-4 and(!$1 !$2)  demeID: 1\n"
        "-4 or(!$1 $2)  demeID: 1\n"
        "-4 or($1 !$2)  demeID: 1\n"
        "-5 true  demeID: 0\n"
        "-5 !$1  demeID: 1\n"
        "-5 !$2  demeID: 1\n"
    )


def test_moses_exec_output_bscore_and_deme_id_defaults_do_not_raise(dataset_csv):
    """The exit criterion's other half: leaving both flags at their
    (shared, `False`) default must still reproduce the ordinary oracle."""
    rc, out = _run(["moses", f"-i{dataset_csv}", "-m1000"])
    assert rc == 0
    assert out == _EXPECTED_IT_NO_LABELS


# ---------------------------------------------------------------------------
# Contin regression end to end.
#
# The whole contin path -- contin_bscore, contin_reduction, contin knob
# building, contin neighbourhood counting/enumeration and contin sampling
# -- was gated off behind four separate raises until 2026-09-09. These
# three cases are the proof that it now reproduces classic byte for byte,
# and they were chosen to discriminate rather than merely to pass:
#
#   * the default case is the one a plain `-H it` on a contin table takes;
#   * `--linear-regression=1` produces a DIFFERENT (shorter) result in
#     both implementations, which is what proves `linear_contin` is
#     actually threaded through to knob building. `deme_expander` used to
#     drop that argument on the floor, so the port silently ran with
#     `representation`'s constructor default (True) while the CLI default
#     is False -- a bug caught precisely because these two cases disagree
#     with each other;
#   * `--contin-depth=3` proves `set_depth` reaches `representation.py`'s
#     module static; that call was missing entirely and the flag was
#     gated instead.
#
# Captured: ( ulimit -v 2097152; timeout 300
#   $CLASSIC/build/moses/moses/main/moses
#   -i<contin_regression_sample.csv> -u y -m200 -W1 [flag] )
# ---------------------------------------------------------------------------

_CONTIN_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "table_data" / "contin_regression_sample.csv"
)

_EXPECTED_CONTIN_M200 = (
    "-0.122694917023181915 0 \n"
    "-0.987496495246887207 $x2 \n"
    "-1.9910207986831665 $x1 \n"
    "-2.11170101165771484 *(-1 $x2) \n"
    "-3.28426837921142578 *(-1 $x1) \n"
    "-4.70610570907592773 *(2 $x2) \n"
    "-15.9446239471435547 +(-1 $x2) \n"
    "-18.8003253936767578 1 \n"
    "-21.4450645446777344 -1 \n"
    "-26.0303688049316406 +(1 $x2) \n"
)

_EXPECTED_CONTIN_LINEAR = (
    "-0.122694917023181915 0 \n"
    "-0.19836975634098053 *(0.5 $x2) \n"
    "-0.987496495246887207 $x2 \n"
    "-0.760471999645233154 *(-0.5 $x2) \n"
    "-0.834387660026550293 *(0.5 +($x1 $x2)) \n"
    "-0.987496495246887207 *(0.5 +($x2 $x2)) \n"
    "-4.26824426651000977 *(0.5 +(-1 $x2)) \n"
    "-18.3381175994873047 +(-1 *(0.5 $x2)) \n"
)

_EXPECTED_CONTIN_DEPTH3 = (
    "-0.122694917023181915 0 \n"
    "-0.987496495246887207 $x2 \n"
    "-1.9910207986831665 $x1 \n"
    "-2.11170101165771484 *(-1 $x2) \n"
    "-3.28426837921142578 *(-1 $x1) \n"
    "-4.70610570907592773 *(2 $x2) \n"
    "-15.9446239471435547 +(-1 $x2) \n"
    "-18.8003253936767578 1 \n"
    "-21.4450645446777344 -1 \n"
    "-26.0303688049316406 +(1 $x2) \n"
)


def test_moses_exec_reproduces_classic_contin_regression_oracle():
    rc, out = _run(["moses", f"-i{_CONTIN_CSV_PATH}", "-u", "y", "-m200", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CONTIN_M200


def test_moses_exec_reproduces_classic_contin_linear_regression_oracle():
    """`--linear-regression=1` must give a DIFFERENT answer from the
    default run above -- that difference is the whole evidence that the
    flag reaches `build_knobs`'s `if (!_linear_contin)`.
    """
    rc, out = _run(
        [
            "moses",
            f"-i{_CONTIN_CSV_PATH}",
            "-u",
            "y",
            "-m200",
            "-W1",
            "--linear-regression=1",
        ]
    )
    assert rc == 0
    assert out == _EXPECTED_CONTIN_LINEAR
    assert out != _EXPECTED_CONTIN_M200


def test_moses_exec_reproduces_classic_contin_depth_oracle():
    rc, out = _run(
        ["moses", f"-i{_CONTIN_CSV_PATH}", "-u", "y", "-m200", "-W1", "--contin-depth=3"]
    )
    assert rc == 0
    assert out == _EXPECTED_CONTIN_DEPTH3


# ---------------------------------------------------------------------------
# -H sr (polynomial regression) and -H cp's contin arm, end to end.
#
# Both raised `NotImplementedError` until 2026-09-09, on a stated reason
# ("ITable/OTable/contin_bscore are S14 territory this port has not built,
# problem_params.contin_reduct is still None") that was false in every
# clause by the time it was read. These are the proof that they now
# reproduce classic.
#
# Every `_EXPECTED_*` below was captured by ACTUALLY RUNNING
#   ( ulimit -v 2097152; ulimit -c 0; timeout 300
#     $CLASSIC/build/moses/moses/main/moses <the flags in the test> )
# and is quoted here byte for byte, trailing spaces included. The full
# sweep behind them was wider than what runs here: 11 `-Hsr` and 14 `-Hcp`
# configurations all matched, varying -k, -m, -r, --nsamples,
# --linear-regression, --it-abs-err, -q/-w, -n, and the shape of -y.
# The subset kept here is the discriminating one -- see each test.
#
# Not kept as a test, deliberately: `-Hsr -k-1`. Classic HANGS on it
# (`timeout 20` -> rc 124), because `demo_params::problem_size` is
# `unsigned int` and `dorepeat` counts in `unsigned int`, so `-k-1` asks
# for 4294967295 iterations per table row. This port reproduces that
# rather than "fixing" it to zero iterations; `_dorepeat_count` in
# demo_problems.py is the extracted, testable half of it (see
# test_demo_problems.py), because a test that actually ran the loop would
# hang here too.
# ---------------------------------------------------------------------------

_EXPECTED_SR_K3 = (
    "-1.38283169269561768 *(+(2 $1) $1) \n"
    "-1.88396143913269043 *(+(2 *(0.75 $1)) $1) \n"
    "-2.50166654586791992 *(+(2 *(1.5 $1)) $1) \n"
    "-3.09212255477905273 *(+(2 *(0.5 $1)) $1) \n"
    "-3.78743267059326172 *(+(2 *(+(1.5 *(-1 $1)) $1)) $1) \n"
    "-4.80392360687255859 *(+(1 *(1.5 $1)) $1) \n"
    "-5.56489133834838867 *(+(1 *(0.75 $1)) $1) \n"
    "-7.62953901290893555 *(2 $1) \n"
    "-8.13655757904052734 *(1.5 $1) \n"
    "-8.31055450439453125 +(1 *(2 $1)) \n"
)

_EXPECTED_SR_K1 = (
    "0 $1 \n"
    "-8.09113025665283203 0 \n"
    "-26.7926368713378906 -1 \n"
    "-29.3896236419677734 1 \n"
    "-32.3645210266113281 *(-1 $1) \n"
)

_EXPECTED_SR_K0 = "0 0 \n"

_EXPECTED_SR_K3_R5 = (
    "-1.01685106754302979 *(1.5 +(1 $1) $1) \n"
    "-1.22611773014068604 *(+(1.5 *(0.75 $1)) $1) \n"
    "-1.69187688827514648 *(+(1.5 *(+(0.75 $1) $1)) $1) \n"
    "-2.06575965881347656 *(+(1.5 *(1.75 $1)) $1) \n"
    "-1.3310776948928833 *(+(1.5 *(+(0.75 *(-1 sin(*(-1 $1)))) $1)) $1) \n"
    "-5.85272073745727539 *(2 $1) \n"
    "-5.84626340866088867 *(+(2.5 *(0.75 $1)) $1) \n"
    "-6.46937274932861328 *(1.5 $1) \n"
    "-6.08637619018554688 *(2 +(1 $1) $1) \n"
    "-7.32205295562744141 *(+(1.5 *(+(0.75 *(-1 $1)) $1)) $1) \n"
)

_EXPECTED_SR_K3_NS7 = (
    "-0.736911177635192871 +(0.25 *(+(2 $1) $1)) \n"
    "-0.54436802864074707 +(0.25 *(+(2 *(0.5 $1)) $1)) \n"
    "-1.16134333610534668 +(0.5 *(2 $1)) \n"
    "-1.39297652244567871 +(0.75 *(2 $1)) \n"
    "-1.55767381191253662 +(0.5 *(1.5 $1)) \n"
    "-1.80471014976501465 +(0.25 *(2 $1)) \n"
    "-1.92749702930450439 +(0.5 *(+(2 $1) $1)) \n"
    "-2.08467888832092285 +(0.5 *(+(1.5 $1) $1)) \n"
    "-2.49960970878601074 +(1 *(2 $1)) \n"
    "-2.59896564483642578 +(0.5 *(2.5 $1)) \n"
)

_EXPECTED_SR_K3_LINREG = (
    "0 *(+(1 *(+(1 $1) $1)) $1) \n"
    "-1.38283169269561768 *(+(2 $1) $1) \n"
    "-1.38283169269561768 *(+(2 *(-1 $1) *(2 $1)) $1) \n"
    "-2.50166654586791992 *(+(2 *(1.5 $1)) $1) \n"
    "-7.62953901290893555 *(2 $1) \n"
    "-8.13655757904052734 *(1.5 $1) \n"
    "-8.31055450439453125 +(1 *(2 $1)) \n"
    "-8.31055450439453125 +(-1 *(2 +(1 $1))) \n"
    "-33.9310073852539062 0 \n"
)

_EXPECTED_SR_K2_ABSERR = (
    "-8.09113025665283203 $1 \n"
    "-8.44435310363769531 *(0.5 $1) \n"
    "-10.7238254547119141 *(2 $1) \n"
    "-11.6908273696899414 0 \n"
    "-11.908869743347168 +(1 $1) \n"
    "-15.1492815017700195 /($1 +(1 *(-1 sin($1)))) \n"
    "-15.1492815017700195 /($1 +(1 sin(*(-1 $1)))) \n"
    "-19.3443717956542969 1 \n"
    "-21.93115234375 *(3 $1) \n"
    "-22.8981533050537109 *(-1 $1) \n"
)

_EXPECTED_CP_CONTIN = "0 *(6 +(1 $1)) \n-45051.44921875 0 \n"

_EXPECTED_CP_ASYM = (
    "0 *(2 $1) \n"
    "-548.24810791015625 +(1 $1) \n"
    "-734.7406005859375 $1 \n"
    "-734.7406005859375 *(3 $1) \n"
    "-961.2330322265625 +(-1 $1) \n"
    "-1653.1663818359375 *(0.5 $1) \n"
    "-2545.9775390625 1 \n"
    "-2594.98828125 /($1 +(1 $1)) \n"
    "-2938.96240234375 0 \n"
    "-3371.947265625 -1 \n"
)

_EXPECTED_CP_IMPULSE = "-51 +(7 *(3 impulse($1))) \n-1836 0 \n"
_EXPECTED_CP_IMPULSE_NS2 = "-5 +(7 *(3 impulse($1))) \n-180 0 \n"
_EXPECTED_CP_UNKNOWN_SLOT = "0 or($1 !$3) \n-2 true \n"


def test_moses_exec_reproduces_classic_sr_oracle():
    """`moses -Hsr -k3 -m1000 -W1` -- classic's own defaults for
    everything this test does not name (`docs/CARRY-FORWARD.md` Sec 5.4).
    """
    rc, out = _run(["moses", "-Hsr", "-k3", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_SR_K3


def test_moses_exec_sr_problem_size_reaches_the_scorer():
    """`-k` is `simple_symbolic_regression`'s `order`, so changing it must
    change the TARGET function, not just the search budget. `-k1` (f(x)=x)
    is solved exactly, `-k0` (f(x)=0, `dorepeat(0)` running zero times) is
    solved by the constant `0`, and `-k3` above is solved by neither --
    three different answers from one flag.
    """
    rc, out = _run(["moses", "-Hsr", "-k1", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_SR_K1

    rc, out = _run(["moses", "-Hsr", "-k0", "-m200", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_SR_K0

    assert _EXPECTED_SR_K1 != _EXPECTED_SR_K3
    assert _EXPECTED_SR_K0 != _EXPECTED_SR_K3


def test_moses_exec_sr_seed_reaches_the_itable_draws():
    """`-r5` changes the sampled `ITable`, not only the optimizer's own
    draws -- the whole reason `ITable`'s random constructor had to be
    verified draw for draw. A different seed must give a different answer.
    """
    rc, out = _run(["moses", "-Hsr", "-k3", "-m1000", "-W1", "-r5"])
    assert rc == 0
    assert out == _EXPECTED_SR_K3_R5
    assert out != _EXPECTED_SR_K3


def test_moses_exec_sr_nsamples_reaches_the_itable_row_count():
    """`--nsamples 7` overrides `default_nsamples` (20), so the table has
    7 rows and 7 `randdouble()` draws instead of 20. It changes both the
    scores and the RNG stream every later decision reads.
    """
    rc, out = _run(["moses", "-Hsr", "-k3", "-m1000", "-W1", "--nsamples", "7"])
    assert rc == 0
    assert out == _EXPECTED_SR_K3_NS7
    assert out != _EXPECTED_SR_K3


def test_moses_exec_sr_overrides_linear_contin():
    """The load-bearing one. `--linear-regression=1` does TWO things in
    classic: `deme_params.linear_contin = linear_regression`
    (problem-params.cc:2264) and `ignore_ops.insert({div, exp, log, sin})`
    (problem-params.cc:2146-2151). `polynomial_problem::run` then
    UNDOES the first (`pms.deme_params.linear_contin = false`,
    demo-problems.cc:259) and leaves the second, "sr is fundamentally a
    kind of non-linear regression!".

    So this output is what classic prints with the reduced operator set
    but NON-linear contin knob building -- and its top entry,
    `*(+(1 *(+(1 $1) $1)) $1)`, is a nested product of `$1` that linear
    contin knob building cannot construct at all. Deleting that one
    assignment from `polynomial_problem.run` makes this test fail (see
    this task's report for the kill count).
    """
    rc, out = _run(["moses", "-Hsr", "-k3", "-m1000", "-W1", "--linear-regression=1"])
    assert rc == 0
    assert out == _EXPECTED_SR_K3_LINREG
    assert out != _EXPECTED_SR_K3


def test_moses_exec_sr_it_abs_err_reaches_contin_bscore():
    """`--it-abs-err=1` selects `contin_bscore::abs_error` over
    `squared_error`. Classic REQUIRES the `=1` (it is a
    `po::value<bool>`): the bare `--it-abs-err` aborts with
    "the required argument for option '--it-abs-err' is missing" and
    dumps core, which is why the flag is written with its value here.
    """
    rc, out = _run(["moses", "-Hsr", "-k2", "-m200", "-W1", "--it-abs-err=1"])
    assert rc == 0
    assert out == _EXPECTED_SR_K2_ABSERR


def test_moses_exec_reproduces_classic_cp_contin_oracle():
    """`MOSESUTest.cxxtest:141`'s own `test_cp_contin` program and flags
    (`-Hcp -y'+(6 *(6 $1))' -nexp`), which its comment calls "challenging
    because if min, max_rand_input aren't set correctly, then the scoring
    table will be on too small a range, and moses will fail to find a
    solution" -- i.e. it is exactly a test of the `largest_const_in_tree`
    hack. Classic finds `*(6 +(1 $1))`, one of the two answers
    `test_cp_contin` accepts.
    """
    rc, out = _run(["moses", "-Hcp", "-y+(6 *(6 $1))", "-nexp", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CP_CONTIN


def test_moses_exec_cp_contin_reproduces_the_swapped_sampling_range():
    """The discriminating case for demo-problems.cc:429's argument-order
    bug: classic passes `max_rand_input` as the constructor's
    `min_contin` and `min_rand_input` as its `max_contin`, so a sampled
    value is `(min_rand - max_rand) * r + max_rand`.

    `-q0 -w10` is chosen precisely because it defeats the "stupid hack"
    (which only fires when the range is still the 0..1 default) AND is
    asymmetric, so the swap is observable: classic samples `10 - 10r`,
    the un-swapped reading would sample `10r`. Passing the two arguments
    the "sensible" way round makes this test fail (see this task's
    report).
    """
    rc, out = _run(["moses", "-Hcp", "-y*(2 $1)", "-m200", "-W1", "-q0", "-w10"])
    assert rc == 0
    assert out == _EXPECTED_CP_ASYM


def test_moses_exec_cp_contin_samples_boolean_columns_as_booleans():
    """`+(6 *(6 impulse($1)))` is `->(boolean contin)` in classic (checked
    against a compiled `infer_type_tree` probe), so `ITable`'s random
    constructor must draw `randGen().randint(2)` for that column, not
    `randdouble()`. Sampling it as contin gives a different table, a
    different RNG stream, and a different answer.
    """
    rc, out = _run(["moses", "-Hcp", "-y+(6 *(6 impulse($1)))", "-m200", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CP_IMPULSE


def test_moses_exec_cp_contin_enumerates_booleans_when_nsamples_is_pow2_barity():
    """`comp_tt` (table.cc:100), the ONE CLI-reachable configuration that
    turns it on: `barity == 1`, so `--nsamples 2 == pow2(1)` and the
    single boolean column is ENUMERATED (`i & (1 << bidx++)`) instead of
    drawn. Classic scores `-5` here against `-51` without the flag, so
    the two branches are not merely different code paths -- they give
    different answers.
    """
    argv = ["moses", "-Hcp", "-y+(6 *(6 impulse($1)))", "-m200", "-W1", "--nsamples", "2"]
    rc, out = _run(argv)
    assert rc == 0
    assert out == _EXPECTED_CP_IMPULSE_NS2
    assert out != _EXPECTED_CP_IMPULSE


def test_moses_exec_cp_boolean_program_with_an_unmentioned_argument():
    """`or($1 !$3)` has arity 3 but only mentions two arguments, so
    classic's real signature is `->(boolean unknown boolean boolean)`.
    Pinned because `combo_problem.run`'s boolean branch deliberately does
    NOT pass that signature (see its own `# DIVERGENCE` marker): this is
    the measurement that the substitution is invisible here.
    """
    rc, out = _run(["moses", "-Hcp", "-yor($1 !$3)", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CP_UNKNOWN_SLOT


def test_moses_exec_cp_rejects_a_cross_argument_type_mismatch():
    """`+(impulse($1) $1)` uses `$1` in boolean and contin position at
    once. Classic's real `infer_type_tree` returns `->(ill_formed
    ill_formed)`, `is_well_formed` is false, and it prints
    "The combo tree +(impulse($1) $1) is not well formed." and exits -1
    (rc 255, measured). This port raises `ValueError` with the same
    message -- the established convention for a C++ path that ends in
    `exit()` rather than returning (see `moses_exec`'s own module
    docstring).

    Note the double space: `combo_tree_to_str` emits a trailing space
    after the tree, exactly as C++'s `operator<<(ostream&, combo_tree)`
    does, so the rendered sentence is identical in both.
    """
    with pytest.raises(ValueError, match="is not well formed"):
        moses_exec(["moses", "-Hcp", "-y+(impulse($1) $1)", "-m200", "-W1"])


# ---------------------------------------------------------------------------
# -H cluster end to end.
#
# `cluster_table_problem` (table-problems.cc:551) + `cluster_bscore`
# (bscores.cc:937). Byte-for-byte against classic's own binary.
#
# The dataset choice is the point. `-Hcluster` on `dataset.csv` prints
# exactly `-inf 0` -- classic's own answer, but a DEGENERATE one that a
# completely broken port would also produce, so it cannot stand alone
# (docs/CARRY-FORWARD.md Sec 5/6, the "-inf corpus" trap). The three
# contin cases below print real, varied, finite scores and are the ones
# that actually discriminate; the `dataset.csv` case is kept only to pin
# that the degenerate path agrees too.
#
# Captured: ( ulimit -v 2097152; timeout 300
#   $CLASSIC/build/moses/moses/main/moses <argv below> )
# ---------------------------------------------------------------------------

_EXPECTED_CLUSTER_M200 = "8.33174324035644531 +(*(-1 log($x2)) $x1) \n-inf 0 \n"

_EXPECTED_CLUSTER_M1000 = (
    "17.0593795776367188 +(*(0.5 log($x2)) $x1 $x2) \n"
    "17.0593795776367188 +(*(2 +($x1 $x2)) log($x2)) \n"
    "16.8451042175292969 +(*(0.5 +(log($x2) $x1)) $x2) \n"
    "17.0605487823486328 +(-2 *(0.5 log($x2)) $x1 $x2) \n"
    "17.0596122741699219 +(*(0.5 log(*(0.5 $x2))) $x1 $x2) \n"
    "17.0594425201416016 +(1 *(0.5 log($x2)) $x1 $x2) \n"
    "17.0594234466552734 +(*(0.5 log(*(2 $x2))) $x1 $x2) \n"
    "17.0593795776367188 *(0.5 +(*(0.5 log($x2)) $x1 $x2)) \n"
    "17.0593643188476562 +(*(0.5 log(*(3 $x2))) $x1 $x2) \n"
    "17.0590305328369141 +(-1 *(0.5 log($x2)) $x1 $x2) \n"
)

_EXPECTED_CLUSTER_M400_SEED11 = (
    "15.0173139572143555 +(log($x2) $x1 $x2) \n"
    "15.0173730850219727 +(-1 log($x2) $x1 $x2) \n"
    "15.0173158645629883 +(1 log($x2) $x1 $x2) \n"
    "15.0173139572143555 *(0.5 +(log($x2) $x1 $x2)) \n"
    "13.7316188812255859 +(*(0.5 $x1) log($x2)) \n"
    "13.6357107162475586 +(*(0.5 log($x2)) $x1) \n"
    "13.1387338638305664 +(log($x2) $x1) \n"
    "14.2640914916992188 /(+(log($x2) $x1) +(1 *(-1 $x2))) \n"
    "-inf 0 \n"
)


@pytest.mark.oracle
def test_moses_exec_reproduces_classic_cluster_oracle_m200():
    rc, out = _run(["moses", f"-i{_CONTIN_CSV_PATH}", "-u", "y", "-Hcluster", "-m200", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CLUSTER_M200


@pytest.mark.oracle
def test_moses_exec_reproduces_classic_cluster_oracle_m1000():
    """Ten surviving candidates, all with distinct finite scores agreeing
    to the last printed digit -- the case that would break first on any
    `score_t` narrowing this port got wrong inside `cluster_bscore`.
    """
    rc, out = _run(["moses", f"-i{_CONTIN_CSV_PATH}", "-u", "y", "-Hcluster", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_CLUSTER_M1000
    assert out != _EXPECTED_CLUSTER_M200


@pytest.mark.oracle
def test_moses_exec_reproduces_classic_cluster_oracle_other_seed():
    """A different `-r` reaches a different part of the search space (a
    `/(...)` candidate appears, which neither run above produces), so
    this is not the same measurement twice.
    """
    rc, out = _run(
        ["moses", f"-i{_CONTIN_CSV_PATH}", "-u", "y", "-Hcluster", "-m400", "-r11", "-W1"]
    )
    assert rc == 0
    assert out == _EXPECTED_CLUSTER_M400_SEED11


@pytest.mark.oracle
def test_moses_exec_reproduces_classic_cluster_degenerate_oracle(dataset_csv):
    """The DEGENERATE case, kept deliberately and labelled as such: on a
    boolean 0/1 table every candidate's output column collapses into too
    few distinct values for three clusters of >= 3.5 members, so
    `cnt[i] < 3.5` (bscores.cc:1008) returns `-inf` for everything. This
    matching proves nothing on its own -- see this section's header
    comment -- and the three tests above are the real evidence.
    """
    rc, out = _run(["moses", f"-i{dataset_csv}", "-Hcluster", "-m200", "-W1"])
    assert rc == 0
    assert out == "-inf 0 \n"


def test_cluster_oracles_are_not_all_degenerate():
    """Guards the guard: if someone later replaces the contin expectations
    with all-`-inf` output, this fails rather than letting the section
    silently become the trap it was written to avoid.
    """
    finite = [
        line.split()[0]
        for block in (
            _EXPECTED_CLUSTER_M200,
            _EXPECTED_CLUSTER_M1000,
            _EXPECTED_CLUSTER_M400_SEED11,
        )
        for line in block.splitlines()
        if line and line.split()[0] not in ("-inf", "inf", "nan")
    ]
    assert len(set(finite)) >= 10, finite


# ---------------------------------------------------------------------------
# -H pre-conj end to end.
#
# `pre_conj_table_problem` (table-problems.cc:410) was a raising stub in
# `problem.py` until 2026-09-09, on the recorded grounds that "what is
# missing is only a pre_conj_table_problem in table_problems.py" -- an
# accurate description of the defect, kept for weeks as though it were a
# discharge of it. Classic RUNS the problem; the port now does too.
#
# The name is `pre-conj`, with a HYPHEN (table-problems.h:134). `-Hpre_conj`
# names nothing in either registry and is refused by both -- see
# `test_moses_exec_pre_conj_UNDERSCORE_is_unsupported_in_both` below.
#
# Captured: ( ulimit -v 2097152; timeout 120
#   $CLASSIC/build/moses/moses/main/moses -Hpre-conj
#   -i$CLASSIC/tests/moses/dataset.csv -m1000 -W1 [flag] )
# ---------------------------------------------------------------------------

_EXPECTED_PRE_CONJ = (
    "0 false \n"
    "0.5 and(!$arg1 !$arg2) \n"
    "0.166666686534881592 and($arg1 $arg2) \n"
    "-0.555555582046508789 true \n"
    "-0.166666656732559204 and(!$arg1 $arg2) \n"
    "-0.166666656732559204 and($arg1 !$arg2) \n"
    "-0.5 $arg1 \n"
    "-0.5 $arg2 \n"
    "0.333333343267440796 or(and(!$arg1 !$arg2) and($arg1 $arg2)) \n"
    "-0.666666626930236816 !$arg1 \n"
)

# -Q3 == `--alpha 3`, i.e. `pms.hardness = 3.0`, the SECOND argument of
# `precision_conj_bscore (ctable, pms.hardness, pms.pre_positive)`.
_EXPECTED_PRE_CONJ_Q3 = (
    "-0.5 and(!$arg1 !$arg2) \n"
    "-0.833333313465118408 and($arg1 $arg2) \n"
    "-0.333333313465118408 or(and(!$arg1 !$arg2) and($arg1 $arg2)) \n"
    "-1.16666662693023682 and(!$arg1 $arg2) \n"
    "-1.16666662693023682 and($arg1 !$arg2) \n"
    "-0.666666626930236816 or(and(!$arg1 $arg2) and($arg1 !$arg2)) \n"
    "-2 false \n"
    "-0.833333313465118408 and(or(and(!$arg1 !$arg2) and($arg1 $arg2)) or(!$arg1 $arg2)) \n"
    "-0.833333313465118408 and(or(and(!$arg1 !$arg2) and($arg1 $arg2)) or($arg1 $arg2)) \n"
    "-2.55555558204650879 true \n"
)

# `--pre-positive=0` is the THIRD argument, `positive`.
_EXPECTED_PRE_CONJ_NEGATIVE = (
    "0 false \n"
    "0.5 and(!$arg1 !$arg2) \n"
    "0.166666686534881592 and(!$arg1 $arg2) \n"
    "0.166666686534881592 and($arg1 !$arg2) \n"
    "-0.444444417953491211 true \n"
    "-0.333333313465118408 !$arg1 \n"
    "-0.333333313465118408 !$arg2 \n"
    "-0.166666656732559204 and($arg1 $arg2) \n"
    "-0.5 $arg1 \n"
    "-0.5 $arg2 \n"
)


@pytest.mark.oracle
def test_moses_exec_reproduces_classic_pre_conj_oracle(dataset_csv):
    rc, out = _run(["moses", "-Hpre-conj", f"-i{dataset_csv}", "-m1000", "-W1"])
    assert rc == 0
    assert out == _EXPECTED_PRE_CONJ


@pytest.mark.oracle
def test_moses_exec_pre_conj_hardness_reaches_the_scorer(dataset_csv):
    """`-Q3` (hardness) changes every printed score AND the surviving
    candidate set. Hardcoding `precision_conj_bscore`'s `hardness`
    argument makes this fail while the default-options oracle above still
    passes -- the two together are what pin `pms.hardness` as the source.
    """
    rc, out = _run(["moses", "-Hpre-conj", f"-i{dataset_csv}", "-m1000", "-W1", "-Q3"])
    assert rc == 0
    assert out == _EXPECTED_PRE_CONJ_Q3
    assert out != _EXPECTED_PRE_CONJ


@pytest.mark.oracle
def test_moses_exec_pre_conj_pre_positive_reaches_the_scorer(dataset_csv):
    """The third `REGRESSION` argument, `pms.pre_positive`. Dropping it
    (so `precision_conj_bscore`'s own `positive=True` default applies)
    makes this fail.
    """
    rc, out = _run(["moses", "-Hpre-conj", f"-i{dataset_csv}", "-m1000", "-W1", "--pre-positive=0"])
    assert rc == 0
    assert out == _EXPECTED_PRE_CONJ_NEGATIVE
    assert out != _EXPECTED_PRE_CONJ


def test_moses_exec_pre_conj_UNDERSCORE_is_unsupported_in_both(dataset_csv, capsys):
    """Measured against the real binary, 2026-09-09:

      ( ulimit -v 2097152; timeout 120 $CLASSIC/build/moses/moses/main/moses
        -Hpre_conj -i<precision.csv> -u y -m100 -W1 )
      -> problem type "pre_conj" is currently unsupported.   (exit 1)

    `pre_conj_table_problem::name()` returns `"pre-conj"` (table-problems.h
    :134), so the underscore spelling names nothing in classic's registry
    either. Wiring `-Hpre_conj` as an alias would make this port ACCEPT
    an invocation classic rejects; this test exists to stop that.
    """
    rc = moses_exec(["moses", "-Hpre_conj", f"-i{dataset_csv}", "-m100", "-W1"])
    assert rc == 1
    assert capsys.readouterr().err == 'problem type "pre_conj" is currently unsupported.\n'


# ---------------------------------------------------------------------------
# --ps-max-particles end to end (-a ps).
#
# `problem_params` parsed and stored `ps_max_particles` and nothing read
# it: `moses_main.metapop_moses_results_b` built `ps_parameters()` fresh
# instead of taking the caller's. C++ threads a `ps_params` bundle through
# both `metapop_moses_results*` signatures (moses_main.h:278,350) and
# `problem-params.cc:2370` sets `ps_params.max_parts = ps_max_particles`.
#
# `calc_swarm_size` (particle-swarm.cc:239-250) is
# `clamp(n_bits * 3/4 + n_contin * 3, 4, max_parts)`, so the flag is
# observable only where the unclamped total EXCEEDS 4 -- which is why the
# 2-input `dataset.csv` shows no difference at any N (measured: identical
# output for N in {4,6,8,12,50}) and cannot be the corpus here. The
# 3-input `trainingBoolean.dat` gives three distinct answers.
#
# Captured: ( ulimit -v 2097152; timeout 120
#   $CLASSIC/build/moses/moses/main/moses
#   -i<trainingBoolean.dat> -u y -aps -m300 -W1 -r1 [--ps-max-particles=N] )
# ---------------------------------------------------------------------------

_TABLE_DATA = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_BOOL3_PATH = _TABLE_DATA / "trainingBoolean.dat"

_EXPECTED_PS_N4 = (
    "0 or(and($x1 $x2) !$x3) \n"
    "-1 !$x3 \n"
    "-2 or(!$x1 $x2 !$x3) \n"
    "-2 or($x1 !$x2 !$x3) \n"
    "-3 true \n"
    "-2 and(or(and(or(!$x2 $x3) !$x1) $x2) !$x3) \n"
    "-3 and($x1 !$x3) \n"
    "-3 or(!$x1 $x2) \n"
    "-3 or($x1 $x2) \n"
    "-4 and(or(!$x1 $x3) $x2) \n"
)

_EXPECTED_PS_N6 = (
    "-1 or($x2 !$x3) \n"
    "-2 or(!$x1 $x2 !$x3) \n"
    "-2 or($x1 !$x2 !$x3) \n"
    "-2 or(and($x1 !$x2) !$x3) \n"
    "-3 true \n"
    "-3 $x1 \n"
    "-2 or(and(or(!$x2 $x3) !$x1) $x2 !$x3) \n"
    "-3 and($x1 $x2) \n"
    "-3 and(or(and(!$x1 !$x2) !$x3) or(!$x1 $x2)) \n"
    "-4 and($x1 $x2 !$x3) \n"
)

# N=8 already saturates the clamp for this field_set, so it is also what
# the DEFAULT (50) prints -- see the two tests below.
_EXPECTED_PS_SATURATED = (
    "-1 or($x2 !$x3) \n"
    "-2 or(!$x1 $x2 !$x3) \n"
    "-2 or($x1 !$x2 !$x3) \n"
    "-3 true \n"
    "-3 $x1 \n"
    "-2 or(and(or(!$x2 $x3) !$x1) $x2 !$x3) \n"
    "-3 or($x1 $x2) \n"
    "-3 or(and(or(!$x1 !$x2) or($x1 !$x3)) $x2 $x3) \n"
    "-4 or(!$x1 !$x2 $x3) \n"
    "-4 or(!$x1 $x2 $x3) \n"
)


def _run_ps(extra):
    return _run(["moses", f"-i{_BOOL3_PATH}", "-u", "y", "-aps", "-m300", "-W1", "-r1", *extra])


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (4, _EXPECTED_PS_N4),
        (6, _EXPECTED_PS_N6),
        (8, _EXPECTED_PS_SATURATED),
    ],
    ids=["N4", "N6", "N8-saturated"],
)
def test_moses_exec_ps_max_particles_reaches_particle_swarm(n, expected):
    """MUTATION PROVEN, and it is the mutation the defect WAS: restoring
    `particle_swarm(opt_params, ps_parameters())` in
    `metapop_moses_results_b` makes the N=4 and N=6 cases fail (both
    print the saturated N>=8 answer) while N=8 still passes -- exactly
    the shape of the original bug, and exactly why sampling only one or
    two configurations missed it.

    Dropping `pms.ps_params` from any single `metapop_moses_results` call
    site is caught the same way, and removing the
    `ps_params.max_parts = self.ps_max_particles` assignment in
    `problem_params` is caught by all three.
    """
    rc, out = _run_ps([f"--ps-max-particles={n}"])
    assert rc == 0
    assert out == expected


def test_moses_exec_ps_oracles_actually_differ():
    """Guards the guard (CARRY-FORWARD.md Sec 5.9 / the "corpus holding
    only values where faithful and broken agree" failure mode): if the
    three expectations above ever collapse to one string, the
    parametrized test above would pass no matter what `ps_params` did.
    """
    assert len({_EXPECTED_PS_N4, _EXPECTED_PS_N6, _EXPECTED_PS_SATURATED}) == 3


@pytest.mark.oracle
def test_moses_exec_ps_default_max_particles_is_50():
    """No flag at all must give classic's default-50 answer, which for
    this field_set is the same saturated output `--ps-max-particles=8`
    produces (the clamp's upper bound stops binding at 8). Setting the
    default to anything below 8 in `problem_params` breaks this.
    """
    rc, out = _run_ps([])
    assert rc == 0
    assert out == _EXPECTED_PS_SATURATED


# ---------------------------------------------------------------------------
# --time-bscore-granularity=month: classic ABORTS, and so must the port.
#
# Reported by an earlier audit, then twice recorded as "did not
# reproduce" -- because the reproduction needs `-H pre` (the only problem
# that constructs a `precision_bscore`, and hence the only one that
# constructs the `bscore_ctable_time_dispersion` base whose constructor
# reads the granularity) AND `-q/-w` bounds that get past
# `precision_bscore`'s own earlier activation-bounds assert
# (precision_bscore.cc:163). With `-H it` (the default problem) the flag
# is genuinely inert in classic too, which is what the "both exit 0,
# byte-identical" check had actually measured.
#
# Measured 2026-09-09:
#   ( ulimit -v 2097152; timeout 120 $CLASSIC/build/moses/moses/main/moses
#     -Hpre -i<dataset.csv> -m300 -W1 -q0.1 -w1.0
#     [--time-bscore-granularity=month] )
#   no flag / =day -> exit 0, the candidate list below
#   =month        -> exit 134, "terminate called after throwing an
#                    instance of 'opencog::AssertionException' / what():
#                    timestamp is not a valid date, are you sure you have
#                    loaded a timestamp feature (option
#                    --timestamp-feature)? (time_dispersion.cc:62)",
#                    core dumped
# ---------------------------------------------------------------------------

_EXPECTED_PRE_DAY = (
    "0.444444447755813599 true \n"
    "0.5 $arg1 \n"
    "0.5 $arg2 \n"
    "0.666666686534881592 and($arg1 $arg2) \n"
    "0.333333313465118408 !$arg1 \n"
    "0.333333313465118408 !$arg2 \n"
    "0.5 or(!$arg1 $arg2) \n"
    "0.5 or($arg1 !$arg2) \n"
    "0.444444447755813599 or($arg1 $arg2) \n"
    "0.333333313465118408 and(!$arg1 $arg2) \n"
)


@pytest.mark.oracle
@pytest.mark.parametrize("extra", [[], ["--time-bscore-granularity=day"]])
def test_moses_exec_pre_day_granularity_matches_classic(dataset_csv, extra):
    rc, out = _run(["moses", "-Hpre", f"-i{dataset_csv}", "-m300", "-W1", "-q0.1", "-w1.0", *extra])
    assert rc == 0
    assert out == _EXPECTED_PRE_DAY


def test_moses_exec_pre_month_granularity_fails_like_classic(dataset_csv):
    """MUTATION PROVEN: deleting the `granularity == month` check in
    `bscore_ctable_time_dispersion.__init__` makes this test fail -- the
    run completes with exit 0 and prints `_EXPECTED_PRE_DAY`, i.e. this
    port silently accepting an invocation that core-dumps classic. That
    was the live behaviour until 2026-09-09.

    The message is classic's own (time_dispersion.cc:62). The exception
    TYPE differs by this project's usual OC_ASSERT convention -- classic
    throws `AssertionException` out of `main` and dumps core, which is
    not reproducible as library behaviour.
    """
    with pytest.raises(AssertionError, match="timestamp is not a valid date"):
        moses_exec(
            [
                "moses",
                "-Hpre",
                f"-i{dataset_csv}",
                "-m300",
                "-W1",
                "-q0.1",
                "-w1.0",
                "--time-bscore-granularity=month",
            ]
        )


@pytest.mark.oracle
def test_moses_exec_it_month_granularity_is_inert_in_both(dataset_csv):
    """The complement, and the measurement that made this defect look
    unreproducible: `-H it` never constructs a `precision_bscore`, so the
    flag is read by nothing and classic runs to completion. Byte-checked
    against `-i<dataset.csv> -m1000 -W1`'s own oracle above. A gate put
    in `problem_params` instead of at the point of use would break this.
    """
    argv = ["moses", f"-i{dataset_csv}", "-m1000", "-W1", "--time-bscore-granularity=month"]
    rc, out = _run(argv)
    assert rc == 0
    assert out == _EXPECTED_IT_LABELS
