"""Oracle and guard tests for pymetta.moses.main.moses_exec (S13 Task 3).

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

import io
from contextlib import redirect_stdout

import pytest

from pymetta.moses.main.moses_exec import moses_exec

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
    assert out == (
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


def test_moses_exec_unregistered_stub_problem_raises():
    """`-H cluster` names one of the four problems `table_problems.py`
    leaves as `problem.py`'s own raising `_unported_table_problem` stub
    (`cluster_bscore` wiring was not attempted -- table_problems.py's own
    module docstring).
    """
    with pytest.raises(NotImplementedError, match="cluster"):
        moses_exec(["moses", "-Hcluster"])


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
