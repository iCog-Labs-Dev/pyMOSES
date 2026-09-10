"""Tests for ``pymetta.feature_selection.main.fs_main`` (the standalone
``feature-selection`` binary's ``main()``) and for the four
feature-selection.cc orchestration functions it drives
(``feature_selection``/``write_results``/``log_selected_features``/
``err_empty_features``) plus ``table_output.ostreamTable``.

## Oracle

Every expected value in this module was produced by running
``$CLASSIC/build/moses/feature-selection/main/feature-selection`` directly,
bounded by ``( ulimit -v 2097152; timeout 300 ... )``, through a bash
harness using a ``"$@"`` array (docs/CARRY-FORWARD.md 5.10: the session
shell is zsh and does NOT word-split an unquoted ``$args``; an inline loop
silently passes ONE argument and reports false agreement -- and it bit this
task once, on the ``eval-features`` probe, before the harness was rewritten
in bash).

The harness compared stdout, stderr, the ``-o`` output file, the produced
log FILE NAME and the log CONTENT (through ``tools/normalize_log.py``) for
every case. 29 of 30 stdout comparisons were byte-identical; the one
exception is ``--subsampling-ratio``, which needs
``pymetta.feature_selection.ctable_subsample`` (see the module docstring of
``feature_selection.py``).

Representative invocations, with what classic printed:

    $ feature-selection -i trainingBoolean.dat -u y -l NONE -C2
    x2,y,x3
    0,1,0
    0,0,1
    1,1,0
    1,0,1
    0,1,0
    0,0,1
    1,1,0
    1,1,1

    $ feature-selection -i trainingBoolean.dat -u y -l NONE -C2 -a bogus
    Fatal Error: Algorithm 'bogus' is unknown, please consult the help ...   (stderr, exit 1)

    $ feature-selection -i bool_a6_mix.csv -u y -l NONE -a inc -T 5
    No features have been selected.                                          (stderr, exit 0)

    $ feature-selection --version
    feature-selection "3"."7"."0"

    $ feature-selection -i trainingBoolean.dat -u y -l NONE -C2 --subsampling-ratio 0.5
    terminate called ... Ratio must be in [0.0, 1.0], but is 4.000000        (SIGABRT)
"""

import re
from pathlib import Path

import pytest

from pymetta.comboreduct.table.table_io import loadTable
from pymetta.feature_selection.main.feature_selection_params import feature_selection_parameters
from pymetta.feature_selection.main.fs_main import fs_main, subsampleTable
from pymetta.feature_selection.main.table_output import ostreamTable, ostreamTableHeader
from pymetta.util.logger import logger
from pymetta.util.rng import randGen

_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_BOOL = str(_TABLE_DATA_DIR / "trainingBoolean.dat")
_XOR = str(_TABLE_DATA_DIR / "trainingBooleanXor.dat")
_MIX6 = str(_TABLE_DATA_DIR / "bool_a6_mix.csv")

# `[<timestamp>] [<LEVEL>] <message>` -- Logger.cc:500's own record shape.
_RECORD_RE = re.compile(r"^\[[^]]*\] \[([A-Z]+)\] (.*)$")

# Classic's stdout for `-i trainingBoolean.dat -u y -l NONE -C2`, verbatim.
_BOOL_C2 = "x2,y,x3\n0,1,0\n0,0,1\n1,1,0\n1,0,1\n0,1,0\n0,0,1\n1,1,0\n1,1,1\n"

# Classic's stdout for `-i trainingBoolean.dat -u y -l NONE -C1`, verbatim.
_BOOL_C1 = "x3,y\n0,1\n1,0\n0,1\n1,0\n0,1\n1,0\n0,1\n1,1\n"

# Classic's stdout for `-i trainingBooleanXor.dat -u y -l NONE -C1`, verbatim.
_XOR_C1 = "x2,y\n0,1\n1,1\n0,1\n1,0\n"


@pytest.fixture(autouse=True)
def _restore_the_logger_singleton():
    """``fs_main`` reconfigures the PROCESS-WIDE logger (``set_filename``,
    ``set_level``, ``set_backtrace_level``), exactly as classic's ``main``
    does -- but classic gets one process per run and pytest does not.
    docs/CARRY-FORWARD.md 5.11 records the same shape for the reduction
    singletons: an in-process harness that drives a ``main()`` more than
    once measures the previous run as much as the current one. Without
    this fixture, the first ``-l NONE`` case here would leave every later
    test in the session with logging switched off.
    """
    lg = logger()
    saved = (lg.get_filename(), lg.get_level(), lg.get_backtrace_level())
    try:
        yield
    finally:
        lg.set_filename(saved[0])
        lg.set_level(saved[1])
        lg.set_backtrace_level(saved[2])


def _run(capsys, *args: str) -> tuple[int, str, str]:
    rc = fs_main(["feature-selection", *args])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("args", "want"),
    [
        ((["-i", _BOOL, "-u", "y", "-l", "NONE", "-C2"]), _BOOL_C2),
        ((["-i", _BOOL, "-u", "y", "-l", "NONE", "-C1"]), _BOOL_C1),
        ((["-i", _XOR, "-u", "y", "-l", "NONE", "-C1"]), _XOR_C1),
    ],
)
def test_fs_main_stdout_is_byte_identical_to_classic(capsys, args, want):
    """The whole binary, end to end: argv -> loadTable -> select ->
    filter -> ostreamTable. See this module's docstring for the classic
    invocations these three came from.
    """
    rc, out, err = _run(capsys, *args)
    assert rc == 0
    assert err == ""
    assert out == want


@pytest.mark.oracle
def test_fs_main_target_column_lands_at_its_filtered_position(capsys):
    """``x2,y,x3`` -- NOT ``x2,x3,y``.

    ``Table::update_pos(3, {1,2}) == 1``, so ``ostreamTableHeader``
    inserts the target label at index 1. A port that appended the target
    instead would produce a table classic never writes; this is the one
    assertion that pins the ``min(target_pos, hsize)`` insert.
    """
    _rc, out, _err = _run(capsys, "-i", _BOOL, "-u", "y", "-l", "NONE", "-C2")
    assert out.splitlines()[0] == "x2,y,x3"


@pytest.mark.oracle
def test_fs_main_force_feature_adds_to_the_selection(capsys):
    """``-C1 -e x1`` selects x3 (the -C1 winner) plus the forced x1.

    Classic prints a 2-input table here, and the forced index comes from
    ``get_indices(force_features_str, table.itable.get_labels())`` --
    INPUT labels only, so the index is 0 for ``x1`` and the union is
    ``{0, 2}``.
    """
    _rc, out, _err = _run(capsys, "-i", _BOOL, "-u", "y", "-l", "NONE", "-C1", "-e", "x1")
    assert out.splitlines()[0] == "x1,y,x3"


@pytest.mark.oracle
def test_fs_main_writes_the_output_file_instead_of_stdout(capsys, tmp_path):
    """``-o <file>`` routes through ``saveTable``, which TRUNCATES (no
    ``ios_base::app``, unlike ``eval-features``' ``output_results``).
    Classic's own out.csv for this invocation was byte-identical to its
    stdout for the same invocation without ``-o``.
    """
    out_file = tmp_path / "out.csv"
    rc, out, err = _run(
        capsys, "-i", _BOOL, "-u", "y", "-l", "NONE", "-C2", "-o", str(out_file)
    )
    assert (rc, out, err) == (0, "", "")
    assert out_file.read_text() == _BOOL_C2


@pytest.mark.oracle
def test_fs_main_version_matches_classic(capsys):
    rc, out, err = _run(capsys, "--version")
    assert (rc, err) == (0, "")
    assert out == 'feature-selection "3"."7"."0"\n'


def test_fs_main_with_no_arguments_prints_help_and_returns_one(capsys):
    """C++: fs-main.cc:362 `if (vm.count("help") || argc == 1)` -- the
    `argc == 1` half, which argparse has no equivalent for. classic
    returns 1 here (not 0), and so does this.

    # The help TEXT itself is boost's, not argparse's, and is a documented
    # divergence (fs_main.py's module docstring); only the return code and
    # the fact that something was printed are asserted.
    """
    rc, out, _err = _run(capsys)
    assert rc == 1
    assert "usage:" in out


@pytest.mark.oracle
def test_fs_main_rejects_a_bad_log_level(capsys):
    """C++: fs-main.cc:398-402. Measured: classic writes exactly this line
    to stderr and exits 1 (56 bytes, byte-compared by the harness).
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(capsys, "-i", _BOOL, "-u", "y", "-l", "BOGUS", "-C2")
    assert excinfo.value.code == 1
    assert capsys.readouterr().err == (
        "Fatal Error: Log level BOGUS is incorrect (see --help).\n"
    )


@pytest.mark.oracle
def test_fs_main_reports_and_exits_zero_when_nothing_is_selected(capsys, tmp_path):
    """``err_empty_features`` -- classic writes "No features have been
    selected." to stderr, writes a one-line comment file, and ``exit(0)``.
    Measured: ``-i bool_a6_mix.csv -u y -a inc -T 5`` selects nothing, and
    classic's stderr for it is exactly 32 bytes.
    """
    out_file = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as excinfo:
        _run(capsys, "-i", _MIX6, "-u", "y", "-l", "NONE", "-a", "inc", "-T", "5",
             "-o", str(out_file))
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "No features have been selected.\n"
    assert out_file.read_text() == "# No features have been selected.\n"


# ---------------------------------------------------------------------------
# subsampleTable -- the two reproduced upstream defects
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_subsampling_ratio_asserts_because_classic_passes_rows_as_a_ratio():
    """fs-main.cc:437 passes ``ratio * table.size()`` into
    ``subsampleTable(float ratio, Table&)``, whose assert wants a ratio.
    Measured on the 8-row table: ``--subsampling-ratio 0.5`` aborts with
    "Ratio must be in [0.0, 1.0], but is 4.000000". DO NOT FIX.

    Both the boundary values are checked, because a port that clamped, or
    that used ``<`` instead of ``<=``, would pass one and fail the other:
    0.125*8 == 1.0 is ACCEPTED and 0.13*8 == 1.04 is REJECTED, and classic
    agrees with both (exit 0 / abort).
    """
    table = loadTable(_BOOL, "y")
    assert table.size() == 8
    with pytest.raises(AssertionError, match=r"Ratio must be in \[0\.0, 1\.0\], but is 4\.000000"):
        subsampleTable(0.5 * table.size(), table)
    with pytest.raises(AssertionError, match=r"but is 1\.040000"):
        subsampleTable(0.13 * table.size(), loadTable(_BOOL, "y"))


@pytest.mark.oracle
def test_subsample_table_keeps_exactly_the_rows_classic_keeps():
    """The RNG DRAW ORDER is the claim here, not just the row count.

    Measured: ``feature-selection -i trainingBoolean.dat -u y -C2
    --subsampling-ratio 0.1 -r 1`` prints a 6-row table whose rows, with
    the feature-selected column x3 dropped, are exactly
    ``(0,0,1) (0,0,0) (0,1,0) (1,0,1) (1,0,0) (1,1,1)``. Those are rows
    0,1,3,4,5,7 of the original -- i.e. classic's two ``randGen().randint(8)``
    draws removed rows 2 and 6, in that order. Compared element for
    element, not set-wise.
    """
    table = loadTable(_BOOL, "y")
    randGen().seed(1)
    subsampleTable(0.1 * table.size(), table)
    assert ostreamTable(table) == (
        "x1,x2,x3,y\n0,0,0,1\n0,0,1,0\n0,1,1,0\n1,0,0,1\n1,0,1,0\n1,1,1,1\n"
    )


@pytest.mark.oracle
def test_subsample_table_at_ratio_one_over_size_removes_nothing():
    """``0.125 * 8 == 1.0`` -> ``nrows == 8`` -> ``nrows < size`` false ->
    no draws at all. Classic prints all 8 rows for
    ``--subsampling-ratio 0.125``, which is the distinguishing case
    against an implementation that removed ``(1-ratio)*size`` rows.
    """
    table = loadTable(_BOOL, "y")
    randGen().seed(1)
    before = ostreamTable(table)
    subsampleTable(0.125 * table.size(), table)
    assert ostreamTable(table) == before


def test_subsample_table_raises_rather_than_reproducing_the_out_of_range_erase():
    """``--subsampling-ratio 0.05`` SIGSEGVs in classic (exit 139,
    measured): ``ridx`` is drawn against the ORIGINAL size on every
    iteration, so after enough removals it names a row past the end and
    ``it.erase(it.begin()+ridx)`` walks off the vector. Undefined
    behaviour is not reproducible (docs/CARRY-FORWARD.md section 3), so
    this raises instead.
    """
    table = loadTable(_BOOL, "y")
    randGen().seed(1)
    with pytest.raises(IndexError, match="past the end"):
        subsampleTable(0.05 * table.size(), table)


# ---------------------------------------------------------------------------
# ostreamTable, on shapes fs_main itself cannot reach
# ---------------------------------------------------------------------------


def test_ostream_table_header_clamps_the_target_position():
    """``min(table.target_pos, hsize)``: a ``target_pos`` past the end of
    the (filtered) input labels must still produce a valid header, with
    the target LAST. Reaching this needs a hand-built table -- every
    ``filtered()`` result keeps ``target_pos`` in range -- so it is not
    covered by the end-to-end cases above.
    """
    table = loadTable(_BOOL, "y")
    table.target_pos = 99
    assert ostreamTableHeader(table) == "x1,x2,x3,y"
    assert ostreamTable(table).splitlines()[1] == "0,0,0,1"


def test_ostream_table_prints_booleans_in_binary_form():
    """table_io.h:192-193's own contract, and the reason a boolean column
    reads ``0``/``1`` rather than ``true``/``false``: C++ streams a
    ``bool`` without ``boolalpha``.
    """
    table = loadTable(_BOOL, "y")
    body = ostreamTable(table).splitlines()[1:]
    assert set("".join(body).replace(",", "")) == {"0", "1"}


def test_fs_scorer_parameters_carry_the_standalone_cli_defaults():
    """fs-main.cc's own ``->default_value(...)`` list, which differs from
    the struct's in three places (``target_size`` 0 vs 1,
    ``hc_crossover`` False vs True, ``mi_confi`` 100 vs 50). A port that
    fell back to the struct defaults would silently change every
    selection, so the wiring is asserted directly rather than only
    through the end-to-end cases.
    """
    import argparse

    from pymetta.feature_selection.main.fs_main import _add_options

    parser = argparse.ArgumentParser(prog="feature-selection", add_help=False)
    _add_options(parser)
    ns = parser.parse_args([])
    assert ns.target_size == 0
    assert ns.hc_crossover is False
    assert ns.mi_confi == 100.0
    assert ns.hc_crossover_pop_size == 120
    assert ns.hc_crossover_min_neighbors == 400
    assert ns.hc_max_score == 1.0
    assert ns.inc_red_intensity == 0.1
    assert ns.inc_target_size_epsilon == 0.001
    assert ns.smd_top_size == 10
    assert ns.log_level == "DEBUG"
    # The struct's own defaults, for contrast -- both are real, neither is
    # "the" default (feature_selection_params.py's module docstring).
    assert feature_selection_parameters().target_size == 1
    assert feature_selection_parameters().hc_crossover is True
    assert feature_selection_parameters().mi_confi == 50.0


@pytest.mark.oracle
def test_fs_main_log_records_match_classic(capsys, tmp_path):
    """The whole INFO-level log, record for record, against classic's.

    Captured from the live binary (timestamps and the argv/path fields
    stripped, since those are not behaviour -- the same reduction
    ``tools/normalize_log.py`` applies)::

        $ feature-selection -i trainingBoolean.dat -u y -l INFO -C2 \
              -a smd -f x1 -f nosuch -r 1
        [INFO] feature-selection version "3"."7"."0"
        [INFO] Command line: <argv>
        [INFO] Read input file <path>
        [WARN] No such a feature nosuch in file <path>. It will be ignored as initial feature.
        [INFO] The search will start with the following feature set: x1
        [INFO] 2 out of 3 features have been selected
        [INFO] log_selected_features(): x1 0.0487949
        [INFO] log_selected_features(): x3 0.548795

    This is the ONLY test that reaches ``initial_features``' two log
    records and ``log_selected_features``' per-feature line, and it was
    added because mutation testing found them uncovered: deleting the
    ``initial_features`` logging killed ZERO tests before this existed.

    The scores print at SIX significant digits, not eighteen: a bare
    ``double`` streamed into a log record goes through a fresh
    ``std::stringstream`` at an ``ostream``'s default precision
    (``ostream_double``), and 0.0487949 vs 0.04879494069539869 is exactly
    that difference.
    """
    log_file = tmp_path / "fs.log"
    rc, _out, _err = _run(
        capsys, "-i", _BOOL, "-u", "y", "-l", "INFO", "-C2",
        "-a", "smd", "-f", "x1", "-f", "nosuch", "-r", "1",
        "-F", str(log_file),
    )
    assert rc == 0
    parsed = [
        _RECORD_RE.match(line).groups()
        for line in log_file.read_text().splitlines()
        if _RECORD_RE.match(line)
    ]
    levels = [level for level, _msg in parsed]
    records = [msg for _level, msg in parsed]
    assert levels == ["INFO", "INFO", "INFO", "WARN", "INFO", "INFO", "INFO", "INFO"]
    assert records[0] == 'feature-selection version "3"."7"."0"'
    assert records[1].startswith("Command line: feature-selection -i ")
    assert records[2] == f"Read input file {_BOOL}"
    assert records[3] == (
        f"No such a feature nosuch in file {_BOOL}. It will be ignored as initial feature."
    )
    assert records[4] == "The search will start with the following feature set: x1"
    assert records[5] == "2 out of 3 features have been selected"
    assert records[6] == "log_selected_features(): x1 0.0487949"
    assert records[7] == "log_selected_features(): x3 0.548795"


@pytest.mark.oracle
def test_initial_features_logs_the_empty_set_line(capsys, tmp_path):
    """The other arm of ``initial_features``' log (feature-selection.cc:126):
    with no ``-f`` at all, classic writes "The search will start with the
    empty feature set". Same mutation gap as the test above.
    """
    log_file = tmp_path / "fs.log"
    _run(capsys, "-i", _BOOL, "-u", "y", "-l", "INFO", "-C2", "-a", "smd",
         "-r", "1", "-F", str(log_file))
    assert "The search will start with the empty feature set" in log_file.read_text()
