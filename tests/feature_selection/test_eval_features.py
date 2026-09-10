"""Tests for ``pymetta.feature_selection.main.eval_features`` (the
standalone ``eval-features`` binary).

## Oracle

Every expected value here came from running
``$CLASSIC/build/moses/feature-selection/main/eval-features`` directly,
bounded by ``( ulimit -v 2097152; timeout 60 ... )``, through a bash
harness using a ``"$@"`` array. That harness form is not optional:
docs/CARRY-FORWARD.md 5.10 records four wrong readings in this project
caused by zsh NOT word-splitting an unquoted ``$args``, and it happened
again while probing THIS binary -- an inline ``$a`` loop made classic
report ``No such a feature  x1 -f x2`` (one argument), which reads like a
port bug and is a harness bug. The values below are from the rewritten
bash version.

22 invocations were compared; 18 were byte-identical on stdout. The four
that were not are all accounted for, and each has a test here:

* ``-f y`` (the LAST label) indexes past the end of every row --
  undefined behaviour in classic, which printed ``0`` on
  ``trainingBoolean.dat`` and ``0.0635228`` on ``bool_a5_maj.csv`` (three
  runs each, stable). This port raises.
* ``-f nosuch`` and a bad ``-i``: classic's ``OC_ASSERT`` throws, nothing
  catches it, and the process dies with SIGABRT (exit 134). This port
  raises ``AssertionError`` with the SAME message text; only the exit
  mechanism differs.
* no arguments at all: boost's help formatter vs argparse's (a
  documented divergence; exit code 1 matches).

Measured on ``trainingBoolean.dat`` (``x1,x2,x3,y``), all with the
default ``-d 0``::

    -f x1              -> 0
    -f x2              -> 0
    -f x3              -> 0.0433733
    -f x1 -f x2        -> 0
    -f x2 -f x3        -> 0.138346
    -f x1 -f x2 -f x3  -> 0.222222
    -f x2 -f x3 -d 1   -> 0.15305
    -f x2 -f x3 -d 100 -> 0.155639
    (no -f at all)     -> "" (one bare newline)
"""

from pathlib import Path

import pytest

from pymetta.feature_selection.main.eval_features import (
    eval_features_main,
    eval_features_parameters,
    get_features_idx,
    output_results,
)

_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_BOOL = str(_TABLE_DATA_DIR / "trainingBoolean.dat")
_MAJ5 = str(_TABLE_DATA_DIR / "bool_a5_maj.csv")


def _run(capsys, *args: str) -> tuple[int, str, str]:
    rc = eval_features_main(["eval-features", *args])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("args", "want"),
    [
        (["-f", "x1"], "0\n"),
        (["-f", "x2"], "0\n"),
        (["-f", "x3"], "0.0433733\n"),
        (["-f", "x1", "-f", "x2"], "0\n"),
        (["-f", "x2", "-f", "x3"], "0.138346\n"),
        (["-f", "x1", "-f", "x2", "-f", "x3"], "0.222222\n"),
        (["-f", "x2", "-f", "x3", "-d", "1"], "0.15305\n"),
        (["-f", "x2", "-f", "x3", "-d", "100"], "0.155639\n"),
        (["-f", "x3", "-d", "-5"], "0.00249569\n"),
        # No -f at all: `ostream_container`'s `empty_lr` defaults to true,
        # so `left`+`right` are printed for an EMPTY container -> one "\n".
        ([], "\n"),
        # A repeated feature is a std::set, so it collapses to one index --
        # same output as `-f x3` alone, NOT a two-element result.
        (["-f", "x3", "-f", "x3"], "0.0433733\n"),
    ],
)
def test_eval_features_stdout_is_byte_identical_to_classic(capsys, args, want):
    rc, out, err = _run(capsys, "-i", _BOOL, *args)
    assert (rc, err) == (0, "")
    assert out == want


@pytest.mark.oracle
@pytest.mark.parametrize("dead", [["-u", "y"], ["-Y", "x1"], ["-H", "pre"]])
def test_the_three_dead_options_change_nothing(capsys, dead):
    """``-u``, ``-Y`` and ``-H`` are all parsed and never read (this
    module's own docstring, defect 1 and the ``-H`` note). Classic prints
    the SAME 0.0433733 with and without each of them -- measured. A port
    that honoured ``-u y`` would change the target column and every
    number with it, so this is the test that pins the defect rather than
    accidentally fixing it. DO NOT FIX.
    """
    rc, out, err = _run(capsys, "-i", _BOOL, *dead, "-f", "x3")
    assert (rc, err) == (0, "")
    assert out == "0.0433733\n"


@pytest.mark.oracle
def test_feature_names_resolve_one_column_to_the_right(capsys):
    """The off-by-one that defines this binary's behaviour.

    With no ``-u``, the target is the FIRST column (``x1``), so the
    inputs are ``[x2, x3, y]`` and ``get_labels()`` is ``[x1, x2, x3, y]``.
    ``-f x3`` therefore scores the INPUT column ``y``, whose mutual
    information with the target ``x1`` is 0.0488; times ``MICScorer``'s
    confidence factor ``8/(8+exp(0)) == 8/9`` that is 0.0433733 -- the
    number classic prints. Both other single-feature names score 0
    because ``x2`` and ``x3`` are independent of ``x1`` in this table.

    Asserted here as the arithmetic identity, not just as the observed
    string, so that a future reader can see the defect is understood and
    not merely recorded.
    """
    _rc, out, _err = _run(capsys, "-i", _BOOL, "-f", "x3")
    assert float(out) == pytest.approx(0.04879494069539869 * (8.0 / 9.0), rel=1e-6)


def test_get_features_idx_message_matches_classic(capsys):
    """C++: eval-features.h:92's ``OC_ASSERT``. Classic dies with SIGABRT
    (exit 134) printing this exact ``what()``; the port raises with the
    same text. Only the mechanism differs -- a C++ ``terminate`` has no
    Python equivalent.
    """
    pa = eval_features_parameters(input_file="/some/file.csv")
    with pytest.raises(AssertionError) as excinfo:
        get_features_idx(["nosuch"], ["a", "b"], pa)
    assert str(excinfo.value) == "No such a feature nosuch in file /some/file.csv"


def test_the_last_label_raises_rather_than_reading_past_the_end(capsys):
    """``-f y`` -> index 3 on a 3-column input table. classic's
    ``seq_filtered`` (opencog/util/algorithm.h:352) is a bare
    ``seq[idx]``, so it reads past the end: measured 0 on
    ``trainingBoolean.dat`` and 0.0635228 on ``bool_a5_maj.csv``, stable
    across three runs each -- i.e. genuinely data-dependent garbage, not
    a benign zero. Undefined behaviour is not reproducible
    (docs/CARRY-FORWARD.md section 3), so this raises.

    Both tables are exercised, because the ``trainingBoolean.dat`` case
    alone reads as an innocuous 0 and would let a "return 0.0 on
    out-of-range" implementation pass.
    """
    for table in (_BOOL, _MAJ5):
        with pytest.raises(IndexError, match="past the end"):
            _run(capsys, "-i", table, "-f", "y")


@pytest.mark.oracle
def test_output_file_is_opened_in_append_mode(capsys, tmp_path):
    """``ofstream of(pa.output_file.c_str(), ios_base::app)``,
    eval-features.h:81 -- unlike ``saveTable``, which truncates. Two runs
    with the same ``-o`` must leave TWO lines.
    """
    out_file = tmp_path / "out.csv"
    for _ in range(2):
        rc, out, err = _run(capsys, "-i", _BOOL, "-f", "x3", "-o", str(out_file))
        assert (rc, out, err) == (0, "", "")
    assert out_file.read_text() == "0.0433733\n0.0433733\n"


@pytest.mark.oracle
def test_features_file_gives_one_result_per_non_empty_line(capsys, tmp_path):
    """``--features-file``: one feature SET per line, results
    space-separated on one output line. Blank lines are skipped
    (eval-features.h:109-110). The line tokenizer is ``tokenizeRow``, so
    commas, spaces and tabs all separate -- the third line below uses a
    tab and must parse the same as the comma-separated first.
    """
    ff = tmp_path / "fss.txt"
    ff.write_text("x2,x3\n\nx2\tx3\nx3\n")
    rc, out, err = _run(capsys, "-i", _BOOL, "-F", str(ff))
    assert (rc, err) == (0, "")
    assert out == "0.138346 0.138346 0.0433733\n"


@pytest.mark.oracle
def test_feature_and_features_file_are_both_used(capsys, tmp_path):
    """``-f`` contributes ONE set (all its values together) and the file
    contributes one per line, ``-f``'s first -- eval-features.h:104-115.
    """
    ff = tmp_path / "fss.txt"
    ff.write_text("x3\n")
    _rc, out, _err = _run(capsys, "-i", _BOOL, "-f", "x2", "-f", "x3", "-F", str(ff))
    assert out == "0.138346 0.0433733\n"


def test_output_results_writes_a_bare_newline_for_an_empty_result(capsys):
    """The ``empty_lr`` default again, isolated from the CLI."""
    output_results(eval_features_parameters(), [])
    assert capsys.readouterr().out == "\n"


def test_eval_features_with_no_arguments_returns_one(capsys):
    """C++: eval-features.cc:101-104 -- ``return 1`` for a successful
    ``--help``. The help TEXT is a documented divergence (boost's
    formatter is not ported); only the return code is asserted.
    """
    rc, out, _err = _run(capsys)
    assert rc == 1
    assert "usage:" in out
