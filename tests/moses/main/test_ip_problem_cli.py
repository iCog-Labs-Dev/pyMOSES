"""Byte-for-byte CLI oracles for `-H ip` (`ip_problem`,
table-problems.cc:229).

Every `_EXPECTED_*` string below is classic's OWN stdout, captured with

    ( ulimit -v 2097152; ulimit -c 0; timeout 300 \
      $CLASSIC/build/moses/moses/main/moses <argv> )

on 2026-09-12, through a `bash` script taking a `"$@"` ARRAY -- never an
inline zsh loop over an `$args` string, which has reported false agreement
three times in this project (docs/CARRY-FORWARD.md Sec 5.10). Each case
asserts a NON-EMPTY expectation for the same reason: two empty captures
compare equal, and `-H ip` at classic's default bounds prints a single
short line, which is exactly the shape a vacuous oracle would take.

Reachability, since it was in doubt. `ip_problem::run` OC_ASSERTs that
the table's signature is exactly `->(boolean ... boolean) contin`
(table-problems.cc:236-239) and classic ABORTS (SIGABRT, rc 134) on
anything else -- measured on `tests/fixtures/table_data/dataset.csv`,
whose output column is boolean. On a conforming table it exits 0. At
classic's DEFAULT activation bounds it prints one `nan true` line,
because `get_activation_penalty` divides 0 by 0 in both arms; that is
what the 16 `--ip_*_weight` rows of this project's value-aware CLI sweep
see, and those 16 argv are replayed verbatim below. Give it real bounds
and it prints a normal ranked list, which is what the rest of this file
pins.
"""
# ruff: noqa: E501

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymetta.moses.main.moses_exec import moses_exec

_TABLE_DATA = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_SWEEP_CSV = str(_TABLE_DATA / "ip_sweep.csv")
_SKEWED_CSV = str(_TABLE_DATA / "ip_boolean_contin.csv")
_SKEWED_REVERSED_CSV = str(_TABLE_DATA / "ip_boolean_contin_reversed.csv")
_BOOLEAN_OUTPUT_CSV = str(_TABLE_DATA / "dataset.csv")


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# The 16 sweep rows, verbatim.
# ---------------------------------------------------------------------------

# Captured: moses -i<ip_sweep.csv> -uy -m200 -r1 -W1 -Hip [--ip_*_weight=V]
# for every one of the 4 weights x 4 values the sweep uses. All 16 print
# the same single line, because the activation penalty is a NaN at the
# default bounds and a NaN dominates every composite score regardless of
# what the weights did -- which is precisely why these 16 rows, on their
# own, are a WEAK oracle and why this file does not stop here.
_EXPECTED_SWEEP_DEFAULT = "nan true \n"

_SWEEP_FLAGS = [
    f"--{flag}={value}"
    for flag in (
        "ip_kld_weight",
        "ip_skewness_weight",
        "ip_stdU_weight",
        "ip_skew_U_weight",
    )
    for value in ("0", "1", "2", "-1")
]


def test_sweep_flag_corpus_is_the_sweep_s_own_sixteen():
    """The sweep's `-Hip` rows are 4 flags x 4 values
    (`tools/flag_sweep_spec.py`). Pin the count so a truncated corpus
    cannot pass silently.
    """
    assert len(_SWEEP_FLAGS) == 16


@pytest.mark.oracle
@pytest.mark.parametrize("flag", _SWEEP_FLAGS)
def test_sweep_rows_match_classic(flag):
    """The 16 rows that were `NotImplementedError: problem 'ip'` before
    this port. Classic exits 0 on every one of them.
    """
    rc, out = _run(["moses", f"-i{_SWEEP_CSV}", "-uy", "-m200", "-r1", "-W1", "-Hip", flag])
    assert rc == 0
    assert out == _EXPECTED_SWEEP_DEFAULT
    assert out.strip(), "empty capture -- two empty outputs compare equal"


@pytest.mark.oracle
def test_sweep_row_without_any_weight_flag_matches_classic():
    rc, out = _run(["moses", f"-i{_SWEEP_CSV}", "-uy", "-m200", "-r1", "-W1", "-Hip"])
    assert rc == 0
    assert out == _EXPECTED_SWEEP_DEFAULT


# ---------------------------------------------------------------------------
# Real activation bounds: the regime where the scorer discriminates.
# ---------------------------------------------------------------------------

# Captured: moses -i<ip_boolean_contin.csv> -uy -m300 -r1 -W1 -Hip -q0.2 -w0.8
_EXPECTED_SKEWED = (
    "2.37031865119934082 and(!$b !$c) \n"
    "1.26839280128479004 and(or($a !$b) !$c) \n"
    "0.379831314086914062 and(!$a !$c) \n"
    "-0.107175886631011963 !$c \n"
    "-0.880655765533447266 $b \n"
    "-0.732480168342590332 and($a !$c) \n"
    "-0.447070300579071045 and(or($b !$c) !$a) \n"
    "-1.2037283182144165 !$a \n"
    "-1.00400936603546143 or(!$a $b) \n"
    "-1.30640280246734619 !$b \n"
)

# Captured: the same, plus -K2 -J3 -U0.5 -X-1. A COMPLETELY different
# candidate list, which is what makes it the proof that the four weights
# reach the scorer rather than being parsed and dropped.
_EXPECTED_SKEWED_WEIGHTED = (
    "-2.01163816452026367 or($b !$c) \n"
    "-2.01163816452026367 or(and(!$a !$c) $b) \n"
    "-2.58383893966674805 $b \n"
    "-2.39732742309570312 or(!$a $b) \n"
    "-2.01163816452026367 or(and(or(!$a $b) !$c) $b) \n"
    "-2.01163816452026367 or(and(or($a $c) $b) !$c) \n"
    "-2.01163816452026367 or(and(or($a $c) $b) and(!$a !$c)) \n"
    "-2.58383893966674805 or(and($a !$c) $b) \n"
    "-2.6083219051361084 or(and($a $b) !$c) \n"
    "-2.39732742309570312 or(and(or($a $c) $b) !$a) \n"
)

# Captured: the same, plus --pre-positive 0. Every tree is the De Morgan
# dual of the positive run's and every score is unchanged -- the predicate
# is inverted, so the SAME rows are selected by the complementary tree.
_EXPECTED_SKEWED_NEGATED = (
    "2.37031865119934082 or($b $c) \n"
    "1.26839280128479004 or(and(!$a $b) $c) \n"
    "0.379831314086914062 or($a $c) \n"
    "-0.107175886631011963 $c \n"
    "0.379831314086914062 or(and($a $b) $c) \n"
    "0.379831314086914062 or(and(or($a $c) $b) $c) \n"
    "-0.107175886631011963 or(and($a !$b) $c) \n"
    "0.379831314086914062 or(and(or(!$a $b) or($a $c)) $c) \n"
    "0.379831314086914062 or(and(or($a $c) or(!$b !$c)) $c) \n"
    "-0.880655765533447266 !$b \n"
)

# Captured: moses -i<ip_sweep.csv> -uy -m300 -r5 -W1 -Hip -q0.3 -w0.7.
# A different table, seed and bounds, so nothing here can be an artefact
# of the one fixture above.
_EXPECTED_SWEEP_REAL_BOUNDS = (
    "0.70849233865737915 $a1 \n"
    "0.551896929740905762 !$a1 \n"
    "0.747367620468139648 and($a1 $a2) \n"
    "0.936396241188049316 and(or($a2 !$a3) $a1) \n"
    "0.559358000755310059 and(!$a1 !$a2) \n"
    "0.296547591686248779 and($a1 !$a3) \n"
    "-0.147346973419189453 $a2 \n"
    "0.123260810971260071 and(!$a1 $a3) \n"
    "-0.233990371227264404 !$a2 \n"
    "0.261445820331573486 or(and($a2 !$a3) !$a1) \n"
)


@pytest.mark.oracle
def test_real_activation_bounds_match_classic():
    rc, out = _run(
        ["moses", f"-i{_SKEWED_CSV}", "-uy", "-m300", "-r1", "-W1", "-Hip", "-q0.2", "-w0.8"]
    )
    assert rc == 0
    assert out == _EXPECTED_SKEWED


@pytest.mark.oracle
def test_ip_weights_reach_the_scorer():
    """`-K2 -J3 -U0.5 -X-1` must change the output. Before `-H ip` was
    wired, `ip_problem_params` was parsed into an object no problem ever
    read; this is the case that would catch a regression to that -- it
    asserts BOTH that the weighted run matches classic AND that it
    differs from the default-weight run.
    """
    rc, out = _run(
        [
            "moses",
            f"-i{_SKEWED_CSV}",
            "-uy",
            "-m300",
            "-r1",
            "-W1",
            "-Hip",
            "-q0.2",
            "-w0.8",
            "-K2",
            "-J3",
            "-U0.5",
            "-X-1",
        ]
    )
    assert rc == 0
    assert out == _EXPECTED_SKEWED_WEIGHTED
    assert out != _EXPECTED_SKEWED


@pytest.mark.oracle
def test_pre_positive_zero_matches_classic():
    """`--pre-positive 0` inverts the predicate (`bool_to_vertex(_positive)`,
    bscores.cc:774).
    """
    rc, out = _run(
        [
            "moses",
            f"-i{_SKEWED_CSV}",
            "-uy",
            "-m300",
            "-r1",
            "-W1",
            "-Hip",
            "-q0.2",
            "-w0.8",
            "--pre-positive",
            "0",
        ]
    )
    assert rc == 0
    assert out == _EXPECTED_SKEWED_NEGATED
    assert out != _EXPECTED_SKEWED


@pytest.mark.oracle
def test_second_table_and_seed_match_classic():
    rc, out = _run(
        ["moses", f"-i{_SWEEP_CSV}", "-uy", "-m300", "-r5", "-W1", "-Hip", "-q0.3", "-w0.7"]
    )
    assert rc == 0
    assert out == _EXPECTED_SWEEP_REAL_BOUNDS


@pytest.mark.oracle
def test_reversed_rows_produce_the_same_output_as_classic():
    """`ip_boolean_contin_reversed.csv` holds the same 14 observations in
    reverse file order. Classic walks the CTable in `std::map` order, so
    its output is unchanged; a port that walked rows in FILE order would
    differ, because `pred_counter`'s ASSIGNMENT (bscores.cc:800) lets the
    last-visited duplicate output value win.

    MUTATION PROVEN: replacing `_ctable_rows_in_map_order`'s `sorted(...)`
    with a bare `list(ctable.items())` makes this case fail while leaving
    `test_real_activation_bounds_match_classic` passing -- the forward
    file happens to be close enough to sorted order that only the
    reversed twin exposes it.
    """
    rc, out = _run(
        [
            "moses",
            f"-i{_SKEWED_REVERSED_CSV}",
            "-uy",
            "-m300",
            "-r1",
            "-W1",
            "-Hip",
            "-q0.2",
            "-w0.8",
        ]
    )
    assert rc == 0
    assert out == _EXPECTED_SKEWED


def test_expected_outputs_are_distinct_and_non_empty():
    """The oracle corpus must SPREAD. Four different expectations, none
    empty, none equal to another -- otherwise a scorer that ignored its
    inputs could satisfy the whole file.
    """
    expectations = [
        _EXPECTED_SWEEP_DEFAULT,
        _EXPECTED_SKEWED,
        _EXPECTED_SKEWED_WEIGHTED,
        _EXPECTED_SKEWED_NEGATED,
        _EXPECTED_SWEEP_REAL_BOUNDS,
    ]
    assert all(e.strip() for e in expectations)
    assert len(set(expectations)) == len(expectations)
    # ... and the discriminating ones must each rank TEN candidates, not
    # collapse to the one-line `nan` shape.
    for expected in expectations[1:]:
        assert len(expected.splitlines()) == 10


# ---------------------------------------------------------------------------
# The OC_ASSERT gate.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_boolean_output_table_is_rejected_with_classics_message():
    """Measured: classic ABORTS on `dataset.csv` (boolean output column)
    with rc 134 and

        terminate called after throwing an instance of
        'opencog::AssertionException'
          what():  The input table doesn't have the right data types. The
          output should be contin and the inputs should be boolean
          (table-problems.cc:236)

    An `OC_ASSERT(false, ...)` is an `AssertionError` here, per this
    project's convention, carrying classic's own text verbatim.
    """
    with pytest.raises(AssertionError) as excinfo:
        _run(["moses", f"-i{_BOOLEAN_OUTPUT_CSV}", "-m200", "-r1", "-W1", "-Hip"])
    assert "The input table doesn't have the right data types" in str(excinfo.value)
    assert "output should be contin" in str(excinfo.value)
    assert "inputs should be boolean" in str(excinfo.value)


@pytest.mark.oracle
def test_boosting_is_rejected_with_classics_message():
    """`OC_ASSERT(not pms.meta_params.do_boosting, "Boosting not supported
    for the ip problem!")` (table-problems.cc:263-264). Classic aborts;
    this port raises the same text.
    """
    with pytest.raises(AssertionError, match="Boosting not supported for the ip problem!"):
        _run(
            [
                "moses",
                f"-i{_SWEEP_CSV}",
                "-uy",
                "-m200",
                "-r1",
                "-W1",
                "-Hip",
                "--boost=1",
            ]
        )


def test_ip_is_no_longer_a_raising_stub():
    """The regression this whole port exists to prevent: `-H ip` used to
    raise `NotImplementedError: problem 'ip' ... is not ported` from
    `problem.py`, on a command line classic exits 0 on.
    """
    rc, out = _run(["moses", f"-i{_SWEEP_CSV}", "-uy", "-m200", "-r1", "-W1", "-Hip"])
    assert rc == 0
    assert out
