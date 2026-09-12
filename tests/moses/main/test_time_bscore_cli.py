r"""`--time-bscore` and `--time-dispersion-pressure` at the CLI.

Both flags were gated in `problem_params.py` until this slice: classic
accepted them and this port refused to start. They run now, and these are
the end-to-end oracles for lifting the gates.

Every expected string is classic's own stdout, captured verbatim from

    ( ulimit -v 2097152; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses \
        -Hpre -i tests/fixtures/table_data/<file> -u <target> \
        [--timestamp-feature=when] -q0.5 -w1.0 -m<M> -W1 -r<R> <flags> )

## Where the byte-for-byte comparison is valid, and where it is not

`TimedValue::operator<` (table.h:482) is not a strict weak ordering, so
classic's `std::map`-based CTable cells are undefined behaviour on some
timestamped data (see `TimedCounter`'s class docstring in table.py). Two
whole families of input are nonetheless immune, and both are used here:

1. **Untimestamped tables.** Every observation carries the same not-a-date
   sentinel, so the comparator's timestamp term never decides anything.
   These are the broadest oracle: `--time-dispersion-pressure` still runs
   the whole dispersion pipeline, with `_Hmax == log2(1) == 0`.
2. **Timestamped tables whose value order and date order agree.**
   `timestamp_bool_clean.csv` and `timestamp_bool_months.csv` were built
   that way on purpose -- within every compressed row, every `true`
   observation is dated before every `false` one.

`timestamp_bool_dup.csv` is deliberately the opposite, and classic's
output on it differs from this port's for reasons that predate these two
flags; it appears here only in the one place where a NaN swallows the
difference.

## What is NOT compared byte for byte, and why

`--time-dispersion-pressure=0.5` at the default `day` granularity makes
classic **SIGSEGV** (rc 139, no diagnostic) on `timestamp_bool_clean.csv`
and `timestamp_bool_months.csv` -- all 18 (file, -m, -r) combinations
tried. The fault is not in the dispersion arithmetic; gdb puts it in
`representation::get_candidate_rec`'s tree-node allocation, downstream of
the heap damage done by `ctable_res`, the `TimedValue`-keyed map
`precision_bscore::do_score` builds (precision_bscore.cc:261). This port
computes the intended result and completes. There is no classic output to
compare against, so those tests assert only that the port produces a
well-formed answer.
"""

from pathlib import Path

import pytest

from pymoses.moses.main.moses_exec import moses_exec

_TABLE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"


def _run(capsys, csv: str, target: str, extra: list[str], m="-m300", seed="-r1", timed=False):
    argv = ["moses", "-Hpre", f"-i{_TABLE_DIR / csv}", "-u", target]
    if timed:
        argv.append("--timestamp-feature=when")
    argv += ["-q0.5", "-w1.0", m, "-W1", seed, *extra]
    assert moses_exec(argv) == 0
    out = capsys.readouterr().out
    assert out, "empty stdout would make every comparison below vacuous"
    return out


# ---------------------------------------------------------------------------
# --time-dispersion-pressure on UNTIMESTAMPED tables.
#
# One timestamp class, so _Hmax is log2(1) == 0 and the penalty is
# degenerate -- but WHICH degenerate value depends on the exponent's parity,
# because `H / _Hmax` is x86-64's negative-signed 0.0f/0.0f and glibc's
# powf clears that sign only for an ODD-INTEGER exponent. Classic prints
# `nan` at the default exponent 1.0 and `-nan` at 2.0 and 0.5. That single
# character is what `_powf` (time_dispersion.py) exists for.
# ---------------------------------------------------------------------------

_UNTIMED = [
    ("precision.csv", "target"),
    ("imbalanced_bool.csv", "y"),
    ("trainingBoolean.dat", "y"),
    ("trainingBooleanXor.dat", "y"),
    ("bool_a4_par.csv", "y"),
    ("bool_a5_maj.csv", "y"),
    ("weighted.csv", "r"),
]


@pytest.mark.oracle
@pytest.mark.parametrize(("csv", "target"), _UNTIMED, ids=lambda v: str(v))
@pytest.mark.parametrize("m", ["-m300", "-m1000"])
@pytest.mark.parametrize("seed", ["-r1", "-r5"])
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--time-dispersion-pressure=0.5"], "nan"),
        (["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=2.0"], "-nan"),
        (["--time-dispersion-pressure=0.25", "--time-dispersion-exponent=0.5"], "-nan"),
        (["--time-bscore=1", "--time-dispersion-pressure=0.5"], "nan"),
    ],
    ids=["e1", "e2", "e05", "time_bscore_too"],
)
def test_dispersion_pressure_on_untimestamped_tables_matches_classic(
    capsys, csv, target, m, seed, flags, expected
):
    """56 of the 168 (dataset x budget x seed x flags) runs this file's
    harness compared; all 168 were byte-identical to classic.

    The whole metapopulation collapses to a single NaN-scored candidate,
    so the expected stdout is `<sign>nan true \\n` for every one of them.
    """
    out = _run(capsys, csv, target, flags, m=m, seed=seed)
    assert out == f"{expected} true \n"


@pytest.mark.oracle
@pytest.mark.parametrize(("csv", "target"), _UNTIMED, ids=lambda v: str(v))
def test_dispersion_pressure_changes_the_answer_on_untimestamped_tables(capsys, csv, target):
    """Non-vacuity guard for the parametrisation above: without the flag
    these datasets produce ordinary finite scores, so the NaN really is the
    penalty arriving, not a table that was degenerate to begin with.
    """
    without = _run(capsys, csv, target, [])
    assert "nan" not in without
    assert without != _run(capsys, csv, target, ["--time-dispersion-pressure=0.5"])


# precision.csv, -q0.5 -w1.0 -m300 -W1 -r1, no time flags at all.
_PRECISION_PLAIN = (
    "0.5 true \n"
    "0.714285731315612793 !$v1 \n"
    "0.489174425601959229 $v2 \n"
    "0.714285731315612793 or(!$v1 $v2) \n"
    "0.285714268684387207 !$v2 \n"
    "0.5 or(!$v1 !$v2) \n"
    "0.5 or($v1 $v2) \n"
    "0.489174425601959229 and(!$v1 $v2) \n"
    "0.285714268684387207 or($v1 !$v2) \n"
    "0.276856482028961182 and(!$v1 !$v2) \n"
)

# ... plus --time-dispersion-pressure=0.5 --time-dispersion-exponent=-1.0.
_PRECISION_EXP_MINUS_ONE = (
    "0.714285731315612793 !$v1 \n"
    "0.489174425601959229 $v2 \n"
    "0.714285731315612793 or(!$v1 $v2) \n"
    "0.285714268684387207 !$v2 \n"
    "0.5 or($v1 $v2) \n"
    "0.489174425601959229 and(!$v1 $v2) \n"
    "0.285714268684387207 or($v1 !$v2) \n"
    "0.276856482028961182 and(!$v1 !$v2) \n"
    "-0.510825574398040771 $v1 \n"
    "-0.510825574398040771 and($v1 !$v2) \n"
)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        # exponent 0.0: `pow(x, +-0) == 1` for EVERY x, NaN included, so the
        # penalty is `pressure * 1.0` and the score is finite. `_powf`
        # answers this before its NaN rule, exactly as C99 and glibc do.
        (["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=0.0"], "1 true \n"),
        # Odd-integer exponents clear the NaN's sign, even and fractional
        # ones keep it. 1e30 is an EVEN integer, and is the row that pins
        # that the parity test is on the VALUE, not on "is it small".
        (["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=3.0"], "nan true \n"),
        (["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=0.25"], "-nan true \n"),
        (["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=1e30"], "-nan true \n"),
        # A negative exponent, where Python's math.pow would raise
        # OverflowError on some bases and C returns a signed infinity.
        (
            ["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=-1.0"],
            _PRECISION_EXP_MINUS_ONE,
        ),
        (
            ["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=-1e30"],
            _PRECISION_EXP_MINUS_ONE,
        ),
        # `if (_pressure > 0.0)` is STRICT: 0.0 and a NEGATIVE pressure are
        # both inert, and give the plain no-flag answer.
        (["--time-dispersion-pressure=0.0"], _PRECISION_PLAIN),
        (["--time-dispersion-pressure=-0.5"], _PRECISION_PLAIN),
        (["--time-dispersion-pressure=1e10"], "nan true \n"),
        (["--time-dispersion-pressure=1.0"], "nan true \n"),
    ],
    ids=["e0", "e3", "e0.25", "e1e30", "e-1", "e-1e30", "p0", "p-0.5", "p1e10", "p1"],
)
def test_dispersion_exponent_and_pressure_edges_match_classic(capsys, flags, expected):
    """The exponent/pressure boundary rows, on an untimestamped table where
    classic is well defined. Between them these cover every arm of `_powf`
    that a CLI can reach: the exponent-zero early return, the odd-integer
    sign clear, the even-integer and fractional sign keep, and a negative
    exponent.
    """
    assert _run(capsys, "precision.csv", "target", flags) == expected


def test_dispersion_exponent_edges_are_not_all_the_same_answer():
    """Non-vacuity guard: the rows above must not all collapse onto one
    expected string, or the parametrisation would prove nothing about the
    exponent at all.
    """
    assert len({"1 true \n", "nan true \n", "-nan true \n", _PRECISION_PLAIN}) == 4


@pytest.mark.oracle
@pytest.mark.parametrize(("csv", "target"), _UNTIMED, ids=lambda v: str(v))
def test_time_bscore_is_inert_on_untimestamped_tables(capsys, csv, target):
    """With no timestamp column every observation shares the sentinel date,
    so `time2res` has exactly one bucket and the time-spread bscore is the
    whole-table sum -- classic's stdout is unchanged by the flag. Measured
    over all 7 datasets x 2 budgets x 2 seeds.
    """
    plain = _run(capsys, csv, target, [])
    assert _run(capsys, csv, target, ["--time-bscore=1"]) == plain


@pytest.mark.oracle
@pytest.mark.parametrize(("csv", "target"), _UNTIMED, ids=lambda v: str(v))
def test_month_granularity_without_a_timestamp_column_fails_as_classic_does(capsys, csv, target):
    """`get_timestamp_class`'s `month` arm OC_ASSERTs the timestamp is a
    real date (time_dispersion.cc:62), and the constructor evaluates it once
    per observation -- so it fires with or without a pressure, on the first
    observation of any table loaded without `--timestamp-feature`.

    Measured 2026-09-10: classic exits 134 on all 7 of these datasets with

        opencog::AssertionException ... timestamp is not a valid date, are
        you sure you have loaded a timestamp feature
        (option --timestamp-feature)?

    An earlier report in this project recorded this as not reproducing. It
    does.
    """
    with pytest.raises(AssertionError, match="timestamp is not a valid date"):
        moses_exec(
            [
                "moses",
                "-Hpre",
                f"-i{_TABLE_DIR / csv}",
                "-u",
                target,
                "-q0.5",
                "-w1.0",
                "-m300",
                "-W1",
                "-r1",
                "--time-dispersion-pressure=0.5",
                "--time-bscore-granularity=month",
            ]
        )
    capsys.readouterr()


# ---------------------------------------------------------------------------
# --time-bscore on TIMESTAMPED tables.
# ---------------------------------------------------------------------------

# -Hpre -i timestamp_bool_months.csv -u y --timestamp-feature=when
# -q0.5 -w1.0 -m300 -W1 -r1, with no time flags.
_MONTHS_PLAIN = (
    "0.555555582046508789 true \n"
    "0.800000011920928955 !$i1 \n"
    "0.632216989994049072 $i2 \n"
    "0.714285731315612793 or(!$i1 $i2) \n"
    "0.400000005960464478 !$i2 \n"
    "0.571428596973419189 or(!$i1 !$i2) \n"
    "0.5 or($i1 $i2) \n"
    "0.428571432828903198 or($i1 !$i2) \n"
    "0.132216975092887878 $i1 \n"
    "0.261201620101928711 and(!$i1 !$i2) \n"
)

# ... the same run with --time-bscore=1. `or($i1 !$i2)` moves.
_MONTHS_TIME_BSCORE = _MONTHS_PLAIN.replace(
    "0.428571432828903198 or($i1 !$i2)", "0.428571403026580811 or($i1 !$i2)"
)

_MONTHS_DIVERSITY = (
    "0.555555582046508789 true \n"
    "0.800000011920928955 !$i1 \n"
    "0.632216989994049072 $i2 \n"
    "0.400000005960464478 !$i2 \n"
    "0.714285731315612793 or(!$i1 $i2) \n"
    "0.571428596973419189 or(!$i1 !$i2) \n"
    "0.189069747924804688 and(!$i1 $i2) \n"
    "0.5 or($i1 $i2) \n"
    "0.132216975092887878 $i1 \n"
    "0.261201620101928711 and(!$i1 !$i2) \n"
)

_MONTHS_DIVERSITY_TIME_BSCORE = (
    "0.555555582046508789 true \n"
    "0.800000011920928955 !$i1 \n"
    "0.632216989994049072 $i2 \n"
    "0.714285731315612793 or(!$i1 $i2) \n"
    "0.400000005960464478 !$i2 \n"
    "0.571428596973419189 or(!$i1 !$i2) \n"
    "0.189069747924804688 and(!$i1 $i2) \n"
    "0.132216975092887878 $i1 \n"
    "0.5 or($i1 $i2) \n"
    "0.261201620101928711 and(!$i1 !$i2) \n"
)


@pytest.mark.oracle
@pytest.mark.parametrize("m", ["-m100", "-m300", "-m1000"])
@pytest.mark.parametrize("seed", ["-r1", "-r7", "-r42"])
def test_time_bscore_matches_classic_on_the_discriminating_corpus(capsys, m, seed):
    """`timestamp_bool_months.csv` is the corpus where `--time-bscore`
    genuinely changes classic's stdout, so a port that ignored the flag
    would fail here. Both answers are reproduced, at three budgets and
    three seeds.
    """
    assert _run(capsys, "timestamp_bool_months.csv", "y", [], m=m, seed=seed, timed=True) == (
        _MONTHS_PLAIN
    )
    assert (
        _run(
            capsys,
            "timestamp_bool_months.csv",
            "y",
            ["--time-bscore=1"],
            m=m,
            seed=seed,
            timed=True,
        )
        == _MONTHS_TIME_BSCORE
    )


def test_time_bscore_actually_changes_the_output():
    """Non-vacuity guard for the parametrisation above. If these two ever
    became equal the oracle would still pass while proving nothing.
    """
    assert _MONTHS_PLAIN != _MONTHS_TIME_BSCORE
    assert _MONTHS_DIVERSITY != _MONTHS_DIVERSITY_TIME_BSCORE


@pytest.mark.oracle
def test_time_bscore_granularity_does_not_affect_the_time_bscore(capsys):
    """`time2res` is keyed on the RAW timestamp (precision_bscore.cc:247),
    not on `get_timestamp_class`, so `--time-bscore-granularity` cannot
    change a `--time-bscore` run even on a table that spans three months.
    Classic agrees: byte-identical with and without `=month`.
    """
    args = ["--time-bscore=1"]
    assert _run(capsys, "timestamp_bool_months.csv", "y", args, timed=True) == (_MONTHS_TIME_BSCORE)
    assert (
        _run(
            capsys,
            "timestamp_bool_months.csv",
            "y",
            [*args, "--time-bscore-granularity=month"],
            timed=True,
        )
        == _MONTHS_TIME_BSCORE
    )


@pytest.mark.oracle
def test_time_bscore_under_diversity_pressure_matches_classic(capsys):
    """`--diversity-pressure` compares whole bscore VECTORS, so it sees the
    regrouping even where the composite score does not. Classic reorders
    two candidates; the port reorders the same two.
    """
    csv = "timestamp_bool_months.csv"
    assert _run(capsys, csv, "y", ["--diversity-pressure=1.0"], timed=True) == _MONTHS_DIVERSITY
    assert (
        _run(capsys, csv, "y", ["--diversity-pressure=1.0", "--time-bscore=1"], timed=True)
        == _MONTHS_DIVERSITY_TIME_BSCORE
    )


# timestamp_bool_desc.csv: the corpus whose ENCOUNTER order of dates is the
# reverse of their calendar order (its first compressed row lists 2012-03-09,
# 2012-02-07, 2012-01-05). classic's `time2res` is a `std::map<date, ...>`,
# so its bscore components come out ascending by date regardless; this file
# is what stops the port getting away with encounter order. Every other
# timestamped corpus here encounters dates ascending, so the two coincide.
_DESC_DIVERSITY = (
    "0.666666686534881592 true \n"
    "0.800000011920928955 !$i2 \n"
    "0.632216989994049072 $i1 \n"
    "0.60000002384185791 !$i1 \n"
    "0.382216989994049072 $i2 \n"
    "0.714285731315612793 or(!$i1 !$i2) \n"
    "0.666666686534881592 or($i1 $i2) \n"
    "0.714285731315612793 or($i1 !$i2) \n"
    "0.189069747924804688 and($i1 !$i2) \n"
    "0.571428596973419189 or(!$i1 $i2) \n"
)

_DESC_DIVERSITY_TIME_BSCORE = (
    "0.666666686534881592 true \n"
    "0.800000011920928955 !$i2 \n"
    "0.632216989994049072 $i1 \n"
    "0.60000002384185791 !$i1 \n"
    "0.382216989994049072 $i2 \n"
    "0.714285731315612793 or(!$i1 !$i2) \n"
    "0.714285731315612793 or($i1 !$i2) \n"
    "0.666666686534881592 or($i1 $i2) \n"
    "0.189069747924804688 and($i1 !$i2) \n"
    "0.571428596973419189 or(!$i1 $i2) \n"
)


@pytest.mark.oracle
@pytest.mark.parametrize("m", ["-m100", "-m300", "-m1000"])
@pytest.mark.parametrize("seed", ["-r1", "-r7", "-r42"])
def test_time_bscore_on_the_reverse_date_order_corpus_matches_classic(capsys, m, seed):
    """45/45 byte-identical across 3 budgets x 3 seeds x 5 flag sets. The
    `--diversity-pressure` pair is the discriminating one: it compares whole
    bscore vectors, so it sees the time-spread regrouping (classic reorders
    `or($i1 $i2)` and `or($i1 !$i2)`), while the composite score alone does
    not change on this corpus.
    """
    csv = "timestamp_bool_desc.csv"
    assert (
        _run(capsys, csv, "y", ["--diversity-pressure=1.0"], m=m, seed=seed, timed=True)
        == _DESC_DIVERSITY
    )
    assert (
        _run(
            capsys,
            csv,
            "y",
            ["--diversity-pressure=1.0", "--time-bscore=1"],
            m=m,
            seed=seed,
            timed=True,
        )
        == _DESC_DIVERSITY_TIME_BSCORE
    )


def test_desc_corpus_diversity_pair_actually_differs():
    """Non-vacuity guard for the parametrisation above."""
    assert _DESC_DIVERSITY != _DESC_DIVERSITY_TIME_BSCORE


@pytest.mark.oracle
@pytest.mark.parametrize("m", ["-m100", "-m300", "-m1000"])
def test_time_bscore_is_score_invariant_on_the_clean_corpus(capsys, m):
    """The other half: on `timestamp_bool_clean.csv` the time-spread bscore
    sums to the same composite score, so classic's stdout is unchanged by
    the flag -- and so is the port's. A port that mangled the regrouping
    would break this even though it cannot break the sum.
    """
    csv = "timestamp_bool_clean.csv"
    plain = _run(capsys, csv, "y", [], m=m, timed=True)
    assert plain == _MONTHS_PLAIN  # the two corpora happen to agree here
    assert _run(capsys, csv, "y", ["--time-bscore=1"], m=m, timed=True) == plain


# ---------------------------------------------------------------------------
# --time-dispersion-pressure on TIMESTAMPED tables.
# ---------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("csv", ["timestamp_bool_clean.csv", "timestamp_bool_dup.csv"])
def test_dispersion_pressure_at_month_granularity_matches_classic(capsys, csv):
    """The one `--time-dispersion-pressure` configuration on a timestamped
    file where classic is NOT undefined: every date in these two files falls
    in 2012-01, so `get_timestamp_class` yields a single class and the
    comparator's timestamp term is never the deciding one.

    `timestamp_bool_dup.csv` is included even though its CTable IS corrupted
    in classic, because the NaN swallows the difference and the two agree
    anyway -- which is the point: the one byte under test is the sign.
    """
    out = _run(
        capsys,
        csv,
        "y",
        ["--time-dispersion-pressure=0.5", "--time-bscore-granularity=month"],
        timed=True,
    )
    assert out == "nan true \n"


@pytest.mark.parametrize("csv", ["timestamp_bool_clean.csv", "timestamp_bool_months.csv"])
def test_dispersion_pressure_at_day_granularity_runs_where_classic_segfaults(capsys, csv):
    """NOT an oracle -- classic has no output here (rc 139, core dumped, at
    every one of -m100/-m300/-m1000 x -r1/-r7/-r42 on both files). See this
    module's own docstring for the cause and `do_score`'s DIVERGENCE marker
    for the decision.

    What this pins is that the port produces a complete, well-formed
    metapopulation listing -- the deliberate divergence, not an accident.
    """
    out = _run(capsys, csv, "y", ["--time-dispersion-pressure=0.5"], timed=True)
    lines = out.splitlines()
    assert len(lines) == 10
    for line in lines:
        score, tree = line.split(" ", 1)
        float(score)  # must parse
        assert tree.strip()


def test_dispersion_pressure_changes_the_ports_own_ranking(capsys):
    """A guard the classic comparison cannot give on this corpus: the
    penalty must actually reach the composite score. If
    `bs.push_back(dispersion_penalty)` were dropped, this run would
    silently equal the no-pressure run.
    """
    csv = "timestamp_bool_months.csv"
    without = _run(capsys, csv, "y", [], timed=True)
    with_pressure = _run(capsys, csv, "y", ["--time-dispersion-pressure=0.5"], timed=True)
    assert without != with_pressure


def test_dispersion_exponent_changes_the_ports_own_ranking(capsys):
    """Likewise for `--time-dispersion-exponent`, which is inert unless the
    pressure is positive and would otherwise be untested on a table where
    the penalty is finite.
    """
    csv = "timestamp_bool_months.csv"
    e1 = _run(capsys, csv, "y", ["--time-dispersion-pressure=0.5"], timed=True)
    e2 = _run(
        capsys,
        csv,
        "y",
        ["--time-dispersion-pressure=0.5", "--time-dispersion-exponent=2.0"],
        timed=True,
    )
    assert e1 != e2
