r"""The `ss-*` subsample family at the CLI, against classic's own binary.

Every one of these command lines made this port raise
`NotImplementedError` from `problem_params.py` until the `ss-*` subsample
task; classic exits 0 on all of them. They are 28 of the value-aware
sweep's own rows plus the configurations that make the otherwise-inert
options actually change classic's answer.

Every expected string is classic's own stdout, captured verbatim from

    ( ulimit -v 2097152; ulimit -c 0; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses \
        -i tests/fixtures/table_data/<file> -u <target> \
        <-m> <-r> -W1 <flags> )

run through a `bash -c '...' _ "$@"` wrapper, never an inline `cmd $args`
expansion (the session shell is zsh, which does not word-split, and that
has produced false agreement in this project three times).

## Which options are OBSERVABLE, and which are not

An `IDENTICAL` verdict on an option that never changes classic's own
output is agreement about nothing. Measured per option on these tables:

* LIVE, and covered below: `--ss-n-subsample-demes`,
  `--ss-n-top-candidates`, `--ss-std-dev-threshold`, `--ss-n-tuples`,
  `--ss-n-subsample-fitnesses`, `--ss-low-dev-pressure`, `--ss-by-time`,
  `--ss-contiguous-time`.
* `--ss-n-tuples` is inert at the default `--ss-n-top-candidates=1`, and
  that is arithmetic, not luck: `ss_score_dev_filter` computes
  `n_possible_tuples = n_top_candidates ** n_ss_demes` and samples
  `min(n_possible_tuples, n_tuples)` of them (subsample.cc:180-184), so
  one possible tuple means one sample whatever `n_tuples` says. It is
  also inert unless `--ss-std-dev-threshold` is set, since nothing calls
  that filter otherwise (subsample.cc:139-141). The `ntup*` cases below
  are inside BOTH of those conditions, and the three `n_tuples` values
  give three different answers.
* NOT observable against this oracle build, and not covered:
  `--ss-tanimoto-mean-threshold`, `--ss-tanimoto-geometric-mean-threshold`,
  `--ss-tanimoto-max-threshold`, `--ss-n-best-bfdemes`,
  `--ss-tanimoto-mean-weight`, `--ss-tanimoto-geometric-mean-weight`,
  `--ss-tanimoto-max-weight`. Every value of the first four that is not
  the default reaches `opencog::tanimoto_distance` (numeric.h:361), and
  `HAVE_BOOST` is defined NOWHERE in the classic checkout, so that
  function's `#else` arm throws `RuntimeException("Compiled without boost
  support")` and classic SIGABRTs (rc 134) with no output. The three
  weights only feed the `n_best_bfdemes > 0` arm, which aborts for the
  same reason at any value. This predates the `ss-*` task and is not
  specific to it: `--diversity-dst=tanimoto` aborts classic identically.
  There is no classic output to compare against, so those seven options
  are reported as unverifiable rather than asserted.

## What classic itself cannot do

`--ss-n-subsample-demes>1` TOGETHER with `--ss-n-subsample-fitnesses>1`
aborts classic: the ss-deme loop calls `ignore_rows` on the cscorer,
which is now the `ss_bscore`, whose own `ignore_rows` is
`OC_ASSERT(false, "Not implemented yet")` (ss_bscore.cc:145). This port
raises `NotImplementedError` with that same message at the same point --
see the last test.
"""

from pathlib import Path

import pytest

from pymetta.moses.main.moses_exec import moses_exec

_TABLE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"


# (id, csv, target, argv tail, classic's own stdout as a tuple of LINES --
# split only to keep this file inside the line-length limit; the assertion
# below rejoins them and compares the whole byte string.)
_CASES = (
    (
        "ssd0",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=0",
        ),
        (
            "-8 true ",
            "-8 false ",
            "-8 !$x1 ",
            "-8 $x1 ",
            "-8 !$x2 ",
            "-8 $x2 ",
            "-8 !$x3 ",
            "-8 $x3 ",
            "-8 !$x4 ",
            "-8 $x4 ",
        ),
    ),
    (
        "ssd1",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=1",
        ),
        (
            "-8 true ",
            "-8 false ",
            "-8 !$x1 ",
            "-8 $x1 ",
            "-8 !$x2 ",
            "-8 $x2 ",
            "-8 !$x3 ",
            "-8 $x3 ",
            "-8 !$x4 ",
            "-8 $x4 ",
        ),
    ),
    (
        "ssd2",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=2",
        ),
        (
            "-8 true ",
            "-8 and(!$x1 !$x3) ",
            "-8 or($x1 $x3) ",
        ),
    ),
    (
        "ssd3",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=3",
        ),
        (
            "-8 true ",
            "-8 false ",
            "-8 and($x1 !$x4) ",
            "-8 or($x1 $x3) ",
        ),
    ),
    (
        "ssd2b",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r3",
            "--ss-n-subsample-demes=2",
        ),
        (
            "-6 or(and(or($x4 $x5) $x1) and($x3 $x4)) ",
            "-8 and(or($x2 $x4) $x5) ",
            "-8 or(and($x1 $x3) $x4) ",
            "-8 or(and($x1 $x4) $x2) ",
            "-16 true ",
        ),
    ),
    (
        "ssd4",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r3",
            "--ss-n-subsample-demes=4",
        ),
        (
            "-6 or(and($x1 $x4) and($x2 $x4) and($x2 $x5)) ",
            "-8 or(and(or($x3 $x4) $x1) and(!$x1 $x3)) ",
            "-10 $x4 ",
            "-10 or($x1 $x4) ",
            "-16 true ",
        ),
    ),
    (
        "ntop1",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=2",
            "--ss-n-top-candidates=1",
        ),
        (
            "-8 true ",
            "-8 and(!$x1 !$x3) ",
            "-8 or($x1 $x3) ",
        ),
    ),
    (
        "ntop3",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=2",
            "--ss-n-top-candidates=3",
        ),
        (
            "-8 true ",
            "-7 or(and(!$x2 $x4) $x1 $x3) ",
            "-7 or(and($x2 !$x4) $x1 $x3) ",
            "-8 !$x3 ",
            "-8 and(!$x1 !$x3) ",
            "-8 or($x1 $x3) ",
            "-8 and(or($x2 !$x3) !$x1) ",
        ),
    ),
    (
        "sdev0",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=2",
            "--ss-std-dev-threshold=0",
        ),
        ("-8 true ",),
    ),
    (
        "sdev1",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-demes=2",
            "--ss-std-dev-threshold=1",
        ),
        (
            "-8 true ",
            "-8 and(!$x1 !$x3) ",
            "-8 or($x1 $x3) ",
        ),
    ),
    (
        "sdev5",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r3",
            "--ss-n-subsample-demes=3",
            "--ss-std-dev-threshold=0.5",
        ),
        ("-16 true ",),
    ),
    (
        "ntupA",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r1",
            "--ss-n-subsample-demes=3",
            "--ss-n-top-candidates=3",
            "--ss-std-dev-threshold=2.3",
            "--ss-n-tuples=1",
        ),
        (
            "-5 or(and(or($x1 $x4) $x2) and($x3 $x5)) ",
            "-5 or(and(or($x2 $x4) $x1) and($x3 $x5)) ",
            "-5 or(and(or($x3 $x4) $x5) and($x1 $x2)) ",
            "-5 or(and(or($x4 $x5) $x3) and($x1 $x2)) ",
            "-5 or(and($x1 $x2) and($x1 $x4) and($x3 $x5)) ",
            "-5 or(and(or(and($x2 $x4) $x5) $x3) and($x1 $x2)) ",
            "-6 or(and($x1 $x2) and($x3 $x5)) ",
            "-6 or(and(or($x1 $x3) $x5) and($x1 $x2)) ",
            "-6 or(and(or($x2 $x5) $x1) and($x3 $x5)) ",
            "-7 or(and($x1 $x2) and($x3 $x4 $x5)) ",
        ),
    ),
    (
        "ntupB",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r1",
            "--ss-n-subsample-demes=3",
            "--ss-n-top-candidates=3",
            "--ss-std-dev-threshold=2.3",
            "--ss-n-tuples=2",
        ),
        (
            "-5 or(and(or($x2 $x4) $x1) and($x3 $x5)) ",
            "-5 or(and(or($x3 $x4) $x5) and($x1 $x2)) ",
            "-5 or(and(or($x4 $x5) $x3) and($x1 $x2)) ",
            "-6 or(and($x1 $x2) and($x3 $x5)) ",
            "-6 or(and(or($x1 $x3) $x5) and($x1 $x2)) ",
            "-6 or(and(or($x1 $x5) $x2) and($x3 $x5)) ",
            "-6 or(and(or($x2 $x3) $x5) and($x1 $x2)) ",
            "-6 or(and(or($x2 $x5) $x1) and($x3 $x5)) ",
            "-7 or(and($x1 $x2) and($x3 $x4 $x5)) ",
            "-7 or(and($x1 $x2) and($x3 $x5) $x4) ",
        ),
    ),
    (
        "ntupC",
        "bool_a5_maj.csv",
        "y",
        (
            "-m1000",
            "-r1",
            "--ss-n-subsample-demes=3",
            "--ss-n-top-candidates=3",
            "--ss-std-dev-threshold=2.3",
            "--ss-n-tuples=27",
        ),
        (
            "-5 or(and(or($x1 $x4) $x2) and($x3 $x5)) ",
            "-5 or(and(or($x2 $x4) $x1) and($x3 $x5)) ",
            "-5 or(and(or($x4 $x5) $x3) and($x1 $x2)) ",
            "-5 or(and($x1 $x2) and($x2 $x4) and($x3 $x5)) ",
            "-6 or(and($x1 $x2) and($x3 $x5)) ",
            "-6 or(and(or($x2 $x5) $x3) and($x1 $x2)) ",
            "-7 or(and($x1 $x2) and($x3 $x4 $x5)) ",
            "-7 or(and($x1 $x2) and($x3 $x5) $x4) ",
            "-7 or(and($x1 $x2 $x4) and($x3 $x5)) ",
            "-8 or(and($x1 $x3) $x5) ",
        ),
    ),
    (
        "ssf2",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-fitnesses=2",
        ),
        (
            "-7 and(or(!$x1 $x2) or($x1 !$x2) $x3 !$x4) ",
            "-7.5 and(or($x1 !$x2) $x3 !$x4) ",
            "-7.5 and(or(and(or(!$x1 $x4) !$x2) $x1) $x3 !$x4) ",
            "-7.5 and(or(and(or($x2 $x3) $x1) !$x2) $x3 !$x4) ",
            "-8 true ",
            "-8 !$x1 ",
            "-8 $x1 ",
            "-8 !$x2 ",
            "-8 $x2 ",
            "-8 !$x3 ",
        ),
    ),
    (
        "ssf3",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-fitnesses=3",
        ),
        (
            "-6.81649684906005859 and(or(and($x1 $x4) $x2 !$x3) or($x1 $x4)) ",
            "-7 and(or($x1 $x4) or($x2 !$x3)) ",
            "-7.4714045524597168 or(and($x1 !$x3 !$x4) $x2) ",
            "-7.4714045524597168 or(and($x1 $x4) $x2 !$x3) ",
            "-7.4714045524597168 or(and(or($x1 $x4) $x2) !$x3) ",
            "-7.81649684906005859 or(!$x1 $x2 !$x3 $x4) ",
            "-7.81649684906005859 or(and($x1 $x2 !$x4) !$x3) ",
            "-7.94280910491943359 and(or(and(!$x3 !$x4) $x2) or($x1 $x4)) ",
            "-8.47140407562255859 true ",
            "-8.47140407562255859 !$x1 ",
        ),
    ),
    (
        "ldp0",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-fitnesses=2",
            "--ss-low-dev-pressure=0",
        ),
        (
            "-8 true ",
            "-8 false ",
            "-8 !$x1 ",
            "-8 $x1 ",
            "-8 !$x2 ",
            "-8 $x2 ",
            "-8 !$x3 ",
            "-8 $x3 ",
            "-8 !$x4 ",
            "-8 $x4 ",
        ),
    ),
    (
        "ldp2",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-fitnesses=2",
            "--ss-low-dev-pressure=2",
        ),
        (
            "-8 true ",
            "-8 !$x1 ",
            "-8 $x1 ",
            "-8 !$x2 ",
            "-8 $x2 ",
            "-8 !$x3 ",
            "-8 $x3 ",
            "-8 !$x4 ",
            "-8 $x4 ",
            "-8 and(!$x1 !$x2) ",
        ),
    ),
    (
        "ssfT",
        "bool_a4_par.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--ss-n-subsample-fitnesses=2",
            "--ss-by-time=1",
        ),
        (
            "-6 or(and(!$x1 !$x2) and($x1 $x2) and(!$x3 !$x4)) ",
            "-6 or(and(!$x1 !$x2) and(!$x1 $x3) and($x1 $x2) and(!$x3 !$x4)) ",
            "-6 or(and(!$x1 !$x2) and(!$x1 $x4) and($x1 $x2) and(!$x3 !$x4)) ",
            "-6 or(and(!$x1 !$x2) and($x1 $x2) and(!$x2 $x3) and(!$x3 !$x4)) ",
            "-7 and(!$x1 $x2 !$x3 !$x4) ",
            "-7 and($x1 !$x2 !$x3 !$x4) ",
            "-7 or(and(!$x3 !$x4) !$x1 $x2) ",
            "-7 or(and(!$x3 !$x4) $x1 !$x2) ",
            "-7 and(or(and(!$x3 !$x4) $x1) $x2) ",
            "-7 and(or(and(!$x3 !$x4) $x2) $x1) ",
        ),
    ),
    (
        "btC0",
        "timestamp_bool_clean.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=0",
        ),
        (
            "-2 !$i1 ",
            "-2 or(!$i1 $i2) ",
            "-4 true ",
        ),
    ),
    (
        "btC1",
        "timestamp_bool_clean.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
        ),
        (
            "-2 !$i1 ",
            "-4 true ",
        ),
    ),
    (
        "ctC0",
        "timestamp_bool_clean.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
            "--ss-contiguous-time=0",
        ),
        (
            "-2 !$i1 ",
            "-2 or(!$i1 $i2) ",
            "-4 true ",
        ),
    ),
    (
        "ctC1",
        "timestamp_bool_clean.csv",
        "y",
        (
            "-m300",
            "-r1",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
            "--ss-contiguous-time=1",
        ),
        (
            "-2 !$i1 ",
            "-4 true ",
        ),
    ),
    (
        "btM0",
        "timestamp_bool_months.csv",
        "y",
        (
            "-m1000",
            "-r5",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=0",
        ),
        (
            "-2 !$i1 ",
            "-3 $i2 ",
            "-4 true ",
        ),
    ),
    (
        "btM1",
        "timestamp_bool_months.csv",
        "y",
        (
            "-m1000",
            "-r5",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
        ),
        (
            "-3 and(!$i1 $i2) ",
            "-4 true ",
        ),
    ),
    (
        "ctM0",
        "timestamp_bool_months.csv",
        "y",
        (
            "-m1000",
            "-r5",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
            "--ss-contiguous-time=0",
        ),
        (
            "-2 !$i1 ",
            "-2 or(!$i1 $i2) ",
            "-4 true ",
        ),
    ),
    (
        "ctM1",
        "timestamp_bool_months.csv",
        "y",
        (
            "-m1000",
            "-r5",
            "--timestamp-feature=when",
            "--ss-n-subsample-demes=2",
            "--ss-by-time=1",
            "--ss-contiguous-time=1",
        ),
        (
            "-3 and(!$i1 $i2) ",
            "-4 true ",
        ),
    ),
    (
        "ssfTs",
        "timestamp_bool_months.csv",
        "y",
        (
            "-m1000",
            "-r5",
            "--timestamp-feature=when",
            "--ss-n-subsample-fitnesses=2",
            "--ss-by-time=1",
        ),
        (
            "-2 !$i1 ",
            "-3 or(!$i1 $i2) ",
            "-3.5 $i2 ",
            "-3.5 and(!$i1 $i2) ",
            "-5 true ",
            "-5 and(!$i1 !$i2) ",
            "-5 or(!$i1 !$i2) ",
            "-5.5 and($i1 $i2) ",
            "-5.5 or($i1 $i2) ",
            "-6 !$i2 ",
        ),
    ),
)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("case_id", "csv", "target", "tail", "expected_lines"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_ss_cli_matches_classic_byte_for_byte(capsys, case_id, csv, target, tail, expected_lines):
    argv = ["moses", f"-i{_TABLE_DIR / csv}", "-u", target, "-W1", *tail]
    assert moses_exec(argv) == 0
    out = capsys.readouterr().out
    assert out, "empty stdout would make the comparison below vacuous"
    assert out == "".join(line + "\n" for line in expected_lines)


def test_the_corpus_is_not_all_the_same_answer():
    """Guard against the failure mode this project keeps hitting: a corpus
    where faithful and broken agree. If every case printed the same thing,
    every assertion above would pass for a port that ignored `ss-*`
    entirely.
    """
    answers = {c[4] for c in _CASES}
    assert len(answers) >= 12, sorted(len(a) for a in answers)

    # And the specific pairs whose whole point is that ONE option differs.
    by_id = {c[0]: c[4] for c in _CASES}
    assert by_id["ssd1"] != by_id["ssd2"], "--ss-n-subsample-demes is inert here"
    assert by_id["ntop1"] != by_id["ntop3"], "--ss-n-top-candidates is inert here"
    assert by_id["sdev0"] != by_id["sdev1"], "--ss-std-dev-threshold is inert here"
    assert len({by_id["ntupA"], by_id["ntupB"], by_id["ntupC"]}) == 3, "--ss-n-tuples is inert here"
    assert by_id["ssf2"] != by_id["ldp0"], "--ss-low-dev-pressure=0 must disable SS-fitness"
    assert by_id["ssf2"] != by_id["ldp2"], "--ss-low-dev-pressure is inert here"
    assert by_id["btC0"] != by_id["btC1"], "--ss-by-time is inert here"
    assert by_id["ctM0"] != by_id["ctM1"], "--ss-contiguous-time is inert here"


def test_ss_demes_plus_ss_fitness_hits_ss_bscore_ignore_rows_exactly_as_classic_aborts():
    """C++: ss_bscore.cc:145, `OC_ASSERT(false, "Not implemented yet")`.

    Combining the two master switches is broken IN CLASSIC: the ss-deme
    loop (deme_expander.cc:487) calls `ignore_rows` on a cscorer that is
    now wrapping an `ss_bscore`, and that method is an unconditional
    abort. Measured: classic exits 134 with
    `what(): Not implemented yet (.../ss_bscore.cc:146)` and no stdout on

        -i bool_a4_par.csv -u y -m300 -r1 -W1 \
            --ss-n-subsample-demes=2 --ss-n-subsample-fitnesses=2

    This port raises with the same message at the same point -- a raise
    where classic aborts, this port's standing convention for OC_ASSERT.
    """
    argv = [
        "moses",
        f"-i{_TABLE_DIR / 'bool_a4_par.csv'}",
        "-u",
        "y",
        "-m300",
        "-r1",
        "-W1",
        "--ss-n-subsample-demes=2",
        "--ss-n-subsample-fitnesses=2",
    ]
    with pytest.raises(NotImplementedError, match="Not implemented yet"):
        moses_exec(argv)
