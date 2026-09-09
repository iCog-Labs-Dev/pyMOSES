"""CLI oracle tests at arity 4, 5 and 6.

`docs/REMAINING-WORK.md` recorded a standing corpus gap: "no argument above
`$3`, no arity >= 4". Every existing oracle in this suite runs on a 2- or
3-input table, so nothing exercised `$4`/`$5`/`$6` at all -- and this
project's most-repeated defect is precisely a corpus that only holds values
on which the faithful and the broken version agree.

These ten cases close that gap. All were captured from classic's own binary:

    ( cd <scratch> && ulimit -v 2097152; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses \
        -i<table>.csv -u y -m<M> -W1 -r<R> )

The tables are committed alongside, under `tests/fixtures/table_data/`, and
are generated deterministically -- full truth tables for the boolean ones,
a fixed arithmetic progression for the contin ones -- so there is no RNG in
the DATA, only in the search.

Boolean tables use targets that genuinely depend on the high-numbered
arguments (5-input majority, and a 6-input mix reading `$1`, `$2`, `$4` and
`$6`), so a port that mishandled arguments above `$3` could not match by
accident.
"""

# ruff: noqa: E501 -- the expected strings below are classic's stdout,
# captured verbatim. Wrapping one to satisfy the line limit would mean
# editing an oracle, which is the one thing these literals must not be.

from pathlib import Path

import pytest

from pymetta.moses.main.moses_exec import moses_exec

_TABLE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"

_CASES = [
    (
        "bool_a4_par",
        1500,
        1,
        "-6 or(and(or(!$x1 $x4) $x3) and(!$x1 $x4) $x2) \n"
        "-6 or(and(or($x2 $x3) $x4) and($x2 $x3) !$x1) \n"
        "-8 true \n"
        "-8 false \n"
        "-6 or(and(or(!$x1 $x2 $x4) $x3) and(!$x1 $x4) $x2) \n"
        "-6 or(and(or(!$x1 $x4) $x3) and(!$x1 !$x3 $x4) $x2) \n"
        "-6 or(and(or(!$x1 $x2) $x4) and(or(!$x1 $x4) $x3) $x2) \n"
        "-6 or(and(or(!$x1 $x3) $x4) and(or(!$x1 $x4) $x3) $x2) \n"
        "-6 or(and(or(!$x1 $x4) or($x2 $x3)) and(!$x1 $x4) $x2) \n"
        "-6 or(and(or(!$x1 $x4) $x3) and(or($x2 $x4) !$x1) $x2) \n",
    ),
    (
        "bool_a4_par",
        1500,
        4,
        "-4 or(and(or(and(!$x1 $x3) $x2) !$x4) and(or(and(!$x2 $x4) !$x3) $x1) and(!$x1 $x2 $x3)) \n"
        "-5 or(and(or(and(!$x1 $x3) $x2) !$x4) and(or(!$x3 $x4) $x1) and($x2 $x3)) \n"
        "-6 or(and(!$x1 $x3) and($x1 !$x3) $x2 $x4) \n"
        "-6 or(and(or($x1 $x2) !$x4) and($x1 $x2) $x3) \n"
        "-6 or(and(or($x2 $x3) !$x4) and($x2 $x3) $x1) \n"
        "-5 or(and(or(and(!$x1 $x3) $x2) !$x4) and(or($x2 !$x3 $x4) $x1) and($x2 $x3)) \n"
        "-5 or(and(or(and(!$x1 $x3) $x2) !$x4) and(or(!$x3 $x4) $x1) and(!$x1 $x2 $x3)) \n"
        "-5 or(and(or(and(!$x1 $x3) $x2) !$x4) and(or(!$x3 $x4) $x1) and($x2 $x3 $x4)) \n"
        "-5 or(and(or(and(or(!$x1 $x2) $x3) $x2) !$x4) and(or(!$x3 $x4) $x1) and($x2 $x3)) \n"
        "-5 or(and(or(and(or(!$x1 $x4) $x3) $x2) !$x4) and(or(!$x3 $x4) $x1) and($x2 $x3)) \n",
    ),
    (
        "bool_a5_maj",
        1500,
        1,
        "-2 or(and(or($x1 $x4) or($x2 $x3) $x5) and($x1 $x4) and($x2 $x3)) \n"
        "-2 or(and(or($x1 $x4) or($x2 $x3) $x5) and(or($x1 $x5) $x2 $x3) and($x1 $x4)) \n"
        "-2 or(and(or(and(or(!$x1 $x2) $x4) $x1) or($x2 $x3) $x5) and($x1 $x4) and($x2 $x3)) \n"
        "-2 or(and(or(and(or($x2 $x5) $x1) $x4) or($x2 $x3) $x5) and($x1 $x4) and($x2 $x3)) \n"
        "-2 or(and(or(and(or(!$x4 $x5) $x1) $x4) or($x2 $x3) $x5) and($x1 $x4) and($x2 $x3)) \n"
        "-2 or(and(or(and($x1 $x5) $x2) $x3) and(or($x1 $x4) or($x2 $x3) $x5) and($x1 $x4)) \n"
        "-2 or(and(or(and($x2 $x3) $x4) $x1) and(or($x1 $x4) or($x2 $x3) $x5) and($x2 $x3)) \n"
        "-2 or(and(or(and($x3 $x5) $x1) $x4) and(or($x1 $x4) or($x2 $x3) $x5) and($x2 $x3)) \n"
        "-2 or(and(or(and($x3 $x5) $x4) $x1) and(or($x1 $x4) or($x2 $x3) $x5) and($x2 $x3)) \n"
        "-2 or(and(or(and($x4 $x5) $x2) $x3) and(or($x1 $x4) or($x2 $x3) $x5) and($x1 $x4)) \n",
    ),
    (
        "bool_a5_maj",
        1500,
        4,
        "-2 and(or(and($x2 $x3) and($x4 $x5) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or($x1 $x2) $x3) and($x4 $x5) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or($x1 $x3) $x2) and($x4 $x5) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or($x1 $x4) $x5) and($x2 $x3) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or($x1 $x5) $x4) and($x2 $x3) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or(!$x1 $x2) $x4 $x5) and($x2 $x3) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or(and($x1 $x2) $x5) $x4) and($x2 $x3) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or(and($x1 $x5) $x3) $x2) and($x4 $x5) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or(and($x2 $x3) $x4) $x5) and($x2 $x3) $x1) or($x2 $x3) or($x4 $x5)) \n"
        "-2 and(or(and(or(and($x4 $x5) $x3) $x2) and($x4 $x5) $x1) or($x2 $x3) or($x4 $x5)) \n",
    ),
    (
        "bool_a6_mix",
        1500,
        1,
        "0 or(and($x1 $x4) and($x2 !$x6)) \n"
        "-12 and($x1 $x4) \n"
        "-12 and(or($x1 $x2) $x4) \n"
        "-12 and(or($x1 !$x6) $x4) \n"
        "-12 and(or($x2 $x4) $x1) \n"
        "-12 or(and($x1 $x4) $x2) \n"
        "-12 or(and($x1 $x4) !$x6) \n"
        "-12 or(and(or($x1 $x2) $x4) $x2) \n"
        "-12 or(and($x1 $x4) and($x3 !$x6)) \n"
        "-12 and(or(and(or($x2 $x4) $x1) $x2) $x4) \n",
    ),
    ("bool_a6_mix", 1500, 4, "0 or(and($x1 $x4) and($x2 !$x6)) \n-36 true \n"),
    (
        "contin_a4",
        400,
        1,
        "-2.51063036918640137 +(1 /($x3 +(2 *(-1 $x4)))) \n"
        "-3.99672222137451172 +(1 *(0.66666666666666663 $x3)) \n"
        "-4.30800008773803711 +(1 *(0.5 $x3)) \n"
        "-3.21285200119018555 +(1 /($x3 +(1 exp(*(-1 $x4))))) \n"
        "-3.4894413948059082 +(1 /($x3 +(2 *(-1 $x4) $x1))) \n"
        "-4.02824831008911133 +(1 /($x3 +(2 *(-1 $x2)))) \n"
        "-5.23449993133544922 +(0.5 $x3) \n"
        "-4.39080810546875 +(1 /($x3 +(1 $x1))) \n"
        "-3.83425569534301758 +(1 /($x3 +(1 exp(*(-1 $x2))))) \n"
        "-3.83755779266357422 +(1 /($x3 +(1 exp(*(-1 $x3))))) \n",
    ),
    (
        "contin_a4",
        400,
        4,
        "-2.51063036918640137 +(1 /($x3 +(2 *(-1 $x4)))) \n"
        "-3.99672222137451172 +(1 *(0.66666666666666663 $x3)) \n"
        "-4.30800008773803711 +(1 *(0.5 $x3)) \n"
        "-3.21285200119018555 +(1 /($x3 +(1 exp(*(-1 $x4))))) \n"
        "-3.4894413948059082 +(1 /($x3 +(2 *(-1 $x4) $x1))) \n"
        "-4.02824831008911133 +(1 /($x3 +(2 *(-1 $x2)))) \n"
        "-5.23449993133544922 +(0.5 $x3) \n"
        "-4.39080810546875 +(1 /($x3 +(1 $x1))) \n"
        "-3.83425569534301758 +(1 /($x3 +(1 exp(*(-1 $x2))))) \n"
        "-3.83755779266357422 +(1 /($x3 +(1 exp(*(-1 $x3))))) \n",
    ),
    (
        "contin_a5",
        400,
        1,
        "-3.09599995613098145 +(1 $x4) \n-4.61600017547607422 +(1 $x5) \n-99.9820022583007812 0 \n",
    ),
    (
        "contin_a5",
        400,
        4,
        "-3.09599995613098145 +(1 $x4) \n-4.61600017547607422 +(1 $x5) \n-99.9820022583007812 0 \n",
    ),
]


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("table", "max_evals", "seed", "expected"), _CASES, ids=lambda v: str(v)[:24]
)
def test_higher_arity_matches_classic(table, max_evals, seed, expected, capsys):
    rc = moses_exec(
        [
            "moses",
            f"-i{_TABLE_DIR / (table + '.csv')}",
            "-u",
            "y",
            f"-m{max_evals}",
            "-W1",
            f"-r{seed}",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out == expected
