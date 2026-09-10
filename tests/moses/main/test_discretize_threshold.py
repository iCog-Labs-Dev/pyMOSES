"""CLI oracle tests for `-R/--discretize-threshold`.

`discretize_contin_bscore` (`bscores.cc:196`) scores a contin-output tree
as a CLASSIFIER: thresholds cut the contin range into classes, and a row
costs `weights[c]` when the tree's class differs from the target's. It was
the last unported scorer reachable from a real command line.

All expected strings are classic's own stdout, captured with

    ( cd <scratch> && ulimit -v 2097152; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses \
        -i<contin_regression_sample.csv> -u y -m300 -W1 <flags> -r<seed> )

**Both flags were checked to discriminate before these were trusted**, which
matters because this project's most-repeated defect is a corpus where the
faithful and broken versions agree:

* adding `-R` changes classic's output versus the same run without it, so
  the scorer really is swapped rather than the contin scorer running anyway;
* adding `--weighted-accuracy=1` changes classic's output again at both
  seeds, so the weighted branch -- the float division and the per-class
  counts -- is genuinely exercised, not just constructed.
"""

# ruff: noqa: E501 -- the expected strings are classic's stdout, captured
# verbatim. Wrapping one to satisfy the line limit would mean editing an
# oracle.

from pathlib import Path

import pytest

from pymetta.moses.main.moses_exec import moses_exec

_CONTIN_CSV = (
    Path(__file__).resolve().parents[2] / "fixtures" / "table_data" / "contin_regression_sample.csv"
)

_CASES = [
    (["-R0.01"], 1, "10 0 \n"),
    (["-R0.01"], 3, "10 0 \n"),
    (["-R0.005", "-R0.02"], 1, "11 0 \n"),
    (["-R0.005", "-R0.02"], 3, "11 0 \n"),
    (["-R0.01", "--weighted-accuracy=1"], 1, "10 0 \n"),
    (["-R0.01", "--weighted-accuracy=1"], 3, "10 0 \n"),
    (["-R0.005", "-R0.02", "-R0.05", "--weighted-accuracy=1"], 1, "15 0 \n"),
    (["-R0.005", "-R0.02", "-R0.05", "--weighted-accuracy=1"], 3, "15 0 \n"),
]


@pytest.mark.oracle
@pytest.mark.parametrize(("flags", "seed", "expected"), _CASES, ids=lambda v: str(v)[:30])
def test_discretize_threshold_matches_classic(flags, seed, expected, capsys):
    rc = moses_exec(["moses", f"-i{_CONTIN_CSV}", "-u", "y", "-m300", "-W1", *flags, f"-r{seed}"])
    assert rc == 0
    assert capsys.readouterr().out == expected
