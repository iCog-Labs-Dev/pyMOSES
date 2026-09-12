"""`--balance` at the CLI: classic accepts it, exits 0, and produces
stdout byte-identical to the same run without it -- because
`CTable::balance()` (table.cc:806) writes its rescaling into temporaries
and drops it (see `CTable.balance`'s own docstring for the direct
measurement).

Before this, the port raised `NotImplementedError` on `--balance`, so a
flag classic accepts was refused. These are the CLI-level oracles for
lifting that gate.

The expected strings are classic's own stdout, captured verbatim:

    ( ulimit -v 2097152; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses \
        -i tests/fixtures/table_data/imbalanced_bool.csv -u y -m1000 -W1 \
        -r1 --balance=<0|1> )

`imbalanced_bool.csv` is a full 3-input truth table with a deliberately
lopsided target (7 `true` rows to 1 `false`), so a working balance would
rescale the two classes by 4/7 and 4/1 respectively -- see
`tests/comboreduct/table/test_ctable_balance.py::
test_balance_multipliers_are_not_all_one`, which proves this corpus is not
one where balanced and unbalanced happen to agree.
"""

from pathlib import Path

import pytest

from pymoses.moses.main.moses_exec import moses_exec

_TABLE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"

# Classic's stdout for BOTH --balance=0 and --balance=1 (they are the same
# bytes; that identity is itself the finding).
_EXPECTED_IMBALANCED = (
    "0 or(!$i1 !$i2 !$i3) \n-1 true \n-1 or(!$i1 !$i2) \n-1 or(!$i1 !$i3) \n-1 or(!$i2 !$i3) \n"
)


@pytest.mark.oracle
@pytest.mark.parametrize("balance", ["0", "1"])
def test_balance_flag_matches_classic(balance, capsys):
    rc = moses_exec(
        [
            "moses",
            f"-i{_TABLE_DIR / 'imbalanced_bool.csv'}",
            "-u",
            "y",
            "-m1000",
            "-W1",
            "-r1",
            f"--balance={balance}",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out, "empty stdout would make the comparison below vacuous"
    assert out == _EXPECTED_IMBALANCED


def test_balance_does_not_change_the_ports_own_output(capsys):
    """The port's `--balance=1` output equals its own `--balance=0` output.
    Asserted separately from the classic comparison above so that a future
    change which broke BOTH in the same way still fails one of the two.
    """
    argv = [
        "moses",
        f"-i{_TABLE_DIR / 'imbalanced_bool.csv'}",
        "-u",
        "y",
        "-m1000",
        "-W1",
        "-r1",
    ]
    assert moses_exec([*argv, "--balance=0"]) == 0
    without = capsys.readouterr().out
    assert moses_exec([*argv, "--balance=1"]) == 0
    with_balance = capsys.readouterr().out
    assert without, "empty stdout would make the comparison vacuous"
    assert without == with_balance
