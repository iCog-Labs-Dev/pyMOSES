"""Oracle tests for mutual information when there is NOTHING TO DIVIDE BY.

## Oracle: tools/cpp_probe/mi_empty_probe.cc -> tests/fixtures/mi_empty.json

`pymetta.feature_selection.scorers.mutual_info` used to short-circuit all
three of its counter-based functions with `if total == 0.0: return NaN`,
under a comment claiming C++ "divides by `total` unconditionally". It does
not. C++ sizes every probability vector from its counter
(`std::vector<double> ip(ic.size())`, table.h:1233) and `transform`s into
it, so an EMPTY counter performs no division at all and `entropy()` --
`double res = 0;` with an empty loop (numeric.h:229) -- returns exactly
0.0.

That was not academic. `--enable-fs=1 --fs-subsampling-ratio=0` makes
`subsampleCTable` (table.cc:899) remove `(1.0 - 0.0) * uncompressed_size`
rows, i.e. every one, and `simple_selection`'s filter is `threshold <= sc`
(simple.h:121): with sc == 0.0 and the default threshold 0, classic keeps
every feature and selects `--fs-target-size` of them; with NaN, `0 <= NaN`
is False, the port selected NOTHING, built a knob-free representation and
returned the unimproved exemplar. Measured on a 6-input boolean table,
seeds 1-5: classic `-8 or($a1 $a5)` (10 candidates), port `-24 true` (1).

THE CORPUS IS THREE STATES OF THE SAME TABLE, deliberately, because a
fixture of empty tables alone is satisfied by `return 0.0`:

* `baseline` -- untouched. `bool_a6_mix` has a spread of non-zero MI
  (0.105843..., 0.28588..., 0.988699...), so a constant-returning port
  fails here.
* `empty` -- every row removed through `CTable::remove_rows` over the
  whole uncompressed index range, which is what `subsampleCTable(0.0)`
  does minus the RNG. Counters empty, `total == 0.0`.
* `zero_counts` -- rows KEPT, every `count_t` set to 0.0. Counters
  non-empty, `total` still 0.0, so C++'s `div_total` really does compute
  0/0 and `weighted_information`'s `p > PROB_EPSILON` (numeric.h:207)
  swallows the NaN. This is the state a bare `c / total` in Python cannot
  reach at all: it raises ZeroDivisionError.

Every comparison is exact equality on the full double, never
`pytest.approx` -- see test_mutual_info.py's module docstring.
"""

import json
from pathlib import Path

import pytest

from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.feature_selection.ctable_subsample import ctable_remove_rows
from pymetta.feature_selection.scorers.mutual_info import (
    _ot_entropy,
    mutual_information,
    mutual_information_between_sets,
    mutual_information_it_ot,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mi_empty.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

# The probe keys each table by its CSV basename, so the two maps line up.
_CSV_FOR = {name: f"{name}.csv" for name in _GOLDEN}


def _base_table(name: str):
    return loadTable(str(_TABLE_DATA_DIR / _CSV_FOR[name]), "y")


def _emptied(ctable: CTable) -> CTable:
    """The `empty` group's state, built the way classic builds it.

    `ctable_remove_rows` over `range(uncompressed_size)` is exactly what
    `subsampleCTable(0.0, ct)` does once its `nremove ==
    (1.0 - 0.0) * ctable_usize` draws have named every index; going through
    the real function rather than `ctable.clear()` keeps the test honest
    about how the state is reached from the CLI.
    """
    copy = CTable(
        ctable.get_output_label(),
        list(ctable.get_input_labels()),
        ctable.get_signature(),
    )
    copy.update({row: cell.copy() for row, cell in ctable.items()})
    ctable_remove_rows(copy, set(range(int(copy.uncompressed_size()))))
    return copy


def _zeroed(ctable: CTable) -> CTable:
    """The `zero_counts` group's state: rows kept, every count set to 0.0.

    Rebuilt cell by cell because this port's `TimedCounter` refuses raw
    `__setitem__` (it would desync its timed and untimed views) -- the same
    reason `ctable_subsample._replace_cell` exists.
    """
    copy = CTable(
        ctable.get_output_label(),
        list(ctable.get_input_labels()),
        ctable.get_signature(),
    )
    for row, cell in ctable.items():
        replacement = type(cell)()
        for key, _count in cell.timed_items():
            replacement.add(getattr(key, "value", key), getattr(key, "timestamp", None), 0.0)
        copy[row] = replacement
    return copy


_STATE_BUILDERS = {
    "baseline": lambda ct: ct,
    "empty": _emptied,
    "zero_counts": _zeroed,
}


def _cases(state: str) -> list[tuple[str, str, dict]]:
    return [(name, state, case) for name in _GOLDEN for case in _GOLDEN[name][state]["mi"]]


def _btw_cases(state: str) -> list[tuple[str, str, dict]]:
    return [(name, state, case) for name in _GOLDEN for case in _GOLDEN[name][state]["btw_sets"]]


def _ids(cases) -> list[str]:
    return [
        f"{name}-{state}-{case.get('fs', (case.get('fs_l'), case.get('fs_r')))}"
        for name, state, case in cases
    ]


_MI_CASES = [c for state in _STATE_BUILDERS for c in _cases(state)]
_BTW_CASES = [c for state in _STATE_BUILDERS for c in _btw_cases(state)]


@pytest.fixture(scope="module")
def ctables() -> dict[str, dict[str, CTable]]:
    out: dict[str, dict[str, CTable]] = {}
    for name in _GOLDEN:
        base = _base_table(name).compressed()
        out[name] = {state: build(base) for state, build in _STATE_BUILDERS.items()}
    return out


def test_fixture_covers_all_three_states_and_is_not_uniformly_zero():
    """Shape guard: a truncated fixture would reduce every parametrized test
    below to an empty set, and a fixture whose `baseline` is all zeros would
    be satisfied by `return 0.0`. Both are pinned here rather than left to
    whatever the tests happen to consume (CARRY-FORWARD.md Sec 6, "Vacuous
    tests recur")."""
    assert sorted(_GOLDEN) == ["bool_a4_par", "bool_a6_mix"]
    for name, table in _GOLDEN.items():
        assert set(table) >= set(_STATE_BUILDERS), name
        for state in _STATE_BUILDERS:
            assert table[state]["mi"], (name, state)
            assert table[state]["btw_sets"], (name, state)
        assert table["empty"]["size"] == 0
        assert table["zero_counts"]["size"] == table["baseline"]["size"] > 0
        assert table["zero_counts"]["uncompressed_size"] == 0.0
    assert any(case["mi"] != 0.0 for case in _GOLDEN["bool_a6_mix"]["baseline"]["mi"])
    assert len({case["mi"] for case in _GOLDEN["bool_a6_mix"]["baseline"]["mi"]}) >= 3


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "state", "case"), _MI_CASES, ids=_ids(_MI_CASES))
def test_mutual_information_matches_classic(ctables, name, state, case):
    assert mutual_information(ctables[name][state], case["fs"]) == case["mi"]


@pytest.mark.oracle
@pytest.mark.parametrize(("name", "state", "case"), _BTW_CASES, ids=_ids(_BTW_CASES))
def test_mutual_information_between_sets_matches_classic(ctables, name, state, case):
    got = mutual_information_between_sets(ctables[name][state], case["fs_l"], case["fs_r"])
    assert got == case["mi"]


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_ot_entropy_of_an_empty_otable_matches_classic(name):
    table = _base_table(name)
    table.otable.clear()
    assert _ot_entropy(table.otable) == _GOLDEN[name]["ot_entropy_empty"]


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_mutual_information_it_ot_of_an_empty_pair_matches_classic(name):
    table = _base_table(name)
    table.itable.clear()
    table.otable.clear()
    got = [
        mutual_information_it_ot(table.itable, table.otable, case["fs"])
        for case in _GOLDEN[name]["mi_it_ot_empty"]
    ]
    assert got == [case["mi"] for case in _GOLDEN[name]["mi_it_ot_empty"]]
