"""Oracle and guard tests for pymoses.feature_selection.ctable_subsample.

## Oracle: tools/cpp_probe/fs_diversity_probe.cc -> tests/fixtures/fs_diversity.json

The fixture's `"ctable"`, `"remove_rows"`, `"selector"` and `"subsample"`
groups run classic's own `CTable::remove_rows` (table.cc:646),
`lazy_random_selector` (lazy_random_selector.cc) and `subsampleCTable`
(table.cc:899) directly.

`tests/fixtures/table_data/ctable_unsorted.csv` exists FOR THIS TEST and
is built so that the two iteration orders `remove_rows`' uncompressed-row
index arithmetic depends on both differ from the port's own `dict`
insertion order:

* rows -- classic's `std::map<multi_type_seq, TimedCounter>` iterates
  `(1,1,1), (1,0,1), (0,1,0), (0,0,0)` (`id::logical_true` is 1 and
  `id::logical_false` is 2, so a `1` cell sorts BEFORE a `0` cell), while
  the compressed table is BUILT in the order
  `(1,1,1), (0,0,0), (1,0,1), (0,1,0)`;
* values inside a row -- two of the four rows have a `TimedCounter` whose
  ascending-key order (`true` then `false`) is the reverse of the order
  the rows were compressed in (`false` then `true`).

`test_fixture_actually_discriminates_iteration_order` asserts both of
those facts up front, so this file fails loudly rather than silently
losing its coverage if the fixture CSV is ever re-ordered.
"""

import copy
import json
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.table.table import CTable
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.feature_selection.ctable_subsample import (
    _replace_cell,
    ctable_remove_rows,
    subsampleCTable,
)
from pymoses.util.lazy_random_selector import lazy_random_selector
from pymoses.util.rng import randGen

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fs_diversity.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

# The probe prints a CTable's INPUT cells through `multi_type_seq::
# to_strings()` ("1"/"0" for a builtin_seq) and its OUTPUT cells through
# `operator<<(ostream&, const vertex&)` ("true"/"false").
_IN = {"0": builtin.logical_false, "1": builtin.logical_true}
_OUT = {"false": builtin.logical_false, "true": builtin.logical_true}


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


@pytest.fixture()
def unsorted_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "ctable_unsorted.csv"), "y").compressed()


@pytest.fixture()
def big_ctable() -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / "ctable_unsorted_big.csv"), "y").compressed()


def _content(ct: CTable) -> dict:
    """The CTable as an ORDER-INSENSITIVE mapping row -> {value: count}.

    Deliberately order-insensitive: this port's `CTable` is a `dict`
    (insertion-ordered) where classic's is a `std::map` (key-ordered), a
    representational difference that predates this module and is disclosed
    on `CTable` itself. What `remove_rows` must reproduce is WHICH (row,
    value) counts get decremented -- which this mapping captures exactly,
    and which a wrong iteration order changes (see
    `test_insertion_order_iteration_would_be_wrong`).
    """
    return {
        tuple(int(v) for v in row): {int(k): c for k, c in counter.items()}
        for row, counter in ct.items()
    }


def _expected(rows: list) -> dict:
    return {
        tuple(int(_IN[s]) for s in row["inputs"]): {int(_OUT[o]): c for o, c in row["outputs"]}
        for row in rows
    }


# --------------------------------------------------------------- guards


@pytest.mark.oracle
def test_fixture_actually_discriminates_iteration_order(golden, unsorted_ctable):
    """The corpus must be able to detect the bug it exists for.

    Both of the two orders `remove_rows` walks must differ between
    classic's `std::map` (the fixture) and the order the table was BUILT
    in, or every mutation of those orders is a zero-kill mutation for a
    reason that has nothing to do with the code under test.

    The row half of that used to read `for row in unsorted_ctable`, back
    when `CTable` was an insertion-ordered `dict`. `CTable` now iterates in
    `std::map` key order itself (`ctable_row_key` / `CTable._order`,
    pymoses/comboreduct/table/table.py), so the *public* walk agrees with
    the fixture by construction and proves nothing -- the discriminating
    fact is that `Table.compressed` still APPENDS in file order, which the
    unbound `dict.__iter__` still shows. Both claims are asserted below:
    the build order differs from classic's, and the public order matches
    it.
    """
    classic_rows = [tuple(int(_IN[s]) for s in r["inputs"]) for r in golden["ctable"]]
    build_rows = [tuple(int(v) for v in row) for row in dict.__iter__(unsorted_ctable)]
    assert classic_rows != build_rows, "fixture rows are already in map order"
    assert sorted(classic_rows) == sorted(build_rows)
    port_rows = [tuple(int(v) for v in row) for row in unsorted_ctable]
    assert port_rows == classic_rows

    classic_values = [[_OUT[o] for o, _ in r["outputs"]] for r in golden["ctable"]]
    differing = 0
    for classic_row, classic_vals in zip(classic_rows, classic_values, strict=True):
        port_key = next(k for k in unsorted_ctable if tuple(int(v) for v in k) == classic_row)
        port_vals = list(unsorted_ctable[port_key].keys())
        if list(classic_vals) != port_vals:
            differing += 1
    assert differing >= 2, f"only {differing} row(s) have a discriminating value order"


@pytest.mark.oracle
def test_compressed_table_matches_classic(golden, unsorted_ctable):
    assert _content(unsorted_ctable) == _expected(golden["ctable"])
    assert unsorted_ctable.uncompressed_size() == 8.0


# ---------------------------------------------------------- remove_rows


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["remove_rows"])))
def test_remove_rows_matches_classic(golden, unsorted_ctable, case_index):
    case = golden["remove_rows"][case_index]
    ctable_remove_rows(unsorted_ctable, set(case["idxs"]))
    assert _content(unsorted_ctable) == _expected(case["result"])
    assert unsorted_ctable.uncompressed_size() == case["uncompressed_size"]


@pytest.mark.oracle
def test_compressed_big_table_matches_classic(golden, big_ctable):
    assert _content(big_ctable) == _expected(golden["ctable_big"])
    assert big_ctable.uncompressed_size() == 30.0


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["remove_rows_big"])))
def test_remove_rows_big_matches_classic(golden, big_ctable, case_index):
    """The 30-row table. Every index set here contains an element >= 8, so
    a Python `set`'s own iteration order over it is NOT ascending (a
    small-int set's slot order is `value % table_size`) -- which is what
    makes the `sorted(idxs)` in `ctable_remove_rows` testable at all. On
    the 8-row table above every index set is a subset of {0..7} and a
    Python set iterates it ascending by accident.
    """
    case = golden["remove_rows_big"][case_index]
    ctable_remove_rows(big_ctable, set(case["idxs"]))
    assert _content(big_ctable) == _expected(case["result"])
    assert big_ctable.uncompressed_size() == case["uncompressed_size"]


@pytest.mark.oracle
def test_big_index_sets_actually_iterate_out_of_order(golden):
    """Guard for the guard: at least three of the big-table index sets must
    genuinely iterate out of ascending order as a Python `set`, or
    `sorted(idxs)` is untested there too.
    """
    out_of_order = sum(
        1 for c in golden["remove_rows_big"] if list(set(c["idxs"])) != sorted(c["idxs"])
    )
    assert out_of_order >= 3, f"only {out_of_order} index set(s) iterate out of order"


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["subsample_big"])))
def test_subsample_ctable_big_matches_classic(golden, big_ctable, case_index):
    case = golden["subsample_big"][case_index]
    randGen().seed(case["seed"])
    subsampleCTable(case["ratio"], big_ctable)
    assert _content(big_ctable) == _expected(case["result"])
    assert big_ctable.uncompressed_size() == case["uncompressed_size"]


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["subsample_f32"])))
def test_subsample_ctable_float32_ratio_matches_classic(golden, unsorted_ctable, case_index):
    """Ratios where the binary32 narrowing of `subsampleCTable`'s own
    `float ratio` PARAMETER changes `nremove`: at ratio 0.750000001 over
    8 uncompressed rows, `(1.0 - ratio) * 8` truncates to 1 in double and
    to 2 once narrowed (0.75f exactly). Classic's answer is 2.
    """
    case = golden["subsample_f32"][case_index]
    randGen().seed(case["seed"])
    subsampleCTable(case["ratio"], unsorted_ctable)
    assert _content(unsorted_ctable) == _expected(case["result"])
    assert unsorted_ctable.uncompressed_size() == case["uncompressed_size"]


@pytest.mark.oracle
def test_float32_ratio_cases_actually_discriminate(golden):
    """Without a case where the double and the float32 forms disagree, the
    `_to_float32` call is a zero-kill mutation site.
    """
    import struct

    discriminating = 0
    for case in golden["subsample_f32"]:
        usize = case["ctable_usize"]
        as_double = int((1.0 - case["ratio"]) * usize)
        as_f32 = int((1.0 - struct.unpack("f", struct.pack("f", case["ratio"]))[0]) * usize)
        if as_double != as_f32:
            discriminating += 1
            assert case["nremove"] == as_f32
    assert discriminating >= 6, f"only {discriminating} float32-discriminating ratio(s)"


@pytest.mark.oracle
def test_insertion_order_iteration_would_be_wrong(golden, unsorted_ctable):
    """A live guard that the two `sorted()` calls in `ctable_remove_rows`
    are load-bearing, not defensive.

    Re-runs every fixture case against a deliberately broken copy of the
    algorithm that walks the rows and each row's values in `dict`
    insertion order (the "obvious" transliteration), and asserts that at
    least one case then disagrees with classic. Without this, a reviewer
    has to take the ordering claim on faith.
    """
    # Snapshot the BUILD order before anything walks the table publicly:
    # every public accessor -- `items()` included, which `copy.deepcopy`
    # itself goes through -- reorders the storage into `std::map` order.
    build_rows = list(dict.__iter__(unsorted_ctable))
    disagreements = 0
    for case in golden["remove_rows"]:
        ct = copy.deepcopy(unsorted_ctable)
        _remove_rows_insertion_order(ct, set(case["idxs"]), build_rows)
        if _content(ct) != _expected(case["result"]):
            disagreements += 1
    assert disagreements >= 1


def _remove_rows_insertion_order(ctable: CTable, idxs: set[int], build_rows: list) -> None:
    """`ctable_remove_rows` with both `sorted()` iteration orders replaced
    by the order the table was BUILT in. Test-local mutant; see the test
    above.

    `build_rows` is passed in rather than read off `ctable` because
    `CTable` is no longer insertion-ordered: reading it here would hand the
    mutant the correct order and silently defang it.

    Reuses the module's own cell read/write helpers so that ONLY the two
    orders differ -- otherwise the comparison would also be measuring the
    write-back path.
    """
    idx_seq = sorted(idxs)
    idx_it = 0
    i = 0
    row_keys = list(build_rows)
    row_it = 0
    while row_it < len(row_keys) and idx_it < len(idx_seq):
        row = row_keys[row_it]
        cell = ctable[row]
        raw = cell.timed_items() if hasattr(cell, "timed_items") else cell.items()
        observations = list(raw)
        counts = [c for _k, c in observations]
        erased = [False] * len(observations)
        i_end = i + int(sum(counts))
        if i <= idx_seq[idx_it] < i_end:
            v_it = 0
            while v_it < len(observations) and idx_it < len(idx_seq):
                i_v_end = int(i + counts[v_it])
                while idx_it < len(idx_seq) and idx_seq[idx_it] < i_v_end:
                    counts[v_it] -= 1
                    idx_it += 1
                i = i_v_end
                if counts[v_it] == 0:
                    erased[v_it] = True
                v_it += 1
            kept = [
                (k, c)
                for (k, _o), c, gone in zip(observations, counts, erased, strict=True)
                if not gone
            ]
            if not kept:
                del ctable[row]
            else:
                _replace_cell(ctable, row, kept)
        row_it += 1
        i = i_end


@pytest.mark.oracle
def test_remove_rows_raises_on_fractional_row_weights():
    """DO NOT FIX -- classic's own `OC_ASSERT(0.0 == (total_row_weights -
    truncate), "This algo is broken!")` (table.cc:668).

    Measured on the same fixture, same code path: classic ABORTS
    (rc 134, `terminate called after throwing an instance of
    'opencog::AssertionException' -- what(): This algo is broken!
    (.../table.cc:668)`), reached via
    `fs_diversity_probe <...> tests/fixtures/table_data/weighted.csv`.
    """
    weighted = loadTable(str(_TABLE_DATA_DIR / "weighted.csv"), "r").compressed("wgt")
    assert weighted.uncompressed_size() != int(weighted.uncompressed_size())
    with pytest.raises(AssertionError, match="This algo is broken!"):
        ctable_remove_rows(weighted, {0})


@pytest.mark.oracle
def test_remove_rows_empty_index_set_is_a_no_op(unsorted_ctable):
    before = _content(unsorted_ctable)
    ctable_remove_rows(unsorted_ctable, set())
    assert _content(unsorted_ctable) == before


# ------------------------------------------------------------- selector


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["selector"])))
def test_lazy_random_selector_draw_order_matches_classic(golden, case_index):
    """Element for element, not set-wise -- draw ORDER is part of this
    port's bit-identity claim, and a set comparison would pass for any
    permutation.
    """
    case = golden["selector"][case_index]
    randGen().seed(case["seed"])
    selector = lazy_random_selector(case["u"])
    draws = [selector() for _ in range(len(case["draws"]))]
    assert draws == case["draws"]


# -------------------------------------------------------- subsampleCTable


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["subsample"])))
def test_subsample_ctable_matches_classic(golden, unsorted_ctable, case_index):
    case = golden["subsample"][case_index]
    randGen().seed(case["seed"])
    subsampleCTable(case["ratio"], unsorted_ctable)
    assert _content(unsorted_ctable) == _expected(case["result"])
    assert unsorted_ctable.uncompressed_size() == case["uncompressed_size"]


@pytest.mark.oracle
@pytest.mark.parametrize("case_index", range(len(_GOLDEN["subsample"])))
def test_subsample_ctable_draw_sequence_matches_classic(golden, unsorted_ctable, case_index):
    """The derived `ctable_usize`/`nremove` and the exact draw SEQUENCE
    `subsampleCTable` consumes, checked separately from the resulting
    table so a compensating pair of errors cannot hide.
    """
    case = golden["subsample"][case_index]
    usize = int(unsorted_ctable.uncompressed_size())
    assert usize == case["ctable_usize"]

    randGen().seed(case["seed"])
    selector = lazy_random_selector(usize)
    expected_draws = [selector() for _ in range(case["nremove"])]

    randGen().seed(case["seed"])
    subsampleCTable(case["ratio"], unsorted_ctable)

    # Re-derive the same stream and confirm subsampleCTable consumed
    # exactly `nremove` draws in exactly that order: replaying it from the
    # SAME seed after the call must reproduce the same prefix.
    randGen().seed(case["seed"])
    replay = lazy_random_selector(usize)
    assert [replay() for _ in range(case["nremove"])] == expected_draws


@pytest.mark.oracle
def test_subsample_ratio_float32_narrowing_is_observable(golden):
    """`subsampleCTable` takes `float ratio` while every caller's field is
    a `double`, so the value narrows to binary32 at the call. The fixture
    pins `nremove` for ratios whose float32 and double forms differ
    (0.9, 0.7, 0.1); this asserts the port derives the same `nremove` from
    the double it is handed.
    """
    import struct

    checked = 0
    for case in golden["subsample"]:
        ratio = case["ratio"]
        narrowed = struct.unpack("f", struct.pack("f", ratio))[0]
        usize = case["ctable_usize"]
        assert int((1.0 - narrowed) * usize) == case["nremove"]
        if narrowed != ratio:
            checked += 1
    assert checked >= 3, "no ratio in the fixture actually narrows"


def test_subsample_ratio_out_of_range_raises(unsorted_ctable):
    """DIVERGENCE: classic's `OC_ASSERT(0.0 <= ratio and ratio <= 1.0,
    ...)` (table.cc:901) becomes a raise here.
    """
    with pytest.raises(ValueError, match="Ratio must be in"):
        subsampleCTable(1.5, unsorted_ctable)
    with pytest.raises(ValueError, match="Ratio must be in"):
        subsampleCTable(-0.5, unsorted_ctable)


def test_subsample_empty_ctable_raises():
    """`lazy_random_selector(0)` is constructed BEFORE the `dorepeat`, so
    an empty CTable fails regardless of `ratio` -- classic's own
    `OC_ASSERT(u - l > 0)` (lazy_selector.cc:41), ported as a ValueError.
    """
    with pytest.raises(ValueError, match="empty list"):
        subsampleCTable(1.0, CTable("y", ["x1"]))
