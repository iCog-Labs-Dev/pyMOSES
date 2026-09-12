"""`CTable` iterates in `std::map` key order, not file order.

C++'s `CTable` is `class CTable : public std::map<multi_type_seq,
TimedCounter>` (table.h:550) -- an ORDERED map, sorted by
`multi_type_seq::operator<` (table.h:383). This port's is a `dict`, which
was filled in file order, so the two agreed only by luck.

The difference is observable in three places, and only one of them is
cosmetic:

* `--output-bscore` prints one element per compressed row
  (`ctruth_table_bscore::operator()`, bscores.cc:283;
  `select_bscore.cc:166`; `precision_bscore.cc:227`), so the printed
  vector IS the key order;
* `CTable::remove_rows` (table.cc:651) defines an uncompressed-row index
  space by walking the map, and that decides WHICH rows `subsampleCTable`
  (table.cc:899) and `ss_bscore` (ss_bscore.cc:88) delete after their RNG
  draws;
* `select_bscore`'s constructor (select_bscore.cc:98) keeps the FIRST
  weight it sees for a duplicated output value.

Everything else that consumes a bscore -- the composite score's sum,
`remove_dominated`'s element-wise compare, the diversity distances -- is
invariant under a permutation applied to every candidate alike, which is
exactly why this stayed hidden until `-t`.

The oracle for the order itself is classic's own `CTable::remove_rows`
output, captured in `tests/fixtures/fs_diversity.json`'s `"ctable"` group
from `tools/cpp_probe/fs_diversity_probe.cc`.
"""

import json
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.vertex import builtin, contin_t, enum_t
from pymoses.comboreduct.table.table import CTable, TimedCounter, ctable_row_key
from pymoses.comboreduct.table.table_io import loadTable

_TABLE_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"
_FS_DIVERSITY = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "fs_diversity.json").read_text()
)

# The probe prints a CTable's INPUT cells through `multi_type_seq::
# to_strings()`, which renders a `builtin_seq` as "1"/"0".
_IN = {"0": builtin.logical_false, "1": builtin.logical_true}


def _load(name: str) -> CTable:
    return loadTable(str(_TABLE_DATA_DIR / name), "y").compressed()


def _insertion_order(ct: CTable) -> list:
    """The raw `dict` order, i.e. the order `Table.compressed` appended in.

    Read through the unbound `dict.__iter__` so it does NOT trigger
    `CTable._order`; every public accessor does, and would destroy the very
    thing this helper exists to observe. Call it BEFORE anything walks
    `ct` publicly, since `_order` reorders the real storage in place.
    """
    return list(dict.__iter__(ct))


# --------------------------------------------------------------- guards


@pytest.mark.parametrize("name", ["ctable_unsorted.csv", "ctable_unsorted_big.csv"])
def test_corpus_file_order_really_differs_from_map_order(name):
    """Without this the ordering tests below are vacuous.

    A 2-input boolean table has only four distinct rows and can easily land
    in map order by accident; both corpora here are built so that the order
    `Table.compressed` appends in is NOT the order `std::map` iterates in.
    `ctable_unsorted_big.csv` is a 4-input table, so the two orders differ
    in more than a single transposition.
    """
    ct = _load(name)
    raw = _insertion_order(ct)
    ordered = list(ct)
    assert raw != ordered, f"{name} is already in map order -- the corpus has lost its teeth"
    assert sorted(raw, key=ctable_row_key) == ordered
    assert set(raw) == set(ordered)


def test_true_sorts_before_false_which_is_the_counterintuitive_half():
    """`id::logical_true` is 1 and `id::logical_false` is 2 (vertex.h:71-74
    -- "constants are put first to be in sync with
    lexicographic_subtree_order"), so a `1` cell sorts BEFORE a `0` cell.

    Reading the CSV digits and sorting those would produce the exact
    reverse, which is why this is pinned on its own rather than left
    implicit in the corpus tests.
    """
    assert int(builtin.logical_true) < int(builtin.logical_false)
    ct = _load("ctable_unsorted.csv")
    first, *_, last = list(ct)
    assert first == (builtin.logical_true,) * 3
    assert last == (builtin.logical_false,) * 3


# --------------------------------------------------------------- oracle


@pytest.mark.oracle
def test_ctable_iterates_in_the_order_classic_measured():
    """Pinned against classic's own `CTable::remove_rows` walk over this
    exact CSV, captured in `tests/fixtures/fs_diversity.json`.

    Classic iterates `(1,1,1), (1,0,1), (0,1,0), (0,0,0)` where the file
    compresses to `(1,1,1), (0,0,0), (1,0,1), (0,1,0)`.
    """
    classic_rows = [tuple(_IN[s] for s in row["inputs"]) for row in _FS_DIVERSITY["ctable"]]
    assert len(classic_rows) == 4
    ct = _load("ctable_unsorted.csv")
    assert _insertion_order(ct) != classic_rows
    assert list(ct) == classic_rows


@pytest.mark.oracle
def test_every_public_accessor_agrees_with_the_map_order():
    """`keys()`, `values()`, `items()`, `__iter__` and `reversed()` must all
    report the same walk: `precision_bscore.bias_selection` walks
    `_wrk_ctable` THREE times in two different methods and indexes a shared
    `hypoth[i]` by position across all three (precision_bscore.cc:409-418),
    so a disagreement between two accessors would misalign an ensemble
    score rather than merely reorder a printout.
    """
    ct = _load("ctable_unsorted_big.csv")
    expected = sorted(dict.__iter__(ct), key=ctable_row_key)
    assert list(ct) == expected
    assert list(ct.keys()) == expected
    assert [k for k, _v in ct.items()] == expected
    assert list(ct.values()) == [ct[k] for k in expected]
    assert list(reversed(ct)) == list(reversed(expected))
    # repr is the nearest analogue of `ostreamCTable` (table_io.cc:1359),
    # which prints in map order too.
    assert repr(ct).index(repr(expected[0])) < repr(ct).index(repr(expected[-1]))


def test_repeated_walks_are_identical():
    """Three separate walks of one table must agree element for element.

    `ctruth_table_bscore` builds the candidate bscore in one walk
    (bscores.cc:283) and `set_best_possible_bscore` builds the vector it is
    subtracted from in another (bscores.cc:352); a walk that re-sorted
    differently on the second pass would silently misalign them.
    """
    ct = _load("ctable_unsorted_big.csv")
    walks = [list(ct) for _ in range(3)]
    assert walks[0] == walks[1] == walks[2]


# --------------------------------------------- the rank-then-value rule


def test_ordering_is_rank_then_value_not_a_bare_comparison():
    """`contin_t` is `float` and `builtin` is an `IntEnum`, so
    `1.0 == builtin.logical_true` is `True` in bare Python where C++'s
    variant alternatives never compare equal (docs/CARRY-FORWARD.md 2.5 --
    both of this project's Criticals were this confusion).

    `boost::variant::operator<` compares `which()` FIRST
    (boost/variant/variant.hpp:2134), and `contin_t` is alternative 0 while
    `builtin` is alternative 2 (vertex.h:128), so the contin row sorts
    first REGARDLESS of the values beside it. A bare `sorted()` sees
    `1.0 == 1` at position 0, falls through to position 1, and puts the
    builtin row first.

    Reachable, not hypothetical: `feature_selector.build_fs_ctable` stamps
    `builtin.null_vertex` into a column that may hold `contin_t`.
    """
    contin_row = (contin_t(1.0), contin_t(5.0))
    builtin_row = (builtin.logical_true, contin_t(0.0))
    ct = CTable("y", ["a", "b"])
    ct[builtin_row] = TimedCounter()
    ct[contin_row] = TimedCounter()

    assert sorted([contin_row, builtin_row]) == [builtin_row, contin_row]  # the trap
    assert list(ct) == [contin_row, builtin_row]  # the rule


def test_ordering_survives_a_column_a_bare_sort_cannot_compare():
    """An `enum_t` (a `str`) beside a `builtin` raises `TypeError` under a
    bare `sorted()`; under the variant rule they simply rank 1 and 2.

    This is the shape `build_fs_ctable` produces when it blanks an ignored
    feature of an enum column with `builtin.null_vertex`.
    """
    enum_row = (enum_t("alpha"),)
    builtin_row = (builtin.null_vertex,)
    ct = CTable("y", ["a"])
    ct[builtin_row] = TimedCounter()
    ct[enum_row] = TimedCounter()

    with pytest.raises(TypeError):
        sorted([enum_row, builtin_row])
    assert list(ct) == [enum_row, builtin_row]


def test_contin_rows_sort_numerically_ascending():
    """`contin_seq` is `std::vector<contin_t>`, so its `operator<` is
    lexicographic over `double`s -- the ordinary numeric order, not the
    textual one the CSV would give.
    """
    ct = _load("contin_a4.csv")
    rows = list(ct)
    assert rows == sorted(rows)
    assert len(rows) > 4


# ------------------------------------------------- mutation and rebuild


def test_a_new_key_inserted_after_a_walk_is_re_sorted():
    """The lazy reorder must not latch. `CTable::filtered` and
    `_copy_ctable` both build a table by inserting into it, and
    `ss_bscore`/`ignore_rows` rebuild `_wrk_ctable` between scoring calls.
    """
    ct = _load("ctable_unsorted.csv")
    assert list(ct)[0] == (builtin.logical_true,) * 3  # force one ordering pass
    ct[(builtin.logical_true, builtin.logical_true, builtin.logical_false)] = TimedCounter()
    rows = list(ct)
    assert rows[1] == (builtin.logical_true, builtin.logical_true, builtin.logical_false)
    assert rows == sorted(rows, key=ctable_row_key)


def test_setdefault_marks_the_table_unordered():
    """`Table.compressed` inserts EVERY row through `setdefault`
    (`++res[*in_it][...]`, table.cc:512), and C-level `dict.setdefault` does
    not route through an overridden `__setitem__` -- so without its own
    override every CTable in this port would be built and left unsorted.
    """
    ct = CTable("y", ["a"])
    ct.setdefault((builtin.logical_false,), TimedCounter())
    assert list(ct) == [(builtin.logical_false,)]
    ct.setdefault((builtin.logical_true,), TimedCounter())
    assert list(ct) == [(builtin.logical_true,), (builtin.logical_false,)]


def test_update_marks_the_table_unordered():
    """Same reason as `setdefault`: C-level `dict.update` bypasses
    `__setitem__` too.
    """
    ct = CTable("y", ["a"])
    ct.update({(builtin.logical_false,): TimedCounter()})
    ct.update({(builtin.logical_true,): TimedCounter()})
    assert list(ct) == [(builtin.logical_true,), (builtin.logical_false,)]


def test_reordering_preserves_every_row_and_count():
    """The reorder rebuilds the underlying `dict` in place; nothing may be
    lost, merged or duplicated on the way through.
    """
    ct = _load("ctable_unsorted_big.csv")
    # Read the content through the unbound `dict.items` so the snapshot is
    # taken BEFORE `_order` rebuilds the storage.
    before = {row: dict(counter) for row, counter in dict.items(ct)}
    after = {row: dict(counter) for row, counter in ct.items()}
    assert before == after
    assert len(before) == len(ct)
    assert ct.uncompressed_size() == 30.0
    assert sum(sum(c.values()) for c in ct.values()) == 30.0


def test_filtered_is_sorted_by_its_own_post_filter_key():
    """`CTable::filtered` builds a FRESH `std::map` keyed on the filtered
    row (table.h:619), so its order is re-derived -- sorted by the pre-filter
    key is not the same thing, and two rows that collide after filtering
    merge into one entry.
    """
    ct = _load("ctable_unsorted_big.csv")
    filtered = ct.filtered([1, 3])
    rows = list(filtered)
    assert rows == sorted(rows, key=ctable_row_key)
    assert len(rows) < len(ct)
    assert filtered.uncompressed_size() == ct.uncompressed_size()
