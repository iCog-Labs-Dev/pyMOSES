"""Oracle tests for the CONTINUOUS branches of
pymetta.feature_selection.scorers.mutual_info.

## Oracle: tools/cpp_probe/mi_contin_probe.cc -> tests/fixtures/mi_contin.json

test_mutual_info.py covers the all-discrete-inputs / discrete-output case.
This module covers the two branches that were gated with
``NotImplementedError`` until the contin slice landed:

* a contin OUTPUT over all-discrete inputs (table.h:1174-1189, :1219-1223)
  -- the ``discretize_contin_feature`` / ``get_discrete_bin`` binning, over
  a min/max scan seeded at ``100000.0`` / ``0.0`` rather than at the data's
  extremes;
* the fully continuous case (table.h:1250-1292) -- ``-KLD(p, q)``.

Plus ``mutualInformationBtwSets`` (table.h:1311) on contin-input tables,
which classic has always handled and this port used to refuse.

THE CORPUS IS BUILT AROUND TIES AND EDGES, deliberately. A corpus of
distinct values would pass with every ordering trap broken:

* ``mi_contin_out_bins.csv`` -- output values 0, 2, 4, 6, 8, 10 land
  EXACTLY on the bin edges the ``(0, 10)`` scan produces, so ``<`` vs
  ``<=`` in ``get_discrete_bin`` is decidable; several rows repeat an
  output so counts exceed 1; every input tuple carries two distinct
  outputs so a compressed row has more than one counter entry.
* ``mi_contin_out_neg.csv`` -- every output is negative, so ``max`` never
  moves off its ``0.0`` seed.
* ``mi_contin_out_big.csv`` -- every output is STRICTLY above 100000
  (100000.5 is the smallest), so ``min`` never moves off its ``100000.0``
  seed AND the seed differs from the data's own minimum -- which is what
  makes "seed at the data's extremes instead" a killable mutation. The
  edges also land on inexact doubles (interval 11/5).
* ``mi_contin_in_ties.csv`` -- duplicate ``x1`` values across rows (ties
  feeding the ``std::multimap`` key), duplicate ``y`` values (ties feeding
  ``Counter``), and one row repeated verbatim so its compressed count is
  2 and ``dorepeat`` runs twice.

Every comparison is exact equality on the full double, never
``pytest.approx`` -- see test_mutual_info.py's module docstring.
"""

import json
import math
from pathlib import Path

import pytest

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import CTable
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, type_node
from pymetta.feature_selection.scorers.mutual_info import (
    TARGET_DISCRETIZED_BINS_NUM,
    discretize_contin_feature,
    get_discrete_bin,
    mutual_information,
    mutual_information_between_sets,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mi_contin.json"
_TABLE_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "table_data"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())

# Keyed the way the probe names them, so a fixture group and a table are
# looked up by the same string.
_CSV_FOR = {
    "bins": "mi_contin_out_bins.csv",
    "neg": "mi_contin_out_neg.csv",
    "big": "mi_contin_out_big.csv",
    "ties": "mi_contin_in_ties.csv",
    "a4": "contin_a4.csv",
}


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


@pytest.fixture(scope="module")
def ctables() -> dict[str, CTable]:
    return {
        name: loadTable(str(_TABLE_DATA_DIR / csv), "y").compressed()
        for name, csv in _CSV_FOR.items()
    }


def _flatten(group: str) -> list[tuple]:
    """(table name, case) pairs for a fixture group, with readable ids."""
    return [(name, case) for name, cases in _GOLDEN[group].items() for case in cases]


def _mi_ids(group: str) -> list[str]:
    return [f"{name}-fs{case['fs']}" for name, case in _flatten(group)]


# --------------------------------------------------------------------------
# discretize_contin_feature / get_discrete_bin
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["discretize"],
    ids=[f"{c['min']}..{c['max']}" for c in _GOLDEN["discretize"]],
)
def test_discretize_contin_feature_matches_classic(case):
    assert discretize_contin_feature(case["min"], case["max"]) == case["intervals"]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case",
    _GOLDEN["discretize"],
    ids=[f"{c['min']}..{c['max']}" for c in _GOLDEN["discretize"]],
)
def test_get_discrete_bin_matches_classic(case):
    intervals = discretize_contin_feature(case["min"], case["max"])
    got = [int(get_discrete_bin(intervals, b["value"])) for b in case["bins"]]
    assert got == [b["bin"] for b in case["bins"]]


@pytest.mark.oracle
def test_the_discretize_corpus_actually_lands_on_bin_edges(golden):
    """Corpus guard. ``<`` vs ``<=`` in get_discrete_bin is decidable ONLY
    for a value exactly equal to an edge; if every probe value sat strictly
    between edges, `test_get_discrete_bin_matches_classic` would pass with
    the comparison broken. Count how many probe values are exactly an
    interior edge (``disc_intvs[1..4]`` -- index 0 is never compared,
    because the loop starts at 1)."""
    on_edge = 0
    for case in golden["discretize"]:
        edges = set(case["intervals"][1:])
        on_edge += sum(1 for b in case["bins"] if b["value"] in edges)
    assert on_edge >= 8, f"only {on_edge} probe values land exactly on a bin edge"


def test_get_discrete_bin_uses_a_strict_less_than():
    """Stated at the value level so the mutant is obvious: with edges
    [0, 2, 4, 6, 8], the value 4.0 is bin 3 (the bin it OPENS), not 2."""
    intervals = discretize_contin_feature(0.0, 10.0)
    assert intervals == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert int(get_discrete_bin(intervals, 4.0)) == 3
    assert int(get_discrete_bin(intervals, math.nextafter(4.0, 0.0))) == 2


def test_get_discrete_bin_returns_builtin_ordinals_one_through_five():
    """The return type is a ``builtin`` carrying the loop counter, per
    table.h:63-64's own "trick vertex" comment. Never ``null_vertex``."""
    intervals = discretize_contin_feature(0.0, 10.0)
    got = [get_discrete_bin(intervals, v) for v in (-1.0, 2.0, 4.0, 6.0, 8.0)]
    assert got == [
        builtin.logical_true,
        builtin.logical_false,
        builtin.logical_and,
        builtin.logical_or,
        builtin.logical_not,
    ]
    assert [int(b) for b in got] == [1, 2, 3, 4, 5]
    assert builtin.null_vertex not in got


def test_discretize_returns_five_lower_edges_and_no_upper_edge():
    intervals = discretize_contin_feature(0.0, 10.0)
    assert len(intervals) == TARGET_DISCRETIZED_BINS_NUM == 5
    assert intervals[0] == 0.0
    assert 10.0 not in intervals


# --------------------------------------------------------------------------
# The min/max scan seeds
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN["scan"]))
def test_scan_min_max_matches_classic(golden, ctables, name):
    """The scan is inline in ``mutualInformation`` (table.h:1172-1187), so
    it is re-derived here the same way and checked against what classic
    arrived at. The END-TO-END oracle for it is the mi_contin_out group
    below; this test localises a failure when that one goes red."""
    min_ = 100000.0
    max_ = 0.0
    for counter in ctables[name].values():
        for value in counter:
            if value < min_:
                min_ = value
            if value > max_:
                max_ = value
    assert min_ == golden["scan"][name]["min"]
    assert max_ == golden["scan"][name]["max"]
    assert discretize_contin_feature(min_, max_) == golden["scan"][name]["intervals"]


@pytest.mark.oracle
def test_the_seeds_are_reached_by_two_of_the_three_tables(golden, ctables):
    """Corpus guard for the ``min = 100000.0; max = 0.0`` seeds. On
    ``big`` the data never lowers ``min``; on ``neg`` it never raises
    ``max``. If both tables were replaced by ordinary positive data, the
    "seed at the data's extremes instead" mutation would kill nothing and
    the DO NOT FIX note would be untested."""
    stuck_min = stuck_max = 0
    for name in ("bins", "neg", "big"):
        values = [v for counter in ctables[name].values() for v in counter]
        if golden["scan"][name]["min"] == 100000.0 and min(values) != 100000.0:
            stuck_min += 1
        if golden["scan"][name]["max"] == 0.0 and max(values) != 0.0:
            stuck_max += 1
    assert stuck_min == 1, "no table leaves `min` stuck at its 100000.0 seed"
    assert stuck_max == 1, "no table leaves `max` stuck at its 0.0 seed"


# --------------------------------------------------------------------------
# The compressed tables themselves
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_GOLDEN["ctables"]))
def test_compressed_table_matches_classic(golden, ctables, name):
    """Guards the corpus rather than the port's arithmetic: if this port's
    loader ever compressed these CSVs differently from classic, every MI
    value below would be comparing two different tables and the agreement
    would mean nothing."""
    want = golden["ctables"][name]
    got = ctables[name]
    assert got.get_arity() == want["arity"]
    got_rows = sorted(
        (
            tuple(float(v) for v in row),
            sorted((float(y), float(n)) for y, n in counter.items()),
        )
        for row, counter in got.items()
    )
    want_rows = sorted(
        (
            tuple(float(v) for v in r["key"]),
            sorted((float(c["y"]), float(c["n"])) for c in r["counts"]),
        )
        for r in want["rows"]
    )
    assert got_rows == want_rows


@pytest.mark.oracle
def test_the_ties_corpus_really_has_ties(ctables):
    """Corpus guard for the multimap/Counter ordering traps: duplicate x
    keys, duplicate y values, and at least one compressed count above 1."""
    ties = ctables["ties"]
    xs = [row[0] for row in ties]
    ys = [y for counter in ties.values() for y in counter]
    counts = [n for counter in ties.values() for n in counter.values()]
    assert len(xs) > len(set(xs)), "no duplicate x values feeding the multimap"
    assert len(ys) > len(set(ys)), "no duplicate y values feeding Counter"
    assert max(counts) > 1.0, "no compressed row with a count above 1"


# --------------------------------------------------------------------------
# mutualInformation -- contin output, discrete inputs
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name,case", _flatten("mi_contin_out"), ids=_mi_ids("mi_contin_out"))
def test_mutual_information_contin_output_matches_classic(ctables, name, case):
    assert mutual_information(ctables[name], case["fs"]) == case["mi"]


# --------------------------------------------------------------------------
# mutualInformation -- the fully continuous branch
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name,case", _flatten("mi_contin_in"), ids=_mi_ids("mi_contin_in"))
def test_mutual_information_contin_input_matches_classic(ctables, name, case):
    assert mutual_information(ctables[name], case["fs"]) == case["mi"]


@pytest.mark.oracle
def test_contin_branch_returns_the_negated_kld(golden, ctables):
    """``contin_t ic = - KLD(p,q);`` (table.h:1287). Half of the fixture's
    contin-input values are NEGATIVE and half positive, so dropping the
    minus is not a sign that cancels out of the comparison anywhere."""
    values = [case["mi"] for cases in golden["mi_contin_in"].values() for case in cases]
    assert any(v < 0 for v in values)
    assert any(v > 0 for v in values)


@pytest.mark.oracle
def test_fractional_counts_truncate_like_an_unsigned(golden, ctables):
    """``unsigned flt_count = val_pair.second;`` (table.h:1266) truncates a
    ``count_t`` double. The fixture holds the SAME table with every count
    scaled off the integers (one below 1.0, so its pair vanishes entirely);
    the counts are replayed onto a real CTable here rather than recomputed,
    so the port and classic are looking at identical inputs."""
    scaled = golden["mi_contin_in_fractional"]["ties_scaled"]
    ctable = loadTable(str(_TABLE_DATA_DIR / _CSV_FOR["ties"]), "y").compressed()
    wanted = {
        tuple(float(v) for v in r["key"]): {float(c["y"]): float(c["n"]) for c in r["counts"]}
        for r in scaled["rows"]
    }
    replaced = 0
    for row, counter in ctable.items():
        target = wanted[tuple(float(v) for v in row)]
        # `add` accumulates, so a delta re-writes the count while keeping
        # the cell's timed and untimed views in step (a plain `__setitem__`
        # is refused by `TimedCounter` for exactly that reason).
        for tv, count in list(counter.timed_items()):
            counter.add(tv.value, tv.timestamp, target[float(tv.value)] - count)
            replaced += 1
    assert replaced == sum(len(r["counts"]) for r in scaled["rows"])
    assert any(c % 1.0 for counter in ctable.values() for c in counter.values())

    for case in golden["mi_contin_in_fractional"]["ties"]:
        assert mutual_information(ctable, case["fs"]) == case["mi"]


def test_contin_branch_rejects_more_than_one_feature(ctables):
    """``OC_ASSERT(0, "Contin MI currently supports only 1 feature.")``
    (table.h:1252-1254)."""
    with pytest.raises(AssertionError, match="only 1 feature"):
        mutual_information(ctables["ties"], {0, 1})


def test_contin_branch_rejects_an_empty_feature_set(ctables):
    """C++ reads past the end of the filtered row here (undefined
    behaviour); this port raises instead -- see the function's own note."""
    with pytest.raises(IndexError):
        mutual_information(ctables["ties"], set())


def test_unsupported_output_type_raises(ctables):
    """Not-all-discrete inputs with a non-contin output is C++'s own
    ``else``: ``OC_ASSERT(0, "Type ... is not supported ...")``
    (table.h:1298-1301)."""
    sig = TypeSignature([type_node.contin_type], type_node.boolean_type)
    ctable = CTable("y", ["x1"], sig)
    with pytest.raises(AssertionError, match="not supported for mutual information"):
        mutual_information(ctable, {0})


# --------------------------------------------------------------------------
# mutualInformationBtwSets
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name,case",
    [(n, c) for n, cases in _GOLDEN["mi_btw_sets"].items() for c in cases],
    ids=[
        f"{n}-{c['fs_l']}|{c['fs_r']}" for n, cases in _GOLDEN["mi_btw_sets"].items() for c in cases
    ],
)
def test_mutual_information_between_sets_matches_classic(ctables, name, case):
    """Includes the contin-input tables (``ties``, ``a4``), which this port
    used to refuse outright. ``mutualInformationBtwSets`` has no contin
    branch in C++ at all -- it gates on the OUTPUT type only and never
    reads the output column -- so contin inputs simply work there."""
    got = mutual_information_between_sets(ctables[name], case["fs_l"], case["fs_r"])
    assert got == case["mi"]


def test_mutual_information_between_sets_rejects_only_the_output_type(ctables):
    """The surviving gate is C++'s: ``OC_ASSERT(0, "Unsupported type for
    mutual information")`` (table.h:1394), keyed on the output type alone.
    An unknown-typed ctable is the only thing that trips it."""
    sig = TypeSignature([type_node.contin_type], type_node.unknown_type)
    ctable = CTable("y", ["x1"], sig)
    with pytest.raises(AssertionError, match="Unsupported type"):
        mutual_information_between_sets(ctable, {0}, set())
