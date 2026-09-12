"""C++: moses/comboreduct/table/table.cc

``subsampleCTable`` (table.cc:899) and the ``CTable::remove_rows``
(table.cc:646) it is the only caller of -- the row-subsampling half of
``--fs-subsampling-ratio`` (problem-params.cc's own
``--fs-subsampling-ratio``, read by
``feature_selector::build_fs_ctable``, feature_selector.cc:205-218).

# DIVERGENCE(C++ moses/comboreduct/table/table.cc:899): PLACEMENT ONLY.
# Both functions' C++ home is ``moses/comboreduct/table/table.{h,cc}``,
# i.e. ``pymoses/comboreduct/table/table.py`` -- which is off limits for
# this port's task partition (another agent owns it), so they land in
# ``pymoses.feature_selection`` instead. Exactly the precedent
# ``pymoses/feature_selection/scorers/mutual_info.py``'s own module
# docstring sets: it ports ``mutualInformation``/
# ``mutualInformationBtwSets``/``OTEntropy`` out of the same comboreduct
# header for the same reason, and notes that in classic itself those are
# feature-selection-only code despite living in a comboreduct header.
# ``subsampleCTable`` is likewise reachable, in the whole of classic,
# from exactly two call sites -- ``feature_selector::build_fs_ctable``
# (feature_selector.cc:217) and ``feature-selection.cc``'s own
# ``subsampleCTable(fs_params.subsampling_ratio, ctable)``
# (feature-selection.cc:110) -- both feature selection. No behaviour is
# changed by the move; only the module a reader finds them in.
#
# NOT moved with them: ``subsampleTable`` (table.cc:872/:892, the
# uncompressed-``Table`` sibling) and ``CTable::remove_rows_at_times``/
# ``remove_rows_at_time`` (table.cc:707/:713, the ``subsampling_by_time``
# path). The first has no feature-selection call site at all
# (``--nsamples`` reaches it through ``table-problems.cc``); the second
# needs ``TTable``/timestamps, which this port's ``CTable`` does not
# carry at all (see ``pymoses.comboreduct.table.table``'s own module
# docstring and CARRY-FORWARD.md section 3).

Scope of what IS reproduced here, deliberately including two upstream
defects:

* ``CTable::remove_rows``'s own ``OC_ASSERT(0.0 == (total_row_weights -
  truncate), "This algo is broken!")`` (table.cc:668) is reproduced as a
  raise, NOT fixed. Classic's own comment three lines above it
  (table.cc:661-665) reads: "XXX this cannot possibly be correct, the
  total count is in general a fraction, not an integer; it is merely the
  sum of the weights of the rows. It is NOT equal to the toal number of
  rows! I cannot figure out what this algo is trying to do, so I can't
  actually fix it :-(". A weighted ``CTable``
  (``Table.compressed(weighting_feature)``, e.g.
  ``tests/fixtures/table_data/weighted.csv``) reaches it.
* ``unsigned i_v_end = i + v_it->second;`` (table.cc:676) narrows a
  ``count_t`` (``double``) sum to ``unsigned`` by C truncation, and
  ``unsigned i_end = i + ((unsigned) truncate);`` (table.cc:669) does the
  same -- both reproduced with an explicit 32-bit mask rather than a
  clamp, per this project's own convention.
"""

from __future__ import annotations

import math
import struct

from pymoses.comboreduct.combo.vertex import vertex, vertex_rank
from pymoses.comboreduct.table.table import CTable
from pymoses.util.lazy_random_selector import lazy_random_selector

__all__ = ["ctable_remove_rows", "subsampleCTable"]

# C's `unsigned` is 32 bits on every target this port compares against.
# `i`, `i_end` and `i_v_end` in `CTable::remove_rows` are all `unsigned`.
_U32_MASK = 0xFFFFFFFF


def _to_float32(x: float) -> float:
    """No direct C++ symbol: the implicit ``double`` -> ``float``
    conversion C++ performs when a ``double`` argument binds to
    ``subsampleCTable``'s ``float ratio`` parameter (table.h:1034,
    table.cc:899).

    ``feature_selector_parameters::subsampling_ratio`` is a ``double``
    (feature_selector.h:120) and ``feature_selection_parameters::
    subsampling_ratio`` is a ``double`` too (feature-selection.h:81), so
    EVERY real caller narrows to IEEE-754 binary32 at the call, before
    ``1.0 - ratio`` is ever computed. Same binary32 rounding as
    ``pymoses.moses.moses.types._to_score_t``, re-implemented here rather
    than imported: this value is a plain C ``float``, not a ``score_t``,
    and ``pymoses.feature_selection`` has no other dependency on
    ``pymoses.moses.moses``.
    """
    return struct.unpack("f", struct.pack("f", x))[0]


def _ctable_row_key(row: tuple[vertex, ...]) -> tuple[tuple[int, vertex], ...]:
    """No direct C++ symbol: the sort key that reproduces
    ``std::map<multi_type_seq, TimedCounter>``'s own ascending-key
    iteration order for this port's ``dict``-based ``CTable``
    (table.h:550, ``CTable : public std::map<...>``).

    ``multi_type_seq::operator<`` (table.h:383) is
    ``get_variant() < r.get_variant()`` over
    ``boost::variant<builtin_seq, contin_seq, string_seq, vertex_seq,
    combo_tree_seq>``: the held alternative first, then that alternative's
    own ``std::vector`` lexicographic ``operator<``. This port's ``CTable``
    key is a flat ``tuple[vertex, ...]`` instead (see that class's own
    docstring), whose every POSITION holds one and only one vertex kind for
    the table's whole lifetime -- so an element-wise
    ``(vertex_rank, value)`` key reproduces the ``builtin_seq`` and
    ``contin_seq`` cases (one kind everywhere -> the rank is constant and
    the comparison reduces to the values') and the ``vertex_seq`` case
    (``boost::variant<...>::operator<`` is exactly rank-then-value) alike.

    For this port's boolean tables the whole thing collapses to plain
    tuple comparison over ``builtin``'s ordinals -- ``logical_true`` is 1
    and ``logical_false`` is 2 (vertex.py:64-65, transcribed from
    vertex.h:74), so a `1` cell sorts BEFORE a `0` cell, which is the
    opposite of the naive reading and is exactly why this is a sort key
    and not a `sorted(ctable)` call.
    """
    return tuple((vertex_rank(v), v) for v in row)


def _observation_sort_key(item: tuple) -> tuple:
    """No direct C++ symbol: the sort key that reproduces
    ``Counter<TimedValue, count_t>``'s (``std::map``'s) own ascending-key
    iteration order for one ``CTable`` cell (table.h:496).

    A cell's key is a ``TimedValue`` -- ``(vertex value, date
    timestamp)`` (table.h:472) -- so the INTENDED order is lexicographic:
    the vertex first (``vertex_rank``/value, as in ``_ctable_row_key``),
    the timestamp second, with the not-a-date sentinel LAST
    (``timestamp_less``'s own measured rule, table.py:804).

    # DIVERGENCE(C++ moses/comboreduct/table/table.h:482): the INTENDED
    # lexicographic order, not ``TimedValue::operator<``'s literal one.
    # That operator reads `(value < r.value) || (timestamp < r.timestamp)`
    # -- no `value == r.value &&` guard -- so it is not a strict weak
    # ordering and the `std::map` it keys is undefined behaviour whenever a
    # cell holds a pair whose value order and date order disagree. This is
    # the SAME divergence `TimedCounter`'s own class docstring
    # (comboreduct/table/table.py) already carries and measures; it is
    # restated here because this function is where the order is chosen for
    # `remove_rows`. For a table with no timestamp column -- every table
    # `subsampleCTable` can reach in this port, since `subsampling_by_time`
    # takes the other branch entirely -- every observation carries the same
    # sentinel date, the timestamp half of the key is constant, and this
    # reduces exactly to ascending vertex order, which classic also
    # produces.
    """
    key = item[0]
    value = getattr(key, "value", key)
    timestamp = getattr(key, "timestamp", None)
    return (vertex_rank(value), value, timestamp is None, timestamp)


def _observations(counter) -> list[tuple]:
    """No direct C++ symbol: one ``CTable`` cell's observations as a list
    of ``(key, count)`` in C++'s own ``std::map`` ascending-key order.

    A cell is a ``TimedCounter`` (``Counter<TimedValue, count_t>``,
    table.h:496) whose keys are ``TimedValue``s. This port's
    ``TimedCounter`` (comboreduct/table/table.py) exposes those through
    ``timed_items()`` while its OWN mapping interface is the untimed
    (vertex-keyed) view -- a divergence documented on that class. Cells
    that carry no timed view at all (a plain ``Counter``-shaped mapping)
    fall back to their vertex-keyed items, which is the same list with the
    timestamp half of every key already collapsed.
    """
    items = counter.timed_items() if hasattr(counter, "timed_items") else counter.items()
    return sorted(items, key=_observation_sort_key)


def _replace_cell(ctable: CTable, row, kept: list[tuple]) -> None:
    """No direct C++ symbol: writes one ``CTable`` cell back after
    ``remove_rows`` decremented and erased entries in it.

    C++ mutates the cell in place (``--v_it->second``,
    ``outputs.erase(v_it)``); this port's ``TimedCounter`` deliberately
    refuses raw ``__setitem__``/``__delitem__`` (they would desync its two
    views), so the surviving observations are rebuilt into a fresh cell of
    the same type -- exactly the pattern ``CTable.remove_rows_at_time``
    (table.py) already uses for the same reason.
    """
    proto = ctable[row]
    replacement = type(proto)()
    for key, count in kept:
        if hasattr(replacement, "add"):
            replacement.add(getattr(key, "value", key), getattr(key, "timestamp", None), count)
        else:
            replacement[key] = count
    ctable[row] = replacement


def _total_count(observations: list[tuple]) -> float:
    """C++: opencog/util/Counter.h:106 Counter::total_count

    ``boost::accumulate(*this | map_values, (CT)0)`` -- a left fold from
    ``(count_t)0`` over the map's own ASCENDING-key iteration order.
    ``count_t`` is ``double``, and double addition is not associative, so
    the order is reproduced (``_observations`` supplies it) rather than
    left to the ``dict``'s insertion order.
    """
    total = 0.0
    for _key, count in observations:
        total += count
    return total


def ctable_remove_rows(ctable: CTable, idxs: set[int]) -> None:
    """C++: moses/comboreduct/table/table.cc:646 CTable::remove_rows

    "Remove the rows of the ctable. It treats rows as if they were
    uncompressed. The indexes follow the order of the rows of the
    uncompressed table" (table.h:678-681's own doc comment). Mutates
    ``ctable`` in place, exactly as the C++ member does.

    Reproduced verbatim, including the parts classic itself flags as
    wrong -- see this module's docstring for the ``OC_ASSERT(0.0 ==
    (total_row_weights - truncate), "This algo is broken!")`` note.

    Two iteration orders are load-bearing and are BOTH restored here from
    C++'s ``std::map``s, because the whole index arithmetic is a running
    offset over them:

    * the rows, ``std::map<multi_type_seq, TimedCounter>`` -- see
      ``_ctable_row_key``;
    * the values inside one row, ``TimedCounter`` ==
      ``Counter<TimedValue, count_t>`` == ``std::map<TimedValue, count_t,
      ...>`` (table.h:496, Counter.h:52).

    Both are plain ``dict``s in this port, i.e. insertion-ordered.
    Measured on ``tests/fixtures/table_data/ctable_unsorted.csv``: the two
    orders disagree for both the rows and the values of two of its four
    compressed rows, and iterating in insertion order there removes a
    different multiset of (row, value) pairs than classic does -- see
    ``tests/feature_selection/test_ctable_subsample.py``, whose oracle
    fixture is the real ``CTable::remove_rows`` output.

    Erasing while iterating is safe in both languages for the same reason:
    C++ advances via ``erase``'s returned iterator (which is the next
    element, leaving every other iterator valid, ``std::map`` being
    node-based), and this port walks a key list snapshotted before the
    loop, which the deletions likewise do not perturb.
    """
    # `auto idx_it = idxs.begin();` -- `idxs` is a `std::set<unsigned>`,
    # so ASCENDING. A Python `set` iterates in hash order.
    idx_seq = sorted(idxs)
    idx_it = 0

    # "iterator index of the CTable from the perspective of an
    # uncompressed table" (table.cc:651-652's own comment)
    i = 0

    row_keys = sorted(ctable.keys(), key=_ctable_row_key)
    row_it = 0

    while row_it < len(row_keys) and idx_it < len(idx_seq):
        row = row_keys[row_it]
        # `auto& outputs = row_it->second;` -- a reference C++ mutates in
        # place. This port collects the same observations up front, in
        # C++'s own map order, mutates the counts here, and writes the
        # surviving ones back once the row is finished (see
        # `_replace_cell`); nothing outside the row is read in between, so
        # the two are equivalent.
        observations = _observations(ctable[row])
        counts = [count for _key, count in observations]
        erased = [False] * len(observations)
        total_row_weights = _total_count(observations)
        truncate = math.floor(total_row_weights)
        # DIVERGENCE(C++ moses/comboreduct/table/table.cc:668): DO NOT FIX --
        # `OC_ASSERT(0.0 == (total_row_weights - truncate), "This algo is
        # broken!")` becomes a raise. Classic's own comment at table.cc:661
        # says the algorithm cannot be right for fractional row weights and
        # that it could not be fixed; a weighted CTable (a `--weighting-
        # feature` / weighted.csv table) reaches this and aborts the classic
        # process. Reproduced, not repaired.
        if total_row_weights - truncate != 0.0:
            raise AssertionError("This algo is broken!")
        i_end = (i + int(truncate)) & _U32_MASK
        if i <= idx_seq[idx_it] < i_end:
            v_it = 0
            while v_it < len(observations) and idx_it < len(idx_seq):
                if not i <= idx_seq[idx_it]:
                    raise AssertionError("There must be a bug")
                # Remove all overlapping indexes with v and advance idx_it
                # of that amount (table.cc:674-675's own comment).
                #
                # `unsigned i_v_end = i + v_it->second;` -- `count_t` is a
                # double, so this is a double sum truncated toward zero on
                # conversion to `unsigned`.
                i_v_end = int(i + counts[v_it]) & _U32_MASK
                while idx_it < len(idx_seq) and idx_seq[idx_it] < i_v_end:
                    if not counts[v_it] > 0:
                        raise AssertionError("There must be a bug")
                    counts[v_it] -= 1
                    idx_it += 1

                # Increment i with the count of that value (table.cc:685).
                i = i_v_end

                # Check if the count went to zero, if so remove v_it
                # entirely (table.cc:689-693). Exact `== 0` on a double,
                # as in C++.
                if counts[v_it] == 0:
                    erased[v_it] = True
                v_it += 1

            kept = [
                (key, count)
                for (key, _old), count, gone in zip(observations, counts, erased, strict=True)
                if not gone
            ]
            # Check if the output is empty, and if so remove the row
            # entirely (table.cc:696-700).
            if not kept:
                del ctable[row]
            else:
                _replace_cell(ctable, row, kept)
        row_it += 1
        i = i_end


def subsampleCTable(ratio: float, ctable: CTable) -> None:
    """C++: moses/comboreduct/table/table.cc:899 subsampleCTable

    "Subsample the input and output tables, after removing the
    indices" -- drops ``(1 - ratio)`` of the UNCOMPRESSED rows, chosen
    without replacement through ``lazy_random_selector`` over the
    process-wide ``randGen()``. Mutates ``ctable`` in place, exactly as
    the C++ free function does (it takes ``CTable&``).

    Every step's typing matters and is reproduced:

    * ``ratio`` is a C ``float`` PARAMETER while every caller's field is a
      ``double`` (``feature_selector_parameters::subsampling_ratio``,
      feature_selector.h:120), so the value is narrowed to binary32 at the
      call -- ``_to_float32`` above.
    * ``unsigned ctable_usize = ctable.uncompressed_size()`` narrows a
      ``count_t`` (``double``) to ``unsigned`` by truncation.
    * ``nremove = (1.0 - ratio) * ctable_usize`` is computed in ``double``
      (the ``float`` ``ratio`` promotes) and then truncated to
      ``unsigned``.

    Draw order is part of this port's bit-identity claim: ``dorepeat`` runs
    ``rm_selector()`` exactly ``nremove`` times, and the ``std::set``
    insert that follows does NOT change how many draws happen or in what
    order (``lazy_random_selector`` never repeats a value, so the set is
    only ever a container). ``lazy_random_selector`` is constructed even
    when ``nremove == 0`` -- and its own ``OC_ASSERT(u - l > 0)``
    (lazy_selector.cc:41) therefore fires on an empty ``CTable``
    regardless of ``ratio``.
    """
    ratio = _to_float32(ratio)
    # DIVERGENCE(C++ moses/comboreduct/table/table.cc:901-902): OC_ASSERT
    # becomes a raise, matching lazy_selector.py's own treatment of the
    # OC_ASSERTs it ports.
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Ratio must be in [0.0, 1.0], but is {ratio:f}")
    rm_row_idxs: set[int] = set()
    ctable_usize = int(ctable.uncompressed_size()) & _U32_MASK
    nremove = int((1.0 - ratio) * ctable_usize) & _U32_MASK
    rm_selector = lazy_random_selector(ctable_usize)
    for _ in range(nremove):
        rm_row_idxs.add(rm_selector())
    ctable_remove_rows(ctable, rm_row_idxs)
