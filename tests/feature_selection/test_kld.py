"""Oracle and guard tests for pymoses.feature_selection.scorers.kld.

## Oracle: tools/cpp_probe/mi_contin_probe.cc -> tests/fixtures/mi_contin.json

Two fixture groups drive this module:

* ``"kld"`` -- classic's ``KLD(p, q)`` helper (``opencog/util/KLD.h:202``)
  on ten hand-built double vectors, chosen so that every branch of
  ``KLDS::next`` is taken by something: ties in ``p`` (the ``Counter``
  collapse plus ``precompute_delta_p``'s running ``p_x_pre``), ties in
  ``q`` (``n_duplicates``), ``q`` wholly below ``p`` (``cit_q`` walks to
  ``cend()`` and ``q_x`` becomes ``x_very_last``), ``q`` wholly above
  ``p`` (``q_x_pre`` never moves off ``x_very_first``), unequal sizes,
  negatives, a singleton ``p``, a wide dynamic range, and one case that
  is a SCRAMBLED restatement of an earlier one.
* ``"klds_direct"`` -- ``KLDS<contin_t>`` driven through its own public
  API (``p_size``, ``p_pdf_size``, the ``Out``-iterator ``operator()``,
  the scalar ``operator()``) with a hand-built ``q_counter``. Two of the
  three cases carry NON-INTEGRAL counts, which is the only way to make
  ``boost::accumulate(q_counter | map_values, 0)``'s ``int`` init
  observable -- ``KLD(p, q)`` itself can never reach it, because it
  always builds ``q_counter`` with ``Counter``.

Every comparison is exact equality on the full double (``%.17g``
round-trips a double bit for bit), never ``pytest.approx`` -- see
test_mutual_info.py's own module docstring for why this project holds
that line.
"""

import json
import math
from pathlib import Path

import pytest

from pymoses.feature_selection.scorers.kld import (
    KLD,
    KLDS,
    _accumulate_with_int_init,
    _counter,
)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mi_contin.json"
_GOLDEN = json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def golden():
    return _GOLDEN


def _ids(cases):
    return [case["name"] for case in cases]


# --------------------------------------------------------------------------
# KLD(p, q) -- the function helper
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["kld"], ids=_ids(_GOLDEN["kld"]))
def test_kld_matches_classic(case):
    assert KLD(case["p"], case["q"]) == case["kld"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["kld"], ids=_ids(_GOLDEN["kld"]))
def test_negated_kld_matches_classic(case):
    """``contin_t ic = - KLD(p,q);`` (table.h:1287) -- the sign flip
    mutual_information's contin branch applies, pinned here at the source
    so that dropping the minus there is a fixture-visible change and not
    merely a sign somebody has to eyeball."""
    assert -KLD(case["p"], case["q"]) == case["neg_kld"]


@pytest.mark.oracle
def test_kld_ignores_input_order(golden):
    """``dup_in_p`` and ``dup_in_p_scrambled`` are the same two multisets
    in different orders, and classic returns the same bits for both. That
    is not an accident of the data: ``KLD`` reads ``p`` only as
    ``Counter(p)`` plus ``len(p)`` and ``q`` only as ``Counter(q)``, so
    the ``std::multimap`` ordering at table.h:1255 -- which decides the
    element order of ``p`` and ``q`` -- cannot reach the result.

    This test exists to keep that claim honest rather than to catch a
    regression: it is the reason the "iterate the multimap unsorted"
    mutation kills nothing (see test_mutual_info_contin.py)."""
    cases = {case["name"]: case for case in golden["kld"]}
    plain = cases["dup_in_p"]
    scrambled = cases["dup_in_p_scrambled"]
    assert sorted(plain["p"]) == sorted(scrambled["p"])
    assert sorted(plain["q"]) == sorted(scrambled["q"])
    assert plain["kld"] == scrambled["kld"]
    assert KLD(plain["p"], plain["q"]) == KLD(scrambled["p"], scrambled["q"])


# --------------------------------------------------------------------------
# KLDS -- the stateful form
# --------------------------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["klds_direct"], ids=_ids(_GOLDEN["klds_direct"]))
def test_klds_scalar_call_matches_classic(case):
    q_counter = {key: value for key, value in case["q_counter"]}
    assert KLDS(case["p"])(q_counter) == case["kld"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["klds_direct"], ids=_ids(_GOLDEN["klds_direct"]))
def test_klds_components_match_classic(case):
    """The ``Out``-iterator overload (KLD.h:170), element for element."""
    q_counter = {key: value for key, value in case["q_counter"]}
    assert KLDS(case["p"]).components(q_counter) == case["components"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", _GOLDEN["klds_direct"], ids=_ids(_GOLDEN["klds_direct"]))
def test_klds_sizes_match_classic(case):
    klds = KLDS(case["p"])
    assert klds.p_size() == case["p_size"]
    assert klds.p_pdf_size() == case["p_pdf_size"]


@pytest.mark.oracle
def test_fractional_counts_reach_the_int_truncation(golden):
    """The ``int``-init accumulation is only observable through ``KLDS``'s
    own ``operator()``, and these two fixture cases are the ones that
    reach it. Guards the corpus itself: if someone replaced their
    q_counters with whole numbers, `test_klds_scalar_call_matches_classic`
    would still pass while covering nothing."""
    reached = 0
    for case in golden["klds_direct"]:
        counts = [value for _, value in case["q_counter"]]
        exact = math.fsum(counts)
        truncated = _accumulate_with_int_init(counts)
        if exact != truncated:
            reached += 1
    assert reached >= 2, "no klds_direct case exercises the int truncation"


def test_accumulate_truncates_at_every_step():
    """``0 + 0.6 -> 0``, three times over, is 0 -- not 1 (which a single
    truncation of the exact 1.8 would give) and not 1.7999999999999998."""
    assert _accumulate_with_int_init([0.6, 0.6, 0.6]) == 0.0
    assert _accumulate_with_int_init([1.5, 2.5]) == 3.0
    assert _accumulate_with_int_init([2.7, 3.9]) == 5.0
    # Whole numbers pass through untouched -- the domain KLD(p,q) reaches.
    assert _accumulate_with_int_init([1.0, 2.0, 3.0]) == 6.0


def test_counter_counts_occurrences_and_keeps_first_appearance_order():
    """``Counter``'s sequence constructor (Counter.h:80) counts, and this
    port deliberately does NOT pre-sort the result -- every ``sorted()``
    in kld.py has to do its own work, so each is independently killable
    by a mutation test."""
    counter = _counter([3.0, 1.0, 3.0, 2.0, 1.0, 3.0])
    assert counter == {1.0: 2.0, 2.0: 1.0, 3.0: 3.0}
    assert list(counter) == [3.0, 1.0, 2.0]


def test_set_p_pdf_does_not_alias_the_callers_counter():
    """``p_pdf = p_counter`` is a ``std::map`` COPY assignment (KLD.h:68),
    and ``precompute_delta_p`` then overwrites every value in place. A
    Python assignment would have aliased the caller's dict and destroyed
    it -- docs/CARRY-FORWARD.md Sec 1.1 is the same shape of bug."""
    mine = {1.0: 2.0, 4.0: 3.0}
    caller_copy = dict(mine)
    klds = KLDS()
    klds.set_p_pdf(caller_copy, 5.0)
    assert caller_copy == mine
    # x_very_first is 1.0 - margin == 0.0, so the gaps are 1.0 and 3.0 and
    # the densities really did replace the counts in the copy.
    assert klds.p_pdf == {1.0: 2.0, 4.0: 1.0}


def test_x_very_first_and_last_come_from_the_extremes_not_the_insertion_order():
    """``p_pdf.cbegin()->first`` / ``crbegin()->first`` on a ``std::map``
    are the smallest and largest keys. Fed a deliberately unsorted ``p``,
    an insertion-ordered ``dict`` would give 3.0 and 2.0 instead."""
    klds = KLDS([3.0, 1.0, 2.0])
    assert klds.x_very_first == 0.0  # min 1.0 - margin 1.0
    assert klds.x_very_last == 4.0  # max 3.0 + margin 1.0


def test_next_asserts_once_p_is_exhausted():
    """``OC_ASSERT(cit_p != p_pdf.end())`` (KLD.h:136) is live in release
    builds, so it is an ``AssertionError`` and not a bare ``assert``."""
    from pymoses.feature_selection.scorers.kld import _kld_iters

    klds = KLDS([1.0, 2.0])
    q_items = sorted(_counter([1.5, 2.5]).items())
    iters = _kld_iters(klds.x_very_first)
    klds.next(q_items, 2.0, iters)
    klds.next(q_items, 2.0, iters)
    with pytest.raises(AssertionError):
        klds.next(q_items, 2.0, iters)
