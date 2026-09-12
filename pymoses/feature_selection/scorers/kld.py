"""C++: opencog/util/KLD.h

Kullback-Leibler divergence of two continuous distributions, estimated by
the algorithm of Perez-Cruz, "Kullback-Leibler Divergence Estimation of
Continuous Distributions" (the reference KLD.h's own header comment
cites): ``KLDS`` (KLD.h:59), which holds a fixed P and can be evaluated
against many Q, and the ``KLD(p, q)`` function helper (KLD.h:202) that
wraps it for the one-shot case.

# DIVERGENCE(C++ opencog/util/KLD.h:59): PLACEMENT, not behaviour. `KLDS`
# and `KLD` live in cogutil (`opencog/util/KLD.h`), not in MOSES at all,
# and this port has no `pymoses.opencog.util` / cogutil package -- every
# cogutil symbol it needs is ported into whichever pymoses package
# actually consumes it (`weighted_information`/`entropy` from
# `opencog/util/numeric.h` are already in this very directory, inside
# `mutual_info.py`, and that module's own docstring records the precedent
# and the reason: `pymoses/comboreduct/**` is off limits to this slice).
# The sole consumer of KLD in all of MOSES is
# `mutualInformation(const CTable&, const FeatureSet&)`
# (`moses/comboreduct/table/table.h:1287`), whose port is
# `mutual_info.py`'s `mutual_information` -- next door. Nothing about the
# maths changes; only the file it is spelled in.

Traps reproduced here deliberately, each verified against
``tests/fixtures/mi_contin.json`` (captured by
``tools/cpp_probe/mi_contin_probe.cc``):

* ``Counter<FloatT, FloatT>`` is a ``std::map<FloatT, FloatT>``
  (``opencog/util/Counter.h:50``), so it iterates **ascending by key**,
  and its sequence constructor counts occurrences. A Python ``dict`` is
  insertion-ordered, which is a different order whenever the input is not
  already sorted. Every place C++ leans on the map's ordering --
  ``precompute_delta_p``'s running ``p_x_pre``, ``cbegin()``/``crbegin()``
  for ``x_very_first``/``x_very_last``, the ``cit_p``/``cit_q`` walks in
  ``next``, and ``boost::accumulate``'s traversal -- sorts explicitly
  here. The counters themselves are deliberately left in first-appearance
  order so that each of those ``sorted()`` calls stays an independently
  killable mutation point rather than being masked by a pre-sorted
  container.
* ``boost::accumulate(q_counter | map_values, 0)`` (KLD.h:157, :171) has
  an **``int``** init value, so the running sum is truncated to an
  integer at every step, not just at the end. See
  ``_accumulate_with_int_init``.

What is NOT reproduced: the ``KLDS(const pdf_t&, FloatT)`` constructor
overload (KLD.h:109) has no separate existence here -- it is exactly
``KLDS()`` followed by ``set_p_pdf(...)``, which this class exposes
directly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = ["KLD", "KLDS", "pdf_t"]

# C++: `KLDS<FloatT>::pdf_t`, `std::map<FloatT, FloatT>` (KLD.h:61) -- a
# mapping from a 1-dimensional observation to its count (before
# `precompute_delta_p`) or to its density (after). Python's `dict` is the
# container; the *ordering* the `std::map` supplies is re-established at
# each use site, never by the container.
pdf_t = dict[float, float]


def _ieee_div(numerator: float, denominator: float) -> float:
    """Not a ported symbol -- C/C++ IEEE-754 division, which never raises.

    A local copy of `pymoses/moses/scoring/scoring_base.py`'s `_ieee_div`,
    not an import: `pymoses.feature_selection` must not depend on
    `pymoses.moses.scoring` (the dependency runs the other way -- see this
    module's own PLACEMENT divergence note above, and the same
    each-file-keeps-its-own-copy convention `_alphabet_size` and
    `_type_to_exemplar` already follow in `pymoses/moses/main/`).

    `x/0` is a signed infinity and `0/0` is the x86-64 "QNaN
    floating-point indefinite", whose SIGN BIT IS SET -- so C++ prints it
    as `-nan`. Python raises `ZeroDivisionError` for both.
    """
    if denominator != 0:
        return numerator / denominator
    if numerator > 0:
        return math.inf
    if numerator < 0:
        return -math.inf
    return -math.nan


def _c_log(x: float) -> float:
    """Not a ported symbol -- C/C++ `std::log(double)`, which never raises.

    `log(0)` is the C99 pole error, `-inf` (`log(-0.0)` too); `log(x)` for
    a negative finite `x` is the domain error, a NEGATIVE NaN on glibc;
    `log(nan)` propagates the NaN, sign and all; `log(+inf)` is `+inf`.
    Python's `math.log` raises `ValueError` for the first two and
    propagates for the rest.

    NOT gathered from a table: only the `log(0.0) == -inf` arm is
    reachable from this module (`delta_p / delta_q` underflows to `+0.0`
    when `delta_q` is `+inf`, i.e. when `q_s == 0`). The negative-argument
    arm cannot be reached -- `delta_p` is a density, `delta_q` a positive
    ratio -- and is written out rather than left to raise so that the
    helper's contract is C's, not a subset of it.
    """
    if math.isnan(x):
        return x
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return -math.nan
    return math.log(x)


def _counter(seq: Iterable[float]) -> pdf_t:
    """C++: opencog/util/Counter.h:50 Counter

    ``Counter<FloatT, FloatT>(c)`` -- the sequence constructor
    (Counter.h:80, delegating to ``init`` at :60), which does
    ``this->operator[](*from) += 1`` for every element. Value-initialized
    ``std::map`` entries start at ``0``, so an absent key counts from
    zero, exactly as ``dict.get(value, 0.0)`` does here.

    The returned ``dict`` is in FIRST-APPEARANCE order, NOT sorted --
    ``std::map``'s ascending order is applied by the callers (see this
    module's docstring). Keys are compared with ``std::less<FloatT>``,
    i.e. plain ``<`` on doubles, which is what Python's ``dict`` equality
    and ``sorted()`` both use for floats too.
    """
    counter: pdf_t = {}
    for value in seq:
        counter[value] = counter.get(value, 0.0) + 1.0
    return counter


def _accumulate_with_int_init(values: Iterable[float]) -> float:
    """C++: opencog/util/KLD.h:157 accumulate

    Breadcrumb cites the expression itself, not a declaration: this helper
    exists only to reproduce ``boost::accumulate(q_counter | map_values,
    0)``, which has no C++ definition of its own inside cogutil. The
    expression is at
    KLD.h:157 (and identically at :171 and :69). ``boost::accumulate`` is
    ``std::accumulate``, whose accumulator type is deduced from the INIT
    VALUE, and the init here is the literal ``0``: an ``int``. So the
    running sum is ``acc = acc + *it`` with ``acc`` an ``int`` and ``*it``
    a ``double`` -- the addition happens in double and the result is
    converted back to ``int`` (truncating toward zero) at EVERY step, not
    once at the end.

    Is that observable? **Through ``KLD(p, q)``, no**: that entry point
    always builds ``q_counter`` with ``Counter<FloatT, FloatT>``, whose
    every value is a whole number of occurrences, so each partial sum is
    already integral and truncation is a no-op. **Through ``KLDS``'s own
    public ``operator()``, yes**: a caller may hand it any ``pdf_t``, and
    a non-integral count changes the result. Measured against classic, for
    p = [1, 2, 3, 4] and q_counter = {1.5: 1.5, 2.5: 2.5}: the exact sum
    is 4.0, the accumulated ``q_s`` is 3.0, and the returned KLD is
    -1.0586093894832422 (``tests/fixtures/mi_contin.json``, group
    ``klds_direct``, case ``fractional_counts``). Reproduced faithfully
    for that reason; the reachable-from-MOSES domain is the one where it
    does not matter.

    ``values`` is traversed in the caller's order, which the callers make
    ascending-by-key to match ``std::map``. That matters here and only
    here in this module for a reason worth naming: truncation makes the
    accumulation NON-ASSOCIATIVE, so a different traversal order can give
    a different total (0.6+0.6 truncating to 1 vs 0 twice).

    C++'s ``int`` overflows (undefined behaviour) past ``INT_MAX``;
    Python's does not. Not reproduced with a mask, because there is no
    defined C++ behaviour to reproduce -- and no reachable ``CTable`` has
    2^31 observations.
    """
    acc = 0
    for value in values:
        acc = int(acc + value)
    return float(acc)


class _kld_iters:
    """The three by-reference parameters of ``KLDS::next``.

    # DIVERGENCE(C++ opencog/util/KLD.h:133 KLDS::next): C++ threads the
    # cursor state through three by-reference out-parameters
    # (``FloatT& q_x_pre``, ``pdf_cit& cit_p``, ``pdf_cit& cit_q``), which
    # Python cannot express for a float or an iterator. They are bundled
    # into this small mutable record instead, and the two ``pdf_cit``
    # iterators become integer indices into the caller's sorted item list
    # (``len(items)`` standing in for ``end()``/``cend()``). Purely a
    # calling-convention change: the same three values are read and
    # written in the same order, and ``next``'s arithmetic is untouched.
    """

    __slots__ = ("cit_p", "cit_q", "q_x_pre")

    def __init__(self, q_x_pre: float) -> None:
        self.q_x_pre = q_x_pre
        self.cit_p = 0
        self.cit_q = 0


class KLDS:
    """C++: opencog/util/KLD.h:59 KLDS

    Holds a fixed distribution P (as a probability-density map keyed on
    the observed values) so that ``KL(P||Q)`` can be evaluated for many
    different Q, and so that the per-observation components are reachable
    one at a time (``next``) or all at once (``components``).

    ``FloatT`` is instantiated only as ``contin_t`` (``double``) by the
    one MOSES caller, so this port carries no type parameter: every value
    here is a Python ``float``, i.e. a binary64 ``double``. It is NOT a
    ``score_t`` (binary32) anywhere -- ``mutualInformation`` assigns the
    result to a ``contin_t`` (``table.h:1287``) and returns ``double``.
    """

    def __init__(self, p: Sequence[float] | None = None) -> None:
        """C++: opencog/util/KLD.h:86 KLDS

        The empty constructor (KLD.h:86, ``KLDS() : margin(1.0) {}``) and
        the sorted-sequence one (KLD.h:94, which additionally calls
        ``set_p``) folded into one signature: ``KLDS()`` leaves P unset
        and ``KLDS(p)`` fills it. ``margin`` is initialized to ``1.0`` by
        every C++ constructor and never assigned again, so it is a plain
        attribute here rather than a parameter.

        C++'s ``p_s`` is left UNINITIALIZED by the empty constructor (it
        is not in the member-initializer list); reading it before
        ``set_p``/``set_p_pdf`` is undefined behaviour there. This port
        starts it at ``0.0``, which is a deliberate refusal to reproduce
        undefined behaviour (see docs/CARRY-FORWARD.md Sec 3 on the
        argument-index case) -- no MOSES path reads it early.
        """
        self.margin = 1.0
        self.p_s = 0.0
        self.p_pdf: pdf_t = {}
        # `sorted(self.p_pdf.items())`, cached: `next` is called once per
        # element of p_pdf and C++'s `std::map` is already ordered, so
        # re-sorting inside `next` would turn an O(n log n) walk into
        # O(n^2 log n) for nothing. Rebuilt by `set_p_pdf` (the only
        # writer of `p_pdf`), never stale.
        self._p_sorted: list[tuple[float, float]] = []
        self.x_very_first = 0.0
        self.x_very_last = 0.0
        if p is not None:
            self.set_p(p)

    def set_p_pdf(self, p_counter: pdf_t, p_s_: float = -1.0) -> None:
        """C++: opencog/util/KLD.h:67 KLDS::set_p_pdf

        "set the density function of P given a counter of the number of
        occurrences of each observations of P" (KLD.h:65-66).

        ``p_pdf = p_counter`` is a ``std::map`` copy assignment, so the
        caller's counter is NOT aliased -- ``precompute_delta_p`` below
        overwrites every value in place, and doing that to the caller's
        own dict would be a live bug (docs/CARRY-FORWARD.md Sec 1.1 is the
        same shape). Copied here for that reason, not for tidiness.

        ``p_pdf.cbegin()->first`` and ``p_pdf.crbegin()->first`` are the
        SMALLEST and LARGEST keys, because the container is a
        ``std::map``; an insertion-ordered ``dict`` would give the first-
        and last-inserted instead, so the keys are sorted explicitly.
        """
        self.p_pdf = dict(p_counter)
        if p_s_ < 0:
            self.p_s = _accumulate_with_int_init(value for _, value in sorted(self.p_pdf.items()))
        else:
            self.p_s = p_s_
        keys = sorted(self.p_pdf)
        self.x_very_first = keys[0] - self.margin
        self.x_very_last = keys[-1] + self.margin
        self._precompute_delta_p()
        self._p_sorted = sorted(self.p_pdf.items())

    def set_p(self, p: Sequence[float]) -> None:
        """C++: opencog/util/KLD.h:76 KLDS::set_p

        "like above but takes a sequence of Float instead of a Counter"
        (KLD.h:75). ``set_p_pdf(Counter<FloatT, FloatT>(p), p.size())`` --
        note that ``p_s`` comes from ``p.size()``, so the
        ``boost::accumulate`` path in ``set_p_pdf`` is NOT taken from
        here, and that ``p`` reaches the computation only through
        ``Counter(p)`` and ``len(p)``. Both are order-insensitive, so
        despite the C++ template parameter being named ``SortedSeq``,
        nothing here requires ``p`` to be sorted and no result changes if
        it is not. Verified against classic: fixture group ``kld``, cases
        ``dup_in_p`` and ``dup_in_p_scrambled``, are the same multisets in
        different orders and agree to the last bit.
        """
        self.set_p_pdf(_counter(p), float(len(p)))

    def p_size(self) -> float:
        """C++: opencog/util/KLD.h:114 KLDS::p_size

        "size of p (duplicted values are not ignored)" (KLD.h:113, sic).
        C++ declares the return type ``size_t`` while ``p_s`` is a
        ``FloatT``, so the value is truncated on the way out; kept as a
        float here because every reachable ``p_s`` is a whole number
        (either ``p.size()`` or an ``int``-truncated accumulation) and
        because a Python ``int`` return would silently change comparisons
        against ``p_s`` itself.
        """
        return self.p_s

    def p_pdf_size(self) -> int:
        """C++: opencog/util/KLD.h:124 KLDS::p_pdf_size

        Number of DISTINCT observations in P ("that is duplicate values
        are ignored", KLD.h:118).
        """
        return len(self.p_pdf)

    def next(self, q_items: Sequence[tuple[float, float]], q_s: float, iters: _kld_iters) -> float:
        """C++: opencog/util/KLD.h:133 KLDS::next

        "computes the components log(delta_p / delta_q) once at a time"
        (KLD.h:127) -- one observation of P per call, advancing
        ``iters.cit_p`` by one and ``iters.cit_q`` past every point of Q
        that lies below the current point of P.

        ``q_items`` is the ``q_counter`` argument, already sorted ascending
        by key: C++ hands ``next`` a ``std::map`` that is ordered by
        construction, so sorting is the caller's job here (``__call__``
        and ``components`` both do it once) rather than a per-call cost.
        See ``_kld_iters`` for the by-reference parameters.

        ``OC_ASSERT(cit_p != p_pdf.end())`` (KLD.h:136) is an
        ``OC_ASSERT``, live in release builds, so it is an
        ``AssertionError`` here rather than a bare ``assert``.

        The ``while (q_x < p_x)`` walk (KLD.h:143) cannot run off the end:
        once ``cit_q`` reaches ``cend()``, ``q_x`` becomes
        ``x_very_last``, which is ``max(P) + 1`` and therefore strictly
        greater than any ``p_x``.
        """
        p_pdf_end = len(self._p_sorted)
        if iters.cit_p == p_pdf_end:
            raise AssertionError("KLDS::next: cit_p != p_pdf.end()")
        q_end = len(q_items)

        # compute delta Q
        p_x = self._p_sorted[iters.cit_p][0]
        q_x = self.x_very_last if iters.cit_q == q_end else q_items[iters.cit_q][0]
        # search the points of q right before and after p_x
        while q_x < p_x:
            iters.q_x_pre = q_x
            iters.cit_q += 1
            q_x = self.x_very_last if iters.cit_q == q_end else q_items[iters.cit_q][0]

        delta_p = self._p_sorted[iters.cit_p][1]
        delta_q_x = q_x - iters.q_x_pre
        n_duplicates = 1.0 if iters.cit_q == q_end else q_items[iters.cit_q][1]
        # `(p_s / q_s) * n_duplicates / delta_q_x` and `log(delta_p /
        # delta_q)` are C++ `FloatT` (double) arithmetic, which NEVER
        # raises: `x/0` is a signed infinity, `0/0` is a negative NaN on
        # x86-64, and `std::log(0.0)` is `-inf`. Python's `/` raises
        # `ZeroDivisionError` and `math.log(0.0)` raises `ValueError`, so
        # both go through the IEEE helpers here. NOT a behaviour change
        # for any input that did not previously CRASH: `_ieee_div(a, b)`
        # is `a / b` whenever `b != 0`, and `_c_log(x)` is `math.log(x)`
        # whenever `x > 0`.
        #
        # The reachable caller is `interesting_predicate_bscore.__call__`
        # (pymoses/moses/scoring/bscores.py): a `-H ip` predicate that
        # selects NO rows hands this an EMPTY `q_counter`, so `q_s` is
        # exactly `0.0`, `delta_q` is `+inf`, `delta_p / delta_q` is
        # `0.0` and the component is `-inf`. Measured in classic --
        # bscore `[-inf, ...]` for `and($1 not($1))`,
        # tests/fixtures/ip_bscore.json group `skewed_table`.
        delta_q = _ieee_div(self.p_s, q_s) * n_duplicates / delta_q_x

        iters.cit_p += 1
        return _c_log(_ieee_div(delta_p, delta_q))

    def __call__(self, q_counter: pdf_t) -> float:
        """C++: opencog/util/KLD.h:156 operator()

        ``KLDS::operator()(const pdf_t& q_counter)`` -- "@return estimate
        of KL(P||Q)" (KLD.h:155). Sums ``next`` over every distinct
        observation of P (``dorepeat(p_pdf.size())``) and returns
        ``res / p_s - 1``; the trailing ``- 1`` carries C++'s own comment,
        "I don't fully understand the -1" (KLD.h:164).

        Breadcrumb cites the member at its own line, spelled
        ``operator()`` -- the form docs/CARRY-FORWARD.md Sec 4.5 allows and
        that ``mutual_info.py``'s own ``MICScorerCTable.__call__`` already
        uses.
        """
        q_items = sorted(q_counter.items())
        q_s = _accumulate_with_int_init(value for _, value in q_items)
        iters = _kld_iters(self.x_very_first)
        res = 0.0
        for _ in range(len(self.p_pdf)):
            res += self.next(q_items, q_s, iters)
        return res / self.p_s - 1  # I don't fully understand the -1

    def components(self, q_counter: pdf_t) -> list[float]:
        """C++: opencog/util/KLD.h:170 operator()

        ``void KLDS::operator()(const pdf_t& q_counter, Out out)`` -- "Fill the
        output iterator with the KLD component corresponding to each data
        point" (KLD.h:167-168). Same walk as ``__call__``, but each
        component is divided by ``p_s`` individually and the trailing
        ``- 1`` is absent, so ``sum(components(q)) - 1`` equals
        ``self(q)`` only up to floating-point re-association.

        The C++ overload writes through an output iterator; a returned
        ``list`` is the Python spelling of the same thing. Unused by
        MOSES (``mutualInformation`` calls only the scalar form) but part
        of ``KLDS``'s public surface, hence ported.

        Breadcrumb cites the member at its own line; the real member is
        the TWO-argument ``KLDS::operator()``, which shares the name with
        the scalar one above.
        """
        q_items = sorted(q_counter.items())
        q_s = _accumulate_with_int_init(value for _, value in q_items)
        iters = _kld_iters(self.x_very_first)
        return [self.next(q_items, q_s, iters) / self.p_s for _ in range(len(self.p_pdf))]

    def _precompute_delta_p(self) -> None:
        """C++: opencog/util/KLD.h:181 KLDS::precompute_delta_p

        "replace the occurrence count in p_pdf by delta_p" (KLD.h:180):
        walking P ascending, each entry's count is divided by the gap to
        the previous observation, with the first gap measured from
        ``x_very_first`` (= ``min(P) - margin``).

        Two things make the ORDER load-bearing rather than incidental:
        ``p_x_pre`` is carried across iterations, and the values are
        overwritten in place. Iterating a Python ``dict`` in insertion
        order here would compute every gap against the wrong predecessor
        (and produce negative gaps, hence negative densities, hence a
        ``math.log`` domain error) the moment ``p`` is not already sorted.
        The in-place ``self.p_pdf[p_x] = delta_p`` deliberately keeps the
        dict's own insertion order intact so that this ``sorted()``
        remains independently observable to a mutation test.
        """
        p_x_pre = self.x_very_first
        for p_x, count in sorted(self.p_pdf.items()):
            delta_p_x = p_x - p_x_pre
            self.p_pdf[p_x] = count / delta_p_x
            p_x_pre = p_x


def KLD(p: Sequence[float], q: Sequence[float]) -> float:
    """C++: opencog/util/KLD.h:202 KLD

    "function helper" (KLD.h:201) -- the one-shot form,
    ``KLDS<FloatT> klds(p); return klds(Counter<FloatT, FloatT>(q));``.
    This is what ``mutualInformation``'s continuous branch calls
    (``moses/comboreduct/table/table.h:1287``, as ``- KLD(p, q)``: note
    the leading minus, which is the whole reason that branch returns a
    positive information content at all).

    Neither ``p`` nor ``q`` needs to be sorted despite the C++ template
    parameter's name (``SortedSeq``): ``p`` is consumed only as
    ``Counter(p)`` plus ``len(p)`` and ``q`` only as ``Counter(q)``. Both
    ARE sorted at the one MOSES call site, as a by-product of the
    ``std::multimap`` walk that builds them.

    Returns a ``contin_t`` (``double``); ``SortedSeq::value_type`` is
    ``contin_t`` for the only instantiation MOSES creates.
    """
    klds = KLDS(p)
    return klds(_counter(q))
