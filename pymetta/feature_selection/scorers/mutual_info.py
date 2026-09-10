"""C++: moses/feature-selection/scorers/mutual_info.h

Mutual information between a feature set and the target column of a
``CTable`` (``mutual_information``, C++ ``mutualInformation(const CTable&,
const FeatureSet&)``, table.h:1132) and between two feature sets
(``mutual_information_between_sets``, C++ ``mutualInformationBtwSets``,
table.h:1311) -- the entropy computation itself lives in
``moses/comboreduct/table/table.h``, not ``scorers/mutual_info.h`` (that
header's own ``MutualInformation``/``MICScorerCTable`` are thin wrappers
around it); ported here, in this package, rather than in
``pymetta.comboreduct.table.table`` because this port's own scope keeps
``comboreduct`` off limits (CLAUDE.md task brief) and because both
entropy computations are, in classic itself, feature-selection-only code
(zero callers outside ``moses/feature-selection/`` and
``moses/moses/deme/feature_selector.cc``) despite living in a
comboreduct header.

Scope: ALL THREE of ``mutualInformation(CTable, FeatureSet)``'s branches
are ported.

1. All-discrete inputs, boolean/enum/contin output (table.h:1153-1245).
   A contin output is first binned into five ``builtin``-valued buckets
   by ``discretize_contin_feature``/``get_discrete_bin``, which live in
   ``moses/comboreduct/table/table.cc`` and are ported into this module
   -- see the DIVERGENCE marker on ``discretize_contin_feature`` for why
   they are here and not there.
2. Not-all-discrete inputs with a contin output (table.h:1250-1292): the
   ``- KLD(p, q)`` estimate over a single feature, using
   ``pymetta.feature_selection.scorers.kld`` (``KLDS``/``KLD``, likewise
   ported into this package because their C++ home is cogutil).
3. Anything else: ``OC_ASSERT(0, ...)`` (table.h:1298-1301), raised as an
   ``AssertionError``.

``mutual_information_between_sets`` has NO continuous branch to port --
see its own docstring.

Also ports the ITable/OTable variant of the same function
(``mutual_information_it_ot``, C++ ``mutualInformation(ITable, OTable,
FeatureSet)``, table.h:1074 -- boolean output only, C++'s own scope
restriction, not this port's) and ``scorers/mutual_info.h`` itself:
``MutualInformation`` (bare MI, the ``"simple"`` algorithm's scorer),
``MICScorer`` (the ITable/OTable-based confidence-weighted scorer,
``eval-features``'s own), and ``MICScorerCTable`` (the CTable-based one,
now built on the shared ``fs_scorer_base``,
``pymetta.feature_selection.scorers.fs_scorer_base``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import CTable, ITable, OTable
from pymetta.comboreduct.type_checker.type_signature import type_node
from pymetta.feature_selection.scorers.fs_scorer_base import fs_scorer_base
from pymetta.feature_selection.scorers.kld import KLD
from pymetta.util.logger import logger

# C++: opencog/util/numeric.h:48 PROB_EPSILON
_PROB_EPSILON = 1e-127

# C++: moses/comboreduct/table/table.h:56 TARGET_DISCRETIZED_BINS_NUM
# "discretize contin type target into # bins" -- a `#define`, so it is a
# plain module constant here rather than an enum member.
TARGET_DISCRETIZED_BINS_NUM = 5

# feature-selection.h's own `feature_set`/`feature_set_pop` (also re-exported
# from pymetta.feature_selection.main.feature_selection_params -- this
# module only needs `feature_set`, not the population type).
feature_set = frozenset  # documentation alias; call sites pass any Iterable[int]


def _weighted_information(p: float) -> float:
    """C++: opencog/util/numeric.h:207 weighted_information

    ``p > PROB_EPSILON ? -p * std::log2(p) : 0``.
    """
    return -p * math.log2(p) if p > _PROB_EPSILON else 0.0


def _entropy(probs: Iterable[float]) -> float:
    """C++: opencog/util/numeric.h:229,239 entropy

    ``- Sum_i p_i log_2(p_i)`` over the given probability distribution.

    The accumulation is a NAIVE left-to-right ``res += ...`` loop, matching
    numeric.h:231-233 exactly, and this is NOT a stylistic choice.
    CPython's own ``sum()`` applies Neumaier compensated summation to a
    float sequence, which is strictly more accurate than C++'s bare
    ``double res = 0; res += ...`` -- and therefore, here, WRONG. Measured:
    with ``sum()``, five of the eight ``mutual_information`` values for
    ``tests/fixtures/table_data/mi_contin_out_bins.csv`` disagreed with
    classic in the last two digits (0.20197066796665197 vs
    0.20197066796665109); with this loop, all eight agree bit for bit.
    See also ``_sorted_counts``: the summation ORDER is the other half of
    the same result.
    """
    res = 0.0
    for p in probs:
        res += _weighted_information(p)
    return res


def _sorted_counts(counter: dict) -> list[float]:
    """C++: opencog/util/Counter.h:50 Counter

    The counts of a ``Counter<K, count_t>``, in the order
    ``transform(counter | map_values, ...)`` produces them -- that is, in
    ASCENDING KEY order, because ``Counter`` is a ``std::map``
    (Counter.h:50) and ``boost::adaptors::map_values`` traverses it in its
    own order.

    A Python ``dict`` is insertion-ordered instead, and since ``entropy``
    accumulates naively (see ``_entropy``) the traversal order changes the
    last bits of every entropy computed here. Measured on
    ``mi_contin_out_bins.csv``: insertion order disagrees with classic on
    five of eight feature sets, sorted order on none.

    Sorting is well defined for every key this module can build. Keys are
    either a ``vertex`` (``std::map<vertex, ...>``, ordered by
    ``boost::variant::operator<``: alternative index first, then value) or
    a tuple of them from one ``multi_type_seq`` (``operator<`` at
    table.h:383 delegates to the variant, whose ``builtin_seq`` /
    ``contin_seq`` / ``string_seq`` alternatives each compare
    lexicographically). Within a single counter every key comes from the
    same columns of the same table, so all keys share one alternative and
    Python's own tuple/scalar ordering agrees element for element:
    ``builtin`` is an ``IntEnum`` ordered by ordinal exactly as the C enum
    is, and ``enum_t`` is a ``str``.
    """
    return [count for _, count in sorted(counter.items())]


def _all_discrete_inputs(ctable: CTable) -> bool:
    sig = ctable.get_signature()
    if sig is None:
        return False
    return all(itype in (type_node.boolean_type, type_node.enum_type) for itype in sig.itypes)


# DIVERGENCE(C++ moses/comboreduct/table/table.cc:573
# discretize_contin_feature): PLACEMENT, not behaviour. This function and
# `get_discrete_bin` below are declared in
# `moses/comboreduct/table/table.h:61,65` and defined in `table.cc`, but
# `pymetta/comboreduct/**` is off limits to this slice (see this module's
# own docstring for the same rationale applied to the `table.h` entropy
# machinery, and `kld.py`'s for the cogutil symbols). Their ONLY caller in
# all of classic is `mutualInformation`'s contin-output branch
# (table.h:1188, :1220) -- the function immediately below -- so they are
# ported next to it. `docs/MODULE-COVERAGE.md`'s `table.cc` row still
# lists both as "unported yet REACHABLE ... fail loudly at their Python
# call sites (... mutual_info.py:113)"; that row is now stale in this
# respect and needs updating by whoever owns that document.
def discretize_contin_feature(min_: float, max_: float) -> list[float]:
    """C++: moses/comboreduct/table/table.cc:573 discretize_contin_feature

    The LOWER EDGES of ``TARGET_DISCRETIZED_BINS_NUM`` (== 5) equal-width
    bins spanning ``[min_, max_)``: ``[min_ + i*interval for i in
    range(5)]`` with ``interval = (max_ - min_) / 5``. Note there are five
    edges, not six -- there is no upper edge, which is why
    ``get_discrete_bin`` treats everything at or above ``res[4]`` as the
    last bin.

    Nothing constrains ``min_ <= max_``. Its one caller seeds the scan at
    ``min = 100000.0, max = 0.0`` (table.h:1174-1175) rather than at the
    data's extremes, so on a column whose values never move both seeds the
    interval can come out NEGATIVE and the returned list DESCENDING. That
    is upstream behaviour and is reproduced, not fixed -- see
    ``mutual_information``.

    ``min``/``max`` are spelled ``min_``/``max_`` because both shadow
    Python builtins; nothing else about them changes.
    """
    res: list[float] = []
    interval = (max_ - min_) / TARGET_DISCRETIZED_BINS_NUM
    for i in range(TARGET_DISCRETIZED_BINS_NUM):
        res.append(min_ + i * interval)
    return res


def get_discrete_bin(disc_intvs: list[float], val: float) -> builtin:
    """C++: moses/comboreduct/table/table.cc:583 get_discrete_bin

    Which of the five bins ``val`` falls in, as an ordinal in ``[1, 5]``,
    returned as a ``builtin``. C++ starts the loop counter at ``1``, so a
    ``val`` below every edge yields 1 and a ``val`` at or above
    ``disc_intvs[4]`` falls out of the loop with the counter at 5.

    The comparison is a STRICT ``<`` (``if (val < disc_intvs[i]) break;``),
    so a value landing EXACTLY on a bin edge belongs to the HIGHER bin --
    ``val == disc_intvs[2]`` gives bin 3, not 2. Verified against classic
    over a corpus built to sit on the edges
    (``tests/fixtures/mi_contin.json``, group ``discretize``).

    WHAT THE RETURN VALUE MEANS. C++ casts an ``unsigned`` loop counter to
    ``builtin`` -- table.h:63-64's own comment is explicit that this is a
    trick, not a truth: "builtin is not used to represent builtins but to
    trick vertex to represent discretized contin values". The cast reaches
    the ordinals 1..5 of the ``builtin`` enum, and in THIS port
    (``pymetta.comboreduct.combo.vertex.builtin``, transcribed ordinal for
    ordinal from ``vertex.h:69``) those are, in order:

        bin 1 -> builtin.logical_true   (ordinal 1)
        bin 2 -> builtin.logical_false  (ordinal 2)
        bin 3 -> builtin.logical_and    (ordinal 3)
        bin 4 -> builtin.logical_or     (ordinal 4)
        bin 5 -> builtin.logical_not    (ordinal 5)

    Those names are noise: the value is used only as a ``ycount`` key and
    as the last element of an ``ioc`` key, where all that matters is that
    distinct bins compare unequal. ``builtin.null_vertex`` (ordinal 0) is
    never produced, because the loop counter starts at 1.
    """
    i = 1
    while i < TARGET_DISCRETIZED_BINS_NUM:
        if val < disc_intvs[i]:
            break
        i += 1
    return builtin(i)


def mutual_information(ctable: CTable, fs: Iterable[int]) -> float:
    """C++: moses/comboreduct/table/table.h:1132 mutualInformation(CTable, FeatureSet)

    Two quite different computations behind one name, selected by the
    ctable's input and output types (see this module's docstring for the
    three-way split):

    * ALL-DISCRETE INPUTS (table.h:1153) -- ``MI(Y;X1,...,Xn) =
      H(X1,...,Xn) + H(Y) - H(X1,...,Xn,Y)``, computed by counting
      occurrences of the filtered input vector (``ic``), the
      filtered-input-plus-output vector (``ioc``), and the output alone
      (``ycount``), then converting counts to probabilities and summing
      ``weighted_information`` over each distribution. A CONTIN output is
      first binned by ``discretize_contin_feature``/``get_discrete_bin``,
      over a min/max scan seeded at ``100000.0``/``0.0`` -- see below.
    * OTHERWISE, WITH A CONTIN OUTPUT (table.h:1250) -- ``-KLD(p, q)``
      for a single feature, where ``p`` is that feature's column and ``q``
      the output column, paired up through a ``std::multimap`` keyed on
      ``p``. C++'s own comments call the sign flip and the >1.0 range
      suspect ("XXX review this, is this really correct?",
      table.h:1283-1286); ported as written.
    * ANY OTHER OUTPUT TYPE (table.h:1298) -- ``OC_ASSERT(0, ...)``,
      raised as an ``AssertionError``.

    ``fs`` is iterated in ASCENDING column-index order regardless of the
    order given -- ``seq_filtered`` (opencog/util/algorithm.h:352, "return
    a sequence of all elements corresponding to the indexes, in the order
    of the indexes") is fed a ``std::set<arity_t>`` in C++, which always
    iterates ascending; ``sorted(set(fs))`` here reproduces that exactly.

    THE MIN/MAX SEEDS ARE NOT THE DATA'S EXTREMES. table.h:1174-1175 is
    ``contin_t min = 100000.0; contin_t max = 0.0;`` and the scan only ever
    lowers ``min`` and raises ``max``. So a column lying entirely above
    100000 keeps ``min == 100000.0``, and a column lying entirely below 0
    keeps ``max == 0.0``; both give an interval list that does not span
    the data. DO NOT FIX -- it is upstream behaviour, it is what the
    fixture pins, and `docs/MODULE-COVERAGE.md`'s `table.h` row already
    records it as a known upstream defect.
    """
    otype = ctable.get_output_type()

    #####################
    # discrete inputs   #
    #####################
    if _all_discrete_inputs(ctable) and otype in (
        type_node.enum_type,
        type_node.boolean_type,
        type_node.contin_type,
    ):
        return _mutual_information_discrete_inputs(ctable, fs, otype)

    #####################
    # continuous case   #
    #####################
    if otype is type_node.contin_type:
        return _mutual_information_contin(ctable, fs)

    ###############################
    # Other non implemented cases #
    ###############################
    # C++: table.h:1298-1301, `OC_ASSERT(0, ss.str())` -- an OC_ASSERT is
    # live in release builds and throws `opencog::AssertionException`, so
    # it is a raise here, not a bare `assert`. The message is C++'s own,
    # with `otype` in place of the streamed `type_node`.
    raise AssertionError(f"Type {otype!r} is not supported for mutual information")


def _mutual_information_discrete_inputs(
    ctable: CTable, fs: Iterable[int], otype: type_node
) -> float:
    """C++: moses/comboreduct/table/table.h:1153 mutualInformation(CTable, FeatureSet)

    The "discrete inputs" branch of ``mutual_information`` above, split
    out only so the three branches read as three blocks; C++ has it inline
    in the one function, and the breadcrumb cites the ``if`` that opens it.
    ``otype`` is passed in rather than re-derived so the two callers cannot
    disagree about it.
    """
    fs_sorted = sorted(set(fs))
    ic: dict[tuple, float] = {}
    ioc: dict[tuple, float] = {}
    ycount: dict[object, float] = {}
    total = 0.0

    disc_intvs: list[float] = []
    if otype is type_node.contin_type:
        # C++: table.h:1174-1187. The seeds are NOT the data's extremes;
        # see mutual_information's own docstring. DO NOT FIX.
        min_ = 100000.0
        max_ = 0.0
        for counter in ctable.values():
            for v in counter:
                if v < min_:
                    min_ = v
                if v > max_:
                    max_ = v
        disc_intvs = discretize_contin_feature(min_, max_)

    for row, counter in ctable.items():
        vec = tuple(row[i] for i in fs_sorted)
        row_total = sum(counter.values())
        ic[vec] = ic.get(vec, 0.0) + row_total
        for y_value, count in counter.items():
            # get_builtin(v)/get_enum_type(v) (table.h:1210-1213) both
            # extract the underlying value from a `vertex` -- already
            # exactly what this port's `vertex` values ARE for boolean
            # (a `builtin`) and enum (a `str`) outputs (no wrapper type).
            # For a contin output (table.h:1219-1222) the key is NOT the
            # value but its BIN, a `builtin` in [1, 5].
            key_value = (
                get_discrete_bin(disc_intvs, y_value) if otype is type_node.contin_type else y_value
            )
            ycount[key_value] = ycount.get(key_value, 0.0) + count
            key = (*vec, key_value)
            ioc[key] = ioc.get(key, 0.0) + count
        total += row_total

    if total == 0.0:
        # C++ divides by `total` unconditionally (`double total = it.size()`
        # is never checked against 0 either) -- an empty ctable produces a
        # 0/0 -> NaN probability distribution, and IEEE754 division carries
        # that through to a NaN entropy. DO NOT FIX: matches C++'s own
        # unguarded division; not expected to be reachable from a real
        # CTable (loadTable never produces zero rows for a real file).
        return float("nan")

    yprob = [c / total for c in _sorted_counts(ycount)]
    ip = [c / total for c in _sorted_counts(ic)]
    iop = [c / total for c in _sorted_counts(ioc)]
    return _entropy(ip) + _entropy(yprob) - _entropy(iop)


def _mutual_information_contin(ctable: CTable, fs: Iterable[int]) -> float:
    """C++: moses/comboreduct/table/table.h:1132 mutualInformation(CTable, FeatureSet)

    The "continuous case" branch of ``mutual_information`` above -- a
    contin output over inputs that are not all discrete. Split out for
    readability only. The branch itself opens at the ``else if
    (id::contin_type == otype)`` on table.h:1250 and runs to :1292; the
    breadcrumb cites the enclosing function's DEFINITION line instead,
    because no line within the checker's window of :1250 names the
    function.

    ``p`` is the selected feature's column and ``q`` the output column,
    paired element for element through a ``std::multimap<contin_t,
    contin_t>`` keyed on ``p`` (table.h:1255). The multimap's ordering is
    reproduced exactly (see the sort below), but it turns out NOT to be
    load-bearing: ``KLD`` consumes ``p`` only as ``Counter(p)`` plus
    ``len(p)`` and ``q`` only as ``Counter(q)``, all three of which are
    order-insensitive. Measured, not assumed -- classic itself returns the
    same bits for a scrambled restatement of the same multisets
    (``tests/fixtures/mi_contin.json``, group ``kld``, cases ``dup_in_p``
    vs ``dup_in_p_scrambled``), and the mutation that iterates the pairs
    unsorted kills nothing. Kept faithful anyway: it costs one ``sort``
    and it is what the C++ says.

    ``contin_t ic = - KLD(p,q);`` (table.h:1287) -- the leading minus is
    load-bearing, and C++'s own comment says why: "KLD is negative; we
    want the IC to be postive ... feature selection utterly fails with
    negative IC" (table.h:1283-1285, sic). ``contin_t`` is ``double``, not
    ``score_t``; nothing here narrows to binary32.
    """
    fs_sorted = sorted(set(fs))
    if len(fs_sorted) > 1:
        # C++: table.h:1252-1254, `OC_ASSERT(0, ...)` -- live in release
        # builds, so a raise rather than a bare `assert`.
        raise AssertionError("Contin MI currently supports only 1 feature.")
    if not fs_sorted:
        # NOT a C++ behaviour: `vec.get_at<contin_t>(0)` (table.h:1259) and
        # `*(fs.begin())` (table.h:1289) both read past the end of an empty
        # container, which is undefined behaviour, not a defined result.
        # docs/CARRY-FORWARD.md Sec 3 settles that a port must not
        # reproduce UB (the argument-index case is the precedent), so this
        # fails loudly instead.
        raise IndexError(
            "mutual_information: the contin branch (table.h:1250) needs "
            "exactly one feature; an empty feature set reads past the end "
            "of the filtered row in C++ (undefined behaviour, not "
            "reproduced -- see docs/CARRY-FORWARD.md Sec 3)."
        )

    sorted_list: list[tuple[float, float]] = []
    for row, counter in ctable.items():
        vec = tuple(row[i] for i in fs_sorted)
        x = vec[0]

        # for each contin counted in the row,
        for y, count in counter.items():
            # C++: `unsigned flt_count = val_pair.second;` (table.h:1266).
            # `count_t` is a DOUBLE, so this is a narrowing conversion that
            # TRUNCATES toward zero: a weighted row counted 1.5 times
            # contributes ONE pair, and one counted 0.75 times contributes
            # NONE. DO NOT FIX -- measured against classic on a ctable
            # whose counts were scaled off the integers
            # (tests/fixtures/mi_contin.json, group
            # `mi_contin_in_fractional`), where it changes the answer.
            # Masked to 32 bits because C++'s `unsigned` is 32 bits here;
            # a negative or >=2^32 `count_t` would be undefined behaviour
            # in C++ and is not reachable from any real CTable.
            flt_count = int(count) & 0xFFFFFFFF
            for _ in range(flt_count):
                sorted_list.append((x, y))

    # `std::multimap` orders by KEY only, and C++11 guarantees that among
    # equivalent keys the elements stay in insertion order. A STABLE sort
    # keyed on the first element -- which `list.sort` is -- is exactly
    # that. Sorting the whole tuple instead would additionally order `y`
    # within each `x` group, which the multimap does not do.
    # NO RESULT DEPENDS ON THIS LINE, and that was measured, not assumed:
    # dropping the sort, and sorting by the whole pair, each kill ZERO
    # tests, because `KLD` reads its two arguments only through `Counter`
    # and `len` (see this function's docstring). Both mutants really do
    # reorder the list -- checked -- so that is "no trap", not "no
    # coverage". Kept because it is what the C++ says.
    sorted_list.sort(key=lambda pr: pr[0])

    # XXX TODO, it would be easier if KLD took a sorted list
    # as the argument.  (C++'s own comment, table.h:1275-1276.)
    p: list[float] = []
    q: list[float] = []
    for pr in sorted_list:
        p.append(pr[0])
        q.append(pr[1])

    # KLD is negative; we want the IC to be postive.
    ic = -KLD(p, q)
    # XXX TODO remove this print, for better performance.  (table.h:1288.)
    idx = fs_sorted[0]  # `*(fs.begin())` on a std::set: the smallest index
    logger().debug("Contin MI for feat=%u ic=%g", idx, ic)
    return ic


def _ot_entropy(ot: OTable) -> float:
    """C++: moses/comboreduct/table/table.cc:916 OTEntropy

    Entropy of the output column alone -- ``Counter<vertex, count_t>``
    over every row of ``ot``, converted to probabilities, then
    ``entropy()``. Only the two callers below (``mutual_information_it_ot``)
    ever need this; not a public API of this module.
    """
    counts: dict[object, int] = {}
    for value in ot:
        counts[value] = counts.get(value, 0) + 1
    total = len(ot)
    return _entropy(c / total for c in _sorted_counts(counts))


def mutual_information_it_ot(it: ITable, ot: OTable, fs: Iterable[int]) -> float:
    """C++: moses/comboreduct/table/table.h:1074 mutualInformation(ITable, OTable, FeatureSet)

    ``MI(Y;X1,...,Xn) = H(X1,...,Xn) + H(Y) - H(X1,...,Xn,Y)`` computed
    directly off an (uncompressed) ``ITable``/``OTable`` pair rather than a
    ``CTable`` -- the variant ``eval-features``'s ``MICScorer`` uses
    (``pymetta.feature_selection.main.eval_features``), not the ``CTable``
    variant ``mutual_information`` (above) that every other caller in this
    package uses. ``@note currently, only works for boolean output columns``
    (table.h:1065) -- C++'s own ``OC_ASSERT(id::boolean_type == otype, ...)``
    (table.h:1077-1078), reproduced here as a raise; the enum/contin
    "cut-n-paste from CTable code" the header's own comment (table.h:1066)
    invites was never done in classic itself, so there is nothing further
    to port for those output types.

    ``fs`` is iterated in ASCENDING column-index order regardless of the
    order given, same as ``mutual_information``'s own docstring explains
    for ``seq_filtered``.
    """
    if ot.get_type() is not type_node.boolean_type:
        raise AssertionError("Only boolean types supported")

    fs_sorted = sorted(set(fs))
    ic: dict[tuple, float] = {}
    ioc: dict[tuple, float] = {}
    total = 0.0

    for row, y_value in zip(it, ot, strict=True):
        vec = tuple(row[i] for i in fs_sorted)
        ic[vec] = ic.get(vec, 0.0) + 1.0
        key = (*vec, y_value)
        ioc[key] = ioc.get(key, 0.0) + 1.0
        total += 1.0

    if total == 0.0:
        # DO NOT FIX -- see mutual_information's own note: C++ divides by
        # `total = it.size()` unconditionally too.
        return float("nan")

    ip = [c / total for c in _sorted_counts(ic)]
    iop = [c / total for c in _sorted_counts(ioc)]
    return _entropy(ip) + _ot_entropy(ot) - _entropy(iop)


def mutual_information_between_sets(
    ctable: CTable, fs_l: Iterable[int], fs_r: Iterable[int]
) -> float:
    """C++: moses/comboreduct/table/table.h:1311 mutualInformationBtwSets

    ``MI(fs_l, fs_r) = H(fs_l) + H(fs_r) - H(fs_l union fs_r)``.

    THERE IS NO CONTINUOUS BRANCH TO PORT HERE, and that is a statement
    about C++, not about this port's scope. ``mutualInformationBtwSets``
    has exactly ONE branch (table.h:1332, "discrete case") plus the
    ``OC_ASSERT(0, "Unsupported type for mutual information")`` else
    (table.h:1392-1396). That single branch:

    * gates on the OUTPUT type only -- ``enum or boolean or contin`` --
      and never inspects the input types at all, so there is no
      ``all_discrete_inputs`` test to mirror;
    * never reads the output column: ``uc``/``lc``/``rc`` count only the
      filtered INPUT vectors, so ``discretize_contin_feature`` and
      ``get_discrete_bin`` have nothing to bin and ``KLD`` nothing to
      estimate.

    Contin inputs therefore just work in C++: they become ``std::map``
    keys like any other vertex. This port used to reject them with a
    ``NotImplementedError`` of its own invention; that guard is GONE, on
    the evidence of ``tests/fixtures/mi_contin.json`` group
    ``mi_btw_sets``, which shows classic returning finite values on
    contin-input tables (2.0159220638351671 for ``fs_l={0}, fs_r={1}`` on
    ``mi_contin_in_ties.csv``). The output-type gate stays, and now raises
    ``AssertionError`` to match ``OC_ASSERT``.

    Used only by ``feature_selector.mi()``
    (``pymetta.moses.deme.feature_selector``) for diversity-penalized
    ranking (``params.diversity_pressure > 0.0``, default ``0.0`` -- not
    exercised by the default ``--enable-fs`` path).
    """
    otype = ctable.get_output_type()
    if otype not in (type_node.enum_type, type_node.boolean_type, type_node.contin_type):
        # C++: table.h:1394, `OC_ASSERT(0, "Unsupported type for mutual
        # information")` -- live in release builds, hence a raise.
        raise AssertionError("Unsupported type for mutual information")

    fs_u = sorted(set(fs_l) | set(fs_r))
    fs_l_sorted = sorted(set(fs_l))
    fs_r_sorted = sorted(set(fs_r))

    uc: dict[tuple, float] = {}
    lc: dict[tuple, float] = {}
    rc: dict[tuple, float] = {}
    total = 0.0

    for row, counter in ctable.items():
        vec_u = tuple(row[i] for i in fs_u)
        vec_l = tuple(row[i] for i in fs_l_sorted)
        vec_r = tuple(row[i] for i in fs_r_sorted)
        row_total = sum(counter.values())
        uc[vec_u] = uc.get(vec_u, 0.0) + row_total
        lc[vec_l] = lc.get(vec_l, 0.0) + row_total
        rc[vec_r] = rc.get(vec_r, 0.0) + row_total
        total += row_total

    if total == 0.0:
        return float("nan")  # DO NOT FIX -- see mutual_information's own note.

    up = [c / total for c in _sorted_counts(uc)]
    lp = [c / total for c in _sorted_counts(lc)]
    rp = [c / total for c in _sorted_counts(rc)]
    return _entropy(lp) + _entropy(rp) - _entropy(up)


class MutualInformation:
    """C++: moses/feature-selection/scorers/mutual_info.h:47 MutualInformation

    Callable wrapper: ``MutualInformation(ctable)(feature_set) ->
    mutual_information(ctable, feature_set)``. This is the scorer
    ``simple_select_feature_sets`` actually uses (``algo/simple.py``) --
    plain MI, no confidence weighting.
    """

    def __init__(self, ctable: CTable) -> None:
        self._ctable = ctable

    def __call__(self, features: Iterable[int]) -> float:
        return mutual_information(self._ctable, features)


class MICScorer:
    """C++: moses/feature-selection/scorers/mutual_info.h:82 MICScorer

    Feature-set scorer combining mutual information and confidence, taking
    an ``ITable``/``OTable`` pair directly (rather than a ``CTable`` --
    that is ``MICScorerCTable`` below). ``confidence = N / (N +
    exp(-confi*|fs|))`` with ``N = len(it)`` -- computed inline, NOT via
    ``fs_scorer_base.confidence`` (mutual_info.h:97's own comment shows
    this class predates that shared base and never adopted it; ported
    verbatim, not refactored onto the base class).

    This is ``eval-features``'s scorer (``pymetta.feature_selection.main.
    eval_features``'s ``FSScorer``), not ``select_feature_sets``'s -- no
    algorithm in ``algo/`` uses it.
    """

    def __init__(self, it: ITable, ot: OTable, confi: float = 100.0) -> None:
        self._it = it
        self._ot = ot
        self._confi = confi

    def __call__(self, fs: Iterable[int]) -> float:
        # Materialise ONCE before using `fs` twice. C++'s `FeatureSet` is a
        # `std::set` (mutual_info.h's own template parameter), so classic can
        # iterate it as often as it likes; this signature accepts any
        # `Iterable`, and the previous body passed `fs` to
        # `mutual_information_it_ot` and THEN called `len(list(fs))` on it.
        # For a generator argument the second pass sees an exhausted
        # iterator, so `len` is 0, `confidence` collapses to
        # `N / (N + 1)` regardless of the feature set's real size, and the
        # score is silently wrong rather than failing. Measured: 0.5488 vs
        # 0.4878 for the same features passed as a generator instead of a
        # list. `MICScorerCTable.__call__` below already did this correctly.
        features = list(fs)
        mi = mutual_information_it_ot(self._it, self._ot, features)
        confidence = len(self._it) / (len(self._it) + math.exp(-self._confi * len(features)))
        return mi * confidence


class MICScorerCTable(fs_scorer_base):
    """C++: moses/feature-selection/scorers/mutual_info.h:108 MICScorerCTable

    ``MI(fs) * confidence``, ``confidence = N / (N + exp(-confi*|fs|))``
    (the class's own comment calls this a heuristic of uncertain
    justification -- ported verbatim, not re-derived). ``N`` is the
    ctable's ``uncompressed_size()``, via the shared
    ``fs_scorer_base.confidence`` (``pymetta.feature_selection.scorers.
    fs_scorer_base`` -- ``MICScorerCTable : public fs_scorer_base
    <FeatureSet>`` in C++ too, unlike the plain ``MICScorer`` above).

    NOT the scorer the ``"simple"`` algorithm uses (that is bare
    ``MutualInformation`` above, per ``simple.cc``'s own
    ``typedef MutualInformation<feature_set> FeatureScorer``) -- this
    class is what ``fs_scorer<FeatureSet>`` (``main/feature_selection.py``'s
    ``_fs_scorer``) wires up for ``scorer="mi"`` on the ``"inc"``/``"smd"``/
    ``"hc"`` algorithms.
    """

    def __init__(self, ctable: CTable, confi: float = 100.0) -> None:
        """C++: moses/feature-selection/scorers/mutual_info.h:113 MICScorerCTable"""
        super().__init__(ctable, confi)

    def __call__(self, features: Iterable[int]) -> float:
        """C++: moses/feature-selection/scorers/mutual_info.h:121 operator()"""
        fs = list(features)
        mi = mutual_information(self._ctable, fs)
        confidence = self.confidence(len(fs))
        return mi * confidence
