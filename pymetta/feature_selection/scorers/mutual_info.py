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

Scope: only the DISCRETE-inputs branch (table.h:1148-1240) is ported --
every input column boolean or enum, output boolean/enum/contin. This is
the only branch this port's own Boolean-only tables (``CTable``s built
by ``pymetta.comboreduct.table.table_io.loadTable(...).compressed()``
from ``trainingBoolean.dat``/``trainingBooleanXor.dat``-shaped data) ever
reach. The continuous-inputs-not-all-discrete branch and the single-contin-
feature KLD branch (table.h:1242-1278, needs ``discretize_contin_feature``/
``KLD``, neither ported) both raise ``NotImplementedError`` -- see
``mutual_information``'s own docstring.

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

from pymetta.comboreduct.table.table import CTable, ITable, OTable
from pymetta.comboreduct.type_checker.type_signature import type_node
from pymetta.feature_selection.scorers.fs_scorer_base import fs_scorer_base

# C++: opencog/util/numeric.h:48 PROB_EPSILON
_PROB_EPSILON = 1e-127

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
    """
    return sum(_weighted_information(p) for p in probs)


def _all_discrete_inputs(ctable: CTable) -> bool:
    sig = ctable.get_signature()
    if sig is None:
        return False
    return all(itype in (type_node.boolean_type, type_node.enum_type) for itype in sig.itypes)


def mutual_information(ctable: CTable, fs: Iterable[int]) -> float:
    """C++: moses/comboreduct/table/table.h:1132 mutualInformation(CTable, FeatureSet)

    ``MI(Y;X1,...,Xn) = H(X1,...,Xn) + H(Y) - H(X1,...,Xn,Y)``, computed by
    counting occurrences of the filtered input vector (``ic``), the
    filtered-input-plus-output vector (``ioc``), and the output alone
    (``ycount``), then converting counts to probabilities and summing
    ``weighted_information`` over each distribution (table.h:1148-1240's
    own "discrete inputs" branch -- this port's ONLY ported branch, see
    module docstring).

    ``fs`` is iterated in ASCENDING column-index order regardless of the
    order given -- ``seq_filtered`` (opencog/util/algorithm.h:352, "return
    a sequence of all elements corresponding to the indexes, in the order
    of the indexes") is fed a ``std::set<arity_t>`` in C++, which always
    iterates ascending; ``sorted(set(fs))`` here reproduces that exactly.
    """
    if not _all_discrete_inputs(ctable):
        raise NotImplementedError(
            "mutual_information: not all input columns are boolean/enum -- "
            "the not-all-discrete-inputs branch (table.h:1242 onward) is "
            "not ported (this port's tables are boolean-only)."
        )
    otype = ctable.get_output_type()
    if otype not in (type_node.enum_type, type_node.boolean_type, type_node.contin_type):
        raise NotImplementedError(
            f"mutual_information: unsupported output type {otype!r} "
            "(table.h:1370's own 'else' branch: OC_ASSERT(0, ...))."
        )
    if otype is type_node.contin_type:
        raise NotImplementedError(
            "mutual_information: a contin OUTPUT with all-discrete inputs "
            "selects table.h:1148's own contin `disc_intvs` "
            "discretization (discretize_contin_feature) -- not ported. "
            "This port's tables are boolean-output only."
        )

    fs_sorted = sorted(set(fs))
    ic: dict[tuple, float] = {}
    ioc: dict[tuple, float] = {}
    ycount: dict[object, float] = {}
    total = 0.0

    for row, counter in ctable.items():
        vec = tuple(row[i] for i in fs_sorted)
        row_total = sum(counter.values())
        ic[vec] = ic.get(vec, 0.0) + row_total
        for y_value, count in counter.items():
            # get_builtin(v)/get_enum_type(v) (table.h:1210-1213) both
            # extract the underlying value from a `vertex` -- already
            # exactly what this port's `vertex` values ARE for boolean
            # (a `builtin`) and enum (a `str`) outputs (no wrapper type).
            ycount[y_value] = ycount.get(y_value, 0.0) + count
            key = (*vec, y_value)
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

    yprob = [c / total for c in ycount.values()]
    ip = [c / total for c in ic.values()]
    iop = [c / total for c in ioc.values()]
    return _entropy(ip) + _entropy(yprob) - _entropy(iop)


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
    return _entropy(c / total for c in counts.values())


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

    ip = [c / total for c in ic.values()]
    iop = [c / total for c in ioc.values()]
    return _entropy(ip) + _ot_entropy(ot) - _entropy(iop)


def mutual_information_between_sets(
    ctable: CTable, fs_l: Iterable[int], fs_r: Iterable[int]
) -> float:
    """C++: moses/comboreduct/table/table.h:1311 mutualInformationBtwSets

    ``MI(fs_l, fs_r) = H(fs_l) + H(fs_r) - H(fs_l union fs_r)``. Same
    discrete-only scope as ``mutual_information`` (table.h:1330's own
    "discrete case" -- boolean/enum/contin output, all-discrete inputs);
    the 'else' branch (``OC_ASSERT(0, "Unsupported type...")``,
    table.h:1394-1396) raises here too.

    Used only by ``feature_selector.mi()``
    (``pymetta.moses.deme.feature_selector``) for diversity-penalized
    ranking (``params.diversity_pressure > 0.0``, default ``0.0`` -- not
    exercised by the default ``--enable-fs`` path).
    """
    if not _all_discrete_inputs(ctable):
        raise NotImplementedError(
            "mutual_information_between_sets: not all input columns are "
            "boolean/enum -- not ported (this port's tables are "
            "boolean-only)."
        )
    otype = ctable.get_output_type()
    if otype not in (type_node.enum_type, type_node.boolean_type, type_node.contin_type):
        raise NotImplementedError(
            f"mutual_information_between_sets: unsupported output type {otype!r} "
            "(table.h:1394's own 'else' branch)."
        )

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

    up = [c / total for c in uc.values()]
    lp = [c / total for c in lc.values()]
    rp = [c / total for c in rc.values()]
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
