"""C++: moses/feature-selection/algo/incremental.{h,cc}

The ``"inc"`` (incremental max-relevancy, min-redundancy) feature-selection
algorithm. Fully deterministic -- no RNG draws anywhere in this file (the
scorer it is given, ``MutualInformation``, is a pure function of the
feature set) -- so, unlike ``random``/``stochastic_max_dependency``, there
is no draw-order fidelity concern here at all.

Algorithm (``incremental_selection``'s own doc comment, incremental.h:46-56):
for each interaction-term size ``i`` from 1 to ``max_interaction_terms``,
score every ``i``-sized subset of the features not yet marked relevant,
mark a subset relevant if it scores above ``threshold``, optionally strip
out "redundant" relevant features (those whose removal doesn't drop a
score by more than ``threshold * red_threshold``), and accumulate the
rest into the result.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable

from pymoses.comboreduct.table.table import CTable
from pymoses.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymoses.feature_selection.scorers.mutual_info import MutualInformation

__all__ = [
    "incremental_selection",
    "cached_incremental_selection",
    "adaptive_incremental_selection",
    "cached_adaptive_incremental_selection",
    "redundant_features",
    "incremental_select_feature_sets",
]

Scorer = Callable[[feature_set], float]


def _powerset_exact(s: Iterable[int], n: int) -> list[feature_set]:
    """C++: opencog/util/algorithm.h:282 powerset(const Set&, size_t, bool)

    Only the ``exact=true`` mode is ported -- every call site in this file
    (``incremental.h:109,129`` and ``redundant_features``'s own
    ``incremental.h:299``) passes ``true``; the ``exact=false``
    ("up to size n") branch has zero call sites anywhere in this port's
    scope and is not ported.

    ``std::set<Set>`` (a set of feature sets, deduplicated and ordered by
    ``Set::operator<`` -- lexicographic on ``std::set<arity_t>``) becomes
    a plain ``list`` here: ``itertools.combinations`` over an ASCENDING-
    sorted input already yields each n-sized subset exactly once, in
    ascending lexicographic order by value -- the same order
    ``std::set<std::set<int>>`` would iterate, which matters for
    ``redundant_features``'s ``max_element`` tie-break (first-occurrence
    in iteration order), see that function's own docstring.
    """
    items = sorted(set(s))
    return [feature_set(c) for c in itertools.combinations(items, n)]


class _MemoizedScorer:
    """C++: opencog/util/lru_cache.h prr_cache_threaded (referenced by
    incremental.h:205,274 cached_incremental_selection/
    cached_adaptive_incremental_selection)

    # DIVERGENCE(C++ opencog/util/lru_cache.h, prr_cache_threaded): a
    plain unbounded ``dict`` keyed by ``frozenset(fs)``, not a "partially
    random replacement" cache bounded to
    ``pow(len(features), max_interaction_terms)`` entries. This changes
    performance (never evicts) but NOT any returned value: the wrapped
    scorer (``MutualInformation``, ``pymoses.feature_selection.scorers.
    mutual_info``) is a pure, side-effect-free function of the feature
    set alone, so caching it -- with any eviction policy, or none -- can
    only ever change how many times it is actually CALLED, never what it
    returns for a given input. `get_misses()`/`get_hits()` (logged only,
    incremental.h has no such counters itself -- those exist on
    ``iscorer_cache``, a different file) are consequently not reproduced
    either.
    """

    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer
        self._cache: dict[frozenset[int], float] = {}

    def __call__(self, fs: Iterable[int]) -> float:
        # `feature_set` (this module's own alias for plain `set[int]`) is
        # unhashable -- `frozenset` is the cache KEY only; the scorer
        # itself is still called with a `feature_set` (the type every
        # scorer in this package's scope actually expects).
        key = frozenset(fs)
        if key not in self._cache:
            self._cache[key] = self._scorer(feature_set(key))
        return self._cache[key]


def redundant_features(features: Iterable[int], scorer: Scorer, threshold: float) -> feature_set:
    """C++: moses/feature-selection/algo/incremental.h:291 redundant_features

    "Return a set of redundant features of a given set of features. It
    looks for a subset of features that do not manage to raise the score
    above a given threshold" (the function's own doc comment). Tries
    subset sizes ``1`` to ``len(features)-1`` ascending ("we start with
    lower i because the fewer the features the better" -- incremental.h:
    296-297); for the first size where SOME subset's complement scores
    within ``threshold`` of the full set, returns that complement.

    ``max_element``'s tie-break (first occurrence under a strict
    ``<`` comparator, for both ``boost::max_element`` and Python's
    ``max(key=...)``) depends on ``_powerset_exact``'s iteration order --
    see that function's own docstring.
    """
    features_set = feature_set(features)
    n = len(features_set)
    for i in range(1, n):
        sfs = _powerset_exact(features_set, i)
        best = max(sfs, key=lambda fs: scorer(fs))
        if scorer(features_set) - scorer(best) < threshold:
            return features_set - best
    return feature_set()


def incremental_selection(
    features: Iterable[int],
    scorer: Scorer,
    threshold: float,
    max_interaction_terms: int = 1,
    red_threshold: float = -1.0,
) -> feature_set:
    """C++: moses/feature-selection/algo/incremental.h:76 incremental_selection

    0) res = empty set
    1.a) Select all relevant features (scoring above threshold) at the
         current interaction-term size, called rel
    1.b) If red_threshold > 0, select redundant features among rel
         (``redundant_features``, at threshold ``threshold *
         red_threshold``), called red; res += rel - red. Otherwise
         res += rel unconditionally.
    2) tf = features - rel (features not yet marked relevant)
    3) Repeat 1-2 at the next interaction-term size (subsets grow by one
       feature each round), up to ``max_interaction_terms``.
    4) return res

    (incremental.h:47-56's own numbered doc comment, condensed.)
    """
    features_set = feature_set(features)
    rel: feature_set = feature_set()
    res: feature_set = feature_set()

    for i in range(1, max_interaction_terms + 1):
        tf = features_set - rel
        fss = _powerset_exact(tf, i)

        rel = feature_set()
        for fs in fss:
            if scorer(fs) > threshold:
                rel |= fs

        if 0.0 < red_threshold:
            nrfss = _powerset_exact(rel, i + 1)
            red: feature_set = feature_set()
            for fs in nrfss:
                red |= redundant_features(fs, scorer, threshold * red_threshold)
            res |= rel - red
        else:
            res |= rel

    return res


def cached_incremental_selection(
    features: Iterable[int],
    scorer: Scorer,
    threshold: float,
    max_interaction_terms: int = 1,
    red_threshold: float = -1.0,
) -> feature_set:
    """C++: moses/feature-selection/algo/incremental.h:196
    cached_incremental_selection
    """
    return incremental_selection(
        features, _MemoizedScorer(scorer), threshold, max_interaction_terms, red_threshold
    )


def adaptive_incremental_selection(
    features: Iterable[int],
    scorer: Scorer,
    features_size_target: int,
    max_interaction_terms: int = 1,
    red_threshold: float = -1.0,
    min_: float = 0.0,
    max_: float = 1.0,
    epsilon: float = 0.001,
) -> feature_set:
    """C++: moses/feature-selection/algo/incremental.h:218
    adaptive_incremental_selection

    "Like incremental_selection but take the number of features to
    select instead of a threshold. It searches it by bisection" (the
    function's own doc comment). ``is_within(min, max, epsilon)`` is
    ``opencog/util/numeric.h:178``'s ``abs(x - y) <= epsilon``.
    """
    mean = (min_ + max_) / 2
    res = incremental_selection(features, scorer, mean, max_interaction_terms, red_threshold)
    rsize = len(res)
    if abs(min_ - max_) <= epsilon or rsize == features_size_target:
        return res
    if rsize < features_size_target:
        nmin, nmax = min_, mean
    else:
        nmin, nmax = mean, max_
    return adaptive_incremental_selection(
        features,
        scorer,
        features_size_target,
        max_interaction_terms,
        red_threshold,
        nmin,
        nmax,
        epsilon,
    )


def cached_adaptive_incremental_selection(
    features: Iterable[int],
    scorer: Scorer,
    features_size_target: int,
    max_interaction_terms: int = 1,
    red_threshold: float = -1.0,
    min_: float = 0.0,
    max_: float = 1.0,
    epsilon: float = 0.01,
) -> feature_set:
    """C++: moses/feature-selection/algo/incremental.h:261
    cached_adaptive_incremental_selection

    Note C++'s own default ``epsilon`` here is ``0.01``, NOT the plain
    ``adaptive_incremental_selection``'s ``0.001`` (incremental.h:224 vs
    :267) -- two genuinely different defaults for the same-named
    parameter on two overloads, ported verbatim as two different Python
    default values, not normalised to one.
    """
    return adaptive_incremental_selection(
        features,
        _MemoizedScorer(scorer),
        features_size_target,
        max_interaction_terms,
        red_threshold,
        min_,
        max_,
        epsilon,
    )


def incremental_select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/incremental.cc:31
    incremental_select_feature_sets

    Uses bare ``MutualInformation`` (feature-selection.h's own
    ``typedef MutualInformation<feature_set> FeatureScorer``,
    incremental.cc:36) -- NOT ``fs_scorer<FeatureSet>`` (the mi/pre
    dispatcher ``stochastic_max_dependency``/``deme_optimize`` use).
    ``scorer=fs_params.scorer`` is consequently never consulted by this
    algorithm at all, exactly as in classic.

    ``threshold <= 0 and target_size <= 0`` returns ALL features
    unranked, scored by MI on the full set (incremental.cc:42-46) --
    unlike ``simple``'s/``random``'s own "return all, fixed score 0"
    special cases, this one DOES compute and return the real MI score of
    ``all_features``, not a fixed ``0``.
    """
    all_features: feature_set = feature_set(range(ctable.get_arity()))
    fsc = MutualInformation(ctable)

    if fs_params.threshold <= 0 and fs_params.target_size <= 0:
        return [(fsc(all_features), all_features)]

    if 0 < fs_params.target_size:
        fs = cached_adaptive_incremental_selection(
            all_features,
            fsc,
            fs_params.target_size,
            fs_params.inc_interaction_terms,
            fs_params.inc_red_intensity,
            0.0,
            1.0,
            fs_params.inc_target_size_epsilon,
        )
    else:
        fs = cached_incremental_selection(
            all_features,
            fsc,
            fs_params.threshold,
            fs_params.inc_interaction_terms,
            fs_params.inc_red_intensity,
        )

    return [(fsc(fs), fs)]
