"""C++: moses/feature-selection/algo/stochastic_max_dependency.{h,cc}

The ``"smd"`` (stochastic max-dependency) feature-selection algorithm:
greedily grow a small pool of top-scoring feature sets one feature at a
time, keeping only the ``top_size`` best after each round, until reaching
``num_features`` or failing to improve by ``threshold``.

RNG: ONE ``std::random_shuffle`` pass over the candidate feature order,
consuming ``len(features) - 1`` draws total (zero for an empty or
single-element ``features``), before the growth loop starts -- see
``_random_shuffle``'s own docstring for the exact algorithm (verified
against the installed libstdc++, not reconstructed from memory).
"""

from __future__ import annotations

from collections.abc import Iterable

from pymetta.comboreduct.table.table import CTable
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
    fs_scorer,
    initial_features,
)
from pymetta.util.rng import randGen

__all__ = ["stochastic_max_dependency_selection", "smd_select_feature_sets"]


def _random_shuffle(seq: list) -> None:
    """C++: <bits/stl_algo.h> std::random_shuffle(first, last, rand) -- in-place

    ``stochastic_max_dependency.h:125``'s own call site
    (``random_shuffle(shuffle.begin(), shuffle.end(), shr)`` with
    ``shr = [&](ptrdiff_t i) { return randGen().randint(i); }``, `using
    namespace std` in scope) resolves to this exact libstdc++ template
    (verified directly against ``/usr/include/c++/*/bits/stl_algo.h``,
    the deprecated-since-C++17-but-still-present two-argument-generator
    overload), NOT ``boost::random_shuffle`` (a different call site,
    ``deme_expander.cc``, uses that one -- same Fisher-Yates shape, but a
    different function this port has not independently verified matches
    byte for byte):

        for (i = first+1; i != last; ++i) {
            j = first + rand((i - first) + 1);
            if (i != j) iter_swap(i, j);
        }

    One ``rand(k)`` draw per loop iteration (``len(seq) - 1`` total),
    REGARDLESS of whether ``i == j`` (the RNG draw always happens; only
    the swap itself is skipped) -- ``rand(k)`` is
    ``randGen().randint_n(k)`` in this port (``opencog::RandGen::
    randint(int)``, rng.py's own docstring).
    """
    for i in range(1, len(seq)):
        j = randGen().randint_n(i + 1)
        if i != j:
            seq[i], seq[j] = seq[j], seq[i]


def stochastic_max_dependency_selection(
    features: Iterable[int],
    init_features: Iterable[int],
    scorer,
    num_features: int,
    threshold: float = 0.0,
    top_size: int = 100,
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/stochastic_max_dependency.h:82
    stochastic_max_dependency_selection

    Returns a ``feature_set_pop`` (NOT a single ``feature_set``, unlike
    ``simple``/``random``/``incremental``) -- the C++ template itself
    returns ``ranks_t`` (the ``feature_set_pop`` typedef), the full
    ranked-by-score population from the LAST growth round, not just its
    best entry (``smd_select_feature_sets`` below returns this directly,
    unlike e.g. ``incremental_select_feature_sets``'s own single-entry
    wrap).

    Growth loop (stochastic_max_dependency.h:131-182): starting from
    ``tops = [(init_sc, init_features)]``, for each target size ``i``
    from ``len(init_features)+1`` to ``num_features``: for every
    ``(score, fs)`` in ``tops``, try adding each feature in ``features``
    NOT already in ``fs`` (in the SHUFFLED order computed once, up
    front, before the loop -- shared across every round and every
    ``tops`` entry, per ``stochastic_max_dependency.h:123-125``), score
    the result, and insert ``(new_score, new_fs)`` into ``ranks``. After
    trying every combination, keep only the ``top_size`` highest-scoring
    entries of ``ranks`` as the new ``tops``; stop early if the best
    score in ``ranks`` did not improve over the previous round's best by
    at least ``threshold``.

    ``ranks`` reproduces ``std::multimap<double, feature_set,
    std::greater<double>>``: descending by score, ties broken by
    INSERTION order (C++11 multimap preserves insertion order among
    equivalent keys) -- built here as a plain list, appended in the same
    nested-loop order C++'s single-threaded ``OMP_ALGO::for_each``
    would run in (outer loop over ``tops`` in its own list order, inner
    loop over ``shuffle`` in its post-shuffle order), then stable-sorted
    descending by score (Python's ``sort``/``sorted`` are stable, so
    ties keep this same insertion order) -- NOT re-shuffled between
    growth rounds.
    """
    features_set = set(features)
    init_set = set(init_features)

    init_sc = scorer(init_set)
    tops: list[tuple[float, feature_set]] = [(init_sc, init_set)]
    previous_high_score = init_sc

    if len(features_set) < num_features:
        num_features = len(features_set)

    # C++: `std::vector<feature_id> shuffle(features.begin(), features.end())`
    # (stochastic_max_dependency.h:122), where `feature_set` is
    # `std::set<arity_t>` (feature-selection.h:124) -- an ORDERED set, so
    # the vector handed to `random_shuffle` is sorted ascending. A Python
    # `set` iterates in hash order, so `list(features_set)` is not the same
    # sequence in general, and `_random_shuffle` permutes a different
    # starting order even with an identical RNG stream.
    #
    # In practice the two coincide for every input this port can currently
    # produce -- the sole caller passes `feature_set(range(arity))`, and
    # CPython iterates a set of small contiguous ints in ascending order
    # (measured: contiguous 0..n for n up to 1000, and 0..49 minus a few,
    # all in order). They diverge as soon as the set mixes small and
    # >=32 values: `{1, 33, 65, 2}` iterates as `[65, 1, 2, 33]`.
    #
    # So this was right for the wrong reason -- relying on a CPython
    # implementation detail rather than on `std::set`'s contract. Sorted
    # explicitly.
    shuffle = sorted(features_set)
    _random_shuffle(shuffle)

    ranks: list[tuple[float, feature_set]] = []

    for _i in range(len(init_set) + 1, num_features + 1):
        ranks = []
        for _score, fs in tops:
            for fid in shuffle:
                if fid not in fs:
                    prod = set(fs)
                    prod.add(fid)
                    sc = scorer(prod)
                    ranks.append((sc, prod))

        if not ranks:
            raise AssertionError("Fatal Error: no ranked feature sets")

        ranks.sort(key=lambda p: p[0], reverse=True)
        tops = ranks[: min(top_size, len(ranks))]

        high_score = ranks[0][0]
        if high_score - previous_high_score < threshold:
            break
        previous_high_score = high_score

    if not tops:
        raise AssertionError("top is empty, there must be a bug")

    # DIVERGENCE(C++ moses/feature-selection/algo/stochastic_max_dependency.h:184-194):
    # DO NOT FIX. C++ returns `ranks` UNCONDITIONALLY (stochastic_max_
    # dependency.h:193), not `tops` -- if the growth loop body never runs
    # at all (`num_features <= len(init_features)`), `ranks` is still its
    # default-constructed EMPTY multimap at this point (never touched),
    # so the function returns an EMPTY population even though `tops`
    # itself holds the (non-empty) initial feature set. Classic's own
    # debug-logging branch right before this dereferences
    # `ranks.begin()->second` unconditionally when
    # `logger().is_debug_enabled()`, which is undefined behaviour on an
    # empty multimap in that edge case -- not reproduced here (no
    # logging surface, see other modules' convention), but the EMPTY
    # RETURN VALUE is reproduced faithfully: a caller hitting this edge
    # case (target_size <= number of forced initial features) gets `[]`
    # back, the same as classic's release-mode (non-crashing) behaviour.
    return ranks


def smd_select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/stochastic_max_dependency.cc:34
    smd_select_feature_sets
    """
    all_features: feature_set = feature_set(range(ctable.get_arity()))
    fs_sc = fs_scorer(ctable, fs_params)

    if fs_params.target_size <= 0:
        return [(fs_sc(all_features), all_features)]

    init_feat = initial_features(ctable.get_input_labels(), fs_params)
    return stochastic_max_dependency_selection(
        all_features,
        init_feat,
        fs_sc,
        int(fs_params.target_size),
        fs_params.threshold,
        fs_params.smd_top_size,
    )
