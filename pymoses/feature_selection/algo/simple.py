"""C++: moses/feature-selection/algo/simple.{h,cc}

The ``"simple"`` feature-selection algorithm (feature_selection_parameters.
algorithm's own default, ``"simple"``) -- the only algorithm ported (see
``main/feature_selection.py``'s dispatcher for the others, all gated).
Score every feature independently by mutual information with the target,
then keep the top ``target_size``.

RNG: ``simple_selection`` draws EXACTLY ONE value from ``randGen()``
whenever ``exp_distrib`` is ``False`` (the default) -- one
``randGen().randint()`` for the tie-break seed below, unconditionally,
even when no two features ever tie. This single draw must still happen
(not skipped as a micro-optimization) to keep this port's RNG stream
advancing in lockstep with classic's for whatever draws a caller makes
afterward. With ``exp_distrib=True`` (never this port's own default), one
additional ``randGen().randdouble()`` is drawn per candidate considered,
in ``sorted_flist`` iteration order, until ``num_desired`` reaches 0 (see
``simple_selection``'s own docstring).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pymoses.comboreduct.table.table import CTable
from pymoses.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymoses.feature_selection.scorers.mutual_info import MutualInformation
from pymoses.util.rng import randGen

__all__ = ["simple_selection", "simple_select_feature_sets"]


def simple_selection(
    features: Iterable[int],
    scorer: Callable[[feature_set], float],
    num_desired: int,
    use_exp_distrib: bool,
    threshold: float,
) -> feature_set:
    """C++: moses/feature-selection/algo/simple.h:96 simple_selection

    1. Score every singleton ``{feat}`` in ``features`` with ``scorer``.
    2. Keep those scoring ``>= threshold``.
    3. Sort them by (score DESC, ``feature ^ seed`` DESC) -- ``seed`` a
       single ``randGen().randint()`` draw, one per call (simple.h:38-41's
       own ``ScoredFeatureSetGreater()`` constructor, invoked once when
       C++'s ``std::set<ScoredFeatureSet, ScoredFeatureSetGreater<...>>
       sorted_flist`` is declared -- BEFORE any feature is scored or
       inserted). This reproduces a `std::set`'s iteration order under
       that comparator exactly: distinct features never compare equal
       under it (`ix != iy => (ix^seed) != (iy^seed)`, XOR by a fixed
       value is injective), so there is no std::set-would-drop-a-
       duplicate case to worry about -- a plain stable sort in the
       comparator's descending order reproduces the set's begin-to-end
       iteration order term for term.
    4. Take the first ``num_desired`` (stair-step, the default) or a
       random subset biased toward the front (``use_exp_distrib=True``,
       simple.h:126-138 -- ``xn *= x`` after EVERY candidate, drawing
       ``randGen().randdouble()`` for each one considered, stopping the
       moment ``num_desired`` reaches 0; simple.h's own "cheat slightly,
       and ask for more" comment about ``x`` is reproduced verbatim, not
       independently re-derived).
    """
    seed = randGen().randint()

    scored: list[tuple[float, int]] = []
    for feat in features:
        sc = scorer(feature_set({feat}))
        if threshold <= sc:
            scored.append((sc, feat))

    sorted_flist = sorted(scored, key=lambda p: (p[0], p[1] ^ seed), reverse=True)

    final: feature_set = feature_set()
    if use_exp_distrib:
        # Exponential distribution with mean of num_desired features.
        x = 1.0 - 1.0 / (float(num_desired) + 1.0)
        xn = 1.0
        for _score, feat in sorted_flist:
            if randGen().randdouble() < xn:
                final.add(feat)
                num_desired -= 1
                if num_desired <= 0:
                    break
            xn *= x
    else:
        # stair-step distribution: keep the top num_desired only.
        for _score, feat in sorted_flist:
            final.add(feat)
            num_desired -= 1
            if num_desired <= 0:
                break

    return final


def simple_select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/simple.cc:31 simple_select_feature_sets

    ``target_size == 0`` returns ALL features, unranked, with score fixed
    at ``0`` -- classic's own comment (simple.cc:41-45) explains the
    fixed-``0`` score: scoring ``all_features`` is skipped on purpose
    because the contin-MI code path (not reachable from this port; see
    ``mutual_info.py``'s own scope note) does not support multi-feature
    sets, so the real score is never computed here either way.

    Otherwise, ``target_size`` is the desired count and the returned
    population always has EXACTLY ONE entry (``{sfs}`` in C++,
    unconditionally) -- the ``"simple"`` algorithm never produces more
    than one feature set, so ``feature_selector_parameters.n_demes > 1``
    has no additional effect when ``algorithm="simple"`` (this port's only
    ported algorithm; see ``pymoses.moses.deme.feature_selector``'s own
    module docstring for where this is exercised).

    The returned score is fixed at ``0``, not the scorer's actual value on
    the final feature set -- simple.cc:49-56's own ``/* fsc(fs) */ 0``
    comment, same "MI over multi-feature contin sets isn't supported so
    it's skipped even here" reasoning. DO NOT FIX: this is not a bug this
    port introduces, it is classic's own explicit choice.
    """
    fsc = MutualInformation(ctable)
    all_features: feature_set = feature_set(range(ctable.get_arity()))

    if fs_params.target_size == 0:
        return [(0.0, all_features)]

    fs = simple_selection(
        all_features, fsc, fs_params.target_size, fs_params.exp_distrib, fs_params.threshold
    )
    return [(0.0, fs)]
