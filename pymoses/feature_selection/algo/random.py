"""C++: moses/feature-selection/algo/random.{h,cc}

The ``"random"`` feature-selection algorithm: pick ``target_size`` features
uniformly at random, WITH replacement per draw (``rand_element`` re-samples
the full original set every time, not a shrinking remainder) -- see
``random_selection``'s own docstring for the DO-NOT-FIX consequence of
that.

RNG: one ``randGen().randint(n)`` draw per feature requested
(``fs_params.target_size`` draws total, regardless of how many distinct
features end up selected) -- ``opencog::rand_element`` (opencog/util/
random.h:53), ``*std::next(c.begin(), rng.randint(c.size()))``.
"""

from __future__ import annotations

from collections.abc import Iterable

from pymoses.comboreduct.table.table import CTable
from pymoses.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymoses.util.rng import randGen

__all__ = ["random_selection", "random_select_feature_sets"]


def _rand_element(sorted_features: list[int]) -> int:
    """C++: opencog/util/random.h:53 rand_element

    ``*std::next(c.begin(), rng.randint(c.size()))`` -- for an ORDERED
    container (``std::set<arity_t>``, which ``feature_set`` always is in
    classic), ``std::next(c.begin(), idx)`` is the ``idx``-th smallest
    element; ``sorted_features[idx]`` reproduces that for any Python
    ``feature_set`` (a plain unordered ``set`` here). ``rng.randint(n)``
    is ``randGen().randint_n(n)`` in this port -- see rng.py's own
    docstring for why ``randint(int n)`` is named ``randint_n`` here.
    """
    idx = randGen().randint_n(len(sorted_features))
    return sorted_features[idx]


def random_selection(features: Iterable[int], num_desired: int) -> feature_set:
    """C++: moses/feature-selection/algo/random.h:46 random_selection

    "Returns a random subset of features. The features are choosen with a
    uniform probability" (the C++ function's own doc comment). Draws
    ``num_desired`` times from the FULL ``features`` set each time (not a
    shrinking remainder, and not Floyd's without-replacement algorithm --
    this is a plain ``rand_element`` call per iteration, with NO
    duplicate check): ``while (0 < num_desired) { final.insert(rand_
    element(features)); num_desired--; }`` (random.h:50-54).

    DO NOT FIX: because ``final`` is a ``std::set``
    (``insert`` of an already-present element is a silent no-op) while
    the loop counter decrements unconditionally, a duplicate draw
    consumes one of the ``num_desired`` "budget" without growing the
    result -- the returned set can legitimately end up SMALLER than
    ``num_desired`` when ``features`` is small relative to it. This is
    classic's own behaviour, not a bug this port introduces.
    """
    sorted_features = sorted(set(features))
    final: feature_set = feature_set()
    while 0 < num_desired:
        final.add(_rand_element(sorted_features))
        num_desired -= 1
    return final


def random_select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/random.cc:32 random_select_feature_sets

    Unlike ``simple``/``smd``/``incremental``, there is NO
    ``target_size <= 0`` "return all features" special case here --
    ``target_size=0`` simply selects zero features (an empty set), via
    ``random_selection``'s own ``while (0 < num_desired)`` loop never
    running. The returned score is fixed at ``0`` (``feature_set_pop::
    value_type sfs(0, fs)``, random.cc:39) -- never the scorer's actual
    value, matching ``simple_select_feature_sets``'s own fixed-``0``
    convention (this algorithm has no scorer to call at all).
    """
    all_features: feature_set = feature_set(range(ctable.get_arity()))
    fs = random_selection(all_features, fs_params.target_size)
    return [(0.0, fs)]
