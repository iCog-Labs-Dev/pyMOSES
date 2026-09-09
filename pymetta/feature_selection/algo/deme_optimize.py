"""C++: moses/feature-selection/algo/deme_optimize.{h,cc}

The ``"hc"`` feature-selection algorithm: run one of MOSES's own
optimization algorithms (in this port's scope, only ``hill_climbing``,
``pymetta.moses.moses.optimization.hill_climbing`` -- read-only import,
out of THIS package's own edit scope) directly over a synthetic deme where
each bit represents "is this feature selected" (``field_set(disc_spec(2),
arity)``), using ``deme_based_scorer``
(``pymetta.feature_selection.scorers.moses_optim``) to turn a feature-set
scorer into an ``iscorer_base`` the optimizer can search.

No new RNG-draw-order claim is made here: every random draw this
algorithm makes flows through ``hill_climbing``/``neighborhood_sampling``
(both already independently verified bit-identical against classic,
CARRY-FORWARD.md/S9), not through any code added in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymetta.comboreduct.table.table import CTable
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set_pop,
    fs_scorer,
    initial_features,
)
from pymetta.feature_selection.scorers.moses_optim import deme_based_scorer, get_feature_set
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_scorer import iscorer_base
from pymetta.moses.representation.instance_set import instance_set

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymetta.moses.moses.optimization.optimization import optimizer_base
    from pymetta.moses.moses.types import composite_score

__all__ = [
    "initial_instance",
    "iscorer_cache",
    "optimize_deme_select_feature_sets",
    "create_deme_select_feature_sets",
    "moses_select_feature_sets",
]


def initial_instance(
    fs_params: feature_selection_parameters,
    fields: field_set,
    ilabels: Sequence[str],
) -> instance:
    """C++: moses/feature-selection/algo/deme_optimize.cc:31 initial_instance

    "For the MOSES algo, generate the initial instance" (deme_optimize.h:
    41's own doc comment): a zero instance with one raw bit set per
    initial (forced-start) feature. ``*(fields.begin_bit(res) + idx) =
    true`` (a bit_iterator assignment at feature-index ``idx`` within the
    bit region) is ``fields.set_raw(res, fields.begin_bit_raw_idx() +
    idx, 1)`` here -- see ``moses_optim.get_feature_set``'s own docstring
    for why the raw-index offset and the feature index coincide only
    because ``field_set(disc_spec(2), arity)`` (this function's only
    caller, ``create_deme_select_feature_sets``) has no contin/multi-bit
    fields preceding the bit region.
    """
    init_features = initial_features(ilabels, fs_params)
    res = instance(fields.packed_width())
    start = fields.begin_bit_raw_idx()
    for idx in init_features:
        fields.set_raw(res, start + idx, 1)
    return res


class iscorer_cache(iscorer_base):
    """C++: moses/feature-selection/main/feature-selection.h:143 iscorer_cache

    "A wrapper, simply so that optimizer gets the iscorer_base base
    class. The only reason for this wrapper is that both iscorer_base,
    and prr_cache_threaded both define operator(), and I need it to be
    clear which operator() needs to be called" (the C++ class's own doc
    comment).

    # DIVERGENCE(C++ opencog/util/lru_cache.h prr_cache_threaded, wrapped
    here): a plain unbounded ``dict`` keyed by the instance's own packed
    words (a ``tuple(inst)``, hashable and value-equal exactly when the
    two instances encode the same feature set), not a capacity-bounded
    "partially random replacement" cache. As in ``incremental.py``'s
    ``_MemoizedScorer`` (same reasoning): the wrapped scorer
    (``deme_based_scorer``) is a pure function of the instance's bit
    pattern, so this changes call COUNT, never any returned value.
    ``get_misses``/``get_hits`` are kept (feature-selection.h:153-154
    logs them, no logging surface here, but the counts themselves are
    real and test-observable).
    """

    def __init__(self, n: int, sc: Callable[[instance], composite_score]) -> None:
        """C++: moses/feature-selection/main/feature-selection.h:146 iscorer_cache

        ``n`` (the C++ cache's capacity) is accepted for signature
        fidelity but otherwise unused -- see the class docstring.
        """
        del n
        self._sc = sc
        self._cache: dict[tuple[int, ...], composite_score] = {}
        self._hits = 0
        self._misses = 0

    def __call__(self, x: instance) -> composite_score:
        """C++: moses/feature-selection/main/feature-selection.h:149 operator()"""
        key = tuple(x)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        result = self._sc(x)
        self._cache[key] = result
        return result

    def get_misses(self) -> int:
        """C++: moses/feature-selection/main/feature-selection.h:153 get_misses"""
        return self._misses

    def get_hits(self) -> int:
        """C++: moses/feature-selection/main/feature-selection.h:154 get_hits"""
        return self._hits


def optimize_deme_select_feature_sets(
    fields: field_set,
    deme: instance_set,
    init_inst: instance,
    optimize: optimizer_base,
    scorer: Callable[[instance], composite_score],
    fs_params: feature_selection_parameters,
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/deme_optimize.h:46
    optimize_deme_select_feature_sets

    Runs ``optimize`` over ``deme`` starting from ``init_inst``, then
    converts the resulting deme into a ``feature_set_pop``, skipping
    exact ``(score, feature_set)`` duplicates (``std::find`` against the
    population built so far -- a plain list-membership check here, since
    equality on ``(float, set[int])`` needs no hashing).

    A ``std::multimap<double, feature_set, std::greater<double>>``
    keeps insertion order among equal keys; built here by APPENDING in
    deme-iteration order (matching each ``multimap::insert``'s own
    "goes after the existing equal-key run" placement) and stable-
    sorting descending by score ONCE at the end (Python's ``sort`` is
    stable, so ties keep this same relative order).
    """
    optimize(deme, scorer, fs_params.hc_max_evals, fs_params.max_time, init_inst=init_inst)

    fs_pop: feature_set_pop = []
    for scored_inst in deme:
        score = scored_inst.second.get_score()
        fs = get_feature_set(fields, scored_inst.first)
        pair = (score, fs)
        if pair not in fs_pop:
            fs_pop.append(pair)
    fs_pop.sort(key=lambda p: p[0], reverse=True)
    return fs_pop


def create_deme_select_feature_sets(
    ctable: CTable,
    optimize: optimizer_base,
    scorer: Callable[[set[int]], float],
    fs_params: feature_selection_parameters,
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/deme_optimize.h:88
    create_deme_select_feature_sets

    "Run feature selection given a moses optimizer and a scorer, create a
    deme and wrap the scorer for that deme. Possibly add cache as well"
    (the function's own doc comment). ``field_set(disc_spec(2), arity)``:
    one 1-bit boolean field per input column -- "is this feature
    selected".
    """
    arity = ctable.get_arity()
    fields = field_set(field_set.disc_spec(2), arity)
    deme = instance_set(fields)
    init_inst = initial_instance(fs_params, fields, ctable.get_input_labels())
    db_sc = deme_based_scorer(scorer, fields)

    if fs_params.hc_cache_size > 0:
        sc_cache = iscorer_cache(fs_params.hc_cache_size, db_sc)
        return optimize_deme_select_feature_sets(
            fields, deme, init_inst, optimize, sc_cache, fs_params
        )
    return optimize_deme_select_feature_sets(fields, deme, init_inst, optimize, db_sc, fs_params)


def moses_select_feature_sets(
    ctable: CTable, optimize: optimizer_base, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/algo/deme_optimize.h:120
    moses_select_feature_sets
    """
    fs_sc = fs_scorer(ctable, fs_params)
    return create_deme_select_feature_sets(ctable, optimize, fs_sc, fs_params)
