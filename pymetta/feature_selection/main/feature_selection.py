"""C++: moses/feature-selection/main/feature-selection.{h,cc}

The algorithm dispatcher (``select_feature_sets``/``select_features``) plus
the small library helper ``score_individual_features``, shared by the
standalone ``feature-selection`` binary and the in-loop ``feature_selector``
(``pymetta.moses.deme.feature_selector``, which calls ``select_feature_sets``
directly -- feature_selector.cc:275). ``initial_features``/``fs_scorer``
(also feature-selection.{h,cc} symbols) live in ``main/
feature_selection_params.py`` instead -- see that module's own docstrings
for why (breaking a would-be circular import between this module and
``algo/``).

Algorithm dispatch (``select_feature_sets``): ALL FIVE algorithms are
ported -- ``"simple"`` (``algo/simple.py``), ``"inc"``
(``algo/incremental.py``), ``"smd"`` (``algo/stochastic_max_dependency.py``),
``"random"`` (``algo/random.py``), and ``"hc"`` (``algo/deme_optimize.py``,
which runs MOSES's own ``hill_climbing`` optimizer as the search --
``pymetta.moses.moses.optimization.hill_climbing``, read-only import).
classic's own "unknown algorithm" branch (feature-selection.cc:170-174,
``cerr`` + ``exit(1)``) is not a real return path here -- an unrecognized
``algorithm`` string raises ``NotImplementedError`` instead.

NOT ported: ``fs-main.cc``'s ``main()`` (argument parsing via
``boost::program_options``, log-level setup, ``randGen().seed(rand_seed)``)
-- no CLI in this port's scope (CLI wiring is ``problem_params.py``'s
concern, another package). Also NOT ported: feature-selection.cc's own
``feature_selection()`` top-level orchestration, ``write_results``,
``log_selected_features``, and ``err_empty_features`` -- all four need
either ``ostreamTable``/``saveTable`` (``moses/comboreduct/table/table_io.h``,
CONFIRMED absent from this port's own ``pymetta.comboreduct.table.
table_io`` -- a search for ``def ostreamTable`` or ``def saveTable`` over
that package returns nothing; its own docstring says explicitly "out of scope")
or a logging surface this project does not provide (matching the
convention every other logger()-only routine in this project follows, see
e.g. ``deme_expander.py``'s module docstring). A caller that wants the
standalone binary's behaviour reproduced calls ``select_features``
directly on a loaded ``Table``/``CTable`` and, if it wants the selected
table's per-feature scores, ``score_individual_features`` -- exactly what
this package's own tests do.
"""

from __future__ import annotations

from pymetta.comboreduct.table.table import CTable, Table
from pymetta.feature_selection.algo.deme_optimize import moses_select_feature_sets
from pymetta.feature_selection.algo.incremental import incremental_select_feature_sets
from pymetta.feature_selection.algo.random import random_select_feature_sets
from pymetta.feature_selection.algo.simple import simple_select_feature_sets
from pymetta.feature_selection.algo.stochastic_max_dependency import smd_select_feature_sets
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
    fs_scorer,
    initial_features,
)
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters

__all__ = [
    "initial_features",
    "score_individual_features",
    "select_feature_sets",
    "select_features",
]


def score_individual_features(
    table: Table, fs_params: feature_selection_parameters
) -> list[float]:
    """C++: moses/feature-selection/main/feature-selection.cc:69 score_individual_features

    Scores every single column of ``table`` (compressed first) with
    ``fs_params.scorer`` -- used by ``log_selected_features`` in C++
    (not ported, see module docstring) and directly, here, by this
    port's own tests as an oracle-comparable per-feature score (matching
    the standalone binary's own ``log_selected_features(): <label> <score>``
    log lines at ``-l INFO``).
    """
    ctable = table.compressed()
    scorer = fs_scorer(ctable, fs_params)
    return [scorer(feature_set({idx})) for idx in range(table.get_arity())]


def select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/main/feature-selection.cc:137 select_feature_sets

    The ``"hc"`` branch builds ``optim_parameters``/``hc_parameters``
    exactly as feature-selection.cc:141-159 does: a fresh
    ``hc_parameters()`` (its own C++ default constructor -- ``widen_
    search=False``, ``single_step=False``, ``crossover=True``,
    ``fraction_of_nn=2.0``, matching this port's own ``hc_parameters()``
    defaults) with ``widen_search``/``fraction_of_nn``/``single_step``/
    ``crossover``/``crossover_pop_size``/``crossover_min_neighbors``/
    ``prefix_stat_deme`` then explicitly overwritten from ``fs_params``
    (or a literal, for ``single_step=False`` and
    ``prefix_stat_deme="FSDemes"``) -- reproduced here as direct
    constructor keyword arguments for ``optim_parameters`` (whose own
    ``__init__`` already ends by calling ``set_min_score_improv``, so
    passing ``min_score_improv=0.0`` to the constructor has the exact
    same effect as C++'s separate ``op_params.set_min_score_improv
    (0.0)`` call after a default-constructed ``optim_parameters()`` --
    not a behavioural difference, just a different way to reach the
    same field values) and direct attribute assignment for
    ``hc_parameters`` (which C++ also does, since only four of its
    fields are even constructor parameters -- see ``hc_parameters.
    __init__``'s own docstring).
    """
    if fs_params.algorithm == "hc":
        op_params = optim_parameters(
            opt_algo="hc",
            pop_size_ratio=20.0,
            terminate_if_gte=fs_params.hc_max_score,
            max_dist=4,
            min_score_improv=0.0,
        )
        hc_params = hc_parameters()
        hc_params.widen_search = fs_params.hc_widen_search
        hc_params.fraction_of_nn = fs_params.hc_fraction_of_nn
        hc_params.single_step = False
        hc_params.crossover = fs_params.hc_crossover
        hc_params.crossover_pop_size = fs_params.hc_crossover_pop_size
        hc_params.crossover_min_neighbors = fs_params.hc_crossover_min_neighbors
        hc_params.prefix_stat_deme = "FSDemes"
        hc = hill_climbing(op_params, hc_params)
        return moses_select_feature_sets(ctable, hc, fs_params)
    if fs_params.algorithm == "inc":
        return incremental_select_feature_sets(ctable, fs_params)
    if fs_params.algorithm == "smd":
        return smd_select_feature_sets(ctable, fs_params)
    if fs_params.algorithm == "random":
        return random_select_feature_sets(ctable, fs_params)
    if fs_params.algorithm == "simple":
        return simple_select_feature_sets(ctable, fs_params)
    raise NotImplementedError(
        f"select_feature_sets: algorithm={fs_params.algorithm!r} is unknown "
        "(feature-selection.cc:170's own cerr+exit(1) path)."
    )


def select_features(
    table_or_ctable: Table | CTable, fs_params: feature_selection_parameters
) -> feature_set:
    """C++: moses/feature-selection/main/feature-selection.cc:159
    select_features(const CTable&, ...) and
    moses/feature-selection/main/feature-selection.cc:171
    select_features(const Table&, ...)

    Accepts either a ``CTable`` (matching the first C++ overload directly)
    or a ``Table`` (matching the second: compress, subsample if
    ``fs_params.subsampling_ratio < 1.0``, then delegate) -- Python has no
    overload resolution, so both C++ signatures collapse onto this one
    function, dispatching on the argument's own type.

    ``subsampling_ratio < 1.0`` needs ``subsampleCTable``
    (moses/comboreduct/table/table.cc:899, row-random-removal) -- not
    ported (out of scope: touching ``pymetta.comboreduct`` is off limits
    for this task). Raises rather than silently skipping the subsample
    step; the default ``subsampling_ratio=1.0`` path (no subsampling) is
    unaffected and fully ported.
    """
    if isinstance(table_or_ctable, Table):
        ctable = table_or_ctable.compressed()
        if fs_params.subsampling_ratio < 1.0:
            raise NotImplementedError(
                "select_features: subsampling_ratio < 1.0 needs "
                "subsampleCTable (moses/comboreduct/table/table.cc:899), "
                "not ported -- pymetta.comboreduct is out of this task's "
                "scope. Leave subsampling_ratio at its default (1.0)."
            )
    else:
        ctable = table_or_ctable

    fs_pop = select_feature_sets(ctable, fs_params)
    if not fs_pop:
        raise AssertionError("select_features: There might a bug")  # OC_ASSERT(!fs_pop.empty())
    return fs_pop[0][1]
