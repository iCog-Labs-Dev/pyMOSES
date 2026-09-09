"""C++: moses/feature-selection/main/feature-selection.{h,cc}

The algorithm dispatcher (``select_feature_sets``/``select_features``) plus
two small library helpers (``initial_features``, ``score_individual_
features``) shared by the standalone ``feature-selection`` binary and the
in-loop ``feature_selector`` (``pymetta.moses.deme.feature_selector``,
which calls ``select_feature_sets`` directly -- feature_selector.cc:275).

NOT ported: ``fs-main.cc``'s ``main()`` (argument parsing via
``boost::program_options``, log-level setup, ``randGen().seed(rand_seed)``)
and feature-selection.cc's own top-level orchestration/output helpers
(``feature_selection()``, ``write_results``, ``log_selected_features``,
``err_empty_features``) -- these are CLI/file-I/O glue with no caller once
there is no CLI in this port's scope (no logging surface either, matching
the convention ``deme_expander.py``'s module docstring already states for
every other logger()-only routine in this project). A caller that wants
the standalone binary's behaviour reproduced calls ``select_features``
directly on a loaded ``Table``/``CTable``, as this package's own tests do.

Algorithm dispatch (``select_feature_sets``): only ``"simple"``
(``algo/simple.py``) is ported. ``"inc"`` (incremental.cc, 310 lines),
``"smd"`` (stochastic_max_dependency.cc, 198 lines), ``"random"``
(random.cc), and ``"hc"`` (deme_optimize.cc -- runs MOSES itself as the
search) all raise ``NotImplementedError`` naming the algorithm and the
C++ file that would need porting. classic's own "unknown algorithm"
branch (feature-selection.cc:170-174, ``cerr`` + ``exit(1)``) is likewise
not a real return path here -- an unrecognized ``algorithm`` string
raises the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymetta.comboreduct.table.table import CTable, Table
from pymetta.feature_selection.algo.simple import simple_select_feature_sets
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
)
from pymetta.feature_selection.scorers.mutual_info import MICScorerCTable

__all__ = [
    "initial_features",
    "score_individual_features",
    "select_feature_sets",
    "select_features",
]

_PORTED_ALGORITHMS = {
    "inc": "moses/feature-selection/algo/incremental.cc",
    "smd": "moses/feature-selection/algo/stochastic_max_dependency.cc",
    "random": "moses/feature-selection/algo/random.cc",
    "hc": "moses/feature-selection/algo/deme_optimize.cc",
}


def _fs_scorer(ctable: CTable, fs_params: feature_selection_parameters):
    """C++: moses/feature-selection/main/feature-selection.h:158 fs_scorer

    Dispatches on ``fs_params.scorer`` -- only ``"mi"`` (``MICScorerCTable``)
    is ported; ``"pre"`` (precision, ``moses/moses/scoring/scoring.h``'s
    ``pre_scorer``) is scoring-subsystem territory out of this package's
    scope and raises.
    """
    if fs_params.scorer == "mi":
        return MICScorerCTable(ctable, fs_params.mi_confi)
    if fs_params.scorer == "pre":
        raise NotImplementedError(
            "fs_scorer: scorer='pre' (precision, moses/moses/scoring/"
            "scoring.h pre_scorer) is not ported -- out of this package's "
            "scope. Use scorer='mi' (the default)."
        )
    raise NotImplementedError(
        f"fs_scorer: unknown feature selection scorer {fs_params.scorer!r} "
        "(feature-selection.h:190's own OC_ASSERT(false, ...))."
    )


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
    scorer = _fs_scorer(ctable, fs_params)
    return [scorer(feature_set({idx})) for idx in range(table.get_arity())]


def initial_features(
    ilabels: Sequence[str], fs_params: feature_selection_parameters
) -> feature_set:
    """C++: moses/feature-selection/main/feature-selection.cc:104 initial_features

    Converts ``fs_params.initial_features`` (label strings) into a
    ``feature_set`` of column indices, silently ignoring (with a would-be
    ``logger().warn`` -- no logging surface, see module docstring) any
    name not present in ``ilabels``.
    """
    res: feature_set = feature_set()
    for name in fs_params.initial_features:
        try:
            idx = ilabels.index(name)
        except ValueError:
            continue
        res.add(idx)
    return res


def select_feature_sets(
    ctable: CTable, fs_params: feature_selection_parameters
) -> feature_set_pop:
    """C++: moses/feature-selection/main/feature-selection.cc:135 select_feature_sets"""
    if fs_params.algorithm == "simple":
        return simple_select_feature_sets(ctable, fs_params)
    if fs_params.algorithm in _PORTED_ALGORITHMS:
        raise NotImplementedError(
            f"select_feature_sets: algorithm={fs_params.algorithm!r} "
            f"({_PORTED_ALGORITHMS[fs_params.algorithm]}) is not ported. "
            "Use algorithm='simple' (the default)."
        )
    raise NotImplementedError(
        f"select_feature_sets: unknown algorithm {fs_params.algorithm!r} "
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
