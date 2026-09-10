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

Now FULLY ported, including the four functions this docstring used to
list as blocked: ``feature_selection()`` (the top-level orchestration
``fs-main.cc``'s ``main()`` calls), ``write_results``,
``log_selected_features`` and ``err_empty_features``. Both blockers were
re-checked rather than inherited:

* ``ostreamTable``/``saveTable`` really ARE absent from this port's
  ``pymetta.comboreduct.table.table_io`` (re-verified 2026-09-10: ``grep
  -n 'def ostreamTable|def saveTable'`` over that package still returns
  nothing). ``pymetta/comboreduct/**`` is off limits to this task, so
  they are ported into ``pymetta.feature_selection.main.table_output``
  instead, carrying a PLACEMENT-ONLY divergence marker -- the same call
  ``scorers/mutual_info.py`` already made for ``mutualInformation``.
* The "no logging surface this project provides" claim was STALE.
  ``pymetta.util.logger`` is a real port of cogutil's ``Logger``
  (``tools/normalize_log.py`` and ``tests/fixtures/moses_log.json``
  exist to diff its output against classic's byte for byte), and
  ``table.py`` / ``feature_selector.py`` / ``deme_expander.py`` all emit
  ``logger().debug`` today. ``log_selected_features``'s two
  ``logger().info`` records are therefore ported for real, not dropped.

``fs-main.cc``'s own ``main()`` lives in
``pymetta.feature_selection.main.fs_main``; ``eval-features.{h,cc}`` in
``pymetta.feature_selection.main.eval_features``.
"""

from __future__ import annotations

import sys

from pymetta.comboreduct.table.table import CTable, Table, get_indices
from pymetta.feature_selection.algo.deme_optimize import moses_select_feature_sets
from pymetta.feature_selection.algo.incremental import incremental_select_feature_sets
from pymetta.feature_selection.algo.random import random_select_feature_sets
from pymetta.feature_selection.algo.simple import simple_select_feature_sets
from pymetta.feature_selection.algo.stochastic_max_dependency import smd_select_feature_sets
from pymetta.feature_selection.ctable_subsample import subsampleCTable
from pymetta.feature_selection.main.feature_selection_params import (
    feature_selection_parameters,
    feature_set,
    feature_set_pop,
    fs_scorer,
    initial_features,
)
from pymetta.feature_selection.main.table_output import ostreamTable, saveTable
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.util.logger import logger, ostream_double

__all__ = [
    "err_empty_features",
    "feature_selection",
    "initial_features",
    "log_selected_features",
    "score_individual_features",
    "select_feature_sets",
    "select_features",
    "write_results",
]


def err_empty_features(outfile: str) -> None:
    """C++: moses/feature-selection/main/feature-selection.cc:51 err_empty_features

    "If the dataset is degenerate, it can happen that feature selection
    will fail to find any features. Rather than just crashing as a
    result, write out an empty file, which at least allows downstream
    processes to infer what happened" (the function's own comment,
    cc:53-57).

    Ends with ``exit(0)`` -- a SUCCESS exit taken from the middle of
    ``feature_selection()``, not an error path. ``SystemExit(0)`` is the
    direct Python analogue and is what ``problem_params.parse_options``
    already uses for classic's ``--version`` ``exit(0)``; unlike a bare
    ``return`` it also stops ``feature_selection()``'s caller, which is
    exactly what ``exit(0)`` does.
    """
    logger().warn("No features have been selected.")
    print("No features have been selected.", file=sys.stderr)
    if outfile:
        with open(outfile, "w") as out:
            out.write("# No features have been selected.\n")
    raise SystemExit(0)


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
    # C++: feature-selection.cc:169-174. Ported for real (stderr + exit 1)
    # rather than as a `NotImplementedError`, which is what it used to be:
    # this is a REACHABLE user-facing error path of the standalone binary,
    # not a gate over an unported subsystem, and it is now compared
    # byte-for-byte against classic's own stderr. Measured 2026-09-10,
    # `feature-selection ... -a bogus`: classic writes exactly this line
    # and exits 1.
    print(
        f"Fatal Error: Algorithm '{fs_params.algorithm}' is unknown, "
        "please consult the help for the list of algorithms.",
        file=sys.stderr,
    )
    raise SystemExit(1)


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

    The ``Table`` overload's own body is `feature-selection.cc:187-196`:
    compress, then -- only when ``subsampling_ratio < 1.0`` -- log the
    size, subsample in place, log it again, and delegate to the ``CTable``
    overload. This used to raise instead, accurately, while
    ``subsampleCTable`` was unported; it is ported now
    (``pymetta.feature_selection.ctable_subsample``).
    """
    if isinstance(table_or_ctable, Table):
        ctable = table_or_ctable.compressed()
        if fs_params.subsampling_ratio < 1.0:
            # C++: feature-selection.cc:190-195. `fs_params.subsampling_ratio`
            # is already a C++ `float` on this path, so `subsampleCTable`'s
            # own `float ratio` parameter narrows nothing further here --
            # unlike the `deme` caller, where a `double` is narrowed at the
            # call and changes how many rows come out.
            logger().debug(
                "Subsample table (size = %d)", ctable.uncompressed_size()
            )
            subsampleCTable(fs_params.subsampling_ratio, ctable)
            logger().debug("New size = %d", ctable.uncompressed_size())
    else:
        ctable = table_or_ctable

    fs_pop = select_feature_sets(ctable, fs_params)
    if not fs_pop:
        raise AssertionError("select_features: There might a bug")  # OC_ASSERT(!fs_pop.empty())
    return fs_pop[0][1]


def log_selected_features(
    old_arity: int, ftable: Table, fs_params: feature_selection_parameters
) -> None:
    """C++: moses/feature-selection/main/feature-selection.cc:82 log_selected_features

    Two ``logger().info`` records: the count, then one line per surviving
    input column. The scores come from ``score_individual_features``
    called on the ALREADY-FILTERED table, so index ``i`` of ``sco``
    lines up with index ``i`` of ``ftable.itable.get_labels()`` -- these
    are the filtered table's own column indices, not the original
    table's.

    ``logger().info() << ... << sco[i]`` streams a bare ``double``
    through a freshly constructed ``std::stringstream``, i.e. at an
    ``ostream``'s default precision of 6 -- ``ostream_double``
    (``pymetta.util.logger``) is that formatting, and the reason this
    line reads ``x3 0.548795`` and not ``x3 0.5487949...``.
    """
    # C++: cc:86-87. `%d` fed `arity_t` (an int) on both sides.
    logger().info("%d out of %d features have been selected", ftable.get_arity(), old_arity)

    labs = ftable.itable.get_labels()
    sco = score_individual_features(ftable, fs_params)
    for i in range(len(labs)):
        logger().info("log_selected_features(): " + labs[i] + " " + ostream_double(sco[i]))


def write_results(
    selected_table: Table, fs_params: feature_selection_parameters
) -> None:
    """C++: moses/feature-selection/main/feature-selection.cc:98 write_results

    ``Table table_wff = selected_table;`` is a full value copy that C++
    then never mutates -- neither ``ostreamTable`` nor ``saveTable``
    takes a non-const reference -- so the copy is dead and is not
    reproduced. ``Table.__init__`` copies both members
    (docs/CARRY-FORWARD.md 1.1), so making the copy would be correct
    too, just wasted.

    Empty ``output_file`` means stdout. ``sys.stdout.write`` (not
    ``print``) because ``ostreamTable`` already ends every row with a
    newline, the last one included.
    """
    if not fs_params.output_file:
        sys.stdout.write(ostreamTable(selected_table))
    else:
        saveTable(fs_params.output_file, selected_table)


def feature_selection(table: Table, fs_params: feature_selection_parameters) -> None:
    """C++: moses/feature-selection/main/feature-selection.cc:199 feature_selection

    The standalone binary's whole body after argument parsing and table
    loading: select, add the forced features, then either bail out via
    ``err_empty_features`` or filter/log/write.

    ``selected_features`` is a ``std::set<arity_t>``, which iterates
    ASCENDING; ``Table::filtered`` consumes that order directly
    (``ITable::filtered`` keeps columns "in the ORDER GIVEN"). A Python
    ``set`` iterates in hash order, so ``sorted()`` is the faithful
    transliteration -- the same trap docs/CARRY-FORWARD.md records
    against ``stochastic_max_dependency.py``. Measured: on
    ``trainingBoolean.dat -u y -C2`` the selected set is ``{1, 2}``,
    whose hash order happens to be ascending too, so this ONE case
    cannot tell the two apart -- ``tests/feature_selection/
    test_fs_main.py`` therefore also covers a case whose hash order is
    NOT ascending.

    ``get_indices(force_features_str, itable.get_labels())`` returns
    indices into the INPUT columns only (``table.itable.get_labels()``,
    not ``table.get_labels()``), so a ``--force-feature`` naming the
    target column selects nothing -- classic's own behaviour, cc:207-208.
    """
    # C++: cc:203
    selected_features = select_features(table, fs_params)

    # C++: cc:205-210 -- "Add the enforced features"
    if fs_params.force_features_str:
        force_idxs = get_indices(fs_params.force_features_str, table.itable.get_labels())
        selected_features.update(force_idxs)

    # C++: cc:212-219
    if not selected_features:
        err_empty_features(fs_params.output_file)
    else:
        filt_table = table.filtered(sorted(selected_features))
        log_selected_features(table.get_arity(), filt_table, fs_params)
        write_results(filt_table, fs_params)
