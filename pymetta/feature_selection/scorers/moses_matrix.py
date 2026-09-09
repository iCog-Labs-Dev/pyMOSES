"""C++: moses/feature-selection/scorers/moses_matrix.h pre_scorer

Wrapper to use MOSES's own precision confusion-matrix scorer
(``pymetta.moses.scoring.precision_bscore`` -- read-only import; that
package is out of THIS package's own edit scope, per the task brief, but
nothing stops reading its public API the same way
``pymetta.moses.deme.feature_selector`` already does) as a feature-set
scorer: filter the ctable down to the given feature set, then ask
``precision_bscore.best_possible_bscore()`` "attempts to maximize the best
possible score one would get (w.r.t. some fitness function) given the
feature set being evaluated" (the C++ class's own doc comment).

This is the scorer ``fs_scorer<FeatureSet>`` (``main/feature_selection.py``'s
``_fs_scorer``) wires up for ``scorer="pre"`` -- reachable from
``stochastic_max_dependency``/``deme_optimize`` (both go through
``fs_scorer``); ``incremental``/``random``/``simple`` never call
``fs_scorer`` at all (they use bare ``MutualInformation`` or nothing).
"""

from __future__ import annotations

from collections.abc import Iterable

from pymetta.comboreduct.table.table import CTable
from pymetta.feature_selection.scorers.fs_scorer_base import fs_scorer_base
from pymetta.moses.scoring.precision_bscore import precision_bscore
from pymetta.moses.scoring.time_dispersion import TemporalGranularity


class pre_scorer(fs_scorer_base):
    """C++: moses/feature-selection/scorers/moses_matrix.h:49 pre_scorer"""

    def __init__(
        self,
        ctable: CTable,
        confi: float = 100.0,
        penalty: float = 1.0,
        min_activation: float = 0.5,
        max_activation: float = 1.0,
        positive: bool = True,
    ) -> None:
        """C++: moses/feature-selection/scorers/moses_matrix.h:53 pre_scorer"""
        super().__init__(ctable, confi)
        self._penalty = penalty
        self._min_activation = min_activation
        self._max_activation = max_activation
        self._positive = positive

    def __call__(self, fs: Iterable[int]) -> float:
        """C++: moses/feature-selection/scorers/moses_matrix.h:63 operator()

        Constructs a fresh ``precision_bscore`` over the ctable FILTERED to
        ``fs`` (``super::_ctable.filtered(fs)``) with the same fixed
        non-default arguments C++ hardcodes at every call
        (``dispersion_pressure=0.0``, ``dispersion_exponent=1.0``,
        ``exact_experts=True``, ``bias_scale=1.0``, ``time_bscore=False``,
        ``granularity=day``, ``disable_debug_log=True``) -- these seven are
        NOT ``pre_scorer``'s own fields, they are moses_matrix.h:71-77's own
        literal constants, reproduced verbatim rather than exposed as
        parameters here.

        ``boost::accumulate(sc.best_possible_bscore(), 0.0)`` is Python's
        builtin ``sum`` over the same sequence -- both are a plain left
        fold with ``+`` starting from ``0.0``.
        """
        fs_list = list(fs)
        filtered_ctable = self._ctable.filtered(fs_list)
        sc = precision_bscore(
            filtered_ctable,
            self._penalty,
            self._min_activation,
            self._max_activation,
            self._positive,
            0.0,  # dispersion_pressure
            1.0,  # dispersion_exponent
            True,  # exact_experts
            1.0,  # bias_scale
            False,  # time_bscore
            TemporalGranularity.day,  # granularity
            True,  # disable_debug_log
        )
        precision = sum(sc.best_possible_bscore())
        confidence = self.confidence(len(fs_list))
        return precision * confidence
