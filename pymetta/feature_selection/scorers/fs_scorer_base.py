"""C++: moses/feature-selection/scorers/fs_scorer_base.h fs_scorer_base

Shared base for feature-set scorers that combine a raw quality measure with
a rough "confidence" factor derived from the table's uncompressed size --
the C++ header's own comment calls this "a very rough approximation of the
confidence in the feature quality measure" (fs_scorer_base.h:47-48). Both
real subclasses in this port's scope inherit it: ``MICScorerCTable``
(``pymetta.feature_selection.scorers.mutual_info``, ``scorers/mutual_info.h:
108``) and ``pre_scorer`` (``pymetta.feature_selection.scorers.moses_matrix``,
``scorers/moses_matrix.h:49``).

``operator()`` is ``= 0`` (pure virtual) in C++ -- reproduced here as an
``abstractmethod``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

from pymetta.comboreduct.table.table import CTable


class fs_scorer_base(ABC):
    """C++: moses/feature-selection/scorers/fs_scorer_base.h:34 fs_scorer_base"""

    def __init__(self, ctable: CTable, confi: float) -> None:
        """C++: moses/feature-selection/scorers/fs_scorer_base.h:37 fs_scorer_base

        # DIVERGENCE(C++ moses/feature-selection/scorers/fs_scorer_base.h:37):
        ``unsigned _usize`` truncates ``_ctable.uncompressed_size()`` (a
        ``double``, since ``count_t`` is ``double`` in this port -- see
        CARRY-FORWARD.md's own type notes) toward zero at the member-
        initializer assignment. DO NOT FIX: ``int(...)`` reproduces that
        truncation exactly (never negative in practice, since
        ``uncompressed_size()`` is a sum of non-negative row counts).
        """
        self._ctable = ctable
        self._confi = confi
        self._usize = int(ctable.uncompressed_size())

    def confidence(self, fs_size: int) -> float:
        """C++: moses/feature-selection/scorers/fs_scorer_base.h:49 confidence

        "Very rough approximation of the confidence in the feature quality
        measure" (the C++ method's own doc comment) --
        ``_usize / (_usize + exp(-_confi * fs_size))``.
        """
        return self._usize / (self._usize + math.exp(-self._confi * fs_size))

    @abstractmethod
    def __call__(self, features: Iterable[int]) -> float:
        """C++: moses/feature-selection/scorers/fs_scorer_base.h:43 operator()"""
        raise NotImplementedError
