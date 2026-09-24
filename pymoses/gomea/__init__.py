"""GP-GOMEA-style population inner optimizer for pyMoses.

This package sits alongside ``pymoses.moses`` and treats a frozen MOSES
representation's logical discrete fields as the search space for a
population-based optimizer. It never modifies ``pymoses.moses`` itself --
see ``docs/gomea/P1.0-AUDIT.md`` for the seam this package plugs into.
"""

from pymoses.gomea.oracle import BudgetExhausted, EvaluationLedger, EvaluationOracle
from pymoses.gomea.schema import FrozenSchema, UnsupportedRepresentation

__all__ = [
    "BudgetExhausted",
    "EvaluationLedger",
    "EvaluationOracle",
    "FrozenSchema",
    "UnsupportedRepresentation",
]
