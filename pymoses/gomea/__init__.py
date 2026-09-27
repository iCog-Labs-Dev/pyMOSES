"""GP-GOMEA-style population inner optimizer for pyMoses.

This package sits alongside ``pymoses.moses`` and treats a frozen MOSES
representation's logical discrete fields as the search space for a
population-based optimizer. It never modifies ``pymoses.moses`` itself --
see ``docs/gomea/P1.0-AUDIT.md`` for the seam this package plugs into.
"""

from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.hybrid import hc_gomea_optimization
from pymoses.gomea.optimizer import gomea_optimization, gomea_parameters
from pymoses.gomea.oracle import BudgetExhausted, EvaluationLedger, EvaluationOracle
from pymoses.gomea.population import build_population, parse_radii
from pymoses.gomea.schema import FrozenSchema, UnsupportedRepresentation

__all__ = [
    "BudgetExhausted",
    "EngineParams",
    "EvaluationLedger",
    "EvaluationOracle",
    "FrozenSchema",
    "GomeaEngine",
    "LinkageModel",
    "UnsupportedRepresentation",
    "build_population",
    "gomea_optimization",
    "gomea_parameters",
    "hc_gomea_optimization",
    "parse_radii",
]
