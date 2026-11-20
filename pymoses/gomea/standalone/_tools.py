"""NOT A PORT: the IT linkage study (BACKLOG 13.9). Access to the two tools of
``tools/gomea`` that the harness reuses (plan section 4, "Reuse, do not copy"):

* ``sr_eval``: the final scoring of the reported model and the cross-check (``run.py``);
* ``sr_linear``: ``independent_columns``, the drop of aliased term columns
  (``evaluator.ITScorer``).

``tools/gomea`` is not a package; the runner (``moses_sr_runner.py``) also puts it on
``sys.path`` and imports these modules by name. The imports are lazy.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "gomea"


def _load(name: str):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return importlib.import_module(name)


def sr_eval():
    return _load("sr_eval")


def sr_linear():
    return _load("sr_linear")
