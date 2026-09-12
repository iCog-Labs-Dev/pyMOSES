"""The `merging.py` half of `metapopulation` must be attached unconditionally.

Classic splits `metapopulation`'s methods across `metapopulation.cc` and
`merging.cc`; the linker guarantees a program sees both halves. This port
mirrors that file split and attaches the second half with `setattr` at import
time, which Python performs only if something actually imports `merging`.

Before `pymoses/moses/metapopulation/__init__.py` imported it for that side
effect, importing `metapopulation` alone yielded a class MISSING `merge_demes`
and `merge_candidates` while `select_exemplar` -- defined in
`metapopulation.py` itself -- was present. An S11 agent blocked on exactly
that. These tests fail if that import is removed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Attached by merging.py's setattr; absent unless merging is imported.
# All 13 grafts (merging.py:895-907) -- pinning only a subset would let a
# partial regression of commit 9b909c9 (which made this attachment
# unconditional) pass silently for whichever grafts happened not to be
# checked.
_MERGING_METHODS = [
    "_min_pool_size",
    "useful_score_range",
    "trim_down_deme",
    "deme_to_trees",
    "merge_candidates",
    "merge_demes",
    "resize_metapop",
    "get_new_candidates",
    "sort_demes",
    "keep_top_unique_candidates",
    "recompute_scores_over_whole_dataset",
    "ss_filter",
    "log_best_candidates",
]


@pytest.mark.parametrize("name", _MERGING_METHODS)
def test_merging_methods_are_attached_without_importing_merging(name):
    """Import `metapopulation` by its own path and require the merging half.

    Run in a FRESH interpreter: within one pytest session some other module
    has almost certainly imported `merging` already, which would make this
    pass for the wrong reason -- the exact vacuity this suite keeps hitting.
    """
    source = (
        "from pymoses.moses.metapopulation.metapopulation import metapopulation\n"
        f"raise SystemExit(0 if hasattr(metapopulation, {name!r}) else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"metapopulation.{name} missing on a bare import of metapopulation.py. "
        "pymoses/moses/metapopulation/__init__.py must import merging for its "
        f"setattr side effect. stderr: {result.stderr}"
    )
