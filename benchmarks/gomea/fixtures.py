"""Frozen-exemplar capture for the frozen-representation experiments
(03_EXPERIMENT_PROTOCOL.md Sec 2's "Frozen-representation experiments").

`capture_exemplars` runs one HC MOSES search and records the first
`n_expansions` exemplars `deme_expander.create_demes` is asked to build a
representation from -- the same combo-tree text `deme_expander.py`'s own
`tree_to_stream_str(exemplar)` calls already use for its debug logging, so a
captured fixture is byte-identical to what a real run would have logged.

`replay` -- rebuilding a representation from a captured exemplar and running
HC/GOMEA on it under an equal local evaluation budget -- is DEFERRED to
Task 10. It is intentionally NOT implemented here: this module only ever
captures fixtures independently of any confirmatory result, per the
protocol's "Use fixtures collected independently of the confirmatory test
results" (Sec 2). Do not add a `replay` function to this module; Task 10
owns it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

__all__ = ["capture_exemplars"]


def _capture_child(argv: list[str], n_expansions: int, out_path: str) -> None:
    """Child-process entry point (top-level, picklable by reference under
    `multiprocessing.get_context("spawn")`). Patches
    `deme_expander.create_demes` for the duration of one in-process moses
    run, recording the exemplar text of its first `n_expansions` calls, then
    writes the captured list to `out_path` as JSON.
    """
    import contextlib
    import io

    from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str
    from pymoses.moses.deme.deme_expander import deme_expander
    from pymoses.moses.main.moses_exec import moses_exec

    exemplars: list[str] = []
    original = deme_expander.create_demes

    def wrapped(self, exemplar, *args, **kwargs):
        if len(exemplars) < n_expansions:
            exemplars.append(tree_to_stream_str(exemplar))
        return original(self, exemplar, *args, **kwargs)

    deme_expander.create_demes = wrapped
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                moses_exec(["moses", *argv])
            except SystemExit:
                pass
    finally:
        deme_expander.create_demes = original

    Path(out_path).write_text(json.dumps(exemplars, indent=2) + "\n")


def capture_exemplars(argv: list[str], n_expansions: int, out_path: str | Path) -> None:
    """Run one HC MOSES search (`argv` should select `-a hc`; this
    function does not itself enforce that -- picking the arm to capture
    from is the caller's job) in a freshly spawned child process, and
    record the combo-tree text of the exemplar handed to the first
    `n_expansions` calls of `deme_expander.create_demes` to `out_path` as a
    JSON list of strings.

    Runs in a CHILD process, the same isolation `runner.py`'s own
    `_child_main` uses, so this capture never touches the calling process's
    RNG state or module-level caches.
    """
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_capture_child, args=(list(argv), n_expansions, str(out_path)))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"capture_exemplars: child process exited with code {p.exitcode}"
        )
