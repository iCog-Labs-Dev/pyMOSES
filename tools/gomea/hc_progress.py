"""NOT A PORT: server round 17 (`SERVER-ROUND17-PREREGISTRATION.md`, "Contrasts",
descriptive): the best-so-far training error of an HC run at deme resolution.

`install(path)` adds a hook to pyMoses's `local_moses.expansion_hooks` (pyMoses only). After
each expansion (one deme for `-a hc`) the hook appends one JSON line to `path`:

* `expansion`: `stats.n_expansions`, the number of expansions so far;
* `n_evals`: `stats.n_evals`, the run's evaluations so far (MOSES's own count, which also
  counts the LM passes of `-a hc --const-opt lm`; the last line has the `n_evals` of MOSES's
  output);
* `best_raw`: `mp.best_score()`, the best raw score (no complexity penalty) of all candidates
  that the metapopulation has merged so far (`metapopulation.update_best_candidates`), or null
  before the first candidate. It never goes down.

These are the values that the "Stats:" row of the MOSES log prints after each expansion
(`local_moses.py`) when the log level is INFO. The runner uses `-l WARN`, so the log does not
have the row; the hook reads the same two values and does not change the log level.

The hook only reads: it calls `mp.best_score()` (a copy of the best composite score) and
reads `stats`. It draws no random number and changes no state, so the search and the output
of MOSES do not change (`tests/gomea/test_hc_progress.py`). It writes and flushes one line per
expansion, so a run that stops early keeps its lines.

`moses_sr_runner.py` installs it for a pyMoses `-a hc` job of a manifest with
`"hc_anytime": true` and turns the lines into the summary fields `demes`, `anytime`,
`anytime_source` ("hc-demes") and `anytime_check` (`hc_anytime_records`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_HOOK = None


class HcProgress:
    """The hook: one JSON line per expansion (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        from pymoses.moses.moses.types import very_worst_score

        self.path = Path(path)
        self.worst = float(very_worst_score)
        self.lines = 0

    def __call__(self, mp, dex, pa, stats) -> None:
        raw = float(mp.best_score())
        ok = math.isfinite(raw) and raw > self.worst
        rec = {"expansion": int(stats.n_expansions), "n_evals": int(stats.n_evals),
               "best_raw": raw if ok else None}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        self.lines += 1


def install(path: str | Path) -> HcProgress:
    """Add an `HcProgress` to `local_moses.expansion_hooks` (replacing one installed before)."""
    global _HOOK
    from pymoses.moses.moses import local_moses

    uninstall()
    _HOOK = HcProgress(path)
    local_moses.expansion_hooks.append(_HOOK)
    return _HOOK


def uninstall() -> None:
    global _HOOK
    from pymoses.moses.moses import local_moses

    if _HOOK is not None and _HOOK in local_moses.expansion_hooks:
        local_moses.expansion_hooks.remove(_HOOK)
    _HOOK = None
