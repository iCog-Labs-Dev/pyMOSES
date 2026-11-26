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
  before the first candidate. It never goes down: the metapopulation replaces its best
  candidate only for a higher raw score (`csc.get_score()`), or the same score with a lower
  complexity. It is read at each deme end, so it lags by up to one deme, and it is not the
  quantity of the GOMEA trace of round 16 (the best raw score over every evaluation).

These are the values that the "Stats:" row of the MOSES log prints after each expansion
(`local_moses.py`) when the log level is INFO. The runner uses `-l WARN`, so the log does not
have the row; the hook reads the same two values and does not change the log level.

The hook only reads: it calls `mp.best_score()` (a copy of the best composite score) and
reads `stats`. It draws no random number and changes no state, so the search and the output
of MOSES do not change (`tests/gomea/test_hc_progress.py`). It writes and flushes one line per
expansion, so a run that stops early keeps its lines.

**A write error does not stop the run (the RV16 review of round 17, F8).** The lines are a
descriptive output, so a fault there must not make a run fail in a primary contrast. When the
open, the write or the close of `path` raises an `OSError` (for example a full disk or a quota
error), the hook:

* stops writing: it writes no more lines for the rest of the run (`failed` is true, `error` has
  the message);
* writes one line to stderr: `ERROR_PREFIX`, the expansion and the message
  (`hc_progress: write error at expansion N: OSError: ...; no more lines`). The runner reads
  stderr through a pipe, so this line does not need the file system. `moses_sr_runner.py`
  copies the message into the summary key `hc_progress_error` (only when there is such a line);
* tries once to append a marker line `{"hc_progress_error": message, "expansion": N}` to
  `path`, after a newline that ends a cut line. The marker has no `n_evals`, so the runner does
  not read it as a point. If this write also fails, the hook ignores it.

MOSES then goes on and ends normally. The lines written before the error stay in the file, so
the summary has the checkpoints up to that deme, and `anytime_check.log_evals_match_n_evals` is
false.

`moses_sr_runner.py` installs it for a pyMoses `-a hc` job of a manifest with
`"hc_anytime": true` and turns the lines into the summary fields `demes`, `anytime`,
`anytime_source` ("hc-demes") and `anytime_check` (`hc_anytime_records`).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HOOK = None
ERROR_PREFIX = "hc_progress: write error"  # the stderr line of a write error (F8)


class HcProgress:
    """The hook: one JSON line per expansion (see the module docstring)."""

    def __init__(self, path: str | Path) -> None:
        from pymoses.moses.moses.types import very_worst_score

        self.path = Path(path)
        self.worst = float(very_worst_score)
        self.lines = 0
        self.failed = False  # true after a write error: no more lines (F8)
        self.error: str | None = None

    def __call__(self, mp, dex, pa, stats) -> None:
        if self.failed:
            return
        raw = float(mp.best_score())
        ok = math.isfinite(raw) and raw > self.worst
        rec = {"expansion": int(stats.n_expansions), "n_evals": int(stats.n_evals),
               "best_raw": raw if ok else None}
        try:
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        except OSError as exc:
            self._fail(rec["expansion"], exc)
            return
        self.lines += 1

    def _fail(self, expansion: int, exc: OSError) -> None:
        """Stop writing and mark the error on stderr and in the file (see the module
        docstring). This never raises."""
        self.failed = True
        self.error = f"{type(exc).__name__}: {exc}"[:300]
        try:
            print(f"{ERROR_PREFIX} at expansion {expansion}: {self.error}; no more lines",
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
        try:
            with self.path.open("a") as fh:
                fh.write("\n" + json.dumps({"hc_progress_error": self.error,
                                             "expansion": expansion}) + "\n")
        except OSError:
            pass


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
