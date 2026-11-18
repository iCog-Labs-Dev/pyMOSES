"""Server round 13: the per-run tables of `phase_report.py` (`SERVER-ROUND13-PREREGISTRATION.md`,
"Rules" and "Arms").

`phase_report.py` calls these functions only when the phase's spec asks for them
(`"size_terms"`, `"run_table"`, `"same_search"`), so the report of every earlier phase does not
change. Each summary is a regression summary of `moses_sr_runner.py`, loaded by
`phase_report._load` (with `_events_path`, the run's `events.jsonl`).

* `terms_size(s)`: what the program size of a TLS arm adds to the node count. The node count
  plus `n_terms` when the final model uses the per-term fit (a `terms` record with 2 or more
  terms and no fallback; with the TLS-G guard only when `terms.guard` is `"terms"`). A final
  model with the plain a + b*f adds 0, as does every summary without a `terms` record.
* `run_rows(runs, size, nodes)`: one row for each run (`RUN_FIELDS`).
* `above_100_rows(runs)`: the ok runs with a test NMSE above 100 (`ABOVE_100_FIELDS`).
* `same_search_rows(runs, a, b, problems)`: for each (dataset, split), whether arms `a` and `b`
  printed the same candidates (the same program strings in the same order, from each run's
  `candidates.json`).

Demes and knobs come from the "end" events of `events.jsonl` with evaluations
(`ledger.requests > 0`, as `phase_report._demes` counts demes). The knob count of a deme is
its `dim` field. "Knobs per deme" is the median over the run's demes, "total knobs" the sum
(the definitions of `SERVER-ROUND11-FINDINGS.md` and `SERVER-ROUND12-FINDINGS.md`). HC-LM logs
no `dim`, so its knob columns are empty.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

LEXICASE_KEYS = ("n_cases", "selections", "cases_used_mean", "cases_used_max",
                 "pool_size_mean", "bscore_recomputes", "bscore_recompute_cpu_s")
RUN_FIELDS = (
    "dataset", "arm", "split", "status", "test_nmse", "test_nmse_clipped", "train_nmse",
    "val_nmse", "size", "nodes", "n_terms", "fallback", "terms_guard", "n_candidates",
    "guard_plain_candidates", "cpu_s", "evals", "demes", "knobs_per_deme", "total_knobs",
    "lm_calls", "lm_share", "lm_guard_checked", "lm_guard_rejected", "lm_pole_checked",
    "lm_pole_rejected", *(f"lexicase_{k}" for k in LEXICASE_KEYS),
)
ABOVE_100_FIELDS = ("dataset", "arm", "split", "test_nmse", "train_nmse", "val_nmse")
SAME_SEARCH_FIELDS = ("a", "b", "dataset", "split", "result", "n_a", "n_b", "first_difference")


def terms_size(s: dict) -> int:
    """What the size of the final model adds to its node count (see the module docstring)."""
    t = s.get("terms")
    if not isinstance(t, dict):
        return 0
    n = t.get("n_terms")
    if not isinstance(n, int) or n < 2 or t.get("fallback") is not False:
        return 0
    if t.get("guard", "terms") != "terms":
        return 0
    return n


def _end_events(s: dict) -> list[dict] | None:
    """The "end" events with evaluations of one run; `None` when it has no `events.jsonl`."""
    path = s.get("_events_path")
    if not path or not Path(path).is_file():
        return None
    out = []
    with open(path) as fh:
        for line in fh:
            if '"end"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if (isinstance(ev, dict) and ev.get("type") == "end"
                    and (ev.get("ledger") or {}).get("requests", 0) > 0):
                out.append(ev)
    return out


def knob_counts(s: dict) -> tuple[int | None, float | None, int | None]:
    """(demes, knobs per deme (median), total knobs) of one run."""
    ends = _end_events(s)
    if ends is None:
        return None, None, None
    dims = [e["dim"] for e in ends if isinstance(e.get("dim"), int)]
    if not dims:
        return len(ends), None, None
    return len(ends), statistics.median(dims), sum(dims)


def _candidates(s: dict) -> list | None:
    """The run's candidates.json (next to its summary.json), or `None`."""
    path = s.get("_events_path")
    if not path:
        return None
    f = Path(path).with_name("candidates.json")
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def _guard_plain(s: dict) -> int | None:
    """TLS-G: the number of candidates whose guard kept the plain a + b*f."""
    if not isinstance(s.get("terms"), dict) or "guard" not in s["terms"]:
        return None
    cands = _candidates(s)
    if cands is None:
        return None
    return sum((c.get("terms") or {}).get("guard") == "plain" for c in cands)


def _evals(s: dict):
    for k in ("n_evals_total", "n_evals"):
        if s.get(k) is not None:
            return s[k]
    return None


def run_rows(runs: dict, size, nodes) -> list[dict]:
    """One row for each run, sorted by dataset, arm and split. `size(s)` and `nodes(s)` are the
    report's program size and node count."""
    rows = []
    for ds in sorted(runs):
        for arm in sorted(runs[ds]):
            for split in sorted(runs[ds][arm]):
                s = runs[ds][arm][split]
                t = s.get("terms") if isinstance(s.get("terms"), dict) else {}
                co = s.get("const_opt") or {}
                lex = s.get("lexicase") or {}
                demes, kpd, total = knob_counts(s)
                row = {
                    "dataset": ds, "arm": arm, "split": split, "status": s.get("status"),
                    "test_nmse": s.get("test_nmse"),
                    "test_nmse_clipped": s.get("test_nmse_clipped"),
                    "train_nmse": s.get("train_nmse"), "val_nmse": s.get("val_nmse"),
                    "size": size(s), "nodes": nodes(s),
                    "n_terms": t.get("n_terms"), "fallback": t.get("fallback"),
                    "terms_guard": t.get("guard"), "n_candidates": s.get("n_candidates"),
                    "guard_plain_candidates": _guard_plain(s),
                    "cpu_s": s.get("cpu_s"), "evals": _evals(s), "demes": demes,
                    "knobs_per_deme": kpd, "total_knobs": total,
                    "lm_calls": co.get("lm_calls"), "lm_share": co.get("tuning_share"),
                    "lm_guard_checked": co.get("lm_guard_checked"),
                    "lm_guard_rejected": co.get("lm_guard_rejected"),
                    "lm_pole_checked": co.get("lm_pole_checked"),
                    "lm_pole_rejected": co.get("lm_pole_rejected"),
                }
                row.update({f"lexicase_{k}": lex.get(k) for k in LEXICASE_KEYS})
                rows.append(row)
    return rows


def above_100_rows(runs: dict) -> list[dict]:
    """The ok runs with a test NMSE above 100 (an infinite one included)."""
    rows = []
    for ds in sorted(runs):
        for arm in sorted(runs[ds]):
            for split in sorted(runs[ds][arm]):
                s = runs[ds][arm][split]
                v = s.get("test_nmse")
                if s.get("status") == "ok" and v is not None and (math.isnan(v) or v > 100):
                    rows.append({"dataset": ds, "arm": arm, "split": split, "test_nmse": v,
                                 "train_nmse": s.get("train_nmse"),
                                 "val_nmse": s.get("val_nmse")})
    return rows


def same_search_rows(runs: dict, a: str, b: str, problems: list[str]) -> list[dict]:
    """For each (dataset, split) of arm `a` or `b`: "same" when both runs printed the same
    program strings in the same order, "differ" when not, "missing" when a run or its
    candidates.json is missing."""
    rows = []
    for ds in problems:
        ra, rb = runs.get(ds, {}).get(a, {}), runs.get(ds, {}).get(b, {})
        for split in sorted(set(ra) | set(rb)):
            ca = _candidates(ra[split]) if split in ra else None
            cb = _candidates(rb[split]) if split in rb else None
            row = {"a": a, "b": b, "dataset": ds, "split": split,
                   "n_a": None if ca is None else len(ca),
                   "n_b": None if cb is None else len(cb), "first_difference": None}
            if ca is None or cb is None:
                row["result"] = "missing"
            else:
                pa, pb = [c.get("program") for c in ca], [c.get("program") for c in cb]
                row["result"] = "same" if pa == pb else "differ"
                if pa != pb:
                    row["first_difference"] = next(
                        (i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
            rows.append(row)
    return rows


def same_search_summary(rows: list[dict]) -> list[dict]:
    """For each arm pair: the (dataset, split) pairs compared, same, differ and missing."""
    out: dict = {}
    for r in rows:
        d = out.setdefault((r["a"], r["b"]), {"a": r["a"], "b": r["b"], "pairs": 0, "same": 0,
                                              "differ": 0, "missing": 0})
        d["pairs"] += 1
        d[r["result"]] += 1
    return list(out.values())
