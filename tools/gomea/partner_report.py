"""Server round 17 (phase 1bl, `SERVER-ROUND17-PREREGISTRATION.md`, "Pairing with round 16"):
the report reads the partner runs of another round beside its own runs. `phase_report.py`
calls these functions only when a spec has the key `"partners"`, so the report of every
earlier phase does not change.

**The spec key `"partners"`**::

    "partners": {"dirs": [DIR, ...], "pairs": {arm: partner arm, ...}}

* `dirs`: result directories of the other round (each with `runs/*/summary.json`), as
  `phase_report.py` takes its own: relative to the working directory of the report (the
  repository root, where `phase_run.sh` runs it) or absolute. Round 17:
  `../gomea-results/phase1bk/phase1bk-sr`, `phase1bk-sr-100K` and `phase1bk-sr-Dow` (round 16).
* `pairs`: each arm of this phase that has a partner, and the arm of its partner. Round 17:
  HC-LM and HC-TLS to BASE-TLS, HC-LM-100K and HC-TLS-100K to BASE-TLS-100K.

**The rule.** The report first loads its own result directories. Then it loads from `dirs` (a
later directory overrides an earlier one, as for its own) exactly the partner runs of its own
runs: the summary of (task, P, seed) is loaded when an own run (task, A, seed) has
`pairs[A] = P`, and the own runs have no run (task, P, seed). Other summaries of `dirs` (other
arms, other tasks or seeds, for example the round 16 RV arms, the BASE-TLS-100K runs on Dow or
the splits that this phase has not run yet) are not loaded. A loaded partner run is then like
an own run: it pairs by (task, seed), and a failed one counts as infinitely bad.

**Missing partners.** An own run whose partner is not found is not paired (the contrasts take
the pairs that have all their runs, as for any missing run). The report does not stop. It
writes OUT_DIR/partners.csv (`PARTNER_FIELDS`), one row for each own run with a partner arm:
the task, the seed, the arm, the partner arm, `partner_status` (the partner's status, or
"missing"), and the directory it came from. OUT_DIR/partners.json has the counts (`expected`,
`found`, `missing`, `not_ok`) and the list of the missing partners ("task/partner arm/seed").
The report prints the counts as one JSON line, and when a partner is missing it prints a
warning with the missing runs to stderr.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

PARTNER_FIELDS = ("task", "seed", "arm", "partner_arm", "partner_status", "partner_dir")


def check_spec(spec: dict) -> str | None:
    """An error message for a malformed `"partners"`, else `None`. A spec without the key gives
    `None`."""
    p = spec.get("partners")
    if p is None:
        return None
    if not isinstance(p, dict) or set(p) != {"dirs", "pairs"}:
        return f"bad partners {p!r} (want {{\"dirs\": [...], \"pairs\": {{arm: partner}}}})"
    dirs, pairs = p["dirs"], p["pairs"]
    if not isinstance(dirs, list) or not dirs or not all(isinstance(d, str) and d for d in dirs):
        return f"bad partners dirs {dirs!r} (want a non-empty list of directories)"
    if not isinstance(pairs, dict) or not pairs or not all(
            isinstance(k, str) and isinstance(v, str) and k != v for k, v in pairs.items()):
        return f"bad partners pairs {pairs!r} (want {{arm: another arm}})"
    if set(pairs) & set(pairs.values()):
        return f"bad partners pairs {pairs!r} (a partner arm cannot have a partner)"
    return None


def load(spec: dict, runs: dict) -> dict:
    """Add the partner runs of `runs` (`{task: {arm: {seed: summary}}}`, the own runs) from the
    spec's `"partners"` (see the module docstring). Returns `{(task, partner arm, seed): dir}`
    for the loaded runs."""
    pairs = spec["partners"]["pairs"]
    want = {(t, pairs[a], k) for t, arms in runs.items() for a, seeds in arms.items()
            if a in pairs for k in seeds}
    want = {(t, p, k) for t, p, k in want if k not in runs.get(t, {}).get(p, {})}
    found: dict = {}
    for d in spec["partners"]["dirs"]:
        for f in sorted(glob.glob(str(Path(d) / "runs/*/summary.json"))):
            try:
                s = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue  # a half-written summary: the partner is reported missing
            key = (s.get("task"), s.get("arm"), s.get("seed"))
            if key in want:
                s["_events_path"] = str(Path(f).with_name("events.jsonl"))
                found[key] = (d, s)
    for (t, p, k), (_, s) in found.items():
        runs.setdefault(t, {}).setdefault(p, {})[k] = s
    return {key: d for key, (d, _) in found.items()}


def rows(spec: dict, runs: dict, loaded: dict) -> list[dict]:
    """The rows of partners.csv: each own run with a partner arm (see the module docstring)."""
    pairs = spec["partners"]["pairs"]
    out = []
    for t in sorted(runs):
        for a in sorted(runs[t]):
            if a not in pairs:
                continue
            for k in sorted(runs[t][a]):
                s = runs[t].get(pairs[a], {}).get(k)
                out.append({"task": t, "seed": k, "arm": a, "partner_arm": pairs[a],
                            "partner_status": "missing" if s is None else s.get("status"),
                            "partner_dir": loaded.get((t, pairs[a], k), "")})
    return out


def summary(table: list[dict]) -> dict:
    """The counts of partners.json, over the distinct partner runs of `table`."""
    status = {}
    for r in table:
        status[(r["task"], r["partner_arm"], r["seed"])] = r["partner_status"]
    missing = sorted(f"{t}/{p}/{k}" for (t, p, k), s in status.items() if s == "missing")
    return {"expected": len(status), "found": len(status) - len(missing),
            "missing": len(missing),
            "not_ok": sum(s not in ("missing", "ok") for s in status.values()),
            "missing_runs": missing}


def write_tables(spec: dict, runs: dict, loaded: dict, out: Path, write_csv) -> None:
    """OUT_DIR/partners.csv and partners.json when the spec has `"partners"`; `write_csv` is
    `phase_report._write_csv`."""
    if not spec.get("partners"):
        return
    table = rows(spec, runs, loaded)
    write_csv(out / "partners.csv", table, list(PARTNER_FIELDS))
    counts = summary(table)
    (out / "partners.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps({"partners": {k: v for k, v in counts.items() if k != "missing_runs"}}))
    if counts["missing"]:
        print(f"WARNING: {counts['missing']} of {counts['expected']} partner runs are missing "
              f"(not paired): {', '.join(counts['missing_runs'])}", file=sys.stderr)
