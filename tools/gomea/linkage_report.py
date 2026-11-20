"""NOT A PORT: the IT linkage study (BACKLOG 13.9), server round 15. Report additions that
`phase_report.py` uses only when a spec asks for them (`SERVER-ROUND15-PREREGISTRATION.md`,
"Metrics" and "Contrasts"). Without the new spec keys and metrics the report does not change.

**The metric `anytime_area`.** The stand-alone harness (`pymoses/gomea/standalone/run.py`)
writes `anytime` into each `summary.json`: one record for each checkpoint 1, 2, 5, 10, 20, 50
and 100 % of E, with `frac` (the fraction of E) and `best_train_nmse` (the training NMSE of the
best score so far at that checkpoint; for a run that stopped before a checkpoint, its final best
score). Let x_i = log10(frac_i) and y_i = ln(best_train_nmse_i). The area A is the trapezoid
rule over the 7 points, A = sum (x_{i+1} - x_i) (y_i + y_{i+1}) / 2, and W = x_7 - x_1 = 2 (two
decades of evaluations). The value of the metric is exp(A / W): the mean level of the
best-so-far curve on a log-evaluation axis, in NMSE units (a weighted geometric mean of the
best-so-far training NMSE). Lower is better. The raw area A can be 0 or negative (an NMSE below
1), so a log ratio of A is not defined; with exp(A / W), a pooled contrast takes
log(metric_A / metric_B) = (A_A - A_B) / W, so its Wilcoxon test is the paired test on the
difference of the two areas. A run that is not ok, a record that does not have the 7 standard
fractions, or a checkpoint without a finite best score gives no value (the metric is then
infinite, as for a failed run). A best NMSE of 0 at a checkpoint gives the value 0.

**The pooled interaction contrast** `[name, A1, B1, A2, B2, metric]`. For each pair (problem,
seed) that has all four runs, d = log(A1 / B1) - log(A2 / B2) on the metric (lower is better).
Each log ratio follows the rule of the pooled contrasts (`phase_report._pooled_contrast`): a run
that failed (metric infinite) gets the largest rank against it, log(A / B) = +1e12 when only A
failed, -1e12 when only B failed, and 0 when both failed. Then d is the difference of the two
log ratios. The test is the paired Wilcoxon signed-rank test (two-sided) on d, and ratio =
exp(median d). d < 0 means that A1 gains more over B1 than A2 gains over B2. Verdict (in its
Holm family): "A1 gains more" if Holm p < ALPHA and ratio <= 0.9; "A2 gains more" if Holm p <
ALPHA and ratio >= 1/0.9; else "tie". In the row, `a` and `b` are A1 and B1, `a2` and `b2` are A2
and B2; `a_lower` counts the pairs with d < 0, `b_lower` those with d > 0, `ties` those with
d = 0; `median_a`, `median_b`, `median_a2` and `median_b2` are the medians of the four metrics;
the metric column is "interaction:<metric>".

**The spec key `"contrast_families"`**: a list of Holm families, each one
`{"name": NAME, "role": "primary" | "secondary" | "descriptive", "contrasts": [...]}` with an
optional `"per_dataset": true`. A contrast with 4 elements, `[name, A, B, metric]`, is a pooled
contrast of round 8 (r = log(A / B), the verdicts "A better", "A worse", "tie"); a contrast with
6 elements is an interaction contrast (above). Holm covers each family alone. Role "primary" or
"secondary": the verdict rule of the pooled contrasts; the rows go to that family of
contrasts.csv, and their verdicts to verdict.json. Role "descriptive": the rows keep p and the
Holm p of their family, their verdict is "descriptive", they go to the family "descriptive" of
contrasts.csv and not to verdict.json. Every row gets `holm_family` = NAME. With `"per_dataset":
true` each contrast gives one row for each problem of `metrics` (problem = the dataset, the pairs
of that dataset only), and the Holm family is all these rows. The names of the pooled contrasts
(those of `pooled_primary` and `pooled_secondary` too) must be distinct; the names in the
per-dataset families must be distinct, and different from the per-problem contrasts of the spec.

**The spec key `"evals_to_reference"`**: `{arm: reference arm}` (round 15: each arm to the RT arm
of its genome). It writes OUT_DIR/evals_to_reference.csv (`EVALS_FIELDS`), descriptive: for each
problem and arm, the target is the median final training NMSE of the reference arm's runs on
that problem (a failed reference run counts as infinite), and the value of a run is the number of
evaluations at the first point of its best-so-far trace with a training NMSE at most the target
(times 1 + 1e-9, for rounding). The trace is the `trace` of the "end" record of the run's
`events.jsonl` (pairs [evaluations, best training NMSE], one at each strict improvement); the
final training NMSE of a run is the last point of its trace (the search's own NMSE; it equals
`train_nmse` to about 1e-12, the harness's cross-check). A run that is not ok, or that never
reaches the target, has an infinite value. The rows give the runs, the failed runs, the runs
that reached the target, the median over all runs (infinite when more than half did not reach
it), the median over the runs that reached it and the median evaluations used. One more row for
each arm (problem "pooled") pools the runs of all problems.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

ANYTIME_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)  # run.py ANYTIME_FRACTIONS
BIG = 1e12  # as phase_report.BIG
REACH_RTOL = 1e-9
ROLES = ("primary", "secondary", "descriptive")
FAMILY_KEYS = {"name", "role", "contrasts", "per_dataset"}
EVALS_FIELDS = ("problem", "arm", "reference", "target_train_nmse", "runs", "failed", "reached",
                "median_evals_to_reference", "median_evals_reached", "median_evals_used")


# ---- the metric anytime_area


def anytime_area(s: dict) -> float | None:
    """The value of the metric `anytime_area` of one summary (see the module docstring), or
    `None` when there is none."""
    pts = s.get("anytime")
    if not isinstance(pts, list) or not all(isinstance(p, dict) for p in pts):
        return None
    pts = sorted(pts, key=lambda p: p.get("frac") or 0.0)
    if tuple(p.get("frac") for p in pts) != ANYTIME_FRACTIONS:
        return None
    ys = []
    for p in pts:
        v = p.get("best_train_nmse")
        if v is None or not isinstance(v, (int, float)) or not v >= 0 or math.isinf(v):
            return None
        ys.append(-math.inf if v == 0 else math.log(v))
    xs = [math.log10(f) for f in ANYTIME_FRACTIONS]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
    return math.exp(area / (xs[-1] - xs[0]))


# ---- the contrasts


def _log(x: float) -> float:
    return -math.inf if x == 0 else math.log(x)


def log_ratio(va: float, vb: float) -> float:
    """log(A / B) with the rule of the pooled contrasts for a failed (infinite) metric."""
    la, lb = _log(va), _log(vb)
    if math.isfinite(la) and math.isfinite(lb):
        return la - lb
    return 0.0 if la == lb else (BIG if la > lb else -BIG)


def _ratio(med: float) -> float:
    return math.exp(med) if abs(med) < 700 else (math.inf if med > 0 else 0.0)


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def interaction_contrast(name, a1, b1, a2, b2, runs, problems, metric, value,
                         problem_label="pooled") -> dict:
    """The interaction contrast over the pairs (problem, seed) of `problems`; `value(metric, s)`
    is the report's metric value (`phase_report._value`)."""
    v = {k: [] for k in ("a", "b", "a2", "b2")}
    d = []
    for p in problems:
        r = runs.get(p, {})
        ra, rb, rc, rd = (r.get(x, {}) for x in (a1, b1, a2, b2))
        for k in sorted(set(ra) & set(rb) & set(rc) & set(rd)):
            xa, xb, xc, xd = (value(metric, x[k]) for x in (ra, rb, rc, rd))
            v["a"].append(xa)
            v["b"].append(xb)
            v["a2"].append(xc)
            v["b2"].append(xd)
            d.append(log_ratio(xa, xb) - log_ratio(xc, xd))
    med = _median(d)
    return {
        "problem": problem_label, "contrast": name, "a": a1, "b": b1, "a2": a2, "b2": b2,
        "metric": f"interaction:{metric}", "pairs": len(d),
        "a_lower": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "median_a": _median(v["a"]), "median_b": _median(v["b"]),
        "median_a2": _median(v["a2"]), "median_b2": _median(v["b2"]),
        "median_log_ratio": med, "ratio": _ratio(med),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


def _decide(rows, role, alpha) -> None:
    """Holm over `rows` (one family) and the verdicts of `role`."""
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        m = r["median_log_ratio"]
        inter = r["metric"].startswith("interaction:")
        if role == "descriptive":
            r["verdict"] = "descriptive"
        elif ph < alpha and m <= math.log(0.9):
            r["verdict"] = "A1 gains more" if inter else "A better"
        elif ph < alpha and m >= math.log(1 / 0.9):
            r["verdict"] = "A2 gains more" if inter else "A worse"
        else:
            r["verdict"] = "tie"


def _shape_error(c, metrics) -> str | None:
    if not isinstance(c, list) or len(c) not in (4, 6):
        return f"bad contrast {c!r} (want [name, A, B, metric] or [name, A1, B1, A2, B2, metric])"
    if not all(isinstance(x, str) for x in c):
        return f"bad contrast {c!r} (names, arms and the metric are strings)"
    if c[-1] not in metrics:
        return f"unknown metric {c[-1]!r} in contrast {c} (choose from {metrics})"
    return None


def check_spec(spec: dict, metrics) -> str | None:
    """An error message for a malformed `"contrast_families"` or `"evals_to_reference"`, else
    `None`. A spec without the keys gives `None`."""
    fams = spec.get("contrast_families")
    if fams is not None:
        if not isinstance(fams, list) or not fams:
            return f"bad contrast_families {fams!r} (want a list of families)"
        pooled = [c[0] for k in ("pooled_primary", "pooled_secondary") for c in spec.get(k, [])]
        per_problem = {c[1] for k in ("primary", "secondary", "descriptive")
                       for c in spec.get(k, [])}
        per_dataset, names = [], set()
        for f in fams:
            if not isinstance(f, dict) or not {"name", "role", "contrasts"} <= set(f) \
                    or not set(f) <= FAMILY_KEYS:
                return (f"bad family {f!r} in contrast_families (want name, role, contrasts "
                        f"and optional per_dataset)")
            if not isinstance(f["name"], str) or f["name"] in names:
                return f"bad family name {f['name']!r} in contrast_families (distinct strings)"
            names.add(f["name"])
            if f["role"] not in ROLES:
                return f"bad role {f['role']!r} of family {f['name']} (choose from {ROLES})"
            if not isinstance(f.get("per_dataset", False), bool):
                return f"per_dataset of family {f['name']} must be true or false"
            if not isinstance(f["contrasts"], list) or not f["contrasts"]:
                return f"family {f['name']} has no contrasts"
            for c in f["contrasts"]:
                err = _shape_error(c, metrics)
                if err:
                    return err
                (per_dataset if f.get("per_dataset") else pooled).append(c[0])
        if len(pooled) != len(set(pooled)):
            return "contrast_families: the names of the pooled contrasts must be distinct"
        if len(per_dataset) != len(set(per_dataset)) or set(per_dataset) & per_problem:
            return ("contrast_families: the names of the per-dataset contrasts must be distinct "
                    "and differ from the per-problem contrasts")
    ref = spec.get("evals_to_reference")
    if ref is not None and (not isinstance(ref, dict) or not ref or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in ref.items())):
        return f"bad evals_to_reference {ref!r} (want {{arm: reference arm}})"
    return None


def family_rows(spec: dict, runs: dict, problems: list[str], value, pooled_contrast,
                alpha: float) -> list[tuple[str, dict]]:
    """The rows of `"contrast_families"` as (contrasts.csv family, row), decided in their Holm
    families. `pooled_contrast` is `phase_report._pooled_contrast`."""
    out = []
    for f in spec.get("contrast_families") or []:
        rows = []
        units = ([([p], p) for p in problems] if f.get("per_dataset")
                 else [(problems, "pooled")])
        for c in f["contrasts"]:
            for problem_set, label in units:
                if len(c) == 6:
                    rows.append(interaction_contrast(*c[:5], runs, problem_set, c[5], value,
                                                     problem_label=label))
                else:
                    r = pooled_contrast(c[0], c[1], c[2], runs, problem_set, c[3])
                    r["problem"] = label
                    rows.append(r)
        _decide(rows, f["role"], alpha)
        for r in rows:
            r["holm_family"] = f["name"]
            out.append((f["role"], r))
    return out


# ---- the table "evaluations to reach the reference"


def _trace(s: dict) -> list | None:
    """The best-so-far trace of the "end" record of the run's events.jsonl, or `None`."""
    path = s.get("_events_path")
    if not path or not Path(path).is_file():
        return None
    trace = None
    with open(path) as fh:
        for line in fh:
            if '"end"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "end" and isinstance(
                    ev.get("trace"), list):
                trace = ev["trace"]
    return trace


def _final(s: dict) -> float:
    """The final training NMSE of a run (the last point of its trace); infinite when the run
    is not ok or has no trace."""
    if s.get("status") != "ok":
        return math.inf
    t = _trace(s)
    if not t or t[-1][1] is None:
        return math.inf
    return float(t[-1][1])


def evals_to_target(s: dict, target: float) -> float:
    """Evaluations at the first trace point with a training NMSE at most `target`."""
    if s.get("status") != "ok":
        return math.inf
    for n, v in _trace(s) or []:
        if v is not None and (math.isinf(target) or v <= target * (1 + REACH_RTOL)):
            return float(n)
    return math.inf


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _evals_used(s: dict):
    return s.get("n_evals") if s.get("status") == "ok" else None


def evals_to_reference_rows(spec: dict, runs: dict, problems: list[str]) -> list[dict]:
    """The rows of evals_to_reference.csv (see the module docstring)."""
    ref = spec["evals_to_reference"]
    rows, pooled = [], {arm: [] for arm in ref}
    for p in problems:
        for arm, rarm in ref.items():
            mine = list(runs.get(p, {}).get(arm, {}).values())
            refs = list(runs.get(p, {}).get(rarm, {}).values())
            if not mine:
                continue
            target = _med([_final(s) for s in refs]) if refs else None
            vals = [evals_to_target(s, target) if target is not None else math.inf
                    for s in mine]
            pooled[arm] += list(zip(mine, vals, strict=True))
            rows.append(_evals_row(p, arm, rarm, target, mine, vals))
    for arm, pairs in pooled.items():
        if pairs:
            rows.append(_evals_row("pooled", arm, ref[arm], None, [s for s, _ in pairs],
                                   [v for _, v in pairs]))
    return rows


def _evals_row(problem, arm, rarm, target, mine, vals) -> dict:
    reached = [v for v in vals if math.isfinite(v)]
    return {
        "problem": problem, "arm": arm, "reference": rarm, "target_train_nmse": target,
        "runs": len(mine), "failed": sum(s.get("status") != "ok" for s in mine),
        "reached": len(reached), "median_evals_to_reference": _med(vals),
        "median_evals_reached": _med(reached),
        "median_evals_used": _med(_evals_used(s) for s in mine),
    }


def write_tables(spec: dict, runs: dict, problems: list[str], out: Path, write_csv) -> None:
    """OUT_DIR/evals_to_reference.csv when the spec asks for it; `write_csv` is
    `phase_report._write_csv`."""
    if spec.get("evals_to_reference"):
        rows = evals_to_reference_rows(spec, runs, problems)
        write_csv(out / "evals_to_reference.csv", rows, list(EVALS_FIELDS))
        print(json.dumps({"evals_to_reference_rows": len(rows)}))
