"""Generic phase report: paired Wilcoxon contrasts, Holm, and the standing verdict rule.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase_report.py --stage {1,2} SPEC_JSON OUT_DIR RESULT_DIRS

RESULT_DIRS: comma-separated result directories, Boolean (`benchmarks.gomea.runner`),
regression (`moses_sr_runner.py`) and stand-alone GP-GOMEA (`gpgomea_sr_runner.py`) mixed
freely. A later directory overrides an earlier one for the same (task, arm, seed). Runs of
different runners pair like any two arms: by (task, seed), a regression run's seed being its
split (both regression runners write `task`, `arm`, `seed`, `status` and `test_nmse`).

SPEC_JSON (one per phase, `configs/gomea/<phase>/spec.json`)::

    {"phase": "1u",
     "metrics": {"mux11": "cpu_to_target", "Con": "test_nmse"},
     "primary":     [[problem, name, arm_a, arm_b(, metric)], ...],
     "secondary":   [[...], ...],
     "descriptive": [[...], ...],          # optional
     "pooled_primary": [[name, arm_a, arm_b, metric], ...],   # optional (server round 8)
     "pooled_secondary": [[name, arm_a, arm_b, metric], ...], # optional (server round 8)
     "secondary_verdicts": false,          # optional (server round 8), default true
     "extend": false,                      # optional (server round 9), default true
     "secondary_families": [[name, ...], ...]}  # optional (server round 9, 1bc)

A contrast takes its problem's metric, or the optional fifth element (server round 3,
1ap: program size as the primary metric and test NMSE as the secondary one for the same
problem). Metrics, paired by seed (a regression run's seed is its split); lower is better:

* `cpu_to_target`: CPU seconds to a solution; a failed or unsolved run = infinite;
* `evals_to_target`: evaluations to a solution; a failed or unsolved run = infinite;
* `test_nmse`: test NMSE of the reported program; a failed run = infinite;
* `program_size`: size of the reported program (`_size`; server round 3, 1ap);
  a failed run = infinite;
* `train_test_gap`: test NMSE minus training NMSE of the reported program (`_gap`;
  server round 9); a failed run = infinite;
* `cpu_s`: CPU seconds of the run (`cpu_s`; for a seeded run it includes GP-GOMEA's CPU;
  server round 9); a failed run = infinite;
* `lm_calls`: LM calls of the run (`const_opt.lm_calls`; server round 9); a failed run,
  or a run without that value, = infinite.

Families: Holm over all primary contrasts, and separately over all secondary ones.
Descriptive contrasts take no verdict (raw p only). With `"secondary_verdicts": false` the
secondary contrasts keep their Holm p but take no verdict (`None`).
With `"secondary_families"` (server round 9, 1bc) each inner list of secondary contrast names
is a Holm family of its own, and the other secondary contrasts form one more family. The rows
keep family "secondary" and get the column `holm_family`: "secondary/<names joined by +>" or
"secondary". Each name must be a secondary contrast, in one list only. Without the key the
secondary contrasts are one family, as before.

Pooled primary contrasts (server round 8, `SERVER-ROUND8-PREREGISTRATION.md`): each one pools
the pairs (problem, seed) of all problems in `metrics`. Its statistic is
r = log(metric_A / metric_B) per pair, with a paired Wilcoxon signed-rank test (two-sided) on r.
A failed run (metric infinite) gets the largest rank against it: r = +1e12 when only A failed,
-1e12 when only B failed, and 0 (dropped from the ranks, a tie in the median) when both
failed. Holm covers the pooled contrasts only. Verdict: "A better" if Holm p < ALPHA and
exp(median r) <= 0.9, "A worse" if Holm p < ALPHA and exp(median r) >= 1/0.9, else "tie"; never
"extend". The rows have problem "pooled", family "primary", metric "log_ratio:<metric>",
`median_log_ratio` and `ratio` = exp(median r). A spec with pooled primary contrasts must have
no per-problem primary ones.

Pooled secondary contrasts (`pooled_secondary`, the control arm of round 8 after the population
round): the same pooled test and verdict rule, with their own Holm family (separate from the
pooled primary and the per-problem secondary contrasts), and family "secondary". They never
extend.

Rule (each look at ALPHA = 0.025): Holm p < ALPHA and a median at least 10 % lower is
"A better" (mirror: "A worse"); primary contrasts at stage 1 with ALPHA <= Holm p <
EXTEND_BELOW are "extend"; otherwise "tie". With `"extend": false` (server round 9, 1bc:
a phase with no extension) no primary contrast is "extend"; such a contrast is "tie". At stage 2 the Holm family stays complete and
only contrasts that were "extend" at stage 1 take a stage-2 verdict (see the phase's
pre-registration and `docs/gomea/QUEUE.md`).

Writes OUT_DIR/contrasts.csv, aggregate.csv (per problem and arm: runs, failures, solved,
medians of the metric, evaluations to solve, CPU, CPU per evaluation, demes, program size,
and for seeded arms the best seed's score and the runs whose program descends from a seed,
the evaluations used, the train-test gap, generations per deme and the share of each deme end
reason) and verdict.json (`{"problem:name": verdict}` for primary and secondary contrasts).

The deme columns come from each run's `events.jsonl`, next to its `summary.json`, and count
only demes with evaluations (an "end" event with `ledger.requests > 0`). A deme that starts
when the time is spent ends at once with 0 evaluations; it is not a real deme:

* `median_deme_count`: a regression run counts these demes. A Boolean run keeps the
  runner's own `deme_count`, the scorers with at least one call;
* `median_generations_per_deme`: pooled over the demes of the arm's runs;
* `deme_end_share_<reason>`: share of the arm's demes with that `termination_reason`;
* `median_pop_size`, `min_pop_size`, `max_pop_size` (server round 6, D1): the population size
  of each deme (`pop_size_effective` of its "end" event, the size drawn), pooled over the demes
  of the arm's runs. A fixed N gives N three times; `--gomea-pop-size dynamic` gives its spread.

A run without `events.jsonl` (a stand-alone GP-GOMEA or classic run) gives no deme values.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.gomea.stats import holm, wilcoxon_signed_rank  # noqa: E402

ALPHA = 0.025
EXTEND_BELOW = 0.25
METRICS = ("cpu_to_target", "evals_to_target", "test_nmse", "program_size",
           "train_test_gap", "cpu_s", "lm_calls")


def _load(dirs: str, into: dict) -> None:
    """Later directories override earlier ones for the same task/arm/seed."""
    for d in dirs.split(","):
        if not d:
            continue
        for f in sorted(glob.glob(str(Path(d) / "runs/*/summary.json"))):
            s = json.loads(Path(f).read_text())
            s["_events_path"] = str(Path(f).with_name("events.jsonl"))
            into.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = s


def _is_regression(s: dict) -> bool:
    return "test_nmse" in s


def _value(metric: str, s: dict) -> float:
    if s.get("status") != "ok":
        return math.inf
    if metric == "program_size":
        v = _size(s)
    elif metric == "test_nmse":
        v = s.get("test_nmse")
    elif metric == "train_test_gap":
        v = _gap(s)
    elif metric == "cpu_s":
        v = s.get("cpu_s")
    elif metric == "lm_calls":
        v = (s.get("const_opt") or {}).get("lm_calls")
    else:
        v = s.get(metric) if s.get("success") else None
    return math.inf if v is None else float(v)


def _size(s: dict) -> int | None:
    """Program size: for a MOSES combo program, its "(" plus its atoms. A GP-GOMEA summary
    (`gpgomea_sr_runner.py`, infix output) gives GP-GOMEA's own node count instead
    (`final_program_nodes`, without its linear scaling), which is not the same count."""
    if s.get("final_program_nodes") is not None:
        return int(s["final_program_nodes"])
    p = s.get("final_program")
    return None if p is None else p.count("(") + len(p.replace("(", " ").replace(")", " ").split())


def _contrast(problem, name, a, b, arms, metric) -> dict:
    ra, rb = arms.get(a, {}), arms.get(b, {})
    keys = sorted(set(ra) & set(rb))
    xa = [_value(metric, ra[k]) for k in keys]
    xb = [_value(metric, rb[k]) for k in keys]
    d = [x - y if math.isfinite(x - y) else (0.0 if x == y else (1e12 if x > y else -1e12))
         for x, y in zip(xa, xb)]
    return {
        "problem": problem, "contrast": name, "a": a, "b": b, "metric": metric,
        "pairs": len(keys),
        "a_lower": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
        "b_lower": sum(x > 0 for x in d),
        "median_a": statistics.median(xa) if xa else float("nan"),
        "median_b": statistics.median(xb) if xb else float("nan"),
        "p": wilcoxon_signed_rank(d)[1] if any(d) else 1.0,
    }


BIG = 1e12  # the difference of a pair with one failed run: larger than any finite one


def _log(x: float) -> float:
    return -math.inf if x == 0 else math.log(x)


def _pooled_contrast(name, a, b, runs, problems, metric) -> dict:
    """Server round 8: the pairs (problem, seed) of all problems, r = log(A / B)."""
    xa, xb, r = [], [], []
    for p in problems:
        ra, rb = runs.get(p, {}).get(a, {}), runs.get(p, {}).get(b, {})
        for k in sorted(set(ra) & set(rb)):
            va, vb = _value(metric, ra[k]), _value(metric, rb[k])
            xa.append(va)
            xb.append(vb)
            la, lb = _log(va), _log(vb)
            d = la - lb if math.isfinite(la) and math.isfinite(lb) else (
                0.0 if la == lb else (BIG if la > lb else -BIG))
            r.append(d)
    med = statistics.median(r) if r else float("nan")
    return {
        "problem": "pooled", "contrast": name, "a": a, "b": b,
        "metric": f"log_ratio:{metric}", "pairs": len(r),
        "a_lower": sum(x < 0 for x in r), "ties": sum(x == 0 for x in r),
        "b_lower": sum(x > 0 for x in r),
        "median_a": statistics.median(xa) if xa else float("nan"),
        "median_b": statistics.median(xb) if xb else float("nan"),
        "median_log_ratio": med,
        "ratio": math.exp(med) if abs(med) < 700 else (math.inf if med > 0 else 0.0),
        "p": wilcoxon_signed_rank(r)[1] if any(r) else 1.0,
    }


def _decide_pooled(rows) -> None:
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        m = r["median_log_ratio"]
        if ph < ALPHA and m <= math.log(0.9):
            r["verdict"] = "A better"
        elif ph < ALPHA and m >= math.log(1 / 0.9):
            r["verdict"] = "A worse"
        else:
            r["verdict"] = "tie"


def _decide(rows, stage, can_extend) -> None:
    for r, ph in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = ph
        ma, mb = r["median_a"], r["median_b"]
        if ph < ALPHA and ma <= 0.9 * mb:
            r["verdict"] = "A better"
        elif ph < ALPHA and mb <= 0.9 * ma:
            r["verdict"] = "A worse"
        elif can_extend and stage == 1 and ph < EXTEND_BELOW:
            r["verdict"] = "extend"
        else:
            r["verdict"] = "tie"


def _gap(s: dict) -> float | None:
    """Test NMSE minus training NMSE of the reported program."""
    if s.get("test_nmse") is None or s.get("train_nmse") is None:
        return None
    return s["test_nmse"] - s["train_nmse"]


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _evals(s: dict) -> float | None:
    """Evaluations a run used: Boolean `n_calls`, regression `n_evals_total` (seeded) or
    `n_evals`, GP-GOMEA `evaluations`."""
    if "n_calls" in s:
        return s.get("n_calls")
    for k in ("n_evals_total", "n_evals", "evaluations"):
        if s.get(k) is not None:
            return s[k]
    return None


def _per_eval(s: dict) -> float | None:
    n = _evals(s)
    c = s.get("cpu_s")
    return c / n if c is not None and n else None


def _demes(s: dict) -> list[dict] | None:
    """The "end" events with evaluations of one run; `None` when it has no telemetry."""
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


def _deme_columns(v: list[dict], reg: bool) -> dict:
    per_run = [_demes(s) for s in v]
    pooled = [e for d in per_run if d for e in d]
    row: dict = {"median_generations_per_deme": _med(e.get("generations") for e in pooled)}
    if reg:
        row["median_deme_count"] = _med(None if d is None else len(d) for d in per_run)
    reasons: dict[str, int] = {}
    for e in pooled:
        r = str(e.get("termination_reason"))
        reasons[r] = reasons.get(r, 0) + 1
    for r in sorted(reasons):
        row[f"deme_end_share_{r}"] = reasons[r] / len(pooled)
    # server round 6 (D1): the population size of each deme
    sizes = [e["pop_size_effective"] for e in pooled
             if isinstance(e.get("pop_size_effective"), int)]
    row["median_pop_size"] = _med(sizes)
    row["min_pop_size"] = min(sizes) if sizes else None
    row["max_pop_size"] = max(sizes) if sizes else None
    return row


def _aggregate(problem: str, arm: str, v: list[dict], metric: str) -> dict:
    reg = any(_is_regression(s) for s in v)
    seeded = any(s.get("seeding") is not None or "seed_test_nmse" in s for s in v)
    row = {
        "problem": problem, "arm": arm, "runs": len(v),
        "not_ok": sum(s.get("status") != "ok" for s in v),
        "solved": None if reg else sum(bool(s.get("success")) for s in v),
        "metric": metric,
        "median_metric": _med(_value(metric, s) for s in v),
        "median_evals_to_target": None if reg else _med(_value("evals_to_target", s) for s in v),
        "median_cpu_s": _med(s.get("cpu_s") for s in v),
        "median_cpu_per_eval": _med(_per_eval(s) for s in v),
        "median_deme_count": _med(s.get("deme_count") for s in v),
        "median_size": _med(_size(s) for s in v) if reg else _med(s.get("best_cpx") for s in v),
        "median_seed_test_nmse": None,
        "median_seed_best_raw": None,
        "median_seed_evals": None,
        "final_from_seed": None,
        "median_evals_used": _med(_evals(s) for s in v),
        "median_train_test_gap": _med(_gap(s) for s in v),
    }
    row.update(_deme_columns(v, reg))
    if any("const_opt" in s for s in v):
        # server round 3: constant tuning (moses_sr_runner.const_opt_summary)
        co = [s.get("const_opt") or {} for s in v]
        row.update(
            median_lm_calls=_med(c.get("lm_calls") for c in co),
            median_lm_improved_share=_med(c.get("lm_improved_share") for c in co),
            median_lm_singular_or_failed=_med(
                (c.get("lm_singular", 0) + c.get("lm_failed", 0)) if c else None for c in co
            ),
            median_tuning_evals=_med(c.get("tuning_evals") for c in co),
            median_tuning_share=_med(c.get("tuning_share") for c in co),
            median_mutate_accepted_share=_med(c.get("mutate_accepted_share") for c in co),
            polish_accepted=sum(bool(c.get("polish_accepted")) for c in co),
        )
        if any("lm_guard_rejected" in c for c in co):
            # server round 5: the LM guard (absent for an arm without it)
            row.update(
                median_lm_guard_checked=_med(c.get("lm_guard_checked") for c in co),
                median_lm_guard_rejected=_med(c.get("lm_guard_rejected") for c in co),
                median_lm_guard_rejected_share=_med(c.get("lm_guard_rejected_share") for c in co),
            )
    if seeded:
        row["median_seed_test_nmse"] = _med(s.get("seed_test_nmse") for s in v)
        row["median_seed_best_raw"] = _med((s.get("seeding") or {}).get("seed_best_raw")
                                           for s in v)
        row["median_seed_evals"] = _med(
            s.get("seed_evals", (s.get("seeding") or {}).get("evaluations_used")) for s in v)
        if any("final_lineage" in s for s in v):
            row["final_from_seed"] = sum(
                str(s.get("final_lineage", "")).startswith("seed") for s in v)
        elif any((s.get("seeding") or {}).get("best_lineages") for s in v):
            # Boolean runner: lineage of the best program(s) at the end of the run
            row["final_from_seed"] = sum(
                str(((s.get("seeding") or {}).get("best_lineages") or [""])[0]).startswith("seed")
                for s in v)
    return row


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 5 or argv[0] != "--stage" or argv[1] not in ("1", "2"):
        print(__doc__, file=sys.stderr)
        return 2
    stage, spec, out = int(argv[1]), json.loads(Path(argv[2]).read_text()), Path(argv[3])
    runs: dict = {}
    _load(argv[4], runs)
    metrics = spec["metrics"]
    for m in metrics.values():
        if m not in METRICS:
            print(f"unknown metric {m!r} (choose from {METRICS})", file=sys.stderr)
            return 2
    families = {k: spec.get(k, []) for k in ("primary", "secondary", "descriptive")}
    problems = sorted(set(metrics) | {c[0] for f in families.values() for c in f})
    missing = [p for p in problems if p not in metrics]
    if missing:
        print(f"spec has no metric for {missing}", file=sys.stderr)
        return 2
    agg = [_aggregate(p, arm, list(runs[p][arm].values()), metrics[p])
           for p in problems if p in runs for arm in sorted(runs[p])]
    if not agg:
        print(f"no Phase {spec.get('phase', '?')} runs found in {argv[4]}", file=sys.stderr)
        return 1
    for f in families.values():
        for c in f:
            if len(c) == 5 and c[4] not in METRICS:
                print(f"unknown metric {c[4]!r} in contrast {c}", file=sys.stderr)
                return 2
    pooled = spec.get("pooled_primary", [])
    if pooled and families["primary"]:
        print("spec has both pooled and per-problem primary contrasts", file=sys.stderr)
        return 2
    pooled2 = spec.get("pooled_secondary", [])
    for c in pooled + pooled2:
        if len(c) != 4 or c[3] not in METRICS:
            print(f"bad pooled contrast {c} (want [name, arm_a, arm_b, metric])", file=sys.stderr)
            return 2
    rows = {k: [_contrast(p, n, a, b, runs.get(p, {}), m[0] if m else metrics[p])
                for p, n, a, b, *m in f]
            for k, f in families.items()}
    if pooled:
        rows["primary"] = [_pooled_contrast(n, a, b, runs, list(metrics), m)
                           for n, a, b, m in pooled]
        _decide_pooled(rows["primary"])
    else:
        _decide(rows["primary"], stage, can_extend=spec.get("extend", True) is not False)
    groups = spec.get("secondary_families")
    if groups is None:
        _decide(rows["secondary"], stage, can_extend=False)
    else:
        names = {c[1] for c in families["secondary"]}
        flat = [n for g in groups if isinstance(g, list) for n in g]
        if (not all(isinstance(g, list) and g for g in groups) or len(flat) != len(set(flat))
                or not set(flat) <= names):
            print(f"bad secondary_families {groups} (want lists of distinct secondary "
                  f"contrast names from {sorted(names)})", file=sys.stderr)
            return 2
        label = {n: "secondary/" + "+".join(g) for g in groups for n in g}
        for r in rows["secondary"]:
            r["holm_family"] = label.get(r["contrast"], "secondary")
        for fam in dict.fromkeys(r["holm_family"] for r in rows["secondary"]):
            _decide([r for r in rows["secondary"] if r["holm_family"] == fam], stage,
                    can_extend=False)
    if spec.get("secondary_verdicts", True) is False:
        for r in rows["secondary"]:
            r["verdict"] = None
    if pooled2:
        prows = [_pooled_contrast(n, a, b, runs, list(metrics), m) for n, a, b, m in pooled2]
        _decide_pooled(prows)
        rows["secondary"] += prows
    for r in rows["descriptive"]:
        r["p_holm"], r["verdict"] = None, None
    for k, rs in rows.items():
        for r in rs:
            r["family"] = k
    allrows = rows["primary"] + rows["secondary"] + rows["descriptive"]
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in allrows for k in r},
                    key=lambda k: (k not in ("problem", "contrast", "family"), k))
    _write_csv(out / "contrasts.csv", allrows, fields)
    afields = list(agg[0]) + [k for r in agg for k in r if k not in agg[0]]
    afields = list(dict.fromkeys(afields))
    _write_csv(out / "aggregate.csv", agg, afields)
    verdict = {f"{r['problem']}:{r['contrast']}": r["verdict"]
               for r in rows["primary"] + rows["secondary"]}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    for r in agg + allrows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
