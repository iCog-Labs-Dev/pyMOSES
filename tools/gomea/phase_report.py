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
     "descriptive": [[...], ...]}          # optional

A contrast takes its problem's metric, or the optional fifth element (server round 3,
1ap: program size as the primary metric and test NMSE as the secondary one for the same
problem). Metrics, paired by seed (a regression run's seed is its split); lower is better:

* `cpu_to_target`: CPU seconds to a solution; a failed or unsolved run = infinite;
* `evals_to_target`: evaluations to a solution; a failed or unsolved run = infinite;
* `test_nmse`: test NMSE of the reported program; a failed run = infinite;
* `program_size`: size of the reported program (`_size`; server round 3, 1ap);
  a failed run = infinite.

Families: Holm over all primary contrasts, and separately over all secondary ones.
Descriptive contrasts take no verdict (raw p only).

Rule (each look at ALPHA = 0.025): Holm p < ALPHA and a median at least 10 % lower is
"A better" (mirror: "A worse"); primary contrasts at stage 1 with ALPHA <= Holm p <
EXTEND_BELOW are "extend"; otherwise "tie". At stage 2 the Holm family stays complete and
only contrasts that were "extend" at stage 1 take a stage-2 verdict (see the phase's
pre-registration and `docs/gomea/QUEUE.md`).

Writes OUT_DIR/contrasts.csv, aggregate.csv (per problem and arm: runs, failures, solved,
medians of the metric, evaluations to solve, CPU, CPU per evaluation, demes, program size,
and for seeded arms the best seed's score and the runs whose program descends from a seed,
and the evaluations used) and verdict.json (`{"problem:name": verdict}` for primary and secondary contrasts).
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
METRICS = ("cpu_to_target", "evals_to_target", "test_nmse", "program_size")


def _load(dirs: str, into: dict) -> None:
    """Later directories override earlier ones for the same task/arm/seed."""
    for d in dirs.split(","):
        if not d:
            continue
        for f in sorted(glob.glob(str(Path(d) / "runs/*/summary.json"))):
            s = json.loads(Path(f).read_text())
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
    }
    if any("const_opt" in s for s in v):
        # server round 3: constant tuning (moses_sr_runner.const_opt_summary)
        co = [s.get("const_opt") or {} for s in v]
        row.update(
            median_train_test_gap=_med(_gap(s) for s in v),
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
    rows = {k: [_contrast(p, n, a, b, runs.get(p, {}), m[0] if m else metrics[p])
                for p, n, a, b, *m in f]
            for k, f in families.items()}
    _decide(rows["primary"], stage, can_extend=True)
    _decide(rows["secondary"], stage, can_extend=False)
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
