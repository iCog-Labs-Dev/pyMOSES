"""Per-arm descriptive extras next to a phase's registered verdicts (phase_report.py).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase_extras.py RESULT_DIR [RESULT_DIR ...] --out OUT_DIR

RESULT_DIR: a result directory with `runs/*/summary.json`, Boolean (`benchmarks.gomea.runner`)
and regression (`moses_sr_runner.py`) mixed freely. As in phase_report.py, a later directory
overrides an earlier one for the same (task, arm, seed).

Per problem and arm (medians over runs; `PHASE1AC-1AG-PREREGISTRATION.md`, D1 in
`DECISIONS.md`):

* regression: test NMSE (a failed run = infinite, as in phase_report.py), train NMSE,
  test - train and test / train (per run, then the median), final program size in nodes
  (phase_report.py's count), CPU per run;
* Boolean: CPU per run and final complexity (`best_cpx`), for reference;
* GOMEA arms with `runs/*/events.jsonl` (per-deme "end" events, or else "generation"
  events counted per "invocation_start"): demes, median and mean generations per deme, the
  share of demes that start 2 or more generations (pooled over the arm's demes), and the
  lowest and highest per-run share. Regression runs from `moses_sr_runner.py` write no
  events file, so these columns are empty for them;
* if any deme carries a `long` (or `long_deme`) flag on its "invocation_start" or "end"
  event (planned 1af telemetry), extra rows per arm split the generation columns into
  `long` and `normal` demes. Without the flag every deme is `normal` and only the `all` row
  is written.

Writes OUT_DIR/extras.csv (one row per problem x arm x deme group) and OUT_DIR/extras.md,
and prints the Markdown.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_report import _is_regression, _med, _size, _value  # noqa: E402

FIELDS = [
    "problem", "arm", "deme_group", "kind", "runs", "not_ok",
    "median_test_nmse", "median_train_nmse", "median_test_minus_train",
    "median_test_over_train", "median_size", "median_cpu_s",
    "telemetry_runs", "demes", "median_gens_per_deme", "mean_gens_per_deme",
    "share_demes_ge2_gens", "min_run_share_ge2", "max_run_share_ge2",
]


def load(dirs: list[str]) -> dict:
    """{task: {arm: {seed: (summary, run_dir)}}}; later directories override earlier ones."""
    runs: dict = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "runs/*/summary.json"))):
            s = json.loads(Path(f).read_text())
            runs.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = (
                s, Path(f).parent)
    return runs


def _argv(s: dict, run_dir: Path) -> list[str]:
    a = s.get("argv")
    if not a:
        cfg = run_dir / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            a = c.get("argv") or c.get("resolved_argv")
    return [str(x) for x in a] if isinstance(a, list) else str(a or "").split()


def is_gomea(s: dict, run_dir: Path) -> bool:
    a = _argv(s, run_dir)
    return any(x == "-a" and y == "gomea" for x, y in zip(a, a[1:], strict=False)) or (
        run_dir / "events.jsonl").exists()


def _flag(e: dict) -> bool | None:
    for k in ("long", "long_deme"):
        if k in e:
            return bool(e[k])
    return None


def demes(run_dir: Path) -> list[dict] | None:
    """Per-deme `{"gens": int, "long": bool | None}` from events.jsonl, or None if absent.

    Uses the "end" event's `generations` (generations started, the last possibly partial);
    without it, the number of "generation" events since the deme's "invocation_start"."""
    p = run_dir / "events.jsonl"
    if not p.exists():
        return None
    out, cur, n_gen = [], None, 0
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("type")
            if t == "invocation_start":
                cur, n_gen = _flag(e), 0
            elif t == "generation":
                n_gen += 1
            elif t == "end":
                g = e.get("generations")
                lf = _flag(e)
                out.append({"gens": int(g) if g is not None else n_gen,
                            "long": lf if lf is not None else cur})
                cur, n_gen = None, 0
    return out


def _ratio(a, b):
    return a / b if a is not None and b else None


def _diff(a, b):
    return a - b if a is not None and b is not None else None


def _gen_cols(per_run: list[list[dict]]) -> dict:
    pooled = [d["gens"] for r in per_run for d in r]
    shares = [sum(d["gens"] >= 2 for d in r) / len(r) for r in per_run if r]
    return {
        "telemetry_runs": len(per_run),
        "demes": len(pooled),
        "median_gens_per_deme": _med(pooled),
        "mean_gens_per_deme": statistics.fmean(pooled) if pooled else None,
        "share_demes_ge2_gens": (sum(g >= 2 for g in pooled) / len(pooled)) if pooled else None,
        "min_run_share_ge2": min(shares) if shares else None,
        "max_run_share_ge2": max(shares) if shares else None,
    }


def arm_rows(problem: str, arm: str, runs: list[tuple[dict, Path]]) -> list[dict]:
    v = [s for s, _ in runs]
    reg = any(_is_regression(s) for s in v)
    ok = [s for s in v if s.get("status") == "ok"]
    row = {k: None for k in FIELDS}
    row.update(problem=problem, arm=arm, deme_group="all",
               kind="regression" if reg else "boolean", runs=len(v),
               not_ok=len(v) - len(ok), median_cpu_s=_med(s.get("cpu_s") for s in v))
    if reg:
        row.update(
            median_test_nmse=_med(_value("test_nmse", s) for s in v),
            median_train_nmse=_med(s.get("train_nmse") for s in ok),
            median_test_minus_train=_med(_diff(s.get("test_nmse"), s.get("train_nmse"))
                                         for s in ok),
            median_test_over_train=_med(_ratio(s.get("test_nmse"), s.get("train_nmse"))
                                        for s in ok),
            median_size=_med(_size(s) for s in ok),
        )
    else:
        row["median_size"] = _med(s.get("best_cpx") for s in v)
    rows = [row]
    if not any(is_gomea(s, d) for s, d in runs):
        return rows
    per_run = [dm for dm in (demes(d) for _, d in runs) if dm is not None]
    if not per_run:
        return rows
    row.update(_gen_cols(per_run))
    if any(d["long"] is not None for r in per_run for d in r):
        for name, want in (("long", True), ("normal", False)):
            sub = {k: None for k in FIELDS}
            sub.update(problem=problem, arm=arm, deme_group=name, kind=row["kind"])
            sub.update(_gen_cols([[d for d in r if bool(d["long"]) is want]
                                  for r in per_run]))
            rows.append(sub)
    return rows


def build(dirs: list[str]) -> list[dict]:
    runs = load(dirs)
    return [r for p in sorted(runs) for a in sorted(runs[p])
            for r in arm_rows(p, a, list(runs[p][a].values()))]


def _f(x, nd=1, comma=False) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:,.{nd}f}" if comma else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def markdown(rows: list[dict], dirs: list[str]) -> str:
    out = ["# Phase extras", "", "Result directories: " + ", ".join(f"`{d}`" for d in dirs), ""]
    reg = [r for r in rows if r["kind"] == "regression" and r["deme_group"] == "all"]
    if reg:
        out += ["## Regression (medians per arm)", "",
                "| Problem | Arm | Runs | Not ok | Test NMSE | Train NMSE | Test − train "
                "| Test / train | Size (nodes) | CPU per run (s) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['problem']} | {r['arm']} | {r['runs']} | {r['not_ok']} "
                f"| {_f(r['median_test_nmse'])} | {_f(r['median_train_nmse'])} "
                f"| {_f(r['median_test_minus_train'])} | {_f(r['median_test_over_train'], 2)} "
                f"| {_f(r['median_size'])} | {_f(r['median_cpu_s'], 0, True)} |" for r in reg]
        out.append("")
    boo = [r for r in rows if r["kind"] == "boolean" and r["deme_group"] == "all"]
    if boo:
        out += ["## Boolean (medians per arm)", "",
                "| Problem | Arm | Runs | Not ok | CPU per run (s) | Final complexity |",
                "|---|---|---|---|---|---|"]
        out += [f"| {r['problem']} | {r['arm']} | {r['runs']} | {r['not_ok']} "
                f"| {_f(r['median_cpu_s'], 0, True)} | {_f(r['median_size'])} |" for r in boo]
        out.append("")
    gen = [r for r in rows if r["demes"] is not None]
    out += ["## GOMEA generations per deme", ""]
    if gen:
        out += ["| Problem | Arm | Demes | Runs | Median gens / deme | Mean gens / deme "
                "| Demes with ≥ 2 gens | Per-run range |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['problem']} | {r['arm']}"
                f"{'' if r['deme_group'] == 'all' else ' (' + r['deme_group'] + ')'} "
                f"| {r['demes']} | {r['telemetry_runs']} | {_f(r['median_gens_per_deme'])} "
                f"| {_f(r['mean_gens_per_deme'], 2)} | {_pct(r['share_demes_ge2_gens'])} "
                f"| {_pct(r['min_run_share_ge2'])} – {_pct(r['max_run_share_ge2'])} |"
                for r in gen]
        out.append("")
        if not any(r["deme_group"] != "all" for r in gen):
            out += ["No deme carries a `long` flag: all demes are treated as normal.", ""]
    none = [f"{r['problem']}:{r['arm']}" for r in rows
            if r["deme_group"] == "all" and r["demes"] is None and r["kind"] == "regression"]
    if none or not gen:
        out += ["No per-generation telemetry (no `events.jsonl`) for: "
                + (", ".join(none) if none else "any arm") + ".", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", metavar="RESULT_DIR")
    ap.add_argument("--out", required=True, metavar="OUT_DIR")
    a = ap.parse_args(argv)
    rows = build(a.dirs)
    if not rows:
        print(f"no runs found in {a.dirs}", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "extras.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    md = markdown(rows, a.dirs)
    (out / "extras.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
