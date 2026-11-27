"""Backlog 15.4, D1b: the report of the Yacht check of ``--gomea-contin-start wide``.

Usage (from the repository root; the script puts the repository on ``sys.path`` itself)::

    .venv/bin/python -I tools/gomea/d1b_report.py RESULTS [RESULTS ...] --out OUT_DIR
        [--task Yac] [--control BASE-TLS] [--treatment TLS-CW] [--reference HC-TLS]

``RESULTS`` is a results directory of ``moses_sr_runner.py`` (``OUTPUT_ROOT/<manifest name>``,
with ``runs/``) or an output root whose subdirectories are such directories (for example
``.d1work/d1b``, with ``d1b-yac`` and ``d1b-yac-hc``). The run lists are
``configs/gomea/d1b/yac.json`` (BASE-TLS, TLS-CW) and ``yac-hc.json`` (HC-TLS); the runner
refuses one list with both ``anytime`` and ``hc_anytime``.

**Per run** (``runs.csv``, from ``summary.json``): the status, the test, clipped test, training
and validation NMSE, the program size (the node count of ``phase_report._nodes`` plus
``run_table.terms_size``: ``n_terms`` when the final model uses the per-term fit; the rule of
rounds 13-17), the CPU time (``cpu_s``), the wall time, the evaluations (``n_evals``), the demes
(see below) and the peak memory of the MOSES process (``ru_maxrss_kb``).

**Contrasts** (``contrasts.csv``, ``ratios.csv``): A against B, paired by split, on the test
NMSE and on the training NMSE. A failed run counts as infinitely bad. The project rule
(``phase_report._pooled_contrast``): the two-sided Wilcoxon signed-rank test on log(A / B)
(``benchmarks.gomea.stats.wilcoxon_signed_rank``, exact for up to 25 pairs) and the ratio
exp(median log(A / B)). The table also has the plain median of the per-split ratios A / B (it
differs from the project ratio only when the two middle ratios differ), the wins (A lower), the
losses (A higher) and the ties.

* Primary: TLS-CW against BASE-TLS. The decision of the rule that was written before the runs
  (``docs/gomea/BACKLOG.md``, 15.4, D1b): **pass** if the test ratio is at most 0.90 and
  p < 0.025; **promising** if the test ratio is at most 0.95 and TLS-CW has the lower training
  NMSE in at least 7 of 10 splits; else **no**. The decision uses the project ratio; the
  summary says when the plain median ratio gives another decision.
* Descriptive: HC-TLS against BASE-TLS (no decision).

**Demes** (``demes.csv`` one row per deme, ``deme_stats.csv`` per arm):

* GOMEA arms, from ``events.jsonl``: an "invocation_start" event starts a deme (``deme_pop_size``,
  ``dim``, ``n_contin_genes``), its "init" event gives ``contin_start`` (absent = ``step``),
  the "generation" events are counted, and its "end" event gives the evaluations of the deme
  (``ledger.requests``, the last ledger), ``generations`` and ``termination_reason``. A deme
  with 0 requests is left out (the rule of ``phase_report._demes``).
* HC arms, from ``hc_progress.jsonl`` (the runner's expansion hook, manifest key
  ``hc_anytime``): one line per expansion; the evaluations of a deme are its ``n_evals`` minus
  that of the line before it. The runner runs MOSES with ``-l WARN``, so ``moses.log`` has no
  per-deme row, and no log gives the number of instances that an HC deme keeps at its end.
  (By the code, HC keeps every evaluated instance in the deme, so that number is the deme's
  evaluations before the merge; this report does not measure it.)
* An HC run with the summary key ``hc_progress_error`` (the hook stopped writing after a write
  error; the RV16 review of round 17, G1): its file has only the first demes, so the run has no
  demes (as a run without the file), and ``summary.txt`` names it in a note.

**Output** (in ``OUT_DIR``): ``runs.csv``, ``contrasts.csv``, ``ratios.csv``, ``demes.csv``,
``deme_stats.csv`` and ``summary.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_report  # noqa: E402
import run_table  # noqa: E402

PASS_RATIO, PASS_P = 0.90, 0.025
PROMISING_RATIO, PROMISING_TRAIN_WINS = 0.95, 7
METRICS = ("test_nmse", "train_nmse")

RUN_FIELDS = (
    "task",
    "arm",
    "split",
    "status",
    "test_nmse",
    "test_nmse_clipped",
    "train_nmse",
    "val_nmse",
    "size",
    "nodes",
    "n_terms",
    "cpu_s",
    "wall_s",
    "evals",
    "demes",
    "max_rss_mb",
    "final_program",
)
CONTRAST_FIELDS = (
    "contrast",
    "kind",
    "a",
    "b",
    "metric",
    "pairs",
    "wins",
    "losses",
    "ties",
    "ratio",
    "median_ratio",
    "p",
    "median_a",
    "median_b",
)
RATIO_FIELDS = ("contrast", "metric", "split", "a_value", "b_value", "ratio")
DEME_FIELDS = (
    "arm",
    "split",
    "deme",
    "source",
    "pop_size",
    "dim",
    "n_contin_genes",
    "contin_start",
    "generations",
    "generation_events",
    "evals",
    "start_evals",
    "termination_reason",
)
STAT_FIELDS = (
    "arm",
    "kind",
    "runs",
    "demes_per_run_median",
    "demes_per_run_min",
    "demes_per_run_max",
    "pop_size_median",
    "pop_size_min",
    "pop_size_max",
    "generations_median",
    "generations_max",
    "evals_per_deme_median",
    "evals_per_deme_max",
    "dim_median",
    "contin_start",
    "source",
)


# ---- loading


def result_dirs(paths: list[Path]) -> list[Path]:
    """Each path with a ``runs/`` directory, or each subdirectory of it that has one."""
    out = []
    for p in paths:
        if (p / "runs").is_dir():
            out.append(p)
            continue
        subs = sorted(d for d in p.iterdir() if (d / "runs").is_dir()) if p.is_dir() else []
        if not subs:
            raise SystemExit(f"{p}: no runs/ directory in it or in its subdirectories")
        out += subs
    return out


def load(paths: list[Path]) -> dict:
    """``{task: {arm: {split: summary}}}`` (``phase_report._load``: a later directory overrides
    an earlier one for the same task, arm and split)."""
    runs: dict = {}
    phase_report._load(",".join(str(d) for d in result_dirs(paths)), runs)
    return runs


def _real(x) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def size(s: dict) -> int | None:
    n = phase_report._nodes(s)
    return None if n is None else n + run_table.terms_size(s)


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # a cut last line of a run that stopped
        if isinstance(ev, dict):
            out.append(ev)
    return out


# ---- demes


def gomea_demes(events: Path) -> list[dict]:
    """The demes with evaluations of one GOMEA run (see the module docstring)."""
    demes, cur, start = [], None, 0
    for ev in _read_jsonl(events):
        t = ev.get("type")
        if t == "invocation_start":
            cur = {
                "source": "events.jsonl",
                "pop_size": ev.get("deme_pop_size"),
                "dim": ev.get("dim"),
                "n_contin_genes": ev.get("n_contin_genes"),
                "contin_start": "step",
                "generation_events": 0,
            }
        elif cur is None:
            continue
        elif t == "init":
            cur["contin_start"] = ev.get("contin_start", "step")
        elif t == "generation":
            cur["generation_events"] += 1
        elif t == "end":
            req = (ev.get("ledger") or {}).get("requests", 0)
            cur.update(
                generations=ev.get("generations"),
                evals=req,
                start_evals=start,
                termination_reason=ev.get("termination_reason"),
            )
            start += req
            if req > 0:
                demes.append(cur)
            cur = None
    return demes


def hc_demes(progress: Path) -> list[dict]:
    """The demes with evaluations of one HC run, from ``hc_progress.jsonl``."""
    demes, prev = [], 0
    for ev in _read_jsonl(progress):
        n = ev.get("n_evals")
        if not isinstance(n, int) or isinstance(n, bool):
            continue
        if n > prev:
            demes.append({"source": "hc_progress.jsonl", "evals": n - prev, "start_evals": prev})
        prev = n
    return demes


def run_demes(s: dict) -> list[dict] | None:
    """The demes of a run, or ``None`` when its source file is missing or cut (a run with
    ``hc_progress_error``)."""
    if s.get("hc_progress_error"):
        return None  # G1: the hook stopped writing, so the file does not have every deme
    rdir = Path(s["_events_path"]).parent
    if (rdir / "hc_progress.jsonl").is_file():
        return hc_demes(rdir / "hc_progress.jsonl")
    if (rdir / "events.jsonl").is_file():
        return gomea_demes(rdir / "events.jsonl")
    return None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _min(xs):
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def deme_stats(arm: str, per_run: dict[int, list[dict] | None]) -> dict:
    lists = [d for d in per_run.values() if d is not None]
    pooled = [x for d in lists for x in d]
    kind = "hc" if pooled and pooled[0]["source"] == "hc_progress.jsonl" else "gomea"
    counts = [len(d) for d in lists]
    starts = sorted({x.get("contin_start") for x in pooled if x.get("contin_start")})
    return {
        "arm": arm,
        "kind": kind,
        "runs": len(lists),
        "demes_per_run_median": _med(counts),
        "demes_per_run_min": _min(counts),
        "demes_per_run_max": _max(counts),
        "pop_size_median": _med(x.get("pop_size") for x in pooled),
        "pop_size_min": _min(x.get("pop_size") for x in pooled),
        "pop_size_max": _max(x.get("pop_size") for x in pooled),
        "generations_median": _med(x.get("generations") for x in pooled),
        "generations_max": _max(x.get("generations") for x in pooled),
        "evals_per_deme_median": _med(x.get("evals") for x in pooled),
        "evals_per_deme_max": _max(x.get("evals") for x in pooled),
        "dim_median": _med(x.get("dim") for x in pooled),
        "contin_start": "/".join(starts),
        "source": pooled[0]["source"] if pooled else "",
    }


# ---- contrasts


def _ratio(a: float, b: float) -> float:
    """A / B with the failed-run rule: inf (A failed), 0 (B failed), 1 (both failed)."""
    if math.isinf(a) and math.isinf(b):
        return 1.0
    if math.isinf(a):
        return math.inf
    if math.isinf(b):
        return 0.0
    if b == 0:
        return 1.0 if a == 0 else math.inf
    return a / b


def contrast(runs: dict, task: str, a: str, b: str, metric: str, kind: str) -> tuple[dict, list]:
    row = phase_report._pooled_contrast(f"{a} vs {b}", a, b, runs, [task], metric)
    ra, rb = runs[task][a], runs[task][b]
    ratios = []
    for split in sorted(set(ra) & set(rb)):
        va = phase_report._value(metric, ra[split])
        vb = phase_report._value(metric, rb[split])
        ratios.append(
            {
                "contrast": f"{a} vs {b}",
                "metric": metric,
                "split": split,
                "a_value": va,
                "b_value": vb,
                "ratio": _ratio(va, vb),
            }
        )
    out = {
        "contrast": f"{a} vs {b}",
        "kind": kind,
        "a": a,
        "b": b,
        "metric": metric,
        "pairs": row["pairs"],
        "wins": row["a_lower"],
        "losses": row["b_lower"],
        "ties": row["ties"],
        "ratio": row["ratio"],
        "median_ratio": statistics.median(r["ratio"] for r in ratios) if ratios else None,
        "p": row["p"],
        "median_a": row["median_a"],
        "median_b": row["median_b"],
    }
    return out, ratios


def decide(test: dict, train: dict, ratio_key: str = "ratio") -> str:
    """The rule written before the runs (see the module docstring)."""
    r = test[ratio_key]
    if r is not None and r <= PASS_RATIO and test["p"] < PASS_P:
        return "pass"
    if r is not None and r <= PROMISING_RATIO and train["wins"] >= PROMISING_TRAIN_WINS:
        return "promising"
    return "no"


# ---- output


def _fmt(v) -> str:
    """A value for ``summary.txt`` (6 significant digits)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _csv(v) -> str:
    """A value for a CSV file (full precision)."""
    return "" if v is None else str(v)


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv(r.get(k)) for k in fields})


def run_row(task: str, arm: str, split: int, s: dict, demes: list[dict] | None) -> dict:
    t = s.get("terms") if isinstance(s.get("terms"), dict) else {}
    rss = s.get("ru_maxrss_kb")
    return {
        "task": task,
        "arm": arm,
        "split": split,
        "status": s.get("status"),
        "test_nmse": _real(s.get("test_nmse")),
        "test_nmse_clipped": _real(s.get("test_nmse_clipped")),
        "train_nmse": _real(s.get("train_nmse")),
        "val_nmse": _real(s.get("val_nmse")),
        "size": size(s),
        "nodes": phase_report._nodes(s),
        "n_terms": t.get("n_terms"),
        "cpu_s": _real(s.get("cpu_s")),
        "wall_s": _real(s.get("wall_s")),
        "evals": phase_report._evals(s),
        "demes": None if demes is None else len(demes),
        "max_rss_mb": None if rss is None else rss / 1024,
        "final_program": s.get("final_program"),
    }


def _line(c: dict) -> str:
    return (
        f"  {c['contrast']:22s} {c['metric']:10s} pairs {c['pairs']:2d}  ratio "
        f"{_fmt(c['ratio']):>9s} (plain median {_fmt(c['median_ratio'])})  "
        f"wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']}  p {_fmt(c['p'])}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="Yac")
    ap.add_argument("--control", default="BASE-TLS")
    ap.add_argument("--treatment", default="TLS-CW")
    ap.add_argument("--reference", default="HC-TLS")
    args = ap.parse_args(argv)
    runs = load(args.results)
    if args.task not in runs:
        raise SystemExit(f"no runs of {args.task} in {[str(p) for p in args.results]}")
    arms = runs[args.task]
    order = [a for a in (args.control, args.treatment, args.reference) if a in arms]
    order += sorted(a for a in arms if a not in order)
    args.out.mkdir(parents=True, exist_ok=True)

    run_rows, deme_rows, stats, notes = [], [], [], []
    for arm in order:
        per_run = {}
        for split in sorted(arms[arm]):
            s = arms[arm][split]
            d = run_demes(s)
            per_run[split] = d
            if s.get("hc_progress_error"):  # G1
                notes.append(f"{arm} split {split}: hc_progress_error (the HC progress hook "
                             f"stopped writing); its demes are left out")
            run_rows.append(run_row(args.task, arm, split, s, d))
            for i, x in enumerate(d or [], 1):
                deme_rows.append({"arm": arm, "split": split, "deme": i, **x})
        stats.append(deme_stats(arm, per_run))
    _write_csv(args.out / "runs.csv", run_rows, RUN_FIELDS)
    _write_csv(args.out / "demes.csv", deme_rows, DEME_FIELDS)
    _write_csv(args.out / "deme_stats.csv", stats, STAT_FIELDS)

    contrasts, ratios = [], []
    pairs = ((args.treatment, "primary"), (args.reference, "descriptive"))
    for a, kind in pairs:
        if a not in arms or args.control not in arms:
            notes.append(f"contrast {a} vs {args.control}: an arm has no runs; left out")
            continue
        for metric in METRICS:
            c, r = contrast(runs, args.task, a, args.control, metric, kind)
            contrasts.append(c)
            ratios += r
    _write_csv(args.out / "contrasts.csv", contrasts, CONTRAST_FIELDS)
    _write_csv(args.out / "ratios.csv", ratios, RATIO_FIELDS)

    lines = [f"D1b report: {args.task}; results {', '.join(str(p) for p in args.results)}", ""]
    lines.append("Runs per arm (median over splits; size = nodes + n_terms with the per-term")
    lines.append("fit; source: summary.json of each run; peak MB = ru_maxrss_kb of the MOSES")
    lines.append("process, the maximum over the splits)")
    for arm in order:
        rows = [r for r in run_rows if r["arm"] == arm]
        ok = sum(r["status"] == "ok" for r in rows)
        lines.append(
            f"  {arm:10s} runs {len(rows)} (ok {ok})"
            f"  test {_fmt(_med(r['test_nmse'] for r in rows))}"
            f"  train {_fmt(_med(r['train_nmse'] for r in rows))}"
            f"  size {_fmt(_med(r['size'] for r in rows))}"
            f"  cpu_s {_fmt(_med(r['cpu_s'] for r in rows))}"
            f"  evals {_fmt(_med(r['evals'] for r in rows))}"
            f"  demes {_fmt(_med(r['demes'] for r in rows))}"
            f"  peak MB {_fmt(_max(r['max_rss_mb'] for r in rows))}"
        )
    lines += [
        "",
        "Contrasts (A vs B, paired by split; ratio = exp(median log(A/B)), the project",
        "rule; p = two-sided Wilcoxon signed-rank on log(A/B); a failed run is infinitely",
        "bad; wins = A lower)",
    ]
    lines += [_line(c) for c in contrasts]
    by = {(c["a"], c["metric"]): c for c in contrasts}
    test, train = by.get((args.treatment, "test_nmse")), by.get((args.treatment, "train_nmse"))
    lines.append("")
    if test and train:
        dec = decide(test, train)
        lines.append(
            f"Decision (pre-written rule, {args.treatment} vs {args.control}): {dec.upper()}  "
            f"[pass: test ratio <= {PASS_RATIO} and p < {PASS_P}; promising: test ratio <= "
            f"{PROMISING_RATIO} and lower training NMSE in >= {PROMISING_TRAIN_WINS} of 10 splits]"
        )
        plain = decide(test, train, "median_ratio")
        if plain != dec:
            lines.append(f"  Note: the plain median of the ratios gives {plain.upper()}.")
        if test["pairs"] != 10:
            lines.append(f"  Note: {test['pairs']} pairs; the rule is written for 10 splits.")
    else:
        lines.append(f"Decision: not computed ({args.treatment} or {args.control} has no runs)")
    lines += [
        "",
        "Demes per arm (GOMEA: events.jsonl, invocation_start/init/generation/end;",
        "HC: hc_progress.jsonl, the evaluations between two expansion lines)",
    ]
    for st in stats:
        lines.append(
            f"  {st['arm']:10s} {st['kind']:5s} runs {st['runs']}  demes/run median "
            f"{_fmt(st['demes_per_run_median'])} ({_fmt(st['demes_per_run_min'])}-"
            f"{_fmt(st['demes_per_run_max'])})  evals/deme median "
            f"{_fmt(st['evals_per_deme_median'])} max {_fmt(st['evals_per_deme_max'])}"
            + (
                f"  pop size median {_fmt(st['pop_size_median'])} ({_fmt(st['pop_size_min'])}-"
                f"{_fmt(st['pop_size_max'])})  generations median {_fmt(st['generations_median'])}"
                f" max {_fmt(st['generations_max'])}  dim median {_fmt(st['dim_median'])}"
                f"  contin_start {st['contin_start'] or '-'}"
                if st["kind"] == "gomea"
                else ""
            )
        )
    lines.append("  HC instances kept at the deme end: not in any log (the runner uses -l WARN);")
    lines.append("  by the code, an HC deme keeps every evaluated instance (= its evaluations).")
    lines += notes
    text = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
