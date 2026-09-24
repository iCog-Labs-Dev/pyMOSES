"""Cross-arm statistical analysis for a GOMEA benchmark result directory.

CLI: `python -m benchmarks.gomea.analyze RESULT_DIR --baseline HC
--contrasts "LT-B:HC,LT-B:LT-MI,LT-B:UNI" [--budget N]`

`RESULT_DIR` is one `<output_root>/<phase>/<name>-<hash>` directory, exactly
as `runner.py` lays it out (`manifest.json` plus `runs/<run_id>/
summary.json`). Writes, under `RESULT_DIR/analysis/`:

  paired_results.csv    -- one row per task x seed, each arm's final
                            best_raw / success / evals_to_target / wall_s.
  aggregate_results.csv -- one row per task x arm: n, successes, success
                            rate + Wilson 95% CI, median/IQR of final error
                            (= -best_raw), median/IQR of wall_s and n_calls,
                            and ERT (right-censored expected running cost).
  contrasts.csv          -- one row per task x contrast ("A:B" in
                            --contrasts): paired n, wins/ties/losses on
                            final error, two-sided Wilcoxon p on final
                            error and on evals_to_target (unsuccessful runs
                            right-censored at budget+1), Holm-adjusted p
                            across every contrast x metric WITHIN a task,
                            and a bootstrap 95% CI of the paired median
                            difference for both metrics.
  failures.md             -- every error/timeout run.
  figures/*.svg            -- per task: success probability vs evaluations
                            (empirical CDF of evals_to_target, one line per
                            arm) and a box plot of final error per arm.

The evaluation budget used for right-censoring is `--budget` if given,
else parsed out of the manifest's own `budget.argv` (`manifest.
budget_evals`); when neither is available, censored runs are dropped from
the affected computations rather than silently treated as zero cost (see
`_task_budget`'s docstring).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from benchmarks.gomea import svgplot
from benchmarks.gomea.manifest import budget_evals
from benchmarks.gomea.stats import (
    bootstrap_median_ci,
    ert,
    holm,
    median_iqr,
    wilcoxon_signed_rank,
    wilson_ci,
)

__all__ = ["main", "load_summaries"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_summaries(result_dir: str | Path) -> list[dict]:
    runs_dir = Path(result_dir) / "runs"
    summaries: list[dict] = []
    if not runs_dir.is_dir():
        return summaries
    for run_dir in sorted(runs_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            with open(summary_path) as f:
                summaries.append(json.load(f))
    return summaries


def _index(summaries: list[dict]) -> dict[str, dict[str, dict]]:
    """task -> arm -> seed -> summary."""
    idx: dict[str, dict[str, dict]] = {}
    for s in summaries:
        idx.setdefault(s["task"], {}).setdefault(s["arm"], {})[s["seed"]] = s
    return idx


def _task_budget(manifest: dict, budget_override: float | None) -> float | None:
    """The single numeric evaluation budget/cap used for right-censoring.
    `--budget` overrides everything; otherwise parsed from the manifest's
    own `budget.argv` (`-m`/`--max-evals`). `None` when neither is
    available -- callers must then treat a censored run's cost as unknown
    (excluded) rather than fabricate a value."""
    if budget_override is not None:
        return budget_override
    return budget_evals(manifest["budget"])


def _final_error(summary: dict | None) -> float | None:
    """Lower-is-better error: -best_raw. `None` when the run has no score
    (e.g. it errored before any evaluation)."""
    if summary is None or summary.get("best_raw") is None:
        return None
    return -float(summary["best_raw"])


def _evals_for_ert(summary: dict | None, budget: float | None) -> float | None:
    """Right-censored evaluation cost for the ERT aggregate: the true
    evals-to-target on success, else the budget/cap itself (timeouts and
    errors are counted, at the cap -- 03_EXPERIMENT_PROTOCOL.md Sec 8)."""
    if summary is not None and summary.get("success"):
        return summary.get("evals_to_target")
    return budget


def _evals_for_contrast(summary: dict | None, censor_value: float | None) -> float | None:
    """Right-censored evaluation cost for the PAIRED Wilcoxon test on
    evals_to_target: the true evals-to-target on success, else the
    rank-worst sentinel `budget + 1` (guaranteed worse than any possible
    success, so a censored run always loses/ties any successful one in the
    paired comparison -- documented per the brief's own "assigned
    rank-worst value budget+1 (document)")."""
    if summary is not None and summary.get("success"):
        return summary.get("evals_to_target")
    return censor_value


# ---------------------------------------------------------------------------
# paired_results.csv
# ---------------------------------------------------------------------------


def write_paired_results_csv(idx: dict, arm_names: list[str], path: Path) -> None:
    fieldnames = ["task", "seed"] + [
        f"{arm}_{field}"
        for arm in arm_names
        for field in ("best_raw", "success", "evals_to_target", "wall_s")
    ]
    rows = []
    for task in sorted(idx):
        seeds = sorted({seed for arm in idx[task].values() for seed in arm})
        for seed in seeds:
            row = {"task": task, "seed": seed}
            for arm in arm_names:
                s = idx[task].get(arm, {}).get(seed)
                row[f"{arm}_best_raw"] = "" if s is None else s.get("best_raw", "")
                row[f"{arm}_success"] = "" if s is None else s.get("success", "")
                row[f"{arm}_evals_to_target"] = (
                    "" if s is None else (s.get("evals_to_target") or "")
                )
                row[f"{arm}_wall_s"] = "" if s is None else s.get("wall_s", "")
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# aggregate_results.csv
# ---------------------------------------------------------------------------


def write_aggregate_results_csv(
    idx: dict, manifest: dict, budget_override: float | None, path: Path
) -> None:
    fieldnames = [
        "task",
        "arm",
        "n",
        "successes",
        "success_rate",
        "wilson_lo",
        "wilson_hi",
        "error_median",
        "error_q1",
        "error_q3",
        "wall_s_median",
        "wall_s_q1",
        "wall_s_q3",
        "n_calls_median",
        "n_calls_q1",
        "n_calls_q3",
        "ert",
    ]
    rows = []
    for task in sorted(idx):
        budget = _task_budget(manifest, budget_override)
        for arm in sorted(idx[task]):
            summaries = list(idx[task][arm].values())
            n = len(summaries)
            successes = [bool(s.get("success")) for s in summaries]
            k = sum(successes)
            wilson_lo, wilson_hi = wilson_ci(k, n)
            errors = [e for e in (_final_error(s) for s in summaries) if e is not None]
            walls = [s["wall_s"] for s in summaries if s.get("wall_s") is not None]
            ncalls = [s["n_calls"] for s in summaries if s.get("n_calls") is not None]
            err_med, err_q1, err_q3 = median_iqr(errors)
            wall_med, wall_q1, wall_q3 = median_iqr(walls)
            n_med, n_q1, n_q3 = median_iqr(ncalls)
            costs = [c for c in (_evals_for_ert(s, budget) for s in summaries) if c is not None]
            ert_val = ert(costs, successes) if costs else float("inf")
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "n": n,
                    "successes": k,
                    "success_rate": (k / n) if n else "",
                    "wilson_lo": wilson_lo,
                    "wilson_hi": wilson_hi,
                    "error_median": err_med,
                    "error_q1": err_q1,
                    "error_q3": err_q3,
                    "wall_s_median": wall_med,
                    "wall_s_q1": wall_q1,
                    "wall_s_q3": wall_q3,
                    "n_calls_median": n_med,
                    "n_calls_q1": n_q1,
                    "n_calls_q3": n_q3,
                    "ert": "inf" if ert_val == float("inf") else ert_val,
                }
            )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# contrasts.csv
# ---------------------------------------------------------------------------


def _paired_values(idx: dict, task: str, arm_a: str, arm_b: str, extractor) -> list[tuple]:
    seeds = sorted(set(idx.get(task, {}).get(arm_a, {})) & set(idx.get(task, {}).get(arm_b, {})))
    pairs = []
    for seed in seeds:
        a = extractor(idx[task][arm_a][seed])
        b = extractor(idx[task][arm_b][seed])
        if a is None or b is None:
            continue
        pairs.append((seed, a, b))
    return pairs


def write_contrasts_csv(
    idx: dict, manifest: dict, contrasts: list[str], budget_override: float | None, path: Path
) -> None:
    fieldnames = [
        "task",
        "contrast",
        "n_paired",
        "wins",
        "ties",
        "losses",
        "p_error",
        "p_error_holm",
        "p_evals_to_target",
        "p_evals_to_target_holm",
        "boot_ci_error_lo",
        "boot_ci_error_hi",
        "boot_ci_evals_lo",
        "boot_ci_evals_hi",
        "censor_value_evals",
    ]
    all_rows: list[dict] = []
    for task in sorted(idx):
        budget = _task_budget(manifest, budget_override)
        censor_value = (budget + 1) if budget is not None else None

        task_start = len(all_rows)
        task_pvals: list[float] = []
        for contrast in contrasts:
            arm_a, arm_b = contrast.split(":")

            err_pairs = _paired_values(idx, task, arm_a, arm_b, _final_error)
            err_diffs = [a - b for _seed, a, b in err_pairs]
            wins = sum(1 for d in err_diffs if d < 0)  # arm_a's error lower = arm_a better
            losses = sum(1 for d in err_diffs if d > 0)
            ties = len(err_diffs) - wins - losses
            _, p_err = wilcoxon_signed_rank(err_diffs)
            ci_err = bootstrap_median_ci(err_diffs) if err_diffs else (float("nan"), float("nan"))

            evals_pairs = _paired_values(
                idx,
                task,
                arm_a,
                arm_b,
                lambda s, cv=censor_value: _evals_for_contrast(s, cv),
            )
            evals_diffs = [a - b for _seed, a, b in evals_pairs]
            _, p_evals = wilcoxon_signed_rank(evals_diffs)
            ci_evals = (
                bootstrap_median_ci(evals_diffs) if evals_diffs else (float("nan"), float("nan"))
            )

            all_rows.append(
                {
                    "task": task,
                    "contrast": contrast,
                    "n_paired": len(err_pairs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "p_error": p_err,
                    "p_evals_to_target": p_evals,
                    "boot_ci_error_lo": ci_err[0],
                    "boot_ci_error_hi": ci_err[1],
                    "boot_ci_evals_lo": ci_evals[0],
                    "boot_ci_evals_hi": ci_evals[1],
                    "censor_value_evals": censor_value if censor_value is not None else "",
                }
            )
            task_pvals.append(p_err)
            task_pvals.append(p_evals)

        # Holm-adjusted across every contrast x metric WITHIN this task.
        holm_adj = holm(task_pvals)
        for i, row in enumerate(all_rows[task_start:]):
            row["p_error_holm"] = holm_adj[2 * i]
            row["p_evals_to_target_holm"] = holm_adj[2 * i + 1]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


# ---------------------------------------------------------------------------
# failures.md
# ---------------------------------------------------------------------------


def write_failures_md(summaries: list[dict], path: Path) -> None:
    bad = [s for s in summaries if s.get("status") != "ok"]
    lines = ["# Failures\n", "\n"]
    if not bad:
        lines.append("No error or timeout runs.\n")
    else:
        lines.append("| run_id | task | arm | seed | status | error |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for s in sorted(bad, key=lambda s: s["run_id"]):
            err = ""
            if s.get("error"):
                err = s["error"].strip().splitlines()[-1]
            err = err.replace("|", "\\|")
            lines.append(
                f"| {s['run_id']} | {s['task']} | {s['arm']} | {s['seed']} | "
                f"{s['status']} | {err} |\n"
            )
    Path(path).write_text("".join(lines))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def write_figures(idx: dict, manifest: dict, budget_override: float | None, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in sorted(idx):
        series = {}
        for arm in sorted(idx[task]):
            summaries = list(idx[task][arm].values())
            n = len(summaries)
            hits = sorted(
                s["evals_to_target"]
                for s in summaries
                if s.get("success") and s.get("evals_to_target") is not None
            )
            pts = [(0.0, 0.0)]
            for i, x in enumerate(hits, start=1):
                pts.append((x, i / n))
            series[arm] = pts
        svgplot.line_chart(
            series,
            f"{task}: success probability vs evaluations",
            "evaluations",
            "P(success)",
            out_dir / f"{task}_success_cdf.svg",
        )

        groups = {}
        for arm in sorted(idx[task]):
            errs = [e for e in (_final_error(s) for s in idx[task][arm].values()) if e is not None]
            groups[arm] = errs
        svgplot.box_chart(
            groups,
            f"{task}: final error by arm",
            "error (-best_raw)",
            out_dir / f"{task}_error_box.svg",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.gomea.analyze")
    parser.add_argument("result_dir")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--contrasts", required=True)
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args(argv)

    result_dir = Path(args.result_dir)
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"{manifest_path} not found", file=sys.stderr)
        return 1
    with open(manifest_path) as f:
        manifest = json.load(f)

    summaries = load_summaries(result_dir)
    idx = _index(summaries)
    arm_names = list(manifest.get("arms", {}).keys()) or sorted({s["arm"] for s in summaries})
    if args.baseline not in arm_names:
        print(f"warning: --baseline {args.baseline!r} is not one of {arm_names}", file=sys.stderr)
    contrasts = [c.strip() for c in args.contrasts.split(",") if c.strip()]

    analysis_dir = result_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = analysis_dir / "figures"

    write_paired_results_csv(idx, arm_names, analysis_dir / "paired_results.csv")
    write_aggregate_results_csv(idx, manifest, args.budget, analysis_dir / "aggregate_results.csv")
    write_contrasts_csv(idx, manifest, contrasts, args.budget, analysis_dir / "contrasts.csv")
    write_failures_md(summaries, analysis_dir / "failures.md")
    write_figures(idx, manifest, args.budget, figures_dir)

    print(f"wrote analysis to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
