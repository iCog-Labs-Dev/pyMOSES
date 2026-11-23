"""Server round 16 (the budget round, phase 1bk): the descriptive tables of
`SERVER-ROUND16-PREREGISTRATION.md` ("Contrasts", "Descriptive tables"). `phase_report.py`
calls these functions only when a spec has the new keys, so the report of every earlier phase
does not change. The tables read the summary fields of `moses_sr_runner.py` (server round 16:
`anytime`, `demes`, `rv`). They give no verdicts.

**The spec key `"anytime_ratios"`**: `{arm: reference arm}` (round 16: each arm to BASE-TLS of
the same budget). It writes OUT_DIR/anytime_ratios.csv (`ANYTIME_FIELDS`): for each problem of
`metrics`, then for all problems together (problem "pooled"), each arm and each checkpoint
`frac` of `ANYTIME_FRACTIONS`, the pairs (problem, seed) where both runs are ok and both have a
value `best_train_nmse` at that checkpoint (the `anytime` record of the summary). `missing`
counts the other pairs (a failed run, or no value). With the trace of `--gomea-trace 1` (the RV16
review, finding 1; the summary's `anytime_source` is "trace") the value is the exact best-so-far
training NMSE, and it is missing only before the first evaluation of the run. Without the trace
(`anytime_source` "generations") the value is missing when the GOMEA log has no record before
the checkpoint. The table reads only `best_train_nmse` and `evals`, so it is the same code for
both sources. `median_ratio` is the median over the pairs of arm / reference (a reference value
0 gives 1 when the arm value is also 0, else infinity); `arm_lower`, `reference_lower` and
`ties` count the pairs; `median_arm` and `median_reference` are the medians of the two values
over the pairs; `evals` is the median checkpoint of the arm's runs (in evaluations).

**The spec key `"deme_table": true`**: OUT_DIR/demes.csv (`DEME_FIELDS`), for each problem and
arm, then for each arm over all problems: the runs, the runs with a `demes` record, the median,
minimum and maximum number of demes, the runs with one deme, and over the runs with a second
deme the median, minimum and maximum run evaluation at which it starts, and the median of that
start as a share of the run's evaluations (`second_start_evals / evals_total`).

**The spec key `"rv_table": true`**: OUT_DIR/rv.csv (`RV_FIELDS`), for each problem and arm that
has runs with an `rv` record, then for each such arm over all problems: the runs, the runs with
an `rv` record, the median `rv_evals`, the median and the maximum `rv_share`, the median number
of RV generations, the sums of the RV counters over the runs (`RV_SUMS`), and the shares of the
RV evaluations that improved (`improving_share`) and that were kept by p_accept
(`kept_p_share`).

NOT A PORT: server round 16 (the budget round).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ANYTIME_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)  # moses_sr_runner.ANYTIME_FRACTIONS
ANYTIME_FIELDS = ("problem", "arm", "reference", "frac", "evals", "pairs", "missing",
                  "median_ratio", "arm_lower", "reference_lower", "ties", "median_arm",
                  "median_reference")
DEME_FIELDS = ("problem", "arm", "runs", "with_record", "median_demes", "min_demes",
               "max_demes", "one_deme_runs", "median_second_start_evals",
               "min_second_start_evals", "max_second_start_evals", "median_second_start_share")
RV_SUMS = ("rv_evals", "improving", "kept_p", "rejected", "ams1_evals", "ams1_improving",
           "ams1_kept_p", "ams2_evals", "ams2_improving", "ams2_kept_p", "ams1_rounds",
           "ams2_rounds", "fi_triggers", "fi_evals", "fi_improved", "skip_struct", "skip_cache",
           "skip_dup", "skipped_p_rv", "skipped_gom_improved")
RV_FIELDS = ("problem", "arm", "runs", "with_rv", "median_rv_evals", "median_rv_share",
             "max_rv_share", "median_rv_generations", *RV_SUMS, "improving_share",
             "kept_p_share")


def check_spec(spec: dict) -> str | None:
    """An error message for a malformed `"anytime_ratios"`, `"deme_table"` or `"rv_table"`,
    else `None`. A spec without the keys gives `None`."""
    ref = spec.get("anytime_ratios")
    if ref is not None and (not isinstance(ref, dict) or not ref or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in ref.items())):
        return f"bad anytime_ratios {ref!r} (want {{arm: reference arm}})"
    for k in ("deme_table", "rv_table"):
        if k in spec and not isinstance(spec[k], bool):
            return f"spec key {k!r} must be true or false, got {spec[k]!r}"
    return None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _checkpoint(s: dict, frac: float) -> tuple[float | None, int | None]:
    """(best_train_nmse, evals) of the `anytime` record of `frac`, or (None, None)."""
    for p in s.get("anytime") or []:
        if isinstance(p, dict) and p.get("frac") == frac:
            v = p.get("best_train_nmse")
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            return (float(v) if ok else None), p.get("evals")
    return None, None


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return 1.0 if a == 0 else math.inf
    return a / b


def anytime_rows(spec: dict, runs: dict, problems: list[str]) -> list[dict]:
    """The rows of anytime_ratios.csv (see the module docstring)."""
    rows = []
    units = [([p], p) for p in problems] + [(problems, "pooled")]
    for arm, ref in spec["anytime_ratios"].items():
        for probs, label in units:
            for frac in ANYTIME_FRACTIONS:
                ratios, va, vb, evals, missing = [], [], [], [], 0
                for p in probs:
                    ra, rb = runs.get(p, {}).get(arm, {}), runs.get(p, {}).get(ref, {})
                    for k in sorted(set(ra) & set(rb)):
                        a, ea = _checkpoint(ra[k], frac)
                        b, _ = _checkpoint(rb[k], frac)
                        if ea is not None:
                            evals.append(ea)
                        ok = ra[k].get("status") == "ok" and rb[k].get("status") == "ok"
                        if not ok or a is None or b is None:
                            missing += 1
                            continue
                        ratios.append(_ratio(a, b))
                        va.append(a)
                        vb.append(b)
                if not ratios and not missing:
                    continue
                rows.append({
                    "problem": label, "arm": arm, "reference": ref, "frac": frac,
                    "evals": _med(evals), "pairs": len(ratios), "missing": missing,
                    "median_ratio": _med(ratios),
                    "arm_lower": sum(r < 1 for r in ratios),
                    "reference_lower": sum(r > 1 for r in ratios),
                    "ties": sum(r == 1 for r in ratios),
                    "median_arm": _med(va), "median_reference": _med(vb),
                })
    return rows


def _deme_row(problem: str, arm: str, v: list[dict]) -> dict:
    recs = [s["demes"] for s in v if isinstance(s.get("demes"), dict)]
    counts = [r.get("count") for r in recs if isinstance(r.get("count"), int)]
    second = [r for r in recs if isinstance(r.get("second_start_evals"), int)]
    starts = [r["second_start_evals"] for r in second]
    shares = [r["second_start_evals"] / r["evals_total"] for r in second
              if isinstance(r.get("evals_total"), int) and r["evals_total"] > 0]
    return {
        "problem": problem, "arm": arm, "runs": len(v), "with_record": len(recs),
        "median_demes": _med(counts), "min_demes": min(counts) if counts else None,
        "max_demes": max(counts) if counts else None,
        "one_deme_runs": sum(c == 1 for c in counts),
        "median_second_start_evals": _med(starts),
        "min_second_start_evals": min(starts) if starts else None,
        "max_second_start_evals": max(starts) if starts else None,
        "median_second_start_share": _med(shares),
    }


def _rv_row(problem: str, arm: str, v: list[dict]) -> dict:
    recs = [s["rv"] for s in v if isinstance(s.get("rv"), dict)]
    row = {
        "problem": problem, "arm": arm, "runs": len(v), "with_rv": len(recs),
        "median_rv_evals": _med(r.get("rv_evals") for r in recs),
        "median_rv_share": _med(r.get("rv_share") for r in recs),
        "max_rv_share": max((r["rv_share"] for r in recs
                             if isinstance(r.get("rv_share"), (int, float))), default=None),
        "median_rv_generations": _med(r.get("generations") for r in recs),
    }
    row.update({k: sum(int(r.get(k) or 0) for r in recs) for k in RV_SUMS})
    n = row["rv_evals"]
    row["improving_share"] = row["improving"] / n if n else None
    row["kept_p_share"] = row["kept_p"] / n if n else None
    return row


def _by_arm(runs: dict, problems: list[str]):
    """(problem, arm, summaries) for each problem and arm, then ("pooled", arm, summaries)."""
    pooled: dict[str, list[dict]] = {}
    for p in problems:
        for arm in sorted(runs.get(p, {})):
            v = list(runs[p][arm].values())
            pooled.setdefault(arm, []).extend(v)
            yield p, arm, v
    for arm in sorted(pooled):
        yield "pooled", arm, pooled[arm]


def deme_rows(runs: dict, problems: list[str]) -> list[dict]:
    return [_deme_row(p, arm, v) for p, arm, v in _by_arm(runs, problems)]


def rv_rows(runs: dict, problems: list[str]) -> list[dict]:
    return [_rv_row(p, arm, v) for p, arm, v in _by_arm(runs, problems)
            if any(isinstance(s.get("rv"), dict) for s in v)]


def write_tables(spec: dict, runs: dict, problems: list[str], out: Path, write_csv) -> None:
    """The tables that the spec asks for; `write_csv` is `phase_report._write_csv`."""
    if spec.get("anytime_ratios"):
        rows = anytime_rows(spec, runs, problems)
        write_csv(out / "anytime_ratios.csv", rows, list(ANYTIME_FIELDS))
        print(json.dumps({"anytime_ratio_rows": len(rows)}))
    if spec.get("deme_table"):
        rows = deme_rows(runs, problems)
        write_csv(out / "demes.csv", rows, list(DEME_FIELDS))
        print(json.dumps({"deme_rows": len(rows)}))
    if spec.get("rv_table"):
        rows = rv_rows(runs, problems)
        write_csv(out / "rv.csv", rows, list(RV_FIELDS))
        print(json.dumps({"rv_rows": len(rows)}))
