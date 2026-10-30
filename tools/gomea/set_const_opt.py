"""Choose the constant-tuning arm for 1ao and 1ar from the 1an results (server round 3).

Usage (from the repository root, after `tools/gomea/phase_run.sh phase1an` and before
`tools/gomea/phase_run.sh phase1ao`; the round-3 plan runs it as the `after_pilot` of
1ao, whose pilot is 1an)::

    .venv/bin/python tools/gomea/set_const_opt.py [--results DIR] [--configs DIR]
        [--source PHASE] [--phases P1,P2,...] [--stage1-only] [--dry-run]

`SERVER-ROUND3-PREREGISTRATION.md`, 1an: "Rule for 1ao: the tuning arm (LM or MUT) with
the lower median test NMSE, pooled over the three datasets' medians, goes to 1ao. A script
writes the choice and logs it." 1ar uses the same choice. This script

1. reads the source phase's runs: every regression manifest `CONFIGS/SOURCE/sr*.json`
   (stage 1 and, unless `--stage1-only`, the extensions `sr-ext-*.json`, so the final
   results), whose runs are `RESULTS/SOURCE/<name>/runs/<dataset>__<arm>__s<split>/
   summary.json` (moses_sr_runner.py); a run whose status is not "ok" counts as an
   infinite test NMSE, as in the phase report;
2. per dataset of the stage-1 manifest, takes the median test NMSE of arm LM
   (`GOMEA-LS-LM`, `--const-opt lm`) and of arm MUT (`GOMEA-LS-MUT`, `--const-opt
   mutate`), then per arm the median of those per-dataset medians (the pooled median);
3. picks the arm with the lower pooled median (a tie, which the rule does not cover,
   goes to `mutate`, the cheaper method);
4. writes the chosen mode after `--const-opt` in every arm whose `--const-opt` value is
   the placeholder `SET-BY-set_const_opt.py` or a mode (the GOMEA-LS-TUNE arms), in every
   regression manifest (`sr*.json`) of each target phase, and logs the numbers and the
   choice to `RESULTS/SOURCE/const-opt-choice.json`.

It exits 1, changing nothing, unless every expected run (each split of each manifest read,
for both arms) has a summary.json, or if a target phase has no TUNE arm. `--dry-run` prints
what it would set and writes nothing. RESULTS defaults to `../gomea-results`, CONFIGS to
`configs/gomea`, SOURCE to `phase1an`, the phases to `phase1ao,phase1ar`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FLAG = "--const-opt"
PLACEHOLDER = "SET-BY-set_const_opt.py"
ARMS = {"lm": "GOMEA-LS-LM", "mutate": "GOMEA-LS-MUT"}
TIE = "mutate"
PHASES = "phase1ao,phase1ar"


def _nmse(summary: dict) -> float:
    if summary.get("status") != "ok" or summary.get("test_nmse") is None:
        return math.inf
    return float(summary["test_nmse"])


def collect(results: Path, source_dir: Path, stage1_only: bool) -> tuple[dict, list[str]]:
    """Per mode and dataset the list of test NMSEs, and the errors (missing runs)."""
    out: dict[str, dict[str, list[float]]] = {m: {} for m in ARMS}
    errors = []
    files = sorted(source_dir.glob("sr*.json"))
    if stage1_only:
        files = [f for f in files if not f.name.startswith("sr-ext-")]
    stage1 = [f for f in files if not f.name.startswith("sr-ext-")]
    if not stage1:
        errors.append(f"no stage-1 regression manifest (sr*.json) in {source_dir}")
    datasets = []
    for f in stage1:
        datasets += json.loads(f.read_text())["datasets"]
    for f in files:
        m = json.loads(f.read_text())
        for mode, arm in ARMS.items():
            if arm not in m["arms"]:
                if f in stage1:
                    errors.append(f"{f}: no arm {arm}")
                continue
            for ds in m["datasets"]:
                for split in m["splits"]:
                    p = (results / source_dir.name / m["name"] / "runs"
                         / f"{ds}__{arm}__s{split}" / "summary.json")
                    if not p.exists():
                        errors.append(f"missing run {p}")
                        continue
                    out[mode].setdefault(ds, []).append(_nmse(json.loads(p.read_text())))
    for mode in ARMS:
        for ds in datasets:
            if ds not in out[mode]:
                errors.append(f"no {ARMS[mode]} runs for {ds}")
    return {m: {ds: out[m][ds] for ds in datasets if ds in out[m]} for m in ARMS}, errors


def choose(values: dict) -> tuple[str, dict]:
    """The mode with the lower pooled median, and the medians behind it."""
    per_ds = {m: {ds: statistics.median(v) for ds, v in d.items()} for m, d in values.items()}
    pooled = {m: statistics.median(list(d.values())) for m, d in per_ds.items()}
    if pooled["lm"] < pooled["mutate"]:
        choice = "lm"
    elif pooled["mutate"] < pooled["lm"]:
        choice = "mutate"
    else:
        choice = TIE
    return choice, {"per_dataset_median": per_ds, "pooled_median": pooled}


def set_mode(manifest: dict, mode: str) -> list[str]:
    """Set the TUNE arms' `--const-opt` value to `mode`; the arms whose value changed."""
    changed = []
    for arm, spec in manifest["arms"].items():
        argv = spec["argv"]
        if FLAG in argv and arm.endswith("-TUNE"):
            i = argv.index(FLAG)
            if argv[i + 1] != mode:
                argv[i + 1] = mode
                changed.append(arm)
    return changed


def plan_writes(configs: Path, phases: list[str]) -> tuple[list, list[str]]:
    writes, errors = [], []
    for ph in phases:
        d = configs / ph
        if not d.is_dir():
            errors.append(f"no configuration directory {d}")
            continue
        found = False
        for f in sorted(d.glob("sr*.json")):
            m = json.loads(f.read_text())
            if any(a.endswith("-TUNE") and FLAG in s["argv"] for a, s in m["arms"].items()):
                writes.append((f, m))
                found = True
        if not found:
            errors.append(f"{d}: no regression manifest with a -TUNE arm")
    return writes, errors


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(REPO.parent / "gomea-results"))
    ap.add_argument("--configs", default=str(REPO / "configs" / "gomea"))
    ap.add_argument("--source", default="phase1an")
    ap.add_argument("--phases", default=PHASES)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    results, configs = Path(a.results).resolve(), Path(a.configs).resolve()
    phases = [p for p in a.phases.split(",") if p]
    values, errors = collect(results, configs / a.source, a.stage1_only)
    writes, werr = plan_writes(configs, phases)
    errors += werr
    if errors:
        for e in errors[:50]:
            print(f"set_const_opt: {e}", file=sys.stderr)
        print("set_const_opt: nothing changed", file=sys.stderr)
        return 1
    choice, med = choose(values)
    for mode in ARMS:
        meds = ", ".join(f"{ds} {v:.6g}" for ds, v in med["per_dataset_median"][mode].items())
        print(f"{ARMS[mode]}: median test NMSE {meds}; pooled {med['pooled_median'][mode]:.6g}")
    print(f"-> --const-opt {choice}")
    log = {
        "rule": "the tuning arm (LM or MUT) with the lower median test NMSE, pooled over "
                "the datasets' medians (median of the per-dataset medians); a failed run "
                f"counts as inf; a tie goes to {TIE}",
        "source": a.source, "stage1_only": a.stage1_only, "arms": ARMS,
        "n_runs": {m: {ds: len(v) for ds, v in d.items()} for m, d in values.items()},
        **med, "choice": choice, "written": [],
    }
    for path, m in writes:
        arms = set_mode(m, choice)
        what = "changed: " + ", ".join(arms) if arms else "unchanged"
        print(f"  {path}: --const-opt {choice} ({what})")
        log["written"].append({"path": str(path), "const_opt": choice, "changed_arms": arms})
        if not a.dry_run:
            path.write_text(json.dumps(m, indent=2) + "\n")
    if not a.dry_run:
        out = results / a.source / "const-opt-choice.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(log), indent=2) + "\n")
        print(f"logged to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
