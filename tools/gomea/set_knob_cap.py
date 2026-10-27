"""Set K of the knob-cap arms of 1ak, 1al and 1am from the 1ak pilot (server round 2).

Usage (from the repository root, after `tools/gomea/phase_run.sh phase1ak-pilot` and before
`tools/gomea/phase_run.sh phase1ak`; the round-2 plan runs it as 1ak's `after_pilot`)::

    .venv/bin/python tools/gomea/set_knob_cap.py [--results DIR] [--configs DIR]
        [--pilot PHASE] [--phases P1,P2,...] [--dry-run]

`SERVER-ROUND2-PREREGISTRATION.md`, 1ak-pilot: "per problem, the largest K whose median
`generations` per deme (over the demes of both runs) is at least 3 ... If no K qualifies,
K = 8. A script writes K into every KC arm of 1ak and 1al and logs the pilot's numbers"
(and 1am, whose arms also use the pilot's K). Per problem, this script

1. finds the pilot's runs: every arm of the pilot manifests (`CONFIGS/PILOT/bool.json`, a
   Boolean task per problem; `CONFIGS/PILOT/sr*.json`, a regression dataset per problem)
   passes `--gomea-knob-cap K`; each run's GOMEA telemetry is `RUN_DIR/events.jsonl`
   (both runners pass `--gomea-log RUN_DIR/events.jsonl`):
   * Boolean: `RESULTS/1/<name>-<hash>/runs/<task>__<arm>__s<seed>/events.jsonl`
     (benchmarks.gomea.runner's result directory);
   * regression: `RESULTS/PILOT/<name>/runs/<dataset>__<arm>__s<split>/events.jsonl`
     (moses_sr_runner.py's);
2. per K, takes `generations` of every deme (`"end"` events) of the problem's runs and their
   median;
3. picks the largest K with a median of at least 3, else 8;
4. writes it after `--gomea-knob-cap` in every arm that has the flag, in every Boolean
   manifest (`bool*.json`, by task) and regression manifest (`sr*.json`, by dataset) of
   each target phase (`CONFIGS/<phase>/`), and logs the pilot's numbers and the choice to
   `RESULTS/PILOT/knob-cap.json`.

A manifest whose problems got different K values cannot take them and is an error. It exits
1, changing nothing, unless every problem has the pilot's runs (one per seed or split of the
pilot manifest, for every K), each with at least one `"end"` event. `--dry-run` prints what
it would set and writes nothing. RESULTS defaults to `../gomea-results`, CONFIGS to
`configs/gomea`, PILOT to `phase1ak-pilot`, the phases to `phase1ak,phase1al,phase1am`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmarks.gomea.manifest import manifest_hash  # noqa: E402

FLAG = "--gomea-knob-cap"
MIN_MEDIAN_GENERATIONS = 3
FALLBACK_K = 8
PHASES = "phase1ak,phase1al,phase1am"


def _k_of(argv: list[str]) -> int | None:
    return int(argv[argv.index(FLAG) + 1]) if FLAG in argv else None


def pilot_runs(results: Path, pilot_dir: Path) -> dict[str, dict[int, list[str]]]:
    """Per problem and K, the expected events.jsonl paths of the pilot's runs."""
    out: dict[str, dict[int, list[str]]] = {}
    pilot = pilot_dir.name
    for f in sorted(pilot_dir.glob("bool*.json")):
        m = json.loads(f.read_text())
        bdir = results / str(m["phase"]) / f"{m['name']}-{manifest_hash(m)}"
        for task in m["tasks"]:
            for arm, spec in m["arms"].items():
                k = _k_of(spec["argv"])
                if k is None:
                    continue
                out.setdefault(task, {}).setdefault(k, []).extend(
                    str(bdir / "runs" / f"{task}__{arm}__s{s}" / "events.jsonl")
                    for s in m["seeds"]
                )
    for f in sorted(pilot_dir.glob("sr*.json")):
        m = json.loads(f.read_text())
        for ds in m["datasets"]:
            for arm, spec in m["arms"].items():
                k = _k_of(spec["argv"])
                if k is None:
                    continue
                out.setdefault(ds, {}).setdefault(k, []).extend(
                    str(results / pilot / m["name"] / "runs" / f"{ds}__{arm}__s{s}"
                        / "events.jsonl")
                    for s in m["splits"]
                )
    return out


def deme_generations(events: Path) -> list[int]:
    """`generations` of every `"end"` event (one per deme)."""
    out = []
    for line in events.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "end" and ev.get("generations") is not None:
            out.append(int(ev["generations"]))
    return out


def choose(medians: dict[int, float]) -> int:
    """The largest K whose median is at least MIN_MEDIAN_GENERATIONS, else FALLBACK_K."""
    ok = [k for k, med in medians.items() if med >= MIN_MEDIAN_GENERATIONS]
    return max(ok) if ok else FALLBACK_K


def measure(results: Path, pilot_dir: Path) -> tuple[dict, list[str]]:
    """Per problem `{"K": chosen, "per_k": {K: {...}}}`, and the errors found."""
    found, errors = {}, []
    for prob, by_k in pilot_runs(results, pilot_dir).items():
        per_k, bad = {}, False
        for k, files in sorted(by_k.items()):
            gens, runs = [], {}
            for f in files:
                if not Path(f).exists():
                    errors.append(f"{prob}: K={k}: missing pilot run telemetry {f}")
                    bad = True
                    continue
                g = deme_generations(Path(f))
                if not g:
                    errors.append(f"{prob}: K={k}: no \"end\" events in {f}")
                    bad = True
                    continue
                runs[str(Path(f).parent)] = {"n_demes": len(g), "median_generations":
                                             statistics.median(g)}
                gens.extend(g)
            if gens:
                per_k[k] = {"median_generations": statistics.median(gens), "n_demes": len(gens),
                            "mean_generations": sum(gens) / len(gens), "runs": runs}
        if not bad:
            found[prob] = {"K": choose({k: v["median_generations"] for k, v in per_k.items()}),
                           "per_k": per_k}
    return found, errors


def set_k(manifest: dict, k: int) -> list[str]:
    """Set FLAG to k in every arm that has it; the arms whose value changed."""
    changed = []
    for arm, spec in manifest["arms"].items():
        argv = spec["argv"]
        if FLAG in argv:
            i = argv.index(FLAG)
            if argv[i + 1] != str(k):
                argv[i + 1] = str(k)
                changed.append(arm)
    return changed


def plan_writes(configs: Path, phases: list[str], found: dict) -> tuple[list, list[str]]:
    """`[(path, manifest, K)]` for every target manifest with a knob-cap arm, and errors."""
    writes, errors = [], []
    for ph in phases:
        d = configs / ph
        if not d.is_dir():
            errors.append(f"no configuration directory {d}")
            continue
        for f in sorted(list(d.glob("bool*.json")) + list(d.glob("sr*.json"))):
            m = json.loads(f.read_text())
            if not any(FLAG in a["argv"] for a in m["arms"].values()):
                continue
            probs = list(m["tasks"]) if "tasks" in m else list(m["datasets"])
            missing = [p for p in probs if p not in found]
            if missing:
                errors.append(f"{f}: no pilot result for {missing}")
                continue
            ks = {found[p]["K"] for p in probs}
            if len(ks) != 1:
                errors.append(f"{f}: its problems {probs} got different K {sorted(ks)}; "
                              "split it by problem")
                continue
            writes.append((f, m, ks.pop()))
    return writes, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(REPO.parent / "gomea-results"))
    ap.add_argument("--configs", default=str(REPO / "configs" / "gomea"))
    ap.add_argument("--pilot", default="phase1ak-pilot")
    ap.add_argument("--phases", default=PHASES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    results, configs = Path(a.results).resolve(), Path(a.configs).resolve()
    phases = [p for p in a.phases.split(",") if p]
    found, errors = measure(results, configs / a.pilot)
    if not found and not errors:
        errors.append(f"no knob-cap arms in the pilot manifests under {configs / a.pilot}")
    writes, werr = plan_writes(configs, phases, found) if not errors else ([], [])
    errors += werr
    if errors:
        for e in errors:
            print(f"set_knob_cap: {e}", file=sys.stderr)
        print("set_knob_cap: nothing changed", file=sys.stderr)
        return 1
    for prob, r in found.items():
        meds = ", ".join(f"K={k}: median {v['median_generations']:g} over {v['n_demes']} demes"
                         for k, v in sorted(r["per_k"].items()))
        print(f"{prob}: {meds} -> K = {r['K']}")
    log = {"rule": f"per problem the largest K with median generations per deme >= "
                   f"{MIN_MEDIAN_GENERATIONS}, else {FALLBACK_K}",
           "pilot": a.pilot, "problems": found, "written": []}
    for path, m, k in writes:
        arms = set_k(m, k)
        print(f"  {path}: K = {k} ({'changed: ' + ', '.join(arms) if arms else 'unchanged'})")
        log["written"].append({"path": str(path), "K": k, "changed_arms": arms})
        if not a.dry_run:
            path.write_text(json.dumps(m, indent=2) + "\n")
    if not a.dry_run:
        out = results / a.pilot / "knob-cap.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(log, indent=2) + "\n")
        print(f"logged to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
