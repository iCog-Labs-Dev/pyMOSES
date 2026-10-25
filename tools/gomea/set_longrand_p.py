"""Set LONG-RAND's p in Phase 1af's manifests from the 2-seed LONG-TOP pilot (1af-pilot).

Usage (from the repository root, after `tools/gomea/phase_run.sh phase1af-pilot` and before
`tools/gomea/phase_run.sh phase1af`)::

    .venv/bin/python tools/gomea/set_longrand_p.py [--results DIR] [--configs DIR] [--dry-run]

`PHASE1AC-1AG-PREREGISTRATION.md`, 1af: "p is set per problem to the median share of long
demes in a 2-seed LONG-TOP pilot (seeds 3001-3002 / splits 1-2)". Per problem, this script

1. finds the pilot's per-run GOMEA telemetry, `RUN_DIR/events.jsonl` (both runners pass
   `--gomea-log RUN_DIR/events.jsonl` to a GOMEA arm):
   * mux11: `RESULTS/1/phase1af-pilot-bool-<hash>/runs/mux11__LONG-TOP__s*/events.jsonl`,
     `<hash>` being the hash of `CONFIGS/../phase1af-pilot/bool.json` (benchmarks.gomea.runner's
     result directory, `<output_root>/<phase>/<name>-<hash>`);
   * Con: `RESULTS/phase1af-pilot/phase1af-pilot-sr/runs/Con__LONG-TOP__s*/events.jsonl`
     (moses_sr_runner.py's `<output_root>/<name>/runs/<run_id>`);
2. takes each run's share of long demes, `n_long_demes / n_demes` from the last event that
   carries both (the long-deme telemetry is on every "invocation_start" and "end" event;
   the last one holds the run's totals), and the median over the runs;
3. writes that p into `--gomea-long-deme-prob` of the LONG-RAND arm of every 1af manifest
   of that problem (mux11: `bool.json`; Con: `sr.json`; the extensions hold only the
   primary arms GOMEA and LONG-TOP) and logs the pilot's result to `RESULTS/phase1af-pilot/longrand-p.json`.

It exits 1, changing nothing, unless it finds exactly 2 pilot runs per problem, each with
the telemetry. `--dry-run` prints what it would set and writes nothing. RESULTS defaults to
`../gomea-results` (phase_run.sh's), CONFIGS to `configs/gomea/phase1af`, both relative to
the repository root.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmarks.gomea.manifest import manifest_hash  # noqa: E402

N_PILOT = 2
ARM = "LONG-RAND"
FLAG = "--gomea-long-deme-prob"
# problem -> the 1af manifests whose LONG-RAND arm takes that problem's p
TARGETS = {"mux11": ("bool.json",), "Con": ("sr.json",)}


def pilot_globs(results: Path, configs: Path) -> dict[str, str]:
    """Per problem, the glob of the pilot's LONG-TOP events files."""
    pilot = configs.parent / "phase1af-pilot"
    b = json.loads((pilot / "bool.json").read_text())
    s = json.loads((pilot / "sr.json").read_text())
    bdir = results / str(b["phase"]) / f"{b['name']}-{manifest_hash(b)}"
    return {
        "mux11": str(bdir / "runs" / "mux11__LONG-TOP__s*" / "events.jsonl"),
        "Con": str(results / "phase1af-pilot" / s["name"] / "runs" / "Con__LONG-TOP__s*"
                   / "events.jsonl"),
    }


def long_share(events: Path) -> tuple[int, int] | None:
    """`(n_long_demes, n_demes)` of the last event that has both, or None."""
    last = None
    for line in events.read_text().splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("n_demes") is not None \
                and ev.get("n_long_demes") is not None:
            last = (int(ev["n_long_demes"]), int(ev["n_demes"]))
    return last


def measure(results: Path, configs: Path) -> tuple[dict, list[str]]:
    """Per problem `{"p": median share, "runs": {run_dir: {...}}}`, and the problems found."""
    out, errors = {}, []
    for prob, pat in pilot_globs(results, configs).items():
        files = sorted(glob.glob(pat))
        if len(files) != N_PILOT:
            errors.append(f"{prob}: expected {N_PILOT} pilot runs, found {len(files)} ({pat})")
            continue
        runs = {}
        for f in files:
            c = long_share(Path(f))
            if c is None or c[1] == 0:
                errors.append(f"{prob}: no n_demes/n_long_demes telemetry in {f}")
                continue
            runs[str(Path(f).parent)] = {"n_long_demes": c[0], "n_demes": c[1],
                                         "share": c[0] / c[1]}
        if len(runs) == N_PILOT:
            out[prob] = {"p": statistics.median(r["share"] for r in runs.values()),
                         "runs": runs}
    return out, errors


def set_p(manifest: dict, p: float) -> bool:
    """Set FLAG of the LONG-RAND arm to p in place; True if the value changed."""
    argv = manifest["arms"][ARM]["argv"]
    i = argv.index(FLAG)
    new = repr(round(p, 6))
    changed = argv[i + 1] != new
    argv[i + 1] = new
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(REPO.parent / "gomea-results"))
    ap.add_argument("--configs", default=str(REPO / "configs" / "gomea" / "phase1af"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    results, configs = Path(a.results).resolve(), Path(a.configs).resolve()
    found, errors = measure(results, configs)
    if errors:
        for e in errors:
            print(f"set_longrand_p: {e}", file=sys.stderr)
        print("set_longrand_p: nothing changed", file=sys.stderr)
        return 1
    for prob, r in found.items():
        for name in TARGETS[prob]:
            path = configs / name
            m = json.loads(path.read_text())
            changed = set_p(m, r["p"])
            print(f"{prob}: p = {r['p']:.6g} -> {path} ({'changed' if changed else 'unchanged'})")
            if not a.dry_run:
                path.write_text(json.dumps(m, indent=2) + "\n")
    if not a.dry_run:
        log = results / "phase1af-pilot" / "longrand-p.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(found, indent=2) + "\n")
        print(f"logged to {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
