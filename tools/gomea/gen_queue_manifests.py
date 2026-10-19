"""Generate the manifests and specs of Phases 1u, 1v, 1w, 1x, 1z and 1aa (docs/gomea/QUEUE.md).

Usage (from the repository root): .venv/bin/python tools/gomea/gen_queue_manifests.py "$PWD"

Rerun after any code commit so every manifest's `commit` is the current HEAD.
"""
import json, subprocess, sys
from pathlib import Path
REPO = Path(sys.argv[1])
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
KR = ["--knob-probe-baseline", "reduced"]
def gomea(fos="lt-b", cap="250"):
    return ["-a", "gomea", "--gomea-fos", fos, "--gomea-pop-size", "16", "--gomea-skip-inactive",
            "0", "--gomea-max-evals-per-deme", cap, "--gomea-genotype-memo", "1"]
HC = ["-a", "hc-capped", "--gomea-max-evals-per-deme", "250"]
SEED = {"binary": "../gomea-results/gpgomea-seed-build/gpgomea-seed", "share": 0.2, "height": 3, "k": 5}
TOPK = lambda mode: ["--gomea-fos-topk", "4", "--gomea-fos-topk-mode", mode]
NC_TOL = "0.1"

def bool_manifest(name, arms, seeds):
    return {"schema_version": 1, "name": name, "phase": 1, "commit": COMMIT,
            "common_argv": ["-l", "WARN"],
            "tasks": {"mux11": {"argv": ["-H", "mux", "-k", "3"], "target_raw": 0}},
            "arms": {a: dict(c, argv=c["argv"] + KR) for a, c in arms.items()},
            "seeds": seeds, "budget": {"argv": ["-m", "40000"], "kind": "evaluations"},
            "wall_cap_s": 36000, "workers": 16, "output_root": "../gomea-results"}

def sr_manifest(name, arms, splits):
    return {"name": name, "datasets_dir": "../gomea-results/phase1f/datasets", "datasets": ["Con"],
            "splits": splits, "wall_cap_s": 36000, "mem_limit_mb": 3072,
            "arms": {a: {"binary": ".venv/bin/moses", "budget": {"evals": 10000},
                         "argv": c["argv"] + ["--gomea-contin", "copy"] * c.get("contin", False) + KR}
                     for a, c in arms.items()}}

S1, S2 = list(range(3001, 3011)), list(range(3011, 3021))
P1, P2 = list(range(1, 11)), list(range(11, 21))

def g(extra=(), fos="lt-b", cap="250"):
    return {"argv": gomea(fos, cap) + list(extra), "contin": True}

phases = {
  "1u": dict(bool={"HC": {"argv": HC}, "HC-S": {"argv": HC, "seed": SEED},
                   "GOMEA": {"argv": gomea()}, "GOMEA-S": {"argv": gomea(), "seed": SEED}},
             primary=[["mux11", "seed-hc", "HC-S", "HC"], ["mux11", "seed-gomea", "GOMEA-S", "GOMEA"]],
             secondary=[["mux11", "seeded-gomea-vs-hc", "GOMEA-S", "HC-S"]],
             descriptive=[["mux11", "gomea-vs-hc", "GOMEA", "HC"]]),
  "1v": dict(arms={"GOMEA": g(), "GOMEA-UNI": g(["--gomea-radii", "uniform"])},
             primary=[[p, "uniform", "GOMEA-UNI", "GOMEA"] for p in ("mux11", "Con")]),
  "1w": dict(arms={"GOMEA": g(), "TOPK-L": g(TOPK("linkage"), fos="lt-mi"),
                   "TOPK-R": g(TOPK("random"), fos="lt-mi")},
             primary=[[p, "topk", "TOPK-L", "GOMEA"] for p in ("mux11", "Con")],
             secondary=[[p, "linkage-vs-random", "TOPK-L", "TOPK-R"] for p in ("mux11", "Con")]),
  "1x": dict(arms={"GOMEA-C125": g(cap="125"), "GOMEA": g(), "GOMEA-C500": g(cap="500")},
             primary=[[p, n, a, "GOMEA"] for p in ("mux11", "Con")
                      for n, a in (("cap125", "GOMEA-C125"), ("cap500", "GOMEA-C500"))]),
  "1z": dict(sr={"GOMEA": g(), "GOMEA-NC": g(["--simplify-exemplar-numeric", NC_TOL])},
             primary=[["Con", "numeric-simplify", "GOMEA-NC", "GOMEA"]]),
  "1aa": dict(bool={"HC": {"argv": HC}, "HC-B3": {"argv": HC + ["-B", "3"]},
                    "GOMEA": {"argv": gomea()}, "GOMEA-B3": {"argv": gomea() + ["-B", "3"]}},
              primary=[["mux11", "b3-hc", "HC-B3", "HC"], ["mux11", "b3-gomea", "GOMEA-B3", "GOMEA"]]),
}
for ph, d in phases.items():
    out = REPO / "configs/gomea" / f"phase{ph}"
    out.mkdir(parents=True, exist_ok=True)
    boolarms = d.get("bool") or ({a: {"argv": c["argv"]} for a, c in d["arms"].items()} if "arms" in d else None)
    srarms = d.get("sr") or d.get("arms")
    if "bool" in d: srarms = None
    probs = {c[0] for c in d["primary"]}
    # Extensions only rerun arms in a primary contrast: secondary contrasts never extend.
    ext_arms = {a for c in d["primary"] for a in c[2:]}
    keep = lambda arms: {a: c for a, c in arms.items() if a in ext_arms}
    metrics = {}
    def dump(fn, obj): (out / fn).write_text(json.dumps(obj, indent=2) + "\n")
    if boolarms:
        metrics["mux11"] = "cpu_to_target"
        dump("bool.json", bool_manifest(f"phase{ph}-bool", boolarms, S1))
        if "mux11" in probs:
            dump("bool-ext-mux11.json", bool_manifest(f"phase{ph}-bool-ext-mux11", keep(boolarms), S2))
    if srarms:
        metrics["Con"] = "test_nmse"
        dump("sr.json", sr_manifest(f"phase{ph}-sr", srarms, P1))
        if "Con" in probs:
            dump("sr-ext-Con.json", sr_manifest(f"phase{ph}-sr-ext-Con", keep(srarms), P2))
    spec = {"phase": ph, "metrics": metrics, "primary": d["primary"], "secondary": d.get("secondary", [])}
    if d.get("descriptive"): spec["descriptive"] = d["descriptive"]
    dump("spec.json", spec)
    print(ph, sorted(p.name for p in out.iterdir()))
