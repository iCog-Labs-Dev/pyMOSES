"""Paired Wilcoxon (two-sided) LT vs RT, LT vs UNI per dataset; Holm over the 6 contrasts per setting."""
import json, sys
from pathlib import Path
from statistics import median
sys.path.insert(0, '/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/pyMoses')
from benchmarks.gomea.stats import holm as repo_holm, wilcoxon_signed_rank
D = Path(sys.argv[1]); P1AI = Path(sys.argv[2])
DS = ["Con", "EnH", "Air"]; ARMS = ["LT", "RT", "UNI", "LTMI"]
def load(s, ds, arm):
    return {k: [json.loads((D/"out"/f"setting{s}"/"runs"/f"{ds}__{arm}__s{i}"/"summary.json").read_text())[k]
                for i in range(1, 11)] for k in ["test_nmse", "final_program_nodes", "evaluations", "train_nmse", "wall_s"]}
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0]*len(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(ps)-r)*ps[i])); adj[i] = run
    return adj
# reproduction check against Phase 1ai
for s, arm in [("A", "GPGOMEA"), ("B", "GPGOMEA-P16")]:
    same = 0; tot = 0
    for ds in DS:
        for i in range(1, 11):
            a = json.loads((D/"out"/f"setting{s}"/"runs"/f"{ds}__LT__s{i}"/"summary.json").read_text())
            b = json.loads((P1AI/"runs"/f"{ds}__{arm}__s{i}"/"summary.json").read_text())
            tot += 1; same += (a["final_program"] == b["final_program"] and a["test_nmse"] == b["test_nmse"])
    print(f"setting {s} LT reproduces Phase 1ai {arm}: {same}/{tot} identical")
for s in "ABC":
    print(f"\n== Setting {s} ==")
    print("median test NMSE | median nodes | median evals | median train NMSE")
    data = {(ds, a): load(s, ds, a) for ds in DS for a in ARMS}
    for ds in DS:
        row = " ".join(f"{a}={median(data[ds,a]['test_nmse']):.2f}" for a in ARMS)
        nodes = " ".join(f"{a}={median(data[ds,a]['final_program_nodes']):.0f}" for a in ARMS)
        ev = " ".join(f"{a}={median(data[ds,a]['evaluations']):.0f}" for a in ARMS)
        tr = " ".join(f"{a}={median(data[ds,a]['train_nmse']):.2f}" for a in ARMS)
        print(f"{ds}: test[{row}]  nodes[{nodes}]  evals[{ev}]  train[{tr}]")
    cons = []; ps = []
    for ds in DS:
        for other in ["RT", "UNI", "LTMI"]:
            x, y = data[ds, "LT"]["test_nmse"], data[ds, other]["test_nmse"]
            w = sum(a < b for a, b in zip(x, y)); l = sum(a > b for a, b in zip(x, y))
            p = wilcoxon_signed_rank([a - b for a, b in zip(x, y)])[1]
            cons.append((ds, other, w, l, p, median(x) - median(y)))
            if other != "LTMI": ps.append(p)
    adj = repo_holm(ps); j = 0
    for ds, other, w, l, p, dm in cons:
        h = adj[j] if other != "LTMI" else None
        if other != "LTMI": j += 1
        hs = f"{h:.3f}" if h is not None else "(not in Holm family)"
        print(f"  {ds} LT vs {other}: LT wins/losses {w}/{l}, median diff {dm:+.2f}, raw p={p:.3f}, Holm p={hs}")
