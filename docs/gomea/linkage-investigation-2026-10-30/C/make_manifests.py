"""Write GP-GOMEA FOS-ablation manifests for tools/gomea/gpgomea_sr_runner.py."""
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
M = "/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/gomea-results"
B = f"{M}/phase1b/gpgomea-build/gpgomea"
BLTMI = f"{M}/phase1b/gpgomea-build/gpgomea-ltmi"
SETTINGS = {"A": (500, 10000), "B": (16, 10000), "C": (2000, 50000)}
for s, (pop, ev) in SETTINGS.items():
    base = ["--linearscaling", "--functions", "+_-_*_aq", "--erc", "--gomea",
            "--initmaxtreeheight", "4", "--ims", "4_1", "--popsize", str(pop),
            "--evaluations", str(ev), "--generations", "-1"]
    arms = {
        "LT": {"binary": B, "train": "train", "argv": base + ["--gomfos", "LT"]},
        "RT": {"binary": B, "train": "train", "argv": base + ["--gomfos", "RT"]},
        "UNI": {"binary": B, "train": "train", "argv": base + ["--gomfos", "U"]},
        "LTMI": {"binary": BLTMI, "train": "train", "argv": base + ["--gomfos", "LT"]},
    }
    m = {"name": f"setting{s}", "datasets_dir": f"{M}/phase1f/datasets",
         "datasets": ["Con", "EnH", "Air"], "splits": list(range(1, 11)),
         "wall_cap_s": 1800, "arms": arms}
    (D / "manifests" / f"setting{s}.json").write_text(json.dumps(m, indent=2))
