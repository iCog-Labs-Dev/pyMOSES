"""Phase 1q deviation: rerun the runs that hit the 7,200 s wall cap, with a 36,000 s cap.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/phase1q_recap.py MANIFEST RESULT_DIR OUT_CONFIG_DIR

Writes one manifest per (task, arm) that has `timeout` runs in RESULT_DIR, identical to
MANIFEST except for its name, `tasks`, `arms`, `seeds` and `wall_cap_s`, and prints their
paths. Pass the rerun directories after the original ones to `phase1q_report.py`: a later
directory's run replaces an earlier one with the same task, arm and seed.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

RECAP_WALL_CAP_S = 36000


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    todo: dict = {}
    for f in glob.glob(str(Path(sys.argv[2]) / "runs/*/summary.json")):
        s = json.loads(Path(f).read_text())
        if s.get("status") == "timeout":
            todo.setdefault((s["task"], s["arm"]), []).append(s["seed"])
    for (task, arm), seeds in sorted(todo.items()):
        m = dict(manifest)
        m["name"] = f"{manifest['name']}-recap-{task}-{arm}"
        m["tasks"] = {task: manifest["tasks"][task]}
        m["arms"] = {arm: manifest["arms"][arm]}
        m["seeds"] = sorted(seeds)
        m["wall_cap_s"] = RECAP_WALL_CAP_S
        p = out / f"recap-{task}-{arm}.json"
        p.write_text(json.dumps(m, indent=2) + "\n")
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
