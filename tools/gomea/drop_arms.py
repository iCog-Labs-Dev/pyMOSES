"""Remove arms from phase configs: the manifests and the spec (server round 17: HC-TLS is left out
if the HC-TLS check fails, `SERVER-ROUND17-PREREGISTRATION.md`, "Arms").

Usage (from the repository root)::

    .venv/bin/python tools/gomea/drop_arms.py CONFIG_DIR [CONFIG_DIR ...] --arms ARM [ARM ...]

For each CONFIG_DIR:

* each manifest `sr*.json`, `bool*.json` and `gp*.json` loses the arms (`"arms"`). A manifest
  that has no arm left is refused, and nothing is written;
* `spec.json` loses each contrast that names one of the arms: in `primary`, `secondary` and
  `descriptive` (arms at positions 2 and 3), in `pooled_primary` and `pooled_secondary`
  (positions 1 and 2), and in each family of `contrast_families` (each arm of the contrast). A
  family that has no contrast left is removed, and so are the names of removed contrasts in
  `secondary_families`, `pooled_secondary_families` and `pooled_descriptive`. The entries of
  `anytime_ratios` and `evals_to_reference` with one of the arms (key or value), the pairs of
  `same_search` with one of the arms, and the entries of `partners.pairs` with one of the arms
  as key are removed. The other keys do not change.

The files are written as `json.dumps(obj, indent=2)` and a newline, the format of the round 17
configs. The script prints what it removed. Round 17::

    .venv/bin/python tools/gomea/drop_arms.py configs/gomea/phase1bl configs/gomea/phase1bl-a \\
        configs/gomea/server-smoke17-1bl configs/gomea/server-smoke17-1bl-a \\
        --arms HC-TLS HC-TLS-100K

Then set `HC_TLS = False` in `tests/gomea/round17_design.py` (the tests then check the design
without HC-TLS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def drop_spec(spec: dict, arms: set[str]) -> tuple[dict, list[str]]:
    """`spec` without the contrasts and entries that name one of `arms`, and the names of the
    removed contrasts."""
    s = json.loads(json.dumps(spec))
    gone: list[str] = []

    def keep(c, idx) -> bool:
        if any(c[i] in arms for i in idx if i < len(c)):
            gone.append(c[1] if idx == (2, 3) else c[0])
            return False
        return True

    for k in ("primary", "secondary", "descriptive"):
        if k in s:
            s[k] = [c for c in s[k] if keep(c, (2, 3))]
    for k in ("pooled_primary", "pooled_secondary"):
        if k in s:
            s[k] = [c for c in s[k] if keep(c, (1, 2))]
    if "contrast_families" in s:
        fams = []
        for f in s["contrast_families"]:
            f["contrasts"] = [c for c in f["contrasts"] if keep(c, tuple(range(1, len(c) - 1)))]
            if f["contrasts"]:
                fams.append(f)
        s["contrast_families"] = fams
    for k in ("secondary_families", "pooled_secondary_families"):
        if k in s:
            s[k] = [g for g in ([n for n in g if n not in gone] for g in s[k]) if g]
    if "pooled_descriptive" in s:
        s["pooled_descriptive"] = [n for n in s["pooled_descriptive"] if n not in gone]
    for k in ("anytime_ratios", "evals_to_reference"):
        if k in s:
            s[k] = {a: b for a, b in s[k].items() if a not in arms and b not in arms}
    if "same_search" in s:
        s["same_search"] = [p for p in s["same_search"] if not set(p) & arms]
    if "partners" in s:
        s["partners"]["pairs"] = {a: b for a, b in s["partners"]["pairs"].items()
                                  if a not in arms}
    return s, gone


def drop_dir(q: Path, arms: set[str]) -> dict[Path, str]:
    """The new text of each changed file of `q` (nothing is written); raises ValueError when a
    manifest would have no arm left."""
    out: dict[Path, str] = {}
    for f in sorted(q.glob("*.json")):
        if f.name == "spec.json" or not f.name.startswith(("sr", "bool", "gp")):
            continue
        m = json.loads(f.read_text())
        if not set(m.get("arms", {})) & arms:
            continue
        m["arms"] = {a: v for a, v in m["arms"].items() if a not in arms}
        if not m["arms"]:
            raise ValueError(f"{f}: no arm left")
        out[f] = json.dumps(m, indent=2) + "\n"
    sp = q / "spec.json"
    if sp.exists():
        spec, gone = drop_spec(json.loads(sp.read_text()), arms)
        out[sp] = json.dumps(spec, indent=2) + "\n"
        print(f"{sp}: removed the contrasts {gone}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args(argv)
    arms = set(a.arms)
    try:
        todo = {f: t for q in a.dirs for f, t in drop_dir(q, arms).items()}
    except ValueError as exc:
        print(f"refused, nothing written: {exc}", file=sys.stderr)
        return 2
    for f, t in todo.items():
        f.write_text(t)
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
