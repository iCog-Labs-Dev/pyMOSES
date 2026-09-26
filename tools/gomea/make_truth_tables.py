"""Write the Phase 1b Boolean tasks as GP-GOMEA data files.

The targets are produced by the very function objects MOSES scores against
(`pymoses.moses.main.demo_problems`: `even_parity`, `majority`, `multiplex`), so the
stand-alone GP-GOMEA runs see exactly the truth tables of `-H pa -k 4`, `-H maj -k 5` and
`-H mux -k 2`. Every one of the 2**arity input rows is written, one per line, as
space-separated 0/1 values with the target in the last column and no header (the format
GP-GOMEA's `--train` reader expects). MOSES's input `$i` is column `x(i-1)`.

Usage: .venv/bin/python tools/gomea/make_truth_tables.py OUT_DIR
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pymoses.moses.main.demo_problems import (  # noqa: E402
    even_parity,
    logical_bscore,
    majority,
    multiplex,
)

TASKS = {
    # name: (MOSES flags, function, arity)
    "pa4": (["-H", "pa", "-k", "4"], even_parity, 4),
    "maj5": (["-H", "maj", "-k", "5"], majority(5), 5),
    "mux6": (["-H", "mux", "-k", "2"], multiplex(2), 2 + 2**2),
}


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (flags, func, arity) in TASKS.items():
        rows = [(bits, bool(func(bits))) for bits in itertools.product([False, True], repeat=arity)]
        # Cross-check against MOSES's own scorer target: same number of rows and the same
        # number of true outputs as logical_bscore builds for this problem.
        target = logical_bscore(func, arity)._target
        assert len(target) == len(rows) and sum(target) == sum(y for _, y in rows), name
        text = "".join(" ".join(str(int(b)) for b in bits) + f" {int(y)}\n" for bits, y in rows)
        path = out / f"{name}.dat"
        path.write_text(text)
        meta[name] = {
            "moses_flags": flags,
            "arity": arity,
            "rows": len(rows),
            "positives": sum(y for _, y in rows),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    (out / "truth_tables.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
