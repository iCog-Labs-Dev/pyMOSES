"""Server round 15: check that the training and validation tables of new splits have the same
column types.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/check_split_types.py DATASETS_DIR LO HI [ABBR ...]

For each dataset (default: all ten of ``sr_datasets.DATASETS``) and each split LO..HI, the tool
reads ``DATASETS_DIR/<Abbr>/splits/s<k>/{train,val,test}.csv`` in two ways:

* ``loadTable`` of pymoses (the type inference of MOSES: the first 20 data rows). This is the
  check that round 14 did by hand on splits 61-70 (a table that does not load counts as a
  difference, with the error text);
* ``sr_eval.read_table`` (types over all rows): the boolean inputs that the stand-alone GOMEA
  harness and the cross-check use (``bool_cols``).

The check passes when, for every split, the validation table has the input types of the
training table in both ways. The test table is also compared and reported, but it does not
decide the result. The tool prints a JSON report and exits 0 when the check passes, else 1.
It reads only the column types; it prints no data values.

NOT A PORT: server round 15 (the IT linkage study, BACKLOG 13.9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_datasets  # noqa: E402
import sr_eval  # noqa: E402

PARTS = ("train", "val", "test")


def _moses_types(path: Path):
    """The input types of ``loadTable`` (as strings), or the error text of a failed load."""
    from pymoses.comboreduct.table.table_io import loadTable

    try:
        return [str(t) for t in loadTable(str(path), "y").itable.get_types()]
    except Exception as exc:  # a table that MOSES cannot load is a difference
        return f"load error: {type(exc).__name__}: {exc}"[:200]


def _bool_cols(path: Path):
    """The boolean inputs of ``sr_eval.read_table`` (sorted indices)."""
    return sorted(sr_eval.read_table(str(path), "y").bool_cols)


def check_split(split_dir: Path) -> dict:
    """The types of one split and the result of the comparison."""
    moses = {p: _moses_types(split_dir / f"{p}.csv") for p in PARTS}
    bools = {p: _bool_cols(split_dir / f"{p}.csv") for p in PARTS}
    return {
        "val_same": moses["val"] == moses["train"] and bools["val"] == bools["train"],
        "test_same": moses["test"] == moses["train"] and bools["test"] == bools["train"],
        "moses_types": moses,
        "bool_cols": bools,
    }


def check(root: Path, abbrs, splits) -> dict:
    """The report over the datasets ``abbrs`` and the splits ``splits`` under ``root``."""
    per = {}
    for a in abbrs:
        for k in splits:
            per[f"{a}/s{k}"] = check_split(Path(root) / a / "splits" / f"s{k}")
    val_diff = sorted(n for n, r in per.items() if not r["val_same"])
    test_diff = sorted(n for n, r in per.items() if not r["test_same"])
    return {
        "splits_checked": len(per),
        "val_differ": val_diff,
        "test_differ": test_diff,
        "passed": not val_diff,
        "details": {n: per[n] for n in sorted(set(val_diff) | set(test_diff))},
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, lo, hi = Path(argv[0]), int(argv[1]), int(argv[2])
    abbrs = argv[3:] or list(sr_datasets.DATASETS)
    unknown = [a for a in abbrs if a not in sr_datasets.DATASETS]
    if unknown:
        print(f"unknown dataset(s) {unknown}", file=sys.stderr)
        return 2
    report = check(root, abbrs, range(lo, hi + 1))
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
