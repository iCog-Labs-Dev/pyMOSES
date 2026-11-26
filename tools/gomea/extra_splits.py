"""Write split seeds of the Phase 1f SR datasets that `sr_datasets.py` does not write.

NOT A PORT, backlog 15.4 (D1b, a local check): `sr_datasets.py` writes the split seeds 1-90
(`SEEDS`) and their `splits.json`, and the server set-up checks that file. The D1b check uses
the fresh seeds 91-100 of Yacht. This tool writes only those extra split folders, with the
split scheme and the file format of `sr_datasets.py` (`split_indices`, `csv_text`,
`dat_text`), so a seed gives the same rows and the same bytes as `sr_datasets.py` would give
for it.

Usage: .venv/bin/python tools/gomea/extra_splits.py DATASETS ABBR [ABBR ...] --seeds A-B
           [--from-full SRC_DIR]

* The rows come from `SRC_DIR/<Abbr>/full.csv` (default: `DATASETS`), after the checks of
  `sr_datasets.load_full`: the SHA-256 against `SRC_DIR/SHA256SUMS`, the header, the size and
  the round trip. `data/gomea-sr` (in git) is a valid `SRC_DIR`.
* It writes `DATASETS/<Abbr>/splits/s<seed>/` with `train`, `val`, `test` and `trainval`, each
  as `.csv` and `.dat`. It never writes `full.csv` or `splits.json`.
* It does not change a file that exists with other content: it stops with an error. A file
  with the same content is left as it is.
* It prints, as JSON, the sizes and the SHA-256 of the row indices of each split (the fields
  of `splits.json`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_datasets as srd  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    """`"91-100"` -> [91, ..., 100]; `"91"` -> [91]. Seeds 1-90 belong to `sr_datasets.py`."""
    lo, sep, hi = text.partition("-")
    a, b = int(lo), int(hi) if sep else int(lo)
    if a > b:
        raise ValueError(f"bad seed range {text!r}")
    seeds = list(range(a, b + 1))
    inside = [s for s in seeds if s in srd.SEEDS]
    if inside:
        raise ValueError(
            f"seeds {inside[0]}-{inside[-1]} are written by sr_datasets.py; use seeds above "
            f"{srd.SEEDS[-1]}"
        )
    return seeds


def _write(path: Path, text: str) -> str:
    """Write `text` to `path`; "same" if the file has it already; an error for other content."""
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with other content; not changed")
        return "same"
    path.write_text(text)
    return "new"


def write_splits(root: Path, abbr: str, seeds: list[int], src: Path) -> dict:
    """Write the split folders of `seeds` for `abbr` under `root`; return their meta."""
    k, n = srd.DATASETS[abbr][3], srd.DATASETS[abbr][4]
    rows, _full = srd.load_full(src, abbr)
    meta = {}
    for seed in seeds:
        tr, va, te = srd.split_indices(abbr, seed, n)
        out = root / abbr / "splits" / f"s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        parts = {"train": tr, "val": va, "test": te, "trainval": tr + va}
        status = set()
        for name, ix in parts.items():
            sub = [rows[i] for i in ix]
            status.add(_write(out / f"{name}.csv", srd.csv_text(sub, k)))
            status.add(_write(out / f"{name}.dat", srd.dat_text(sub)))
        meta[f"s{seed}"] = {
            "rng": f'random.Random("{abbr}-s{seed}").shuffle(range({n}))',
            "sizes": {name: len(ix) for name, ix in parts.items()},
            "row_indices_sha256": {
                name: srd._sha(",".join(map(str, ix))) for name, ix in parts.items()
            },
            "files": "/".join(sorted(status)),
        }
    return meta


def main(argv: list[str]) -> int:
    opts = {}
    for opt in ("--seeds", "--from-full"):
        if opt in argv:
            i = argv.index(opt)
            if i + 1 >= len(argv):
                raise SystemExit(f"{opt} needs a value")
            opts[opt] = argv[i + 1]
            del argv[i : i + 2]
    if "--seeds" not in opts or len(argv) < 2:
        raise SystemExit(__doc__)
    root = Path(argv[0])
    wanted = argv[1:]
    unknown = [a for a in wanted if a not in srd.DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {list(srd.DATASETS)}")
    seeds = parse_seeds(opts["--seeds"])
    src = Path(opts.get("--from-full", root))
    report = {abbr: write_splits(root, abbr, seeds, src) for abbr in wanted}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
