"""Build the Phase 1f symbolic-regression datasets and their 30 train/val/test splits.

The ten real-world datasets of Virgolin et al. (2021), "Improving Model-based Genetic
Programming for Symbolic Regression of Small Expressions", Evolutionary Computation 29(2),
Table 2 (Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac). The raw downloads live untouched
under `DATASETS/<Abbr>/raw/` (see `DATASETS/PROVENANCE.md` for URLs and SHA-256s); this tool
never downloads anything.

Usage: .venv/bin/python tools/gomea/sr_datasets.py DATASETS [ABBR ...] [--no-splits]
       .venv/bin/python tools/gomea/sr_datasets.py DATASETS ABBR ... --from-full SRC_DIR

`--from-full SRC_DIR` (the server set-up, `tools/server/setup.sh`) needs no raw files: it
takes `SRC_DIR/<Abbr>/full.csv` (e.g. the committed copies in `data/gomea-sr/`), checks
its SHA-256 against `SRC_DIR/SHA256SUMS`, copies it to `DATASETS/<Abbr>/full.csv` and
writes the 30 splits from it as in step 3 below. It then checks that the regenerated
`splits.json` equals `SRC_DIR/<Abbr>/splits.json` (or `SRC_DIR/<Abbr>/splits/splits.json`)
exactly, and exits non-zero on any mismatch. full.csv prints every value with its
shortest round-trip repr, so re-reading it gives the same floats and the same split files,
byte for byte.

For every requested dataset (default: all ten) it

1. parses the raw file into `DATASETS/<Abbr>/full.csv` (header `x1,...,xk,y`, target last,
   every value numeric; values are re-printed with Python's shortest round-trip `repr`, with
   integral values printed without a trailing `.0`), and asserts the (#features, #examples)
   of the paper's Table 2 exactly;
2. cross-checks the parsed rows against the independent copies in `raw/xcheck/` (Martins et
   al. 2018 fold-0 train+test files, grantdick/gpzgd, MultiGPG, GPFormulasInterpretability;
   see PROVENANCE.md): the multisets of rows must be identical after rounding every value to
   6 significant digits, except the investigated entries of KNOWN_XCHECK_DIFFS;
3. writes, for split seeds 1..30, `DATASETS/<Abbr>/splits/s<seed>/` containing
   `train`, `val`, `test` and `trainval` (train rows followed by val rows, in that order),
   each as `.csv` (MOSES format: header `x1,...,xk,y`, comma separated) and as `.dat`
   (GP-GOMEA format: no header, space separated, target in the last column; GP-GOMEA reads
   it with Armadillo's `mat::load` auto-detection, `Fitness::ReadFitnessCases`, and takes the
   last column as the target, `Fitness::SetFitnessCases`).

Split scheme (paper Table 1: 50%-25%-25%): with n examples, n_val = n_test = n // 4 and
n_train = n - 2 * (n // 4). The row order is `random.Random(f"{abbr}-s{seed}")` shuffling
`list(range(n))` of full.csv; the first n_train shuffled rows are train, the next n_val are
val, the last n_test are test. `splits/splits.json` records the sizes, the row indices'
SHA-256 per split, and `gpgomea_validation_fraction`, the value to pass to GP-GOMEA's
`--validation` so that it holds out exactly n_val rows of trainval.dat. NOTE: GP-GOMEA's
`--validation` is a FRACTION (not a percentage) and picks those rows uniformly at random
with its own run-seeded RNG, so they are NOT the rows of val.dat.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

SEEDS = range(1, 31)


def _num(tok: str) -> float:
    v = float(tok)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {tok!r}")
    return v


def parse_ws(data: bytes) -> list[list[float]]:
    """Whitespace-separated numeric rows, blank lines skipped (UCI .data/.dat files)."""
    text = data.decode("utf-8")
    return [[_num(t) for t in line.split()] for line in text.splitlines() if line.strip()]


def parse_csv_plain(data: bytes) -> list[list[float]]:
    """Comma-separated numeric rows without header (the Martins et al. 2018 fold files)."""
    text = data.decode("utf-8")
    return [[_num(t) for t in r] for r in csv.reader(io.StringIO(text)) if r]


def parse_csv_header(data: bytes) -> list[list[float]]:
    """Comma-separated numeric rows after one header line (the MultiGPG copies)."""
    text = data.decode("utf-8")
    return [[_num(t) for t in r] for r in list(csv.reader(io.StringIO(text)))[1:] if r]


def parse_semicolon_header(data: bytes) -> list[list[float]]:
    """Semicolon-separated rows with one header line (UCI wine quality)."""
    text = data.decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=";") if r]
    return [[_num(t) for t in r] for r in rows[1:]]


def parse_gpzgd(data: bytes) -> list[list[float]]:
    """grantdick/gpzgd format: `N M`, then M column-name lines, then N rows of M values."""
    text = data.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n, m = (int(t) for t in lines[0].split())
    rows = [[_num(t) for t in ln.split()] for ln in lines[1 + m :]]
    assert len(rows) == n and all(len(r) == m for r in rows), "gpzgd header mismatch"
    return rows


def _xlsx_rows(data: bytes) -> list[dict[str, str]]:
    """Cells of the first worksheet of an .xlsx, as one {column letter: raw value} per row."""
    z = zipfile.ZipFile(io.BytesIO(data))
    sheet = z.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for row in re.findall(r"<row\b[^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for ref, attrs, body in re.findall(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', row, re.S):
            v = re.search(r"<v>(.*?)</v>", body, re.S)
            if v is not None and 't="s"' not in attrs:  # skip shared-string (header) cells
                cells[ref] = v.group(1)
        if cells:
            rows.append(cells)
    return rows


def _energy(target_col: str):
    def parse(data: bytes) -> list[list[float]]:
        feats = "ABCDEFGH"  # X1..X8
        out = []
        for cells in _xlsx_rows(data):
            if all(c in cells for c in feats + target_col):
                out.append([_num(cells[c]) for c in feats] + [_num(cells[target_col])])
        return out

    return parse


def _martins(prefix: str) -> tuple:
    return (parse_csv_plain, [f"{prefix}-train-0.csv", f"{prefix}-test-0.csv"])


# abbr: (raw file under raw/, zip member or None, parser(bytes), #features, #examples,
#        cross-checks: [(parser, [files under raw/xcheck/, concatenated])])
DATASETS = {
    "Air": (
        "airfoil+self+noise.zip",
        "airfoil_self_noise.dat",
        parse_ws,
        5,
        1503,
        [_martins("airfoil")],
    ),
    "Bos": ("housing.data", None, parse_ws, 13, 506, [(parse_gpzgd, ["housing"])]),
    "Con": ("concrete", None, parse_gpzgd, 8, 1030, [_martins("concrete")]),
    "Dow": (
        "dowchem",
        None,
        parse_gpzgd,
        57,
        1066,
        [
            (parse_csv_header, ["multigpg-dowchemical.csv"]),
            (parse_ws, ["gpfi-dowchemical_full.dat"]),
        ],
    ),
    "EnC": (
        "energy+efficiency.zip",
        "ENB2012_data.xlsx",
        _energy("J"),
        8,
        768,
        [_martins("energyCooling")],
    ),
    "EnH": (
        "energy+efficiency.zip",
        "ENB2012_data.xlsx",
        _energy("I"),
        8,
        768,
        [_martins("energyHeating"), (parse_gpzgd, ["energy"])],
    ),
    "Tow": (
        "tower",
        None,
        parse_gpzgd,
        25,
        4999,
        [_martins("towerData"), (parse_csv_header, ["multigpg-tower.csv"])],
    ),
    "WiR": (
        "wine+quality.zip",
        "winequality-red.csv",
        parse_semicolon_header,
        11,
        1599,
        [_martins("wineRed"), (parse_gpzgd, ["winequality-red"])],
    ),
    "WiW": (
        "wine+quality.zip",
        "winequality-white.csv",
        parse_semicolon_header,
        11,
        4898,
        [_martins("wineWhite")],
    ),
    "Yac": (
        "yacht+hydrodynamics.zip",
        "yacht_hydrodynamics.data",
        parse_ws,
        6,
        308,
        [_martins("yacht"), (parse_gpzgd, ["yacht"])],
    ),
}


def fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def load_raw(ddir: Path, abbr: str) -> list[list[float]]:
    raw, member, parser, k, n, _ = DATASETS[abbr]
    data = (ddir / "raw" / raw).read_bytes()
    if member is not None:
        data = zipfile.ZipFile(io.BytesIO(data)).read(member)
    rows = parser(data)
    widths = Counter(len(r) for r in rows)
    got = (next(iter(widths)) - 1 if len(widths) == 1 else widths, len(rows))
    if got != (k, n):
        raise SystemExit(f"{abbr}: expected {k} features x {n} examples, got {got}")
    return rows


def _key(row: list[float]) -> tuple:
    return tuple(float(f"{v:.6g}") for v in row)


# Known, investigated disagreements between the canonical source and a cross-check copy,
# as {abbr: {canonical row key: reference row key}}. They are tolerated and reported, never
# "fixed": full.csv always keeps the canonical source's value.
KNOWN_XCHECK_DIFFS = {
    # UCI ENB2012_data.xlsx cell I32 (row 31, Y1) stores 6.366 (6.3659999999999997), shown as
    # 6.37 under the sheet's 2-decimal number format; the Martins et al. and gpzgd copies were
    # exported from the displayed value and carry 6.37.
    "EnH": {
        (0.71, 710.5, 269.5, 220.5, 3.5, 4.0, 0.0, 0.0, 6.366): (
            (0.71, 710.5, 269.5, 220.5, 3.5, 4.0, 0.0, 0.0, 6.37)
        ),
    },
}


def cross_check(ddir: Path, abbr: str, rows: list[list[float]]) -> list[str]:
    notes = []
    for parser, files in DATASETS[abbr][5]:
        ref = []
        for f in files:
            ref += parser((ddir / "raw" / "xcheck" / f).read_bytes())
        ours, theirs = Counter(map(_key, rows)), Counter(map(_key, ref))
        only_ours, only_theirs = ours - theirs, theirs - ours
        known = KNOWN_XCHECK_DIFFS.get(abbr, {})
        tolerated = [
            k for k in only_ours if k in known and only_ours[k] == only_theirs.get(known[k])
        ]
        for k in tolerated:
            del only_theirs[known[k]]
            del only_ours[k]
        exact = Counter(map(tuple, ref)) == Counter(map(tuple, rows))
        status = "exact" if exact else "equal at 6 significant digits"
        if tolerated:
            status += f" except known diffs {[(k, known[k]) for k in tolerated]}"
        notes.append(f"{'+'.join(files)}: {len(ref)} rows, row multiset {status}")
        if only_ours or only_theirs:
            raise SystemExit(
                f"{abbr}: cross-check against {files} failed: only in canonical "
                f"{list(only_ours.items())[:5]}, only in reference {list(only_theirs.items())[:5]}"
            )
    return notes


def csv_text(rows: list[list[float]], k: int) -> str:
    head = ",".join([f"x{i}" for i in range(1, k + 1)] + ["y"])
    return head + "\n" + "".join(",".join(map(fmt, r)) + "\n" for r in rows)


def dat_text(rows: list[list[float]]) -> str:
    return "".join(" ".join(map(fmt, r)) + "\n" for r in rows)


def split_indices(abbr: str, seed: int, n: int) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    random.Random(f"{abbr}-s{seed}").shuffle(idx)
    q = n // 4
    n_train = n - 2 * q
    return idx[:n_train], idx[n_train : n_train + q], idx[n_train + q :]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_full(src: Path, abbr: str) -> tuple[list[list[float]], str]:
    """Rows of `src/<abbr>/full.csv`, after checking its SHA-256 against
    `src/SHA256SUMS` (line `<sha>  <abbr>/full.csv`), its header and (k, n)."""
    k, n = DATASETS[abbr][3], DATASETS[abbr][4]
    text = (src / abbr / "full.csv").read_text()
    sums = {}
    for line in (src / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = h
    want = sums.get(f"{abbr}/full.csv")
    if want is None:
        raise SystemExit(f"{abbr}: no {abbr}/full.csv line in {src / 'SHA256SUMS'}")
    if _sha(text) != want:
        raise SystemExit(f"{abbr}: full.csv SHA-256 {_sha(text)} != {want} (SHA256SUMS)")
    lines = text.splitlines()
    head = ",".join([f"x{i}" for i in range(1, k + 1)] + ["y"])
    if lines[0] != head:
        raise SystemExit(f"{abbr}: full.csv header {lines[0]!r} != {head!r}")
    rows = [[_num(t) for t in line.split(",")] for line in lines[1:] if line.strip()]
    if len(rows) != n or any(len(r) != k + 1 for r in rows):
        raise SystemExit(f"{abbr}: full.csv is not {k} features x {n} examples")
    if csv_text(rows, k) != text:
        raise SystemExit(f"{abbr}: full.csv does not re-print identically")
    return rows, text


def build(root: Path, abbr: str, splits: bool, from_full: Path | None = None) -> dict:
    ddir = root / abbr
    k, n = DATASETS[abbr][3], DATASETS[abbr][4]
    if from_full is None:
        rows = load_raw(ddir, abbr)
        notes = cross_check(ddir, abbr, rows)
        full = csv_text(rows, k)
    else:
        rows, full = load_full(from_full, abbr)
        notes = [f"from {from_full / abbr / 'full.csv'} (SHA-256 checked; no raw files)"]
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "full.csv").write_text(full)
    info = {"features": k, "examples": n, "full_csv_sha256": _sha(full), "xcheck": notes}
    dups = n - len(set(map(tuple, rows)))
    info["duplicate_rows"] = dups
    if splits:
        meta = {}
        for seed in SEEDS:
            tr, va, te = split_indices(abbr, seed, n)
            out = ddir / "splits" / f"s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            parts = {"train": tr, "val": va, "test": te, "trainval": tr + va}
            for name, ix in parts.items():
                sub = [rows[i] for i in ix]
                (out / f"{name}.csv").write_text(csv_text(sub, k))
                (out / f"{name}.dat").write_text(dat_text(sub))
            meta[f"s{seed}"] = {
                "rng": f'random.Random("{abbr}-s{seed}").shuffle(range({n}))',
                "sizes": {name: len(ix) for name, ix in parts.items()},
                "row_indices_sha256": {
                    name: _sha(",".join(map(str, ix))) for name, ix in parts.items()
                },
            }
        q = n // 4
        summary = {
            "abbr": abbr,
            "n": n,
            "n_train": n - 2 * q,
            "n_val": q,
            "n_test": q,
            "gpgomea_validation_fraction": (q + 0.5) / (n - q),
            "splits": meta,
        }
        text = json.dumps(summary, indent=1) + "\n"
        (ddir / "splits" / "splits.json").write_text(text)
        info["sizes"] = {"train": n - 2 * q, "val": q, "test": q}
        if from_full is not None:
            ref = next(
                (p for p in (from_full / abbr / "splits.json",
                             from_full / abbr / "splits" / "splits.json") if p.exists()),
                None,
            )
            if ref is None:
                raise SystemExit(f"{abbr}: no splits.json under {from_full / abbr}")
            if ref.read_text() != text:
                raise SystemExit(f"{abbr}: regenerated splits.json differs from {ref}")
            info["splits_json_matches"] = str(ref)
    return info


def main() -> int:
    argv = sys.argv[1:]
    from_full = None
    if "--from-full" in argv:
        i = argv.index("--from-full")
        if i + 1 >= len(argv):
            raise SystemExit("--from-full needs SRC_DIR")
        from_full = Path(argv[i + 1])
        del argv[i : i + 2]
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    root = Path(args[0])
    wanted = args[1:] or list(DATASETS)
    unknown = [a for a in wanted if a not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {list(DATASETS)}")
    splits = "--no-splits" not in argv
    report = {abbr: build(root, abbr, splits, from_full) for abbr in wanted}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
