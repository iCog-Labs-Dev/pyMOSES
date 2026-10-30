<!-- Copied from ../gomea-results/phase1f/datasets/PROVENANCE.md (local machine). Paths
     such as <Abbr>/raw/ refer to that local directory; the raw files are not committed.
     Dow and Tow have no committed full.csv: tools/server/setup.sh downloads their raw
     gpzgd files (rows below) and builds full.csv from them. See README.md in this
     directory. -->

# Phase 1f SR datasets: provenance

These are the ten datasets from Virgolin et al. 2021 (EvCo 29(2)), Table 2. All files were
downloaded on **2026-09-28**. Raw files sit untouched under `<Abbr>/raw/`. Independent
copies used only for cross-checking sit under `<Abbr>/raw/xcheck/`.
`pyMoses/tools/gomea/sr_datasets.py` builds `<Abbr>/full.csv` and `<Abbr>/splits/` from them,
and does no downloading. To rebuild:

    cd pyMoses && .venv/bin/python tools/gomea/sr_datasets.py ../gomea-results/phase1f/datasets

`SHA256SUMS` in this directory lists the `full.csv` hashes (check with `sha256sum -c SHA256SUMS`).

## Sources

- **UCI**: UCI Machine Learning Repository, `https://archive.ics.uci.edu/static/public/<id>/<name>.zip`.
- **Martins**: the folder behind the paper's footnote 3 (`https://goo.gl/tn6Zxv` redirects to
  Google Drive folder `1cUU7f23z_lBPQCOX7h1unZWKZH3YO_EB`). These are the Martins et al. 2018
  5-fold files. Fold 0's `train` and `test` together make the full dataset. The folder has
  **no Boston and no Dow** files. Download URL: `https://drive.google.com/uc?export=download&id=<file id>`.
- **gpzgd**: `github.com/grantdick/gpzgd` (Grant Dick, Univ. of Otago; GECCO 2020/2022 GP
  benchmark code), commit `42e5a1829a0b8a2eb50e116000001f82c770791e`, file
  `problems/data/<name>`. The format is an `N M` line, then M column-name lines, then N
  whitespace rows.
- **MultiGPG**: `github.com/matigekunstintelligentie/MultiGPG` (GP-GOMEA lineage, CWI),
  commit `53b831d008eb0db41536a72a1a78b9320d3eb8af`, `dataset/<name>.csv`.
- **GPFI**: `github.com/MaLeLabTs/GPFormulasInterpretability` (Virgolin co-author), commit
  `5a969c0276710dc10937bcd057aa2ab14dc213b5`, `datasets/<name>_full.dat`.

## Per dataset

In every `full.csv`, the header is `x1..xk,y` and the target is the last column. Values are
reprinted as Python's shortest round-trip float, with integers written without `.0`. Rows
keep the order of the canonical source. No rows were removed, added or deduplicated.
"Dups" is the number of exact duplicate rows that the original data already contains.

| Abbr | Canonical raw file (SHA-256) | Preprocessing | k x n (verified) | Dups | Cross-checks (row multisets) |
|---|---|---|---|---|---|
| Air | UCI 291 `airfoil+self+noise.zip` (`5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c`), member `airfoil_self_noise.dat` | tab-separated, 6 cols, target = sound pressure (col 6) | 5 x 1503 | 0 | Martins airfoil fold 0: exact |
| Bos | UCI legacy `https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data` (`baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a`) | whitespace, 14 cols, target = MEDV | 13 x 506 | 0 | gpzgd `housing`: exact |
| Con | gpzgd `concrete` (`a26b866b78da9349ce40b682c05fecaef4f46894eaa8bc4d6c76bba458d8c83f`) | gpzgd format, 9 cols, target = strength | 8 x 1030 | 25 | Martins concrete fold 0: exact. The UCI 165 zip (`dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7`) is kept in raw/ for reference, but its legacy `.xls` cannot be parsed with the stdlib. |
| Dow | gpzgd `dowchem` (`0e55bcd11740d65ab7c3fe80df8e2636b0e84840c46fc9182086868a5c9f52aa`) | gpzgd format, 58 cols, target = last | 57 x 1066 | 2 | MultiGPG `dowchemical.csv` (`ed9e3b3d…`): exact. GPFI `dowchemical_full.dat` (`b9c555a4…`): exact. The original EvoStar 2010 competition file (`casnew.iti.upv.es/cas/competitions/symreg/SymbolicRegressionCompetition.xls`, on the Wayback Machine only) is legacy `.xls` and was not used. |
| EnC | UCI 242 `energy+efficiency.zip` (`499441eee27929a4b00417f58fd8c63c9cc14b8a71520cd0dd27fcb626738351`), member `ENB2012_data.xlsx` | stdlib xlsx (zip+XML) parse of sheet 1. Columns A..H = X1..X8, target = column J = **Y2 (cooling)**. Header row and empty rows skipped. | 8 x 768 | 0 | Martins energyCooling fold 0: exact |
| EnH | same zip | same, target = column I = **Y1 (heating)** | 8 x 768 | 0 | Martins energyHeating fold 0 and gpzgd `energy`: equal except **one known diff**. xlsx cell I32 (X = 0.71, 710.5, 269.5, 220.5, 3.5, 4, 0, 0) stores Y1 = **6.366**, which the sheet's 2-decimal format displays as 6.37. Both copies carry 6.37. full.csv keeps UCI's 6.366 (see `KNOWN_XCHECK_DIFFS` in the tool). |
| Tow | gpzgd `tower` (`7d353f109f0d13c4c372ec9c3b7580580d351f0b29e395c9b05a0e89bc6492db`) | gpzgd format, 26 cols, target = last | 25 x 4999 | 0 | Martins towerData fold 0: exact. MultiGPG `tower.csv` (`a591f1d9…`): exact. |
| WiR | UCI 186 `wine+quality.zip` (`3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`), member `winequality-red.csv` | `;`-separated, header dropped, target = quality | 11 x 1599 | 240 | Martins wineRed fold 0: exact. gpzgd `winequality-red`: exact. |
| WiW | same zip, member `winequality-white.csv` | same | 11 x 4898 | 937 | Martins wineWhite fold 0: exact |
| Yac | UCI 243 `yacht+hydrodynamics.zip` (`aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac`), member `yacht_hydrodynamics.data` | whitespace, blank lines skipped, target = residuary resistance | 6 x 308 | 0 | Martins yacht fold 0: exact. gpzgd `yacht`: exact. |

Each cross-check compares the multiset of rows after rounding to 6 significant digits.
"exact" means the rows are also identical at full float precision. Every (k, n) matches
Table 2 exactly.

### Martins Drive file IDs (fold 0: train, test)

airfoil `1TQZDUYIYJhr3BaK3YifVKoGGId1eKCkS`, `14rg6TG-gt7LWHhw5oaAf7jrL_kvfr8DX`;
concrete `1DigbVMfmNoxgdyNHGuldnxp1G-GaLywF`, `15tDns8ZrWR4iFewv49eHxWX7mrVj9Hqg`;
energyCooling `1E49kC-vYApNQ9Ik595jzIL0PhhjgR9nH`, `1pFPD4Ujs-bZwjhUj3nPByIXmYE0Q_AJz`;
energyHeating `1-2OKw7-DBt-JimLOq7aPYLhcMILaKOti`, `1g5loTIZpc8nL_NKF0DGs17rkDocHiXv9`;
towerData `1pl0YQt9sIKkmYG34Tobn5GpUkRv4GK9I`, `13BzLFTEsICbfDbfteAMQaKJ7JENoqkOG`;
wineRed `1EGUBGQUkGwV8YWNdGlWIbbol7XVFxFJT`, `1C1seuJamEfFngGYjRteotKBxoMNFmpFJ`;
wineWhite `15o2OpOPlEhJBoAcyjcCGtHyie9dfKbBH`, `1LrfP9V13ElxAvSDKvHrAbDC_20FF0h3a`;
yacht `1cgZwANcMSPEOWZVIPhrmjY7eTPO6Xvwe`, `1p-6GqcQ06kP4JOhpHygxC49DD-QRe6_C`.
The SHA-256 of every raw and xcheck file can be reproduced with
`sha256sum */raw/* */raw/xcheck/*`.

## Splits

For each seed s = 1..30, `<Abbr>/splits/s<s>/` holds `{train,val,test,trainval}.{csv,dat}`.
`idx = list(range(n)); random.Random(f"{abbr}-s{s}").shuffle(idx)`. With q = n // 4, the
first n - 2q indices are train, the next q are val, and the last q are test. `trainval` is
train followed by val. The `.csv` files are MOSES format (header, commas). The `.dat` files
are GP-GOMEA format (no header, spaces, target last). `splits/splits.json` records the sizes,
the SHA-256 of the row indices, and `gpgomea_validation_fraction = (q + 0.5) / (n - q)`.

**GP-GOMEA `--validation` caveat.** The value is a fraction, not a percent: the code
computes `size_t nrows_validation = TR.n_rows * validation_perc` (GP-GOMEA
`src/Evolution/EvolutionState.cpp:20`). Passing `33.333` aborts with
`Mat::row(): index out of bounds`. The held-out rows are drawn uniformly at random with
Armadillo `randu()`, which is seeded by `--seed` (lines 21-26 and 189). So they are **not**
the rows of `val.dat`.
