# Symbolic-regression datasets for the server round

The ten Phase 1f datasets the server rounds use. Eight of them (Air, Bos, Con, EnC, EnH,
WiR, WiW, Yac) are committed so that a fresh clone can rebuild them without the raw
downloads; Dow and Tow are not committed (see "Dow and Tow" below):

* `<Abbr>/full.csv`: the file `tools/gomea/sr_datasets.py` built from the raw downloads on
  the local machine (header `x1..xk,y`, target last). `SHA256SUMS` lists their hashes
  (also those of Dow and Tow, which are built on the server).
* `<Abbr>/splits.json`: the local `splits/splits.json` (split sizes and the SHA-256 of
  every split's row indices).
* `PROVENANCE.md`: where every file came from and how it was parsed (the local
  `PROVENANCE.md`).

`tools/server/setup.sh` rebuilds `../gomea-results/phase1f/datasets/<Abbr>/` with

    .venv/bin/python tools/gomea/sr_datasets.py ../gomea-results/phase1f/datasets \
        Air Bos Con EnC EnH WiR WiW Yac --from-full data/gomea-sr

which checks each `full.csv` against `SHA256SUMS`, writes the 30 splits and fails unless
the regenerated `splits.json` equals the committed one. On the local machine the rebuilt
`splits/` directories (s1..s30, `.csv` and `.dat`) were byte-identical to the originals
(`diff -r`, 2026-10-25).

## Licences and citations

| Abbr | Dataset | Source | Licence |
|---|---|---|---|
| Air | Airfoil Self-Noise (Brooks, Pope and Marcolini, 1989) | UCI ML Repository, id 291 | CC BY 4.0 |
| Bos | Boston Housing (Harrison and Rubinfeld, 1978) | UCI legacy archive (`housing.data`, originally StatLib) | none stated by UCI; widely redistributed |
| Con | Concrete Compressive Strength (Yeh, 1998) | UCI ML Repository, id 165 (rows taken from the grantdick/gpzgd copy, identical to the Martins et al. copy) | CC BY 4.0 (UCI) |
| EnC | Energy Efficiency, cooling load Y2 (Tsanas and Xifara, 2012) | UCI ML Repository, id 242 | CC BY 4.0 |
| EnH | Energy Efficiency, heating load Y1 (Tsanas and Xifara, 2012) | UCI ML Repository, id 242 | CC BY 4.0 |
| WiR | Wine Quality, red (Cortez, Cerdeira, Almeida, Matos and Reis, 2009) | UCI ML Repository, id 186 | CC BY 4.0 |
| WiW | Wine Quality, white (Cortez, Cerdeira, Almeida, Matos and Reis, 2009) | UCI ML Repository, id 186 | CC BY 4.0 |
| Yac | Yacht Hydrodynamics (Gerritsma, Onnink and Versluis, 1981) | UCI ML Repository, id 243 | CC BY 4.0 |

CC BY 4.0 requires attribution: cite the UCI Machine Learning Repository (Dua and Graff /
Kelly, Longjohn and Nottingham, <https://archive.ics.uci.edu>) and the dataset's creators
above. The datasets are unmodified apart from the format conversion described in
`PROVENANCE.md`. For WiR and WiW the conversion removes the header line and changes the
separator `;` to `,` (and prints the values as described there). Wine Quality citation:
Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009). Wine Quality
[Dataset]. UCI Machine Learning Repository. doi:10.24432/C56S3T; and the paper "Modeling
wine preferences by data mining from physicochemical properties", Decision Support
Systems 47(4), 2009 (<https://archive.ics.uci.edu/dataset/186/wine+quality>). The Boston Housing data has no explicit licence and carries a known
ethical problem (the `B` feature); it is used here only because Virgolin et al. (2021)
use it.

## Dow and Tow

Dow (Dow Chemical, "All rights reserved") and Tow (tower) have no licence that permits
redistribution, and neither do the repositories that hold copies (gpzgd, MultiGPG). So
their data is not committed. Only `Dow/splits.json`, `Tow/splits.json` (split sizes and
SHA-256 values of row indices, no data) and their `SHA256SUMS` lines are here.
`tools/server/setup.sh` downloads the raw files from github.com/grantdick/gpzgd at commit
`42e5a1829a0b8a2eb50e116000001f82c770791e` (`problems/data/dowchem`,
`problems/data/tower`) to `../gomea-results/phase1f/datasets/<Abbr>/raw/`, checks their
SHA-256 against the values in `PROVENANCE.md`, and builds them with

    .venv/bin/python tools/gomea/sr_datasets.py ../gomea-results/phase1f/datasets \
        Dow Tow --from-raw data/gomea-sr

which checks the built `full.csv` against `SHA256SUMS` and the regenerated `splits.json`
against the committed one. If a download or a check fails, set-up fails and names the
dataset, and phase 1ar (the only phase that uses them) does not start.
