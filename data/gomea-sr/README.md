# Symbolic-regression datasets for the server round

The six Phase 1f datasets the server round uses (Air, Bos, Con, EnC, EnH, Yac), committed so
that a fresh clone can rebuild them without the raw downloads:

* `<Abbr>/full.csv`: the file `tools/gomea/sr_datasets.py` built from the raw downloads on
  the local machine (header `x1..xk,y`, target last). `SHA256SUMS` lists their hashes.
* `<Abbr>/splits.json`: the local `splits/splits.json` (split sizes and the SHA-256 of
  every split's row indices).
* `PROVENANCE.md`: where every file came from and how it was parsed (the local
  `PROVENANCE.md`, cut down to these six datasets).

`tools/server/setup.sh` rebuilds `../gomea-results/phase1f/datasets/<Abbr>/` with

    .venv/bin/python tools/gomea/sr_datasets.py ../gomea-results/phase1f/datasets \
        Air Bos Con EnC EnH Yac --from-full data/gomea-sr

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
| Yac | Yacht Hydrodynamics (Gerritsma, Onnink and Versluis, 1981) | UCI ML Repository, id 243 | CC BY 4.0 |

CC BY 4.0 requires attribution: cite the UCI Machine Learning Repository (Dua and Graff /
Kelly, Longjohn and Nottingham, <https://archive.ics.uci.edu>) and the dataset's creators
above. The datasets are unmodified apart from the format conversion described in
`PROVENANCE.md`. The Boston Housing data has no explicit licence and carries a known
ethical problem (the `B` feature); it is used here only because Virgolin et al. (2021)
use it.
