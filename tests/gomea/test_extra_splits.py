"""NOT A PORT, backlog 15.4 (D1b): ``tools/gomea/extra_splits.py`` writes split seeds above 90
with the scheme and the format of ``sr_datasets.py``.

* Seed 1 written by the tool has the meta of the committed ``data/gomea-sr/Yac/splits.json``
  (sizes and SHA-256 of the row indices), so the tool and ``sr_datasets.py`` give the same rows.
* The tool refuses seeds 1-90, never writes ``full.csv`` or ``splits.json``, leaves a file with
  the same content as it is and refuses to change a file with other content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "gomea-sr"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import extra_splits as es  # noqa: E402


def test_seed_1_has_the_meta_of_sr_datasets(tmp_path):
    meta = es.write_splits(tmp_path, "Yac", [1, 91], SRC)
    ref = json.loads((SRC / "Yac" / "splits.json").read_text())["splits"]["s1"]
    assert {k: v for k, v in meta["s1"].items() if k != "files"} == ref
    assert meta["s91"]["sizes"] == {"train": 154, "val": 77, "test": 77, "trainval": 231}
    assert meta["s91"]["row_indices_sha256"] != ref["row_indices_sha256"]
    written = sorted(p.name for p in (tmp_path / "Yac" / "splits" / "s91").iterdir())
    assert written == sorted(f"{n}.{e}" for n in ("train", "val", "test", "trainval")
                             for e in ("csv", "dat"))  # fmt: skip
    assert not (tmp_path / "Yac" / "full.csv").exists()
    assert not (tmp_path / "Yac" / "splits" / "splits.json").exists()


def test_seeds_and_existing_files(tmp_path):
    assert es.parse_seeds("91-100") == list(range(91, 101))
    assert es.parse_seeds("101") == [101]
    with pytest.raises(ValueError, match="written by sr_datasets.py"):
        es.parse_seeds("85-95")
    assert es.write_splits(tmp_path, "Yac", [91], SRC)["s91"]["files"] == "new"
    assert es.write_splits(tmp_path, "Yac", [91], SRC)["s91"]["files"] == "same"
    (tmp_path / "Yac" / "splits" / "s91" / "test.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="exists with other content"):
        es.write_splits(tmp_path, "Yac", [91], SRC)
