"""NOT A PORT, backlog 15.4 (D1b): ``tools/gomea/extra_splits.py`` writes split seeds above 90
with the scheme and the format of ``sr_datasets.py``.

* Seed 1 written by the tool has the meta of the committed ``data/gomea-sr/Yac/splits.json``
  (sizes and SHA-256 of the row indices), so the tool and ``sr_datasets.py`` give the same rows.
* The tool refuses seeds 1-90, never writes ``full.csv`` or ``splits.json``, leaves a file with
  the same content as it is and refuses to change a file with other content.
* A refusal writes nothing (the RV16 review of round 17, G5): it checks every target file of
  every dataset and seed before it writes one. The files and the stdout are the same as those
  of the tool at b6d7618 (the commit before G5).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "gomea-sr"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import extra_splits as es  # noqa: E402

PRE_G5 = "b6d7618"  # the commit before the G5 fix (the RV16 review of round 17)


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


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_a_refusal_writes_nothing(tmp_path):
    """G5: an old test.csv in s95 stops the tool before it writes a file of any seed."""
    s95 = tmp_path / "Yac" / "splits" / "s95"
    s95.mkdir(parents=True)
    (s95 / "test.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="s95/test.csv exists with other content"):
        es.write_splits(tmp_path, "Yac", list(range(91, 101)), SRC)
    assert _tree(tmp_path) == ["Yac", "Yac/splits", "Yac/splits/s95", "Yac/splits/s95/test.csv"]
    assert (s95 / "test.csv").read_text() == "x\n"
    # the command line: a refusal for the second dataset writes nothing of the first
    root = tmp_path / "m"
    (root / "Yac" / "splits" / "s92").mkdir(parents=True)
    (root / "Yac" / "splits" / "s92" / "val.dat").write_text("x\n")
    with pytest.raises(SystemExit, match="s92/val.dat exists with other content"):
        es.main([str(root), "Con", "Yac", "--seeds", "91-92", "--from-full", str(SRC)])
    assert _tree(root) == ["Yac", "Yac/splits", "Yac/splits/s92", "Yac/splits/s92/val.dat"]


def test_the_output_is_byte_identical_to_pre_g5(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for n in ("extra_splits.py", "sr_datasets.py"):
        r = subprocess.run(["git", "show", f"{PRE_G5}:tools/gomea/{n}"], cwd=REPO,
                           capture_output=True)
        if r.returncode != 0:  # a missing reference must never let the test pass (no skip)
            pytest.fail(f"git show {PRE_G5}:tools/gomea/{n} failed: {r.stderr.decode()!r}; "
                        f"use a clone with the full history")
        (old / n).write_bytes(r.stdout)
    got = {}
    for tag, script in (("old", old / "extra_splits.py"),
                        ("new", REPO / "tools" / "gomea" / "extra_splits.py")):
        root = tmp_path / f"out-{tag}"
        outs = []
        for _ in range(2):  # the second run finds every file with the same content
            r = subprocess.run([sys.executable, "-I", str(script), str(root), "Yac", "Con",
                                "--seeds", "91-93", "--from-full", str(SRC)],
                               capture_output=True, text=True, env=dict(os.environ))
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(r.stdout)
        files = {p.relative_to(root).as_posix(): p.read_bytes()
                 for p in root.rglob("*") if p.is_file()}
        got[tag] = (outs, files)
    assert got["new"] == got["old"]
    outs, files = got["new"]
    assert len(files) == 2 * 3 * 8 and '"files": "new"' in outs[0] and '"files": "same"' in outs[1]
