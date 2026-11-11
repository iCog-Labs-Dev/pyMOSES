"""Server round 11: the fresh splits 31-40 (``SERVER-ROUND11-PREREGISTRATION.md``, "Rules").

``tools/gomea/sr_datasets.py`` writes splits 1-40, with the same split rule as before. Splits
1-30 must stay byte-identical: the committed ``data/gomea-sr/<Abbr>/splits.json`` without the
entries s31-s40 is the file of commit 5da6731, and the rebuilt split files of s1-s30 of each
committed dataset have the digest of a build at 5da6731. ``tools/server/setup.sh`` verifies
the 320 split files (40 splits x 8 files) of each dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "gomea-sr"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import sr_datasets as sd  # noqa: E402

COMMITTED = ["Air", "Bos", "Con", "EnC", "EnH", "WiR", "WiW", "Yac"]  # Dow, Tow: no full.csv
FILES = [f"{name}.{ext}" for name in ("train", "val", "test", "trainval")
         for ext in ("csv", "dat")]

# SHA-256 of data/gomea-sr/<Abbr>/splits.json at commit 5da6731 (splits 1-30).
OLD_SPLITS_JSON = {
    "Air": "1bf3ed05547129abc730985aeebb707be181ac04782e1e04653ebe2a95d3db3b",
    "Bos": "360fb3c043c1b11bdb0f0ef6519f72366d02d9b61fc7bc219b3bd23085b01c0e",
    "Con": "69f56993231fdcce00ce16d512808783933a70fba12e87eb694047231921da69",
    "Dow": "bc7981fcef452f536faf7e9c44aad4859b3bf436aa066fba1070b20f1955fccd",
    "EnC": "4882493d62ebc5ae44b4d16cae1b356881c030bb3b2f963b2d5d0fb0295facf5",
    "EnH": "fe84c0170f8bc3b4aac7eba88b4adf0c577885522878df3afba5a0f3b6c80350",
    "Tow": "1f7db6ba167ce662fc025582a794f4bed4a7554c62bcc2390e082237dbec57c5",
    "WiR": "e45e334c470194ca51fcb45b740c56daf9c8c29d747bc53f524915420c8e1fb1",
    "WiW": "44fe24a23f505e1217629fe8e35fcef4dfedc8c21894b734649451cadb8c9609",
    "Yac": "242c77cdc185b2a30fcf5bdfd3936158cf8e40b162c381f67c594f78a64004be",
}
# _digest(<build>/<Abbr>, range(1, 31)) of a build with sr_datasets.py at commit 5da6731
# (``sr_datasets.py OUT ABBR --from-full data/gomea-sr``).
OLD_FILES = {
    "Air": "21d49f94f544f910c42ff101390e6abe85fdf6cb5767d4bb5460effa76f0188e",
    "Bos": "1762623f2f9d4ffe913d2188e6f12d70fca81e8766f91175dc773883a7870aae",
    "Con": "e4e2872318e6db968bc5ccec76c00629ca9a79b207d81f8b55abf45560c1f455",
    "EnC": "9fa41cb1187efcaa0463841373434991fba5af507d6c8b61d91b8ca97c8dbd4e",
    "EnH": "5afe7910b1b85b05daf2592dc334f89e4f8ef4535930d7718abecbb8a4408cdd",
    "WiR": "badb3f2eac6d1b90cba5ecc3d553edf9c16cf7c1996f9557d3d74bbe8086f7ba",
    "WiW": "18daa7f98e879d1fc3d1be24b4ae9dc90feecc0350022e92946fae6223a2183b",
    "Yac": "234ca24ee652af2c469cb2f94314e8c76f53e0674df159279f7258337aeb72fd",
}


def _digest(ddir: Path, seeds) -> str:
    """SHA-256 over the name and the bytes of every split file of ``seeds``, in order."""
    h = hashlib.sha256()
    for seed in seeds:
        for f in FILES:
            h.update(f"s{seed}/{f}\n".encode())
            h.update((ddir / "splits" / f"s{seed}" / f).read_bytes())
    return h.hexdigest()


def test_split_seeds_are_1_to_40():
    assert sd.SEEDS == range(1, 41)


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_committed_splits_json_keeps_splits_1_to_30(abbr):
    d = json.loads((SRC / abbr / "splits.json").read_text())
    assert list(d["splits"]) == [f"s{k}" for k in range(1, 41)]
    d["splits"] = {k: v for k, v in d["splits"].items() if int(k[1:]) <= 30}
    old = json.dumps(d, indent=1) + "\n"
    assert hashlib.sha256(old.encode()).hexdigest() == OLD_SPLITS_JSON[abbr]


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_splits_31_to_40_are_new_rows_orders(abbr):
    """Each fresh split has its own row order: no test set of s31-s40 is one of s1-s30."""
    s = json.loads((SRC / abbr / "splits.json").read_text())["splits"]
    tests = [s[f"s{k}"]["row_indices_sha256"]["test"] for k in range(1, 41)]
    assert len(set(tests)) == 40
    assert s["s31"]["rng"] == s["s1"]["rng"].replace(f'"{abbr}-s1"', f'"{abbr}-s31"')
    assert f'"{abbr}-s31"' in s["s31"]["rng"]
    assert all(s[f"s{k}"]["sizes"] == s["s1"]["sizes"] for k in range(31, 41))


@pytest.mark.parametrize("abbr", COMMITTED)
def test_rebuilt_split_files_1_to_30_are_byte_identical(abbr, tmp_path):
    info = sd.build(tmp_path, abbr, True, from_full=SRC)
    assert info["splits_json_matches"] == str(SRC / abbr / "splits.json")
    assert _digest(tmp_path / abbr, range(1, 31)) == OLD_FILES[abbr]
    for seed in range(31, 41):
        assert sorted(p.name for p in (tmp_path / abbr / "splits" / f"s{seed}").iterdir()) == \
            sorted(FILES)


def test_setup_verifies_the_files_of_40_splits():
    text = (REPO / "tools" / "server" / "setup.sh").read_text()
    body = text.split("dataset_ok() {", 1)[1].split("\n}\n", 1)[0]
    assert f'-type f | wc -l)" -eq {8 * len(sd.SEEDS)} ]' in body
    assert 'cmp -s "$D/$a/splits/splits.json" "data/gomea-sr/$a/splits.json"' in body
