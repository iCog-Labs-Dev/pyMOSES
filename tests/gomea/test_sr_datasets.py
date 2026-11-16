"""Server rounds 11 and 12: the fresh splits 31-40 and 41-50 (``SERVER-ROUND11-PREREGISTRATION.md``
and ``SERVER-ROUND12-PREREGISTRATION.md``, "Rules").

``tools/gomea/sr_datasets.py`` writes splits 1-50, with the same split rule as before. Splits
1-40 must stay byte-identical: the committed ``data/gomea-sr/<Abbr>/splits.json`` without the
entries s31-s50 is the file of commit 5da6731, and without the entries s41-s50 it is the file
of commit ae876e7 (round 11). The rebuilt split files of s1-s30 of each committed dataset have
the digest of a build at 5da6731, and those of s31-s40 the digest of a build at ae876e7.
``tools/server/setup.sh`` verifies the 400 split files (50 splits x 8 files) of each dataset."""

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
# SHA-256 of data/gomea-sr/<Abbr>/splits.json at commit ae876e7 (splits 1-40, round 11).
ROUND11_SPLITS_JSON = {
    "Air": "6aa7ac984fc9ab8b831495ca1c7969de6020385ccebb3de07a7b726fde7b0a93",
    "Bos": "b0e0e37cf3e8f9ce18a3df3a4277636fd79eb7529c9ffd0904cfdd8b91108496",
    "Con": "f8b7292b18efadb438de05e7f748fb9c95c7554fa86a018c990b1e48b8bd78c4",
    "Dow": "b494bbcafd917c6ead355d2677a0860f6aaaec8bac5da582406843ce72c8ede1",
    "EnC": "5460738241b8bfb56f864dd1625bc3422aca4b269a17c4ffea6689d317dc4f76",
    "EnH": "6552f9545c83508a62caabd1cfd5b5b12fe077b9a5bbf8a7a287e90adefca779",
    "Tow": "01936cae5a41939ee8cc0eaaf327f6bd23d5c4b9cd85dc82779308369b1f237e",
    "WiR": "696216dfb34361059187d0b01aae5ef61a52705cc1e3d587fccfd0e1db81729c",
    "WiW": "e5eda579f6e851be623df5cc67f0788a0862cfad72338aa425c8d4b1c2078594",
    "Yac": "6ee427996d6da0f659498af7f5f19b485becb0c20c1af493d4ebde49eeb7a88a",
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


# _digest(<build>/<Abbr>, range(31, 41)) of a build with sr_datasets.py at commit ae876e7
# (round 11, splits 1-40; ``sr_datasets.build(OUT, ABBR, True, from_full=data/gomea-sr)``).
ROUND11_FILES = {
    "Air": "b25cbb02c57de7f33975b0de09204c6d82d860c5f9c1adaee108f71d5aed59d8",
    "Bos": "9c2e07aded91014230a245f813a3d2b9870f8ddbba4fc7ee2e7dfe17218e2ef6",
    "Con": "fa91e3b53bbc873f443b5cccea1a303b79876ac2ea833d10c7d063086817b5c9",
    "EnC": "2d7b46730ac789316830f08bc084a43b1a721b5c5bd9c9a2111784a5a13ce0d0",
    "EnH": "2030bdc9e7e09b5d6852e466d131407b5ed9dc55f707f3aa9f23cc14e3d14362",
    "WiR": "97728e8444d24e2602490997440ebc9875e320dc594331eba84120a91b59bba3",
    "WiW": "ec4e31fb8a289b91e0ee9d625b3035fdd6c928ebadf21bdad7f283d99015cedb",
    "Yac": "cf7a9761a6f1ac0d477275e44eed314c40a195ba8c26be3778754e4b3c39a232",
}


def _digest(ddir: Path, seeds) -> str:
    """SHA-256 over the name and the bytes of every split file of ``seeds``, in order."""
    h = hashlib.sha256()
    for seed in seeds:
        for f in FILES:
            h.update(f"s{seed}/{f}\n".encode())
            h.update((ddir / "splits" / f"s{seed}" / f).read_bytes())
    return h.hexdigest()


def test_split_seeds_are_1_to_50():
    assert sd.SEEDS == range(1, 51)


def _sha_without_splits_above(d: dict, last: int) -> str:
    d = {**d, "splits": {k: v for k, v in d["splits"].items() if int(k[1:]) <= last}}
    return hashlib.sha256((json.dumps(d, indent=1) + "\n").encode()).hexdigest()


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_committed_splits_json_keeps_splits_1_to_40(abbr):
    d = json.loads((SRC / abbr / "splits.json").read_text())
    assert list(d["splits"]) == [f"s{k}" for k in range(1, 51)]
    assert _sha_without_splits_above(d, 30) == OLD_SPLITS_JSON[abbr]
    assert _sha_without_splits_above(d, 40) == ROUND11_SPLITS_JSON[abbr]


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_splits_31_to_50_are_new_rows_orders(abbr):
    """Each fresh split has its own row order: no test set of s31-s50 is one of another
    split."""
    s = json.loads((SRC / abbr / "splits.json").read_text())["splits"]
    tests = [s[f"s{k}"]["row_indices_sha256"]["test"] for k in range(1, 51)]
    assert len(set(tests)) == 50
    for k in (31, 41):
        assert s[f"s{k}"]["rng"] == s["s1"]["rng"].replace(f'"{abbr}-s1"', f'"{abbr}-s{k}"')
        assert f'"{abbr}-s{k}"' in s[f"s{k}"]["rng"]
    assert all(s[f"s{k}"]["sizes"] == s["s1"]["sizes"] for k in range(31, 51))


@pytest.mark.parametrize("abbr", COMMITTED)
def test_rebuilt_split_files_1_to_40_are_byte_identical(abbr, tmp_path):
    info = sd.build(tmp_path, abbr, True, from_full=SRC)
    assert info["splits_json_matches"] == str(SRC / abbr / "splits.json")
    assert _digest(tmp_path / abbr, range(1, 31)) == OLD_FILES[abbr]
    assert _digest(tmp_path / abbr, range(31, 41)) == ROUND11_FILES[abbr]
    for seed in range(41, 51):
        assert sorted(p.name for p in (tmp_path / abbr / "splits" / f"s{seed}").iterdir()) == \
            sorted(FILES)


def test_setup_verifies_the_files_of_50_splits():
    text = (REPO / "tools" / "server" / "setup.sh").read_text()
    body = text.split("dataset_ok() {", 1)[1].split("\n}\n", 1)[0]
    assert f'-type f | wc -l)" -eq {8 * len(sd.SEEDS)} ]' in body
    assert 'cmp -s "$D/$a/splits/splits.json" "data/gomea-sr/$a/splits.json"' in body
