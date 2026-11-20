"""Server rounds 11, 12, 13, 14 and 15: the fresh splits 31-40, 41-50, 51-60, 61-70 and 71-80
(``SERVER-ROUND11-PREREGISTRATION.md`` to ``SERVER-ROUND15-PREREGISTRATION.md``, "Rules").

``tools/gomea/sr_datasets.py`` writes splits 1-80, with the same split rule as before. Splits
1-70 must stay byte-identical: the committed ``data/gomea-sr/<Abbr>/splits.json`` without the
entries s31-s80 is the file of commit 5da6731, without the entries s41-s80 it is the file of
commit ae876e7 (round 11), without the entries s51-s80 it is the file of commit cb1ccfe
(round 12), without the entries s61-s80 it is the file of commit ad0c8d8 (round 13), and without
the entries s71-s80 it is the file of commit 3b18ca6 (round 14). The rebuilt split files of
s1-s30 of each committed dataset have the digest of a build at 5da6731, those of s31-s40 the
digest of a build at ae876e7, those of s41-s50 the digest of a build at cb1ccfe, those of s51-s60
the digest of a build at ad0c8d8, and those of s61-s70 the digest of a build at 3b18ca6.
``tools/server/setup.sh`` verifies the 640 split files (80 splits x 8 files) of each dataset.

Server round 15 also checks the column types of the new splits (``check_split_types.py``): for
each committed dataset and split 71-80, the validation and test tables have the input types of
the training table (``loadTable`` of pymoses and ``sr_eval.read_table``)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "gomea-sr"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import check_split_types  # noqa: E402
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
# SHA-256 of data/gomea-sr/<Abbr>/splits.json at commit cb1ccfe (splits 1-50, round 12).
ROUND12_SPLITS_JSON = {
    "Air": "7a117514e160deb19e41bf81b608b38ec4938bbddbfac948a3b173ceec51db35",
    "Bos": "2c18cc29cd69b8f35f1cf8343f927c9268e12a31190d3783e716a66e8ebcf488",
    "Con": "494113495a7c96aed4872bca3124d6936fa25def88d75eccad4a8a205d2e4b8b",
    "Dow": "47f35bdbe9ce012868d0d0b70adc322129629a270112b3fd6ec86c64b24015a0",
    "EnC": "f00809bbc8c9daf86dfcbb67d72fe3ddf10c7e83e350d392976a415f660b9146",
    "EnH": "35d4369f2fe0d75ab5243c1bf94294d9eb1e194995d5e7ae24b840ddc0782d91",
    "Tow": "5272ea7aa0aaecc8539547e10a2177a4a258cabc0fece3517257333e749b8333",
    "WiR": "b316d9c7c155e9302e9a5727d9bb6d6b6e4dfd95ad0cb4ae4e714ec51135c543",
    "WiW": "3ef1cdc12f898b08307240e79f7e4432ea18401bc60ea7e0d163a464131a88a5",
    "Yac": "a1a34f440752e1d6ad2a855ae1d96f84c2edaa0af086e4bf831b30ac827bfa6b",
}
# SHA-256 of data/gomea-sr/<Abbr>/splits.json at commit ad0c8d8 (splits 1-60, round 13).
ROUND13_SPLITS_JSON = {
    "Air": "8a62f913c50239de52d3e611016fb70888806924c367fcf278850a81a7898d01",
    "Bos": "f22e4cc08e34e5c15299270b295d6cae2202b62a2533671e2812a7022685d34a",
    "Con": "b5ba4202c0cb9d66e6330909edadc57de846875aa115edc473803a48297a3b86",
    "Dow": "ac40df7fbdc4f0f1def429955aa0d35929233e07b859870fd923b814a2c8ff62",
    "EnC": "e2a2ed1f87550368af69133f7598c0b1bf92125b504788048482c5ab3064b299",
    "EnH": "9ebdafde65529fddfe6fdf4c124983b05dff4a2d4882db183987deb48339b1eb",
    "Tow": "5f1af8c19f4da08625e670535440c4aa981a4d54f7d636770b61f0b461af6f3b",
    "WiR": "4ef88593a09f4acfa359c5ed863ac1bd72b3623cc33c2628f612a945929b44b8",
    "WiW": "c4c34764d2cff72f9ac65f226837eacae252ca2d78bf625a1ef841254c30c717",
    "Yac": "006085306c5879c1c70e6661cf0fa7359de03e8e16b9b73e85c7c552d8f685e4",
}
# SHA-256 of data/gomea-sr/<Abbr>/splits.json at commit 3b18ca6 (splits 1-70, round 14).
ROUND14_SPLITS_JSON = {
    "Air": "db1b20f96214572b00e542f4d8c4cf0fa33ec541f1e00e5dd21aef95a7992c57",
    "Bos": "337616740ac3979a0e2fb1a65b593b8c5f3fd8a0f089c76536def4d436c7aee6",
    "Con": "e60b06930e09094cc8f02ebfccd41163c9d46688f79159d4844fc945a8ce02af",
    "Dow": "85f793f22543ffe243119e1854d913a53f465635a5d12ab4db6e7e565a411103",
    "EnC": "5fe53773b3870d934c8cc972e5ff0bdb51eeac1cd431a103aedcb6057efd4c29",
    "EnH": "17ed9f8aa1d171f9ea147c29965621f5d649212f44b370a805eec855d2ceb0dc",
    "Tow": "02ef0376cf9fa847e6a15d651be8e18cfa525c636d1cf504489ee52367df4124",
    "WiR": "0e1a57af726245813270ad039735592f37841a04c67e91758d4b83b7aa57676e",
    "WiW": "b5092efa712fde6063d780371a3d7751b23f4af165907b3acc65ca3251d12ddb",
    "Yac": "5d37f129955d7414bf54e9b06962cc588b0065bc84c4d4835466b9853a5d0ccf",
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


# _digest(<build>/<Abbr>, range(41, 51)) of a build with sr_datasets.py at commit cb1ccfe
# (round 12, splits 1-50; ``sr_datasets.build(OUT, ABBR, True, from_full=data/gomea-sr)``).
ROUND12_FILES = {
    "Air": "15cb9c9755bcd13aca09ece3eceb2d06e1acc4aaa4801d671cf02a9d7553085e",
    "Bos": "1209a6e6ff77ef97a668f84cb1b72144b4da37ab16075029ce3d226cc95ba26f",
    "Con": "dc3b3e98e0a0cae6612574cd88bcc69a9df0866017f75b9b8eaaf37d00474170",
    "EnC": "362041975d2accec9b6f1c2c86becea40657df8d44dd546f027c4f83d5f1c2b3",
    "EnH": "850c95b63e3dc0c026ad7e4d994671f4ed735b5504fdbd6a205a3b399e7312dd",
    "WiR": "d54d004a0c2ac5f32cb24a3fb06523d0f69fd40ef2022db656a54267e40d8298",
    "WiW": "252144dec993f8b23622506a9dd4931e599c6f8a62fa07a546ce1ebc3eb29a33",
    "Yac": "b89859aa2c55be32fb0e049405f4bb5ee02f75e1cf78dde04adf59c046f55954",
}


# _digest(<build>/<Abbr>, range(51, 61)) of a build with sr_datasets.py at commit ad0c8d8
# (round 13, splits 1-60; ``sr_datasets.build(OUT, ABBR, True, from_full=data/gomea-sr)``).
ROUND13_FILES = {
    "Air": "518e0dbce97d71f5e1def29ba0ffa0fd92fd69a058d6cd0cf177aa9ce855eb55",
    "Bos": "26e77f2a58723dd9c4ab0ae868d4a461f72f97e1d036595b2da3eaee6745a223",
    "Con": "dce4c8078cc2e79c52e5c7f50d2b74c0d8f4ae08015918147912c4c8ad892027",
    "EnC": "63777d4d06b5a1c1cb89d69a9703d429f96732230d28aac9af9875af6f031ec3",
    "EnH": "24255112309a3c5d6d2b7a0a352b697580da943e60b42be577dcc4682eb6db8a",
    "WiR": "73214f869ae7fb6ccd59ff755906192ad17d4eac63ca0746f4d976a0c7213c3d",
    "WiW": "65bd5442a783cc9c87fdc66034c717046fb3e2018281596f2ece6896a387771e",
    "Yac": "ba595381af34fb6cb61e8250044101fc518309de4c06b5cdb997b7d25c3941ae",
}


# _digest(<build>/<Abbr>, range(61, 71)) of a build with sr_datasets.py at commit 3b18ca6
# (round 14, splits 1-70; ``sr_datasets.build(OUT, ABBR, True, from_full=data/gomea-sr)``).
ROUND14_FILES = {
    "Air": "4274bdfdc59336826fafccb53bb5f1e0ce104c0e5c168710fce2b044e305b172",
    "Bos": "b07edca0f4133d151879a0b28198acba81b03b8e588e187b1eee9a2c555b459a",
    "Con": "04c1783357f4d8ac69ddc5ae2d7b0fd178bf98e6badc3992742c53e2e7556f87",
    "EnC": "6cfb5c2df04a1389717f436ab8f8412c721a28b855118d1e5094776500e2f252",
    "EnH": "f55b0473a1679bae0cb75e62722aff89bb7c595b8fa6e2f8bb3b07bb9b1c777c",
    "WiR": "f4d17ee92ae279d944ac8d253c6d52a56533547934ff37e93675f669738abbd0",
    "WiW": "06f4ca223639e0789b816fe9d1eb19979e29c5a7a3e484245c9a959824277cec",
    "Yac": "cd868894d96bc8cf762c9eeb472a2dc526304607eb6b39c41446b0bbb6eb269b",
}


def _digest(ddir: Path, seeds) -> str:
    """SHA-256 over the name and the bytes of every split file of ``seeds``, in order."""
    h = hashlib.sha256()
    for seed in seeds:
        for f in FILES:
            h.update(f"s{seed}/{f}\n".encode())
            h.update((ddir / "splits" / f"s{seed}" / f).read_bytes())
    return h.hexdigest()


def test_split_seeds_are_1_to_80():
    assert sd.SEEDS == range(1, 81)


def _sha_without_splits_above(d: dict, last: int) -> str:
    d = {**d, "splits": {k: v for k, v in d["splits"].items() if int(k[1:]) <= last}}
    return hashlib.sha256((json.dumps(d, indent=1) + "\n").encode()).hexdigest()


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_committed_splits_json_keeps_splits_1_to_70(abbr):
    d = json.loads((SRC / abbr / "splits.json").read_text())
    assert list(d["splits"]) == [f"s{k}" for k in range(1, 81)]
    assert _sha_without_splits_above(d, 30) == OLD_SPLITS_JSON[abbr]
    assert _sha_without_splits_above(d, 40) == ROUND11_SPLITS_JSON[abbr]
    assert _sha_without_splits_above(d, 50) == ROUND12_SPLITS_JSON[abbr]
    assert _sha_without_splits_above(d, 60) == ROUND13_SPLITS_JSON[abbr]
    assert _sha_without_splits_above(d, 70) == ROUND14_SPLITS_JSON[abbr]


@pytest.mark.parametrize("abbr", list(sd.DATASETS))
def test_splits_31_to_80_are_new_rows_orders(abbr):
    """Each fresh split has its own row order: no test set of s31-s80 is one of another
    split."""
    s = json.loads((SRC / abbr / "splits.json").read_text())["splits"]
    tests = [s[f"s{k}"]["row_indices_sha256"]["test"] for k in range(1, 81)]
    assert len(set(tests)) == 80
    for k in (31, 41, 51, 61, 71):
        assert s[f"s{k}"]["rng"] == s["s1"]["rng"].replace(f'"{abbr}-s1"', f'"{abbr}-s{k}"')
        assert f'"{abbr}-s{k}"' in s[f"s{k}"]["rng"]
    assert all(s[f"s{k}"]["sizes"] == s["s1"]["sizes"] for k in range(31, 81))


@pytest.mark.parametrize("abbr", COMMITTED)
def test_rebuilt_split_files_1_to_70_are_byte_identical(abbr, tmp_path):
    info = sd.build(tmp_path, abbr, True, from_full=SRC)
    assert info["splits_json_matches"] == str(SRC / abbr / "splits.json")
    assert _digest(tmp_path / abbr, range(1, 31)) == OLD_FILES[abbr]
    assert _digest(tmp_path / abbr, range(31, 41)) == ROUND11_FILES[abbr]
    assert _digest(tmp_path / abbr, range(41, 51)) == ROUND12_FILES[abbr]
    assert _digest(tmp_path / abbr, range(51, 61)) == ROUND13_FILES[abbr]
    assert _digest(tmp_path / abbr, range(61, 71)) == ROUND14_FILES[abbr]
    for seed in range(71, 81):
        assert sorted(p.name for p in (tmp_path / abbr / "splits" / f"s{seed}").iterdir()) == \
            sorted(FILES)
    # server round 15: the new splits have the column types of their training table
    report = check_split_types.check(tmp_path, [abbr], range(71, 81))
    assert report["splits_checked"] == 10
    assert report["passed"] and report["val_differ"] == [] and report["test_differ"] == []


def test_the_type_check_finds_a_validation_table_with_other_types(tmp_path):
    """check_split_types.py reports a split whose validation table has a boolean column where
    the training table has a numeric one (made-up tables, no dataset rows)."""
    d = tmp_path / "X" / "splits" / "s1"
    d.mkdir(parents=True)
    (d / "train.csv").write_text("x1,x2,y\n" + "".join(f"{i}.5,{i % 2},{i}\n" for i in range(30)))
    (d / "test.csv").write_text((d / "train.csv").read_text())
    (d / "val.csv").write_text("x1,x2,y\n" + "".join(f"{i % 2},{i % 2},{i}\n" for i in range(30)))
    report = check_split_types.check(tmp_path, ["X"], [1])
    assert not report["passed"] and report["val_differ"] == ["X/s1"]
    assert report["test_differ"] == []
    assert report["details"]["X/s1"]["bool_cols"] == {"train": [1], "val": [0, 1], "test": [1]}


def test_setup_verifies_the_files_of_80_splits():
    text = (REPO / "tools" / "server" / "setup.sh").read_text()
    body = text.split("dataset_ok() {", 1)[1].split("\n}\n", 1)[0]
    assert f'-type f | wc -l)" -eq {8 * len(sd.SEEDS)} ]' in body
    assert 'cmp -s "$D/$a/splits/splits.json" "data/gomea-sr/$a/splits.json"' in body
