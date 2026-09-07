"""Parity and edge tests for discrete and continuous field storage."""

import json
import struct

import pytest

from pymetta.moses.representation.field_set import (
    contin_spec,
    disc_spec,
    field_set,
    nbits_to_pack,
)
from pymetta.moses.representation.instance import instance


def _layout(fs: field_set) -> dict[str, object]:
    return {
        "canonical_contin": [
            {
                "kind": "contin",
                "mean": item.mean,
                "step_size": item.step_size,
                "expansion": item.expansion,
                "depth": item.depth,
            }
            for item in fs.contin()
        ],
        "canonical_disc": [
            {"kind": "disc", "multy": item.multy} for item in fs.disc_and_bit()
        ],
        "contin_raw_offsets": [
            fs.contin_to_raw_idx(idx) for idx in range(len(fs.contin()))
        ],
        "fields": [
            {
                "width": item.width,
                "major_offset": item.major_offset,
                "minor_offset": item.minor_offset,
            }
            for item in fs.fields()
        ],
        "packed_width": fs.packed_width(),
        "raw_size": fs.raw_size(),
        "dim_size": fs.dim_size(),
        "starts": {
            "contin": [fs.begin_contin_raw_idx(), fs.end_contin_raw_idx()],
            "disc": [fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()],
            "bit": [fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()],
        },
        "counts": {
            "contin_raw": fs.n_contin_fields(),
            "disc": fs.n_disc_fields(),
            "bits": fs.n_bits(),
        },
    }


def _spec_from_json(value: dict[str, object]):
    if value["kind"] == "disc":
        return disc_spec(value["multy"])
    return contin_spec(
        value["mean"], value["step_size"], value["expansion"], value["depth"]
    )


def _double_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def _assert_spec_shape(value: dict[str, object]) -> None:
    if value["kind"] == "disc":
        assert set(value) == {"kind", "multy"}
    else:
        assert value["kind"] == "contin"
        assert set(value) == {"kind", "mean", "step_size", "expansion", "depth"}


def test_cpp_field_set_layout_oracle(fixture_dir):
    fixture = json.loads((fixture_dir / "field_set.json").read_text())
    assert set(fixture) == {
        "abi",
        "nbits_to_pack",
        "layouts",
        "continuous",
        "set_continuous",
        "pack_cases",
        "hamming_cases",
        "merge_cases",
    }
    assert fixture["abi"] == {"sizeof_packed_t": 8, "bits_per_packed_t": 64}
    assert fixture["nbits_to_pack"] == [
        {"multy": multy, "width": nbits_to_pack(multy)}
        for multy in (1, 2, 3, 4, 5, 16, 17, 50, 257)
    ]
    assert [row["name"] for row in fixture["layouts"]] == [
        "empty",
        "65-bits",
        "33-ternary",
        "straddle-following",
        "same-depth-continuous-order",
        "canonical-mixed",
    ]
    assert [len(row["writes"]) for row in fixture["layouts"]] == [1] * 6
    layouts = {row["name"]: row for row in fixture["layouts"]}
    assert layouts["empty"]["input_specs"] == []
    assert layouts["65-bits"]["input_specs"] == [
        {"kind": "disc", "multy": 2}
    ] * 65
    assert layouts["33-ternary"]["input_specs"] == [
        {"kind": "disc", "multy": 3}
    ] * 33
    assert layouts["straddle-following"]["input_specs"] == [
        {"kind": "contin", "mean": 0, "step_size": 1, "expansion": 2, "depth": 5},
        *([{"kind": "disc", "multy": 5}] * 15),
    ]
    contin_ties = [
        {"kind": "contin", "mean": 0, "step_size": 1, "expansion": 2, "depth": 4},
        {"kind": "contin", "mean": 0, "step_size": 1, "expansion": 3, "depth": 4},
        {"kind": "contin", "mean": 0, "step_size": 2, "expansion": 3, "depth": 4},
        {"kind": "contin", "mean": 1, "step_size": 2, "expansion": 3, "depth": 4},
    ]
    assert layouts["same-depth-continuous-order"]["input_specs"] == contin_ties
    assert layouts["same-depth-continuous-order"]["canonical_contin"] == list(reversed(contin_ties))
    assert layouts["canonical-mixed"]["input_specs"] == [
        {"kind": "disc", "multy": 2},
        {"kind": "contin", "mean": 0, "step_size": 1, "expansion": 2, "depth": 3},
        {"kind": "disc", "multy": 9},
        {"kind": "contin", "mean": 1, "step_size": 0.5, "expansion": 3, "depth": 5},
        {"kind": "disc", "multy": 3},
        {"kind": "disc", "multy": 1},
        {"kind": "disc", "multy": 9},
    ]
    assert {name: [write["raw_ops"] for write in row["writes"]]
            for name, row in layouts.items()} == {
        "empty": [[]],
        "65-bits": [[[63, 1], [64, 1]]],
        "33-ternary": [[[31, 2], [32, 1]]],
        "straddle-following": [[[18, 15], [19, 9]]],
        "same-depth-continuous-order": [[]],
        "canonical-mixed": [[[0, 1], [7, 2], [8, 9], [12, 1]]],
    }

    for row in fixture["layouts"]:
        assert set(row) == {
            "name", "input_specs", "canonical_contin", "canonical_disc",
            "contin_raw_offsets", "fields", "packed_width", "raw_size",
            "dim_size", "starts", "counts", "writes",
        }
        for spec_json in [
            *row["input_specs"], *row["canonical_contin"], *row["canonical_disc"]
        ]:
            _assert_spec_shape(spec_json)
        assert all(set(item) == {"raw_ops", "words_hex", "raw"} for item in row["writes"])
        fs = field_set(_spec_from_json(item) for item in row["input_specs"])
        assert _layout(fs) == {
            key: row[key]
            for key in (
                "canonical_contin",
                "canonical_disc",
                "contin_raw_offsets",
                "fields",
                "packed_width",
                "raw_size",
                "dim_size",
                "starts",
                "counts",
            )
        }
        for write in row["writes"]:
            packed = instance(fs.packed_width())
            for idx, value in write["raw_ops"]:
                fs.set_raw(packed, idx, value)
            assert [f"0x{word:016x}" for word in packed] == write["words_hex"]
            assert list(fs.iter_raw(packed)) == write["raw"]


def test_cpp_continuous_codec_oracle(fixture_dir):
    fixture = json.loads((fixture_dir / "field_set.json").read_text())
    assert len(fixture["continuous"]) == 7
    assert len(fixture["set_continuous"]) == 5
    assert [row["name"] for row in fixture["continuous"]] == [
        "all-stop", "left-stop", "left-left-right", "right-right-left",
        "stop-ignores-tail", "right-left-stop", "contracting-expansion",
    ]
    assert [row["name"] for row in fixture["set_continuous"]] == [
        "mean", "one-left", "two-left", "far-right", "tie-keeps-shallow",
    ]
    base_spec = {"kind": "contin", "mean": 0, "step_size": 1,
                 "expansion": 2, "depth": 5}
    contracting_spec = {"kind": "contin", "mean": 0, "step_size": 0.5,
                        "expansion": 0.5, "depth": 4}
    assert [(row["name"], row["spec"], row["raw"])
            for row in fixture["continuous"]] == [
        ("all-stop", base_spec, [0, 0, 0, 0, 0]),
        ("left-stop", base_spec, [1, 0, 0, 0, 0]),
        ("left-left-right", base_spec, [1, 1, 2, 0, 0]),
        ("right-right-left", base_spec, [2, 2, 1, 0, 0]),
        ("stop-ignores-tail", base_spec, [1, 2, 0, 1, 2]),
        ("right-left-stop", base_spec, [2, 1, 0, 0, 0]),
        ("contracting-expansion", contracting_spec, [2, 0, 0, 0]),
    ]
    tie_spec = {"kind": "contin", "mean": 0, "step_size": 1,
                "expansion": 1, "depth": 2}
    assert [(row["name"], row["spec"], row["target"])
            for row in fixture["set_continuous"]] == [
        ("mean", base_spec, 0), ("one-left", base_spec, -1),
        ("two-left", base_spec, -2), ("far-right", base_spec, 100),
        ("tie-keeps-shallow", tie_spec, -1.5),
    ]
    for row in fixture["continuous"]:
        assert set(row) == {"name", "spec", "raw", "value_bits_hex", "length"}
        _assert_spec_shape(row["spec"])
        cs = _spec_from_json(row["spec"])
        fs = field_set(cs, 1)
        packed = fs.pack(row["raw"])
        assert _double_bits_hex(fs.get_contin(packed, 0)) == row["value_bits_hex"]
        assert fs.contin_length(packed, 0) == row["length"]

    for row in fixture["set_continuous"]:
        assert set(row) == {"name", "spec", "target", "raw_after", "decoded_bits_hex"}
        _assert_spec_shape(row["spec"])
        cs = _spec_from_json(row["spec"])
        fs = field_set(cs, 1)
        packed = instance(fs.packed_width())
        fs.set_contin(packed, 0, row["target"])
        assert list(fs.iter_raw(packed)) == row["raw_after"]
        assert _double_bits_hex(fs.get_contin(packed, 0)) == row["decoded_bits_hex"]


def test_cpp_pack_hamming_and_merge_oracles(fixture_dir):
    fixture = json.loads((fixture_dir / "field_set.json").read_text())
    assert [row["name"] for row in fixture["pack_cases"]] == [
        "33-ternary-rollover", "straddle-following"
    ]
    assert all(set(row) == {"name", "input_specs", "raw", "words_hex"}
               for row in fixture["pack_cases"])
    rollover, classic_straddle = fixture["pack_cases"]
    base_spec = {"kind": "contin", "mean": 0, "step_size": 1,
                 "expansion": 2, "depth": 5}
    assert rollover["input_specs"] == [{"kind": "disc", "multy": 3}] * 33
    assert rollover["raw"] == [0] * 31 + [2, 1]
    assert classic_straddle["input_specs"] == [base_spec, *([{"kind": "disc", "multy": 5}] * 15)]
    assert classic_straddle["raw"] == [0] * 18 + [15, 9]
    rollover_fs = field_set(_spec_from_json(item) for item in rollover["input_specs"])
    assert [f"0x{word:016x}" for word in rollover_fs.pack(rollover["raw"])] == rollover["words_hex"]

    # The classic loop advances only at offset == 64 and the measured x86-64
    # ABI masks the following over-wide shift count modulo 64.
    straddle_fs = field_set(
        _spec_from_json(item) for item in classic_straddle["input_specs"]
    )
    packed_words = straddle_fs.pack(classic_straddle["raw"])
    assert [f"0x{word:016x}" for word in packed_words] == classic_straddle["words_hex"]
    assert packed_words == [0xC000000000000024, 0]

    assert fixture["hamming_cases"] == [{
        "name": "raw-field-distance", "lhs_raw": [1, 0, 2],
        "rhs_raw": [0, 1, 2], "distance": 2,
    }]
    hamming = fixture["hamming_cases"][0]
    fs = field_set(disc_spec(3), 3)
    assert fs.hamming_distance(fs.pack(hamming["lhs_raw"]), fs.pack(hamming["rhs_raw"])) == 2

    assert fixture["merge_cases"] == [{
        "name": "copy-reference-differences", "base_raw": [0, 1, 2],
        "reference_raw": [0, 2, 2], "target_before_raw": [2, 0, 1],
        "target_after_raw": [2, 2, 1],
    }]
    merge = fixture["merge_cases"][0]
    target = fs.pack(merge["target_before_raw"])
    fs.merge_instance(target, fs.pack(merge["base_raw"]), fs.pack(merge["reference_raw"]))
    assert list(fs.iter_raw(target)) == merge["target_after_raw"]


def test_continuous_and_discrete_comparators_are_descending():
    assert sorted([disc_spec(2), disc_spec(9), disc_spec(3)]) == [
        disc_spec(9),
        disc_spec(3),
        disc_spec(2),
    ]
    specs = [
        contin_spec(0.0, 1.0, 2.0, 4),
        contin_spec(0.0, 1.0, 3.0, 4),
        contin_spec(0.0, 2.0, 3.0, 4),
        contin_spec(1.0, 2.0, 3.0, 4),
        contin_spec(0.0, 1.0, 2.0, 5),
    ]
    assert sorted(specs) == [specs[4], specs[3], specs[2], specs[1], specs[0]]


def test_width_one_classifies_multiplicity_one_and_two_as_bits():
    fs = field_set([disc_spec(1), disc_spec(2), disc_spec(3)])
    assert [item.multy for item in fs.disc_and_bit()] == [3, 2, 1]
    assert fs.n_disc_fields() == 1
    assert fs.n_bits() == 2


def test_field_set_copy_rebuilds_the_same_canonical_layout():
    original = field_set(
        [disc_spec(2), contin_spec(1.0, 0.5, 2.0, 3), disc_spec(3)]
    )
    copied = field_set(original)
    assert copied == original
    assert copied.fields() == original.fields()
    assert copied is not original


def test_set_raw_masks_value_and_preserves_neighbor():
    fs = field_set(disc_spec(3), 2)
    packed = instance(fs.packed_width())
    fs.set_raw(packed, 1, 2)
    fs.set_raw(packed, 0, 7)
    assert list(fs.iter_raw(packed)) == [3, 2]


def test_continuous_offsets_and_inverse_lookup():
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 3), contin_spec(0.0, 1.0, 2.0, 5)])
    assert fs.contin_to_raw_idx(0) == 0
    assert fs.contin_to_raw_idx(1) == 5
    assert [fs.raw_to_contin_idx(idx) for idx in range(8)] == [0] * 5 + [1] * 3
    with pytest.raises(IndexError):
        fs.raw_to_contin_idx(8)


def test_hamming_is_per_raw_field_and_validates_widths():
    fs = field_set(disc_spec(3), 2)
    lhs = instance(fs.packed_width())
    rhs = instance(fs.packed_width())
    fs.set_raw(rhs, 0, 3)
    assert fs.hamming_distance(lhs, rhs) == 1
    assert fs.count(rhs) == 1
    with pytest.raises(ValueError, match="requires"):
        fs.hamming_distance(lhs, instance(0))


def test_merge_instance_copies_only_reference_differences():
    fs = field_set(disc_spec(3), 3)
    base = fs.pack([0, 1, 2])
    reference = fs.pack([0, 2, 2])
    target = fs.pack([2, 0, 1])
    fs.merge_instance(target, base, reference)
    assert list(fs.iter_raw(target)) == [2, 2, 1]


def test_pack_ors_into_existing_output():
    fs = field_set(disc_spec(3), 2)
    packed = fs.pack([1, 0])
    fs.pack([0, 2], packed)
    assert list(fs.iter_raw(packed)) == [1, 2]


def test_continuous_tie_keeps_shallower_encoding():
    fs = field_set(contin_spec(0.0, 1.0, 1.0, 2), 1)
    packed = instance(fs.packed_width())
    fs.set_contin(packed, 0, -1.5)
    assert list(fs.iter_raw(packed)) == [contin_spec.Left, contin_spec.Stop]
    assert fs.get_contin(packed, 0) == -1.0


def test_invalid_continuous_direction_and_switch_are_rejected():
    fs = field_set(contin_spec(0.0, 1.0, 2.0, 1), 1)
    packed = fs.pack([3])
    with pytest.raises(ValueError, match="unknown"):
        fs.get_contin(packed, 0)
    with pytest.raises(ValueError, match="Left or Right"):
        contin_spec.switchLR(contin_spec.Stop)


def test_raw_field_straddle_reproduces_classic_single_word_truncation():
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 5), *([disc_spec(5)] * 14)])
    descriptor = fs.fields()[-1]
    assert (descriptor.major_offset, descriptor.minor_offset, descriptor.width) == (0, 62, 4)
    packed = instance(fs.packed_width())
    fs.set_raw(packed, fs.raw_size() - 1, 15)
    assert fs.get_raw(packed, fs.raw_size() - 1) == 3


def test_public_field_indexes_reject_negative_and_upper_bounds_without_mutation():
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2), disc_spec(3), disc_spec(2)])
    packed = fs.pack([1, 0, 2, 1])
    before = packed.copy()

    for idx in (-1, fs.raw_size()):
        with pytest.raises(IndexError):
            fs.get_raw(packed, idx)
        with pytest.raises(IndexError):
            fs.set_raw(packed, idx, 1)
        assert packed == before

    for idx in (-1, len(fs.contin())):
        with pytest.raises(IndexError):
            fs.get_contin(packed, idx)
        with pytest.raises(IndexError):
            fs.set_contin(packed, idx, 1.0)
        with pytest.raises(IndexError):
            fs.contin_to_raw_idx(idx)
        with pytest.raises(IndexError):
            fs.contin_length(packed, idx)
        assert packed == before

    for method in (fs.raw_to_contin_idx, fs.raw_to_disc_idx):
        for idx in (-1, fs.raw_size()):
            with pytest.raises(IndexError):
                method(idx)


def test_false_classic_begin_iterator_aliases_are_not_exposed():
    fs = field_set(disc_spec(2), 1)
    for name in ("begin_raw", "begin_disc", "begin_bit", "begin_contin"):
        assert not hasattr(fs, name)
