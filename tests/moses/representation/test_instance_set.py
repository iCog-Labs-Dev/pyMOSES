"""C++ oracle and aliasing tests for instance_set and demeID_t."""

import json
import struct

import pytest

from pymetta.moses.moses.types import composite_score, very_worst_score
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.instance_set import demeID_t, instance_set


def _hex(x: float) -> str:
    return "0x" + struct.pack(">f", x).hex()


def _fs(n: int = 6) -> field_set:
    return field_set(field_set.disc_spec(2), n)


# --- demeID_t ---------------------------------------------------------------


def test_demeID_default_is_string_zero_not_empty():
    assert demeID_t() == "0"
    assert isinstance(demeID_t(), str)


def test_demeID_formats_all_three_overloads():
    assert demeID_t(2) == "2"
    assert demeID_t(2, 1) == "2.1"
    assert demeID_t(2, 1, 3) == "2.1.SS-3"


def test_demeID_rejects_ss_deme_without_breadth_first():
    with pytest.raises(TypeError):
        demeID_t(2, ss_deme=3)


# --- construction -------------------------------------------------------


def test_n_fs_constructor_fills_n_null_instances_with_fresh_worst_scores():
    fs = _fs()
    deme = instance_set(3, fs)
    assert len(deme) == 3
    assert deme.fields() is fs
    assert deme.getID() == "0"
    assert deme.n_evals == 0
    assert deme.n_best_evals == 0
    for entry in deme:
        assert entry.first == instance(fs.packed_width())
        assert entry.second.get_score() == very_worst_score


def test_n_fs_constructor_gives_every_slot_an_independent_score():
    # C++-measured in containers.json's "default_fill": mutating slot 0's
    # score must not be observable through slot 1 or 2.
    fs = _fs()
    deme = instance_set(3, fs)
    deme[0].second.set_score(-1.0)
    assert deme[1].second.get_score() == very_worst_score
    assert deme[2].second.get_score() == very_worst_score


def test_n_inst_fs_constructor_copies_the_prototype_n_times_independently():
    fs = _fs()
    proto = instance(fs.packed_width())
    fs.set_raw(proto, 0, 1)
    deme = instance_set(3, proto, fs)
    assert len(deme) == 3
    for entry in deme:
        assert entry.first == proto
        assert entry.first is not proto
    deme[0].first[0] = 0xFF
    assert deme[1].first == proto
    assert deme[2].first == proto


def test_n_negative_is_rejected():
    with pytest.raises(ValueError):
        instance_set(-1, _fs())


def test_fs_constructor_makes_an_empty_deme_with_default_or_given_id():
    fs = _fs()
    empty = instance_set(fs)
    assert len(empty) == 0
    assert empty.getID() == "0"

    named = instance_set(fs, demeID_t(4, 2))
    assert named.getID() == "4.2"


def test_constructor_rejects_unrecognized_argument_shapes():
    with pytest.raises(TypeError):
        instance_set("not-an-int-or-field-set")
    with pytest.raises(TypeError):
        instance_set(3, "not-a-field-set")
    with pytest.raises(TypeError):
        instance_set(_fs(), _fs())  # id must be a demeID_t, not a field_set


# --- resize ---------------------------------------------------------------


def test_resize_grows_with_independent_null_instances():
    fs = _fs()
    deme = instance_set(fs)
    deme.resize(2)
    assert len(deme) == 2
    for entry in deme:
        assert entry.first == instance(fs.packed_width())
        assert entry.second.get_score() == very_worst_score
    deme[0].second.set_score(-3.0)
    assert deme[1].second.get_score() == very_worst_score


def test_resize_shrinks_by_truncating_from_the_end():
    fs = _fs()
    deme = instance_set(3, fs)
    deme[0].second.set_score(-1.0)
    deme[1].second.set_score(-2.0)
    deme[2].second.set_score(-3.0)
    deme.resize(1)
    assert len(deme) == 1
    assert deme[0].second.get_score() == -1.0


def test_resize_rejects_negative_size():
    with pytest.raises(ValueError):
        instance_set(_fs()).resize(-1)


# --- score/instance views ---------------------------------------------------


def test_begin_scores_view_reads_and_writes_through_to_the_deme():
    fs = _fs()
    deme = instance_set(3, fs)
    view = deme.begin_scores()
    assert len(view) == 3
    assert view[0].get_score() == very_worst_score
    view[0] = composite_score(-4.0, 1, 0.0, 0.0)
    assert deme[0].second.get_score() == -4.0


def test_begin_scores_offset_and_slicing_match_std_next_semantics():
    fs = _fs()
    deme = instance_set(3, fs)
    for i, entry in enumerate(deme):
        entry.second.set_score(-float(i))
    offset_view = deme.begin_scores()
    assert [s.get_score() for s in offset_view[1:]] == [-1.0, -2.0]
    assert len(deme.end_scores()) == 0


def test_begin_instances_view_is_read_only():
    # C++'s own mutable `instance_iterator begin_instances()` overload is
    # commented out in instance_set.h (:122-123) -- only the const one is
    # live. Faithfully ported as read-only: unlike `_ScoreView`, this view
    # defines no `__setitem__` at all.
    fs = _fs()
    deme = instance_set(2, fs)
    view = deme.begin_instances()
    assert len(view) == 2
    assert view[0] == deme[0].first
    with pytest.raises(TypeError):
        view[0] = instance(1)


def test_end_instances_is_an_empty_view_past_the_last_slot():
    fs = _fs()
    deme = instance_set(2, fs)
    assert len(deme.end_instances()) == 0


# --- C++ oracle -------------------------------------------------------------


@pytest.mark.oracle
def test_cpp_default_fill_oracle(fixture_dir):
    """`instance_set(3, fs)` and mutating slot 0's score afterward must
    match classic bit-exactly -- see containers_probe.cc's
    `emit_default_fill`.
    """
    fixture = json.loads((fixture_dir / "containers.json").read_text())
    section = fixture["default_fill"]

    fs = field_set(field_set.disc_spec(2), 6)
    deme = instance_set(3, fs)
    assert len(deme) == len(section["entries"])
    for entry, expected in zip(deme, section["entries"], strict=True):
        assert [f"0x{w:016x}" for w in entry.first] == expected["words"]
        assert _hex(entry.second.get_score()) == expected["score"]["score"]
        assert entry.second.get_complexity() == expected["score"]["complexity"]

    deme[0].second.set_score(-1.0)
    for entry, expected in zip(deme, section["after_mutating_slot0"], strict=True):
        assert _hex(entry.second.get_score()) == expected["score"]


@pytest.mark.oracle
def test_cpp_fill_with_instance_oracle(fixture_dir):
    """`instance_set(3, proto, fs)` then mutating slot 0's instance in
    place must match classic bit-exactly -- see containers_probe.cc's
    `emit_fill_with_instance`.
    """
    fixture = json.loads((fixture_dir / "containers.json").read_text())
    section = fixture["fill_with_instance"]["entries"]

    fs = field_set(field_set.disc_spec(2), 6)
    proto = instance(fs.packed_width())
    fs.set_raw(proto, 0, 1)
    fs.set_raw(proto, 3, 1)
    deme = instance_set(3, proto, fs)
    deme[0].first[0] = 0xFF

    for entry, expected in zip(deme, section, strict=True):
        assert [f"0x{w:016x}" for w in entry.first] == expected
