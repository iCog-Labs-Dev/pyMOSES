import pytest

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.knob_mapper import knob_mapper
from pymetta.moses.representation.knobs import logical_subtree_knob


def _knob(tr, idx):
    subtree = combo_tree(argument(idx))
    return logical_subtree_knob(tr, tr.begin(), subtree.begin())


def test_mapper_stably_orders_duplicate_specs_and_indexes_by_identity():
    tr = str_to_combo_tree("and($1 $2 $9)")
    first = _knob(tr, 1)
    second = _knob(tr, 2)
    lower_multiplicity = _knob(tr, 3)
    lower_multiplicity.disallow(2)
    mapper = knob_mapper()
    mapper.add_disc_knob(lower_multiplicity)
    mapper.add_disc_knob(second)
    mapper.add_disc_knob(first)

    for frozen in (lower_multiplicity, second, first):
        with pytest.raises(RuntimeError, match="specification is frozen"):
            frozen.disallow(0)
        with pytest.raises(RuntimeError, match="specification is frozen"):
            frozen.allow(0)

    with pytest.raises(RuntimeError, match="not finalized"):
        mapper.find_disc_knob(first.get_loc())
    mapper.finalize()

    assert [entry.spec.multy for entry in mapper.disc] == [3, 3, 2]
    assert all(entry.spec == entry.knob.spec() for entry in mapper.disc)
    assert [entry.knob for entry in mapper.disc] == [second, first, lower_multiplicity]
    assert mapper.find_disc_knob(first.get_loc()) is first
    assert mapper.find_disc_knob(second.get_loc()) is second
    assert mapper.find_disc_knob(lower_multiplicity.get_loc()) is lower_multiplicity
    assert mapper.find_disc_knob(None) is None
    assert not hasattr(mapper, "it_disc_idx")
    assert not hasattr(mapper, "disc_idx")
    mapper.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        mapper.add_disc_knob(first)


def test_mapper_rejects_duplicate_node_identity():
    tr = str_to_combo_tree("and($1 $9)")
    first = _knob(tr, 1)
    duplicate = _knob(tr, 1)
    lower_multiplicity = _knob(tr, 2)
    lower_multiplicity.disallow(2)
    assert first.get_loc() is duplicate.get_loc()
    mapper = knob_mapper()
    mapper.add_disc_knob(lower_multiplicity)
    mapper.add_disc_knob(first)
    mapper.add_disc_knob(duplicate)
    before_entries = mapper.disc
    before_identity_map = dict(mapper._it_disc_knob)

    with pytest.raises(ValueError, match="already has a knob"):
        mapper.finalize()

    assert mapper.disc == before_entries
    assert mapper._it_disc_knob == before_identity_map == {}
    with pytest.raises(RuntimeError, match="not finalized"):
        mapper.find_disc_knob(first.get_loc())


def test_mapper_rejects_a_finalized_mapping_after_location_is_erased():
    tr = str_to_combo_tree("and($1 $9)")
    knob = _knob(tr, 2)
    location = knob.get_loc()
    mapper = knob_mapper()
    mapper.add_disc_knob(knob)
    mapper.finalize()

    knob.clear_exemplar()

    assert not tr.is_valid(location)
    with pytest.raises(ValueError, match="erased or unset"):
        mapper.find_disc_knob(location)


def test_finalized_specs_align_with_field_set_raw_offsets_and_bit_boundary():
    tr = str_to_combo_tree("and($1 $9)")
    multi_bit = _knob(tr, 1)
    one_bit = _knob(tr, 2)
    one_bit.disallow(2)
    mapper = knob_mapper()
    mapper.add_disc_knob(one_bit)
    mapper.add_disc_knob(multi_bit)
    mapper.finalize()

    specs = tuple(entry.spec for entry in mapper.disc)
    fields = field_set(specs)

    assert [entry.spec.multy for entry in mapper.disc] == [3, 2]
    assert all(entry.spec == entry.knob.spec() for entry in mapper.disc)
    assert fields.disc_and_bit() == specs
    assert fields.raw_size() == len(mapper.disc) == 2
    assert fields.n_disc_fields() == 1
    assert fields.n_bits() == 1
    assert fields.begin_disc_raw_idx() == 0
    assert fields.begin_bit_raw_idx() == 1
    assert [
        fields.fields()[fields.begin_disc_raw_idx() + offset].width
        for offset, _entry in enumerate(mapper.disc)
    ] == [2, 1]
