import pytest

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.knob_mapper import knob_mapper
from pymoses.moses.representation.knobs import contin_knob, logical_subtree_knob


def _knob(tr, idx):
    subtree = combo_tree(argument(idx))
    return logical_subtree_knob(tr, tr.begin(), subtree.begin())


def _contin_knob(tr, node, *, step_size=1.0, expansion=2.0, depth=5):
    return contin_knob(tr, node, step_size=step_size, expansion=expansion, depth=depth)


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


# ***************************************************************************
# contin collection (S14 Task 2)


def test_contin_mapper_stably_orders_duplicate_specs_and_indexes_by_identity():
    tr = str_to_combo_tree("plus(1.0 1.0 0.0)")
    leaves = list(tr.children(tr.begin()))
    tied_a = _contin_knob(tr, leaves[0], depth=5)
    tied_b = _contin_knob(tr, leaves[1], depth=5)
    lower = _contin_knob(tr, leaves[2], depth=2)
    mapper = knob_mapper()
    # Insert the lower-depth knob FIRST, and the tied pair out of the order
    # they'll be re-sorted into -- proving finalize() actually reorders
    # rather than the insertion order already happening to match.
    mapper.add_contin_knob(lower)
    mapper.add_contin_knob(tied_b)
    mapper.add_contin_knob(tied_a)

    with pytest.raises(RuntimeError, match="not finalized"):
        mapper.find_contin_knob(tied_a.get_loc())
    mapper.finalize()

    # Descending by depth first (field_set.h's own contin_spec::operator<
    # sorts descending by depth, then expansion, then step_size, then mean)
    # -- `lower` (depth=2) sorts LAST despite being inserted first.
    assert [entry.spec.depth for entry in mapper.contin] == [5, 5, 2]
    assert all(entry.spec == entry.knob.spec() for entry in mapper.contin)
    # Ties (tied_a, tied_b: identical spec) keep INSERTION order --
    # `tied_b` was added before `tied_a` above.
    assert [entry.knob for entry in mapper.contin] == [tied_b, tied_a, lower]
    assert mapper.find_contin_knob(tied_a.get_loc()) is tied_a
    assert mapper.find_contin_knob(tied_b.get_loc()) is tied_b
    assert mapper.find_contin_knob(lower.get_loc()) is lower
    assert mapper.find_contin_knob(None) is None
    mapper.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        mapper.add_contin_knob(lower)


def test_contin_mapper_rejects_duplicate_node_identity():
    tr = str_to_combo_tree("plus(1.0 0.0)")
    node = tr.begin().first_child
    first = _contin_knob(tr, node)
    duplicate = _contin_knob(tr, node)
    other = _contin_knob(tr, tr.begin().last_child)
    assert first.get_loc() is duplicate.get_loc()
    mapper = knob_mapper()
    mapper.add_contin_knob(other)
    mapper.add_contin_knob(first)
    mapper.add_contin_knob(duplicate)
    before_entries = mapper.contin
    before_identity_map = dict(mapper._it_contin_knob)

    with pytest.raises(ValueError, match="already has a knob"):
        mapper.finalize()

    assert mapper.contin == before_entries
    assert mapper._it_contin_knob == before_identity_map == {}
    with pytest.raises(RuntimeError, match="not finalized"):
        mapper.find_contin_knob(first.get_loc())


def test_contin_mapper_rejects_a_finalized_mapping_after_location_is_erased():
    tr = str_to_combo_tree("plus(1.0 0.0)")
    node = tr.begin().first_child
    knob = _contin_knob(tr, node)
    other_node = tr.begin().last_child
    mapper = knob_mapper()
    mapper.add_contin_knob(knob)
    mapper.add_contin_knob(_contin_knob(tr, other_node))
    mapper.finalize()

    tr.erase(node)

    assert not tr.is_valid(node)
    with pytest.raises(ValueError, match="erased or unset"):
        mapper.find_contin_knob(node)


def test_finalized_contin_specs_align_with_field_sets_own_layout():
    tr = str_to_combo_tree("plus(1.0 0.0)")
    deep = _contin_knob(tr, tr.begin().first_child, depth=5)
    shallow = _contin_knob(tr, tr.begin().last_child, depth=2)
    mapper = knob_mapper()
    mapper.add_contin_knob(shallow)
    mapper.add_contin_knob(deep)
    mapper.finalize()

    specs = tuple(entry.spec for entry in mapper.contin)
    fields = field_set(specs)

    assert [entry.spec.depth for entry in mapper.contin] == [5, 2]
    assert fields.contin() == specs
    assert fields.raw_size() == 5 + 2
    assert fields.n_contin_fields() == 7
    assert fields.begin_contin_raw_idx() == 0
    assert fields.n_disc_fields() == 0
    assert fields.n_bits() == 0
    # C++: boost::variant<term_spec, contin_spec, disc_spec>'s own
    # which()-first multiset order places every contin_spec before every
    # disc_spec regardless of value -- disc knobs (none, here) would land
    # AFTER all 7 raw contin fields.
    assert fields.begin_disc_raw_idx() == 7
