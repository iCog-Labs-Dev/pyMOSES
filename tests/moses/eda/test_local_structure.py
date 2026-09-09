"""Unit tests for pymetta.moses.eda.local_structure.

No C++ probe of its own (BOA/`local_structure` is P2, not built as a
standalone `test_*` tool anywhere in classic's own tree): pinned by hand
against local_structure.{h,cc}'s own documented algorithm, plus one
end-to-end check (`test_univariate.py`) that this machinery, driven through
`univariate_optimization`, converges the same way classic's own `-aun`
oracle (`test_un_3_pa_multi_thread`) does.
"""

import pytest

from pymetta.moses.eda.local_structure import (
    local_structure_model,
    local_structure_probs_learning,
    univariate,
)
from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.field_set import contin_spec, disc_spec, field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance
from pymetta.util.rng import randGen


def _pop(fs: field_set, raw_values: list[list[int]]) -> list[scored_instance]:
    """Build instances whose raw fields are exactly `raw_values[i]`."""
    out = []
    for i, row in enumerate(raw_values):
        inst = instance(fs.packed_width())
        for idx, v in enumerate(row):
            fs.set_raw(inst, idx, v)
        out.append(scored_instance(inst, composite_score(float(i))))
    return out


# --------------------------------------------------------------- disc-only model


def test_disc_only_model_has_one_leaf_dtree_per_raw_field():
    fs = field_set([disc_spec(3), disc_spec(2)])
    pop = _pop(fs, [[0, 0], [1, 1], [2, 0]])
    model = local_structure_model(fs, pop)
    assert len(model.dtrees()) == fs.raw_size() == 2
    for dtr in model.dtrees():
        assert dtr.is_childless(dtr.begin())


def test_univariate_structure_learner_is_a_true_no_op():
    fs = field_set([disc_spec(3), disc_spec(2)])
    pop = _pop(fs, [[0, 0], [1, 1], [2, 0]])
    model = local_structure_model(fs, pop)
    before = [list(dtr.begin().data) for dtr in model.dtrees()]
    univariate()(fs, pop, model)
    after = [list(dtr.begin().data) for dtr in model.dtrees()]
    assert before == after


def test_probs_learning_counts_and_grand_total_are_correct():
    fs = field_set([disc_spec(3)])
    # values: 1,1,0,2,1 -> counts [1,3,1], total 5.
    pop = _pop(fs, [[1], [1], [0], [2], [1]])
    model = local_structure_model(fs, pop)
    local_structure_probs_learning()(fs, pop, model)
    leaf = model.dtrees()[0].begin()
    assert leaf.data == [1, 3, 1, 5]


def test_sampling_deterministic_when_only_one_bucket_has_mass():
    fs = field_set([disc_spec(3), disc_spec(4)])
    pop = _pop(fs, [[1, 2]] * 6)  # every instance identical
    model = local_structure_model(fs, pop)
    local_structure_probs_learning()(fs, pop, model)
    randGen().seed(123)
    for _ in range(10):
        sampled = model()
        assert fs.get_raw(sampled, 0) == 1
        assert fs.get_raw(sampled, 1) == 2


def test_sampling_an_all_zero_leaf_returns_zero_without_drawing_rng():
    """DO NOT FIX marker in local_structure.py: `if node.data[-1] > 0: ...
    else: return 0` -- an all-zero leaf (never learned on) must return 0
    UNCONDITIONALLY, consuming NO `randGen()` draw. Proven by comparing
    the RNG's internal state before/after.
    """
    fs = field_set([disc_spec(3)])
    model = local_structure_model(fs, [])  # empty population -> all-zero leaf

    randGen().seed(9)
    draw_without_model_call = randGen().genrand_int32()

    randGen().seed(9)
    sampled = model()
    draw_after_model_call = randGen().genrand_int32()

    assert fs.get_raw(sampled, 0) == 0
    # If model() had consumed any draw, the NEXT draw after it would differ
    # from the first draw of a fresh, equally-seeded stream.
    assert draw_after_model_call == draw_without_model_call


def test_mutation_returning_uniform_draw_instead_of_zero_is_detectable():
    """Mutation proof for the same DO NOT FIX marker: if `_sample`'s
    all-zero branch instead did `return randGen().randint_n(dtr->size()-1)`
    (the "fixed" behaviour the C++ comment itself wonders about but does
    NOT do), it WOULD consume an RNG draw here -- demonstrated directly.
    """
    randGen().seed(9)
    before = randGen().genrand_int32()
    randGen().seed(9)

    def mutant_sample_all_zero():
        return randGen().randint_n(3)

    mutant_sample_all_zero()
    after = randGen().genrand_int32()
    assert after != before  # the mutant DOES consume an extra draw


# --------------------------------------------------------------- contin splitting


def test_contin_depth_two_splits_on_non_uniform_second_position():
    """`rec_split_contin` splits dtree index 1 (the field's SECOND depth
    position) on raw index 0 (its parent) when the population disagrees
    on raw index 1's value. Structural check: the dtree's root becomes an
    internal (3-child) node keyed on src_idx == 0.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2)])
    # raw idx 0 in {0,1,2}; raw idx 1 varies too, so is_uniform_on(idx=1) is False.
    pop = _pop(fs, [[0, 1], [1, 2], [2, 0], [0, 2]])
    model = local_structure_model(fs, pop)
    dtr1 = model.dtrees()[1]
    root = dtr1.begin()
    assert not dtr1.is_childless(root)
    assert root.data == [0]  # src_idx
    assert dtr1.number_of_children(root) == 3


def test_contin_depth_two_stays_a_leaf_when_second_position_is_uniform():
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2)])
    pop = _pop(fs, [[0, 1], [1, 1], [2, 1]])  # raw idx 1 always 1 -> uniform
    model = local_structure_model(fs, pop)
    dtr1 = model.dtrees()[1]
    root = dtr1.begin()
    assert dtr1.is_childless(root)


def test_contin_probs_learning_partitions_by_parent_value():
    """After splitting, `local_structure_probs_learning` must route each
    instance into the CHILD matching its raw idx-0 (Stop/Left/Right)
    value, and each leaf's counts must sum back to that subgroup's size.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2)])
    # 2 Stop(0), 3 Left(1), 1 Right(2); raw idx 1 arbitrary but non-uniform overall.
    pop = _pop(
        fs,
        [
            [0, 0], [0, 1],
            [1, 0], [1, 1], [1, 2],
            [2, 0],
        ],
    )
    model = local_structure_model(fs, pop)
    local_structure_probs_learning()(fs, pop, model)
    dtr1 = model.dtrees()[1]
    root = dtr1.begin()
    child0 = dtr1.child(root, 0)
    child1 = dtr1.child(root, 1)
    child2 = dtr1.child(root, 2)
    assert child0.data[-1] == 2
    assert child1.data[-1] == 3
    assert child2.data[-1] == 1
    assert sum(child0.data[:-1]) == 2
    assert sum(child1.data[:-1]) == 3
    assert sum(child2.data[:-1]) == 1


def test_split_raises_when_target_node_already_has_children():
    """C++: OC_ASSERT(tgt.is_childless(), "dtree node should be childless
    (split)") -- ported as AssertionError.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2)])
    pop = _pop(fs, [[0, 1], [1, 2], [2, 0]])
    model = local_structure_model(fs, pop)
    dtr1 = model.dtrees()[1]
    root = dtr1.begin()  # already split by the constructor above
    with pytest.raises(AssertionError):
        model._split(0, 1, root)


def test_sampling_a_split_contin_dtree_reads_the_parent_margs_value():
    """`_sample` on an internal node reads `margs[src_idx]` (the ALREADY-
    sampled value of the parent field) to select which child to descend
    into -- verified by forcing every leaf under one specific child to be
    the only leaf with any mass, then checking the value assigned to that
    field only ever comes from the corresponding child's distribution.
    """
    fs = field_set([contin_spec(0.0, 1.0, 2.0, 2)])
    pop = _pop(fs, [[0, 5], [1, 6], [2, 7]])  # non-uniform idx1 -> forces a split
    model = local_structure_model(fs, pop)
    dtr1 = model.dtrees()[1]
    root = dtr1.begin()
    # Force raw idx 0 (dtree[0], a plain never-split leaf) to always sample
    # value 1, so the parent value driving the split below is deterministic.
    model.dtrees()[0].begin().data = [0, 7, 0, 7]
    # Manually seed all-mass onto child 1 (the "Left" partition) only.
    for i in range(3):
        child = dtr1.child(root, i)
        child.data = [0, 0, 0, 0]
    left_child = dtr1.child(root, 1)
    left_child.data = [0, 7, 0, 7]  # bucket 1 has all the mass

    randGen().seed(1)
    for _ in range(5):
        sampled = model()
        raw0 = fs.get_raw(sampled, 0)
        raw1 = fs.get_raw(sampled, 1)
        assert raw0 == 1
        assert raw1 == 1  # sampled via the correctly-selected child 1
