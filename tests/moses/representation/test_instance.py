"""Tests for classic-compatible packed instance storage."""

import pytest

from pymetta.moses.representation.instance import bits_per_packed_t, instance


def test_instance_width_constructs_zero_words():
    assert bits_per_packed_t == 64
    assert instance() == []
    assert instance(2) == [0, 0]


def test_instance_iterable_construction_copies_and_masks_words():
    words = [1, (1 << 65) + 3]
    packed = instance(words)
    words[0] = 9
    assert packed == [1, 3]


def test_instance_rejects_negative_width_and_is_unhashable():
    with pytest.raises(ValueError, match="non-negative"):
        instance(-1)
    with pytest.raises(TypeError):
        hash(instance(1))


def test_all_public_mutations_preserve_unsigned_64_bit_words():
    mask = (1 << bits_per_packed_t) - 1
    packed = instance([0, 0])
    packed[0] = -1
    packed[1:] = [(1 << 64) + 2, -2]
    packed.append(1 << 65)
    packed.extend([-3, (1 << 64) + 4])
    packed.insert(0, -5)
    packed += [-6]
    assert packed == [mask - 4, mask, 2, mask - 1, 0, mask - 2, 4, mask - 5]
