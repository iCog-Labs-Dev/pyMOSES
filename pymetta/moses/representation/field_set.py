"""Packed discrete and continuous fields for supported MOSES domains.

C++ stores term-algebra, continuous, discrete, and one-bit fields in one
``field_set``. The term-algebra branch is excluded from this port; this module
retains the exact continuous-before-discrete ordering and raw layouts used by
Boolean logical knobs, continuous knobs, and future action knobs.

"""

# DIVERGENCE(C++ moses/moses/representation/field_set.h:72-90): term_spec and
# term-algebra accessors are omitted outside Boolean/continuous/action scope.

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cmp_to_key
from typing import ClassVar

from pymetta.moses.representation.instance import (
    _PACKED_MASK,
    bits_per_packed_t,
    contin_t,
    disc_t,
    instance,
)

multiplicity_t = int
width_t = int
depth_t = int
size_t = int


def nbits_to_pack(multy: int) -> int:
    """C++: opencog/util/numeric.h:127 nbits_to_pack

    Return the next power-of-two bit width capable of storing ``multy``
    distinct values. In particular, multiplicities 1 and 2 both use one bit.
    """
    # DIVERGENCE(C++ opencog/util/numeric.h:127): classic relies on an
    # assertion/precondition; Python makes invalid multiplicity deterministic.
    if multy <= 0:
        raise ValueError("multiplicity must be positive")
    required = max(1, (multy - 1).bit_length())
    return 1 << (required - 1).bit_length()


@dataclass(frozen=True, slots=True)
class field:
    """C++: moses/moses/representation/field_set.h:137 field

    Descriptor for one raw field within an instance word array.
    """

    width: width_t
    major_offset: size_t
    minor_offset: size_t


@dataclass(frozen=True, slots=True)
class disc_spec:
    """C++: moses/moses/representation/field_set.h:151 disc_spec"""

    multy: multiplicity_t

    def __lt__(self, rhs: object) -> bool:
        """C++: moses/moses/representation/field_set.h:151 disc_spec"""
        if not isinstance(rhs, disc_spec):
            return NotImplemented
        return self.multy > rhs.multy


@dataclass(frozen=True, slots=True)
class contin_spec:
    """C++: moses/moses/representation/field_set.h:180 contin_spec"""

    mean: contin_t
    step_size: contin_t
    expansion: contin_t
    depth: depth_t

    Stop: ClassVar[disc_t] = 0
    Left: ClassVar[disc_t] = 1
    Right: ClassVar[disc_t] = 2

    def __lt__(self, rhs: object) -> bool:
        """C++: moses/moses/representation/field_set.h:180 contin_spec"""
        if not isinstance(rhs, contin_spec):
            return NotImplemented
        return self.depth > rhs.depth or (
            self.depth == rhs.depth
            and (
                self.expansion > rhs.expansion
                or (
                    self.expansion == rhs.expansion
                    and (
                        self.step_size > rhs.step_size
                        or (self.step_size == rhs.step_size and self.mean > rhs.mean)
                    )
                )
            )
        )

    def epsilon(self) -> contin_t:
        """C++: moses/moses/representation/field_set.h:208 epsilon"""
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:208): reject
        # signed Python depths rather than allowing an invalid shift.
        if self.depth < 0:
            raise ValueError("continuous depth must be non-negative")
        return self.step_size / contin_t(1 << self.depth)

    @staticmethod
    def switchLR(lr: disc_t) -> disc_t:
        """C++: moses/moses/representation/field_set.h:220 switchLR"""
        if lr == contin_spec.Left:
            return contin_spec.Right
        if lr == contin_spec.Right:
            return contin_spec.Left
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:220): OC_ASSERT becomes ValueError.
        raise ValueError("continuous direction must be Left or Right")


class contin_stepper:
    """C++: moses/moses/representation/field_set.h:233 contin_stepper"""

    __slots__ = ("c", "value", "_all_left", "_all_right", "_step_size")

    def __init__(self, c: contin_spec) -> None:
        self.c = c
        self.value = c.mean
        self._all_left = True
        self._all_right = True
        self._step_size = c.step_size

    def left(self) -> None:
        """C++: moses/moses/representation/field_set.h:241 left"""
        if self._all_left:
            self.value -= self._step_size
            self._step_size *= self.c.expansion
            self._all_right = False
        else:
            if self._all_right:
                self._all_right = False
                self._step_size /= self.c.expansion * 2
            self.value -= self._step_size
            self._step_size /= 2

    def right(self) -> None:
        """C++: moses/moses/representation/field_set.h:256 right"""
        if self._all_right:
            self.value += self._step_size
            self._step_size *= self.c.expansion
            self._all_left = False
        else:
            if self._all_left:
                self._all_left = False
                self._step_size /= self.c.expansion * 2
            self.value += self._step_size
            self._step_size /= 2


spec = contin_spec | disc_spec


def _compare_specs(lhs: spec, rhs: spec) -> int:
    if isinstance(lhs, contin_spec) and isinstance(rhs, disc_spec):
        return -1
    if isinstance(lhs, disc_spec) and isinstance(rhs, contin_spec):
        return 1
    if lhs < rhs:
        return -1
    if rhs < lhs:
        return 1
    return 0


class field_set:
    """C++: moses/moses/representation/field_set.h:99 field_set

    Describe the canonical raw layout of continuous and discrete knobs without
    storing knob values themselves.
    """

    field = field
    disc_spec = disc_spec
    contin_spec = contin_spec
    contin_stepper = contin_stepper

    def __init__(
        self,
        specs: Iterable[spec] | spec | field_set | None = None,
        n: size_t | None = None,
    ) -> None:
        """C++: moses/moses/representation/field_set.h:315 field_set()
        (``specs=None``), field_set.h:321 field_set(const field_set&)
        (``specs`` is a ``field_set``), field_set.h:329 field_set(const
        spec&, size_t) (``specs`` is a single ``disc_spec``/``contin_spec``),
        and field_set.h:337 field_set(It, It) (``specs`` an iterable) -- one
        Python constructor dispatches on argument type/shape across all four
        C++ overloads instead of overloading.
        """
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:295-344): reject
        # negative counts/depths before unsigned conversion.
        if specs is None:
            if n is not None:
                raise TypeError("a repetition count requires a field specification")
            source: list[spec] = []
        elif isinstance(specs, field_set):
            if n is not None:
                raise TypeError("a repetition count cannot be used when copying a field set")
            source = [*specs._contin, *specs._disc]
        elif isinstance(specs, (disc_spec, contin_spec)):
            repeat = 1 if n is None else n
            if repeat < 0:
                raise ValueError("specification count must be non-negative")
            source = [specs] * repeat
        else:
            if n is not None:
                raise TypeError("a repetition count is only valid for one specification")
            source = list(specs)

        for item in source:
            if not isinstance(item, (disc_spec, contin_spec)):
                raise TypeError(f"unsupported field specification: {type(item).__name__}")

        ordered = sorted(source, key=cmp_to_key(_compare_specs))
        self._fields: list[field] = []
        self._contin: list[contin_spec] = []
        self._disc: list[disc_spec] = []
        self._nbool = 0

        for item in ordered:
            if isinstance(item, contin_spec):
                self._build_contin_spec(item)
            else:
                self._build_disc_spec(item)
        self._compute_starts()

    def __eq__(self, rhs: object) -> bool:
        """C++: moses/moses/representation/field_set.cc:51 field_set

        The real member is `field_set::operator==` -- an operator overload,
        which `tools/check_alignment.py` cannot match by identifier (see
        CARRY-FORWARD.md 4.5), so the enclosing class is cited instead.
        """
        if not isinstance(rhs, field_set):
            return NotImplemented
        return self._disc == rhs._disc and self._contin == rhs._contin and self._nbool == rhs._nbool

    def _back_offset(self) -> int:
        if not self._fields:
            return 0
        last = self._fields[-1]
        return last.major_offset * bits_per_packed_t + last.minor_offset + last.width

    def _build_disc_spec(self, ds: disc_spec) -> None:
        width = nbits_to_pack(ds.multy)
        # DIVERGENCE(C++ moses/moses/representation/field_set.cc:159-169):
        # build_disc_spec there has no such check and would silently place a
        # field wider than one packed word (whose bit offsets the rest of
        # this class assumes never crosses a word boundary). Python rejects
        # it before it can corrupt later get_raw/set_raw bit arithmetic.
        if width > bits_per_packed_t:
            raise ValueError("discrete field width exceeds packed word size")
        base = self._back_offset()
        self._fields.append(field(width, base // bits_per_packed_t, base % bits_per_packed_t))
        self._disc.append(ds)
        if width == 1:
            self._nbool += 1

    def _build_contin_spec(self, cs: contin_spec) -> None:
        if cs.depth < 0:
            raise ValueError("continuous depth must be non-negative")
        base = self._back_offset()
        for _ in range(cs.depth):
            self._fields.append(field(2, base // bits_per_packed_t, base % bits_per_packed_t))
            base += 2
        self._contin.append(cs)

    def _compute_starts(self) -> None:
        self._begin_contin_raw_idx = 0
        self._end_contin_raw_idx = sum(item.depth for item in self._contin)
        self._begin_disc_raw_idx = self._end_contin_raw_idx
        self._end_disc_raw_idx = len(self._fields) - self._nbool
        self._begin_bit_raw_idx = self._end_disc_raw_idx
        self._end_bit_raw_idx = len(self._fields)
        self._contin_raw_offsets: list[int] = []
        raw_idx = self._begin_contin_raw_idx
        for item in self._contin:
            self._contin_raw_offsets.append(raw_idx)
            raw_idx += item.depth

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:521): immutable
    # descriptor tuple replaces classic's begin/end `field_iterator` pair.
    # `field_iterator` (field_set.h:145) is itself already a
    # `std::vector<field>::const_iterator`, not a mutable one -- this port
    # does not relax any existing const-ness, it just returns a value
    # instead of an iterator pair.
    def fields(self) -> tuple[field, ...]:
        """C++: moses/moses/representation/field_set.h:521 begin_fields

        """
        return tuple(self._fields)

    def packed_width(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:357 packed_width"""
        return 0 if not self._fields else self._fields[-1].major_offset + 1

    def empty(self) -> bool:
        """C++: moses/moses/representation/field_set.h:361 empty"""
        return not self._fields

    def raw_size(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:365 raw_size"""
        return len(self._fields)

    def dim_size(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:383 dim_size"""
        return self.n_bits() + self.n_disc_fields() + len(self._contin)

    def count(self, inst: instance) -> size_t:
        """C++: moses/moses/representation/field_set.h:388 count"""
        return sum(value != 0 for value in self.iter_raw(inst))

    def disc_and_bit(self) -> tuple[disc_spec, ...]:
        """C++: moses/moses/representation/field_set.h:396 disc_and_bit"""
        return tuple(self._disc)

    def contin(self) -> tuple[contin_spec, ...]:
        """C++: moses/moses/representation/field_set.h:400 contin"""
        return tuple(self._contin)

    def get_raw(self, inst: instance, idx: size_t) -> disc_t:
        """C++: moses/moses/representation/field_set.h:410 get_raw"""
        self._validate_raw_idx(idx)
        descriptor = self._fields[idx]
        mask = (1 << descriptor.width) - 1
        return (inst[descriptor.major_offset] >> descriptor.minor_offset) & mask

    def set_raw(self, inst: instance, idx: size_t, value: disc_t) -> None:
        """C++: moses/moses/representation/field_set.h:416 set_raw"""
        # Like classic unsigned-long arithmetic, a field that straddles a word
        # is truncated to the low portion held by its descriptor's first word.
        self._validate_raw_idx(idx)
        descriptor = self._fields[idx]
        shifted_mask = (((1 << descriptor.width) - 1) << descriptor.minor_offset) & _PACKED_MASK
        major = descriptor.major_offset
        word = inst[major] & _PACKED_MASK
        shifted_value = (int(value) << descriptor.minor_offset) & _PACKED_MASK
        inst[major] = (word ^ ((word ^ shifted_value) & shifted_mask)) & _PACKED_MASK

    def get_contin(self, inst: instance, idx: size_t) -> contin_t:
        """C++: moses/moses/representation/field_set.cc:75 get_contin"""
        self._validate_contin_idx(idx)
        raw_idx = self.contin_to_raw_idx(idx)
        descriptor = self._contin[idx]
        stepper = contin_stepper(descriptor)
        for offset in range(descriptor.depth):
            direction = self.get_raw(inst, raw_idx + offset)
            if direction == contin_spec.Left:
                stepper.left()
            elif direction == contin_spec.Right:
                stepper.right()
            elif direction == contin_spec.Stop:
                break
            else:
                # DIVERGENCE(C++ moses/moses/representation/field_set.cc:92): classic asserts on an
                # invalid two-bit direction; Python raises ValueError.
                raise ValueError(f"unknown continuous direction: {direction}")
        return stepper.value

    def set_contin(self, inst: instance, idx: size_t, target: contin_t) -> None:
        """C++: moses/moses/representation/field_set.cc:98 set_contin"""
        self._validate_contin_idx(idx)
        raw_idx = self.contin_to_raw_idx(idx)
        descriptor = self._contin[idx]
        best_distance = math.fabs(descriptor.mean - target)
        best_depth = 0
        stepper = contin_stepper(descriptor)
        offset = 0
        while offset < descriptor.depth and best_distance > descriptor.epsilon():
            if target < stepper.value:
                stepper.left()
                self.set_raw(inst, raw_idx + offset, contin_spec.Left)
            else:
                stepper.right()
                self.set_raw(inst, raw_idx + offset, contin_spec.Right)
            distance = math.fabs(stepper.value - target)
            if distance < best_distance:
                best_distance = distance
                best_depth = offset + 1
            offset += 1
        for offset in range(best_depth, descriptor.depth):
            self.set_raw(inst, raw_idx + offset, contin_spec.Stop)

    def pack(self, values: Iterable[disc_t], out: instance | None = None) -> instance:
        """C++: moses/moses/representation/field_set.h:1326 pack

        OR raw values into packed storage. A newly allocated zero instance is
        used when ``out`` is omitted.
        """
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:1326): Python
        # validates the input count and exact output width before mutation.
        raw = list(values)
        if len(raw) != self.raw_size():
            raise ValueError(f"expected {self.raw_size()} raw values, got {len(raw)}")
        result = instance(self.packed_width()) if out is None else out
        if len(result) != self.packed_width():
            raise ValueError("output instance width does not match field set")
        major = 0
        offset = 0
        for descriptor, value in zip(self._fields, raw, strict=True):
            result[major] = (
                result[major]
                | ((int(value) << (offset % bits_per_packed_t)) & _PACKED_MASK)
            ) & _PACKED_MASK
            offset += descriptor.width
            if offset == bits_per_packed_t:
                offset = 0
                major += 1
        return result

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:442): classic's
    # assertion compares `inst1.size()` with itself, then may read past inst2.
    # Python validates both operands against the field set before walking raw
    # fields, turning undefined access into a deterministic ValueError.
    def hamming_distance(self, inst1: instance, inst2: instance) -> int:
        """C++: moses/moses/representation/field_set.h:440 hamming_distance"""
        self._validate_instance_width(inst1)
        self._validate_instance_width(inst2)
        return sum(
            lhs != rhs
            for lhs, rhs in zip(self.iter_raw(inst1), self.iter_raw(inst2), strict=True)
        )

    def merge_instance(self, target: instance, base: instance, reference: instance) -> None:
        """C++: moses/moses/representation/field_set.h:468 merge_instance"""
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:468): classic
        # only asserts equal sizes; Python also requires exact packed width.
        self._validate_instance_width(target)
        self._validate_instance_width(base)
        self._validate_instance_width(reference)
        for idx, (base_value, ref_value) in enumerate(
            zip(self.iter_raw(base), self.iter_raw(reference), strict=True)
        ):
            if base_value != ref_value:
                self.set_raw(target, idx, ref_value)

    def _validate_instance_width(self, inst: instance) -> None:
        if len(inst) != self.packed_width():
            raise ValueError(
                f"instance has {len(inst)} words; field set requires {self.packed_width()}"
            )

    def _validate_raw_idx(self, idx: size_t) -> None:
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:410-430):
        # `get_raw`/`set_raw` there have no assertion of any kind, not even
        # an OC_ASSERT -- they index `_fields[idx]` directly, so an
        # out-of-range `idx` is undefined behaviour in classic (a raw
        # `std::vector::operator[]`, not `.at()`). Python rejects every
        # negative/out-of-range public field index before access or mutation.
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < self.raw_size():
            raise IndexError(f"raw field index out of range: {idx!r}")

    def _validate_contin_idx(self, idx: size_t) -> None:
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(self._contin):
            raise IndexError(f"continuous field index out of range: {idx!r}")

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:1273): read-only
    # generator facade replaces the proxy iterator pair.
    def iter_raw(self, inst: instance) -> Iterator[disc_t]:
        """C++: moses/moses/representation/field_set.h:1273 begin_raw

        """
        for idx in range(self.raw_size()):
            yield self.get_raw(inst, idx)

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:1217): read-only
    # generator facade replaces the proxy iterator pair.
    def iter_disc(self, inst: instance) -> Iterator[disc_t]:
        """C++: moses/moses/representation/field_set.h:1217 begin_disc

        """
        for idx in range(self.begin_disc_raw_idx(), self.end_disc_raw_idx()):
            yield self.get_raw(inst, idx)

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:1189): read-only
    # generator facade replaces the proxy iterator pair.
    def iter_bit(self, inst: instance) -> Iterator[bool]:
        """C++: moses/moses/representation/field_set.h:1189 begin_bit

        """
        for idx in range(self.begin_bit_raw_idx(), self.end_bit_raw_idx()):
            yield bool(self.get_raw(inst, idx))

    # DIVERGENCE(C++ moses/moses/representation/field_set.h:1237): read-only
    # generator facade replaces the proxy iterator pair.
    def iter_contin(self, inst: instance) -> Iterator[contin_t]:
        """C++: moses/moses/representation/field_set.h:1237 begin_contin

        """
        for idx in range(len(self._contin)):
            yield self.get_contin(inst, idx)

    def begin_contin_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:537 begin_contin_raw_idx"""
        return self._begin_contin_raw_idx

    def end_contin_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:540 end_contin_raw_idx"""
        return self._end_contin_raw_idx

    def begin_disc_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:544 begin_disc_raw_idx"""
        return self._begin_disc_raw_idx

    def end_disc_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:547 end_disc_raw_idx"""
        return self._end_disc_raw_idx

    def begin_bit_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:551 begin_bit_raw_idx"""
        return self._begin_bit_raw_idx

    def end_bit_raw_idx(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:554 end_bit_raw_idx"""
        return self._end_bit_raw_idx

    def n_bits(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:560 n_bits"""
        return self._nbool

    def n_disc_fields(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:567 n_disc_fields"""
        return self._end_disc_raw_idx - self._begin_disc_raw_idx

    def n_contin_fields(self) -> size_t:
        """C++: moses/moses/representation/field_set.h:574 n_contin_fields"""
        return self._end_contin_raw_idx - self._begin_contin_raw_idx

    def contin_to_raw_idx(self, spec_idx: size_t) -> size_t:
        """C++: moses/moses/representation/field_set.h:591 contin_to_raw_idx"""
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:591-594):
        # classic indexes `_contin_raw_offsets[spec_idx]` directly with no
        # bounds check at all. Python validates first.
        self._validate_contin_idx(spec_idx)
        return self._contin_raw_offsets[spec_idx]

    def raw_to_contin_idx(self, raw_idx: size_t) -> size_t:
        """C++: moses/moses/representation/field_set.h:601 raw_to_contin_idx"""
        self._validate_raw_idx(raw_idx)
        if not self.begin_contin_raw_idx() <= raw_idx < self.end_contin_raw_idx():
            raise IndexError("raw index is not a continuous field")
        offset = raw_idx - self.begin_contin_raw_idx()
        for idx, descriptor in enumerate(self._contin):
            offset -= descriptor.depth
            if offset < 0:
                return idx
        raise AssertionError("continuous raw-index lookup failed")

    def raw_to_disc_idx(self, raw_idx: size_t) -> size_t:
        """C++: moses/moses/representation/field_set.h:633 raw_to_disc_idx"""
        self._validate_raw_idx(raw_idx)
        if not self.begin_disc_raw_idx() <= raw_idx < self.end_disc_raw_idx():
            raise IndexError("raw index is not a multi-bit discrete field")
        return raw_idx - self.begin_disc_raw_idx()

    def contin_length(self, inst: instance, idx: size_t) -> size_t:
        """C++: moses/moses/representation/field_set.h:665 contin_length"""
        # DIVERGENCE(C++ moses/moses/representation/field_set.h:665-675):
        # classic indexes `_contin[idx]` directly with no bounds check at
        # all. Python validates first.
        self._validate_contin_idx(idx)
        raw_begin = self.contin_to_raw_idx(idx)
        depth = self._contin[idx].depth
        for offset in range(depth):
            if self.get_raw(inst, raw_begin + offset) == contin_spec.Stop:
                return offset
        return depth
