"""C++: moses/moses/representation/instance.h:34 packed_t

Storage aliases and the packed instance container used by representation.
Classic's ``instance`` is a mutable ``std::vector<unsigned long>``. This
module pins ``unsigned long`` to the 64-bit ABI measured by the S8 C++ probe;
all writes performed by :mod:`field_set` are masked to that word width.

Term-algebra storage is intentionally omitted. It belongs to excluded object
and procedure domains, while Boolean, continuous, and action knobs all use the
packed integer fields represented here.
"""

from __future__ import annotations

from collections.abc import Iterable

packed_t = int  # C++: moses/moses/representation/instance.h:34 packed_t
bits_per_packed_t = 64  # C++: moses/moses/representation/instance.h:35 bits_per_packed_t
contin_t = float  # C++: moses/moses/representation/instance.h:38 contin_t
disc_t = int  # C++: moses/moses/representation/instance.h:39 disc_t

_PACKED_MASK = (1 << bits_per_packed_t) - 1


# DIVERGENCE(C++ moses/moses/representation/instance.h:34-43): classic uses the
# platform width of unsigned long. This port deliberately pins the measured
# 64-bit ABI and coerces every public mutation to that unsigned word range.
#
# DIVERGENCE(C++ moses/moses/representation/instance.h:43): classic aliases
# `instance` directly to `std::vector<packed_t>`. A small list subclass keeps
# the same mutable value semantics while making the ubiquitous C++
# `instance(packed_width)` zero-filled construction available in Python.
class instance(list[packed_t]):
    """C++: moses/moses/representation/instance.h:43 instance

    ``instance(n)`` creates ``n`` zero words; iterable construction makes a
    masked value copy. Instances remain deliberately unhashable while mutable.
    The classic hash specialization is deferred with ``instance_set``.
    """

    __hash__ = None

    def __init__(self, width_or_words: int | Iterable[int] = 0) -> None:
        if isinstance(width_or_words, bool):
            width_or_words = int(width_or_words)
        if isinstance(width_or_words, int):
            if width_or_words < 0:
                raise ValueError("instance width must be non-negative")
            super().__init__([0] * width_or_words)
            return
        super().__init__(_coerce_word(word) for word in width_or_words)

    def __setitem__(self, key: int | slice, value: int | Iterable[int]) -> None:
        """C++: moses/moses/representation/instance.h:43 instance"""
        if isinstance(key, slice):
            if isinstance(value, int):
                raise TypeError("can only assign an iterable to an instance slice")
            super().__setitem__(key, [_coerce_word(word) for word in value])
        else:
            if not isinstance(value, int):
                raise TypeError("instance words must be integers")
            super().__setitem__(key, _coerce_word(value))

    def append(self, value: int) -> None:
        """C++: moses/moses/representation/instance.h:43 instance"""
        super().append(_coerce_word(value))

    def extend(self, values: Iterable[int]) -> None:
        """C++: moses/moses/representation/instance.h:43 instance"""
        super().extend(_coerce_word(value) for value in values)

    def insert(self, index: int, value: int) -> None:
        """C++: moses/moses/representation/instance.h:43 instance"""
        super().insert(index, _coerce_word(value))

    def __iadd__(self, values: Iterable[int]):
        """C++: moses/moses/representation/instance.h:43 instance"""
        self.extend(values)
        return self


def _coerce_word(value: int) -> packed_t:
    return int(value) & _PACKED_MASK
