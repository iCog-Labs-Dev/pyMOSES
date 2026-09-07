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
#
# DIVERGENCE(C++ moses/moses/representation/instance.h:48-63): the
# `std::hash<instance>` specialization ("This is to enable std::unordered_map
# and similar", the header's own comment) is examined, not silently skipped,
# by S9 Task 0 -- and NOT ported. `instance_set` (instance_set.py), the
# container S9 Task 0 adds specifically to close this gap
# (docs/MODULE-COVERAGE.md's own `instance.h` row named it as such), turns
# out to need no hash at all: `instance_set` is `public std::vector<
# scored_instance<ScoreT>>` (instance_set.h:52), not an unordered_map/
# unordered_set keyed on `instance` or anything containing one. Searching
# the ENTIRE classic tree (`grep -rn 'unordered_map\|unordered_set'
# moses/ tests/`, both narrowly for "instance" nearby and broadly for any
# hit at all near a `scored_instance`/`instance_set`/`iscorer_base` name)
# finds exactly one match: the hash specialization's own comment line,
# instance.h:48. No unordered_map/unordered_set anywhere in classic MOSES
# -- moses/, feature-selection/, or tests/ -- is ever keyed on `instance`
# (or on `scored_instance`, which has no hash specialization of its own
# either). The specialization is dead code in the upstream sources this
# fork tracks: written to make `instance` hashable, never actually used
# for that purpose anywhere the code that would need it ships. `instance`
# therefore stays deliberately unhashable here -- not because hashing it
# would be hard (the classic mixing formula is trivial to reproduce; see
# the commented-out sketch this note replaces), but because no consumer
# in scope for S9 (this task's three containers) or reachable from them
# (Task 1's neighborhood_sampling, Task 2's hill-climbing -- neither
# greps up an unordered container of instances either) needs it. Revisit
# only if a future task finds a real `unordered_map</unordered_set<`
# keyed on `instance` that this search missed.
class instance(list[packed_t]):
    """C++: moses/moses/representation/instance.h:43 instance

    ``instance(n)`` creates ``n`` zero words; iterable construction makes a
    masked value copy. Instances remain deliberately unhashable while
    mutable -- see the DIVERGENCE note above this class for why S9 examined
    and rejected porting the classic hash specialization, rather than
    leaving it deferred.
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
