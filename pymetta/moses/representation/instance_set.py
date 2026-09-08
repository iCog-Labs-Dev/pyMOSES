"""C++: moses/moses/representation/instance_set.h:52 instance_set

A `deme`: `public std::vector<scored_instance<ScoreT>>` plus a `field_set`
back-reference, a deme id, and two evaluation counters. This port subclasses
`list`, the same choice `instance.py` makes for `std::vector<packed_t>`.

Every real constructor call in classic reaches this class through one of the
three overloads below -- none of them ever calls `scored_instance(...)`
directly (see scored_instance.py's module docstring); `std::vector`'s own
fill-construction converts a bare `instance` into a `scored_instance<ScoreT>`
implicitly, via the single-argument constructor, once per slot. That
conversion default-constructs a *fresh* `ScoreT()` (a worst `composite_score`)
for every slot -- not one shared prototype copied by reference -- exactly
the aliasing distinction CARRY-FORWARD.md 1.1/2.4 flags: `instance_set(3,
fs)` must produce three *independent* `composite_score` objects, so scoring
slot 0 can never be observed through slot 1 or 2.

`demeID_t` (`moses/moses/moses/types.h:190`) is ported here, inline, rather
than in `pymetta/moses/moses/types.py` where it otherwise belongs: this S9
task owns only `pymetta/moses/representation/` (a concurrent S9 task owns
`pymetta/moses/moses/`), and `instance_set` cannot be constructed at all
without *some* `demeID_t` to default to. Only the subset `instance_set`
actually calls is ported -- default-construction (`demeID_t()`, used as
`instance_set`'s own third-constructor default argument) and the two
numeric-formatting overloads (types.cc:43-50) it composes from. This is the
"port the minimum required, and say so" case the task brief names for
`moses/moses/types.{cc,h}` PARTIAL-status gaps.
"""

# DIVERGENCE(C++ moses/moses/representation/instance_set.h:38-46,86-94,
# 106-110): the `GCC46_EMPLACE_BACK_WORKAROUND` branch (`#if (__GNUC__ ==
# 4) && (__GNUC_MINOR__ == 6)`) is a compiler-bug workaround for a GCC
# version from 2011. It changes `MAYBE_CONST field_set&` to a non-const
# reference, adds a hand-written `operator=`, and adds a mutable `field_set&
# fields()` overload -- none of which exist on any compiler this port will
# ever run under. Excluded outright, matching CLAUDE.md's treatment of
# other dead `#ifdef` branches (`EXEMPLAR_INST_IS_UNDEAD`, `LATER`,
# `HAVE_MPI`). Only the live (`MAYBE_CONST` = `const`) path is ported:
# `_fields` is a read-only reference and `fields()` has no mutable overload.

from __future__ import annotations

from collections.abc import Sequence

from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.scored_instance import scored_instance


class demeID_t(str):
    """C++: moses/moses/moses/types.h:190 demeID_t

    See this module's own docstring for why this lives here rather than in
    `pymetta/moses/moses/types.py`. `struct demeID_t : public std::string`
    with three numbered constructors (types.h:192-194); Python subclasses
    `str` the same way `instance.py` subclasses `list` for `std::vector`.
    """

    def __new__(
        cls,
        expansion: int = 0,
        breadth_first: int | None = None,
        ss_deme: int | None = None,
    ) -> demeID_t:
        """C++: moses/moses/moses/types.cc:43 demeID_t::demeID_t

        Three overloads by trailing-argument presence, matching
        types.cc:43-50 exactly: `str(expansion)`,
        `f"{expansion}.{breadth_first}"`, or
        `f"{expansion}.{breadth_first}.SS-{ss_deme}"`. `demeID_t()` (the
        default used throughout this module) is `demeID_t(0)` == `"0"`,
        not the empty string.
        """
        if breadth_first is None:
            if ss_deme is not None:
                raise TypeError("demeID_t requires breadth_first before ss_deme")
            text = str(expansion)
        elif ss_deme is None:
            text = f"{expansion}.{breadth_first}"
        else:
            text = f"{expansion}.{breadth_first}.SS-{ss_deme}"
        return super().__new__(cls, text)


class _ScoreView(Sequence):
    """Not itself a ported C++ symbol -- stands in for `instance_set::
    score_iterator`/`const_score_iterator` (instance_set.h:56-63), a
    `boost::transform_iterator` pair selecting `.second` out of each
    `scored_instance`. Python has no free-standing "iterator position"
    distinct from the sequence it walks, so `begin_scores()`/`end_scores()`
    (below) return a live, indexable view over `owner[start:]` instead of an
    opaque iterator: `view[k]` replaces `*std::next(it, k)`, and slicing
    (`view[k:]`) replaces `std::next(it, k)` as the start of a sub-range --
    both read through to the SAME underlying `composite_score` objects
    `owner` holds (matching `score_iterator`'s `ScoreT&` reference
    semantics), and `view[k] = new_score` (`__setitem__`) writes through the
    same way `*out_it = new_score` would via the iterator's reference.
    """

    def __init__(self, owner: instance_set, start: int) -> None:
        self._owner = owner
        self._start = start

    def __len__(self) -> int:
        return max(0, len(self._owner) - self._start)

    def __getitem__(self, index):
        if isinstance(index, slice):
            tail = list.__getitem__(self._owner, slice(self._start, None))
            return [entry.second for entry in tail[index]]
        if index < 0:
            index += len(self)
        return self._owner[self._start + index].second

    def __setitem__(self, index: int, value) -> None:
        if index < 0:
            index += len(self)
        self._owner[self._start + index].second = value


class _InstanceView(Sequence):
    """Not itself a ported C++ symbol -- stands in for `instance_set::
    const_instance_iterator` (instance_set.h:64-71) the same way `_ScoreView`
    stands in for `score_iterator`; see that class's docstring.

    Read-only: unlike `_ScoreView`, this has no `__setitem__`.
    `instance_iterator begin_instances()` -- the mutable overload -- is
    commented out in the header (instance_set.h:122-123, `//instance_iterator
    begin_instances() { ... }`); only the `const_instance_iterator`
    (read-only) overload is live. Faithfully ported as read-only, not merely
    left unimplemented.
    """

    def __init__(self, owner: instance_set, start: int) -> None:
        self._owner = owner
        self._start = start

    def __len__(self) -> int:
        return max(0, len(self._owner) - self._start)

    def __getitem__(self, index):
        if isinstance(index, slice):
            tail = list.__getitem__(self._owner, slice(self._start, None))
            return [entry.first for entry in tail[index]]
        if index < 0:
            index += len(self)
        return self._owner[self._start + index].first


class instance_set(list):
    """C++: moses/moses/representation/instance_set.h:52 instance_set"""

    def __init__(
        self,
        a: int | field_set,
        b: field_set | instance | None = None,
        c: field_set | None = None,
    ) -> None:
        """C++: moses/moses/representation/instance_set.h:74 instance_set

        Dispatches on argument shape across the header's three live
        constructors (the same single-`__init__`-covers-several-C++-
        overloads style `field_set.__init__` already uses):

        * `instance_set(n, fs)` (:74-76) -- `n` null (zero-filled)
          instances, each with a fresh worst `composite_score`.
        * `instance_set(n, inst, fs)` (:78-80) -- `n` independent copies of
          `inst`, each with its own fresh worst `composite_score`.
        * `instance_set(fs, id=demeID_t())` (:82-84) -- an empty deme.
        """
        if isinstance(a, int):
            n = a
            if n < 0:
                raise ValueError("instance_set size must be non-negative")
            if isinstance(b, field_set) and c is None:
                fs = b
                entries = [scored_instance(instance(fs.packed_width())) for _ in range(n)]
            elif isinstance(b, instance) and isinstance(c, field_set):
                fs = c
                entries = [scored_instance(b) for _ in range(n)]
            else:
                raise TypeError(
                    "instance_set(n, fs) or instance_set(n, inst, fs) expected"
                )
            self._id = demeID_t()
        elif isinstance(a, field_set):
            if c is not None:
                raise TypeError("instance_set(fs, id=...) takes no third argument")
            if b is not None and not isinstance(b, demeID_t):
                raise TypeError("instance_set(fs, id) requires a demeID_t id")
            fs = a
            entries = []
            self._id = b if b is not None else demeID_t()
        else:
            raise TypeError("instance_set requires (n, fs), (n, inst, fs), or (fs[, id])")

        self._fields = fs
        self.n_evals = 0  # C++: moses/moses/representation/instance_set.h:141 n_evals
        self.n_best_evals = 0  # C++: moses/moses/representation/instance_set.h:145 n_best_evals
        super().__init__(entries)

    def resize(self, n: int) -> None:
        """C++: moses/moses/representation/instance_set.h:98 instance_set::resize

        "Insert or erase instances so that the size becomes n. In case of
        insertions, it will insert null instances." (the header's own
        comment.) Growth appends `n - len(self)` freshly zero-filled,
        independently worst-scored instances -- not `n - len(self)` copies
        of ONE prototype, for the same aliasing reason the constructor's
        own docstring gives. Shrinkage truncates from the end, matching
        `std::vector::resize`.
        """
        if n < 0:
            raise ValueError("instance_set size must be non-negative")
        current = len(self)
        if n < current:
            del self[n:]
        elif n > current:
            width = self._fields.packed_width()
            self.extend(scored_instance(instance(width)) for _ in range(n - current))

    def fields(self) -> field_set:
        """C++: moses/moses/representation/instance_set.h:102 instance_set::fields"""
        return self._fields

    def begin_scores(self) -> _ScoreView:
        """C++: moses/moses/representation/instance_set.h:112 instance_set::begin_scores"""
        return _ScoreView(self, 0)

    def end_scores(self) -> _ScoreView:
        """C++: moses/moses/representation/instance_set.h:115 instance_set::end_scores"""
        return _ScoreView(self, len(self))

    def begin_instances(self) -> _InstanceView:
        """C++: moses/moses/representation/instance_set.h:124 instance_set::begin_instances"""
        return _InstanceView(self, 0)

    def end_instances(self) -> _InstanceView:
        """C++: moses/moses/representation/instance_set.h:127 instance_set::end_instances"""
        return _InstanceView(self, len(self))

    def setID(self, id_: demeID_t) -> None:
        """C++: moses/moses/representation/instance_set.h:131 instance_set::setID"""
        self._id = id_

    def getID(self) -> demeID_t:
        """C++: moses/moses/representation/instance_set.h:132 instance_set::getID"""
        return self._id
