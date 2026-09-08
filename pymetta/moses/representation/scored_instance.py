"""C++: moses/moses/representation/scored_instance.h:33 scored_instance

`scored_instance<ScoreT>` inherits `opencog::tagged_item<instance, ScoreT>`
(``opencog/util/functional.h:53``), itself `public std::pair<Item, Tag>` plus
`boost::less_than_comparable<tagged_item<Item, Tag>>`. Python has no template
mechanism, so both layers are folded into one class here rather than mirrored
as a base/derived pair -- there is nothing else in this port that would
specialize `tagged_item` on its own.

`.first`/`.second` are kept as the attribute names (not e.g. `.instance`/
`.score`) because every real call site addresses a scored_instance this way
-- `deme[i].first` (the instance), `deme[i].second` (the composite_score) --
throughout hill-climbing.cc, merging.cc and subsample.cc.

C++'s `std::pair` copies both members by value at construction; `tagged_item`
and `scored_instance` add nothing that changes this. This port copies `i`
(via `instance(i)`, which already value-copies -- see instance.py) and `s`
(via `copy.copy`, since `composite_score` holds only plain float/int fields,
so a shallow copy is a full value copy) for the same reason CARRY-FORWARD.md
1.1 gives for `Table.__init__`: two scored_instance entries built from the
same source objects (e.g. `instance_set`'s fill-construction, one prototype
instance copied into every deme slot) must never end up sharing one mutable
`instance` list or one mutable `composite_score` -- mutating one entry would
then silently mutate every other, the same aliasing-defect shape S8 hit in
`representation.py`'s `get_candidate_rec` (CARRY-FORWARD.md 2.4).
"""

# DIVERGENCE(C++ moses/moses/representation/scored_instance.h:33, and
# opencog/util/functional.h:53): `scored_instance` is a class template over
# `ScoreT`; this port fixes `ScoreT = composite_score` directly. Every
# instantiation anywhere in classic MOSES -- moses/moses/ (optimization.h:171
# `typedef scored_instance<composite_score> deme_inst_t`, hill-climbing.cc,
# merging.cc:507 `std::greater<scored_instance<composite_score>>`,
# subsample.cc), feature-selection/ (deme_optimize.h), and tests/
# (optimizationUTest.cxxtest) alike -- uses `composite_score`; grepping the
# whole classic tree for `scored_instance<` turns up no other type argument.
#
# DIVERGENCE(C++ opencog/util/functional.h:41-42): the generic
# `template<class T1, class T2> scored_instance(const std::pair<T1, T2>&)`
# conversion overload is dropped. `git grep 'scored_instance('` over
# classic's `moses/` and `tests/` finds zero explicit constructor calls
# anywhere -- every scored_instance in the real codebase is built implicitly
# by `std::vector`'s own fill/copy machinery (see instance_set.py's module
# docstring), never by a literal `scored_instance(...)` call site, so there
# is no real-world shape this overload would need to match.

from __future__ import annotations

import copy

from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.instance import instance


class scored_instance:
    """C++: moses/moses/representation/scored_instance.h:33 scored_instance"""

    __slots__ = ("first", "second")

    def __init__(self, i: instance | None = None, s: composite_score | None = None) -> None:
        """C++: moses/moses/representation/scored_instance.h:38 scored_instance

        Covers three of the header's four constructors by argument shape:

        * `scored_instance()` (:40) -- both `i` and `s` omitted. `tagged_item`'s
          own empty constructor (functional.h:61) default-constructs the pair,
          i.e. an empty `instance` and a fresh (worst) `composite_score`.
        * `scored_instance(const instance& i)` (:39) -- `s` omitted. Copies
          `i`; `tagged_item(const Item&)` (functional.h:60) default-constructs
          `Tag()` for the score, i.e. a fresh `composite_score()`.
        * `scored_instance(const instance& i, const ScoreT& s)` (:38) -- both
          given. Copies both.

        The fourth overload, the generic pair-conversion constructor
        (scored_instance.h:41-42), is not ported -- see the module
        docstring's DIVERGENCE note.
        """
        if i is None:
            if s is not None:
                raise TypeError("scored_instance requires an instance to pair with a score")
            self.first = instance()
            self.second = composite_score()
            return
        if not isinstance(i, instance):
            raise TypeError("scored_instance requires an instance")
        self.first = instance(i)
        self.second = composite_score() if s is None else copy.copy(s)

    def __lt__(self, other: object) -> bool:
        """C++: moses/moses/representation/scored_instance.h:44 scored_instance

        Non-identifier member per the project's convention (see `__eq__`
        below, and `types.py`'s `composite_score.__lt__` docstring, for
        the same pattern): cites the enclosing class at the member's own
        line. The real member is `scored_instance::operator<`
        (scored_instance.h:44-47): `this->second < other.second`.
        """
        if not isinstance(other, scored_instance):
            return NotImplemented
        return self.second < other.second

    def __gt__(self, other: object) -> bool:
        """C++: moses/moses/representation/scored_instance.h:48 scored_instance

        Non-identifier member (see `__lt__` above). The real member is
        `scored_instance::operator>` (scored_instance.h:48-51): `this->
        second > other.second`.
        """
        if not isinstance(other, scored_instance):
            return NotImplemented
        return self.second > other.second

    def __eq__(self, other: object) -> bool:
        """C++: opencog/util/functional.h:53 tagged_item

        Non-identifier member per the project's convention (see
        `types.py`'s `composite_score.__lt__` docstring for the same
        pattern): cites the enclosing class at its own declaration line.
        The real member is `std::pair<Item, Tag>::operator==`, inherited
        unchanged by both `tagged_item` (functional.h:53-54, `public
        std::pair<Item, Tag>`) and `scored_instance` (scored_instance.h:34,
        `public tagged_item<instance, ScoreT>`).
        """
        # DIVERGENCE(C++ opencog/util/functional.h:53-85): no
        # `scored_instance` call site anywhere in classic (moses/,
        # feature-selection/, tests/) ever compares two instances with
        # `==` -- see module docstring. Ported anyway: Python's default
        # `object.__eq__`
        # (identity) would otherwise silently diverge from C++'s inherited
        # `std::pair` value equality, the same class of gap `__hash__ =
        # None` below exists to flag rather than leave implicit.
        if not isinstance(other, scored_instance):
            return NotImplemented
        return self.first == other.first and self.second == other.second

    def __le__(self, other: object) -> bool:
        """C++: opencog/util/functional.h:53 tagged_item

        Non-identifier member (see `__eq__` above for the citation
        convention). The real member is `boost::less_than_comparable
        <tagged_item<Item, Tag>>`'s CRTP-generated `operator<=` --
        `a<=b := !(b<a)` -- using `tagged_item::operator<` (`this->second <
        r.second`, functional.h:65-67), found via the derived-class ADL
        C++ grants a base class's friend operators. `scored_instance` does
        not redefine `<=`, so this is exactly that inherited comparison --
        identical in effect to `not (other.second < self.second)`, i.e. to
        this class's own `<` on `.second` alone.
        """
        if not isinstance(other, scored_instance):
            return NotImplemented
        return not (other.second < self.second)

    def __ge__(self, other: object) -> bool:
        """C++: opencog/util/functional.h:53 tagged_item

        Non-identifier member (see `__eq__` above). The real member is
        `boost::less_than_comparable<tagged_item<Item, Tag>>`'s CRTP-
        generated `operator>=` -- `a>=b := !(a<b)` -- same derivation as
        `__le__` above.
        """
        if not isinstance(other, scored_instance):
            return NotImplemented
        return not (self.second < other.second)

    __hash__ = None  # type: ignore[assignment]
