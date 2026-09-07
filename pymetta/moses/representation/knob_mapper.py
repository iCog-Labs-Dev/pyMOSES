"""Identity-indexed logical knob collection used by representation."""

from __future__ import annotations

from dataclasses import dataclass

from pymetta.moses.representation.field_set import disc_spec
from pymetta.moses.representation.knobs import disc_knob_base
from pymetta.util.tree import tree_node

# DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:46-94): Task 2
# contains the logical discrete collection and identity lookup only. Continuous
# entries are deferred to that allowed domain slice. The representation that
# owns the final field_set assigns raw indexes after this mapper is finalized.


@dataclass(frozen=True, slots=True)
class disc_entry:
    """C++: moses/moses/representation/knob_mapper.h:53 knob_mapper::disc_map"""

    spec: disc_spec
    knob: disc_knob_base


class knob_mapper:
    """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper

    This mapper finalizes stable spec order and node-identity lookup only. Raw
    field indexes depend on the owning representation's completed ``field_set``.
    """

    def __init__(self) -> None:
        """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper"""
        self._disc: list[disc_entry] = []
        self._it_disc_knob: dict[tree_node, disc_knob_base] = {}
        self._finalized = False

    @property
    def disc(self) -> tuple[disc_entry, ...]:
        """C++: moses/moses/representation/knob_mapper.h:63 knob_mapper::disc"""
        return tuple(self._disc)

    def add_disc_knob(self, knob: disc_knob_base) -> None:
        """C++: moses/moses/representation/build_knobs.cc:530 add_logical_knobs"""
        if self._finalized:
            raise RuntimeError("add_disc_knob: mapper is finalized")
        # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:580-587):
        # classic constructs a multimap entry only after probing/disallowing is
        # complete. Python exposes staging explicitly and freezes multiplicity
        # at this equivalent insertion point so stored field specs cannot drift.
        spec = knob._freeze_spec()
        self._disc.append(disc_entry(spec, knob))

    def finalize(self) -> None:
        """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper"""
        if self._finalized:
            return
        # DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:63-79):
        # Python stages multimap ordering and identity-map construction behind
        # one explicit finalization call. Validation is atomic: failure leaves
        # both the insertion-ordered entries and identity index unchanged.
        pending = list(self._disc)
        locations: dict[tree_node, disc_knob_base] = {}
        for entry in pending:
            if entry.spec != entry.knob.spec():
                raise RuntimeError("finalize: stored knob specification is stale")
            loc = entry.knob.get_loc()
            if loc in locations:
                # DIVERGENCE(C++ moses/moses/representation/representation.cc:181):
                # the only write to classic's it_disc_knob map is
                # `it_disc_knob[pit] = dit;` -- std::map::operator[], which
                # silently OVERWRITES on a duplicate key, so the *last* knob
                # at a given location wins, not the first (knob_mapper.h:76
                # only declares the it_disc_knob member -- an
                # it_disc_knob_map -- it contains no insert() call of its
                # own). Python rejects the ambiguous identity mapping
                # explicitly instead of picking either "first" or "last".
                raise ValueError("finalize: location already has a knob")
            locations[loc] = entry.knob
        ordered = sorted(pending, key=lambda entry: entry.spec.multy, reverse=True)
        self._disc = ordered
        self._it_disc_knob = locations
        self._finalized = True

    def find_disc_knob(self, loc: tree_node | None) -> disc_knob_base | None:
        """C++: moses/moses/representation/knob_mapper.h:83 knob_mapper::find_disc_knob"""
        # DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:83-86):
        # Python returns the knob object or None instead of a multimap iterator
        # or disc.end(), and rejects lookup before explicit finalization.
        if not self._finalized:
            raise RuntimeError("find_disc_knob: mapper is not finalized")
        if loc is None:
            return None
        knob = self._it_disc_knob.get(loc)
        if knob is not None and knob.get_loc() is not loc:
            raise ValueError("find_disc_knob: mapped knob location is stale")
        return knob
