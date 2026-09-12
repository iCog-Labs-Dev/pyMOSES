"""Identity-indexed discrete and continuous knob collection used by
representation."""

from __future__ import annotations

from dataclasses import dataclass

from pymoses.moses.representation.field_set import contin_spec, disc_spec
from pymoses.moses.representation.knobs import contin_knob, disc_knob_base
from pymoses.util.tree import tree_node

# DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:46-94): S14 Task 2
# adds the continuous collection and identity lookup alongside the logical
# discrete ones already here. The representation that owns the final
# field_set still assigns raw indexes after this mapper is finalized.


@dataclass(frozen=True, slots=True)
class disc_entry:
    """C++: moses/moses/representation/knob_mapper.h:53 knob_mapper::disc_map"""

    spec: disc_spec
    knob: disc_knob_base


@dataclass(frozen=True, slots=True)
class contin_entry:
    """C++: moses/moses/representation/knob_mapper.h:59 knob_mapper::contin_map"""

    spec: contin_spec
    knob: contin_knob


class knob_mapper:
    """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper

    This mapper finalizes stable spec order and node-identity lookup only. Raw
    field indexes depend on the owning representation's completed ``field_set``.
    """

    def __init__(self) -> None:
        """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper"""
        self._disc: list[disc_entry] = []
        self._contin: list[contin_entry] = []
        self._it_disc_knob: dict[tree_node, disc_knob_base] = {}
        self._it_contin_knob: dict[tree_node, contin_knob] = {}
        self._finalized = False

    @property
    def disc(self) -> tuple[disc_entry, ...]:
        """C++: moses/moses/representation/knob_mapper.h:63 knob_mapper::disc"""
        return tuple(self._disc)

    @property
    def contin(self) -> tuple[contin_entry, ...]:
        """C++: moses/moses/representation/knob_mapper.h:64 knob_mapper::contin"""
        return tuple(self._contin)

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

    def add_contin_knob(self, knob: contin_knob) -> None:
        """C++: moses/moses/representation/build_knobs.cc:769 build_contin
        (``_rep.contin.insert(make_pair(kb.spec(), kb))``)

        Unlike ``add_disc_knob``, there is no staged probe/disallow step to
        freeze against: a ``contin_knob``'s spec is fixed at its own
        construction (knobs.py) and never mutates afterward, so it is simply
        recorded here in build order.
        """
        if self._finalized:
            raise RuntimeError("add_contin_knob: mapper is finalized")
        self._contin.append(contin_entry(knob.spec(), knob))

    def finalize(self) -> None:
        """C++: moses/moses/representation/knob_mapper.h:46 knob_mapper"""
        if self._finalized:
            return
        # DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:63-79):
        # Python stages multimap ordering and identity-map construction behind
        # one explicit finalization call. Validation is atomic: failure leaves
        # both the insertion-ordered entries and identity index unchanged.
        pending_disc = list(self._disc)
        disc_locations: dict[tree_node, disc_knob_base] = {}
        for entry in pending_disc:
            if entry.spec != entry.knob.spec():
                raise RuntimeError("finalize: stored knob specification is stale")
            loc = entry.knob.get_loc()
            if loc in disc_locations:
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
            disc_locations[loc] = entry.knob

        pending_contin = list(self._contin)
        contin_locations: dict[tree_node, contin_knob] = {}
        for entry in pending_contin:
            if entry.spec != entry.knob.spec():
                raise RuntimeError("finalize: stored knob specification is stale")
            loc = entry.knob.get_loc()
            if loc in contin_locations:
                # Same duplicate-key policy as disc_locations above, applied
                # to classic's SEPARATE it_contin_knob map
                # (representation.cc:196's `it_contin_knob[pit] = cit;`).
                raise ValueError("finalize: location already has a knob")
            contin_locations[loc] = entry.knob

        # C++: disc/contin are two SEPARATE std::multimaps, each ordered by
        # its own spec's operator< (field_set.h:158/191, both "sort
        # descending"). Python sorts each list independently on the same
        # descending key, `sorted`'s stability preserving insertion order
        # for exact ties -- the same "insertion order wins a libstdc++
        # multiset tie" reasoning this class's own disc sort already
        # documents (see representation.py's combined field_set build for
        # where these two independently-ordered sequences are merged).
        self._disc = sorted(pending_disc, key=lambda entry: entry.spec.multy, reverse=True)
        self._contin = sorted(
            pending_contin,
            key=lambda entry: (
                entry.spec.depth,
                entry.spec.expansion,
                entry.spec.step_size,
                entry.spec.mean,
            ),
            reverse=True,
        )
        self._it_disc_knob = disc_locations
        self._it_contin_knob = contin_locations
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

    def find_contin_knob(self, loc: tree_node | None) -> contin_knob | None:
        """C++: moses/moses/representation/knob_mapper.h:88 knob_mapper::find_contin_knob"""
        # DIVERGENCE(C++ moses/moses/representation/knob_mapper.h:88-91): same
        # shape as find_disc_knob's own divergence note above.
        if not self._finalized:
            raise RuntimeError("find_contin_knob: mapper is not finalized")
        if loc is None:
            return None
        knob = self._it_contin_knob.get(loc)
        if knob is not None and knob.get_loc() is not loc:
            raise ValueError("find_contin_knob: mapped knob location is stale")
        return knob
