"""Freeze a MOSES representation's logical discrete fields into a fixed
tuple of categorical genotype positions.

A GOMEA-style population optimizer needs to treat one candidate as a plain
tuple of small integers with a fixed per-position domain -- it has no notion
of MOSES's own packed instance words, knob objects, or continuous step
encodings. ``FrozenSchema`` is the read-only adapter that makes one
``representation`` look like that: it fixes the genotype's length and each
position's legal value range once, up front, and then only ever translates
between that genotype tuple and MOSES's own ``instance``/``combo_tree``
types -- it never mutates the representation it wraps.

See ``docs/gomea/P1.0-AUDIT.md`` Sec 4 for the representation/field/sampler
facts this module relies on: fields are ordered contin-raw-fields-then-disc-
then-bit, a discrete field's legal values are ``0 .. multy-1``, and the
zero instance always decodes to the exemplar.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.representation import representation
from pymoses.moses.scoring.behave_cscore import _cache_key


class UnsupportedRepresentation(ValueError):
    """A representation cannot be frozen into a discrete schema.

    Raised when a representation has any continuous (contin) field. A
    contin field's raw slots pack a variable-length ``{Stop, Left, Right}``
    step-instruction path to a floating-point value, not a fixed-domain
    category, so it has no place in a tuple-of-categories genotype -- see
    ``FrozenSchema``'s own docstring.
    """


class FrozenSchema:
    """A read-only categorical-genotype view of one MOSES representation's
    logical discrete fields.

    Every logical field -- a multi-valued discrete knob or a one-bit knob --
    becomes one genotype position with a fixed integer domain
    ``0 .. domain - 1``. Continuous fields are unsupported: the wrapped
    representation must have none, or construction raises
    ``UnsupportedRepresentation``.

    Because contin fields are rejected, MOSES's own raw field index
    (``field_set.get_raw``/``set_raw``) already IS the logical field's
    genotype position: raw field ``k`` is logical field ``k``, with nothing
    ahead of it to offset the numbering (contin fields, when present, are
    always ordered first).
    """

    def __init__(self, rep: representation) -> None:
        if not isinstance(rep, representation):
            raise TypeError("FrozenSchema requires a representation")

        fields = rep.fields()
        if fields.n_contin_fields() > 0:
            raise UnsupportedRepresentation(
                "FrozenSchema does not support representations with continuous fields"
            )

        self.rep: representation = rep
        self.fields: field_set = fields
        self.dim: int = fields.raw_size()
        self.domains: tuple[int, ...] = tuple(spec.multy for spec in fields.disc_and_bit())
        self.schema_id: str = self._compute_schema_id()

    def _compute_schema_id(self) -> str:
        layout = [(f.width, f.major_offset, f.minor_offset) for f in self.fields.fields()]
        text = (
            self.rep.ostream_prototype() + "|" + repr(self.domains) + "|" + repr(layout)
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def default_genotype(self) -> tuple[int, ...]:
        """The all-zero genotype, matching MOSES's own default instance.

        Value 0 of every knob is the exemplar's own setting (see this
        module's docstring), so this decodes to the exemplar itself.
        """
        return self.to_genotype(instance(self.fields.packed_width()))

    def to_genotype(self, inst: instance) -> tuple[int, ...]:
        """Read every logical field's current raw value out of ``inst``."""
        return tuple(self.fields.get_raw(inst, k) for k in range(self.dim))

    def to_instance(self, g: Sequence[int]) -> instance:
        """Pack a genotype into a fresh MOSES ``instance``.

        Raises ``ValueError`` if ``g`` has the wrong length or any value is
        outside its field's domain.
        """
        values = list(g)
        if len(values) != self.dim:
            raise ValueError(f"genotype has {len(values)} values; schema requires {self.dim}")
        for k, value in enumerate(values):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"genotype value at field {k} is not an int: {value!r}")
            if not 0 <= value < self.domains[k]:
                raise ValueError(
                    f"genotype value {value} at field {k} is out of domain "
                    f"0..{self.domains[k] - 1}"
                )
        return self.fields.pack(values)

    def phenotype_key(self, g: Sequence[int]) -> tuple:
        """A collision-free key identifying ``g``'s scored phenotype.

        Two genotypes share a ``phenotype_key`` iff their UNREDUCED decoded
        trees are structurally identical -- which, since
        ``get_candidate(inst, True) == reducer(get_candidate(inst, False))``
        (``representation.clean_combo_tree``), also proves their REDUCED
        trees (what a scorer actually sees) are identical.
        """
        inst = self.to_instance(g)
        return _cache_key(self.rep.get_candidate(inst, False))

    def decode(self, g: Sequence[int]) -> combo_tree:
        """The reduced combo tree ``g`` decodes to."""
        inst = self.to_instance(g)
        return self.rep.get_candidate(inst, True)
