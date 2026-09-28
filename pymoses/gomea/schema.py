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

GOMEA extension: with ``contin="copy"`` (``--gomea-contin copy``) a
representation with contin knobs is accepted too. Each contin knob (one
``contin_spec``, i.e. ``depth`` consecutive raw fields) becomes ONE gene
whose code packs all of its raw ``{Stop, Left, Right}`` fields, two bits per
raw field (raw field ``k`` of the knob at bits ``2k .. 2k+1``), so its domain
is ``4 ** depth`` and the code round-trips every raw value exactly. GOMEA
only ever copies whole codes between individuals, so a constant is copied,
never mutated -- the paper-style constant handling of Virgolin et al. 2021,
Sec 6. Codes whose digits are not a legal step path (a digit 3) are never
produced: every code comes from a MOSES instance or from
``sample_contin_code``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.moses.representation.field_set import contin_spec, contin_stepper, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.representation import representation
from pymoses.moses.scoring.behave_cscore import _cache_key
from pymoses.util.rng import MT19937RandGen

# GOMEA extension: how a representation's contin knobs are handled
# (``--gomea-contin``): ``reject`` refuses them (the default), ``copy`` makes
# each knob one copy-only gene.
CONTIN_MODES = ("reject", "copy")


class UnsupportedRepresentation(ValueError):
    """A representation cannot be frozen into a discrete schema.

    Raised when a representation has any continuous (contin) field and the
    schema was built with ``contin="reject"`` (the default). A contin
    field's raw slots pack a variable-length ``{Stop, Left, Right}``
    step-instruction path to a floating-point value, not a fixed-domain
    category; ``contin="copy"`` (a GOMEA extension) turns each such knob into
    one copy-only gene instead -- see ``FrozenSchema``'s own docstring.
    """


class FrozenSchema:
    """A read-only categorical-genotype view of one MOSES representation's
    logical discrete fields.

    Every logical field -- a multi-valued discrete knob or a one-bit knob --
    becomes one genotype position with a fixed integer domain
    ``0 .. domain - 1``. With ``contin="reject"`` (the default) continuous
    fields are unsupported: the wrapped representation must have none, or
    construction raises ``UnsupportedRepresentation``, and MOSES's own raw
    field index (``field_set.get_raw``/``set_raw``) IS the logical field's
    genotype position: raw field ``k`` is logical field ``k``.

    GOMEA extension: with ``contin="copy"`` each contin knob is one gene at
    the front of the genotype (``contin_positions``, in ``fields.contin()``
    order, domain ``4 ** depth``; see this module's docstring), and the
    discrete/bit genes follow, one per raw field. Without contin knobs both
    modes give the identical schema.
    """

    def __init__(self, rep: representation, contin: str = "reject") -> None:
        if not isinstance(rep, representation):
            raise TypeError("FrozenSchema requires a representation")
        if contin not in CONTIN_MODES:
            raise ValueError(f"contin must be one of {CONTIN_MODES}, got {contin!r}")

        fields = rep.fields()
        if fields.n_contin_fields() > 0 and contin != "copy":
            raise UnsupportedRepresentation(
                "FrozenSchema does not support representations with continuous fields"
            )

        self.rep: representation = rep
        self.fields: field_set = fields
        self.contin_mode: str = contin
        specs = fields.contin()
        # Genes 0 .. len(specs)-1 are the contin knobs (MOSES orders contin
        # raw fields first); the discrete/bit genes follow, one per raw
        # field. Without contin knobs this is exactly the discrete layout.
        self.contin_positions: tuple[int, ...] = tuple(range(len(specs)))
        self.contin_depths: tuple[int, ...] = tuple(c.depth for c in specs)
        self._contin_specs: tuple[contin_spec, ...] = specs
        self._contin_raw: tuple[int, ...] = tuple(
            fields.contin_to_raw_idx(i) for i in range(len(specs))
        )
        self._disc_raw0: int = fields.n_contin_fields()
        self._contin_values: dict[tuple[int, int], float] = {}
        self.dim: int = len(specs) + fields.raw_size() - self._disc_raw0
        self.domains: tuple[int, ...] = tuple(4**c.depth for c in specs) + tuple(
            spec.multy for spec in fields.disc_and_bit()
        )
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
        """Read every logical field's current raw value out of ``inst``
        (a contin gene reads all of its knob's raw fields into one code)."""
        get_raw = self.fields.get_raw
        if not self.contin_positions:
            return tuple(get_raw(inst, k) for k in range(self.dim))
        codes = []
        for begin, depth in zip(self._contin_raw, self.contin_depths, strict=True):
            code = 0
            for k in range(depth):
                code |= get_raw(inst, begin + k) << (2 * k)
            codes.append(code)
        n_raw = self.fields.raw_size()
        return tuple(codes) + tuple(get_raw(inst, k) for k in range(self._disc_raw0, n_raw))

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
        if self.contin_positions:
            n = len(self.contin_positions)
            raw = []
            for code, depth in zip(values[:n], self.contin_depths, strict=True):
                raw.extend((code >> (2 * k)) & 3 for k in range(depth))
            values = raw + values[n:]
        return self.fields.pack(values)

    # GOMEA extension: contin genes (``--gomea-contin copy``).
    def contin_value(self, pos: int, code: int) -> float:
        """The constant that contin gene ``pos`` with ``code`` decodes to
        (``field_set.get_contin``'s step walk, memoized per code)."""
        key = (pos, code)
        value = self._contin_values.get(key)
        if value is None:
            stepper = contin_stepper(self._contin_specs[pos])
            for k in range(self.contin_depths[pos]):
                step = (code >> (2 * k)) & 3
                if step == contin_spec.Left:
                    stepper.left()
                elif step == contin_spec.Right:
                    stepper.right()
                elif step == contin_spec.Stop:
                    break
                else:
                    raise ValueError(f"contin gene {pos} code {code} has an invalid step")
            value = stepper.value
            self._contin_values[key] = value
        return value

    def sample_contin_code(self, pos: int, rng: MT19937RandGen) -> int:
        """A code drawn uniformly over contin gene ``pos``'s legal step paths.

        A legal path is ``L`` Left/Right steps followed by Stops, for
        ``0 <= L <= depth``: ``2 ** (depth + 1) - 1`` paths, each one of the
        knob's representable constants. One ``rng.randint_n`` draw picks
        path number ``j``; ``L`` is the largest length with ``2 ** L - 1 <=
        j`` and the bits of ``j - (2 ** L - 1)`` choose Left (0) or Right
        (1) step by step.
        """
        depth = self.contin_depths[pos]
        j = rng.randint_n((1 << (depth + 1)) - 1)
        length = (j + 1).bit_length() - 1
        bits = j - ((1 << length) - 1)
        code = 0
        for k in range(length):
            step = contin_spec.Right if (bits >> k) & 1 else contin_spec.Left
            code |= step << (2 * k)
        return code

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
