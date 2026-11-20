"""NOT A PORT: the IT linkage study (BACKLOG 13.9). ``ITGenome``, the IT (interaction -
transformation) form of de Franca and Aldeia (Evolutionary Computation 2021) as a
fixed-length genome, with the sparse factor encoding of ``docs/gomea/T3.4-FEASIBILITY.md``
(plan section 5.3). The transformation is the identity only.

* m term slots (default 8), f factor slots per term (default 2), two boxes per factor: the
  **input** box and the **power** box. Position ``(t * f + k) * 2`` is the input box of factor
  k of term t, the next position is its power box (32 positions by default).
* An input box has n + 1 codes: 0 = "none", j + 1 = input j (inputs are 0-based here and
  ``$(j + 1)`` in MOSES syntax). A power box has one code for each power of ``powers``
  (default -2, -1, 1, 2).
* A factor is input^power, or nothing when its input is "none". A term is the product of its
  factors; a term with no factor is off. The powers of one input in one term add up (x^1 x^1 =
  x^2; x^1 x^-1 = 1, and then the input leaves the term). A term whose powers all cancel is
  off, as a constant term is aliased with the intercept.

**Rules applied before the phenotype key** (so equal models get equal keys):

* An input with a zero in the training rows (``zero_inputs``) never takes a negative power:
  the power becomes its absolute value (-2 -> 2, -1 -> 1). ``decode`` counts these
  (``forced``). The domains do not change.
* A 0/1 input (``zero_one_inputs``: every training value is 0 or 1) has x^k = x, so its power
  is always 1 (after the rule above, as it has zeros).
* Equal monomials in two slots are merged into one term (the fit then never gets two equal
  columns). Positions never move: the slot-swap problem is handled only by this merge.

**Arithmetic.** ``monomial_column`` does exactly the operations that the MOSES interpreter
does on the printed term (``model_to_moses``), so ``tools/gomea/sr_eval.py`` gets
bit-identical term columns: x^k (k > 0) is ``*($j ... $j)`` (a left-to-right product), x^-k is
``/(1 x^k)``, and a term of several inputs is ``*(F1 F2 ...)`` in input order. The interpreter
stops a product at a zero factor and raises an overflow for a quotient that is not finite;
``nan`` marks the overflow here (as ``sr_eval`` does for the row).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_POWERS = (-2, -1, 1, 2)

Monomial = tuple[tuple[int, int], ...]  # sorted (input, power) pairs, powers not 0


@dataclass(frozen=True)
class ITModel:
    """A decoded IT genome: ``monomials`` (the distinct terms, sorted; the phenotype key),
    ``slots`` (the monomial of each term slot, ``None`` for a slot that is off) and
    ``forced`` (the negative powers made positive by the zero rule)."""

    monomials: tuple[Monomial, ...]
    slots: tuple[Monomial | None, ...]
    forced: int


def nodes_of(program: str) -> int:
    """The node count of a MOSES program: its "(" plus its atoms (the rule of
    ``tools/gomea/phase_report.py``, ``_nodes``)."""
    return program.count("(") + len(program.replace("(", " ").replace(")", " ").split())


def _power_column(x: Sequence[float], k: int) -> list[float]:
    """x^k for k >= 1, as the interpreter computes ``*($j ... $j)`` (left to right)."""
    if k == 1:
        return list(x)
    if k == 2:
        return [v * v for v in x]
    if k == 3:
        return [v * v * v for v in x]
    if k == 4:
        return [v * v * v * v for v in x]
    out = []
    for v in x:
        r = v
        for _ in range(k - 1):
            r *= v
        out.append(r)
    return out


def _reciprocal(v: float) -> float:
    """``/(1 v)`` with the interpreter's rules: an overflow (``nan``) when ``v`` is 0 or the
    quotient is not finite."""
    if v == 0.0 or v != v:
        return math.nan
    r = 1.0 / v
    return r if math.isfinite(r) else math.nan


class ITGenome:
    """The IT genome over ``n_inputs`` inputs (see the module docstring).

    ``zero_inputs``, ``zero_one_inputs``: the inputs with a zero in the training rows, and
    those whose training values are all 0 or 1. ``bool_inputs``: those that
    ``sr_eval.read_table`` types as boolean on the training table; ``model_to_moses`` writes
    them as ``impulse($k)`` (the number 0 or 1).
    """

    kind = "it"

    def __init__(
        self,
        n_inputs: int,
        terms: int = 8,
        factors: int = 2,
        powers: Sequence[int] = DEFAULT_POWERS,
        zero_inputs=frozenset(),
        zero_one_inputs=frozenset(),
        bool_inputs=frozenset(),
    ) -> None:
        powers = tuple(int(p) for p in powers)
        if n_inputs < 1:
            raise ValueError("the IT genome needs at least one input")
        if terms < 1 or factors < 1:
            raise ValueError("the IT genome needs at least one term slot and one factor slot")
        if not powers or 0 in powers or len(set(powers)) != len(powers):
            raise ValueError(f"the powers must be distinct and not 0, got {powers}")
        self.n_inputs = n_inputs
        self.terms = terms
        self.factors = factors
        self.powers = powers
        self.length = terms * factors * 2
        self.domains = tuple([n_inputs + 1, len(powers)] * (terms * factors))
        self.zero_inputs = frozenset(zero_inputs)
        self.zero_one_inputs = frozenset(zero_one_inputs)
        self.bool_inputs = frozenset(bool_inputs)
        self._zero = [j in self.zero_inputs for j in range(n_inputs)]
        self._zero_one = [j in self.zero_one_inputs for j in range(n_inputs)]
        self._intern: dict[Monomial, Monomial] = {}

    # ---- decoding

    def decode(self, g: Sequence[int]) -> ITModel:
        f, powers, zero, zero_one = self.factors, self.powers, self._zero, self._zero_one
        intern = self._intern
        forced = 0
        slots: list[Monomial | None] = []
        for t in range(self.terms):
            acc: dict[int, int] = {}
            base = t * f * 2
            for k in range(f):
                code = g[base + 2 * k]
                if code == 0:
                    continue
                j = code - 1
                p = powers[g[base + 2 * k + 1]]
                if p < 0 and zero[j]:
                    p = -p
                    forced += 1
                if zero_one[j]:
                    p = 1
                acc[j] = acc.get(j, 0) + p
            if not acc:
                slots.append(None)
                continue
            mono = tuple(sorted((j, 1 if zero_one[j] else p) for j, p in acc.items() if p != 0))
            if not mono:
                slots.append(None)
                continue
            slots.append(intern.setdefault(mono, mono))
        monomials = tuple(sorted({m for m in slots if m is not None}))
        return ITModel(monomials, tuple(slots), forced)

    def phenotype_key(self, g: Sequence[int]) -> tuple[Monomial, ...]:
        """The sorted tuple of the distinct active monomials (each a sorted tuple of (input,
        power) pairs)."""
        return self.decode(g).monomials

    def active_mask(self, g: Sequence[int]) -> tuple[bool, ...]:
        """Which positions can affect the model: every input box, and the power box of a
        factor whose input is not "none" and not a 0/1 input."""
        mask = []
        for i in range(0, self.length, 2):
            code = g[i]
            mask.append(True)
            mask.append(code != 0 and not self._zero_one[code - 1])
        return tuple(mask)

    # ---- MOSES syntax and size

    def _input_moses(self, j: int) -> str:
        return f"impulse(${j + 1})" if j in self.bool_inputs else f"${j + 1}"

    def monomial_to_moses(self, m: Monomial) -> str:
        parts = []
        for j, p in m:
            x = self._input_moses(j)
            k = abs(p)
            xk = x if k == 1 else "*(" + " ".join([x] * k) + ")"
            parts.append(xk if p > 0 else f"/(1 {xk})")
        return parts[0] if len(parts) == 1 else "*(" + " ".join(parts) + ")"

    def model_to_moses(self, monomials: Sequence[Monomial]) -> str:
        """``+(t1 t2 ...)``; one term alone without the ``+``; ``0`` for the empty model."""
        if not monomials:
            return "0"
        terms = [self.monomial_to_moses(m) for m in monomials]
        return terms[0] if len(terms) == 1 else "+(" + " ".join(terms) + ")"

    def to_moses(self, g: Sequence[int]) -> str:
        """All distinct active terms of ``g`` in MOSES syntax. The run prints only the terms
        that the fit keeps (``evaluator.ITScorer.fit``, ``model_to_moses``)."""
        return self.model_to_moses(self.decode(g).monomials)

    def model_size(self, monomials: Sequence[Monomial], terms_fit: bool = True) -> int:
        """The node count of the printed model, plus the number of terms when the per-term
        fit is used for 2 or more terms (``run_table.terms_size``)."""
        n = nodes_of(self.model_to_moses(monomials))
        return n + (len(monomials) if terms_fit and len(monomials) >= 2 else 0)

    def size(self, g: Sequence[int]) -> int:
        return self.model_size(self.decode(g).monomials)

    # ---- evaluation

    def monomial_column(self, m: Monomial, cols: Sequence[Sequence[float]]) -> list[float]:
        """The value of monomial ``m`` on every row (``cols[j]`` is input j), with the
        interpreter's rules; ``nan`` marks an overflow."""
        out: list[float] | None = None
        for j, p in m:
            c = _power_column(cols[j], abs(p))
            if p < 0:
                c = [_reciprocal(v) for v in c]
            out = c if out is None else [a * b if a else a for a, b in zip(out, c, strict=True)]
        assert out is not None
        return out
