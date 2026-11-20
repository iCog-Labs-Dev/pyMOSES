"""NOT A PORT: the IT linkage study (BACKLOG 13.9). ``TemplateGenome``, GP-GOMEA's tree
template as a fixed-length genome (plan section 5.2).

* A full binary tree of height h (default 4): 2^(h+1) - 1 positions in breadth-first order.
  Position 0 is the root; the children of p are 2p + 1 and 2p + 2. The first 2^h - 1
  positions are internal, the others are leaves.
* Codes. An internal position holds a function (codes 0 .. 3: ``+ - * aq``) or an input
  (code 4 + j for input j). A leaf holds an input (code j). Inputs are 0-based here and
  ``$(j + 1)`` in MOSES syntax.
* Decoding starts at the root. A position that holds an input is a leaf of the expression;
  all positions below it are inactive (introns), as in GP-GOMEA.

**Arithmetic.** The column-wise evaluation does exactly the operations that the MOSES
interpreter does on ``to_moses(g)`` (``pymoses.comboreduct.interpreter``), so that
``tools/gomea/sr_eval.py`` gives bit-identical predictions:

* ``a - b`` is printed ``+(a *(-1 b))``; the interpreter computes ``a + (-b)``, which IEEE 754
  makes equal to ``a - b``.
* ``a * b``: the interpreter returns ``a`` at once when ``a == 0`` and does not evaluate ``b``.
* ``aq(a, b) = a / sqrt(1 + b^2)`` (GP-GOMEA) is printed
  ``/(a exp(*(0.5 log(+(1 *(b b))))))``, so the value is ``a / exp(0.5 log(1 + b b))``. The
  interpreter returns ``a`` at once when ``a == 0``. It raises an overflow when the log or the
  quotient is not finite.
* An interpreter overflow makes the row ``nan`` in ``sr_eval``. Here ``nan`` marks it. A
  ``nan`` goes through every operation, except where the interpreter skips the second operand
  (``a == 0`` in ``*`` and in ``aq``), and there the result is the same as in the interpreter.

A model with a value that is not finite on a training row is invalid (score -inf, in
``evaluator.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

FUNCTIONS = ("+", "-", "*", "aq")
ADD, SUB, MUL, AQ = -1, -2, -3, -4  # postfix op tokens; an input is its index j >= 0
_OP_OF_CODE = (ADD, SUB, MUL, AQ)
_OP_NAME = {ADD: "+", SUB: "-", MUL: "*", AQ: "aq"}


def _aq(a: float, b: float) -> float:
    """``/(a exp(*(0.5 log(+(1 *(b b))))))`` with the interpreter's rules."""
    if not a:  # 0.0 or -0.0: the interpreter returns a at once
        return a
    t = 1.0 + b * b
    if not math.isfinite(t):
        return math.nan  # log(inf) or log(nan): an overflow
    r = a / math.exp(0.5 * math.log(t))
    return r if math.isfinite(r) else math.nan


def _aq_column(A: Sequence[float], B: Sequence[float]) -> list[float]:
    """``_aq`` on every row. When every ``a`` and every ``1 + b b`` is finite, the log and the
    exp cannot fail and every quotient is finite (the divisor is at least 1), so the same
    values come from whole-column calls; otherwise each row takes ``_aq``."""
    t = [1.0 + b * b for b in B]
    if all(map(math.isfinite, t)) and all(map(math.isfinite, A)):
        e = list(map(math.exp, [0.5 * v for v in map(math.log, t)]))
        return [a / d if a else a for a, d in zip(A, e, strict=True)]
    return [_aq(a, b) for a, b in zip(A, B, strict=True)]


class TemplateGenome:
    """GP-GOMEA's template of height ``height`` over ``n_inputs`` inputs.

    ``bool_inputs`` are the inputs that ``sr_eval.read_table`` types as boolean (every token 0
    or 1) on the training table. MOSES passes them as logical values, so ``to_moses`` writes
    them as ``impulse($k)`` (the number 0 or 1). It does not change the arithmetic.
    """

    kind = "template"

    def __init__(self, n_inputs: int, height: int = 4, bool_inputs=frozenset()) -> None:
        if n_inputs < 1:
            raise ValueError("the template needs at least one input")
        if height < 1:
            raise ValueError("the template height must be at least 1")
        self.n_inputs = n_inputs
        self.height = height
        self.length = 2 ** (height + 1) - 1
        self.n_internal = 2**height - 1
        self.domains = tuple(
            [len(FUNCTIONS) + n_inputs] * self.n_internal
            + [n_inputs] * (self.length - self.n_internal)
        )
        self.bool_inputs = frozenset(bool_inputs)

    # ---- decoding

    def node(self, g: Sequence[int], p: int) -> tuple[str, int]:
        """``("f", op)`` for a function (op is the postfix token) or ``("x", j)`` for input
        ``j`` at position ``p``."""
        c = g[p]
        if p < self.n_internal:
            if c < len(FUNCTIONS):
                return "f", _OP_OF_CODE[c]
            return "x", c - len(FUNCTIONS)
        return "x", c

    def decode(self, g: Sequence[int]) -> tuple[str, tuple[int, ...]]:
        """``(key, program)``: the active expression as a string (the phenotype key) and as
        a postfix tuple (an input is its index, a function its negative token)."""
        n_int, nf = self.n_internal, len(FUNCTIONS)
        post: list[int] = []

        def walk(p: int) -> str:
            c = g[p]
            if p < n_int and c < nf:
                left = walk(2 * p + 1)
                right = walk(2 * p + 2)
                op = _OP_OF_CODE[c]
                post.append(op)
                return f"{FUNCTIONS[c]}({left},{right})"
            j = c - nf if p < n_int else c
            post.append(j)
            return f"x{j}"

        key = walk(0)
        return key, tuple(post)

    def phenotype_key(self, g: Sequence[int]) -> str:
        """The string of the active expression: equal keys only for equal expressions."""
        return self.decode(g)[0]

    def active_mask(self, g: Sequence[int]) -> tuple[bool, ...]:
        """Which positions affect the model: the nodes reachable from the root through
        function nodes."""
        mask = [False] * self.length
        stack = [0]
        n_int, nf = self.n_internal, len(FUNCTIONS)
        while stack:
            p = stack.pop()
            mask[p] = True
            if p < n_int and g[p] < nf:
                stack.append(2 * p + 1)
                stack.append(2 * p + 2)
        return tuple(mask)

    def size(self, g: Sequence[int]) -> int:
        """The number of active nodes (functions and inputs), as GP-GOMEA's "Number of
        nodes"."""
        return sum(self.active_mask(g))

    # ---- MOSES syntax

    def _input_moses(self, j: int) -> str:
        return f"impulse(${j + 1})" if j in self.bool_inputs else f"${j + 1}"

    def program_to_moses(self, program: Sequence[int]) -> str:
        stack: list[str] = []
        for t in program:
            if t >= 0:
                stack.append(self._input_moses(t))
                continue
            b = stack.pop()
            a = stack.pop()
            if t == ADD:
                stack.append(f"+({a} {b})")
            elif t == SUB:
                stack.append(f"+({a} *(-1 {b}))")
            elif t == MUL:
                stack.append(f"*({a} {b})")
            else:
                stack.append(f"/({a} exp(*(0.5 log(+(1 *({b} {b}))))))")
        return stack[0]

    def to_moses(self, g: Sequence[int]) -> str:
        """The model in MOSES combo syntax (see the module docstring for ``-`` and ``aq``)."""
        return self.program_to_moses(self.decode(g)[1])

    def program_to_infix(self, program: Sequence[int]) -> str:
        """The model in infix form with GP-GOMEA's names (for the reports)."""
        stack: list[str] = []
        for t in program:
            if t >= 0:
                stack.append(f"x{t + 1}")
                continue
            b = stack.pop()
            a = stack.pop()
            stack.append(f"aq({a},{b})" if t == AQ else f"({a}{_OP_NAME[t]}{b})")
        return stack[0]

    # ---- evaluation

    def predict(self, program: Sequence[int], cols: Sequence[Sequence[float]]) -> list[float]:
        """The output of ``program`` on every row; ``cols[j]`` is input j. ``nan`` marks an
        interpreter overflow (see the module docstring)."""
        stack: list = []
        for t in program:
            if t >= 0:
                stack.append(cols[t])
                continue
            b = stack.pop()
            a = stack.pop()
            if t == ADD:
                stack.append([x + y for x, y in zip(a, b, strict=True)])
            elif t == SUB:
                stack.append([x - y for x, y in zip(a, b, strict=True)])
            elif t == MUL:
                stack.append([x * y if x else x for x, y in zip(a, b, strict=True)])
            else:
                stack.append(_aq_column(a, b))
        out = stack[0]
        return list(out) if program and program[-1] >= 0 else out
