"""C++: moses/comboreduct/combo/argument.h

`argument` -- the index of an input variable ($1, $2, ...) of a function
coded as a combo tree. A negative index is the negated literal: idx == -3
is NOT($3), the third argument's negation. idx == 0 is invalid (1-based
indices; asserted impossible in the constructor).

Two details here are load-bearing for reduct, which depends on both (see
task-3-brief.md):

1. Sign encodes negation -- negate()/is_negated() below.
2. `operator<` (ported as `__lt__`) orders by *absolute* value via
   opencog::absolute_value_order<int> (opencog/util/numeric.h:56-60), not by
   signed value. $1 and !$1 sort adjacent; a signed-int comparison would
   instead put every negated argument before every positive one, silently
   producing different clause orderings in flat_normal_form and therefore
   different reductions.
"""

import functools

from pymoses.comboreduct.combo.common_def import arity_t


@functools.total_ordering
class argument:
    """C++: moses/comboreduct/combo/argument.h:43 argument

    `boost::less_than_comparable<argument>` (argument.h:44) and
    `boost::equality_comparable<argument>` (argument.h:45) are CRTP mixins
    that generate >, <=, >= from operator< and != from operator==.
    `functools.total_ordering` is the Python equivalent: applied to a class
    defining __eq__ and __lt__, it generates the same four derived
    comparisons.

    Scoped to argument.h:43-82 (the class through abs_idx_from_zero): the
    remaining member, `is_idx_valid` (argument.h:88-90), is not ported here
    -- it has exactly one call site in all of classic MOSES/
    (procedure_call.h:184, itself part of the legacy procedure-call
    machinery), which is out of scope for this task.
    """

    __slots__ = ("idx",)

    def __init__(self, i: arity_t) -> None:
        """C++: moses/comboreduct/combo/argument.h:50 argument

        FIX WAVE (final review, Important): `OC_ASSERT(idx != 0, ...)`
        (argument.h:51) is active in release C++ builds. A bare Python
        `assert` is not -- `python -O` strips it, which would silently
        accept `i == 0` and construct an `argument` whose `idx` classic
        would never allow to exist (verified: under `-O`, `argument(0)`
        was accepted and evaluated to `boolean_inputs[-1]`, the LAST
        input, not a rejected construction). Raised explicitly instead so
        this check survives `-O` the same way `OC_ASSERT` survives
        `NDEBUG`.
        """
        if i == 0:
            raise AssertionError("idx should be different than zero.")
        self.idx = i

    def negate(self) -> None:
        """C++: moses/comboreduct/combo/argument.h:54 negate"""
        self.idx = -self.idx

    def is_negated(self) -> bool:
        """C++: moses/comboreduct/combo/argument.h:57 is_negated"""
        return self.idx < 0

    def __lt__(self, other: argument) -> bool:
        """C++: moses/comboreduct/combo/argument.h:61 argument

        The checker matches plain identifiers only, so this cites the
        enclosing class at the line where the member actually lives (see
        docs/superpowers/specs/2026-09-02-pymoses-foundation-design.md
        section 5.2) -- the real member is `argument::operator<`
        (argument.h:61-64).

        Ports `opencog::absolute_value_order<int>::operator()` verbatim
        (opencog/util/numeric.h:56-60: `(a == -b) ? a < b : abs(a) <
        abs(b)`): order by absolute value; when two indices are exact
        negations of each other (a == -b, e.g. $1 vs !$1) the negative one
        sorts first. Do not replace this with a sort-key tuple without
        re-checking it reproduces the same a == -b tie-break -- see the
        module docstring and task-3-brief.md for why this matters.
        """
        a, b = self.idx, other.idx
        if a == -b:
            return a < b
        return abs(a) < abs(b)

    def __copy__(self) -> argument:
        """No C++ counterpart: what `copy.copy` builds through `__reduce_ex__`
        for this slotted class (a new object with the same `idx`), without
        the generic reconstruction machinery. Tree copies call this for
        every argument leaf.
        """
        new = object.__new__(type(self))
        new.idx = self.idx
        return new

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/combo/argument.h:66 argument

        Cites the enclosing class for the same reason as __lt__ above; the
        real member is `argument::operator==` (argument.h:66-68).
        """
        if not isinstance(other, argument):
            return NotImplemented
        return self.idx == other.idx

    def __hash__(self) -> int:
        """C++: moses/comboreduct/combo/argument.h:43 argument

        No C++ counterpart: classic argument is only ever ordered into a
        std::set (via operator<, ported above as __lt__), never hashed into
        an unordered container. Defining __eq__ without __hash__ makes a
        class unhashable in Python, and elsewhere in this port `vertex`
        values (of which argument is one alternative) are used as plain,
        hashable values -- so this is required for parity with how the
        rest of the port treats a vertex, not a ported symbol as such.
        Hashes on idx, keeping `a == b => hash(a) == hash(b)`.
        """
        return hash(self.idx)

    def abs_idx(self) -> arity_t:
        """C++: moses/comboreduct/combo/argument.h:70 abs_idx"""
        return -self.idx if self.idx < 0 else self.idx

    @staticmethod
    def idx_to_abs_idx_from_zero(other_idx: arity_t) -> arity_t:
        """C++: moses/comboreduct/combo/argument.h:73 idx_to_abs_idx_from_zero"""
        return (-other_idx if other_idx < 0 else other_idx) - 1

    @staticmethod
    def idx_from_zero_to_idx(idx_from_zero: arity_t) -> arity_t:
        """C++: moses/comboreduct/combo/argument.h:76 idx_from_zero_to_idx"""
        return idx_from_zero + 1

    def abs_idx_from_zero(self) -> arity_t:
        """C++: moses/comboreduct/combo/argument.h:80 abs_idx_from_zero

        Returns 0 for argument of idx 1 or -1, 1 for idx 2 or -2, and so on.
        """
        return argument.idx_to_abs_idx_from_zero(self.idx)

    def __repr__(self) -> str:
        return f"argument({self.idx})"
