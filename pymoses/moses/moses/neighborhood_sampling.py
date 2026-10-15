"""Enumerate and sample the knob-setting neighborhood of an instance.

Ports `moses/moses/moses/neighborhood_sampling.{h,cc}`, Boolean/discrete
paths only. This is the first ported call site that actually draws from
`pymoses.util.rng.randGen()` -- S0-S8 proved the RNG bit-identical but never
exercised it. `generate_initial_sample` and `sample_from_neighborhood` call
that process-wide singleton directly, exactly as their C++ signatures do
(neither takes an RNG parameter); a caller wanting a reproducible sequence
reseeds it first with ``randGen().seed(s)``, matching how the C++ oracle
probe (`tools/cpp_probe/neighborhood_probe.cc`) does it.

S14 Task 3 ports the three continuous SAMPLING helpers -- `flip_LR`,
`twiddle_contin_bit`, `generate_contin_neighbor` (neighborhood_sampling.cc:
32, 38, 55) -- and wires the last of them into `sample_from_neighborhood`'s
own "modify contin" branch (neighborhood_sampling.h:172-178), which
previously raised ``NotImplementedError`` there. This is NOT full
continuous-field support for the rest of this module: `count_neighborhood_
size_from_index`, `vary_n_knobs` (hence `generate_all_in_neighborhood`) and
`_randomize`'s `occam_randomize_contin` call are a DIFFERENT feature
(exhaustive contin-neighborhood enumeration / uniform contin
randomization, not among this task's three named functions) and still
raise ``NotImplementedError`` on a continuous field_set exactly as before.
Term fields need no equivalent guard anywhere in this module: they do not
exist in this port (see field_set.py's own term exclusion), so the C++
"terms" branch -- always an empty index range here -- is omitted outright
rather than guarded.

`flip_LR`/`twiddle_contin_bit` take ``(fs, inst, idx)``/``(fs, inst, idx,
next_idx)`` in place of C++'s `field_set::disc_iterator` parameter(s) --
the same raw-index substitution `vary_n_knobs`/`sample_from_neighborhood`
already use for every OTHER disc_iterator/bit_iterator dereference in this
file (via `fs.get_raw`/`fs.set_raw`), applied here for the first time to a
FUNCTION PARAMETER rather than a local loop variable. `generate_contin_
neighbor` takes ``(fs, inst, idx, dist)`` where ``idx`` is C++'s
`it.idx()` -- the CONTIN SPEC index (`fs.contin()`'s own indexing, as
`fs.contin_to_raw_idx`/`fs.contin_length` already take) -- NOT a raw
index; `flip_LR`/`twiddle_contin_bit`'s own ``idx``/``next_idx``, by
contrast, ARE raw indices (`fs.get_raw`/`fs.set_raw`'s own indexing).
This mirrors the real type distinction in C++ between
`field_set::contin_iterator::idx()` (spec-indexed) and
`field_set::disc_iterator` (raw-indexed).

`twiddle_contin_bit`/`generate_contin_neighbor` both take an optional
trailing ``rng`` parameter defaulting to ``None`` (resolved to
``randGen()`` when omitted), reproducing C++'s own
``RandGen& rng = randGen()`` default argument (neighborhood_sampling.h:
87, 116) -- UNLIKE `generate_initial_sample`/`sample_from_neighborhood`
above, which have NO rng parameter in C++ at all and so always call
``randGen()`` directly (see this docstring's own paragraph on that). Since
``randGen()`` is a process-wide singleton, resolving the default at each
call site (Python) is behaviourally identical to C++ re-resolving its own
default argument at each call site: both always land on the SAME live
object.

`sample_new_instances` (both overloads, neighborhood_sampling.cc:214, 261)
IS ported (S9 Task 2), now that `instance_set` (S9 Task 0) exists --
`hill_climbing.py:hill_climbing.__call__` (neighborhood_sampling.cc's real
caller, hill-climbing.cc:188) is the reason it is required rather than
merely convenient. It writes through `_DemeWriter`, a third adapter (after
the plain-list `out`/`end` convention below) standing in for C++'s
`deme.begin() + current_number_of_instances` output iterator: it satisfies
the same `.append(value)`/`len(...)` protocol `_emit` already expects, but
each `append` writes `scored_instance(value)` into the correspondingly-
resized `instance_set` slot instead of growing a Python list -- mirroring
`*out++ = new_inst` implicitly converting through `scored_instance(const
instance&)` (scored_instance.h:39, not marked `explicit`) the way C++'s
own assignment does. `instance_set` is presumed already resized to fit
every write before `_DemeWriter` is used, exactly as `deme.resize(...)`
guarantees in both `sample_new_instances` overloads below.

Output-iterator convention: C++ threads a same-container ``Out out, Out
end`` pair (a current position and a fixed one-past-the-end sentinel
bounding a pre-sized buffer) through every generator here, re-assigning
``out = f(..., out, end)`` on each recursive call and asserting
``OC_ASSERT(out != end)`` immediately before every ``*out++ = value``. This
port replaces that pair with a plain mutable ``list[instance]`` (``out``,
appended to in place -- every real caller starts it empty) plus an ``int``
capacity (``end``): ``len(out)`` plays the role of the C++ iterator's
current position, so nothing needs to be returned and re-assigned across
recursive calls. `_emit` is the shared helper that performs the write and
the bounds check together, exactly where C++ performs them together.
"""

from __future__ import annotations

import math

from pymoses.moses.representation.field_set import contin_spec, field_set
from pymoses.moses.representation.instance import instance
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.scored_instance import scored_instance
from pymoses.util.lazy_random_selector import lazy_random_selector
from pymoses.util.logger import logger
from pymoses.util.rng import MT19937RandGen, randGen

# DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h:491,514,520):
# std::numeric_limits<size_t>::max() (three default-argument occurrences;
# h:27 is only the `#include <limits>` that makes the expression
# available, not the expression itself) pinned to the 64-bit ABI
# instance.py already measures and pins (bits_per_packed_t = 64).
_SIZE_T_MAX = (1 << 64) - 1

# C++ `unsigned` is 32 bits; `cni` (neighborhood_sampling.cc:133) and
# `remRLs` (neighborhood_sampling.h:337) are both `unsigned` and both can
# wrap, so the port masks rather than letting Python's arbitrary-precision
# ints run away. See each use site for the reachable input that wraps it.
_UINT32_MAX = (1 << 32) - 1

# `unchecked_factorial<double>` covers 0..170 in boost; past that
# `binomial_coefficient` switches to its beta-function branch. Boost's
# table entries are the correctly-rounded doubles of i!, so
# `float(math.factorial(i))` reproduces them -- verified as part of the
# 115/115 agreement recorded on `_boost_binomial_coefficient_double`.
_BOOST_MAX_FACTORIAL_DOUBLE = 170
_FACTORIAL_DOUBLE = tuple(float(math.factorial(i)) for i in range(_BOOST_MAX_FACTORIAL_DOUBLE + 1))

# `cvttsd2si`'s "integer indefinite" result, 0x8000000000000000, is what a
# double outside int64's range converts to on x86-64; the low 32 bits of
# that are zero. See `_double_to_unsigned`.
_INT64_LIMIT = 9223372036854775808.0


def _pow2(x: int) -> int:
    """C++: opencog/util/numeric.h:266 pow2

    `pow2<unsigned int>` -- `OC_ASSERT(8*sizeof(OutInt) - 0 > x)` then
    `static_cast<unsigned>(1) << x`. The assert is NOT decoration: classic
    aborts for `x >= 32`, and `count_neighborhood_size_from_index` reaches
    it with a contin field whose `depth - length >= 32` (a depth-35 field
    left all-Stop, at `dist >= 32`, is the smallest case). Reproduced as a
    raise so the port fails where classic fails instead of returning a
    number classic never produces.
    """
    # DIVERGENCE(C++ opencog/util/numeric.h:268): OC_ASSERT aborts the
    # process; this raises, matching field_set.py's convention for
    # OC_ASSERT on out-of-range shifts.
    if not 32 > x:
        raise ValueError("pow2: Amount to shift is out of range.")
    return (1 << x) & _UINT32_MAX


def _boost_binomial_coefficient_double(n: int, k: int) -> float:
    """C++: boost/math/special_functions/binomial.hpp:23 binomial_coefficient

    `boost::math::binomial_coefficient<double>(n, k)`, transliterated from
    the installed boost header rather than approximated, because
    `count_neighborhood_size_from_index` casts its result to `unsigned`
    (neighborhood_sampling.cc:141,149) and a one-ULP difference survives
    that cast as a difference of several counted instances.

    The two details that a "compute the exact binomial" implementation
    gets wrong, both measured: boost divides in DOUBLE
    (`fact[n] /= fact[n-k] /= fact[k]`), which is not correctly rounded
    for larger `n`; and it finishes with `ceil(result - 0.5f)`, a
    round-to-nearest step. Emulating only the first, or neither, disagrees
    with boost on 10-11 of a 121-row corpus. With both, agreement is exact:

        115/115 rows byte-identical as doubles AND after the `(unsigned)`
        cast, measured 2026-09-09 against
        `tools/cpp_probe/contin_enumeration_probe.cc`'s "binomial_double"
        section, which dumps boost's own return value as a hex float.

    The remaining 6 rows of that corpus are the `n > 170` beta-function
    branch; see the raise below.
    """
    if k > n:
        # boost raises a domain_error; `safe_binomial_coefficient`'s own
        # docstring records that classic never reaches it with k > n.
        raise ValueError(f"binomial_coefficient: undefined for k > n (n={n}, k={k})")
    if k == 0 or k == n:
        return 1.0
    if k == 1 or k == n - 1:
        return float(n)
    if n > _BOOST_MAX_FACTORIAL_DOUBLE:
        # DIVERGENCE(C++ boost/math/special_functions/binomial.hpp:44-53):
        # boost falls back to `beta()` here and returns a (garbage but
        # defined) number. Reproducing boost's `beta` bit-for-bit is a
        # substantial numeric port of its own, and this branch needs a
        # contin field of depth > 170 -- i.e. `--contin-depth 171` or
        # higher, a search space of 2**171 settings -- to be reached at
        # all. Gated rather than approximated: a silently-close number
        # here would be indistinguishable from a correct one.
        raise NotImplementedError(
            "binomial_coefficient: boost's beta-function branch (n > 170, "
            f"got n={n}) is not ported; reachable only at --contin-depth "
            "above 170 -- boost/math/special_functions/binomial.hpp:44-53"
        )
    result = _FACTORIAL_DOUBLE[n]
    result /= _FACTORIAL_DOUBLE[n - k]
    result /= _FACTORIAL_DOUBLE[k]
    # `ceil(result - 0.5f)`: the float literal 0.5 is exact in double, so
    # the promotion changes nothing.
    return float(math.ceil(result - 0.5))


def _double_to_unsigned(value: float) -> int:
    """The C-style `(unsigned)` cast applied to
    `boost::math::binomial_coefficient<double>` at
    neighborhood_sampling.cc:141 and :149.

    Formally undefined behaviour when the double does not fit; in practice
    GCC on x86-64 emits a 64-bit `cvttsd2si` and keeps the low 32 bits, so
    an out-of-range double becomes 0x8000000000000000 -> 0. Measured: the
    probe's C(70,35) and C(80,40) rows both report 0, which a plain
    truncate-and-mask would have turned into 796475392 and 2499805184.
    """
    if math.isnan(value) or value >= _INT64_LIMIT or value < -_INT64_LIMIT:
        return 0
    return int(value) & _UINT32_MAX


def _emit(out: list[instance], end: int, value: instance) -> None:
    """Append ``value`` to ``out``, mirroring the ``OC_ASSERT(out != end);
    *out++ = value;`` pair that precedes every write in
    neighborhood_sampling.{h,cc}. See the module docstring for how
    ``out``/``end`` replace C++'s output-iterator pair.
    """
    # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h): every
    # `OC_ASSERT(out != end)` in this file becomes IndexError, matching
    # field_set.py's convention for out-of-range/overflowing raw access.
    if len(out) >= end:
        raise IndexError("write past end of neighborhood output buffer")
    out.append(value)


def safe_binomial_coefficient(k: int, n: int) -> int:
    """C++: moses/moses/moses/neighborhood_sampling.cc:87 safe_binomial_coefficient

    Despite the parameter names, this computes "k choose n": C++ forwards to
    ``boost::math::binomial_coefficient<double>(k, n)``, i.e. boost's own
    ``binomial_coefficient(n_arg=k, k_arg=n)``, which requires ``k >= n``.
    """
    # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.cc:87-96): C++'s
    # try/catch there only catches boost::numeric::positive_overflow, not the
    # std::domain_error boost::math::binomial_coefficient<double> throws for
    # k < n -- measured directly: calling safe_binomial_coefficient(3, 4)
    # through a C++ probe aborts the process with an uncaught
    # boost::wrapexcept<std::domain_error> ("k > n"), not a clamped result.
    # This is not exercised by classic's own working code: there are TWO
    # call sites in the whole classic tree (`git grep
    # safe_binomial_coefficient`), NOT one -- this file's own
    # `count_neighborhood_size_from_index` bit branch (below, guarding
    # `dist <= rb` before calling) AND hill-climbing.cc:516, inside
    # `estimate_neighborhood`, ported at `hill_climbing.py`'s own
    # `estimate_neighborhood` (in the same S9 range as this file). That
    # second call site guards the k>=n precondition differently -- by
    # clamping `distance` down to `nn_estimate` a few lines earlier
    # (`hill-climbing.cc:493-502`) rather than skipping the call outright
    # -- but the effect is the same: `nn_estimate >= distance` always
    # holds by the time `safe_binomial_coefficient(nn_estimate, distance)`
    # runs. Both real call sites respect the k >= n precondition, so k < n
    # never actually happens in classic's own working code at either one.
    # Python's math.comb(k, n) returns the mathematically-correct 0 for
    # k < n instead of raising; this port relies on callers respecting the
    # same k >= n precondition classic's real call sites do, and does not
    # attempt to reproduce a crash for an input the real code never sends.
    return min(math.comb(k, n), _SIZE_T_MAX)


def count_neighborhood_size_from_index(
    fs: field_set,
    inst: instance,
    dist: int,
    starting_index: int,
    max_count: int = _SIZE_T_MAX,
) -> int:
    """C++: moses/moses/moses/neighborhood_sampling.cc:99 count_neighborhood_size_from_index

    See the module docstring for the continuous-field guard and the term
    omission.
    """
    if dist == 0:
        return 1

    number_of_instances = 0

    if fs.begin_contin_raw_idx() <= starting_index < fs.end_contin_raw_idx():
        # C++: neighborhood_sampling.cc:117-160. `itc += contin_idx` then
        # `fs.contin()[itc.idx()]`, and `contin_iterator::idx()` after
        # advancing from `begin_contin` by `contin_idx` IS `contin_idx`, so
        # the iterator round-trip collapses to a direct index here.
        contin_idx = fs.raw_to_contin_idx(starting_index)
        # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.cc:122):
        # OC_ASSERT becomes ValueError. This one is load-bearing rather
        # than decorative -- it is what makes `starting_index + depth`
        # below a correct "skip to the next contin field", and it holds
        # only because every caller enters a contin field at its first raw
        # index. `vary_n_knobs` (which takes an arbitrary starting_index)
        # has no such assert and must not be assumed to share it.
        if starting_index - fs.contin_to_raw_idx(contin_idx) != 0:
            raise ValueError(
                "count_neighborhood_size_from_index: starting_index must be the "
                f"first raw index of contin field {contin_idx}"
            )

        # `int depth` / `int length` (cc:125-126), deliberately SIGNED:
        # `i - length` and `depth - length` below are allowed to go
        # negative, and `std::max(0, ...)`/`std::min(...)` are what bring
        # them back. Python ints are signed, so this needs no mask -- but
        # it does need the `max`/`min` to be applied in the same places.
        depth = fs.contin()[contin_idx].depth
        length = fs.contin_length(inst, contin_idx)

        # Calculate number_of_instances for each possible distance i of the
        # current contin.
        for i in range(0, min(dist, depth) + 1):
            # Number of instances for this contin, at distance i. `unsigned`
            # (cc:133), and it genuinely wraps: a depth-35 field at length
            # 35 makes `(unsigned)binomial_coefficient<double>(35, 17)`
            # alone 242600354 rather than the exact 4537567650.
            cni = 0

            # Count combinations when Left or Right are switched and added
            # after Stop, where j represents the number of Left or Right
            # added after Stop.
            for j in range(max(0, i - length), min(i, depth - length) + 1):
                cni = (
                    cni
                    + _double_to_unsigned(_boost_binomial_coefficient_double(length, i - j))
                    * _pow2(j)
                ) & _UINT32_MAX

            # Count combinations when Left or Right are switched and
            # removed before Stop, where j represents the number of removed
            # Left or Right before Stop.
            if i <= length:
                for j in range(1, min(i, length) + 1):
                    cni = (
                        cni
                        + _double_to_unsigned(_boost_binomial_coefficient_double(length - j, i - j))
                    ) & _UINT32_MAX

            # Recursive call.
            number_of_instances = (
                number_of_instances
                + cni
                * count_neighborhood_size_from_index(
                    fs, inst, dist - i, starting_index + depth, max_count
                )
            ) & _SIZE_T_MAX
            # Stop prematurely if above max_count.
            if number_of_instances > max_count:
                return number_of_instances

    elif fs.begin_disc_raw_idx() <= starting_index < fs.end_disc_raw_idx():
        # Recursive call, moved for one position.
        number_of_instances = count_neighborhood_size_from_index(
            fs, inst, dist, starting_index + 1, max_count
        )

        # stop prematurely if above max_count
        if number_of_instances > max_count:
            return number_of_instances

        # count all legal values of the knob
        multy = fs.disc_and_bit()[fs.raw_to_disc_idx(starting_index)].multy
        number_of_instances += (multy - 1) * count_neighborhood_size_from_index(
            fs, inst, dist - 1, starting_index + 1, max_count
        )

    elif fs.begin_bit_raw_idx() <= starting_index < fs.end_bit_raw_idx():
        # Since bits all have the same multiplicity (viz. 2), and are
        # the last in the field set, there is no need for recursive call.
        rb = fs.end_bit_raw_idx() - starting_index
        # Ported verbatim (neighborhood_sampling.cc:99-...'s own `if (dist
        # <= rb)`) though behaviourally REDUNDANT in Python: `math.comb`
        # already returns 0 for `dist > rb` (see `safe_binomial_
        # coefficient`'s own docstring), the same value `number_of_
        # instances` already holds when this guard is false and the
        # assignment is skipped. No test can distinguish "guard present"
        # from "guard removed" here -- this is a fidelity choice (matching
        # C++'s own structure, in case a future change to either side
        # makes the two branches diverge), not a covered behaviour; say so
        # rather than imply a test exists that would catch its removal.
        if dist <= rb:
            number_of_instances = safe_binomial_coefficient(rb, dist)
    # else: harmless; this recursive algo is designed to over-run by exactly one.

    return number_of_instances


def count_neighborhood_size(
    fs: field_set,
    dist: int,
    max_count: int = _SIZE_T_MAX,
    inst: instance | None = None,
) -> int:
    """C++: moses/moses/moses/neighborhood_sampling.cc:201 count_neighborhood_size
    (``inst`` given) and neighborhood_sampling.cc:207 count_neighborhood_size
    (``inst`` omitted -- "for backward compatibility, like above but with
    null instance"). One Python function merges both overloads, matching
    field_set.py's merged-constructor convention; ``inst`` trails as an
    optional keyword-shaped parameter rather than sitting at C++'s overload-1
    position 2, since ``dist`` is what every real caller supplies either way.
    """
    if inst is None:
        inst = instance(fs.packed_width())
    return count_neighborhood_size_from_index(fs, inst, dist, 0, max_count)


def vary_n_knobs(
    fs: field_set,
    inst: instance,
    dist: int,
    starting_index: int,
    out: list[instance],
    end: int,
) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.h:265 vary_n_knobs

    See the module docstring for the output-iterator convention, the
    continuous-field guard, and the term omission.
    """
    # Record the current instance.
    if dist == 0:
        _emit(out, end, instance(inst))
        return

    tmp_inst = instance(inst)  # `instance tmp_inst = inst;` -- a value copy.

    if fs.begin_contin_raw_idx() <= starting_index < fs.end_contin_raw_idx():
        # C++: neighborhood_sampling.h:289-349. Modify the contin knob at
        # `starting_index`, then recurse on starting_index and dist.
        contin_idx = fs.raw_to_contin_idx(starting_index)

        # The 'depth' is the max possible size of this contin field, while
        # length is the actual size, for this instance.
        depth = fs.contin()[contin_idx].depth
        length = fs.contin_length(tmp_inst, contin_idx)
        relative_raw_idx = starting_index - fs.contin_to_raw_idx(contin_idx)

        if fs.get_raw(tmp_inst, starting_index) == contin_spec.Stop:
            # Assume that this is the first stop encountered for this
            # contin field. Skip straight to the next contin field.
            next_contin = starting_index + depth - relative_raw_idx
            vary_n_knobs(fs, tmp_inst, dist, next_contin, out, end)

            # Turn this stop pseudo-bit to Left, and Right, and recurse to
            # next.
            fs.set_raw(tmp_inst, starting_index, contin_spec.Left)
            vary_n_knobs(fs, tmp_inst, dist - 1, starting_index + 1, out, end)
            fs.set_raw(tmp_inst, starting_index, contin_spec.Right)
            vary_n_knobs(fs, tmp_inst, dist - 1, starting_index + 1, out, end)
        else:
            # Recursive call, moved for one position.
            vary_n_knobs(fs, tmp_inst, dist, starting_index + 1, out, end)
            # Left<->Right
            fs.set_raw(
                tmp_inst,
                starting_index,
                contin_spec.switchLR(fs.get_raw(tmp_inst, starting_index)),
            )
            vary_n_knobs(fs, tmp_inst, dist - 1, starting_index + 1, out, end)

            # Set all remaining 'pseudo-bits' in this contin field to Stop;
            # remRLs is the number of remaining non-Stop pseudo-bits,
            # including this one.
            #
            # `unsigned remRLs = length - relative_raw_idx` (h:337) with
            # both operands `size_t`: when `relative_raw_idx > length` the
            # size_t subtraction wraps near 2**64 and the assignment then
            # truncates to 32 bits, so `remRLs` becomes ~0xFFFFFFFF and the
            # `remRLs <= dist` test below is false. That is NOT unreachable
            # -- `vary_n_knobs` takes an arbitrary `starting_index` and an
            # arbitrary center instance through its own public signature,
            # so a contin field holding a non-Stop pseudo-bit AFTER a Stop
            # (e.g. [Left, Stop, Right]) entered past that Stop produces
            # it. Reproduced with an explicit mask rather than clamped.
            remRLs = (length - relative_raw_idx) & _UINT32_MAX
            if remRLs <= dist:
                # `for(; relative_raw_idx < length; --length, ++itr)` --
                # note it decrements `length`, NOT `relative_raw_idx`.
                # Transliterated as-is: the loop therefore runs exactly
                # `length - relative_raw_idx` times, clearing this field's
                # remaining non-Stop pseudo-bits from `starting_index`
                # forward. `relative_raw_idx` keeps its original value
                # afterwards, which the `next_contin` line below relies on.
                itr = starting_index
                remaining_length = length
                while relative_raw_idx < remaining_length:
                    fs.set_raw(tmp_inst, itr, contin_spec.Stop)
                    remaining_length -= 1
                    itr += 1
                # Skip straight to the next contin field.
                next_contin = starting_index + depth - relative_raw_idx
                vary_n_knobs(fs, tmp_inst, dist - remRLs, next_contin, out, end)

    elif fs.begin_disc_raw_idx() <= starting_index < fs.end_disc_raw_idx():
        # For distance == 1, take the tail call and explicitly turn it into a
        # loop (avoids deep stacks). This loop starts at the far end and
        # works backwards through the recursion, unlike the loop below.
        if dist == 1:
            end_idx = fs.end_disc_raw_idx()
            vary_n_knobs(fs, tmp_inst, dist, end_idx, out, end)

            idx = starting_index
            while idx < end_idx:
                multy = fs.disc_and_bit()[fs.raw_to_disc_idx(idx)].multy
                tmp_val = fs.get_raw(tmp_inst, idx)
                for i in range(1, multy):
                    # Vary to all legal values. The neighborhood should not
                    # equal itself, so if it is the same, set it to 0.
                    fs.set_raw(tmp_inst, idx, 0 if tmp_val == i else i)
                    _emit(out, end, instance(tmp_inst))
                fs.set_raw(tmp_inst, idx, tmp_val)  # put the bit back.
                idx += 1
            return

        # Recursive call, moved for one position. Note this steps past all
        # the disc knobs, and goes to do the single-bit knobs. Only after
        # returning from that does it do the disc knobs, below.
        vary_n_knobs(fs, tmp_inst, dist, starting_index + 1, out, end)

        # Modify the disc and recursive call, moved for one position.
        multy = fs.disc_and_bit()[fs.raw_to_disc_idx(starting_index)].multy
        tmp_val = fs.get_raw(tmp_inst, starting_index)
        for i in range(1, multy):
            fs.set_raw(tmp_inst, starting_index, 0 if tmp_val == i else i)
            vary_n_knobs(fs, tmp_inst, dist - 1, starting_index + 1, out, end)
        # NOTE: C++ does not restore tmp_val here (neighborhood_sampling.h:
        # 395-407) -- only the dist == 1 unrolled loop above does, because it
        # alone reuses tmp_inst across more than one disc field in the same
        # loop. tmp_inst is discarded when this call returns, so leaving it
        # mutated is harmless; restoring it here would be an unfaithful "fix".

    elif fs.begin_bit_raw_idx() <= starting_index < fs.end_bit_raw_idx():
        if dist == 1:
            end_idx = fs.end_bit_raw_idx()
            idx = starting_index
            while idx < end_idx:
                cur = fs.get_raw(tmp_inst, idx)
                fs.set_raw(tmp_inst, idx, 0 if cur else 1)  # change one bit.
                _emit(out, end, instance(tmp_inst))  # record the resulting inst.
                fs.set_raw(tmp_inst, idx, cur)  # put the bit back.
                idx += 1
            return

        # Recursive call, moved for one position.
        vary_n_knobs(fs, tmp_inst, dist, starting_index + 1, out, end)

        # Modify tmp_inst at starting_index, change to the opposite value.
        cur = fs.get_raw(tmp_inst, starting_index)
        fs.set_raw(tmp_inst, starting_index, 0 if cur else 1)

        # Recursive call, moved for one position.
        vary_n_knobs(fs, tmp_inst, dist - 1, starting_index + 1, out, end)
        # (Same no-restore note as the disc branch above applies here too.)

    # else: the current recursive algo used here will over-run the end of
    # the array by one. This is normal. Do nothing. If 'dist' was zero, the
    # very first statement at the top takes care of things for us.


def generate_all_in_neighborhood(
    fs: field_set,
    dist: int,
    out: list[instance],
    end: int,
    center_inst: instance | None = None,
) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.h:203 generate_all_in_neighborhood
    (``center_inst`` given) and neighborhood_sampling.h:227
    generate_all_in_neighborhood (``center_inst`` omitted -- classic builds a
    fresh null/zero instance for it). One Python function merges both
    overloads, matching count_neighborhood_size's merged-overload convention
    above.
    """
    if center_inst is None:
        center_inst = instance(fs.packed_width())
    # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h:207-208):
    # OC_ASSERT becomes ValueError.
    if len(center_inst) != fs.packed_width():
        raise ValueError("the size of center_instance should be equal to the width of fs")
    vary_n_knobs(fs, center_inst, dist, 0, out, end)


# `_uniform_randomize_disc` and `_uniform_randomize_bit` used to live
# here, as byte-identical copies of `eda/initialization.py`'s
# `uniform_randomize_disc`/`uniform_randomize_bit`. They existed because
# `eda/initialization.{h,cc}` was unported when `generate_initial_sample`
# first needed them. It is ported now, `_randomize` below delegates to it,
# and nothing outside this module referenced either copy -- so they are
# gone rather than left as a second definition that could drift.


def _randomize(fs: field_set, inst: instance) -> None:
    """C++: moses/moses/eda/initialization.cc:96 randomize

    Delegates to `pymoses.moses.eda.initialization.randomize`, which is the
    real port of this function, so the two cannot drift. This used to be a
    partial copy that raised on any field_set with contin fields, because
    `occam_randomize_contin` had not been ported; it has been since, and
    the copy's disc-then-bit tail was the same code twice.

    The import is function-local on purpose. `eda/initialization.py`
    imports `generate_contin_neighbor` from THIS module at module scope,
    mirroring C++'s own dependency direction (`initialization.cc:25`
    includes `neighborhood_sampling.h`, not the reverse), so a module-scope
    import here would close the cycle. Deferring it to call time is the
    least-bad option available and costs one dict lookup per call.

    `occam_randomize_term`'s call is omitted there rather than stubbed:
    term fields do not exist anywhere in this port, so that call's
    iteration range is always empty and it draws nothing. Draw ORDER
    across the remaining three calls (contin, disc, bit) is the property
    that matters, since every one of them draws from the shared RNG.
    """
    from pymoses.moses.eda.initialization import randomize

    randomize(fs, inst)


def generate_initial_sample(fs: field_set, n: int, out: list[instance], end: int) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.h:52 generate_initial_sample

    Draws `randGen().randbool()` in both bias loops below, plus whatever
    `_randomize` draws first. See the module docstring for the output-
    iterator convention and for why `randGen()` is called directly rather
    than accepting an RNG parameter (matching the real C++ signature, which
    has none either).
    """
    rng = randGen()
    for _ in range(n):
        inst = instance(fs.packed_width())

        _randomize(fs, inst)

        # Bias towards the exemplar instance.
        for idx in range(fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()):
            if rng.randbool():
                fs.set_raw(inst, idx, 0)
        for idx in range(fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()):
            if rng.randbool():
                fs.set_raw(inst, idx, 0)

        _emit(out, end, inst)


def flip_LR(fs: field_set, inst: instance, idx: int) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.cc:32 flip_LR

    Flips a single contin "pseudo-bit" between `contin_spec.Left` and
    `contin_spec.Right`. ``idx`` is a RAW field index (`fs.get_raw`/
    `fs.set_raw`'s own indexing) -- see the module docstring's own
    paragraph on this substitution for C++'s `field_set::disc_iterator`
    parameter.
    """
    is_left = fs.get_raw(inst, idx) == contin_spec.Left
    fs.set_raw(inst, idx, contin_spec.Right if is_left else contin_spec.Left)


def twiddle_contin_bit(
    fs: field_set,
    inst: instance,
    idx: int,
    next_idx: int,
    rng: MT19937RandGen | None = None,
) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.cc:38 twiddle_contin_bit

    "Twiddle one contin bit only, according to the following rules: 1. If
    it is Stop then select Left or Right. 2. Otherwise if the next one is
    Stop (or invalid) then select Stop or flip. 3. Otherwise just flip."
    (neighborhood_sampling.h:79-85's own doc comment, reproduced
    verbatim.) ``idx``/``next_idx`` are RAW field indices, like `flip_LR`
    above; ``next_idx`` is "invalid" exactly when it is not literally
    ``idx + 1`` -- the caller (`generate_contin_neighbor` below) passes
    `fs.raw_size()` there for that case, mirroring C++'s own
    `fs.end_raw(inst)` sentinel.

    Draws exactly one `rng.randbool()` in the `Stop` branch and in the
    "next is Stop or invalid" branch; draws NOTHING in the plain-flip
    branch (`flip_LR` itself draws nothing). Draw order/count here is
    part of the bit-identity claim this task exists to establish -- see
    the module docstring's RNG paragraph on why an omitted ``rng`` still
    resolves to the live process-wide singleton.

    The `or` in C++'s `itr + 1 != next_itr // next_itr is invalid or
    *next_itr == field_set::contin_spec::Stop` short-circuits exactly like
    Python's own `or` below: when ``idx + 1 != next_idx`` is already
    True, `fs.get_raw(inst, next_idx)` is never evaluated -- load-bearing
    when ``next_idx`` is the `fs.raw_size()` sentinel, which
    `fs.get_raw` would otherwise reject as out of range.

    DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.cc:46-47):
    when ``idx + 1`` genuinely EQUALS `fs.raw_size()` (this contin field
    is the very last thing in the whole raw layout -- no disc/bit fields
    follow it, e.g. a contin-only regression deme -- and ``idx`` is its
    final pseudo-bit), C++'s short-circuit does not fire and it
    dereferences `fs.end_raw(inst)`: `field_set::get_raw` reads
    `_fields[_fields.size()]` (field_set.h:414, unchecked
    `std::vector::operator[]`) and then `inst[<garbage major_offset>]`
    -- undefined behaviour whose value (and hence whether one
    `randbool()` is drawn) depends on heap contents, so it cannot be
    mirrored. This port instead treats the end sentinel as "invalid"
    whenever it is passed, which is (a) what the C++ header documents
    (neighborhood_sampling.h:82-83: "if the next one is Stop (or
    invalid) then select Stop or flip"), and (b) exactly what C++ itself
    does, deterministically, for the same ``up == depth`` call whenever
    any field follows the contin field. Previously this edge raised
    `IndexError` (field_set.get_raw's bounds check) and crashed whole
    hill-climbing runs; no run that completed is affected, since every
    other path is unchanged.
    """
    if rng is None:
        rng = randGen()
    if fs.get_raw(inst, idx) == contin_spec.Stop:
        fs.set_raw(inst, idx, contin_spec.Left if rng.randbool() else contin_spec.Right)
    elif (
        idx + 1 != next_idx  # next_itr is invalid
        or next_idx == fs.raw_size()  # ... or is the end sentinel (see DIVERGENCE above)
        or fs.get_raw(inst, next_idx) == contin_spec.Stop
    ):
        if rng.randbool():
            fs.set_raw(inst, idx, contin_spec.Stop)
        else:
            flip_LR(fs, inst, idx)
    else:
        flip_LR(fs, inst, idx)


def generate_contin_neighbor(
    fs: field_set,
    inst: instance,
    idx: int,
    dist: int,
    rng: MT19937RandGen | None = None,
) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.cc:55 generate_contin_neighbor

    Modifies ``inst`` in place so its contin field at spec index ``idx``
    (``fs.contin()``'s own indexing -- C++'s `it.idx()`, NOT a raw index;
    see the module docstring) has one of its "significant pseudo-bits"
    changed, choosing among the (up to) ``2 * dist`` neighbouring
    encodings uniformly via `lazy_random_selector` -- see
    neighborhood_sampling.h:90-104's own doc comment for the worked
    [L R S S] example this reproduces.

    Computes `begin`/`depth`/`length`/`lowest_low`/`uppest_low` BEFORE
    the ``dist == 0`` early return, exactly matching C++'s comma-separated
    declaration order (neighborhood_sampling.cc:58-65) -- those reads can
    themselves raise (an invalid ``idx``) regardless of ``dist``, matching
    C++'s own UB-on-invalid-iterator behaviour for the same case.

    `lowest_low`/`uppest_low` are computed via SIGNED arithmetic
    (`max(length - dist, 0)`, matching C++'s explicit `(int)length -
    (int)dist` casts at neighborhood_sampling.cc:61) specifically to
    avoid the `size_t`-wrap this project's fidelity discipline calls out
    (`length - dist` alone, in unsigned arithmetic, would wrap to a huge
    positive value whenever ``dist > length``) -- Python's arbitrary-
    precision `int` never wraps, so this is a direct, faithful
    transliteration rather than a guard this port had to add.

    DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.cc:67):
    `OC_ASSERT(dist <= depth)` becomes `ValueError`, matching this
    module's own convention (`sample_from_neighborhood`/
    `generate_all_in_neighborhood` above).

    Draws exactly one value from `lazy_random_selector` (one
    `rng.randint_n` draw internally -- see that class's own docstring),
    then makes ``dist`` calls to `twiddle_contin_bit` above, each of
    which draws zero or one `rng.randbool()` depending on the pseudo-bit
    it lands on -- draw order matters and is verified element-for-element
    against classic (see this module's own test file).
    """
    if rng is None:
        rng = randGen()

    begin = fs.contin_to_raw_idx(idx)
    depth = fs.contin()[idx].depth
    length = fs.contin_length(inst, idx)
    lowest_low = max(length - dist, 0)
    uppest_low = min(length + 1, depth + 1 - dist)

    if dist == 0:
        return

    if not dist <= depth:
        raise ValueError(f"generate_contin_neighbor: dist ({dist}) must not exceed depth ({depth})")

    # Randomly choose the interval to modify, the intervals can be
    # [length - dist, length) to [length, length + dist), as long as
    # it is within [0, depth).
    low = lazy_random_selector(uppest_low, lowest_low, rng)()
    up = low + dist

    # Twiddle all pseudo bits in the selected interval.
    itr = begin + up - 1
    next_itr = fs.raw_size() if up == depth else itr + 1
    low_itr = begin + low
    while next_itr != low_itr:
        twiddle_contin_bit(fs, inst, itr, next_itr, rng)
        next_itr = itr
        itr -= 1


def sample_from_neighborhood(
    fs: field_set,
    dist: int,
    sample_size: int,
    out: list[instance],
    end: int,
    center_inst: instance,
) -> None:
    """C++: moses/moses/moses/neighborhood_sampling.h:134 sample_from_neighborhood

    Sample (with replacement) `sample_size` instances from the neighborhood
    at distance `dist` from `center_inst`. See the module docstring for the
    output-iterator convention. The "modify contin" branch below delegates
    to `generate_contin_neighbor` (S14 Task 3) -- see that function's own
    docstring, and the module docstring's RNG paragraph, for what it draws.

    Sharp edge, load-bearing for the bit-identity claim: the selector index
    `r` is numbered bit-range-first, then disc-range-second
    (``r < fs.n_bits()`` is a bit; ``fs.n_bits() <= r < fs.n_bits() +
    fs.n_disc_fields()`` is a disc) -- the OPPOSITE of the raw field order
    (disc-before-bit) that `count_neighborhood_size_from_index` and
    `vary_n_knobs` use. This is exactly what neighborhood_sampling.h:156-172
    does, driven by `fs.dim_size()`/`fs.n_bits()`, not by raw field position;
    it is not a bug to "fix" to match raw order.
    """
    # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h:138-140):
    # OC_ASSERT becomes ValueError.
    if len(center_inst) != fs.packed_width():
        raise ValueError(
            "Please make sure that the center_inst have the same size with the field_set"
        )

    dim = fs.dim_size()

    # DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h:144-146):
    # OC_ASSERT becomes ValueError.
    if dist > dim:
        raise ValueError(
            f"the sampling distance {dist} cannot be greater than the field dimension {dim}"
        )

    rng = randGen()
    for _ in range(sample_size):
        new_inst = instance(center_inst)
        select = lazy_random_selector(dim, rng)

        i = 1
        while i <= dist:
            r = select()
            if r < fs.n_bits():
                # modify bit
                bit_idx = fs.begin_bit_raw_idx() + r
                cur = fs.get_raw(new_inst, bit_idx)
                fs.set_raw(new_inst, bit_idx, 0 if cur else 1)
                i += 1
            elif fs.n_bits() <= r < (fs.n_bits() + fs.n_disc_fields()):
                # modify disc
                disc_pos = r - fs.n_bits()
                disc_idx = fs.begin_disc_raw_idx() + disc_pos
                multy = fs.disc_and_bit()[disc_pos].multy
                temp = 1 + rng.randint_n(multy - 1)
                cur = fs.get_raw(new_inst, disc_idx)
                fs.set_raw(new_inst, disc_idx, 0 if cur == temp else temp)
                i += 1
            elif r >= (fs.n_bits() + fs.n_disc_fields()):
                # modify contin
                #
                # DO NOT FIX: `dist` (this function's own parameter) is
                # NOT threaded through here -- C++ hardcodes `1` at this
                # exact call (neighborhood_sampling.h:178,
                # "@todo: now the distance is 1, choose the distance of
                # contin possibly different than 1"), so every contin
                # pseudo-bit change this function makes is distance-1
                # regardless of the OUTER `dist` this loop is trying to
                # reach overall -- ``i`` still only advances by 1 per
                # such change, exactly as it does for a bit/disc change.
                contin_idx = r - fs.n_bits() - fs.n_disc_fields()
                generate_contin_neighbor(fs, new_inst, contin_idx, 1, rng)
                i += 1

        _emit(out, end, new_inst)


class _DemeWriter:
    """Not itself a ported C++ symbol -- see the module docstring's
    "Output-iterator convention" section. Stands in for `deme.begin() +
    current_number_of_instances`, an `instance_set<composite_score>`
    output iterator: each `append(value)` writes `scored_instance(value)`
    (a fresh default-worst `composite_score` alongside the new instance,
    exactly what implicit construction-then-assignment produces in C++)
    into the next slot of a `deme` that the caller has already resized to
    fit every write, mirroring `*out++ = new_inst` for a
    `scored_instance<composite_score>`-valued iterator instead of a plain
    `instance`-valued one.
    """

    __slots__ = ("_count", "_deme", "_start")

    def __init__(self, deme: instance_set, start: int) -> None:
        self._deme = deme
        self._start = start
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def append(self, value: instance) -> None:
        """Not itself a ported C++ symbol -- the `*out++ = value` write
        `_emit` performs against this adapter; see the class docstring.
        Named `append` (not `_append`) so `_emit`'s own `out.append(value)`
        call works unchanged against either a plain list or this adapter.
        """
        self._deme[self._start + self._count] = scored_instance(value)
        self._count += 1


def sample_new_instances(
    total_number_of_neighbours: int | None,
    number_of_new_instances: int,
    current_number_of_instances: int,
    center_inst: instance,
    deme: instance_set,
    dist: int,
) -> int:
    """C++: moses/moses/moses/neighborhood_sampling.cc:214 sample_new_instances
    (``total_number_of_neighbours`` given -- the overload hill-climbing.cc:188
    actually calls) and neighborhood_sampling.cc:261 sample_new_instances
    (``total_number_of_neighbours=None`` -- computed via
    `count_neighborhood_size` first, exactly mirroring the second overload's
    own one-line body, which does nothing but that computation before
    delegating to the first overload with the now-known total). One Python
    function merges both, matching this module's established overload-
    merging convention (`count_neighborhood_size`, `generate_all_in_
    neighborhood` above) -- ``total_number_of_neighbours`` keeps the first
    overload's own leading position rather than moving to a trailing
    keyword, since that is the overload every real caller (hill-climbing.cc)
    reaches; a caller wanting the second overload's behaviour passes
    ``None`` in that same slot, exactly where it already knows to look.

    Resizes ``deme`` to fit the new instances, then samples (if the
    neighborhood is large relative to what's wanted) or exhaustively
    enumerates (otherwise) directly into the newly-resized slots via
    `_DemeWriter` -- see the module docstring's "Output-iterator
    convention" section for why a third out/end adapter exists alongside
    the plain-list one `generate_all_in_neighborhood`/
    `sample_from_neighborhood` themselves still use for their non-deme
    callers (Task 1's own tests).
    """
    if total_number_of_neighbours is None:
        # neighborhood_sampling.cc:261-270's own one-line body.
        total_number_of_neighbours = count_neighborhood_size(
            deme.fields(), dist, number_of_new_instances, inst=center_inst
        )

    # We assume that the total number of neighbors was just an estimate.
    # If the number of requested new instances is even close to the
    # estimate, then we need an accurate count of the total. We need
    # an accurate count for the case of generating the entire
    # neighborhood (as otherwise, the deme.resize will be bad).
    #
    # DO NOT FIX: when the caller already passed ``None`` above, this is a
    # redundant re-computation of the exact same value with the exact same
    # arguments -- classic pays this same double-computation cost, since
    # overload 2 calls overload 1 with its own freshly-computed total,
    # which then re-runs this very check (see the docstring above).
    if 2 * number_of_new_instances > total_number_of_neighbours:
        total_number_of_neighbours = count_neighborhood_size(
            deme.fields(), dist, number_of_new_instances, inst=center_inst
        )

    if number_of_new_instances < total_number_of_neighbours:
        # Resize the deme so it can take new instances.
        deme.resize(current_number_of_instances + number_of_new_instances)
        # Sample number_of_new_instances instances at
        # distance 'distance' from the exemplar.
        logger().debug("Sampling new instances")
        writer = _DemeWriter(deme, current_number_of_instances)
        sample_from_neighborhood(
            deme.fields(),
            dist,
            number_of_new_instances,
            writer,
            number_of_new_instances,
            center_inst,
        )
    else:
        number_of_new_instances = total_number_of_neighbours
        # Resize the deme so it can take new instances
        deme.resize(current_number_of_instances + number_of_new_instances)
        # Add all instances on the distance dist from the initial instance.
        logger().debug("Generating all instances")
        writer = _DemeWriter(deme, current_number_of_instances)
        generate_all_in_neighborhood(
            deme.fields(), dist, writer, number_of_new_instances, center_inst=center_inst
        )
    if logger().is_debug_enabled():
        logger().debug(
            f"SampleNewInstances: {number_of_new_instances}"
            f" With deme size: {deme.fields().dim_size()}"
        )
    return number_of_new_instances
