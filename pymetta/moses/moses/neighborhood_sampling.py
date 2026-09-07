"""Enumerate and sample the knob-setting neighborhood of an instance.

Ports `moses/moses/moses/neighborhood_sampling.{h,cc}`, Boolean/discrete
paths only. This is the first ported call site that actually draws from
`pymetta.util.rng.randGen()` -- S0-S8 proved the RNG bit-identical but never
exercised it. `generate_initial_sample` and `sample_from_neighborhood` call
that process-wide singleton directly, exactly as their C++ signatures do
(neither takes an RNG parameter); a caller wanting a reproducible sequence
reseeds it first with ``randGen().seed(s)``, matching how the C++ oracle
probe (`tools/cpp_probe/neighborhood_probe.cc`) does it.

Continuous fields are out of scope for this port (`flip_LR`,
`twiddle_contin_bit`, `generate_contin_neighbor` at neighborhood_sampling.cc:32,
38, 55 are DEFERRED -- not ported at all, no placeholder dispatch for them).
Every function below that has a continuous-field branch in C++
(`count_neighborhood_size_from_index`, `vary_n_knobs`,
`sample_from_neighborhood`) raises ``NotImplementedError`` if it actually
reaches a continuous raw index, rather than silently skipping or
mishandling it -- this never fires for a Boolean/discrete-only field_set
(every field_set exercised by this port's tests), since the guard only
triggers when ``fs`` genuinely has continuous fields. Term fields need no
equivalent guard: they do not exist anywhere in this port (see
field_set.py's own term exclusion), so the C++ "terms" branch -- always an
empty index range here -- is omitted outright rather than guarded.

`sample_new_instances` (both overloads, neighborhood_sampling.cc:214, 261)
is NOT ported: both need `instance_set<composite_score>`, the S9 Task 0
bridge container, which does not exist in this port yet.

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

from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.util.lazy_random_selector import lazy_random_selector
from pymetta.util.rng import randGen

# DIVERGENCE(C++ moses/moses/moses/neighborhood_sampling.h:27):
# std::numeric_limits<size_t>::max() pinned to the 64-bit ABI
# instance.py already measures and pins (bits_per_packed_t = 64).
_SIZE_T_MAX = (1 << 64) - 1


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
    # This is not exercised by classic's own working code: the sole call
    # site (count_neighborhood_size_from_index's bit branch, below) always
    # guards `dist <= rb` before calling, so k < n never actually happens.
    # Python's math.comb(k, n) returns the mathematically-correct 0 for
    # k < n instead of raising; this port relies on callers respecting the
    # same k >= n precondition classic's real call site does, and does not
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
        raise NotImplementedError(
            "count_neighborhood_size_from_index: continuous fields are deferred "
            "(see generate_contin_neighbor)"
        )

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
        raise NotImplementedError(
            "vary_n_knobs: continuous fields are deferred (see generate_contin_neighbor)"
        )

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


def _uniform_randomize_disc(fs: field_set, inst: instance) -> None:
    """C++: moses/moses/eda/initialization.cc:88 uniform_randomize_disc

    Reached from `generate_initial_sample` via `randomize` (see
    `_randomize` below) -- `moses/moses/eda/initialization.{h,cc}` itself is
    not otherwise ported; only the Boolean/discrete subset this call chain
    needs is.
    """
    rng = randGen()
    for idx in range(fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()):
        multy = fs.disc_and_bit()[fs.raw_to_disc_idx(idx)].multy
        fs.set_raw(inst, idx, rng.randint_n(multy))


def _uniform_randomize_bit(fs: field_set, inst: instance) -> None:
    """C++: moses/moses/eda/initialization.cc:80 uniform_randomize_bit"""
    rng = randGen()
    for idx in range(fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()):
        fs.set_raw(inst, idx, 1 if rng.randbool() else 0)


def _randomize(fs: field_set, inst: instance) -> None:
    """C++: moses/moses/eda/initialization.cc:96 randomize

    Only the Boolean/discrete tail of C++'s
    ``occam_randomize_term; occam_randomize_contin; uniform_randomize_disc;
    uniform_randomize_bit;`` sequence is ported. `occam_randomize_term` has
    no equivalent here at all: term fields do not exist anywhere in this
    port (field_set.py excludes them entirely), so that call's iteration
    range is always empty and it draws nothing -- there is nothing to guard.
    `occam_randomize_contin` is deferred along with the rest of the
    continuous domain (see generate_contin_neighbor's DEFER marker): a
    field_set with continuous fields raises here rather than silently
    skipping their randomization, which would silently desync the RNG
    stream from C++'s. Draw order for the disc-then-bit tail that remains
    matches C++ exactly.
    """
    if fs.n_contin_fields() > 0:
        raise NotImplementedError(
            "_randomize: continuous fields are deferred (see generate_contin_neighbor)"
        )
    _uniform_randomize_disc(fs, inst)
    _uniform_randomize_bit(fs, inst)


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
    output-iterator convention and the continuous-field guard.

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
                raise NotImplementedError(
                    "sample_from_neighborhood: continuous fields are deferred "
                    "(see generate_contin_neighbor)"
                )

        _emit(out, end, new_inst)
