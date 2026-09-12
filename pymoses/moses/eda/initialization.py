"""C++: moses/moses/eda/initialization.h

"various routines for initializing instances" (the header's own comment).
This is the last file of `eda/`'s remainder -- see `eda/__init__.py`'s
module docstring for the rest of the package and for a correction to that
docstring's own earlier claim about this file (below).

Ported: `occam_randomize_contin` (both overloads, initialization.cc:30,69),
`uniform_randomize_bit` (initialization.cc:80), `uniform_randomize_disc`
(initialization.cc:88), and `randomize` (initialization.cc:96, the
combining wrapper).

NOT ported: `occam_randomize_term` (both overloads, initialization.cc:38,63).
Term fields do not exist anywhere in this port -- `field_set.py` excludes
the whole term-algebra domain outright (its own module docstring) -- so
there is no `const_term_iterator`/`term_tree` type to give either overload
a real signature against, and the loop `randomize()` itself would run
(`for it in fs.begin_term(inst)...end_term(inst)`) is *always empty* for
every `field_set` this port can construct. `randomize()` below omits the
call entirely rather than keeping a phantom no-op, matching the existing
convention in this call chain's OTHER prior partial port (see next
paragraph): term draws nothing to skip.

Correction to `eda/__init__.py`'s "NOT ported" claim about this file
(written when `univariate.py` landed): that entry says this whole file is
"genuinely unreachable" because `git grep -rn 'occam_randomize\\|
uniform_randomize' $CLASSIC/moses` finds no call site outside
`initialization.cc` itself. That grep is accurate for those two names, but
incomplete: it did not also search for `randomize(` (the fifth, combining
function), which DOES have an external call site --
`neighborhood_sampling.h:58`'s `generate_initial_sample`, the very function
`univariate.cc` uses to seed its initial population. That call site was
already handled, independently, by an earlier (S9/S14) port: rather than
importing this file, `pymoses/moses/moses/neighborhood_sampling.py` carries
its own private `_randomize`/`_uniform_randomize_disc`/
`_uniform_randomize_bit` (module-private, underscore-prefixed) that
reimplements exactly the Boolean/discrete tail of C++'s `randomize()` --
and, because `occam_randomize_contin` was not portable at the time
(`generate_contin_neighbor` had not yet landed), its `_randomize` raises
`NotImplementedError` on any field_set with contin fields. `generate_
contin_neighbor` landed in S14 Task 3, so that gap is now closeable, but
`neighborhood_sampling.py` is outside this task's owned tree (see this
project's task brief) and is not edited here -- reported as a follow-up:
the public `randomize`/`uniform_randomize_disc`/`uniform_randomize_bit`
below are, apart from `randomize`'s new contin branch, faithful
transliterations of the exact same C++ that `neighborhood_sampling.py`'s
private trio already reproduces, so switching that module to import from
here (instead of maintaining a second copy) would be a safe, mechanical
follow-up, and would also lift its `NotImplementedError`-on-contin gap for
free.

Every function below draws from the process-wide `randGen()` singleton via
its `rng` parameter, defaulting to `None` (resolved to `randGen()` at call
time) -- matching C++'s own `RandGen& rng = randGen()` default argument,
re-resolved at each call site to the same live singleton either way (see
`neighborhood_sampling.py`'s module docstring for the identical reasoning
this port already applies elsewhere).
"""

from __future__ import annotations

from pymoses.moses.moses.neighborhood_sampling import generate_contin_neighbor
from pymoses.moses.representation.field_set import field_set
from pymoses.moses.representation.instance import instance
from pymoses.util.rng import MT19937RandGen, randGen

__all__ = [
    "occam_randomize_contin",
    "uniform_randomize_bit",
    "uniform_randomize_disc",
    "randomize",
]


def occam_randomize_contin(
    fs: field_set,
    inst: instance,
    idx: int | None = None,
    rng: MT19937RandGen | None = None,
) -> None:
    """C++: moses/moses/eda/initialization.cc:30 occam_randomize_contin

    "Occam randomize a particular contin ... field. Note that it is not
    strictly occam in the Solomonoff sense because the size is uniformely
    distributed." (the header's own comment, term part elided.)

    One Python function merges both C++ overloads (matching this module's
    own `randomize`, and `neighborhood_sampling.py`'s merged-overload
    convention for its own iterator-pair functions): passing ``idx``
    (C++'s `field_set::contin_iterator::idx()`, i.e. the SPEC index into
    `fs.contin()` -- NOT a raw index, exactly as `generate_contin_neighbor`
    already takes it) calls the single-field overload
    (initialization.cc:30-35); omitting it (``idx=None``) calls the
    all-fields overload (initialization.cc:69-74), which loops
    `fs.contin()` in spec order and recurses into the single-field form
    once per spec -- draw order across specs therefore follows `fs.contin()`
    order exactly, matching C++'s `contin_iterator` traversal order.

    Single-field body: `unsigned int n = rng.randint(fs.contin()[it.idx()].
    depth); moses::generate_contin_neighbor(fs, inst, it, n, rng);` -- one
    `rng.randint_n(depth)` draw (C++'s `RandGen::randint(int n)`, see
    `pymoses/util/rng.py`'s own docstring on that mapping), then whatever
    `generate_contin_neighbor` itself draws (see that function's own
    docstring in `neighborhood_sampling.py`).
    """
    if rng is None:
        rng = randGen()
    if idx is None:
        for spec_idx in range(len(fs.contin())):
            occam_randomize_contin(fs, inst, spec_idx, rng)
        return
    depth = fs.contin()[idx].depth
    n = rng.randint_n(depth)
    generate_contin_neighbor(fs, inst, idx, n, rng)


def uniform_randomize_bit(
    fs: field_set, inst: instance, rng: MT19937RandGen | None = None
) -> None:
    """C++: moses/moses/eda/initialization.cc:80 uniform_randomize_bit

    `generate(fs.begin_bit(inst), fs.end_bit(inst), bind(&RandGen::randbool,
    ref(rng)));` -- one `rng.randbool()` draw per bit field, in raw order.
    Reproduces `neighborhood_sampling.py`'s private `_uniform_randomize_bit`
    exactly (see this module's own docstring on that duplication).
    """
    if rng is None:
        rng = randGen()
    for idx in range(fs.begin_bit_raw_idx(), fs.end_bit_raw_idx()):
        fs.set_raw(inst, idx, 1 if rng.randbool() else 0)


def uniform_randomize_disc(
    fs: field_set, inst: instance, rng: MT19937RandGen | None = None
) -> None:
    """C++: moses/moses/eda/initialization.cc:88 uniform_randomize_disc

    `for (... it = fs.begin_disc(inst); it != fs.end_disc(inst); ++it)
    it.randomize(rng);` where `disc_iterator::randomize` (field_set.h:1053)
    is `_fs->set_raw(*_inst, _idx, rng.randint(multy()))` -- one
    `rng.randint_n(multy)` draw per disc field, in raw order. Reproduces
    `neighborhood_sampling.py`'s private `_uniform_randomize_disc` exactly.
    """
    if rng is None:
        rng = randGen()
    for idx in range(fs.begin_disc_raw_idx(), fs.end_disc_raw_idx()):
        multy = fs.disc_and_bit()[fs.raw_to_disc_idx(idx)].multy
        fs.set_raw(inst, idx, rng.randint_n(multy))


def randomize(fs: field_set, inst: instance, rng: MT19937RandGen | None = None) -> None:
    """C++: moses/moses/eda/initialization.cc:96 randomize

    "Occam randomize all contin and term fields, and uniformly randomize
    all bit and disc fields" (the header's own comment) -- C++ body,
    verbatim order: `occam_randomize_term(fs, inst, rng);
    occam_randomize_contin(fs, inst, rng); uniform_randomize_disc(fs, inst,
    rng); uniform_randomize_bit(fs, inst, rng);`.

    `occam_randomize_term`'s call is OMITTED, not merely a no-op stand-in:
    term fields do not exist in this port at all (see this module's own
    docstring), so that call's own iteration range is always empty and it
    draws nothing, for every `field_set` constructible here -- there is
    nothing to reproduce. The remaining three calls run in C++'s exact
    order (contin, then disc, then bit) -- this ordering is the single
    most important property of this function to preserve, since a caller
    (`neighborhood_sampling.py`'s `generate_initial_sample`, once/if it is
    switched to import this module -- see this module's own docstring) can
    reseed and diff the whole instance against C++, and any reordering
    would desync every draw from the first contin field onward.
    """
    if rng is None:
        rng = randGen()
    occam_randomize_contin(fs, inst, None, rng)
    uniform_randomize_disc(fs, inst, rng)
    uniform_randomize_bit(fs, inst, rng)
