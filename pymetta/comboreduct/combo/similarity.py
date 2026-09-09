"""C++: moses/comboreduct/combo/similarity.h

"This file implements a combo tree similarity measure, obtained by
directly comparing combo trees." (the .cc file's own comment, which goes
on for ~90 lines describing the algorithm -- read it at
`$CLASSIC/moses/comboreduct/combo/similarity.cc` before touching this
file; it is not repeated here in full.) Roughly: convert an n-ary combo
tree to an (implicit) right-leaning binary tree, count occurrences of each
2-child "branch" shape as a string-keyed histogram (`tree_flatten`), then
take the Manhattan (lp_1) distance between two such histograms
(`tree_similarity`).

Oracle: `$CLASSIC/tests/comboreduct/combo/similarityUTest.cxxtest`, a
REGISTERED test (`ADD_CXXTEST(similarityUTest)`,
tests/comboreduct/combo/CMakeLists.txt:22 -- verified, not assumed) that
calls `tree_flatten`/`tree_similarity`/`toString` directly with the
configuration this module ports. Its own cases are reproduced verbatim in
`tests/moses/... /test_similarity.py`, plus `tools/cpp_probe/
similarity_probe.cc`, which links the REAL exported symbols from
`libcomboreduct.so` for additional adversarial cases (see below).

## A genuine, measured bug in classic's own `tree_similarity`, ported verbatim

`tree_similarity(av, bv)`'s first loop is, verbatim (similarity.cc:180-197):

    size_t dist = 0;
    tree_branch_vector diff = bv;
    for (auto pr : av) {
        size_t acnt = pr.second;
        size_t bcnt = 0;
        diff.erase(pr.first);
        try { bcnt = bv.at(pr.first); } catch (const std::out_of_range&) {}
        dist += fabs(acnt - bcnt);
    }

`acnt`/`bcnt`/`dist` are ALL `size_t` (unsigned, 64-bit on this project's
reference toolchain -- see hill_climbing.py's own `_SIZE_T_MAX`). Whenever
`bcnt > acnt` for a branch key present in both histograms, `acnt - bcnt`
WRAPS (unsigned underflow) to a value within a few units of 2**64, NOT a
negative number `fabs` could act on -- `fabs` only ever sees the
already-wrapped, already-positive `size_t` value (implicitly converted to
`double` for the call), so it is a no-op here, not a fix. That double,
almost always indistinguishable from 2**64.0 after IEEE-754 round-to-
nearest (any wrap margin under 2**11 rounds UP to exactly 2**64.0 -- see
`_double_to_uint64`'s own docstring), is then added to `dist` (itself
promoted to `double` for the `+=`, then the double SUM is converted back
to `size_t`) -- converting a double `>= 2**64` back to a 64-bit unsigned
integer is UNDEFINED BEHAVIOUR in C++. MEASURED (not merely reasoned
about) on this project's reference toolchain (x86-64 GCC, see this
module's own test file for the exact snippets that pinned each step):
`(uint64_t)2**64.0 == 0`, and more generally x86-64 GCC compiles
`(uint64_t)d` for out-of-signed-range `d` via "subtract 2**63, convert via
the SIGNED `cvttsd2si` instruction (which yields the x86 'integer
indefinite' value `INT64_MIN` on overflow), add 2**63 back, mod 2**64" --
`_double_to_uint64` below reproduces exactly this compiled sequence, not
merely "the target statistic".

CONFIRMED end to end against the real, linked `tree_similarity` symbol
(`tools/cpp_probe/similarity_probe.cc`): `tree_similarity("+(1 1 1)",
"+(1 1 1 1 1 1 1)")` (branch `"+(1 +)"` occurs 1 time in the first tree,
5 times in the second) returns **0** -- and the ARGUMENT-SWAPPED call,
`tree_similarity("+(1 1 1 1 1 1 1)", "+(1 1 1)")` (same branch, now
`acnt=5 > bcnt=1`, no wrap), returns **4**. `tree_similarity` is therefore
**not symmetric** in classic, for realistic inputs, and (per the
"absorption" property `_double_to_uint64`'s docstring proves) a SINGLE
wraparound resets everything the first loop had accumulated up to that
point back to exactly 0, in `std::map`'s sorted-key iteration order --
NOT merely that one term. Ported verbatim, with a DO NOT FIX marker, as
`_first_loop_distance` below. Nothing in this file "fixes" the asymmetry;
it is reproduced precisely, in both directions, and is why
`tree_similarity` here takes its two arguments in a specific, ORDER-
SENSITIVE way exactly as classic's parameter names (`av`, `bv` -- "a
histogram" first, "b histogram" second) already suggest.
"""

from __future__ import annotations

import math

from pymetta.comboreduct.combo.iostream_combo import _vertex_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import combo_tree

__all__ = ["tree_branch_vector", "tree_flatten", "tree_similarity", "toString"]

tree_branch_vector = dict[str, int]

_SIZE_T_MOD = 1 << 64  # C++: 2**64, size_t wraparound modulus (64-bit size_t)
_INT64_MIN = -(1 << 63)


def _cvttsd2si_signed(d: float) -> int:
    """Not itself a ported C++ symbol -- models the x86-64 `cvttsd2si`
    instruction (scalar double -> signed 64-bit int, truncating toward
    zero) that GCC compiles a `(int64_t)double` conversion to. For `d`
    within `[-2**63, 2**63)` this is plain truncation; for `d` outside
    that range (or NaN), the hardware produces the "integer indefinite"
    result `INT64_MIN` (`0x8000000000000000`), NOT a saturated value --
    this is the specific, measured behaviour `_double_to_uint64` below
    depends on. MEASURED via a direct C++ snippet on this project's
    reference toolchain (see test_similarity.py's own conversion tests
    for the exact inputs/outputs pinned).
    """
    if d != d or d >= 2**63 or d < _INT64_MIN:  # d != d is the NaN check
        return _INT64_MIN
    return math.trunc(d)


def _double_to_uint64(d: float) -> int:
    """Not itself a ported C++ symbol -- models GCC/x86-64's compiled
    sequence for `(uint64_t)d` where `d` is a non-negative `double` that
    may exceed `INT64_MAX` (2**63 - 1): "subtract 2**63, convert the
    (now possibly out-of-signed-range) remainder via `_cvttsd2si_signed`,
    add 2**63 back, mod 2**64" -- the standard "unsigned via signed"
    lowering GCC uses on targets without a direct scalar double-to-
    unsigned-int64 instruction available at this codebase's baseline.

    MEASURED directly (not derived from the x86 manual alone) against a
    compiled C++ probe over six values spanning the boundary --
    `4.0 -> 4`, `0.0 -> 0`, `2**64 -> 0`, `2**64 - 2048 -> 2**64 - 2048`
    (exact, not zero), `2**63 -> 2**63` (exact), and `~2**64 + 1e10 -> 0`
    -- all six reproduced exactly by this function (test_similarity.py).

    Consequence for `_first_loop_distance` below, proven algebraically and
    confirmed against the two end-to-end oracle cases in this module's own
    docstring: once `d >= 2**64` (which any REALISTIC branch-count wrap
    margin, i.e. under 2**11, always rounds up to when converted from the
    wrapped `size_t` -- see `_first_loop_distance`), `d - 2**63 >= 2**63`,
    `_cvttsd2si_signed` returns `INT64_MIN`, and `(INT64_MIN + 2**63) %
    2**64 == 0` unconditionally -- so ANY sum that reaches `>= 2**64`
    collapses to exactly 0, and adding a further non-negative value to an
    already `>= 2**64` sum can only keep it `>= 2**64` (never bring it back
    down), so the result stays 0 regardless of how large the prior
    accumulator was. This "absorption" is what makes a single wraparound
    erase everything accumulated before it in iteration order.
    """
    if d < 2**63:
        return math.trunc(d) % _SIZE_T_MOD
    reduced = d - 2**63
    signed = _cvttsd2si_signed(reduced)
    return (signed + 2**63) % _SIZE_T_MOD


def _first_loop_distance(av: tree_branch_vector, bv: tree_branch_vector) -> tuple[int, dict]:
    """C++: moses/comboreduct/combo/similarity.cc:180-190 (the `for (auto
    pr : av)` loop, up to but not including the `for (auto p : diff)`
    tail -- see `tree_similarity` below for the whole function).

    Iterates `av` in SORTED KEY order -- `std::map`'s own iteration order,
    which is what decides WHICH terms are still live when a wraparound
    resets the accumulator (see this module's own docstring). Returns
    `(dist, diff)` where `diff` is `bv` with every key also present in
    `av` removed (`dict.pop(key, None)` mirrors `std::map::erase` on a
    possibly-absent key: both are silent no-ops when the key is missing).

    `bv.at(pr.first)` wrapped in `try/except KeyError` here reproduces
    `bv.at(pr.first)` wrapped in `try/catch (std::out_of_range&)` in C++
    exactly -- `dict.get(key, 0)` is equivalent and used directly, since
    the `try/except` shape adds nothing observable here (unlike a case
    where the caught branch does something the plain default could not).
    """
    dist = 0
    diff = dict(bv)
    for key in sorted(av):
        acnt = av[key]
        diff.pop(key, None)
        bcnt = bv.get(key, 0)
        raw_diff = (acnt - bcnt) % _SIZE_T_MOD  # size_t subtraction: wraps
        term = float(raw_diff)  # implicit size_t -> double conversion
        # fabs(term) is a genuine no-op: term is already non-negative
        # (raw_diff is a size_t reduced mod 2**64, always in [0, 2**64)).
        dist = _double_to_uint64(float(dist) + term)
    return dist, diff


def tree_flatten(tree_or_str: combo_tree | str) -> tree_branch_vector:
    """C++: moses/comboreduct/combo/similarity.cc:141 tree_flatten (combo_tree
    overload) and :156 tree_flatten (std::string overload) -- merged into
    one function here (matching this port's existing overload-merging
    convention, e.g. `occam_randomize_contin` in eda/initialization.py),
    since the string overload is a one-line wrapper
    (`str_to_combo_tree(str)` then the combo_tree overload) in C++ too.
    """
    if isinstance(tree_or_str, str):
        tree = str_to_combo_tree(tree_or_str)
    else:
        tree = tree_or_str
    counter: tree_branch_vector = {}
    root = tree.begin()
    if root is not None:
        _tree_flatten_rec(counter, tree, root)
    return counter


def _tree_flatten_rec(ctr: tree_branch_vector, t: combo_tree, node) -> None:
    """C++: moses/comboreduct/combo/similarity.cc:117 tree_flatten_rec (`static`,
    file-local in C++; not exported here either -- not in `__all__`).

    See this module's docstring and similarity.cc's own ~90-line algorithm
    comment for the "convert to a right-leaning binary tree, count 2-child
    branch shapes" idea; this is a direct transliteration of the four
    cases (0, 1, 2, >2 children) similarity.cc:117-165 handles, using
    `_vertex_to_str` (iostream_combo.py) for each individual vertex's own
    token text -- matching `ss << *root` etc., which prints ONLY the
    node's own label, never a subtree, exactly as `_vertex_to_str`
    reproduces `operator<<(ostream&, const vertex&)` (`ostream_vertex`
    with no labels, `output_format.combo`).

    `t.number_of_children(first) != 0` (guarding each recursive call, e.g.
    similarity.cc:129) is reproduced as `len(children) != 0` on the
    children list already fetched for that node below, rather than a
    second `t.number_of_children` call -- behaviourally identical, one
    fewer tree traversal.
    """
    children = list(t.children(node))
    numch = len(children)
    root_str = _vertex_to_str(node.data)

    if numch == 0:
        ctr[root_str] = ctr.get(root_str, 0) + 1
        return

    if numch == 1:
        first = children[0]
        key = f"{root_str}({_vertex_to_str(first.data)})"
        ctr[key] = ctr.get(key, 0) + 1
        first_children = list(t.children(first))
        if first_children:
            _tree_flatten_rec(ctr, t, first)
        return

    # numch >= 2: distribute associatively -- see similarity.cc:139-150's
    # own comment (f *does not* have to actually be associative; this is
    # purely an accounting trick, never evaluated as combo).
    i = 0
    first = children[i]
    while i < numch - 2:
        first = children[i]
        key = f"{root_str}({_vertex_to_str(first.data)} {root_str})"
        ctr[key] = ctr.get(key, 0) + 1
        if list(t.children(first)):
            _tree_flatten_rec(ctr, t, first)
        i += 1

    first = children[i]
    second = children[i + 1]
    key = f"{root_str}({_vertex_to_str(first.data)} {_vertex_to_str(second.data)})"
    ctr[key] = ctr.get(key, 0) + 1
    if list(t.children(first)):
        _tree_flatten_rec(ctr, t, first)
    if list(t.children(second)):
        _tree_flatten_rec(ctr, t, second)


def tree_similarity(
    a: combo_tree | str | tree_branch_vector, b: combo_tree | str | tree_branch_vector
) -> int:
    """C++: moses/comboreduct/combo/similarity.cc:167 tree_similarity
    (tree_branch_vector overload), :199 (combo_tree overload), :206
    (std::string overload) -- all THREE merged into one function here
    (see `tree_flatten`'s own docstring for this port's overload-merging
    convention): a plain `dict` argument is treated as an already-flattened
    histogram; a `str` is parsed then flattened; anything else (a
    `combo_tree`) is flattened directly.

    "compute the manhattan distance between trees. The manhattan distance
    is the same as the lp_1 distance." (similarity.cc:169-172's own
    comment) -- see this module's own docstring for why that description
    is only accurate when `bv`'s counts never exceed `av`'s for a shared
    branch; ARGUMENT ORDER MATTERS here, unlike a true distance metric.

    Second loop (similarity.cc:192-196, "add anything in bv that was not
    in av"): plain, un-wrapped `unsigned` accumulation -- `p.second` is a
    map value (`unsigned`), never a `double`, so no float round-trip
    applies here; only `dist = (dist + diff[key]) % 2**64` for symmetry
    with the first loop's own wraparound convention (unreachable in
    practice: it would need `dist` to already be within `2**64 -
    max(diff.values())` of the modulus, impossible for any realistic
    combo tree).
    """
    av = _as_tree_branch_vector(a)
    bv = _as_tree_branch_vector(b)
    dist, diff = _first_loop_distance(av, bv)
    for key in sorted(diff):
        dist = (dist + diff[key]) % _SIZE_T_MOD
    return dist


def _as_tree_branch_vector(x: combo_tree | str | tree_branch_vector) -> tree_branch_vector:
    """Not itself a ported C++ symbol -- dispatch helper for
    `tree_similarity`'s three merged overloads (see its own docstring)."""
    if isinstance(x, dict):
        return x
    return tree_flatten(x)


def toString(tbv: tree_branch_vector) -> str:
    """C++: moses/comboreduct/combo/similarity.h:44 toString

    `std::stringstream ss; ss << tbv; return ss.str();` where `operator<<`
    (similarity.cc:222-230) iterates `tbv` (a `std::map`, hence SORTED key
    order) writing `"key : count"` pairs joined by `", "`.

    # DIVERGENCE(C++ moses/comboreduct/combo/similarity.h:44-49): classic
    # defines this function's BODY directly in the header, with no
    # `inline` keyword -- a real ODR violation waiting to happen the
    # moment more than one translation unit includes similarity.h (none
    # currently does; only similarity.cc does). Ported as an ordinary
    # Python function; Python has no translation-unit/ODR concept for this
    # to matter, so there is nothing to reproduce or diverge from
    # behaviourally -- noted here only because it is a genuine defect in
    # the header worth knowing about if this file is ever touched again.
    """
    return ", ".join(f"{key} : {tbv[key]}" for key in sorted(tbv))
