"""C++: moses/moses/scoring/behave_cscore.h and behave_cscore.cc

`behave_cscore` -- "Composite score calculated from the behavioral score.
The score is calculated as the sum of the bscore over all samples: score =
sum_x BScore(x) + penalty ... This is a 'minor' helper class, and exists
for three reasons: 1) avoids some redundancy of having the summation in
many places 2) Helps with keeping the score-caching code cleaner. 3) When
boosting, the summation above is no longer just a simple sum."
(behave_cscore.h:35-49's own doc comment, reproduced verbatim.)

Ports exactly `get_cscore` (behave_cscore.cc:51-56) and
`get_cscore_nocache` (behave_cscore.cc:63-98) per task-6-brief.md's
Reference line, plus `get_bscore` (behave_cscore.cc:41-44,
task-6-brief.md's own interface line) and `__init__`.

NOT ported, all deliberately out of scope for this task:

* `get_cscore`/`get_bscore` overloads taking a `scored_combo_tree_set`
  (ensembles) -- ensembles/boosting are out of scope for this whole cycle
  (types.py's module docstring).
* The `prr_cache_threaded` LRU cache (`_cscore_cache`, `wrapper`,
  `_have_cache`) -- an optimisation, not semantics, per task-6-brief.md's
  own "Global Constraints" note. `cache_size` is accepted (to match the
  constructor's interface line, `__init__(self, bscorer, cache_size=0)`)
  and stored, but a plain Python `dict` keyed on `_cache_key(tr)` (a
  collision-free structural key over the tree's own nodes -- see that
  function's docstring for why it is NOT keyed on the tree's printed
  form) stands in for the real cache -- see `__init__`'s docstring.
* `best_possible_score`/`worst_possible_score`/`min_improv`/`ignore_cols`/
  `ignore_rows`/`ignore_rows_at_times`/`get_ctable_usize`/`get_ctable`/
  `get_bscorer` -- thin one-line delegations to the underlying
  `bscore_base`, not named in task-6-brief.md's interface line for this
  class. Call `.get_bscorer()` (i.e. the `behave_cscore` instance's own
  `_bscorer` attribute) directly instead; every one of these was a single
  delegating call in C++ too.
"""

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.moses.moses.complexity import complexity_t
from pymetta.moses.moses.types import _to_score_t, composite_score, score_t
from pymetta.moses.scoring.scoring_base import bscore_base
from pymetta.util.tree import tree_node


def _cache_key(tr: combo_tree) -> tuple:
    """Not itself a ported C++ symbol -- a collision-free cache key for
    `get_cscore`'s tree cache below.

    FIX WAVE (final review, Important): this replaces the previous key,
    `combo_tree_to_str(tr)`. That printer (iostream_combo.py) is a
    faithful, deliberate port of `ostream_vertex`'s C++ formatting, which
    rounds a contin leaf to ~6 significant digits (`format(c, ".6g")`) --
    exactly right for what it is FOR (printing), but wrong as a cache key:
    `1.0`, `1.0000000001` and `1.00000000001` all print `"1 "`, so three
    structurally-different trees collided on one cache slot and
    `get_cscore` would return one tree's `composite_score` for another,
    silently. Contin `0.0` is MOSES's most common contin constant
    (complexity.py's own module docstring), so this was not a corner case.
    This function instead recurses over the tree's own node links
    directly -- no string formatting anywhere in the path -- so every
    vertex value, including a contin leaf's full float precision, is
    carried into the key bit-exactly: two keys compare equal iff the two
    trees are structurally identical down to that precision, matching
    what classic's own `prr_cache_threaded` achieves by hashing the tree
    object itself rather than a printed rendering of it.

    FIX WAVE ROUND 2 (re-review Critical): a bare `node.data` is not
    enough -- it is exactly the "compare a vertex by value without
    checking its type first" trap the rest of this fix wave exists to
    close. `contin_t` is a bare Python `float` and `builtin` an
    `IntEnum`, so `1.0 == builtin.logical_true` (and `hash(1.0) ==
    hash(builtin.logical_true)`) is True -- a contin leaf `1.0`
    (`str_to_combo_tree("1")`) and a boolean leaf `builtin.logical_true`
    (`str_to_combo_tree("true")`) produced the IDENTICAL key under the
    round-1 fix, so `get_cscore` on one tree silently returned the
    other's cached `composite_score`. Every node's value is now paired
    with `type(node.data)` -- the concrete `vertex` alternative's own
    class (`builtin`, `argument`, `float`, `enum_t`/`str`, ...) -- so two
    vertices of different variant types can never produce equal keys:
    `float == builtin` (the CLASSES, not instances) is always False, so
    `(float, 1.0, ...) != (builtin, builtin.logical_true, ...)`
    regardless of what the second element compares as.
    """
    root = tr.begin()
    if root is None:
        return ()

    def _walk(node: tree_node) -> tuple:
        children = []
        child = node.first_child
        while child is not None:
            children.append(_walk(child))
            child = child.next_sibling
        return (type(node.data), node.data, tuple(children))

    return _walk(root)


class behave_cscore:
    """C++: moses/moses/scoring/behave_cscore.h:50 behave_cscore"""

    def __init__(self, bscorer: bscore_base, cache_size: int = 0) -> None:
        """C++: moses/moses/scoring/behave_cscore.cc:33 behave_cscore

        `cache_size` matches task-6-brief.md's interface line
        (`__init__(self, bscorer, cache_size=0)`) but the cache itself is
        an optimisation, not semantics (task-6-brief.md's own "Global
        Constraints" note) -- a plain `dict`, keyed on `_cache_key(tr)`
        (see that function's own docstring for why, since a bare
        `combo_tree`/`tree` object here is not hashable: `util/tree.py`
        defines no `__hash__`, matching C++'s `tree<T>` having no
        `operator<`/hash either -- classic's own `prr_cache_threaded`
        hashes on a caller-supplied functor, not on the tree object's own
        identity). `_have_cache` mirrors C++'s `0 < initial_cache_size`
        gate exactly: `cache_size=0` (the default) disables caching
        altogether, same as classic.
        """
        self._bscorer = bscorer
        self._have_cache = cache_size > 0
        self._cscore_cache: dict[tuple, composite_score] = {}

    def get_bscore(self, tr: combo_tree) -> list[score_t]:
        """C++: moses/moses/scoring/behave_cscore.cc:41 behave_cscore::get_bscore"""
        return self._bscorer(tr)

    def get_cscore(self, tr: combo_tree) -> composite_score:
        """C++: moses/moses/scoring/behave_cscore.cc:51 behave_cscore::get_cscore

        "Use the cache, if it is enabled." (behave_cscore.cc:53's own
        comment.)
        """
        if self._have_cache:
            key = _cache_key(tr)
            if key not in self._cscore_cache:
                self._cscore_cache[key] = self.get_cscore_nocache(tr)
            return self._cscore_cache[key]
        return self.get_cscore_nocache(tr)

    def get_cscore_nocache(self, tr: combo_tree) -> composite_score:
        """C++: moses/moses/scoring/behave_cscore.cc:63 behave_cscore::get_cscore_nocache

        # DIVERGENCE(C++ moses/moses/scoring/behave_cscore.cc:66-86): classic
        # wraps `get_bscore(tr)` in a try/catch for
        # `combo::EvalException` (raised by the CONTIN interpreter on an
        # out-of-domain operand -- negative log, division by zero -- or
        # an unrepresentable result), returning `worst_composite_score`
        # on catch. This port's scoring path is boolean-only (see
        # scoring_base.py's module docstring): `boolean_interpreter`
        # (interpreter.py) has no contin-domain failure mode to raise an
        # equivalent exception for, so there is nothing here to catch.
        # If a future task adds contin scoring, this is the place its
        # own evaluation exceptions would need to be caught the same way.

        `_bscorer.sum_bscore(bs)`, `_bscorer.get_complexity(tr)` and
        `_bscorer.get_complexity_coef()` are exactly `bscore_base`'s own
        methods (scoring_base.py) -- `res`/`cpxy`/`cpxy_coef` below are
        named to match behave_cscore.cc:87-97's own local variables.

        `cpxy * cpxy_coef` (behave_cscore.cc:97) is narrowed to `score_t`
        right here: `cpxy` is `complexity_t` (`unsigned`) and `cpxy_coef`
        is `score_t` (`float`), so C++'s usual arithmetic conversions
        promote `cpxy` to `float` and perform the multiplication in
        float32 -- not in double. `composite_score.__init__` (types.py)
        ALSO narrows its own `complexity_penalty` parameter now (fix round
        2's "eleventh site" fix), so this call's explicit narrowing is no
        longer the only thing standing between an unnarrowed product and
        the constructor -- it is kept anyway, both because it is the
        direct, literal analogue of the C++ multiplication itself
        happening in float32 (independent of whatever the constructor
        does with its result), and as defense in depth: narrowing an
        already-narrow value is idempotent, so keeping both narrowing
        points costs nothing and loses no precision either way.
        """
        bs = self.get_bscore(tr)
        res: score_t = self._bscorer.sum_bscore(bs)

        cpxy: complexity_t = self._bscorer.get_complexity(tr)
        cpxy_coef: score_t = self._bscorer.get_complexity_coef()

        return composite_score(res, cpxy, _to_score_t(cpxy * cpxy_coef), 0.0)
