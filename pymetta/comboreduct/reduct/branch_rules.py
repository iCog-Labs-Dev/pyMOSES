"""C++: moses/comboreduct/reduct/branch_rules.{h,cc}

S14 Task 4. Four rules over `cond`/`contin_if`'s branching structure --
`cond` and `contin_if` share the same (predicate, consequent) pair-list
plus trailing "else" shape (`eval.py`'s already-ported `builtin.contin_if |
builtin.cond` evaluation case), which is why `reduce_cond_else` and
`reduce_cond_const` both fire on either vertex, while `reduce_cond_arg`
(boolean-reduce every predicate) and `reduce_cond_adjacent` (merge
adjacent identical consequents) are `cond`-only in C++ -- ported that way
here, verbatim, not "fixed" to also cover `contin_if` symmetrically.

MEASURED UPSTREAM BUG in `zap_one` (branch_rules.cc:47-52), the private
`cond(v) -> v` helper both `reduce_cond_else` and `reduce_cond_const` call
when a cond/contin_if has been reduced down to its lone trailing "else"
value: it silently DISCARDS that value's own children when the value is a
compound (non-leaf) expression, rather than splicing them up. Confirmed
directly against the compiled `libcomboreduct.so`, not inferred from
reading the source alone -- `contin_reduce(cond(+($1 $2)), 2, {})` (an
already-single-child cond wrapping a compound `+`) turns into a bare,
CHILDLESS `+` node (tree size 1, not the `+($1 $2)` a "collapse the
redundant wrapper" reading would expect), and a later `complete_factorize`
pass hits that childless `+` and throws `combo_tree node should not be
childless` -- an OC_ASSERT in `reduce_factorize` (contin_rules.cc:294), a
DIFFERENT rule than the one that corrupted the tree. See `_zap_one`'s own
docstring below for the mechanism. Ported verbatim -- DO NOT FIX: the
correct-looking change (flatten `v`'s children up before erasing it, the
way `reduce_plus_times_one_child`'s own `tr.erase(tr.flatten(it_child))`
idiom does) would silently diverge this port from measured C++ behaviour.
"""

from __future__ import annotations

import copy

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, vertex_equal
from pymetta.comboreduct.reduct.logical_reduction import logical_reduce
from pymetta.comboreduct.reduct.reduct import rule
from pymetta.util.tree import tree_node


def _zap_one(tr: combo_tree, it: tree_node) -> None:
    """C++: moses/comboreduct/reduct/branch_rules.cc:47 zap_one (static, private)

    `cond(v) -> v`: `it` has exactly one child, `v`. C++: `pre_it top = it;
    *it = *it.begin(); tr.erase(++top);` -- `top` starts at `it`, THEN its
    data is overwritten with a copy of `v`'s (`*it = *it.begin()`, BEFORE
    the erase, so `top`/`it` still has its original one child at that
    point), and only then does `++top` (a pre-order step from `it`, which
    still has exactly one child, so this descends into `v`) land on `v`
    itself for `tr.erase` to remove.

    `tr.erase(v)` -- see this module's own docstring's measured-bug
    section -- destroys `v`'s ENTIRE subtree (`erase()` always calls
    `erase_children()` first, regardless of which iterator type
    instantiates the template; confirmed in $COGUTIL's
    `opencog/util/tree.h:805-825`), so any of `v`'s OWN children are
    silently discarded rather than promoted onto `it`. Reproduced
    verbatim: `it.data` becomes a copy of `v`'s vertex, then `v` (and
    everany descendants) is erased outright, leaving `it` with `v`'s old
    ROOT vertex but NONE of `v`'s old children.

    `copy.copy` on `it.data = copy.copy(child.data)` matches this port's
    uniform subtree-copy convention (`tree.py`'s own `_copy_value`) for
    the one mutable vertex alternative, `argument` -- `child` is about to
    be erased outright on the next line regardless, so this guards only
    against `it`'s new value being a shared object with anything ELSE
    still live in the tree (there is nothing else here, but the
    convention is applied uniformly rather than selectively skipped).
    """
    child = it.first_child
    it.data = copy.copy(child.data)
    tr.erase(child)


class reduce_cond_arg(rule):
    """C++: moses/comboreduct/reduct/branch_rules.h:33 reduce_cond_arg

    Apply boolean reduction to each predicate of a `cond` (same class
    structure as `reduce_impulse_arg`, contin_rules.py).
    """

    def __init__(self, effort: int, ignore_ops: set, name: str = "reduce_cond_arg") -> None:
        """C++: moses/comboreduct/reduct/branch_rules.h:37 reduce_cond_arg::reduce_cond_arg

        `ignore_ops` is stored by REFERENCE in C++ (`const vertex_set&`,
        branch_rules.h:35) -- the same non-owning-reference convention
        `contin_rules.py`'s `reduce_impulse_arg` uses, for the identical
        reason (the caller's `ignore_ops` set outlives every rule built
        from it in a `contin_reduction`/`mixed_reduction` call).
        """
        super().__init__(name)
        self.reduct_effort = effort
        self.ignore_ops = ignore_ops

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/branch_rules.cc:36 reduce_cond_arg::operator()

        `size_t nc = tr.number_of_children(it); for (sib_it sib =
        it.begin(); 3 <= nc; nc -= 2) { ...; ++sib; ++sib; }` -- the loop
        guard `3 <= nc` is checked BEFORE each `nc -= 2`, so `nc` is only
        ever decremented from a value already `>= 3` (down to `>= 1`),
        never wrapping past zero -- no `size_t`-wrap hazard here (S9's
        Critical 1, checked and cleared, not merely assumed).
        """
        if not vertex_equal(it.data, builtin.cond):
            return
        nc = tr.number_of_children(it)
        sib = it.first_child
        while 3 <= nc:
            logical_reduce(self.reduct_effort, tr, sib, self.ignore_ops)
            sib = sib.next_sibling
            sib = sib.next_sibling
            nc -= 2


class reduce_cond_else(rule):
    """C++: moses/comboreduct/reduct/branch_rules.h:47 reduce_cond_else

    If the last two consequents are identical, omit the last clause:
    `cond(p1 x1 ... pn xn p x x) -> cond(p1 x1 ... pn xn x)`;
    `cond(v) -> v`. Fires on both `cond` and `contin_if` (this obsoletes
    `reduce_contin_if_equal_branch`, per C++'s own comment).
    """

    def __init__(self, name: str = "reduce_cond_else") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/branch_rules.cc:59 reduce_cond_else::operator()

        `last -= 3` where `last` is C++ `size_t`: reachable only once
        `1 == last` has already been ruled out this same iteration, and a
        well-formed `cond`/`contin_if` always holds an ODD child count
        (pair-list plus one trailing else) -- every rule that ever
        shrinks one (this one, `reduce_cond_arg`, `reduce_cond_const`)
        removes children two at a time, or via `_zap_one` at exactly
        `last == 1`, so the invariant survives every real reduction path.
        `last == 2` (the only even value that would drive `last -= 3`
        negative) is therefore not reachable from a well-formed input;
        `tr.child` still raises `IndexError` on a negative index rather
        than reproducing C++'s own weaker, one-step-late bounds check
        (S9's Critical 1 policy -- checked, not assumed unreachable
        without checking).
        """
        if not (vertex_equal(it.data, builtin.cond) or vertex_equal(it.data, builtin.contin_if)):
            return

        while True:
            last = tr.number_of_children(it)
            if last == 1:
                _zap_one(tr, it)
                return
            last -= 3
            cond = tr.child(it, last)
            c1 = tr.child(it, last + 1)
            c2 = tr.child(it, last + 2)
            if tr.equal_subtree(c1, c2):
                tr.erase(cond)
                tr.erase(c2)
            else:
                break


class reduce_cond_adjacent(rule):
    """C++: moses/comboreduct/reduct/branch_rules.h:57 reduce_cond_adjacent

    If two neighbouring consequents are identical, merge them:
    `cond(... p x q x ...) -> cond(... or(p q) x ...)`. `cond`-only.
    """

    def __init__(self, name: str = "reduce_cond_adjacent") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/branch_rules.cc:93 reduce_cond_adjacent::operator()

        C++: `size_t sz = tr.number_of_children(it); sib_it sib =
        it.begin(); for (; 5 <= sz; sz -= 2) { sib_it p=sib, x1=++sib,
        q=++sib; sib_it x2=q; x2++; if (!equal_subtree(x1,x2)) continue;
        ...merge...; sib=x2; ++sib; sz -= 2; tr.erase(x2); }` -- TWO
        decrements per successful merge: the `for`-header's own automatic
        `sz -= 2` (reached by simply falling off the end of the body, the
        same path a `continue` also takes) PLUS one EXPLICIT `sz -= 2`
        written inline in the body before `tr.erase(x2)`
        (branch_rules.cc:118). Ported as a `while` with both kept exactly
        where C++ places them: the "not equal" path gets only the
        header's implicit decrement (spelled out here since Python's
        `while` has no automatic per-iteration step); the merge path gets
        that SAME implicit decrement (at the end, after `tr.erase`) plus
        the explicit one before it -- four total per merge, matching `it`
        losing exactly two direct children that iteration (`q` becomes a
        grandchild under the new `or`, `x2` is destroyed outright).
        Neither decrement can underflow: both only ever run once the loop
        guard has already confirmed `sz >= 5`.

        `sib`'s advance mirrors `++sib` used TWICE to walk `p -> x1 -> q`
        (`sib` itself, re-read after each step, rather than three
        independently-computed offsets) and `x2 = q; x2++` (a NEW
        position one step past `q`, not `sib`'s own further advance) --
        `sib` stays parked AT `q`'s position on the "not equal" path,
        exactly matching C++'s `continue` never touching `sib` again,
        so the NEXT iteration's `p` becomes THIS iteration's `q`.
        """
        if not vertex_equal(it.data, builtin.cond):
            return

        sz = tr.number_of_children(it)
        sib = it.first_child
        while 5 <= sz:
            p = sib
            sib = sib.next_sibling
            x1 = sib
            sib = sib.next_sibling
            q = sib
            x2 = q.next_sibling

            if not tr.equal_subtree(x1, x2):
                sz -= 2
                continue

            por = tr.insert_above(p, builtin.logical_or)
            tr.move_after(por.first_child, q)

            sib = x2.next_sibling
            sz -= 2

            tr.erase(x2)
            sz -= 2


class reduce_cond_const(rule):
    """C++: moses/comboreduct/reduct/branch_rules.h:71 reduce_cond_const

    Reduce `cond`/`contin_if` with constant predicates: a `true` predicate
    truncates the rest of the condition down to its own consequent
    (`cond(true x1 ... y) -> x1`); a `false` predicate drops the whole
    clause (`cond(false x1 ...pn xn y) -> cond(p2 x2 ...pn xn y)`). Also
    handles `cond(v) -> v` via `_zap_one` when this leaves exactly one
    child. Fires on both `cond` and `contin_if`. This partially obsoletes
    `reduce_contin_if` (C++'s own comment).
    """

    def __init__(self, name: str = "reduce_cond_const") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/branch_rules.cc:117 reduce_cond_const::operator()

        `sib` walks `p1, x1, p2, x2, ...` two steps at a time
        (`pred=sib; quent=++sib; ...; ++sib`) REGARDLESS of whether this
        pair turns out to be true/false/neither -- so on the next `while`
        pass, `pred` always starts at what would be the FOLLOWING
        predicate (or lands on the trailing else, at which point the
        `sib is None` check below -- `if (sib == it.end()) return;` in
        C++ -- stops the walk rather than misreading the else value as a
        predicate).

        # DIVERGENCE(C++ branch_rules.cc:139-140): C++'s trailing cleanup
        # loop (`for (; sib != it.end(); ++sib) { tr.erase(sib); num--; }`)
        # never reassigns `sib` from `tr.erase`'s own return value -- it
        # erases the CURRENT node, then reads `++sib` (`sib->next_sibling`)
        # off the iterator's OWN copy of the now-freed node pointer. Real
        # C++ relies on that read landing on still-intact (if
        # technically-freed) memory -- undefined behaviour that happens to
        # work in practice, not a documented guarantee. This port's
        # `tree.erase` POISONS the erased node's own `next_sibling` with a
        # `_DEAD` sentinel specifically so a stale handle fails LOUDLY
        # (`AttributeError`) rather than silently reading freed memory --
        # a literal transliteration would crash every time this branch
        # runs. The successor is captured BEFORE erasing instead
        # (`nxt = sib.next_sibling` then `tr.erase(sib)` then `sib = nxt`),
        # reproducing the code's plain observable intent (erase every
        # child from here through the end) rather than its
        # undefined-behaviour mechanism -- the two agree on every input
        # that does not itself depend on freed-memory contents, which none
        # of this rule's real callers do.
        """
        if not (vertex_equal(it.data, builtin.cond) or vertex_equal(it.data, builtin.contin_if)):
            return

        sib = it.first_child
        num = tr.number_of_children(it)
        while True:
            if num == 1:
                _zap_one(tr, it)
                return

            pred = sib
            sib = sib.next_sibling
            quent = sib
            if sib is None:
                return
            sib = sib.next_sibling

            if vertex_equal(pred.data, builtin.logical_true):
                tr.erase(pred)
                num -= 1
                while sib is not None:
                    nxt = sib.next_sibling
                    tr.erase(sib)
                    num -= 1
                    sib = nxt
                if num == 1:
                    _zap_one(tr, it)
                return

            if vertex_equal(pred.data, builtin.logical_false):
                tr.erase(pred)
                tr.erase(quent)
                num -= 2
