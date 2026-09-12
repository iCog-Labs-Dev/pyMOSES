"""C++: moses/comboreduct/reduct/mixed_rules.{h,cc}

S14 Task 4 -- the largest single file in this slice. Rules over trees that
mix logical and continuous vertices: `greater_than_zero` predicates,
`impulse`, `contin_if`, and the "very complex" assumption-based rules
that reduce an inequality/branch using OTHER inequalities already known
to hold (the tree's assumption set -- `combo/assumption.py`).

No `move_ontop` call anywhere in this file (checked directly: `grep -n
move_ontop mixed_rules.cc` is empty) -- CLAUDE.md's traced-and-cleared
`move_ontop` self-cycle hazard does not apply here at all.

Three generic tree operations this file needs that neither `tree.py` nor
this port's other reduct modules provide yet, all reproduced here as
private module-level helpers (not added to the shared `tree.py` -- see
`vertex.py`'s own `lexicographic_subtree_cmp` precedent for porting a
`opencog/util/tree.h` algorithm as an external function rather than a
`tree` method):

* `_tree_swap` -- `opencog/util/tree.h:2055,2076` `tree<T>::swap` (both
  overloads): exchange two nodes' POSITIONS (parent/sibling links), not
  their data. Two real call sites (`reduce_op_contin_if`,
  `reduce_contin_if_not`).
* `_find_subtree` -- `opencog/util/tree.h:2141` `tree<T>::find_subtree`:
  linear pre-order scan for a structurally-equal subtree within a bounded
  range. One real call site (`reduce_contin_if_substitute_cond`), which
  also needs `tree.py`'s own `subtree_pre_order_end` to correctly convert
  a `sibling_iterator`-shaped "end of this node's children" bound into
  the pre-order position that really means (see that rule's own
  docstring for the derivation -- a naive `None` bound would let the
  search spill past the branch it is meant to be confined to).
* `_with_copied_assumptions` -- the "make a disposable one-node standalone
  tree, wrap it, carry the ORIGINAL tree's assumption set along" idiom
  that EVERY assumption-aware rule below repeats
  (`reduce_from_assumptions`, `reduce_contin_if_not`, `reduce_gt_zero_sum`,
  `reduce_gt_zero_prod`, `reduce_gt_zero_div`, `reduce_gt_zero_sum_sin`,
  `reduce_gt_zero_sin`): C++'s `tr.subtree(sib, tr.next_sibling(sib))`
  (always a SINGLE-node sibling range in every real call site here) is
  `combo_tree(sib)` (tree.py's iterator-copy constructor), and the
  "copy old assumptions" loop that follows it every time is exactly
  `copy_all_assumptions` (`combo/assumption.py`, added by S14 Task 1 for
  this precise call pattern).

`may_have_side_effects` (`vertex.h:793`) is a real, disclosed C++ TODO
stub that always returns `False` -- ported as such, not a simplification
this port introduces.

MEASURED DEFAULT-BUILTIN QUIRK carried in from `contin_rules.py`: `*it =
FP_NAN`/`FP_INFINITE` write `0.0`/`1.0`, never an actual NaN/infinity (see
that module's own docstring) -- this file has no such assignments of its
own (`reduce_gt_zero_*` write `id::logical_true`/`id::logical_false`
instead), so the quirk does not recur here, but is noted for anyone
reading this file expecting the same pattern.
"""

from __future__ import annotations

import copy
import functools
import math

from pymoses.comboreduct.combo.assumption import copy_all_assumptions
from pymoses.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    combo_tree,
    contin_t,
    get_builtin,
    get_contin,
    is_argument,
    is_boolean,
    is_builtin,
    is_contin,
    lexicographic_subtree_cmp,
    vertex_equal,
    vertex_to_bool,
)
from pymoses.comboreduct.reduct.contin_rules import (
    _find_child_by_vertex,
    _LexicographicSubtreeMap,
    _LexicographicSubtreeSet,
)
from pymoses.comboreduct.reduct.reduct import rule
from pymoses.comboreduct.table.table import type_node
from pymoses.comboreduct.type_checker.type_tree import get_output_type_tree, type_tree
from pymoses.util.tree import tree_node

# ---------------------------------------------------------------------------
# Shared helpers -- see module docstring.
# ---------------------------------------------------------------------------


def may_have_side_effects(it: tree_node) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:793 may_have_side_effects

    Always False -- a real, disclosed C++ `//TODO` stub, not a
    simplification introduced by this port.
    """
    return False


def _tree_swap_adjacent(node: tree_node) -> None:
    """C++: opencog/util/tree.h:2055 tree<T>::swap(sibling_iterator)

    Swap `node` with its own next sibling, in place. No-op if `node` has
    no next sibling.
    """
    nxt = node.next_sibling
    if nxt is None:
        return
    if node.prev_sibling is not None:
        node.prev_sibling.next_sibling = nxt
    else:
        node.parent.first_child = nxt
    nxt.prev_sibling = node.prev_sibling
    nxtnxt = nxt.next_sibling
    if nxtnxt is not None:
        nxtnxt.prev_sibling = node
    else:
        node.parent.last_child = node
    nxt.next_sibling = node
    node.prev_sibling = nxt
    node.next_sibling = nxtnxt


def _tree_swap(one: tree_node, two: tree_node) -> None:
    """C++: opencog/util/tree.h:2076 tree<T>::swap(iterator, iterator)

    Exchange two nodes' POSITIONS (parent/prev/next-sibling links), not
    their data -- delegates to the single-node adjacent-sibling swap
    above when the two are already adjacent siblings in either order,
    exactly matching C++'s own dispatch.
    """
    if one is two:
        return
    if one.next_sibling is two:
        _tree_swap_adjacent(one)
        return
    if two.next_sibling is one:
        _tree_swap_adjacent(two)
        return

    nxt1, nxt2 = one.next_sibling, two.next_sibling
    pre1, pre2 = one.prev_sibling, two.prev_sibling
    par1, par2 = one.parent, two.parent

    one.parent = par2
    one.next_sibling = nxt2
    if nxt2 is not None:
        nxt2.prev_sibling = one
    else:
        par2.last_child = one
    one.prev_sibling = pre2
    if pre2 is not None:
        pre2.next_sibling = one
    else:
        par2.first_child = one

    two.parent = par1
    two.next_sibling = nxt1
    if nxt1 is not None:
        nxt1.prev_sibling = two
    else:
        par1.last_child = two
    two.prev_sibling = pre1
    if pre1 is not None:
        pre1.next_sibling = two
    else:
        par1.first_child = two


def _find_subtree(
    tr: combo_tree, target: tree_node, begin: tree_node | None, end: tree_node | None
) -> tree_node | None:
    """C++: opencog/util/tree.h:2141 tree<T>::find_subtree

    Linear pre-order scan from `begin` (inclusive) through `end`
    (exclusive), returning the first node whose subtree is
    `vertex_equal`-structurally equal to `target`'s -- `None` ("an
    invalid iterator" in C++) if none is found. `begin=None` is treated as
    an EMPTY range (this port's stand-in for a `sibling_iterator`'s
    end-of-children position used as a start bound -- e.g. a childless
    branch -- which real C++'s converting constructor resolves to a
    position already equal to any `end` bound derived from the SAME
    node, so the loop already runs zero times there either way).
    """
    if begin is None:
        return None
    i = begin
    while i is not end:
        if tr.equal_subtree(target, i, vertex_equal):
            return i
        i = tr.next_pre_order(i)
    return None


def _with_copied_assumptions(tr: combo_tree, node: tree_node) -> combo_tree:
    """No direct single C++ symbol -- see this module's own docstring.
    `combo_tree sib_tr = tr.subtree(sib, tr.next_sibling(sib));` (always a
    single-node range at every real call site in this file) plus the
    "copy old assumptions" loop that follows it every time.
    """
    sib_tr = combo_tree(node)
    copy_all_assumptions(sib_tr, tr)
    return sib_tr


def _copy_tree_with_assumptions(src: combo_tree) -> combo_tree:
    """No direct C++ symbol: reproduces `combo_tree dst = src;` -- a C++
    COPY-ASSIGNMENT/copy-construction of a WHOLE `combo_tree` that
    already carries an assumption set, which C++'s real value semantics
    copy along with the main tree (same head/feet node graph). This
    port's `combo_tree(node)` constructor (`tree.py`'s own iterator-copy
    constructor) copies only the given node's own subtree, never a
    separate `.assumptions` side list -- `meta_rules.py`'s
    `assum_iterative.apply` established this exact two-step workaround
    first (`combo_tree(tr.begin())` plus `copy_all_assumptions`), for the
    identical reason. Every `combo_tree trN = existing_tree;` in this
    file (as opposed to a fresh `tr.subtree(...)`-shaped copy straight
    off the MAIN tree, which `_with_copied_assumptions` above already
    handles) needs this, not the bare constructor.
    """
    dst = combo_tree(src.begin())
    copy_all_assumptions(dst, src)
    return dst


# ---------------------------------------------------------------------------
# Simple mixed rules (no recursive reduction argument)
# ---------------------------------------------------------------------------


class reduce_gt_zero_times_const(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:31 reduce_gt_zero_times_const

    `0<c*x -> 0<x` if `c>0`; `0<c*x -> 0<-1*x` if `c<0`. WARNING (C++'s own
    comment): deprecated, superseded by `reduce_gt_zero_prod` -- ported
    anyway since it is still a real, compiled symbol (used nowhere in
    `mixed_reduction.cc`'s own assembly, but declared and defined in
    `mixed_rules.{h,cc}` regardless -- CLAUDE.md's port-faithfully rule
    applies to unused-but-real code same as used code).
    """

    def __init__(self, name: str = "reduce_gt_zero_times_const") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:36 reduce_gt_zero_times_const::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_times_const)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.times):
            return
        sib = it_child.first_child
        while sib is not None:
            if is_contin(sib.data):
                c = get_contin(sib.data)
                if c > 0:
                    if tr.has_one_child(it_child):
                        tr.erase_children(it)
                        it.data = builtin.logical_true
                        return
                    sib = tr.erase(sib)
                elif c < 0:
                    if tr.has_one_child(it_child):
                        tr.erase_children(it)
                        it.data = builtin.logical_false
                        return
                    sib.data = -1.0
                    sib = sib.next_sibling
                else:  # c == 0
                    tr.erase_children(it)
                    it.data = builtin.logical_false
                    return
            else:
                sib = sib.next_sibling


class reduce_gt_zero_pair_power(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:37 reduce_gt_zero_pair_power

    `0<c*x*x -> false` if `c<0`, or more generally `0<c*x^p_x*y^p_y*...`
    is false when `c<0` and every exponent (parity of occurrence count
    within the product) is even.
    """

    def __init__(self, name: str = "reduce_gt_zero_pair_power") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:79 reduce_gt_zero_pair_power::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_pair_power)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.times):
            return

        is_neg_const = False
        sp = _LexicographicSubtreeMap()
        sib = it_child.first_child
        while sib is not None:
            if is_contin(sib.data):
                if get_contin(sib.data) < 0:
                    is_neg_const = not is_neg_const
            else:
                entry = sp.find(sib)
                if entry is None:
                    sp.insert(sib, False)
                else:
                    entry[1] = not entry[1]
            sib = sib.next_sibling

        if is_neg_const:
            if all(parity for _key, parity in sp):
                tr.erase_children(it)
                it.data = builtin.logical_false


class reduce_gt_zero_const_div(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:44 reduce_gt_zero_const_div

    `0<c/x -> 0<x` if `c>0`; `0<c/x -> 0<-1*x` if `c<0`. WARNING
    (deprecated, superseded by `reduce_gt_zero_div`) -- ported anyway,
    same reasoning as `reduce_gt_zero_times_const`.
    """

    def __init__(self, name: str = "reduce_gt_zero_const_div") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:128 reduce_gt_zero_const_div::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_const_div)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.div):
            return
        if tr.number_of_children(it_child) != 2:
            raise AssertionError(
                "combo_tree child node should have exactly two children (reduce_gt_zero_const_div)."
            )
        num = it_child.first_child
        if not is_contin(num.data):
            return
        c = get_contin(num.data)
        if c > 0:
            denom = it_child.last_child
            tr.move_before(it_child, denom)
            tr.erase(it_child)
        elif c < 0:
            num.data = -1.0
            it_child.data = builtin.times
        else:  # c == 0
            tr.erase_children(it)
            it.data = builtin.logical_false


class reduce_gt_zero_log(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:53 reduce_gt_zero_log

    `0<log(x) -> 0<-1+x`. Guarded in C++ by `#ifndef ABS_LOG`; `ABS_LOG`
    is commented out at `vertex.h:49`, so always compiled and always
    ported here (same reasoning as `contin_rules.py`'s `reduce_exp_log`).
    """

    def __init__(self, name: str = "reduce_gt_zero_log") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:161 reduce_gt_zero_log::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_log)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.log):
            return
        if not tr.has_one_child(it_child):
            raise AssertionError(
                "combo_tree child node should have exactly one child (reduce_gt_zero_log)."
            )
        it_child.data = builtin.plus
        tr.append_child(it_child, -1.0)


class reduce_gt_zero_exp(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:60 reduce_gt_zero_exp

    `0<exp(x) -> true`.
    """

    def __init__(self, name: str = "reduce_gt_zero_exp") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:178 reduce_gt_zero_exp::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_exp)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.exp):
            return
        if not tr.has_one_child(it_child):
            raise AssertionError(
                "combo_tree child node should have exactly one child (reduce_gt_zero_exp)."
            )
        tr.erase_children(it)
        it.data = builtin.logical_true


class reduce_gt_zero_minus_exp(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:67 reduce_gt_zero_minus_exp

    `0<-1*exp(x) -> false`. WARNING (deprecated, superseded by
    `reduce_gt_zero_prod`) -- ported anyway.
    """

    def __init__(self, name: str = "reduce_gt_zero_minus_exp") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:195 reduce_gt_zero_minus_exp::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_minus_exp)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.times):
            return
        is_neg_const = False
        sib = it_child.first_child
        while sib is not None:
            if is_contin(sib.data):
                if get_contin(sib.data) < 0:
                    is_neg_const = not is_neg_const
            elif not vertex_equal(sib.data, builtin.exp):
                return
            sib = sib.next_sibling
        if is_neg_const:
            tr.erase_children(it)
            it.data = builtin.logical_false


class reduce_gt_zero_prod_exp(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:75 reduce_gt_zero_prod_exp

    `0<y*exp(x) -> 0<y`. WARNING (deprecated, superseded by
    `reduce_gt_zero_prod`) -- ported anyway.
    """

    def __init__(self, name: str = "reduce_gt_zero_prod_exp") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:221 reduce_gt_zero_prod_exp::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError("combo_tree node should have exactly one child.")
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.times):
            return
        sib = it_child.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.exp):
                sib = tr.erase(sib)
            else:
                sib = sib.next_sibling
        if tr.is_childless(it_child):
            tr.erase_children(it)
            it.data = builtin.logical_true


class reduce_gt_zero_const_sum_sin(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:82 reduce_gt_zero_const_sum_sin

    `0<c+sin(x) -> true` if `c>1`; `0<c+sin(x) -> false` if `c<=1`.
    WARNING (deprecated, superseded by `reduce_gt_zero_sum_sin`) --
    ported anyway.
    """

    def __init__(self, name: str = "reduce_gt_zero_const_sum_sin") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:245 reduce_gt_zero_const_sum_sin"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_const_sum_sin)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.plus):
            return
        total = 0.0
        sin_count = 0
        sib = it_child.first_child
        while sib is not None:
            if is_contin(sib.data):
                total = total + get_contin(sib.data)
            elif vertex_equal(sib.data, builtin.sin):
                sin_count += 1
            else:
                return
            sib = sib.next_sibling
        if sin_count > 0:
            if total > contin_t(sin_count):
                tr.erase_children(it)
                it.data = builtin.logical_true
            elif total <= contin_t(-sin_count):
                tr.erase_children(it)
                it.data = builtin.logical_false


class reduce_gt_zero_impulse(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:90 reduce_gt_zero_impulse

    `0<impulse(x) -> x` (remember `x` is boolean).
    """

    def __init__(self, name: str = "reduce_gt_zero_impulse") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:276 reduce_gt_zero_impulse::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_impulse)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.impulse):
            return
        if not tr.has_one_child(it_child):
            raise AssertionError(
                "combo_tree child node should have exactly one child (reduce_gt_zero_impulse)."
            )
        new_it = it_child.first_child
        it.data = new_it.data
        tr.reparent(it, new_it)
        tr.erase(it_child)


class reduce_impulse_power(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:96 reduce_impulse_power

    `prod impulse(x_i)^p_i -> prod impulse(x_i)` -- duplicate `impulse`
    factors collapse to one (an `impulse` squared is still 0 or 1).
    """

    def __init__(self, name: str = "reduce_impulse_power") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:294 reduce_impulse_power::operator()"""
        if not vertex_equal(it.data, builtin.times):
            return
        sq = _LexicographicSubtreeSet()
        sib = it.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.impulse):
                if not tr.has_one_child(sib):
                    raise AssertionError(
                        "combo_tree sibling node should have exactly one child "
                        "(reduce_impulse_power)."
                    )
                if sq.contains(sib):
                    sib = tr.erase(sib)
                else:
                    sq.add(sib)
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling
        if tr.has_one_child(it):
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))


class reduce_impulse_prod(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:102 reduce_impulse_prod

    `prod impulse(x_i) * z -> impulse(and x_i) * z`.
    """

    def __init__(self, name: str = "reduce_impulse_prod") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:322 reduce_impulse_prod::operator()"""
        if not vertex_equal(it.data, builtin.times):
            return
        fic: tree_node | None = None
        new_and: tree_node | None = None
        sib = it.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.impulse):
                if not tr.has_one_child(sib):
                    raise AssertionError(
                        "combo_tree sibling node should have exactly one child "
                        "(reduce_impulse_prod)."
                    )
                if fic is not None:
                    if new_and is None:
                        new_and = tr.wrap(fic, builtin.logical_and)
                    tr.move_after(fic, sib.first_child)
                    sib = tr.erase(sib)
                else:
                    fic = sib.first_child
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling
        if tr.has_one_child(it):
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))


class reduce_impulse_sum(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:108 reduce_impulse_sum

    `0<(sum impulse(x_i)) -> or x_i`.
    """

    def __init__(self, name: str = "reduce_impulse_sum") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:352 reduce_impulse_sum::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree sibling node should have exactly one child (reduce_impulse_sum)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.plus):
            return
        sib = it_child.first_child
        while sib is not None:
            if not vertex_equal(sib.data, builtin.impulse):
                return
            sib = sib.next_sibling
        # we are now sure that all arguments of + are impulses
        it.data = builtin.logical_or
        tr.erase(tr.flatten(it_child))
        # C++: `for (sib_it sib = it.begin(); sib != it.end(); ++sib) sib
        # = tr.erase(tr.flatten(sib));` -- the SAME double-advance shape
        # `contin_rules.py`'s `reduce_times_div` docstring explains: the
        # body's own reassignment (to the just-promoted argument) is
        # followed by the for-loop's own automatic `++sib`, so each
        # already-flattened argument is visited once, never re-flattened.
        sib = it.first_child
        while sib is not None:
            sib = tr.erase(tr.flatten(sib))
            sib = sib.next_sibling if sib is not None else None


class reduce_contin_if_to_impulse(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:115 reduce_contin_if_to_impulse

    `if(x 1 0) -> impulse(x)`; `if(x 0 1) -> impulse(NOT(x))`.
    """

    def __init__(self, name: str = "reduce_contin_if_to_impulse") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:373 reduce_contin_if_to_impulse"""
        if not vertex_equal(it.data, builtin.contin_if):
            return
        if tr.number_of_children(it) != 3:
            raise AssertionError(
                "combo_tree node should have exactly three children (reduce_contin_if_to_impulse)."
            )
        cond = tr.child(it, 0)
        b1 = tr.child(it, 1)
        b2 = tr.child(it, 2)
        if is_contin(b1.data) and is_contin(b2.data):
            cb1 = get_contin(b1.data)
            cb2 = get_contin(b2.data)
            if cb1 == 1.0 and cb2 == 0.0:
                it.data = builtin.impulse
                tr.erase(b1)
                tr.erase(b2)
            elif cb1 == 0.0 and cb2 == 1.0:
                tr.wrap(cond, builtin.logical_not)
                it.data = builtin.impulse
                tr.erase(b1)
                tr.erase(b2)


class reduce_contin_if(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:122 reduce_contin_if

    `if(true x y) -> x`; `if(false x y) -> y`; `if(x if(x y z) w) -> if(x y
    w)`; `if(x y if(x z w)) -> if(x y w)`.
    """

    def __init__(self, name: str = "reduce_contin_if") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:403 reduce_contin_if::operator()"""
        if not vertex_equal(it.data, builtin.contin_if):
            return
        if tr.number_of_children(it) != 3:
            raise AssertionError(
                "combo_tree node should have exactly three children (reduce_contin_if)."
            )
        cond = tr.child(it, 0)
        b1 = tr.child(it, 1)
        b2 = tr.child(it, 2)
        if vertex_equal(cond.data, builtin.logical_true):
            it.data = b1.data
            tr.flatten(b1)
            tr.erase(cond)
            tr.erase(b1)
            tr.erase(b2)
        elif vertex_equal(cond.data, builtin.logical_false):
            it.data = b2.data
            tr.flatten(b2)
            tr.erase(cond)
            tr.erase(b1)
            tr.erase(b2)
        elif vertex_equal(b1.data, builtin.contin_if):
            if tr.number_of_children(b1) != 3:
                raise AssertionError(
                    "combo_tree node should have exactly three children (reduce_contin_if - b1)."
                )
            b1_cond = tr.child(b1, 0)
            b1_b1 = tr.child(b1, 1)
            if tr.equal_subtree(cond, b1_cond, vertex_equal):
                tr.move_before(b1, b1_b1)
                tr.erase(b1)
        elif vertex_equal(b2.data, builtin.contin_if):
            if tr.number_of_children(b2) != 3:
                raise AssertionError(
                    "combo_tree node should have exactly three children (reduce_contin_if - b2)."
                )
            b2_cond = tr.child(b2, 0)
            b2_b2 = tr.child(b2, 2)
            if tr.equal_subtree(cond, b2_cond, vertex_equal):
                tr.move_before(b2, b2_b2)
                tr.erase(b2)


class reduce_op_contin_if(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:129 reduce_op_contin_if

    `op(if(x y1 z1) if(x y2 z2)) -> if(x op(y1 y2) op(z1 z2))`.
    """

    def __init__(self, name: str = "reduce_op_contin_if") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:449 reduce_op_contin_if::operator()"""
        if vertex_equal(it.data, builtin.div):
            if tr.number_of_children(it) != 2:
                raise AssertionError(
                    "combo_tree node should have exactly two children (reduce_op_contin_if)."
                )
            num = tr.child(it, 0)
            denum = tr.child(it, 1)
            if vertex_equal(num.data, builtin.contin_if) and vertex_equal(
                denum.data, builtin.contin_if
            ):
                if tr.number_of_children(num) != 3:
                    raise AssertionError(
                        "combo_tree node should have exactly three children "
                        "(reduce_op_contin_if - num)."
                    )
                if tr.number_of_children(denum) != 3:
                    raise AssertionError(
                        "combo_tree node should have exactly three children "
                        "(reduce_op_contin_if - denum)."
                    )
                num_cond = tr.child(num, 0)
                denum_cond = tr.child(denum, 0)
                if tr.equal_subtree(num_cond, denum_cond, vertex_equal):
                    it.data = builtin.contin_if
                    tr.move_before(num, num_cond)
                    tr.erase(denum_cond)
                    num.data = builtin.div
                    denum.data = builtin.div
                    _tree_swap(tr.child(num, 1), tr.child(denum, 0))
            return

        if vertex_equal(it.data, builtin.plus) or vertex_equal(it.data, builtin.times):
            sq = _LexicographicSubtreeSet()
            sib = it.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.contin_if):
                    if tr.number_of_children(sib) != 3:
                        raise AssertionError(
                            "combo_tree sibling node should have exactly three children "
                            "(reduce_op_contin_if)."
                        )
                    sib_cond = tr.child(sib, 0)
                    match = sq.find(sib_cond)
                    if match is None:
                        sq.add(sib_cond)
                        sib = sib.next_sibling
                    else:
                        pivot_cond = match
                        pivot_b1 = pivot_cond.next_sibling
                        pivot_b2 = pivot_b1.next_sibling
                        tr.erase(tr.child(sib, 0))
                        it_builtin = get_builtin(it.data)
                        if vertex_equal(pivot_b1.data, it_builtin):
                            pivot_b1 = pivot_b1.first_child
                        else:
                            tr.wrap(pivot_b1, it.data)
                        tr.move_before(pivot_b1, tr.child(sib, 0))
                        if vertex_equal(pivot_b2.data, it_builtin):
                            pivot_b2 = pivot_b2.first_child
                        else:
                            tr.wrap(pivot_b2, it.data)
                        tr.move_before(pivot_b2, tr.child(sib, 0))
                        sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            if tr.has_one_child(it):
                it.data = it.first_child.data
                tr.erase(tr.flatten(it.first_child))


class reduce_contin_if_inner_op(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:136 reduce_contin_if_inner_op

    `contin_if(x op(y z) op(y w)) -> op(y contin_if(x z w))`, `op` in
    `{+, *, /}`; if `op` is `/` the argument order is respected.
    """

    def __init__(self, name: str = "reduce_contin_if_inner_op") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:517 reduce_contin_if_inner_op::operator()

        `sq.erase(sq_it)` (mixed_rules.cc:574) is the ITERATOR-based
        `std::multiset::erase` overload -- removes exactly the referenced
        element, unlike `contin_rules.py`'s `reduce_fraction`, which uses
        the KEY-based overload (removes every equal element). Ported here
        via `_LexicographicSubtreeSet.discard_one`, not `discard_all_
        equal` -- the two rules genuinely use different `std::multiset`
        member functions, not the same one applied inconsistently.
        """
        if not vertex_equal(it.data, builtin.contin_if):
            return
        if tr.number_of_children(it) != 3:
            raise AssertionError(
                "combo_tree node should have exactly three children (reduce_contin_if_inner_op)."
            )
        b1 = tr.child(it, 1)
        b2 = tr.child(it, 2)
        if not is_builtin(b1.data):
            return
        b1_builtin = get_builtin(b1.data)
        if not vertex_equal(b2.data, b1_builtin):
            return

        if vertex_equal(b1_builtin, builtin.div):
            if tr.number_of_children(b1) != 2:
                raise AssertionError(
                    "combo_tree node should have exactly two children "
                    "(reduce_contin_if_inner_op - b1)."
                )
            if tr.number_of_children(b2) != 2:
                raise AssertionError(
                    "combo_tree node should have exactly two children "
                    "(reduce_contin_if_inner_op - b2)."
                )
            b1_num = tr.child(b1, 0)
            b1_denum = tr.child(b1, 1)
            b2_num = tr.child(b2, 0)
            b2_denum = tr.child(b2, 1)
            if tr.equal_subtree(b1_num, b2_num, vertex_equal):
                it.data = builtin.div
                new_if = tr.prepend_child(it, builtin.contin_if)
                tr.reparent(new_if, it.first_child.next_sibling, None)
                tr.move_before(new_if, b1_num)
                tr.erase(b2_num)
                tr.erase(tr.flatten(b1))
                tr.erase(tr.flatten(b2))
            elif tr.equal_subtree(b1_denum, b2_denum, vertex_equal):
                it.data = builtin.div
                new_if = tr.prepend_child(it, builtin.contin_if)
                tr.reparent(new_if, it.first_child.next_sibling, None)
                tr.move_after(new_if, b1_denum)
                tr.erase(b2_denum)
                tr.erase(tr.flatten(b1))
                tr.erase(tr.flatten(b2))
            return

        if vertex_equal(b1_builtin, builtin.plus) or vertex_equal(b1_builtin, builtin.times):
            sq = _LexicographicSubtreeSet()
            new_if: tree_node | None = None
            sib = b1.first_child
            while sib is not None:
                sq.add(sib)
                sib = sib.next_sibling

            sib = b2.first_child
            while sib is not None:
                match = sq.find(sib)
                if match is not None:
                    if new_if is None:
                        it.data = b1_builtin
                        new_if = tr.prepend_child(it, builtin.contin_if)
                        tr.reparent(new_if, it.first_child.next_sibling, None)
                    tr.move_after(new_if, match)
                    sib = tr.erase(sib)
                    sq.discard_one(match)
                else:
                    sib = sib.next_sibling

            if tr.has_one_child(b1):
                tr.erase(tr.flatten(b1))
            elif tr.is_childless(b1):
                b1.data = 0.0 if vertex_equal(b1_builtin, builtin.plus) else 1.0
            if tr.has_one_child(b2):
                tr.erase(tr.flatten(b2))
            elif tr.is_childless(b2):
                b2.data = 0.0 if vertex_equal(b1_builtin, builtin.plus) else 1.0


class reduce_contin_if_substitute_cond(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:144 reduce_contin_if_substitute_cond

    `contin_if(x exp1[x] exp2[x]) -> if(x exp1[true] exp2[false])` --
    within the "then" branch, every occurrence of the condition itself is
    known to be true; within "else", known false.
    """

    def __init__(self, name: str = "reduce_contin_if_substitute_cond") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:594 reduce_contin_if_substitute_cond

        See this module's own docstring for `_find_subtree`'s bound
        conversions -- `pre_it(b2.begin())`/`pre_it(b2.end())` both need
        the `None`-means-empty-range / `subtree_pre_order_end` treatment
        that a literal `b2.first_child`/`None` pair would get wrong.
        """
        if not vertex_equal(it.data, builtin.contin_if):
            return
        if tr.number_of_children(it) != 3:
            raise AssertionError(
                "combo_tree node should have exactly three children "
                "(reduce_contin_if_substitute_cond)."
            )
        cond = tr.child(it, 0)
        b1 = tr.child(it, 1)
        b2 = tr.child(it, 2)
        if may_have_side_effects(cond):
            return
        if vertex_equal(cond.data, builtin.logical_true) or vertex_equal(
            cond.data, builtin.logical_false
        ):
            return

        cond_in_b1 = _find_subtree(tr, cond, b1.first_child, b2)
        while cond_in_b1 is not None:
            cond_in_b1.data = builtin.logical_true
            tr.erase_children(cond_in_b1)
            cond_in_b1 = _find_subtree(tr, cond, tr.next_pre_order(cond_in_b1), b2)

        b2_end = tr.subtree_pre_order_end(b2)
        cond_in_b2 = _find_subtree(tr, cond, b2.first_child, b2_end)
        while cond_in_b2 is not None:
            cond_in_b2.data = builtin.logical_false
            tr.erase_children(cond_in_b2)
            cond_in_b2 = _find_subtree(tr, cond, tr.next_pre_order(cond_in_b2), b2_end)


class reduce_junction_gt_zero_sum_constant(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:155 reduce_junction_gt_zero_sum_constant

    `0<x+c1 and 0<x+c2 -> 0<x+min(c1, c2)`, or more generally `0<sum x_i
    +c1 and 0<sum x_i +c2 and ... -> 0<sum x_i + min(c1, c2, ...)`; `or`
    is the symmetric `max` version. C++'s own comment: "no longer
    necessary due to the extension of reduce_from_assumptions to deal
    with inequalities" -- ported anyway, it is still wired into
    `mixed_reduction`'s real `non_recursive` sequence.
    """

    def __init__(self, name: str = "reduce_junction_gt_zero_sum_constant") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:637 reduce_junction_gt_zero_sum_constant"""
        if not (
            vertex_equal(it.data, builtin.logical_and) or vertex_equal(it.data, builtin.logical_or)
        ):
            return

        sq = _LexicographicSubtreeMap()
        sib = it.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.greater_than_zero):
                if not tr.has_one_child(sib):
                    raise AssertionError(
                        "combo_tree node should have exactly one child "
                        "(reduce_junction_gt_zero_const_sum_constante)."
                    )
                sib_child = sib.first_child
                if vertex_equal(sib_child.data, builtin.plus):
                    c = 0.0
                    plus_sib = sib_child.first_child
                    while plus_sib is not None:
                        if is_contin(plus_sib.data):
                            c = c + get_contin(plus_sib.data)
                            plus_sib = tr.erase(plus_sib)
                        else:
                            plus_sib = plus_sib.next_sibling
                    if tr.is_childless(sib_child):
                        sib_child.data = c
                        sib = sib.next_sibling
                    else:
                        entry = sq.find(sib_child)
                        if entry is None:
                            sq.insert(sib_child, c)
                            sib = sib.next_sibling
                        else:
                            c_ref = entry[1]
                            entry[1] = (
                                min(c, c_ref)
                                if vertex_equal(it.data, builtin.logical_and)
                                else max(c, c_ref)
                            )
                            sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling

        # add the right constant [that is min/max(c1, c2, ...)] to each
        # reduced inequality
        for key, val in sq:
            if val != 0.0:
                tr.append_child(key, val)


# ---------------------------------------------------------------------------
# "Very complex" recursive / assumption-based rules
# ---------------------------------------------------------------------------


class reduce_from_assumptions(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:178 reduce_from_assumptions

    Look up the assumptions and replace `it` by `true` if it is implied by
    one (or an assumption implies it), or `false` if `not(it)` is implied.
    The rule given at construction (`_reduction`) is used by `implies`
    and may be fully recursive.
    """

    def __init__(self, r: rule, name: str = "reduce_from_assumptions") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:181 reduce_from_assumptions

        `_reduction` is a raw, NON-OWNING reference (no `.clone()`) --
        same convention as every other `(const rule& r)`-constructed rule
        in this file and in `logical_rules.py`.
        """
        super().__init__(name)
        self._reduction = r

    def implies(self, tr: combo_tree, it1: tree_node, it2: tree_node) -> bool:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:697 reduce_from_assumptions::implies

        Tells whether `it1` implies `it2`: trivially true if they are
        syntactically equal; otherwise, if both are inequalities
        (`greater_than_zero`, possibly wrapped in `not` for the
        non-strict form), reduces each to a normalized `x [+ c]` shape
        and compares.
        """
        if tr.equal_subtree(it1, it2, vertex_equal):
            return True

        ine1_child: tree_node | None = None
        is_strict1 = True
        ine2_child: tree_node | None = None
        is_strict2 = True

        if vertex_equal(it1.data, builtin.greater_than_zero):
            if not tr.has_one_child(it1):
                raise AssertionError("combo_tree node should have exactly one child")
            ine1_child = it1.first_child
            is_strict1 = True
        elif vertex_equal(it1.data, builtin.logical_not):
            if not tr.has_one_child(it1):
                raise AssertionError("combo_tree node should have exactly one child")
            it1_child = it1.first_child
            if vertex_equal(it1_child.data, builtin.greater_than_zero):
                if not tr.has_one_child(it1_child):
                    raise AssertionError("combo_tree child node should have exactly one child")
                ine1_child = it1_child.first_child
                is_strict1 = False

        if vertex_equal(it2.data, builtin.greater_than_zero):
            if not tr.has_one_child(it2):
                raise AssertionError("combo_tree node should have exactly one child")
            ine2_child = it2.first_child
            is_strict2 = True
        elif vertex_equal(it2.data, builtin.logical_not):
            if not tr.has_one_child(it2):
                raise AssertionError("combo_tree node should have exactly one child")
            it2_child = it2.first_child
            if vertex_equal(it2_child.data, builtin.greater_than_zero):
                if not tr.has_one_child(it2_child):
                    raise AssertionError("combo_tree child node should have exactly one child")
                ine2_child = it2_child.first_child
                is_strict2 = False

        if ine1_child is None or ine2_child is None:
            return False

        c1 = 0.0
        c2 = 0.0
        tr1 = combo_tree(ine1_child)
        tr1_it = tr1.begin()
        tr2 = combo_tree(ine2_child)
        tr2_it = tr2.begin()

        if vertex_equal(tr1_it.data, builtin.plus):
            sib1 = tr1_it.first_child
            while sib1 is not None:
                if is_contin(sib1.data):
                    c1 += (1.0 if is_strict1 else -1.0) * get_contin(sib1.data)
                    sib1 = tr1.erase(sib1)
                else:
                    sib1 = sib1.next_sibling
        elif vertex_equal(tr1_it.data, builtin.times) and tr1.number_of_children(tr1_it) == 2:
            child0 = tr1.child(tr1_it, 0)
            child1 = tr1.child(tr1_it, 1)
            if is_contin(child0.data) and get_contin(child0.data) == -1.0:
                if vertex_equal(child1.data, builtin.plus):
                    s1p = child1.first_child
                    while s1p is not None:
                        if is_contin(s1p.data):
                            c1 += (-1.0 if is_strict1 else 1.0) * get_contin(s1p.data)
                            s1p = tr1.erase(s1p)
                        else:
                            s1p = s1p.next_sibling
            elif is_contin(child1.data) and get_contin(child1.data) == -1.0:
                if vertex_equal(child0.data, builtin.plus):
                    s1p = child0.first_child
                    while s1p is not None:
                        if is_contin(s1p.data):
                            c1 += (-1.0 if is_strict1 else 1.0) * get_contin(s1p.data)
                            s1p = tr1.erase(s1p)
                        else:
                            s1p = s1p.next_sibling
        if not is_strict1:
            tr1.append_child(tr1.wrap(tr1_it, builtin.times), -1.0)

        if vertex_equal(tr2_it.data, builtin.plus):
            sib2 = tr2_it.first_child
            while sib2 is not None:
                if is_contin(sib2.data):
                    c2 += (1.0 if is_strict2 else -1.0) * get_contin(sib2.data)
                    sib2 = tr2.erase(sib2)
                else:
                    sib2 = sib2.next_sibling
        elif vertex_equal(tr2_it.data, builtin.times) and tr2.number_of_children(tr2_it) == 2:
            child0 = tr2.child(tr2_it, 0)
            child1 = tr2.child(tr2_it, 1)
            if is_contin(child0.data) and get_contin(child0.data) == -1.0:
                if vertex_equal(child1.data, builtin.plus):
                    s2p = child1.first_child
                    while s2p is not None:
                        if is_contin(s2p.data):
                            c2 += (-1.0 if is_strict2 else 1.0) * get_contin(s2p.data)
                            s2p = tr2.erase(s2p)
                        else:
                            s2p = s2p.next_sibling
            elif is_contin(child1.data) and get_contin(child1.data) == -1.0:
                if vertex_equal(child0.data, builtin.plus):
                    s2p = child0.first_child
                    while s2p is not None:
                        if is_contin(s2p.data):
                            c2 += (-1.0 if is_strict2 else 1.0) * get_contin(s2p.data)
                            s2p = tr2.erase(s2p)
                        else:
                            s2p = s2p.next_sibling
        if not is_strict2:
            # C++ (mixed_rules.cc:867-868) calls `tr1.append_child(tr1.wrap(
            # tr2_it, ...), ...)` -- `tr1`'s own method table, applied to a
            # node (`tr2_it`) that belongs to `tr2`'s graph, not `tr1`'s.
            # Ported via `tr2`'s own methods instead: neither `append_child`
            # nor `wrap` reads anything from the receiving tree's own state
            # (both are pure node-pointer splices), so this has no
            # observable effect difference from the literal C++ call.
            tr2.append_child(tr2.wrap(tr2_it, builtin.times), -1.0)

        self._reduction(tr1)
        self._reduction(tr2)

        if tr1.equal_subtree(tr1.begin(), tr2.begin(), vertex_equal):
            if not is_strict1 and is_strict2:
                return c1 < c2
            return c1 <= c2
        return False

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:888 reduce_from_assumptions::operator()"""
        is_bool = get_output_type_tree(it.data) == type_tree(type_node.boolean_type)
        if not (
            (is_bool and not vertex_equal(it.data, builtin.logical_true)
             and not vertex_equal(it.data, builtin.logical_false))
            or is_argument(it.data)
        ):
            return

        for assumption in getattr(tr, "assumptions", []):
            assum = assumption.begin()
            if self.implies(tr, assum, it):
                it.data = builtin.logical_true
                tr.erase_children(it)
                return
            if vertex_equal(assum.data, builtin.logical_not):
                if not tr.has_one_child(assum):
                    raise AssertionError(
                        "combo_tree node should have exactly one child "
                        "(reduce_from_assumptions - assum)."
                    )
                if self.implies(tr, it, assum.first_child):
                    it.data = builtin.logical_false
                    tr.erase_children(it)
                    return
            elif is_argument(assum.data) and is_argument(it.data):
                assum_arg = copy.copy(assum.data)
                assum_arg.negate()
                # don't need to treat the case when same idx because
                # already treated in the beginning of the function
                if assum_arg.idx == it.data.idx:
                    it.data = builtin.logical_false
                    tr.erase_children(it)
                    return


class reduce_contin_if_not(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:196 reduce_contin_if_not

    `if(x y z) -> if(NOT(x) z y)` if `|rule(NOT(x))| < |rule(x)|` -- swap
    the branches when negating and reducing the condition makes it
    shorter.
    """

    def __init__(self, r: rule, name: str = "reduce_contin_if_not") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:197 reduce_contin_if_not"""
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:928 reduce_contin_if_not::operator()

        `cond_tr.wrap(cond_tr.begin(), id::logical_not)` replaces C++'s
        manual `prepend_child` + `reparent` dance (mixed_rules.cc:943-946:
        prepend a COPY of the root's own data as a new first child, then
        reparent the root's ORIGINAL children onto that copy, then
        overwrite the root's own data with `id::logical_not`) --
        `tree.wrap`/`insert_above` already performs exactly this
        "splice a new node in above the current root, root becomes its
        sole child" operation directly (well-defined even at a tree's own
        root -- `tree.py`'s own `insert_above` docstring), with no
        aliasing risk to guard against (the manual dance needs an
        explicit value-copy of the root's data to avoid sharing a mutable
        `argument` object between the original root and its own copy;
        `wrap` never duplicates any vertex value in the first place).
        Every later reader of `cond_tr` (`self._reduction`, `.size()`,
        `tr.replace`) only ever reads STRUCTURE/VALUE, never depends on
        which specific node object ended up as the tree's root, so this
        is a proven equivalent, not merely a plausible one.
        """
        if not vertex_equal(it.data, builtin.contin_if):
            return
        if tr.number_of_children(it) != 3:
            raise AssertionError(
                "combo_tree node should have exactly three children (reduce_contin_if_not)."
            )
        cond = tr.child(it, 0)
        b1 = tr.child(it, 1)
        b2 = tr.child(it, 2)

        cond_tr = _with_copied_assumptions(tr, cond)
        cond_tr.wrap(cond_tr.begin(), builtin.logical_not)

        self._reduction(cond_tr)
        self._reduction(tr, cond)

        if cond_tr.size() < tr.count(cond):
            tr.replace(cond, cond_tr.begin())
            _tree_swap(b1, b2)


class reduce_gt_zero_sum(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:203 reduce_gt_zero_sum

    `0<sum x_i -> true` if `0<x_i -> true` for all `i`; `0<sum x_i ->
    false` if `0<x_i -> false` for all `i`.
    """

    def __init__(self, r: rule, name: str = "reduce_gt_zero_sum") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:204 reduce_gt_zero_sum::reduce_gt_zero_sum"""
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:960 reduce_gt_zero_sum::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_sum)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.plus):
            return
        if tr.is_childless(it_child):
            raise AssertionError(
                "combo_tree child node should have at least one child (reduce_gt_zero_sum)."
            )

        res = True
        sib = it_child.first_child
        while sib is not None:
            sib_tr = _with_copied_assumptions(tr, sib)
            sib_tr.wrap(sib_tr.begin(), builtin.greater_than_zero)
            self._reduction(sib_tr)
            sib_tr_root = sib_tr.begin()
            if is_boolean(sib_tr_root.data):
                if tr.sibling_index(sib) == 0:
                    res = vertex_to_bool(sib_tr_root.data)
                elif res != vertex_to_bool(sib_tr_root.data):
                    return
            else:
                return
            sib = sib.next_sibling

        it.data = bool_to_vertex(res)
        tr.erase_children(it)


class reduce_gt_zero_prod(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:211 reduce_gt_zero_prod

    `0<prod x_i -> 0<prod x_i` with `x_i=1` if `0<x_i -> true`, `x_i=-1`
    if `0<-x_i -> true`; `0<prod x_i -> false` if some `x_i==0` (that is
    `0<x_i -> false` and `0<-1*x_i -> false`).
    """

    def __init__(
        self, r: rule, r2: rule | None = None, name: str = "reduce_gt_zero_prod"
    ) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:213 reduce_gt_zero_prod::reduce_gt_zero_prod

        Two C++ constructors collapsed into one Python signature (this
        port's usual convention): `reduce_gt_zero_prod(const rule& r)`
        (:213, `_complete_reduction = _reduction_without_itself = &r`)
        and `reduce_gt_zero_prod(const rule& r1, const rule& r2)` (:216,
        the two roles given separately -- the shape `mixed_reduction.cc`
        actually uses, `r` for the complete recursive reduction and
        `r_without_reduce_gt_zero_prod` to avoid infinite recursion when
        re-reducing a NEGATED single factor).
        """
        super().__init__(name)
        self._complete_reduction = r
        self._reduction_without_itself = r if r2 is None else r2

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:998 reduce_gt_zero_prod::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_prod)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.times):
            return

        sib = it_child.first_child
        while sib is not None:
            sib_tr = _with_copied_assumptions(tr, sib)
            sib_tr.wrap(sib_tr.begin(), builtin.greater_than_zero)

            tr1 = _copy_tree_with_assumptions(sib_tr)
            self._complete_reduction(tr1)
            tr1_root = tr1.begin()
            if vertex_equal(tr1_root.data, builtin.logical_true):
                sib = tr.erase(sib)
            else:
                tr2 = _copy_tree_with_assumptions(sib_tr)
                x_i = tr2.begin().first_child
                tr2.append_child(tr2.wrap(x_i, builtin.times), -1.0)
                # reduction without reduce_gt_zero_prod is used to avoid
                # infinite recursion
                self._reduction_without_itself(tr2)
                tr2_root = tr2.begin()
                if vertex_equal(tr2_root.data, builtin.logical_true):
                    sib.data = -1.0
                    tr.erase_children(sib)
                    sib = sib.next_sibling
                elif vertex_equal(tr1_root.data, builtin.logical_false) and vertex_equal(
                    tr2_root.data, builtin.logical_false
                ):
                    it.data = builtin.logical_false
                    tr.erase_children(it)
                    return
                else:
                    sib = sib.next_sibling

        if tr.is_childless(it_child):
            tr.erase_children(it)
            it.data = builtin.logical_true


class reduce_gt_zero_div(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:229 reduce_gt_zero_div

    `0<x/y -> 0<x` if `0<y -> true`; `0<x/y -> 0<y` if `0<x -> true`;
    `0<x/y -> 0<-1*y` if `0<-1*x -> true`; `0<x/y -> 0<-1*x` if `0<-1*y ->
    true`; `0<x/y -> false` if `x==0` (i.e. `not(0<x)` and `not(0<-x)`
    both hold).
    """

    def __init__(self, r: rule, name: str = "reduce_gt_zero_div") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:230 reduce_gt_zero_div::reduce_gt_zero_div"""
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1056 reduce_gt_zero_div::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_div)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.div):
            return
        if tr.number_of_children(it_child) != 2:
            raise AssertionError(
                "combo_tree child node should have exactly two children (reduce_gt_zero_div)."
            )
        num = tr.child(it_child, 0)
        denom = tr.child(it_child, 1)

        # case of the numerator
        num_tr = _with_copied_assumptions(tr, num)
        num_tr.wrap(num_tr.begin(), builtin.greater_than_zero)
        num_tr1 = _copy_tree_with_assumptions(num_tr)
        self._reduction(num_tr1)
        num_tr1_root = num_tr1.begin()
        if vertex_equal(num_tr1_root.data, builtin.logical_true):
            tr.erase(num)
            tr.erase(tr.flatten(it_child))
            return

        num_tr2 = _copy_tree_with_assumptions(num_tr)
        num_tr2.append_child(num_tr2.wrap(num_tr2.begin().first_child, builtin.times), -1.0)
        self._reduction(num_tr2)
        num_tr2_root = num_tr2.begin()
        if vertex_equal(num_tr2_root.data, builtin.logical_true):
            tr.erase_children(num)
            it_child.data = builtin.times
            num.data = -1.0
            return

        if vertex_equal(num_tr1_root.data, builtin.logical_false) and vertex_equal(
            num_tr2_root.data, builtin.logical_false
        ):
            it.data = builtin.logical_false
            tr.erase_children(it)
            return

        # case of the denominator
        denom_tr = _with_copied_assumptions(tr, denom)
        denom_tr.wrap(denom_tr.begin(), builtin.greater_than_zero)
        denom_tr1 = _copy_tree_with_assumptions(denom_tr)
        self._reduction(denom_tr1)
        denom_tr1_root = denom_tr1.begin()
        if vertex_equal(denom_tr1_root.data, builtin.logical_true):
            tr.erase(denom)
            tr.erase(tr.flatten(it_child))
            return

        denom_tr2 = _copy_tree_with_assumptions(denom_tr)
        denom_tr2.append_child(denom_tr2.wrap(denom_tr2.begin().first_child, builtin.times), -1.0)
        self._reduction(denom_tr2)
        denom_tr2_root = denom_tr2.begin()
        if vertex_equal(denom_tr2_root.data, builtin.logical_true):
            tr.erase_children(denom)
            it_child.data = builtin.times
            denom.data = -1.0


class reduce_gt_zero_sum_sin(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:239 reduce_gt_zero_sum_sin

    `0<x+sin(y) -> true` if `0<x-1 -> true`; `0<x+sin(y) -> false` if
    `0<x+1 -> false`.
    """

    def __init__(self, r: rule, name: str = "reduce_gt_zero_sum_sin") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:240 reduce_gt_zero_sum_sin"""
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1139 reduce_gt_zero_sum_sin::operator()"""
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_sum_sin)."
            )
        it_child = it.first_child
        if not (
            vertex_equal(it_child.data, builtin.plus)
            and tr.is_valid(_find_child_by_vertex(it_child, builtin.sin))
        ):
            return

        copy_tr = _with_copied_assumptions(tr, it)
        copy_plus = copy_tr.begin().first_child
        number_of_sin = 0
        sib = copy_plus.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.sin):
                number_of_sin += 1
                sib = copy_tr.erase(sib)
            else:
                sib = sib.next_sibling
        if tr.is_childless(copy_plus):
            return

        copy1_tr = _copy_tree_with_assumptions(copy_tr)
        copy1_plus = copy1_tr.begin().first_child
        copy1_tr.append_child(copy1_plus, contin_t(-number_of_sin))
        self._reduction(copy1_tr)
        if vertex_equal(copy1_tr.begin().data, builtin.logical_true):
            it.data = builtin.logical_true
            tr.erase_children(it)
            return

        copy2_tr = _copy_tree_with_assumptions(copy_tr)
        copy2_plus = copy2_tr.begin().first_child
        copy2_tr.append_child(copy2_plus, contin_t(number_of_sin))
        self._reduction(copy2_tr)
        if vertex_equal(copy2_tr.begin().data, builtin.logical_false):
            it.data = builtin.logical_false
            tr.erase_children(it)


class reduce_gt_zero_sin(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:246 reduce_gt_zero_sin

    `0<sin(y) -> true` if `0<y -> true` and `0<pi-y -> true`; `0<sin(y)
    -> false` if `0<y -> false` and `0<-(y+pi) -> false`.
    """

    def __init__(self, r: rule, name: str = "reduce_gt_zero_sin") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:247 reduce_gt_zero_sin::reduce_gt_zero_sin"""
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1189 reduce_gt_zero_sin::operator()

        C++'s `it_child.begin();` (mixed_rules.cc:1198) is a dead
        statement -- its result is computed and immediately discarded,
        with no assignment and no side effect. Omitted here; there is
        nothing for its absence to change.
        """
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_zero_sin)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.sin):
            return
        if not tr.has_one_child(it_child):
            raise AssertionError(
                "combo_tree child node should have exactly one child (reduce_gt_zero_sin)."
            )

        copy_tr = _with_copied_assumptions(tr, it)
        # turn copy_tr into 0<y and check if 0<y
        copy_tr.erase(copy_tr.flatten(copy_tr.begin().first_child))
        self._reduction(copy_tr)

        if vertex_equal(copy_tr.begin().data, builtin.logical_true):
            # check if 0<pi-y -> true
            copy1_tr = _with_copied_assumptions(tr, it)
            sin1_it = copy1_tr.begin().first_child
            sin1_it.data = builtin.times
            copy1_tr.append_child(sin1_it, -1.0)
            new_plus = copy1_tr.wrap(sin1_it, builtin.plus)
            copy1_tr.append_child(new_plus, contin_t(math.pi))
            self._reduction(copy1_tr)
            if vertex_equal(copy1_tr.begin().data, builtin.logical_true):
                it.data = builtin.logical_true
                tr.erase_children(it)
        elif vertex_equal(copy_tr.begin().data, builtin.logical_false):
            # check if 0<-(y+pi) -> false
            copy2_tr = _with_copied_assumptions(tr, it)
            sin2_it = copy2_tr.begin().first_child
            sin2_it.data = builtin.plus
            copy2_tr.append_child(sin2_it, contin_t(math.pi))
            copy2_tr.append_child(copy2_tr.wrap(sin2_it, builtin.times), -1.0)
            self._reduction(copy2_tr)
            if vertex_equal(copy2_tr.begin().data, builtin.logical_false):
                it.data = builtin.logical_false
                tr.erase_children(it)


class reduce_gt_division_of_constants(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:257 reduce_gt_division_of_constants

    `0<c*x+d -> 0<x+d/c` if `0<c -> true`; `0<c*x+d -> 0<-x+d/abs(c)` if
    `c<0 -> true`; `0<c*x+d -> true` if `c==0` and `0<d`; `0<c*x+d ->
    false` if `c==0` and `d<=0`.
    """

    def __init__(self, r: rule, name: str = "reduce_gt_division_of_constants") -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.h:258 reduce_gt_division_of_constants

        `_reduction` is stored (matching the constructor's own C++ shape,
        `_reduction(&r)`) but genuinely never read by `apply` below --
        checked directly against mixed_rules.cc:1253-1298, not assumed;
        the parameter/member exist in real, compiled C++ and are ported
        for structural fidelity even though nothing in this rule's own
        body ever calls through `_reduction`.
        """
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1253 reduce_gt_division_of_constants

        The `for (...;;) { if (is_contin(*sib)) {...} ++sib; }` shape
        (mixed_rules.cc:1280-1296) has an EMPTY for-header increment and
        an UNCONDITIONAL `++sib;` statement at the end of the loop BODY
        (not inside any branch) -- so the `c>0` branch's own `sib =
        tr.erase(sib)` is followed by that unconditional `++sib` too, a
        double advance (same shape as `contin_rules.py`'s
        `reduce_times_div`); the `c==0` branch instead `break`s, skipping
        it entirely; `c<0` and the non-contin case just fall through to
        the single unconditional step.
        """
        if not vertex_equal(it.data, builtin.greater_than_zero):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_gt_division_of_constants)."
            )
        it_child = it.first_child
        if not vertex_equal(it_child.data, builtin.plus) or tr.number_of_children(it_child) != 2:
            return

        left_argument = tr.child(it_child, 0)
        right_argument = tr.child(it_child, 1)
        if vertex_equal(left_argument.data, builtin.times) and is_contin(right_argument.data):
            contin_child = right_argument
            times_child = left_argument
        elif is_contin(left_argument.data) and vertex_equal(right_argument.data, builtin.times):
            contin_child = left_argument
            times_child = right_argument
        else:
            return

        sib = times_child.first_child
        while sib is not None:
            stop = False
            if is_contin(sib.data):
                c = get_contin(sib.data)
                if c < 0:
                    contin_child.data = get_contin(contin_child.data) / (-1 * c)
                    sib.data = -1.0
                elif c > 0:
                    contin_child.data = get_contin(contin_child.data) / c
                    sib = tr.erase(sib)
                else:  # c is zero
                    it.data = (
                        builtin.logical_true
                        if get_contin(contin_child.data) > 0
                        else builtin.logical_false
                    )
                    tr.erase_children(it)
                    stop = True
            if stop:
                break
            sib = sib.next_sibling if sib is not None else None


class reduce_inequality_from_assumptions(rule):
    """C++: moses/comboreduct/reduct/mixed_rules.h:265 reduce_inequality_from_assumptions

    Reduce an inequality from the assumptions: check whether the main
    term's linear decomposition, as a combination of the assumptions
    (each `greater_than_zero` or `not(greater_than_zero(...))`), forces
    it to a definite sign. Only `gaussianElimination2` and
    `backSubstitution` are actually called by `operator()`
    (`gaussJordanElimination`/`gaussianElimination` are declared, real,
    compiled methods with zero call sites within this class itself --
    checked directly, not assumed -- ported anyway for API completeness).
    """

    def __init__(self, name: str = "reduce_inequality_from_assumptions") -> None:
        super().__init__(name)

    def split_coef_expression(
        self, tr: combo_tree, it: tree_node
    ) -> tuple[contin_t, tree_node | None]:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1301 splitCoefExpression

        C++ takes `it` by REFERENCE (`combo_tree::iterator&`) and may
        reassign it to `tr.end()` ("to be recognized as not valid") --
        ported as a `(coef, expression_or_None)` return pair, since
        Python cannot mutate a caller's plain variable through a
        parameter the way a C++ reference can.
        """
        coef = 1.0
        if vertex_equal(it.data, builtin.times):
            sib = it.first_child
            while sib is not None:
                if is_contin(sib.data):
                    coef = coef * get_contin(sib.data)
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            if tr.has_one_child(it):
                it = tr.erase(tr.flatten(it))
            elif tr.is_childless(it):
                tr.erase(it)
                it = None
        elif is_contin(it.data):
            coef = get_contin(it.data)
            tr.erase(it)
            it = None
        return coef, it

    def gauss_jordan_elimination(self, dm: list[list[float]], eps: float = 1.0e-10) -> bool:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1327 gaussJordanElimination

        Zero call sites within this class's own `operator()` (see class
        docstring) -- ported for API completeness, not exercised by the
        real reduction pipeline.
        """
        h = len(dm)
        if not h > 0:
            raise AssertionError("Number of rows should be greater than 0 (gaussJordanElimitation)")
        w = len(dm[h - 1])
        for y in range(h):
            maxrow = y
            for y2 in range(y + 1, h):
                if abs(dm[y2][y]) > abs(dm[maxrow][y]):
                    maxrow = y2
            dm[y], dm[maxrow] = dm[maxrow], dm[y]
            if abs(dm[y][y]) <= eps:
                return False
            for y2 in range(y + 1, h):
                c = dm[y2][y] / dm[y][y]
                for x in range(y, w):
                    dm[y2][x] -= c * dm[y][x]
        for y in range(h - 1, -1, -1):
            c = dm[y][y]
            for y2 in range(y):
                for x in range(w - 1, y - 1, -1):
                    dm[y2][x] -= dm[y][x] * dm[y2][y] / c
            dm[y][y] /= c
            lo, hi = min(h, w), max(h, w)
            for x in range(lo, hi):
                dm[y][x] /= c
        return True

    def gaussian_elimination(self, dm: list[list[float]]) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1364 gaussianElimination

        Zero call sites within this class's own `operator()` (see class
        docstring) -- ported for API completeness, not exercised by the
        real reduction pipeline.
        """
        m = len(dm)
        if not m > 0:
            raise AssertionError("Number of rows should be greater than 0 (gaussianElimitation)")
        n = len(dm[m - 1])
        for i in range(m):
            for j in range(n):
                maxi = i
                for k in range(i + 1, m):
                    if abs(dm[k][j]) > abs(dm[maxi][j]):
                        maxi = k
                if dm[maxi][j] != 0.0:
                    dm[i], dm[maxi] = dm[maxi], dm[i]
                    c = dm[i][j]
                    if c == 0.0:
                        raise AssertionError(f"dm[{i}][{j}] == 0.0. Should be diferent")
                    for k in range(n):
                        dm[i][k] = dm[i][k] / c
                    for u in range(i + 1, m):
                        d = dm[u][j]
                        for k in range(n):
                            dm[u][k] = dm[u][k] - d * dm[i][k]

    def gaussian_elimination2(self, dm: list[list[float]], eps: float = 1.0e-10) -> bool:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1394 gaussianElimination2

        The variant `operator()` actually calls.
        """
        m = len(dm)
        if not m > 0:
            raise AssertionError("Number of rows should be greater than 0 (gaussianElimitation2)")
        n = len(dm[m - 1])
        if not n > 0:
            raise AssertionError(
                "Number of columns should be greater than 0 (gaussianElimitation2)"
            )
        for i in range(min(m, n - 1)):
            ipivot = -1
            for i2 in range(i, m):
                if abs(dm[i2][i]) > eps:
                    ipivot = i2
            if ipivot == -1:
                return False
            dm[i], dm[ipivot] = dm[ipivot], dm[i]
            c = dm[i][i]
            for j in range(n):
                dm[i][j] /= c
            for i2 in range(i + 1, m):
                c2 = dm[i2][i]
                for j in range(n):
                    dm[i2][j] -= c2 * dm[i][j]
        return True

    def back_substitution(
        self, dm: list[list[float]], eps: float = 1.0e-10
    ) -> tuple[bool, list[float]]:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1426 backSubstitution

        C++ takes `solution` as an out-parameter (`std::vector<double>&`,
        asserted empty on entry) -- ported as a `(ok, solution)` return
        pair.
        """
        m = len(dm)
        if not m > 0:
            raise AssertionError("Number of rows should be greater than 0 (backSubstitution)")
        n = len(dm[m - 1])
        if not n > 0:
            raise AssertionError("Number of rows should be greater than 0 (backSubstitution)")
        solution = [0.0] * (n - 1)
        all_null = True
        for i in range(m - 1, -1, -1):
            for j in range(n - 1):
                if abs(dm[i][j]) > eps:
                    all_null = False
            if all_null and abs(dm[i][n - 1]) > eps:
                return False, solution
            for j in range(n - 1):
                if dm[i][j] == 1.0:
                    solution[j] = dm[i][n - 1]
                    for i2 in range(i - 1, -1, -1):
                        dm[i2][n - 1] -= solution[j] * dm[i2][j]
                        dm[i2][j] = 0
        return True, solution

    @staticmethod
    def _record_linear_term(
        expression: tree_node | None,
        coef: contin_t,
        sr: _LexicographicSubtreeMap,
        free_coefs: list[float],
        column: int,
        total_columns: int,
    ) -> None:
        """No direct C++ symbol: the shared "add or update (expression,
        coef) in the map, else fold into free_coefs" idiom
        `operator()`'s own body repeats three times (mixed_rules.cc:
        1558-1574, 1589-1599, 1661-1679) -- one function here in place of
        three copies.
        """
        if expression is not None:
            entry = sr.find(expression)
            if entry is None:
                row = [0.0] * total_columns
                row[column] = coef
                sr.insert(expression, row)
            else:
                entry[1][column] = entry[1][column] + coef
        else:
            free_coefs[column] = free_coefs[column] + coef

    def _accumulate_plus_terms(
        self,
        ctr: combo_tree,
        plus_it: tree_node,
        coef_multiplier: contin_t,
        sr: _LexicographicSubtreeMap,
        free_coefs: list[float],
        column: int,
        total_columns: int,
    ) -> None:
        """No direct C++ symbol: the shared "walk a `+` node's children,
        split each into (coef, expression), record at `column`" idiom
        `operator()`'s own body repeats twice (mixed_rules.cc:1552-1584,
        1624-1649) -- once for the main term, once per assumption.
        """
        sib = plus_it.first_child
        while sib is not None:
            expression = sib
            next_sib = sib.next_sibling
            coef, expression = self.split_coef_expression(ctr, expression)
            coef = coef * coef_multiplier
            sib = next_sib
            self._record_linear_term(expression, coef, sr, free_coefs, column, total_columns)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/mixed_rules.cc:1455 reduce_inequality_from_assumptions

        `combo_tree ctr = tr;` is a C++ value-copy of the WHOLE tree
        (main tree plus its assumption set) -- `_copy_tree_with_
        assumptions(tr)`. `term_it = ctr.begin().begin()` (or `...
        .begin().begin()` for the `not(...)` case) navigates from `ctr`'s
        OWN ROOT, not from any position corresponding to `it` specifically
        -- this only reaches the right node when `it` IS `tr`'s own root,
        which every real caller (`mixed_reduction.cc`'s `downwards(reduce_
        inequality_from_assumptions())`, applied over a whole predicate
        tree) satisfies for the outermost `greater_than_zero`/`not(...)`
        node, but is a real, disclosed reading of the C++ shape, not a
        verified-by-probe claim about every possible nesting.

        `sr`'s rows are extracted for the final matrix in `std::map`'s
        own COMPARATOR-SORTED order (mixed_rules.cc:1684, `for (sci =
        sr.begin(); ...)`), NOT insertion order -- `_LexicographicSubtreeMap`'s
        own class docstring notes that no OTHER caller in this port
        depends on that distinction, but THIS one does:
        `gaussianElimination2`'s pivot search (mixed_rules.cc:1403-1405)
        keeps overwriting `ipivot` with the LAST row (from the current
        row downward) whose column value exceeds `eps`, not the first or
        the largest, so row order can change which row becomes the pivot
        -- sorted here via `lexicographic_subtree_cmp` before extraction,
        not left in insertion order.
        """
        term_strict_positive = True
        ctr = _copy_tree_with_assumptions(tr)
        term_it: tree_node | None = None

        if vertex_equal(it.data, builtin.greater_than_zero):
            if not tr.has_one_child(it):
                raise AssertionError(
                    "combo_tree node should have exactly one child "
                    "(reduce_inequality_from_assumptions)."
                )
            term_it = ctr.begin().first_child
            term_strict_positive = True
        elif vertex_equal(it.data, builtin.logical_not):
            if not tr.has_one_child(it):
                raise AssertionError(
                    "combo_tree node should have exactly one child "
                    "(reduce_inequality_from_assumptions)."
                )
            it_child = it.first_child
            if vertex_equal(it_child.data, builtin.greater_than_zero):
                if not tr.has_one_child(it_child):
                    raise AssertionError(
                        "combo_tree child node should have exactly one child "
                        "(reduce_inequality_from_assumptions)."
                    )
                term_it = ctr.begin().first_child.first_child
                term_strict_positive = False

        if term_it is None or not ctr.is_valid(term_it):
            return

        ctr_assumptions = getattr(ctr, "assumptions", [])
        if not ctr_assumptions:
            return

        assumptions: list[tree_node] = []
        is_strict_positives: list[bool] = []
        for assumption_tree in ctr_assumptions:
            assum = assumption_tree.begin()
            if vertex_equal(assum.data, builtin.greater_than_zero):
                assumptions.append(assum)
                is_strict_positives.append(True)
            elif vertex_equal(assum.data, builtin.logical_not):
                if not ctr.has_one_child(assum):
                    raise AssertionError(
                        "combo_tree node should have exactly one child "
                        "(reduce_inequality_from_assumptions - assum)."
                    )
                assum_child = assum.first_child
                if vertex_equal(assum_child.data, builtin.greater_than_zero):
                    assumptions.append(assum_child)
                    is_strict_positives.append(False)
        # the last entry is true because a constant is considered
        # strictly positive as a term 1
        is_strict_positives.append(True)

        if not assumptions:
            return

        num_assumptions = len(assumptions)
        total_columns = num_assumptions + 2
        sr = _LexicographicSubtreeMap()
        free_coefs = [0.0] * total_columns
        free_coefs[num_assumptions] = 1.0

        # this reduction only handles linear combinations -- the case
        # when term_it is `*` is not treated in general, only the
        # -1*(sum) shape (a positive constant factor would already have
        # been eliminated by earlier reduction rules).
        main_coef = 1.0
        if vertex_equal(term_it.data, builtin.times) and ctr.number_of_children(term_it) == 2:
            child0 = ctr.child(term_it, 0)
            child1 = ctr.child(term_it, 1)
            if is_contin(child0.data) and get_contin(child0.data) == -1.0:
                term_it = child1
                main_coef = -1.0
            elif is_contin(child1.data) and get_contin(child1.data) == -1.0:
                term_it = child0
                main_coef = -1.0

        if vertex_equal(term_it.data, builtin.plus):
            self._accumulate_plus_terms(
                ctr, term_it, main_coef, sr, free_coefs, num_assumptions + 1, total_columns
            )
        else:
            expression = term_it
            coef, expression = self.split_coef_expression(ctr, expression)
            coef = coef * main_coef
            self._record_linear_term(
                expression, coef, sr, free_coefs, num_assumptions + 1, total_columns
            )

        for i, assum in enumerate(assumptions):
            if not (
                vertex_equal(assum.data, builtin.greater_than_zero) and ctr.has_one_child(assum)
            ):
                raise AssertionError(
                    "Assum diferent from greater_than_zero or has none or more than one child."
                )
            assum_child = assum.first_child
            assum_coef = 1.0
            if (
                vertex_equal(assum_child.data, builtin.times)
                and ctr.number_of_children(assum_child) == 2
            ):
                child0 = ctr.child(assum_child, 0)
                child1 = ctr.child(assum_child, 1)
                if is_contin(child0.data) and get_contin(child0.data) == -1.0:
                    assum_child = child1
                    assum_coef = -1.0
                elif is_contin(child1.data) and get_contin(child1.data) == -1.0:
                    assum_child = child0
                    assum_coef = -1.0

            if vertex_equal(assum_child.data, builtin.plus):
                self._accumulate_plus_terms(
                    ctr, assum_child, assum_coef, sr, free_coefs, i, total_columns
                )
            else:
                expression = assum_child
                coef, expression = self.split_coef_expression(ctr, expression)
                coef = coef * assum_coef
                self._record_linear_term(expression, coef, sr, free_coefs, i, total_columns)

        # generate matrix to be solved -- sr's rows in std::map's own
        # comparator-sorted order (see this method's own docstring)
        ordered_entries = sorted(
            sr, key=functools.cmp_to_key(lambda a, b: -lexicographic_subtree_cmp(a[0], b[0]))
        )
        mat = [row for _key, row in ordered_entries]
        mat.append(free_coefs)

        if not self.gaussian_elimination2(mat):
            return

        bs_ok, solution = self.back_substitution(mat)
        if not bs_ok:
            return

        exists_strict_positive = False
        exists_strict_from_strict_positive = False
        all_positive_or_null = True
        for i, sol in enumerate(solution):
            if is_strict_positives[i]:
                if sol > 0:
                    exists_strict_positive = True
                    exists_strict_from_strict_positive = True
                elif sol < 0:
                    all_positive_or_null = False
            else:
                if sol < 0:
                    exists_strict_positive = True
                elif sol > 0:
                    all_positive_or_null = False

        if all_positive_or_null and exists_strict_from_strict_positive:
            it.data = builtin.logical_true if term_strict_positive else builtin.logical_false
            tr.erase_children(it)
        elif not exists_strict_positive:
            it.data = builtin.logical_false if term_strict_positive else builtin.logical_true
            tr.erase_children(it)
