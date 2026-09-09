"""C++: moses/comboreduct/reduct/contin_rules.{h,cc}

S14 Task 4. Sixteen rules over `+`/`*`/`/`/`log`/`exp`/`sin`/`impulse`
continuous-arithmetic subtrees. `contin_t` is `double` -- see this
module's own arithmetic below for where that matters (never narrowed to
`float`, unlike `score_t`).

TWO REAL UPSTREAM QUIRKS, ported verbatim, DO NOT FIX -- both measured
directly against the compiled `libcomboreduct.so`, not inferred from
reading the source alone:

1. `reduce_plus_zero`/`reduce_times_one_zero`/`reduce_invert_constant`/
   `reduce_log_div_times` assign a childless node's vertex to `FP_NAN` or
   `FP_INFINITE` on overflow/divide-by-zero/log(0). Those are NOT the
   floating-point special values `NAN`/`INFINITY` from `<cmath>` -- they
   are `<math.h>`'s `fpclassify()` CATEGORY TAGS (`FP_NAN`, `FP_INFINITE`,
   ... small ints used with `std::fpclassify`), which on this toolchain
   are `0` and `1` respectively (measured: a tiny C probe printing
   `FP_NAN`/`FP_INFINITE` from `<math.h>` prints `0`/`1`). Both compile
   only because they implicitly convert to `contin_t` -- `*it = FP_NAN;`
   assigns the vertex to `0.0`, and `*it = FP_INFINITE;` assigns it to
   `1.0`, NOT to an actual NaN or an actual infinity. Reproduced as bare
   `0.0`/`1.0` literals below, never `math.nan`/`math.inf` -- "fixing"
   this to the presumably-intended `NAN`/`INFINITY` would silently
   diverge this port from what classic's own binary actually computes.
2. `branch_rules.py`'s `_zap_one` (ported in the previous commit) can
   leave a genuinely CHILDLESS `+`/`*` node behind it (a real upstream
   bug of its own); `reduce_factorize`/`reduce_factorize_fraction` both
   `OC_ASSERT(!it.is_childless(), ...)` on exactly that shape and throw.
   Ported as a raised `AssertionError`, matching `OC_ASSERT`'s own
   release-build behaviour (this port's uniform convention).

Two `std::map<pre_it, V, lexicographic_subtree_order<vertex>>` (and, in
`mixed_rules.py`, `std::set`/`std::multiset` of the same shape) are used
throughout the factorisation rules -- `_LexicographicSubtreeMap`/
`_LexicographicSubtreeSet` below reproduce their FIND/INSERT semantics
(two keys are the SAME entry whenever neither's subtree sorts before the
other's, i.e. `lexicographic_subtree_cmp(a, b) == 0`, regardless of
Python object identity) via a small linear-scan list rather than a
balanced tree. Every real caller here reduces small, knob-building-scale
trees (CLAUDE.md's own hot-spot hazard note), and none of these
algorithms ever depends on `std::map`'s SORTED iteration order for its
final tree-mutation result -- only on find-or-insert-by-structural-
equality, which a linear scan reproduces exactly, just less efficiently
than a real balanced tree would.
"""

from __future__ import annotations

import math

from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    contin_t,
    get_contin,
    is_contin,
    lexicographic_subtree_cmp,
    vertex_equal,
)
from pymetta.comboreduct.reduct.logical_reduction import logical_reduce
from pymetta.comboreduct.reduct.reduct import replace_without_changing_it, rule
from pymetta.util.tree import tree_node


class _LexicographicSubtreeMap:
    """No direct C++ symbol -- see this module's own docstring. A
    find-or-insert-by-structural-equality map, keyed by a `tree_node`
    subtree via `lexicographic_subtree_cmp(a, b) == 0`. `.find(node)`
    returns the stored `[key_node, value]` pair (mutable in place --
    Python list, not tuple, so callers can update `.value` the way C++
    mutates `spit->second` through a reference) or `None`.
    """

    def __init__(self) -> None:
        self._entries: list[list] = []

    def find(self, node: tree_node) -> list | None:
        for entry in self._entries:
            if lexicographic_subtree_cmp(entry[0], node) == 0:
                return entry
        return None

    def insert(self, node: tree_node, value) -> list:
        entry = [node, value]
        self._entries.append(entry)
        return entry

    def __iter__(self):
        return iter(self._entries)


class _LexicographicSubtreeSet:
    """No direct C++ symbol -- the `std::set`/`std::multiset` twin of
    `_LexicographicSubtreeMap` above, for the mixed_rules.py callers that
    need membership/erase-one rather than a keyed value. A `multiset`
    caller simply never checks `.contains` before `.add`; `.discard_one`
    removes exactly one structurally-equal element (an `iterator`-based
    `erase`, not `erase(key)`, which for a `multiset` would remove every
    match) -- both share this one implementation since a plain Python
    list already tolerates duplicate entries.
    """

    def __init__(self) -> None:
        self._items: list[tree_node] = []

    def contains(self, node: tree_node) -> bool:
        return any(lexicographic_subtree_cmp(x, node) == 0 for x in self._items)

    def find(self, node: tree_node) -> tree_node | None:
        for x in self._items:
            if lexicographic_subtree_cmp(x, node) == 0:
                return x
        return None

    def add(self, node: tree_node) -> None:
        self._items.append(node)

    def __len__(self) -> int:
        return len(self._items)

    def discard_one(self, node: tree_node) -> tree_node | None:
        for i, x in enumerate(self._items):
            if lexicographic_subtree_cmp(x, node) == 0:
                del self._items[i]
                return x
        return None

    def discard_all_equal(self, node: tree_node) -> None:
        """`std::multiset::erase(const key_type&)` -- removes EVERY
        element considered equal to `node` by the comparator, unlike
        `discard_one`'s single-element `erase(iterator)`. This is the
        overload `reduce_fraction`'s real, measured bug depends on -- see
        that class's own docstring.
        """
        self._items = [x for x in self._items if lexicographic_subtree_cmp(x, node) != 0]


class reduce_plus_zero(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:30 reduce_plus_zero

    `x+0 -> x`; a non-finite child collapses the whole sum to NaN
    (`x+NAN -> NAN`, per the class's own C++ comment -- but see this
    module's own docstring: the vertex actually written is `0.0`, not a
    real NaN, matching `*it = FP_NAN`'s real meaning).
    """

    def __init__(self, name: str = "reduce_plus_zero") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:37 reduce_plus_zero::operator()"""
        if not vertex_equal(it.data, builtin.plus):
            return
        sib = it.first_child
        while sib is not None:
            if is_contin(sib.data):
                c = get_contin(sib.data)
                if c == 0.0:
                    sib = tr.erase(sib)
                elif not math.isfinite(c):
                    tr.erase_children(it)
                    it.data = 0.0  # DO NOT FIX -- FP_NAN is really 0.0, see module docstring
                    return
                else:
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling
        if tr.is_childless(it):
            it.data = 0.0


class reduce_times_one_zero(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:34 reduce_times_one_zero

    `x*1 -> x`; `x*0 -> 0`; a non-finite child collapses the whole
    product (per this module's own docstring: to `1.0`, not a real
    infinity -- `*it = FP_INFINITE`'s real meaning).
    """

    def __init__(self, name: str = "reduce_times_one_zero") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:64 reduce_times_one_zero::operator()"""
        if not vertex_equal(it.data, builtin.times):
            return
        sib = it.first_child
        while sib is not None:
            if is_contin(sib.data):
                c = get_contin(sib.data)
                if c == 1.0:
                    sib = tr.erase(sib)
                elif c == 0.0:
                    tr.erase_children(it)
                    it.data = 0.0
                    return
                elif not math.isfinite(c):
                    tr.erase_children(it)
                    it.data = 1.0  # DO NOT FIX -- `*it = FP_INFINITE` is really 1.0, see docstring
                    return
                else:
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling
        if vertex_equal(it.data, builtin.times) and tr.is_childless(it):
            it.data = 1.0


class reduce_factorize_fraction(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:41 reduce_factorize_fraction

    `x/z+y/z -> (x+y)/z`, or more generally `sum 1/z*x_i + sum y_j ->
    1/z*(sum x_i) + sum y_j` -- unify a common denominator FACTOR shared
    by two or more `div` terms under a `+`. When several choices are
    possible, picks the one that shortens the expression the most; ties
    go to whichever was found first while scanning left to right.
    """

    def __init__(self, name: str = "reduce_factorize_fraction") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:97 reduce_factorize_fraction::operator()

        `es = spit->second` / `es_vec = es.second` are C++ VALUE COPIES
        (plain `equiv_subtrees`/`pre_it_vector`, not references) taken
        right before the factorize step begins -- ported as ordinary
        Python list/tuple reads, which already behave as independent
        snapshots for this rule's purposes (`es_vec` is only ever
        iterated afterward, never mutated by the mutation loop touching
        the SAME map entry it came from, since the map itself is a local
        that goes out of scope with this call).
        """
        if not vertex_equal(it.data, builtin.plus):
            return

        if tr.is_childless(it):
            raise AssertionError(
                "combo_tree node should not be childless (reduce_factorize_fraction)."
            )

        sp = _LexicographicSubtreeMap()
        chosen_factor: tree_node | None = None
        reduced_size = 0

        plus_child = it.first_child
        while plus_child is not None:
            if vertex_equal(plus_child.data, builtin.div):
                if tr.number_of_children(plus_child) != 2:
                    raise AssertionError(
                        "combo_tree child node should have exactly two children "
                        "(reduce_factorize_fraction)."
                    )
                denom = plus_child.last_child
                if vertex_equal(denom.data, builtin.times):
                    times_child = denom.first_child
                    while times_child is not None:
                        pf = times_child
                        if is_contin(pf.data) and get_contin(pf.data) == 1.0:
                            times_child = times_child.next_sibling
                            continue
                        entry = sp.find(pf)
                        if entry is None:
                            sp.insert(pf, (tr.count(pf), []))
                        else:
                            key = entry[0]
                            es_subtree_size, es_vec = entry[1]
                            last_it = key if not es_vec else es_vec[-1]
                            last_it_parent = last_it.parent
                            if last_it_parent is None:
                                raise AssertionError(
                                    "last combo_tree node invalid "
                                    "(reduce_factorize_fraction)."
                                )
                            if last_it_parent is not denom:
                                es_vec.append(pf)
                                prs = len(es_vec) * es_subtree_size
                                if prs > reduced_size:
                                    reduced_size = prs
                                    chosen_factor = pf
                        times_child = times_child.next_sibling
                else:
                    pf = denom
                    if is_contin(pf.data) and get_contin(pf.data) == 1.0:
                        break
                    entry = sp.find(pf)
                    if entry is None:
                        sp.insert(pf, (tr.count(pf), []))
                    else:
                        es_subtree_size, es_vec = entry[1]
                        es_vec.append(pf)
                        prs = len(es_vec) * es_subtree_size
                        if prs > reduced_size:
                            reduced_size = prs
                            chosen_factor = pf
            plus_child = plus_child.next_sibling

        if chosen_factor is None:
            return
        entry = sp.find(chosen_factor)
        if entry is None:
            return
        key = entry[0]
        _es_subtree_size, es_vec = entry[1]
        if not es_vec:
            return

        key_parent = key.parent
        if key_parent is None:
            raise AssertionError("key parent node is invalid (reduce_factorize_fraction).")
        kp_parent = key_parent.parent
        if kp_parent is None:
            raise AssertionError("kp parent node is invalid (reduce_factorize_fraction).")

        if vertex_equal(key_parent.data, builtin.div):
            main_div = key_parent
            remain_plus = tr.wrap(main_div.first_child, builtin.plus)
        else:
            main_div = tr.append_child(it, builtin.div)
            remain_plus = tr.append_child(main_div, builtin.plus)
            tr.move_ontop(tr.append_child(main_div, contin_t(0.0)), key)
            if tr.is_childless(key_parent):
                tr.move_ontop(tr.append_child(remain_plus), kp_parent.first_child)
                tr.erase(kp_parent)
            else:
                tr.move_ontop(tr.append_child(remain_plus), key_parent)

        for factor in es_vec:
            factor_parent = factor.parent
            if factor_parent is None:
                raise AssertionError(
                    "factor parent node is invalid (reduce_factorize_fraction)."
                )
            fp_parent = factor_parent.parent
            if fp_parent is None:
                raise AssertionError("fp parent node is invalid (reduce_factorize_fraction).")
            tr.erase(factor)
            if vertex_equal(factor_parent.data, builtin.div):
                new_loc = tr.append_child(remain_plus, contin_t(0.0))
                tr.move_ontop(new_loc, factor_parent.first_child)
                tr.erase(factor_parent)
            else:
                if tr.is_childless(factor_parent):
                    tr.move_ontop(
                        tr.append_child(remain_plus, contin_t(0.0)), fp_parent.first_child
                    )
                    tr.erase(fp_parent)
                else:
                    # C++ moves `fp_parent` here (contin_rules.cc:263), NOT
                    # `factor_parent` -- the whole remaining `x/(reduced
                    # product)` div subtree, preserving both the numerator
                    # and the surviving factors, not just the bare product.
                    tr.move_ontop(tr.append_child(remain_plus, contin_t(0.0)), fp_parent)


class reduce_factorize(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:53 reduce_factorize

    `x*y+x*z -> x*(y+z)`, or more generally `sum x*y_i + sum z_j ->
    x*(sum y_i) + sum z_j` -- unify a common MULTIPLICATIVE factor shared
    by two or more `+` terms (also works if `x` is a `div`'s numerator).
    Same tie-breaking convention as `reduce_factorize_fraction`.
    """

    def __init__(self, name: str = "reduce_factorize") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:277 reduce_factorize::operator()

        `key_parent`/`kp_parent`/`key_div` (and their `factor_*` twins in
        the second loop) are captured ONCE, before `tr.wrap`/`tr.move_ontop`
        run, and read again AFTER -- exactly mirroring C++'s own pattern of
        holding a `pre_it` (a stable node handle) across mutations that
        move OTHER nodes around it, never `key_parent` itself directly
        (only what `key_parent` used to contain).
        """
        if not vertex_equal(it.data, builtin.plus):
            return
        if tr.is_childless(it):
            raise AssertionError("combo_tree node should not be childless")

        sp = _LexicographicSubtreeMap()
        chosen_factor: tree_node | None = None
        reduced_size = 0

        plus_child = it.first_child
        while plus_child is not None:
            pos = (
                plus_child.first_child
                if vertex_equal(plus_child.data, builtin.div)
                else plus_child
            )
            if vertex_equal(pos.data, builtin.times):
                times_child = pos.first_child
                while times_child is not None:
                    pf = times_child
                    if is_contin(pf.data) and get_contin(pf.data) == 1.0:
                        times_child = times_child.next_sibling
                        continue
                    entry = sp.find(pf)
                    if entry is None:
                        sp.insert(pf, (tr.count(pf), []))
                    else:
                        key = entry[0]
                        es_subtree_size, es_vec = entry[1]
                        last_it = key if not es_vec else es_vec[-1]
                        last_it_parent = last_it.parent
                        if last_it_parent is None:
                            raise AssertionError("last combo_tree node is invalid")
                        if last_it_parent is not pos:
                            es_vec.append(pf)
                            prs = len(es_vec) * es_subtree_size
                            if prs > reduced_size:
                                reduced_size = prs
                                chosen_factor = pf
                    times_child = times_child.next_sibling
            else:
                pf = pos
                if is_contin(pf.data) and get_contin(pf.data) == 1.0:
                    break
                entry = sp.find(pf)
                if entry is None:
                    sp.insert(pf, (tr.count(pf), []))
                else:
                    es_subtree_size, es_vec = entry[1]
                    es_vec.append(pf)
                    prs = len(es_vec) * es_subtree_size
                    if prs > reduced_size:
                        reduced_size = prs
                        chosen_factor = pf
            plus_child = plus_child.next_sibling

        if chosen_factor is None:
            return
        entry = sp.find(chosen_factor)
        if entry is None:
            return
        key = entry[0]
        _es_subtree_size, es_vec = entry[1]
        first_one: tree_node | None = None
        if not es_vec:
            return

        key_parent = key.parent
        if key_parent is None:
            raise AssertionError("key parent node is invalid (reduce_factorize).")
        kp_parent = key_parent.parent
        key_div: tree_node | None = None
        if vertex_equal(key_parent.data, builtin.div):
            key_div = key_parent
        elif kp_parent is not None and vertex_equal(kp_parent.data, builtin.div):
            key_div = kp_parent

        if key_parent is not it:
            tr.move_ontop(tr.append_child(it, contin_t(0.0)), key)
        tr.wrap(key, builtin.times)
        remain_plus = tr.insert_after(key, builtin.plus)

        if tr.is_childless(key_parent):
            tr.erase(key_parent)
            key_parent = kp_parent

        if key_parent is key_div:
            tr.prepend_child(key_parent, contin_t(1.0))

        if key_parent is it:
            first_one = tr.append_child(remain_plus, contin_t(1.0))
        elif key_div is not None:
            tr.move_ontop(tr.append_child(remain_plus, contin_t(0.0)), key_div)
        else:
            tr.move_ontop(tr.append_child(remain_plus, contin_t(0.0)), key_parent)

        for factor in es_vec:
            factor_parent = factor.parent
            tr.erase(factor)
            if factor_parent is None:
                raise AssertionError("factor parent node is invalid")
            fp_parent = factor_parent.parent
            factor_div: tree_node | None = None
            if vertex_equal(factor_parent.data, builtin.div):
                factor_div = factor_parent
            elif fp_parent is not None and vertex_equal(fp_parent.data, builtin.div):
                factor_div = fp_parent

            if tr.is_childless(factor_parent):
                tr.erase(factor_parent)
                factor_parent = fp_parent

            if factor_parent is factor_div:
                tr.prepend_child(factor_parent, contin_t(1.0))

            if factor_parent is it:
                if first_one is None:
                    first_one = tr.append_child(remain_plus, contin_t(1.0))
                else:
                    first_one.data = get_contin(first_one.data) + 1.0
            elif factor_div is not None:
                tr.move_ontop(tr.append_child(remain_plus, contin_t(0.0)), factor_div)
            else:
                tr.move_ontop(tr.append_child(remain_plus, contin_t(0.0)), factor_parent)


def _find_child_by_vertex(parent: tree_node, value) -> tree_node | None:
    """No direct C++ symbol: `std::find(it.begin(), it.end(),
    vertex(id::xxx))` (`reduce_distribute`'s two real call sites,
    contin_rules.cc:464,475) over `parent`'s own children, comparing each
    child's `.data` against `value`. Ported via `vertex_equal`, NOT bare
    `==` -- `boost::variant::operator==` is type-safe in C++ (this port's
    usual "type before value" concern), and `value` here is always a
    `builtin` (`id::plus`), so a bare Python `==` against a `contin_t`
    child could otherwise false-positive on an ordinal collision.
    """
    child = parent.first_child
    while child is not None:
        if vertex_equal(child.data, value):
            return child
        child = child.next_sibling
    return None


class reduce_distribute(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:65 reduce_distribute

    `*(+(X1 ... Xn) Y)`: distribute `Y` into `+(X1 ... Xn)`, then reduce
    `+(*(X1 Y) ... *(Xn Y))` (without re-factorizing the root, to avoid
    infinite recursion against `reduce_factorize`) and keep the result
    only if it is no LARGER than before.
    """

    def __init__(self, r: rule, name: str = "reduce_distribute") -> None:
        """C++: moses/comboreduct/reduct/contin_rules.h:66 reduce_distribute::reduce_distribute

        `_reduction` is a raw, NON-OWNING `const rule*` in C++ (no
        `.clone()`), the same convention `logical_rules.py`'s
        `reduce_and_assumptions` uses for the identical reason -- the
        caller's reduction rule (`iter_without_factorize_distribute` in
        `contin_reduction.py`) outlives every `reduce_distribute` built
        from it.
        """
        super().__init__(name)
        self._reduction = r

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:461 reduce_distribute::operator()"""
        if not (vertex_equal(it.data, builtin.times) and tr.number_of_children(it) >= 2):
            return
        plus_it = _find_child_by_vertex(it, builtin.plus)
        if plus_it is None:
            return

        idx = 0
        fac_it = it.first_child
        while fac_it is not None:
            if not vertex_equal(fac_it.data, builtin.plus):
                tr_copy = combo_tree(it)
                size_before_reduct = tr_copy.size()
                tr_cr = tr_copy.begin()
                fac_copy_it = tr_copy.child(tr_cr, idx)
                plus_ci = _find_child_by_vertex(tr_cr, builtin.plus)

                child = plus_ci.first_child
                while child is not None:
                    if not vertex_equal(child.data, builtin.times):
                        child = tr_copy.insert_above(child, builtin.times)
                    tr_copy.replace(
                        tr_copy.append_child(child, contin_t(0.0)), fac_copy_it
                    )
                    child = child.next_sibling

                tr_copy.erase(fac_copy_it)
                self._reduction(tr_copy)
                size_after_reduct = tr_copy.size()
                if size_after_reduct <= size_before_reduct:
                    replace_without_changing_it(tr, it, tr_cr)
                    return
            fac_it = fac_it.next_sibling
            idx += 1


class reduce_invert_constant(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:97 reduce_invert_constant

    `x/c -> 1/c * x`; `x/(c*y) -> 1/c * x/y`; `0/x -> 0`; `x/0 -> NAN`
    (see this module's own docstring on `FP_INFINITE`'s real meaning).
    """

    def __init__(self, name: str = "reduce_invert_constant") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:503 reduce_invert_constant::operator()"""
        if not vertex_equal(it.data, builtin.div):
            return
        if tr.number_of_children(it) != 2:
            raise AssertionError(
                "combo_tree node should have exactly two children (reduce_invert_constant)."
            )
        sib = it.first_child
        if is_contin(sib.data) and get_contin(sib.data) == 0.0:  # 0/x -> 0
            tr.erase_children(it)
            it.data = 0.0
            return

        sib = sib.next_sibling
        if is_contin(sib.data):  # x/c -> 1/c * x
            divisor = get_contin(sib.data)
            if divisor != 0.0:
                sib.data = 1.0 / divisor
                it.data = builtin.times
            else:
                tr.erase_children(it)
                it.data = 1.0  # DO NOT FIX -- FP_INFINITE is really 1.0, see docstring
            return

        if vertex_equal(sib.data, builtin.times):  # x/(c*y) -> 1/c * x/y
            divisor = 1.0
            mul_sib = sib.first_child
            while mul_sib is not None:
                if is_contin(mul_sib.data):
                    divisor *= get_contin(mul_sib.data)
                    mul_sib = tr.erase(mul_sib)
                else:
                    mul_sib = mul_sib.next_sibling
            if divisor != 1.0:
                if divisor != 0.0:
                    it.data = builtin.times
                    if tr.is_childless(sib):
                        sib.data = 1.0 / divisor
                    else:
                        if tr.has_one_child(sib):
                            sib = tr.erase(tr.flatten(sib))
                        new_div = tr.append_child(it, builtin.div)
                        tr.reparent(new_div, it.first_child, new_div)
                        tr.append_child(it, 1.0 / divisor)
                else:
                    tr.erase_children(it)
                    it.data = 1.0  # DO NOT FIX -- FP_INFINITE is really 1.0
            elif tr.is_childless(sib):
                sib.data = 1.0


class reduce_fraction(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:106 reduce_fraction

    `(x*y)/(x*z) -> y/z`, or more generally `(prod x_i*prod y_j)/(prod
    x_i*prod z_k) -> prod y_j/prod z_k`.

    MEASURED UPSTREAM BUG, ported verbatim -- DO NOT FIX: C++'s tracking
    structure is a real `std::multiset<pre_it, lexicographic_subtree_
    order<vertex>>` of the numerator's own factors, and `nc.erase(
    denom_el)` uses the KEY-based erase overload, which removes EVERY
    element considered equal to `denom_el` by the comparator -- not just
    the one `nc.find(denom_el)` actually located. So when the numerator
    holds MULTIPLE copies of a factor that also appears in the
    denominator, only the ONE copy `find` happens to return is physically
    erased from the tree; every OTHER equal copy stays in the numerator,
    untouched, but is silently dropped from `nc`'s own bookkeeping and can
    never be matched again -- even against a SECOND, later-encountered
    equal factor in the denominator. Net effect: with a duplicated common
    factor on both sides, only ONE copy cancels per side, not all of them.
    Confirmed directly against the compiled `libcomboreduct.so`
    (`downwards(reduce_fraction())` alone, not the full pipeline):
    `(x*x*y)/(x*x)` reduces to `(x*y)/(x)`, NOT the fully-cancelled `y` a
    naive reading of the rule's own comment would expect.
    """

    def __init__(self, name: str = "reduce_fraction") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:570 reduce_fraction::operator()"""
        if not vertex_equal(it.data, builtin.div):
            return
        if tr.number_of_children(it) != 2:
            raise AssertionError(
                "combo_tree node should have exactly two children (reduce_fraction)."
            )
        numerator = it.first_child
        denominator = it.last_child

        nc = _LexicographicSubtreeSet()
        if vertex_equal(numerator.data, builtin.times):
            sib = numerator.first_child
            while sib is not None:
                nc.add(sib)
                sib = sib.next_sibling
        else:
            nc.add(numerator)

        if vertex_equal(denominator.data, builtin.times):
            sib = denominator.first_child
            while sib is not None and len(nc):
                denom_el = sib
                sib = sib.next_sibling
                mirror_el = nc.find(denom_el)
                if mirror_el is not None:
                    nc.discard_all_equal(denom_el)
                    if mirror_el is numerator:
                        tr.erase_children(mirror_el)
                        mirror_el.data = 1.0
                    else:
                        tr.erase(mirror_el)
                        if tr.is_childless(numerator):
                            numerator.data = 1.0
                    tr.erase(denom_el)
                    if tr.is_childless(denominator):
                        denominator.data = 1.0
        else:
            mirror_el = nc.find(denominator)
            if mirror_el is not None:
                if mirror_el is numerator:
                    tr.erase_children(mirror_el)
                    mirror_el.data = 1.0
                else:
                    tr.erase(mirror_el)
                    if tr.is_childless(numerator):
                        numerator.data = 1.0
                tr.erase_children(denominator)
                denominator.data = 1.0


class reduce_times_div(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:113 reduce_times_div

    `(x/y)/z -> x/(y*z)`; `x/(y/z) -> (x*z)/y`; `x*(y/z) -> (x*y)/z`, or
    more generally `prod x_i * prod y_j/z_j -> (prod x_i * prod y_j)/(prod
    z_j)`.
    """

    def __init__(self, name: str = "reduce_times_div") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:640 reduce_times_div::operator()

        The `*it==id::times` branch's `for(sib_it sib = it.begin(); sib !=
        it.end(); ++sib)` walks a `sib` whose underlying node gets
        PHYSICALLY RELOCATED mid-walk (into a newly-prepended `times`
        wrapper, on the first match) -- `sib`'s own `next_sibling` chain
        then continues through that NEW parent's children rather than
        `it`'s original ones, which is exactly what makes a single walk
        correctly visit every remaining `y_j/z_j` term after the first one
        moves them all under `new_times`. Ported as a `while` with the
        SAME two-step advance C++'s for-loop gives every div match (the
        explicit `sib = tr.erase(tr.flatten(sib))` reassignment, THEN the
        loop's own automatic next-sibling step) -- both run only for a
        match; a non-matching `sib` gets just the ordinary single step.
        """
        if vertex_equal(it.data, builtin.div):
            if tr.number_of_children(it) != 2:
                raise AssertionError(
                    "combo_tree node should have exactly two children (reduce_times_div)."
                )
            sib = it.first_child
            if vertex_equal(sib.data, builtin.div):  # (x/y)/z -> x/(y*z)
                if tr.number_of_children(sib) != 2:
                    raise AssertionError(
                        "combo_tree sibiling node should have exactly two children "
                        "(reduce_times_div)."
                    )
                y = sib.last_child
                z = it.last_child
                tr.insert_above(z, builtin.times)
                tr.move_after(z, y)
                tr.erase(tr.flatten(sib))
                return

            sib = sib.next_sibling
            if vertex_equal(sib.data, builtin.div):  # x/(y/z) -> (x*z)/y
                if tr.number_of_children(sib) != 2:
                    raise AssertionError(
                        "combo_tree sibiling node should have exactly two children "
                        "(reduce_times_div)."
                    )
                x = it.first_child
                z = sib.last_child
                tr.insert_above(x, builtin.times)
                tr.move_after(x, z)
                tr.erase(tr.flatten(sib))
            return

        if vertex_equal(it.data, builtin.times):
            denom: tree_node | None = None
            denom_times: tree_node | None = None
            new_times: tree_node | None = None
            sib = it.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.div):
                    if not vertex_equal(it.data, builtin.div):
                        if tr.is_childless(it):
                            raise AssertionError(
                                "combo_tree node should not be childless (reduce_time_div)."
                            )
                        new_times = tr.prepend_child(it, builtin.times)
                        second_child = tr.child(it, 1)
                        tr.reparent(new_times, second_child, None)
                        it.data = builtin.div
                    if tr.number_of_children(sib) != 2:
                        raise AssertionError(
                            "combo_tree sibiling node should have exactly two children "
                            "(reduce_times_div)."
                        )
                    z_j = sib.last_child
                    if denom is None:
                        denom = tr.move_after(new_times, z_j)
                    else:
                        if denom_times is None:
                            denom_times = tr.insert_above(denom, builtin.times)
                        tr.move_after(denom, z_j)
                    sib = tr.erase(tr.flatten(sib))
                sib = sib.next_sibling if sib is not None else None


class reduce_plus_times_one_child(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:120 reduce_plus_times_one_child

    `+(x) -> x`; `*(x) -> x`.
    """

    def __init__(self, name: str = "reduce_plus_times_one_child") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:703 reduce_plus_times_one_child::operator()

        `it.data = it_child.data` needs no defensive copy despite reusing
        `tree.py`'s aliasing-prone plain-data-assignment pattern: `it_child`
        is discarded by `tr.erase(tr.flatten(it_child))` on the very next
        line, and nothing else in the tree references that same object
        (the same reasoning `logical_rules.py`'s `reduce_nots` docstring
        gives for its own in-place `is_argument` branch).
        """
        if (
            vertex_equal(it.data, builtin.plus) or vertex_equal(it.data, builtin.times)
        ) and tr.has_one_child(it):
            it_child = it.first_child
            it.data = it_child.data
            tr.erase(tr.flatten(it_child))


class reduce_sum_log(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:128 reduce_sum_log

    `log(x)+log(y) -> log(x*y)`; `log(x)-log(y) -> log(x/y)`, or more
    generally `sum log(x_i) - sum log(y_j) -> log((prod x_i)/(prod
    y_j))`. Works only if at least one `log(x_i)` exists (otherwise it
    would conflict with `reduce_log_div_times`'s `log(c/x) ->
    -log(c^-1*x)`).
    """

    def __init__(self, name: str = "reduce_sum_log") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:718 reduce_sum_log::operator()"""
        if not vertex_equal(it.data, builtin.plus):
            return
        first_log = _find_child_by_vertex(it, builtin.log)
        if first_log is None:
            return
        if not tr.has_one_child(first_log):
            raise AssertionError("combo_tree node should have exactly one child.")

        num = first_log.first_child
        denom: tree_node | None = None
        num_times: tree_node | None = None
        denom_times: tree_node | None = None

        sib = it.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.log) and sib is not first_log:
                if not tr.has_one_child(sib):
                    raise AssertionError(
                        "combo_tree sibiling node should have exactly one child."
                    )
                if num_times is None:
                    num_times = tr.wrap(num, builtin.times)
                tr.move_after(num, sib.first_child)
                sib = tr.erase(sib)
            elif vertex_equal(sib.data, builtin.times) and tr.number_of_children(sib) == 2:
                minus1: tree_node | None = None
                log_node: tree_node | None = None
                times_child = sib.first_child
                while times_child is not None and not (minus1 is not None and log_node is not None):
                    if minus1 is None and is_contin(times_child.data) and get_contin(
                        times_child.data
                    ) == -1.0:
                        minus1 = times_child
                    if log_node is None and vertex_equal(times_child.data, builtin.log):
                        log_node = times_child
                    times_child = times_child.next_sibling
                is_minus_log = minus1 is not None and log_node is not None
                if is_minus_log:
                    if denom is None:
                        if not tr.has_one_child(first_log):
                            raise AssertionError("combo_tree node should have exactly one child")
                        tmp_child_log = first_log.first_child
                        div_node = tr.wrap(tmp_child_log, builtin.div)
                        if not tr.has_one_child(div_node):
                            raise AssertionError(
                                "combo_tree sibiling node should have exactly one child"
                            )
                        denom = tr.move_after(div_node.first_child, log_node.first_child)
                        sib = tr.erase(sib)
                    else:
                        if denom_times is None:
                            denom_times = tr.wrap(denom, builtin.times)
                        if not tr.has_one_child(log_node):
                            raise AssertionError(
                                "combo_tree node (log) should have exactly one child"
                            )
                        tr.move_after(denom, log_node.first_child)
                        sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            else:
                sib = sib.next_sibling


class reduce_log_div_times(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:136 reduce_log_div_times

    `log(c/x) -> -log(c^-1*x)`; `log(0) -> NAN` (this module's own
    docstring: really `1.0`, `FP_INFINITE`'s real meaning); `log(exp(x))
    -> x`; `log(prod exp(x_i)*prod y_j) -> sum x_i + log(prod y_j)`.
    """

    def __init__(self, name: str = "reduce_log_div_times") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:793 reduce_log_div_times::operator()"""
        if not vertex_equal(it.data, builtin.log):
            return
        if not tr.has_one_child(it):
            raise AssertionError("combo_tree node id::log should have exactly one child")

        log_child = it.first_child
        if vertex_equal(log_child.data, builtin.div):  # log(c/x) -> log(c^-1*x)
            if tr.number_of_children(log_child) != 2:
                raise AssertionError("combo_tree node id::div should have exactly two children")
            num = log_child.first_child
            if is_contin(num.data):
                c = get_contin(num.data)
                if c != 0.0:
                    num.data = 1.0 / c
                    log_child.data = builtin.times
                    it.data = builtin.times  # trick to keep it as root of the subtree
                    tr.insert_after(tr.wrap(log_child, builtin.log), -1.0)
                else:
                    tr.erase_children(it)
                    it.data = 1.0  # DO NOT FIX -- FP_INFINITE is really 1.0, see module docstring
                    return
            return

        if vertex_equal(log_child.data, builtin.exp):  # log(exp(x)) -> x
            if not tr.has_one_child(log_child):
                raise AssertionError("combo_tree node id::exp should have exactly one child")
            tr.erase(tr.flatten(log_child))
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))
            return

        if vertex_equal(log_child.data, builtin.times):
            # log(prod exp(x_i)*prod y_j) -> sum x_i + log(prod y_j)
            new_log: tree_node | None = None
            sib = log_child.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.exp):
                    if not vertex_equal(it.data, builtin.plus):
                        new_log = tr.insert_above(log_child, builtin.log)
                        it.data = builtin.plus
                    if not tr.has_one_child(sib):
                        raise AssertionError(
                            "combo_tree sibling node should have exactly one child"
                        )
                    tr.reparent(it, sib)
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            # sum x_i + log(*()) -> sum x_i
            if new_log is not None and tr.has_one_child(new_log) and tr.is_childless(
                new_log.first_child
            ):
                tr.erase(new_log)
            return


class reduce_exp_times(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:143 reduce_exp_times

    `prod exp(x_i) -> exp(sum x_i)`.
    """

    def __init__(self, name: str = "reduce_exp_times") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:861 reduce_exp_times::operator()"""
        if not vertex_equal(it.data, builtin.times):
            return
        x_1: tree_node | None = None
        sib = it.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.exp):
                if not tr.has_one_child(sib):
                    raise AssertionError(
                        "combo_tree sibling node should have exactly one child "
                        "(reduce_exp_times)."
                    )
                if x_1 is None:
                    x_1 = sib.first_child
                    sib = sib.next_sibling
                else:
                    if not vertex_equal(x_1.parent.data, builtin.plus):
                        tr.insert_above(x_1, builtin.plus)
                    tr.move_after(x_1, sib.first_child)
                    sib = tr.erase(sib)
            else:
                sib = sib.next_sibling


class reduce_exp_div(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:150 reduce_exp_div

    `x/exp(y) -> x*exp(-y)`.
    """

    def __init__(self, name: str = "reduce_exp_div") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:888 reduce_exp_div::operator()"""
        if not vertex_equal(it.data, builtin.div):
            return
        if tr.number_of_children(it) != 2:
            raise AssertionError(
                "combo_tree node should have exactly two childrem (reduce_exp_div)."
            )
        denom = it.last_child
        if not vertex_equal(denom.data, builtin.exp):
            return

        it.data = builtin.times
        if not tr.has_one_child(denom):
            raise AssertionError(
                "combo_tree node should have exactly one child (reduce_exp_times - demon)."
            )
        y = denom.first_child
        if vertex_equal(y.data, builtin.times):
            if tr.is_childless(y):
                raise AssertionError(
                    "combo_tree sibling node should have exactly one child "
                    "(reduce_exp_times - y)."
                )
            y = y.first_child
        else:
            tr.insert_above(y, builtin.times)

        tr.insert_after(y, -1.0)


class reduce_exp_log(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:170 reduce_exp_log

    `exp(log(x)+y) -> x*exp(y)`, or more generally `exp(sum log(x_i) + sum
    y_j) -> prod x_i * exp(sum y_j)`. Only valid when `log` is NOT
    `log(abs(x))` -- guarded in C++ by `#ifndef ABS_LOG`, and `ABS_LOG` is
    commented out at `vertex.h:49` (`// #define ABS_LOG`), so this rule is
    always compiled into classic and is always ported here.
    """

    def __init__(self, name: str = "reduce_exp_log") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:919 reduce_exp_log::operator()"""
        if not vertex_equal(it.data, builtin.exp):
            return
        if not tr.has_one_child(it):
            raise AssertionError("combo_tree node should have exactly one child (reduce_exp_log).")
        exp_child = it.first_child
        if vertex_equal(exp_child.data, builtin.plus):
            x_1: tree_node | None = None
            new_exp: tree_node | None = None
            if tr.is_childless(exp_child):
                raise AssertionError(
                    "combo_tree child node should have exactly one child (reduce_exp_log)."
                )
            sib = exp_child.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.log):
                    if not tr.has_one_child(sib):
                        raise AssertionError(
                            "combo_tree sibling node should have exactly one child "
                            "(reduce_exp_log)."
                        )
                    if x_1 is None:
                        it.data = builtin.times
                        x_1 = tr.move_after(exp_child, sib.first_child)
                        new_exp = tr.insert_above(exp_child, builtin.exp)
                    else:
                        tr.move_after(x_1, sib.first_child)
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
            if new_exp is not None and tr.has_one_child(new_exp) and tr.is_childless(
                new_exp.first_child
            ):
                tr.erase(tr.flatten(new_exp.first_child))
                tr.erase(tr.flatten(new_exp))
        elif vertex_equal(exp_child.data, builtin.log):
            tr.erase(tr.flatten(exp_child))
            it.data = it.first_child.data
            tr.erase(tr.flatten(it.first_child))


class reduce_sin(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:180 reduce_sin

    `sin(x + c) -> sin(x + (c>pi? c-pi : (c<=pi? c+pi)))`, or more
    generally `sin(sum x_i + sum c_j) -> sin(sum x_i + ((sum c_j)+pi)%2pi
    - pi)` -- fold the constant term of a `sin`'s argument into the range
    `(-pi, pi]`.
    """

    def __init__(self, name: str = "reduce_sin") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:965 reduce_sin::operator()"""
        if not vertex_equal(it.data, builtin.sin):
            return
        if not tr.has_one_child(it):
            raise AssertionError(
                "combo_tree node id::sin should have exactly one child (reduce_sin)."
            )

        c_it: tree_node | None = None
        sin_child = it.first_child
        if vertex_equal(sin_child.data, builtin.plus):
            if tr.is_childless(sin_child):
                raise AssertionError(
                    "combo_tree node id::plus should not be childless (reduce_sin)."
                )
            sib = sin_child.first_child
            while sib is not None:
                if is_contin(sib.data):
                    if c_it is None:
                        c_it = sib
                        sib = sib.next_sibling
                    else:
                        c_it.data = get_contin(c_it.data) + get_contin(sib.data)
                        sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
        elif is_contin(sin_child.data):
            c_it = sin_child

        if c_it is not None:
            if not is_contin(c_it.data):
                raise AssertionError("sin_child isn't of type contin (reduce_sin).")
            c = get_contin(c_it.data)
            if not math.isfinite(c):
                tr.erase_children(it)
                it.data = 0.0  # DO NOT FIX -- `*it = FP_NAN` is really 0.0, see module docstring
                return
            if c <= -math.pi or c > math.pi:
                c_it.data = math.fmod((c + math.pi), 2.0 * math.pi) - math.pi


class reduce_impulse_arg(rule):
    """C++: moses/comboreduct/reduct/contin_rules.h:187 reduce_impulse_arg

    Apply boolean reduction to the argument of `impulse` (same class
    structure as `branch_rules.py`'s `reduce_cond_arg`).
    """

    def __init__(self, effort: int, ignore_ops: set, name: str = "reduce_impulse_arg") -> None:
        """C++: moses/comboreduct/reduct/contin_rules.h:191 reduce_impulse_arg::reduce_impulse_arg

        `ignore_ops` is stored by REFERENCE in C++, not cloned -- same
        convention as `branch_rules.py`'s `reduce_cond_arg`.
        """
        super().__init__(name)
        self.reduct_effort = effort
        self.ignore_ops = ignore_ops

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/contin_rules.cc:1010 reduce_impulse_arg::operator()"""
        if not vertex_equal(it.data, builtin.impulse):
            return
        if not tr.has_one_child(it):
            raise AssertionError("impulse should have exactly one child (reduce_impulse_arg).")
        it_child = it.first_child
        logical_reduce(self.reduct_effort, tr, it_child, self.ignore_ops)
