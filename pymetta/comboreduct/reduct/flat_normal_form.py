"""C++: moses/comboreduct/reduct/flat_normal_form.{h,cc}

The flat clause-set representation `general_rules.py`'s type-independent
rules feed into and out of, plus `nf_mapper`, which converts between
combo's tree-shaped `and(...)`/`or(...)` boolean expressions and this flat
form.

A LITERAL is a nonzero `int`: positive `n` means the n-th item
`nf_mapper.add_item` interned; negative `-n` means that same item's
NEGATION. A CLAUSE (`clause`, below) is a `std::set<int,
opencog::absolute_value_order<int>>` -- literals ordered by ABSOLUTE VALUE,
`-1, 1, -2, 2, -3, 3, ...`, not by signed value (docs/CARRY-FORWARD.md
§4.4). This is the load-bearing ordering fact this whole module exists to
get right: a natural (signed) sort would put every negated literal before
every positive one, silently producing a different clause order -- and
therefore, downstream, a different reduction -- than C++. It also means a
literal and its own negation always sort ADJACENT to each other (nothing
else can fall between them by absolute value), which is exactly what makes
`tautology`'s adjacent-pairs-only scan below correct and complete.

`nf` (a normal form: a CNF or DNF, depending which of `nf_mapper.add_cnf`/
`add_dnf` built it) is `list[clause]` -- one clause per top-level
conjunct/disjunct. `list`, not any dedup/order-preserving-set type: C++'s
own `nf` is `std::list<clause>` (an ordered sequence, deliberately
NOT a set of clauses -- duplicate or dominated clauses are a normal,
expected transient state that later logical_rules.py subsumption passes
clean up, not something this representation prevents).

Ground truth for every non-obvious piece of this file (`nf_mapper::
add_item`'s id-assignment/sign-folding, `extract_cnf`/`extract_dnf`'s
single-child collapsing and 0-children/empty-clause handling, `partition`'s
non-stability as it interacts with these clauses) was measured directly
against the real compiled `libcomboreduct.so`
(`tools/cpp_probe/flat_normal_form_probe.cc`, bounded per CLAUDE.md's
OOM-safety rule), not inferred from reading the header alone -- see
task-3-report.md for the full transcript this file's tests are pinned
against.
"""

from __future__ import annotations

import copy

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    is_argument,
    lexicographic_subtree_cmp,
    vertex_equal,
)
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# clause / nf
# ---------------------------------------------------------------------------


def _absolute_value_order_int_less(a: int, b: int) -> bool:
    """C++: opencog/util/numeric.h:56 absolute_value_order

    The checker matches plain identifiers only, so this cites the
    enclosing class at the line where the member actually lives -- the
    real member is `absolute_value_order<int>::operator()`
    (numeric.h:56-59).

    `(a == -b) ? a < b : abs(a) < abs(b)` -- ported verbatim. This is the
    genuine `absolute_value_order<int>` template instantiated DIRECTLY
    over plain `int`s (unlike `argument.__lt__`, which borrows the same
    formula for a different type, per that method's own docstring).
    Module-level (not a `clause` method): `_std_includes` below needs it
    too, over plain lists that are not themselves `clause` instances.
    """
    return a < b if a == -b else abs(a) < abs(b)


class clause:
    """C++: moses/comboreduct/reduct/flat_normal_form.h:41 clause

    `typedef std::set<int, opencog::absolute_value_order<int>> clause;`
    -- see this module's own docstring for what a literal means and why
    the ordering matters. Implemented as a Python object (not a plain
    `list`/`set`) so its sort order is enforced at every mutation site,
    the same way `std::set` enforces it at every `insert`, rather than
    left to whichever caller happens to build one correctly.

    Not a `frozenset`/`set`: those hash by value with no ordering
    guarantee at all, and this file's callers (`extract_cnf`/
    `extract_dnf`/`extract_conjunction`/`create`) genuinely depend on
    ITERATION ORDER (the reconstructed tree's own child order comes
    straight from iterating a clause), not just membership.
    """

    __slots__ = ("_items",)

    def __init__(self, items: tuple[int, ...] = ()) -> None:
        self._items: list[int] = []
        for x in items:
            self.insert(x)

    def _bisect(self, x: int) -> tuple[int, bool]:
        """Binary search `_items` (invariant: always sorted by
        `_absolute_value_order_int_less`, maintained by `insert`/`erase`
        below) for `x`. Returns (the index `x` belongs at, whether it is
        already present).
        """
        lo, hi = 0, len(self._items)
        while lo < hi:
            mid = (lo + hi) // 2
            v = self._items[mid]
            if v == x:
                return mid, True
            if _absolute_value_order_int_less(v, x):
                lo = mid + 1
            else:
                hi = mid
        return lo, False

    def insert(self, x: int) -> None:
        """C++: std::set<int, absolute_value_order<int>>::insert"""
        idx, found = self._bisect(x)
        if not found:
            self._items.insert(idx, x)

    def assign(self, other: clause) -> None:
        """No direct C++ symbol: `std::set::operator=` -- replaces this
        clause's own contents with `other`'s, IN PLACE, preserving this
        clause's own object identity. Added task-4 (S7) for
        `logical_rules.py`'s `reduce_ors`/`reduce_ands`
        (logical_rules.cc:322-325/384-387's `*c1 = *impl` / `*c2 = *impl`)
        -- both translate a C++ `nf::iterator` dereference-assignment
        (mutates the clause a `std::list<clause>::iterator` POINTS AT,
        without changing which list node it points at) into Python, where
        a caller may be holding an external reference to this exact
        `clause` object (e.g. a loop variable `c1`/`c2` compared by
        identity against other elements of the same `nf` later) and must
        keep observing updates through that SAME reference afterward, not
        a newly-constructed one.
        """
        self._items = list(other._items)

    def erase(self, x: int) -> None:
        """C++: std::set<int, absolute_value_order<int>>::erase(const key_type&)

        A no-op if `x` is not present, matching `std::set::erase`'s own
        permissive contract (it returns a count, never throws, for a
        missing key).
        """
        idx, found = self._bisect(x)
        if found:
            del self._items[idx]

    def first(self) -> int:
        """No direct C++ symbol: `*c.begin()` -- this file's own only use
        of `clause::begin()` is to immediately dereference it (never to
        keep the iterator itself), so this returns the literal VALUE
        directly rather than porting an iterator/dereference distinction
        `clause` (a set of plain `int`s, not `tree_node`s) has no use
        for.
        """
        return self._items[0]

    def empty(self) -> bool:
        """C++: std::set::empty"""
        return not self._items

    def size(self) -> int:
        """C++: std::set::size"""
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, x: int) -> bool:
        return self._bisect(x)[1]

    def __repr__(self) -> str:
        """C++: moses/comboreduct/reduct/flat_normal_form.cc:62 clause

        The checker matches plain identifiers only, so this cites the
        type the free function operates on rather than the un-parseable
        `operator<<` itself -- the real symbol is
        `operator<<(std::ostream&, const opencog::reduct::clause&)`
        (flat_normal_form.cc:62-72).

        `"(" + " ".join(literals) + ")"` -- ported for parity/debugging;
        no real C++ caller under moses/comboreduct/reduct/ actually uses
        this operator (grep confirms), it exists only for ad-hoc logging
        in classic. `__repr__`, not `__str__`: this port has no separate
        "informal" rendering to distinguish it from.
        """
        return "(" + " ".join(str(x) for x in self._items) + ")"


# C++: moses/comboreduct/reduct/flat_normal_form.h:42 nf
nf = list[clause]


def nf_to_str(f: nf) -> str:
    """C++: moses/comboreduct/reduct/flat_normal_form.cc:74 nf

    Cites `nf` (the type this free function operates on), not the
    un-parseable `operator<<` itself -- the real symbol is
    `operator<<(std::ostream&, const opencog::reduct::nf&)`
    (flat_normal_form.cc:74-78).

    Concatenates each clause's own `repr` with NO separator between them,
    matching the real operator's `for (...) out << *c;` loop exactly (not
    this module's own probe-script convenience of joining with spaces).
    """
    return "".join(repr(c) for c in f)


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def tautology(c: clause) -> bool:
    """C++: moses/comboreduct/reduct/flat_normal_form.cc:29 tautology

    Does `c` contain both a literal `p` and its negation `!p`? Scans only
    ADJACENT pairs (`std::adjacent_find`) -- correct and complete because
    `clause`'s absolute-value order always places `p`/`!p` next to each
    other (this module's own docstring; docs/CARRY-FORWARD.md §4.4), with
    nothing else able to fall between them.
    """
    items = list(c)
    for i in range(len(items) - 1):
        if items[i] == -items[i + 1]:
            return True
    return False


def _std_includes(haystack: list[int], needle: list[int]) -> bool:
    """No direct C++ symbol: `std::includes(first1,last1, first2,last2,
    comp)` (`subset_eq`/`subset` below, C++ flat_normal_form.cc:41-42/
    48-49) -- both `haystack`/`needle` are already sorted by `clause`'s
    own order (they come from `clause` objects). Ported as an explicit
    while-loop, not a `for x2 in needle` (which would wrongly force the
    needle pointer to advance every outer step): the real algorithm
    (cppreference's own reference implementation) advances `first1`
    (`haystack`) UNCONDITIONALLY every iteration -- written as the C++
    FOR loop's own increment clause, easy to misread as conditional --
    while `first2` (`needle`) advances only when the current haystack
    element actually matched, so a single needle element can require
    several haystack advances first.
    """
    i1, n1 = 0, len(haystack)
    i2, n2 = 0, len(needle)
    while i2 != n2:
        if i1 == n1 or _absolute_value_order_int_less(needle[i2], haystack[i1]):
            return False
        if not _absolute_value_order_int_less(haystack[i1], needle[i2]):
            i2 += 1
        i1 += 1
    return True


def subset_eq(c1: clause, c2: clause) -> bool:
    """C++: moses/comboreduct/reduct/flat_normal_form.cc:38 subset_eq

    Is `c1` a subset of (or equal to) `c2`?
    """
    return len(c2) >= len(c1) and _std_includes(list(c2), list(c1))


def subset(c1: clause, c2: clause) -> bool:
    """C++: moses/comboreduct/reduct/flat_normal_form.cc:45 subset

    Is `c1` a STRICT subset of `c2`?
    """
    return len(c2) > len(c1) and _std_includes(list(c2), list(c1))


def number_of_literals(f: nf) -> int:
    """C++: moses/comboreduct/reduct/flat_normal_form.cc:52 number_of_literals"""
    return sum(c.size() for c in f)


def implications(c1: clause, c2: clause):
    """C++: moses/comboreduct/reduct/flat_normal_form.h:53 implications

    `(p X)(!p Y)` implies `(X Y)`: for every literal `p` in `c1` whose
    negation appears in `c2`, yields the resolvent clause `(c1 U c2) -
    {p, !p}` -- unless that resolvent is itself a tautology (in which
    case it carries no information and is dropped). Note distinct
    complementary pairs between the SAME `c1`/`c2` each produce their OWN
    separate resolvent -- this is standard propositional resolution, not
    a loop bug (measured: `tools/cpp_probe/flat_normal_form_probe.cc`'s
    `implications_multi` case, where `c1={1,2}`/`c2={-1,-2,4}` have TWO
    complementary pairs, yields ZERO resolvents -- both would-be
    resolvents (`{-2,2,4}` and `{-1,1,4}`) are themselves tautologies via
    the OTHER pair, correctly suppressed).

    # DIVERGENCE(C++ flat_normal_form.h:53): yields via a generator, not an output iterator.
    # C++'s `template<typename Out> void implications(c1, c2, Out out)` writes each resolvent via
    # `(*out++) = c`, taking whatever output-iterator adapter the caller passes; this port drops
    # the `out` parameter and `yield`s each resolvent instead. Every real call site
    # (`logical_rules.cc:313,375`, Task 4) uses `std::back_inserter` into a fresh `nf` --
    # `list(implications(c1, c2))` here reproduces exactly that, and no other output-iterator
    # shape is used anywhere in $CLASSIC, so the collapse changes no real caller's observable
    # result -- only the C++ TEMPLATE PARAMETER this port has no equivalent mechanism for.
    """
    for lit1 in c1:
        for lit2 in c2:
            if lit1 == -lit2:
                merged = clause()
                for x in c1:
                    merged.insert(x)
                for x in c2:
                    merged.insert(x)
                merged.erase(lit1)
                merged.erase(lit2)
                if not tautology(merged):
                    yield merged


def pairwise_erase_if(f: nf, p) -> None:
    """C++: moses/comboreduct/reduct/flat_normal_form.h:71 pairwise_erase_if

    For every distinct pair of clauses in `f` (`c1` later in the list,
    `c2` earlier -- `p` is checked in both orders, so which one is
    "first" does not change the result): if `p(c1, c2)`, drop `c2`; elif
    `p(c2, c1)`, drop `c1` (and stop comparing the now-gone `c1` against
    anything else). Real call sites (`logical_rules.cc:304,369`, Task 4)
    use `p = subset_eq`: a clause subsumed by (or equal to) another is
    redundant and is dropped, keeping the smaller/equal one.

    Mutates `f` IN PLACE (`nf& f`), like C++. Ported via explicit index
    bookkeeping rather than `list`-with-node-identity, since Python's
    `list.pop`/`del` already give O(1)-per-step-adjusted indices for this
    algorithm's own access pattern (this file's clause counts are small;
    a linked-list is not needed for that reason either) -- `i1` is
    decremented whenever an EARLIER element (`f[i2]`) is erased, keeping
    it pointing at the same logical `c1` throughout the inner loop,
    mirroring how C++'s `nf::iterator c1` is unaffected by erasing any
    OTHER list element.
    """
    i1 = 0
    while i1 < len(f):
        i2 = 0
        erased_c1 = False
        while i2 < i1:
            if p(f[i1], f[i2]):
                del f[i2]
                i1 -= 1
            elif p(f[i2], f[i1]):
                del f[i1]
                erased_c1 = True
                break
            else:
                i2 += 1
        if not erased_c1:
            i1 += 1


# ---------------------------------------------------------------------------
# nf_mapper
# ---------------------------------------------------------------------------


class nf_mapper:
    """C++: moses/comboreduct/reduct/flat_normal_form.h:96 nf_mapper

    Not ported as a generic class (`nf_mapper<T>`): this port's `tree`/
    `combo_tree` carries no runtime type parameter (vertex.py's
    `combo_tree = tree[vertex]` note), and `nf_mapper<vertex>`
    (logical_rules.cc's own only instantiation) is the only one this
    codebase ever needs.

    Converts between combo's tree-shaped boolean expressions and this
    module's flat clause-set form, INTERNING each distinct leaf/atom it
    encounters as a small positive int id (`add_item`) so a clause can be
    a plain `set<int>` instead of a set of subtrees.
    """

    def __init__(self) -> None:
        # C++: Item2Int = std::map<sib_it, int, lexicographic_subtree_order<T>>
        # -- STRUCTURAL lookup by subtree CONTENT, not iterator/node
        # identity. Ported as a plain list of (tree_node, id) pairs,
        # linear-scanned via vertex.py's promoted `lexicographic_subtree_cmp`
        # (== 0 means structurally equivalent) rather than a balanced
        # tree keyed by that same comparator: clause/nf sizes in this
        # domain are small, and a linear scan is a direct, obviously
        # correct translation of "find an equivalent existing key" with
        # no separate data structure to get subtly wrong.
        self._item2int: list[tuple[tree_node, int]] = []
        # C++: Int2Item = std::unordered_map<int, tree> -- id -> a
        # standalone deep copy of the interned subtree (`tree(item)`,
        # this port's `combo_tree(tree_node)` constructor).
        self._int2item: dict[int, combo_tree] = {}

    def _find_item2int(self, item: tree_node) -> int | None:
        for node, id_ in self._item2int:
            if lexicographic_subtree_cmp(node, item) == 0:
                return id_
        return None

    def add_item(self, item: tree_node) -> int:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:229 nf_mapper::add_item

        Returns `item`'s interned id: a POSITIVE int the first time any
        STRUCTURALLY equivalent subtree is seen (whether that is `item`
        itself, or -- for an `argument` only -- its own negation seen
        earlier under the opposite sign); a NEGATIVE int (the negation of
        that first id) on every later encounter of the opposite sign.

        MEASURED (`tools/cpp_probe/flat_normal_form_probe.cc`'s
        `add_cnf_ids_mixed_args`), not assumed from reading alone: the id
        that ends up "positive" is whichever SIGN of a given argument was
        encountered FIRST, not always the positive one -- `_int2item`'s
        stored canonical copy can itself be a NEGATED argument (e.g. `!$2`
        interned before plain `$2` is ever seen). `create`/
        `extract_conjunction` below are written to not assume otherwise.
        """
        found = self._find_item2int(item)
        if found is not None:
            return found

        # Perhaps it's under minus-the-index?
        if isinstance(item.data, argument):
            item.data.negate()
            found = self._find_item2int(item)
            item.data.negate()
            if found is not None:
                return -found

        # Add it directly.
        next_id = len(self._int2item) + 1
        self._int2item[next_id] = combo_tree(item)
        self._item2int.append((item, next_id))
        return next_id

    def create(self, t: combo_tree, at: tree_node, idx: int) -> None:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:259 nf_mapper::create

        Appends a copy of the interned item `abs(idx)` as a new child of
        `at`, negated if `idx < 0` -- a literal translation of
        `get_argument(*t.append_child(...)).negate()`: append the
        subtree copy, then negate the FRESH copy's own root in place.

        This is safe -- the negate cannot reach back into `_int2item`'s
        stored original -- only because `tree.append_child`'s
        subtree-copy path genuinely value-copies a node's `.data`
        (`tree.py`'s `_copy_value`, task-3 review Critical fix); before
        that fix, this exact line would have silently corrupted
        `_int2item[-idx]` on every call (`_int2item`'s only
        negative-`idx`-reachable items are always themselves `argument`s
        -- see `add_item`'s own negate-and-relookup branch above -- so
        every real call into this branch would have hit it). See
        `tree.py`'s own `_copy_value` docstring for the full trace.
        """
        if idx < 0:
            new_node = t.append_child(at, self._int2item[-idx].begin())
            new_node.data.negate()
        else:
            t.append_child(at, self._int2item[idx].begin())

    def add_cnf(self, node: tree_node) -> nf:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:124 nf_mapper::add_cnf

        # DIVERGENCE(C++ flat_normal_form.h:124): takes a node, not a (from,to) sibling sub-range.
        # C++'s `add_cnf(sib_it from, sib_it to)` accepts an arbitrary sibling sub-range; this
        # port narrows the parameter to `node` (its FULL child range only) because that is the
        # only shape the real call site (`logical_rules.cc:348`, Task 4: `tr.begin(it),
        # tr.end(it)` -- `it`'s complete child list) ever uses -- the same scoping precedent this
        # port's `tree.sort`/`tree.partition`/`tree.sort_on_subtrees` already established.
        # Re-widening to an explicit sub-range is a mechanical addition (slice `node`'s children
        # by a `from`/`to` pair before the walk below) if a future caller ever needs it; no
        # observable behaviour differs for any input this narrower signature can still express.

        One clause per child of `node`: a child that is itself
        `logical_or` contributes one literal per ITS OWN child (a genuine
        disjunction); any other child contributes a single-literal clause
        (a bare atom).
        """
        result: nf = []
        child = node.first_child
        while child is not None:
            c = clause()
            if vertex_equal(child.data, builtin.logical_or):
                item = child.first_child
                while item is not None:
                    c.insert(self.add_item(item))
                    item = item.next_sibling
            else:
                c.insert(self.add_item(child))
            result.append(c)
            child = child.next_sibling
        return result

    def add_dnf(self, node: tree_node) -> nf:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:137 nf_mapper::add_dnf

        # DIVERGENCE(C++ flat_normal_form.h:137): takes a node, not a (from,to) sibling sub-range.
        # Same scoping collapse as `add_cnf` above, for the identical reason -- see that
        # method's own DIVERGENCE marker for the full explanation.

        As `add_cnf`, with `logical_and` in place of `logical_or`.
        """
        result: nf = []
        child = node.first_child
        while child is not None:
            c = clause()
            if vertex_equal(child.data, builtin.logical_and):
                item = child.first_child
                while item is not None:
                    c.insert(self.add_item(item))
                    item = item.next_sibling
            else:
                c.insert(self.add_item(child))
            result.append(c)
            child = child.next_sibling
        return result

    def extract_cnf(self, f: nf, t: combo_tree, sib: tree_node) -> None:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:151 nf_mapper::extract_cnf

        # DIVERGENCE(C++ flat_normal_form.h:151): takes the whole `f`, not an (It,It) sub-range.
        # C++'s `extract_cnf(It from, It to, tree&, sib_it)` accepts an arbitrary `nf` sub-range;
        # this port narrows to the whole `f` because its sole real call site
        # (`logical_rules.cc:426`, Task 4) passes `f.begin(), f.end()` -- the same scoping
        # collapse `add_cnf` above makes, for the same reason. The call at
        # `logical_rules.cc:331` is to `extract_dnf`, not this method.

        Rebuilds `sib` (and its descendants) as `and(or(...) or(...) ...)`
        from `f`. An EMPTY clause anywhere in `f` is silently SKIPPED -- it
        contributes NO conjunct at all, rather than making the whole
        result `logical_false` (which is what an empty OR-clause means in
        real boolean algebra, and IS how `extract_dnf` below treats an
        empty clause for ITS OWN identity). Ported exactly as measured
        (`tools/cpp_probe/flat_normal_form_probe.cc`'s
        `extract_cnf_with_empty_clause_present`): this genuine asymmetry
        with `extract_dnf` is a real, disclosed observation about C++'s
        own code, not something this port corrects.

        If `f` was entirely empty (zero clauses) OR every clause in it
        was empty (hence skipped), `sib` ends up with ZERO children --
        C++ sets `*sib = id::logical_false` for that case, mathematically
        odd for what is meant to be an "AND of nothing" (whose identity
        is TRUE, not FALSE) but ported verbatim (measured:
        `extract_cnf_empty_nf`) per "port faithfully, document upstream
        bugs, never silently fix them".
        """
        t.erase_children(sib)
        for c in f:
            if c.size() == 1:
                if c.first() != 0:
                    self.create(t, sib, c.first())
            elif not c.empty():
                or_node = t.append_child(sib, builtin.logical_or)
                for item in c:
                    self.create(t, or_node, item)
        if t.has_one_child(sib):
            sib.data = sib.first_child.data
            t.erase(t.flatten(sib.first_child))
        else:
            sib.data = builtin.logical_false if t.is_childless(sib) else builtin.logical_and

    def extract_dnf(self, f: nf, t: combo_tree, sib: tree_node) -> None:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:175 nf_mapper::extract_dnf

        # DIVERGENCE(C++ flat_normal_form.h:175): takes the whole `f`, not an (It,It) sub-range.
        # Same scoping collapse as `extract_cnf` above, for the identical reason -- see that
        # method's own DIVERGENCE marker for the full explanation.

        As `extract_cnf`, with the CNF/DNF roles of `logical_and`/
        `logical_or` swapped -- PLUS one real asymmetry, not a
        symmetric mirror of `extract_cnf` (see that method's own
        docstring): an EMPTY clause ANYWHERE in `f` makes the WHOLE
        result `logical_true` immediately (an empty AND-clause is
        trivially TRUE, and `OR(..., TRUE, ...)` is TRUE), discarding
        whatever conjuncts/disjuncts were already appended to `sib` from
        EARLIER clauses in `f` (measured:
        `extract_dnf_with_empty_clause_present` re-clears `sib`'s
        children even though a real prior clause had already added one).
        """
        t.erase_children(sib)
        for c in f:
            if c.empty():
                t.erase_children(sib)
                sib.data = builtin.logical_true
                return
            if c.size() == 1:
                if c.first() != 0:
                    self.create(t, sib, c.first())
            else:
                and_node = t.append_child(sib, builtin.logical_and)
                for item in c:
                    self.create(t, and_node, item)
        if t.has_one_child(sib):
            sib.data = sib.first_child.data
            t.erase(t.flatten(sib.first_child))
        else:
            sib.data = builtin.logical_false if t.is_childless(sib) else builtin.logical_or

    def extract_conjunction(self, c: clause, t: combo_tree, sib: tree_node) -> None:
        """C++: moses/comboreduct/reduct/flat_normal_form.h:202 nf_mapper::extract_conjunction

        Rebuilds `sib` as the single interned item `c` names (if `c` has
        exactly one literal), or as `and(...)` of every literal in `c`
        otherwise. A singleton clause holding literally `0` is a no-op --
        `sib` is left untouched, matching C++'s own `if (*c.begin()!=0)`
        guard exactly (this is the one call in this file where `sib` may
        legitimately arrive already holding real, meaningful data that
        must survive untouched: `logical_rules.cc:435` (Task 4) always
        calls this on a freshly-appended node, but this method's own
        contract does not depend on that).

        # DIVERGENCE(C++ flat_normal_form.h:202): sib.data=... bypasses tree.py's value-copy fix.
        # `*sib = *tmp` in both of C++'s size()==1 branches (flat_normal_form.h:206-217) copies a
        # `vertex` BY VALUE. `sib.data = tmp.data` here is a manual Python attribute assignment,
        # written directly in THIS file rather than going through `tree.append_child` -- so it
        # does NOT benefit from `tree.py`'s own `_copy_value` fix (task-3 review Critical; that
        # fix lives inside `append_child`'s subtree-copy branch and friends, none of which this
        # line calls). Left unfixed, it would alias `sib`'s value with `_int2item`'s stored
        # original directly, and the negative branch's own `.negate()` would then corrupt that
        # original exactly like `create`'s own (now-fixed) hazard did. `copy.copy(...)` is
        # applied at both assignments below for the same reason `tree.py`'s `_copy_value` uses
        # it: a no-op for the immutable alternatives, a genuine independent copy for `argument`.
        """
        if c.size() == 1:
            first = c.first()
            if first != 0:
                if first < 0:
                    interned = self._int2item[-first].begin().data
                    if not is_argument(interned):
                        raise AssertionError(
                            "Tree node referenced by iterator isn't an argument."
                        )
                    sib.data = copy.copy(interned)
                    sib.data.negate()
                else:
                    tmp = self._int2item[first].begin()
                    sib.data = copy.copy(tmp.data)
                    t.append_children(sib, list(t.children(tmp)))
        else:
            sib.data = builtin.logical_and
            for item in c:
                self.create(t, sib, item)
