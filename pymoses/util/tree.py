"""tree<T> -- n-ary linked tree with the splice operations MOSES uses.

C++: opencog/util/tree.h (Kasper Peeters' tree.hh v2.3, 3420 lines). This
ports a scoped subset -- see
.superpowers/sdd/2026-09-03-pymoses-s0-s6-foundation/task-2-brief.md -- not
the whole header: no forests/multiple roots (no head/feet sentinels), no
fixed-depth/breadth-first/upwards iterators, no merge/swap, no allocator
template parameter.

`partition` and `sort_on_subtrees` (task-3, S7) were added later, once
`comboreduct/reduct/general_rules.py`'s `eval_constants`/
`reorder_commutative` needed them -- both scoped, like this file's own
`sort`, to a NODE's full child range rather than C++'s arbitrary
sibling sub-range, since that is the only shape either real caller ever
uses. `partition` is NOT a stable partition -- see its own docstring.

Iterators collapse to plain `tree_node` objects. C++ keeps several distinct
iterator *classes* (iterator_base, pre_order_iterator, post_order_iterator,
sibling_iterator, leaf_iterator) mostly so each can carry its own transient
per-step state (chiefly `skip_current_children_`) alongside its own
operator++. None of the methods in this port's scoped API need that
per-iterator state to advance -- pre-order, post-order and leaf successors
are all pure functions of the node graph -- so a bare `tree_node` doubles as
every kind of "position" this module hands out, and Python object identity
(`is`) replaces C++'s pointer-identity-keyed `obj_ptr_cmp<It>`
(opencog/util/hashing.h:81).

No head/feet sentinels: C++ flanks the whole node graph with two dummy nodes
so that even the tree's actual root has a real (non-null) prev_sibling /
next_sibling, which is what lets a handful of methods (`insert` on the root,
forests in general) work via the same code path as any other sibling splice.
combo_tree is always single-rooted, so this port tracks the root directly as
`self._root` instead.

# DIVERGENCE(C++ opencog/util/tree.h:111): no head/feet sentinels -- a
# single root is tracked directly as `self._root`. Consequence: `insert`,
# `insert_after`, `move_after`, and `move_before` all raise
# ValueError if asked to give the tree's own root a new sibling (or to move
# the root itself into a sibling position), since either would require a
# forest (multiple top-level roots), which this port does not support. In
# C++ these same calls quietly succeed by growing a momentary or permanent
# forest under head/feet. `move_ontop(root, source)` is rejected too, but for
# a different reason: the C++ precondition forbids the target from containing
# the source, and violating it destroys the target subtree before reading the
# now-freed source (undefined behaviour), rather than creating a forest.
# `insert_above` and `wrap` are unaffected -- they
# replace the root in place rather than adding a sibling of it, which
# stays well-defined without sentinels.
"""

import copy
import functools


def _copy_value(data):
    """No direct C++ symbol: every C++ `tree<T>` subtree-copy operation
    (`tree(const iterator_base&)`, `replace(iter, const iterator_base&)`,
    `append_child(iter,iter)`, `insert_subtree`/`insert_subtree_after`)
    copies `T` BY VALUE -- for this port's real `T` (`vertex`, a
    `boost::variant`, vertex.py), that means the source node and the copy
    never share state afterward, even for the one variant alternative
    that is itself mutable in place (`argument.negate()`).

    # DIVERGENCE(C++ opencog/util/tree.h:1397): subtree-copy shared mutable data; now copies it.
    # Found in review (task-3, S7), not a deliberate design: `tree.__init__`'s iterator
    # constructor, `append_child`'s and `replace`'s subtree-copy branches,
    # `insert_subtree`/`insert_subtree_after`, and `_copy_descendants`'s own walk all passed a
    # source node's `.data` straight through unwrapped, silently sharing one mutable `argument`
    # object between the original node and every copy of it. C++ copies `vertex` BY VALUE at
    # every one of those call sites, so two different copies can never influence each other by
    # negating one in place -- exactly what `logical_rules.cc:204`/`logical_rules.h:235`'s "copy
    # a subtree, then negate the copy's own root" idiom (`push_back_negated_arguments`, Task 4)
    # depends on. Demonstrated live before this fix: `combo_tree(some_node_holding_$1)` followed
    # by negating the COPY's root argument also flipped the ORIGINAL node's sign
    # (`and($1 $2)` became `and(!$1 $2)`, not just the copy) -- see task-3-report.md's
    # aliasing-root-cause section for the full before/after trace and the regression tests this
    # fix added. `copy.copy` is applied uniformly rather than special-cased per alternative: for
    # this port's three IMMUTABLE `vertex` alternatives (`builtin` -- an IntEnum member,
    # `contin_t` -- a bare `float`, `enum_t`/plain `str`), `copy.copy` is already a documented
    # no-op returning the SAME object (verified: `copy.copy(3.0) is 3.0`), so this costs nothing
    # for the common case and only ever does real work for the one mutable alternative,
    # `argument` (`copy.copy(argument(5))` returns a genuinely distinct object with the same
    # `idx`, confirmed directly).
    """
    cls = type(data)
    if cls in _COPY_RETURNS_SELF:
        return data
    copier = _COPIER_BY_TYPE.get(cls)
    if copier is None:
        copier = _COPIER_BY_TYPE[cls] = _copier_for(cls)
    return copier(data)


# Speed only, same result as `copy.copy`: exact types `copy.copy` returns
# unchanged (some of its atomic types), and, per type, the
# function `copy.copy` would end up calling.
_COPY_RETURNS_SELF = frozenset({int, float, bool, str, type(None)})
_COPIER_BY_TYPE: dict = {}


def _copier_for(cls):
    """The callable `copy.copy(x)` resolves to for an `x` of exact type `cls`.

    `copy.copy` tries its own atomic types and builtin containers (none of
    which defines `__copy__`), then metaclasses, then `cls.__copy__`; only
    the last is short-circuited here. An `Enum` member's `__copy__`
    returns the member itself; `argument.__copy__` builds a new argument.
    """
    if not issubclass(cls, type):
        own_copy = getattr(cls, "__copy__", None)
        if own_copy is not None:
            return own_copy
    return copy.copy


def three_way_comparator(comp):
    """No C++ counterpart: turn a strict-weak-order `comp(a, b)` ("a sorts
    before b") into a three-way `cmp(a, b)`: -1 if `comp(a, b)`, else 1 if
    `comp(b, a)`, else 0 -- the two-test pattern every merge, set and sort
    helper here uses. A comparator may supply its own one-pass version as
    a `three_way` attribute (`lexicographic_subtree_less` does).
    """
    own = getattr(comp, "three_way", None)
    if own is not None:
        return own

    def _cmp(a, b):
        if comp(a, b):
            return -1
        if comp(b, a):
            return 1
        return 0

    return _cmp


class _DeadLink:
    """Sentinel poisoning a discarded node's own `parent`/`prev_sibling`/
    `next_sibling` after `tree.erase()` or the subtree-copy branch of
    `tree.replace()` detaches it -- see the DIVERGENCE markers on those two
    methods. `__slots__ = ()` gives every attribute access (or assignment)
    on this object an immediate `AttributeError`: there is deliberately no
    `.data`, no `.parent`, nothing to read -- so a stale handle that tries
    to navigate one step further through a poisoned link fails loudly at
    that exact point, rather than silently returning `None` (which already
    means "no parent"/"no sibling" for a perfectly live node) or a real
    neighbor object still attached to the live tree.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<erased tree_node link>"


_DEAD = _DeadLink()

# No direct C++ symbol. Sentinel for `tree.reparent`'s omitted-`to_node`
# default (task-3, S7) -- distinguishes "3rd argument not given at all" (the
# 2-arg C++ overload) from "3rd argument explicitly `None`" (the 3-arg C++
# overload's `to` meaning "through the last sibling", `_reparent_range`'s own
# `end=None` convention) -- the same sentinel shape as `reduct.py`'s `_NO_IT`
# and for the same reason: plain `None` cannot serve as the default here
# without conflating those two, semantically different, calls.
_NO_TO_NODE = object()


class tree_node:
    """C++: opencog/util/tree.h:102 tree_node_

    Links only; no head/feet sentinels (see module docstring). A node with
    `parent is None` is the tree's root.
    """

    __slots__ = ("data", "parent", "first_child", "last_child", "prev_sibling", "next_sibling")

    def __init__(self, data=None):
        self.data = data
        self.parent = None
        self.first_child = None
        self.last_child = None
        self.prev_sibling = None
        self.next_sibling = None


def _require_live_node(node: tree_node | None, operation: str) -> None:
    """Reject a retained handle whose node was destroyed by a tree splice."""
    if node is not None and (
        node.parent is _DEAD or node.prev_sibling is _DEAD or node.next_sibling is _DEAD
    ):
        raise ValueError(
            f"{operation}: cannot use an erased/detached node -- its own links are poisoned"
        )


def _require_live_subtree_source(node: tree_node, operation: str) -> None:
    """Reject an erased node before reading it as a subtree-copy source."""
    try:
        _require_live_node(node, operation)
    except ValueError as exc:
        raise ValueError(
            f"{operation}: cannot copy from an erased/detached node -- "
            "its own links are poisoned"
        ) from exc


class tree:
    """C++: opencog/util/tree.h:111 tree

    See the module docstring for what is, and is not, ported, and why.
    """

    def __init__(self, root_data=None):
        """C++: opencog/util/tree.h:646 tree (default ctor, empty tree); the
        root ctor is opencog/util/tree.h:651 tree(const T&). `root_data=None`
        means "start empty," mirroring the default-constructed C++ tree --
        this signature has no way to distinguish that from "a root whose
        data happens to be Python None," matching what the brief asks for.

        A third C++ overload is folded in here too:
        `explicit tree(const iterator_base&)` (tree.h:131, implemented
        tree.h:658 as `set_head(*other); replace(begin(), other);`) --
        deep-copies `other`'s entire subtree into this new, standalone
        tree. Added for S7 (Task 1's eval.py -- eval.cc's `combo_tree
        tr(it)`/`combo_tree cb_tr(f)` in the foldr/foldl cases -- and
        assumption.py's copy-on-insert), the same way Task 0 added the
        subtree-rooted post-order walk: a real, narrowly-scoped C++
        constructor this port was missing, discovered by a task that
        needs it. Collapsed into this one constructor rather than a
        second method, the same convention `boolean_interpreter.__call__`
        (interpreter.py) uses for overloaded C++ methods -- Python
        dispatches on `root_data`'s type instead of overloading. A
        `tree_node` is never itself a legal `combo_tree` root *value*
        (`vertex` is `builtin | argument | contin_t | str`, vertex.py), so
        this dispatch cannot misfire against a genuine plain-data root.

        FIX WAVE (task-1 review, Minor): a `root_data` whose own
        `parent`/`prev_sibling`/`next_sibling` were poisoned to `_DEAD` by
        `erase()`/`replace()`'s subtree-copy branch is rejected outright,
        rather than silently deep-copying it as if it were still live.
        Without this check, `tree(erased_node)` was a NEW way to read
        through a dead handle that neither of those two methods' own
        DIVERGENCE markers anticipated -- the whole point of poisoning
        those links is that a stale handle fails loudly at the first
        attempt to navigate through it, and this constructor was the one
        path that never tried to navigate through them at all (it only
        reads `.data`/`.first_child`, none of which `erase()`/`replace()`
        poison), so it slipped past silently. A genuine, live tree root
        (`parent is None`) is unaffected -- only `is _DEAD` specifically
        is rejected, not `is None`.
        """
        if isinstance(root_data, tree_node):
            _require_live_subtree_source(root_data, "tree()")
            self._root = tree_node(_copy_value(root_data.data))
            self._copy_descendants(self._root, root_data)
            return
        self._root = None if root_data is None else tree_node(root_data)

    def __class_getitem__(cls, item):
        """C++: opencog/util/tree.h:111 tree

        C++'s `tree` is a class template, so a specific element type is
        named by instantiating it: `tree<vertex>` (vertex.h:527's
        `combo_tree` typedef). This class carries no runtime type
        parameter of its own -- like the C++ template, it is duck-typed
        over whatever `.data` each tree_node happens to hold -- so the
        subscript is accepted purely so a consumer can write `tree[vertex]`
        (task-3-brief.md's `combo_tree = tree[vertex]`) and have it read as
        that same instantiation, without this module needing to become
        Generic[T] to support it.
        """
        return cls

    # ---- traversal bounds -------------------------------------------------

    def begin(self):
        """C++: opencog/util/tree.h:830 begin"""
        return self._root

    def end(self):
        """C++: opencog/util/tree.h:836 end"""
        return None

    def begin_leaf(self):
        """C++: opencog/util/tree.h:930 begin_leaf"""
        node = self._root
        if node is not None:
            while node.first_child is not None:
                node = node.first_child
        return node

    def end_leaf(self):
        """C++: opencog/util/tree.h:942 end_leaf"""
        return None

    def begin_post(self):
        """C++: opencog/util/tree.h:856 begin_post"""
        node = self._root
        if node is not None:
            while node.first_child is not None:
                node = node.first_child
        return node

    def end_post(self):
        """C++: opencog/util/tree.h:868 end_post"""
        return None

    # ---- pre/post/leaf stepping (private: no per-iterator state needed) ---

    def _skip_children_successor(self, node):
        """Pre-order successor of `node` with its own children skipped --
        i.e. the first node after `node`'s entire subtree. This is exactly
        the bound the sibling_iterator->pre_order_iterator "end of range"
        conversion computes in C++ (used, e.g., by equal_subtree's walk).
        C++: opencog/util/tree.h:2411 pre_order_iterator
        """
        n = node
        while n is not None and n.next_sibling is None:
            n = n.parent
        return n.next_sibling if n is not None else None

    def _preorder_step(self, node):
        """C++: opencog/util/tree.h:2427 pre_order_iterator (operator++)"""
        if node.first_child is not None:
            return node.first_child
        return self._skip_children_successor(node)

    def _postorder_step(self, node):
        """C++: opencog/util/tree.h:2535 post_order_iterator (operator++)"""
        if node.next_sibling is None:
            return node.parent
        nxt = node.next_sibling
        while nxt.first_child is not None:
            nxt = nxt.first_child
        return nxt

    def _leaf_step(self, node):
        """C++: opencog/util/tree.h:3053 leaf_iterator (operator++)

        C++ relies on the head/feet sentinels to give even the tree's real
        root a non-null next_sibling (the dummy `feet` node), so climbing
        the parent chain always finds a next_sibling before parent runs
        out, and the walk terminates by landing on `feet` == end_leaf().
        This port has no sentinels (see module docstring), so climbing that
        reaches a node with neither a next_sibling nor a parent -- the
        actual root, already exhausted -- returns None (this port's
        end_leaf()) directly instead.
        """
        n = node
        while n.next_sibling is None:
            if n.parent is None:
                return None
            n = n.parent
        n = n.next_sibling
        while n.first_child is not None:
            n = n.first_child
        return n

    def __iter__(self):
        """C++: opencog/util/tree.h:2427 pre_order_iterator (the operator++
        that begin()/end() bound, driving the whole-tree walk)"""
        node = self._root
        while node is not None:
            yield node
            node = self._preorder_step(node)

    def iter_post(self):
        """C++: opencog/util/tree.h:2535 post_order_iterator

        Post-order traversal generator. Not itself in the brief's literal
        API-name list. begin_post() / end_post() above mirror the C++
        signatures exactly (a single boundary node each), but C++'s
        post_order_iterator carries its own operator++ as a method OF THE
        ITERATOR OBJECT -- a place for per-step advance state this port
        deliberately has no equivalent of (see module docstring). Without
        *some* way to advance, begin_post()/end_post() could only ever
        identify the two ends of a walk, never enumerate it -- which would
        make the "post-order traversal order" case
        tools/cpp_probe/tree_probe.cc is required to cover untestable
        through the public API. Added for exactly that reason; flagged
        here and in the task report rather than silently.
        """
        node = self.begin_post()
        end = self.end_post()
        while node is not end:
            yield node
            node = self._postorder_step(node)

    def iter_post_at(self, node):
        """C++: opencog/util/tree.h:2610 post_order_iterator::descend_all

        A post-order walk rooted at an arbitrary SUBTREE rather than the
        whole tree -- `iter_post()` only ever starts from `begin_post()`/
        `end_post()`, the tree's own root. `reduct::upwards`
        (moses/comboreduct/reduct/meta_rules.cc:111) needs exactly this:
        `post_order_iterator at=it, end=it; ++end; at.descend_all(); for
        (;at!=end;++at) ...`. `at` starts at `node`'s own
        leftmost-deepest descendant (`descend_all`, tree.h:2610 -- follow
        `first_child` until there is none) and steps with the ordinary
        post-order `operator++` (`_postorder_step`) until it reaches
        `end`: `node`'s own post-order successor, computed by stepping
        ONCE from `node` itself, UNDESCENDED (`++end` on a copy still
        sitting at `node`, before `at.descend_all()` touches the other
        copy) -- i.e. exactly the node that would follow `node` in a
        post-order walk of the WHOLE enclosing tree, continued past
        `node`. A node's post-order position is fixed by its own
        descendants alone, regardless of what encloses it, so walking
        from `node`'s leftmost-deepest descendant up to that successor
        visits precisely `node`'s own subtree, in standard post-order,
        ending on `node` itself -- never touching anything outside it.

        Restores the same `node is not None` guard `children()` restores
        for `tree.h:914`'s `tree_assert(pos.node != 0)` -- split into this
        eager-checking wrapper plus `_iter_post_at_iter` for the identical
        reason `children()`/`_children_iter()` are split (see that
        docstring): a generator's body does not run until the first
        `next()`, so the check must live outside it to raise on the call
        itself rather than on first iteration.
        """
        if node is None:
            raise ValueError("iter_post_at: node must not be None")
        _require_live_node(node, "iter_post_at")
        return self._iter_post_at_iter(node)

    def _iter_post_at_iter(self, node):
        end = self._postorder_step(node)
        cur = node
        while cur.first_child is not None:
            cur = cur.first_child
        while cur is not end:
            yield cur
            cur = self._postorder_step(cur)

    def iter_pre_at(self, node):
        """C++: opencog/util/tree.h:2373 iterator_base::skip_children

        A pre-order walk rooted at an arbitrary SUBTREE rather than the
        whole tree -- `__iter__` only ever starts from `begin()`, the
        tree's own root. `reduct::downwards`
        (moses/comboreduct/reduct/meta_rules.cc:79-105) needs exactly this:
        `iterator end = it; end.skip_children(); ++end; for (; it != end;
        ++it) ...` -- `end` is `it`'s own post-subtree successor
        (`_skip_children_successor`, computed ONCE up front, before `it`
        has moved at all: climb from `it` through ancestors until one has
        a `next_sibling`, and land on that), and the walk visits `it`
        itself FIRST (pre-order: parent before children), then descends
        via ordinary pre-order stepping (`_preorder_step`) until it
        reaches `end`. Added this task (Task 2, S7) as the pre-order
        counterpart of `iter_post_at` (Task 0, S7) -- `downwards` needed
        it and no public API offered a subtree-bounded pre-order walk,
        exactly the gap `iter_post_at`'s own docstring already explains
        for the post-order case.

        Like `children()`/`iter_post_at`, this is a LIVE walk, not a
        snapshot: `_preorder_step` re-reads `first_child`/`next_sibling`/
        `parent` at the moment each step actually runs, so a caller that
        mutates the tree BETWEEN two `next()` calls on this generator (as
        `downwards::operator()` does -- it applies its wrapped rule at
        each node before advancing) sees the walk continue over the
        POST-mutation structure, including newly inserted nodes that fall
        within `node`'s remaining, not-yet-visited range. Measured against
        the real library (`tools/cpp_probe/meta_rules_probe.cc`, the
        "downwards(live-mutation insert-after)" case): inserting a new
        sibling immediately after the current node mid-walk gets visited
        by that SAME walk. A materialised `list(t.iter_pre_at(node))`
        upfront would silently lose that -- deliberately not done here or
        at `downwards`'s own call site (meta_rules.py).

        Restores the same `node is not None` guard `iter_post_at` restores
        for `tree.h:914`'s `tree_assert(pos.node != 0)`, split into this
        eager-checking wrapper plus `_iter_pre_at_iter` for the identical
        reason (a generator's body does not run until the first `next()`).
        """
        if node is None:
            raise ValueError("iter_pre_at: node must not be None")
        _require_live_node(node, "iter_pre_at")
        return self._iter_pre_at_iter(node)

    def _iter_pre_at_iter(self, node):
        end = self._skip_children_successor(node)
        cur = node
        while cur is not end:
            yield cur
            cur = self._preorder_step(cur)

    def next_pre_order(self, node):
        """C++: opencog/util/tree.h:2427 pre_order_iterator (operator++)

        A single pre-order advance from an arbitrary node -- public
        wrapper around `_preorder_step` (used internally by `__iter__`
        and `iter_pre_at`). Added for `comboreduct/reduct/logical_rules.
        py`'s `reduce_remove_subtree_equal_tt` (logical_rules.cc:852's
        `++pit`, a single explicit pre-order step) -- the first real
        external caller that needs one bare step rather than a whole
        generator walk or a private implementation detail. That rule's
        walk is bounded by a subtree end, but computing the bound is
        `subtree_pre_order_end`'s job (below), not this method's -- this
        one is called only for the per-node advance in between.
        """
        if node is None:
            raise ValueError("next_pre_order: node must not be None")
        _require_live_node(node, "next_pre_order")
        return self._preorder_step(node)

    def subtree_pre_order_end(self, node):
        """C++: opencog/util/tree.h:2412 pre_order_iterator(const sibling_iterator&)

        Public wrapper around `_skip_children_successor` -- the pre-order
        node immediately following `node`'s own entire subtree, i.e. what
        `it.end()` (a `sibling_iterator` bounding `node`'s CHILD range)
        becomes once converted to a `pre_it`/`pre_order_iterator` in C++.
        That conversion (tree.h:2412-2423) is easy to misread as "keeps
        only the underlying node pointer, which is null for `end()`" --
        it does not: when the incoming node pointer is null it falls back
        to `other.range_last()` (`node`'s own last child, or `node`
        itself if childless), calls `skip_children()`, then steps once --
        which lands EXACTLY on `_skip_children_successor(node)`, a real,
        non-null boundary in every case except "node is the last node in
        the whole tree in pre-order" (where it is `None`, matching
        `_skip_children_successor`'s own `None` return for that case).

        Added for `comboreduct/reduct/logical_rules.py`'s
        `reduce_remove_subtree_equal_tt` (`for (pre_it pit = it.begin();
        pit != it.end();)`, logical_rules.cc:844): that rule's walk over
        `it`'s children IS bounded to `it`'s own subtree, via this exact
        mechanism -- a Task 5 fix-round correction of an earlier claim
        (in this rule's own docstring) that the walk was unbounded. See
        that rule's own docstring for the full trace, including a bounded
        probe against the real compiled `libcomboreduct.so` pinning both
        the wrong and the corrected behaviour on the same input tree.
        """
        if node is None:
            raise ValueError("subtree_pre_order_end: node must not be None")
        _require_live_node(node, "subtree_pre_order_end")
        return self._skip_children_successor(node)

    def iter_leaf(self):
        """C++: opencog/util/tree.h:3023 leaf_iterator

        Leaf traversal generator -- see iter_post()'s docstring for why
        this exists alongside begin_leaf()/end_leaf() rather than
        replacing them.
        """
        node = self.begin_leaf()
        end = self.end_leaf()
        while node is not end:
            yield node
            node = self._leaf_step(node)

    # ---- size / navigation -------------------------------------------------

    def size(self):
        """C++: opencog/util/tree.h:1979 size"""
        return sum(1 for _ in self)

    def empty(self):
        """C++: opencog/util/tree.h:1999 empty"""
        return self._root is None

    def is_valid(self, node):
        """C++: opencog/util/tree.h:2166 is_valid"""
        return node is not None and not (
            node.parent is _DEAD
            or node.prev_sibling is _DEAD
            or node.next_sibling is _DEAD
        )

    def child(self, node, i):
        """C++: opencog/util/tree.h:2215 child

        C++'s `tree_assert(tmp != 0)` inside the walk only ever fires one
        step *past* the true out-of-range boundary (an artifact of the
        post-decrementing `while(num--)` loop: asking for exactly
        `number_of_children(node)` silently returns null in C++, with no
        assert at all). Restored here as a full, symmetric bounds check
        instead of reproducing that gap -- raise for any `i` that is not a
        real child index, which is strictly more of what the "raise rather
        than assert-away" instruction asks for than the letter of the
        C++ loop.
        """
        _require_live_node(node, "child")
        if i < 0:
            raise IndexError(f"child: index {i} out of range")
        n = node.first_child
        for _ in range(i):
            if n is None:
                raise IndexError(f"child: index {i} out of range")
            n = n.next_sibling
        if n is None:
            raise IndexError(f"child: index {i} out of range")
        return n

    def children(self, node):
        """C++: opencog/util/tree.h:910 begin

        The sibling range tree::begin(pos)/end(pos) bound. A plain
        generator over live links -- deliberately not a snapshot: see the
        module docstring / task report on preserving the
        mutate-while-iterating hazard.

        Restores `tree.h:914`'s `tree_assert(pos.node != 0)`. Split into
        this eager-checking wrapper plus `_children_iter` rather than
        putting the check inline in the generator body: a generator's body
        does not run at all until the first `next()`, so a bare `if node is
        None: raise` inside one would let `t.children(None)` return
        successfully and only blow up later, on first iteration --
        inconsistent with every other guard in this module, which raises
        on the call itself.
        """
        if node is None:
            raise ValueError("children: node must not be None")
        _require_live_node(node, "children")
        return self._children_iter(node)

    def _children_iter(self, node):
        child = node.first_child
        while child is not None:
            yield child
            child = child.next_sibling

    def number_of_children(self, node):
        """C++: opencog/util/tree.h:2019 number_of_children"""
        _require_live_node(node, "number_of_children")
        n = 0
        child = node.first_child
        while child is not None:
            n += 1
            child = child.next_sibling
        return n

    def is_childless(self, node):
        """C++: opencog/util/tree.h:170 sibling_iterator::is_childless

        Added task-3 (S7) for `comboreduct/reduct/general_rules.py`'s
        `eval_constants` (`it.is_childless()`).
        """
        _require_live_node(node, "is_childless")
        return node.first_child is None

    def has_one_child(self, node):
        """C++: opencog/util/tree.h:171 sibling_iterator::has_one_child

        `node->first_child != NULL && node->first_child == node->last_child`
        -- ported as identity (`is`), matching C++'s raw-pointer `==` here.
        Added task-3 (S7) for the same call site as `is_childless` above.
        """
        _require_live_node(node, "has_one_child")
        return node.first_child is not None and node.first_child is node.last_child

    def number_of_siblings(self, node):
        """C++: opencog/util/tree.h:2035 number_of_siblings"""
        _require_live_node(node, "number_of_siblings")
        n = 0
        pos = node.next_sibling
        while pos is not None:
            n += 1
            pos = pos.next_sibling
        pos = node.prev_sibling
        while pos is not None:
            n += 1
            pos = pos.prev_sibling
        return n

    def sibling_index(self, node):
        """C++: opencog/util/tree.h:2193 sibling_index

        Added task-4 (S7) for `comboreduct/reduct/logical_rules.py`'s
        `reduce_and_assumptions` (logical_rules.cc:252's
        `tr.sibling_index(sib)`). No head/feet sentinels here (module
        docstring) -- C++'s root-vs-non-root branch exists only to walk
        through to its own artificial head sentinel when `node` IS the
        root; since a root has no siblings at all in this port
        (single-rooted, forests unsupported), the walk below (stopping at
        `prev_sibling is None`) already gives the right answer -- 0 -- for
        the root too, without needing a separate branch.
        """
        _require_live_node(node, "sibling_index")
        ind = 0
        pos = node
        while pos.prev_sibling is not None:
            pos = pos.prev_sibling
            ind += 1
        return ind

    def next_sibling(self, node):
        """C++: opencog/util/tree.h:974 next_sibling

        Restores `tree.h:978`'s `tree_assert(position.node != 0)`.
        """
        if node is None:
            raise ValueError("next_sibling: node must not be None")
        _require_live_node(node, "next_sibling")
        return node.next_sibling

    def previous_sibling(self, node):
        """C++: opencog/util/tree.h:964 previous_sibling

        Restores `tree.h:968`'s `tree_assert(position.node != 0)`.
        """
        if node is None:
            raise ValueError("previous_sibling: node must not be None")
        _require_live_node(node, "previous_sibling")
        return node.prev_sibling

    def max_depth(self, node=None):
        """C++: opencog/util/tree.h:538 max_depth"""
        if node is None:
            node = self._root
        _require_live_node(node, "max_depth")
        d = -1
        for child in self.children(node):
            depth = self.max_depth(child)
            if depth > d:
                d = depth
        return d + 1

    def count(self, node):
        """C++: opencog/util/tree.h:1991 subtree_size

        The brief's `count(node)` -- no `tree<T>::count` exists anywhere in
        tree.h; the only method matching "take a node, return the number of
        nodes in its subtree" is `subtree_size`. See the task report for
        how this reading was reached.
        """
        _require_live_node(node, "count")
        total = 1
        for child in self.children(node):
            total += self.count(child)
        return total

    def sort(self, node, key):
        """C++: opencog/util/tree.h:1781 sort (StrictWeakOrdering overload)

        Scoped to sorting ALL of `node`'s children (from=begin(node),
        to=end(node)) using a Python `key` function rather than a
        comparator, matching the brief's `sort(node, key)` signature --
        C++'s general form also takes an arbitrary sibling sub-range and a
        two-argument comparator (`sort(from, to, comp, deep=False)` at
        tree.h:1773/1781), neither of which is in the brief's API list.
        Because the range is always a node's *entire* child list, the
        "prev/next node outside the sorted range" bookkeeping that the
        general C++ version needs (it reconnects to real siblings before
        and after an arbitrary sub-range) collapses to None/None here,
        simplifying the multiset-based reassembly to a direct relink.
        `key` is applied to each child's `.data` (matching C++'s own
        `compare_nodes`, which compares `a->data`/`b->data`, never the
        node pointers themselves). `sorted()` is stable, matching the C++
        comment's expectation that multiset preserves insertion order for
        equal keys (deep=False: children move as whole subtrees,
        grandchildren are not sorted).
        """
        _require_live_node(node, "sort")
        children = list(self.children(node))
        if not children:
            return
        ordered = sorted(children, key=lambda child: key(child.data))
        self._relink_full_children(node, ordered)

    def _relink_full_children(self, node, ordered):
        """No direct C++ symbol: the direct-relink reassembly `sort`'s own
        docstring above describes (the C++ `sort_on_subtrees`/`sort`
        multiset-based reassembly collapsed to this, since the range is
        always a node's *entire* child list here -- see that docstring
        for the full reasoning). Shared by `sort`, `partition` and
        `sort_on_subtrees` (task-3, S7): `ordered` is a non-empty list of
        `node`'s own children (already validated as such by the caller),
        in the new desired order; this rewires `first_child`/
        `last_child`/`prev_sibling`/`next_sibling` to match.
        """
        node.first_child = ordered[0]
        node.last_child = ordered[-1]
        prev = None
        for child in ordered:
            child.prev_sibling = prev
            if prev is not None:
                prev.next_sibling = child
            prev = child
        prev.next_sibling = None

    def partition(self, node, pred):
        """C++: opencog/util/tree.h:1843 tree::partition

        Scoped to partitioning ALL of `node`'s children (from=begin(node),
        to=end(node)) -- like this file's own `sort` above, C++'s general
        form also takes an arbitrary sibling sub-range; this port's one
        real caller (`comboreduct/reduct/general_rules.py`'s
        `eval_constants`, task-3) always passes the full range
        (`tr.partition(it.begin(), it.end(), is_constant<vertex>)`).

        NOT a stable partition -- measured directly
        (`tools/cpp_probe/partition_probe.cc`), not assumed: the real
        algorithm is `std::partition`'s own two-pointer forward/backward
        swap scan, which can and does reorder elements WITHIN each of the
        two output groups relative to their original order. A stable
        "gather the matches, then the non-matches, each in original
        order" implementation is a DIFFERENT function and would silently
        diverge from C++ whenever the input interleaves matches and
        non-matches (measured: children `[C1,V1,C2,V2,C3]` with `pred`
        true only on the `C*` ones partitions to `[C1,C3,C2,V2,V1]`, NOT
        the stable `[C1,C2,C3,V1,V2]`).

        Ported as a position-index simulation of the real scan: C++'s
        `this->swap(to, from)` (physically exchange the two nodes'
        positions) followed immediately by `std::swap(to, from)`
        (exchange which *iterator variable* refers to which node) is,
        net, equivalent -- for the purpose of tracking `from`/`to` as
        plain positions -- to swapping the two positions' CONTENTS and
        leaving the position variables themselves unchanged; both index
        variables below (`lo`, `hi`) are therefore never swapped, only
        `children[lo]`/`children[hi]` are.

        Returns the boundary node -- the first child, in the FINAL order,
        for which `pred` is false -- or `None` if every child satisfies
        `pred` (C++'s `to` sentinel, `end()`).
        """
        _require_live_node(node, "partition")
        children = list(self.children(node))
        n = len(children)
        if n == 0:
            return None
        lo, hi = 0, n
        while True:
            while lo != hi and pred(children[lo].data):
                lo += 1
            while lo != hi:
                hi -= 1
                if pred(children[hi].data):
                    break
            if lo == hi:
                break
            children[lo], children[hi] = children[hi], children[lo]
        self._relink_full_children(node, children)
        return None if lo == n else children[lo]

    def sort_on_subtrees(self, node, comp, deep=False):
        """C++: opencog/util/tree.h:1858 tree::sort_on_subtrees

        Scoped to sorting ALL of `node`'s children (from=begin(node),
        to=end(node)), the same scoping decision `sort` above already
        makes and for the same reason: this port's one real caller
        (`comboreduct/reduct/general_rules.py`'s `reorder_commutative`,
        task-3) always passes the full range (`tr.sort_on_subtrees(
        it.begin(), it.end(), lexicographic_subtree_order<vertex>(),
        false)`). Unlike `sort`, `comp` compares two whole subtrees
        (`tree_node`s -- e.g. vertex.py's `lexicographic_subtree_less`),
        not `.data` alone: C++'s `StrictWeakOrdering` here is
        instantiated against a `sibling_iterator`, never against `T`
        itself.

        The real algorithm reassembles the sorted order via a
        `std::multiset<tree_node*, ...>` keyed on `comp` (tree.h:1861);
        this port's own real caller only ever uses a `comp` that is a
        genuine STRICT TOTAL order over distinct subtrees (ties occur
        only between two STRUCTURALLY IDENTICAL subtrees -- verified
        against `lexicographic_subtree_order`'s own definition), for
        which any correct sort reproduces the same output regardless of
        stability. `sorted()` (Python's stable sort) is used regardless,
        matching a real `std::multiset`'s own insertion-order-preserving
        behaviour for equivalent keys -- the C++ code's own comment
        flags this as unverified in the abstract ("CHECK: if multiset
        stores equivalent nodes in the order in which they are inserted,
        then this routine should be called 'stable_sort'"), but real
        libstdc++ does exactly that (measured:
        `tools/cpp_probe/partition_probe.cc`'s `sort_on_subtrees_ties`
        case preserves original relative order between two same-length,
        hence tied, children).

        `deep=True` recurses into each child's OWN children FIRST,
        matching the C++ implementation's stated order ("sort the
        children of each node too, *first*", tree.h:1865).
        """
        _require_live_node(node, "sort_on_subtrees")
        if deep:
            for child in list(self.children(node)):
                self.sort_on_subtrees(child, comp, True)
        children = list(self.children(node))
        if not children:
            return

        ordered = sorted(children, key=functools.cmp_to_key(three_way_comparator(comp)))
        self._relink_full_children(node, ordered)

    def equal_subtree(self, a, b, predicate=None):
        """C++: opencog/util/tree.h:1923 equal_subtree

        Ports the literal algorithm (tree.h:1931 equal(), tree.h:1949
        equal_subtree(..., BinaryPredicate)) rather than a "cleaner"
        recursive structural-equality check: this walks both subtrees in
        lockstep pre-order, comparing data and immediate children-counts at
        each step, bounded on the `a` side by `_skip_children_successor`
        and on the `b` side only by "not run out of nodes." For two
        genuinely equal-shaped subtrees both walks land on their true
        boundary simultaneously (the common, correct case); for mismatched
        shapes the `b`-side walk can in principle read past `b`'s own
        subtree into whatever follows it in `b`'s tree, exactly as the C++
        can -- an accepted, inherited imprecision of the original
        algorithm, not something this port corrects.

        ``predicate`` mirrors C++'s two-overload split: the no-argument
        `equal_subtree(iter, iter)` (tree.h:1923) defaults its comparator to
        `std::equal_to<T>` (tree.h:1926), and the `BinaryPredicate` overload
        (tree.h:1949) lets a caller supply another one. This module is
        generic over `T` (module docstring) and stays that way by defaulting
        `predicate` to plain `==`/`!=` rather than importing any one `T`'s
        notion of equality -- but for this port's actual `T` (`vertex`, a
        `boost::variant`), C++'s `std::equal_to<T>` is NOT bare `==`: a
        `boost::variant`'s compiler-generated `operator==` compares which
        alternative is held before ever comparing a value, so it is
        unconditionally false across alternatives. Python's bare `==`/`!=`
        on `.data` has no such discrimination (`contin_t` is a bare `float`,
        `builtin` an `IntEnum`, so e.g. `1.0 == builtin.logical_true` is
        `True`) -- callers holding vertex data of unproven type MUST pass
        `vertex_equal` (comboreduct/combo/vertex.py) here to get the real
        C++ default's actual behaviour; the plain default below is only
        correct for callers whose data's `==` already discriminates by type.
        """
        _require_live_node(a, "equal_subtree")
        _require_live_node(b, "equal_subtree")
        if predicate is None:
            def predicate(x, y):
                return x == y
        if a is b:
            return True
        if not predicate(a.data, b.data):
            return False
        if self.number_of_children(a) != self.number_of_children(b):
            return False
        boundary = self._skip_children_successor(a)
        one, three = a.first_child, b.first_child
        while one is not None and one is not boundary and three is not None:
            if not predicate(one.data, three.data):
                return False
            if self.number_of_children(one) != self.number_of_children(three):
                return False
            one = self._preorder_step(one)
            three = self._preorder_step(three)
        return True

    def validate(self, node=None):
        """C++: opencog/util/tree.h:2175 validate

        Added task-4 (S7) for `comboreduct/reduct/logical_rules.py`'s
        `subtree_to_enf::reduce_to_enf::reduce_and` (logical_rules.cc:
        693-694's unconditional `tr.validate(child); tr.validate();` --
        NOT inside that function's `#if DEBUG` guard, so it runs on every
        real call, not just debug builds).

        A pure consistency CHECK, no side effects on the tree: raises
        `AssertionError` (this port's uniform stand-in for `tree_assert`,
        matching every other restored-assert in this module) if `node`'s
        own parent/sibling links disagree with what its neighbours report,
        then recurses into every child. `node` omitted (C++'s no-arg
        `validate()` overload, tree.h:536: `validate(begin())`) validates
        the whole tree from its root; an empty tree is a silent no-op,
        matching `begin()` being a harmless no-op start point for an empty
        C++ tree too.

        Same blind spot as real C++ tree.h:2175: recursion reaches
        children via `first_child`/`next_sibling` forward traversal, so a
        corrupted `first_child` pointer silently orphans everything below
        it from this very walk rather than being detected (`tests/util/
        test_tree.py`'s own, stronger, test-only `validate()` helper
        deliberately does not share this limitation -- see its docstring).
        """
        if node is None:
            node = self._root
            if node is None:
                return
        _require_live_node(node, "validate")
        if node.parent is not None:
            if node.prev_sibling is None and node.parent.first_child is not node:
                raise AssertionError("validate: parent.first_child mismatch")
            if node.next_sibling is None and node.parent.last_child is not node:
                raise AssertionError("validate: parent.last_child mismatch")
        if node.prev_sibling is not None and node.prev_sibling.next_sibling is not node:
            raise AssertionError("validate: prev_sibling.next_sibling mismatch")
        if node.next_sibling is not None and node.next_sibling.prev_sibling is not node:
            raise AssertionError("validate: next_sibling.prev_sibling mismatch")
        child = node.first_child
        while child is not None:
            self.validate(child)
            child = child.next_sibling

    # ---- internal splice helpers -------------------------------------------

    def _append_child_leaf(self, parent, data):
        """Raw last-child append, no dispatch on `data`'s type.
        C++: opencog/util/tree.h:1091 append_child
        """
        node = tree_node(data)
        node.parent = parent
        if parent.last_child is not None:
            parent.last_child.next_sibling = node
        else:
            parent.first_child = node
        node.prev_sibling = parent.last_child
        parent.last_child = node
        node.next_sibling = None
        return node

    @classmethod
    def pruned_copy(cls, node, prune):
        """No C++ counterpart: `tree(node)` (a standalone deep copy of `node`'s
        subtree) that leaves out every proper descendant `d` for which
        `prune(d)` is true, together with `d`'s own subtree. `prune` is asked
        once about each child of each kept node, and never about `node`
        itself.
        """
        _require_live_subtree_source(node, "pruned_copy")
        out = cls()
        out._root = tree_node(_copy_value(node.data))
        append = out._append_child_leaf
        stack = [(node, out._root)]
        while stack:
            src, dst = stack.pop()
            kept = []
            child = src.first_child
            while child is not None:
                if not prune(child):
                    kept.append((child, append(dst, _copy_value(child.data))))
                child = child.next_sibling
            stack.extend(reversed(kept))
        return out

    def _copy_descendants(self, dest_root, source_root):
        """Deep-copy source_root's descendants onto dest_root, whose own
        .data the caller has already set. Bounded to exactly source_root's
        own subtree via `last = source_root.next_sibling` (source_root's
        real next sibling, if any -- NOT a "climb up" successor).
        C++: opencog/util/tree.h:1397 replace(iter, const iterator_base&)
        """
        last = source_root.next_sibling
        current_from = source_root
        toit = dest_root
        while True:
            if current_from.first_child is not None:
                current_from = current_from.first_child
                toit = self._append_child_leaf(toit, _copy_value(current_from.data))
            else:
                while current_from.next_sibling is None and current_from is not source_root:
                    current_from = current_from.parent
                    toit = toit.parent
                current_from = current_from.next_sibling
                if current_from is not last:
                    toit = self._append_child_leaf(toit.parent, _copy_value(current_from.data))
            if current_from is last:
                break
        return dest_root

    def _reparent_range(self, position, first, end):
        """Move the sibling range [first, end) to become children of
        `position` (end excluded, may be None for "through the last
        sibling").
        C++: opencog/util/tree.h:1525 reparent

        Restores `tree.h:1531`'s `tree_assert(first != position.node)`:
        `position` must not be the very first node of the range being
        moved -- that specific self-containment (reparenting a node's
        children onto one of those same children) would relink `first`'s
        own prev/next pointers into a self-referential cycle. Raises
        rather than silently detaching the subtree.
        """
        if first is position:
            raise ValueError(
                "reparent: new_parent cannot be from_node's own first "
                "child -- that would reparent a node's children onto one "
                "of those very children"
            )
        if first is end:
            return first
        last = first
        n = first.next_sibling
        while n is not end:
            last = n
            n = n.next_sibling
        if first.prev_sibling is None:
            first.parent.first_child = last.next_sibling
        else:
            first.prev_sibling.next_sibling = last.next_sibling
        if last.next_sibling is None:
            last.parent.last_child = first.prev_sibling
        else:
            last.next_sibling.prev_sibling = first.prev_sibling
        if position.first_child is None:
            position.first_child = first
            position.last_child = last
            first.prev_sibling = None
        else:
            position.last_child.next_sibling = first
            first.prev_sibling = position.last_child
            position.last_child = last
        last.next_sibling = None
        n = first
        while True:
            n.parent = position
            if n is last:
                break
            n = n.next_sibling
        return first

    # ---- public splice operations ------------------------------------------

    def set_head(self, data):
        """C++: opencog/util/tree.h:1239 set_head

        Insert the sole top-level node into an otherwise empty tree. C++
        copies ``T`` by value; ``_copy_value`` matters for mutable Combo
        ``argument`` payloads.
        """
        if not self.empty():
            # DIVERGENCE(C++ opencog/util/tree.h:1239-1242): classic raises
            # tree_assert when set_head is called on a nonempty tree; Python
            # raises ValueError and guarantees the existing root is untouched.
            raise ValueError("set_head: tree must be empty")
        self._root = tree_node(_copy_value(data))
        return self._root

    def append_child(self, node, data_or_subtree):
        """C++: opencog/util/tree.h:1091 append_child (plain data); the
        subtree-copy overload is opencog/util/tree.h:1183
        append_child(iter, iter), itself implemented via tree.h:1397
        replace(iter, const iterator_base&)'s copy loop -- ported here as
        `_copy_descendants`.

        Restores `tree.h:1097-1098`'s `tree_assert(position.node != head);
        tree_assert(position.node);` (the plain-data overload) and
        `tree.h:1185-1186`'s identical pair on the subtree-copy overload --
        one check here covers both, since Python dispatches on
        `data_or_subtree`'s type rather than overloading.
        """
        if node is None:
            raise ValueError("append_child: node must not be None")
        _require_live_node(node, "append_child")
        if isinstance(data_or_subtree, tree_node):
            _require_live_subtree_source(data_or_subtree, "append_child")
            new_child = self._append_child_leaf(node, _copy_value(data_or_subtree.data))
            self._copy_descendants(new_child, data_or_subtree)
            return new_child
        return self._append_child_leaf(node, data_or_subtree)

    def prepend_child(self, node, data):
        """C++: opencog/util/tree.h:1119 prepend_child

        Plain data only, matching the brief's signature -- unlike
        append_child, prepend_child's subtree-copy overload
        (tree.h:1194) is not in the brief's API list.

        Restores `tree.h:1121-1122`'s `tree_assert(position.node != head);
        tree_assert(position.node);`.
        """
        if node is None:
            raise ValueError("prepend_child: node must not be None")
        _require_live_node(node, "prepend_child")
        new_node = tree_node(data)
        new_node.parent = node
        if node.first_child is not None:
            node.first_child.prev_sibling = new_node
        else:
            node.last_child = new_node
        new_node.next_sibling = node.first_child
        node.first_child = new_node
        new_node.prev_sibling = None
        return new_node

    def append_children(self, node, iterable):
        """C++: opencog/util/tree.h:1162 append_children (initializer_list
        overload)

        The initializer_list overload we port has no `tree_assert` of its
        own in C++ -- it delegates per-element to `append_child`
        (tree.h:1097-1098), which does. That leaves one gap C++ itself
        does not close: an empty `iterable` skips the loop entirely and
        falls straight through to `position.node->last_child`, a null
        deref with nothing guarding it. Guarded directly here instead of
        only inheriting the per-element check, so `append_children(None,
        [])` raises the same way `append_children(None, [x])` already
        would via the delegated call.
        """
        if node is None:
            raise ValueError("append_children: node must not be None")
        _require_live_node(node, "append_children")
        for data in iterable:
            self.append_child(node, data)
        return node.last_child

    def insert(self, node, data):
        """C++: opencog/util/tree.h:1247 insert (previous sibling)

        See the module-level DIVERGENCE note: raises if `node` is the
        tree's own root, since C++'s head/feet sentinels (not ported here)
        are what let this splice give even the root a new sibling.
        """
        _require_live_node(node, "insert")
        if node is self._root:
            raise ValueError(
                "insert: cannot insert a previous sibling of the tree root -- "
                "forests of multiple roots are out of scope for this port"
            )
        new_node = tree_node(data)
        new_node.parent = node.parent
        new_node.next_sibling = node
        new_node.prev_sibling = node.prev_sibling
        node.prev_sibling = new_node
        if new_node.prev_sibling is None:
            new_node.parent.first_child = new_node
        else:
            new_node.prev_sibling.next_sibling = new_node
        return new_node

    def insert_after(self, node, data):
        """C++: opencog/util/tree.h:1301 insert_after

        See the module-level DIVERGENCE note: raises if `node` is the
        tree's own root, for the same reason as `insert`.
        """
        _require_live_node(node, "insert_after")
        if node is self._root:
            raise ValueError(
                "insert_after: cannot insert a next sibling of the tree root -- "
                "forests of multiple roots are out of scope for this port"
            )
        new_node = tree_node(data)
        new_node.parent = node.parent
        new_node.prev_sibling = node
        new_node.next_sibling = node.next_sibling
        node.next_sibling = new_node
        if new_node.next_sibling is None:
            new_node.parent.last_child = new_node
        else:
            new_node.next_sibling.prev_sibling = new_node
        return new_node

    def insert_subtree(self, node, subtree):
        """C++: opencog/util/tree.h:1358 insert_subtree"""
        _require_live_subtree_source(subtree, "insert_subtree")
        dummy = self.insert(node, _copy_value(subtree.data))
        self._copy_descendants(dummy, subtree)
        return dummy

    def insert_subtree_after(self, node, subtree):
        """C++: opencog/util/tree.h:1368 insert_subtree_after"""
        _require_live_subtree_source(subtree, "insert_subtree_after")
        new_node = self.insert_after(node, _copy_value(subtree.data))
        self._copy_descendants(new_node, subtree)
        return new_node

    def insert_above(self, node, data):
        """C++: opencog/util/tree.h:1324 insert_above

        Well-defined even when `node` is the tree's own root (replaces the
        root in place rather than adding a sibling of it), unlike
        `insert`/`insert_after` -- so `self._root` is updated here when
        needed.
        """
        _require_live_node(node, "insert_above")
        new_node = tree_node(data)
        new_node.first_child = node
        new_node.last_child = node
        new_node.parent = node.parent
        new_node.prev_sibling = node.prev_sibling
        new_node.next_sibling = node.next_sibling
        if new_node.prev_sibling is not None:
            new_node.prev_sibling.next_sibling = new_node
        elif new_node.parent is not None:
            new_node.parent.first_child = new_node
        if new_node.next_sibling is not None:
            new_node.next_sibling.prev_sibling = new_node
        elif new_node.parent is not None:
            new_node.parent.last_child = new_node
        if node is self._root:
            self._root = new_node
        node.parent = new_node
        node.prev_sibling = None
        node.next_sibling = None
        return new_node

    def erase(self, node):
        """C++: opencog/util/tree.h:805 erase

        FIX WAVE (Task 4 review, Critical): C++'s `erase` is a TEMPLATE
        over the iterator type (`template<class iter> iter erase(iter
        it)`) -- its return value is `it` advanced by THAT ITERATOR
        TYPE's own `operator++`, via `ret.skip_children(); ++ret;`
        (tree.h:808-810). `skip_children()` (tree.h:2373) only sets a flag
        `pre_order_iterator::operator++` reads; `sibling_iterator::
        operator++` (tree.h:2926-2938) ignores it completely and just
        does `node = node->next_sibling` (or NULL at the end of the
        range). Every real reduct call site passes a `sib_it`
        (`combo_tree::sibling_iterator`), so the correct successor here is
        `node.next_sibling` (`None` at the end of a sibling range) --
        NOT `_skip_children_successor(node)`, which climbs to the
        node's PRE-ORDER successor (the right answer for a `pre_it`, the
        template instantiation this method never previously distinguished
        from `sib_it`). The bug this fixes: erasing the LAST child of a
        node returned that node's own PARENT's next sibling instead of
        `None`, so `x = tr.erase(x)` walked OUTSIDE the sibling range and
        a subsequent write went through `x` into unrelated tree state.
        Measured (Task 4 review): on `and(or($1 $2) $3)`, erasing the
        `or`'s last child (`$2`) used to return `argument(3)` instead of
        `None`.

        `_skip_children_successor` itself is untouched and remains
        correct for its other two real uses (`iter_pre_at`'s own
        pre-order-bounded walk, `equal_subtree`'s lockstep boundary) --
        both are genuinely pre-order-shaped, unlike this one.

        S7 Task 5 follow-up: `reduce_remove_subtree_equal_tt`
        (logical_rules.cc:845-853) walks with a `pre_it`, not a `sib_it`
        (`for (pre_it pit = it.begin(); pit != it.end();) { pit =
        tr.insert_above(...); ...; pit = tr.erase(pit); }`) -- that call
        site needs the OLD (pre-`erase()`-fix) pre-order-successor
        behaviour instead, ported separately below as
        `erase_pre_order` rather than silently reusing this
        sibling-successor version. Both share the same unlink mechanics
        via `_unlink`; only the computed `successor` differs.
        """
        _require_live_node(node, "erase")
        successor = node.next_sibling
        self._unlink(node)
        return successor

    def erase_pre_order(self, node):
        """No direct C++ symbol of its own -- the SAME C++ template method
        `tree<T>::erase` (opencog/util/tree.h:805) instantiated for a
        `pre_it` (`combo_tree::iterator`/`pre_order_iterator`) rather than
        the `sib_it` every OTHER real reduct caller passes (see `erase`'s
        own docstring for the general `ret.skip_children(); ++ret;`
        mechanism and why the two iterator types disagree).
        `pre_order_iterator::operator++` DOES respect the `skip_children`
        flag `ret.skip_children()` sets (unlike `sibling_iterator::
        operator++`, which ignores it) -- descending into `node`'s
        children is exactly what must NOT happen once they are about to
        be erased, so the resulting successor is `node`'s own PRE-ORDER
        successor computed AS IF `node` were childless: precisely
        `_skip_children_successor(node)`, this port's existing helper for
        that exact shape (already used by `iter_pre_at`'s subtree
        boundary and `equal_subtree`'s lockstep boundary -- both
        genuinely pre-order-shaped, like this one, unlike plain `erase`).

        Added for `logical_rules.py`'s `reduce_remove_subtree_equal_tt`
        (logical_rules.cc:845-853, `pit = tr.erase(pit)` where `pit` is a
        `pre_it`) -- the one real call site in this port that walks with a
        pre-order iterator while mutating via `erase`, per the TODO
        `erase()`'s own docstring left for this task (Task 4 review fix
        round 1).
        """
        _require_live_node(node, "erase_pre_order")
        successor = self._skip_children_successor(node)
        self._unlink(node)
        return successor

    def _unlink(self, node):
        """No direct C++ symbol: the unlink-and-poison body shared by
        `erase` and `erase_pre_order` -- both are the SAME C++ `erase`
        template, differing only in which iterator type computes the
        returned successor (see both callers' own docstrings). Splitting
        this out avoids duplicating the splice/poison logic between them.
        """
        self.erase_children(node)
        if node.prev_sibling is None:
            if node.parent is not None:
                node.parent.first_child = node.next_sibling
        else:
            node.prev_sibling.next_sibling = node.next_sibling
        if node.next_sibling is None:
            if node.parent is not None:
                node.parent.last_child = node.prev_sibling
        else:
            node.next_sibling.prev_sibling = node.prev_sibling
        if node is self._root:
            self._root = None
        # DIVERGENCE(C++ opencog/util/tree.h:805): C++ destroys and
        # deallocates the erased `tree_node` (tree.h:824-825) after
        # unlinking it -- any pointer a
        # caller kept is now dangling, and dereferencing it is undefined
        # behaviour (typically a crash, sometimes silent corruption).
        # Python cannot reproduce "dangling": the object stays alive as
        # long as anything references it, so a naive port would leave its
        # outward links holding their stale pre-erase values -- a
        # stale handle would then silently navigate back into the LIVE
        # tree instead of crashing. That is exactly the hazard S8's
        # knob_mapper (one retained iterator per knob) must not hit:
        # ported knob code would silently reattach where C++ crashes.
        # Poisoning those three links with `_DEAD` below makes the
        # failure loud (`AttributeError`, from `_DEAD`'s empty
        # `__slots__`) at the first attempt to navigate through them --
        # safer than C++'s silent corruption, but a deliberate, disclosed
        # divergence from it, not an equivalent implementation of the
        # same dangling-pointer behaviour.
        node.parent = _DEAD
        node.prev_sibling = _DEAD
        node.next_sibling = _DEAD

    def erase_children(self, node):
        """C++: opencog/util/tree.h:787 erase_children"""
        _require_live_node(node, "erase_children")
        cur = node.first_child
        while cur is not None:
            nxt = cur.next_sibling
            self.erase_children(cur)
            cur.parent = _DEAD
            cur.prev_sibling = _DEAD
            cur.next_sibling = _DEAD
            cur = nxt
        node.first_child = None
        node.last_child = None

    def replace(self, node, data_or_subtree):
        """C++: opencog/util/tree.h:1388 replace (plain data, same node
        identity); the subtree-copy overload is opencog/util/tree.h:1397
        replace(iter, const iterator_base&), which allocates a brand-new
        node -- `node` becomes invalid in that branch, matching the C++
        docs' "'position' becomes invalid."

        Restores `tree.h:1399`'s `tree_assert(position.node != head)`.
        C++'s check there is strictly "not the head sentinel," not "not
        null" (a genuinely null `position.node` would slip past it and
        crash later); applied here as a plain null guard regardless, for
        the same reason `append_children` guards past what its own
        delegation covers -- one consistent check per public entry point,
        covering both the plain-data and subtree-copy branches below.
        """
        if node is None:
            raise ValueError("replace: node must not be None")
        _require_live_node(node, "replace")
        if isinstance(data_or_subtree, tree_node):
            _require_live_subtree_source(data_or_subtree, "replace")
            if node is data_or_subtree:
                return node
            new_node = tree_node(_copy_value(data_or_subtree.data))
            new_node.parent = node.parent
            new_node.prev_sibling = node.prev_sibling
            new_node.next_sibling = node.next_sibling
            if new_node.prev_sibling is not None:
                new_node.prev_sibling.next_sibling = new_node
            elif new_node.parent is not None:
                new_node.parent.first_child = new_node
            if new_node.next_sibling is not None:
                new_node.next_sibling.prev_sibling = new_node
            elif new_node.parent is not None:
                new_node.parent.last_child = new_node
            if node is self._root:
                self._root = new_node
            self._copy_descendants(new_node, data_or_subtree)
            # DIVERGENCE(C++ opencog/util/tree.h:1397): this subtree-copy
            # branch destroys and deallocates `position` and its old
            # descendants; C++'s own docs therefore call the iterator
            # "invalid". Any retained C++ handle is dangling and reading
            # it is undefined behaviour. Python retains referenced objects,
            # so the discarded node and all former descendants are poisoned
            # with `_DEAD` instead: a loud, deterministic failure rather
            # than undefined behaviour.
            self.erase_children(node)
            node.parent = _DEAD
            node.prev_sibling = _DEAD
            node.next_sibling = _DEAD
            return new_node
        node.data = data_or_subtree
        return node

    def flatten(self, node):
        """C++: opencog/util/tree.h:1500 flatten"""
        _require_live_node(node, "flatten")
        if node.first_child is None:
            return node
        child = node.first_child
        while child is not None:
            child.parent = node.parent
            child = child.next_sibling
        if node.next_sibling is not None:
            node.last_child.next_sibling = node.next_sibling
            node.next_sibling.prev_sibling = node.last_child
        elif node.parent is not None:
            node.parent.last_child = node.last_child
        node.next_sibling = node.first_child
        node.next_sibling.prev_sibling = node
        node.first_child = None
        node.last_child = None
        return node

    def wrap(self, node, data):
        """C++: opencog/util/tree.h:1580 wrap

        wrap(position, x) in C++ is `insert(position, x)` (previous
        sibling) followed by `reparent`-ing exactly the one-element range
        [position, position.next_sibling) into the new node -- i.e. splice
        a new node into position's old slot and make position its sole
        child. That is exactly insert_above's direct splice
        (tree.h:1324): for a single moved node the two are provably the
        same resulting shape, and delegating avoids depending on the
        sentinel-forest trick that is what lets plain `insert()` work on
        the tree's actual root in C++ (this port has none -- see the
        module docstring).

        Restores `tree.h:1582`'s `tree_assert(position.node != 0)`.
        """
        if node is None:
            raise ValueError("wrap: node must not be None")
        _require_live_node(node, "wrap")
        return self.insert_above(node, data)

    def reparent(self, new_parent, from_node, to_node=_NO_TO_NODE):
        """C++: opencog/util/tree.h:1572 reparent(iter, iter) [2-arg: move
        ALL of `from_node`'s own children onto `new_parent`] collapsed
        with opencog/util/tree.h:1525 reparent(iter, sibling_iterator,
        sibling_iterator) [3-arg, added task-3/S7: move the sibling RANGE
        [`from_node`, `to_node`) directly onto `new_parent` -- here
        `from_node` is already a first CHILD, not the parent whose
        children get gathered]. Collapsed into one Python signature the
        same way this port collapses other overloaded C++ methods (e.g.
        `tree(iterator)`, `downwards.__init__`): the 2-arg form is what
        the original (S0-S6) `tree_probe.cc`-verified behaviour needs;
        the 3-arg form is what `comboreduct/reduct/general_rules.py`'s
        `eval_constants` needs (`tr.reparent(tr.append_child(it, *it),
        it.begin(), to)`) -- moving only the just-`partition`ed constant
        PREFIX of `it`'s children onto a freshly appended sibling,
        without gathering `it`'s remaining (non-constant) children too.

        Both forms delegate to `_reparent_range`, which is the direct
        port of the 3-arg C++ overload's own body.
        """
        _require_live_node(new_parent, "reparent")
        _require_live_node(from_node, "reparent")
        if to_node is not _NO_TO_NODE:
            _require_live_node(to_node, "reparent")
        if to_node is _NO_TO_NODE:
            if from_node.first_child is None:
                return new_parent
            return self._reparent_range(new_parent, from_node.first_child, None)
        return self._reparent_range(new_parent, from_node, to_node)

    def move_after(self, target, source):
        """C++: opencog/util/tree.h:1592 move_after

        Restores `tree.h:1596-1597`'s `tree_assert(dst); tree_assert(src);`
        (null checks). See the module-level DIVERGENCE note: also raises
        if `target` or `source` is the tree's own root, since either would
        require a forest -- unlike C++, which has head/feet sentinels
        giving the root a real (non-null) prev/next_sibling to splice
        through the very same code path.
        """
        if target is None or source is None:
            raise ValueError("move_after: target and source must not be None")
        _require_live_node(target, "move_after")
        _require_live_node(source, "move_after")
        if target is source:
            return source
        if target.next_sibling is not None and target.next_sibling is source:
            return source
        if target is self._root or source is self._root:
            raise ValueError(
                "move_after: cannot move the tree root, or move a node to "
                "become a sibling of the tree root -- forests of multiple "
                "roots are out of scope for this port"
            )
        if source.prev_sibling is not None:
            source.prev_sibling.next_sibling = source.next_sibling
        else:
            source.parent.first_child = source.next_sibling
        if source.next_sibling is not None:
            source.next_sibling.prev_sibling = source.prev_sibling
        else:
            source.parent.last_child = source.prev_sibling
        if target.next_sibling is not None:
            target.next_sibling.prev_sibling = source
        else:
            target.parent.last_child = source
        source.next_sibling = target.next_sibling
        target.next_sibling = source
        source.prev_sibling = target
        source.parent = target.parent
        return source

    def move_before(self, target, source):
        """C++: opencog/util/tree.h:1628 move_before

        Restores `tree.h:1632-1633`'s `tree_assert(dst); tree_assert(src);`
        (null checks). See the module-level DIVERGENCE note: also raises
        if `target` or `source` is the tree's own root, for the same
        reason as `move_after`.
        """
        if target is None or source is None:
            raise ValueError("move_before: target and source must not be None")
        _require_live_node(target, "move_before")
        _require_live_node(source, "move_before")
        if target is source:
            return source
        if target.prev_sibling is not None and target.prev_sibling is source:
            return source
        if target is self._root or source is self._root:
            raise ValueError(
                "move_before: cannot move the tree root, or move a node to "
                "become a sibling of the tree root -- forests of multiple "
                "roots are out of scope for this port"
            )
        if source.prev_sibling is not None:
            source.prev_sibling.next_sibling = source.next_sibling
        else:
            source.parent.first_child = source.next_sibling
        if source.next_sibling is not None:
            source.next_sibling.prev_sibling = source.prev_sibling
        else:
            source.parent.last_child = source.prev_sibling
        if target.prev_sibling is not None:
            target.prev_sibling.next_sibling = source
        else:
            target.parent.first_child = source
        source.prev_sibling = target.prev_sibling
        target.prev_sibling = source
        source.next_sibling = target
        source.parent = target.parent
        return source

    def move_ontop(self, target, source):
        """C++: opencog/util/tree.h:1710 move_ontop"""
        # DIVERGENCE(C++ opencog/util/tree.h:1708): inherited upstream
        # defect, deliberately not fixed here. move_ontop(target, source)
        # corrupts the sibling chain when `source` is `target`'s immediate
        # next sibling: `b_next_sibling` is captured as target.next_sibling
        # *before* target is erased, and when source IS that next sibling,
        # the final `source.next_sibling = b_next_sibling` sets
        # source.next_sibling to source itself -- a self-cycle -- while
        # `parent.last_child` is left stale (never corrected). A traversal
        # that later follows next_sibling through this node -- including
        # this class's own __iter__ -- does not terminate, in Python
        # exactly as in C++. This is deliberate fidelity to a real upstream
        # bug (confirmed against tools/cpp_probe/tree_probe.cc's
        # move_ontop_adjacent_sibling_bug case and
        # test_move_ontop_adjacent_sibling_self_cycle_is_an_inherited_bug),
        # not an oversight.
        #
        # DO NOT "FIX" THIS. The correct-looking change -- special-case
        # `target.next_sibling is source` and relink correctly, or assert
        # against it -- would silently diverge this port from the C++
        # baseline it exists to mirror. If upstream tree.h is ever
        # corrected, port that fix here in the same change, not before,
        # and update the regression test above to match.
        #
        # Restores `tree.h:1714-1715`'s `tree_assert(dst); tree_assert(src);`
        # (null checks), and enforces tree.h:1708's documented precondition
        # that source is not contained in target's subtree. All guards run
        # BEFORE mutation, so a rejected call leaves the tree untouched.
        if target is None or source is None:
            raise ValueError("move_ontop: target and source must not be None")
        _require_live_node(target, "move_ontop")
        _require_live_node(source, "move_ontop")
        if target is source:
            return source
        if target is self._root or source is self._root:
            raise ValueError(
                "move_ontop: cannot move the tree root, or move a node to "
                "replace the tree root -- forests of multiple roots are "
                "out of scope for this port"
            )
        ancestor = source.parent
        while ancestor is not None:
            if ancestor is target:
                raise ValueError(
                    "move_ontop: source must not be a descendant of target"
                )
            ancestor = ancestor.parent
        b_prev_sibling = target.prev_sibling
        b_next_sibling = target.next_sibling
        b_parent = target.parent
        self.erase(target)
        if source.prev_sibling is not None:
            source.prev_sibling.next_sibling = source.next_sibling
        else:
            source.parent.first_child = source.next_sibling
        if source.next_sibling is not None:
            source.next_sibling.prev_sibling = source.prev_sibling
        else:
            source.parent.last_child = source.prev_sibling
        if b_prev_sibling is not None:
            b_prev_sibling.next_sibling = source
        else:
            b_parent.first_child = source
        if b_next_sibling is not None:
            b_next_sibling.prev_sibling = source
        else:
            b_parent.last_child = source
        source.prev_sibling = b_prev_sibling
        source.next_sibling = b_next_sibling
        source.parent = b_parent
        return source
