"""C++: moses/comboreduct/combo/assumption.{h,cc}

This file contains the functions needed to treat assumptions attached to
a combo_tree. In C++, "the assumption set of a combo_tree" (assumption.h's
own header comment, lines 31-37) means: every child (subtree) of the
combo_tree's *hidden head node* except the first one, which is the
"conclusion" or main tree -- i.e. a combo_tree here is used as a tiny
FOREST: one top-level "real" program plus zero or more top-level
"assumption" subtrees, all siblings of each other, held together purely
by `tree<T>`'s head/feet sentinels (`opencog/util/tree.h:734-740`). The
assumption set is kept lexicographically ordered (`lexicographic_subtree_
order`, tree.h:3257) and deduplicated -- `insert_assumption` is documented
as "the only way to add an element in that set," so nothing else needs to
re-check that invariant.

# DIVERGENCE(C++ moses/comboreduct/combo/assumption.cc, whole file): this
# port's `util/tree.py` deliberately does not support that forest
# structure at all -- "no forests/multiple roots (no head/feet
# sentinels)" is stated in tree.py's own module docstring, is enforced at
# runtime (`insert`/`insert_after`/`insert_subtree`/`insert_subtree_after`/
# `move_after`/`move_before`/`move_ontop` all raise `ValueError` if asked
# to give the tree's own root a sibling), and is regression-tested
# (`test_insert_after_on_the_tree_root_is_rejected`,
# tests/util/test_tree.py) as a decision already made in the prior
# (S0-S6) cycle. `combo/iostream_combo.py`'s `str_to_combo_tree`
# independently enforces the same thing at the text-format level ("a
# combo_tree is always single-rooted... rejects trailing content"). None
# of this is something Task 1 can or should reopen -- CARRY-FORWARD.md
# section 5 records "'forests unsupported'... is uniform" as an accepted,
# load-bearing property of this whole port, not an oversight.

Consequence: the assumption set is instead attached to a combo_tree as a
plain Python list, `tr.assumptions` (created lazily, on first
`insert_assumption`; absent entirely on a tree nothing has ever inserted
into, which every other function here treats the same as "empty").
Each element is a standalone `combo_tree` holding a deep copy of one
assumption subtree -- mirroring C++'s own copy-on-insert semantics
(`insert_subtree`/`insert_subtree_after` always copy, never alias, per
`tree.h`'s own doc comments) -- just relocated to a side list instead of
genuine tree-node siblings of the root. The four functions below
reproduce the *documented contract* (insert as a deduplicated,
lexicographically-ordered set; find every vertex-match anywhere in the
set; compare two sets for structural equality; clear the set) against
that list, not against tree.py's node graph.

This divergence is safe against every actual caller in S7's scope,
checked directly against $CLASSIC (`grep -rn insert_assumption|
find_vertices_in_assumptions|equal_assumptions|delete_all_assumptions
$CLASSIC/moses/comboreduct/reduct/`):

* `general_rules.cc`'s `remove_all_assumptions` rule and `meta_rules.cc`'s
  `assum_iterative` combinator only ever call `delete_all_assumptions`/
  `equal_assumptions` on the tree *as a whole* -- never by walking
  assumption siblings as raw `combo_tree::iterator`s themselves.
* `logical_rules.cc`'s `reduce_and_assumptions` (the one real call site of
  `insert_assumption` in S7's scope) builds a disposable *scratch*
  `combo_tree` (`copy_tr`) purely to run the ordinary rule-reduction
  engine over a subtree "assuming" its siblings, then keeps only
  `copy_tr.begin()` (the reduced conclusion) and discards the rest --
  it never inspects the assumption region as tree nodes either.
* `find_vertices_in_assumptions` -- the one function whose contract is
  genuinely iterator-shaped (it returns `combo_tree::iterator` handles) --
  has its only real call site in `perception_rules.cc`, itself legacy
  embodiment/perception machinery out of scope for the entire S7 cycle
  (CLAUDE.md's "Known sharp edges"; not in the S7 plan's file list).
  Ported here for completeness against assumption.h's declared API (and
  because a future task could plausibly need it), returning
  `tree_node`s from the internal per-assumption `combo_tree`s -- callers
  get real, navigable handles, just not ones aliasing the main tree's own
  graph the way C++'s would.

The ordering used by `insert_assumption` below is a deliberately
SIMPLIFIED stand-in for `lexicographic_subtree_order<vertex>`
(tree.h:3257) -- see `_vertex_rank`/`_less_vertex`/`_less_node`'s own
docstrings for why a byte-exact port of that comparator (which orders by
`boost::variant::which()` across all 14 alternatives, most of which this
port never constructs) is neither practical nor load-bearing here: C++'s
own comment on it is "Not really lexicographic, but a valid ordering...
can be changed if it's important," and no C++ test or in-scope oracle
exercises the exact order at all (grep across $CLASSIC/tests turns up no
assumption.cc test whatsoever).
"""

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.common_def import contin_t
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, vertex, vertex_equal
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# Internal ordering -- a simplified stand-in for lexicographic_subtree_order
# (see module docstring)
# ---------------------------------------------------------------------------


def _vertex_rank(v: vertex) -> int:
    """Relative order among this port's four vertex alternatives, matching
    their DECLARATION ORDER in the real 14-way `boost::variant` (vertex.h:
    128: `contin_t, enum_t, builtin, wild_card, argument, ...`) -- only
    relative order among the four alternatives this port constructs is
    reproduced (the intervening `wild_card` is simply skipped, so
    `argument`'s rank here is 3, not 4); absolute values are never
    compared or serialised. A bare, non-`enum_t` `str` (this port's
    stand-in for `definite_object`/`message`, vertex.py's module
    docstring) is unreachable in a combo_tree's assumption set in
    practice (iostream_combo.py's own docstring: combo-tree grammar never
    produces one) and is ranked alongside `enum_t` for simplicity, not out
    of any claim to reproduce `definite_object`'s real (much later)
    position in the full variant.
    """
    if isinstance(v, contin_t):
        return 0
    if isinstance(v, str):
        return 1
    if isinstance(v, builtin):
        return 2
    if isinstance(v, argument):
        return 3
    raise TypeError(f"vertex {v!r} is not one of this port's four alternatives")


def _less_vertex(v1: vertex, v2: vertex) -> bool:
    """A total order over single vertices: rank first (see `_vertex_rank`),
    then each alternative's own natural order for a same-rank pair --
    `argument.__lt__` (`absolute_value_order`, argument.py) for two
    arguments, plain `<` for two `contin_t`/`str`/`builtin` values (the
    last ordering by ordinal, since `builtin` is an `IntEnum`). Safe by
    construction: `_vertex_rank` already confirmed both operands share the
    same Python type before either `<` here ever runs, so this never
    risks the `contin_t`/`builtin` cross-type trap `vertex_equal` guards
    against.
    """
    r1, r2 = _vertex_rank(v1), _vertex_rank(v2)
    if r1 != r2:
        return r1 < r2
    return v1 < v2


def _cmp_node(n1: tree_node, n2: tree_node) -> int:
    """C++: opencog/util/tree.h:3277 lexicographic_subtree_order::cmp

    Recursive pre-order comparison: 1 if `n1`'s subtree sorts first, -1 if
    `n2`'s does, 0 if they are structurally equal. A node with fewer
    children than its counterpart (once their own vertices compare equal)
    sorts first -- ported directly from the C++ recursion using this
    port's `first_child`/`next_sibling` node links in place of
    `sibling_iterator`.
    """
    if _less_vertex(n1.data, n2.data):
        return 1
    if _less_vertex(n2.data, n1.data):
        return -1
    sib1, sib2 = n1.first_child, n2.first_child
    while True:
        if sib1 is None:
            return 0 if sib2 is None else 1
        if sib2 is None:
            return -1
        res = _cmp_node(sib1, sib2)
        if res != 0:
            return res
        sib1, sib2 = sib1.next_sibling, sib2.next_sibling


def _less_node(n1: tree_node, n2: tree_node) -> bool:
    """C++: opencog/util/tree.h:3273 lexicographic_subtree_order::operator()"""
    return _cmp_node(n1, n2) > 0


def _standalone_copy(node: tree_node) -> combo_tree:
    """Deep-copy `node`'s entire subtree into a fresh, standalone
    combo_tree -- what `insert_assumption` stores per assumption (see
    module docstring). `combo_tree(node)` (tree.py's `__init__`, extended
    this same task for this exact need) already performs the copy; this
    wrapper exists only to name the operation at its call sites.
    """
    return combo_tree(node)


# ---------------------------------------------------------------------------
# The four functions assumption.h declares
# ---------------------------------------------------------------------------


def insert_assumption(tr: combo_tree, assum_it: tree_node) -> None:
    """C++: moses/comboreduct/combo/assumption.cc:35 insert_assumption

    Inserts a (deep copy of the) subtree rooted at `assum_it` into `tr`'s
    assumption set. It is a SET: a structural duplicate of an assumption
    already present is silently dropped, matching C++'s own `else
    return;` when neither ordering comparison holds (`cmp == 0`, i.e.
    structurally equal). See module docstring for what "assumption set"
    means in this port (`tr.assumptions`, not tree siblings) and for the
    ordering used.

    It is assumed that `tr` is not empty -- mirrors `OC_ASSERT(tr.
    is_valid(main_tree), ...)` (assumption.cc:38).
    """
    main_tree = tr.begin()
    if main_tree is None:
        raise AssertionError("combo_tree isn't valid (insert_assumption).")

    assumptions = getattr(tr, "assumptions", None)
    if assumptions is None:
        assumptions = tr.assumptions = []

    new_copy = _standalone_copy(assum_it)
    new_root = new_copy.begin()
    for i, existing in enumerate(assumptions):
        existing_root = existing.begin()
        if _less_node(existing_root, new_root):
            continue
        if _less_node(new_root, existing_root):
            assumptions.insert(i, new_copy)
            return
        return  # neither a<b nor b<a: structurally equal, already present
    assumptions.append(new_copy)


def find_vertices_in_assumptions(
    tr: combo_tree, v: vertex, res: list[tree_node]
) -> bool:
    """C++: moses/comboreduct/combo/assumption.cc:77 find_vertices_in_assumptions

    Appends to `res` every node, ANYWHERE in ANY assumption subtree (not
    just assumption roots -- C++'s own walk is a full pre-order
    `combo_tree::iterator`, not a `sibling_iterator` restricted to
    top-level assumptions), whose vertex equals `v`. Returns whether
    anything was found. Uses `vertex_equal` rather than bare `==` --
    C++'s `*assum_it == v` can never true-positive across
    `boost::variant` alternatives the way bare Python `==` can between a
    `contin_t` and a `builtin` (see vertex.py's `vertex_equal` docstring
    and eval.py's module docstring for the same trap).

    It is assumed that `tr` is not empty -- mirrors `OC_ASSERT(tr.
    is_valid(tr_it), ...)` (assumption.cc:80).
    """
    if tr.begin() is None:
        raise AssertionError(
            "combo_tree isn't valid (find_vertices_in_assumptions)."
        )

    is_there = False
    for assumption in getattr(tr, "assumptions", []):
        for node in assumption:
            if vertex_equal(node.data, v):
                res.append(node)
                is_there = True
    return is_there


def equal_assumptions(tr1: combo_tree, tr2: combo_tree) -> bool:
    """C++: moses/comboreduct/combo/assumption.cc:94 equal_assumptions

    True iff `tr1` and `tr2` carry the same assumption set, compared
    POSITIONALLY (element `i` of one against element `i` of the other,
    per C++'s own lock-step `++assum1, ++assum2` walk) rather than as an
    unordered set comparison -- safe here because `insert_assumption` is
    the only way either list is ever populated, so two equal SETS always
    end up in the same canonical order (same reasoning `flat_normal_form`
    relies on for `argument`'s own ordering, docs/CARRY-FORWARD.md 4.4).

    SURPRISING INHERITED BEHAVIOUR, ported as-is (not a `# DIVERGENCE` --
    this is what C++ itself does): assumption.cc:98-106's own `if
    (tr.is_valid(assum1) && tr.is_valid(assum2)) {...} else return true;`
    returns `true` whenever EITHER side has zero assumptions -- even if
    the OTHER side has some. A tree with three assumptions therefore
    compares "equal" to one with none, not just to another empty one.
    This looks like an asymmetric-comparison bug, but per "port
    faithfully; document upstream bugs, never silently fix them" it is
    reproduced exactly, not corrected to a symmetric "both empty"
    check. Unreachable in S7's actual scope either way:
    `equal_assumptions`'s only caller, `meta_rules.assum_iterative`, is
    itself only used by `perception_reduction.cc` (legacy
    embodiment/perception machinery, out of scope for the whole S7
    cycle -- see module docstring), so no in-scope oracle exercises this
    branch in either direction.

    It is assumed that neither tree is empty -- mirrors `OC_ASSERT(!tr1.
    empty() && !tr2.empty(), ...)` (assumption.cc:95).
    """
    if tr1.begin() is None or tr2.begin() is None:
        raise AssertionError(
            "assumptions should not be empty (equal_assumptions)."
        )

    a1 = getattr(tr1, "assumptions", [])
    a2 = getattr(tr2, "assumptions", [])
    if not a1 or not a2:
        return True  # see the docstring's "surprising inherited behaviour"
    if len(a1) != len(a2):
        return False
    return all(
        tr1.equal_subtree(x.begin(), y.begin()) for x, y in zip(a1, a2, strict=True)
    )


def delete_all_assumptions(tr: combo_tree) -> None:
    """C++: moses/comboreduct/combo/assumption.cc:109 delete_all_assumptions

    Clears `tr`'s assumption set entirely. A tree with no assumption set
    yet (`tr.assumptions` never created) is left with none, same as if it
    had been created empty and then cleared -- no error, matching C++'s
    own `if (tr.is_valid(assum)) for (...)` guard around an already-empty
    walk.

    It is assumed that `tr` is not empty -- mirrors `OC_ASSERT(!tr.
    empty(), ...)` (assumption.cc:110).
    """
    if tr.begin() is None:
        raise AssertionError(
            "combo_tree should not be empty (delete_all_assumptions)."
        )
    tr.assumptions = []
