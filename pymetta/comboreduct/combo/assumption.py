"""C++: moses/comboreduct/combo/assumption.{h,cc}

This file contains the functions needed to treat assumptions attached to
a combo_tree. In C++, "the assumption set of a combo_tree" (assumption.h's
own header comment, lines 31-37) means: every child (subtree) of the
combo_tree's *hidden head node* except the first one, which is the
"conclusion" or main tree -- i.e. a combo_tree here is used as a tiny
FOREST: one top-level "real" program plus zero or more top-level
"assumption" subtrees, all siblings of each other, held together purely
by `tree<T>`'s head/feet sentinels (`opencog/util/tree.h:734-740`).

# DIVERGENCE(C++ moses/comboreduct/combo/assumption.cc, whole file): no forest -> plain list.
# `util/tree.py` deliberately does not support that forest
# structure -- "no forests/multiple roots (no head/feet
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
genuine tree-node siblings of the root. The functions below reproduce the
REAL C++ behaviour against that list (see `insert_assumption`'s own
docstring for a measured, not assumed, correction to an earlier version
of this file that silently "fixed" two upstream bugs instead).

Checked directly against $CLASSIC (`grep -rn insert_assumption|
find_vertices_in_assumptions|equal_assumptions|delete_all_assumptions
$CLASSIC/moses/comboreduct/reduct/`) for what this reshaping is, and is
not, safe against:

* `general_rules.cc`'s `remove_all_assumptions` rule and `meta_rules.cc`'s
  `assum_iterative` combinator only ever call `delete_all_assumptions`/
  `equal_assumptions` on the tree *as a whole* -- never by walking
  assumption siblings as raw `combo_tree::iterator`s themselves.
* `logical_rules.cc`'s `reduce_and_assumptions` -- the one real S7-scope
  call site of `insert_assumption` -- builds a disposable scratch
  `combo_tree` (`copy_tr`) to run the ordinary rule-reduction engine over
  a subtree "assuming" its siblings, then keeps only `copy_tr.begin()`
  (the reduced conclusion) and discards the rest. It DOES also inspect
  the assumption region as raw tree nodes for one step first
  (`logical_rules.cc:236-238`): it copies `tr`'s OWN existing assumptions
  into `copy_tr`, in order, via a raw `insert_subtree_after` chain that
  bypasses `insert_assumption` entirely -- `copy_all_assumptions` below
  is this port's equivalent of that exact step, added for Task 4 to use
  (an earlier version of this docstring claimed this rule "never
  inspects the assumption region as tree nodes either", which was false;
  corrected here after re-reading logical_rules.cc line by line).
  `reduce_and_assumptions`/`reduce_or_assumptions` are wired into
  `mixed_reduction.cc`/`full_reduction.cc` (S14 scope -- CLAUDE.md's
  "Known gap, deliberate" -- not legacy embodiment code, an earlier
  version of this project's task-1-report.md mischaracterised this),
  never into `logical_reduction.cc` (checked directly: zero matches),
  so this is still unreachable from S7's own boolean-only rule sets.
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

UNSETTLED, flagged rather than silently assumed (task-1 review, Important
#3): copying a `combo_tree` via `combo_tree(some_node)` (tree.py's
iterator-constructor, extended this same task) never carries an
`assumptions` list forward -- a bare `tree_node` has no back-reference to
whatever tree it came from, so there is nothing FOR this constructor to
even look up. This is not a gap introduced by narrowing scope: C++'s own
`tree(const iterator_base&)` (tree.h:658) *also* never captures a
source's siblings -- its copy loop is bounded to exactly the source
node's own descendants (see tree.py's `_copy_descendants` docstring) --
so an assumption region hanging off `tr.begin()`'s siblings was never in
scope for THIS constructor in C++ either. What C++ has that this port
does not is a WHOLE-TREE copy (`tree::operator=`/copy-constructor, the
mechanism `meta_rules.cc:147`'s `assum_iterative`'s `tmp = tr;` uses),
which genuinely would carry the assumption region along in C++, and which
no code in this port constructs today (`assum_iterative` itself is not
in Task 2's scope -- only `downwards`/`upwards`/`sequential`/`iterative`
are, and `iterative`'s own `tmp = combo_tree(it)` never touches
assumptions, per `meta_rules.cc:130-135`). Decision, made deliberately
rather than left as a silent trap: no whole-tree-copy-with-assumptions
mechanism is added speculatively here. Whoever eventually ports
`assum_iterative` (currently not on any task's list) must add one then
and decide whether it carries `.assumptions` -- this paragraph is that
decision point's flag.
"""

from pymetta.comboreduct.combo.vertex import (
    combo_tree,
    lexicographic_subtree_less,
    vertex,
    vertex_equal,
)
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _c_trunc_div(a: int, b: int) -> int:
    """No direct C++ symbol: C++'s `/` on two `int`s truncates toward
    zero; Python's `//` floors toward negative infinity. The two
    disagree exactly when the operands have opposite sign -- which
    `insert_assumption`'s binary search (assumption.cc:42, 50, 56) can
    drive `(max_pos - min_pos)` into. Needed to reproduce that
    arithmetic (and the bug it is part of) exactly, not approximately.
    """
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _standalone_copy(node: tree_node) -> combo_tree:
    """Deep-copy `node`'s entire subtree into a fresh, standalone
    combo_tree -- what `insert_assumption`/`copy_all_assumptions` store
    per assumption (see module docstring). `combo_tree(node)` (tree.py's
    `__init__`, extended this same task for this exact need) already
    performs the copy; this wrapper exists only to name the operation at
    its call sites.
    """
    return combo_tree(node)


def _equal_subtree_vertex_safe(a: tree_node, b: tree_node) -> bool:
    """No direct C++ symbol: a type-safe replacement for `tree.
    equal_subtree` (tree.py:447), used ONLY here because that method
    itself compares node data with a bare `a.data != b.data`
    (tree.py:465) -- verified live (`equal_subtree(combo_tree(1.0).
    begin(), combo_tree(builtin.logical_true).begin())` returns `True`,
    the exact `contin_t`/`IntEnum` collision this whole task's brief
    warns about). `tree.equal_subtree` is shared, foundational code from
    the prior (S0-S6) cycle, used elsewhere in this port (e.g. Task 2's
    `iterative` fixpoint check) -- fixing it globally is out of THIS
    task's scope and is flagged separately (task-1-report.md) rather than
    changed here; this local helper closes the hole at the one call site
    this file controls (`equal_assumptions` below) without touching
    shared code. Same recursive-descent shape as `tree.equal_subtree`,
    `vertex_equal` (not bare `!=`) at each node.
    """
    if a is b:
        return True
    if not vertex_equal(a.data, b.data):
        return False
    ca, cb = a.first_child, b.first_child
    while ca is not None and cb is not None:
        if not _equal_subtree_vertex_safe(ca, cb):
            return False
        ca, cb = ca.next_sibling, cb.next_sibling
    return ca is None and cb is None


# ---------------------------------------------------------------------------
# The four functions assumption.h declares, plus one addition
# ---------------------------------------------------------------------------


def insert_assumption(tr: combo_tree, assum_it: tree_node) -> None:
    """C++: moses/comboreduct/combo/assumption.cc:35 insert_assumption

    KNOWN UPSTREAM BUG, ported faithfully -- measured directly against
    the compiled C++ library (`tools/cpp_probe/assumption_probe.cc`,
    `libcomboreduct.so`), not inferred from the class's own doc comment.
    That comment (assumption.h:34-37) claims the assumption set is kept
    "lexicographically ordered" and that `insert_assumption` is "the only
    way to add an element... and respect that order" -- true only once
    `tr` already holds MORE than one assumption. The real algorithm
    guards its ordered binary-search insert with `if (max_pos > 1)`
    (assumption.cc:40, `max_pos` being the CURRENT assumption count); with
    zero or exactly one assumption already present it takes the
    unconditional `tr.insert_subtree_after(main_tree, assum_it)`
    (assumption.cc:74) instead -- no comparison against what is already
    there, no dedup, and it always inserts as the new FIRST assumption
    (immediately after the conclusion, before whatever was already
    there). Measured:

        insert("true"), insert("true")           -> [true, true]
                                                     (not deduplicated)
        insert("$1"), insert("$2"), insert("$3") -> [$2, $1, $3]
                                                     (not sorted)
        insert("$3"), insert("$2"), insert("$1") -> [$1, $2, $3]
                                                     (same SET, different
                                                     order -- depends on
                                                     insertion history)
        insert("$3"), insert("$4"), insert("$3") -> [$3, $4, $3]
                                                     (max_pos > 1 on the
                                                     third insert -- the
                                                     binary search runs --
                                                     yet the duplicate is
                                                     STILL not caught)
        insert("true") x3                        -> [true, true]
                                                     (max_pos > 1 here
                                                     too, and THIS
                                                     duplicate IS caught)

    Per "port faithfully; document upstream bugs, never silently fix
    them" (the same treatment `equal_assumptions`'s asymmetric-empty
    quirk below gets) this reproduces the real algorithm exactly: the
    `max_pos <= 1` unconditional-prepend bypass AND the `max_pos > 1`
    binary search, translating C++'s sibling-iterator position
    arithmetic into 1-based list-index arithmetic one-for-one against
    `tr.assumptions` (`_c_trunc_div` reproduces C++'s toward-zero integer
    division, which the search's position arithmetic can drive negative).

    `max_pos > 1` does NOT guarantee dedup, and an earlier version of
    this docstring (and of the test suite) wrongly claimed it did,
    without re-checking against the real library first -- exactly the
    mistake this whole file exists to avoid repeating. The binary search
    only compares the NEW assumption against the specific positions its
    search path happens to probe; with as few as two existing elements it
    can converge to an insertion point (the "search range collapsed,
    insert unconditionally" branches, assumption.cc:62/65-70) without
    ever comparing against the element that is actually the duplicate --
    the `[$3, $4, $3]` case above. It only dedups when the search's own
    probe sequence happens to land exactly on the equal element first --
    the `[true, true]` (x3) case above, where the very first (and only)
    position checked already holds "true". Both are measured, both are
    reproduced; neither is "corrected" toward the other.

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

    max_pos = len(assumptions)  # assumption.cc:39 number_of_siblings(main_tree)
    if max_pos > 1:
        min_pos = 1
        diff = _c_trunc_div(max_pos - min_pos, 2)
        cur_pos = min_pos + diff
        while min_pos < max_pos:
            cur_root = assumptions[cur_pos - 1].begin()
            if lexicographic_subtree_less(cur_root, new_root):
                min_pos = cur_pos + 1
                diff = 1 + _c_trunc_div(max_pos - min_pos, 2)
                cur_pos += diff
            elif lexicographic_subtree_less(new_root, cur_root):
                max_pos = cur_pos - 1
                diff = 1 + _c_trunc_div(max_pos - min_pos, 2)
                cur_pos -= diff
            else:
                return  # assumption.cc:60 -- already present, no-op
        if max_pos < min_pos:
            assumptions.insert(cur_pos, new_copy)  # insert_subtree_after
        elif min_pos == max_pos:
            cur_root = assumptions[cur_pos - 1].begin()
            if lexicographic_subtree_less(cur_root, new_root):
                assumptions.insert(cur_pos, new_copy)  # insert_subtree_after
            elif lexicographic_subtree_less(new_root, cur_root):
                assumptions.insert(cur_pos - 1, new_copy)  # insert_subtree
            # else: neither -- already present, no-op. assumption.cc has
            # no final `else` at all here (lines 65-70); falling through
            # without inserting is the correct, faithful translation.
        else:
            raise AssertionError("should never get here (insert_assumption).")
    else:
        # assumption.cc:74 -- THE BUG: unconditional prepend whenever 0
        # or 1 assumptions already exist. No comparison, no dedup.
        assumptions.insert(0, new_copy)


def copy_all_assumptions(dest: combo_tree, src: combo_tree) -> None:
    """No direct single C++ symbol: mirrors the raw copy loop inside
    `logical_rules.cc:236-238` (`reduce_and_assumptions`) --

        sib_it bna = copy_tr.begin();
        for(sib_it a = tr.next_sibling(tr.begin()); a != tr.end(); ++a)
            bna = copy_tr.insert_subtree_after(bna, a);

    -- the one place in $CLASSIC that copies an existing assumption set
    into another tree WITHOUT going through `insert_assumption` at all,
    bypassing its rules (buggy or not) entirely: a raw, order-preserving
    append chain. Added here (not declared in assumption.h) because this
    port's forest-free representation (see module docstring) has no
    `tr.next_sibling(tr.begin())` for `logical_rules.py` (Task 4) to walk
    the way C++ does, and Task 4 needs exactly this operation for its own
    port of `reduce_and_assumptions`. Appends deep copies of every one of
    `src`'s assumptions onto `dest`'s assumption list, in `src`'s own
    order, with NO reordering and NO dedup check against what `dest`
    already has.
    """
    dest_assumptions = getattr(dest, "assumptions", None)
    if dest_assumptions is None:
        dest_assumptions = dest.assumptions = []
    for src_assumption in getattr(src, "assumptions", []):
        dest_assumptions.append(_standalone_copy(src_assumption.begin()))


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
    unordered set comparison.

    Uses `_equal_subtree_vertex_safe`, NOT `tree.equal_subtree`, for the
    per-element comparison -- `tree.equal_subtree` compares node data
    with a bare `!=` (tree.py:465), which is exactly the `contin_t`/
    `IntEnum` collision this task's brief warns about (verified live:
    `tr.equal_subtree` says a `[1.0]` assumption equals a `[logical_true]`
    one). That defect lives in shared, foundational `tree.py` code out of
    this task's scope to change globally (flagged in task-1-report.md for
    separate scoping); this function closes it at the one call site this
    file controls.

    SURPRISING INHERITED BEHAVIOUR, ported as-is (not a `# DIVERGENCE` --
    this is what C++ itself does): assumption.cc:98-106's own `if
    (tr.is_valid(assum1) && tr.is_valid(assum2)) {...} else return true;`
    returns `true` whenever EITHER side has zero assumptions -- even if
    the OTHER side has some. A tree with three assumptions therefore
    compares "equal" to one with none, not just to another empty one.
    This looks like an asymmetric-comparison bug, but per "port
    faithfully; document upstream bugs, never silently fix them" it is
    reproduced exactly, not corrected to a symmetric "both empty" check.
    Unreachable in S7's actual scope either way: `equal_assumptions`'s
    only caller, `meta_rules.assum_iterative`, is itself only used by
    `perception_reduction.cc` (legacy embodiment/perception machinery,
    out of scope for the whole S7 cycle -- see module docstring), so no
    in-scope oracle exercises this branch in either direction.

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
        _equal_subtree_vertex_safe(x.begin(), y.begin())
        for x, y in zip(a1, a2, strict=True)
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
