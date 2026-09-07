"""Task 6 (S7 cycle) oracle support -- test-only helpers, not part of
`pymetta` itself. Three independent verification layers laid over
`pymetta.comboreduct.reduct.logical_reduction.logical_reduce`, none of
them the byte-comparison against tests/fixtures/reduct.json (that lives
directly in test_reduct_oracle.py):

1. `tt_unsound_by_construction` -- a STRUCTURAL indicator for the
   shape docs/CARRY-FORWARD.md Β§3 (amended 2026-09-07) identifies as
   classic's `eval_logical_identities` unsoundness *mechanism*: "a
   junctor's entire child list consists of childless junctors."
   It is neither a necessary nor sufficient classifier for end-to-end
   non-preservation. The mechanism ignores parent context, and earlier
   reductions can create the shape even when it was absent from the input.
   Measured directly against `$CLASSIC`'s own `test_reduct` (pinned as
   `test_pinned_context_dependent_rows` in test_reduct_oracle.py): an
   emptied `and()` AT THE ROOT folds to `true` (nothing lost, preserved);
   the SAME emptied `and()` under an `or` parent is erased as "skip",
   destroying a definite TRUE (a real divergence); and a genuinely
   diverging subtree can be MASKED by an enclosing context that supplies
   the same truth value some other way. Conversely, the effort-1 input
   `or(and(or($1 and(!$1))) !$2)` is initially unflagged, but reduction
   creates the lossy shape and turns a tautology into `!$2`. Neither
   boolean result is therefore a semantic guarantee. Computed fresh from whatever
   combo_tree it is handed -- deliberately independent of
   tests/fixtures/reduct.json's own `known_tt_divergent` flag (a
   hardcoded id list rots as the corpus grows, and cannot flag a NEW tree
   that hits the same mechanism -- the task brief's own reasoning).

2. `evaluate_truth_table` -- a three-outcome evaluator over
   `complete_truth_table` (comboreduct/table/table.py) that additionally
   detects, and reports as its own outcome rather than as a crash, the
   ONE shape from which a raw builtin operator token (not `logical_true`/
   `logical_false`) can reach the OUTPUT of a whole-tree evaluation --
   see `root_is_bare_junctor_token`'s docstring for the proof that this
   shape is exactly "the tree's root is a childless and()/or()", nothing
   more general.

3. The seven ENF (elegant normal form) structural axioms, transliterated
   from `elegant-normal-forms-python`'s `Tests/EnfRuleCheckers.py`
   (`ruleOne`..`ruleSeven`) against combo_tree's actual and/or/argument
   shape -- see the section below the imports for the translation this
   uses instead of that module's own `DataStructures.Trees.TreeNode`
   (guardSet + children, a data structure this project's `combo_tree`
   does not have and does not need to grow just to reuse this logic).
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_argument
from pymetta.comboreduct.interpreter.eval import arity as _tree_arity
from pymetta.comboreduct.table.table import complete_truth_table
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# 1. The truth-table-unsoundness predicate (CARRY-FORWARD.md Β§3)
# ---------------------------------------------------------------------------


def _is_and_or(v) -> bool:
    """`v` is a bare `and`/`or` builtin -- house convention (interpreter.py,
    vertex.py) of an `isinstance` guard before any `==` against a
    `builtin` member, since `contin_t == builtin` can be `True` in Python
    purely by ordinal coincidence (CLAUDE.md's "type before value")."""
    return isinstance(v, builtin) and (v == builtin.logical_and or v == builtin.logical_or)


def _is_childless_junctor(tr: combo_tree, node: tree_node) -> bool:
    """A childless `and()`/`or()` -- the shape whose C++ evaluation
    (interpreter.cc:67-68, 76-77; ported at interpreter.py's `boolean_and`/
    `boolean_or` branches) returns its OWN operator token instead of a
    genuine `logical_true`/`logical_false`."""
    return _is_and_or(node.data) and tr.is_childless(node)


def _all_children_are_childless_junctors(tr: combo_tree, node: tree_node) -> bool:
    """True iff `node` is itself a junctor (`and`/`or`) with a NON-EMPTY
    child list, every one of whose direct children is a childless
    junctor.

    Deliberately requires a non-empty child list: a childless junctor
    vacuously has "all zero of its children" satisfy any per-child
    property, and this project's corpus is full of harmless trailing
    empty `and()`/`or()` siblings that must NOT be flagged just for being
    childless themselves -- only a junctor whose entire (real, non-empty)
    child list is made of OTHER childless junctors triggers the mechanism
    `tt_unsound_by_construction` below documents.
    """
    if not _is_and_or(node.data):
        return False
    children = list(tr.children(node))
    if not children:
        return False
    return all(_is_childless_junctor(tr, c) for c in children)


def tt_unsound_by_construction(tr: combo_tree) -> bool:
    """A structural indicator for one immediately-present unsound shape.
    (Task 6 fix round, review Important finding 1 -- see this module's
    own docstring, item 1, for the measured evidence): True iff `tr`
    contains a node whose entire (non-empty) child list consists of
    childless junctors -- the shape docs/CARRY-FORWARD.md Β§3 (amended
    2026-09-07) identifies as classic's `eval_logical_identities`
    unsoundness *mechanism* under `logical_reduce`. The mechanism
    firing does NOT guarantee the WHOLE TREE's truth table actually
    diverges -- root-level occurrences can fold harmlessly, and a
    genuinely diverging subtree can be masked by an enclosing context
    that supplies the same value another way (see
    `test_pinned_context_dependent_rows` in test_reduct_oracle.py for
    five rows measured directly against `$CLASSIC`'s own `test_reduct`:
    three flagged rows are truth-table-preserved, one flagged row diverges,
    and one initially unflagged row diverges after an earlier rewrite).
    Absence is not a guarantee either: an earlier rule can simplify an
    ordinary subtree into this shape before `eval_logical_identities`
    sees it. The same pinned test includes an effort-1 example that is
    initially unflagged and nevertheless diverges.

    Mechanism (Β§3, Β§4.3 of CARRY-FORWARD.md): a childless junctor
    evaluates to its OWN token (interpreter.cc:67-68/76-77), which a
    PARENT's short-circuit test (`== logical_false` inside an `and`'s
    evaluation loop, `== logical_true` inside an `or`'s) never matches --
    so the childless child is silently treated as "skip me", regardless
    of which operator it actually is. When EVERY child of some junctor J
    is itself a childless junctor, none of them can ever supply a value
    J's own evaluation could depend on -- all of them are "skipped" -- so
    J's evaluated result is forced to whichever constant an empty fold of
    J's OWN operator produces (`logical_true` for `and`, `logical_false`
    for `or`; interpreter.py's own "for correct foldr behaviour" comment).
    `eval_logical_identities::operator()`'s own later erasure
    (logical_rules.cc:134 for `and`'s empty-`or()` child, :155 for `or`'s
    empty-`and()` child -- NOT `reduce_to_enf`; this function opens at
    logical_rules.cc:130 and runs, via `pre_subtree_to_enf`, BEFORE
    `subtree_to_enf`, not after it) does not preserve this in general:
    each erasure is individually faithful to "this child can be
    skipped", but J itself becomes childless once all its children are
    erased this way, and is THEN erased from ITS OWN parent by the exact
    same rule on the next pass -- discarding the forced-constant fact the
    composition of otherwise-correct single steps was supposed to
    preserve, UNLESS J's parent (or an ancestor) happens to already
    assert the same fact through some other branch, in which case the
    loss is invisible at the whole-tree level -- exactly the masking the
    pinned rows demonstrate.

    Computed fresh, every call, from tree SHAPE alone -- independent by
    design of tests/fixtures/reduct.json's own `known_tt_divergent` field
    (see the module docstring's opening paragraph for why that
    independence, not merely a value that happens to agree with it, is
    the actual requirement here).
    """
    return any(_all_children_are_childless_junctors(tr, n) for n in tr)


# ---------------------------------------------------------------------------
# 2. Three-outcome truth-table evaluation (Β§4.3's token semantics)
# ---------------------------------------------------------------------------


class _RootIsBareJunctorToken:
    """Sentinel type: `tr`'s root itself is a childless `and()`/`or()`, so
    evaluating `tr` returns a raw operator token, not `logical_true`/
    `logical_false`. `builtin_to_bool` (vertex.h:708-713) `OC_ASSERT`s on
    exactly this input -- precisely: `OC_ASSERT` (`opencog::cassert`,
    oc_assert.cc:46) *throws* `opencog::AssertionException`, it does not
    itself call `abort()`. Whether the PROCESS then dies is up to the
    caller, and both outcomes are real, measured facts, not just one of
    them: `$CLASSIC`'s own `tests/comboreduct/reduct/test_reduct.cc` does
    NOT catch it -- confirmed by actually running it on `and()`/`or()`
    input: `terminate called after throwing an instance of
    'opencog::AssertionException'`, `what(): builtin should be
    'id::logical_true' or 'id::logical_false'.`, then a core dump (exit
    134) -- while this project's own `tools/cpp_probe/
    reduct_effort_probe.cc` wraps the same call in `catch (const
    std::exception&)` and handles it cleanly, recovering to report the
    next input. `complete_truth_table` (table.py) calls `builtin_to_bool`
    unconditionally on every row, matching the C++ side's own unguarded
    call rather than working around it. This sentinel lets a caller
    detect and report that case as its own, distinct outcome instead of
    treating the resulting `AssertionError` as an ordinary, unrelated
    bug.
    """


ROOT_TOKEN = _RootIsBareJunctorToken()


def root_is_bare_junctor_token(tr: combo_tree) -> bool:
    """True iff `tr`'s root is a childless `and()`/`or()` -- the ONLY
    shape from which `boolean_interpreter.boolean_eval` can return a raw
    operator token as the OUTPUT of a whole-tree evaluation.

    Proof (by exhaustive case analysis of `interpreter.py::boolean_eval`'s
    own `match`): the `argument` branch returns an already-resolved
    `boolean_inputs[...]` entry or its `negate_vertex` (itself raises
    unless that entry already IS `logical_true`/`logical_false`); the
    `logical_not` branch returns `negate_vertex(...)` of its child's own
    evaluation (same constraint); the non-empty `logical_and`/`logical_or`
    branches only ever `return builtin.logical_true`/`logical_false`
    literally, after a loop that COMPARES each child's evaluation against
    `logical_false`/`logical_true` (`==`) but never forwards a child's raw
    result as its own return value. The ONLY branches that return a raw,
    unresolved token AT ALL are the childless-`and`/`or` branches
    themselves (`return builtin.logical_and` / `return builtin.logical_or`,
    "for correct foldr behaviour"). A raw token can therefore only reach
    the very top of `boolean_eval(tr.begin())` -- the value
    `complete_truth_table`/`builtin_to_bool` actually consumes -- if
    `tr.begin()` (the root) IS itself one of those two branches, i.e. the
    root is a childless `and`/`or`. No other tree shape, at any depth, can
    produce this outcome: every other branch either resolves a child's
    token into a definite `logical_true`/`logical_false` before returning,
    or raises trying.
    """
    root = tr.begin()
    return _is_and_or(root.data) and tr.is_childless(root)


def evaluate_truth_table(tr: combo_tree, arity: int | None = None):
    """Evaluate `tr` over every input combination, Β§4.3-faithfully.

    Returns `ROOT_TOKEN` (not a tuple) when `root_is_bare_junctor_token
    (tr)` -- the distinct third outcome the task brief requires -- instead
    of raising, matching what a real, non-aborting observer of C++'s
    `OC_ASSERT` would see (this port raises a catchable `AssertionError`
    from `builtin_to_bool` rather than calling `abort()`, but the POINT
    the task brief makes is architectural: this is not "preserved" and
    not "violated", it is a third thing). For every other tree, returns
    exactly what `tuple(complete_truth_table(tr, arity))` would -- this
    function does not reimplement row evaluation, it only adds the
    root-token guard `complete_truth_table` itself deliberately does not
    have (see that class's own module-level comment: it mirrors table.h's
    unguarded `OC_ASSERT` on purpose).
    """
    if root_is_bare_junctor_token(tr):
        return ROOT_TOKEN
    return tuple(complete_truth_table(tr, arity))


def truth_table_preserved(before: combo_tree, after: combo_tree) -> bool | None:
    """True/False if `after` (typically `before` post-`logical_reduce`)
    preserves `before`'s truth table, `None` if either tree's evaluation
    hits the third, root-token outcome (`ROOT_TOKEN` -- see
    `evaluate_truth_table`'s own docstring).

    LOAD-BEARING DETAIL, mirroring `$CLASSIC/tests/comboreduct/reduct/
    test_reduct.cc` and this project's own `tools/cpp_probe/reduct_effort_
    probe.cc` exactly: `after` is evaluated at `before`'s OWN arity
    (`complete_truth_table tt2(tr, integer_log2(tt1.size()))` in both C++
    probes), never at whatever `after`'s own arity would independently
    infer. Reduction can make a referenced variable disappear entirely
    (e.g. `and($1 !$1)` reducing to `false`, which alone has inferred
    arity 0) -- comparing two truth tables inferred at DIFFERENT arities
    would silently compare tables of different lengths (`list.__eq__`
    between a length-1 and a length-8 list is simply `False`, not an
    error), which is a distinct bug from an actual semantic divergence
    and must not be conflated with one.
    """
    if root_is_bare_junctor_token(before) or root_is_bare_junctor_token(after):
        return None
    n = _tree_arity(before)
    return evaluate_truth_table(before, n) == evaluate_truth_table(after, n)


# ---------------------------------------------------------------------------
# 3. The seven ENF axioms (elegant-normal-forms-python's EnfRuleCheckers.py)
#
# elegant-normal-forms-python's own tree (DataStructures.Trees.TreeNode)
# gives every AND node a `guardSet` (its literal conjuncts, held OUT OF
# `.children`) and a `children` list holding only OR/AND substructure; a
# bare literal that would otherwise sit directly under an OR gets AUTO-
# WRAPPED, by its own `GatherJunctors.py`, into a synthetic childless AND
# node with a singleton guardSet, so `.children` entries are ALWAYS
# AND/OR-typed there, never a bare literal.
#
# combo_tree has no such split: an `and`/`or` node's DIRECT children are a
# flat mix of literals (`argument` vertices, sign encoding negation --
# `argument.py`'s own docstring) and further and/or subtrees (confirmed
# empirically against this project's own fixture: e.g. effort-3's real,
# byte-identical `and(or(!$4 $5) $3)`, `or(!$1 !$3 !$4)` both have a bare
# literal as a DIRECT child of an `or`). The translation used throughout
# this section:
#
#   * `_guard_set(tr, node)` -- node's own literal conjuncts:
#       - node is a bare literal          -> {node} (itself, as a
#         singleton "implicit childless AND" -- exactly GatherJunctors's
#         own auto-wrap, done virtually instead of by mutating the tree)
#       - node is `and`                   -> its literal DIRECT children
#       - node is `or`                    -> {} (an OR carries no guard
#         set of its own; only AND-shaped nodes, real or implicit, do)
#   * `_struct_children(tr, node)` -- node's substructure for recursion:
#       - node is a bare literal          -> [] (a leaf)
#       - node is `and`                   -> its NON-literal children
#         (literal children are guard-set members, not substructure)
#       - node is `or`                    -> ALL direct children (every
#         non-literal one is real substructure; every literal one is an
#         implicit terminal AND, per `_guard_set` above)
#   * `_is_and_type(node)` -- "this node behaves as an AND for rule
#     2/4/6/7's purposes" -- true for a real `and` OR a bare literal
#     (the implicit-AND case); `_is_or_type(node)` -- true for a real
#     `or` only.
#
# Verified (not merely assumed) against this project's own committed
# fixture BEFORE writing the rules below: `reduced` strings at effort 2/3
# never contain the substring `not(` (reduce_nots + subtree_to_enf leave
# every negation encoded as a signed `argument`, never a `not(...)`
# wrapper) -- see test_reduct_oracle.py's own
# test_no_residual_not_wrapper_in_enf_shaped_output for that same check
# kept live as a regression, not just asserted once here in a comment.
# ---------------------------------------------------------------------------


def _is_literal(node: tree_node) -> bool:
    return is_argument(node.data)


def _is_and(node: tree_node) -> bool:
    return isinstance(node.data, builtin) and node.data == builtin.logical_and


def _is_or(node: tree_node) -> bool:
    return isinstance(node.data, builtin) and node.data == builtin.logical_or


def _is_and_type(node: tree_node) -> bool:
    return _is_and(node) or _is_literal(node)


def _literal_key(node: tree_node) -> tuple[int, bool]:
    """A literal's identity for set membership: (variable, is_positive).
    `argument.idx`'s sign already encodes negation (argument.py's own
    module docstring) -- `abs(idx)` names the variable, `idx > 0` its
    polarity in THIS occurrence."""
    idx = node.data.idx
    return (abs(idx), idx > 0)


def _guard_set(tr: combo_tree, node: tree_node) -> set[tuple[int, bool]]:
    if _is_literal(node):
        return {_literal_key(node)}
    if _is_and(node):
        return {_literal_key(c) for c in tr.children(node) if _is_literal(c)}
    return set()


def _struct_children(tr: combo_tree, node: tree_node) -> list[tree_node]:
    if _is_literal(node):
        return []
    if _is_and(node):
        return [c for c in tr.children(node) if not _is_literal(c)]
    return list(tr.children(node))


def _iter_enf_nodes(tr: combo_tree):
    """Every node this section's rules should independently visit/check:
    the root, every real `and`/`or`, and every literal that is NOT itself
    a guard-set member of some enclosing real AND (a literal reached only
    via `_struct_children` -- i.e. one sitting directly under an OR, or
    the tree's sole root node).

    THIS IS THE FIX for a real bug found empirically while validating this
    module against the project's own committed fixture (not merely
    reasoned about): an earlier version of rules 2/3/4/5/6/7 iterated
    `for node in tr:` -- combo_tree's own flat pre-order walk over EVERY
    node, with no notion of "guard-set member, already absorbed by its
    parent". That over-counted every literal DIRECT CHILD OF A REAL AND
    as its OWN independent implicit-AND-to-check, on top of it already
    being counted as part of its parent's own guard set -- e.g. `and($1
    !$3)` (`testTrees.sh#4`'s reduced form) has root guard_set = {(1,T),
    (3,F)}; blindly visiting `$1` itself as a second, independent
    "implicit AND" found `_guard_set($1) = {(1,T)}` intersecting the
    ROOT's own dominant-set walk (which starts at $1's PARENT, i.e. root
    itself, whose guard_set already contains (1,T) BECAUSE it contains
    $1) -- a false rule_six violation on 82 of 158 effort-2/3 cases before
    this fix, traced to exactly this double-counting, not a real defect.
    `elegant-normal-forms-python`'s own model never has this hazard: its
    `TreeNode.children` list is, BY CONSTRUCTION (`GatherJunctors.py`),
    never populated with a literal that is a guard-set member of an AND
    -- only `_struct_children` reproduces that same exclusion here, so
    every rule below walks via this helper (or, for rule_one, via
    `_struct_children` directly during its own recursive descent) instead
    of raw tree iteration.
    """

    def walk(node: tree_node):
        yield node
        for child in _struct_children(tr, node):
            yield from walk(child)

    yield from walk(tr.begin())


def rule_one(tr: combo_tree) -> bool:
    """ENF Rule #1: every root-to-leaf branch's accumulated guard set is
    internally consistent -- no variable appears both positive and
    negated along the same branch.

    Checked INCREMENTALLY, as each node's own guard set is merged in
    (rather than deferred to the leaf, `EnfRuleCheckers.ruleOne`'s own
    approach): the merged branch dict only ever grows root-to-leaf, so a
    conflict is detected the first time it can possibly exist, which is
    provably the same answer a leaf-deferred check would give -- a
    conflict present at a leaf must already have been introduced by some
    ancestor merge on the way there.
    """

    def walk(node: tree_node, branch: dict[int, bool]) -> bool:
        branch = dict(branch)
        if _is_and_type(node):
            for var, sign in _guard_set(tr, node):
                if var in branch and branch[var] != sign:
                    return False
                branch[var] = sign
        for child in _struct_children(tr, node):
            if not walk(child, branch):
                return False
        return True

    return walk(tr.begin(), {})


def rule_two(tr: combo_tree) -> bool:
    """ENF Rule #2: every terminal AND node (real or implicit-literal)
    has a non-empty guard set."""
    for node in _iter_enf_nodes(tr):
        if _is_and_type(node) and not _struct_children(tr, node):
            if not _guard_set(tr, node):
                return False
    return True


def rule_three(tr: combo_tree) -> bool:
    """ENF Rule #3: every OR node has at least two children."""
    for node in _iter_enf_nodes(tr):
        if _is_or(node) and tr.number_of_children(node) < 2:
            return False
    return True


def rule_four(tr: combo_tree) -> bool:
    """ENF Rule #4: every non-root, non-terminal, empty-guard-set AND
    node has at least two (structural) children -- i.e. no redundant
    unary `and(X)` wrapper survives (that shape is what `remove_unary_
    junctors` exists to clean up; this rule is meaningful precisely on
    output that has already passed through it -- see test_reduct_oracle.py
    for where these checks are actually applied)."""
    root = tr.begin()
    for node in _iter_enf_nodes(tr):
        if node is root or not _is_and(node):
            continue
        struct = _struct_children(tr, node)
        if struct and not _guard_set(tr, node) and len(struct) < 2:
            return False
    return True


def rule_five(tr: combo_tree) -> bool:
    """ENF Rule #5: for every OR node, the intersection of its children's
    own guard sets is empty (no literal common to every disjunct -- that
    would be a factorable redundancy)."""
    for node in _iter_enf_nodes(tr):
        if not _is_or(node):
            continue
        children = _struct_children(tr, node)
        if len(children) < 2:
            continue
        guard_sets = [_guard_set(tr, c) for c in children]
        if set.intersection(*guard_sets):
            return False
    return True


def _dominant_set(tr: combo_tree, node: tree_node) -> set[tuple[int, bool]]:
    """Union of the guard sets of every AND ancestor of `node` (a tree has
    a unique root-to-node path, so "same branch, closer to the root" --
    `EnfRuleCheckers.ruleSix`'s own definition of "dominates" -- reduces
    to "is a proper ancestor"). Only a REAL `and` node can dominate
    anything (an implicit-literal AND is always a leaf, with no
    descendants to dominate)."""
    result: set[tuple[int, bool]] = set()
    p = node.parent
    while p is not None:
        if _is_and(p):
            result |= _guard_set(tr, p)
        p = p.parent
    return result


def rule_six(tr: combo_tree) -> bool:
    """ENF Rule #6: every AND node's (real or implicit) guard set is
    disjoint from its dominant set -- no literal is re-asserted by a
    node whose ancestor already guarantees it."""
    for node in _iter_enf_nodes(tr):
        if _is_and_type(node) and _guard_set(tr, node) & _dominant_set(tr, node):
            return False
    return True


def _command_set(tr: combo_tree, node: tree_node) -> set[tuple[int, bool]]:
    """Union of the guard sets of every node A such that: A is a
    STRUCTURAL child (`_struct_children` -- i.e. one `_iter_enf_nodes`
    would itself visit, NOT a guard-set member already absorbed by its
    own parent, see that function's docstring) of some ancestor of
    `node`, A is a terminal AND (real or implicit) with exactly one guard
    literal, and A is not itself on the root-to-`node` path
    (`EnfRuleCheckers.ruleSeven`'s own "A commands B" definition, walked
    directly via `.parent` rather than that module's flat-list/level-
    counter reimplementation of the same walk).

    Uses `_struct_children(parent)`, not raw `tr.children(parent)`: the
    same double-counting hazard `_iter_enf_nodes` documents applies here
    too -- a literal sibling that is really a guard-set member of a real
    AND `parent` is not an independent "commanding" node A in its own
    right (elegant-normal-forms-python's own `.children` never contains
    one either, by `GatherJunctors.py`'s construction).
    """
    result: set[tuple[int, bool]] = set()
    child_on_path = node
    parent = node.parent
    while parent is not None:
        for sib in _struct_children(tr, parent):
            if sib is child_on_path:
                continue
            guard = _guard_set(tr, sib)
            if _is_and_type(sib) and not _struct_children(tr, sib) and len(guard) == 1:
                result |= guard
        child_on_path = parent
        parent = parent.parent
    return result


def rule_seven(tr: combo_tree) -> bool:
    """ENF Rule #7: every AND node's (real or implicit) guard set is
    disjoint from its command set."""
    for node in _iter_enf_nodes(tr):
        if _is_and_type(node) and _guard_set(tr, node) & _command_set(tr, node):
            return False
    return True


_RULES = {
    "rule_one": rule_one,
    "rule_two": rule_two,
    "rule_three": rule_three,
    "rule_four": rule_four,
    "rule_five": rule_five,
    "rule_six": rule_six,
    "rule_seven": rule_seven,
}


def enf_violations(tr: combo_tree) -> list[str]:
    """Names of every ENF axiom `tr` fails, or `[]` if `tr` is fully ENF."""
    return [name for name, check in _RULES.items() if not check(tr)]
