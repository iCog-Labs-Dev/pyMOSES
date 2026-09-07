"""C++: moses/comboreduct/reduct/general_rules.{h,cc}

The type-independent reduction rules: `level` (flatten associative
functions), `eval_constants` (fold pure-constant sub-expressions),
`reorder_commutative` (canonical child order for commutative operators),
`remove_null_vertices` (drop `null_vertex` sentinels), and
`remove_all_assumptions`.

Seven vertex.h algebraic-property predicates general_rules.cc calls
(`is_associative`, `is_commutative`, `is_symmetric`, `is_constant`,
`is_boolean`, `is_action_result`, `is_indefinite_object`) live here rather
than in vertex.py: general_rules.cc is their only call site anywhere under
`moses/comboreduct/reduct/` (checked directly -- `grep -rn
"is_associative\\|is_commutative\\|is_constant\\b\\|is_indefinite_object\\|
is_boolean\\b\\|is_action_result" moses/comboreduct/reduct/` matches only
general_rules.cc; `is_boolean` also appears in `mixed_rules.cc`, S14/out of
scope). This mirrors eval.py's own `is_procedure_call` precedent: a real,
narrow C++ predicate ported at the one file that needs it, not promoted to
shared vertex.py speculatively.

Every one of these predicates is a C++ *template* (`template<typename T>
inline bool is_associative(const T& v)`, instantiated here as
`is_associative(v)` over `v: vertex`) that compares `v` against several
`id::` constants with bare `==`. Real `vertex`, being a `boost::variant`,
never true-positives across held alternatives; this port's `vertex_equal`
(vertex.py) reproduces exactly that discrimination for the `contin_t`/
`builtin` collision `1.0 == builtin.logical_and` would otherwise produce in
bare Python -- so every `v == id::xxx` below is ported as
`vertex_equal(v, builtin.xxx)`, never bare `==` (this file's own "type
before value" case, CLAUDE.md's Global Constraints).

`is_associative`/`is_commutative` also compare `v` against
`sequential_and`/`sequential_or`/`sequential_exec` (`combo/action.h:38`'s
`action` enum -- legacy embodiment action sequencing) and, via
`is_symmetric`, against whatever `is_perception(v)` guards. Neither
`action` nor `perception` is one of this port's four vertex alternatives
(vertex.py's module docstring), so `vertex_equal` against any of them can
never be `True` for any vertex this port constructs -- those comparisons
are omitted below as permanently-unreachable, rather than ported as dead
`vertex_equal(v, ...)` calls with nothing on the right-hand side to
construct.

`general_rules_probe.cc`/`flat_normal_form_probe.cc` under
`tools/cpp_probe/` are the ground truth this file's tests are pinned
against -- see task-3-report.md for the full measured-output listing.
"""

from pymetta.comboreduct.combo.assumption import delete_all_assumptions
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    is_contin,
    is_enum_type,
    lexicographic_subtree_less,
    vertex,
    vertex_equal,
)
from pymetta.comboreduct.crutil.exception import OverflowException
from pymetta.comboreduct.interpreter.eval import eval_throws_binding
from pymetta.comboreduct.reduct.reduct import rule
from pymetta.util.tree import tree_node

# ---------------------------------------------------------------------------
# vertex.h algebraic-property predicates general_rules.cc needs -- see this
# module's own docstring for why they live here rather than in vertex.py.
# ---------------------------------------------------------------------------


def is_associative(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:548 is_associative

    Scoped to this port's four vertex alternatives -- see module
    docstring on the omitted `sequential_and`/`sequential_or`/
    `sequential_exec` (`action`-typed) comparisons.
    """
    return (
        vertex_equal(v, builtin.logical_and)
        or vertex_equal(v, builtin.logical_or)
        or vertex_equal(v, builtin.plus)
        or vertex_equal(v, builtin.times)
    )


def is_symmetric(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:599 is_symmetric

    `is_perception(v) ? get_perception(v)->is_symmetric() : false` --
    `perception` is not one of this port's vertex alternatives (vertex.py's
    module docstring), so `is_perception` is always False and this always
    returns False. Ported only because `is_commutative` below calls it
    directly (vertex.h:560).
    """
    return False


def is_commutative(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:557 is_commutative"""
    return (
        vertex_equal(v, builtin.logical_and)
        or vertex_equal(v, builtin.logical_or)
        or vertex_equal(v, builtin.plus)
        or vertex_equal(v, builtin.times)
        or is_symmetric(v)
    )


def is_boolean(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:732 is_boolean"""
    return vertex_equal(v, builtin.logical_true) or vertex_equal(v, builtin.logical_false)


def is_action_result(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:180 is_action_result

    Declared at vertex.h:681, defined here. `v == id::action_failure ||
    v == id::action_success` -- both
    `action_result_type` values, not one of this port's vertex
    alternatives (vertex.py's module docstring; CLAUDE.md's "Known sharp
    edges" on the stale `action_result_type` knob path), so always False
    here. Ported only because `is_constant` below calls it directly
    (vertex.h:756).
    """
    return False


def is_constant(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.h:753 is_constant"""
    return is_boolean(v) or is_contin(v) or is_enum_type(v) or is_action_result(v)


def is_indefinite_object(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:65 is_indefinite_object

    Declared at vertex.h:626, defined here.
    `boost::get<indefinite_object>(&v)` -- `indefinite_object` is not one
    of this port's vertex alternatives (vertex.py's module docstring), so
    always False here. Mirrors eval.py's `is_procedure_call` precedent: a
    real, exercised (`eval_constants`'s childless-node branch below) but
    permanently-False C++ predicate.
    """
    return False


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


class level(rule):
    """C++: moses/comboreduct/reduct/general_rules.h:38 level

    Flattens all associative functions: f(a,f(b,c)) -> f(a,b,c). Note
    that `level` is recursive, i.e. f(a,f(b,f(c,d))) -> f(a,b,c,d) --
    because `sib = tr.erase(tr.flatten(sib))` reassigns `sib` to erase()'s
    own returned successor (the first newly-promoted grandchild) rather
    than blindly `++sib`-ing past it, a grandchild that is ITSELF the
    same associative operator is re-examined by this SAME pass, one
    flatten at a time, all the way down. Measured directly
    (`tools/cpp_probe/general_rules_probe.cc`'s `level_flatten_deep`):
    `and($1 and($2 and($3 $4)))` reduces to `and($1 $2 $3 $4)` in one
    `apply()` call.
    """

    def __init__(self, name: str = "level") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/general_rules.cc:34 level::operator()"""
        if is_associative(it.data):
            sib = it.first_child
            while sib is not None:
                if vertex_equal(sib.data, it.data):
                    sib = tr.erase(tr.flatten(sib))
                else:
                    sib = sib.next_sibling


class eval_constants(rule):
    """C++: moses/comboreduct/reduct/general_rules.h:44 eval_constants

    Evaluates sub-expressions when possible. If an operator is
    commutative, op(const,var,const) becomes op(op(const,const),var),
    e.g. +(2,x,1) -> +(3,x).
    """

    def __init__(self, name: str = "eval_constants") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/general_rules.cc:47 eval_constants::operator()

        `tr.partition(it, is_constant)` -- scoped to `it`'s FULL child
        range, matching `tr.partition`'s own scoping (tree.py) and the
        one real C++ call site (`tr.partition(it.begin(), it.end(),
        is_constant<vertex>)`). NOT a stable partition -- see
        `tree.partition`'s own docstring and task-3-report.md; the
        constants-vs-non-constants boundary this returns, and the
        resulting order WITHIN each group, is pinned against
        `tools/cpp_probe/general_rules_probe.cc`'s
        `eval_constants_mixed_plus_interleaved` case, not derived from
        first principles.

        `it = it.last_child()` (general_rules.cc:66) is ported as
        `it = new_op` directly (the node `tr.append_child` just
        returned), rather than re-reading `it.last_child` off the
        tree_node afterward -- both are equal after `tr.reparent` (the
        moved range never reaches `it`'s own last child, `new_op`, since
        `to != it.end()` is required to enter this branch at all), but
        holding the already-known reference is more direct and does not
        depend on that bookkeeping invariant continuing to hold under
        future tree.py changes.
        """
        if tr.is_childless(it):
            if is_indefinite_object(it.data):
                it.data = eval_throws_binding([], it)
            return

        if is_associative(it.data):
            if is_commutative(it.data):
                to = tr.partition(it, is_constant)
                n_consts = 0
                sib = it.first_child
                while sib is not to:
                    n_consts += 1
                    sib = sib.next_sibling
                if n_consts < 2 and not (n_consts == 1 and tr.has_one_child(it)):
                    return
                if to is not None:
                    new_op = tr.append_child(it, it.data)
                    tr.reparent(new_op, it.first_child, to)
                    it = new_op
            else:
                raise AssertionError("Not implemented yet")
        else:
            sib = it.first_child
            while sib is not None:
                if not is_constant(sib.data):
                    return
                sib = sib.next_sibling

        # We pass an empty vertex sequence, as there should be no
        # arguments in the tree below 'it'. Viz, the only things we
        # expect to evaluate are things like 'not(true)', '*(-1.75 1)'
        # '+(2 0 6)' and so on. If there are args, then hopefully an
        # exception will be thrown :-)
        #
        # Err, well, its common for knob-building to generate things
        # log(0) or /(1 0) and so we want to catch that inf, and
        # propagate it along. (later on, there may be a divde-by-inf
        # or maybe a 0<(-inf) predicate, etc. all of which can be legally
        # evaluated to return valid results.
        try:
            it.data = eval_throws_binding([], it)
        except OverflowException as e:
            it.data = e.get_vertex()
        tr.erase_children(it)


class reorder_commutative(rule):
    """C++: moses/comboreduct/reduct/general_rules.h:53 reorder_commutative

    Reorder children of commutative operators (should be applied
    upwards). SORTS THE PRINTED CHILDREN of every commutative node in
    the final reduced tree -- feeds Task 6's byte-identical oracle
    directly (vertex.py's own note on `lexicographic_subtree_order`).
    """

    def __init__(self, name: str = "reorder_commutative") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/general_rules.cc:101 reorder_commutative::operator()

        `tr.sort_on_subtrees(it.begin(), it.end(),
        lexicographic_subtree_order<vertex>(), false)` -- `deep=false`
        (the C++ literal argument); `lexicographic_subtree_less`
        (vertex.py) is the promoted comparator, used here verbatim per
        the task brief ("use the promoted one, do not write a second").
        """
        if is_commutative(it.data):
            tr.sort_on_subtrees(it, lexicographic_subtree_less, False)


class remove_null_vertices(rule):
    """C++: moses/comboreduct/reduct/general_rules.h:59 remove_null_vertices

    Get rid of subtrees marked with a `null_vertex` in their roots.
    """

    def __init__(self, name: str = "remove_null_vertices") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/general_rules.cc:110 remove_null_vertices::operator()

        Most nodes take simple lists; but not `cond`. `cond` takes
        clauses, which are pairs. If we remove the condition, we must
        also remove the consequent.

        FIX WAVE (self-review): the `cond` branch's non-null `else`
        walks the pair-stride by advancing `sib` TWICE
        (`general_rules.cc:128-129`'s `++sib; ++sib;`). A `cond` node's
        own real shape (eval.cc's own cond-evaluation loop, already
        ported at `eval.py`'s `builtin.contin_if | builtin.cond` case)
        is (condition, consequent) PAIRS plus one final, UNPAIRED "else"
        value -- an ODD total child count. Landing exactly on that
        trailing "else" during the pair-stride walk is real, reachable
        input (this class's own `remove_null_vertices_cond` test), and
        C++'s `sibling_iterator::operator++` (opencog/util/tree.h:2930)
        silently no-ops once already at `end()` (`if (this->node) {...}`,
        no `else`) -- so C++'s SECOND `++sib` there is a safe no-op, not
        a crash. A literal `sib = sib.next_sibling` twice is NOT safe the
        same way (`None.next_sibling` raises `AttributeError`) -- this
        guards the second advance to reproduce operator++'s own
        past-the-end no-op, confirmed against the real library
        (`tools/cpp_probe/general_rules_probe.cc`'s
        `remove_null_vertices_cond` case: this exact odd-count shape
        completes without crashing in C++, producing `cond($2 $3 $4)`).

        # DIVERGENCE(C++ general_rules.cc:123-125): a trailing-else null
        # segfaults real C++; we raise.
        # CORRECTION (task-3 review): an earlier version of this docstring
        # claimed "a null CONDITION always has a real paired consequent by
        # cond's own structural contract", used to justify NOT guarding the
        # two-erase branch the same way the advance branch above is
        # guarded. That claim does not cover the trailing "else" slot
        # itself, which the pair-stride walk visits (and null-checks) as
        # if it were a condition whenever it lands there -- a genuinely
        # malformed but real, constructible input, not merely
        # hypothetical. Measured directly, isolated in its own binary so a
        # crash could not take out other probe cases
        # (`tools/cpp_probe/remove_null_vertices_cond_crash_probe.cc`,
        # `bad1`/`bad2` cases, `( ulimit -v 1048576; timeout 20 <binary>
        # bad1 )` -> exit 139, `timeout: the monitored command dumped
        # core`): `cond($1 $2 null)` (null IS the trailing else) and
        # `cond(null $1 null)` (a real null condition followed by a SECOND
        # null in the trailing else slot) both segfault the real
        # `libcomboreduct.so` -- `tr.erase(sib)` erasing the trailing null
        # returns an `end()`/null successor, and the SECOND `tr.erase(sib)`
        # then dereferences that null iterator. This port's `tr.erase(None)`
        # raises `AttributeError` at the same point instead of dereferencing
        # a null pointer -- a strictly safer failure mode (loud, catchable,
        # no memory corruption) than C++'s undefined behaviour, but a
        # real, disclosed difference in KIND of failure, not a "both just
        # crash so it doesn't matter" equivalence. `good1`
        # (`cond($1 null $2)`, null as a plain consequent -- never
        # inspected) and `good2` (`cond($1 $2 $3 null)`, null lands on a
        # CONSEQUENT-position skip for this particular child count) both
        # complete identically in C++ and this port (verified by the same
        # probe), confirming the crash is specific to a null actually
        # occupying the trailing else slot, not something this whole
        # branch does wrong in general.
        """
        if not vertex_equal(it.data, builtin.cond):
            sib = it.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.null_vertex):
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
        else:
            sib = it.first_child
            while sib is not None:
                if vertex_equal(sib.data, builtin.null_vertex):
                    sib = tr.erase(sib)
                    sib = tr.erase(sib)
                else:
                    sib = sib.next_sibling
                    if sib is not None:
                        sib = sib.next_sibling


class remove_all_assumptions(rule):
    """C++: moses/comboreduct/reduct/general_rules.h:65 remove_all_assumptions

    Remove all assumptions.
    """

    def __init__(self, name: str = "remove_all_assumptions") -> None:
        super().__init__(name)

    def apply(self, tr: combo_tree, it: tree_node) -> None:
        """C++: moses/comboreduct/reduct/general_rules.cc:134 remove_all_assumptions::operator()"""
        delete_all_assumptions(tr)
