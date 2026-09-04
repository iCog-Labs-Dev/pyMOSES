"""C++: moses/comboreduct/interpreter/interpreter.{h,cc}

`boolean_interpreter` (interpreter.h:46, interpreter.cc:37-102) -- evaluates
a combo tree of booleans (`argument`, `logical_and/or/not`,
`logical_true/false`) against a vector of `builtin` inputs.

`interpreter_visitor` lives elsewhere in classic (table.h:304-351, not this
directory -- see task-4-brief.md) but is ported alongside
`boolean_interpreter` here because its whole job is choosing which
sub-interpreter to run. Only its `std::vector<builtin>` overload is
implemented (see its class docstring): `contin_interpreter` and
`mixed_interpreter` are out of scope for the entire S0-S6 foundation cycle,
not just this task (docs/superpowers/plans/2026-09-03-pymetta-s0-s6-foundation.md's
"Known gap, deliberate" note scopes this cycle to the boolean path only).

Both classes dispatch on vertex kind with `match` (mirroring
`boost::apply_visitor`/the C++ `switch` on a `boost::get<T>`), per house
style (see vertex.py, iostream_combo.py's `_vertex_to_str`).

NOT a DIVERGENCE (this is a reading of an informally-worded planning
document, not a deviation from C++ behaviour -- so it gets a plain note,
not a `# DIVERGENCE(C++ ...)` marker): task-4-brief.md's interface line
describes `boolean_interpreter.__call__(self, tr, inputs)` as a single
two-argument call. The actual C++ usage -- both in interpreter.cc's own
callers and in the oracle test this task ports (interpreterUTest.cxxtest:36,
`boolean_interpreter(ir.first)(tr)`) and in table.cc:340
(`interpreter_visitor`'s own use, `boolean_interpreter(inputs)(_it)`) --
is uniformly constructor-then-call: `inputs` is bound in `__init__`
(interpreter.h:49), and `__call__` (interpreter.h:52-53) takes only the
tree or iterator. Ported that way here, matching every real call site and
preserving the same two C++ overloads (`operator()(const combo_tree&)` and
`operator()(const combo_tree::iterator)`) as one Python `__call__` that
dispatches on whether it was handed a `tree` or a bare node -- the brief's
shorthand line is read as describing the calling convention informally,
not a literal signature (see the task's own "Read that C++ directly. Do
not port from prose" instruction).
"""

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    is_contin_expr,
    negate_vertex,
    vertex,
)
from pymetta.util.tree import tree, tree_node


class boolean_interpreter:
    """C++: moses/comboreduct/interpreter/interpreter.h:46 boolean_interpreter"""

    def __init__(self, inputs: list[builtin] | None = None) -> None:
        """C++: moses/comboreduct/interpreter/interpreter.cc:37 boolean_interpreter

        The constructor (`boolean_interpreter::boolean_interpreter`).
        `inputs` defaults to an empty list, mirroring the C++ default
        constructor argument `std::vector<builtin>()` (interpreter.h:49).
        C++ stores a `const std::vector<builtin>&` reference to the
        caller's vector; this port just keeps the list object itself
        (Python has no reference/value distinction to reproduce here).
        """
        self.boolean_inputs: list[builtin] = [] if inputs is None else inputs

    def __call__(self, tr_or_it: combo_tree | tree_node) -> builtin:
        """C++: moses/comboreduct/interpreter/interpreter.cc:40 boolean_interpreter::operator()

        Covers both C++ overloads: `operator()(const combo_tree& tr)`
        (interpreter.cc:40-42, calls `boolean_eval(tr.begin())`) and
        `operator()(const combo_tree::iterator it)` (interpreter.cc:43-45,
        calls `boolean_eval(it)` directly) -- see the module docstring's
        note on why this is one Python method rather than two.
        """
        it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
        return self.boolean_eval(it)

    def boolean_eval(self, it: tree_node) -> builtin:
        """C++: moses/comboreduct/interpreter/interpreter.cc:47 boolean_interpreter::boolean_eval

        `it` is a `combo_tree::iterator` in C++; here, a plain `tree_node`
        (pymetta/util/tree.py's module docstring -- iterators collapse to
        node objects in this port). Sibling children are walked directly
        via `it.first_child`/`.next_sibling` -- the same links
        `combo_tree::sibling_iterator::begin()`/`end()` wrap in C++ -- since
        this method, like its C++ counterpart, never has a `tree` object
        to call `.children()` on, only the node itself.
        """
        v = it.data

        # FIX WAVE (final review, Critical 2): the outer `match` dispatches
        # on TYPE ONLY (`argument()` / `builtin()` / the catch-all `_`) --
        # mirroring `_vertex_to_str`'s `match v: case argument(): ...; case
        # builtin(): ...` in iostream_combo.py, and matching C++'s own
        # `boost::get<builtin>(&v)` type-discriminate-before-switch
        # (interpreter.cc:57). Only once `case builtin():` has confirmed
        # `isinstance(v, builtin)` does the INNER `match` compare `v`
        # against specific builtin members by value. Previously the value
        # patterns (`case builtin.logical_false | builtin.logical_true`,
        # etc.) ran directly against `v` with no preceding type check:
        # since `contin_t` is a bare Python `float` and `builtin` an
        # `IntEnum`, a contin whose value happened to equal a builtin's
        # ordinal (1.0/2.0/3.0/4.0/5.0 vs. logical_true/logical_false/
        # logical_and/logical_or/logical_not) was silently misrouted into
        # these branches -- returning a float where the declared `->
        # builtin` contract promises a `builtin`, or worse, recursing into
        # `.first_child` of what is actually a childless contin leaf
        # (`None.data` -- a crash). C++ cannot exhibit this at all:
        # `boost::variant` alternatives never compare equal across types.
        # Verified against the real C++ (which OC_ASSERTs/aborts instead)
        # via a contin reaching `interpreter_visitor`/`boolean_interpreter`
        # -- see test_interpreter.py's integral-contin regression cases.
        match v:
            case argument():
                idx = v.idx
                if idx > 0:
                    return self.boolean_inputs[idx - 1]
                return negate_vertex(self.boolean_inputs[-idx - 1])

            case builtin():
                match v:
                    case builtin.logical_false | builtin.logical_true:
                        return v

                    case builtin.logical_and:
                        # C++: interpreter.cc:67-68 -- "For correct foldr
                        # behaviour": and() over no children is the
                        # identity, true.
                        if it.first_child is None:
                            return builtin.logical_and
                        sib = it.first_child
                        while sib is not None:
                            if self.boolean_eval(sib) == builtin.logical_false:
                                return builtin.logical_false
                            sib = sib.next_sibling
                        return builtin.logical_true

                    case builtin.logical_or:
                        # C++: interpreter.cc:76-77 -- or() over no
                        # children is the identity, false.
                        if it.first_child is None:
                            return builtin.logical_or
                        sib = it.first_child
                        while sib is not None:
                            if self.boolean_eval(sib) == builtin.logical_true:
                                return builtin.logical_true
                            sib = sib.next_sibling
                        return builtin.logical_false

                    case builtin.logical_not:
                        return negate_vertex(self.boolean_eval(it.first_child))

                    case _:
                        # C++: interpreter.cc:87-93 -- default switch case,
                        # OC_ASSERT(false, ...) naming the unhandled
                        # builtin. Silence here surfaces much later as a
                        # wrong score, so this raises rather than
                        # returning a placeholder.
                        raise ValueError(
                            f"boolean_interpreter does not handle builtin {v.name!r}"
                        )

            case _:
                # C++: interpreter.cc:95-101 -- the vertex is not even a
                # builtin (e.g. a bare contin_t leaf, or an argument
                # already handled above).
                raise ValueError(f"boolean_interpreter does not handle vertex {v!r}")


class interpreter_visitor:
    """C++: moses/comboreduct/table/table.h:304 interpreter_visitor

    Chooses the interpreter type based on the type of vertices appearing
    in the combo tree, exactly as classic does -- but only the
    `std::vector<builtin>` overload (table.h:328-331) is implemented.
    `contin_interpreter`/`mixed_interpreter` are out of scope for this
    whole port cycle (see module docstring), so every other input-type
    overload (`vector<contin_t>`, `vector<vertex>`, `string_seq`,
    `vector<combo_tree>` -- table.h:332-347) and the `mixed` case within
    the `vector<builtin>` overload both raise `NotImplementedError` naming
    the gap rather than silently mishandling it.
    """

    def __init__(self, tr_or_it: combo_tree | tree_node) -> None:
        """C++: moses/comboreduct/table/table.h:306 interpreter_visitor::interpreter_visitor

        Two real C++ overloads -- the combo_tree ctor (table.h:306-319)
        and the iterator ctor (table.h:321-326) -- both ported here as one
        constructor dispatching on whether it was handed a whole `tree` or
        a bare node -- same convention as `boolean_interpreter.__call__`
        above.

        KNOWN SHARP EDGE, ported faithfully (the C++ comment at table.h:323
        says so itself: "Cannot check the entire tree with this ctor"):
        given a bare node, only *that* node's own
        vertex is checked for mixed-ness, not its subtree. A `plus`
        several levels below the given node is invisible to this
        constructor even though the whole-tree constructor would have
        found it. See test_interpreter.py's
        test_interpreter_visitor_constructed_from_non_root_iterator_can_miss_mixedness.
        """
        # FIX WAVE (sweep for Critical 2's pattern): both
        # `... == builtin.greater_than_zero` comparisons below are guarded
        # by `isinstance(..., builtin)`. They are not currently reachable
        # with a contin operand -- `is_contin_expr` above always returns
        # True for one first, via its own `is_contin(v) or ...`
        # short-circuit, so `mixed` is already True and the loop/branch
        # exits before this line runs -- but that safety is an incidental
        # consequence of evaluation ORDER, not a type check at this
        # comparison's own site, so it is guarded directly too rather than
        # left to depend on the line above never being reordered.
        if isinstance(tr_or_it, tree):
            self._it = tr_or_it.begin()
            self.mixed = False
            for node in tr_or_it:
                self.mixed = is_contin_expr(node.data)
                if self.mixed:
                    break
                self.mixed = (
                    isinstance(node.data, builtin) and node.data == builtin.greater_than_zero
                )
                if self.mixed:
                    break
        else:
            self._it = tr_or_it
            self.mixed = is_contin_expr(self._it.data)
            if not self.mixed:
                self.mixed = (
                    isinstance(self._it.data, builtin)
                    and self._it.data == builtin.greater_than_zero
                )

    def __call__(self, inputs: list[builtin]) -> vertex:
        """C++: moses/comboreduct/table/table.h:328 interpreter_visitor::operator()

        The `operator()(const std::vector<builtin>&)` overload
        (table.h:328-331) -- the only one implemented, see class
        docstring. Requires both that `inputs` actually is a
        `vector<builtin>` (this port has no `multi_type_seq` variant to
        dispatch on at runtime the way `boost::apply_visitor` does;
        `list[builtin]` is the only concrete input type this method
        recognises) and that the tree is not `mixed` (a `plus`/`div`/...
        or `greater_than_zero` anywhere would route to
        `mixed_interpreter` in C++, table.h:329).
        """
        if not (isinstance(inputs, list) and all(isinstance(x, builtin) for x in inputs)):
            raise NotImplementedError(
                "interpreter_visitor in this port implements only the "
                "vector<builtin> overload (table.h:328) -- "
                "contin_interpreter/mixed_interpreter are out of scope for "
                f"this port cycle; got input type {type(inputs)!r}"
            )
        if self.mixed:
            raise NotImplementedError(
                "interpreter_visitor: this combo_tree contains a contin "
                "expression or greater_than_zero, so C++ would route to "
                "mixed_interpreter (table.h:329) -- mixed_interpreter is "
                "out of scope for this port cycle"
            )
        return boolean_interpreter(inputs)(self._it)
