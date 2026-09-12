"""C++: moses/comboreduct/interpreter/interpreter.{h,cc}

`boolean_interpreter` (interpreter.h:46, interpreter.cc:37-102) -- evaluates
a combo tree of booleans (`argument`, `logical_and/or/not`,
`logical_true/false`) against a vector of `builtin` inputs. Ported S4
(S0-S6 foundation cycle).

`contin_interpreter` (interpreter.h:65, interpreter.cc:108-221) and
`mixed_interpreter` (interpreter.h:79, interpreter.cc:227-385) -- ported
S14 Task 1. Both were out of scope for the whole S0-S6 foundation cycle
(the module docstring said so, until now); see
docs/superpowers/plans/2026-09-08-pymoses-s14-contin.md's Task 1.

`interpreter_visitor` lives elsewhere in classic (table.h:304-351, not this
directory) but is ported alongside these three here because its whole job
is choosing which sub-interpreter to run. All five `operator()` overloads
now have a real branch (table.h:328-350): `vector<builtin>` dispatches to
either `boolean_interpreter` or `mixed_interpreter` depending on `mixed`;
`vector<contin_t>` and `vector<vertex>` ALWAYS dispatch to
`mixed_interpreter`, regardless of `mixed` (table.h:335-336's own comment:
"Can't use contin, since the output might be non-contin, e.g. a boolean,
or an enum" -- true for `vector<vertex>` too, unconditionally, per its own
one-line `operator()` body); `string_seq` and `vector<combo_tree>` still
raise (table.h:344-350's own `OC_ASSERT(false, "Not implemented")` --
these two are genuinely unimplemented in classic itself, not merely
out-of-scope for this port).

Both `contin_interpreter` and `mixed_interpreter` are `virtual`-dispatch
classes in C++ (`contin_eval`/`boolean_eval` are declared `virtual` in
`boolean_interpreter`/`contin_interpreter` and overridden in
`mixed_interpreter`). This matters for how `mixed_interpreter::mixed_eval`
delegates to its two bases: `boolean_interpreter::boolean_eval(it)` /
`contin_interpreter::contin_eval(it)` at interpreter.cc:303/313 are
EXPLICITLY QUALIFIED calls -- they run the BASE class's own method body
directly, bypassing the override for that one entry -- but every
RECURSIVE call inside that base method's own body (`boolean_eval(sib)`,
`contin_eval(sib)`) is unqualified, so C++ virtual dispatch sends it back
to `mixed_interpreter`'s override, letting a `plus` node's arguments
resolve through `_mixed_inputs`/`_use_contin_inputs` etc. rather than the
base's own (empty, in mixed mode) `contin_inputs`. Ported by calling the
unbound base method explicitly (`contin_interpreter.contin_eval(self,
it)`) at exactly those two call sites, and relying on Python's own
always-dynamic method dispatch (`self.contin_eval(...)`) everywhere else
to reproduce the "unqualified call re-enters the most-derived override"
behaviour for free. Getting this backwards is not a subtle behavioural
difference: calling `self.contin_eval(it)` instead of the unbound base
call at those two sites recurses `mixed_eval` back into itself
indefinitely (`mixed_eval` -> `contin_eval` override -> `mixed_eval` ->
...) for the exact `+($1 $2)`-shaped trees `test_mixed_interpreter`
(this module's own oracle, ported from interpreterUTest.cxxtest:83-111)
exercises -- a `RecursionError`, not a wrong number.

All three classes dispatch on vertex kind with `match` (mirroring
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

import math

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    combo_tree,
    contin_t,
    get_builtin,
    get_contin,
    is_contin,
    is_contin_expr,
    is_definite_object,
    negate_vertex,
    vertex,
    vertex_equal,
)
from pymoses.comboreduct.crutil.exception import OverflowException
from pymoses.util.rng import randGen
from pymoses.util.tree import tree, tree_node


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
        (pymoses/util/tree.py's module docstring -- iterators collapse to
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
                # DIVERGENCE(C++ moses/comboreduct/interpreter/interpreter.cc:54):
                # DO NOT "FIX" BY MATCHING CLASSIC. C++ indexes a
                # `std::vector` with a bare `operator[]` and no bounds
                # check, so an argument beyond the table's arity is
                # UNDEFINED BEHAVIOUR there -- it reads past the end.
                #
                # Measured on a 2-input boolean table with
                # `-e 'and($1 $99)'`: classic prints `0 and($1 $99)` and
                # exits 0 -- a PERFECT score for a tree it cannot evaluate
                # -- and the same command with `-W1` segfaults (rc 139)
                # instead, deterministically, five runs each way. A port
                # cannot faithfully reproduce undefined behaviour, and
                # reproducing the "perfect score" would be worse than
                # failing: it would silently rank an ill-formed exemplar
                # first.
                #
                # Python's list indexing raises `IndexError` here already;
                # this makes the failure say what actually happened, since
                # a bare `IndexError: list index out of range` from inside
                # a recursive evaluator gives the user nothing to act on.
                n = len(self.boolean_inputs)
                if not -n <= idx <= n or idx == 0:
                    raise IndexError(
                        f"boolean_eval: argument ${idx} is out of range for a "
                        f"{n}-input row. C++ has no bounds check here "
                        "(interpreter.cc:54) and reads out of bounds instead; "
                        "see this line's DIVERGENCE note."
                    )
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
                        raise ValueError(f"boolean_interpreter does not handle builtin {v.name!r}")

            case _:
                # C++: interpreter.cc:95-101 -- the vertex is not even a
                # builtin (e.g. a bare contin_t leaf, or an argument
                # already handled above).
                raise ValueError(f"boolean_interpreter does not handle vertex {v!r}")


class contin_interpreter:
    """C++: moses/comboreduct/interpreter/interpreter.h:65 contin_interpreter

    Evaluates a combo tree of continuous arithmetic (`argument`,
    `plus/times/div/log/exp/sin/rand`, a bare `contin_t` leaf) against a
    vector of `contin_t` inputs. `contin_eval` is declared `virtual` in
    C++ (interpreter.h:73) -- see this module's own docstring for why
    that matters to `mixed_interpreter` below, and why every recursive
    call here is a plain `self.contin_eval(...)` (never a direct/bound
    call), matching C++'s own unqualified-call-is-virtual behaviour.
    """

    def __init__(self, inputs: list[contin_t] | None = None) -> None:
        """C++: moses/comboreduct/interpreter/interpreter.cc:108 contin_interpreter

        Mirrors `boolean_interpreter.__init__` exactly: `inputs` defaults
        to an empty list (C++ default `std::vector<contin_t>()`,
        interpreter.h:68).
        """
        self.contin_inputs: list[contin_t] = [] if inputs is None else inputs

    def __call__(self, tr_or_it: combo_tree | tree_node) -> contin_t:
        """C++: moses/comboreduct/interpreter/interpreter.cc:111 contin_interpreter::operator()

        Covers both C++ overloads (interpreter.cc:111-116), same
        `tr_or_it` convention as `boolean_interpreter.__call__` -- see
        that method's own docstring and this module's docstring.
        """
        it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
        return self.contin_eval(it)

    def contin_eval(self, it: tree_node) -> contin_t:
        """C++: moses/comboreduct/interpreter/interpreter.cc:118 contin_interpreter::contin_eval

        Same outer type-first `match` discipline as
        `boolean_interpreter.boolean_eval` (see that method's own FIX WAVE
        comment): the outer `match` dispatches on TYPE ONLY
        (`argument()` / `builtin()` / the numeric/catch-all cases),
        confirming `isinstance(v, builtin)` before the INNER `match`
        compares `v` against specific builtin members by value -- the
        same `contin_t == builtin` collision risk applies here (e.g. a
        contin leaf `6.0` colliding with `builtin.plus`'s ordinal 6).
        """
        v = it.data

        match v:
            case argument():
                # C++: interpreter.cc:123-125 -- `return
                # contin_inputs[a->idx - 1];`, with NO `idx > 0` branch
                # unlike `boolean_interpreter.boolean_eval`'s argument
                # case: a contin argument is never a negated-boolean
                # reference, so C++ never checks the sign here at all.
                #
                # DO NOT FIX: for a negative `idx` this is C++ UB
                # (`arity_t` is `int`; `std::vector::operator[]` takes a
                # `size_type`, so a negative `idx - 1` implicitly
                # converts to a huge `size_t` -- an out-of-bounds access
                # with no defined result, not a controlled negation).
                # This port mirrors the literal expression
                # `contin_inputs[idx - 1]`, which for a negative idx
                # performs Python's own negative-index semantics instead
                # of touching invalid memory -- a DIFFERENT (safer, more
                # deterministic) outcome than C++'s UB, not a match to
                # it, because UB has no fixed outcome to match. Not
                # reachable in practice: negative argument indices encode
                # boolean negation (argument.py's module docstring) and
                # never appear as a contin operator's own argument in any
                # exemplar this port's callers build.
                return self.contin_inputs[v.idx - 1]

            case builtin():
                match v:
                    case builtin.plus:
                        # C++: interpreter.cc:133-140 -- `return *b;` where
                        # `*b` is a `const builtin&` and the function's
                        # return type is `contin_t` (double): this is NOT
                        # "return the operator token as a sentinel" the
                        # way boolean_interpreter's and()/or() childless
                        # case is (there, the return type IS `builtin`, so
                        # no conversion happens) -- `contin_eval` returns
                        # `contin_t`, so `*b` implicitly converts
                        # builtin(enum) -> int -> double, i.e. `id::plus`'s
                        # ordinal (6) becomes the contin_t VALUE `6.0`.
                        # Measured directly (tools/cpp_probe/
                        # contin_interpreter_probe.cc,
                        # "plus_over_no_children=VALUE:6"): a nested
                        # `plus($1 plus())` really does add 6.0 into the
                        # sum, not 0.0 -- this is a real, if likely
                        # unintended, upstream quirk, ported verbatim, not
                        # "fixed" into a true additive identity.
                        # `contin_t(v)` reproduces the same conversion
                        # Python's own `IntEnum` already performs under
                        # bare arithmetic (`0.0 + builtin.plus == 6.0` --
                        # verified) -- made explicit here so the RETURN
                        # TYPE also matches (a bare top-level childless
                        # `+()` must come back as a genuine `contin_t`,
                        # not a `builtin` enum member, for any caller that
                        # inspects the result's type, e.g. `is_contin`).
                        if it.first_child is None:
                            return contin_t(v)
                        res = 0.0
                        sib = it.first_child
                        while sib is not None:
                            res += self.contin_eval(sib)
                            sib = sib.next_sibling
                        return res

                    case builtin.times:
                        # C++: interpreter.cc:143-155 -- same implicit
                        # builtin->contin_t conversion as `plus` above
                        # (measured: "times_over_no_children=VALUE:7",
                        # `id::times`'s ordinal).
                        if it.first_child is None:
                            return contin_t(v)
                        res = 1.0
                        sib = it.first_child
                        while sib is not None:
                            res *= self.contin_eval(sib)
                            if res == 0.0:  # avoid pointless evals
                                return res
                            sib = sib.next_sibling
                        return res

                    case builtin.div:
                        # C++: interpreter.cc:157-168
                        sib = it.first_child
                        x = self.contin_eval(sib)
                        if x == 0.0:  # avoid pointless evals
                            return x
                        sib = sib.next_sibling
                        y = self.contin_eval(sib)
                        # DIVERGENCE(C++ interpreter.cc:164): `x / y` in
                        # C++ is IEEE754 float division -- a zero divisor
                        # produces a signed infinity, not an exception.
                        # Python's `float / float` raises
                        # ZeroDivisionError for a zero divisor instead, so
                        # the signed-infinity result is reconstructed by
                        # hand from the operands' signs (`x` is already
                        # known nonzero here). Same reconstruction
                        # `eval.py`'s `eval_throws_binding` uses for its
                        # own (separate) `div` case -- eval.cc:217 -- kept
                        # as an independent copy here per this module's
                        # own "each mirrors its own C++ source file"
                        # convention (see eval.py's module docstring).
                        try:
                            res = x / y
                        except ZeroDivisionError:
                            same_sign = math.copysign(1.0, x) == math.copysign(1.0, y)
                            res = math.inf if same_sign else -math.inf
                        if math.isnan(res) or math.isinf(res):
                            raise OverflowException(res)
                        return res

                    case builtin.log:
                        # C++: interpreter.cc:170-180. `#ifdef ABS_LOG` is
                        # NOT defined by default (vertex.h:49, `#define
                        # ABS_LOG` is commented out), so classic's default
                        # build takes the plain `log(x)` branch, not
                        # `log(abs(x))` -- ported here as the only branch,
                        # matching eval.py's own log case (eval.cc:228)
                        # exactly.
                        x = self.contin_eval(it.first_child)
                        # DIVERGENCE(C++ interpreter.cc:175): `log(0)`/
                        # `log(negative)` are IEEE754 `-inf`/`nan` in
                        # C++; Python's `math.log` raises `ValueError`
                        # for both instead, so both are reconstructed by
                        # hand -- same shape as eval.py's `log` case.
                        try:
                            res = math.log(x)
                        except ValueError:
                            res = -math.inf if x == 0.0 else math.nan
                        if math.isnan(res) or math.isinf(res):
                            raise OverflowException(res)
                        return res

                    case builtin.exp:
                        # C++: interpreter.cc:182-188 -- only `isinf`
                        # is checked here (exp() of a finite real
                        # argument is never nan), matching eval.py's
                        # `exp` case (eval.cc:237) exactly.
                        x = self.contin_eval(it.first_child)
                        # DIVERGENCE(C++ interpreter.cc:183): C's exp()
                        # overflows to IEEE754 +inf for a too-large
                        # argument; Python's `math.exp` raises
                        # `OverflowError` instead.
                        try:
                            res = math.exp(x)
                        except OverflowError:
                            res = math.inf
                        if math.isinf(res):
                            raise OverflowException(res)
                        return res

                    case builtin.sin:
                        # C++: interpreter.cc:190-192 -- NO isnan/isinf
                        # check at all: `sin(inf)`/`sin(nan)` is a
                        # silently valid `nan` result in C, not raised.
                        # DIVERGENCE(C++ interpreter.cc:191): Python's
                        # `math.sin` raises `ValueError` for a
                        # non-finite argument instead of returning `nan`;
                        # reconstructed by hand, matching eval.py's `sin`
                        # case (eval.cc:246) exactly.
                        x = self.contin_eval(it.first_child)
                        try:
                            return math.sin(x)
                        except ValueError:
                            return math.nan

                    case builtin.rand:
                        # C++: interpreter.cc:194-195. `randfloat()` returns
                        # a C `float`, and C++ then widens it into the
                        # `contin_t` (double) this function returns -- so
                        # the value carries exactly 24 significand bits of
                        # information no matter which language reads it.
                        # See `MT19937RandGen.randfloat`'s own docstring.
                        return randGen().randfloat()

                    case _:
                        # C++: interpreter.cc:197-204 -- classic throws a
                        # genuine `ComboException` here (a checked
                        # exception), not an `OC_ASSERT` the way
                        # `boolean_interpreter`'s default case does,
                        # matching the same convention (this project's
                        # exception-type mapping is inconsistent across
                        # OC_ASSERT/ComboException already --
                        # docs/CARRY-FORWARD.md Sec 5) -- ported as
                        # `ValueError` for consistency with this file's
                        # own `boolean_interpreter.boolean_eval`.
                        raise ValueError(f"contin_interpreter does not handle builtin {v.name!r}")

            case _ if is_contin(v):
                # C++: interpreter.cc:208-212 -- a bare contin_t leaf.
                if math.isnan(v) or math.isinf(v):
                    raise OverflowException(v)
                return v

            case _:
                # C++: interpreter.cc:213-219 -- neither argument, builtin
                # nor contin_t; same ComboException-to-ValueError mapping
                # as the builtin default case above.
                raise ValueError(f"contin_interpreter does not handle vertex {v!r}")


class mixed_interpreter(boolean_interpreter, contin_interpreter):
    """C++: moses/comboreduct/interpreter/interpreter.h:79 mixed_interpreter

    `struct mixed_interpreter : public boolean_interpreter, public
    contin_interpreter` -- ported as real multiple inheritance, matching
    C++'s own class hierarchy (not merely composition), because
    `mixed_eval`'s boolean/contin operator branches genuinely reuse the
    two base classes' own method BODIES via explicit-qualified calls --
    see this module's own docstring for exactly how and why.

    C++ has THREE constructors (interpreter.cc:227-245), chosen by the
    STATIC type of the argument at the call site -- something Python's
    single `list` type cannot reproduce by runtime inspection alone (an
    empty list, or a list holding only `builtin`/only `contin_t` values,
    is genuinely ambiguous between more than one of the three C++
    overloads). Rather than guess, this port exposes the three
    constructors as: the default `__init__` (mirrors `mixed_interpreter
    (const std::vector<vertex>& inputs=empty_vertex_seq)`, interpreter.cc:
    227-231 -- the only one of the three with a default argument, and the
    one `interpreterUTest.cxxtest:105`'s own `mixed_interpreter(ir.first)
    (tr)` call uses, where `ir.first` is a genuine `vector<vertex>`) plus
    two named alternate constructors, `from_contin_inputs`/
    `from_boolean_inputs`, mirroring the other two (interpreter.cc:
    233-238, :240-245) unambiguously. `interpreter_visitor.__call__`
    below is the one caller that must pick among the three, and it does
    so explicitly rather than by guessing from list contents wherever the
    choice is unambiguous from ITS OWN overload (see that method's own
    docstring for the one place a real ambiguity remains, inherited from
    `multi_type_seq` itself).
    """

    def __init__(self, inputs: list[vertex] | None = None) -> None:
        """C++: moses/comboreduct/interpreter/interpreter.cc:227 mixed_interpreter

        `mixed_interpreter(const std::vector<vertex>& inputs=empty_vertex_seq)`
        -- both bases default-construct with empty inputs (C++ member
        init list does not mention them, so `boolean_interpreter()`/
        `contin_interpreter()`'s own default arguments apply).
        """
        contin_interpreter.__init__(self)
        boolean_interpreter.__init__(self)
        self._use_boolean_inputs = False
        self._use_contin_inputs = False
        self._mixed_inputs: list[vertex] = [] if inputs is None else inputs

    @classmethod
    def from_contin_inputs(cls, inputs: list[contin_t]) -> mixed_interpreter:
        """C++: moses/comboreduct/interpreter/interpreter.cc:233 mixed_interpreter

        `mixed_interpreter(const std::vector<contin_t>& inputs)` --
        initializes the `contin_interpreter` base with `inputs`
        (interpreter.cc:234), leaves `boolean_interpreter` at its own
        default (empty), and empties `_mixed_inputs`.
        """
        obj = cls.__new__(cls)
        contin_interpreter.__init__(obj, inputs)
        boolean_interpreter.__init__(obj)
        obj._use_boolean_inputs = False
        obj._use_contin_inputs = True
        obj._mixed_inputs = []
        return obj

    @classmethod
    def from_boolean_inputs(cls, inputs: list[builtin]) -> mixed_interpreter:
        """C++: moses/comboreduct/interpreter/interpreter.cc:240 mixed_interpreter

        `mixed_interpreter(const std::vector<builtin>& inputs)` --
        initializes the `boolean_interpreter` base with `inputs`
        (interpreter.cc:241), leaves `contin_interpreter` at its own
        default (empty), and empties `_mixed_inputs`.
        """
        obj = cls.__new__(cls)
        contin_interpreter.__init__(obj)
        boolean_interpreter.__init__(obj, inputs)
        obj._use_boolean_inputs = True
        obj._use_contin_inputs = False
        obj._mixed_inputs = []
        return obj

    def __call__(self, tr_or_it: combo_tree | tree_node) -> vertex:
        """C++: moses/comboreduct/interpreter/interpreter.cc:247 mixed_interpreter::operator()

        Covers both C++ overloads (interpreter.cc:247-254), same
        `tr_or_it` convention as `boolean_interpreter.__call__`.
        """
        it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
        return self.mixed_eval(it)

    def boolean_eval(self, it: tree_node) -> builtin:
        """C++: moses/comboreduct/interpreter/interpreter.cc:256 mixed_interpreter::boolean_eval

        Overrides `boolean_interpreter.boolean_eval` -- `return
        get_builtin(mixed_eval(it));`. This is what makes the explicit
        `boolean_interpreter.boolean_eval(self, it)` call in `mixed_eval`'s
        own logical-operator branch (below) re-enter `mixed_eval` for
        every recursive child, via ordinary Python/C++ virtual dispatch --
        see this module's own docstring.
        """
        return get_builtin(self.mixed_eval(it))

    def contin_eval(self, it: tree_node) -> contin_t:
        """C++: moses/comboreduct/interpreter/interpreter.cc:261 mixed_interpreter::contin_eval

        Overrides `contin_interpreter.contin_eval` -- `return
        get_contin(mixed_eval(it));`. Symmetric to `boolean_eval` above.
        """
        return get_contin(self.mixed_eval(it))

    def mixed_eval(self, it: tree_node) -> vertex:
        """C++: moses/comboreduct/interpreter/interpreter.cc:266 mixed_interpreter::mixed_eval

        Same outer type-first `match` discipline as `boolean_eval`/
        `contin_eval` above.
        """
        v = it.data

        match v:
            case argument():
                # C++: interpreter.cc:271-292
                idx = v.idx
                # The mixed interpreter could be getting an array of all
                # contins, if the signature of the problem is
                # ->(contin ... contin boolean) or
                # ->(contin ... contin enum_t) -- deal with this case.
                if self._use_contin_inputs:
                    return self.contin_inputs[idx - 1]
                # ... or an array of all booleans, if the signature is
                # ->(bool ... bool contin) or ->(bool ... bool enum).
                if self._use_boolean_inputs:
                    return (
                        self.boolean_inputs[idx - 1]
                        if idx > 0
                        else negate_vertex(self.boolean_inputs[-idx - 1])
                    )
                # A negative index means boolean-negate.
                return (
                    self._mixed_inputs[idx - 1]
                    if idx > 0
                    else negate_vertex(self._mixed_inputs[-idx - 1])
                )

            case builtin():
                match v:
                    case (
                        builtin.logical_false
                        | builtin.logical_true
                        | builtin.logical_and
                        | builtin.logical_or
                        | builtin.logical_not
                    ):
                        # C++: interpreter.cc:298-303 -- explicit-qualified
                        # base call, see this module's own docstring.
                        return boolean_interpreter.boolean_eval(self, it)

                    case (
                        builtin.plus
                        | builtin.times
                        | builtin.div
                        | builtin.log
                        | builtin.exp
                        | builtin.sin
                        | builtin.rand
                    ):
                        # C++: interpreter.cc:306-313 -- explicit-qualified
                        # base call, see this module's own docstring.
                        return contin_interpreter.contin_eval(self, it)

                    case builtin.greater_than_zero:
                        # C++: interpreter.cc:316-329
                        sib = it.first_child
                        try:
                            # A divide-by-zero in the contin could throw.
                            # We want to return a boolean in this case,
                            # anyway. Values might be +inf -inf or nan and
                            # we can still get a sign bit off two of these
                            # cases...
                            x = self.mixed_eval(sib)
                        except OverflowException as e:
                            x = e.get_vertex()
                        return bool_to_vertex(0 < get_contin(x))

                    case builtin.impulse:
                        # C++: interpreter.cc:331-334
                        i = self.mixed_eval(it.first_child)
                        return 1.0 if vertex_equal(i, builtin.logical_true) else 0.0

                    case builtin.contin_if | builtin.cond:
                        # C++: interpreter.cc:337-359. XXX TODO:
                        # contin_if should go away (C++ comment, still
                        # true).
                        sib = it.first_child
                        while True:
                            if sib is None:
                                raise AssertionError("Error: mal-formed cond statement")
                            vcond = self.mixed_eval(sib)
                            sib = sib.next_sibling  # move past the condition
                            # The very last value is the universal "else"
                            # clause, taken when none of the previous
                            # predicates were true.
                            if sib is None:
                                return vcond
                            # If condition is true, return the consequent
                            # (the value immediately following). Else,
                            # skip the consequent, and loop around again.
                            if vertex_equal(vcond, builtin.logical_true):
                                return self.mixed_eval(sib)
                            sib = sib.next_sibling  # move past the consequent

                    case _:
                        # C++: interpreter.cc:361-367
                        raise ValueError(f"mixed_interpreter does not handle builtin {v.name!r}")

            case float():
                # C++: interpreter.cc:371-373 -- `else if (is_contin(v))`.
                # `contin_t` IS `float` in this port (common_def.py), so
                # `case float():` covers exactly `is_contin(v)` -- see
                # `_vertex_to_str` (iostream_combo.py) for the same
                # convention.
                return contin_interpreter.contin_eval(self, it)

            case str() if is_definite_object(v):
                # C++: interpreter.cc:374-377 -- "string constant".
                return v

            case _:
                # C++: interpreter.cc:378-384
                raise ValueError(f"mixed_interpreter does not handle vertex {v!r}")


class interpreter_visitor:
    """C++: moses/comboreduct/table/table.h:304 interpreter_visitor

    Chooses the interpreter type based on the type of vertices appearing
    in the combo tree, exactly as classic does. C++ is a real
    `boost::static_visitor<vertex>` applied (via `boost::apply_visitor`)
    to a `multi_type_seq` -- a `boost::variant` of `vector<builtin>`,
    `vector<contin_t>`, `vector<vertex>`, `string_seq` and
    `vector<combo_tree>`, whose STATIC alternative is chosen once, when
    the caller constructs the `multi_type_seq`. Python has no equivalent
    static-variant machinery, so `__call__` below inspects `inputs`'s
    runtime content to decide which of the five C++ `operator()`
    overloads it stands in for -- see its own docstring for the one
    genuine ambiguity this introduces (an input list that is entirely one
    element type looks the same whether the caller "meant"
    `vector<builtin>`/`vector<contin_t>` or a `vector<vertex>` that
    merely happens to be homogeneous).

    `string_seq` and `vector<combo_tree>` (table.h:344-350) are the two
    overloads classic ITSELF leaves as `OC_ASSERT(false, "Not
    implemented")` -- ported the same way, as `NotImplementedError`, not
    as an out-of-scope marker.
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

    def __call__(self, inputs: list[builtin] | list[contin_t] | list[vertex]) -> vertex:
        """C++: moses/comboreduct/table/table.h:328 interpreter_visitor::operator()

        Dispatches among the five `operator()` overloads (table.h:
        328-350) by inspecting `inputs`'s runtime content -- see class
        docstring for why a real ambiguity exists:

        * All elements `builtin`: the `vector<builtin>` overload
          (table.h:328-331) -- `mixed_interpreter` if `self.mixed`, else
          plain `boolean_interpreter`.
        * All elements `contin_t` (and not all `builtin`, so an empty
          list is NOT misrouted here -- see below): the
          `vector<contin_t>` overload (table.h:332-336), which ALWAYS
          uses `mixed_interpreter` regardless of `self.mixed` (that
          function's own comment: "Can't use contin, since the output
          might be non-contin, e.g. a boolean, or an enum").
        * `inputs` is empty: matches BOTH of the checks above vacuously.
          Resolved the same way C++'s own default-argument overload
          resolution would for a bare `mixed_interpreter()`/
          `boolean_interpreter()` call -- but since a real caller
          passing an explicitly empty sequence could equally mean
          "no boolean inputs" or "no contin inputs", and this port's
          actual callers (table.py's future row-evaluation path) always
          pass a genuinely populated per-row sequence, this case falls
          through to the `vector<vertex>` branch below rather than
          guessing further.
        * Anything else (a genuine mix of types, i.e. `vector<vertex>`,
          or empty): the `vector<vertex>` overload (table.h:337-339),
          which -- like `vector<contin_t>` -- ALWAYS uses
          `mixed_interpreter`.
        * A `string_seq` (`list[str]`, no `enum_t`/contin/builtin) or a
          `list` of `combo_tree` objects: table.h:341-343/:344-350,
          genuinely unimplemented in classic itself -- raises
          `NotImplementedError`, matching `OC_ASSERT(false, "Not
          implemented")`.
        """
        if not isinstance(inputs, list):
            raise TypeError(f"interpreter_visitor expects a list of inputs, got {type(inputs)!r}")
        if not inputs:
            return mixed_interpreter([])(self._it)
        if all(isinstance(x, combo_tree) for x in inputs):
            raise NotImplementedError(
                "interpreter_visitor: the vector<combo_tree> overload "
                '(table.h:344-350) is OC_ASSERT(false, "Not implemented") '
                "in classic itself, not merely out of scope for this port"
            )
        if all(isinstance(x, str) and not isinstance(x, builtin) for x in inputs):
            # `is_contin`/`is_builtin` never match a bare `str` (nor
            # `enum_t`, its subclass -- vertex.py's module docstring), so
            # this check alone separates a genuine string_seq from
            # anything the branches below would otherwise claim.
            raise NotImplementedError(
                "interpreter_visitor: the string_seq overload (table.h:341-343) "
                'is OC_ASSERT(false, "Not implemented") in classic itself, not '
                "merely out of scope for this port"
            )
        if all(isinstance(x, builtin) for x in inputs):
            if self.mixed:
                return mixed_interpreter.from_boolean_inputs(inputs)(self._it)
            return boolean_interpreter(inputs)(self._it)
        if all(is_contin(x) for x in inputs):
            return mixed_interpreter.from_contin_inputs(inputs)(self._it)
        return mixed_interpreter(inputs)(self._it)
