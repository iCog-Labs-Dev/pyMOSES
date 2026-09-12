"""C++: moses/comboreduct/interpreter/eval.{h,cc}

`eval` (this module) and `interpreter.py`'s already-ported
`boolean_interpreter` are TWO DIFFERENT C++ evaluators that happen to
overlap on booleans, not two names for the same thing -- do not fold one
into the other or route one through the other:

* `boolean_interpreter` (interpreter.h/.cc) only ever runs against a
  `vector<builtin>` -- pure boolean inputs -- and only recognises
  `argument`/`logical_and`/`logical_or`/`logical_not`/`logical_true`/
  `logical_false`. Every other vertex kind is a hard error there
  (interpreter.cc:95-101's own "OC_ASSERT(false)"-equivalent).
* `eval` (this module) takes a `vertex_seq` -- a MIXED sequence, one
  `vertex` of any kind per argument slot -- and additionally evaluates
  the full contin arithmetic (`plus/times/div/log/exp/sin/rand`), the
  list/fold operators (`list/car/cdr/cons/foldr/foldl`), `cond`/
  `contin_if`, `lambda`/`apply`, and (in classic) procedure calls. It is
  the evaluator `reduct/general_rules.cc`'s `eval_constants` rule calls
  to fold constant sub-expressions of ANY type during boolean reduction
  -- which is exactly why S7 needs it even though S7 itself only reduces
  boolean trees: a boolean expression can still contain a constant contin
  sub-expression (e.g. `0<(+(2 3))`) that reduct must be able to evaluate.

Where the two genuinely overlap (plain boolean and/or/not/true/false over
an argument binding), `eval`'s behaviour is deliberately IDENTICAL --
same empty-and()/or() sentinel-return "for correct foldr behaviour"
(eval.cc:136-137/144-145, interpreter.cc:66-68/75-77), same negate-on-
negative-argument-index rule. This is not code reuse in C++ (the two
files duplicate that logic independently) and is not code reuse here
either, for the same reason interpreter.py's own module docstring gives
for not sharing `boolean_interpreter`/`interpreter_visitor` internals:
each mirrors its own C++ source file.

`eval_binding` is the "never throws" entry point (catches
`OverflowException`/`EvalException` and returns the payload vertex
instead, eval.cc:290-299); `eval_throws_binding`/`eval_throws_tree`/
`eval_throws_vertex` are the throwing core it wraps. Per the task brief's
"Semantics to get right": an unbound argument must still raise even
through `eval_binding` -- `IndexError` is neither of the two exception
types `eval_binding` rescues, so it always propagates, matching "raise
now, not a wrong-value crash three functions later."

Every C++ iterator/combo_tree overload pair below (`eval_throws_binding`,
`eval_binding`, `eval_throws_tree` each have a `combo_tree::iterator` form
and a `const combo_tree&` form, eval.h:50-69) is collapsed into ONE Python
function taking `tr_or_it: combo_tree | tree_node` and dispatching on
`isinstance(tr_or_it, tree)` -- the same convention interpreter.py's
`boolean_interpreter.__call__` already established for exactly this
situation (see that module's docstring for the fuller justification, and
its own citation of task-4-brief.md's informally-worded interface line as
precedent for reading task-1-brief.md's `eval_throws_tree(tr)` shorthand
the same way).

Every value comparison against a specific `builtin` member below goes
through `vertex_equal` (vertex.py), not bare `==` -- the task brief's
named trap: `contin_t` is a bare `float` and `builtin` an `IntEnum`, so
`1.0 == builtin.logical_false` is `True` in bare Python even though
`boost::variant` can never agree across held alternatives. eval.cc is
FULL of exactly this shape (`eval_throws_binding(bmap, sib) ==
id::logical_false` inside the and()/or() loops, `i == id::logical_true`
in impulse, `vcond == id::logical_true` in cond, `*lp != id::list` in
car/cdr) -- every one of those compares a freshly-evaluated, TYPE-UNPROVEN
sub-result against a builtin literal, which is precisely the shape the
brief warns two prior files got silently wrong for.
"""

import math

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    combo_tree,
    get_contin,
    is_enum_type,
    negate_vertex,
    vertex,
    vertex_equal,
)
from pymoses.comboreduct.crutil.exception import EvalException, OverflowException
from pymoses.util.rng import randGen
from pymoses.util.tree import tree, tree_node


def is_procedure_call(v: vertex) -> bool:
    """C++: moses/comboreduct/combo/vertex.cc:43 is_procedure_call

    `boost::get<procedure_call>(&v)` -- always `False` in this port:
    `procedure_call` is not one of the four vertex alternatives this port
    constructs (vertex.py's module docstring narrows the 14-way
    `boost::variant` and explicitly excludes `procedure_call` as legacy
    machinery -- the same scope line as argument.py's `is_idx_valid`
    note). Ported as a real function (rather than inlined away or
    dropped) so `eval_procedure_tree` below, and the module docstring's
    account of why the `is_procedure_call` branch of `eval_throws_tree`
    (eval.cc:563) is never taken, both have something concrete to point
    at -- not a rhetorical "trust me."
    """
    return False


def eval_procedure_tree(bmap: list[vertex], it: tree_node) -> combo_tree:
    """C++: moses/comboreduct/interpreter/eval.cc:33 eval_procedure_tree

    Only the initial sanity check (eval.cc:36-40) is reachable in this
    port: `is_procedure_call(*it)` (immediately above) is always `False`
    for any vertex this port can construct, so C++'s own guard always
    fires here -- faithfully, not a divergence introduced by this port.
    The remainder of the C++ body (eval.cc:41-82: arity-checking against
    `procedure_call::arity()`/`abs_min_arity` from
    `type_checker/type_tree.h`, evaluating call arguments, then
    `eval_throws_tree`-ing the substituted procedure body) depends on the
    `procedure_call` class (`combo/procedure_call.h`) and the type
    checker, neither of which exists in this port -- both are out of
    scope for the whole S7 cycle (see argument.py's `is_idx_valid` note
    and CLAUDE.md's "Known sharp edges" on legacy procedure-call
    machinery) -- so it is unreachable and not ported.
    """
    if not is_procedure_call(it.data):
        raise ValueError(
            "ProcedureEvaluator - combo_tree node does not represent a "
            "combo procedure call."
        )
    raise NotImplementedError(
        "eval_procedure_tree: procedure_call machinery (combo/procedure_call.h, "
        "type_checker/type_tree.h's abs_min_arity) is out of scope for this port "
        "-- unreachable in practice since is_procedure_call is always False (see "
        "its own docstring)"
    )


def eval_throws_vertex(bmap: list[vertex], it: tree_node) -> vertex:
    """C++: moses/comboreduct/interpreter/eval.cc:113 eval_throws_vertex

    Evaluates operators that can only ever produce a single `vertex`
    (never a whole list/lambda combo_tree) -- `eval_throws_tree` below
    handles the list-shaped operators first and falls through to this
    function for everything else.
    """
    v = it.data
    match v:
        case argument():
            idx = v.idx
            # Assumption (eval.cc:121-123): when idx is negative, the
            # argument is necessarily boolean, and must be negated.
            # "Unbound arguments must raise": bmap[idx-1]/bmap[-idx-1] is
            # a plain Python list index -- IndexError on an out-of-range
            # idx, satisfying "must raise" with no bounds-check of this
            # port's own.
            # DIVERGENCE(C++ eval.cc:124): C++'s vertex_seq is
            # std::vector<vertex>, and operator[] on an out-of-range index
            # is undefined behaviour, not a throw -- Python's list[idx]
            # raising IndexError here is strictly safer than that UB, not
            # a behaviour C++ specifically guarantees.
            return bmap[idx - 1] if idx > 0 else negate_vertex(bmap[-idx - 1])

        case builtin():
            match v:
                case builtin.logical_false | builtin.logical_true:
                    return v

                case builtin.logical_and:
                    if it.first_child is None:  # for correct foldr behaviour
                        return builtin.logical_and
                    sib = it.first_child
                    while sib is not None:
                        if vertex_equal(eval_throws_binding(bmap, sib), builtin.logical_false):
                            return builtin.logical_false
                        sib = sib.next_sibling
                    return builtin.logical_true

                case builtin.logical_or:
                    if it.first_child is None:  # for correct foldr behaviour
                        return builtin.logical_or
                    sib = it.first_child
                    while sib is not None:
                        if vertex_equal(eval_throws_binding(bmap, sib), builtin.logical_true):
                            return builtin.logical_true
                        sib = sib.next_sibling
                    return builtin.logical_false

                case builtin.logical_not:
                    return negate_vertex(eval_throws_binding(bmap, it.first_child))

                case builtin.greater_than_zero:
                    sib = it.first_child
                    try:
                        # A divide-by-zero in the contin could throw. We
                        # want to return a boolean in this case, anyway.
                        x = eval_throws_binding(bmap, sib)
                    except OverflowException as e:
                        x = e.get_vertex()
                    return bool_to_vertex(0 < get_contin(x))

                case builtin.impulse:
                    i = eval_throws_binding(bmap, it.first_child)
                    return 1.0 if vertex_equal(i, builtin.logical_true) else 0.0

                case builtin.plus:
                    if it.first_child is None:  # for correct foldr behaviour
                        return v
                    res = 0.0
                    sib = it.first_child
                    while sib is not None:
                        res += get_contin(eval_throws_binding(bmap, sib))
                        sib = sib.next_sibling
                    return res

                case builtin.times:
                    if it.first_child is None:  # for correct foldr behaviour
                        return v
                    res = 1.0
                    sib = it.first_child
                    while sib is not None:
                        res *= get_contin(eval_throws_binding(bmap, sib))
                        if res == 0.0:  # avoid pointless evals
                            return res
                        sib = sib.next_sibling
                    return res

                case builtin.div:
                    sib = it.first_child
                    x = get_contin(eval_throws_binding(bmap, sib))
                    if x == 0.0:  # avoid pointless evals
                        return x
                    sib = sib.next_sibling
                    y = get_contin(eval_throws_binding(bmap, sib))
                    # DIVERGENCE(C++ eval.cc:217): `x / y` in C++ is
                    # IEEE754 float division -- a zero divisor produces a
                    # signed infinity, not an exception. Python's `float
                    # / float` raises ZeroDivisionError for a zero
                    # divisor instead, so the signed-infinity result is
                    # reconstructed by hand from the operands' signs
                    # (x is already known nonzero here).
                    try:
                        res = x / y
                    except ZeroDivisionError:
                        same_sign = math.copysign(1.0, x) == math.copysign(1.0, y)
                        res = math.inf if same_sign else -math.inf
                    if math.isnan(res) or math.isinf(res):
                        raise OverflowException(res)
                    return res

                case builtin.log:
                    x = get_contin(eval_throws_binding(bmap, it.first_child))
                    # `#define ABS_LOG` (vertex.h:49) is commented out, so
                    # classic's default build takes the plain `log(x)`
                    # branch (eval.cc:228), not `log(abs(x))` -- ported
                    # here as the only branch.
                    # DIVERGENCE(C++ eval.cc:228): `log(0)`/`log(negative)`
                    # are IEEE754 `-inf`/`nan` in C++; Python's `math.log`
                    # raises ValueError for both instead, so both are
                    # reconstructed by hand.
                    try:
                        res = math.log(x)
                    except ValueError:
                        res = -math.inf if x == 0.0 else math.nan
                    if math.isnan(res) or math.isinf(res):
                        raise OverflowException(res)
                    return res

                case builtin.exp:
                    x = get_contin(eval_throws_binding(bmap, it.first_child))
                    # DIVERGENCE(C++ eval.cc:237): C++'s exp() overflows
                    # to IEEE754 +inf for a too-large argument; Python's
                    # math.exp raises OverflowError instead (exp() is
                    # never nan for a finite real argument, matching
                    # C++'s own isinf-only check here).
                    try:
                        res = math.exp(x)
                    except OverflowError:
                        res = math.inf
                    if math.isinf(res):
                        raise OverflowException(res)
                    return res

                case builtin.sin:
                    x = get_contin(eval_throws_binding(bmap, it.first_child))
                    # DIVERGENCE(C++ eval.cc:246): C's sin(inf)/sin(nan)
                    # is nan, silently -- eval.cc itself never isnan/
                    # isinf-checks sin()'s result the way log/exp/div
                    # above do, so a nan here is a plain, valid returned
                    # vertex, not raised. Python's math.sin instead
                    # raises ValueError for a non-finite argument (task-1
                    # review, Minor: a bare ValueError here would have
                    # been indistinguishable from car/cdr's "not a
                    # list!" ValueError below -- returning nan instead
                    # removes the ambiguity by removing the exception).
                    try:
                        return math.sin(x)
                    except ValueError:
                        return math.nan

                case builtin.rand:
                    return randGen().randfloat()

                case builtin.equ:
                    raise AssertionError("The equ operator is not handled yet...")

                case _:
                    raise AssertionError(f"That case is not handled: {v.name}")

        case float():
            if math.isnan(v) or math.isinf(v):
                raise OverflowException(v)
            return v

        case str() if is_enum_type(v):
            # Enums are constants too (eval.cc:272-274).
            return v

        case _:
            # Don't know what to do with this. It might just be some
            # un-interpreted string (this port's stand-in for
            # definite_object/message, see vertex.py's module docstring),
            # such as those used in listUTest. If so, then the list
            # functions should catch this exception. Everyone else should
            # be surprised by this.
            raise EvalException(v, "unrecognized expression")


def eval_throws_binding(bmap: list[vertex], tr_or_it: combo_tree | tree_node) -> vertex:
    """C++: moses/comboreduct/interpreter/eval.cc:89 eval_throws_binding

    Also covers the `const combo_tree&` overload (eval.cc:285-288, which
    is just `eval_throws_binding(bmap, tr.begin())`) -- see module
    docstring on collapsing overload pairs.

    Evaluate a combo tree, using the argument values supplied in the
    `bmap` list. This procedure does not do any type-checking; the static
    type-checker should be used for that purpose.
    """
    it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
    ret = eval_throws_tree(bmap, it)
    root = ret.begin()
    if ret.number_of_children(root) != 0:
        raise AssertionError(
            "Invalid use of eval_throws_binding: expression evaluates to a "
            "whole combo_tree, not just one vertex"
        )
    return root.data


def eval_binding(bmap: list[vertex], tr_or_it: combo_tree | tree_node) -> vertex:
    """C++: moses/comboreduct/interpreter/eval.cc:290 eval_binding

    Also covers the `const combo_tree&` overload (eval.cc:301-304, which
    is just `eval_binding(bmap, tr.begin())`). As above, but
    `OverflowException`/`EvalException` are never thrown out of this
    entry point -- the offending vertex is returned instead. Any OTHER
    exception (e.g. `IndexError` from an unbound argument, or an
    `AssertionError` mirroring an `OC_ASSERT`) still propagates: this is
    not a catch-all.
    """
    it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
    try:
        return eval_throws_binding(bmap, it)
    except OverflowException as e:
        return e.get_vertex()
    except EvalException as e:
        return e.get_vertex()


def eval_throws_tree(bmap: list[vertex], tr_or_it: combo_tree | tree_node) -> combo_tree:
    """C++: moses/comboreduct/interpreter/eval.cc:306 eval_throws_tree

    Also covers the `const combo_tree&` overload (eval.cc:587-590, which
    is just `eval_throws_tree(bmap, tr.begin())`). As `eval_throws_binding`,
    but returns a combo_tree instead of a vertex -- the non-tree variant
    cannot be used when the result is a list, since the only way to
    represent a combo list is as a tree.

    First handles the operators that allow/require returning a whole
    combo_tree (a combo list or lambda expression); then falls through to
    `eval_throws_vertex` for the operators that can only return a single
    vertex.
    """
    it = tr_or_it.begin() if isinstance(tr_or_it, tree) else tr_or_it
    v = it.data

    if isinstance(v, builtin):
        match v:
            case builtin.list:
                result = combo_tree(builtin.list)
                loc = result.begin()
                sib = it.first_child
                while sib is not None:
                    rr = eval_throws_tree(bmap, sib)
                    result.append_child(loc, rr.begin())
                    sib = sib.next_sibling
                return result

            case builtin.car:
                lp = it.first_child
                if not vertex_equal(lp.data, builtin.list):
                    evo = eval_throws_tree(bmap, lp)
                    lp = evo.begin()
                if not vertex_equal(lp.data, builtin.list):
                    raise ValueError("not a list!")
                # If the list is empty, return an empty list (use an
                # empty list to represent nil).
                if lp.first_child is None:
                    return combo_tree(builtin.list)
                return eval_throws_tree(bmap, lp.first_child)

            case builtin.cdr:
                top = it.first_child
                if not vertex_equal(top.data, builtin.list):
                    evo = eval_throws_tree(bmap, top)
                    top = evo.begin()
                if not vertex_equal(top.data, builtin.list):
                    raise ValueError("not a list!")
                # Skip over the first elt. C++ (eval.cc:372-375) does this
                # unconditionally, even for an empty list -- ported
                # unguarded to match: an empty-list cdr fails loudly here
                # too (AttributeError on None.next_sibling) rather than
                # this port silently making it well-defined.
                sib = top.first_child.next_sibling
                result = combo_tree(builtin.list)
                loc = result.begin()
                while sib is not None:
                    rest = eval_throws_tree(bmap, sib)
                    result.append_child(loc, rest.begin())
                    sib = sib.next_sibling
                return result

            case builtin.cons:
                result = combo_tree(builtin.list)
                loc = result.begin()
                if it.first_child is None:
                    return combo_tree(builtin.cons)
                head = it.first_child
                ht = eval_throws_tree(bmap, head)
                result.append_child(loc, ht.begin())
                head = head.next_sibling
                rest = eval_throws_tree(bmap, head)
                sib = rest.begin().first_child
                while sib is not None:
                    result.append_child(loc, sib)
                    sib = sib.next_sibling
                return result

            case builtin.foldr:
                # base case: foldr(f v list) = v, i.e. list is empty list.
                tr_copy = combo_tree(it)
                itend = tr_copy.begin().first_child
                while itend.next_sibling is not None:
                    itend = itend.next_sibling
                if itend.first_child is None:
                    v_node = tr_copy.begin().first_child
                    while v_node.next_sibling is not itend:
                        v_node = v_node.next_sibling
                    return eval_throws_tree(bmap, v_node)

                # main case: foldr(f v l) = f(car l, foldr(f v cdr l))
                f = tr_copy.begin().first_child
                cb_tr = combo_tree(f)  # the call to f
                loc = cb_tr.begin()

                car_lst = combo_tree(builtin.car)
                car_lst_it = car_lst.begin()
                car_lst.append_child(car_lst_it, itend)
                car_lst = eval_throws_tree(bmap, car_lst_it)
                cb_tr.append_child(loc, car_lst.begin())  # cb_tr == f(<x> ...)

                cdr_call = combo_tree(builtin.cdr)
                cdr_call_it = cdr_call.begin()
                cdr_call.append_child(cdr_call_it, itend)
                tr_copy.erase(itend)
                tr_it = tr_copy.begin()

                cdr_result = eval_throws_tree(bmap, cdr_call_it)
                tr_copy.append_child(tr_it, cdr_result.begin())

                tr_copy = eval_throws_tree(bmap, tr_it)  # recurse on the cdr
                cb_tr.append_child(loc, tr_copy.begin())

                return eval_throws_tree(bmap, loc)

            case builtin.foldl:
                # base case: foldl(f v list) = v, i.e. list is empty list.
                tr_copy = combo_tree(it)
                itend = tr_copy.begin().first_child
                while itend.next_sibling is not None:
                    itend = itend.next_sibling
                if itend.first_child is None:
                    v_node = tr_copy.begin().first_child
                    while v_node.next_sibling is not itend:
                        v_node = v_node.next_sibling
                    return eval_throws_tree(bmap, v_node)

                # new tree: foldl(f, f(v car), cdr)
                f = tr_copy.begin().first_child
                v_node = f.next_sibling
                lst = eval_throws_tree(bmap, itend)
                lst_it = lst.begin()

                rec = combo_tree(f)  # f(v ...)
                rec_it = rec.begin()
                rec.append_child(rec_it, v_node)

                car_lst = combo_tree(builtin.car)
                car_lst_it = car_lst.begin()
                car_lst.append_child(car_lst_it, lst_it)
                car_lst = eval_throws_tree(bmap, car_lst_it)
                rec.append_child(rec_it, car_lst.begin())  # f(v car)

                cdr_lst = combo_tree(builtin.cdr)
                cdr_lst_it = cdr_lst.begin()
                cdr_lst.append_child(cdr_lst_it, lst_it)
                cdr_lst = eval_throws_tree(bmap, cdr_lst_it)

                new_tr = combo_tree(builtin.foldl)
                new_it = new_tr.begin()
                new_tr.append_child(new_it, f)
                new_tr.append_child(new_it, rec_it)
                new_tr.append_child(new_it, cdr_lst.begin())

                return eval_throws_tree(bmap, new_it)

            case builtin.lambda_:
                # lambda constructor -- not evaluated, just copied.
                return combo_tree(it)

            case builtin.apply:
                # The apply() operator is a sensible thing to have, but
                # this code is bad so it was disabled in C++ too
                # (eval.cc:511-512's own comment). Ported as the same
                # unconditional assertion failure, not the dead code
                # after it.
                raise AssertionError("apply() is not implemented")

            case builtin.contin_if | builtin.cond:
                # XXX TODO: contin_if should go away (C++ comment,
                # eval.cc:530, still true).
                sib = it.first_child
                while True:
                    if sib is None:
                        raise AssertionError("Error: mal-formed cond statement")
                    trcond = eval_throws_tree(bmap, sib)
                    sib = sib.next_sibling  # move past the condition

                    # The very last value is the universal "else" clause,
                    # taken when none of the previous predicates matched.
                    if sib is None:
                        return trcond

                    # If the condition is true, return the consequent
                    # (the value immediately following). Else skip the
                    # consequent and loop around again.
                    if vertex_equal(trcond.begin().data, builtin.logical_true):
                        return eval_throws_tree(bmap, sib)
                    sib = sib.next_sibling  # move past the consequent

    # `else if (is_procedure_call(v))` (eval.cc:563-568) is omitted here:
    # is_procedure_call is always False in this port (see its own
    # docstring), so that branch can never be taken.

    # Operators which only return a single vertex.
    try:
        return combo_tree(eval_throws_vertex(bmap, it))
    except EvalException as e:
        return combo_tree(e.get_vertex())


def arity(tr: combo_tree) -> int:
    """C++: moses/comboreduct/interpreter/eval.h:72 arity

    Return the arity of a tree: the largest absolute argument index that
    appears anywhere in it (0 if it contains no arguments at all).
    """
    a = 0
    for node in tr:
        if isinstance(node.data, argument):
            a = max(a, abs(node.data.idx))
    return a
