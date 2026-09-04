"""C++: moses/moses/moses/complexity.h and complexity.cc

`tree_complexity` -- the algorithmic complexity of a combo program. See
the doc comment reproduced verbatim at `tree_complexity` itself below for
the governing design note ("For a Boolean formula, the complexity is the
neg(# of literals) ... Note Bene: this function returns a POSITIVE
number!", complexity.cc:33-55).

`complexity_t` (complexity.h:35, `typedef unsigned`) and `least_complexity`
(complexity.h:37) are re-exported from types.py as well (see its module
docstring) purely for convenience -- this module is where they are
actually defined, matching complexity.h being its own header rather than
folded into types.h in C++.
"""

from collections.abc import Callable

from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    get_contin,
    is_argument,
    is_contin,
)
from pymetta.util.tree import tree, tree_node

complexity_t = int  # C++: moses/moses/moses/complexity.h:35 complexity_t

least_complexity: complexity_t = 0  # C++: moses/moses/moses/complexity.h:37 least_complexity

# Builtins with complexity 1 that recurse into their children (as opposed
# to logical_and/logical_or/plus/times/greater_than_zero/impulse, which
# fall through to complexity 0 -- see tree_complexity's own body for why,
# quoting complexity.cc:92-99 verbatim).
_UNARY_OR_BINARY_COST_ONE = frozenset(
    {
        builtin.div,
        builtin.exp,
        builtin.log,
        builtin.sin,
        builtin.rand,
        builtin.equ,
        builtin.cond,
    }
)


def tree_complexity(
    tr_or_it: combo_tree | tree_node,
    stopper: Callable[[tree_node], bool] | None = None,
) -> complexity_t:
    """C++: moses/moses/moses/complexity.cc:56 tree_complexity
    and moses/moses/moses/complexity.cc:113 tree_complexity

    Merges both C++ overloads into one function, the same convention
    `boolean_interpreter.__call__` (interpreter.py) uses -- dispatching on
    whether `tr_or_it` is a whole `combo_tree` (the `const combo_tree&`
    overload, complexity.cc:113-120: empty-tree guard, then delegates to
    the iterator overload on `tr.begin()`) or a bare node (the
    `combo_tree::iterator` overload, complexity.cc:56-111, the actual
    recursive algorithm, reproduced below in the same branch order).

    `stopper`, when given, is a `Callable[[tree_node], bool]` predicate --
    standing in for C++'s raw function pointer
    (`bool (*)(const combo_tree::iterator&)`, complexity.h:41-42) since
    Python has no equivalent restriction to a free function; `None`
    matches the `NULL` default.

    "For a Boolean formula, the complexity is the neg(# of literals) ...
    Note Bene: this function returns a POSITIVE number!" (complexity.cc:
    33-55's own doc comment, reproduced verbatim -- the sign convention
    matters: `composite_score`'s own doc comment (types.py) says
    complexity must be positive, and `get_penalized_score()` SUBTRACTS
    `complexity * complexity_coef` from the raw score).

    NARROWED versus C++ (see vertex.py's module docstring for why): the
    `is_builtin_action`/`is_ann_type`/`is_action_result` checks
    (complexity.cc:83-85) are dropped -- those `vertex` alternatives
    (`builtin_action`, `ann_type`, `action_result_type`) are never
    constructed anywhere in this port, so `is_argument`/`is_contin` alone
    cover every case this port's narrowed `vertex` union can actually
    hold.
    """
    if isinstance(tr_or_it, tree):
        it = tr_or_it.begin()
        if it is None:  # tr.begin() == tr.end(): an empty tree
            return 0
        return tree_complexity(it, stopper)

    it = tr_or_it
    v = it.data

    # base cases -- complexity.cc:59-86

    # null_vertex marks the location of a logical knob; halt recursion
    # past logical knobs (complexity.cc:60-65).
    if v == builtin.logical_true or v == builtin.logical_false or v == builtin.null_vertex:
        return 0

    # Needed for knob-probing across boundaries between logical and
    # contin expressions (complexity.cc:67-70).
    if stopper is not None and stopper(it):
        return 0

    # *(0 stuff) marks the location of a contin knob; halt recursion past
    # contin knobs, also for knob-probing (complexity.cc:72-76).
    if (
        v == builtin.times
        and it.first_child is not None
        and is_contin(it.first_child.data)
        and get_contin(it.first_child.data) == 0
    ):
        return 0

    # Contins (and, in classic, builtin_action/ann_type/action_result --
    # see the docstring's NARROWED note) get a complexity of 1
    # (complexity.cc:78-86).
    if is_argument(v) or is_contin(v):
        return 1

    # recursive cases -- complexity.cc:88-110
    if v == builtin.logical_not:
        return tree_complexity(it.first_child, stopper)

    # "If an operator is not listed below, it has a complexity of zero.
    # Note that logical_and, logical_or are not listed, these have a
    # complexity of zero. div and trigonometric functions have complexity
    # one. But greater_than_zero, impulse, plus, times are all treated
    # with complexity zero." (complexity.cc:92-99's own comment.)
    c = 1 if v in _UNARY_OR_BINARY_COST_ONE else 0

    sib = it.first_child
    while sib is not None:
        c += tree_complexity(sib, stopper)
        sib = sib.next_sibling
    return c
