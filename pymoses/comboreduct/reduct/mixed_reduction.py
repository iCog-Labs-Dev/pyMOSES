"""C++: moses/comboreduct/reduct/mixed_reduction.cc

S14 Task 4. `mixed_reduction()` assembles the mixed logical/continuous
rule set -- classic's own default `-Hcp`/predicate-tree reduction, used
directly (via `mixed_reduce`) wherever a tree can mix `greater_than_zero`
predicates with ordinary boolean structure. Takes no arguments in C++
(unlike `contin_reduction`) -- there is nothing analogous to `ignore_ops`
gating here. COMPOSITION ORDER IS THE NORMAL FORM (same governing fact as
`logical_reduction.py`/`contin_reduction.py`'s own module docstrings).

SELF-REFERENTIAL STATIC CONSTRUCTION, ported via a two-phase build --
C++'s `static iterative r_without_reduce_gt_zero_prod = iterative(
sequential(non_recursive, downwards(reduce_from_assumptions(
r_without_reduce_gt_zero_prod)), ...));` takes the ADDRESS of
`r_without_reduce_gt_zero_prod` INSIDE its own initializer -- legal in
C++ because a `static` variable's storage is fixed before its
initializer runs, and `reduce_from_assumptions(const rule& r)` only
STORES `&r` (no dereference at construction time; the address becomes
valid to actually CALL through only once the whole static initialization
completes, by which point `r_without_reduce_gt_zero_prod` is a real,
finished object). Python objects have no comparable "storage exists
before the constructor runs" property, so this is reproduced as: build
an EMPTY `iterative` shell first (`iterative()`'s own `r=None` no-wrapped-
rule state, meta_rules.py's own documented-as-real-if-unreachable-in-C++
default constructor), pass THAT SAME shell object to every
`reduce_from_assumptions`/`reduce_contin_if_not`/etc. constructed inside
its own body (each just stores the reference, matching C++'s non-owning
`const rule*`), THEN assign the fully-built `sequential(...)` onto the
shell's own `.r` attribute after the fact. Every wrapping rule
(`downwards`, `sequential`) that clones its wrapped rules via
`rule.clone()` (`copy.copy`, a SHALLOW copy) preserves this reference
correctly -- a shallow copy of `reduce_from_assumptions` still has its
`_reduction` attribute pointing at the SAME shell object, and mutating
the shell's `.r` after cloning is visible through every one of those
clones, exactly mirroring what taking `&r_without_reduce_gt_zero_prod`
achieves in C++.

No explicit `name=` argument is passed to ANY `sequential`/`iterative`
constructed here in the real C++ (checked directly -- unlike
`contin_reduction.cc`'s `"seq_without_factorize_distribute"` etc.), so
none is passed here either; every combinator keeps its class's own
default name.
"""

from __future__ import annotations

from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.comboreduct.reduct.branch_rules import reduce_cond_else
from pymoses.comboreduct.reduct.general_rules import eval_constants, level, reorder_commutative
from pymoses.comboreduct.reduct.logical_rules import reduce_and_assumptions
from pymoses.comboreduct.reduct.meta_rules import downwards, iterative, sequential, upwards
from pymoses.comboreduct.reduct.mixed_rules import (
    reduce_contin_if,
    reduce_contin_if_inner_op,
    reduce_contin_if_not,
    reduce_contin_if_substitute_cond,
    reduce_contin_if_to_impulse,
    reduce_from_assumptions,
    reduce_gt_division_of_constants,
    reduce_gt_zero_div,
    reduce_gt_zero_exp,
    reduce_gt_zero_impulse,
    reduce_gt_zero_log,
    reduce_gt_zero_minus_exp,
    reduce_gt_zero_pair_power,
    reduce_gt_zero_prod,
    reduce_gt_zero_sin,
    reduce_gt_zero_sum,
    reduce_gt_zero_sum_sin,
    reduce_impulse_power,
    reduce_impulse_prod,
    reduce_impulse_sum,
    reduce_inequality_from_assumptions,
    reduce_junction_gt_zero_sum_constant,
    reduce_op_contin_if,
)
from pymoses.comboreduct.reduct.reduct import _NO_IT, rule
from pymoses.util.tree import tree_node

_cached_rule: rule | None = None


def _reset_cache_for_testing() -> None:
    """No direct C++ symbol -- see `contin_reduction.py`'s own identical
    hook for the reasoning. `mixed_reduction` takes no arguments, so the
    caching quirk that makes `contin_reduction`'s hook load-bearing does
    not apply the same way here; this exists purely so this port's own
    test suite can force a fresh build (e.g. to prove the self-
    referential construction above does not error on a second build).
    """
    global _cached_rule
    _cached_rule = None


def mixed_reduction() -> rule:
    """C++: moses/comboreduct/reduct/mixed_reduction.cc:35 mixed_reduction"""
    global _cached_rule
    if _cached_rule is not None:
        return _cached_rule

    non_recursive = sequential(
        downwards(level()),
        upwards(eval_constants()),
        # The following rule is not valid if log has the semantics
        # log(abs(x)). ABS_LOG is commented out at vertex.h:49, so this
        # is always compiled into classic and always ported here.
        downwards(reduce_gt_zero_log()),
        downwards(reduce_gt_zero_exp()),
        downwards(reduce_gt_zero_minus_exp()),
        # downwards(reduce_gt_zero_times_const()), // deprecate
        # downwards(reduce_gt_zero_const_div()), // deprecate
        # downwards(reduce_gt_zero_prod_exp()), // deprecate
        # downwards(reduce_gt_zero_const_sum_sin()), // deprecate
        downwards(reduce_gt_zero_impulse()),
        downwards(reduce_impulse_sum()),
        downwards(reduce_contin_if_to_impulse()),
        # Complex mixed
        upwards(reorder_commutative()),
        downwards(reduce_gt_zero_pair_power()),
        downwards(reduce_impulse_power()),
        downwards(reduce_impulse_prod()),
        downwards(reduce_contin_if()),
        downwards(reduce_op_contin_if()),
        downwards(reduce_cond_else()),
        downwards(reduce_contin_if_inner_op()),
        downwards(reduce_contin_if_substitute_cond()),
        downwards(reduce_junction_gt_zero_sum_constant()),
    )

    # This set of rules is defined to avoid infinite recursion of the
    # rule reduce_gt_zero_prod -- see this module's own docstring for
    # the self-referential construction.
    r_without_reduce_gt_zero_prod = iterative()
    r_without_reduce_gt_zero_prod.r = sequential(
        non_recursive,
        # Very complex recursive mixed
        downwards(reduce_from_assumptions(r_without_reduce_gt_zero_prod)),
        downwards(reduce_inequality_from_assumptions()),
        downwards(reduce_contin_if_not(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sum(r_without_reduce_gt_zero_prod)),
        # Commented out to avoid infinite recursion:
        # downwards(reduce_gt_zero_prod(r)),
        downwards(reduce_gt_zero_div(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sum_sin(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sin(r_without_reduce_gt_zero_prod)),
        downwards(reduce_and_assumptions(r_without_reduce_gt_zero_prod)),
        # The following is commented because there are no boolean rules
        # here. The double negation cannot be reduced and it leads to
        # infinite recursion.
        # downwards(reduce_or_assumptions(r))
    )

    r = iterative()
    r.r = sequential(
        non_recursive,
        # Very complex mixed
        downwards(reduce_from_assumptions(r)),
        downwards(reduce_inequality_from_assumptions()),
        downwards(reduce_contin_if_not(r)),
        downwards(reduce_gt_zero_sum(r)),
        downwards(reduce_gt_zero_prod(r, r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_div(r)),
        downwards(reduce_gt_zero_sum_sin(r)),
        downwards(reduce_gt_zero_sin(r)),
        downwards(reduce_gt_division_of_constants(r)),
        downwards(reduce_and_assumptions(r)),
        # downwards(reduce_or_assumptions(r))
    )

    if _cached_rule is None:
        _cached_rule = r
    return _cached_rule


def mixed_reduce(tr: combo_tree, it: tree_node = _NO_IT) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:165 mixed_reduce

    Two C++ overloads collapsed into one Python signature (this port's
    usual convention): `mixed_reduce(tr, it)` (reduct.h:165-168) and
    `mixed_reduce(tr)` (reduct.h:170-173, `it` omitted -> the 1-arg
    `rule.__call__` overload). `it`'s default is `reduct.py`'s own
    `_NO_IT` sentinel.
    """
    mixed_reduction()(tr, it)
