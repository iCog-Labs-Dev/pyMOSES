"""C++: moses/comboreduct/reduct/full_reduction.cc

S14 Task 4 -- the last of the nine files. `full_reduction()` assembles the
"predicate tree" rule set: logical structure, continuous arithmetic, and
mixed greater_than_zero/impulse/contin_if rules all in ONE composition
(reduct.h's own doc comment on `full_reduce`: "trees containing mixtures
of logical operators, boolean constants, boolean-valued predicates,
boolean-valued literals, arithmetic operators, contin-typed constants,
and contin-valued literals"). COMPOSITION ORDER IS THE NORMAL FORM, same
governing fact as every other reduction-assembly file in this slice.

Self-referential static construction -- same two-phase build as
`mixed_reduction.py` (an empty `iterative()` shell, wired into every rule
that needs a `const rule&` reference, then the real `sequential` assigned
onto the shell's own `.r` afterward); see that module's own docstring for
the full reasoning.

THREE RULES `mixed_reduction.cc` WIRES IN THAT `full_reduction.cc` DOES
NOT, verified by reading both files side by side, not assumed from one
alone:

* `reduce_gt_zero_log` -- commented out here (`// downwards(reduce_gt_
  zero_log()),`, full_reduction.cc:102/194) in BOTH `r_without_reduce_
  gt_zero_prod` and `r`; `mixed_reduction.cc`'s own `non_recursive`
  includes it uncommented. `full_reduce` therefore never folds
  `0<log(x)` into `0<-1+x` the way `mixed_reduce` does.
* `reduce_junction_gt_zero_sum_constant` -- commented out here
  (`//downwards(reduce_junction_gt_zero_sum_constant()),`,
  full_reduction.cc:125/217) in both blocks; wired into `mixed_
  reduction.cc`'s `non_recursive`.
* `reduce_gt_division_of_constants` -- absent from BOTH of this file's
  blocks entirely (not even as a commented-out line); `mixed_
  reduction.cc`'s own `r` block wires it in (`r_without_reduce_gt_zero_
  prod` does not, there either).

Ported as three real, disclosed absences -- not restored to match
`mixed_reduction.py`'s own composition, which would silently diverge
this file from what `full_reduction.cc` actually builds.
"""

from __future__ import annotations

from pymoses.comboreduct.combo.vertex import combo_tree
from pymoses.comboreduct.reduct.branch_rules import reduce_cond_else
from pymoses.comboreduct.reduct.contin_rules import (
    reduce_exp_div,
    reduce_exp_log,
    reduce_exp_times,
    reduce_factorize,
    reduce_factorize_fraction,
    reduce_fraction,
    reduce_invert_constant,
    reduce_log_div_times,
    reduce_plus_times_one_child,
    reduce_plus_zero,
    reduce_sin,
    reduce_sum_log,
    reduce_times_div,
    reduce_times_one_zero,
)
from pymoses.comboreduct.reduct.general_rules import eval_constants, level, reorder_commutative
from pymoses.comboreduct.reduct.logical_rules import (
    eval_logical_identities,
    insert_ands,
    reduce_and_assumptions,
    reduce_ands,
    reduce_nots,
    reduce_or_assumptions,
    reduce_ors,
    remove_dangling_junctors,
    remove_unary_junctors,
)
from pymoses.comboreduct.reduct.meta_rules import downwards, iterative, sequential, upwards
from pymoses.comboreduct.reduct.mixed_rules import (
    reduce_contin_if,
    reduce_contin_if_inner_op,
    reduce_contin_if_not,
    reduce_contin_if_substitute_cond,
    reduce_contin_if_to_impulse,
    reduce_from_assumptions,
    reduce_gt_zero_const_div,
    reduce_gt_zero_const_sum_sin,
    reduce_gt_zero_div,
    reduce_gt_zero_exp,
    reduce_gt_zero_impulse,
    reduce_gt_zero_minus_exp,
    reduce_gt_zero_pair_power,
    reduce_gt_zero_prod,
    reduce_gt_zero_sin,
    reduce_gt_zero_sum,
    reduce_gt_zero_sum_sin,
    reduce_gt_zero_times_const,
    reduce_impulse_power,
    reduce_impulse_prod,
    reduce_impulse_sum,
    reduce_inequality_from_assumptions,
    reduce_op_contin_if,
)
from pymoses.comboreduct.reduct.reduct import _NO_IT, rule
from pymoses.comboreduct.table.table import type_node
from pymoses.util.tree import tree_node

_cached_rule: rule | None = None


def _reset_cache_for_testing() -> None:
    """No direct C++ symbol -- see `mixed_reduction.py`'s identical hook
    for the reasoning.
    """
    global _cached_rule
    _cached_rule = None


def _logical_block() -> list:
    """No direct C++ symbol: the "general" + "logical" prefix
    (full_reduction.cc:59-75/152-167) is IDENTICAL between the `r_
    without_reduce_gt_zero_prod` and `r` blocks -- factored out once
    here rather than written twice, matching real C++'s own byte-
    identical duplication (the two blocks really are copy-pasted in the
    source) but without repeating the transcription risk twice in this
    port. Returns a plain list (like `_contin_block`/`_simple_mixed_
    block`), spliced directly into the caller's own flat `sequential(...)`
    -- real C++ has ONE flat sequential here, not a sequential-of-
    sequentials; a nested `sequential` would in fact be behaviorally
    identical (composition is associative: `sequential` has no logic of
    its own beyond calling each wrapped rule in order at the same point,
    so nesting one changes nothing observable), but a flat list keeps
    this helper's own shape consistent with its two siblings rather than
    relying on that equivalence.

    The ONE genuinely nested combinator here (`iterative(sequential(...))`,
    full_reduction.cc:66-74) is a real, distinct fixpoint sub-loop in
    C++ itself, not organizational -- kept nested exactly as C++ has it,
    as one element of the returned list.
    """
    return [
        # general
        downwards(level()),
        upwards(eval_constants()),
        # logical
        downwards(reduce_nots(), type_node.boolean_type),
        upwards(remove_dangling_junctors()),
        iterative(
            sequential(
                upwards(eval_logical_identities()),
                downwards(level()),
                downwards(insert_ands(), type_node.boolean_type),
                downwards(reduce_ands(), type_node.boolean_type),
                downwards(reduce_ors(), type_node.boolean_type),
            )
        ),
        downwards(remove_unary_junctors(), type_node.boolean_type),
    ]


def _contin_block() -> list:
    """No direct C++ symbol: the "simple contin" + "complex contin"
    section, identical between both blocks -- same factoring reasoning
    as `_logical_block`. Returns a plain list of rules (not a
    `sequential`) so callers can splice it directly into their own
    `sequential(...)` argument list, matching how C++ inlines these
    lines directly rather than nesting a sub-sequential.
    """
    return [
        downwards(reduce_plus_times_one_child()),
        downwards(reduce_plus_zero()),
        downwards(reduce_times_one_zero()),
        downwards(reduce_sin()),
        downwards(reduce_invert_constant()),
        downwards(reduce_log_div_times()),
        downwards(reduce_exp_times()),
        downwards(reduce_exp_div()),
        # The following rule is not valid if log has the semantics
        # log(abs(x)). ABS_LOG is commented out at vertex.h:49, so this
        # is always compiled into classic and always ported here.
        downwards(reduce_exp_log()),
        downwards(reduce_times_div()),
        downwards(reduce_sum_log()),
        # complex contin
        upwards(reorder_commutative()),
        downwards(reduce_factorize()),
        downwards(reduce_factorize_fraction()),
        downwards(reduce_fraction()),
    ]


def _simple_mixed_block() -> list:
    """No direct C++ symbol: the "simple mixed" + "complex mixed" section
    up to (not including) the assumption-drawing rules -- identical
    between both blocks. `reduce_gt_zero_log` and `reduce_junction_gt_
    zero_sum_constant` are commented out in REAL C++ here (see this
    module's own docstring); NOT included.
    """
    return [
        downwards(reduce_gt_zero_exp()),
        downwards(reduce_gt_zero_minus_exp()),
        downwards(reduce_gt_zero_times_const()),
        downwards(reduce_gt_zero_const_div()),
        downwards(reduce_gt_zero_const_sum_sin()),
        downwards(reduce_gt_zero_impulse()),
        downwards(reduce_impulse_sum()),
        downwards(reduce_contin_if_to_impulse()),
        # complex mixed
        upwards(reorder_commutative()),
        downwards(reduce_gt_zero_pair_power()),
        downwards(reduce_impulse_power()),
        downwards(reduce_impulse_prod()),
        downwards(reduce_contin_if()),
        downwards(reduce_op_contin_if()),
        downwards(reduce_cond_else()),
        downwards(reduce_contin_if_inner_op()),
        downwards(reduce_contin_if_substitute_cond()),
    ]


def full_reduction() -> rule:
    """C++: moses/comboreduct/reduct/full_reduction.cc:35 full_reduction"""
    global _cached_rule
    if _cached_rule is not None:
        return _cached_rule

    r_without_reduce_gt_zero_prod = iterative()
    r_without_reduce_gt_zero_prod.r = sequential(
        *_logical_block(),
        *_contin_block(),
        *_simple_mixed_block(),
        # very complex mixed (exponential of exponential)
        downwards(reduce_from_assumptions(r_without_reduce_gt_zero_prod)),
        downwards(reduce_inequality_from_assumptions()),
        downwards(reduce_contin_if_not(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sum(r_without_reduce_gt_zero_prod)),
        # commented to avoid infinite recursivity:
        # downwards(reduce_gt_zero_prod()),
        downwards(reduce_gt_zero_div(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sum_sin(r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_sin(r_without_reduce_gt_zero_prod)),
        downwards(reduce_and_assumptions(r_without_reduce_gt_zero_prod)),
        downwards(reduce_or_assumptions(r_without_reduce_gt_zero_prod)),
    )

    r = iterative()
    r.r = sequential(
        *_logical_block(),
        *_contin_block(),
        *_simple_mixed_block(),
        # very complex mixed
        downwards(reduce_from_assumptions(r)),  # maybe better at end
        downwards(reduce_inequality_from_assumptions()),
        downwards(reduce_contin_if_not(r)),
        downwards(reduce_gt_zero_sum(r)),
        downwards(reduce_gt_zero_prod(r, r_without_reduce_gt_zero_prod)),
        downwards(reduce_gt_zero_div(r)),
        downwards(reduce_gt_zero_sum_sin(r)),
        downwards(reduce_gt_zero_sin(r)),
        downwards(reduce_and_assumptions(r)),
        downwards(reduce_or_assumptions(r)),
    )

    if _cached_rule is None:
        _cached_rule = r
    return _cached_rule


def full_reduce(tr: combo_tree, it: tree_node = _NO_IT) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:181 full_reduce

    Two C++ overloads collapsed into one Python signature (this port's
    usual convention): `full_reduce(tr, it)` (reduct.h:181-184) and
    `full_reduce(tr)` (reduct.h:186-189, `it` omitted -> the 1-arg
    `rule.__call__` overload).
    """
    full_reduction()(tr, it)


def clean_and_full_reduce(tr: combo_tree, it: tree_node = _NO_IT) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:209 clean_and_full_reduce

    C++'s own body: `clean_and_full_reduction()(tr, it)` is COMMENTED
    OUT (the class does not exist -- `reduct.h:105`'s own `//const rule&
    clean_and_full_reduction();` is likewise commented out); the real
    body is `clean_reduce(tr, it); full_reduce(tr, it);` -- two separate
    calls, not one combined rule. Ported as that same two-call sequence.
    """
    from pymoses.comboreduct.reduct.clean_reduction import clean_reduce

    clean_reduce(tr, it)
    full_reduce(tr, it)
