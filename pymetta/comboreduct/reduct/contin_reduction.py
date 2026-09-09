"""C++: moses/comboreduct/reduct/contin_reduction.cc

S14 Task 4. `contin_reduction(reduct_effort, ignore_ops)` assembles the
continuous-arithmetic rule set -- this is what `problem-params.cc:2394`
calls to build `cand_reduct`, and `logical_rules.py`'s `simplify_predicates`
calls (via `contin_reduce`) to reduce a `greater_than_zero` predicate's own
continuous argument. COMPOSITION ORDER IS THE NORMAL FORM, the same
governing fact `logical_reduction.py`'s own module docstring states for
its own effort levels -- every rule, `when`-gate and combinator below is
ordered EXACTLY as `contin_reduction.cc` orders it.

STATIC-SINGLETON CACHING, ported faithfully -- DO NOT FIX: C++'s own
comment explains the mechanism (a lock-guarded `static rule *pr`, built
from whichever `(reduct_effort, ignore_ops)` the FIRST call anywhere in
the process happened to pass, avoiding the cost of re-locking on every
later call). The consequence, real and disclosed rather than papered
over: EVERY later call to `contin_reduction`, anywhere in the same
process, for the REST OF THAT PROCESS's lifetime, silently ignores its
own `reduct_effort`/`ignore_ops` arguments and returns the SAME rule
object the first call built. `_cached_rule` (module-level, below)
reproduces exactly that: a real, disclosed divergence from "just call
the function again with new arguments and get a rule built from them" --
which is what a naive reading of the signature would suggest, but not
what either C++ or this port actually does.

`_reset_cache_for_testing()` exists ONLY so this port's own test suite
(a single long-lived Python process exploring MANY `(reduct_effort,
ignore_ops)` combinations, unlike a real one-shot CLI invocation of
classic) can observe more than the first configuration ever requested.
No direct C++ symbol -- a real C++ process has no equivalent (the cache
lives until the process exits), and the caching behaviour ITSELF (not
this reset hook) is what `test_contin_reduction.py`'s own dedicated
test proves.
"""

from __future__ import annotations

from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.reduct.branch_rules import (
    reduce_cond_adjacent,
    reduce_cond_arg,
    reduce_cond_const,
    reduce_cond_else,
)
from pymetta.comboreduct.reduct.contin_rules import (
    reduce_distribute,
    reduce_exp_div,
    reduce_exp_log,
    reduce_exp_times,
    reduce_factorize,
    reduce_factorize_fraction,
    reduce_fraction,
    reduce_impulse_arg,
    reduce_invert_constant,
    reduce_log_div_times,
    reduce_plus_times_one_child,
    reduce_plus_zero,
    reduce_sin,
    reduce_sum_log,
    reduce_times_div,
    reduce_times_one_zero,
)
from pymetta.comboreduct.reduct.general_rules import eval_constants, level, reorder_commutative
from pymetta.comboreduct.reduct.meta_rules import (
    downwards,
    ignore_size_increase,
    iterative,
    sequential,
    upwards,
    when,
)
from pymetta.comboreduct.reduct.reduct import _NO_IT, ignores, rule
from pymetta.util.tree import tree_node

_cached_rule: rule | None = None


def _reset_cache_for_testing() -> None:
    """No direct C++ symbol -- see this module's own docstring."""
    global _cached_rule
    _cached_rule = None


def contin_reduction(reduct_effort: int, ignore_ops: set) -> rule:
    """C++: moses/comboreduct/reduct/contin_reduction.cc:36 contin_reduction

    See this module's own docstring for the static-singleton-caching
    behaviour: only the FIRST call's arguments, ever, actually shape the
    returned rule.
    """
    global _cached_rule
    if _cached_rule is not None:
        return _cached_rule

    seq_without_factorize_distribute = sequential(
        # These next two below are performed first, because
        # representation building puts in lots of 0*#n terms,
        # and getting rid of these early saves headaches later.
        downwards(reduce_times_one_zero()),
        downwards(reduce_plus_zero()),
        downwards(level()),
        upwards(eval_constants()),
        downwards(reduce_plus_times_one_child()),
        downwards(reduce_plus_zero()),
        downwards(reduce_times_one_zero()),
        downwards(reduce_cond_const()),
        downwards(reduce_cond_else()),
        downwards(reduce_cond_adjacent()),
        when(downwards(reduce_sin()), not ignores(ignore_ops, builtin.sin)),
        when(downwards(reduce_invert_constant()), not ignores(ignore_ops, builtin.div)),
        when(downwards(reduce_log_div_times()), not ignores(ignore_ops, builtin.log)),
        when(downwards(reduce_exp_times()), not ignores(ignore_ops, builtin.exp)),
        when(
            downwards(reduce_exp_div()),
            not ignores(ignore_ops, builtin.exp) and not ignores(ignore_ops, builtin.div),
        ),
        # The following rule is not valid if log has the semantics
        # log(abs(x)). ABS_LOG is commented out at vertex.h:49, so this
        # is always compiled into classic and always ported here.
        when(
            downwards(reduce_exp_log()),
            not ignores(ignore_ops, builtin.sin) and not ignores(ignore_ops, builtin.log),
        ),
        when(downwards(reduce_times_div()), not ignores(ignore_ops, builtin.div)),
        when(downwards(reduce_sum_log()), not ignores(ignore_ops, builtin.log)),
        when(
            downwards(reduce_impulse_arg(reduct_effort, ignore_ops)),
            not ignores(ignore_ops, builtin.impulse),
        ),
        when(
            downwards(reduce_cond_arg(reduct_effort, ignore_ops)),
            not ignores(ignore_ops, builtin.cond),
        ),
        upwards(reorder_commutative()),
        when(downwards(reduce_fraction()), not ignores(ignore_ops, builtin.div)),
        name="seq_without_factorize_distribute",
    )

    iter_without_factorize_distribute = iterative(
        seq_without_factorize_distribute, name="iter_without_factorize_distribute"
    )

    complete_factorize = sequential(
        downwards(reduce_factorize()),
        when(downwards(reduce_factorize_fraction()), not ignores(ignore_ops, builtin.div)),
        seq_without_factorize_distribute,
        name="complete_factorize",
    )

    complete_distribute = downwards(reduce_distribute(iter_without_factorize_distribute))

    res = iterative(
        sequential(
            seq_without_factorize_distribute,
            ignore_size_increase(complete_factorize),
            # ignore_size_increase(sequential(complete_factorize, complete_distribute)),
            ignore_size_increase(sequential(complete_distribute, complete_factorize)),
        ),
        name="contin_reduction",
    )

    if _cached_rule is None:
        _cached_rule = res
    return _cached_rule


def contin_reduce(
    tr: combo_tree, it: tree_node = _NO_IT, *, reduct_effort: int = 2, ignore_ops: set | None = None
) -> None:
    """C++: moses/comboreduct/reduct/reduct.h:141 contin_reduce

    Two C++ overloads collapsed into one Python signature (this port's
    usual convention): `contin_reduce(tr, it, reduct_effort, ignore_ops)`
    (reduct.h:141-146) and `contin_reduce(tr, reduct_effort, ignore_ops)`
    (reduct.h:148-153, `it` omitted -> the 1-arg `rule.__call__` overload).
    `tr` first, matching C++'s own parameter order exactly (unlike
    `logical_reduce`'s different, effort-first order in
    `logical_reduction.py`). `reduct_effort`/`ignore_ops` are made
    KEYWORD-ONLY here specifically so omitting `it` (the C++ 3-arg
    overload) never needs `it`'s own `_NO_IT` sentinel threaded through
    positionally -- callers write `contin_reduce(tr, reduct_effort=2,
    ignore_ops=ops)` for that overload, and `contin_reduce(tr, node,
    reduct_effort=2, ignore_ops=ops)` for the other. `it`'s default is
    `reduct.py`'s own `_NO_IT` sentinel, matching `logical_reduce`'s
    identical reasoning; `ignore_ops` defaults to an empty set (matching
    every real C++ call site under `moses/comboreduct/main/
    contin-reductor.cc`'s own `vertex_set()`).
    """
    if ignore_ops is None:
        ignore_ops = set()
    contin_reduction(reduct_effort, ignore_ops)(tr, it)
