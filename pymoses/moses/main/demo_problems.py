"""C++: moses/moses/main/demo-problems.{h,cc}

The toy/demo problems selected by `-H pa|dj|maj|mux|sr|cp|ann-cp`:
`pa` (even parity), `dj` (logical disjunction), `maj` (majority), `mux`
(multiplexer), `sr` (polynomial regression), `cp` (learn a given combo
program) and `ann-cp` (the ANN variant of `cp`). Ported per the S13 P2
"demo problems" task brief: unblocks the seven `MOSESUTest` cases that
have been stuck since S11 on two missing dependencies -- this file itself,
and `logical_bscore` (bscores.h:52) -- both of which `pymoses/moses/main/
problem.py`'s own module docstring already names as the reason
`test_2_parity`, `test_2_parity_with_labels`, `test_3_parity_multi_thread`
(`-Hpa`), `test_1_mux`, `test_2_mux` (`-Hmux`) and `test_cp_simple`,
`test_cp` (`-Hcp`) are unreachable, and as the reason `pymoses/moses/moses/
moses_main.py`'s own test suite could not exercise them either.

## Where `logical_bscore` lives, and why

`bscores.h:52`'s `logical_bscore` is used ONLY by this file in classic
(verified: `git grep -n logical_bscore` inside `$CLASSIC` returns exactly
11 hits, all in `demo-problems.cc` or `bscores.{h,cc}` itself -- no other
`.cc` file references it). `pymoses/moses/scoring/bscores.py` is owned,
for this task, by a concurrent agent porting `ctruth_table_bscore`, and
this task's own brief says explicitly: put `logical_bscore` in this
module instead of editing that file, and say so here. Done: `logical_bscore`
below is a complete, independent port of `bscores.h:52-79` and
`bscores.cc:64-134`, needing nothing from `bscores.py` (it does not share
a base-class *file* with `ctruth_table_bscore`, only the `bscore_base`
class both extend, from `scoring_base.py`, which this task does not touch
either). Anyone completing a from-scratch `pymoses/moses/scoring/
bscores.py` port later should either import `logical_bscore` from here, or
delete this copy and update this module's import -- this docstring is the
seam.

## Small dependencies this task also had to supply, and why they live here

Three more C++ symbols this file's own logic needs turned out not to be
ported ANYWHERE in this project as of this task (grepped, not assumed):

* `alphabet_size` (problem.cc:138-196). `pymoses/moses/main/problem.py`'s
  own module docstring claims this helper is "used only from
  `table-problems.cc`'s concrete `*_table_problem::run()` bodies" and is
  therefore out of scope for that file -- but `demo-problems.cc:112`
  (`bool_problem_base::run`) calls it too, so that claim is incomplete
  (not something this task's brief asked it to fix; noted here because it
  is directly why this port needs its own copy). `problem.py` is not
  owned by either of this task's two concurrent sibling agents (only
  `problem_params.py` and the whole `pymoses/moses/scoring/` tree are),
  but it is a stable, already-committed S13 Task 0 deliverable this task
  was not asked to touch either -- so `_alphabet_size` below is a private,
  local copy scoped to exactly what `bool_problem_base`/`combo_problem`
  need, not a change to that file.
* `type_to_exemplar` (representation.cc:75-92, declared representation.h:
  42). Grepped: no Python port exists. `_type_to_exemplar` below ports
  only the two branches this file's in-scope problems can ever reach
  (`boolean_type`, `contin_type`) -- see its own docstring.
* `even_parity`/`disjunction`/`multiplex`/`majority`/
  `simple_symbolic_regression` (examples/example-progs/
  scoring_iterators.h) -- plain iterator-based functors with no Python
  port anywhere. The first four are reimplemented below as ordinary
  functions/callables over a plain `list[bool]`, since that is all
  `logical_bscore`'s functional constructor ever feeds them (see
  `_iterator_function_truth_table`); the fifth over a
  `Sequence[vertex]`, since that is what `contin_bscore`'s own functional
  constructor feeds it. All five live HERE rather than in some
  `examples/`-shaped package of their own because `demo-problems.cc` is
  the ONLY file in the entire classic tree that includes
  `scoring_iterators.h` -- measured, not assumed: `grep -rn
  scoring_iterators $CLASSIC --include='*.cc' --include='*.h'
  --include='*.txt'` returns exactly two hits, the header itself and
  `moses/moses/main/demo-problems.cc:32`. Not one of the sixteen programs
  in `examples/example-progs/` uses it. A header whose sole consumer is
  this file belongs with this file.

## `sr` and `cp`'s contin arm: LIVE since 2026-09-09; `ann-cp` still a stub

This section previously said `sr` and `cp`'s `contin_type` arm both
raised because "`ITable`/`OTable`/`contin_bscore` are S14 territory this
port has not built (`problem_params.contin_reduct` is still `None`)".
**Every clause of that was false by the time it was read**: `contin_bscore`
is `pymoses/moses/scoring/bscores.py`'s `contin_bscore`, `ITable`/`OTable`
are `pymoses/comboreduct/table/table.py`'s, and
`problem_params.contin_reduct` is a real `contin_reduction` object. Both
problems are now ported for real and reproduce classic byte for byte; see
`tests/moses/main/test_moses_exec.py`'s `-Hsr`/`-Hcp` oracle block for
the captured commands.

What `sr` genuinely still needed, and where it came from:

* `simple_symbolic_regression` (scoring_iterators.h:110) -- see the
  bullet above for why all five `scoring_iterators.h` functors live here.
* `ITable(const type_tree&, int nsamples, contin_t, contin_t)`
  (table.cc:93), the random-sampling constructor. Added to `table.py` (it
  belongs with the class, not here) and, being an RNG CONSUMER, verified
  draw for draw against a compiled probe rather than set-wise -- see
  `ITable.__init__`'s own docstring there.
* `pms.deme_params.linear_contin = False` (demo-problems.cc:257-259,
  "sr is fundamentally a kind of non-linear regression! over-ride any
  flag settings regarding this"). Load-bearing: with the override
  dropped, `-Hsr -k3 -m1000` finds a different program.
  `test_polynomial_problem_run_overrides_linear_contin` pins it.

`ann-cp` (`ann_combo_problem`, demo-problems.cc:451-496) still raises: on
top of the contin machinery that now exists, it needs the stale ANN
combo-vocabulary path CLAUDE.md's own "Known sharp edges" section flags as
legacy. It stays registered (so a `-H ann-cp` lookup succeeds, matching
classic's own successful `find_problem`) with the SAME
name()/description() strings as its real C++ class and a `run()` that
raises `NotImplementedError` immediately -- the same "port the gate,
raise inside it" discipline `problem.py`'s own `_unported_table_problem`
established for the nine unreachable `table-problems.cc` problems. The
SAME gap is why `OTable.__init__`'s combo-tree branch (table.py) does not
port `OTable::OTable(const combo_tree&, const ITable&)`'s own
`is_ann_type(*tr.begin())` arm (table.cc:323-338).

## The missing general type-inference engine, and the narrow substitute

`combo_problem`/`combo_problem_base` need `infer_type_tree` (a whole-tree
type-unification pass, type_tree.cc, ~1468 lines) to determine a parsed
`-y` combo program's output type, INPUT types, arity, and
well-formedness (demo-problems.cc:310-344, 373-380).
`pymoses/comboreduct/type_checker/type_tree.py`'s own module docstring is
explicit that this general engine is NOT ported and is out of scope: only
`get_output_type_tree`, a SINGLE-VERTEX lookup, exists.
`_infer_signature` below is a narrow, disclosed substitute sufficient for
exactly what `combo_problem` consults:

* the ROOT vertex's output type, via the one function that IS ported;
* the highest argument index anywhere in the tree, which for a combo
  program IS `type_tree_arity`'s "how many input slots" count;
* each argument's INPUT type, taken from the type its PARENT vertex
  expects of its children (`_BUILTIN_INPUT_TYPE` below). Classic gets the
  same answer by unification; for the fixed-input-type builtins this port
  can parse there is nothing to unify, since every child of `and`/`or`/
  `not`/`impulse` is boolean and every child of `plus`/`times`/`div`/
  `exp`/`log`/`sin`/`greater_than_zero` is contin.

Measured against classic's real `infer_type_tree` (a compiled probe over
nine programs), this substitute agrees on all four of the facts above for
every program it accepts, INCLUDING the two non-obvious ones:

* an argument index that never appears gets `unknown_type`, not the
  surrounding type -- `or($1 !$3)` really is `->(boolean unknown boolean
  boolean)` in classic, so its `boolean_arity` is 2 while its
  `type_tree_arity` is 3;
* a genuine cross-argument type mismatch is reported as ill-formed --
  `+(impulse($1) $1)` really is `->(ill_formed ill_formed)`, `wf=0`.

Two disclosed differences remain, neither reachable by anything this file
runs:

* On a mismatch classic marks BOTH the input slot and the OUTPUT
  `ill_formed`; this substitute marks only the input slot, because its
  output type comes from the root vertex alone and no unification can
  reach back to it. `is_well_formed` asks only "does any `ill_formed`
  appear anywhere" (type_tree.cc:1210-1218), so the OBSERVABLE outcome --
  `check_args` rejecting the program -- is identical.
* Position-dependent builtins (`cond`, `contin_if`) and the container/
  higher-order family (`list`/`car`/`cdr`/`cons`/`foldr`/`foldl`/
  `apply`/`lambda`) have no `_BUILTIN_INPUT_TYPE` entry at all, and an
  argument under one raises `NotImplementedError` rather than guessing.
  `type_tree_arity`'s `-1` (ill-formed-arity) sentinel is likewise still
  unreachable, so `check_args` still relies entirely on the OTHER half of
  demo-problems.cc:337's OR condition (no literal `$` in the raw `-y`
  string) to catch the "forgot to escape the $'s in the shell" case; see
  that method's own docstring.

## `demo_params`: registered per-problem, not through `problem_params`

Classic's `demo_params` (demo-problems.cc:39-45) is a SEPARATE
`option_base` from `problem_params`, holding just `-y`/`--combo-program`
and `-k`/`--problem-size`; `register_demo_problems` builds ONE `demo_params`
instance, registers it for CLI parsing via `option_manager& mgr`, and
passes the SAME instance into every demo problem's constructor (so
`pa_problem`/`dj_problem`/.../`combo_problem` all read `_dparms.problem_size`/
`_dparms.combo_str` off that one shared object, never off `problem_params`
itself). `option_base`/`option_manager` are deliberately not ported
anywhere in this project (`problem.py`'s own module docstring), so there is
no CLI-registration step to port here either -- `demo_params` below is a
plain data holder with the same two fields and the same two CLI defaults
(`problem_size=5`, `combo_str=""`, matching demo-problems.cc:59,54's
`po::value<...>()->default_value(5)` / a default-constructed empty
`std::string`), and `register_demo_problems` RETURNS the instance it
built and registered problems against (no C++ equivalent -- callers need
a handle on it to configure `-k`/`-y`-equivalent values themselves, since
there is no `option_manager` to do that for them; this is this file's own
seam for whenever `moses_exec.py`'s CLI wiring exists).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence

from pymoses.comboreduct.combo.argument import argument
from pymoses.comboreduct.combo.common_def import contin_t
from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    get_contin,
    is_argument,
    is_builtin,
    is_contin,
    vertex,
)
from pymoses.comboreduct.table.table import ITable, OTable, complete_truth_table
from pymoses.comboreduct.type_checker.type_signature import (
    TypeSignature,
    gen_signature,
    ostream_type_node,
    type_node,
    type_tree_arity,
)
from pymoses.comboreduct.type_checker.type_tree import get_output_type_tree
from pymoses.moses.main.problem import problem_base, problem_manager
from pymoses.moses.moses.moses_main import metapop_moses_results
from pymoses.moses.moses.types import _to_score_t, behavioral_score, score_t
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import contin_bscore
from pymoses.moses.scoring.scoring_base import bscore_base
from pymoses.util.logger import logger

__all__ = [
    "combo_problem",
    "demo_params",
    "dj_problem",
    "logical_bscore",
    "majority_problem",
    "mux_problem",
    "pa_problem",
    "polynomial_problem",
    "register_demo_problems",
    "simple_symbolic_regression",
]


# ---------------------------------------------------------------------------
# demo_params -- see module docstring's own section.
# ---------------------------------------------------------------------------


class demo_params:
    """C++: moses/moses/main/demo-problems.cc:39 demo_params

    Not an `option_base` (see module docstring): a plain holder for the
    two demo-problem-only CLI options, `-y`/`--combo-program` and
    `-k`/`--problem-size`, with their real CLI defaults (demo-problems.cc:
    53-59) as constructor defaults here.
    """

    def __init__(self, combo_str: str = "", problem_size: int = 5) -> None:
        self.combo_str = combo_str
        self.problem_size = problem_size


# ---------------------------------------------------------------------------
# set_noise_or_ratio -- demo-problems.cc:72-79. A free template function in
# C++ (`template <typename BScorer> void set_noise_or_ratio(...)`); ported as
# a plain function since Python has no separate template-instantiation step
# and `bscore_base.set_complexity_coef`'s own argument-count dispatch (see
# scoring_base.py) already collapses the two C++ overloads this calls into
# one method.
# ---------------------------------------------------------------------------


def _set_noise_or_ratio(
    scorer: bscore_base, alphabet_size: int, noise: float, ratio: score_t
) -> None:
    """C++: moses/moses/main/demo-problems.cc:73 set_noise_or_ratio"""
    if noise >= 0.0:
        scorer.set_complexity_coef(alphabet_size, noise)
    else:
        scorer.set_complexity_coef(ratio)


# ---------------------------------------------------------------------------
# alphabet_size -- see module docstring's own section for why this is a
# private copy rather than an edit to problem.py.
# ---------------------------------------------------------------------------


def _log_alphabet_size(alphabet_size: int, output_type: type_node) -> int:
    """C++: moses/moses/main/problem.cc:138 alphabet_size

    Not a C++ symbol; the shared tail of `_alphabet_size` below, which
    returns early from each arm where C++ falls through to a single
    `logger().info() << "Alphabet size = " << as << " output = " <<
    output_type;`. Own copy of `table_problems.py`'s identical helper, for
    the reason that file's own `_alphabet_size` docstring gives.
    """
    logger().info(f"Alphabet size = {alphabet_size} output = {ostream_type_node(output_type)}")
    return alphabet_size


def _alphabet_size(sig: TypeSignature, ignore_ops: Sequence[object]) -> int:
    """C++: moses/moses/main/problem.cc:138 alphabet_size

    "Determine the alphabet size given the type_tree of the problem and a
    list of operators to ignore. This is a somewhat ad-hoc value, meant to
    help compute the complexity ratio." (problem.cc:132-134's own comment.)

    `sig.otype` is used directly in place of C++'s
    `get_type_node(get_signature_output(tt))`: `get_type_node` is not
    ported anywhere in this project (grepped), and
    `get_signature_output(sig).node` -- the ONE thing it would be called
    on here -- is definitionally `sig.otype` (`type_signature.py`'s own
    `get_signature_output`), so this is a direct equivalent, not an
    approximation.

    Only the branches this file's in-scope problems can reach are real
    implementations: `boolean_type` (every `bool_problem_base` subclass,
    and `cp` on a boolean-output program) and `contin_type` (`sr`, and
    `cp` on a contin-output program -- both live since 2026-09-09).
    `ann_type` is ported for fidelity, though only the unported `ann-cp`
    could reach it.
    `enum_type` (problem.cc:157-171) would need `enum_t.size()` plus the
    `contains_type` scan this port has no call site for anywhere else
    either; raised explicitly rather than guessed at. Every other
    `type_node` falls to the same "not supported" `else` real C++ reaches
    (problem.cc:186-195, `log_output_error_exit` -- an unconditional
    `exit(1)`, not a raised/catchable error there; ported as `ValueError`
    per this port's "port the gate, raise inside it" convention for a
    hard CLI-input failure, not a P2 in-scope-but-unported gap).
    """
    arity = type_tree_arity(sig)
    output_type = sig.otype
    # C++ assigns each branch to a single `as` and falls through to ONE
    # `logger().info()` at problem.cc:198-200 before its single `return`;
    # `_log_alphabet_size` is that shared tail. See `table_problems.py`'s
    # own copy of this function, which does the same.
    if output_type == type_node.boolean_type:
        # 3 operators: and, or, not
        return _log_alphabet_size(3 + arity, output_type)
    if output_type == type_node.contin_type:
        return _log_alphabet_size(8 + arity - len(ignore_ops), output_type)
    if output_type == type_node.ann_type:
        return _log_alphabet_size(2 + arity * arity, output_type)
    if output_type == type_node.enum_type:
        raise NotImplementedError(
            "alphabet_size: enum_type needs enum_t.size()/contains_type, not "
            "wired for alphabet computation anywhere in this port; unreachable "
            "by any demo problem in this file's scope (problem.cc:157-171)"
        )
    raise ValueError(f"alphabet_size: type {output_type!r} currently not supported.")


# ---------------------------------------------------------------------------
# type_to_exemplar -- see module docstring's own section.
# ---------------------------------------------------------------------------


def _type_to_exemplar(output_type: type_node) -> combo_tree:
    """C++: moses/moses/representation/representation.cc:75 type_to_exemplar

    Only the two branches this file's problems can ever request:
    `boolean_type` (every `bool_problem_base` subclass, always, plus
    `cp` on a boolean-output program) and `contin_type`
    (`polynomial_problem.run`'s own default exemplar, and `cp` on a
    contin-output program). `enum_type`
    (`combo_tree(cond); append_child(..., enum_t::get_random_enum())`)
    and the `ill_formed_type`/default `OC_ASSERT(false, ...)` branches
    are not reachable by anything in this file and are not ported.
    """
    if output_type == type_node.boolean_type:
        return combo_tree(builtin.logical_and)
    if output_type == type_node.contin_type:
        return combo_tree(builtin.plus)
    raise NotImplementedError(
        f"type_to_exemplar: type {output_type!r} not supported by this port "
        "(only boolean_type/contin_type are ever requested by this file's "
        "in-scope problems; representation.cc:75's enum_type/default branches "
        "are not ported)"
    )


# ---------------------------------------------------------------------------
# even_parity / disjunction / multiplex / majority --
# examples/example-progs/scoring_iterators.h. Not ported anywhere else in
# this project (grepped) -- reimplemented here over a plain `list[bool]`,
# which is all `_iterator_function_truth_table` ever feeds them (unlike
# `complete_truth_table`'s own tree-based `_populate`, table.py, which
# evaluates a combo_tree through `boolean_interpreter` and therefore deals in
# builtin-encoded vertices -- these four operate on raw Python `bool`,
# exactly like their C++ originals operate on raw `bool` iterators, never on
# `combo::builtin`).
# ---------------------------------------------------------------------------


def even_parity(bits: Sequence[bool]) -> bool:
    """C++: examples/example-progs/scoring_iterators.h:44 even_parity

    "even_parity(x1, ..., xn) = true iff (int)x1 + ... + (int)xn is even"
    (scoring_iterators.h:42-43's own comment) -- `parity ^= b` over every
    bit, seeded `True`, is exactly `bool(count_of_true_bits % 2 == 0)`.
    """
    parity = True
    for b in bits:
        parity ^= bool(b)
    return parity


def disjunction(bits: Sequence[bool]) -> bool:
    """C++: examples/example-progs/scoring_iterators.h:58 disjunction

    "disjunction(x1, ..., xn) = true iff there exists i such that xi is
    true" (scoring_iterators.h:55's own comment).
    """
    return any(bits)


class multiplex:
    """C++: examples/example-progs/scoring_iterators.h:73 multiplex

    "multiplex(a1, ..., an, d1, ..., dm) = 1 iff m = 2^n and di = 1 if i
    is the address of the string bit described by a1, ..., an."
    (scoring_iterators.h:66-67's own comment.) `arity` here is the NUMBER
    OF ADDRESS BITS (`n` above), matching the C++ constructor parameter
    name and meaning exactly -- NOT the full input width `mux_problem.
    get_arity` computes (`n + 2**n`), which is a different, larger value
    passed separately to `logical_bscore`.
    """

    def __init__(self, arity: int) -> None:
        self.arity = arity

    def __call__(self, bits: Sequence[bool]) -> bool:
        addr = 0
        for i in range(self.arity):
            if bits[i]:
                addr += 1 << i
        return bool(bits[self.arity + addr])


class majority:
    """C++: examples/example-progs/scoring_iterators.h:92 majority

    "majority(x1, ..., xn) = 0 iff n/2 or more arguments are false"
    (scoring_iterators.h:93's own comment) -- ported as the literal
    `count(true) > arity // 2` the C++ body computes, not simplified to
    "more than half are true" (the two read the same for this predicate,
    but this keeps the exact C++ expression shape, including its integer
    `arity / 2` floor division, e.g. `arity=5` needs a strict majority of
    3, not 2.5).
    """

    def __init__(self, arity: int) -> None:
        self.arity = arity

    def __call__(self, bits: Sequence[bool]) -> bool:
        return sum(1 for b in bits if b) > self.arity // 2


_UNSIGNED_MAX = 0xFFFFFFFF


def _dorepeat_count(n: int) -> int:
    """C++: opencog/util/dorepeat.h:41 dorepeat

    `#define dorepeat(N) for (unsigned int <uniq>=N; <uniq>>0;--<uniq>)`
    -- the loop counter is `unsigned int`, so a NEGATIVE `N` does not
    mean "zero iterations" (Python's `range(-1)`), it means `N` converted
    modulo 2**32. Extracted as a named function purely so the conversion
    is testable without actually running the loop: the only interesting
    input, `-1`, would run 4294967295 iterations.

    Reachable from this module through `-k`: classic declares
    `demo_params::problem_size` as `unsigned int` (demo-problems.cc:44),
    hands it to `simple_symbolic_regression(int o)` (scoring_iterators.h:
    112) -- which converts it BACK to `int` -- and then `dorepeat(order)`
    converts it to `unsigned int` a third time, so the iteration count is
    always exactly the `unsigned int` bit pattern `-k` produced.
    Confirmed live, not inferred: `( ulimit -v 2097152; timeout 20
    $CLASSIC/build/moses/moses/main/moses -Hsr -k-1 -m200 -W1 )` produces
    no output and exits 124 (timeout) -- classic really does try to run
    the loop 4294967295 times per table row. This port reproduces the
    hang rather than "fixing" it into 0 iterations.
    """
    return n & _UNSIGNED_MAX


class simple_symbolic_regression:
    """C++: examples/example-progs/scoring_iterators.h:110 simple_symbolic_regression

    "simple function : f(x)_o = sum_{i={1,o}} x^i" (scoring_iterators.h:
    104-108's own comment: `f(x)_3 = x+x^2+x^3`, `f(x)_0 = 0`), computed
    by Horner's rule: `contin_t res = 0; dorepeat(order) res = (res + 1) *
    get_contin(*from); return res;`.

    Two properties of that body are easy to lose and are deliberate here:

    * it reads ONLY `*from`, the FIRST input of the row, and never
      advances the iterator -- `to` is unused entirely. `-Hsr` always
      builds an arity-1 signature (demo-problems.cc:261,
      `gen_signature(id::contin_type, 1)`) so no other input exists, but
      the functor would ignore them if one did;
    * `order` is the `-k`/`--problem-size` value, and `dorepeat` counts in
      `unsigned int` -- see `_dorepeat_count` above.

    Takes a `Sequence[vertex]` (one table row) rather than C++'s
    `(It from, It to)` iterator pair, matching the four Boolean functors
    above and `contin_bscore`'s own functional constructor, which is the
    only caller: `contin_bscore(const Scoring& score, const ITable& r,
    eft)` (bscores.h:203) forwards to `OTable(const Func& f, const
    ITable& it, ...)` (table.h:886), whose body is
    `push_back(f(vs.get_seq<vertex>().begin(), vs.get_seq<vertex>().end()))`
    -- i.e. exactly the row, as vertices.
    """

    def __init__(self, o: int = 4) -> None:
        """C++: examples/example-progs/scoring_iterators.h:112 simple_symbolic_regression

        `simple_symbolic_regression(int o = 4) : order(o) { }` -- the `4`
        default is never used by `polynomial_problem::run`, which always
        passes `_dparms.problem_size`; kept because it is part of the C++
        declaration.
        """
        self.order = o

    def __call__(self, inputs: Sequence[vertex]) -> contin_t:
        """C++: examples/example-progs/scoring_iterators.h:115 simple_symbolic_regression

        Cites the `operator()` definition line (the project's convention
        for a symbol the breadcrumb matcher cannot name).
        """
        res: contin_t = 0.0
        for _ in range(_dorepeat_count(self.order)):
            res = (res + 1) * get_contin(inputs[0])
        return res


def _iterator_function_truth_table(
    func: Callable[[Sequence[bool]], bool], arity: int
) -> list[bool]:
    """C++: moses/comboreduct/table/table.h:1458 complete_truth_table(Func, arity)

    The `(Func, arity)` truth-table constructor `table.py`'s own
    `complete_truth_table` deliberately does not port (its module
    docstring names it explicitly as out of scope for that task) -- this
    is that missing overload, scoped to exactly what `logical_bscore`'s
    own functional constructor needs. Same bit order as table.h:1462-1466
    AND as `complete_truth_table._populate` (table.py): input `i`'s j-th
    bit is `(i >> j) % 2`.
    """
    size = 1 << arity
    result = [False] * size
    for i in range(size):
        bits = [bool((i >> j) & 1) for j in range(arity)]
        result[i] = bool(func(bits))
    return result


# ---------------------------------------------------------------------------
# logical_bscore -- bscores.h:52, bscores.cc:64-134. See module docstring.
# ---------------------------------------------------------------------------


class logical_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:52 logical_bscore

    "Each feature corresponds to an input tuple, 0 if the output of the
    candidate matches the output of the intended function, -1 otherwise."
    (bscores.h:48-50's own doc comment.)

    C++ has two constructor overloads, distinguished by a templated
    `const Func&` vs. a concrete `const combo_tree&` (bscores.h:55,61):
    collapsed here into one `__init__` dispatching on `isinstance(...,
    combo_tree)`, this port's usual convention for a C++ overload set
    distinguished only by parameter type. `reset_weights()` (bscores.h:
    57,63) is NOT called here -- boosting-weight infrastructure is out of
    scope for this whole port cycle, exactly as `ctruth_table_bscore.
    __init__`'s own docstring already documents for the same reason
    (bscores.py); `call_ensemble` (the `operator()(const
    scored_combo_tree_set&)` overload, bscores.cc:85-113) IS overridden
    here -- see its own docstring. It is the only such override in this
    port's reachable scope, which is exactly why `--boost` works on the
    demo problems and aborts on the table problems, as in classic.
    """

    def __init__(
        self, func_or_tree: Callable[[Sequence[bool]], bool] | combo_tree, arity: int
    ) -> None:
        """C++: moses/moses/scoring/bscores.h:55 logical_bscore (Func overload)
        and moses/moses/scoring/bscores.h:61 logical_bscore (combo_tree overload)
        """
        super().__init__()
        self._arity = arity
        if isinstance(func_or_tree, combo_tree):
            self._target: list[bool] = list(complete_truth_table(func_or_tree, arity))
        else:
            self._target = _iterator_function_truth_table(func_or_tree, arity)
        self._size = len(self._target)
        # C++ calls this at the end of BOTH ctors (bscores.h:59,65). It was
        # omitted while boosting was out of scope; with `call_ensemble`
        # ported it matters, because `_weights` must exist before
        # `update_weights` is ever called.
        self.reset_weights()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:67 logical_bscore::operator()

        "Compare the predictions of the tree to that of the desired
        result. A correct prediction gets a score of 0, an incorrect
        prediction gets a score of -1." (bscores.cc:70-71's own comment,
        typo in the original ("prdiction") not reproduced.)
        """
        tt = complete_truth_table(tr, self._arity)
        return [-_to_score_t(b1 != b2) for b1, b2 in zip(tt, self._target, strict=True)]

    def call_ensemble(self, ensemble) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:85 logical_bscore::operator()
        (the `const scored_combo_tree_set&` overload)

        "Boolean ensemble scorer. Assumes that the ensemble signature
        outputs a boolean value. All of the trees in the ensmble get a
        weighted vote, that vote is totalled to get the prediction of the
        ensemble, and then compared to the desired output." (bscores.cc:
        81-84's own comment; typo in the original not reproduced.)

        This is what makes `--boost` reachable at all:
        `bscore_base.call_ensemble` raises unconditionally, and
        `logical_bscore` is the only scorer in this port's reachable scope
        that overrides it. That matches classic, where `--boost` on a
        TABLE problem aborts inside `bscore_base::get_error` ("bscore
        error not implemented for bscorer ... ctruth_table_bscore",
        scoring_base.cc:115). Measured: classic exits 134 on
        `-i dataset.csv --boost=1` and exits 0 on `-Hpa -k3 --boost=1`.

        `sct.get_weight()` is a **double**; `hypoth` and the resulting
        bscore are `score_t` (binary32). C++ accumulates into a
        `behavioral_score`, i.e. in float, so every `+=` narrows --
        reproduced here rather than summing in double and narrowing once
        at the end, which would not be the same number.
        """
        # Step 1: accumulate the weighted prediction of each tree in the
        # ensemble.
        hypoth = [_to_score_t(0.0)] * self._size
        for sct in ensemble:
            tt = complete_truth_table(sct.get_tree(), self._arity)
            weight = sct.get_weight()
            for i in range(self._size):
                # Add +1 if the prediction is true, -1 if it is false.
                hypoth[i] = _to_score_t(
                    hypoth[i] + weight * (2.0 * _to_score_t(tt[i]) - 1.0)
                )

        # Step 2: compare the ensemble's prediction to the desired result.
        # `hypoth` is positive to predict true, negative to predict false;
        # the resulting score is 0 if correct and -1 if incorrect.
        return [
            -_to_score_t((hyp > 0.0) != b2)
            for hyp, b2 in zip(hypoth, self._target, strict=True)
        ]

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:115 logical_bscore::best_possible_bscore"""
        return [0.0] * self._size

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:120 logical_bscore::worst_possible_bscore"""
        return [-1.0] * self._size

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:125 logical_bscore::min_improv"""
        return 0.5

    def get_error(self, bs: behavioral_score) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:130 logical_bscore::get_error

        "It's minus the score: 0.0 is perfect score, 1.0 is worst score."
        (bscores.cc:132's own comment, typo "Its" not reproduced.) Not
        part of `bscore_base`'s ported interface (`scoring_base.py`'s own
        module docstring lists exactly what is) and nothing in this
        port's reachable scope calls it (feature-selection/boosting are
        both out of scope) -- ported anyway for fidelity, at zero
        maintenance cost, as a plain extra method.
        """
        return _to_score_t(-self.sum_bscore(bs) / self._size)


# ---------------------------------------------------------------------------
# bool_problem_base -- demo-problems.cc:84-122.
# ---------------------------------------------------------------------------


class bool_problem_base(problem_base):
    """C++: moses/moses/main/demo-problems.cc:84 bool_problem_base"""

    def __init__(self, dp: demo_params) -> None:
        self._dparms = dp

    @abstractmethod
    def get_arity(self, problem_size: int) -> int:
        """C++: moses/moses/main/demo-problems.cc:88 bool_problem_base::get_arity"""
        raise NotImplementedError

    @abstractmethod
    def get_bscore(self, problem_size: int) -> logical_bscore:
        """C++: moses/moses/main/demo-problems.cc:89 bool_problem_base::get_bscore"""
        raise NotImplementedError

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/demo-problems.cc:95 bool_problem_base::run

        `pms` is `ob`, duck-typed (see `problem_base.run`'s own docstring
        in `problem.py` for why): this port's `problem_params` has no
        `.printer` field yet (S13 CLI-output territory), so it stands in
        here for classic's `pms.mmr_pa` (the `metapop_printer` instance,
        itself S13/not ported) -- exactly the convention `moses_main.py`'s
        `metapop_moses_results`/`metapop_moses_results_b` already
        established: callers supply whatever callable they need.

        `logger().warn("Feature selection is not supported for the demo
        problems")` (demo-problems.cc:99-100) is ported below: note it
        precedes the exemplar default, and that classic only WARNS -- it
        does not refuse the flag.
        """
        pms = ob
        if pms.enable_feature_selection:
            logger().warn("Feature selection is not supported for the demo problems")

        if not pms.exemplars:
            pms.exemplars = [_type_to_exemplar(type_node.boolean_type)]

        bscore = self.get_bscore(self._dparms.problem_size)
        # C++: demo-problems.cc:109 -- `if (pms.meta_params.do_boosting)
        # bscore.use_weighted_scores();`. This line was REPLACED by a gate
        # while boosting was out of scope, so restoring it is part of
        # un-gating: without it `_return_weighted_score` stays false,
        # `reset_weights()` leaves `_weights` empty, and the first
        # `update_weights` from the AdaBoost loop asserts "Unexpected use
        # of weights in the bscorer!" (scoring_base.cc:169) -- measured.
        if pms.meta_params.do_boosting:
            bscore.use_weighted_scores()
            # `use_weighted_scores` only flips the flag; `_weights` is
            # sized by `reset_weights`, which the ctor already called
            # BEFORE the flag was set, so it sized an empty vector. C++
            # has the same ordering and the same consequence, and
            # `ensemble`'s own first action on the boosting path is
            # `_bscorer.reset_weights()` (ensemble.cc), which re-sizes it
            # now that the flag is true.
            bscore.reset_weights()
        # `--boost` is NOT gated here any more. The gate claimed
        # `logical_bscore` lacked `use_weighted_scores()`/`reset_weights()`;
        # both are ported on `bscore_base` (scoring_base.py) and the ctor
        # now calls `reset_weights()` as C++ does. What actually blocked
        # boosting was the missing `call_ensemble` override, which
        # `logical_bscore` now has -- see its own docstring for why that
        # one override is exactly what makes `--boost` work on the demo
        # problems and abort on the table problems, as in classic.

        sig = gen_signature(type_node.boolean_type, self.get_arity(self._dparms.problem_size))
        as_ = _alphabet_size(sig, pms.ignore_ops)
        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)

        cscore = behave_cscore(bscore)
        metapop_moses_results(
            pms.exemplars,
            sig,
            pms.bool_reduct,
            pms.bool_reduct_rep,
            cscore,
            pms.opt_params,
            pms.hc_params,
            pms.ps_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
        )


# ---------------------------------------------------------------------------
# pa / dj / maj / mux -- demo-problems.cc:124-222.
# ---------------------------------------------------------------------------


class pa_problem(bool_problem_base):
    """C++: moses/moses/main/demo-problems.cc:128 pa_problem"""

    def name(self) -> str:
        return "pa"

    def description(self) -> str:
        return "Learn parity function demo"

    def get_arity(self, sz: int) -> int:
        return sz

    def get_bscore(self, problem_size: int) -> logical_bscore:
        return logical_bscore(even_parity, problem_size)


class dj_problem(bool_problem_base):
    """C++: moses/moses/main/demo-problems.cc:149 dj_problem"""

    def name(self) -> str:
        return "dj"

    def description(self) -> str:
        return "Learn logicical disjunction demo"

    def get_arity(self, sz: int) -> int:
        return sz

    def get_bscore(self, problem_size: int) -> logical_bscore:
        return logical_bscore(disjunction, problem_size)


class majority_problem(bool_problem_base):
    """C++: moses/moses/main/demo-problems.cc:171 majority_problem"""

    def name(self) -> str:
        return "maj"

    def description(self) -> str:
        return "Majority problem demo"

    def get_arity(self, sz: int) -> int:
        return sz

    def get_bscore(self, problem_size: int) -> logical_bscore:
        return logical_bscore(majority(problem_size), problem_size)


class mux_problem(bool_problem_base):
    """C++: moses/moses/main/demo-problems.cc:194 mux_problem"""

    def name(self) -> str:
        return "mux"

    def description(self) -> str:
        return "Multiplex problem demo"

    def get_arity(self, sz: int) -> int:
        """C++: moses/moses/main/demo-problems.cc:202 mux_problem::get_arity

        "A mux of 5 would multiplex 32 bits, requiring a truth table of
        2^37 rows" (demo-problems.cc:205-206's own comment) -- classic
        logs, prints to stderr, and `exit(-1)`s; ported as `ValueError`
        (a hard CLI-input failure, this port's usual substitute for a
        non-recoverable `exit()` on bad input -- see `_alphabet_size`'s
        own docstring for the same convention).
        """
        if sz > 4:
            raise ValueError(
                "Error: maximum mux demo problem size is 4; use the -k flag "
                "to pick a smaller size"
            )
        return sz + (1 << sz)

    def get_bscore(self, problem_size: int) -> logical_bscore:
        func = multiplex(problem_size)
        return logical_bscore(func, self.get_arity(problem_size))


# ---------------------------------------------------------------------------
# sr -- demo-problems.cc:224-285.
# ---------------------------------------------------------------------------


class polynomial_problem(problem_base):
    """C++: moses/moses/main/demo-problems.cc:233 polynomial_problem"""

    def __init__(self, dp: demo_params) -> None:
        self._dparms = dp

    def name(self) -> str:
        return "sr"

    def description(self) -> str:
        return "Simple regression of f_n(x) = sum_{k={1,n}} x^k"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/demo-problems.cc:246 polynomial_problem::run

        Statement order is C++'s exactly, and that is load-bearing in two
        places:

        * the `ITable` construction is the ONLY RNG consumer in this
          method, so anything reordered around it would shift the draw
          stream that the whole rest of the run depends on;
        * classic's `OC_ASSERT(not pms.meta_params.do_boosting, ...)`
          comes AFTER the scorer is built (demo-problems.cc:272), not
          before, so a `--boost` run still draws its table first. Kept
          in that position rather than hoisted to the top of the method
          where it would read more naturally.

        `pms.deme_params.linear_contin = False` (demo-problems.cc:259) is
        an unconditional override of whatever `--linear-regression` set --
        "sr is fundamentally a kind of non-linear regression! over-ride
        any flag settings regarding this" (demo-problems.cc:257-258's own
        comment). See the module docstring for why this one assignment
        gets its own test.

        `logger().warn("Feature selection is not supported for the
        polynomial problem")` (demo-problems.cc:250-251) is dropped, not
        ported -- see `bool_problem_base.run`'s docstring for the same,
        already-established omission.
        """
        pms = ob

        # If no exemplar has been provided in the options, use the
        # default contin_type exemplar (+)
        if not pms.exemplars:
            pms.exemplars = [_type_to_exemplar(type_node.contin_type)]

        # sr is fundamentally a kind of non-linear regression!
        # over-ride any flag settings regarding this.
        pms.deme_params.linear_contin = False

        sig = gen_signature(type_node.contin_type, 1)

        it = ITable(sig, pms.nsamples if pms.nsamples > 0 else pms.default_nsamples)

        as_ = _alphabet_size(sig, pms.ignore_ops)

        eft = (
            contin_bscore.abs_error if pms.it_abs_err else contin_bscore.squared_error
        )
        bscore = contin_bscore(simple_symbolic_regression(self._dparms.problem_size), it, eft)

        if getattr(pms.meta_params, "do_boosting", False):
            raise NotImplementedError("Boosting not supported for this problem!")

        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)
        cscore = behave_cscore(bscore)
        metapop_moses_results(
            pms.exemplars,
            sig,
            pms.contin_reduct,
            pms.contin_reduct,
            cscore,
            pms.opt_params,
            pms.hc_params,
            pms.ps_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
        )


# ---------------------------------------------------------------------------
# cp / ann-cp -- demo-problems.cc:287-496.
# ---------------------------------------------------------------------------


def _infer_output_type_and_arity(tr: combo_tree) -> tuple[type_node, int]:
    """Not a single C++ function -- the narrow substitute for
    `infer_type_tree`/`type_tree_arity` this module's own docstring
    explains (see its "missing general type-inference engine" section).
    Reads the ROOT vertex's output type via the one really-ported
    function that can answer that (`get_output_type_tree`,
    `type_tree.py`) and the highest `argument` index anywhere in the
    tree (a plain walk -- `type_tree_arity` itself is just "how many
    input slots does this signature have", and for a combo program the
    highest referenced `$n` IS that count, with no type unification
    needed to find it).
    """
    root = tr.begin()
    output_type = get_output_type_tree(root.data).node
    arity = 0
    for node in tr:
        v = node.data
        if is_argument(v):
            assert isinstance(v, argument)
            arity = max(arity, abs(v.idx))
    return output_type, arity


# The type every child of a given builtin must have. Stands in for C++'s
# `get_input_type_tree(const vertex&, arity_t)` (type_tree.cc), which reads
# it out of the `builtins_properties` singleton that PARSES arrow-type
# strings like `"->(arg_list(boolean) boolean)"` (`descriptions.cc`) --
# machinery `type_tree.py`'s own module docstring rules out of scope, and
# which this file needs only for the fixed-input-type builtins.
#
# Deliberately PARTIAL, and every omission is a raise rather than a guess:
# `cond` and `contin_if` take position-dependent types, and the container/
# higher-order family (`list`/`car`/`cdr`/`cons`/`foldr`/`foldl`/`apply`/
# `lambda`) is parametric. `equ` is omitted for the same reason -- it has
# no fixed child type either. `rand`/`logical_true`/`logical_false` are
# nullary, so they can never be an argument's parent at all.
_BUILTIN_INPUT_TYPE: dict[builtin, type_node] = {
    builtin.logical_and: type_node.boolean_type,
    builtin.logical_or: type_node.boolean_type,
    builtin.logical_not: type_node.boolean_type,
    builtin.impulse: type_node.boolean_type,
    builtin.plus: type_node.contin_type,
    builtin.times: type_node.contin_type,
    builtin.div: type_node.contin_type,
    builtin.exp: type_node.contin_type,
    builtin.log: type_node.contin_type,
    builtin.sin: type_node.contin_type,
    builtin.greater_than_zero: type_node.contin_type,
}


def _infer_signature(tr: combo_tree) -> TypeSignature:
    """Not a single C++ function -- the narrow substitute for
    `infer_type_tree` (type_tree.cc:1416) this module's own docstring
    explains (see its "missing general type-inference engine" section for
    what it agrees with classic on, measured, and the two disclosed
    differences).

    Input types come from `_BUILTIN_INPUT_TYPE[parent]` for each
    `argument` node's parent. An index that appears nowhere keeps
    `unknown_type`, which is what classic's real unifier leaves there too
    (`or($1 !$3)` -> `->(boolean unknown boolean boolean)`); an index used
    at two different types becomes `ill_formed_type`, which is how
    classic reports the same mismatch (`+(impulse($1) $1)` ->
    `->(ill_formed ill_formed)`).
    """
    output_type, arity = _infer_output_type_and_arity(tr)
    itypes: list[type_node] = [type_node.unknown_type] * arity
    for node in tr:
        v = node.data
        if not is_argument(v):
            continue
        assert isinstance(v, argument)
        parent = node.parent
        if parent is None or not is_builtin(parent.data):
            # A bare `$n` at the root has no enclosing expression to pin
            # its type down; classic leaves it `unknown` too (`$1` ->
            # `->(unknown unknown)`), so this is not an error.
            continue
        if parent.data not in _BUILTIN_INPUT_TYPE:
            raise NotImplementedError(
                f"_infer_signature: no input-type entry for builtin "
                f"{parent.data!r} (parent of {v!r}); this port has no general "
                "type-unification engine and does not guess -- see the module "
                "docstring's 'missing general type-inference engine' section"
            )
        want = _BUILTIN_INPUT_TYPE[parent.data]
        slot = abs(v.idx) - 1
        if itypes[slot] == type_node.unknown_type:
            itypes[slot] = want
        elif itypes[slot] != want:
            itypes[slot] = type_node.ill_formed_type
    return TypeSignature(itypes, output_type)


def _is_well_formed(sig: TypeSignature) -> bool:
    """C++: moses/comboreduct/type_checker/type_tree.cc:1210 is_well_formed

    "if (tt.empty()) return false; else for (it : tt) if (*it ==
    id::ill_formed_type) return false; return true;" -- over a
    `TypeSignature`, whose only nodes are its input types plus its output
    type. The `tt.empty()` half has no analogue: a `TypeSignature` always
    has an `otype`, and `str_to_combo_tree` never returns an empty tree
    (classic's own `combo_problem::run` would `OC_ASSERT` on one inside
    `OTable`'s constructor long before this).
    """
    return type_node.ill_formed_type not in sig.itypes and sig.otype != type_node.ill_formed_type


def _largest_const_in_tree(tr: combo_tree) -> contin_t:
    """C++: moses/moses/main/demo-problems.cc:347 largest_const_in_tree

    "Get largest contin constant in a combo tree" (demo-problems.cc:346's
    own comment). Note `contin_t rc = 0.0;` and `if (rc < val) rc = val;`
    -- an all-negative tree (or one with no contin constants at all)
    yields `0.0`, NOT the largest negative value and not a "no constants"
    sentinel. `combo_problem::run` divides its sampling range on that
    result, so a `-y` program whose only constants are negative samples
    `[-0.0, 0.0]` -- degenerate, and faithful.
    """
    rc: contin_t = 0.0
    for node in tr:
        v = node.data
        if is_contin(v):
            val = get_contin(v)
            if rc < val:
                rc = val
    return rc


class combo_problem_base(problem_base):
    """C++: moses/moses/main/demo-problems.cc:301 combo_problem_base"""

    def __init__(self, dp: demo_params) -> None:
        self._dparms = dp

    def check_args(self) -> tuple[combo_tree, TypeSignature]:
        """C++: moses/moses/main/demo-problems.cc:310 combo_problem_base::check_args

        Returns `(tr, tt)` instead of C++'s `void` -- `combo_problem::run`
        (demo-problems.cc:373-376) re-parses `_dparms.combo_str` and
        re-infers its type tree independently right after calling this in
        C++; since both computations are pure functions of the same
        `combo_str`, this port does the work once and returns it, a
        non-observable internal restructuring (same class of change
        `representation::get_candidate`'s own docstring already accepts
        elsewhere in this project). Callers take `output_type` from
        `tt.otype` and `arity` from `type_tree_arity(tt)`, exactly as
        `combo_problem::run` does (demo-problems.cc:375-376).

        `logger().warn("Feature selection is not supported...")`
        (demo-problems.cc:312-313) is dropped -- see `bool_problem_base.
        run`'s own docstring for the same, already-established omission.

        The "is_well_formed" check (demo-problems.cc:324-328) now has a
        real, if narrow, engine to call: `_is_well_formed(_infer_signature
        (tr))`. It genuinely fires for a cross-argument type mismatch
        (`-y '+(impulse($1) $1)'`, which classic also rejects -- measured
        against a compiled `infer_type_tree` probe, `->(ill_formed
        ill_formed)`, `wf=0`). The `except AssertionError` arm around it
        catches the OTHER, structural failure -- `get_output_type_tree`
        raising for a vertex kind it does not handle -- and remains
        UNREACHABLE by any string `str_to_combo_tree` can actually parse:
        that parser's grammar only ever produces `builtin`/`argument`/
        `contin_t` vertices (`iostream_combo.py`'s own module docstring),
        and every `builtin` member it can produce has a
        `_BUILTIN_OUTPUT_TYPE` entry except the internal `builtin_count`
        sentinel, which no token string maps to. Kept for structural
        fidelity and covered by `test_demo_problems.py` via a direct
        monkeypatch of `_infer_output_type_and_arity`, not a real
        `combo_str`.

        A genuinely UNPARSEABLE `combo_str` (mismatched parens, e.g.)
        fails earlier still: `str_to_combo_tree` itself raises `ValueError`
        for that (a stricter failure mode than classic's own `stringstream
        >> tr`, which does not throw on malformed input at all -- an
        already-disclosed property of that function, not introduced here)
        -- left unwrapped, so that `ValueError` (with `str_to_combo_tree`'s
        own message) propagates directly out of this method too.
        """
        combo_str = self._dparms.combo_str
        if not combo_str:
            raise ValueError("You must specify a combo tree to learn (option -y).")

        tr = str_to_combo_tree(combo_str)
        try:
            tt = _infer_signature(tr)
        except AssertionError as exc:
            raise ValueError(
                f"The combo tree {combo_tree_to_str(tr)} is not well formed."
            ) from exc
        if not _is_well_formed(tt):
            raise ValueError(f"The combo tree {combo_tree_to_str(tr)} is not well formed.")

        # DIVERGENCE(C++ moses/moses/main/demo-problems.cc:337): classic's
        # OR condition is `-1 == arity || NULL == strchr(combo_str, '$')`.
        # This port's `_infer_output_type_and_arity` can never produce
        # `-1` (module docstring's own "missing general type-inference
        # engine" section explains why), so only the second half is ever
        # reachable here -- still sufficient for every in-scope oracle
        # fixture ("and($1 $2)", "or(and($2 !$3) $1 !$4 $5)",
        # "+(6 *(6 $1))"), which all contain '$' and have a real positive
        # arity either way.
        if "$" not in combo_str:
            raise ValueError(
                f"Error: the combo program {combo_tree_to_str(tr)} appears not "
                "to contain any arguments. Did you forget to escape the $'s in "
                "the shell command line?"
            )
        return tr, tt


class combo_problem(combo_problem_base):
    """C++: moses/moses/main/demo-problems.cc:361 combo_problem"""

    def name(self) -> str:
        return "cp"

    def description(self) -> str:
        return "Demo: Learn a given combo program"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/demo-problems.cc:371 combo_problem::run

        Both branches are implemented: `boolean_type`
        (demo-problems.cc:390-400) and `contin_type`
        (demo-problems.cc:401-442).

        The contin branch's `min`/`max_rand_input` block
        (demo-problems.cc:403-421) is classic's own self-described "stupid
        hack" and is ported verbatim, wart included -- see the inline
        comments and the `# DIVERGENCE` marker on the `ITable` call for
        the ARGUMENT-ORDER swap that is genuinely there in C++.

        The contin branch feeds `tt`'s real INPUT types (not
        `gen_signature(otype, arity)`) to everything downstream, because
        `ITable`'s random-sampling constructor dispatches per column on
        exactly those (table.cc:104-118): `+(6 *(6 impulse($1)))` is
        `->(boolean contin)` in classic, and sampling it as a contin
        column instead would silently produce a different table. The
        boolean branch keeps `gen_signature(boolean_type, arity)` -- see
        its own `# DIVERGENCE` marker for the one measured case where
        that is not the same thing.
        """
        pms = ob
        tr, tt = self.check_args()
        output_type = tt.otype
        arity = type_tree_arity(tt)

        if getattr(pms.meta_params, "do_boosting", False):
            raise NotImplementedError("Boosting not supported for this problem!")

        if not pms.exemplars:
            pms.exemplars = [_type_to_exemplar(output_type)]

        if output_type == type_node.boolean_type:
            # DIVERGENCE(C++ moses/moses/main/demo-problems.cc:396): classic
            # passes `tt` -- `infer_type_tree`'s real signature -- while this
            # branch passes an all-boolean one of the same arity. The two
            # differ only when an argument index is never mentioned:
            # `or($1 !$3)` really is `->(boolean unknown boolean boolean)`
            # in classic (measured against a compiled `infer_type_tree`
            # probe), so `boolean_arity` is 2 there and 3 here.
            # `boolean_arity(_signature)` is read in exactly ONE place in
            # classic, `build_knobs.cc:95`'s `if (5 * boolean_arity >
            # type_tree_arity) _skip_disc_probe = false`, so the difference
            # is observable only when `5 * real_barity <= arity`. It is NOT
            # observable for `or($1 !$3)` (10 > 3 and 15 > 3 take the same
            # branch), and `-Hcp -yor($1 !$3) -m1000 -W1` reproduces classic
            # byte for byte.
            #
            # It IS observable, measured: `-Hcp -y'or($1 !$11)' -m2000 -W1`
            # (barity 2, arity 11, so `10 > 11` is false and classic leaves
            # disc_probe ON) prints 9 lines in classic and 2 here. Both find
            # `0 or($1 !$11)`; the metapopulation tail differs. Note also
            # that the neighbouring `or($1 !$10)` -- the smallest program
            # that reaches that branch -- SEGFAULTS in classic (rc 139,
            # reproduced under `ulimit -v 2097152`), while `!$9` and `!$11`
            # both run clean. Upstream crash; recorded, not chased.
            #
            # Kept rather than fixed because the faithful signature is
            # currently REJECTED downstream: `build_knobs.py`'s
            # `_validate_boolean_exemplar` raises "non-Boolean inputs
            # require their corresponding domain builder slice" for any
            # non-boolean itype, `unknown_type` included, so passing `tt`
            # here turns a working `-Hcp` run into a NotImplementedError.
            # That validator is outside this task's ownership. The contin
            # branch below has no such problem (`_validate_contin_exemplar`
            # accepts an `unknown` slot -- `-Hcp -y'+($1 *($3 $3))'`
            # matches classic).
            sig = gen_signature(type_node.boolean_type, arity)
            bscore = logical_bscore(tr, arity)
            cscore = behave_cscore(bscore)
            metapop_moses_results(
                pms.exemplars,
                sig,
                pms.bool_reduct,
                pms.bool_reduct_rep,
                cscore,
                pms.opt_params,
                pms.hc_params,
                pms.ps_params,
                pms.deme_params,
                pms.filter_params,
                pms.meta_params,
                pms.moses_params,
                pms.printer,
            )
        elif output_type == type_node.contin_type:
            # Naive users of the combo regression mode will fail
            # to understand that the input program must be sampled
            # in order for the fitness function to be evaluated.
            # The default sample range 0<x<1 is probably too small
            # for any fancy-pants input program, so try to make
            # a reasonable guess.  Yes, this is a stupid hack, but
            # it does avoid the problem of naive users saying
            # "aww moses sucks" when they fail to invoke it correctly.
            if pms.min_rand_input == 0.0 and pms.max_rand_input == 1.0:
                pms.max_rand_input = 2.0 * _largest_const_in_tree(tr)
                pms.min_rand_input = -pms.max_rand_input
                if pms.nsamples <= 0 and pms.default_nsamples < 2 * arity * pms.max_rand_input:
                    # `pms.nsamples` is an `int` member and the right-hand
                    # side is a `contin_t`: C++ truncates toward zero on
                    # assignment, which is `int()`, not `round()`.
                    pms.nsamples = int(2 * arity * pms.max_rand_input)

            if pms.nsamples <= 0:
                pms.nsamples = pms.default_nsamples

            # logger().info() << "Will sample combo program ..." is dropped,
            # like every other logger() call in this port.

            # DIVERGENCE(C++ moses/moses/main/demo-problems.cc:429): classic
            # writes `ITable it(tt, pms.nsamples, pms.max_rand_input,
            # pms.min_rand_input);` and the constructor's parameters are
            # `(..., contin_t min_contin, contin_t max_contin)`
            # (table.h:761-762) -- so MAX is passed as MIN and vice versa.
            # Every drawn value is therefore `(min_rand - max_rand) * r +
            # max_rand`, i.e. the intended range traversed BACKWARDS. This
            # is invisible whenever the range is symmetric (which the
            # "stupid hack" above makes it), and observable the moment it
            # is not: `-q0 -w10` samples `10 - 10r`, not `10r`. Reproduced
            # verbatim, argument for argument, rather than "fixed" -- and
            # pinned by an asymmetric-range oracle in
            # tests/moses/main/test_moses_exec.py.
            it = ITable(tt, pms.nsamples, pms.max_rand_input, pms.min_rand_input)
            ot = OTable(tr, it)

            as_ = _alphabet_size(tt, pms.ignore_ops)

            bscore = contin_bscore(ot, it)
            _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)
            cscore = behave_cscore(bscore)
            metapop_moses_results(
                pms.exemplars,
                tt,
                pms.contin_reduct,
                pms.contin_reduct,
                cscore,
                pms.opt_params,
                pms.hc_params,
                pms.ps_params,
                pms.deme_params,
                pms.filter_params,
                pms.meta_params,
                pms.moses_params,
                pms.printer,
            )
        else:
            raise ValueError(f"combo_problem: type {output_type!r} not supported.")


class ann_combo_problem(combo_problem_base):
    """C++: moses/moses/main/demo-problems.cc:451 ann_combo_problem

    Raising stub -- see module docstring's own "`sr` and `cp`'s contin
    arm" section. Does not call `check_args()` first (unlike classic, which
    validates the combo string before doing ANN-specific work): ANN
    regression is unsupported regardless of whether `-y`'s argument would
    otherwise validate, so validating it first would only mask the real
    reason `-H ann-cp` fails behind an unrelated, possibly-misleading
    `ValueError` about `-y` -- raising unconditionally here is more
    honest about which gap actually blocks this problem.
    """

    def name(self) -> str:
        return "ann-cp"

    def description(self) -> str:
        return "Demo: Learn a given combo program using ANN"

    def run(self, ob: object) -> None:
        del ob
        raise NotImplementedError(
            "ann_combo_problem ('ann-cp', demo-problems.cc:451): needs the ANN "
            "combo-vocabulary path CLAUDE.md's own 'Known sharp edges' section "
            "already flags as stale/legacy (tree_transform().decodify_tree, "
            "ann::feedforward_depth, ann::propagate) -- -H ann-cp is "
            "unsupported. The contin/ITable/OTable/contin_bscore half of what "
            "this problem needs DOES now exist; the ANN half does not"
        )


# ---------------------------------------------------------------------------
# register_demo_problems -- demo-problems.cc:500-512.
# ---------------------------------------------------------------------------


def register_demo_problems(pmr: problem_manager) -> demo_params:
    """C++: moses/moses/main/demo-problems.cc:500 register_demo_problems

    `void register_demo_problems(problem_manager&, option_manager&)` in
    C++; the `option_manager&` parameter is dropped (nothing to register
    CLI options against, see module docstring) and the freshly built
    `demo_params` is RETURNED instead (no C++ equivalent -- see module
    docstring's own "`demo_params`" section for why a caller needs it).
    """
    dp = demo_params()
    pmr.register_problem(pa_problem(dp))
    pmr.register_problem(dj_problem(dp))
    pmr.register_problem(majority_problem(dp))
    pmr.register_problem(mux_problem(dp))
    pmr.register_problem(polynomial_problem(dp))
    pmr.register_problem(combo_problem(dp))
    pmr.register_problem(ann_combo_problem(dp))
    return dp
