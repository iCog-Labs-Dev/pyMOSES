"""C++: moses/moses/main/demo-problems.{h,cc}

The toy/demo problems selected by `-H pa|dj|maj|mux|sr|cp|ann-cp`:
`pa` (even parity), `dj` (logical disjunction), `maj` (majority), `mux`
(multiplexer), `sr` (polynomial regression), `cp` (learn a given combo
program) and `ann-cp` (the ANN variant of `cp`). Ported per the S13 P2
"demo problems" task brief: unblocks the seven `MOSESUTest` cases that
have been stuck since S11 on two missing dependencies -- this file itself,
and `logical_bscore` (bscores.h:52) -- both of which `pymetta/moses/main/
problem.py`'s own module docstring already names as the reason
`test_2_parity`, `test_2_parity_with_labels`, `test_3_parity_multi_thread`
(`-Hpa`), `test_1_mux`, `test_2_mux` (`-Hmux`) and `test_cp_simple`,
`test_cp` (`-Hcp`) are unreachable, and as the reason `pymetta/moses/moses/
moses_main.py`'s own test suite could not exercise them either.

## Where `logical_bscore` lives, and why

`bscores.h:52`'s `logical_bscore` is used ONLY by this file in classic
(verified: `git grep -n logical_bscore` inside `$CLASSIC` returns exactly
11 hits, all in `demo-problems.cc` or `bscores.{h,cc}` itself -- no other
`.cc` file references it). `pymetta/moses/scoring/bscores.py` is owned,
for this task, by a concurrent agent porting `ctruth_table_bscore`, and
this task's own brief says explicitly: put `logical_bscore` in this
module instead of editing that file, and say so here. Done: `logical_bscore`
below is a complete, independent port of `bscores.h:52-79` and
`bscores.cc:64-134`, needing nothing from `bscores.py` (it does not share
a base-class *file* with `ctruth_table_bscore`, only the `bscore_base`
class both extend, from `scoring_base.py`, which this task does not touch
either). Anyone completing a from-scratch `pymetta/moses/scoring/
bscores.py` port later should either import `logical_bscore` from here, or
delete this copy and update this module's import -- this docstring is the
seam.

## Small dependencies this task also had to supply, and why they live here

Three more C++ symbols this file's own logic needs turned out not to be
ported ANYWHERE in this project as of this task (grepped, not assumed):

* `alphabet_size` (problem.cc:138-196). `pymetta/moses/main/problem.py`'s
  own module docstring claims this helper is "used only from
  `table-problems.cc`'s concrete `*_table_problem::run()` bodies" and is
  therefore out of scope for that file -- but `demo-problems.cc:112`
  (`bool_problem_base::run`) calls it too, so that claim is incomplete
  (not something this task's brief asked it to fix; noted here because it
  is directly why this port needs its own copy). `problem.py` is not
  owned by either of this task's two concurrent sibling agents (only
  `problem_params.py` and the whole `pymetta/moses/scoring/` tree are),
  but it is a stable, already-committed S13 Task 0 deliverable this task
  was not asked to touch either -- so `_alphabet_size` below is a private,
  local copy scoped to exactly what `bool_problem_base`/`combo_problem`
  need, not a change to that file.
* `type_to_exemplar` (representation.cc:75-92, declared representation.h:
  42). Grepped: no Python port exists. `_type_to_exemplar` below ports
  only the two branches this file's in-scope problems can ever reach
  (`boolean_type`, `contin_type`) -- see its own docstring.
* `even_parity`/`disjunction`/`multiplex`/`majority` (moses/example-progs/
  scoring_iterators.h) -- plain iterator-based functors with no Python
  port anywhere. Reimplemented below as ordinary functions/callables over
  a plain `list[bool]`, since that is all `logical_bscore`'s functional
  constructor ever feeds them (see `_iterator_function_truth_table`).

## `sr` and `ann-cp`: registered, but as raising stubs

Per this task's own brief: `sr` (`polynomial_problem`, demo-problems.cc:
233-285) builds an `ITable`/`OTable` and a `contin_bscore` over a
`contin_type` signature -- S14 (continuous) territory this port has not
built (`problem_params.contin_reduct` is still `None`, per
`problem_params.py:799`). `ann-cp` (`ann_combo_problem`, demo-problems.cc:
451-496) needs the same contin/ITable/OTable/`contin_bscore` machinery
PLUS the stale ANN combo-vocabulary path CLAUDE.md's own "Known sharp
edges" section already flags as legacy. Both are registered here (so a
`-H sr` / `-H ann-cp` lookup succeeds, matching classic's own successful
`find_problem`) with the SAME name()/description() strings as their real
C++ classes, and a `run()` that raises `NotImplementedError` immediately
-- the same "port the gate, raise inside it" discipline `problem.py`'s own
`_unported_table_problem` already established for the nine unreachable
`table-problems.cc` problems.

## `cp`'s own contin branch: also a raising stub, for the same reason

`combo_problem` (`cp`) is genuinely in scope -- but only for a
BOOLEAN-output combo program. Its `contin_type` branch (demo-problems.cc:
401-442) needs exactly the same un-ported `ITable`/`OTable`/`contin_bscore`
trio `sr` does, so it raises too. Both `MOSESUTest` cases this task can
actually reach (`test_cp_simple`: `"and($1 $2)"`, `test_cp`:
`"or(and($2 !$3) $1 !$4 $5)"`) are boolean-output programs, so this
narrowing costs nothing against this task's own oracle list; `test_cp_contin`
(`"+(6 *(6 $1))"`) is already named S14 in `docs/CARRY-FORWARD.md`'s own
`MOSESUTest` classification and stays blocked for that reason, not a new
one this file introduces.

## The missing general type-inference engine, and the narrow substitute

`combo_problem`/`combo_problem_base` need `infer_type_tree` (a whole-tree
type-unification pass, type_tree.cc, ~1468 lines) to determine a parsed
`-y` combo program's output type, arity, and well-formedness
(demo-problems.cc:310-344, 376-380). `pymetta/comboreduct/type_checker/
type_tree.py`'s own module docstring is explicit that this general engine
is NOT ported and is out of scope: only `get_output_type_tree`, a
SINGLE-VERTEX lookup, exists. `_infer_output_type_and_arity` below is a
narrow, disclosed substitute sufficient for exactly what `combo_problem`
consults -- the ROOT vertex's output type (via the one function that IS
ported) and the highest argument index anywhere in the tree (a plain
tree walk, needing no type unification at all since combo arguments are
untyped placeholders whose OWN type is never inspected by anything this
file does). This is NOT a full is_well_formed/infer_type_tree port:

* It cannot detect a genuine cross-argument type mismatch (e.g. an
  argument used once in boolean position and once in contin position) --
  classic's real type checker would catch that and report "not well
  formed"; this substitute has no such check at all and would happily
  return whatever the root vertex's own type is. No demo problem in this
  task's scope constructs such a program (both reachable `cp` fixtures
  are single-typed boolean expressions), so this is a disclosed, inert
  gap, not a silently "fixed" one.
* It can never produce `type_tree_arity`'s `-1` (ill-formed) sentinel the
  way real `infer_type_tree` can -- there is no failure mode in this
  substitute that yields it. `_check_combo_args` below therefore relies
  entirely on the OTHER half of demo-problems.cc:337's OR condition (no
  literal `$` in the raw `-y` string) to catch the "forgot to escape the
  $'s in the shell" case; see that function's own docstring.

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

from pymetta.comboreduct.combo.argument import argument
from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymetta.comboreduct.combo.vertex import builtin, combo_tree, is_argument
from pymetta.comboreduct.table.table import complete_truth_table
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    gen_signature,
    type_node,
    type_tree_arity,
)
from pymetta.comboreduct.type_checker.type_tree import get_output_type_tree
from pymetta.moses.main.problem import problem_base, problem_manager
from pymetta.moses.moses.moses_main import metapop_moses_results
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.scoring_base import bscore_base

__all__ = [
    "combo_problem",
    "demo_params",
    "dj_problem",
    "logical_bscore",
    "majority_problem",
    "mux_problem",
    "pa_problem",
    "register_demo_problems",
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

    Only the branches this file's in-scope problems can reach
    (`boolean_type`, always; `contin_type`/`ann_type`, ported for
    fidelity though nothing in this file's reachable scope calls them
    today -- `cp`'s own `contin_type` branch raises before ever
    constructing a signature to pass here) are real implementations.
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
    if output_type == type_node.boolean_type:
        # 3 operators: and, or, not
        return 3 + arity
    if output_type == type_node.contin_type:
        return 8 + arity - len(ignore_ops)
    if output_type == type_node.ann_type:
        return 2 + arity * arity
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
    `boolean_type` (every `bool_problem_base` subclass, always) and
    `contin_type` (`combo_problem`'s own contin branch would request it,
    though that branch raises before this default would ever run for
    `cp` today -- `sr`/`ann-cp` also raise before reaching it). `enum_type`
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
# moses/moses/example-progs/scoring_iterators.h. Not ported anywhere else in
# this project (grepped) -- reimplemented here over a plain `list[bool]`,
# which is all `_iterator_function_truth_table` ever feeds them (unlike
# `complete_truth_table`'s own tree-based `_populate`, table.py, which
# evaluates a combo_tree through `boolean_interpreter` and therefore deals in
# builtin-encoded vertices -- these four operate on raw Python `bool`,
# exactly like their C++ originals operate on raw `bool` iterators, never on
# `combo::builtin`).
# ---------------------------------------------------------------------------


def even_parity(bits: Sequence[bool]) -> bool:
    """C++: moses/moses/example-progs/scoring_iterators.h:44 even_parity

    "even_parity(x1, ..., xn) = true iff (int)x1 + ... + (int)xn is even"
    (scoring_iterators.h:42-43's own comment) -- `parity ^= b` over every
    bit, seeded `True`, is exactly `bool(count_of_true_bits % 2 == 0)`.
    """
    parity = True
    for b in bits:
        parity ^= bool(b)
    return parity


def disjunction(bits: Sequence[bool]) -> bool:
    """C++: moses/moses/example-progs/scoring_iterators.h:56 disjunction

    "disjunction(x1, ..., xn) = true iff there exists i such that xi is
    true" (scoring_iterators.h:55's own comment).
    """
    return any(bits)


class multiplex:
    """C++: moses/moses/example-progs/scoring_iterators.h:68 multiplex

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
    """C++: moses/moses/example-progs/scoring_iterators.h:94 majority

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
    scored_combo_tree_set&)` overload, bscores.cc:85-113) is likewise not
    overridden here, for the same reason `ctruth_table_bscore` does not
    override it either -- falling back to `bscore_base.call_ensemble`'s
    own unconditional raise (scoring_base.py).
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

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:67 logical_bscore::operator()

        "Compare the predictions of the tree to that of the desired
        result. A correct prediction gets a score of 0, an incorrect
        prediction gets a score of -1." (bscores.cc:70-71's own comment,
        typo in the original ("prdiction") not reproduced.)
        """
        tt = complete_truth_table(tr, self._arity)
        return [-_to_score_t(b1 != b2) for b1, b2 in zip(tt, self._target, strict=True)]

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

        `logger().warn("Feature selection is not supported...")`
        (demo-problems.cc:100) is dropped, not ported -- no logging
        surface exists anywhere in this port (`scoring_base.py`'s own
        docstring notes the same omission for a `logger().info()` call).
        """
        pms = ob
        if not pms.exemplars:
            pms.exemplars = [_type_to_exemplar(type_node.boolean_type)]

        bscore = self.get_bscore(self._dparms.problem_size)
        if getattr(pms.meta_params, "do_boosting", False):
            raise NotImplementedError(
                "bool_problem_base: --boost needs logical_bscore."
                "use_weighted_scores()/reset_weights(), which are boosting-"
                "weight infrastructure out of scope for this whole port "
                "cycle (scoring_base.py's own module docstring) -- this is "
                "why test_4_parity_boosting/test_5_parity_boosting stay "
                "blocked (docs/CARRY-FORWARD.md's own MOSESUTest classification)"
            )

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
# sr -- demo-problems.cc:224-285. Raising stub; see module docstring.
# ---------------------------------------------------------------------------


class polynomial_problem(problem_base):
    """C++: moses/moses/main/demo-problems.cc:233 polynomial_problem

    Registered (so `-H sr` resolves via `find_problem`, matching real
    C++) but `run()` raises immediately -- see module docstring's own
    "`sr` and `ann-cp`" section for why: `contin_bscore`/`ITable`/`OTable`
    are S14, not built here (`problem_params.contin_reduct` is still
    `None`).
    """

    def __init__(self, dp: demo_params) -> None:
        self._dparms = dp

    def name(self) -> str:
        return "sr"

    def description(self) -> str:
        return "Simple regression of f_n(x) = sum_{k={1,n}} x^k"

    def run(self, ob: object) -> None:
        del ob
        raise NotImplementedError(
            "polynomial_problem ('sr', demo-problems.cc:233): contin regression "
            "needs ITable/OTable/contin_bscore, all S14 territory this port has "
            "not built (problem_params.contin_reduct is still None) -- -H sr is "
            "unsupported; docs/CARRY-FORWARD.md's own MOSESUTest classification "
            "already lists test_sr_3 as S14"
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


class combo_problem_base(problem_base):
    """C++: moses/moses/main/demo-problems.cc:301 combo_problem_base"""

    def __init__(self, dp: demo_params) -> None:
        self._dparms = dp

    def check_args(self) -> tuple[combo_tree, type_node, int]:
        """C++: moses/moses/main/demo-problems.cc:310 combo_problem_base::check_args

        Returns `(tr, output_type, arity)` instead of C++'s `void` --
        `combo_problem::run` (demo-problems.cc:371-380) re-parses
        `_dparms.combo_str` and re-infers its type tree independently
        right after calling this in C++; since both computations are pure
        functions of the same `combo_str`, this port does the work once
        and returns it, a non-observable internal restructuring (same
        class of change `representation::get_candidate`'s own docstring
        already accepts elsewhere in this project).

        `logger().warn("Feature selection is not supported...")`
        (demo-problems.cc:312-313) is dropped -- see `bool_problem_base.
        run`'s own docstring for the same, already-established omission.

        The "is_well_formed" check (demo-problems.cc:324-328) has no
        general engine to call (module docstring) -- approximated here as
        "the root vertex's output type could be determined at all"
        (`get_output_type_tree` raises `AssertionError` for a vertex kind
        it does not handle); see module docstring for exactly what this
        does and does not catch. In practice this `except` clause is
        CURRENTLY UNREACHABLE by any string `str_to_combo_tree` can
        actually parse: that parser's grammar only ever produces
        `builtin`/`argument`/`contin_t` vertices (`iostream_combo.py`'s
        own module docstring), and every `builtin` member it can produce
        has a `_BUILTIN_OUTPUT_TYPE` entry except the internal
        `builtin_count` sentinel, which no token string maps to (checked
        directly: `builtin_count` is the only member missing from that
        table). Kept for structural fidelity (a real is_well_formed
        engine, or a widened grammar, could make it reachable) and
        covered by `test_demo_problems.py` via a direct monkeypatch of
        `_infer_output_type_and_arity`, not a real `combo_str`.

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
            output_type, arity = _infer_output_type_and_arity(tr)
        except AssertionError as exc:
            raise ValueError(
                f"The combo tree {combo_tree_to_str(tr)} is not well formed."
            ) from exc

        # DIVERGENCE(C++ moses/moses/main/demo-problems.cc:337): classic's
        # OR condition is `-1 == arity || NULL == strchr(combo_str, '$')`.
        # This port's `_infer_output_type_and_arity` can never produce
        # `-1` (module docstring's own "missing general type-inference
        # engine" section explains why), so only the second half is ever
        # reachable here -- still sufficient for both in-scope oracle
        # fixtures ("and($1 $2)", "or(and($2 !$3) $1 !$4 $5)"), which both
        # contain '$' and have a real positive arity either way.
        if "$" not in combo_str:
            raise ValueError(
                f"Error: the combo program {combo_tree_to_str(tr)} appears not "
                "to contain any arguments. Did you forget to escape the $'s in "
                "the shell command line?"
            )
        return tr, output_type, arity


class combo_problem(combo_problem_base):
    """C++: moses/moses/main/demo-problems.cc:361 combo_problem"""

    def name(self) -> str:
        return "cp"

    def description(self) -> str:
        return "Demo: Learn a given combo program"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/demo-problems.cc:371 combo_problem::run

        Only the `boolean_type` branch (demo-problems.cc:390-400) is
        implemented; the `contin_type` branch (demo-problems.cc:401-442)
        raises -- see module docstring's own "`cp`'s own contin branch"
        section.
        """
        pms = ob
        tr, output_type, arity = self.check_args()

        if getattr(pms.meta_params, "do_boosting", False):
            raise NotImplementedError("Boosting not supported for this problem!")

        if not pms.exemplars:
            pms.exemplars = [_type_to_exemplar(output_type)]

        if output_type == type_node.boolean_type:
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
                pms.deme_params,
                pms.filter_params,
                pms.meta_params,
                pms.moses_params,
                pms.printer,
            )
        elif output_type == type_node.contin_type:
            raise NotImplementedError(
                "combo_problem ('cp'): a contin-output combo program needs "
                "ITable/OTable/contin_bscore, all S14 territory this port has "
                "not built (problem_params.contin_reduct is still None) -- "
                "-H cp -y '<contin program>' is unsupported; boolean-output "
                "combo programs (e.g. 'and($1 $2)') are fully supported. "
                "docs/CARRY-FORWARD.md's own MOSESUTest classification already "
                "lists test_cp_contin as S14"
            )
        else:
            raise ValueError(f"combo_problem: type {output_type!r} not supported.")


class ann_combo_problem(combo_problem_base):
    """C++: moses/moses/main/demo-problems.cc:451 ann_combo_problem

    Raising stub -- see module docstring's own "`sr` and `ann-cp`"
    section. Does not call `check_args()` first (unlike classic, which
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
            "ann_combo_problem ('ann-cp', demo-problems.cc:451): needs "
            "ITable/OTable/contin_bscore (S14, not built here) plus the ANN "
            "combo-vocabulary path CLAUDE.md's own 'Known sharp edges' section "
            "already flags as stale/legacy -- -H ann-cp is unsupported"
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
