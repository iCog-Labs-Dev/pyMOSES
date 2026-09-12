"""C++: moses/moses/scoring/bscores.h and bscores.cc

`ctruth_table_bscore` -- boolean classification scorer over a compressed
truth table. Ports exactly `bscores.h:360-383` (the class declaration) and
its out-of-line members at `bscores.cc:276-424`, per task-6-brief.md's
Reference line. Boolean scoring via `ctruth_table_bscore` was the entire
ported surface for that task (see the plan's "Known gap, deliberate" note
and that task's own Context section on `CTable.get_output_type()`); two
more classes have since been added to this module by later tasks
(`contin_bscore`, `cluster_bscore` -- both described below).

Still NOT ported from bscores.h, as of this module's latest revision:
`logical_bscore` (bscores.h:52), `discretize_contin_bscore` (bscores.h:91
-- the `-H it --discretize-threshold` arm, which `table_problems.py`
raises on), the enum family `enum_table_bscore`/`enum_filter_bscore`/
`enum_graded_bscore`/`enum_effective_bscore` (bscores.h:409,446,490,516)
and `interesting_predicate_bscore` (bscores.h:553, the `-H ip` problem's
scorer). The precision/multi-objective
scorers are not in this file at all: they live in the neighbouring
`precision_bscore.py`/`discriminating_bscore.py`/`select_bscore.py`, and
ARE ported there.

Regarding the sign/weighting convention this scorer relies on:
`ctruth_table_bscore::operator()`'s doc comment at the top of bscores.h
(reproduced where `__call__` below cites it) explains why each bscore
element is the NEGATIVE count of a row's WRONG answers, and why
`best_possible_bscore`/`worst_possible_bscore`/`min_improv` all read off
of `fmin`/`fabs` over the same `{logical_true: count, logical_false:
count}` pair per compressed row.

`contin_bscore` (S14 Task 3) -- continuous-regression scorer over an
(uncompressed) `Table`/`OTable`+`ITable` pair. Ports `bscores.h:181-224`
(the class declaration, including its own `err_function_type` nested enum
and `init`) and its out-of-line members at `bscores.cc:140-190`
(`operator()`, `best_possible_bscore`, `min_improv`,
`set_complexity_coef`). `contin_complexity_coef` (scoring_base.h:53,
scoring_base.cc:45) is ported here too, as a private module-level
function, rather than in scoring_base.py: it has exactly one caller
anywhere in classic (`contin_bscore::set_complexity_coef`,
bscores.cc:184-185; confirmed by `git grep contin_complexity_coef` over
the whole classic tree) and scoring_base.py explicitly defers it as out of
its own task's scope (see that module's docstring) -- porting it there
would mean editing a file this task does not own for a function nothing
else needs.

Three C++ constructor overloads (bscores.h:203-224) merge into one Python
`__init__`, dispatched by argument TYPE (matching C++'s own static
overload resolution) rather than argument count, since the (Table, eft)
and (OTable, ITable) shapes are both two-positional-argument calls:
`Table` first argument -> the `contin_bscore(const Table&, eft)` overload
(bscores.h:219-224, the one `table-problems.cc`'s `it` problem actually
uses on `trainingRegression.dat`); `OTable` first argument -> `contin_bscore
(const OTable&, const ITable&, eft)` (bscores.h:211-217); otherwise (a
plain callable) -> the templated `contin_bscore(const Scoring&, const
ITable&, eft)` (bscores.h:203-209), which itself constructs its `target`
via `OTable`'s own templated `Func`-taking constructor
(table.h, `OTable(const Func& f, const ITable& it, ...)` --
`for (row : it) push_back(f(row.begin(), row.end()));`). This port
collapses that two-iterator call into a single ``score(row)`` call taking
the whole row as a `Sequence[vertex]` -- the same iterator-pair-to-Sequence
substitution already established for `roulette_select`
(`docs/ALIGNMENT.md`'s entry for `util/selection.py:28`) -- built directly
from `OTable`'s existing plain-values constructor (`OTable(values, label)`,
table.py) with no change to table.py itself. This first overload has ZERO
real callers within this port's scope (`demo-problems.cc`'s `sr` problem
is the only classic caller, and `pymetta/moses/main/` is out of scope for
this task) -- ported for fidelity to the full declared class, and
covered by a direct (non-demo-problem) unit test, not claimed as
exercising any wired problem.

`cluster_bscore` (S13 `-H cluster`) -- the scorer behind
`cluster_table_problem` (table-problems.cc:551), ported from
`bscores.h:613-621` (the class declaration) and its out-of-line members at
`bscores.cc:932`, `:937`, `:1074`, `:1079`. A one-dimensional k-means
(Lloyd's algorithm) over the single column the candidate tree produces
from the input table, scoring a tree by how tightly that column clusters.
It carries three upstream defects that classic's own printed scores depend
on, all marked DO NOT FIX on the class -- see its docstrings. It also
needs `OTable(tr, itable)` (table.cc:319), a constructor this port's
`OTable` does not have; the non-ANN half of that constructor's body is
written out inline in `__call__`, exactly as `contin_bscore.__call__`
already writes out its own per-row `interpreter_visitor` loop, so
`table.py` needs no change.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from enum import IntEnum

from pymetta.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    combo_tree,
    get_contin,
    negate_vertex,
    vertex,
    vertex_equal,
)
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import (
    CTable,
    ITable,
    OTable,
    Table,
    timed_items_of,
    type_node,
)
from pymetta.feature_selection.scorers.kld import KLDS
from pymetta.moses.moses.types import (
    _to_score_t,
    behavioral_score,
    ostream_behavioral_score,
    score_t,
    very_best_score,
    very_worst_score,
)
from pymetta.moses.scoring.scoring_base import (
    _copy_ctable,
    _cpp_reciprocal,
    _ieee_div,
    _to_usize,
    bscore_base,
    bscore_ctable_base,
    log_candidate_bscore,
)
from pymetta.util.logger import logger, ostream_double


class ctruth_table_bscore(bscore_ctable_base):
    """C++: moses/moses/scoring/bscores.h:360 ctruth_table_bscore

    Extends `bscore_ctable_base` (scoring_base.h:236), matching bscores.h:
    360 exactly. It used to extend `bscore_base` directly, per
    task-6-brief.md's interface line, on the stated grounds that
    `bscore_ctable_base`'s row/feature subsetting was out of scope; every
    `_wrk_ctable` in the C++ member functions below then read as a single
    private table attribute. That is no longer tenable:
    `--ss-n-subsample-demes>1` and `--ss-n-subsample-fitnesses>1` both
    drive `bscore_ctable_base.ignore_rows` (scoring_base.py) on the *it*
    problem's boolean scorer -- i.e. on THIS class -- so the
    `_orig_ctable`/`_wrk_ctable`/`_all_rows_wrk_ctable` distinction is
    load-bearing here, and every member below reads `self._wrk_ctable`
    exactly where C++ does.

    Consequence worth naming: `_wrk_ctable` is a VALUE COPY of the
    caller's table (`_copy_ctable`, scoring_base.py), where this class
    used to alias it. That matches C++ (`_wrk_ctable(_orig_ctable)` is a
    `std::map` copy-construction, scoring_base.cc:188) and is what lets
    `ignore_rows` restore the full table between subsamples;
    `get_ctable()` still hands back `_orig_ctable`, the caller's own
    object, exactly as scoring_base.cc:323 does.
    """

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/bscores.h:362 ctruth_table_bscore

        `bscore_ctable_base(ctt)` (bscores.h:363) runs first and does the
        `_orig_ctable`/`_wrk_ctable`/`_all_rows_wrk_ctable`/
        `_ctable_usize`/`_ctable_weight` work; this body then re-does
        `_size = _wrk_ctable.size()` (bscores.h:365) -- the compressed-row
        count, i.e. `dict.__len__`, not `uncompressed_size()` -- exactly
        as C++ does, redundantly, since the base already set the same
        value from the same table.

        `reset_weights()` (bscores.h:366) IS called now. It used to be
        skipped here as "boosting-only infrastructure this port does not
        implement"; `bscore_base.reset_weights` (scoring_base.py) is real,
        and with `_return_weighted_score` false -- which it always is for
        this class, nothing ever calls `use_weighted_scores()` on it --
        its whole body is `self._weights = []`, so calling it is both
        faithful and observationally inert. Verified inert by mutation:
        removing the call again changes no test's outcome and no CLI
        byte.

        # DIVERGENCE(C++ moses/moses/scoring/bscores.h:362-368): classic's
        # constructor never calls `get_output_type()`, it simply assumes a
        # boolean-output CTable, silently misbehaving otherwise
        # (`counter.get(id::logical_true)`/`get(id::logical_false)` against
        # a non-boolean counter just always returns the "not present"
        # default -- i.e. classic would silently return an all-zero
        # bscore rather than raising). This port raises `ValueError`
        # instead, added here as a deliberate, disclosed strengthening --
        # and, per task-6-brief.md's Context section, to be the FIRST real
        # consumer of `CTable.get_signature()`/`get_output_type()`
        # (table.py -- no earlier task's tests ever called a compressed
        # table's `.get_output_type()`), so as to answer its question:
        # does the "signature landmine" (a weighted `Table.compressed()`'s
        # `CTable` carrying an un-trimmed signature next to trimmed
        # `ilabels`, table.py's `Table.compressed` docstring) actually
        # fire here? It does not: `get_output_type()` reads only
        # `self._signature.otype`, wholly unaffected by any
        # `itypes`/`ilabels` LENGTH mismatch, and this class never calls
        # `CTable.filtered()` (the only method with a length cross-check
        # -- and even there, this port's own `filtered()` has no such
        # cross-check to begin with; see table.py). Verified directly
        # against tests/fixtures/table_data/weighted.csv (loaded with a
        # weighting feature, so the mismatch is actually present) in
        # tests/moses/scoring/test_bscores.py.
        """
        super().__init__(ctable)
        self._size = len(self._wrk_ctable)
        self.reset_weights()
        output_type = ctable.get_output_type()
        if output_type != type_node.boolean_type:
            raise ValueError(
                f"ctruth_table_bscore requires a boolean-output CTable, "
                f"got output type {output_type.name!r}"
            )
        self._best_possible_score: behavioral_score = []
        self.set_best_possible_bscore()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:276 ctruth_table_bscore::operator()

        For each compressed row, evaluate `tr` over that row's inputs,
        NEGATE the predicted answer, and look up HOW MANY actual
        observations at that row hold the negated (i.e. WRONG) answer --
        that count, itself negated again (so a perfect row scores 0.0 and
        a row with wrong answers scores negative), is the row's bscore
        element. "By convention, we expect score to be negative (so that
        higher scores==better scores)" (types.py's composite_score
        docstring, quoting types.h verbatim) is exactly this convention.

        `interpreter_visitor(tr)` is constructed once per call (matching
        C++'s `interpreter_visitor iv(tr); auto interpret_tr =
        boost::apply_visitor(iv);`, done once before the per-row loop,
        not once per row) and applied to each row's input tuple, coerced
        to `list` since `interpreter_visitor.__call__` requires a `list`
        (interpreter.py) where a CTable row key is a `tuple`.
        """
        iv = interpreter_visitor(tr)
        bs: behavioral_score = []
        for row, counter in self._wrk_ctable.items():
            sc = _to_score_t(counter.get(negate_vertex(iv(list(row))), 0.0))
            bs.append(-sc)

        if self._return_weighted_score:  # if boosting
            # "Report the score only relative to the best-possible
            # score." (bscores.cc:290's own comment.) Unreachable in
            # this port -- see scoring_base.py's module docstring -- kept
            # for fidelity to bscores.cc:289-292.
            bs = [a - b for a, b in zip(bs, self._best_possible_score, strict=True)]

        log_candidate_bscore(tr, bs)  # C++: bscores.cc:294
        return bs

    def set_best_possible_bscore(self) -> None:
        """C++: moses/moses/scoring/bscores.cc:350 ctruth_table_bscore::set_best_possible_bscore

        "CTable is a compressed table; different rows may have identical
        inputs, differing only in output. Clearly, in such a case, both
        outputs cannot be simultaneously satisfied, but we can try to
        satisfy the one of which there is more (the max). Thus, we take
        the fmin of the two possibilities as the number of wrong answers
        we are doomed to get. fmin is used, because the rows may be
        weighted by fractional values." (bscores.cc:356-368's own comment,
        reproduced verbatim.) Python's builtin `min` over two floats is
        `fmin`-equivalent for this port's corpus (neither operand is ever
        NaN).
        """
        self._best_possible_score = [
            -_to_score_t(
                min(counter.get(builtin.logical_true, 0.0), counter.get(builtin.logical_false, 0.0))
            )
            for counter in self._wrk_ctable.values()
        ]

        # C++: bscores.cc:373-374, the last statement of the function.
        logger().info(
            "ctruth_table_bscore: Best possible: "
            + ostream_behavioral_score(self._best_possible_score)
        )

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:377 ctruth_table_bscore::best_possible_bscore

        "The returned best score will always be zero [when boosting],
        because the actual best score is subtracted; this is required to
        get boosting to work." (bscores.cc:379-382's own comment.) The
        `_return_weighted_score` branch is unreachable in this port (see
        `__call__`'s docstring) but kept for fidelity.

        Returns a COPY of `self._best_possible_score`, matching C++'s
        by-value `behavioral_score` return (bscores.h:375 declares this
        `behavioral_score best_possible_bscore() const`, not a reference)
        -- returning the live list itself would let a caller's in-place
        mutation of the result (e.g. the very `bs -= _best_possible_score`
        pattern `__call__` uses when boosting) corrupt this scorer's own
        cached state for every future call.
        """
        if self._return_weighted_score:
            return [0.0] * self._size
        self.set_best_possible_bscore()
        return list(self._best_possible_score)

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:389 ctruth_table_bscore::worst_possible_bscore

        "The most that the score can improve is to flip true to false, or
        v.v. The worst score is to get the majority wrong. This works
        correctly even for weighted tables, where the counts could be
        arbitrary float-point values." (bscores.cc:395-398's own comment.)
        """
        result = []
        for counter in self._wrk_ctable.values():
            t = counter.get(builtin.logical_true, 0.0)
            f = counter.get(builtin.logical_false, 0.0)
            result.append(-_to_score_t(abs(t - f)))
        return result

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:406 ctruth_table_bscore::min_improv

        "A return value of 0.5 would be correct only if all rows had a
        weight of exactly 1.0. Otherwise, we look for the row with the
        smallest (positive) weight, and return that." (bscores.cc:408-410's
        own comment.) `min_weight` is seeded with `very_best_score`
        (types.py) -- this port's bit-exact stand-in for C++'s `FLT_MAX`
        sentinel (bscores.cc:413), so the degenerate case of a CTable
        where every row is split exactly 50/50 between `logical_true` and
        `logical_false` returns the same `0.5 * FLT_MAX` classic does,
        not `math.inf`.
        """
        min_weight = very_best_score
        for counter in self._wrk_ctable.values():
            t = counter.get(builtin.logical_true, 0.0)
            f = counter.get(builtin.logical_false, 0.0)
            w = _to_score_t(abs(t - f))
            if w != 0.0 and w < min_weight:
                min_weight = w
        # `0.5 * min_weight` is computed in double precision in C++ too
        # (0.5 is a double literal; float*double promotes to double) and
        # narrowed only at the return, since `min_improv`'s C++ return
        # type is itself `score_t`.
        return _to_score_t(0.5 * min_weight)


def _sq(x: float) -> float:
    """C++: opencog/util/numeric.h:263 sq

    ``template<typename T> T sq(T x) { return x * x; }`` -- ported directly
    here rather than as a project-wide utility, since `contin_bscore`'s
    ``squared_error`` err_func (below) is its only user anywhere in this
    port's scope.
    """
    return x * x


def _contin_complexity_coef(alphabet_size: int, stdev: score_t) -> score_t:
    """C++: moses/moses/scoring/scoring_base.cc:45 contin_complexity_coef
    (declared scoring_base.h:53)

    "Used to define the complexity scoring component given that stdev is
    the standard deviation of the noise of the [output] we're trying to
    predict" (scoring_base.h:47-49's own doc comment, reproduced verbatim).
    Ported HERE, not in scoring_base.py, which explicitly defers it as out
    of ITS OWN task's scope (see that module's own docstring) -- its only
    caller anywhere in classic is `contin_bscore::set_complexity_coef`
    (bscores.cc:184-185; confirmed by `git grep contin_complexity_coef`
    over the whole classic tree, exactly two matches: the declaration/
    definition and this one call site), so porting it in scoring_base.py
    would mean editing a file this task does not own for a function
    nothing else needs.

    Computed in DOUBLE precision throughout (`log`/`sq` both take and
    return `double` in C++; `alphabet_size` -- `unsigned` -- promotes to
    `double` for `log` the same way any C integer argument does) and
    narrowed to `score_t` exactly once, at THIS function's own C++ return
    (its declared return type). ``stdev`` arrives here already narrowed to
    float32 precision by `contin_bscore.set_complexity_coef`'s own `float`
    parameter (mirrored there via `_to_score_t` before this is called) --
    squaring an already-float32-precision Python `float` in double
    arithmetic reproduces C++ squaring a `float` promoted to `double`
    exactly, since that promotion is lossless.
    """
    return _to_score_t(math.log(alphabet_size) * 2 * _sq(stdev))


class contin_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:181 contin_bscore

    Continuous-regression behavioural scorer: `__call__` evaluates a combo
    tree over each row of `self._cti` (the input `ITable`) and compares
    the result against `self._target`'s (the output `OTable`) same-row
    value via a per-instance error function -- squared or absolute error,
    selected by `err_function_type` (bscores.h:183-186) at construction.
    See the module docstring for how this port's one `__init__` merges
    C++'s three constructor overloads (bscores.h:203-224), dispatched by
    argument TYPE rather than count.
    """

    class err_function_type(IntEnum):
        """C++: moses/moses/scoring/bscores.h:183 contin_bscore::err_function_type

        An unscoped `enum` in C++ (bscores.h:183-186), so
        `contin_bscore::squared_error`/`contin_bscore::abs_error` are
        valid C++ spellings alongside the fully-qualified enum-typed ones
        -- mirrored below as plain class attributes on `contin_bscore`
        itself (`contin_bscore.squared_error`), not only on this nested
        enum, so both spellings work here too.
        """

        squared_error = 0
        abs_error = 1

    squared_error = err_function_type.squared_error
    abs_error = err_function_type.abs_error

    def __init__(
        self,
        first: Table | OTable | Callable[[Sequence[vertex]], vertex],
        second: ITable | None = None,
        eft: contin_bscore.err_function_type = err_function_type.squared_error,
    ) -> None:
        """C++: moses/moses/scoring/bscores.h:203 contin_bscore
        (``Scoring`` functor + `ITable`), moses/moses/scoring/bscores.h:211
        contin_bscore (`OTable` + `ITable`), and
        moses/moses/scoring/bscores.h:219 contin_bscore (`Table`). See the
        module docstring for the overload-dispatch rationale and the
        first overload's iterator-pair-to-Sequence substitution.

        Every branch copies `cti`/`target` (`ITable`/`OTable` are C++
        value types; C++'s member-initializer list copy-constructs both
        `target(...)`/`cti(r)` from a `const&` argument, matching
        `Table.__init__`'s own established copying convention -- see that
        class's docstring) rather than aliasing the caller's table.
        `_size` mirrors each overload's own `_size = r.size()` /
        `_size = table.size()` (bscores.h:207, 216, 223) -- the INPUT row
        count in every case, not `target`'s.
        """
        super().__init__()
        if isinstance(first, Table):
            # contin_bscore(const Table& table, err_function_type eft):
            #     target(table.otable), cti(table.itable)
            if second is not None:
                eft = second
            table = first
            self._target = table.otable.copy()
            self._cti = table.itable.copy()
            self._size = table.size()
        elif isinstance(first, OTable):
            # contin_bscore(const OTable& t, const ITable& r, eft):
            #     target(t), cti(r)
            if not isinstance(second, ITable):
                raise TypeError(
                    "contin_bscore(otable, itable, ...): second argument must be an ITable"
                )
            self._target = first.copy()
            self._cti = second.copy()
            self._size = len(second)
        elif callable(first):
            # contin_bscore(const Scoring& score, const ITable& r, eft):
            #     target(score, r), cti(r)
            if not isinstance(second, ITable):
                raise TypeError(
                    "contin_bscore(score_func, itable, ...): second argument must be an ITable"
                )
            itable = second
            self._target = OTable([first(row) for row in itable])
            self._cti = itable.copy()
            self._size = len(itable)
        else:
            raise TypeError(
                "contin_bscore: first argument must be a Table, an OTable (with an ITable "
                f"second argument), or a callable (with an ITable second argument); "
                f"got {type(first).__name__}"
            )
        self.init(eft)

    def init(self, eft: contin_bscore.err_function_type = err_function_type.squared_error) -> None:
        """C++: moses/moses/scoring/bscores.h:186 contin_bscore::init

        Selects the per-row error function -- `sq(y1 - y2)` for
        `squared_error`, `abs(y1 - y2)` for `abs_error` (bscores.h:188-196).
        C++ stores this in a `std::function<score_t(contin_t, contin_t)>`
        member: the lambda body itself computes in `contin_t` (double)
        precision, and the DECLARED `score_t` return type narrows the
        result exactly once, at the point the `std::function` wrapper is
        invoked -- reproduced here by narrowing once, inside the lambda's
        own return, rather than narrowing `y1 - y2` first.

        DIVERGENCE(C++ moses/moses/scoring/bscores.h:194-195): C++'s
        `default: OC_ASSERT(false);` inside a `switch` over its own
        two-member enum is unreachable from any REAL caller (every
        `err_function_type` value passed here is one of the two enum
        members) -- ported as `AssertionError` rather than silently
        picking a default, matching this port's convention for a direct
        `OC_ASSERT(false, ...)` (docs/CARRY-FORWARD.md Sec 5).
        """
        if eft == self.err_function_type.squared_error:
            self._err_func: Callable[[float, float], score_t] = lambda y1, y2: _to_score_t(
                _sq(y1 - y2)
            )
        elif eft == self.err_function_type.abs_error:
            self._err_func = lambda y1, y2: _to_score_t(abs(y1 - y2))
        else:
            raise AssertionError(f"contin_bscore.init: bad err_function_type {eft!r}")

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:140 contin_bscore::operator()

        "boost::transform(cti, target, back_inserter(bs), [&](mts, v) {
        contin_t tar = get_contin(v), res = get_contin(interpret_tr(mts));
        return -err_func(res, tar); });" (bscores.cc:151-156, paraphrased)
        -- `cti` (this row's inputs) and `target` (this row's expected
        output) are walked IN LOCKSTEP, `tar` computed before `res` in
        each iteration (C++'s comma-declaration order), and the element
        appended is `-err_func(res, tar)`: note the ARGUMENT ORDER is
        `(res, tar)`, not `(tar, res)` -- immaterial for `squared_error`
        (symmetric) but NOT for `abs_error` (also symmetric, since
        `abs(a-b) == abs(b-a)` -- so, in fact, immaterial for both
        `err_function_type`s this port implements; preserved anyway for
        fidelity, in case a future `err_function_type` is added that
        is not symmetric).

        `zip(..., strict=True)` (rather than C++'s unchecked
        `boost::transform`, which iterates only as many times as `cti`
        has elements and is undefined behaviour if `target` is shorter)
        is a deliberate, disclosed strengthening -- same convention
        `ctruth_table_bscore.__call__`'s own `_best_possible_score` zip
        already uses (bscores.py, this file) -- surfacing a length
        mismatch loudly instead of silently reading past `target`'s end
        or silently dropping `cti`'s tail.
        """
        iv = interpreter_visitor(tr)
        bs: behavioral_score = []
        for row, target_v in zip(self._cti, self._target, strict=True):
            tar = get_contin(target_v)
            res = get_contin(iv(list(row)))
            bs.append(-self._err_func(res, tar))
        log_candidate_bscore(tr, bs)  # C++: bscores.cc:158
        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:163 contin_bscore::best_possible_bscore

        "return behavioral_score(target.size(), 0);" -- a vector of
        `target.size()` zeros, matching classic's `target.size()` exactly
        (NOT `self._size`, which is the INPUT row count -- see `__init__`'s
        own docstring on why the two can differ for the first two
        constructor overloads, though every real caller keeps them equal).
        "The best possible bscore is a vector of zeros. That's probably
        not quite true, because there could be duplicated inputs, but
        that's acceptable for now." (bscores.h:198-200's own comment.)
        """
        return [0.0] * len(self._target)

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:168 contin_bscore::min_improv

        "The backwards compat version of this is 0.0. But for
        continuously-variable scores, this is crazy, as the system falls
        into a state of tweaking the tenth decimal place. Limit any such
        tweaking to 4 decimal places of precision." (bscores.cc:170-174's
        own comment, reproduced verbatim.) Narrowed via `_to_score_t`
        since `min_improv`'s C++ return type is itself `score_t`.
        """
        return _to_score_t(-1.0e-4)

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/bscores.cc:181 contin_bscore::set_complexity_coef
        (the `(alphabet_size, stdev)` overload -- OVERRIDES
        `bscore_base::set_complexity_coef`'s own two-argument overload,
        per `contin_bscore`'s `virtual void set_complexity_coef(unsigned,
        float)` declaration, bscores.h:216) and
        moses/moses/scoring/scoring_base.cc:63 bscore_base::set_complexity_coef
        (the one-argument `complexity_ratio` overload, brought into this
        class's scope UNCHANGED by C++'s own `using bscore_base::
        set_complexity_coef;`, bscores.h:217).

        DIVERGENCE (a default that differs from the inherited one, even
        though this class always overrides it explicitly): `bscore_base`'s
        OWN two-argument overload uses `discrete_complexity_coef` and
        gates on `0.0 < p < 0.5`; THIS override uses
        `_contin_complexity_coef` (`log(alphabet_size) * 2 * stdev**2`)
        and gates on `stdev > 0.0` alone -- a completely different formula
        and a completely different singularity condition, not merely a
        parameter rename. Delegating the two-argument case to
        `super().set_complexity_coef(*args)` here would silently apply the
        WRONG formula for every real caller of THIS class (table-
        problems.cc's `it`/regression problems, the only real callers of
        `contin_bscore::set_complexity_coef` in classic).

        `stdev` is narrowed to `score_t` (float32) FIRST -- matching
        C++'s own `float stdev` PARAMETER type, which narrows at the call
        boundary itself, before the `stdev > 0.0` gate or the
        `_contin_complexity_coef` call ever see it (same discipline
        `bscore_base.set_complexity_coef`'s own docstring already
        documents for its sibling overload).

        The `logger().info()` line at bscores.cc:187-189 is emitted below.
        """
        if len(args) == 2:
            alphabet_size, stdev = args
            stdev = _to_score_t(stdev)
            self._complexity_coef = 0.0
            if stdev > 0.0:
                self._complexity_coef = _contin_complexity_coef(alphabet_size, stdev)
            # C++: bscores.cc:187-189. "alphabest" is classic's own typo,
            # and the ratio is the RECIPROCAL of the stored coefficient --
            # `inf` when the coefficient stayed 0.0.
            logger().info(
                f"contin_bscore noise = {ostream_double(stdev)}"
                f" alphabest size = {alphabet_size}"
                f" complexity ratio = {ostream_double(_cpp_reciprocal(self._complexity_coef))}"
            )
        else:
            super().set_complexity_coef(*args)


def _c_div(x: score_t, y: score_t) -> score_t:
    """Not a ported symbol -- IEEE-754 division with C's semantics.

    Python raises `ZeroDivisionError` where C++ yields `+/-inf` (finite
    numerator) or `nan` (`0/0`). `cluster_bscore.__call__` below performs
    three divisions that C++ lets run to an IEEE result; this reproduces
    that rather than letting an exception escape. Same reconstruct-by-hand
    shape `interpreter.py`'s own `div` case already uses
    (interpreter.cc:157-167), minus its `OverflowException` throw, which
    belongs to the interpreter and not to arithmetic.
    """
    try:
        return _to_score_t(x / y)
    except ZeroDivisionError:
        if x == 0.0 or math.isnan(x):
            return math.nan
        return math.copysign(math.inf, x) * math.copysign(1.0, y)


class cluster_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:613 cluster_bscore

    "Cluster scoring. Experimental attempt at using moses for cluster
    discovery. When this scorer is presented with a combo tree, it
    attempts to see if the tree, when applied to the input table,
    naturally clusters scores into disparate groups. Since the combo
    tree, applied to the input table, results in a 1-d array of values,
    the clustering is judged by using the one-dimensional k-means
    clustering algo.

    This is considered experimental because it doesn't yet work very
    well, is likely to be redisigned, and finally, doesn't even output
    all the data that is required to use the resulting formula (the
    edges, with are printed by hand, below).

    XXX this should probably be removed! TODO FIXME"
    (bscores.cc:916-930's own comment, reproduced verbatim, typos and
    all.) The "printed by hand, below" it refers to is an `#if 0` block
    inside `operator()` (bscores.cc:1059-1067) -- dead in every real
    build, and not ported.

    DO NOT FIX -- three upstream defects in `operator()` below are
    reproduced deliberately, because classic's own printed scores depend
    on them (see `__call__`'s docstring for each):

    1. a value that crosses a cluster edge is dropped from BOTH clusters;
    2. the LAST cluster's variance never reaches `final`;
    3. `centers` seeds from the first three table rows in table order,
       not from any spread heuristic.
    """

    def __init__(self, itable: ITable) -> None:
        """C++: moses/moses/scoring/bscores.cc:932 cluster_bscore::cluster_bscore

        `: _itable(itable)` -- `ITable _itable;` (bscores.h:626) is a
        by-VALUE member, so the member-initializer list copy-constructs
        it. `.copy()` here for the same reason `contin_bscore.__init__`
        copies its own `cti`/`target` (see that constructor's docstring):
        a bare assignment would alias the caller's mutable list where C++
        copies a value type.

        `bscore_base`'s `_size` stays 0: C++'s constructor never sets it
        either, and nothing in this class reads it -- `operator()` sizes
        everything off `oned.size()`, computed per call.
        """
        super().__init__()
        self._itable = itable.copy()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:937 cluster_bscore::operator()

        One-dimensional k-means (Lloyd's algorithm) over the column that
        `tr` produces from `self._itable`; the single returned bscore
        element is the RECIPROCAL of the binwidth-normalised RMS cluster
        width, so that "the narrower the peaks, the higher the score"
        (bscores.cc:1053-1054's own comment).

        ## `nclusters` is a hard-coded 3, and that removes every wrap

        `size_t nclusters = 3;` (bscores.cc:942) is a local literal, not
        a parameter. Every `size_t` subtraction in the function is
        therefore checked and none can wrap: `nclusters-1` is 2;
        `numvals-1` (`vals[numvals-1]`, bscores.cc:1051) is safe because
        `OC_ASSERT(nclusters < oned.size())` (bscores.cc:944) already
        guarantees `numvals >= 4`; and `edge_idx` is only ever indexed
        below `nclusters-1`, under an `OC_ASSERT` of exactly that
        (bscores.cc:996). No mask is needed anywhere here -- see this
        docstring's own assertion notes below for why that last
        `OC_ASSERT` cannot fire.

        ## DO NOT FIX 1: a boundary-crossing value is dropped entirely

        C++: bscores.cc:989-1000. When `sc > edges[i]` the `else` branch
        advances `i` and moves straight on to the next `j` -- it never
        re-tests `sc` against the NEW `edges[i]`, so that one value is
        counted into NEITHER cluster. Every cluster boundary therefore
        loses exactly one sample from `cnt`/`sum`. Reproduced as written:
        classic's printed scores are computed from these truncated sums.

        ## DO NOT FIX 2: the last cluster never contributes to `final`

        C++: bscores.cc:1026-1046. The RMS loop accumulates `cnt`/`sum`/
        `squ` and folds them into `final` only in its `else` branch, i.e.
        only when a boundary is crossed. There is no flush after the
        loop ends, so the final cluster's variance is computed and then
        discarded. With `nclusters == 3`, `final` is the sum of exactly
        TWO cluster variances, never three. Reproduced as written.

        ## DO NOT FIX 3: the initial centroids are the first three rows

        C++: bscores.cc:946-952. `centers[i] = get_contin(oned[i])` for
        `i` in 0..2 -- the first three rows of the table in table order,
        sorted. Not a spread heuristic, not k-means++; whatever the input
        file happens to list first. Reproduced as written, because it
        decides which fixed point Lloyd's converges to.

        ## `std::sort` (bscores.cc:953, :968): tie order is NOT observable

        Both sorts are over `vector<score_t>` -- bare float values, not
        records with a payload -- so this port uses `list.sort()`, NOT
        `merging.py`'s `_sort` introsort transliteration (see
        docs/CARRY-FORWARD.md 5b.1, which requires that decision to be
        made deliberately rather than by default). The reasoning, in
        three parts:

        1. `std::sort` and `list.sort` both produce the unique
           non-decreasing arrangement of the same multiset. Two
           `<`-equal floats that are bit-identical are indistinguishable
           once placed, so no permutation among them is observable.
        2. The one pair that compares equal while remaining
           distinguishable is `+0.0`/`-0.0`. Neither is observable here:
           `sc <= edges[i]`, `cnt[i] += 1.0` and `squ += sc*sc` treat
           them identically, and `sum[i] += sc` starts from `+0.0`, for
           which `+0.0 + -0.0` and `+0.0 + +0.0` are both `+0.0`. The
           only site where the sign could survive is `binwidth =
           vals[numvals-1] - vals[0]` (bscores.cc:1051), and reaching it
           with a signed zero at BOTH ends requires every value to be
           zero -- which returns `-inf` from the `cnt[i] < 3.5` guard
           (bscores.cc:1008) long before `binwidth` is computed. With a
           zero at only one end the subtraction gives the same result
           for either sign.
        3. NaN makes `std::sort` undefined behaviour in C++ (it breaks
           the strict weak ordering), so no permutation is "the"
           faithful one. That case is moot: a NaN (or an infinity)
           ANYWHERE in `vals` makes the sweep at bscores.cc:982-986
           return `[-INFINITY]` when it reaches that element, and the
           sweep visits every element, so the returned value does not
           depend on where the sort put it. `centers` draws its three
           values from the same column, so the same argument covers it.
           This port's `list.sort()` orders NaN by position rather than
           value; the guard makes that unobservable.

        ## Termination

        The `while (changed)` loop (bscores.cc:973) has no iteration cap
        in classic and none is added here. Its state is the partition
        recorded in `edge_idx`, and the next partition is a deterministic
        function of it, so the loop terminates iff that map reaches a
        fixed point rather than a longer cycle. Textbook Lloyd's
        guarantees this via a monotonically decreasing objective, but
        DO NOT FIX 1 above drops a sample per boundary each pass, so that
        argument does not carry over unmodified and a cycle of length > 1
        cannot be ruled out on paper. Classic would spin forever on such
        an input; so does this port. Adding a cap classic does not have
        would change the printed score on every input that needs one more
        pass than the cap allows.

        ## Not ported

        The ANN branch of `OTable(tr, itable)` (table.cc:322-338, taken
        only when `is_ann_type(*tr.begin())`) -- the ANN domain is
        excluded project-wide, and `cluster_table_problem` forces
        `contin_type` (table-problems.cc:555), so no ANN tree reaches
        here. The non-ANN branch (table.cc:339-343) is the
        `interpreter_visitor`-per-row loop written out below, identical
        in shape to `contin_bscore.__call__`'s own.

        `log_candidate_bscore(tr, bs)` (bscores.cc:1068) is a logging
        call -- no logging surface exists in this port.
        """
        # C++: bscores.cc:940 `OTable oned(tr, _itable);` -- table.cc:319.
        iv = interpreter_visitor(tr)
        oned = OTable([iv(list(row)) for row in self._itable])

        nclusters = 3

        # C++: bscores.cc:944 OC_ASSERT(nclusters < oned.size()).
        if not nclusters < len(oned):
            raise AssertionError(
                f"cluster_bscore: nclusters ({nclusters}) < oned.size() ({len(oned)})"
            )

        # Initial guess for the centroids. C++: bscores.cc:946-953.
        # `centers` is `vector<score_t>` (float) while `get_contin`
        # returns `contin_t` (double): the assignment narrows.
        centers = [_to_score_t(get_contin(oned[i])) for i in range(nclusters)]
        centers.sort()

        # C++: bscores.cc:955-960. `centers[i] + centers[i+1]` is
        # float+float (rounded to float); `0.5 *` then promotes to double
        # and the assignment back into `edges` narrows again -- two
        # roundings, reproduced by the nested `_to_score_t`.
        edges = [score_t(0.0)] * nclusters
        for i in range(nclusters - 1):
            edges[i] = _to_score_t(0.5 * _to_score_t(centers[i] + centers[i + 1]))
        edges[nclusters - 1] = math.inf

        # sort the values. This makes assignment easier.
        # C++: bscores.cc:962-969.
        numvals = len(oned)
        vals = [_to_score_t(get_contin(oned[j])) for j in range(numvals)]
        vals.sort()

        # One-dimensional k-means algorithm (LLoyd's algorithm)
        # C++: bscores.cc:970-1017.
        edge_idx = [0] * (nclusters - 1)
        changed = True
        while changed:
            cnt = [score_t(0.0)] * nclusters
            sum_ = [score_t(0.0)] * nclusters
            changed = False
            i = 0
            for j in range(numvals):
                sc = vals[j]

                if math.isinf(sc) or math.isnan(sc):
                    return [-math.inf]

                if sc <= edges[i]:
                    # `cnt[i] += 1.0` / `sum[i] += sc`: the accumulators
                    # are `score_t` (float), so EVERY step rounds to
                    # float32 -- accumulating in Python's native double
                    # here would drift from classic on any table long
                    # enough for the sum to lose low bits.
                    cnt[i] = _to_score_t(cnt[i] + 1.0)
                    sum_[i] = _to_score_t(sum_[i] + sc)
                else:
                    # C++: bscores.cc:996 OC_ASSERT(i<nclusters-1). It
                    # cannot fire: `edges[nclusters-1]` is set to
                    # `INFINITY` at bscores.cc:960 and is never
                    # reassigned (the update loop at bscores.cc:1016 runs
                    # only to `nclusters-1`), so once `i == nclusters-1`
                    # every non-NaN `sc <= inf` takes the other branch --
                    # and NaN has already returned above. Ported anyway,
                    # for fidelity to a real `OC_ASSERT`.
                    if not i < nclusters - 1:
                        raise AssertionError(
                            f"cluster_bscore: i ({i}) < nclusters-1 ({nclusters - 1})"
                        )
                    if j != edge_idx[i]:
                        changed = True
                    edge_idx[i] = j
                    i += 1

            # Compute cluster centers. C++: bscores.cc:1003-1015.
            for i in range(nclusters):
                # A cluster must have at least two points in it,
                # as otherwise the RMS would be zero.  Heck, lets make it three.
                if cnt[i] < 3.5:
                    return [-math.inf]
                sum_[i] = _to_score_t(sum_[i] / cnt[i])
            for i in range(nclusters - 1):
                edges[i] = _to_score_t(0.5 * _to_score_t(sum_[i] + sum_[i + 1]))

        # Compute the RMS width of each cluster. C++: bscores.cc:1020-1046.
        final = score_t(0.0)
        cnt_s = score_t(0.0)
        sum_s = score_t(0.0)
        squ = score_t(0.0)
        i = 0
        for j in range(numvals):
            sc = vals[j]
            if sc <= edges[i]:
                cnt_s = _to_score_t(cnt_s + 1.0)
                sum_s = _to_score_t(sum_s + sc)
                # `squ += sc*sc`: `sc*sc` rounds to float first, then the
                # accumulation rounds again.
                squ = _to_score_t(squ + _to_score_t(sc * sc))
            else:
                sum_s = _to_score_t(sum_s / cnt_s)
                squ = _to_score_t(squ / cnt_s)

                final = _to_score_t(final + _to_score_t(squ - _to_score_t(sum_s * sum_s)))
                i += 1
                cnt_s = score_t(0.0)
                sum_s = score_t(0.0)
                squ = score_t(0.0)

        # normalize by bind-width. C++: bscores.cc:1048-1055 (comment typo
        # "bind-width" for bin-width is classic's own).
        #
        # `sqrt(final)`: `final` can be NEGATIVE here -- it is a sum of
        # `squ - sum*sum` terms, i.e. float32 variances, each of which
        # rounds below zero when a cluster's values are near-identical.
        # C's `sqrt` returns NaN for that; Python's `math.sqrt` raises
        # `ValueError`, so the NaN is reconstructed by hand, the same way
        # interpreter.py reconstructs `log`'s (interpreter.cc:177).
        # Double-rounding is not an issue for `sqrt`: binary64 carries 53
        # bits, more than the 2*24+2 needed for a binary32 square root to
        # round identically through binary64.
        final = math.nan if final < 0.0 else _to_score_t(math.sqrt(final))
        # score_t binwidth = edges[nclusters-2] - edges[0];
        binwidth = _to_score_t(vals[numvals - 1] - vals[0])
        # `binwidth` cannot be zero: it is zero only when `vals[0] ==
        # vals[numvals-1]`, i.e. every value is equal, which returns
        # `-inf` from the `cnt[i] < 3.5` guard above. `_c_div` covers it
        # regardless, so an IEEE result reaches the caller instead of a
        # `ZeroDivisionError` if that reasoning is ever wrong.
        final = _c_div(final, binwidth)
        # The narrower the peaks, the higher the score.
        # This way of doing it works better with the complexity penalty
        #
        # `final` here IS reachably zero -- a partition whose first two
        # clusters each hold a single repeated value gives `squ ==
        # sum*sum` exactly -- and C++ then yields `+inf`, not an
        # exception. Hence `_c_div` rather than a bare `1.0 / final`.
        final = _c_div(score_t(1.0), final)

        bs: behavioral_score = [final]
        # C++: bscores.cc:1059-1067's `#if 0` debug dump is dead code and is
        # not ported; bscores.cc:1068's `log_candidate_bscore` is live.
        log_candidate_bscore(tr, bs)
        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:1074 cluster_bscore::best_possible_bscore

        "Return the best possible bscore. Used as one of the termination
        conditions (when the best bscore is reached)." (bscores.h:619-620's
        own comment.) `behavioral_score(1, very_best_score)` -- a
        one-element vector holding `very_best_score` (types.h:71, this
        port's `very_best_score` in types.py, bit-exactly C's `FLT_MAX`).

        Note this is UNREACHABLE as a termination condition in practice:
        `operator()` returns a reciprocal of an RMS width, which is finite
        and nowhere near `FLT_MAX` for any real table.
        """
        return [very_best_score]

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:1079 cluster_bscore::min_improv

        A flat `0.1` -- far coarser than `contin_bscore`'s `-1.0e-4`,
        because this scorer's own score is a reciprocal RMS on the order
        of 1..100 rather than a summed error. Narrowed via `_to_score_t`
        since `min_improv`'s C++ return type is `score_t`; `0.1` is not
        exactly representable in binary32, so the narrowing is not a
        no-op.
        """
        return _to_score_t(0.1)


# ---------------------------------------------------------------------------
# discretize_contin_bscore -- bscores.h:91 / bscores.cc:196.
# ---------------------------------------------------------------------------


class discretize_contin_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:91 discretize_contin_bscore

    Scores a contin-output tree as a CLASSIFIER: each threshold cuts the
    contin range into a class, and a row scores `weights[c]` when the
    tree's class differs from the target's, `0` when it agrees.

    Two things about this scorer are the opposite of every other one here,
    and both are classic's, not a port choice:

    * **Its bscore entries are non-negative penalties**, so a PERFECT
      candidate scores all zeros -- which is exactly what
      `best_possible_bscore` returns (bscores.cc:222-225). Elsewhere in
      this file a better candidate scores closer to zero from BELOW.
    * **`min_improv()` returns 0.0**, with classic's own comment saying
      "not necessarily right, just the backwards-compat behavior"
      (bscores.cc:227-231). Ported verbatim, comment included.

    Reached only through `-R/--discretize-threshold` on a contin-output
    table (`table-problems.cc:520-524`).
    """

    def __init__(
        self,
        ot: OTable,
        it: ITable,
        thres: Sequence[float],
        wa: bool = False,
    ) -> None:
        """C++: moses/moses/scoring/bscores.cc:196 discretize_contin_bscore"""
        super().__init__()
        # `target(ot), cit(it)` copy-construct in C++'s member-initializer
        # list, matching this file's established copying convention.
        self._target = ot.copy()
        self._cit = it.copy()
        # `boost::sort(thresholds)` (cc:204) sorts the MEMBER in place,
        # after it was copy-initialised from the argument -- so the
        # caller's vector is untouched. `sorted()` gives both halves.
        self._thresholds: list[float] = sorted(thres)
        self._weighted_accuracy = wa
        # `weights(thresholds.size() + 1, 1)` (cc:201) -- one weight per
        # class, and there is one more class than there are thresholds.
        self._weights: list[score_t] = [_to_score_t(1.0)] * (len(self._thresholds) + 1)

        # `boost::transform(target, classes.begin(), ...)` (cc:206-207).
        self._classes: list[int] = [self.class_idx(get_contin(v)) for v in self._target]

        if self._weighted_accuracy:
            # C++: cc:209-213. `cs.count(i)` is a `multiset` count, i.e.
            # how many rows fall in class `i`.
            #
            # The division is `classes.size() / (float)(weights.size() *
            # cs.count(i))` -- a FLOAT divide, so the result is narrowed to
            # binary32 here and not at some later comparison. A class with
            # no members makes `cs.count(i)` zero and C++ divides by zero
            # in float, yielding +inf rather than trapping; `_c_div`
            # reproduces that rather than raising ZeroDivisionError.
            n_classes = len(self._weights)
            for i in range(n_classes):
                count = self._classes.count(i)
                self._weights[i] = _c_div(
                    _to_score_t(len(self._classes)),
                    _to_score_t(n_classes * count),
                )

        # DIVERGENCE(C++ moses/moses/scoring/bscores.cc:215-218): the
        # OC_ASSERT on mismatched table sizes aborts the process; this
        # raises ValueError, matching this file's convention.
        if len(ot) != len(it):
            raise ValueError(
                "Error: discretize_contin_bscore: input and output table size "
                f"do not match: {len(it)} {len(ot)}"
            )
        self._size = len(ot)

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:222 discretize_contin_bscore::best_possible_bscore"""
        return behavioral_score([_to_score_t(0.0)] * len(self._target))

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:228 discretize_contin_bscore::min_improv"""
        # "not necessarily right, just the backwards-compat behavior" --
        # classic's own comment (cc:229-230).
        return _to_score_t(0.0)

    def class_idx(self, v: float) -> int:
        """C++: moses/moses/scoring/bscores.cc:233 discretize_contin_bscore::class_idx"""
        if v < self._thresholds[0]:
            return 0
        s = len(self._thresholds)
        if v >= self._thresholds[s - 1]:
            return s
        return self.class_idx_within(v, 1, s)

    def class_idx_within(self, v: float, l_idx: int, u_idx: int) -> int:
        """C++: moses/moses/scoring/bscores.cc:242 discretize_contin_bscore::class_idx_within

        A binary search over `thresholds`, transliterated including its
        recursion. `u_idx - l_idx` is a `size_t` subtraction in C++, but
        every caller keeps `l_idx < u_idx` (`class_idx` enters with
        `(1, s)` after ruling out both ends, and each recursive call
        preserves it), so it cannot wrap -- no mask is needed here, and
        this note records that it was checked rather than overlooked.
        """
        # base case
        if u_idx - l_idx == 1:
            return l_idx
        # recursive case
        m_idx = l_idx + (u_idx - l_idx) // 2
        t = self._thresholds[m_idx - 1]
        if v < t:
            return self.class_idx_within(v, l_idx, m_idx)
        return self.class_idx_within(v, m_idx, u_idx)

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:257 discretize_contin_bscore::operator()"""
        ct = OTable(tr, self._cit)
        # `(c_idx != this->class_idx(get_contin(v))) * this->weights[c_idx]`
        # (cc:266-267) -- a bool promoted to int and multiplied by a float
        # weight, so a MATCH contributes exactly 0 and a MISS contributes
        # the class weight. Note it indexes `weights` by the TARGET's class
        # `c_idx`, not by the candidate's.
        bs = behavioral_score(
            [
                _to_score_t((c_idx != self.class_idx(get_contin(v))) * self._weights[c_idx])
                for v, c_idx in zip(ct, self._classes, strict=True)
            ]
        )
        log_candidate_bscore(tr, bs)  # C++: bscores.cc:269
        return bs


# ===========================================================================
# interesting_predicate_bscore -- bscores.h:553 / bscores.cc:726-905, and
# the two cogutil statistics it is the only MOSES caller of.
# ===========================================================================


def _fmax(x: float, y: float) -> float:
    """Not a ported symbol -- C99 `fmax`, which `bscores.cc:897-900` calls
    three times and which Python's `max` is NOT a drop-in for.

    `fmax` returns the OTHER operand when one is a NaN (`max` returns
    whichever the comparison happens to pick, and `max(nan, 0.0)` is
    `nan`). That difference is load-bearing here, not theoretical: with
    `--min-rand-input 0` (the DEFAULT) the first quotient in
    `get_activation_penalty` is `0.0f/0.0f`, a NaN, and classic's `dst`
    comes out as the SECOND quotient -- measured, `tests/fixtures/
    ip_bscore.json` group `activation`, case
    `(0.5, min=0.0, max=0.8) -> dst 0.0`, where `max` would give `nan`.

    When neither is a NaN this is glibc's `x < y ? y : x`, so `fmax(+0.0,
    -0.0)` returns `+0.0` and `fmax(-0.0, +0.0)` returns `-0.0` -- C
    leaves that unspecified and nothing here observes it.
    """
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return y if x < y else x


def _c_pow(base: float, exponent: float) -> float:
    """Not a ported symbol -- `std::pow(double, double)` as called at
    `bscores.cc:903`, i.e. glibc's `pow`. `time_dispersion.py`'s `_powf`
    is the binary32 sibling; this is the binary64 one, and the two are NOT
    interchangeable (`bscores.cc:903` is `pow(1.0 - dst, _penalty)`, whose
    first argument is a `double` because of the `1.0` literal, so the
    `double` overload is selected and `_penalty` is PROMOTED, not the
    other way round).

    The same four families where `math.pow` is not a drop-in, all MEASURED
    against classic's own toolchain here rather than carried over from
    `powf` (`tools/cpp_probe/ip_bscore_probe.cc`'s `activation` mode,
    pinned in `tests/fixtures/ip_bscore.json`):

    1. **NaN base.** glibc XORs its internal `sign_bias` -- set only for a
       negative base with an ODD-INTEGER exponent -- into the result's
       sign, so a NEGATIVE NaN base comes back POSITIVE for an odd integer
       exponent and negative otherwise. Measured, with `dst == -nan` (what
       the DEFAULT activation bounds produce): `penalty 1.0 -> nan`,
       `penalty 2.0 -> -nan`, `penalty 0.5 -> -nan`. That single character
       is the whole difference between classic's `nan true` and `-nan
       true` on the `-H ip` sweep rows.
    2. **Negative finite base, non-integer exponent** -- C99 domain error,
       a negative NaN; Python raises `ValueError`. Unreachable from
       `get_activation_penalty` (`dst <= 1` for any activation in
       `[0, 1]`, so `1.0 - dst >= 0`), written out for the contract.
    3. **Zero base, negative exponent** -- C99 pole error, a signed
       infinity; Python raises `ValueError`.
    4. **Overflow** -- `+-inf`; Python raises `OverflowError`.

    `pow(x, +-0) == 1` for every `x`, NaN included, so the exponent-zero
    case is answered first, matching C99 and keeping the NaN rule from
    firing on it.
    """
    if exponent == 0.0:
        return 1.0

    odd_int = exponent.is_integer() and int(exponent) % 2 != 0
    negative_base = math.copysign(1.0, base) < 0.0

    if math.isnan(base):
        return -math.nan if (negative_base and not odd_int) else math.nan

    try:
        return math.pow(base, exponent)
    except ValueError:
        if base == 0.0:
            return -math.inf if (negative_base and odd_int) else math.inf
        return -math.nan
    except OverflowError:
        return -math.inf if (negative_base and odd_int) else math.inf


def _c_log(x: float) -> float:
    """Not a ported symbol -- `std::log(double)` as called at
    `bscores.cc:903`, which never raises where Python's `math.log` does.

    `log(0)` is `-inf` (the C99 pole error), `log(x < 0)` is a negative
    NaN (the domain error), and a NaN propagates sign and all. Reachable
    here: `dst == 1.0` exactly -- an activation of 0 against a positive
    `--min-rand-input` -- makes `pow(0.0, penalty)` zero and the penalty
    `-inf`. Measured, `tests/fixtures/ip_bscore.json` group `activation`.

    A local sibling of the identically-shaped `_c_log` in
    `pymetta/feature_selection/scorers/kld.py`, not an import: that module
    is a `pymetta.feature_selection` one, and small C-semantics helpers
    are kept per-file in this project (`_alphabet_size`,
    `_type_to_exemplar`, `_sq`, `_contin_complexity_coef` all follow the
    same convention).
    """
    if math.isnan(x):
        return x
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return -math.nan
    return math.log(x)


def _accumulate_with_int_init(values: Iterable[float]) -> float:
    """C++: opencog/util/MannWhitneyU.h:53 accumulate

    Breadcrumb cites the expression, not a declaration: this reproduces
    `boost::accumulate(c2 | map_values, 0)`, which has no definition of
    its own in cogutil. The expression appears three times --
    MannWhitneyU.h:53 (inside `MannWhitneyU`) and :67/:69 (inside
    `standardizedMannWhitneyU`).

    `boost::accumulate` is `std::accumulate`, whose accumulator type comes
    from the INIT VALUE, and the init is the literal `0`: an `int`. So the
    running sum is `acc = acc + *it` with `acc` an `int` and `*it` a
    `contin_t` -- the addition happens in double and the result is
    converted back to `int`, TRUNCATING TOWARD ZERO, at EVERY step.

    Observable: yes, and not only in theory. A `--score-weight` column
    makes every `CTable` count fractional, and `n1`/`n2` then truncate.
    Measured (`tests/fixtures/ip_bscore.json`, group `stats`, case
    `fractional_counts`): for `c1 = {1:1.5, 2:2.5, 3:0.25}` the exact
    total is 4.25 and the accumulated `n1` is 3; for
    `c2 = {2:0.75, 3:0.25}` the exact total is 1.0 and `n2` is **0**,
    which makes `sU` zero and `standardizedMannWhitneyU` return `+inf`
    instead of a finite z-score.

    Traversal is the caller's order, which every caller here makes
    ascending-by-key to match `std::map`. That matters: truncation makes
    the accumulation NON-ASSOCIATIVE (0.6+0.6 truncating to 1, versus 0
    twice).

    A local sibling of `kld.py`'s identically-shaped helper, per this
    project's per-file convention for cogutil-adjacent helpers; it is
    cited to MannWhitneyU.h here because that is the expression this copy
    exists for.
    """
    acc = 0
    for value in values:
        acc = int(acc + value)
    return float(acc)


def _ranking(counter: dict[float, float]) -> dict[float, float]:
    """C++: opencog/util/ranking.h:40 ranking

    "Return a rank given a counter. Ties ranks are averaged. Given a
    Counter {'blue': 3, 'red': 2, 'green': 1}, it will produce a new
    Counter {'blue': 3, 'red': 4.5, 'green': 6}" (ranking.h:33-38's own
    comment -- note the example is written in Python-Counter insertion
    order, but the C++ container is a `std::map`, so the walk is ascending
    BY KEY, which for those three strings happens to be the same order).

    `lrank` is carried across iterations and `res.insert(res.end(), ...)`
    appends with a hint, so BOTH the running rank and the result's order
    depend on the ascending walk; iterating a Python `dict` in insertion
    order here would assign every rank to the wrong key. Sorted
    explicitly, leaving the returned `dict` in that same ascending order
    (which is what `MannWhitneyU`'s `r[v.first]` lookups and this port's
    own tests then see).

    `(2 * lrank + v.second - 1) / 2` is evaluated in `FloatT` (double)
    throughout: the `2` and `1` literals are `int`s promoted by the
    `double` operand, and the trailing `/ 2` is likewise a double divide.
    """
    res: dict[float, float] = {}
    lrank = 1.0
    for key, count in sorted(counter.items()):
        res[key] = (2 * lrank + count - 1) / 2
        lrank += count
    return res


def _counter_add(c1: dict[float, float], c2: dict[float, float]) -> dict[float, float]:
    """C++: opencog/util/Counter.h:127 Counter

    Breadcrumb cites the bare enclosing CLASS: the real symbol is
    `Counter::operator+=`, and `operator+=` is not an identifier the
    alignment checker can look for (it strips to `operator+=`, which
    matches nothing).

    `c1 + c2`, which `boost::arithmetic<Counter>` (Counter.h:55) defines
    as "copy c1, then `+= c2`" -- and `Counter::operator+=` is
    `for (const auto& v : other) this->operator[](v.first) += v.second;`
    (Counter.h:127-131), where `std::map::operator[]` VALUE-INITIALIZES a
    missing key to `0` before adding. `dict.get(key, 0.0)` is that.

    Breadcrumb cites `operator+=`, the member that carries the behaviour;
    the free `operator+` it is used through is generated by
    `boost::arithmetic` and has no line of its own.

    `other` is walked ascending (it is a `std::map`), which decides the
    order the summed values are computed in -- the result is
    order-insensitive for exact doubles but not in general, so the walk is
    reproduced rather than left to insertion order. The RESULT is left in
    "c1's keys in c1's order, then c2's new keys" order; every consumer
    here (`_ranking`) sorts.
    """
    res = dict(c1)
    for key, count in sorted(c2.items()):
        res[key] = res.get(key, 0.0) + count
    return res


def _mann_whitney_u(c1: dict[float, float], c2: dict[float, float], n2: float = -1.0) -> float:
    """C++: opencog/util/MannWhitneyU.h:44 MannWhitneyU

    "return the Mann-Whitney U (U_2 to be precise) given 2 distributions
    described by Counters. If n2, the size of c2, is not provided it is
    automatically calculated." (MannWhitneyU.h:39-42's own comment.)

    Two things are easy to get wrong and are reproduced exactly:

    * `sum2 += r[v.first]` sums each of c2's ranks ONCE, unweighted --
      the count `v.second` is not multiplied in, even though `r` was built
      from the COMBINED counter `c1 + c2`. Verified against classic on the
      `identical` case (c1 == c2 == {1:1, 2:2, 5:1}): ranks 1.5/4.5/7.5
      sum to 13.5, and U is 3.5, not the 20.5 a count-weighted sum would
      give.
    * `r[v.first]` is `std::map::operator[]`, which INSERTS a
      value-initialized `0` for a missing key rather than throwing. No key
      of c2 can be missing from `c1 + c2`, so the insert never fires; it
      is written out (`res.setdefault`) so the port does not raise where
      C++ would silently continue.

    `n2 * (n2 + 1) / 2` is `(n2 * (n2 + 1)) / 2` in doubles.
    """
    c = _counter_add(c1, c2)
    r = _ranking(c)
    sum2 = 0.0
    for key, _ in sorted(c2.items()):
        sum2 += r.setdefault(key, 0.0)
    if n2 < 0:
        n2 = _accumulate_with_int_init(count for _, count in sorted(c2.items()))
    return sum2 - n2 * (n2 + 1) / 2


def _standardized_mann_whitney_u(
    c1: dict[float, float], c2: dict[float, float], n1: float = -1.0, n2: float = -1.0
) -> float:
    """C++: opencog/util/MannWhitneyU.h:64 standardizedMannWhitneyU

    "return the standardized Mann-Whitney U given 2 distributions
    described by Counters" (MannWhitneyU.h:60-61's own comment). The one
    MOSES caller is `interesting_predicate_bscore::operator()`
    (bscores.cc:856), which passes neither `n1` nor `n2`, so both go
    through `_accumulate_with_int_init`'s truncation.

    `n1`/`n2` are computed BEFORE `MannWhitneyU` is called and the already
    -truncated `n2` is passed down, so `MannWhitneyU` never recomputes it.

    `(U - mU) / sU` divides by zero whenever `n1` or `n2` truncates to 0 --
    `+inf` when `U > mU`, and `0/0`'s NEGATIVE NaN when the predicate
    selected nothing at all. Both measured; see `_ieee_div`. `sqrt` of a
    negative argument is C's domain error (a negative NaN) rather than
    Python's `ValueError`; unreachable for non-negative counts, written
    out for the contract.
    """
    if n1 < 0:
        n1 = _accumulate_with_int_init(count for _, count in sorted(c1.items()))
    if n2 < 0:
        n2 = _accumulate_with_int_init(count for _, count in sorted(c2.items()))
    u = _mann_whitney_u(c1, c2, n2)
    m_u = n1 * n2 / 2
    radicand = m_u * (n1 + n2 + 1) / 6
    s_u = -math.nan if radicand < 0.0 else math.sqrt(radicand)
    return _ieee_div(u - m_u, s_u)


def _weighted_skewness(counter: dict[float, float]) -> float:
    """C++: boost/accumulators/statistics/weighted_skewness.hpp:59 result

    `weighted_skewness(acc)` on a
    `accumulator_set<contin_t, stats<tag::weighted_skewness>, contin_t>`
    (bscores.h:558-561), fed by
    `for (const auto& v : pred_counter) acc(v.first, weight = v.second);`
    (bscores.cc:751-752 and :827-828) -- so the SAMPLE is the observed
    value and the WEIGHT is its count, and the feed order is the
    `std::map`'s, ascending by key.

    Breadcrumb cites `weighted_skewness_impl::result`, the member that
    holds the formula, under the header's own path (the alignment checker
    resolves `boost/...` against `/usr/include`, the third of its roots --
    the same route `neighborhood_sampling.py`'s
    `_boost_binomial_coefficient_double` already uses).

    Boost's own documented formula, transcribed with its EXACT
    associativity because every operation is a double rounding:

        (m3 - 3.*m2*mean + 2.*mean*mean*mean)
        / ((m2 - mean*mean) * sqrt(m2 - mean*mean))

    where `mean` is `weighted_mean` = `weighted_sum / sum_of_weights`
    (weighted_mean.hpp:56, the LAZY default) and `m_n` is
    `weighted_moment<n>` = `sum(w_i * pow(x_i, n)) / sum_of_weights`
    (weighted_moment.hpp:60). `numeric::pow(x, mpl::int_<2>)` is `x * x`
    and `numeric::pow(x, mpl::int_<3>)` is `(x * x) * x`
    (moment.hpp:28-42) -- repeated multiplication, NOT `std::pow`, which
    rounds differently.

    Every accumulator is a running sum in the order the samples arrive, so
    the ascending walk is load-bearing beyond tidiness.

    Degenerate inputs, both reachable and both measured
    (`tests/fixtures/ip_bscore.json`, group `stats`): an EMPTY counter
    gives `0/0` for `mean`/`m2`/`m3` and a NEGATIVE NaN overall (the
    predicate selected no rows); a SINGLE-point counter gives a zero
    denominator and likewise a negative NaN. `numeric::fdiv` on two
    `double`s is plain `/`, so those go through `_ieee_div`.
    """
    sum_of_weights = 0.0
    weighted_sum = 0.0
    moment2_sum = 0.0
    moment3_sum = 0.0
    for x, w in sorted(counter.items()):
        sum_of_weights += w
        weighted_sum += w * x
        moment2_sum += w * (x * x)
        moment3_sum += w * ((x * x) * x)

    mean = _ieee_div(weighted_sum, sum_of_weights)
    m2 = _ieee_div(moment2_sum, sum_of_weights)
    m3 = _ieee_div(moment3_sum, sum_of_weights)

    numerator = m3 - 3.0 * m2 * mean + 2.0 * mean * mean * mean
    variance = m2 - mean * mean
    root = -math.nan if variance < 0.0 else math.sqrt(variance)
    return _ieee_div(numerator, variance * root)


def _ctable_rows_in_map_order(ctable: CTable) -> list[tuple[tuple, object]]:
    """No C++ symbol -- re-establishes `std::map`'s ordering over a
    `CTable`, which this port stores as an insertion-ordered `dict`.

    `CTable` is `std::map<multi_type_seq, TimedCounter>` (table.h:550), so
    every `boost::for_each(_ctable | map_values, ...)` in
    `interesting_predicate_bscore` walks the rows ASCENDING BY INPUT
    TUPLE. The port's `Table.compressed` inserts in FILE ROW ORDER, which
    is a different order for any table whose rows are not already sorted.

    That difference is observable here, and only here among this port's
    scorers, because `operator()` builds `pred_counter` with an
    ASSIGNMENT (`pred_counter[...] = mv.second`, bscores.cc:800) rather
    than a `+=`: when two selected rows report the same output value, the
    LAST one visited wins, so the visiting order decides the value.
    MEASURED both ways: classic's `-H ip` output is byte-identical for a
    14-row table and for the same 14 rows reversed, which it could not be
    if it walked them in file order.

    `multi_type_seq::operator<` is `get_variant() < r.get_variant()`
    (table.h:383). `-H ip` admits only all-boolean inputs
    (`table-problems.cc:233-237`), so every key's variant alternative is
    the same one -- a `builtin_seq`, i.e. `vector<builtin>`, confirmed by
    measurement: `get_seq<vertex>()` throws `boost::bad_get` on these keys
    -- and the comparison reduces to a lexicographic walk over the
    `builtin` enum's ordinals. `builtin` is an `IntEnum` here with the
    same ordinals (`logical_true = 1` BEFORE `logical_false = 2`,
    vertex.h:74), so a plain tuple `sorted()` IS that comparison. Classic
    therefore visits `(true, true, true)` first and `(false, false,
    false)` last, and so does this.
    """
    return sorted(ctable.items())


def _cell_items_in_value_order(cell) -> list[tuple[object, float]]:
    """No C++ symbol -- re-establishes `std::map`'s ordering over ONE
    `CTable` cell, the companion to `_ctable_rows_in_map_order` above.

    A cell is a `TimedCounter`, i.e. `Counter<TimedValue, count_t>`
    (table.h:496), so C++ walks it ascending by `TimedValue`. This port
    carries no timestamps inside a `CTable` cell (every `TimedValue`
    collapses to its bare vertex -- docs/CARRY-FORWARD.md Sec 3), so that
    ordering reduces here to ascending by the observation's `contin_t`
    value, which is also what `TimedValue::operator<` (table.h:482) gives
    for a set of same-timestamp entries.

    It matters in both of this scorer's walks: the constructor's `+=`
    histogram accumulates in this order (a float-rounding dependency), and
    `operator()`'s `pred_counter` ASSIGNS in it (a which-value-wins
    dependency).
    """
    return sorted(timed_items_of(cell), key=lambda item: get_contin(item[0].value))


class interesting_predicate_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:553 interesting_predicate_bscore

    The `-H ip` ("find interesting patterns") scorer. It does NOT score a
    candidate against a target column: it treats the candidate as a
    PREDICATE that selects rows, and rewards the selection whose output
    distribution is most unlike the unconditioned one -- by
    Kullback-Leibler divergence, by a shift in skewness, and by a
    standardized Mann-Whitney U. `__call__`'s own docstring walks the
    four weighted components.

    `ip_problem::run` (table-problems.cc:229) is the only constructor
    site, and it `OC_ASSERT`s first that the table's signature is exactly
    `->(boolean ... boolean) contin` (table-problems.cc:233-237), so the
    predicate's inputs are Boolean and the histogram's values are contin.

    REACHABILITY, measured rather than assumed. Classic exits 0 on this
    problem, but at its DEFAULT activation bounds (`--min-rand-input 0`,
    `--max-rand-input 1`) every candidate's activation penalty is a NaN --
    `0.0f/0.0f` in both arms of `get_activation_penalty` -- so every
    composite score is `nan` and classic prints a single `nan true` line.
    The four `--ip_*` weights make no difference there, which is exactly
    what the 16 `-H ip` rows of this project's CLI sweep observe. Give it
    real bounds (`-q 0.2 -w 0.8`) and the scorer discriminates normally.
    Both regimes are covered by this port's tests; neither is a
    degenerate-output path (classic prints a real, ranked candidate list
    in the second).
    """

    def __init__(
        self,
        ctable: CTable,
        kld_weight: float = 1.0,
        skewness_weight: float = 1.0,
        stdU_weight: float = 1.0,
        skew_U_weight: float = 1.0,
        min_activation: float = 0.0,
        max_activation: float = 1.0,
        penalty: float = 1.0,
        positive: bool = True,
        abs_skewness: bool = False,
        decompose_kld: bool = False,
    ) -> None:
        """C++: moses/moses/scoring/bscores.cc:726 interesting_predicate_bscore

        Every weight and bound is a `score_t`/`weight_t` PARAMETER in C++
        (`typedef score_t weight_t;`, bscores.h:555), so each narrows to
        binary32 at the call boundary -- before any `> 0` gate below ever
        sees it. `ip_problem::run` passes `double`s
        (`ip_problem_params`' four fields are `double`,
        table-problems.h:79-82), so the narrowing is real, not notional.

        `CTable _ctable;` (bscores.h:590) is a by-VALUE member, so the
        member-initializer list copy-constructs the caller's table;
        `_copy_ctable` here for the same reason (docs/CARRY-FORWARD.md
        Sec 1.1/1.2: a bare assignment would alias where C++ copies).

        The histogram loop (cc:743-747) is a nested `boost::for_each` over
        the ctable's cells, `+=`-ing each `(value, count)` into
        `_counter` -- an ADDITION, unlike `operator()`'s `pred_counter`,
        which assigns. Both walks are ascending, per
        `_ctable_rows_in_map_order`; the inner one is ascending by the
        cell's own key, which for this port (no timestamps in a `CTable`
        cell -- docs/CARRY-FORWARD.md Sec 3) is ascending by contin value.

        `_skewness` IS ONLY ASSIGNED WHEN `_kld_w > 0` (cc:749-758): the
        whole precompute block, `_pdf` included, sits inside that `if`.
        `contin_t _skewness;` (bscores.h:591) has no default member
        initializer and is not in the constructor's initializer list, so
        `--ip_kld_weight=0` leaves classic READING AN UNINITIALIZED
        `double` in `operator()`'s `diff_skewness` -- undefined behaviour.

        # DIVERGENCE(C++ moses/moses/scoring/bscores.cc:749): this port
        # initialises `_skewness` to 0.0 rather than leaving it
        # uninitialised, the same refusal-to-reproduce-UB this project
        # already applies to an out-of-range argument index
        # (docs/CARRY-FORWARD.md Sec 3) and to `KLDS::p_s`
        # (feature_selection/scorers/kld.py). Measured on this toolchain,
        # classic's uninitialised `_skewness` READS AS 0.0 -- `-Hip -K0
        # -J0 -U0 -X1` scores every candidate exactly 0 on a symmetric
        # table, which requires `_skewness == pred_skewness == 0` -- so
        # 0.0 is also the value that agrees with the measurement. It is
        # still a divergence and not a port of anything: another build,
        # another allocator or another stack layout can hand classic any
        # bit pattern there, and no port can be faithful to that.

        `abs_skewness` and `decompose_kld` are the two constructor
        parameters `ip_problem::run` never passes (it supplies nine of the
        eleven, cc:248-256), so both keep their declared defaults on every
        CLI-reachable path. Ported anyway, because they change
        `operator()`'s shape rather than just a constant.
        """
        super().__init__()
        self._ctable = _copy_ctable(ctable)
        self._kld_w = _to_score_t(kld_weight)
        self._skewness_w = _to_score_t(skewness_weight)
        self._abs_skewness = abs_skewness
        self._stdU_w = _to_score_t(stdU_weight)
        self._skew_U_w = _to_score_t(skew_U_weight)
        self._min_activation = _to_score_t(min_activation)
        self._max_activation = _to_score_t(max_activation)
        self._penalty = _to_score_t(penalty)
        self._positive = positive
        self._decompose_kld = decompose_kld

        # Define counter (mapping between observation and its number of
        # occurences). That is, create a historgram showing how often each
        # output value occurs in the ctable. (cc:739-742's own comment,
        # typos and all.)
        self._counter: dict[float, float] = {}
        for _row, cell in _ctable_rows_in_map_order(self._ctable):
            for tv, count in _cell_items_in_value_order(cell):
                value = get_contin(tv.value)
                self._counter[value] = self._counter.get(value, 0.0) + count

        self._pdf: dict[float, float] = {}
        self._klds = KLDS()
        self._skewness: float = 0.0
        # Precompute pdf (probability distribution function). C++: cc:749-758.
        if self._kld_w > 0:
            self._pdf = dict(self._counter)
            self._klds.set_p_pdf(self._pdf)

            # Compute the skewness of the pdf. `_pdf` is still the raw
            # COUNTS here: `set_p_pdf` copies into KLDS's own `p_pdf` and
            # only that copy is turned into densities (KLD.h:67-73).
            self._skewness = _weighted_skewness(self._pdf)
            logger().debug("interesting_predicate_bscore::_skewness = %f", self._skewness)

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:761 interesting_predicate_bscore::operator()

        "OK, here's the deal. The combo tree evaluates to T/F on each
        input table row. That is, the combo tree is a predicate that
        selects certain rows of the input table. Here, pred_cache is just
        a cache, to avoid multiple evaluations of the combo tree: its just
        a table with just one column, equal to the value of the combo tree
        on each input row." (cc:763-769's own comment.)

        Bscore layout, in push order -- between two and five elements
        depending on the weights:

        0. `_kld_w * KLD(unconditioned || selected)`, if `_kld_w > 0`.
           With `_decompose_kld` it is instead one element PER DISTINCT
           OBSERVATION of the unconditioned distribution.
        1. `_skewness_w * (skewness(selected) - _skewness)`, if
           `_skewness_w > 0`.
        2. `_stdU_w * |standardizedMannWhitneyU|`, if `_stdU_w > 0`.
        3. `_skew_U_w * stdU * diff_skewness` -- "the product of the
           relative differences of the shift (stdU) and the skewness (so
           that if both go in the same direction the value if positive,
           and negative otherwise)" (cc:858-861's own comment).
        4. the activation penalty, ALWAYS.

        Note elements 1-3 are gated as a BLOCK on
        `_skewness_w > 0 || _stdU_w > 0 || _skew_U_w > 0` (cc:824) and
        then individually, so `-J0 -U0 -X1` still computes both
        statistics; and `stdU` is computed whenever `_stdU_w > 0 ||
        _skew_U_w > 0` but only PUSHED when `_stdU_w > 0`.

        ## `OTable pred_cache(tr, _ctable)` is written out inline

        `OTable(const combo_tree&, const CTable&, const string&)`
        (table.cc:350) is not ported (table.py's own class docstring says
        so), and this is its only MOSES call site. Its body is the
        `interpreter_visitor`-per-KEY loop below -- the same
        write-it-out-inline treatment `cluster_bscore.__call__` already
        gives the ITable overload. Its first statement, `arity_set as =
        get_argument_abs_idx_set(tr);` (table.cc:352), is DEAD -- `as` is
        never read -- and is not reproduced.

        ## DO NOT FIX: `pred_counter` ASSIGNS where `_counter` ADDS

        `pred_counter[get_contin(mv.first.value)] = mv.second;`
        (cc:800) -- an assignment. Two different selected rows that share
        an output value do NOT accumulate: the later one in `std::map`
        order overwrites the earlier. The constructor's histogram, three
        dozen lines up, uses `+=` on the same shape of data. So the
        "conditioned" distribution this scorer compares against the
        unconditioned one is not a sub-histogram of it -- it under-counts
        every value that appears in more than one selected row.
        Reproduced as written: classic's printed scores are computed from
        it, and it is the reason `_ctable_rows_in_map_order` exists.

        ## The `pred_counter.size() == 1` guard does not cover size 0

        cc:805-811 returns `[very_worst_score]` for a ONE-value filtered
        distribution ("Statistics like skewness need a distribution that
        isn't a single spike"), but a predicate that selects NOTHING falls
        straight through. Measured in classic for `and($1 not($1))`:
        `[-inf, -nan, nan, -nan, -inf]`. Every one of those comes from a
        division by zero this port routes through `_ieee_div` /
        `KLDS.next`'s own IEEE arithmetic.
        """
        # C++: cc:771 `OTable pred_cache(tr, _ctable);` -- table.cc:350.
        rows = _ctable_rows_in_map_order(self._ctable)
        iv = interpreter_visitor(tr)
        pred_cache = [iv(list(row)) for row, _cell in rows]

        # target simply negates (inverts) the predicate. C++: cc:773-774.
        target = bool_to_vertex(self._positive)

        # Count how many rows the predicate selected. C++: cc:776-787.
        # `unsigned tc = c.total_count();` narrows `count_t` (double) to
        # `unsigned`, truncating -- which a `--score-weight` column makes
        # observable, exactly as `_ctable_usize` does elsewhere.
        total = 0
        actives = 0
        for (_row, cell), v in zip(rows, pred_cache, strict=True):
            tc = _to_usize(sum(count for _tv, count in timed_items_of(cell)))
            if vertex_equal(v, target):
                actives += tc
            total += tc

        logger().fine("ip scorer: total = %u", total)
        logger().fine("ip scorer: actives = %u", actives)

        # Create a histogram of output values, ignoring non-selected rows.
        # C++: cc:794-802. NOTE the assignment -- see this method's own
        # DO NOT FIX note.
        pred_counter: dict[float, float] = {}
        for (_row, cell), v in zip(rows, pred_cache, strict=True):
            if vertex_equal(v, target):
                for tv, count in _cell_items_in_value_order(cell):
                    pred_counter[get_contin(tv.value)] = count

        logger().fine("ip scorer: pred_cache.size() = %u", len(pred_cache))
        logger().fine("ip scorer: pred_counter.size() = %u", len(pred_counter))

        # If there's only one output value left, then punt. Statistics
        # like skewness need a distribution that isn't a single spike.
        # C++: cc:805-811.
        if len(pred_counter) == 1:
            bs: behavioral_score = [very_worst_score]
            log_candidate_bscore(tr, bs)
            return bs

        bs = behavioral_score()
        # Compute Kullback-Leibler divergence (KLD) of the filetered
        # distribution. C++: cc:813-823.
        if self._kld_w > 0.0:
            if self._decompose_kld:
                # `_klds(pred_counter, back_inserter(bs))` then
                # `boost::transform(bs, bs.begin(), _kld_w * arg1)`
                # (cc:816-817): the phoenix lambda multiplies each element
                # by the weight IN PLACE, after the fact.
                bs.extend(self._klds.components(pred_counter))
                for i, element in enumerate(bs):
                    bs[i] = _to_score_t(self._kld_w * _to_score_t(element))
            else:
                pred_klds = _to_score_t(self._klds(pred_counter))
                logger().fine("ip scorer: klds = %f", pred_klds)
                bs.append(_to_score_t(self._kld_w * pred_klds))

        # Compute skewness of the filtered distribution. C++: cc:825-863.
        if self._skewness_w > 0 or self._stdU_w > 0 or self._skew_U_w > 0:
            diff_skewness: score_t = 0.0
            if self._skewness_w > 0 or self._skew_U_w > 0:
                # push the absolute difference between the unconditioned
                # skewness and conditioned one. `pred_skewness` is a
                # `score_t` LOCAL (cc:838) -- the double result of
                # `weighted_skewness` narrows BEFORE the subtraction.
                pred_skewness = _to_score_t(_weighted_skewness(pred_counter))
                diff_skewness = _to_score_t(pred_skewness - self._skewness)
                val_skewness = abs(diff_skewness) if self._abs_skewness else diff_skewness
                logger().fine("ip scorer: pred_skewness = %f", pred_skewness)
                if self._skewness_w > 0:
                    bs.append(_to_score_t(self._skewness_w * val_skewness))

            stdU: score_t = 0.0
            if self._stdU_w > 0 or self._skew_U_w > 0:
                # Compute the standardized Mann-Whitney U. C++: cc:856.
                stdU = _to_score_t(_standardized_mann_whitney_u(self._counter, pred_counter))
                logger().fine("ip scorer: stdU = %f", stdU)
                if self._stdU_w > 0.0:
                    bs.append(_to_score_t(self._stdU_w * abs(stdU)))

            # push the product of the relative differences of the shift
            # (stdU) and the skewness (so that if both go in the same
            # direction the value if positive, and negative otherwise)
            if self._skew_U_w > 0:
                bs.append(_to_score_t(_to_score_t(self._skew_U_w * stdU) * diff_skewness))

        # add activation_penalty component. C++: cc:865-871.
        activation = _to_score_t(_ieee_div(float(actives), _to_score_t(float(total))))
        activation_penalty = self._get_activation_penalty(activation)
        logger().fine("ip scorer: activation = %f", activation)
        logger().fine("ip scorer: activation penalty = %e", activation_penalty)
        bs.append(activation_penalty)

        log_candidate_bscore(tr, bs)  # C++: cc:873
        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/bscores.cc:876 best_possible_bscore

        Breadcrumb cites the BARE member name, which is all the alignment
        checker keeps anyway (it strips everything before the last `::`);
        the real symbol is
        `interesting_predicate_bscore::best_possible_bscore`, and the
        fully-qualified spelling every sibling here uses would not fit the
        100-column limit.

        "the KLD has no upper boundary so the best of possible score is
        the maximum value a behavioral_score can represent"
        (bscores.h:576-577's own comment) -- `behavioral_score(1,
        very_best_score)`, a ONE-element score regardless of how many
        elements `operator()` actually returns.
        """
        return [very_best_score]

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:906 interesting_predicate_bscore::min_improv

        "not necessarily right, just the backwards-compatible behavior"
        (cc:907-908's own comment) -- 0.0, the same value
        `bscore_base::min_improv` already returns, spelled out here
        because classic spells it out.
        """
        return 0.0

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/bscores.cc:880 interesting_predicate_bscore::set_complexity_coef
        (the `(alphabet_size, stdev)` overload, which OVERRIDES
        `bscore_base`'s own two-argument one per the `virtual void
        set_complexity_coef(unsigned, float)` declaration at bscores.h:582)
        and moses/moses/scoring/scoring_base.cc:63
        bscore_base::set_complexity_coef (the one-argument
        `complexity_ratio` overload, brought into this class's scope
        UNCHANGED by C++'s own `using bscore_base::set_complexity_coef;`,
        bscores.h:583).

        Byte-for-byte the same body as `contin_bscore.set_complexity_coef`
        above, including `_contin_complexity_coef` and the `stdev > 0`
        gate -- NOT `bscore_base`'s `discrete_complexity_coef` with its
        `0 < p < 0.5` gate, which would be a different formula AND a
        different singularity. Only the log line's prefix differs, and
        classic misspells it: "intersting_predicate_bscore" (cc:888), plus
        the same "alphabest" typo the other two carry.

        Which overload `-H ip` takes is decided by `set_noise_or_ratio`
        (table-problems.cc:41): `--noise >= 0` takes the two-argument one,
        otherwise `--complexity-ratio` takes the one-argument one. The
        default `--noise` is negative, so the ONE-argument form is the
        default path here.
        """
        if len(args) == 2:
            alphabet_size, stdev = args
            stdev = _to_score_t(stdev)
            self._complexity_coef = 0.0
            if stdev > 0:
                self._complexity_coef = _contin_complexity_coef(int(alphabet_size), stdev)
            # C++: cc:886-889. "intersting"/"alphabest" are classic's own
            # typos, and the ratio is the RECIPROCAL of the stored
            # coefficient -- `inf` when it stayed 0.0.
            logger().info(
                f"intersting_predicate_bscore noise = {ostream_double(stdev)}"
                f" alphabest size = {alphabet_size}"
                f" complexity ratio = {ostream_double(_cpp_reciprocal(self._complexity_coef))}"
            )
        else:
            super().set_complexity_coef(*args)

    def _get_activation_penalty(self, activation: score_t) -> score_t:
        """C++: moses/moses/scoring/bscores.cc:896 get_activation_penalty

        The real symbol is
        `interesting_predicate_bscore::get_activation_penalty`; the
        breadcrumb cites the bare member name because the qualified one
        does not fit the 100-column limit (see `best_possible_bscore`).

        A `private` member (bscores.h:608), so it is private here too.
        Every type in these two statements is load-bearing and none of
        them is uniform:

            score_t dst = fmax(fmax(_min_activation - activation, score_t(0))
                              / _min_activation,
                              fmax(activation - _max_activation, score_t(0))
                              / (1.0 - _max_activation));
            return log(pow((1.0 - dst), _penalty));

        * the FIRST quotient is `float / float` -- both operands are
          `score_t`, so it rounds to binary32;
        * the SECOND is `float / double`, because the `1.0` literal
          promotes `_max_activation`, so it stays binary64;
        * the outer `fmax` therefore resolves to the `double` overload and
          narrows only on the store to `score_t dst`;
        * `1.0 - dst` promotes back to `double`, so `pow` is the
          `(double, double)` overload with `_penalty` promoted -- NOT
          `powf`.

        Both quotients are `0/0` at classic's DEFAULT bounds
        (`--min-rand-input 0`, `--max-rand-input 1`), which is the whole
        reason `-H ip` prints `nan` out of the box. `fmax` returning the
        non-NaN operand is what keeps a HALF-degenerate configuration
        (only one bound at its default) finite -- see `_fmax`.

        `dst <= 1` for any `activation` in `[0, 1]`, so `1.0 - dst` is
        never negative and `pow`'s domain error is unreachable from here;
        `dst == 1` IS reachable (activation 0 against a positive low
        bound) and gives `log(0) == -inf`.
        """
        dst = _to_score_t(
            _fmax(
                _to_score_t(
                    _ieee_div(
                        _fmax(_to_score_t(self._min_activation - activation), 0.0),
                        self._min_activation,
                    )
                ),
                _ieee_div(
                    _fmax(_to_score_t(activation - self._max_activation), 0.0),
                    1.0 - self._max_activation,
                ),
            )
        )
        logger().fine("ip scorer: dst = %f", dst)
        return _to_score_t(_c_log(_c_pow(1.0 - dst, self._penalty)))
