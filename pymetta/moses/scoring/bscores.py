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
from collections.abc import Callable, Sequence
from enum import IntEnum

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, get_contin, negate_vertex, vertex
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, ITable, OTable, Table, type_node
from pymetta.moses.moses.types import (
    _to_score_t,
    behavioral_score,
    ostream_behavioral_score,
    score_t,
    very_best_score,
)
from pymetta.moses.scoring.scoring_base import (
    _cpp_reciprocal,
    bscore_base,
    log_candidate_bscore,
)
from pymetta.util.logger import logger, ostream_double


class ctruth_table_bscore(bscore_base):
    """C++: moses/moses/scoring/bscores.h:360 ctruth_table_bscore

    Extends `bscore_base` directly, per task-6-brief.md's interface line
    (`class ctruth_table_bscore(bscore_base)`) -- NOT `bscore_ctable_base`
    (bscores.h's own base, scoring_base.h:236), which is out of scope (see
    scoring_base.py's module docstring). Every `_wrk_ctable` reference in
    the C++ member functions below reads, in this port, as `self._ctable`
    -- there is no separate `_orig_ctable`/`_wrk_ctable` distinction to
    maintain without `ignore_cols`/`ignore_rows`.
    """

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/bscores.h:362 ctruth_table_bscore

        `_size = _wrk_ctable.size()` (bscores.h:365) is `len(ctable)`
        (compressed-row count, i.e. `dict.__len__`, not
        `uncompressed_size()`); `reset_weights()` (bscores.h:366) is not
        called -- it is boosting-only infrastructure this port does not
        implement (scoring_base.py's module docstring).

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
        super().__init__()
        self._ctable = ctable
        self._size = len(ctable)
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
        for row, counter in self._ctable.items():
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
            for counter in self._ctable.values()
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
        for counter in self._ctable.values():
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
        for counter in self._ctable.values():
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
