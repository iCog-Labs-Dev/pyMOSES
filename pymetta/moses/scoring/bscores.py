"""C++: moses/moses/scoring/bscores.h and bscores.cc

`ctruth_table_bscore` -- boolean classification scorer over a compressed
truth table. Ports exactly `bscores.h:360-383` (the class declaration) and
its out-of-line members at `bscores.cc:276-424`, per task-6-brief.md's
Reference line. Every other `*_bscore` class in bscores.h (`logical_bscore`,
`discretize_contin_bscore`, `enum_table_bscore`/`enum_filter_bscore`/
`enum_graded_bscore`, and the precision/multi-objective scorers in the
neighboring `precision_bscore.h`) is out of scope -- boolean scoring via
`ctruth_table_bscore` was the entire ported surface for that task (see the
plan's "Known gap, deliberate" note and that task's own Context section on
`CTable.get_output_type()`).

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
"""

import math
from collections.abc import Callable, Sequence
from enum import IntEnum

from pymetta.comboreduct.combo.vertex import builtin, combo_tree, get_contin, negate_vertex, vertex
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, ITable, OTable, Table, type_node
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t, very_best_score
from pymetta.moses.scoring.scoring_base import bscore_base


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

    def init(
        self, eft: contin_bscore.err_function_type = err_function_type.squared_error
    ) -> None:
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

        NOT ported: the `logger().info()` diagnostic line
        (bscores.cc:187-189) -- no logging surface exists in this port.
        """
        if len(args) == 2:
            alphabet_size, stdev = args
            stdev = _to_score_t(stdev)
            self._complexity_coef = 0.0
            if stdev > 0.0:
                self._complexity_coef = _contin_complexity_coef(alphabet_size, stdev)
        else:
            super().set_complexity_coef(*args)
