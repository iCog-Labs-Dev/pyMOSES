"""C++: moses/moses/scoring/select_bscore.h and select_bscore.cc

Ports `select_bscore`, the contin-table range-selection scorer registered
as the `-H select` problem.

## A real, reproducible read of UNINITIALIZED memory in classic's own
## constructor -- pinned via measurement, not simulated

`select_bscore`'s constructor (select_bscore.cc:67-149) does, in order:
`_return_weighted_score = true;` then `reset_weights();` -- BEFORE
`_lower_bound`/`_upper_bound` (`score_t` members with no member-initializer,
select_bscore.h:78-79) are ever assigned (that happens much later in the
SAME constructor, after the percentile-ranking loop). `reset_weights()` is
`select_bscore::reset_weights()` (the override; virtual dispatch inside a
constructor body resolves to the most-derived override once construction
of that class's own body has begun -- `select_bscore` has no further
subclass in this port), which calls `worst_possible_bscore()`, which reads
`_lower_bound <= val && val <= _upper_bound` for every observation in the
table -- using whatever bit pattern those two float members happen to hold
at that point. This is a genuine, textbook C++ "read before write" defect
in classic MOSES, not a port artifact.

Measured directly (2026-09-08, `tools/cpp_probe/scorers_probe.cc`'s
`select` mode, `probe_select_bscore` exposing `_lower_bound`/`_upper_bound`/
`_effective_length`/`_weights` via a subclass, on
`tests/fixtures/table_data/select_regression.csv`, TWO independent runs):
byte-identical output both times -- `_weights` uniformly `1.0`,
`_effective_length` exactly equal to the row count. That is exactly the
signature of every row's contin value being read as OUTSIDE whatever
`[_lower_bound, _upper_bound]` window existed at that moment (`worst_
possible_bscore` returns `-fabs(n_inside - n_outside)` per row; if every
row is entirely "outside", `n_inside=0` for all of them, so every element
is `-(that row's own total weight)`, and `reset_weights`'s `_effective_
length = -sum(bs)` collapses to exactly the row count when weights are
1.0 -- consistent with, though not a byte-for-byte proof of, the two
members reading as `0.0`/`0.0` on this toolchain/build: ANY combination
where the (garbage) interval is empty or does not overlap the real data
range gives the identical observable result, and `trainingRegression.dat`-
derived data is never exactly `0.0`). Reproducing genuine undefined
behaviour bit-for-bit is not possible in Python (there is no "uninitialized
memory" to read), so this port PINS `_lower_bound = _upper_bound = 0.0`
immediately before the premature `reset_weights()` call -- the simplest
value consistent with every independent measurement above -- and documents
it here as DO NOT FIX: this reproduces MEASURED behaviour on this
toolchain, not a theoretical worst case, and a different compiler/allocator
could in principle read something else. The REST of the constructor (the
percentile-ranking loop, `_lower_bound`/`_upper_bound`'s REAL values, and
every method that runs after construction completes) is unaffected: the
bug's only externally-visible trace is `_weights`/`_effective_length`,
which feed `sum_bscore`'s WEIGHTED branch (`scoring_base.py`) -- i.e. this
corrupts the scalar fitness `select_bscore` reports for the actual search,
not merely some inspectable-but-unused internal state.

## `__call__` cannot be exercised end-to-end in this port -- a project-wide
## gap, not a select_bscore-specific one

`select_bscore` REQUIRES a contin-output CTable (`OC_ASSERT(id::contin_type
== _wrk_ctable.get_output_type(), ...)`, this class's own constructor) --
by definition, every row's INPUT columns are also contin-valued in every
real dataset this scorer is used on (that is what "select a range of a
continuous-valued distribution" means). Evaluating the combo tree passed to
`__call__` therefore requires the "mixed" interpreter path (contin inputs,
boolean output). `interpreter_visitor` (interpreter.py) implements ONLY the
`vector<builtin>` overload -- "`contin_interpreter` and `mixed_interpreter`
are out of scope for the entire S0-S6 foundation cycle, not just this
task" (interpreter.py's own module docstring) -- and raises
`NotImplementedError` for any other input shape. So `__call__` below is
ported in full (its own logic is straightforward and correct once given a
working interpreter) but is UNREACHABLE on any real contin dataset today,
purely because of this pre-existing, already-disclosed, whole-project
interpreter gap -- not a defect introduced here. Every OTHER method
(`best_possible_bscore`, `worst_possible_bscore`, `min_improv`, `get_error`,
`reset_weights`, `update_weights`, `set_best_possible_bscore`, and the
constructor itself) needs no tree evaluation at all and is fully portable,
tested, and measured exact against classic (tests/moses/scoring/
test_select_bscore.py).
"""

from pymoses.comboreduct.combo.vertex import builtin, combo_tree, get_contin, vertex_equal
from pymoses.comboreduct.interpreter.interpreter import interpreter_visitor
from pymoses.comboreduct.table.table import CTable, type_node
from pymoses.moses.moses.types import (
    _to_score_t,
    behavioral_score,
    epsilon_score,
    ostream_behavioral_score,
    score_t,
    very_worst_score,
)
from pymoses.moses.scoring.scoring_base import _ieee_div, bscore_ctable_base
from pymoses.util.logger import logger, ostream_double


class select_bscore(bscore_ctable_base):
    """C++: moses/moses/scoring/select_bscore.h:50 select_bscore

    "Fitness function for selecting a range of rows from a continuous-
    valued distribution." (select_bscore.h:37-40's own doc comment.)
    """

    def __init__(
        self,
        ctable: CTable,
        lower_percentile: float = 0.8,
        upper_percentile: float = 0.9,
        hardness: float = 1.0,
        positive: bool = True,
    ) -> None:
        """C++: moses/moses/scoring/select_bscore.cc:67 select_bscore

        `hardness` is accepted (matching C++'s own signature) but NEVER
        STORED OR USED -- `select_bscore.h` declares no `_hardness`
        member at all, and its own class doc comment says so explicitly:
        "The 'hardness' indicates just how sharp its selection will be
        TBD. Not yet implemented." Ported as a faithfully-ignored
        parameter, not silently dropped from the signature.

        `lower_percentile`/`upper_percentile`/`hardness` are C++ `double`
        PARAMETERS (not `score_t`) -- no narrowing needed for them.
        Every CTABLE-DERIVED value (`val`, `weight`, `total_weight`,
        `_lower_bound`, `_upper_bound`) IS `score_t`, and is narrowed at
        every point classic's own code holds it in a `score_t` local or
        member -- in particular `score_t weight = tcv.second;`
        (select_bscore.cc:105 and reused verbatim at every one of this
        class's four scanning loops) narrows `count_t` (double) down to
        float32 at that EXACT declaration, and `total_weight`/`running_
        weight`/`fail`/`n_inside`/`n_outside` are all `score_t`
        ACCUMULATORS narrowed on every single `+=`, not once at the end
        -- the same "real float32 struct member" pattern discriminating_
        bscore.py's own `discriminator.count` docstring documents; a
        naive double accumulation was measured to diverge here (many-row,
        fractional-weight data makes this far more likely to be
        observable than it was for `discriminating_bscore`'s own boolean,
        unweighted fixture).
        """
        del hardness  # accepted, never used -- see docstring
        super().__init__(ctable)
        self._positive = positive

        if self._wrk_ctable.get_output_type() != type_node.contin_type:
            raise AssertionError("The selection scorer can only be used with contin-valued tables!")

        # This is needed even if not boosted, as otherwise the
        # ignore_cols(), ignore_rows() methods strip out rows, and cause
        # mis-matches during scoring. (select_bscore.cc:78-79's own
        # comment, reproduced verbatim -- ignore_cols/ignore_rows are
        # raising gates in this port, scoring_base.py, so the ORIGINAL
        # reason no longer applies operationally, but the flag itself is
        # still set unconditionally, exactly as classic does.)
        self._return_weighted_score = True

        # See this module's own docstring: classic reads _lower_bound/
        # _upper_bound here BEFORE they are ever assigned. Pinned to the
        # measured value. DO NOT FIX.
        self._lower_bound: score_t = 0.0
        self._upper_bound: score_t = 0.0
        self._best_possible_score: behavioral_score = []
        self._effective_length: float = 0.0
        self.reset_weights()

        lower_percentile = max(0.0, min(1.0, lower_percentile))
        upper_percentile = max(0.0, min(1.0, upper_percentile))
        if not (lower_percentile < upper_percentile):
            raise AssertionError(
                f"Selection scorer, invalid percentiles: {lower_percentile} and {upper_percentile}"
            )

        # "Maps are implicitly ordered, so the below has the effect of
        # putting the rows into sorted order, by output score."
        # (select_bscore.cc:93-94's own comment.) `std::map::insert`
        # silently KEEPS the first-inserted value for a colliding key and
        # discards the second -- unlike a plain Python dict assignment,
        # which would overwrite with the LAST one. Ported as an explicit
        # `if val not in ranked_out` guard to preserve "first wins".
        # NOTE: this loops the raw CONSTRUCTOR PARAMETER `ctable`
        # (select_bscore.cc:98, `for (io_row : ctable)`), not
        # `self._wrk_ctable` -- both hold identical rows at this point in
        # construction (`_wrk_ctable` is copy-constructed from it), so
        # this is observationally identical either way; ctable is used
        # here to mirror the C++ source's own variable name exactly.
        ranked_out: dict[score_t, score_t] = {}
        total_weight: score_t = 0.0
        for _row, counter in ctable.items():
            for v, raw_weight in counter.items():
                val = _to_score_t(get_contin(v))
                if not self._positive:
                    val = -val
                weight = _to_score_t(raw_weight)
                if val not in ranked_out:
                    ranked_out[val] = weight
                total_weight = _to_score_t(total_weight + weight)

        # C++: select_bscore.cc:107-109. "wieght" is classic's own typo.
        logger().info(
            f"select_bscore: lower_percentile = {ostream_double(lower_percentile)}"
            f" upper percentile = {ostream_double(upper_percentile)}"
            f" total wieght = {ostream_double(total_weight)}"
        )

        # "Now, carve out the selected range." (select_bscore.cc:115's
        # own comment.) `upper_percentile`/`lower_percentile` are C++
        # `double` locals throughout (`*= total_weight` promotes the
        # `score_t total_weight` to double for the multiply, with NO
        # narrowing back to float32 -- unlike every ctable-derived value
        # above) -- narrowing them here was a measured, real bug in an
        # earlier version of this port (shifted `_lower_bound`/
        # `_upper_bound` into the ~8th significant digit on
        # select_regression.csv). Kept as plain Python float (double).
        upper_percentile = upper_percentile * total_weight
        lower_percentile = lower_percentile * total_weight
        running_weight: score_t = 0.0
        last_val: score_t = very_worst_score
        found_lower = False
        found_upper = False
        for val in sorted(ranked_out):
            weight = ranked_out[val]
            running_weight = _to_score_t(running_weight + weight)
            if not found_lower and lower_percentile <= running_weight:
                self._lower_bound = val
                found_lower = True
            if not found_upper and upper_percentile <= running_weight:
                self._upper_bound = val
                found_upper = True
            last_val = val
        if not found_upper:
            self._upper_bound = _to_score_t(last_val * (1.0 + 2.0 * epsilon_score))

        if not (self._lower_bound < self._upper_bound):
            raise AssertionError(
                f"Selection scorer, invalid bounds: {self._lower_bound} and {self._upper_bound}"
            )

        # C++: select_bscore.cc:145-146.
        logger().info(
            f"select_bscore: lower_bound = {ostream_double(self._lower_bound)}"
            f" upper bound = {ostream_double(self._upper_bound)}"
        )

        self._best_possible_score = []
        self.set_best_possible_bscore()

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/select_bscore.cc:159 select_bscore::operator()

        "If the boolean tree correctly predicts that a row is inside/
        outside the selection range, then scorer rewards this with a
        score of zero. Otherwise, the score is minus the weight of that
        row." (select_bscore.cc:154-157's own doc comment.)

        `bs -= _best_possible_score` (select_bscore.cc:185) is an
        ELEMENT-WISE vector subtraction (`behavioral_score`'s own
        `operator-=`, types.h) -- ported as a per-element, per-step
        narrowed loop (`score_t -= score_t` is a native float32 op).
        """
        bs: behavioral_score = []
        iv = interpreter_visitor(tr)
        for row, counter in self._wrk_ctable.items():
            predict_inside = vertex_equal(iv(list(row)), builtin.logical_true)
            fail: score_t = 0.0
            for v, raw_weight in counter.items():
                val = _to_score_t(get_contin(v))
                if not self._positive:
                    val = -val
                weight = _to_score_t(raw_weight)
                inside = self._lower_bound <= val <= self._upper_bound
                if (predict_inside and not inside) or (not predict_inside and inside):
                    fail = _to_score_t(fail + weight)
            bs.append(-fail)

        for i in range(len(bs)):
            bs[i] = _to_score_t(bs[i] - self._best_possible_score[i])
        return bs

    def call_ensemble(self, ensemble) -> behavioral_score:
        """C++: moses/moses/scoring/select_bscore.cc:191 select_bscore
        (the `operator()(const scored_combo_tree_set&)` overload)

        "Scorer, suitable for use with boosting." (select_bscore.cc:190's
        own comment.) Reached from `-H select --boost=1`, via
        `ensemble.add_adaboost` -> `behave_cscore.get_bscore_ensemble` ->
        here; `select_bscore` is NOT an experts scorer (`table-problems.
        cc` never sets `ensemble_params.experts` for it), so AdaBoost is
        the path, and `sct.get_weight()` really does vary per member.

        `behavioral_score hypoth(_size, 0.0);` (select_bscore.cc:195) is a
        `std::vector<score_t>` -- **float32**, not the `std::vector<double>`
        that `precision_bscore::bias_selection` uses for its own `hypoth`.
        So every `+=` narrows here, and `score_t trweight = sct.get_
        weight();` (:198) narrows the double weight first.  `2.0 * ((score_t)
        select) - 1.0` is computed in DOUBLE (both literals are double),
        then multiplied by the float `trweight` -- so exactly one rounding
        per addend plus one per store, reproduced term by term.

        Step 2 is `__call__`'s own loop with `predict_inside` taken from
        `hypoth` instead of from interpreting one tree; C++ duplicates the
        loop rather than factoring it, and so does this.
        """
        # Step 1: accumulate the weighted prediction of each tree in
        # the ensemble.
        hypoth: behavioral_score = [0.0] * self._size
        for sct in ensemble:
            tr = sct.get_tree()
            trweight = _to_score_t(sct.get_weight())

            iv = interpreter_visitor(tr)

            i = 0
            for row in self._wrk_ctable:
                select = vertex_equal(iv(list(row)), builtin.logical_true)
                hypoth[i] = _to_score_t(hypoth[i] + trweight * (2.0 * _to_score_t(select) - 1.0))
                i += 1

        # Step 2: compare the prediction of the ensemble to the desired
        # result. The array "hypoth" is positive to predict true, and
        # negative to predict false.  The resulting score is 0 if correct,
        # and -1 if incorrect.
        bs: behavioral_score = []
        i = 0
        for _row, orow in self._wrk_ctable.items():
            predict_inside = 0 < hypoth[i]
            fail: score_t = 0.0
            for v, raw_weight in orow.items():
                val = _to_score_t(get_contin(v))
                if not self._positive:
                    val = -val
                weight = _to_score_t(raw_weight)
                inside = self._lower_bound <= val <= self._upper_bound
                if (predict_inside and not inside) or (not predict_inside and inside):
                    fail = _to_score_t(fail + weight)
            bs.append(-fail)
            i += 1

        # Report the score only relative to the best-possible score.
        for j in range(len(bs)):
            bs[j] = _to_score_t(bs[j] - self._best_possible_score[j])

        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/select_bscore.cc:243 select_bscore::best_possible_bscore"""
        return [0.0] * self._size

    def set_best_possible_bscore(self) -> None:
        """C++: moses/moses/scoring/select_bscore.cc:248 select_bscore::set_best_possible_bscore

        "Any row with a multiplicity of just one can be inherently
        classified ... for rows with multiplicities greater than one, the
        situation is trickier." (select_bscore.cc:254-258's own comment.)
        """
        for _row, counter in self._wrk_ctable.items():
            n_inside: score_t = 0.0
            n_outside: score_t = 0.0
            for v, raw_weight in counter.items():
                val = _to_score_t(get_contin(v))
                if not self._positive:
                    val = -val
                weight = _to_score_t(raw_weight)
                if self._lower_bound <= val <= self._upper_bound:
                    n_inside = _to_score_t(n_inside + weight)
                else:
                    n_outside = _to_score_t(n_outside + weight)
            total = _to_score_t(n_inside + n_outside)
            split = _to_score_t(abs(n_inside - n_outside))
            self._best_possible_score.append(_to_score_t(-0.5 * abs(total - split)))

        # C++: select_bscore.cc:276-277.
        logger().info(
            "select_bscore: Best possible: " + ostream_behavioral_score(self._best_possible_score)
        )

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/select_bscore.cc:280 select_bscore::worst_possible_bscore

        Called PREMATURELY once, from the constructor's own `reset_
        weights()`, before `_lower_bound`/`_upper_bound` are set -- see
        this module's own docstring.
        """
        bs: behavioral_score = []
        for _row, counter in self._wrk_ctable.items():
            n_inside: score_t = 0.0
            n_outside: score_t = 0.0
            for v, raw_weight in counter.items():
                val = _to_score_t(get_contin(v))
                if not self._positive:
                    val = -val
                weight = _to_score_t(raw_weight)
                if self._lower_bound <= val <= self._upper_bound:
                    n_inside = _to_score_t(n_inside + weight)
                else:
                    n_outside = _to_score_t(n_outside + weight)
            bs.append(_to_score_t(-abs(n_inside - n_outside)))
        return bs

    def get_error(self, bs: behavioral_score) -> score_t:
        """C++: moses/moses/scoring/select_bscore.cc:309 select_bscore::get_error

        Not reachable from anywhere else in this port (boosting-error/
        feature-selection plumbing, both out of scope -- scoring_base.py's
        module docstring) -- ported for completeness since it is short.
        """
        return _to_score_t(-_ieee_div(self.sum_bscore(bs), self._ctable_weight))

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/select_bscore.cc:317 select_bscore::min_improv

        "This is not quite right, for weighted rows ... (But I think
        boosting should not affect min-improv, right?)"
        (select_bscore.cc:314-316's own comment, reproduced verbatim.)
        """
        return _to_score_t(_ieee_div(1.0, self._ctable_weight))

    def reset_weights(self) -> None:
        """C++: moses/moses/scoring/select_bscore.cc:322 select_bscore::reset_weights

        Called TWICE in a real run: once prematurely from the constructor
        (this module's own docstring), and again -- correctly, with real
        bounds -- only if something outside this port ever calls it a
        second time (boosting only; never happens here). `znorm` is a C++
        `double` local, so no per-step narrowing is needed for it beyond
        each `bs[i]` term already being `score_t`-clean.
        """
        bs = self.worst_possible_bscore()
        znorm = 0.0
        for i in range(self._size):
            znorm -= bs[i]
        self._effective_length = znorm
        znorm = _ieee_div(znorm, self._size)
        self._weights = [znorm] * self._size

        # C++: select_bscore.cc:331. `znorm` is a `double` here, not a
        # `score_t`, but it is streamed bare either way -- six digits.
        logger().info(f"select_bscore::reset_weights() norm={ostream_double(znorm)}")

    def update_weights(self, rew: list[float]) -> None:
        """C++: moses/moses/scoring/select_bscore.cc:334 select_bscore::update_weights

        Boosting-only -- never called anywhere in this port.
        """
        znorm = 0.0
        for i in range(self._size):
            self._weights[i] *= rew[i]
            znorm += self._weights[i]
        znorm = _ieee_div(self._effective_length, znorm)
        for i in range(self._size):
            self._weights[i] *= znorm
