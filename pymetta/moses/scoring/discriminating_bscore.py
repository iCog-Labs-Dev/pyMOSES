"""C++: moses/moses/scoring/discriminating_bscore.h and discriminating_bscore.cc

Ports `discriminator` (the shared true/false-positive/negative counting
helper) and `discriminating_bscore` (the base class for `recall_bscore`,
`prerec_bscore`, `bep_bscore`, `f_one_bscore`) -- the P2 scorer-family
task's own priority ("Port `discriminating_bscore` first -- the others
derive from it, so getting its contract right is the whole slice's
foundation"). `discriminating_bscore` extends BOTH `bscore_ctable_base`
(scoring_base.py) and `discriminator` in C++ (`struct discriminating_bscore
: public bscore_ctable_base, discriminator`) -- ported as the same
multiple inheritance, since neither base conflicts (`discriminator` is not
itself a `bscore_base`).

## A genuine, reproducible upstream defect: `f_one_bscore` cannot be
## constructed at all, and `bep_bscore`'s own declared defaults crash too

`discriminating_bscore`'s own constructor (discriminating_bscore.cc:222-229)
asserts, whenever `hardness > 0`, that `0.0 < min_threshold <= max_threshold`.
`f_one_bscore`'s constructor (discriminating_bscore.cc:645-650) calls the
base constructor with HARDCODED `discriminating_bscore(ct, 0.0, 1.0,
1.0e-20)` -- `min_threshold=0.0`, `hardness=1.0e-20` (which IS `> 0.0`,
both in float32 and double). This combination ALWAYS fails the assert:
there are no constructor arguments a caller could pass to avoid it, since
none of the three values are caller-supplied. Verified live (2026-09-08,
this task's own `tools/cpp_probe/scorers_probe.cc`, `disc ... f_one ...`):
the probe binary aborts with `opencog::AssertionException`, message
"Discriminating scorer, invalid thresholds. The minimum threshold must be
greater than zero...", naming `discriminating_bscore.cc:224` -- i.e.
`f_one_bscore` is UNCONSTRUCTIBLE in classic MOSES as it stands today, and
`-H f_one` (registered in `table-problems.cc`'s `f_one_table_problem::run`,
which calls `REGRESSION(ctable, f_one_bscore, (ctable));` unconditionally)
crashes classic itself on first use, on any dataset. `bep_bscore`'s own
DECLARED DEFAULTS (`min_diff=0.0f, max_diff=0.5f, hardness=1.0f`,
discriminating_bscore.h:209-212) hit the exact same assert for the same
reason (`0.0 < 1.0` is true, so the branch is entered, and `0.0 < 0.0` is
false) -- ALSO verified live with the same probe. Unlike `f_one_bscore`,
`bep_bscore` is NOT permanently broken: `bep_table_problem::run()`
(table-problems.cc) always overrides `min_diff` to a nonzero value before
constructing (`if (0.0 == pms.hardness) { pms.hardness = 1.0;
pms.min_rand_input = 0.0; ...}` -- wait, that ALSO sets `min_rand_input`
(i.e. `min_diff`) to `0.0`, reproducing the SAME crash for the `-H bep`
CLI problem's own hardcoded fallback; a caller passing an explicit
`--hardness`/threshold pair different from that fallback avoids it).
DO NOT FIX either: this port raises the identical `AssertionError` at the
identical point, for the identical reason, rather than silently choosing
safer defaults classic itself does not have.
"""

import math

from pymetta.comboreduct.combo.vertex import (
    builtin,
    combo_tree,
    get_contin,
    is_contin,
    vertex_equal,
)
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, type_node
from pymetta.moses.moses.types import (
    _to_score_t,
    behavioral_score,
    score_t,
    very_best_score,
    very_worst_score,
)
from pymetta.moses.scoring.scoring_base import (
    _ieee_div,
    _to_usize,
    bscore_ctable_base,
    discrete_complexity_coef,
    log_candidate_bscore,
)
from pymetta.util.logger import logger, ostream_double


class discriminator:
    """C++: moses/moses/scoring/discriminating_bscore.h:51 discriminator

    "Perform summations commonly used in binary discriminators ... The
    count() method performs the actual counting." (discriminating_bscore.h
    :37-44's own doc comment.) Not a `bscore_base` itself -- a plain mixin
    `discriminating_bscore` multiply-inherits alongside `bscore_ctable_base`.
    """

    class d_counts:
        """C++: moses/moses/scoring/discriminating_bscore.h:55 discriminator::d_counts

        All six fields are `score_t` in C++ (discriminating_bscore.h:57-63)
        -- even `positive_count`/`negative_count`, despite being fed from
        an `unsigned totalc` at every call site (`count`/`counts` below):
        the widening `unsigned -> score_t` addition needs no narrowing
        (every representable `unsigned` row count fits exactly in
        float32's range for the corpora this port scores).
        """

        __slots__ = (
            "true_positive_sum",
            "false_positive_sum",
            "positive_count",
            "true_negative_sum",
            "false_negative_sum",
            "negative_count",
        )

        def __init__(self) -> None:
            """C++: moses/moses/scoring/discriminating_bscore.cc:107 d_counts::d_counts"""
            self.true_positive_sum: score_t = 0.0
            self.false_positive_sum: score_t = 0.0
            self.positive_count: score_t = 0.0
            self.true_negative_sum: score_t = 0.0
            self.false_negative_sum: score_t = 0.0
            self.negative_count: score_t = 0.0

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:55 discriminator::discriminator

        `sum_true`/`sum_false` (discriminating_bscore.h:85,87) are
        `std::function<score_t(const CTable::counter_t&)>` closures in
        C++, dispatched ONCE here on `_output_type` -- ported as bound
        methods selected the same way, not re-dispatched per call.
        `OC_ASSERT(false, "Discriminator, unsupported output type")`
        (discriminating_bscore.cc:90) is ported as `ValueError` (this
        port's convention for a data-validation `OC_ASSERT`, matching
        `ctruth_table_bscore.__init__`'s own boolean-output check,
        bscores.py).
        """
        self._ctable = ctable
        self._output_type = ctable.get_output_type()
        if self._output_type == type_node.boolean_type:
            self.sum_true = self._sum_true_boolean
            self.sum_false = self._sum_false_boolean
        elif self._output_type == type_node.contin_type:
            self.sum_true = self._sum_true_contin
            self.sum_false = self._sum_false_contin
        else:
            raise ValueError("Discriminator, unsupported output type")

        # `_true_total`/`_false_total` are `score_t` (float) MEMBERS
        # (discriminating_bscore.h:75-76) accumulated via `+=` over every
        # ctable row (discriminating_bscore.cc:99-100) -- narrowed on
        # every store, same reasoning as `count()`'s own docstring below.
        self._true_total: score_t = 0.0
        self._false_total: score_t = 0.0
        for counter in ctable.values():
            self._true_total = _to_score_t(self._true_total + self.sum_true(counter))
            self._false_total = _to_score_t(self._false_total + self.sum_false(counter))
        # C++: discriminating_bscore.cc:102-103.
        logger().info(
            f"Discriminator: num_true={ostream_double(self._true_total)}"
            f" num_false={ostream_double(self._false_total)}"
        )

    @staticmethod
    def _sum_true_boolean(counter) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:62 discriminator::sum_true (lambda)"""
        return counter.get(builtin.logical_true, 0.0)

    @staticmethod
    def _sum_false_boolean(counter) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:68 discriminator::sum_false (lambda)"""
        return counter.get(builtin.logical_false, 0.0)

    @staticmethod
    def _sum_true_contin(counter) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:74 discriminator::sum_true (lambda)

        "For contin tables, we return the sum of the row values > 0"
        (discriminating_bscore.cc:73's own comment) -- `std::max(0.0, val
        * weight)`, summed in `score_t` (`res` is declared `score_t`, so
        this port narrows the accumulator after every term, matching a
        real float32 variable rather than a double one).
        """
        res: score_t = 0.0
        for v, weight in counter.items():
            res = _to_score_t(res + max(0.0, get_contin(v) * weight))
        return res

    @staticmethod
    def _sum_false_contin(counter) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:82 discriminator::sum_false (lambda)

        "For contin tables, we return the sum of the row values < 0"
        (discriminating_bscore.cc:81's own comment.)
        """
        res: score_t = 0.0
        for v, weight in counter.items():
            res = _to_score_t(res + min(0.0, get_contin(v) * weight))
        return res

    def count(self, tr: combo_tree) -> discriminator.d_counts:
        """C++: moses/moses/scoring/discriminating_bscore.cc:117 discriminator::count

        `unsigned totalc = vct.second.total_count();` (discriminating_
        bscore.cc:130) TRUNCATES `total_count()`'s `count_t` (double)
        return to `unsigned` -- same C++ implicit narrowing `_to_usize`
        (scoring_base.py) documents in full, applied here at the same
        kind of call site (`Counter.total_count()`'s Python equivalent is
        `sum(counter.values())`, since this port's `CTable` counter is a
        plain `collections.Counter`, table.py's own class docstring).
        Observable on a WEIGHTED table (e.g. `weighted.csv`): a row whose
        weights sum to a fractional total has that fraction discarded
        here, before it ever reaches `positive_count`/`negative_count`.

        `pos`/`neg` are C++ LOCAL `double`s (discriminating_bscore.cc:
        128-129, `double pos = sum_true(...)`) but `ctr.true_positive_sum`
        etc. are `score_t` (float) STRUCT MEMBERS (discriminating_bscore.h
        :57-63) -- `ctr.true_positive_sum += pos;` is therefore `float +=
        double`, computed at double precision and narrowed back to
        float32 on EVERY SINGLE STORE, not accumulated across the whole
        ctable in double and narrowed once at the end. This matters: a
        naive Python `+=` across many rows accumulates in double the
        entire way through, which is NOT bit-identical to narrowing after
        every individual addition (measured: diverges from the C++ probe
        starting at the 7th significant digit on `trainingBoolean.dat`).
        Each field is therefore narrowed via `_to_score_t` after every
        single addition below, matching a real float32 struct member.
        """
        ctr = discriminator.d_counts()
        iv = interpreter_visitor(tr)
        for row, counter in self._ctable.items():
            pos = self.sum_true(counter)
            neg = self.sum_false(counter)
            totalc = _to_usize(sum(counter.values()))
            if vertex_equal(iv(list(row)), builtin.logical_true):
                ctr.true_positive_sum = _to_score_t(ctr.true_positive_sum + pos)
                ctr.false_positive_sum = _to_score_t(ctr.false_positive_sum + neg)
                ctr.positive_count = _to_score_t(ctr.positive_count + totalc)
            else:
                ctr.true_negative_sum = _to_score_t(ctr.true_negative_sum + neg)
                ctr.false_negative_sum = _to_score_t(ctr.false_negative_sum + pos)
                ctr.negative_count = _to_score_t(ctr.negative_count + totalc)

        # C++: discriminating_bscore.cc:146-161 -- ONE guard, three printf
        # records, all `%f`.
        if logger().is_fine_enabled():
            logger().fine(
                "counter: tp = %f  fp = %f  tp+fp = %f pos = %f",
                ctr.true_positive_sum,
                ctr.false_positive_sum,
                ctr.true_positive_sum + ctr.false_positive_sum,
                ctr.positive_count,
            )
            logger().fine(
                "counter: fn = %f  tn = %f  fn+tn = %f neg=%f",
                ctr.false_negative_sum,
                ctr.true_negative_sum,
                ctr.false_negative_sum + ctr.true_negative_sum,
                ctr.negative_count,
            )
            logger().fine(
                "counter: tp+fn = %f  fp+tn = %f  total = %f",
                ctr.true_positive_sum + ctr.false_negative_sum,
                ctr.false_positive_sum + ctr.true_negative_sum,
                ctr.positive_count + ctr.negative_count,
            )
        return ctr

    def counts(self, tr: combo_tree) -> list[discriminator.d_counts]:
        """C++: moses/moses/scoring/discriminating_bscore.cc:165 discriminator::counts

        "Like count but do the counting for each datapoint (each row of
        the ctable)." (discriminating_bscore.h:67-69's own comment.) Same
        `unsigned totalc` truncation as `count` above (discriminating_
        bscore.cc:178).
        """
        ctr_seq: list[discriminator.d_counts] = []
        iv = interpreter_visitor(tr)
        for row, counter in self._ctable.items():
            pos = self.sum_true(counter)
            neg = self.sum_false(counter)
            totalc = _to_usize(sum(counter.values()))
            ctr = discriminator.d_counts()
            if vertex_equal(iv(list(row)), builtin.logical_true):
                ctr.true_positive_sum = pos
                ctr.false_positive_sum = neg
                ctr.positive_count = totalc
            else:
                ctr.true_negative_sum = neg
                ctr.false_negative_sum = pos
                ctr.negative_count = totalc
            ctr_seq.append(ctr)
        return ctr_seq


class discriminating_bscore(bscore_ctable_base, discriminator):
    """C++: moses/moses/scoring/discriminating_bscore.h:95 discriminating_bscore

    "Base class for precision, recall, senstivity, specificty, F-score,
    etc. type discriminator scorers." (discriminating_bscore.h:91-93's own
    doc comment.)
    """

    def __init__(
        self,
        ctable: CTable,
        min_threshold: float = 0.5,
        max_threshold: float = 1.0,
        hardness: float = 1.0,
    ) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:202 discriminating_bscore

        `min_threshold`/`max_threshold`/`hardness` are `float` C++
        PARAMETERS -- narrowed at the call boundary via `_to_score_t`
        before any comparison uses them, same convention
        `set_complexity_coef` (scoring_base.py) already established.
        `_ctable_usize(ct.uncompressed_size())` (discriminating_bscore.cc:
        207) is discriminating_bscore's OWN, separately-declared member
        that SHADOWS `bscore_ctable_base`'s -- both truncate identically
        (see `_to_usize`'s docstring, scoring_base.py) so this port simply
        overwrites the inherited `self._ctable_usize` with the same
        (redundant but faithful) computation, matching C++ shadowing's
        observable effect exactly.

        `_size = 2` (discriminating_bscore.cc:214) OVERWRITES
        `bscore_ctable_base.__init__`'s own `_size = ctable.size()` --
        "XXX Currently, this scorer does not return a true behavioral
        score" (discriminating_bscore.cc:213's own comment) is reproduced
        verbatim: every subclass below re-overwrites `_size` again in its
        OWN constructor to its own bscore's real length, matching C++'s
        member-initializer-then-body-reassignment order exactly.
        """
        bscore_ctable_base.__init__(self, ctable)
        discriminator.__init__(self, self._wrk_ctable)

        self._ctable_usize = _to_usize(ctable.uncompressed_size())
        self._min_threshold = _to_score_t(min_threshold)
        self._max_threshold = _to_score_t(max_threshold)
        self._hardness = _to_score_t(hardness)
        self._full_bscore = True
        self._size = 2

        # C++: discriminating_bscore.cc:216-219, before the hardness check.
        logger().info(
            "Discriminating scorer, hardness = %f, min_threshold = %f, max_threshold = %f",
            self._hardness,
            self._min_threshold,
            self._max_threshold,
        )

        if not (0.0 <= self._hardness):
            raise AssertionError("Hardness must be non-negative")
        if 0.0 < self._hardness:
            if not (0.0 < self._min_threshold <= self._max_threshold):
                raise AssertionError(
                    "Discriminating scorer, invalid thresholds.  "
                    "The minimum threshold must be greater than zero, "
                    "and the maximum threshold must be greater "
                    "than or equal to the minimum threshold.\n"
                )

        if self._output_type == type_node.boolean_type:
            self._max_output: score_t = 1.0
            self._min_output: score_t = 0.0
        else:
            self._max_output = very_worst_score
            self._min_output = very_best_score
            for counter in self._wrk_ctable.values():
                for v in counter:
                    if is_contin(v):
                        val = get_contin(v)
                        self._max_output = max(self._max_output, val)
                        self._min_output = min(self._min_output, val)

        # C++: discriminating_bscore.cc:251-252.
        logger().info(
            "Discriminating scorer, min_output = %f, max_output = %f",
            self._min_output,
            self._max_output,
        )

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/discriminating_bscore.cc:262 best_possible_bscore

        "The best_possible_bscore is a pair of (best_score, fix_penalty)."
        (discriminating_bscore.cc:255-261's own comment, reproduced in
        full at the top of that function.) `max_vary` (a
        `std::multimap<double, pos_neg_cnt>`, discriminating_bscore.cc:
        273) is REPLACED here by a plain Python `list` of `(vary, pos,
        neg, count)` tuples sorted ASCENDING by `vary` -- `std::multimap`
        iterates in ascending key order by construction, and the C++ loop
        immediately below walks it `| reversed` (descending); `list.sort()`
        + Python's own reversed iteration reproduces exactly that access
        pattern. Python's `sort()` is stable; ties in `vary` therefore
        preserve INSERTION order (ascending original ctable-row order)
        rather than `std::multimap`'s unspecified-but-consistent
        equal-range order -- both are valid orderings under `vary`, and
        since `>` (strict) decides `best_sc` updates below (never `>=`),
        a tie in `vary` can only change WHICH tied candidate is examined
        first, never whether a strictly-better score is found. Untested:
        no fixture case in tests/fixtures/scorers.json has two ctable rows
        with an EXACTLY tied `vary`.
        """
        max_vary: list[tuple[score_t, score_t, score_t, int]] = []
        for counter in self._wrk_ctable.values():
            pos = self.sum_true(counter)
            neg = self.sum_false(counter)
            # unsigned total = c.total_count() (discriminating_bscore.cc:282)
            # -- same truncation as discriminator.count/counts above.
            total = _to_usize(sum(counter.values()))
            vary = self.get_variable(pos, neg, total)
            # C++: discriminating_bscore.cc:286-287.
            logger().fine(
                f"Disc: total={total} pos={ostream_double(pos)}"
                f" neg={ostream_double(neg)} vary={ostream_double(vary)}"
            )
            max_vary.append((vary, pos, neg, total))
        max_vary.sort(key=lambda t: t[0])

        acc_cnt = 0  # unsigned acc_cnt = 0 (discriminating_bscore.cc:300)
        acc_pos: score_t = 0.0
        acc_neg: score_t = 0.0
        best_sc: score_t = very_worst_score
        best_vary: score_t = 0.0
        best_fixation_penalty: score_t = 0.0
        # `best_fixed` (discriminating_bscore.cc:305) exists in C++ only to
        # feed the fourth INFO record below; tracked again now that the
        # record is emitted. It feeds no decision.
        best_fixed: score_t = 0.0

        # C++: discriminating_bscore.cc:308.
        logger().fine("Disc: min_thresh=" + ostream_double(self._min_threshold))

        for _vary, pos, neg, total in reversed(max_vary):
            acc_pos = _to_score_t(acc_pos + pos)
            acc_neg = _to_score_t(acc_neg + neg)
            acc_cnt += total

            vary = self.get_variable(acc_pos, acc_neg, acc_cnt)
            fixed = self.get_fixed(acc_pos, acc_neg, acc_cnt)
            fixation_penalty = self.get_threshold_penalty(fixed)
            sc = _to_score_t(vary + fixation_penalty)

            if sc > best_sc:
                best_sc = sc
                best_vary = vary
                best_fixed = fixed
                best_fixation_penalty = fixation_penalty

            # C++: discriminating_bscore.cc:329-330, AFTER the update block
            # and before the halt test.
            logger().fine(
                f"Disc: vary={ostream_double(vary)} fixed={ostream_double(fixed)}"
                f" score={ostream_double(sc)}"
            )

            if self._min_threshold <= fixed:
                break

        # C++: discriminating_bscore.cc:337-340, four separate records.
        logger().info("Discriminating scorer, best score = %f", best_sc)
        logger().info("Discriminating scorer, variable component of best score = %f", best_vary)
        logger().info("Discriminating scorer, fixed component of best score = %f", best_fixed)
        logger().info(
            "Discriminating scorer, fixation penalty of best score = %f", best_fixation_penalty
        )

        return [best_vary, best_fixation_penalty]

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:345 discriminating_bscore::min_improv

        `1.0 / _ctable_usize` promotes the `size_t` denominator to
        `double` and divides -- IEEE-754 `1.0/0.0 == +inf`, not a crash,
        for the (real, see `_to_usize`'s docstring) case of a fractional-
        weight table whose `_ctable_usize` truncates to exactly 0. Ported
        via `_ieee_div` (scoring_base.py), not plain Python `/`, which
        raises `ZeroDivisionError` even for two floats.
        """
        return _to_score_t(_ieee_div(1.0, self._ctable_usize))

    def get_threshold_penalty(self, value: score_t) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:352 get_threshold_penalty

        "Note that the logarithm is always negative, so this method
        always returns a value that is zero or negative."
        (discriminating_bscore.cc:350-351's own comment.) The
        `0.99999999 < dst` short-circuit (discriminating_bscore.cc:363)
        avoids `log(1.0 - dst)` blowing up to `-inf` for a `dst` extremely
        close to (but not exactly) 1.0 -- ported with the exact same
        literal threshold and exact same substitute value
        (`-hardness * 18`, an arbitrary-but-fixed "very negative" stand-in
        classic itself chose, NOT `-inf`).

        DOUBLE-ROUNDING NOTE: `value / _min_threshold` and `(value -
        _max_threshold)` are each a `score_t / score_t` or `score_t -
        score_t` operation in C++ -- BOTH operands already float, so the
        operation itself happens at NATIVE float32 precision (rounded
        ONCE), and only the SURROUNDING `1.0 - (...)` (a double literal
        forcing promotion) rounds a second time at the assignment to
        `dst`. Computing the whole `1.0 - value / _min_threshold`
        expression in one shot at double precision and narrowing only
        once would round the intermediate DIFFERENTLY than C++'s two
        separate roundings in the general case -- measured directly: on
        `trainingBoolean.dat`, `recall_bscore(ct, 0.8, 1.0, 1.0)` scoring
        `or($1 $2)` gives a threshold penalty of `-0.18232150375843048`
        with the two-step narrowing below, versus `-0.18232154846191406`
        (WRONG) with a single narrow at the end -- both against the same
        C++ probe run. Each branch below therefore narrows the
        both-operands-score_t sub-expression FIRST, then narrows the
        double-promoted `1.0 - (...)`/division AGAIN, matching C++'s two
        distinct rounding points exactly.
        """
        if self._hardness > 0:
            dst: score_t = 0.0
            if value < self._min_threshold:
                ratio = _to_score_t(value / self._min_threshold)  # score_t / score_t
                dst = _to_score_t(1.0 - ratio)
            if self._max_threshold < value:
                numer = _to_score_t(value - self._max_threshold)  # score_t - score_t
                dst = _to_score_t(numer / (1.0 - self._max_threshold))
            if 0.99999999 < dst:
                return _to_score_t(-self._hardness * 18)
            return _to_score_t(self._hardness * math.log(1.0 - dst))
        return 0.0

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:370 set_complexity_coef(unsigned)
        and discriminating_bscore.cc:387 set_complexity_coef(score_t)

        "Over-ride the default complexity setters, since our scoring is
        normalized to 1.0, unlike other scorers." (discriminating_bscore.h
        :107-108's own doc comment.) Merges both C++ overloads the same
        way `bscore_base.set_complexity_coef` does (scoring_base.py) --
        one positional `ratio`, or two positional `(alphabet_size, p)`.
        Both divide by `self._ctable_usize` where `bscore_base`'s own
        two overloads do not -- this is the entire reason discriminating
        scorers override the base method at all.
        """
        if len(args) == 1:
            (ratio,) = args
            ratio = _to_score_t(ratio)
            self._complexity_coef = 0.0
            if ratio > 0:
                self._complexity_coef = _to_score_t(_ieee_div(1.0, self._ctable_usize * ratio))
            # C++: discriminating_bscore.cc:402-403 -- TWO records, and the
            # second uses `1.0f/`, a float32 reciprocal, where the noise
            # overload at :384 uses `1.0/` in double. Both are streamed at
            # an ostream's six digits, so the difference is invisible here;
            # it is reproduced anyway because guessing it wrong later would
            # cost a puzzle.
            logger().info(f"Discriminating scorer, table size = {self._ctable_usize}")
            ratio_str = ostream_double(_to_score_t(_ieee_div(1.0, self._complexity_coef)))
            logger().info(f"Discriminating scorer, complexity ratio = {ratio_str}")
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                self._complexity_coef = _to_score_t(
                    _ieee_div(discrete_complexity_coef(alphabet_size, p), self._ctable_usize)
                )
            # C++: discriminating_bscore.cc:382-384. "alphabest" is classic's
            # own typo, kept in every scorer that copies this line.
            logger().info(
                f"Discriminating scorer, noise = {ostream_double(p)}"
                f" alphabest size = {alphabet_size}"
                f" complexity ratio = {ostream_double(_ieee_div(1.0, self._complexity_coef))}"
            )
        else:
            raise TypeError(
                "set_complexity_coef expects either (ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )

    def get_fixed(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.h:120 discriminating_bscore::get_fixed

        Pure virtual in C++ (`= 0`) -- every concrete subclass below
        overrides this.
        """
        raise NotImplementedError

    def get_variable(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.h:121 discriminating_bscore::get_variable

        Pure virtual in C++ (`= 0`) -- every concrete subclass below
        overrides this.
        """
        raise NotImplementedError


class recall_bscore(discriminating_bscore):
    """C++: moses/moses/scoring/discriminating_bscore.h:167 recall_bscore

    "scorer that attempts to maximize recall, while holding precision at
    or above a minimum acceptable value." (discriminating_bscore.h:
    163-165's own doc comment.)
    """

    def __init__(
        self,
        ctable: CTable,
        min_precision: float = 0.8,
        max_precision: float = 1.0,
        hardness: float = 1.0,
    ) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:411 recall_bscore::recall_bscore"""
        super().__init__(ctable, min_precision, max_precision, hardness)
        self._size = 2

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/discriminating_bscore.cc:421 recall_bscore::operator()"""
        ctr = self.count(tr)
        tp_fp = _to_score_t(ctr.true_positive_sum + ctr.false_positive_sum)
        precision = _to_score_t(ctr.true_positive_sum / tp_fp) if 0.0 < tp_fp else 0.0
        tp_fn = _to_score_t(ctr.true_positive_sum + ctr.false_negative_sum)
        recall = _to_score_t(ctr.true_positive_sum / tp_fn) if 0.0 < tp_fn else 0.0

        bs: behavioral_score = [recall]
        precision_penalty = self.get_threshold_penalty(precision)
        bs.append(precision_penalty)
        # C++: discriminating_bscore.cc:438-441. "recall_bcore" is classic's
        # own typo, and the penalty uses `%e`, not `%f`.
        if logger().is_fine_enabled():
            logger().fine(
                "recall_bcore: precision = %f  recall=%f  precision penalty=%e",
                precision,
                recall,
                precision_penalty,
            )
        log_candidate_bscore(tr, bs)
        return bs

    def get_fixed(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:462 recall_bscore::get_fixed"""
        return _to_score_t(pos / cnt)

    def get_variable(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:471 recall_bscore::get_variable"""
        return _to_score_t(pos / self._true_total)


class prerec_bscore(discriminating_bscore):
    """C++: moses/moses/scoring/discriminating_bscore.h:185 prerec_bscore

    "scorer that attempts to maximize precision, while holding recall at
    or above a minimum acceptable value." (discriminating_bscore.h:
    181-183's own doc comment.)
    """

    def __init__(
        self,
        ctable: CTable,
        min_recall: float = 0.5,
        max_recall: float = 1.0,
        hardness: float = 1.0,
    ) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:480 prerec_bscore::prerec_bscore"""
        super().__init__(ctable, min_recall, max_recall, hardness)
        self._size = len(ctable) + 2

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/discriminating_bscore.cc:493 prerec_bscore::operator()

        `_full_bscore` (set `True` by the base constructor, discriminating_
        bscore.h:160/discriminating_bscore.cc's own default) selects the
        per-datapoint branch below -- the `else` branch (discriminating_
        bscore.cc:534-547, `_full_bscore == false`) is UNREACHABLE in this
        port: nothing anywhere ever sets `_full_bscore` to `False` (no
        setter is even ported, matching `_full_bscore` being written only
        once, in the base constructor, in C++ too). Ported anyway as
        `_full_bscore_disabled_branch` (kept private, never called) purely
        for completeness/fidelity, not because any test can reach it.
        """
        bs: behavioral_score = []
        ctr_seq = self.counts(tr)
        pos_total: score_t = 0.0
        for ctr in ctr_seq:
            bs.append(_to_score_t((ctr.true_positive_sum - ctr.false_positive_sum) / 2))
            pos_total += ctr.positive_count

        # `boost::transform(bs, bs.begin(), arg1 / pos_total)`
        # (discriminating_bscore.cc:513) is IEEE division, so `pos_total ==
        # 0.0` gives NaN for every element, not an error. Python's `/`
        # raises ZeroDivisionError instead, which `complexity_based_scorer`
        # then catches and turns into `worst_composite_score` -- a REAL,
        # measurable divergence, found by comparing classic's own
        # `-l DEBUG` log against this port's on `-Hprerec`: classic logs
        # `Score: -nan` for `and(!$1 !$2)` where the port logged
        # `Score: -3.40282e+38`, and the two runs then kept different
        # numbers of candidates (25 vs 23). `_ieee_div` restores C++'s
        # result. `pos_total` is zero whenever no ctable row is positive,
        # which a candidate that selects nothing produces routinely.
        bs = [_to_score_t(_ieee_div(x, pos_total)) for x in bs]
        bs.append(0.5)

        precision = 0.0
        for x in bs:
            precision += x
        precision = _to_score_t(precision)

        recall: score_t = 0.0
        for ctr in ctr_seq:
            recall += ctr.true_positive_sum
        if 0.0 < self._true_total:
            recall = _to_score_t(recall / self._true_total)

        recall_penalty = self.get_threshold_penalty(recall)
        bs.append(recall_penalty)
        # C++: discriminating_bscore.cc:554-557.
        if logger().is_fine_enabled():
            logger().fine(
                "prerec_bscore: precision = %f  recall = %f  recall penalty = %f",
                precision,
                recall,
                recall_penalty,
            )
        log_candidate_bscore(tr, bs)
        return bs

    def get_variable(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:565 prerec_bscore::get_variable"""
        return _to_score_t(pos / cnt)

    def get_fixed(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:571 prerec_bscore::get_fixed"""
        return _to_score_t(pos / self._true_total)


class bep_bscore(discriminating_bscore):
    """C++: moses/moses/scoring/discriminating_bscore.h:207 bep_bscore

    "scorer that attempts to maximize 'BEP', the 'precision-recall
    break-even point'" (discriminating_bscore.h:200-201's own doc
    comment.) See this module's own docstring for why `bep_bscore()`
    constructed with its OWN declared defaults (`min_diff=0.0`) raises.
    """

    def __init__(
        self, ctable: CTable, min_diff: float = 0.0, max_diff: float = 0.5, hardness: float = 1.0
    ) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:580 bep_bscore::bep_bscore"""
        super().__init__(ctable, min_diff, max_diff, hardness)
        self._size = 2

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/discriminating_bscore.cc:590 bep_bscore::operator()"""
        ctr = self.count(tr)
        tp_fp = _to_score_t(ctr.true_positive_sum + ctr.false_positive_sum)
        precision = _to_score_t(ctr.true_positive_sum / tp_fp) if 0.0 < tp_fp else 0.0
        tp_fn = _to_score_t(ctr.true_positive_sum + ctr.false_negative_sum)
        recall = _to_score_t(ctr.true_positive_sum / tp_fn) if 0.0 < tp_fn else 0.0

        bep = _to_score_t((precision + recall) / 2)
        bs: behavioral_score = [bep]

        bep_diff = _to_score_t(abs(precision - recall))
        bep_penalty = self.get_threshold_penalty(bep_diff)
        bs.append(bep_penalty)
        # C++: discriminating_bscore.cc:609-610.
        if logger().is_fine_enabled():
            logger().fine("bep = %f  diff=%f  bep penalty=%e", bep, bep_diff, bep_penalty)
        log_candidate_bscore(tr, bs)
        return bs

    def get_variable(self, pos: score_t, neg: score_t, cnt: int) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:618 bep_bscore::get_variable

        "XXX TODO FIXME is this really correct?" (discriminating_bscore.cc
        :620's own comment) -- reproduced verbatim, not investigated or
        fixed further here.

        `cnt * _true_total` is `unsigned * score_t` -- `cnt` converts to
        `float` (matching `_true_total`'s type, NOT promoted all the way
        to `double`), so the multiplication is a NATIVE float32 op,
        narrowed here before the (double) division that follows, same
        double-rounding reasoning as `get_threshold_penalty`'s own
        docstring. `pos / (...)` is then `score_t / float` -- also a
        native float32 division, narrowed again before being treated as
        the `double best_possible_precision` C++ widens it into.
        """
        denom = _to_score_t(cnt * self._true_total)
        best_possible_precision = _to_score_t(pos / denom)
        best_possible_recall = 1.0 / self._true_total
        return _to_score_t((best_possible_precision + best_possible_recall) / 2)

    def get_fixed(self, pos: score_t, neg: score_t, cnt: int) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:627 bep_bscore::get_fixed

        `pos / cnt` is `score_t / unsigned` -- `cnt` converts to `float`,
        so this is a native float32 division, narrowed before being
        treated as the `double best_possible_precision` C++ widens it
        into (same reasoning as `get_variable` above).
        """
        best_possible_precision = _to_score_t(pos / cnt)
        best_possible_recall = 1.0 if 0.0 < pos else 0.0
        return _to_score_t(abs(best_possible_precision - best_possible_recall))


class f_one_bscore(discriminating_bscore):
    """C++: moses/moses/scoring/discriminating_bscore.h:225 f_one_bscore

    "scorer that attempts to maximize the F_1 score, harmonic mean of the
    precision and recall scores." (discriminating_bscore.h:221-223's own
    doc comment.) See this module's own docstring: THIS CLASS CANNOT BE
    CONSTRUCTED, in classic or in this port -- `__init__` always raises.
    Ported in full anyway (including `__call__`/`get_fixed`/`get_variable`,
    none of them reachable from a real instance) for fidelity, matching
    this port's "port the gate, raise inside it" discipline even where the
    gate is classic's own upstream defect rather than an out-of-scope
    subsystem.
    """

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/discriminating_bscore.cc:645 f_one_bscore::f_one_bscore

        `discriminating_bscore(ct, 0.0, 1.0, 1.0e-20)` -- the base
        constructor's own threshold assert (see this module's docstring)
        ALWAYS fires here, since `min_threshold=0.0` and `hardness=
        1.0e-20 > 0` are both hardcoded, not caller-supplied. This line
        therefore always raises `AssertionError` before `self._size = 1`
        (discriminating_bscore.cc:649) is ever reached.
        """
        super().__init__(ctable, 0.0, 1.0, 1.0e-20)
        self._size = 1

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/discriminating_bscore.cc:652 f_one_bscore::operator()

        NEVER ACTUALLY REACHABLE (see class docstring: `__init__` always
        raises) -- ported for completeness. `2 * precision * recall /
        (precision + recall)` is ENTIRELY native float32 arithmetic in
        C++ (`2` promotes to `float`, not `double`, since it multiplies a
        `score_t`) -- narrowed after every individual operation, matching
        `get_threshold_penalty`'s own double-rounding reasoning.
        """
        ctr = self.count(tr)
        tp_fp = _to_score_t(ctr.true_positive_sum + ctr.false_positive_sum)
        precision = _to_score_t(ctr.true_positive_sum / tp_fp) if 0.0 < tp_fp else 0.0
        tp_fn = _to_score_t(ctr.true_positive_sum + ctr.false_negative_sum)
        recall = _to_score_t(ctr.true_positive_sum / tp_fn) if 0.0 < tp_fn else 0.0

        numer = _to_score_t(_to_score_t(2 * precision) * recall)
        denom = _to_score_t(precision + recall)
        f_one = _to_score_t(numer / denom)
        bs: behavioral_score = [f_one]
        # C++: discriminating_bscore.cc:669-670.
        if logger().is_fine_enabled():
            logger().fine(
                "f_one_bscore: precision = %f recall = %f f_one=%f", precision, recall, f_one
            )
        log_candidate_bscore(tr, bs)
        return bs

    def get_fixed(self, pos: score_t, neg: score_t, cnt: float) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:679 f_one_bscore::get_fixed

        "Quasi-meaningless for f_one, but needed for automatic generation
        of best-possible score." (discriminating_bscore.cc:677-678's own
        comment.)
        """
        return 1.0

    def get_variable(self, pos: score_t, neg: score_t, cnt: int) -> score_t:
        """C++: moses/moses/scoring/discriminating_bscore.cc:686 f_one_bscore::get_variable

        `pos / cnt` is `score_t / unsigned` -- `cnt` converts to `float`,
        a native float32 division, narrowed before being widened into the
        `double best_possible_precision` C++ declares (same reasoning as
        `bep_bscore.get_fixed`). Everything after that is `double`
        arithmetic in C++ (`best_possible_recall`/`f_one` are both
        declared `double`), so no further per-step narrowing is needed
        until the final return.
        """
        best_possible_precision = _to_score_t(pos / cnt)
        best_possible_recall = 1.0
        f_one = 2 * best_possible_precision * best_possible_recall / (
            best_possible_recall + best_possible_precision
        )
        f_one /= self._true_total
        return _to_score_t(f_one)
