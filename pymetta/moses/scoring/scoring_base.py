"""C++: moses/moses/scoring/scoring_base.h and scoring_base.cc

`bscore_base` -- "Abstract base class for behavioral scoring. A behavioral
score is a vector of scores, one per sample of a dataset." (scoring_base.h:
55-56's own doc comment, reproduced verbatim.) Narrowed to exactly the
surface task-6-brief.md's Interfaces section lists (`__call__`,
`get_complexity`, `best_possible_bscore`, `min_improv`,
`set_complexity_coef`, `get_complexity_coef`) plus the handful of other
methods `ctruth_table_bscore`/`behave_cscore` genuinely call
(`sum_bscore`, `worst_possible_bscore`, `size`) -- see each method's own
docstring for its C++ line.

NOT ported, all deliberately out of scope for this task:

* `ignore_cols`/`ignore_rows`/`ignore_rows_at_times` -- the feature/row
  subsetting infrastructure `bscore_ctable_base` (below) uses to compress
  a CTable during feature selection. `filtered_preserve_idxs`, the CTable
  method the real C++ bodies call, does not exist in this port's `CTable`
  (table.py) -- feature selection itself is an unscheduled subsystem
  (REMAINING-WORK.md Sec 3d). Ported as raising gates on
  `bscore_ctable_base` (below), not silently inherited as bscore_base's
  own harmless no-op default -- silently doing nothing would misrepresent
  a base class whose whole point is to actually compress the table.
* `get_complexity(const scored_combo_tree_set&)` (scoring_base.h:204) --
  it averages `sct.get_weight()` (scoring_base.cc:96-107), for ensembles.
  CORRECTED: this bullet used to justify the omission with "the boosting
  weight this port's own `scored_combo_tree` (types.py) does not carry".
  That reason is gone -- `scored_combo_tree` carries `weight` now
  (types.h:274,291,295) -- but the overload itself is still unported, and
  `ensemble.py` still keeps its real weights in a private side dict
  rather than on the `scored_combo_tree`, so a port of this overload
  would have to be rewired along with it.
* `contin_complexity_coef` (scoring_base.h:53) -- used only by
  `contin_bscore`, which is not part of this task's ported surface
  (bscores.py ports `ctruth_table_bscore` only, per task-6-brief.md).

`bscore_ctable_base` (scoring_base.h:236) -- the base class
`discriminating_bscore`/`select_bscore`/`bscore_ctable_time_dispersion`
(discriminating_bscore.py/select_bscore.py/time_dispersion.py) all extend
-- IS now ported below, by the P2 scorer-family task: `ctruth_table_bscore`
(bscores.py) still extends `bscore_base` directly per its own
task-6-brief.md interface line, so this class gains its first real users
only now.

The `_weights`/`use_weighted_scores`/`reset_weights`/`update_weights`
apparatus behind `_return_weighted_score` was ALSO believed out of scope
for this whole cycle ("boosting only") until the P2 scorer-family task
found that assumption false: `select_bscore`'s own constructor
(select_bscore.cc:80-82) sets `_return_weighted_score = true`
UNCONDITIONALLY, in ORDINARY (non-boosting) use -- the comment there is
explicit: "This is needed even if not boosted, as otherwise the
ignore_cols(), ignore_rows() methods strip out rows, and cause
mis-matches during scoring." So `sum_bscore`'s weighted branch
(scoring_base.cc:141-156) IS reachable in this port after all, and is
ported below alongside `reset_weights`/`update_weights`/
`use_weighted_scores`. `precision_bscore` (precision_bscore.py), by
contrast, only sets `_return_weighted_score = true` via the still-out-of-
scope `use_weighted_scores()` boosting entry point (never called anywhere
in this port), so its own weighted-branch code in `do_score` stays
unreachable/untested, exactly as originally assumed for this whole
apparatus.

S10 Task 0 ports `operator()(const scored_combo_tree_set&)`
(scoring_base.h:66) -- as `call_ensemble` below, see its own docstring for
the naming -- now that `scored_combo_tree_set` (types.py) exists to type
its parameter. Its C++ body is an unconditional `OC_ASSERT(false, ...)`:
no bscorer in this port overrides it (boosting stays out of scope), so
this is a straight "port the gate, raise inside it" case, the same
discipline the S10 metapopulation plan uses for every other P2 subsystem.
"""

import math
from abc import ABC, abstractmethod

from pymetta.comboreduct.combo.iostream_combo import tree_to_stream_str
from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.table.table import CTable
from pymetta.moses.moses.complexity import complexity_t, tree_complexity
from pymetta.moses.moses.types import (
    _to_score_t,
    behavioral_score,
    ostream_behavioral_score,
    score_t,
    scored_combo_tree_set,
)
from pymetta.util.logger import logger, ostream_double


def _cpp_reciprocal(x: float) -> float:
    """C++: moses/moses/scoring/scoring_base.cc:50 set_complexity_coef

    Not a C++ symbol: `1.0/_complexity_coef` in IEEE-754 `double`. Both
    `set_complexity_coef` overloads stream exactly that, and both leave
    `_complexity_coef` at `0.0` whenever their singularity guard rejects the
    argument -- so classic really does divide by zero here, and really does
    log `inf`. Python raises `ZeroDivisionError` for the same expression, so
    the IEEE result is produced explicitly.

    The sign is preserved because C++'s is: `-0.0` gives `-inf`. A
    `_complexity_coef` of `-0.0` is not reachable through either overload
    (both assign the literal `0.0`), but reproducing the sign costs nothing
    and guessing it wrong later would cost a puzzle.
    """
    if x == 0.0:
        return math.copysign(math.inf, x)
    return 1.0 / x


def discrete_complexity_coef(alphabet_size: int, p: float) -> score_t:
    """C++: moses/moses/scoring/scoring_base.cc:39 discrete_complexity_coef

    "Used to define the complexity scoring component given that p is the
    probability of having an observation being wrong ... Note: this
    returns NEGATIVE values." (scoring_base.h:41-45's own doc comment.)

    C++'s own return type is `score_t` (float), so the double-precision
    `-log(alphabet_size) / log(p/(1-p))` computation is narrowed at this
    function's own `return` (scoring_base.cc:41), not deferred to whatever
    later assigns the result to `_complexity_coef` -- see
    `set_complexity_coef` below, whose two branches both simply store this
    already-narrowed value without narrowing it a second time.
    """
    return _to_score_t(-math.log(alphabet_size) / math.log(p / (1 - p)))


class bscore_base(ABC):
    """C++: moses/moses/scoring/scoring_base.h:57 bscore_base"""

    def __init__(self) -> None:
        """C++: moses/moses/scoring/scoring_base.h:59 bscore_base

        `_weights` (scoring_base.h:231, `std::vector<double>`) starts
        empty, matching C++'s default-constructed `std::vector` -- it is
        populated by `reset_weights()`, which every `_return_weighted_
        score`-using constructor calls (see module docstring: this is no
        longer boosting-only, `select_bscore` needs it in ordinary use).
        """
        self._return_weighted_score = False
        self._weights: list[float] = []
        self._complexity_coef: score_t = 0.0
        self._size = 0

    @abstractmethod
    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/scoring_base.h:63 bscore_base::operator()

        "Return the behavioral score for the combo_tree" (scoring_base.h:
        62's own comment) -- pure virtual in C++ (`= 0`), so every
        concrete subclass (here, only `ctruth_table_bscore`, bscores.py)
        must implement it.
        """
        raise NotImplementedError

    def call_ensemble(self, ensemble: scored_combo_tree_set) -> behavioral_score:
        """C++: moses/moses/scoring/scoring_base.h:66 bscore_base::operator()
        (declared here; defined scoring_base.cc:72-77)

        "Return the behavioral score for the ensemble" (scoring_base.h:
        65's own comment) -- NOT `@abstractmethod`/pure virtual in C++
        (contrast `__call__` above, `= 0` at scoring_base.h:63): this is a
        concrete base-class body every bscorer inherits unless it chooses
        to override it, exactly like `worst_possible_bscore`/`min_improv`
        below. Named `call_ensemble`, not `__call__`: Python's `__call__`
        can bind only one signature, and `__call__` above already
        implements the OTHER `operator()` overload (the pure-virtual,
        per-tree one every concrete subclass must provide) -- this is C++'s
        SECOND overload, distinguished there only by parameter type, which
        needs its own name here.

        `OC_ASSERT(false, "Ensemble scoring not implemented for bscorer
        %s", typeid(*this).name())` (scoring_base.cc:74-75) -- unconditional
        in C++ (the `ensemble` parameter is read only for its type, never
        its contents), and no subclass in this port overrides it: boosting/
        ensemble scoring is out of scope for this whole cycle (module
        docstring). Ported as a straight raise, matching this port's
        "port the gate, raise inside it" convention for every other P2
        subsystem the S10 plan names.
        """
        raise NotImplementedError(
            f"Ensemble scoring not implemented for bscorer {type(self).__name__}"
        )

    def size(self) -> int:
        """C++: moses/moses/scoring/scoring_base.h:70 bscore_base::size

        "Return the size (length) of the behavioral_score that operator()
        above would return." (scoring_base.h:68-69's own comment.)
        """
        return self._size

    @abstractmethod
    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/scoring_base.h:75 bscore_base::best_possible_bscore

        "Return the best possible bscore achievable with this fitness
        function." (scoring_base.h:72-74's own comment.) Pure virtual in
        C++, same as `__call__` above.
        """
        raise NotImplementedError

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/scoring_base.cc:81 bscore_base::worst_possible_bscore

        The base-class default is the empty vector ("Can't assert; this
        will fail during ensemble setup", scoring_base.cc:83's own
        comment, left as a comment rather than an assertion) --
        `ctruth_table_bscore` overrides this with a real implementation
        (bscores.py).
        """
        return []

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/scoring_base.h:90 bscore_base::min_improv

        "Return the smallest change in the score which can be considered
        to be an improvement ... Returns 0.0 by default." (scoring_base.h:
        82-89's own comment.)
        """
        return 0.0

    def sum_bscore(self, bs: behavioral_score) -> score_t:
        """C++: moses/moses/scoring/scoring_base.cc:135 bscore_base::sum_bscore

        "Return the (possibly weighted) sum of the behavioral score."
        (scoring_base.h:97's own comment.) BOTH branches are reachable in
        this port -- see module docstring: `select_bscore` sets
        `_return_weighted_score = True` unconditionally, in ordinary
        (non-boosting) use.

        `boost::accumulate(bs, 0.0)` (scoring_base.cc:139, the UNWEIGHTED
        branch) is a NAIVE left-to-right double-precision fold --
        deliberately NOT Python's builtin `sum()`, which (CPython >=3.12,
        confirmed on 3.14.7 here: `sum([0.1]*10) == 1.0` exactly, where a
        naive fold gives `0.9999999999999999`) uses Neumaier/Kahan-style
        compensated summation and so is systematically MORE accurate than
        `boost::accumulate` -- a second, independent source of
        double-vs-C++ disagreement beyond float32 narrowing, invisible on
        this port's small (<=8-row) fixture corpus but not in general.
        The double-precision accumulator itself is intentionally never
        narrowed per-element (C++'s accumulator is `double`, initialized
        from the `0.0` literal in `boost::accumulate(bs, 0.0)`) -- only
        the final return value is narrowed to `score_t`, exactly once.

        The WEIGHTED branch (scoring_base.cc:141-156) is different in
        kind, not just in formula: `score_t res = 0.0;` declares the
        accumulator itself as `score_t` (float), so EVERY `+=` narrows,
        not just the final return -- narrowed here via `_to_score_t`
        inside each loop iteration, matching a real float32 variable
        being repeatedly written. `_weights[i] * bs[i]` is a
        `double * score_t` product (C++ promotes `bs[i]` to `double` for
        the multiply), computed here at full Python-`float` (double)
        precision before that per-iteration narrow -- exactly one
        narrowing point per term, matching C++'s one truncating store per
        loop iteration. `OC_ASSERT(_size <= bsz, ...)` (scoring_base.cc:
        146) is ported as `AssertionError`, this port's convention for a
        direct `OC_ASSERT(false, ...)` (see docs/CARRY-FORWARD.md Sec 5).
        """
        if not self._return_weighted_score or self._size == 0 or len(self._weights) == 0:
            total = 0.0
            for x in bs:
                total += x
            return _to_score_t(total)

        bsz = len(bs)
        if not (self._size <= bsz):
            raise AssertionError("Behavioral score too small!")

        res: score_t = 0.0
        i = 0
        for i in range(self._size):
            res = _to_score_t(res + self._weights[i] * bs[i])
        for i in range(self._size, bsz):
            res = _to_score_t(res + bs[i])
        return res

    def use_weighted_scores(self) -> None:
        """C++: moses/moses/scoring/scoring_base.h:95 bscore_base::use_weighted_scores

        "Return weighted scores instead of flat scores. The weighted
        scores are needed by the boosting algorithms" (scoring_base.h:
        92-94's own comment) -- inline in C++ (`{ _return_weighted_score =
        true; }`). Never called anywhere in THIS port (boosting itself is
        out of scope) -- kept because it is the one entry point besides
        `select_bscore`'s own constructor that can set
        `_return_weighted_score`, and `precision_bscore.do_score`'s own
        `_return_weighted_score` branch (precision_bscore.py) is written
        against it existing, even though nothing calls it.
        """
        self._return_weighted_score = True

    def reset_weights(self) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:163 bscore_base::reset_weights

        "Reset the weights to a uniform distribution." (scoring_base.h:
        119's own comment.) `select_bscore` OVERRIDES this with its own
        version (select_bscore.py); this base implementation is what
        `discriminating_bscore`/`precision_bscore` would inherit if either
        ever called it (neither does in this port -- boosting-only for
        them, see module docstring).
        """
        if self._return_weighted_score:
            self._weights = [1.0] * self._size
        else:
            self._weights = []

    def update_weights(self, rew: list[float]) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:171 bscore_base::update_weights

        Only the `#if 1` branch (scoring_base.cc:177-180, "Simple
        rescaling, without normalization") is ported -- the `#else`
        branch (scoring_base.cc:182-190) is dead code, never compiled
        (matching this port's own convention of porting only what the
        active preprocessor branch actually builds).
        """
        if not self._return_weighted_score:
            raise AssertionError("Unexpected use of weights in the bscorer!")
        if len(rew) != self._size:
            raise AssertionError("Unexpected size of weight array!")
        for i in range(self._size):
            self._weights[i] *= rew[i]

    def get_complexity(self, tr: combo_tree) -> complexity_t:
        """C++: moses/moses/scoring/scoring_base.h:200 bscore_base::get_complexity

        "Get the appropriate complexity measure for the indicated combo
        tree. By default, this is the tree complexity" (scoring_base.h:
        197-199's own comment) -- `ctruth_table_bscore` does not override
        this, so classic's own default (calling `tree_complexity`) is
        exactly what every case in this port also uses.
        """
        return tree_complexity(tr)

    def get_complexity_ensemble(self, ensemble) -> complexity_t:
        """C++: moses/moses/scoring/scoring_base.cc:96 bscore_base::get_complexity
        (the `const scored_combo_tree_set&` overload)

        A weight-normalised mean of the members' tree complexities. Named
        `get_complexity_ensemble` rather than overloading `get_complexity`
        above: Python has no overload resolution, and dispatching on
        `isinstance(x, combo_tree)` inside one method would make the
        C++ line each branch corresponds to harder to cite, not easier.

        The empty case returns 0.0 BEFORE the division (scoring_base.cc:
        98) -- otherwise `norm` is zero. Note `cpxy`/`norm` are `double`
        here even though `complexity_t` is an int; classic's own comment
        at :108 calls that out ("XXX FIXME complexity_t should be a double
        not an int") and the `floor(x + 0.5)` round-half-up is what
        converts, NOT C's truncating cast, so a mean of 2.5 gives 3.
        """
        if not ensemble:
            return 0.0

        cpxy = 0.0
        norm = 0.0
        for sct in ensemble:
            w = sct.get_weight()
            cpxy += w * tree_complexity(sct.get_tree())
            norm += w
        return complexity_t(math.floor(cpxy / norm + 0.5))

    def get_complexity_coef(self) -> score_t:
        """C++: moses/moses/scoring/scoring_base.h:214 bscore_base::get_complexity_coef

        "Return the complexity coefficient. This is used to obtain the
        complexity penalty for the score ... penalty =
        get_complexity_coef() * get_complexity(tree)" (scoring_base.h:
        206-209's own comment).
        """
        return self._complexity_coef

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:50 bscore_base::set_complexity_coef
        and moses/moses/scoring/scoring_base.cc:63 bscore_base::set_complexity_coef(score_t)

        Merges both C++ overloads into one method dispatching on
        argument count -- one positional `complexity_ratio` (the
        scoring_base.cc:63-70 overload), or two positional
        `(alphabet_size, p)` (the scoring_base.cc:50-61 overload) --
        the same convention `ITable.__init__`/`boolean_interpreter.
        __call__` use elsewhere in this port for merging C++ overloads
        that classic itself distinguishes only by parameter list.

        Both branches share the same "0.0 unless a strictly-positive
        condition holds" structure classic uses, in each case to dodge a
        singularity in the underlying formula (the `p==0.0` and `p==0.5`
        cases for the two-argument form; `complexity_ratio<=0.0` for the
        one-argument form -- scoring_base.cc:52-53's own comment: "Both
        p==0.0 and p==0.5 are singularities in the forumla [sic]. See the
        explanation in the comment above ctruth_table_bscore.").

        Each overload ends with a `logger().info()` line
        (scoring_base.cc:58-60 and :69) that streams `1.0/_complexity_coef`
        -- note that is the RECIPROCAL of the stored coefficient, computed
        in `double` from the `score_t` member, and that a zero coefficient
        therefore prints `inf` rather than dividing by zero at any earlier
        point. Both are reproduced below at the same point in each branch.

        Both `complexity_ratio` (one-arg form) and `p` (two-arg form) are
        `score_t`/`float`-typed C++ PARAMETERS (`set_complexity_coef
        (score_t complexity_ratio)`, `set_complexity_coef(unsigned
        alphabet_size, float p)`) -- narrowed to float32 at the C++ call
        boundary itself, BEFORE either the `>`/`<` singularity checks or
        the reciprocal/`discrete_complexity_coef` call ever see them. This
        port narrows each argument first, for the same reason, rather than
        only narrowing the final `_complexity_coef` assignment.
        """
        if len(args) == 1:
            (complexity_ratio,) = args
            complexity_ratio = _to_score_t(complexity_ratio)
            self._complexity_coef = 0.0
            if complexity_ratio > 0.0:
                self._complexity_coef = _to_score_t(1.0 / complexity_ratio)
            # C++: scoring_base.cc:69. A bare `<<` on a fresh stream, so the
            # double is rendered at an ostream's DEFAULT precision of 6, not
            # at `io_score_precision`.
            ratio = ostream_double(_cpp_reciprocal(self._complexity_coef))
            logger().info(f"BScore complexity ratio = {ratio}")
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                # discrete_complexity_coef already returns a score_t-
                # narrowed value (see its own docstring) -- no second
                # narrowing needed here.
                self._complexity_coef = discrete_complexity_coef(alphabet_size, p)
            # C++: scoring_base.cc:58-60. "alphabest" is classic's own typo.
            logger().info(
                f"BScore noise = {ostream_double(p)}"
                f" alphabest size = {alphabet_size}"
                f" complexity ratio = {ostream_double(_cpp_reciprocal(self._complexity_coef))}"
            )
        else:
            raise TypeError(
                "set_complexity_coef expects either (complexity_ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )


def log_candidate_bscore(tr: combo_tree, bs: behavioral_score) -> None:
    """C++: moses/moses/scoring/scoring_base.h:339 log_candidate_bscore

    The shared "here is the tree and the behavioural score it just got"
    record, called at the tail of sixteen `operator()` bodies across
    bscores.cc, precision_bscore.cc and discriminating_bscore.cc. One
    record, two physical lines: the `"\n"` is inside the message, so
    `Logger::log`'s own newline terminates the second line.

    The `is_fine_enabled()` guard is C++'s own and is what keeps the tree
    from being rendered on every single candidate at lower levels -- the
    one place in this port where a guard is worth more than the record.
    """
    if logger().is_fine_enabled():
        logger().fine(
            "Evaluate candidate: "
            + tree_to_stream_str(tr)
            + "\n"
            + f"\tBScore size={len(bs)} bscore: {ostream_behavioral_score(bs)}"
        )


def _copy_ctable(ctable: CTable) -> CTable:
    """Not a ported symbol -- stands in for C++'s implicit copy constructor
    (`CTable _wrk_ctable(_orig_ctable);`/`_all_rows_wrk_ctable(_wrk_ctable)`,
    `bscore_ctable_base`'s own member-initializer list below), the same
    "not a ported symbol" convention `ITable.copy`/`OTable.copy` (table.py)
    use for the same reason: `CTable` is a mutable Python `dict` subclass,
    so `new = old` would alias rather than copy (docs/CARRY-FORWARD.md Sec
    1.2, flagged since S7 and closed here). `CTable` itself defines no
    `.copy()` -- `dict.copy()` would return a bare `dict` (losing the
    subclass, `olabel`/`ilabels`/`_signature`) and only shallow-copy each
    row's `Counter` (aliasing it between old and new) -- neither of which
    is a value copy, so this helper is written from scratch rather than
    delegating to it.

    Each CELL is copied through its own `.copy()`, not rebuilt as a plain
    `collections.Counter`. In C++ a `CTable` cell is a `TimedCounter`
    (`std::map<TimedValue, count_t>`, table.h:496) and the map's copy
    constructor keeps every `(value, timestamp)` observation; rebuilding it
    from the untimed mapping view here silently dropped the timestamps, so
    `_wrk_ctable.get_timestamps()` came back empty on a table loaded WITH
    `--timestamp-feature`. That is exactly what `precision_bscore.do_score`'s
    `time_bscore` / dispersion branches read (precision_bscore.cc:243, :257),
    and it is why they were gated. `TimedCounter.copy()` (table.py) and
    `collections.Counter.copy()` both return their own type, so this single
    call is correct for a real cell and for a hand-built test one alike.
    """
    new = CTable(ctable.olabel, list(ctable.ilabels), ctable.get_signature())
    for row, counter in ctable.items():
        new[row] = counter.copy()
    return new


def _ieee_div(numerator: float, denominator: float) -> float:
    """Not a ported symbol -- emulates C/C++ IEEE-754 floating-point
    division, which NEVER raises: division by zero yields signed
    infinity (a NEGATIVE `nan` for `0/0` on x86-64 -- see the comment
    on that branch), unlike Python's `float.__truediv__`,
    which raises `ZeroDivisionError` even for two `float` operands
    (`1.0 / 0.0` in plain Python -- confirmed, not merely `1.0 // 0` or
    an int divisor). Needed wherever a ported formula divides by a
    table-derived quantity (`_ctable_usize`, `_ctable_weight`) that can
    genuinely be zero or empty -- `_ctable_usize` in particular can be
    EXACTLY 0 for a real, non-degenerate weighted table whose total
    weight truncates below 1.0 (`_to_usize`'s own docstring above; see
    `discriminating_bscore.min_improv`, whose C++ original,
    `1.0 / _ctable_usize`, promotes the `size_t` denominator to `double`
    and divides -- IEEE-754 `1.0 / 0.0 == +inf`, not a crash).
    """
    if denominator != 0:
        return numerator / denominator
    if numerator > 0:
        return math.inf
    if numerator < 0:
        return -math.inf
    # 0/0 is the invalid operation, and x86-64's default result is the
    # "QNaN floating-point indefinite", whose SIGN BIT IS SET -- so C++
    # prints it as `-nan`, not `nan`. Measured with a compiled probe
    # 2026-09-09: `std::signbit(0.0f/0.0f)` is 1, and `std::cout` writes
    # `-nan`. Python's `math.nan` is positive, so the sign is applied
    # explicitly; `discriminating_bscore`'s own `-Hprerec` log line was
    # what exposed the difference.
    return -math.nan


def _to_usize(x: float) -> int:
    """Not a ported symbol -- reproduces an IMPLICIT C++ narrowing
    conversion this port's callers must perform explicitly: every one of
    `bscore_ctable_base::_ctable_usize` (scoring_base.h:332,
    `mutable size_t`) and `discriminating_bscore::_ctable_usize`
    (discriminating_bscore.h:146, its OWN, separately-declared `size_t`
    member that SHADOWS the base class's) is initialized directly from
    `CTable::uncompressed_size()` -- whose C++ return type is `count_t`
    (`double`, table.h:578), not `size_t`. `size_t x = count_t_value;` is
    therefore a TRUNCATING double-to-unsigned narrowing at both call sites
    (`bscore_ctable_base(const CTable&)`, scoring_base.cc:190; and
    `discriminating_bscore`'s own constructor, discriminating_bscore.cc:
    207) -- not a bug callers introduced, an implicit conversion the
    initializer list performs silently. For a WEIGHTED table
    (`uncompressed_size()` sums fractional row weights, table.py's own
    `CTable.uncompressed_size` docstring), this discards the fractional
    part: `12.7` truncates to `12`. Ported verbatim (DO NOT FIX) -- every
    downstream formula that reads `_ctable_usize` (`min_improv`,
    `set_complexity_coef(ratio)`) inherits this truncation exactly as
    classic does. Python's `int()` on a non-negative float truncates
    toward zero exactly like a C++ narrowing conversion from a
    non-negative `double` to `size_t` (`count_t` is never negative in
    practice: it accumulates weights/counts, never subtracts them below
    zero) -- see test_scoring_base.py's dedicated regression test on
    `weighted.csv`, which has a genuinely fractional `uncompressed_size()`.
    """
    return int(x)


class bscore_ctable_base(bscore_base):
    """C++: moses/moses/scoring/scoring_base.h:236 bscore_ctable_base

    "Base class for fitness functions that use a ctable. Provides useful
    table compression." (scoring_base.h:234-235's own doc comment.) The
    table-COMPRESSION half of that promise (`ignore_cols`/`ignore_rows`/
    `ignore_rows_at_times`, driving `_wrk_ctable`/`_all_rows_wrk_ctable`
    down to a genuinely smaller table) is NOT ported -- see this module's
    own docstring's "NOT ported" list -- so those three methods are raising
    gates here, not the real, functional C++ bodies. Every OTHER member
    (`_orig_ctable`/`_wrk_ctable` themselves, `get_ctable_usize`,
    `get_ctable`, `recompute_weight`/`_ctable_weight`) IS ported, because
    `discriminating_bscore`/`select_bscore`/`bscore_ctable_time_dispersion`
    all genuinely need them for their own (non-compression) constructors
    and formulas.
    """

    def __init__(self, ctable: CTable) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:186 bscore_ctable_base::bscore_ctable_base

        `_orig_ctable(ctable)` (scoring_base.cc:188) binds a C++
        REFERENCE -- no copy -- so `self._orig_ctable = ctable` (a plain
        alias) is the faithful port: mutating the caller's original
        `CTable` object after construction is visible through
        `_orig_ctable` in C++ too, via the same reference. `_wrk_ctable
        (_orig_ctable)` and `_all_rows_wrk_ctable(_wrk_ctable)`
        (scoring_base.cc:188-189), by contrast, are genuine C++ VALUE
        copies (`CTable`'s copy constructor) -- ported via `_copy_ctable`
        (this module), per docs/CARRY-FORWARD.md Sec 1.2.

        `_ctable_usize(_orig_ctable.uncompressed_size())`
        (scoring_base.cc:189) truncates -- see `_to_usize`'s own
        docstring (this module) for why and the exact C++ narrowing this
        reproduces.

        `_size = ctable.size()` (scoring_base.cc:191) is the COMPRESSED
        row count (`len(ctable)`), matching `bscore_base._size`'s own
        semantics elsewhere in this port (bscores.py's `ctruth_table_
        bscore.__init__`, e.g., which does exactly `len(ctable)` for the
        same field).
        """
        super().__init__()
        self._orig_ctable = ctable
        self._wrk_ctable = _copy_ctable(ctable)
        self._all_rows_wrk_ctable = _copy_ctable(self._wrk_ctable)
        self._ctable_usize = _to_usize(ctable.uncompressed_size())
        self._size = len(ctable)
        self._ctable_weight: float = 0.0
        self.recompute_weight()

    def recompute_weight(self) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:196 bscore_ctable_base::recompute_weight

        "Sum of all of the weights in the working table. Note that these
        weights can be fractional!" (scoring_base.cc:197-198's own
        comment.) `count_t` (`double`) throughout -- no narrowing, unlike
        `_ctable_usize` above (`_ctable_weight` is declared `count_t`, not
        `size_t`, scoring_base.h:333).
        """
        total = 0.0
        for counter in self._wrk_ctable.values():
            total += sum(counter.values())
        self._ctable_weight = total

    def ignore_cols(self, idxs: set) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:220 bscore_ctable_base::ignore_cols

        NOT functionally ported -- see this module's own docstring's "NOT
        ported" list: the real C++ body calls `CTable::
        filtered_preserve_idxs`, which does not exist in this port's
        `CTable` (table.py). Raises unconditionally rather than silently
        doing nothing, so a caller cannot mistake "no-op" for "there was
        nothing to ignore".
        """
        raise NotImplementedError(
            "bscore_ctable_base.ignore_cols: feature-selection table "
            "compression is out of scope for this port (CTable has no "
            "filtered_preserve_idxs; see scoring_base.py's module docstring)"
        )

    def ignore_rows(self, idxs: set) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:267 bscore_ctable_base::ignore_rows

        NOT functionally ported -- see `ignore_cols` above.
        """
        raise NotImplementedError(
            "bscore_ctable_base.ignore_rows: row-subsetting is out of "
            "scope for this port; see scoring_base.py's module docstring"
        )

    def ignore_rows_at_times(self, timestamps: set) -> None:
        """C++: moses/moses/scoring/scoring_base.cc:290 bscore_ctable_base::ignore_rows_at_times

        NOT functionally ported -- see `ignore_cols` above. Also blocked
        on there being no `TTable`/timestamp representation at all in
        this port (table.py's own module docstring).
        """
        raise NotImplementedError(
            "bscore_ctable_base.ignore_rows_at_times: out of scope for "
            "this port (no TTable/timestamp representation; see "
            "scoring_base.py's and table.py's module docstrings)"
        )

    def get_ctable_usize(self) -> int:
        """C++: moses/moses/scoring/scoring_base.cc:318 bscore_ctable_base::get_ctable_usize"""
        return self._ctable_usize

    def get_ctable(self) -> CTable:
        """C++: moses/moses/scoring/scoring_base.cc:323 bscore_ctable_base::get_ctable"""
        return self._orig_ctable
