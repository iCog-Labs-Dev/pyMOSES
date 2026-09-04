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

* `bscore_ctable_base` (scoring_base.h:235) -- the `ignore_cols`/
  `ignore_rows`/`ignore_rows_at_times`/`get_ctable_usize`/`get_ctable`
  infrastructure for compressing a CTable by feature/row subset during
  search. `ctruth_table_bscore` (bscores.py) extends `bscore_base`
  directly and stores its CTable as a plain attribute instead of through
  this intermediate base -- exactly what task-6-brief.md's interface line
  (`class ctruth_table_bscore(bscore_base)`) specifies.
* `operator()(const scored_combo_tree_set&)` (ensemble/boosting scoring)
  and the whole `_weights`/`use_weighted_scores`/`reset_weights`/
  `update_weights` apparatus behind `_return_weighted_score` -- boosting
  is out of scope for this whole cycle (types.py's module docstring makes
  the same narrowing for `scored_combo_tree_set`). `_return_weighted_score`
  is kept (defaulting to `False`, exactly its C++ default) purely because
  `sum_bscore`'s own branch reads it; it is never set to `True` anywhere
  in this port.
* `ignore_cols`/`ignore_rows`/`ignore_rows_at_times`/`get_ctable_usize`/
  `get_ctable`/`get_error` -- feature-selection and boosting-error
  plumbing, likewise out of scope.
* `contin_complexity_coef` (scoring_base.h:53) -- used only by
  `contin_bscore`, which is not part of this task's ported surface
  (bscores.py ports `ctruth_table_bscore` only, per task-6-brief.md).
"""

import math
from abc import ABC, abstractmethod

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.moses.moses.complexity import complexity_t, tree_complexity
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t


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

        `_weights` (scoring_base.h:230) is not initialized here -- it
        backs the weighted-boosting branch of `sum_bscore`, which this
        port never takes (`_return_weighted_score` is never set `True`;
        see module docstring), so there is no `reset_weights()` call site
        to populate it.
        """
        self._return_weighted_score = False
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
        (scoring_base.h:96's own comment.) Only the unweighted branch
        (`not _return_weighted_score`) is reachable in this port -- see
        module docstring -- so the `_weights`-driven branch
        (scoring_base.cc:141-156) is not ported.

        `boost::accumulate(bs, 0.0)` (scoring_base.cc:139) is a NAIVE
        left-to-right double-precision fold -- deliberately NOT Python's
        builtin `sum()`, which (CPython >=3.12, confirmed on 3.14.7 here:
        `sum([0.1]*10) == 1.0` exactly, where a naive fold gives
        `0.9999999999999999`) uses Neumaier/Kahan-style compensated
        summation and so is systematically MORE accurate than
        `boost::accumulate` -- a second, independent source of
        double-vs-C++ disagreement beyond float32 narrowing, invisible on
        this port's small (<=8-row) fixture corpus but not in general.
        The double-precision accumulator itself is intentionally never
        narrowed per-element (C++'s accumulator is `double`, initialized
        from the `0.0` literal in `boost::accumulate(bs, 0.0)`) -- only
        the final return value is narrowed to `score_t`, exactly once,
        matching `sum_bscore`'s own `score_t` return type.
        """
        total = 0.0
        for x in bs:
            total += x
        return _to_score_t(total)

    def get_complexity(self, tr: combo_tree) -> complexity_t:
        """C++: moses/moses/scoring/scoring_base.h:200 bscore_base::get_complexity

        "Get the appropriate complexity measure for the indicated combo
        tree. By default, this is the tree complexity" (scoring_base.h:
        197-199's own comment) -- `ctruth_table_bscore` does not override
        this, so classic's own default (calling `tree_complexity`) is
        exactly what every case in this port also uses.
        """
        return tree_complexity(tr)

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

        NOT ported: the `logger().info()` diagnostic line each C++
        overload ends with (scoring_base.cc:58-60, 69) -- no logging
        surface exists in this port.

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
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                # discrete_complexity_coef already returns a score_t-
                # narrowed value (see its own docstring) -- no second
                # narrowing needed here.
                self._complexity_coef = discrete_complexity_coef(alphabet_size, p)
        else:
            raise TypeError(
                "set_complexity_coef expects either (complexity_ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )
