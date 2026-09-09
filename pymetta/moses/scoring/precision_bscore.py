"""C++: moses/moses/scoring/precision_bscore.h and precision_bscore.cc

Ports `precision_bscore` (the `-H pre` problem's scorer) and
`precision_conj_bscore` (the `-H pre-conj` problem's scorer -- the
registered name has a HYPHEN, table-problems.h:134; `-Hpre_conj` names
nothing in classic's registry and is refused there and here alike).

## Scope: the boolean, non-time, non-boosting path -- exactly what the
## registered `-H pre` problem actually constructs

`table-problems.cc`'s `pre_table_problem::run()` calls `common_type_setup
(pms, id::boolean_type)` UNCONDITIONALLY before constructing
`precision_bscore` -- despite this class's own header doc comment
extensively describing contin-table support ("This scorer works for both
boolean regression and contin regression..."), the constructor's own
`OC_ASSERT(output_type == id::boolean_type, ...)` and the registered
problem's forced boolean setup together mean the contin path the header
describes is UNREACHABLE from the one place this class is actually wired
up in classic. This port therefore targets exactly the boolean,
non-`time_bscore`, non-boosting path -- the reachable one -- and raises
clear, EARLY gates for everything else, rather than letting an
unavailable dependency fail deep inside a loop:

* `time_bscore=True` or `dispersion_pressure > 0.0`: both require a REAL
  per-observation timestamp (`time2res[tcv.first.timestamp]`/
  `get_timestamp_class(tcv.first.timestamp)`, precision_bscore.cc:242-263)
  that this port's CTable can never supply -- see time_dispersion.py's own
  module docstring for the whole-project reason. `do_score` raises
  `NotImplementedError` up front if either is set, rather than failing
  with an `AttributeError` deep inside its own row loop.
* Ensemble/boosting scoring (`operator()(const scored_combo_tree_set&)`,
  `exact_selection`, `bias_selection`, `get_error(const combo_tree&)`,
  `reset_weights`/`update_weights`): all boosting-only, out of scope for
  the whole port cycle (scoring_base.py's module docstring) -- NOT
  overridden here, so they inherit `bscore_base.call_ensemble`'s raising
  gate. `_return_weighted_score` is consequently NEVER `True` for this
  class in this port (nothing calls `use_weighted_scores()` on a
  `precision_bscore`), so `do_score`'s own `if (_return_weighted_score)`
  branch (precision_bscore.cc:283-289) is a raising gate too, kept for
  structural fidelity though it is provably unreachable.
"""

import math

from pymetta.comboreduct.combo.vertex import (
    argument,
    bool_to_vertex,
    builtin,
    combo_tree,
    negate_vertex,
    vertex_equal,
)
from pymetta.comboreduct.interpreter.interpreter import interpreter_visitor
from pymetta.comboreduct.table.table import CTable, type_node
from pymetta.moses.moses.types import _to_score_t, behavioral_score, score_t, very_worst_score
from pymetta.moses.scoring.scoring_base import _ieee_div, bscore_base, discrete_complexity_coef
from pymetta.moses.scoring.time_dispersion import TemporalGranularity, bscore_ctable_time_dispersion


class precision_bscore(bscore_ctable_time_dispersion):
    """C++: moses/moses/scoring/precision_bscore.h:107 precision_bscore"""

    def __init__(
        self,
        ctable: CTable,
        activation_pressure: float = 1.0,
        min_activation: float = 0.5,
        max_activation: float = 1.0,
        positive: bool = True,
        dispersion_pressure: float = 0.0,
        dispersion_exponent: float = 1.0,
        exact_experts: bool = True,
        bias_scale: float = 1.0,
        time_bscore: bool = False,
        granularity: TemporalGranularity = TemporalGranularity.day,
        disable_debug_log: bool = False,
    ) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:114 precision_bscore

        `min_activation`/`max_activation` are CLAMPED to `[0.0, 1.0]`
        (`clamp(x, 0.0f, 1.0f)`) in the member-initializer list, BEFORE
        `reset_weights()` and the threshold sanity check -- ported in
        that exact order.

        `activation_pressure`/`min_activation`/`max_activation` are C++
        `float` PARAMETERS -- narrowed at the call boundary via
        `_to_score_t`, same convention as `discriminating_bscore`'s own
        constructor.
        """
        super().__init__(ctable, dispersion_pressure, dispersion_exponent, granularity)
        self.min_activation: score_t = _to_score_t(max(0.0, min(1.0, min_activation)))
        self.max_activation: score_t = _to_score_t(max(0.0, min(1.0, max_activation)))
        self.activation_pressure: score_t = _to_score_t(activation_pressure)
        self.positive = positive
        self.bias_scale = bias_scale
        self.exact_experts = exact_experts
        self.time_bscore = time_bscore
        self._disable_debug_log = disable_debug_log

        self.reset_weights()

        self._target = bool_to_vertex(positive)
        self._neg_target = negate_vertex(self._target)

        self.output_type = self._wrk_ctable.get_output_type()
        if self.output_type != type_node.boolean_type:
            raise AssertionError("Error: Precision scorer: output type must be boolean!")

        if not (0.0 <= self.activation_pressure):
            raise AssertionError("activation pressure must be non-negative")
        if 0.0 < self.activation_pressure:
            if not (0.0 < self.min_activation <= self.max_activation):
                raise AssertionError(
                    "Precision scorer, invalid activation bounds.  "
                    "The minimum activation must be greater than zero, "
                    "and the maximum activation must be greater "
                    "than or equal to the minimum activation.\n"
                )

    def sum_outputs(self, counter) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:175 precision_bscore::sum_outputs

        "For boolean tables, sum the total number of 'T' values in the
        output. This sum represents the best possible score i.e. we
        found all of the true values correcty. The 'F's are false
        positives." (precision_bscore.cc:171-174's own doc comment.)
        """
        pos = counter.get(self._target, 0.0)
        neg = counter.get(self._neg_target, 0.0)
        return _to_score_t(0.5 * (pos - neg))

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:180 set_complexity_coef(unsigned, float)
        and precision_bscore.cc:197 set_complexity_coef(score_t)

        Divides by `self._ctable_weight` (a `count_t`/double, never
        truncated) where `discriminating_bscore`'s own two overloads
        divide by `self._ctable_usize` (truncated) instead -- these are
        genuinely different denominators in C++ too, not a copy-paste
        variant of the same method.
        """
        if len(args) == 1:
            (ratio,) = args
            ratio = _to_score_t(ratio)
            self._complexity_coef = 0.0
            if ratio > 0:
                self._complexity_coef = _to_score_t(_ieee_div(1.0, ratio))
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                self._complexity_coef = _to_score_t(
                    _ieee_div(discrete_complexity_coef(alphabet_size, p), self._ctable_weight)
                )
        else:
            raise TypeError(
                "set_complexity_coef expects either (ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )

    def do_score(self, select) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:207 precision_bscore::do_score

        `select` is a `Callable[[tuple], bool]` over a ctable ROW KEY
        (mirroring `std::function<bool(const multi_type_seq&)>`).

        `score_t acto = orow.total_count();` (precision_bscore.cc:238)
        narrows a `count_t` (double) total down to float32 AT THAT
        DECLARATION -- unlike `discriminator.count`'s own `totalc`
        (discriminating_bscore.py), this one is NOT further truncated to
        an integer (`_to_score_t`, not `_to_usize`: `acto` stays a
        `score_t`, matching C++'s own `score_t acto`, not an `unsigned`).
        `sao`/`active` are `score_t` ACCUMULATORS narrowed on every `+=`,
        same "real float32 member" pattern as `discriminator.count`'s own
        docstring.
        """
        if self.time_bscore:
            raise NotImplementedError(
                "precision_bscore.do_score: time_bscore=True requires a real "
                "per-observation timestamp this port's CTable cannot supply -- "
                "see time_dispersion.py's own module docstring"
            )
        if self._pressure > 0.0:
            raise NotImplementedError(
                "precision_bscore.do_score: dispersion_pressure > 0.0 requires "
                "get_timestamp_class, which is unreachable in this port -- "
                "see time_dispersion.py's own module docstring"
            )

        bs: behavioral_score = []
        ac: behavioral_score = []
        active: score_t = 0.0
        sao: score_t = 0.0

        for row, counter in self._wrk_ctable.items():
            sumo: score_t = 0.0
            acto: score_t = 0.0
            selected = select(row)
            if selected:
                sumo = self.sum_outputs(counter)
                sao = _to_score_t(sao + sumo)
                acto = _to_score_t(sum(counter.values()))
                active = _to_score_t(active + acto)

            bs.append(sumo)
            ac.append(acto)

        if self._return_weighted_score:
            # Unreachable in this port -- see module docstring.
            raise NotImplementedError(
                "precision_bscore.do_score: the weighted-score branch is "
                "boosting-only and out of scope for this port"
            )

        if 0.0 < active:
            iac = _to_score_t(_ieee_div(1.0, active))
            bs = [_to_score_t(v * iac) for v in bs]
            bs.append(0.5)
        else:
            bs.append(0.0)

        if not self._return_weighted_score:
            activation = _to_score_t(_ieee_div(active, self._ctable_weight))
            activation_penalty = self.get_activation_penalty(activation)
            bs.append(activation_penalty)

        return bs

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:352 precision_bscore::operator()"""
        iv = interpreter_visitor(tr)

        def selector(row):
            return vertex_equal(iv(list(row)), builtin.logical_true)

        return self.do_score(selector)

    def get_error(self, tr: combo_tree) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:501 precision_bscore::get_error

        `OC_ASSERT(exact_experts, "Precision scorer: inexact experts not
        yet supported!");` (precision_bscore.cc:503-504) -- ported
        verbatim. The `_return_weighted_score` toggle around the
        `operator()` call (precision_bscore.cc:507-509) is meaningless in
        this port (always `False`) but kept for structural fidelity.
        """
        if not self.exact_experts:
            raise AssertionError("Precision scorer: inexact experts not yet supported!")

        self._return_weighted_score = False
        bs = self(tr)
        self._return_weighted_score = True

        accum = 0.0
        for i in range(self._size):
            accum += bs[i]
        return _to_score_t(0.5 - accum)

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:520 precision_bscore::best_possible_bscore

        `max_precisions` (a `std::multimap<contin_t, pair<contin_t,
        count_t>>`, ascending by `contin_t`/double precision) is a plain
        Python `list` of `(precision, sumo, total)` tuples, sorted
        ascending, then walked in reverse -- same convention
        `discriminating_bscore.best_possible_bscore` uses (see its own
        docstring for the tie-order note, which applies identically
        here: ties in `precision` are UNTESTED on the current fixture).
        """
        max_precisions: list[tuple[float, float, float]] = []
        for counter in self._wrk_ctable.values():
            # count_t sumo = sum_outputs(c); -- widening score_t -> double, no narrow
            sumo = self.sum_outputs(counter)
            total = sum(counter.values())
            precision = sumo / total  # double precision = sumo / total;
            max_precisions.append((precision, sumo, total))
        max_precisions.sort(key=lambda t: t[0])

        active = 0.0  # count_t active = 0.0;
        sao: score_t = 0.0
        best_sc: score_t = very_worst_score
        best_precision: score_t = 0.0
        best_activation: score_t = 0.0
        best_activation_penalty: score_t = 0.0

        for _precision, sumo, total in reversed(max_precisions):
            sao = _to_score_t(sao + sumo)
            active += total  # count_t accumulator, no narrow

            precision = _to_score_t(_ieee_div(sao, active) + 0.5)
            activation = _to_score_t(_ieee_div(active, self._ctable_weight))
            activation_penalty = self.get_activation_penalty(activation)
            sc = _to_score_t(precision + activation_penalty)

            if sc > best_sc:
                best_sc = sc
                best_precision = precision
                best_activation = activation
                best_activation_penalty = activation_penalty

            if self._ctable_weight * self.min_activation <= active:
                break

        del best_precision, best_activation, best_activation_penalty  # diagnostics only in C++
        return [best_sc]

    def worst_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:614 precision_bscore::worst_possible_bscore

        "The worst possible score is to fail to identify any of the
        positive rows. This renders TP = 0." (precision_bscore.cc:
        611-613's own comment.)
        """
        return [0.0]

    def get_activation_penalty(self, activation: score_t) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:622 precision_bscore::get_activation_penalty

        "Note that it returns -inf if activation is 0.0 -- but that's OK,
        it won't hurt anything." (precision_bscore.cc:621's own comment.)
        Same two-step, double-rounding-aware narrowing as
        `discriminating_bscore.get_threshold_penalty` -- see that
        method's own docstring -- but with NO `0.99999999 < dst`
        saturation guard: this class's own C++ source does not have one,
        so `log(1.0 - dst)` for `dst >= 1.0` IS reachable here (unlike
        the discriminating-scorer family). C's `log(0.0)` returns
        `-HUGE_VAL` (`-inf`, no exception) and `log(negative)` returns
        `NaN` (a domain error, also no exception) -- Python's `math.log`
        RAISES `ValueError` for both, so both are handled explicitly
        rather than narrowing precision alone via `_to_score_t`.
        """
        if self.activation_pressure > 0:
            dst: score_t = 0.0
            if activation < self.min_activation:
                ratio = _to_score_t(activation / self.min_activation)
                dst = _to_score_t(1.0 - ratio)
            if self.max_activation < activation:
                numer = _to_score_t(activation - self.max_activation)
                dst = _to_score_t(_ieee_div(numer, 1.0 - self.max_activation))
            log_arg = 1.0 - dst
            if log_arg == 0.0:
                log_val = -math.inf
            elif log_arg < 0.0:
                log_val = math.nan
            else:
                log_val = math.log(log_arg)
            return _to_score_t(self.activation_pressure * log_val)
        return 0.0

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:639 precision_bscore::min_improv"""
        return _to_score_t(_ieee_div(1.0, self._ctable_weight))

    def gen_canonical_best_candidate(self) -> combo_tree:
        """C++: moses/moses/scoring/precision_bscore.cc:644 gen_canonical_best_candidate

        "we generate a massive combo tree that is supposed to maximize
        the precision" (precision_bscore.h:141-143's own doc comment.)
        `ptc` is sorted the same way `best_possible_bscore` sorts
        `max_precisions` -- ascending by (double) precision, walked in
        reverse.
        """
        ptc: list[tuple[float, tuple, float]] = []
        for row, counter in self._wrk_ctable.items():
            total = sum(counter.values())
            precision = self.sum_outputs(counter) / total
            ptc.append((precision, row, total))
        ptc.sort(key=lambda t: t[0])

        active = 0.0
        tr = combo_tree(builtin.logical_or)
        head = tr.begin()
        for _precision, row, total in reversed(ptc):
            active += total
            dch = tr.append_child(head, builtin.logical_and)
            idx = 1
            for v in row:
                is_true = vertex_equal(v, builtin.logical_true)
                tr.append_child(dch, argument(idx if is_true else -idx))
                idx += 1
            if self._ctable_weight * self.min_activation <= active:
                break
        return tr

    def reset_weights(self) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:698 precision_bscore::reset_weights"""
        self.wnorm = 1.0
        if self._return_weighted_score:
            self._weights = [1.0] * self._size
        else:
            self._weights = []

    def update_weights(self, rew: list[float]) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:707 precision_bscore::update_weights

        Boosting-only -- never called anywhere in this port.
        """
        if not self._return_weighted_score:
            raise AssertionError("Unexpected use of weights in the bscorer!")
        if len(rew) != self._size:
            raise AssertionError("Unexpected size of weight array!")

        znorm = 0.0
        for i in range(self._size):
            self._weights[i] *= self.wnorm
            if rew[i] < 1.0:
                self._weights[i] *= rew[i]
            znorm += self._weights[i]

        znorm = _ieee_div(float(self._size), znorm)
        self.wnorm = _ieee_div(1.0, znorm)
        for i in range(self._size):
            self._weights[i] *= znorm


class precision_conj_bscore(bscore_base):
    """C++: moses/moses/scoring/precision_bscore.h:202 precision_conj_bscore

    "A bit like precision but tries to maximize the number conjunctions
    after precision. It blatantly assumes that the domain is boolean."
    (precision_bscore.h:198-201's own doc comment.) No time/boosting
    dependency at all -- fully portable.
    """

    def __init__(self, ctable: CTable, hardness: float, positive: bool = True) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:739 precision_conj_bscore"""
        super().__init__()
        self.ctable = ctable
        self.ctable_usize = ctable.uncompressed_size()
        self.hardness = hardness
        self.positive = positive
        self._target = bool_to_vertex(positive)

    def set_complexity_coef(self, *args: float) -> None:
        """C++: moses/moses/scoring/precision_bscore.cc:750 set_complexity_coef(unsigned, float)
        and precision_bscore.cc:767 set_complexity_coef(score_t)
        """
        if len(args) == 1:
            (ratio,) = args
            ratio = _to_score_t(ratio)
            self._complexity_coef = 0.0
            if ratio > 0:
                self._complexity_coef = _to_score_t(_ieee_div(1.0, ratio))
        elif len(args) == 2:
            alphabet_size, p = args
            p = _to_score_t(p)
            self._complexity_coef = 0.0
            if 0.0 < p < 0.5:
                self._complexity_coef = _to_score_t(
                    _ieee_div(discrete_complexity_coef(alphabet_size, p), self.ctable_usize)
                )
        else:
            raise TypeError(
                "set_complexity_coef expects either (ratio) or "
                f"(alphabet_size, p); got {len(args)} arguments"
            )

    def __call__(self, tr: combo_tree) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:776 precision_conj_bscore::operator()"""
        active = 0.0  # count_t
        sao: score_t = 0.0
        iv = interpreter_visitor(tr)
        for row, counter in self.ctable.items():
            if vertex_equal(iv(list(row)), builtin.logical_true):
                sumo = _to_score_t(counter.get(self._target, 0.0))
                totalc = sum(counter.values())
                sao = _to_score_t(sao + sumo)
                active += totalc

        precision: score_t = 1.0
        if 0 < active:
            precision = _to_score_t(_ieee_div(sao, active))
        bs: behavioral_score = [precision]

        conj_n = 0
        head = tr.begin()
        if vertex_equal(head.data, builtin.logical_and):
            conj_n += 1
        sib = head.first_child
        while sib is not None:
            if vertex_equal(sib.data, builtin.logical_and):
                conj_n += 1
            sib = sib.next_sibling
        conj_n_penalty = _to_score_t(self.hardness * (-1.0 / (1.0 + conj_n)))
        bs.append(conj_n_penalty)
        return bs

    def best_possible_bscore(self) -> behavioral_score:
        """C++: moses/moses/scoring/precision_bscore.cc:826 best_possible_bscore"""
        return [1.0, 0.0]

    def min_improv(self) -> score_t:
        """C++: moses/moses/scoring/precision_bscore.cc:831 precision_conj_bscore::min_improv"""
        return 0.0
