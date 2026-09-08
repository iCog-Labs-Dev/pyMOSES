"""C++: moses/moses/scoring/time_dispersion.h and time_dispersion.cc

Ports `TemporalGranularity` and `bscore_ctable_time_dispersion`, the base
class `precision_bscore` (precision_bscore.py) uses for its optional
time-dispersion penalty.

## A whole-project architectural gap, not a time_dispersion-specific one:
## this port's CTable carries no timestamp at all

table.py's own module docstring is explicit: "No `TTable`/`CTableTime`.
Timestamped compression is out of scope." Every observation in classic's
`CTable` is keyed by a `TimedValue` (`vertex` + a `boost::gregorian::date`
timestamp, table.h:472-490); this port's `CTable` counter keys are bare
`vertex` values with no timestamp field at all (table.py's own `CTable`
class docstring). This is a PERMANENT, already-decided scope boundary for
the whole port, not something raised or deferred by this task.

Two consequences, handled differently:

1. `bscore_ctable_time_dispersion.__init__` (time_dispersion.cc:34-54)
   ALWAYS runs -- `precision_bscore` extends this class unconditionally,
   regardless of whether time-dispersion is ever actually used
   (`time_dispersion_pressure` defaults to `0.0`). Its constructor loop
   (`for (iorow : _ctable) for (tcv : iorow.second) timestamp_classes.
   insert(get_timestamp_class(tcv.first.timestamp));`) collects the SET
   of distinct timestamp classes across every observation in the table.
   Classic's OWN `Table::compress()` (table.cc:513, `TimedValue(*out_it)`)
   constructs every `TimedValue` with the DEFAULT-CONSTRUCTED timestamp
   (`boost::gregorian::date()`, "not-a-date-time") whenever no
   `--timestamp-feature` column was loaded -- which is EVERY table this
   port can construct, since timestamp loading itself is out of scope
   (table_io.py's own module docstring: a timestamp column's VALUES are
   discarded at parse time, never captured). So in classic itself,
   restricted to the only input shape this port can ever produce, EVERY
   observation's timestamp is the SAME single sentinel value, and
   `timestamp_classes` always has EXACTLY 0 elements (an empty table) or
   EXACTLY 1 (any non-empty table) -- never more. `_Hmax = log2(1) == 0.0`
   or `log2(0) == -inf` are therefore not approximations of what this
   port's constructor computes: they are the EXACT values classic itself
   computes for the only case this port (or, for that matter, an
   un-flagged classic run) can ever construct. Measured directly
   (tools/cpp_probe/scorers_probe.cc's `timedisp_hmax` mode, a real
   `bscore_ctable_time_dispersion` over `trainingBoolean.dat` and
   `weighted.csv`): `_Hmax == 0.0` in both cases, confirming this.

2. `get_timestamp_class`/`get_time_dispersion_penalty`'s CALLER
   (`precision_bscore::do_score`, precision_bscore.py) needs a REAL
   per-observation timestamp to bucket by whenever `time_bscore=True` or
   `dispersion_pressure > 0.0` -- which this port's CTable can never
   supply. `get_timestamp_class` below is therefore a genuine raising
   gate (there is no reachable input to give it). `get_time_dispersion_
   penalty` itself, however, is PORTED IN FULL and independently tested:
   its signature takes a `CTableTime`-shaped mapping directly (`Mapping[
   Any, Counter[vertex]]`, mirroring `std::map<TTable::value_type,
   Counter<vertex, count_t>>`, table.h:518) rather than deriving one from
   a `CTable` -- so the pure entropy-penalty MATH is fully portable and
   measured exact against classic (`tools/cpp_probe/scorers_probe.cc`'s
   `timedisp_penalty` mode builds a synthetic `CTableTime` directly in
   C++, bypassing the unreachable timestamp-collection pipeline
   entirely), even though NOTHING in this port can yet construct a real
   `CTableTime` from actual data.
"""

import math
from collections.abc import Mapping
from enum import IntEnum
from typing import Any

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.moses.moses.types import _to_score_t, score_t
from pymetta.moses.scoring.scoring_base import _ieee_div, _to_usize, bscore_ctable_base


class TemporalGranularity(IntEnum):
    """C++: moses/moses/scoring/time_dispersion.h:33 TemporalGranularity

    `enum class TemporalGranularity {day, month};` -- `day=0`, `month=1`,
    C++'s own unscoped-within-class-scope enumerator ordinals.
    """

    day = 0
    month = 1


class bscore_ctable_time_dispersion(bscore_ctable_base):
    """C++: moses/moses/scoring/time_dispersion.h:40 bscore_ctable_time_dispersion

    "WARNING: that class does not support dynamical subsampling."
    (time_dispersion.h:36-38's own doc comment.)
    """

    def __init__(
        self,
        ctable,
        time_dispersion_pressure: float = 0.0,
        time_dispersion_exponent: float = 1.0,
        granularity: TemporalGranularity = TemporalGranularity.day,
        multiplier: int = 1,
    ) -> None:
        """C++: moses/moses/scoring/time_dispersion.cc:34 bscore_ctable_time_dispersion

        `OC_ASSERT(_multiplier == 1, "Multiplier other than 1 is not
        supported yet");` (time_dispersion.cc:44) -- ported as an
        unconditional guard, matching the TODO comment above it in C++
        ("multiplier other than 1 is not supported at the moment",
        time_dispersion.h:57-61) exactly: this is classic's OWN
        not-yet-implemented feature, not something this port narrows
        further.

        `_Hmax = log2(timestamp_classes.size())` (time_dispersion.cc:53)
        -- see this module's own docstring for why `timestamp_classes`
        can only ever have 0 or 1 elements when built from THIS port's
        CTable, and why that is not an approximation. `math.log2(0)`
        raises `ValueError` in Python where C++'s `std::log2(0.0)`
        returns `-HUGE_VAL` (`-inf`) -- handled explicitly here rather
        than via `_to_score_t`, which only narrows precision, not this
        library-behaviour difference.
        """
        if multiplier != 1:
            raise AssertionError("Multiplier other than 1 is not supported yet")
        super().__init__(ctable)
        self._granularity = granularity
        self._multiplier = multiplier
        self._pressure = _to_score_t(time_dispersion_pressure)
        self._exponent = _to_score_t(time_dispersion_exponent)

        n_observations = sum(len(counter) for counter in ctable.values())
        n_timestamp_classes = 1 if n_observations > 0 else 0
        self._Hmax: score_t = (
            _to_score_t(math.log2(n_timestamp_classes)) if n_timestamp_classes > 0 else -math.inf
        )

    def get_timestamp_class(self, timestamp: Any) -> Any:
        """C++: moses/moses/scoring/time_dispersion.cc:56 get_timestamp_class

        Genuinely unreachable in this port -- see this module's own
        docstring, point 2. There is no `TTable::value_type` (a
        timestamp) this port can ever hand it; every caller that would
        need one (`precision_bscore.do_score`'s `time_bscore`/
        `_pressure > 0.0` branches) raises before reaching here instead
        (precision_bscore.py), so this exists only for structural
        completeness/fidelity.
        """
        raise NotImplementedError(
            "get_timestamp_class: no TTable/timestamp representation exists in this "
            "port (table.py's module docstring) -- there is no reachable input for "
            "this method to classify"
        )

    def get_time_dispersion_penalty(self, ctt: Mapping[Any, Any]) -> score_t:
        """C++: moses/moses/scoring/time_dispersion.cc:76 get_time_dispersion_penalty

        "the penalty is defined as pressure * (1 - H(time|active) /
        Hmax)^exponent" (time_dispersion.h:89-91's own doc comment.)

        `ctt` mirrors `CTableTime` (table.h:518, `std::map<TTable::
        value_type, Counter<vertex, count_t>>`) -- a mapping from
        timestamp-class to a `Counter` of `builtin.logical_true`/
        `logical_false` observation counts. See this module's own
        docstring for why nothing in this port can yet CONSTRUCT one from
        real data, and why the formula itself is still fully ported and
        tested independent of that gap.

        `unsigned row_count = tio.second.total_count();`
        (time_dispersion.cc:87) truncates a `count_t` (double) total down
        to `unsigned` -- same pattern as `_to_usize` (scoring_base.py) and
        `discriminator.count`'s own `totalc` (discriminating_bscore.py).
        `act_time_dist` is `std::vector<float>` (time_dispersion.cc:84) --
        each `weighted_information(p)` term ($COGUTIL/opencog/util/
        numeric.h:206-208, `p > PROB_EPSILON ? -p*log2(p) : 0`) is
        computed at `FloatT=float` (native float32: BOTH the `log2(p)`
        call and the `-p * (...)` product are float32 operations, each
        narrowed separately here, same double-rounding reasoning as
        discriminating_bscore.py's `get_threshold_penalty`), and the
        running SUM (`entropy`'s own `double res`, numeric.h:229-233) is
        double-precision, narrowed once when assigned to `float H`
        (time_dispersion.cc:96).
        """
        act_time_dist: list[score_t] = []
        total = 0
        for _timestamp_class, counter in ctt.items():
            row_count = _to_usize(sum(counter.values()))
            act_time_dist.append(_to_score_t(counter.get(builtin.logical_true, 0.0)))
            total += row_count

        act_time_dist = [_to_score_t(_ieee_div(p, total)) for p in act_time_dist]

        res = 0.0  # double accumulator, entropy()'s own `double res = 0;`
        for p in act_time_dist:
            if p > 1e-127:  # PROB_EPSILON
                log2p = _to_score_t(math.log2(p))
                term = _to_score_t(-p * log2p)
            else:
                term = 0.0
            res += term
        H: score_t = _to_score_t(res)

        normalized_H = _to_score_t(_ieee_div(H, self._Hmax))
        penalty = _to_score_t(1.0 - normalized_H)
        # std::pow(negative, non-integer) is a C99 domain error that
        # returns NaN (no exception); Python's math.pow raises ValueError
        # for the identical input -- narrowed to NaN here rather than
        # letting a different exception type escape. `penalty` is
        # negative whenever `_Hmax == 0.0` and `H > 0.0` (this port's own
        # single-timestamp-class case, see module docstring): `normalized_
        # H` is then `+inf`, so `penalty = 1.0 - inf == -inf`.
        try:
            distorted_penalty = _to_score_t(math.pow(penalty, self._exponent))
        except ValueError:
            distorted_penalty = math.nan
        scaled_distorted_penalty = _to_score_t(self._pressure * distorted_penalty)
        return scaled_distorted_penalty
