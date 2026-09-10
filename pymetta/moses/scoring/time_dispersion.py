"""C++: moses/moses/scoring/time_dispersion.h and time_dispersion.cc

Ports `TemporalGranularity` and `bscore_ctable_time_dispersion`, the base
class `precision_bscore` (precision_bscore.py) uses for its optional
time-dispersion penalty.

## The CTable now carries real timestamps

Until the timestamp slice this port's `CTable` had no timestamp field at
all, so `get_timestamp_class` was a raising gate and the constructor's
`timestamp_classes` loop was COUNTED rather than run. Both are real now:
`pymetta.comboreduct.table.table` ports `TTable`/`TimedValue`/
`TimedCounter`/`CTableTime`, `loadTable` fills `Table.ttable` from a
`--timestamp-feature` column, and `Table.compressed` keys each
observation on `TimedValue(output, timestamp)`.

Two consequences worth stating, because they decide what the numbers
below mean:

1. When no `--timestamp-feature` was given, EVERY observation carries the
   default not-a-date timestamp (`None` here; `TimedValue(*out_it)`,
   table.cc:513), so `timestamp_classes` has exactly 1 element for any
   non-empty table and `_Hmax == log2(1) == 0.0`. That is not an
   approximation of classic's answer -- it IS classic's answer for that
   input shape, confirmed against a real `bscore_ctable_time_dispersion`
   (`tools/cpp_probe/scorers_probe.cc`'s `timedisp_hmax` mode over
   `trainingBoolean.dat` and `weighted.csv`: `_Hmax == 0.0` in both).
   At `month` granularity classic instead ABORTS on the first
   observation, because `get_timestamp_class` OC_ASSERTs the timestamp is
   a real date; that is reproduced.
2. `get_time_dispersion_penalty` takes a `CTableTime`-shaped mapping
   directly, and `CTable.ordered_by_time()` now builds real ones. Its
   arithmetic was already measured exact against classic
   (`tools/cpp_probe/scorers_probe.cc`'s `timedisp_penalty` mode, which
   builds a synthetic `CTableTime` in C++); what changed is that there is
   now a way to reach it from actual data.

Both CLI flags now RUN. `--time-bscore` and `--time-dispersion-pressure`
were gated in `problem_params.py` because `precision_bscore.do_score`'s
two timestamp branches (precision_bscore.cc:242-263) were not ported, and
because `bscore_ctable_base._wrk_ctable` is built by
`scoring_base._copy_ctable`, which rebuilt each cell as a plain `Counter`
and dropped the observation list. Both are fixed; the gates are gone.

Classic's own `--time-dispersion-pressure` is not in good shape, and this
port is deliberately more defined than it. Measured 2026-09-10, `-Hpre
-q0.5 -w1.0 --timestamp-feature=when --time-dispersion-pressure=0.5 -m300
-W1 -r1`: classic SIGSEGVs (rc 139, no diagnostic) on
`tests/fixtures/table_data/timestamp_bool_clean.csv` while completing on
`timestamp_bool_dup.csv`. The cause is not the dispersion arithmetic --
gdb puts the fault in `representation::get_candidate_rec`'s tree-node
allocation -- but the `ctable_res` that `do_score` builds, a
`TimedValue`-keyed `std::map` under classic's non-strict-weak comparator,
whose corruption damages the heap. See `precision_bscore.do_score`'s own
DIVERGENCE marker.
"""

import math
from collections.abc import Mapping
from datetime import date
from enum import IntEnum
from typing import Any

from pymetta.comboreduct.combo.vertex import builtin
from pymetta.comboreduct.table.table import timed_keys_of
from pymetta.moses.moses.types import _to_score_t, score_t
from pymetta.moses.scoring.scoring_base import _ieee_div, _to_usize, bscore_ctable_base


def _is_odd_integer(x: float) -> bool:
    """No C++ symbol: glibc `powf`'s own `checkint(iy) == 1` test
    (sysdeps/ieee754/flt-32/e_powf.c), i.e. "y is an odd integer". `inf`
    and `nan` are not integers, and `float.is_integer()` says so.
    """
    return x.is_integer() and int(x) % 2 != 0


def _powf(base: float, exponent: float) -> float:
    """No direct C++ symbol: `std::pow(float, float)` as called at
    moses/moses/scoring/time_dispersion.cc:104, i.e. glibc's `powf`.

    Python's `math.pow` is NOT a drop-in for it. Four families differ, and
    `--time-dispersion-exponent` can reach every one of them. All values
    below are MEASURED against classic's own toolchain and pinned in
    `tests/fixtures/scorers.json`'s `powf` section (26 cases,
    `tools/cpp_probe/scorers_probe.cc`'s `powf` mode):

    1. **NaN base.** glibc XORs powf's internal `sign_bias` -- set only
       for a negative base with an ODD-INTEGER exponent -- into the
       result's sign bit, so a NEGATIVE NaN comes back POSITIVE for an odd
       integer exponent and negative otherwise: `powf(-nan, 1) == +nan`,
       `powf(-nan, 3) == +nan`, `powf(-nan, -1) == +nan`, but
       `powf(-nan, 2) == -nan`, `powf(-nan, 0.5) == -nan`,
       `powf(-nan, 1e30) == -nan` (1e30 is an EVEN integer). A positive
       NaN base always gives a positive NaN. Python preserves the sign in
       every one of those.
       This is not a curiosity: `H / _Hmax` is `0.0f / 0.0f` whenever the
       table has exactly one timestamp class, and x86-64's 0/0 is the
       "QNaN floating-point indefinite" whose sign bit is SET (see
       `_ieee_div`, scoring_base.py). The default `--time-dispersion-
       exponent` is 1.0, an odd integer, so classic prints `nan` where a
       sign-preserving port prints `-nan` -- a one-character stdout
       divergence, measured end to end.
    2. **Negative finite base, non-integer exponent.** C99 domain error:
       glibc returns a NEGATIVE NaN (`powf(-3, 0.5)`, `powf(-3, 1.5)`,
       `powf(-0.5, 0.5)` all measured sign-bit-set). Python raises
       `ValueError`.
    3. **Zero base, negative exponent.** C99 pole error: `powf(0, -1)` is
       `+inf`, `powf(-0., -1)` is `-inf` (odd integer exponent), and
       `powf(-0., -2)` is `+inf`. Python raises `ValueError` for all three.
    4. **Overflow.** `powf(2, 1e30)`, `powf(0.5, -1e30)` and
       `powf(-0.5, -1e30)` are all `+inf`. Python raises `OverflowError`.

    `pow(x, +-0) == 1` for every `x`, NaN included, so the exponent-zero
    case is answered before anything else -- matching C99 and glibc, and
    keeping the NaN rule from firing on it.
    """
    if exponent == 0.0:
        return 1.0

    odd_int = _is_odd_integer(exponent)
    negative_base = math.copysign(1.0, base) < 0.0

    if math.isnan(base):
        return -math.nan if (negative_base and not odd_int) else math.nan

    try:
        return math.pow(base, exponent)
    except ValueError:
        # Two distinct C behaviours land on Python's one exception.
        if base == 0.0:
            # Pole error -- the sign follows the usual negative-base rule.
            return -math.inf if (negative_base and odd_int) else math.inf
        # Domain error: negative finite base, non-integer exponent.
        return -math.nan
    except OverflowError:
        return -math.inf if (negative_base and odd_int) else math.inf


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

        The constructor loop is COUNTED rather than run only for
        `granularity == day`, where `get_timestamp_class` is the identity
        (time_dispersion.cc:59-60) and cannot fail. For `month` classic
        evaluates `OC_ASSERT(!timestamp.is_not_a_date(), ...)`
        (time_dispersion.cc:61-64) once per observation, and every
        timestamp this port (or an un-flagged classic run) can produce IS
        not-a-date -- so the assertion fires on the first observation of
        any non-empty table. Reproduced below, with classic's own message
        text, because skipping the loop silently made the port MORE
        permissive than the thing it ports. Measured 2026-09-09,
        `-Hpre -i $CLASSIC/tests/moses/dataset.csv -m300 -W1 -q0.1 -w1.0
        --time-bscore-granularity=month`: classic exits 134
        (AssertionException at time_dispersion.cc:62, core dumped) and
        this port used to exit 0 printing the `day` answer.
        """
        if multiplier != 1:
            raise AssertionError("Multiplier other than 1 is not supported yet")
        super().__init__(ctable)
        self._granularity = granularity
        self._multiplier = multiplier
        self._pressure = _to_score_t(time_dispersion_pressure)
        self._exponent = _to_score_t(time_dispersion_exponent)

        # C++: time_dispersion.cc:47-50 -- the real loop, over every
        # OBSERVATION (`timed_keys()`, not this port's untimed mapping
        # view; see `TimedCounter`'s class docstring for why they differ).
        timestamp_classes = set()
        for counter in ctable.values():
            for tv in timed_keys_of(counter):
                timestamp_classes.add(self.get_timestamp_class(tv.timestamp))

        # C++: time_dispersion.cc:53, `_Hmax = log2(timestamp_classes.size())`.
        # `math.log2(0)` raises `ValueError` in Python where C++'s
        # `std::log2(0.0)` returns `-HUGE_VAL`; handled explicitly here
        # rather than via `_to_score_t`, which narrows precision, not this
        # library-behaviour difference. An empty CTable is the only input
        # that reaches it.
        self._Hmax: score_t = (
            _to_score_t(math.log2(len(timestamp_classes))) if timestamp_classes else -math.inf
        )

    def get_timestamp_class(self, timestamp: Any) -> Any:
        """C++: moses/moses/scoring/time_dispersion.cc:56 get_timestamp_class

        Buckets one observation's timestamp at the configured granularity:
        `day` returns it unchanged; `month` returns the FIRST of its month
        (`TTable::value_type(timestamp.year(), timestamp.month(), 1)`).

        The `month` arm asserts the timestamp is a real date first
        (time_dispersion.cc:61-64). Every observation of a table loaded
        WITHOUT `--timestamp-feature` carries the not-a-date sentinel
        (`None` here), so that assertion fires on the first observation of
        any such table -- reproduced with classic's own message text.
        Measured 2026-09-09: `-Hpre -i dataset.csv -m300 -W1 -q0.1 -w1.0
        --time-bscore-granularity=month` exits 134 in classic
        (AssertionException at time_dispersion.cc:62, core dumped).

        The `default:` arm (time_dispersion.cc:65-70) is unreachable while
        `TemporalGranularity` has exactly two members, but is ported so a
        third one added later fails loudly instead of silently getting
        `day`'s answer.
        """
        if self._granularity == TemporalGranularity.day:
            return timestamp
        if self._granularity == TemporalGranularity.month:
            if timestamp is None:
                raise AssertionError(
                    "timestamp is not a valid date, "
                    "are you sure you have loaded a timestamp feature "
                    "(option --timestamp-feature)?"
                )
            return date(timestamp.year, timestamp.month, 1)
        raise AssertionError(f"Case {int(self._granularity)} not implemented")

    def get_time_dispersion_penalty(self, ctt: Mapping[Any, Any]) -> score_t:
        """C++: moses/moses/scoring/time_dispersion.cc:76 get_time_dispersion_penalty

        "the penalty is defined as pressure * (1 - H(time|active) /
        Hmax)^exponent" (time_dispersion.h:89-91's own doc comment.)

        `ctt` mirrors `CTableTime` (table.h:518, `std::map<TTable::
        value_type, Counter<vertex, count_t>>`) -- a mapping from
        timestamp-class to a `Counter` of `builtin.logical_true`/
        `logical_false` observation counts. Its one real caller is
        `precision_bscore.do_score`, which passes
        `ctable_res.ordered_by_time()` (precision_bscore.cc:342).

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
        # `std::pow` here is glibc's `powf`, which differs from Python's
        # `math.pow` in four ways this expression can reach -- above all
        # the SIGN of the NaN that `_Hmax == 0.0 and H == 0.0` produces.
        # See `_powf` (this module) for the whole measured table. Both
        # degenerate `_Hmax == 0.0` cases are live from the CLI: `H > 0.0`
        # makes `normalized_H` `+inf` and `penalty` `-inf`, while
        # `H == 0.0` makes both NaN.
        distorted_penalty = _to_score_t(_powf(penalty, self._exponent))
        scaled_distorted_penalty = _to_score_t(self._pressure * distorted_penalty)
        return scaled_distorted_penalty
