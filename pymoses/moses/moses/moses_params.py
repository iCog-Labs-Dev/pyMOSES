"""C++: moses/moses/moses/moses_params.h

`moses_parameters` ("parameters to decide how to run moses", the header's
own comment) and `moses_statistics` ("Keep track of miscellaneous solver
statistics") -- ported verbatim, field for field and default for default,
per S11 Task 0. This is the ENTIRE termination surface of the outer search
loop: `local_moses.cc`'s four termination lambdas (`not_max_evals`,
`not_max_gens`, `not_max_score`, `not_max_time`, local_moses.cc:131-134)
read `moses_parameters`'s four `max_*` fields against `moses_statistics`'s
three counters, so a wrong default here silently changes when the whole
learner stops -- not merely how it logs.

Unsigned-subtraction hazard check (S9's Critical 1 was a `size_t
nleft = max_evals - current` wrapping to `SIZE_MAX`, hill-climbing.cc:541).
`local_moses.cc:142-144` performs exactly two subtractions against these
fields:

    pa.max_evals - stats.n_evals        # local_moses.cc:143
    pa.max_time  - stats.elapsed_secs   # local_moses.cc:144

Measured against the ACTUAL declared C++ types below (not assumed):
`max_evals` is a plain (signed) `int` (moses_params.h:63) and `n_evals` is
`std::atomic<int>` (moses_params.h:89) -- an `int`, not an unsigned type,
underneath the atomic wrapper -- so `int - int` is a signed subtraction
that can go negative like any signed integer subtraction; it does NOT wrap
to a huge positive value the way an unsigned/`size_t` subtraction would.
`max_time` is `time_t` (moses_params.h:73) and `elapsed_secs` is a plain
`double` (moses_params.h:95); `time_t - double` promotes the `time_t`
operand to `double` and performs floating-point subtraction, which has no
integer-wraparound semantics at all. Confirmed on this project's actual
build target (Linux x86-64, glibc, gcc): `time_t` is a signed 8-byte
integer (`sizeof(time_t) == 8`, `(time_t)-1 < 0`, verified by compiling and
running a probe against `<time.h>`), so even reinterpreted as a plain
integer subtraction it would still be signed. **Conclusion: NEITHER
subtraction `local_moses.cc` performs has an unsigned operand, so neither
can reproduce S9's `size_t`-wraparound shape.** No field ported by this
module has a C++ type that permits an unsigned wrap; see each field's own
note below for its exact declared type regardless, since Task 1 (which
ports `local_moses.cc` itself) should verify this against whatever it
actually writes rather than trust this note alone -- this describes the
fields as declared here, not the loop code once ported.

`max_gens`'s own termination lambda uses `!=`, not `<`
(`local_moses.cc:132`, `pa.max_gens != stats.n_expansions`) -- ported
verbatim in the FUTURE loop task, not fixed to `<`, per the plan's own
instruction. Whether `!=` can ever miss its target and loop forever
depends on how `n_expansions` is mutated, which this task can already
verify from the field's type and the (Task-1-scope) call site: within
`local_moses.cc`, `stats.n_expansions` is written ONLY by
`stats.n_expansions++` (local_moses.cc:80), executed exactly once per
`while`-loop iteration, and `deme_expander::create_demes`'s own
`n_expansions` parameter is taken BY VALUE (`deme_expander.h:56`,
`int n_expansions = 0`), so it cannot mutate `stats.n_expansions` through
that call either. A counter that starts at its default `0`
(`moses_statistics.__init__` below) and only ever increments by exactly 1
cannot skip over any non-negative integer, so `!=` and `<` agree for every
`max_gens >= 0`; for the documented default `max_gens = -1` ("if negative,
then no limit", moses_params.h:65-66), `n_expansions` (never negative) is
never equal to `-1`, so the lambda is unconditionally `True` -- correctly
unbounded, matching the header comment. Re-verify this reasoning if Task 1
ever finds `n_expansions` mutated somewhere this note did not check.

`jobs_t` (moses_params.h:35, `std::map<std::string, unsigned>`) is a plain
type alias here, the same convention `types.py`'s `score_t`/
`behavioral_score` use for their own C++ typedefs -- its value type
(`unsigned`, a job COUNT) is not used in any subtraction this module or
`local_moses.cc` performs, and no field of it is read within this whole
S11 slice (`local_moses.{h,cc}`, `moses_main.{h,cc}`, this file) -- only
`distributed_moses.cc` (a different, out-of-scope `-j host:N` mechanism)
and `problem-params.cc` (S13 CLI wiring) touch `.jobs` at all.
"""

from __future__ import annotations

from pymoses.moses.moses.types import _to_score_t, score_t

__all__ = ["jobs_t", "moses_parameters", "moses_statistics"]

# C++: moses/moses/moses/moses_params.h:35 jobs_t
# "A map between hostname and number of jobs allocated." (the header's own
# comment.) Value type is `unsigned`; see the module docstring's `jobs_t`
# paragraph for why that is never exercised by this slice.
jobs_t = dict[str, int]

# C++: <climits> INT_MAX (32-bit `int`) -- `moses_parameters.max_time`'s
# hardcoded default (moses_params.h:52, `max_time(INT_MAX)`). Measured on
# this project's build target: `int` is 4 bytes, `INT_MAX == 2147483647`.
_INT32_MAX = 2147483647


class moses_parameters:
    """C++: moses/moses/moses/moses_params.h:40 moses_parameters

    "parameters to decide how to run moses" (the header's own comment).
    See the module docstring for the unsigned-wrap analysis and the
    `max_gens`/`jobs_t` notes shared by every field below.
    """

    def __init__(
        self,
        vm: object | None = None,
        jobs: jobs_t | None = None,
        local: bool = True,
        max_evals: int = 10000,
        max_gens: int = -1,
        max_score: score_t = 0,
        max_cnd_output: int = -1,
    ) -> None:
        """C++: moses/moses/moses/moses_params.h:42 moses_parameters

        Constructor-parameter order and defaults match C++ exactly
        (moses_params.h:42-49): `_vm`, `_jobs`, `_local=true`,
        `_max_evals=10000`, `_max_gens=-1`, `_max_score=0`,
        `_max_cnd_output=-1`. `mpi`, `force_feed` and `max_time` are NOT
        constructor parameters in C++ -- they are fixed member-initializer
        values (`mpi(false), force_feed(false), ... max_time(INT_MAX)`,
        moses_params.h:50-52) that a caller can only change afterward by
        mutating the constructed object's public field, ported the same
        way as plain assignments below rather than additional constructor
        parameters (the same convention `hc_parameters.__init__`'s own
        docstring documents for its five member-initializer-only fields).

        Field types, verbatim from the header, for the loop task to reason
        about (see the module docstring for the full wrap analysis):
        `local`/`mpi`/`force_feed` are `bool`; `jobs` is `jobs_t` =
        `map<string, unsigned>`; `vm` is
        `boost::program_options::variables_map`; `max_evals`/`max_gens`/
        `max_cnd_output` are plain (signed) `int`; `max_score` is
        `score_t` (`float`, narrowed via `_to_score_t` below, matching
        every other `score_t`-typed parameter in this port); `max_time` is
        `time_t` (signed on this project's build target, see module
        docstring).

        # DIVERGENCE(C++ moses/moses/moses/moses_params.h:42-43,60):
        # `vm` (`const boost::program_options::variables_map&`, C++'s
        # parsed-command-line-options container) has no ported Python
        # equivalent yet -- `docs/MODULE-COVERAGE.md:291` explicitly
        # marks `problem-params.cc`'s whole `boost::program_options`
        # surface "PLANNED: S13 ... via argparse", not this cycle. This
        # field is therefore an opaque stand-in (`None` by default,
        # stored by reference, not copied, whatever a caller passes) --
        # verified inert for this whole S11 slice: `pa.vm` is read ONLY
        # by `distributed_moses.cc:360` (`git grep`, `moses/moses/`, zero
        # other hits), a different, out-of-scope `-j host:N` mechanism
        # (see this module's own docstring and the S11 plan's Task 2
        # notes). C++ itself copies `_vm` by value into the `vm` member
        # (moses_params.h:42-43,60, a `variables_map`, not a reference)
        # -- storing the caller's object by reference here instead is
        # itself a second, smaller divergence, accepted because there is
        # no ported type to copy and nothing in scope mutates or reads
        # this field either way.
        """
        self.local = local
        self.mpi = False
        self.force_feed = False
        self.jobs: jobs_t = dict(jobs) if jobs is not None else {}
        self.vm = vm

        self.max_evals = max_evals
        self.max_gens = max_gens
        self.max_score: score_t = _to_score_t(max_score)
        self.max_time = _INT32_MAX
        self.max_cnd_output = max_cnd_output


class moses_statistics:
    """C++: moses/moses/moses/moses_params.h:83 moses_statistics

    "Keep track of miscellaneous solver statistics" (the header's own
    comment). See the module docstring's unsigned-wrap analysis for
    `n_evals`/`elapsed_secs`'s role in `local_moses.cc`'s two
    subtractions, and for the exact non-skipping increment pattern
    `n_expansions` follows within this slice.

    # DIVERGENCE(C++ moses/moses/moses/moses_params.h:89,92): `n_evals`
    # and `n_expansions` are declared `std::atomic<int>` in C++, giving
    # thread-safe increments when multiple worker threads score
    # candidates concurrently. This port represents both as plain Python
    # `int` attributes with no synchronization, the same way
    # `hill_climbing.total_evals` and `deme_expander.total_evals()`
    # already represent C++ counters that exist for concurrency reasons
    # -- accepted because no threading is wired up anywhere in this port
    # yet (`import threading` appears in zero files under `pymoses/`, so
    # every existing caller updates these counters from a single
    # thread of control). Underneath the wrapper the C++ element type is
    # still plain (signed) `int`, not `unsigned` -- see the module
    # docstring for why that rules out an unsigned-wrap reproduction of
    # S9's Critical 1 regardless of the synchronization question. Whoever
    # introduces real concurrent evaluation in a later task must add real
    # synchronization here (a `threading.Lock`, or an atomics-like
    # primitive) before that becomes observable; nothing in this task's
    # scope exercises more than one thread of control.
    """

    def __init__(self) -> None:
        """C++: moses/moses/moses/moses_params.h:85 moses_statistics

        `n_evals(0), n_expansions(0), elapsed_secs(0)` -- all three
        default to `0` per the initializer list, ported verbatim.
        `elapsed_secs` is a plain `double` (moses_params.h:95); the `0`
        C++ literal is `int` there but implicitly converts to `double` at
        the member-initializer, so `0.0` here matches exactly.
        """
        self.n_evals = 0
        self.n_expansions = 0
        self.elapsed_secs = 0.0
