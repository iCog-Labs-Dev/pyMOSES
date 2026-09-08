"""C++: moses/moses/moses/local_moses.{h,cc}

Ports `expand_deme` (local_moses.cc:46, `static` -- translation-unit-private
in real C++; kept as an ordinary, UNPREFIXED module-level function here per
this project's "C++ identifiers preserved verbatim" rule, which pyproject.
toml's own ruff config comment calls "the single most load-bearing
convention in the project" -- not renamed to `_expand_deme`) and
`local_moses` itself (local_moses.cc:107, declared local_moses.h:36-39).
`__all__` below exports only `local_moses`, the one symbol local_moses.h
itself exposes -- `expand_deme` stays importable (nothing in this port
enforces C++ `static` linkage), the same public-surface-vs-file-scope
distinction `merging.py`'s own module-level helper functions draw before
being bound onto `metapopulation`.

## What is dropped, and why (module-wide "no logging surface" convention)

Every `logger()` call in BOTH functions is dropped, the same convention
every prior slice in this port follows (see e.g. `hill_climbing.py`'s own
module docstring). This drops, in full:

* The ENTIRE `if (logger().is_info_enabled()) { ... }` block inside
  `expand_deme` (local_moses.cc:84-88): two `logger().info()` calls and
  `mp.log_best_candidates()`. `log_best_candidates` itself IS still ported
  -- on `merging.py`, its real C++ home (merging.cc:652) -- but
  `expand_deme` does not call it back, since its only C++ call site is
  inside this dropped block; see merging.py's own docstring for that
  method's separate "port control flow, not text" treatment.
* `print_stats_header` (local_moses.cc:118) -- S13 output formatting, per
  this task's own scope note -- and the `optim_stats *os = dynamic_cast<
  optim_stats*>(&dex._optimize);` computed ONLY to feed it and the "Stats:"
  line's own `if (os) {...}` branch below (local_moses.cc:115,165-169):
  nothing outside those two dropped call sites ever reads `os`.
* The per-generation "Stats:" tab-separated line (local_moses.cc:153-189),
  INCLUDING the two `mp.diversity_enabled()`-gated blocks nested inside it
  (local_moses.cc:170-187) -- diversity is P2 and out of scope, per this
  task's own explicit instruction to "Skip the diversity_enabled() stats
  blocks entirely".
* The debug-level top-candidates dump (local_moses.cc:194-200, itself
  gated on `mp.diversity_enabled()` too) and the final "MOSES ends. ..."
  diagnostic (local_moses.cc:203-220).

None of these seven blocks computes anything a LATER statement reads --
every one is print-only, confirmed by reading local_moses.cc top to bottom
-- so dropping all of them changes no state this port's own scope tracks.

## What is load-bearing, and ported exactly

`stats.elapsed_secs`'s ACCUMULATION (local_moses.cc:120-122,147-151) feeds
`not_max_time()`, one of the four real termination predicates -- behaviour,
not logging, per this task's own instruction. C++'s `gettimeofday`/
`timersub` give a `struct timeval` whose `tv_sec + 1.0e-6*tv_usec`
reconstructs a FULL-PRECISION (microsecond-resolution) elapsed-time float
-- unlike `hill_climbing.py`'s own `elapsed_tv_sec`, which truncates to
WHOLE seconds for a DIFFERENT budget check in a different file
(hill-climbing.cc's own max_time countdown). This port's replacement,
`time.monotonic()` (the same "mechanism differs, semantics don't" host-
clock substitution `hill_climbing.py`'s module docstring already
establishes for `gettimeofday`), therefore needs no analogous
truncate-to-int step here: `now - start` is used directly as the
elapsed-seconds float to accumulate.

# DIVERGENCE(C++ moses/moses/moses/local_moses.cc:120-122,147-151):
# `gettimeofday` has microsecond resolution; `time.monotonic()` has
# whatever resolution the host clock provides (typically nanosecond on
# Linux) -- STRICTLY MORE precise, never less. Immaterial to every test in
# this port's fidelity suite: `pa.max_time` defaults to `_INT32_MAX`
# seconds (moses_params.py) and every test bounds a run via
# `max_evals`/`max_gens` instead, so `not_max_time()` never actually
# decides a termination in any scenario this port's own tests construct.

`pa.max_time - stats.elapsed_secs` (local_moses.cc:144) computes a
`double` (`time_t` promotes to `double` under the usual arithmetic
conversions, since `elapsed_secs` is a plain `double`) that C++ then
passes into `expand_deme`'s own `time_t max_time` PARAMETER
(local_moses.cc:46-49) -- an IMPLICIT double-to-`time_t` NARROWING
(truncation toward zero) at that exact call boundary, confirmed against
the actual declared signatures: `expand_deme` forwards this same value,
unchanged, straight into `deme_expander::optimize_demes(int, time_t)`
(deme_expander.h:65). Ported via an explicit `int(...)` at the call site
inside `local_moses` below -- matching where C++'s own type actually
changes -- NOT inside `expand_deme` itself, which receives an
already-`time_t` (already-`int`, in this port) value and performs no
further conversion of its own, exactly like C++. Python's `int(x)`
truncates toward zero for a float `x`, matching C++'s own implicit
floating-to-integral conversion (both round toward zero, not down) --
verified for the negative case too, though `not_max_time()`'s own guard
just above this call keeps the subtrahend smaller than the minuend in
every reachable call.

`pa.max_evals - stats.n_evals` (local_moses.cc:143) is `int - int`: plain
signed subtraction, no unsigned-wrap hazard -- see moses_params.py's own
module docstring for the full analysis this task was handed (S9's Critical
1 shape needs an UNSIGNED operand, and neither operand here is one),
confirmed again here since this is the file that actually performs it.

`mp.best_score() < pa.max_score` (`not_max_score`, local_moses.cc:133) --
both operands are already `score_t`-narrowed at their own point of origin
(`composite_score.score`/`moses_parameters.max_score`, both narrowed at
construction time in other modules) -- comparing two already-narrowed
floats performs no NEW arithmetic that would itself need `_to_score_t`
here; narrowing a comparison's BOOLEAN result would be meaningless.

`pa.max_gens != stats.n_expansions` (`not_max_gens`, local_moses.cc:132) is
ported VERBATIM, `!=` and not `<` -- per this task's own explicit
instruction not to "fix" it. See moses_params.py's own docstring for why
`!=` and `<` provably agree for the reachable range of `n_expansions`
within this whole slice: a counter that starts at 0 (moses_statistics.
__init__) and is written ONLY by the single `stats.n_expansions += 1`
below, executed at most once per `while`-loop iteration, cannot skip past
any non-negative `max_gens` -- the divergence risk `!=` carries in
principle (a counter that COULD skip past `max_gens` would loop forever
under `!=` where `<` would still stop it) is real, just not reachable from
`n_expansions`'s own single write site.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymetta.moses.deme.deme_expander import deme_expander
    from pymetta.moses.metapopulation.metapopulation import metapopulation
    from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics

__all__ = ["local_moses"]


def expand_deme(
    mp: metapopulation,
    dex: deme_expander,
    max_evals: int,
    max_time: int,
    stats: moses_statistics,
) -> bool:
    """C++: moses/moses/moses/local_moses.cc:46 expand_deme

    "Run one deme-creation and optimization step. A single step consists
    of representation-building, to create a deme, followed by
    optimization, ... and finally, a merger of the unique (and possibly
    non-dominated) trees back into the metapopulation, for potential use
    as exemplars for future demes." (local_moses.cc:32-39's own doc
    comment, condensed.) Returns whether expansion should be treated as a
    PREMATURE termination signal for `local_moses`'s own outer loop -- see
    that function's own docstring.

    `mp.empty()` (local_moses.cc:51) short-circuits to `True` -- "nothing
    left to expand from" is itself reported as premature termination.

    The `while (1)` loop (local_moses.cc:57-76) retries `select_exemplar()`
    until either no exemplar remains (`exemplar == mp.end()`,
    local_moses.cc:61 -- this port's own `metapopulation.end()`/
    `select_exemplar()` return `None` for this case, see their own
    docstrings) or one successfully produces a non-empty representation
    (`dex.create_demes(...)` returns `True`, local_moses.cc:71-72).
    `exemplar->get_tree()` (local_moses.cc:71) is `exemplar.get_tree()`
    here -- `exemplar` is the DEREFERENCED `scored_combo_tree` itself in
    this port, not a C++ iterator needing `->` (see `select_exemplar`'s
    own docstring). The two `logger().warn()` calls on the retry path
    (local_moses.cc:62-66,74-75) are dropped -- module docstring's "no
    logging surface" convention; no state either would compute is read
    again.

    `dex.optimize_demes(max_evals, max_time)` / `stats.n_evals +=
    dex.total_evals()` / `stats.n_expansions++` (local_moses.cc:78-80) are
    ported as direct statements -- `+=`/`+= 1` are the exact Python
    analogues of C++'s `+=`/`++` here (both operands plain `int`s, no
    narrowing needed).

    `bool done = mp.merge_demes(dex._demes, dex._reps);` (local_moses.cc:
    82) reads `deme_expander`'s own `_demes`/`_reps` members directly, from
    OUTSIDE the class -- real C++ does the exact same direct member access
    at this exact call site (they are public in the real header), so this
    is not itself a fidelity gap; `dex._demes`/`dex._reps` are ordinary
    (non name-mangled, "private by convention" only) attributes on this
    port's own `deme_expander` for the identical reason.

    The `if (logger().is_info_enabled()) {...}` block (local_moses.cc:
    84-88, two `logger().info()` calls plus `mp.log_best_candidates()`) is
    dropped in full -- see module docstring. `dex.free_demes()`
    (local_moses.cc:89) always runs, unconditionally, exactly as in C++
    (no logging-block early-return sits between it and the merge above in
    either language).

    `return done || mp.empty();` (local_moses.cc:92) -- "Might be empty,
    if the eval fails and throws an exception" (the C++ comment). Ported
    verbatim as `done or mp.empty()`.
    """
    if mp.empty():
        return True

    while True:
        exemplar = mp.select_exemplar()

        if exemplar == mp.end():
            return True

        if dex.create_demes(exemplar.get_tree(), stats.n_expansions):
            break

    dex.optimize_demes(max_evals, max_time)
    stats.n_evals += dex.total_evals()
    stats.n_expansions += 1

    done = mp.merge_demes(dex._demes, dex._reps)

    dex.free_demes()

    return done or mp.empty()


def local_moses(
    mp: metapopulation,
    dex: deme_expander,
    pa: moses_parameters,
    stats: moses_statistics,
) -> None:
    """C++: moses/moses/moses/local_moses.cc:107 local_moses

    "The main function of MOSES, non-distributed version. This is termed
    'local' because it runs only on the local CPU node." (local_moses.cc:
    96-102's own doc comment, condensed.) See module docstring for
    `print_stats_header`/the per-generation "Stats:" line/the final
    diagnostic -- all dropped, all print-only.

    `stats.elapsed_secs = 0.0;` (local_moses.cc:122) resets the counter
    UNCONDITIONALLY at the start of every call -- ported verbatim; a
    caller that invokes `local_moses` more than once (e.g. to resume a
    search) gets a FRESH elapsed-time count starting from that call, not a
    running total across calls, in both languages.

    `bool premature_termination = false;` (local_moses.cc:128) is a LOCAL
    variable in both languages, reset on every call for the same reason
    `elapsed_secs` is -- see this file's own header comment in
    `tools/cpp_probe/local_moses_probe.cc` for the measured consequence
    this has for a probe that calls the real C++ `local_moses()`
    repeatedly to get per-generation snapshots.

    The four termination lambdas (local_moses.cc:131-134) are ported as
    nested closures over `mp`/`pa`/`stats`, matching C++'s own `[&]`
    capture-by-reference exactly (Python closures over enclosing-scope
    names are references, not copies, by construction) -- `not_max_gens`
    keeps its `!=`; see module docstring for the other three's own
    narrowing/overflow notes.

    The `while` loop body (local_moses.cc:140-201) calls `expand_deme`
    with `pa.max_evals - stats.n_evals` and `int(pa.max_time -
    stats.elapsed_secs)` (see module docstring for the latter's narrowing),
    assigns its return to `premature_termination`, then accumulates
    elapsed time (local_moses.cc:147-151) -- see module docstring for the
    `time.monotonic()` substitution. Every remaining statement in the real
    loop body (local_moses.cc:153-200) is a dropped print block (module
    docstring).
    """
    start = time.monotonic()
    stats.elapsed_secs = 0.0

    premature_termination = False

    def not_max_evals() -> bool:
        return stats.n_evals < pa.max_evals

    def not_max_gens() -> bool:
        return pa.max_gens != stats.n_expansions

    def not_max_score() -> bool:
        return mp.best_score() < pa.max_score

    def not_max_time() -> bool:
        return stats.elapsed_secs < pa.max_time

    while (
        not_max_evals()
        and not_max_gens()
        and not_max_score()
        and not_max_time()
        and not premature_termination
    ):
        premature_termination = expand_deme(
            mp,
            dex,
            pa.max_evals - stats.n_evals,
            int(pa.max_time - stats.elapsed_secs),
            stats,
        )

        stop = time.monotonic()
        elapsed = stop - start
        start = stop
        stats.elapsed_secs += elapsed
