"""Uniform per-call instrumentation for every MOSES inner-optimizer arm.

Wraps ``pymoses.moses.representation.instance_scorer.complexity_based_scorer.
__call__`` -- the one seam every representation-level candidate evaluation
passes through, for hill climbing and for GOMEA alike. Both reach it the
same way, from ``deme_expander.optimize_demes``'s

    cpx_scorer = complexity_based_scorer(self._cscorer, self._reps[i], ...)
    self._optimize(self._demes[i][j], cpx_scorer, max_evals_per_deme, max_time)

-- ``self._optimize`` is the injected ``optimizer_base`` (hill climbing or
GOMEA), and neither one constructs its own scorer. Patching the CLASS's
``__call__`` here therefore instruments every instance any optimizer ever
builds, with no optimizer-specific code anywhere in this module. This is the
ONLY instrumentation the benchmark harness installs.
"""

from __future__ import annotations

import contextlib
import copy
import time
from collections.abc import Callable, Iterator

from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance_scorer import complexity_based_scorer

__all__ = ["ScorerTrace", "instrument_scorer", "summarize_events"]


class ScorerTrace:
    """Records every scorer call's best-so-far trajectory.

    ``n_calls`` counts every call, whether or not it improved anything.
    ``events`` holds one dict per call that improved either the best RAW
    score or the best MOSES-comparator score --

        {"n": <1-based call index>, "t": <seconds since the trace started>,
         "raw": <this call's raw score>, "pen": <this call's penalized
         score>, "cpx": <this call's complexity>}

    Both "best" trackers are kept independently, on purpose: the MOSES
    comparator (``composite_score.__lt__``, "a < b" means a is WORSE than
    b -- see that method's own docstring, types.py) breaks ties by
    complexity and applies the complexity penalty, so it can prefer a call
    with a LOWER raw score than an earlier call once penalties are
    accounted for. The raw-score trajectory and the comparator's own
    trajectory are genuinely different curves, and this harness needs both:
    raw score for task-success thresholds (``first_hit``), the comparator
    for what MOSES itself actually treats as progress.
    """

    def __init__(self, sink: Callable[[dict], None] | None = None) -> None:
        """`sink`, if given, is called with each event dict right after it
        is appended to `events` -- `runner.py` uses this to stream every
        improvement to `trace.jsonl` on disk AS IT HAPPENS (fix round 1,
        finding 4), so a run killed at the wall cap, or one that crashes
        outright, still leaves the partial trace behind for the parent to
        recover `best_raw`/`best_pen`/`best_cpx`/`evals_to_target` from
        (see `summarize_events` below)."""
        self.n_calls = 0
        self.events: list[dict[str, float]] = []
        self.best_raw: float | None = None
        self.best_composite: composite_score | None = None
        self._start: float = 0.0
        self._sink = sink

    def best_pen(self) -> float | None:
        """The penalized score of the best-by-comparator call so far, or
        ``None`` before the first call."""
        return None if self.best_composite is None else self.best_composite.get_penalized_score()

    def best_cpx(self) -> int | None:
        """The complexity of the best-by-comparator call so far, or
        ``None`` before the first call."""
        return None if self.best_composite is None else self.best_composite.get_complexity()

    def first_hit(self, target_raw: float) -> tuple[int, float] | None:
        """The (call index, elapsed seconds) of the first call whose raw
        score reached ``target_raw``, or ``None`` if it never did.

        ``best_raw`` is a running max, so ANY call whose raw score is a new
        maximum is, by definition, a raw-score improvement and is recorded
        as an event; a call that is not such an improvement can never
        exceed the running best and so cannot be the first crossing of
        ``target_raw`` either. The first call that first reaches
        ``target_raw`` is therefore always itself a recorded event, and
        scanning the (call-ordered) event list for the first one whose own
        "raw" field already meets the target finds exactly that call --
        without needing to store every call's raw score, only the
        improving ones.
        """
        for ev in self.events:
            if ev["raw"] >= target_raw:
                return ev["n"], ev["t"]
        return None

    def record(self, score: composite_score) -> None:
        """Called once per scorer invocation, with that call's returned
        ``composite_score``. Not itself part of the wrapped signature --
        kept separate from ``instrument_scorer`` so a test can drive it
        directly against hand-built scores (see
        ``tests/gomea/test_benchmarks.py``)."""
        self.n_calls += 1
        raw = float(score.get_score())
        pen = float(score.get_penalized_score())
        cpx = score.get_complexity()

        improved = False
        if self.best_raw is None or raw > self.best_raw:
            self.best_raw = raw
            improved = True
        if self.best_composite is None or self.best_composite < score:
            # `composite_score` is mutated in place elsewhere in this port
            # (`set_score`/`set_uniformity_penalty`); keep an independent
            # snapshot, the same `copy.copy` treatment
            # `instance_scorer.py`'s own `worst_composite_score` handling
            # uses, since every field is a plain float/int.
            self.best_composite = copy.copy(score)
            improved = True

        if improved:
            ev = {
                "n": self.n_calls,
                "t": time.perf_counter() - self._start,
                "raw": raw,
                "pen": pen,
                "cpx": cpx,
            }
            self.events.append(ev)
            if self._sink is not None:
                self._sink(ev)


@contextlib.contextmanager
def instrument_scorer(trace: ScorerTrace) -> Iterator[ScorerTrace]:
    """Install the instrumentation for the duration of the ``with`` block.

    Patches ``complexity_based_scorer.__call__`` at the CLASS level, so
    every instance -- constructed by whichever optimizer is running -- is
    measured identically, and restores the original method on exit,
    including when the ``with`` block raises.
    """
    original = complexity_based_scorer.__call__
    trace._start = time.perf_counter()

    def wrapped(self, inst):
        score = original(self, inst)
        trace.record(score)
        return score

    complexity_based_scorer.__call__ = wrapped
    try:
        yield trace
    finally:
        complexity_based_scorer.__call__ = original


def summarize_events(events: list[dict], target_raw: float) -> dict:
    """Reconstruct `n_calls`/`best_raw`/`best_pen`/`best_cpx`/
    `evals_to_target`/`time_to_target` from a list of event dicts read back
    from a `trace.jsonl` file -- the same schema `ScorerTrace.record`
    writes. Used by `runner.py` to recover as much as possible from a
    PARTIAL trace when a run was killed at the wall cap or a child died
    without writing its own `summary.json` (fix round 1, finding 4): with
    streaming (`ScorerTrace`'s `sink`), the file on disk holds exactly the
    events that had already happened at the moment of death, so this
    recovers the same values `ScorerTrace` itself would report had the run
    been asked for them at that exact point.

    `n_calls` here is "the last call index seen in the trace" -- a LOWER
    BOUND on the true total call count, since only improving calls are
    ever recorded (documented convention, not the true count).

    Returns all-`None` fields for an empty (or all-unparseable) event list.
    """
    if not events:
        return {
            "n_calls": None,
            "best_raw": None,
            "best_pen": None,
            "best_cpx": None,
            "evals_to_target": None,
            "time_to_target": None,
        }
    best_raw = max(ev["raw"] for ev in events)
    best_pen: float | None = None
    best_cpx: int | None = None
    for ev in events:
        # Mirrors `composite_score.__lt__`: higher penalized score is
        # better, and at an EQUAL penalized score, LOWER complexity is
        # better (types.py's own tie-break, "decreasing order" of
        # complexity meaning the more complex one is WORSE).
        key = (ev["pen"], -ev["cpx"])
        if best_pen is None or key > (best_pen, -best_cpx):  # type: ignore[operator]
            best_pen, best_cpx = ev["pen"], ev["cpx"]
    evals_to_target = None
    time_to_target = None
    for ev in events:
        if ev["raw"] >= target_raw:
            evals_to_target, time_to_target = ev["n"], ev["t"]
            break
    return {
        "n_calls": events[-1]["n"],
        "best_raw": best_raw,
        "best_pen": best_pen,
        "best_cpx": best_cpx,
        "evals_to_target": evals_to_target,
        "time_to_target": time_to_target,
    }
