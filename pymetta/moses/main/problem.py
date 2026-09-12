"""C++: moses/moses/main/problem.{h,cc}

The problem registry: `problem_base` (the interface every `-H <name>` problem
implements) and `problem_manager` (the name -> `problem_base` map
`moses_exec.cc` builds once and queries once). `moses_exec.cc`'s own body is
exactly:

    problem_manager pmr;
    register_table_problems(pmr, mgr);
    register_demo_problems(pmr, mgr);
    ...
    problem_base* probm = pmr.find_problem(pms.problem);
    if (probm) { probm->run(&pms); return 0; }
    ... log "unsupported" and return 1 ...

so `problem_manager`/`problem_base` -- register_problem/find_problem plus the
interface they traffic in -- is the WHOLE dispatch mechanism; adding a problem
elsewhere is a subclass plus one `register_problem` call. This task ports
exactly `problem_base`, `problem_manager`, `problem_manager::register_problem`
and `problem_manager::find_problem` (problem.h:52-69, problem.cc:92-112), per
this task's own brief.

Declared in the same two C++ files but deliberately NOT ported here:

* `option_base`/`option_manager` (problem.h:33-50, problem.cc:44-83) -- the
  boost::program_options `add_options`/`parse_options` plumbing every CLI
  option struct (`problem_params`, `table_problem_params`, ...) derives from.
  This port's CLI options are argparse-based (`docs/MODULE-COVERAGE.md`'s own
  non-goal note on `problem-params.cc`), which has no boost::program_options
  object to wrap -- if any of this machinery is needed at all, it belongs
  with `problem_params.py` (owned concurrently with this task; not touched
  here). `problem_base.run` below still takes an `ob` parameter, unannotated
  beyond `object`, exactly mirroring `run(option_base*)`'s shape without
  committing to a concrete type this file does not define.
* `alphabet_size` (problem.cc:138-202, with its `contains_type` helper at
  :126-131 and its unsupported-type fallback `log_output_error_exit` --
  :117-122, an unconditional `exit(1)` -- called only from the `default:`
  switch arm at :193-196) -- used only inside `table-problems.cc`'s concrete
  `*_table_problem::run()` bodies (S13 Task 2, `table_problems.py`, not yet
  ported). None of the raising stubs this file registers call it: each
  raises immediately, before doing any real work that would need it.

## Fidelity: duplicate registration and unknown lookup

Read directly off problem.cc (quoted in the two methods' own docstrings
below) rather than inferred: neither `register_problem` nor `find_problem`
has an assert or `exit()` anywhere in its body. `register_problem` builds a
`std::pair` and passes it to `std::map::insert`, which -- per the container's
own well-defined semantics, not a project-specific quirk -- is a no-op when
an element with that key already exists: the FIRST problem registered under
a given name wins permanently; every later registration under the same name
is silently dropped, with no exception, log line, or return value indicating
that anything was discarded. `find_problem` does `map::find` and returns
`NULL` on a miss. The only place `-H <unknown>` actually produces a visible
error is one layer up, in `moses_exec.cc` (`if (probm) ... else { log
"unsupported"; return 1; }`) and in `option_manager::parse_options`'s
`OC_ASSERT`/`exit(1)` paths for malformed CLI syntax -- both outside this
file's ported scope (`moses_exec.py` is S13 Task 3; `option_manager` is not
ported at all, see above). Because problem.cc's OWN two methods never
assert or exit, there is nothing to reproduce with a raise here: this port's
`register_problem` and `find_problem` are faithful, gap-free translations,
not divergences.

## `~problem_manager()` (problem.cc:107-112) has no Python equivalent

C++ deletes every registered `problem_base*` when the manager is destroyed.
This port relies on ordinary garbage collection instead -- no `__del__` is
added (matching `moses_main.py`'s own "nothing to free in Python; ordinary
garbage collection reclaims it" note for a similar case). Checked whether
that changes anything OBSERVABLE, per this task's own fidelity checklist:
`problem_base`'s destructor is `virtual ~problem_base() {}` (problem.h:55),
an empty body, and neither `table-problems.h` nor `demo-problems.h` (the
only two files that ever define a concrete `problem_base` subclass in
classic) overrides it -- grepped for `~.*problem`/`virtual ~` in both files,
zero matches. So no concrete problem class in classic ever does anything
observable when destroyed, and this port's reliance on GC instead of an
explicit delete loop has no behavioural difference from C++.

## Stubs for the three unreachable `table-problems.cc` problems

`register_table_problems` (table-problems.cc:562-580) registers eleven
problems. This file registers none of the ones that are ported for real:
that is `table_problems.py`'s job, and this file leaves it a clean seam by
simply not naming them anywhere below. What is left here are
`_unported_table_problem` stubs for the ones that still need something
this port has not built: same `name()`/`description()` strings as their
real C++ classes (table-problems.h), so a registry lookup and its reported
description agree with classic; `run()` raises `NotImplementedError`
instead of silently doing nothing or dispatching elsewhere. This is the
same "port the gate, raise inside it" discipline `ensemble.py`,
`merging.py` and `deme_expander.py` use for their own P2 subsystems -- so
an unported `-H` fails loudly rather than acting as a no-op.

That leaves exactly two: `ip` and `ann-it`.
`table_problems.py` registers its nine real problems FIRST and
`problem_manager.register_problem` is `dict.setdefault`, so the ordering
would tolerate extra stubs here -- but a stub for a problem that runs has
nothing true to say, so the seven that had become dead were removed rather
than left behind, and `pre-conj`'s went the same way once
`pre_conj_table_problem` landed in `table_problems.py`. That last one is
worth recording: its reason string said, accurately, that "what is
missing is only a pre_conj_table_problem in table_problems.py" -- and an
accurate description of a defect is not a discharge of it. The stub sat
there blocking a problem classic runs fine (measured 2026-09-09:
`-Hpre-conj -i tests/fixtures/table_data/precision.csv -m100 -W1` prints
a normal candidate list in classic). Each surviving stub carries its OWN
reason string; see `_unported_table_problem.run` for why a single shared
reason was wrong.

CORRECTED (this whole paragraph): it used to say only `it_table_problem`
was reachable and that "the other ten need subsystems this port has not
built (precision/discriminating/select bscores for `pre`/`pre-conj`/
`prerec`/`recall`/`bep`/`f_one`/`select`/`cluster`; ...)". Every clause of
that is now false. `precision_bscore`/`discriminating_bscore`/
`select_bscore`/`precision_conj_bscore` are all ported; `cluster`'s
scorer was never any of those -- it is `cluster_bscore` (bscores.h:613), a
one-dimensional k-means, and it is ported too. All nine of those problems
are wired for real in `table_problems.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = [
    "problem_base",
    "problem_manager",
]


class problem_base(ABC):
    """C++: moses/moses/main/problem.h:52 problem_base

    The interface every `-H <name>` problem implements. Pure virtual in
    C++ (all three members `= 0`); `ABC` + `@abstractmethod` is this port's
    equivalent -- instantiating a subclass that leaves any of the three
    unimplemented raises `TypeError`, matching C++'s refusal to instantiate
    a class with an unoverridden pure virtual.
    """

    @abstractmethod
    def name(self) -> str:
        """C++: moses/moses/main/problem.h:56 problem_base::name

        `virtual const std::string name() const = 0;` -- the string a
        `-H` command-line flag names this problem by, and the key
        `register_problem` files it under.
        """
        raise NotImplementedError

    @abstractmethod
    def description(self) -> str:
        """C++: moses/moses/main/problem.h:57 problem_base::description

        `virtual const std::string description() const = 0;` -- the
        one-line help text `-h` would print next to this problem's name.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, ob: object) -> None:
        """C++: moses/moses/main/problem.h:58 problem_base::run

        `virtual void run(option_base*) = 0;`. `ob` is left typed as plain
        `object`, not `option_base`: that class is not ported here (module
        docstring) -- concrete subclasses (S13 Task 2's `table_problems.py`)
        are responsible for downcasting it to whatever options struct they
        actually need, exactly as classic's own `run()` overrides do
        (e.g. `table_problem_params& pms = *static_cast<table_problem_params*>(ob);`,
        table-problems.cc).
        """
        raise NotImplementedError


class problem_manager:
    """C++: moses/moses/main/problem.h:61 problem_manager

    Owns the `{name: problem_base}` registry `moses_exec.cc` builds once
    (via `register_table_problems`/`register_demo_problems`) and queries
    once (`find_problem(pms.problem)`). See the module docstring's
    `~problem_manager()` section for why this port adds no destructor.
    """

    def __init__(self) -> None:
        """No C++ line to cite: `problem_manager` declares no constructor
        of its own (only `~problem_manager()`, problem.h:64), so C++ uses
        the compiler-generated default, which default-constructs
        `_problem_set` (problem.h:68) as an empty `std::map`. `{}` here is
        that same starting state, not a divergence from an explicit line.
        """
        self._problem_set: dict[str, problem_base] = {}

    def register_problem(self, prob: problem_base) -> None:
        """C++: moses/moses/main/problem.cc:93 problem_manager::register_problem

            std::pair<std::string, problem_base*> pr(prob->name(), prob);
            _problem_set.insert(pr);

        `std::map::insert` is a no-op when the key already exists (see the
        module docstring's fidelity section) -- ported via `dict.setdefault`,
        which has the identical "insert only if the key is absent" contract:
        the first problem registered under a given `name()` is kept
        permanently; a later call with a different `problem_base` under the
        SAME name is silently dropped, exactly as in C++ (no exception, no
        log line, nothing to indicate the second registration had no
        effect).
        """
        self._problem_set.setdefault(prob.name(), prob)

    def find_problem(self, name: str) -> problem_base | None:
        """C++: moses/moses/main/problem.cc:99 problem_manager::find_problem

            auto it = _problem_set.find(name);
            if (it != _problem_set.end())
                return it->second;
            return NULL;

        No assert, no exit, on a miss -- just `NULL`. `dict.get(name)`
        returns `None` in that same case, `None` being this language's
        `NULL`: this is a faithful translation, not a divergence (see the
        module docstring's fidelity section for why turning a miss into a
        visible failure is a DIFFERENT layer's job, `moses_exec.cc`, not
        ported in this file).
        """
        return self._problem_set.get(name)


class _unported_table_problem(problem_base):
    """Not one specific C++ class -- a stand-in `problem_base` for a
    `table-problems.cc` problem this port has not wired.
    `name()`/`description()` are fixed at construction to match the real
    C++ class's strings; `reason` states what is missing for THAT problem
    specifically; `run()` always raises, regardless of what `ob` is, since
    there is no real work this stub could correctly do.

    CORRECTED: `run()` used to raise one shared message claiming every
    stubbed problem's scoring was "a precision/discriminating/select
    bscore, the ANN domain, or interesting-predicates scoring", "P2 or
    excluded in this port". That was false for most of the list, and
    flatly wrong for `cluster`, whose scorer is neither -- it is
    `cluster_bscore` (bscores.h:613), one-dimensional k-means, and it is
    now ported (bscores.py) and wired (table_problems.py). `reason` was
    added so no stub can inherit another's excuse again.
    """

    def __init__(self, name: str, description: str, cpp_class: str, reason: str) -> None:
        self._name = name
        self._description = description
        self._cpp_class = cpp_class
        self._reason = reason

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:294 ann_table_problem::run

        Breadcrumb added 2026-09-12, having been ABSENT since this class
        was written. That absence was not cosmetic: `tools/coverage_totals.
        py`'s `check_ported_rows_do_not_gate` attributes every `raise
        NotImplementedError` to the C++ file named by its NEAREST ENCLOSING
        BREADCRUMBED symbol, so a gate with no breadcrumb anywhere above it
        is invisible to that check -- it can never be flagged, no matter
        what MODULE-COVERAGE.md claims about the file it stands in for.
        This is the only gate in the package that stood in for a C++
        function while citing none.

        The cited symbol is the ONE C++ `run` this stub still stands in
        for: `ann_table_problem::run` (cc:294), the ANN-domain problem.
        It stood in for `ip_problem::run` (cc:229) too until
        `ip_table_problem` (table_problems.py) was written, at which point
        the `ip` entry was deleted from `_unported_table_problems` below
        rather than left shadowed -- see that function's own docstring on
        why a stub for a ported problem has nothing true to say.

        `ob` is discarded rather than dispatched on: there is no real work
        this stub could correctly do with any `problem_params`.
        """
        del ob
        raise NotImplementedError(
            f"problem {self._name!r} ({self._cpp_class}) is not ported: "
            f"{self._reason}; -H {self._name} is unsupported"
        )


def _unported_table_problems() -> list[problem_base]:
    """Not a C++ function -- builds one freshly-constructed
    `_unported_table_problem` per `table-problems.cc` problem this port
    still cannot run, mirroring the `new ip_problem(...)`-per-registration
    shape `register_table_problems` (table-problems.cc:562-580) uses.
    `(name, description, C++ class, reason)` tuples: the first three are
    copied verbatim from table-problems.h's `name()`/`description()`
    overrides, the fourth is measured, per-problem, against classic's own
    binary (see each entry's own comment).

    CORRECTED TWICE. It returned TEN stubs -- every problem except `it` --
    until `table_problems.py` grew to wire eight of the eleven for real.
    Seven of those ten were already dead (`register_problem` is
    `dict.setdefault` and `table_problems.py` registers first), so they
    were unreachable stubs asserting a falsehood about a scorer that was
    in fact ported. They were dropped rather than corrected: a stub for a
    ported problem has nothing true to say.

    The second correction removed `ip`, on the same rule, when
    `ip_table_problem` (table_problems.py) landed. Its reason string had
    said "it needs interesting_predicate_bscore (bscores.h:553), which
    bscores.py does not port", which is now false -- bscores.py ports it,
    along with cogutil's `standardizedMannWhitneyU` and boost's
    `weighted_skewness`. `ann-it` is the ONE stub left, and it is the ANN
    domain, excluded project-wide rather than unscheduled.
    """
    return [
        _unported_table_problem(*args)
        for args in (
            (
                "ann-it",
                "ANN-based regression on input table",
                "table-problems.h:101 ann_table_problem",
                # The ANN vocabulary (`moses/comboreduct/ant_combo_vocabulary`,
                # `ann_type`, `tree_transform::decodify_tree`) is excluded
                # project-wide. Classic itself is broken here on an
                # ordinary table: measured, `-Hann-it
                # -i$CLASSIC/tests/moses/dataset.csv` aborts with
                # `boost::bad_get` before printing anything.
                "it needs the ANN vocabulary (ann_type, tree_transform::decodify_tree), "
                "excluded project-wide -- and classic itself aborts with boost::bad_get "
                "on an ordinary table",
            ),
        )
    ]


def _register_unported_table_problems(pmr: problem_manager) -> None:
    """Not a C++ function -- the stub half of what
    `register_table_problems` (table-problems.cc:562-580) would do in a
    complete port; see the module docstring. `table_problems.py` owns
    registering the nine real problems and calls this afterwards for the
    remaining two; it will supersede this function entirely once
    `ip`/`ann-it` are ported for real. Nothing outside
    `table_problems.py` and this module's own tests calls it.
    """
    for prob in _unported_table_problems():
        pmr.register_problem(prob)
