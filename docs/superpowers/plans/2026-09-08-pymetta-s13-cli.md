# pyMoses S13: CLI parity and output formatting

**Date:** 2026-09-08
**Status:** Proposed
**Depends on:** S11 (end-to-end loop), S12 (crossover — so the default configuration is real)

## Scope

The layer between a shell command and the search:

```
argv -> problem-params (options + defaults) -> problem_manager registry
     -> it_table_problem -> moses_main -> metapop_printer -> stdout
```

After S13 the port is drivable as `moses -i data.csv -u y` rather than only
from Python, and its output can be diffed against classic's line for line.

**Files (11 / 4,037 lines):** `main/moses-exec.cc`, `moses_exec.{h,cc}`,
`moses_exec_def.{h,cc}`, `problem-params.{h,cc}`, `problem.{h,cc}`,
`table-problems.{h,cc}`.

## What the line count hides — measured

`problem-params.cc` alone is **2,407 lines, 60% of the slice**, and it is
almost entirely option declarations: **168** `add_options`/`po::value` sites.
That is broad-but-shallow work, not deep logic.

The important consequence: **most of those options configure subsystems this
port has not built.** Feature selection, diversity, domination,
ensembles/boosting, subsampling and the three non-hill-climbing optimizers are
all P2; contin/mixed is S14.

**Design rule:** port the option *surface* faithfully — every flag parses, every
default matches C++ — but any option whose subsystem is unported must **raise a
clear error when set to a non-default value**, not be silently ignored. A flag
that parses and does nothing is worse than one that fails, because a user
diffing against classic would see a silent behavioural difference. This is the
same gate-and-raise discipline S8, S10 and S11 used.

## Two measured scoping facts

- **`metapop_printer` is `moses_main.h:60`**, not a `main/` file. `moses_main.py`
  is already PARTIAL, so the printer belongs there and its row is already open.
- **`table-problems.cc` registers 11 problems** (`:570-580`), of which only
  `it_table_problem` is reachable now. `pre`, `pre_conj`, `prerec`, `recall`,
  `bep`, `f_one`, `select`, `cluster` need the precision/discriminating/select
  bscores (all P2); `ann_table_problem` is an excluded domain; `ip_problem`
  likewise. Port the registry and `it`; register the rest as raising stubs so
  `-H recall` fails loudly instead of silently doing something else.

## Task 0: `problem_manager` and `problem_base`

Port the registry (`problem.{h,cc}`, 286 lines). `moses_exec.cc` does one
`find_problem(pms.problem)` and calls `run()`, so the registry is the whole
dispatch mechanism — adding a problem is a subclass plus one `register_problem`
line.

**Exit:** registering and finding a problem matches C++; an unknown `-H` value
fails the way classic fails.

## Task 1: `problem-params` — options and defaults

The 168 option declarations, their defaults, and the `moses_parameters` /
`metapop_parameters` / `hc_parameters` / `deme_parameters` wiring they drive.

Defaults are behaviour: they decide which paths run. **Pin every default with a
test.** S9's review found *every* `hc_parameters` default unpinned, including
the one its own DIVERGENCE depended on, and I later had to confirm four of them
by hand.

Note this is where the deferred uninitialized fields land:
`diversity_parameters.autoscale` and the ten `subsample_deme_filter_parameters`
fields are genuinely uninitialized in C++ and the port leaves them unset so
reading one raises. **The CLI is what sets them** (`problem-params.cc:2295`,
`moses_main.cc:106`), so whichever options touch them must populate them before
`diversity.enabled()` can be true.

**Exit:** every flag parses, every default byte-matches C++, and every
unsupported option raises rather than being ignored.

## Task 2: `it_table_problem` and `table-problems` plumbing

The one reachable problem, plus the shared table setup.

**Exit:** `-H it` reproduces what the Python-level S11 path already produces.

## Task 3: `moses_exec` and `metapop_printer`

The entry point and output formatting. `metapop_printer` goes in
`moses_main.py`.

**Exit:** stdout matches classic's for the same flags, line for line, on
`test_it`'s dataset.

## Task 4: Review and closeout

Ruff, `git diff --check`, `bash tools/ci.sh` (now including
`coverage_totals.py --check` and `check_fixture_manifests.py`). Update
`docs/ALIGNMENT.md`, `docs/MODULE-COVERAGE.md` and carry-forward notes.

**Closeout is part of the slice, not an afterthought.** S9, S10 and S11 were
each merged with their coverage rows still saying PLANNED; the checker now
catches that (`check_planned_rows_against_reality`), but do not rely on it.

## Hazards

- **A parsing flag that silently does nothing** is the specific risk here. See
  the design rule above.
- **`score_t` narrowing** — any option feeding a score comparison must narrow.
- **"Correct but nothing tests it"** — fourteen occurrences. An options table is
  exactly the shape that gets a smoke test and no discrimination: prove each
  default fires by mutation.
- **Output diffing is fragile**: classic prints binary32 values at double width
  (`-3.57142853736877441`). Match the formatting, not a rounded value.

## What S13 does NOT do

No contin/mixed (S14). No feature selection, diversity, domination, ensembles,
subsampling, or the other optimizers (P2). No distributed or MPI. The seven
`MOSESUTest` demo-problem cases stay blocked — they need `demo-problems.cc` and
`logical_bscore`, both P2, and S13 does not change that.
