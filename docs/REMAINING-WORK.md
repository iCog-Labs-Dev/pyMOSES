# Remaining work to a complete port

Written 2026-09-08 after S12 merged. **Rewritten 2026-09-09** against a full
re-audit of every PLANNED and PARTIAL row in `docs/MODULE-COVERAGE.md`. This is
the durable roadmap: it survives context loss and says what "done" means, in
what order, and why.

**Scope reminder.** 88 files / 12,212 lines are **EXCLUDED by decision**, not
deferred: legacy embodiment, dead `#ifdef` code, C++-only tools, the enum, ANN,
list/fold, object, message and procedure-call domains, and — new on 2026-09-09 —
`partial.{cc,h}` and `distributed_moses.{cc,h}` (see §3f). The port's stated
domains are Boolean, continuous and action, though **the action half is
contradicted by the code and needs settling** (§5). "Complete" therefore means
every non-excluded file ported, with its named gaps closed.

## Status

Regenerate with `python3 tools/coverage_totals.py`; never hand-edit.

| Status | Files | Lines |
|---|---:|---:|
| PORTED | 73 | 12,112 |
| PARTIAL | 82 | 37,332 |
| PLANNED: P2 | 4 | 767 |
| EXCLUDED | 88 | 12,212 |
| **Total** | **247** | **62,423** |

The 2026-09-08 table in this file's previous version said PORTED 17 / PARTIAL 67
/ PLANNED 79; it was already stale when written. Measured against the previous
commit, the 2026-09-09 audit **changed the status of 71 of the 246 existing rows**
and added one: PARTIAL→PORTED 30, PLANNED→PORTED 19, PLANNED→PARTIAL 10,
PORTED→PARTIAL 7, PLANNED→EXCLUDED 4, PARTIAL→EXCLUDED 1. Many more rows kept
their status and had a wrong or stale REASON replaced. Note the total moved too,
246 → 247: `opencog/util/digraph.h` was fully ported, with its own module and
tests, and had **no row at all**.

**S13 and S14 are done as slices**; there is no `PLANNED: S<n>` row left. Four
files have no Python whatsoever, and they are the whole of `PLANNED: P2`:
`fs-main.cc`, `eval-features.{cc,h}`, `type_tree_def.cc`.

## State handoff — end of the 2026-09-09 session

Written so the next execution resumes from measurement rather than re-deriving
it. `bash tools/ci.sh` is **green (exit 0, 5242 passed, 5 skipped)** at this
point, and the working tree is clean.

### The differential baseline — run this FIRST after any change

Every one of these is byte-identical to `$CLASSIC/build/moses/moses/main/moses`
as of this session. They are the regression net; a change that breaks one is
wrong until proven otherwise.

| Invocation | Notes |
|---|---|
| `-i<dataset.csv> -m1000 -W1` and without `-W1` | the `-H it` baseline |
| `-Hpa -k3`, `-Hdj -k3`, `-Hmux -k2`, `-Hmaj -k3` | demo problems |
| `-Hcp -y "and($1 $2)"`, `-Hcp -y "*(2 $1)"` | combo problem, bool + contin |
| `-Hsr -k1..k4` at seeds 1/2/5/9 | needs `ITable`'s RNG-drawing ctor |
| `-Hcluster` on boolean, contin-sample and `trainingRegression.dat` | |
| `-i<contin_regression_sample.csv> -u y` at `-m` 20→500 | contin regression |
| `-i$CLASSIC/trainingRegression.dat -u y -m120` | 8000 rows |
| `-ahc`, `-asa`, `-aun`, `-aps` | all four optimizers, boolean and contin |
| `--linear-regression=1`, `--contin-depth=3/8`, `--it-abs-err=1` | contin flags |
| `--discard-dominated=1`, `--diversity-pressure=0.1`, `--well-enough=1` | |
| `--ignore-operator=div --ignore-operator=sin`, `--complexity-ratio=1.5` | |

A 90-flag sweep, a 25-case flag-COMBINATION sweep and a 12-case error-path
sweep each found **zero** cases where both binaries succeed and disagree.

**Every `-H` problem, surveyed 2026-09-10.** Byte-identical: `it`, `pre`,
`pre-conj`, `prerec`, `recall`, `bep`, `select`, `cluster`, `pa`, `dj`,
`maj`, `mux`, `sr`, `cp` — 14 of 18. The other four are not port gaps:
`f_one` reproduces classic's own `AssertionException` (its bscore is
unconstructible upstream), and `ip`/`ann-it`/`ann-cp` are excluded domains
where both binaries fail. Note six of these ABORT in classic at their
defaults and only run once given `-q`/`-w`/`-Q`; a survey that omits those
flags will read them as "both fail" and prove nothing.

**Measure against a clean checkout when other agents are running.** A dirty
worktree produced 23 false "port fails" in one sweep, all of them a concurrent
agent's half-finished edit:

    rm -rf /tmp/clean && mkdir -p /tmp/clean && git archive HEAD | tar -x -C /tmp/clean

### Dispatched 2026-09-09/10, all three landed

1. **Logging surface** — landed; see §4.1, which this closed.
2. **Mixed predicate knob building** — landed; see §1-2, which this closed.
3. **`--ps-max-particles`, `-H pre_conj`, `--time-bscore-granularity`** —
   landed. The first was silently dropped and is now threaded; the second was
   a stale gate on an already-ported scorer and `-H pre-conj` is byte-identical
   to classic; the third did not reproduce as reported and the behaviour was
   pinned rather than "fixed".

### Known-but-unfixed, carried forward

- The `action` half of the port's stated scope is contradicted by the code in
  four places (§5) — an open product decision, not a bug.
- `fs-main.cc` and `eval-features.{cc,h}` are genuinely unported and left
  PLANNED rather than EXCLUDED, because the exclusion boilerplate used for
  `comboreduct/main/*` is factually false for them (§3).
- Coverage rows for `table-problems.*`, `demo-problems.*`, `bscores.*` and
  `table.*` were re-audited after `-H cluster`/`-H sr` landed. All four keep
  status PARTIAL -- each still has a real gap (`--nsamples`/`--balance`,
  the excluded ANN domain, `discretize_contin_bscore`, `CTable::balance`) --
  but their REASONS had gone stale and were rewritten.

### Method notes worth keeping

- **Bisect a scalar budget.** The `linear_contin` bug was localised in three
  runs by bisecting `-m` between an agreeing and a disagreeing run, then
  dumping the field_set on both sides. Much cheaper than diffing final answers.
- **Mutation batches need isolated bytecode.** A same-size file restored within
  one second let CPython reuse a stale `.pyc` and report 0 kills where the true
  answer was 21 (CARRY-FORWARD §5.8). Use `PYTHONDONTWRITEBYTECODE=1`.
- **A zero-kill mutation is ambiguous** — no coverage, or no real trap. Always
  determine which; both occurred this session.

## Order of work, and the reasoning

### 1-2. CLOSED 2026-09-10 — the mixed predicate path and `simplify_predicates`

Both landed together (`215577a`, `8b2e1de`), since item 1 shadowed item 2: the
reduct gap was unreachable through the CLI until knob building opened.

- `build_logical`'s `is_predicate(cur)` branch (`build_knobs.cc:640-663`),
  `logical_canonize`'s (`:160-172`), and the contin branches of
  `insert_typed_arg` (`:386-393`) and `sample_logical_perms` (`:425-430`).
- `simplify_predicates`' `contin_reduce`/`mixed_reduce` tail
  (`logical_rules.cc:76-81`) — the last raise in the whole reduct package.
  Its module-level import cycle is resolved with a function-local import.

Measured: 75/75 byte-identical CLI runs (5 tables x 3 budgets x 5 seeds),
8/8 across flag variations, 40/40 knob-building constructions compared on
decorated exemplar, `ostream_prototype`, and every disc AND contin knob's
spec, order and location, and 280/280 `logical_reduce` results over 137
predicate trees at efforts 2 and 3. `-H select`, which a test previously
asserted *raised*, now reproduces classic's seven lines.

The reproducer this section was written around — a two-contin-column table
with a boolean target — matches, and independently so on a second such
table (8/8 across seeds, budgets, `--linear-regression` and
`--contin-depth`).

**Ten mutants survived and are recorded in the code as equivalent or
unreachable rather than as coverage gaps** — including one that proves
`build_logical`'s `logical_not` assertion arm is never entered at all,
while its sibling check fails 24 tests when forced, so the arm is live and
only that condition is unsatisfiable.

### 3. The P2 tail — what actually remains of 3a-3f

Every item below was re-verified against the tree on 2026-09-09, not inferred
from the plan.

**3a. Scorers — LANDED.** `precision_bscore`, `discriminating_bscore`,
`select_bscore`, `ss_bscore` and `time_dispersion` are all implemented with
oracle fixtures (5/5, 14/14, 4/4, 45 rows, 8/8 respectively), and the port now
registers seven `-H` problems (`it`, `pre`, `prerec`, `recall`, `bep`, `f_one`,
`select`). `bscore_ctable_base` is ported and CARRY-FORWARD §1.2's copy-vs-alias
trap is handled. **Remaining, per row:**
- `discretize_contin_bscore` (`bscores.cc:196-274`) — unported entirely; blocks
  `-R/--discretize-threshold`, which classic runs.
- The `scored_combo_tree_set` ensemble overloads in `behave_cscore`,
  `precision_bscore` and `select_bscore`, plus `exact_selection`/
  `bias_selection` — all of `--boost`.
- `get_timestamp_class` (`time_dispersion.cc:56-74`) and the `TTable` machinery
  behind it (§4.3).
- `ss_bscore` does not subclass `bscore_base` and its ctor raises; it is
  implemented but unconstructible.
- `scored_combo_tree`'s `_deme_id`/`get_demeID()` (`types.h:280-281`), which
  gates the real `--output-deme-id` flag. Its sibling gap, `get_weight()`/
  `set_weight()`, closed DURING this audit in concurrent commit `74fff6c` --
  a live instance of exactly the drift this pass exists to correct.
- Float32 narrowing missing at `discriminating_bscore.py:150,155,244-252,571,583`
  (`--score-weight` path; measured `2.66` vs C++'s `2.6600000858306885`).

**3b. `demo-problems.cc` + `logical_bscore` — LANDED.** All 7 problems register;
`logical_bscore` is ported (relocated into `demo_problems.py`). `-H pa -k3`,
`-H dj -k3`, `-H maj -k3`, `-H mux -k2` and `-H cp -y 'and($1 $2)'` are all
byte-identical to classic. **Remaining:** `-H sr` and `-H cp`'s contin branch,
which need `ITable(const type_tree&, int nsamples, contin_t min, contin_t max)`
(`table.cc:93`), `simple_symbolic_regression` and `largest_const_in_tree` — all
three grep-absent. Their raise messages are stale and blame the wrong thing.

**3c. Metapopulation subsystems — LANDED.** `diversity.cc`, `domination.cc`,
`subsample.cc`, `ensemble.{cc,h}` and `ensemble_params.h` are all PORTED, with
real AdaBoost in `ensemble.py` and `metapopulation.py` containing zero
`NotImplementedError`. The "ported as gates that raise" description is obsolete.
**Remaining:** `merge_demes`' `keep_bscore` recompute arm (`-t/--output-bscore`),
and `--boost`/`--ss-*` remain refused by gaps that live in *other* rows
(`behave_cscore`, `deme_expander`).

**3d. Feature selection — MOSTLY LANDED.** All five algorithms run
(`simple`, `incremental`, `stochastic_max_dependency`, `random`, `hc`); four are
oracle-checked against live classic. `feature_selector.py` reaches all five and
both scorers — the old row claimed only `simple` worked. **Remaining:**
- `--enable-fs` still does not work end to end: `deme_expander.cc:432-450`'s
  real-`fstor` arm needs `behave_cscore.ignore_cols()`/`best_possible_score()`.
- `prune_xmplr` (`deme_expander.cc:308-374`, 67 lines) is unported —
  `--fs-prune-exemplar`. This was previously **misattributed** to
  `feature_selector.cc`.
- `--fs-subsampling-ratio < 1.0` and `--fs-diversity-pressure > 0` raise.
- `mutual_information`'s two continuous branches (`table.h:1168-1188`,
  `1248-1291`) raise.
- `err_empty_features`, `log_selected_features`, `write_results` and the
  top-level `feature_selection()` need `ostreamTable`/`saveTable` (§4.4).
- `fs-main.cc` and `eval-features.{cc,h}` are **verified unported** — see §5.2.
- **No oracle is possible for `hc`**: classic's `feature-selection --algo hc`
  dumps core unconditionally (re-confirmed live 2026-09-09, exit 139).

**3e. The other optimizers + `eda/` — LANDED, with two measured divergences.**
All four optimizers are constructed for real and selectable. Measured through
the CLI (5 seeds × 2 budgets on `trainingBooleanXor.dat`, plus
`trainingBoolean.dat`): **`-ahc` 10/10 byte-identical, `-asa` 10/10, `-aun` 6/10,
`-aps` 4/10** (0/6 on `trainingBoolean.dat`). **Remaining:**
- **`-aps` port defect, root cause isolated and proven by patch.**
  `particle-swarm.cc:326-327`'s increment list omits `disc_best_it`, so classic
  always dereferences `disc_parts.best_personal[0]`; `particle_swarm.py:474`
  passes `[i]`. Freezing that index takes XOR 4/10 → 10/10 and
  `trainingBoolean` 0/6 → 6/6.
- **`-aun`: two causes, both in `eda/`.** (i) `digraph.h:96`'s bare
  `std::random_shuffle` over unseeded `std::rand()` — reimplementing glibc's
  TYPE_3 `rand()` raises `-aun` to 9/10; (ii) `roulette_select` is instantiated
  at `local_structure.cc:146` with `ScoreT = int`, where C++ truncates and
  `selection.py` does not — 10–14 disagreements per 4,768 leaf calls.
- `rtr_replacement` (`replacement.h:68-89`) unported but unreachable
  (its only reference is a commented-out line).
- `print_stats_header` (`optimization.cc:140-186`) unported, one live call site.

**3f. The remainder — RESOLVED, mostly by exclusion.**
- `similarity.{cc,h}` — **PORTED** (51 tests), was stale as PLANNED.
- `partial.{cc,h}` — **EXCLUDED.** `partial_solver` is enum-domain-only by its
  own typedefs (`partial.h:130`, `:134`) and its sole construction site sits
  inside `output_type == id::enum_type`. Measured: classic `--well-enough=1` on
  a boolean table is byte-identical to the no-flag baseline.
- `distributed_moses.{cc,h}` — **EXCLUDED by scope decision, as instructed.**
  It shells out (`ssh` at `cc:96`, `system()` at `:173`, `popen("pgrep")` at
  `:182`, `popen("fuser")` at `:197`, `killall` at `:315`/`:324`) and
  re-serialises `boost::program_options::variables_map` back into argv at
  `cc:84`. Zero combo/knob/scoring semantics; a Python version would be a
  subprocess rewrite, not a word-for-word port. **Do not port it.**
- Residual vertex variants — see §5.1, the action-domain question.

### 4. Cross-cutting gaps that no single row owns

These surfaced repeatedly across the audit and are bigger than the rows that
mention them.

1. **CLOSED 2026-09-10 — the logging surface exists.** cogutil's `Logger` is
   ported (`pymetta/util/logger.py`), `-l`/`-f`/`-F` are wired (`-F` is not a
   filename: it is `determine_log_name`, which names the log after every
   option the user actually TYPED), and the port has 196 `logger()` call
   sites where it had zero. `tools/normalize_log.py --compare` is the oracle
   and `tests/moses/main/test_log_differential.py` makes it permanent; it
   asserts EQUALITY after normalising volatile fields, not that the port is a
   subsequence. Measured on two configurations: 139/139 and 149/149 records
   matched, **zero unexplained in either direction**, with a 34-entry
   allow-list of records whose emitting function is not ported (enumerated in
   the test). What remains is that allow-list shrinking, not a missing
   surface.
2. **The type-inference engine is unported** — 41 of `type_tree.cc`'s 73
   definitions, verified by exact-word grep returning zero definitions.
   `demo_problems.py:94-121` reaches it through a disclosed narrow substitute
   that cannot detect a cross-argument type mismatch.
3. **Timestamps are parsed and discarded** (`table_io.py:501-502`, `:532`).
   No `TTable`/`TimedValue`/`CTableTime`, and `CTable` has no `get_timestamps()`
   -- concurrent commit `9793e5a` pinned that absence rather than closing it.
   This alone blocks `ss_bscore`, `time_dispersion`'s `get_timestamp_class`, and
   time-based subsampling.
4. **The `ostreamTable`/`saveTable` half of `table_io.cc` is unported** (17
   functions). Mostly EXCLUDED-binary-only, but it is the blocker for four
   `feature-selection.cc` functions.
5. **`CTable.filtered_preserve_idxs` is absent**, which is what makes
   `scoring_base`'s `ignore_cols`/`ignore_rows`/`ignore_rows_at_times` raise.
6. **`randfloat` raises** (`rng.py:107`) and is no longer unreachable: three
   live call sites (`eval.py:295`, `interpreter.py:434`,
   `feature_selector.py:391`). Needs C float32 truncation.

### 5. Decisions someone must make (not code)

1. **Is the action domain in scope?** This document and `MODULE-COVERAGE.md`
   both say the port's domains are "Boolean, continuous and action", but the
   code has already taken the excluding reading everywhere:
   `deme_expander.py:311` raises for perceptions/actions,
   `general_rules.py:112-119` hard-codes `is_action_result` to `False`,
   `complexity.py:79` lists `action_result_type` among types never seen, and
   `build_knobs.py:65` puts it in `_FUTURE_OUTPUT_TYPES`. Four rows
   (`vertex.{cc,h}`, `knobs.{cc,h}`) are PARTIAL solely on the stated reading
   and would be PORTED under the other. **Settle it in prose before anyone
   ports `is_action`/`get_action`.**
2. **Does pyMeTTa ship a standalone `feature-selection` CLI?** `fs-main.cc`
   (453 lines) and `eval-features.{cc,h}` (273) are verified unported — zero
   lines of Python, no `[project.scripts]` in `pyproject.toml`, no
   `__main__.py`. They were left `PLANNED: P2` rather than EXCLUDED because the
   boilerplate used to exclude `moses/comboreduct/main/*` ("the spec's layout
   lists no Python equivalent") is **factually false** here — a
   `pymetta/feature_selection/main/` package exists with two ported modules —
   and because the analogous C++ `main()`, `moses/moses/main/moses-exec.cc`,
   IS ported. Note `eval-features` is the only oracle for `MICScorer`, which
   currently has zero test coverage.
3. **`type_tree_def.cc`** (41 lines, 11 of body: `is_argument_type`,
   `arg_to_idx`) is a strict subset of the `type_tree.cc` gap. All three of its
   classic call sites are inside the unported inference engine. Consider folding
   its row into §4.2 rather than tracking it separately.
4. **The CogUtil side of the inventory is not exhaustive.** The port carries
   breadcrumbs into 18 `opencog/util/` files; 7 have rows. The eleven without
   are listed in `MODULE-COVERAGE.md`'s CogUtil section with their breadcrumb
   counts. `digraph.h` was one of them until 2026-09-09.

### 6. Port defects found during the audit, reported and NOT fixed

Logged here because each is a live behavioural difference, not a missing
feature. None was fixed — that work belongs to whoever owns `pymetta/**`.

- **`particle_swarm.py:474`** misses classic's frozen `disc_best_it` (§3e).
- **`--ps-max-particles` is silently dropped.** Parsed (`problem_params.py:418`),
  stored (`:635`), then discarded — `moses_main.py:553` constructs
  `particle_swarm(opt_params, ps_parameters())`. Classic wires it at
  `problem-params.cc:2370` and it changes classic's output. This violates S13's
  own governing rule: an unwired flag must raise, not be ignored.
- **`--well-enough` over-gates.** `problem_params.py:1057` raises at parse time,
  but classic accepts and ignores the flag off the enum path (measured: exit 0
  with real results on a boolean table). The gate belongs behind the
  enum-output check.
- **`-H pre-conj` is a stale gate.** `precision_conj_bscore` is fully ported
  (`precision_bscore.py:415`) with the exact signature `table-problems.cc:421`
  needs, yet `problem.py`'s stub still blocks it. Classic emits 5 result lines.
- **`--time-bscore-granularity` makes the port MORE permissive than classic.**
  Because the ctor never calls `get_timestamp_class`, it never reaches
  `time_dispersion.cc:62`'s assert. Measured: classic aborts with an
  AssertionException and a core dump; the port completes and prints candidates.
- **`MICScorer.__call__` consumes `fs` twice** (`mutual_info.py:305-308`),
  silently degrading confidence for a generator argument. Measured:
  `0.5487949406953987` vs `0.4878177250625766`. No caller passes a generator today.
- **`mutual_information_between_sets` raises for non-discrete inputs**
  (`mutual_info.py:227-232`) where C++ `mutualInformationBtwSets`
  (`table.h:1311-1391`) has no input-type check at all. Undocumented divergence.
- **`stochastic_max_dependency.py:116`** relies on incidental `set` ordering
  where C++ iterates a `std::set` ascending. Safe for the in-tree call path,
  wrong for non-contiguous input; the function is public in `__all__`.
- **`foldl` (`eval.py:480-519`) carries no `# DIVERGENCE` marker** although it
  builds a fresh tree where `eval.cc:462-503` mutates in place (and thereby
  avoids classic's use-after-free at `cc:479-481`).

**Stale claims in `pymetta/**` source** — each is a confident, checkable, false
statement of the kind CARRY-FORWARD §6 names as this project's worst defect:
`feature_selection/__init__.py:9-13`, `algo/simple.py:4-5`,
`main/feature_selection_params.py:67-69,159-161`,
`moses/deme/feature_selector.py:6-9,27-37` (all: "only `simple` is ported");
`scorers/mutual_info.py:173,295` (cite a module that does not exist);
`moses/main/problem_params.py:56-58,304`, `moses/moses/moses_main.py:19-22,90-95`
(`ps_parameters`/optimizers "not ported"); `moses/moses/neighborhood_sampling.py:16-22`
(contin helpers "still raise"); `moses/scoring/select_bscore.py:54-76`
(mixed interpreter "out of scope"); `moses/main/table_problems.py:546,627,697`
(`feature_selector`/`partial_solver` "not ported anywhere");
`reduct/reduct.py:32-37`; `reduct/logical_rules.py:63-64,540-579`;
`moses/main/demo_problems.py:659,818`; `moses/main/problem_params.py:980-1001`;
`tests/moses/deme/test_feature_selector.py:225` (test name now false).

### 7. Documentation debt this pass did not clear

- **`docs/PARTIAL-AUDIT.md` is stale.** It classifies 62 PARTIAL rows; there are
  now 82, and at least four of its verdicts are overtaken (`instance_scorer.h`
  is now PORTED, `bscore_ctable_base` is done, `logical_bscore` is ported, and
  the real survivor of its item 3 is `discretize_contin_bscore`). It was not
  rewritten here because it was outside this task's ownership.
- **`docs/CARRY-FORWARD.md` §1.2** still says of `bscore_ctable_base` "Nothing
  ports it yet". It is ported and the trap is handled.
- **Section intros in `MODULE-COVERAGE.md`** were rewritten for fourteen
  sections on 2026-09-09, but `coverage_totals.py --check` still validates only
  the rows. Prose can go stale again and nothing will catch it.
- **No fixture manifest records a compiler/stdlib version** (CARRY-FORWARD
  §5.1). S12's fixtures depend on libstdc++'s sort permutations, and §3e's
  `-aun` finding adds glibc's `rand()` to the same exposure.
- **Error types are inconsistent** — `OC_ASSERT(false)` maps to six Python
  types; "column not found" is `IndexError` in `table.py` and `ValueError` in
  `table_io.py`. Needs one C++-grounded mapping decision, then a wide sweep.
- ~~**Corpus gaps** — no argument above `$3`, no arity ≥ 4.~~ **CLOSED
  2026-09-09** by `tests/moses/main/test_higher_arity_oracles.py`: ten CLI
  oracles at arity 4/5/6, boolean and contin, captured from classic's own
  binary and byte-identical. Their expected outputs reference `$x4` 97
  times, `$x5` 41 and `$x6` 5, so the coverage is real rather than
  nominal, and the boolean targets genuinely depend on the high-numbered
  arguments (5-input majority; a 6-input mix reading `$1`, `$2`, `$4`,
  `$6`) so a port mishandling them could not match by accident.
- **`tree.json` has no shape guard**, and `test_tree.py`'s 33 oracle tests do
  not use `parametrize`, so `empty_parameter_set_mark` does not cover them.

## Standing rules for whoever continues this

- **Closeout is part of the slice.** S9, S10 and S11 were each merged with their
  coverage rows still saying PLANNED — and the 2026-09-09 audit found 71 more
  stale rows on top of that, including rows stale in the INVERSE direction
  (PORTED while the module gated, PARTIAL while the gap had closed).
  `coverage_totals.py` now catches four mechanical shapes of this; it cannot
  catch "this row's named gap has closed", and the tool says so in
  `WHY_NO_PARTIAL_TO_PORTED_CHECK`. Do not rely on it.
- **A row is PORTED only if the whole C++ file is ported.** A file narrowed by
  an excluded domain is PARTIAL with the exclusion named — that is this
  document's convention, and it is what keeps "PORTED" meaning something.
- **A deliberately-unasserted field is where a real divergence hides.** The
  13-vs-17 deme size was waived twice as benign tie-order and turned out to be
  the stable-sort bug (CARRY-FORWARD §5.2).
- **A marker must state what C++ actually does.** This project's most-repeated
  defect; §6 above lists sixteen live instances.
- **Verify the adjacent dimension.** Three times an oracle or fix was confirmed
  along one axis and wrong along another (CARRY-FORWARD §5.6). The 2026-09-09
  audit added a fourth shape: a row whose *reason* named a real gap while its
  *status* claimed completeness, and which every automated check passed.
- **A large matching corpus is not coverage.** S9's oracle matched 4,500
  instances while walking one control-flow path.
