# Remaining work to a complete port

Written 2026-09-08 after S12 merged. Rewritten 2026-09-09 against a full re-audit
of every PLANNED and PARTIAL row in `docs/MODULE-COVERAGE.md`. **Rewritten again
2026-09-11**, against a second such re-audit that re-measured all 82 remaining
PARTIAL/PLANNED rows and found the inventory stale in one direction throughout:
rows describing work as missing after it had landed. This is the durable roadmap:
it survives context loss and says what "done" means, in what order, and why.

**Scope reminder.** 88 files / 12,212 lines are **EXCLUDED by decision**, not
deferred: legacy embodiment, dead `#ifdef` code, C++-only tools, and the enum,
ANN, list/fold, object, message, procedure-call and term domains, plus
`partial.{cc,h}` and `distributed_moses.{cc,h}` (see §3(b)). The port's domains are
**Boolean and continuous**; the action domain was settled as EXCLUDED on
2026-09-10 by measurement (§5.1) and is no longer an open question. "Complete"
therefore means every non-excluded file ported, with its named gaps closed.

## Status

Regenerate with `python3 tools/coverage_totals.py`; never hand-edit.

| Status | Files | Lines |
|---|---:|---:|
| PORTED | 89 | 16,081 |
| PARTIAL | 70 | 34,130 |
| PLANNED: P2 | 0 | 0 |
| EXCLUDED | 88 | 12,212 |
| **Total** | **247** | **62,423** |

**The PLANNED bucket is empty. Every non-EXCLUDED file in the inventory now has
Python behind it.** The last PLANNED row, `type_tree_def.cc` (41 lines), was found
already ported at `pymetta/comboreduct/type_checker/type_signature.py:115,:126`.
`tools/coverage_totals.py` keeps rendering the bucket at zero rather than dropping
it -- see its `MAY_BE_EMPTY` note -- so the milestone stays visible in the table.

This file's own Status table was itself stale when the 2026-09-11 audit started:
it read PORTED 73 / PARTIAL 82 / PLANNED 4 against a real 77 / 81 / 1, and its
"four files have no Python whatsoever" sentence named `fs-main.cc` and
`eval-features.{cc,h}`, all three of which had been PORTED since 2026-09-10.

Measured against the previous commit, the 2026-09-11 audit **changed the status of
12 rows** -- PARTIAL->PORTED 11, PLANNED->PORTED 1 -- and rewrote the REASON on
essentially all of the other 70. Promoted: `logical_rules.{cc,h}`,
`type_tree_def.{cc,h}`, `eda/optimize.h`, `demo-problems.h`, `local_moses.h`,
`hill-climbing.{cc,h}`, `univariate.cc`, `time_dispersion.{cc,h}`.

**A 13th promotion was attempted and REVERTED by the tooling**, which is worth
recording because it is the guard working: `univariate.h` was promoted to PORTED,
and `check_ported_rows_do_not_gate` caught it -- `univariate.py:157` raises, and
`_gates_in` attributes that raise to the header. It is held at PARTIAL with the
reason naming both the raise and the fact that the raise is *faithful*
(`univariate.cc:88-90` is itself `OC_ASSERT(false)`).

For the prior pass's record: the 2026-09-09 audit changed 71 of 246 rows and added
one (`opencog/util/digraph.h`, fully ported with no row at all), moving the total
246 -> 247.

## State handoff — end of the 2026-09-09 session

Written so the next execution resumes from measurement rather than re-deriving
it. `bash tools/ci.sh` is **green (exit 0, 5242 passed, 5 skipped)** at this
point, and the working tree is clean.

### COMPLETION PLAN — CLOSED 2026-09-11

All six workstreams landed. Measured at the closing commit:

| Measure | Result as claimed | Re-measured 2026-09-11 |
|---|---|---|
| CLI flags where both binaries succeed and disagree | **0** | **NOT 0** — see §6, at least `--boost` on three demo problems |
| Flags byte-identical (90-flag sweep) | **82** — the rest classic rejects at every value | unchanged for the 82 |
| Flags classic accepts that this port refuses | **0** | **NOT 0** — at least 6: `--output-format=scheme`, `-t1`, `--nsamples`, `--ss-n-subsample-demes`, `--ss-n-subsample-fitnesses`, `-K/-J/-U/-X` |
| End-to-end differential (20 configurations × 3 seeds) | **60 / 60 byte-identical** | unchanged |
| Optimizers (`hc`/`sa`/`un`/`ps`) | all byte-identical, including `-m1500` | confirmed |
| `-H` problems | 14 byte-identical; 4 fail exactly where classic fails | `-H ip` reclassified: it is LIVE in classic, not a shared failure |
| `tools/ci.sh` | green — 7345 passed, 5 skipped | green |
| `PLANNED` coverage | 1 file / 41 lines | **0 files** |

> **The two "0" rows above were wrong, and the way they were wrong is this
> project's documented failure mode (CARRY-FORWARD §5.10): a sweep that finds
> nothing must be checked for whether it *could* have found anything.** The
> 90-flag sweep exercised `--output-format` only at `combo` and `python`, never
> at `scheme`; `scheme` is the one value that diverges. `--output-format` appears
> nowhere in this file, and no test in the port mentions `scheme` at all. The
> re-measured column above was produced by running both binaries on each flag,
> not by re-reading the sweep.

The rest of this section's original text is kept for the record, and its
"four `raise`s remain in `problem_params.py`" framing is accurate as far as it
goes -- but the claim that "none blocks a command line classic accepts" is
false for two of the four (`--ss-*` and `-t1`), as §3(a) now records:
an unknown `--opt-algo` (classic `exit(1)`s too),
`-j host:N` distributed dispatch (EXCLUDED by scope decision), and the
`ss-*` family (`deme_expander`'s subsample-splitting branch).

**What is NOT byte-identical, and cannot be.** Three places where classic's
own behaviour is unspecified or undefined, each marked DO NOT FIX with the
measurement behind it:

1. **Boosting member order** — `add_adaboost` picks with `min_element` over
   a `std::unordered_set`, so ties are broken by hash-bucket order. Measured:
   4 of 4 candidates tied on one call, 12 of 32 on the next; 11 of 18 runs
   match classic's score, and the port sometimes scores *better*, which is
   what rules out an arithmetic error.
2. **`TimedCounter` insertion order** — `TimedValue::operator<` is not a
   strict weak ordering (`table.h:482`), so the `std::map` behind every
   timestamped CTable cell is UB. 1554 insertion sequences measured: 533
   reorder, 190 lose observations, 30 duplicate a key.
3. **`std::unordered_set` iteration** wherever else it decides output order.

**Upstream defects reproduced, not fixed** (each with a probe): classic
SIGSEGVs under `--time-dispersion-pressure` on a timestamped table (rc 139,
no diagnostic, 18/18 configurations) where the port completes; aborts on
`--time-bscore-granularity=month` without a timestamp column; `--fs-algo hc`
null-derefs; `--fs-demes=2` aborts; `--fs-seed=xmplr` SIGSEGVs;
`logical_reduce` never terminates on `0<(log(-1))`; and an argument index
past the table's arity reads out of bounds and reports a *perfect* score.

### Historical: the plan as written 2026-09-10

Everything below is measured, not estimated. Six workstreams, partitioned so
no two touch the same files. A/B/C/D dispatched in parallel; E/F follow because
they contend for `table_problems.py`.

| # | Workstream | Closes | Owns |
|---|---|---|---|
| A | `-aun` univariate divergence | the LAST behavioural divergence | `moses/eda/**`, `util/selection.py`, `util/digraph.py`, `optimization/univariate.py` |
| B | `TTable`/timestamps + `CTable::balance` | `--balance`, `--nsamples`, `--timestamp-feature`, `--time-bscore`, `--time-dispersion-pressure` | `comboreduct/table/**`, `scoring/time_dispersion.py` |
| C | Boosting + demeID | `--boost`, `--output-deme-id` | `metapopulation/**`, `moses/types.py`, `scoring/behave_cscore.py`, `precision_bscore.py`, `select_bscore.py`, `moses_main.py`, `main/demo_problems.py` |
| D | Type-inference engine | 41 of `type_tree.cc`'s 73 defs, `type_tree_def.cc` | `comboreduct/type_checker/**` |
| E | Feature selection end to end | `--enable-fs`, `prune_xmplr`, `fs-main.cc`, `eval-features.{h,cc}` | `feature_selection/**`, `deme/deme_expander.py` |
| F | `discretize_contin_bscore` | `--discretize-threshold` | `scoring/bscores.py` |

**Done means:** zero flags where both binaries succeed and disagree; zero flags
classic accepts that the port refuses, except where the refusal is itself
faithful; no file with zero Python outside EXCLUDED; and `tools/ci.sh` green.

**Not code, and not mine to settle:** the action-domain question (§5). Four
coverage rows hinge on it.

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

### 3. What is actually left — the definitive inventory, 2026-09-11

Every `raise NotImplementedError` in `pymetta/` was located by AST walk (not
grep) and attributed to a C++ file using `tools/coverage_totals.py::_gates_in`,
which reads the enclosing symbol's `C++:` breadcrumb. **70 raises**, classified
into exactly four kinds. The counts partition: 21 + 20 + 28 + 1 = 70.

| Kind | Count | Meaning |
|---|---:|---|
| **(a)** real unported work in scope | **21** | a user could ask for this and not get it |
| **(b)** excluded-domain narrowing | **20** | correct and permanent; the domain is named |
| **(c)** deliberate divergence | **28** | abstract-base stand-in, or a C++ `OC_ASSERT(false)`/UB reproduced |
| **(d)** stale claim corrected | **1** | the guard is now dead code |

Three raises carry **no C++ breadcrumb** and are therefore invisible to
`check_ported_rows_do_not_gate`: `demo_problems.py:1030`, `problem.py:260`,
`partial.py:178`. A PORTED row could be introduced above any of them without the
tool noticing. That is a tooling gap, recorded in §7.

#### (a) Real unported work in scope — 21 raises, 12 distinct features

Listed in full, because this is the "what is left" list. **Ten of these are
CLI-reachable divergences where classic succeeds and the port does not** --
which falsifies the completion plan's "zero flags classic accepts that the port
refuses". Each was measured with both binaries.

| # | Site(s) | Feature | Measured |
|---|---|---|---|
| a1 | `iostream_combo.py:435`, `:467`, `:559` | `--output-format=scheme` | **classic exit 0** printing `0 (OrLink (NotLink (PredicateNode "$1")) ...)`, **port exit 1**. `combo` and `python` formats are byte-identical; only `scheme` diverges. No test anywhere in the port mentions `scheme`. |
| a2 | `deme_expander.py:363`, `:380`, `:852`, `:925` | `--ss-n-subsample-demes` | deme-splitting arms of `subsample_by_time`/`subsample_by_row`; classic exit 0, port exit 1 |
| a3 | `moses_main.py:766` + `problem_params.py:1020` | `--ss-n-subsample-fitnesses` | `metapop_moses_results`' SS-fitness branch (`moses_main.h:364-378`) unwired; classic exit 0, port exit 1 |
| a4 | `merging.py:615` + `problem_params.py:1092` | `-t` / `--output-bscore` | `keep_bscore` is the 4th disjunct of `merging.cc:265`; classic `-t1` exits 0 with 10 result lines, port exits 1 |
| a5 | `table_problems.py:510` | `--nsamples` | classic exit 0, port exit 1. The raise text blames `subsampleTable`, which IS ported at `fs_main.py:183` -- only the call is missing |
| a6 | `demo_problems.py:1030` | `-H cp` with an un-modelled builtin | `_infer_signature`/`_BUILTIN_INPUT_TYPE` covers 11 builtins as a stand-in for `infer_type_tree`. `-H cp -y 'contin_if($1 1 2)'`: classic exit 0 with 2 result lines, port exit 1 |
| a7 | `problem.py:260` (the `ip` half) | `-H ip` | `interesting_predicate_bscore` (`bscores.h:553`) unported. **`-H ip` is live in classic** (exit 0, prints `nan true` on boolean-in/contin-out), so this is reachable, not academic. The same raise's `ann-it` half is (b), ANN |
| a8 | `scoring_base.py:629`, `:640`, `:652` | `ignore_cols`/`ignore_rows`/`ignore_rows_at_times` | real C++ bodies at `scoring_base.cc:220/:267/:290`; blocked on `CTable.filtered_preserve_idxs` (`table.h:661`). NOT observable via `--enable-fs` (byte-identical) but fatal on the `ss_bscore` path |
| a9 | `precision_bscore.py:318` | `do_score` weighted normalization | `precision_bscore.cc:282-289`; `-Hpre -q0.1 -w0.9 --boost=1`: classic exit 0 printing `-inf or`, port exit 1 |
| a10 | `rng.py:107` | `randfloat` | needs C float32 truncation (`struct.pack('f', ...)`). Three live call sites: `eval.py:295`, `interpreter.py:460`, `feature_selector.py:748`. Reachable via `--fs-enforce-features-filename`, where classic exits 0 |
| a11 | `moses_main.py:404` | `--output-format=python` with `--boost` | needs per-member `get_weight()`; `get_weight` now exists (`types.py:708`) but `ensemble.py` keeps AdaBoost weights in a private identity-keyed dict, so every member would print `1.0`. Gap belongs to `ensemble.py` |
| a12 | `neighborhood_sampling.py:183` | boost `binomial.hpp:44-53` beta fallback | first argument > 170. **Real but not CLI-demonstrable**: `--contin-depth` 171/200/400 all exit 0 byte-identically without reaching it |

**Real in-scope gaps that are NOT raises** (they are plain absences, which is
why no automated check can find them -- see `WHY_NO_PARTIAL_TO_PORTED_CHECK`):

- `ip_problem_params` (`table-problems.cc:204-227`) entirely unported, so
  classic accepts `-K`/`-J`/`-U`/`-X` (`--ip_kld_weight` and friends) at exit 0
  where **the port exits 2 with "unrecognized arguments"**. This is the one
  place the option surface is not symmetric, and the old intro claimed it was.
- `mutual_info.h:126-127`'s `logger().fine` records: classic writes 16 under
  `--fs-algo=smd -l FINE`, the port writes 0, though `logger().fine()` exists.
- `build_knobs.py` has **zero** `logger()` call sites against 27 in the `.cc`;
  a `-l DEBUG` run drops exactly 30 records.
- `moses_main.py` has **zero** `logger()` call sites, so `moses_main.cc:82`,
  `:84`, `:92`, `:110` are dropped -- classic's default log carries
  `Inferred target score = 0` and the port's does not.
- `eda/logging.py:36` is a bare `return`; `-aun -m1000 -r1 -l DEBUG` drops
  classic's 2 `Generation:` + 2 `Best instance:` records.
- `particle-swarm.cc:216`'s `log_stats_legend` plus all 7 `logger()` sites:
  5 records missing at `-aps -l DEBUG`.
- `metapopulation.cc:236`'s `ostream_metapop`, observable at the **default** log
  level via `moses_main.h:220` (3-line "Best candidate:" record absent).
- `local_moses.cc:194-199` is a bare `pass`: under
  `-l DEBUG -Z1 --diversity-pressure=1` classic emits a 7-line "Top 10
  candidates" record and the port emits none.
- `opencog/util/RandGen.h:72` / `mt19937ar.cc:74` `rand_positive_negative` has
  **no Python counterpart at all** and was never named by any row. Zero
  in-scope callers, so not reachable -- but it is a real absence, not a stale
  claim.
- `types.cc`'s parsing half -- `string_to_scored_combo_tree` (cc:209) and
  `istream_scored_combo_trees` (cc:291). Real; the only callers are
  `eval-diversity.cc:156` (EXCLUDED) and four cxxtests.
- `field_set.cc:205-263`'s `ostream_field_set` (59 lines). Real, and dead on
  **both** sides -- nothing under `moses/` or `tests/` streams a `field_set`.
- `tree.h`: 4 of 8 iterator kinds ported; `fixed_depth_iterator`,
  `breadth_first_queued_iterator` and `upwards_iterator` absent, along with
  ~20 members (`clear`, `merge`, `depth`, both `equal`, both `subtree`, all
  four `prepend_children`, …). The row's old list understated this.
- `selection.h`'s 3-argument `accumulate` overload (h:103-110) and
  `NodeSelector` (h:112-150); only consumer is `tree_gen.cc:50` (EXCLUDED).

#### (b) Excluded-domain narrowing — 20 raises, correct and permanent

Each names its domain. These are why rows stay PARTIAL rather than PORTED, and
that convention is what keeps "PORTED" meaningful.

| Domain | Sites |
|---|---|
| **enum** | `table.py:373`, `deme_expander.py:157`, `merging.py:667` (the only `merge_callback` setter in classic is `partial.cc:86`), `partial.py:178`, `table_problems.py:344`, `:372`, `:720`, `demo_problems.py:339`, `:368` |
| **procedure-call** | `eval.py:124` (`eval_procedure_tree`; unreachable because `is_procedure_call` returns False for every constructible vertex) |
| **ANN** | `demo_problems.py:1343` (`-H ann-cp`; classic itself aborts 134) |
| **action / perception** (embodiment) | `deme_expander.py:493`, `build_knobs.py:265` |
| **enum + object + message + action-symbol/wild-card** | `type_tree.py:504` |
| **all non-boolean/contin input or output** | `build_knobs.py:268`, `:313`, `:382` |
| **MPI** (dead `#ifdef`, EXCLUDED row) | `moses_main.py:275` |
| **`distributed_moses`** (scope decision, EXCLUDED row -- not one of the eight domains) | `moses_main.py:284`, `problem_params.py:940`. Measured: `-j 4:somehost` makes classic *hang to timeout* (rc 124), so classic does not succeed here either |

#### (c) Deliberate divergences — 28 raises

Two shapes, both faithful.

**Abstract-base stand-ins for a C++ pure virtual** (15): `fs_scorer_base.py:55`,
`lazy_random_selector.py:109`, `optimization.py:243`, `instance_scorer.py:42`,
`discriminating_bscore.py:572`, `:580`, `scoring_base.py:152`, `:199`,
`problem.py:149`, `:158`, `:172`, `demo_problems.py:694`, `:699`. Each mirrors a
`= 0` declaration whose concrete overrides all exist.

**C++ `OC_ASSERT(false)` or UB reproduced** (13) -- classic fails at the same
point, so no user can get behaviour from classic that the port withholds:

- `interpreter.py:831`, `:841` -- `table.h:341-350` is `OC_ASSERT(false, "Not implemented")`
- `ss_bscore.py:444`, `:452` -- `ss_bscore.cc:146`, `:150` are the same stub
- `table.py:391` -- `table.cc:118`, release-live `OC_ASSERT(false, "Not implemented yet")`
- `univariate.py:157` -- `univariate.cc:88-90`; unreachable because `eda_parameters` has no call site under `$CLASSIC/moses/moses/main`, so `selection` is always its default 2
- `feature_selector.py:493` -- `feature_selector.cc:196`
- `feature_selection_params.py:208` -- `feature-selection.h:178`
- `scoring_base.py:179` -- `scoring_base.cc:73-77`; measured, classic's own message on `-Hit --boost=1`
- `demo_problems.py:924`, `:1200`, `table_problems.py:897` -- the three "Boosting not supported" asserts (`demo-problems.cc:275`, `:382`, `table-problems.cc:419`). Measured: classic exit 134, port exit 1, same message
- `problem_params.py:935` -- unknown `--opt-algo`; both binaries exit 1
- `rng.py:115`, `:122` -- `std::uniform_real_distribution` / `std::discrete_distribution`, both implementation-defined and with zero call sites in classic `moses/`

Non-raise divergences in the same class, kept for the record: `tree.py` poisons
discarded nodes where C++ dangles (CARRY-FORWARD §1.3); `update_best_candidates`
emits 4 DEBUG records in classic and 2 in the port because
`scored_combo_tree_set` is a `std::unordered_set` (final state and all stdout
identical); `TimedValue::operator<` is not a strict weak ordering (`table.h:482`)
so every timestamped `CTable` is UB upstream.

#### (d) Stale claims corrected — 1 raise

`ss_bscore.py:365`. Its guard is `hasattr(ctable, "get_timestamps")`, and
`CTable.get_timestamps` now exists at `table.py:1595`, so **the branch is dead
code**: `ss_bscore(precision_bscore(ct), 2, 0.5, True)` constructs successfully
with `_timestamps == set()`. The raise should be deleted by whoever owns
`pymetta/**`; it is reported, not fixed.

### 4. Cross-cutting gaps that no single row owns

1. **The logging surface exists but is not complete.** 231 real `logger().x()`
   call sites across 27 modules, counted by AST walk. What remains is the
   per-module holes listed under §3(a) -- `moses_main.py` and `build_knobs.py`
   at zero, `eda/logging.py` a bare `return`, `particle-swarm`'s seven sites,
   `ostream_metapop`, `local_moses.cc:194-199`, `mutual_info.h:126-127`. The
   2026-09-09 text calling this "no logging surface at all" across seven rows
   was already false and has been removed from all of them.
2. ~~**The type-inference engine is unported**~~ -- **CLOSED.** `type_tree.cc`
   is 0 of 73 definitions unported, down from the claimed 41. Verified by
   running it: `infer_type_tree` on `and($1 plus($1 1))` returns
   `->(ill_formed ill_formed)`, `is_well_formed` False.
3. ~~**Timestamps are parsed and discarded**~~ -- **CLOSED.** `TTable`,
   `TimedValue`, `TimedCounter` and `CTableTime` are all in `table.py`
   (`:828`, `:960`, `:1050`, `:1372`), and `table_io.py:556-575` fills
   `Table.ttable`.
4. ~~**The `ostreamTable`/`saveTable` half of `table_io.cc` is unported**~~ --
   **CLOSED**, into `main/table_output.py:73/:96/:133`. A placement divergence,
   not an absence.
5. **`CTable.filtered_preserve_idxs` is still absent**, and is the single
   blocker behind §3's a8.
6. **`randfloat` still raises** (`rng.py:107`) -- §3's a10.
7. **The CogUtil side of the inventory is not exhaustive, and is worse than
   recorded.** 24 breadcrumbed `opencog/util/` files, 7 rowed, **17 unrowed**
   (4,194 C++ lines) -- the previous count of 18/7/11 was wrong in every term.
   Two of the unrowed are dedicated ports meeting the exact bar `digraph.h` got
   a row for: `Logger.{cc,h}` (1,052 lines -> `util/logger.py`, 609 lines) and
   `lazy_selector.{cc,h}` (203 lines -> inside `lazy_random_selector.py:22-122`).


### 5. Decisions someone must make (not code)

1. **Is the action domain in scope? — SETTLED 2026-09-10 BY MEASUREMENT: no,
   and the code was right.** `moses` itself cannot run an action-domain
   problem. `alphabet_size` (`problem.cc:184-196`) lists `action_result_type`
   among the types that fall through to `log_output_error_exit`, and driving
   classic's own binary confirms it:

       moses -Hcp -y 'and_seq($1 action_success)' -m200 -W1
       moses -Hcp -y 'action_boolean_if($1 action_success action_failure)' ...
       moses -Hcp -y 'and_seq($1 $2)' ...
         -> all three: AssertionException, rc 134,
            'Error: type "action_result" not supported'

   No `-H` problem can produce action output either: the table problems force
   `boolean_type` or infer from a CSV, and a CSV cannot hold an action value.
   The action domain in classic lives only in the embodiment code
   (`ant_combo_vocabulary`, reached by `AntUTest`), which this project's own
   CLAUDE.md already flags as legacy -- `ActionReductUTest` and
   `PerceptionReductUTest` are commented out of the build.

   So the four code sites below are FAITHFUL, and what needs correcting is the
   PROSE in this document and `MODULE-COVERAGE.md` claiming action is a
   supported domain. The port refuses these programs too, at a different point
   (its combo parser, `ValueError`, rc 1, rather than type inference) -- worth
   a marker if anyone cares, but the observable contract is the same: neither
   binary runs an action program.

   The original wording, for the record -- the code sites it lists are all
   still there and are all now known to be correct:
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

#### Found by the 2026-09-11 re-audit (new)

Each was measured with both binaries. **These are the ones that falsify the
"zero flags where both binaries succeed and disagree" claim**, because in the
first case both sides exit 0 and print different answers.

- **`--boost=1` diverges with BOTH binaries at exit 0.** `-H pa -k3 -m100`,
  `-H maj -k3 -m500`, `-H mux -k2 -m500` produce different ensembles (classic
  `-2 0<(...3 terms)` vs port `0 0<(...4 terms)` at `-m500`). `-m20`/`-m50`,
  `-H pa -k2` and `-H dj -k3` agree, which is why a small-budget sweep missed
  it. `logical_bscore.call_ensemble` (`demo_problems.py:609`) transcribes
  `bscores.cc:85` faithfully, so the locus is likely `ensemble.py` or the
  merging port.
- **`-Hrecall -q 0.5 --score-weight=<col>` raises where classic returns a
  score.** `discriminating_bscore.py:511` does a Python division by zero;
  `discriminating_bscore.cc:358` does IEEE `x/0.0f` and gets `inf`. Classic
  exit 0 printing `1 true`, port exit 1 `ZeroDivisionError`.
- **`--gen-best-tree=1` yields a different tree on 3/3 fixtures.** Root cause:
  C++ `CTable` is a `std::map` (`table.h:550`) and the port's is a dict, so ties
  in `precision_bscore.cc:655`'s `multimap` break differently. Re-sorting
  reproduces classic exactly, so this is an ordering bug, not arithmetic.
- **`-Hselect --boost=1`**: classic exit 0 `0 or(0<($1) 0<($2))`, port exit 1.
- **`--ss-n-subsample-fitnesses=2 --ss-low-dev-pressure=0.5`**: classic exit 0,
  port exit 1.
- **`behave_cscore(ss_bscore(...)).get_cscore(tr)` is unconstructible in
  practice** — `AttributeError: 'ss_bscore' object has no attribute
  'sum_bscore'`. The port's `ctruth_table_bscore` extends `bscore_base` where
  `bscores.h:360` extends `bscore_ctable_base`, so `ss_bscore(ctruth_table_bscore(ct))`
  also fails on `get_ctable`. A structural divergence, not a missing function.
- **`--version` prints nothing** in the port.
- **`--help` exit code differs**: classic exits **1**, port exits **0**.
  `option_manager`'s `Command line:` log IS reproduced
  (`moses_exec.py:337`); only its `exit(1)` is not.
- **`float32` narrowing still missing in `counts()`** (`discriminating_bscore.py:277-284`):
  `2.66` where C++ gives `2.6600000858306885`. `count()` itself is now exact.

**Stale in-source claims found in `pymetta/**` and NOT fixed** (the defect class
CARRY-FORWARD §6 names as the worst kind, because a confident false claim stops
the next reader looking):

- `representation.py:103-114` still says `deme_expander` "never forwards
  `linear_contin` at all". It does, at `deme_expander.py:669` — CARRY-FORWARD
  §5.7 is fixed and this docstring was not updated with it.
- `feature_selection_params.py:169` still calls `iscorer_cache` unported; it is
  at `deme_optimize.py:76` and used at `:190`.
- `eda/initialization.py:39-50` describes a `NotImplementedError`-on-contin that
  lives in `neighborhood_sampling.py:183`; `initialization.py` raises nothing.
- `ss_bscore.py:365`'s raise is now dead code (§3(d)) and its message is false.
- `table_problems.py:510`'s raise says `subsampleTable` is "not ported anywhere
  in this project"; it is ported at `fs_main.py:183`.
- `iostream_combo.py:559`'s message says "no test in this port's scope needs
  scheme-format output" — true, but the flag is CLI-reachable and classic
  serves it, so the message understates the gap.
- Three `# DIVERGENCE` comments now rest on stale premises and could be lifted:
  `merging.py:396` (the code at `:435` already passes `deme.getID()`),
  `merging.py:469` and `metapopulation.py:900-907` (both cite a missing
  `get_demeID`, now at `types.py:685`).
- `fs_main.py:140` drops `table.cc:872`'s `TTable&` parameter on a premise that
  is now false, so the ttable desyncs under `--subsampling-ratio`.

#### Found by the 2026-09-09 audit (carried forward)

- **`particle_swarm.py:474`** misses classic's frozen `disc_best_it`.
  **RESOLVED since** — `particle_swarm.py:492` now freezes `best_personal[0]`
  and `-aps` is 12/12 byte-identical.
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

- **`docs/PARTIAL-AUDIT.md` is stale, and now doubly so.** It classifies 62
  PARTIAL rows; there are now **70**, and its verdicts are two audits behind
  (`instance_scorer.h` is PORTED, `bscore_ctable_base` is done, `logical_bscore`
  is ported, and `discretize_contin_bscore` — which it named as the survivor of
  its item 3 — is itself now ported at `bscores.py:922`). Not rewritten here
  because it is outside this task's ownership. **Recommend deleting it**: it has
  never once been correct at read time, and `MODULE-COVERAGE.md`'s rows now carry
  everything it was for.
- **`docs/CARRY-FORWARD.md` §1.2** still says of `bscore_ctable_base` "Nothing
  ports it yet". It is ported and the trap is handled. §5's bullet on the
  logging surface is also stale in the same direction.
- **Three `raise` sites carry no `C++:` breadcrumb** — `demo_problems.py:1030`,
  `problem.py:260`, `partial.py:178` — so `check_ported_rows_do_not_gate` cannot
  see them. A row above any of those could be promoted to PORTED while the module
  still gates, and the tool would stay green. This is the one mechanical hole
  left in the guard; closing it means adding breadcrumbs to those three symbols,
  which is a `pymetta/**` edit.
- **Section intros in `MODULE-COVERAGE.md` went stale for the SECOND consecutive
  pass.** Rewritten for fourteen sections on 2026-09-09, nine of them were wrong
  again by 2026-09-11 and were rewritten for twelve sections. `coverage_totals.py
  --check` still validates only the rows. The pattern is now established rather
  than suspected: **prose in that file should be assumed stale unless it carries
  a date later than the rows it describes.**
- **No fixture manifest records a compiler/stdlib version** (CARRY-FORWARD
  §5.1). S12's fixtures depend on libstdc++'s sort permutations, and the `-aun`
  fix adds glibc's `rand()` to the same exposure -- `digraph.py:103-235` now
  transliterates glibc TYPE_3 `rand()` and libstdc++'s `std::random_shuffle`, so
  the port reproduces a toolchain-specific ordering by design.
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
