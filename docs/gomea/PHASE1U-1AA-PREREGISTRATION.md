# Phases 1u, 1v, 1w, 1x and 1aa: pre-registration (queue segment A)

Written 2026-10-19, before any of these runs. Shared rules: `QUEUE.md`.

## Common to every phase

* **Run order:** strictly one at a time, after Phase 1t has logged "all done":
  1u → 1v → 1w → 1x → 1aa.
* **How:** `tools/gomea/queue_run.sh`. Within a phase the mux11 part runs first, then Con.
  16 workers.
* **Code:** the merge of the knob-building speed-ups and the seeding options (integration
  commit `f8bc41a`), merged after Phase 1t and before any of these runs.
  * Outputs are byte-identical to `ef88825` on all 8 identity cases; the suite passes.
  * CPU is about 0.34–0.53× on the identity cases.
  * The manifests' `commit` field is the HEAD after the merge, produced by
    `tools/gomea/gen_queue_manifests.py`.
  * No further code is merged until 1aa is done.
* **Base settings:**
  * GOMEA is LT-B, N = 16, cap 250, memo; `--gomea-contin copy` on Con.
  * HC is `-a hc-capped` with cap 250.
  * **Every arm** adds `--knob-probe-baseline reduced` (candidate default, `BACKLOG.md`
    item 15).
* **Seeds:** mux11 seeds 3001–3010, Con splits 1–10. Extension: 3011–3020 / 11–20, only for a
  problem with an "extend" primary contrast.
* **Metrics:**
  * **mux11:** CPU seconds to solve at `-m 40000` (unsolved counts as +∞), with evaluations
    to solve reported.
  * **Con:** test NMSE at 10,000 evaluations, with CPU per run reported.
* **Rule** (`tools/gomea/phase_report.py`):
  * Holm correction over each phase's primary family; two looks at α = 0.025.
  * **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
  * **Extend:** a primary contrast at stage 1 with Holm p < 0.25.
  * Otherwise a **tie**.
  * A pair where both runs are unsolved counts as a tie.
  * Contrasts that were not extended keep their stage-1 verdict.

## 1u: GP-GOMEA seeding on mux11 (backlog item 14)

* **Question:** does seeding cut CPU to solve on the Boolean problem, as it cut CPU on
  regression in Phase 1r?
* **Arms:** `HC`, `HC-S`, `GOMEA`, `GOMEA-S`.
* **Seeding:** `-S` arms run GP-GOMEA first with 20 % of the evaluations (height 3, top 5
  elites, the seeder's Boolean function set AND/OR/NOT). MOSES gets the rest and starts from
  the seeds plus the empty program.
  * **Charging:** GP-GOMEA's evaluations and CPU are charged to the run
    (`runner._add_seed_cost`).
  * **Seed solves:** a seed that already solves mux11 counts as solved at GP-GOMEA's cost.
* **Primary** (Holm over 2): `seed-hc` (HC-S vs HC) and `seed-gomea` (GOMEA-S vs GOMEA).
* **Secondary:** `seeded-gomea-vs-hc` (GOMEA-S vs HC-S).
* **Descriptive:** GOMEA vs HC.
* **Pilot caution:** one earlier seeded-HC pilot (seed 3001) needed about twice the
  evaluations.

## 1v: uniform-distance start at equal cost, end to end (item 11)

* **Question:** Phase 1i found the uniform start helps on frozen maj5, at about 6–8× the
  evaluations. Does it help inside full MOSES runs at the same budget?
* **Arms:** `GOMEA`, `GOMEA-UNI` (`--gomea-radii uniform`).
* **Problems:** mux11 and Con.
* **Primary** (Holm over 2): `uniform` per problem.

## 1w: top-k linkage subsets per generation (item 2.4)

* **Question:** if each generation mixes only the K = 4 most strongly linked subsets, a
  capped deme completes about 3 generations and relearns its linkage. Does that help?
* **Arms:**
  * `GOMEA`.
  * `TOPK-L`: LT-MI with the top 4 subsets by linkage. LT-MI is used because LT-B's
    first-generation ranking is arbitrary.
  * `TOPK-R`: LT-MI with 4 random subsets.
* **Problems:** mux11 and Con.
* **Primary** (Holm over 2): `topk`, TOPK-L vs GOMEA, per problem.
* **Secondary** (Holm over 2): `linkage-vs-random`, TOPK-L vs TOPK-R, per problem.
* **Extension runs:** only GOMEA and TOPK-L, because secondary contrasts are never extended
  (amended 2026-10-19, before any 1w run).

**Deviation (2026-10-19 20:25, during the 1w mux11 runs, before any analysis):**

* **The problem:**
  * TOPK runs on mux11 cost about 1.2 s of CPU per evaluation, about 5× GOMEA's, because the
    linkage model is relearned every generation.
  * After about 55 min of CPU they had reached only 2,500–2,900 of 40,000 evaluations,
    still 360–430 rows wrong.
  * One TOPK-L run ended on its own after 1,016 evaluations, unsolved.
  * Left alone, the remaining runs would take until the 36,000 s wall cap.
* **The rule:** the registered metric is CPU to solve (unsolved = +∞). Once every GOMEA run
  is done and every TOPK-L run has either finished or already spent more CPU than its paired
  GOMEA run needed to solve, every pair's sign is fixed.
* **What happens then:** `tools/gomea/phase1w_earlystop.py` records each pair's state in
  `earlystop.json` and stops the mux11 runner, which terminates its active runs. The phase
  continues with Con as registered.
* **The mux11 primary verdict:** computed from `earlystop.json` with the registered rule.
  The automatic stage-1 report's mux11 row is partial and is superseded.
* **Step 2 (20:45):** all 16 slots were held by slow TOPK runs, so GOMEA s3009 and TOPK-L
  s3003 had not started. `tools/gomea/phase1w_finish.sh` does the following:
  1. Waits for GOMEA s3003 to finish.
  2. Pauses `phase_run.sh` (so no Con run can overlap).
  3. Interrupts the mux11 runner. Interrupted TOPK runs keep their CPU traces as lower
     bounds.
  4. Runs only GOMEA s3009 and TOPK-L s3003, until the early-stop rule is met.
  5. Resumes the phase.
* **Step 3 (21:13):**
  * SIGINT did not stop the runner, because background jobs of a non-interactive shell
    ignore it. Its child runs were then terminated with SIGTERM, followed by the runner.
  * The runner had already written `error` summaries for some of the terminated runs. These
    count as unsolved (+∞), the same outcome the early-stop rule assigns.
  * The paused launcher left the runner a zombie, which stalled `phase1w_finish.sh`.
    `phase1w_finish2.sh` replaced it and runs the remaining two runs plus the early stop.
* **The mux11 secondary (TOPK-L vs TOPK-R):** cannot be decided this way. It is reported
  descriptively: solved counts, and best error at the stop.

## 1x: deme length by problem type

* **Question:** is the 250-evaluation cap right for both problem types? Phase 1m found long
  demes (4,000) bad for mux11 but neutral for regression.
* **Arms:** `GOMEA-C125`, `GOMEA` (cap 250), `GOMEA-C500`. Only the cap differs.
* **Problems:** mux11 and Con.
* **Primary** (Holm over 4): `cap125` and `cap500` vs cap 250, per problem.

## 1aa: `-B 3`, effort-3 reduction in knob building (low priority)

* **Question:** does a stronger knob-building reducer
  (`--reduct-knob-building-effort 3`, default 2) change cost or speed?
* **Arms:** `HC`, `HC-B3`, `GOMEA`, `GOMEA-B3`.
* **Problem:** mux11.
* **Primary** (Holm over 2): `b3-hc` and `b3-gomea`.

**1aa stopped (deviation, 2026-10-20 07:48, after 2 of 40 runs, before any analysis).**

* **The bug:** the first `-B 3` run crashed with an AssertionError in effort-3 Boolean
  reduction. The trace:
  1. `logical_rules.py:1681`;
  2. `same_complete_truth_table`;
  3. `builtin_to_bool` received a non-Boolean;
  4. reached from `build_knobs._disc_probe_reduced_baseline`, the `--knob-probe-baseline
     reduced` path that every arm carries.
* **Why the phase was stopped:** crashes count as unsolved, so they would make `-B 3` look
  worse because of a bug, not because of the search. The phase was stopped and its runs
  discarded.
* **Not yet known:** whether the crash also happens with the classic probe (`-B 3` alone).
* **Status:** 1aa is **blocked by this bug** (`BACKLOG.md` section 4). It can be rerun once
  the crash is understood, either with the classic probe or with a fix.

## Phase 1z (near-constant simplification): settled by pilot, not run

* **Pilot results:**
  * In tooling smoke tests (Con split 1, at TOL 0.1 and 0.5), `--simplify-exemplar-numeric`
    found 0 candidate subtrees in every exemplar. The run was byte-identical to plain GOMEA.
  * The option's development pilot had found it fires rarely and is slightly worse when it
    does.
* **Decision:** to save running time, as the user asked, 1z is not run. Its manifests stay in
  `configs/gomea/phase1z/` in case a different TOL or dataset is wanted.
* **Recorded result:** "no effect at the tested tolerances: the pass almost never fires on
  Con".
