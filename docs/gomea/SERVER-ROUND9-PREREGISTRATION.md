# Server round 9: pre-registration (2026-11-09)

**Status:** the design is approved by the user (see `APPROVALS.md`, 2026-11-09). Implemented in be758a9 and a2ba7dd. Not run yet.
Agent-written, not user-approved in detail: the evaluation budget of 20,000, the datasets,
the splits and the adoption rules. Changed on 2026-11-09 after the Dow memory check (user-approved,
see `APPROVALS.md`): the memory limits in "Before the round: Dow memory" and the Holm families of
1bc. Implemented in 13c9f63, 8b519d3, 16ce6d6 and e97d27f. Dow does not get a plan step of its
own: the 1ba step gives the two Dow manifests their own W (plan key `manifest_per_run_mb`, 8,192),
so the 1ba report keeps all 8 datasets in one Holm family. Not run yet.

**Question:** three stage 1 tasks of `ROADMAP.md`, all with the new base that the user chose on
2026-11-09:

1. **1ba:** does the dynamic population hold up on other datasets? (`BACKLOG.md` 9.2, 9.3)
2. **1bb:** does GP-GOMEA seeding still help with the new base? (`BACKLOG.md` 7.5, roadmap T1.1)
3. **1bc:** does LM-VAL gain from the cap of 250 (more demes, so more LM calls)? (`BACKLOG.md` 9.1)

## Rules

* As in rounds 1 to 8: on the server, one phase at a time, W from `resources.py`, state in
  `server-run9/`. Round 9 never overlaps with another round.
* **Budget: evaluations, no time limit (user, 2026-11-09).** Each run gets 20,000 evaluations
  (`-m 20000`) and no `--max-time`. LM's passes over the training rows count as evaluations
  (`USAGE.md`), and so do GP-GOMEA's seed evaluations in 1bb. CPU time is measured and reported,
  not capped. A wall cap of 36,000 s per run stays only as a guard against a hung process, as in
  the earlier evaluation-budget phases.
* **Why 20,000:** in 1,000 s, the N = 16 LM-VAL arms of rounds 7 and 8 used a median of 5,000–
  47,000 evaluations (most 12,000–17,000), and the dynamic arms 16,000–42,000 (most 25,000–
  35,000). 20,000 is between the two groups.
* **Statistics:** paired Wilcoxon signed-rank, two-sided, Holm across the primary contrasts of each
  phase, α = 0.025. "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1
  extends that dataset (splits 11–20) where the phase has an extension. Secondary contrasts never
  extend. Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, evaluations used, LM calls and
  guard rejections, deme count, generations per deme, population size; for 1bb also the seed cost
  and the lineage.
* **S0:** full pytest and the reproducibility check.
* **Splits:** no fresh splits are left (round 8 used 21–30). The phases reuse splits 1–20, so they
  are selection data, not a confirmation.

## The new base (all arms start from it)

* **Flags:** `-a gomea --gomea-fos lt-b --gomea-pop-size dynamic --hc-fraction-of-nn 0.05
  --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced
  --linear-scaling 1 --const-opt lm --const-opt-guard val --gomea-accept-complexity-ratio 0.125
  --gomea-deme-store accepted`, no evaluation cap per deme.
* **Budget:** 20,000 evaluations, no time limit.
* **Name:** BASE. Its N = 16 form is BASE-N16: `--gomea-pop-size 16` in place of the dynamic flags.

## 1ba. The new base against N = 16 on other datasets

* **Arms:** BASE, BASE-N16.
* **Datasets:** Air, Bos, Dow, EnC, Tow, WiR, WiW, Yac (the 8 datasets that 1ay did not use).
  Splits 1–10, extension 11–20.
* **Primary (8):** test NMSE, BASE vs BASE-N16, on each dataset.
* **Secondary (24):** program size, train–test gap and CPU, BASE vs BASE-N16, on each dataset.
* **Adoption rule:** the dynamic population stays the base unless BASE is worse on at least 3 of 8
  datasets. In that case the report recommends N = 16 for regression, and the user decides.
* **Runs:** 160 (+ up to 160).

## 1bb. GP-GOMEA seeding with the new base

* **Arms:**
  * **BASE.**
  * **BASE-S:** BASE, started from GP-GOMEA elites. The seed block of 1r and 1ab: `"share": 0.2`
    (4,000 of the 20,000 evaluations), height 3, k = 5 elites, functions `+_-_*`, with the empty
    program kept. MOSES gets the rest of the evaluations (the existing runner code).
* **Datasets:** Con, EnH (the datasets of the earlier seeding phases), Air, Yac. Splits 1–10,
  extension 11–20.
* **Primary (4):** test NMSE, BASE-S vs BASE, on each dataset.
* **Secondary (8):** program size and CPU (GP-GOMEA's CPU included), BASE-S vs BASE, on each
  dataset.
* **Adoption rule:** seeding joins the combined run (roadmap T1.2) if BASE-S is better on at least
  2 of 4 datasets and worse on none. If not, T1.2 runs without seeding, and the seeded arm stays a
  pair there only if the user asks.
* **Runs:** 80 (+ up to 80).

## 1bc. LM-VAL with the cap of 250

* **Arms:**
  * **BASE.**
  * **BASE-C250:** BASE + `--gomea-max-evals-per-deme 250`.
  * **N16:** BASE-N16.
  * **N16-C250:** BASE-N16 + `--gomea-max-evals-per-deme 250`.
* **Why four arms:** with the dynamic population, a deme with 250 evaluations gets N = 100 and has
  room for about one GOM generation. The 1aw hint came from N = 16. Two N = 16 arms show the cap's
  effect at N = 16 on new data, apart from the population.
* **Datasets and splits:** data that 1aw did not use, so the hint is not tested on its own data:
  Con, Tow, Yac, WiR on splits 11–20; Bos and EnC on splits 1–10. **No extension:** it would reuse
  1aw's data.
* **Primary (6):** test NMSE, BASE-C250 vs BASE, on each dataset.
* **Secondary, family A (6, own Holm family):** test NMSE, N16-C250 vs N16, on each dataset (the
  1aw hint on new data). With 10 pairs the smallest p is 2/1024, so Holm over 6 gives at least
  0.012, and a verdict is possible. (The first draft had one family of 18; there the smallest Holm
  p was 0.035, so no verdict was possible.)
* **Secondary, family B (12):** program size and LM calls per run, BASE-C250 vs BASE, on each
  dataset. "Better" for LM calls means fewer; the hypothesis expects more, so the report reads
  this verdict as a direction only.
* **Adoption rule:** the cap of 250 joins the base for regression if BASE-C250 is better on at
  least 2 of 6 datasets and worse on none. If only N16-C250 vs N16 is better, the report says so,
  and the user decides.
* **Runs:** 240.

## Before the round: Dow memory

In 1ax, 3 of 30 Dow runs of the GOMEA arms ended at a peak of 3.07–3.11 GB (limit 3,072 MB), after
about 12,000 evaluations. With 20,000 evaluations the use can be higher. Dow is in 1ba.

1. Find where the memory goes, locally, on a failed split (FINAL, split 22).
2. If the cause is a bug or an unbounded store, fix it as opt-in or prove that the fix does not
   change results (identity test). Then the limit stays 3,072 MB.
3. If the use is real, the Dow manifest gets a higher `mem_limit_mb`, and W follows from
   `resources.py` with that limit.

The user approves the outcome of step 1 before the round starts.

**Outcome (2026-11-09, user-approved):**
* Cause: the reduced-candidate cache (`representation.py:59`, 10,000 entries per deme). Each key
  holds the unreduced tree, so an entry is 185–520 KB in Dow demes of 600–1,800 knobs. A pure
  cache: with it off, the output of one run was identical. On Dow it had 0 hits in the search.
* Decision: step 3, the cache stays (the same code for GOMEA and HC). Limits:
  * Dow (1ba stage 1 and extension): `mem_limit_mb` 8,192, in its own plan step with
    `PER_RUN_MB` 8192, so W = floor((65,536 − 6,553) / 8,192) = 7.
  * Every other round 9 manifest: `mem_limit_mb` 5,120. Bos reached 2,609 MB at about 12,000
    evaluations in 1ax. W = floor(58,983 / 5,120) = 11, so W stays 11 (the CPU limit).
* Limit: the Dow need (about 7 GB) is an extrapolation from one split.

## Implementation (by a subagent, in a separate clone)

* **Configs:** `configs/gomea/phase1ba/`, `phase1bb/`, `phase1bc/`, with spec files, budget
  `{"evals": 20000}`, wall cap 36,000 s. A Dow manifest of its own if step 3 applies.
* **Plan:** `tools/server/plan9.json` (S0, 1ba, 1bb, 1bc), a smoke plan, and tests. No choice step:
  the three phases are independent. No chain to another round.
* **No change under `pymoses`,** unless the Dow check finds a bug. Seeding with an evaluation budget
  exists (`moses_sr_runner.py`).

## Size and time

* Stage 1: 480 runs. At the CPU per evaluation of rounds 7 and 8 (about 0.02–0.15 s), a run of
  20,000 evaluations takes about 500–3,000 s; Tow and WiW are the slowest. About 16 hours at
  W = 11, plus 1.3 hours for S0.
* With all extensions: up to 720 runs, about 24 hours.

## Limits

* One evaluation budget. The arms use different CPU per evaluation, so equal evaluations are not
  equal CPU. The CPU is reported for each arm.
* The splits are selection data. A claim about the final setting needs fresh data, which we do not
  have.
