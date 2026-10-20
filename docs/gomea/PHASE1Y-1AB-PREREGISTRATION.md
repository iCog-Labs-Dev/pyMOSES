# Phases 1y and 1ab: pre-registration (queue segment B)

Written 2026-10-20, before any of these runs. Shared rules: `QUEUE.md`. Common settings,
metric and decision rule as in `PHASE1U-1AA-PREREGISTRATION.md`.

* **Run order:** strictly one at a time, after Phase 1aa: 1y → 1ab.
* **Problem:** regression only, Con splits 1–10, test NMSE at 10,000 evaluations in total.
  Extension: splits 11–20, with only the arms in primary contrasts.
* **Code:** the merged HEAD, which already includes both options (`f8bc41a`, identity
  IDENTICAL on all 8 cases).

## Background

* **Phase 1r:** seeding MOSES with GP-GOMEA elites helped capped HC on Con and EnH, and
  GOMEA on EnH.
* **On Con it hurt GOMEA by median:** 42.3 against 35.2, worse in 14 of 20 splits, a
  registered tie.
* **Why that may happen:** every seeded run's final program descended from a seed, and MOSES
  never built on the empty program. Seeds may pull GOMEA into a worse basin.

## 1y: can seeding stop hurting GOMEA on Con? (backlog item 9 refinements)

* **Arms:**
  * `GOMEA` (unseeded).
  * `GOMEA-S`: the Phase 1r seeding (20 % share, height 3, top 5, `+ - *`).
  * `GOMEA-S-RES`: GOMEA-S, with every 3rd expansion taken from the empty program's lineage
    (`reserve_empty_every: 3`).
  * `GOMEA-S10`: GOMEA-S with a 10 % seeding share.
* **Primary** (Holm over 2): `reserve-vs-unseeded` (GOMEA-S-RES vs GOMEA) and
  `share10-vs-unseeded` (GOMEA-S10 vs GOMEA).
* **Secondary** (Holm over 2): each variant vs GOMEA-S.
* **Descriptive:**
  * GOMEA-S vs GOMEA, which repeats Phase 1r's contrast;
  * expansions by lineage;
  * final program size.
* **Reading:** a variant "fixes" seeding on Con if it is not worse than unseeded GOMEA and is
  better than GOMEA-S. Seeding's size and CPU advantages (Phase 1r) are reported alongside.

## 1ab: re-seed from GP-GOMEA when MOSES stagnates (item 9, later extension)

* **Arms:**
  * `GOMEA`.
  * `GOMEA-S`.
  * `GOMEA-S-RESEED`: GOMEA-S, re-seeding after 5 expansions without improvement, at most
    twice.
* **What a re-seed does:**
  * GP-GOMEA runs again with a new random seed, using 30 % of the evaluations left (a
    re-seed is skipped below 1,000).
  * Its elites are checked with MOSES's scorer and merged into the metapopulation.
  * The total budget stays 10,000; GP-GOMEA's evaluations count against it.
* **Design note:** the default minimum (the initial seeding budget, 2,000) with a 20 % share
  would skip every re-seed on a 10,000 budget. The tooling smoke test showed this. Hence 30 %
  and a minimum of 1,000, set before any run.
* **Primary:** `reseed-vs-seeded` (GOMEA-S-RESEED vs GOMEA-S).
* **Secondary:** `reseed-vs-unseeded`.
* **Descriptive:** GOMEA-S vs GOMEA; number of re-seeds per run; lineage of the final program.

## Limits known in advance

* **One dataset (Con).** It was chosen because it is where seeding hurt GOMEA. A positive
  result would need a second dataset.
* **Re-seeding design:** it uses a fresh GP-GOMEA run, not one started from MOSES's programs.
  The latter would need C++ changes.
