# Phase 1l pre-registration: does GOMEA tie hill climbing on mux11 at equal time after the speed-ups?

Written 2026-10-15, before any Phase 1l run.

**Requested by the user:** make GOMEA "at least tie with the HC" while keeping its regression
lead, starting with speed-ups that lower its cost per evaluation ("do the improvement to
speed things up for a reduced CPU cost"), carried out autonomously.

## Background

* **Phase 1j (equal wall-clock time, 3,600 s, 10 seeds):** hill climbing (HC) solved mux11 10/10
  and was faster on every seed (median 615 s against 2,498 s). GOMEA solved 7/10.
* **GOMEA needed fewer evaluations to solve** (median 4,354 against 7,953), but each one cost
  about 5.5× the CPU.

## Code changes since Phase 1j

Each change is port-only and result-identical: the identity harness
(`tools/gomea/identity_check.py`, 8 cases, HC and GOMEA, Boolean and regression) reports every
case identical to the pre-change baseline. The changes:

* **`87cec7c`:** bit-parallel Boolean truth tables (about 180× faster table building on
  mux11-sized trees; helps HC and GOMEA).
* **`2f49251`:** GOMEA packs each genotype once per evaluation, and computes linkage marginals
  in one pass.
* **`4d04599`:** reduced candidates are cached per representation (62–70 % hit rate on maj5;
  helps HC and GOMEA).
* **`4e9b9ed`:** new opt-in options, unused here.
* **Not merged:** a disc-probe memo got 0 hits.

Because results are identical, the same seeds reach the same programs at the same
evaluation counts as before; only CPU and wall time change.

## Design

* **Arms and setup:** as Phase 1j's mux11 arm (`configs/gomea/phase1l/mux11.json`): `HC` vs
  `LTB-C250-M`, `--max-time 3600 -m 1000000000`, 10 workers on an otherwise idle machine.
* **Seeds:** 3001–3010, extended to 3011–3020 if borderline.
* **Code:** `a42d5c8`. The same code is used for Phase 1m.

## Analysis (`tools/gomea/phase1j_report.py`, unchanged)

* **Metric:** wall time to a solution; an unsolved run counts as 3,600 s.
* **Test:** paired Wilcoxon, LT-B GOMEA vs HC, with two looks at α = 0.025.
  * **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
  * **Extend:** 0.025 ≤ p < 0.25 at stage 1.
  * Otherwise a **tie**.
* **H (the user's target):** GOMEA is not worse, i.e. "tie" or "better". Phase 1j's verdict
  was "worse".
* **Descriptive:**
  * CPU per evaluation for each arm, against Phase 1j;
  * the speed-up factor of each arm;
  * solved counts.
