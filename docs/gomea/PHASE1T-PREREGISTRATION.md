# Phase 1t pre-registration: does the same penalty help capped hill climbing?

Written 2026-10-19, before any Phase 1t run.

**Requested by the user:** run the hill-climbing counterpart of the penalty test as its own
experiment, quick and conclusive.

## Question

If capped hill climbing (HC) ranks a deme's neighbours with the same strong complexity
penalty PEN gives GOMEA, does it solve mux11 with less CPU?

* **Why ask:** if it does, PEN's benefit comes from keeping exemplars small, not from
  anything specific to GOMEA.
* **Default HC:** it already uses MOSES's composite score. On mux11 that is a penalty of
  0.29 per complexity unit, negligible against scores in wrong rows. PEN with R = 0.125 is
  8 wrong rows per unit.

## Code

Commit `ebd77c0` adds the option for HC: `-a hc-capped` with `--gomea-accept-complexity-ratio R`.

* **What changes:** the climb ranks new instances, and tests improvement, by
  `raw − complexity / R`, the key GOMEA's PEN uses.
* **What does not change:** the scores stored in the deme, MOSES's merge and exemplar
  selection.
* **Default:** with R = 0 the code path is the ported one. The identity harness is
  byte-identical on all 8 cases, and there are unit tests in `tests/gomea/test_capped_hc.py`.
* This is an extension option, not a C++ divergence. It is disclosed in `APPROVALS.md` as
  agent-built.

## Design

* **Arms:**
  * `HC-C250`: `-a hc-capped --gomea-max-evals-per-deme 250`.
  * `HC-PEN`: HC-C250 + `--gomea-accept-complexity-ratio 0.125`.
* **Problem and budget:** mux11, `-m 40000`.
* **Seeds:** 3021–3030, 10 pairs and 20 runs, with 8 workers. It runs alongside Phase 1s.
* **Metric:** CPU seconds to solve (`cpu_to_target`); unsolved counts as +∞. As in Phase 1s,
  CPU time is used because the machine is shared.

**Analysis** (`tools/gomea/phase1t_report.py`):

* **Primary:** one paired Wilcoxon test, HC-PEN vs HC-C250, at α = 0.025.
  * **Better / worse:** p < 0.025 and a median at least 10 % lower / higher.
  * Otherwise a **tie**.
  * Single look, no extension.
* **Descriptive:** solved count, evaluations to solve, CPU per evaluation, demes per run,
  best complexity.

**Why 10 pairs is enough here:**

* Capped HC's runs on mux11 are cheap (median about 1,100 s of CPU), so the phase is quick.
* A penalty strong enough to matter changes HC's trajectory on every seed. A real effect
  should show in most pairs, and 9 of 10 can already give p ≈ 0.004.
* If the effect is too small to show in 10 pairs, it is too small to matter next to
  GOMEA's cost gap, so a tie is a usable answer.

**Pilot, before registration (tooling smoke tests, 2026-10-19):**

* **The result:** HC-PEN and HC-C250 produced identical scorer traces on seeds 3021–3023 at
  `-m 1500`, and on seed 3022 at `-m 8000`. Both solved 3022 at exactly 6,121 evaluations.
* **The check:** an in-process check showed the hook is active. It was called 600 times in
  a `-m 600` run and differed from the default key 598 times. It just never changed which
  neighbour won.
* **Likely reason:**
  * HC moves to the best neighbour, and MOSES's default penalty already breaks ties between
    equal raw scores towards the simpler program.
  * At R = 0.125, a penalty of 8 rows per unit rarely outweighs a mux11 improvement step of
    8–32 rows.
  * GOMEA, in contrast, accepts any trial that is not worse, so the penalty changes its
    decisions on neutral moves.
* **Expected outcome:** a tie with near-identical runs. The phase is kept because it is
  cheap, and it documents over 10 seeds whether the penalty acts only through GOMEA's
  acceptance.
* **Descriptive addition:** the number of seeds whose runs are identical in evaluations to
  solve.

**Reading, with Phase 1s:**

| Phase 1s (GOMEA) | Phase 1t (HC) | Conclusion |
|---|---|---|
| better | better | The penalty works for any optimizer: smaller exemplars are the lever. |
| better | tie or worse | The gain is specific to GOMEA's acceptance. |
| tie | better | The penalty helps HC but not GOMEA; GOMEA's cost lies elsewhere. |
| tie | tie | The penalty is a CPU saver at equal evaluations only, not a time-to-solve win. |
