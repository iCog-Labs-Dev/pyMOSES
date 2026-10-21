# Experiment queue: finishing the backlog (from 2026-10-19)

**Requested by the user:** finish every backlog item so the next quarter's build can be chosen.
Use the minimum running time, keep findings consistent and comparable with earlier phases,
and spend more running time only on inconclusive results.

## Rules for every phase below

1. **Seeds:** mux11 seeds 3001–3010, Con splits 1–10. These are the seeds and splits of
   Phases 1n–1r, so medians compare directly. Extension: 3011–3020 / 11–20, only for a
   primary contrast at stage 1 with 0.025 ≤ Holm p < 0.25.
2. **Problems:** mux11 (Boolean) and Con (regression), unless the question concerns only one
   type. EnH is added only where a phase's question needs a second regression problem.
3. **Metrics,** chosen so they are robust to a shared machine:
   * **mux11:** CPU seconds to solve (`cpu_to_target`) at `-m 40000`; unsolved counts as +∞.
     Evaluations to solve is reported alongside.
   * **Con:** test NMSE at `-m 10000`, with CPU per run reported.
   * **No wall-clock (equal-time) phases.** CPU to solve is the time measure.
4. **Base settings:** GOMEA is `LTB-C250-M` (LT-B, N = 16, cap 250, memo; `--gomea-contin
   copy` on Con). HC is capped at the same 250, per the user's rule.
   * From Phase 1u on, **every arm** adds `--knob-probe-baseline reduced` (candidate default,
     `BACKLOG.md` item 15). It has no effect on regression knobs.
   * Every phase reruns its own baseline arms, so no cross-phase pooling is needed.
5. **Rule:** two looks at α = 0.025 with Holm correction over each phase's primary family.
   "Better" also needs a 10 % lower median. Contrasts that were not extended keep their
   stage-1 verdict.
6. **No overlap, at any cost** (user, 2026-10-19).
   * Exactly one experiment runs at a time. Within a phase, the Boolean and regression parts
     also run one after the other.
   * Each part uses 16 workers.
   * `tools/gomea/queue_run.sh` starts the next phase only when no other experiment runner
     is alive.
   * Code work in the clones (`../pyMoses-dev`-style clones) runs under `nice -n 19`, with at
     most 2 jobs while an experiment is running.
   * Code is merged only between phases, so a running phase's code check never breaks.
7. **Process:** each phase gets a short pre-registration before it runs. A subagent builds and
   smoke-tests its tooling. The phase gets a fact-checked findings file and a row in
   `FINDINGS-LOG.md`.

## Order

| # | Phase | Backlog item | Question | Arms | Problems | Runs (stage 1) | Code needed |
|---|---|---|---|---|---|---|---|
| 0 | **1s**, then **1t** | 13 | Does the acceptance penalty cut CPU to solve, for GOMEA (1s) and for HC (1t)? | GOMEA / PEN; HC / HC-PEN | mux11 | 20 + 20 | done |
| 1 | **1u** | 14 | Does GP-GOMEA seeding close the Boolean gap? | HC, HC-S, GOMEA, GOMEA-S | mux11 | 40 | none (seed block exists) |
| 1 | **1v** | 11 | Uniform-distance start at equal cost, end to end | GOMEA, GOMEA-UNIFORM | mux11, Con | 40 | none |
| 2 | **1w** | 2.4 | Top-k linkage subsets per generation (lt-mi) | GOMEA, TOPK | mux11, Con | 40 | none |
| 2 | **1x** | — | Deme length by problem type | GOMEA at caps 125 / 250 / 500 | mux11, Con | 60 | none |
| 3 | **1y** | 9 (refinement) | Can seeding stop hurting GOMEA on Con? Reserve expansions for the empty program; a smaller seeding share | GOMEA, GOMEA-S, GOMEA-S-RES, GOMEA-S-10% | Con | 40 | `--reserve-empty` option |
| 3 | **1z** | 10 | Near-constant subtree simplification | GOMEA, GOMEA-NC | Con | 20 | none |
| 4 | **1aa** | — | `-B 3` (effort-3 knob-building reduct) | HC, HC-B3, GOMEA, GOMEA-B3 | mux11 | 40 | none |
| 4 | **1ab** | 9 (later extension) | Re-seed from GP-GOMEA when MOSES stagnates | GOMEA-S, GOMEA-S-RESEED | Con | 20 | re-seed hook |
| — | engineering | 1.x | Cheaper knob building | — | identity harness + CPU per evaluation | — | yes, in `../pyMoses-dev` |

**Why this order:**

* Items that inform the quarter's build decision most come first: penalty (0), seeding on
  Boolean (1), and the structural GOMEA options (1–2).
* Code-free phases come before code-dependent ones, so code is built while earlier phases run.
* Low-priority items (1z, 1aa) and the largest new code (1ab) come last.

**Estimated wall time,** run strictly one at a time on a quiet machine: about 1–2 h per
phase at stage 1, so roughly 14–20 h for the whole queue without extensions. A shared machine
slows wall time but not the CPU metrics.

**Run order:** 1s → 1t → 1u → 1v → 1w → 1x → 1y → 1z → 1aa → 1ab. The `#` column groups
phases by priority only; nothing in it runs at the same time.

## Incident: orphaned runs overlapped later phases (found 2026-10-20 09:25)

* **What happened:** when runners were stopped early, by terminating their child runs first
  (Phase 1w at 21:13 on 2026-10-19, Phase 1aa at 07:48), some runs were orphaned and kept
  computing.
  * 1 orphan (1w TOPK-R s3007) ran during 1w's Con part, 1x and 1y.
  * 6 orphans (1aa) ran during 1y.
  * All were killed at 09:25.
* **Effect on results:**
  * Con results (test NMSE at a fixed budget) are unaffected.
  * mux11 CPU-to-solve in 1x ran with one extra busy process among 16, so it is a little
    noisier.
  * Two orphans wrote late summaries into 1w; its findings are corrected and its verdict
    is unchanged.
* **Fix for future early stops:** terminate the runner process first (so it cannot launch
  queued runs), then its children. Afterwards, check for leftover `spawn_main` processes
  whose parent is init.

## Status

**Queue complete (2026-10-20 17:10).** Every queued phase has a findings file: 1s, 1t, 1u, 1v,
1w, 1x, 1y, 1aa (rerun on the effort-3 fix) and 1ab. 1z was settled by pilot.


* **Segment A** (merged code after `f8bc41a`, with the knob speed-ups and seeding options; queued with `tools/gomea/queue_run.sh`):
  1s → 1t → 1u → 1v → 1w → 1x → 1aa.
  * 1s started 2026-10-19 10:55.
  * Pre-registrations: `PHASE1S-`, `PHASE1T-` and `PHASE1U-1AA-PREREGISTRATION.md`.
* **1z:** settled by pilot and not run. The pass found 0 candidates on Con; see
  `PHASE1U-1AA-PREREGISTRATION.md`.
* **Merged early** (after 1t, before 1u): `../pyMoses-seedopts` and `../pyMoses-knobfast`,
  via `../pyMoses-int`. Identity check: IDENTICAL on all 8 cases; suite: 8602 passed.
* **Segment B:** 1y → 1ab (`PHASE1Y-1AB-PREREGISTRATION.md`), queued behind 1aa.

**CPU comparability:** 1s and 1t ran on the old code. Every phase from 1u on, segment B
included, runs on the same merged code.

## Round 2 (from 2026-10-21): phases 1ac–1ag

Protocol: `PHASE1AC-1AG-PREREGISTRATION.md`. The rules above apply unchanged.

* **1ac:** ran 10:09–11:30. Verdict: tie (`PHASE1AC-FINDINGS.md`).
* **Stop at 11:46 (user request: free the CPU):**
  * 1ad had been running for 16 minutes, with no run finished.
  * Stopped in this order: the queue, the phase launcher, the runner, then its 16 workers and
    their runs, by PID.
  * No orphans were left.
  * The partial output was moved to `../gomea-results/phase1ad-aborted-20261021` and is not
    used.
* **Restart:** 1ad from scratch, then 1ae, from 12:19 (`queue-1ad-1ae.log`).
* **Code for the later phases:**
  * The long-deme option and regression-runner telemetry (1af) are done in their clone.
  * The linear-scaling work (1ag) and the knob-probe default change (backlog 15) are
    committed in their clone, but their verification was cut off by the stop.
