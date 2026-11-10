# Server round 10 (T1.2): pre-registration (2026-11-10)

**Status:** a draft by the agent. Not approved by the user. Not implemented.

**Question:** roadmap task T1.2, the last task of stage 1. On splits that no earlier round used,
how does the new base compare with MOSES hill climbing and with plain GOMEA? The result fixes the
reference setting for stage 2. **Seeding is a separate control experiment** (user, 2026-11-10): the
main result uses only the unseeded base, so a weak seeding result cannot lower it.

## Rules

* As in rounds 1 to 9: on the server, one phase at a time, W from `resources.py`, state in
  `server-run10/`. Round 10 never overlaps with another round.
* **Budget: evaluations, no time limit (user, 2026-11-09).** Each run gets 20,000 evaluations
  (`-m 20000`), as in round 9. LM passes and GP-GOMEA seed evaluations count. CPU is measured and
  reported, not capped. A wall cap of 36,000 s per run is only a guard against a hung process.
* **Fresh splits 31–40** on all 10 datasets. Rounds 1 to 9 used splits 1–30. The split builder
  (`tools/gomea/sr_datasets.py`) makes seeds 1–30 today; seeds 31–40 use the same rule
  (`random.Random("<Abbr>-s<seed>")`). This is a change in `tools`, not in `pymoses`.
* **No extension:** the primary contrasts pool 100 pairs.
* **Statistics:** see "Contrasts". Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, evaluations used, LM calls and
  guard rejections, deme count, generations per deme, population size; for BASE-S also the seed
  cost and the lineage.
* **S0:** full pytest and the reproducibility check.
* **No change under `pymoses`.**

## Arms

| Arm | Flags |
|---|---|
| **BASE** | `-a gomea --gomea-fos lt-b --gomea-pop-size dynamic --hc-fraction-of-nn 0.05 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val --gomea-accept-complexity-ratio 0.125 --gomea-deme-store accepted` (as in round 9) |
| **BASE-S** (control) | BASE, started from GP-GOMEA elites: the seed block of 1bb (`"share": 0.2`, height 3, k = 5, functions `+_-_*`) |
| **NOCAP** (plain GOMEA) | `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1` (as in round 8) |
| **HC-LS** | `-a hc --knob-probe-baseline reduced --linear-scaling 1`, no cap (the same cap as GOMEA; as in round 8) |

* Datasets: Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac. Splits 31–40.
* HC-LS has no LM: `--const-opt` works only with `-a gomea` and `-a hc-gomea`. So BASE vs HC-LS
  measures GOMEA, LM-VAL and PEN125 together.

## Contrasts

* **Primary (2), the main experiment, pooled over 100 pairs (10 datasets × 10 splits), test
  NMSE:**
  1. BASE vs HC-LS;
  2. BASE vs NOCAP.
  * **Test:** as in round 8. Paired Wilcoxon signed-rank on r = log(NMSE_A / NMSE_B), two-sided,
    Holm across the 2, α = 0.025. A failed run gets the largest rank against it.
  * **Verdict:** "better" if Holm p < 0.025 and exp(median r) ≤ 0.9; "worse" if Holm p < 0.025
    and exp(median r) ≥ 1/0.9; otherwise "tie".
* **Control, seeding (3), pooled, the same test, its own Holm family:**
  3. BASE-S vs BASE;
  4. BASE-S vs HC-LS;
  5. BASE-S vs NOCAP.
  The control family never enters the Holm correction of the primary family, so the control arm
  does not change the main result.
* **Secondary, pooled (6), the same test, Holm across the 6:** program size and CPU, each for
  BASE vs HC-LS, BASE vs NOCAP and BASE-S vs BASE.
* **Per dataset, descriptive:** contrasts 1, 2 and 3 on each dataset (30), paired Wilcoxon on test
  NMSE, Holm across the 30. These give no verdicts (the lowest Holm p is above 0.025). They
  show where each arm wins or loses.

## Reference setting for stage 2 (adoption rule)

| Result of contrast 3 (BASE-S vs BASE) | Reference for stage 2 |
|---|---|
| "better" | BASE-S |
| "tie" or "worse" | BASE (no GP-GOMEA build needed) |

The headline result of T1.2 is contrasts 1 and 2, with BASE. They do not change the reference.
The user decides in all cases.

## Memory

Round 9 showed that N = 16 needs much memory at 20,000 evaluations: Dow N16 runs failed at 8 GB,
Tow N16 runs at 5 GB. NOCAP uses N = 16. A failed NOCAP run counts as a BASE win, so failures
bias contrast 2.

| Manifest | `mem_limit_mb` | W on the server (64 GB, reserve 6,553 MB) |
|---|---|---|
| Dow | 16,384 | 3 |
| Tow | 8,192 | 7 |
| All other datasets | 5,120 | 11 (the CPU limit) |

* The round 9 mechanism (`manifest_per_run_mb`) gives each manifest its own W.
* If a NOCAP run still fails on memory, the report lists it, and the findings give contrast 2 with
  and without the failed pairs (the second form is descriptive).

## Implementation (by a subagent, in a separate clone)

* `tools/gomea/sr_datasets.py` and `setup.sh`: splits 31–40. The splits 1–30 must stay
  byte-identical (the `splits.json` check).
* `configs/gomea/phase1bd/`: manifests (one each for Dow and Tow, one for the 8 others), spec with
  pooled contrasts, budget `{"evals": 20000}`, wall cap 36,000 s.
* `tools/server/plan10.json` (S0, 1bd), a smoke plan, tests, and the SERVER-RUN section.
* The pooled test of round 8 (`set_final.py`, `phase_report.py`): reuse it for the 2 + 3 + 6
  pooled contrasts, in three Holm families.

## Size and time

* 400 runs (4 arms × 10 datasets × 10 splits).
* CPU per run from round 9 at 20,000 evaluations: BASE and BASE-S 200–1,400 s; N = 16 arms
  300–5,500 s. HC-LS is the fastest arm.
* About 10–14 hours at W = 11, plus about 1.3 hours for S0. The Dow runs at W = 3 add about 1–2
  hours.

## Limits

* One evaluation budget. Equal evaluations are not equal CPU; CPU is reported for each arm.
* The fresh splits are new partitions of the same rows. They are new for selection, but the rows
  are the same as in earlier rounds.
* HC-LS has no LM (see "Arms").
* The decision to test BASE-S came from 1bb, where seeding was better on 1 of 4 datasets (the user
  chose it against the rule, 2026-11-10). It is a control arm for this reason.

## Open points for the user

1. **Arms:** BASE, NOCAP and HC-LS in the main experiment, and BASE-S as the seeding control, as
   above?
2. **Splits:** fresh splits 31–40 (a tools change)?
3. **Memory:** Dow 16,384 MB (W = 3) and Tow 8,192 MB (W = 7)?
4. **Adoption rule:** the reference is BASE-S only if contrast 3 gives "better"?
