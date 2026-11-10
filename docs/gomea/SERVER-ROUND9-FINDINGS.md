# Server round 9: findings (2026-11-10)

**Pre-registration:** `SERVER-ROUND9-PREREGISTRATION.md`. **Results:** bundle
`bundle-final-20261110T043237Z.tar.gz` (local copy: `../server-results-round9/`). **Code:**
`efcea29`. **Decisions:** `APPROVALS.md`, 2026-11-10.

## Summary

| Phase | Question | Rule result | User decision |
|---|---|---|---|
| 1ba | Dynamic population (BASE) against N = 16 on 8 other datasets | Dynamic stays: BASE worse on 1 of 8 (Air); the rule needs 3 | Keep the dynamic population |
| 1bb | GP-GOMEA seeding (BASE-S) against BASE | Not adopted: better on 1 of 4 (Yac); the rule needs 2 | BASE-S is an arm of T1.2 |
| 1bc | Cap of 250 evaluations per deme with LM-VAL | Not adopted: better on 1 of 6 (Tow); the rule needs 2 | Leave it out |

## Run health

* **Steps:** S0 1.33 h, 1ba 6.27 h, 1bb 1.03 h, 1bc 8.12 h. W = 11; W = 7 for the two Dow
  manifests (`manifest_workers` in `state.json`).
* **S0:** reproducibility 4 of 4 matched. pytest: 9,251 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`). The same test failed in rounds 6, 7 and 8:
  the server's toolchain (Ubuntu g++ 13.3) differs from the toolchain that the fixtures record. It
  is not a code fault.
* **Failed runs:** 6 of 500, all N = 16 arms, all "lost sys.stderr" at the memory limit:

| Run | Peak RSS | Limit |
|---|---|---|
| Dow BASE-N16, splits 5, 6, 7, 8 | 8.04 GB | 8,192 MB |
| Tow BASE-N16, split 9 (1ba) | 5.09 GB | 5,120 MB |
| Tow N16, split 18 (1bc) | 5.02 GB | 5,120 MB |

* **Memory of the other runs:** every dynamic-population run stayed below 2.5 GB. Bos N16 reached
  3.5 GB, so it would have failed at the old 3,072 MB limit. The user chose not to add a backlog
  item for the cache size (2026-11-10).
* **Consistency:** BASE and N16 on Bos and EnC, splits 1–10, ran in both 1ba and 1bc with the same
  argv. Their medians are identical in the two phases.
* **Extensions:** only 1bb Yac (splits 11–20). No 1ba primary Holm p fell in [0.025, 0.25). 1bc
  had no extension by design.

Test NMSE in the tables is × 100 and is the median over the splits.

## 1ba. Dynamic population against N = 16

| Dataset | BASE | BASE-N16 | Pairs (BASE lower) | Holm p | Verdict | Size BASE / N16 | CPU s BASE / N16 |
|---|---|---|---|---|---|---|---|
| Air | 39.1 | 32.7 | 0/10 | 0.016 | **BASE worse** | 77 / 115 | 651 / 1,272 |
| Bos | 19.6 | 19.2 | 2/8 | 1.0 | tie | 64 / 93 | 348 / 948 |
| Dow | 39.8 | 35.1 | 4/6 | 1.0 | tie | 26 / 122 | 311 / 1,978 |
| EnC | 11.5 | 10.8 | 4/6 | 1.0 | tie | 45 / 74 | 337 / 869 |
| Tow | 18.8 | 15.7 | 1/9 | 0.59 | tie | 26 / 164 | 928 / 5,535 |
| WiR | 66.0 | 65.2 | 3/7 | 1.0 | tie | 54 / 131 | 522 / 1,504 |
| WiW | 72.1 | 71.6 | 4/6 | 1.0 | tie | 76 / 140 | 1,408 / 4,169 |
| Yac | 0.8 | 0.6 | 2/8 | 1.0 | tie | 34 / 56 | 257 / 341 |

* **Rule:** the dynamic population stays the base (BASE worse on 1 of 8; the rule needs 3).
* **The trade-off (D1):** the BASE median is higher on all 8 datasets: by 1–2 % on WiR, WiW and
  Bos, 6 % on EnC, 13–20 % on Air, Dow and Tow, and 0.8 against 0.6 on Yac (small in absolute
  terms). BASE programs are 30–85 % smaller, and BASE uses 1.3–6×
  less CPU.
* **Dow:** the 4 failed N16 runs count as infinitely bad, so they count as 4 BASE wins. N16 won
  all 6 pairs without a failure. Read the Dow tie with this in mind.
* **Secondary (size, gap, CPU; Holm over 24):** no verdict, as expected with 10 pairs. Size: BASE
  smaller in 8–10 of 10 pairs on all 8 datasets. CPU: BASE lower in 10 of 10 pairs on 7 datasets.
  Gap: no clear direction.

## 1bb. Seeding with the new base

| Dataset | BASE | BASE-S | Pairs (S lower) | Holm p | Verdict | Size BASE / S | CPU s BASE / S |
|---|---|---|---|---|---|---|---|
| Con | 36.1 | 30.9 | 8/2 | 0.25 | tie | 47 / 86 | 431 / 508 |
| EnH | 8.3 | 8.0 | 6/4 | 0.25 | tie | 44 / 51 | 345 / 369 |
| Air | 39.1 | 36.5 | 8/2 | 0.25 | tie | 77 / 84 | 652 / 568 |
| Yac (20 pairs) | 0.8 | 0.7 | 18/2 | 0.001 | **BASE-S better** | 34 / 34 | 245 / 185 |

* **Rule:** seeding does not join T1.2 (better on 1 of 4; the rule needs 2).
* **Pattern:** the BASE-S medians are lower on all 4 datasets, at about the same CPU (GP-GOMEA's CPU
  included). The Con loss of 1y and 1ab did not come back: Con is now 8 of 10 for BASE-S.
* **User decision:** BASE-S is an arm of T1.2.

## 1bc. Cap of 250 evaluations per deme

| Dataset | BASE | BASE-C250 | N16 | N16-C250 | Pairs (C250 lower) | Holm p | Verdict | Size BASE / C250 | CPU s BASE / C250 |
|---|---|---|---|---|---|---|---|---|---|
| Con | 26.9 | 19.9 | 23.0 | 19.2 | 8/2 | 0.039 | tie | 48 / 87 | 411 / 729 |
| Tow | 19.0 | 12.4 | 14.8 | 11.1 | 10/0 | 0.012 | **C250 better** | 31 / 100 | 936 / 3,608 |
| Yac | 0.8 | 0.6 | 0.7 | 0.7 | 9/1 | 0.11 | tie | 35 / 54 | 268 / 285 |
| WiR | 63.9 | 64.5 | 64.9 | 63.0 | 5/5 | 0.77 | tie | 68 / 90 | 558 / 1,260 |
| Bos | 19.6 | 16.0 | 19.2 | 18.8 | 8/2 | 0.75 | tie | 64 / 108 | 362 / 707 |
| EnC | 11.5 | 9.6 | 10.8 | 9.5 | 9/1 | 0.029 | tie | 45 / 72 | 344 / 570 |

* **Rule:** the cap does not join the base (better on 1 of 6; the rule needs 2).
* **Pattern:** the BASE-C250 median is lower on 5 of 6 datasets; two Holm p (0.029, 0.039) are just
  above 0.025. The programs are 30–220 % larger, and the CPU is 1.1–3.9× higher.
* **Secondary family A (N16-C250 vs N16, own Holm family of 6):** no verdict. Medians lower on 5 of
  6 (Tow 9/10, p 0.059).
* **Secondary family B (Holm over 12):** size: C250 larger, no verdict (smallest p 0.035). LM calls:
  C250 more on all 6 datasets, 10 of 10 pairs (verdict "A worse" means more calls), as the
  hypothesis said.
* **User decision:** leave the cap out.

## Limits

* The splits 1–20 are selection data. The claims hold for these splits only.
* One evaluation budget (20,000). Equal evaluations are not equal CPU: N16 and C250 use more CPU.
* The N16 memory failures make the Dow comparison of 1ba weaker.
