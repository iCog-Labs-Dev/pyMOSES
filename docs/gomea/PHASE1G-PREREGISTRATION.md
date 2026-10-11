# Phase 1g pre-registration: short demes or linkage? (quick, exploratory)

Written 2026-10-11, before any Phase 1g run.

**Requested by the user:** "let's just check whether the performance gain we observed is
actually from the smaller deme or from the usage of a linkage tree in the first place with
as small of a benchmark as possible ... to make the run finish with in the next hour".

## Question

Inside pyMoses, capped GOMEA beat hill climbing (Phases 1e and 1f), and on regression the
random-tree FOS did as well as the linkage tree. Is the gain from **short demes** (more
representation rebuilds per run) or from **GOMEA's linkage-driven mixing**?

## Design (exploratory; sized for about an hour)

* **Code:** `-a hc-capped` (new, opt-in) is MOSES's own hill climbing with each deme's budget
  capped at `--gomea-max-evals-per-deme`. With no cap it is byte-identical to `-a hc`.
* **Arms** (all `tools/gomea/moses_sr_runner.py`, the Phase 1f regression setup; manifest
  `configs/gomea/phase1g/quick.json`):
  * `HC`: stock hill climbing;
  * `HC-C250`: hill climbing capped at 250 evaluations per deme;
  * `LTB-C250`: the Phase 1f GOMEA arm (lt-b, cap 250, memo, `--gomea-contin copy`);
  * `UNI-C250`: the same with `--gomea-fos uni`, i.e. no linkage at all.
* **Problems:** Con and EnH, the two fastest Phase 1f datasets on which GOMEA was *better*
  than HC. The Boolean problem where GOMEA won (mux11) takes about 45 min and 0.8 GB per
  GOMEA run, too long for the hour.
* **Budget:** 2,000 evaluations per run (Phase 1f used 10,000), splits 1–6 (paired), 10
  workers, a 1,536 MB cap per run.

## Analysis (exploratory; no confirmatory claim)

Per dataset, test NMSE, paired by split: median per arm, how many of the 6 pairs one arm is
lower in, and the Wilcoxon p (the smallest possible with 6 pairs is 0.031). Contrasts:

1. `HC-C250` vs `HC`: does capping alone help hill climbing?
2. `LTB-C250` vs `HC-C250`: what GOMEA adds over short-deme hill climbing.
3. `LTB-C250` vs `UNI-C250`: what linkage adds within GOMEA.

**Reading.** An arm "leads" a contrast on a dataset if it is lower in at least 5 of 6 pairs.
* "Short demes explain the gain" if `HC-C250` leads `HC` on both datasets and `LTB-C250`
  does not lead `HC-C250` on either.
* "GOMEA's mixing matters" if `LTB-C250` leads `HC-C250` on both.
* "Linkage matters" if `LTB-C250` also leads `UNI-C250` on both.
* Anything else is "inconclusive at this size".

**Caveats fixed in advance:** 2,000 evaluations is a fifth of Phase 1f's budget, so the
GOMEA-vs-HC gap may differ from Phase 1f's; 6 pairs on 2 datasets can only indicate.
