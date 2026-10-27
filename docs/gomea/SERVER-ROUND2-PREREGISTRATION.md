# Server round 2: pre-registration (2026-10-27)

**Why:** the first server round (`SERVER-ROUND-PREREGISTRATION.md`) showed that 90–99 % of
GOMEA demes stop inside generation 0. The FOS has 400–2,000 subsets (hundreds of knobs) and a
deme may use 250 evaluations, so a deme never reaches the generation in which the linkage
tree is learned from a selected population. Thus no phase so far has tested learned linkage
in a setting where it can act. The user asked for a second round that tests (A) a cap on the
number of knobs in the FOS, as a stable core plus a few rotating "explore" knobs, and (B) the
linkage ablation (learned linkage tree against univariate and random-tree FOS) at that cap.
The user then asked for (C) a test of whether the fitness ranking in generation 0 helps,
against other ways to rank the knobs in generation 0.
The designs are the agent's, written at the user's request (see `APPROVALS.md`).

**What earlier phases predict:** at N = 16, learned linkage did not beat a random tree
(1g, 1h at N = 16–256, 1m). It did at N = 1,024 on a frozen problem (1h). Keeping the 4
strongest subsets per generation (1w) was worse on mux11 and Con, and linkage-ranked subsets
tied random ones. So the expected result of part B at N = 16 is a tie. The N = 64 arms test
the sample-size explanation. The knob cap differs from 1w: 1w kept 4 subsets of a FOS over
all knobs; the knob cap keeps the full FOS (2K − 2 subsets) over K knobs.

## Rules for the whole round

* As in round 1: everything runs on the server, one phase at a time, W from
  `tools/server/resources.py`.
* **Statistics** as in every phase since 1u: paired Wilcoxon signed-rank, Holm across a
  phase's primary contrasts, α = 0.025. "Better" also needs a 10 % lower median. Holm p in
  [0.025, 0.25) at stage 1 extends that problem (seeds 3011–3020 or splits 11–20). Secondary
  contrasts never extend. Failed runs count as infinitely bad.
* **Problems:** mux11 (seeds 3001–3010, `-m 40000`, metric CPU to solve, as 1ae), Con and EnH
  (splits 1–10, 10,000 evaluations, metric test NMSE, as 1ai). Also reported: size,
  train–test gap, CPU, and per deme the generations completed (D1 plus telemetry).
* **Base setting:** LT-B, N = 16, deme cap 250, memo, `--gomea-contin copy`,
  `--knob-probe-baseline reduced` (the round-1 GOMEA arm).
* **Order:** S0 (tests and the round-1 reproducibility check: with the knob cap off, the new
  code must reproduce round 1 exactly), 1ak-pilot, 1ak, 1al, 1am.

## The knob cap (new code, off by default)

* **Flags:** `--gomea-knob-cap K` (0 = off), `--gomea-knob-select {explore|static|random}`
  (default `explore`), `--gomea-knob-explore E` (default ⌈K/4⌉),
  `--gomea-knob-rank0 {contrast|entropy|random}` (default `contrast`; the score used in
  generation 0).
* **Candidate knobs:** the deme's search positions (knobs that vary in the initial
  population; constant ones are already excluded today).
* **Score in generation 0 (fitness contrast):** for each knob, |mean fitness of the
  individuals whose value differs from the exemplar's − mean fitness of the others|; 0 when
  all or none differ. Ties are broken at random. This is `--gomea-knob-rank0 contrast`.
* **Other generation-0 scores (1am):** `entropy`: the knob's entropy in the initial
  population (`H0`, already computed for LT-B; no fitness is used). `random`: a random
  order.
* **Score from generation 1 (`explore` only):** linkage strength, the knob's largest LT-B
  similarity to any other search position, from the current population. It is computed with
  LT-B in every arm, so the knob choice does not depend on the arm's FOS type.
* **Selection:**
  * `static`: the K best by fitness contrast in generation 0, fixed for the deme.
  * `random`: K random search positions in generation 0, fixed for the deme.
  * `explore`: each generation, a core of the K − E best (generation 0: fitness contrast;
    later: linkage strength) and E random explore knobs drawn from the rest.
* **FOS:** built by the arm's `--gomea-fos` on the selected knobs only (LT-B uses the rows of
  the initial-population entropies `H0` of those knobs, so a knob that enters later already
  has its baseline). Knobs outside the selection keep each individual's value; GOM does not
  copy them.
* **Random numbers:** a separate stream, so that with K = 0 every other stream and every
  output are unchanged.
* **Telemetry:** per generation the selected knobs, the core, and how many knobs entered the
  core; per deme the generations completed.

## 1ak-pilot. Choosing K

* Arm KC (`explore`, LT-B, N = 16, cap 250) with K ∈ {8, 16, 32}, 2 seeds or splits (3001–3002,
  splits 1–2) per problem: 18 runs.
* **Rule:** per problem, the largest K whose median `generations` per deme (over the demes of
  both runs) is at least 3 (so the tree is learned from a selected population at least twice).
  If no K qualifies, K = 8. A script writes K into every KC arm of 1ak and 1al and logs the
  pilot's numbers. The chosen K is reported.

## 1ak. Does a knob cap help? (part A)

* **Arms:** GOMEA (base), KC (`explore`), KC-STATIC, KC-RANDOM (`random`, static).
* **Primary (6):** KC vs GOMEA and KC-STATIC vs GOMEA, on each problem.
* **Secondary:** KC vs KC-STATIC (does rotation avoid a lock-in?), KC-STATIC vs KC-RANDOM
  (does the ranking matter, or only the smaller FOS?).
* **Also reported:** generations per deme, knobs entering the core per deme.
* **Runs:** 120 (+ up to 90).

## 1al. Does learned linkage help when demes run several generations? (part B)

* **Arms**, all with the knob cap (`explore`, the pilot's K):
  * KC: LT-B (the same configuration as 1ak's KC; rerun inside this phase).
  * KC-UNI: `--gomea-fos uni`.
  * KC-RT: `--gomea-fos rt`.
  * KC-N64: LT-B, N = 64, deme cap 1,000 (the cap grows with N, so generations stay
    comparable).
  * KC-RT-N64: random tree, N = 64, deme cap 1,000.
* **Primary (6):** KC vs KC-UNI and KC vs KC-RT, on each problem.
* **Secondary:** KC-N64 vs KC-RT-N64 (does a larger sample make learned linkage pay?), KC-N64
  vs KC.
* **Reading:** the evaluation budget is equal in all arms, so the univariate FOS (K subsets
  against 2K − 2) gets about twice the generations; generations are reported per arm.
* **Runs:** 150 (+ up to 90).

## 1am. Does the fitness ranking in generation 0 help? (part C)

* **Arms**, all `explore` with the pilot's K, LT-B, N = 16, cap 250:
  * KC: `--gomea-knob-rank0 contrast` (the same configuration as 1ak's KC; rerun inside
    this phase).
  * KC-R0-ENT: `--gomea-knob-rank0 entropy`.
  * KC-R0-RAND: `--gomea-knob-rank0 random`.
* **Primary (6):** KC vs KC-R0-ENT and KC vs KC-R0-RAND, on each problem.
* **Secondary:** KC-R0-ENT vs KC-R0-RAND.
* **Reading:** from generation 1 all three arms rank by linkage strength, so they differ
  only in the generation-0 core. A tie means that the fitness ranking gives no benefit
  over a ranking that uses no fitness. 1ak's KC-STATIC vs KC-RANDOM (secondary) is the
  same question with a fixed selection.
* **Runs:** 90 (+ up to 90).

## Size and time

At most 18 + 210 + 240 + 180 = 648 runs, at about 900 CPU-seconds each (1ae, 1ai): about
160 CPU-hours, so about 15 hours at W = 11, plus about 1.3 hours for S0.

## Logging

As round 1: `server-run/events.jsonl`, `run.log`, scoreboards, hourly and final bundles, and
the pilot's log (`phase1ak-pilot/knob-cap.json`). The chosen K is written into the KC arms
of 1ak, 1al and 1am.

## Implementation notes (added before the round starts)

* Round 2 writes its state to `server-run2/`, not `server-run/`, because round 1's DONE marker
  and finished steps are there. Start it with `bash tools/server/start.sh --plan round2`.
* The knob-cap stream is seeded by one extra draw from MOSES's global generator, only when
  K > 0. K = 0 stays byte-identical (golden test against 04c2020).
* The generation-0 contrast uses the penalized score. A NaN contrast scores 0.
* 1al and 1am do not start until `phase1ak-pilot/knob-cap.json` exists. K can differ per
  problem; the regression manifests are split per dataset for this.
