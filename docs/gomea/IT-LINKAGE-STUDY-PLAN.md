# IT linkage study: implementation plan (2026-11-20)

**Status:** a plan for an implementing agent. Not a pre-registration, and not approved for a
server run. Nothing is built. Backlog item: `BACKLOG.md` 13.9. Background: `T3.4-FEASIBILITY.md`.

**The user's request (2026-11-20, quoted):**

> "I want to test the first two options. I was thinking we test the IT only as a reference,
> replace the GOMEA's template with IT to see the linkage learning it uses has an effect, keeping
> this switch we test it with different type of linkage learning, the UNI, RT, LT, LT-B and the
> structural (which I assume to be the positional proximity based linkage tree building) and if
> we don't have the numbers for the GOMEA's performance with the template used alone for those
> different tests, we also need to run that experiment as well. [...] The objective is to see if
> the IT approach is indeed better for the linkage learning than the template alone and find
> linkage models that might have a better result when used with IT as opposed to the current
> version of the linkage model."
>
> "This is in hopes of finding a suitable structural way of learning globally a better model for
> the structure we are learning when it comes to the current version of MOSES we are planning to
> test. Don't do any implementation yet, just write a document that another agent can use to
> start it's implementation in detail from"

Earlier decisions of the user (2026-11-20, `APPROVALS.md`): the reason for this work is the
linkage research question; IT is explored with GOMEA outside MOSES first; the code is option (b),
the GOMEA engine already ported in `pymoses/gomea`.

**Origin of the parts:** the IT form is from a paper (de França and Aldeia, Evolutionary
Computation 2021). The template genome is GP-GOMEA's (Virgolin et al.). The study design (2
genomes × 5 linkage models) is the user's. Everything marked "agent proposal" below is the
agent's and needs the user's decision.

---

## 1. The research questions

| # | Question | How the study answers it |
|---|---|---|
| Q1 | Does learned linkage help more on the IT genome than on GP-GOMEA's template? | The genome × linkage interaction (section 7.2, family P1) |
| Q2 | Which linkage model works best with IT? | The five linkage models on the IT genome (family P2) |
| Q3 | Does our engine reproduce the known template result (learned linkage beats a random tree on training error)? | A gate before the main study (section 8.3) and family S1 |
| Q4 (later) | Which linkage model could work as a global, run-long model in MOSES? | Phase B (section 11), only after Q1–Q3 |

## 2. What we already know (do not run again without a reason)

| Source | Genome and engine | Linkage models | Result |
|---|---|---|---|
| Phase 1F part A (`PHASE1F-FINDINGS.md`) | template, height 4, the authors' C++ GP-GOMEA, population 2,000, 20 generations, no ERC, linear scaling, `+ - * aq`; 10 datasets × 30 splits (splits 1–30) | LT-MIb (= our LT-B), LT-MI (= our LT), RT | LT-MIb better than RT on **training** NMSE on 10 of 10 datasets, on test on 4 of 10; LT-MIb better than LT-MI on train on 5 of 10 (1 worse) |
| Koch, Alderliesten and Bosman, GECCO 2026 (`LITERATURE-2026-11-06.md`) | template, C++; Airfoil, Bike, Concrete, Dow, Tower; N 1,024 and IMS from 64; 10^7 evaluations | MI, bias-adjusted MI, random, node proximity, intron-masked MI | P(learned better than random) = 0.57; P(node proximity better than the MI variants and random) ≈ 0.80 |
| pyMOSES findings 1g, 1h, 1m, 1al, 1ay | MOSES knobs (change in every deme) | LT-B, RT, UNI | LT-B ties RT in every real run; LT-B beat RT only on one frozen representation with N = 1,024 (1h) |

**Gap:** no numbers exist for the template under **our** engine (`pymoses/gomea`), for UNI and NP
on the template with our datasets, or for any linkage model on an IT genome. So the study runs
both genomes under all five models (as the user asked).

## 3. The experiment design

### 3.1 Factors

| Factor | Levels |
|---|---|
| Genome | `template` (GP-GOMEA's tree template), `it` (the IT form) |
| Linkage model (FOS) | `uni`, `rt`, `lt-mi` ("LT"), `lt-b` ("LT-B"), `np` ("structural": node proximity on the genome's own tree) |

These map one to one to the existing modes of `pymoses.gomea.fos.LinkageModel` (`MODES = ("uni",
"rt", "lt-mi", "lt-b", "np")`). No new linkage code is needed for phase A, only a distance
function for `np` on each genome (section 5.4).

10 arms = 2 genomes × 5 linkage models. Every arm runs on the same datasets and splits, so arms
pair by (dataset, split).

### 3.2 Fixed settings (agent proposal; the pilot can change the sizes)

| Setting | Template | IT | Why |
|---|---|---|---|
| Genome length | 31 positions (full binary tree, height 4) | 32 positions (8 term slots × 2 factors × 2 boxes) | The same length, so the FOS sizes and the cost per generation match |
| Operators | `+ - * aq` (aq(a, b) = a / √(1 + b²)) | products of inputs with integer powers; the top-level sum | The Phase 1F part A set; no sin, exp, log on either side |
| Constants | none (no ERC), as in Phase 1F part A | none; least squares gives the weights | No constant genes, so no binning of constants in the statistics |
| Scaling of the output | linear scaling a + b·f on the training rows (GP-GOMEA `--linearscaling`) | per-term least squares w0 + Σ wi·termi (as `--linear-scaling terms`) | Each genome with its natural scorer (see decision D4) |
| Population | N = 1,000 (pilot also tests 2,000) | N = 1,000 | 1h: learned linkage needed about 1,024 |
| Budget | equal evaluations, E per run (pilot sets E; first value 500,000) | the same E | FOS sizes differ (UNI ≈ ℓ subsets, trees ≈ 2ℓ − 2), so equal generations would favour UNI |
| Forced improvements | as GP-GOMEA@750220c does by default (read `src/GOMEA/GOMVariator.cpp`; set `EngineParams.forced_improvement` to match) | the same | Match GP-GOMEA |
| Acceptance | `nonworse` (GP-GOMEA) | the same | |
| Inactive changes | `skip_inactive=True` with a phenotype key (GP-GOMEA does not evaluate a change of inactive nodes) | the same | Saves evaluations; required for a fair count |
| Stop rule | E evaluations, or convergence (all genomes equal); no IMS, no restart | the same | Phase B adds restarts |

### 3.3 Datasets and splits

* Datasets: the 10 regression datasets (Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac), with
  the same train/val/test split files as the server rounds (`tools/gomea/sr_datasets.py`).
* Splits: **fresh splits 71–80** for the main study (extend `SEEDS` to `range(1, 81)`, add the 80
  split files to `tools/server/setup.sh`; splits 1–70 stay byte-identical, as round 14 did for
  61–70). Rule of the project: 10 splits, then 10 more if a primary result is borderline
  (memory "10 seeds, then 10 more").
* If the CPU estimate (section 9) does not fit, the fallback is 5 datasets: Air, Con, Dow, Tow (4
  of Koch's 5; Bike is not in our set) and EnH. The user decides (decision D2).

## 4. Where the code goes

All new code is **opt-in and outside the MOSES search path**. No existing output may change.

| File (new unless marked) | Content |
|---|---|
| `pymoses/gomea/standalone/__init__.py` | Package marker; docstring "NOT A PORT: the IT linkage study (BACKLOG 13.9)" |
| `pymoses/gomea/standalone/genome_template.py` | `TemplateGenome` (section 5.2) |
| `pymoses/gomea/standalone/genome_it.py` | `ITGenome` (section 5.3) |
| `pymoses/gomea/standalone/positions.py` | The position tree of each genome and the `np` distance (section 5.4) |
| `pymoses/gomea/standalone/evaluator.py` | `RegressionEvaluator`, the `gom.Evaluator` protocol (section 5.5) |
| `pymoses/gomea/standalone/init.py` | The start population of each genome (section 5.6) |
| `pymoses/gomea/standalone/run.py` | The stand-alone run: CLI, engine call, outputs (section 6) |
| `pymoses/gomea/standalone/telemetry.py` | Linkage telemetry per generation (section 6.3) |
| `tools/gomea/moses_sr_runner.py` (change) | A runner-only arm key `"standalone"`, like `"linear"` in round 14 (section 6.4) |
| `tools/gomea/sr_datasets.py`, `tools/server/setup.sh` (change) | Splits 71–80 |
| `configs/gomea/<phase>/`, `tools/server/plans/planNN.json` | Phase configs and the plan (after the pilot) |
| `tests/gomea/standalone/test_*.py` | Tests (section 8) |

**Reuse, do not copy:**

* `pymoses.gomea.engine.GomeaEngine` and `EngineParams`: the generation loop. It is independent of
  MOSES: it takes `domains`, an `Evaluator`, two RNG streams, an evaluated start population and,
  for `np`, a `distance` callable.
* `pymoses.gomea.gom`: `Individual`, `Evaluator`, `GomStats`, `GomInterrupted`.
* `pymoses.gomea.oracle.BudgetExhausted`: raise it from `evaluate` when E is spent.
* `pymoses.gomea.fos.LinkageModel`: the five modes. Note: `calibrate` excludes every position
  that is constant in the start population for the whole run, so the start population must vary
  every position (section 5.6).
* `pymoses.moses.scoring.least_squares.fit_terms` and `tools/gomea/sr_linear.py`
  (`independent_columns`): the per-term fit and the drop of aliased columns.
* `pymoses.util.rng.MT19937RandGen`: all randomness, seeded from the split seed.
* `tools/gomea/sr_eval.py`: the final scoring cross-check (section 6.2).

## 5. Components in detail

### 5.1 Common rules

* A genotype is a `tuple[int, ...]` of codes; `domains[p]` is the number of codes at position p.
* Every genome class gives: `length`, `domains`, `decode(g) -> Model`, `phenotype_key(g) ->
  Hashable` (equal keys only for equal decoded models), `active_mask(g) -> tuple[bool, ...]`
  (which positions affect the model), `size(g)` and `to_moses(g) -> str` (the model in MOSES combo
  syntax, for the cross-check and the reports).
* Decoding is pure and deterministic. No MOSES reduction is applied.

### 5.2 `TemplateGenome` (GP-GOMEA's template)

* Full binary tree of height h = 4: 31 positions in breadth-first order (position 0 is the root;
  the children of p are 2p + 1 and 2p + 2). 15 internal positions, 16 leaves.
* Domain of an internal position: the functions `+ - * aq` (codes 0–3) and the n inputs (codes
  4 … n + 3). Domain of a leaf: the n inputs only (codes 0 … n − 1).
* Decoding: from the root; a position that holds an input is a leaf of the expression, and all
  positions below it are inactive (introns), as in GP-GOMEA (`GOMEATreeInitializer.cpp` always
  builds the full template; unused children stay as introns).
* Evaluation: column-wise over the rows (lists of floats; no numpy on the server). `aq(a, b) = a /
  sqrt(1 + b*b)`. A non-finite output makes the model invalid (score −∞).
* `phenotype_key`: the string of the active expression.
* `size`: the number of active nodes (functions and inputs), as GP-GOMEA's "Number of nodes".
* `to_moses`: `aq(a, b)` written as `/(a exp(*(0.5 log(+(1 *(b b))))))`. The MOSES interpreter
  has `exp` and `log` (the search excludes them only through `-n`), so `sr_eval.py` can score the
  model. This is only for the cross-check.

### 5.3 `ITGenome` (the IT form)

* m = 8 term slots, f = 2 factor slots per term, 2 boxes per factor: **input** and **power**.
  Position order: term 0 factor 0 input, term 0 factor 0 power, term 0 factor 1 input, …
  (32 positions).
* Domain of an input box: n + 1 codes; code 0 = "none", code j = input j. Domain of a power box:
  4 codes for the powers (−2, −1, 1, 2).
* Rule for negative powers (from `T3.4-FEASIBILITY.md`): an input with a zero in the training rows
  never takes a negative power. Implement it in the decoder: such a power becomes its absolute
  value (−2 → 2, −1 → 1), and record the count in the telemetry. Do not change the domains, so
  all genomes keep the same length and meaning.
* Decoding: a factor is input^power, or nothing when the input is "none" (its power box is then
  inactive). A term is the product of its factors; a term with no factor is off (all its boxes
  are inactive except the input boxes). Equal monomials in two slots are merged into one term
  (the decoder sums them), so the fit never gets two equal columns. For a 0/1 input, x² = x and
  x⁻¹ is not allowed.
* Fit: columns = the active, distinct terms; drop aliased columns (`sr_linear.independent_columns(cols)` returns the indices to keep);
  then `least_squares.fit_terms`. If the fit falls back (singular, non-finite), score the model
  with linear scaling of the sum of its terms, as `bscores` does for `--linear-scaling terms`.
  The empty model scores as the mean (w0 only).
* `phenotype_key`: the sorted tuple of (sorted (input, power) pairs) of the distinct active terms.
* `size`: the node count of `to_moses` plus the number of terms (the round 13–14 report rule).
  Also report the number of terms and the number of fitted weights.
* `to_moses`: `+(t1 t2 …)`, where x^2 is `*($j $j)`, x^−1 is `/(1 $j)` and x^−2 is
  `/(1 *($j $j))`.
* **The slot-swap problem (design choice for phase A, agent proposal):** do not reorder slots in
  the genome; positions never move. Duplicates are handled only by the merge in the decoder.
  Measure the effect with the telemetry "slot dispersion" (section 6.3). Reordering would change
  what a position means during a run, and that breaks the copy semantics of GOM.

### 5.4 The structural model (`np`) on each genome

* `LinkageModel` mode `np` builds the linkage tree on S(i, j) = −d(i, j), where d is the number of
  edges between positions i and j in a tree. The engine takes this as `distance(positions) ->
  list[list[int]]`.
* **Template:** d is the edge count between the two template positions through their lowest
  common ancestor, in the full binary tree of section 5.2. This is Koch et al.'s node proximity.
* **IT:** the implicit tree is root → 8 terms → 2 factors each → 2 boxes each. So d = 2 for the
  two boxes of one factor, 4 for boxes in the same term but different factors, 6 for boxes in
  different terms. The resulting static FOS groups each factor, then each term, then pairs of
  terms.
* Implement both with a parent array (`positions.py`). Check against
  `pymoses.gomea.node_proximity.tree_distances` on one shared small tree in a test, so the
  definition is the same as in pyMOSES.

### 5.5 `RegressionEvaluator`

Implements `pymoses.gomea.gom.Evaluator`:

* `evaluate(g)`: if the evaluation count has reached E, raise `BudgetExhausted("global_budget")`.
  Otherwise decode, fit on the **training rows**, and return the score = −MSE of the scaled
  model on the training rows (higher is better). Invalid models score −∞.
* `worse(a, b)`: `a < b`, with −∞ handled; never NaN (map NaN to −∞).
* `phenotype_key(g)`: the genome's key (sections 5.2, 5.3).
* `check()`: `"global_budget"` when E is spent, else `None`.
* `record_noop_skip`, `record_phenotype_skip`: counters for the telemetry.
* A score cache keyed by phenotype (a cache hit does **not** count as an evaluation, as in GP-GOMEA
  with caching; record hits). Agent proposal for speed on IT: cache each distinct monomial column
  and the dot products between columns, so a new model only computes its new parts. The cache
  must give bit-identical scores (a test compares cached and uncached runs).
* Training rows: the split's `train` rows with the target scaled as the runner does (`y' = 10 (y −
  mean) / sd`). Validation and test rows are used only after the run (section 6.2).

### 5.6 Start population

* **Template:** GP-GOMEA's GOMEA default, half-and-half (`--inittype` default HH for GOMEA,
  `EvolutionState.cpp:458`): half of the trees with `GenerateTreeFull`, half with
  `GenerateTreeGrow`, at height h, every node with all its children so that the full template
  exists (`GOMEATreeInitializer.cpp`). Port these two functions with `MT19937RandGen` draws and a
  provenance note (Apache-2.0, as the other ported files).
* **IT (agent proposal):** each term slot is active with probability 0.5; in an active slot, each
  factor's input is uniform over the n inputs with probability 0.5 and "none" otherwise, and at
  least one factor is active; every power is uniform over its 4 values (also in inactive boxes, so
  that every position varies). The pilot reports the start diversity (section 9).
* Both: reject duplicate genomes up to a try cap, as `pymoses.gomea.population.build_population`
  does; every position must vary in the start population, otherwise `LinkageModel.calibrate`
  drops it for the whole run (check and log it).

## 6. The run, its outputs and the runner

### 6.1 CLI

```
python -m pymoses.gomea.standalone \
    --genome {template,it} --fos {uni,rt,lt-mi,lt-b,np} \
    --train train_scaled.csv --val val.csv --test test.csv --target y \
    --pop 1000 --max-evals 500000 --seed SPLIT_SEED \
    [--template-height 4] [--it-terms 8 --it-factors 2 --it-powers=-2,-1,1,2] \
    --out RUN_DIR
```

* RNG streams: seed a master `MT19937RandGen` with `--seed`, then draw three streams from it in
  this order: start population, linkage, variation, as `pymoses/gomea/optimizer.py:774-776` does
  (`rng_init`, `rng_link`, `rng_var`). A run is then reproducible from the seed.
* The engine: `GomeaEngine(domains, EngineParams(fos_mode=..., forced_improvement=..., skip_inactive=True,
  max_generations=a large number, stall_generations=a large number), ev, rng_var, rng_link,
  distance=<only for np>, on_generation=<telemetry hook>)`. The run ends on E or convergence.

### 6.2 Outputs (`RUN_DIR`)

* `summary.json`, with the keys of the runner's summaries where they apply: `status`, `arm`,
  `task`, `seed`, `final_program` (`to_moses`), `train_nmse`, `val_nmse`, `test_nmse`,
  `test_nmse_clipped`, `size`, `nodes`, `n_terms` (IT), `n_evals`, `cpu_s`, `wall_s`,
  `ru_maxrss_kb`, `generations`, `termination_reason`, `cache_hits`, and a `standalone` block
  (genome, fos, pop, E, template or IT sizes).
* The reported model is the **best training score** of the final population (no selection on
  validation; this is an optimization study). Also report the validation NMSE of that model.
* NMSE as in the rounds: 100 × MSE / var(y) of the split, with the scaling fitted on the training
  rows. **Cross-check:** score `to_moses` with `sr_eval.evaluate` (with `terms=True` for IT); the
  two NMSE values must agree to a relative 1e-9, else the run fails with a clear error.
* `events.jsonl`: one record per generation (section 6.3) and one `end` record.

### 6.3 Telemetry per generation (needed for Q2 and phase B)

* best and median training NMSE; evaluations so far; cache hits; noop and phenotype skips;
  GOM accepted improving and neutral; FI events;
* the FOS: size, mean subset size, a hash; the Jaccard similarity to the previous generation's FOS
  (stability of the learned model);
* the top 20 position pairs by MI (for `lt-mi`, `lt-b`), and their **structural alignment**:
  the share that lies in the same factor or term (IT) or that are parent and child or siblings
  (template);
* accepted improvements by subset size (which subset sizes carry the useful changes);
* the share of active positions in the population;
* IT only: **slot dispersion**, for each distinct monomial in the population, the number of slots
  that hold it (median and maximum); the count of negative powers forced positive.

### 6.4 Runner integration

* Add the arm key `"standalone"` to `tools/gomea/moses_sr_runner.py`, following the `"linear"`
  arm of round 14 (`_run_linear`, `_check_linear`, the `expand` branch): the arm is
  `{"standalone": {"genome": "it", "fos": "lt-b", "pop": 1000, "max_evals": 500000, ...}}`, with no
  other key. The runner writes the scaled CSVs as for MOSES, runs the module, and keeps its
  `summary.json`. The plan, `chain.py`, `start.sh`, `status.sh` and `phase_report.py` then work as
  in the earlier rounds.
* Do not change any existing arm type. Identity tests: the round 14 manifests expand to the same
  jobs, byte for byte.

## 7. Analysis plan (to be fixed in the pre-registration)

### 7.1 Metrics

| Metric | Role |
|---|---|
| Training NMSE of the reported model at E | **primary** (linkage is an optimization effect; Phase 1F part A and Table 3 also use it) |
| Test NMSE (and clipped) | secondary (relevance for MOSES) |
| Anytime: the area under best-so-far log(training NMSE) over the evaluations (checkpoints at 1, 2, 5, 10, 20, 50, 100 % of E) | secondary (Koch et al.: node proximity reached the final accuracy at about one third of the budget) |
| Evaluations to reach the median final training NMSE of RT on the same genome | secondary |
| Size, n_terms, CPU | descriptive |

### 7.2 Contrasts (agent proposal)

Statistics as in rounds 12–14: paired Wilcoxon signed-rank on log ratios over the (dataset,
split) pairs, two-sided, Holm, α = 0.025; "better" or "worse" also needs a ratio at least 10 %
from 1 (see decision D5).

* **P1, the interaction, Q1 (4 contrasts, primary):** for each f in {UNI, LT, LT-B, NP}, the
  per-pair value d = log(IT_f / IT_RT) − log(T_f / T_RT) on training NMSE. d < 0 means that model
  f gains more over the random tree on IT than on the template. Ratio = exp(median d).
* **P2, the linkage models on IT, Q2 (5 contrasts, primary):** IT-UNI, IT-LT, IT-LT-B, IT-NP each
  against IT-RT; IT-NP against IT-LT-B (structure against learned).
* **S1, the same 5 on the template, Q3 (secondary):** to compare with Phase 1F part A and Koch et
  al.
* **S2:** IT-LT-B against T-LT-B and IT-NP against T-NP on test NMSE (the absolute level of the two
  genomes; confounded by the scorer, see D4).
* **S3:** P1 and P2 on test NMSE and on the anytime area.
* **Per dataset:** P1 and P2 per dataset, descriptive (no verdicts).

## 8. Tests and gates for the implementing agent

### 8.1 Unit tests

* Domains and lengths of both genomes; `decode` on hand-written genomes (for example the IT form A
  of the lessons: x1·x2 + x3, and a template with an input at the root, which makes all other
  positions inactive).
* `active_mask` agrees with `phenotype_key`: changing an inactive position never changes the key.
* The IT merge of equal monomials; the negative-power rule; 0/1 inputs; the empty model.
* `to_moses` scored by `sr_eval.evaluate` equals the genome's own NMSE (template with `aq`; IT
  with `terms=True`).
* The `np` distances (both genomes), including the check against
  `node_proximity.tree_distances`.
* `RegressionEvaluator`: the budget raises `BudgetExhausted` at exactly E; cache hits are not
  counted; NaN maps to −∞.
* Reproducibility: the same seed gives the same `summary.json` (except times) for each of the 10
  arms on a tiny dataset.
* The cached and uncached IT scorers give bit-identical runs.

### 8.2 Identity tests

* No existing output changes: run the existing identity harness (`tools/gomea/identity_check.py`)
  and the full suite. The runner change must leave every existing manifest's job list unchanged.

### 8.3 Gates before any server run (report to the user if one fails)

1. **Template replication gate:** on 3 datasets (Con, EnH, Tow) × 5 splits from 1–30, template
   with N = 2,000 and 20 generations (the Phase 1F part A settings): T-LT-B must have lower median
   training NMSE than T-RT on each dataset, as the C++ code did on 10 of 10. If not, the engine or
   the template port differs from GP-GOMEA, and the study stops.
2. **IT recovery gate:** on a synthetic table (n = 5 inputs, 500 rows, y = 3·x1·x2 + 5·x3 + small
   noise), IT-LT-B finds the true terms in at least 4 of 5 seeds within 50,000 evaluations.
3. **Cross-check gate:** in the pilot, every run passes the `sr_eval` cross-check of section 6.2.
4. **Resources gate:** the peak memory of each arm is at most 50 % of its manifest limit.

## 9. Pilot and CPU estimate

* **Measured (agent, 2026-11-20, local machine, pure Python, 500 rows; measurement only):** one
  evaluation of a random full height-4 template tree with linear scaling took 0.46 ms; one
  evaluation of an 8-term IT model (2 factors per term) with `fit_terms` took 2.43 ms. The server
  CPU is slower; assume about 2× (template ~1 ms, IT ~5 ms) until the pilot measures it.
* **Estimate:** CPU per run ≈ E × (cost per evaluation) + the linkage learning per generation.
  At E = 500,000 and 500 rows: template about 8 min, IT about 40 min (before caching).
* **Correction (2026-11-20, before any decision on the sizes):** the first estimate assumed 500
  training rows. The datasets have 154–2,501 training rows (Air 753, Bos 254, Con 516, Dow 534,
  EnC 384, EnH 384, Tow 2,501, WiR 801, WiW 2,450, Yac 154; mean 873, 1.75× the benchmark), and the
  cost of an evaluation grows about linearly with the rows. Corrected, at E = 500,000:

  | Design | CPU hours | Server time at W = 11 |
  |---|---|---|
  | 10 datasets, no IT cache | about 730 | about 66 h (2.8 days) |
  | 5 datasets (Air, Con, Dow, Tow, EnH; mean 938 rows), no IT cache | about 390 | about 36 h (1.5 days) |
  | 10 datasets, IT column cache 3× faster (an assumption) | about 320 | about 29 h (1.2 days) |

  These numbers assume a server 2× slower than the local machine, and that every run uses all E
  evaluations; the pilot measures both. Under D2, 10 datasets fit only if the pilot shows about two
  chained rounds (about 30 h); so the IT column cache of section 5.5 is required, not optional,
  for the 10-dataset design. Other levers: a smaller E, or the 5-dataset fallback.
* **The pilot (clone, local, at nice 19):** all 10 arms on Con and Dow, split 71, with N = 1,000 and
  E = 500,000. Report: evaluations and CPU per arm; cache hit rate; generations reached; start
  diversity (every position varies); the gate results. From it, the agent proposes E, N and the
  dataset set so that the study fits the user's round rules (one round of about 15 h, or rounds
  that chain one after the other).
* The server venv has no numpy (round 14 `pip-freeze.txt`). Do not add it without the user's
  approval.

## 10. Implementation order and project rules

1. **The user decides the open decisions** (section 12).
2. **Pre-registration** (the agent who talks to the user writes it; the user approves).
3. **Clone K** (subagent, separate clone, nice 19): sections 4–6 and the tests of section 8.1–8.2.
4. **Clone L** (subagent, after K merges): splits 71–80, configs, the plan, the pilot, the gates of
   section 8.3, the dry run, a local smoke run of the chain.
5. Merge, full suite, the RV16 review (all findings to the user unchanged), the push only with the
   user's fresh approval, then the server round(s) in one go (`server-round-one-go` memory).
6. Findings document; then the user decides on phase B.

Rules for every implementing agent (from the user's global rules and this project's memory):

* Commit as the user only (the repo's `user.name` and `user.email`); no attribution lines of any
  kind. Write in ASD-STE100 Simplified Technical English.
* Work in a separate clone at nice 19; never use git worktree; never merge to main, rebase or
  force-push; never push.
* Never commit `MOSES_GOMEA_LOCAL_AGENT_HANDOFF/`, `reports/`, `research_notes/`, or Dow or Tow
  data rows (they have no licence).
* Mark new code "NOT A PORT" (or give the GP-GOMEA provenance for the ported initializer).
  Every new behaviour is opt-in; identity tests prove that existing outputs do not change.
* Python: `.venv/bin/python` with `PYTHONPATH=<clone>` and `-o addopts=""`. Put suite temp files
  in `../pyMoses-suite-tmp`, not in `/tmp`. Check `df -h /tmp` before heavy steps; stop at 75 %.
* Kill processes only by PID or PGID. Read untrusted bundles with `python3 -I`, each into a new
  folder.
* Do not read any NMSE of the main splits (71–80) before the pre-registration is approved; the
  pilot uses split 71 only for cost and gates, and its NMSE is not read.

## 11. Phase B (later; agent ideas, not approved)

Only after phase A, and only with the user's decision. Candidates for "a global, run-long linkage
model" for MOSES (Q4), all on the IT genome:

| Candidate | Idea | Why it fits MOSES |
|---|---|---|
| Pooled LT-B across restarts | The run restarts the population (like MOSES demes) several times; the MI comes from an archive of the good genomes of all restarts | It tests the pooling of the lessons (step 7) directly |
| NP plus data | Start from the `np` tree and let MI re-order merges inside each term | A structural prior with data |
| Intron-masked MI | Inactive boxes get their own "masked" code in the statistics (Koch et al. 2026) | IT marks inactive boxes exactly |
| Template-free check | IT with more slots (16 × 3) | Tests whether the result depends on the form size |

## 12. Decisions for the user before the pre-registration

**The user accepted D1–D7 as recommended (2026-11-20: "Accept D1–D7 as recommended").**

| # | Decision | Agent recommendation |
|---|---|---|
| D1 | Population and budget: N = 1,000 and E = 500,000 as the start, fixed after the pilot | yes; the pilot sets the final values |
| D2 | Datasets: all 10, or the 5-dataset fallback if the CPU does not fit | all 10 if the pilot fits two chained rounds; else 5 |
| D3 | Genome sizes: template height 4 (31) and IT 8 × 2 (32) | yes; equal lengths make the linkage comparison fair |
| D4 | Scorer: each genome with its natural scaling (template a + b·f, IT per-term least squares) | yes for phase A; P1 compares gains within each genome, so the scorer difference affects it less than S2 |
| D5 | The 10 % band for verdicts | keep the project rule, but also report direction and Holm p for every contrast, because linkage effects on training NMSE were small in Phase 1F (often below 1 NMSE point) |
| D6 | The primary metric: training NMSE (optimization) rather than test NMSE | training NMSE, with test as secondary |
| D7 | The IT start population of section 5.6 | yes, checked in the pilot |
