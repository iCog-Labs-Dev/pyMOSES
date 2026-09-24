# pyMoses + GOMEA Implementation Plan (Phase 1 discrete, Phase 2 mixed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GP-GOMEA-style population inner optimizer (`-a gomea`) to pyMoses. It
optimizes within each frozen MOSES representation and leaves exemplar selection,
representation building and metapopulation merging untouched. It is benchmarked against
the matched MOSES hill climber (Phase 1: discrete fields). Phase 2 extends it with
per-individual real constants and interleaved RV-GOMEA.

**Architecture:** A MOSES-independent engine (`pymoses/gomea/{stats,fos,gom,engine}.py`)
works on tuples of categorical field values through an injected evaluator. A thin adapter
(`schema.py`, `oracle.py`, `population.py`, `optimizer.py`) freezes one representation's
logical fields and samples the initial population with MOSES's own neighbourhood sampler.
Every evaluation goes through the stock `complexity_based_scorer`, and every evaluated
trial is appended to the deme, which is exactly the contract hill climbing uses. The
linkage statistics, UPGMA/RNN linkage tree and GOM are narrow Python ports of GP-GOMEA
[C1] `750220c` (Apache-2.0), with golden parity fixtures from a g++ harness built on
verbatim upstream code.

**Tech Stack:** Python 3.14 standard library only (pyMoses has no runtime dependencies),
pytest, ruff, g++ for the golden harness only, and hand-written SVG for plots.

## Global Constraints

- Python `>=3.14`; **no new runtime dependencies** (`pyproject.toml` `dependencies = []` stays empty).
- `-a hc` (and every existing flag) must stay byte-identical to the pre-change behaviour; **no edits to `hill_climbing.py`, `neighborhood_sampling.py`, `representation.py`, scoring or metapopulation modules**.
- New code lives in `pymoses/gomea/`; it must not contain `DIVERGENCE(` markers (those are reserved for the C++ port and are machine-checked); upstream provenance uses the first-docstring-line form `GP-GOMEA: <path>:<line> <symbol>`.
- Ruff: line length 100, rules `E,F,W,I,N,UP,B` (see `pyproject.toml`).
- Branches: `experiment/moses-gomea-discrete` (Phase 1) from `0410881`; `experiment/moses-gomea-mixed` (Phase 2) from the approved Phase 1 commit. Never `git worktree`, never push/merge/rebase.
- Commits authored as the repo's configured user only; **no attribution trailers** (user's global CLAUDE.md).
- Bulk results outside the tree: `../gomea-results/phase1/…`, `../gomea-results/phase2/…`; commit manifests, checksums, reports only.
- One local invocation = one frozen schema; population, linkage calibration and (Phase 2) RV state are rebuilt per invocation and never shared across schemas.
- Budget unit = one call of the MOSES `iscorer` (identical to HC's `deme.n_evals` accounting); cache hits/misses, no-op skips and proven-phenotype skips are counted separately.

## File Structure (Phase 1)

| File | Responsibility |
|---|---|
| `pymoses/gomea/__init__.py` | package docstring + public exports |
| `pymoses/gomea/schema.py` | `FrozenSchema`: logical fields, per-field domains, default genotype, instance⇄genotype codec, `schema_id`, decode helpers |
| `pymoses/gomea/oracle.py` | `EvaluationLedger`, `EvaluationOracle`: the single evaluation path; global/inner/time budget checks; deme archive; proven-phenotype key |
| `pymoses/gomea/stats.py` | entropy / MI / initialization-biased similarity (port of `GOMEAFOS::GenerateLinkageTreeFOS` statistics) |
| `pymoses/gomea/fos.py` | univariate, random-tree and UPGMA/RNN linkage-tree FOS (ports) + `LinkageModel` (4 modes, fixed calibration) |
| `pymoses/gomea/gom.py` | GOM of one recipient against an immutable donor snapshot; optional forced improvement |
| `pymoses/gomea/engine.py` | `GomeaEngine`: generation loop, termination reasons, diagnostics |
| `pymoses/gomea/population.py` | sampler-based multi-radius initial population, dedup/coverage policy, per-field diagnostics |
| `pymoses/gomea/optimizer.py` | `gomea_parameters`, `gomea_optimization(optimizer_base)`: the MOSES seam |
| `pymoses/gomea/telemetry.py` | JSONL event sink (no-op when disabled) |
| `pymoses/moses/main/problem_params.py` | `-a gomea` + `--gomea-*` options |
| `pymoses/moses/moses/moses_main.py` | dispatch `gomea` |
| `tools/gomea_golden/golden.cpp`, `tools/gomea_golden/capture.py` | verbatim-upstream C++ harness + fixture capture |
| `tests/fixtures/gomea_golden.json` | golden LT / similarity fixtures |
| `tests/gomea/test_*.py` | contract, engine, parity, integration tests |
| `benchmarks/gomea/{runner,instrument,fixtures,analyze,svgplot,manifest}.py` | experiment harness |
| `configs/gomea/phase1/*.json` | frozen resolved configurations and manifests |
| `docs/gomea/*.md` | audit, approvals, gates, provenance, preregistration, findings, Phase 2 handoff |

---

## Phase 1

### Task 1: Docs scaffold: audit, approvals, plan, provenance (P1.0 / P1.1 entry)

**Files:**
- Create: `docs/gomea/P1.0-AUDIT.md`, `docs/gomea/APPROVALS.md`, `docs/gomea/PROVENANCE.md`, `docs/gomea/PHASE1-GATES.md`, this plan
- Modify: none

- [ ] **Step 1:** Write `PROVENANCE.md` with the component table: requirement, source + commit, file:symbol, licence, choice (port), modifications, parity test, and unresolved behaviour. The rows are entropy, bias, LT, UNI FOS, RT FOS, GOM, FI (reconstruction), and experiment utilities (new code).
- [ ] **Step 2:** Record the stock baseline: `pytest -q` log at `../gomea-results/phase1/baseline/pytest_stock_0410881.log`, and the Python/g++ versions in `../gomea-results/phase1/baseline/environment.json`.
- [ ] **Step 3:** Commit: `git add docs/gomea docs/superpowers/plans/2026-09-24-moses-gomea.md && git commit -m "Record the GOMEA audit, approvals and plan"`

### Task 2: FrozenSchema (contracts 1, 4, 6)

**Files:**
- Create: `pymoses/gomea/__init__.py`, `pymoses/gomea/schema.py`, `tests/gomea/__init__.py` (empty, only if the tests package needs it — check `tests/moses/` layout; the repo uses rootdir-relative imports without `__init__`), `tests/gomea/conftest.py`, `tests/gomea/test_schema.py`

**Interfaces:**
- Produces:
  ```python
  class UnsupportedRepresentation(ValueError): ...
  class FrozenSchema:
      def __init__(self, rep: representation) -> None   # raises UnsupportedRepresentation if rep.fields().n_contin_fields() > 0
      rep: representation
      fields: field_set
      dim: int                               # == fields.raw_size() (disc + bit, no contin)
      domains: tuple[int, ...]               # multy per logical field (bits -> 2)
      schema_id: str                         # sha256 hex of prototype text + domains + packed layout
      def default_genotype(self) -> tuple[int, ...]            # from instance(fields.packed_width())
      def to_genotype(self, inst: instance) -> tuple[int, ...]
      def to_instance(self, g: Sequence[int]) -> instance      # raises ValueError on out-of-domain value / wrong length
      def phenotype_key(self, g: Sequence[int]) -> tuple        # behave_cscore._cache_key(rep.get_candidate(inst, False))
      def decode(self, g: Sequence[int]) -> combo_tree          # rep.get_candidate(inst, True)
  ```
- Shared test helper (`tests/gomea/conftest.py`): `make_boolean_rep(exemplar: str, arity: int) -> representation`. It uses the same simplifiers and type signature the `pa`/`maj` demo problems use, so find how `tests/moses/deme/` builds a representation and reuse that construction.

- [ ] **Step 1: Write failing tests** covering:
  * the default genotype decodes to the same program and score as the exemplar;
  * encode/decode round-trips every field;
  * `to_instance` rejects code 3 on a multy-3 field;
  * two different exemplars with the same `dim` give different `schema_id`s;
  * `schema_id` is stable across two constructions of the same representation;
  * a contin representation raises `UnsupportedRepresentation`;
  * `phenotype_key` is equal for two genotypes that differ only in a field nested under a disabled knob, and for them `decode` gives equal reduced trees;
  * `to_genotype(to_instance(g)) == g` for 200 random legal genotypes.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/gomea/test_schema.py -q`. Expected: fails with an ImportError.
- [ ] **Step 3:** Implement `schema.py`. Raw index `k` is logical field `k`, because contin fields are rejected. Domains come from `fields.disc_and_bit()[k].multy`. `schema_id = sha256(rep.ostream_prototype() + "|" + repr(domains) + "|" + repr([(f.width, f.major_offset, f.minor_offset) for f in fields.fields()]))`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Run `ruff check pymoses/gomea tests/gomea`.
- [ ] **Step 5:** Commit with the message "Freeze a MOSES representation's logical fields for GOMEA".

### Task 3: EvaluationOracle and ledger (contracts 2, 3, 5)

**Files:** Create `pymoses/gomea/oracle.py` and `tests/gomea/test_oracle.py`.

**Interfaces:**
- Consumes: `FrozenSchema` (Task 2).
- Produces:
  ```python
  class BudgetExhausted(Exception): reason: str        # 'global_budget' | 'inner_budget' | 'time_limit'
  @dataclass
  class EvaluationLedger:
      requests: int = 0            # evaluate() calls that reached the scorer (== budget unit)
      invalid: int = 0             # scorer returned worst_composite_score (penalized score == very_worst_score)
      noop_skips: int = 0          # trial genotype == recipient genotype
      phenotype_skips: int = 0     # proven equal unreduced decoded tree
      cache_hits: int = 0          # delta of behave_cscore._cache_hits over the call (if present)
      cache_misses: int = 0
      def as_dict(self) -> dict
  class EvaluationOracle:
      def __init__(self, schema, iscorer, deme, max_evals: int, max_time: float,
                   inner_cap: int | None = None, clock=time.monotonic) -> None
      ledger: EvaluationLedger
      def check(self) -> str | None          # budget reason or None; checked before EVERY evaluation
      def evaluate(self, g: tuple[int, ...]) -> composite_score   # raises BudgetExhausted; appends scored_instance to deme
      def best_so_far(self) -> composite_score
      def evals_at_best(self) -> int
      def finalize(self) -> None             # deme.n_evals = requests; deme.n_best_evals = evals_at_best
  ```
- [ ] **Step 1: Write failing tests** covering:
  * the oracle score equals `iscorer(inst)` for 50 random genotypes, calling HC's scorer directly (contract 3);
  * two evaluations never alias (mutating a returned instance or score does not change the deme entry) (contract 2);
  * `max_evals=0` raises `BudgetExhausted('global_budget')` before calling the scorer;
  * an inner cap of 3 stops at exactly 3 requests even when global = 100;
  * the time limit uses an injected fake clock;
  * `finalize` sets `deme.n_evals == len(deme) == requests`;
  * invalid counting uses a scorer stub that raises inside `get_candidate`, which `complexity_based_scorer` turns into worst.
- [ ] **Step 2:** Run them. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement. Cache deltas are read from `getattr(iscorer, "_cscorer", None)` and `_cache_hits/_cache_misses`; they are 0 if absent. The deme entry is `scored_instance(schema.to_instance(g), copy.copy(score))`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Route every GOMEA evaluation through one budgeted oracle".

### Task 4: Linkage statistics, FOS builders, golden harness

**Files:**
- Create: `pymoses/gomea/stats.py`, `pymoses/gomea/fos.py`, `tools/gomea_golden/golden.cpp`, `tools/gomea_golden/capture.py`, `tests/fixtures/gomea_golden.json`, `tests/gomea/test_stats_fos.py`, `tests/gomea/test_golden_parity.py`

**Interfaces:**
- Produces:
  ```python
  # stats.py  (natural log, like upstream)
  def entropy_matrix(pop: Sequence[Sequence[int]], positions: Sequence[int]) -> list[list[float]]
      # H[i][i] = H(field i), H[i][j] = joint entropy (i != j), over positions (indices into genotype)
  def mi_similarity(H) -> list[list[float]]                       # S[i][j] = H_ii + H_jj - H_ij ; S[i][i] = H_ii
  def biased_similarity(H, H0) -> list[list[float]]               # H_ii/H0_ii + H_jj/H0_jj - 2 H_ij/H0_ij ; diag H_ii/H0_ii
  # fos.py
  def univariate_fos(d: int) -> list[list[int]]
  def random_tree_fos(d: int, rng: MT19937RandGen) -> list[list[int]]
  def linkage_tree_fos(S: list[list[float]], order: Sequence[int]) -> list[list[int]]  # UPGMA/RNN incl. root
  class LinkageModel:
      MODES = ('uni', 'rt', 'lt-mi', 'lt-b')
      def __init__(self, mode: str, rng: MT19937RandGen) -> None
      def calibrate(self, pop: Sequence[tuple[int, ...]], domains: Sequence[int]) -> None
          # once per invocation: fixes search positions = fields with H0 > 0; stores H0 over them
      search_positions: tuple[int, ...]
      constant_positions: tuple[int, ...]
      def learn(self, pop) -> list[list[int]]      # FOS over genotype indices; root excluded when d>1; d==1 -> [[p]]; d==0 -> []
  ```
- Upstream-faithful rules (from `GOMEAFOS.cpp@750220c`):
  * The LT uses `BuildLinkageTreeFromSimilarityMatrix` with `mpm` initialized in `order`. The chain starts at `mpm[0]` (upstream's `(int)randu()*len` is always 0). The NN tie rule prefers the smaller cluster, then the lower index, and the chain tie rule is copied as is. Merge weights are cardinality-weighted. There is no clamping.
  * `linkage_tree_fos` returns all `2d-1` sets. `LinkageModel.learn` drops the last (root) for `d > 1`.
  * The random tree merges random pairs while there are more than 2 clusters (upstream loop), so it has no root.
  * Univariate returns all singletons.
  * Positions are mapped back to genotype indices.
- Golden harness: `golden.cpp` holds verbatim copies of `BuildLinkageTreeFromSimilarityMatrix`, `DetermineNearestNeighbour` and the entropy/adjustment loops. The only substitutions are `std::vector<std::vector<double>>` for `arma::mat` and an injected `random_order` for `random_shuffle`. It reads cases on stdin and prints JSON. `capture.py` compiles it with `g++ -O2 -std=c++17` and writes `tests/fixtures/gomea_golden.json` (≥ 12 cases: unequal alphabets, dependent pairs, a second-generation population with fixed H0, negative similarities, d = 2, 3, 12, ties).

- [ ] **Step 1:** Write `golden.cpp` and `capture.py`. Run the capture and commit the fixture JSON together with its source commit hash.
- [ ] **Step 2: Write failing tests:**
  * parity with the golden cases (similarity matrices to `1e-12`, FOS exactly equal);
  * the identity `biased_similarity(H0, H0)` off-diagonal == 0;
  * label permutation within a field leaves `H` unchanged;
  * a negative similarity still merges (not blocked);
  * FOS coverage: every position appears in some set, and the root is excluded when `d > 1`;
  * `d = 1 → [[p]]`, `d = 0 → []`;
  * constant fields are excluded from `search_positions`;
  * a seeded tie-break is deterministic across two runs.
- [ ] **Step 3:** Implement `stats.py` and `fos.py`. Each ported function's first docstring line is `GP-GOMEA: src/GOMEA/GOMEAFOS.cpp:<line> <Symbol>`.
- [ ] **Step 4:** Run `pytest tests/gomea/test_stats_fos.py tests/gomea/test_golden_parity.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Port GP-GOMEA's linkage statistics and FOS builders with golden parity".

### Task 5: GOM, forced improvement, engine

**Files:** Create `pymoses/gomea/gom.py`, `pymoses/gomea/engine.py`, `tests/gomea/test_gom.py`, `tests/gomea/test_engine.py`, `tests/gomea/diagnostic_objectives.py`.

**Interfaces:**
- Consumes: `LinkageModel` (Task 4). `MT19937RandGen` (`pymoses/util/rng.py`: `randint_n(n)`, `randdouble()`).
- Produces:
  ```python
  class Evaluator(Protocol):
      def check(self) -> str | None
      def evaluate(self, g: tuple[int, ...]) -> Score          # may raise BudgetExhausted
      def phenotype_key(self, g) -> Hashable | None            # None disables proven skips
      def worse(self, a: Score, b: Score) -> bool              # a strictly worse than b  (MOSES: a < b)
  @dataclass
  class Individual: genotype: tuple[int, ...]; score: Any; pkey: Hashable | None
  @dataclass
  class GomStats: trials: int; evaluated: int; accepted_improving: int; accepted_neutral: int;
                  rejected: int; noop_skips: int; phenotype_skips: int; fi_events: int; fi_elite_copies: int
  def gom(recipient: Individual, donors: Sequence[Individual], fos, ev: Evaluator,
          rng, stats: GomStats, skip_inactive: bool) -> tuple[Individual, bool, bool]  # (offspring, changed, strictly_improved)
  def forced_improvement(offspring, elite, fos, ev, rng, stats) -> Individual
  @dataclass
  class EngineParams: fos_mode='lt-b'; forced_improvement=False; skip_inactive=True;
                      max_generations=100; stall_generations=5
  @dataclass
  class EngineResult: population: list[Individual]; best: Individual; generations: int;
                      termination_reason: str; history: list[dict]
  class GomeaEngine:
      def __init__(self, domains, params: EngineParams, ev: Evaluator, rng_var, rng_link,
                   target: Callable[[Score], bool] | None = None, on_generation=None) -> None
      def run(self, population: list[Individual]) -> EngineResult
  ```
- GOM semantics (port of `GOMVariator::GOM`, no-coefficient path):
  * Clone the recipient.
  * Shuffle the FOS order per recipient.
  * For each subset, pick a donor uniformly from the **immutable generation-start snapshot** (self allowed).
  * Copy the donor's values at the subset's positions. If no position changed, record a no-op skip and move on.
  * If `skip_inactive` holds and `pkey(trial) == pkey(current)`, keep the trial with the recipient's score (a neutral, proven skip).
  * Otherwise evaluate, and accept iff `not worse(trial, current)`. On rejection, restore the genotype, score and key.
- FI (reconstruction, labelled): after GOM, let `NIS` be the number of generations without strict improvement of the population best. If `not changed or NIS > 1 + floor(log10(N))`, then for each subset in a shuffled FOS, copy the elite's values. Accept only a strict improvement and stop at the first one. If there is none, the offspring becomes an elite copy (genotype, score and key).
- Engine loop:
  * `calibrate(P0)`. If `d == 0`, stop with `no_search_fields`.
  * Each generation: `donors = P`, `F = learn(P)`, each offspring `= gom(P[i], donors, F)` (optionally FI), then `P = O`.
  * Stop on: `target_reached`, `converged` (all genotypes identical), `stagnation` (`stall_generations` without strict best improvement), `max_generations`, `global_budget`/`inner_budget`/`time_limit` (a BudgetExhausted mid-generation keeps the partially built O plus the untouched parents), `no_valid_moves` (N < 2 or FOS empty).
  * `KeyboardInterrupt` records `interrupted` and re-raises.

- [ ] **Step 1: Write failing tests** with a scripted fake evaluator (exact call counts):
  * worsening rollback;
  * neutral acceptance;
  * donors frozen (mutating offspring never changes the donor snapshot);
  * no-op skip counted without evaluation;
  * phenotype skip counted without evaluation;
  * budget exhaustion mid-generation returns a valid population with `termination_reason == 'global_budget'`;
  * `d = 0` and `d = 1` boundaries;
  * FI accepts only strict improvement and otherwise copies the elite;
  * OneMax over 30 ternary fields reaches the optimum with N=32;
  * on a planted 4 × concatenated deceptive trap (bits, block 4), LT-B or LT-MI recovers the blocks (each block is a FOS subset in ≥ 80 % of learned FOS after gen 2) and beats UNI in success count over 10 seeds.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `gom.py` and `engine.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Add the GOM engine with optional forced improvements".

### Task 6: Sampler-based initial population

**Files:** Create `pymoses/gomea/population.py` and `tests/gomea/test_population.py`.

**Interfaces:**
- Consumes: `FrozenSchema`, `sample_from_neighborhood(fs, dist, 1, out, 1, center_inst)` (draws from `randGen()`).
- Produces:
  ```python
  RadiusMixture = tuple[tuple[int | str, float], ...]     # e.g. ((1,.4),(2,.3),(4,.2),('max',.1))
  def parse_radii(text: str) -> RadiusMixture             # "1:0.4,2:0.3,4:0.2,max:0.1"
  def resolve_radii(mix, dim) -> tuple[tuple[int, float], ...]   # 'max' -> dim; r > dim -> dim; merged; weights renormalised
  @dataclass
  class InitResult: genotypes: list[tuple[int, ...]]; radii: list[int]; attempts: int;
                    duplicates_rejected: int; coverage_replacements: int; feasible_short: bool; diagnostics: dict
  def build_population(schema, n: int, mix: RadiusMixture, rng: MT19937RandGen,
                       dedupe: bool = True, attempt_cap_multiple: int = 20,
                       coverage_cap_multiple: int = 2) -> InitResult
  def field_diagnostics(genotypes, domains) -> dict   # per field: domain, observed values, entropy (bits), default frequency; unique genotypes
  ```
- Policy:
  * Index 0 is the default genotype (the exemplar).
  * For each remaining slot, pick a radius from the mixture with `rng`, then draw one sample from the MOSES sampler around the default instance.
  * Reject a duplicate if `dedupe` holds, up to `attempt_cap_multiple*n` draws. If still short, return fewer individuals and set `feasible_short=True`.
  * Coverage: for each field constant in the population (random order), make up to `coverage_cap_multiple*n` extra draws in total. Take the first sample that varies that field and is not a duplicate, and let it replace a random non-default member.
  * There is no fitness preselection and no out-of-schema value.

- [ ] **Step 1: Write failing tests:**
  * every genotype is legal;
  * index 0 is the default;
  * the radius distribution follows the weights (chi-square-free check: each component's count within ±5σ over 2000 draws on a large schema);
  * sampler distance == radius (Hamming distance to the default equals the chosen radius);
  * no duplicates when `dedupe`;
  * a tiny schema (d=2, 9 legal genotypes) with n=64 returns ≤ 9 individuals, `feasible_short` set, and terminates;
  * coverage makes a previously constant field vary when feasible;
  * the result is deterministic given the seeds of both `randGen()` and `rng`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Initialise GOMEA populations with MOSES's neighbourhood sampler".

### Task 7: The MOSES optimizer seam

**Files:** Create `pymoses/gomea/optimizer.py`, `pymoses/gomea/telemetry.py` and `tests/gomea/test_optimizer.py`.

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  ```python
  class gomea_parameters:
      pop_size=64; fos='lt-b'; forced_improvement=False; skip_inactive=True
      radii='1:0.4,2:0.3,4:0.2,max:0.1'; dedupe=True; max_generations=100; stall_generations=5
      max_evals_per_deme: int | None = None; telemetry_path: str | None = None
  class gomea_optimization(optimizer_base):
      def __init__(self, op: optim_parameters | None = None, gp: gomea_parameters | None = None)
      def __call__(self, deme, iscorer, max_evals: int, max_time: int) -> None
      last_result: dict        # termination_reason, ledger, generations, diagnostics (for tests/telemetry)
  ```
- Behaviour:
  * `schema = FrozenSchema(iscorer.get_rep())`.
  * The RNG streams `rng_init`, `rng_link` and `rng_var` are each seeded from one `randGen().randint()` draw, in that order, so a run is reproducible from `-r`.
  * Build the population, then evaluate it through the oracle. The default individual is evaluated first, and the initial evaluations count against the budget.
  * Run the engine with `target = raw score >= opt_params.terminate_if_gte`.
  * Call `oracle.finalize()` and set `optim_stats`: `nsteps` = generations, `total_steps += generations`, `total_evals += requests`, `over_budget = reason in {'global_budget','inner_budget','time_limit'}`.
  * Telemetry events go to JSONL: `invocation_start`, `init`, `generation`, `end`.
- [ ] **Step 1: Write failing tests** on a real `pa`-3 representation:
  * the deme holds exactly the evaluated trials, with `n_evals == len(deme) == ledger.requests`;
  * the budget is never exceeded (max_evals 10/50/200);
  * deterministic given `randGen().seed(s)`;
  * all termination reasons are distinct strings;
  * a contin representation raises `UnsupportedRepresentation`;
  * the population and linkage are rebuilt per call (two calls on different reps give different `schema_id`s in telemetry, and the second call's calibration does not read the first's).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with the message "Expose GOMEA as a MOSES inner optimizer".

### Task 8: CLI wiring + lifecycle integration tests (P1.3)

**Files:**
- Modify: `pymoses/moses/main/problem_params.py` (`_SUPPORTED_OPT_ALGOS` gains `"gomea"`; add options `--gomea-pop-size INT=64`, `--gomea-fos {lt-b,lt-mi,rt,uni}=lt-b`, `--gomea-fi INT=0`, `--gomea-radii STR`, `--gomea-skip-inactive INT=1`, `--gomea-max-gens INT=100`, `--gomea-stall-gens INT=5`, `--gomea-max-evals-per-deme INT=0` (0 = none), `--gomea-log PATH`), and `pymoses/moses/moses/moses_main.py` (dispatch).
- Test: `tests/gomea/test_integration.py`.

**Interfaces:** `metapop_moses_results_b` receives a `gomea_parameters`. Thread it via a new attribute on `problem_params` (e.g. `pms.gomea_params`), following how `hc_params` is threaded (grep `hc_params` in `problem_params.py`, `table_problems.py`, `demo_problems.py`), with a keyword default `gomea_params=None` so existing call sites and tests stay valid.

- [ ] **Step 1: Write failing tests:**
  * `moses -H pa -k 3 -a gomea -m 3000 -r 1` completes with ≥ 2 expansions, and a `--gomea-log` shows ≥ 2 distinct `schema_id`s with independent calibrations;
  * the same run twice gives identical stdout;
  * `-a hc` stdout on `pa -k 3 -m 2000 -r 1`, `maj -k 3 -m 1500 -r 2` and `tests/fixtures/table_data/dataset.csv -m 1000` is byte-identical to the pre-change output, captured at the base commit and stored in the test;
  * `--help` still lists every classic option unchanged (the diff of old vs new help contains only added `--gomea-*` lines);
  * candidate program, score and complexity agree between the deme entry and the printed metapopulation (re-score the printed best with the scorer);
  * a regression table problem with `-a gomea` fails with a clear `UnsupportedRepresentation` message;
  * `terminate_if_gte` reached yields `target_reached`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the wiring.
- [ ] **Step 4:** Run `pytest tests/gomea -q`, then the full `pytest -q`. Expected: everything that passed at the base still passes. Record the full-suite log in `../gomea-results/phase1/baseline/pytest_after_integration.log`.
- [ ] **Step 5:** Commit with the message "Select GOMEA with -a gomea".

### Task 9: Benchmark harness (P1.4 tooling)

**Files:**
- Create in `benchmarks/gomea/`:
  * `instrument.py`: wraps `complexity_based_scorer.__call__` to record the best-so-far `(eval_index, raw_score, penalized, complexity, t)`, identically for every arm;
  * `runner.py`: expands a manifest into runs; `multiprocessing` with `maxtasksperchild=1`; a single thread per process (`OMP_NUM_THREADS=1` etc.); a per-run wall cap via a child process timeout; writes `runs/<run_id>/{config.json,summary.json,trace.jsonl,best_programs.jsonl,events.jsonl}`; resumable (skips completed run ids); SIGINT stops cleanly;
  * `manifest.py`: validates the manifest schema and fails closed on nulls; computes the manifest hash; writes `environment.json`;
  * `fixtures.py`: captures frozen exemplars from HC runs and replays HC/GOMEA on an identical rebuilt representation under an equal local budget;
  * `analyze.py`: paired results CSV, success rates with Wilson CIs, medians/IQR, paired Wilcoxon signed-rank (exact for n ≤ 25 else normal with tie correction) + Holm, bootstrap CIs of the paired median difference, and ERT with right-censoring;
  * `svgplot.py`: line/box plots as plain SVG.
- Create: `configs/gomea/phase1/smoke.json`, `pilot.json`, `confirm.json` (written in Tasks 10–11).
- Test: `tests/gomea/test_benchmarks.py`, covering the Wilcoxon statistic against hand-computed small cases, the Holm order, ERT censoring, and manifest fail-closed.

- [ ] Steps: write the failing tests → implement → run the tests → commit with the message "Add the GOMEA benchmark harness".

### Task 10: Smoke, pilot, profiling, preregistration (P1.4 runs)

- [ ] **Smoke:** 2 seeds × {`pa -k 3`, `maj -k 5`} × {HC, LT-B} at `-m 2000`. Check the logs, termination reasons and ledger sums.
- [ ] **Profile:** use `cProfile` on one LT-B run and one HC run, and record the share of time in representation building, sampling, entropy/LT, decoding/reduction, scoring, GOM/FI and merge.
- [ ] **Frozen-fixture comparison:** capture 6 exemplars per task from HC runs, then replay HC vs UNI/RT/LT-MI/LT-B on identical fixtures with an equal local budget. Use 5 seeds.
- [ ] **Pilot:** 5 seeds; tasks `pa4, maj5, mux6` (plus `maj7`/`pa5` if the cost allows); N ∈ {64, 128, 256}; FI off/on; the end-to-end budget is set from the profile. Diagnose before changing anything.
- [ ] **Write `docs/gomea/PREREGISTRATION.md`:** the frozen N, radii, FI choice, tasks, seeds 1001–1030, budgets, primary contrasts (LT-B vs HC; LT-B vs LT-MI; LT-B vs UNI), metrics, tests and practical thresholds. Commit it **before** any confirmatory run.

### Task 11: Confirmation, analysis, findings, Phase 2 handoff (P1.5)

- [ ] Run `confirm.json` (the equal-evaluation cohort; the equal-wall-time cohort if affordable).
- [ ] Run `analyze.py` to produce tables and SVG figures under `../gomea-results/phase1/<hash>/analysis/`, and write the checksums.
- [ ] Write `docs/gomea/PHASE1-FINDINGS.md` (300–500 words, using the template, citing run IDs), `docs/gomea/PHASE1-GATES.md` (the review packets), and `docs/gomea/PHASE2-HANDOFF.md` (the exact adapter APIs, the missing continuous pieces, measured bottlenecks, and dependency choices).
- [ ] Run the full test suite and ruff, then commit. The Phase 1 completion commit is the branch head. It is not merged or pushed.

---

## Phase 2 (separate execution run; plan only — do not start without explicit approval naming the Phase 1 commit)

Branch `experiment/moses-gomea-mixed` from the approved Phase 1 commit. Each task ends at a
user gate.

### Task P2.0: Verification and source decisions (read-only + decision record)
- Verify the approved commit, a clean tree, and the Phase 1 suite green.
- Audit: `pymoses/moses/representation/build_knobs.py` contin knob building (`build_contin`, `contin_canonize`); `field_set` contin step encoding (`get_contin`/`set_contin`, `contin_stepper`); contin interpreter operators and division semantics (`pymoses/comboreduct/interpreter/`); `contin_bscore`; contin reduction.
- Look for the exact [P2] source: the paper's supplementary material, the author repos (JFKakaJFK/GOMREG and branches), and DOI search. Time-box 2 h and record URLs and commits.
- Compare [C3] (CWI `gomea`, pinned release) with [C4] (`RV-GOMEA.zip`, checksum) for AMS, AVS, FI and the distribution update. The deliverable is a table of mechanism → source file:symbol → port/bridge → parity test.
- **Decision record:** `docs/gomea/P2.0-DECISIONS.md` covering
  * the constant representation (native contin-knob adapter vs reference pool `c[0..K-1]`);
  * the scheduler conventions (`p_RV = 1 - 0.5*E_RV/E_GP`, the startup pseudocounts if unverified);
  * the IA covariance rule;
  * the FI clocks.
- Gate.

### Task P2.1: Mixed codec, scorer, activity masks, matched baselines
- Files: `pymoses/gomea/mixed_schema.py` (`MixedSchema(z_schema: FrozenSchema, K: int)`, `materialize(z, c) -> combo_tree`, `active_mask(z) -> tuple[bool, ...]`), `pymoses/gomea/mixed_oracle.py` (the recipient-specific evaluation of `(z, c)`, with caching keyed by `(schema_id, z, c)`), and the baselines `pymoses/gomea/constants/{fixed,coeff_mut,gradient,posthoc}.py`.
- Tests: exemplar reproduction, repeated references, inactive-slot invariance, atomic rollback, serialization, and cache keys that differ only in `c`.
- Linear scaling is an explicit option that is off by default. Division semantics are pinned to the interpreter's.

### Task P2.2: Stateful RV engine (non-IA)
- `pymoses/gomea/rv.py` holds `RVState(K, params)` with `.step(pop_c, masks, evaluator, budget)`. It implements Gaussian ML on the top 35 % (with the reference rounding), AMS, AVS, RV-FI (stagnation threshold 20) and a single full-K FOS, ported from [C3]/[C4] with parity fixtures.
- Numerical tests: singular covariance, N < K, identical constants, non-finite objective values, and rejected steps. Fixed-structure benchmarks: a separable quadratic, a rotated quadratic, and one nonlinear expression.

### Task P2.3: Interleaved coordinator
- `pymoses/gomea/mixed_engine.py` does subset-major GP steps against the immutable mixed snapshot, runs full RV generations, and uses the evaluation-balanced scheduler with progress guards and mixed FI. Its distinct stop reasons are: structural convergence, numerical convergence, stagnation, no active constants, and budget.
- Tests: frozen donors across RV steps, counters matching oracle calls, budget stops inside each substep, no zero-evaluation loops, checkpoint/resume, structure-only, constants-only, and mixed tasks.

### Task P2.4: Intron-aware RV
- IA moments come from per-index active top-35 % values, following the approved covariance rule, with PSD repair and logged fallbacks.
- Tests: inactive-value invariance, reactivation identity, and symmetry/PSD/finite checks. It is compared with non-IA on a synthetic masked population.

### Task P2.5: Experiments, report, freeze
- Run the six [P2] synthetic targets and the five real datasets (with checksummed loaders and fold manifests).
- The arms are HC-native, G-FIX, G-CM, G-GRAD, G-POST, G-RV and G-RV-IA. The analysis is hierarchical/correlated.
- Deliver the 300–500 word findings note and freeze the branch.

---

## Self-review notes

* Spec coverage:
  * P1.0 → Task 1 (audit already done);
  * P1.1 contracts 1–7 → Tasks 2, 3, 8 (contract 7: hc byte-identity);
  * P1.2 → Tasks 4, 5, 6 (the engine matrix is in the Task 5 tests; golden parity is Task 4);
  * P1.3 → Tasks 7, 8;
  * P1.4 → Tasks 9, 10;
  * P1.5 → Task 11;
  * Phase 2 increments → P2.0–P2.5.
* Deliberate scope limits in Phase 1 are recorded in the findings: IMS, multi-deme and immigrants are only run if the pilot justifies them; probing on/off is shared representation behaviour that GOMEA does not change.
