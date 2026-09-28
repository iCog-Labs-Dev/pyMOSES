# Source and reuse register (Phase 1)

All upstream sources were read from local checkouts at pinned commits. "Port" means a
narrow, function-level transliteration into pure Python. Each ported function's first
docstring line names its upstream file:line and symbol, and the golden fixtures prove it
behaves the same (Task 4). No upstream file is vendored, and no native bridge is built
(the reasons are in `P1.0-AUDIT.md` §8).

| Requirement | Source | Commit / checksum | File : symbol | Licence | Choice | Modifications | Parity test | Unresolved / disclosed |
|---|---|---|---|---|---|---|---|---|
| Per-field and joint entropy | [C1] github.com/marcovirgolin/GP-GOMEA | `750220c23d599e515ca337dcca617ae5faab8b4d` | `src/GOMEA/GOMEAFOS.cpp` : `GOMEAFOS::GenerateLinkageTreeFOS` (frequency loops) | Apache-2.0 | port | GP node symbols are replaced by MOSES logical-field codes. Constant binning is dropped from this function; the opt-in `--gomea-contin copy` ports it separately (next row). Entropy terms are summed in per-field code order rather than upstream's global symbol-encoding order, so results can differ from a real GP-GOMEA run in the last floating-point digit. The golden harness uses the same order as the port | `tests/gomea/test_golden_parity.py` | none |
| Constants in linkage learning: all-const, no-const, bin-const (Sec 6 of [P1]; `--gomea-contin copy`, `--gomea-contin-stats`) | [C1] and [P1] Sec 6 | `750220c` | `src/GOMEA/GOMEAFOS.cpp` : `GOMEAFOS::GenerateLinkageTreeFOS` (constant binning, lines 316-364) → `pymoses/gomea/stats.py` : `contin_symbols` | Apache-2.0 | port (bin) and reconstruction from [P1] (all, none) | a MOSES contin knob is one copy-only gene (never mutated), as GP-GOMEA's constants are only mixed by GOM. bin: γ = 100; the first γ distinct constants in individual-then-gene order keep their symbol, later ones take the nearest retained constant's symbol (ties to the lower). none: every constant is one symbol, and such genes join the FOS as singletons outside the tree (upstream has no separate no-const code at this commit). | `tests/gomea/test_contin.py` | upstream's nearest search starts `least_dist` at a constant's value instead of its distance, and stores the result via `std::string = double` (`operator=(char)`), which collapses binned constants into one-character symbols. Both are disclosed and not reproduced: the paper's rule is implemented |
| Initialization-biased similarity (Eq. 2 of [P1]) | [C1] | same | same function, `MI_distribution_adjustments` (computed once) | Apache-2.0 | port | `H0` comes from the actual local initial population. Fields with `H0 = 0` are excluded from linkage (upstream would divide by zero) | golden + identity test | the upstream divide-by-zero is disclosed, not reproduced |
| Ordinary-MI similarity | [C1] | same | the dead `if (false)` branch of the same function | Apache-2.0 | port of the dead branch | enabled as the LT-MI control arm | golden | this branch is not live upstream at this commit |
| UPGMA / RNN linkage tree | [C1] | same | `BuildLinkageTreeFromSimilarityMatrix`, `DetermineNearestNeighbour` | Apache-2.0 | port | `random_shuffle` is replaced by an injected order drawn from a seeded MT19937 stream. The chain start `(int)randu()*len == 0` is kept | golden | none |
| Root exclusion | [C1] | same | `FOS.pop_back()` | Apache-2.0 | port | for d = 1 the singleton is kept (an adaptation, see the plan) | unit test | adaptation |
| Univariate FOS | [C1] | same | `GenerateUnivariateFOS` | Apache-2.0 | port | none (`fos_no_root_swap` is off) | unit | none |
| Random-tree FOS | [C1] | same | `GenerateRandomTreeFOS` | Apache-2.0 | port | armadillo `randu()` is replaced by the seeded MT19937 `randdouble()`. The index is capped at `size-1`, because `randdouble()` can return exactly 1.0 and `randu()` cannot | unit | none |
| GOM | [C1] | same | `src/GOMEA/GOMVariator.cpp` : `GOMVariator::GOM` (the path without coefficient mutation) | Apache-2.0 | port | MOSES composite comparator (accept iff not worse). "Active node" is replaced by a proof of an equal unreduced decoded tree. The donor snapshot is immutable | scripted-evaluator tests | none |
| Generation loop / convergence | [C1] | same | `src/GOMEA/GOMEAGenerationHandler.cpp` : `PerformGeneration`, `CheckPopulationConverged` | Apache-2.0 | adapt | no re-initialization on convergence (control returns to MOSES). Elitism is implicit because GOM never accepts a worse individual | engine tests | adaptation |
| Efficient GOM variant | [C2] github.com/marcovirgolin/gpg | `65eb040aaf3d316257ea373897548a56db744a35` (branch `pybind`) | `src/variation.hpp` : `efficient_gom` | MIT | reference only | its tournament "FI variant" is **not** inherited | n/a | none |
| Forced improvements | [P2] text; LT-GOMEA convention | n/a | neither [C1] nor [C2] implements it | n/a | **reconstruction** | trigger: no change, or no-improvement stretch > 1 + ⌊log10 N⌋. Elite-donor GOM accepts only a strict improvement and stops at the first one, else the elite is copied | unit | a labelled reconstruction, off in the core arms |
| MOSES semantics (sampler, scorer, merge) | pyMoses itself (a port of classic MOSES [C6], `../OpenCog/Moses@5a5bf75`) | `0410881` | unchanged modules | Apache-2.0 OR AGPL-3.0 | reuse as is | none | existing suite (7898 passed) | none |
| Experiment utilities (Wilcoxon, Holm, bootstrap, ERT, SVG) | new code | n/a | `benchmarks/gomea/*` | project licence | new | n/a | `tests/gomea/test_benchmarks.py` | [C5] SRBench is not needed for Boolean tasks |

Licence compatibility: Apache-2.0 ([C1]) and MIT ([C2], reference only) can both be
incorporated into a project distributed as `Apache-2.0 OR AGPL-3.0-only`. The ported
functions carry an attribution line pointing to [C1]'s Apache-2.0 licence.

## Disclosed behaviours (from the final whole-branch review)

* **Generation-0 LT-B tree.** On the calibration population itself, H = H0, so every
  off-diagonal biased similarity is 0 up to floating-point residue (measured values
  {0, 2.2e-16}). The first LT-B tree is therefore decided by 1-ulp rounding noise and the
  seeded order, not by linkage. This matches upstream: the golden fixtures show bit parity.
  It is inherent to Eq. 2 of [P1] at g = 0.
* **The `max` initial-population radius** equals the representation dimension, so it can
  exceed hill climbing's `-D/--max-dist` search distance. The GOMEA initial population can
  therefore contain instances HC's neighbourhood search would never visit. This is a
  design difference between the arms, not a bug.
* **The score cache is off in both arms for the demo problems** (see the correction in
  `P1.0-AUDIT.md`). `EvaluationLedger.duplicate_requests` reports how many GOMEA requests
  re-evaluated an already-seen genotype.
