# Remaining work to a complete port

Written 2026-09-08, after S12 merged. This is the durable roadmap: it survives
context loss and says what "done" means, in what order, and why.

**Scope reminder.** 83 files / 11,048 lines are **EXCLUDED by decision**, not
deferred: legacy embodiment, dead `#ifdef` code, C++-only tools, and the enum,
ANN, list/fold, object, message and procedure-call domains. The port's stated
domains are Boolean, continuous and action. "Complete" therefore means every
non-excluded file ported, with its named gaps closed.

## Status at the time of writing

| Status | Files | Lines |
|---|---:|---:|
| PORTED | 17 | 4,901 |
| PARTIAL | 67 | 25,975 |
| PLANNED: S13 | 11 | 4,037 |
| PLANNED: S14 | 9 | 4,057 |
| PLANNED: P2 | 59 | 12,281 |
| EXCLUDED | 83 | 11,048 |

## Order of work, and the reasoning

### 1. S13 — CLI parity (in progress)
Plan: `docs/superpowers/plans/2026-09-08-pymetta-s13-cli.md`.
Governing rule: every flag parses and every default matches, but an option whose
subsystem is unported **raises**. A silently-ignored flag is worse than a
failing one.

### 2. S14 — continuous and mixed
Plan: `docs/superpowers/plans/2026-09-08-pymetta-s14-contin.md`.
**Its file list understates it**: six earlier slices deferred their continuous
halves here (interpreter, `build_contin`, `contin_knob`, knob-mapper entries,
three RNG sampling helpers, `contin_bscore`) and none appear in the list. Task 0
is CARRY-FORWARD §2.5, a hard prerequisite.

### 3. P2 — the deferred tail, 59 files / 12,281 lines

Sequenced by what unblocks the most, not by size:

**3a. Scorers** — `precision_bscore` (838+235), `discriminating_bscore`
(698+238), `select_bscore` (348+86), `ss_bscore` (162+83), `time_dispersion`
(118+109). ~2,900 lines.
Unblocks `-H pre`, `prerec`, `recall`, `bep`, `f_one`, `select` — six of the
nine problems S13 stubs as raising — and four `MOSESUTest` cases
(`test_kl`, `test_multi_kl`, `test_precision`,
`test_negative_predictive_value`). Independent of S13/S14, so it can run in
parallel with them.

**3b. `demo-problems.cc` + `logical_bscore`** — 516 + ~60 lines.
Unblocks the seven `MOSESUTest` cases blocked since S11 (`-Hpa`, `-Hmux`,
`-Hcp`). Small and high-leverage: it converts most of classic's own end-to-end
suite into a usable oracle.

**3c. Metapopulation subsystems** — `diversity.cc` (307), `domination.cc` (309),
`subsample.cc` (488), `ensemble.{h,cc}` (438+98). ~1,640 lines.
All four are already ported **as gates that raise**, so this is filling in
bodies behind existing, tested boundaries — the cheapest kind of P2 work.

**3d. Feature selection** — `feature_selector.{h,cc}` (477+233),
`moses/feature-selection/` (algos + scorers + main). ~2,400 lines.
Reached via `--enable-fs`. Self-contained subsystem.

**3e. The other optimizers** — `particle-swarm` (380+356), `star-anneal`
(165+160), `univariate` (104+80), plus `eda/`. ~1,600 lines.
Selected by a CLI flag; `deme_expander` already injects `optimizer_base`, so the
seam exists. Note `test_ps_2_parity` **fails in classic itself** — do not chase
it.

**3f. The remainder** — `partial.cc` (352), `distributed_moses.cc` (475),
`similarity.cc` (257), and the residual vertex variants.
`distributed_moses` shells out to `ssh`/`pgrep`; consider whether it belongs in
scope at all before porting it.

### 4. Close the PARTIAL gaps

67 files carry a named gap. `docs/PARTIAL-AUDIT.md` classifies them: **55 are
roadmap** (scheduled to a slice above, or excluded by scope) and **7 are genuine
unscheduled gaps** — `exception.{cc,h}`, `instance_scorer.h`, `bscores.{cc,h}`,
`scoring_base.{cc,h}`. Most of the 55 close automatically as their slice lands.

Known genuine debt not tied to a slice:
- **Error types are inconsistent** — `OC_ASSERT(false)` maps to six Python types;
  "column not found" is `IndexError` in `table.py` and `ValueError` in
  `table_io.py`. Needs one C++-grounded mapping decision, then a wide sweep.
- **Corpus gaps** — no argument above `$3`, no arity ≥ 4.
- **`tree.json` has no shape guard**, and `test_tree.py`'s 33 oracle tests do not
  use `parametrize`, so `empty_parameter_set_mark` does not cover them.
- **No manifest records a compiler/stdlib version** (CARRY-FORWARD §5.1). S12's
  fixtures depend on libstdc++'s sort permutations, so a toolchain upgrade could
  leave them silently stale.

## Standing rules for whoever continues this

- **Closeout is part of the slice.** S9, S10 and S11 were each merged with their
  coverage rows still saying PLANNED. `coverage_totals.py` now catches that; do
  not rely on it.
- **A deliberately-unasserted field is where a real divergence hides.** The
  13-vs-17 deme size was waived twice as benign tie-order and turned out to be
  the stable-sort bug (§5.2).
- **A marker must state what C++ actually does.** Four markers in this project
  have made confident false claims; that is its most-repeated defect.
- **Verify the adjacent dimension.** Three times an oracle or fix was confirmed
  along one axis and wrong along another: `metapopulationUTest` built and passed
  but did not cover the slice; nine `MOSESUTest` cases used the right optimizer
  but the wrong problem registry; a `setattr` composition preserved
  introspection but not import order.
- **A large matching corpus is not coverage.** S9's oracle matched 4,500
  instances while walking one control-flow path.
