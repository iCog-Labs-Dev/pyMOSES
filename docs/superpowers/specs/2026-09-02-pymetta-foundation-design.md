# pyMeTTa — Foundation and First Runnable Slice

**Date:** 2026-09-02
**Status:** Approved design, pending implementation plan
**Scope of this spec:** repository foundation, fidelity conventions, validation harness, and slices S0–S6 (combo tree → boolean interpreter → table → scoring). Slices S7+ are listed as backlog input for the next spec cycle, not designed here.

---

## 1. Context

MOSES (Meta-Optimizing Semantic Evolutionary Search) is OpenCog's evolutionary program
learner: it evolves short `combo` programs that fit a dataset. The reference
implementation is ~58k lines of C++ at
`/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses` (referred to
throughout as **classic**).

pyMeTTa is a Python port of classic MOSES whose purpose is experimentation: making it
cheap to change one part of the algorithm and observe the effect on search behaviour.
That purpose only holds if the baseline is trustworthy, so fidelity to classic is the
governing constraint, not a nice-to-have.

### 1.1 Prior art surveyed, and why none of it is reused

Four sibling projects were analysed before this design:

| Project | Verdict |
|---|---|
| `MOSES/metta-moses` | MeTTa/PeTTa port, ~27k lines. Real coverage of hill climbing, boolean knob building, boolean reduct, feature selection. Not reused: pyMeTTa aligns to C++ structure, not to metta-moses's. |
| `MOSES/MOSES-MORK` | 10.8k lines Python, but a from-scratch Boolean-only re-imagining: `Instance.value` is a program *string*, no `field_set`, no bit-packed instance, PLN factor-graph EDA in place of hill climbing. |
| `MOSES/NMOSES` | ~1.1k lines MeTTa + 261 lines disconnected Python GP/EDA tutorial scripts. No knob/instance encoding, no outer loop. |
| `MOSES/elegant-normal-forms-python` | 4.6k lines; implements Holman ENF only, on a bespoke binary tree type. Code unusable (wrong tree type), **test assets valuable**. |

`grep` for `field_set|build_knobs|hill_climb|combo_tree|contin_t` returns **zero hits**
across MOSES-MORK and NMOSES, confirming neither mirrors classic's architecture.

**One asset is carried forward** (at S7, not in this spec's scope):
`elegant-normal-forms-python/Tests/` contributes 44 curated boolean expressions
(including Holman p.19 and Moshe Looks examples), a truth-table equivalence harness
(`TestHelpers.py`), and seven ENF axiom checkers (`EnfRuleCheckers.py`) — 89 tests that
pass today. These are rewritten against pyMeTTa's tree type when reduct is ported.

## 2. Decisions

These four were settled with the project owner and are not open for re-litigation
during implementation:

1. **Alignment target: classic C++ only.** Mirror the C++ file layout
   (`comboreduct/` + `moses/`). metta-moses's module organisation is explicitly not
   followed.
2. **Fidelity bar: bit-identical where possible.** Same `-r seed` should produce the
   same result as the C++ binary. Where infeasible, the divergence is documented rather
   than silently absorbed. Section 6 shows this is achievable for every RNG call MOSES
   makes.
3. **Scope of this cycle: foundation + first runnable slice** (S0–S6).
4. **Git: local repository only, no remote.** Sole author `Yab <yabuman46@gmail.com>`
   on every commit. No co-author or generated-by trailers of any kind.

## 3. Non-goals

- Any MeTTa/Hyperon integration. pyMeTTa has no `hyperon` dependency.
- Performance parity with C++. Correctness first; a parallel-scoring pass is S-late.
- Porting classic's dead or legacy paths: `#ifdef HAVE_MPI` (`mpi_moses.cc` — MPI
  detection is commented out of classic's CMake, so it compiles to nothing),
  `#ifdef EXEMPLAR_INST_IS_UNDEAD`, `#ifdef LATER`, `moses/pleasure/` (does not
  compile), `ant_combo_vocabulary/`, and the `action_result_type` / `ann_type` knob
  branches.
- Porting `problem-params.cc` line-by-line. It is 2407 lines of
  `boost::program_options` boilerplate; only the option *names and defaults* are
  reproduced, via `argparse`.

## 4. Repository layout

Mirrors classic's source tree so that a C++ path maps to a Python path by inspection.

```
pyMeTTa/
  pymetta/
    util/
      tree.py                 # cogutil opencog/util/tree.h  -> tree<T>
      rng.py                  # cogutil mt19937ar.cc         -> MT19937RandGen
    comboreduct/
      combo/                  vertex.py argument.py common_def.py iostream_combo.py
      interpreter/            interpreter.py eval.py
      table/                  table.py table_io.py
      type_checker/           type_tree.py type_tree_def.py
      reduct/                 reduct.py meta_rules.py general_rules.py
                              logical_rules.py flat_normal_form.py logical_reduction.py
    moses/
      representation/         field_set.py instance.py instance_set.py
                              scored_instance.py knobs.py knob_mapper.py
                              build_knobs.py representation.py instance_scorer.py
      optimization/           optimization.py hill_climbing.py
      moses/                  types.py complexity.py neighborhood_sampling.py
                              local_moses.py moses_main.py moses_params.py
      scoring/                scoring_base.py bscores.py behave_cscore.py
      metapopulation/         metapopulation.py merging.py metapop_params.py
      deme/                   deme_expander.py deme_params.py
      main/                   moses_exec.py problem_params.py table_problems.py
  tests/
    <mirrors pymetta/ 1:1>    one test module per source module
    oracle/                   differential tests against the C++ binaries
    fixtures/                 golden data captured from C++, committed
  tools/
    cpp_probe/                C++ probes linked against the real MOSES libraries
    capture_golden.py         runs probes, writes tests/fixtures/
    check_alignment.py        validates breadcrumbs, generates docs/ALIGNMENT.md
  docs/
    ALIGNMENT.md              generated: C++ <-> Python coverage and divergences
    superpowers/specs/        design documents
  pyproject.toml
  README.md
```

Note two path flattenings, both deliberate: classic's `moses/comboreduct/...` and
`moses/moses/...` lose their redundant top `moses/` segment, and `cogutil`'s two
MOSES-relevant headers are collected under `pymetta/util/`. Every other directory name
matches classic exactly.

## 5. Fidelity conventions

### 5.1 Identifier naming

**C++ identifiers are preserved verbatim**, including class names. `field_set` stays
`field_set`, not `FieldSet`; `build_logical` stays `build_logical`; `combo_tree` stays
`combo_tree`.

This is a deliberate PEP 8 deviation. The justification is that the entire value of
this port rests on a reader being able to hold the C++ and the Python side by side and
diff them by eye; renaming to satisfy a style guide destroys that for no functional
gain. `ruff`/`pycodestyle` naming rules (N801, N802, N803, N806) are disabled
repo-wide in `pyproject.toml` with a comment pointing at this section.

### 5.2 Provenance breadcrumbs

Every ported symbol records where it came from, on the first line of its docstring:

```python
def build_logical(self, subtree, it):
    """C++: moses/moses/representation/build_knobs.cc:602 build_knobs::build_logical

    Decorate a logical subtree with knobs.
    """
```

Format: `C++: <path relative to a source root>:<line> <C++ qualified symbol>`.

**Two source roots, not one** (amended 2026-09-03, during Task 1). `pymetta/util/`
ports CogUtil, not Moses, so a breadcrumb path resolves against either:

| Root | Path prefix | Example |
|---|---|---|
| `$CLASSIC` — the Moses checkout | `moses/` | `moses/moses/representation/build_knobs.cc:602` |
| `$COGUTIL` — the CogUtil checkout | `opencog/` | `opencog/util/mt19937ar.cc:66` |

The prefixes are disjoint, so there is no ambiguity about which root a path
belongs to. `check_alignment.py` tries each root in turn and reports every root
it tried when a path resolves under none. The original single-root wording was
written before it was noticed that `tree.h` and `mt19937ar.cc` — two of the
first three files ported — live in CogUtil.

### 5.3 Divergence markers

Anywhere the Python cannot or should not mirror the C++, it says so inline:

```python
# DIVERGENCE(C++ moses/comboreduct/table/table_io.cc:382): classic parses rows with
# OMP_ALGO::for_each; we parse serially. No observable difference — parse order does
# not affect the resulting ITable.
```

Format: `DIVERGENCE(C++ <path>:<line>): <reason and observable impact>`.

A divergence is legitimate only when it has **no observable effect on results**, or
when it is recorded as a known, accepted difference. "It was easier" is not a reason;
such cases must instead be raised for a decision.

### 5.4 Enforcement

`tools/check_alignment.py` runs in CI and:

1. Parses every breadcrumb and every divergence marker.
2. Verifies the referenced C++ file exists and the named symbol occurs within ±25 lines
   of the cited line (tolerating drift, catching rot). Failure is a CI error.
3. Emits `docs/ALIGNMENT.md`: a table of every C++ symbol in the in-scope files,
   marked ported / not-ported / diverged, with a coverage percentage per module.

This converts "word for word" from an intention into a build-enforced invariant, and it
keeps working as classic moves.

## 6. RNG — bit-identical reproduction (verified)

This is the foundation of the fidelity bar, so it was settled empirically before this
design was written rather than assumed.

### 6.1 What classic actually does

`RandGen : public std::mt19937` (`CogUtil/opencog/util/RandGen.h:49`) — the generator is
literally `std::mt19937`. `randGen()` returns a thread-local singleton seeded `0`,
reseeded from `-r <seed>`. From `CogUtil/opencog/util/mt19937ar.cc:44-88`:

| Method | Definition |
|---|---|
| `randint()` | `std::uniform_int_distribution<int>()(*this)` — default range `[0, INT_MAX]` |
| `randint(n)` | `n == 0 ? 0 : randint() % n` |
| `randfloat()` | `operator()() / (float)max()` |
| `randdouble()` | `operator()() / (double)max()` |
| `randbool()` | `randint() % 2 == 0` |
| `randdouble_one_excluded()` | `std::uniform_real_distribution<double>()(*this)` |
| `rand_discrete(w)` | `std::discrete_distribution<int>(...)(*this)` |

### 6.2 Call-site audit

Counted across `moses/moses` and `moses/comboreduct`:

| Call | Sites | Status |
|---|---|---|
| `randint(...)` | 17 | reproducible (6.3) |
| `randdouble()` | 14 | reproducible — raw draw / `max()` |
| `randbool()` | 9 | reproducible — derived from `randint()` |
| `randfloat()` | 3 | P2 only — all three implement the combo `rand` builtin (`eval.cc:250`, `interpreter.cc:195`, and the `tree_gen` side binary), unreachable on the boolean table path |
| `randdouble_one_excluded()` | **0** | not used |
| `rand_discrete()` | **0** | not used |

The two calls whose algorithms are implementation-defined by the C++ standard are never
called. This eliminates the one substantial risk to the bit-identity goal.

### 6.3 Why `randint()` is reproducible

`std::uniform_int_distribution<int>`'s algorithm is implementation-defined in general.
But for this specific instantiation it collapses to a shift:

- range `[0, INT_MAX]` gives `__uerange = 2^31`; the generator gives `__uerngrange = 2^32`
- `scaling = __uerngrange / __uerange = 2`
- `past = __uerange * scaling = 2^32`, and a raw draw is always `< 2^32`, so **the
  rejection branch can never be taken**
- result is `raw / scaling`, i.e. `raw >> 1`, consuming exactly one draw

So `randint() == mt() >> 1` deterministically, independent of libstdc++ version.

### 6.4 Verification performed

A C++ probe (`std::mt19937` + `std::uniform_int_distribution<int>`, built with the local
g++ 16.2.1) was diffed against a 30-line Python MT19937 using the classic `init_genrand`
single-seed initialisation:

- **12,000 `randint()` draws across seeds {0, 1, 42, 7, 12345, 4294967295}: bit-identical.**
- Raw 32-bit stream, `randint(n) % 7`, and `randdouble()` also matched exactly.

**Critical implementation note:** Python's `random` module cannot be used. `random.seed(int)`
routes through MT19937's `init_by_array` seeding, whereas C++ `std::mt19937::seed(unsigned)`
uses `init_genrand`. The streams diverge from the first draw. `pymetta/util/rng.py` must
implement MT19937 directly.

`randfloat()` carries one deferred caveat: it divides by `(float)max()`, producing a
float32. Python has no float32, so exact reproduction requires
`struct.unpack('f', struct.pack('f', x))[0]`. Deferred with the `rand` builtin to P2.

## 7. Validation architecture

Three tiers, each catching what the others cannot.

### Tier 1 — ported unit tests

Classic's `.cxxtest` files transliterated to `pytest`, preserving test names so a
failure maps back to its C++ original. In scope for this cycle: `vertexUTest`,
`iostream_comboUTest`, `interpreterUTest`, `tableUTest`, `scoringUTest`,
`type_treeUTest`.

Caveat inherited from classic: **many of its tests are commented out** of
`tests/moses/CMakeLists.txt` (`representationUTest`, `weightedUTest`, `mixedUTest`,
`irisUTest`, `selectionUTest`, `dynFeatSelUTest`, `diversityUTest`) and
`tests/comboreduct/reduct/CMakeLists.txt` (`ActionReductUTest`,
`PerceptionReductUTest`). A commented-out C++ test is *not* evidence of correctness on
either side; where one covers in-scope behaviour it is ported anyway and treated as new
coverage.

### Tier 2 — C++ oracle (differential testing)

The primary mechanism. Small C++ probe programs, built against the local toolchain and
linked to the real MOSES libraries, dump internal state as JSON; Python must reproduce
it exactly.

| Probe | Slice | Dumps |
|---|---|---|
| `rng_probe` | S2 | draw sequences for `randint`/`randint(n)`/`randdouble`/raw, over a seed matrix |
| `tree_probe` | S1 | tree state after each splice op (`insert_above`, `flatten`, `wrap`, `move_after`, `reparent`) |
| `table_probe` | S5 | `ITable` / `OTable` / `CTable` contents after `loadTable` + `compressed()` |
| `score_probe` | S6 | behavioral score vector, complexity, and composite score for a (tree, CTable) pair |
| `combo-fmt-converter` | S3 | **already built** — combo tree parse/print round-trip |
| `eval-table` | S4 | **already built** — evaluates a combo tree over a table |
| `eval-candidate` | S6 | **already built** — scores candidate trees |
| `test_reduct` | S7 | **already exists** in classic at `tests/comboreduct/reduct/test_reduct.cc` |
| `neighborhood_test` | S9 | **already exists** in classic at `tests/moses/neighborhood_test.cc` |

Four of the nine already exist, three of them as the project owner's own additions to
classic.

Golden outputs are captured once into `tests/fixtures/` and committed, so the suite runs
with no C++ toolchain present. `tools/capture_golden.py --regenerate` re-captures when
classic changes; an opt-in `--live` mode rebuilds the probes and diffs against them
directly, for use when touching a ported algorithm.

**Building the probes** uses classic's `scripts/rebuild-moses-local.sh`, which forces
`RelWithDebInfo` and strips LTO (classic's `OpenCogGccOptions.cmake` injects `-flto=auto`
into `Release`, which breaks incremental local builds). Probes link against
`$HOME/opt/moses-legacy`. The probe build is never invoked by the default test run.

### Tier 3 — trajectory diff

From S9 onward (outside this spec, specified here so the harness is built to accommodate
it): same seed, same flags, diff the full run — per-generation best score, metapopulation
contents, final program — against the C++ binary. Given section 6, this is expected to
be an exact match, and any divergence is a bug rather than noise.

## 8. Dependencies

**Runtime: Python standard library only.**

- Python floats are IEEE-754 doubles, identical to C++ `contin_t`; introducing numpy
  adds dtype and float-semantics risk against the bit-identity goal for no benefit.
- `field_set`'s bit-packing needs its *semantic* layout reproduced (field ordering,
  spec sort order) but not its *physical* packing: a `list[int]` holding one value per
  raw field is behaviourally equivalent, and `hamming_distance` is field-wise anyway.
- A zero-dependency core is directly in service of the stated purpose — an
  experimentation testbed people can read and modify.

**Development:** `pytest` (9.1.0 present). **Python 3.14** (3.14.7 present); `match`
statements are used for `vertex` dispatch, mirroring `boost::apply_visitor`.

numpy remains available later as an optional vectorised scoring fast path, behind a
flag and validated against the pure-Python path.

## 9. Slices in this cycle

Each slice is one commit, and must be green before the next begins.

### S0 — Scaffold
`pyproject.toml`, pytest config, ruff config with the naming-rule exemptions from 5.1,
`.gitignore`, `README.md`, `tools/check_alignment.py`, initial `docs/ALIGNMENT.md`, and
a CI entry point runnable as a single command.
**Exit:** `pytest` runs green on an empty suite; `check_alignment.py` runs clean.

### S1 — `pymetta/util/tree.py`
Port of `tree<T>` (cogutil `opencog/util/tree.h`, Kasper Peeters' `tree.hh` v2.3). Nodes
are `parent / first_child / last_child / prev_sibling / next_sibling / data`. Built as a
real linked object graph, not a flat array — `insert_above` / `flatten` / `wrap` /
`reparent` are pointer-splicing operations that are natural on links and awkward on
arrays.

API surface is scoped by measured call-site frequency in classic, not by porting the
whole 3420-line header: `begin`/`end` (1841 uses), `size` (462), `erase` (235), `empty`
(194), `append_child` (160), `insert` (128), `child` (98), `number_of_children` (91),
`erase_children` (68), `flatten` (60), `is_valid` (49), `insert_above` (42),
`next_sibling` (40), `replace` (37), `wrap` / `equal_subtree` (27 each), `move_after` /
`count` (18), `insert_subtree_after` (15), `reparent` / `insert_subtree` (13),
`move_ontop` (11), `prepend_child` (10), `begin_leaf` / `end_leaf` (9), plus
`move_before`, `insert_after`, `append_children`, `begin_post` / `end_post`,
`previous_sibling`, `number_of_siblings`, `sort`, `max_depth`.

Python object identity replaces C++'s `obj_ptr_cmp<It>` (`CogUtil/opencog/util/hashing.h:81`),
which classic uses to key knobs by tree position in `knob_mapper.h:69-74` — in Python the
node object is directly usable as a dict key, which is simpler than the C++.

**Hazard carried over, not solved by the language change:** classic's `build_knobs`
mutates the exemplar while iterating it, and sibling loops can observe siblings replaced
by `null_vertex` sentinels mid-iteration (classic commit `98ed6fe0f` fixed one such
premature `break`). A Python port over live object references has exactly the same
hazard. Iterators must be re-derived after any `insert_above` / `append_child` /
`move_after`, in the same places classic does it.

**Exit:** `tree_probe` differential tests pass for every splice operation.

### S2 — `pymetta/util/rng.py`
MT19937 with `init_genrand` seeding, plus the `RandGen` surface: `randint()`,
`randint(n)`, `randdouble()`, `randbool()`. Per section 6, `randint()` is `raw >> 1`.
`randfloat()`, `randdouble_one_excluded()`, and `rand_discrete()` raise
`NotImplementedError` with a pointer to section 6.2 — they are unreachable on the
in-scope path, and a stub that silently returns a plausible wrong number would be worse
than a crash.

**Exit:** `rng_probe` differential test passes over a seed matrix of at least
{0, 1, 42, 7, 12345, 2^32-1} × 2000 draws for each supported method. **This slice is
the gate on the project's bit-identity guarantee** — if it fails, that guarantee is
withdrawn and section 6 is rewritten before any further slice proceeds.

### S3 — `comboreduct/combo/`
`vertex` as a tagged union over the subset actually reachable: `builtin` (the enum at
`vertex.h:69-105`), `argument` (`argument.h:44-52`), and `contin_t`. The other
ten variant members (`action`, `perception`, `message`, `procedure_call`, `ann_type`,
`definite_object`, `indefinite_object`, `action_symbol`, `wild_card`, `enum_t`) are
declared for completeness but unimplemented. Plus `combo_tree = tree[vertex]` and
`iostream_combo` printing and parsing.

Two `argument` details are load-bearing and must be ported exactly, because reduct
depends on both:

- **Sign encodes negation.** `negate()` is `idx = -idx`; `is_negated()` is `idx < 0`
  (`argument.h:54-59`). Indices are 1-based and `idx == 0` is asserted impossible.
- **`operator<` orders by _absolute_ value**, via `absolute_value_order<int>`
  (`argument.h:62-65`). So `$1` and `!$1` are adjacent under this ordering rather than
  separated by every other argument. Anywhere arguments are sorted — notably the clause
  sets in `flat_normal_form` — a natural Python `__lt__` on the signed integer would
  silently produce a different order and therefore different reductions. Port the
  comparator, do not infer it.
**Exit:** round-trip parse/print matches `combo-fmt-converter` on a corpus including
classic's own `testTrees.sh` fixtures.

### S4 — `comboreduct/interpreter/`
`boolean_interpreter` (`interpreter.cc:47`) and the `interpreter_visitor` dispatcher
(which lives in `table.h:304`, not in the interpreter directory).
**Exit:** ported `interpreterUTest` passes; outputs match `eval-table` over a corpus.

### S5 — `comboreduct/table/`
`loadTable` (`table_io.cc:1296`) with CSV parsing and per-column type inference;
`ITable`, `OTable`, `Table`; and `Table::compressed()` (`table.cc:501`) producing the
`CTable` (`table.h:550`) — the row-deduplicated, counted representation that makes
scoring O(distinct rows).
**Exit:** ported `tableUTest` passes; `table_probe` differential test matches on
classic's own fixtures (`tests/comboreduct/table/*.csv`) plus the project owner's
root-level `trainingBoolean.dat` and `trainingBooleanXor.dat`.

### S6 — `moses/scoring/` + `moses/moses/types.py` + `complexity.py`
`bscore_base` (`scoring_base.h`), `ctruth_table_bscore` (`bscores.cc:276-296`),
`behave_cscore` (`behave_cscore.cc:51`), `composite_score` (`types.h`), and
`tree_complexity` (`complexity.cc`). The `behave_cscore` cache is an optimisation, not
semantics — a plain `dict` or nothing at all.
**Exit:** for a corpus of (CSV, combo tree) pairs, Python and C++ produce **identical**
behavioral score vectors, complexity values, and composite scores, verified by
`score_probe` and `eval-candidate`.

### Cycle exit criterion

`pytest` green; RNG bit-identical across the seed matrix; and identical scoring against
C++ for the (CSV, tree) corpus. At that point every deterministic input to the search
loop is proven correct, and S7+ can build on it.

## 10. Execution workflow

Per slice, four steps:

1. **Porting agent** — implements the slice from the C++ source, given the exact C++
   files and line ranges, the conventions in section 5, and the oracle contract.
2. **Test agent** — independently ports the corresponding `cxxtest` and writes the
   probe plus fixture capture, working **from the C++ tests and C++ source only**, never
   from the Python implementation. This independence is the point: an agent that reads
   the implementation first will write tests that ratify it.
3. **Verification** — the full suite plus `check_alignment.py` are run, and the real
   output is recorded. No slice is described as complete on the basis of expected output.
4. **Review agent** — fresh context, given the C++ source and the Python diff, hunting
   the specific failure modes that matter for a transliteration: dropped conditional
   branches, reordered operations, wrong iteration order, off-by-one in loop bounds,
   silently-skipped edge cases, and divergence markers that hide a real behavioural
   difference. Findings are fixed and the suite re-run before the commit.

S3, S4 and S5 are largely independent once S1 lands and can run in parallel; S0→S1→S2
and S5→S6 are strictly ordered.

**Commits.** One per slice, message `port(<module>): <summary>`, body listing the C++
files mirrored and any `DIVERGENCE` markers introduced. Author `Yab
<yabuman46@gmail.com>` and nobody else — no co-author, generated-by, or session
trailers. The spec commit precedes S0.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Reduct (S7) is the highest silent-divergence surface — classic's rule set fires to a fixpoint in a specific order, and a "morally equivalent" engine converges differently | `test_reduct` already exists as an oracle; port `meta_rules` (the traversal combinators) before any individual rule, since they control application order |
| `build_knobs` (S8) mutates the tree it iterates | Documented in 9/S1; port the iterator re-derivation points explicitly, one-to-one |
| `field_set` field ordering must match classic's `std::multiset` ordering (disc descending by multiplicity, contin descending by depth), or `it_disc_idx`/`it_contin_idx` break silently | Port `disc_spec::operator<` and `contin_spec::operator<` verbatim; assert order in tests rather than trusting insertion order |
| Breadcrumb line numbers rot as classic moves | `check_alignment.py` tolerates ±25 lines and fails CI beyond that |
| An agent "completes" a slice that does not actually pass | Step 3 of section 10 requires recorded real output; step 4 is a separate agent with no stake in the implementation |

## 12. Backlog — input for the next spec cycle

Not designed here. Listed so the next planning cycle has its input, in dependency order.

| Slice | Content | Oracle |
|---|---|---|
| S7 | `reduct/`: `reduct.py`, `meta_rules` (combinators first), `general_rules`, `flat_normal_form`, `logical_rules`, `logical_reduction` | `test_reduct`; plus the 44 expressions, truth-table harness and 7 ENF axiom checkers lifted from `elegant-normal-forms-python` |
| S8 | `representation/`: `field_set`, `instance`, `instance_set`, `knobs` (`logical_subtree_knob` first), `knob_mapper`, `build_knobs` (`logical_canonize` → `build_logical` → `logical_cleanup`), `representation`, `instance_scorer` | knob-count and `get_candidate` round-trip probe |
| S9 | `optimization/hill_climbing.py` (serial, no crossover) + `moses/neighborhood_sampling.py` | `neighborhood_test` (exists) |
| S10 | `deme/deme_expander.py` (single-deme path, no feature selection) + `metapopulation/` + `merging.py` | trajectory diff |
| S11 | `local_moses.py`, `moses_main.py` — **first end-to-end `moses -i data.csv -u y`** | full trajectory diff vs C++ |
| S12 | Hill-climbing crossover (on by default in classic; read `doc/cross_top_two_explanation.md` and `doc/cross_top_three_explanation.md` first) | trajectory diff |
| S13 | CLI parity (`argparse`) + `metapop_printer` output formatting | CLI-level output diff |
| S14 | contin/mixed: `contin_rules`, `mixed_rules`, `contin_spec` fields, `contin_bscore` | `test_reduct` at contin/mixed levels |

Then the P2 tail, each its own vertical slice: feature selection (`--enable-fs`),
diversity, domination, ensembles/boosting, subsampling, the other optimizers
(`star-anneal`, `univariate`, `particle-swarm`), and the precision/recall/BEP/F1/select
scorers.
