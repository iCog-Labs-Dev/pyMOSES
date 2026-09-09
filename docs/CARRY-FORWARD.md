# Carry-forward from the S0–S6 foundation cycle

Written 2026-09-04, at the close of the foundation cycle (commits `569e801..4ecba1f`).

This is what the next cycle needs to know and would otherwise have to rediscover.
Everything here was found by building or reviewing the foundation; none of it is
speculation. Items are grouped by when they bite.

---

## 1. Must be settled before S7 (reduct) and S8 (knob building)

### 1.1 `Table.__init__` aliases where C++ copies (resolved in S7 Task 0)

`table.py:574` stores `otable`/`itable` by reference; `table.cc:485` copies both.
This is the only genuine divergence in what was originally logged as a broad
"aliasing cluster" — the other members turned out to be faithful (see 4.1).

The reason it was parked is that `_load_table_old` (`table_io.py:614-619`)
*depends* on the aliasing to observe `delete_column`, so fixing the constructor
naively breaks the sparse loader. **That trap dissolves**: C++'s
`istreamTable_OLD` never calls that constructor — it fills `tab.itable` in place
(`table_io.cc:860`). Mirror that (`table = Table(); table.itable = itable`) and
the constructor is free to copy with zero loader breakage.

Resolved in S7 Task 0: `Table.__init__` copies both inputs, and
`_load_table_old` now fills the constructed table in place. Regression coverage
is in `tests/comboreduct/table/test_table.py`.

### 1.2 The same trap, one layer up, not yet ported

`bscore_ctable_base` (`scoring_base.h:236`) does `_wrk_ctable = _orig_ctable` and
then filters it. That is a **value copy** in C++ and would be an alias in Python.
Nothing ports it yet. When S8 does, copy explicitly.

### 1.3 `erase()` / `replace()` leave discarded nodes live (resolved in S7)

`tree.py:659` and `replace()` never null the removed node's own `parent` /
`next_sibling`, so a stale handle stays pointing into the tree where C++ dangles.

Resolved in S7: every destroyed node in the discarded subtree has its outward
links poisoned, and every subtree-copy entry point rejects a poisoned source.
This deliberately fails fast where a retained C++ iterator would dangle. The
regression coverage is in `tests/util/test_tree.py`.

### 1.4 `reduct::upwards` needs a subtree-rooted post-order walk

`meta_rules.cc:111` uses `post_order_iterator at=it, end=it; ++end;
at.descend_all();` — a walk rooted at a subtree, which `iter_post()` does not
offer. The private `_postorder_step` in `tree.py` makes this cheap to add.

### 1.5 The reduct test corpus is already available

`MOSES/elegant-normal-forms-python` contributes 44 curated boolean expressions
(including Holman p.19 and Moshe Looks examples), a truth-table equivalence
harness (`Tests/TestHelpers.py`), and seven ENF axiom checkers
(`Tests/EnfRuleCheckers.py`) — 89 tests that pass today. Its tree type is
unusable here, but the assets rewrite cleanly against `pymetta`'s.

`test_reduct` (`$CLASSIC/tests/comboreduct/reduct/test_reduct.cc`) is already
built and is the oracle for S7.

---

## 2. Latent defects — correct today, wrong the moment something reaches them

### 2.1 `_vertex_type_node` maps bare `str` to `enum_type` (scope-closed)

`table.py:123`. C++'s `get_type_tree` returns `definite_object_type` for a
`definite_object`. Unreachable from `loadTable`'s CSV path today, and the
values+label `OTable` overload is unused by `loadTable`. Both enum and object
domains are now explicitly excluded, so this is retained as historical context
rather than a future implementation item.

### 2.2 `CTable`'s dict key is type-blind

`dict[tuple[vertex,...], Counter[vertex]]`. Because `contin_t = float` and
`builtin` is an `IntEnum`, keys of different vertex types can collide — the same
class of bug that produced both Criticals in the final review. Not practically
reachable because each column's values are type-consistent by construction from
`table_io` parsing. It stops being safe if a column can ever hold mixed types.

### 2.4 `append_child`'s plain-data path does not value-copy, but `set_head` does

`tree.py:1052` passes a caller's `data` straight into the new node. **Measured
across the whole package (S8 review): `set_head` (`tree.py:1028`) is the ONLY
plain-data entry point that copies.** Every other plain-data insertion --
`append_child`, `prepend_child`, `insert`, `insert_after`, `replace` -- stores
the caller's object. The subtree-taking overloads all route through
`_copy_value`, so the split is *plain data vs. subtree*, not one stray function.
C++ copies by value at all of them (`append_child(pos, const T&)`), so **every
caller passing an existing node's payload must copy explicitly**. `knobs.py:286` already does; `representation.py`'s
`get_candidate_rec` did not.

Consequence, measured in S8: `get_candidate(inst, reduce=True)` on a negated
composite knob shared the exemplar's `argument` objects with the candidate, and
the reducer's De Morgan step rewrote them in place via `argument.negate()` --
turning the *exemplar* `or(and($1 $2) $3)` into `or(and(!$1 !$2) $3)`. C++ calls
this function thread-safe and says it "does not modify the rep at all"
(`representation.cc:313`). Fixed at the call site, not in `tree.py`, because the
plain path's no-copy behaviour is relied on elsewhere -- but the asymmetry is the
trap, and it will reappear wherever a new caller appends a node's `.data`.

Rule for S9+: appending `some_node.data` to another tree is always a bug unless
it is wrapped in `copy.copy`.

### 2.5 `logical_rules.py:1288` still compares vertices without `vertex_equal`

S8 closed two instances of this in `knobs.py` (a knob on contin `0.0` crashed;
a knob on `logical_true` mis-bound to a contin `1.0` sibling) by giving
`tree.equal_subtree` the optional `predicate` parameter C++ already has
(`tree.h:1923` / `:1949`). `logical_rules.py:1288` has the same shape and was
left untouched as out-of-scope for that slice.

It is latent for the same reason the `knobs.py` pair was: reduct currently sees
only Boolean trees. The continuous slice makes contin and builtin vertices
co-occur, and `contin_t = float` with `builtin` an `IntEnum` means
`1.0 == builtin.logical_true` is `True`. Fix it in the same change that enables
contin reduction, not after.

### 2.3 The weighted-CTable arity landmine

`Table::compressed` (`table.cc:527`) passes an **un-trimmed** `get_signature()`
alongside trimmed `ilabels`, and `CTable::filtered` (`table.h:594-595`) asserts
`number_of_children == get_arity()+1`. Ported faithfully. Confirmed present
(arity 5 vs 6 itypes) and inert, because scoring reads only `.otype` and classic's
own scoring never calls `get_signature()`/`filtered()` either. It fires the moment
something does.

---

## 3. Traps in classic MOSES and its tooling

These cost real time to find. They are properties of the C++ side, not of the port.

- **`eval-table`'s positional `$N` arguments are broken.** They always resolve to
  a zero column regardless of the `-L/--labels` flag, contrary to its help text.
  On XOR data, `-c 'and($1 $2)'` returns `0,0,0,0`; `-c 'and($x1 $x2)'` returns
  the correct `0,0,0,1`. Always substitute header names before invoking it.
  Nearly poisoned the Task 4 oracle.
- **`tree.hh`'s `move_ontop` corrupts the tree** when `source` is `target`'s
  immediate next sibling: `src->next_sibling = src`, and `parent->last_child` is
  left stale. Any later traversal hangs; destruction aborts. **Ported faithfully
  with a "DO NOT FIX" marker.** Of 11 call sites in `contin_rules.cc`, ten are
  safe by construction. **`contin_rules.cc:254` is now CLEARED TOO (traced
  2026-09-08) — S14 no longer needs to.** The precise trigger, from
  `tree.h:1710-1750`: `b_next_sibling = dst->next_sibling` is captured *before*
  `erase(target)`, so when `source == target->next_sibling` the code runs
  `b_next_sibling->prev_sibling = src` and `src->next_sibling = b_next_sibling`
  with `b_next_sibling == src` — the self-cycle. The condition is therefore
  exactly **`source == target->next_sibling`**. At `contin_rules.cc:254` the
  target is `tr.append_child(remain_plus)`, a freshly appended **last** child
  whose `next_sibling` is null, while `source` is `factor_parent.begin()`, which
  `tree_assert(src)` requires non-null — so it cannot hold.
  `contin_rules.cc:259-260` has the identical shape and is safe for the same
  reason. **All 11 call sites are clear**: no reachable heap corruption from
  these callers. Keep the DO NOT FIX marker — the bug is real, just unreachable.
- **Classic's own `optimizationUTest` CORE-DUMPS.** Confirmed by the controller
  on 2026-09-07: `./build/tests/moses/optimizationUTest` prints the `Hill:`
  debug header and then dies ("dumped core"). Cause, found during S9 Task 2:
  `distance_based_scorer` unconditionally dereferences through
  `iscorer_base::get_rep()` inside `hill-climbing.cc`'s own debug logging, and
  the test constructs a scorer with no representation. It is an upstream bug,
  not a port defect -- so **`make test` on classic is not fully green**, and
  any S9+ oracle that drives `hill_climbing` directly must supply a throwaway
  `representation` for `get_rep()` or hit the same crash. Do not "fix" it in
  the port; reproduce the behaviour and mark DO NOT FIX.
- **`metapopulationUTest` is NOT an oracle for the metapopulation slice.** It
  builds and passes, but its only two tests (`metapopulationUTest.cxxtest:74`
  `test_remove_dominated` and `:119` `test_merge_nondominated_rec`) exercise
  `remove_dominated`/`get_nondominated_disjoint` — both in `domination.cc`,
  which is P2 and outside S10. They cover none of `metapopulation.cc` or
  `merging.cc`'s default path. S10 must build its own probes.
  **The controller asserted otherwise in the first S10 plan after checking only
  that it built and passed.** "The oracle exists" and "the oracle covers what I
  am porting" are different claims; verify the second before planning around it.
- **`MOSESUTest` is S11's oracle, and it is NOT fully green in classic.**
  Measured 2026-09-08: 23 tests, **1 fails** — `test_ps_2_parity`
  (`moses-framework.h:169`, `fabs(result.first - expected_sc) < 1.0e-8` found
  `1.0000 >= 0.0000`). It uses `-aps` (particle swarm), a **P2 optimizer S11
  does not port**, so the failure is outside S11's scope. Do not chase it.
  The 23 cases map to slices by their flags (`tests/moses/MOSESUTest.cxxtest`):
  * **S11-usable now — CORRECTED 2026-09-08, only TWO:** `test_it` and
    `test_it_target_ignore`. My first version of this list named nine, chosen by
    checking that each uses the default hill-climbing optimizer. **I did not
    check which problem registry each goes through**, and seven of them are
    unreachable: `test_2_parity`, `test_2_parity_with_labels` and
    `test_3_parity_multi_thread` use `-Hpa`; `test_1_mux`/`test_2_mux` use
    `-Hmux`; `test_cp_simple`/`test_cp` use `-Hcp`. All six of those problems
    are registered in `demo-problems.cc` (**PLANNED: P2, not ported**) and need
    `logical_bscore` (`bscores.h:52`, also not ported). Only the `it` problem
    comes from `table-problems.cc`.
    This is the SECOND time I asserted an oracle after verifying one dimension
    and not another — the first was `metapopulationUTest`, verified to build and
    pass but not to cover the slice. Before naming a test as an oracle, check
    every layer it traverses: optimizer, problem registry, and scorer.
  * **P2 optimizers** (`-asa` star-anneal, `-aps` particle swarm, `-aun`
    univariate): `test_sa_2_parity`, `test_sa_1_mux`, `test_ps_2_parity`,
    `test_un_3_pa_multi_thread`.
  * **P2 boosting:** `test_4_parity_boosting`, `test_5_parity_boosting`.
  * **S14 (contin):** `test_cp_contin`, `test_sr_3`.
  * **P2 scorers:** `test_kl`, `test_multi_kl`, `test_precision`,
    `test_negative_predictive_value`.
  * **S13 (CLI/output):** `test_output_labels`, `test_python_output`.
  Note also that `MOSESUTest`'s own output prints a penalized score of
  `-3.57142853736877441` with complexity penalty `0.571428596973419189` --
  those are binary32 values printed at double width, independent confirmation
  that `score_t` narrowing (4.2) is observable in classic's own output.
- **Which classic `tests/moses` oracles actually exist.** Registered and
  active: `metapopulationUTest` (verified: builds, 2 tests, OK -- the S10
  oracle), `MOSESUTest` (the S11 end-to-end oracle), `optimizationUTest` (see
  above -- crashes), `FieldSetUTest`, `KnobTurningUTest`, `scoringUTest`,
  `iostream_scored_combo_treeUTest`, `AntUTest`, and `NeighborSamplingUTest`
  (gated on `Boost_MATH_C99_FOUND`). Commented OUT, so unavailable as oracles:
  `representationUTest`, `weightedUTest`, `mixedUTest`, `irisUTest`,
  `selectionUTest`, `dynFeatSelUTest`, `diversityUTest`.
- **The installed header tree at `~/opt/moses-legacy/include` is incomplete** —
  `table.h` includes `../interpreter/eval.h`, which was never installed. Probes
  must compile against `$CLASSIC`, and `libcogutil.so` lives in `lib/opencog`,
  not `lib`. Verified recipe is in `tools/cpp_probe/` build comments.
- **`rebuild-moses-local.sh` needs `-j <N>` with a space.** The `-j4` form in
  classic's own `CLAUDE.md` is rejected by its own argument parser.
- **The prebuilt binaries can silently rot.** They were linked against boost
  1.91, which no longer exists on this machine; every one failed at load until a
  full `--fresh` rebuild. Check `ldd` before trusting an oracle run.
- **`eval_logical_identities` is unsound, and `logical_reduce` is therefore
  semantically lossy** (added and corrected 2026-09-07). One directly visible
  trigger is **a junctor whose entire child list consists of childless
  junctors**, but input-shape detection is neither necessary nor sufficient for
  end-to-end truth-table divergence. Whether an occurrence matters depends on
  its parent, and earlier rules can create the shape during reduction. Measured with classic's
  own `test_reduct`: `and(or())` -> `true` and `and(or() or())` -> `true` both
  **preserve** (an emptied `and()` at the root folds to `true`, which is what
  the original evaluated to anyway), while `or(and(or()) $1)` -> `$1`
  **diverges** (the emptied `and()` is *erased* by its `or` parent as "skip",
  destroying a definite TRUE). A diverging subtree can also be **masked**:
  `and($1 or(and(or()) $1))` contains the diverging tree and still preserves.
  The effort-1 input `or(and(or($1 and(!$1))) !$2)` supplies the opposite
  counterexample: it is initially unflagged, but reduces from a tautology to
  `!$2` after an intermediate simplification creates the lossy shape.
  Mechanism: a childless junctor returns its *own token*
  (`interpreter.cc:66-82`), which the parent's short-circuit never matches, so
  the child is skipped — making `and(or())` evaluate TRUE. Reduct then erases
  the childless `or()` from its `and` parent (`logical_rules.cc:134`), which is
  faithful to "skip", but leaves `and()` itself childless; the grandparent
  erases *that* (`:155`). A definite TRUE has been rewritten into "skip me".
  Each rewrite is individually correct; their composition is not.
  Verified with an independent evaluator and a direct `boolean_interpreter`
  probe, not via `test_reduct`'s own comparison: `or(and(or()) $1)` gives
  original `11`, reduced `01`. 13 of the 29 `testTrees.sh` trees hit this;
  the 12 that carry the same `or()`/`and()` litter but keep a real child
  (`and(X or())`) are unaffected. `git log` shows this fork never touched
  `logical_rules.cc` or `interpreter.cc`, so it is upstream behaviour.
  **Consequence:** truth-table preservation is not a property `logical_reduce`
  has, and no current pre-reduction structural predicate completely classifies
  its failures. Differential checks must compare Python's semantic outcome to
  C++'s measured `tt_computed`/`tt_preserved` result directly, and must handle
  a bare `and()`/`or()` reaching the root as a distinct outcome —
  `builtin_to_bool` (`vertex.h:708-713`) `OC_ASSERT`s on a non-true/false
  builtin, which **throws `opencog::AssertionException`** (`oc_assert.cc:46`)
  out of `complete_truth_table` rather than returning. Whether that ends the
  process depends on the caller: `test_reduct` does not catch it (observed:
  `terminate called ... AssertionException`, core dumped), while a probe with
  `catch (std::exception&)` handles it cleanly. Do not describe this as
  "aborts the process" — the mechanism is a throw, and the difference decides
  whether a harness can measure the case at all.

---

## 4. Settled questions — do not relitigate

### 4.1 Most of the "aliasing cluster" is faithful, not a bug

`ITable.get_labels`/`get_types` and `CTable.get_input_labels` return internals,
but C++ returns `const string_seq&` / `const type_seq&` — **the aliasing is
correct**; only const-ness is lost. Accept permanently; treat as read-only.
`CTable.get_signature`/`__init__` alias a `Signature` that has `__slots__`, a
tuple `itypes`, and is never mutated post-construction — accept.

### 4.2 `score_t` is `float`, and it changes results

`types.h:63` is `typedef float score_t`. Keeping Python at double produced
demonstrated exemplar-selection inversions — 20,656 inverted pairs at MOSES's
default `complexity_ratio 3.5`. Narrowing is applied at 14 traced sites. The
`sum_bscore` accumulator stays in **double** and narrows once at return, matching
`boost::accumulate(bs, 0.0)`. `count_t` (double) and `complexity_t` (unsigned)
are not narrowed.

### 4.3 Empty `and()`/`or()` return the operator token

Not `logical_true`/`logical_false`. The identity only appears via a parent's
short-circuit (`interpreter.cc:66-82`). The port matches the real behaviour.

### 4.4 `argument`'s ordering is by absolute value

`-1, 1, -2, 2, -3, 3` — `absolute_value_order<int>` with tie-break
`(a == -b) ? a < b : abs(a) < abs(b)` (`argument.h:61-64`,
`$COGUTIL/opencog/util/numeric.h:56-60`). Ported as a verbatim comparator, not a
sort key. `flat_normal_form` clause ordering depends on it.

### 4.5 Breadcrumbs resolve against two roots

`moses/` paths against `$CLASSIC`, `opencog/` paths against `$COGUTIL`. Symbols
the matcher cannot handle — operators, destructors, templated names, inherited
members — cite the **enclosing class** at the member's line and name the real
member in the docstring body.

---

## 5. Known weaknesses in our own harness

- ~~**No fixture carries a checksum, version, or input manifest**, and `ci.sh`
  cannot rebuild the probes.~~ ~~**`tree.json` has no capture script in-repo.**~~
  **Both FIXED** (fixture-integrity task, 2026-09-08, commit `fddb6b6`). All 17
  fixtures now carry a manifest recording the capture script, the probe source
  and its SHA-256, and the inputs used — two embedded, fifteen as
  `<name>.manifest.json` sidecars (five fixtures have exact `set(golden) ==`
  tests that an embedded key would break, and three are top-level JSON lists).
  `tools/check_fixture_manifests.py` runs in `ci.sh`, in pure Python with no
  compiler, and **a new fixture without a manifest turns CI red** — deliberate
  enforcement. Capture scripts are now path-independent
  (`PYMETTA_CLASSIC_DIR` / `PYMETTA_COGUTIL_DIR` / `PYMETTA_MOSES_PREFIX`), so
  a fixture can actually be regenerated; `tree.json` finally has one and
  rebuilds byte-identically.
  Rot detection was proven in **both** directions, not merely the passing one:
  ~20 synthetic perturbation tests plus a live demo that zeroed a
  `probe_source_sha256`, saw `FAIL: 1/17`, reverted, and saw `OK: 17`.
  `rng.json`'s probe had been rewritten to link the real cogutil
  `MT19937RandGen`; a bounded rebuild proved the output byte-identical, so
  nothing was overwritten — the manifest just records that verification.
  **Note `tree.json` is still the only fixture without a shape guard**, and
  `test_tree.py`'s 33 oracle tests do not use `parametrize`, so
  `empty_parameter_set_mark` does not cover them. That half stands.
- **`check_alignment.py` checks 218 of 282 C++ citations.** It parses only
  Function/Class docstring first-lines, so module headers, `#`-comment
  breadcrumbs on constants, and `and <file>:<line>` overload continuations go
  unvalidated. Module headers come in five incompatible shapes.
- **Error types are inconsistent.** One C++ construct (`OC_ASSERT(false)`) maps to
  **six** Python types across the package — `AssertionError`, `IndexError`,
  `NotImplementedError`, `RuntimeError`, `TypeError`, `ValueError` (counted
  2026-09-08; this bullet said four, which was already stale). "Column not found" is `IndexError` at
  `table.py:302` but `ValueError` at `table_io.py:431,486,491,676`. The
  "forests unsupported" and `boost::bad_get`→`TypeError` families *are* uniform.
- ~~**`parse_divergences` captures only the first physical line** of a marker;
  two of nine markers use multi-line references.~~ **Fixed** (alignment-parser
  task, S7): `parse_divergences` now merges a `# DIVERGENCE(...)` marker with
  its same-indentation, unbroken-run-of-`#`-lines continuation into one
  `reason` — see that function's docstring in `tools/check_alignment.py` for
  the exact continuation rule. Authors do **not** need to front-load the
  essential statement onto the marker's first physical line anymore; a normal
  multi-line explanation now reaches `docs/ALIGNMENT.md` in full, so long as
  two adjacent markers are not run together (each marker still ends where the
  next one's `# DIVERGENCE(...)` begins).
- **A breadcrumb can cite the wrong overload and still pass.**
  `check_alignment.py` confirms the cited *identifier* appears within
  `DRIFT_TOLERANCE = 25` lines of the cited line — it does **not** confirm the
  cited line *is* that symbol. In overload-dense headers the window swallows
  every candidate: `table.h`'s five `Table` constructors sit inside ~15 lines,
  so a citation naming the un-ported `(It,It)` ctor instead of the `(tr,arity)`
  one two lines down is indistinguishable to the checker. Found in S7 as a
  systematic off-by-a-few cluster in `table.py:771-803` and `vertex.py:262,272`.
  The tolerance exists to survive upstream line drift and is right to; the
  consequence is that **overload choice is unverified and must be reviewed by
  eye**. Tightening it would trade a real protection for a narrow one.
- **Corpus gaps:** no argument index above `$3`, no arity ≥ 4.

---

## 5b. Settled by S12 (crossover) -- do not relitigate

### 5b.1 `std::partial_sort`'s tie order IS reachable, and it decides outcomes

`hill_climbing.py` now carries `_partial_sort`, a line-by-line
transliteration of libstdc++'s `__heap_select` + `__sort_heap`
(`/usr/include/c++/16/bits/stl_algo.h:1589,1871`, `bits/stl_heap.h`).
Measured against classic over 15 samples
(`tests/fixtures/crossover.json`'s `partial_sort` group): it reproduces
libstdc++'s permutation element for element, including ties, NaN, +/-inf,
`mid == 0`, `mid == size` and `size == 1`.

**The "a stable sort can only be at least as well-ordered as an unspecified
one" argument is FALSE for `partial_sort` and has been withdrawn** in
`resize_deme` (where the sort is immediately followed by a truncation, so
tie order decides which tied instances survive). It survives ONLY where the
result is not truncated or otherwise order-sensitive. `merging.py:674`
`sort_demes` still uses `deme.sort(reverse=True)` for `boost::sort`; that is
a DIFFERENT algorithm (introsort-shaped, not heap-select) and was not
re-examined by S12 -- if a slice ever makes that ordering observable, it
needs the same treatment, and `_partial_sort` is not the right model for it.

Two `composite_score`s tie under `operator<` exactly when penalized score
AND complexity are equal (`types.cc:139-148`) -- the normal state of a
Boolean deme, not a corner case.

### 5b.2 C's `sqrtf`/`cbrtf` conversions must be reproduced, with counterexamples

`unsigned n = sqrtf(2 * x) + 3;` converts `x` to binary32 BEFORE the call,
computes in binary32, and truncates toward zero. Computing it in Python
binary64 gives a different integer at `x = 8380417` (4097 vs 4096); the
`cbrtf(6 * x) + 3` twin differs at `x = 26244` (57 vs 56). Both confirmed
against a C++ snippet on this toolchain. `_sqrtf_plus_3`/`_cbrtf_plus_3` in
`hill_climbing.py` do it correctly; `math.cbrt` is not a
correctly-rounded operation, so that one rests on measurement (5012 inputs)
rather than an error bound.

### 5b.3 Two claims in `docs/CROSSOVER-INVENTORY.md` were wrong

Both are corrected inline there, with the measurement:

1. "small problems reach crossover ONLY via `last_chance`" -- no.
   `distance++` (`hill-climbing.cc:367-368`) runs before the `last_chance`
   block (`:434`), so a `last_chance` crossover is always at
   `distance >= 2`, where the estimate is `C(nn, 2)` and already exceeds
   400 for `nn >= 29`. `last_chance` is the deciding conjunct only for
   `nn <= 28`.
2. `cross_top_three`'s `k < i` bug "collides or skips" -- it never skips.
   Every slot below `num_to_make` is written; many are written repeatedly
   and the LAST write wins, with a `(j, k)` pair the index formula never
   intended. See the DO NOT FIX marker on `cross_top_three`.

### 5b.4 `(iteration > 2)` in the crossover gate is redundant

`current_number_of_instances >= 3` already excludes iterations 1 and 2:
iteration 1 runs at distance 0 and appends exactly one instance
(`hill-climbing.cc:521-522`), so the deme holds 0 then 1. Relaxing the
conjunct to `iteration > 1` changes nothing measurable -- an equivalent
mutant, documented as such in `test_crossover.py` rather than papered over
with a test that does not actually discriminate.

### 5b.5 `check_alignment.py` was silently dropping `) / DO NOT FIX:` markers

A marker written as

    # DIVERGENCE(C++ some/file.cc:687) / DO NOT FIX: ...

never parses: `_read_marker` scans for `"):"`, which that form never
contains. Two markers in `hill_climbing.py` had been missing from
`docs/ALIGNMENT.md` since S9 for exactly this reason, undetected -- the
same class of loss CARRY-FORWARD Sec 5 records for the earlier 49.
`tools/check_alignment.py` now reports any `# DIVERGENCE(`-opening comment
line that `parse_divergences` does not turn into a `Divergence`, which
turns CI red; `tests/test_alignment_checker.py` pins it, including
repo-wide. **Write markers as `# DIVERGENCE(C++ <ref>): <reason>` and put
any `DO NOT FIX` note AFTER the colon.**

### 5b.6 Untestable-by-construction masks in `cross_top_two`/`cross_top_three`

The `& _UNSIGNED_MAX` masks on `n` and `ntgt` are faithful but PROVABLY
unreachable, and no test covers them: the loop returns as soon as
`num_to_make <= n`, so `n` can only reach 2**32 if `num_to_make` exceeds
2**32, which would require `deme.resize()` of >4.29e9 `scored_instance`
objects. The same holds for `max`'s truncation (no `sample_size` below
200000 makes the wrapped value small enough to change the clamp). Kept as
faithful reproduction, reported as untested rather than claimed as covered.

### 5.1 A NEW class of fixture risk arrived with S12

Every fixture before S12 was a snapshot of **classic's pinned source**, so it
could only rot if classic changed. `crossover.json` is different: its
correctness depends on the **host toolchain**. `_partial_sort` is a
transliteration of libstdc++ 16's `__heap_select` + `__sort_heap`, chosen
because `std::partial_sort` gives no stability guarantee and its actual tie
order decides which instances get crossed — a stable sort turns 27 tests red.

No manifest carries a compiler or stdlib version field (schema-wide), and
`ci.sh` never rebuilds probes. **A libstdc++ upgrade could therefore leave the
fixture silently stale while the port stays green** — the exact rot the
manifest work was meant to prevent, arriving through a door that work did not
cover. The pin is disclosed in prose, which is honest but not
machine-checkable.

Whoever next touches the manifest schema should add the toolchain fields. Until
then, treat an unexplained `test_crossover.py` failure after a system upgrade
as a suspected toolchain change, not a port regression.

### 5.2 The 13-vs-17 deme size was a REAL divergence, waived twice

Worth recording as a process failure, because two rounds of scrutiny let it
through.

S10 Task 3 measured its merge probe matching classic on trees, order, scores,
`best_composite_score`, `best_tree` and return value — but **deliberately did
not assert** one field: an intermediate deme size of 13 (C++) vs 17 (port). It
waived that as harmless tie-order, since `deme_to_trees` dedups by tree.

I checked the argument and accepted it. S10's branch review then measured the
generalisation false — tie-order *does* decide which distinct trees survive
once `keep_top_unique_candidates` truncates — but concluded the waiver was
still "right for its scenario". **Both of us were wrong about the cause.**

S12 disproved the underlying premise for `std::partial_sort`, and a follow-up
investigation disproved it for `sort_demes`' own `std::sort` too: libstdc++'s
introsort matches a stable sort only for `n <= 16` (`_S_threshold`) and
measurably reorders ties above it. 17 all-tied elements give
`[8,16,15,14,13,12,11,10,9,0,7,6,5,4,3,2,1]`, not `[0..16]` — and `n > 16` is
the ordinary deme size, not an edge case.

Porting libstdc++'s real introsort makes the port produce **13, exactly
matching classic**. Proven by mutation: restoring `deme.sort(reverse=True)`
gives `assert 17 == 13`.

**The lesson is the one I stated at the time and then did not act on: a
deliberately-unasserted field is exactly where a real divergence hides.** An
unasserted mismatch should be treated as an open finding until its cause is
identified — not until an argument for its harmlessness sounds convincing.

### 5.3 The `pa` (even-parity) block was in S8, not S9-S11 -- and was already documented

Recorded because the misattribution cost a whole investigation, and because
the divergence had been WRITTEN DOWN and never costed.

`MOSESUTest`'s `test_2_parity` failed while classic solved it in one deme
expansion. The S13 demo-problems report correctly ruled out scoring (the
same stall reproduced through `ctruth_table_bscore`) and then attributed the
remainder to `local_moses`/`hill_climbing`/`metapopulation`. Wrong: those
were faithful. The cause was **`build_knobs::sample_logical_perms`' random
PAIR perms (`build_knobs.cc:435-518`), never ported**, plus `perm_ratio`
defaulting to `-1.0` in `build_knobs.py`/`representation.py` where classic
defaults to `0.0` (`build_knobs.h:56`, `representation.h:62`,
`problem-params.cc:299`). Classic built 16 knobs for the `-Hpa -k2`
exemplar; the port built 8, and even parity is the first problem in the
corpus unreachable from single-literal knobs.

Two things are worth carrying forward:

1. **`build_knobs.py`'s own comment already said classic's default was 0.0
   and that 0.0 generates pairs.** The divergence was documented, honestly
   and accurately. What nobody did was ask what it COST. A documented
   divergence is not a discharged one -- same failure mode as 5.2's
   13-vs-17 waiver.
2. **Ruling out one layer does not implicate the next.** Scoring was
   correctly excluded; the layer BETWEEN scoring and the search loop --
   representation building -- was never checked.

Rule: when an end-to-end oracle fails, **diff the search SPACE before
diffing the search**. Classic prints it for free at `-l DEBUG`
("Total number of field specs", "Created prototype:"), it is one line to
compare, and it is upstream of every optimizer decision.

Fixed 2026-09-08; full trajectory diff, the ruled-out list and the mutation
table are in
`.superpowers/sdd/2026-09-08-pymetta-s13-cli/parity-divergence-report.md`.

**Concurrency hazard, seen live on this branch (2026-09-08).** This fix is
commit `1ddb097` -- but it first landed, twice, inside a concurrent agent's
scorer commit, because `git add <paths>` followed by `git commit` is a
TWO-STEP RACE on a shared index: the other agent's `git commit` (and later
its `reset`/re-commit) picked up whatever this task had staged in between.
The project's "explicit pathspecs, never `git add -A`" rule is necessary
and not sufficient. **Use the one-step `git commit -F msg -- <paths>`,
which ignores the index entirely**; `git add` only for genuinely new files,
in the same shell invocation as the commit. Related: do NOT `git stash` on
a shared branch either -- a `git stash push --include-untracked -- <paths>`
here swallowed another agent's in-flight untracked file and the pop then
aborted on a generated doc, leaving a partially-restored tree.

### 5.4 A scope decision that silently became a correctness bug

The most instructive failure in this project, and it was mine.

S8's plan said: "Start with `perm_ratio=-1` to exclude random pair-permutation
behavior from this cycle." Sensible for S8 — it kept the first representation
slice deterministic. But `perm_ratio`'s real default is **`0.0`**
(`build_knobs.h:56`, `representation.h:62`, `problem-params.cc:299`), and `0.0`
*generates* the argument-pair perms at `build_knobs.cc:435-518`. The port not
only defaulted to `-1.0`, it **hard-rejected** any other value in
`_validate_request` — so classic's actual default was unreachable.

Consequence: the port built **8 knobs where classic builds 16**. Measured at the
first divergence, deme 1 iteration 2: `estimate_neighborhood(d=1)` returned 24
against classic's 50, and `sample_new_instances` produced 16 against 32. 2-parity
became unsolvable — the port stalled at -1.0 where classic reaches 0.

**It survived S8's review and every review since**, plus S9-S12's oracles,
because no oracle ran a problem where pair perms mattered. `test_it` converges
almost immediately and is seed-invariant; the trajectory oracles diffed the port
against a probe built from *the port's own* configuration, so both sides were
equally wrong. Only classic's own `-Hpa` case, reached in the P2 tail, exposed it.

Three rules follow:
1. **A scope simplification must be recorded as a DIVERGENCE, not just a plan
   sentence.** Had `perm_ratio=-1` carried a marker, it would have surfaced in
   `ALIGNMENT.md` at every later review.
2. **A default that differs from classic's is a divergence even when nothing
   reads it yet** — it becomes live the moment a CLI wires the real one in.
3. **An oracle configured by the port cannot detect the port's configuration
   being wrong.** At least one oracle per slice must run classic's own defaults.

After the fix the port reproduces classic's exact `-r1` tree for `-Hmux -k2`,
`-Hpa -k3` and `-Hmaj -k3` — three problems where it previously found a
different one — and `-Hpa -k3` now scores 0.

### 5.5 A false commit message of my own, and how it happened

Commit `1760611` says it rewrote an invalid `"\|"` escape in `table.py` as a
`grep -E` alternation. **It does not.** The diff adds 25 lines and removes
none, the added text uses `grep -rn "push_back"` rather than any `grep -E`, and
`ae4ea9f:table.py` never contained that escape at all. Found by the S13 review.

What actually happened: the escape existed only in a *concurrent agent's
uncommitted buffer*. I edited that buffer, and my commit — which commits the
working-tree state of the named paths — captured their whole 25-line CTable
note. Their rewrite of the same passage superseded my one-line fix, so the
escape is genuinely gone (`python3 -W error::SyntaxWarning` is clean
package-wide) but not for the reason the message gives.

Nothing was lost, and the outcome is correct. The record is not. This is §6's
own failure mode applied to a commit message rather than a docstring: a
confident, checkable, false claim, which is worse than no claim because it
stops the next reader looking.

**Rule:** on a shared worktree, `git commit -- <paths>` captures whatever is in
those files *now*, including another agent's edits. Before writing the message,
read the actual staged diff — `git show --stat` and the hunks — rather than
describing what you intended to change.

## 6. Process notes that paid for themselves

- **Prove a guard fires.** Requiring "revert it and show the test fail" caught a
  validator that did not actually validate, and a corpus that structurally could
  not detect the bug it was written for.
- **Vacuous tests recur.** A parametrised oracle over an empty fixture passes
  silently. Fixed locally in Task 1, it came back in Task 6; now fixed globally
  via `empty_parameter_set_mark = "fail_at_collect"`. `test_tree.py` is still
  outside that guard (see 5).
- **Per-task review cannot see cross-task defects.** Both final Criticals came
  from Task 3's type choices meeting Tasks 4 and 6's comparisons. Neither task
  was wrong alone.
- **A fix can reintroduce the bug it fixes.** The final fix wave's new cache key
  reopened the same cross-type collision at a wider scope. The scoped re-review
  caught it.
- **A wrong port asserted as verified fidelity is worse than a wrong port.** S7
  Task 5 shipped an unbounded iterator walk whose docstring claimed the C++
  end-condition "reduces to a null check" — a specific, confident, checkable
  claim that was simply false. Because it read as measured, it carried no
  `# DIVERGENCE` marker, so `ALIGNMENT.md` reported the file as faithful and
  the next reader had every reason not to look. Prefer "not yet verified" to a
  plausible reconstruction: a gap invites a check, a false claim forecloses it.
