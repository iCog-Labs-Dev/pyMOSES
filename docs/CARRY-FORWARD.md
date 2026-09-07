# Carry-forward from the S0–S6 foundation cycle

Written 2026-09-04, at the close of the foundation cycle (commits `569e801..4ecba1f`).

This is what the next cycle needs to know and would otherwise have to rediscover.
Everything here was found by building or reviewing the foundation; none of it is
speculation. Items are grouped by when they bite.

---

## 1. Must be settled before S7 (reduct) and S8 (knob building)

### 1.1 `Table.__init__` aliases where C++ copies

`table.py:574` stores `otable`/`itable` by reference; `table.cc:485` copies both.
This is the only genuine divergence in what was originally logged as a broad
"aliasing cluster" — the other members turned out to be faithful (see 4.1).

The reason it was parked is that `_load_table_old` (`table_io.py:614-619`)
*depends* on the aliasing to observe `delete_column`, so fixing the constructor
naively breaks the sparse loader. **That trap dissolves**: C++'s
`istreamTable_OLD` never calls that constructor — it fills `tab.itable` in place
(`table_io.cc:860`). Mirror that (`table = Table(); table.itable = itable`) and
the constructor is free to copy with zero loader breakage.

Fix this before anything holds a `Table` across mutations, which S7+ will.

### 1.2 The same trap, one layer up, not yet ported

`bscore_ctable_base` (`scoring_base.h:236`) does `_wrk_ctable = _orig_ctable` and
then filters it. That is a **value copy** in C++ and would be an alias in Python.
Nothing ports it yet. When S8 does, copy explicitly.

### 1.3 `erase()` / `replace()` leave discarded nodes live

`tree.py:659` and `replace()` never null the removed node's own `parent` /
`next_sibling`, so a stale handle stays pointing into the tree where C++ dangles.

S8's `knob_mapper` stores one iterator per knob. Ported knob code will silently
reattach to a live tree where C++ would crash — a wrong result instead of a
signal. Decide deliberately: null the links, or document that handles must not
outlive an erase.

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

### 2.1 `_vertex_type_node` maps bare `str` to `enum_type`

`table.py:123`. C++'s `get_type_tree` returns `definite_object_type` for a
`definite_object`. Unreachable from `loadTable`'s CSV path today, and the
values+label `OTable` overload is unused by `loadTable`. Revisit when
`definite_object_type` becomes reachable.

### 2.2 `CTable`'s dict key is type-blind

`dict[tuple[vertex,...], Counter[vertex]]`. Because `contin_t = float` and
`builtin` is an `IntEnum`, keys of different vertex types can collide — the same
class of bug that produced both Criticals in the final review. Not practically
reachable because each column's values are type-consistent by construction from
`table_io` parsing. It stops being safe if a column can ever hold mixed types.

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
  safe by construction; `contin_rules.cc:254` could not be cleared — **S14 must
  trace it.** If reachable, classic has a latent heap-corruption bug.
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
  semantically lossy** on a characterised class of input (added 2026-09-07).
  Divergence occurs **iff a junctor's entire child list consists of childless
  junctors**. Mechanism: a childless junctor returns its *own token*
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
  has. Any check must assert it on the complement of this class, and must
  handle a bare `and()`/`or()` reaching the root as a distinct outcome —
  `builtin_to_bool` (`vertex.h:708-713`) `OC_ASSERT`s on a non-true/false
  builtin, so that case aborts `complete_truth_table` rather than returning.

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

- **No fixture carries a checksum, version, or input manifest**, and `ci.sh`
  cannot rebuild the probes. All six fixtures can rot silently.
- **`tree.json` has no capture script in-repo** — the recipe lives only in the
  plan document — and it is the only fixture without a shape guard.
  `test_tree.py`'s 33 oracle tests do not use `parametrize`, so
  `empty_parameter_set_mark` does not cover them.
- **`check_alignment.py` checks 218 of 282 C++ citations.** It parses only
  Function/Class docstring first-lines, so module headers, `#`-comment
  breadcrumbs on constants, and `and <file>:<line>` overload continuations go
  unvalidated. Module headers come in five incompatible shapes.
- **Error types are inconsistent.** One C++ construct (`OC_ASSERT(false)`) maps to
  four Python types across the package. "Column not found" is `IndexError` at
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
