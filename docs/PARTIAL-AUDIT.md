# PARTIAL Audit: MODULE-COVERAGE.md Gap Classification

**Audit date:** 2026-09-08  
**Scope:** 62 rows marked `PARTIAL` in `docs/MODULE-COVERAGE.md`

## Summary

Of the 62 PARTIAL entries, **55 are Category A** (gaps scheduled to named slices or excluded by scope), and **7 are Category B** (genuine unscheduled gaps not excluded by scope). No entries are ambiguous.

- **A (Deliberate roadmap):** 55 items — gaps scheduled to S11, S12, S13, S14, or P2, or explicitly excluded by port scope (enum, ANN, list/fold, object, message, procedure-call, legacy embodiment, or current-slice narrowing).
- **B (Genuine gap):** 7 items — unscheduled, in-scope gaps that lack a slice assignment and are not explicitly excluded.
- **?:** 0 items.

## Classification Table

| File | Lines | Category | Deciding phrase or reason (B only) |
|---|---:|---|---|
| `moses/comboreduct/combo/combo.h` | 35 | A | "vertex.h/iostream_combo.h/table.h parts are ported (S3/S5)" |
| `moses/comboreduct/combo/iostream_combo.cc` | 646 | A | "per spec S3"; excluded vertex variants named (action, perception, message, procedure_call, ann_type, etc.) |
| `moses/comboreduct/combo/iostream_combo.h` | 287 | A | Same as iostream_combo.cc |
| `moses/comboreduct/combo/vertex.cc` | 272 | A | "enum_t, ann_type, object, message, and procedure-call variants are explicitly excluded" |
| `moses/comboreduct/combo/vertex.h` | 828 | A | Same as vertex.cc |
| `moses/comboreduct/crutil/exception.cc` | 55 | B | No Python exception hierarchy; maps inconsistently (OC_ASSERT → 4 types) |
| `moses/comboreduct/crutil/exception.h` | 81 | B | Same as exception.cc |
| `moses/comboreduct/interpreter/eval.cc` | 593 | A | "contin/mixed interpreter coverage remains assigned to S14" |
| `moses/comboreduct/interpreter/eval.h` | 85 | A | Same as eval.cc |
| `moses/comboreduct/interpreter/interpreter.cc` | 387 | A | "contin_interpreter and mixed_interpreter... needed for S14" |
| `moses/comboreduct/interpreter/interpreter.h` | 101 | A | Same as interpreter.cc |
| `moses/comboreduct/reduct/flat_normal_form.cc` | 79 | A | S7 scope: concrete shapes ported, generic iterator surface intentionally narrowed |
| `moses/comboreduct/reduct/flat_normal_form.h` | 281 | A | Same as flat_normal_form.cc |
| `moses/comboreduct/reduct/general_rules.cc` | 141 | A | "evaluation remains limited to the builtin/argument/contin vertex grammar" |
| `moses/comboreduct/reduct/general_rules.h` | 77 | A | Same as general_rules.cc |
| `moses/comboreduct/reduct/logical_reduction.cc` | 164 | A | "remains part of S14" |
| `moses/comboreduct/reduct/logical_rules.cc` | 860 | A | "contin/mixed predicate behavior... remain outside S7" |
| `moses/comboreduct/reduct/logical_rules.h` | 286 | A | Same as logical_rules.cc |
| `moses/comboreduct/reduct/reduct.cc` | 33 | A | "full contin/mixed rule composition remains S14" |
| `moses/comboreduct/reduct/reduct.h` | 269 | A | Same as reduct.cc |
| `moses/comboreduct/table/table.cc` | 928 | A | "Generic vertex variants remain outside the current grammar; CARRY-FORWARD 2.2, 2.3" |
| `moses/comboreduct/table/table.h` | 1511 | A | "reachable S7 path has no call sites for them" |
| `moses/comboreduct/table/table_io.cc` | 1441 | A | "scoped to... Boolean/contin table grammar" |
| `moses/comboreduct/table/table_io.h` | 250 | A | Same as table_io.cc |
| `moses/comboreduct/type_checker/type_tree.cc` | 1468 | A | "only the signature helpers S8 calls... inference engine NOT ported" (S8 scope) |
| `moses/comboreduct/type_checker/type_tree.h` | 442 | A | Same as type_tree.cc |
| `moses/comboreduct/type_checker/type_tree_def.h` | 109 | A | "S8 Task 0 MOVED... everything else remains unported" (S0-S6/S8 scope) |
| `moses/moses/deme/deme_expander.cc` | 533 | A | "single-deme path" (S10 scope; feature-selection/subsampling excluded) |
| `moses/moses/deme/deme_expander.h` | 158 | A | Same as deme_expander.cc |
| `moses/moses/deme/deme_params.h` | 88 | A | "`fstor` (feature selection)... any non-None value raises" (P2 excluded from S10) |
| `moses/moses/metapopulation/ensemble.h` | 98 | A | "Full boosting stays P2 with ensemble.cc" |
| `moses/moses/metapopulation/ensemble_params.h` | 54 | A | "boosting algorithm it configures stays P2" |
| `moses/moses/metapopulation/merging.cc` | 679 | A | "Domination/diversity/boosting/subsampling arms raise" (P2 excluded from S10) |
| `moses/moses/metapopulation/metapop_params.cc` | 90 | A | "The S13 CLI wiring that sets them must populate them first" (S13 responsibility) |
| `moses/moses/metapopulation/metapop_params.h` | 250 | A | Same as metapop_params.cc |
| `moses/moses/metapopulation/metapopulation.cc` | 248 | A | "Diversity, domination, boosting and subsampling are ported as GATES that RAISE" (P2 excluded from S10) |
| `moses/moses/metapopulation/metapopulation.h` | 574 | A | Same as metapopulation.cc |
| `moses/moses/moses/moses_params.h` | 102 | A | S11 scope: fields ported, atomicity and threading acknowledged as out-of-scope |
| `moses/moses/moses/neighborhood_sampling.cc` | 276 | A | "Contin-domain helpers... deferred to S14" |
| `moses/moses/moses/neighborhood_sampling.h` | 543 | A | Same as neighborhood_sampling.cc |
| `moses/moses/moses/types.cc` | 299 | A | "demeID_t... (-> S10/S11); scored_combo_tree containers... (-> S10); ostream_* functions (-> S13)" |
| `moses/moses/moses/types.h` | 449 | A | Same as types.cc |
| `moses/moses/optimization/hill-climbing.cc` | 846 | A | "serial hill-climbing with `crossover=False` (S12 turns it on)" |
| `moses/moses/optimization/hill-climbing.h` | 280 | A | Same as hill-climbing.cc |
| `moses/moses/optimization/optimization.cc` | 190 | A | "Term-field contribution not ported: term fields are excluded from this port's `field_set`" (S9 scope) |
| `moses/moses/optimization/optimization.h` | 192 | A | Same as optimization.cc |
| `moses/moses/representation/build_knobs.cc` | 1302 | A | "Boolean path only... excluded domains raise" (S8 scope) |
| `moses/moses/representation/build_knobs.h` | 193 | A | Same as build_knobs.cc |
| `moses/moses/representation/field_set.cc` | 270 | A | "Enum-only and object-domain specifications omitted" |
| `moses/moses/representation/field_set.h` | 1383 | A | Same as field_set.cc |
| `moses/moses/representation/instance.h` | 65 | A | "arrive with `instance_set` in S9" |
| `moses/moses/representation/instance_scorer.h` | 111 | B | Reproduces upstream bug (iscorer_base::get_rep() → nullptr); marked DO NOT FIX for fidelity |
| `moses/moses/representation/instance_set.h` | 154 | A | "reachable S9 path has no call sites for them" |
| `moses/moses/representation/knob_mapper.h` | 98 | A | "Continuous entries deferred to the continuous slice" |
| `moses/moses/representation/knobs.cc` | 424 | A | "`action_subtree_knob`... (narrow action slice); `contin_knob` (continuous slice)" |
| `moses/moses/representation/knobs.h` | 347 | A | Same as knobs.cc |
| `moses/moses/representation/representation.cc` | 422 | A | "deliberately not ported (it would weaken thread-safe design); continuous/action-result maps deferred" (S8 scope decision) |
| `moses/moses/representation/representation.h` | 217 | A | Same as representation.cc |
| `moses/moses/scoring/bscores.cc` | 1085 | B | "logical_bscore, interesting_predicate_bscore and cluster_bscore have none [assigned slice]. SURPRISE." |
| `moses/moses/scoring/bscores.h` | 632 | B | Same as bscores.cc |
| `moses/moses/scoring/scoring_base.cc` | 329 | B | "`bscore_ctable_base`... CARRY-FORWARD 1.2 flags... [slice assignment] unresolved discrepancy" |
| `moses/moses/scoring/scoring_base.h` | 351 | B | Same as scoring_base.cc |

## Category B Items: Candidate Work Queue

These unscheduled gaps are in-scope and not assigned to any named future slice. Each would involve:

1. **`moses/comboreduct/crutil/exception.{cc,h}`** — Create a unified exception hierarchy mapping all ComboReductException/OverflowException/EvalException/TypeCheckException variants to corresponding Python types, replacing ad hoc ValueError/IndexError/TypeError maps.

2. **`moses/moses/representation/instance_scorer.h`** — *Do not close.* Marked DO NOT FIX; faithful reproduction of classic bug where `distance_based_scorer` dereferences nullptr via `iscorer_base::get_rep()`. Requires throwaway `representation` object to exercise in S9+ oracles.

3. **`moses/moses/scoring/bscores.cc`/`.h`** — Port three allowed-domain scorer classes (`logical_bscore`, `interesting_predicate_bscore`, `cluster_bscore`) that are neither assigned to S14 nor excluded by domain. Not named in S7–S14 backlog or P2 tail.

4. **`moses/moses/scoring/scoring_base.cc`/`.h`** — Port `bscore_ctable_base` (CTable ignore_cols/rows filtering used by feature-selection). CARRY-FORWARD 1.2 tentatively attributed to "S8" but "real consumer looks like feature-selection (P2)" — slice assignment unresolved.

## CARRY-FORWARD §5 Cross-Check

**CARRY-FORWARD §5** ("Known weaknesses in our own harness") lists 7 items:

| §5 Item | Corresponding PARTIAL Row(s) | Status |
|---|---|---|
| 1. Fixture checksums/versions unsupported | None (not module coverage) | No PARTIAL row |
| 2. `tree.json` capture script absent | None (not module coverage) | No PARTIAL row |
| 3. `check_alignment.py` incomplete coverage | None (not module coverage) | No PARTIAL row |
| **4. Error types inconsistent** (OC_ASSERT → 4 types) | **exception.cc, exception.h** | **B item — recorded** |
| 5. Parse divergences multi-line capture (FIXED S7) | None (already resolved) | No open gap |
| 6. Breadcrumb can cite wrong overload | None (not module coverage) | No PARTIAL row |
| 7. Corpus gaps | None (not module coverage) | No PARTIAL row |

**Finding:** Only 1 of 7 CARRY-FORWARD §5 items corresponds to a PARTIAL row (exception.{cc,h}). The other six address test infrastructure, validation tooling, or documentation — not module coverage. No §5 items remain undocumented; no PARTIAL rows refer to unrecorded §5 debts.

**Note:** Other B items are documented in CARRY-FORWARD but outside §5:
- `instance_scorer.h`: CARRY-FORWARD §3 (upstream classic bug, reproduced faithfully)
- `bscores`/`scoring_base`: CARRY-FORWARD §1.2 and MODULE-COVERAGE "Surprises" (gaps noted but slice assignment unresolved)
