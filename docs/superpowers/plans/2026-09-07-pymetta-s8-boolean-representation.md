# pyMoses S8: Boolean Representation Plan

**Date:** 2026-09-07
**Status:** Proposed; implementation starts after approval
**Depends on:** completed S7 reduct slice

## Scope

Build the first deterministic MOSES representation vertical slice:

`Boolean exemplar -> logical knobs -> field_set -> packed instance -> candidate tree`

The only supported output domains for the overall Python port are Boolean,
continuous, and action. S8 implements Boolean first and leaves reusable seams for
continuous and action follow-ons. Enum, ANN, list/fold, object, message,
procedure-call, and legacy ant/pet embodiment domains are not deferred work;
they are excluded.

The following classic paths must therefore not be ported in S8:

- `build_knobs::enum_canonize` and `build_knobs::build_enum`
- ANN node enumeration or ANN knob construction
- definite/indefinite object, message, or procedure repository support
- `#ifdef LATER` and `#ifdef EXEMPLAR_INST_IS_UNDEAD` branches

## Fidelity Rules

- Preserve classic field ordering and raw bit layout exactly.
- Preserve prototype tree bytes and knob index order, not only semantics.
- Keep the S7 single-leaf `type_tree` wrapper intact. It is not a signature tree.
- Add only the signature helpers required by Boolean representation.
- Use C++ probes and committed JSON fixtures for every stateful boundary.
- Build candidate trees from fresh copies so concurrent calls cannot share mutation.

## Task 0: Signature Prerequisite

Introduce an immutable `TypeSignature` with ordered input types and one output
type. Port only the helpers S8 calls: `gen_signature`, `get_signature_inputs`,
`get_signature_output`, `type_tree_arity`, `boolean_arity`, `contin_arity`, and
`action_result_arity`.

Do not broaden `pymoses/comboreduct/type_checker/type_tree.py` into the classic
1,468-line inference engine. Reconcile the existing `table.Signature` so the port
has one signature model, by migration or a compatibility re-export.

**Likely files:**

- `pymoses/comboreduct/type_checker/type_signature.py`
- `pymoses/comboreduct/table/table.py`
- `tests/comboreduct/type_checker/test_type_signature.py`
- `tools/cpp_probe/type_signature_probe.cc`

**Exit:** helper outputs and arities match the C++ probe for Boolean signatures;
continuous and action leaf types round-trip without enabling their builders.

## Task 1: Instance and Field Storage

Port packed instance storage and the field-set subset required by Boolean logical
knobs. Include `disc_spec` and the reusable `contin_spec` layout, but omit enum-only
or object-domain specifications. Port comparator order verbatim: discrete
multiplicity descending; continuous `(depth, expansion, step_size, mean)`
descending.

Cover bit packing across 64-bit boundaries, raw get/set, continuous encode/decode,
offset calculation, iteration, and Hamming distance.

**Likely files:**

- `pymoses/moses/representation/instance.py`
- `pymoses/moses/representation/field_set.py`
- `tests/moses/representation/test_instance.py`
- `tests/moses/representation/test_field_set.py`
- `tools/cpp_probe/field_set_probe.cc`

**Exit:** field ordering, offsets, raw values, and round trips match pinned C++
fixture rows exactly.

## Task 2: Logical Knob Primitive

Port the knob base/discrete mapping and `logical_subtree_knob` only. Cover absent,
present, and negated states; existing exemplar nodes; newly appended defaults; and
invalid values. Add `knob_mapper` using tree-node identity and verify that S7's dead
handle checks reject mappings to removed nodes.

**Likely files:**

- `pymoses/moses/representation/knobs.py`
- `pymoses/moses/representation/knob_mapper.py`
- `tests/moses/representation/test_knobs.py`
- `tools/cpp_probe/logical_knob_probe.cc`

**Exit:** every knob turn matches classic's exact tree output and index mapping.

## Task 3: Deterministic Boolean Builder

Port the Boolean path of `build_knobs` in classic order:

`logical_canonize -> sample positive literals -> add/build_logical -> logical_cleanup`

Start with `perm_ratio=-1` to exclude random pair-permutation behavior from this
cycle. Do not add placeholder dispatch for unsupported domains; reject them with a
clear exception.

**Likely files:**

- `pymoses/moses/representation/build_knobs.py`
- `tests/moses/representation/test_build_knobs.py`
- `tools/cpp_probe/representation_probe.cc`

**Exit:** prototype tree bytes, knob count, field specifications, and knob indexes
match C++ for bare arguments and small AND/OR exemplars.

## Task 4: Candidate Round Trip

Port the minimal `representation` surface needed to assemble sorted specs and
indexes and implement `get_candidate(instance, reduce)`. Rebuild a fresh tree for
each call, apply knob values, then run `clean_reduce` and the configured logical
reducer when requested.

Do not port `instance_set`, `scored_instance`, or `instance_scorer` in this first
vertical slice; they become the bridge from representation to S9 optimization.

**Likely files:**

- `pymoses/moses/representation/representation.py`
- `tests/moses/representation/test_representation.py`
- `tests/fixtures/representation.json`
- `tools/capture_representation.py`

**Exit:** all small packed instances reproduce C++ reduced and unreduced candidate
strings, knob order, and Boolean truth tables.

## Task 5: Review and Closeout

Run focused unit tests, fixture-shape tests, a bounded live differential, Ruff,
`git diff --check`, and `bash tools/ci.sh`. Update `docs/ALIGNMENT.md`,
`docs/MODULE-COVERAGE.md`, and carry-forward notes with measured S8 results.

The next approved cycle after this one is either:

1. representation-to-optimizer containers (`instance_set`, `scored_instance`,
   `instance_scorer`) for S9; or
2. continuous representation, followed by a separately designed narrow action
   representation.

Neither option may introduce an excluded output domain as a hidden dependency.
