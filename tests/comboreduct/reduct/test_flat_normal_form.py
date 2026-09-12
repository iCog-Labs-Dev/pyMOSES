"""Tests for pymoses/comboreduct/reduct/flat_normal_form.py -- the flat
clause-set representation (`clause`, `nf`), its free functions
(`tautology`, `subset_eq`, `subset`, `number_of_literals`, `implications`,
`pairwise_erase_if`), and `nf_mapper` (converts between combo's tree-shaped
boolean expressions and this flat form).

C++: moses/comboreduct/reduct/flat_normal_form.{h,cc}.

Every literal expected value below is copied verbatim from a REAL run of
`tools/cpp_probe/flat_normal_form_probe.cc` against $CLASSIC's compiled
`libcomboreduct.so` (bounded:
`( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/flat_normal_form_probe
)`), not hand-derived from the header/source comments alone -- see that
file and flat_normal_form.py's own module docstring for the full
transcript, and task-3-report.md for the annotated derivation of the
non-obvious cases (`add_item`'s sign-folding, the `extract_cnf`/
`extract_dnf` asymmetry over an empty clause).
"""

from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin, combo_tree
from pymoses.comboreduct.reduct.flat_normal_form import (
    clause,
    implications,
    nf_mapper,
    nf_to_str,
    number_of_literals,
    pairwise_erase_if,
    subset,
    subset_eq,
    tautology,
)

# ---------------------------------------------------------------------------
# clause -- ordering
# ---------------------------------------------------------------------------


def test_clause_orders_by_absolute_value_not_signed_value():
    """REQUIRED test (task-3-brief.md): clause ordering for a set
    containing both `$1` and `!$1` -- the case a natural (signed) sort
    gets wrong. `-1, 1, -2, 2, -3, 3` (docs/CARRY-FORWARD.md §4.4), not a
    natural ascending sort (which would give `-3,-2,-1,1,2,3`) and not a
    "negatives grouped together" sort either. Measured directly
    (`flat_normal_form_probe.cc`'s `clause_ordering_mixed`).
    """
    c = clause()
    for x in (3, -1, 2, 1, -2, -3):
        c.insert(x)
    assert list(c) == [-1, 1, -2, 2, -3, 3]
    # The property under test, spelled out explicitly: a natural sort of
    # the SAME inputs would NOT match.
    assert sorted([3, -1, 2, 1, -2, -3]) != list(c)


def test_clause_orders_a_bare_pair_with_the_negative_one_first():
    c = clause()
    c.insert(1)
    c.insert(-1)
    assert list(c) == [-1, 1]


def test_clause_treats_a_literal_and_its_negation_as_distinct_elements():
    # (a == -b) is only a TIE-BREAK in absolute_value_order, not an
    # equivalence -- 1 and -1 are still two separate set elements, in
    # the negative-first order (not just "size 2", which alone would not
    # catch a bug that let one silently displace or reorder the other).
    c = clause()
    c.insert(1)
    c.insert(-1)
    assert list(c) == [-1, 1]


def test_clause_insert_dedups_equal_literals():
    c = clause()
    c.insert(2)
    c.insert(2)
    assert list(c) == [2]


def test_clause_erase_is_a_noop_when_absent():
    c = clause([1, 2])
    c.erase(99)
    assert list(c) == [1, 2]


def test_clause_repr_matches_the_real_stream_operator_format():
    # C++: flat_normal_form.cc:62 operator<<(clause) -- "(a b c)"
    c = clause()
    for x in (3, -1, 2, 1, -2, -3):
        c.insert(x)
    assert repr(c) == "(-1 1 -2 2 -3 3)"
    assert repr(clause()) == "()"


def test_nf_to_str_concatenates_clauses_with_no_separator():
    # C++: flat_normal_form.cc:74 operator<<(nf) -- no space between clauses.
    f = [clause([1, 2]), clause([-1, 3]), clause([4])]
    assert nf_to_str(f) == "(1 2)(-1 3)(4)"


# ---------------------------------------------------------------------------
# tautology / subset_eq / subset / number_of_literals
# ---------------------------------------------------------------------------


def test_tautology_true_when_a_literal_and_its_negation_are_both_present():
    assert tautology(clause([1, -1, 2])) is True


def test_tautology_false_without_a_complementary_pair():
    assert tautology(clause([1, 2])) is False


def test_tautology_false_for_an_empty_clause():
    assert tautology(clause()) is False


def test_subset_eq_true_for_a_genuine_subset():
    assert subset_eq(clause([1, 2]), clause([-1, 1, 2, 3])) is True


def test_subset_eq_true_for_equal_clauses():
    c = clause([1, 2])
    assert subset_eq(c, c) is True


def test_subset_eq_false_when_not_a_subset():
    assert subset_eq(clause([1, 5]), clause([1, 2, 3])) is False


def test_subset_strict_true_for_a_genuine_subset():
    assert subset(clause([1, 2]), clause([-1, 1, 2, 3])) is True


def test_subset_strict_false_for_equal_clauses():
    c = clause([1, 2])
    assert subset(c, c) is False


def test_number_of_literals_sums_clause_sizes():
    f = [clause([1, 2]), clause([3]), clause()]
    assert number_of_literals(f) == 3


# ---------------------------------------------------------------------------
# implications
# ---------------------------------------------------------------------------


def test_implications_resolves_a_single_complementary_pair():
    # (p X)(!p Y) implies (X Y): c1={1,2}, c2={-1,3} -> resolvent {2,3}
    result = list(implications(clause([1, 2]), clause([-1, 3])))
    assert [list(c) for c in result] == [[2, 3]]


def test_implications_multiple_complementary_pairs_can_all_be_tautologies():
    # C++: flat_normal_form_probe.cc "implications_multi" -- c1={1,2},
    # c2={-1,-2,4} have TWO complementary pairs; resolving on EITHER pair
    # still leaves the OTHER pair in the resolvent, making BOTH would-be
    # resolvents tautologies. Real propositional-resolution behaviour,
    # not a bug -- measured, not assumed.
    result = list(implications(clause([1, 2]), clause([-1, -2, 4])))
    assert result == []


def test_implications_suppresses_a_tautological_resolvent():
    result = list(implications(clause([1, 2]), clause([-1, -2])))
    assert result == []


def test_implications_no_complementary_literal_yields_nothing():
    result = list(implications(clause([1, 2]), clause([3, 4])))
    assert result == []


# ---------------------------------------------------------------------------
# pairwise_erase_if
# ---------------------------------------------------------------------------


def test_pairwise_erase_if_drops_a_clause_subsumed_by_an_earlier_one():
    # C++: flat_normal_form_probe.cc "pairwise_erase_if_drop_subsumed_by_earlier"
    # -- p = subset_eq (the real call site's own predicate,
    # logical_rules.cc:304,369): {1} subsumes {1,2} (the smaller clause
    # implies more), so the bigger one is redundant and gets dropped.
    f = [clause([1]), clause([1, 2])]
    pairwise_erase_if(f, subset_eq)
    assert nf_to_str(f) == "(1)"


def test_pairwise_erase_if_drops_a_clause_subsumed_by_a_later_one():
    # C++: flat_normal_form_probe.cc "pairwise_erase_if_drop_subsumed_by_later"
    f = [clause([1, 2]), clause([1])]
    pairwise_erase_if(f, subset_eq)
    assert nf_to_str(f) == "(1)"


def test_pairwise_erase_if_keeps_incomparable_clauses():
    # C++: flat_normal_form_probe.cc "pairwise_erase_if_keeps_incomparable"
    f = [clause([1, 2]), clause([3, 4])]
    pairwise_erase_if(f, subset_eq)
    assert nf_to_str(f) == "(1 2)(3 4)"


def test_pairwise_erase_if_collapses_duplicate_clauses_to_one():
    # C++: flat_normal_form_probe.cc "pairwise_erase_if_collapses_duplicate"
    # -- the surviving duplicate is the LATER one (index 2), not the
    # first: when the outer (later, index-2) element and an earlier one
    # compare subset_eq BOTH ways (they are equal), `p(c1, c2)` -- c1 the
    # outer, LATER element -- is checked before `p(c2, c1)`, so the
    # EARLIER one (index 0) is what gets erased. {3} (index 1) is
    # untouched and ends up first once index 0 is gone. Traced by hand
    # against pairwise_erase_if's own docstring and confirmed against
    # the real library (this exact case, measured).
    f = [clause([1, 2]), clause([3]), clause([1, 2])]
    pairwise_erase_if(f, subset_eq)
    assert nf_to_str(f) == "(3)(1 2)"


# ---------------------------------------------------------------------------
# nf_mapper -- add_item id assignment and sign-folding
# ---------------------------------------------------------------------------


def test_add_cnf_assigns_ids_in_first_seen_order_and_folds_argument_signs():
    # C++: flat_normal_form_probe.cc "add_cnf_ids_mixed_args"
    tr = str_to_combo_tree("or($2 $1 !$2 $3 !$1)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    assert nf_to_str(f) == "(1)(2)(-1)(3)(-2)"


def test_add_dnf_over_or_of_ands():
    # C++: flat_normal_form_probe.cc "add_dnf_or_of_ands"
    tr = str_to_combo_tree("or(and($1 $2) and(!$1 $3) $4)")
    m = nf_mapper()
    f = m.add_dnf(tr.begin())
    assert nf_to_str(f) == "(1 2)(-1 3)(4)"


def test_add_item_interns_a_structurally_repeated_compound_subtree():
    # C++: flat_normal_form_probe.cc "add_cnf_interning_repeat_subtree" --
    # the SAME compound subtree ("or($1 $2)") appearing twice, as
    # DIFFERENT tree nodes but structurally identical, must be assigned
    # the SAME id both times.
    tr = str_to_combo_tree("or(or($1 $2) $3 or($1 $2))")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    assert nf_to_str(f) == "(1 2)(3)(1 2)"


def test_add_item_returns_the_same_id_for_the_same_argument_seen_twice():
    tr = str_to_combo_tree("or($1 $1)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    assert nf_to_str(f) == "(1)(1)"


# ---------------------------------------------------------------------------
# nf_mapper -- extract_cnf / extract_dnf round trips
# ---------------------------------------------------------------------------


def test_extract_cnf_roundtrip():
    # C++: flat_normal_form_probe.cc "extract_cnf_roundtrip"
    tr = str_to_combo_tree("or(or($1 !$2) $3)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    m.extract_cnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "and(or($1 !$2) $3) "


def test_extract_dnf_roundtrip():
    # C++: flat_normal_form_probe.cc "extract_dnf_roundtrip"
    tr = str_to_combo_tree("or(and($1 $2) $3)")
    m = nf_mapper()
    f = m.add_dnf(tr.begin())
    m.extract_dnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "or(and($1 $2) $3) "


def test_extract_cnf_collapses_a_single_clause_single_literal_to_a_bare_leaf():
    # C++: flat_normal_form_probe.cc "extract_cnf_single_clause_single_literal"
    tr = str_to_combo_tree("or($1)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    m.extract_cnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "$1 "


def test_extract_dnf_collapses_a_single_clause_single_literal_to_a_bare_leaf():
    # C++: flat_normal_form_probe.cc "extract_dnf_single_clause_single_literal"
    # -- extract_cnf's own twin (immediately above) was pinned; this one
    # was not, and the two ARE genuinely different code paths (deletable
    # independently -- task-3 review, Important).
    tr = str_to_combo_tree("and($1)")
    m = nf_mapper()
    f = m.add_dnf(tr.begin())
    m.extract_dnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "$1 "


def test_extract_cnf_of_an_entirely_empty_nf_becomes_logical_false():
    # C++: flat_normal_form_probe.cc "extract_cnf_empty_nf" -- ported
    # verbatim even though "AND of nothing" is mathematically TRUE, not
    # FALSE (see extract_cnf's own docstring: port faithfully, document
    # upstream oddities, never silently fix them).
    tr = str_to_combo_tree("$1")
    m = nf_mapper()
    m.extract_cnf([], tr, tr.begin())
    assert combo_tree_to_str(tr) == "false "


def test_extract_dnf_of_an_entirely_empty_nf_becomes_logical_false():
    # C++: flat_normal_form_probe.cc "extract_dnf_empty_nf"
    tr = str_to_combo_tree("$1")
    m = nf_mapper()
    m.extract_dnf([], tr, tr.begin())
    assert combo_tree_to_str(tr) == "false "


def test_extract_cnf_silently_skips_an_empty_clause():
    # C++: flat_normal_form_probe.cc "extract_cnf_with_empty_clause_present"
    # -- a real, disclosed ASYMMETRY with extract_dnf below: an empty
    # clause in a CNF contributes NOTHING (silently), rather than making
    # the whole conjunction logical_false the way an empty OR-clause
    # "should" in real boolean algebra.
    tr = str_to_combo_tree("$1")
    m = nf_mapper()
    seed = str_to_combo_tree("or($5)")
    m.add_cnf(seed.begin())  # interns $5 as id 1, matching the probe's seed step
    f = [clause(), clause([1])]
    m.extract_cnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "$5 "


def test_extract_dnf_any_empty_clause_makes_the_whole_result_true():
    # C++: flat_normal_form_probe.cc "extract_dnf_with_empty_clause_present"
    # -- the asymmetric counterpart of the CNF case above: ANY empty
    # clause collapses the WHOLE dnf to logical_true, discarding
    # whatever real clauses were already appended as children.
    tr = str_to_combo_tree("$1")
    m = nf_mapper()
    seed = str_to_combo_tree("and($5 $6)")
    seed_f = m.add_dnf(seed.begin())
    f = [seed_f[0], clause()]
    m.extract_dnf(f, tr, tr.begin())
    assert combo_tree_to_str(tr) == "true "


# ---------------------------------------------------------------------------
# nf_mapper -- extract_conjunction
# ---------------------------------------------------------------------------


def _fresh_and_scaffold():
    """A `logical_and` root with one placeholder child, for
    `extract_conjunction` to overwrite -- matches the probe's own scaffold
    (a manually-built outer `and(...)` wraps the single node under test,
    so the printed result shows what `extract_conjunction` did to that
    one node without disturbing anything above it).
    """
    tr = combo_tree(builtin.logical_and)
    sib = tr.append_child(tr.begin(), 0.0)
    return tr, sib


def test_extract_conjunction_single_positive_literal():
    # C++: flat_normal_form_probe.cc "extract_conjunction_single_positive"
    tr = str_to_combo_tree("or($1 $2)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    c1 = f[0]

    dst, sib = _fresh_and_scaffold()
    m.extract_conjunction(c1, dst, sib)
    assert combo_tree_to_str(dst) == "and($1) "


def test_extract_conjunction_single_negative_literal():
    # C++: flat_normal_form_probe.cc "extract_conjunction_single_negative"
    tr = str_to_combo_tree("or($1 $2)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    id1 = f[0].first()

    dst, sib = _fresh_and_scaffold()
    m.extract_conjunction(clause([-id1]), dst, sib)
    assert combo_tree_to_str(dst) == "and(!$1) "


def test_extract_conjunction_multi_literal_becomes_a_nested_and():
    # C++: flat_normal_form_probe.cc "extract_conjunction_multi"
    tr = str_to_combo_tree("or($1 $2 $3)")
    m = nf_mapper()
    f = m.add_cnf(tr.begin())
    all_literals = clause()
    for c in f:
        all_literals.insert(c.first())

    dst, sib = _fresh_and_scaffold()
    m.extract_conjunction(all_literals, dst, sib)
    assert combo_tree_to_str(dst) == "and(and($1 $2 $3)) "


def test_extract_conjunction_singleton_zero_literal_is_a_noop():
    # C++: flat_normal_form.h:205-206's own `if (*c.begin()!=0)` guard --
    # a clause holding literally {0} leaves `sib` completely untouched.
    m = nf_mapper()
    dst, sib = _fresh_and_scaffold()
    m.extract_conjunction(clause([0]), dst, sib)
    assert combo_tree_to_str(dst) == "and(0) "


# ---------------------------------------------------------------------------
# nf_mapper -- create()'s argument-aliasing hazard (self-review finding)
# ---------------------------------------------------------------------------


def test_create_does_not_corrupt_the_interned_copy_across_repeated_calls():
    """Regression test for a translation hazard `create`'s own docstring
    documents: C++'s `*sib=*tmp; get_argument(*sib).negate();` copies a
    `vertex` BY VALUE before negating the copy (safe in C++ -- the
    interned original in `_int2item` is never touched). This port's
    `argument` is a mutable OBJECT, and `tree.append_child`'s
    subtree-copy path aliases `.data` by reference (an existing, pervasive
    property of this port) -- a literal "copy, then mutate the copy"
    translation would therefore negate the SAME object still sitting in
    `_int2item`, corrupting every LATER `create()` call for that id. This
    asserts two independent `create()` calls for the same negative id
    produce IDENTICAL, uncorrupted output.
    """
    tr = str_to_combo_tree("!$2")
    m = nf_mapper()
    neg_id = m.add_item(tr.begin())  # !$2 interned as the canonical (positive-slot) id

    out1 = combo_tree(builtin.logical_and)
    m.create(out1, out1.begin(), -neg_id)
    out2 = combo_tree(builtin.logical_and)
    m.create(out2, out2.begin(), -neg_id)

    assert combo_tree_to_str(out1) == "and($2) "
    assert combo_tree_to_str(out1) == combo_tree_to_str(out2)


def test_extract_conjunction_negative_branch_does_not_corrupt_the_interned_copy():
    """Regression test for `extract_conjunction`'s own twin of `create`'s
    aliasing hazard (task-3 review, Important: the `create` fix had this
    pin, `extract_conjunction`'s did not). Reverting `extract_conjunction`'s
    negative branch to the naive `sib.data = tmp.data; sib.data.negate()`
    passes every OTHER test in this file, yet three successive calls with
    the SAME negative singleton clause then alternate `and(!$1)` /
    `and($1)` / `and(!$1)` instead of returning `and(!$1)` every time --
    each call's `.negate()` flips the shared, aliased `_int2item` entry
    for the NEXT call to read. This asserts all three stay consistent.
    """
    tr = str_to_combo_tree("$1")
    m = nf_mapper()
    id1 = m.add_item(tr.begin())
    neg = clause([-id1])

    results = []
    for _ in range(3):
        dst, sib = _fresh_and_scaffold()
        m.extract_conjunction(neg, dst, sib)
        results.append(combo_tree_to_str(dst))

    assert results == ["and(!$1) ", "and(!$1) ", "and(!$1) "]
