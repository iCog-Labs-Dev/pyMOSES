// tools/cpp_probe/logical_rules_probe.cc
//
// Measures the REAL output of opencog::reduct's logical_rules.{h,cc} rules
// -- above all subtree_to_enf::reduce_to_enf -- for pymetta's
// comboreduct/reduct/logical_rules.py port (task-4). Ground truth, not
// inference.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/logical_rules_probe tools/cpp_probe/logical_rules_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/logical_rules_probe )

#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/reduct/meta_rules.h>
#include <moses/comboreduct/reduct/general_rules.h>
#include <moses/comboreduct/reduct/logical_rules.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/type_checker/type_tree.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

static string show(combo_tree& tr)
{
    ostringstream oss;
    ostream_combo_tree(oss, tr);
    return oss.str();
}

static string quote(const string& s)
{
    ostringstream oss;
    oss << "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') oss << '\\';
        oss << c;
    }
    oss << "\"";
    return oss.str();
}

static void run_rule(const string& label, const string& prog, const rule& r)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    r(tr, it);
    cout << quote(label) << ": " << quote(show(tr)) << "," << endl;
}

// subtree_to_enf's own docstring: reduce_nots()/eval_logical_identities()/
// level() should run first. Canonicalize with the untyped combinators
// (matches what logical_rules.py's own tests exercise), then apply
// subtree_to_enf() ONCE at the root -- exactly logical_reduction.cc's own
// "pre_subtree_to_enf; subtree_to_enf(); post_subtree_to_enf" shape, minus
// insert_ands/reduce_ands/reduce_ors (kept OUT so this probe isolates
// reduce_to_enf's own contribution specifically).
static void run_enf(const string& label, const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    upwards up1{eval_logical_identities()};
    up1(tr, it);
    downwards dn1{level()};
    dn1(tr, it);
    subtree_to_enf enf;
    enf(tr, it);
    cout << quote(label) << ": " << quote(show(tr)) << "," << endl;
}

// subtree_to_enf() ALONE, with NO pre-canonicalization -- unlike run_enf
// above. Needed for the Task 4 review Important's "trailing 0-subsumption
// Disconnect" and "reduce_or Disconnect propagation" regression cases:
// their own repro shape (redundant same-operator nesting, e.g.
// and(and(and()))) is exactly what level()'s own flattening collapses
// away BEFORE subtree_to_enf ever sees it, so run_enf's pre-pass would
// silently route around the very branch being tested. subtree_to_enf's
// own docstring frames the pre-pass as needed for "proper reduction"
// quality, not as a precondition for calling it at all.
static void run_enf_raw(const string& label, const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    subtree_to_enf enf;
    enf(tr, it);
    cout << quote(label) << ": " << quote(show(tr)) << "," << endl;
}

// Full effort-3 "complexe" reduction, TYPED (the real thing, using real
// type inference -- confirms this probe's "untyped" replica pipeline below
// visits the exact same nodes for these BOOLEAN-ONLY trees).
static string boolean_reduce_typed(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    vertex_set ignore_ops;
    logical_reduce(3, tr, ignore_ops);
    return show(tr);
}

// Same pipeline, UNTYPED (no id::boolean_type filter on any downwards) --
// this is what logical_rules.py's own test harness can build today, since
// meta_rules.py's typed-downwards branch is not implemented until Task 5.
// simplify_predicates is dropped entirely (never reachable: no predicate
// ever appears in these trees, so its own C++ guard would no-op anyway).
static string boolean_reduce_untyped_replica(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();

    sequential pre_subtree_to_enf =
        sequential(upwards(eval_logical_identities()),
                   downwards(level()),
                   downwards(insert_ands()));
    sequential post_subtree_to_enf =
        sequential(downwards(reduce_ands()),
                   downwards(reduce_ors()));

    sequential medium =
        sequential(downwards(reduce_nots()),
                   iterative(sequential(pre_subtree_to_enf,
                                        subtree_to_enf(),
                                        post_subtree_to_enf)),
                   downwards(remove_unary_junctors()));

    iterative complexe = iterative(sequential(medium, reduce_remove_subtree_equal_tt()));
    // reduce_remove_subtree_equal_tt needs no predicates present (true for
    // this corpus) but ALSO needs a clean tree for its own truth-table
    // amputation walk; since these trees have none of the S14 predicate
    // machinery, run it -- if it ever changes anything for this corpus,
    // that is itself worth knowing.
    complexe(tr, it);
    return show(tr);
}

// "medium"-only (real C++'s effort 2), UNTYPED, WITHOUT
// reduce_remove_subtree_equal_tt -- this port's own logical_rules.py test
// harness target, since reduce_remove_subtree_equal_tt itself needs
// complete_truth_table (comboreduct/table, out of this file's scope) and
// clean_reduce (Task 5, unbuilt). This is what logical_rules.py's own
// tests are pinned against exactly (byte-for-byte), NOT the real
// logical_reduce.
static string boolean_reduce_medium_untyped(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();

    sequential pre_subtree_to_enf =
        sequential(upwards(eval_logical_identities()),
                   downwards(level()),
                   downwards(insert_ands()));
    sequential post_subtree_to_enf =
        sequential(downwards(reduce_ands()),
                   downwards(reduce_ors()));

    sequential medium =
        sequential(downwards(reduce_nots()),
                   iterative(sequential(pre_subtree_to_enf,
                                        subtree_to_enf(),
                                        post_subtree_to_enf)),
                   downwards(remove_unary_junctors()));

    medium(tr, it);
    return show(tr);
}

// Same as boolean_reduce_medium_untyped, but TYPED (boolean_type on every
// downwards) -- the real "effort 2" composition, for comparison.
static string boolean_reduce_medium_typed(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    vertex_set ignore_ops;
    logical_reduction r(ignore_ops);
    const rule& medium = r(2);
    medium(tr);
    return show(tr);
}

int main()
{
    insert_ands ia;
    remove_unary_junctors ruj;
    remove_dangling_junctors rdj;
    eval_logical_identities eli;
    reduce_nots rn;
    reduce_ors ro;
    reduce_ands ra;

    cout << "{" << endl;

    // ---- insert_ands ----
    run_rule("insert_ands_bare_argument_at_root", "$1", ia);
    run_rule("insert_ands_or_under_not", "not(or($1 $2))", ia);
    run_rule("insert_ands_argument_under_or", "or($1 $2)", ia);

    // ---- remove_unary_junctors ----
    run_rule("remove_unary_junctors_and_one_child", "and($1)", ruj);
    run_rule("remove_unary_junctors_or_one_child", "or(and($1 $2))", ruj);

    // ---- remove_dangling_junctors ----
    {
        combo_tree tr(vertex(id::logical_and));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(id::logical_and)); // childless -- dangling
        tr.append_child(root, vertex(argument(2)));
        rdj(tr, root);
        cout << quote("remove_dangling_junctors_plain") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(id::logical_and)); // childless condition 1 (dropped)
        tr.append_child(root, vertex(argument(1)));      // consequent 1 (dropped too)
        tr.append_child(root, vertex(argument(2)));       // condition 2 (kept)
        tr.append_child(root, vertex(argument(3)));       // consequent 2 (kept)
        tr.append_child(root, vertex(argument(4)));       // else clause (kept)
        rdj(tr, root);
        cout << quote("remove_dangling_junctors_cond") << ": " << quote(show(tr)) << "," << endl;
    }

    // ---- eval_logical_identities ----
    run_rule("eval_logical_identities_and_true_or_childless", "and(or() $1 true)", eli);
    run_rule("eval_logical_identities_and_false_shortcircuit", "and($1 false $2)", eli);
    run_rule("eval_logical_identities_or_one_child", "or(and($1 $2))", eli);
    run_rule("eval_logical_identities_or_true_shortcircuit", "or($1 true $2)", eli);
    run_rule("eval_logical_identities_or_and_childless_false", "or(and() false $1)", eli);

    // ---- reduce_nots ----
    run_rule("reduce_nots_double_not", "not(not($1))", rn);
    run_rule("reduce_nots_triple_not", "not(not(not($1)))", rn);
    run_rule("reduce_nots_distribute_and", "not(and($1 $2))", rn);
    run_rule("reduce_nots_distribute_or", "not(or($1 $2))", rn);
    run_rule("reduce_nots_argument", "not($1)", rn);
    run_rule("reduce_nots_negated_argument", "not(!$1)", rn);
    run_rule("reduce_nots_true", "not(true)", rn);
    run_rule("reduce_nots_false", "not(false)", rn);

    // ---- reduce_ors / reduce_ands ----
    run_rule("reduce_ors_dup_clause", "or($1 $1 $2)", ro);
    run_rule("reduce_ors_subset", "or(and($1 $2) $1)", ro);
    run_rule("reduce_ors_implication", "or(and($1 $3) and(!$1 $3))", ro);
    run_rule("reduce_ands_dup_clause", "and($1 $1 $2)", ra);
    run_rule("reduce_ands_superset", "and(or($1 $2) $1)", ra);
    run_rule("reduce_ands_implication", "and(or($1 $3) or(!$1 $3))", ra);
    run_rule("reduce_ands_common_factor", "and(or($1 $2) or($1 $3))", ra);
    // Task 4 review Critical #2 regression cases -- the naive Python
    // "while i2 < i1" (a stale snapshot of c1's index) instead of C++'s
    // live "c2 != c1" bound: pre-fix these crashed with IndexError.
    run_rule("reduce_ors_stale_bound_regression", "or(!$1 or(and($2 !$1 !$2)) !$2 $2)", ro);
    run_rule("reduce_ands_stale_bound_regression", "and($1 and(or($2 $1 $2)) $2 !$2)", ra);

    // ---- subtree_to_enf, in isolation (canonicalized first) ----
    run_enf("enf_and_arg_negarg_contradiction", "and($1 not($1))");
    run_enf("enf_and_dup_children", "and($1 $1 $2)");
    run_enf("enf_or_dup_children", "or($1 $1 $2)");
    run_enf("enf_and_or_same_arg", "and($1 or($1 $2))");
    run_enf("enf_and_or_neg_arg", "and($1 or(not($1) $2))");
    run_enf("enf_or_and_dominant", "or(and($1 $2) and($1 $3))");
    run_enf("enf_nested_and_or_and", "and($1 or($2 and($1 $3)))");
    run_enf("enf_command_promotion", "or($1 and(not($1) $2))");
    run_enf("enf_and_cut_single_child_or", "and($1 and(or($2 $3)))");
    run_enf("enf_all_true", "and(true true)");
    run_enf("enf_dominant_command_conflict", "or($1 and(not($1) not($1)))");
    run_enf("enf_and_bare_complement_contradiction", "and($1 !$1)");

    // ---- Task 4 review Important: 5 load-bearing branches that survived
    // mutation testing (26 mutants / 74 tests, 16 survivors, 5 of them
    // behaviour-changing) because nothing exercised them. Each input below
    // was found by instrumenting the Python port with a print() at the
    // target line and searching by hand until each one, and ONLY that one
    // (plus, in one case, its own dependency), actually fired.
    run_enf("enf_one_subsumption_delete", "or(and($1) and($1 $2))");
    run_enf_raw("enf_trailing_zero_subsumption_disconnect", "and(and(and()))");
    run_enf_raw("enf_reduce_or_disconnect_propagation", "or($1 and(and(and())))");
    // The remaining two survivors (and_cut(child) itself, and the
    // reprocess-all-kids reset that follows it) are each guarded by a
    // SEPARATE mutation-kill case: and_cut(child)'s own effect can be
    // silently redone by the OTHER and_cut call site (the "else" branch)
    // on a second pass through the reset loop, so a case that merely
    // *reaches* this branch is not enough -- found by differential
    // testing the Python port against itself with each line separately
    // mutated away, then confirmed here.
    run_enf("enf_and_cut_after_res_promotion", "and(or($1 $2) or(and($2 $1) and($2 !$1)))");
    run_enf("enf_reprocess_all_kids_reset",
            "and(!$1 and(or(or($3 $2) and($3 !$2)) !$2 or(and($3 !$3) and($3 !$1))))");

    // ---- reduce_and_assumptions / reduce_or_assumptions ----
    {
        reduce_and_assumptions raa; // default ctor: _reduction = this
        run_rule("reduce_and_assumptions_default_no_shrink", "and($1 $2 $3)", raa);
        run_rule("reduce_and_assumptions_not_an_and_noop", "or($1 $2)", raa);
    }
    {
        reduce_or_assumptions roa; // default ctor: _reduction = this
        run_rule("reduce_or_assumptions_default", "or($1 $2)", roa);
        run_rule("reduce_or_assumptions_not_an_or_noop", "and($1 $2)", roa);
    }
    {
        reduce_and_assumptions inner_raa;
        reduce_or_assumptions roa2(inner_raa);
        run_rule("reduce_or_assumptions_with_and_assumptions_reduction", "or($1 $2 $3)", roa2);
    }

    // ---- BooleanReductUTest's own 6 trees: typed vs untyped-replica ----
    {
        static const char* trees[] = {
            "and(and(not(not(and($4 $3))) or(not(or($5 $4)) or(and($1 $5) and($1 $5)))) and(and(and(not($1) not($2)) not(not($3))) or(and(not($4) not($5)) or(or($2 $1) not($5)))))",
            "not(or(and(and(not($5) or($5 $3)) not(or($5 $2))) and(or(and($5 $3) or($5 $3)) not(not($1)))))",
            "and(or(not(and(or($5 $3) not($2))) or(or(or($1 $2) and($2 $3)) or(and($2 $4) not($5)))) not(not(and(or($3 $3) or($3 $4)))))",
            "or(and(not($1) $3 or($3 not($1)) or(not($1) and($2 $1) and($3 $2))) and($3 $1) and(not($3) or($3 $1) not(or($3 not($1))) not(or($2 $1))))",
            "or(and(or(not($1) not($2)) $3) and(or(not($1) $3) not($2)))",
            "and(or(and(or(not($1) $3) not($2)) and(not($1) $3)) or(not($2) $3))",
        };
        for (int i = 0; i < 6; ++i) {
            string typed = boolean_reduce_typed(trees[i]);
            string untyped = boolean_reduce_untyped_replica(trees[i]);
            string med_typed = boolean_reduce_medium_typed(trees[i]);
            string med_untyped = boolean_reduce_medium_untyped(trees[i]);
            cout << quote("boolean_reduct_effort3_typed_" + to_string(i + 1)) << ": "
                 << quote(typed) << "," << endl;
            cout << quote("boolean_reduct_effort3_untyped_" + to_string(i + 1)) << ": "
                 << quote(untyped) << "," << endl;
            cout << quote("boolean_reduct_effort3_agree_" + to_string(i + 1)) << ": "
                 << (typed == untyped ? "true" : "false") << "," << endl;
            cout << quote("boolean_reduct_medium_typed_" + to_string(i + 1)) << ": "
                 << quote(med_typed) << "," << endl;
            cout << quote("boolean_reduct_medium_untyped_" + to_string(i + 1)) << ": "
                 << quote(med_untyped) << "," << endl;
            cout << quote("boolean_reduct_medium_agree_" + to_string(i + 1)) << ": "
                 << (med_typed == med_untyped ? "true" : "false") << "," << endl;
        }
    }

    cout << quote("_end") << ": true" << endl;
    cout << "}" << endl;
    return 0;
}
