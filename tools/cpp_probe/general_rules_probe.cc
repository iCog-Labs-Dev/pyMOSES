// tools/cpp_probe/general_rules_probe.cc
//
// Measures the REAL output of opencog::reduct's general_rules.cc rules
// (level, eval_constants, reorder_commutative, remove_null_vertices) for
// pymoses's comboreduct/reduct/general_rules.py port (task-3). Ground
// truth, not inference -- particularly for eval_constants's
// partition+reparent dance, which tree.h's `partition` was separately
// confirmed (partition_probe.cc) to NOT be a stable partition.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/general_rules_probe tools/cpp_probe/general_rules_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/general_rules_probe )
#include <moses/comboreduct/reduct/general_rules.h>
#include <moses/comboreduct/reduct/flat_normal_form.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>

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

// small local helper (matching other probes' inline JSON style)
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

static void run_case(const string& label, const string& prog, const rule& r, bool at_root = true)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    if (at_root)
        r(tr, it);
    else
        r(tr, it.begin()); // apply at the root's first child instead
    cout << quote(label) << ": " << quote(show(tr)) << "," << endl;
}

int main()
{
    level lv;
    eval_constants ec;
    reorder_commutative rc;
    remove_null_vertices rnv;

    cout << "{" << endl;

    // ---- level: flatten nested associative calls ----
    run_case("level_flatten_and", "and($1 and($2 $3) $4)", lv);
    run_case("level_flatten_deep", "and($1 and($2 and($3 $4)))", lv);
    run_case("level_flatten_deep_left_nested", "and($1 and(and($2 $3) $4))", lv);
    run_case("level_no_flatten_or_inside_and", "and($1 or($2 $3))", lv);
    run_case("level_plus", "plus(1 plus(2 3) 4)", lv);

    // ---- eval_constants: fold pure-constant subtrees ----
    run_case("eval_constants_all_const_and", "and(true false)", ec);
    run_case("eval_constants_all_const_plus", "plus(1 2 3)", ec);
    run_case("eval_constants_mixed_and", "and($1 true $2 false)", ec);
    run_case("eval_constants_mixed_plus", "plus(2 $1 1)", ec);
    run_case("eval_constants_mixed_plus_interleaved", "plus(2 $1 3 $2 1)", ec);
    run_case("eval_constants_single_const_only_child", "and(true)", ec);
    run_case("eval_constants_single_const_with_siblings", "and(true $1)", ec);
    run_case("eval_constants_not", "not(true)", ec);
    run_case("eval_constants_greater_than_zero", "0<(3)", ec);
    run_case("eval_constants_no_consts", "and($1 $2)", ec);
    run_case("eval_constants_childless_arg", "$1", ec);

    // ---- reorder_commutative: sort children lexicographically ----
    run_case("reorder_commutative_args", "and($2 $1 $3)", rc);
    run_case("reorder_commutative_arg_and_negation", "and($1 !$1)", rc);
    run_case("reorder_commutative_negation_and_arg", "and(!$1 $1)", rc);
    run_case("reorder_commutative_mixed_neg", "and($2 !$1 $1 !$2)", rc);
    run_case("reorder_commutative_or", "or($3 $1 $2)", rc);
    run_case("reorder_commutative_subtree_vs_leaf", "and(or($1 $2) $1)", rc);
    run_case("reorder_commutative_two_neg_only", "and(!$2 !$1)", rc);
    // multi-child NON-commutative operator (malformed combo -- not() only
    // ever has one real child -- but reorder_commutative's own guard is
    // purely `is_commutative(*it)`, arity-blind; built via append_child
    // directly since str2combo_tree may not parse a 2-arg not()).
    {
        combo_tree tr(vertex(id::logical_not));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(2)));
        tr.append_child(root, vertex(argument(1)));
        rc(tr, root);
        cout << quote("reorder_commutative_multi_child_non_commutative") << ": " << quote(show(tr))
             << "," << endl;
    }

    // ---- remove_null_vertices ----
    // (built by hand -- combo's own str2combo_tree parser has no null_vertex
    // literal token, so this constructs the tree directly.)
    {
        combo_tree tr(vertex(id::logical_and));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(id::null_vertex));
        tr.append_child(root, vertex(argument(2)));
        rnv(tr, root);
        cout << quote("remove_null_vertices_plain") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        // cond takes (condition, consequent) pairs plus a final else clause;
        // a null_vertex condition drops BOTH it and its paired consequent.
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(id::null_vertex)); // condition 1 (dropped)
        tr.append_child(root, vertex(argument(1)));      // consequent 1 (dropped too)
        tr.append_child(root, vertex(argument(2)));       // condition 2 (kept)
        tr.append_child(root, vertex(argument(3)));       // consequent 2 (kept)
        tr.append_child(root, vertex(argument(4)));       // else clause (kept)
        rnv(tr, root);
        cout << quote("remove_null_vertices_cond") << ": " << quote(show(tr)) << endl;
    }

    cout << "}" << endl;
    return 0;
}
