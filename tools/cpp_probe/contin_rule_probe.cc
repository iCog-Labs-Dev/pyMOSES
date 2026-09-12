// tools/cpp_probe/contin_rule_probe.cc
//
// S14 Task 4 oracle probe: apply exactly ONE named contin_rules.cc /
// branch_rules.cc rule, in isolation, pre-order (`downwards(rule())`)
// over a parsed combo tree, and print the resulting tree text. Used to
// validate pymoses's own per-rule unit tests against the real compiled
// libcomboreduct.so, one rule at a time -- NOT a substitute for the
// full contin_reduction/mixed_reduction end-to-end oracle (see
// contin_mixed_reduct_probe.cc for that).
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/contin_rule_probe \
//       tools/cpp_probe/contin_rule_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 \
//     tools/cpp_probe/build/contin_rule_probe <rule_name> "<tree>" )
//
// <rule_name> is one of: reduce_plus_zero, reduce_times_one_zero,
// reduce_factorize_fraction, reduce_factorize, reduce_invert_constant,
// reduce_fraction, reduce_times_div, reduce_plus_times_one_child,
// reduce_sum_log, reduce_log_div_times, reduce_exp_times, reduce_exp_div,
// reduce_exp_log, reduce_sin, reduce_cond_else, reduce_cond_const,
// reduce_cond_adjacent.

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/reduct/meta_rules.h>
#include <moses/comboreduct/reduct/contin_rules.h>
#include <moses/comboreduct/reduct/branch_rules.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

static string show(combo_tree& tr)
{
    ostringstream oss;
    ostream_combo_tree(oss, tr);
    string s = oss.str();
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << "usage: " << argv[0] << " <rule_name> \"<tree>\"" << endl;
        return 2;
    }
    string rule_name = argv[1];
    stringstream ss(argv[2]);
    combo_tree tr;
    ss >> tr;

    try {
#define APPLY(NAME, EXPR) if (rule_name == NAME) { EXPR; }
        if (rule_name == "reduce_plus_zero") { reduce_plus_zero r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_times_one_zero") { reduce_times_one_zero r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_factorize_fraction") { reduce_factorize_fraction r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_factorize") { reduce_factorize r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_invert_constant") { reduce_invert_constant r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_fraction") { reduce_fraction r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_times_div") { reduce_times_div r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_plus_times_one_child") { reduce_plus_times_one_child r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_sum_log") { reduce_sum_log r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_log_div_times") { reduce_log_div_times r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_exp_times") { reduce_exp_times r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_exp_div") { reduce_exp_div r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_exp_log") { reduce_exp_log r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_sin") { reduce_sin r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_cond_else") { reduce_cond_else r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_cond_const") { reduce_cond_const r; downwards d(r); d(tr, tr.begin()); }
        else if (rule_name == "reduce_cond_adjacent") { reduce_cond_adjacent r; downwards d(r); d(tr, tr.begin()); }
        else { cerr << "unknown rule: " << rule_name << endl; return 2; }
#undef APPLY
    } catch (std::exception& e) {
        cout << "EXCEPTION: " << e.what() << endl;
        return 0;
    }
    cout << show(tr) << endl;
    return 0;
}
