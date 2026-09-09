// tools/cpp_probe/mixed_assumption_probe.cc
//
// S14 Task 4: four hand-picked mixed_reduce cases that exercise the
// assumption-set-aware rules (reduce_from_assumptions, reduce_gt_zero_sum
// drawing on assumptions, negated-assumption handling) -- these need
// `insert_assumption`, which combo-tree text syntax cannot express, so
// they are a separate probe from contin_mixed_reduct_probe.cc's plain-text
// corpus. Prints a JSON array, one object per case, to stdout.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/mixed_assumption_probe \
//       tools/cpp_probe/mixed_assumption_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/mixed_assumption_probe )

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/combo/assumption.h>
#include <moses/comboreduct/reduct/reduct.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

// Prints ONLY the main conclusion (tr's own root subtree), never the
// assumption siblings real C++ attaches as extra top-level roots under
// head/feet -- pymetta's own combo_tree_to_str has nothing analogous to
// print (assumption.py's own module docstring: "no forests -> plain
// list", assumptions kept in a side list, never printed as tree
// content), so a fair byte-identical comparison needs the SAME
// main-tree-only text on both sides. `combo_tree(tr.begin())` (the
// iterator constructor) copies exactly tr's own root subtree, discarding
// every assumption sibling.
static string show(combo_tree& tr)
{
    combo_tree main_only(tr.begin());
    ostringstream oss;
    ostream_combo_tree(oss, main_only);
    string s = oss.str();
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

static string json_quote(const string& s)
{
    ostringstream oss;
    oss << "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') oss << '\\';
        else oss << c;
    }
    oss << "\"";
    return oss.str();
}

struct Case {
    string label;
    string main_tree;
    vector<string> assumptions;
};

int main()
{
    vector<Case> cases = {
        {"trivial_self_implication", "0<($1)", {"0<($1)"}},
        {"assumption_x_gt_1_leaves_x_gt_0_term_reduced",
         "0<(+($1 -1.0))", {"0<($1)"}},
        {"gt_zero_sum_from_two_positive_assumptions",
         "0<(+($1 $2))", {"0<($1)", "0<($2)"}},
        {"negated_assumption_forces_false", "0<($1)", {"not(0<($1))"}},
    };

    cout << "[" << endl;
    bool first = true;
    for (auto& c : cases) {
        if (!first) cout << "," << endl;
        first = false;

        combo_tree tr;
        { stringstream ss(c.main_tree); ss >> tr; }
        for (auto& a : c.assumptions) {
            combo_tree atr;
            stringstream ss(a);
            ss >> atr;
            insert_assumption(tr, atr.begin());
        }
        mixed_reduce(tr);

        cout << "{\"label\": " << json_quote(c.label)
             << ", \"main_tree\": " << json_quote(c.main_tree)
             << ", \"assumptions\": [";
        for (size_t i = 0; i < c.assumptions.size(); ++i) {
            if (i) cout << ", ";
            cout << json_quote(c.assumptions[i]);
        }
        cout << "], \"reduced\": " << json_quote(show(tr)) << "}";
    }
    cout << endl << "]" << endl;
    return 0;
}
