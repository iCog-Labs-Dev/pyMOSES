// tools/cpp_probe/assumption_probe.cc
//
// Measures insert_assumption's REAL behaviour (not its header comment)
// for review of pymetta/comboreduct/combo/assumption.py.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/assumption_probe tools/cpp_probe/assumption_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil

#include <moses/comboreduct/combo/assumption.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>

#include <iostream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

static void dump_assumptions(combo_tree& tr, const string& label)
{
    typedef combo_tree::sibling_iterator sib_it;
    cout << label << ": [";
    sib_it a = tr.next_sibling(tr.begin());
    bool first = true;
    while (tr.is_valid(a)) {
        if (!first) cout << ", ";
        first = false;
        combo_tree sub(a);
        ostream_combo_tree(cout, sub);
        ++a;
    }
    cout << "]" << endl;
}

static void run_case(const string& label, const vector<string>& insert_order)
{
    combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
    for (const string& s : insert_order) {
        combo_tree a = str2combo_tree(s, vector<string>());
        insert_assumption(tr, a.begin());
    }
    dump_assumptions(tr, label);
}

int main()
{
    run_case("case1 insert true,true", {"true", "true"});
    run_case("case2 insert $1,$2,$3", {"$1", "$2", "$3"});
    run_case("case3 insert $3,$2,$1", {"$3", "$2", "$1"});
    run_case("case4 insert $3,$4,$3(dup)", {"$3", "$4", "$3"});
    run_case("case5 insert true,true,true", {"true", "true", "true"});
    return 0;
}
