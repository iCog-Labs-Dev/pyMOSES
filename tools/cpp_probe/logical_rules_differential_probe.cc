// tools/cpp_probe/logical_rules_differential_probe.cc
//
// Differential oracle for pymoses's comboreduct/reduct/logical_rules.py
// (Task 4 review, Important: "land the differential harness in-repo" --
// both Criticals in that review were found by randomized differential
// testing, neither reachable from the hand-picked ENF corpus, and the
// reviewer's own scratchpad artifacts were lost when the session was
// stopped; this is that harness, rebuilt and committed).
//
// Reads combo-tree text-format strings, one per line, from a file (see
// gen_boolean_trees.py, which generates that file) and for EACH one runs
// two pipelines, printing one JSON-lines object per input:
//
//   "raw"   -- subtree_to_enf() ALONE, no pre-canonicalization. Directly
//              exercises reduce_to_enf's own state machine with no
//              typed/untyped confound (subtree_to_enf contains no
//              downwards/upwards of its own).
//   "canon" -- upwards(eval_logical_identities()); downwards(level());
//              subtree_to_enf() -- matches logical_rules.py's own `_enf`
//              test helper and this probe's own `run_enf`.
//
// Malformed/crashing inputs are not expected (the generator only emits
// well-formed and/or/not-free boolean trees), but if str2combo_tree or a
// rule throws, this reports it as an error string rather than crashing
// the whole run, so one bad input does not lose the rest of the corpus.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/logical_rules_differential_probe \
//       tools/cpp_probe/logical_rules_differential_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 60 \
//     tools/cpp_probe/build/logical_rules_differential_probe corpus.txt )
//
// See run_differential.py for the end-to-end orchestration (generate the
// corpus, build+run this probe, run the same corpus through the Python
// port, compare, report).

#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/reduct/meta_rules.h>
#include <moses/comboreduct/reduct/general_rules.h>
#include <moses/comboreduct/reduct/logical_rules.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>

#include <fstream>
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
    string s = oss.str();
    // Trim the trailing space ostream_combo_tree always appends, to match
    // the Python side's own .strip().
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n'))
        s.pop_back();
    return s;
}

static string json_quote(const string& s)
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

static string run_raw(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    subtree_to_enf enf;
    enf(tr, it);
    return show(tr);
}

static string run_canon(const string& prog)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    combo_tree::iterator it = tr.begin();
    upwards up1{eval_logical_identities()};
    up1(tr, it);
    downwards dn1{level()};
    dn1(tr, it);
    subtree_to_enf enf;
    enf(tr, it);
    return show(tr);
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " <corpus-file>" << endl;
        return 2;
    }

    ifstream in(argv[1]);
    if (!in) {
        cerr << "cannot open " << argv[1] << endl;
        return 2;
    }

    string line;
    cout << "[" << endl;
    bool first = true;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (!first) cout << "," << endl;
        first = false;

        string raw_result, canon_result;
        bool raw_ok = true, canon_ok = true;
        try {
            raw_result = run_raw(line);
        } catch (const std::exception& e) {
            raw_result = string("ERROR: ") + e.what();
            raw_ok = false;
        } catch (...) {
            raw_result = "ERROR: unknown exception";
            raw_ok = false;
        }
        try {
            canon_result = run_canon(line);
        } catch (const std::exception& e) {
            canon_result = string("ERROR: ") + e.what();
            canon_ok = false;
        } catch (...) {
            canon_result = "ERROR: unknown exception";
            canon_ok = false;
        }

        cout << "{\"input\": " << json_quote(line)
             << ", \"raw\": " << json_quote(raw_result)
             << ", \"raw_ok\": " << (raw_ok ? "true" : "false")
             << ", \"canon\": " << json_quote(canon_result)
             << ", \"canon_ok\": " << (canon_ok ? "true" : "false")
             << "}";
    }
    cout << endl << "]" << endl;
    return 0;
}
