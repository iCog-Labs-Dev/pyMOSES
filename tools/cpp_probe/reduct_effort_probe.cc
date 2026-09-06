// tools/cpp_probe/reduct_effort_probe.cc
//
// Oracle probe for Task 6's exit criterion: at logical-reduction efforts
// 1 (simple), 2 (medium) and 3 (complexe), pyMeTTa's reduced tree must be
// byte-identical to classic MOSES's own `logical_reduce()` output, and
// every reduction must preserve the boolean tree's truth table.
//
// Mirrors $CLASSIC/tests/comboreduct/reduct/test_reduct.cc (a local,
// dev-only fork of the upstream cxxtest probe -- see also
// moses/comboreduct/main/logical-reductor.cc, which hardcodes effort=3):
// same `logical_reduce(effort, tr, ignore_ops)` call and the same
// before/after complete_truth_table comparison, made batch- and
// effort-parametric, with JSON-lines output instead of test_reduct's
// human-readable printf block.
//
// Effort levels ($CLASSIC's moses/comboreduct/reduct/logical_reduction.cc,
// logical_reduction::operator()(int)): 0 extra_simple, 1 simple, 2 medium
// (requires ignore_ops -- always empty here, matching every corpus source
// used by reduct_corpus.py), 3 complexe. Task 6's exit criterion exercises
// 1, 2 and 3 only; effort 0 is out of scope and not captured.
//
// Reads combo-tree text-format strings, one per line, from a file (see
// reduct_corpus.py, which writes that file) and for the single effort
// given on argv[2], runs logical_reduce over every line and prints one
// JSON-lines object per input into a JSON array.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/reduct_effort_probe \
//       tools/cpp_probe/reduct_effort_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 \
//     tools/cpp_probe/build/reduct_effort_probe corpus.txt 3 )

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <opencog/util/numeric.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

// Preserve ostream_combo_tree's own trailing space verbatim -- this
// fixture is meant to be byte-compared against pymetta's
// combo_tree_to_str, which reproduces that exact trailing space (see
// tests/fixtures/combo_roundtrip.json's committed entries). Do NOT
// `.strip()` here, unlike logical_rules_differential_probe.cc's `show()`
// (that probe feeds a different, already-.strip()'d Python comparison).
static string show(combo_tree& tr)
{
    ostringstream oss;
    ostream_combo_tree(oss, tr);
    return oss.str();
}

static string json_quote(const string& s)
{
    ostringstream oss;
    oss << "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') oss << '\\';
        else if (c == '\n') { oss << "\\n"; continue; }
        oss << c;
    }
    oss << "\"";
    return oss.str();
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << "usage: " << argv[0] << " <corpus-file> <effort 1|2|3>" << endl;
        return 2;
    }

    ifstream in(argv[1]);
    if (!in) {
        cerr << "cannot open " << argv[1] << endl;
        return 2;
    }
    int effort = atoi(argv[2]);

    // Empty ignore_ops throughout: every corpus source used by
    // reduct_corpus.py is a pure and/or/not/$n boolean tree with no
    // predicates to ignore -- same as test_reduct.cc, BooleanReductUTest
    // and BooleanFullUTest's own `vertex_set ignore_ops;`.
    vertex_set ignore_ops;

    string line;
    cout << "[" << endl;
    bool first = true;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (!first) cout << "," << endl;
        first = false;

        bool ok = true;
        string error, reduced;
        long size_before = -1, size_after = -1;
        bool tt_preserved = false;

        try {
            stringstream ss(line);
            combo_tree tr;
            ss >> tr;

            complete_truth_table tt1(tr);
            size_before = (long)tr.size();

            logical_reduce(effort, tr, ignore_ops);

            size_after = (long)tr.size();
            complete_truth_table tt2(tr, integer_log2(tt1.size()));
            tt_preserved = (tt1 == tt2);
            reduced = show(tr);
        } catch (const std::exception& e) {
            ok = false;
            error = string("ERROR: ") + e.what();
        } catch (...) {
            ok = false;
            error = "ERROR: unknown exception";
        }

        cout << "{\"input\": " << json_quote(line)
             << ", \"effort\": " << effort
             << ", \"ok\": " << (ok ? "true" : "false");
        if (ok) {
            cout << ", \"reduced\": " << json_quote(reduced)
                 << ", \"size_before\": " << size_before
                 << ", \"size_after\": " << size_after
                 << ", \"tt_preserved\": " << (tt_preserved ? "true" : "false");
        } else {
            cout << ", \"error\": " << json_quote(error);
        }
        cout << "}";
    }
    cout << endl << "]" << endl;
    return 0;
}
