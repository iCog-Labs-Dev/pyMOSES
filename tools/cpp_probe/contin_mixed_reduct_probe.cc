// tools/cpp_probe/contin_mixed_reduct_probe.cc
//
// S14 Task 4's end-to-end oracle probe: full-pipeline `contin_reduce`/
// `mixed_reduce` output, byte-identical comparison against pymoses's own
// contin_reduction.py/mixed_reduction.py.
//
// $CLASSIC's own tests/comboreduct/reduct/test_reduct.cc (the pre-existing,
// already-built "test_reduct" binary this project's plan document expected
// to reuse here) does NOT drive contin/mixed reduction at all -- verified
// directly by reading its source: it calls ONLY `logical_reduce(reduce_
// arity, tr, ignore_ops)`, and its `complete_truth_table` unconditionally
// assumes a boolean-typed tree (confirmed live: feeding it a `+(...)` tree
// throws "boolean_interpreter does not handle builtin +"). Its `reduce_
// arity` CLI argument is `logical_reduce`'s own boolean effort level (0-3),
// not a contin/mixed selector. So this probe exists in place of extending
// test_reduct.cc -- mirrors reduct_effort_probe.cc's own JSON-lines corpus
// convention instead (Task 6's oracle for logical_reduce).
//
// Reads combo-tree text-format strings, one per line, from a file and for
// the given <mode> ("contin", "mixed" or "full") and (contin only)
// <effort>, runs contin_reduce/mixed_reduce/full_reduce with an EMPTY
// ignore_ops (classic's own real default -- CLAUDE.md's own §5.4 lesson:
// "at least one oracle must run classic's own defaults") over every line,
// printing one JSON-lines object per input.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/contin_mixed_reduct_probe \
//       tools/cpp_probe/contin_mixed_reduct_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 \
//     tools/cpp_probe/build/contin_mixed_reduct_probe corpus.txt contin 2 )
//   ( ulimit -v 2097152; timeout 300 \
//     tools/cpp_probe/build/contin_mixed_reduct_probe corpus.txt mixed )

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/reduct/reduct.h>

#include <cstdlib>
#include <fstream>
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
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <corpus-file> contin|mixed [effort (contin only)]" << endl;
        return 2;
    }
    string mode = argv[2];
    int effort = 2;
    if (mode == "contin" && argc >= 4) effort = atoi(argv[3]);

    ifstream in(argv[1]);
    if (!in) {
        cerr << "cannot open " << argv[1] << endl;
        return 2;
    }

    cout << "[" << endl;
    bool first = true;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (!first) cout << "," << endl;
        first = false;

        cout << "{\"input\": " << json_quote(line) << ", \"mode\": " << json_quote(mode);
        if (mode == "contin") cout << ", \"effort\": " << effort;

        combo_tree tr;
        bool parse_ok = true;
        try {
            stringstream ss(line);
            ss >> tr;
        } catch (std::exception& e) {
            parse_ok = false;
            cout << ", \"parse_error\": " << json_quote(e.what());
        }

        if (parse_ok) {
            unsigned size_before = tr.size();
            bool ok = true;
            string error_msg;
            try {
                vertex_set ignore_ops;  // classic's own real default: empty
                if (mode == "contin") {
                    contin_reduce(tr, effort, ignore_ops);
                } else if (mode == "mixed") {
                    mixed_reduce(tr);
                } else {
                    full_reduce(tr);
                }
            } catch (std::exception& e) {
                ok = false;
                error_msg = e.what();
            }
            cout << ", \"ok\": " << (ok ? "true" : "false");
            if (ok) {
                cout << ", \"reduced\": " << json_quote(show(tr))
                     << ", \"size_before\": " << size_before
                     << ", \"size_after\": " << tr.size();
            } else {
                cout << ", \"error\": " << json_quote(error_msg);
            }
        }
        cout << "}";
    }
    cout << endl << "]" << endl;
    return 0;
}
