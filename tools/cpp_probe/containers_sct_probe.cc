// tools/cpp_probe/containers_sct_probe.cc
//
// Measures REAL classic ordering/deduplication behaviour of
// scored_combo_tree_set/_tset/_ptr_set and their four functors
// (sct_score_greater, sct_tree_greater, scored_combo_tree_hash,
// scored_combo_tree_equal) for pyMoses's S10 Task 0 port (types.py).
//
// The interesting case this probe exists to measure, not infer: whether
// boost::ptr_set really DISCARDS a candidate when score, raw score, tree
// size AND tree lexicographic order all tie simultaneously -- exactly the
// scenario types.cc:86-88's own comment warns about ("we are desperate to
// break a tie ... otherwise ... the scored_combo_tree_ptr_set will discard
// anything that compares equal"). A contin_t NaN vs. a finite contin_t
// leaf produces exactly this: IEEE-754 NaN comparisons are false in BOTH
// directions, so size_tree_order<vertex> (which lexicographic_subtree_
// order falls back to for equal-size trees) ties too.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/containers_sct_probe \
//       tools/cpp_probe/containers_sct_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/containers_sct_probe )
//
// Build/run through tools/capture_containers_sct.py.
#include <moses/moses/moses/types.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace opencog::combo;
using namespace std;

static string show(const combo_tree& tr)
{
    ostringstream oss;
    combo_tree t = tr;
    ostream_combo_tree(oss, t);
    return oss.str();
}

static string quote(const string& s)
{
    ostringstream oss;
    oss << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') oss << '\\';
        oss << c;
    }
    oss << '"';
    return oss.str();
}

static scored_combo_tree mk(const combo_tree& tr, score_t score, complexity_t complexity,
                            score_t cpen = 0.0f, score_t upen = 0.0f)
{
    return scored_combo_tree(tr, demeID_t(), composite_score(score, complexity, cpen, upen));
}

static const scored_combo_tree& find_by_name(
        const vector<pair<string, scored_combo_tree>>& table, const string& name)
{
    for (auto& p : table)
        if (p.first == name) return p.second;
    throw std::runtime_error("not found: " + name);
}

int main()
{
    cout << unitbuf;
    const double nan = numeric_limits<double>::quiet_NaN();

    vertex v_and(id::logical_and);
    vertex v_or(id::logical_or);
    vertex v_arg1 = vertex(argument(1));
    vertex v_arg2 = vertex(argument(2));
    vertex v_nan = vertex(contin_t(nan));
    vertex v_five = vertex(contin_t(5.0));

    combo_tree t_and(v_and);      // size 1
    combo_tree t_or(v_or);        // size 1
    combo_tree t_arg1(v_arg1);    // size 1
    combo_tree t_arg2(v_arg2);    // size 1
    combo_tree t_nan(v_nan);      // size 1, contin leaf
    combo_tree t_five(v_five);    // size 1, contin leaf

    combo_tree t_and2(v_and);   // size 2 (and($1))
    t_and2.append_child(t_and2.begin(), vertex(argument(1)));

    vector<pair<string, scored_combo_tree>> table;
    table.push_back(make_pair(string("a-worse-score"), mk(t_and, -10.0f, 0)));
    table.push_back(make_pair(string("b-better-score"), mk(t_and, -2.0f, 0)));
    table.push_back(make_pair(string("c-tie-score-high-complexity"), mk(t_and, -5.0f, 10)));
    table.push_back(make_pair(string("d-tie-score-low-complexity"), mk(t_and, -5.0f, 3)));
    table.push_back(make_pair(string("e-tie-everything-bigger-tree"), mk(t_and2, -5.0f, 3)));
    table.push_back(make_pair(string("f-tie-everything-arg1"), mk(t_arg1, -5.0f, 3)));
    table.push_back(make_pair(string("g-tie-everything-arg2"), mk(t_arg2, -5.0f, 3)));
    table.push_back(make_pair(string("h-tie-everything-nan-leaf"), mk(t_nan, -5.0f, 3)));
    table.push_back(make_pair(string("i-tie-everything-five-leaf"), mk(t_five, -5.0f, 3)));

    sct_score_greater score_gt;
    sct_tree_greater tree_gt;

    cout << "{\n";

    cout << "  \"names\": [";
    for (size_t i = 0; i < table.size(); ++i) {
        if (i) cout << ',';
        cout << quote(table[i].first);
    }
    cout << "],\n";

    cout << "  \"sct_score_greater_pairs\": [";
    bool first = true;
    for (size_t i = 0; i < table.size(); ++i) {
        for (size_t j = 0; j < table.size(); ++j) {
            if (i == j) continue;
            if (!first) cout << ',';
            first = false;
            bool gt = score_gt(table[i].second, table[j].second);
            cout << "\n    {\"a\":" << quote(table[i].first) << ",\"b\":" << quote(table[j].first)
                 << ",\"gt\":" << (gt ? "true" : "false") << "}";
        }
    }
    cout << "\n  ],\n";

    cout << "  \"sct_tree_greater_pairs\": [";
    first = true;
    for (size_t i = 0; i < table.size(); ++i) {
        for (size_t j = 0; j < table.size(); ++j) {
            if (i == j) continue;
            if (!first) cout << ',';
            first = false;
            bool gt = tree_gt(table[i].second, table[j].second);
            cout << "\n    {\"a\":" << quote(table[i].first) << ",\"b\":" << quote(table[j].first)
                 << ",\"gt\":" << (gt ? "true" : "false") << "}";
        }
    }
    cout << "\n  ],\n";

    // --- scored_combo_tree_ptr_set: does the full-tie NaN/five pair get
    // deduplicated (one discarded)? -----------------------------------
    {
        vector<string> insert_order = {
            "h-tie-everything-nan-leaf", "i-tie-everything-five-leaf",
            "a-worse-score", "b-better-score", "c-tie-score-high-complexity"};
        scored_combo_tree_ptr_set pset;
        vector<bool> results;
        for (auto& name : insert_order) {
            auto result = pset.insert(new scored_combo_tree(find_by_name(table, name)));
            results.push_back(result.second);
        }
        cout << "  \"ptr_set_insert_order\": [";
        for (size_t i = 0; i < insert_order.size(); ++i) {
            if (i) cout << ',';
            cout << quote(insert_order[i]);
        }
        cout << "],\n";
        cout << "  \"ptr_set_insert_results\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) cout << ',';
            cout << (results[i] ? "true" : "false");
        }
        cout << "],\n";
        cout << "  \"ptr_set_size\": " << pset.size() << ",\n";
        cout << "  \"ptr_set_order\": [";
        first = true;
        for (scored_combo_tree_ptr_set::const_iterator it = pset.begin(); it != pset.end(); ++it) {
            if (!first) cout << ',';
            first = false;
            cout << "{\"tree\":" << quote(show(it->get_tree()))
                 << ",\"score\":" << it->get_score() << "}";
        }
        cout << "],\n";
    }

    // --- scored_combo_tree_tset: same full-tie pair, tree-only comparator
    {
        vector<string> insert_order = {
            "h-tie-everything-nan-leaf", "i-tie-everything-five-leaf",
            "a-worse-score", "f-tie-everything-arg1", "g-tie-everything-arg2"};
        scored_combo_tree_tset tset;
        vector<bool> results;
        for (auto& name : insert_order) {
            auto result = tset.insert(new scored_combo_tree(find_by_name(table, name)));
            results.push_back(result.second);
        }
        cout << "  \"tset_insert_order\": [";
        for (size_t i = 0; i < insert_order.size(); ++i) {
            if (i) cout << ',';
            cout << quote(insert_order[i]);
        }
        cout << "],\n";
        cout << "  \"tset_insert_results\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) cout << ',';
            cout << (results[i] ? "true" : "false");
        }
        cout << "],\n";
        cout << "  \"tset_size\": " << tset.size() << ",\n";
    }

    // --- scored_combo_tree_set (unordered, hash+equal by TREE ONLY) -----
    {
        scored_combo_tree_set uset;
        auto r1 = uset.insert(mk(t_and, -1.0f, 3));
        auto r2 = uset.insert(mk(t_and, -9.0f, 7));   // same tree text, different score
        auto r3 = uset.insert(mk(t_or, -1.0f, 3));    // different tree
        cout << "  \"unordered_set_insert_results\": ["
             << (r1.second ? "true" : "false") << ","
             << (r2.second ? "true" : "false") << ","
             << (r3.second ? "true" : "false") << "],\n";
        cout << "  \"unordered_set_size\": " << uset.size() << "\n";
    }

    cout << "}\n";
    return 0;
}
