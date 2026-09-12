// tools/cpp_probe/partition_probe.cc
//
// Measures the REAL child-reordering behaviour of opencog::tree<T>::partition
// and ::sort_on_subtrees for pymoses's util/tree.py port (task-3, needed by
// general_rules.py's eval_constants and reorder_commutative). Ground truth,
// not inference from reading tree.h alone -- partition's two-pointer swap
// scan is not a textbook stable partition (verified below).
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/partition_probe tools/cpp_probe/partition_probe.cc
//
// (header-only against tree.h -- no link libs needed, same as tree_probe.cc)
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/partition_probe )
#define IGNORE_OC_ASSERT
#include <opencog/util/tree.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
typedef tree<std::string> stree;

static std::string quote(const std::string &s) { return "\"" + s + "\""; }

static std::string str_array(const std::vector<std::string> &xs)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ",";
        os << quote(xs[i]);
    }
    os << "]";
    return os.str();
}

static std::vector<std::string> dump_children(stree &t)
{
    std::vector<std::string> out;
    for (stree::sibling_iterator sib = t.begin(t.begin()); sib != t.end(t.begin()); ++sib)
        out.push_back(*sib);
    return out;
}

// pred: is this a constant, marked by an upper-case first letter (matches
// this probe's own convention below: "C..." = constant, "V..." = variable).
static bool is_const_str(const std::string &s) { return !s.empty() && s[0] == 'C'; }

int main()
{
    std::vector<std::pair<std::string, std::string>> cases;

    // -- partition case A: C1 C2 V1 C3 V2 --
    {
        stree t("root");
        for (const char *s : {"C1", "C2", "V1", "C3", "V2"})
            t.append_child(t.begin(), std::string(s));
        stree::sibling_iterator to =
            t.partition(t.begin(t.begin()), t.end(t.begin()), is_const_str);
        std::string boundary = (to == t.end(t.begin())) ? "<end>" : *to;
        std::ostringstream os;
        os << "{\"result\":" << str_array(dump_children(t)) << ",\"boundary\":" << quote(boundary) << "}";
        cases.push_back({"partition_case_a", os.str()});
    }

    // -- partition case B: C1 V1 C2 V2 C3 (the case worked out by hand) --
    {
        stree t("root");
        for (const char *s : {"C1", "V1", "C2", "V2", "C3"})
            t.append_child(t.begin(), std::string(s));
        stree::sibling_iterator to =
            t.partition(t.begin(t.begin()), t.end(t.begin()), is_const_str);
        std::string boundary = (to == t.end(t.begin())) ? "<end>" : *to;
        std::ostringstream os;
        os << "{\"result\":" << str_array(dump_children(t)) << ",\"boundary\":" << quote(boundary) << "}";
        cases.push_back({"partition_case_b", os.str()});
    }

    // -- partition case C: all constants --
    {
        stree t("root");
        for (const char *s : {"C1", "C2", "C3"})
            t.append_child(t.begin(), std::string(s));
        stree::sibling_iterator to =
            t.partition(t.begin(t.begin()), t.end(t.begin()), is_const_str);
        std::string boundary = (to == t.end(t.begin())) ? "<end>" : *to;
        std::ostringstream os;
        os << "{\"result\":" << str_array(dump_children(t)) << ",\"boundary\":" << quote(boundary) << "}";
        cases.push_back({"partition_case_c", os.str()});
    }

    // -- partition case D: no constants at all --
    {
        stree t("root");
        for (const char *s : {"V1", "V2", "V3"})
            t.append_child(t.begin(), std::string(s));
        stree::sibling_iterator to =
            t.partition(t.begin(t.begin()), t.end(t.begin()), is_const_str);
        std::string boundary = (to == t.end(t.begin())) ? "<end>" : *to;
        std::ostringstream os;
        os << "{\"result\":" << str_array(dump_children(t)) << ",\"boundary\":" << quote(boundary) << "}";
        cases.push_back({"partition_case_d", os.str()});
    }

    // -- partition case E: single child, constant --
    {
        stree t("root");
        t.append_child(t.begin(), std::string("C1"));
        stree::sibling_iterator to =
            t.partition(t.begin(t.begin()), t.end(t.begin()), is_const_str);
        std::string boundary = (to == t.end(t.begin())) ? "<end>" : *to;
        std::ostringstream os;
        os << "{\"result\":" << str_array(dump_children(t)) << ",\"boundary\":" << quote(boundary) << "}";
        cases.push_back({"partition_case_e", os.str()});
    }

    // -- sort_on_subtrees: string length order, with a genuine tie (two
    // equal-length children) to observe whether relative order among ties
    // is preserved (stable) or not. --
    {
        stree t("root");
        for (const char *s : {"bb", "a", "cc", "ddd"})
            t.append_child(t.begin(), std::string(s));
        auto by_len = [](stree::iterator_base x, stree::iterator_base y) { return (*x).size() < (*y).size(); };
        t.sort_on_subtrees(t.begin(t.begin()), t.end(t.begin()), by_len, false);
        cases.push_back({"sort_on_subtrees_ties", "{\"result\":" + str_array(dump_children(t)) + "}"});
    }

    // -- sort_on_subtrees deep=true: sort grandchildren first, then children
    // -- to observe recursion order/effect. --
    {
        stree t("root");
        stree::iterator p = t.append_child(t.begin(), std::string("P"));
        for (const char *s : {"z", "y", "x"})
            t.append_child(p, std::string(s));
        stree::iterator q = t.append_child(t.begin(), std::string("Q"));
        for (const char *s : {"n", "m"})
            t.append_child(q, std::string(s));
        auto by_val = [](stree::iterator_base x, stree::iterator_base y) { return (*x) < (*y); };
        t.sort_on_subtrees(t.begin(t.begin()), t.end(t.begin()), by_val, true);

        std::vector<std::string> top;
        for (stree::sibling_iterator sib = t.begin(t.begin()); sib != t.end(t.begin()); ++sib)
            top.push_back(*sib);
        std::vector<std::string> p_children, q_children;
        for (stree::sibling_iterator sib = p.begin(); sib != p.end(); ++sib)
            p_children.push_back(*sib);
        for (stree::sibling_iterator sib = q.begin(); sib != q.end(); ++sib)
            q_children.push_back(*sib);

        std::ostringstream os;
        os << "{\"top\":" << str_array(top) << ",\"p_children\":" << str_array(p_children)
           << ",\"q_children\":" << str_array(q_children) << "}";
        cases.push_back({"sort_on_subtrees_deep", os.str()});
    }

    printf("{\n");
    for (size_t i = 0; i < cases.size(); ++i) {
        printf("  %s: %s%s\n", quote(cases[i].first).c_str(), cases[i].second.c_str(),
               (i + 1 < cases.size()) ? "," : "");
    }
    printf("}\n");
    return 0;
}
