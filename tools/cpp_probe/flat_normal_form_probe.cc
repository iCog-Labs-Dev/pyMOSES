// tools/cpp_probe/flat_normal_form_probe.cc
//
// Measures the REAL behaviour of opencog::reduct::nf_mapper<vertex> and its
// free functions (tautology, subset_eq, subset, implications) for pymetta's
// comboreduct/reduct/flat_normal_form.py port (task-3). Ground truth, not
// inference -- particularly clause ORDERING (std::set<int,
// absolute_value_order<int>>) for a clause containing both a literal and
// its negation, and the id-assignment order of nf_mapper::add_item.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/flat_normal_form_probe tools/cpp_probe/flat_normal_form_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/flat_normal_form_probe )
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

static string show(combo_tree& tr)
{
    ostringstream oss;
    ostream_combo_tree(oss, tr);
    return oss.str();
}

static string clause_str(const clause& c)
{
    ostringstream oss;
    oss << c;
    return oss.str();
}

static string nf_str(const nf& f)
{
    ostringstream oss;
    bool first = true;
    for (const clause& c : f) {
        if (!first) oss << " ";
        first = false;
        oss << c;
    }
    return oss.str();
}

int main()
{
    cout << "{" << endl;

    // ---- clause ordering: a set built by hand, mixing $1 and !$1 ----
    {
        clause c;
        c.insert(3);
        c.insert(-1);
        c.insert(2);
        c.insert(1);
        c.insert(-2);
        c.insert(-3);
        cout << quote("clause_ordering_mixed") << ": " << quote(clause_str(c)) << "," << endl;
    }
    {
        // does the set consider 1 and -1 the SAME key (as a std::set with
        // absolute_value_order would, since neither compares less than the
        // other's negation under that specific formula's *reflexive* case)?
        // -- No: (a==-b) branch only fires for the TIE-BREAK; a==-b with
        // a=1,b=-1 gives a<b (1<-1 = false) meaning 1 is NOT less than -1,
        // and separately -1 IS less than 1, so they are NOT equivalent
        // (exactly one is less than the other) -- both a distinct elements.
        clause c;
        c.insert(1);
        c.insert(-1);
        cout << quote("clause_distinct_pos_neg") << ": " << c.size() << "," << endl;
    }
    {
        // tautology: does a clause containing both p and !p get flagged?
        clause c1;
        c1.insert(1);
        c1.insert(-1);
        c1.insert(2);
        clause c2;
        c2.insert(1);
        c2.insert(2);
        cout << quote("tautology_with_pair") << ": " << (tautology(c1) ? "true" : "false") << "," << endl;
        cout << quote("tautology_without_pair") << ": " << (tautology(c2) ? "true" : "false") << "," << endl;
    }
    {
        // subset_eq / subset
        clause c1;
        c1.insert(1);
        c1.insert(2);
        clause c2;
        c2.insert(-1);
        c2.insert(1);
        c2.insert(2);
        c2.insert(3);
        cout << quote("subset_eq_true") << ": " << (subset_eq(c1, c2) ? "true" : "false") << "," << endl;
        cout << quote("subset_strict_true") << ": " << (subset(c1, c2) ? "true" : "false") << "," << endl;
        cout << quote("subset_eq_equal_not_strict") << ": " << (subset(c1, c1) ? "true" : "false") << ","
             << endl;
        cout << quote("subset_eq_equal_is_eq") << ": " << (subset_eq(c1, c1) ? "true" : "false") << ","
             << endl;
    }
    {
        // implications: (p X)(!p Y) => (X Y)
        clause c1;
        c1.insert(1);
        c1.insert(2);
        clause c2;
        c2.insert(-1);
        c2.insert(3);
        vector<clause> impls;
        implications(c1, c2, back_inserter(impls));
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < impls.size(); ++i) {
            if (i) oss << ",";
            oss << quote(clause_str(impls[i]));
        }
        oss << "]";
        cout << quote("implications_simple") << ": " << oss.str() << "," << endl;
    }
    {
        // implications: multiple complementary pairs between the two clauses
        clause c1;
        c1.insert(1);
        c1.insert(2);
        clause c2;
        c2.insert(-1);
        c2.insert(-2);
        c2.insert(4);
        vector<clause> impls;
        implications(c1, c2, back_inserter(impls));
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < impls.size(); ++i) {
            if (i) oss << ",";
            oss << quote(clause_str(impls[i]));
        }
        oss << "]";
        cout << quote("implications_multi") << ": " << oss.str() << "," << endl;
    }
    {
        // implications producing a tautology -- must be suppressed
        clause c1;
        c1.insert(1);
        c1.insert(2);
        clause c2;
        c2.insert(-1);
        c2.insert(-2);
        vector<clause> impls;
        implications(c1, c2, back_inserter(impls));
        cout << quote("implications_tautology_suppressed") << ": " << impls.size() << "," << endl;
    }

    // ---- nf_mapper::add_item id assignment order + argument sign folding ----
    {
        combo_tree tr = str2combo_tree("or($2 $1 !$2 $3 !$1)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        cout << quote("add_cnf_ids_mixed_args") << ": " << quote(nf_str(f)) << "," << endl;
    }
    {
        // add_dnf over an or-of-ands tree
        combo_tree tr = str2combo_tree("or(and($1 $2) and(!$1 $3) $4)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_dnf(root.begin(), root.end());
        cout << quote("add_dnf_or_of_ands") << ": " << quote(nf_str(f)) << "," << endl;
    }
    {
        // add_item interning: the SAME compound subtree appearing twice
        // (structurally, not by identity) should be assigned the SAME id.
        combo_tree tr = str2combo_tree("or(or($1 $2) $3 or($1 $2))", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        cout << quote("add_cnf_interning_repeat_subtree") << ": " << quote(nf_str(f)) << "," << endl;
    }

    // ---- extract_cnf / extract_dnf round trips ----
    {
        combo_tree tr = str2combo_tree("or(or($1 !$2) $3)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        mapper.extract_cnf(f.begin(), f.end(), tr, root);
        cout << quote("extract_cnf_roundtrip") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        combo_tree tr = str2combo_tree("or(and($1 $2) $3)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_dnf(root.begin(), root.end());
        mapper.extract_dnf(f.begin(), f.end(), tr, root);
        cout << quote("extract_dnf_roundtrip") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        // extract_cnf collapsing to a single child (one clause, single literal)
        combo_tree tr = str2combo_tree("or($1)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        mapper.extract_cnf(f.begin(), f.end(), tr, root);
        cout << quote("extract_cnf_single_clause_single_literal") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        // extract_cnf: a clause list built to be entirely empty (0 clauses)
        // -- what does sib become? (sib.is_childless() branch)
        combo_tree tr = str2combo_tree("$1", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf empty_f;
        mapper.extract_cnf(empty_f.begin(), empty_f.end(), tr, root);
        cout << quote("extract_cnf_empty_nf") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        combo_tree tr = str2combo_tree("$1", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf empty_f;
        mapper.extract_dnf(empty_f.begin(), empty_f.end(), tr, root);
        cout << quote("extract_dnf_empty_nf") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        // extract_cnf: a clause list containing one EMPTY clause plus others
        // -- silently skipped (no early return, unlike extract_dnf).
        combo_tree tr = str2combo_tree("$1", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        clause empty_clause;
        clause c1;
        c1.insert(1);
        nf f;
        f.push_back(empty_clause);
        f.push_back(c1);
        // need real interned items for c1's literal 1 -- reuse a fresh mapper
        // seeded via add_cnf first so id 1 refers to a real argument.
        combo_tree seed = str2combo_tree("or($5)", vector<string>());
        nf seed_f = mapper.add_cnf(seed.begin().begin(), seed.begin().end());
        mapper.extract_cnf(f.begin(), f.end(), tr, root);
        cout << quote("extract_cnf_with_empty_clause_present") << ": " << quote(show(tr)) << "," << endl;
    }
    {
        // extract_dnf: an empty clause anywhere -> whole result collapses to
        // logical_true, even with other non-trivial clauses present.
        combo_tree tr = str2combo_tree("$1", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        combo_tree seed = str2combo_tree("and($5 $6)", vector<string>());
        nf seed_f = mapper.add_dnf(seed.begin().begin(), seed.begin().end());
        clause empty_clause;
        nf f;
        f.push_back(*seed_f.begin());
        f.push_back(empty_clause);
        mapper.extract_dnf(f.begin(), f.end(), tr, root);
        cout << quote("extract_dnf_with_empty_clause_present") << ": " << quote(show(tr)) << "," << endl;
    }

    // ---- extract_conjunction ----
    {
        combo_tree tr = str2combo_tree("or($1 $2)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        clause c1 = *f.begin();       // {id for $1}
        clause c2 = *(++f.begin());   // {id for $2}
        clause both;
        both.insert(*c1.begin());
        both.insert(*c2.begin());

        combo_tree tr2(vertex(id::logical_and));
        combo_tree::iterator dst_root = tr2.begin();
        combo_tree::iterator sib1 = tr2.append_child(dst_root, vertex(0.0));
        mapper.extract_conjunction(c1, tr2, sib1); // single positive literal
        cout << quote("extract_conjunction_single_positive") << ": " << quote(show(tr2)) << "," << endl;
    }
    {
        combo_tree tr = str2combo_tree("or($1 $2)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        clause c1 = *f.begin();
        int id1 = *c1.begin();
        clause neg;
        neg.insert(-id1);

        combo_tree tr2(vertex(id::logical_and));
        combo_tree::iterator dst_root = tr2.begin();
        combo_tree::iterator sib1 = tr2.append_child(dst_root, vertex(0.0));
        mapper.extract_conjunction(neg, tr2, sib1); // single negative literal
        cout << quote("extract_conjunction_single_negative") << ": " << quote(show(tr2)) << "," << endl;
    }
    {
        combo_tree tr = str2combo_tree("or($1 $2 $3)", vector<string>());
        combo_tree::iterator root = tr.begin();
        nf_mapper<vertex> mapper;
        nf f = mapper.add_cnf(root.begin(), root.end());
        clause all;
        for (const clause& c : f)
            all.insert(*c.begin());

        combo_tree tr2(vertex(id::logical_and));
        combo_tree::iterator dst_root = tr2.begin();
        combo_tree::iterator sib1 = tr2.append_child(dst_root, vertex(0.0));
        mapper.extract_conjunction(all, tr2, sib1); // multi-literal -> logical_and
        cout << quote("extract_conjunction_multi") << ": " << quote(show(tr2)) << endl;
    }

    cout << "}" << endl;
    return 0;
}
