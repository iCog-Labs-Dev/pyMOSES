// Measures representation::get_candidate_rec (representation.cc:331) over a
// directly-constructed COMPOSITE logical_subtree_knob -- a knob whose bound
// exemplar subtree has children of its own, which `--perm_ratio=-1` (this
// port's only supported deterministic mode) never lets build_knobs produce
// on its own (build_knobs.cc:435-437 returns before pair sampling, and every
// knob it does build sits on a single literal). A composite knob only
// arises the way this probe makes one: a `logical_subtree_knob` whose
// constructor subtree argument already equals an existing sibling
// (knobs.cc:86's "existing immediate match" branch), so `_default ==
// present` and `get_loc()` has real children.
//
// This exists to back
// tests/moses/representation/test_representation.py's
// test_candidate_recursion_descends_into_a_composite_knob_subtree, whose
// expected trees were originally "derived from classic's algorithm read
// line by line ... NOT measured against a binary" (S8 fix-round Minor).
// `get_candidate_rec`'s own recursion shape is generic and not what was in
// question; what IS measured here, through the real compiled
// `logical_subtree_knob::append_to` (knobs.h:255, knobs.cc:163 --
// note it takes `combo_tree::iterator&`, mutating a caller's local by
// reference, which this probe's `candidate_rec` relies on exactly as
// `get_candidate_rec` does), is whether that composite-knob recursion
// actually descends into the bound subtree's real children with the right
// parent, and whether the exemplar is left untouched.
//
// Build/run through tools/capture_representation.py's sibling recipe (see
// that file's `build_probe`/`CLASSIC`/`PREFIX` constants) -- this probe
// takes the same include/link flags as candidate_roundtrip_probe.cc.
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/representation/knobs.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

static combo_tree parse(const string& value)
{
    stringstream in(value);
    combo_tree tr;
    in >> tr;
    return tr;
}

static string rendered(const combo_tree& tr)
{
    stringstream out;
    out << tr;
    return out.str();
}

static string json_string(const string& value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
}

static string reduced(const combo_tree& source)
{
    combo_tree tr(source);
    vertex_set ignore_ops;
    clean_reduce(tr);
    logical_reduce(2, tr, ignore_ops);
    return rendered(tr);
}

// Transcribes representation::get_candidate_rec (representation.cc:331) for
// the single-knob case this probe needs. `find_disc_knob` there is just
// "does src == the one knob's location" here; everything else -- in
// particular the `append_to`/`is_valid`/recursive_call shape -- matches the
// real function line for line.
static void candidate_rec(logical_subtree_knob& knob, combo_tree& exemplar,
                          combo_tree::iterator src,
                          combo_tree::iterator parent_dst,
                          combo_tree& candidate, int idx)
{
    typedef combo_tree::iterator pre_it;
    typedef combo_tree::sibling_iterator sib_it;

    auto recursive_call = [&](pre_it new_parent_dst, pre_it node) {
        for (sib_it child = node.begin(); child != node.end(); ++child)
            candidate_rec(knob, exemplar, child, new_parent_dst, candidate, idx);
    };

    if (src == knob.get_loc()) {
        pre_it new_src = knob.append_to(candidate, parent_dst, idx);
        if (exemplar.is_valid(new_src))
            recursive_call(parent_dst, new_src);
        return;
    }

    auto append_child = [&candidate](pre_it parent_dst, const vertex& v) {
        return candidate.empty() ? candidate.set_head(v)
                                 : candidate.append_child(parent_dst, v);
    };

    recursive_call(append_child(parent_dst, *src), src);
}

int main()
{
    combo_tree tr = parse("or(and($1 $2) $3)");
    combo_tree subtree = parse("and($1 $2)");
    string exemplar_before = rendered(tr);

    logical_subtree_knob knob(tr, tr.begin(), subtree.begin());

    // Sanity: the constructor must have bound to the EXISTING `and($1 $2)`
    // sibling (in_exemplar) rather than appending a fresh null_vertex
    // wrapper -- otherwise this is not the composite-knob case at all.
    if (!knob.in_exemplar()) {
        cerr << "composite_knob_probe: expected knob to bind to the existing "
                "sibling (in_exemplar() was false)\n";
        return 1;
    }
    if (tr.number_of_children(knob.get_loc()) != 2) {
        cerr << "composite_knob_probe: expected the bound subtree to keep "
                "its two children, got "
             << tr.number_of_children(knob.get_loc()) << '\n';
        return 1;
    }

    cout << "{\n  \"exemplar_before\":" << json_string(exemplar_before)
         << ",\n  \"rows\":[\n";
    for (int idx = 0; idx < 3; ++idx) {
        combo_tree candidate;
        candidate_rec(knob, tr, tr.begin(), candidate.end(), candidate, idx);
        string unreduced = rendered(candidate);
        string reduced_tree = reduced(candidate);
        cout << "    {\"idx\":" << idx
             << ",\"unreduced\":" << json_string(unreduced)
             << ",\"reduced\":" << json_string(reduced_tree) << '}'
             << (idx == 2 ? "\n" : ",\n");
    }
    cout << "  ],\n  \"exemplar_after\":" << json_string(rendered(tr))
         << "\n}\n";
    return 0;
}
