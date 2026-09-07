// Captures classic logical_subtree_knob construction and mutation behavior.
// Build/run through tools/capture_logical_knob.py.
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/representation/knobs.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
        if (c == '\\' || c == '\"') out += '\\';
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

static void emit_state(logical_subtree_knob& knob, combo_tree& tr, int idx)
{
    knob.turn(idx);
    cout << "{\"idx\":" << idx << ",\"tree\":" << json_string(rendered(tr))
         << ",\"reduced\":" << json_string(reduced(tr))
         << ",\"complexity\":" << knob.complexity_bound() << '}';
}

static void emit_transition_case(const string& name, const string& base,
                                 int arg_idx, const vector<int>& sequence,
                                 bool last)
{
    combo_tree tr = parse(base);
    vertex_set ignore_ops;
    logical_reduce(2, tr, ignore_ops);
    combo_tree subtree{argument(arg_idx)};
    logical_subtree_knob knob(tr, tr.begin(), subtree.begin());
    cout << "    {\"name\":" << json_string(name)
         << ",\"base\":" << json_string(base) << ",\"arg\":" << arg_idx
         << ",\"initial_tree\":" << json_string(rendered(tr))
         << ",\"knob\":" << json_string(knob.toStr())
         << ",\"in_exemplar\":" << (knob.in_exemplar() ? "true" : "false")
         << ",\"multiplicity\":" << knob.multiplicity()
         << ",\"spec_multy\":" << knob.spec().multy << ",\"states\":[";
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (i) cout << ',';
        emit_state(knob, tr, sequence[i]);
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

static void emit_active_cases()
{
    combo_tree tr = parse("and($1 not($2))");
    vertex_set ignore_ops;
    logical_reduce(2, tr, ignore_ops);
    combo_tree t1(argument(1)), t2(argument(2)), t3(argument(3));
    logical_subtree_knob k1(tr, tr.begin(), t1.begin());
    logical_subtree_knob k2(tr, tr.begin(), t2.begin());
    logical_subtree_knob k3(tr, tr.begin(), t3.begin());
    logical_subtree_knob* knobs[] = {&k1, &k2, &k3};
    cout << "  \"active\":{\"base\":\"and($1 not($2))\",\"prototype\":"
         << json_string(rendered(tr))
         << ",\"knobs\":[" << json_string(k1.toStr()) << ','
         << json_string(k2.toStr()) << ',' << json_string(k3.toStr())
         << "],\"turns\":[";
    bool first = true;
    for (int ki = 0; ki < 3; ++ki) {
        for (int idx = 0; idx < 3; ++idx) {
            if (!first) cout << ',';
            first = false;
            knobs[ki]->turn(idx);
            cout << "{\"knob\":" << ki + 1 << ",\"idx\":" << idx
                 << ",\"tree\":" << json_string(rendered(tr))
                 << ",\"reduced\":" << json_string(reduced(tr)) << '}';
        }
        knobs[ki]->turn(0);
    }
    cout << "]},\n";
}

static void emit_append_case(const string& name, const string& base,
                             int arg_idx, bool last)
{
    combo_tree tr = parse(base);
    vertex_set ignore_ops;
    logical_reduce(2, tr, ignore_ops);
    combo_tree subtree{argument(arg_idx)};
    logical_subtree_knob knob(tr, tr.begin(), subtree.begin());
    cout << "    {\"name\":" << json_string(name)
         << ",\"base\":" << json_string(base) << ",\"arg\":" << arg_idx
         << ",\"rows\":[";
    bool first = true;
    for (int idx = 0; idx < 3; ++idx) {
        for (int nested = 0; nested < 2; ++nested) {
            if (!first) cout << ',';
            first = false;
            combo_tree candidate;
            combo_tree::iterator parent = candidate.end();
            if (nested) {
                candidate.set_head(id::logical_and);
                candidate.append_child(candidate.begin(), argument(9));
                parent = candidate.begin();
            }
            combo_tree::iterator source = knob.append_to(candidate, parent, idx);
            bool source_valid = tr.is_valid(source);
            bool source_is_exemplar_end = source == tr.end();
            bool source_is_candidate_end = source == candidate.end();
            string source_tree = source_valid ? rendered(combo_tree(source)) : "";
            stringstream parent_value;
            if (candidate.is_valid(parent)) parent_value << *parent;
            cout << "{\"idx\":" << idx << ",\"nested\":"
                 << (nested ? "true" : "false") << ",\"candidate\":"
                 << json_string(rendered(candidate)) << ",\"source_valid\":"
                 << (source_valid ? "true" : "false")
                 << ",\"source_is_exemplar_end\":"
                 << (source_is_exemplar_end ? "true" : "false")
                 << ",\"source_is_candidate_end\":"
                 << (source_is_candidate_end ? "true" : "false")
                 << ",\"source\":" << json_string(source_tree) << ",\"parent\":"
                 << json_string(parent_value.str()) << '}';
        }
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

static void emit_disallow_case(const string& name, const string& base,
                               int arg_idx, int setting, bool last)
{
    combo_tree tr = parse(base);
    combo_tree subtree{argument(arg_idx)};
    logical_subtree_knob knob(tr, tr.begin(), subtree.begin());
    knob.disallow(setting);
    cout << "    {\"name\":" << json_string(name)
         << ",\"base\":" << json_string(base) << ",\"arg\":" << arg_idx
         << ",\"setting\":" << setting << ",\"multiplicity\":"
         << knob.multiplicity() << ",\"spec_multy\":" << knob.spec().multy
         << ",\"knob\":" << json_string(knob.toStr()) << "}"
         << (last ? "\n" : ",\n");
}

static void emit_disallow_cases()
{
    cout << "  \"disallow\":[\n";
    for (int setting = 0; setting < 3; ++setting)
        emit_disallow_case("missing", "and($1 $9)", 2, setting, false);
    for (int setting = 0; setting < 3; ++setting)
        emit_disallow_case("existing-positive", "and($1 $9)", 1, setting,
                           setting == 2);
    cout << "  ]\n";
}

static void emit_copy_case(const string& name, const string& base,
                           int arg_idx, int source_turn, int disallowed,
                           bool destination_from_source,
                           const vector<int>& sequence, bool last)
{
    combo_tree source = parse(base);
    vertex_set ignore_ops;
    logical_reduce(2, source, ignore_ops);
    combo_tree subtree{argument(arg_idx)};
    logical_subtree_knob source_knob(source, source.begin(), subtree.begin());
    source_knob.turn(source_turn);
    source_knob.disallow(disallowed);

    combo_tree destination = destination_from_source ? combo_tree(source)
                                                     : parse(base);
    if (!destination_from_source)
        logical_reduce(2, destination, ignore_ops);
    string destination_before = rendered(destination);
    logical_subtree_knob copied(destination, destination.begin(), source_knob);
    cout << "    {\"name\":" << json_string(name)
         << ",\"base\":" << json_string(base) << ",\"arg\":" << arg_idx
         << ",\"source_turn\":" << source_turn
         << ",\"disallowed\":" << disallowed
         << ",\"destination_mode\":"
         << json_string(destination_from_source ? "source-copy" : "fresh-base")
         << ",\"source_tree\":" << json_string(rendered(source))
         << ",\"destination_before\":" << json_string(destination_before)
         << ",\"destination_after\":" << json_string(rendered(destination))
         << ",\"knob\":" << json_string(copied.toStr())
         << ",\"in_exemplar\":" << (copied.in_exemplar() ? "true" : "false")
         << ",\"multiplicity\":" << copied.multiplicity()
         << ",\"spec_multy\":" << copied.spec().multy << ",\"states\":[";
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (i) cout << ',';
        emit_state(copied, destination, sequence[i]);
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

static void emit_clear_case(const string& name, const string& base,
                            int arg_idx, int turn, int disallowed, bool last)
{
    combo_tree tr = parse(base);
    vertex_set ignore_ops;
    logical_reduce(2, tr, ignore_ops);
    combo_tree subtree{argument(arg_idx)};
    logical_subtree_knob knob(tr, tr.begin(), subtree.begin());
    if (disallowed >= 0) knob.disallow(disallowed);
    if (turn >= 0) knob.turn(turn);
    string before = rendered(tr);
    bool retained = knob.in_exemplar();
    knob.clear_exemplar();
    cout << "    {\"name\":" << json_string(name)
         << ",\"base\":" << json_string(base) << ",\"arg\":" << arg_idx
         << ",\"turn\":" << turn << ",\"disallowed\":" << disallowed
         << ",\"before\":" << json_string(before)
         << ",\"after\":" << json_string(rendered(tr))
         << ",\"retained\":" << (retained ? "true" : "false")
         << ",\"after_knob\":"
         << json_string(retained ? knob.toStr() : "") << "}"
         << (last ? "\n" : ",\n");
}

int main()
{
    cout << unitbuf;
    cout << "{\n";
    emit_active_cases();
    cout << "  \"transitions\":[\n";
    emit_transition_case("existing-positive", "and($1 $9)", 1,
                         {0, 1, 2, 0, 2, 1, 0}, false);
    emit_transition_case("existing-negated", "and(not($1) $9)", 1,
                         {0, 1, 2, 0, 2, 1, 0}, false);
    emit_transition_case("missing", "and($1 $9)", 2,
                         {0, 1, 2, 0, 2, 1, 0}, true);
    cout << "  ],\n  \"append\":[\n";
    emit_append_case("existing-positive", "and($1 $9)", 1, false);
    emit_append_case("existing-negated", "and(not($1) $9)", 1, false);
    emit_append_case("missing", "and($1 $9)", 2, true);
    cout << "  ],\n";
    emit_disallow_cases();
    cout << ",\n  \"copy\":[\n";
    emit_copy_case("existing-absent-disallowed-negated", "and($1 $9)", 1,
                   1, 2, true, {0, 1}, false);
    emit_copy_case("missing-negated-disallowed-present", "and($1 $9)", 2,
                   2, 1, false, {0, 1}, true);
    cout << "  ],\n  \"clear\":[\n";
    emit_clear_case("existing-negated", "and(not($1) $9)", 1, 2, -1,
                    false);
    emit_clear_case("existing-disallowed-absent", "and($1 $9)", 1, -1, 0,
                    false);
    emit_clear_case("missing-absent", "and($1 $9)", 2, 0, -1, false);
    emit_clear_case("missing-present", "and($1 $9)", 2, 1, -1, false);
    emit_clear_case("missing-negated-disallowed-absent", "and($1 $9)", 2,
                    1, 0, true);
    cout << "  ]\n";
    cout << "}\n";
    return 0;
}
