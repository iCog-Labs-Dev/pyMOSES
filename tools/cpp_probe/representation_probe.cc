// Captures the deterministic pure-Boolean build_knobs boundary for S8 Task 3.
// Full classic representation is used only to reach its internal builder; no
// packed fields or candidate-generation results are emitted by this probe.
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/representation/representation.h>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

struct probe_event {
    string before;
    string after;
};

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

static combo_tree parse(const string& value)
{
    stringstream in(value);
    combo_tree tr;
    in >> tr;
    return tr;
}

struct recording_rule : public crule<recording_rule> {
    recording_rule(const rule& delegate, vector<probe_event>& events)
        : crule<recording_rule>("recording_rule"), _delegate(&delegate), _events(&events) {}

    void operator()(combo_tree& tr, combo_tree::iterator it) const
    {
        probe_event event;
        event.before = rendered(tr);
        (*_delegate)(tr, it);
        event.after = rendered(tr);
        _events->push_back(event);
    }

private:
    const rule* _delegate;
    vector<probe_event>* _events;
};

static void emit_case(const string& name, const string& source, int arity,
                      const vector<int>& ignored, bool last)
{
    combo_tree exemplar = parse(source);
    type_tree signature = gen_signature(id::boolean_type, arity);
    vertex_set ignore_ops;
    for (int idx : ignored) ignore_ops.insert(argument(idx));

    logical_reduction reduction(ignore_ops);
    const rule& delegate = reduction(2);
    vector<probe_event> events;
    recording_rule recorder(delegate, events);
    representation rep(recorder, recorder, exemplar, signature, ignore_ops,
                       nullptr, nullptr, false, -1.0);

    stringstream prototype;
    rep.ostream_prototype(prototype);

    cout << "    {\"name\":" << json_string(name)
         << ",\"source\":" << json_string(source) << ",\"arity\":" << arity
         << ",\"ignored_arguments\":[";
    for (size_t idx = 0; idx < ignored.size(); ++idx) {
        if (idx) cout << ',';
        cout << ignored[idx];
    }
    cout << "],\"decorated_exemplar\":" << json_string(rendered(rep.exemplar()))
         << ",\"prototype\":" << json_string(prototype.str())
         << ",\"probe_calls\":[";
    for (size_t idx = 0; idx < events.size(); ++idx) {
        if (idx) cout << ',';
        cout << "{\"before\":" << json_string(events[idx].before)
             << ",\"after\":" << json_string(events[idx].after) << '}';
    }
    cout << "],\"knobs\":[";
    size_t order = 0;
    for (auto it = rep.disc.cbegin(); it != rep.disc.cend(); ++it, ++order) {
        if (order) cout << ',';
        combo_tree location(it->second->get_loc());
        size_t preorder = distance(rep.exemplar().begin(), it->second->get_loc());
        cout << "{\"order\":" << order << ",\"multy\":" << it->first.multy
             << ",\"text\":" << json_string(it->second->toStr())
             << ",\"in_exemplar\":"
             << (it->second->in_exemplar() ? "true" : "false")
             << ",\"location_preorder\":" << preorder
             << ",\"location_subtree\":" << json_string(rendered(location)) << '}';
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

int main()
{
    cout << unitbuf;
    cout << "{\n  \"perm_ratio\":-1,\n  \"cases\":[\n";
    emit_case("constant-true", "true", 0, {}, false);
    emit_case("bare-argument", "$1", 1, {}, false);
    emit_case("small-and", "and($1 $2)", 2, {}, false);
    emit_case("small-or-negated", "or($1 !$2)", 2, {}, false);
    emit_case("nested-three", "and(or($1 $2) !$3)", 3, {}, false);
    emit_case("ignored-argument", "and($1 $2)", 2, {2}, true);
    cout << "  ]\n}\n";
    return 0;
}
