// Captures the deterministic contin-output build_knobs boundary for S14
// Task 2 -- the contin-output counterpart of representation_probe.cc's
// pure-Boolean capture. Full classic `representation` is used, at its own
// real defaults (stepsize=1.0, expansion=2.0, depth=5 -- representation.cc's
// own module statics, never overridden here) so this is not an oracle
// configured by the port: every case is also run at `linear_contin=false`,
// classic's own CLI default (`--linear-regression`, problem-params.cc:292),
// not just the constructor's own `linear_contin=true` default.
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/representation/representation.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

// `ostream_combo_tree` (not the generic `opencog::tree<T>::operator<<`,
// which omits the trailing space `ostream_combo_tree` -- and this port's own
// `combo_tree_to_str`, which mirrors it -- always appends after the whole
// tree) matches `candidate_roundtrip_probe.cc`'s own convention, the probe
// `tests/fixtures/representation.json` is captured from.
static string rendered(const combo_tree& tr)
{
    stringstream out;
    ostream_combo_tree(out, tr);
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

static void emit_case(const string& name, const string& source,
                      const vector<type_node>& itypes, bool linear_contin,
                      bool last)
{
    combo_tree exemplar = parse(source);
    type_tree signature = gen_signature(itypes, id::contin_type);

    // This probe never turns a knob, so the simplifier rule is threaded
    // through but never actually invoked (matching representation_probe.cc's
    // own comment on the analogous Boolean-only reduction rule).
    const rule& simplifier = mixed_reduction();

    representation rep(simplifier, simplifier, exemplar, signature,
                       operator_set(), nullptr, nullptr, linear_contin);

    stringstream prototype;
    rep.ostream_prototype(prototype);

    cout << "    {\"name\":" << json_string(name)
         << ",\"source\":" << json_string(source) << ",\"itypes\":[";
    for (size_t idx = 0; idx < itypes.size(); ++idx) {
        if (idx) cout << ',';
        stringstream tn;
        tn << itypes[idx];
        cout << json_string(tn.str());
    }
    cout << "],\"linear_contin\":" << (linear_contin ? "true" : "false")
         << ",\"decorated_exemplar\":" << json_string(rendered(rep.exemplar()))
         << ",\"prototype\":" << json_string(prototype.str())
         << ",\"disc_knob_count\":" << rep.disc.size()
         << ",\"contin_knobs\":[";
    size_t order = 0;
    for (auto it = rep.contin.cbegin(); it != rep.contin.cend(); ++it, ++order) {
        if (order) cout << ',';
        const field_set::contin_spec& spec = it->first;
        const contin_knob& kb = it->second;
        combo_tree location(kb.get_loc());
        size_t preorder = distance(rep.exemplar().begin(), kb.get_loc());
        cout << "{\"order\":" << order
             << ",\"mean\":" << spec.mean
             << ",\"step_size\":" << spec.step_size
             << ",\"expansion\":" << spec.expansion
             << ",\"depth\":" << spec.depth
             << ",\"text\":" << json_string(kb.toStr())
             << ",\"location_preorder\":" << preorder
             << ",\"location_subtree\":" << json_string(rendered(location)) << '}';
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

int main()
{
    cout << unitbuf;
    cout << "{\n  \"stepsize\":1.0,\n  \"expansion\":2.0,\n  \"depth\":5,\n  \"cases\":[\n";

    struct spec { const char* name; const char* source; vector<type_node> itypes; };
    vector<spec> specs = {
        {"bare-argument", "$1", {id::contin_type}},
        {"sum-two-args", "plus($1 $2)", {id::contin_type, id::contin_type}},
        {"sin-of-arg", "sin($1)", {id::contin_type}},
        {"impulse-of-boolean-arg", "impulse($1)", {id::boolean_type}},
        {"mixed-sum", "plus($1 impulse($2))", {id::contin_type, id::boolean_type}},
        {"div-two-args", "div($1 $2)", {id::contin_type, id::contin_type}},
        {"times-arg-and-constant", "times($1 0.5)", {id::contin_type}},
        // Constant appears BEFORE the value, unlike the case above --
        // exercises canonize_times' genuine tree::swap (build_knobs.cc:803)
        // rather than the swap(x, x) no-op the case above happens to reach
        // (0.5 there is already `it`'s last child, so no positions actually
        // move). Found by mutation testing: a no-op `swap` left every OTHER
        // case in this fixture unchanged. See the S14 Task 2 report.
        {"times-constant-and-arg", "times(0.5 $1)", {id::contin_type}},
        {"bare-childless-plus", "plus", {id::contin_type, id::contin_type}},
    };

    for (size_t i = 0; i < specs.size(); ++i) {
        for (bool linear_contin : {true, false}) {
            bool last = (i + 1 == specs.size()) && linear_contin == false;
            string name = string(specs[i].name) + (linear_contin ? "-linear" : "-nonlinear");
            emit_case(name, specs[i].source, specs[i].itypes, linear_contin, last);
        }
    }
    cout << "  ]\n}\n";
    return 0;
}
