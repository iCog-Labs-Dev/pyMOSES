// Captures the MIXED build_knobs boundary -- Boolean OUTPUT over contin
// (and/or Boolean) INPUTS, i.e. the `is_predicate` branches of
// `logical_canonize` (build_knobs.cc:160) and `build_logical`
// (build_knobs.cc:640), plus the contin branches of `insert_typed_arg`
// (build_knobs.cc:386) and `sample_logical_perms` (build_knobs.cc:425).
//
// Where representation_probe.cc captures the pure-Boolean boundary and
// contin_representation_probe.cc the pure-contin one, this probe captures
// the path that joins them: a Boolean representation whose knob set holds
// BOTH disc knobs (over `0<($i)` perms) and contin knobs (inside each
// predicate's canonized contin argument).
//
// Full classic `representation` is used, at its own real module-static
// defaults (stepsize=1.0, expansion=2.0, depth=5 -- representation.cc:59-68,
// never overridden here), with `logical_reduction(ignore_ops)(2)` as both
// rules -- which is what problem-params.cc actually hands a Boolean-output
// problem, and which routes `disc_probe` through `simplify_predicates`'
// own contin/mixed tail (logical_rules.cc:71-81).
//
// Both `perm_ratio` regimes are captured, because they reach DIFFERENT
// code: -1.0 takes `sample_logical_perms`' early return (single-argument
// perms only, no RNG draws), while 0.0 -- classic's own constructor and CLI
// default (build_knobs.h:56, representation.h:62, problem-params.cc:299) --
// runs the pair loop, which is the ONLY caller of `insert_typed_arg`. The
// pair loop draws from the process-wide `randGen()` singleton, so every
// case reseeds it immediately before construction, exactly as
// neighborhood_probe.cc does.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/mixed_predicate_probe \
//       tools/cpp_probe/mixed_predicate_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 tools/cpp_probe/build/mixed_predicate_probe )
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/representation/representation.h>
#include <opencog/util/mt19937ar.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

// `ostream_combo_tree`, not `operator<<`: the former appends the trailing
// space this port's own `combo_tree_to_str` mirrors. Same convention as
// contin_representation_probe.cc.
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

struct spec {
    const char* name;
    const char* source;
    vector<type_node> itypes;
    vector<int> ignored_arguments;   // argument($i) added to ignore_ops
    bool ignore_gt_zero;             // id::greater_than_zero added to ignore_ops
};

static void emit_case(const spec& sp, float perm_ratio, unsigned seed, bool last,
                      type_node output_type = id::boolean_type)
{
    combo_tree exemplar = parse(sp.source);
    type_tree signature = gen_signature(sp.itypes, output_type);

    vertex_set ignore_ops;
    for (int idx : sp.ignored_arguments) ignore_ops.insert(argument(idx));
    if (sp.ignore_gt_zero) ignore_ops.insert(id::greater_than_zero);

    // For a Boolean output, `logical_reduction(ignore_ops)(2)` is what
    // problem-params.cc actually hands the builder -- and it is what routes
    // `disc_probe` through `simplify_predicates`' own contin/mixed tail.
    // For a contin output, `mixed_reduction()`, mirroring
    // contin_representation_probe.cc (the contin path never calls
    // `disc_probe`, so the rule is threaded through but never invoked).
    logical_reduction reduction(ignore_ops);
    const rule& simplifier = (output_type == id::boolean_type)
                                 ? static_cast<const rule&>(reduction(2))
                                 : static_cast<const rule&>(mixed_reduction());

    randGen().seed(seed);
    representation rep(simplifier, simplifier, exemplar, signature, ignore_ops,
                       nullptr, nullptr, false /* linear_contin */, perm_ratio);

    stringstream prototype;
    rep.ostream_prototype(prototype);

    stringstream otype_ss;
    otype_ss << output_type;
    cout << "    {\"name\":" << json_string(string(sp.name))
         << ",\"otype\":" << json_string(otype_ss.str())
         << ",\"source\":" << json_string(sp.source) << ",\"itypes\":[";
    for (size_t idx = 0; idx < sp.itypes.size(); ++idx) {
        if (idx) cout << ',';
        stringstream tn;
        tn << sp.itypes[idx];
        cout << json_string(tn.str());
    }
    cout << "],\"ignored_arguments\":[";
    for (size_t idx = 0; idx < sp.ignored_arguments.size(); ++idx) {
        if (idx) cout << ',';
        cout << sp.ignored_arguments[idx];
    }
    cout << "],\"ignore_greater_than_zero\":"
         << (sp.ignore_gt_zero ? "true" : "false")
         << ",\"perm_ratio\":" << perm_ratio
         << ",\"seed\":" << seed
         << ",\"decorated_exemplar\":" << json_string(rendered(rep.exemplar()))
         << ",\"prototype\":" << json_string(prototype.str())
         << ",\"disc_knobs\":[";
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
    cout << "],\"contin_knobs\":[";
    order = 0;
    for (auto it = rep.contin.cbegin(); it != rep.contin.cend(); ++it, ++order) {
        if (order) cout << ',';
        const field_set::contin_spec& cspec = it->first;
        const contin_knob& kb = it->second;
        combo_tree location(kb.get_loc());
        size_t preorder = distance(rep.exemplar().begin(), kb.get_loc());
        cout << "{\"order\":" << order
             << ",\"mean\":" << cspec.mean
             << ",\"step_size\":" << cspec.step_size
             << ",\"expansion\":" << cspec.expansion
             << ",\"depth\":" << cspec.depth
             << ",\"text\":" << json_string(kb.toStr())
             << ",\"location_preorder\":" << preorder
             << ",\"location_subtree\":" << json_string(rendered(location)) << '}';
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

// The second half of this probe: `logical_reduce` over predicate-bearing
// trees. That is the ONLY route to `simplify_predicates`' own contin/mixed
// tail (logical_rules.cc:76-81), since `simplify_predicates` is the first
// element of `logical_reduction`'s `p_medium` (logical_reduction.cc) and
// `p_medium` is what efforts 2 and 3 both return.
//
// `logical-reductor` (moses/comboreduct/main/) cannot serve as this oracle:
// it wraps every reduction in a `complete_truth_table` equivalence check,
// and `boolean_interpreter` aborts on `0<` ("boolean_interpreter does not
// handle builtin 0<", interpreter.cc:90) -- confirmed live, exit 134.
static void emit_reduce_case(const string& source, int effort, bool last)
{
    combo_tree tr = parse(source);
    unsigned size_before = tr.size();
    vertex_set ignore_ops;  // classic's own real default: empty
    string error_msg;
    bool ok = true;
    try {
        logical_reduce(effort, tr, ignore_ops);
    } catch (std::exception& e) {
        ok = false;
        error_msg = e.what();
    }
    cout << "    {\"input\":" << json_string(source)
         << ",\"effort\":" << effort
         << ",\"ok\":" << (ok ? "true" : "false");
    if (ok) {
        cout << ",\"reduced\":" << json_string(rendered(tr))
             << ",\"size_before\":" << size_before
             << ",\"size_after\":" << tr.size();
    } else {
        cout << ",\"error\":" << json_string(error_msg);
    }
    cout << "}" << (last ? "\n" : ",\n");
}

int main()
{
    cout << unitbuf;
    cout << "{\n  \"stepsize\":1.0,\n  \"expansion\":2.0,\n  \"depth\":5,\n"
         << "  \"linear_contin\":false,\n  \"reduct_effort\":2,\n  \"cases\":[\n";

    const type_node C = id::contin_type;
    const type_node B = id::boolean_type;

    vector<spec> specs = {
        // The shape every real contin-input run starts from:
        // type_to_exemplar(boolean_type) == combo_tree(logical_and).
        {"bare-and-1contin", "and", {C}, {}, false},
        {"bare-and-2contin", "and", {C, C}, {}, false},
        // logical_canonize's own is_predicate branch: a bare predicate at
        // the root, negated and not.
        {"root-predicate", "0<($1)", {C}, {}, false},
        {"root-predicate-negated", "not(0<($1))", {C}, {}, false},
        {"root-predicate-sum", "0<(+($1 $2))", {C, C}, {}, false},
        // build_logical's own is_predicate branch: a predicate as a CHILD
        // of a junctor, negated and not.
        {"and-of-predicates", "and(0<($1) 0<($2))", {C, C}, {}, false},
        {"and-negated-predicate", "and(not(0<($1)) 0<($2))", {C, C}, {}, false},
        {"or-nested", "or(and(not(0<($1)) 0<($2)) 0<($1))", {C, C}, {}, false},
        // A predicate over a non-trivial contin expression -- contin_canonize
        // reaches its `plus`/`times`/`div` arms rather than the bare-argument
        // one.
        {"predicate-over-times", "0<(*($1 $2))", {C, C}, {}, false},
        {"predicate-over-div", "0<(/($1 $2))", {C, C}, {}, false},
        {"predicate-over-sin", "0<(sin($1))", {C}, {}, false},
        {"predicate-over-constant-sum", "0<(+(1 $1))", {C}, {}, false},
        // MIXED inputs: Boolean literals live alongside predicates, and
        // append_linear_combination wraps the Boolean ones in `impulse`.
        {"mixed-inputs", "and($2 0<($1))", {C, B}, {}, false},
        {"mixed-inputs-bare-and", "and", {C, B, C}, {}, false},
        // ignore_ops: an ignored ARGUMENT, and an ignored greater_than_zero
        // (which makes both `sample_logical_perms` and `insert_typed_arg`
        // silently emit nothing for every contin argument).
        {"ignored-argument", "and", {C, C}, {2}, false},
        {"ignored-greater-than-zero", "and", {C, C}, {}, true},
        {"ignored-gt-zero-mixed", "and", {C, B}, {}, true},
    };

    // CONTIN-output exemplars that nonetheless carry a predicate. Classic
    // reaches `logical_canonize`'s own is_predicate branch from
    // `rec_canonize`'s `impulse` arm (build_knobs.cc:903-905) -- and note
    // what it does NOT do: the contin path never calls `build_logical`, so
    // such a predicate is canonized but gets no disc knobs at all.
    vector<spec> contin_specs = {
        {"contin-impulse-of-predicate", "impulse(0<($1))", {C}, {}, false},
        {"contin-sum-with-impulse-predicate", "+($1 impulse(0<($2)))", {C, C}, {}, false},
        {"contin-impulse-of-negated-predicate", "impulse(not(0<($1)))", {C}, {}, false},
    };

    size_t emitted = 0;
    const size_t total = (specs.size() + contin_specs.size()) * 2;
    for (const spec& sp : specs) {
        for (float perm_ratio : {-1.0f, 0.0f}) {
            ++emitted;
            spec named = sp;
            string name = string(sp.name)
                + (perm_ratio < 0.0f ? "-nopairs" : "-pairs");
            named.name = name.c_str();
            emit_case(named, perm_ratio, 1u, emitted == total);
        }
    }
    for (const spec& sp : contin_specs) {
        for (float perm_ratio : {-1.0f, 0.0f}) {
            ++emitted;
            spec named = sp;
            string name = string(sp.name)
                + (perm_ratio < 0.0f ? "-nopairs" : "-pairs");
            named.name = name.c_str();
            emit_case(named, perm_ratio, 1u, emitted == total, id::contin_type);
        }
    }
    cout << "  ],\n  \"logical_reduce\":[\n";

    vector<string> reduce_corpus = {
        // The three shapes docs/REMAINING-WORK.md Sec 2 names as reproduced
        // against classic.
        "0<(+($1 $2))",
        "and(0<(+($1 $2)) $3)",
        "or(0<(*($1 $2)) $3)",
        // logical_rules.cc:78-80's own two named "pathological" examples.
        "0<(1)",
        "0<(/(1 $1))",
        // The is_argument early return (logical_rules.cc:74): the tail must
        // NOT run for a bare argument.
        "0<($1)",
        "not(0<($1))",
        // Constant folding through the predicate.
        "0<(-1)",
        "0<(+(1 -1))",
        "0<(*(2 $1))",
        "0<(+(*(2 $1) *(3 $1)))",
        // Predicates over the transcendental ops and over a division.
        "or(0<(sin($1)) 0<(exp($2)))",
        "and(0<(log($1)) $2)",
        "0<(/($1 $1))",
        "0<(/($1 2))",
        // A predicate nested inside a contin expression inside a predicate.
        "0<(+(impulse(0<($1)) $2))",
        // Boolean-level reduction interacting with predicates.
        "and(0<($1) not(0<($1)))",
        "or(0<(+($1 1)) 0<(+($1 1)))",
        "and(0<(+($1 1)) 0<(+($1 2)))",
        "not(0<(+(0.5 $1)))",
    };
    size_t r_emitted = 0;
    const size_t r_total = reduce_corpus.size() * 2;
    for (int effort : {2, 3}) {
        for (const string& source : reduce_corpus) {
            ++r_emitted;
            emit_reduce_case(source, effort, r_emitted == r_total);
        }
    }
    cout << "  ]\n}\n";
    return 0;
}
