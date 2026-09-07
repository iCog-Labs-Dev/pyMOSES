// Captures the immutable Boolean representation candidate boundary for S8.
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/moses/representation/representation.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

static string json_string(const string& value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
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

static string rendered(const combo_tree& tr)
{
    stringstream out;
    ostream_combo_tree(out, tr);
    return out.str();
}

static void emit_truth_table(const combo_tree& tr, int arity)
{
    try {
        complete_truth_table tt(tr, arity);
        cout << "{\"computed\":true,\"bits\":\"";
        for (bool value : tt) cout << (value ? '1' : '0');
        cout << "\"}";
    } catch (const std::exception& error) {
        cout << "{\"computed\":false,\"error\":"
             << json_string(error.what()) << '}';
    } catch (...) {
        cout << "{\"computed\":false,\"error\":\"unknown exception\"}";
    }
}

static void emit_case(const string& name, const string& source, int arity,
                      const vector<int>& ignored, bool last)
{
    combo_tree exemplar = parse(source);
    type_tree signature = gen_signature(id::boolean_type, arity);
    vertex_set ignore_ops;
    for (int idx : ignored) ignore_ops.insert(argument(idx));

    logical_reduction reduction(ignore_ops);
    const rule& reducer = reduction(2);
    representation rep(reducer, reducer, exemplar, signature, ignore_ops,
                       nullptr, nullptr, false, -1.0);

    vector<unsigned> multiplicities;
    for (auto it = rep.disc.cbegin(); it != rep.disc.cend(); ++it)
        multiplicities.push_back(it->first.multy);

    size_t combinations = 1;
    for (unsigned multy : multiplicities) combinations *= multy;
    const bool exhaustive = combinations <= 1024;

    cout << "    {\"name\":" << json_string(name)
         << ",\"source\":" << json_string(source)
         << ",\"arity\":" << arity << ",\"ignored_arguments\":[";
    for (size_t idx = 0; idx < ignored.size(); ++idx) {
        if (idx) cout << ',';
        cout << ignored[idx];
    }
    cout << "],\"decorated_exemplar\":" << json_string(rendered(rep.exemplar()))
         << ",\"packed_width\":" << rep.fields().packed_width()
         << ",\"raw_size\":" << rep.fields().raw_size()
         << ",\"exhaustive\":" << (exhaustive ? "true" : "false")
         << ",\"knobs\":[";

    size_t order = 0;
    for (auto it = rep.disc.cbegin(); it != rep.disc.cend(); ++it, ++order) {
        if (order) cout << ',';
        auto loc = it->second->get_loc();
        auto raw = rep.it_disc_idx.find(loc);
        cout << "{\"order\":" << order
             << ",\"multy\":" << it->first.multy
             << ",\"text\":" << json_string(it->second->toStr())
             << ",\"location_preorder\":"
             << distance(rep.exemplar().begin(), loc)
             << ",\"raw_idx\":" << raw->second << '}';
    }
    cout << "],\"rows\":[";

    vector<vector<unsigned>> settings;
    if (exhaustive) {
        vector<unsigned> current(multiplicities.size(), 0);
        function<void(size_t)> visit = [&](size_t idx) {
            if (idx == multiplicities.size()) {
                settings.push_back(current);
                return;
            }
            for (unsigned value = 0; value < multiplicities[idx]; ++value) {
                current[idx] = value;
                visit(idx + 1);
            }
        };
        visit(0);
    } else {
        settings.push_back(vector<unsigned>(multiplicities.size(), 0));
        for (size_t idx = 0; idx < multiplicities.size(); ++idx) {
            for (unsigned value = 1; value < multiplicities[idx]; ++value) {
                vector<unsigned> row(multiplicities.size(), 0);
                row[idx] = value;
                settings.push_back(row);
            }
        }
        vector<unsigned> maximum;
        for (unsigned multy : multiplicities) maximum.push_back(multy - 1);
        if (!maximum.empty()) settings.push_back(maximum);
    }

    for (size_t row_idx = 0; row_idx < settings.size(); ++row_idx) {
        if (row_idx) cout << ',';
        instance inst(rep.fields().packed_width(), 0);
        for (size_t idx = 0; idx < settings[row_idx].size(); ++idx)
            rep.fields().set_raw(inst, idx, settings[row_idx][idx]);

        combo_tree raw_candidate = rep.get_candidate(inst, false);
        combo_tree reduced_candidate = rep.get_candidate(inst, true);
        cout << "{\"raw\":[";
        for (size_t idx = 0; idx < settings[row_idx].size(); ++idx) {
            if (idx) cout << ',';
            cout << settings[row_idx][idx];
        }
        cout << "],\"packed\":[";
        for (size_t idx = 0; idx < inst.size(); ++idx) {
            if (idx) cout << ',';
            cout << inst[idx];
        }
        cout << "],\"unreduced\":" << json_string(rendered(raw_candidate))
             << ",\"reduced\":" << json_string(rendered(reduced_candidate))
             << ",\"unreduced_tt\":";
        emit_truth_table(raw_candidate, arity);
        cout << ",\"reduced_tt\":";
        emit_truth_table(reduced_candidate, arity);
        cout << '}';
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
    emit_case("ignored-argument", "and($1 $2)", 2, {2}, false);
    // Discriminates knob_mapper's descending-multy sort (Task 3's exit
    // criterion): insertion order here is [3,3,3,2,2,3,3] (measured), which
    // is NOT already sorted, unlike every other case above. Reversing the
    // sort in knob_mapper.py must change this case's "knobs" order.
    emit_case("and-wraps-or", "and($1 or($2 $3))", 3, {}, true);
    cout << "  ]\n}\n";
    return 0;
}
