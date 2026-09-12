// tools/cpp_probe/table_probe.cc
//
// Dumps loadTable()/Table::compressed("") output as JSON, for
// differential testing against pymoses/comboreduct/table/{table,
// table_io}.py.
//
// Build (see task-5-brief.md / CLAUDE.md's verified probe recipe --
// compiling against the installed headers alone fails: table.h includes
// ../interpreter/eval.h, which is not installed):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/table_probe tools/cpp_probe/table_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Usage: table_probe <csv-path> [target-feature] [timestamp-feature] [ignore-feature...]
// (pass "" for target-feature/timestamp-feature to leave them unset while
// still specifying ignore-features)

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

// ---- tiny JSON helpers ----------------------------------------------

static std::string json_escape(const std::string &s)
{
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out;
}

static std::string quote(const std::string &s) { return "\"" + json_escape(s) + "\""; }

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

static std::string type_node_name(type_node t)
{
    switch (t) {
        case id::lambda_type: return "lambda_type";
        case id::application_type: return "application_type";
        case id::union_type: return "union_type";
        case id::arg_list_type: return "arg_list_type";
        case id::boolean_type: return "boolean_type";
        case id::contin_type: return "contin_type";
        case id::enum_type: return "enum_type";
        case id::list_type: return "list_type";
        case id::action_result_type: return "action_result_type";
        case id::definite_object_type: return "definite_object_type";
        case id::action_definite_object_type: return "action_definite_object_type";
        case id::indefinite_object_type: return "indefinite_object_type";
        case id::message_type: return "message_type";
        case id::action_symbol_type: return "action_symbol_type";
        case id::wild_card_type: return "wild_card_type";
        case id::ann_type: return "ann_type";
        case id::unknown_type: return "unknown_type";
        case id::ill_formed_type: return "ill_formed_type";
        case id::argument_type: return "argument_type";
        default: return "unknown";
    }
}

static std::string type_array(const std::vector<type_node> &ts)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < ts.size(); ++i) {
        if (i) os << ",";
        os << quote(type_node_name(ts[i]));
    }
    os << "]";
    return os.str();
}

// vertex -> JSON value. Booleans as JSON true/false; contin as a JSON
// number at full round-trip precision; enum_t (and the ill_formed_type
// "ugly hack" columns, which are also enum_t under the hood -- see
// table_io.cc's token_to_vertex) as a JSON string of its raw content
// (NOT via vertex's own ostream operator, which wraps enum_t as
// `enum:(<id>):"<content>"` -- see enum_type.cc:66-70 -- useless for a
// differential oracle).
static std::string vertex_to_json(const vertex &v)
{
    if (const builtin *b = boost::get<builtin>(&v)) {
        if (*b == id::logical_true) return "true";
        if (*b == id::logical_false) return "false";
        std::ostringstream os;
        os << *b;
        return quote(os.str());
    }
    if (const contin_t *c = boost::get<contin_t>(&v)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", (double)*c);
        return std::string(buf);
    }
    if (const enum_t *e = boost::get<enum_t>(&v)) {
        return quote(e->getContent());
    }
    std::ostringstream os;
    os << v;
    return quote(os.str());
}

static std::string vertex_array(const vertex_seq &vs)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < vs.size(); ++i) {
        if (i) os << ",";
        os << vertex_to_json(vs[i]);
    }
    os << "]";
    return os.str();
}

static std::string itable_to_json(const ITable &it)
{
    std::ostringstream os;
    os << "[";
    for (size_t r = 0; r < it.size(); ++r) {
        if (r) os << ",";
        const ITable::value_type &row = it[r];
        os << "[";
        for (size_t c = 0; c < row.size(); ++c) {
            if (c) os << ",";
            os << vertex_to_json(row.get_at<vertex>(c));
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

static std::string otable_to_json(const OTable &ot)
{
    std::ostringstream os;
    os << "[";
    for (size_t r = 0; r < ot.size(); ++r) {
        if (r) os << ",";
        os << vertex_to_json(ot[r]);
    }
    os << "]";
    return os.str();
}

static std::string ctable_to_json(const CTable &ct)
{
    std::ostringstream os;
    os << "[";
    bool first_row = true;
    for (const CTable::value_type &kv : ct) {
        if (!first_row) os << ",";
        first_row = false;
        const multi_type_seq &key = kv.first;
        const CTable::counter_t &counter = kv.second;

        os << "{\"inputs\":[";
        for (size_t i = 0; i < key.size(); ++i) {
            if (i) os << ",";
            os << vertex_to_json(key.get_at<vertex>(i));
        }
        os << "],\"counts\":[";
        bool first_c = true;
        for (const auto &vc : counter) {
            if (!first_c) os << ",";
            first_c = false;
            char buf[64];
            snprintf(buf, sizeof(buf), "%.17g", (double)vc.second);
            os << "[" << vertex_to_json(vc.first.value) << "," << buf << "]";
        }
        os << "]}";
    }
    os << "]";
    return os.str();
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <csv-path> [target-feature] [timestamp-feature] "
                "[ignore-feature...]\n",
                argv[0]);
        return 1;
    }
    std::string path = argv[1];
    std::string target_feature = argc > 2 ? argv[2] : "";
    std::string timestamp_feature = argc > 3 ? argv[3] : "";
    std::vector<std::string> ignore_features;
    for (int i = 4; i < argc; ++i) ignore_features.push_back(argv[i]);

    Table table = loadTable(path, target_feature, timestamp_feature, ignore_features);
    CTable ctable = table.compressed("");

    printf("{\n");
    printf("  \"itable_labels\": %s,\n", str_array(table.itable.get_labels()).c_str());
    printf("  \"itable_types\": %s,\n", type_array(table.itable.get_types()).c_str());
    printf("  \"otable_label\": %s,\n", quote(table.otable.get_label()).c_str());
    printf("  \"otable_type\": %s,\n", quote(type_node_name(table.otable.get_type())).c_str());
    printf("  \"arity\": %d,\n", (int)table.get_arity());
    printf("  \"target_pos\": %u,\n", table.target_pos);
    printf("  \"itable\": %s,\n", itable_to_json(table.itable).c_str());
    printf("  \"otable\": %s,\n", otable_to_json(table.otable).c_str());
    printf("  \"ctable_labels\": %s,\n", str_array(ctable.get_labels()).c_str());
    printf("  \"ctable_uncompressed_size\": %.17g,\n", (double)ctable.uncompressed_size());
    printf("  \"ctable\": %s\n", ctable_to_json(ctable).c_str());
    printf("}\n");
    return 0;
}
