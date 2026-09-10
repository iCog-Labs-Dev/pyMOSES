// tools/cpp_probe/ttable_probe.cc
//
// Dumps the TIMESTAMP half of loadTable()/Table::compressed(), which
// tools/cpp_probe/table_probe.cc deliberately drops: the TTable itself
// (table.h:452), Table::timestamp_pos, the CTable's TimedCounter cells
// with their TimedValue timestamps intact (table.h:472/:496),
// CTable::get_timestamps() (table.cc:735) and CTable::ordered_by_time()
// (table.cc:794).
//
// It ALSO dumps, alongside each real TimedCounter, the "intended"
// counter -- the same insertions replayed into a map ordered by a
// genuinely lexicographic `(value, timestamp)` comparator, i.e. what
// `TimedValue::operator<` (table.h:482) would give if it carried the
// `value == r.value &&` guard it is missing.  The two differ (in
// CONTENTS, not merely order) whenever a cell holds observations whose
// value order and timestamp order disagree; see
// tools/cpp_probe/timed_counter_probe.cc for the systematic measurement.
//
// Build (see table_probe.cc's header comment for why the installed
// header tree alone is not enough):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/ttable_probe tools/cpp_probe/ttable_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Usage: ttable_probe <csv-path> [target-feature] [timestamp-feature] [ignore-feature...]

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

static string json_escape(const string &s)
{
    string out;
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

static string quote(const string &s) { return "\"" + json_escape(s) + "\""; }

static string date_json(const TTable::value_type &d)
{
    if (d.is_not_a_date()) return "null";
    return quote(TTable::to_string(d));
}

static string vertex_to_json(const vertex &v)
{
    if (const builtin *b = boost::get<builtin>(&v)) {
        if (*b == id::logical_true) return "true";
        if (*b == id::logical_false) return "false";
        ostringstream os; os << *b; return quote(os.str());
    }
    if (const contin_t *c = boost::get<contin_t>(&v)) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", (double)*c);
        return string(buf);
    }
    if (const enum_t *e = boost::get<enum_t>(&v)) return quote(e->getContent());
    ostringstream os; os << v; return quote(os.str());
}

static string count_json(count_t c)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", (double)c);
    return string(buf);
}

// The comparator TimedValue::operator< was meant to be.
struct LexLess {
    bool operator()(const TimedValue &a, const TimedValue &b) const
    {
        if (a.value < b.value) return true;
        if (b.value < a.value) return false;
        return a.timestamp < b.timestamp;
    }
};

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <csv-path> [target-feature] [timestamp-feature] "
                "[ignore-feature...]\n", argv[0]);
        return 1;
    }
    string path = argv[1];
    string target_feature = argc > 2 ? argv[2] : "";
    string timestamp_feature = argc > 3 ? argv[3] : "";
    vector<string> ignore_features;
    for (int i = 4; i < argc; ++i) ignore_features.push_back(argv[i]);

    Table table = loadTable(path, target_feature, timestamp_feature, ignore_features);

    // Replay the compression by hand, in Table::compressed's own row
    // order (table.cc:509-517), into a lexicographically-ordered map, so
    // the fixture carries the INTENDED result next to the real one.
    map<multi_type_seq, map<TimedValue, count_t, LexLess>> intended;
    {
        ITable::const_iterator in_it = table.itable.begin();
        OTable::const_iterator out_it = table.otable.begin();
        if (table.ttable.empty()) {
            for (; in_it != table.itable.end(); ++in_it, ++out_it)
                intended[*in_it][TimedValue(*out_it)] += 1.0;
        } else {
            TTable::const_iterator t_it = table.ttable.begin();
            for (; in_it != table.itable.end(); ++in_it, ++out_it, ++t_it)
                intended[*in_it][TimedValue(*out_it, *t_it)] += 1.0;
        }
    }

    CTable ctable = table.compressed("");

    printf("{\n");
    printf("  \"arity\": %d,\n", (int)table.get_arity());
    printf("  \"target_pos\": %u,\n", table.target_pos);
    printf("  \"timestamp_pos\": %u,\n", table.timestamp_pos);
    printf("  \"ttable_label\": %s,\n", quote(table.ttable.get_label()).c_str());
    printf("  \"ttable\": [");
    for (size_t i = 0; i < table.ttable.size(); ++i) {
        if (i) printf(",");
        printf("%s", date_json(table.ttable[i]).c_str());
    }
    printf("],\n");

    // The real CTable, timestamps intact, in the container's OWN
    // iteration order.  `map_size` is std::map::size(); `iterated` is
    // how many cells iteration actually visits -- they disagree when the
    // broken comparator has corrupted the tree.
    printf("  \"ctable_timed\": [");
    bool first_row = true;
    size_t total_iterated = 0;
    double total_count = 0.0;
    for (const CTable::value_type &kv : ctable) {
        if (!first_row) printf(",");
        first_row = false;
        printf("{\"inputs\":[");
        for (size_t i = 0; i < kv.first.size(); ++i) {
            if (i) printf(",");
            printf("%s", vertex_to_json(kv.first.get_at<vertex>(i)).c_str());
        }
        printf("],\"cell_map_size\":%zu,\"counts\":[", kv.second.size());
        bool first_c = true;
        size_t iterated = 0;
        for (const auto &vc : kv.second) {
            if (!first_c) printf(",");
            first_c = false;
            ++iterated;
            total_count += vc.second;
            printf("[%s,%s,%s]", vertex_to_json(vc.first.value).c_str(),
                   date_json(vc.first.timestamp).c_str(),
                   count_json(vc.second).c_str());
        }
        total_iterated += iterated;
        printf("],\"cell_iterated\":%zu}", iterated);
    }
    printf("],\n");
    printf("  \"ctable_total_iterated_cells\": %zu,\n", total_iterated);
    printf("  \"ctable_total_count\": %s,\n", count_json(total_count).c_str());
    printf("  \"ctable_uncompressed_size\": %s,\n",
           count_json(ctable.uncompressed_size()).c_str());

    // The intended (lexicographic) result, same shape.
    printf("  \"ctable_timed_intended\": [");
    first_row = true;
    double intended_total = 0.0;
    for (const auto &kv : intended) {
        if (!first_row) printf(",");
        first_row = false;
        printf("{\"inputs\":[");
        for (size_t i = 0; i < kv.first.size(); ++i) {
            if (i) printf(",");
            printf("%s", vertex_to_json(kv.first.get_at<vertex>(i)).c_str());
        }
        printf("],\"counts\":[");
        bool first_c = true;
        for (const auto &vc : kv.second) {
            if (!first_c) printf(",");
            first_c = false;
            intended_total += vc.second;
            printf("[%s,%s,%s]", vertex_to_json(vc.first.value).c_str(),
                   date_json(vc.first.timestamp).c_str(),
                   count_json(vc.second).c_str());
        }
        printf("]}");
    }
    printf("],\n");
    printf("  \"ctable_intended_total_count\": %s,\n", count_json(intended_total).c_str());

    // CTable::get_timestamps() -- note table.cc:740 filters out the
    // default not-a-date timestamp.
    printf("  \"get_timestamps\": [");
    {
        set<TTable::value_type> ts = ctable.get_timestamps();
        bool f = true;
        for (const auto &t : ts) {
            if (!f) printf(",");
            f = false;
            printf("%s", date_json(t).c_str());
        }
    }
    printf("],\n");

    // CTable::ordered_by_time()
    printf("  \"ordered_by_time\": [");
    {
        CTableTime ctt = ctable.ordered_by_time();
        bool f = true;
        for (const auto &tio : ctt) {
            if (!f) printf(",");
            f = false;
            printf("{\"timestamp\":%s,\"counts\":[", date_json(tio.first).c_str());
            bool fc = true;
            for (const auto &vc : tio.second) {
                if (!fc) printf(",");
                fc = false;
                printf("[%s,%s]", vertex_to_json(vc.first).c_str(),
                       count_json(vc.second).c_str());
            }
            printf("]}");
        }
    }
    printf("]\n}\n");
    return 0;
}
