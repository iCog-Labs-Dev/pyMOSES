// tools/cpp_probe/ctable_balance_probe.cc
//
// What does `CTable::balance()` (table.cc:806) actually DO to a CTable?
//
// The body reads
//
//     for (auto iorow : *this)
//         for (auto tvc : iorow.second)
//             tvc.second *= (usize / n) / class_count[tvc.first.value];
//
// -- both range-for loops bind BY VALUE, so `tvc.second` is a field of a
// copy of a copy.  This probe dumps every cell's counts before and after
// the call, on a deliberately class-imbalanced table, so the answer is
// measured rather than read off the source.
//
// It also prints the `uncompressed_size()` and the per-class totals the
// function itself computes, so an implementer can see that the
// *computation* runs (and is correct) even if the write-back does not.
//
// Build (see table_probe.cc's header comment for why the installed
// header tree alone is not enough):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/ctable_balance_probe \
//       tools/cpp_probe/ctable_balance_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// A second mode, `discretize`, dumps discretize_contin_feature
// (table.cc:573) and get_discrete_bin (table.cc:583) over a fixed grid --
// the pair `mutualInformation(const CTable&, FeatureSet)` (table.h:1188/
// :1220) uses to bucket a contin target, and the only callers either
// function has.
//
// Usage: ctable_balance_probe <csv-path> [target-feature] [timestamp-feature]
//        ctable_balance_probe discretize

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

static string quote(const string &s) { return "\"" + s + "\""; }

static string vertex_str(const vertex &v)
{
    if (const builtin *b = boost::get<builtin>(&v)) {
        if (*b == id::logical_true) return "true";
        if (*b == id::logical_false) return "false";
    }
    ostringstream os; os << v; return os.str();
}

// Typed JSON for a vertex, matching tools/cpp_probe/table_probe.cc's own
// encoder: booleans as JSON true/false, contin as a full-precision JSON
// number, everything else as a string.
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

static string ctable_counts_json(const CTable &ct)
{
    ostringstream os;
    os << "[";
    bool first_row = true;
    for (const CTable::value_type &kv : ct) {
        if (!first_row) os << ",";
        first_row = false;
        os << "{\"inputs\":[";
        for (size_t i = 0; i < kv.first.size(); ++i) {
            if (i) os << ",";
            os << vertex_to_json(kv.first.get_at<vertex>(i));
        }
        os << "],\"counts\":[";
        bool first_c = true;
        for (const auto &vc : kv.second) {
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

// ---- mode: discretize ----------------------------------------------

static int mode_discretize()
{
    // (min, max) pairs, chosen to cover an ordinary positive range, a
    // range straddling zero, a negative range, a degenerate min==max
    // (interval 0), and an inverted min>max (negative interval), since
    // nothing in table.cc validates the arguments.
    const double ranges[][2] = {
        {0.0, 10.0}, {-1.0, 1.0}, {-10.0, -2.0}, {3.0, 3.0}, {5.0, 1.0},
        {0.0, 1.0}, {-2.5, 7.25}, {1e-8, 1e8},
    };
    // Values probed against each range's bin edges.
    const double vals[] = {
        -1e9, -10.0, -2.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.25, 10.0, 1e9,
    };

    printf("{\n  \"TARGET_DISCRETIZED_BINS_NUM\": %d,\n  \"cases\": [\n",
           TARGET_DISCRETIZED_BINS_NUM);
    bool first = true;
    for (const auto &r : ranges) {
        vector<contin_t> intvs = discretize_contin_feature(r[0], r[1]);
        if (!first) printf(",\n");
        first = false;
        printf("    {\"min\": %.17g, \"max\": %.17g, \"intervals\": [", r[0], r[1]);
        for (size_t i = 0; i < intvs.size(); ++i) {
            if (i) printf(",");
            printf("%.17g", (double)intvs[i]);
        }
        printf("], \"bins\": [");
        bool fv = true;
        for (double v : vals) {
            if (!fv) printf(",");
            fv = false;
            builtin b = get_discrete_bin(intvs, v);
            ostringstream os; os << b;
            printf("[%.17g,%d,%s]", v, (int)b, quote(os.str()).c_str());
        }
        printf("]}");
    }
    printf("\n  ]\n}\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <csv-path> [target-feature] [timestamp-feature]\n", argv[0]);
        fprintf(stderr, "       %s discretize\n", argv[0]);
        return 1;
    }
    if (string(argv[1]) == "discretize") return mode_discretize();
    string path = argv[1];
    string target_feature = argc > 2 ? argv[2] : "";
    string timestamp_feature = argc > 3 ? argv[3] : "";
    vector<string> ignore_features;

    Table table = loadTable(path, target_feature, timestamp_feature, ignore_features);
    CTable ctable = table.compressed("");

    string before = ctable_counts_json(ctable);
    double usize_before = ctable.uncompressed_size();

    // Re-run balance()'s own class_count computation here, so the
    // numbers it works from are visible even though it never writes
    // them back.
    Counter<vertex, count_t> class_count;
    for (auto iorow : ctable) class_count += iorow.second.untimedCounter();

    ctable.balance();

    string after = ctable_counts_json(ctable);
    double usize_after = ctable.uncompressed_size();

    printf("{\n");
    printf("  \"output_type\": \"%s\",\n",
           [&]{ ostringstream os; os << ctable.get_output_type(); return os.str(); }().c_str());
    printf("  \"class_count\": [");
    bool f = true;
    for (const auto &vc : class_count) {
        if (!f) printf(",");
        f = false;
        printf("[%s,%.17g]", vertex_to_json(vc.first).c_str(), (double)vc.second);
    }
    printf("],\n");
    printf("  \"n_classes\": %zu,\n", class_count.size());
    printf("  \"uncompressed_size_before\": %.17g,\n", usize_before);
    printf("  \"uncompressed_size_after\": %.17g,\n", usize_after);
    printf("  \"ctable_before\": %s,\n", before.c_str());
    printf("  \"ctable_after\": %s,\n", after.c_str());
    printf("  \"balance_changed_anything\": %s\n", (before == after) ? "false" : "true");
    printf("}\n");
    return 0;
}
