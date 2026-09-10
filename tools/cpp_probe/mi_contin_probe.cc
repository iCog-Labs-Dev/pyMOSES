// tools/cpp_probe/mi_contin_probe.cc
//
// Oracle for the TWO CONTINUOUS branches of
// `mutualInformation(const CTable&, const FeatureSet&)`
// (moses/comboreduct/table/table.h:1132) that
// `pymetta/feature_selection/scorers/mutual_info.py` previously gated with
// `NotImplementedError`, plus the cogutil machinery each branch needs:
//
//   * contin OUTPUT with all-discrete inputs -- table.h:1170-1189 (the
//     `min = 100000.0 / max = 0.0` scan) and table.h:1219-1223 (the
//     `get_discrete_bin` push into the `ioc`/`ycount` keys). Needs
//     `discretize_contin_feature` (table.cc:573) and `get_discrete_bin`
//     (table.cc:583), both keyed off `TARGET_DISCRETIZED_BINS_NUM` == 5
//     (table.h:56).
//
//   * the fully continuous case -- table.h:1250-1292, `-KLD(p,q)` over a
//     `std::multimap<contin_t,contin_t>` walk. Needs `KLD`/`KLDS`
//     (opencog/util/KLD.h:202 / :59), which table.h:44 includes directly.
//
// Also captures `mutualInformationBtwSets` (table.h:1311) on the SAME
// contin tables: that function has no contin-specific branch at all (it
// never touches the output column and never checks the input types), so
// the port's own `_all_discrete_inputs` guard there is a port-only
// restriction. These rows are the measurement that decides it.
//
// JSON groups, all from the REAL classic code, none recomputed by hand:
//
//   "discretize"  -- discretize_contin_feature(min,max) for a spread of
//                    (min,max) pairs, including the two degenerate ones the
//                    `100000.0 / 0.0` seeds produce (a column entirely above
//                    100000, a column entirely negative), plus
//                    get_discrete_bin(intvs, val) over values that land
//                    EXACTLY on a bin boundary -- `if (val < disc_intvs[i])`
//                    is a strict `<`, so a boundary value falls in the
//                    HIGHER bin, and `<=` there is a live mutant.
//
//   "kld"         -- KLD(p,q) on hand-built vectors: duplicates in p (the
//                    `Counter` collapse plus `precompute_delta_p`'s running
//                    `p_x_pre`), duplicates in q (`n_duplicates`), q wholly
//                    below p (cit_q walks to `cend()` and `q_x` becomes
//                    `x_very_last`), q wholly above p (`q_x_pre` stays at
//                    `x_very_first`), unequal sizes, negatives, and a
//                    DELIBERATELY UNSORTED p/q pair -- KLD reads both only
//                    through `Counter<FloatT,FloatT>` and `p.size()`, so
//                    input ORDER cannot matter and the fixture says so.
//
//   "scan"        -- the `min`/`max` a contin-output table's own scan
//                    (table.h:1172-1187) arrives at, transliterated here so
//                    the port has a reference for the seeds themselves. The
//                    end-to-end oracle for the scan is the "mi_contin_out"
//                    MI value, not this row.
//
//   "mi_contin_out" -- mutualInformation(ctable, fs) on the three
//                    boolean-input/contin-output tables, for every single
//                    feature and for every pair/triple.
//
//   "mi_contin_in"  -- mutualInformation(ctable, fs) on the two
//                    contin-input/contin-output tables, one feature at a
//                    time (table.h:1252 asserts `fs.size() <= 1`).
//
//   "mi_contin_in_fractional" -- the same, on a ctable whose counts have
//                    been scaled to non-integral `count_t` values, so that
//                    `unsigned flt_count = val_pair.second;` (table.h:1266)
//                    actually truncates. A weighted CSV would reach the same
//                    place; mutating the counts in-place keeps the fixture's
//                    input a plain CSV.
//
//   "mi_btw_sets" -- mutualInformationBtwSets(ctable, fs_l, fs_r) on the
//                    contin tables AND on a contin-output/discrete-input
//                    table.
//
// Build/run: tools/capture_mi_contin.py (bounded per this project's
// standing OOM caution: `( ulimit -v 2097152; timeout 300 <cmd> )`).

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/type_checker/type_tree.h>
#include <opencog/util/KLD.h>

#include <cstdio>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

typedef set<arity_t> probe_feature_set;

static string json_string(const string &value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

static void print_doubles(const vector<contin_t> &v)
{
    printf("[");
    for (size_t i = 0; i < v.size(); i++) {
        if (i) printf(",");
        printf("%.17g", (double)v[i]);
    }
    printf("]");
}

static void print_feature_set(const probe_feature_set &fs)
{
    printf("[");
    bool first = true;
    for (arity_t f : fs) {
        if (!first) printf(",");
        first = false;
        printf("%d", (int)f);
    }
    printf("]");
}

// ---------------------------------------------------------------------
// "discretize"
// ---------------------------------------------------------------------

static void dump_discretize_case(contin_t lo, contin_t hi,
                                 const vector<contin_t> &vals)
{
    vector<contin_t> intvs = discretize_contin_feature(lo, hi);
    printf("{\"min\":%.17g,\"max\":%.17g,\"intervals\":", (double)lo, (double)hi);
    print_doubles(intvs);
    printf(",\"bins\":[");
    for (size_t i = 0; i < vals.size(); i++) {
        if (i) printf(",");
        // `builtin` is an int-valued enum; the ordinal is the whole point
        // (table.h:63-65's own comment: "builtin is not used to represent
        // builtins but to trick vertex to represent discretized contin
        // values"), so dump the ordinal, not a name.
        printf("{\"value\":%.17g,\"bin\":%d}", (double)vals[i],
               (int)get_discrete_bin(intvs, vals[i]));
    }
    printf("]}");
}

// ---------------------------------------------------------------------
// "kld"
// ---------------------------------------------------------------------

static void dump_kld_case(const char *name, const vector<contin_t> &p,
                          const vector<contin_t> &q)
{
    printf("{\"name\":%s,\"p\":", json_string(name).c_str());
    print_doubles(p);
    printf(",\"q\":");
    print_doubles(q);
    printf(",\"kld\":%.17g", (double)KLD(p, q));
    printf(",\"neg_kld\":%.17g}", (double)(-KLD(p, q)));
}

// ---------------------------------------------------------------------
// "klds_direct" -- KLDS<contin_t> driven through its own public API with a
// HAND-BUILT q_counter. This is the only way to reach `boost::accumulate(
// q_counter | map_values, 0)`'s `int` init with non-integral counts:
// `KLD(p,q)` always builds q_counter with `Counter<FloatT,FloatT>`, whose
// every value is a whole number, so the per-step truncation is invisible
// through that entry point. Here it is not.
// ---------------------------------------------------------------------

static void dump_klds_direct(const char *name, const vector<contin_t> &p,
                             const map<contin_t, contin_t> &q_counter)
{
    KLDS<contin_t> klds(p);
    printf("{\"name\":%s,\"p\":", json_string(name).c_str());
    print_doubles(p);
    printf(",\"q_counter\":[");
    bool first = true;
    for (const auto &kv : q_counter) {
        if (!first) printf(",");
        first = false;
        printf("[%.17g,%.17g]", (double)kv.first, (double)kv.second);
    }
    printf("],\"p_size\":%u,\"p_pdf_size\":%u",
           (unsigned)klds.p_size(), (unsigned)klds.p_pdf_size());
    vector<contin_t> comps;
    {
        // The Out-iterator overload (KLD.h:170) shares `next` with
        // operator() but divides each component by p_s and drops the -1.
        KLDS<contin_t> klds_comp(p);
        klds_comp(q_counter, back_inserter(comps));
    }
    printf(",\"components\":");
    print_doubles(comps);
    printf(",\"kld\":%.17g}", (double)klds(q_counter));
}

// ---------------------------------------------------------------------
// "scan" -- table.h:1172-1187, verbatim
// ---------------------------------------------------------------------

static void dump_scan(const char *name, const CTable &ctable)
{
    contin_t min = 100000.0;
    contin_t max = 0.0;
    for (const auto &row : ctable) {
        for (const auto &val_pair : row.second) {
            const vertex &v = val_pair.first.value;
            if (get_contin(v) < min) min = get_contin(v);
            if (get_contin(v) > max) max = get_contin(v);
        }
    }
    vector<contin_t> intvs = discretize_contin_feature(min, max);
    printf("%s:{\"min\":%.17g,\"max\":%.17g,\"intervals\":",
           json_string(name).c_str(), (double)min, (double)max);
    print_doubles(intvs);
    printf("}");
}

// ---------------------------------------------------------------------
// "mi_contin_out" / "mi_contin_in"
// ---------------------------------------------------------------------

static void dump_mi_cases(const char *name, const CTable &ctable,
                          const vector<probe_feature_set> &sets)
{
    printf("%s:[", json_string(name).c_str());
    for (size_t i = 0; i < sets.size(); i++) {
        if (i) printf(",");
        printf("{\"fs\":");
        print_feature_set(sets[i]);
        printf(",\"mi\":%.17g}", mutualInformation(ctable, sets[i]));
    }
    printf("]");
}

static void dump_btw_sets(const char *name, const CTable &ctable,
                          const vector<pair<probe_feature_set, probe_feature_set>> &pairs)
{
    printf("%s:[", json_string(name).c_str());
    for (size_t i = 0; i < pairs.size(); i++) {
        if (i) printf(",");
        printf("{\"fs_l\":");
        print_feature_set(pairs[i].first);
        printf(",\"fs_r\":");
        print_feature_set(pairs[i].second);
        printf(",\"mi\":%.17g}",
               mutualInformationBtwSets(ctable, pairs[i].first, pairs[i].second));
    }
    printf("]");
}

// Dump the compressed table itself, so the Python side compares against the
// SAME rows classic saw rather than against its own re-load of the CSV.
// Boolean inputs are dumped as their `builtin` ordinal, contin as a double.
static void dump_ctable(const char *name, const CTable &ctable)
{
    printf("%s:{\"arity\":%u,\"rows\":[", json_string(name).c_str(),
           (unsigned)ctable.get_arity());
    bool first_row = true;
    for (const auto &row : ctable) {
        if (!first_row) printf(",");
        first_row = false;
        printf("{\"key\":[");
        const multi_type_seq &key = row.first;
        const type_tree &tsig = ctable.get_signature();
        unsigned i = 0;
        for (const type_tree &in_tt : get_signature_inputs(tsig)) {
            if (i) printf(",");
            type_node tn = get_type_node(in_tt);
            if (tn == id::contin_type)
                printf("%.17g", (double)key.get_at<contin_t>(i));
            else
                printf("%d", (int)key.get_at<builtin>(i));
            i++;
        }
        printf("],\"counts\":[");
        bool first_v = true;
        for (const auto &val_pair : row.second) {
            if (!first_v) printf(",");
            first_v = false;
            const vertex &v = val_pair.first.value;
            if (ctable.get_output_type() == id::contin_type)
                printf("{\"y\":%.17g,\"n\":%.17g}", (double)get_contin(v),
                       (double)val_pair.second);
            else
                printf("{\"y\":%d,\"n\":%.17g}", (int)get_builtin(v),
                       (double)val_pair.second);
        }
        printf("]}");
    }
    printf("]}");
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <bins-csv> <neg-csv> <big-csv> <ties-csv> <a4-csv>\n",
                argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Table bins_t = loadTable(argv[1], "y");
    CTable bins = bins_t.compressed();
    Table neg_t = loadTable(argv[2], "y");
    CTable neg = neg_t.compressed();
    Table big_t = loadTable(argv[3], "y");
    CTable big = big_t.compressed();
    Table ties_t = loadTable(argv[4], "y");
    CTable ties = ties_t.compressed();
    Table a4_t = loadTable(argv[5], "y");
    CTable a4 = a4_t.compressed();

    printf("{\n");

    // ---------------- discretize ----------------
    printf("\"discretize\":[\n");
    dump_discretize_case(0.0, 10.0,
                         {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          -1.0, 11.0, 1.9999999999999998, 2.0000000000000004});
    printf(",\n");
    // A column entirely above 100000: `min` never moves off its seed.
    dump_discretize_case(100000.0, 100011.0,
                         {100000.0, 100002.2, 100004.4, 100006.6, 100008.8,
                          100011.0, 100001.0, 100010.0});
    printf(",\n");
    // A column entirely negative: `max` never moves off its seed.
    dump_discretize_case(-10.0, 0.0,
                         {-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, -0.5, -10.5, 0.0});
    printf(",\n");
    // min == max (a constant column): every interval collapses onto min.
    dump_discretize_case(3.0, 3.0, {2.0, 3.0, 4.0});
    printf(",\n");
    // max < min -- reachable only through the seeds, and it makes `interval`
    // NEGATIVE, so `disc_intvs` is DESCENDING and every `val < intvs[i]`
    // decision flips. Not a hypothetical: a column entirely in (0, 100000)
    // cannot produce it, but the seeds themselves are (100000, 0).
    dump_discretize_case(100000.0, 0.0, {-1.0, 0.0, 50000.0, 100000.0, 200000.0});
    printf("\n],\n");

    // ---------------- kld ----------------
    printf("\"kld\":[\n");
    dump_kld_case("identical", {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0});
    printf(",\n");
    dump_kld_case("dup_in_p", {1.0, 1.0, 2.0, 3.0, 3.0, 3.0},
                  {2.0, 2.0, 2.0, 5.0, 5.0, 7.0});
    printf(",\n");
    dump_kld_case("dup_in_q", {1.0, 2.0, 3.0, 4.0}, {2.0, 2.0, 2.0, 2.0});
    printf(",\n");
    dump_kld_case("q_below_p", {10.0, 11.0, 12.0}, {1.0, 2.0, 3.0});
    printf(",\n");
    dump_kld_case("q_above_p", {1.0, 2.0, 3.0}, {10.0, 11.0, 12.0});
    printf(",\n");
    dump_kld_case("unequal_sizes", {1.0, 2.0, 3.0, 4.0, 5.0}, {1.5, 2.5});
    printf(",\n");
    dump_kld_case("negatives", {-3.0, -2.0, -2.0, 0.5, 4.0},
                  {-2.5, -2.5, 1.0, 3.0});
    printf(",\n");
    dump_kld_case("singleton_p", {2.0}, {1.0, 3.0, 3.0});
    printf(",\n");
    // Same multisets as "dup_in_p", presented in a scrambled order. KLD reads
    // p only via Counter(p)/p.size() and q only via Counter(q), so this MUST
    // agree with "dup_in_p" to the last bit.
    dump_kld_case("dup_in_p_scrambled", {3.0, 1.0, 3.0, 2.0, 1.0, 3.0},
                  {5.0, 2.0, 7.0, 2.0, 5.0, 2.0});
    printf(",\n");
    dump_kld_case("wide_range", {0.125, 0.25, 0.25, 8.0, 1024.0},
                  {0.5, 0.5, 2.0, 512.0, 4096.0});
    printf("\n],\n");

    // ---------------- klds_direct ----------------
    printf("\"klds_direct\":[\n");
    // Whole-number counts: the `int` init is invisible here (sum stays
    // integral at every step), which is the domain KLD(p,q) itself reaches.
    dump_klds_direct("integral_counts", {1.0, 2.0, 3.0, 4.0},
                     {{1.5, 1.0}, {2.5, 2.0}, {3.5, 1.0}});
    printf(",\n");
    // Non-integral counts: 0 + 1.5 -> 1, 1 + 2.5 -> 3. Exact sum is 4.0,
    // the accumulated q_s is 3.
    dump_klds_direct("fractional_counts", {1.0, 2.0, 3.0, 4.0},
                     {{1.5, 1.5}, {2.5, 2.5}});
    printf(",\n");
    // 0 + 2.7 -> 2, 2 + 3.9 -> 5. Exact sum is 6.6, accumulated q_s is 5.
    dump_klds_direct("fractional_counts2", {0.5, 1.0, 2.0},
                     {{0.75, 2.7}, {1.75, 3.9}});
    printf("\n],\n");

    // ---------------- scan ----------------
    printf("\"scan\":{");
    dump_scan("bins", bins);
    printf(",");
    dump_scan("neg", neg);
    printf(",");
    dump_scan("big", big);
    printf("},\n");

    // ---------------- ctables ----------------
    printf("\"ctables\":{");
    dump_ctable("bins", bins);
    printf(",");
    dump_ctable("neg", neg);
    printf(",");
    dump_ctable("big", big);
    printf(",");
    dump_ctable("ties", ties);
    printf("},\n");

    // ---------------- mi_contin_out ----------------
    {
        vector<probe_feature_set> sets = {
            {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}, {},
        };
        printf("\"mi_contin_out\":{");
        dump_mi_cases("bins", bins, sets);
        printf(",");
        dump_mi_cases("neg", neg, sets);
        printf(",");
        dump_mi_cases("big", big, sets);
        printf("},\n");
    }

    // ---------------- mi_contin_in ----------------
    {
        printf("\"mi_contin_in\":{");
        dump_mi_cases("ties", ties, {{0}, {1}});
        printf(",");
        dump_mi_cases("a4", a4, {{0}, {1}, {2}, {3}});
        printf("},\n");
    }

    // ---------------- mi_contin_in_fractional ----------------
    // `unsigned flt_count = val_pair.second;` (table.h:1266) truncates a
    // `count_t` (double). Scale every count by 1.5 so the truncation is
    // observable: a count of 1 becomes 1.5 -> 1, a count of 2 becomes
    // 3.0 -> 3. Also drop one count below 1.0 so a pair vanishes entirely.
    {
        CTable frac = ties;
        bool first_cell = true;
        for (auto &row : frac) {
            for (auto &val_pair : row.second) {
                val_pair.second = first_cell ? 0.75 : val_pair.second * 1.5;
                first_cell = false;
            }
        }
        printf("\"mi_contin_in_fractional\":{");
        dump_ctable("ties_scaled", frac);
        printf(",");
        dump_mi_cases("ties", frac, {{0}, {1}});
        printf("},\n");
    }

    // ---------------- mi_btw_sets ----------------
    {
        vector<pair<probe_feature_set, probe_feature_set>> contin_pairs = {
            {{0}, {1}}, {{1}, {0}}, {{0}, {0}}, {{0}, {}},
        };
        vector<pair<probe_feature_set, probe_feature_set>> bins_pairs = {
            {{0}, {1}}, {{0, 1}, {2}}, {{0}, {0, 1, 2}}, {{}, {1}},
        };
        printf("\"mi_btw_sets\":{");
        dump_btw_sets("ties", ties, contin_pairs);
        printf(",");
        dump_btw_sets("a4", a4, contin_pairs);
        printf(",");
        dump_btw_sets("bins", bins, bins_pairs);
        printf("}\n");
    }

    printf("}\n");
    return 0;
}
