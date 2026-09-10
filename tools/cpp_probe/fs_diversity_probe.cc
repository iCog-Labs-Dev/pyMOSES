// tools/cpp_probe/fs_diversity_probe.cc
//
// Oracle for the two gates this slice closes in
// pymetta/moses/deme/feature_selector.py:
//
//   (a) --fs-diversity-pressure > 0  -- feature_selector::rank_feature_sets
//       (feature_selector.cc:343), feature_selector::mi (cc:434) and the
//       ranked branch of feature_selector::select_top_feature_sets
//       (cc:235-245).
//
//   (b) --fs-subsampling-ratio < 1.0 -- subsampleCTable
//       (moses/comboreduct/table/table.cc:899) and the CTable::remove_rows
//       it drives (table.cc:646).
//
// Five sections are dumped as JSON, all from the REAL classic code:
//
//   "ctable"        -- the compressed ctable_unsorted.csv, IN MAP ORDER
//                      (with each row's TimedCounter also in map order).
//                      This is the ground truth for the two iteration
//                      orders remove_rows' index arithmetic depends on; the
//                      fixture is built so map order != insertion order for
//                      both the rows and two of the four rows' counters.
//
//   "remove_rows"   -- CTable::remove_rows(idxs) on a fresh copy of that
//                      ctable for a list of index sets.
//
//   "selector"      -- the raw lazy_random_selector(u) draw SEQUENCE for a
//                      few (u, seed, n) triples. subsampleCTable's own
//                      draws go through this, and draw order is part of the
//                      port's bit-identity claim, so it is compared element
//                      for element, not set-wise.
//
//   "subsample"     -- subsampleCTable(ratio, ctable) on a fresh copy, for
//                      several (ratio, seed) pairs, reporting the derived
//                      nremove/ctable_usize as well as the resulting table.
//
//   "rank"          -- rank_feature_sets(fs_pop) and
//                      select_top_feature_sets(fs_pop) over hand-built
//                      feature_set_pops that contain DELIBERATE TIES (equal
//                      scores, symmetric/duplicate feature sets), for
//                      jaccard and mi variants, several diversity_pressure
//                      values, several diversity_interaction values, and
//                      several n_demes.
//
//   "end_to_end"    -- feature_selector::operator()(exemplar) with
//                      diversity_pressure > 0 and/or subsampling_ratio < 1,
//                      i.e. both gates live at once through the real entry
//                      point.
//
// Build:
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/fs_diversity_probe \
//       tools/cpp_probe/fs_diversity_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lfeature_selection -lmoses -lcomboreduct -lcogutil \
//       -lboost_program_options
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).
//
// Usage: fs_diversity_probe <unsorted-csv> <big-csv> <boolean-csv> <a5-csv>
//        <skew-csv> [weighted-csv]
//        (with the optional 4th argument the probe runs ONLY the
//        OC_ASSERT demonstration described in main(), and aborts.)

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/feature-selection/main/feature-selection.h>
#include <moses/moses/deme/feature_selector.h>
#include <opencog/util/lazy_random_selector.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;

// Exposes the two protected members this probe measures.
struct fs_access : public feature_selector
{
    using feature_selector::feature_selector;
    using feature_selector::mi;
    using feature_selector::rank_feature_sets;
    using feature_selector::select_top_feature_sets;
};

static string json_string(const string &value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

static string vertex_str(const vertex &v)
{
    stringstream ss;
    ss << v;
    return ss.str();
}

static void print_feature_set(const feature_set &fs)
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

// Dumps the ctable in ITS OWN (std::map) order: rows ascending by
// multi_type_seq, each row's TimedCounter ascending by TimedValue.
static void print_ctable(const CTable &ct)
{
    printf("[");
    bool first_row = true;
    for (const auto &vct : ct) {
        if (!first_row) printf(",");
        first_row = false;
        printf("{\"inputs\":[");
        bool first_in = true;
        for (const string &s : vct.first.to_strings()) {
            if (!first_in) printf(",");
            first_in = false;
            printf("%s", json_string(s).c_str());
        }
        printf("],\"outputs\":[");
        bool first_out = true;
        for (const auto &tv : vct.second) {
            if (!first_out) printf(",");
            first_out = false;
            printf("[%s,%.17g]", json_string(vertex_str(tv.first.value)).c_str(),
                   (double)tv.second);
        }
        printf("]}");
    }
    printf("]");
}

static void print_idx_set(const set<unsigned> &idxs)
{
    printf("[");
    bool first = true;
    for (unsigned u : idxs) {
        if (!first) printf(",");
        first = false;
        printf("%u", u);
    }
    printf("]");
}

static const CTable &pick_table(const char *name, const CTable &boolean, const CTable &a5,
                                const CTable &skew)
{
    string n(name);
    if (n == "a5") return a5;
    if (n == "skew") return skew;
    return boolean;
}

// ---------------------------------------------------------------- sections

static void dump_remove_rows(const char *section, const CTable &base,
                             const vector<set<unsigned>> &cases)
{
    printf("\"%s\":[", section);
    bool first = true;
    for (const set<unsigned> &idxs : cases) {
        if (!first) printf(",\n");
        first = false;
        CTable ct = base;
        ct.remove_rows(idxs);
        printf("{\"idxs\":");
        print_idx_set(idxs);
        printf(",\"result\":");
        print_ctable(ct);
        printf(",\"uncompressed_size\":%.17g}", ct.uncompressed_size());
    }
    printf("]");
}

static void dump_selector(const vector<vector<unsigned>> &cases)
{
    // each case is {u, seed, n}
    printf("\"selector\":[");
    bool first = true;
    for (const auto &c : cases) {
        if (!first) printf(",");
        first = false;
        unsigned u = c[0], seed = c[1], n = c[2];
        randGen().seed(seed);
        lazy_random_selector sel(u);
        printf("{\"u\":%u,\"seed\":%u,\"draws\":[", u, seed);
        for (unsigned k = 0; k < n; k++) {
            if (k) printf(",");
            printf("%u", sel());
        }
        printf("]}");
    }
    printf("]");
}

static void dump_subsample(const char *section, const CTable &base,
                           const vector<pair<double, unsigned>> &cases)
{
    printf("\"%s\":[", section);
    bool first = true;
    for (const auto &c : cases) {
        if (!first) printf(",\n");
        first = false;
        double ratio = c.first;
        unsigned seed = c.second;

        // Recompute the two derived unsigneds exactly as subsampleCTable
        // does, so the port can be compared on them separately from the
        // final table.
        unsigned usize = base.uncompressed_size();
        unsigned nremove = (1.0 - (float)ratio) * usize;

        CTable ct = base;
        randGen().seed(seed);
        subsampleCTable((float)ratio, ct);

        printf("{\"ratio\":%.17g,\"seed\":%u,\"ctable_usize\":%u,\"nremove\":%u,"
               "\"result\":",
               ratio, seed, usize, nremove);
        print_ctable(ct);
        printf(",\"uncompressed_size\":%.17g}", ct.uncompressed_size());
    }
    printf("]");
}

static void print_composite_score(const composite_score &cs)
{
    printf("{\"score\":%.17g,\"complexity\":%d,\"complexity_penalty\":%.17g,"
           "\"uniformity_penalty\":%.17g,\"penalized_score\":%.17g}",
           (double)cs.get_score(), (int)cs.get_complexity(),
           (double)cs.get_complexity_penalty(), (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

struct rank_case
{
    const char *name;
    const char *table;   // "boolean" or "a5"
    vector<pair<double, vector<arity_t>>> pop;  // insertion order matters
    double diversity_pressure;
    bool diversity_jaccard;
    int diversity_interaction;
    unsigned n_demes;
};

static void dump_rank(const CTable &boolean, const CTable &a5, const CTable &skew,
                      const vector<rank_case> &cases)
{
    printf("\"rank\":[");
    bool first = true;
    for (const rank_case &rc : cases) {
        if (!first) printf(",\n");
        first = false;

        feature_selector_parameters params;
        params.diversity_pressure = rc.diversity_pressure;
        params.diversity_jaccard = rc.diversity_jaccard;
        params.diversity_interaction = rc.diversity_interaction;
        params.n_demes = rc.n_demes;
        const CTable &ctable = pick_table(rc.table, boolean, a5, skew);
        fs_access fsa(ctable, params);

        feature_set_pop pop;
        for (const auto &entry : rc.pop) {
            feature_set fs(entry.second.begin(), entry.second.end());
            pop.insert({entry.first, fs});
        }

        csc_feature_set_pop ranked = fsa.rank_feature_sets(pop);
        feature_set_pop top = fsa.select_top_feature_sets(pop);

        printf("{\"name\":%s,", json_string(rc.name).c_str());
        printf("\"table\":%s,", json_string(rc.table).c_str());
        printf("\"diversity_pressure\":%.17g,", rc.diversity_pressure);
        printf("\"diversity_jaccard\":%s,", rc.diversity_jaccard ? "true" : "false");
        printf("\"diversity_interaction\":%d,", rc.diversity_interaction);
        printf("\"n_demes\":%u,", rc.n_demes);

        // The input pop, in multimap order -- the port must reproduce this
        // ordering too (its feature_set_pop is a list built by a stable sort).
        printf("\"pop\":[");
        bool f2 = true;
        for (const auto &p : pop) {
            if (!f2) printf(",");
            f2 = false;
            printf("[%.17g,", p.first);
            print_feature_set(p.second);
            printf("]");
        }
        printf("],");

        printf("\"ranked\":[");
        f2 = true;
        for (const auto &p : ranked) {
            if (!f2) printf(",");
            f2 = false;
            printf("{\"cs\":");
            print_composite_score(p.first);
            printf(",\"fs\":");
            print_feature_set(p.second);
            printf("}");
        }
        printf("],");

        printf("\"top\":[");
        f2 = true;
        for (const auto &p : top) {
            if (!f2) printf(",");
            f2 = false;
            printf("[%.17g,", p.first);
            print_feature_set(p.second);
            printf("]");
        }
        printf("]}");
    }
    printf("]");
}

struct mi_case
{
    const char *table;
    vector<arity_t> fs_l;
    vector<arity_t> fs_r;
    int diversity_interaction;
};

// feature_selector::mi() at FULL double precision. rank_feature_sets
// narrows its result to a `float`, which erases the last-ULP differences a
// wrong powerset summation order produces -- so the method is measured
// directly here as well.
static void dump_mi(const CTable &boolean, const CTable &a5, const CTable &skew,
                    const vector<mi_case> &cases)
{
    printf("\"mi\":[");
    bool first = true;
    for (const mi_case &mc : cases) {
        if (!first) printf(",\n");
        first = false;
        feature_selector_parameters params;
        params.diversity_interaction = mc.diversity_interaction;
        fs_access fsa(pick_table(mc.table, boolean, a5, skew), params);
        feature_set fs_l(mc.fs_l.begin(), mc.fs_l.end());
        feature_set fs_r(mc.fs_r.begin(), mc.fs_r.end());
        printf("{\"table\":%s,\"diversity_interaction\":%d,\"fs_l\":",
               json_string(mc.table).c_str(), mc.diversity_interaction);
        print_feature_set(fs_l);
        printf(",\"fs_r\":");
        print_feature_set(fs_r);
        printf(",\"mi\":%.17g}", fsa.mi(fs_l, fs_r));
    }
    printf("]");
}

struct e2e_case
{
    const char *table;       // "boolean" or "a5"
    const char *algorithm;   // "simple" or "hc"
    const char *exemplar;
    unsigned base_target_size;
    bool increase_target_size;
    double diversity_pressure;
    bool diversity_jaccard;
    int diversity_interaction;
    unsigned n_demes;
    unsigned diversity_cap;
    double subsampling_ratio;
    unsigned seed;
};

static void dump_end_to_end(const CTable &boolean, const CTable &a5, const CTable &skew,
                            const vector<e2e_case> &cases)
{
    printf("\"end_to_end\":[");
    bool first = true;
    for (const e2e_case &ec : cases) {
        if (!first) printf(",\n");
        first = false;

        const CTable &ctable = pick_table(ec.table, boolean, a5, skew);

        feature_selector_parameters params;
        params.fs_params.algorithm = ec.algorithm;
        params.fs_params.target_size = ec.base_target_size;
        params.increase_target_size = ec.increase_target_size;
        params.diversity_pressure = ec.diversity_pressure;
        params.diversity_jaccard = ec.diversity_jaccard;
        params.diversity_interaction = ec.diversity_interaction;
        params.n_demes = ec.n_demes;
        params.diversity_cap = ec.diversity_cap;
        params.subsampling_ratio = ec.subsampling_ratio;

        combo_tree xmplr = str2combo_tree(ec.exemplar, std::vector<std::string>());

        feature_selector festor(ctable, params);
        randGen().seed(ec.seed);
        feature_set_pop top = festor(xmplr);

        printf("{\"table\":%s,", json_string(ec.table).c_str());
        printf("\"algorithm\":%s,", json_string(ec.algorithm).c_str());
        printf("\"exemplar\":%s,", json_string(ec.exemplar).c_str());
        printf("\"base_target_size\":%u,", ec.base_target_size);
        printf("\"increase_target_size\":%s,", ec.increase_target_size ? "true" : "false");
        printf("\"diversity_pressure\":%.17g,", ec.diversity_pressure);
        printf("\"diversity_jaccard\":%s,", ec.diversity_jaccard ? "true" : "false");
        printf("\"diversity_interaction\":%d,", ec.diversity_interaction);
        printf("\"n_demes\":%u,", ec.n_demes);
        printf("\"diversity_cap\":%u,", ec.diversity_cap);
        printf("\"subsampling_ratio\":%.17g,", ec.subsampling_ratio);
        printf("\"seed\":%u,", ec.seed);
        printf("\"top\":[");
        bool f2 = true;
        for (const auto &p : top) {
            if (!f2) printf(",");
            f2 = false;
            printf("[%.17g,", p.first);
            print_feature_set(p.second);
            printf("]");
        }
        printf("]}");
    }
    printf("]");
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <unsorted-csv> <big-csv> <boolean-csv> <a5-csv> "
                "<skew-csv> [weighted-csv]\n",
                argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Optional 4th argument: run ONLY the OC_ASSERT demonstration. A
    // weighted CTable has fractional row counts, which trips
    // remove_rows' own `OC_ASSERT(0.0 == (total_row_weights - truncate),
    // "This algo is broken!")` (table.cc:668) and aborts the process --
    // so it cannot share a run with the JSON dump.
    if (argc >= 7) {
        CTable weighted = loadTable(argv[6], "r").compressed("wgt");
        fprintf(stderr, "weighted ctable usize=%.17g rows=%zu\n",
                weighted.uncompressed_size(), weighted.size());
        set<unsigned> idxs{0};
        weighted.remove_rows(idxs);
        fprintf(stderr, "remove_rows returned normally\n");
        return 0;
    }

    CTable unsorted = loadTable(argv[1], "y").compressed();
    CTable big = loadTable(argv[2], "y").compressed();
    CTable boolean = loadTable(argv[3], "y").compressed();
    CTable a5 = loadTable(argv[4], "y").compressed();
    CTable skew = loadTable(argv[5], "y").compressed();

    printf("{\n");

    printf("\"ctable\":");
    print_ctable(unsorted);
    printf(",\n");

    vector<set<unsigned>> rr_cases = {
        {},
        {0},
        {1},
        {0, 1},
        {2, 3},
        {4},
        {5},
        {5, 6, 7},
        {0, 2, 4, 6},
        {1, 3, 5, 7},
        {2, 5},
        {0, 1, 2, 3, 4, 5, 6, 7},
    };
    dump_remove_rows("remove_rows", unsorted, rr_cases);
    printf(",\n");

    printf("\"ctable_big\":");
    print_ctable(big);
    printf(",\n");

    // Index sets whose Python `set` iteration order is NOT ascending (any
    // element >= 8 puts a small-int set's slot order out of value order),
    // so the `sorted(idxs)` in ctable_remove_rows is exercised rather than
    // accidentally satisfied. The big table also has EIGHT compressed rows
    // whose map order (1111, 1100, 1010, 1001, 0110, 0101, 0011, 0000) is
    // nothing like the order they were compressed in.
    vector<set<unsigned>> rr_big_cases = {
        {3, 9},
        {5, 13},
        {7, 15},
        {3, 11, 19},
        {2, 3, 9},
        {4, 12, 20, 28},
        {1, 10, 17},
        {9, 3, 26, 5},
        {29, 2, 11},
        {0, 8, 16, 24},
        {13, 5, 21, 29, 2},
        {4, 5, 6, 7, 20, 21, 22},
        {28, 29},
        {0, 29},
        {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
    };
    dump_remove_rows("remove_rows_big", big, rr_big_cases);
    printf(",\n");

    vector<vector<unsigned>> sel_cases = {
        {8, 1, 8}, {8, 2, 4}, {8, 7, 6}, {16, 1, 10}, {3, 5, 3},
    };
    dump_selector(sel_cases);
    printf(",\n");

    vector<pair<double, unsigned>> ss_cases = {
        {1.0, 1},  {0.9, 1},  {0.75, 1}, {0.75, 2},  {0.5, 1},
        {0.5, 3},  {0.25, 1}, {0.1, 1},  {0.0, 1},   {0.7, 4},
    };
    dump_subsample("subsample", unsorted, ss_cases);
    printf(",\n");

    // Ratios chosen so the binary32 narrowing of the `float ratio`
    // PARAMETER changes `nremove`: with usize 8, (1.0 - 0.750000001) * 8
    // truncates to 1 in double and to 2 once `ratio` is narrowed to
    // float32 (0.75f exactly). Every such pair is a live discriminator for
    // the `_to_float32` call in subsampleCTable.
    vector<pair<double, unsigned>> ss_f32_cases = {
        {0.875000001, 1}, {0.750000001, 1}, {0.750000001, 5}, {0.625000001, 1},
        {0.500000001, 1}, {0.375000001, 2}, {0.250000001, 1}, {0.125000001, 1},
    };
    dump_subsample("subsample_f32", unsorted, ss_f32_cases);
    printf(",\n");

    vector<pair<double, unsigned>> ss_big_cases = {
        {0.9, 1},  {0.8, 1},  {0.75, 1}, {0.5, 1},  {0.5, 11},
        {0.25, 1}, {0.1, 2},  {0.0, 1},  {0.966666667, 1}, {0.033333333, 4},
    };
    dump_subsample("subsample_big", big, ss_big_cases);
    printf(",\n");

    // Feature-set populations with DELIBERATE TIES:
    //  * equal scores across several sets (multimap insertion-order tie),
    //  * symmetric sets of equal size (equal jaccard index against a common
    //    neighbour -> equal uniformity penalty -> composite_score ties),
    //  * a duplicated set (identical score AND identical members).
    vector<rank_case> rank_cases = {
        {"tie3_jaccard_p1", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         1.0, true, 0, 3},
        {"tie3_jaccard_p0.25", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.25, true, 0, 3},
        {"tie3_jaccard_reversed_insert", "boolean",
         {{0.5, {2}}, {0.5, {1}}, {0.5, {0}}},
         1.0, true, 0, 3},
        {"dup_set_jaccard", "boolean",
         {{0.5, {0}}, {0.5, {0}}, {0.5, {1}}},
         1.0, true, 0, 3},
        {"mixed_sizes_jaccard", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.5, true, 0, 5},
        {"mixed_sizes_jaccard_n2", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.5, true, 0, 2},
        {"mixed_sizes_jaccard_n1", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.5, true, 0, 1},
        {"tie3_mi_p1", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         1.0, false, 0, 3},
        {"tie3_mi_interaction_neg", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         1.0, false, -1, 3},
        {"mixed_sizes_mi_interaction_neg", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.5, false, -1, 5},
        {"mixed_sizes_mi_interaction_0", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.5, false, 0, 5},
        {"mixed_sizes_mi_interaction_1", "boolean",
         {{0.9, {0, 1}}, {0.9, {1, 2}}, {0.5, {0, 2}}, {0.5, {1, 2}}},
         0.5, false, 1, 4},
        {"distinct_scores_jaccard", "boolean",
         {{0.9, {0}}, {0.7, {1}}, {0.3, {2}}},
         1.0, true, 0, 3},
        {"big_pressure_jaccard", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         100.0, true, 0, 3},
        // The a5 table (bool_a5_maj.csv) is a 5-feature MAJORITY function --
        // every input column is interchangeable, so every same-size feature
        // set has the SAME mutual information with the target and with any
        // other same-size set. That symmetry is what makes these cases
        // exercise tie-breaking rather than merely ranking.
        {"a5_tie5_mi_p1", "a5",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}, {0.5, {3}}, {0.5, {4}}},
         1.0, false, 0, 5},
        {"a5_tie5_mi_interaction_neg", "a5",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}, {0.5, {3}}, {0.5, {4}}},
         1.0, false, -1, 5},
        {"a5_pairs_mi_interaction_0", "a5",
         {{0.8, {0, 1}}, {0.8, {2, 3}}, {0.8, {1, 2}}, {0.8, {3, 4}}},
         0.5, false, 0, 4},
        {"a5_pairs_mi_interaction_1", "a5",
         {{0.8, {0, 1}}, {0.8, {2, 3}}, {0.8, {1, 2}}, {0.8, {3, 4}}},
         0.5, false, 1, 4},
        {"a5_pairs_mi_interaction_2", "a5",
         {{0.8, {0, 1, 2}}, {0.8, {2, 3, 4}}, {0.8, {1, 2, 3}}},
         0.5, false, 2, 3},
        {"a5_pairs_jaccard", "a5",
         {{0.8, {0, 1}}, {0.8, {2, 3}}, {0.8, {1, 2}}, {0.8, {3, 4}}},
         0.5, true, 0, 4},
        {"a5_triples_jaccard", "a5",
         {{0.8, {0, 1, 2}}, {0.8, {2, 3, 4}}, {0.8, {1, 2, 3}}, {0.8, {0, 3, 4}}},
         0.25, true, 0, 4},
        {"a5_mixed_mi_neg", "a5",
         {{0.7, {0, 1, 2}}, {0.7, {0}}, {0.7, {1, 2}}, {0.7, {3, 4}}, {0.7, {2}}},
         0.3, false, -1, 5},
        // Same population, n_demes BELOW its size: `select_top_feature_sets`
        // must rank the WHOLE population and then take the first n, not take
        // the first n and rank those (the two differ precisely because this
        // population is reordered by ranking).
        {"a5_mixed_mi_neg_n2", "a5",
         {{0.7, {0, 1, 2}}, {0.7, {0}}, {0.7, {1, 2}}, {0.7, {3, 4}}, {0.7, {2}}},
         0.3, false, -1, 2},
        {"a5_mixed_mi_neg_n3", "a5",
         {{0.7, {0, 1, 2}}, {0.7, {0}}, {0.7, {1, 2}}, {0.7, {3, 4}}, {0.7, {2}}},
         0.3, false, -1, 3},
        // Three tied singletons, n_demes 2: the final multimap keyed on the
        // penalized score has to keep ranked order among equal keys.
        {"tie3_jaccard_p1_n2", "boolean",
         {{0.5, {0}}, {0.5, {1}}, {0.5, {2}}},
         1.0, true, 0, 2},
        // Non-dyadic jaccard indexes against non-dyadic diversity_pressure:
        // `float sim` then `float last_dp` -- with `sim` left in double the
        // product lands on a different float32. |cap|/|cup| = 1/5, 2/5, 3/5.
        {"f32_jaccard_1over5_p0.3", "a5",
         {{0.9, {0, 1, 2}}, {0.5, {2, 3, 4}}},
         0.3, true, 0, 2},
        {"f32_jaccard_2over5_p0.6", "a5",
         {{0.9, {0, 1, 2}}, {0.5, {1, 2, 3, 4}}},
         0.6, true, 0, 2},
        {"f32_jaccard_3over5_p0.7", "a5",
         {{0.9, {0, 1, 2, 3}}, {0.5, {1, 2, 3, 4}}},
         0.7, true, 0, 2},
        // fs_mi_skewed.csv is NOT a full truth-table enumeration, so mutual
        // information between its input sets is irrational rather than a
        // whole number of bits -- which is what makes `float sim =
        // mi(...)`'s narrowing observable at all.
        {"skew_mi_neg_p0.9", "skew",
         {{0.9, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.9, false, -1, 3},
        {"skew_mi_neg_p0.13", "skew",
         {{0.9, {0}}, {0.5, {1}}, {0.5, {2}}},
         0.13, false, -1, 3},
        {"skew_mi_interaction_1", "skew",
         {{0.9, {0, 1, 2}}, {0.5, {0, 2, 3}}, {0.5, {1, 3, 4}}},
         0.7, false, 1, 3},
        {"skew_mi_interaction_2", "skew",
         {{0.9, {0, 1, 2}}, {0.5, {0, 2, 3}}, {0.5, {1, 3, 4}}},
         0.7, false, 2, 3},
        {"skew_jaccard_p0.23", "skew",
         {{0.9, {0, 1, 2}}, {0.5, {2, 3, 4}}, {0.5, {0, 4}}},
         0.23, true, 0, 3},
    };
    dump_rank(boolean, a5, skew, rank_cases);
    printf(",\n");

    // table, algorithm, exemplar, base_target_size, increase_target_size,
    // diversity_pressure, diversity_jaccard, diversity_interaction, n_demes,
    // diversity_cap, subsampling_ratio, seed
    vector<e2e_case> e2e_cases = {
        {"boolean", "simple", "$1", 2, true, 0.5, true, 0, 2, 0, 1.0, 1},
        {"boolean", "simple", "$1", 2, true, 0.5, false, 0, 2, 0, 1.0, 1},
        {"boolean", "simple", "and($1 $2)", 2, true, 1.0, true, 0, 3, 0, 1.0, 1},
        {"boolean", "simple", "$1", 2, true, 0.0, true, 0, 2, 0, 0.75, 1},
        {"boolean", "simple", "$1", 2, true, 0.0, true, 0, 2, 0, 0.5, 1},
        {"boolean", "simple", "$1", 2, true, 0.5, true, 0, 2, 0, 0.75, 1},
        {"boolean", "simple", "and($1 $2)", 1, true, 0.25, true, 0, 2, 0, 0.5, 3},
        // NOT dumped: the "hc" algorithm, the ONLY one whose feature_set_pop
        // can hold more than one feature set (simple/inc/smd/random all
        // `return {sfs}`). classic's own "hc" feature-selection path
        // SEGFAULTS unconditionally -- already measured and documented in
        // tools/cpp_probe/fs_standalone_probe.cc's header comment, and
        // re-confirmed here (rc 139 on the first such case). So there is NO
        // C++-reachable end-to-end route that drives rank_feature_sets over a
        // population of more than one; the "rank" section above drives it
        // directly instead.
        {"a5", "simple", "$1", 1, false, 0.0, true, 0, 1, 0, 0.75, 1},
        {"a5", "simple", "$1", 1, false, 0.0, true, 0, 1, 0, 0.5, 1},
        {"a5", "simple", "$1", 1, false, 0.0, true, 0, 1, 0, 0.25, 7},
    };
    dump_end_to_end(boolean, a5, skew, e2e_cases);
    printf(",\n");

    // table, fs_l, fs_r, diversity_interaction
    vector<mi_case> mi_cases = {
        {"skew", {0}, {1}, -1},
        {"skew", {0}, {2}, -1},
        {"skew", {1}, {4}, -1},
        {"skew", {0, 1}, {2, 3}, -1},
        {"skew", {0, 1, 2}, {2, 3, 4}, -1},
        {"skew", {0, 1, 2}, {0, 1, 2}, 0},
        {"skew", {0, 1, 2}, {2, 3, 4}, 0},
        {"skew", {0, 1, 2}, {0, 1, 2}, 1},
        {"skew", {0, 1, 2}, {0, 2, 3}, 1},
        {"skew", {0, 1, 2}, {0, 2, 4}, 1},
        {"skew", {0, 1, 2}, {1, 2, 4}, 1},
        {"skew", {0, 1, 2}, {1, 3, 4}, 1},
        {"skew", {0, 1, 2}, {2, 3, 4}, 1},
        {"skew", {0, 1, 3}, {0, 3, 4}, 1},
        {"skew", {0, 1, 3}, {1, 3, 4}, 1},
        {"skew", {0, 1, 2, 3}, {1, 2, 3, 4}, 1},
        {"skew", {0, 1, 2, 3}, {1, 2, 3, 4}, 2},
        {"skew", {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, 2},
        {"a5", {0, 1}, {1, 2}, 0},
        {"a5", {0, 1, 2}, {2, 3, 4}, 1},
        {"boolean", {0}, {1}, -1},
        {"boolean", {0, 1}, {1, 2}, 0},
    };
    dump_mi(boolean, a5, skew, mi_cases);
    printf("\n}\n");
    return 0;
}
