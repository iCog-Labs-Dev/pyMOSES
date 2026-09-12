// tools/cpp_probe/mi_empty_probe.cc
//
// Oracle for what `mutualInformation` / `mutualInformationBtwSets`
// (moses/comboreduct/table/table.h:1132 / :1311) and `OTEntropy`
// (table.cc:916) return when the table they are given has NOTHING TO
// DIVIDE BY -- the case `pymoses/feature_selection/scorers/mutual_info.py`
// used to short-circuit with `if total == 0.0: return NaN`.
//
// That short-circuit was a guess, and it was wrong in the direction that
// costs a whole MOSES run. C++ never tests `total`: it sizes each
// probability vector from its counter (`std::vector<double> ip(ic.size())`)
// and then `transform`s with `div_total`. When the counter is EMPTY the
// vector is empty, `transform` is a no-op, NO DIVISION HAPPENS AT ALL, and
// `entropy()` -- `double res = 0; for(...) res += ...` (numeric.h:229) --
// returns exactly 0.0. So MI of an empty table is 0.0, not NaN.
//
// Why that matters, concretely: `--enable-fs=1 --fs-subsampling-ratio=0`
// makes `subsampleCTable` (table.cc:899) remove `(1.0 - 0.0) *
// uncompressed_size` rows, i.e. ALL of them, and `feature_selector::
// build_fs_ctable` (feature_selector.cc:205) then hands that empty ctable
// to `simple_selection`, whose filter is `threshold <= sc`
// (simple.h:121). With sc == 0.0 and the default threshold 0, EVERY
// feature passes and classic selects `--fs-target-size` of them; with a
// NaN, `0 <= NaN` is false, no feature passes, and the port built an
// empty representation and lost the search.
//
// The three groups below:
//
//   "empty"       -- a real CSV loaded and compressed, then emptied with
//                    `CTable::remove_rows` over every uncompressed index
//                    (exactly how `subsampleCTable(0.0, ct)` empties it,
//                    minus the RNG). Signature, labels and arity survive;
//                    only the rows are gone. MI is dumped for every
//                    singleton and for `mutualInformationBtwSets`.
//
//   "zero_counts" -- the SAME table with every `count_t` set to 0.0 but
//                    the rows left in place: counters non-empty, `total`
//                    still 0.0, so `div_total` DOES run and produces 0/0.
//                    This is the branch a plain `c / total` in Python
//                    cannot reproduce (it raises ZeroDivisionError) and
//                    that `weighted_information`'s `p > PROB_EPSILON`
//                    (numeric.h:207) swallows, NaN not being > anything.
//
//   "baseline"    -- the untouched table, so the fixture is not composed
//                    entirely of degenerate cases and a port that returned
//                    0.0 unconditionally would still fail.
//
// Also dumps `OTEntropy` (table.cc:916) on an empty OTable, the same
// empty-range-no-division shape one layer down, and the ITable/OTable
// `mutualInformation` overload (table.h:1074) on an empty pair.
//
// Build/run: tools/capture_mi_empty.py (bounded per this project's
// standing OOM caution: `( ulimit -v 2097152; timeout 300 <cmd> )`).

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/type_checker/type_tree.h>

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

typedef set<arity_t> probe_feature_set;

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

// Every singleton, every adjacent pair, and the full set.
static vector<probe_feature_set> feature_sets(arity_t arity)
{
    vector<probe_feature_set> sets;
    for (arity_t i = 0; i < arity; i++)
        sets.push_back(probe_feature_set{i});
    for (arity_t i = 0; i + 1 < arity; i++)
        sets.push_back(probe_feature_set{i, (arity_t)(i + 1)});
    probe_feature_set all;
    for (arity_t i = 0; i < arity; i++) all.insert(i);
    sets.push_back(all);
    return sets;
}

static void dump_group(const char *name, const CTable &ctable)
{
    arity_t arity = ctable.get_arity();
    printf("\"%s\":{\"size\":%u,\"uncompressed_size\":%.17g,\"arity\":%d,\"mi\":[",
           name, (unsigned)ctable.size(), (double)ctable.uncompressed_size(),
           (int)arity);
    vector<probe_feature_set> sets = feature_sets(arity);
    for (size_t i = 0; i < sets.size(); i++) {
        if (i) printf(",");
        printf("{\"fs\":");
        print_feature_set(sets[i]);
        printf(",\"mi\":%.17g}", mutualInformation(ctable, sets[i]));
    }
    printf("],\"btw_sets\":[");
    bool first = true;
    for (arity_t l = 0; l < arity; l++) {
        for (arity_t r = 0; r < arity; r++) {
            if (!first) printf(",");
            first = false;
            probe_feature_set fs_l{l}, fs_r{r};
            printf("{\"fs_l\":");
            print_feature_set(fs_l);
            printf(",\"fs_r\":");
            print_feature_set(fs_r);
            printf(",\"mi\":%.17g}",
                   mutualInformationBtwSets(ctable, fs_l, fs_r));
        }
    }
    printf("]}");
}

// One CSV -> the three groups above, under a "<name>" key.
static void dump_table(const char *name, const char *path)
{
    Table table = loadTable(path, "y");
    CTable base = table.compressed();

    // Emptied exactly as `subsampleCTable(0.0f, ct)` empties it:
    // `nremove == (1.0 - 0.0) * uncompressed_size`, i.e. every index.
    CTable emptied = base;
    set<unsigned> all_idxs;
    for (unsigned i = 0; i < (unsigned)emptied.uncompressed_size(); i++)
        all_idxs.insert(i);
    emptied.remove_rows(all_idxs);

    // Rows kept, counts zeroed: `total` is 0.0 with NON-empty counters, so
    // `div_total` actually divides by zero.
    CTable zeroed = base;
    for (auto &row : zeroed)
        for (auto &val_pair : row.second)
            val_pair.second = 0.0;

    printf("\"%s\":{\n", name);
    dump_group("baseline", base);
    printf(",\n");
    dump_group("empty", emptied);
    printf(",\n");
    dump_group("zero_counts", zeroed);

    // One layer down: OTEntropy over an empty OTable, and the
    // ITable/OTable mutualInformation overload over an empty pair.
    // `table.otable` keeps its type; only the rows are dropped.
    Table it_ot = table;
    it_ot.itable.clear();
    it_ot.otable.clear();
    printf(",\n\"ot_entropy_empty\":%.17g", OTEntropy(it_ot.otable));
    printf(",\n\"mi_it_ot_empty\":[");
    for (arity_t i = 0; i < (arity_t)table.get_arity(); i++) {
        if (i) printf(",");
        probe_feature_set fs{i};
        printf("{\"fs\":");
        print_feature_set(fs);
        printf(",\"mi\":%.17g}",
               mutualInformation(it_ot.itable, it_ot.otable, fs));
    }
    printf("]");
    printf("\n}");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <csv> [<csv>...]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    printf("{\n");
    for (int i = 1; i < argc; i++) {
        if (i > 1) printf(",\n");
        // The fixture key is the CSV's basename without its extension.
        string path(argv[i]);
        size_t slash = path.find_last_of('/');
        string name = (slash == string::npos) ? path : path.substr(slash + 1);
        size_t dot = name.find_last_of('.');
        if (dot != string::npos) name = name.substr(0, dot);
        dump_table(name.c_str(), argv[i]);
    }
    printf("\n}\n");
    return 0;
}
