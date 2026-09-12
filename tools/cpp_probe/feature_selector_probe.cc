// tools/cpp_probe/feature_selector_probe.cc
//
// Oracle for the P2 feature-selection port (moses/moses/deme/
// feature_selector.{h,cc} -- the "--enable-fs" in-loop path -- plus the
// "simple" algorithm + "mi" scorer it calls into via
// moses/feature-selection/main/feature-selection.{h,cc} and
// moses/feature-selection/algo/simple.{h,cc}).
//
// Three independent things are dumped, as JSON, all from the REAL classic
// code (not reimplemented here):
//
//   "mi"      -- mutualInformation(ctable, {single feature}) for every
//                input column of trainingBoolean.dat and
//                trainingBooleanXor.dat. Ground truth for
//                pymoses.feature_selection.scorers.mutual_info.
//
//   "simple"  -- simple_select_feature_sets(ctable, fs_params) for a range
//                of target_size values on both tables, with the DEFAULT
//                fs_params (exp_distrib=false, threshold=0). This exercises
//                simple.cc's RNG-seeded tie-break
//                (ScoredFeatureSetGreater's one randGen().randint() draw)
//                whenever two features tie on MI -- both fixture tables do
//                this at target_size=2/1 respectively. Ground truth for
//                pymoses.feature_selection.algo.simple.
//
//   "in_loop" -- feature_selector::operator()(exemplar), the actual
//                --enable-fs entry point, for a handful of
//                (exemplar, restrict_incorrect, ignore_xmplr_features,
//                increase_target_size, base target_size) cases on
//                trainingBoolean.dat. A `feature_selector_probe_access`
//                subclass exposes the protected preprocess_params/
//                build_fs_ctable so the fixture also records the POST-
//                preprocessing target_size, the ignored-feature set, and
//                the filtered ctable's uncompressed_size() -- everything
//                pymoses.moses.deme.feature_selector's own
//                preprocess_params/build_fs_ctable must reproduce, not just
//                the final selected set. Ground truth for
//                pymoses.moses.deme.feature_selector.
//
// randGen() is (re)seeded to a fixed value immediately before every case
// that can draw from it (every "simple" and "in_loop" case), matching a
// fresh process's own randGen() state under `-r <seed>` -- see fs-main.cc:
// `randGen().seed(rand_seed)` runs once, before any feature-selection call,
// and (for algorithm="simple", the only one this probe/port touches) no
// other draw happens first.
//
// Build:
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/feature_selector_probe \
//       tools/cpp_probe/feature_selector_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lfeature_selection -lmoses -lcomboreduct -lcogutil \
//       -lboost_program_options
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).
//
// Usage: feature_selector_probe <boolean-csv> <xor-csv> <seed>

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/type_checker/type_tree.h>
#include <moses/feature-selection/algo/simple.h>
#include <moses/feature-selection/main/feature-selection.h>
#include <moses/moses/deme/feature_selector.h>
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

static string json_string(const string &value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

static string rendered(const combo_tree &tr)
{
    stringstream out;
    out << tr;
    return out.str();
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

// Exposes the protected members needed to inspect preprocess_params/
// build_fs_ctable's OWN effect, separately from the final selected set --
// see this file's header comment. Never calls operator() itself, so it
// never double-applies preprocess_params's mutations (target_size += ...
// is not idempotent).
struct feature_selector_probe_access : public feature_selector
{
    using feature_selector::feature_selector;
    using feature_selector::preprocess_params;
    using feature_selector::build_fs_ctable;
};

static void dump_mi(const char *name, const CTable &ctable)
{
    printf("\"%s\":{", name);
    unsigned arity = ctable.get_arity();
    auto ilabels = ctable.get_input_labels();
    for (unsigned i = 0; i < arity; i++) {
        if (i) printf(",");
        feature_set fs{(arity_t)i};
        double mi = mutualInformation(ctable, fs);
        printf("%s:%.17g", json_string(ilabels[i]).c_str(), mi);
    }
    printf("}");
}

static void dump_simple(const char *name, const CTable &ctable, unsigned seed,
                        const vector<unsigned> &target_sizes)
{
    auto ilabels = ctable.get_input_labels();
    printf("\"%s\":{", name);
    bool first_ts = true;
    for (unsigned ts : target_sizes) {
        if (!first_ts) printf(",");
        first_ts = false;

        feature_selection_parameters fs_params;
        fs_params.target_size = ts;

        randGen().seed(seed);
        feature_set_pop pop = simple_select_feature_sets(ctable, fs_params);

        printf("\"%u\":[", ts);
        // pop always has exactly one entry for the simple algorithm
        // (simple.cc: `return {sfs};`).
        const feature_set &fs = pop.begin()->second;
        print_feature_set(fs);
        printf("]");
    }
    printf("}");
}

static void dump_in_loop_case(const CTable &ctable, const combo_tree &exemplar,
                              unsigned base_target_size, bool restrict_incorrect,
                              bool restrict_true, bool ignore_xmplr_features,
                              bool increase_target_size, unsigned seed)
{
    feature_selector_parameters params;
    params.fs_params.target_size = base_target_size;
    params.restrict_incorrect = restrict_incorrect;
    params.restrict_true = restrict_true;
    params.ignore_xmplr_features = ignore_xmplr_features;
    params.increase_target_size = increase_target_size;

    // Pass 1: introspect preprocess_params/build_fs_ctable in isolation.
    feature_selector_probe_access probe_a(ctable, params);
    probe_a.preprocess_params(exemplar);
    CTable fs_ctable = probe_a.build_fs_ctable(exemplar);

    // Pass 2: a FRESH feature_selector for the real operator() call, so
    // preprocess_params's target_size += mutation is applied exactly once
    // (matching what a real deme_expander::create_representations call
    // site does: one festor(exemplar) call on one festor copy).
    feature_selector festor_b(ctable, params);
    randGen().seed(seed);
    feature_set_pop top = festor_b(exemplar);

    printf("{");
    printf("\"exemplar\":%s,", json_string(rendered(exemplar)).c_str());
    printf("\"base_target_size\":%u,", base_target_size);
    printf("\"restrict_incorrect\":%s,", restrict_incorrect ? "true" : "false");
    printf("\"restrict_true\":%s,", restrict_true ? "true" : "false");
    printf("\"ignore_xmplr_features\":%s,", ignore_xmplr_features ? "true" : "false");
    printf("\"increase_target_size\":%s,", increase_target_size ? "true" : "false");
    printf("\"effective_target_size\":%u,", probe_a.params.fs_params.target_size);
    printf("\"ignored_features\":");
    print_feature_set(probe_a.params.ignored_features);
    printf(",\"fs_ctable_uncompressed_size\":%.17g,", fs_ctable.uncompressed_size());
    printf("\"fs_ctable_compressed_size\":%zu,", fs_ctable.size());
    printf("\"selected\":");
    print_feature_set(top.begin()->second);
    printf(",\"n_feature_sets\":%zu", top.size());
    printf("}");
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <boolean-csv> <xor-csv> <seed>\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    string bool_path = argv[1];
    string xor_path = argv[2];
    unsigned seed = (unsigned)atoi(argv[3]);

    Table bool_table = loadTable(bool_path, "y");
    CTable bool_ctable = bool_table.compressed();

    Table xor_table = loadTable(xor_path, "y");
    CTable xor_ctable = xor_table.compressed();

    printf("{\n");

    printf("\"mi\":{");
    dump_mi("boolean", bool_ctable);
    printf(",");
    dump_mi("xor", xor_ctable);
    printf("},\n");

    printf("\"simple\":{");
    dump_simple("boolean", bool_ctable, seed, {0, 1, 2, 3});
    printf(",");
    dump_simple("xor", xor_ctable, seed, {0, 1, 2});
    printf("},\n");

    printf("\"in_loop\":[\n");
    {
        combo_tree ex_and = str2combo_tree("and($1 $2)", std::vector<std::string>());
        combo_tree ex_arg1 = str2combo_tree("$1", std::vector<std::string>());

        dump_in_loop_case(bool_ctable, ex_and, /*base_target_size=*/1,
                          /*restrict_incorrect=*/true, /*restrict_true=*/false,
                          /*ignore_xmplr_features=*/true,
                          /*increase_target_size=*/true, seed);
        printf(",\n");
        dump_in_loop_case(bool_ctable, ex_and, /*base_target_size=*/1,
                          /*restrict_incorrect=*/false, /*restrict_true=*/false,
                          /*ignore_xmplr_features=*/true,
                          /*increase_target_size=*/true, seed);
        printf(",\n");
        dump_in_loop_case(bool_ctable, ex_arg1, /*base_target_size=*/1,
                          /*restrict_incorrect=*/true, /*restrict_true=*/false,
                          /*ignore_xmplr_features=*/false,
                          /*increase_target_size=*/false, seed);
        printf(",\n");
        dump_in_loop_case(bool_ctable, ex_arg1, /*base_target_size=*/0,
                          /*restrict_incorrect=*/true, /*restrict_true=*/false,
                          /*ignore_xmplr_features=*/true,
                          /*increase_target_size=*/true, seed);
    }
    printf("\n]\n");

    printf("}\n");
    return 0;
}
