// tools/cpp_probe/deme_expander_probe.cc
//
// Oracle for pyMeTTa S10 Task 4 (moses/moses/deme/deme_expander.{h,cc} and
// deme/deme_params.h, single-deme path).
//
// Drives the REAL opencog::moses::deme_expander end to end: loads a small
// real dataset through the real Table/CTable/ctruth_table_bscore/
// behave_cscore stack (same recipe as tools/cpp_probe/score_probe.cc), builds
// a deme_expander over it with a real hill_climbing optimizer (hc_parameters
// cross=false, same LANDMINE workaround shape as tools/cpp_probe/
// hillclimbing_probe.cc is not needed here -- complexity_based_scorer's own
// get_rep() returns the real representation deme_expander itself builds, so
// there is no null-get_rep() hazard to work around), and dumps, as JSON:
//   - the built representation's field_set shape (disc multiplicities,
//     n_bits) and its knob count/order/text -- same shape as
//     representation_probe.cc's own per-case "knobs" dump.
//   - the decorated exemplar.
//   - after optimize_demes(): total_evals(), the deme size, and the full
//     trajectory of every scored_instance ever appended to the ONE deme
//     (single representation, single ss-deme, since n_subsample_demes==0
//     and fstor==nullptr -- deme_expander.py's own scope) -- same shape as
//     hillclimbing_probe.cc's print_deme_trajectory.
//
// This is what a Python port (pymetta.moses.deme.deme_expander, driven by
// pymetta's own ctruth_table_bscore/behave_cscore/hill_climbing) must
// reproduce field-for-field, knob-for-knob, instance-for-instance, for a
// FIXED randGen() seed.
//
// Build:
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/deme_expander_probe \
//       tools/cpp_probe/deme_expander_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and this session's own standing OOM caution. The dataset
// (trainingBooleanXor.dat, 2 boolean inputs) and exemplar ("and()", arity 2)
// are both tiny by construction -- deme sizes stay in the tens, not
// thousands.
//
// Usage: deme_expander_probe <csv-path> <target-feature> <exemplar-str>
//                             <arity> <seed> <max-evals>

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/moses/deme/deme_expander.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <moses/moses/scoring/bscores.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
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

static void print_field_shape(const field_set &fs)
{
    printf("{\"disc_multiplicities\":[");
    for (size_t i = fs.begin_disc_raw_idx(); i < fs.end_disc_raw_idx(); ++i) {
        if (i > fs.begin_disc_raw_idx()) printf(",");
        printf("%u", (unsigned)fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy);
    }
    printf("],\"n_bits\":%zu,\"packed_width\":%zu}", fs.n_bits(), fs.packed_width());
}

static void print_knobs(const representation &rep)
{
    printf("[");
    size_t order = 0;
    for (auto it = rep.disc.cbegin(); it != rep.disc.cend(); ++it, ++order) {
        if (order) printf(",");
        size_t preorder = distance(rep.exemplar().begin(), it->second->get_loc());
        printf("{\"order\":%zu,\"multy\":%u,\"text\":%s,\"location_preorder\":%zu}", order,
               (unsigned)it->first.multy, json_string(it->second->toStr()).c_str(), preorder);
    }
    printf("]");
}

static void print_raw(const field_set &fs, const instance &inst)
{
    printf("[");
    for (size_t i = 0; i < fs.raw_size(); ++i) {
        if (i) printf(",");
        printf("%lu", (unsigned long)fs.get_raw(inst, i));
    }
    printf("]");
}

// %.17g (full double round-trip precision), NOT %.9g -- matching
// score_probe.cc's own convention. `score_t` (float32) values are widened
// to `double` for printing; %.17g round-trips a double exactly, so the
// JSON fixture reproduces the SAME double bit pattern
// `_to_score_t(...)`'s widened float32 gives in the Python port, letting
// test_deme_expander.py compare with exact `==` rather than a tolerance.
// (measured directly: %.9g of a fractional complexity_penalty like
// 0.8571428656578064 prints "0.857142866", which parses back to a
// DIFFERENT double than the original -- %.17g does not have this problem.)
static void print_score(const composite_score &cs)
{
    printf("{\"score\":%.17g,\"complexity\":%u,\"complexity_penalty\":%.17g,"
           "\"uniformity_penalty\":%.17g,\"penalized_score\":%.17g}",
           (double)cs.get_score(), (unsigned)cs.get_complexity(),
           (double)cs.get_complexity_penalty(), (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

static void print_trajectory(const field_set &fs, const deme_t &deme)
{
    printf("[");
    for (size_t i = 0; i < deme.size(); ++i) {
        if (i) printf(",");
        printf("{\"raw\":");
        print_raw(fs, deme[i].first);
        printf(",\"score\":");
        print_score(deme[i].second);
        printf("}");
    }
    printf("]");
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <csv-path> <target-feature> <exemplar-str> <arity> "
                "<seed> <max-evals>\n",
                argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    string path = argv[1];
    string target_feature = argv[2];
    string exemplar_str = argv[3];
    int arity = atoi(argv[4]);
    unsigned seed = (unsigned)atoi(argv[5]);
    int max_evals = atoi(argv[6]);

    Table table = loadTable(path, target_feature);
    CTable ctable = table.compressed();

    ctruth_table_bscore bsc(ctable);
    bsc.set_complexity_coef((score_t)3.5);
    behave_cscore cscorer(bsc);

    type_tree sig = gen_signature(id::boolean_type, arity);

    logical_reduction reduction(empty_ignore_ops);
    const rule &r2 = reduction(2);

    deme_parameters dparams;
    dparams.perm_ratio = -1.0f;
    subsample_deme_filter_parameters fparams;

    hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/false);
    hill_climbing hc((optim_parameters()), hcp);

    deme_expander dex(sig, r2, r2, cscorer, hc, dparams, fparams);

    combo_tree exemplar = str2combo_tree(exemplar_str, std::vector<std::string>());

    randGen().seed(seed);
    bool created = dex.create_demes(exemplar, 0);

    printf("{\n");
    printf("\"created\":%s,\n", created ? "true" : "false");
    printf("\"n_reps\":%zu,\n", dex._reps.size());
    if (!dex._reps.empty()) {
        const representation &rep = dex._reps[0];
        printf("\"decorated_exemplar\":%s,\n", json_string(rendered(rep.exemplar())).c_str());
        printf("\"field_shape\":");
        print_field_shape(rep.fields());
        printf(",\n\"knobs\":");
        print_knobs(rep);
        printf(",\n");
    }

    dex.optimize_demes(max_evals, 86400);

    printf("\"total_evals\":%u,\n", dex.total_evals());
    if (!dex._demes.empty()) {
        const field_set &fs = dex._reps[0].fields();
        const deme_t &deme = dex._demes[0][0];
        printf("\"deme_size\":%zu,\n", deme.size());
        printf("\"n_evals\":%u,\n", deme.n_evals);
        printf("\"trajectory\":");
        print_trajectory(fs, deme);
        printf("\n");
    }
    printf("}\n");
    return 0;
}
