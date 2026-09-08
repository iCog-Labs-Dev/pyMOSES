// tools/cpp_probe/merging_probe.cc
//
// Oracle probe for pyMeTTa S10 Task 3 (moses/moses/metapopulation/merging.cc:
// merge_candidates, merge_demes, and every private helper merge_demes calls
// that lives in merging.cc itself -- trim_down_deme, deme_to_trees,
// get_new_candidates, sort_demes, keep_top_unique_candidates, resize_metapop).
//
// `metapopulationUTest.cxxtest` covers none of this (see CARRY-FORWARD.md:
// its two tests exercise domination.cc, outside this whole S10 slice). This
// probe drives the REAL classic pipeline end to end, reusing exactly the
// "xor_empty_and" scenario tools/capture_deme_expander.py already captured
// for S10 Task 4's own oracle (tests/fixtures/deme_expander.json) -- same
// dataset, exemplar, arity, seed, max_evals -- so `create_demes`+
// `optimize_demes` reproduce an ALREADY oracle-verified deme+representation
// (deme_size 17), and this probe only needs to add what Task 4's own probe
// does not: constructing a real `metapopulation` and calling `merge_demes`
// on that deme, then dumping the resulting pool.
//
// Part 1 ("single_merge"): a fresh metapopulation seeded with the exemplar
// "and()" (the same tree create_demes/optimize_demes decorate and search
// around), merged once against the one deme optimize_demes produced. Dumps
// pre-merge size, merge_demes's own return value, post-merge size,
// _scored_trees in full sct_score_greater order (tree/score/complexity/
// penalized_score for each), best_composite_score, and best_tree.
//
// Part 2 ("repeat_merge"): merge_demes called a SECOND time against a FRESH
// re-run of create_demes/optimize_demes with the IDENTICAL seed (so an
// identical deme, independent object) on the Part-1 metapopulation, which
// already contains every tree the deme could produce -- exercises
// get_new_candidates' dedup returning nothing new, so merge_demes must
// return false and leave the pool size unchanged.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I "$C" -I "$P/include" \
//       -o tools/cpp_probe/build/merging_probe \
//       tools/cpp_probe/merging_probe.cc \
//       -L "$P/lib" -L "$P/lib/opencog" \
//       -Wl,-rpath,"$P/lib" -Wl,-rpath,"$P/lib/opencog" \
//       -lmoses -lcomboreduct -lcogutil
//
// Run, bounded (CARRY-FORWARD.md's standing OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 tools/cpp_probe/build/merging_probe )
//
// (no arguments -- the scenario is fixed, matching tests/fixtures/
// deme_expander.json's own "xor_empty_and" case exactly, run from the repo
// root so the relative dataset path resolves)

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/moses/deme/deme_expander.h>
#include <moses/moses/metapopulation/metapopulation.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <moses/moses/scoring/bscores.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

static string rendered(const combo_tree &tr)
{
    stringstream out;
    out << tr;
    return out.str();
}

static string json_string(const string &value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

// %.17g, not %.9g -- see deme_expander_probe.cc's own print_score comment
// for the measured reason (a fractional complexity_penalty does not
// round-trip through %.9g to the same double bit pattern _to_score_t's
// widened float32 gives Python-side).
static void print_score(const composite_score &cs)
{
    printf("{\"score\":%.17g,\"complexity\":%u,\"complexity_penalty\":%.17g,"
           "\"uniformity_penalty\":%.17g,\"penalized_score\":%.17g}",
           (double)cs.get_score(), (unsigned)cs.get_complexity(),
           (double)cs.get_complexity_penalty(), (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

static void print_pool(const metapopulation &mp)
{
    printf("[");
    bool first = true;
    for (const scored_combo_tree &sct : mp.get_trees()) {
        if (!first) printf(",");
        first = false;
        printf("{\"tree\":%s,\"cscore\":", json_string(rendered(sct.get_tree())).c_str());
        print_score(sct.get_composite_score());
        printf("}");
    }
    printf("]");
}

// `deme_expander` holds a `boost::ptr_vector<representation>`, and
// `representation` is `boost::noncopyable` (representation.h:47) -- so
// `deme_expander` cannot be returned by value, passed by value, or stored
// in most containers. Built via a `unique_ptr` factory instead, one per
// scenario; each caller keeps its own `hill_climbing` alive alongside it
// (`deme_expander` stores `optimizer_base& _optimize` as a REFERENCE,
// deme_expander.h:44,83 -- it does not own the optimizer). `r2`
// (the simplify rule) is likewise a REFERENCE parameter, not built
// locally: `deme_expander` also stores `si_ca`/`si_kb` as references, and
// `representation` forwards them into its own reference members -- an
// earlier version of this probe built `logical_reduction`/`r2` INSIDE
// this function and crashed (measured: SIGSEGV inside
// `reduct::rule::operator()`, called from `representation::
// clean_combo_tree`, called from `keep_top_unique_candidates`) the moment
// anything tried to reduce a candidate after this function returned and
// destroyed its own local `logical_reduction` -- the caller must keep the
// reduction rule alive at least as long as every representation built
// from it.
static unique_ptr<deme_expander> build_and_run(behave_cscore &cscorer, const rule &r2,
                                                hill_climbing &hc, unsigned seed)
{
    type_tree sig = gen_signature(id::boolean_type, 2);

    deme_parameters dparams;
    dparams.perm_ratio = -1.0f;
    subsample_deme_filter_parameters fparams;

    auto dex = make_unique<deme_expander>(sig, r2, r2, cscorer, hc, dparams, fparams);

    combo_tree exemplar = str2combo_tree("and()", vector<string>());

    randGen().seed(seed);
    dex->create_demes(exemplar, 0);
    dex->optimize_demes(200, 86400);

    return dex;
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    Table table = loadTable("tests/fixtures/table_data/trainingBooleanXor.dat", "y");
    CTable ctable = table.compressed();

    ctruth_table_bscore bsc(ctable);
    bsc.set_complexity_coef((score_t)3.5);
    behave_cscore cscorer(bsc);

    // NOTE, measured: the metapopulation's OWN seed exemplar must be
    // something `_cscorer.get_cscore` can score DIRECTLY (metapopulation::
    // init(), metapopulation.cc:59, scores the raw exemplar with no
    // build_knobs/reduction pass first -- unlike deme_to_trees, which
    // always goes through rep.get_candidate's clean_combo_tree). A bare
    // "and()" crashes here: CARRY-FORWARD.md's "empty and()/or() return
    // the operator token" quirk (interpreter.cc:66-82) feeds a non-bool
    // token straight into complete_truth_table's builtin_to_bool, which
    // OC_ASSERTs and aborts the process uncaught. "and()" is fine as the
    // DEME's own exemplar in build_and_run below because create_demes
    // canonizes it (build_knobs) before anything is ever scored. "$1" (a
    // bare argument read) has no such problem.
    combo_tree exemplar = str2combo_tree("$1", vector<string>());

    // IMPORTANT, measured (see tools/cpp_probe/metapop_core_probe.cc's own
    // comment): metapopulation::_params/_filter_params are C++ REFERENCE
    // members bound directly to whatever pa/subp argument is passed --
    // omitting them binds to a destroyed temporary. Always pass explicit,
    // named, outliving parameter objects.
    metapop_parameters pa;
    subsample_deme_filter_parameters subp;

    // Kept alive in main()'s own scope for the whole run -- see
    // build_and_run's own comment for why a LOCAL logical_reduction
    // inside that function dangles the moment it returns.
    operator_set empty_ignore_ops;
    logical_reduction reduction(empty_ignore_ops);
    const rule &r2 = reduction(2);

    printf("{\n");

    // ----------------------------------------------------------------
    // Part 1: single merge_demes() call.
    metapopulation mp(exemplar, cscorer, pa, subp);
    size_t pre_size = mp.size();

    hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/false);
    hill_climbing hc1((optim_parameters()), hcp);
    unique_ptr<deme_expander> dex1 = build_and_run(cscorer, r2, hc1, /*seed=*/1);
    bool done1 = mp.merge_demes(dex1->_demes, dex1->_reps);

    printf("  \"single_merge\": {\n");
    printf("    \"pre_size\":%zu,\n", pre_size);
    printf("    \"deme_size\":%zu,\n", dex1->_demes.empty() ? 0 : dex1->_demes[0][0].size());
    printf("    \"merge_demes_returned\":%s,\n", done1 ? "true" : "false");
    printf("    \"post_size\":%zu,\n", mp.size());
    printf("    \"pool\":");
    print_pool(mp);
    printf(",\n");
    printf("    \"best_composite_score\":");
    print_score(mp.best_composite_score());
    printf(",\n");
    printf("    \"best_tree\":%s\n", json_string(rendered(mp.best_tree())).c_str());
    printf("  },\n");

    // ----------------------------------------------------------------
    // Part 2: a SECOND, independently-built deme from the identical seed
    // (so identical content, different object) merged into the SAME
    // metapopulation, which by now already contains every tree that deme
    // could produce -- get_new_candidates must find nothing new.
    hill_climbing hc2((optim_parameters()), hcp);
    unique_ptr<deme_expander> dex2 = build_and_run(cscorer, r2, hc2, /*seed=*/1);
    size_t pre_size_2 = mp.size();
    bool done2 = mp.merge_demes(dex2->_demes, dex2->_reps);

    printf("  \"repeat_merge\": {\n");
    printf("    \"pre_size\":%zu,\n", pre_size_2);
    printf("    \"merge_demes_returned\":%s,\n", done2 ? "true" : "false");
    printf("    \"post_size\":%zu\n", mp.size());
    printf("  }\n");

    printf("}\n");
    return 0;
}
