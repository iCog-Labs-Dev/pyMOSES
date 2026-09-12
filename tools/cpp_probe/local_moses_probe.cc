// tools/cpp_probe/local_moses_probe.cc
//
// Oracle for pyMoses S11 Task 1 (moses/moses/moses/local_moses.{h,cc}:
// expand_deme + local_moses). Drives the REAL exported
// opencog::moses::local_moses() function directly -- not a
// reimplementation of its loop -- over a real Table/CTable/
// ctruth_table_bscore/behave_cscore/deme_expander/hill_climbing/
// metapopulation stack, the same "drive the real classic pipeline"
// convention every other S10/S11 probe in this directory follows
// (deme_expander_probe.cc, merging_probe.cc, metapop_core_probe.cc).
//
// PER-GENERATION granularity, and its one measured caveat:
// -----------------------------------------------------------------------
// `expand_deme()` is `static` (internal linkage) inside local_moses.cc, so
// it cannot be linked from an external probe -- only the exported
// `local_moses()` itself is callable here. This probe instead calls
// `local_moses()` REPEATEDLY, bumping `moses_parameters::max_gens` by
// EXACTLY one before each call, with the SAME `metapopulation`/
// `deme_expander`/`moses_statistics` objects threaded through every call
// (all three passed BY REFERENCE and mutated in place, so state persists
// across calls exactly as it would across iterations of one continuous
// while-loop -- local_moses.cc:131-134's own termination lambdas only
// ever read CURRENT `pa`/`stats` state, nothing `local_moses()` itself
// resets between calls). `pa.max_score` is set far out of reach so only
// `max_gens` ever controls how many generations one call may run.
//
// The one thing that IS reset at the top of every `local_moses()` call:
// `stats.elapsed_secs = 0.0;` (local_moses.cc:122) and the LOCAL variable
// `premature_termination` (local_moses.cc:128, always `false` on entry).
// The latter would only matter if some earlier `expand_deme()` call had
// returned `true` (metapopulation went empty, or every exemplar failed to
// expand).
//
// `advanced` in the JSON output below (`true` when `stats.n_expansions`
// increased by EXACTLY one relative to the previous generation) does NOT
// settle whether that happened: `expand_deme()` increments `stats.
// n_expansions` (local_moses.cc:80) BEFORE computing and returning `done
// || mp.empty()` (local_moses.cc:92), so a call that DOES terminate
// prematurely still advances `n_expansions` by exactly one first --
// `advanced == true` every generation is consistent with EITHER outcome
// and does not, by itself, prove `premature_termination` never fired.
// This was measured directly instead: `local_moses_single_call_probe.cc`
// runs the IDENTICAL scenario as a single `local_moses(..., max_gens=5,
// ...)` call rather than five max_gens-bumped-by-one calls, and gives
// BYTE-FOR-BYTE the same final state this probe's own `gen:5` line
// reports -- `n_expansions=5, n_evals=113, metapop_size=20`, the same
// `best_composite_score`, and the same `best_tree`. That measured
// equality, not the `advanced` argument above, is what establishes this
// repeated-call construction reproduces one continuous
// `local_moses(..., max_gens=N, ...)` call.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I "$C" -I "$P/include" \
//       -o tools/cpp_probe/build/local_moses_probe \
//       tools/cpp_probe/local_moses_probe.cc \
//       -L "$P/lib" -L "$P/lib/opencog" \
//       -Wl,-rpath,"$P/lib" -Wl,-rpath,"$P/lib/opencog" \
//       -lmoses -lcomboreduct -lcogutil -lboost_program_options
//
// `-lboost_program_options` is REQUIRED, not optional: `libmoses.so` pulls
// in `problem-params.cc`'s `boost::program_options::variables_map`
// machinery transitively, and this is the one probe in this directory
// that actually reaches something depending on it at link time -- without
// it, `ld` fails with "undefined reference to symbol
// '_ZN5boost15program_options13variables_mapC1Ev'" /
// "DSO missing from command line" (measured directly: the recipe as
// previously written did not link at all).
//
// Run, bounded (CARRY-FORWARD.md's standing OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 tools/cpp_probe/build/local_moses_probe )
//
// (no arguments -- fixed scenario, run from the repo root so the relative
// dataset path resolves: trainingBoolean.dat, arity 3, exemplar "and()",
// seed 42, 5 generations)

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/moses/deme/deme_expander.h>
#include <moses/moses/metapopulation/metapopulation.h>
#include <moses/moses/moses/local_moses.h>
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
// widened float32 produces Python-side).
static void print_score(const composite_score &cs)
{
    printf("{\"score\":%.17g,\"complexity\":%u,\"complexity_penalty\":%.17g,"
           "\"uniformity_penalty\":%.17g,\"penalized_score\":%.17g}",
           (double)cs.get_score(), (unsigned)cs.get_complexity(),
           (double)cs.get_complexity_penalty(), (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    Table table = loadTable("tests/fixtures/table_data/trainingBoolean.dat", "y");
    CTable ctable = table.compressed();

    ctruth_table_bscore bsc(ctable);
    bsc.set_complexity_coef((score_t)3.5);
    behave_cscore cscorer(bsc);

    // NOTE, measured (see merging_probe.cc's own identical comment):
    // `metapopulation::init()` (metapopulation.cc:59) scores the raw seed
    // exemplar DIRECTLY, with no build_knobs/reduction pass first -- unlike
    // a deme's own exemplar, which `create_demes` always canonizes before
    // anything is scored. A bare "and()" here crashes: CARRY-FORWARD.md's
    // "empty and()/or() return the operator token" quirk feeds a non-bool
    // token into `complete_truth_table`'s `builtin_to_bool`, which
    // OC_ASSERTs and terminates the process uncaught. "$1" (a bare argument
    // read) has no such problem and is what this probe's metapopulation is
    // seeded with; deme exemplars (chosen later by `select_exemplar()`
    // and passed through `create_demes`) are unaffected either way.
    combo_tree exemplar = str2combo_tree("$1", vector<string>());

    metapop_parameters mpp;
    subsample_deme_filter_parameters subp;
    metapopulation mp(exemplar, cscorer, mpp, subp);

    type_tree sig = gen_signature(id::boolean_type, 3);

    operator_set empty_ignore_ops;
    logical_reduction reduction(empty_ignore_ops);
    const rule &r2 = reduction(2);

    deme_parameters dparams;
    dparams.perm_ratio = -1.0f;
    subsample_deme_filter_parameters fparams;

    hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/false);
    hill_climbing hc((optim_parameters()), hcp);

    deme_expander dex(sig, r2, r2, cscorer, hc, dparams, fparams);

    unsigned seed = 42;
    randGen().seed(seed);

    moses_parameters pa;
    pa.max_evals = 20000;
    pa.max_score = (score_t)1000.0;  // unreachable -- max_gens drives this probe
    moses_statistics stats;

    const int N_GENS = 5;

    printf("{\"seed\":%u,\"generations\":[\n", seed);
    int prev_expansions = 0;
    for (int g = 1; g <= N_GENS; ++g) {
        pa.max_gens = g;
        local_moses(mp, dex, pa, stats);

        int n_expansions = (int)stats.n_expansions;
        printf("%s{\"gen\":%d,\"n_expansions\":%d,\"n_evals\":%d,\"advanced\":%s,"
               "\"metapop_size\":%zu,\"best_composite_score\":",
               g > 1 ? "," : "", g, n_expansions, (int)stats.n_evals,
               (n_expansions == prev_expansions + 1) ? "true" : "false", mp.size());
        print_score(mp.best_composite_score());
        printf(",\"best_tree\":%s}\n", json_string(rendered(mp.best_tree())).c_str());
        prev_expansions = n_expansions;
    }
    printf("]}\n");
    return 0;
}
