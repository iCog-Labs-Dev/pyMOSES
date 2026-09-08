// tools/cpp_probe/local_moses_single_call_probe.cc
//
// Companion measurement for local_moses_probe.cc, NOT a checked-in oracle
// of its own (no expected values live in any test file for this probe's
// output -- see that file's own header comment for why it exists).
//
// local_moses_probe.cc calls `local_moses()` REPEATEDLY (bumping
// `max_gens` by one before each call) because `expand_deme()` is `static`
// and cannot be linked from outside local_moses.cc, so per-generation
// snapshots need per-generation calls. Its header comment used to argue
// that repeated-call construction is equivalent to one continuous
// `local_moses(..., max_gens=5, ...)` call because `advanced` (`stats.
// n_expansions` grew by exactly one) was `true` for every one of its five
// generations, therefore `premature_termination` never fired. THAT
// ARGUMENT DOES NOT FOLLOW: `expand_deme()` increments `stats.
// n_expansions` (local_moses.cc:80, `stats.n_expansions++`) BEFORE
// computing and returning `done || mp.empty()` (local_moses.cc:92) --
// so a call that DOES terminate prematurely (merge collapses the deme,
// or the metapopulation goes empty) still increments `n_expansions` by
// exactly one first. `advanced == true` is therefore consistent with
// EITHER outcome and proves nothing about `premature_termination` on its
// own.
//
// This probe settles the question by measurement instead of argument: it
// builds the IDENTICAL scenario (same table, same seed exemplar, same
// seed 42, same hill-climbing/deme/metapop parameters) as local_moses_
// probe.cc, but calls `local_moses()` exactly ONCE with `pa.max_gens = 5`
// set from the start, and prints the same final-state fields local_moses_
// probe.cc's own gen-5 line reports. Measured 2026-09-08: this single-call
// run gives `n_expansions=5, n_evals=113, metapop_size=20`, the identical
// `best_composite_score` (score 0, complexity 3, complexity_penalty
// 0.85714292526245117, penalized_score -0.85714292526245117), and the
// identical `best_tree`, "or(and($1 $2) !$3)" -- byte-for-byte the same
// as local_moses_probe.cc's own `gen:5` line. That equality is the actual
// proof the repeated-call construction reproduces one continuous run;
// `advanced` being `true` every generation is not, and is no longer cited
// as if it were (see local_moses_probe.cc's own header comment and this
// module's own test file, test_local_moses.py).
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I "$C" -I "$P/include" \
//       -o tools/cpp_probe/build/local_moses_single_call_probe \
//       tools/cpp_probe/local_moses_single_call_probe.cc \
//       -L "$P/lib" -L "$P/lib/opencog" \
//       -Wl,-rpath,"$P/lib" -Wl,-rpath,"$P/lib/opencog" \
//       -lmoses -lcomboreduct -lcogutil -lboost_program_options
//
// Run, bounded (CARRY-FORWARD.md's standing OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 tools/cpp_probe/build/local_moses_single_call_probe )
//
// (no arguments -- fixed scenario, run from the repo root so the relative
// dataset path resolves: trainingBoolean.dat, arity 3, exemplar "$1",
// seed 42, ONE call with max_gens=5)

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

// %.17g -- see local_moses_probe.cc's own print_score comment.
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

    // See local_moses_probe.cc's own identical comment: "$1", not "and()".
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
    pa.max_gens = 5;                 // ONE call, not five bumped-by-one calls
    moses_statistics stats;

    local_moses(mp, dex, pa, stats);

    printf("{\"seed\":%u,\"n_expansions\":%d,\"n_evals\":%d,"
           "\"metapop_size\":%zu,\"best_composite_score\":",
           seed, (int)stats.n_expansions, (int)stats.n_evals, mp.size());
    print_score(mp.best_composite_score());
    printf(",\"best_tree\":%s}\n", json_string(rendered(mp.best_tree())).c_str());
    return 0;
}
