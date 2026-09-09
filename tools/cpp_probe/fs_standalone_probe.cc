// tools/cpp_probe/fs_standalone_probe.cc
//
// Oracle for the P2 feature-selection STANDALONE port: the four
// algorithms `pymetta.feature_selection.main.feature_selection.
// select_feature_sets` newly dispatches to beyond "simple" (already
// covered by feature_selector_probe.cc/feature_selection.json) --
// "random" (algo/random.{h,cc}), "smd" (algo/stochastic_max_dependency.
// {h,cc}), "inc" (algo/incremental.{h,cc}), and "hc" (algo/deme_optimize.
// {h,cc}, which runs MOSES's own hill_climbing optimizer as the search --
// exactly the construction feature-selection.cc:141-159 builds for
// `--algo hc`).
//
// Five independent JSON groups, all from the REAL classic code:
//
//   "shuffle" -- the RAW std::random_shuffle(v.begin(), v.end(), shr)
//               pass over a plain {0,...,n-1} vector, isolated from any
//               scorer, for a few (seed, n) pairs -- an element-for-
//               element draw-order oracle for
//               pymetta.feature_selection.algo.stochastic_max_dependency.
//               _random_shuffle, independent of the "smd" group below
//               (which only checks the shuffle's downstream EFFECT on
//               selected feature sets, not its own output sequence).
//
//   "random" -- random_select_feature_sets(ctable, fs_params) at a few
//               target_sizes, seed reset immediately before each call.
//               Exercises `rand_element`'s with-replacement draw
//               (opencog/util/random.h:53).
//
//   "smd"    -- smd_select_feature_sets(ctable, fs_params) at a few
//               target_sizes, seed reset immediately before each call.
//               Exercises the std::random_shuffle pass
//               (stochastic_max_dependency.h:125) plus fs_scorer(mi).
//
//   "inc"    -- incremental_select_feature_sets(ctable, fs_params) at a
//               few target_size/threshold/inc_interaction_terms
//               combinations. Fully deterministic (no RNG at all), so no
//               seed reset is needed, but one is done anyway before each
//               case for uniformity and so a later RNG-sensitive case
//               added to this same probe can never accidentally inherit
//               state from an inc case.
//
//   "smd_pre" -- smd_select_feature_sets(ctable, fs_params) with
//               `scorer="pre"` -- exercises fs_scorer<FeatureSet>'s OTHER
//               branch (scorers/moses_matrix.h's pre_scorer, wrapping
//               moses/moses/scoring/precision_bscore.h), reachable
//               because "smd" (unlike "hc") never touches hill_climbing/
//               get_rep() at all. Ground truth for
//               pymetta.feature_selection.scorers.moses_matrix.
//               CRITICAL: `feature_selection_parameters`'s own
//               `pre_penalty`/`pre_min_activation`/`pre_max_activation`/
//               `pre_positive` fields are OMITTED from its struct ctor's
//               initializer list (feature-selection.h:47-68) --
//               genuinely indeterminate stack garbage on a bare
//               `feature_selection_parameters fs_params;` in C++. This
//               probe sets all four explicitly to the real CLI defaults
//               (1.0, 0.5, 1.0, true) before calling `scorer="pre"` --
//               confirmed necessary the hard way: omitting them made
//               `pre_scorer` crash with an unrelated-looking
//               `AssertionException` from garbage memory, while the REAL
//               installed binary (whose CLI always sets these four)
//               runs the identical case cleanly.
//
//   "hc"     -- NOT DUMPED. `dump_hc` below is dead code, kept only as
//               evidence: classic's OWN "hc" feature-selection path
//               (feature-selection.cc's own "hc" branch, exactly what
//               `dump_hc` builds) SEGFAULTS unconditionally, in classic
//               itself, confirmed two ways: (1) the REAL installed
//               standalone binary --
//                   feature-selection -i trainingBoolean.dat -u y
//                       --algo hc --target-size=1 -m 200 -o /dev/null
//                       -l ERROR -r 1
//               dumps core every time, `-l ERROR` included (so this is
//               NOT merely a debug-log-level thing); (2) a gdb backtrace
//               of that exact crash: `hill_climbing::operator()`
//               (hill-climbing.cc:69) calls
//               `iscorer.get_rep()->get_candidate(init_inst, true)`
//               UNCONDITIONALLY as an argument to its own
//               `logger().debug() << ...` line (no `is_debug_enabled()`
//               guard -- `LAZY_LOG_DEBUG` exists in Logger.h precisely
//               for that guard, and this call site does not use it), and
//               `iscorer.get_rep()` is `nullptr` for EVERY scorer
//               `moses_select_feature_sets`/`create_deme_select_
//               feature_sets` ever builds (`deme_based_scorer`/
//               `iscorer_cache`, neither overrides `iscorer_base::
//               get_rep()`'s `nullptr` default) -- so this is a 100%
//               reproducible null-pointer dereference on EVERY
//               `--algo hc` invocation, any dataset, any target_size,
//               confirmed by gdb backtrace through
//               `main -> feature_selection() -> select_features() ->
//               select_feature_sets() -> moses_select_feature_sets() ->
//               create_deme_select_feature_sets() ->
//               optimize_deme_select_feature_sets() ->
//               hill_climbing::operator()() ->
//               representation::get_candidate() [this=0x0]`. There is
//               therefore no live classic output to compare the ported
//               "hc" algorithm against -- see
//               pymetta.feature_selection.algo.deme_optimize's own
//               module docstring for how the Python port's "hc" runs
//               to completion anyway (an already-established, PRE-
//               EXISTING project-wide convention of dropping every
//               logger() call, not a new fix applied for this task).
//
// Build:
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/fs_standalone_probe \
//       tools/cpp_probe/fs_standalone_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lfeature_selection -lmoses -lcomboreduct -lcogutil \
//       -lboost_program_options
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).
//
// Usage: fs_standalone_probe <boolean-csv> <xor-csv> <seed>

#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/feature-selection/algo/deme_optimize.h>
#include <moses/feature-selection/algo/incremental.h>
#include <moses/feature-selection/algo/random.h>
#include <moses/feature-selection/algo/stochastic_max_dependency.h>
#include <moses/feature-selection/main/feature-selection.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/optimization/optimization.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <set>
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

static void print_feature_set_pop(const feature_set_pop &pop)
{
    printf("[");
    bool first = true;
    for (const auto &p : pop) {
        if (!first) printf(",");
        first = false;
        printf("{\"score\":%.17g,\"fs\":", p.first);
        print_feature_set(p.second);
        printf("}");
    }
    printf("]");
}

static void dump_random(const char *name, const CTable &ctable, unsigned seed,
                        const vector<unsigned> &target_sizes)
{
    printf("\"%s\":{", name);
    bool first_ts = true;
    for (unsigned ts : target_sizes) {
        if (!first_ts) printf(",");
        first_ts = false;

        feature_selection_parameters fs_params;
        fs_params.target_size = ts;

        randGen().seed(seed);
        feature_set_pop pop = random_select_feature_sets(ctable, fs_params);

        printf("\"%u\":", ts);
        print_feature_set_pop(pop);
    }
    printf("}");
}

static void dump_smd(const char *name, const CTable &ctable, unsigned seed,
                     const vector<unsigned> &target_sizes)
{
    printf("\"%s\":{", name);
    bool first_ts = true;
    for (unsigned ts : target_sizes) {
        if (!first_ts) printf(",");
        first_ts = false;

        feature_selection_parameters fs_params;
        fs_params.target_size = ts;

        randGen().seed(seed);
        feature_set_pop pop = smd_select_feature_sets(ctable, fs_params);

        printf("\"%u\":", ts);
        print_feature_set_pop(pop);
    }
    printf("}");
}

static void dump_smd_pre(const char *name, const CTable &ctable, unsigned seed,
                         const vector<unsigned> &target_sizes)
{
    // Exercises fs_scorer<FeatureSet>'s "pre" (precision) branch --
    // scorers/moses_matrix.h's pre_scorer -- through the SAME dispatcher
    // "mi" uses (feature-selection.h:159-200's fs_scorer<FeatureSet>),
    // via the "smd" algorithm (which never touches hill_climbing/
    // get_rep(), unlike "hc" -- see this file's header comment for why
    // "hc" itself cannot be captured at all). Ground truth for
    // pymetta.feature_selection.scorers.moses_matrix.pre_scorer.
    printf("\"%s\":{", name);
    bool first_ts = true;
    for (unsigned ts : target_sizes) {
        if (!first_ts) printf(",");
        first_ts = false;

        feature_selection_parameters fs_params;
        fs_params.target_size = ts;
        fs_params.scorer = opencog::pre;
        // feature-selection.h's own struct ctor OMITS these four fields
        // from its member-initializer list (genuinely indeterminate on a
        // bare `feature_selection_parameters()` in C++ -- see
        // feature_selection_params.py's own module docstring). Every
        // real CLI (fs-main.cc, problem-params.cc) sets them via
        // `po::value<...>()->default_value(...)` before `notify()`
        // returns; this probe must do the same or read uninitialized
        // stack memory -- confirmed the hard way: omitting this
        // initialization made `pre_scorer` crash with an unrelated
        // `AssertionException` ("Builtin 66 unknown") from garbage
        // memory, while the REAL installed binary (`--algo smd --scorer
        // pre`, which DOES go through fs-main.cc's CLI defaults) runs
        // cleanly on the identical dataset/seed/target_size.
        fs_params.pre_penalty = 1.0;
        fs_params.pre_min_activation = 0.5;
        fs_params.pre_max_activation = 1.0;
        fs_params.pre_positive = true;

        randGen().seed(seed);
        feature_set_pop pop = smd_select_feature_sets(ctable, fs_params);

        printf("\"%u\":", ts);
        print_feature_set_pop(pop);
    }
    printf("}");
}

static void dump_inc_case(const CTable &ctable, unsigned seed, unsigned target_size,
                          double threshold, unsigned interaction_terms)
{
    feature_selection_parameters fs_params;
    fs_params.target_size = target_size;
    fs_params.threshold = threshold;
    fs_params.inc_interaction_terms = interaction_terms;

    randGen().seed(seed);
    feature_set_pop pop = incremental_select_feature_sets(ctable, fs_params);

    printf("{\"target_size\":%u,\"threshold\":%.17g,\"interaction_terms\":%u,\"pop\":",
          target_size, threshold, interaction_terms);
    print_feature_set_pop(pop);
    printf("}");
}

static void dump_hc(const char *name, const CTable &ctable, unsigned seed,
                    unsigned target_size, unsigned max_evals)
{
    feature_selection_parameters fs_params;
    fs_params.target_size = target_size;
    fs_params.hc_max_evals = max_evals;

    // Exactly feature-selection.cc:141-159's own "hc" branch.
    double pop_size_ratio = 20;
    size_t max_dist = 4;
    score_t min_score_improv = 0.0;
    optim_parameters op_params;
    op_params.opt_algo = opencog::hc;
    op_params.pop_size_ratio = pop_size_ratio;
    op_params.terminate_if_gte = fs_params.hc_max_score;
    op_params.max_dist = max_dist;
    op_params.set_min_score_improv(min_score_improv);
    hc_parameters hc_params;
    hc_params.widen_search = fs_params.hc_widen_search;
    hc_params.fraction_of_nn = fs_params.hc_fraction_of_nn;
    hc_params.single_step = false;
    hc_params.crossover = fs_params.hc_crossover;
    hc_params.crossover_pop_size = fs_params.hc_crossover_pop_size;
    hc_params.crossover_min_neighbors = fs_params.hc_crossover_min_neighbors;
    hc_params.prefix_stat_deme = "FSDemes";
    hill_climbing hc_optim(op_params, hc_params);

    randGen().seed(seed);
    feature_set_pop pop = moses_select_feature_sets(ctable, hc_optim, fs_params);

    printf("\"%s\":{\"target_size\":%u,\"max_evals\":%u,\"pop\":", name, target_size, max_evals);
    print_feature_set_pop(pop);
    printf("}");
}

static void dump_shuffle(unsigned seed, unsigned n)
{
    vector<int> v;
    for (unsigned i = 0; i < n; i++) v.push_back((int)i);
    randGen().seed(seed);
    random_shuffle(v.begin(), v.end(),
                   [](ptrdiff_t i) { return randGen().randint(i); });
    printf("[");
    for (unsigned i = 0; i < v.size(); i++) {
        if (i) printf(",");
        printf("%d", v[i]);
    }
    printf("]");
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

    printf("\"shuffle\":{");
    printf("\"n5_seed1\":");
    dump_shuffle(1, 5);
    printf(",\"n8_seed1\":");
    dump_shuffle(1, 8);
    printf(",\"n5_seed42\":");
    dump_shuffle(42, 5);
    printf(",\"n1_seed1\":");
    dump_shuffle(1, 1);
    printf(",\"n0_seed1\":");
    dump_shuffle(1, 0);
    printf("},\n");

    printf("\"random\":{");
    dump_random("boolean", bool_ctable, seed, {0, 1, 2, 3});
    printf(",");
    dump_random("xor", xor_ctable, seed, {0, 1, 2});
    printf("},\n");

    printf("\"smd\":{");
    dump_smd("boolean", bool_ctable, seed, {0, 1, 2, 3});
    printf(",");
    dump_smd("xor", xor_ctable, seed, {0, 1, 2});
    printf("},\n");

    printf("\"smd_pre\":{");
    dump_smd_pre("boolean", bool_ctable, seed, {1, 2});
    printf("},\n");

    printf("\"inc\":[\n");
    dump_inc_case(bool_ctable, seed, /*target_size=*/1, /*threshold=*/0.0, /*terms=*/1);
    printf(",\n");
    dump_inc_case(bool_ctable, seed, /*target_size=*/2, /*threshold=*/0.0, /*terms=*/1);
    printf(",\n");
    dump_inc_case(bool_ctable, seed, /*target_size=*/0, /*threshold=*/0.0, /*terms=*/1);
    printf(",\n");
    dump_inc_case(bool_ctable, seed, /*target_size=*/0, /*threshold=*/0.1, /*terms=*/1);
    printf(",\n");
    dump_inc_case(bool_ctable, seed, /*target_size=*/2, /*threshold=*/0.0, /*terms=*/2);
    printf(",\n");
    dump_inc_case(xor_ctable, seed, /*target_size=*/1, /*threshold=*/0.0, /*terms=*/1);
    printf(",\n");
    dump_inc_case(xor_ctable, seed, /*target_size=*/0, /*threshold=*/0.0, /*terms=*/1);
    printf("\n]\n");

    // "hc" is deliberately NOT dumped -- see this file's header comment:
    // classic's own "hc" feature-selection path segfaults unconditionally
    // (confirmed against the real installed binary + a gdb backtrace),
    // so `dump_hc` below would crash this probe too. Kept, unused, as
    // documentation of exactly what WOULD be dumped if classic's own bug
    // were ever fixed upstream.
    (void)&dump_hc;

    printf("}\n");
    return 0;
}
