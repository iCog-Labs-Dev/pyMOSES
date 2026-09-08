// tools/cpp_probe/hillclimbing_probe.cc
//
// Oracle for pyMeTTa S9 Task 2 (moses/moses/optimization/{hill-climbing,
// optimization}.{h,cc} and moses/moses/moses/neighborhood_sampling.cc's
// sample_new_instances, both overloads).
//
// Drives the REAL opencog::moses::hill_climbing::operator() directly against
// a real field_set + instance_set<composite_score> + a local `hamming_scorer`
// (below), with hc_parameters(cross=false), and dumps, as JSON, the full
// trajectory: every scored_instance ever appended to the deme, in evaluation
// order, as {"raw": [...], "score": {...}} -- this is the step-for-step
// sequence a Python port must reproduce element-for-element, and where
// sample_new_instances writes into the deme is exercised for real (not just
// unit-tested in isolation).
//
// LANDMINE FOUND (measured, not inferred): `iscorer_base::get_rep()`
// defaults to returning `nullptr` (instance_scorer.h:37), and BOTH
// `hill_climbing::operator()` overloads unconditionally dereference it --
// `iscorer.get_rep()->get_candidate(init_inst, true)` -- as a plain argument
// to a `logger().debug() << ...` chain (hill-climbing.cc:69 in the 5-arg
// "real" overload, and again at hill-climbing.h:267 in the 4-arg
// "backward-compatible" one). C++ evaluates function-call arguments eagerly
// regardless of whether the enclosing stream chain's message ends up
// printed, so this line executes -- and null-pointer-dereferences --
// EVERY time hill_climbing is driven with a scorer whose get_rep() returns
// null, at EVERY log level, through EITHER overload. `distance_based_scorer`
// (instance_scorer.h:49) never overrides get_rep(), so it is unconditionally
// incompatible with hill_climbing as shipped. This is not a probe artifact:
// classic's own registered unit test, tests/moses/optimizationUTest.cxxtest
// (`test_hc`, ADD_CXXTEST'd and linked in tests/moses/CMakeLists.txt, NOT
// one of the commented-out tests CLAUDE.md warns about), does exactly
// `distance_based_scorer scorer(fs, target_inst); ... hc(deme, init_inst,
// scorer, eval_n, max_time);` and was confirmed, by building and running it
// directly under this session's memory bound, to core-dump at the same
// hill-climbing.cc:69 for the same reason -- immediately after printing
// "Local Search Optimization", before doing anything else. A passing `make
// test` would never catch this (cxxtest treats a crashed test process as a
// build/run problem, not a reported assertion failure) -- another instance
// of CLAUDE.md's "a passing make test is a much weaker signal than it
// looks". `hamming_scorer` below works around this the same way any REAL
// moses run does: it gives get_rep() something non-null to return.
//
// `hamming_scorer` computes the exact same "-hamming_distance(target, inst)"
// score `distance_based_scorer` does, over its OWN field_set/target (whatever
// hill_climbing is actually searching), but overrides get_rep() to return a
// pointer to a small, throwaway, REAL `representation` (built the same way
// tools/cpp_probe/representation_probe.cc's `emit_case` builds one: a
// 2-argument boolean exemplar via build_knobs) that has nothing to do with
// the search itself -- its only job is to survive `get_candidate()` without
// crashing. This is safe: get_candidate_rec only reads raw instance bits at
// the indices ITS OWN small knob map uses (a handful, for a 2-argument
// exemplar), and every `inst` this probe ever passes is at least that wide
// (packed_width() is a WORD count, and every field_set used below --
// 6 to 30 single-bit knobs -- fits in one 64-bit word, same as the
// 2-knob dummy representation's own field_set does). No combo tree / table /
// CTable / behave_cscore machinery is needed for the SCORE itself -- only
// for satisfying get_rep()'s contract -- so this probe still sidesteps the
// much larger, separately-oracle-tested combo/table stack (S9 Task 0
// already covers instance_scorer.h's own two real scorers).
//
// Two cases:
//   - "deterministic": a small (6-knob) all-boolean field_set. Every
//     iteration's neighborhood is small enough that sample_new_instances
//     always takes the exhaustive generate_all_in_neighborhood path -- no
//     RNG draw anywhere in the run. Confirms the base algorithm end to end.
//   - "sampled": a larger (30-knob) all-boolean field_set with a low
//     hc_parameters.max_nn_evals(50) cap and widen_search=true. distance 0-1
//     stays exhaustive (30 true neighbors, under the cap), climbs to the
//     exact global optimum (a pure negative-Hamming-distance landscape has
//     no local optima under single/multi-bit-flip neighborhoods), then
//     widen_search pushes distance to 2/3/4, where the TRUE neighbor count
//     (C(30,2)=435, C(30,3)=4060, C(30,4)=27405) exceeds the 50-instance
//     cap, forcing sample_new_instances into its RNG-sampling branch
//     (sample_from_neighborhood, drawing through lazy_random_selector) for
//     real, inside a real hill_climbing run -- not just Task 1's isolated
//     per-function calls. optim_parameters.terminate_if_gte is set far above
//     the scorer's actual maximum (0) so the run cannot terminate early via
//     the "found best possible score" exit and is forced to widen.
//
// Also covers sample_new_instances directly (both overloads) against a
// prefilled instance_set, independent of the full hill_climbing loop.
//
// Build (see tools/capture_hillclimbing.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/hillclimbing_probe \
//       tools/cpp_probe/hillclimbing_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and this session's own standing OOM caution. Both cases
// above are sized to stay well within that bound (deme sizes in the low
// thousands at most).

#include <moses/comboreduct/reduct/reduct.h>
#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/representation/field_set.h>
#include <moses/moses/representation/instance_scorer.h>
#include <moses/moses/representation/instance_set.h>
#include <moses/moses/representation/representation.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

static field_set make_bit_fs(size_t n)
{
    return field_set(field_set::disc_spec(2), n);
}

// See the file header's LANDMINE note. Builds a REAL representation over an
// `arity`-argument boolean exemplar -- same recipe as
// tools/cpp_probe/representation_probe.cc's emit_case() -- and uses ITS OWN
// field_set (rep.fields()) as the ENTIRE search space for hill_climbing,
// rather than a separately-constructed synthetic field_set. This is load-
// bearing, not cosmetic: get_candidate() is called by hill-climbing.cc's
// (dropped-in-Python) debug logging on EVERY logged instance, EVERY
// iteration (init_inst, center_inst, and every deme[i].first) -- real
// logical knobs have multiplicity 3 (knobs.cc's own "Index greater than 2"
// assertion, measured directly), so a knob's raw 2-bit field only accepts
// {0,1,2}, not the full {0,1,2,3} a 2-bit field can otherwise hold. Reusing
// a SEPARATE, differently-shaped field_set for the actual search (as a first
// attempt at this probe did) reinterprets whatever bit pattern the search
// produces through the representation's OWN field layout and reliably
// produces an out-of-range raw value once the search explores enough
// distinct instances -- measured directly: it aborts with exactly that
// assertion. Searching over rep.fields() itself instead means every value
// neighborhood_sampling.py ever writes into a disc knob is already
// range-checked against that knob's own multiplicity (vary_n_knobs/
// sample_from_neighborhood only ever assign `0 <= v < multy`), so
// get_candidate() never sees an invalid setting.
static representation make_representation(size_t arity)
{
    combo_tree exemplar = type_to_exemplar(id::boolean_type); // combo_tree(id::logical_and)
    type_tree signature = gen_signature(id::boolean_type, arity);
    vertex_set ignore_ops;
    static logical_reduction reduction(ignore_ops);
    const rule& delegate = reduction(2);
    return representation(delegate, delegate, exemplar, signature, ignore_ops, nullptr,
                          nullptr, false, -1.0);
}

// Computes the same score as instance_scorer.h's distance_based_scorer
// (-hamming_distance(target, inst), null complexity/penalties), but
// overrides get_rep() to return a non-null (if otherwise irrelevant)
// representation, working around the LANDMINE documented in the file
// header. `fs`/`target` are references, same lifetime contract
// distance_based_scorer itself has (caller must outlive the scorer).
struct hamming_scorer : public iscorer_base {
    hamming_scorer(const field_set& fs_, const instance& target_,
                  const representation& rep_)
        : fs(fs_), target(target_), rep(rep_) {}

    composite_score operator()(const instance& inst) const override
    {
        score_t sc = -fs.hamming_distance(target, inst);
        return composite_score(sc, 0, 0, 0);
    }

    const representation* get_rep() const override { return &rep; }

protected:
    const field_set& fs;
    const instance& target;
    const representation& rep;
};

static void print_score(const composite_score& cs)
{
    printf("{\"score\":%.9g,\"complexity\":%u,\"complexity_penalty\":%.9g,"
           "\"uniformity_penalty\":%.9g,\"penalized_score\":%.9g}",
           (double)cs.get_score(), (unsigned)cs.get_complexity(),
           (double)cs.get_complexity_penalty(),
           (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

static void print_raw(const field_set& fs, const instance& inst)
{
    printf("[");
    for (size_t i = 0; i < fs.raw_size(); ++i) {
        if (i) printf(",");
        printf("%lu", (unsigned long)fs.get_raw(inst, i));
    }
    printf("]");
}

// Dumps enough of `fs`'s structure -- NOT going through build_knobs/
// representation at all -- for a Python test to reconstruct an equivalent
// field_set directly via field_set([disc_spec(m) for m in ...]), without
// this task needing to reach into S8's already-separately-tested
// build_knobs/representation pipeline at all. Every knob `make_representation`
// produces is a multi-valued disc knob (verified: this exemplar/arity
// combination produces zero single-bit knobs), but bit knobs are dumped too
// for completeness/robustness against that changing.
static void print_field_shape(const field_set& fs)
{
    printf("{\"disc_multiplicities\":[");
    for (size_t i = fs.begin_disc_raw_idx(); i < fs.end_disc_raw_idx(); ++i) {
        if (i > fs.begin_disc_raw_idx()) printf(",");
        printf("%u", (unsigned)fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy);
    }
    printf("],\"n_bits\":%zu}", fs.n_bits());
}

static void print_deme_trajectory(const field_set& fs, const deme_t& deme)
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

// Runs hill_climbing to completion and dumps its full trajectory plus
// optim_stats. `hcp` and `op` are passed by value (cheap, small structs) so
// each case gets an independent, freshly-seeded run.
static void run_case(const char* label, size_t n_knobs, unsigned seed,
                      optim_parameters op, hc_parameters hcp,
                      unsigned max_evals, time_t max_time)
{
    representation rep = make_representation(n_knobs);
    const field_set& fs = rep.fields();

    instance target(fs.packed_width());
    for (size_t i = 0; i < fs.raw_size(); ++i) {
        if (fs.begin_disc_raw_idx() <= i && i < fs.end_disc_raw_idx())
            fs.set_raw(target, i, fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy - 1);
        else if (fs.begin_bit_raw_idx() <= i && i < fs.end_bit_raw_idx())
            fs.set_raw(target, i, 1);
    }

    hamming_scorer scorer(fs, target, rep);

    randGen().seed(seed);
    deme_t deme(fs);
    hill_climbing hc(op, hcp);
    // The 4-arg operator()(deme, iscorer, max_evals, max_time) overload --
    // "all of the current code" (hill-climbing.h:260-261) -- constructs its
    // own null init_inst and forwards; using it here (rather than the 5-arg
    // form) matches that real-world call shape exactly, now that
    // hamming_scorer's get_rep() is non-null (see the file header).
    hc(deme, scorer, max_evals, max_time);

    printf("\"%s\":{\"seed\":%u,\"n_knobs\":%zu,\"n_evals\":%zu,"
           "\"nsteps\":%u,\"over_budget\":%s,\"field_shape\":",
           label, seed, n_knobs, deme.n_evals, hc.nsteps,
           hc.over_budget ? "true" : "false");
    print_field_shape(fs);
    printf(",\"trajectory\":");
    print_deme_trajectory(fs, deme);
    printf("}");
}

// Exercises sample_new_instances directly (both overloads) against a
// prefilled instance_set, independent of the full hill_climbing loop.
static void run_sample_new_instances_cases()
{
    printf("[");
    struct Case {
        unsigned multiplicity;
        size_t n_knobs;
        unsigned dist;
        size_t number_of_new_instances;
        size_t current_number_of_instances;
        unsigned seed;
        bool give_total; // true => overload 1 (total given); false => overload 2
    };
    vector<Case> cases = {
        {2, 6, 1, 3, 0, 1, true},    // small, exhaustive-eligible, total given
        {2, 6, 1, 3, 0, 1, false},   // same, but via overload 2 (total omitted)
        {2, 20, 2, 10, 5, 42, true}, // larger: forces the RNG-sampling branch
        {2, 20, 2, 10, 5, 42, false},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        const Case& cs = cases[c];
        if (c) printf(",");

        field_set fs = make_bit_fs(cs.n_knobs);
        instance center(fs.packed_width());
        // A mixed center (not all-zero) so raw-value output isn't trivial.
        for (size_t i = 0; i < cs.n_knobs; i += 3)
            fs.set_raw(center, i, 1);

        deme_t deme(cs.current_number_of_instances, fs);

        randGen().seed(cs.seed);
        size_t total_estimate =
            count_neighborhood_size(fs, center, cs.dist, cs.number_of_new_instances);

        size_t returned;
        if (cs.give_total)
            returned = sample_new_instances(total_estimate, cs.number_of_new_instances,
                                            cs.current_number_of_instances, center, deme,
                                            cs.dist);
        else
            returned = sample_new_instances(cs.number_of_new_instances,
                                            cs.current_number_of_instances, center, deme,
                                            cs.dist);

        printf("{\"multiplicity\":%u,\"n_knobs\":%zu,\"dist\":%u,"
               "\"number_of_new_instances\":%zu,\"current_number_of_instances\":%zu,"
               "\"seed\":%u,\"give_total\":%s,\"returned\":%zu,\"deme_size\":%zu,"
               "\"new_slice\":[",
               cs.multiplicity, cs.n_knobs, cs.dist, cs.number_of_new_instances,
               cs.current_number_of_instances, cs.seed,
               cs.give_total ? "true" : "false", returned, deme.size());
        for (size_t i = cs.current_number_of_instances; i < deme.size(); ++i) {
            if (i > cs.current_number_of_instances) printf(",");
            print_raw(fs, deme[i].first);
        }
        printf("]}");
    }
    printf("]");
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    printf("{\n");

    printf("\"sample_new_instances\":");
    run_sample_new_instances_cases();
    printf(",\n");

    // ---- deterministic hill_climbing case ----
    {
        optim_parameters op; // defaults: terminate_if_gte=0
        hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/false);
        run_case("deterministic", 6, 1, op, hcp, 500, 86400);
    }
    printf(",\n");

    // ---- RNG-sampling hill_climbing case ----
    {
        optim_parameters op("hc", 20, /*terminate_if_gte=*/1000.0f);
        hc_parameters hcp(/*widen=*/true, /*step=*/false, /*cross=*/false,
                          /*max_evals=*/50);
        run_case("sampled", 30, 1, op, hcp, 4000, 86400);
    }

    printf("\n}\n");
    return 0;
}
