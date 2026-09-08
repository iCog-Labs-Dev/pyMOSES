// tools/cpp_probe/crossover_probe.cc
//
// Oracle for pyMeTTa S12 (moses/moses/optimization/hill-climbing.cc's
// cross_top_one :593, cross_top_two :624, cross_top_three :664 and
// crossover :701, plus the crossover gate at :139-152 and the last_chance
// block at :434-458).
//
// Five independent groups of cases, all emitted as one JSON object:
//
//   "partial_sort" -- raw std::partial_sort(first, middle, last,
//       std::greater<deme_inst_t>()) over a hand-built vector of
//       deme_inst_t, dumping the resulting order.  std::partial_sort is
//       NOT a stable sort and the standard grants no tie order at all, so
//       this measures what libstdc++ actually does, element for element,
//       for tie configurations the cross_top_* functions really hit.  The
//       identity of each element is carried in its INSTANCE bits (an
//       integer id), which the comparator never looks at (scored_instance::
//       operator> compares .second only, scored_instance.h:48-51), so the
//       dumped id order IS the permutation the algorithm chose.
//
//   "cross_top_one"/"cross_top_two"/"cross_top_three"/"crossover" --
//       direct calls into the REAL (protected, hence the hc_probe subclass
//       below) member functions against a prefilled deme, dumping the whole
//       deme afterwards plus the return value.  Each case is run four
//       times, once per function, from an identical freshly built deme.
//
//   "hill_climbing" -- full hill_climbing::operator() runs with
//       hc_parameters.crossover = TRUE (classic's own default,
//       problem-params.cc:424), covering BOTH gate entry paths:
//         * "last_chance_only"/"large_nbh_at_distance_2"{,_immediate}: small
//           field_sets whose DISTANCE-1 neighborhood estimate is far below
//           crossover_min_neighbors (400), so the FIRST crossover attempt
//           can only come through the last_chance one-shot at
//           hill-climbing.cc:450 -- though for the latter two, by the time
//           it fires, large_nbh has already become true at distance 2 (see
//           the cases' own comments below, measured).
//         * "large_nbh_*": a field_set sized so that
//           estimate_neighborhood(1) == 2 * information_theoretic_bits
//           straddles the 400 boundary exactly (hill-climbing.cc:504-505),
//           giving a true/false pair on the boundary itself.
//       Ground truth for WHICH iterations crossed over is taken from
//       classic's own DEBUG log ("Iteration: %u" at hill-climbing.cc:108
//       and "Crossover enabled (...)"/"Crossover disabled (...)" at
//       hill-climbing.cc:162/184), captured to a file and parsed back here
//       -- not inferred from scores.
//
//   "local_moses" -- the S11 end-to-end scenario (the same dataset,
//       exemplar, seed and five-generation loop as
//       tools/cpp_probe/local_moses_probe.cc) re-run with
//       hc_parameters.crossover = TRUE, i.e. classic's DEFAULT
//       configuration.  S11's own probe pinned the crossover-DISABLED
//       trajectory; this pins the default one, and records the same
//       iteration-by-iteration crossover ground truth from the DEBUG log,
//       so a port can assert crossover fired rather than infer it from a
//       matching score.
//
// LANDMINE (inherited verbatim from tools/cpp_probe/hillclimbing_probe.cc,
// see its file header for the full measurement): both
// hill_climbing::operator() overloads unconditionally dereference
// iscorer.get_rep(), which iscorer_base defaults to null, as an eagerly
// evaluated argument of a logger() chain.  So every hill_climbing run needs
// a scorer with a non-null get_rep(), and the searched field_set must be the
// representation's OWN field_set (reinterpreting foreign bit patterns
// through a representation's knob layout trips knobs.cc's multiplicity
// assertion).  hamming_scorer below is the same workaround that probe uses.
//
// Build (see tools/capture_crossover.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/crossover_probe \
//       tools/cpp_probe/crossover_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil -lboost_program_options
//
// `-lboost_program_options` is REQUIRED because of the "local_moses" group
// below -- see tools/cpp_probe/local_moses_probe.cc's own header for the
// measured link error without it.
//
// Usage: crossover_probe [logdir]   (logdir defaults to /tmp)
// Run it FROM THE REPO ROOT: the "local_moses" group loads the relative
// path tests/fixtures/table_data/trainingBoolean.dat.
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and this session's standing OOM caution.  Every case
// below is deliberately sized so the largest deme stays in the low
// thousands of instances.

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/moses/deme/deme_expander.h>
#include <moses/moses/metapopulation/metapopulation.h>
#include <moses/moses/moses/local_moses.h>
#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <moses/moses/scoring/bscores.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/representation/field_set.h>
#include <moses/moses/representation/instance_scorer.h>
#include <moses/moses/representation/instance_set.h>
#include <moses/moses/representation/representation.h>
#include <opencog/util/Logger.h>
#include <opencog/util/mt19937ar.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

// cross_top_one/two/three/crossover are `protected` members of
// hill_climbing (hill-climbing.h:191/222/228/235, under the `protected:` at
// :180).  A derived class can re-export them with `using`, which is plain
// standard C++ -- no `#define private public` and therefore no risk of a
// layout/ODR mismatch against the already-compiled libmoses.so.
struct hc_probe : public hill_climbing {
    hc_probe(const optim_parameters& op = optim_parameters(),
             const hc_parameters& hc = hc_parameters())
        : hill_climbing(op, hc) {}
    using hill_climbing::cross_top_one;
    using hill_climbing::cross_top_two;
    using hill_climbing::cross_top_three;
    using hill_climbing::crossover;
    using hill_climbing::estimate_neighborhood;
    using hill_climbing::n_new_instances;
};

static field_set make_bit_fs(size_t n)
{
    return field_set(field_set::disc_spec(2), n);
}

// %.9g prints the C library's "nan"/"inf" spellings, which are not JSON.
// Python's json module accepts NaN/Infinity/-Infinity (its own non-standard
// but default-enabled extension), so emit those instead; every other value
// stays %.9g, enough to round-trip a float32 score exactly.
static void print_double(double v)
{
    if (std::isnan(v)) {
        printf("NaN");
        return;
    }
    if (std::isinf(v)) {
        printf(v > 0 ? "Infinity" : "-Infinity");
        return;
    }
    printf("%.9g", v);
}

static void print_score(const composite_score& cs)
{
    printf("{\"score\":");
    print_double((double)cs.get_score());
    printf(",\"complexity\":%u,\"complexity_penalty\":", (unsigned)cs.get_complexity());
    print_double((double)cs.get_complexity_penalty());
    printf(",\"uniformity_penalty\":");
    print_double((double)cs.get_uniformity_penalty());
    printf(",\"penalized_score\":");
    print_double((double)cs.get_penalized_score());
    printf("}");
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

static void print_deme(const field_set& fs, const deme_t& deme)
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

static void print_field_shape(const field_set& fs)
{
    printf("{\"disc_multiplicities\":[");
    for (size_t i = fs.begin_disc_raw_idx(); i < fs.end_disc_raw_idx(); ++i) {
        if (i > fs.begin_disc_raw_idx()) printf(",");
        printf("%u", (unsigned)fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy);
    }
    printf("],\"n_bits\":%zu}", fs.n_bits());
}

// --------------------------------------------------------------------------
// Group 1: std::partial_sort tie order
// --------------------------------------------------------------------------

// One element of a synthetic sample.  `id` is written into the instance bits
// (so it survives any permutation and identifies the element); the remaining
// fields build the composite_score the comparator actually reads.
struct Sample {
    unsigned long id;
    double score;
    unsigned complexity;
    double cpen;
    double upen;
};

static instance id_instance(const field_set& fs, unsigned long id)
{
    instance inst(fs.packed_width());
    for (size_t i = 0; i < fs.n_bits(); ++i)
        fs.set_raw(inst, fs.begin_bit_raw_idx() + i, (id >> i) & 1UL);
    return inst;
}

static unsigned long instance_id(const field_set& fs, const instance& inst)
{
    unsigned long id = 0;
    for (size_t i = 0; i < fs.n_bits(); ++i)
        if (fs.get_raw(inst, fs.begin_bit_raw_idx() + i)) id |= (1UL << i);
    return id;
}

struct PSCase {
    const char* label;
    size_t mid;
    vector<Sample> items;
};

static void run_partial_sort_cases()
{
    const double nan_v = numeric_limits<double>::quiet_NaN();
    const double inf_v = numeric_limits<double>::infinity();

    vector<PSCase> cases;

    // Every score equal AND every complexity equal => every pair is a tie
    // under composite_score::operator< (types.cc:131-149), which is exactly
    // the configuration a Boolean deme reaches constantly (many neighbours
    // score identically with identical complexity).
    {
        PSCase c{"all_tied_8_mid4", 4, {}};
        for (unsigned long i = 0; i < 8; ++i) c.items.push_back({i, -3.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    {
        PSCase c{"all_tied_12_mid5", 5, {}};
        for (unsigned long i = 0; i < 12; ++i) c.items.push_back({i, -3.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    {
        PSCase c{"all_tied_40_mid9", 9, {}};
        for (unsigned long i = 0; i < 40; ++i) c.items.push_back({i, -3.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    // Two tied blocks: ids 0..4 score -1, ids 5..9 score -2.
    {
        PSCase c{"two_blocks_10_mid4", 4, {}};
        for (unsigned long i = 0; i < 10; ++i)
            c.items.push_back({i, i < 5 ? -1.0 : -2.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    // Strictly descending / ascending: no ties at all, the control cases.
    {
        PSCase c{"descending_6_mid3", 3, {}};
        for (unsigned long i = 0; i < 6; ++i) c.items.push_back({i, -(double)i, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    {
        PSCase c{"ascending_6_mid3", 3, {}};
        for (unsigned long i = 0; i < 6; ++i)
            c.items.push_back({i, -(double)(6 - i), 0, 0.0, 0.0});
        cases.push_back(c);
    }
    // Equal penalized score but different complexity: NOT a tie -- operator<
    // breaks it by complexity (types.cc:148, `lef == rig and complexity >
    // r.complexity`), so higher complexity sorts EARLIER under greater<>.
    {
        PSCase c{"equal_score_diff_complexity_6_mid3", 3, {}};
        for (unsigned long i = 0; i < 6; ++i) c.items.push_back({i, -1.0, (unsigned)i, 0.0, 0.0});
        cases.push_back(c);
    }
    // Non-zero penalties, so raw score and penalized score differ and a test
    // built on the raw score alone would order these differently: raw score
    // ascends with id while the penalty makes the penalized score descend.
    {
        PSCase c{"penalties_invert_order_8_mid4", 4, {}};
        for (unsigned long i = 0; i < 8; ++i)
            c.items.push_back({i, (double)i, 0, 2.0 * (double)i, 0.5 * (double)i});
        cases.push_back(c);
    }
    // NaN: composite_score::operator< (types.cc:136-137) says nan < x for
    // every non-nan x, and nan vs nan is a tie.
    {
        PSCase c{"with_nan_8_mid4", 4, {}};
        for (unsigned long i = 0; i < 8; ++i)
            c.items.push_back({i, (i % 3 == 0) ? nan_v : -(double)i, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    { // +/-inf alongside a tie block
        PSCase c{"with_inf_7_mid3", 3,
                 {{0, -inf_v, 0, 0.0, 0.0},
                  {1, -1.0, 0, 0.0, 0.0},
                  {2, -1.0, 0, 0.0, 0.0},
                  {3, inf_v, 0, 0.0, 0.0},
                  {4, -1.0, 0, 0.0, 0.0},
                  {5, -inf_v, 0, 0.0, 0.0},
                  {6, -1.0, 0, 0.0, 0.0}}};
        cases.push_back(c);
    }
    // Degenerate middles: mid == 0, mid == size, size == 1.
    {
        PSCase c{"all_tied_5_mid0", 0, {}};
        for (unsigned long i = 0; i < 5; ++i) c.items.push_back({i, -1.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    {
        PSCase c{"all_tied_5_mid5", 5, {}};
        for (unsigned long i = 0; i < 5; ++i) c.items.push_back({i, -1.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }
    {
        PSCase c{"single_1_mid1", 1, {{0, -1.0, 0, 0.0, 0.0}}};
        cases.push_back(c);
    }
    {
        PSCase c{"pair_tied_2_mid1", 1, {{0, -1.0, 0, 0.0, 0.0}, {1, -1.0, 0, 0.0, 0.0}}};
        cases.push_back(c);
    }
    // A shuffled tie/non-tie mix at a size big enough for the heap phases to
    // actually interleave (make_heap over [first,middle), then the sift-in
    // scan over [middle,last), then sort_heap).
    {
        PSCase c{"mixed_20_mid7", 7, {}};
        const double sc[20] = {-2, -1, -2, -3, -1, -1, -4, -2, -3, -1,
                               -2, -5, -1, -3, -2, -1, -4, -2, -3, -1};
        for (unsigned long i = 0; i < 20; ++i) c.items.push_back({i, sc[i], 0, 0.0, 0.0});
        cases.push_back(c);
    }

    printf("[");
    for (size_t c = 0; c < cases.size(); ++c) {
        if (c) printf(",");
        const PSCase& pc = cases[c];
        // 6 bits is enough to give all 40 ids a distinct pattern.
        field_set fs = make_bit_fs(6);
        vector<deme_inst_t> v;
        for (const Sample& s : pc.items) {
            deme_inst_t si(id_instance(fs, s.id));
            si.second = composite_score(s.score, s.complexity, s.cpen, s.upen);
            v.push_back(si);
        }
        std::partial_sort(v.begin(), v.begin() + pc.mid, v.end(), std::greater<deme_inst_t>());

        printf("{\"label\":\"%s\",\"mid\":%zu,\"size\":%zu,\"input\":[", pc.label, pc.mid,
               pc.items.size());
        for (size_t i = 0; i < pc.items.size(); ++i) {
            if (i) printf(",");
            const Sample& s = pc.items[i];
            printf("{\"id\":%lu,\"score\":", s.id);
            print_double(s.score);
            printf(",\"complexity\":%u,\"complexity_penalty\":", s.complexity);
            print_double(s.cpen);
            printf(",\"uniformity_penalty\":");
            print_double(s.upen);
            printf("}");
        }
        printf("],\"order\":[");
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) printf(",");
            printf("%lu", instance_id(fs, v[i].first));
        }
        printf("]}");
    }
    printf("]");
}

// --------------------------------------------------------------------------
// Group 2: direct cross_top_* / crossover calls
// --------------------------------------------------------------------------

struct XCase {
    const char* label;
    size_t n_bits;
    size_t sample_start; // == the number of filler instances ahead of the sample
    size_t num_to_make;
    unsigned long base_id;
    vector<Sample> samples;
};

// Builds the deme the C++ functions expect: `sample_start` filler entries,
// then `samples.size()` sample entries, and nothing else -- i.e. deme.size()
// == deme_size == sample_start + sample_size, exactly the invariant
// hill_climbing::operator() maintains when it calls crossover() at :165.
static void build_xdeme(const XCase& c, const field_set& fs, deme_t& deme)
{
    deme.resize(c.sample_start + c.samples.size());
    for (size_t i = 0; i < c.sample_start; ++i) {
        // Filler ids are pushed far away from sample ids (bit 5 set) so a
        // crossed instance can never be confused with a filler one.
        deme[i].first = id_instance(fs, 32UL + i);
        deme[i].second = composite_score(-1000.0 - (double)i, 0, 0.0, 0.0);
    }
    for (size_t i = 0; i < c.samples.size(); ++i) {
        const Sample& s = c.samples[i];
        deme[c.sample_start + i].first = id_instance(fs, s.id);
        deme[c.sample_start + i].second = composite_score(s.score, s.complexity, s.cpen, s.upen);
    }
}

static void emit_xcase(const XCase& c, const char* which)
{
    field_set fs = make_bit_fs(c.n_bits);
    deme_t deme(fs);
    build_xdeme(c, fs, deme);
    instance base = id_instance(fs, c.base_id);

    hc_probe hc;
    size_t deme_size = c.sample_start + c.samples.size();
    size_t returned = 0;
    if (!strcmp(which, "cross_top_one"))
        returned = hc.cross_top_one(deme, deme_size, c.num_to_make, c.sample_start,
                                    c.samples.size(), base);
    else if (!strcmp(which, "cross_top_two"))
        returned = hc.cross_top_two(deme, deme_size, c.num_to_make, c.sample_start,
                                    c.samples.size(), base);
    else if (!strcmp(which, "cross_top_three"))
        returned = hc.cross_top_three(deme, deme_size, c.num_to_make, c.sample_start,
                                      c.samples.size(), base);
    else
        returned = hc.crossover(deme, deme_size, c.num_to_make, c.sample_start,
                                c.samples.size(), base);

    printf("{\"label\":\"%s\",\"n_bits\":%zu,\"sample_start\":%zu,\"sample_size\":%zu,"
           "\"num_to_make\":%zu,\"base_id\":%lu,\"deme_size\":%zu,\"returned\":%zu,"
           "\"crossover_pop_size\":%u,\"samples\":[",
           c.label, c.n_bits, c.sample_start, c.samples.size(), c.num_to_make, c.base_id,
           deme_size, returned, (unsigned)hc_parameters().crossover_pop_size);
    for (size_t i = 0; i < c.samples.size(); ++i) {
        if (i) printf(",");
        const Sample& s = c.samples[i];
        printf("{\"id\":%lu,\"score\":", s.id);
        print_double(s.score);
        printf(",\"complexity\":%u,\"complexity_penalty\":", s.complexity);
        print_double(s.cpen);
        printf(",\"uniformity_penalty\":");
        print_double(s.upen);
        printf("}");
    }
    printf("],\"deme\":");
    print_deme(fs, deme);
    printf("}");
}

static vector<XCase> make_xcases()
{
    vector<XCase> cases;

    // A canonical single-bit-flip sample: sample id i has exactly bit i set,
    // base is all-zero.  merge_instance then copies exactly the flipped bit,
    // so a crossed child's id is the OR of the contributing ids -- readable
    // by eye in the fixture.
    auto flips = [](size_t n) {
        vector<Sample> v;
        for (size_t i = 0; i < n; ++i) v.push_back({1UL << i, -(double)i, 0, 0.0, 0.0});
        return v;
    };
    // Same shape but every score identical: forces the partial_sort ties to
    // decide the outcome.
    auto flips_tied = [](size_t n) {
        vector<Sample> v;
        for (size_t i = 0; i < n; ++i) v.push_back({1UL << i, -1.0, 0, 0.0, 0.0});
        return v;
    };

    cases.push_back({"flips8_start0_make6", 8, 0, 6, 0UL, flips(8)});
    cases.push_back({"flips8_start3_make6", 8, 3, 6, 0UL, flips(8)});
    // num_to_make above every clamp: cross_top_one clamps to sample_size-1,
    // cross_top_two to C(n,2), cross_top_three to C(n,3).
    cases.push_back({"flips6_start0_make100", 6, 0, 100, 0UL, flips(6)});
    // num_to_make == 0 and == 1: the degenerate ends.
    cases.push_back({"flips6_start0_make0", 6, 0, 0, 0UL, flips(6)});
    cases.push_back({"flips6_start0_make1", 6, 0, 1, 0UL, flips(6)});
    // Exactly on cross_top_one's clamp boundary (sample_size - 1 == 5).
    cases.push_back({"flips6_start0_make5", 6, 0, 5, 0UL, flips(6)});
    cases.push_back({"flips6_start0_make6", 6, 0, 6, 0UL, flips(6)});
    // Exactly on cross_top_two's clamp boundary: C(5,2) == 10.
    cases.push_back({"flips5_start0_make10", 5, 0, 10, 0UL, flips(5)});
    cases.push_back({"flips5_start0_make11", 5, 0, 11, 0UL, flips(5)});
    // Exactly on cross_top_three's clamp boundary: C(6,3) == 20.
    cases.push_back({"flips6_start0_make20", 6, 0, 20, 0UL, flips(6)});
    cases.push_back({"flips6_start0_make21", 6, 0, 21, 0UL, flips(6)});
    // Smallest sample sizes: sample_size 1/2/3 exercise cross_top_two's and
    // cross_top_three's `max` computations at their zero/one edges.
    cases.push_back({"flips1_start0_make4", 4, 0, 4, 0UL, flips(1)});
    cases.push_back({"flips2_start0_make4", 4, 0, 4, 0UL, flips(2)});
    cases.push_back({"flips3_start0_make4", 4, 0, 4, 0UL, flips(3)});
    // All scores tied: the partial_sort tie order alone decides which
    // instances get crossed and in what order.
    cases.push_back({"tied8_start0_make6", 8, 0, 6, 0UL, flips_tied(8)});
    cases.push_back({"tied12_start2_make20", 12, 2, 20, 0UL, flips_tied(12)});
    // A non-zero base: merge_instance's `if (*bit != *rit) *tit = *rit;`
    // then also has to CLEAR bits, not only set them.
    cases.push_back({"flips8_base_ff_make6", 8, 0, 6, 0xFFUL, flips(8)});
    // Multi-bit sample patterns with non-zero penalties, so that raw and
    // penalized scores order the sample differently.  Penalized score is
    // score - cpen - upen, so this orders id 7 best, id 0 worst, while the
    // raw score orders them the other way.
    {
        vector<Sample> v;
        for (unsigned long i = 0; i < 8; ++i)
            v.push_back({(i * 37UL) & 0xFFUL, (double)i, (unsigned)(i % 3), 3.0 * (double)i,
                         0.25 * (double)i});
        cases.push_back({"penalized8_start1_make9", 8, 1, 9, 0x0FUL, v});
    }
    // Large enough for cross_top_three's index collisions to be visible over
    // several i-blocks.
    cases.push_back({"flips10_start0_make40", 10, 0, 40, 0UL, flips(10)});
    cases.push_back({"tied10_start0_make40", 10, 0, 40, 0UL, flips_tied(10)});
    // The default budget split: crossover() with num_to_make == 120 gives
    // 40/40/40 exactly; 121 and 122 push the remainder into cross_top_one.
    cases.push_back({"flips24_start0_make120", 24, 0, 120, 0UL, flips(24)});
    cases.push_back({"flips24_start0_make121", 24, 0, 121, 0UL, flips(24)});
    cases.push_back({"flips24_start0_make122", 24, 0, 122, 0UL, flips(24)});
    // Budget below crossover_pop_size, so crossover() clamps to the budget.
    cases.push_back({"flips24_start0_make100", 24, 0, 100, 0UL, flips(24)});
    return cases;
}

static void run_xcases()
{
    static const char* which[] = {"cross_top_one", "cross_top_two", "cross_top_three",
                                  "crossover"};
    vector<XCase> cases = make_xcases();
    printf("{");
    for (size_t w = 0; w < 4; ++w) {
        if (w) printf(",");
        printf("\"%s\":[", which[w]);
        for (size_t c = 0; c < cases.size(); ++c) {
            if (c) printf(",");
            emit_xcase(cases[c], which[w]);
        }
        printf("]");
    }
    printf("}");
}

// --------------------------------------------------------------------------
// Group 3: full hill_climbing runs with crossover enabled
// --------------------------------------------------------------------------

static representation make_representation(size_t arity)
{
    combo_tree exemplar = type_to_exemplar(id::boolean_type);
    type_tree signature = gen_signature(id::boolean_type, arity);
    vertex_set ignore_ops;
    static logical_reduction reduction(ignore_ops);
    const rule& delegate = reduction(2);
    return representation(delegate, delegate, exemplar, signature, ignore_ops, nullptr, nullptr,
                          false, -1.0);
}

struct hamming_scorer : public iscorer_base {
    hamming_scorer(const field_set& fs_, const instance& target_, const representation& rep_)
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

// Reads back the DEBUG log classic itself wrote and extracts, per iteration,
// whether hill-climbing.cc:152's `if (xover)` branch was taken and the reason
// string it logged.  This is ground truth straight out of the C++ control
// flow, not an inference from the resulting scores.
static void print_xover_log(const string& path)
{
    ifstream in(path);
    printf("[");
    string line;
    long iteration = -1;
    bool first = true;
    while (getline(in, line)) {
        size_t p = line.find("Iteration: ");
        if (p != string::npos) {
            iteration = strtol(line.c_str() + p + 11, nullptr, 10);
            continue;
        }
        bool enabled = line.find("Crossover enabled (") != string::npos;
        bool disabled = line.find("Crossover disabled (") != string::npos;
        if (!enabled && !disabled) continue;
        size_t open = line.rfind('(');
        size_t close = line.rfind(')');
        string why = (open != string::npos && close != string::npos && close > open)
                         ? line.substr(open + 1, close - open - 1)
                         : string();
        if (!first) printf(",");
        first = false;
        printf("{\"iteration\":%ld,\"xover\":%s,\"why\":\"", iteration,
               enabled ? "true" : "false");
        for (char ch : why) {
            if (ch == '"' || ch == '\\') printf("\\");
            printf("%c", ch);
        }
        printf("\"}");
    }
    printf("]");
}

// `null_target` makes the all-zero instance -- which is exactly the
// `init_inst` the 4-arg operator() builds (hill-climbing.h:264) -- the
// global optimum.  The search then plateaus on its SECOND iteration, which
// arms last_chance at hill-climbing.cc:450 and fires crossover on iteration
// 3, in a run small enough to read by eye.  With the default (far) target
// the same plateau is only reached after the climb finishes, exercising the
// same path from a realistic trajectory instead.
static void run_hc_case(const char* label, size_t arity, unsigned seed, optim_parameters op,
                        hc_parameters hcp, unsigned max_evals, time_t max_time,
                        const string& logdir, bool null_target = false)
{
    representation rep = make_representation(arity);
    const field_set& fs = rep.fields();

    instance target(fs.packed_width());
    if (!null_target) {
        for (size_t i = 0; i < fs.raw_size(); ++i) {
            if (fs.begin_disc_raw_idx() <= i && i < fs.end_disc_raw_idx())
                fs.set_raw(target, i, fs.disc_and_bit()[fs.raw_to_disc_idx(i)].multy - 1);
            else if (fs.begin_bit_raw_idx() <= i && i < fs.end_bit_raw_idx())
                fs.set_raw(target, i, 1);
        }
    }

    hamming_scorer scorer(fs, target, rep);

    string logpath = logdir + "/crossover_probe_" + label + ".log";
    remove(logpath.c_str());
    logger().set_print_to_stdout_flag(false);
    logger().set_sync_flag(true);
    logger().set_filename(logpath);
    logger().set_level(Logger::DEBUG);

    randGen().seed(seed);
    deme_t deme(fs);
    hc_probe hc(op, hcp);
    hc(deme, scorer, max_evals, max_time);

    logger().flush();
    logger().set_level(Logger::NONE);

    printf("\"%s\":{\"seed\":%u,\"arity\":%zu,\"max_evals\":%u,\"null_target\":%s,"
           "\"info_bits\":%.17g,\"nn_estimate_d1\":%zu,"
           "\"hc_crossover\":%s,\"hc_widen_search\":%s,\"hc_max_nn_evals\":%u,"
           "\"hc_crossover_pop_size\":%u,\"hc_crossover_min_neighbors\":%u,"
           "\"terminate_if_gte\":%.9g,\"n_evals\":%zu,\"nsteps\":%u,"
           "\"over_budget\":%s,\"field_shape\":",
           label, seed, arity, max_evals, null_target ? "true" : "false",
           information_theoretic_bits(fs),
           hc.estimate_neighborhood(1, fs), hcp.crossover ? "true" : "false",
           hcp.widen_search ? "true" : "false", (unsigned)hcp.max_nn_evals,
           (unsigned)hcp.crossover_pop_size, (unsigned)hcp.crossover_min_neighbors,
           (double)op.terminate_if_gte, deme.n_evals, hc.nsteps,
           hc.over_budget ? "true" : "false");
    print_field_shape(fs);
    printf(",\"xover_log\":");
    print_xover_log(logpath);
    printf(",\"trajectory\":");
    print_deme(fs, deme);
    printf("}");
    remove(logpath.c_str());
}

// --------------------------------------------------------------------------
// Group 4: local_moses end to end, with crossover at its default (on)
// --------------------------------------------------------------------------

static string rendered(const combo_tree& tr)
{
    stringstream out;
    out << tr;
    return out.str();
}

static string json_string(const string& value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

// %.17g, not %.9g -- see local_moses_probe.cc's own print_score comment for
// the measured reason (a fractional complexity_penalty does not round-trip
// through %.9g).
static void print_score17(const composite_score& cs)
{
    printf("{\"score\":%.17g,\"complexity\":%u,\"complexity_penalty\":%.17g,"
           "\"uniformity_penalty\":%.17g,\"penalized_score\":%.17g}",
           (double)cs.get_score(), (unsigned)cs.get_complexity(),
           (double)cs.get_complexity_penalty(), (double)cs.get_uniformity_penalty(),
           (double)cs.get_penalized_score());
}

// Byte-for-byte the same scenario as tools/cpp_probe/local_moses_probe.cc
// (same dataset, complexity coef, seed-42 exemplar "$1", signature, reduction
// effort, perm_ratio and five repeated local_moses calls), with ONE change:
// hc_parameters' crossover flag is `cross`, so passing true gives classic's
// real default configuration.  See that probe's header for why the exemplar
// is "$1" and not "and()".
static void run_local_moses_case(const char* label, bool cross, const string& logdir)
{
    Table table = loadTable("tests/fixtures/table_data/trainingBoolean.dat", "y");
    CTable ctable = table.compressed();

    ctruth_table_bscore bsc(ctable);
    bsc.set_complexity_coef((score_t)3.5);
    behave_cscore cscorer(bsc);

    combo_tree exemplar = str2combo_tree("$1", vector<string>());

    metapop_parameters mpp;
    subsample_deme_filter_parameters subp;
    metapopulation mp(exemplar, cscorer, mpp, subp);

    type_tree sig = gen_signature(id::boolean_type, 3);

    operator_set empty_ignore_ops;
    logical_reduction reduction2(empty_ignore_ops);
    const rule& r2 = reduction2(2);

    deme_parameters dparams;
    dparams.perm_ratio = -1.0f;
    subsample_deme_filter_parameters fparams;

    hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/cross);
    hill_climbing hc((optim_parameters()), hcp);

    deme_expander dex(sig, r2, r2, cscorer, hc, dparams, fparams);

    string logpath = logdir + "/crossover_probe_" + label + ".log";
    remove(logpath.c_str());
    logger().set_print_to_stdout_flag(false);
    logger().set_sync_flag(true);
    logger().set_filename(logpath);
    logger().set_level(Logger::DEBUG);

    unsigned seed = 42;
    randGen().seed(seed);

    moses_parameters pa;
    pa.max_evals = 20000;
    pa.max_score = (score_t)1000.0; // unreachable -- max_gens drives this
    moses_statistics stats;

    const int N_GENS = 5;

    printf("\"%s\":{\"seed\":%u,\"hc_crossover\":%s,\"generations\":[", label, seed,
           cross ? "true" : "false");
    int prev_expansions = 0;
    for (int g = 1; g <= N_GENS; ++g) {
        pa.max_gens = g;
        local_moses(mp, dex, pa, stats);

        int n_expansions = (int)stats.n_expansions;
        printf("%s{\"gen\":%d,\"n_expansions\":%d,\"n_evals\":%d,\"advanced\":%s,"
               "\"metapop_size\":%zu,\"best_composite_score\":",
               g > 1 ? "," : "", g, n_expansions, (int)stats.n_evals,
               (n_expansions == prev_expansions + 1) ? "true" : "false", mp.size());
        print_score17(mp.best_composite_score());
        printf(",\"best_tree\":%s}", json_string(rendered(mp.best_tree())).c_str());
        prev_expansions = n_expansions;
    }
    logger().flush();
    logger().set_level(Logger::NONE);
    printf("],\"xover_log\":");
    print_xover_log(logpath);
    printf("}");
    remove(logpath.c_str());
}

int main(int argc, char** argv)
{
    string logdir = (argc > 1) ? argv[1] : "/tmp";

    printf("{\"partial_sort\":");
    run_partial_sort_cases();
    printf(",\"direct\":");
    run_xcases();
    printf(",\"hill_climbing\":{");

    // ---- large_nbh_at_distance_2{,_immediate}: a small field_set whose
    // DISTANCE-1 neighborhood estimate is far below crossover_min_neighbors
    // (400), so `large_nbh` is false at distance 1 and the run must plateau
    // and arm `last_chance` (hill-climbing.cc:450) before crossover can fire
    // at all -- but (see the MEASURED note below) by the time last_chance
    // actually opens the gate, distance has already reached 2, where
    // `large_nbh` -- not `last_chance` -- is what admits it. Renamed from
    // "last_chance"/"last_chance_immediate": those names implied
    // `last_chance` was the deciding conjunct, which classic's own log
    // ("n'bhood too large for exhaustive srch: 703>=400") contradicts; see
    // "last_chance_only" below for the one case where that name would be
    // accurate. terminate_if_gte is pushed above the scorer's maximum (0)
    // so the run does not stop the moment it reaches the optimum -- it has
    // to plateau, which is what arms last_chance in the first place.
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        run_hc_case("large_nbh_at_distance_2", 6, 1, op, hcp, 3000, 86400, logdir);
    }
    // Same gate path reached on iteration 3 instead of ~26, by making the
    // all-zero init_inst the optimum (see run_hc_case's null_target note).
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        printf(",");
        run_hc_case("large_nbh_at_distance_2_immediate", 6, 1, op, hcp, 3000, 86400, logdir, true);
    }
    // The ONLY configuration in which `last_chance` is the conjunct that
    // actually decides the gate -- hence keeping that name for THIS case.
    // MEASURED, and sharper than docs/CROSSOVER-INVENTORY.md's "small
    // problems reach crossover only via last_chance": `distance` is
    // incremented at hill-climbing.cc:367-368 BEFORE the last_chance block
    // at :434 runs, so a last_chance crossover always happens at
    // distance >= 2, where estimate_neighborhood returns
    // safe_binomial_coefficient(nn, distance) -- which for a Boolean field
    // set is already >= 400 at distance 2 once nn >= 29 (C(29,2) = 406).  So
    // "large_nbh false AND last_chance true" needs a field set with fewer
    // than ~29 information-theoretic bits: arity 4 => 16 knobs => 25 bits,
    // C(25,2) = 300 < 400.  (arity 6 => 38 bits => C(38,2) = 703 >= 400, so
    // the two "large_nbh_at_distance_2*" cases above reach crossover through
    // large_nbh at distance 2 even though they are armed by last_chance --
    // exactly the reason the C++ logs "n'bhood too large" for them, and
    // exactly why they are NOT named "last_chance*" any more.)
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        printf(",");
        run_hc_case("last_chance_only", 4, 1, op, hcp, 3000, 86400, logdir, true);
    }
    // Control: identical to "large_nbh_at_distance_2_immediate" but with
    // crossover off. Isolates exactly what the crossover code changes.
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = false;
        printf(",");
        run_hc_case("no_crossover", 6, 1, op, hcp, 3000, 86400, logdir, true);
    }
    // ---- large_nbh path, straddling the 400-neighbour boundary exactly.
    // make_representation(arity) yields FOUR multiplicity-3 disc knobs per
    // argument (measured: arity 6 -> 24 knobs, arity 30 -> 120), and a
    // multiplicity-3 knob carries log2(3) = 1.584963 bits, so
    // information_theoretic_bits == 4 * arity * 1.584963 and
    // estimate_neighborhood(1) == 2 * (size_t)that (hill-climbing.cc:504-505).
    //   arity 31 -> 124 knobs -> 196 bits -> 392  (just UNDER 400: FALSE)
    //   arity 32 -> 128 knobs -> 202 bits -> 404  (just OVER  400: TRUE)
    // Measured in the fixture as info_bits/nn_estimate_d1, so the boundary is
    // pinned by measurement rather than by this comment.  max_nn_evals is
    // held down to 40 so each iteration still only produces
    // MINIMUM_DEME_SIZE(100) instances and the deme stays small.
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        hcp.max_nn_evals = 40;
        printf(",");
        run_hc_case("large_nbh_below", 31, 3, op, hcp, 400, 86400, logdir);
    }
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        hcp.max_nn_evals = 40;
        printf(",");
        run_hc_case("large_nbh_above", 32, 3, op, hcp, 400, 86400, logdir);
    }
    // widen_search on top of the large_nbh path: after a crossover iteration
    // that does not improve, `distance` is NOT incremented (hill-climbing.cc:
    // 367-368) and the run does not terminate (widen_search suppresses the
    // "No improvement" exit at :477), so the NEXT iteration hits the gate
    // with already_xover still true -- the "already done from that center"
    // branch (:177), which no other case here reaches.
    {
        optim_parameters op;
        op.terminate_if_gte = 1.0;
        hc_parameters hcp;
        hcp.crossover = true;
        hcp.widen_search = true;
        hcp.max_nn_evals = 40;
        printf(",");
        run_hc_case("large_nbh_widen", 32, 5, op, hcp, 700, 86400, logdir, true);
    }
    printf("},\"local_moses\":{");
    // Classic's real default (--hc-crossover=1) ...
    run_local_moses_case("crossover_on", true, logdir);
    // ... and S11's crossover-disabled configuration, re-measured here so
    // the two can be diffed directly and the effect of crossover on the
    // end-to-end trajectory is visible rather than asserted.
    printf(",");
    run_local_moses_case("crossover_off", false, logdir);
    printf("}}\n");
    return 0;
}
