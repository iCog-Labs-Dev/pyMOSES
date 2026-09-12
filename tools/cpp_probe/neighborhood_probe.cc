// tools/cpp_probe/neighborhood_probe.cc
//
// Oracle for pyMoses S9 Task 1 (moses/moses/moses/neighborhood_sampling.{h,cc}).
// Dumps, as JSON:
//   - "deterministic": count_neighborhood_size (both overloads) and
//     generate_all_in_neighborhood (both overloads) over small disc-only
//     field_sets at several (multiplicity, num_knobs, distance) points,
//     including a distance-0 base case and a distance-exceeds-dimension
//     empty-result case.
//   - "safe_binomial_coefficient": direct calls to the real exported
//     function at a handful of (k, n) points.
//   - "generate_initial_sample": RNG-drawn initial demes for a fixed seed,
//     over an all-bit, an all-disc, and a mixed bit+disc field_set, so both
//     the uniform_randomize_bit/disc AND the bias loops in
//     generate_initial_sample are exercised in both branches.
//   - "sample_from_neighborhood": RNG-drawn neighbor samples (WITH
//     replacement, exact draw order matters) over the same mixed field_set,
//     at several seeds/distances/sample sizes.
//
// generate_initial_sample and sample_from_neighborhood both call the
// process-wide opencog::randGen() singleton directly (no RNG parameter in
// either signature) -- so determinism here comes from reseeding that
// singleton with randGen().seed(seed) immediately before each case, exactly
// mirroring what a Python oracle test does to pymoses.util.rng.randGen().
//
// Build (see tools/capture_neighborhood.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/neighborhood_probe \
//       tools/cpp_probe/neighborhood_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and the task brief on generate_all_in_neighborhood's
// combinatorial blowup at large distance/knob counts. Every case below is
// deliberately kept small (num_knobs <= 8, distance <= 5, sample sizes
// <= 30).

#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/moses/types.h>
#include <moses/moses/representation/field_set.h>
#include <moses/moses/representation/instance_set.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

static void print_raw_array(const field_set& fs, const instance& inst)
{
    printf("[");
    for (size_t i = 0; i < fs.raw_size(); ++i) {
        if (i) printf(",");
        printf("%lu", (unsigned long)fs.get_raw(inst, i));
    }
    printf("]");
}

static void print_int_vector(const vector<unsigned>& v)
{
    printf("[");
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) printf(",");
        printf("%u", v[i]);
    }
    printf("]");
}

static void print_int_vector(const vector<int>& v)
{
    printf("[");
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) printf(",");
        printf("%d", v[i]);
    }
    printf("]");
}

static field_set make_uniform_fs(unsigned multiplicity, size_t num_knobs)
{
    return field_set(field_set::disc_spec(multiplicity), num_knobs);
}

static field_set make_mixed_fs(const vector<unsigned>& multiplicities)
{
    vector<field_set::spec> specs;
    for (unsigned m : multiplicities)
        specs.push_back(field_set::disc_spec(m));
    return field_set(specs.begin(), specs.end());
}

int main()
{
    printf("{\n");

    // ---------------- deterministic ----------------
    struct DetCase {
        unsigned multiplicity;
        size_t num_knobs;
        unsigned distance;
        vector<int> center; // empty => use the null-exemplar overloads
    };
    vector<DetCase> det_cases = {
        {2, 3, 0, {}},         // distance == 0 base case
        {2, 3, 1, {}},
        {2, 3, 1, {1, 0, 1}},
        {2, 6, 3, {}},
        {3, 4, 1, {}},
        {3, 4, 2, {1, 2, 0, 1}},
        {4, 3, 2, {}},
        {5, 3, 1, {}},
        {5, 4, 2, {}},
        {2, 8, 1, {}},
        {2, 3, 5, {}},         // distance > dimension => empty result
    };

    printf("  \"deterministic\": [\n");
    bool first = true;
    for (auto& c : det_cases) {
        field_set fs = make_uniform_fs(c.multiplicity, c.num_knobs);
        bool has_center = !c.center.empty();
        instance center(fs.packed_width());
        if (has_center) {
            field_set::disc_iterator itd = fs.begin_disc(center);
            for (int v : c.center) {
                *itd = v;
                ++itd;
            }
        }

        size_t count_with_inst = count_neighborhood_size(fs, center, c.distance);
        size_t count_null = count_neighborhood_size(fs, c.distance);
        size_t deme_size = has_center ? count_with_inst : count_null;

        instance_set<composite_score> deme(deme_size, center, fs);
        if (has_center)
            generate_all_in_neighborhood(fs, c.distance, deme.begin(), deme.end(), center);
        else
            generate_all_in_neighborhood(fs, c.distance, deme.begin(), deme.end());

        if (!first) printf(",\n");
        first = false;
        printf("    {\"multiplicity\": %u, \"num_knobs\": %zu, \"distance\": %u, "
               "\"has_center\": %s, \"center\": ",
               c.multiplicity, c.num_knobs, c.distance, has_center ? "true" : "false");
        print_int_vector(c.center);
        printf(", \"count\": %zu, \"count_null\": %zu, \"neighbors\": [", count_with_inst,
               count_null);
        bool firstn = true;
        for (const instance& inst : deme) {
            if (!firstn) printf(",");
            firstn = false;
            print_raw_array(fs, inst);
        }
        printf("]}\n");
    }
    printf("  ],\n");

    // ---------------- safe_binomial_coefficient ----------------
    // safe_binomial_coefficient(k, n) forwards to
    // boost::math::binomial_coefficient<double>(k, n), i.e. boost's own
    // binomial_coefficient(n_arg=k, k_arg=n) -- which requires k_arg <= n_arg,
    // i.e. n <= k here. safe_binomial_coefficient's try/catch only catches
    // boost::numeric::positive_overflow, NOT the std::domain_error boost
    // throws for k > n (measured: k=3, n=4 aborts the process with an
    // uncaught boost::wrapexcept<std::domain_error>). Every real call site
    // (count_neighborhood_size_from_index's bit branch) guards `dist <= rb`
    // before calling, so k < n never happens in practice; every case below
    // respects that same k >= n precondition.
    vector<pair<unsigned, unsigned>> combo_cases = {
        {5, 2}, {6, 3}, {10, 0}, {10, 10}, {7, 4}, {20, 5}, {4, 3}, {0, 0},
    };
    printf("  \"safe_binomial_coefficient\": [\n");
    first = true;
    for (auto& p : combo_cases) {
        size_t r = safe_binomial_coefficient(p.first, p.second);
        if (!first) printf(",\n");
        first = false;
        printf("    {\"k\": %u, \"n\": %u, \"result\": %zu}", p.first, p.second, r);
    }
    printf("\n  ],\n");

    // ---------------- generate_initial_sample ----------------
    struct FsCase {
        string label;
        vector<unsigned> multiplicities;
    };
    vector<FsCase> fs_cases = {
        {"all_bit", {2, 2, 2, 2, 2}},
        {"all_disc", {3, 4, 5}},
        {"mixed", {2, 2, 2, 3, 4, 5}},
    };
    vector<unsigned> seeds = {0, 1, 42, 12345};

    printf("  \"generate_initial_sample\": [\n");
    first = true;
    for (auto& fc : fs_cases) {
        for (unsigned seed : seeds) {
            field_set fs = make_mixed_fs(fc.multiplicities);
            randGen().seed(seed);
            int n = 20;
            instance_set<composite_score> deme(n, fs);
            generate_initial_sample(fs, n, deme.begin(), deme.end());

            if (!first) printf(",\n");
            first = false;
            printf("    {\"label\": \"%s\", \"seed\": %u, \"n\": %d, \"multiplicities\": ",
                   fc.label.c_str(), seed, n);
            print_int_vector(fc.multiplicities);
            printf(", \"instances\": [");
            bool firsti = true;
            for (const instance& inst : deme) {
                if (!firsti) printf(",");
                firsti = false;
                print_raw_array(fs, inst);
            }
            printf("]}\n");
        }
    }
    printf("  ],\n");

    // ---------------- sample_from_neighborhood ----------------
    struct SfnCase {
        unsigned seed;
        unsigned dist;
        unsigned sample_size;
        vector<int> center_raw;
    };
    vector<unsigned> mixed_mults = {2, 2, 2, 3, 4, 5};
    vector<SfnCase> sfn_cases = {
        {0, 1, 15, {3, 2, 1, 1, 0, 1}},
        {0, 2, 20, {3, 2, 1, 1, 0, 1}},
        {1, 3, 25, {0, 0, 0, 0, 0, 0}},
        {42, 2, 30, {4, 3, 2, 1, 1, 0}},
        {12345, 4, 30, {2, 1, 0, 0, 1, 1}},
        {7, 1, 10, {0, 0, 0, 0, 0, 0}},
        {4294967295u, 6, 20, {4, 3, 2, 1, 1, 1}}, // dist == dim_size()
    };

    printf("  \"sample_from_neighborhood\": [\n");
    first = true;
    for (auto& c : sfn_cases) {
        field_set fs = make_mixed_fs(mixed_mults);
        instance center(fs.packed_width());
        for (size_t i = 0; i < c.center_raw.size(); ++i)
            fs.set_raw(center, i, c.center_raw[i]);

        randGen().seed(c.seed);
        instance_set<composite_score> deme(c.sample_size, center, fs);
        sample_from_neighborhood(fs, c.dist, c.sample_size, deme.begin(), deme.end(), center);

        if (!first) printf(",\n");
        first = false;
        printf("    {\"seed\": %u, \"dist\": %u, \"sample_size\": %u, \"multiplicities\": ",
               c.seed, c.dist, c.sample_size);
        print_int_vector(mixed_mults);
        printf(", \"center_raw\": ");
        print_int_vector(c.center_raw);
        printf(", \"instances\": [");
        bool firsti = true;
        for (const instance& inst : deme) {
            if (!firsti) printf(",");
            firsti = false;
            print_raw_array(fs, inst);
        }
        printf("]}\n");
    }
    printf("  ]\n");

    printf("}\n");
    return 0;
}
