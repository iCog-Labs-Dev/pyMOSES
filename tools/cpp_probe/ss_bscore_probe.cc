// tools/cpp_probe/ss_bscore_probe.cc
//
// Oracle for pyMeTTa's port of moses/moses/scoring/ss_bscore.{h,cc}
// (pymetta/moses/scoring/ss_bscore.py). ss_bscore's own class cannot be
// exercised end to end in this port (its constructor needs
// CTable::get_timestamps(), which does not exist -- see ss_bscore.py's own
// module docstring), so this probe targets the THREE self-contained,
// RNG-drawing computations ss_bscore::operator() actually performs,
// independent of any CTable:
//
//   - "accumulator": boost::accumulators::accumulator_set<float,
//     stats<count, mean, variance>> fed a sequence of float samples one at
//     a time (ss_bscore.cc:57-63's `sc_var_acc`), dumping mean/variance
//     after each sample -- no RNG, but float32 arithmetic order matters
//     (tag::variance's default is the ITERATIVE accumulator, NOT the lazy
//     moment-based one -- see ss_bscore.py's own docstring for the
//     recurrence this pins).
//   - "low_dev_penalty": `-low_dev_pressure * sqrtf(sc_var)` (ss_bscore.cc:
//     122), float32 throughout -- no RNG.
//   - "partition_by_row": the shuffle-then-split branch (ss_bscore.cc:
//     90-100), `std::random_shuffle(first, last, gen)` with
//     `gen(i) = randGen().randint(i)` -- RNG-drawing, draw order matters.
//   - "partition_by_time": the `rand_element_erase`-then-round-robin branch
//     (ss_bscore.cc:66-76), driven here over a std::set<unsigned> standing
//     in for std::set<TTable::value_type> (both are ordered sets; the
//     algorithm only depends on `<` and `size()`, never on the concrete
//     value type) -- RNG-drawing, draw order matters.
//
// Build (see tools/capture_ss_bscore.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/ss_bscore_probe \
//       tools/cpp_probe/ss_bscore_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and CARRY-FORWARD.md.

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <opencog/util/mt19937ar.h>
#include <opencog/util/random.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace opencog;
using namespace std;
using namespace boost::accumulators;

// ---------------- accumulator (count, mean, variance) ----------------

static void dump_accumulator()
{
    typedef accumulator_set<float, stats<tag::count, tag::mean, tag::variance>> stat_acc_t;

    vector<vector<float>> sample_sets = {
        {1.0f},
        {1.0f, 2.0f},
        {1.0f, 2.0f, 3.0f},
        {5.0f, 5.0f, 5.0f, 5.0f},
        {0.0f, 10.0f, -3.5f, 7.25f, 2.0f},
        {-1.0f, -2.0f, -3.0f},
    };

    printf("  \"accumulator\": [\n");
    bool first_case = true;
    for (auto& samples : sample_sets) {
        stat_acc_t acc;
        printf("%s    {\"samples\": [", first_case ? "" : ",\n");
        first_case = false;
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i) printf(",");
            printf("%.9g", (double)samples[i]);
        }
        printf("], \"after_each\": [");
        for (size_t i = 0; i < samples.size(); ++i) {
            acc(samples[i]);
            if (i) printf(",");
            float mean_val = boost::accumulators::mean(acc);
            float var_val = boost::accumulators::variance(acc);
            printf("{\"mean\": %.9g, \"variance\": %.9g}", (double)mean_val, (double)var_val);
        }
        printf("]}");
    }
    printf("\n  ],\n");
}

// ---------------- low_dev_penalty ----------------

static void dump_low_dev_penalty()
{
    printf("  \"low_dev_penalty\": [\n");
    bool first = true;
    vector<float> variances = {0.0f, 1.0f, 4.0f, 2.5f, 100.0f, 0.0001f};
    vector<float> pressures = {0.0f, 0.5f, 1.0f, 2.0f};
    for (float sc_var : variances) {
        for (float pressure : pressures) {
            float low_dev_penalty = -pressure * sqrtf(sc_var);
            if (!first) printf(",\n");
            first = false;
            printf("    {\"sc_var\": %.9g, \"low_dev_pressure\": %.9g, \"low_dev_penalty\": %.9g}",
                   (double)sc_var, (double)pressure, (double)low_dev_penalty);
        }
    }
    printf("\n  ],\n");
}

// ---------------- partition_by_row ----------------

static void dump_partition_by_row()
{
    printf("  \"partition_by_row\": [\n");
    bool first = true;
    struct Case { unsigned usize; unsigned n_subsamples; unsigned seed; };
    vector<Case> cases = {
        {10, 2, 0}, {10, 2, 1}, {10, 3, 42}, {7, 3, 7},
        {20, 4, 99}, {5, 5, 3}, {1, 1, 0}, {6, 4, 5},
    };
    for (auto& c : cases) {
        vector<unsigned> row_idxs(c.usize);
        for (unsigned i = 0; i < c.usize; ++i) row_idxs[i] = i;

        randGen().seed(c.seed);
        auto shr = [&](ptrdiff_t i) { return randGen().randint(i); };
        random_shuffle(row_idxs.begin(), row_idxs.end(), shr);

        unsigned seg_size = c.n_subsamples ? c.usize / c.n_subsamples : 0;
        vector<vector<unsigned>> groups;
        for (auto it = row_idxs.cbegin();
             seg_size > 0 && it != row_idxs.cbegin() + (ptrdiff_t)(c.n_subsamples * seg_size);
             it += seg_size)
            groups.emplace_back(it, it + seg_size);

        if (!first) printf(",\n");
        first = false;
        printf("    {\"usize\": %u, \"n_subsamples\": %u, \"seed\": %u, \"shuffled\": [",
               c.usize, c.n_subsamples, c.seed);
        for (size_t i = 0; i < row_idxs.size(); ++i) {
            if (i) printf(",");
            printf("%u", row_idxs[i]);
        }
        printf("], \"groups\": [");
        for (size_t g = 0; g < groups.size(); ++g) {
            if (g) printf(",");
            printf("[");
            for (size_t i = 0; i < groups[g].size(); ++i) {
                if (i) printf(",");
                printf("%u", groups[g][i]);
            }
            printf("]");
        }
        printf("]}");
    }
    printf("\n  ],\n");
}

// ---------------- partition_by_time (rand_element_erase round robin) ----------------

static void dump_partition_by_time()
{
    printf("  \"partition_by_time\": [\n");
    bool first = true;
    struct Case { vector<unsigned> timestamps; unsigned n_subsamples; unsigned seed; };
    vector<Case> cases = {
        {{100, 200, 300, 400, 500}, 2, 0},
        {{100, 200, 300, 400, 500}, 2, 1},
        {{10, 20, 30}, 3, 42},
        {{1, 2, 3, 4, 5, 6, 7}, 3, 7},
        {{5}, 1, 0},
        {{}, 2, 0},
        {{9, 1, 5, 3, 7}, 2, 99}, // inserted out of sorted order; std::set sorts anyway
    };
    for (auto& c : cases) {
        set<unsigned> timestamps(c.timestamps.begin(), c.timestamps.end());
        unsigned tt_size = timestamps.size();
        vector<set<unsigned>> groups(c.n_subsamples);

        randGen().seed(c.seed);
        for (unsigned i = 0; i < tt_size; ++i) {
            unsigned time = rand_element_erase(timestamps);
            groups[i % c.n_subsamples].insert(time);
        }

        if (!first) printf(",\n");
        first = false;
        printf("    {\"timestamps\": [");
        // dump the SORTED input (matches std::set's own construction order)
        set<unsigned> sorted_in(c.timestamps.begin(), c.timestamps.end());
        bool ti = true;
        for (unsigned t : sorted_in) { printf(ti ? "%u" : ",%u", t); ti = false; }
        printf("], \"n_subsamples\": %u, \"seed\": %u, \"groups\": [", c.n_subsamples, c.seed);
        for (size_t g = 0; g < groups.size(); ++g) {
            if (g) printf(",");
            printf("[");
            bool gi = true;
            for (unsigned t : groups[g]) { printf(gi ? "%u" : ",%u", t); gi = false; }
            printf("]");
        }
        printf("]}");
    }
    printf("\n  ]\n");
}

int main()
{
    printf("{\n");
    dump_accumulator();
    dump_low_dev_penalty();
    dump_partition_by_row();
    dump_partition_by_time();
    printf("}\n");
    return 0;
}
