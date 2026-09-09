// tools/cpp_probe/eda_initialization_probe.cc
//
// Oracle for pyMeTTa's port of moses/moses/eda/initialization.{h,cc}
// (pymetta/moses/eda/initialization.py): uniform_randomize_bit:80,
// uniform_randomize_disc:88, occam_randomize_contin (single-field overload,
// initialization.cc:30, and the all-fields overload, initialization.cc:69),
// and randomize:96 (the combining wrapper). occam_randomize_term is NOT
// probed: term fields do not exist anywhere in this port (field_set.py
// excludes them entirely), so that call's iteration range is always empty
// and there is nothing to compare -- see initialization.py's own module
// docstring.
//
// All four functions draw from the process-wide opencog::randGen()
// singleton (no RNG parameter passed explicitly below, matching how a real
// caller in this port's own code resolves the same singleton via
// pymetta.util.rng.randGen()) -- determinism comes from reseeding
// immediately before each case. Every case brackets the call with a
// pre/post randbool() marker (seeded, then re-seeded before the real call)
// to catch an extra or missing draw independent of the resulting value --
// same convention as contin_neighborhood_probe.cc.
//
// Build (see tools/capture_eda_initialization.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/eda_initialization_probe \
//       tools/cpp_probe/eda_initialization_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and CARRY-FORWARD.md.

#include <moses/moses/eda/initialization.h>
#include <moses/moses/moses/types.h>
#include <moses/moses/representation/field_set.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
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

// ---------------- uniform_randomize_bit ----------------

static void dump_uniform_randomize_bit()
{
    printf("  \"uniform_randomize_bit\": [\n");
    bool first = true;
    for (size_t n_bits : {0u, 1u, 3u, 8u}) {
        for (unsigned seed : {0u, 1u, 42u}) {
            vector<field_set::spec> specs(n_bits, field_set::disc_spec(2));
            field_set fs(specs.begin(), specs.end());
            instance inst(fs.packed_width());

            randGen().seed(seed);
            bool pre = randGen().randbool();
            randGen().seed(seed);
            (void)pre;

            uniform_randomize_bit(fs, inst, randGen());

            bool post = randGen().randbool();

            if (!first) printf(",\n");
            first = false;
            printf("    {\"n_bits\": %lu, \"seed\": %u, \"raw\": ",
                   (unsigned long)n_bits, seed);
            print_raw_array(fs, inst);
            printf(", \"post_randbool\": %s}", post ? "true" : "false");
        }
    }
    printf("\n  ],\n");
}

// ---------------- uniform_randomize_disc ----------------

static void dump_uniform_randomize_disc()
{
    printf("  \"uniform_randomize_disc\": [\n");
    bool first = true;
    vector<vector<unsigned>> multy_lists = {
        {}, {3}, {2, 5}, {4, 3, 7},
    };
    for (auto& multys : multy_lists) {
        for (unsigned seed : {0u, 1u, 42u}) {
            vector<field_set::spec> specs;
            for (unsigned m : multys)
                specs.push_back(field_set::disc_spec(m));
            field_set fs(specs.begin(), specs.end());
            instance inst(fs.packed_width());

            randGen().seed(seed);
            bool pre = randGen().randbool();
            randGen().seed(seed);
            (void)pre;

            uniform_randomize_disc(fs, inst, randGen());

            bool post = randGen().randbool();

            if (!first) printf(",\n");
            first = false;
            printf("    {\"multys\": [");
            for (size_t i = 0; i < multys.size(); ++i) {
                if (i) printf(",");
                printf("%u", multys[i]);
            }
            printf("], \"seed\": %u, \"raw\": ", seed);
            print_raw_array(fs, inst);
            printf(", \"post_randbool\": %s}", post ? "true" : "false");
        }
    }
    printf("\n  ],\n");
}

// ---------------- occam_randomize_contin (single field) ----------------

static void dump_occam_randomize_contin_single()
{
    printf("  \"occam_randomize_contin_single\": [\n");
    bool first = true;
    for (unsigned depth : {0u, 1u, 3u, 6u}) {
        for (unsigned seed : {0u, 1u, 42u, 99u}) {
            // Trailing disc(3) so the contin field is never the very last
            // raw position (see contin_neighborhood_probe.cc's own note
            // on why: avoids a genuine C++ UB read at the raw-array
            // boundary that generate_contin_neighbor's callee can hit).
            vector<field_set::spec> specs;
            specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, depth));
            specs.push_back(field_set::disc_spec(3));
            field_set fs(specs.begin(), specs.end());
            instance inst(fs.packed_width());

            randGen().seed(seed);
            bool pre = randGen().randbool();
            randGen().seed(seed);
            (void)pre;

            field_set::contin_iterator it = fs.begin_contin(inst);
            occam_randomize_contin(fs, inst, it, randGen());

            bool post = randGen().randbool();

            if (!first) printf(",\n");
            first = false;
            printf("    {\"depth\": %u, \"seed\": %u, \"raw\": ", depth, seed);
            print_raw_array(fs, inst);
            printf(", \"post_randbool\": %s}", post ? "true" : "false");
        }
    }
    printf("\n  ],\n");
}

// ---------------- occam_randomize_contin (all fields overload) ----------------

static void dump_occam_randomize_contin_all()
{
    printf("  \"occam_randomize_contin_all\": [\n");
    bool first = true;
    vector<vector<unsigned>> depth_lists = {
        {}, {4}, {3, 5}, {2, 4, 6},
    };
    for (auto& depths : depth_lists) {
        for (unsigned seed : {0u, 1u, 42u}) {
            vector<field_set::spec> specs;
            for (unsigned d : depths)
                specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, d));
            specs.push_back(field_set::disc_spec(3)); // raw-boundary guard
            field_set fs(specs.begin(), specs.end());
            instance inst(fs.packed_width());

            randGen().seed(seed);
            bool pre = randGen().randbool();
            randGen().seed(seed);
            (void)pre;

            occam_randomize_contin(fs, inst, randGen());

            bool post = randGen().randbool();

            if (!first) printf(",\n");
            first = false;
            printf("    {\"depths\": [");
            for (size_t i = 0; i < depths.size(); ++i) {
                if (i) printf(",");
                printf("%u", depths[i]);
            }
            printf("], \"seed\": %u, \"raw\": ", seed);
            print_raw_array(fs, inst);
            printf(", \"post_randbool\": %s}", post ? "true" : "false");
        }
    }
    printf("\n  ],\n");
}

// ---------------- randomize (combining wrapper) ----------------
//
// occam_randomize_term draws nothing observable for a term-free field_set
// (its own iteration range is empty), so a field_set with NO term fields
// (every field_set constructible in this port) isolates exactly the
// contin -> disc -> bit draw order that occam_randomize_contin,
// uniform_randomize_disc and uniform_randomize_bit (in that order) are
// documented to produce.

static void dump_randomize()
{
    printf("  \"randomize\": [\n");
    bool first = true;
    struct Case { vector<unsigned> contin_depths; vector<unsigned> disc_multys; unsigned n_bits; };
    vector<Case> cases = {
        {{}, {}, 0},
        {{3}, {}, 0},
        {{}, {2}, 0},
        {{}, {}, 4},
        {{3, 5}, {2, 4}, 3},
        {{4}, {3, 5, 2}, 6},
    };
    for (auto& c : cases) {
        for (unsigned seed : {0u, 1u, 42u}) {
            vector<field_set::spec> specs;
            for (unsigned d : c.contin_depths)
                specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, d));
            for (unsigned m : c.disc_multys)
                specs.push_back(field_set::disc_spec(m));
            for (unsigned b = 0; b < c.n_bits; ++b)
                specs.push_back(field_set::disc_spec(2));
            field_set fs(specs.begin(), specs.end());
            instance inst(fs.packed_width());

            randGen().seed(seed);
            bool pre = randGen().randbool();
            randGen().seed(seed);
            (void)pre;

            randomize(fs, inst, randGen());

            bool post = randGen().randbool();

            if (!first) printf(",\n");
            first = false;
            printf("    {\"contin_depths\": [");
            for (size_t i = 0; i < c.contin_depths.size(); ++i) {
                if (i) printf(",");
                printf("%u", c.contin_depths[i]);
            }
            printf("], \"disc_multys\": [");
            for (size_t i = 0; i < c.disc_multys.size(); ++i) {
                if (i) printf(",");
                printf("%u", c.disc_multys[i]);
            }
            printf("], \"n_bits\": %u, \"seed\": %u, \"raw\": ", c.n_bits, seed);
            print_raw_array(fs, inst);
            printf(", \"post_randbool\": %s}", post ? "true" : "false");
        }
    }
    printf("\n  ]\n");
}

int main()
{
    printf("{\n");
    dump_uniform_randomize_bit();
    dump_uniform_randomize_disc();
    dump_occam_randomize_contin_single();
    dump_occam_randomize_contin_all();
    dump_randomize();
    printf("}\n");
    return 0;
}
