// tools/cpp_probe/subsample_selector_probe.cc
//
// Oracle for the P2 metapopulation subsample task's own acceptance
// criterion: "Verify draw *order*, not just the resulting set" for
// `lazy_random_selector` as `subsample.cc` actually drives it --
// `ss_score_dev_filter`/`ss_average_agreement`/`ss_tanimoto_stats`
// (subsample.cc:180-186,257-263,393-399) all share the identical shape:
//
//   lazy_random_selector selector(n_possible_tuples);
//   for (i_tuple = 0; i_tuple < sample_tuples; ++i_tuple) {
//       unsigned rnd_tuple = selector();
//       for (i_deme = 0; i_deme < n_demes; ++i_deme) {
//           tuple_idxs.push_back(rnd_tuple % base);
//           rnd_tuple /= base;
//       }
//   }
//
// `pymoses/util/lazy_random_selector.py`'s own Floyd's-algorithm
// bookkeeping is already covered by hand-traced fixed-candidate tests
// (tests/util/test_lazy_random_selector.py), and `MT19937RandGen.randint_n`
// is already covered by rng_probe.cc/rng.json (12000 draws / 6 seeds,
// module docstring) -- what neither of those exercises is THIS specific
// composition: real `opencog::MT19937RandGen` draws, through the real
// `opencog::lazy_random_selector::select()` (which calls `rng.randint(_u)`,
// i.e. the SAME overload rng_probe.cc's own "randint_n"-equivalent field
// exercises), decomposed via the exact modulo/divide loop above -- for
// configurations small enough to force real collisions (the `_l`-advance
// path), which no existing fixture drives through a REAL RandGen (only
// `test_lazy_random_selector.py`'s own hand-scripted `_FixedSelector`
// does, with a synthetic candidate sequence, not real RNG draws).
//
// Build (see tools/capture_subsample_selector.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   g++ -O2 -std=c++17 -I$P/include \
//       -o tools/cpp_probe/build/subsample_selector_probe \
//       tools/cpp_probe/subsample_selector_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcogutil
//
// Only cogutil (mt19937ar.h, lazy_random_selector.h) is needed -- no
// MOSES/comboreduct headers or libraries -- since the loop shape above is
// reproduced directly here rather than calling the real (private,
// unexported) metapopulation methods.
#include <opencog/util/mt19937ar.h>
#include <opencog/util/lazy_random_selector.h>

#include <cstdio>
#include <vector>

using namespace opencog;

struct Config {
    const char* name;
    unsigned seed;
    unsigned n_possible_tuples;
    unsigned sample_tuples;
    unsigned n_demes;
    unsigned base;
};

int main() {
    // Every config draws sample_tuples <= n_possible_tuples times, matching
    // subsample.cc's own `sample_tuples = std::min(n_possible_tuples,
    // n_tuples)` invariant -- the selector is never asked to over-draw.
    const std::vector<Config> configs = {
        // Exhaustive draws (sample_tuples == n_possible_tuples): forces
        // every one of Floyd's-algorithm's collisions for a small range.
        {"exhaustive_n4_demes1_base4", 42u, 4u, 4u, 1u, 4u},
        {"exhaustive_n64_demes3_base4", 42u, 64u, 64u, 3u, 4u},
        {"exhaustive_n9_demes2_base3", 7u, 9u, 9u, 2u, 3u},
        // Partial draws: realistic n_tuples-clamped shape.
        {"partial_n64_demes3_base4", 1234u, 64u, 20u, 3u, 4u},
        {"partial_n1000_demes2_base10", 0u, 1000u, 50u, 2u, 10u},
        // n_demes == 0 (edge: ss_demes.size() == 0 -> tuple_idxs always
        // empty regardless of the raw draw -- base ** 0 == 1).
        {"edge_demes0", 99u, 1u, 1u, 0u, 1u},
        // base == 1 (n_top_candidates == 1, subsample_deme_filter_
        // parameters's own default): every tuple_idx is always 0.
        {"base1_demes5", 5u, 1u, 1u, 5u, 1u},
    };

    printf("{\n  \"configs\": {\n");
    bool first = true;
    for (const Config& cfg : configs) {
        if (!first) printf(",\n");
        first = false;
        printf("    \"%s\": {\n", cfg.name);
        printf("      \"seed\": %u,\n", cfg.seed);
        printf("      \"n_possible_tuples\": %u,\n", cfg.n_possible_tuples);
        printf("      \"sample_tuples\": %u,\n", cfg.sample_tuples);
        printf("      \"n_demes\": %u,\n", cfg.n_demes);
        printf("      \"base\": %u,\n", cfg.base);

        MT19937RandGen rng(cfg.seed);
        lazy_random_selector selector(cfg.n_possible_tuples, rng);

        printf("      \"draws\": [");
        std::vector<unsigned> draws;
        for (unsigned i = 0; i < cfg.sample_tuples; ++i) {
            unsigned d = selector();
            draws.push_back(d);
            printf(i ? ",%u" : "%u", d);
        }
        printf("],\n");

        printf("      \"tuple_idxs\": [");
        for (unsigned i = 0; i < cfg.sample_tuples; ++i) {
            unsigned rnd_tuple = draws[i];
            if (i) printf(",");
            printf("[");
            for (unsigned d = 0; d < cfg.n_demes; ++d) {
                unsigned idx = rnd_tuple % cfg.base;
                rnd_tuple /= cfg.base;
                printf(d ? ",%u" : "%u", idx);
            }
            printf("]");
        }
        printf("]\n    }");
    }
    printf("\n  }\n}\n");
    return 0;
}
