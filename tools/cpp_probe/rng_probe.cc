// tools/cpp_probe/rng_probe.cc
//
// Dumps real cogutil MT19937RandGen draw sequences as JSON, for
// differential/bit-identity testing against pymetta/util/rng.py.
//
// FIX WAVE (final review, Important): this probe used to re-implement
// MT19937RandGen's algorithm from scratch against a bare <random>
// std::mt19937 (no cogutil include or link at all) rather than calling
// the real cogutil class. That reimplementation was faithful -- RandGen
// IS-A std::mt19937 with no extra state (RandGen.h's own comment: "RandGen
// inherits std::mt19937 ... it's clear we don't need that home made random
// generator structure anymore"), and every MT19937RandGen method this
// probe exercises is a one-line call into an STL distribution/operator()
// applied to `*this` (mt19937ar.cc:44-79) -- so the 36,000-draw fixture
// this produced was, and remains, correct. But a probe built that way
// can never notice a REAL change to cogutil's own code (a different
// distribution, a changed edge case, a future algorithm swap): it would
// keep silently reproducing its own hand-copy forever. This version
// includes and links the genuine opencog::MT19937RandGen instead, so a
// cogutil change shows up here the next time this probe is rebuilt and
// re-run, the same guarantee score_probe.cc/table_probe.cc/tree_probe.cc
// already have for the classic headers/libraries they each depend on.
//
// Build (see tools/capture_golden.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   g++ -O2 -std=c++17 -I$P/include \
//       -o tools/cpp_probe/build/rng_probe tools/cpp_probe/rng_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcogutil
//
// No MOSES/comboreduct headers or libraries are needed -- mt19937ar.h/
// RandGen.h and libcogutil alone are enough to construct MT19937RandGen
// and call its public methods directly.
//
// 2026-09-12: a "randfloat" series was added (pymetta/util/rng.py's
// randfloat() stopped raising NotImplementedError and needed an oracle --
// it is reached from the CLI by --fs-enforce-features-filename, via
// biased_randbool, $COGUTIL/opencog/util/random.h:127). The rebuild that
// captured it also re-emitted the five pre-existing series, and every one
// of them came back byte-for-byte identical to the previously committed
// tests/fixtures/rng.json -- so this probe's link against the real
// cogutil is still producing exactly the values the older fixture
// pinned, and only the new series was added to it.
#include <opencog/util/mt19937ar.h>
#include <opencog/util/random.h>

#include <cstdio>
#include <vector>

using namespace opencog;

int main() {
    const std::vector<unsigned> seeds = {0u, 1u, 7u, 42u, 12345u, 4294967295u};
    const int N = 2000;
    printf("{\n  \"draws\": {\n");
    bool first_seed = true;
    for (unsigned s : seeds) {
        if (!first_seed) printf(",\n");
        first_seed = false;
        printf("    \"%u\": {\n", s);

        // Each field below calls the real MT19937RandGen method it is
        // named for, on its own freshly-seeded generator (matching the
        // fixture's own per-field independence: pymetta/util/rng.py's
        // oracle test draws each field from an independently-seeded
        // generator too, not one shared stream split across fields).

        MT19937RandGen g_raw(s);
        printf("      \"raw\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%u" : "%u", g_raw());
        printf("],\n");

        MT19937RandGen g_ri(s);
        printf("      \"randint\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%d" : "%d", g_ri.randint());
        printf("],\n");

        MT19937RandGen g_rn(s);
        printf("      \"randint_7\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%d" : "%d", g_rn.randint(7));
        printf("],\n");

        MT19937RandGen g_rb(s);
        printf("      \"randbool\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%d" : "%d", g_rb.randbool() ? 1 : 0);
        printf("],\n");

        MT19937RandGen g_rd(s);
        printf("      \"randdouble\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%.17g" : "%.17g", g_rd.randdouble());
        printf("],\n");

        // randfloat() returns a C `float` (binary32); it is printed with
        // %.17g as a double so the fixture carries the EXACT binary32
        // value -- a shorter format would round it and silently loosen the
        // bit-identity claim pymetta/util/rng.py's randfloat() makes.
        MT19937RandGen g_rf(s);
        printf("      \"randfloat\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%.17g" : "%.17g", (double) g_rf.randfloat());
        printf("]\n    }");
    }
    printf("\n  },\n");

    // biased_randbool(float b) -- `return b > rng.randfloat();`
    // ($COGUTIL/opencog/util/random.h:127). The ONLY reader of
    // randfloat() in the MOSES core, reached from the CLI by
    // --fs-enforce-features-filename via
    // feature_selector::sample_enforced_features (feature_selector.cc:455).
    //
    // Every weight below is chosen to make the float32 narrowing DECIDABLE:
    // `b` binds to a `float` parameter, so a `double` weight is rounded
    // BEFORE the comparison. Pairs like (0.4170219898223877,
    // 0.41702198982238775) are the draw itself and the very next double
    // above it -- both narrow to the same float32, so C++ answers the same
    // for both, while a port comparing doubles answers differently.
    // One generator per seed, weights consumed in order: each call draws
    // exactly once, so the k-th weight faces the k-th randfloat draw.
    {
        const std::vector<double> weights = {
            0.0, 1.0, 0.5, 0.25,
            // seed 0's first four randfloat draws, then each one's next
            // double up (same float32).
            0.54881352186203, 0.5488135218620301,
            0.5928446054458618, 0.5928446054458619,
            0.7151893377304077, 0.7151893377304078,
            0.8442657589912415, 0.8442657589912416,
            // seed 1's and seed 42's first draws, same treatment.
            0.4170219898223877, 0.41702198982238775,
            0.3745401203632355, 0.37454012036323553,
            // a few plain decimals whose float32 value differs from the
            // double in the 9th digit.
            0.1, 0.3, 0.7, 0.9,
        };
        printf("  \"biased_randbool\": [");
        bool first_case = true;
        for (unsigned s : seeds) {
            if (!first_case) printf(",");
            first_case = false;
            MT19937RandGen g(s);
            printf("\n    {\"seed\":%u,\"weights\":[", s);
            for (size_t i = 0; i < weights.size(); i++)
                printf(i ? ",%.17g" : "%.17g", weights[i]);
            printf("],\"results\":[");
            for (size_t i = 0; i < weights.size(); i++)
                printf(i ? ",%d" : "%d", biased_randbool((float) weights[i], g) ? 1 : 0);
            printf("]}");
        }
        printf("\n  ],\n");

        // The same weights, but a FRESH generator per weight, so EVERY
        // call faces that seed's FIRST randfloat draw. This is what makes
        // the float32 narrowing killable: for the seed whose first draw
        // appears in the list, the draw itself and the next double above
        // it are two different doubles that narrow to the SAME float32, so
        // C++ answers identically for both -- and a port that compared
        // doubles would not. In the shared-generator group above the pair
        // faces two different draws, so it proves nothing about narrowing.
        printf("  \"biased_randbool_first_draw\": [");
        first_case = true;
        for (unsigned s : seeds) {
            if (!first_case) printf(",");
            first_case = false;
            printf("\n    {\"seed\":%u,\"weights\":[", s);
            for (size_t i = 0; i < weights.size(); i++)
                printf(i ? ",%.17g" : "%.17g", weights[i]);
            printf("],\"results\":[");
            for (size_t i = 0; i < weights.size(); i++) {
                MT19937RandGen g(s);
                printf(i ? ",%d" : "%d", biased_randbool((float) weights[i], g) ? 1 : 0);
            }
            printf("]}");
        }
        printf("\n  ]\n}\n");
    }
    return 0;
}
