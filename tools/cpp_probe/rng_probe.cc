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
#include <opencog/util/mt19937ar.h>

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
        printf("]\n    }");
    }
    printf("\n  }\n}\n");
    return 0;
}
