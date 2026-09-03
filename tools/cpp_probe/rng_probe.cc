// tools/cpp_probe/rng_probe.cc
// Dumps cogutil RandGen draw sequences as JSON for differential testing.
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const std::vector<unsigned> seeds = {0u, 1u, 7u, 42u, 12345u, 4294967295u};
    const int N = 2000;
    printf("{\n  \"draws\": {\n");
    bool first_seed = true;
    for (unsigned s : seeds) {
        if (!first_seed) printf(",\n");
        first_seed = false;
        printf("    \"%u\": {\n", s);

        std::mt19937 g_raw(s);
        printf("      \"raw\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%u" : "%u", g_raw());
        printf("],\n");

        std::mt19937 g_ri(s);
        printf("      \"randint\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", d(g_ri));
        }
        printf("],\n");

        std::mt19937 g_rn(s);
        printf("      \"randint_7\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", d(g_rn) % 7);
        }
        printf("],\n");

        std::mt19937 g_rb(s);
        printf("      \"randbool\": [");
        for (int i = 0; i < N; ++i) {
            std::uniform_int_distribution<int> d;
            printf(i ? ",%d" : "%d", (d(g_rb) % 2 == 0) ? 1 : 0);
        }
        printf("],\n");

        std::mt19937 g_rd(s);
        printf("      \"randdouble\": [");
        for (int i = 0; i < N; ++i) printf(i ? ",%.17g" : "%.17g", g_rd() / (double)g_rd.max());
        printf("]\n    }");
    }
    printf("\n  }\n}\n");
    return 0;
}
