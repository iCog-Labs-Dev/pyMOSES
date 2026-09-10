// tools/cpp_probe/univariate_rng_probe.cc
//
// Oracle for the two RNG streams the `-aun` (univariate) optimizer reads that
// nothing else in MOSES reads:
//
//   1. `std::rand()`, via the BARE `std::random_shuffle(nodes.begin(),
//      nodes.end())` at opencog/util/digraph.h:96 inside
//      `randomized_topological_sort`. libstdc++'s two-argument
//      `std::random_shuffle` is specified to use an unspecified source of
//      randomness and implements it as `std::rand() % (i - first + 1)`
//      (libstdc++ `bits/stl_algo.h`, `random_shuffle(_RandomAccessIterator,
//      _RandomAccessIterator)`). `git grep 'srand(' $CLASSIC $COGUTIL` finds
//      ZERO call sites, so on glibc that stream starts from the statically
//      initialised default state, which is exactly the state `srand(1)`
//      produces (glibc `stdlib/random_r.c`: `randtbl` is the TYPE_3 table and
//      `unsafe_state` points into it; `srandom_r` special-cases `seed == 0` to
//      1). Sections "rand_default" and "rand_srand1" MEASURE that claim
//      rather than assuming it: they are captured separately and the test
//      that consumes them asserts they are equal.
//
//   2. `opencog::roulette_select` instantiated with `ScoreT = int` at
//      moses/moses/eda/local_structure.cc:146 -- `dtree_node` is
//      `vector<int>` (local_structure.h:68), so `sum = ScoreT(double(sum) *
//      rng.randdouble())` TRUNCATES TOWARD ZERO and the whole loop runs in
//      integer arithmetic. Section "roulette_int" dumps the real template's
//      answer for a grid of (counts, randdouble) pairs so the port can be
//      compared index for index, and "roulette_double" dumps the SAME grid
//      through the `ScoreT = double` instantiation so the disagreement rate
//      between the two is measurable rather than asserted.
//
// Build (see tools/capture_univariate_rng.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   g++ -O2 -std=gnu++17 -I$CLASSIC -I$P/include \
//       -o tools/cpp_probe/build/univariate_rng_probe \
//       tools/cpp_probe/univariate_rng_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcogutil
//
// Only cogutil is needed: digraph.h and selection.h are header-only and
// MT19937RandGen lives in libcogutil.

#include <opencog/util/digraph.h>
#include <opencog/util/mt19937ar.h>
#include <opencog/util/selection.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace opencog;

namespace {

// The exact grid of leaf-count vectors used by both roulette sections. Each
// entry is a `dtree_node` MINUS its trailing grand total, i.e. what
// `local_structure.cc:146` passes as `[dtr->begin(), --(dtr->end()))`; the
// total is recomputed as the sum, which is what `rec_learn`
// (local_structure.h:312) stores in `back()`.
const std::vector<std::vector<int>> kCounts = {
    {1},          {0},          {5},          {1, 1},       {1, 0},
    {0, 1},       {0, 0},       {3, 7},       {7, 3},       {1, 9},
    {9, 1},       {2, 2},       {10, 10},     {1, 2, 3},    {3, 2, 1},
    {0, 5, 5},    {5, 0, 5},    {5, 5, 0},    {1, 1, 1},    {4, 4, 4, 4},
    {1, 2, 4, 8}, {8, 4, 2, 1}, {0, 0, 0, 1}, {1, 0, 0, 0}, {13, 1, 1, 1, 1},
    {1, 1, 1, 1, 13},
};

// The randdouble values swept against every count vector. Chosen to bracket
// each integer boundary of every `sum` in kCounts: the truncation in
// `ScoreT(double(sum) * rng.randdouble())` only shows up when
// `double(sum) * r` lands strictly between two integers.
std::vector<double> sweep_draws() {
  std::vector<double> out;
  for (int i = 0; i <= 200; ++i) out.push_back(double(i) / 200.0);
  // Values just inside each side of the k/20 boundaries, to catch the
  // exact-integer case (where int() and the float compare agree) versus the
  // just-below case (where they do not).
  for (int k = 1; k < 20; ++k) {
    out.push_back(double(k) / 20.0 - 1e-12);
    out.push_back(double(k) / 20.0 + 1e-12);
  }
  return out;
}

// A RandGen that hands back a scripted randdouble value, so the roulette
// sections isolate roulette_select's own arithmetic from MT19937's. Only
// randdouble() is ever called by roulette_select (selection.h:91). Derived
// from the concrete MT19937RandGen so every other pure virtual of RandGen
// keeps its real implementation.
struct ScriptedRng : public MT19937RandGen {
  ScriptedRng() : MT19937RandGen(0u) {}
  double value = 0.0;
  double randdouble() { return value; }
};

void print_int_vector(const std::vector<int>& v) {
  printf("[");
  for (size_t i = 0; i < v.size(); ++i) printf(i ? ",%d" : "%d", v[i]);
  printf("]");
}

}  // namespace

int main() {
  const int kRandDraws = 512;
  const std::vector<int> kShuffleSizes = {1, 2, 3, 4, 5, 6, 7, 8,
                                          9, 10, 12, 16, 20, 32, 64};
  const int kShuffleReps = 12;

  printf("{\n");

  // ---- section 1: the raw std::rand() stream ------------------------------
  // First kRandDraws values of the DEFAULT (never-srand'd) stream. This is
  // the stream digraph.h:96 consumes.
  printf("  \"rand_default\": [");
  for (int i = 0; i < kRandDraws; ++i) printf(i ? ",%d" : "%d", std::rand());
  printf("],\n");

  // ---- section 2: srand(1) reproduces the default state -------------------
  // Measured, not assumed: glibc's static `randtbl` is documented to be the
  // post-warm-up TYPE_3 state for seed 1.
  std::srand(1);
  printf("  \"rand_srand1\": [");
  for (int i = 0; i < kRandDraws; ++i) printf(i ? ",%d" : "%d", std::rand());
  printf("],\n");

  // ---- section 3: srand(0) is srand(1) ------------------------------------
  // `srandom_r` maps seed 0 to 1. Included because a port that reimplements
  // the generator has to reproduce that special case too.
  std::srand(0);
  printf("  \"rand_srand0\": [");
  for (int i = 0; i < 64; ++i) printf(i ? ",%d" : "%d", std::rand());
  printf("],\n");

  // ---- section 4: a few other seeds ---------------------------------------
  printf("  \"rand_seeded\": {\n");
  const std::vector<unsigned> kSeeds = {2u, 7u, 42u, 12345u, 2147483647u,
                                        4294967295u};
  bool first_seed = true;
  for (unsigned s : kSeeds) {
    if (!first_seed) printf(",\n");
    first_seed = false;
    std::srand(s);
    printf("    \"%u\": [", s);
    for (int i = 0; i < 64; ++i) printf(i ? ",%d" : "%d", std::rand());
    printf("]");
  }
  printf("\n  },\n");

  // ---- section 5: randomized_topological_sort over an EDGELESS graph ------
  // The shape `local_structure_model`'s constructor actually produces for a
  // Boolean-only field_set: `_initial_deps` gets no `insert` calls at all
  // (local_structure.h:257-258 only calls `make_dtree`), so every node is
  // ready immediately and the emitted order IS `std::random_shuffle`'s
  // permutation. Reset to the process-start state first, then call the real
  // function repeatedly, exactly as one `optimize()` run does once per
  // generation.
  printf("  \"topo_sort_edgeless\": {\n");
  bool first_size = true;
  for (int n : kShuffleSizes) {
    if (!first_size) printf(",\n");
    first_size = false;
    // Reset per SIZE, not once for the whole section: each entry is then a
    // self-contained replay from the process-start state, and the 12 reps
    // within it still chain, which is what pins the per-call draw COUNT.
    std::srand(1);
    printf("    \"%d\": [", n);
    for (int rep = 0; rep < kShuffleReps; ++rep) {
      digraph g(n);
      std::vector<digraph::value_type> out;
      randomized_topological_sort(g, std::back_inserter(out));
      if (rep) printf(",");
      printf("[");
      for (size_t i = 0; i < out.size(); ++i) printf(i ? ",%u" : "%u", out[i]);
      printf("]");
    }
    printf("]");
  }
  printf("\n  },\n");

  // ---- section 6: randomized_topological_sort WITH a dependency chain -----
  // The shape a contin field produces: `_initial_deps.insert(base+i-1,
  // base+i)` (local_structure.h:240) builds one chain per contin field, so
  // the shuffle only decides the interleaving of the chain heads.
  printf("  \"topo_sort_chains\": {\n");
  // Each entry: total node count, then chains given as [start, length].
  const std::vector<std::pair<int, std::vector<std::pair<int, int>>>> kChains = {
      {6, {{0, 3}, {3, 3}}},
      {8, {{0, 4}}},
      {9, {{0, 2}, {2, 3}, {5, 4}}},
      {12, {{0, 5}, {6, 5}}},
  };
  bool first_chain = true;
  for (const auto& spec : kChains) {
    if (!first_chain) printf(",\n");
    first_chain = false;
    std::srand(1);  // per-entry reset, as above
    printf("    \"%d", spec.first);
    for (const auto& c : spec.second) printf("_%d.%d", c.first, c.second);
    printf("\": [");
    for (int rep = 0; rep < kShuffleReps; ++rep) {
      digraph g(spec.first);
      for (const auto& c : spec.second)
        for (int i = 1; i < c.second; ++i) g.insert(c.first + i - 1, c.first + i);
      std::vector<digraph::value_type> out;
      randomized_topological_sort(g, std::back_inserter(out));
      if (rep) printf(",");
      printf("[");
      for (size_t i = 0; i < out.size(); ++i) printf(i ? ",%u" : "%u", out[i]);
      printf("]");
    }
    printf("]");
  }
  printf("\n  },\n");

  // ---- section 7/8: roulette_select at ScoreT=int and ScoreT=double -------
  ScriptedRng rng;
  const std::vector<double> draws = sweep_draws();

  printf("  \"roulette_draws\": [");
  for (size_t i = 0; i < draws.size(); ++i)
    printf(i ? ",%.17g" : "%.17g", draws[i]);
  printf("],\n");

  printf("  \"roulette_counts\": [");
  for (size_t i = 0; i < kCounts.size(); ++i) {
    if (i) printf(",");
    print_int_vector(kCounts[i]);
  }
  printf("],\n");

  // ScoreT = int: the REAL instantiation at local_structure.cc:146.
  printf("  \"roulette_int\": [");
  for (size_t c = 0; c < kCounts.size(); ++c) {
    if (c) printf(",");
    printf("[");
    const std::vector<int>& counts = kCounts[c];
    int total = 0;
    for (int x : counts) total += x;
    for (size_t d = 0; d < draws.size(); ++d) {
      rng.value = draws[d];
      std::vector<int>::const_iterator it =
          roulette_select(counts.begin(), counts.end(), total, rng);
      printf(d ? ",%ld" : "%ld", long(it - counts.begin()));
    }
    printf("]");
  }
  printf("],\n");

  // ScoreT = double: what a float-typed port computes instead. Same grid, so
  // the disagreement count is a measurement.
  printf("  \"roulette_double\": [");
  for (size_t c = 0; c < kCounts.size(); ++c) {
    if (c) printf(",");
    printf("[");
    const std::vector<int>& counts = kCounts[c];
    std::vector<double> dcounts(counts.begin(), counts.end());
    double total = 0.0;
    for (double x : dcounts) total += x;
    for (size_t d = 0; d < draws.size(); ++d) {
      rng.value = draws[d];
      std::vector<double>::const_iterator it =
          roulette_select(dcounts.begin(), dcounts.end(), total, rng);
      printf(d ? ",%ld" : "%ld", long(it - dcounts.begin()));
    }
    printf("]");
  }
  printf("]\n");

  printf("}\n");
  return 0;
}
