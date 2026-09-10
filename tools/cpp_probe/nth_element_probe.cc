// tools/cpp_probe/nth_element_probe.cc
//
// Oracle for `OMP_ALGO::nth_element` -- the one line of
// `moses/moses/eda/replacement.h:54` (`replace_the_worst::operator()`) whose
// output permutation IS observed by the rest of MOSES.
//
// Two things are measured here, not assumed:
//
//  1. WHICH `nth_element` `OMP_ALGO::` names in the build classic actually
//     produces. `OMP_ALGO` is `__gnu_parallel` when cogutil's `OC_OMP` is
//     defined and `std` otherwise (`opencog/util/oc_omp.h:53-60`), and
//     `OC_OMP` is defined only `#elif HAVE_PARALLEL_STL` (oc_omp.h:49) --
//     an identifier nothing in either checkout ever `#define`s, so the
//     preprocessor evaluates it as 0. Section "omp_algo_is_std" reports
//     the compiled-in answer rather than relying on that reading, and
//     section "omp_matches_std" runs the SAME cases through
//     `OMP_ALGO::nth_element` and `std::nth_element` and reports whether
//     the two permutations agree element for element.
//
//  2. The exact permutation libstdc++'s `std::nth_element` leaves behind.
//     `nth_element` only PARTITIONS: `[first, nth)` all compare <= `*nth`
//     and `(nth, last)` all compare >= it, with the order WITHIN each part
//     unspecified by the standard. `replace_the_worst` overwrites
//     `[first, nth)` on the very next line, so that half is unobservable --
//     but the tail is NOT: it stays in the deme, and `sort_demes`'
//     `std::sort` is introsort, which is not stable, so the tail's order
//     decides which of several equal-scoring candidates survives
//     `keep_top_unique_candidates`' ADJACENT-duplicate scan. Hence this
//     fixture pins the whole array.
//
// Elements are `pair<int,int>`: `.first` is the sort key (deliberately full
// of TIES -- ties are the only place the permutation is observable) and
// `.second` is a unique tag that makes the permutation readable. The
// comparator compares `.first` only, so the code path exercised is
// `std::__introselect(..., __iter_comp_iter(comp))`, the same one
// `replace_the_worst`'s comparator-less call reaches via
// `__iter_less_iter()`.
//
// Build (see tools/capture_nth_element.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   g++ -O2 -std=gnu++17 -fopenmp -I$P/include \
//       -o tools/cpp_probe/build/nth_element_probe \
//       tools/cpp_probe/nth_element_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcogutil
//
// `-fopenmp` matches classic's own CXX_FLAGS
// (build/moses/moses/CMakeFiles/moses.dir/flags.make), so `OMP_ALGO` here
// resolves exactly as it does in the library under test.

#include <opencog/util/oc_omp.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using Elem = std::pair<int, int>;

bool by_key(const Elem& a, const Elem& b) { return a.first < b.first; }

// One case: a name, the key pattern, and the `nth` offset.
struct Case {
  std::string name;
  std::vector<int> keys;
  size_t nth;
};

std::vector<Elem> tagged(const std::vector<int>& keys) {
  std::vector<Elem> out;
  out.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) out.push_back(Elem(keys[i], int(i)));
  return out;
}

// Deterministic pseudo-random key stream, independent of any library RNG so
// the fixture depends on nothing but this file. A plain LCG.
struct Lcg {
  unsigned long long s;
  explicit Lcg(unsigned long long seed) : s(seed) {}
  unsigned next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return unsigned(s >> 33);
  }
};

std::vector<int> random_keys(size_t n, int distinct, unsigned long long seed) {
  Lcg g(seed);
  std::vector<int> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = int(g.next() % unsigned(distinct));
  return out;
}

std::vector<int> iota_keys(size_t n) {
  std::vector<int> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = int(i);
  return out;
}

std::vector<int> reversed_keys(size_t n) {
  std::vector<int> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = int(n - i);
  return out;
}

std::vector<int> organ_pipe_keys(size_t n) {
  std::vector<int> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = int(i < n / 2 ? i : n - i);
  return out;
}

void print_elems(const std::vector<Elem>& v) {
  printf("[");
  for (size_t i = 0; i < v.size(); ++i)
    printf(i ? ",[%d,%d]" : "[%d,%d]", v[i].first, v[i].second);
  printf("]");
}

std::vector<Case> build_cases() {
  std::vector<Case> cases;
  // Degenerate and boundary sizes. `nth == size` is C++'s documented
  // early-return (`if (first == last || nth == last) return;`).
  const size_t tiny[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17};
  for (size_t n : tiny)
    for (size_t k = 0; k <= n; ++k) {
      char buf[64];
      snprintf(buf, sizeof buf, "tiny_n%zu_k%zu", n, k);
      cases.push_back(Case{buf, random_keys(n, 3, 1000 + n * 17 + k), k});
    }

  // Realistic shapes: the `-aun` deme is 597 with `n_generate == 298`, and
  // scores collapse onto very few distinct values.
  const size_t sizes[] = {20, 33, 64, 100, 298, 597};
  const int distincts[] = {1, 2, 3, 7, 50};
  for (size_t n : sizes)
    for (int d : distincts) {
      char buf[64];
      snprintf(buf, sizeof buf, "rand_n%zu_d%d", n, d);
      cases.push_back(Case{buf, random_keys(n, d, 7 + n * 131 + unsigned(d)), n / 2});
    }

  // Adversarial-ish orderings, at the real deme size.
  cases.push_back(Case{"sorted_n597", iota_keys(597), 298});
  cases.push_back(Case{"reversed_n597", reversed_keys(597), 298});
  cases.push_back(Case{"organ_pipe_n597", organ_pipe_keys(597), 298});
  cases.push_back(Case{"all_equal_n597", random_keys(597, 1, 5), 298});
  // `nth` at both ends of a big range.
  cases.push_back(Case{"rand_n597_k0", random_keys(597, 4, 99), 0});
  cases.push_back(Case{"rand_n597_klast", random_keys(597, 4, 99), 596});
  cases.push_back(Case{"rand_n597_kend", random_keys(597, 4, 99), 597});
  return cases;
}

}  // namespace

int main() {
  printf("{\n");

#ifdef OC_OMP
  const bool omp_algo_is_std = false;
#else
  const bool omp_algo_is_std = true;
#endif
  printf("  \"omp_algo_is_std\": %s,\n", omp_algo_is_std ? "true" : "false");

  const std::vector<Case> cases = build_cases();

  bool omp_matches_std = true;
  for (const Case& c : cases) {
    std::vector<Elem> a = tagged(c.keys);
    std::vector<Elem> b = a;
    OMP_ALGO::nth_element(a.begin(), a.begin() + long(c.nth), a.end(), by_key);
    std::nth_element(b.begin(), b.begin() + long(c.nth), b.end(), by_key);
    if (a != b) omp_matches_std = false;
  }
  printf("  \"omp_matches_std\": %s,\n", omp_matches_std ? "true" : "false");

  printf("  \"cases\": [\n");
  bool first = true;
  for (const Case& c : cases) {
    if (!first) printf(",\n");
    first = false;
    std::vector<Elem> v = tagged(c.keys);
    OMP_ALGO::nth_element(v.begin(), v.begin() + long(c.nth), v.end(), by_key);
    printf("    {\"name\": \"%s\", \"nth\": %zu, \"input\": ", c.name.c_str(), c.nth);
    print_elems(tagged(c.keys));
    printf(", \"output\": ");
    print_elems(v);
    printf("}");
  }
  printf("\n  ]\n}\n");
  return 0;
}
