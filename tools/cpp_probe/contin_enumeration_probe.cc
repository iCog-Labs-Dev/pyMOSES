// tools/cpp_probe/contin_enumeration_probe.cc
//
// Oracle for the last three continuous-domain gaps in
// pymoses/moses/moses/neighborhood_sampling.py:
//
//   1. count_neighborhood_size_from_index's contin branch
//      ($CLASSIC/moses/moses/moses/neighborhood_sampling.cc:117-160)
//   2. vary_n_knobs' contin branch
//      ($CLASSIC/moses/moses/moses/neighborhood_sampling.h:289-349)
//   3. randomize()'s occam_randomize_contin call
//      ($CLASSIC/moses/moses/eda/initialization.cc:96) -- covered here only
//      as the ORDER check (contin, then disc, then bit); the per-function
//      draws already have their own fixture (contin_neighborhood.json /
//      eda_initialization.json).
//
// The point of this probe is the ARITHMETIC, not just the shape. Sections:
//
//   "binomial_double" -- boost::math::binomial_coefficient<double>(n, k)
//     itself, dumped as an exact hex float PLUS the `(unsigned)` C-style
//     cast neighborhood_sampling.cc:141/149 actually applies to it. That
//     cast is a 32-bit truncation, NOT std::min(..., SIZE_MAX) the way
//     safe_binomial_coefficient (cc:87) clamps, and NOT Python's exact
//     math.comb. This section is the direct oracle for the port's own
//     double-emulation helper.
//
//   "count" -- count_neighborhood_size_from_index over field_sets that
//     include real contin_specs, varying (center content, dist,
//     starting_index, max_count). Reaches, deliberately:
//       * length == 0 (all Stop), 0 < length < depth, length == depth
//       * `i - length` negative AND positive (cc:138's std::max(0, ...))
//       * `depth - length` == 0 AND > 0 (cc:138's std::min(i, ...))
//       * the `if (i <= length)` second inner loop both taken and skipped
//       * pow2(j) for j up to 31
//       * a depth-35, length-35 field where
//         (unsigned)binomial_coefficient<double>(35, 17) wraps mod 2^32
//         (exact C(35,17) = 4537567650 > UINT_MAX) and where `cni` itself
//         (an `unsigned`, cc:133) wraps while accumulating
//       * the `number_of_instances > max_count` early return (a strict >)
//
//   "count_throws" -- inputs where classic's own OC_ASSERT fires, recorded
//     as a thrown-exception marker rather than a number. pow2<unsigned>
//     (cogutil numeric.h:266) asserts `32 > x`, so a depth >= 32 contin
//     with length 0 and dist >= 32 aborts classic. Recorded so the port
//     can reproduce the failure rather than silently returning a number.
//
//   "vary" -- the FULL vary_n_knobs output sequence, element for element in
//     emission order (not set-wise), over field_sets with contin fields.
//     Reaches the `*itr == Stop` arm and the Left/Right arm; `remRLs <=
//     dist` true and false; and -- via a deliberately MALFORMED contin
//     field ([Left, Stop, Right], i.e. a non-Stop pseudo-bit after a Stop)
//     entered at a starting_index past the first Stop -- the case where
//     `size_t length - size_t relative_raw_idx` UNDERFLOWS and is then
//     narrowed to `unsigned remRLs` (h:337). vary_n_knobs takes an
//     arbitrary starting_index and an arbitrary center instance, so this is
//     reachable through its own public signature.
//
//   "randomize_order" -- eda/initialization.cc:96 randomize() over a
//     field_set with contin, disc and bit fields, at fixed seeds: the whole
//     resulting raw array. Any reordering of its four calls desyncs the RNG
//     stream, so this pins the order, not just the values.
//
// Build (see tools/capture_contin_enumeration.py, which runs exactly this):
//
//   P=$PYMOSES_MOSES_PREFIX
//   C=$PYMOSES_CLASSIC_DIR
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/contin_enumeration_probe \
//       tools/cpp_probe/contin_enumeration_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and docs/CARRY-FORWARD.md. Every case below is sized so the
// vary_n_knobs output stays in the low thousands of instances and the
// count_neighborhood_size recursion stays shallow.

#include <moses/moses/eda/initialization.h>
#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/moses/types.h>
#include <moses/moses/representation/field_set.h>
#include <opencog/util/mt19937ar.h>

#include <boost/math/special_functions/binomial.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

// ---------------------------------------------------------------- helpers

// A JSON-friendly description of a field_set: a list of ["contin", depth]
// / ["disc", multy] pairs, in the order handed to the field_set(It, It)
// constructor.  field_set SORTS its specs internally, so the resulting raw
// layout is dumped alongside every case and re-asserted on the Python side
// rather than assumed.
struct SpecItem {
  bool is_contin;
  int a; // depth for contin, multiplicity for disc
};

static field_set make_fs(const vector<SpecItem> &items) {
  vector<field_set::spec> specs;
  for (const SpecItem &s : items) {
    if (s.is_contin)
      specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, s.a));
    else
      specs.push_back(field_set::disc_spec(s.a));
  }
  return field_set(specs.begin(), specs.end());
}

static void print_spec_items(const vector<SpecItem> &items) {
  printf("[");
  for (size_t i = 0; i < items.size(); ++i) {
    if (i)
      printf(",");
    printf("[\"%s\",%d]", items[i].is_contin ? "contin" : "disc", items[i].a);
  }
  printf("]");
}

// The resolved raw layout, so the Python side can assert its own field_set
// landed on the same one before comparing any number.
static void print_layout(const field_set &fs) {
  printf("{\"raw_size\": %zu, \"begin_contin\": %zu, \"end_contin\": %zu, "
         "\"begin_disc\": %zu, \"end_disc\": %zu, \"begin_bit\": %zu, "
         "\"end_bit\": %zu, \"contin_depths\": [",
         fs.raw_size(), fs.begin_contin_raw_idx(), fs.end_contin_raw_idx(),
         fs.begin_disc_raw_idx(), fs.end_disc_raw_idx(),
         fs.begin_bit_raw_idx(), fs.end_bit_raw_idx());
  for (size_t i = 0; i < fs.contin().size(); ++i) {
    if (i)
      printf(",");
    printf("%d", fs.contin()[i].depth);
  }
  printf("]}");
}

static void print_raw_array(const field_set &fs, const instance &inst) {
  printf("[");
  for (size_t i = 0; i < fs.raw_size(); ++i) {
    if (i)
      printf(",");
    printf("%lu", (unsigned long)fs.get_raw(inst, i));
  }
  printf("]");
}

static void print_int_vector(const vector<int> &v) {
  printf("[");
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      printf(",");
    printf("%d", v[i]);
  }
  printf("]");
}

static instance make_center(const field_set &fs, const vector<int> &raw) {
  instance inst(fs.packed_width());
  for (size_t i = 0; i < raw.size(); ++i)
    fs.set_raw(inst, i, (disc_t)raw[i]);
  return inst;
}

// ------------------------------------------------------- binomial_double

// Exactly the expression neighborhood_sampling.cc:141 and :149 evaluate:
// a double-valued binomial coefficient put through a C-style cast to
// `unsigned`.  Kept in its own noinline function so -O2 cannot constant
// fold it into something the real code would not do.
__attribute__((noinline)) static unsigned trunc_binom(unsigned n, unsigned k) {
  return (unsigned)boost::math::binomial_coefficient<double>(n, k);
}

static void dump_binomial_double() {
  struct NK {
    unsigned n, k;
  };
  vector<NK> cases;
  // Dense small grid: every (n, k) with k <= n <= 12.
  for (unsigned n = 0; n <= 12; ++n)
    for (unsigned k = 0; k <= n; ++k)
      cases.push_back({n, k});
  // Around and past the 2^32 boundary of the `(unsigned)` cast.
  // C(34,17) = 2333606220 < 2^32 <= C(35,17) = 4537567650.
  const NK big[] = {{30, 15}, {32, 16}, {33, 16}, {34, 16}, {34, 17},
                    {35, 17}, {35, 18}, {36, 18}, {38, 19}, {40, 20},
                    {45, 22}, {50, 25}, {52, 26}, {55, 27}, {60, 30},
                    {62, 31}, {64, 32}, {66, 33}, {70, 35}, {80, 40},
                    // Past 2^53 (the exact-integer range of a double).
                    {100, 50}, {120, 60}, {150, 75}, {170, 85},
                    // n > max_factorial<double>::value (170): boost
                    // switches from the factorial table to the beta
                    // function here.
                    {171, 2},  {171, 3},  {171, 85}, {175, 4},
                    {200, 5},  {200, 100}};
  for (const NK &c : big)
    cases.push_back(c);

  printf("  \"binomial_double\": [\n");
  bool first = true;
  for (const NK &c : cases) {
    double d = boost::math::binomial_coefficient<double>(c.n, c.k);
    unsigned t = trunc_binom(c.n, c.k);
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"n\": %u, \"k\": %u, \"hex\": \"%a\", \"repr\": \"%.17g\", "
           "\"trunc_unsigned\": %u}",
           c.n, c.k, d, d, t);
  }
  printf("\n  ],\n");
}

// ------------------------------------------------------------------ count

struct CountCase {
  const char *label;
  vector<SpecItem> specs;
  vector<int> center; // raw values, index 0 upward; missing tail is 0
  unsigned dist;
  unsigned starting_index;
  // max_count as an offset from "effectively unbounded": use_max_count
  // false means classic's own default (SIZE_MAX).
  bool use_max_count;
  size_t max_count;
};

static vector<int> repeat_then(int head_val, int head_n, int tail_val,
                              int tail_n) {
  vector<int> v;
  for (int i = 0; i < head_n; ++i)
    v.push_back(head_val);
  for (int i = 0; i < tail_n; ++i)
    v.push_back(tail_val);
  return v;
}

static void dump_count() {
  // A: one small contin (depth 3), one disc(3), two bits.
  const vector<SpecItem> A = {{true, 3}, {false, 3}, {false, 2}, {false, 2}};
  // B: one contin (depth 5), one disc(3), one bit.
  const vector<SpecItem> B = {{true, 5}, {false, 3}, {false, 2}};
  // C: two contins of different depth, plus disc and bits.
  const vector<SpecItem> C = {
      {true, 2}, {true, 4}, {false, 3}, {false, 2}, {false, 2}};
  // D: one deep contin (depth 35) plus four bits.  Depth 35 is what makes
  // C(35, 17) = 4537567650 exceed UINT_MAX, so the `(unsigned)` cast at
  // cc:141 actually truncates.
  const vector<SpecItem> D = {
      {true, 35}, {false, 2}, {false, 2}, {false, 2}, {false, 2}};

  const vector<int> A_stop = {0, 0, 0};
  const vector<int> A_len1 = {1, 0, 0};
  const vector<int> A_len2 = {1, 2, 0};
  const vector<int> A_full = {1, 2, 1};

  const vector<int> B_stop = {0, 0, 0, 0, 0};
  const vector<int> B_len2 = {2, 1, 0, 0, 0};
  const vector<int> B_len3 = {1, 1, 2, 0, 0};
  const vector<int> B_full = {2, 2, 1, 1, 2};

  const vector<int> C_mixed = {1, 0, /* contin2 */ 1, 2, 0, 0};

  const vector<int> D_full = repeat_then(1, 35, 0, 0);
  const vector<int> D_len20 = repeat_then(2, 20, 0, 15);
  const vector<int> D_stop = repeat_then(0, 35, 0, 0);

  vector<CountCase> cases;
  // --- A: all four contin contents, every distance 0..5, from index 0.
  for (unsigned d = 0; d <= 5; ++d) {
    cases.push_back({"A_stop", A, A_stop, d, 0, false, 0});
    cases.push_back({"A_len1", A, A_len1, d, 0, false, 0});
    cases.push_back({"A_len2", A, A_len2, d, 0, false, 0});
    cases.push_back({"A_full", A, A_full, d, 0, false, 0});
  }
  // --- A: starting_index past the contin field (disc/bit branches, with a
  // contin present in the field_set) -- a regression guard that the contin
  // branch's index range test does not swallow them.
  for (unsigned d = 0; d <= 3; ++d) {
    cases.push_back({"A_from_disc", A, A_full, d, 3, false, 0});
    cases.push_back({"A_from_bit", A, A_full, d, 4, false, 0});
  }
  // --- B: depth 5, four contin contents, distances 0..6.
  for (unsigned d = 0; d <= 6; ++d) {
    cases.push_back({"B_stop", B, B_stop, d, 0, false, 0});
    cases.push_back({"B_len2", B, B_len2, d, 0, false, 0});
    cases.push_back({"B_len3", B, B_len3, d, 0, false, 0});
    cases.push_back({"B_full", B, B_full, d, 0, false, 0});
  }
  // --- C: two contin fields -- exercises the recursive call landing on a
  // SECOND contin field (starting_index + depth), not on disc/bit.
  for (unsigned d = 0; d <= 5; ++d)
    cases.push_back({"C_mixed", C, C_mixed, d, 0, false, 0});
  // --- max_count early return (a strict `>`), on a case whose unbounded
  // count is well above every threshold below.
  for (size_t mc : {(size_t)0, (size_t)1, (size_t)2, (size_t)5, (size_t)10,
                    (size_t)50, (size_t)1000}) {
    cases.push_back({"B_full_maxcount", B, B_full, 4, 0, true, mc});
    cases.push_back({"C_mixed_maxcount", C, C_mixed, 4, 0, true, mc});
  }
  // --- D: the 32-bit truncation of the double binomial, and cni's own
  // `unsigned` accumulation wrap.  length == depth == 35 forces
  // depth - length == 0, so j is pinned to 0 and the first inner loop is
  // exactly (unsigned)C(35, i).
  for (unsigned d : {13u, 14u, 15u, 16u, 17u, 18u, 20u})
    cases.push_back({"D_full", D, D_full, d, 0, false, 0});
  // --- D with length 20 < depth 35: depth - length == 15, so pow2(j) runs
  // up to 2^15 and both inner loops contribute.
  for (unsigned d : {4u, 10u, 17u, 20u, 25u})
    cases.push_back({"D_len20", D, D_len20, d, 0, false, 0});
  // --- D all Stop: length == 0, so j == i and pow2(j) reaches 2^31 at
  // dist 31 -- one below cogutil's own pow2<unsigned> assertion bound.
  for (unsigned d : {4u, 20u, 30u, 31u})
    cases.push_back({"D_stop", D, D_stop, d, 0, false, 0});

  printf("  \"count\": [\n");
  bool first = true;
  for (const CountCase &c : cases) {
    field_set fs = make_fs(c.specs);
    instance center = make_center(fs, c.center);
    size_t result =
        c.use_max_count
            ? count_neighborhood_size_from_index(fs, center, c.dist,
                                                 c.starting_index, c.max_count)
            : count_neighborhood_size_from_index(fs, center, c.dist,
                                                 c.starting_index);
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"label\": \"%s\", \"specs\": ", c.label);
    print_spec_items(c.specs);
    printf(", \"layout\": ");
    print_layout(fs);
    printf(", \"center\": ");
    print_int_vector(c.center);
    printf(", \"dist\": %u, \"starting_index\": %u, \"use_max_count\": %s, "
           "\"max_count\": %zu, \"result\": %zu}",
           c.dist, c.starting_index, c.use_max_count ? "true" : "false",
           c.max_count, result);
  }
  printf("\n  ],\n");
}

// ------------------------------------------------------------ count_throws

static void dump_count_throws() {
  // pow2<unsigned>(x) (cogutil util/numeric.h:266) OC_ASSERTs 32 > x.  In
  // the contin branch j <= std::min(i, depth - length) and i <= std::min
  // (dist, depth), so length == 0 with depth >= 32 and dist >= 32 puts
  // j == 32 into pow2 and classic throws.
  struct ThrowCase {
    const char *label;
    vector<SpecItem> specs;
    vector<int> center;
    unsigned dist;
    unsigned starting_index;
  };
  const vector<SpecItem> D = {
      {true, 35}, {false, 2}, {false, 2}, {false, 2}, {false, 2}};
  vector<ThrowCase> cases = {
      {"D_stop_dist32", D, repeat_then(0, 35, 0, 0), 32, 0},
      {"D_stop_dist35", D, repeat_then(0, 35, 0, 0), 35, 0},
  };

  printf("  \"count_throws\": [\n");
  bool first = true;
  for (const ThrowCase &c : cases) {
    field_set fs = make_fs(c.specs);
    instance center = make_center(fs, c.center);
    bool threw = false;
    size_t result = 0;
    try {
      result = count_neighborhood_size_from_index(fs, center, c.dist,
                                                  c.starting_index);
    } catch (...) {
      threw = true;
    }
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"label\": \"%s\", \"specs\": ", c.label);
    print_spec_items(c.specs);
    printf(", \"center\": ");
    print_int_vector(c.center);
    printf(", \"dist\": %u, \"starting_index\": %u, \"threw\": %s, "
           "\"result\": %zu}",
           c.dist, c.starting_index, threw ? "true" : "false", result);
  }
  printf("\n  ],\n");
}

// ------------------------------------------------------------------- vary

struct VaryCase {
  const char *label;
  vector<SpecItem> specs;
  vector<int> center;
  unsigned dist;
  unsigned starting_index;
};

static void dump_vary() {
  const vector<SpecItem> A = {{true, 3}, {false, 3}, {false, 2}, {false, 2}};
  const vector<SpecItem> B = {{true, 5}, {false, 3}, {false, 2}};
  const vector<SpecItem> C = {
      {true, 2}, {true, 4}, {false, 3}, {false, 2}, {false, 2}};
  // Contin only -- no disc/bit tail at all, so the recursion runs straight
  // off the end of the raw array from inside the contin branch.
  const vector<SpecItem> E = {{true, 3}};

  vector<VaryCase> cases;

  // --- A, Stop arm at index 0 (whole contin field is Stop).
  for (unsigned d = 0; d <= 3; ++d)
    cases.push_back({"A_stop", A, {0, 0, 0, 0, 0, 0}, d, 0});
  // --- A, Left/Right arm, length 1: remRLs == 1, so `remRLs <= dist` is
  // FALSE at dist 0 (unreachable, dist 0 returns early) and TRUE from 1 up.
  for (unsigned d = 1; d <= 3; ++d)
    cases.push_back({"A_len1", A, {1, 0, 0, 1, 1, 0}, d, 0});
  // --- A, full-length contin: remRLs == 3, so `remRLs <= dist` is FALSE
  // at dist 1 and 2, TRUE at dist 3.
  for (unsigned d = 1; d <= 3; ++d)
    cases.push_back({"A_full", A, {1, 2, 1, 2, 0, 1}, d, 0});
  // --- A, starting_index INSIDE the contin field: relative_raw_idx > 0.
  cases.push_back({"A_full_mid1", A, {1, 2, 1, 2, 0, 1}, 1, 1});
  cases.push_back({"A_full_mid1", A, {1, 2, 1, 2, 0, 1}, 2, 1});
  cases.push_back({"A_full_mid2", A, {1, 2, 1, 2, 0, 1}, 1, 2});
  cases.push_back({"A_full_mid2", A, {1, 2, 1, 2, 0, 1}, 2, 2});
  // --- A, starting_index past the contin field entirely.
  cases.push_back({"A_from_disc", A, {1, 2, 1, 2, 0, 1}, 1, 3});
  cases.push_back({"A_from_disc", A, {1, 2, 1, 2, 0, 1}, 2, 3});

  // --- B (depth 5): more room between length and depth.
  for (unsigned d = 1; d <= 3; ++d) {
    cases.push_back({"B_stop", B, {0, 0, 0, 0, 0, 0, 0}, d, 0});
    cases.push_back({"B_len2", B, {2, 1, 0, 0, 0, 1, 1}, d, 0});
    cases.push_back({"B_full", B, {2, 2, 1, 1, 2, 1, 0}, d, 0});
  }
  cases.push_back({"B_len2_mid", B, {2, 1, 0, 0, 0, 1, 1}, 2, 1});
  cases.push_back({"B_full_mid3", B, {2, 2, 1, 1, 2, 1, 0}, 2, 3});
  cases.push_back({"B_full_mid4", B, {2, 2, 1, 1, 2, 1, 0}, 2, 4});

  // --- C: two contin fields, so `next_contin` lands on another contin.
  for (unsigned d = 1; d <= 2; ++d) {
    cases.push_back({"C_stop", C, {0, 0, 0, 0, 0, 0, 0, 0, 0}, d, 0});
    cases.push_back({"C_mixed", C, {1, 0, 1, 2, 0, 0, 1, 0, 1}, d, 0});
    cases.push_back({"C_full", C, {1, 2, 2, 1, 1, 2, 2, 1, 0}, d, 0});
  }
  cases.push_back({"C_mixed_mid", C, {1, 0, 1, 2, 0, 0, 1, 0, 1}, 2, 2});

  // --- E: contin only, no trailing fields.
  for (unsigned d = 1; d <= 3; ++d) {
    cases.push_back({"E_stop", E, {0, 0, 0}, d, 0});
    cases.push_back({"E_full", E, {2, 1, 2}, d, 0});
  }

  // --- MALFORMED contin fields: a non-Stop pseudo-bit AFTER a Stop, so
  // contin_length() (which returns the index of the FIRST Stop) is smaller
  // than a starting_index that still points at a Left/Right.  This makes
  // h:337's `unsigned remRLs = length - relative_raw_idx` a genuine
  // unsigned underflow: size_t wraps to ~2^64, then narrows to ~2^32, so
  // `remRLs <= dist` is FALSE and the whole Stop-fill block is skipped.
  // Clamping the subtraction to 0 instead would make it TRUE and emit
  // extra instances.
  cases.push_back({"A_malformed_LSR", A, {1, 0, 2, 1, 1, 1}, 1, 2});
  cases.push_back({"A_malformed_LSR", A, {1, 0, 2, 1, 1, 1}, 2, 2});
  cases.push_back({"A_malformed_LSR", A, {1, 0, 2, 1, 1, 1}, 3, 2});
  cases.push_back({"A_malformed_SSR", A, {0, 0, 2, 1, 1, 1}, 2, 2});
  // A wider gap: length 1, relative_raw_idx 4 -> remRLs narrows to
  // 0xFFFFFFFD.
  cases.push_back({"B_malformed_far", B, {1, 0, 0, 0, 2, 1, 1}, 2, 4});
  cases.push_back({"B_malformed_far", B, {1, 0, 0, 0, 2, 1, 1}, 3, 4});
  // Malformed in the SECOND contin field of a two-contin field_set.
  cases.push_back({"C_malformed", C, {1, 2, 1, 0, 0, 2, 1, 0, 1}, 2, 5});

  const size_t CAP = 200000;

  printf("  \"vary\": [\n");
  bool first = true;
  for (const VaryCase &c : cases) {
    field_set fs = make_fs(c.specs);
    instance center = make_center(fs, c.center);
    vector<instance> buf(CAP, instance(fs.packed_width()));
    vector<instance>::iterator it =
        vary_n_knobs(fs, center, c.dist, c.starting_index, buf.begin(),
                     buf.end());
    size_t n = (size_t)(it - buf.begin());
    if (n >= CAP) {
      fprintf(stderr, "case %s dist=%u start=%u filled the buffer (%zu)\n",
              c.label, c.dist, c.starting_index, n);
      exit(1);
    }
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"label\": \"%s\", \"specs\": ", c.label);
    print_spec_items(c.specs);
    printf(", \"layout\": ");
    print_layout(fs);
    printf(", \"center\": ");
    print_int_vector(c.center);
    printf(", \"dist\": %u, \"starting_index\": %u, \"n\": %zu, "
           "\"instances\": [",
           c.dist, c.starting_index, n);
    for (size_t i = 0; i < n; ++i) {
      if (i)
        printf(",");
      print_raw_array(fs, buf[i]);
    }
    printf("]}");
  }
  printf("\n  ],\n");
}

// -------------------------------------------------------- randomize_order

static void dump_randomize_order() {
  // eda/initialization.cc:96 randomize() -- occam_randomize_term (a no-op
  // here: no term fields exist), occam_randomize_contin, then
  // uniform_randomize_disc, then uniform_randomize_bit.  Contin fields of
  // different depths, plus disc and bit fields, so a reordering of the
  // three real calls changes every draw from the first contin onward.
  struct RCase {
    const char *label;
    vector<SpecItem> specs;
    int seed;
  };
  const vector<SpecItem> A = {{true, 3}, {false, 3}, {false, 2}, {false, 2}};
  const vector<SpecItem> C = {
      {true, 2}, {true, 4}, {false, 3}, {false, 4}, {false, 2}, {false, 2}};
  const vector<SpecItem> NOCONTIN = {{false, 3}, {false, 4}, {false, 2}};

  vector<RCase> cases;
  for (int seed : {0, 1, 2, 7, 42}) {
    cases.push_back({"A", A, seed});
    cases.push_back({"C", C, seed});
    cases.push_back({"nocontin", NOCONTIN, seed});
  }

  printf("  \"randomize_order\": [\n");
  bool first = true;
  for (const RCase &c : cases) {
    field_set fs = make_fs(c.specs);
    instance inst(fs.packed_width());
    randGen().seed(c.seed);
    randomize(fs, inst);
    // One extra draw AFTER the call, so a port that draws a different
    // NUMBER of values (not merely different ones) is caught even when the
    // instance happens to match.
    int post = randGen().randint(1000000);
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"label\": \"%s\", \"specs\": ", c.label);
    print_spec_items(c.specs);
    printf(", \"layout\": ");
    print_layout(fs);
    printf(", \"seed\": %d, \"raw\": ", c.seed);
    print_raw_array(fs, inst);
    printf(", \"post_randint\": %d}", post);
  }
  printf("\n  ],\n");
}

// ------------------------------------------ generate_initial_sample(contin)

static void dump_generate_initial_sample() {
  // neighborhood_sampling.h:52 generate_initial_sample -- randomize() plus
  // its own two bias loops, over a field_set WITH contin fields.  This is
  // the real caller that _randomize()'s NotImplementedError was blocking.
  struct GCase {
    const char *label;
    vector<SpecItem> specs;
    int seed;
    int n;
  };
  const vector<SpecItem> A = {{true, 3}, {false, 3}, {false, 2}, {false, 2}};
  const vector<SpecItem> C = {
      {true, 2}, {true, 4}, {false, 3}, {false, 4}, {false, 2}, {false, 2}};

  vector<GCase> cases;
  for (int seed : {0, 1, 5}) {
    cases.push_back({"A", A, seed, 4});
    cases.push_back({"C", C, seed, 4});
  }

  printf("  \"generate_initial_sample\": [\n");
  bool first = true;
  for (const GCase &c : cases) {
    field_set fs = make_fs(c.specs);
    vector<instance> buf(c.n, instance(fs.packed_width()));
    randGen().seed(c.seed);
    generate_initial_sample(fs, c.n, buf.begin(), buf.end());
    int post = randGen().randint(1000000);
    if (!first)
      printf(",\n");
    first = false;
    printf("    {\"label\": \"%s\", \"specs\": ", c.label);
    print_spec_items(c.specs);
    printf(", \"layout\": ");
    print_layout(fs);
    printf(", \"seed\": %d, \"n\": %d, \"instances\": [", c.seed, c.n);
    for (int i = 0; i < c.n; ++i) {
      if (i)
        printf(",");
      print_raw_array(fs, buf[i]);
    }
    printf("], \"post_randint\": %d}", post);
  }
  printf("\n  ]\n");
}

int main() {
  printf("{\n");
  dump_binomial_double();
  dump_count();
  dump_count_throws();
  dump_vary();
  dump_randomize_order();
  dump_generate_initial_sample();
  printf("}\n");
  return 0;
}
