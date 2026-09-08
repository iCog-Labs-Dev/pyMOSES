// tools/cpp_probe/sort_demes_probe.cc
//
// Oracle for the question S12 raised but left open (docs/CARRY-FORWARD.md
// Sec 5b.1, task-report.md Sec 9 item 1): does `metapopulation::sort_demes`
// (moses/moses/metapopulation/merging.cc:503) reorder ties?
//
//   boost::sort(deme, std::greater<scored_instance<composite_score>>());
//
// `boost::range::sort` (boost/range/algorithm/sort.hpp) is a bare forward to
// `std::sort(begin, end, pred)` -- confirmed by reading that header on this
// machine, not assumed. `std::sort` in libstdc++ is introsort: quicksort
// (median-of-3 pivot, `__unguarded_partition_pivot`) down to chunks of
// `_S_threshold == 16` elements (confirmed: bits/stl_algo.h:1807 on THIS
// toolchain, gcc 16.2.1), then a single final insertion-sort pass over the
// whole range, with a heapsort fallback if the recursion depth budget
// (`2 * log2(n)`) is exhausted.
//
// This is a DIFFERENT algorithm shape from `std::partial_sort`'s
// heap-select + sort-heap (S12's `_partial_sort`), so S12's counterexample
// does not transfer automatically -- this probe measures `std::sort` (via
// `boost::sort`'s exact call shape) directly, using the same technique S12
// used for partial_sort: each element's identity is carried in its
// INSTANCE BITS (an integer id) that the comparator never reads
// (`scored_instance::operator>` / `composite_score::operator<`, types.cc:
// 131-149, read only `.second`), so the dumped id order after sorting IS
// the permutation libstdc++ chose.
//
// Two groups:
//
//   "full_sort" -- `std::sort(v.begin(), v.end(), std::greater<deme_inst_t>())`
//       over hand-built vectors of `deme_inst_t`, at sizes straddling
//       libstdc++'s own `_S_threshold = 16` boundary (insertion-sort-only
//       below/at it, quicksort-partitioned above it) and at sizes realistic
//       for an actual MOSES deme (up to ~2000 instances -- ordinary boolean
//       runs routinely produce demes in the hundreds to low thousands, per
//       docs/CARRY-FORWARD.md's own `large_nbh` neighbourhood-size
//       measurements).
//
//   "keep_top_unique_forward" -- takes the SAME operation
//       `keep_top_unique_candidates`'s own final line performs --
//       `deme.erase(deme.begin() + top_cnd, deme.end())` (merging.cc:600) --
//       against a realistic, heavily-tied, freshly-`std::sort`-ed deme, for
//       a `top_cnd` that falls INSIDE the tied block. This is deliberately
//       the narrower, unconditional half of the forward path (it does not
//       simulate the dedup scan's combo-tree-equality half, which this
//       probe has no combo trees to exercise and which is oracled instead
//       by tests/moses/metapopulation/test_merging.py) -- it exists only to
//       show, without approximation, that `std::sort`'s chosen PERMUTATION
//       decides which `top_cnd` ids occupy the surviving prefix.
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and this session's standing OOM caution. The largest
// case here is 2000 `deme_inst_t` elements (a few dozen bytes each): trivial
// memory, and std::sort on 2000 elements is microseconds.

#include <moses/moses/moses/types.h>
#include <moses/moses/optimization/optimization.h>
#include <moses/moses/representation/field_set.h>
#include <moses/moses/representation/instance.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

static field_set make_bit_fs(size_t n)
{
    return field_set(field_set::disc_spec(2), n);
}

static void print_double(double v)
{
    if (std::isnan(v)) {
        printf("NaN");
        return;
    }
    if (std::isinf(v)) {
        printf(v > 0 ? "Infinity" : "-Infinity");
        return;
    }
    printf("%.9g", v);
}

// Identity is carried in the instance's raw bits, exactly as
// tools/cpp_probe/crossover_probe.cc's own partial_sort group does (see
// that file's header for the rationale): the comparator never reads them.
static instance id_instance(const field_set& fs, unsigned long id)
{
    instance inst(fs.packed_width());
    for (size_t i = 0; i < fs.n_bits(); ++i)
        fs.set_raw(inst, fs.begin_bit_raw_idx() + i, (id >> i) & 1UL);
    return inst;
}

static unsigned long instance_id(const field_set& fs, const instance& inst)
{
    unsigned long id = 0;
    for (size_t i = 0; i < fs.n_bits(); ++i)
        if (fs.get_raw(inst, fs.begin_bit_raw_idx() + i)) id |= (1UL << i);
    return id;
}

struct Sample {
    unsigned long id;
    double score;
    unsigned complexity;
    double cpen;
    double upen;
};

struct SortCase {
    string label;
    vector<Sample> items;
};

// n_bits must be enough to give every id in the case a distinct pattern.
static size_t bits_for(size_t n) { return (size_t)std::ceil(std::log2((double)max(n, (size_t)2))) + 1; }

static void run_full_sort_cases()
{
    vector<SortCase> cases;

    // Straddle libstdc++'s own _S_threshold = 16 (bits/stl_algo.h:1807 on
    // this toolchain, confirmed by reading the header, not assumed): sizes
    // AT and BELOW it take std::sort's plain-insertion-sort-only path
    // (introsort_loop's `while (last - first > 16)` never fires), sizes
    // ABOVE it go through at least one quicksort partition first.
    for (size_t n : {size_t(15), size_t(16), size_t(17), size_t(20), size_t(32)}) {
        SortCase c{"all_tied_" + to_string(n), {}};
        for (unsigned long i = 0; i < n; ++i) c.items.push_back({i, -3.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }

    // Realistic MOSES deme sizes: ordinary boolean runs' demes/neighbourhood
    // estimates land in the hundreds to low thousands (docs/CARRY-FORWARD.md
    // / crossover.json's own large_nbh cases measure exactly this range,
    // e.g. estimate_neighborhood(1) == 703 for one real field_set). All
    // scores AND complexities tied -- composite_score::operator< ties
    // exactly when penalized score and complexity are both equal
    // (types.cc:139-148), the ordinary state of a boolean deme, not a
    // corner case.
    for (size_t n : {size_t(50), size_t(100), size_t(703), size_t(2000)}) {
        SortCase c{"all_tied_" + to_string(n), {}};
        for (unsigned long i = 0; i < n; ++i) c.items.push_back({i, -3.0, 0, 0.0, 0.0});
        cases.push_back(c);
    }

    // Several tied blocks of realistic size, interleaved by id rather than
    // grouped, so a reordering WITHIN a block is visible even though the
    // blocks themselves must still separate correctly (different scores are
    // NOT a tie and must sort strictly).
    {
        SortCase c{"interleaved_blocks_120", {}};
        for (unsigned long i = 0; i < 120; ++i)
            c.items.push_back({i, -(double)(i % 4), 0, 0.0, 0.0});
        cases.push_back(c);
    }

    // Equal penalized score, differing complexity: NOT a tie -- operator<
    // breaks it by complexity (types.cc:148), higher complexity sorts
    // EARLIER under greater<>. Control case: no ties present at all despite
    // equal raw score.
    {
        SortCase c{"equal_score_diff_complexity_40", {}};
        for (unsigned long i = 0; i < 40; ++i) c.items.push_back({i, -1.0, (unsigned)i, 0.0, 0.0});
        cases.push_back(c);
    }

    // Strictly descending / ascending, realistic size: no ties at all, the
    // controls -- with a strict total order the result is unique regardless
    // of algorithm, so these must match a stable sort even if ties
    // elsewhere do not.
    {
        SortCase c{"strictly_ordered_200", {}};
        for (unsigned long i = 0; i < 200; ++i)
            c.items.push_back({i, -(double)i, 0, 0.0, 0.0});
        cases.push_back(c);
    }

    // NaN block tied among themselves (composite_score::operator< says nan
    // vs nan is a tie, types.cc:136-137), realistic size.
    {
        const double nan_v = numeric_limits<double>::quiet_NaN();
        SortCase c{"with_nan_60", {}};
        for (unsigned long i = 0; i < 60; ++i)
            c.items.push_back({i, (i % 5 == 0) ? nan_v : -(double)(i % 3), 0, 0.0, 0.0});
        cases.push_back(c);
    }

    printf("[");
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        if (ci) printf(",");
        const SortCase& sc = cases[ci];
        field_set fs = make_bit_fs(bits_for(sc.items.size()));
        vector<deme_inst_t> v;
        for (const Sample& s : sc.items) {
            deme_inst_t si(id_instance(fs, s.id));
            si.second = composite_score(s.score, s.complexity, s.cpen, s.upen);
            v.push_back(si);
        }
        // The exact call shape of merging.cc:507's
        // boost::sort(deme, std::greater<scored_instance<composite_score>>())
        // -- boost::range::sort is a bare forward to std::sort (confirmed by
        // reading boost/range/algorithm/sort.hpp on this machine).
        std::sort(v.begin(), v.end(), std::greater<deme_inst_t>());

        printf("{\"label\":\"%s\",\"size\":%zu,\"input\":[", sc.label.c_str(), sc.items.size());
        for (size_t i = 0; i < sc.items.size(); ++i) {
            if (i) printf(",");
            const Sample& s = sc.items[i];
            printf("{\"id\":%lu,\"score\":", s.id);
            print_double(s.score);
            printf(",\"complexity\":%u,\"complexity_penalty\":", s.complexity);
            print_double(s.cpen);
            printf(",\"uniformity_penalty\":");
            print_double(s.upen);
            printf("}");
        }
        printf("],\"order\":[");
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) printf(",");
            printf("%lu", instance_id(fs, v[i].first));
        }
        printf("]}");
    }
    printf("]");
}

// --------------------------------------------------------------------------
// Group 2: does the reordering reach an observable outcome through the
// forward path sort_demes actually feeds? `keep_top_unique_candidates`'s
// own final, UNCONDITIONAL line is `deme.erase(deme.begin() + top_cnd,
// deme.end())` (merging.cc:600) -- run directly here, with no dedup
// approximation, against a realistic all-tied deme.
// --------------------------------------------------------------------------

static void run_keep_top_unique_forward_case()
{
    // A realistic heavily-tied deme (matches "all_tied_100" above) with a
    // top_cnd that falls INSIDE the tied block, so the truncation boundary
    // is meaningful.
    const size_t n = 100;
    const size_t top_cnd = 17; // arbitrary, mid-deme

    field_set fs = make_bit_fs(bits_for(n));
    vector<deme_inst_t> v;
    for (unsigned long i = 0; i < n; ++i) {
        deme_inst_t si(id_instance(fs, i));
        si.second = composite_score(-3.0, 0, 0.0, 0.0);
        v.push_back(si);
    }
    std::sort(v.begin(), v.end(), std::greater<deme_inst_t>());

    // merging.cc:600's own unconditional truncation, verbatim.
    v.erase(v.begin() + (long)top_cnd, v.end());

    printf("{\"label\":\"all_tied_100_top17\",\"deme_size\":%zu,\"top_cnd\":%zu,\"survivors\":[",
           n, top_cnd);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) printf(",");
        printf("%lu", instance_id(fs, v[i].first));
    }
    printf("]}");
}

int main()
{
    printf("{\"full_sort\":");
    run_full_sort_cases();
    printf(",\"keep_top_unique_forward\":");
    run_keep_top_unique_forward_case();
    printf("}\n");
    return 0;
}
