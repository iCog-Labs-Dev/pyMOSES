// tools/cpp_probe/contin_neighborhood_probe.cc
//
// Oracle for pyMoses S14 Task 3 (moses/moses/moses/neighborhood_sampling.cc's
// three continuous sampling helpers -- flip_LR:32, twiddle_contin_bit:38,
// generate_contin_neighbor:55 -- plus sample_from_neighborhood's own
// "modify contin" branch, neighborhood_sampling.h:172-178, that calls the
// last of them).
//
// Dumps, as JSON:
//   - "flip_LR": direct calls over every raw pseudo-bit value (Stop/Left/
//     Right) -- no RNG, deterministic.
//   - "twiddle_contin_bit": direct calls over (current value, next-is-
//     invalid, next value, seed) combinations, covering all three branches
//     documented at neighborhood_sampling.h:79-85 -- RNG-drawing.
//   - "generate_contin_neighbor": direct calls over several field_sets
//     (contin-only and contin-followed-by-disc, several depths), several
//     (idx, dist, seed) triples -- RNG-drawing, draw order matters.
//   - "sample_from_neighborhood_contin": end-to-end calls over a MIXED
//     field_set that includes a contin_spec, several (seed, dist,
//     sample_size) triples -- the task's actual exit criterion ("for a
//     fixed seed, contin neighbourhoods match C++ element for element").
//
// generate_contin_neighbor/twiddle_contin_bit/sample_from_neighborhood all
// draw from the process-wide opencog::randGen() singleton (no RNG
// parameter passed explicitly below, matching how a real caller in this
// port's own code resolves the same singleton via pymoses.util.rng.
// randGen()) -- determinism comes from reseeding immediately before each
// case, exactly mirroring the Python oracle tests.
//
// Build (see tools/capture_contin_neighborhood.py, which runs exactly
// this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/contin_neighborhood_probe \
//       tools/cpp_probe/contin_neighborhood_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and CARRY-FORWARD.md. Every case below uses a small contin
// depth (<= 8) and small dist/sample_size, per the task brief's own
// memory-bound instruction.

#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/moses/types.h>
#include <moses/moses/representation/field_set.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <string>
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

// ---------------- flip_LR ----------------

static void dump_flip_LR()
{
    field_set fs(field_set::contin_spec(0.0, 1.0, 2.0, 3), 1);
    printf("  \"flip_LR\": [\n");
    bool first = true;
    for (int raw_idx = 0; raw_idx < 3; ++raw_idx) {
        for (disc_t val : {0, 1, 2}) { // Stop, Left, Right
            instance inst(fs.packed_width());
            fs.set_raw(inst, raw_idx, val);
            field_set::disc_iterator itr = fs.begin_raw(inst);
            itr += raw_idx;
            flip_LR(itr);
            if (!first) printf(",\n");
            first = false;
            printf("    {\"raw_idx\": %d, \"before\": %lu, \"after\": %lu}",
                   raw_idx, (unsigned long)val, (unsigned long)fs.get_raw(inst, raw_idx));
        }
    }
    printf("\n  ],\n");
}

// ---------------- twiddle_contin_bit ----------------

struct TwiddleCase {
    disc_t cur;          // value at itr before the call
    bool next_invalid;   // true => next_itr = fs.end_raw(inst)
    disc_t next_val;     // value at next_itr, when not invalid
    unsigned seed;
};

static void dump_twiddle_contin_bit()
{
    // depth=6 so there is always a genuine "next" raw position to hold
    // next_val, plus one extra trailing position for the not-invalid
    // "next is real but its OWN value matters" cases.
    field_set fs(field_set::contin_spec(0.0, 1.0, 2.0, 6), 1);

    vector<TwiddleCase> cases = {
        // Stop branch: draws one randbool regardless of next_itr.
        {0, false, 0, 0}, {0, false, 0, 1}, {0, false, 0, 2}, {0, false, 0, 3},
        {0, true, 0, 0}, {0, true, 0, 1},
        // next invalid (end_raw sentinel): draws one randbool.
        {1, true, 0, 0}, {1, true, 0, 1}, {1, true, 0, 2}, {1, true, 0, 3},
        {2, true, 0, 0}, {2, true, 0, 1},
        // next valid and Stop: draws one randbool.
        {1, false, 0, 0}, {1, false, 0, 1}, {1, false, 0, 2}, {1, false, 0, 3},
        {2, false, 0, 0}, {2, false, 0, 1},
        // next valid and NOT Stop: plain flip, draws nothing (seed is
        // irrelevant here but included to confirm the RNG stream is
        // untouched -- see capture_contin_neighborhood.py's own check).
        {1, false, 1, 0}, {1, false, 2, 0}, {2, false, 1, 5}, {2, false, 2, 7},
    };

    printf("  \"twiddle_contin_bit\": [\n");
    bool first = true;
    for (auto& c : cases) {
        instance inst(fs.packed_width());
        fs.set_raw(inst, 0, c.cur);
        if (!c.next_invalid) fs.set_raw(inst, 1, c.next_val);

        randGen().seed(c.seed);
        // Draw one extra randbool() before and after, to detect any
        // extra/missing draw this call itself makes (a stream-length
        // check independent of the resulting value).
        bool pre = randGen().randbool();
        randGen().seed(c.seed);
        (void)pre;

        field_set::disc_iterator itr = fs.begin_raw(inst);
        field_set::disc_iterator next_itr =
            c.next_invalid ? fs.end_raw(inst) : (fs.begin_raw(inst) + 1);
        twiddle_contin_bit(itr, next_itr, randGen());

        disc_t after = fs.get_raw(inst, 0);
        bool post = randGen().randbool(); // stream-position marker

        if (!first) printf(",\n");
        first = false;
        printf("    {\"cur\": %lu, \"next_invalid\": %s, \"next_val\": %lu, \"seed\": %u, "
               "\"after\": %lu, \"post_randbool\": %s}",
               (unsigned long)c.cur, c.next_invalid ? "true" : "false",
               (unsigned long)c.next_val, c.seed, (unsigned long)after,
               post ? "true" : "false");
    }
    printf("\n  ],\n");
}

// ---------------- generate_contin_neighbor ----------------

struct GcnCase {
    string label;
    vector<unsigned> depths; // one contin_spec per depth, then (if trailing_disc) a disc_spec(3)
    bool trailing_disc;
    size_t idx;             // which contin spec to modify
    vector<int> raw;        // initial raw values (contin range only; disc left 0)
    unsigned dist;
    unsigned seed;
};

static field_set make_gcn_fs(const vector<unsigned>& depths, bool trailing_disc)
{
    vector<field_set::spec> specs;
    for (unsigned d : depths)
        specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, d));
    if (trailing_disc)
        specs.push_back(field_set::disc_spec(3));
    return field_set(specs.begin(), specs.end());
}

static void dump_generate_contin_neighbor()
{
    // NOTE on trailing_disc=true below, even for the "single_depth*" cases
    // that are conceptually about a lone contin field: whenever the
    // randomly-chosen interval's `up` reaches this contin field's own
    // `depth`, generate_contin_neighbor sets `next_itr` to the GLOBAL
    // `fs.end_raw(inst)` sentinel regardless of what (if anything) follows
    // this contin field in the raw layout. If this contin field were also
    // the LAST thing in the whole raw array, that sentinel would equal
    // `idx + 1` too, defeating twiddle_contin_bit's short-circuit guard
    // (`idx + 1 != next_idx`) and forcing a dereference of the sentinel
    // itself -- `field_set::get_raw`'s `_fields[idx]` is a bare
    // `std::vector::operator[]`, so that read is genuine UB in C++ (it
    // happened not to crash when measured, landing on unspecified memory
    // -- NOT a reproducible oracle value; see
    // test_neighborhood_sampling.py's own direct, non-oracle test for how
    // this port's `fs.get_raw`'s explicit bounds check turns the same
    // situation into a deterministic `IndexError` instead). A trailing
    // disc(3) field on every case below guarantees `idx + 1` always names
    // a REAL raw position (the disc field's own start) distinct from the
    // global sentinel, so every case here is well-defined in BOTH
    // languages and safely exercises the `up == depth` branch itself.
    vector<GcnCase> cases = {
        {"single_depth3_allstop", {3}, true, 0, {0, 0, 0}, 1, 0},
        {"single_depth3_allstop", {3}, true, 0, {0, 0, 0}, 1, 1},
        {"single_depth3_LRS", {3}, true, 0, {1, 2, 0}, 1, 0},
        {"single_depth3_LRS", {3}, true, 0, {1, 2, 0}, 1, 7},
        {"single_depth3_full_LRL_dist2", {3}, true, 0, {1, 2, 1}, 2, 0},
        {"single_depth3_full_LRL_dist3", {3}, true, 0, {1, 2, 1}, 3, 0},
        {"single_depth6_LRLSSS_dist1", {6}, true, 0, {1, 2, 1, 0, 0, 0}, 1, 3},
        {"single_depth6_LRLSSS_dist2", {6}, true, 0, {1, 2, 1, 0, 0, 0}, 2, 3},
        {"single_depth6_LRLSSS_dist3", {6}, true, 0, {1, 2, 1, 0, 0, 0}, 3, 3},
        {"single_depth6_LRLRLR_dist2", {6}, true, 0, {1, 2, 1, 2, 1, 2}, 2, 42},
        {"single_depth8_allstop_dist4", {8}, true, 0, {0, 0, 0, 0, 0, 0, 0, 0}, 4, 12345},
        // Contin followed by a disc field -- exercises the "up == depth"
        // end_raw-sentinel branch with REAL data sitting right after the
        // contin range (twiddle_contin_bit must not read into it).
        {"contin_then_disc_full_dist1", {4}, true, 0, {1, 2, 1, 2}, 1, 0},
        {"contin_then_disc_full_dist2", {4}, true, 0, {1, 2, 1, 2}, 2, 0},
        {"contin_then_disc_full_dist4", {4}, true, 0, {1, 2, 1, 2}, 4, 9},
        // Two contin specs -- idx selects the SECOND one, spec-indexed --
        // plus the same trailing disc(3) for the same reason as above.
        {"two_contin_second_idx", {3, 4}, true, 1, {1, 2, 0, 1, 2, 1, 0}, 2, 1},
    };

    printf("  \"generate_contin_neighbor\": [\n");
    bool first = true;
    for (auto& c : cases) {
        field_set fs = make_gcn_fs(c.depths, c.trailing_disc);
        instance inst(fs.packed_width());
        for (size_t i = 0; i < c.raw.size(); ++i)
            fs.set_raw(inst, i, c.raw[i]);
        if (c.trailing_disc)
            fs.set_raw(inst, c.raw.size(), 2); // nonzero disc value, must survive untouched

        randGen().seed(c.seed);
        field_set::contin_iterator itc = fs.begin_contin(inst);
        itc += c.idx;
        generate_contin_neighbor(fs, inst, itc, c.dist, randGen());

        if (!first) printf(",\n");
        first = false;
        printf("    {\"label\": %s, \"depths\": [", ("\"" + c.label + "\"").c_str());
        for (size_t i = 0; i < c.depths.size(); ++i) {
            if (i) printf(",");
            printf("%u", c.depths[i]);
        }
        printf("], \"trailing_disc\": %s, \"idx\": %zu, \"dist\": %u, \"seed\": %u, \"raw\": [",
               c.trailing_disc ? "true" : "false", c.idx, c.dist, c.seed);
        for (size_t i = 0; i < c.raw.size(); ++i) {
            if (i) printf(",");
            printf("%d", c.raw[i]);
        }
        printf("], \"trailing_disc_val\": %s, \"result\": ", c.trailing_disc ? "2" : "null");
        print_raw_array(fs, inst);
        printf("}\n");
    }
    printf("  ],\n");
}

// ---------------- sample_from_neighborhood (contin-inclusive) ----------------

struct SfnCase {
    unsigned seed;
    unsigned dist;
    unsigned sample_size;
    vector<int> center_raw; // contin(depth5), contin(depth3), disc(3), disc(4), 2 bits
};

static field_set make_sfn_fs()
{
    vector<field_set::spec> specs;
    specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, 5));
    specs.push_back(field_set::contin_spec(0.0, 1.0, 2.0, 3));
    specs.push_back(field_set::disc_spec(3));
    specs.push_back(field_set::disc_spec(4));
    specs.push_back(field_set::disc_spec(2)); // multiplicity 2 -> folds into bit range
    specs.push_back(field_set::disc_spec(2));
    return field_set(specs.begin(), specs.end());
}

static void dump_sample_from_neighborhood_contin()
{
    field_set fs = make_sfn_fs();
    size_t raw = fs.raw_size();

    vector<SfnCase> cases = {
        {0, 1, 15, {}},
        {0, 2, 20, {}},
        {1, 3, 25, {}},
        {42, 2, 30, {}},
        {12345, 4, 30, {}},
        {7, 1, 10, {}},
        {4294967295u, 1, 20, {}},
    };

    printf("  \"sample_from_neighborhood_contin\": [\n");
    bool first = true;
    for (auto& c : cases) {
        instance center(fs.packed_width());
        // Fixed, nontrivial center: mix of Left/Right/Stop across both
        // contin fields, plus nonzero disc values.
        vector<int> pattern = {1, 2, 1, 0, 0,   // contin depth5: LRL SS
                               2, 1, 0,          // contin depth3: RL S
                               1, 2, 0};         // disc(3), disc(4), bit-folded disc(2)
        for (size_t i = 0; i < pattern.size() && i < raw; ++i)
            fs.set_raw(center, i, pattern[i]);

        randGen().seed(c.seed);
        instance_set<composite_score> deme(c.sample_size, center, fs);
        sample_from_neighborhood(fs, c.dist, c.sample_size, deme.begin(), deme.end(), center);

        if (!first) printf(",\n");
        first = false;
        printf("    {\"seed\": %u, \"dist\": %u, \"sample_size\": %u, \"center_raw\": ",
               c.seed, c.dist, c.sample_size);
        print_raw_array(fs, center);
        printf(", \"instances\": [");
        bool firsti = true;
        for (const instance& inst : deme) {
            if (!firsti) printf(",");
            firsti = false;
            print_raw_array(fs, inst);
        }
        printf("]}\n");
    }
    printf("  ]\n");
}

int main()
{
    printf("{\n");
    dump_flip_LR();
    dump_twiddle_contin_bit();
    dump_generate_contin_neighbor();
    dump_sample_from_neighborhood_contin();
    printf("}\n");
    return 0;
}
