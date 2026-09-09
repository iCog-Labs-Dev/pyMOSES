// tools/cpp_probe/cluster_bscore_probe.cc
//
// Oracle for pyMeTTa's `cluster_bscore` (moses/moses/scoring/bscores.h:613,
// bscores.cc:932/937/1074/1079) -- the `-H cluster` scorer.
//
// Follows contin_bscore_probe.cc's shape exactly (tiny local JSON
// helpers, one probe binary with subcommand "modes", one capture_*.py
// driving it) in a SEPARATE file rather than editing that one or the
// shared scorers_probe.cc, per docs/CARRY-FORWARD.md Sec 5.1's
// shared-index concurrency warning.
//
// Why a probe at all, when `-H cluster` is drivable end to end through
// classic's own `moses` binary: stdout only ever shows the FINAL
// composite scores of surviving candidates. These cases pin
// `operator()`'s single bscore element directly, for inputs the search
// loop will not reach on its own -- float32 overflow, an underpopulated
// cluster, an exactly-zero RMS, a table too small for the OC_ASSERT, and
// an accumulator long enough for float32 summation to drift from double.
//
// Modes:
//   synthetic
//       Hand-built single-column ITables, evaluated with the tree "$1"
//       so `oned` IS the column verbatim and every case is a controlled
//       input to the k-means itself. Covers, case by case (see each
//       case's own "reaches:" note in the code):
//         three_clusters      normal convergence, one pass
//         multi_pass          convergence needing more than one pass
//         underpopulated      the `cnt[i] < 3.5` guard (bscores.cc:1008)
//         zero_rms            final == 0, i.e. `1.0 / 0.0` -> +inf
//         four_rows           the smallest table the OC_ASSERT admits
//         float32_sum         60 values near 2^23, where accumulating
//                             `sum[i] += sc` in double instead of float
//                             changes the answer
//   overflow
//       Tree "*($1 $1)" over a column near 1e30: the products are finite
//       DOUBLES (1e60) that overflow when narrowed to `score_t`
//       (binary32), so `std::isinf(sc)` fires (bscores.cc:982) purely
//       because of the narrowing. Split out from `synthetic` only
//       because it is the case a port that keeps `contin_t` precision
//       gets wrong, and it deserves to be visible on its own.
//   nan
//       Tree "+(*(1e300 1e300) *(-1e300 1e300))" -- +inf + -inf -> nan,
//       for EVERY row. `plus`/`times` do not throw OverflowException
//       (interpreter.cc:131-155 check nothing), so the nan reaches
//       `vals`. NOTE: `std::sort` over a range containing nan violates
//       its strict-weak-ordering precondition, i.e. this mode is
//       undefined behaviour in C++ by construction. It is here to record
//       what this toolchain actually does, and is run in its own process
//       so a crash cannot take the other modes' output with it.
//   too_small
//       A 3-row table: `OC_ASSERT(nclusters < oned.size())`
//       (bscores.cc:944) must fire. Caught and reported rather than
//       allowed to terminate the process.
//   table <csv> <target> <tree>...
//       Real tables. `cluster_bscore bsc(table.itable);` for each tree
//       given on the command line. Trees are POSITIONAL ($1, $2, ...);
//       column names are not accepted.
//
// Build (see tools/capture_cluster_bscore.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/cluster_bscore_probe \
//       tools/cpp_probe/cluster_bscore_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).

#include <moses/moses/scoring/bscores.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <opencog/util/exceptions.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;

// ---- tiny JSON helpers (mirrors contin_bscore_probe.cc) ----------------

static std::string float_json(double v)
{
    if (std::isnan(v)) return "\"nan\"";
    if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    std::string s(buf);
    // Keep "-0"/"0"/"123" from parsing back as a Python int, which would
    // discard the sign of zero -- see contin_bscore_probe.cc's own note.
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

static std::string bscore_array_to_json(const behavioral_score &bs)
{
    std::string out = "[";
    for (size_t i = 0; i < bs.size(); ++i) {
        if (i) out += ",";
        out += float_json((double)bs[i]);
    }
    out += "]";
    return out;
}

static std::string escape(const std::string &s)
{
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

// Dump one (scorer, tree) evaluation. `OC_ASSERT` throws
// opencog::AssertionException (oc_assert.cc:46), so it is catchable here
// even though an uncaught one aborts the real `moses` binary.
static void dump_eval(const char *label, cluster_bscore &bsc, const combo_tree &tr,
                      const std::string &tree_str)
{
    printf("  \"%s\": {\n", label);
    printf("    \"tree\": \"%s\",\n", escape(tree_str).c_str());
    try {
        behavioral_score bs = bsc(tr);
        printf("    \"bscore\": %s,\n", bscore_array_to_json(bs).c_str());
        printf("    \"sum_bscore\": %s,\n", float_json((double)bsc.sum_bscore(bs)).c_str());
    } catch (const std::exception &e) {
        printf("    \"exception\": \"%s\",\n", escape(e.what()).c_str());
    }
    printf("    \"best_possible_bscore\": %s,\n",
           bscore_array_to_json(bsc.best_possible_bscore()).c_str());
    printf("    \"min_improv\": %s\n", float_json((double)bsc.min_improv()).c_str());
    printf("  }");
}

static ITable one_col(const std::vector<double> &vals)
{
    ITable it;
    for (double v : vals) it.push_back(vertex_seq{contin_t(v)});
    return it;
}

static void dump_column(const std::vector<double> &vals)
{
    printf("    \"column\": [");
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i) printf(",");
        printf("%s", float_json(vals[i]).c_str());
    }
    printf("],\n");
}

// One synthetic case: dump the exact input column alongside the score, so
// the port's test does not have to re-derive the column from this file.
static void dump_case(const char *label, const std::vector<double> &vals,
                      const std::string &tree_str, bool last)
{
    ITable it = one_col(vals);
    cluster_bscore bsc(it);
    combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());
    printf("  \"%s\": {\n", label);
    dump_column(vals);
    printf("    \"tree\": \"%s\",\n", escape(tree_str).c_str());
    try {
        behavioral_score bs = bsc(tr);
        printf("    \"bscore\": %s,\n", bscore_array_to_json(bs).c_str());
    } catch (const std::exception &e) {
        printf("    \"exception\": \"%s\",\n", escape(e.what()).c_str());
    }
    printf("    \"best_possible_bscore\": %s,\n",
           bscore_array_to_json(bsc.best_possible_bscore()).c_str());
    printf("    \"min_improv\": %s\n", float_json((double)bsc.min_improv()).c_str());
    printf(last ? "  }\n" : "  },\n");
}

// ---- modes ---------------------------------------------------------------

static int mode_synthetic(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("{\n");

    // reaches: normal convergence. First three rows are 1/2/3 so the
    // initial centroids (bscores.cc:946-952, DO NOT FIX 3) already
    // straddle the three groups and one pass suffices.
    dump_case("three_clusters",
              {1.0, 2.0, 3.0,
               1.25, 1.5, 1.75, 0.75,
               2.25, 2.5, 1.9, 2.1,
               3.25, 3.5, 2.9, 3.1},
              "$1", false);

    // reaches: the `while (changed)` loop more than once. The first three
    // rows are all in the LOW group, so pass 1's edges are far too low
    // and the partition has to migrate.
    dump_case("multi_pass",
              {10.0, 14.0, 30.0,
               10.5, 11.0, 11.5, 12.0, 12.5,
               19.0, 19.5, 20.0, 20.5, 21.0,
               29.0, 29.5, 30.5, 31.0, 31.5},
              "$1", false);

    // reaches: `cnt[i] < 3.5` (bscores.cc:1008). The top group has only
    // three members, one of which the boundary-crossing defect (DO NOT
    // FIX 1) eats, leaving cnt == 2.
    dump_case("underpopulated",
              {1.0, 2.0, 3.0,
               1.1, 1.2, 1.3, 1.4,
               2.1, 2.2, 2.3, 2.4,
               3.1, 3.2},
              "$1", false);

    // reaches: `final == 0` and therefore `1.0 / 0.0` -> +inf
    // (bscores.cc:1055). Each of the first two clusters holds one
    // repeated value, so `squ/cnt` and `(sum/cnt)^2` agree exactly.
    dump_case("zero_rms",
              {1.0, 2.0, 3.0,
               1.0, 1.0, 1.0, 1.0,
               2.0, 2.0, 2.0, 2.0,
               3.0, 3.0, 3.0, 3.0},
              "$1", false);

    // reaches: the smallest table `OC_ASSERT(nclusters < oned.size())`
    // admits -- 4 rows, which then cannot fill three clusters of >= 3.5.
    dump_case("four_rows", {1.0, 2.0, 3.0, 4.0}, "$1", false);

    // reaches: float32 accumulation. 60 values just above 2^23, where a
    // float32 `sum[i] += sc` loses the fractional bits a double
    // accumulator would keep. A port that accumulates in Python's native
    // double disagrees here.
    {
        std::vector<double> v;
        v.push_back(8388609.0);      // first three rows seed the centroids
        v.push_back(8388640.0);
        v.push_back(8388672.0);
        for (int k = 0; k < 19; ++k) v.push_back(8388609.0 + 0.25 * k);
        for (int k = 0; k < 19; ++k) v.push_back(8388640.0 + 0.25 * k);
        for (int k = 0; k < 19; ++k) v.push_back(8388672.0 + 0.25 * k);
        dump_case("float32_sum", v, "$1", true);
    }

    printf("}\n");
    return 0;
}

static int mode_overflow(int argc, char **argv)
{
    (void)argc; (void)argv;
    // "*($1 $1)": 1e30 * 1e30 == 1e60 is a perfectly finite contin_t
    // (double), and `times` does not throw (interpreter.cc:143-155). It
    // becomes +inf only at `vals[j] = get_contin(oned[j])`
    // (bscores.cc:967), where score_t is binary32. So the `std::isinf`
    // guard at bscores.cc:982 fires purely because of that narrowing.
    // The `three_clusters` shape, scaled: three clean clusters, so
    // nothing here is degenerate for a reason of its own.
    static const double shape[15] = {1.0, 2.0, 3.0,
                                     1.25, 1.5, 1.75, 0.75,
                                     2.25, 2.5, 1.9, 2.1,
                                     3.25, 3.5, 2.9, 3.1};
    std::vector<double> big, small;
    for (double x : shape) { big.push_back(x * 1e30); small.push_back(x * 1e3); }

    printf("{\n");
    dump_case("overflow_times", big, "*($1 $1)", false);
    // Control 1: the SAME column with the identity tree. Each value is
    // finite in binary32 (3.5e30 < FLT_MAX), so the isinf guard does NOT
    // fire -- but `squ += sc*sc` (bscores.cc:1033) is float*float and
    // 1e60 is not representable, so `squ` and `sum*sum` both become +inf
    // and `final` becomes `inf - inf` == nan. A DIFFERENT answer from
    // the case above, which is what makes it a control.
    dump_case("overflow_control_identity", big, "$1", false);
    // Control 2: the same shape three orders of magnitude down, same
    // squaring tree. Nothing overflows anywhere and the score is finite,
    // proving neither the tree nor the shape is inherently degenerate.
    dump_case("overflow_control_scaled", small, "*($1 $1)", true);
    printf("}\n");
    return 0;
}

static int mode_nan(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("{\n");
    static const double shape[15] = {1.0, 2.0, 3.0,
                                     1.25, 1.5, 1.75, 0.75,
                                     2.25, 2.5, 1.9, 2.1,
                                     3.25, 3.5, 2.9, 3.1};
    std::vector<double> v(shape, shape + 15);
    dump_case("nan_everywhere", v, "+(*(1e300 1e300) *(-1e300 1e300))", false);
    // Control: the SAME column with the identity tree scores finitely,
    // so the -inf above is the nan guard and not the shape.
    dump_case("nan_control_finite", v, "$1", true);
    printf("}\n");
    return 0;
}

static int mode_too_small(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("{\n");
    dump_case("three_rows", {1.0, 2.0, 3.0}, "$1", false);
    dump_case("two_rows", {1.0, 2.0}, "$1", true);
    printf("}\n");
    return 0;
}

static int mode_table(int argc, char **argv)
{
    // table <csv> <target> <tree>...
    if (argc < 5) {
        fprintf(stderr, "table: need csv target tree...\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    Table table = loadTable(path, target);
    cluster_bscore bsc(table.itable);

    printf("{\n");
    printf("  \"n_rows\": %zu,\n", table.size());
    printf("  \"arity\": %d,\n", (int)table.get_arity());
    for (int i = 4; i < argc; ++i) {
        std::string tree_str = argv[i];
        // Empty label vector: trees are given POSITIONALLY ($1, $2, ...),
        // not by column name, so the same tree strings work against any
        // table regardless of its header row.
        combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());
        char label[64];
        snprintf(label, sizeof(label), "tree_%d", i - 4);
        dump_eval(label, bsc, tr, tree_str);
        printf(i + 1 < argc ? ",\n" : "\n");
    }
    printf("}\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s synthetic | overflow | nan | too_small | "
                "table <csv> <target> <tree>...\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "synthetic") return mode_synthetic(argc, argv);
    if (mode == "overflow") return mode_overflow(argc, argv);
    if (mode == "nan") return mode_nan(argc, argv);
    if (mode == "too_small") return mode_too_small(argc, argv);
    if (mode == "table") return mode_table(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
