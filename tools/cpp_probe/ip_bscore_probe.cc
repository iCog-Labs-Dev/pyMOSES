// tools/cpp_probe/ip_bscore_probe.cc
//
// Oracle for pyMeTTa's `interesting_predicate_bscore`
// (moses/moses/scoring/bscores.h:553, bscores.cc:726/761/880/886/896) --
// the `-H ip` ("find interesting patterns") scorer -- and for the two
// cogutil helpers it needs that nothing in this port had yet:
// `standardizedMannWhitneyU` (opencog/util/MannWhitneyU.h:64) and
// boost's `weighted_skewness` accumulator.
//
// Follows cluster_bscore_probe.cc's shape exactly (tiny local JSON
// helpers, one probe binary with subcommand "modes", one capture_*.py
// driving it) in a SEPARATE file rather than editing that one or the
// shared scorers_probe.cc, per docs/CARRY-FORWARD.md Sec 5.1's
// shared-index concurrency warning.
//
// Why a probe when `-H ip` IS drivable end to end through classic's own
// `moses` binary: stdout shows only the final composite scores of the
// surviving candidates, which fold the whole bscore into one float and
// hide which of the four weighted components produced it. These modes
// pin `operator()`'s bscore ELEMENT BY ELEMENT, plus the two statistics
// in isolation, plus the constructor's `_counter`/`_skewness`, for
// inputs the search loop does not reach on its own (a one-value filtered
// distribution, a zero-activation predicate, the default
// `min_activation == 0` division that makes classic print `nan`).
//
// Modes:
//   activation
//       `get_activation_penalty`'s two statements (bscores.cc:896-904)
//       over a grid of (activation, min, max, penalty). It is `private`,
//       so a derived class cannot call it; the grid pins the whole
//       expression -- both `fmax` overloads, the `1.0 - _max_activation`
//       promotion to double, `pow`'s C99 domain/pole errors and the
//       NEGATIVE NaN that the DEFAULT bounds (0.0, 1.0) produce.
//   stats
//       `weighted_skewness` and `standardizedMannWhitneyU` on hand-built
//       `Counter<contin_t, contin_t>`s, with no scorer in sight. Isolates
//       the two green-field statistics so a port can be tested against
//       them before any CTable exists. Also dumps `ranking()`
//       (opencog/util/ranking.h:40) and `MannWhitneyU` itself, since
//       `standardizedMannWhitneyU` is a thin wrapper over both.
//   table <csv> <target> <K> <J> <U> <X> <minact> <maxact> <penalty> <positive> <tree>...
//       A real table. Builds `Table::compressed()` exactly as
//       `table_problem_base::common_setup` (table-problems.cc:101) does,
//       constructs the scorer with the ten arguments `ip_problem::run`
//       (table-problems.cc:229) passes, and evaluates each tree. Dumps
//       the constructor's own state (`_counter`, `_skewness`) and, per
//       tree, the bscore, `sum_bscore`, and the CTable rows in
//       `std::map` order so a port can check its own iteration order
//       against C++'s.
//       Trees are POSITIONAL ($1, $2, ...); column names are not
//       accepted.
//
// Build (see tools/capture_ip_bscore.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/ip_bscore_probe \
//       tools/cpp_probe/ip_bscore_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).

#include <moses/moses/scoring/bscores.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <opencog/util/Counter.h>
#include <opencog/util/MannWhitneyU.h>
#include <opencog/util/exceptions.h>
#include <opencog/util/ranking.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/weighted_skewness.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;
using namespace boost::accumulators;

typedef Counter<contin_t, contin_t> counter_t;
typedef accumulator_set<contin_t, stats<tag::weighted_skewness>, contin_t> accumulator_t;

// ---- tiny JSON helpers (mirrors cluster_bscore_probe.cc) ---------------

static std::string float_json(double v)
{
    if (std::isnan(v)) return std::signbit(v) ? "\"-nan\"" : "\"nan\"";
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

// A Counter<contin_t, contin_t> as a JSON list of [key, count] pairs, in
// the container's own (std::map, ascending) order -- which is the whole
// point: the port has to reproduce that order, not a dict's.
static std::string counter_to_json(const counter_t &c)
{
    std::string out = "[";
    bool first = true;
    for (const auto &v : c) {
        if (!first) out += ",";
        first = false;
        out += "[" + float_json(v.first) + "," + float_json(v.second) + "]";
    }
    out += "]";
    return out;
}

// ---- the scorer's protected state, exposed -----------------------------

// `_counter`, `_pdf` and `_skewness` are `protected` (bscores.h:585-591),
// so a derived class is the only way to read what the constructor
// computed. Adds no behaviour.
struct ip_probe_bscore : public interesting_predicate_bscore {
    using interesting_predicate_bscore::interesting_predicate_bscore;
    const counter_t &counter() const { return _counter; }
    const counter_t &pdf() const { return _pdf; }
    contin_t skewness() const { return _skewness; }
};

// ---- mode: stats -------------------------------------------------------

static counter_t make_counter(const std::vector<std::pair<double, double>> &kv)
{
    counter_t c;
    for (const auto &p : kv) c[p.first] = p.second;
    return c;
}

static void dump_stats_case(const char *label,
                            const std::vector<std::pair<double, double>> &kv1,
                            const std::vector<std::pair<double, double>> &kv2,
                            bool last)
{
    counter_t c1 = make_counter(kv1), c2 = make_counter(kv2);

    printf("  \"%s\": {\n", label);
    printf("    \"c1\": %s,\n", counter_to_json(c1).c_str());
    printf("    \"c2\": %s,\n", counter_to_json(c2).c_str());
    printf("    \"ranking_c1_plus_c2\": %s,\n", counter_to_json(ranking(c1 + c2)).c_str());

    // `weighted_skewness` over each counter, accumulated in ASCENDING KEY
    // order -- the order `interesting_predicate_bscore` itself uses
    // (`for (const auto& v : pred_counter) acc(v.first, weight = v.second);`,
    // bscores.cc:827-828), because the container is a std::map.
    accumulator_t a1;
    for (const auto &v : c1) a1(v.first, weight = v.second);
    accumulator_t a2;
    for (const auto &v : c2) a2(v.first, weight = v.second);
    printf("    \"weighted_skewness_c1\": %s,\n",
           float_json(weighted_skewness(a1)).c_str());
    printf("    \"weighted_skewness_c2\": %s,\n",
           float_json(c2.empty() ? nan("") : weighted_skewness(a2)).c_str());

    printf("    \"mann_whitney_u\": %s,\n",
           float_json(MannWhitneyU<contin_t, contin_t>(c1, c2)).c_str());
    printf("    \"standardized_mann_whitney_u\": %s\n",
           float_json(standardizedMannWhitneyU<contin_t, contin_t>(c1, c2)).c_str());
    printf(last ? "  }\n" : "  },\n");
}

static int mode_stats(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("{\n");

    // reaches: the ordinary path. c2 is a strict sub-distribution of c1,
    // which is the shape `operator()` always produces (pred_counter's
    // keys are a subset of _counter's).
    dump_stats_case("subset",
                    {{0.5, 4.0}, {0.75, 3.0}, {1.25, 2.0}, {2.0, 1.0}, {7.5, 1.0},
                     {13.5, 1.0}, {22.0, 1.0}},
                    {{0.5, 1.0}, {1.25, 1.0}, {7.5, 1.0}},
                    false);

    // reaches: a symmetric distribution, whose weighted skewness is
    // exactly 0 -- the value an uninitialised `_skewness` happens to
    // share on this toolchain, so it is worth having pinned.
    dump_stats_case("symmetric",
                    {{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}, {5.0, 1.0}},
                    {{2.0, 1.0}, {4.0, 1.0}},
                    false);

    // reaches: fractional counts (what `--score-weight` produces), where
    // `boost::accumulate(c | map_values, 0)`'s INT init value
    // (MannWhitneyU.h:53, :67, :69) truncates n1/n2 at every step.
    dump_stats_case("fractional_counts",
                    {{1.0, 1.5}, {2.0, 2.5}, {3.0, 0.25}},
                    {{2.0, 0.75}, {3.0, 0.25}},
                    false);

    // reaches: an EMPTY c2 -- what a predicate selecting no rows gives.
    // n2 == 0 makes `sU` zero and the standardized U a 0/0 nan.
    dump_stats_case("empty_c2",
                    {{1.0, 1.0}, {2.0, 2.0}, {5.0, 1.0}},
                    {},
                    false);

    // reaches: c2 == c1, the `true` predicate. U == mU exactly, so the
    // standardized U is exactly 0 (or -0).
    dump_stats_case("identical",
                    {{1.0, 1.0}, {2.0, 2.0}, {5.0, 1.0}},
                    {{1.0, 1.0}, {2.0, 2.0}, {5.0, 1.0}},
                    false);

    // reaches: a single-point c2, where the skewness denominator
    // `(m2 - mean^2)^{3/2}` is 0 and weighted_skewness is 0/0.
    dump_stats_case("single_point_c2",
                    {{1.0, 1.0}, {2.0, 2.0}, {5.0, 1.0}},
                    {{2.0, 2.0}},
                    false);

    // reaches: a heavily right-skewed distribution, so `weighted_skewness`
    // is large and positive and the sign of `diff_skewness` is decided by
    // the conditioned half.
    dump_stats_case("skewed",
                    {{1.0, 10.0}, {2.0, 5.0}, {3.0, 2.0}, {50.0, 1.0}},
                    {{1.0, 2.0}, {50.0, 1.0}},
                    true);

    printf("}\n");
    return 0;
}

// ---- mode: activation --------------------------------------------------

// A verbatim copy of `interesting_predicate_bscore::get_activation_penalty`
// (bscores.cc:896-904). It is `private`, so a derived class cannot reach
// it; copying the two statements is the only way to pin them, and the
// copy is checked against the real thing by the `table` mode's own
// bscores (the last bscore element is exactly this value).
//
// Every type here is load-bearing and is reproduced exactly as written:
// `_min_activation`/`_max_activation`/`_penalty`/`activation` are all
// `score_t` (binary32), `score_t(0)` is a float zero, but `1.0 -
// _max_activation` promotes to DOUBLE -- so the second `fmax` argument
// is a double and the outer `fmax` is the double overload, narrowed only
// on the store to `score_t dst`.
static score_t probe_activation_penalty(score_t activation, score_t min_activation,
                                        score_t max_activation, score_t penalty)
{
    score_t dst = fmax(fmax(min_activation - activation, score_t(0))
                      / min_activation,
                      fmax(activation - max_activation, score_t(0))
                      / (1.0 - max_activation));
    return log(pow((1.0 - dst), penalty));
}

static void dump_activation_case(score_t act, score_t lo, score_t hi, score_t pen, bool last)
{
    printf("  {\"activation\": %s, \"min_activation\": %s, \"max_activation\": %s, "
           "\"penalty\": %s, \"dst\": %s, \"activation_penalty\": %s}%s",
           float_json((double)act).c_str(), float_json((double)lo).c_str(),
           float_json((double)hi).c_str(), float_json((double)pen).c_str(),
           float_json((double)(score_t)fmax(fmax(lo - act, score_t(0)) / lo,
                                            fmax(act - hi, score_t(0)) / (1.0 - hi))).c_str(),
           float_json((double)probe_activation_penalty(act, lo, hi, pen)).c_str(),
           last ? "\n" : ",\n");
}

static int mode_activation(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("[\n");
    // reaches: the DEFAULT bounds (`--min-rand-input` 0.0 /
    // `--max-rand-input` 1.0, problem-params.cc:1147/:1158), where BOTH
    // quotients are 0/0. This is what the 16 `-Hip` sweep rows hit, and
    // why classic prints `nan`.
    dump_activation_case(1.0f, 0.0f, 1.0f, 1.0f, false);
    dump_activation_case(0.5f, 0.0f, 1.0f, 1.0f, false);
    dump_activation_case(0.0f, 0.0f, 1.0f, 1.0f, false);
    dump_activation_case(0.5f, 0.0f, 1.0f, 2.0f, false);
    dump_activation_case(0.5f, 0.0f, 1.0f, 0.5f, false);
    // reaches: real bounds, activation inside them -> dst == 0 -> penalty 0.
    dump_activation_case(0.5f, 0.2f, 0.8f, 1.0f, false);
    dump_activation_case(0.2f, 0.2f, 0.8f, 1.0f, false);
    dump_activation_case(0.8f, 0.2f, 0.8f, 1.0f, false);
    // reaches: below the low bound -> the FIRST quotient wins.
    dump_activation_case(0.1f, 0.2f, 0.8f, 1.0f, false);
    dump_activation_case(0.05f, 0.2f, 0.8f, 3.0f, false);
    // reaches: dst == 1 exactly (activation 0 with a positive low bound)
    // -> `pow(0, penalty)` -> `log(0)` -> -inf.
    dump_activation_case(0.0f, 0.2f, 0.8f, 1.0f, false);
    dump_activation_case(0.0f, 0.2f, 0.8f, 2.5f, false);
    // reaches: above the high bound -> the SECOND quotient wins.
    dump_activation_case(0.9f, 0.2f, 0.8f, 1.0f, false);
    dump_activation_case(1.0f, 0.2f, 0.8f, 1.0f, false);
    // reaches: dst > 1 -> `1.0 - dst` NEGATIVE -> `pow(negative,
    // non-integer)` is the C99 domain error.
    dump_activation_case(0.0f, 0.5f, 0.8f, 0.5f, false);
    dump_activation_case(0.1f, 0.9f, 0.95f, 0.5f, false);
    dump_activation_case(0.1f, 0.9f, 0.95f, 3.0f, false);
    dump_activation_case(0.1f, 0.9f, 0.95f, 2.0f, false);
    // reaches: only the HIGH bound degenerate (max_activation == 1).
    dump_activation_case(0.5f, 0.2f, 1.0f, 1.0f, false);
    // reaches: only the LOW bound degenerate (min_activation == 0).
    dump_activation_case(0.5f, 0.0f, 0.8f, 1.0f, false);
    dump_activation_case(0.9f, 0.0f, 0.8f, 1.0f, false);
    // reaches: a NEGATIVE penalty (`--alpha=-1`, which classic accepts).
    dump_activation_case(0.1f, 0.2f, 0.8f, -1.0f, false);
    dump_activation_case(0.5f, 0.2f, 0.8f, -1.0f, true);
    printf("]\n");
    return 0;
}

// ---- mode: table -------------------------------------------------------

static void dump_ctable_rows(const CTable &ct)
{
    // The CTable in `std::map` order -- key first (as a list of the
    // printed vertices), then the cell's (value, count) pairs. This is
    // the order `operator()`'s `boost::for_each(_ctable | map_values,
    // pred_cache, ...)` walks, and the order the
    // `pred_counter[...] = mv.second` ASSIGNMENT (bscores.cc:800, note:
    // assignment, not +=) makes observable.
    printf("  \"ctable_rows\": [");
    bool first_row = true;
    for (const auto &row : ct) {
        if (!first_row) printf(",");
        first_row = false;
        printf("\n    {\"key\": [");
        // `-H ip` admits only all-boolean inputs (table-problems.cc:233-237),
        // so the key's variant alternative is a `builtin_seq`
        // (`vector<builtin>`), NOT a `vertex_seq` -- `get_seq<vertex>()`
        // throws `boost::bad_get` on it. Measured, not assumed.
        const builtin_seq bs_key = row.first.get_seq<builtin>();
        for (size_t i = 0; i < bs_key.size(); ++i) {
            if (i) printf(",");
            stringstream ss;
            ss << vertex(bs_key[i]);
            printf("\"%s\"", escape(ss.str()).c_str());
        }
        printf("], \"cell\": [");
        bool first_cell = true;
        for (const auto &mv : row.second) {
            if (!first_cell) printf(",");
            first_cell = false;
            printf("[%s,%s]", float_json(get_contin(mv.first.value)).c_str(),
                   float_json(mv.second).c_str());
        }
        printf("]}");
    }
    printf("\n  ],\n");
}

static int mode_table(int argc, char **argv)
{
    // table <csv> <target> <K> <J> <U> <X> <minact> <maxact> <penalty> <positive> <tree>...
    if (argc < 12) {
        fprintf(stderr,
                "table: need csv target K J U X minact maxact penalty positive tree...\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    score_t kld_w = (score_t)atof(argv[4]);
    score_t skew_w = (score_t)atof(argv[5]);
    score_t stdU_w = (score_t)atof(argv[6]);
    score_t skewU_w = (score_t)atof(argv[7]);
    score_t minact = (score_t)atof(argv[8]);
    score_t maxact = (score_t)atof(argv[9]);
    score_t penalty = (score_t)atof(argv[10]);
    bool positive = atoi(argv[11]) != 0;

    Table table = loadTable(path, target);
    CTable ctable = table.compressed("");

    ip_probe_bscore bsc(ctable, kld_w, skew_w, stdU_w, skewU_w,
                        minact, maxact, penalty, positive);

    printf("{\n");
    printf("  \"n_rows\": %zu,\n", table.size());
    printf("  \"arity\": %d,\n", (int)table.get_arity());
    printf("  \"weights\": [%s,%s,%s,%s],\n",
           float_json((double)kld_w).c_str(), float_json((double)skew_w).c_str(),
           float_json((double)stdU_w).c_str(), float_json((double)skewU_w).c_str());
    printf("  \"activation\": [%s,%s],\n",
           float_json((double)minact).c_str(), float_json((double)maxact).c_str());
    printf("  \"penalty\": %s,\n", float_json((double)penalty).c_str());
    printf("  \"positive\": %s,\n", positive ? "true" : "false");
    dump_ctable_rows(ctable);
    printf("  \"counter\": %s,\n", counter_to_json(bsc.counter()).c_str());
    // `_pdf` is `_counter` BEFORE `set_p_pdf` ran -- `set_p_pdf` copies
    // into KLDS's own `p_pdf` and mutates only that copy (KLD.h:67-73),
    // so this must still be the raw counts, not densities.
    printf("  \"pdf\": %s,\n", counter_to_json(bsc.pdf()).c_str());
    printf("  \"skewness\": %s,\n", float_json(bsc.skewness()).c_str());
    printf("  \"best_possible_bscore\": %s,\n",
           bscore_array_to_json(bsc.best_possible_bscore()).c_str());
    printf("  \"min_improv\": %s,\n", float_json((double)bsc.min_improv()).c_str());

    printf("  \"trees\": [");
    for (int i = 12; i < argc; ++i) {
        std::string tree_str = argv[i];
        // Empty label vector: trees are given POSITIONALLY ($1, $2, ...),
        // not by column name, so the same tree strings work against any
        // table regardless of its header row.
        combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());
        printf(i == 12 ? "\n    {" : ",\n    {");
        printf("\"tree\": \"%s\", ", escape(tree_str).c_str());
        try {
            behavioral_score bs = bsc(tr);
            printf("\"bscore\": %s, ", bscore_array_to_json(bs).c_str());
            printf("\"sum_bscore\": %s", float_json((double)bsc.sum_bscore(bs)).c_str());
        } catch (const std::exception &e) {
            printf("\"exception\": \"%s\"", escape(e.what()).c_str());
        }
        printf("}");
    }
    printf("\n  ]\n}\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s stats | activation | table <csv> <target> <K> <J> <U> <X> "
                "<minact> <maxact> <penalty> <positive> <tree>...\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "stats") return mode_stats(argc, argv);
    if (mode == "activation") return mode_activation(argc, argv);
    if (mode == "table") return mode_table(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
