// tools/cpp_probe/scorers_probe.cc
//
// Dumps discriminating_bscore/{recall,prerec,bep,f_one}_bscore,
// select_bscore, precision_bscore, and bscore_ctable_time_dispersion
// output as JSON, for differential testing against pymetta/moses/scoring/
// {discriminating_bscore,select_bscore,precision_bscore,time_dispersion}.py.
//
// Build (see score_probe.cc's own comment for the verified recipe --
// compiling against the installed headers alone fails):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/scorers_probe tools/cpp_probe/scorers_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lmoses -lcogutil
//
// Usage: scorers_probe <mode> ...
//
//   scorers_probe disc <csv> <target> <scorer> <min_thr> <max_thr> <hardness> <tree> [ratio]
//       <scorer> in {recall, prerec, bep, f_one} (f_one ignores thresholds).
//
//   scorers_probe select <csv> <target> <lower_pct> <upper_pct> <hardness> <positive:0|1> <tree>
//
//   scorers_probe precision <csv> <target> <act_pressure> <min_act> <max_act> <positive:0|1> <tree> [ratio]
//
//   scorers_probe timedisp_hmax <csv> <target>
//       Constructs bscore_ctable_time_dispersion on a real (untimestamped)
//       boolean CTable and dumps the resulting _Hmax -- this port's CTable
//       carries no per-observation timestamp at all (see table.py's module
//       docstring), so every observation collapses into the SAME default
//       TimedValue timestamp (table.cc:513, `TimedValue(*out_it)` uses the
//       default-constructed `boost::gregorian::date()` for every row when
//       no timestamp column is loaded) -- this mode measures what _Hmax
//       classic actually computes for that (the only reachable) case.
//
//   scorers_probe timedisp_penalty <pressure> <exponent> <hmax> <true_counts...>
//       Directly exercises bscore_ctable_time_dispersion::
//       get_time_dispersion_penalty on a SYNTHETIC CTableTime built by
//       hand (one bucket per <true_counts> entry, false count fixed at 1.0
//       per bucket so entropy varies with the true/false split) -- this
//       bypasses the timestamp-collection pipeline entirely (see the mode
//       above and time_dispersion.py's own module docstring for why that
//       pipeline cannot be exercised from real data in this port) and
//       isolates the pure entropy-penalty formula, which the Python port
//       DOES fully implement (given a CTableTime-shaped mapping).
//       Because _Hmax is fixed by construction rather than measured from
//       real data here, this mode also accepts an explicit override via a
//       tiny subclass -- see probe_time_dispersion below.
//
// IMPORTANT: like score_probe.cc, CTable is a std::map, so all "rows"
// output is dumped keyed by its own input tuple, not positionally -- see
// that probe's own comment for why.

#include <moses/moses/scoring/discriminating_bscore.h>
#include <moses/moses/scoring/select_bscore.h>
#include <moses/moses/scoring/precision_bscore.h>
#include <moses/moses/scoring/time_dispersion.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;

// ---- tiny JSON helpers (mirrors score_probe.cc) -----------------------

static std::string json_escape(const std::string &s)
{
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out;
}

static std::string quote(const std::string &s) { return "\"" + json_escape(s) + "\""; }

static std::string float_json(double v)
{
    if (std::isnan(v)) return "\"nan\"";
    if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

static std::string vertex_to_json(const vertex &v)
{
    if (const builtin *b = boost::get<builtin>(&v)) {
        if (*b == id::logical_true) return "true";
        if (*b == id::logical_false) return "false";
        std::ostringstream os;
        os << *b;
        return quote(os.str());
    }
    if (const contin_t *c = boost::get<contin_t>(&v)) {
        return float_json((double)*c);
    }
    std::ostringstream os;
    os << v;
    return quote(os.str());
}

static std::string key_to_json(const CTable::key_type &key)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < key.size(); ++i) {
        if (i) os << ",";
        os << vertex_to_json(key.get_at<vertex>(i));
    }
    os << "]";
    return os.str();
}

static std::string bscore_rows_to_json(const CTable &ctable, const behavioral_score &bs)
{
    // Zips `bs` positionally against `ctable`'s own iteration order --
    // valid only when `bs` was produced by iterating this exact CTable
    // (true for every bscore below: one CTable, one operator() call).
    std::ostringstream os;
    os << "[";
    size_t i = 0;
    bool first = true;
    for (const CTable::value_type &kv : ctable) {
        if (i >= bs.size()) break;
        if (!first) os << ",";
        first = false;
        os << "{\"inputs\": " << key_to_json(kv.first)
           << ", \"value\": " << float_json((double)bs[i]) << "}";
        ++i;
    }
    os << "]";
    return os.str();
}

static std::string bscore_array_to_json(const behavioral_score &bs)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < bs.size(); ++i) {
        if (i) os << ",";
        os << float_json((double)bs[i]);
    }
    os << "]";
    return os.str();
}

// ---- probe subclasses exposing protected state -------------------------

struct probe_select_bscore : public select_bscore
{
    probe_select_bscore(const CTable &ct, double lo, double hi, double hard, bool pos)
        : select_bscore(ct, lo, hi, hard, pos) {}
    double lower_bound() const { return (double)_lower_bound; }
    double upper_bound() const { return (double)_upper_bound; }
    double effective_length() const { return _effective_length; }
    const std::vector<double> &weights() const { return _weights; }
};

struct probe_time_dispersion : public bscore_ctable_time_dispersion
{
    probe_time_dispersion(const CTable &ct, float pressure, float exponent,
                          TemporalGranularity g)
        : bscore_ctable_time_dispersion(ct, pressure, exponent, g) {}
    // Trivial overrides -- this probe never calls either; they exist
    // purely so this concrete subclass can be instantiated at all
    // (bscore_base::operator()/best_possible_bscore are pure virtual).
    behavioral_score operator()(const combo_tree&) const override { return behavioral_score(); }
    behavioral_score best_possible_bscore() const override { return behavioral_score(); }
    float hmax() const { return _Hmax; }
    // Exposes the protected instance method directly for the
    // timedisp_penalty mode; also lets that mode override _Hmax so the
    // formula can be probed independently of what the constructor itself
    // measures (see timedisp_hmax for the constructor's own value).
    float penalty(const CTableTime &ctt) const { return get_time_dispersion_penalty(ctt); }
    void set_hmax(float h) { _Hmax = h; }
};

// ---- mode implementations ----------------------------------------------

static int mode_disc(int argc, char **argv)
{
    // disc <csv> <target> <scorer> <min_thr> <max_thr> <hardness> <tree> [ratio]
    if (argc < 9) {
        fprintf(stderr, "disc: need csv target scorer min_thr max_thr hardness tree [ratio]\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3], scorer = argv[4];
    float min_thr = atof(argv[5]), max_thr = atof(argv[6]), hardness = atof(argv[7]);
    std::string tree_str = argv[8];
    bool have_ratio = argc > 9;
    double ratio = have_ratio ? atof(argv[9]) : 0.0;

    Table table = loadTable(path, target);
    CTable ctable = table.compressed();
    combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());

    discriminating_bscore *bsc = nullptr;
    if (scorer == "recall") bsc = new recall_bscore(ctable, min_thr, max_thr, hardness);
    else if (scorer == "prerec") bsc = new prerec_bscore(ctable, min_thr, max_thr, hardness);
    else if (scorer == "bep") bsc = new bep_bscore(ctable, min_thr, max_thr, hardness);
    else if (scorer == "f_one") bsc = new f_one_bscore(ctable);
    else { fprintf(stderr, "disc: unknown scorer %s\n", scorer.c_str()); return 1; }

    if (have_ratio) bsc->set_complexity_coef((score_t)ratio);

    behavioral_score bs = (*bsc)(tr);
    behavioral_score best = bsc->best_possible_bscore();
    score_t mi = bsc->min_improv();
    score_t coef = bsc->get_complexity_coef();
    score_t sum = bsc->sum_bscore(bs);

    printf("{\n");
    printf("  \"scorer\": %s,\n", quote(scorer).c_str());
    printf("  \"bscore\": %s,\n", bscore_array_to_json(bs).c_str());
    // Row-keyed dump too: prerec_bscore's bscore is one element PER
    // CTABLE ROW (plus a fixed 2-element tail) -- see this probe's own
    // module comment (score_probe.cc's identical note) on why CTable's
    // std::map iteration order is not reproduced by this port's
    // insertion-ordered CTable, so a positional comparison of the
    // per-row prefix is not meaningful across the two languages.
    // Harmless (if unused) for recall/bep/f_one, whose 2-element bscore
    // is NOT row-keyed at all.
    printf("  \"bscore_rows\": %s,\n", bscore_rows_to_json(ctable, bs).c_str());
    printf("  \"best_possible_bscore\": %s,\n", bscore_array_to_json(best).c_str());
    printf("  \"min_improv\": %s,\n", float_json((double)mi).c_str());
    printf("  \"complexity_coef\": %s,\n", float_json((double)coef).c_str());
    printf("  \"sum_bscore\": %s\n", float_json((double)sum).c_str());
    printf("}\n");
    delete bsc;
    return 0;
}

static int mode_select(int argc, char **argv)
{
    // select <csv> <target> <lower_pct> <upper_pct> <hardness> <positive> <tree>
    if (argc < 9) {
        fprintf(stderr, "select: need csv target lower_pct upper_pct hardness positive tree\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    double lower_pct = atof(argv[4]), upper_pct = atof(argv[5]), hardness = atof(argv[6]);
    bool positive = atoi(argv[7]) != 0;
    std::string tree_str = argv[8];

    Table table = loadTable(path, target);
    CTable ctable = table.compressed();
    combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());

    probe_select_bscore bsc(ctable, lower_pct, upper_pct, hardness, positive);

    behavioral_score bs = bsc(tr);
    behavioral_score best = bsc.best_possible_bscore();
    behavioral_score worst = bsc.worst_possible_bscore();
    score_t mi = bsc.min_improv();
    score_t sum_unweighted = 0.0;
    for (score_t x : bs) sum_unweighted += x;
    score_t sum_weighted = bsc.sum_bscore(bs);

    printf("{\n");
    printf("  \"lower_bound\": %s,\n", float_json(bsc.lower_bound()).c_str());
    printf("  \"upper_bound\": %s,\n", float_json(bsc.upper_bound()).c_str());
    printf("  \"effective_length\": %s,\n", float_json(bsc.effective_length()).c_str());
    behavioral_score weights_bs;
    for (double w : bsc.weights()) weights_bs.push_back((score_t)w);
    printf("  \"weights\": %s,\n", bscore_array_to_json(weights_bs).c_str());
    // Every element of select_bscore's bscore is row-keyed (exactly one
    // per ctable row, no fixed-position tail) -- see mode_disc's own
    // comment on why a positional array is not comparable across the two
    // languages' differing CTable iteration order.
    printf("  \"bscore_rows\": %s,\n", bscore_rows_to_json(ctable, bs).c_str());
    printf("  \"best_possible_bscore_rows\": %s,\n", bscore_rows_to_json(ctable, best).c_str());
    printf("  \"worst_possible_bscore_rows\": %s,\n", bscore_rows_to_json(ctable, worst).c_str());
    printf("  \"min_improv\": %s,\n", float_json((double)mi).c_str());
    printf("  \"sum_bscore_unweighted\": %s,\n", float_json((double)sum_unweighted).c_str());
    printf("  \"sum_bscore_weighted\": %s\n", float_json((double)sum_weighted).c_str());
    printf("}\n");
    return 0;
}

static int mode_precision(int argc, char **argv)
{
    // precision <csv> <target> <act_pressure> <min_act> <max_act> <positive> <tree> [ratio]
    if (argc < 9) {
        fprintf(stderr, "precision: need csv target act_pressure min_act max_act positive tree [ratio]\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    float act_pressure = atof(argv[4]), min_act = atof(argv[5]), max_act = atof(argv[6]);
    bool positive = atoi(argv[7]) != 0;
    std::string tree_str = argv[8];
    bool have_ratio = argc > 9;
    double ratio = have_ratio ? atof(argv[9]) : 0.0;

    Table table = loadTable(path, target);
    CTable ctable = table.compressed();
    combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());

    precision_bscore bsc(ctable, act_pressure, min_act, max_act, positive);
    if (have_ratio) bsc.set_complexity_coef((score_t)ratio);

    behavioral_score bs = bsc(tr);
    behavioral_score best = bsc.best_possible_bscore();
    behavioral_score worst = bsc.worst_possible_bscore();
    score_t mi = bsc.min_improv();
    score_t coef = bsc.get_complexity_coef();
    score_t sum = bsc.sum_bscore(bs);

    // bs = one row-keyed element per ctable row, THEN a fixed 2-element
    // tail (phantom 0.5-or-0.0, activation_penalty) -- see mode_disc's
    // own comment for why the row-keyed prefix is dumped by key, not
    // position. The tail is small and fixed-position, dumped plainly.
    behavioral_score tail;
    for (size_t i = ctable.size(); i < bs.size(); ++i) tail.push_back(bs[i]);

    printf("{\n");
    printf("  \"bscore_rows\": %s,\n", bscore_rows_to_json(ctable, bs).c_str());
    printf("  \"bscore_tail\": %s,\n", bscore_array_to_json(tail).c_str());
    printf("  \"best_possible_bscore\": %s,\n", bscore_array_to_json(best).c_str());
    printf("  \"worst_possible_bscore\": %s,\n", bscore_array_to_json(worst).c_str());
    printf("  \"min_improv\": %s,\n", float_json((double)mi).c_str());
    printf("  \"complexity_coef\": %s,\n", float_json((double)coef).c_str());
    printf("  \"sum_bscore\": %s\n", float_json((double)sum).c_str());
    printf("}\n");
    return 0;
}

static int mode_timedisp_hmax(int argc, char **argv)
{
    // timedisp_hmax <csv> <target>
    if (argc < 4) {
        fprintf(stderr, "timedisp_hmax: need csv target\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    Table table = loadTable(path, target);
    CTable ctable = table.compressed();

    probe_time_dispersion bsc(ctable, 0.0f, 1.0f, TemporalGranularity::day);
    printf("{\n");
    printf("  \"hmax\": %s,\n", float_json((double)bsc.hmax()).c_str());
    printf("  \"ctable_size\": %u\n", (unsigned)ctable.size());
    printf("}\n");
    return 0;
}

static int mode_timedisp_penalty(int argc, char **argv)
{
    // timedisp_penalty <csv-for-base-ctor> <target> <pressure> <exponent> <true_counts...>
    // (false count fixed at 1.0 per bucket; base ctor's own ctable just
    // needs to be a valid boolean CTable -- its _Hmax is overridden below
    // so this mode tests the formula independent of what a real ctable's
    // own (always-degenerate, see module comment) _Hmax would be.)
    if (argc < 6) {
        fprintf(stderr, "timedisp_penalty: need csv target pressure exponent true_counts...\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    float pressure = atof(argv[4]), exponent = atof(argv[5]);

    Table table = loadTable(path, target);
    CTable ctable = table.compressed();
    probe_time_dispersion bsc(ctable, pressure, exponent, TemporalGranularity::day);

    // Explicit Hmax override: log2(number of buckets), matching what a
    // genuinely time-diverse ctable (never constructible in this port,
    // see time_dispersion.py's module docstring) would have produced.
    int n_buckets = argc - 6;
    bsc.set_hmax((float)log2((double)n_buckets));

    CTableTime ctt;
    for (int i = 0; i < n_buckets; ++i) {
        double true_count = atof(argv[6 + i]);
        TTable::value_type ts = boost::gregorian::date(2000, 1, 1) + boost::gregorian::date_duration(i);
        Counter<vertex, count_t> c;
        c[vertex(id::logical_true)] = true_count;
        c[vertex(id::logical_false)] = 1.0;
        ctt[ts] = c;
    }

    float pen = bsc.penalty(ctt);
    printf("{\n");
    printf("  \"hmax\": %s,\n", float_json((double)bsc.hmax()).c_str());
    printf("  \"penalty\": %s\n", float_json((double)pen).c_str());
    printf("}\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mode> ...\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "disc") return mode_disc(argc, argv);
    if (mode == "select") return mode_select(argc, argv);
    if (mode == "precision") return mode_precision(argc, argv);
    if (mode == "timedisp_hmax") return mode_timedisp_hmax(argc, argv);
    if (mode == "timedisp_penalty") return mode_timedisp_penalty(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
