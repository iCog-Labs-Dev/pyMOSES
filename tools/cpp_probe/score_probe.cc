// tools/cpp_probe/score_probe.cc
//
// Dumps ctruth_table_bscore/behave_cscore/composite_score/tree_complexity
// output as JSON, for differential testing against pymoses/moses/scoring/
// {scoring_base,bscores,behave_cscore}.py and pymoses/moses/moses/
// {types,complexity}.py.
//
// Build (see task-6-brief.md / CLAUDE.md's verified probe recipe --
// compiling against the installed headers alone fails, see table_probe.cc's
// own comment for why):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/score_probe tools/cpp_probe/score_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lmoses -lcogutil
//
// Usage: score_probe <csv-path> <target-feature> <combo-tree-str>
//                     [weighting-feature] [complexity-ratio]
//
// `complexity-ratio`, if given, is passed to bscore_base::set_complexity_
// coef(score_t) (scoring_base.cc:63) before scoring, so
// composite_score.complexity_penalty is exercised with a non-zero value
// too, not just the class's own default of 0.0. Pass "" for
// weighting-feature to skip it while still giving a complexity-ratio.
//
// The target is always resolved BY NAME (never by position): table_io.py's
// module docstring / task-5's "Target by name vs. by position" finding
// established that a positional-looking target only works on the sparse
// table format, and every dataset this probe is pointed at is a headed,
// dense CSV. `weighting-feature`, if given, is passed to Table::compressed
// (NOT to loadTable's ignore_features) exactly the way tableUTest.cxxtest's
// test_weightCTable does it.
//
// best_possible_bscore()/worst_possible_bscore()/min_improv() depend only
// on the CTable, not on the tree being scored -- they are dumped on every
// invocation regardless of which tree was passed, so a single throwaway
// tree argument (e.g. "logical_true") is enough to capture them for a
// given dataset.
//
// IMPORTANT ordering note: CTable is a std::map, so ctruth_table_bscore's
// per-row vectors (bs/best_bs/worst_bs) are produced in the map's *sorted*
// key order -- an order this port's CTable (a plain, insertion-ordered
// dict, see table.py) does not reproduce. So this probe does not emit a
// bare parallel array: each row's own input tuple is dumped alongside its
// three per-row values, and the *Python* side re-associates by key rather
// than by position. See tests/moses/scoring/test_bscores.py.

#include <moses/moses/scoring/bscores.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;

// ---- tiny JSON helpers (mirrors table_probe.cc) ----------------------

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
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Same rules as table_probe.cc's vertex_to_json: this probe's whole
// corpus is boolean-only (ctruth_table_bscore requires a boolean-output
// CTable), so only the builtin branch is ever exercised in practice, but
// the others are kept for the same "don't silently mishandle an
// unexpected type" reason table_probe.cc keeps them.
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
    if (const enum_t *e = boost::get<enum_t>(&v)) {
        return quote(e->getContent());
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

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <csv-path> <target-feature> <combo-tree-str> "
                "[weighting-feature]\n",
                argv[0]);
        return 1;
    }
    std::string path = argv[1];
    std::string target_feature = argv[2];
    std::string tree_str = argv[3];
    std::string weighting_feature = argc > 4 ? argv[4] : "";
    bool have_ratio = argc > 5;
    double complexity_ratio = have_ratio ? atof(argv[5]) : 0.0;

    Table table = loadTable(path, target_feature);
    CTable ctable = table.compressed(weighting_feature);

    combo_tree tr = str2combo_tree(tree_str, std::vector<std::string>());

    ctruth_table_bscore bsc(ctable);
    if (have_ratio) bsc.set_complexity_coef((score_t)complexity_ratio);

    behavioral_score bs = bsc(tr);
    score_t sum = bsc.sum_bscore(bs);
    complexity_t cpxy = bsc.get_complexity(tr);
    score_t coef = bsc.get_complexity_coef();

    behavioral_score best_bs = bsc.best_possible_bscore();
    behavioral_score worst_bs = bsc.worst_possible_bscore();
    score_t min_improv = bsc.min_improv();

    behave_cscore behsc(bsc);
    composite_score cs = behsc.get_cscore(tr);

    // One pass over ctable, zipped positionally against bs/best_bs/
    // worst_bs -- valid because all four were produced by iterating this
    // exact same CTable instance, so std::map's iteration order (whatever
    // it is) is identical across all of them. See the module comment for
    // why the Python side must not assume that order matches its own.
    printf("{\n");
    printf("  \"rows\": [\n");
    size_t i = 0;
    bool first = true;
    for (const CTable::value_type &kv : ctable) {
        if (!first) printf(",\n");
        first = false;
        printf("    {\"inputs\": %s, \"bscore\": %s, \"best_possible_bscore\": %s, "
               "\"worst_possible_bscore\": %s}",
               key_to_json(kv.first).c_str(), float_json((double)bs[i]).c_str(),
               float_json((double)best_bs[i]).c_str(), float_json((double)worst_bs[i]).c_str());
        ++i;
    }
    printf("\n  ],\n");
    printf("  \"sum_bscore\": %s,\n", float_json((double)sum).c_str());
    printf("  \"tree_complexity\": %u,\n", (unsigned)cpxy);
    printf("  \"complexity_coef\": %s,\n", float_json((double)coef).c_str());
    printf("  \"min_improv\": %s,\n", float_json((double)min_improv).c_str());
    printf("  \"composite_score\": {\n");
    printf("    \"score\": %s,\n", float_json((double)cs.get_score()).c_str());
    printf("    \"complexity\": %u,\n", (unsigned)cs.get_complexity());
    printf("    \"complexity_penalty\": %s,\n",
           float_json((double)cs.get_complexity_penalty()).c_str());
    printf("    \"uniformity_penalty\": %s,\n",
           float_json((double)cs.get_uniformity_penalty()).c_str());
    printf("    \"penalized_score\": %s\n", float_json((double)cs.get_penalized_score()).c_str());
    printf("  }\n");
    printf("}\n");
    return 0;
}
