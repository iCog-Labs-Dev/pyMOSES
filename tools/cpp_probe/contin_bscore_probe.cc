// tools/cpp_probe/contin_bscore_probe.cc
//
// Oracle for pyMoses S14 Task 3's `contin_bscore` (moses/moses/scoring/
// bscores.h:181, bscores.cc:140-190) -- extends the scorers_probe.cc/
// capture_scorers.py JSON-dump pattern (tiny local JSON helpers, one
// probe binary with subcommand "modes", one capture_*.py driving it) in
// a SEPARATE file rather than editing the shared scorers_probe.cc/
// capture_scorers.py, which back the (differently-owned) discriminating_
// bscore/select_bscore/precision_bscore/time_dispersion family -- see
// docs/CARRY-FORWARD.md Sec 5.1's shared-index concurrency warning.
//
// Modes:
//   synthetic
//       A small (5-row) hand-built OTable/ITable, run through BOTH
//       squared_error and abs_error. No file I/O.
//
//   table_default <csv> <target>
//       `contin_bscore bsc(table);` -- NO err_function_type argument at
//       all, i.e. classic's own real default (squared_error), not a
//       value this probe or the port chose. Satisfies the task's own
//       "run classic's own defaults" requirement.
//
//   table <csv> <target> <eft: squared|abs>
//       `contin_bscore bsc(table, eft);` with an EXPLICIT eft, covering
//       the abs_error branch table_default cannot reach.
//
//   complexity <alphabet_size> <stdev...>
//       `bsc.set_complexity_coef(alphabet_size, stdev)` for each stdev,
//       dumping `get_complexity_coef()` after each call. Uses a trivial
//       throwaway table just to construct a contin_bscore instance.
//
//   complexity_ratio <ratio...>
//       `bsc.set_complexity_coef(ratio)` (the ONE-argument overload,
//       brought into contin_bscore's scope unchanged via `using
//       bscore_base::set_complexity_coef` -- bscores.h:217) for each
//       ratio, dumping `get_complexity_coef()` after each call.
//
// Build (see tools/capture_contin_bscore.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/contin_bscore_probe \
//       tools/cpp_probe/contin_bscore_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).

#include <moses/moses/scoring/bscores.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/comboreduct/combo/iostream_combo.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace std;

// ---- tiny JSON helpers (mirrors scorers_probe.cc) ----------------------

static std::string float_json(double v)
{
    if (std::isnan(v)) return "\"nan\"";
    if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    std::string s(buf);
    // Preserve float-ness (and the sign of zero) through a JSON round
    // trip: a bare "-0"/"0"/"123" with no '.'/'e' is valid JSON but
    // Python's json module parses it as an int, silently discarding the
    // sign of zero -- observed directly on this exact probe's synthetic
    // mode, where sq(0.0) legitimately produces -0.0 for one row and a
    // bit-exactness check against the re-parsed golden value failed
    // ONLY on that formatting round trip, not on the actual computed
    // value. Appending ".0" makes every such token parse back as a
    // Python float, sign of zero included.
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

static void dump_scorer(const char *label, contin_bscore &bsc, const combo_tree &tr)
{
    behavioral_score bs = bsc(tr);
    behavioral_score best = bsc.best_possible_bscore();
    score_t mi = bsc.min_improv();
    score_t sum = bsc.sum_bscore(bs);
    printf("  \"%s\": {\n", label);
    printf("    \"bscore\": %s,\n", bscore_array_to_json(bs).c_str());
    printf("    \"best_possible_bscore\": %s,\n", bscore_array_to_json(best).c_str());
    printf("    \"min_improv\": %s,\n", float_json((double)mi).c_str());
    printf("    \"sum_bscore\": %s\n", float_json((double)sum).c_str());
    printf("  }");
}

// ---- modes ---------------------------------------------------------------

static int mode_synthetic(int argc, char **argv)
{
    (void)argc; (void)argv;
    // tar = 2*x1 - 0.5 (roughly); tree = *(2 $1); irrational-in-binary
    // values on both sides so squaring in double vs. any premature
    // float32 narrowing would measurably disagree.
    ITable it;
    it.push_back(vertex_seq{contin_t(1.0 / 3.0)});
    it.push_back(vertex_seq{contin_t(2.0 / 3.0)});
    it.push_back(vertex_seq{contin_t(0.1)});
    it.push_back(vertex_seq{contin_t(-0.7)});
    it.push_back(vertex_seq{contin_t(1.23456789)});
    OTable ot;
    ot.push_back(vertex(contin_t(0.5)));
    ot.push_back(vertex(contin_t(1.5)));
    ot.push_back(vertex(contin_t(0.05)));
    ot.push_back(vertex(contin_t(-1.4)));
    ot.push_back(vertex(contin_t(2.469)));

    combo_tree tr = str2combo_tree("*(2 $1)", std::vector<std::string>());

    printf("{\n");
    contin_bscore sq_bsc(ot, it); // default eft: squared_error
    dump_scorer("squared_error", sq_bsc, tr);
    printf(",\n");
    contin_bscore abs_bsc(ot, it, contin_bscore::abs_error);
    dump_scorer("abs_error", abs_bsc, tr);
    printf("\n}\n");
    return 0;
}

static int mode_table_default(int argc, char **argv)
{
    // table_default <csv> <target>
    if (argc < 4) {
        fprintf(stderr, "table_default: need csv target\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3];
    Table table = loadTable(path, target);

    // "+($1 $2)" (not "*($1 $2)"): trainingRegression.dat's own target is
    // exactly x1*x2 (verified: row 1 gives 1.567521326983612948e-02 *
    // 5.028885506232726871e-02 == 7.882885281978582260e-04, the recorded
    // y), so a "*" tree would score an all-zero bscore here -- a poor
    // discriminator for narrowing bugs, which only show up on genuinely
    // NONZERO residuals. "+" gives nonzero residuals throughout.
    combo_tree tr = str2combo_tree("+($1 $2)", std::vector<std::string>());

    printf("{\n");
    contin_bscore bsc(table); // NO eft argument -- classic's own default.
    printf("  \"n_rows\": %zu,\n", table.size());
    dump_scorer("default_eft_squared", bsc, tr);
    printf("\n}\n");
    return 0;
}

static int mode_table(int argc, char **argv)
{
    // table <csv> <target> <eft>
    if (argc < 5) {
        fprintf(stderr, "table: need csv target eft\n");
        return 1;
    }
    std::string path = argv[2], target = argv[3], eft_name = argv[4];
    Table table = loadTable(path, target);
    contin_bscore::err_function_type eft =
        eft_name == "abs" ? contin_bscore::abs_error : contin_bscore::squared_error;

    combo_tree tr = str2combo_tree("+($1 $2)", std::vector<std::string>());

    printf("{\n");
    contin_bscore bsc(table, eft);
    printf("  \"n_rows\": %zu,\n", table.size());
    dump_scorer(eft_name.c_str(), bsc, tr);
    printf("\n}\n");
    return 0;
}

static int mode_complexity(int argc, char **argv)
{
    // complexity <alphabet_size> <stdev...>
    if (argc < 4) {
        fprintf(stderr, "complexity: need alphabet_size stdev...\n");
        return 1;
    }
    unsigned alphabet_size = (unsigned)atoi(argv[2]);

    ITable it;
    it.push_back(vertex_seq{contin_t(1.0)});
    OTable ot;
    ot.push_back(vertex(contin_t(1.0)));
    contin_bscore bsc(ot, it);

    printf("{\n  \"alphabet_size\": %u,\n  \"results\": [\n", alphabet_size);
    for (int i = 3; i < argc; ++i) {
        float stdev = (float)atof(argv[i]);
        bsc.set_complexity_coef(alphabet_size, stdev);
        if (i > 3) printf(",\n");
        printf("    {\"stdev\": %s, \"complexity_coef\": %s}",
               float_json((double)stdev).c_str(),
               float_json((double)bsc.get_complexity_coef()).c_str());
    }
    printf("\n  ]\n}\n");
    return 0;
}

static int mode_complexity_ratio(int argc, char **argv)
{
    // complexity_ratio <ratio...>
    if (argc < 3) {
        fprintf(stderr, "complexity_ratio: need ratio...\n");
        return 1;
    }
    ITable it;
    it.push_back(vertex_seq{contin_t(1.0)});
    OTable ot;
    ot.push_back(vertex(contin_t(1.0)));
    contin_bscore bsc(ot, it);

    printf("{\n  \"results\": [\n");
    for (int i = 2; i < argc; ++i) {
        float ratio = (float)atof(argv[i]);
        bsc.set_complexity_coef(ratio);
        if (i > 2) printf(",\n");
        printf("    {\"ratio\": %s, \"complexity_coef\": %s}",
               float_json((double)ratio).c_str(),
               float_json((double)bsc.get_complexity_coef()).c_str());
    }
    printf("\n  ]\n}\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mode> ...\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "synthetic") return mode_synthetic(argc, argv);
    if (mode == "table_default") return mode_table_default(argc, argv);
    if (mode == "table") return mode_table(argc, argv);
    if (mode == "complexity") return mode_complexity(argc, argv);
    if (mode == "complexity_ratio") return mode_complexity_ratio(argc, argv);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
