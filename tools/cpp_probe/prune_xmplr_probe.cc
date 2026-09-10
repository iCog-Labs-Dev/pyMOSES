// tools/cpp_probe/prune_xmplr_probe.cc
//
// Oracle for `deme_expander::prune_xmplr` (moses/moses/deme/deme_expander.cc:308)
// and `deme_expander::fs_to_names` (moses/moses/deme/deme_expander.cc:48).
//
// Both are PROTECTED members of `struct deme_expander` (deme_expander.h:85's
// `protected:` section covers h:115 and h:128), NOT public as first assumed --
// so this probe derives a tiny struct that re-exposes them with `using`, which
// is legal for a protected member accessed through a derived type. Everything
// else about the setup (real Table/CTable/ctruth_table_bscore/behave_cscore,
// real logical_reduction(2), real hill_climbing injected only because the
// constructor demands an optimizer reference) is copied verbatim from
// tools/cpp_probe/deme_expander_probe.cc.
//
// The corpus is HARDCODED rather than passed on argv, deliberately: every
// exemplar contains `$1`/`$2`/`$3`, and CARRY-FORWARD.md Sec 5.10 documents
// three separate false readings in this project caused by a shell expanding
// exactly those tokens before the binary ever saw them. Nothing here is
// shell-quoted, so nothing can be shell-mangled.
//
// ONE CASE PER PROCESS. `prune_xmplr` can hand `logical_reduce` a `not` with
// zero children -- prune the only child of a `not` and that is exactly what is
// left -- and classic SEGFAULTS on that tree (measured; see the test module
// that consumes this probe). A whole-corpus loop therefore dies on the first
// such case and reports nothing for the rest, so the driver runs `--case N`
// per index and records each exit status separately.
//
// Output: one JSON object per invocation on stdout:
//   {"exemplar": "...", "selected": [..], "pruned": "...",
//    "pruned_empty": bool, "names": [..]}
// where "names" is fs_to_names(selected, ilabels) over the loaded table's own
// input labels -- the ORDER of that list is the thing under test (a C++
// std::set<arity_t> iterates ascending; a Python set does not).
//
// Build:
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/prune_xmplr_probe \
//       tools/cpp_probe/prune_xmplr_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ).
//
// Usage: prune_xmplr_probe <csv-path> <target-feature> <arity> [case-index]
//        prune_xmplr_probe <csv-path> <target-feature> <arity> --count

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <moses/comboreduct/table/table_io.h>
#include <moses/moses/deme/deme_expander.h>
#include <moses/moses/optimization/hill-climbing.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <moses/moses/scoring/bscores.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace opencog::moses;
using namespace opencog::reduct;
using namespace std;

// prune_xmplr / fs_to_names are protected (deme_expander.h:85 `protected:`),
// so re-expose them through a derived type.
struct probe_dex : public deme_expander {
    using deme_expander::deme_expander;
    using deme_expander::fs_to_names;
    using deme_expander::prune_xmplr;
};

static string json_string(const string &value)
{
    string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '\"') out += '\\';
        out += c;
    }
    return out + "\"";
}

static string rendered(const combo_tree &tr)
{
    stringstream out;
    out << tr;
    return out.str();
}

struct corpus_case {
    const char *exemplar;
    vector<arity_t> selected;
};

// The corpus. Every trap this port has to reproduce:
//  - sibling arguments (an erase that returns the WRONG successor
//    silently truncates the walk -- see `or(and($1 $2) $3)`),
//  - arguments at several depths,
//  - arguments under `not` (cases 6, 7, 8, 9, 12, 20 -- the ones that
//    SEGFAULT classic, see this file's header),
//  - an exemplar that is a bare argument, so pruning it EMPTIES the
//    tree and the `type_to_exemplar` seed fallback fires,
//  - selecting everything / selecting nothing.
static const vector<corpus_case> corpus = {
    {"and($1 $2 $3)", {0, 2}},
    {"and($1 $2 $3)", {}},
    {"and($1 $2 $3)", {0, 1, 2}},
    {"or(and($1 $2) $3)", {}},
    {"or(and($1 $2) $3)", {2}},
    {"or(and($1 $2) $3)", {0}},
    {"or(and($1 $2) not($3))", {1}},
    {"and(not($1) $2)", {}},
    {"and(not($1) $2)", {1}},
    {"and(not($1) not($2))", {0}},
    {"$1", {}},
    {"$1", {0}},
    {"not($1)", {}},
    {"or($1 and($2 or($3 $1)))", {0}},
    {"or($1 and($2 or($3 $1)))", {2}},
    {"or($1 and($2 or($3 $1)))", {}},
    {"and(or($1 $2) or($2 $3))", {1}},
    {"and($1 $1 $2)", {0}},
    {"and($1 $1 $2)", {1}},
    {"or(and($1 $2 $3) and($1 $2 $3))", {}},
    {"and(not($1) or($2 not($3)))", {2}},
    {"not($1)", {0}},
    {"and(not($1) not($2))", {0, 1}},
    {"or(and($1 $2) $3)", {0, 1, 2}},
};

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <csv-path> <target-feature> <arity> [case-index|--count]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    string path = argv[1];
    string target_feature = argv[2];
    int arity = atoi(argv[3]);

    if (argc >= 5 && string(argv[4]) == "--count") {
        printf("%zu\n", corpus.size());
        return 0;
    }
    // No index given: run every case in one process. That only works when no
    // case in the range crashes -- see this file's header.
    int only = (argc >= 5) ? atoi(argv[4]) : -1;
    if (only >= (int)corpus.size()) {
        fprintf(stderr, "case index %d out of range (%zu cases)\n", only, corpus.size());
        return 1;
    }

    Table table = loadTable(path, target_feature);
    CTable ctable = table.compressed();

    ctruth_table_bscore bsc(ctable);
    behave_cscore cscorer(bsc);

    type_tree sig = gen_signature(id::boolean_type, arity);

    logical_reduction reduction(empty_ignore_ops);
    const rule &r2 = reduction(2);

    deme_parameters dparams;
    dparams.perm_ratio = -1.0f;
    subsample_deme_filter_parameters fparams;

    hc_parameters hcp(/*widen=*/false, /*step=*/false, /*cross=*/false);
    hill_climbing hc((optim_parameters()), hcp);

    probe_dex dex(sig, r2, r2, cscorer, hc, dparams, fparams);

    const string_seq &ilabels = ctable.get_input_labels();

    printf("[\n");
    bool first = true;
    for (size_t ci = 0; ci < corpus.size(); ++ci) {
        if (only >= 0 && (int)ci != only) continue;
        const corpus_case &cc = corpus[ci];
        combo_tree xmplr = str2combo_tree(cc.exemplar, std::vector<std::string>());
        feature_set selected(cc.selected.begin(), cc.selected.end());

        combo_tree pruned = dex.prune_xmplr(xmplr, selected);
        string_seq names = dex.fs_to_names(selected, ilabels);

        if (!first) printf(",\n");
        first = false;
        printf("  {\"case\":%zu,\"exemplar\":%s,\"selected\":[", ci,
               json_string(cc.exemplar).c_str());
        {
            bool f2 = true;
            for (arity_t f : selected) {
                if (!f2) printf(",");
                f2 = false;
                printf("%d", (int)f);
            }
        }
        printf("],\"pruned\":%s,\"pruned_empty\":%s,\"xmplr_after\":%s,\"names\":[",
               json_string(rendered(pruned)).c_str(), pruned.empty() ? "true" : "false",
               json_string(rendered(xmplr)).c_str());
        {
            bool f3 = true;
            for (const string &n : names) {
                if (!f3) printf(",");
                f3 = false;
                printf("%s", json_string(n).c_str());
            }
        }
        printf("]}");
    }
    printf("\n]\n");
    return 0;
}
