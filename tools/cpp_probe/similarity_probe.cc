// tools/cpp_probe/similarity_probe.cc
//
// Oracle for pyMoses's port of moses/comboreduct/combo/similarity.{h,cc}
// (pymoses/comboreduct/combo/similarity.py). Links directly against the
// real, exported opencog::combo::tree_flatten/tree_similarity symbols in
// libcomboreduct.so -- not reimplemented here.
//
// Dumps, as JSON:
//   - "flatten": tree_flatten(str) -> its toString() rendering, over every
//     case classic's own tests/comboreduct/combo/similarityUTest.cxxtest
//     uses (test_simple/test_nest/test_long) plus a few extra shapes.
//   - "similarity": tree_similarity(str, str) over every case
//     similarityUTest's own test_distance uses, PLUS adversarial cases
//     designed to probe whether `size_t acnt - size_t bcnt` (ss_bscore-
//     style unsigned wraparound, see similarity.cc:191's `dist +=
//     fabs(acnt - bcnt)`) actually manifests when a branch present in
//     BOTH trees is MORE common in the second tree than the first -- this
//     determines whether tree_similarity(a, b) == tree_similarity(b, a)
//     for such inputs, which the intended "Manhattan distance" semantics
//     require but the literal size_t-subtraction-before-fabs shape does
//     not guarantee.
//
// Build (see tools/capture_similarity.py, which runs exactly this):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/similarity_probe \
//       tools/cpp_probe/similarity_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and CARRY-FORWARD.md.

#include <moses/comboreduct/combo/similarity.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace opencog::combo;
using namespace std;

static string json_escape(const string& s)
{
    string out;
    char buf[8];
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\t') out += "\\t";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c < 0x20) { snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
        else out += (char)c;
    }
    return out;
}

static void dump_flatten()
{
    vector<string> cases = {
        "+($1 $2)",
        "+ ($1 2)",
        "+		(1  $2)",
        "+  ($1)",
        "+( 1)",
        "+ (  )",
        "+( +( $2))",
        "+( +(1 $2))",
        "+(3 +(1 $2))",
        "+(+(1 $2) 3)",
        "+(1 2 3)",
        "+(1 2 3 4)",
        "+(1 1 1 1 1)",
        "+(1 1 *(1 1) 1)",
        "+(1 1 *(2 2) 3)",
        "and($1 $1 or($2 $2) $3)",
    };
    printf("  \"flatten\": [\n");
    bool first = true;
    for (auto& s : cases) {
        tree_branch_vector tbv = tree_flatten(s);
        string rendered = toString(tbv);
        if (!first) printf(",\n");
        first = false;
        printf("    {\"input\": \"%s\", \"result\": \"%s\"}",
               json_escape(s).c_str(), json_escape(rendered).c_str());
    }
    printf("\n  ],\n");
}

static void dump_similarity()
{
    struct Case { string a, b; };
    vector<Case> cases = {
        // similarityUTest.cxxtest's own test_distance cases, with bare
        // identifier placeholders (a, b, c, d, e -- indefinite_object
        // tokens in classic's own wider grammar) replaced 1:1 by
        // arguments ($1=a, $2=b, $3=c, $4=d, $5=e): this port's combo
        // grammar is narrowed to builtin | argument | contin_t token
        // forms only (vertex.py's own module docstring), with no
        // indefinite_object/string-literal vertex kind at all. The
        // algorithm is purely structural/token-identity-based (same
        // distinctness relation either way), and this substitution is
        // confirmed, not merely assumed, by reproducing similarityUTest's
        // own expected distances EXACTLY (0,2,2,8,3,1,1,2,3,5,2) below.
        {"+($1 $2 $3)", " +  ( $1  $2  $3 )  "},
        {"+($1 $2 $3)", "+($1 $2 $4)"},
        {"+($1 $2 $3 $1 $1 $1 $1)", "+($1 $2 $4 $1 $1 $1 $1)"},
        {"+($1 $2 $3 $1 $1 $1 $1)", "+($1 $2 $4 $4 $4 $4 $4)"},
        {"+($1 $2 $3)", "+($1 $2)"},
        {"+($1 $2 $3)", "+($2 $3)"},
        {"+($1 $2 $3 $4 $5)", "+($2 $3 $4 $5)"},
        {"+($1 $2 $3 $4 $5)", "+($3 $4 $5)"},
        {"+($1 $2 $3 $4 $5)", "+($4 $5)"},
        {"+($1 $2 $3 $4 $5)", "+($5)"},
        {"+($1 $2 $1 $1 $1 $1)", "+($1 $3 $1 $1 $1 $1)"},
        // Adversarial: does tree_similarity(a,b) == tree_similarity(b,a)?
        // "+(a b c)" vs "+(a b)" above is asymmetric input SIZE but let's
        // also directly swap the ORDER of an already-known asymmetric
        // pair, and construct one where the SECOND tree has strictly MORE
        // of a branch that also appears in the first (not just "extra"
        // branches missing from the first entirely).
        {"+($1 $2)", "+($1 $2 $3)"},          // swap of "+($1 $2 $3)"/"+($1 $2)"
        {"+(1 1 1 1 1)", "+(1 1)"},           // '+(1 +)' count 3 vs 0
        {"+(1 1)", "+(1 1 1 1 1)"},           // swapped
        {"+(1 1 1)", "+(1 1 1 1 1 1 1)"},     // second has MANY more of a shared branch
        {"+(1 1 1 1 1 1 1)", "+(1 1 1)"},     // swapped
    };
    printf("  \"similarity\": [\n");
    bool first = true;
    for (auto& c : cases) {
        size_t dist = tree_similarity(c.a, c.b);
        if (!first) printf(",\n");
        first = false;
        printf("    {\"a\": \"%s\", \"b\": \"%s\", \"dist\": %zu}",
               json_escape(c.a).c_str(), json_escape(c.b).c_str(), dist);
    }
    printf("\n  ]\n");
}

int main()
{
    printf("{\n");
    dump_flatten();
    dump_similarity();
    printf("}\n");
    return 0;
}
