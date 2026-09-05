// tools/cpp_probe/remove_null_vertices_cond_crash_probe.cc
//
// Isolated probe (own binary, not general_rules_probe.cc) for a
// SPECIFIC crash scenario in opencog::reduct::remove_null_vertices's
// `cond` branch: a null_vertex landing in the trailing, UNPAIRED "else"
// slot. Run ONE labelled case per invocation (argv[1]) so a segfault in
// one case does not take out the others' results.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/remove_null_vertices_cond_crash_probe \
//       tools/cpp_probe/remove_null_vertices_cond_crash_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule; a SIGSEGV is expected
// for the "bad" cases and is not an OOM condition, so ulimit -v is not
// what protects against it -- timeout guards a hang, not a crash):
//   for case in bad1 bad2 good1 good2; do
//     ( ulimit -v 1048576; timeout 20 \
//       tools/cpp_probe/build/remove_null_vertices_cond_crash_probe $case )
//     echo "exit: $?"
//   done
#include <moses/comboreduct/reduct/general_rules.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;

static std::string show(combo_tree& tr)
{
    std::ostringstream oss;
    ostream_combo_tree(oss, tr);
    return oss.str();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <bad1|bad2|good1|good2>\n", argv[0]);
        return 2;
    }
    remove_null_vertices rnv;
    std::string which = argv[1];

    if (which == "bad1") {
        // cond($1 $2 null) -- null in the trailing (unpaired) else slot.
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(argument(2)));
        tr.append_child(root, vertex(id::null_vertex));
        rnv(tr, root);
        printf("bad1 completed (no crash): %s\n", show(tr).c_str());
    } else if (which == "bad2") {
        // cond(null $1 null) -- a genuine null condition (pos0, paired
        // with $1) PLUS a second null in the trailing else slot (pos2).
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(id::null_vertex));
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(id::null_vertex));
        rnv(tr, root);
        printf("bad2 completed (no crash): %s\n", show(tr).c_str());
    } else if (which == "good1") {
        // cond($1 null $2) -- null as a CONSEQUENT, not touched by the
        // null-check at all (only conditions are ever tested).
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(id::null_vertex));
        tr.append_child(root, vertex(argument(2)));
        rnv(tr, root);
        printf("good1 completed: %s\n", show(tr).c_str());
    } else if (which == "good2") {
        // cond($1 $2 $3 null) -- null in the trailing else slot, but an
        // EVEN number of (condition,consequent) pairs before it, so the
        // pair-stride walk lands on null at an ODD overall position that
        // this shape happens to make safe (it is visited as a
        // CONSEQUENT-position skip, not as a condition) -- included as a
        // contrasting "looks similar, does not crash" case.
        combo_tree tr(vertex(id::cond));
        combo_tree::iterator root = tr.begin();
        tr.append_child(root, vertex(argument(1)));
        tr.append_child(root, vertex(argument(2)));
        tr.append_child(root, vertex(argument(3)));
        tr.append_child(root, vertex(id::null_vertex));
        rnv(tr, root);
        printf("good2 completed: %s\n", show(tr).c_str());
    } else {
        fprintf(stderr, "unknown case %s\n", which.c_str());
        return 2;
    }
    return 0;
}
