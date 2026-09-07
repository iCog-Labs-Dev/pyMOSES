// tools/cpp_probe/reduct_effort_probe.cc
//
// Oracle probe for Task 6's exit criterion: at logical-reduction efforts
// 1 (simple), 2 (medium) and 3 (complexe), pyMeTTa's reduced tree must be
// byte-identical to classic MOSES's own `logical_reduce()` output. The
// before/after truth-table result is diagnostic data to reproduce, not a
// universal preservation requirement: classic itself has known lossy cases.
//
// Mirrors $CLASSIC/tests/comboreduct/reduct/test_reduct.cc (a local,
// dev-only fork of the upstream cxxtest probe -- see also
// moses/comboreduct/main/logical-reductor.cc, which hardcodes effort=3):
// same `logical_reduce(effort, tr, ignore_ops)` call and the same
// before/after complete_truth_table comparison, made batch- and
// effort-parametric, with JSON-lines output instead of test_reduct's
// human-readable printf block.
//
// Effort levels ($CLASSIC's moses/comboreduct/reduct/logical_reduction.cc,
// logical_reduction::operator()(int)): 0 extra_simple, 1 simple, 2 medium
// (requires ignore_ops -- always empty here, matching every corpus source
// used by reduct_corpus.py), 3 complexe. Task 6's exit criterion exercises
// 1, 2 and 3 only; effort 0 is out of scope and not captured.
//
// Reads combo-tree text-format strings, one per line, from a file (see
// reduct_corpus.py, which writes that file) and for the single effort
// given on argv[2], runs logical_reduce over every line and prints one
// JSON-lines object per input into a JSON array.
//
// Per-line, THREE INDEPENDENT phases, each in its own try/catch (Task 6
// review, Important finding 3 -- a single shared try/catch used to let a
// POST-hoc abort masquerade as "logical_reduce never ran", discarding a
// `reduced` value that genuinely existed):
//   1. `complete_truth_table tt1(tr)` on the ORIGINAL tree, purely to
//      learn the arity to re-evaluate the reduced tree at. Can throw
//      (`builtin_to_bool`, vertex.h:708-713, OC_ASSERTs whenever a
//      whole-tree evaluation's root is a bare childless and()/or()) if
//      the INPUT itself has that shape -- a "pre-reduction" abort.
//   2. `logical_reduce(effort, tr, ignore_ops)` -- the thing actually
//      under test. `"ok"` in the JSON output reflects ONLY this phase.
//   3. A post-hoc `complete_truth_table` on the REDUCED tree, purely
//      this probe's own `tt_preserved` diagnostic -- NOT part of
//      logical_reduce's own contract. Can independently throw the same
//      way, on the REDUCED tree's root, without affecting `"ok"` or
//      `"reduced"` (both already committed by phase 2 before phase 3
//      even starts) -- see `"tt_computed"`/`"tt_error"` in the output
//      schema below.
//
// Output schema, per JSON-lines object:
//   always: "input", "effort", "ok" (phase 2 alone).
//   if ok:  "reduced", "size_before", "size_after", "tt_computed"
//           (phase 3 alone), then either "tt_preserved" (if tt_computed)
//           or "tt_error" (if not -- phase 1 or phase 3 aborted; phase 2
//           still ran to completion and "reduced" above is still real).
//   if not ok: "error" (the parse step, or phase 2 itself).
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/reduct_effort_probe \
//       tools/cpp_probe/reduct_effort_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 2097152; timeout 300 \
//     tools/cpp_probe/build/reduct_effort_probe corpus.txt 3 )

#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/table/table.h>
#include <opencog/util/numeric.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

// Preserve ostream_combo_tree's own trailing space verbatim -- this
// fixture is meant to be byte-compared against pymetta's
// combo_tree_to_str, which reproduces that exact trailing space (see
// tests/fixtures/combo_roundtrip.json's committed entries). Do NOT
// `.strip()` here, unlike logical_rules_differential_probe.cc's `show()`
// (that probe feeds a different, already-.strip()'d Python comparison).
static string show(combo_tree& tr)
{
    ostringstream oss;
    ostream_combo_tree(oss, tr);
    return oss.str();
}

static string json_quote(const string& s)
{
    ostringstream oss;
    oss << "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') oss << '\\';
        else if (c == '\n') { oss << "\\n"; continue; }
        oss << c;
    }
    oss << "\"";
    return oss.str();
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        cerr << "usage: " << argv[0] << " <corpus-file> <effort 1|2|3>" << endl;
        return 2;
    }

    ifstream in(argv[1]);
    if (!in) {
        cerr << "cannot open " << argv[1] << endl;
        return 2;
    }
    int effort = atoi(argv[2]);

    // Empty ignore_ops throughout: every corpus source used by
    // reduct_corpus.py is a pure and/or/not/$n boolean tree with no
    // predicates to ignore -- same as test_reduct.cc, BooleanReductUTest
    // and BooleanFullUTest's own `vertex_set ignore_ops;`.
    vertex_set ignore_ops;

    string line;
    cout << "[" << endl;
    bool first = true;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (!first) cout << "," << endl;
        first = false;

        // `reduce_ok` is the ONLY thing "ok" means below: did
        // `logical_reduce` itself (the thing under test) run to
        // completion. It is set to true ONLY as the very last statement
        // of phase 2 (below), after `reduced`/`size_after` have both
        // already been captured successfully -- so there is no window
        // where `reduce_ok` is true but `reduced` is not yet a real
        // value.
        //
        // `tt_computed`/`tt_preserved`/`tt_error` are a SEPARATE outcome:
        // this probe's own post-hoc truth-table-preservation diagnostic
        // (phase 3), which is instrumentation ON TOP of logical_reduce,
        // not part of what it means for logical_reduce to have
        // succeeded. Task 6 review, Important finding 3: the previous
        // single try/catch conflated these two -- a phase-3-only throw
        // (the reduced tree's root being a bare childless and()/or(),
        // which `complete_truth_table`/`builtin_to_bool`, vertex.h:
        // 708-713, OC_ASSERTs on) was indistinguishable from
        // logical_reduce itself never having run, silently discarding a
        // `reduced` value that genuinely existed. Restructured into
        // three independent phases so a phase-3 (or phase-1) abort can
        // never retract phase 2's own result.
        bool reduce_ok = false;
        bool tt1_ok = false;
        bool tt_computed = false;
        bool tt_preserved = false;
        string error, reduced, tt_error;
        long size_before = -1, size_after = -1;

        combo_tree tr;
        complete_truth_table tt1;

        try {
            stringstream ss(line);
            ss >> tr;
        } catch (const std::exception& e) {
            error = string("ERROR (parse): ") + e.what();
        } catch (...) {
            error = "ERROR (parse): unknown exception";
        }

        if (!error.empty()) {
            // Parse itself failed -- nothing below can run.
        } else {
            size_before = (long)tr.size();

            // Phase 1: truth table of the ORIGINAL tree, needed only to
            // learn what arity to re-evaluate the REDUCED tree at
            // (`truth_table_preserved`'s own docstring on the Python
            // side: "before's OWN arity"). Kept in ITS OWN try/catch --
            // a "pre-reduction" throw here (the INPUT tree's root is
            // itself a bare childless and()/or()) must never be
            // confused with logical_reduce itself failing, nor does it
            // prevent phase 2 (logical_reduce does not need `tt1`) from
            // still running and reporting a real `reduced` value.
            try {
                tt1 = complete_truth_table(tr);
                tt1_ok = true;
            } catch (const std::exception& e) {
                tt_error = string("ERROR (pre-reduction complete_truth_table): ") + e.what();
            } catch (...) {
                tt_error = "ERROR (pre-reduction complete_truth_table): unknown exception";
            }

            // Phase 2: the thing actually under test. `reduce_ok` is set
            // true only once `reduced` itself is a valid, fully-computed
            // string -- see the comment above the variable declarations.
            try {
                logical_reduce(effort, tr, ignore_ops);
                size_after = (long)tr.size();
                reduced = show(tr);
                reduce_ok = true;
            } catch (const std::exception& e) {
                error = string("ERROR (logical_reduce): ") + e.what();
            } catch (...) {
                error = "ERROR (logical_reduce): unknown exception";
            }

            // Phase 3: the post-hoc truth-table-preservation diagnostic,
            // attempted only when phase 2 produced a reduced tree AND
            // phase 1 produced an original-arity table to compare it
            // against. Its own abort (same OC_ASSERT, now hit by the
            // REDUCED tree's root instead of the original's) only ever
            // populates `tt_error`/leaves `tt_computed` false -- it can
            // no longer retract `reduced`.
            if (reduce_ok && tt1_ok) {
                try {
                    complete_truth_table tt2(tr, integer_log2(tt1.size()));
                    tt_preserved = (tt1 == tt2);
                    tt_computed = true;
                } catch (const std::exception& e) {
                    tt_error = string("ERROR (post-hoc complete_truth_table): ") + e.what();
                } catch (...) {
                    tt_error = "ERROR (post-hoc complete_truth_table): unknown exception";
                }
            }
        }

        cout << "{\"input\": " << json_quote(line)
             << ", \"effort\": " << effort
             << ", \"ok\": " << (reduce_ok ? "true" : "false");
        if (reduce_ok) {
            cout << ", \"reduced\": " << json_quote(reduced)
                 << ", \"size_before\": " << size_before
                 << ", \"size_after\": " << size_after
                 << ", \"tt_computed\": " << (tt_computed ? "true" : "false");
            if (tt_computed) {
                cout << ", \"tt_preserved\": " << (tt_preserved ? "true" : "false");
            } else {
                cout << ", \"tt_error\": " << json_quote(tt_error);
            }
        } else {
            cout << ", \"error\": " << json_quote(error);
        }
        cout << "}";
    }
    cout << endl << "]" << endl;
    return 0;
}
