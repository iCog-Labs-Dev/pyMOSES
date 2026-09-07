// tools/cpp_probe/meta_rules_probe.cc
//
// Measures the REAL node-visit order and fixpoint termination of
// opencog::reduct's meta-rule combinators (downwards, upwards, sequential,
// iterative, assum_iterative) for pymetta's comboreduct/reduct/meta_rules.py
// port (task-2). Ground truth, not inference.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/meta_rules_probe tools/cpp_probe/meta_rules_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/meta_rules_probe )

#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/reduct/meta_rules.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/combo/assumption.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::reduct;
using namespace opencog::combo;
using namespace std;

// ---------------------------------------------------------------------
// record_visit: non-mutating probe rule. Appends a tag plus the printed
// subtree at the point of application to a shared log. Clones share the
// same underlying vector (held by reference), so wrapping this in
// downwards/upwards/sequential/etc. (which each clone their wrapped rule
// on construction) still all log into ONE sequence.
// ---------------------------------------------------------------------
struct record_visit : public crule<record_visit> {
    string tag;
    vector<string>& log;
    record_visit(string tag_, vector<string>& log_)
        : crule<record_visit>::crule("record_visit"), tag(tag_), log(log_) {}
    void operator()(combo_tree& tr, combo_tree::iterator it) const {
        combo_tree sub(it);
        ostringstream oss;
        ostream_combo_tree(oss, sub);
        log.push_back(tag + oss.str());
    }
};

static void print_log(const string& label, const vector<string>& log) {
    cout << label << ": [";
    for (size_t i = 0; i < log.size(); ++i) {
        if (i) cout << ", ";
        cout << log[i];
    }
    cout << "]" << endl;
}

// ---------------------------------------------------------------------
// step_arg: mutating probe rule for `iterative`'s fixpoint. Bumps an
// argument's index by 1 each call, until it reaches `cap`, then no-ops.
// ---------------------------------------------------------------------
struct step_arg : public crule<step_arg> {
    int cap;
    int& calls;
    step_arg(int cap_, int& calls_)
        : crule<step_arg>::crule("step_arg"), cap(cap_), calls(calls_) {}
    void operator()(combo_tree& tr, combo_tree::iterator it) const {
        ++calls;
        argument* a = boost::get<argument>(&*it);
        if (a && a->idx < cap) {
            *it = argument(a->idx + 1);
        }
    }
};

// ---------------------------------------------------------------------
// step_target: mutating probe rule for the NON-ROOT `it` question --
// always bumps a FIXED target iterator (captured at construction,
// independent of whatever `it` the combinator actually passes),
// regardless of the point of application. Used to check what
// `iterative`/`assum_iterative` do when the wrapped rule mutates
// something OUTSIDE `it`'s own subtree.
// ---------------------------------------------------------------------
struct step_target : public crule<step_target> {
    combo_tree::iterator target;
    int cap;
    int& calls;
    step_target(combo_tree::iterator target_, int cap_, int& calls_)
        : crule<step_target>::crule("step_target"),
          target(target_), cap(cap_), calls(calls_) {}
    void operator()(combo_tree& tr, combo_tree::iterator it) const {
        ++calls;
        argument* a = boost::get<argument>(&*target);
        if (a && a->idx < cap) {
            *target = argument(a->idx + 1);
        }
    }
};

// ---------------------------------------------------------------------
// grow_assumptions: mutating probe rule for `assum_iterative`'s fixpoint.
// Inserts one new (distinct) assumption per call until `tr` already has
// `cap` assumptions, then no-ops. Never touches the main tree.
// ---------------------------------------------------------------------
struct grow_assumptions : public crule<grow_assumptions> {
    int cap;
    int& calls;
    int next_idx;
    grow_assumptions(int cap_, int& calls_, int start_idx)
        : crule<grow_assumptions>::crule("grow_assumptions"),
          cap(cap_), calls(calls_), next_idx(start_idx) {}
    void operator()(combo_tree& tr, combo_tree::iterator it) const {
        ++calls;
        int count = tr.number_of_siblings(tr.begin());
        if (count < cap) {
            combo_tree a{argument(next_idx)};
            const_cast<grow_assumptions*>(this)->next_idx++;
            insert_assumption(tr, a.begin());
        }
    }
};

static int assumption_count(combo_tree& tr) {
    return tr.number_of_siblings(tr.begin());
}

// ---------------------------------------------------------------------
// insert_sibling_once: mutating probe rule for the LIVE-MUTATION hazard
// during a downwards pass. The first time it sees an argument with
// idx==2, inserts a brand-new sibling ($99) immediately after it, and
// records the visit sequence -- to see whether the newly-inserted node
// is itself picked up by the SAME still-running downwards traversal
// (live re-derivation) or skipped (a snapshot would skip it).
// ---------------------------------------------------------------------
struct insert_sibling_once : public crule<insert_sibling_once> {
    bool& done;
    vector<string>& log;
    insert_sibling_once(bool& done_, vector<string>& log_)
        : crule<insert_sibling_once>::crule("insert_sibling_once"),
          done(done_), log(log_) {}
    void operator()(combo_tree& tr, combo_tree::iterator it) const {
        combo_tree sub(it);
        ostringstream oss;
        ostream_combo_tree(oss, sub);
        log.push_back(oss.str());
        argument* a = boost::get<argument>(&*it);
        if (a && a->idx == 2 && !done) {
            const_cast<insert_sibling_once*>(this)->done = true;
            tr.insert_after(it, vertex(argument(99)));
        }
    }
};

int main()
{
    // ---- 1. downwards: pre-order, parent before children -------------
    {
        combo_tree tr = str2combo_tree("and($1 or($2 $3) not($4))", vector<string>());
        vector<string> log;
        const rule& r_down = downwards(record_visit("", log));
        r_down(tr);
        print_log("downwards", log);
    }

    // ---- 2. upwards: post-order, children before parent ---------------
    {
        combo_tree tr = str2combo_tree("and($1 or($2 $3) not($4))", vector<string>());
        vector<string> log;
        const rule& r_up = upwards(record_visit("", log));
        r_up(tr);
        print_log("upwards", log);
    }

    // ---- 3. sequential: declaration order, same point of application --
    {
        combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
        vector<string> log;
        const rule& r_seq = sequential(record_visit("A:", log), record_visit("B:", log), record_visit("C:", log));
        r_seq(tr);
        print_log("sequential", log);
    }

    // ---- 4. downwards(sequential(...)): interleaving of the two -------
    {
        combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
        vector<string> log;
        const rule& r_ds = downwards(sequential(record_visit("A:", log), record_visit("B:", log)));
        r_ds(tr);
        print_log("downwards(sequential)", log);
    }

    // ---- 5. iterative: fixpoint at the point of application -----------
    {
        combo_tree tr = str2combo_tree("$1", vector<string>());
        int calls = 0;
        const rule& r_it = iterative(step_arg(3, calls));
        r_it(tr);
        ostringstream oss;
        ostream_combo_tree(oss, tr);
        cout << "iterative: calls=" << calls << " final=" << oss.str() << endl;
    }

    // ---- 6. assum_iterative: fixpoint over BOTH main tree and ----------
    //         assumption set. Pre-seed with one assumption so the
    //         either-side-zero quirk in equal_assumptions never masks a
    //         real change (see assumption.py's own documented quirk).
    {
        combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
        combo_tree seed(argument(9));
        insert_assumption(tr, seed.begin());
        cout << "assum_iterative: initial assumption count="
             << assumption_count(tr) << endl;
        int calls = 0;
        const rule& r_ai = assum_iterative(grow_assumptions(3, calls, 10));
        r_ai(tr);
        cout << "assum_iterative: calls=" << calls
             << " final assumption count=" << assumption_count(tr) << endl;
    }

    // ---- 7. assum_iterative sanity check: if assumptions never grew
    //         past the pre-seeded 1 (a no-op rule), it must terminate
    //         after exactly one call (both sides always equal).
    {
        combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
        combo_tree seed(argument(9));
        insert_assumption(tr, seed.begin());
        int calls = 0;
        const rule& r_ai0 = assum_iterative(grow_assumptions(0, calls, 10));  // cap=0: instant no-op
        r_ai0(tr);
        cout << "assum_iterative(no-op rule): calls=" << calls
             << " final assumption count=" << assumption_count(tr) << endl;
    }

    // ---- 8. downwards live-mutation hazard: does inserting a new
    //         sibling mid-walk get visited by the SAME downwards pass? ---
    {
        combo_tree tr = str2combo_tree("and($1 $2 $3)", vector<string>());
        vector<string> log;
        bool done = false;
        const rule& r_lm = downwards(insert_sibling_once(done, log));
        r_lm(tr);
        print_log("downwards(live-mutation insert-after)", log);
        ostringstream oss;
        ostream_combo_tree(oss, tr);
        cout << "downwards(live-mutation) final tree=" << oss.str() << endl;
    }

    // ---- 9. iterative at a NON-ROOT it, rule mutates a SIBLING (outside
    //         it's own subtree): iterative's fixpoint check is local to
    //         `it` (`tr.equal_subtree(it, tmp.begin())`), so it must not
    //         even notice the sibling changing -- stop after 1 call.
    {
        combo_tree tr(vertex(id::logical_and));
        combo_tree::iterator root = tr.begin();
        combo_tree::iterator c1 = tr.append_child(root, vertex(argument(1)));
        combo_tree::iterator c2 = tr.append_child(root, vertex(argument(2)));
        tr.append_child(root, vertex(argument(3)));
        int calls = 0;
        const rule& r_it_nr = iterative(step_target(c1, 4, calls));
        r_it_nr(tr, c2);  // it = c2 (middle child, non-root); rule mutates c1
        ostringstream oss;
        ostream_combo_tree(oss, tr);
        cout << "iterative(non-root it, mutate sibling): calls=" << calls
             << " final=" << oss.str() << endl;
    }

    // ---- 10. assum_iterative at the SAME non-root it, SAME sibling-only
    //          mutation: assum_iterative's check is the WHOLE tree at
    //          tr.begin() regardless of `it`, so it DOES see the sibling
    //          change and keeps iterating until it stops.
    {
        combo_tree tr(vertex(id::logical_and));
        combo_tree::iterator root = tr.begin();
        combo_tree::iterator c1 = tr.append_child(root, vertex(argument(1)));
        combo_tree::iterator c2 = tr.append_child(root, vertex(argument(2)));
        tr.append_child(root, vertex(argument(3)));
        int calls = 0;
        const rule& r_ai_nr = assum_iterative(step_target(c1, 4, calls));
        r_ai_nr(tr, c2);  // it = c2 (middle child, non-root); rule mutates c1
        ostringstream oss;
        ostream_combo_tree(oss, tr);
        cout << "assum_iterative(non-root it, mutate sibling): calls=" << calls
             << " final=" << oss.str() << endl;
    }

    // ---- 11. downwards TYPED dispatch (task-5): does filtering on
    //          id::boolean_type skip bare argument leaves? An argument's
    //          get_output_type_tree is ALWAYS unknown_type in isolation
    //          (type_tree.cc:253-254), so a typed downwards(r, boolean_type)
    //          over "and($1 $2)" should visit ONLY the root -- never $1/$2.
    {
        combo_tree tr = str2combo_tree("and($1 $2)", vector<string>());
        vector<string> log;
        const rule& r_typed = downwards(record_visit("", log), id::boolean_type);
        r_typed(tr);
        print_log("downwards(typed boolean_type)", log);
    }

    // ---- 12. downwards TYPED dispatch over a tree with a nested `or` --
    //          `or` also resolves to boolean_type, so it must ALSO be
    //          visited (not just the root `and`), while the leaves still
    //          are not.
    {
        combo_tree tr = str2combo_tree("and($1 or($2 $3))", vector<string>());
        vector<string> log;
        const rule& r_typed2 = downwards(record_visit("", log), id::boolean_type);
        r_typed2(tr);
        print_log("downwards(typed boolean_type, nested or)", log);
    }

    return 0;
}
