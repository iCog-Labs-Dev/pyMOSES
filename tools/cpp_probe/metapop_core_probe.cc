// tools/cpp_probe/metapop_core_probe.cc
//
// Oracle probe for pyMeTTa S10 Task 2 (metapopulation core: init,
// best_candidates, best_tree, best_composite_score, get_trees, size,
// update_best_candidates).
//
// metapopulationUTest.cxxtest (the slice's registered cxxtest oracle)
// exercises only metapopulation::remove_dominated/get_nondominated_disjoint
// -- domination.cc code that lives outside every file this whole S10 slice
// touches (deme_expander.{h,cc}, deme_params.h, metapopulation.{h,cc},
// merging.cc, metapop_params.{h,cc} -- see the S10 plan's own file list).
// It exercises none of Task 2's own scope. This probe fills that gap for
// the observable PUBLIC surface: the constructor (both the single-tree and
// vector<combo_tree> overloads), init()'s default-path candidate scoring +
// insertion, and update_best_candidates()'s tie-break-by-complexity branch
// (public in the header, called directly here with a hand-built
// scored_combo_tree_set). rescore() and init() itself are private in
// metapopulation.h and not friended to anything but ::metapopulationUTest,
// so they are not probed directly here -- they are ported straight from
// merging.cc/metapopulation.cc's own source (see metapopulation.py's
// docstrings) and exercised with direct Python unit tests instead.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I "$C" -I "$P/include" \
//       -o tools/cpp_probe/build/metapop_core_probe \
//       tools/cpp_probe/metapop_core_probe.cc \
//       -L "$P/lib" -L "$P/lib/opencog" \
//       -Wl,-rpath,"$P/lib" -Wl,-rpath,"$P/lib/opencog" \
//       -lmoses -lcomboreduct -lcogutil
//
// Run, bounded (CARRY-FORWARD.md's standing OOM-safety rule):
//   ( ulimit -v 2097152; timeout 60 tools/cpp_probe/build/metapop_core_probe )

#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/moses/metapopulation/metapopulation.h>
#include <moses/moses/scoring/behave_cscore.h>
#include <opencog/util/mt19937ar.h>

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace combo;

namespace {

std::string show(const combo_tree &tr) {
  combo_tree t = tr;
  std::ostringstream oss;
  ostream_combo_tree(oss, t);
  return oss.str();
}

// Concrete bscore_base: score(tree) looked up by its printed form (an
// empty bscore vector isn't enough here, unlike metapopulationUTest's own
// dummy_bscore -- this probe needs DISTINCT, controllable scores per tree
// to exercise ordering/tie-break, so operator() returns a single-element
// bscore equal to the tree's assigned score; sum_bscore(bs) == that score
// exactly, since it's the only element). get_complexity is NOT overridden
// -- bscore_base's own default (tree_complexity(tr)) is used, so the
// complexity values below are measured, not asserted.
struct probe_bscore : public bscore_base {
  std::map<std::string, double> scores;

  behavioral_score operator()(const combo_tree &tr) const {
    std::string key = show(tr);
    auto it = scores.find(key);
    double s = (it == scores.end()) ? 0.0 : it->second;
    return {s};
  }
  behavioral_score best_possible_bscore() const { return {0.0}; }
};

void print_cscore(const composite_score &csc) {
  printf("{\"score\":%.9g,\"complexity\":%d,\"complexity_penalty\":%.9g,"
         "\"uniformity_penalty\":%.9g,\"penalized_score\":%.9g}",
         csc.get_score(), csc.get_complexity(), csc.get_complexity_penalty(),
         csc.get_uniformity_penalty(), csc.get_penalized_score());
}

} // namespace

int main() {
  randGen().seed(1);

  // IMPORTANT, measured: `metapopulation::_params`/`_filter_params` are
  // C++ REFERENCE members (metapopulation.h:532,534) bound directly to
  // whatever `pa`/`subp` argument is passed. Calling the constructor with
  // its DEFAULT `pa = metapop_parameters()` binds `_params` to a temporary
  // whose lifetime is NOT extended (lifetime extension applies only to a
  // local reference variable, never to a reference class MEMBER) -- so
  // `_params` dangles the instant the constructor call's full expression
  // ends. Confirmed empirically here: an earlier version of this probe
  // called `metapopulation mp(t1, csc);` (relying on the default `pa`) and
  // crashed inside the very next statement, `mp.best_composite_score()`,
  // with "Ensemble scoring not implemented" -- i.e. `_params.do_boosting`
  // read back `true` from already-destroyed stack memory, even though no
  // boosting was ever requested. Every metapopulation built below is
  // therefore given an explicit, named `metapop_parameters` that outlives
  // it.
  metapop_parameters pa;

  // Four distinct trees, distinct complexities by construction (nested
  // logical_and of increasing arity), fed fixed scores via probe_bscore
  // keyed on their printed form so their relative order is controlled by
  // this probe, not by the interpreter.
  vertex v_and(id::logical_and);
  vertex v_or(id::logical_or);

  combo_tree t1(vertex(argument(1))); // "t1": single argument, score 2.0
  combo_tree t2(v_and);
  t2.append_child(t2.begin(), vertex(argument(1)));
  t2.append_child(t2.begin(), vertex(argument(2))); // "and($1 $2)", score 5.0
  combo_tree t3(v_and);
  t3.append_child(t3.begin(), vertex(argument(1)));
  t3.append_child(t3.begin(), vertex(argument(2)));
  t3.append_child(t3.begin(),
                  vertex(argument(3))); // "and($1 $2 $3)", score 5.0 (tie)
  combo_tree t4(v_or);
  t4.append_child(t4.begin(), vertex(argument(1)));
  t4.append_child(t4.begin(), vertex(argument(2))); // "or($1 $2)", score 1.0

  probe_bscore bsc;
  bsc.scores[show(t1)] = 2.0;
  bsc.scores[show(t2)] = 5.0;
  bsc.scores[show(t3)] = 5.0;
  bsc.scores[show(t4)] = 1.0;

  behave_cscore csc(bsc);

  printf("{\n");

  // -------------------------------------------------------------------
  // Part 1: single-tree constructor overload.
  {
    metapopulation mp(t1, csc, pa);
    printf("  \"single_ctor\": {\"size\":%zu,\"best_tree\":%s,"
           "\"best_composite_score\":",
           mp.size(), ("\"" + show(mp.best_tree()) + "\"").c_str());
    print_cscore(mp.best_composite_score());
    printf("},\n");
  }

  // -------------------------------------------------------------------
  // Part 2: vector<combo_tree> constructor overload, tie-break exercised
  // through init() itself (t2 and t3 tie for the best raw score; t3 is
  // strictly more complex, so update_best_candidates must keep only t2).
  {
    std::vector<combo_tree> bases = {t1, t2, t3, t4};
    metapopulation mp(bases, csc, pa);

    printf("  \"multi_ctor\": {\"size\":%zu,\n", mp.size());

    printf("    \"trees_in_order\": [");
    bool first = true;
    for (const scored_combo_tree &sct : mp.get_trees()) {
      if (!first) printf(",");
      first = false;
      printf("{\"tree\":\"%s\",\"score\":%.9g,\"complexity\":%d}",
             show(sct.get_tree()).c_str(), sct.get_score(),
             sct.get_complexity());
    }
    printf("],\n");

    printf("    \"best_composite_score\":");
    print_cscore(mp.best_composite_score());
    printf(",\n");

    printf("    \"best_tree\":\"%s\",\n", show(mp.best_tree()).c_str());

    printf("    \"best_candidates\": [");
    first = true;
    for (const scored_combo_tree &sct : mp.best_candidates()) {
      if (!first) printf(",");
      first = false;
      printf("{\"tree\":\"%s\",\"score\":%.9g,\"complexity\":%d}",
             show(sct.get_tree()).c_str(), sct.get_score(),
             sct.get_complexity());
    }
    printf("]\n");

    printf("  },\n");
  }

  // -------------------------------------------------------------------
  // Part 3: update_best_candidates() called directly (it is public),
  // starting from a fresh metapopulation over a single low-scoring tree,
  // then fed a hand-built scored_combo_tree_set with a strictly-better
  // candidate followed (in insertion order) by a same-score, higher
  // complexity one -- to confirm the "insert but don't replace" tie-break
  // arm (the `cpx <= best_cpx` outer / `cpx < best_cpx` inner split).
  {
    metapopulation mp(t4, csc, pa); // best so far: t4, score 1.0

    scored_combo_tree_set more;
    more.insert(scored_combo_tree(t2, demeID_t(), composite_score(5.0f, 2)));
    more.insert(scored_combo_tree(t3, demeID_t(), composite_score(5.0f, 3)));
    mp.update_best_candidates(more);

    printf("  \"update_best_candidates_direct\": {\n");
    printf("    \"best_composite_score\":");
    print_cscore(mp.best_composite_score());
    printf(",\n");
    printf("    \"best_candidates_size\":%zu\n", mp.best_candidates().size());
    printf("  }\n");
  }

  printf("}\n");
  return 0;
}
