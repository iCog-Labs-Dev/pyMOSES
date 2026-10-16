import sys, collections
sys.setrecursionlimit(100000)
import pymoses; assert "pyMoses-knobs" in pymoses.__file__
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.moses.moses.complexity import tree_complexity
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.reduct.clean_reduction import clean_reduce
from pymoses.comboreduct.combo.vertex import combo_tree
import pymoses.moses.representation.build_knobs as bk
from pymoses.moses.representation.representation import representation
red = logical_reduction(set())(2)
recs = []
orig = bk.build_knobs.disc_probe
def dp(self, subtree, knob):
    if self._skip_disc_probe: return True
    knob.turn(0)
    c0 = tree_complexity(subtree, bk.is_predicate)
    r = combo_tree(subtree); clean_reduce(r); cl = tree_to_stream_str(r); self._simplify_knob_building(r)
    c0r = tree_complexity(r, bk.is_predicate)
    res = orig(self, subtree, knob)
    recs.append((c0, c0r, res, tree_to_stream_str(r) if c0r < c0 else None, cl))
    return res
bk.build_knobs.disc_probe = dp
from pymoses.moses.main.moses_exec import moses_exec
n = sys.argv[1]
ex = open(f"/tmp/claude-1000/knobs-investigation/{n}.ex").read().strip()
moses_exec(["moses", "-H", "mux", "-k", "3", "-r", "1", "-e", ex, "-a", "hc", "-m", "1", "-l", "WARN", "-f", f"/tmp/claude-1000/knobs-investigation/instr_{n}.log"])
built = sum(r[2] for r in recs)
print(n, "probes", len(recs), "built", built)
first = next((i for i, r in enumerate(recs) if r[1] < r[0]), None)
print("first probe whose BASELINE (knob at 0) reduces:", first)
seg = collections.Counter((r[1] < r[0], r[2]) for r in recs)
print("(baseline_reduces, built) counts:", dict(seg))
if first is not None:
    print("baseline cx", recs[first][0], "->", recs[first][1], recs[first][3])
    # runs
    print("built before first:", sum(r[2] for r in recs[:first]), "built after:", sum(r[2] for r in recs[first:]))

if first is not None:
    print("clean before:", recs[first-1][4])
    print("clean at    :", recs[first][4])
    print("exemplar    :", ex)
