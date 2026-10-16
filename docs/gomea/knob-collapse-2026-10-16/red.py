import pymoses; assert "pyMoses-knobs" in pymoses.__file__, pymoses.__file__
import json, sys
sys.setrecursionlimit(100000)
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.moses.moses.complexity import tree_complexity
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.combo.vertex import combo_tree
lr = logical_reduction(set())
red = lr(2)
for f in ['g_mux_3001','g_mux_3002','g_mux_3003']:
    sel = {}; cre = {}
    for l in open(f'/tmp/claude-1000/exemplar-growth/runs/{f}.jsonl'):
        d = json.loads(l)
        if d['kind']=='select': sel[d['exp']] = d
        if d['kind']=='create': cre[d['exp']] = d
    for e in sorted(cre):
        t = str_to_combo_tree(sel[e]['tree'])
        c0 = tree_complexity(t); n0 = t.size()
        t2 = combo_tree(t.begin()); red(t2)
        print(f, e, n0, c0, cre[e]['knobs'][0], '-> reduced', t2.size(), tree_complexity(t2), 'REDUCIBLE' if tree_complexity(t2) < c0 else '')
