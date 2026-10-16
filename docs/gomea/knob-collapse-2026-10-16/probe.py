import sys, collections
sys.setrecursionlimit(100000)
import pymoses; assert "pyMoses-knobs" in pymoses.__file__
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.moses.moses.complexity import tree_complexity
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.combo.vertex import combo_tree, argument
red = logical_reduction(set())(2)
n = sys.argv[1]
s = open(n + '.ex').read().strip()
base = str_to_combo_tree(s); c0 = tree_complexity(base)
nodes = [nd for nd in base.iter_pre_at(base.begin()) if str(nd.data) in ('and','or') or 'and' in repr(nd.data) or 'or' in repr(nd.data)]
# enumerate junctors by preorder index
pre = list(base.iter_pre_at(base.begin()))
tot = dis = 0; examples = collections.Counter(); shown=0
for i, nd in enumerate(pre):
    if nd.first_child is None: continue
    if 'not' in repr(nd.data): continue
    for k in range(1, 12):
        for neg in (False, True):
            t = str_to_combo_tree(s)
            p = list(t.iter_pre_at(t.begin()))[i]
            a = argument(k)
            if neg: a.negate()
            t.append_child(p, a)
            ic = tree_complexity(t)
            r = combo_tree(t.begin()); red(r)
            rc = tree_complexity(r)
            tot += 1
            if ic > rc:
                dis += 1
                if rc < c0 and shown < 3:
                    shown += 1
                    print('  junctor', i, 'add', ('!' if neg else '') + f'${k}', 'cx', ic, '->', rc, '(exemplar', c0, ')')
                    print('     reduced:', tree_to_stream_str(r))
                examples['below_exemplar' if rc < c0 else 'eq_exemplar' if rc == c0 else 'other'] += 1
print(n, 'exemplar cx', c0, 'literal-add probes', tot, 'reduced', dis, dict(examples))
