import sys
sys.setrecursionlimit(100000)
import pymoses; assert "pyMoses-knobs" in pymoses.__file__
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree, tree_to_stream_str
from pymoses.moses.moses.complexity import tree_complexity
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.reduct.clean_reduction import clean_reduce
from pymoses.comboreduct.combo.vertex import combo_tree
lr = logical_reduction(set())
for n in ['n3001_6','c3001_7','n3002_10','c3002_11','c3003_11']:
    s = open('/tmp/claude-1000/knobs-investigation/'+n+'.ex').read().strip()
    out = [n]
    for eff in (2,3):
        t = str_to_combo_tree(s); lr(eff)(t); out.append(f'raw e{eff}:{tree_complexity(t)}')
    t = str_to_combo_tree('or(' + s + ')'); clean_reduce(t); lr(2)(t); out.append(f'or(X) e2:{tree_complexity(t)}')
    print(*out)
