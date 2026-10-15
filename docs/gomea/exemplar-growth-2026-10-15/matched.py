import sys, glob
sys.path.insert(0, '/tmp/claude-1000/exemplar-growth')
from analyse import table

R = '/tmp/claude-1000/exemplar-growth/runs/'


def at(name, E):
    rows = table(name)
    done = [r for r in rows if r['evals'] <= E]
    if not rows or rows[-1]['evals'] < E and not open(R + name + '.jsonl').read().count('"done"'):
        tag = '*'  # run not yet at E
    else:
        tag = ''
    if not done:
        return None
    last = done[-1]
    best = max(done, key=lambda r: (r['best_raw'], -r['best_cx']))
    return dict(tag=tag, exps=len(done), evals=last['evals'], ex_nodes=last['nodes'], ex_cx=last['cx'], knobs=last['knobs'],
                best_raw=best['best_raw'], best_cx=best['best_cx'], best_nodes=best['best_nodes'],
                createcpu=round(sum(r['create_cpu'] for r in done)), max_knobs=max(r['knobs'] for r in done))


for E in map(int, sys.argv[1:]):
    print(f'--- at <= {E} evals')
    print('run | demes evals | last exemplar nodes/cx knobs | best raw cx nodes | sum knob-build CPU s')
    for pre in ['hc_mux', 'g_mux', 'gz0125_mux', 'gz0125_mux6k', 'gz05_mux', 'hc_maj', 'g_maj']:
        for f in sorted(glob.glob(R + pre + '_*.jsonl')):
            if 'events' in f:
                continue
            n = f.split('/')[-1][:-6]
            a = at(n, E)
            if a:
                print(f"{n}{a['tag']} | {a['exps']} {a['evals']} | {a['ex_nodes']}/{a['ex_cx']} {a['knobs']} | {a['best_raw']} {a['best_cx']} {a['best_nodes']} | {a['createcpu']}")
