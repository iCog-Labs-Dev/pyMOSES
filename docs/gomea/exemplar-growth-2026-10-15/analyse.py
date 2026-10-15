import json, sys, glob, os, statistics as st
from collections import defaultdict

R = '/tmp/claude-1000/exemplar-growth/runs/'


def load(name):
    recs = [json.loads(l) for l in open(R + name + '.jsonl')]
    exps = defaultdict(dict)
    for r in recs:
        if r['kind'] in ('select', 'create', 'merge'):
            exps[r['exp']][r['kind']] = r
    return [exps[k] for k in sorted(exps)]


def table(name):
    rows = []
    cum = 0
    for e in load(name):
        s, c, m = e.get('select'), e.get('create'), e.get('merge')
        if not (s and c and m):
            continue
        d = m['demes'][0]
        cum += d['n_evals']
        rows.append(dict(exp=s['exp'], evals=cum, nodes=s['nodes'], cx=s['cx'], raw=s['raw'], pen=round(s['pen'], 2),
                         knobs=c['knobs'][0], create_cpu=c['create_cpu'], deme_n=d['n'],
                         best_raw=d['best']['raw'], best_cx=d['best']['cx'], best_nodes=d['best']['nodes'],
                         merged=m.get('new', 0), merged_cx_med=(m['new_cx'][len(m['new_cx']) // 2] if m.get('new_cx') else None),
                         merged_cx_max=(max(m['new_cx']) if m.get('new_cx') else None), mp=m['mp_size'],
                         near=s['n_near_best'], traj=d['traj'], gom=m.get('gom')))
    return rows


def show(name):
    rows = table(name)
    print(f'### {name}')
    print('exp evals | ex_nodes ex_cx ex_raw ex_pen | knobs createCPU | deme_best raw/cx/nodes | merged med/max cx | mp')
    for r in rows:
        print(f"{r['exp']:>3} {r['evals']:>5} | {r['nodes']:>4} {r['cx']:>3} {r['raw']:>7} {r['pen']:>8} | {r['knobs']:>5} {r['create_cpu']:>6} | "
              f"{r['best_raw']:>7}/{r['best_cx']:>3}/{r['best_nodes']:>3} | {r['merged']:>4} {r['merged_cx_med']}/{r['merged_cx_max']} | {r['mp']}")
    return rows


if __name__ == '__main__':
    for n in sys.argv[1:]:
        show(n)
