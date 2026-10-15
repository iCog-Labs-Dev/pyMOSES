import json, glob, statistics as st, sys
sys.path.insert(0, '/tmp/claude-1000/exemplar-growth')
from analyse import table

R = '/tmp/claude-1000/exemplar-growth/runs/'


def agg(prefix, ths):
    names = sorted(f.split('/')[-1][:-6] for f in glob.glob(R + prefix + '_*.jsonl') if 'events' not in f)
    print(f'===== {prefix}: {names}')
    tabs = {n: table(n) for n in names}
    # final state
    for n, rows in tabs.items():
        if rows:
            l = rows[-1]
            print(f'  {n}: expansions={len(rows)} evals={l["evals"]} last exemplar nodes={l["nodes"]} cx={l["cx"]} raw={l["raw"]} knobs={l["knobs"]} '
                  f'deme_best raw={l["best_raw"]} cx={l["best_cx"]}; sum createCPU={sum(r["create_cpu"] for r in rows):.0f}')
    # exemplar cx at first exemplar with raw >= th
    for th in ths:
        v = []
        for rows in tabs.values():
            for r in rows:
                if r['raw'] >= th:
                    v.append((r['cx'], r['nodes'], r['knobs'], r['evals'] - 0))
                    break
        if v:
            print(f'  first exemplar raw>={th}: n={len(v)} cx={[x[0] for x in v]} nodes={[x[1] for x in v]} knobs={[x[2] for x in v]}')
    # efficiency per expansion: cx gained per raw gained (deme best vs exemplar)
    dcx = draw = 0
    steps = []
    acc_merged = []
    for rows in tabs.values():
        for r in rows:
            dcx += r['best_cx'] - r['cx']
            draw += r['best_raw'] - r['raw']
            steps += r['traj']
            acc_merged.append(r['merged'])
    print(f'  per-deme: sum dcx={dcx} sum draw={draw} -> literals per 100 rows gained = {100 * dcx / max(draw, 1):.2f}')
    if not steps:
        return
    up = [s for s in steps if s['dcx'] > 0]
    print(f'  running-best steps: n={len(steps)} cx_up={len(up)} mean dcx={st.mean([s["dcx"] for s in steps]):.2f} '
          f'mean draw={st.mean([s["draw"] for s in steps]):.1f} median draw={st.median([s["draw"] for s in steps])} '
          f'mean ham={st.mean([s["ham"] for s in steps]):.2f} ham>=3: {sum(s["ham"] >= 3 for s in steps)}  '
          f'steps with draw<=8: {sum(s["draw"] <= 8 for s in steps)}')
    print(f'  merged per expansion: mean={st.mean(acc_merged):.1f} median={st.median(acc_merged)} total={sum(acc_merged)}')
    # GOM acceptance
    tot = {}
    for rows in tabs.values():
        for r in rows:
            for k, d in (r['gom'] or {}).items():
                t = tot.setdefault(k, {"n": 0, "cx_up": 0, "cx_eq": 0, "cx_down": 0, "sum_dcx": 0, "sum_ham": 0, "ham_hist": {}})
                for kk in ("n", "cx_up", "cx_eq", "cx_down", "sum_dcx", "sum_ham"):
                    t[kk] += d[kk]
                for hb, c in d['ham_hist'].items():
                    t['ham_hist'][hb] = t['ham_hist'].get(hb, 0) + c
    for k, t in sorted(tot.items()):
        print(f'  GOM {k}: n={t["n"]} cx_up={t["cx_up"]} cx_eq={t["cx_eq"]} cx_down={t["cx_down"]} mean dcx={t["sum_dcx"] / t["n"]:.2f} '
              f'mean ham={t["sum_ham"] / t["n"]:.2f} hist={dict(sorted(t["ham_hist"].items()))}')


if __name__ == '__main__':
    agg('g_mux', [-768, -512, -384, -256, -128, -64, 0])
    agg('hc_mux', [-768, -512, -384, -256, -128, -64, 0])
    agg('g_maj', [-12, -8, -6, -4, -2, 0])
    agg('hc_maj', [-12, -8, -6, -4, -2, 0])
