import json,glob,statistics as st,sys
base='/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/gomea-results/1/'
for arm in ['HC','LTB-C250-M']:
    runs=[]
    for f in sorted(glob.glob(base+f'phase1e-stage2-*/runs/mux11__{arm}__s*/trace.jsonl')):
        runs.append([json.loads(l) for l in open(f)])
    print(arm,'nruns',len(runs))
    # complexity of best-so-far at eval checkpoints
    for n in [500,1000,2000,3000,4000,6000]:
        cx=[];raw=[]
        for r in runs:
            last=None
            for e in r:
                if e['n']<=n: last=e
            if last: cx.append(last['cpx']); raw.append(last['raw'])
        print(f'  evals<={n}: median cpx {st.median(cx)}  median raw {st.median(raw)}')
    # complexity at first time raw >= threshold
    for th in [-512,-256,-128,-64,-32,-16,0]:
        cx=[];ns=[]
        for r in runs:
            for e in r:
                if e['raw']>=th: cx.append(e['cpx']); ns.append(e['n']); break
        if cx: print(f'  first raw>={th}: n={len(cx)} median cpx {st.median(cx)} mean {st.mean(cx):.1f} median evals {st.median(ns)}')
