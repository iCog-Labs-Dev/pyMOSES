import sys, time, cProfile, signal, os
sys.setrecursionlimit(100000)
from pymoses.moses.main.moses_exec import moses_exec
from pymoses.moses.deme.deme_expander import deme_expander
from pymoses.moses.metapopulation.metapopulation import metapopulation
out = sys.argv[1]; snap_every = int(sys.argv[2]); args = sys.argv[3:]
log = open(out + '.demes.tsv', 'w', buffering=1)
log.write("deme\tphase\tcpu_s\tinfo\n")
state = {'n': 0}
def wrap(cls, name, info=lambda self, a, r: ''):
    f = getattr(cls, name)
    def w(self, *a, **k):
        t = time.process_time(); r = f(self, *a, **k)
        log.write(f"{state['n']}\t{name}\t{time.process_time()-t:.3f}\t{info(self,a,r)}\n")
        if name == 'merge_demes': state['n'] += 1
        return r
    setattr(cls, name, w)
def ci(self, a, r):
    try:
        from pymoses.comboreduct.combo.io import tree_to_stream_str
    except Exception:
        tree_to_stream_str = str
    try:
        reps = self._reps; return f"exemplar_len={len(str(a[0]))} dim={[rp.fields().raw_size() for rp in reps]}"
    except Exception as e: return repr(e)
def oi(self, a, r):
    try: return f"evals={sum(d.n_evals for ds in self._demes for d in ds)} sizes={[len(d) for ds in self._demes for d in ds]}"
    except Exception as e: return repr(e)
def mi(self, a, r):
    return f"mp_size={self.size()}"
wrap(deme_expander, 'create_demes', ci)
wrap(deme_expander, 'optimize_demes', oi)
import pymoses.moses.metapopulation.merging as mg
wrap(metapopulation, 'merge_demes', mi)
pr = cProfile.Profile()
k = {'i': 0}
def snap(sig, frm):
    pr.disable(); pr.dump_stats(out); pr.enable()
    log.write(f"#snapshot cpu={time.process_time():.1f}\n")
signal.signal(signal.SIGALRM, snap); signal.setitimer(signal.ITIMER_REAL, snap_every, snap_every)
t0 = time.process_time()
pr.enable()
try:
    rc = moses_exec(['moses', *args])
except SystemExit as e:
    rc = e.code
pr.disable()
signal.setitimer(signal.ITIMER_REAL, 0)
print("CPU", time.process_time()-t0, "rc", rc, file=sys.stderr)
pr.dump_stats(out)
log.write("#done\n")
