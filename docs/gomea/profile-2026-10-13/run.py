import sys, time, cProfile, pstats
sys.setrecursionlimit(100000)
from pymoses.moses.main.moses_exec import moses_exec
out = sys.argv[1]; args = sys.argv[2:]
pr = cProfile.Profile()
t0 = time.process_time(); w0=time.time()
pr.enable()
try:
    rc = moses_exec(['moses', *args])
except SystemExit as e:
    rc = e.code
pr.disable()
print("CPU", time.process_time()-t0, "WALL", time.time()-w0, "rc", rc, file=sys.stderr)
pr.dump_stats(out)
