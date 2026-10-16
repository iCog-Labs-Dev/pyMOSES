import sys
sys.setrecursionlimit(100000)
import pymoses; assert "pyMoses-knobs" in pymoses.__file__, pymoses.__file__
from pymoses.moses.main.moses_exec import moses_exec
n = sys.argv[1]
ex = open(f"/tmp/claude-1000/knobs-investigation/{n}.ex").read().strip()
moses_exec(["moses", "-H", "mux", "-k", "3", "-r", "1", "-e", ex, "-a", "hc", "-m", "1", "-l", "DEBUG", "-f", f"/tmp/claude-1000/knobs-investigation/py_{n}.log"])
