#!/usr/bin/env bash
# Remove the block install_autostart.sh added to ~/.jupyter/jupyter_server_config.py.
set -u
CFG=${JUPYTER_CONFIG_DIR:-$HOME/.jupyter}/jupyter_server_config.py
[ -f "$CFG" ] || { echo "no $CFG"; exit 0; }
python3 - "$CFG" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
t = re.sub(r"# >>> pyMOSES server round autostart.*?# <<< pyMOSES server round autostart <<<\n?", "", s, flags=re.S)
open(p, "w").write(t)
print("removed" if t != s else "nothing to remove", "in", p)
PY
