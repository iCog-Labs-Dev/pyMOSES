#!/usr/bin/env bash
# Opt-in: make the Jupyter server start (resume) the round whenever it starts, e.g. after a
# pod restart. Appends a delimited block to ~/.jupyter/jupyter_server_config.py that runs
# tools/server/start.sh in the background unless this plan and each later plan of its chain are
# DONE. start.sh goes on with the first plan that is not DONE, and it refuses when an
# orchestrator of the chain is already running. Undo: uninstall_autostart.sh.
#   bash tools/server/install_autostart.sh [--plan round|smoke|round2|smoke2|round3|smoke3|round4|smoke4|round5|smoke5|round6|smoke6|round7|smoke7|round8|smoke8|round9|smoke9|round10|smoke10|FILE]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CFG=${JUPYTER_CONFIG_DIR:-$HOME/.jupyter}/jupyter_server_config.py
mkdir -p "$(dirname "$CFG")"
if grep -q ">>> pyMOSES server round autostart" "$CFG" 2>/dev/null; then
    echo "already installed in $CFG (run uninstall_autostart.sh first to change it)"; exit 0
fi
SDS=$(chain_cmd state-dirs "$PLAN" | "$PY3" -c 'import sys; print(repr(sys.stdin.read().split("\n")[:-1]))')
[ "$SDS" != "[]" ] || { echo "cannot read the chain of $PLAN; nothing installed" >&2; exit 2; }
cat >> "$CFG" <<PYEOF
# >>> pyMOSES server round autostart (tools/server/install_autostart.sh) >>>
def _pymoses_round_autostart():
    import os, subprocess
    repo, sd, chain_dirs = "$REPO", "$SD", $SDS
    if not os.path.exists(os.path.join(repo, ".venv")):
        return
    if all(os.path.exists(os.path.join(d, "DONE")) for d in chain_dirs):
        return
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "autostart.log"), "a") as log:
        subprocess.Popen(["bash", os.path.join(repo, "tools/server/start.sh"), "--plan", "$PLAN"],
                         cwd=repo, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                         start_new_session=True)
try:
    _pymoses_round_autostart()
except Exception as _e:  # never break the Jupyter server
    print("pyMOSES round autostart failed:", _e)
# <<< pyMOSES server round autostart <<<
PYEOF
echo "installed in $CFG"
