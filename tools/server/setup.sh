#!/usr/bin/env bash
# Set up a fresh clone for the unattended server round (docs/gomea/SERVER-RUN.md).
#
#   bash tools/server/setup.sh
#
# Idempotent: rerunning it skips what is already in place. No sudo and nothing
# system-wide: every download goes under ../gomea-results/toolchain/ (or $TOOLCHAIN_DIR).
# The results root is ../gomea-results next to the clone; with GOMEA_RESULTS=/some/dir set
# and no ../gomea-results yet, ../gomea-results is created as a symlink to it.
#
# Steps (log: ../gomea-results/server-run/setup.log; summary at the end):
#   1. disk     free space under the results root (warning below 20 GB)
#   2. uv       uv $UV_VERSION (pinned; downloaded and checksum-verified if not on PATH)
#   3. python   CPython $PY_VERSION exactly, uv-managed, under the toolchain dir
#   4. venv     .venv from that Python; pip install -e . pytest==9.1.1
#   5. datasets ../gomea-results/phase1f/datasets/<Abbr>/ for Air Bos Con EnC EnH WiR WiW
#               Yac, rebuilt from data/gomea-sr/ and verified; and for Dow and Tow (no
#               licence, not committed: data/gomea-sr/README.md), the raw gpzgd files are
#               downloaded to <Abbr>/raw/ at a pinned commit, checked against their SHA-256,
#               and built with sr_datasets.py --from-raw (full.csv and splits.json checked
#               against data/gomea-sr). A missing or wrong file fails this step and names
#               the dataset; phase 1ar (which needs them) then does not start.
#   6. gpgomea  the authors' GP-GOMEA (750220c) and the seed build, for phases 1ai/1aj
#               only, so a failure here is NOT fatal; result in server-run/gpgomea-build.json.
#               Uses the system toolchain if it is complete, else (or with FORCE_CONDA=1)
#               a pinned micromamba + conda-forge environment. REBUILD_GPGOMEA=1 rebuilds.
#   7. env      server-run/environment.json (machine, limits, versions, git, pip freeze)
# Exit code 0 when every required step (all but 1 and 6) is OK.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO" || exit 1

UV_VERSION=0.13.0
UV_SHA256=b0828fc4cdc47d65951e94b5f0a15a5364420e8dcd88723f3b21e7a2eb30bbc3  # musl tarball
PY_VERSION=3.14.8
MM_VERSION=2.8.1-1
MM_SHA256=77b7790ec97f64581118f103585b175df4306f95829b0fa6bfe4a19cc88a1182  # micromamba-linux-64
GPG_URL=https://github.com/marcovirgolin/GP-GOMEA
GPG_COMMIT=750220c
# Reference (netlib) BLAS/LAPACK, as on the local machine: with OpenBLAS the tiny-problem
# output differs from the local build from generation 2 on; with netlib it is identical.
CONDA_SPECS=(python=3.14.9 numpy=2.5.4 libboost-devel=1.92.0 libboost-python-devel=1.92.0
             gxx=16.2.0 sysroot_linux-64=2.28 "libblas=3.11.0=*netlib" "liblapack=3.11.0=*netlib")
DATASETS=(Air Bos Con EnC EnH WiR WiW Yac)
# Dow and Tow: downloaded (abbr, gpzgd file, raw SHA-256 from the local PROVENANCE.md).
GPZGD_COMMIT=42e5a1829a0b8a2eb50e116000001f82c770791e
GPZGD_RAW=https://raw.githubusercontent.com/grantdick/gpzgd/$GPZGD_COMMIT/problems/data
DOWNLOADED=("Dow dowchem 0e55bcd11740d65ab7c3fe80df8e2636b0e84840c46fc9182086868a5c9f52aa"
            "Tow tower 7d353f109f0d13c4c372ec9c3b7580580d351f0b29e395c9b05a0e89bc6492db")

if [ -n "${GOMEA_RESULTS:-}" ] && [ ! -e ../gomea-results ]; then
    mkdir -p "$GOMEA_RESULTS" && ln -s "$(cd "$GOMEA_RESULTS" && pwd)" ../gomea-results
fi
mkdir -p ../gomea-results/server-run || exit 1
RES=$(cd ../gomea-results && pwd)
SR=$RES/server-run
TC=${TOOLCHAIN_DIR:-$RES/toolchain}
mkdir -p "$TC/downloads"
exec > >(tee -a "$SR/setup.log") 2>&1
echo "=== setup.sh $(date -u +%FT%TZ) repo $REPO results $RES toolchain $TC"

STEPS=(); STATUS=(); NOTES=()
record() { STEPS+=("$1"); STATUS+=("$2"); NOTES+=("$3"); echo "--- step $1: $2 ${3:+($3)}"; }
note() { echo "    $*"; }

fetch() {  # url dest sha256
    local url=$1 dest=$2 sha=$3
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status - 2>/dev/null; then return 0; fi
    rm -f "$dest.part"
    local i
    for i in 1 2 3; do
        if curl -fsSL --retry 3 --connect-timeout 30 -o "$dest.part" "$url"; then break; fi
        note "download attempt $i of $url failed"; sleep 5
    done
    [ -f "$dest.part" ] || return 1
    if ! echo "$sha  $dest.part" | sha256sum -c --status -; then
        note "checksum mismatch for $url: $(sha256sum "$dest.part" | cut -d' ' -f1) != $sha"
        rm -f "$dest.part"; return 1
    fi
    mv "$dest.part" "$dest"
}

# ---- 1. disk
step_disk() {
    local kb gb
    kb=$(df -Pk "$RES" | awk 'NR==2 {print $4}')
    gb=$((kb / 1024 / 1024))
    if [ "$gb" -lt 20 ]; then record disk WARN "only $gb GB free under $RES (want >= 20 GB)"
    else record disk OK "$gb GB free under $RES"; fi
}

# ---- 2. uv
UV=""
step_uv() {
    local c
    for c in "$TC/uv-$UV_VERSION/uv" "$(command -v uv 2>/dev/null)"; do
        if [ -n "$c" ] && [ -x "$c" ] && "$c" --version 2>/dev/null | grep -q "^uv $UV_VERSION\b"; then
            UV=$c; record uv OK "$("$UV" --version) at $UV"; return 0
        fi
    done
    local tgz=$TC/downloads/uv-$UV_VERSION-x86_64-unknown-linux-musl.tar.gz
    if ! fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-musl.tar.gz" \
            "$tgz" "$UV_SHA256"; then
        record uv FAILED "could not download uv $UV_VERSION"; return 1
    fi
    mkdir -p "$TC/uv-$UV_VERSION"
    tar xzf "$tgz" -C "$TC/uv-$UV_VERSION" --strip-components=1 || { record uv FAILED "untar"; return 1; }
    UV=$TC/uv-$UV_VERSION/uv
    "$UV" --version >/dev/null 2>&1 || { record uv FAILED "$UV does not run"; return 1; }
    record uv OK "$("$UV" --version) installed at $UV"
}

# ---- 3. python
PYBIN=""
step_python() {
    [ -n "$UV" ] || { record python FAILED "no uv"; return 1; }
    export UV_PYTHON_INSTALL_DIR=$TC/python UV_CACHE_DIR=$TC/uv-cache
    PYBIN=$(cd "$TC" && env -u VIRTUAL_ENV "$UV" python find --managed-python "$PY_VERSION" 2>/dev/null)
    if [ -z "$PYBIN" ]; then
        "$UV" python install --no-bin "$PY_VERSION" || { record python FAILED "uv python install $PY_VERSION"; return 1; }
        PYBIN=$(cd "$TC" && env -u VIRTUAL_ENV "$UV" python find --managed-python "$PY_VERSION" 2>/dev/null)
    fi
    local v
    v=$("$PYBIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)
    if [ "$v" != "$PY_VERSION" ]; then record python FAILED "found '$PYBIN' version '$v'"; return 1; fi
    record python OK "CPython $v at $PYBIN"
}

# ---- 4. venv
step_venv() {
    [ -n "$PYBIN" ] || { record venv FAILED "no Python $PY_VERSION"; return 1; }
    local want have
    want=$("$PYBIN" -c 'import sys; print(sys.base_prefix)')
    if [ -x .venv/bin/python ]; then
        have=$(.venv/bin/python -c 'import sys; print(sys.base_prefix)' 2>/dev/null)
        if [ "$have" != "$want" ]; then
            note ".venv is from '$have', not '$want': moving it to .venv.old-$(date +%s)"
            mv .venv ".venv.old-$(date +%s)"
        fi
    fi
    if [ ! -x .venv/bin/python ]; then
        "$PYBIN" -m venv .venv || { record venv FAILED "python -m venv"; return 1; }
    fi
    .venv/bin/python -m pip install -q --disable-pip-version-check -e . pytest==9.1.1 \
        || { record venv FAILED "pip install -e . pytest==9.1.1"; return 1; }
    if ! .venv/bin/python -c "import pathlib, pymoses; assert pathlib.Path(pymoses.__file__).resolve().is_relative_to(pathlib.Path('$REPO').resolve())"; then
        record venv FAILED "pymoses does not import from $REPO"; return 1
    fi
    if ! .venv/bin/moses --help 2>&1 | grep -q "usage: moses"; then record venv FAILED ".venv/bin/moses --help fails"; return 1; fi
    .venv/bin/python -m pip freeze --all > "$SR/pip-freeze.txt"
    record venv OK "$(.venv/bin/python -V) pytest $(.venv/bin/python -c 'import pytest; print(pytest.__version__)'), pymoses editable from $REPO"
}

# ---- 5. datasets
dataset_ok() {  # dir abbr: full.csv, splits.json and the 240 split files verified
    local D=$1 a=$2
    [ -f "$D/$a/full.csv" ] && [ -f "$D/$a/splits/splits.json" ] \
        && [ "$(sha256sum < "$D/$a/full.csv" | cut -d' ' -f1)" = "$(grep " $a/full.csv" data/gomea-sr/SHA256SUMS | cut -d' ' -f1)" ] \
        && cmp -s "$D/$a/splits/splits.json" "data/gomea-sr/$a/splits.json" \
        && [ "$(find "$D/$a/splits" -mindepth 2 -type f | wc -l)" -eq 240 ]
}
step_datasets() {
    [ -x .venv/bin/python ] || { record datasets FAILED "no venv"; return 1; }
    local D=$RES/phase1f/datasets todo=() a f sha line built=() bad=()
    mkdir -p "$D"
    for a in "${DATASETS[@]}"; do
        dataset_ok "$D" "$a" || todo+=("$a")
    done
    if [ ${#todo[@]} -gt 0 ]; then
        if .venv/bin/python tools/gomea/sr_datasets.py "$D" "${todo[@]}" --from-full data/gomea-sr > "$SR/datasets.json"; then
            built+=("${todo[@]}")
        else
            bad+=("${todo[*]}: sr_datasets.py --from-full failed")
        fi
    fi
    for line in "${DOWNLOADED[@]}"; do
        read -r a f sha <<< "$line"
        dataset_ok "$D" "$a" && continue
        mkdir -p "$D/$a/raw"
        if ! fetch "$GPZGD_RAW/$f" "$D/$a/raw/$f" "$sha"; then
            note "DATASET $a: download of $GPZGD_RAW/$f failed, or its SHA-256 is not $sha"
            bad+=("$a: download or raw SHA-256 check failed ($GPZGD_RAW/$f)"); continue
        fi
        if .venv/bin/python tools/gomea/sr_datasets.py "$D" "$a" --from-raw data/gomea-sr > "$SR/datasets-$a.json"; then
            built+=("$a")
        else
            note "DATASET $a: sr_datasets.py --from-raw failed (full.csv or splits.json differs)"
            bad+=("$a: sr_datasets.py --from-raw failed")
        fi
    done
    cp data/gomea-sr/SHA256SUMS data/gomea-sr/PROVENANCE.md "$D/" 2>/dev/null
    if [ ${#bad[@]} -gt 0 ]; then
        record datasets FAILED "$(IFS=';'; echo "${bad[*]}") (phase 1ar needs Dow and Tow and will not start)"; return 1
    fi
    if [ ${#built[@]} -eq 0 ]; then
        record datasets OK "all $(( ${#DATASETS[@]} + ${#DOWNLOADED[@]} )) present and verified"
    else
        record datasets OK "built ${built[*]} (verified against data/gomea-sr)"
    fi
}

# ---- 6. GP-GOMEA (non-fatal)
GPG_SRC=$RES/src/GP-GOMEA
GPG_B1=$RES/phase1b/gpgomea-build
GPG_B2=$RES/gpgomea-seed-build
GPG_JSON=$SR/gpgomea-build.json
GPG_LOG=$SR/gpgomea-build.log
system_toolchain_ok() {
    command -v g++ >/dev/null && command -v python3 >/dev/null || return 1
    python3 -c 'import sys, numpy; assert sys.version_info[:2] == (3, 14)' 2>/dev/null || return 1
    local t; t=$(mktemp -d)
    cat > "$t/t.cpp" <<'EOF'
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
int main() { boost::program_options::options_description d("x"); return 0; }
EOF
    local PYINC NPINC
    PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
    NPINC=$(python3 -c "import numpy;print(numpy.get_include())")
    g++ -std=c++17 "$t/t.cpp" -I "$PYINC" -I "$NPINC" -o "$t/t" -llapack -lblas -lboost_program_options \
        -lboost_filesystem -lboost_python314 -lboost_numpy314 -lpython3.14 >/dev/null 2>&1 && "$t/t"
    local rc=$?
    rm -rf "$t"
    return $rc
}
MM=""
conda_env() {  # sets CENV
    MM=$TC/micromamba-$MM_VERSION/micromamba
    mkdir -p "$(dirname "$MM")"
    fetch "https://github.com/mamba-org/micromamba-releases/releases/download/$MM_VERSION/micromamba-linux-64" \
        "$MM" "$MM_SHA256" || { note "micromamba download failed"; return 1; }
    chmod +x "$MM"
    export MAMBA_ROOT_PREFIX=$TC/mamba-root
    CENV=$TC/gpgomea-env
    if [ ! -f "$CENV/.specs" ] || [ "$(cat "$CENV/.specs")" != "${CONDA_SPECS[*]}" ]; then
        rm -rf "$CENV"
        "$MM" create -y -q -p "$CENV" -c conda-forge --override-channels "${CONDA_SPECS[@]}" \
            || { note "micromamba create failed"; return 1; }
        echo "${CONDA_SPECS[*]}" > "$CENV/.specs"
    fi
    "$MM" env export -p "$CENV" --explicit > "$SR/gpgomea-conda-explicit.txt" 2>/dev/null || true
    return 0
}
build_with() {  # route -> builds both binaries
    local route=$1
    (
        set -e
        if [ "$route" = conda ]; then
            export PATH="$CENV/bin:$PATH"
            if [ -x "$CENV/bin/x86_64-conda-linux-gnu-g++" ]; then export GPGOMEA_CXX=$CENV/bin/x86_64-conda-linux-gnu-g++
            else export GPGOMEA_CXX=$CENV/bin/g++; fi
            export GPGOMEA_CXXFLAGS="-I$CENV/include"
            export GPGOMEA_LDFLAGS="-L$CENV/lib -Wl,-rpath,$CENV/lib -Wl,-rpath-link,$CENV/lib"
        fi
        echo "route $route: $("${GPGOMEA_CXX:-g++}" --version | head -1); python3 = $(command -v python3)"
        bash tools/gomea/build_gpgomea.sh "$GPG_SRC" "$GPG_B1"
        bash tools/gomea/build_gpgomea_seed.sh "$GPG_SRC" "$GPG_B2" "$GPG_B1/armadillo-code-14.0.2"
    ) >> "$GPG_LOG" 2>&1
}
gpg_json() {  # ok route error
    .venv/bin/python - "$1" "$2" "$3" "$GPG_JSON" "$GPG_B1/gpgomea" "$GPG_B2/gpgomea-seed" "$SR/gpgomea-smoke.json" "$GPG_SRC" <<'EOF'
import hashlib, json, os, subprocess, sys
ok, route, err, out, b1, b2, smoke, src = sys.argv[1:]
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
def ldd(p):
    try: return subprocess.run(["ldd", p], capture_output=True, text=True).stdout.splitlines()
    except OSError: return None
s = json.load(open(smoke)) if os.path.exists(smoke) else None
head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
rec = {"ok": ok == "1", "route": route or None, "error": err or None, "gp_gomea_commit": head or None,
       "binaries": {b1: sha(b1), b2: sha(b2)},
       "smoke_ok": s["ok"] if s else None, "smoke_matches_local": s["matches_local"] if s else None,
       "smoke": {k: {f: v.get(f) for f in ("ok", "rc", "best", "local_best", "evaluations", "matches_local", "sha256")}
                 for k, v in (s or {}).items() if isinstance(v, dict)},
       "ldd": {b1: ldd(b1), b2: ldd(b2)}}
json.dump(rec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
EOF
}
step_gpgomea() {
    if [ -z "${REBUILD_GPGOMEA:-}" ] && [ -f "$GPG_JSON" ] && [ -x "$GPG_B1/gpgomea" ] && [ -x "$GPG_B2/gpgomea-seed" ] \
       && .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$GPG_JSON'))['ok'] else 1)" 2>/dev/null \
       && .venv/bin/python tools/server/gpgomea_smoke.py "$GPG_B1/gpgomea" "$GPG_B2/gpgomea-seed" \
            "$RES/phase1f/datasets" --out "$SR/gpgomea-smoke.json" > /dev/null; then
        record gpgomea OK "already built ($(.venv/bin/python -c "import json; d=json.load(open('$GPG_JSON')); print('route', d['route'], '; output matches local:', d['smoke_matches_local'])"))"
        return 0
    fi
    rm -f "$SR/gpgomea-smoke.json"
    echo "=== $(date -u +%FT%TZ)" >> "$GPG_LOG"
    if [ ! -d "$GPG_SRC/.git" ]; then
        mkdir -p "$(dirname "$GPG_SRC")"
        git clone -q "$GPG_URL" "$GPG_SRC" >> "$GPG_LOG" 2>&1 \
            || { gpg_json 0 "" "git clone $GPG_URL failed"; record gpgomea FAILED "git clone failed (phases 1ai, 1aj will be skipped)"; return 1; }
    fi
    if ! git -C "$GPG_SRC" checkout -q "$GPG_COMMIT" >> "$GPG_LOG" 2>&1; then
        git -C "$GPG_SRC" fetch -q origin >> "$GPG_LOG" 2>&1
        git -C "$GPG_SRC" checkout -q "$GPG_COMMIT" >> "$GPG_LOG" 2>&1 \
            || { gpg_json 0 "" "checkout $GPG_COMMIT failed"; record gpgomea FAILED "checkout $GPG_COMMIT failed"; return 1; }
    fi
    # A route whose tiny-problem output matches the local build wins; otherwise the first
    # route that works is used (rebuilt if a later route overwrote it).
    local routes=() r err="" fallback="" last=""
    if [ "${FORCE_CONDA:-0}" != 1 ] && system_toolchain_ok; then routes+=(system); fi
    routes+=(conda)
    for r in "${routes[@]}"; do
        note "GP-GOMEA: trying the $r toolchain (log $GPG_LOG)"
        if [ "$r" = conda ] && ! conda_env >> "$GPG_LOG" 2>&1; then err="$err conda env failed;"; continue; fi
        last=$r
        if ! build_with "$r"; then err="$err $r build failed;"; continue; fi
        if ! .venv/bin/python tools/server/gpgomea_smoke.py "$GPG_B1/gpgomea" "$GPG_B2/gpgomea-seed" \
               "$RES/phase1f/datasets" --out "$SR/gpgomea-smoke.json" >> "$GPG_LOG" 2>&1; then
            err="$err $r smoke run failed;"; continue
        fi
        if .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$SR/gpgomea-smoke.json'))['matches_local'] else 1)"; then
            gpg_json 1 "$r" ""
            record gpgomea OK "built with the $r toolchain; tiny-problem output matches the local build"
            return 0
        fi
        note "GP-GOMEA: the $r build works but its output differs from the local build"
        fallback=${fallback:-$r}
    done
    if [ -n "$fallback" ]; then
        if [ "$last" != "$fallback" ]; then
            build_with "$fallback" && .venv/bin/python tools/server/gpgomea_smoke.py "$GPG_B1/gpgomea" \
                "$GPG_B2/gpgomea-seed" "$RES/phase1f/datasets" --out "$SR/gpgomea-smoke.json" >> "$GPG_LOG" 2>&1 \
                || { gpg_json 0 "" "$err rebuild with $fallback failed"; record gpgomea FAILED "rebuild with $fallback failed"; return 1; }
        fi
        gpg_json 1 "$fallback" ""
        record gpgomea OK "built with the $fallback toolchain; WARNING tiny-problem output differs from the local build"
        return 0
    fi
    gpg_json 0 "" "$err"
    record gpgomea FAILED "$err (phases 1ai, 1aj will be skipped; see $GPG_LOG)"
    return 1
}

# ---- 7. environment
step_env() {
    if UV_BIN=$UV .venv/bin/python tools/server/envinfo.py "$SR/environment.json"; then
        record env OK "$SR/environment.json"
    else record env FAILED "envinfo.py"; return 1; fi
}

step_disk
step_uv && step_python && step_venv
if [ -x .venv/bin/python ]; then
    step_datasets
    step_gpgomea
    step_env
fi

echo
echo "=== setup summary ($(date -u +%FT%TZ))"
FAIL=0
for i in "${!STEPS[@]}"; do
    printf '  %-9s %-7s %s\n' "${STEPS[$i]}" "${STATUS[$i]}" "${NOTES[$i]}"
    case "${STEPS[$i]}:${STATUS[$i]}" in disk:*|gpgomea:*|*:OK) ;; *) FAIL=1 ;; esac
done
for s in uv python venv datasets env; do
    printf '%s\n' "${STEPS[@]}" | grep -qx "$s" || { printf '  %-9s %-7s %s\n' "$s" "NOT RUN" "(an earlier step failed)"; FAIL=1; }
done
if [ $FAIL = 0 ]; then echo "SETUP OK: start the round with: bash tools/server/start.sh"
else echo "SETUP FAILED: see the steps above and $SR/setup.log"; fi
exit $FAIL
