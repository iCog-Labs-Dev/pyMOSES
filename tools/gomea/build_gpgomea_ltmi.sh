#!/usr/bin/env bash
# Build GP-GOMEA with the plain (uncorrected) mutual-information linkage tree, LT-MI.
#
# Usage: tools/gomea/build_gpgomea_ltmi.sh GP_GOMEA_CHECKOUT BUILD_DIR
#
# At GP-GOMEA@750220c the Eq. 2 bias correction (LT-MIb) is hard-wired on: the plain-MI
# branch in src/GOMEA/GOMEAFOS.cpp is `if (false)`. Phase 1f replicates Table 3 of
# Virgolin et al. (2021), which also reports LT-MI, so this copies the sources, flips
# that one condition and builds BUILD_DIR/gpgomea-ltmi. The diff is written to
# BUILD_DIR/ltmi.diff. BUILD_DIR must already hold armadillo-code-14.0.2 (run
# build_gpgomea.sh first). The checkout is never modified.
set -euo pipefail
G=$(realpath "${1:?GP-GOMEA checkout}")
B=$(realpath "${2:?build dir}")
cd "$B"
ARMA=armadillo-code-14.0.2
[ -d "$ARMA" ] || { echo "missing $B/$ARMA; run build_gpgomea.sh first" >&2; exit 1; }
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
NPINC=$(python3 -c "import numpy;print(numpy.get_include())")
PYVER=$(python3 -c "import sys;print(f'{sys.version_info[0]}{sys.version_info[1]}')")
rm -rf src-ltmi && cp -r "$G/src" src-ltmi
python3 - <<'PY'
p = "src-ltmi/GOMEA/GOMEAFOS.cpp"
s = open(p).read()
old = "    //if (!MI_distribution_adjustments) {    // we always apply the correction\n    if (false){"
assert s.count(old) == 1
s = s.replace(old, "    //if (!MI_distribution_adjustments) {    // we always apply the correction\n    if (true){")
open(p, "w").write(s)
PY
diff -u "$G/src/GOMEA/GOMEAFOS.cpp" src-ltmi/GOMEA/GOMEAFOS.cpp > ltmi.diff || true
mapfile -t srcs < <(find src-ltmi -name "*.cpp" ! -name gpgomea_python.cpp | sort)
g++ -std=c++17 -O2 -fopenmp -w -DARMA_DONT_USE_WRAPPER -I src-ltmi/Include -I "$ARMA/include" \
    -I "$PYINC" -I "$NPINC" "${srcs[@]}" -o gpgomea-ltmi -llapack -lblas -lboost_program_options \
    -lboost_filesystem "-lboost_python$PYVER" "-lboost_numpy$PYVER" "-lpython${PYVER:0:1}.${PYVER:1}"
sha256sum gpgomea-ltmi
