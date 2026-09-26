#!/usr/bin/env bash
# Build the authors' GP-GOMEA CLI out of tree, without installing anything system-wide.
#
# Usage: tools/gomea/build_gpgomea.sh GP_GOMEA_CHECKOUT BUILD_DIR
#
# The result is BUILD_DIR/gpgomea (the unmodified sources) and BUILD_DIR/gpgomea-instr
# (identical except that each generation log line also prints the cumulative evaluation
# count; the diff is written to BUILD_DIR/instrumentation.diff).
#
# Requires g++ (C++17), BLAS/LAPACK, Boost (program_options, filesystem, python, numpy)
# and Python 3.14 development headers, all from the system. It downloads the Armadillo
# 14.0.2 source (Apache-2.0) and uses it header-only (ARMA_DONT_USE_WRAPPER).
# The checkout used for Phase 1b is GP-GOMEA@750220c.
set -euo pipefail
G=$(realpath "${1:?GP-GOMEA checkout}")
B=$(realpath -m "${2:?build dir}")
mkdir -p "$B"
cd "$B"
ARMA=armadillo-code-14.0.2
if [ ! -d "$ARMA" ]; then
  curl -sL -o armadillo-14.0.2.tar.gz \
    https://gitlab.com/conradsnicta/armadillo-code/-/archive/14.0.2/armadillo-code-14.0.2.tar.gz
  echo "a067616d88af8d993fce1c73e1ebcf66aa57551a8bc6ba5b7f62b7b99991d207  armadillo-14.0.2.tar.gz" | sha256sum -c -
  tar xzf armadillo-14.0.2.tar.gz
fi
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
NPINC=$(python3 -c "import numpy;print(numpy.get_include())")
PYVER=$(python3 -c "import sys;print(f'{sys.version_info[0]}{sys.version_info[1]}')")
build() {  # $1 = source dir, $2 = output name
  mapfile -t srcs < <(find "$1" -name "*.cpp" ! -name gpgomea_python.cpp | sort)
  g++ -std=c++17 -O2 -fopenmp -w -DARMA_DONT_USE_WRAPPER -I "$1/Include" -I "$ARMA/include" \
      -I "$PYINC" -I "$NPINC" "${srcs[@]}" -o "$2" -llapack -lblas -lboost_program_options \
      -lboost_filesystem "-lboost_python$PYVER" "-lboost_numpy$PYVER" "-lpython${PYVER:0:1}.${PYVER:1}"
}
build "$G/src" gpgomea
rm -rf src-instr && cp -r "$G/src" src-instr
python3 - <<'EOF'
p = "src-instr/RunHandling/IMSHandler.cpp"
s = open(p).read()
old = 'cout << " > generation " << macro_generation << " - best fit: " << elitist_fit << endl;'
assert s.count(old) == 1
s = s.replace(old, 'cout << " > generation " << macro_generation << " - best fit: " << elitist_fit'
              ' << " - evaluations: " << st->fitness->evaluations << endl;')
open(p, "w").write(s)
EOF
diff -u "$G/src/RunHandling/IMSHandler.cpp" src-instr/RunHandling/IMSHandler.cpp > instrumentation.diff || true
build src-instr gpgomea-instr
sha256sum gpgomea gpgomea-instr
