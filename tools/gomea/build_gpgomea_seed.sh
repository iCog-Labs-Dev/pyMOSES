#!/usr/bin/env bash
# Build GP-GOMEA with an elite dump for seeding MOSES (backlog item 9).
#
# Usage: tools/gomea/build_gpgomea_seed.sh GP_GOMEA_CHECKOUT BUILD_DIR [ARMADILLO_DIR]
#
# The result is BUILD_DIR/gpgomea-seed: the sources of build_gpgomea.sh's gpgomea-instr
# (each generation line also prints the cumulative evaluation count) plus, at
# termination, one line per candidate seed, printed before the `Evaluations:` line:
#
#     SEED_ELITE <run> <fit> <a> <b> <prefix expression>   (each IMS run's elitist)
#     SEED_POP   <run> <fit> <a> <b> <prefix expression>   (each individual of the
#                                                           still-active populations)
#
# `fit` is GP-GOMEA's own training fitness, as cached during the run (no re-evaluation, so
# the dump costs no evaluations); `a b` are the linear-scaling terms of that program on the
# training rows (`0 1` without --linearscaling); the expression is prefix, only the
# active nodes, with constants printed at 17 significant digits, e.g. `(+ x0 (* 2.5 x3))`.
# The diff is written to BUILD_DIR/seed.diff. ARMADILLO_DIR defaults to
# BUILD_DIR/armadillo-code-14.0.2 (run build_gpgomea.sh first, or pass the one it made).
# The checkout is never modified.
set -euo pipefail
G=$(realpath "${1:?GP-GOMEA checkout}")
B=$(realpath -m "${2:?build dir}")
ARMA=$(realpath -m "${3:-$B/armadillo-code-14.0.2}")
mkdir -p "$B"
cd "$B"
[ -d "$ARMA/include" ] || { echo "missing $ARMA/include; run build_gpgomea.sh first" >&2; exit 1; }
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
NPINC=$(python3 -c "import numpy;print(numpy.get_include())")
PYVER=$(python3 -c "import sys;print(f'{sys.version_info[0]}{sys.version_info[1]}')")
rm -rf src-seed && cp -r "$G/src" src-seed
python3 - <<'PY'
p = "src-seed/RunHandling/IMSHandler.cpp"
s = open(p).read()
# 1. the instrumentation of build_gpgomea.sh (cumulative evaluations per generation)
old = 'cout << " > generation " << macro_generation << " - best fit: " << elitist_fit << endl;'
assert s.count(old) == 1
s = s.replace(old, 'cout << " > generation " << macro_generation << " - best fit: " << elitist_fit'
              ' << " - evaluations: " << st->fitness->evaluations << endl;')
# 2. the seed dump
inc = '#include "GPGOMEA/RunHandling/IMSHandler.h"\n'
assert s.count(inc) == 1
s = s.replace(inc, inc + r'''#include "GPGOMEA/Operators/Regression/OpRegrConstant.h"
#include <sstream>
#include <iomanip>

static void SeedDumpExpr(Node * n, std::ostringstream & o) {
    if (n->op->type == OperatorType::opTermConstant) {
        OpRegrConstant * c = dynamic_cast<OpRegrConstant *>(n->op);
        if (c)
            o << std::setprecision(17) << c->GetConstant();
        else
            o << n->op->name;
        return;
    }
    if (n->op->type == OperatorType::opTermVariable) {
        o << n->op->name;
        return;
    }
    o << "(" << n->op->name;
    for (size_t i = 0; i < n->GetArity(); i++) {
        o << " ";
        SeedDumpExpr(n->children[i], o);
    }
    o << ")";
}

static void SeedDumpLine(const char * tag, int run, Node * n, double_t fit, EvolutionState * st) {
    std::ostringstream o;
    o << std::setprecision(17) << tag << "\t" << run << "\t" << fit << "\t";
    if (st->config->linear_scaling) {
        std::pair<double_t, double_t> ab = Utils::ComputeLinearScalingTerms(
            n->GetOutput(st->fitness->TrainX, false), st->fitness->TrainY,
            &st->fitness->trainY_mean, &st->fitness->var_comp_trainY);
        o << ab.first << "\t" << ab.second;
    } else {
        o << "0\t1";
    }
    o << "\t";
    SeedDumpExpr(n, o);
    std::cout << o.str() << std::endl;
}
''')
old = '''    // Evaluations
    out = "Evaluations:\\t"'''
assert s.count(old) == 1, "evaluations anchor"
s = s.replace(old, '''    // Seed dump (no evaluations: cached fitnesses only)
    for (size_t i = 0; i < elitist_per_run.size(); i++)
        if (elitist_per_run[i])
            SeedDumpLine("SEED_ELITE", (int) i, elitist_per_run[i], elitist_per_run_fit[i], st);
    for (size_t i = 0; i < runs.size(); i++)
        if (runs[i])
            for (Node * n : runs[i]->population)
                SeedDumpLine("SEED_POP", (int) i, n, n->cached_fitness, st);

''' + old)
open(p, "w").write(s)
PY
diff -u "$G/src/RunHandling/IMSHandler.cpp" src-seed/RunHandling/IMSHandler.cpp > seed.diff || true
mapfile -t srcs < <(find src-seed -name "*.cpp" ! -name gpgomea_python.cpp | sort)
g++ -std=c++17 -O2 -fopenmp -w -DARMA_DONT_USE_WRAPPER -I src-seed/Include -I "$ARMA/include" \
    -I "$PYINC" -I "$NPINC" "${srcs[@]}" -o gpgomea-seed -llapack -lblas -lboost_program_options \
    -lboost_filesystem "-lboost_python$PYVER" "-lboost_numpy$PYVER" "-lpython${PYVER:0:1}.${PYVER:1}"
sha256sum gpgomea-seed
