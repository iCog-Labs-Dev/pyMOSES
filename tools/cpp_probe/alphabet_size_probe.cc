// Probe: calls the REAL opencog::moses::alphabet_size (problem.cc:138) and
// reproduces the exact `int as = alphabet_size(...)` assignment done at
// table-problems.cc:335 and :372. Nothing here re-derives the arithmetic;
// the value printed comes out of the linked library.
#include <iostream>
#include <moses/comboreduct/type_checker/type_tree.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/moses/main/problem.h>

using namespace opencog;
using namespace opencog::combo;
using opencog::moses::alphabet_size;

int main() {
  // The full set of operators --ignore-operator can name, so the corpus can
  // push ignore_ops.size() past 8 + arity.
  vertex all[] = {id::plus, id::times, id::div, id::exp, id::log, id::sin,
                  id::impulse, id::logical_and, id::logical_or,
                  id::logical_not, id::greater_than_zero, id::cond};
  const int n_all = 12;

  for (int arity = 1; arity <= 3; ++arity) {
    type_tree tt = gen_signature(id::contin_type, id::contin_type, arity);
    for (int n = 0; n <= n_all; ++n) {
      vertex_set ig;
      for (int i = 0; i < n; ++i) ig.insert(all[i]);
      unsigned as_u = alphabet_size(tt, ig);   // problem.cc:139 `unsigned as`
      int as_i = alphabet_size(tt, ig);        // table-problems.cc:335 `int as`
      std::cout << arity << " " << n << " " << as_u << " " << as_i << "\n";
    }
  }
  return 0;
}
