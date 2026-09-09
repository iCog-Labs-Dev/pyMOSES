// Probe: does logical_reduce terminate on a predicate over log(-1)?
#include <moses/comboreduct/reduct/reduct.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <cstdio>
#include <sstream>
using namespace opencog;
using namespace opencog::combo;
int main(int argc, char** argv) {
  std::stringstream ss(argv[1]);
  combo_tree tr; ss >> tr;
  printf("parsed, reducing...\n"); fflush(stdout);
  vertex_set ignore_ops;
  reduct::logical_reduction lr(ignore_ops);
  lr(2)(tr);
  std::stringstream out; out << tr;
  printf("DONE: %s\n", out.str().c_str());
  return 0;
}
