// tools/cpp_probe/contin_interpreter_probe.cc
//
// Measures the REAL output of contin_interpreter/mixed_interpreter
// (moses/comboreduct/interpreter/interpreter.{h,cc}) for pyMoses's S14
// Task 1 port (interpreter.py), for edge cases interpreterUTest.cxxtest's
// own three tests (already reproduced byte-for-byte in
// tests/comboreduct/interpreter/test_interpreter.py) do NOT reach:
// overflow reconstruction (div-by-zero, log domain errors, exp overflow),
// sin's silent-nan-no-throw behaviour, a bare nan/inf contin_t leaf, and
// mixed_interpreter's greater_than_zero recovering a sign from a caught
// OverflowException. Ground truth, not inference -- this port's Python
// reconstruction of C's IEEE754 div/log/exp semantics (eval.py's
// eval_throws_binding already established the same idioms for a
// DIFFERENT C++ source file, eval.cc; this probe confirms
// interpreter.cc's own contin_interpreter/mixed_interpreter really do
// the same thing, not merely "probably, by analogy").
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/contin_interpreter_probe \
//       tools/cpp_probe/contin_interpreter_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run (bounded, per CLAUDE.md's OOM-safety rule):
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/contin_interpreter_probe )
#include <moses/comboreduct/interpreter/interpreter.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/crutil/exception.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

// Prints a contin_t the same way ostream_vertex's own fallback does
// (defaultfloat, precision 6 -- iostream_combo.cc:463-464), so this
// probe's own text output matches what the Python side's `.6g` format
// spec is verified against elsewhere (iostream_combo.py's _contin_to_str).
static string show_contin(contin_t c)
{
    ostringstream oss;
    oss << c;
    return oss.str();
}

static void contin_case(const string& label, const string& prog,
                        const vector<contin_t>& inputs)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    contin_interpreter ci(inputs);
    try {
        contin_t res = ci(tr);
        cout << label << "=VALUE:" << show_contin(res) << endl;
    } catch (OverflowException& e) {
        contin_t v = get_contin(e.get_vertex());
        cout << label << "=OVERFLOW:" << show_contin(v) << endl;
    }
}

static void mixed_case(const string& label, const string& prog,
                       const vector<vertex>& inputs)
{
    combo_tree tr = str2combo_tree(prog, vector<string>());
    mixed_interpreter mi(inputs);
    vertex res = mi(tr);
    ostringstream oss;
    oss << res;
    cout << label << "=VALUE:" << oss.str() << endl;
}

int main()
{
    // ---- contin_interpreter: overflow reconstruction ----
    contin_case("div_by_zero_positive_numerator", "/($1 $2)", {1.0, 0.0});
    contin_case("div_by_zero_negative_numerator", "/($1 $2)", {-1.0, 0.0});
    contin_case("div_zero_numerator_shortcuts_denominator", "/($1 log($2))", {0.0, 0.0});
    contin_case("log_of_zero", "log($1)", {0.0});
    contin_case("log_of_negative", "log($1)", {-1.0});
    contin_case("exp_overflow", "exp($1)", {1000.0});
    contin_case("sin_of_infinity_no_throw", "sin($1)", {INFINITY});
    contin_case("contin_constant_nan_leaf", "nan", {});
    contin_case("contin_constant_inf_leaf", "inf", {});

    // ---- contin_interpreter: foldr identities over no children ----
    // Printed via the tree's own root vertex rather than contin_case
    // (the return type here really is `builtin`, not `contin_t` -- see
    // interpreter.cc:133-134/143-146's "for correct foldr behaviour").
    {
        combo_tree tr = str2combo_tree("+()", vector<string>());
        contin_interpreter ci((vector<contin_t>()));
        vertex res = ci(tr);
        ostringstream oss;
        oss << res;
        cout << "plus_over_no_children=VALUE:" << oss.str() << endl;
    }
    {
        combo_tree tr = str2combo_tree("*()", vector<string>());
        contin_interpreter ci((vector<contin_t>()));
        vertex res = ci(tr);
        ostringstream oss;
        oss << res;
        cout << "times_over_no_children=VALUE:" << oss.str() << endl;
    }

    // ---- mixed_interpreter: greater_than_zero recovers a sign from a
    // caught OverflowException (interpreter.cc:316-329's own comment) ----
    mixed_case("mixed_gtz_recovers_positive_infinity", "0<(/(1.0 0.0))", {});
    mixed_case("mixed_gtz_recovers_negative_infinity", "0<(/(-1.0 0.0))", {});

    // ---- mixed_interpreter: cond's universal else clause ----
    mixed_case("mixed_cond_true_predicate", "cond($1 10.0 20.0)",
               {vertex(id::logical_true)});
    mixed_case("mixed_cond_false_predicate_falls_to_else", "cond($1 10.0 20.0)",
               {vertex(id::logical_false)});

    return 0;
}
