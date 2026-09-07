// Measures classic type-signature construction and arity helpers used by S8.
//
// Build:
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/type_signature_probe \
//       tools/cpp_probe/type_signature_probe.cc \
//       -L$P/lib -L$P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Run:
//   ( ulimit -v 1048576; timeout 20 tools/cpp_probe/build/type_signature_probe )
#include <moses/comboreduct/type_checker/type_tree.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog::combo;
using namespace std;

static string name(type_node node)
{
    ostringstream out;
    out << node;
    return out.str();
}

static void emit(const string& label, const string& form, const type_tree& signature, bool last)
{
    type_tree_seq inputs = get_signature_inputs(signature);
    cout << "  {\"name\":\"" << label << "\",\"form\":\"" << form << "\",\"inputs\":[";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i != 0) cout << ',';
        cout << "\"" << name(get_type_node(inputs[i])) << "\"";
    }
    cout << "],\"output\":\"" << name(get_type_node(get_signature_output(signature))) << "\""
         << ",\"arity\":" << type_tree_arity(signature)
         << ",\"boolean_arity\":" << boolean_arity(signature)
         << ",\"contin_arity\":" << contin_arity(signature)
         << ",\"action_result_arity\":" << action_result_arity(signature) << '}';
    cout << (last ? "\n" : ",\n");
}

int main()
{
    type_tree boolean_leaf(id::boolean_type);
    type_tree contin_leaf(id::contin_type);
    type_tree action_leaf(id::action_result_type);
    vector<type_node> mixed = {
        id::boolean_type,
        id::contin_type,
        id::action_result_type,
        id::boolean_type,
    };
    type_tree_seq leaf_inputs = {boolean_leaf, contin_leaf};

    cout << "[\n";
    emit("boolean_zero", "type_node_uniform", gen_signature(id::boolean_type, 0), false);
    emit("boolean_uniform_three", "type_node_uniform", gen_signature(id::boolean_type, 3), false);
    emit("contin_to_boolean_two", "type_node_io_arity",
         gen_signature(id::contin_type, id::boolean_type, 2), false);
    emit("mixed_to_action", "type_node_seq", gen_signature(mixed, id::action_result_type), false);
    emit("action_uniform_two", "type_node_uniform",
         gen_signature(id::action_result_type, 2), false);
    emit("leaf_contin_to_action_two", "type_tree_io_arity",
         gen_signature(contin_leaf, action_leaf, 2), false);
    emit("leaf_sequence_to_boolean", "type_tree_seq",
         gen_signature(leaf_inputs, boolean_leaf), true);
    cout << "]\n";
    return 0;
}
