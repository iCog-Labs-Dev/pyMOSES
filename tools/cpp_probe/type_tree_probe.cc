// tools/cpp_probe/type_tree_probe.cc
//
// Oracle for the combo TYPE-INFERENCE ENGINE, i.e. the 41 unported
// definitions of ($CLASSIC/moses/comboreduct/type_checker/type_tree.cc)
// plus the two in ($CLASSIC/moses/comboreduct/type_checker/type_tree_def.cc)
// and the `builtins_properties` / `actions_properties` singletons they
// read out of ($CLASSIC/moses/comboreduct/combo/descriptions.cc:100,180).
//
// This links classic's OWN functions out of libcomboreduct.so; nothing
// below reimplements anything.
//
// Sections, and which C++ arm each one reaches:
//
//   "builtins" -- descriptions.cc:100 builtins_properties::builtins_properties,
//     dumped through get_arity(builtin) (cc:47), get_type_tree(builtin)
//     (cc:53), get_output_type_tree(builtin) (cc:59) and
//     get_input_type_tree(builtin, i) (cc:65) for every one of the 26
//     id::builtin values, i in 0..3. Reaches, deliberately: the arity==-1
//     arg_list arm of cc:65 (id::lambda, whose type is
//     "->(arg_list(unknown) ->)" and whose type_tree_arity is -1); the
//     `i < arity` arm; and the past-the-end `return type_tree()` arm
//     (an EMPTY type_tree, which prints as the empty string). Also
//     reaches descriptions.cc:135's arg_list flattening, which DROPS
//     every element of an arg_list after the first -- visible on
//     id::cond, whose arg_list has two.
//
//   "actions" -- the same four facts for every one of the 14 id::action
//     values, via descriptions.cc:180 actions_properties::actions_properties
//     and get_type_tree(action) (cc:79) / get_output_type_tree(action)
//     (cc:85) / get_input_type_tree(action, i) (cc:91). Included as the
//     EVIDENCE for the action-domain scoping decision: it records exactly
//     what the port would have to reproduce, and shows that every one of
//     these takes an `id::action`, a vertex alternative this port's
//     four-way `vertex` union does not have.
//
//   "vertices" -- get_type_tree(const vertex&) (cc:201),
//     get_output_type_tree(const vertex&) (cc:250), get_arity(const vertex&)
//     (cc:403) and get_input_type_tree(const vertex&, i) (cc:283) over
//     single builtin / argument / contin vertices. Reaches cc:201's
//     is_builtin / is_argument / is_contin arms, cc:283's `is_contin ->
//     type_tree()` (empty) arm, and cc:250's is_argument -> unknown arm.
//
//   "programs" -- the corpus. Per program: get_type_tree(const combo_tree&)
//     (cc:1169, the UNREDUCED application tree), infer_type_tree (cc:1200,
//     the reduced signature), is_well_formed (cc:1210), type_tree_arity
//     (cc:352), contin_arity (cc:316), boolean_arity (cc:328),
//     action_result_arity (cc:340), infer_arity (cc:1242), explicit_arity
//     (cc:1260), get_signature_output (cc:1272), get_signature_inputs
//     (cc:1282), get_argument_abs_idx_set (cc:1115),
//     get_argument_abs_idx_from_zero_set (cc:1124),
//     does_contain_all_arg_up_to (cc:1220) for n in 0..5,
//     infer_arg_type_tree (cc:1135) and, per pre-order position,
//     infer_vertex_type (cc:1082).
//
//     The corpus is deliberately HALF ILL-TYPED. It reaches every
//     ill_formed producer in reduce_type_tree (cc:537):
//       * cc:660 wrong number of applied arguments (`not($1 $2)`,
//         `/($1)`, `/($1 $2 $3)`, `impulse($1 $2)`, `0<($1 $2)`)
//       * cc:778 a lambda-typed applied argument whose OUTPUT does not
//         inherit the expected input (`+(and($1 $2) 1)`,
//         `and(+($1 $2) $3)`, `exp(and($1 $2))`, `or($1 +($2 3))`,
//         `and(impulse($1) $2)`, `+(0<($1) 1)`)
//       * the CROSS-ARGUMENT mismatch that reaches ill_formed through
//         get_intersection (cc:951) inside infer_arg_type_tree rather
//         than through reduce_type_tree at all: `+(impulse($1) $1)`,
//         `and($1 0<($1))`, `or($1 +($1 1))`, `and($1 impulse($1))`.
//         These are the cases a per-argument "what does my parent want"
//         heuristic can get right and an operator/operand check cannot,
//         and vice versa -- both families are present on purpose.
//     Well-typed members pin the other direction, including the two
//     non-obvious facts a substitute is most likely to get wrong: a gap
//     in the argument indices stays `unknown` (`or($1 !$3)`), and a
//     childless junctor is well formed (`and()`, `+()`).
//
//   "inherit" -- inherit_type_tree (cc:440/:446), equal_type_tree (cc:435)
//     and get_intersection (cc:941/:951) over hand-written type-tree
//     STRINGS. Reaches cc:446's union arms (both directions), the lambda
//     arm with matching and mismatched child counts, the arg_list arm,
//     the application arm's unconditional `return false`, and all three
//     base cases; and cc:951's union-of-unions, lambda, unknown and
//     ill_formed arms.
//
//   "reduce" -- reduce_type_tree(tt, arg_types, tr, name) (cc:517) over
//     type-tree strings, with and without an arg_types vector. Reaches
//     cc:549's lambda(X)->X arm, cc:571's application(X)->X arm, the
//     arg_list cycling used by id::cond (cc:686-701), the `ils` "funky
//     special case" (cc:722), the input-substitution branch at cc:759,
//     the union dedup at cc:884, and the argument-substitution base case
//     at cc:930 (which needs type_tree_def.cc's is_argument_type/
//     arg_to_idx).
//
//   "parse" -- the type-tree STRING reader, opencog/util/tree.cc:70's
//     operator>>(istream&, tree<string>&) plus tree.h:3312's tree_convert,
//     printed back through tree.h:3305. Reaches both halves of that
//     reader's hand-written pre-pass (tree.cc:93-119): whitespace before
//     an open paren is dropped, and a "( )" holding only whitespace loses
//     BOTH parens, leaving the operator childless -- "union()" reads as a
//     childless `union`, not as `union(union)`.
//
//   "type_node_io" -- operator<<(ostream&, type_node&) (cc:1360) for every
//     ordinal 0..24 (so past id::argument_type, which prints "$1", "$2",
//     ... , and including id::list_type and id::ann_type, which have NO
//     case and print "UNKNOWN_HANDLE"), and operator>>(istream&,
//     type_node&) (cc:1409) for every spelling it accepts.
//
//   "misc" -- get_type_node (cc:372), convert_index (cc:380),
//     abs_min_arity (cc:389), argument_type_list_input_type (cc:396),
//     default_vertex_value (cc:35) and type_tree_def.cc's
//     is_argument_type (cc:29) / arg_to_idx (cc:34).
//
// Every case is wrapped in try/catch: an OC_ASSERT throws
// opencog::AssertionException (oc_assert.cc:46), and a case that throws
// is recorded as `"threw": true` rather than dropped, so the port can
// reproduce the failure instead of silently returning a value. The
// exception MESSAGE is deliberately not emitted: OC_ASSERT's text is a
// TRACE_INFO string carrying this machine's absolute $CLASSIC path, which
// would make the fixture bytes machine-specific for no gain.
//
// The logger is silenced (set_level(NONE), set_print_to_stdout_flag(false))
// because reduce_type_tree writes its type errors through logger().error()
// and that would otherwise interleave with the JSON on stdout. It changes
// no computed value.
//
// Build (see tools/capture_type_tree.py, which runs exactly this):
//
//   P=$PYMOSES_MOSES_PREFIX
//   C=$PYMOSES_CLASSIC_DIR
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/type_tree_probe \
//       tools/cpp_probe/type_tree_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CLAUDE.md and docs/CARRY-FORWARD.md. Every case below is a handful of
// nodes; the whole run is milliseconds.

#include <moses/comboreduct/combo/descriptions.h>
#include <moses/comboreduct/combo/iostream_combo.h>
#include <moses/comboreduct/combo/vertex.h>
#include <moses/comboreduct/type_checker/type_tree.h>
#include <moses/comboreduct/type_checker/type_tree_def.h>
#include <opencog/util/Logger.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

// ---------------------------------------------------------------------
// small JSON helpers
// ---------------------------------------------------------------------

static string json_escape(const string& s) {
  string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", (int)(unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

static void jstr(const string& s) { printf("\"%s\"", json_escape(s).c_str()); }

static string tt_str(const type_tree& tt) {
  stringstream ss;
  ss << tt;
  return ss.str();
}

static string tr_str(const combo_tree& tr) {
  stringstream ss;
  ss << tr;
  return ss.str();
}

static string node_str(type_node n) {
  stringstream ss;
  ss << n;
  return ss.str();
}

static type_tree parse_tt(const string& s) {
  stringstream ss;
  ss << s;
  type_tree tt;
  ss >> tt;
  return tt;
}

static combo_tree parse_tr(const string& s) {
  stringstream ss;
  ss << s;
  combo_tree tr;
  ss >> tr;
  return tr;
}

// ---------------------------------------------------------------------
// "builtins" -- descriptions.cc:100 builtins_properties
// ---------------------------------------------------------------------

static const char* builtin_names[] = {
    "null_vertex", "logical_true", "logical_false", "logical_and", "logical_or",
    "logical_not", "plus",         "times",         "div",         "exp",
    "log",         "sin",          "greater_than_zero", "impulse", "rand",
    "list",        "car",          "cdr",           "cons",        "foldr",
    "foldl",       "cond",         "equ",           "lambda",      "apply",
    "contin_if"};

static void dump_builtins() {
  printf("  \"builtins\": [\n");
  for (int b = 0; b < (int)id::builtin_count; ++b) {
    builtin bi = (builtin)b;
    printf("    {\"ord\": %d, \"name\": ", b);
    jstr(builtin_names[b]);
    printf(", \"arity\": %d, \"type_tree\": ", (int)get_arity(bi));
    jstr(tt_str(get_type_tree(bi)));
    printf(", \"output_type_tree\": ");
    jstr(tt_str(get_output_type_tree(bi)));
    printf(", \"input_type_trees\": [");
    for (int i = 0; i < 4; ++i) {
      if (i) printf(", ");
      jstr(tt_str(get_input_type_tree(bi, (arity_t)i)));
    }
    printf("]}%s\n", b + 1 == (int)id::builtin_count ? "" : ",");
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "actions" -- descriptions.cc:180 actions_properties
// ---------------------------------------------------------------------

static const char* action_names[] = {
    "sequential_and",  "sequential_or",    "sequential_exec", "action_not",
    "action_if",       "boolean_action_if", "contin_action_if", "action_action_if",
    "action_success",  "action_failure",   "action_while",    "boolean_while",
    "return_success",  "repeat_n"};

static void dump_actions() {
  printf("  \"actions\": [\n");
  for (int a = 0; a < (int)id::action_count; ++a) {
    action ac = (action)a;
    printf("    {\"ord\": %d, \"name\": ", a);
    jstr(action_names[a]);
    printf(", \"arity\": %d, \"type_tree\": ", (int)get_arity(ac));
    jstr(tt_str(get_type_tree(ac)));
    printf(", \"output_type_tree\": ");
    jstr(tt_str(get_output_type_tree(ac)));
    printf(", \"input_type_trees\": [");
    for (int i = 0; i < 4; ++i) {
      if (i) printf(", ");
      jstr(tt_str(get_input_type_tree(ac, (arity_t)i)));
    }
    printf("]}%s\n", a + 1 == (int)id::action_count ? "" : ",");
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "vertices" -- cc:201 / :250 / :283 / :403 over single vertices
// ---------------------------------------------------------------------

static const char* vertex_corpus[] = {
    "true", "false", "and", "or", "not", "+", "*", "/", "exp", "log", "sin",
    "0<", "impulse", "rand", "cond", "contin_if", "equ", "list", "car", "cdr",
    "cons", "foldr", "foldl", "apply", "$1", "$2", "$3", "!$1", "!$4", "$10",
    "0", "1", "-1", "2.5", "0.5"};

static void dump_vertices() {
  printf("  \"vertices\": [\n");
  const int n = (int)(sizeof(vertex_corpus) / sizeof(vertex_corpus[0]));
  for (int k = 0; k < n; ++k) {
    combo_tree tr = parse_tr(vertex_corpus[k]);
    vertex v = *tr.begin();
    printf("    {\"vertex\": ");
    jstr(vertex_corpus[k]);
    printf(", \"printed\": ");
    jstr(tr_str(tr));
    printf(", \"get_type_tree\": ");
    jstr(tt_str(get_type_tree(v)));
    printf(", \"get_output_type_tree\": ");
    jstr(tt_str(get_output_type_tree(v)));
    printf(", \"get_arity\": %d, \"input_type_trees\": [", (int)get_arity(v));
    for (int i = 0; i < 4; ++i) {
      if (i) printf(", ");
      jstr(tt_str(get_input_type_tree(v, (arity_t)i)));
    }
    printf("]}%s\n", k + 1 == n ? "" : ",");
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "programs" -- the corpus
// ---------------------------------------------------------------------

static const char* program_corpus[] = {
    // --- well typed, boolean -------------------------------------------
    "and($1 $2)",
    "or(and($2 !$3) $1 !$4 $5)",
    "not($1)",
    "$1",
    "!$1",
    "true",
    "false",
    "or($1 !$3)",
    "and(or($1 $2) $3)",
    "0<($1)",
    "and(0<($1) $2)",
    "and(0<(+($1 $2)) $3)",
    "or(and($1 !$2) and($3 $4) !$5)",
    "and()",
    "or()",
    "not(and($1 $2))",
    // --- well typed, contin --------------------------------------------
    "+(6 *(6 $1))",
    "*(2 $1)",
    "+(6 *(6 impulse($1)))",
    "impulse($1)",
    "/($1 $2)",
    "log($1)",
    "exp($1)",
    "sin($1)",
    "0",
    "1.5",
    "rand",
    "+($1)",
    "+()",
    "*()",
    "+($1 $2 $3 $4)",
    "impulse(and($1 $2))",
    "0<(*($1 $2))",
    "contin_if($1 $2 $3)",
    // --- ill typed: cross-argument mismatch (via get_intersection) -----
    "+(impulse($1) $1)",
    "and($1 0<($1))",
    "or($1 +($1 1))",
    "and($1 impulse($1))",
    "*(impulse($2) $2)",
    "and(0<($1) not($1))",
    // --- ill typed: operand's OUTPUT does not inherit the input --------
    "+(and($1 $2) 1)",
    "and(+($1 $2) $3)",
    "exp(and($1 $2))",
    "or($1 +($2 3))",
    "and(impulse($1) $2)",
    "+(0<($1) 1)",
    "not(+($1 $2))",
    "log(or($1 $2))",
    // --- ill typed: wrong number of applied arguments ------------------
    "not($1 $2)",
    "/($1)",
    "/($1 $2 $3)",
    "impulse($1 $2)",
    "0<($1 $2)",
    "exp($1 $2)",
    "contin_if($1 $2)",
    // --- excluded-domain shapes, recorded rather than ported -----------
    "cond($1 2 3)",
    "cond($1 2 $2 3 4)",
    "equ($1 $2)",
    "list($1 $2)",
    "car(list($1))",
    // --- higher argument indices ---------------------------------------
    "and($1 $9)",
    "or($2 $4 $6)",
};

static void dump_one_program(const char* combo, bool last) {
  printf("    {\"combo\": ");
  jstr(combo);
  try {
    combo_tree tr = parse_tr(combo);
    printf(", \"parsed\": ");
    jstr(tr_str(tr));
    if (tr.empty()) {
      printf(", \"empty\": true}%s\n", last ? "" : ",");
      return;
    }
    type_tree raw = get_type_tree(tr);
    printf(", \"get_type_tree\": ");
    jstr(tt_str(raw));

    type_tree tt = infer_type_tree(tr);
    printf(", \"infer_type_tree\": ");
    jstr(tt_str(tt));
    printf(", \"is_well_formed\": %d", is_well_formed(tt) ? 1 : 0);
    printf(", \"type_tree_arity\": %d", (int)type_tree_arity(tt));
    printf(", \"contin_arity\": %d", (int)contin_arity(tt));
    printf(", \"boolean_arity\": %d", (int)boolean_arity(tt));
    printf(", \"action_result_arity\": %d", (int)action_result_arity(tt));
    printf(", \"infer_arity\": %d", (int)infer_arity(tr));
    printf(", \"explicit_arity\": %d", (int)explicit_arity(tr));
    printf(", \"signature_output\": ");
    jstr(tt_str(get_signature_output(tt)));
    printf(", \"signature_inputs\": [");
    {
      type_tree_seq ins = get_signature_inputs(tt);
      for (size_t i = 0; i < ins.size(); ++i) {
        if (i) printf(", ");
        jstr(tt_str(ins[i]));
      }
    }
    printf("], \"arg_abs_idx\": [");
    {
      arity_set s = get_argument_abs_idx_set(tr);
      bool first = true;
      for (arity_t i : s) {
        if (!first) printf(", ");
        first = false;
        printf("%d", (int)i);
      }
    }
    printf("], \"arg_abs_idx_from_zero\": [");
    {
      arity_set s = get_argument_abs_idx_from_zero_set(tr);
      bool first = true;
      for (arity_t i : s) {
        if (!first) printf(", ");
        first = false;
        printf("%d", (int)i);
      }
    }
    printf("], \"contains_all_arg_up_to\": [");
    for (int n = 0; n <= 5; ++n) {
      if (n) printf(", ");
      printf("%d", does_contain_all_arg_up_to(tr, (arity_t)n) ? 1 : 0);
    }
    printf("], \"infer_arg_types\": [");
    {
      type_tree_seq ats;
      infer_arg_type_tree(tr, ats);
      for (size_t i = 0; i < ats.size(); ++i) {
        if (i) printf(", ");
        jstr(tt_str(ats[i]));
      }
    }
    printf("], \"infer_vertex_type\": [");
    {
      bool first = true;
      for (combo_tree::iterator it = tr.begin(); it != tr.end(); ++it) {
        if (!first) printf(", ");
        first = false;
        printf("{\"idx\": %u, \"vertex\": ", pre_order_index(tr, it));
        {
          stringstream ss;
          ss << *it;
          jstr(ss.str());
        }
        printf(", \"type\": ");
        jstr(tt_str(infer_vertex_type(tr, it)));
        printf("}");
      }
    }
    printf("]}%s\n", last ? "" : ",");
  } catch (const std::exception&) {
    printf(", \"threw\": true}%s\n", last ? "" : ",");
  }
}

static void dump_programs() {
  printf("  \"programs\": [\n");
  const int n = (int)(sizeof(program_corpus) / sizeof(program_corpus[0]));
  for (int k = 0; k < n; ++k) dump_one_program(program_corpus[k], k + 1 == n);
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "inherit" -- cc:435 / :440 / :446 / :941 / :951
// ---------------------------------------------------------------------

struct tt_pair {
  const char* a;
  const char* b;
};

static const tt_pair inherit_corpus[] = {
    {"boolean", "boolean"},
    {"boolean", "contin"},
    {"contin", "boolean"},
    {"boolean", "unknown"},
    {"unknown", "boolean"},
    {"unknown", "unknown"},
    {"ill_formed", "boolean"},
    {"boolean", "ill_formed"},
    {"ill_formed", "ill_formed"},
    {"ill_formed", "unknown"},
    {"unknown", "ill_formed"},
    {"action_result", "action_result"},
    {"action_result", "unknown"},
    {"action_result", "boolean"},
    {"enum_t", "enum_t"},
    {"$1", "$1"},
    {"$1", "$2"},
    {"$1", "unknown"},
    {"unknown", "$1"},
    {"union(boolean contin)", "boolean"},
    {"boolean", "union(boolean contin)"},
    {"union(boolean contin)", "union(boolean contin)"},
    {"union(boolean contin)", "union(contin action_result)"},
    {"union(boolean contin)", "unknown"},
    {"union(boolean)", "boolean"},
    {"arg_list(boolean)", "arg_list(boolean)"},
    {"arg_list(boolean)", "arg_list(contin)"},
    {"arg_list(boolean)", "boolean"},
    {"->(boolean boolean)", "->(boolean boolean)"},
    {"->(boolean boolean)", "->(unknown boolean)"},
    {"->(unknown boolean)", "->(boolean boolean)"},
    {"->(boolean contin)", "->(boolean boolean)"},
    {"->(boolean boolean boolean)", "->(boolean boolean)"},
    {"->(arg_list(boolean) boolean)", "->(boolean boolean boolean)"},
    {"->(boolean boolean)", "boolean"},
    {"application(boolean)", "boolean"},
    {"boolean", "application(boolean)"},
    {"->(contin contin)", "->(contin contin)"},
    {"union(->(boolean boolean) contin)", "union(->(boolean boolean) contin)"},
    // Neither side inherits the other (contin is absent from the second,
    // action_result from the first), so get_intersection reaches its
    // union-of-unions branch (cc:960) and collects TWO well-formed
    // intersections of DIFFERENT sizes -- `boolean` (1 node) and
    // `->(contin contin)` (3). That makes the direction of
    // size_tree_order's `s1 < s2` (tree.h:3219) decide the output order.
    {"union(boolean ->(contin contin) contin)",
     "union(boolean ->(contin contin) action_result)"},
    {"union(boolean ->(contin contin) ->(boolean boolean) contin)",
     "union(boolean ->(contin contin) ->(boolean boolean) action_result)"},
};

static void dump_inherit() {
  printf("  \"inherit\": [\n");
  const int n = (int)(sizeof(inherit_corpus) / sizeof(inherit_corpus[0]));
  for (int k = 0; k < n; ++k) {
    printf("    {\"a\": ");
    jstr(inherit_corpus[k].a);
    printf(", \"b\": ");
    jstr(inherit_corpus[k].b);
    try {
      type_tree a = parse_tt(inherit_corpus[k].a);
      type_tree b = parse_tt(inherit_corpus[k].b);
      printf(", \"a_inherits_b\": %d", inherit_type_tree(a, b) ? 1 : 0);
      printf(", \"b_inherits_a\": %d", inherit_type_tree(b, a) ? 1 : 0);
      printf(", \"equal\": %d", equal_type_tree(a, b) ? 1 : 0);
      printf(", \"intersection\": ");
      jstr(tt_str(get_intersection(a, b)));
      printf("}%s\n", k + 1 == n ? "" : ",");
    } catch (const std::exception&) {
      printf(", \"threw\": true}%s\n", k + 1 == n ? "" : ",");
    }
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "reduce" -- cc:517 / :529 / :537
// ---------------------------------------------------------------------

struct reduce_case {
  const char* tt;
  const char* args[4];  // NULL-terminated list of arg types
};

static const reduce_case reduce_corpus[] = {
    {"boolean", {0}},
    {"->(boolean)", {0}},
    {"->(->(boolean))", {0}},
    {"application(boolean)", {0}},
    {"application(->(boolean boolean) boolean)", {0}},
    {"application(->(boolean boolean) contin)", {0}},
    {"application(->(arg_list(boolean) boolean) boolean boolean boolean)", {0}},
    {"application(->(arg_list(boolean) boolean))", {0}},
    {"application(->(arg_list(boolean) boolean) contin)", {0}},
    {"application(->(contin contin contin) contin contin)", {0}},
    {"application(->(contin contin contin) contin)", {0}},
    {"application(->(contin contin contin) contin contin contin)", {0}},
    {"application(->(contin boolean) contin)", {0}},
    {"application(->(boolean contin) boolean)", {0}},
    {"application(->(arg_list(boolean unknown) unknown unknown) boolean contin "
     "boolean contin contin)",
     {0}},
    {"application(->(arg_list(boolean unknown) unknown unknown) contin)", {0}},
    {"application(->(arg_list(contin) contin) application(->(boolean contin) "
     "boolean) contin)",
     {0}},
    {"application(->(boolean boolean) ->(contin boolean))", {0}},
    {"application(->(boolean boolean) ->(contin contin))", {0}},
    {"application(boolean boolean)", {0}},
    {"union(boolean boolean)", {0}},
    {"union(boolean unknown)", {0}},
    {"union(boolean contin)", {0}},
    {"union(boolean)", {0}},
    {"->(boolean application(->(boolean boolean) boolean))", {0}},
    {"$1", {"boolean", 0}},
    {"$2", {"boolean", "contin", 0}},
    {"$3", {"boolean", 0}},
    {"application(->(boolean boolean) $1)", {"boolean", 0}},
    {"application(->(boolean boolean) $1)", {"contin", 0}},
    {"application(->(contin contin) $1)", {"unknown", 0}},
    {"->($1 $2)", {"boolean", "contin", 0}},
    // An arg_list whose CHILD is itself an application. This is the only
    // shape found that distinguishes cc:917's `for (tit = it.begin(); tit
    // != it.end(); ++tit)` -- a WHOLE-TREE pre-order walk, because
    // `it.end()`'s node is null and a pre-order ++ walks up and out of the
    // subtree -- from the subtree-restricted walk it reads like. Both rows
    // were found by differential search over the two implementations, then
    // measured here.
    {"application(->(arg_list(application(->(boolean boolean) boolean)) boolean) boolean)",
     {0}},
    {"->(arg_list(application(->(boolean boolean) boolean)) boolean)", {0}},
    {"application(->(arg_list(boolean) application(->(boolean boolean) boolean)) boolean)",
     {0}},
    {"application(->(arg_list(contin) contin) application(->(arg_list(contin) contin) "
     "contin contin) contin)",
     {0}},
    {"application(->(arg_list($1) boolean) $1 $1)", {"boolean", 0}},
};

static void dump_reduce() {
  printf("  \"reduce\": [\n");
  const int n = (int)(sizeof(reduce_corpus) / sizeof(reduce_corpus[0]));
  for (int k = 0; k < n; ++k) {
    printf("    {\"tt\": ");
    jstr(reduce_corpus[k].tt);
    printf(", \"arg_types\": [");
    type_tree_seq ats;
    for (int i = 0; i < 4 && reduce_corpus[k].args[i]; ++i) {
      if (i) printf(", ");
      jstr(reduce_corpus[k].args[i]);
      ats.push_back(parse_tt(reduce_corpus[k].args[i]));
    }
    printf("]");
    try {
      type_tree tt = parse_tt(reduce_corpus[k].tt);
      reduce_type_tree(tt, ats, combo_tree(), string("PROBE"));
      printf(", \"reduced\": ");
      jstr(tt_str(tt));
      printf(", \"is_well_formed\": %d", is_well_formed(tt) ? 1 : 0);
      printf("}%s\n", k + 1 == n ? "" : ",");
    } catch (const std::exception&) {
      printf(", \"threw\": true}%s\n", k + 1 == n ? "" : ",");
    }
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "parse" -- opencog/util/tree.cc:70 operator>>(istream&, tree<string>&)
// plus tree.h:3312's tree_convert, then printed back through tree.h:3305.
// Pins the two halves of that function's hand-written pre-pass
// (tree.cc:93-119): whitespace before an open paren is dropped, and a
// "( )" containing only whitespace loses BOTH parens so the operator ends
// up childless.
// ---------------------------------------------------------------------

static const char* parse_corpus[] = {
    "boolean",
    "  boolean  ",
    "->(boolean boolean)",
    "->(boolean  boolean)",
    "-> (boolean boolean)",
    "->  ( boolean boolean )",
    "->(arg_list(boolean) boolean)",
    "->( arg_list( boolean ) boolean )",
    "union()",
    "->()",
    "boolean()",
    "arg_list()",
    "->(arg_list() boolean)",
    "application(->(boolean boolean) boolean)",
    "union(boolean contin)",
    "$1",
    "->($1 $2)",
    "->(arg_list(boolean unknown) unknown unknown)",
};

static void dump_parse() {
  printf("  \"parse\": [\n");
  const int n = (int)(sizeof(parse_corpus) / sizeof(parse_corpus[0]));
  for (int k = 0; k < n; ++k) {
    printf("    {\"text\": ");
    jstr(parse_corpus[k]);
    try {
      type_tree tt = parse_tt(parse_corpus[k]);
      printf(", \"printed\": ");
      jstr(tt_str(tt));
      printf(", \"size\": %d", tt.size());
      printf("}%s\n", k + 1 == n ? "" : ",");
    } catch (const std::exception&) {
      printf(", \"threw\": true}%s\n", k + 1 == n ? "" : ",");
    }
  }
  printf("  ],\n");
}

// ---------------------------------------------------------------------
// "type_node_io" -- cc:1360 / :1409
// ---------------------------------------------------------------------

static const char* type_node_spellings[] = {
    "->",         "lambda",       "lambda_type",  "application", "application_type",
    "union",      "union_type",   "arg_list",     "arg_list_type",
    "boolean",    "boolean_type", "contin",       "contin_t",    "contin_type",
    "enum",       "enum_t",       "enum_type",    "list",        "list_t",
    "list_type",  "action_result", "action_result_type",
    "definite_object", "definite_object_type",
    "action_definite_object", "action_definite_object_type",
    "indefinite_object", "indefinite_object_type",
    "message",    "message_type", "action_symbol", "action_symbol_type",
    "wild_card",  "wild_card_type", "unknown",    "unknown_type",
    "ill_formed", "ill_formed_type", "$1",        "$2",          "$10",
    "$-1"};

static void dump_type_node_io() {
  printf("  \"type_node_io\": {\n");
  printf("    \"ostream\": [");
  for (int i = 0; i <= 24; ++i) {
    if (i) printf(", ");
    printf("{\"ord\": %d, \"str\": ", i);
    jstr(node_str((type_node)i));
    printf("}");
  }
  printf("],\n    \"istream\": [");
  const int n = (int)(sizeof(type_node_spellings) / sizeof(type_node_spellings[0]));
  for (int k = 0; k < n; ++k) {
    if (k) printf(", ");
    printf("{\"str\": ");
    jstr(type_node_spellings[k]);
    try {
      stringstream ss;
      ss << type_node_spellings[k];
      type_node tn;
      ss >> tn;
      printf(", \"ord\": %d", (int)tn);
    } catch (const std::exception&) {
      printf(", \"threw\": true");
    }
    printf("}");
  }
  printf("]\n  },\n");
}

// ---------------------------------------------------------------------
// "misc" -- cc:35 / :372 / :380 / :389 / :396 and type_tree_def.cc
// ---------------------------------------------------------------------

static void dump_misc() {
  printf("  \"misc\": {\n");

  printf("    \"convert_index\": [");
  {
    bool first = true;
    for (int arity = -4; arity <= 4; ++arity) {
      for (int index = 0; index <= 5; ++index) {
        if (!first) printf(", ");
        first = false;
        printf("{\"arity\": %d, \"index\": %d, \"result\": %d}", arity, index,
               (int)convert_index((arity_t)arity, (arity_t)index));
      }
    }
  }
  printf("],\n");

  printf("    \"abs_min_arity\": [");
  for (int arity = -4; arity <= 4; ++arity) {
    if (arity != -4) printf(", ");
    printf("{\"arity\": %d, \"result\": %d}", arity,
           (int)abs_min_arity((arity_t)arity));
  }
  printf("],\n");

  printf("    \"argument_type_list_input_type\": [");
  {
    type_tree_seq atl;
    atl.push_back(parse_tt("boolean"));
    atl.push_back(parse_tt("contin"));
    atl.push_back(parse_tt("action_result"));
    bool first = true;
    // Only combinations whose convert_index result is a valid atl index.
    int cases[][2] = {{3, 0}, {3, 1}, {3, 2}, {-3, 0}, {-3, 1}, {-3, 2},
                      {-3, 3}, {-3, 5}, {-2, 1}, {-2, 4}};
    for (auto& c : cases) {
      if (!first) printf(", ");
      first = false;
      printf("{\"arity\": %d, \"index\": %d, \"result\": ", c[0], c[1]);
      jstr(tt_str(argument_type_list_input_type(atl, (arity_t)c[0],
                                                (arity_t)c[1])));
      printf("}");
    }
  }
  printf("],\n");

  printf("    \"get_type_node\": [");
  {
    const char* cases[] = {"boolean", "contin", "unknown", "ill_formed", "$1",
                           "->(boolean boolean)"};
    for (int k = 0; k < 6; ++k) {
      if (k) printf(", ");
      printf("{\"tt\": ");
      jstr(cases[k]);
      try {
        type_tree tt = parse_tt(cases[k]);
        printf(", \"result\": %d", (int)get_type_node(tt));
      } catch (const std::exception&) {
        printf(", \"threw\": true");
      }
      printf("}");
    }
  }
  printf("],\n");

  printf("    \"default_vertex_value\": [");
  for (int i = 0; i <= 18; ++i) {
    if (i) printf(", ");
    printf("{\"ord\": %d, \"value\": ", i);
    stringstream ss;
    ss << default_vertex_value((type_node)i);
    jstr(ss.str());
    printf("}");
  }
  printf("],\n");

  printf("    \"is_argument_type\": [");
  for (int i = 0; i <= 22; ++i) {
    if (i) printf(", ");
    printf("{\"ord\": %d, \"result\": %d", i,
           is_argument_type((type_node)i) ? 1 : 0);
    if (is_argument_type((type_node)i))
      printf(", \"arg_to_idx\": %u", arg_to_idx((type_node)i));
    printf("}");
  }
  printf("]\n  }\n");
}

// ---------------------------------------------------------------------

int main() {
  logger().set_level(Logger::NONE);
  logger().set_print_to_stdout_flag(false);
  printf("{\n");
  dump_builtins();
  dump_actions();
  dump_vertices();
  dump_programs();
  dump_inherit();
  dump_reduce();
  dump_parse();
  dump_type_node_io();
  dump_misc();
  printf("}\n");
  return 0;
}
