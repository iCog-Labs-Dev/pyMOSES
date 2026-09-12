// tools/cpp_probe/tree_probe.cc
//
// Dumps opencog::tree<std::string> splice-operation and traversal-order
// results as JSON, for differential testing against pymoses/util/tree.py.
//
// tree.h declares a few OC_ASSERT call sites but is otherwise header-only;
// IGNORE_OC_ASSERT avoids pulling in opencog::cassert() from libcogutil,
// which this probe does not link against.
#define IGNORE_OC_ASSERT
#include <opencog/util/tree.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
typedef tree<std::string> stree;

// ---- tiny JSON helpers (data values here are plain identifiers: no
// escaping needed beyond wrapping in quotes) ----

static std::string quote(const std::string &s)
{
    return "\"" + s + "\"";
}

// C++: opencog/util/tree.h:911 tree::begin(const iterator_base&) / :922 end
// -- mirrors the brief's to_sexpr() helper, walking sibling ranges via
// iterator_base::begin()/end() (tree.h:2346-2361).
static std::string sexpr(const stree &t, stree::iterator_base node)
{
    if (!t.is_valid(node)) return "null";
    std::ostringstream os;
    os << "{\"data\":" << quote(*node) << ",\"children\":[";
    bool first = true;
    for (stree::sibling_iterator sib = node.begin(); sib != node.end(); ++sib) {
        if (!first) os << ",";
        first = false;
        os << sexpr(t, stree::iterator_base(sib.node));
    }
    os << "]}";
    return os.str();
}

static std::string sexpr_tree(const stree &t)
{
    if (t.empty()) return "null";
    return sexpr(t, t.begin());
}

static std::string str_array(const std::vector<std::string> &xs)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ",";
        os << quote(xs[i]);
    }
    os << "]";
    return os.str();
}

// ---- small tree-building helpers, using only the ported API surface ----

static stree::iterator add_child(stree &t, stree::iterator parent, const std::string &data)
{
    return t.append_child(parent, data);
}

int main()
{
    std::vector<std::pair<std::string, std::string>> cases;

    // -- append_child (plain data) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        add_child(t, t.begin(), "b");
        cases.push_back({"append_child_data", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- append_child (subtree: deep copy, source untouched) --
    {
        stree src("s");
        add_child(src, src.begin(), "s1");
        add_child(src, src.begin(), "s2");
        stree dst("d");
        dst.append_child(dst.begin(), src.begin());
        cases.push_back({"append_child_subtree", "{\"dest\":" + sexpr_tree(dst) +
                                                      ",\"source_untouched\":" + sexpr_tree(src) + "}"});
    }

    // -- prepend_child --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        t.prepend_child(t.begin(), "z");
        cases.push_back({"prepend_child", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- append_children (iterable of plain data) --
    {
        stree t("root");
        t.append_children(t.begin(), {"x", "y", "z"});
        cases.push_back({"append_children", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert (previous sibling) --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        (void)a;
        t.insert(b, "n");
        cases.push_back({"insert", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert_after --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        t.insert_after(b, "n");
        cases.push_back({"insert_after", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert_subtree --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        stree src("s");
        add_child(src, src.begin(), "s1");
        t.insert_subtree(b, src.begin());
        cases.push_back({"insert_subtree", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert_subtree_after --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        stree src("s");
        add_child(src, src.begin(), "s1");
        t.insert_subtree_after(b, src.begin());
        cases.push_back({"insert_subtree_after", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert_above (internal node with children) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, b, "b2");
        add_child(t, t.begin(), "c");
        t.insert_above(b, "n");
        cases.push_back({"insert_above_internal", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- insert_above on the tree root itself --
    {
        stree t("R");
        add_child(t, t.begin(), "r1");
        add_child(t, t.begin(), "r2");
        t.insert_above(t.begin(), "N");
        cases.push_back({"insert_above_root", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- erase (subtree + returned successor) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, b, "b2");
        add_child(t, t.begin(), "c");
        stree::iterator ret = t.erase(b);
        std::string ret_data = t.is_valid(ret) ? *ret : std::string("<end>");
        cases.push_back({"erase", "{\"result\":" + sexpr_tree(t) + ",\"returned\":" + quote(ret_data) + "}"});
    }

    // -- erase the sole root of a tree --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator ret = t.erase(t.begin());
        bool ret_is_end = !t.is_valid(ret);
        cases.push_back({"erase_root", "{\"result\":" + sexpr_tree(t) + ",\"empty\":" +
                                            (t.empty() ? std::string("true") : std::string("false")) +
                                            ",\"returned_is_end\":" + (ret_is_end ? "true" : "false") + "}"});
    }

    // -- erase_children --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, b, "b2");
        add_child(t, t.begin(), "c");
        t.erase_children(b);
        cases.push_back({"erase_children", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- erase_children on an already-childless node (no-op) --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        t.erase_children(a);
        cases.push_back({"erase_children_leaf", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- replace (plain data, same node identity) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        t.replace(b, std::string("B2"));
        cases.push_back({"replace_data", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- replace (subtree, source untouched) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, t.begin(), "c");
        stree src("s");
        add_child(src, src.begin(), "s1");
        add_child(src, src.begin(), "s2");
        t.replace(b, src.begin());
        cases.push_back({"replace_subtree", "{\"result\":" + sexpr_tree(t) +
                                                 ",\"source_untouched\":" + sexpr_tree(src) + "}"});
    }

    // -- flatten (node with children) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, b, "b2");
        add_child(t, t.begin(), "c");
        t.flatten(b);
        cases.push_back({"flatten", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- flatten a childless node (no-op) --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        add_child(t, t.begin(), "b");
        t.flatten(a);
        cases.push_back({"flatten_childless", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- wrap (node with children moves under the new node) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");
        add_child(t, b, "b2");
        add_child(t, t.begin(), "c");
        t.wrap(b, std::string("W"));
        cases.push_back({"wrap", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- reparent(new_parent, from) --
    {
        stree t("root");
        stree::iterator src = add_child(t, t.begin(), "src");
        add_child(t, src, "s1");
        add_child(t, src, "s2");
        stree::iterator dst = add_child(t, t.begin(), "dst");
        t.reparent(dst, src);
        cases.push_back({"reparent", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- move_after --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        stree::iterator d = add_child(t, t.begin(), "d");
        (void)a;
        t.move_after(b, d);
        cases.push_back({"move_after", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- move_before --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, t.begin(), "c");
        stree::iterator d = add_child(t, t.begin(), "d");
        t.move_before(b, d);
        cases.push_back({"move_before", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- move_ontop (source NOT adjacent to target -- the normal, in-use shape) --
    {
        stree t("root");
        add_child(t, t.begin(), "a");
        add_child(t, t.begin(), "b");
        stree::iterator c = add_child(t, t.begin(), "c");
        add_child(t, c, "c1");
        add_child(t, c, "c2");
        add_child(t, t.begin(), "d"); // sits between target and source: keeps them non-adjacent
        stree::iterator e = add_child(t, t.begin(), "e");
        add_child(t, e, "e1");
        t.move_ontop(c, e);
        cases.push_back({"move_ontop", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- move_ontop: KNOWN UPSTREAM DEFECT when source is target's immediate next
    // sibling (opencog/util/tree.h:1708-1750). b_next_sibling is captured as
    // target->next_sibling *before* target is erased; when that equals source
    // itself, the final `src->next_sibling = b_next_sibling` assigns
    // src->next_sibling = src -- a self-cycle -- and parent->last_child is left
    // stale. Any traversal that walks off this node via next_sibling (including
    // this tree's own destructor) never terminates / corrupts the heap, so this
    // case is deliberately probed by raw pointer inspection ONLY, on a
    // deliberately leaked tree (never destructed, never traversed). Do not
    // "clean this up" into a normal stack tree + sexpr_tree() dump -- that
    // hangs or aborts, which is the whole point being demonstrated here.
    {
        stree *t = new stree("root"); // leaked on purpose -- see comment above
        add_child(*t, t->begin(), "a");
        add_child(*t, t->begin(), "b");
        stree::iterator c = add_child(*t, t->begin(), "c");
        add_child(*t, c, "c1");
        add_child(*t, c, "c2");
        stree::iterator e = add_child(*t, t->begin(), "e"); // e is c's immediate next sibling
        add_child(*t, e, "e1");

        bool target_next_was_source = (c.node->next_sibling == e.node);
        auto *prev_of_c_before = c.node->prev_sibling; // "b"

        t->move_ontop(c, e);

        bool source_next_is_self = (e.node->next_sibling == e.node);
        std::string source_prev_data = e.node->prev_sibling ? e.node->prev_sibling->data : std::string("(null)");
        std::string last_child_data =
            t->begin().node->last_child ? t->begin().node->last_child->data : std::string("(null)");
        bool last_child_is_stale = (t->begin().node->last_child == prev_of_c_before);

        std::ostringstream os;
        os << "{\"target_next_was_source\":" << (target_next_was_source ? "true" : "false")
           << ",\"source_next_sibling_is_self\":" << (source_next_is_self ? "true" : "false")
           << ",\"source_prev_sibling_data\":" << quote(source_prev_data)
           << ",\"parent_last_child_data\":" << quote(last_child_data)
           << ",\"parent_last_child_is_stale\":" << (last_child_is_stale ? "true" : "false") << "}";
        cases.push_back({"move_ontop_adjacent_sibling_bug", os.str()});
        // t is intentionally never deleted: destructing it would traverse the
        // now-cyclic sibling chain (~tree -> clear() -> erase()) and hang/abort.
    }

    // -- sort (by key = string length; children carry their own child along) --
    {
        stree t("root");
        stree::iterator bb = add_child(t, t.begin(), "bb");
        add_child(t, bb, "bb1");
        stree::iterator a = add_child(t, t.begin(), "a");
        add_child(t, a, "a1");
        stree::iterator ccc = add_child(t, t.begin(), "ccc");
        add_child(t, ccc, "ccc1");
        auto by_len = [](const std::string &x, const std::string &y) { return x.size() < y.size(); };
        t.sort(t.begin(t.begin()), t.end(t.begin()), by_len, false);
        cases.push_back({"sort", "{\"result\":" + sexpr_tree(t) + "}"});
    }

    // -- equal_subtree --
    {
        stree t1("root");
        stree::iterator a1 = add_child(t1, t1.begin(), "a");
        add_child(t1, a1, "x");
        add_child(t1, a1, "y");

        stree t2("root");
        stree::iterator a2 = add_child(t2, t2.begin(), "a");
        add_child(t2, a2, "x");
        add_child(t2, a2, "y");

        stree t3("root");
        stree::iterator a3 = add_child(t3, t3.begin(), "a");
        add_child(t3, a3, "x");
        add_child(t3, a3, "DIFFERENT");

        stree t4("root");
        stree::iterator a4 = add_child(t4, t4.begin(), "a");
        add_child(t4, a4, "x");

        bool identical_shape = t1.equal_subtree(t1.begin(), t2.begin());
        bool different_data = t1.equal_subtree(t1.begin(), t3.begin());
        bool different_children_count = t1.equal_subtree(t1.begin(), t4.begin());
        bool same_object = t1.equal_subtree(t1.begin(), t1.begin());

        std::ostringstream os;
        os << "{\"identical_shape\":" << (identical_shape ? "true" : "false")
           << ",\"different_data\":" << (different_data ? "true" : "false")
           << ",\"different_children_count\":" << (different_children_count ? "true" : "false")
           << ",\"same_object\":" << (same_object ? "true" : "false") << "}";
        cases.push_back({"equal_subtree", os.str()});
    }

    // -- max_depth --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        stree::iterator a1 = add_child(t, a, "a1");
        add_child(t, t.begin(), "leaf");

        std::ostringstream os;
        os << "{\"root\":" << t.max_depth(t.begin()) << ",\"a\":" << t.max_depth(a)
           << ",\"a1\":" << t.max_depth(a1) << "}";
        cases.push_back({"max_depth", os.str()});
    }

    // -- max_depth of a single (childless) node --
    {
        stree t("only");
        std::ostringstream os;
        os << "{\"result\":" << t.max_depth(t.begin()) << "}";
        cases.push_back({"max_depth_single", os.str()});
    }

    // -- count (subtree_size) --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        stree::iterator a1 = add_child(t, a, "a1");
        add_child(t, a, "a2");
        add_child(t, t.begin(), "b");

        std::ostringstream os;
        os << "{\"root\":" << t.subtree_size(t.begin()) << ",\"a\":" << t.subtree_size(a)
           << ",\"a1\":" << t.subtree_size(a1) << "}";
        cases.push_back({"count", os.str()});
    }

    // -- child / children / number_of_children / number_of_siblings / next_sibling /
    // previous_sibling --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        stree::iterator b = add_child(t, t.begin(), "b");
        stree::iterator c = add_child(t, t.begin(), "c");

        std::vector<std::string> kids;
        for (stree::sibling_iterator sib = t.begin(t.begin()); sib != t.end(t.begin()); ++sib)
            kids.push_back(*sib);

        std::ostringstream os;
        os << "{\"child0\":" << quote(*t.child(t.begin(), 0)) << ",\"child2\":" << quote(*t.child(t.begin(), 2))
           << ",\"children\":" << str_array(kids) << ",\"number_of_children_root\":" << t.number_of_children(t.begin())
           << ",\"number_of_children_a\":" << t.number_of_children(a)
           << ",\"number_of_siblings_b\":" << t.number_of_siblings(b) << ",\"next_sibling_a\":" << quote(*t.next_sibling(a))
           << ",\"previous_sibling_c\":" << quote(*t.previous_sibling(c))
           << ",\"next_sibling_c_is_end\":" << (t.is_valid(t.next_sibling(c)) ? "false" : "true")
           << ",\"previous_sibling_a_is_end\":" << (t.is_valid(t.previous_sibling(a)) ? "false" : "true") << "}";
        cases.push_back({"navigation", os.str()});
    }

    // -- pre-order / post-order / leaf traversal orders --
    {
        stree t("root");
        stree::iterator a = add_child(t, t.begin(), "a");
        add_child(t, a, "a1");
        add_child(t, a, "a2");
        stree::iterator b = add_child(t, t.begin(), "b");
        add_child(t, b, "b1");

        std::vector<std::string> pre, post, leaf;
        for (stree::pre_order_iterator it = t.begin(); it != t.end(); ++it)
            pre.push_back(*it);
        for (stree::post_order_iterator it = t.begin_post(); it != t.end_post(); ++it)
            post.push_back(*it);
        for (stree::leaf_iterator it = t.begin_leaf(); it != t.end_leaf(); ++it)
            leaf.push_back(*it);

        cases.push_back({"traversal_orders", "{\"pre_order\":" + str_array(pre) + ",\"post_order\":" +
                                                  str_array(post) + ",\"leaf_order\":" + str_array(leaf) + "}"});
    }

    // -- hazard: erasing the next sibling mid sibling-iteration --
    {
        stree t("root");
        add_child(t, t.begin(), "x");
        stree::iterator y = add_child(t, t.begin(), "y");
        add_child(t, t.begin(), "z");

        std::vector<std::string> observed;
        stree::sibling_iterator sib = t.begin(t.begin());
        while (sib != t.end(t.begin())) {
            observed.push_back(*sib);
            if (*sib == "x") t.erase(y); // erase the *next* sibling before we reach it
            ++sib;
        }
        cases.push_back({"hazard_erase_next_sibling", "{\"observed\":" + str_array(observed) + "}"});
    }

    // -- hazard: replacing the next sibling's data mid sibling-iteration --
    {
        stree t("root");
        add_child(t, t.begin(), "x");
        stree::iterator y = add_child(t, t.begin(), "y");
        add_child(t, t.begin(), "z");

        std::vector<std::string> observed;
        stree::sibling_iterator sib = t.begin(t.begin());
        while (sib != t.end(t.begin())) {
            observed.push_back(*sib);
            if (*sib == "x") t.replace(y, std::string("REPLACED"));
            ++sib;
        }
        cases.push_back({"hazard_replace_next_sibling", "{\"observed\":" + str_array(observed) + "}"});
    }

    // ---- emit ----
    printf("{\n");
    for (size_t i = 0; i < cases.size(); ++i) {
        printf("  %s: %s%s\n", quote(cases[i].first).c_str(), cases[i].second.c_str(),
               (i + 1 < cases.size()) ? "," : "");
    }
    printf("}\n");
    return 0;
}
