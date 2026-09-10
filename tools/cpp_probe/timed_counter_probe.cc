// tools/cpp_probe/timed_counter_probe.cc
//
// What does `Counter<TimedValue, count_t>` (table.h:496) -- i.e.
// `std::map<TimedValue, double, std::less<TimedValue>>` -- ACTUALLY do,
// given that `TimedValue::operator<` (table.h:482) is
//
//     return (value < r.value) || (timestamp < r.timestamp);
//
// which is missing the `value == r.value &&` guard that would make it
// lexicographic, and is therefore not a strict weak ordering?
// `tools/cpp_probe/timed_value_ordering_probe.cc` already shows the
// asymmetry violation on one pair; this probe measures the CONSEQUENCE
// for the container, which is what a port has to decide about.
//
// Modes:
//   ordering   -- the comparator's own truth table over a pool of
//                 TimedValues, plus boost::gregorian::date ordering
//                 facts (including the default not-a-date value, which
//                 table.cc:513 constructs for every untimestamped row).
//   sequences  -- for every insertion sequence of length 1..MAXLEN over
//                 a pool of TimedValues, build a real TimedCounter with
//                 `counter[tv] += 1.0` (exactly what Table::compressed
//                 does at table.cc:548) and report BOTH the container's
//                 iteration order/contents AND the "intended"
//                 lexicographic-map result.
//
// Build (see table_probe.cc's header comment for why the installed
// header tree alone is not enough):
//
//   P=$HOME/opt/moses-legacy
//   C=<classic checkout>
//   g++ -O2 -std=c++17 -I$C -I$P/include \
//       -o tools/cpp_probe/build/timed_counter_probe \
//       tools/cpp_probe/timed_counter_probe.cc \
//       -L$P/lib -L$P/lib/opencog -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lcomboreduct -lcogutil
//
// Usage: timed_counter_probe <ordering|sequences> [maxlen]

#include <moses/comboreduct/table/table.h>

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::combo;
using namespace std;

// ---- the pool -------------------------------------------------------
//
// Two boolean output values (id::logical_true == 1 and
// id::logical_false == 2 as enumerators -- vertex.h:74 puts the
// constants first -- so logical_true < logical_false as vertices) and
// three ordered dates.  A CTable cell holding a `true` at a LATER date
// and a `false` at an EARLIER date is exactly the shape that makes both
// `a<b` and `b<a` true.

struct PoolEntry { const char *name; vertex v; TTable::value_type t; };

static vector<PoolEntry> make_pool()
{
    using boost::gregorian::date;
    vector<PoolEntry> pool;
    static const char *names[] = {
        "T@d1", "T@d2", "T@d3", "F@d1", "F@d2", "F@d3",
    };
    vertex vs[2] = { vertex(id::logical_true), vertex(id::logical_false) };
    date ds[3] = { date(2012, 1, 3), date(2012, 1, 11), date(2012, 1, 19) };
    int k = 0;
    for (int vi = 0; vi < 2; ++vi)
        for (int di = 0; di < 3; ++di, ++k)
            pool.push_back({names[k], vs[vi], ds[di]});
    return pool;
}

// ---- JSON helpers ---------------------------------------------------

static string quote(const string &s) { return "\"" + s + "\""; }

static string date_str(const TTable::value_type &d)
{
    if (d.is_not_a_date()) return "not-a-date-time";
    return TTable::to_string(d);
}

static string vertex_str(const vertex &v)
{
    if (const builtin *b = boost::get<builtin>(&v)) {
        if (*b == id::logical_true) return "true";
        if (*b == id::logical_false) return "false";
    }
    ostringstream os; os << v; return os.str();
}

static string entry_json(const vertex &v, const TTable::value_type &t, double c)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", c);
    return "[" + quote(vertex_str(v)) + "," + quote(date_str(t)) + "," + buf + "]";
}

// ---- mode: ordering -------------------------------------------------

static int mode_ordering()
{
    using boost::gregorian::date;
    vector<PoolEntry> pool = make_pool();

    printf("{\n");

    // boost::gregorian::date facts, including the default (not-a-date)
    // value that table.cc:513 gives every untimestamped observation.
    date nad;                       // default-constructed == not_a_date_time
    date d1(2012, 1, 3), d3(2012, 1, 19);
    printf("  \"date_facts\": {\n");
    printf("    \"default_is_not_a_date\": %s,\n", nad.is_not_a_date() ? "true" : "false");
    printf("    \"default_lt_real\": %s,\n", (nad < d1) ? "true" : "false");
    printf("    \"real_lt_default\": %s,\n", (d1 < nad) ? "true" : "false");
    printf("    \"default_eq_default\": %s,\n", (nad == date()) ? "true" : "false");
    printf("    \"d1_lt_d3\": %s\n", (d1 < d3) ? "true" : "false");
    printf("  },\n");

    // TimedValue's own comparator, every ordered pair from the pool.
    printf("  \"pool\": [");
    for (size_t i = 0; i < pool.size(); ++i) {
        if (i) printf(",");
        printf("{\"name\":%s,\"value\":%s,\"timestamp\":%s}",
               quote(pool[i].name).c_str(),
               quote(vertex_str(pool[i].v)).c_str(),
               quote(date_str(pool[i].t)).c_str());
    }
    printf("],\n");

    printf("  \"pairs\": [");
    bool first = true;
    for (size_t i = 0; i < pool.size(); ++i) {
        for (size_t j = 0; j < pool.size(); ++j) {
            TimedValue a(pool[i].v, pool[i].t), b(pool[j].v, pool[j].t);
            bool ab = (a < b), ba = (b < a), eq = (a == b);
            if (!first) printf(",");
            first = false;
            printf("{\"a\":%s,\"b\":%s,\"a_lt_b\":%s,\"b_lt_a\":%s,\"a_eq_b\":%s}",
                   quote(pool[i].name).c_str(), quote(pool[j].name).c_str(),
                   ab ? "true" : "false", ba ? "true" : "false", eq ? "true" : "false");
        }
    }
    printf("],\n");

    // The default-timestamp TimedValue, which is what an untimestamped
    // CTable cell is built from (table.cc:513/:538).
    TimedValue u1(vertex(id::logical_true)), u2(vertex(id::logical_false));
    printf("  \"untimestamped\": {\"true_lt_false\":%s,\"false_lt_true\":%s,\"eq\":%s}\n",
           (u1 < u2) ? "true" : "false", (u2 < u1) ? "true" : "false",
           (u1 == u2) ? "true" : "false");
    printf("}\n");
    return 0;
}

// ---- mode: sequences ------------------------------------------------

// The "intended" container: what a lexicographic `(value, timestamp)`
// comparator -- the one the missing `value == r.value &&` guard would
// have produced -- gives for the same insertion sequence.
struct LexLess {
    bool operator()(const TimedValue &a, const TimedValue &b) const
    {
        if (a.value < b.value) return true;
        if (b.value < a.value) return false;
        return a.timestamp < b.timestamp;
    }
};

static int mode_sequences(int maxlen)
{
    vector<PoolEntry> pool = make_pool();
    const size_t P = pool.size();

    printf("{\n  \"maxlen\": %d,\n  \"pool_size\": %zu,\n  \"cases\": [\n", maxlen, P);

    bool first_case = true;
    size_t n_total = 0, n_contents_differ = 0, n_order_differs = 0;

    for (int len = 1; len <= maxlen; ++len) {
        vector<size_t> idx(len, 0);
        for (;;) {
            // Build the real container, exactly as Table::compressed does.
            TimedCounter actual;
            map<TimedValue, double, LexLess> intended;
            for (int i = 0; i < len; ++i) {
                TimedValue tv(pool[idx[i]].v, pool[idx[i]].t);
                actual[tv] += 1.0;
                intended[tv] += 1.0;
            }

            // Serialize both in their own iteration order.
            string a_json = "[", i_json = "[";
            bool f = true;
            size_t n_iterated = 0;
            double actual_total = 0.0;
            for (const auto &kv : actual) {
                if (!f) a_json += ",";
                f = false;
                ++n_iterated;
                actual_total += kv.second;
                a_json += entry_json(kv.first.value, kv.first.timestamp, kv.second);
            }
            a_json += "]";
            f = true;
            for (const auto &kv : intended) {
                if (!f) i_json += ",";
                f = false;
                i_json += entry_json(kv.first.value, kv.first.timestamp, kv.second);
            }
            i_json += "]";

            // Contents comparison, order-insensitive: same (value,
            // timestamp) -> count mapping?
            map<TimedValue, double, LexLess> actual_as_lex;
            bool dup = false;
            for (const auto &kv : actual) {
                auto it = actual_as_lex.find(kv.first);
                if (it != actual_as_lex.end()) dup = true;
                actual_as_lex[kv.first] += kv.second;
            }
            bool contents_equal = (!dup) && (actual_as_lex.size() == intended.size());
            if (contents_equal) {
                auto ai = actual_as_lex.begin();
                auto ii = intended.begin();
                for (; ai != actual_as_lex.end(); ++ai, ++ii) {
                    if (!(ai->first == ii->first) || ai->second != ii->second) {
                        contents_equal = false;
                        break;
                    }
                }
            }
            bool order_equal = (a_json == i_json);

            ++n_total;
            if (!contents_equal) ++n_contents_differ;
            if (!order_equal) ++n_order_differs;

            string seq = "[";
            for (int i = 0; i < len; ++i) {
                if (i) seq += ",";
                seq += quote(pool[idx[i]].name);
            }
            seq += "]";

            if (!first_case) printf(",\n");
            first_case = false;
            char nbuf[128];
            snprintf(nbuf, sizeof(nbuf),
                     ",\"actual_map_size\":%zu,\"actual_iterated\":%zu,"
                     "\"actual_total_count\":%.17g,\"intended_total_count\":%.17g",
                     actual.size(), n_iterated, actual_total, (double)len);
            printf("    {\"seq\":%s,\"actual\":%s,\"intended\":%s,"
                   "\"contents_equal\":%s,\"order_equal\":%s,\"physical_duplicate\":%s%s}",
                   seq.c_str(), a_json.c_str(), i_json.c_str(),
                   contents_equal ? "true" : "false",
                   order_equal ? "true" : "false",
                   dup ? "true" : "false", nbuf);

            // odometer
            int p = len - 1;
            for (; p >= 0; --p) {
                if (++idx[p] < P) break;
                idx[p] = 0;
            }
            if (p < 0) break;
        }
    }

    printf("\n  ],\n");
    printf("  \"summary\": {\"n_sequences\": %zu, \"n_contents_differ\": %zu, "
           "\"n_order_differs\": %zu}\n}\n",
           n_total, n_contents_differ, n_order_differs);
    return 0;
}

int main(int argc, char **argv)
{
    string mode = argc > 1 ? argv[1] : "ordering";
    int maxlen = argc > 2 ? atoi(argv[2]) : 3;
    if (mode == "ordering") return mode_ordering();
    if (mode == "sequences") return mode_sequences(maxlen);
    fprintf(stderr, "usage: %s <ordering|sequences> [maxlen]\n", argv[0]);
    return 1;
}
