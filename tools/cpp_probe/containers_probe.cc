// Captures classic instance_set/scored_instance ordering, the implicit
// default-construction path instance_set's fill constructors rely on, and
// distance_based_scorer's numeric output, for S9 Task 0.
//
// Build/run through tools/capture_containers.py.
#include <moses/moses/moses/types.h>
#include <moses/moses/representation/field_set.h>
#include <moses/moses/representation/instance.h>
#include <moses/moses/representation/instance_scorer.h>
#include <moses/moses/representation/instance_set.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

static string float_bits_hex(float value)
{
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "binary32 expected");
    memcpy(&bits, &value, sizeof(bits));
    ostringstream out;
    out << "0x" << hex << setw(8) << setfill('0') << bits;
    return out.str();
}

static void emit_score(const composite_score& cs)
{
    cout << "{\"score\":\"" << float_bits_hex(cs.get_score())
         << "\",\"complexity\":" << cs.get_complexity()
         << ",\"complexity_penalty\":\"" << float_bits_hex(cs.get_complexity_penalty())
         << "\",\"uniformity_penalty\":\"" << float_bits_hex(cs.get_uniformity_penalty())
         << "\",\"penalized_score\":\"" << float_bits_hex(cs.get_penalized_score()) << "\"}";
}

static void emit_words(const instance& inst)
{
    cout << '[';
    for (size_t idx = 0; idx < inst.size(); ++idx) {
        if (idx) cout << ',';
        cout << "\"0x" << hex << setw(16) << setfill('0') << inst[idx] << dec << "\"";
    }
    cout << ']';
}

// --- (1) instance_set's implicit fill-construction path -------------------
// Nobody in classic calls scored_instance(...) directly (see scored_instance.
// py's own module docstring); instance_set(n, fs) and instance_set(n, inst,
// fs) reach scored_instance's single/none-arg constructors only through
// std::vector's implicit fill-conversion. Measure what that actually
// produces, and that mutating one slot's score cannot be observed through
// another (the aliasing question CARRY-FORWARD.md 1.1/2.4 raises).
static void emit_default_fill()
{
    field_set fs(field_set::disc_spec(2), 6);
    instance_set<composite_score> deme(3, fs);

    cout << "\"default_fill\":{\"entries\":[";
    for (size_t i = 0; i < deme.size(); ++i) {
        if (i) cout << ',';
        cout << "{\"words\":";
        emit_words(deme[i].first);
        cout << ",\"score\":";
        emit_score(deme[i].second);
        cout << '}';
    }
    cout << "],";

    // Mutate slot 0's score in place; 1 and 2 must be untouched if the
    // vector fill-constructed three INDEPENDENT composite_score objects
    // rather than three references to one.
    deme[0].second.set_score(-1.0f);
    cout << "\"after_mutating_slot0\":[";
    for (size_t i = 0; i < deme.size(); ++i) {
        if (i) cout << ',';
        emit_score(deme[i].second);
    }
    cout << "]}";
}

// --- (2) instance_set(n, inst, fs): n independent copies of one prototype -
static void emit_fill_with_instance()
{
    field_set fs(field_set::disc_spec(2), 6);
    instance proto(fs.packed_width());
    fs.set_raw(proto, 0, 1);
    fs.set_raw(proto, 3, 1);

    instance_set<composite_score> deme(3, proto, fs);
    // Mutate slot 0's instance content directly; 1 and 2 must be untouched.
    deme[0].first[0] = 0xFF;

    cout << "\"fill_with_instance\":{\"entries\":[";
    for (size_t i = 0; i < deme.size(); ++i) {
        if (i) cout << ',';
        emit_words(deme[i].first);
    }
    cout << "]}";
}

// --- (3) scored_instance<composite_score> pairwise ordering ---------------
struct synth {
    const char* name;
    float score;
    unsigned complexity;
    float cpen;
    float upen;
};

static void emit_pairwise_compare()
{
    const float nan = numeric_limits<float>::quiet_NaN();
    vector<synth> table = {
        {"a-worse-score", -10.0f, 0u, 0.0f, 0.0f},
        {"b-better-score", -2.0f, 0u, 0.0f, 0.0f},
        {"c-tie-score-high-complexity", -5.0f, 10u, 0.0f, 0.0f},
        {"d-tie-score-low-complexity", -5.0f, 3u, 0.0f, 0.0f},
        {"e-tie-score-tie-complexity", -5.0f, 3u, 0.0f, 0.0f},
        {"f-same-raw-score-but-penalized", -5.0f, 10u, 1.0f, 0.0f},
        {"g-near-tie-within-epsilon", -5.0000005f, 3u, 0.0f, 0.0f},
        {"h-nan-score", nan, 5u, 0.0f, 0.0f},
        {"i-nan-score-again", nan, 1u, 0.0f, 0.0f},
    };

    field_set fs(field_set::disc_spec(2), 4);
    vector<scored_instance<composite_score>> entries;
    for (size_t i = 0; i < table.size(); ++i) {
        instance inst(fs.packed_width());
        fs.set_raw(inst, 0, (disc_t)(i % 4));
        composite_score cs(table[i].score, table[i].complexity, table[i].cpen, table[i].upen);
        entries.push_back(scored_instance<composite_score>(inst, cs));
    }

    cout << "\"pairwise_compare\":{\"names\":[";
    for (size_t i = 0; i < table.size(); ++i) {
        if (i) cout << ',';
        cout << '"' << table[i].name << '"';
    }
    cout << "],\"pairs\":[";
    bool first = true;
    for (size_t i = 0; i < entries.size(); ++i) {
        for (size_t j = 0; j < entries.size(); ++j) {
            if (i == j) continue;
            if (!first) cout << ',';
            first = false;
            bool lt = entries[i] < entries[j];
            bool gt = entries[i] > entries[j];
            bool std_greater = greater<scored_instance<composite_score>>()(entries[i], entries[j]);
            cout << "{\"i\":" << i << ",\"j\":" << j << ",\"lt\":" << (lt ? "true" : "false")
                 << ",\"gt\":" << (gt ? "true" : "false")
                 << ",\"std_greater\":" << (std_greater ? "true" : "false") << '}';
        }
    }
    cout << "]}";
}

// --- (4) distance_based_scorer ---------------------------------------------
static void emit_distance_scorer()
{
    field_set fs(field_set::disc_spec(2), 8);
    instance target(fs.packed_width());
    for (size_t idx = 0; idx < 8; ++idx) fs.set_raw(target, idx, (disc_t)(idx % 2));

    vector<instance> probes;
    instance same = target;
    probes.push_back(same);
    instance one_off = target;
    fs.set_raw(one_off, 0, (disc_t)((fs.get_raw(one_off, 0) + 1) % 2));
    probes.push_back(one_off);
    instance all_off(fs.packed_width());
    for (size_t idx = 0; idx < 8; ++idx)
        fs.set_raw(all_off, idx, (disc_t)((fs.get_raw(target, idx) + 1) % 2));
    probes.push_back(all_off);

    distance_based_scorer scorer(fs, target);
    cout << "\"distance_scorer\":{\"results\":[";
    for (size_t i = 0; i < probes.size(); ++i) {
        if (i) cout << ',';
        composite_score cs = scorer(probes[i]);
        emit_score(cs);
    }
    cout << "]}";
}

int main()
{
    cout << unitbuf;
    cout << "{\n  ";
    emit_default_fill();
    cout << ",\n  ";
    emit_fill_with_instance();
    cout << ",\n  ";
    emit_pairwise_compare();
    cout << ",\n  ";
    emit_distance_scorer();
    cout << "\n}\n";
    return 0;
}
