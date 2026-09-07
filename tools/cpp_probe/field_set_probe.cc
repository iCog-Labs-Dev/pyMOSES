// Captures classic packed-field layouts and continuous codec behavior for S8.
//
// Build/run through tools/capture_field_set.py.
#include <moses/moses/representation/field_set.h>

#include <boost/variant/get.hpp>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

using raw_op = pair<size_t, disc_t>;

static string double_bits_hex(double value)
{
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value), "binary64 expected");
    memcpy(&bits, &value, sizeof(bits));
    ostringstream out;
    out << "0x" << hex << setw(16) << setfill('0') << bits;
    return out.str();
}

static void emit_spec(const field_set::spec& item)
{
    if (const auto* ds = boost::get<field_set::disc_spec>(&item)) {
        cout << "{\"kind\":\"disc\",\"multy\":" << ds->multy << '}';
        return;
    }
    const auto& cs = boost::get<field_set::contin_spec>(item);
    cout << "{\"kind\":\"contin\",\"mean\":" << setprecision(17) << cs.mean
         << ",\"step_size\":" << cs.step_size << ",\"expansion\":" << cs.expansion
         << ",\"depth\":" << cs.depth << '}';
}

static void emit_raw(const field_set& fs, const instance& inst)
{
    cout << '[';
    for (size_t idx = 0; idx < fs.raw_size(); ++idx) {
        if (idx) cout << ',';
        cout << fs.get_raw(inst, idx);
    }
    cout << ']';
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

static void emit_layout(const string& name,
                        const vector<field_set::spec>& specs,
                        const vector<vector<raw_op>>& writes,
                        bool last)
{
    field_set fs(specs.begin(), specs.end());
    cout << "    {\"name\":\"" << name << "\",\"input_specs\":[";
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        if (idx) cout << ',';
        emit_spec(specs[idx]);
    }
    cout << "],\"canonical_contin\":[";
    for (size_t ci = 0; ci < fs.contin().size(); ++ci) {
        if (ci) cout << ',';
        emit_spec(field_set::spec(fs.contin()[ci]));
    }
    cout << "],\"canonical_disc\":[";
    for (size_t di = 0; di < fs.disc_and_bit().size(); ++di) {
        if (di) cout << ',';
        emit_spec(field_set::spec(fs.disc_and_bit()[di]));
    }
    cout << "],\"contin_raw_offsets\":[";
    for (size_t ci = 0; ci < fs.contin().size(); ++ci) {
        if (ci) cout << ',';
        cout << fs.contin_to_raw_idx(ci);
    }
    cout << "],\"fields\":[";
    size_t idx = 0;
    for (auto it = fs.begin_fields(); it != fs.end_fields(); ++it, ++idx) {
        if (idx) cout << ',';
        cout << "{\"width\":" << it->width << ",\"major_offset\":" << it->major_offset
             << ",\"minor_offset\":" << it->minor_offset << '}';
    }
    cout << "],\"packed_width\":" << fs.packed_width()
         << ",\"raw_size\":" << fs.raw_size() << ",\"dim_size\":" << fs.dim_size()
         << ",\"starts\":{\"contin\":[" << fs.begin_contin_raw_idx() << ','
         << fs.end_contin_raw_idx() << "],\"disc\":[" << fs.begin_disc_raw_idx() << ','
         << fs.end_disc_raw_idx() << "],\"bit\":[" << fs.begin_bit_raw_idx() << ','
         << fs.end_bit_raw_idx() << "]},\"counts\":{\"contin_raw\":"
         << fs.n_contin_fields() << ",\"disc\":" << fs.n_disc_fields()
         << ",\"bits\":" << fs.n_bits() << "},\"writes\":[";
    for (size_t wi = 0; wi < writes.size(); ++wi) {
        if (wi) cout << ',';
        instance inst(fs.packed_width());
        cout << "{\"raw_ops\":[";
        for (size_t oi = 0; oi < writes[wi].size(); ++oi) {
            if (oi) cout << ',';
            const auto [raw_idx, value] = writes[wi][oi];
            cout << '[' << raw_idx << ',' << value << ']';
            fs.set_raw(inst, raw_idx, value);
        }
        cout << "],\"words_hex\":";
        emit_words(inst);
        cout << ",\"raw\":";
        emit_raw(fs, inst);
        cout << '}';
    }
    cout << "]}" << (last ? "\n" : ",\n");
}

static void emit_continuous(const string& name,
                            const field_set::contin_spec& cs,
                            const vector<disc_t>& raw,
                            bool last)
{
    field_set fs(cs, 1);
    instance inst(fs.packed_width());
    for (size_t idx = 0; idx < raw.size(); ++idx) fs.set_raw(inst, idx, raw[idx]);
    cout << "    {\"name\":\"" << name << "\",\"spec\":";
    emit_spec(field_set::spec(cs));
    cout << ",\"raw\":[";
    for (size_t idx = 0; idx < raw.size(); ++idx) {
        if (idx) cout << ',';
        cout << raw[idx];
    }
    cout << "],\"value_bits_hex\":\"" << double_bits_hex(fs.get_contin(inst, 0))
         << "\",\"length\":" << fs.contin_length(inst, 0) << '}'
         << (last ? "\n" : ",\n");
}

static void emit_set_continuous(const string& name,
                                const field_set::contin_spec& cs,
                                double target,
                                bool last)
{
    field_set fs(cs, 1);
    instance inst(fs.packed_width());
    fs.set_contin(inst, 0, target);
    cout << "    {\"name\":\"" << name << "\",\"spec\":";
    emit_spec(field_set::spec(cs));
    cout << ",\"target\":" << setprecision(17) << target << ",\"raw_after\":";
    emit_raw(fs, inst);
    cout << ",\"decoded_bits_hex\":\"" << double_bits_hex(fs.get_contin(inst, 0)) << "\"}"
         << (last ? "\n" : ",\n");
}

static void emit_pack_case(const string& name,
                           const vector<field_set::spec>& specs,
                           const vector<disc_t>& raw,
                           bool last)
{
    field_set fs(specs.begin(), specs.end());
    instance packed(fs.packed_width());
    fs.pack(raw.begin(), packed.begin());
    cout << "    {\"name\":\"" << name << "\",\"input_specs\":[";
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        if (idx) cout << ',';
        emit_spec(specs[idx]);
    }
    cout << "],\"raw\":[";
    for (size_t idx = 0; idx < raw.size(); ++idx) {
        if (idx) cout << ',';
        cout << raw[idx];
    }
    cout << "],\"words_hex\":";
    emit_words(packed);
    cout << "}" << (last ? "\n" : ",\n");
}

static void emit_hamming_case()
{
    field_set fs(field_set::disc_spec(3), 3);
    instance lhs(fs.packed_width()), rhs(fs.packed_width());
    fs.set_raw(lhs, 0, 1);
    fs.set_raw(lhs, 2, 2);
    fs.set_raw(rhs, 1, 1);
    fs.set_raw(rhs, 2, 2);
    cout << "    {\"name\":\"raw-field-distance\",\"lhs_raw\":";
    emit_raw(fs, lhs);
    cout << ",\"rhs_raw\":";
    emit_raw(fs, rhs);
    cout << ",\"distance\":" << fs.hamming_distance(lhs, rhs) << "}\n";
}

static void emit_merge_case()
{
    field_set fs(field_set::disc_spec(3), 3);
    const vector<disc_t> base_raw = {0, 1, 2};
    const vector<disc_t> ref_raw = {0, 2, 2};
    const vector<disc_t> target_raw = {2, 0, 1};
    instance base(fs.packed_width()), reference(fs.packed_width()), target(fs.packed_width());
    fs.pack(base_raw.begin(), base.begin());
    fs.pack(ref_raw.begin(), reference.begin());
    fs.pack(target_raw.begin(), target.begin());
    cout << "    {\"name\":\"copy-reference-differences\",\"base_raw\":";
    emit_raw(fs, base);
    cout << ",\"reference_raw\":";
    emit_raw(fs, reference);
    cout << ",\"target_before_raw\":";
    emit_raw(fs, target);
    fs.merge_instance(target, base, reference);
    cout << ",\"target_after_raw\":";
    emit_raw(fs, target);
    cout << "}\n";
}

int main()
{
    cout << "{\n  \"abi\":{\"sizeof_packed_t\":" << sizeof(packed_t)
         << ",\"bits_per_packed_t\":" << bits_per_packed_t << "},\n";
    cout << "  \"nbits_to_pack\":[";
    const vector<size_t> multiplicities = {1, 2, 3, 4, 5, 16, 17, 50, 257};
    for (size_t idx = 0; idx < multiplicities.size(); ++idx) {
        if (idx) cout << ',';
        cout << "{\"multy\":" << multiplicities[idx]
             << ",\"width\":" << nbits_to_pack(multiplicities[idx]) << '}';
    }
    cout << "],\n  \"layouts\":[\n";

    emit_layout("empty", {}, {{}}, false);
    emit_layout("65-bits", vector<field_set::spec>(65, field_set::disc_spec(2)),
                {{{63, 1}, {64, 1}}}, false);
    emit_layout("33-ternary", vector<field_set::spec>(33, field_set::disc_spec(3)),
                {{{31, 2}, {32, 1}}}, false);
    vector<field_set::spec> straddle = {
        field_set::contin_spec(0.0, 1.0, 2.0, 5),
    };
    straddle.insert(straddle.end(), 15, field_set::disc_spec(5));
    emit_layout("straddle-following", straddle, {{{18, 15}, {19, 9}}}, false);
    vector<field_set::spec> contin_ties = {
        field_set::contin_spec(0.0, 1.0, 2.0, 4),
        field_set::contin_spec(0.0, 1.0, 3.0, 4),
        field_set::contin_spec(0.0, 2.0, 3.0, 4),
        field_set::contin_spec(1.0, 2.0, 3.0, 4),
    };
    emit_layout("same-depth-continuous-order", contin_ties, {{}}, false);
    vector<field_set::spec> mixed = {
        field_set::disc_spec(2),
        field_set::contin_spec(0.0, 1.0, 2.0, 3),
        field_set::disc_spec(9),
        field_set::contin_spec(1.0, 0.5, 3.0, 5),
        field_set::disc_spec(3),
        field_set::disc_spec(1),
        field_set::disc_spec(9),
    };
    emit_layout("canonical-mixed", mixed, {{{0, 1}, {7, 2}, {8, 9}, {12, 1}}}, true);

    const field_set::contin_spec base(0.0, 1.0, 2.0, 5);
    cout << "  ],\n  \"continuous\":[\n";
    emit_continuous("all-stop", base, {0, 0, 0, 0, 0}, false);
    emit_continuous("left-stop", base, {1, 0, 0, 0, 0}, false);
    emit_continuous("left-left-right", base, {1, 1, 2, 0, 0}, false);
    emit_continuous("right-right-left", base, {2, 2, 1, 0, 0}, false);
    emit_continuous("stop-ignores-tail", base, {1, 2, 0, 1, 2}, false);
    emit_continuous("right-left-stop", base, {2, 1, 0, 0, 0}, false);
    emit_continuous("contracting-expansion",
                    field_set::contin_spec(0.0, 0.5, 0.5, 4),
                    {2, 0, 0, 0}, true);

    cout << "  ],\n  \"set_continuous\":[\n";
    emit_set_continuous("mean", base, 0.0, false);
    emit_set_continuous("one-left", base, -1.0, false);
    emit_set_continuous("two-left", base, -2.0, false);
    emit_set_continuous("far-right", base, 100.0, false);
    emit_set_continuous("tie-keeps-shallow",
                        field_set::contin_spec(0.0, 1.0, 1.0, 2), -1.5, true);

    vector<disc_t> rollover_raw(33, 0);
    rollover_raw[31] = 2;
    rollover_raw[32] = 1;
    vector<disc_t> straddle_raw(20, 0);
    straddle_raw[18] = 15;
    straddle_raw[19] = 9;
    cout << "  ],\n  \"pack_cases\":[\n";
    emit_pack_case("33-ternary-rollover",
                   vector<field_set::spec>(33, field_set::disc_spec(3)),
                   rollover_raw, false);
    emit_pack_case("straddle-following", straddle, straddle_raw, true);
    cout << "  ],\n  \"hamming_cases\":[\n";
    emit_hamming_case();
    cout << "  ],\n  \"merge_cases\":[\n";
    emit_merge_case();
    cout << "  ]\n}\n";
    return 0;
}
