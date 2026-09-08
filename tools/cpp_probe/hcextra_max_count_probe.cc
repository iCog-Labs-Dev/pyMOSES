// tools/cpp_probe/hcextra_max_count_probe.cc
//
// Extra oracle for the S9 fix-round coverage pass (mutation survivor (i)):
// no existing test in this port ever calls
// `count_neighborhood_size_from_index` with a non-default `max_count`, so
// its early-stop guard (neighborhood_sampling.cc, "stop counting once
// 'max_count' is exceeded" -- neighborhood_sampling.h:474-475's own doc
// comment) was untested. This probe calls the REAL classic function
// directly, at a genuine multi-valued-disc field_set (multiplicity 3, 5
// knobs -- NOT multiplicity 2, which field_set folds into the fast bit
// path and never reaches the recursive disc branch the guard lives in;
// see field_set.py's own `_build_disc_spec`/`_nbool` classification for
// why the Python port has the same split), at one distance (3) and a
// spread of max_count values straddling the two internal recursion
// thresholds, and dumps both the capped and the uncapped (default
// max_count) results as JSON.
//
// Build (mirrors tools/cpp_probe/neighborhood_probe.cc's own recipe):
//
//   P=$HOME/opt/moses-legacy
//   C=/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses
//   g++ -O2 -std=c++17 -I $C -I $P/include \
//       -o tools/cpp_probe/build/hcextra_max_count_probe \
//       tools/cpp_probe/hcextra_max_count_probe.cc \
//       -L $P/lib -L $P/lib/opencog \
//       -Wl,-rpath,$P/lib -Wl,-rpath,$P/lib/opencog \
//       -lmoses -lcomboreduct -lcogutil
//
// Bound every run: ( ulimit -v 2097152; timeout 300 <cmd> ) -- see
// CARRY-FORWARD.md and this session's own standing OOM caution. The field
// here is tiny (5 knobs, distance <=3) so this finishes instantly.

#include <moses/moses/moses/neighborhood_sampling.h>
#include <moses/moses/representation/field_set.h>

#include <cstdio>
#include <vector>

using namespace opencog;
using namespace opencog::moses;
using namespace std;

int main()
{
    field_set fs(field_set::disc_spec(3), 5);
    instance inst(fs.packed_width());

    printf("{\"begin_disc_raw_idx\":%zu,\"end_disc_raw_idx\":%zu,",
           fs.begin_disc_raw_idx(), fs.end_disc_raw_idx());
    printf("\"dist\":3,\"uncapped\":%zu,\"capped\":[",
           count_neighborhood_size_from_index(fs, inst, 3, 0));

    vector<size_t> caps = {1, 5, 10, 20, 50, 100};
    for (size_t i = 0; i < caps.size(); ++i) {
        if (i) printf(",");
        size_t v = count_neighborhood_size_from_index(fs, inst, 3, 0, caps[i]);
        printf("{\"max_count\":%zu,\"result\":%zu}", caps[i], v);
    }
    printf("]}\n");
    return 0;
}
