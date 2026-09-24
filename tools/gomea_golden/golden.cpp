/*
 * Golden harness for pymoses/gomea/{stats,fos}.py.
 *
 * Source:  GP-GOMEA, https://github.com/marcovirgolin/GP-GOMEA
 *          src/GOMEA/GOMEAFOS.cpp (and src/Include/GPGOMEA/GOMEA/GOMEAFOS.h)
 * Commit:  750220c23d599e515ca337dcca617ae5faab8b4d
 * Licence: Apache-2.0 (Copyright Marco Virgolin). The functions below are
 *          copied from that file; this harness only exists to produce
 *          parity fixtures and is never linked into pyMoses.
 *
 * Copied verbatim (body text character-for-character, apart from the
 * substitutions listed here):
 *   GOMEAFOS.cpp:289-301  GOMEAFOS::DetermineNearestNeighbour     (no change)
 *   GOMEAFOS.cpp:41-181   GOMEAFOS::BuildLinkageTreeFromSimilarityMatrix
 *   GOMEAFOS.cpp:303-490  GOMEAFOS::GenerateLinkageTreeFOS (the entropy
 *                         loops 382-426 live in GOMEAFOS::EntropyLoops so the
 *                         raw entropy matrix can be printed too)
 *
 * Substitutions (every one of them):
 *   S1. `arma::mat` (MI_distribution_adjustments, frequencies) is replaced by
 *       `std::vector<std::vector<double>>` (typedef `mat_t`). Consequently
 *       `frequencies.at(a, b)` / `frequencies(k, l)` become
 *       `frequencies[a][b]` / `frequencies[k][l]`,
 *       `MI_distribution_adjustments(i, j)` becomes
 *       `MI_distribution_adjustments[i][j]`, `mat frequencies(n, n,
 *       fill::zeros)` becomes `mat_t frequencies(n, std::vector<double>(n,
 *       0.0))`, and `arma::accu(...)` becomes the local `accu(...)` (plain
 *       sum of all entries). `uword` is typedef'd to `std::size_t`.
 *   S2. BuildLinkageTreeFromSimilarityMatrix: `random_shuffle(random_order
 *       .begin(), random_order.end());` (line 50) is replaced by copying an
 *       injected permutation (`random_order = injected_order;`), passed as
 *       an extra parameter.
 *   S3. BuildLinkageTreeFromSimilarityMatrix line 84: `(int) randu() *
 *       mpm_length` becomes `(int) 0.0 * mpm_length`. The C-style cast binds
 *       tighter than `*`, and armadillo's randu() is in [0, 1), so upstream's
 *       `(int) randu()` is always 0 and the chain always starts at index 0;
 *       using the literal 0.0 reproduces that exactly without armadillo.
 *   S4. GenerateLinkageTreeFOS takes the population as an integer-coded
 *       matrix `pop_nodes` (rows = individuals, columns = positions) plus
 *       `encode_number` (1 + the largest code), instead of `std::vector<Node*>`.
 *       The symbol-encoding / constant-binning preprocessing (lines 312-380)
 *       is therefore dropped: the codes are already small integers. Codes are
 *       per position (code k in one position is unrelated to code k in
 *       another); the entropies are unchanged by this because the joint table
 *       is indexed by (value at i, value at j) either way.
 *   S5. The dead `if (false)` at line 430 becomes `if (use_mi_branch)`, a new
 *       parameter, so the ordinary-MI branch can be exercised (it is the
 *       LT-MI control arm of pyMoses). `use_mi_branch == false` is the live
 *       upstream path.
 *   S6. GenerateLinkageTreeFOS additionally copies the similarity matrix
 *       (`out_sim`, just before the tree is built) and the full FOS
 *       (`out_full_fos`, just before `FOS.pop_back()`), so both can be
 *       printed. The `fos_no_root_swap` and `no_univar` tails are kept.
 *   S7. main() is new: it reads cases from stdin and prints JSON.
 *
 * Input format (whitespace separated integers):
 *   T                                   number of cases
 *   then per case:
 *     d N0 N1                           positions, H0-population size,
 *                                       current-population size
 *     N0 rows of d codes                H0 population (first call)
 *     N1 rows of d codes                current population
 *     d integers                        injected order (a permutation of 0..d-1)
 *
 * Output: one JSON object {"cases": [...]}; per case:
 *   H      raw entropy matrix of the current population (lines 387-426)
 *   H0     raw entropy matrix of the H0 population
 *   S_mi   similarity after the (dead) ordinary-MI branch, current population
 *   S_b    biased similarity: adjustments set on a first call with the H0
 *          population (accu == 0), then applied to the current population
 *   lt_mi_full / lt_mi : FOS before / after pop_back for S_mi
 *   lt_b_full  / lt_b  : FOS before / after pop_back for S_b
 */

#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

typedef std::size_t uword;                          // S1
typedef std::vector<std::vector<double>> mat_t;     // S1

static double accu(const mat_t & m) {               // S1
    double s = 0.0;
    for (const auto & row : m)
        for (double v : row)
            s += v;
    return s;
}

struct GOMEAFOS {
    static std::vector<std::vector<size_t>> BuildLinkageTreeFromSimilarityMatrix(size_t number_of_nodes, vector<vector<double_t>> &sim_matrix, const vector<size_t> & injected_order);
    static int DetermineNearestNeighbour(int index, vector<vector<double_t>> &S_matrix, vector<int> & mpm_number_of_indices, int mpm_length);
    static vector<vector<double_t>> EntropyLoops(size_t number_of_nodes, const vector<vector<uword>> & pop_nodes, uword encode_number);
    static std::vector<std::vector<size_t> > GenerateLinkageTreeFOS(size_t number_of_nodes, const vector<vector<uword>> & pop_nodes, uword encode_number, mat_t & MI_distribution_adjustments, bool fos_no_root_swap, bool use_mi_branch, const vector<size_t> & injected_order, vector<vector<double_t>> & out_sim, vector<vector<size_t>> & out_full_fos);
};

// ---- verbatim: GOMEAFOS.cpp:41-181 (S2, S3) --------------------------------
std::vector<std::vector<size_t>> GOMEAFOS::BuildLinkageTreeFromSimilarityMatrix(size_t number_of_nodes, vector<vector<double_t>> &sim_matrix, const vector<size_t> & injected_order) {
    // code taken from Peter's

    vector<vector<size_t> > FOS;

    vector<size_t> random_order;
    random_order.reserve(number_of_nodes);
    for (size_t i = 0; i < number_of_nodes; i++)
        random_order.push_back(i);
    random_order = injected_order; // S2: was random_shuffle(random_order.begin(), random_order.end());

    vector<vector < int >> mpm(number_of_nodes, vector<int>(1));
    vector<int> mpm_number_of_indices(number_of_nodes);
    int mpm_length = number_of_nodes;

    for (size_t i = 0; i < number_of_nodes; i++) {
        mpm[i][0] = random_order[i];
        mpm_number_of_indices[i] = 1;
    }

    /* Initialize LT to the initial MPM */
    FOS.resize(number_of_nodes + number_of_nodes - 1);
    size_t FOSs_index = 0;
    for (size_t i = 0; i < mpm_length; i++) {
        FOS[FOSs_index] = vector<size_t>(mpm[i].begin(), mpm[i].end());
        FOSs_index++;
    }

    /* Initialize similarity matrix */
    vector<vector < double_t >> S_matrix(number_of_nodes, vector<double_t>(number_of_nodes));
    for (size_t i = 0; i < mpm_length; i++)
        for (size_t j = 0; j < mpm_length; j++)
            S_matrix[i][j] = sim_matrix[mpm[i][0]][mpm[j][0]];
    for (size_t i = 0; i < mpm_length; i++)
        S_matrix[i][i] = 0;

    vector<vector < int >> mpm_new;
    vector<int> NN_chain(number_of_nodes + 2, 0);
    int NN_chain_length = 0;
    short done = 0;

    while (!done) {
        if (NN_chain_length == 0) {
            NN_chain[NN_chain_length] = (int) 0.0 * mpm_length; // S3: was (int) randu() * mpm_length, always 0
            NN_chain_length++;
        }

        while (NN_chain_length < 3) {
            NN_chain[NN_chain_length] = DetermineNearestNeighbour(NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length);
            NN_chain_length++;
        }

        while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]) {
            NN_chain[NN_chain_length] = DetermineNearestNeighbour(NN_chain[NN_chain_length - 1], S_matrix, mpm_number_of_indices, mpm_length);
            if (((S_matrix[NN_chain[NN_chain_length - 1]][NN_chain[NN_chain_length]] == S_matrix[NN_chain[NN_chain_length - 1]][NN_chain[NN_chain_length - 2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];
            NN_chain_length++;
            if (NN_chain_length > number_of_nodes)
                break;
        }
        int r0 = NN_chain[NN_chain_length - 2];
        int r1 = NN_chain[NN_chain_length - 1];
        int rswap;
        if (r0 > r1) {
            rswap = r0;
            r0 = r1;
            r1 = rswap;
        }
        NN_chain_length -= 3;

        if (r1 < mpm_length) { /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
            vector<size_t> indices(mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            //indices.resize((mpm_number_of_indices[r0] + mpm_number_of_indices[r1]));
            //indices.clear();

            int i = 0;
            for (int j = 0; j < mpm_number_of_indices[r0]; j++) {
                indices[i] = mpm[r0][j];
                i++;
            }
            for (int j = 0; j < mpm_number_of_indices[r1]; j++) {
                indices[i] = mpm[r1][j];
                i++;
            }

            FOS[FOSs_index] = indices;
            FOSs_index++;

            double_t mul0 = ((double_t) mpm_number_of_indices[r0]) / ((double_t) mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            double_t mul1 = ((double_t) mpm_number_of_indices[r1]) / ((double_t) mpm_number_of_indices[r0] + mpm_number_of_indices[r1]);
            for (i = 0; i < mpm_length; i++) {
                if ((i != r0) && (i != r1)) {
                    S_matrix[i][r0] = mul0 * S_matrix[i][r0] + mul1 * S_matrix[i][r1];
                    S_matrix[r0][i] = S_matrix[i][r0];
                }
            }

            mpm_new = vector<vector < int >> (mpm_length - 1);
            vector<int> mpm_new_number_of_indices(mpm_length - 1);
            int mpm_new_length = mpm_length - 1;
            for (i = 0; i < mpm_new_length; i++) {
                mpm_new[i] = mpm[i];
                mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
            }

            mpm_new[r0] = vector<int>(indices.begin(), indices.end());

            mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0] + mpm_number_of_indices[r1];
            if (r1 < mpm_length - 1) {
                mpm_new[r1] = mpm[mpm_length - 1];
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length - 1];

                for (i = 0; i < r1; i++) {
                    S_matrix[i][r1] = S_matrix[i][mpm_length - 1];
                    S_matrix[r1][i] = S_matrix[i][r1];
                }

                for (int j = r1 + 1; j < mpm_new_length; j++) {
                    S_matrix[r1][j] = S_matrix[j][mpm_length - 1];
                    S_matrix[j][r1] = S_matrix[r1][j];
                }
            }

            for (i = 0; i < NN_chain_length; i++) {
                if (NN_chain[i] == mpm_length - 1) {
                    NN_chain[i] = r1;
                    break;
                }
            }

            mpm = mpm_new;
            mpm_number_of_indices = mpm_new_number_of_indices;
            mpm_length = mpm_new_length;

            if (mpm_length == 1)
                done = 1;
        }
    }

    return FOS;
}

// ---- verbatim: GOMEAFOS.cpp:289-301 ----------------------------------------
int GOMEAFOS::DetermineNearestNeighbour(int index, vector<vector<double_t>> &S_matrix, vector<int> & mpm_number_of_indices, int mpm_length) {
    int i, result;

    result = 0;
    if (result == index)
        result++;
    for (i = 1; i < mpm_length; i++) {
        if (((S_matrix[index][i] > S_matrix[index][result]) || ((S_matrix[index][i] == S_matrix[index][result]) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index))
            result = i;
    }

    return ( result);
}

// ---- verbatim: GOMEAFOS.cpp:310 and 382-426 (S1, S4) -----------------------
vector<vector<double_t>> GOMEAFOS::EntropyLoops(size_t number_of_nodes, const vector<vector<uword>> & pop_nodes, uword encode_number) {
    size_t pop_size = pop_nodes.size();

    // phase 1: estimate similarity matrix
    vector<vector < double_t >> mi_matrix(number_of_nodes, vector<double_t>(number_of_nodes, 0.0));

    // build frequency table for symbol pairs
    mat_t frequencies(encode_number, std::vector<double>(encode_number, 0.0)); // S1: was mat frequencies(values_to_int_map.size(), values_to_int_map.size(), fill::zeros);
    uword val_i, val_j;

    // measure frequencies of pairs & compute joint entropy
    for (uword i = 0; i < number_of_nodes; i++) {
        for (uword j = i + 1; j < number_of_nodes; j++) {
            for (uword p = 0; p < pop_size; p++) {
                val_i = pop_nodes[p][i];
                val_j = pop_nodes[p][j];

                frequencies[val_i][val_j] += 1.0;
            }

            double_t freq;
            for (uword k = 0; k < encode_number; k++) {
                for (uword l = 0; l < encode_number; l++) {
                    freq = frequencies[k][l];
                    if (freq > 0.0) {
                        freq = freq / pop_size;
                        mi_matrix[i][j] += -freq * log(freq);
                        frequencies[k][l] = 0.0; // reset the freq;
                    }
                }
            }
            mi_matrix[j][i] = mi_matrix[i][j];
        }

        for (uword p = 0; p < pop_size; p++) {
            val_i = pop_nodes[p][i];
            frequencies[val_i][val_i] += 1.0;
        }

        double_t freq;
        for (uword k = 0; k < encode_number; k++) {
            for (uword l = 0; l < encode_number; l++) {
                freq = frequencies[k][l];
                if (freq > 0) {
                    freq = freq / pop_size;
                    mi_matrix[i][i] += -freq * log(freq);
                    frequencies[k][l] = 0.0; // reset the freq;
                }
            }
        }
    }

    return mi_matrix;
}

// ---- verbatim: GOMEAFOS.cpp:303-530 minus 312-380 (S1, S4, S5, S6) ---------
std::vector<std::vector<size_t> > GOMEAFOS::GenerateLinkageTreeFOS(size_t number_of_nodes, const vector<vector<uword>> & pop_nodes, uword encode_number, mat_t & MI_distribution_adjustments, bool fos_no_root_swap, bool use_mi_branch, const vector<size_t> & injected_order, vector<vector<double_t>> & out_sim, vector<vector<size_t>> & out_full_fos) {

    vector<vector < size_t>> FOS;

    vector<vector < double_t >> mi_matrix = EntropyLoops(number_of_nodes, pop_nodes, encode_number); // S4

    // transform entropy into mutual information
    //if (!MI_distribution_adjustments) {    // we always apply the correction
    if (use_mi_branch){ // S5: was if (false){

        for (size_t i = 0; i < number_of_nodes; i++) {
            for (size_t j = i + 1; j < number_of_nodes; j++) {
                mi_matrix[i][j] = mi_matrix[i][i] + mi_matrix[j][j] - mi_matrix[i][j];
                mi_matrix[j][i] = mi_matrix[i][j];
            }
        }

    } else {

        // first call, set it all up
        if (accu(MI_distribution_adjustments)==0) {
            for (size_t i = 0; i < number_of_nodes; i++) {
                MI_distribution_adjustments[i][i] = 1.0 / mi_matrix[i][i];
                for (size_t j = i + 1; j < number_of_nodes; j++) {
                    MI_distribution_adjustments[i][j] = 2.0 / mi_matrix[i][j];
                }
            }
        }

        for (size_t i = 0; i < number_of_nodes; i++) {
            mi_matrix[i][i] = mi_matrix[i][i] * MI_distribution_adjustments[i][i];
            for (size_t j = i + 1; j < number_of_nodes; j++) {
                mi_matrix[i][j] = mi_matrix[i][j] * MI_distribution_adjustments[i][j];
            }
        }


        for (size_t i = 0; i < number_of_nodes; i++) {
            for (size_t j = i + 1; j < number_of_nodes; j++) {
                mi_matrix[i][j] = mi_matrix[i][i] + mi_matrix[j][j] - mi_matrix[i][j];
                mi_matrix[j][i] = mi_matrix[i][j];
            }
        }
    }

    out_sim = mi_matrix; // S6

    // assemble the Linkage Tree with UPGMA
    FOS = BuildLinkageTreeFromSimilarityMatrix(number_of_nodes, mi_matrix, injected_order); // S2

    out_full_fos = FOS; // S6

    FOS.pop_back(); // remove the root of the Linkage Tree (otherwise entire solution can be swapped during GOM)

    if (fos_no_root_swap) {
        for (size_t i = 0; i < FOS.size(); i++) {
            vector<size_t> * set = &FOS[i];
            auto it = find(set->begin(), set->end(), 0);
            if (it != set->end()) {
                set->erase(it);
            }
        }
        // remove empty sets and duplicates
        set<vector < size_t>> fos_set;
        for (vector<size_t> set : FOS) {
            if (!set.empty())
                fos_set.insert(set);
        }
        FOS = vector<vector < size_t >> (fos_set.begin(), fos_set.end());
    }

    bool no_univar = false;
    if (no_univar) {
        set<vector < size_t>> fos_set;
        for (vector<size_t> set : FOS) {
            if (set.size() > 1)
                fos_set.insert(set);
        }
        FOS = vector<vector < size_t >> (fos_set.begin(), fos_set.end());
    }

    return FOS;
}

// ---- new: I/O (S7) ---------------------------------------------------------
static void print_matrix(const vector<vector<double_t>> & m) {
    printf("[");
    for (size_t i = 0; i < m.size(); i++) {
        printf(i ? ",[" : "[");
        for (size_t j = 0; j < m[i].size(); j++)
            printf(j ? ",%.17g" : "%.17g", (double) m[i][j]);
        printf("]");
    }
    printf("]");
}

static void print_fos(const vector<vector<size_t>> & fos) {
    printf("[");
    for (size_t i = 0; i < fos.size(); i++) {
        printf(i ? ",[" : "[");
        for (size_t j = 0; j < fos[i].size(); j++)
            printf(j ? ",%zu" : "%zu", fos[i][j]);
        printf("]");
    }
    printf("]");
}

static vector<vector<uword>> read_pop(size_t n, size_t d, uword & encode_number) {
    vector<vector<uword>> pop(n, vector<uword>(d));
    for (size_t p = 0; p < n; p++)
        for (size_t j = 0; j < d; j++) {
            cin >> pop[p][j];
            if (pop[p][j] + 1 > encode_number)
                encode_number = pop[p][j] + 1;
        }
    return pop;
}

int main() {
    size_t T;
    cin >> T;
    printf("{\"cases\":[");
    for (size_t c = 0; c < T; c++) {
        size_t d, n0, n1;
        cin >> d >> n0 >> n1;
        uword enc0 = 0, enc1 = 0;
        vector<vector<uword>> pop0 = read_pop(n0, d, enc0);
        vector<vector<uword>> pop1 = read_pop(n1, d, enc1);
        vector<size_t> order(d);
        for (size_t i = 0; i < d; i++)
            cin >> order[i];
        if (!cin) {
            fprintf(stderr, "malformed input in case %zu\n", c);
            return 1;
        }

        vector<vector<double_t>> H = GOMEAFOS::EntropyLoops(d, pop1, enc1);
        vector<vector<double_t>> H0 = GOMEAFOS::EntropyLoops(d, pop0, enc0);

        vector<vector<double_t>> S_mi, S_b, S_first;
        vector<vector<size_t>> full_mi, full_b, full_first;
        mat_t unused(d, std::vector<double>(d, 0.0));
        vector<vector<size_t>> lt_mi = GOMEAFOS::GenerateLinkageTreeFOS(d, pop1, enc1, unused, false, true, order, S_mi, full_mi);

        mat_t adjustments(d, std::vector<double>(d, 0.0));
        // first call (H0 population) sets the adjustments; its FOS is discarded
        GOMEAFOS::GenerateLinkageTreeFOS(d, pop0, enc0, adjustments, false, false, order, S_first, full_first);
        vector<vector<size_t>> lt_b = GOMEAFOS::GenerateLinkageTreeFOS(d, pop1, enc1, adjustments, false, false, order, S_b, full_b);

        printf(c ? ",{" : "{");
        printf("\"H\":"); print_matrix(H);
        printf(",\"H0\":"); print_matrix(H0);
        printf(",\"S_mi\":"); print_matrix(S_mi);
        printf(",\"S_b\":"); print_matrix(S_b);
        printf(",\"lt_mi_full\":"); print_fos(full_mi);
        printf(",\"lt_mi\":"); print_fos(lt_mi);
        printf(",\"lt_b_full\":"); print_fos(full_b);
        printf(",\"lt_b\":"); print_fos(lt_b);
        printf("}");
    }
    printf("]}\n");
    return 0;
}
