// this file provides helper functions that process raw result
// arrays [n_save * N] from the GPU on the CPU

#include <cstdlib>
#include <vector>

// normalized local energy
inline void normalized_energy(const std::vector<double> &e, std::vector<double> &normalized_e,
                              int n_save, int N) {
    // the stride ist t*N <=> the spatial reslution is the stride

    // only spreading is interesting to us

    for (int t = 0; t < n_save; t++) {

        double total_excess_per_t = 0.0;
        // extract total (excess) energy per time
        for (int j = 0; j < N; ++j) {
            const double excess = e[t * N + j] - e[j]; // e_j = e_j(t) - e_j(0)
            total_excess_per_t += excess;
        }

        // avoid division by zero:
        if (std::abs(total_excess_per_t) < 1e-14) {
            for (int j = 0; j < N; ++j) {
                normalized_e[t * N + j] = 0.0;
            }
            continue;
        }

        // normalize
        for (int j = 0; j < N; ++j) {
            const double excess = e[t * N + j] - e[j];
            normalized_e[t * N + j] = excess / total_excess_per_t;
        }
    }
}