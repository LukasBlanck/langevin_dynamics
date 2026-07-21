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

// first moment or centre of mass/energy
inline void first_moment(const std::vector<double> &normalized_e,
                         std::vector<double> &first_moments, int n_save, int N) {

    for (int t = 0; t < n_save; t++) {
        for (int j = 0; j < N; j++) {
            first_moments[t] += normalized_e[t * N + j] * static_cast<double>(j);
        }
    }
}

// sqrt(second moment), spread
inline void spread(const std::vector<double> &normalized_e, std::vector<double> &spread,
                   std::vector<double> &first_moments, int n_save, int N) {
    std::vector<double> first_moments_squared(n_save);
    for (int t = 0; t < n_save; t++) {
        for (int j = 0; j < N; j++) {
            first_moments_squared[t] += normalized_e[t * N + j] * static_cast<double>(j * j);
        }
        spread[t] = std::sqrt(first_moments_squared[t] - (first_moments[t] * first_moments[t]));
    }
}

inline void
process_pearson_correlators(std::vector<double> &corr_p0, const std::vector<double> &pj0,
                            const std::vector<double> &pj, const std::vector<double> &p0,
                            const std::vector<double> &pj2, const std::vector<double> &p02,
                            int n_save, int N, double inv_ensemble) {
    for (int i = 0; i < n_save * N; i++) {
        const double mean_pj0 = pj0[i] * inv_ensemble;
        const double mean_pj = pj[i] * inv_ensemble;
        const double mean_pj2 = pj2[i] * inv_ensemble;
        const double mean_p0 = p0[i] * inv_ensemble;
        const double mean_p02 = p02[i] * inv_ensemble;

        const double var_pj = mean_pj2 - mean_pj * mean_pj;
        const double var_p0 = mean_p02 - mean_p0 * mean_p0;
        const double cov = mean_pj0 - mean_pj * mean_p0;

        const double eps = 1e-14;
        if (var_pj < -eps || var_p0 < -eps) {
            throw std::runtime_error("Negative variance in Pearson correlation calculation");
        }

        if (var_pj > eps && var_p0 > eps) {
            corr_p0[i] = cov / std::sqrt(var_pj * var_p0);
        } else {
            corr_p0[i] = 0.0;
        }
    }
}