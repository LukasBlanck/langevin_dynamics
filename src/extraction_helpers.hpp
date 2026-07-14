// this file provides functions to extract observables at runtime
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

template <class Potential>
inline void symmetric_energy(std::vector<double> &e, const std::vector<double> &q,
                             const std::vector<double> &p, int count, int N, double m,
                             const Potential &potential) {

    // the stride ist count*N <=> the spatial reslution is the stride

    int stride = count * N;
    // inside sites
    for (int i = 1; i < N - 1; i++) {
        e[i + stride] += p[i] * p[i] / (2 * m) + 0.5 * potential.V(q[i - 1] - q[i]) +
                         0.5 * potential.V(q[i] - q[i + 1]);
    }
    // boundary
    e[0 + stride] += p[0] * p[0] / (2 * m) + 0.5 * potential.V(q[0] - q[1]);
    e[N - 1 + stride] += p[N - 1] * p[N - 1] / (2 * m) + 0.5 * potential.V(q[N - 2] - q[N - 1]);
}

// only kinetic energy
inline void kinetic_energy(std::vector<double> &kin_e, const std::vector<double> &p, int count,
                           int N, double m) {
    // the stride ist count*N <=> the spatial reslution is the stride

    int stride = count * N;
    for (int i = 0; i < N; i++) {
        kin_e[i + stride] += p[i] * p[i] / (2 * m);
    }
}

// only potential energy
template <class Potential>
inline void potential_energy(std::vector<double> &pot_e, const std::vector<double> &q, int count,
                             int N, const Potential &potential) {
    // the stride ist count*N <=> the spatial reslution is the stride

    int stride = count * N;
    // inside sites
    for (int i = 1; i < N - 1; i++) {
        pot_e[i + stride] +=
            0.5 * potential.V(q[i - 1] - q[i]) + 0.5 * potential.V(q[i] - q[i + 1]);
    }
    // boundary
    pot_e[0 + stride] += 0.5 * potential.V(q[0] - q[1]);
    pot_e[N - 1 + stride] += 0.5 * potential.V(q[N - 2] - q[N - 1]);
}

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
inline void spread(const std::vector<double> &normalized_e,
                          std::vector<double> &spread, std::vector<double> &first_moments,
                          int n_save, int N) {
    std::vector<double> first_moments_squared(n_save);
    for (int t = 0; t < n_save; t++) {
        for (int j = 0; j < N; j++) {
            first_moments_squared[t] += normalized_e[t * N + j] * static_cast<double>(j * j);
        }
        spread[t] =
            std::sqrt(first_moments_squared[t] - (first_moments[t] * first_moments[t]));
    }
}

// pearson correlators
inline void pearson_correlators(std::vector<double> &pj0, std::vector<double> &pj,
                                std::vector<double> &p0, std::vector<double> &pj2,
                                std::vector<double> &p02, const std::vector<double> &p, int count,
                                int N) {
    // the stride ist count*N <=> the spatial reslution is the stride

    int stride = count * N;

    for (int i = 0; i < N; i++) {
        pj0[i + stride] += p[i] * p[0];
        pj[i + stride] += p[i];
        p0[i + stride] += p[0];

        // normalize
        pj2[i + stride] += p[i] * p[i];
        p02[i + stride] += p[0] * p[0];
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

inline void pearson_bond_correlators(std::vector<double> &rj0, std::vector<double> &rj,
                                     std::vector<double> &r0, std::vector<double> &rj2,
                                     std::vector<double> &r02, const std::vector<double> &q,
                                     int count, int N_bond) {
    const int stride = count * N_bond;

    const double r_left = q[0] - q[1];

    for (int b = 0; b < N_bond; ++b) {
        const double rb = q[b] - q[b + 1];

        rj0[stride + b] += rb * r_left;
        rj[stride + b] += rb;
        r0[stride + b] += r_left;

        rj2[stride + b] += rb * rb;
        r02[stride + b] += r_left * r_left;
    }
}