// this file provides functions to extract observables at runtime
#pragma once

#include <vector>
#include <cmath>
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
        e[i + stride] = p[i] * p[i] / (2 * m) + 0.5 * potential.V(q[i - 1] - q[i]) +
                        0.5 * potential.V(q[i] - q[i + 1]);
    }
    // boundary
    e[0 + stride] = p[0] * p[0] / (2 * m) + 0.5 * potential.V(q[0] - q[1]);
    e[N - 1 + stride] = p[N - 1] * p[N - 1] / (2 * m) + 0.5 * potential.V(q[N - 2] - q[N - 1]);
}

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

inline void process_pearson_correlators(std::vector<double>& corr_p0, const std::vector<double> &pj0,
                                        const std::vector<double> &pj,
                                        const std::vector<double> &p0,
                                        const std::vector<double> &pj2,
                                        const std::vector<double> &p02, int n_save, int N,
                                        double inv_ensemble) {
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