// this file provides functions to extract observables at runtime
#pragma once

#include <vector>

template <class Potential>
inline void symmetric_energy(std::vector<double> &e, const std::vector<double> &q, const std::vector<double> &p,
                             int count, int N, double m, const Potential potential) {

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