#pragma once

#include "input/input.hpp"
#include <vector>

inline Config create_config() {
    Config config;

    // standard params for testing
    config.grid.N = 128;

    config.conventions.m = 1.0;
    config.conventions.kB = 1.0;

    config.time.end_time = 5.0;
    config.time.N = 100000;
    config.time.save_every = 10;

    config.ensemble.N = 10;

    config.model.beta = 1.0;
    config.model.EJ = 1.0;
    config.model.lambda = 1.0;
    config.model.left_bath_T = 2.0;
    config.model.omega = 1.0;
    config.model.potential = "FPU";

    return config;
}

// calculate total energy
template <class Potential>
inline double symmetric_energy(std::vector<double> &q, std::vector<double> &p,
                             int N, double m, Potential potential) {
    double e = 0.0;
    
    // inside sites
    for (int i = 1; i < N - 1; i++) {
        e += p[i] * p[i] / (2 * m) + 0.5 * potential.V(q[i - 1] - q[i]) +
               0.5 * potential.V(q[i] - q[i + 1]);
    }
    // boundary
    e += p[0] * p[0] / (2 * m) + 0.5 * potential.V(q[0] - q[1]);
    e += p[N - 1] * p[N - 1] / (2 * m) + 0.5 * potential.V(q[N - 2] - q[N - 1]);

    return e;
}