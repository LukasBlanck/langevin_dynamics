// this file runs the BAOAB solver and tests, whether the ensemble observables converge with at
// least order of two

#include "../../src/CPU/BAOAB.hpp"
#include "../../src/input/input.hpp"
#include "../../src/potentials.hpp"
#include "test_helpers.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// TODO:
// Adapt this test for Monte-Carlo error estimation with batches
// This should enable to say confidently, wether the ensemble size
// and N_time big enough to not be dominated by stochastic noise
// and therefore cleanly inspect weak 2nd order convergence

TEST(test_convergence, ensemble_observables) {

    Config config = create_config();

    const int N = config.grid.N; // 64
    const double m = config.conventions.m;
    config.model.lambda = 0; // for later weak second order convergence this must be > 0.0
    const int N_time = 100;
    const double end_time = config.time.end_time; // 5
    const int N_ensemble = 10;

    // create initial grid
    std::vector<double> q0(N, 0.0);
    std::vector<double> p0(N, 0.0);
    std::vector<double> F_fpu(N, 0.0);
    std::vector<double> F_jos(N, 0.0);

    // non trivial initialization
    for (int i = 0; i < N; ++i) {
        q0[i] = 0.1 * std::sin(0.3 * static_cast<double>(i));
        p0[i] = 0.2 * std::cos(0.2 * static_cast<double>(i));
    }
    std::vector<double> q_fpu = q0;
    std::vector<double> p_fpu = p0;
    std::vector<double> q_jos = q0;
    std::vector<double> p_jos = p0;

    // helper arrays for storing observables for different dt
    const int N_dt = 10;
    std::vector<int> N_time_vec(N_dt);
    std::vector<double> dt_vec(N_dt);
    std::vector<double> total_energy(N_dt);
    for (int i = 0; i < N_dt; i++) {
        N_time_vec[i] = N_time * (1 << i);
        dt_vec[i] = end_time / static_cast<double>(N_time_vec[i]);
    }

    // create potentials
    FPUPotential fpu_potential(config);
    JosephsonPotential jos_potential(config);

    // initialize rng
    int seed = 67;

    std::mt19937_64 rng_jos(seed);

    // FPU
    for (int b = 0; b < N_dt; b++) {
        double sum_energy = 0.0;
        for (int n = 0; n < N_ensemble; n++) {
            // re-initialize starting params
            std::vector<double> q_fpu = q0;
            std::vector<double> p_fpu = p0;
            std::vector<double> F_fpu(N, 0.0);
            BAOAB<FPUPotential> fpu_integrator(config, q_fpu, p_fpu, F_fpu,
                                               dt_vec[b] /*dt always halfed per ensemble iter*/);
            std::mt19937_64 rng_fpu(seed + n);
            for (int i = 0; i < N_time_vec[b]; i++) {
                fpu_integrator.step(rng_fpu);
            }
            sum_energy += symmetric_energy(q_fpu, p_fpu, N, m, fpu_potential);
        }
        total_energy[b] = sum_energy / static_cast<double>(N_ensemble);
    }

    // assume smallest dt is perfect
    const double total_energy_smallest_dt = total_energy[N_dt - 1];
    std::vector<double> err(N_dt - 1);

    for (int b = 0; b < N_dt - 1; b++) {
        err[b] = std::abs(total_energy[b] - total_energy_smallest_dt);
        std::cout << std::left << std::setw(15) << "FPU:"
                  << "error at dt = " << std::left << std::setw(15) << dt_vec[b] << std::right
                  << std::setw(10) << "error: " << err[b] << '\n';
    }

    for (int b = 0; b < N_dt - 2; ++b) {
        if (err[b + 1] > 0.0) {
            const double ratio = err[b] / err[b + 1];
            std::cout << "FPU error ratio err[" << b << "] / err[" << b + 1 << "] = " << ratio << "\n";
            // For lambda = 0, we must get Velocity-Verlet second order convergence
            EXPECT_GT(ratio, 4.0);
        }
    }
}