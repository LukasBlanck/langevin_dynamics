#include "../../src/CPU/BAOAB.hpp"
#include "../../src/input/input.hpp"
#include "../../src/potentials.hpp"
#include "test_helpers.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <iostream>

// test global energy conservation for lamda=0
// depending on dt, it should give let`s say at least 1e-8 relative error

TEST(test_energy, conservation_for_lambda_equal_0) {

    Config config = create_config();
    config.model.lambda = 0;

    const int N = config.grid.N;
    const double m = config.conventions.m;
    const int N_time = config.time.N;
    const double end_time = config.time.end_time;
    const int save_every = config.time.save_every;
    const double dt = end_time / static_cast<double>(N_time);

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

    // create potentials
    FPUPotential fpu_potential(config);
    JosephsonPotential jos_potential(config);

    // extract initial energy
    double fpu_initial_energy = symmetric_energy(q_fpu, p_fpu, N, m, fpu_potential);
    double jos_initial_energy = symmetric_energy(q_jos, p_jos, N, m, jos_potential);

    // run integrator
    double fpu_current_energy = 0.0;
    double jos_current_energy = 0.0;
    double fpu_rel_error = 0.0;
    double jos_rel_error = 0.0;
    double fpu_max_rel_error = 0.0;
    double jos_max_rel_error = 0.0;

    int seed = 67;
    std::mt19937_64 rng_fpu(seed);
    std::mt19937_64 rng_jos(seed);
    const double tolerance = 1e-10;

    // FPU
    BAOAB<FPUPotential> fpu_integrator(config, q_fpu, p_fpu, F_fpu, dt);
    for (int k = 0; k < N_time; k++) {
        fpu_integrator.step(rng_fpu);

        if ((k + 1) % save_every == 0 || k + 1 == N_time) {
            fpu_current_energy = symmetric_energy(q_fpu, p_fpu, N, m, fpu_potential);
            fpu_rel_error =
                std::abs((fpu_initial_energy - fpu_current_energy) / fpu_initial_energy);
            fpu_max_rel_error = std::max(fpu_max_rel_error, fpu_rel_error);
        }
    }
    EXPECT_LE(fpu_max_rel_error, tolerance);
    std::cout << std::left << std::setw(15) << "\nFPU:" << std::left << std::setw(24)
              << "max relative error:" << std::right << std::setw(10) << fpu_max_rel_error;

    // Josephson
    BAOAB<JosephsonPotential> jos_integrator(config, q_jos, p_jos, F_jos, dt);
    for (int k = 0; k < N_time; k++) {
        jos_integrator.step(rng_jos);

        if ((k + 1) % save_every == 0 || k + 1 == N_time) {
            jos_current_energy = symmetric_energy(q_jos, p_jos, N, m, jos_potential);
            jos_rel_error =
                std::abs((jos_initial_energy - jos_current_energy) / jos_initial_energy);
            jos_max_rel_error = std::max(jos_max_rel_error, jos_rel_error);
        }
    }
    EXPECT_LE(jos_max_rel_error, tolerance);
    std::cout << std::left << std::setw(15) << "\nJosephson:" << std::left << std::setw(24)
              << "max relative error:" << std::right << std::setw(10) << jos_max_rel_error
              << "\n\n";
}