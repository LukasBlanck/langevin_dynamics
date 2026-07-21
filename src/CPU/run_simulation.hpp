// this file contains the highest logical abstraction of the run_simulation() function
// it is seperate for visibility reasons of the main.cpp
#pragma once

#include "../input/input.hpp"
#include "../io/netCDF_writer.hpp"
#include "BAOAB.hpp"
#include "extraction_helpers.hpp"
#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

template <class Potential>
inline void run_simulation(const Config &config, const std::string &output_path) {
    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    // saving helpers
    constexpr std::int64_t target_n_save =
        1000; // ensure good visual resolution and small enouggh memory demand
    if (N_time < target_n_save - 1) {
        throw std::invalid_argument("N_time must be at least 999");
    }
    const int save_every = static_cast<int>((N_time - 1) / (target_n_save - 2));
    const int n_save = static_cast<int>(1 + (N_time + save_every - 1) / save_every);

    if (n_save < target_n_save) {
        throw std::logic_error("Internal error: n_save is below target");
    }
    std::vector<double> e(n_save * N, 0.0);
    std::vector<double> time(n_save, 0.0);
    std::vector<double> kin_e(n_save * N, 0.0);
    std::vector<double> pot_e(n_save * N, 0.0);
    std::vector<double> normalized_tot_e(n_save * N, 0.0);
    std::vector<double> normalized_pot_e(n_save * N, 0.0); // might not be numerically reliable
    std::vector<double> normalized_kin_e(n_save * N, 0.0); // due to divisions with small numbers
    std::vector<double> first_moment_tot_e(n_save, 0.0);
    std::vector<double> tot_energy_spread(n_save, 0.0);

    // pearson correlators
    std::vector<double> pj0(n_save * N, 0.0);
    std::vector<double> pj(n_save * N, 0.0);
    std::vector<double> p0(n_save * N, 0.0);

    // pearson normalize vectors
    std::vector<double> pj2(n_save * N, 0.0); // <pj^2>
    std::vector<double> p02(n_save * N, 0.0); // <p0^2>

    // pearson correlators
    std::vector<double> qj0(n_save * N, 0.0);
    std::vector<double> qj(n_save * N, 0.0);
    std::vector<double> q0(n_save * N, 0.0);

    // pearson normalize vectors
    std::vector<double> qj2(n_save * N, 0.0); // <qj^2>
    std::vector<double> q02(n_save * N, 0.0); // <q0^2>

    const int N_bond = N - 1;
    // Pearson bond-displacement correlators r_j with r_0
    std::vector<double> rj0(n_save * N_bond, 0.0);
    std::vector<double> rj(n_save * N_bond, 0.0);
    std::vector<double> r0(n_save * N_bond, 0.0);

    std::vector<double> rj2(n_save * N_bond, 0.0);
    std::vector<double> r02(n_save * N_bond, 0.0);

    int seed = 67;
    Potential potential(config);

    // ETA
    const int eta_sample = 100;
    const auto wall_start = std::chrono::steady_clock::now();
    bool eta_printed = false;

    for (int n = 0; n < N_ensemble; n++) {
        int count = 0;

        // create initial grid
        std::vector<double> q(N, 0.0);
        std::vector<double> p(N, 0.0);
        std::vector<double> F(N, 0.0);

        std::mt19937_64 rng(seed + n);

        // integrate
        BAOAB<Potential> integrator(config, q, p, F, dt);

        if (n == 0) {
            time[count] = 0.0;
        }

        // save initial condition
        symmetric_energy(e, q, p, count, N, m, potential);
        kinetic_energy(kin_e, p, count, N, m);
        potential_energy(pot_e, q, count, N, potential);

        pearson_correlators(pj0, pj, p0, pj2, p02, p, count, N);
        pearson_correlators(qj0, qj, q0, qj2, q02, q, count, N);
        pearson_bond_correlators(rj0, rj, r0, rj2, r02, q, count, N_bond);
        count++;

        // per trajectory
        for (int k = 0; k < N_time; k++) {
            integrator.step(rng);

            if ((k + 1) % save_every == 0 || k + 1 == N_time) {
                double t = (k + 1) * dt; // time after step
                // save observables
                symmetric_energy(e, q, p, count, N, m, potential);
                kinetic_energy(kin_e, p, count, N, m);
                potential_energy(pot_e, q, count, N, potential);

                pearson_correlators(pj0, pj, p0, pj2, p02, p, count, N);
                pearson_correlators(qj0, qj, q0, qj2, q02, q, count, N);
                pearson_bond_correlators(rj0, rj, r0, rj2, r02, q, count, N_bond);

                if (n == 0) {
                    time[count] = t; // store exact time seperately
                }
                count++;
            }
        }

        // safety check
        if (count != n_save) {
            throw std::runtime_error("count != n_save after trajectory");
        }
        // ETA
        if (!eta_printed && n + 1 == eta_sample) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
                    .count();
            const long long eta = static_cast<long long>(
                std::round(elapsed * (N_ensemble - eta_sample) / eta_sample));
            std::cout << "ETA: " << eta / 60 << ":" << std::setw(2) << std::setfill('0') << eta % 60
                      << " min:s" << std::setfill(' ') << "\n\n";
            eta_printed = true;
        }
    }

    // normalize
    const double inv_ensemble = 1.0 / static_cast<double>(N_ensemble);
    for (int i = 0; i < n_save * N; i++) {
        e[i] = e[i] * inv_ensemble;
        kin_e[i] = kin_e[i] * inv_ensemble;
        pot_e[i] = pot_e[i] * inv_ensemble;
    }

    // process weighted energies
    normalized_energy(e, normalized_tot_e, n_save, N);
    normalized_energy(kin_e, normalized_kin_e, n_save, N);
    normalized_energy(pot_e, normalized_pot_e, n_save, N);

    first_moment(normalized_tot_e, first_moment_tot_e, n_save, N);
    spread(normalized_tot_e, tot_energy_spread, first_moment_tot_e, n_save, N);

    // process pearson
    std::vector<double> corr_p0(n_save * N, 0.0);
    std::vector<double> corr_q0(n_save * N, 0.0);
    std::vector<double> corr_r0(n_save * N_bond, 0.0);
    process_pearson_correlators(corr_p0, pj0, pj, p0, pj2, p02, n_save, N, inv_ensemble);
    process_pearson_correlators(corr_q0, qj0, qj, q0, qj2, q02, n_save, N, inv_ensemble);
    process_pearson_correlators(corr_r0, rj0, rj, r0, rj2, r02, n_save, N_bond, inv_ensemble);

    // write results
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    NetCDFWriter writer(output_path, config, n_save, N, dt);

    writer.write_time(time);
    // energy observables
    writer.write_time_site_array("local_total_energy", "ensemble averaged local total energy",
                                 "energy", e);
    writer.write_time_site_array("local_kinetic_energy", "ensemble averaged local kinetic energy",
                                 "energy", kin_e);
    writer.write_time_site_array("local_potential_energy",
                                 "ensemble averaged local potential energy", "energy", pot_e);

    // weighted energies
    writer.write_time_site_array("normalized_total_energy",
                                 "ensemble averaged normalized local total energy", "dimensionless",
                                 normalized_tot_e);
    writer.write_time_site_array("normalized_kinetic_energy",
                                 "ensemble averaged normalized local kinetic energy",
                                 "dimensionless", normalized_kin_e);
    writer.write_time_site_array("normalized_potential_energy",
                                 "ensemble averaged normalized local potential energy",
                                 "dimensionless", normalized_pot_e);

    // moments
    writer.write_time_data_array("first_moment_total_energy",
                                 "ensemble averaged first momentum of total energy", "site",
                                 first_moment_tot_e);
    writer.write_time_data_array("total_energy_spread", "ensemble averaged spread of total energy",
                                 "site", tot_energy_spread);

    // pearson correlation
    writer.write_time_site_array("pearson_momentum_correlation",
                                 "Pearson momentum correlation with left boundary (site at 0)",
                                 "dimensionless", corr_p0);
    writer.write_time_site_array("pearson_position_correlation",
                                 "Pearson position correlation with left boundary (site at 0)",
                                 "dimensionless", corr_q0);
    writer.write_time_bond_array("pearson_bond_correlation",
                                 "Pearson bond displacement correlation with left boundary bond",
                                 "dimensionless", corr_r0);

    // finished simulation
    std::cout << "Finished simulation.\n";
    std::cout << "Output written to: " << output_path << "\n";
}