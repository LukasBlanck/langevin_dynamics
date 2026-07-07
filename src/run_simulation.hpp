// this file contains the highest logical abstraction of the run_simulation() function
// it is seperate for visibility reasons of the main.cpp
#pragma once

#include "BAOAB.hpp"
#include "extraction_helpers.hpp"
#include "input/input.hpp"
#include "io/netCDF_writer.hpp"
#include <filesystem>
#include <stdexcept>
#include <vector>

template <class Potential> inline void run_simulation(const Config &config, const std::string& output_path) {
    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const int save_every = config.time.save_every;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    // saving helpers
    const int n_save = 1 + (N_time + save_every - 1) / save_every;
    std::vector<double> e(n_save * N, 0.0);
    std::vector<double> sum_e(n_save * N, 0.0);
    std::vector<double> time(n_save, 0.0);

    int seed = 67;
    Potential potential(config);

    for (int n = 0; n < N_ensemble; n++) {
        int count = 0;

        std::fill(e.begin(), e.end(), 0.0);

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
        count++;

        // per trajectory
        for (int k = 0; k < N_time; k++) {
            integrator.step(rng);

            if ((k + 1) % save_every == 0 || k + 1 == N_time) {
                double t = (k + 1) * dt; // time after step
                // save observables
                symmetric_energy(e, q, p, count, N, m, potential);

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

        // accumulate over ensembles
        for (int i = 0; i < n_save * N; i++) {
            sum_e[i] += e[i];
        }
    }

    // normalize
    for (int i = 0; i < n_save * N; i++) {
        sum_e[i] = sum_e[i] / N_ensemble;
    }

    // write results
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    NetCDFWriter writer(output_path, config, n_save, N, dt);

    writer.write_time(time);
    writer.write_time_site_array("local_energy", "ensemble averaged local energy", "energy", sum_e);

    // finished simulation
    std::cout << "Finished simulation.\n";
    std::cout << "Output written to: " << output_path << "\n";
}