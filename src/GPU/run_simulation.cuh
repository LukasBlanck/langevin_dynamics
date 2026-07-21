// this file only contains functions that launch kernels, but do not instantiate it;
// the definition of those kernels should be in kernels/

#include "../input/input.hpp"
#include "GPU/cuda_check.hpp"
#include "GPU/kernels/extraction.cuh"
#include "GPU/kernels/integration.cuh"
#include "GPU/kernels/pearson.cuh"
#include "GPU/kernels/reduction.cuh"
#include "GPU/kernels/rng.cuh"
#include "io/netCDF_writer.hpp"
#include "process/helpers.hpp"
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// layout:
// Block - trajectory
// per batch:
// integrate up to save every
// store observables    mem: [batch_size * N]
// reduction kernel: reduces batch_size Blocks to N in mem:[n_save * N] Matrix

// strided LOOP:
// Block always 256 threads, then
// for (std::size_t site = threadIdx.x;
//      site < N;
//      site += blockDim.x) {
//     // Process site.
// }

template <class Potential>
inline void run_simulation(const Config &config, const std::string &output_path) {

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const int save_every = config.time.save_every;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    const double left_bath_T = config.model.left_bath_T;
    const double gamma = config.model.lambda / m;

    // saving helpers
    const int n_save = 1 + (N_time + save_every - 1) / save_every;
    std::vector<double> tot_e(n_save * N, 0.0);
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

    // helpers for OU step
    const double c = (std::exp(-gamma * dt));
    const double eta = std::sqrt(m * kB * left_bath_T * (1 - c * c));

    // device constants
    const int batch_size = 256;
    constexpr int threads_per_block = 256; // TODO: test for 128 and 512 and correspondant runtim
    static_assert(threads_per_block > 0 && (threads_per_block & (threads_per_block - 1)) == 0,
                  "threads_per_block must be a power of two ! ()");

    const int number_of_batches = (N_ensemble + batch_size - 1) / batch_size;

    // device pointers
    double *d_p = nullptr;
    double *d_q = nullptr;
    double *d_tot_e_temporary = nullptr;
    double *d_tot_e = nullptr;
    double *d_pot_e_temporary = nullptr;
    double *d_pot_e = nullptr;
    double *d_kin_e_temporary = nullptr;
    double *d_kin_e = nullptr;

    // pearson correlators
    double *d_pj0 = nullptr;
    double *d_pj = nullptr;
    double *d_p0 = nullptr;

    double *d_pj2 = nullptr;
    double *d_p02 = nullptr;

    // rng device allocation
    curandStatePhilox4_32_10_t *d_rng_states = nullptr; // rng state (persistent per trajectory)
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_rng_states),
                   static_cast<std::size_t>(batch_size) * sizeof(curandStatePhilox4_32_10_t)));

    // bytes of shared memory on block
    const std::size_t shared_bytes = static_cast<std::size_t>(N) * sizeof(double);
    const int num_of_observables =
        3; // in the simplest form this is really the number of observables, but with later HPC
           // improvements the number can be smaller then the number of observables
    const std::size_t reduction_shared_bytes =
        static_cast<std::size_t>(threads_per_block) * num_of_observables * sizeof(double);
    const int num_of_pearson_observables = 5;
    const std::size_t pearson_reduction_shared_bytes =
        static_cast<std::size_t>(threads_per_block) * num_of_pearson_observables * sizeof(double);

    // "temporary" q and p arrays [batch_size * N]
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q),
                          static_cast<std::size_t>(N) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p),
                          static_cast<std::size_t>(N) * batch_size * sizeof(double)));

    // reusable (temporary observables) [batch_size * N]
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tot_e_temporary),
                          static_cast<std::size_t>(N) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pot_e_temporary),
                          static_cast<std::size_t>(N) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_kin_e_temporary),
                          static_cast<std::size_t>(N) * batch_size * sizeof(double)));

    // Final observables [n_save * N]
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tot_e),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pot_e),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_kin_e),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pj0),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pj),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p0),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pj2),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p02),
                          static_cast<std::size_t>(N) * n_save * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_tot_e, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pot_e, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_kin_e, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_pj0, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pj, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p0, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_pj2, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_p02, 0, static_cast<std::size_t>(N) * n_save * sizeof(double)));

    // TODO:
    // at n_save_index=10 a ETA?
    // add n_save == 1000
    // add timing?
    // here maybe stable dt calculation? or seperate with stable dt at runtime?

    // begin itegration (per batch)
    for (int batch = 0; batch < number_of_batches; ++batch) {

        const int batch_begin = batch * batch_size;
        const int current_batch_size =
            std::min(batch_size, N_ensemble - batch_begin); // last batch might be smaller

        // Initialize q, p, F and RNG for this batch.
        CUDA_CHECK(cudaMemset(d_q, 0, static_cast<std::size_t>(N) * batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_p, 0, static_cast<std::size_t>(N) * batch_size * sizeof(double)));

        // initialize the rng states per batch
        constexpr int rng_threads_per_block = threads_per_block;
        const int rng_blocks = (current_batch_size + rng_threads_per_block - 1) /
                               rng_threads_per_block; // this batch has currently #rng_blocks blocks
        initialize_rng_states<<<rng_blocks, rng_threads_per_block>>>(
            d_rng_states, static_cast<unsigned long long>(seed), batch_begin, current_batch_size);
        CUDA_CHECK(cudaGetLastError());

        int completed_steps = 0;
        int n_save_index = 0;

        // initial measurement at t = 0.
        extract_observables<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
            d_p, d_q, d_tot_e_temporary, d_pot_e_temporary, d_kin_e_temporary, potential,
            current_batch_size, N, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // pearson reduction
        pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
           d_p, d_pj0, d_pj, d_p0, d_pj2, d_p02, N, current_batch_size, n_save_index);

        // launch N blocks - one block is one site
        perform_reduction<<<N, threads_per_block, reduction_shared_bytes>>>(
            d_tot_e_temporary, d_pot_e_temporary, d_kin_e_temporary, d_tot_e, d_pot_e, d_kin_e,
            current_batch_size, N, n_save_index);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        ++n_save_index;

        while (completed_steps < N_time) { // regular reduction also in time resolution

            const int steps_this_interval =
                std::min(save_every, N_time - completed_steps); // might be less at end

            integrate<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
                d_p, d_q, d_rng_states, potential, current_batch_size, N, steps_this_interval, m,
                eta, c, dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            extract_observables<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
                d_p, d_q, d_tot_e_temporary, d_pot_e_temporary, d_kin_e_temporary, potential,
                current_batch_size, N, m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // pearson reduction
            pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
                d_p, d_pj0, d_pj, d_p0, d_pj2, d_p02, N, current_batch_size, n_save_index);

            // launch N blocks - one block is one site
            perform_reduction<<<N, threads_per_block, reduction_shared_bytes>>>(
                d_tot_e_temporary, d_pot_e_temporary, d_kin_e_temporary, d_tot_e, d_pot_e, d_kin_e,
                current_batch_size, N, n_save_index);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            completed_steps += steps_this_interval;
            ++n_save_index;
        }
    }

    // Copy reduced [n_save, N] arrays to CPU.
    CUDA_CHECK(cudaMemcpy(tot_e.data(), d_tot_e,
                          static_cast<std::size_t>(N) * n_save * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pot_e.data(), d_pot_e,
                          static_cast<std::size_t>(N) * n_save * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kin_e.data(), d_kin_e,
                          static_cast<std::size_t>(N) * n_save * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // free GPU
    CUDA_CHECK(cudaFree(d_tot_e));
    CUDA_CHECK(cudaFree(d_tot_e_temporary));
    CUDA_CHECK(cudaFree(d_pot_e));
    CUDA_CHECK(cudaFree(d_pot_e_temporary));
    CUDA_CHECK(cudaFree(d_kin_e));
    CUDA_CHECK(cudaFree(d_kin_e_temporary));

    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));

    // Normalize by N_ensemble.
    for (double &value : tot_e) {
        value /= static_cast<double>(N_ensemble);
    }
    for (double &value : pot_e) {
        value /= static_cast<double>(N_ensemble);
    }
    for (double &value : kin_e) {
        value /= static_cast<double>(N_ensemble);
    }

    // Compute derived observables.

    // process weighted energies
    normalized_energy(tot_e, normalized_tot_e, n_save, N);
    normalized_energy(kin_e, normalized_kin_e, n_save, N);
    normalized_energy(pot_e, normalized_pot_e, n_save, N);

    first_moment(normalized_tot_e, first_moment_tot_e, n_save, N);
    spread(normalized_tot_e, tot_energy_spread, first_moment_tot_e, n_save, N);

    // for now compute time on host
    for (int save_index = 1; save_index < n_save; ++save_index) {
        const int completed = std::min(save_index * save_every, N_time);

        time[save_index] = completed * dt;
    }

    // write results (helper arrays) to .nc file
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    NetCDFWriter writer(output_path, config, n_save, N, dt);

    writer.write_time(time);
    // energy observables
    writer.write_time_site_array("local_total_energy", "ensemble averaged local total energy",
                                 "energy", tot_e);
    writer.write_time_site_array("local_potential_energy",
                                 "ensemble averaged local potential energy", "energy", pot_e);
    writer.write_time_site_array("local_kinetic_energy", "ensemble averaged local kinetic energy",
                                 "energy", kin_e);

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

    // simulation finished
    std::cout << "Finished simulation.\n";
    std::cout << "Output written to: " << output_path << "\n";
}
