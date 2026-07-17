// this file only contains functions that launch kernels, but do not instantiate it;
// the definition of those kernels should be in kernels/

#include "../input/input.hpp"
#include "GPU/cuda_check.hpp"
#include "GPU/kernels/extraction.cuh"
#include "GPU/kernels/integration.cuh"
#include <cstddef>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

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

    // helpers for OU step
    const double c = (std::exp(-gamma * dt));
    const double eta = std::sqrt(m * kB * left_bath_T * (1 - c * c));

    // device constants
    const int batch_size = 256;
    const int threads_per_block = 256;

    const int number_of_batches = (batch_size + N_ensemble - 1) / batch_size;

    // device pointers
    double *d_p = nullptr;
    double *d_q = nullptr;
    double *d_tot_e = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_q),
                           static_cast<std::size_t>(N) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p),
                           static_cast<std::size_t>(N) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tot_e),
                           static_cast<std::size_t>(N) * batch_size * sizeof(double)));

    // Allocate final device observables:
    // [n_save, N]

    // Allocate reusable temporary observables:
    // [max_batch_size, N]

    for (int batch = 0; batch < number_of_batches; ++batch) {

        const int batch_begin = batch * batch_size;
        const int current_batch_size =
            std::min(batch_size, N_ensemble - batch_begin); // last batch might be smaller

        // Initialize q, p, F and RNG for this batch.
        CUDA_CHECK(cudaMemset(d_q, 0, static_cast<std::size_t>(N) * batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_p, 0, static_cast<std::size_t>(N) * batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_tot_e, 0, static_cast<std::size_t>(N) * batch_size * sizeof(double)));

        int completed_steps = 0;
        int n_save_index = 0;

        // initial measurement at t = 0.
        // extract_observables(current_batch_size, threads_per_block,
        //                     N); // first reduction, then extract?? better HPC?
        // reduction(current_batch_size, N, n_save_index);
        ++n_save_index;

        while (completed_steps < N_time) { // regular reduction also in time resolution

            const int steps_this_interval =
                std::min(save_every, N_time - completed_steps); // might be less at end

            const std::size_t shared_bytes = static_cast<std::size_t>(N) * sizeof(double);

            integrate<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
                d_p, d_q, potential, current_batch_size, N, steps_this_interval, completed_steps,
                static_cast<unsigned long long>(seed), m, eta, c, batch_begin, dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            extract_observables(d_p, d_q, d_tot_e, potential, current_batch_size,  N,  m,  int n_save_index);
            // reduction(current_batch_size, N, n_save_index);

            completed_steps += steps_this_interval;
            ++n_save_index;
        }
    }

    // Copy reduced [n_save, N] arrays to CPU.
    // Normalize by N_ensemble.
    // Compute derived observables.
    // write results (helper arrays) to .nc file
}
