// CUDA backend simulation for the pilot run that returns the energy observables PER statistical
// batch - it returns then observables * [statistic_batches * N]

#include "../../GPU/host_device/structs.hpp"
#include "../../GPU/kernels/extraction.cuh"
#include "../../GPU/kernels/integration.cuh"
#include "../../GPU/kernels/pearson.cuh"
#include "../../GPU/kernels/reduction.cuh"
#include "../../GPU/kernels/rng.cuh"
#include "../../input/input.hpp"
#include "../../io/netCDF_writer.hpp"
#include "../cuda_check.hpp"
#include "../host_device/copy_data.hpp"
#include "../host_device/structs.hpp"
#include "../process/helpers.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <limits>


#include <cuda_runtime.h>
#include <curand_kernel.h>

// layout:
// Block - trajectory
// per batch:
// integrate up to save every
// store temporary observables    mem: [batch_size * N]
// reduction kernel: reduces batch_size Blocks to N in mem:[n_save * N] Matrix

// strided LOOP:
// Block always 256 threads, then
// for (std::size_t site = threadIdx.x;
//      site < N;
//      site += blockDim.x) {
//     // Process site.
// }

// ALL buffers necessary for the simulation
struct DevicePilotSimulationBuffers {
    DeviceBuffer<double> q;
    DeviceBuffer<double> p;

    DeviceEnergyBuffers energy;

    DeviceBuffer<curandStatePhilox4_32_10_t> rng_states;

    DevicePilotSimulationBuffers(std::size_t temporary_size /* [batch_size * N] */,
                                 std::size_t final_size /* [statistical_batches * N] */,
                                 int batch_size)
        : q(temporary_size), p(temporary_size), energy(temporary_size, final_size),
          rng_states(static_cast<std::size_t>(batch_size)) {}
};

struct SimulationResults {
    std::vector<double> total_energy;      // [batches * N]
    std::vector<double> kinetic_energy;    // [batches * N]
    std::vector<double> potential_energy;  // [batches * N]
    std::vector<double> tot_energy_spread; // [batches]
    std::vector<double> tot_energy_mean;   // [batches]
};

template <class Potential>
inline SimulationResults run_pilot_simulation(const int statistical_batches, const double dt,
                                              const int target_steps, const Config &config,
                                              const int trajectories_per_statistical_batch) {

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double end_time = target_steps * dt;
    const int N_time = end_time / dt;

    const int N_ensemble = config.ensemble.N;

    const double left_bath_T = config.model.left_bath_T;
    const double gamma = config.model.lambda / m;

    // check if the given trajectories_per_statistical_batch * statistical_batches = N_ensemble
    const std::size_t pilot_ensemble = static_cast<std::size_t>(statistical_batches) *
                                       static_cast<std::size_t>(trajectories_per_statistical_batch);

    if (pilot_ensemble != static_cast<std::size_t>(config.ensemble.N)) {
        throw std::invalid_argument("statistical_batches * "
                                    "trajectories_per_statistical_batch "
                                    "must equal config.ensemble.N");
    }

    // -----------------------------------------------------------------------
    // saving helpers
    // generic helpers
    int seed = 67;
    Potential potential(config);

    // helpers for OU step
    const double c = (std::exp(-gamma * dt));
    const double eta = std::sqrt(m * kB * left_bath_T * (1 - c * c));

    // -----------------------------------------------------------------------
    // device constants
    // TODO: inspect depedence on performance/throughput on device
    const int batch_size = 256;
    constexpr int threads_per_block = 256; // TODO: test for 128 and 512 and correspondant runtim
    static_assert(threads_per_block > 0 && (threads_per_block & (threads_per_block - 1)) == 0,
                  "threads_per_block must be a power of two !");
    const int number_of_batches =
        (trajectories_per_statistical_batch + batch_size - 1) / batch_size;

    // bytes of shared memory on block
    const std::size_t shared_bytes = static_cast<std::size_t>(N) * sizeof(double); // for shared q
    const int num_of_observables =
        3; // in the simplest form this is really the number of observables, but with later
           // HPC improvements the number can be smaller then the number of observables
    const std::size_t reduction_shared_bytes =
        static_cast<std::size_t>(threads_per_block) * num_of_observables * sizeof(double);

    // -----------------------------------------------------------------------
    // --------------
    // |    HOST    |
    // --------------

    // allocate final results (observables) [statistical_batches * N] host buffers
    const std::size_t final_size =
        static_cast<std::size_t>(statistical_batches) * static_cast<std::size_t>(N);
    HostEnergyBuffers host_energy(
        final_size,
        static_cast<std::size_t>(statistical_batches)); // tot, pot and kin energy (+ normalized)

    // -----------------------------------------------------------------------
    // --------------
    // |   DEVICE   |
    // --------------

    const std::size_t temporary_size =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(N); // reusable (temporary observables) [batch_size * N]

    // allocate ALL simulation buffers
    DevicePilotSimulationBuffers device{
        temporary_size, final_size,
        batch_size}; // allocates ALL temporaray: q, p, tot, pot, kin of size
                     // [batch_size * N] and ALL final: tot, pot, kin,
                     // of size [statistical_batches*N] and the rng_states of size [batch_size]

    // ETA
    const int eta_sample = 10; // number of n_save_index
    const auto wall_start = std::chrono::steady_clock::now();
    bool eta_printed = false;

    // -----------------------------------------------------------------------
    // -------------------
    // |   INTEGRATION   |
    // -------------------
    // begin itegration (per statistical batch)
    for (int statistical_batch_index = 0; statistical_batch_index < statistical_batches;
         ++statistical_batch_index) {

        // integrate number_of_batches (within a statistical batch) trajectories and reduce the
        // final time end result into [N] in [statistical_batches * N]
        for (int batch = 0; batch < number_of_batches; ++batch) {
            const int batch_begin = batch * batch_size;
            const int current_batch_size =
                std::min(batch_size, trajectories_per_statistical_batch -
                                         batch_begin); // last batch might be smaller

            // Initialize q, p, F and RNG for this batch.
            device.q.set_to_zero();
            device.p.set_to_zero();

            // initialize the rng states per batch
            const std::size_t trajectories_per_stat_batch =
                static_cast<std::size_t>(trajectories_per_statistical_batch);

            // for rng
            const std::size_t global_trajectory_begin =
                static_cast<std::size_t>(statistical_batch_index) * trajectories_per_stat_batch +
                static_cast<std::size_t>(batch_begin);

            constexpr int rng_threads_per_block = threads_per_block;
            const int rng_blocks =
                (current_batch_size + rng_threads_per_block - 1) /
                rng_threads_per_block; // this batch has currently #rng_blocks blocks
            initialize_rng_states<<<rng_blocks, rng_threads_per_block>>>(
                device.rng_states.data(), static_cast<unsigned long long>(seed),
                global_trajectory_begin, current_batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize()); // delete after ensured correctness

            const int steps_this_interval = target_steps; // integrate to end_time
            integrate<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
                device.p.data(), device.q.data(), device.rng_states.data(), potential,
                current_batch_size, N, steps_this_interval, m, eta, c, dt);
            CUDA_CHECK(cudaGetLastError());

            // --- measurements ---
            // energy observables
            extract_observables<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
                device.p.data(), device.q.data(), device.energy.total_temporary.data(),
                device.energy.potential_temporary.data(), device.energy.kinetic_temporary.data(),
                potential, current_batch_size, N, m);
            CUDA_CHECK(cudaGetLastError());

            // launch N blocks - one block is one site
            perform_reduction<<<N, threads_per_block, reduction_shared_bytes>>>(
                device.energy.total_temporary.data(), device.energy.potential_temporary.data(),
                device.energy.kinetic_temporary.data(), device.energy.total.data(),
                device.energy.potential.data(), device.energy.kinetic.data(), current_batch_size, N,
                statistical_batch_index);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // ETA
            // if (!eta_printed && statistical_batch_index >= eta_sample) {
            //     const double elapsed =
            //         std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
            //             .count();
            //     const double progress =
            //         (static_cast<double>(batch_begin) * N_time +
            //          static_cast<double>(current_batch_size) * completed_steps) /
            //         (static_cast<double>(N_ensemble) * N_time);
            //     const long long eta = static_cast<long long>(elapsed * (1.0 - progress) /
            //     progress); std::cout << "ETA: " << eta / 60 << ":" << std::setw(2) <<
            //     std::setfill('0')
            //               << eta % 60 << " min:s" << std::setfill(' ') << "\n\n"
            //               << std::flush;
            //     eta_printed = true;
            // }
        }
    }

    // Copy reduced [n_save, N] arrays to from device to host.
    copy_energy_to_host(device.energy, host_energy);

    // Normalize by N_ensemble
    const double inv_statistical_batch =
        1.0 / static_cast<double>(trajectories_per_statistical_batch);
    for (double &value : host_energy.total) {
        value *= inv_statistical_batch;
    }
    for (double &value : host_energy.potential) {
        value *= inv_statistical_batch;
    }
    for (double &value : host_energy.kinetic) {
        value *= inv_statistical_batch;
    }


    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%" << "\n verify manually the pilot GPU results:\n"
    for (int batch = 0; batch < statistical_batches; ++batch) {
        const std::size_t offset = static_cast<std::size_t>(batch) * static_cast<std::size_t>(N);

        double total_row_energy = 0.0;
        double minimum_energy = std::numeric_limits<double>::infinity();
        double maximum_energy = -std::numeric_limits<double>::infinity();

        for (int site = 0; site < N; ++site) {
            const double value = host_energy.total[offset + static_cast<std::size_t>(site)];

            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite total energy in statistical batch " +
                                         std::to_string(batch) + ", site " + std::to_string(site));
            }

            total_row_energy += value;
            minimum_energy = std::min(minimum_energy, value);
            maximum_energy = std::max(maximum_energy, value);
        }

        std::cout << "batch " << batch << ": sum(E) = " << total_row_energy
                  << ", min(E) = " << minimum_energy << ", max(E) = " << maximum_energy << '\n';

        if (!(total_row_energy > 0.0) || !std::isfinite(total_row_energy)) {

            throw std::runtime_error("Cannot normalize non-positive or non-finite "
                                     "energy profile in batch " +
                                     std::to_string(batch));
        }
    }

    // Compute derived observables.

    // process weighted energies
    normalized_energy(host_energy.total, host_energy.normalized_total, statistical_batches, N);

    first_moment(host_energy.normalized_total, host_energy.first_moment_total, statistical_batches,
                 N);
    spread(host_energy.normalized_total, host_energy.total_spread, host_energy.first_moment_total,
           statistical_batches, N);

    SimulationResults simul_results;
    simul_results.total_energy = host_energy.total;
    simul_results.kinetic_energy = host_energy.kinetic;
    simul_results.potential_energy = host_energy.potential;
    simul_results.tot_energy_mean = host_energy.first_moment_total;
    simul_results.tot_energy_spread = host_energy.total_spread;

    return simul_results;
}
