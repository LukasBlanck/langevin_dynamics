// this file only contains functions that launch kernels, but do not instantiate it;
// the definition of those kernels should be in kernels/

#include "../input/input.hpp"
#include "cuda_check.hpp"
#include "GPU/kernels/extraction.cuh"
#include "GPU/kernels/integration.cuh"
#include "GPU/kernels/pearson.cuh"
#include "GPU/kernels/reduction.cuh"
#include "GPU/kernels/rng.cuh"
#include "host_device/copy_data.hpp"
#include "host_device/structs.hpp"
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
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    const double left_bath_T = config.model.left_bath_T;
    const double gamma = config.model.lambda / m;

    // -----------------------------------------------------------------------
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

    // bond count
    const int N_bond = N - 1;

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
    const int number_of_batches = (N_ensemble + batch_size - 1) / batch_size;

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

    // -----------------------------------------------------------------------
    // --------------
    // |    HOST    |
    // --------------

    // allocate final results (observables) [n_save * N] host buffers
    const std::size_t final_size = static_cast<std::size_t>(n_save) * static_cast<std::size_t>(N);
    const std::size_t bond_size =
        static_cast<std::size_t>(n_save) * static_cast<std::size_t>(N_bond);
    HostEnergyBuffers host_energy(final_size, n_save); // tot, pot and kin energy (+ normalized)

    HostPearsonBuffers host_momentum(final_size); // p0j, pj, p0, pj2, p02
    HostPearsonBuffers host_position(final_size); // q0j, qj, q0, qj2, q02
    HostPearsonBuffers host_bond(bond_size);      // r0j, rj, r0, rj2, r02

    std::vector<double> time(n_save, 0.0);

    // -----------------------------------------------------------------------
    // --------------
    // |   DEVICE   |
    // --------------

    const std::size_t temporary_size =
        static_cast<std::size_t>(batch_size) *
        static_cast<std::size_t>(N); // reusable (temporary observables) [batch_size * N]

    // allocate ALL simulation buffers
    DeviceSimulationBuffers device{
        temporary_size, final_size, bond_size,
        batch_size}; // allocates ALL temporaray: q, p, tot, pot, kin of size [batch_size * N] and
                     // ALL final: tot, pot, kin, 5*pearson*3 of size [n_save*N] and
                     // the rng_states of size [batch_size]

    // TODO:
    // at n_save_index=10 a ETA?
    // add timing?
    // here maybe stable dt calculation? or seperate with stable dt at runtime?

    // begin itegration (per batch)
    for (int batch = 0; batch < number_of_batches; ++batch) {

        const int batch_begin = batch * batch_size;
        const int current_batch_size =
            std::min(batch_size, N_ensemble - batch_begin); // last batch might be smaller

        // Initialize q, p, F and RNG for this batch.
        device.q.set_to_zero();
        device.p.set_to_zero();

        // initialize the rng states per batch
        constexpr int rng_threads_per_block = threads_per_block;
        const int rng_blocks = (current_batch_size + rng_threads_per_block - 1) /
                               rng_threads_per_block; // this batch has currently #rng_blocks blocks
        initialize_rng_states<<<rng_blocks, rng_threads_per_block>>>(
            device.rng_states.data(), static_cast<unsigned long long>(seed), batch_begin,
            current_batch_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int completed_steps = 0;
        int n_save_index = 0;

        // initial measurement at t = 0.
        // energy observables
        extract_observables<Potential><<<current_batch_size, threads_per_block, shared_bytes>>>(
            device.p.data(), device.q.data(), device.energy.total_temporary.data(),
            device.energy.potential_temporary.data(), device.energy.kinetic_temporary.data(),
            potential, current_batch_size, N, m);
        CUDA_CHECK(cudaGetLastError());

        // pearson reduction
        pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
            device.p.data(), device.momentum_pearson.xj0.data(), device.momentum_pearson.xj.data(),
            device.momentum_pearson.x0.data(), device.momentum_pearson.xj2.data(),
            device.momentum_pearson.x02.data(), N, current_batch_size, n_save_index);
        CUDA_CHECK(cudaGetLastError());
        pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
            device.p.data(), device.position_pearson.xj0.data(), device.position_pearson.xj.data(),
            device.position_pearson.x0.data(), device.position_pearson.xj2.data(),
            device.position_pearson.x02.data(), N, current_batch_size, n_save_index);
        CUDA_CHECK(cudaGetLastError());
        pearson_bond_reduction<<<N_bond, threads_per_block, pearson_reduction_shared_bytes>>>(
            device.p.data(), device.bond_pearson.xj0.data(), device.bond_pearson.xj.data(),
            device.bond_pearson.x0.data(), device.bond_pearson.xj2.data(),
            device.bond_pearson.x02.data(), N, current_batch_size, n_save_index);
        CUDA_CHECK(cudaGetLastError());

        // launch N blocks - one block is one site
        perform_reduction<<<N, threads_per_block, reduction_shared_bytes>>>(
            device.energy.total_temporary.data(), device.energy.potential_temporary.data(),
            device.energy.kinetic_temporary.data(), device.energy.total.data(),
            device.energy.potential.data(), device.energy.kinetic.data(), current_batch_size, N,
            n_save_index);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        ++n_save_index;

        while (completed_steps < N_time) { // regular reduction also in time resolution

            const int steps_this_interval =
                std::min(save_every, N_time - completed_steps); // might be less at end

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

            // pearson reduction
            pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
                device.p.data(), device.momentum_pearson.xj0.data(),
                device.momentum_pearson.xj.data(), device.momentum_pearson.x0.data(),
                device.momentum_pearson.xj2.data(), device.momentum_pearson.x02.data(), N,
                current_batch_size, n_save_index);
            CUDA_CHECK(cudaGetLastError());
            pearson_reduction<<<N, threads_per_block, pearson_reduction_shared_bytes>>>(
                device.p.data(), device.position_pearson.xj0.data(),
                device.position_pearson.xj.data(), device.position_pearson.x0.data(),
                device.position_pearson.xj2.data(), device.position_pearson.x02.data(), N,
                current_batch_size, n_save_index);
            CUDA_CHECK(cudaGetLastError());
            pearson_bond_reduction<<<N_bond, threads_per_block, pearson_reduction_shared_bytes>>>(
                device.p.data(), device.bond_pearson.xj0.data(), device.bond_pearson.xj.data(),
                device.bond_pearson.x0.data(), device.bond_pearson.xj2.data(),
                device.bond_pearson.x02.data(), N, current_batch_size, n_save_index);
            CUDA_CHECK(cudaGetLastError());

            // launch N blocks - one block is one site
            perform_reduction<<<N, threads_per_block, reduction_shared_bytes>>>(
                device.energy.total_temporary.data(), device.energy.potential_temporary.data(),
                device.energy.kinetic_temporary.data(), device.energy.total.data(),
                device.energy.potential.data(), device.energy.kinetic.data(), current_batch_size, N,
                n_save_index);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            completed_steps += steps_this_interval;
            ++n_save_index;
        }
    }

    // Copy reduced [n_save, N] arrays to from device to host.
    copy_energy_to_host(device.energy, host_energy);
    copy_pearson_to_host(device.momentum_pearson, host_momentum);
    copy_pearson_to_host(device.position_pearson, host_position);
    copy_pearson_to_host(device.bond_pearson, host_bond);

    // Normalize by N_ensemble
    const double inv_ensemble = 1.0 / static_cast<double>(N_ensemble);
    for (double &value : host_energy.total) {
        value *= inv_ensemble;
    }
    for (double &value : host_energy.potential) {
        value *= inv_ensemble;
    }
    for (double &value : host_energy.kinetic) {
        value *= inv_ensemble;
    }

    // Compute derived observables.

    // process weighted energies
    normalized_energy(host_energy.total, host_energy.normalized_total, n_save, N);
    normalized_energy(host_energy.kinetic, host_energy.normalized_kinetic, n_save, N);
    normalized_energy(host_energy.potential, host_energy.normalized_potential, n_save, N);

    first_moment(host_energy.normalized_total, host_energy.first_moment_total, n_save, N);
    spread(host_energy.normalized_total, host_energy.total_spread, host_energy.first_moment_total,
           n_save, N);

    // process pearson
    process_pearson_correlators(host_momentum.correlation, host_momentum.xj0, host_momentum.xj,
                                host_momentum.x0, host_momentum.xj2, host_momentum.x02, n_save, N,
                                inv_ensemble);
    process_pearson_correlators(host_position.correlation, host_position.xj0, host_position.xj,
                                host_position.x0, host_position.xj2, host_position.x02, n_save, N,
                                inv_ensemble);
    process_pearson_correlators(host_bond.correlation, host_bond.xj0, host_bond.xj, host_bond.x0,
                                host_bond.xj2, host_bond.x02, n_save, N_bond, inv_ensemble);

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
                                 "energy", host_energy.total);
    writer.write_time_site_array("local_potential_energy",
                                 "ensemble averaged local potential energy", "energy",
                                 host_energy.potential);
    writer.write_time_site_array("local_kinetic_energy", "ensemble averaged local kinetic energy",
                                 "energy", host_energy.kinetic);

    // weighted energies
    writer.write_time_site_array("normalized_total_energy",
                                 "ensemble averaged normalized local total energy", "dimensionless",
                                 host_energy.normalized_total);
    writer.write_time_site_array("normalized_kinetic_energy",
                                 "ensemble averaged normalized local kinetic energy",
                                 "dimensionless", host_energy.normalized_potential);
    writer.write_time_site_array("normalized_potential_energy",
                                 "ensemble averaged normalized local potential energy",
                                 "dimensionless", host_energy.normalized_kinetic);

    // moments
    writer.write_time_data_array("first_moment_total_energy",
                                 "ensemble averaged first momentum of total energy", "site",
                                 host_energy.first_moment_total);
    writer.write_time_data_array("total_energy_spread", "ensemble averaged spread of total energy",
                                 "site", host_energy.total_spread);

    // pearson correlation
    writer.write_time_site_array("pearson_momentum_correlation",
                                 "Pearson momentum correlation with left boundary (site at 0)",
                                 "dimensionless", host_momentum.correlation);
    writer.write_time_site_array("pearson_position_correlation",
                                 "Pearson position correlation with left boundary (site at 0)",
                                 "dimensionless", host_position.correlation);
    writer.write_time_bond_array("pearson_bond_correlation",
                                 "Pearson bond displacement correlation with left boundary bond",
                                 "dimensionless", host_bond.correlation);

    // simulation finished
    std::cout << "Finished simulation.\n";
    std::cout << "Output written to: " << output_path << "\n";
}
