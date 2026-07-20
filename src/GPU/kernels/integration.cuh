// this file integrates every site N to a given time.
// This given time t is (in scope of the whole simulation)
// not the final integration time but the current n_save_index

// the position array q should be block memory __shared__
// for the force calculation for faster access at neighbouring q vals

#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

template <class Potential>
__device__ inline double compute_site_force(const double *shared_q, const int N,
                                            const Potential &potential, const int i) {
    // i is threadIdx.x, so the site index
    double force_at_site = 0.0;

    // evaluate forces at neighboring bonds
    if (i == N - 1) {
        const double bond_force_left = potential.dV(shared_q[i - 1] - shared_q[i]);
        force_at_site = bond_force_left;
    } else if (i > 0) {
        const double bond_force_right = potential.dV(shared_q[i] - shared_q[i + 1]);
        const double bond_force_left = potential.dV(shared_q[i - 1] - shared_q[i]);
        force_at_site = bond_force_left - bond_force_right;
    } else if (i == 0) {
        const double bond_force_right = potential.dV(shared_q[i] - shared_q[i + 1]);
        force_at_site = -bond_force_right;
    }

    return force_at_site;
}

template <class Potential>
__global__ void integrate(double *p, double *q, curandStatePhilox4_32_10_t *rng_states,
                          const Potential potential, const int current_batch_size, const int N,
                          const int steps_this_interval, const double m, const double eta,
                          const double c, const double dt) {

    // one trajectory - one block
    const int trajectory = blockIdx.x;
    if (trajectory >= current_batch_size) {
        return;
    }

    // Initialize one seed per trajectory (like in CPU version)
    // only thread 0 (thermal batch and random energy injection)
    // handles the state of the rng in that trajectory for all time steps
    curandStatePhilox4_32_10_t local_rng; // tread 0 curand state
    if (threadIdx.x == 0) {
        local_rng = rng_states[trajectory];
    }

    // block shared q for fast access within block
    extern __shared__ double shared_q[]; // shared per block

    for (int step = 0; step < steps_this_interval; ++step) {

        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            const std::size_t batch_index = static_cast<std::size_t>(trajectory) * N + site;
            shared_q[site] = q[batch_index]; // site accesses the block specific shared q
        }
        __syncthreads(); // wait for all threads within a block to have initialized/updated the
                         // shared q

        // --- update q, p and F ---

        // -- B -- update p
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            const double force =
                compute_site_force(shared_q, N, potential, site); // force evaluation at site i
            p[trajectory * N + site] += 0.5 * dt * force;
        }
        __syncthreads();

        // -- A -- update q
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            q[trajectory * N + site] =
                q[trajectory * N + site] + 0.5 * dt * (p[trajectory * N + site] / m);
        }
        __syncthreads();

        // -- O -- OU step
        // for (int site = threadIdx.x; site < N; site += blockDim.x) {
        if (threadIdx.x == 0) {
            const double Z = curand_normal_double(&local_rng);
            p[trajectory * N] = c * p[trajectory * N] + eta * Z;
        }
        // }
        __syncthreads();

        // -- A -- update q
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            q[trajectory * N + site] =
                q[trajectory * N + site] + 0.5 * dt * (p[trajectory * N + site] / m);
        }
        __syncthreads();

        // update shared q for next force evaluation
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            const std::size_t batch_index = static_cast<std::size_t>(trajectory) * N + site;
            shared_q[site] = q[batch_index]; // site accesses the block specific shared q
        }
        __syncthreads();

        // -- B -- update p
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            const double force =
                compute_site_force(shared_q, N, potential, site); // force evaluation at site i
            p[trajectory * N + site] += 0.5 * dt * force;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        rng_states[trajectory] = local_rng;
    }
}