// this file integrates every site N to a given time.
// This given time t is (in scope of the whole simulation)
// not the final integration time but the current n_save_index

// the position array q should be block memory __shared__
// for the force calculation

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

// rng helper
__device__ inline double normal(unsigned long long seed, unsigned long long ensemble_id,
                                unsigned long long global_step, unsigned long long bath_id) {
    curandStatePhilox4_32_10_t state;
    const unsigned long long stream_id = 2ULL * ensemble_id + bath_id;
    curand_init(seed, stream_id, global_step, &state);
    return curand_normal_double(&state);
}

template <class Potential>
__global__ void integrate(double *p, double *q, const Potential potential,
                          const int current_batch_size, const int N, const int steps_this_interval,
                          const int completed_steps, unsigned long long seed, const double m,
                          const double eta, const double c, const int batch_begin,
                          const double dt) {

    // Initialize one seed per trajectory (like in CPU version)
    const int trajectory = blockIdx.x;
    if (trajectory >= current_batch_size) {
        return;
    }
    // for rng
    const unsigned long long ensemble_id =
        static_cast<unsigned long long>(batch_begin + trajectory);

    // block shared q
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
        for (int site = threadIdx.x; site < N; site += blockDim.x) {
            if (site == 0) {
                const unsigned long long global_step =
                    static_cast<unsigned long long>(completed_steps + step);

                const double Z = normal(seed, ensemble_id, global_step,
                                        0ULL // left bath
                );
                // only left bath and noise
                p[trajectory * N] = c * p[trajectory * N] + eta * Z;
            }
        }
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
}