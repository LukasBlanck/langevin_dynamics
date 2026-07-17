// this file extracta from every block in a batch -> [batch_size * N]
// the obervables and writes them into observables * [batch_size * N]

#pragma once

#include <cuda_runtime.h>

template <class Potential>
__device__ inline void symmetric_energy_at_site(double *tot_e, const double *shared_q,
                                                const double *p, const int N, const double m,
                                                const Potential &potential, const int trajectory,
                                                const int site) {

    // tot_e is of form [batch_size, N]
    // site = threadIdx.x
    const int stride = trajectory * N;

    // boundary
    if (site == N - 1) {
        tot_e[N - 1 + stride] = p[stride + N - 1] * p[stride + N - 1] / (2 * m) +
                                0.5 * potential.V(shared_q[N - 2] - shared_q[N - 1]);
    } else if (site > 0) {
        // inside sites
        tot_e[stride + site] = p[stride + site] * p[stride + site] / (2 * m) +
                               0.5 * potential.V(shared_q[site - 1] - shared_q[site]) +
                               0.5 * potential.V(shared_q[site] - shared_q[site + 1]);
    } else if (site == 0) {
        tot_e[0 + stride] =
            p[stride + 0] * p[stride + 0] / (2 * m) + 0.5 * potential.V(shared_q[0] - shared_q[1]);
    }
}

template <class Potential>
__global__ inline void extract_observables(double *p, double *q, double *tot_e,
                                           const Potential &potential, const int current_batch_size,
                                           const int N, const double m, const int n_save_index) {

    const int trajectory = blockIdx.x;
    if (trajectory >= current_batch_size) {
        return;
    }

    // block shared q
    extern __shared__ double shared_q[]; // shared per block

    // TODO: calculate stride = trajectory * N before looop
    for (int site = threadIdx.x; site < N; site += blockDim.x) {
        const std::size_t batch_index = static_cast<std::size_t>(trajectory) * N + site;
        shared_q[site] = q[batch_index]; // site accesses the block specific shared q
    }
    __syncthreads(); // wait for all threads within a block to have initialized/updated the
                     // shared q

    for (int site = threadIdx.x; site < N; site += blockDim.x) {
        symmetric_energy_at_site(tot_e, shared_q, p, N, m, potential, trajectory, site);
    }
}