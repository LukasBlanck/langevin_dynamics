// kernel to perform the reduction within a batch

// the observables have form [batch_size * N]
// those should be reduced to [N] and furthermore
// be written into [n_save * N] at the correct n_save_index

// ATTENTION: block layout
// one block = one site
// threads within block = trajectories

#pragma once

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

template <class Potential>
__global__ inline void perform_reduction(double *tot_e_temporary, double *tot_e,
                                         const int current_batch_size, const int N,
                                         const int n_save_index) {

    // tot_e_temporary has size [batch_size * N]
    // tot_e has size [n_save * N]

    const int site = blockIdx.x; // one block per site
    if (site >= N) {
        return;
    }
    // trajectory = threadIdx.x;

    // create sum for all sites - size: [batch_size]
    extern __shared__ double batch_sum[]; // contains all trajectories for one site

    batch_sum[threadIdx.x] = 0.0;

    for (int trajectory = threadIdx.x; trajectory < current_batch_size; trajectory += blockDim.x) {
        batch_sum[threadIdx.x] +=
            tot_e_temporary[static_cast<std::size_t>(trajectory) * N +
                            site]; // accumulate different trajectories into one site (local_sum)
    }
    __syncthreads();

    // now those different trajectories are in shared memory and we can
    // add them together - ATTENTION: race conditions in writing sum
    // SOLUTION: tree based add/reduction (see https://en.wikipedia.org/wiki/Tree_contraction)
    for (unsigned int stride = blockDim.x / 2; stride > 0;
         stride >>= 1) { // ATTENTION Requires blockDim.x to be a power of two. !-!-!
        if (threadIdx.x < stride) {
            batch_sum[threadIdx.x] += batch_sum[threadIdx.x + stride];
        }
        __syncthreads(); // ensure every tree add stage is completed before proceeding to next stage
    }

    // accumulate now different batches if N_ensemble > batch_size
    if (threadIdx.x == 0) {
        const std::size_t output_index = static_cast<std::size_t>(n_save_index) * N + site;
        // += because successive batches contribute to the same site.
        tot_e[output_index] += batch_sum[0];
    }
}