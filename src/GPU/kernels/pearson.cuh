// this file provides the reduction(s) for the pearson observables
// it takes the [batch_size * N] unprocessed raw d_p and d_q (and bonds)
// and returns the reduced unprocessed observables [n_save * N]

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// struct helper
struct Pearson {
    double xj0;
    double xj;
    double x0;
    double xj2;
    double x02;
};

__global__ inline void pearson_reduction(const double *x, double *xj0, double *xj, double *x0,
                                         double *xj2, double *x02, const int N,
                                         const int current_batch_size, const int n_save_index) {

    // same as in reduction.cuh: one thread - one trajectory
    // one block - one site

    // const int trajectory = threadIdx.x;
    const int site = blockIdx.x;

    if (site >= N) {
        return;
    }

    extern __shared__ Pearson batch_sum[];

    // local sums (to load mem into registers and not to shared memory)
    Pearson local_sum{0, 0, 0, 0, 0};

    for (int trajectory = threadIdx.x; trajectory < current_batch_size;
         trajectory += static_cast<int>(blockDim.x)) {
        // indices
        const std::size_t base = static_cast<std::size_t>(trajectory) * static_cast<std::size_t>(N);
        const double value_j = x[base + static_cast<std::size_t>(site)];
        const double value_0 = x[base];

        local_sum.xj0 += value_j * value_0;
        local_sum.xj += value_j;
        local_sum.x0 += value_0;
        local_sum.xj2 += value_j * value_j;
        local_sum.x02 += value_0 * value_0;
    }

    // load registers (local sum) into shared mem
    batch_sum[trajectory] = local_sum;
    __syncthreads();

    // perform tree based addition as in reduction.cuh
    for (unsigned int stride = blockDim.x / 2; stride > 0;
         stride >>= 1) { // ATTENTION Requires blockDim.x to be a power of two. !-!-!
        if (threadIdx.x < stride) {
            batch_sum[threadIdx.x].xj0 += batch_sum[threadIdx.x + stride].xj0;
            batch_sum[threadIdx.x].xj += batch_sum[threadIdx.x + stride].xj;
            batch_sum[threadIdx.x].x0 += batch_sum[threadIdx.x + stride].x0;
            batch_sum[threadIdx.x].xj2 += batch_sum[threadIdx.x + stride].xj2;
            batch_sum[threadIdx.x].x02 += batch_sum[threadIdx.x + stride].x02;
        }
        __syncthreads(); // ensure every tree add stage is completed before proceeding to next stage
    }

    // accumulate now different batches if N_ensemble > batch_size
    if (threadIdx.x == 0) {
        const std::size_t output_index = static_cast<std::size_t>(n_save_index) * N + site;
        // += because successive batches contribute to the same site.
        xj0[output_index] += batch_sum[0].xj0;
        xj[output_index] += batch_sum[0].xj;
        x0[output_index] += batch_sum[0].x0;
        xj2[output_index] += batch_sum[0].xj2;
        x02[output_index] += batch_sum[0].x02;
    }
}