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

    extern __shared__ Pearson pearson_batch_sum[];

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
    pearson_batch_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // perform tree based addition as in reduction.cuh
    for (unsigned int stride = blockDim.x / 2; stride > 0;
         stride >>= 1) { // ATTENTION Requires blockDim.x to be a power of two. !-!-!
        if (threadIdx.x < stride) {
            pearson_batch_sum[threadIdx.x].xj0 += pearson_batch_sum[threadIdx.x + stride].xj0;
            pearson_batch_sum[threadIdx.x].xj += pearson_batch_sum[threadIdx.x + stride].xj;
            pearson_batch_sum[threadIdx.x].x0 += pearson_batch_sum[threadIdx.x + stride].x0;
            pearson_batch_sum[threadIdx.x].xj2 += pearson_batch_sum[threadIdx.x + stride].xj2;
            pearson_batch_sum[threadIdx.x].x02 += pearson_batch_sum[threadIdx.x + stride].x02;
        }
        __syncthreads(); // ensure every tree add stage is completed before proceeding to next stage
    }

    // accumulate now different batches if N_ensemble > batch_size
    if (threadIdx.x == 0) {
        const std::size_t output_index = static_cast<std::size_t>(n_save_index) * N + site;
        // += because successive batches contribute to the same site.
        xj0[output_index] += pearson_batch_sum[0].xj0;
        xj[output_index] += pearson_batch_sum[0].xj;
        x0[output_index] += pearson_batch_sum[0].x0;
        xj2[output_index] += pearson_batch_sum[0].xj2;
        x02[output_index] += pearson_batch_sum[0].x02;
    }
}

// pearson bond correlation
__global__ inline void pearson_bond_reduction(const double *q, double *rj0, double *rj, double *r0,
                                              double *rj2, double *r02, const int N_bond,
                                              const int current_batch_size,
                                              const int n_save_index) {

    // similair as "normal" pearson as: one thread - one trajectory
    // one block - one bond

    // const int trajectory = threadIdx.x;
    const int bond = blockIdx.x;

    if (bond >= N_bond) {
        return;
    }

    extern __shared__ Pearson pearson_bond_batch_sum[];

    // local sums (to load mem into registers and not to shared memory)
    Pearson local_sum{0, 0, 0, 0, 0};

    for (int trajectory = threadIdx.x; trajectory < current_batch_size;
         trajectory += static_cast<int>(blockDim.x)) {
        // indices
        const std::size_t base = static_cast<std::size_t>(trajectory) * static_cast<std::size_t>(N_bond);
        const double value_j =
            q[base + static_cast<std::size_t>(bond)] - q[base + static_cast<std::size_t>(bond + 1)];
        const double value_0 = q[base] - q[base + 1];

        local_sum.xj0 += value_j * value_0;
        local_sum.xj += value_j;
        local_sum.x0 += value_0;
        local_sum.xj2 += value_j * value_j;
        local_sum.x02 += value_0 * value_0;
    }

    // load registers (local sum) into shared mem
    pearson_bond_batch_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // perform tree based addition as in reduction.cuh
    for (unsigned int stride = blockDim.x / 2; stride > 0;
         stride >>= 1) { // ATTENTION Requires blockDim.x to be a power of two. !-!-!
        if (threadIdx.x < stride) {
            pearson_bond_batch_sum[threadIdx.x].xj0 +=
                pearson_bond_batch_sum[threadIdx.x + stride].xj0;
            pearson_bond_batch_sum[threadIdx.x].xj +=
                pearson_bond_batch_sum[threadIdx.x + stride].xj;
            pearson_bond_batch_sum[threadIdx.x].x0 +=
                pearson_bond_batch_sum[threadIdx.x + stride].x0;
            pearson_bond_batch_sum[threadIdx.x].xj2 +=
                pearson_bond_batch_sum[threadIdx.x + stride].xj2;
            pearson_bond_batch_sum[threadIdx.x].x02 +=
                pearson_bond_batch_sum[threadIdx.x + stride].x02;
        }
        __syncthreads(); // ensure every tree add stage is completed before proceeding to next stage
    }

    // accumulate now different batches if N_ensemble > batch_size
    if (threadIdx.x == 0) {
        const std::size_t output_index = static_cast<std::size_t>(n_save_index) * N_bond + bond;
        // += because successive batches contribute to the same site.
        rj0[output_index] += pearson_bond_batch_sum[0].xj0;
        rj[output_index] += pearson_bond_batch_sum[0].xj;
        r0[output_index] += pearson_bond_batch_sum[0].x0;
        rj2[output_index] += pearson_bond_batch_sum[0].xj2;
        r02[output_index] += pearson_bond_batch_sum[0].x02;
    }
}
