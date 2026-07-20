// this file initializes #batch_size rng states (one rng state per trajectory)

#pragma once

__global__ void initialize_rng_states(curandStatePhilox4_32_10_t *states, unsigned long long seed,
                                      int batch_begin, int current_batch_size) {
    const int trajectory = blockIdx.x * blockDim.x + threadIdx.x;

    if (trajectory >= current_batch_size) {
        return;
    }

    const unsigned long long trajectory_id_in_batch =
        static_cast<unsigned long long>(batch_begin + trajectory);

    // One independent subsequence for each physical trajectory.
    curand_init(seed,
                trajectory_id_in_batch, // subsequence
                0ULL,                   // initial offset
                &states[trajectory]);
}