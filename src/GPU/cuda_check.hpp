#pragma once

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        cudaError_t error = (call);                                                                          \
        if (error != cudaSuccess) {                                                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                             \
                      << cudaGetErrorString(error) << '\n';                                                  \
            std::exit(EXIT_FAILURE);                                                                         \
        }                                                                                                    \
    } while (0)