#pragma once

#include "cuda_check.hpp"

#include <cstddef>
#include <utility>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();

        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)));

        count_ = count;
    }

    void set_to_zero() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T)));
        }
    }

    void copy_to_host(T *destination, std::size_t count) const {
        if (count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_to_host count exceeds allocation");
        }

        CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copy_to_host(std::vector<T> &destination) const {
        if (destination.size() > count_) {
            throw std::out_of_range("Host vector is larger than device allocation");
        }

        copy_to_host(destination.data(), destination.size());
    }

    void copy_from_host(const T *source, std::size_t count) {
        if (count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_from_host count exceeds allocation");
        }

        CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    [[nodiscard]] T *data() noexcept { return data_; }

    [[nodiscard]] const T *data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

  private:
    void release() noexcept {
        if (data_ != nullptr) {
            // Destructors should not throw.
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    T *data_ = nullptr;
    std::size_t count_ = 0;
};
