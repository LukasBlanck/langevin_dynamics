#pragma once

#include "device_buffers.cuh"

#include <cstddef>
#include <curand_kernel.h>

// energy
struct DeviceEnergyBuffers {
    DeviceBuffer<double> total_temporary;
    DeviceBuffer<double> potential_temporary;
    DeviceBuffer<double> kinetic_temporary;

    DeviceBuffer<double> total;
    DeviceBuffer<double> potential;
    DeviceBuffer<double> kinetic;

    DeviceEnergyBuffers(std::size_t temporary_size, std::size_t final_size)
        : total_temporary(temporary_size), potential_temporary(temporary_size),
          kinetic_temporary(temporary_size), total(final_size), potential(final_size),
          kinetic(final_size) {
        total.set_to_zero();
        potential.set_to_zero();
        kinetic.set_to_zero();
    }
};

// pearson correlators
struct DevicePearsonBuffers {
    DeviceBuffer<double> xj0;
    DeviceBuffer<double> xj;
    DeviceBuffer<double> x0;
    DeviceBuffer<double> xj2;
    DeviceBuffer<double> x02;

    explicit DevicePearsonBuffers(std::size_t count)
        : xj0(count), xj(count), x0(count), xj2(count), x02(count) {
        set_to_zero();
    }

    void set_to_zero() {
        xj0.set_to_zero();
        xj.set_to_zero();
        x0.set_to_zero();
        xj2.set_to_zero();
        x02.set_to_zero();
    }
};

// ALL buffers necessary for the simulation
struct DeviceSimulationBuffers {
    DeviceBuffer<double> q;
    DeviceBuffer<double> p;

    DeviceEnergyBuffers energy;
    DevicePearsonBuffers momentum_pearson;
    DevicePearsonBuffers position_pearson;
    DevicePearsonBuffers bond_pearson;

    DeviceBuffer<curandStatePhilox4_32_10_t> rng_states;

    DeviceSimulationBuffers(std::size_t temporary_size /* [batch_size * N] */,
                            std::size_t final_size /* [n_save * N] */,
                            std::size_t bond_size /* [n_save * N_bond] */, int batch_size)
        : q(temporary_size), p(temporary_size), energy(temporary_size, final_size),
          momentum_pearson(final_size), position_pearson(final_size), bond_pearson(bond_size),
          rng_states(static_cast<std::size_t>(batch_size)) {}
};
