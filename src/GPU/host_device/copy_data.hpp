#pragma once

#include "structs.hpp"

inline void copy_pearson_to_host(const DevicePearsonBuffers &device, HostPearsonBuffers &host) {
    device.xj0.copy_to_host(host.xj0);
    device.xj.copy_to_host(host.xj);
    device.x0.copy_to_host(host.x0);
    device.xj2.copy_to_host(host.xj2);
    device.x02.copy_to_host(host.x02);
}

inline void copy_energy_to_host(const DeviceEnergyBuffers &device, HostEnergyBuffers &host) {
    device.total.copy_to_host(host.total);
    device.potential.copy_to_host(host.potential);
    device.kinetic.copy_to_host(host.kinetic);
}