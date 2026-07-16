// file for evaluating potentials V(r) and derivatives dV/dr
#pragma once

#include "input/input.hpp"
#include <cmath>
#include <string>

#if defined(LANGEVIN_COMPILE_FOR_GPU)
#define LANGEVIN_HD __host__ __device__
#else
#define LANGEVIN_HD
#endif

struct FPUPotential {
    double omega;
    double beta;

    explicit FPUPotential(const Config &config)
        : omega(config.model.omega), beta(config.model.beta) {}

    // potential V
    LANGEVIN_HD inline double V(double r) const {
        return 0.5 * omega * omega * r * r + 0.25 * beta * r * r * r * r;
    }

    // derivative dV/dr
    LANGEVIN_HD inline double dV(double r) const { return omega * omega * r + beta * r * r * r; }

    static std::string name() { return "FPU"; }
};

struct JosephsonPotential {
    double EJ;

    explicit JosephsonPotential(const Config &config) : EJ(config.model.EJ) {}

    // potential V
    LANGEVIN_HD inline double V(double r) const { return -EJ * std::cos(r); }

    // derivative dV/dr
    LANGEVIN_HD inline double dV(double r) const { return EJ * std::sin(r); }

    static std::string name() { return "Josephson"; }
};