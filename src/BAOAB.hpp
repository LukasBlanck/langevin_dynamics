// BAOAB Integrator

#pragma once

#include "input/input.hpp"
#include "potentials.hpp"
#include <algorithm>
#include <random>
#include <vector>

class BAOAB {
  public:
    BAOAB(const Config config, std::vector<double> &q, std::vector<double> &p,
          std::vector<double> &F, const double dt)
        : N_(config.grid.N), m_(config.conventions.m), kB_(config.conventions.kB), dt_(dt),
          w_(config.model.omega), lambda_(config.model.lambda), beta_(config.model.beta),
          gamma_(lambda_ / m_), EJ_(config.model.EJ), left_bath_T_(config.model.left_bath_T),
          c_(std::exp(-gamma_ * dt_)), eta_(std::sqrt(m_ * kB_ * left_bath_T_ * (1 - c_ * c_))),
          q_(q), p_(p), F_(F), normal_(0.0, 1.0) {}

    inline void compute_force_fpu() {
        std::fill(F_.begin(), F_.end(), 0.0);
        // evaluate forces at bonds
        for (int i = 0; i < N_ - 1; i++) {
            double r = q_[i] - q_[i + 1];
            double bond_force = dV_FPU(r, w_, beta_);
            F_[i] -= bond_force;
            F_[i + 1] += bond_force;
        }
    }

    inline void compute_force_J() {
        std::fill(F_.begin(), F_.end(), 0.0);
        // evaluate forces at bonds
        for (int i = 0; i < N_ - 1; i++) {
            double r = q_[i] - q_[i + 1];
            double bond_force = dV_J(r, EJ_);
            F_[i] -= bond_force;
            F_[i + 1] += bond_force;
        }
    }

    inline void step_fpu(std::mt19937_64 &rng) {

        compute_force_fpu();

        // -- B -- update p
        for (int i = 0; i < N_; i++) {
            p_[i] = p_[i] + 0.5 * dt_ * F_[i];
        }

        // -- A -- update q
        for (int i = 0; i < N_; i++) {
            q_[i] = q_[i] + 0.5 * dt_ * (p_[i] / m_);
        }

        // -- O -- OU step
        const double Z = normal_(rng); // create random number from same seed
        // only left bath and noise
        p_[0] = c_ * p_[0] + eta_ * Z;

        // -- A -- update q
        for (int i = 0; i < N_; i++) {
            q_[i] = q_[i] + 0.5 * dt_ * (p_[i] / m_);
        }

        compute_force_fpu();

        // -- B -- update p
        for (int i = 0; i < N_; i++) {
            p_[i] = p_[i] + 0.5 * dt_ * F_[i];
        }
    }

  private:
    const int N_;

    const double m_;
    const double kB_;

    const double dt_;

    const double w_;
    const double lambda_;
    const double beta_;
    const double gamma_;
    const double EJ_;
    const double left_bath_T_;

    const double c_;   // OU step
    const double eta_; // OU step

    std::vector<double> &q_;
    std::vector<double> &p_;
    std::vector<double> &F_;

    std::normal_distribution<double> normal_;
};
