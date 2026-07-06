
#include "input/input.hpp"
#include "potentials.hpp"
#include <cmath>
#include <random>
#include <vector>

// initialize grid -> q,p,F,T arrays m?
// create result arrays -> ej, pearson_corr

// ensemble

// evaluate forces
// iterate over j
// evaluate dV/dq
// update p with dt/2
// update q with dt/2

// update p with dt O.U step
// update q with dt/2
// evaluate forces
// iterate over j
// evaluate dV/dq
// update p with dt/2

// at extract times - write current state into result arrays

int main() {

    // load input params
    const Config config = load_config("src/input/input.toml");
    print_config(config);
    const double w = config.model.omega;
    const double beta = config.model.beta;

    const double dt = config.time.end_time / config.time.N;
    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double lambda = config.model.lambda;
    const double left_bath_T = config.model.left_bath_T;
    const double gamma = lambda / m;

    // initialize grid/arrays
    const int N = config.grid.N;

    std::vector<double> q(N);
    std::vector<double> p(N);
    std::vector<double> F(N);
    std::vector<double> T(N);

    int seed = 67;
    std::mt19937_64 rng(seed);

    // initialize with q=p=T=F=0 for first try
    for (int i = 0; i < N; i++) {
        q[i] = 0;
        p[i] = 0;
        F[i] = 0;
    }

    // evaluate potentials
    // evaluate forces at bonds
    for (int i = 0; i < N - 1; i++) {
        double r = q[i] - q[i + 1];
        double bond_force = dV_FPU(r, w, beta);
        F[i] -= bond_force;
        F[i + 1] += bond_force;
    }

    // update p
    for (int i = 0; i < N; i++) {
        p[i] = p[i] + 0.5 * dt * F[i];
    }

    // update q
    for (int i = 0; i < N; i++) {
        q[i] = q[i] + 0.5 * dt * (p[i] / m);
    }

    // Ornstein-Uhlenbeck step
    std::normal_distribution<double> normal(0.0, 1.0);
    const double Z = normal(rng);
    // only left bath and noise
    double c = std::exp(-gamma * dt);
    p[0] = c * p[0] + std::sqrt(m * kB * left_bath_T * (1 - c * c)) * Z;

    // update q
    for (int i = 0; i < N; i++) {
        q[i] = q[i] + 0.5 * dt * (p[i] / m);
    }

    // evaluate forces at bonds
    std::fill(F.begin(), F.end(), 0.0);
    for (int i = 0; i < N - 1; i++) {
        double r = q[i] - q[i + 1];
        double bond_force = dV_FPU(r, w, beta);
        F[i] -= bond_force;
        F[i + 1] += bond_force;
    }

    // update p
    for (int i = 0; i < N; i++) {
        p[i] = p[i] + 0.5 * dt * F[i];
    }

    return 0;
}