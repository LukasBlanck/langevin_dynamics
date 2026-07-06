
#include "BAOAB.hpp"
#include "input/input.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

// create result arrays -> ej, pearson_corr

// ensemble

// at extract times - write current state into result arrays

int main() {

    // load input params
    const Config config = load_config("src/input/input.toml");
    print_config(config);

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const int save_every = config.time.save_every;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    const double w = config.model.omega;
    const double beta = config.model.beta;
    const double lambda = config.model.lambda;
    const double left_bath_T = config.model.left_bath_T;
    const double gamma = lambda / m;

    // initialize grid/arrays

    std::vector<double> q(N);
    std::vector<double> p(N);
    std::vector<double> F(N);

    int seed = 67;
    std::mt19937_64 rng(seed);

    // initialize with q=p=F=0 for first try
    std::fill(q.begin(), q.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);

    // integrate
    BAOAB integrator(config, q, p, F, dt);

    // save initial condition


    for (int k = 0; k < N_time; k++) {
        integrator.step_fpu(rng);

        if ((k + 1) % save_every == 0 || k + 1 == N_time) {
            double t = (k + 1) * dt;    // time after step
            // save observables
        }
    }

    return 0;
}